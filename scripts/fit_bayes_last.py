#!/usr/bin/env python
"""Fit Bayesian last layers on a frozen surrogate from one pass over its training events.
Usage: scripts/fit_bayes_last.py MODEL_DIR [--max-events 2000000] [--draws 2]"""
import argparse, glob, pathlib, sys, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, torch
from torch.utils.data import DataLoader
from sim2reco.data.compact import load_compact, CompactDataset, collate
from sim2reco.data.dataset import split_by_subrun
from sim2reco.train.m3 import load_model
from sim2reco.train.m2 import to_dev, apply_holdout
from sim2reco.models.bayes_last import FeatureTap, GaussianLastLayer, HEADS, head_layer

ap = argparse.ArgumentParser(); ap.add_argument("model_dir"); ap.add_argument("--slim-dir", default="data/slim_1A"); ap.add_argument("--max-events", type=int, default=2_000_000); ap.add_argument("--draws", type=int, default=2)
a = ap.parse_args(); D = pathlib.Path(a.model_dir); dev = "cuda"
model, tf, ptf = load_model(D / "model.pt"); ck = torch.load(D / "model.pt", map_location="cpu", weights_only=False)["config"]
stems = sorted(p[:-len(".truth.parquet")] for p in glob.glob(f"{a.slim_dir}/*.truth.parquet"))
d = load_compact(stems, ke_cut_mev=ck.get("ke_cut_mev", 50.0), keep_neutrons=ck.get("keep_neutrons", False))
split = apply_holdout(d, split_by_subrun(d["subrun"], seed=0), ck.get("exclude_inttype") or None)
tr = np.where(split == 0)[0]; rng = np.random.default_rng(0); tr = rng.permutation(tr)[:a.max_events]
ds = CompactDataset(d, tr, tf, 0, prong_tf=ptf); ld = DataLoader(ds, batch_size=2048, collate_fn=collate, num_workers=4)
layers = {k: head_layer(model, p) for k, (p, _) in HEADS.items()}; taps = {k: FeatureTap(l) for k, l in layers.items()}
post = {k: GaussianLastLayer(l.in_features, l.out_features, dev) for k, l in layers.items()}
t0 = time.time()
with torch.no_grad():
    for i, b in enumerate(ld):
        b = to_dev(b, dev); z, h = model.enc(b["cls"], b["mom"], b["mask"], b["ctx"])
        t0b = b["tier0"]; reco = t0b[:, 0] > 0; minos = reco & (t0b[:, 1] > 0); fl = reco & (b["valid"] > 0)
        # classification heads: Laplace weights p(1-p) (binary) / mean over classes (softmax)
        lg = model.tier0(z); p = torch.sigmoid(lg); hz = taps["tier0"].h
        w = torch.stack([p[:, 0] * (1 - p[:, 0]), (p[:, 1] * (1 - p[:, 1])) * reco, (p[:, 2] * (1 - p[:, 2])) * minos], 1)
        post["tier0"].accumulate(hz, None, w.mean(1))
        if reco.any():
            pc = torch.softmax(model.card(z[reco]), -1); post["card"].accumulate(taps["card"].h, None, (pc * (1 - pc)).sum(1))
        cond = model.flow_cond(z[fl], t0b[fl, 1:3], b["nprong"][fl])
        if fl.any():
            pv = torch.softmax(model.vtx_logits(cond, b["vphase"][fl]), -1); post["vtx"].accumulate(taps["vtx"].h, None, (pv * (1 - pv)).sum(1))
            # flows: regression of the FM target on the last-layer features, several (t, noise) draws per event
            x1 = b["x1"][fl]
            for _ in range(a.draws):
                x0 = torch.randn_like(x1); t = torch.rand(len(x1), device=dev); sm = model.flow.sigma_min
                xt = (1 - (1 - sm) * t)[:, None] * x0 + t[:, None] * x1; target = x1 - (1 - sm) * x0
                model.flow.v(xt, t, cond); post["flow"].accumulate(taps["flow"].h, target)
            pm = b["pmask"][fl]; has = pm.any(1)
            if has.any():
                P1 = b["prongs"][fl][has]; pc_ = torch.cat([cond[has], x1[has]], -1); hh = h[fl][has]; hm = b["mask"][fl][has]; pmm = pm[has]
                for _ in range(a.draws):
                    P0 = torch.randn_like(P1); t = torch.rand(len(P1), device=dev); sm = model.prong.sigma_min; tt = t[:, None, None]
                    Pt = (1 - (1 - sm) * tt) * P0 + tt * P1; target = (P1 - (1 - sm) * P0)
                    model.prong.v(Pt, t, pc_, hh, hm, pmm); feat = taps["prong"].h[pmm]; post["prong"].accumulate(feat, target[pmm])
        if i % 200 == 0: print(f"  batch {i}/{len(ld)}  {time.time()-t0:.0f} s", flush=True)
for k, (_, kind) in HEADS.items():
    post[k].fit_regression() if kind == "regression" else post[k].fit_laplace()
    print(f"{k:6s}: n={post[k].n:,} lambda={post[k].lam}" + (f" noise var={post[k].noise.mean().item():.3f}" if post[k].noise is not None else ""), flush=True)
torch.save({k: v.state() for k, v in post.items()}, D / "bayes_last.pt"); print(f"saved {D/'bayes_last.pt'} in {time.time()-t0:.0f} s")
