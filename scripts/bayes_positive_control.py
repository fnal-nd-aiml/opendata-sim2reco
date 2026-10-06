#!/usr/bin/env python
"""Positive control for the epistemic flag: perturb in-distribution events into regions the training data cannot
contain and check that the per-event epistemic variance rises. Perturbations: hadron momenta x4; hadron tokens
tripled (duplicated with 10% jitter); vertex moved far upstream (z = 2500 mm); all three.
Usage: scripts/bayes_positive_control.py OUT_DIR --model DIR [--n 30000]"""
import argparse, glob, json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, torch
from sklearn.metrics import roc_auc_score
from sim2reco.data.compact import load_compact, CompactDataset, collate
from sim2reco.data.dataset import split_by_subrun
from sim2reco.train.m3 import load_model
from sim2reco.train.m2 import to_dev
from sim2reco.models.bayes_last import FeatureTap, GaussianLastLayer, HEADS, head_layer, heads_for
from sim2reco.eval import plots

ap = argparse.ArgumentParser(); ap.add_argument("out_dir"); ap.add_argument("--model", required=True); ap.add_argument("--slim-dir", default="data/slim_1A"); ap.add_argument("--n", type=int, default=30000); ap.add_argument("--steps", type=int, default=32); ap.add_argument("--tag", default="")
a = ap.parse_args(); out = pathlib.Path(a.out_dir); (out / "figures").mkdir(parents=True, exist_ok=True); dev = "cuda"; rng = np.random.default_rng(0)
D = pathlib.Path(a.model); model, tf, ptf = load_model(D / "model.pt"); ck = torch.load(D / "model.pt", map_location="cpu", weights_only=False)["config"]
post = {k: GaussianLastLayer.from_state(s, dev) for k, s in torch.load(D / "bayes_last.pt", map_location="cpu", weights_only=False).items()}
layers = {k: head_layer(model, p) for k, (p, _) in heads_for(model).items()}; taps = {k: FeatureTap(l) for k, l in layers.items()}
stems = sorted(p[:-len(".truth.parquet")] for p in glob.glob(f"{a.slim_dir}/*.truth.parquet"))
d = load_compact(stems, ke_cut_mev=ck.get("ke_cut_mev", 50.0), keep_neutrons=ck.get("keep_neutrons", False)); split = split_by_subrun(d["subrun"], seed=0)
idx = rng.permutation(np.where((split == 2) & ~np.isin(d["intType"], [8]))[0])[:a.n]
ds = CompactDataset(d, idx, tf, 0, prong_tf=ptf); batches = [b for b in torch.utils.data.DataLoader(ds, batch_size=512, collate_fn=collate, num_workers=4)]

def perturb(b, mode):
    b = {k: v.clone() for k, v in b.items()}; had = b["mask"] & (b["cls"] != 1)
    if mode in ("p_x4", "all"): b["mom"] = torch.where(had[..., None], b["mom"] * 4.0, b["mom"])
    if mode in ("tokens_x3", "all"):
        cls, mom, mask = b["cls"], b["mom"], b["mask"]; hc = torch.where(had, cls, torch.zeros_like(cls)); hm = mom * had[..., None]
        jit = lambda: hm * (1 + 0.1 * torch.randn_like(hm))
        b["cls"] = torch.cat([cls, hc, hc], 1); b["mom"] = torch.cat([mom, jit(), jit()], 1); b["mask"] = torch.cat([mask, had, had], 1)
    if mode in ("vtx_upstream", "all"): b["ctx"][:, 2] = 2500.0
    return b

def flags(mode):
    o = {"exist": [], "card": [], "flow": []}
    with torch.no_grad():
        for b0 in batches:
            b = to_dev(perturb(b0, mode) if mode != "nominal" else b0, dev); z, _ = model.enc(b["cls"], b["mom"], b["mask"], b["ctx"])
            lg = model.tier0(z); o["exist"].append(post["tier0"].var(taps["tier0"].h)[:, 0].cpu())
            pn = torch.softmax(model.card(z), -1); o["card"].append(post["card"].var(taps["card"].h).mean(1).cpu())
            fl = torch.stack([torch.rand(len(z), device=dev) < torch.sigmoid(lg)[:, 1], torch.zeros(len(z), dtype=torch.bool, device=dev)], -1).float()
            cond = model.flow_cond(z, model.full_flags(z, fl), torch.multinomial(pn, 1)[:, 0]); x = torch.randn(len(z), model.flow.dim, device=dev); dt = 1 / a.steps; g = torch.zeros(len(z), layers["flow"].in_features, device=dev)
            for i in range(a.steps):
                t = torch.full((len(z),), i * dt, device=dev); k1 = model.flow.v(x, t, cond); k2 = model.flow.v(x + 0.5 * dt * k1, t + 0.5 * dt, cond); g += dt * taps["flow"].h; x = (x + dt * k2).clamp(-20, 20)
            o["flow"].append(post["flow"].var(g).sum(1).cpu())
    return {k: torch.cat(v).numpy() for k, v in o.items()}

modes = ["nominal", "p_x4", "tokens_x3", "vtx_upstream", "all"]; F = {m: flags(m) for m in modes}
R = {}
for m in modes[1:]:
    R[m] = {k: {"median_ratio": float(np.median(F[m][k]) / np.median(F["nominal"][k])), "auc_vs_nominal": float(roc_auc_score(np.r_[np.ones(len(F[m][k])), np.zeros(len(F["nominal"][k]))], np.r_[F[m][k], F["nominal"][k]]))} for k in F[m]}
    print(m, {k: (round(v["median_ratio"], 2), round(v["auc_vs_nominal"], 3)) for k, v in R[m].items()}, flush=True)
json.dump(R, open(out / f"bayes_positive_control{a.tag}.json", "w"), indent=1)
fig, axs = plots.plt.subplots(1, 3, figsize=(11, 3.2)); cols = [plots.PALETTE["third"], plots.PALETTE["real"], plots.PALETTE["model"], plots.PALETTE["fourth"], plots.PALETTE["gray"]]
for ax, k, lab in zip(axs, ["exist", "card", "flow"], ["epistemic var., reco-exists logit", "epistemic var., multiplicity logits", "epistemic var., event flow sample"]):
    allv = np.concatenate([F[m][k] for m in modes]); lo, hi = np.percentile(allv, [0.5, 99.9]); b = np.logspace(np.log10(max(lo, 1e-9)), np.log10(hi), 50)
    for m, c in zip(modes, cols): ax.hist(F[m][k], bins=b, histtype="step", density=True, color=c, label=m.replace("_", " "))
    ax.set_xscale("log"); ax.set_xlabel(lab); ax.set_ylabel("density"); ax.legend(frameon=False, fontsize=7)
plots.save(fig, out / "figures" / f"bayes_positive_control{a.tag}.png"); print("done")
