#!/usr/bin/env python
"""Quantify the fidelity loss of a held-out-mode-blind surrogate with evaluation error bars.

Samples models A and B on the held-out test events, then: (i) closure AUC data-vs-A, data-vs-B and the direct
A-vs-B AUC, marginal and truth-conditional, each with a bootstrap 68% interval over test events; (ii) the same
for the W1 of key variables. Usage: scripts/holdout_uncertainty.py OUT_DIR --a DIR_A --b DIR_B [--inttype 8] [--n-boot 200]
"""
import argparse, glob, json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, torch
from torch.utils.data import DataLoader
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from scipy.stats import wasserstein_distance
from sim2reco.data.compact import load_compact, CompactDataset, collate, MODEL_NAMES
from sim2reco.data.dataset import split_by_subrun
from sim2reco.train.m3 import load_model
from sim2reco.train.m2 import to_dev, _pad
from sim2reco.prep.features import event_features

ap = argparse.ArgumentParser(); ap.add_argument("out_dir"); ap.add_argument("--a", required=True); ap.add_argument("--b", required=True)
ap.add_argument("--inttype", type=int, nargs="*", default=[8]); ap.add_argument("--n-boot", type=int, default=200); ap.add_argument("--slim-dir", default="data/slim_1A"); ap.add_argument("--steps", type=int, default=64)
a = ap.parse_args(); out = pathlib.Path(a.out_dir); out.mkdir(parents=True, exist_ok=True); rng = np.random.default_rng(0)
stems = sorted(p[:-len(".truth.parquet")] for p in glob.glob(f"{a.slim_dir}/*.truth.parquet"))
mA, tfA, ptfA = load_model(pathlib.Path(a.a) / "model.pt")
ck = torch.load(pathlib.Path(a.a) / "model.pt", map_location="cpu", weights_only=False)["config"]
d = load_compact(stems, ke_cut_mev=ck.get("ke_cut_mev", 50.0), keep_neutrons=ck.get("keep_neutrons", False))
split = split_by_subrun(d["subrun"], seed=0); te = np.where((split == 2) & np.isin(d["intType"], a.inttype))[0]
reco = d["reco_exists"][te]; ir = te[reco]
print(f"{len(te):,} held-out test events, {reco.sum():,} reconstructed", flush=True)

def sample(model, tf, ptf, seed):
    torch.manual_seed(seed); ds = CompactDataset(d, ir, tf, 0, prong_tf=ptf); S = []
    for b in DataLoader(ds, batch_size=2048, collate_fn=collate, num_workers=4):
        s = model.sample(to_dev(b, "cuda"), a.steps)
        S.append(torch.cat([s["minos"][:, None].float(), s["charge"][:, None].float(), s["nprong"][:, None].float(), s["x1"]], 1).cpu())
    return torch.cat(S).numpy()
mB, tfB, ptfB = load_model(pathlib.Path(a.b) / "model.pt")
SA, SB = sample(mA, tfA, ptfA, 1), sample(mB, tfB, ptfB, 2)
# B's continuous part lives in B's standardised space: map it to physical units and back into A's space
yB = tfB.inverse(SB[:, 3:], d["mu_true"][ir], d["ctx"][ir]); xB, vB = tfA.forward(yB, d["mu_true"][ir], d["ctx"][ir], rng)
SB = np.concatenate([SB[:, :3], xB], 1)
xr, valid = tfA.forward(d["tier1"][ir], d["mu_true"][ir], d["ctx"][ir], rng); valid = valid & vB
t0 = d["tier0"][ir]; R = np.concatenate([t0[:, 1:3], d["nprong"][ir][:, None].astype(float), xr], 1)[valid]; SA, SB = SA[valid], SB[valid]
cls_p, mom_p, mask_p = _pad(d, ir[valid]); X, _ = event_features(cls_p, mom_p, mask_p, d["ctx"][ir[valid]])
n = len(R); print(f"{n:,} events in the comparison", flush=True)

def auc(P, Q, cond):
    """Classifier P-vs-Q fitted on a random half of the events; AUC on the other half with a bootstrap over the
    held-out events only (no event appears on both sides, so duplicates cannot leak)."""
    A = np.concatenate([P, Q]); y = np.r_[np.ones(n), np.zeros(n)]
    if cond: A = np.concatenate([np.concatenate([X, X]), A], 1)
    ev = rng.permutation(n); h = n // 2; tr = np.r_[ev[:h], n + ev[:h]]; tst = np.r_[ev[h:], n + ev[h:]]
    c = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1, early_stopping=True, random_state=0).fit(A[tr], y[tr])
    s_ = c.predict_proba(A[tst])[:, 1]; yt = y[tst]
    point = roc_auc_score(yt, s_); m = len(tst)
    boots = [roc_auc_score(yt[i], s_[i]) for i in (rng.integers(0, m, m) for _ in range(a.n_boot))]
    return point, float(np.percentile(boots, 16)), float(np.percentile(boots, 84))

res = {}
pairs = {"data_vs_A": (R, SA), "data_vs_B": (R, SB), "A_vs_B": (SA, SB)}
for name, (P, Q) in pairs.items():
    for cond in (False, True):
        point, lo, hi = auc(P, Q, cond)
        res[f"auc_{name}_{'cond' if cond else 'marg'}"] = {"point": float(point), "boot_lo": lo, "boot_hi": hi, "n_boot": a.n_boot}
        print(f"AUC {name:10s} {'cond' if cond else 'marg'}: {point:.3f}  [{lo:.3f}, {hi:.3f}]", flush=True)
# W1 with bootstrap for the key variables (columns of x1 after the 3 discrete ones)
for j, nm in enumerate(MODEL_NAMES):
    col = 3 + j; w = {}
    for name, (P, Q) in pairs.items():
        vals = [wasserstein_distance(P[idx, col], Q[idx, col]) for idx in (rng.integers(0, n, n) for _ in range(a.n_boot))]
        w[name] = {"point": float(wasserstein_distance(P[:, col], Q[:, col])), "lo": float(np.percentile(vals, 16)), "hi": float(np.percentile(vals, 84))}
    res[f"w1_{nm}"] = w
json.dump(res, open(out / "uncertainty.json", "w"), indent=1)
print("W1 (point [16%,84%]) data_vs_A / data_vs_B / A_vs_B:")
for nm in MODEL_NAMES:
    w = res[f"w1_{nm}"]; print(f"  {nm:26s} " + "  ".join(f"{w[k]['point']:.3f} [{w[k]['lo']:.3f},{w[k]['hi']:.3f}]" for k in pairs))
