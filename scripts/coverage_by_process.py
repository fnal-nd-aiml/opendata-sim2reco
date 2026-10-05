#!/usr/bin/env python
"""For each GENIE interaction type: how much of it lives in regions of the model's input space where all other
processes together are rare?  Density ratio from classifier odds: r = P/(1-P) * (1-prior)/prior = f_proc / f_rest.
A reweighting from 'rest' to 'proc' needs weight r on those events.  Writes reports/coverage_2p2h/by_process.json."""
import glob, json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sim2reco.data.compact import load_compact
from sim2reco.prep.features import event_features
from sim2reco.train.m2 import _pad
NAMES = {1: "QE", 2: "RES", 3: "DIS", 4: "COH", 5: "diffractive", 6: "nu-e", 7: "IMD", 8: "2p2h/MEC", 9: "AMNuGamma", 10: "unknown-10"}
rng = np.random.default_rng(0); stems = sorted(p[:-len(".truth.parquet")] for p in glob.glob("data/slim_1A/*.truth.parquet")); d = load_compact(stems)
it = d["intType"]; N = len(it); sub = rng.permutation(N)[:2000000]
cls, mom, mask = _pad(d, sub); X, names = event_features(cls, mom, mask, d["ctx"][sub])
P = np.linalg.norm(mom, axis=-1); isp = mask & (cls == 4); Pp = np.where(isp, P, -1.0); Ps = -np.sort(-Pp, axis=1); X = np.column_stack([X, Ps[:, 0], Ps[:, 1]])
half = len(sub) // 2; out = {}
print(f"{'process':12s} {'share':>6s} {'AUC':>6s} | fraction of the process in regions where rest/proc density < 0.1 | < 0.01 | median rest/proc | p10")
for code in sorted(np.unique(it)):
    y = it[sub] == code; share = (it == code).mean()
    if y[:half].sum() < 500: print(f"{NAMES.get(int(code), str(code)):12s} {100*share:5.2f}%  too few events"); continue
    m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, max_leaf_nodes=63, random_state=0).fit(X[:half], y[:half])
    p = np.clip(m.predict_proba(X[half:])[:, 1], 1e-6, 1 - 1e-6); yt = y[half:]; prior = y[:half].mean()
    r_rest_over_proc = (1 - p) / p * prior / (1 - prior)  # f_rest / f_proc at the event's location
    rr = r_rest_over_proc[yt]; auc = roc_auc_score(yt, p)
    out[NAMES.get(int(code), str(code))] = {"code": int(code), "share": float(share), "auc": float(auc), "frac_lt_0.1": float((rr < 0.1).mean()), "frac_lt_0.01": float((rr < 0.01).mean()), "median": float(np.median(rr)), "p10": float(np.percentile(rr, 10)), "n_test": int(yt.sum())}
    print(f"{NAMES.get(int(code), str(code)):12s} {100*share:5.2f}% {auc:6.3f} | {100*(rr<0.1).mean():6.1f}% | {100*(rr<0.01).mean():6.2f}% | {np.median(rr):8.2f} | {np.percentile(rr,10):6.3f}", flush=True)
json.dump(out, open("reports/coverage_2p2h/by_process.json", "w"), indent=1)
