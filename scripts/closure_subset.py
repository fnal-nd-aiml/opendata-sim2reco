#!/usr/bin/env python
"""Closure AUC (marginal / truth-conditional) with the evaluation's classifier recipe, for all nine generated variables
and for the non-calorimetry observable set (muon, vertex, recoil; without the two non-vertex calorimetric quantities),
on the full test split and on the held-out 2p2h test events, for models A and B.
Writes reports/<out>/tables/closure_ccinc.tex and closure_ccinc_macros.tex. Usage: scripts/closure_subset.py --a reports/m3_1A_z --b reports/m3_1A_z_no2p2h --out reports/m3_1A_z"""
import argparse, glob, json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, torch
from torch.utils.data import DataLoader
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sim2reco.data.compact import load_compact, CompactDataset, collate
from sim2reco.data.dataset import split_by_subrun
from sim2reco.prep.features import event_features
from sim2reco.train.m3 import load_model
from sim2reco.train.m2 import to_dev, _pad
ap = argparse.ArgumentParser(); ap.add_argument("--a", default="reports/m3_1A_z"); ap.add_argument("--b", default="reports/m3_1A_z_no2p2h"); ap.add_argument("--out", default="reports/m3_1A_z"); ap.add_argument("--steps", type=int, default=64); ap.add_argument("--seed", type=int, default=0); a = ap.parse_args()
CI = [0, 1, 2, 3, 4, 5, 6]; ALL = list(range(9))
d = load_compact(sorted(p[:-len(".truth.parquet")] for p in glob.glob("data/slim_1A/*.truth.parquet"))); split = split_by_subrun(d["subrun"], seed=0); te = np.where(split == 2)[0]
def aucs(idx, model, tf, ptf, seed):
    ds = CompactDataset(d, idx, tf, 0, prong_tf=ptf); S, XR = [], []
    with torch.no_grad():
        for b in DataLoader(ds, batch_size=4096, collate_fn=collate, num_workers=4):
            s = model.sample(to_dev(b, "cuda"), a.steps, prongs=False); S.append(torch.cat([s["minos"][:, None].float(), s["charge"][:, None].float(), s["nprong"][:, None].float(), s["x1"]], 1).cpu()); XR.append(b["x1"])
    S = torch.cat(S).numpy(); x_real_all = torch.cat(XR).numpy(); reco = d["reco_exists"][idx]; valid = ds.valid; sel = reco & valid
    x_real = x_real_all[sel]; x_fake = S[sel, 3:]; rv = idx[sel]; cls_p, mom_p, mask_p = _pad(d, rv); X, _ = event_features(cls_p, mom_p, mask_p, d["ctx"][rv]); t0 = d["tier0"][rv]
    Xr = np.concatenate([X, t0[:, 1:3], d["nprong"][rv][:, None], x_real], 1); Xf = np.concatenate([X, S[sel, :2], S[sel, 2:3], x_fake], 1)
    y = np.r_[np.ones(len(Xr)), np.zeros(len(Xf))]; rng = np.random.default_rng(seed); perm = rng.permutation(len(y)); half = len(perm) // 2; out = {}
    for name, cols in (("all", ALL), ("ccinc", CI)):
        Xm = np.concatenate([x_real[:, cols], x_fake[:, cols]]); c = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, early_stopping=True, random_state=seed).fit(Xm[perm[:half]], y[perm[:half]])
        out[f"{name}_marg"] = float(roc_auc_score(y[perm[half:]], c.predict_proba(Xm[perm[half:]])[:, 1]))
        nX = X.shape[1] + 3; keep = list(range(nX)) + [nX + j for j in cols]; Xc = np.concatenate([Xr, Xf])[:, keep]
        c = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, early_stopping=True, random_state=seed).fit(Xc[perm[:half]], y[perm[:half]])
        out[f"{name}_cond"] = float(roc_auc_score(y[perm[half:]], c.predict_proba(Xc[perm[half:]])[:, 1]))
    return out, int(sel.sum())
R = {}; M = {}
for tag, D in (("A", a.a), ("B", a.b)):
    model, tf, ptf = load_model(f"{D}/model.pt")
    for sname, idx in (("full", te), ("twop", te[d["intType"][te] == 8])):
        r, n = aucs(idx, model, tf, ptf, a.seed); R[f"{tag}_{sname}"] = dict(r, n=n); print(tag, sname, n, {k: round(v, 4) for k, v in r.items()}, flush=True)
        for k, v in r.items(): M[f"nCL{tag}{'Full' if sname == 'full' else 'TwoP'}{k.replace('_marg', 'Marg').replace('_cond', 'Cond').replace('all', 'All').replace('ccinc', 'CI')}"] = f"{v:.3f}"
out = pathlib.Path(a.out); (out / "tables").mkdir(exist_ok=True)
tex = ("\\begin{tabular}{lcccc}\n\\toprule\n & \\multicolumn{2}{c}{all nine variables} & \\multicolumn{2}{c}{non-calorimetry set (muon, vertex, recoil)} \\\\\n & marginal & + truth & marginal & + truth \\\\\n\\midrule\n"
       + "\n".join(f"{lab} & {R[k]['all_marg']:.3f} & {R[k]['all_cond']:.3f} & {R[k]['ccinc_marg']:.3f} & {R[k]['ccinc_cond']:.3f} \\\\" for k, lab in (("A_full", "model A, full test split"), ("B_full", "model B (2p2h blind), full test split"), ("A_twop", "model A, held-out 2p2h events"), ("B_twop", "model B, held-out 2p2h events")))
       + "\n\\bottomrule\n" + "\\end{tabular}\n")
(out / "tables" / "closure_ccinc.tex").write_text(tex); (out / "closure_ccinc_macros.tex").write_text("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in M.items())); json.dump(R, open(out / "closure_ccinc.json", "w"), indent=1); print(tex)
