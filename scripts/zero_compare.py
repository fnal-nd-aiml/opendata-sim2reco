#!/usr/bin/env python
"""Previous (dequantised-zero) vs zero-flag models: closure, 2p2h holdout, marginal pulls, training time.
Writes reports/m3_1A_z/tables/zero_compare.tex and reports/m3_1A_z/zero_compare_macros.tex."""
import json, pathlib, numpy as np
R = pathlib.Path("reports"); M = {}
def auc(d): m = json.load(open(R / d / "metrics.json")); return m["classifier_auc_marginal"], m["classifier_auc_conditional"]
def mpull(d, tag, sp, key=None):
    U = json.load(open(R / d / f"bayes_uncertainty_{tag}.json"))["binned"][sp]
    ks = [key] if key else [k for k, v in U.items() if v.get("marginal")]
    pl = np.concatenate([np.array(U[k]["pull"], float) for k in ks]); pl = pl[np.isfinite(pl)]; return float(np.sqrt(np.mean(pl ** 2)))
def hours(d): h = json.load(open(R / d / "history.json")); return h[-1]["time"] / 3600
rows = []
for name, old, new in (("A, full test", "m3_1A", "m3_1A_z"), ("B, full test", "m3_1A_no2p2h", "m3_1A_z_no2p2h"), ("A on held-out 2p2h", "m3_1A_on2p2h", "m3_1A_z_on2p2h"), ("B on held-out 2p2h", "m3_1A_no2p2h_on2p2h", "m3_1A_z_no2p2h_on2p2h")):
    (om, oc), (nm, nc) = auc(old), auc(new); rows.append(f"closure AUC, {name} & {om:.3f} / {oc:.3f} & {nm:.3f} / {nc:.3f} \\\\")
    key = name.replace(",", "").replace(" ", "").replace("-", "")
    M[f"nZC{key}OldM"], M[f"nZC{key}OldC"], M[f"nZC{key}NewM"], M[f"nZC{key}NewC"] = f"{om:.3f}", f"{oc:.3f}", f"{nm:.3f}", f"{nc:.3f}"
rows.append("\\midrule")
for label, key in (("all marginal bins", None), ("prong count", "marg_nprong"), ("recoil", "marg_recoil"), ("non-vertex energy", "marg_nonvtx100"), ("isolated blobs", "marg_blobs"), ("muon $\\log P$ ratio", "marg_muP"), ("muon $\\Delta\\theta_x$", "marg_dthx")):
    o = (mpull("bayes_1A", "A", "control", key), mpull("bayes_1A", "A", "heldout", key)); n = (mpull("bayes_1A_z", "A", "control", key), mpull("bayes_1A_z", "A", "heldout", key))
    rows.append(f"marginal pull RMS, {label} (control / 2p2h) & {o[0]:.2f} / {o[1]:.2f} & {n[0]:.2f} / {n[1]:.2f} \\\\")
    k2 = (key or "all").replace("marg_", "").replace("100", "").title(); M[f"nZCPull{k2}OldCtl"], M[f"nZCPull{k2}OldOOD"], M[f"nZCPull{k2}NewCtl"], M[f"nZCPull{k2}NewOOD"] = f"{o[0]:.2f}", f"{o[1]:.2f}", f"{n[0]:.2f}", f"{n[1]:.2f}"
rows.append("\\midrule")
ho = hours("m2_1A") + hours("m3_1A"); hn = hours("m2_1A_z") + hours("m3_1A_z"); rows.append(f"training time per model, both stages & {ho:.1f}\\,h & {hn:.1f}\\,h \\\\")
M["nZCHoursOld"], M["nZCHoursNew"] = f"{ho:.1f}", f"{hn:.1f}"; M["nZCHoursMtwoNew"], M["nZCHoursMthreeNew"] = f"{hours('m2_1A_z'):.1f}", f"{hours('m3_1A_z'):.1f}"
tex = "\\begin{tabular}{lcc}\n\\toprule\n & previous model & zero-flag model \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n" + "\\end{tabular}\n"
(R / "m3_1A_z" / "tables").mkdir(exist_ok=True); (R / "m3_1A_z" / "tables" / "zero_compare.tex").write_text(tex)
(R / "m3_1A_z" / "zero_compare_macros.tex").write_text("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in M.items())); print(tex); print(len(M), "macros")
