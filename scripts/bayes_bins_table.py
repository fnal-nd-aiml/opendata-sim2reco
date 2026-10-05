#!/usr/bin/env python
"""LaTeX table of the bins used in the calibration pulls (edges as defined in bayes_uncertainty_eval.py) and the
number of populated bins per sample. Usage: scripts/bayes_bins_table.py reports/bayes_1A [tag=B]"""
import json, pathlib, sys
import numpy as np
bd = pathlib.Path(sys.argv[1]); tag = sys.argv[2] if len(sys.argv) > 2 else "B"
U = json.load(open(bd / f"bayes_uncertainty_{tag}.json"))["binned"]
ke = np.logspace(np.log10(30), np.log10(20000), 17); pe = np.logspace(np.log10(0.5), np.log10(40), 15); ne = np.arange(-0.5, 12.5, 1)
def fmt(e, kind):
    if kind == "int": return "integers 0, 1, \\ldots, 11 (one bin each)"
    if kind == "mev": return ", ".join(f"{v:.0f}" for v in e) + "\\,MeV"
    return ", ".join(f"{v:.2g}" if v < 10 else f"{v:.0f}" for v in e) + "\\,GeV"
rows = [("reconstruction probability", "true hadrons after cuts", "eff_vs_nhad", ne, "int"), ("mean reco prongs", "true hadrons after cuts", "nprong_vs_nhad", ne, "int"),
        ("mean $\\log E_\\mathrm{recoil}$", "true hadronic KE", "recoil_vs_ke", ke, "mev"), ("mean $\\log E$ non-vertex 100\\,mm", "true hadronic KE", "nonvtx100_vs_ke", ke, "mev"),
        ("mean $\\log E$ isolated blobs", "true hadronic KE", "blobs_vs_ke", ke, "mev"), ("mean $\\log(P/P_\\mathrm{true})$ muon", "true muon momentum", "muP_vs_P", pe, "gev"), ("mean $\\Delta\\theta_x$ muon", "true muon momentum", "dthx_vs_P", pe, "gev")]
lines = []
for name, var, key, e, kind in rows:
    nc = int(np.isfinite(U["control"][key]["pull"]).sum()); nh = int(np.isfinite(U["heldout"][key]["pull"]).sum())
    lines.append(f"{name} & {var} & {len(e)-1} & {nc} / {nh} & \\parbox[t]{{7.5cm}}{{\\raggedright\\footnotesize {fmt(e, kind)}}} \\\\")
lines.append("\\midrule\n\\multicolumn{5}{l}{\\emph{Marginal histograms (fraction of reconstructed events per bin, binned in the observable itself)}} \\\\")
for key, name in (("marg_nprong", "reco prong count"), ("marg_recoil", "$\\log E_\\mathrm{recoil}$"), ("marg_nonvtx100", "$\\log E$ non-vertex 100\\,mm"), ("marg_blobs", "$\\log E$ isolated blobs"), ("marg_muP", "$\\log(P/P_\\mathrm{true})$ muon"), ("marg_dthx", "$\\Delta\\theta_x$ muon")):
    v = U["control"][key]; e = np.array(v["edges"]); nc = int(np.isfinite(U["control"][key]["pull"]).sum()); nh = int(np.isfinite(U["heldout"][key]["pull"]).sum())
    edges = "integers 0, 1, \\ldots, 8 (one bin each)" if key == "marg_nprong" else f"{len(e)-1} equal bins from {e[0]:.2f} to {e[-1]:.2f} (model space; 0.5th to 99.5th percentile of the control sample)"
    lines.append(f"{name} & itself & {len(e)-1} & {nc} / {nh} & \\parbox[t]{{7.5cm}}{{\\raggedright\\footnotesize {edges}}} \\\\")
tex = ("\\begin{tabular}{llccl}\n\\toprule\nObservable & binned against & bins & populated (control / held-out) & bin edges \\\\\n\\midrule\n" + "\n".join(lines) +
       "\n\\bottomrule\n\\end{tabular}\n")
(bd / "tables").mkdir(exist_ok=True); (bd / "tables" / "bins.tex").write_text(tex); print(tex)
