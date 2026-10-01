#!/usr/bin/env python
"""Compare input-definition variants: table of key metrics on the full test split and on 2p2h, plus figures of the
quantities the neutron question touches (calorimetry blocks, zero-charged-hadron prong rate, 2p2h recoil).

Usage: scripts/variants_compare.py OUT_DIR --variants NAME:FULL_DIR:TWOP2H_DIR [...]
"""
import argparse, json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from sim2reco.eval import plots

ap = argparse.ArgumentParser(); ap.add_argument("out_dir"); ap.add_argument("--variants", nargs="+", required=True)
a = ap.parse_args(); out = pathlib.Path(a.out_dir); (out / "figures").mkdir(parents=True, exist_ok=True); (out / "tables").mkdir(exist_ok=True)
V = {}
for spec in a.variants:
    name, full, twop = spec.split(":")
    V[name] = {"full": (json.load(open(f"{full}/metrics.json")), json.load(open(f"{full}/metrics_tier2.json"))),
               "2p2h": (json.load(open(f"{twop}/metrics.json")), json.load(open(f"{twop}/metrics_tier2.json")))}
def metrics(m, t):
    byv = m["classifier_auc_by_variable"]; pv = {r[0]: r for r in t["validation"]["prongs_vs_true_charged"]}
    return {"auc_marg": m["classifier_auc_marginal"], "auc_cond": m["classifier_auc_conditional"], "auc_cond_calo": m["classifier_auc_conditional_by_block"]["calorimetry"],
            "auc_blobs": byv.get("log_nonvtx_iso_blobs_E", np.nan), "auc_nv100": byv.get("log_recoil_nonvtx100", np.nan), "auc_recoil": byv.get("log_recoil_E", np.nan),
            "prong_auc": t["closure_prongs"]["event_summary_only"], "prong_auc_cond": t["closure_prongs"]["truth_plus_event_summary"],
            "reco_ll": m["tier0"]["reco_exists"]["logloss"], "mult_ll": m["multiplicity"]["logloss"],
            "w1_recoil": m["tier1_model_space"]["log_recoil_E"]["w1"], "w1_blobs": m["tier1_model_space"]["log_nonvtx_iso_blobs_E"]["w1"],
            "zero_ch_prongs_real": pv[0][3] if 0 in pv else np.nan, "zero_ch_prongs_fake": pv[0][4] if 0 in pv else np.nan, "n_test": m["n_test"]}
rows = {(n, s): metrics(*V[n][s]) for n in V for s in ("full", "2p2h")}
json.dump({f"{n}/{s}": r for (n, s), r in rows.items()}, open(out / "summary.json", "w"), indent=1, default=float)
labels = [("closure AUC, reco only", "auc_marg", ".3f"), ("closure AUC, truth + reco", "auc_cond", ".3f"), ("cond.\\ AUC calorimetry block", "auc_cond_calo", ".3f"),
          ("single-variable AUC: isolated blobs", "auc_blobs", ".3f"), ("single-variable AUC: non-vertex 100\\,mm", "auc_nv100", ".3f"), ("single-variable AUC: recoil", "auc_recoil", ".3f"),
          ("prong closure AUC (summary / + truth)", None, None), ("reconstruction log loss", "reco_ll", ".4f"), ("multiplicity log loss", "mult_ll", ".4f"),
          ("$W_1$ $\\log E_\\mathrm{recoil}$ / isolated blobs", None, None), ("prongs per event at 0 true charged hadrons (data / surrogate)", None, None), ("test events", "n_test", ",d")]
for s in ("full", "2p2h"):
    cols = list(V); lines = []
    for lab, k, f in labels:
        if k: vals = [format(rows[(n, s)][k], f) for n in cols]
        elif lab.startswith("prong"): vals = [f"{rows[(n,s)]['prong_auc']:.3f} / {rows[(n,s)]['prong_auc_cond']:.3f}" for n in cols]
        elif lab.startswith("$W_1$"): vals = [f"{rows[(n,s)]['w1_recoil']:.3f} / {rows[(n,s)]['w1_blobs']:.3f}" for n in cols]
        else: vals = [f"{rows[(n,s)]['zero_ch_prongs_real']:.2f} / {rows[(n,s)]['zero_ch_prongs_fake']:.2f}" for n in cols]
        lines.append(lab + " & " + " & ".join(vals) + " \\\\")
    (out / "tables" / f"variants_{s}.tex").write_text("\\begin{tabular}{l" + "c" * len(cols) + "}\n\\toprule\n & " + " & ".join(cols) + " \\\\\n\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")
# figure: closure AUCs and the zero-charged-hadron prong rate per variant, full vs 2p2h
fig, axs = plots.plt.subplots(1, 3, figsize=(12, 3.3)); x = np.arange(len(V)); w = 0.38
for ax, key, yl in ((axs[0], "auc_cond", "closure AUC, truth + reco"), (axs[1], "auc_cond_calo", "conditional AUC, calorimetry block"), (axs[2], "zero_ch_prongs_fake", "surrogate prongs/event, 0 true charged hadrons")):
    ax.bar(x - w / 2, [rows[(n, "full")][key] for n in V], w, color=plots.PALETTE["third"], label="full test split")
    ax.bar(x + w / 2, [rows[(n, "2p2h")][key] for n in V], w, color=plots.PALETTE["model"], label="2p2h test events")
    if key == "zero_ch_prongs_fake":
        ax.axhline(rows[(list(V)[0], "full")]["zero_ch_prongs_real"], color=plots.PALETTE["third"], ls=":", lw=1); ax.axhline(rows[(list(V)[0], "2p2h")]["zero_ch_prongs_real"], color=plots.PALETTE["model"], ls=":", lw=1, label="open dataset (dotted)")
    else: ax.axhline(0.5, color="gray", ls=":", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(list(V)); ax.set_ylabel(yl, fontsize=9); ax.legend(frameon=False, fontsize=7)
plots.save(fig, out / "figures" / "variants_summary.png")
print(json.dumps({f"{n}/{s}": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()} for (n, s), r in rows.items()}, indent=1))
