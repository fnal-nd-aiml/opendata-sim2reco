#!/usr/bin/env python
"""Combined calibration figure for models A (informed) and B (held-out-mode blind) from bayes_uncertainty_{A,B}.json.
Usage: scripts/bayes_calibration_figure.py reports/bayes_1A"""
import json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from sim2reco.eval import plots

bd = pathlib.Path(sys.argv[1]); U = {t: json.load(open(bd / f"bayes_uncertainty_{t}.json")) for t in ("A", "B")}
C_CTL, C_HO = plots.PALETTE["third"], plots.PALETTE["model"]
# two rows (A, B) x three columns: recoil vs KE, efficiency vs n_had, pull histogram over all observables and bins
fig, axs = plots.plt.subplots(2, 3, figsize=(12.5, 6.4))
for row, tag, title in ((0, "A", "model A (trained with 2p2h)"), (1, "B", "model B (2p2h blind)")):
    for col, key in ((0, "recoil_vs_ke"), (1, "eff_vs_nhad")):
        ax = axs[row, col]
        for sp, c, ls, lab in (("control", C_CTL, "s--", "control"), ("heldout", C_HO, "^:", "held-out 2p2h")):
            v = U[tag]["binned"][sp][key]; x = np.array(v["centers"]); r_, se, mu, ep = (np.array(v[k]) for k in ("real", "real_err", "pred", "epistemic"))
            ax.errorbar(x, r_, yerr=se, fmt="o", color=c, ms=3, alpha=0.7, label=f"open dataset, {lab}"); ax.plot(x, mu, ls, color=c, ms=4, label=f"surrogate, {lab}"); ax.fill_between(x, mu - ep, mu + ep, color=c, alpha=0.2)
        if v["logx"]: ax.set_xscale("log")
        ax.set_xlabel(v["xlabel"], fontsize=9); ax.set_ylabel(v["ylabel"], fontsize=9); ax.set_title(title, fontsize=9, loc="left"); ax.legend(frameon=False, fontsize=6)
    ax = axs[row, 2]; b = np.linspace(-4, 4, 33)
    for sp, c, lab in (("control", C_CTL, "control"), ("heldout", C_HO, "held-out 2p2h")):
        pulls = np.concatenate([np.array(v["pull"]) for v in U[tag]["binned"][sp].values()]); pulls = pulls[np.isfinite(pulls)]
        ax.hist(pulls, bins=b, histtype="step", color=c, label=f"{lab}: {len(pulls)} bins, RMS {np.sqrt(np.mean(pulls**2)):.2f}")
    xx = np.linspace(-4, 4, 200); n_ref = len(pulls); ax.plot(xx, n_ref * (b[1] - b[0]) * np.exp(-xx**2 / 2) / np.sqrt(2 * np.pi), ":", color="gray", lw=1, label="unit normal")
    ax.set_xlabel("pull: (surrogate - open dataset) / total uncertainty", fontsize=9); ax.set_ylabel("bins", fontsize=9); ax.set_title(title, fontsize=9, loc="left"); ax.legend(frameon=False, fontsize=6)
plots.save(fig, bd / "figures" / "bayes_calibration_AB.png")
# all observables for model B, held-out, as a supplementary grid
keys = list(U["B"]["binned"]["heldout"]); fig, axs = plots.plt.subplots(2, 4, figsize=(14, 6)); axs = axs.ravel()
for ax, key in zip(axs, keys):
    for sp, c, ls, lab in (("control", C_CTL, "s--", "control"), ("heldout", C_HO, "^:", "held-out 2p2h")):
        v = U["B"]["binned"][sp][key]; x = np.array(v["centers"]); r_, se, mu, ep = (np.array(v[k]) for k in ("real", "real_err", "pred", "epistemic"))
        ax.errorbar(x, r_, yerr=se, fmt="o", color=c, ms=3, alpha=0.7, label=f"open dataset, {lab}"); ax.plot(x, mu, ls, color=c, ms=3, label=f"surrogate B, {lab}"); ax.fill_between(x, mu - ep, mu + ep, color=c, alpha=0.2)
    if v["logx"]: ax.set_xscale("log")
    ax.set_xlabel(v["xlabel"], fontsize=8); ax.set_ylabel(v["ylabel"], fontsize=8); ax.legend(frameon=False, fontsize=5)
for ax in axs[len(keys):]: ax.axis("off")
plots.save(fig, bd / "figures" / "bayes_calibration_B_all.png"); print("figures written")
