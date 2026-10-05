#!/usr/bin/env python
"""One figure per calibration observable (model A left, model B right): open dataset as points with statistical
errors, surrogate binned mean as a line, epistemic band (dark) and total surrogate-side band (light; epistemic plus
sampling error over K draws, the quantity in the pull denominator), for control and held-out 2p2h events.
Also a two-panel pull summary. Usage: scripts/bayes_binned_figures.py reports/bayes_1A [--draws 8]"""
import json, pathlib, sys, argparse
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from sim2reco.eval import plots

ap = argparse.ArgumentParser(); ap.add_argument("dir"); ap.add_argument("--draws", type=int, default=8); a = ap.parse_args()
bd = pathlib.Path(a.dir); U = {t: json.load(open(bd / f"bayes_uncertainty_{t}.json")) for t in ("A", "B")}
C = {"control": plots.PALETTE["third"], "heldout": plots.PALETTE["model"]}; LAB = {"control": "control (in distribution)", "heldout": "held-out 2p2h"}
ALL = U["B"]["binned"]["heldout"]; keys = [k for k in ALL if not ALL[k].get("marginal")]; mkeys = [k for k in ALL if ALL[k].get("marginal")]
for key in keys + mkeys:
    fig, axs = plots.plt.subplots(2, 2, figsize=(10, 5.6), sharex="col", gridspec_kw={"height_ratios": (2, 1), "hspace": 0.06, "wspace": 0.22, "bottom": 0.2})
    for col, (tag, title) in enumerate((("A", "model A (trained with 2p2h)"), ("B", "model B (2p2h blind)"))):
        ax, axd = axs[0, col], axs[1, col]
        for sp in ("control", "heldout"):
            v = U[tag]["binned"][sp][key]; x = np.array(v["centers"]); r_, se, mu, ep = (np.array(v[k], float) for k in ("real", "real_err", "pred", "epistemic"))
            tot = np.sqrt(ep ** 2 + (np.array(v["surrogate_stat"], float) ** 2 / a.draws if "surrogate_stat" in v else 0))
            ok = np.isfinite(np.array(v["pull"], float)); x, r_, se, mu, ep, tot = (q[ok] for q in (x, r_, se, mu, ep, tot))
            if v.get("marginal"): e_ = np.array(v["edges"]); ax.stairs(np.where(np.isfinite(np.array(v["pred"], float)), np.array(v["pred"], float), np.nan), e_, color=C[sp], lw=1.2, label=f"surrogate, {LAB[sp]}"); ax.set_yscale("log")
            else: ax.plot(x, mu, "-", color=C[sp], lw=1.2, label=f"surrogate, {LAB[sp]}")
            ax.errorbar(x, r_, yerr=se, fmt="o", color=C[sp], ms=3.5, mfc="white", mew=1.2, capsize=0, label=f"open dataset ± stat, {LAB[sp]}")
            # difference panel: bands are the surrogate-side uncertainty around zero, points carry the open-dataset error
            axd.fill_between(x, -tot, tot, color=C[sp], alpha=0.12, linewidth=0, label=f"{LAB[sp]}: ± epistemic ⊕ sampling/K (light)" if col == 0 else None)
            axd.fill_between(x, -ep, ep, color=C[sp], alpha=0.35, linewidth=0, label=f"{LAB[sp]}: ± epistemic (dark)" if col == 0 else None)
            axd.errorbar(x, mu - r_, yerr=se, fmt="o", color=C[sp], ms=3.5, mfc="white", mew=1.2, capsize=0)
        if v["logx"]: ax.set_xscale("log")
        axd.axhline(0, color="k", lw=0.6); axd.set_xlabel(v["xlabel"]); ax.set_title(title, fontsize=10, loc="left"); ax.legend(frameon=False, fontsize=6.5)
        lim = 1.1 * max(abs(l) for l in axd.get_ylim()); axd.set_ylim(-lim, lim)
        if v.get("marginal"): ax.set_ylim(bottom=max(ax.get_ylim()[0], 1e-4))
    axs[0, 0].set_ylabel(v["ylabel"]); axs[1, 0].set_ylabel("surrogate − open")
    h, l = axs[1, 0].get_legend_handles_labels(); fig.legend(h, l, frameon=False, fontsize=7, ncol=2, loc="lower center", bbox_to_anchor=(0.5, -0.02), title="bands in the lower panels: surrogate-side uncertainty around zero; points carry the open-dataset statistical error", title_fontsize=7)
    fig.savefig(bd / "figures" / f"bayes_binned_{key}.png", bbox_inches="tight"); plots.plt.close(fig)
# pull summaries: conditional means (bayes_pulls_AB) and marginal histograms (bayes_pulls_marginal_AB)
def pull_summary(sel, fname, what):
    fig, axs = plots.plt.subplots(1, 2, figsize=(10, 3.4), sharey=True); b = np.linspace(-6, 6, 49)
    for ax, tag, title in ((axs[0], "A", "model A (trained with 2p2h)"), (axs[1], "B", "model B (2p2h blind)")):
        n_ref = 0
        for sp in ("control", "heldout"):
            pulls = np.concatenate([np.array(v["pull"], float) for k, v in U[tag]["binned"][sp].items() if k in sel]); pulls = pulls[np.isfinite(pulls)]; n_ref = max(n_ref, len(pulls))
            ax.hist(np.clip(pulls, -5.9, 5.9), bins=b, histtype="step", color=C[sp], lw=1.4, label=f"{LAB[sp]}: {len(pulls)} bins, RMS {np.sqrt(np.mean(pulls**2)):.2f}, {100*(np.abs(pulls)<2).mean():.0f}% within 2")
        xx = np.linspace(-6, 6, 300); ax.plot(xx, n_ref * (b[1] - b[0]) * np.exp(-xx ** 2 / 2) / np.sqrt(2 * np.pi), ":", color="gray", lw=1, label="unit normal (control count)")
        ax.set_xlabel("pull: (surrogate − open dataset) / total uncertainty"); ax.set_title(f"{title}, {what}", fontsize=10, loc="left"); ax.legend(frameon=False, fontsize=6.5, loc="upper left")
    axs[0].set_ylabel("bins"); plots.save(fig, bd / "figures" / fname)
pull_summary(keys, "bayes_pulls_AB.png", "binned means"); pull_summary(mkeys, "bayes_pulls_marginal_AB.png", "marginal bin fractions")
print("binned figures:", len(keys), "marginal figures:", len(mkeys), "+ 2 pull summaries")
