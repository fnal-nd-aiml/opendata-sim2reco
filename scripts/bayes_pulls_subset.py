#!/usr/bin/env python
"""Pull summaries and floor transfer with a subset of observables (e.g. without the isolated-blob energy), recomputed
from the stored calibration JSONs. Usage: scripts/bayes_pulls_subset.py reports/bayes_1A_z reports/bayes_1A_z_trainctl --exclude blobs [--draws 8]"""
import argparse, json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("test_dir"); ap.add_argument("train_dir"); ap.add_argument("--exclude", nargs="*", default=["blobs"]); ap.add_argument("--draws", type=int, default=8); ap.add_argument("--tag", default="NB", help="macro prefix / file tag (letters only)"); a = ap.parse_args()
K = a.draws
def arr(v, k): return np.array(v[k], float)
def rms(x): x = x[np.isfinite(x)]; return float(np.sqrt(np.mean(x ** 2))), float((np.abs(x) < 2).mean()), len(x)
def floor_from(v):
    p, r, se, ep, st = (arr(v, k) for k in ("pred", "real", "real_err", "epistemic", "surrogate_stat")); ok = np.isfinite(arr(v, "pull"))
    rel = np.sqrt(np.clip((p - r) ** 2 - se ** 2 - st ** 2 / K - ep ** 2, 0, None)) / np.where(r > 0, r, np.nan); rel[~ok] = np.nan; out = np.full_like(rel, np.nan)
    for i in range(len(rel)):
        w = rel[max(0, i - 1):i + 2]; w = w[np.isfinite(w)]
        if len(w): out[i] = np.sqrt(np.mean(w ** 2))
    g = np.sqrt(np.nanmean(rel ** 2)); out[~np.isfinite(out)] = g; return out
def pulls_floor(v, fl):
    p, r, se, ep, st = (arr(v, k) for k in ("pred", "real", "real_err", "epistemic", "surrogate_stat")); ok = np.isfinite(arr(v, "pull")); return ((p - r) / np.sqrt(ep ** 2 + st ** 2 / K + se ** 2 + (fl * p) ** 2))[ok]
M = {}
for tag in "AB":
    S = json.load(open(pathlib.Path(a.test_dir) / f"bayes_uncertainty_{tag}.json"))["binned"]; T = json.load(open(pathlib.Path(a.train_dir) / f"bayes_uncertainty_{tag}.json"))["binned"]
    for label, keep in (("all observables", lambda k: True), (f"without {', '.join(a.exclude)}", lambda k: not any(x in k for x in a.exclude))):
        for kind, isk in (("conditional means", lambda v: not v.get("marginal")), ("marginal bins", lambda v: v.get("marginal"))):
            row = {}
            for sp in ("control", "heldout"):
                ks = [k for k, v in S[sp].items() if keep(k) and isk(v)]; pl = np.concatenate([arr(S[sp][k], "pull") for k in ks]); row[sp] = rms(pl)
                if kind == "marginal bins": row[sp + "_floor"] = rms(np.concatenate([pulls_floor(S[sp][k], floor_from(T["control"][k])) for k in ks]))
            fl = f" | with floor: control {row['control_floor'][0]:.2f}, 2p2h {row['heldout_floor'][0]:.2f} ({100*row['heldout_floor'][1]:.0f}% within 2)" if kind == "marginal bins" else ""
            print(f"model {tag}, {label:22s}, {kind:17s}: control RMS {row['control'][0]:.2f} ({row['control'][2]} bins), 2p2h RMS {row['heldout'][0]:.2f} ({row['heldout'][2]} bins, {100*row['heldout'][1]:.0f}% within 2){fl}")
            if label.startswith("without"):
                kn = "Marg" if kind == "marginal bins" else "Cond"; M[f"n{a.tag}{tag}{kn}CtlRMS"] = f"{row['control'][0]:.2f}"; M[f"n{a.tag}{tag}{kn}OODRMS"] = f"{row['heldout'][0]:.2f}"; M[f"n{a.tag}{tag}{kn}OODWithin"] = f"{100*row['heldout'][1]:.0f}\\%"; M[f"n{a.tag}{tag}{kn}CtlBins"] = str(row['control'][2]); M[f"n{a.tag}{tag}{kn}OODBins"] = str(row['heldout'][2])
                if kind == "marginal bins": M[f"n{a.tag}{tag}FloorCtlRMS"] = f"{row['control_floor'][0]:.2f}"; M[f"n{a.tag}{tag}FloorOODRMS"] = f"{row['heldout_floor'][0]:.2f}"; M[f"n{a.tag}{tag}FloorOODWithin"] = f"{100*row['heldout_floor'][1]:.0f}\\%"
(pathlib.Path(a.test_dir) / f"subset_{a.tag}_macros.tex").write_text("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in M.items())); print(len(M), "macros")
from sim2reco.eval import plots
fig, axs = plots.plt.subplots(1, 2, figsize=(10, 3.4), sharey=True); b = np.linspace(-6, 6, 49)
for ax, tag in zip(axs, "AB"):
    S = json.load(open(pathlib.Path(a.test_dir) / f"bayes_uncertainty_{tag}.json"))["binned"]; T = json.load(open(pathlib.Path(a.train_dir) / f"bayes_uncertainty_{tag}.json"))["binned"]
    for sp, col, lab in (("control", plots.PALETTE["third"], "control"), ("heldout", plots.PALETTE["model"], "held-out 2p2h")):
        ks = [k for k, v in S[sp].items() if v.get("marginal") and not any(x in k for x in a.exclude)]
        p0 = np.concatenate([arr(S[sp][k], "pull") for k in ks]); p0 = p0[np.isfinite(p0)]; p1 = np.concatenate([pulls_floor(S[sp][k], floor_from(T["control"][k])) for k in ks])
        ax.hist(np.clip(p0, -5.9, 5.9), b, histtype="step", color=col, lw=1.0, ls=":", label=f"{lab}, epistemic only: RMS {np.sqrt(np.mean(p0**2)):.2f}")
        ax.hist(np.clip(p1, -5.9, 5.9), b, histtype="step", color=col, lw=1.6, label=f"{lab}, + floor: RMS {np.sqrt(np.mean(p1**2)):.2f}, {100*(np.abs(p1)<2).mean():.0f}% within 2")
    xx = np.linspace(-6, 6, 300); ax.plot(xx, len(p1) * (b[1] - b[0]) * np.exp(-xx ** 2 / 2) / np.sqrt(2 * np.pi), ":", color="gray", lw=1)
    ax.set_title(f"model {tag}, marginal bins without {', '.join(a.exclude)}", fontsize=10, loc="left"); ax.set_xlabel("pull"); ax.legend(frameon=False, fontsize=6.5, loc="upper left")
axs[0].set_ylabel("bins"); plots.save(fig, pathlib.Path(a.test_dir) / "figures" / f"bayes_pulls_subset_{a.tag}.png")
