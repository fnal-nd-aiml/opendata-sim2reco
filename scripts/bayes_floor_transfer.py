#!/usr/bin/env python
"""Model-error floor test.  A per-bin relative floor on the marginal bin fractions is derived on the TRAINING-split
control (the excess of (surrogate - open dataset)^2 beyond statistics and epistemic variance, smoothed over 3 bins),
and applied, in quadrature with the epistemic and statistical terms, to the TEST-split control and to the held-out
2p2h events.  Usage: scripts/bayes_floor_transfer.py reports/bayes_1A reports/bayes_1A_trainctl [--draws 8]"""
import argparse, json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from sim2reco.eval import plots
ap = argparse.ArgumentParser(); ap.add_argument("test_dir"); ap.add_argument("train_dir"); ap.add_argument("--draws", type=int, default=8); a = ap.parse_args()
K = a.draws; KEYS = ("marg_nprong", "marg_recoil", "marg_nonvtx100", "marg_blobs", "marg_muP", "marg_dthx")
def arr(v, k): return np.array(v[k], float)
def floor_from(v, smooth):
    p, r, se, ep, st = (arr(v, k) for k in ("pred", "real", "real_err", "epistemic", "surrogate_stat")); ok = np.isfinite(arr(v, "pull"))
    rel = np.sqrt(np.clip((p - r) ** 2 - se ** 2 - st ** 2 / K - ep ** 2, 0, None)) / np.where(r > 0, r, np.nan); rel[~ok] = np.nan
    out = np.full_like(rel, np.nan)
    for i in range(len(rel)):
        w = rel[max(0, i - 1):i + 2]; w = w[np.isfinite(w)]
        if len(w): out[i] = np.sqrt(np.mean(w ** 2)) if smooth == "rms" else w.max()
    g = np.sqrt(np.nanmean(rel ** 2)); out[~np.isfinite(out)] = g; return out, g
def pulls(v, fl):
    p, r, se, ep, st = (arr(v, k) for k in ("pred", "real", "real_err", "epistemic", "surrogate_stat")); ok = np.isfinite(arr(v, "pull"))
    return ((p - r) / np.sqrt(ep ** 2 + st ** 2 / K + se ** 2 + (fl * p) ** 2))[ok]
R = {}
for tag in "AB":
    T = json.load(open(pathlib.Path(a.train_dir) / f"bayes_uncertainty_{tag}.json"))["binned"]; S = json.load(open(pathlib.Path(a.test_dir) / f"bayes_uncertainty_{tag}.json"))["binned"]
    R[tag] = {}
    for variant, (smooth, fac) in {"rms": ("rms", 1.0), "max": ("max", 1.0), "rms_x1.5": ("rms", 1.5)}.items():
        res = {}; allc, allh = [], []
        for key in KEYS:
            if key not in T["control"]: continue
            fl, g = floor_from(T["control"][key], smooth); fl = fl * fac
            pc, ph = pulls(S["control"][key], fl), pulls(S["heldout"][key], fl); pc0, ph0 = arr(S["control"][key], "pull"), arr(S["heldout"][key], "pull"); pc0, ph0 = pc0[np.isfinite(pc0)], ph0[np.isfinite(ph0)]
            res[key] = {"floor_rms": float(g * fac), "ctl_rms_before": float(np.sqrt(np.mean(pc0 ** 2))), "ctl_rms": float(np.sqrt(np.mean(pc ** 2))), "ctl_within2": float((np.abs(pc) < 2).mean()),
                        "ood_rms_before": float(np.sqrt(np.mean(ph0 ** 2))), "ood_rms": float(np.sqrt(np.mean(ph ** 2))), "ood_within2": float((np.abs(ph) < 2).mean()), "n_ctl": int(len(pc)), "n_ood": int(len(ph))}
            allc.append(pc); allh.append(ph)
        allc, allh = np.concatenate(allc), np.concatenate(allh)
        res["all"] = {"ctl_rms": float(np.sqrt(np.mean(allc ** 2))), "ctl_within2": float((np.abs(allc) < 2).mean()), "ood_rms": float(np.sqrt(np.mean(allh ** 2))), "ood_within2": float((np.abs(allh) < 2).mean()), "n_ctl": int(len(allc)), "n_ood": int(len(allh))}
        R[tag][variant] = res
        print(f"model {tag} floor variant {variant:8s}: test control RMS {res['all']['ctl_rms']:.2f} ({100*res['all']['ctl_within2']:.0f}% within 2) | held-out 2p2h RMS {res['all']['ood_rms']:.2f} ({100*res['all']['ood_within2']:.0f}% within 2) | per observable 2p2h: " + ", ".join(f"{k[5:]} {res[k]['ood_rms']:.2f}" for k in KEYS if k in res))
out = pathlib.Path(a.test_dir); json.dump(R, open(out / "floor_transfer.json", "w"), indent=1)
# figure: pull distributions before / after the floor, split into the non-calorimetry set (recoil, muon, prong count)
# and the two non-vertex calorimetric quantities (hard-edge residual), model A left, model B right
GROUPS = (("non-calorimetry set: recoil, muon $\\log P$ ratio, $\\Delta\\theta_x$, prong count", ("marg_nprong", "marg_recoil", "marg_muP", "marg_dthx")), ("non-vertex energy within 100 mm and isolated-blob energy", ("marg_nonvtx100", "marg_blobs")))
fig, axs = plots.plt.subplots(2, 2, figsize=(10, 6.4), sharey="row"); b = np.linspace(-6, 6, 49)
for col, tag in enumerate("AB"):
    T = json.load(open(pathlib.Path(a.train_dir) / f"bayes_uncertainty_{tag}.json"))["binned"]; S = json.load(open(pathlib.Path(a.test_dir) / f"bayes_uncertainty_{tag}.json"))["binned"]
    for row, (gname, keys) in enumerate(GROUPS):
        ax = axs[row, col]
        for sp, c_ in (("control", plots.PALETTE["third"]), ("heldout", plots.PALETTE["model"])):
            ks = [k for k in keys if k in S[sp]]; p0 = np.concatenate([arr(S[sp][k], "pull")[np.isfinite(arr(S[sp][k], "pull"))] for k in ks]); p1 = np.concatenate([pulls(S[sp][k], floor_from(T["control"][k], "rms")[0]) for k in ks])
            lab = "control (test split)" if sp == "control" else "held-out 2p2h"
            ax.hist(np.clip(p0, -5.9, 5.9), b, histtype="step", color=c_, lw=1.0, ls=":", label=f"{lab}, epistemic only: RMS {np.sqrt(np.mean(p0**2)):.2f}")
            ax.hist(np.clip(p1, -5.9, 5.9), b, histtype="step", color=c_, lw=1.6, label=f"{lab}, + floor: RMS {np.sqrt(np.mean(p1**2)):.2f}, {100*(np.abs(p1)<2).mean():.0f}% within 2")
        xx = np.linspace(-6, 6, 300); ax.plot(xx, len(p1) * (b[1] - b[0]) * np.exp(-xx ** 2 / 2) / np.sqrt(2 * np.pi), ":", color="gray", lw=1)
        ax.set_title(f"model {tag}: {gname}", fontsize=8.5, loc="left"); ax.legend(frameon=False, fontsize=6.3, loc="upper left")
        if row == 1: ax.set_xlabel("pull")
        if col == 0: ax.set_ylabel("bins")
plots.save(fig, out / "figures" / "bayes_floor_transfer.png"); print("figure written")
