#!/usr/bin/env python
"""Pull summaries and floor transfer with a subset of observables (e.g. without the isolated-blob energy), recomputed
from the stored calibration JSONs. Usage: scripts/bayes_pulls_subset.py reports/bayes_1A_z reports/bayes_1A_z_trainctl --exclude blobs [--draws 8]"""
import argparse, json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("test_dir"); ap.add_argument("train_dir"); ap.add_argument("--exclude", nargs="*", default=["blobs"]); ap.add_argument("--draws", type=int, default=8); a = ap.parse_args()
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
                kn = "Marg" if kind == "marginal bins" else "Cond"; M[f"nNB{tag}{kn}CtlRMS"] = f"{row['control'][0]:.2f}"; M[f"nNB{tag}{kn}OODRMS"] = f"{row['heldout'][0]:.2f}"; M[f"nNB{tag}{kn}OODWithin"] = f"{100*row['heldout'][1]:.0f}\\%"; M[f"nNB{tag}{kn}CtlBins"] = str(row['control'][2]); M[f"nNB{tag}{kn}OODBins"] = str(row['heldout'][2])
                if kind == "marginal bins": M[f"nNB{tag}FloorCtlRMS"] = f"{row['control_floor'][0]:.2f}"; M[f"nNB{tag}FloorOODRMS"] = f"{row['heldout_floor'][0]:.2f}"; M[f"nNB{tag}FloorOODWithin"] = f"{100*row['heldout_floor'][1]:.0f}\\%"
(pathlib.Path(a.test_dir) / "noblobs_macros.tex").write_text("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in M.items())); print(len(M), "macros")
