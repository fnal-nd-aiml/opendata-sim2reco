#!/usr/bin/env python
"""GENIE vs NuWro through the surrogate, on matched fiducial-tracker CC nu_mu samples.
Three things side by side: the open dataset's reconstruction of the GENIE events (ref_* columns of the GENIE truth
file), the surrogate on GENIE, and the surrogate on NuWro. Truth-level differences between the generators are shown
first, so reconstructed-level differences can be read against them. Outputs figures, tables, macros and a JSON in
reports/nuwro_compare/. Usage: scripts/nuwro_compare.py [--genie-truth ... --genie-sur ... --nuwro-truth ... --nuwro-sur ...]"""
import argparse, json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import awkward as ak, numpy as np, uproot
from sklearn.metrics import roc_auc_score
from sim2reco.prep.particles import select_from_tuple
from sim2reco.prep.frames import theta_phi_beam
from sim2reco.eval import plots
ap = argparse.ArgumentParser()
ap.add_argument("--genie-truth", default="data/genie_tracker/genie_me_fhc_tracker.truth.parquet"); ap.add_argument("--genie-sur", default="data/genie_tracker/surrogate_Az_genie_me_fhc_tracker.root")
ap.add_argument("--nuwro-truth", default="data/nuwro_2026-10/nuwro_me_fhc_tracker.truth.parquet"); ap.add_argument("--nuwro-sur", default="data/nuwro_2026-10/surrogate_Az_nuwro_me_fhc_tracker.root")
ap.add_argument("--out", default="reports/nuwro_compare"); a = ap.parse_args()
out = pathlib.Path(a.out); (out / "figures").mkdir(parents=True, exist_ok=True); (out / "tables").mkdir(exist_ok=True)
MP = 938.272; CH = {1: "QE", 2: "RES", 3: "DIS", 4: "COH", 8: "2p2h", 0: "other"}
C = {"ref": plots.PALETTE["real"], "genie": plots.PALETTE["model"], "nuwro": plots.PALETTE["third"]}
LAB = {"ref": "open dataset (GENIE events)", "genie": "surrogate on GENIE", "nuwro": "surrogate on NuWro"}

def load(truth_path, sur_path, has_ref):
    t = ak.from_parquet(truth_path); n = len(t)
    mu = ak.to_numpy(t["mc_primFSLepton"]); th, _ = theta_phi_beam(mu[:, 0], mu[:, 1], mu[:, 2])
    Enu = ak.to_numpy(t["mc_incomingE"]); pmu = np.linalg.norm(mu[:, :3], axis=1)
    Q2 = 2 * Enu * (mu[:, 3] - pmu * np.cos(th)) - 105.66 ** 2  # lepton-defined, same for both generators (GENIE's stored Q2 has sentinels)
    W = np.sqrt(np.clip(MP ** 2 + 2 * MP * (Enu - mu[:, 3]) - Q2, 0, None))
    parts = select_from_tuple(t, 10.0, True); cls = parts["cls"]; nhad = ak.to_numpy(ak.sum(cls >= 4, axis=1))
    # hadronic KE of the kept tokens from the class masses (the selector returns cls, px, py, pz)
    MASS = np.array([0, 105.66, 0.511, 0, 938.27, 139.57, 139.57, 134.98, 493.68, 497.61, 1115.7, 0, 939.57])
    P = np.sqrt(parts["px"] ** 2 + parts["py"] ** 2 + parts["pz"] ** 2); m = ak.Array(MASS[ak.to_numpy(ak.flatten(cls)).astype(int)]); m = ak.unflatten(m, ak.num(cls))
    keh = ak.to_numpy(ak.sum(ak.where(cls >= 4, np.sqrt(P ** 2 + m ** 2) - m, 0.0), axis=1))
    T = {"n": n, "Enu": Enu / 1e3, "muP": pmu / 1e3, "muth": th, "Q2": Q2 / 1e6, "W": W / 1e3, "nhad": nhad, "keh": keh, "ch": ak.to_numpy(t["mc_intType"]), "Z": ak.to_numpy(t["mc_targetZ"])}
    key = np.stack([ak.to_numpy(t["mc_run"]), ak.to_numpy(t["mc_subrun"]), ak.to_numpy(t["mc_nthEvtInFile"])], 1)
    s = uproot.open(sur_path)["MasterAnaDev"]
    cols = ["mc_run", "mc_subrun", "mc_nthEvtInFile", "MasterAnaDev_muon_P", "MasterAnaDev_muon_theta", "MasterAnaDev_minos_trk_is_ok", "MasterAnaDev_recoil_E", "recoil_energy_nonmuon_nonvtx100mm", "nonvtx_iso_blobs_energy", "n_prongs", "surrogate_p_reco_exists",
            "surrogate_epi_logit_reco_exists", "surrogate_epi_flow_flag", "surrogate_epi_multiplicity_logits", "surrogate_epi_recoil_E", "surrogate_epi_muon_P", "surrogate_epi_p_reco_exists"]
    S = {k: np.asarray(v, float) for k, v in s.arrays(cols, library="np").items()}
    skey = np.stack([S["mc_run"], S["mc_subrun"], S["mc_nthEvtInFile"]], 1).astype(np.int64)
    # map surrogate rows back to truth rows
    tk = {tuple(k): i for i, k in enumerate(key)}; row = np.array([tk[tuple(k)] for k in skey]); assert len(np.unique(row)) == len(row)
    sur = {"row": row, "P": S["MasterAnaDev_muon_P"] / 1e3, "th": S["MasterAnaDev_muon_theta"], "minos": S["MasterAnaDev_minos_trk_is_ok"] > 0, "recoil": S["MasterAnaDev_recoil_E"], "nv100": S["recoil_energy_nonmuon_nonvtx100mm"], "blobs": S["nonvtx_iso_blobs_energy"], "npr": S["n_prongs"],
           "epi_exist": S["surrogate_epi_logit_reco_exists"], "epi_flow": S["surrogate_epi_flow_flag"], "epi_card": S["surrogate_epi_multiplicity_logits"], "epi_recoil_rel": S["surrogate_epi_recoil_E"] / np.maximum(S["MasterAnaDev_recoil_E"], 1), "epi_muP_rel": S["surrogate_epi_muon_P"] / np.maximum(S["MasterAnaDev_muon_P"], 1), "epi_preco": S["surrogate_epi_p_reco_exists"]}
    ref = None
    if has_ref:
        ex = ak.to_numpy(t["ref_reco_exists"]); ref = {"row": np.where(ex)[0], "P": ak.to_numpy(t["ref_muon_P"])[ex] / 1e3, "th": ak.to_numpy(t["ref_muon_theta"])[ex], "recoil": ak.to_numpy(t["ref_recoil_E"])[ex], "nv100": ak.to_numpy(t["ref_recoil_nonvtx100"])[ex], "blobs": ak.to_numpy(t["ref_nonvtx_iso_blobs_E"])[ex], "npr": ak.to_numpy(t["ref_n_prongs"])[ex].astype(float) + 1}  # + muon track, as n_prongs in the ntuple
    return T, sur, ref

print("loading GENIE", flush=True); TG, SG, RG = load(a.genie_truth, a.genie_sur, True)
print("loading NuWro", flush=True); TN, SN, _ = load(a.nuwro_truth, a.nuwro_sur, False)
R = {"n_genie": int(TG["n"]), "n_nuwro": int(TN["n"])}

# ---------- truth-level comparison
fig, axs = plots.plt.subplots(2, 4, figsize=(14, 6)); axs = axs.ravel()
specs = [("Enu", np.linspace(0, 25, 51), r"true $E_\nu$ [GeV]", True), ("muP", np.linspace(0, 20, 51), "true muon momentum [GeV]", True), ("muth", np.linspace(0, 0.6, 49), "true muon angle to the beam [rad]", False),
         ("Q2", np.linspace(0, 4, 41), r"true $Q^2$ [GeV$^2$]", True), ("W", np.linspace(0.8, 4.5, 38), "lepton-defined $W$ [GeV]", False), ("nhad", np.arange(-0.5, 15.5, 1), "hadron tokens after cuts", True), ("keh", np.logspace(1, 4.3, 34), "true hadronic KE [MeV]", True)]
for ax, (k, e, xl, logy) in zip(axs, specs):
    for g, T in (("genie", TG), ("nuwro", TN)):
        v = T[k]; v = v[np.isfinite(v)]; ax.hist(v, e, histtype="step", color=C[g], lw=1.3, density=True, label="GENIE" if g == "genie" else "NuWro")
    ax.set_xlabel(xl); ax.set_ylabel("density")
    if logy: ax.set_yscale("log")
    if k == "keh": ax.set_xscale("log")
axs[0].legend(frameon=False, fontsize=8)
ax = axs[7]; codes = [1, 2, 3, 4, 8]; w = 0.4
for i, (g, T) in enumerate((("genie", TG), ("nuwro", TN))):
    f = [np.mean(T["ch"] == c) for c in codes]; ax.bar(np.arange(len(codes)) + (i - 0.5) * w, f, w, color=C[g], label="GENIE" if g == "genie" else "NuWro"); R[f"channel_frac_{g}"] = {CH[c]: float(x) for c, x in zip(codes, f)}
ax.set_xticks(np.arange(len(codes))); ax.set_xticklabels([CH[c] for c in codes]); ax.set_ylabel("fraction of events"); ax.legend(frameon=False, fontsize=8); ax.set_title("interaction channel", fontsize=9, loc="left")
plots.save(fig, out / "figures" / "truth_level.png")
R["truth_means"] = {k: {"genie": float(np.nanmean(TG[k])), "nuwro": float(np.nanmean(TN[k]))} for k in ("Enu", "muP", "muth", "Q2", "W", "nhad", "keh")}

# ---------- reconstructed-level: three-way, normalised per truth event
def per_truth(v, n, e): h = np.histogram(v[np.isfinite(v)], e)[0]; return h / n, np.sqrt(h) / n
specs = [("P", np.linspace(0, 20, 41), "reco muon momentum [GeV]", True), ("th", np.linspace(0, 0.6, 31), "reco muon angle [rad]", False), ("recoil", np.logspace(1, 4.3, 34), r"reco recoil $E$ [MeV]", True),
         ("nv100", np.r_[0.3, np.logspace(0, 4, 29)], "non-vertex $E$, 100 mm [MeV]", True), ("blobs", np.r_[0.3, np.logspace(0.5, 4, 29)], "isolated-blob $E$ [MeV]", True), ("npr", np.arange(-0.5, 9.5, 1), "reco prong count", True)]
fig, axs = plots.plt.subplots(2, 6, figsize=(20, 6.2), gridspec_kw={"height_ratios": (2, 1), "hspace": 0.08}); SM = {}
for j, (k, e, xl, logx) in enumerate(specs):
    ax, axr = axs[0, j], axs[1, j]; H = {}
    for g, S, n in (("ref", RG, TG["n"]), ("genie", SG, TG["n"]), ("nuwro", SN, TN["n"])):
        v = S[k].copy()
        if k in ("nv100", "blobs"): v = np.where(v <= 0, 0.5, v)  # exact zeros land in the first bin [0.3, 1)
        h, err = per_truth(v, n, e); H[g] = (h, err); c = 0.5 * (e[:-1] + e[1:]) if not logx or k == "npr" else np.sqrt(e[:-1] * e[1:])
        ax.stairs(h, e, color=C[g], lw=1.3, label=LAB[g]); ax.errorbar(c, h, err, fmt="none", color=C[g], lw=0.8)
    for g, ls in (("genie", "-"), ("nuwro", "-")):
        den = H["ref"][0] if g == "genie" else H["genie"][0]; num = H[g][0]; ok = den > 0; c = 0.5 * (e[:-1] + e[1:]) if not logx or k == "npr" else np.sqrt(e[:-1] * e[1:])
        r = np.where(ok, num / np.where(ok, den, 1), np.nan); re = np.where(ok, r * np.sqrt((H[g][1] / np.maximum(num, 1e-12)) ** 2 + ((H["ref"][1] if g == "genie" else H["genie"][1]) / np.where(ok, den, 1)) ** 2), np.nan)
        axr.errorbar(c, r, re, fmt="o", ms=2.5, color=C[g], label="surrogate(GENIE) / open dataset" if g == "genie" else "surrogate(NuWro) / surrogate(GENIE)")
    axr.axhline(1, color="k", lw=0.6); axr.set_ylim(0.3, 1.7); axr.set_xlabel(xl); ax.set_yscale("log")
    if logx and k != "npr":
        ax.set_xscale("log"); axr.set_xscale("log")
        if k in ("nv100", "blobs"):
            for a_ in (ax, axr): a_.set_xticks([0.5, 10, 100, 1000, 10000]); a_.set_xticklabels(["0", "10", "$10^2$", "$10^3$", "$10^4$"])
            ax.annotate("exact zeros", (0.5, H["ref"][0][0]), textcoords="offset points", xytext=(6, 6), fontsize=7)
    SM[k] = {g: {"frac_total": float(H[g][0].sum())} for g in H}
axs[0, 0].set_ylabel("events per truth event"); axs[1, 0].set_ylabel("ratio"); axs[0, 0].legend(frameon=False, fontsize=7); axs[1, 0].legend(frameon=False, fontsize=6.5)
plots.save(fig, out / "figures" / "reco_level.png")

# ---------- conditional response: should be generator-independent if the surrogate is a function of the final state
def binned_mean(x, y, e):
    m, s = [], []
    for lo, hi in zip(e[:-1], e[1:]):
        sel = (x >= lo) & (x < hi) & np.isfinite(y); m.append(y[sel].mean() if sel.sum() >= 50 else np.nan); s.append(y[sel].std() / np.sqrt(max(sel.sum(), 1)) if sel.sum() >= 50 else np.nan)
    return np.array(m), np.array(s)
fig, axs = plots.plt.subplots(1, 4, figsize=(16, 3.6))
e_nu = np.linspace(1, 20, 20); e_nh = np.arange(-0.5, 12.5, 1); e_ke = np.logspace(1.5, 4.3, 17)
for g, T, S, R_ in (("ref", TG, None, RG), ("genie", TG, SG, None), ("nuwro", TN, SN, None)):
    src = R_ if R_ is not None else S; reco = np.zeros(T["n"], bool); reco[src["row"]] = True
    m, s = binned_mean(T["Enu"], reco.astype(float), e_nu); axs[0].errorbar(0.5 * (e_nu[:-1] + e_nu[1:]), m, s, fmt="o-", ms=3, color=C[g], label=LAB[g])
    m, s = binned_mean(T["nhad"].astype(float), reco.astype(float), e_nh); axs[1].errorbar(np.arange(0, 12), m, s, fmt="o-", ms=3, color=C[g])
    keh = T["keh"][src["row"]]; m, s = binned_mean(keh, np.log10(np.clip(src["recoil"], 1, None)), e_ke); axs[2].errorbar(np.sqrt(e_ke[:-1] * e_ke[1:]), m, s, fmt="o-", ms=3, color=C[g])
    muP = T["muP"][src["row"]]; ratio = src["P"] / np.maximum(muP, 1e-3); e_p = np.linspace(1, 15, 15); med = []
    for lo, hi in zip(e_p[:-1], e_p[1:]):
        sel = (muP >= lo) & (muP < hi) & np.isfinite(ratio); med.append(np.median(ratio[sel]) if sel.sum() >= 50 else np.nan)
    axs[3].plot(0.5 * (e_p[:-1] + e_p[1:]), med, "o-", ms=3, color=C[g])
axs[0].set_xlabel(r"true $E_\nu$ [GeV]"); axs[0].set_ylabel("P(reco muon candidate)"); axs[1].set_xlabel("hadron tokens after cuts"); axs[1].set_ylabel("P(reco muon candidate)")
axs[2].set_xlabel("true hadronic KE [MeV]"); axs[2].set_ylabel(r"mean $\log_{10}$ reco recoil $E$"); axs[2].set_xscale("log"); axs[3].set_xlabel("true muon momentum [GeV]"); axs[3].set_ylabel(r"median $P_\mathrm{reco}/P_\mathrm{true}$"); axs[3].set_ylim(0.9, 1.1); axs[0].legend(frameon=False, fontsize=7)
plots.save(fig, out / "figures" / "conditional_response.png")

# ---------- summary table and epistemic flags
def frac0(v): return float(np.mean(v <= 0))
rows = []
for g, T, S in (("ref", TG, RG), ("genie", TG, SG), ("nuwro", TN, SN)):
    d = {"reco_frac": len(S["row"]) / T["n"], "mean_recoil": float(np.nanmean(S["recoil"])), "zero_nv100": frac0(S["nv100"]), "zero_blobs": frac0(S["blobs"]), "mean_npr": float(np.nanmean(S["npr"])), "minos_frac": float(np.mean(S["minos"])) if "minos" in S else np.nan}
    R[f"reco_{g}"] = d; rows.append((g, d))
tex = "\\begin{tabular}{lcccccc}\n\\toprule\n & events & reco fraction & MINOS-matched & mean recoil $E$ [MeV] & zero non-vtx / blobs & mean prongs \\\\\n\\midrule\n"
for g, d in rows:
    n = TG["n"] if g != "nuwro" else TN["n"]; tex += f"{LAB[g]} & {n:,} & {d['reco_frac']:.3f} & {('%.3f' % d['minos_frac']) if np.isfinite(d['minos_frac']) else '--'} & {d['mean_recoil']:.0f} & {d['zero_nv100']:.3f} / {d['zero_blobs']:.3f} & {d['mean_npr']:.3f} \\\\\n"
tex += "\\bottomrule\n" + "\\end{tabular}\n"; (out / "tables" / "summary.tex").write_text(tex.replace(",", "{,}") if False else tex)
# truth composition table
tex = "\\begin{tabular}{lcccccccc}\n\\toprule\n & QE & RES & DIS & COH & 2p2h & mean $E_\\nu$ [GeV] & mean hadron tokens & mean hadronic KE [MeV] \\\\\n\\midrule\n"
for g, T in (("genie", TG), ("nuwro", TN)):
    f = R[f"channel_frac_{g}"]; tex += f"{'GENIE (open dataset)' if g == 'genie' else 'NuWro'} & {100*f['QE']:.1f}\\% & {100*f['RES']:.1f}\\% & {100*f['DIS']:.1f}\\% & {100*f['COH']:.1f}\\% & {100*f['2p2h']:.1f}\\% & {np.nanmean(T['Enu']):.2f} & {np.nanmean(T['nhad']):.2f} & {np.nanmean(T['keh']):.0f} \\\\\n"
tex += "\\bottomrule\n" + "\\end{tabular}\n"; (out / "tables" / "truth_composition.tex").write_text(tex)
# does the model see NuWro as novel? epistemic flags GENIE vs NuWro
E = {}
for k in ("epi_exist", "epi_flow", "epi_card", "epi_recoil_rel", "epi_muP_rel", "epi_preco"):
    g, n_ = SG[k], SN[k]; g, n_ = g[np.isfinite(g)], n_[np.isfinite(n_)]; y = np.r_[np.zeros(len(g)), np.ones(len(n_))]; E[k] = {"median_genie": float(np.median(g)), "median_nuwro": float(np.median(n_)), "ratio": float(np.median(n_) / max(np.median(g), 1e-12)), "auc": float(roc_auc_score(y, np.r_[g, n_]))}
R["epistemic_flags"] = E
fig, axs = plots.plt.subplots(1, 3, figsize=(12, 3.3))
for ax, k, xl in zip(axs, ("epi_exist", "epi_card", "epi_flow"), ("reco-flag logit variance", "mean multiplicity-logit variance", "event-flow epistemic flag")):
    allv = np.r_[SG[k], SN[k]]; allv = allv[np.isfinite(allv) & (allv > 0)]; b = np.logspace(np.log10(np.percentile(allv, 0.5)), np.log10(np.percentile(allv, 99.5)), 50)
    for g, S in (("genie", SG), ("nuwro", SN)): v = S[k]; ax.hist(v[np.isfinite(v) & (v > 0)], b, histtype="step", density=True, color=C[g], lw=1.3, label=LAB[g])
    ax.set_xscale("log"); ax.set_xlabel(xl); ax.set_title(f"NuWro/GENIE median ratio {E[k]['ratio']:.2f}, AUC {E[k]['auc']:.2f}", fontsize=9, loc="left")
axs[0].legend(frameon=False, fontsize=7); axs[0].set_ylabel("density"); plots.save(fig, out / "figures" / "epistemic_flags.png")
json.dump(R, open(out / "summary.json", "w"), indent=1, default=float)
# macros (no digits in names)
M = {"nNWnGenie": f"{TG['n']:,}".replace(",", "{,}"), "nNWnNuwro": f"{TN['n']:,}".replace(",", "{,}")}
for g, nm in (("ref", "Ref"), ("genie", "Gen"), ("nuwro", "Nuw")):
    d = R[f"reco_{g}"]; M[f"nNW{nm}RecoFrac"] = f"{d['reco_frac']:.3f}"; M[f"nNW{nm}ZeroNv"] = f"{100*d['zero_nv100']:.1f}\\%"; M[f"nNW{nm}ZeroBlobs"] = f"{100*d['zero_blobs']:.1f}\\%"; M[f"nNW{nm}MeanRecoil"] = f"{d['mean_recoil']:.0f}"; M[f"nNW{nm}MeanNpr"] = f"{d['mean_npr']:.2f}"
for g, nm in (("genie", "Gen"), ("nuwro", "Nuw")):
    for c, cn in (("QE", "QE"), ("RES", "RES"), ("DIS", "DIS"), ("COH", "COH"), ("2p2h", "MEC")): M[f"nNW{nm}Frac{cn}"] = f"{100*R[f'channel_frac_{g}'][c]:.1f}\\%"
    M[f"nNW{nm}MeanEnu"] = f"{R['truth_means']['Enu'][g]:.2f}"; M[f"nNW{nm}MeanNhad"] = f"{R['truth_means']['nhad'][g]:.2f}"; M[f"nNW{nm}MeanKeh"] = f"{R['truth_means']['keh'][g]:.0f}"
for k, kn in (("epi_exist", "Exist"), ("epi_card", "Card"), ("epi_flow", "Flow"), ("epi_recoil_rel", "RecoilRel"), ("epi_preco", "Preco")): M[f"nNWEpi{kn}Ratio"] = f"{E[k]['ratio']:.2f}"; M[f"nNWEpi{kn}AUC"] = f"{E[k]['auc']:.2f}"
(out / "macros.tex").write_text("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in M.items()))
print(json.dumps({k: v for k, v in R.items() if k.startswith("reco_") or k.startswith("channel") or k == "epistemic_flags"}, indent=1, default=float))
