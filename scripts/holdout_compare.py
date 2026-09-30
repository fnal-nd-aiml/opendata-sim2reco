#!/usr/bin/env python
"""Three-way comparison for the 2p2h holdout: open dataset vs model A (trained on all) vs model B (2p2h blind),
evaluated on the 2p2h test events, plus B on the non-2p2h control and an input-coverage figure.

Usage: scripts/holdout_compare.py OUT_DIR --a reports/m3_1A_on2p2h --b reports/m3_1A_no2p2h_on2p2h
                                  [--b-control reports/m3_1A_no2p2h_control] [--a-full reports/m3_1A] [--slim-dir data/slim_1A]
Reads metrics.json / metrics_tier2.json written by run_m3.py; no re-sampling.
"""
import argparse, glob, json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from sim2reco.eval import plots

C_REAL, C_A, C_B = plots.PALETTE["real"], plots.PALETTE["third"], plots.PALETTE["model"]


def load(d):
    d = pathlib.Path(d)
    return json.load(open(d / "metrics.json")), json.load(open(d / "metrics_tier2.json"))


def profile(ax, rows_a, rows_b, key_real, key_fake, xlabel, ylabel, log=True):
    x = [np.sqrt(r["lo"] * r["hi"]) if log else 0.5 * (r["lo"] + r["hi"]) for r in rows_a]
    for rows, key, c, ls, lab in ((rows_a, key_real, C_REAL, "o-", "open dataset (2p2h)"), (rows_a, key_fake, C_A, "s--", "surrogate A (trained with 2p2h)"), (rows_b, key_fake, C_B, "^:", "surrogate B (2p2h blind)")):
        med = [r[key][1] if r[key] else np.nan for r in rows]; lo = [r[key][0] if r[key] else np.nan for r in rows]; hi = [r[key][2] if r[key] else np.nan for r in rows]
        ax.plot(x, med, ls, color=c, ms=4, label=lab); ax.fill_between(x, lo, hi, color=c, alpha=0.12)
    if log: ax.set_xscale("log")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.legend(frameon=False, fontsize=7)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir"); ap.add_argument("--a", required=True); ap.add_argument("--b", required=True)
    ap.add_argument("--b-control", default=None); ap.add_argument("--a-full", default=None); ap.add_argument("--slim-dir", default="data/slim_1A")
    a = ap.parse_args()
    out = pathlib.Path(a.out_dir); (out / "figures").mkdir(parents=True, exist_ok=True); (out / "tables").mkdir(exist_ok=True)
    MA, TA = load(a.a); MB, TB = load(a.b)
    S = {}
    # ---- summary table: A vs B on 2p2h, B on control, A on full test ----
    def row(M, T):
        return {"reco_exists_logloss": M["tier0"]["reco_exists"]["logloss"], "minos_logloss": M["tier0"]["minos_ok"]["logloss"], "mult_logloss": M["multiplicity"]["logloss"],
                "auc_marginal": M["classifier_auc_marginal"], "auc_conditional": M["classifier_auc_conditional"],
                "auc_cond_muon": M["classifier_auc_conditional_by_block"]["muon"], "auc_cond_vertex": M["classifier_auc_conditional_by_block"]["vertex"], "auc_cond_calo": M["classifier_auc_conditional_by_block"]["calorimetry"],
                "prong_auc": T["closure_prongs"]["event_summary_only"], "prong_auc_cond": T["closure_prongs"]["truth_plus_event_summary"],
                "w1_logP": M["tier1_model_space"]["log_P_ratio"]["w1"], "w1_recoil": M["tier1_model_space"]["log_recoil_E"]["w1"], "w1_prong_pP": T["prong_marginals"]["log_p_P"]["w1"],
                "pid_auc_real": T["pid"]["score_auc_p_vs_pi"][0] if "pid" in T else np.nan, "pid_auc_fake": T["pid"]["score_auc_p_vs_pi"][1] if "pid" in T else np.nan, "n_test": M["n_test"]}
    S["A_on_2p2h"] = row(MA, TA); S["B_on_2p2h"] = row(MB, TB)
    if a.b_control: MC, TC = load(a.b_control); S["B_on_control"] = row(MC, TC)
    if a.a_full: MF, TF = load(a.a_full); S["A_on_full_test"] = row(MF, TF)
    keys = [("reco\\_exists log loss", "reco_exists_logloss", ".4f"), ("MINOS log loss", "minos_logloss", ".4f"), ("multiplicity log loss", "mult_logloss", ".4f"),
            ("closure AUC, reco only", "auc_marginal", ".3f"), ("closure AUC, truth + reco", "auc_conditional", ".3f"), ("cond.\\ AUC muon / vertex / calo", None, None),
            ("prong closure AUC, summary / + truth", None, None), ("$W_1$ $\\log P$ ratio / $\\log E_\\mathrm{recoil}$ / $\\log P_p$", None, None), ("PID AUC (open dataset / surrogate)", None, None), ("test events", "n_test", ",d")]
    cols = list(S); hdr = " & ".join(c.replace("_", " ") for c in cols)
    lines = []
    for lab, k, f in keys:
        if k: vals = [format(S[c][k], f) for c in cols]
        elif lab.startswith("cond"): vals = [f"{S[c]['auc_cond_muon']:.3f} / {S[c]['auc_cond_vertex']:.3f} / {S[c]['auc_cond_calo']:.3f}" for c in cols]
        elif lab.startswith("prong"): vals = [f"{S[c]['prong_auc']:.3f} / {S[c]['prong_auc_cond']:.3f}" for c in cols]
        elif lab.startswith("$W_1$"): vals = [f"{S[c]['w1_logP']:.3f} / {S[c]['w1_recoil']:.3f} / {S[c]['w1_prong_pP']:.3f}" for c in cols]
        else: vals = [f"{S[c]['pid_auc_real']:.3f} / {S[c]['pid_auc_fake']:.3f}" for c in cols]
        lines.append(lab + " & " + " & ".join(vals) + " \\\\")
    (out / "tables" / "holdout_summary.tex").write_text("\\begin{tabular}{l" + "c" * len(cols) + "}\n\\toprule\n & " + hdr + " \\\\\n\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")
    json.dump(S, open(out / "summary.json", "w"), indent=1, default=float)
    # ---- three-way conditionals on 2p2h ----
    fig, axs = plots.plt.subplots(1, 3, figsize=(12, 3.3))
    profile(axs[0], MA["muon_resp_by_P"], MB["muon_resp_by_P"], "real_q16_50_84", "fake_q16_50_84", "true muon momentum [GeV]", "log P ratio [model space]")
    profile(axs[1], MA["recoil_by_sumKE"], MB["recoil_by_sumKE"], "real_q16_50_84", "fake_q16_50_84", "true hadronic KE [MeV]", "log recoil_E [model space]")
    profile(axs[2], TA["lead_proton_P_by_true_KE"], TB["lead_proton_P_by_true_KE"], "real_q16_50_84", "fake_q16_50_84", "true leading proton KE [MeV]", "leading proton-hypothesis P [MeV]")
    axs[2].set_yscale("log"); plots.save(fig, out / "figures" / "holdout_conditionals.png")
    # ---- efficiency and prong counts vs truth ----
    fig, axs = plots.plt.subplots(1, 3, figsize=(12, 3.3))
    for rows, c, ls, lab, key in ((MA["eff_calibration_nhad"], C_REAL, "o-", "open dataset (2p2h)", "observed"), (MA["eff_calibration_nhad"], C_A, "s--", "surrogate A", "predicted"), (MB["eff_calibration_nhad"], C_B, "^:", "surrogate B (blind)", "predicted")):
        axs[0].plot([(r["lo"] + r["hi"]) / 2 for r in rows], [r[key] for r in rows], ls, color=c, ms=4, label=lab)
    axs[0].set_xlabel("true hadrons after cuts"); axs[0].set_ylabel("P(reco muon candidate)"); axs[0].set_ylim(0, 1); axs[0].legend(frameon=False, fontsize=7)
    for rows, c, ls, lab, j in ((TA["validation"]["prongs_vs_true_charged"], C_REAL, "o-", "open dataset (2p2h)", 3), (TA["validation"]["prongs_vs_true_charged"], C_A, "s--", "surrogate A", 4), (TB["validation"]["prongs_vs_true_charged"], C_B, "^:", "surrogate B (blind)", 4)):
        axs[1].plot([r[0] for r in rows], [r[j] for r in rows], ls, color=c, ms=4, label=lab)
    axs[1].set_xlabel("true charged hadrons after FSI"); axs[1].set_ylabel("mean reco prongs per event"); axs[1].legend(frameon=False, fontsize=7)
    for rows, c, ls, lab, key in ((TA["lead_proton_P_by_true_KE"], C_REAL, "o-", "open dataset (2p2h)", 0), (TA["lead_proton_P_by_true_KE"], C_A, "s--", "surrogate A", 1), (TB["lead_proton_P_by_true_KE"], C_B, "^:", "surrogate B (blind)", 1)):
        axs[2].plot([np.sqrt(r["lo"] * r["hi"]) for r in rows], [r["frac_with_pfit"][key] for r in rows], ls, color=c, ms=4, label=lab)
    axs[2].set_xscale("log"); axs[2].set_xlabel("true leading proton KE [MeV]"); axs[2].set_ylabel("fraction with a proton fit"); axs[2].legend(frameon=False, fontsize=7)
    plots.save(fig, out / "figures" / "holdout_efficiency_prongs.png")
    # ---- multiplicity confusion: open dataset / A / B ----
    fig, axs = plots.plt.subplots(1, 3, figsize=(12, 3.6)); labels = [str(k) for k in range(6)] + ["6+"]
    mats = [(np.array(TA["confusion_true_charged_vs_reco"]["masteranadev"], float), "open dataset (2p2h)"), (np.array(TA["confusion_true_charged_vs_reco"]["surrogate"], float), "surrogate A"), (np.array(TB["confusion_true_charged_vs_reco"]["surrogate"], float), "surrogate B (blind)")]
    for ax, (C, title) in zip(axs, mats):
        C = C / np.clip(C.sum(1, keepdims=True), 1, None); im = ax.imshow(C, vmin=0, vmax=1, cmap="Blues", origin="lower")
        for i in range(7):
            for j in range(7):
                if C[i, j] >= 0.005: ax.text(j, i, f"{C[i,j]:.2f}", ha="center", va="center", fontsize=6.5, color="white" if C[i, j] > 0.6 else "black")
        ax.set_xticks(range(7)); ax.set_yticks(range(7)); ax.set_xticklabels(labels, fontsize=8); ax.set_yticklabels(labels, fontsize=8); ax.grid(False); ax.set_xlabel("reco prongs"); ax.set_ylabel("true charged hadrons"); ax.set_title(title, fontsize=9, loc="left")
    fig.colorbar(im, ax=axs.tolist(), shrink=0.8); fig.savefig(out / "figures" / "holdout_multiplicity.png", dpi=150, bbox_inches="tight"); plots.plt.close(fig)
    # ---- input coverage: 2p2h final states vs the rest of the training population ----
    try:
        from sim2reco.data.compact import load_compact
        stems = sorted(p[:-len(".truth.parquet")] for p in glob.glob(f"{a.slim_dir}/*.truth.parquet"))
        d = load_compact(stems); it = d["intType"]; off = d["offsets"]; cls = d["cls"]; seg = np.repeat(np.arange(len(off) - 1), np.diff(off))
        def count(c): o = np.zeros(len(off) - 1, int); np.add.at(o, seg, cls == c); return o
        n_p = count(4); n_pi = count(5) + count(6) + count(7)
        mom = d["mom"]; KE = np.sqrt((mom ** 2).sum(1) + 938.272 ** 2) - 938.272; isp = cls == 4
        lead = np.full(len(off) - 1, -1.0); np.maximum.at(lead, seg[isp], KE[isp])
        cats = [("0p 0$\\pi$", (n_p == 0) & (n_pi == 0)), ("1p 0$\\pi$", (n_p == 1) & (n_pi == 0)), ("2p 0$\\pi$", (n_p == 2) & (n_pi == 0)), ("3+p 0$\\pi$", (n_p >= 3) & (n_pi == 0)), ("any $\\pi$", n_pi >= 1)]
        fig, axs = plots.plt.subplots(1, 2, figsize=(8.5, 3.3))
        k = np.arange(len(cats)); w = 0.4; m2 = it == 8; mo = it != 8
        axs[0].bar(k - w / 2, [(c & m2).sum() / m2.sum() for _, c in cats], w, color=C_REAL, label="2p2h (held out)"); axs[0].bar(k + w / 2, [(c & mo).sum() / mo.sum() for _, c in cats], w, color=plots.PALETTE["gray"], label="all other modes (training)")
        axs[0].set_xticks(k); axs[0].set_xticklabels([n for n, _ in cats]); axs[0].set_ylabel("fraction of events"); axs[0].set_title("final state after input cuts", fontsize=9, loc="left"); axs[0].legend(frameon=False, fontsize=7)
        s2 = (n_p == 2) & (n_pi == 0); b = np.linspace(0, 1200, 50)
        for mask, c, lab in ((s2 & m2, C_REAL, "2p2h"), (s2 & (it == 1), C_A, "QE"), (s2 & (it == 2), C_B, "RES"), (s2 & (it == 3), plots.PALETTE["gray"], "DIS")):
            axs[1].hist(lead[mask], bins=b, histtype="step", density=True, color=c, label=f"{lab} (n={mask.sum():,})")
        axs[1].set_xlabel("leading proton KE [MeV], 2p 0$\\pi$ final states"); axs[1].set_ylabel("density"); axs[1].legend(frameon=False, fontsize=7)
        plots.save(fig, out / "figures" / "holdout_input_coverage.png")
        S["coverage"] = {"n_2p2h": int(m2.sum()), "frac_2p0pi_2p2h": float((s2 & m2).sum() / m2.sum()), "frac_2p0pi_other": float((s2 & mo).sum() / mo.sum()),
                         "n_2p0pi_other": int((s2 & mo).sum()), "lead_KE_2p0pi_median": {n: float(np.median(lead[s2 & (it == c)])) for c, n in ((8, "2p2h"), (1, "QE"), (2, "RES"), (3, "DIS"))}}
        json.dump(S, open(out / "summary.json", "w"), indent=1, default=float)
    except Exception as e:
        print("coverage figure skipped:", e)
    print(json.dumps({k: {kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in vv.items()} for k, vv in S.items() if k != "coverage"}, indent=1))


if __name__ == "__main__":
    main()
