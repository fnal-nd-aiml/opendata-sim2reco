#!/usr/bin/env python
"""Write LaTeX macros with every number quoted in the technote/paper text, from the evaluation JSON files.

Usage: scripts/technote_numbers.py MODEL_DIR OUT.tex [--m1 reports/m1_1A/metrics.json] [--holdout reports/holdout_2p2h_1A]
Then \\input{OUT.tex} and use the macros (\\nAUCmarg, ...) instead of typing numbers.
"""
import argparse, json, pathlib, numpy as np

def pct(x, d=0): return f"{100*x:.{d}f}\\%"

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("model_dir"); ap.add_argument("out"); ap.add_argument("--m1", default="reports/m1_1A/metrics.json"); ap.add_argument("--holdout", default=None)
    a = ap.parse_args(); D = pathlib.Path(a.model_dir)
    m = json.load(open(D / "metrics.json")); t = json.load(open(D / "metrics_tier2.json")); m1 = json.load(open(a.m1)) if pathlib.Path(a.m1).exists() else None
    M = {}
    t0 = m["tier0"]; M["nRecoLL"] = f"{t0['reco_exists']['logloss']:.3f}"; M["nRecoLLtree"] = f"{m1['tier0']['reco_exists']['logloss']:.3f}" if m1 else "--"
    M["nMinosLL"] = f"{t0['minos_ok']['logloss']:.3f}"; M["nChargeLL"] = f"{t0['charge_neg']['logloss']:.3f}"; M["nMultLL"] = f"{m['multiplicity']['logloss']:.3f}"
    M["nTest"] = f"{m['n_test']:,}".replace(",", "{,}")
    w = m["tier1_model_space"]; M["nWoneMax"] = f"{max(v['w1'] for k, v in w.items() if 'minos' not in k):.2f}"
    M["nWoneMuMatched"] = f"{w['log_P_ratio_minos_ok']['w1']:.3f}"; M["nWoneMuUnmatched"] = f"{w['log_P_ratio_no_minos']['w1']:.2f}"; M["nWoneRecoil"] = f"{w['log_recoil_E']['w1']:.3f}"
    M["nCorrDiff"] = f"{m['corr_max_abs_diff']:.3f}"
    M["nAUCmarg"] = f"{m['classifier_auc_marginal']:.3f}"; M["nAUCcond"] = f"{m['classifier_auc_conditional']:.3f}"
    bb = m["classifier_auc_by_block"]; cb = m["classifier_auc_conditional_by_block"]
    M["nAUCblockMax"] = f"{max(bb.values()):.3f}"; M["nAUCcondBlockMax"] = f"{max(cb['muon'], cb['vertex'], cb['calorimetry']):.3f}"; M["nAUCflags"] = f"{cb['flags_and_N_only']:.3f}"
    c = t["closure_prongs"]; M["nAUCprong"] = f"{c['event_summary_only']:.3f}"; M["nAUCprongCond"] = f"{c['truth_plus_event_summary']:.3f}"; M["nAUCvtxz"] = f"{c['truth_plus_vertex_z']:.3f}"
    # multiplicity confusion cells
    C = np.array(t["confusion_true_charged_vs_reco"]["masteranadev"], float); R = C / np.clip(C.sum(1, keepdims=True), 1, None)
    S = np.array(t["confusion_true_charged_vs_reco"]["surrogate"], float); RS = S / np.clip(S.sum(1, keepdims=True), 1, None)
    M["nConfOneZero"] = pct(R[1, 0]); M["nConfOneOne"] = pct(R[1, 1]); M["nConfFourOne"] = pct(R[4, 1]); M["nConfMaxDiff"] = f"{np.abs(R - RS).max():.2f}"
    ev = np.array(t["confusion_event_by_event"]); M["nEvZero"] = f"{int(ev[0].sum()):,}".replace(",", "{,}"); M["nEvSixPlus"] = f"{int(ev[6].sum()):,}"
    J = ev / ev.sum(); M["nJointAsym"] = f"{np.abs(J - J.T).max():.4f}"; M["nJointMax"] = f"{J.max():.3f}"
    # prongs
    pc = t["prong_counts"]; M["nKinFrac"] = pct(pc["has_kin"][0]); M["nPfitFrac"] = pct(pc["has_p"][0]); M["nProngWoneMax"] = f"{max(v['w1'] for v in t['prong_marginals'].values()):.3f}"
    rows = t["lead_proton_P_by_true_KE"]; M["nPfitLow"] = pct(rows[0]["frac_with_pfit"][0]); M["nPfitPeak"] = pct(max(r["frac_with_pfit"][0] for r in rows)); M["nPfitHiReal"] = pct(rows[-1]["frac_with_pfit"][0]); M["nPfitHiFake"] = pct(rows[-1]["frac_with_pfit"][1])
    pv = t["validation"]["prongs_vs_true_charged"]; by = {r[0]: r for r in pv}
    M["nProngsOne"] = f"{by[1][3]:.2f}"; M["nProngsFour"] = f"{by[4][3]:.2f}"; M["nProngsSix"] = f"{by[6][3]:.2f}"; M["nProngsZeroReal"] = f"{by[0][3]:.2f}"; M["nProngsZeroFake"] = f"{by[0][4]:.2f}"; M["nEventsZeroCh"] = f"{by[0][5]:,}".replace(",", "{,}")
    p = t["pid"]; Mr = np.array(p["matrix_masteranadev"]); Mf = np.array(p["matrix_surrogate"])
    M["nPIDevents"] = f"{p['n_events'][0]:,}".replace(",", "{,}"); M["nPIDprotonHigh"] = pct(Mr[0, 3]); M["nPIDpionHigh"] = pct(Mr[1, 3]); M["nPIDmaxDiff"] = f"{p['max_abs_diff']:.3f}"
    M["nPIDaucReal"] = f"{p['score_auc_p_vs_pi'][0]:.3f}"; M["nPIDaucFake"] = f"{p['score_auc_p_vs_pi'][1]:.3f}"; M["nPIDaucGap"] = f"{p['score_auc_p_vs_pi'][0]-p['score_auc_p_vs_pi'][1]:.3f}"
    sm = t["validation"]["score_medians"]; M["nScoreMedPReal"] = f"{sm['lead_p_real']:.2f}"; M["nScoreMedPFake"] = f"{sm['lead_p_fake']:.2f}"; M["nScoreMedPiReal"] = f"{sm['lead_pi_real']:.3f}"; M["nScoreMedPiFake"] = f"{sm['lead_pi_fake']:.3f}"
    v = t["vertex_class"]; o = t["vertex_offset_to_nearest_plane"]
    M["nSnapReal"] = pct(o["real_frac_within_0.1mm"], 1); M["nSnapFake"] = pct(o["fake_frac_within_0.1mm"], 1); M["nSnapNearReal"] = pct(v["snapped_frac_real"], 1); M["nSnapNearFake"] = pct(v["snapped_frac_sampled"], 1)
    M["nSnapWone"] = f"{o['w1_mm']:.2f}"; M["nVtxLL"] = f"{v['logloss']:.3f}"; M["nVtxLLmarg"] = f"{v['logloss_marginal']:.3f}"
    if a.holdout:
        H = json.load(open(pathlib.Path(a.holdout) / "summary.json")); A, B, Cc, F = H["A_on_2p2h"], H["B_on_2p2h"], H["B_on_control"], H["A_on_full_test"]
        for tag, X in (("A", A), ("B", B), ("Ctl", Cc), ("Full", F)):
            M[f"nH{tag}AUCmarg"] = f"{X['auc_marginal']:.3f}"; M[f"nH{tag}AUCcond"] = f"{X['auc_conditional']:.3f}"; M[f"nH{tag}Calo"] = f"{X['auc_cond_calo']:.3f}"; M[f"nH{tag}Prong"] = f"{X['prong_auc']:.3f}"
            M[f"nH{tag}WoneRecoil"] = f"{X['w1_recoil']:.3f}"; M[f"nH{tag}WoneMu"] = f"{X['w1_logP']:.3f}"; M[f"nH{tag}RecoLL"] = f"{X['reco_exists_logloss']:.3f}"; M[f"nH{tag}MultLL"] = f"{X['mult_logloss']:.3f}"
        M["nHtest"] = f"{A['n_test']:,}".replace(",", "{,}"); cov = H["coverage"]
        M["nHtwoP"] = f"{cov['n_2p2h']:,}".replace(",", "{,}"); M["nHfracTwoPtwoph"] = pct(cov["frac_2p0pi_2p2h"]); M["nHfracTwoPother"] = pct(cov["frac_2p0pi_other"], 1); M["nHnTwoPother"] = f"{cov['n_2p0pi_other']:,}".replace(",", "{,}")
        for k, nm in (("2p2h", "TwoPTwoH"), ("QE", "QE"), ("RES", "RES"), ("DIS", "DIS")): M[f"nHleadKE{nm}"] = f"{cov['lead_KE_2p0pi_median'][k]:.0f}"
    lines = [f"\\newcommand{{\\{k}}}{{{vv}}}" for k, vv in M.items()]
    pathlib.Path(a.out).write_text("% generated by scripts/technote_numbers.py; do not edit\n" + "\n".join(lines) + "\n")
    print(f"{len(M)} macros -> {a.out}")

if __name__ == "__main__":
    main()
