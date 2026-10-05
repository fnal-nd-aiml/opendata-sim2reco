#!/usr/bin/env python
"""Write LaTeX macros with every number quoted in the technote/paper text, from the evaluation JSON files.

Usage: scripts/technote_numbers.py MODEL_DIR OUT.tex [--m1 reports/m1_1A/metrics.json] [--holdout reports/holdout_2p2h_1A]
Then \\input{OUT.tex} and use the macros (\\nAUCmarg, ...) instead of typing numbers.
"""
import argparse, json, pathlib, numpy as np

def pct(x, d=0): return f"{100*x:.{d}f}\\%"

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("model_dir"); ap.add_argument("out"); ap.add_argument("--m1", default="reports/m1_1A/metrics.json"); ap.add_argument("--holdout", default=None); ap.add_argument("--holdout-old", default=None, help="holdout summary of the previous input definition (50 MeV, no neutrons)"); ap.add_argument("--variants", default=None, help="variants_compare summary.json"); ap.add_argument("--bayes", default=None, help="directory with bayes_uncertainty_{A,B}.json and bayes_positive_control_B.json")
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
    if a.holdout_old:
        H = json.load(open(pathlib.Path(a.holdout_old) / "summary.json"))
        for tag, X in (("A", H["A_on_2p2h"]), ("B", H["B_on_2p2h"]), ("Full", H["A_on_full_test"])):
            M[f"nHold{tag}AUCmarg"] = f"{X['auc_marginal']:.3f}"; M[f"nHold{tag}AUCcond"] = f"{X['auc_conditional']:.3f}"; M[f"nHold{tag}Calo"] = f"{X['auc_cond_calo']:.3f}"; M[f"nHold{tag}WoneRecoil"] = f"{X['w1_recoil']:.3f}"
    if a.variants:
        Vv = json.load(open(a.variants))
        for key in Vv:
            name, split = key.split("/"); X = Vv[key]
            words = {"baseline": "KEfifty", "ke50": "KEfifty", "ke10": "KEten", "ke10n": "KEtenn"}  # macro names cannot contain digits
            tag = words.get(name, name) + {"full": "Full", "2p2h": "TwoP"}[split]
            M[f"nV{tag}AUCmarg"] = f"{X['auc_marg']:.3f}"; M[f"nV{tag}AUCcond"] = f"{X['auc_cond']:.3f}"; M[f"nV{tag}Calo"] = f"{X['auc_cond_calo']:.3f}"
            M[f"nV{tag}WoneRecoil"] = f"{X['w1_recoil']:.3f}"; M[f"nV{tag}WoneBlobs"] = f"{X['w1_blobs']:.3f}"; M[f"nV{tag}FakeReal"] = f"{X['zero_ch_prongs_real']:.2f}"; M[f"nV{tag}FakeSur"] = f"{X['zero_ch_prongs_fake']:.2f}"
    if a.holdout and (pathlib.Path(a.holdout) / "uncertainty.json").exists():
        U = json.load(open(pathlib.Path(a.holdout) / "uncertainty.json"))
        for key, tag in (("data_vs_A", "DA"), ("data_vs_B", "DB"), ("A_vs_B", "AB")):
            for c in ("marg", "cond"):
                u = U[f"auc_{key}_{c}"]; M[f"nU{tag}{c.capitalize()}"] = f"{u['point']:.3f}"; M[f"nU{tag}{c.capitalize()}Lo"] = f"{u['boot_lo']:.3f}"; M[f"nU{tag}{c.capitalize()}Hi"] = f"{u['boot_hi']:.3f}"
        M["nUnBoot"] = str(U["auc_data_vs_A_marg"]["n_boot"])
    if a.bayes:
        bd = pathlib.Path(a.bayes)
        for tag in ("A", "B"):
            f = bd / f"bayes_uncertainty_{tag}.json"
            if not f.exists(): continue
            U = json.load(open(f)); M[f"nBay{tag}nOOD"] = f"{U['n_ood']:,}".replace(",", "{,}"); M[f"nBay{tag}nCtl"] = f"{U['n_control']:,}".replace(",", "{,}")
            for k, nm in (("exist", "Exist"), ("card", "Card"), ("flow", "Flow")):
                fl = U["flags"][k]; M[f"nBay{tag}{nm}AUC"] = f"{fl['ood_auc']:.2f}"; M[f"nBay{tag}{nm}Ratio"] = f"{fl['median_ood']/max(fl['median_control'],1e-12):.2f}"
            for sp, nm in (("heldout", "OOD"), ("control", "Ctl")):
                c = U[f"calibration_{sp}"]; M[f"nBay{tag}{nm}PullRMS"] = f"{c['pull_rms']:.2f}"; M[f"nBay{tag}{nm}Within"] = f"{100*c['frac_abs_pull_lt2']:.0f}\\%"; M[f"nBay{tag}{nm}EpiStat"] = f"{c['median_epistemic_over_stat']:.2f}"; _r = np.concatenate([(np.array(v["epistemic"], float) / np.array(v["real_err"], float))[np.isfinite(np.array(v["pull"], float)) & (np.array(v["real_err"], float) > 0)] for v in U["binned"][sp].values()]); M[f"nBay{tag}{nm}EpiStatPNinety"] = f"{np.percentile(_r, 90):.2f}"; M[f"nBay{tag}{nm}EpiStatFracHalf"] = f"{100*(_r > 0.5).mean():.0f}\\%"; M[f"nBay{tag}{nm}Bins"] = str(c["n_bins"]); M[f"nBay{tag}{nm}PullRMSnoEpi"] = f"{c.get('pull_rms_no_epistemic', float('nan')):.2f}"
        f = bd / "bayes_positive_control_B.json"
        if f.exists():
            P = json.load(open(f))
            for mode, nm in (("p_x4", "Pfour"), ("tokens_x3", "Tokthree"), ("vtx_upstream", "Vtx"), ("all", "All")):
                for k, kn in (("exist", "Exist"), ("card", "Card"), ("flow", "Flow")):
                    M[f"nPC{nm}{kn}Ratio"] = f"{P[mode][k]['median_ratio']:.1f}"; M[f"nPC{nm}{kn}AUC"] = f"{P[mode][k]['auc_vs_nominal']:.2f}"
    lines = [f"\\newcommand{{\\{k}}}{{{vv}}}" for k, vv in M.items()]
    pathlib.Path(a.out).write_text("% generated by scripts/technote_numbers.py; do not edit\n" + "\n".join(lines) + "\n")
    print(f"{len(M)} macros -> {a.out}")

if __name__ == "__main__":
    main()
