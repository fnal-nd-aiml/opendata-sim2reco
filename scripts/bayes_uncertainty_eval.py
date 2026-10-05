#!/usr/bin/env python
"""Epistemic-uncertainty validation of a surrogate with Bayesian last layers.

(1) Per-event epistemic flag on a held-out-mode sample vs an in-distribution control; AUC of the flag as an
    out-of-distribution detector (no truth needed).
(2) Calibration against the open dataset: K posterior weight draws -> K generated samples per event -> binned
    predictions with an epistemic spread; pulls (surrogate - data) / sqrt(epistemic^2 + stat^2).
Usage: scripts/bayes_uncertainty_eval.py OUT_DIR --model DIR [--ood-inttype 8] [--n-control 60000] [--draws 10]
"""
import argparse, glob, json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from sim2reco.data.compact import load_compact, CompactDataset, collate
from sim2reco.data.dataset import split_by_subrun
from sim2reco.train.m3 import load_model
from sim2reco.train.m2 import to_dev, _pad
from sim2reco.models.bayes_last import FeatureTap, GaussianLastLayer, HEADS, head_layer
from sim2reco.prep.features import event_features
from sim2reco.eval import plots

ap = argparse.ArgumentParser(); ap.add_argument("out_dir"); ap.add_argument("--model", required=True); ap.add_argument("--slim-dir", default="data/slim_1A")
ap.add_argument("--ood-inttype", type=int, nargs="*", default=[8]); ap.add_argument("--n-control", type=int, default=60000); ap.add_argument("--draws", type=int, default=10); ap.add_argument("--steps", type=int, default=64); ap.add_argument("--tag", default="")
a = ap.parse_args(); out = pathlib.Path(a.out_dir); (out / "figures").mkdir(parents=True, exist_ok=True); (out / "tables").mkdir(exist_ok=True); dev = "cuda"; rng = np.random.default_rng(0)
D = pathlib.Path(a.model); model, tf, ptf = load_model(D / "model.pt"); ck = torch.load(D / "model.pt", map_location="cpu", weights_only=False)["config"]
post = {k: GaussianLastLayer.from_state(s, dev) for k, s in torch.load(D / "bayes_last.pt", map_location="cpu", weights_only=False).items()}
layers = {k: head_layer(model, p) for k, (p, _) in HEADS.items()}; taps = {k: FeatureTap(l) for k, l in layers.items()}
stems = sorted(p[:-len(".truth.parquet")] for p in glob.glob(f"{a.slim_dir}/*.truth.parquet"))
d = load_compact(stems, ke_cut_mev=ck.get("ke_cut_mev", 50.0), keep_neutrons=ck.get("keep_neutrons", False)); split = split_by_subrun(d["subrun"], seed=0)
te = np.where(split == 2)[0]; ood = te[np.isin(d["intType"][te], a.ood_inttype)]; ctl = rng.permutation(te[~np.isin(d["intType"][te], a.ood_inttype)])[:a.n_control]
print(f"OOD events {len(ood):,}, control {len(ctl):,}", flush=True)

def flag_pass(idx):
    """Per-event epistemic variances: reco-exists logit, multiplicity logits (mean), event-flow final state (sum over outputs)."""
    ds = CompactDataset(d, idx, tf, 0, prong_tf=ptf); o = {"exist": [], "card": [], "flow": []}
    with torch.no_grad():
        for b in DataLoader(ds, batch_size=2048, collate_fn=collate, num_workers=4):
            b = to_dev(b, dev); z, _ = model.enc(b["cls"], b["mom"], b["mask"], b["ctx"])
            model.tier0(z); o["exist"].append(post["tier0"].var(taps["tier0"].h)[:, 0].cpu())
            pn = torch.softmax(model.card(z), -1); o["card"].append(post["card"].var(taps["card"].h).mean(1).cpu())
            # event flow: sample with the mean weights while accumulating g = sum_s dt h_s (features at the k2 evaluation)
            flags = torch.stack([torch.rand(len(z), device=dev) < torch.sigmoid(model.tier0(z))[:, 1], torch.zeros(len(z), dtype=torch.bool, device=dev)], -1).float()
            nprong = torch.multinomial(pn, 1)[:, 0]; cond = model.flow_cond(z, flags, nprong)
            x = torch.randn(len(z), model.flow.dim, device=dev); dt = 1.0 / a.steps; g = torch.zeros(len(z), layers["flow"].in_features, device=dev)
            for i in range(a.steps):
                t = torch.full((len(z),), i * dt, device=dev); k1 = model.flow.v(x, t, cond); k2 = model.flow.v(x + 0.5 * dt * k1, t + 0.5 * dt, cond); g += dt * taps["flow"].h; x = (x + dt * k2).clamp(-20, 20)
            o["flow"].append(post["flow"].var(g).sum(1).cpu())
    return {k: torch.cat(v).numpy() for k, v in o.items()}

F_ood, F_ctl = flag_pass(ood), flag_pass(ctl)
R = {"n_ood": int(len(ood)), "n_control": int(len(ctl)), "flags": {}}
for k in F_ood:
    y = np.r_[np.ones(len(ood)), np.zeros(len(ctl))]; s = np.r_[F_ood[k], F_ctl[k]]
    R["flags"][k] = {"median_ood": float(np.median(F_ood[k])), "median_control": float(np.median(F_ctl[k])), "ood_auc": float(roc_auc_score(y, s)),
                     "frac_ood_above_control_p95": float((F_ood[k] > np.percentile(F_ctl[k], 95)).mean())}
    print(f"flag {k:6s}: median OOD {np.median(F_ood[k]):.4g} vs control {np.median(F_ctl[k]):.4g}; OOD-detection AUC {R['flags'][k]['ood_auc']:.3f}; OOD above control 95th pct: {R['flags'][k]['frac_ood_above_control_p95']:.3f}", flush=True)
fig, axs = plots.plt.subplots(1, 3, figsize=(11, 3.2))
for ax, k, lab in zip(axs, ["exist", "card", "flow"], ["epistemic var., reco-exists logit", "epistemic var., multiplicity logits", "epistemic var., event flow sample"]):
    lo, hi = np.percentile(np.r_[F_ood[k], F_ctl[k]], [0.5, 99.5]); b = np.logspace(np.log10(max(lo, 1e-8)), np.log10(hi), 50)
    ax.hist(F_ctl[k], bins=b, histtype="step", density=True, color=plots.PALETTE["third"], label=f"control (in distribution), n={len(ctl):,}")
    ax.hist(F_ood[k], bins=b, histtype="step", density=True, color=plots.PALETTE["model"], label=f"held-out 2p2h, n={len(ood):,}, separation AUC {R['flags'][k]['ood_auc']:.2f}")
    ax.set_xscale("log"); ax.set_xlabel(lab); ax.set_ylabel("density"); ax.legend(frameon=False, fontsize=7)
plots.save(fig, out / "figures" / f"bayes_flags{a.tag}.png")

# ---------- calibration with K posterior draws ----------
def binned(idx, draws):
    """For each posterior draw: sample Tier 0, multiplicity and the event flow with perturbed last layers; return
    binned means of several observables against truth variables (fine binning; bins need >= 30 events)."""
    reco_real = d["reco_exists"][idx]; ir = idx[reco_real]
    cls_p, mom_p, mask_p = _pad(d, idx); X, names = event_features(cls_p, mom_p, mask_p, d["ctx"][idx])
    nhad = X[:, names.index("n_had")]; ke = X[:, names.index("sumKE_had")]; muP = X[:, names.index("mu_P")] / 1000
    xr, valid = tf.forward(d["tier1"][ir], d["mu_true"][ir], d["ctx"][ir], rng)
    ke_edges = np.logspace(np.log10(30), np.log10(20000), 17); p_edges = np.logspace(np.log10(0.5), np.log10(40), 15); n_edges = np.arange(-0.5, 12.5, 1)
    def bin_mean(x, y, edges):
        m, e_ = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            s = (x >= lo) & (x < hi) & np.isfinite(y)
            m.append(y[s].mean() if s.sum() >= 30 else np.nan); e_.append(y[s].std() / np.sqrt(s.sum()) if s.sum() >= 30 else np.nan)
        return np.array(m), np.array(e_)
    nreal = d["nprong"][idx].astype(float); nreal[~reco_real] = np.nan
    spec = {  # name: (x for all events or reco events, real y, edges, x-axis label, y label)
        "eff_vs_nhad": (nhad, reco_real.astype(float), n_edges, "true hadrons after cuts", "P(reco muon candidate)"),
        "nprong_vs_nhad": (nhad, nreal, n_edges, "true hadrons after cuts", "mean reco prongs"),
        "recoil_vs_ke": (ke[reco_real][valid], xr[:, 6], ke_edges, "true hadronic KE [MeV]", "mean log recoil_E [model space]"),
        "nonvtx100_vs_ke": (ke[reco_real][valid], xr[:, 7], ke_edges, "true hadronic KE [MeV]", "mean log non-vertex E [model space]"),
        "blobs_vs_ke": (ke[reco_real][valid], xr[:, 8], ke_edges, "true hadronic KE [MeV]", "mean log isolated-blob E [model space]"),
        "muP_vs_P": (muP[reco_real][valid], xr[:, 0], p_edges, "true muon momentum [GeV]", "mean log P ratio [model space]"),
        "dthx_vs_P": (muP[reco_real][valid], xr[:, 1], p_edges, "true muon momentum [GeV]", "mean dtheta_x [model space]"),
    }
    real = {k: bin_mean(v[0], v[1], v[2]) for k, v in spec.items()}
    ds = CompactDataset(d, idx, tf, 0, prong_tf=ptf); preds = {k: [] for k in spec}; pstat = {k: [] for k in spec}
    W0 = {k: (layers[k].weight.data.clone(), layers[k].bias.data.clone()) for k in ("tier0", "card", "flow")}
    gen = torch.Generator(device=dev); gen.manual_seed(123)
    with torch.no_grad():
        for kdraw in range(draws):
            for k in ("tier0", "card", "flow"):
                dW, db = post[k].sample_delta(gen); layers[k].weight.data = W0[k][0] + dW; layers[k].bias.data = W0[k][1] + db
            torch.manual_seed(kdraw + 7); S = []
            for b in DataLoader(ds, batch_size=2048, collate_fn=collate, num_workers=4):
                s_ = model.sample(to_dev(b, dev), a.steps); S.append(torch.cat([s_["exist"][:, None].float(), s_["nprong"][:, None].float(), s_["x1"]], 1).cpu())
            S = torch.cat(S).numpy(); ex = S[:, 0] > 0; npr = S[:, 1].copy(); npr[~ex] = np.nan
            Sv = S[reco_real][valid]; kv = ke[reco_real][valid]; pv_ = muP[reco_real][valid]
            draws_ = {"eff_vs_nhad": bin_mean(nhad, S[:, 0], n_edges), "nprong_vs_nhad": bin_mean(nhad, npr, n_edges),
                      "recoil_vs_ke": bin_mean(kv, Sv[:, 2 + 6], ke_edges), "nonvtx100_vs_ke": bin_mean(kv, Sv[:, 2 + 7], ke_edges), "blobs_vs_ke": bin_mean(kv, Sv[:, 2 + 8], ke_edges),
                      "muP_vs_P": bin_mean(pv_, Sv[:, 2 + 0], p_edges), "dthx_vs_P": bin_mean(pv_, Sv[:, 2 + 1], p_edges)}
            for k, (m_, e_) in draws_.items(): preds[k].append(m_); pstat[k].append(e_)
    for k in ("tier0", "card", "flow"): layers[k].weight.data, layers[k].bias.data = W0[k]
    out_ = {}
    for k, v in spec.items():
        P = np.array(preds[k]); mu = np.nanmean(P, 0); r_, se = real[k]
        # across-draw variance = epistemic + the surrogate's own sampling noise; the K-draw mean keeps only 1/K of the latter
        stat_s2 = np.nanmean(np.array(pstat[k]) ** 2, 0)
        ep = np.sqrt(np.clip(np.nanvar(P, 0) - stat_s2, 0, None))
        pull = (mu - r_) / np.sqrt(ep ** 2 + stat_s2 / draws + se ** 2)
        e = v[2]; centers = (np.sqrt(e[:-1] * e[1:]) if e[0] > 0 else 0.5 * (e[:-1] + e[1:])).tolist()
        out_[k] = {"centers": centers, "xlabel": v[3], "ylabel": v[4], "logx": bool(e[0] > 0 and e[1] / e[0] > 1.1), "real": r_.tolist(), "real_err": se.tolist(), "pred": mu.tolist(), "epistemic": ep.tolist(), "surrogate_stat": np.sqrt(stat_s2).tolist(), "pull": pull.tolist()}
    return out_

C = {"heldout": binned(ood, a.draws), "control": binned(ctl, a.draws)}
for s_, res in C.items():
    pulls = np.concatenate([np.array(v["pull"]) for v in res.values()]); pulls = pulls[np.isfinite(pulls)]
    eps = np.concatenate([np.array(v["epistemic"]) / np.maximum(np.array(v["real_err"]), 1e-9) for v in res.values()]); eps = eps[np.isfinite(eps)]
    # pull without the epistemic term: tells whether the epistemic band is needed at all
    pulls0 = np.concatenate([(np.array(v["pred"]) - np.array(v["real"])) / np.sqrt(np.array(v["surrogate_stat"]) ** 2 / a.draws + np.array(v["real_err"]) ** 2) for v in res.values()]); pulls0 = pulls0[np.isfinite(pulls0)]
    R[f"calibration_{s_}"] = {"n_bins": int(len(pulls)), "pull_rms": float(np.sqrt(np.mean(pulls ** 2))), "pull_mean": float(pulls.mean()), "frac_abs_pull_lt2": float((np.abs(pulls) < 2).mean()),
                              "median_epistemic_over_stat": float(np.median(eps)), "pull_rms_no_epistemic": float(np.sqrt(np.mean(pulls0 ** 2)))}
    print(f"calibration {s_:8s}: {len(pulls)} bins, pull RMS {R[f'calibration_{s_}']['pull_rms']:.2f} (without epistemic term {R[f'calibration_{s_}']['pull_rms_no_epistemic']:.2f}), |pull|<2 in {R[f'calibration_{s_}']['frac_abs_pull_lt2']:.2f}, epistemic/stat median {np.median(eps):.2f}", flush=True)
R["binned"] = C
json.dump(R, open(out / f"bayes_uncertainty{a.tag}.json", "w"), indent=1, default=float)
print("done")
