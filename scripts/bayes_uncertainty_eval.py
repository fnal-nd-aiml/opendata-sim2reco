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
    ax.hist(F_ood[k], bins=b, histtype="step", density=True, color=plots.PALETTE["model"], label=f"held-out 2p2h, n={len(ood):,}, AUC {R['flags'][k]['ood_auc']:.2f}")
    ax.set_xscale("log"); ax.set_xlabel(lab); ax.set_ylabel("density"); ax.legend(frameon=False, fontsize=7)
plots.save(fig, out / "figures" / f"bayes_flags{a.tag}.png")

# ---------- calibration with K posterior draws ----------
def binned(idx, draws):
    """For each draw: sample Tier 0 and the event flow with perturbed last layers; return binned means."""
    reco_real = d["reco_exists"][idx]; ir = idx[reco_real]
    cls_p, mom_p, mask_p = _pad(d, idx); X, names = event_features(cls_p, mom_p, mask_p, d["ctx"][idx])
    nhad = X[:, names.index("n_had")]; ke = X[:, names.index("sumKE_had")]; muP = X[:, names.index("mu_P")] / 1000
    xr, valid = tf.forward(d["tier1"][ir], d["mu_true"][ir], d["ctx"][ir], rng)
    ke_edges = np.array([0, 100, 200, 400, 800, 1500, 3000, 6000, 20000]); p_edges = np.array([0, 1, 1.5, 2, 3, 4, 6, 8, 12, 20, 40]); n_edges = np.arange(-0.5, 8.5, 1)
    def bin_mean(x, y, edges, mask=None):
        m = []; e_ = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            s = (x >= lo) & (x < hi) & (mask if mask is not None else True); s = s & np.isfinite(y)
            m.append(y[s].mean() if s.sum() > 30 else np.nan); e_.append(y[s].std() / np.sqrt(s.sum()) if s.sum() > 30 else np.nan)
        return np.array(m), np.array(e_)
    real = {"eff": bin_mean(nhad, reco_real.astype(float), n_edges), "recoil": bin_mean(ke[reco_real][valid], xr[:, 6], ke_edges), "muP": bin_mean(muP[reco_real][valid], xr[:, 0], p_edges)}
    ds = CompactDataset(d, idx, tf, 0, prong_tf=ptf); preds = {k: [] for k in real}
    W0 = {k: (layers[k].weight.data.clone(), layers[k].bias.data.clone()) for k in ("tier0", "flow")}
    gen = torch.Generator(device=dev); gen.manual_seed(123)
    with torch.no_grad():
        for kdraw in range(draws):
            for k in ("tier0", "flow"):
                dW, db = post[k].sample_delta(gen); layers[k].weight.data = W0[k][0] + dW; layers[k].bias.data = W0[k][1] + db
            S = []
            for b in DataLoader(ds, batch_size=2048, collate_fn=collate, num_workers=4):
                s = model.sample(to_dev(b, dev), a.steps); S.append(torch.cat([s["exist"][:, None].float(), s["x1"]], 1).cpu())
            S = torch.cat(S).numpy()
            preds["eff"].append(bin_mean(nhad, S[:, 0], n_edges)[0])
            preds["recoil"].append(bin_mean(ke[reco_real][valid], S[reco_real][valid][:, 7], ke_edges)[0])
            preds["muP"].append(bin_mean(muP[reco_real][valid], S[reco_real][valid][:, 1], p_edges)[0])
            torch.manual_seed(kdraw + 7)
    for k in ("tier0", "flow"): layers[k].weight.data, layers[k].bias.data = W0[k]
    out_ = {}
    for k in real:
        P = np.array(preds[k]); mu, ep = np.nanmean(P, 0), np.nanstd(P, 0); r_, se = real[k]
        pull = (mu - r_) / np.sqrt(ep ** 2 + se ** 2 + (ep * 0 + np.nanstd(P, 0) ** 2 / draws))  # include MC error of the draw mean
        out_[k] = {"real": r_.tolist(), "real_err": se.tolist(), "pred": mu.tolist(), "epistemic": ep.tolist(), "pull": pull.tolist()}
    return out_

C = {"ood": binned(ood, a.draws), "control": binned(ctl, a.draws)}
for s_, res in C.items():
    pulls = np.concatenate([np.array(v["pull"]) for v in res.values()]); pulls = pulls[np.isfinite(pulls)]
    eps = np.concatenate([np.array(v["epistemic"]) / np.maximum(np.array(v["real_err"]), 1e-9) for v in res.values()]); eps = eps[np.isfinite(eps)]
    R[f"calibration_{s_}"] = {"n_bins": int(len(pulls)), "pull_rms": float(np.sqrt(np.mean(pulls ** 2))), "pull_mean": float(pulls.mean()), "frac_abs_pull_lt2": float((np.abs(pulls) < 2).mean()), "median_epistemic_over_stat": float(np.median(eps))}
    print(f"calibration {s_:8s}: {len(pulls)} bins, pull RMS {R[f'calibration_{s_}']['pull_rms']:.2f}, |pull|<2 in {R[f'calibration_{s_}']['frac_abs_pull_lt2']:.2f}, epistemic/stat median {np.median(eps):.2f}", flush=True)
R["binned"] = C
json.dump(R, open(out / f"bayes_uncertainty{a.tag}.json", "w"), indent=1, default=float)
# figure: binned recoil and efficiency with epistemic bands, OOD vs control, plus pull histogram
fig, axs = plots.plt.subplots(1, 3, figsize=(12, 3.3))
ke_c = np.sqrt(np.array([0, 100, 200, 400, 800, 1500, 3000, 6000])[0:] * np.array([100, 200, 400, 800, 1500, 3000, 6000, 20000])); n_c = np.arange(0, 8)
for (ax, key, xc, xl, yl, logx) in ((axs[0], "recoil", ke_c, "true hadronic KE [MeV]", "mean log recoil_E [model space]", True), (axs[1], "eff", n_c, "true hadrons after cuts", "P(reco muon candidate)", False)):
    for s_, c, ls in (("control", plots.PALETTE["third"], "s--"), ("ood", plots.PALETTE["model"], "^:")):
        v = C[s_][key]; r_, se, mu, ep = (np.array(v[k]) for k in ("real", "real_err", "pred", "epistemic"))
        ax.errorbar(xc, r_, yerr=se, fmt="o", color=c, ms=3, alpha=0.6, label=f"open dataset, {s_}"); ax.plot(xc, mu, ls, color=c, ms=4, label=f"surrogate mean, {s_}"); ax.fill_between(xc, mu - ep, mu + ep, color=c, alpha=0.2, label=f"epistemic band, {s_}" if key == "recoil" else None)
    if logx: ax.set_xscale("log")
    ax.set_xlabel(xl); ax.set_ylabel(yl); ax.legend(frameon=False, fontsize=6)
allp = {s_: np.concatenate([np.array(v["pull"]) for v in C[s_].values()]) for s_ in C}
b = np.linspace(-6, 6, 25)
axs[2].hist(allp["control"][np.isfinite(allp["control"])], bins=b, histtype="step", color=plots.PALETTE["third"], label=f"control, RMS {R['calibration_control']['pull_rms']:.2f}")
axs[2].hist(allp["ood"][np.isfinite(allp["ood"])], bins=b, histtype="step", color=plots.PALETTE["model"], label=f"held-out 2p2h, RMS {R['calibration_ood']['pull_rms']:.2f}")
axs[2].set_xlabel("pull: (surrogate - open dataset) / total uncertainty"); axs[2].set_ylabel("bins"); axs[2].legend(frameon=False, fontsize=7)
plots.save(fig, out / "figures" / f"bayes_calibration{a.tag}.png")
print("done")
