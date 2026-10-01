"""M2 training and evaluation: set encoder + Tier 0 / cardinality heads + Tier 1 flow matching."""
from __future__ import annotations

import json
import pathlib
import time

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from ..data.compact import MODEL_NAMES, TIER1_NAMES, CompactDataset, Tier1Transform, collate, load_compact
from ..data.dataset import split_by_subrun
from ..eval import plots
from ..eval.metrics import binary_metrics, calibration_by_bin, confusion, multiclass_metrics, sample_vs_real_1d
from ..models.surrogate import Surrogate
from ..prep.features import event_features


def make_loaders(d, split, tf, bs, seed, workers=4):
    idx = {k: np.where(split == v)[0] for k, v in (("train", 0), ("val", 1), ("test", 2))}
    ds = {k: CompactDataset(d, v, tf, seed) for k, v in idx.items()}
    ld = {k: DataLoader(ds[k], batch_size=bs, shuffle=(k == "train"), collate_fn=collate, num_workers=workers,
                        drop_last=(k == "train"), persistent_workers=workers > 0) for k in ds}
    return idx, ds, ld


def to_dev(b, dev):
    return {k: v.to(dev, non_blocking=True) for k, v in b.items()}


def apply_holdout(d, split, exclude_inttype):
    """Remove interaction modes (GENIE mc_intType codes) from train AND validation; they stay in test."""
    if exclude_inttype:
        held = np.isin(d["intType"], exclude_inttype)
        split = split.copy(); split[held & (split != 2)] = -1
        print(f"holdout: {held.sum():,} events of intType {exclude_inttype} removed from train/val "
              f"({(held & (split == 2)).sum():,} remain in test)", flush=True)
    return split


def train(stems, out_dir, epochs=20, bs=1024, lr=3e-4, seed=0, device="cuda", d_model=128, n_layers=4,
          flow_hidden=768, flow_layers=5, max_train_events=None, log_every=200, exclude_inttype=None,
          ke_cut_mev=50.0, keep_neutrons=False):
    out = pathlib.Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed); np.random.seed(seed)
    t0 = time.time()
    d = load_compact(stems, ke_cut_mev=ke_cut_mev, keep_neutrons=keep_neutrons)
    split = apply_holdout(d, split_by_subrun(d["subrun"], seed=seed), exclude_inttype)
    if max_train_events:
        tr = np.where(split == 0)[0]
        drop = np.random.default_rng(seed).permutation(tr)[max_train_events:]
        split[drop] = -1
    ok_tr = (split == 0) & d["reco_exists"]
    tf = Tier1Transform().fit(d["tier1"][ok_tr], d["mu_true"][ok_tr], d["ctx"][ok_tr], seed)
    idx, ds, ld = make_loaders(d, split, tf, bs, seed)
    print(f"data: {len(d['subrun'])} events ({(split==0).sum()} train / {(split==1).sum()} val / {(split==2).sum()} test), "
          f"{d['offsets'][-1]} particles, {time.time()-t0:.0f} s", flush=True)

    model = Surrogate(d_model, 4, n_layers, flow_hidden, flow_layers).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * len(ld["train"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)
    hist, best, step = [], np.inf, 0
    print(f"model: {n_par/1e6:.2f} M parameters, {steps} steps", flush=True)
    for ep in range(epochs):
        model.train(); agg = {}
        for b in ld["train"]:
            b = to_dev(b, device)
            L = model.losses(b); loss = sum(L.values())
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); step += 1
            for k, v in L.items(): agg[k] = agg.get(k, 0) + v.item()
            if step % log_every == 0:
                print(f"  step {step} " + " ".join(f"{k} {v.item():.4f}" for k, v in L.items()), flush=True)
        model.eval(); va = {}
        with torch.no_grad():
            for b in ld["val"]:
                L = model.losses(to_dev(b, device))
                for k, v in L.items(): va[k] = va.get(k, 0) + v.item() / len(ld["val"])
        rec = {"epoch": ep, "train": {k: v / len(ld["train"]) for k, v in agg.items()}, "val": va, "time": time.time() - t0}
        hist.append(rec); tot = sum(va.values())
        if not np.isfinite(tot):
            print("  WARNING: non-finite validation loss", flush=True)
        print(f"epoch {ep}: val " + " ".join(f"{k} {v:.4f}" for k, v in va.items()) + f"  total {tot:.4f}  [{rec['time']:.0f} s]", flush=True)
        if np.isfinite(tot) and tot < best:
            best = tot
            torch.save({"model": model.state_dict(), "transform": tf.state(), "config": {"d_model": d_model, "n_layers": n_layers, "flow_hidden": flow_hidden, "flow_layers": flow_layers, "exclude_inttype": list(exclude_inttype or []), "ke_cut_mev": ke_cut_mev, "keep_neutrons": keep_neutrons}}, out / "model.pt")
    (out / "history.json").write_text(json.dumps(hist, indent=1))
    return d, split, tf, idx, ld, out


def write_data_table(stems, d, split, out_dir, epochs=None, n_params=None):
    """One-line data summary for the report (reports/m2/tables/data.tex)."""
    f = lambda n: f"{int(n):,}".replace(",", "{,}")
    n = len(split); tr, va, te = [(split == k).sum() for k in (0, 1, 2)]
    reco = d["reco_exists"].mean()
    txt = (f"{len(stems)} ME FHC MC files, {f(n)} CC~$\\nu_\\mu$ events with $z \\ge 4000$~mm ({reco*100:.1f}\\% reconstructed), "
           f"{f(d['offsets'][-1])} particle tokens; subrun split {f(tr)} / {f(va)} / {f(te)} train / validation / test"
           + (f"; {epochs} epochs" if epochs else "") + (f"; {n_params/1e6:.2f}\\,M parameters" if n_params else "") + ".")
    pathlib.Path(out_dir, "tables").mkdir(parents=True, exist_ok=True)
    (pathlib.Path(out_dir) / "tables" / "data.tex").write_text(txt + "\n")


def load_model(path, device="cuda"):
    ck = torch.load(path, map_location=device, weights_only=False)
    c = ck["config"]; m = Surrogate(c["d_model"], 4, c["n_layers"], c["flow_hidden"], c["flow_layers"]).to(device)
    m.load_state_dict(ck["model"]); m.eval()
    return m, Tier1Transform.from_state(ck["transform"])


@torch.no_grad()
def evaluate(model, tf, d, idx_test, ld_test, out_dir, device="cuda", n_steps=64, seed=0, m1_metrics=None):
    """Closure on the test split: Tier 0 / cardinality metrics, Tier 1 marginals + conditionals, classifier test."""
    out = pathlib.Path(out_dir); (out / "figures").mkdir(exist_ok=True); (out / "tables").mkdir(exist_ok=True)
    torch.manual_seed(seed)
    P0, PN, S = [], [], []
    for b in ld_test:
        b = to_dev(b, device)
        s = model.sample(b, n_steps)
        P0.append(s["p0"].cpu()); PN.append(s["pn"].cpu())
        S.append(torch.cat([s["exist"][:, None].float(), s["minos"][:, None].float(), s["charge"][:, None].float(), s["nprong"][:, None].float(), s["x1"]], 1).cpu())
    P0 = torch.cat(P0).numpy(); PN = torch.cat(PN).numpy(); S = torch.cat(S).numpy()
    t0 = d["tier0"][idx_test]; reco = t0[:, 0] > 0; minos = reco & (t0[:, 1] > 0)
    M = {"n_test": int(len(idx_test))}
    M["tier0"] = {"reco_exists": binary_metrics(t0[:, 0].astype(int), P0[:, 0]),
                  "minos_ok": binary_metrics(t0[reco, 1].astype(int), P0[reco, 1]),
                  "charge_neg": binary_metrics(t0[minos, 2].astype(int), P0[minos, 2])}
    yn = d["nprong"][idx_test].astype(int)
    M["multiplicity"] = multiclass_metrics(yn[reco], PN[reco])
    samp_n = S[reco, 3].astype(int)
    M["multiplicity"]["confusion_true_vs_sampled"] = confusion(yn[reco], np.minimum(samp_n, 8), 9).tolist()
    M["multiplicity"]["marginal_true"] = np.bincount(yn[reco], minlength=9).tolist()
    M["multiplicity"]["marginal_sampled"] = np.bincount(np.minimum(samp_n, 8), minlength=9).tolist()
    # Tier 1: compare on reconstructed test events, teacher-forcing nothing (flags and N are the sampled ones)
    x_real, valid = tf.forward(d["tier1"][idx_test[reco]], d["mu_true"][idx_test[reco]], d["ctx"][idx_test[reco]], np.random.default_rng(seed))
    x_fake = S[reco, 4:]
    finite_fake = np.isfinite(x_fake).all(1)
    M["n_nonfinite_samples"] = int((~finite_fake).sum())
    valid = valid & finite_fake
    x_real, x_fake = x_real[valid], x_fake[valid]
    y_real = d["tier1"][idx_test[reco]][valid]
    y_fake = tf.inverse(x_fake, d["mu_true"][idx_test[reco]][valid], d["ctx"][idx_test[reco]][valid])
    M["n_invalid_tuple_rows"] = int((~valid).sum())
    names_model = MODEL_NAMES
    M["tier1_model_space"] = {n: sample_vs_real_1d(x_real[:, j], x_fake[:, j]) for j, n in enumerate(names_model)}
    M["tier1_raw"] = {n: sample_vs_real_1d(y_real[:, j], y_fake[:, j]) for j, n in enumerate(TIER1_NAMES)}
    mm = (t0[reco, 1] > 0)[valid]; sm = (S[reco, 1] > 0)[valid]
    M["tier1_model_space"]["log_P_ratio_minos_ok"] = sample_vs_real_1d(x_real[mm, 0], x_fake[sm, 0])
    M["tier1_model_space"]["log_P_ratio_no_minos"] = sample_vs_real_1d(x_real[~mm, 0], x_fake[~sm, 0])
    # correlations
    M["corr_real"] = np.corrcoef(x_real.T).round(3).tolist(); M["corr_fake"] = np.corrcoef(x_fake.T).round(3).tolist()
    M["corr_max_abs_diff"] = float(np.abs(np.array(M["corr_real"]) - np.array(M["corr_fake"])).max())
    # derived columns: how well does recoil_E x median-ratio(z) reproduce the tuple's own values?
    for c, nm in ((7, "recoil_passivecorrected"), (8, "hadron_recoil")):
        rr = y_real[:, c] / np.clip(y_real[:, 6], 1e-3, None); rf = y_fake[:, c] / np.clip(y_fake[:, 6], 1e-3, None)
        M[f"derived_{nm}_ratio"] = {"real_q16_50_84": np.percentile(rr, [16, 50, 84]).tolist(), "derived_q16_50_84": np.percentile(rf, [16, 50, 84]).tolist()}

    # classifier test on reconstructed events: (truth summary features, reco vector) real vs fake
    rv = idx_test[reco][valid]
    cls_p, mom_p, mask_p = _pad(d, rv)
    X, _ = event_features(cls_p, mom_p, mask_p, d["ctx"][rv])
    Xr = np.concatenate([X, t0[reco][valid, 1:3], yn[reco][valid][:, None], x_real], 1)
    Xf = np.concatenate([X, S[reco][valid, 1:3], S[reco][valid, 3:4], x_fake], 1)
    Xc = np.concatenate([Xr, Xf]); yc = np.concatenate([np.ones(len(Xr)), np.zeros(len(Xf))])
    rng = np.random.default_rng(seed); perm = rng.permutation(len(Xc)); half = len(perm) // 2
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, early_stopping=True, random_state=seed)
    clf.fit(Xc[perm[:half]], yc[perm[:half]])
    auc = roc_auc_score(yc[perm[half:]], clf.predict_proba(Xc[perm[half:]])[:, 1])
    clf2 = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, early_stopping=True, random_state=seed)
    Xc2 = np.concatenate([x_real, x_fake]); clf2.fit(Xc2[perm[:half]], yc[perm[:half]])
    auc2 = roc_auc_score(yc[perm[half:]], clf2.predict_proba(Xc2[perm[half:]])[:, 1])
    M["classifier_auc_conditional"] = float(auc); M["classifier_auc_marginal"] = float(auc2)
    # which variables carry the discrimination: single columns and blocks (reco vector only, subsample)
    sub = perm[:min(len(perm), 400_000)]; half2 = len(sub) // 2

    def _auc(cols):
        c = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1, early_stopping=True, random_state=seed)
        c.fit(Xc2[sub[:half2]][:, cols], yc[sub[:half2]]); return float(roc_auc_score(yc[sub[half2:]], c.predict_proba(Xc2[sub[half2:]][:, cols])[:, 1]))
    M["classifier_auc_by_variable"] = {n: _auc([j]) for j, n in enumerate(names_model)}
    M["classifier_auc_by_block"] = {"muon": _auc([0, 1, 2]), "vertex": _auc([3, 4, 5]), "calorimetry": _auc([6, 7, 8])}
    # conditional versions: truth summary features + a subset of (flags, N, x). Separates head calibration
    # (flags/N given truth) from the flow's conditional fidelity (x given truth).
    nX = X.shape[1]

    def _auc_cond(extra_cols):
        cols = list(range(nX)) + [nX + c for c in extra_cols]
        c = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1, early_stopping=True, random_state=seed)
        c.fit(Xc[sub[:half2]][:, cols], yc[sub[:half2]]); return float(roc_auc_score(yc[sub[half2:]], c.predict_proba(Xc[sub[half2:]][:, cols])[:, 1]))
    # layout of Xc columns after X: [minos, charge, N, x0..x8]
    M["classifier_auc_conditional_by_block"] = {
        "flags_and_N_only": _auc_cond([0, 1, 2]), "muon": _auc_cond([0, 1, 2, 3, 4, 5]), "vertex": _auc_cond([0, 1, 2, 6, 7, 8]),
        "calorimetry": _auc_cond([0, 1, 2, 9, 10, 11]), "truth_only": _auc_cond([])}

    # conditional checks: muon response vs true P, recoil vs sum KE
    names = event_features(cls_p[:1], mom_p[:1], mask_p[:1], d["ctx"][rv[:1]])[1]
    muP = X[:, names.index("mu_P")] / 1000; ke = X[:, names.index("sumKE_had")]
    M["muon_resp_by_P"] = _by_bin(muP, x_real[:, 0], x_fake[:, 0], np.array([0, 1, 1.5, 2, 3, 4, 6, 8, 12, 20, 40]))
    M["recoil_by_sumKE"] = _by_bin(ke, x_real[:, 6], x_fake[:, 6], np.array([0, 100, 200, 400, 800, 1500, 3000, 6000, 20000]))
    M["n_model_cols"] = len(MODEL_NAMES)
    M["eff_calibration_theta"] = calibration_by_bin(np.degrees(_theta(d["mu_true"][idx_test])), t0[:, 0].astype(int), P0[:, 0], np.arange(0, 61, 5))
    M["eff_calibration_nhad"] = calibration_by_bin(_nhad(d, idx_test), t0[:, 0].astype(int), P0[:, 0], np.arange(-0.5, 10.5, 1))
    (out / "metrics.json").write_text(json.dumps(M, indent=1, default=float))
    _figures(M, x_real, x_fake, y_real, y_fake, mm, sm, out / "figures")
    _tables(M, m1_metrics, out / "tables")
    return M


def _pad(d, idx, n_max=32):
    n = (d["offsets"][idx + 1] - d["offsets"][idx]).astype(int)
    cls = np.zeros((len(idx), n_max), np.int64); mom = np.zeros((len(idx), n_max, 3), np.float32)
    for i, e in enumerate(idx):
        a, b = d["offsets"][e], d["offsets"][e + 1]; k = min(b - a, n_max)
        cls[i, :k] = d["cls"][a:a + k]; mom[i, :k] = d["mom"][a:a + k]
    return cls, mom, cls > 0


def _theta(mu):
    from ..prep.frames import theta_phi_beam
    return theta_phi_beam(mu[:, 0], mu[:, 1], mu[:, 2] + (np.abs(mu).sum(1) == 0))[0]


def _nhad(d, idx):
    n = d["offsets"][idx + 1] - d["offsets"][idx]
    return (n - 1).astype(float)


def _by_bin(x, real, fake, edges):
    rows = []
    for b in range(len(edges) - 1):
        s = (x >= edges[b]) & (x < edges[b + 1])
        if s.sum() > 50:
            rows.append({"lo": float(edges[b]), "hi": float(edges[b + 1]), "n": int(s.sum()),
                         "real_q16_50_84": np.percentile(real[s], [16, 50, 84]).tolist(), "fake_q16_50_84": np.percentile(fake[s], [16, 50, 84]).tolist()})
    return rows


def _figures(M, xr, xf, yr, yf, mm, sm, fdir):
    fig, axs = plots.plt.subplots(1, 3, figsize=(11, 3.2))
    plots.hist_compare(axs[0], xr[mm, 0], xf[sm, 0], np.linspace(-5, 5, 80), "log(P_reco/P_true) [model space], MINOS matched", (plots.REAL_LABEL, "surrogate"))
    plots.hist_compare(axs[1], xr[~mm, 0], xf[~sm, 0], np.linspace(-8, 8, 80), "log(P_reco/P_true) [model space], not matched", (plots.REAL_LABEL, "surrogate")); axs[1].set_yscale("log")
    plots.hist_compare(axs[2], xr[:, 1], xf[:, 1], np.linspace(-8, 8, 80), "dtheta_x [model space]", (plots.REAL_LABEL, "surrogate")); axs[2].set_yscale("log")
    plots.save(fig, fdir / "tier1_muon.png")
    fig, axs = plots.plt.subplots(1, 3, figsize=(11, 3.2))
    for j, (ax, nm) in enumerate(zip(axs, ["dvtx_x", "dvtx_y", "dvtx_z"])):
        plots.hist_compare(ax, xr[:, 3 + j], xf[:, 3 + j], np.linspace(-8, 8, 80), f"{nm} [model space]", (plots.REAL_LABEL, "surrogate")); ax.set_yscale("log")
    plots.save(fig, fdir / "tier1_vertex.png")
    fig, axs = plots.plt.subplots(1, 5, figsize=(16, 3.0))
    for ax, j in zip(axs, (6, 9, 10, 7, 8)):
        nm = TIER1_NAMES[j] + (" (derived)" if j in (7, 8) else "")
        plots.hist_compare(ax, np.log(yr[:, j] + 1), np.log(yf[:, j] + 1), np.linspace(0, 12, 60), f"log({nm}+1)", (plots.REAL_LABEL, "surrogate"))
    plots.save(fig, fdir / "tier1_calorimetry.png")
    fig, axs = plots.plt.subplots(1, 2, figsize=(8, 3.4))
    im = axs[0].imshow(np.array(M["corr_real"]), vmin=-1, vmax=1, cmap="RdBu_r"); axs[0].set_title(f"{plots.REAL_LABEL} correlations", fontsize=9)
    axs[1].imshow(np.array(M["corr_fake"]), vmin=-1, vmax=1, cmap="RdBu_r"); axs[1].set_title("surrogate correlations", fontsize=9)
    k = len(MODEL_NAMES)
    for ax in axs: ax.set_xticks(range(k)); ax.set_yticks(range(k)); ax.set_xticklabels(range(k), fontsize=7); ax.set_yticklabels(range(k), fontsize=7); ax.grid(False)
    fig.colorbar(im, ax=axs, shrink=0.8); fig.savefig(fdir / "tier1_correlations.png", dpi=150); plots.plt.close(fig)
    fig, axs = plots.plt.subplots(1, 2, figsize=(7.5, 3.2))
    plots.bar_compare(axs[0], np.array(M["multiplicity"]["marginal_true"]), np.array(M["multiplicity"]["marginal_sampled"]), "reco hadron prongs", (plots.REAL_LABEL, "surrogate"))
    plots.calibration_panel(axs[1], M["eff_calibration_nhad"], "true hadrons after cuts"); axs[1].set_title("P(reco exists)", fontsize=9, loc="left")
    plots.save(fig, fdir / "heads_multiplicity_eff.png")
    fig, axs = plots.plt.subplots(1, 2, figsize=(7.5, 3.2))
    for ax, rows, xl, log in ((axs[0], M["muon_resp_by_P"], "true muon momentum [GeV]", True), (axs[1], M["recoil_by_sumKE"], "true hadronic KE [MeV]", True)):
        x = [(r["lo"] + r["hi"]) / 2 for r in rows]
        for k, (key, c, lab, ls) in enumerate((("real_q16_50_84", plots.PALETTE["real"], plots.REAL_LABEL, "o-"), ("fake_q16_50_84", plots.PALETTE["model"], "surrogate", "s--"))):
            ax.plot(x, [r[key][1] for r in rows], ls, color=c, ms=4, label=lab); ax.fill_between(x, [r[key][0] for r in rows], [r[key][2] for r in rows], color=c, alpha=0.15)
        ax.set_xscale("log"); ax.set_xlabel(xl); ax.legend(frameon=False)
    axs[0].set_ylabel("log P ratio [model space]"); axs[1].set_ylabel("log recoil_E [model space]")
    plots.save(fig, fdir / "tier1_conditionals.png")


def _tables(M, m1, tdir):
    t0 = M["tier0"]; g = (m1 or {}).get("tier0", {})
    def row(name, k):
        v = t0[k]; b = g.get(k, {})
        return f"{name} & {v['logloss_marginal']:.4f} & {b.get('logloss', float('nan')):.4f} & {v['logloss']:.4f} & {b.get('auc', float('nan')):.3f} & {v['auc']:.3f} \\\\"
    (tdir / "tier0.tex").write_text("\\begin{tabular}{lccccc}\n\\toprule\nTarget & marginal & tree baseline & surrogate & AUC tree & AUC surrogate \\\\\n\\midrule\n"
        + "\n".join([row("reconstructed", "reco_exists"), row("MINOS matched $\\mid$ reco", "minos_ok"), row("negative charge $\\mid$ matched", "charge_neg")]) + "\n\\bottomrule\n\\end{tabular}\n")
    mm = M["multiplicity"]; gm = (m1 or {}).get("multiplicity", {})
    (tdir / "multiplicity.tex").write_text("\\begin{tabular}{lccc}\n\\toprule\n & marginal & tree baseline & surrogate \\\\\n\\midrule\n"
        f"log loss & {mm['logloss_marginal']:.4f} & {gm.get('logloss', float('nan')):.4f} & {mm['logloss']:.4f} \\\\\naccuracy & {mm['accuracy_marginal']:.3f} & {gm.get('accuracy', float('nan')):.3f} & {mm['accuracy']:.3f} \\\\\n" + "\\bottomrule\n\\end{tabular}\n")
    rows = []
    for n, v in M["tier1_model_space"].items():
        r, f = v["real_q16_50_84"], v["fake_q16_50_84"]
        rows.append(f"{n.replace('_', chr(92)+'_')} & {r[1]:.3f} [{r[0]:.3f}, {r[2]:.3f}] & {f[1]:.3f} [{f[0]:.3f}, {f[2]:.3f}] & {v['w1']:.4f} \\\\")
    (tdir / "tier1_marginals.tex").write_text("\\begin{tabular}{lccc}\n\\toprule\nVariable (model space) & MasterAnaDev median [16\\%, 84\\%] & surrogate median [16\\%, 84\\%] & $W_1$ \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    blk = M.get("classifier_auc_by_block", {}); byv = M.get("classifier_auc_by_variable", {})
    cblk = M.get("classifier_auc_conditional_by_block", {})
    blk_rows = "".join(f"block: {k} only & {v:.3f} \\\\\n" for k, v in blk.items())
    blk_rows += "".join(f"truth features + {k.replace('_', ' ')} & {v:.3f} \\\\\n" for k, v in cblk.items())
    var_rows = "".join(f"single variable: {k.replace('_', chr(92)+'_')} & {v:.3f} \\\\\n" for k, v in byv.items())
    (tdir / "closure.tex").write_text("\\begin{tabular}{lc}\n\\toprule\nTest & AUC \\\\\n\\midrule\n"
        f"real vs surrogate, reco vector only & {M['classifier_auc_marginal']:.3f} \\\\\nreal vs surrogate, (truth features, reco vector) & {M['classifier_auc_conditional']:.3f} \\\\\n"
        + "\\midrule\n" + blk_rows + var_rows +
        f"\\midrule\nmax $|\\Delta$corr$|$ over the {M.get('n_model_cols', 9)} generated variables & {M['corr_max_abs_diff']:.3f} \\\\\n" + "\\bottomrule\n\\end{tabular}\n")
