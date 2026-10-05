#!/usr/bin/env python
"""Does 2p2h populate final-state phase space that the other processes leave empty?  Compares intType 8 against
all other CC nu_mu events in (a) the model's input features (classifier 2p2h vs rest), (b) the two-proton
momentum plane for events with >= 2 protons and no pions, (c) the (Q2, W) plane.  Writes a JSON summary and
figures to reports/coverage_2p2h/."""
import glob, json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, awkward as ak
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sim2reco.data.compact import load_compact
from sim2reco.prep.particles import in_training_population
from sim2reco.prep.features import event_features
from sim2reco.train.m2 import _pad
from sim2reco.eval import plots

out = pathlib.Path("reports/coverage_2p2h"); rng = np.random.default_rng(0)
stems = sorted(p[:-len(".truth.parquet")] for p in glob.glob("data/slim_1A/*.truth.parquet"))
d = load_compact(stems)  # adopted selection: 10 MeV, neutrons kept
Q2, W = [], []
MP = 938.272
for s in stems:
    t = ak.from_parquet(s + ".truth.parquet", columns=["mc_vtx", "mc_current", "mc_incoming", "mc_incomingE", "mc_Q2", "mc_FSPartPDG", "mc_FSPartE", "mc_intType"]); pop = in_training_population(t); t = t[pop]
    # the tuple's mc_w is a sentinel (1e8 MeV) for 2p2h, so W is computed from the lepton: W^2 = M^2 + 2 M (E_nu - E_mu) - Q2
    Emu = ak.to_numpy(ak.fill_none(ak.firsts(t["mc_FSPartE"][t["mc_FSPartPDG"] == 13]), np.nan)); Enu = ak.to_numpy(t["mc_incomingE"]); q2 = ak.to_numpy(t["mc_Q2"])
    Q2.append(q2); W.append(np.sqrt(np.clip(MP ** 2 + 2 * MP * (Enu - Emu) - q2, 0, None)))
Q2 = np.concatenate(Q2) / 1e6; W = np.concatenate(W) / 1e3  # GeV^2, GeV (lepton-defined W)
assert len(Q2) == len(d["intType"]), (len(Q2), len(d["intType"]))
is2 = d["intType"] == 8; N = len(is2); print(f"events {N:,}, 2p2h {is2.sum():,} ({100*is2.mean():.1f}%)", flush=True)
R = {"n_events": int(N), "n_2p2h": int(is2.sum()), "frac_2p2h": float(is2.mean())}

# ---- per-event proton kinematics from the model's token set
idx = np.arange(N); cls, mom, mask = _pad(d, idx); P = np.linalg.norm(mom, axis=-1)
isp = mask & (cls == 4); npi = (mask & np.isin(cls, [5, 6, 7])).sum(1); n_p = isp.sum(1)
Pp = np.where(isp, P, -1.0); order = np.argsort(-Pp, axis=1); Ps = np.take_along_axis(Pp, order, 1); p1, p2 = Ps[:, 0], Ps[:, 1]
m1 = np.take_along_axis(mom, order[:, :1, None], 1)[:, 0]; m2 = np.take_along_axis(mom, order[:, 1:2, None], 1)[:, 0]
cos12 = np.where(n_p >= 2, (m1 * m2).sum(1) / np.maximum(p1 * p2, 1e-9), np.nan)
two_p = (n_p >= 2) & (npi == 0)
R["frac_2p0pi"] = {"2p2h": float(two_p[is2].mean()), "rest": float(two_p[~is2].mean())}
R["share_of_2p0pi_that_is_2p2h"] = float(is2[two_p].mean())
print(f"2p (>=2 protons, 0 pions) topology: {100*two_p[is2].mean():.1f}% of 2p2h, {100*two_p[~is2].mean():.1f}% of the rest; 2p2h share of that topology {100*is2[two_p].mean():.1f}%", flush=True)

def coverage(x, y, sel, xe, ye, name):
    """2D occupancy of 2p2h vs rest; a bin is 'untouched' if the rest has no event there, 'dominated' if the rest's
    density (per sample) is below 10% of 2p2h's.  Returns the fraction of 2p2h events in such bins."""
    a = sel & is2; b = sel & ~is2
    H2 = np.histogram2d(x[a], y[a], [xe, ye])[0]; Hr = np.histogram2d(x[b], y[b], [xe, ye])[0]
    f2 = H2 / H2.sum(); fr = Hr / Hr.sum(); ratio = np.where(f2 > 0, fr / np.maximum(f2, 1e-12), np.nan)
    untouched = (Hr == 0) & (H2 > 0); dominated = (H2 > 0) & (ratio < 0.1)
    # how big would a reweighting factor be: for 2p2h events, the ratio of 2p2h sample density to rest density
    w = np.where(fr > 0, f2 / np.maximum(fr, 1e-12), np.inf); ix = np.clip(np.searchsorted(xe, x[a]) - 1, 0, len(xe) - 2); iy = np.clip(np.searchsorted(ye, y[a]) - 1, 0, len(ye) - 2); wev = w[ix, iy]
    res = {"frac_2p2h_in_untouched_bins": float(H2[untouched].sum() / H2.sum()), "frac_2p2h_in_dominated_bins": float(H2[dominated].sum() / H2.sum()),
           "n_bins_2p2h": int((H2 > 0).sum()), "n_untouched": int(untouched.sum()), "n_dominated": int(dominated.sum()),
           "weight_2p2h_over_rest_median": float(np.median(wev)), "weight_p90": float(np.percentile(wev, 90)), "frac_weight_gt10": float((wev > 10).mean()), "frac_weight_inf": float(np.isinf(wev).mean()),
           "n_2p2h": int(a.sum()), "n_rest": int(b.sum())}
    print(f"{name}: 2p2h {a.sum():,} rest {b.sum():,}; 2p2h in untouched bins {100*res['frac_2p2h_in_untouched_bins']:.2f}%, in bins where rest density < 10% of 2p2h {100*res['frac_2p2h_in_dominated_bins']:.1f}%; "
          f"density ratio 2p2h/rest for 2p2h events: median {res['weight_2p2h_over_rest_median']:.1f}, p90 {res['weight_p90']:.1f}, >10 in {100*res['frac_weight_gt10']:.1f}%", flush=True)
    return res, H2, Hr, ratio

pe = np.arange(0, 1501, 50.0); ce = np.linspace(-1, 1, 21); qe = np.linspace(0, 2.0, 41); we = np.linspace(0.8, 2.5, 35); print("W (lepton-defined) quantiles 2p2h", np.nanpercentile(W[is2], [5, 50, 95]).round(2), "rest", np.nanpercentile(W[~is2], [5, 50, 95]).round(2), flush=True)
R["p1_p2"], H2a, Hra, Ra = coverage(p1, p2, two_p, pe, pe, "p1 vs p2 [MeV], 2p0pi")
R["p2_cos12"], H2b, Hrb, Rb = coverage(p2, cos12, two_p, pe, ce, "p2 vs cos(opening), 2p0pi")
R["Q2_W"], H2c, Hrc, Rc = coverage(Q2, W, np.ones(N, bool), qe, we, "Q2 vs W, all")
R["Q2_W_2p0pi"], H2d, Hrd, Rd = coverage(Q2, W, two_p, qe, we, "Q2 vs W, 2p0pi")

# ---- classifier in the model's input features: how separable is 2p2h from the rest, and how pure are its regions
sub = rng.permutation(N)[:1500000]; X, names = event_features(cls[sub], mom[sub], mask[sub], d["ctx"][sub]); y = is2[sub]
X = np.column_stack([X, p1[sub], p2[sub], np.nan_to_num(cos12[sub], nan=-2)]); names = names + ["p1", "p2", "cos12"]
half = len(sub) // 2
def fit(cols, label):
    c = [names.index(n) for n in cols]; m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, max_leaf_nodes=63, random_state=0).fit(X[:half][:, c], y[:half])
    pp = m.predict_proba(X[half:][:, c])[:, 1]; a_ = roc_auc_score(y[half:], pp); print(f"  classifier on {label}: AUC {a_:.3f}", flush=True); return pp, a_
had_cols = [n for n in names if not n.startswith("mu_") and n not in ("vtx_x", "vtx_y", "vtx_z", "target_Z", "target_A", "Z", "A")]
p_had, auc_had = fit(had_cols, f"hadron tokens only ({len(had_cols)} features)"); p_mu, auc_mu = fit(["mu_P", "mu_theta"], "muon P and angle only")
p, auc = fit(names, "all inputs"); yt = y[half:]
R["classifier_variants"] = {"auc_hadrons_only": float(auc_had), "auc_muon_only": float(auc_mu), "auc_all": float(auc), "hadron_cols": had_cols}
R["classifier"] = {"auc": float(auc), "prior": float(y.mean()), "frac_2p2h_with_p_gt_0.5": float((p[yt] > 0.5).mean()), "frac_2p2h_with_p_gt_0.9": float((p[yt] > 0.9).mean()), "frac_2p2h_with_p_gt_0.99": float((p[yt] > 0.99).mean()),
                   "frac_2p2h_with_p_gt_0.5_2p0pi": float((p[yt & two_p[sub][half:]] > 0.5).mean()), "frac_2p2h_with_p_gt_0.9_2p0pi": float((p[yt & two_p[sub][half:]] > 0.9).mean())}
print(f"classifier 2p2h vs rest on input features: AUC {auc:.3f} (prior {y.mean():.3f}); 2p2h events with p>0.5: {100*(p[yt]>0.5).mean():.1f}%, p>0.9: {100*(p[yt]>0.9).mean():.1f}%, p>0.99: {100*(p[yt]>0.99).mean():.2f}%", flush=True)
json.dump(R, open(out / "summary.json", "w"), indent=1)

# ---- figures
def panel(ax, H, xe, ye, title, log=True):
    im = ax.pcolormesh(xe, ye, np.where(H > 0, H, np.nan).T, cmap="viridis", norm=plots.plt.matplotlib.colors.LogNorm() if log else None); ax.set_title(title, fontsize=9, loc="left"); return im
fig, axs = plots.plt.subplots(1, 3, figsize=(12, 3.6))
panel(axs[0], H2a, pe, pe, "2p2h: events"); panel(axs[1], Hra, pe, pe, "all other processes: events")
im = axs[2].pcolormesh(pe, pe, np.where(H2a > 0, Ra, np.nan).T, cmap="coolwarm", norm=plots.plt.matplotlib.colors.LogNorm(vmin=0.03, vmax=30)); axs[2].set_title("density ratio rest / 2p2h (per sample)", fontsize=9, loc="left"); fig.colorbar(im, ax=axs[2])
for ax in axs: ax.set_xlabel("leading proton P [MeV]"); ax.set_ylabel("subleading proton P [MeV]")
fig.suptitle("Events with ≥2 protons (KE ≥ 10 MeV) and no pions", fontsize=10); plots.save(fig, out / "figures" / "coverage_p1_p2.png")
fig, axs = plots.plt.subplots(1, 3, figsize=(12, 3.6))
panel(axs[0], H2c, qe, we, "2p2h: events"); panel(axs[1], Hrc, qe, we, "all other processes: events")
im = axs[2].pcolormesh(qe, we, np.where(H2c > 0, Rc, np.nan).T, cmap="coolwarm", norm=plots.plt.matplotlib.colors.LogNorm(vmin=0.03, vmax=30)); axs[2].set_title("density ratio rest / 2p2h (per sample)", fontsize=9, loc="left"); fig.colorbar(im, ax=axs[2])
for ax in axs: ax.set_xlabel("true $Q^2$ [GeV$^2$]"); ax.set_ylabel("true $W$ from the lepton [GeV]")
plots.save(fig, out / "figures" / "coverage_Q2_W.png")
fig, ax = plots.plt.subplots(figsize=(5.5, 3.4)); b = np.linspace(0, 1, 51)
ax.hist(p[~yt], b, histtype="step", color=plots.PALETTE["real"], label="other processes", density=True); ax.hist(p[yt], b, histtype="step", color=plots.PALETTE["model"], label="2p2h", density=True)
ax.set_yscale("log"); ax.set_xlabel("classifier P(2p2h | model inputs)"); ax.set_ylabel("density"); ax.legend(frameon=False, fontsize=8); ax.set_title(f"AUC {auc:.3f}", fontsize=9, loc="left"); plots.save(fig, out / "figures" / "coverage_classifier.png")
print("done")

# ---- within the shared 2p0pi topology: what separates 2p2h from the rest, and is there a region only 2p2h fills?
tp = two_p[sub]; Xt, yt2 = X[tp], y[tp]; h = len(Xt) // 2
def fit2(cols, label):
    c = [names.index(n) for n in cols]; m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, max_leaf_nodes=63, random_state=0).fit(Xt[:h][:, c], yt2[:h])
    pp = m.predict_proba(Xt[h:][:, c])[:, 1]; a_ = roc_auc_score(yt2[h:], pp); return pp, a_
groups = {"all inputs": names, "hadron tokens only": had_cols, "counts only": [n for n in had_cols if n.startswith("n_")], "KE sums/max only": [n for n in had_cols if "KE" in n],
          "proton pair (p1, p2, cos12)": ["p1", "p2", "cos12"], "muon only": ["mu_P", "mu_theta"], "hadron tokens without counts": [n for n in had_cols if not n.startswith("n_")]}
R["within_2p0pi"] = {"prior": float(yt2.mean()), "n": int(len(Xt))}
print(f"within the 2p0pi topology ({len(Xt):,} events, 2p2h share {100*yt2.mean():.1f}%):", flush=True)
for g, cols in groups.items():
    pp, a_ = fit2(cols, g); t_ = yt2[h:]; R["within_2p0pi"][g] = {"auc": float(a_), "frac_2p2h_p_gt_0.9": float((pp[t_] > 0.9).mean()), "frac_2p2h_p_gt_0.99": float((pp[t_] > 0.99).mean()), "frac_rest_p_gt_0.9": float((pp[~t_] > 0.9).mean())}
    print(f"  {g:32s} AUC {a_:.3f}; 2p2h events with P>0.9: {100*(pp[t_]>0.9).mean():5.1f}%, P>0.99: {100*(pp[t_]>0.99).mean():5.2f}%; rest with P>0.9: {100*(pp[~t_]>0.9).mean():.2f}%", flush=True)
# multiplicity composition within 2p0pi
nh = X[:, names.index("n_had")]; nn = X[:, names.index("n_neutron")] if "n_neutron" in names else None
for lab, m_ in (("2p2h", tp & y), ("rest", tp & ~y)):
    print(f"  {lab}: n_had distribution within 2p0pi", {int(k): round(float(v), 3) for k, v in zip(*np.unique(nh[m_], return_counts=True)) if v / m_.sum() > 0.005 for v in [v / m_.sum()]}, flush=True)
    if nn is not None: print(f"  {lab}: n_neutron distribution", {int(k): round(float(v / m_.sum()), 3) for k, v in zip(*np.unique(nn[m_], return_counts=True)) if v / m_.sum() > 0.005}, flush=True)
json.dump(R, open(out / "summary.json", "w"), indent=1); print("done2")
