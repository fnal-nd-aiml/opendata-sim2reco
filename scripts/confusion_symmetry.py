#!/usr/bin/env python
"""Event-by-event multiplicity confusion: open dataset vs one surrogate draw, two surrogate draws, and the joint
matrix (symmetry check). Usage: scripts/confusion_symmetry.py MODEL_DIR [--slim-dir data/slim_1A] [--n 300000]"""
import argparse, glob, json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, torch
from torch.utils.data import DataLoader
from sim2reco.data.compact import load_compact, CompactDataset, collate
from sim2reco.data.dataset import split_by_subrun
from sim2reco.train.m3 import load_model
from sim2reco.train.m2 import to_dev
from sim2reco.eval import plots

ap = argparse.ArgumentParser(); ap.add_argument("model_dir"); ap.add_argument("--slim-dir", default="data/slim_1A"); ap.add_argument("--n", type=int, default=300000)
a = ap.parse_args(); out = pathlib.Path(a.model_dir)
stems = sorted(p[:-len(".truth.parquet")] for p in glob.glob(f"{a.slim_dir}/*.truth.parquet"))
d = load_compact(stems); split = split_by_subrun(d["subrun"], seed=0); te = np.where(split == 2)[0][:a.n]
model, tf, ptf = load_model(out / "model.pt"); ds = CompactDataset(d, te, tf, 0, prong_tf=ptf)
def draw(seed):
    torch.manual_seed(seed); o = []
    for b in DataLoader(ds, batch_size=4096, collate_fn=collate, num_workers=4):
        _, pn, _ = model.predict_probs(to_dev(b, "cuda")); o.append(torch.multinomial(pn, 1)[:, 0].cpu())
    return torch.cat(o).numpy()
x, y = draw(1), draw(2); reco = d["reco_exists"][te]; nr = np.minimum(d["nprong"][te][reco], 6); x, y = np.minimum(x[reco], 6), np.minimum(y[reco], 6)
def conf(p, q): M = np.zeros((7, 7)); np.add.at(M, (p, q), 1); return M
Crs, Css = conf(nr, x), conf(x, y); J = Crs / Crs.sum(); rn = lambda M: M / np.clip(M.sum(1, keepdims=True), 1, None)
labels = [str(k) for k in range(6)] + ["6+"]
fig, axs = plots.plt.subplots(1, 3, figsize=(12, 3.6))
for ax, M, title, yl, xl, vmax in ((axs[0], rn(Crs), f"{plots.REAL_LABEL} vs one surrogate draw", f"{plots.REAL_LABEL} reco prongs", "surrogate reco prongs", 1),
                                   (axs[1], rn(Css), "two independent surrogate draws", "surrogate draw 1", "surrogate draw 2", 1),
                                   (axs[2], J, f"joint (unnormalised) {plots.REAL_LABEL} vs surrogate", f"{plots.REAL_LABEL} reco prongs", "surrogate reco prongs", J.max())):
    im = ax.imshow(M, vmin=0, vmax=vmax, cmap="Blues", origin="lower")
    for i in range(7):
        for j in range(7):
            if M[i, j] >= 0.005: ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=6.5, color="white" if M[i, j] > 0.6 * vmax else "black")
    ax.set_xticks(range(7)); ax.set_yticks(range(7)); ax.set_xticklabels(labels, fontsize=8); ax.set_yticklabels(labels, fontsize=8); ax.grid(False); ax.set_xlabel(xl, fontsize=9); ax.set_ylabel(yl, fontsize=9); ax.set_title(title, fontsize=9, loc="left")
fig.tight_layout(); fig.savefig(out / "figures" / "m3_confusion_symmetry.png", dpi=150); plots.plt.close(fig)
res = {"joint_max_asymmetry": float(np.abs(J - J.T).max()), "joint_max_entry": float(J.max()), "sample_vs_sample_max_asymmetry": float(np.abs(Css / Css.sum() - (Css / Css.sum()).T).max()), "n_reco_events": int(reco.sum())}
json.dump(res, open(out / "confusion_symmetry.json", "w"), indent=1); print(res)
