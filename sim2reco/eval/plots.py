"""Matplotlib helpers for the performance report. Okabe-Ito colorblind-safe palette, thin marks, one axis."""
from __future__ import annotations

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REAL_LABEL = os.environ.get("SIM2RECO_REAL_LABEL", "MasterAnaDev")  # legend label of the real reconstruction
PALETTE = {"real": "#0072B2", "model": "#E69F00", "third": "#009E73", "fourth": "#D55E00", "gray": "#7f7f7f"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "lines.linewidth": 1.6, "figure.dpi": 150})


def hist_compare(ax, real, fake, bins, xlabel, labels=None, density=True):
    labels = labels or (REAL_LABEL, "surrogate")
    ax.hist(real, bins=bins, histtype="step", color=PALETTE["real"], label=labels[0], density=density)
    ax.hist(fake, bins=bins, histtype="step", color=PALETTE["model"], label=labels[1], density=density, ls="--")
    ax.set_xlabel(xlabel); ax.set_ylabel("density" if density else "events"); ax.legend(frameon=False)


def calibration_panel(ax, rows, xlabel):
    x = [(r["lo"] + r["hi"]) / 2 for r in rows]
    ax.plot(x, [r["observed"] for r in rows], "o-", color=PALETTE["real"], ms=4, label="observed")
    ax.plot(x, [r["predicted"] for r in rows], "s--", color=PALETTE["model"], ms=4, label="predicted")
    ax.set_xlabel(xlabel); ax.set_ylabel("fraction"); ax.set_ylim(0, 1); ax.legend(frameon=False)


def bar_compare(ax, counts_real, counts_fake, xlabel, labels=None):
    labels = labels or (REAL_LABEL, "surrogate")
    k = np.arange(len(counts_real)); w = 0.4
    ax.bar(k - w / 2, counts_real / counts_real.sum(), w, color=PALETTE["real"], label=labels[0])
    ax.bar(k + w / 2, counts_fake / counts_fake.sum(), w, color=PALETTE["model"], label=labels[1])
    ax.set_xlabel(xlabel); ax.set_ylabel("fraction"); ax.set_xticks(k); ax.legend(frameon=False)


def save(fig, path):
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
