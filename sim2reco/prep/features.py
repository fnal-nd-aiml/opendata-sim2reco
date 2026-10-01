"""Hand-crafted event summaries for the M1 baselines (docs/PROJECT.md §8).

These are NOT the model interface; they are a convenience for tree models and the MDN baselines.
Input: the padded arrays of `EventDataset` (cls [N,n_max], mom [N,n_max,3], mask, ctx [N,5]).
"""
from __future__ import annotations

import numpy as np

from ..constants import CLASS_NAMES, M_MU, N_CLASSES
from .frames import theta_phi_beam

_MASS_OF_CLASS = np.array([0.0, M_MU, 0.511, 0.0, 938.272, 139.570, 139.570, 134.977, 493.677, 497.611,
                           1115.683, 0.0, 939.565])


def class_mass(cls: np.ndarray) -> np.ndarray:
    return _MASS_OF_CLASS[cls]


def muon_true(cls, mom, mask):
    """True muon (px,py,pz) per event: the highest-momentum class-1 particle. Returns (mom[N,3], found[N])."""
    P = np.linalg.norm(mom, axis=-1)
    is_mu = (cls == 1) & mask
    P_mu = np.where(is_mu, P, -1.0)
    j = P_mu.argmax(1)
    found = P_mu[np.arange(len(cls)), j] > 0
    return mom[np.arange(len(cls)), j] * found[:, None], found


def event_features(cls, mom, mask, ctx):
    """Return (X [N,F] float32, names)."""
    N = len(cls)
    P = np.linalg.norm(mom, axis=-1)
    KE = np.where(mask, np.sqrt(P**2 + class_mass(cls) ** 2) - class_mass(cls), 0.0)
    th, ph = theta_phi_beam(mom[..., 0], mom[..., 1], mom[..., 2] + (~mask))  # +1 avoids 0/0 on padding
    mu, found = muon_true(cls, mom, mask)
    mu_P = np.linalg.norm(mu, axis=-1)
    mu_th, mu_ph = theta_phi_beam(mu[:, 0], mu[:, 1], mu[:, 2] + (~found))
    feats = {"mu_P": mu_P, "mu_theta": mu_th, "mu_phi": mu_ph, "mu_found": found.astype(float)}
    had = mask & (cls != 1)
    for c in range(2, N_CLASSES):
        sel = mask & (cls == c)
        name = CLASS_NAMES[c].replace("+", "p").replace("-", "m")
        feats[f"n_{name}"] = sel.sum(1)
        feats[f"sumKE_{name}"] = (KE * sel).sum(1)
        feats[f"maxKE_{name}"] = np.where(sel, KE, 0.0).max(1)
    feats["n_had"] = had.sum(1)
    feats["sumKE_had"] = (KE * had).sum(1)
    feats["maxKE_had"] = np.where(had, KE, 0.0).max(1)
    feats["n_p_ke100"] = (mask & (cls == 4) & (KE > 100)).sum(1)
    feats["n_charged_had"] = (mask & ((cls == 4) | (cls == 5) | (cls == 6) | (cls == 8))).sum(1)
    pt = np.sqrt(mom[..., 0] ** 2 + mom[..., 1] ** 2)
    feats["sum_pt_had"] = (pt * had).sum(1)
    feats["max_theta_had"] = np.where(had, th, 0.0).max(1)
    lead = np.where(had, KE, -1.0).argmax(1)
    feats["lead_had_theta"] = np.where(had.any(1), th[np.arange(N), lead], 0.0)
    feats["lead_had_cls"] = np.where(had.any(1), cls[np.arange(N), lead], 0)
    for i, n in enumerate(["vtx_x", "vtx_y", "vtx_z", "target_Z", "target_A"]):
        feats[n] = ctx[:, i]
    names = list(feats)
    X = np.stack([np.asarray(feats[n], dtype=np.float32) for n in names], 1)
    return X, names
