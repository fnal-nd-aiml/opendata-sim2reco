"""Input pipeline (docs/PROJECT.md 13.1): generator final-state particles -> model input particle set.

Works on awkward arrays with one entry per event. Generator-agnostic: needs only pdg, px, py, pz, E.
"""
from __future__ import annotations

import awkward as ak
import numpy as np

from ..constants import PDG_CLASS, PDG_MASS

DEFAULT_KE_CUT_MEV = 10.0   # adopted 2026-10-02 (was 50); see docs/PROJECT.md §18
DEFAULT_KEEP_NEUTRONS = True  # neutrons are input tokens (adopted 2026-10-02)
DROP_PDG_ABS = {12, 14, 16}                 # neutrinos (always); neutrons are dropped unless keep_neutrons
HADRON_CLASSES = {4, 5, 6, 7, 8, 9, 10, 12} # p, pi+, pi-, pi0, K+-, K0, hyperon, n: KE cut applies
OTHER_CLASS = 11


def _isin(arr: ak.Array, values) -> ak.Array:
    """Elementwise membership test for a jagged array against a small set of values."""
    out = None
    for v in values:
        m = arr == v
        out = m if out is None else (out | m)
    return out


def pdg_to_class(pdg):
    """Vectorised PDG -> species class. Unknown -> OTHER_CLASS (11)."""
    flat = np.asarray(ak.flatten(pdg)) if isinstance(pdg, ak.Array) else np.asarray(pdg)
    out = np.full(flat.shape, OTHER_CLASS, dtype=np.int32)
    for p, c in PDG_CLASS.items():
        out[flat == p] = c
    if isinstance(pdg, ak.Array):
        return ak.unflatten(out, ak.num(pdg))
    return out


def pdg_to_mass(pdg):
    flat = np.asarray(ak.flatten(pdg)) if isinstance(pdg, ak.Array) else np.asarray(pdg)
    out = np.zeros(flat.shape, dtype=np.float64)
    for p, m in PDG_MASS.items():
        out[flat == p] = m
    if isinstance(pdg, ak.Array):
        return ak.unflatten(out, ak.num(pdg))
    return out


def select_particles(pdg, px, py, pz, E, ke_cut_mev: float = DEFAULT_KE_CUT_MEV, keep_neutrons: bool = DEFAULT_KEEP_NEUTRONS) -> ak.Array:
    """Apply the 13.1 selection and return a record array {cls, px, py, pz} per event.

    Steps: drop neutrinos, GENIE pseudo-particles (2000000101), nuclear remnants (pdg > 1e9) and, unless
    keep_neutrons, neutrons; drop hadrons (neutrons included when kept) with KE < ke_cut_mev; map pdg -> class.
    Leptons and photons are never KE-cut.
    """
    apdg = abs(pdg)
    keep = ~_isin(apdg, sorted(DROP_PDG_ABS)) & (pdg != 2000000101) & (apdg < 1_000_000_000)
    if not keep_neutrons:
        keep = keep & (pdg != 2112)
    cls = pdg_to_class(pdg)
    ke = E - pdg_to_mass(pdg)
    is_had = _isin(cls, sorted(HADRON_CLASSES))
    keep = keep & ~(is_had & (ke < ke_cut_mev))
    return ak.zip({"cls": cls[keep], "px": px[keep], "py": py[keep], "pz": pz[keep]})


def select_from_tuple(events: ak.Array, ke_cut_mev: float = DEFAULT_KE_CUT_MEV, keep_neutrons: bool = DEFAULT_KEEP_NEUTRONS) -> ak.Array:
    """Convenience wrapper on slimmed tuple arrays (mc_FSPart* branches)."""
    return select_particles(events["mc_FSPartPDG"], events["mc_FSPartPx"], events["mc_FSPartPy"],
                            events["mc_FSPartPz"], events["mc_FSPartE"], ke_cut_mev, keep_neutrons)


def context_from_tuple(events: ak.Array) -> np.ndarray:
    """Context vector (13.2): vtx_x, vtx_y, vtx_z, target_Z, target_A. Shape (n_events, 5)."""
    vtx = ak.to_numpy(events["mc_vtx"])[:, :3]
    Z = ak.to_numpy(events["mc_targetZ"]).astype(np.float64)[:, None]
    A = ak.to_numpy(events["mc_targetA"]).astype(np.float64)[:, None]
    return np.concatenate([vtx, Z, A], axis=1)


def is_cc_numu(events: ak.Array) -> np.ndarray:
    return (ak.to_numpy(events["mc_current"]) == 1) & (ak.to_numpy(events["mc_incoming"]) == 14)


def in_training_population(events: ak.Array, z_min: float = 4000.0, z_max: float = 8700.0) -> np.ndarray:
    """v1 training population (docs/PROJECT.md 14.2): CC nu_mu with true vertex z in [z_min, z_max]."""
    z = ak.to_numpy(events["mc_vtx"])[:, 2]
    return is_cc_numu(events) & (z >= z_min) & (z <= z_max)
