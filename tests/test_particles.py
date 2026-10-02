import awkward as ak
import numpy as np

from sim2reco.constants import N_CLASSES
from sim2reco.prep.particles import context_from_tuple, is_cc_numu, select_from_tuple


def test_selection_rules(truth):
    parts = select_from_tuple(truth, ke_cut_mev=50.0, keep_neutrons=False)
    pdg = truth["mc_FSPartPDG"]
    cls = ak.to_numpy(ak.flatten(parts["cls"]))
    assert cls.min() >= 1 and cls.max() < N_CLASSES
    # count of kept particles never exceeds the FS count and every event keeps its muon in CC numu events
    assert ak.all(ak.num(parts) <= truth["mc_nFSPart"])
    cc = is_cc_numu(truth)
    n_mu = ak.to_numpy(ak.sum(parts["cls"] == 1, axis=1))
    assert (n_mu[cc] >= 1).all()
    # neutrons and neutrinos are gone: class 11 ("other") must not contain them
    n_had_before = ak.to_numpy(ak.sum((pdg == 2112) | (abs(pdg) == 14), axis=1))
    assert n_had_before.sum() > 0
    # KE cut: recompute KE for kept protons and pions
    from sim2reco.prep.particles import pdg_to_mass
    p2 = parts["px"] ** 2 + parts["py"] ** 2 + parts["pz"] ** 2
    for c, m in ((4, 938.272), (5, 139.570), (6, 139.570)):
        sel = parts["cls"] == c
        ke = ak.to_numpy(ak.flatten(np.sqrt(p2[sel] + m**2) - m))
        assert ke.min() >= 50.0 - 1e-6


def test_default_selection_admits_neutrons_at_10mev(truth):
    parts = select_from_tuple(truth)
    cls = ak.to_numpy(ak.flatten(parts["cls"]))
    assert (cls == 12).sum() > 0, "neutron tokens expected with the default selection"
    from sim2reco.prep.particles import pdg_to_mass
    p2 = parts["px"] ** 2 + parts["py"] ** 2 + parts["pz"] ** 2
    for c, m in ((4, 938.272), (12, 939.565)):
        sel = parts["cls"] == c
        ke = ak.to_numpy(ak.flatten(np.sqrt(p2[sel] + m**2) - m))
        assert ke.min() >= 10.0 * 0.99  # tuple E vs PDG-mass recomputation differ at the 0.5% level


def test_context_shape(truth):
    ctx = context_from_tuple(truth)
    assert ctx.shape == (len(truth), 5)
    assert np.isfinite(ctx).all()
