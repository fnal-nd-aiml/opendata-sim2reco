import awkward as ak
import numpy as np
import pytest

from sim2reco.prep.prongs import decode, encode

FLOAT_TOL = 1e-6


def _cmp(name, a, b, tol=FLOAT_TOL):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    assert a.shape == b.shape, name
    bad = ~np.isclose(a, b, rtol=tol, atol=tol)
    assert not bad.any(), f"{name}: {bad.sum()} mismatches, e.g. {a[bad][:3]} vs {b[bad][:3]}"


def _cmp_jag(name, a, b, tol=FLOAT_TOL):
    assert ak.all(ak.num(a) == ak.num(b)), f"{name}: counts differ"
    _cmp(name, ak.flatten(a), ak.flatten(b), tol)


def test_round_trip_exact(reco):
    prongs = encode(reco)
    assert ak.all(ak.num(prongs) == reco["MasterAnaDev_hadron_number"])
    out = decode(prongs, choose_primary="flag")
    # "direction only" sentinel slots (P == 1, E == 0) are decoded as no-fit (-1); exclude them from the comparison
    sentinel = (ak.to_numpy(reco["MasterAnaDev_pion_E"]) == 0) & (ak.to_numpy(reco["MasterAnaDev_pion_P"]) > 0)
    psent = (ak.to_numpy(reco["MasterAnaDev_proton_E_fromdEdx"]) == 0) & (ak.to_numpy(reco["MasterAnaDev_proton_P_fromdEdx"]) > 0)
    ssent = ak.to_numpy(ak.sum(reco["MasterAnaDev_sec_protons_E_fromdEdx"] == 0, axis=1)) > 0
    # secondary protons whose theta matches no prong (corrupt rows, dropped by the encoder with a warning)
    th = ak.to_numpy(reco["MasterAnaDev_pion_theta"]); hn = ak.to_numpy(reco["MasterAnaDev_hadron_number"])
    sth = reco["MasterAnaDev_sec_protons_theta_fromdEdx"]
    for e in np.where(ak.to_numpy(ak.num(sth)) > 0)[0]:
        for t in sth[e]:
            if hn[e] == 0 or np.abs(th[e, :hn[e]] - t).min() >= 1e-6: ssent[e] = True
    print(f"\nsentinel slots excluded: pion {sentinel.sum()}, primary proton {psent.sum()}, events with sec-proton sentinel {ssent.sum()}")
    for k, v in out.items():
        if isinstance(v, ak.Array):
            _cmp_jag(k, v[~ssent], reco[k][~ssent])
        else:
            tol = 1e-4 if ("_E_" in k or "_T_" in k or k.endswith("_E") or k.endswith("_T")) else FLOAT_TOL
            ref = np.asarray(reco[k], dtype=float)
            if k.startswith("MasterAnaDev_pion_") and ref.ndim == 2 and k not in ("MasterAnaDev_pion_theta", "MasterAnaDev_pion_phi"):
                ref = np.where(sentinel, v, ref)
            if k.startswith("MasterAnaDev_proton_"):
                ref = np.where(psent, v, ref)
            _cmp(k, v, ref, tol)


def test_primary_by_score_agreement(reco):
    """How often does 'highest proton score1' reproduce MasterAnaDev's primary-proton choice?"""
    prongs = encode(reco)
    by_flag = decode(prongs, "flag")["MasterAnaDev_proton_P_fromdEdx"]
    by_score = decode(prongs, "score")["MasterAnaDev_proton_P_fromdEdx"]
    has = by_flag > 0
    agree = np.isclose(by_flag[has], by_score[has]).mean()
    print(f"\nprimary-by-score agreement: {agree:.4f} over {has.sum()} events with a primary proton")
    assert agree > 0.9


def test_prong_fields_sane(reco):
    prongs = encode(reco)
    flat = ak.flatten(prongs)
    assert ak.all((flat["pi_P"] > 0) == flat["has_pion_fit"])
    assert ak.all((flat["p_P"] > 0) == flat["has_proton_fit"])
    assert ak.all(flat["is_primary_proton"] <= flat["has_proton_fit"])
    assert ak.sum(prongs["is_primary_proton"], axis=1).to_numpy().max() == 1
