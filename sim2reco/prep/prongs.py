"""Canonical hadron prong table (docs/PROJECT.md 13.3): tuple <-> per-event prong records.

encode(): tuple arrays -> awkward record array, one list of prongs per event, fields
    theta, phi          direction of the prong in the beam frame (radians), as in the tuple
    has_pion_fit        pion-hypothesis momentum available (tuple: pion_E > 0; P == 1 with E == 0 is a
                        "direction only" sentinel and counts as no fit)
    pi_P                pion-hypothesis momentum magnitude (MeV; -1 if no fit)
    has_proton_fit      the prong appears in the proton block (primary or secondary)
    p_P, p_score1       proton-hypothesis momentum and score (-1 if no proton fit)
    is_primary_proton   the prong is the tuple's MasterAnaDev_proton_* candidate
    is_exiting          MasterAnaDev_hadron_isExiting
    tm_pdg, tm_fraction truth match (validation only; not modelled)

decode(): prong records -> dict of tuple branches with the tuple's shapes and fill conventions.
Round trip encode->decode reproduces the tuple exactly (see tests/test_prongs.py).
"""
from __future__ import annotations

import awkward as ak
import numpy as np

from ..constants import (FILL_PRONG_ANG, FILL_PRONG_INT, FILL_PRONG_MOM, FILL_PRONG_T, FILL_PROTON, M_P, M_PI,
                         N_PRONG_SLOTS)
from .frames import momentum_from_beam_angles

_THETA_TOL = 1e-6


def encode(ev: ak.Array) -> ak.Array:
    n_ev = len(ev)
    hn = ak.to_numpy(ev["MasterAnaDev_hadron_number"]).astype(np.int64)
    P = ak.to_numpy(ev["MasterAnaDev_pion_P"])
    E_pi = ak.to_numpy(ev["MasterAnaDev_pion_E"])
    # The tuple marks "direction known, momentum not fitted" with P == 1 and E == 0 (unit-vector Px,Py,Pz).
    # Treat those as no pion fit; they are ~0.3% of prongs.
    P = np.where(E_pi > 0, P, -1.0)
    th = ak.to_numpy(ev["MasterAnaDev_pion_theta"])
    ph = ak.to_numpy(ev["MasterAnaDev_pion_phi"])
    ex = ak.to_numpy(ev["MasterAnaDev_hadron_isExiting"])
    tmp = ak.to_numpy(ev["MasterAnaDev_hadron_tm_PDGCode"])
    tmf = ak.to_numpy(ev["MasterAnaDev_hadron_tm_fraction"])
    valid = np.arange(N_PRONG_SLOTS)[None, :] < hn[:, None]

    # proton hypothesis, primary
    pP = ak.to_numpy(ev["MasterAnaDev_proton_P_fromdEdx"])
    pE = ak.to_numpy(ev["MasterAnaDev_proton_E_fromdEdx"])
    pth = ak.to_numpy(ev["MasterAnaDev_proton_theta"])
    psc = ak.to_numpy(ev["MasterAnaDev_proton_score1"])
    has_prim = (pP > 0) & (pE > 0)   # same "P == 1, E == 0" sentinel as for pions
    p_P = np.full(P.shape, -1.0)
    p_sc = np.full(P.shape, -1.0)
    is_prim = np.zeros(P.shape, dtype=bool)
    d = np.abs(th - pth[:, None])
    d[~valid] = np.inf
    j = d.argmin(1)
    rows = np.where(has_prim & (d[np.arange(n_ev), j] < _THETA_TOL))[0]
    is_prim[rows, j[rows]] = True
    p_P[rows, j[rows]] = pP[rows]
    p_sc[rows, j[rows]] = psc[rows]
    unmatched = has_prim.sum() - len(rows)
    if unmatched:  # corrupt rows (a handful per million): treat as no proton fit
        import warnings
        warnings.warn(f"{unmatched} primary protons did not match a prong by theta; treated as no proton fit")

    # proton hypothesis, secondaries
    sfit = ev["MasterAnaDev_sec_protons_E_fromdEdx"] > 0
    sP = ev["MasterAnaDev_sec_protons_P_fromdEdx"][sfit]
    sth = ev["MasterAnaDev_sec_protons_theta_fromdEdx"][sfit]
    ssc = ev["MasterAnaDev_sec_protons_proton_scores1"][sfit]
    ev_idx = np.asarray(ak.flatten(ak.broadcast_arrays(np.arange(n_ev), sP)[0]))
    f_th = np.asarray(ak.flatten(sth)); f_P = np.asarray(ak.flatten(sP)); f_sc = np.asarray(ak.flatten(ssc))
    if len(ev_idx):
        d2 = np.abs(th[ev_idx] - f_th[:, None])
        d2[~valid[ev_idx]] = np.inf
        j2 = d2.argmin(1)
        bad = d2[np.arange(len(ev_idx)), j2] >= _THETA_TOL
        if bad.any():
            import warnings
            warnings.warn(f"{bad.sum()} secondary protons did not match a prong by theta; dropped")
        good = ~bad
        # a prong may already hold a proton fit (duplicate prongs at identical theta occur): assign greedily to
        # the first free prong with the same theta instead of overwriting
        for i in np.where(good)[0]:
            e, j = ev_idx[i], j2[i]
            if p_P[e, j] > 0:
                cands = np.where(valid[e] & (np.abs(th[e] - f_th[i]) < _THETA_TOL) & (p_P[e] <= 0))[0]
                if len(cands) == 0:
                    continue  # genuinely redundant entry
                j = cands[0]
            p_P[e, j] = f_P[i]; p_sc[e, j] = f_sc[i]

    rec = ak.zip({
        "theta": th, "phi": ph,
        "has_pion_fit": P > 0, "pi_P": np.where(P > 0, P, -1.0),
        "has_proton_fit": p_P > 0, "p_P": p_P, "p_score1": p_sc, "is_primary_proton": is_prim,
        "is_exiting": ex, "tm_pdg": tmp, "tm_fraction": tmf,
    }, depth_limit=2)
    return ak.drop_none(ak.mask(rec, valid))  # drop invalid slots -> jagged lists


def _fill_slots(vals: ak.Array, fill, dtype=np.float64) -> np.ndarray:
    padded = ak.fill_none(ak.pad_none(vals, N_PRONG_SLOTS, clip=True), fill)
    return ak.to_numpy(padded).astype(dtype)


def decode(prongs: ak.Array, choose_primary: str = "flag") -> dict:
    """Prong records -> tuple branches.

    choose_primary: "flag" uses is_primary_proton (exact round trip of real data);
                    "score" picks the highest p_score1 among prongs with a proton fit (for generated events).
    """
    n_ev = len(prongs)
    n = ak.to_numpy(ak.num(prongs)).astype(np.int32)
    out = {"MasterAnaDev_hadron_number": n, "n_prongs": n + 1}

    th = _fill_slots(prongs["theta"], FILL_PRONG_ANG)
    ph = _fill_slots(prongs["phi"], FILL_PRONG_ANG)
    P = _fill_slots(prongs["pi_P"], FILL_PRONG_MOM)
    fit = P > 0
    E = np.where(fit, np.sqrt(P**2 + M_PI**2), FILL_PRONG_MOM)
    out["MasterAnaDev_pion_P"] = P
    out["MasterAnaDev_pion_E"] = E
    out["MasterAnaDev_pion_T"] = np.where(fit, E - M_PI, FILL_PRONG_T)
    out["MasterAnaDev_pion_theta"] = th
    out["MasterAnaDev_pion_phi"] = ph
    px, py, pz = momentum_from_beam_angles(P, th, ph)
    out["MasterAnaDev_pion_Px"] = np.where(fit, px, FILL_PRONG_MOM)
    out["MasterAnaDev_pion_Py"] = np.where(fit, py, FILL_PRONG_MOM)
    out["MasterAnaDev_pion_Pz"] = np.where(fit, pz, FILL_PRONG_MOM)
    out["MasterAnaDev_hadron_isExiting"] = _fill_slots(prongs["is_exiting"], FILL_PRONG_INT, np.int32)

    has_p = _fill_slots(prongs["has_proton_fit"], False, bool)
    pP = _fill_slots(prongs["p_P"], -1.0)
    psc = _fill_slots(prongs["p_score1"], -1.0)
    if choose_primary == "flag":
        prim = _fill_slots(prongs["is_primary_proton"], False, bool)
    else:
        sc = np.where(has_p, psc, -np.inf)
        prim = np.zeros_like(has_p)
        rows = np.where(has_p.any(1))[0]
        prim[rows, sc[rows].argmax(1)] = True
    has_prim = prim.any(1)
    jp = np.where(has_prim, prim.argmax(1), 0)
    r = np.arange(n_ev)

    def scal(vals, fill=FILL_PROTON):
        return np.where(has_prim, vals[r, jp], fill)

    Pp = scal(pP); thp = scal(th); php = scal(ph)
    Ep = np.where(has_prim, np.sqrt(Pp**2 + M_P**2), FILL_PROTON)
    out["MasterAnaDev_proton_P_fromdEdx"] = Pp
    out["MasterAnaDev_proton_E_fromdEdx"] = Ep
    out["MasterAnaDev_proton_T_fromdEdx"] = np.where(has_prim, Ep - M_P, FILL_PROTON)
    ppx, ppy, ppz = momentum_from_beam_angles(Pp, thp, php)
    out["MasterAnaDev_proton_Px_fromdEdx"] = np.where(has_prim, ppx, FILL_PROTON)
    out["MasterAnaDev_proton_Py_fromdEdx"] = np.where(has_prim, ppy, FILL_PROTON)
    out["MasterAnaDev_proton_Pz_fromdEdx"] = np.where(has_prim, ppz, FILL_PROTON)
    out["MasterAnaDev_proton_theta"] = thp
    out["MasterAnaDev_proton_phi"] = php
    out["MasterAnaDev_proton_score1"] = scal(psc)

    sec = has_p & ~prim
    counts = sec.sum(1)
    ei, si = np.where(sec)
    sPf, sthf, sphf, sscf = pP[ei, si], th[ei, si], ph[ei, si], psc[ei, si]
    sEf = np.sqrt(sPf**2 + M_P**2)
    spx, spy, spz = momentum_from_beam_angles(sPf, sthf, sphf)

    def jag(flat):
        return ak.unflatten(flat, counts)

    out["MasterAnaDev_sec_protons_P_fromdEdx"] = jag(sPf)
    out["MasterAnaDev_sec_protons_E_fromdEdx"] = jag(sEf)
    out["MasterAnaDev_sec_protons_T_fromdEdx"] = jag(sEf - M_P)
    out["MasterAnaDev_sec_protons_Px_fromdEdx"] = jag(spx)
    out["MasterAnaDev_sec_protons_Py_fromdEdx"] = jag(spy)
    out["MasterAnaDev_sec_protons_Pz_fromdEdx"] = jag(spz)
    out["MasterAnaDev_sec_protons_theta_fromdEdx"] = jag(sthf)
    out["MasterAnaDev_sec_protons_proton_scores1"] = jag(sscf)
    return out
