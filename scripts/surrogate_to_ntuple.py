#!/usr/bin/env python
"""Run the surrogate on truth events and write a pruned MasterAnaDev-style ntuple (docs/PROJECT.md 13.4).

Usage: scripts/surrogate_to_ntuple.py MODEL_DIR TRUTH.parquet OUT.root [--n N] [--steps 64] [--seed 0]
Input: a slimmed Truth-tree Parquet (or any file with the same truth branches). Output tree `MasterAnaDev`
with the truth passthrough branches, the sampled Tier 0 flags, the muon/vertex/calorimetry blocks, the prong
multiplicity and, for an M3 model, the prong branches (pion/proton/sec_protons blocks) and a plane-snapped
vertex z. Only events the surrogate marks as reconstructed are written (like the real tuple).
"""
import argparse, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import awkward as ak, numpy as np, torch
from torch.utils.data import DataLoader

from sim2reco.data.compact import CompactDataset, collate
from sim2reco.io.writer import write_ntuple
from sim2reco.prep.muon import decode_muon
from sim2reco.prep.particles import context_from_tuple, in_training_population, select_from_tuple
from sim2reco.prep.prongs import decode as decode_prongs
from sim2reco.train.m2 import to_dev
from sim2reco.train import m2, m3

PASSTHROUGH = ["mc_run", "mc_subrun", "mc_nthEvtInFile", "eventID", "mc_vtx", "mc_targetZ", "mc_targetA", "truth_targetID",
               "mc_nFSPart", "mc_FSPartPDG", "mc_FSPartPx", "mc_FSPartPy", "mc_FSPartPz", "mc_FSPartE", "mc_primFSLepton",
               "mc_incoming", "mc_current", "mc_intType"]


def truth_to_compact(truth, ke_cut_mev=10.0, keep_neutrons=True):
    parts = select_from_tuple(truth, ke_cut_mev, keep_neutrons); n = ak.to_numpy(ak.num(parts["cls"])).astype(np.int32)
    pdg = truth["mc_FSPartPDG"]; mu = pdg == 13
    Pmu = np.sqrt(truth["mc_FSPartPx"] ** 2 + truth["mc_FSPartPy"] ** 2 + truth["mc_FSPartPz"] ** 2)
    lead = ak.argmax(ak.where(mu, Pmu, -1.0), axis=1, keepdims=True)
    mu_true = np.stack([ak.to_numpy(ak.flatten(ak.fill_none(truth[f"mc_FSPart{c}"][lead], 0.0))) for c in ("Px", "Py", "Pz")], 1).astype(np.float32)
    N = len(truth)
    return {"offsets": np.concatenate([[0], np.cumsum(n)]).astype(np.int64),
            "cls": ak.to_numpy(ak.flatten(parts["cls"])).astype(np.int8),
            "mom": np.stack([ak.to_numpy(ak.flatten(parts[c])) for c in ("px", "py", "pz")], 1).astype(np.float32),
            "ctx": context_from_tuple(truth).astype(np.float32), "mu_true": mu_true,
            "reco_exists": np.zeros(N, bool), "tier0": np.zeros((N, 3), np.float32), "nprong": np.zeros(N, np.int8),
            "tier1": np.full((N, 11), np.nan, np.float32), "subrun": ak.to_numpy(truth["mc_subrun"]).astype(np.int32),
            "prongs": np.zeros((0, 9), np.float32), "p_offsets": np.zeros(N + 1, np.int64)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir"); ap.add_argument("truth"); ap.add_argument("out")
    ap.add_argument("--n", type=int, default=None); ap.add_argument("--steps", type=int, default=64); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    truth = ak.from_parquet(a.truth)
    truth = truth[in_training_population(truth)]
    if a.n: truth = truth[:a.n]
    ck = torch.load(pathlib.Path(a.model_dir) / "model.pt", map_location="cpu", weights_only=False)
    tier2 = ck["config"].get("tier2", False)
    d = truth_to_compact(truth, ck["config"].get("ke_cut_mev", 50.0), ck["config"].get("keep_neutrons", False))
    if tier2:
        model, tf, ptf = m3.load_model(pathlib.Path(a.model_dir) / "model.pt")
    else:
        model, tf = m2.load_model(pathlib.Path(a.model_dir) / "model.pt"); ptf = None
    ds = CompactDataset(d, np.arange(len(truth)), tf, a.seed, prong_tf=ptf)  # ptf=None -> Tier 1 only
    S, PR, PM, VC = [], [], [], []
    for b in DataLoader(ds, batch_size=2048, collate_fn=collate, num_workers=4):
        s = model.sample(to_dev(b, "cuda"), a.steps)
        S.append(torch.cat([s["exist"][:, None].float(), s["minos"][:, None].float(), s["charge"][:, None].float(), s["nprong"][:, None].float(), s["x1"]], 1).cpu())
        if tier2:
            PR.append(s["prongs"].cpu()); PM.append(s["pmask"].cpu()); VC.append(s["vclass"].cpu())
    S = torch.cat(S).numpy()
    exist = S[:, 0] > 0
    y = tf.inverse(S[exist, 4:], d["mu_true"][exist], d["ctx"][exist])
    if tier2:  # snap the vertex z to a plane when the class head says so
        vc = torch.cat(VC).numpy()[exist]
        zs = tf.planes.z_for_class(vc, d["ctx"][exist, 2], y[:, 5]); y[:, 5] = np.where(vc > 0, zs, y[:, 5])
    minos, charge = S[exist, 1] > 0, S[exist, 2] > 0
    out = {k: truth[k][exist] for k in PASSTHROUGH}
    out.update(decode_muon(y[:, 0], y[:, 1], y[:, 2], minos, charge))
    vtx = np.zeros((exist.sum(), 4)); vtx[:, :3] = y[:, 3:6]
    out["MasterAnaDev_vtx"] = vtx; out["vtx"] = vtx.copy()
    out["MasterAnaDev_recoil_E"] = y[:, 6]; out["MasterAnaDev_recoil_E_wide_window"] = y[:, 6]; out["MasterAnaDev_hadron_recoil_CCInc"] = y[:, 6]
    out["MasterAnaDev_recoil_passivecorrected"] = y[:, 7]; out["MasterAnaDev_hadron_recoil_default"] = y[:, 7]
    out["MasterAnaDev_hadron_recoil"] = y[:, 8]; out["recoil_energy_nonmuon_nonvtx100mm"] = y[:, 9]; out["nonvtx_iso_blobs_energy"] = y[:, 10]
    n = S[exist, 3].astype(np.int32)
    out["MasterAnaDev_hadron_number"] = n; out["n_prongs"] = n + 1; out["multiplicity"] = n + 1
    if tier2:  # prong branches via the canonical prong table (docs/PROJECT.md 13.3)
        Npad = max(p.shape[1] for p in PR)
        padp = lambda p, v=0.0: torch.nn.functional.pad(p, (0, 0, 0, Npad - p.shape[1]) if p.dim() == 3 else (0, Npad - p.shape[1]), value=v)
        pr = torch.cat([padp(p) for p in PR]).numpy()[exist]; pm = torch.cat([padp(p, False) for p in PM]).numpy()[exist]
        flat = ptf.inverse(pr[pm]); counts = pm.sum(1)
        rec = ak.unflatten(ak.zip({"theta": flat[:, 0], "phi": flat[:, 1], "has_pion_fit": flat[:, 2] > 0.5, "pi_P": flat[:, 3],
                                   "has_proton_fit": flat[:, 4] > 0.5, "p_P": flat[:, 5], "p_score1": flat[:, 6],
                                   "is_primary_proton": flat[:, 7], "is_exiting": flat[:, 8].astype(np.int32),
                                   "tm_pdg": np.zeros(len(flat), np.int32), "tm_fraction": np.zeros(len(flat))}), counts)
        # primary proton: highest is_primary score among prongs with a proton fit
        prim = ak.to_numpy(ak.fill_none(ak.pad_none(rec["is_primary_proton"], Npad, clip=True), -np.inf))
        best = np.zeros_like(prim, dtype=bool); has = np.isfinite(prim).any(1)
        best[np.where(has)[0], prim[has].argmax(1)] = True
        rec = ak.with_field(rec, ak.unflatten(best[pm], counts), "is_primary_proton")
        out.update(decode_prongs(rec, choose_primary="flag"))
        out["MasterAnaDev_hadron_number"] = n; out["n_prongs"] = n + 1
    out["surrogate_p_reco_exists"] = np.zeros(exist.sum())  # placeholder for downstream weighting; filled below
    p0 = None
    # per-event efficiency probability (useful for reweighting) from a second pass over probabilities
    probs = []
    for b in DataLoader(ds, batch_size=2048, collate_fn=collate, num_workers=4):
        p, _, _ = model.predict_probs(to_dev(b, "cuda")); probs.append(p[:, 0].cpu())
    out["surrogate_p_reco_exists"] = torch.cat(probs).numpy()[exist].astype(np.float64)
    write_ntuple(a.out, out)
    print(f"{len(truth)} truth events -> {exist.sum()} reconstructed ({exist.mean()*100:.1f}%) -> {a.out}")


if __name__ == "__main__":
    main()
