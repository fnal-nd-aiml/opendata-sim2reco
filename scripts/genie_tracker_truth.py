#!/usr/bin/env python
"""GENIE reference sample for generator comparisons: open-dataset Truth events (CC nu_mu) with the true vertex in the
tracker fiducial volume (hexagon of apothem 850 mm, 5990 < z < 8340 mm), subsampled to N events, with the open
dataset's own reconstruction of each event attached as `ref_*` columns (reco_exists, muon P and beam-frame angle,
recoil_E, recoil_nonvtx100, nonvtx_iso_blobs_E, prong count). Usage: scripts/genie_tracker_truth.py OUT.parquet [--n 2000000]"""
import argparse, glob, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import awkward as ak, numpy as np
from sim2reco.prep.particles import in_training_population
from sim2reco.data.compact import load_compact, TIER1_NAMES
from sim2reco.prep.frames import theta_phi_beam
ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--n", type=int, default=2000000); ap.add_argument("--slim-dir", default="data/slim_1A"); ap.add_argument("--seed", type=int, default=0); a = ap.parse_args()
stems = sorted(p[:-len(".truth.parquet")] for p in glob.glob(f"{a.slim_dir}/*.truth.parquet")); d = load_compact(stems)  # compact rows follow the truth files (population-selected) in order
def fiducial(v):
    x, y, z = v[:, 0], v[:, 1], v[:, 2]; ap_ = 850.0  # pointy-top hexagon, apothem 850 mm
    return (np.abs(x) <= ap_) & (np.abs(y) <= 2 * ap_ / np.sqrt(3) - np.abs(x) / np.sqrt(3)) & (z > 5990) & (z < 8340)
parts, off = [], 0
for s in stems:
    t = ak.from_parquet(s + ".truth.parquet"); pop = in_training_population(t); t = t[pop]; n = len(t)
    v = ak.to_numpy(t["mc_vtx"]); fid = fiducial(v); rows = np.arange(off, off + n); off += n
    assert np.allclose(ak.to_numpy(t["mc_vtx"])[:, 2], d["ctx"][rows, 2]), "compact/truth row mismatch"
    t = t[fid]; r = rows[fid]; t1 = d["tier1"][r]; ex = d["reco_exists"][r]
    th, _ = theta_phi_beam(t1[:, 0], t1[:, 1], t1[:, 2])
    extra = {"ref_reco_exists": ex, "ref_muon_P": np.where(ex, np.linalg.norm(t1[:, :3], axis=1), np.nan), "ref_muon_theta": np.where(ex, th, np.nan),
             "ref_recoil_E": np.where(ex, t1[:, 6], np.nan), "ref_recoil_nonvtx100": np.where(ex, t1[:, 9], np.nan), "ref_nonvtx_iso_blobs_E": np.where(ex, t1[:, 10], np.nan),
             "ref_n_prongs": np.where(ex, d["nprong"][r], -1).astype(np.int32)}
    parts.append(ak.Array({**{k: t[k] for k in t.fields}, **extra})); print(f"  {pathlib.Path(s).name}: {fid.sum():,} fiducial tracker events of {n:,}", flush=True)
allt = ak.concatenate(parts); print(f"{len(allt):,} fiducial tracker CC nu_mu events", flush=True)
sel = np.sort(np.random.default_rng(a.seed).permutation(len(allt))[:a.n]); allt = allt[sel]
ak.to_parquet(allt, a.out); print(f"wrote {len(allt):,} -> {a.out}")
