#!/usr/bin/env python
"""Convert NuWro output trees to the open-dataset Truth-tree layout the surrogate reads (docs/PROJECT.md 13).

Final-state particles are NuWro's post-FSI list (`post`), rotated from the beam frame (NuWro: beam along +z)
into the detector frame. NuWro has no detector geometry, so each event gets a vertex drawn from the open
dataset's fiducial tracker events with the same target nucleus, which preserves the position-nucleus
correlation of the real detector. Extra branches `nuwro_dyn`, `nuwro_weight`, `nuwro_job` are carried through.

Usage: scripts/nuwro_to_truth.py NUWRO_DIR OUT.truth.parquet [--slim-dir data/slim_1A] [--seed 0]
"""
import argparse, glob, json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import awkward as ak, numpy as np, uproot
from sim2reco.prep.frames import from_beam

M_P = 938.272; DYN_TO_INTTYPE = {0: 1, 2: 2, 4: 3, 6: 4, 8: 8, 10: 0}  # NuWro dyn (CC codes) -> GENIE intType; hyperon production -> 0
ap = argparse.ArgumentParser(); ap.add_argument("nuwro_dir"); ap.add_argument("out"); ap.add_argument("--slim-dir", default="data/slim_1A"); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args(); rng = np.random.default_rng(a.seed)

# ---- vertex pool from the open dataset: fiducial tracker CC nu_mu events, keyed by target Z
pool = {}
for f in sorted(glob.glob(f"{a.slim_dir}/*.truth.parquet")):
    t = ak.from_parquet(f, columns=["mc_vtx", "mc_targetZ", "truth_is_fiducial", "truth_vtx_module", "mc_current", "mc_incoming"])
    v = ak.to_numpy(t["mc_vtx"]); Z = ak.to_numpy(t["mc_targetZ"]); mod = ak.to_numpy(t["truth_vtx_module"])
    sel = ak.to_numpy(t["truth_is_fiducial"]) & (ak.to_numpy(t["mc_current"]) == 1) & (ak.to_numpy(t["mc_incoming"]) == 14) & (v[:, 2] > 5990) & (v[:, 2] < 8340)
    for z in np.unique(Z[sel]): m = sel & (Z == z); pool.setdefault(int(z), []).append(np.column_stack([v[m], mod[m]]))
pool = {z: np.concatenate(p) for z, p in pool.items()}; print("vertex pool (fiducial tracker) per Z:", {z: len(p) for z, p in pool.items()}, flush=True)

cols = {"pdg": "e/post/post.pdg", "px": "e/post/post.x", "py": "e/post/post.y", "pz": "e/post/post.z", "E": "e/post/post.t",
        "in_pdg": "e/in/in.pdg", "in_E": "e/in/in.t", "dyn": "e/dyn", "weight": "e/weight", "Z": "e/par/par.nucleus_p", "N": "e/par/par.nucleus_n"}
parts = []
for f in sorted(glob.glob(f"{a.nuwro_dir}/nuwro_*.root")):
    job = int(pathlib.Path(f).stem.split("_")[1]); arr = uproot.open(f)["treeout"].arrays(list(cols.values()), library="ak"); arr = {k: arr[v] for k, v in cols.items()}
    n = len(arr["dyn"]); px, py, pz = from_beam(arr["px"], arr["py"], arr["pz"])  # beam frame -> detector frame
    nu = arr["in_pdg"] == 14; Enu = ak.to_numpy(ak.fill_none(ak.firsts(arr["in_E"][nu]), np.nan))
    mu = arr["pdg"] == 13; assert ak.all(ak.sum(mu, axis=1) >= 1), f"{f}: events without a muon"
    lead = ak.argmax(ak.where(mu, arr["E"], -1.0), axis=1, keepdims=True)  # primary muon = most energetic mu- (rare DIS events carry an extra mu+mu- pair)
    mu4 = np.column_stack([ak.to_numpy(ak.fill_none(ak.flatten(c[lead]), np.nan)) for c in (px, py, pz, arr["E"])])  # plain float columns, no option type
    # Q2 and lepton-defined W: neutrino along the beam axis, detector frame
    bx, by, bz = from_beam(np.zeros(n), np.zeros(n), Enu); q = np.column_stack([bx - mu4[:, 0], by - mu4[:, 1], bz - mu4[:, 2], Enu - mu4[:, 3]])
    Q2 = q[:, 0] ** 2 + q[:, 1] ** 2 + q[:, 2] ** 2 - q[:, 3] ** 2; W = np.sqrt(np.clip(M_P ** 2 + 2 * M_P * q[:, 3] - Q2, 0, None))
    Z = ak.to_numpy(arr["Z"]).astype(np.int32); A = (Z + ak.to_numpy(arr["N"])).astype(np.int32)
    vtx = np.empty((n, 4)); mod = np.empty(n, np.int32)
    for z in np.unique(Z):
        m = Z == z; pick = pool[int(z)][rng.integers(0, len(pool[int(z)]), m.sum())]; vtx[m] = pick[:, :4]; mod[m] = pick[:, 4].astype(np.int32)
    parts.append(ak.Array({
        "mc_run": np.full(n, 900000 + job, np.int32), "mc_subrun": np.full(n, job, np.int32), "mc_nthEvtInFile": np.arange(n, dtype=np.int32), "eventID": (900000 + job) * 1e7 + np.arange(n, dtype=np.float64),
        "mc_vtx": vtx, "mc_targetZ": Z, "mc_targetA": A, "truth_targetID": np.zeros(n, np.int32), "truth_vtx_module": mod, "truth_is_fiducial": np.ones(n, bool),
        "mc_nFSPart": ak.to_numpy(ak.num(arr["pdg"])).astype(np.int32), "mc_FSPartPDG": ak.values_astype(arr["pdg"], np.int32), "mc_FSPartPx": px, "mc_FSPartPy": py, "mc_FSPartPz": pz, "mc_FSPartE": arr["E"],
        "mc_primFSLepton": mu4, "mc_incoming": np.full(n, 14, np.int32), "mc_incomingE": Enu, "mc_current": np.ones(n, np.int32),
        "mc_intType": np.array([DYN_TO_INTTYPE.get(int(d), 0) for d in ak.to_numpy(arr["dyn"])], np.int32), "mc_Q2": Q2, "mc_w": W,
        "nuwro_dyn": ak.to_numpy(arr["dyn"]).astype(np.int32), "nuwro_weight": ak.to_numpy(arr["weight"]), "nuwro_job": np.full(n, job, np.int32)}))
    print(f"  {pathlib.Path(f).name}: {n:,} events, Z={Z[0]} A={A[0]}", flush=True)
out = ak.concatenate(parts); ak.to_parquet(out, a.out); print(f"wrote {len(out):,} events -> {a.out}")
