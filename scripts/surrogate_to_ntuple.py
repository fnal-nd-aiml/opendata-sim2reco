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
    ap.add_argument("--epi-draws", type=int, default=16, help="posterior weight draws for per-event epistemic uncertainties (0 = off); needs bayes_last.pt in MODEL_DIR")
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
    loader = lambda: DataLoader(ds, batch_size=2048, collate_fn=collate, num_workers=4)

    def generate(seed, prongs=True):
        """One full generation pass; a fixed seed gives the same base noise, so repeated passes differ only through the weights.
        prongs=False skips the prong-set flow (the epistemic draws use only event-level outputs and the vertex class)."""
        torch.manual_seed(seed); S, PR, PM, VC, P0 = [], [], [], [], []
        for b in loader():
            s = model.sample(to_dev(b, "cuda"), a.steps, prongs=prongs)
            S.append(torch.cat([s["exist"][:, None].float(), s["minos"][:, None].float(), s["charge"][:, None].float(), s["nprong"][:, None].float(), s["x1"]], 1).cpu())
            P0.append(s["p0"].cpu())
            if tier2: PR.append(s["prongs"].cpu()); PM.append(s["pmask"].cpu()); VC.append(s["vclass"].cpu())
        return torch.cat(S).numpy(), PR, PM, VC, torch.cat(P0).numpy()

    S, PR, PM, VC, P0 = generate(a.seed)
    # ---- epistemic uncertainties: analytic per-variable variance (model space) + K common-noise posterior draws (physical units)
    epi = None
    bl_path = pathlib.Path(a.model_dir) / "bayes_last.pt"
    if a.epi_draws > 0 and bl_path.exists():
        from sim2reco.models.bayes_last import FeatureTap, GaussianLastLayer, HEADS, head_layer, heads_for
        post = {k: GaussianLastLayer.from_state(v, "cuda") for k, v in torch.load(bl_path, map_location="cpu", weights_only=False).items()}
        layers = {k: head_layer(model, p) for k, (p, _) in heads_for(model).items()}; taps = {k: FeatureTap(l) for k, l in layers.items()}
        # analytic: logit variances of the three flags, mean multiplicity-logit variance, and the event-flow flag along the nominal trajectory
        A_exist, A_card, A_flow = [], [], []
        with torch.no_grad():
            torch.manual_seed(a.seed)
            for b in loader():
                b = to_dev(b, "cuda"); z, _ = model.enc(b["cls"], b["mom"], b["mask"], b["ctx"]); lg = model.tier0(z); A_exist.append(post["tier0"].var(taps["tier0"].h).cpu())
                pn = torch.softmax(model.card(z), -1); A_card.append(post["card"].var(taps["card"].h).mean(1).cpu())
                u = torch.rand_like(lg); flags = torch.stack([u[:, 1] < torch.sigmoid(lg)[:, 1], (u[:, 1] < torch.sigmoid(lg)[:, 1]) & (u[:, 2] < torch.sigmoid(lg)[:, 2])], -1).float()
                cond = model.flow_cond(z, model.full_flags(z, flags), torch.multinomial(pn, 1)[:, 0]); x = torch.randn(len(z), model.flow.dim, device="cuda"); dt = 1.0 / a.steps; g = torch.zeros(len(z), layers["flow"].in_features, device="cuda")
                for i in range(a.steps):
                    t = torch.full((len(z),), i * dt, device="cuda"); k1 = model.flow.v(x, t, cond); k2 = model.flow.v(x + 0.5 * dt * k1, t + 0.5 * dt, cond); g += dt * taps["flow"].h; x = (x + dt * k2).clamp(-20, 20)
                A_flow.append(post["flow"].var(g).cpu())
        epi = {"logit_var": torch.cat(A_exist).numpy(), "card_var": torch.cat(A_card).numpy(), "flow_var": torch.cat(A_flow).numpy()}
        # K draws with common random numbers: perturb the output layers of flag, multiplicity and event flow (and prong flow if present)
        W0 = {k: (layers[k].weight.data.clone(), layers[k].bias.data.clone()) for k in layers}
        gen = torch.Generator(device="cuda"); gen.manual_seed(a.seed + 1000); draws = []
        for k_ in range(a.epi_draws):
            for k in ("tier0", "card", "flow") + (("prong",) if tier2 else ()) + (("zero",) if model.zero_flags else ()):
                dW, db = post[k].sample_delta(gen); layers[k].weight.data = W0[k][0] + dW; layers[k].bias.data = W0[k][1] + db
            draws.append(generate(a.seed, prongs=False))
        for k in layers: layers[k].weight.data, layers[k].bias.data = W0[k]
        epi["draws"] = draws
        print(f"epistemic uncertainties from {a.epi_draws} posterior draws (common noise)", flush=True)
    exist = S[:, 0] > 0
    y = tf.inverse(S[exist, 4:], d["mu_true"][exist], d["ctx"][exist])
    if tier2:  # snap the vertex z to a plane when the class head says so
        vc = torch.cat(VC).numpy()[exist]
        zs = tf.planes.z_for_class(vc, d["ctx"][exist, 2], y[:, 5]); y[:, 5] = np.where(vc > 0, zs, y[:, 5])
    minos, charge = S[exist, 1] > 0, S[exist, 2] > 0
    out = {k: truth[k][exist] for k in PASSTHROUGH + [c for c in truth.fields if c.startswith(("nuwro_", "ref_"))]}  # generator-specific extras pass through
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
    out["surrogate_p_reco_exists"] = P0[exist, 0].astype(np.float64)
    if epi is not None:
        # analytic (model space): std of the flag logits, mean multiplicity-logit variance, event-flow flag (sum of 9 variances)
        out["surrogate_epi_logit_reco_exists"] = np.sqrt(epi["logit_var"][exist, 0]); out["surrogate_epi_logit_minos"] = np.sqrt(epi["logit_var"][exist, 1]); out["surrogate_epi_logit_charge"] = np.sqrt(epi["logit_var"][exist, 2])
        out["surrogate_epi_multiplicity_logits"] = np.sqrt(epi["card_var"][exist]); out["surrogate_epi_flow_flag"] = epi["flow_var"][exist].sum(1)
        # K-draw std in physical units for the written event-level variables, and of the probabilities
        Ys = []; Ps = []; Ns = []
        for Sk, PRk, PMk, VCk, P0k in epi["draws"]:
            yk = tf.inverse(Sk[exist, 4:], d["mu_true"][exist], d["ctx"][exist])
            if tier2:
                vck = torch.cat(VCk).numpy()[exist]; zk = tf.planes.z_for_class(vck, d["ctx"][exist, 2], yk[:, 5]); yk[:, 5] = np.where(vck > 0, zk, yk[:, 5])
            Ys.append(yk); Ps.append(P0k[exist]); Ns.append(Sk[exist, 3])
        Ys = np.array(Ys); sd = Ys.std(0)
        for j, c in enumerate("xyz"): out[f"surrogate_epi_muon_P{c}"] = sd[:, j]
        out["surrogate_epi_muon_P"] = np.linalg.norm(Ys[:, :, :3], axis=2).std(0)
        for j, c in enumerate("xyz"): out[f"surrogate_epi_vtx_{c}"] = sd[:, 3 + j]
        out["surrogate_epi_recoil_E"] = sd[:, 6]; out["surrogate_epi_recoil_nonvtx100"] = sd[:, 9]; out["surrogate_epi_nonvtx_iso_blobs_energy"] = sd[:, 10]
        Ps = np.array(Ps); out["surrogate_epi_p_reco_exists"] = Ps[:, :, 0].std(0); out["surrogate_epi_p_minos"] = Ps[:, :, 1].std(0); out["surrogate_epi_p_charge_neg"] = Ps[:, :, 2].std(0)
        out["surrogate_epi_n_prongs"] = np.array(Ns).std(0); out["surrogate_epi_draws"] = np.full(exist.sum(), a.epi_draws, dtype=np.int32)
    write_ntuple(a.out, out)
    print(f"{len(truth)} truth events -> {exist.sum()} reconstructed ({exist.mean()*100:.1f}%) -> {a.out}")


if __name__ == "__main__":
    main()
