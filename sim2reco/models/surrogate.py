"""The M2 surrogate: set encoder -> Tier 0 heads, cardinality head, Tier 1 conditional flow matching."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import SetEncoder
from .flow import FlowMatcher
from .prongflow import ProngFlow

N_PRONG_CLASSES = 9  # 0..8
TIER1_DIM = 9  # MODEL_COLS of sim2reco.data.compact
PRONG_DIM = 10  # sim2reco.data.prongs_tf.PRONG_DIM
N_VTX_CLASSES = 9  # sim2reco.data.prongs_tf.VertexPlaneTable.N_CLASSES
PRONG_DIM_CAP = 8  # sim2reco.data.compact.N_PRONG_CAP
ZERO_COLS = [7, 8]  # Tier1Transform.ZERO_MODEL_COLS: log_recoil_nonvtx100, log_nonvtx_iso_blobs_E


class Surrogate(nn.Module):
    def __init__(self, d_model=128, n_heads=4, n_layers=4, flow_hidden=512, flow_layers=4, tier2=False,
                 prong_layers=3, zero_flags=False):
        super().__init__()
        self.enc = SetEncoder(d_model, n_heads, n_layers)
        self.tier0 = nn.Sequential(nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, 3))
        self.card = nn.Sequential(nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, N_PRONG_CLASSES))
        # zero flags: Bernoulli heads for "non-vertex energy within 100 mm is exactly zero" and "isolated-blob
        # energy is exactly zero" (conditional on reconstruction); the flow then models only the positive part
        # of those two columns, with the flags in its conditioning, and the sampler fills exact zeros.
        self.zero_flags = zero_flags
        if zero_flags:
            self.zero = nn.Sequential(nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, 2))
        self.register_buffer("zero_fill", torch.zeros(2))  # model-space value for E = 0 (set from the transform)
        self.zero_band = None  # [2, 2] model-space [lo, hi] of the dequantisation band; set from the transform at load/train
        self.flag_emb = nn.Linear(4 if zero_flags else 2, 32)
        self.n_emb = nn.Embedding(N_PRONG_CLASSES, 32)
        self.flow = FlowMatcher(TIER1_DIM, d_model + 64, flow_hidden, flow_layers)
        self.tier2 = tier2
        if tier2:
            # vertex-plane class head (unsnapped / snapped to nearest plane + delta) on z, flags and N
            # input: flow conditioning + the phase of the true z within the plane cycle (+ sin/cos of it)
            self.vtx = nn.Sequential(nn.Linear(d_model + 64 + 3, d_model), nn.SiLU(), nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, N_VTX_CLASSES))
            # prong set flow, conditioned on z, flags, N, and the Tier 1 vector; cross-attends particle tokens
            self.prong = ProngFlow(PRONG_DIM, d_model, d_model + 64 + TIER1_DIM, prong_layers)

    def encode(self, b, return_tokens=False):
        z, h = self.enc(b["cls"], b["mom"], b["mask"], b["ctx"])
        return (z, h) if return_tokens else z

    def full_flags(self, z, flags, zflags=None):
        """Flow-conditioning flags: (minos, charge) plus, for a zero-flag model, the two zero flags, sampled from
        the zero heads when not given (teacher forcing passes the true ones)."""
        if not self.zero_flags: return flags
        if flags.shape[1] == 4: return flags
        if zflags is None: zflags = (torch.rand(len(z), 2, device=z.device) < torch.sigmoid(self.zero(z))).float()
        return torch.cat([flags, zflags], -1)

    def dim_mask(self, flags):
        """[n, TIER1_DIM] mask: the zero-spike energy columns are switched off when their zero flag is set."""
        m = torch.ones(len(flags), TIER1_DIM, device=flags.device)
        if self.zero_flags: m[:, ZERO_COLS] = 1.0 - flags[:, 2:4]
        return m

    def flow_cond(self, z, flags, nprong):
        return torch.cat([z, self.flag_emb(flags), self.n_emb(nprong.clamp(0, N_PRONG_CLASSES - 1))], -1)

    def vtx_logits(self, cond, phase):
        ph = phase[:, None] * 3.14159265
        return self.vtx(torch.cat([cond, phase[:, None], ph.sin(), ph.cos()], -1))

    def losses(self, b):
        z, h = self.encode(b, return_tokens=True)
        t0 = b["tier0"]; reco = t0[:, 0] > 0; minos = reco & (t0[:, 1] > 0)
        logits = self.tier0(z)
        l_exist = F.binary_cross_entropy_with_logits(logits[:, 0], t0[:, 0])
        l_minos = F.binary_cross_entropy_with_logits(logits[reco, 1], t0[reco, 1]) if reco.any() else logits.sum() * 0
        l_charge = F.binary_cross_entropy_with_logits(logits[minos, 2], t0[minos, 2]) if minos.any() else logits.sum() * 0
        l_card = F.cross_entropy(self.card(z[reco]), b["nprong"][reco]) if reco.any() else logits.sum() * 0
        fl = reco & (b["valid"] > 0)   # exclude the rare corrupt tuple entries from the flow loss
        flags = self.full_flags(z[fl], t0[fl, 1:3], b["zflags"][fl]) if self.zero_flags else t0[fl, 1:3]
        cond = self.flow_cond(z[fl], flags, b["nprong"][fl])
        l_flow = self.flow.loss(b["x1"][fl], cond, self.dim_mask(flags) if self.zero_flags else None).mean() if fl.any() else logits.sum() * 0
        out = {"exist": l_exist, "minos": l_minos, "charge": l_charge, "card": l_card, "flow": l_flow}
        if self.zero_flags:
            out["zero"] = F.binary_cross_entropy_with_logits(self.zero(z[fl]), b["zflags"][fl]) if fl.any() else logits.sum() * 0
        if self.tier2:
            out["vtx"] = F.cross_entropy(self.vtx_logits(cond, b["vphase"][fl]), b["vclass"][fl]) if fl.any() else logits.sum() * 0
            pm = b["pmask"][fl]; has = pm.any(1)
            if has.any():
                pc = torch.cat([cond[has], b["x1"][fl][has]], -1)
                out["prong"] = self.prong.loss(b["prongs"][fl][has], pc, h[fl][has], b["mask"][fl][has], pm[has])
            else:
                out["prong"] = logits.sum() * 0
        return out

    @torch.no_grad()
    def predict_probs(self, b):
        z = self.encode(b)
        return torch.sigmoid(self.tier0(z)), F.softmax(self.card(z), -1), z

    @torch.no_grad()
    def sample(self, b, n_steps=64, teacher_flags=None, teacher_nprong=None, prongs=True):
        """Sample Tier 0 flags, N prongs and Tier 1 (model space). Optionally condition on given flags/N.
        prongs=False skips the prong-set flow (85% of the sampling cost) when only event-level outputs are needed."""
        p0, pn, z = self.predict_probs(b)
        u = torch.rand_like(p0)
        # The heads are conditional probabilities: p(reco exists | X), p(MINOS ok | reco), p(charge<0 | MINOS ok).
        # `exist` is a mask for the caller; minos/charge and the Tier 1 vector describe the event *if* it is
        # reconstructed and must not be zeroed by a False `exist` (that biased the closure test toward the
        # unmatched-muon mode).
        exist = u[:, 0] < p0[:, 0]
        minos = u[:, 1] < p0[:, 1]
        charge = minos & (u[:, 2] < p0[:, 2])
        flags = torch.stack([minos, charge], -1).float() if teacher_flags is None else teacher_flags
        flags = self.full_flags(z, flags)  # adds sampled zero flags for a zero-flag model
        nprong = torch.multinomial(pn, 1)[:, 0] if teacher_nprong is None else teacher_nprong
        cond = self.flow_cond(z, flags, nprong)
        if self.zero_flags:
            x1 = self.flow.sample(cond, n_steps, dim_mask=self.dim_mask(flags))
            zf = flags[:, 2:4] > 0
            if self.zero_band is not None:  # uniform over the dequantisation band, as the real zeros are in model space
                lo, hi = self.zero_band[:, 0][None, :], self.zero_band[:, 1][None, :]; fill = lo + torch.rand(len(z), 2, device=z.device) * (hi - lo)
            else: fill = self.zero_fill[None, :].expand(len(z), -1)
            x1[:, ZERO_COLS] = torch.where(zf, fill, x1[:, ZERO_COLS])
        else:
            x1 = self.flow.sample(cond, n_steps)
        out = {"exist": exist, "minos": flags[:, 0] > 0, "charge": flags[:, 1] > 0, "nprong": nprong, "x1": x1, "p0": p0, "pn": pn, "flags": flags}
        if self.zero_flags: out["zflags"] = flags[:, 2:4] > 0
        if self.tier2:
            pv = F.softmax(self.vtx_logits(cond, b["vphase"]), -1)
            out["vclass"] = torch.multinomial(pv, 1)[:, 0]; out["pv"] = pv
            h = self.encode(b, return_tokens=True)[1] if prongs else None
            N = int(min(nprong.max().item(), PRONG_DIM_CAP)) if len(nprong) else 0
            pmask = torch.arange(max(N, 1), device=z.device)[None, :] < nprong.clamp(max=PRONG_DIM_CAP)[:, None]
            out["pmask"] = pmask
            out["prongs"] = self.prong.sample(torch.cat([cond, x1], -1), h, b["mask"], pmask, n_steps) if (N > 0 and prongs) else torch.zeros(len(z), 1, PRONG_DIM, device=z.device)
        return out
