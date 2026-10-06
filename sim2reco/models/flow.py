"""Conditional flow matching (linear/OT path, Gaussian source) for a fixed-size continuous target."""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class TimeEmbedding(nn.Module):
    def __init__(self, dim=64):
        super().__init__(); self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(1000.0) * torch.arange(half, device=t.device) / half)
        a = t[:, None] * freqs[None] * 2 * math.pi
        return torch.cat([a.sin(), a.cos()], -1)


class VelocityNet(nn.Module):
    def __init__(self, dim, cond_dim, hidden=512, n_layers=4, t_dim=64):
        super().__init__()
        self.temb = TimeEmbedding(t_dim)
        layers, d = [], dim + cond_dim + t_dim
        for _ in range(n_layers):
            layers += [nn.Linear(d, hidden), nn.SiLU()]; d = hidden
        self.body = nn.Sequential(*layers); self.out = nn.Linear(hidden, dim)

    def forward(self, x, t, cond):
        return self.out(self.body(torch.cat([x, cond, self.temb(t)], -1)))


class FlowMatcher(nn.Module):
    def __init__(self, dim, cond_dim, hidden=512, n_layers=4, sigma_min=1e-3):
        super().__init__()
        self.dim, self.sigma_min = dim, sigma_min
        self.v = VelocityNet(dim, cond_dim, hidden, n_layers)

    def loss(self, x1, cond, dim_mask=None):
        """dim_mask [n, dim] in {0,1}: masked dimensions are zeroed at the input and excluded from the loss, so the
        flow models only the unmasked part (used for energies that are exactly zero in an event)."""
        x0 = torch.randn_like(x1)
        t = torch.rand(len(x1), device=x1.device)
        xt = (1 - (1 - self.sigma_min) * t)[:, None] * x0 + t[:, None] * x1
        target = x1 - (1 - self.sigma_min) * x0
        if dim_mask is None: return ((self.v(xt, t, cond) - target) ** 2).mean(-1)
        err = ((self.v(xt * dim_mask, t, cond) - target) ** 2) * dim_mask
        return err.sum(-1) / dim_mask.sum(-1).clamp(min=1)

    @torch.no_grad()
    def sample(self, cond, n_steps=64, bound=20.0, dim_mask=None):
        """Midpoint (RK2) integration from t=0 to 1. The state is clamped to +-bound after each step: in
        ~3e-5 of events the learned velocity diverges late in the trajectory (|x| -> 1e36), which would
        otherwise yield NaN rows."""
        x = torch.randn(len(cond), self.dim, device=cond.device)
        m = 1.0 if dim_mask is None else dim_mask
        dt = 1.0 / n_steps
        for i in range(n_steps):
            t = torch.full((len(cond),), i * dt, device=cond.device)
            k1 = self.v(x * m, t, cond)
            k2 = self.v((x + 0.5 * dt * k1) * m, t + 0.5 * dt, cond)
            x = (x + dt * k2).clamp(-bound, bound)
        return torch.nan_to_num(x * m, nan=0.0, posinf=bound, neginf=-bound)
