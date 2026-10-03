"""Bayesian last layers on a frozen surrogate (epistemic uncertainty without retraining).

For each head the final nn.Linear is given a Gaussian posterior over its weights:
  * flows (velocity nets): the flow-matching objective is a linear regression of the target velocity on the
    last-layer features h, so the posterior is the exact Bayesian-linear-regression posterior
    Sigma_o = (lambda I + H^T H / s_o^2)^-1 per output o, mean = the trained weights. This is the optimum the
    variational Bayesian last layer (VBLL) converges to with a fixed backbone.
  * classification heads: Laplace approximation, precision_o = lambda I + sum_n p_n(1-p_n) h_n h_n^T
    (binary) or with softmax probabilities per class.
Epistemic quantities: logit variance h^T Sigma h for heads; for a flow sample, the first-order variance of the
final state under a constant weight perturbation along the trajectory, g_o^T Sigma_o g_o with g_o = sum_s dt h_s.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class FeatureTap:
    """Forward hook capturing the input of a Linear layer."""
    def __init__(self, layer: nn.Linear):
        self.h = None; self.handle = layer.register_forward_hook(self._hook)
    def _hook(self, mod, inp, out): self.h = inp[0].detach()
    def remove(self): self.handle.remove()


def last_linear(seq):
    for m in reversed(list(seq.modules())):
        if isinstance(m, nn.Linear): return m
    raise ValueError("no Linear layer")


class GaussianLastLayer:
    """Posterior over the weights of one Linear(d -> k): shared feature covariance per output."""
    def __init__(self, d: int, k: int, device):
        self.d, self.k = d, k
        self.HtH = torch.zeros(d + 1, d + 1, dtype=torch.float64, device=device)   # with bias column
        self.Hty = torch.zeros(d + 1, k, dtype=torch.float64, device=device)
        self.yty = torch.zeros(k, dtype=torch.float64, device=device); self.n = 0
        self.Sigma = None; self.noise = None; self.lam = None

    def accumulate(self, h, y, w=None):
        """h [n,d] features, y [n,k] regression targets (or for Laplace: pass y=None and w=p(1-p) weights)."""
        hb = torch.cat([h.double(), torch.ones(len(h), 1, dtype=torch.float64, device=h.device)], 1)
        if w is None:
            self.HtH += hb.T @ hb; self.Hty += hb.T @ y.double(); self.yty += (y.double() ** 2).sum(0); self.n += len(h)
        else:  # Laplace: weighted outer products (w may be [n] or [n,k]; use mean over classes if [n,k])
            ww = w.double() if w.dim() == 1 else w.double().mean(1)
            self.HtH += (hb * ww[:, None]).T @ hb; self.n += len(h)

    def fit_regression(self, lam_grid=(1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0, 10000.0)):
        """Noise variance from the residual of the ML fit; lambda by maximising the Gaussian evidence."""
        I = torch.eye(self.d + 1, dtype=torch.float64, device=self.HtH.device)
        w_ml = torch.linalg.solve(self.HtH + 1e-6 * I, self.Hty)
        rss = self.yty - 2 * (w_ml * self.Hty).sum(0) + torch.einsum("ik,ij,jk->k", w_ml, self.HtH, w_ml)
        self.noise = (rss / self.n).clamp_min(1e-8)                              # [k]
        best = None
        for lam in lam_grid:
            # evidence per output (up to constants): -0.5[ n log s^2 + log det(A) - (d+1) log lam + rss_post/s^2 + lam ||w||^2 ]
            A = lam * I[None] + self.HtH[None] / self.noise[:, None, None]       # [k,d+1,d+1]
            L = torch.linalg.cholesky(A); logdet = 2 * torch.log(torch.diagonal(L, dim1=1, dim2=2)).sum(1)
            w_post = torch.cholesky_solve((self.Hty / self.noise).T[..., None], L)[..., 0]   # [k,d+1]
            rss_post = self.yty - 2 * (w_post.T * self.Hty).sum(0) + torch.einsum("ki,ij,kj->k", w_post, self.HtH, w_post)
            ev = -0.5 * (self.n * torch.log(self.noise) + logdet - (self.d + 1) * torch.log(torch.tensor(lam, device=A.device)) + rss_post / self.noise + lam * (w_post ** 2).sum(1))
            tot = float(ev.sum())
            if best is None or tot > best[0]: best = (tot, lam, A)
        self.lam = best[1]; self.Sigma = torch.linalg.inv(best[2]).float()      # [k,d+1,d+1]
        return self

    def fit_laplace(self, lam=1.0):
        I = torch.eye(self.d + 1, dtype=torch.float64, device=self.HtH.device)
        self.lam = lam; self.Sigma = torch.linalg.inv(lam * I + self.HtH).float()[None].expand(self.k, -1, -1).contiguous()
        return self

    def var(self, h):
        """Predictive variance of each output for features h [n,d] -> [n,k]."""
        hb = torch.cat([h, torch.ones(len(h), 1, device=h.device)], 1)
        return torch.einsum("ni,kij,nj->nk", hb, self.Sigma, hb)

    def sample_delta(self, generator=None):
        """One weight perturbation dW [k,d], db [k] drawn from the posterior (zero-mean)."""
        L = torch.linalg.cholesky(self.Sigma.double() + 1e-10 * torch.eye(self.d + 1, dtype=torch.float64, device=self.Sigma.device)[None]).float()
        eps = torch.randn(self.k, self.d + 1, device=self.Sigma.device, generator=generator)
        d = torch.einsum("kij,kj->ki", L, eps)
        return d[:, :-1], d[:, -1]

    def state(self):
        return {"Sigma": self.Sigma.cpu(), "noise": None if self.noise is None else self.noise.cpu(), "lam": self.lam, "n": self.n, "d": self.d, "k": self.k}

    @classmethod
    def from_state(cls, s, device):
        g = cls(s["d"], s["k"], device); g.Sigma = s["Sigma"].to(device); g.noise = None if s["noise"] is None else s["noise"].to(device); g.lam = s["lam"]; g.n = s["n"]; return g


HEADS = {"tier0": ("tier0", "laplace"), "card": ("card", "laplace"), "vtx": ("vtx", "laplace"), "flow": ("flow.v.out", "regression"), "prong": ("prong.v.out", "regression")}


def head_layer(model, path):
    mod = model
    for p in path.split("."): mod = getattr(mod, p)
    return last_linear(mod) if not isinstance(mod, nn.Linear) else mod
