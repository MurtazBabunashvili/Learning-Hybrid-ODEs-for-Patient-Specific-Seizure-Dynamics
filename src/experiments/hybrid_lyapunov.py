"""Experimental Lyapunov-constrained hybrid.

Same dynamics as hybrid.py:

    dx/dt = f_Epileptor(x; theta_i) + g_phi(x)

plus an isolated Lyapunov apparatus (matrix + hinge loss) that lives ONLY
here -- hybrid.py is untouched. The runner (main.py) adds extra_loss() to
the forecast objective whenever the built experiment provides it.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .hybrid import PureHybridODE


class LyapunovP(nn.Module):
    """P = L L^T + eps I (positive definite by construction)."""

    def __init__(self, dim: int = 5, eps: float = 1e-3):
        super().__init__()
        self.dim = int(dim)
        self.eps = float(eps)
        self.L = nn.Parameter(torch.eye(dim) * 0.1)

    def P(self) -> torch.Tensor:
        L = self.L
        return L @ L.T + self.eps * torch.eye(
            self.dim, device=L.device, dtype=L.dtype)

    def normalized(self) -> torch.Tensor:
        """Trace-normalized P: prevents the trivial P -> 0 solution."""
        P = self.P()
        return self.dim / P.trace() * P


class HybridLyapunovODE(PureHybridODE):
    """Hybrid field + Lyapunov hinge on interictal rollout steps.

    extra_loss(future_traj, kinds, pidx, lambda_l, margin):
    V(x) = (x-x*)^T P (x-x*) with x* = detached per-sample trajectory mean
    and trace-normalized P; penalizes Vdot > -margin on interictal samples.
    Per-step loop keeps per-sample theta exact; only runs when lambda_l > 0.
    """

    def __init__(self, init_H: torch.Tensor, mu: torch.Tensor, patient_ids,
                 residual_hidden: int = 128, residual_layers: int = 2,
                 residual_activation: str = "tanh", residual_skip: bool = False,
                 latent_dim: int = 5):
        super().__init__(init_H, mu, patient_ids,
                         residual_hidden=residual_hidden,
                         residual_layers=residual_layers,
                         residual_activation=residual_activation,
                         residual_skip=residual_skip)
        self.lyapunovP = LyapunovP(latent_dim)

    def extra_loss(self, future_traj: torch.Tensor, kinds: list[str] | None,
                   pidx: torch.Tensor, lambda_l: float,
                   margin: float = 0.0) -> torch.Tensor:
        if not (lambda_l > 0.0):
            return torch.zeros((), device=future_traj.device)
        theta = self.patient.batch(pidx)  # each [B]
        P = self.lyapunovP.normalized()
        x_star = future_traj.detach().mean(dim=1)  # [B, D]
        D = future_traj - x_star.unsqueeze(1)
        T = future_traj.shape[1]
        vd = []
        for k in range(T):
            Fk = self.field_with_theta(future_traj[:, k], theta)
            vd.append(2.0 * (D[:, k] @ P * Fk).sum(dim=-1))
        Vd = torch.stack(vd, dim=1)  # [B, T]
        if kinds is None:
            mask = torch.ones(Vd.shape[0], dtype=torch.bool,
                              device=future_traj.device)
        else:
            mask = torch.tensor([kk == "interictal" for kk in kinds],
                                device=future_traj.device)
        if not bool(mask.any()):
            return torch.zeros((), device=future_traj.device)
        return torch.relu(Vd[mask] + margin).mean()
