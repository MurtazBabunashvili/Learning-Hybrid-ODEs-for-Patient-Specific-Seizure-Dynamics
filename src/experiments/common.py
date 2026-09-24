"""Shared components identical across all experiments.

Moved verbatim from main.py: only former module-global constants are now
explicit arguments (same defaults), so numerics are bit-identical.
"""
from __future__ import annotations
from contextlib import contextmanager

import torch
import torch.nn as nn


class PatientParameters(nn.Module):
    """
    theta_i = [x0_i, I1_i, I2_i]

    Same initialization used by the project.
    """

    def __init__(self, patient_ids: list[str]):
        super().__init__()
        self.patient_ids = list(patient_ids)
        n = len(patient_ids)

        self.x0 = nn.Parameter(torch.full((n,), -1.6))
        self.I1 = nn.Parameter(torch.full((n,), 3.1))
        self.I2 = nn.Parameter(torch.full((n,), 0.45))

    def batch(self, pidx: torch.Tensor):
        return {
            "x0": self.x0[pidx],
            "I1": self.I1[pidx],
            "I2": self.I2[pidx],
        }


class ObservationOperator(nn.Module):
    """
    y = mu + Hx

    H is only 22x5 = 110 parameters.
    No neural decoder.

    Columns are normalized to unit Euclidean norm. This removes the trivial
    column-scale ambiguity without adding a learned decoder.
    """

    def __init__(self, init_H: torch.Tensor, mu: torch.Tensor,
                 n_channels: int = 22, state_dim: int = 5):
        super().__init__()
        if init_H.shape != (n_channels, state_dim):
            raise ValueError(init_H.shape)
        self.H_raw = nn.Parameter(init_H.clone())
        self.register_buffer("mu", mu.clone())

    def matrix(self):
        return self.H_raw / (
            self.H_raw.norm(dim=0, keepdim=True) + 1e-8
        )

    def forward(self, x):
        # x [...,5] -> y [...,22]
        return self.mu + x @ self.matrix().T


class _ResidualBlock(nn.Module):
    """x + act(Linear(x)). Identity at init when the Linear is small."""

    def __init__(self, width: int, act: nn.Module):
        super().__init__()
        self.lin = nn.Linear(width, width)
        self.act = act

    def forward(self, x):
        return x + self.act(self.lin(x))


class ResidualMLP(nn.Module):
    """
    ONLY neural component.

        g_phi: R^5 -> R^5

    Defaults (hidden=128, n_layers=2, activation="tanh", skip=False)
    reproduce the original 5->128->128->5 Tanh MLP exactly.
    The deep variant (e.g. hidden=256, n_layers=4, silu, skip=True) keeps a
    smooth vector field with good gradients: SiLU is C-infinity and the
    skip connections bound each block near identity.
    """

    def __init__(self, hidden: int = 128, n_layers: int = 2,
                 activation: str = "tanh", skip: bool = False):
        super().__init__()
        act_cls = {"tanh": nn.Tanh, "silu": nn.SiLU}[activation]
        layers: list[nn.Module] = [nn.Linear(5, hidden), act_cls()]
        for _ in range(n_layers - 1):
            if skip:
                layers.append(_ResidualBlock(hidden, act_cls()))
            else:
                layers += [nn.Linear(hidden, hidden), act_cls()]
        layers.append(nn.Linear(hidden, 5))
        self.net = nn.Sequential(*layers)

        # Start close to the mechanistic prior, but not identically zero.
        nn.init.normal_(
            self.net[-1].weight,
            mean=0.0,
            std=1e-3,
        )

        nn.init.zeros_(
            self.net[-1].bias
        )

    def forward(self, x):
        return self.net(x)


def euler_rollout(
    model,
    x0: torch.Tensor,
    pidx: torch.Tensor,
    steps: int,
    dt: float,
    theta_override: dict | None = None,
):
    """
    Returns [B, steps, 5].

    Fully batched: one vectorized field evaluation per Euler step.
    """
    x = x0
    out = [x]

    if theta_override is None:
        theta = model.patient.batch(pidx)
    else:
        theta = theta_override

    for _ in range(1, steps):
        dx = model.field_with_theta(x, theta)
        x = x + dt * dx
        out.append(x)

    return torch.stack(out, dim=1)


def ridge_initial_state(model, y_last: torch.Tensor,
                        ridge: float = 1e-4, clamp: float = 2.5,
                        state_dim: int = 5):
    """
    Initial guess only.

        x = argmin ||Hx - (y_last-mu)||^2 + alpha ||x||^2

    This is not a learned encoder.
    """
    H = model.obs.matrix().detach()
    target = y_last - model.obs.mu

    A = H.T @ H + ridge * torch.eye(
        state_dim,
        device=H.device,
        dtype=H.dtype,
    )
    b = H.T @ target.T

    x = torch.linalg.solve(A, b).T
    # Ridge can extrapolate far outside the Epileptor operating range on
    # saturated (clipped ±8) prefixes. Project into the trust region.
    return x.clamp(-clamp, clamp)


@contextmanager
def frozen_parameters(module: nn.Module):
    old = [p.requires_grad for p in module.parameters()]
    try:
        for p in module.parameters():
            p.requires_grad_(False)
        yield
    finally:
        for p, flag in zip(module.parameters(), old):
            p.requires_grad_(flag)


def estimate_boundary_state(
    model,
    y_prefix: torch.Tensor,
    pidx: torch.Tensor,
    prefix_steps: int,
    dt: float,
    inner_steps: int = 3,
    state_lr: float = 2e-2,
    ridge: float = 1e-4,
    clamp: float = 2.5,
    state_dim: int = 5,
):
    """
    Solve

        min_x0 mean_t ||H Phi_t(x0) - y_prefix(t)||^2

    using ONLY the prefix.

    Model parameters are frozen during this inner solve. Only the 5D start-of-window
    state is optimized. The function returns the state at the forecast
    boundary, detached so the outer forecast loss cannot secretly train the
    state estimate using future targets.
    """
    x0 = ridge_initial_state(
        model,
        y_prefix[..., -1],
        ridge=ridge,
        clamp=clamp,
        state_dim=state_dim,
    )

    x0 = x0.detach()

    with frozen_parameters(model):
        for step_i in range(inner_steps):
            x0 = x0.detach().requires_grad_(True)

            prefix_traj = euler_rollout(
                model,
                x0,
                pidx,
                prefix_steps,
                dt,
            )

            y_hat_prefix = model.observe(
                prefix_traj
            ).permute(0, 2, 1)

            loss = torch.mean(
                (y_hat_prefix - y_prefix) ** 2
            )

            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(
                    f"Non-finite inner prefix loss at GD step {step_i} "
                    f"(x0 maxabs={float(x0.abs().max()):.3f})."
                )

            grad_x = torch.autograd.grad(
                loss,
                x0,
                retain_graph=False,
                create_graph=False,
            )[0]

            if not bool(torch.isfinite(grad_x).all()):
                raise FloatingPointError(
                    f"Non-finite inner gradient at GD step {step_i}."
                )

            x0 = (
                x0 - state_lr * grad_x
            ).detach().clamp(-clamp, clamp)

    # x0 is the state at the START of the prefix.  The forecast must
    # start at the END of the prefix, so return Phi(x0), not x0 itself.
    with torch.no_grad():
        prefix_traj = euler_rollout(
            model,
            x0,
            pidx,
            prefix_steps,
            dt,
        )
        x_boundary = prefix_traj[:, -1].detach()

    return x_boundary


def forecast_loss(
    model,
    future_traj: torch.Tensor,
    y_future: torch.Tensor,
    lambda_residual: float,
    dt: float,
):
    """
    Data loss:
        MSE(y_hat, y_future)

    Residual integral:
        dt * sum_t ||g_phi(x_t)||^2
    averaged across the batch.

    Models without a residual (mechanistic) contribute 0 regularization.
    """
    y_hat = model.observe(
        future_traj
    ).permute(0, 2, 1)

    data_loss = torch.mean(
        (y_hat - y_future) ** 2
    )

    if (
        lambda_residual > 0
        and getattr(model, "residual", None) is not None
    ):
        B, T, D = future_traj.shape

        residual_out = model.residual(
            future_traj.reshape(B * T, D)
        )

        residual_integral_per_sample = (
            dt * residual_out.pow(2).sum(dim=-1).sum(dim=-1)
        )

        residual_integral = (
            residual_integral_per_sample.mean()
        )
    else:
        residual_integral = torch.zeros(
            (),
            device=y_hat.device,
        )

    total = (
        data_loss
        + lambda_residual * residual_integral
    )

    return total, data_loss, residual_integral


@torch.no_grad()
def persistence_mse(y_prefix, y_future):
    """
    Simple causal baseline:
        y_hat_future(t) = y(last prefix sample)
    """
    last = y_prefix[..., -1:].expand_as(y_future)
    return torch.mean(
        (last - y_future) ** 2
    ).item()


@torch.no_grad()
def residual_ratio(
    model,
    trajectory: torch.Tensor,
    pidx: torch.Tensor,
):
    """Mean ||g|| / (||f_mech|| + eps); (0, mech, 0) if residual is None."""
    B, T, D = trajectory.shape
    x = trajectory.reshape(B * T, D)

    idx = pidx[:, None].expand(B, T).reshape(-1)
    theta = model.patient.batch(idx)

    if getattr(model, "mech", None) is None:
        # Neural-only experiment: no mechanistic component to compare to.
        g = model.residual(x)
        nr = torch.linalg.vector_norm(g, dim=-1).mean()
        return float(nr), float("nan"), float("nan")
    f = model.mech(x, theta)
    if getattr(model, "residual", None) is None:
        nm = torch.linalg.vector_norm(f, dim=-1).mean()
        return 0.0, float(nm), 0.0
    g = model.residual(x)

    nr = torch.linalg.vector_norm(g, dim=-1).mean()
    nm = torch.linalg.vector_norm(f, dim=-1).mean()

    return float(nr), float(nm), float(nr / (nm + 1e-8))
