"""Neural-only dynamics: dx/dt = g_phi(x). No Epileptor."""
from __future__ import annotations

import torch
import torch.nn as nn

from .common import ObservationOperator, PatientParameters, ResidualMLP


class NeuralField(nn.Module):
    """Neural-only vector field shared by the pure-neural experiment."""

    def __init__(self, hidden: int = 128, n_layers: int = 2,
                 activation: str = "tanh", skip: bool = False):
        super().__init__()
        self.net = ResidualMLP(hidden=hidden, n_layers=n_layers,
                               activation=activation, skip=skip)

    def forward(self, x: torch.Tensor):
        return self.net(x)


class NeuralODE(nn.Module):
    """dx/dt = g_phi(x). Patient table kept so the shared runner
    (estimate_boundary_state / euler_rollout) works unchanged; theta is
    unused by the field."""

    def __init__(self, init_H: torch.Tensor, mu: torch.Tensor, patient_ids,
                 hidden: int = 128, n_layers: int = 2,
                 activation: str = "tanh", skip: bool = False):
        super().__init__()
        self.mech = None
        self.residual = NeuralField(hidden=hidden, n_layers=n_layers,
                                    activation=activation, skip=skip)
        self.obs = ObservationOperator(init_H, mu)
        self.patient = PatientParameters(patient_ids)

    def field(self, x: torch.Tensor, pidx: torch.Tensor):
        return self.residual(x)

    def field_with_theta(self, x: torch.Tensor, theta: dict):
        return self.residual(x)

    def observe(self, x):
        return self.obs(x)
