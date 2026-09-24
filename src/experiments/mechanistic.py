"""Pure mechanistic dynamics: dx/dt = f_Epileptor(x; theta_i). No neural network."""
from __future__ import annotations

import torch
import torch.nn as nn

from .common import ObservationOperator, PatientParameters


class ReducedEpileptor(nn.Module):
    """Exact reduced 5D field used by the project."""

    def __init__(self):
        super().__init__()
        self.tau0 = 2857.0
        self.tau2 = 10.0
        self.y0 = 1.0

    @staticmethod
    def f1(x1):
        return 3.0 * x1**3 - 2.0 * x1**2

    @staticmethod
    def f2(x2):
        return 6.0 * (x2 + 0.25) * torch.sigmoid(
            20.0 * (x2 + 0.25)
        )

    @staticmethod
    def g(x1):
        return torch.sigmoid(
            10.0 * (x1 + 0.5)
        )

    def forward(self, x: torch.Tensor, theta: dict):
        x1, y1, z, x2, y2 = x.unbind(dim=-1)

        x0 = theta["x0"]
        I1 = theta["I1"]
        I2 = theta["I2"]

        dx1 = y1 - self.f1(x1) - z + I1
        dy1 = self.y0 - 5.0 * x1 ** 2 - y1
        dz = (4.0 * (x1 - x0) - z) / self.tau0
        dx2 = (
            -y2
            + x2
            - x2 ** 3
            + I2
            + 0.002 * self.g(x1)
            - 0.3 * (z - 3.5)
        )
        dy2 = (-y2 + self.f2(x2)) / self.tau2

        return torch.stack(
            [dx1, dy1, dz, dx2, dy2],
            dim=-1,
        )


class MechanisticODE(nn.Module):
    """dx/dt = f_mech(x; theta_i). residual is None by construction."""

    def __init__(self, init_H: torch.Tensor, mu: torch.Tensor, patient_ids):
        super().__init__()
        self.mech = ReducedEpileptor()
        self.residual = None
        self.obs = ObservationOperator(init_H, mu)
        self.patient = PatientParameters(patient_ids)

    def field(self, x: torch.Tensor, pidx: torch.Tensor):
        theta = self.patient.batch(pidx)
        return self.mech(x, theta)

    def field_with_theta(self, x: torch.Tensor, theta: dict):
        return self.mech(x, theta)

    def observe(self, x):
        return self.obs(x)
