"""Hybrid dynamics: dx/dt = f_Epileptor(x; theta_i) + g_phi(x).

This is the current production model (moved verbatim from main.py).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .common import ObservationOperator, PatientParameters, ResidualMLP
from .mechanistic import ReducedEpileptor


class PureHybridODE(nn.Module):
    def __init__(self, init_H: torch.Tensor, mu: torch.Tensor, patient_ids,
                 residual_hidden: int = 128, residual_layers: int = 2,
                 residual_activation: str = "tanh", residual_skip: bool = False):
        super().__init__()
        self.mech = ReducedEpileptor()
        self.residual = ResidualMLP(hidden=residual_hidden,
                                    n_layers=residual_layers,
                                    activation=residual_activation,
                                    skip=residual_skip)
        self.obs = ObservationOperator(init_H, mu)
        self.patient = PatientParameters(patient_ids)

    def field(self, x: torch.Tensor, pidx: torch.Tensor):
        theta = self.patient.batch(pidx)
        return self.mech(x, theta) + self.residual(x)

    def field_with_theta(self, x: torch.Tensor, theta: dict):
        return self.mech(x, theta) + self.residual(x)

    def observe(self, x):
        return self.obs(x)
