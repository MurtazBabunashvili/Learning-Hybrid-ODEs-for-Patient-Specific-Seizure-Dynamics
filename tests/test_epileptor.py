"""Field is finite with expected dimensions (main.py ReducedEpileptor)."""
import os

os.environ["MAIN_SILENT"] = "1"

import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import main as M


def test_field_finite_batched():
    from src.experiments.mechanistic import ReducedEpileptor
    mech = ReducedEpileptor()
    B = 4
    x = torch.randn(B, 5)
    theta = {"x0": torch.full((B,), -1.6),
             "I1": torch.full((B,), 3.1),
             "I2": torch.full((B,), 0.45)}
    dx = mech(x, theta)
    assert tuple(dx.shape) == (B, 5)
    assert bool(torch.isfinite(dx).all())


def test_residual_small_init():
    from src.experiments.common import ResidualMLP
    res = ResidualMLP(hidden=128)
    x = torch.randn(8, 5)
    g = res(x)
    assert tuple(g.shape) == (8, 5)
    assert float(g.abs().mean()) < 0.1


def test_residual_variants_equivalent_default():
    import torch.nn as nn
    from src.experiments.common import ResidualMLP
    res_new = ResidualMLP(hidden=128, n_layers=2, activation="tanh", skip=False)
    assert [type(m) for m in res_new.net] == [
        nn.Linear, nn.Tanh, nn.Linear, nn.Tanh, nn.Linear]
    assert res_new.net[0].in_features == 5 and res_new.net[-1].out_features == 5


def test_residual_deep_silu_skip():
    from src.experiments.common import ResidualMLP
    res = ResidualMLP(hidden=256, n_layers=4, activation="silu", skip=True)
    x = torch.randn(6, 5, requires_grad=True)
    g = res(x)
    assert tuple(g.shape) == (6, 5)
    assert bool(torch.isfinite(g).all())
    g.sum().backward()
    assert x.grad is not None and bool(torch.isfinite(x.grad).all())
    n = sum(p.numel() for p in res.parameters())
    print(f"deep residual params={n}")
    assert n > 200000
