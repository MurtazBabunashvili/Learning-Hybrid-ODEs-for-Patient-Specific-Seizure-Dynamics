"""10s prefix -> 20s future: finite rollout, prefix-only encoder blindness."""
import os

os.environ["MAIN_SILENT"] = "1"

import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import main as M  # config constants only
from src.experiments.common import estimate_boundary_state, euler_rollout, ridge_initial_state
from src.experiments.hybrid import PureHybridODE


def _model():
    n = 4
    H = torch.linalg.qr(torch.randn(22, 5))[0]
    mu = torch.zeros(22)
    pids = [f"p{i}" for i in range(n)]
    return PureHybridODE(init_H=H, mu=mu, patient_ids=pids), pids


def _forecast(model, prefix, pidx):
    """Same explicit path as train_epoch: ridge+shoot -> rollout -> decode."""
    x0 = estimate_boundary_state(model, prefix, pidx, M.PREFIX_STEPS, M.DT)
    traj = euler_rollout(model, x0, pidx, M.HORIZON_STEPS, M.DT)
    y_hat = model.observe(traj).permute(0, 2, 1)
    return y_hat, x0, traj


def test_forecast_shapes_finite():
    model, pids = _model()
    B = 4
    prefix = torch.randn(B, 22, M.PREFIX_STEPS)
    pidx = torch.tensor([0, 1, 2, 3])
    y_hat, x0, traj = _forecast(model, prefix, pidx)
    assert tuple(y_hat.shape) == (B, 22, M.HORIZON_STEPS)
    assert tuple(x0.shape) == (B, 5)
    assert tuple(traj.shape) == (B, M.HORIZON_STEPS, 5)
    assert bool(torch.isfinite(y_hat).all())


def test_encoder_blind_to_future():
    # Determinism of the explicit forecast path (no grad needed).
    torch.manual_seed(0)
    model, _ = _model()
    model.eval()
    prefix = torch.randn(2, 22, M.PREFIX_STEPS)
    pidx = torch.tensor([0, 1])
    a, _, _ = _forecast(model, prefix, pidx)
    b, _, _ = _forecast(model, prefix, pidx)
    assert torch.equal(a, b)


def test_boundary_state_in_trust_region():
    model, _ = _model()
    prefix = 8.0 * torch.randn(4, 22, M.PREFIX_STEPS)  # saturated-scale input
    pidx = torch.tensor([0, 1, 2, 3])
    x0 = ridge_initial_state(model, prefix[..., -1])
    assert bool(torch.isfinite(x0).all())
    assert float(x0.abs().max()) <= M.STATE_CLAMP + 1e-6
