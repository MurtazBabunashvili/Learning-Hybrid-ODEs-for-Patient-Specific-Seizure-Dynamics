from __future__ import annotations

"""
PURE MATHEMATICAL HYBRID ODE FORECASTER
=======================================

No neural encoder.
No neural decoder.
No hypernetwork.
No Lyapunov machinery.
No full-window reconstruction objective.
No hidden fallback solver.

Model
-----
    dx/dt = f_Epileptor(x; theta_i) + g_phi(x)

    y(t) = mu + H x(t)

where
    x in R^5       : reduced Epileptor state
    y in R^22      : observed EEG
    theta_i        : patient-specific (x0, I1, I2)
    g_phi          : ONLY neural component, a 5D residual vector field
    H in R^(22x5) : linear observation operator

Initial-state estimation
-------------------------
For each forecast window, the 5D ODE initial state is estimated ONLY from
its 10-second prefix by solving

    x_0* = argmin_x  (1/Tp) sum_k || mu + H Phi_tk(x) - y_prefix(tk) ||^2

using a few batched gradient-descent iterations in 5D.

Then the forecast starts from the estimated boundary state

    x_b = Phi_10(x_0*)
    y_hat_future = mu + H Phi_t(x_b),  t in [0, 20s].

The future is NEVER used to estimate the state.

Residual penalty
----------------
The regularizer is the literal time integral

    L_R = integral ||g_phi(x(t))||^2 dt

approximated on the Euler grid by

    L_R ~= dt * sum_k ||g_phi(x_k)||^2.

This script intentionally keeps the implementation mathematically explicit.
It reuses the project's existing CHB-MIT split/window/cache pipeline so the
same 22-channel data are used as in the previous runs.
"""

from contextlib import contextmanager
from pathlib import Path
from collections import Counter
import os
import pickle
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# -----------------------------------------------------------------------------
# 0. PROJECT ROOT
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


# -----------------------------------------------------------------------------
# 1. EXPERIMENT CONFIG (config/*.yaml controls every parameter)
# -----------------------------------------------------------------------------
def _load_config() -> dict:
    import yaml
    default = ROOT / "config" / "hybrid.yaml"
    # CLI: python main.py [config.yaml] [--resume] [--epochs N]
    # (argv entries that are not yaml files, e.g. pytest args, are ignored
    # for the config path but still scanned for flags)
    override = None
    for a in sys.argv[1:]:
        if a.lower().endswith((".yaml", ".yml")):
            override = Path(a)
    with open(default, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if override is not None:
        with open(override, encoding="utf-8") as f:
            other = yaml.safe_load(f)
        cfg.update(other)
    return cfg


CFG = _load_config()

SEED = int(CFG.get("seed", 0))
_device_cfg = str(CFG.get("device", "auto"))
DEVICE = torch.device(
    "cuda" if (torch.cuda.is_available() and _device_cfg in ("auto", "cuda"))
    else "cpu"
)

_data = CFG.get("data", {})
N_CHANNELS = int(_data.get("n_channels", 22))
STATE_DIM = int(_data.get("state_dim", 5))

TARGET_FS = float(_data.get("target_fs", 64.0))
ODE_FS = float(_data.get("ode_fs", 16.0))
DT = 1.0 / ODE_FS

WINDOW_SEC = float(_data.get("window_sec", 30.0))
PREFIX_SEC = float(_data.get("prefix_sec", 10.0))
HORIZON_SEC = float(_data.get("horizon_sec", 20.0))

PREFIX_STEPS = int(round(PREFIX_SEC * ODE_FS))       # 160
HORIZON_STEPS = int(round(HORIZON_SEC * ODE_FS))     # 320

_model = CFG.get("model", {})
RESIDUAL_HIDDEN = int(_model.get("residual_hidden", 128))

_train = CFG.get("train", {})
BATCH_SIZE = int(_train.get("batch_size", 128))
# Sampler: "quota" (legacy) | "diverse" (hierarchy+caps) | "targeted"
# (episode-first 50k construction with documented strides).
SAMPLER = str(_train.get("sampler", "quota"))
# Micro-batch for VRAM: effective batch stays BATCH_SIZE; gradients are
# exactly accumulated over chunks of MICRO_BATCH (0/null = no chunking).
MICRO_BATCH = int(_train.get("micro_batch", 0) or 0)
N_TRAIN = int(_train.get("n_train", 1500))
N_VAL = int(_train.get("n_val", 120))

EPOCHS = int(_train.get("epochs", 5))
LR = float(_train.get("lr", 3e-4))

# Pure state-estimation inner solve.  It uses PREFIX ONLY.
_se = CFG.get("state_estimation", {})
STATE_INNER_STEPS = int(_se.get("inner_steps", 3))
STATE_LR = float(_se.get("lr", 2e-2))
STATE_RIDGE = float(_se.get("ridge", 1e-4))
# Trust region for the shooting state: the cubic Epileptor terms make
# explicit Euler at dt=1/16 unstable outside roughly ±2.5 (measured: ridge
# solutions up to |x|=14.7 on saturated ictal prefixes blow up even at ±4).
# Clamping keeps the inner solve inside the model's valid domain; it does
# not touch the learned dynamics.
STATE_CLAMP = float(_se.get("clamp", 2.5))

# Your mathematical residual regularizer.
_loss = CFG.get("loss", {})
LAMBDA_RESIDUAL = float(_loss.get("lambda_residual", 1e-3))
LAMBDA_LYAPUNOV = float(_loss.get("lambda_lyapunov", 0.0))
LYAPUNOV_MARGIN = float(_loss.get("lyapunov_margin", 0.0))

EXPERIMENT_NAME = str(CFG.get("experiment", {}).get("name", "hybrid"))

NUM_WORKERS = 0
PIN_MEMORY = DEVICE.type == "cuda"

_split = CFG.get("split", {})
TRAIN_PATIENTS = list(_split.get("train_patients", ["chb01"]))
VAL_PATIENTS = list(_split.get("val_patients", ["chb04"]))
TEST_PATIENTS = list(_split.get("test_patients", ["chb05"]))

QUOTAS = dict(_split.get("quotas", {
    "interictal": 0.4,
    "preictal_far": 0.2,
    "preictal_near": 0.2,
    "ictal": 0.1,
    "postictal": 0.1,
}))

_paths = CFG.get("paths", {})
CKPT_PATH = ROOT / _paths.get("ckpt", "artifacts/checkpoints/pure_hybrid_ode.pt")
LOG_PATH = ROOT / _paths.get("log", "artifacts/metrics/pure_hybrid_ode.log")

# Estimator settings bundle (prefix-only shooting, same values as config).
EST = dict(
    prefix_steps=PREFIX_STEPS,
    dt=DT,
    inner_steps=STATE_INNER_STEPS,
    state_lr=STATE_LR,
    ridge=STATE_RIDGE,
    clamp=STATE_CLAMP,
    state_dim=STATE_DIM,
)


# -----------------------------------------------------------------------------
# 2. REPRODUCIBILITY
# -----------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


set_seed(SEED)


# -----------------------------------------------------------------------------
# 3. LOGGING
# -----------------------------------------------------------------------------
class Logger:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = open(path, "w", encoding="utf-8")

    def log(self, msg: str) -> None:
        s = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(s, flush=True)
        self.file.write(s + "\n")
        self.file.flush()

    def close(self) -> None:
        self.file.close()


logger = Logger(LOG_PATH) if os.environ.get("MAIN_SILENT") != "1" else None
if logger is None:
    # Test/analysis imports: no file logging side effects.
    class _NullLogger:
        def log(self, msg: str) -> None:
            pass

        def close(self) -> None:
            pass

    logger = _NullLogger()


# -----------------------------------------------------------------------------
# 4. DATA: SAME PROJECT WINDOWS/CACHE PIPELINE
# -----------------------------------------------------------------------------
def load_data():
    split_path = ROOT / "artifacts" / "splits" / "splits.pkl"
    if not split_path.exists():
        raise FileNotFoundError(
            f"Missing {split_path}. Build the project splits first."
        )

    with open(split_path, "rb") as f:
        splits = pickle.load(f)

    from src.evaluation.splits import (
        sample_diverse, sample_onset_relative, sample_targeted)
    from src.data.loader import get_batches_cached

    train_pool = splits["train"]
    val_pool = splits["val"]

    if SAMPLER == "diverse":
        train_windows, drep = sample_diverse(
            train_pool, TRAIN_PATIENTS, N_TRAIN, SEED)
        logger.log(f"diverse-train: total={drep['total']} "
                   f"phases={drep['phases']} episodes={drep['episodes']} "
                   f"recordings={drep['recordings']}")
        logger.log(f"diverse-train per-patient={drep['per_patient']}")
    elif SAMPLER == "targeted":
        import json as _json
        manifest_recs = [
            _json.loads(l) for l in
            open(ROOT / "artifacts" / "manifests" / "chbmit.jsonl")]
        train_windows, trep = sample_targeted(
            manifest_recs, TRAIN_PATIENTS, N_TRAIN, SEED)
        # ONE consistent population: phases counted post-trim == total.
        logger.log(f"targeted-train: total={trep['total']} "
                   f"phases={trep['phases']} episodes={trep['episodes']} "
                   f"recordings={trep['recordings']} "
                   f"bad_spans={trep['skipped_bad_spans']}")
        logger.log(f"targeted-train per-patient={trep['per_patient']}")
        logger.log(f"targeted-train preictal_bands={trep['preictal_bands']}")
    else:
        train_windows = sample_onset_relative(
            train_pool,
            TRAIN_PATIENTS,
            N_TRAIN,
            QUOTAS,
            SEED,
        )

    val_windows = sample_onset_relative(
        val_pool,
        VAL_PATIENTS,
        N_VAL,
        QUOTAS,
        SEED + 1,
    )

    logger.log(
        f"train_windows={len(train_windows)} "
        f"{dict(Counter(w['kind'] for w in train_windows))}"
    )
    logger.log(
        f"val_windows={len(val_windows)} "
        f"{dict(Counter(w['kind'] for w in val_windows))}"
    )

    # Lazy sharded views (peak RAM flat); metadata lists stay in memory.
    train_items = get_batches_cached(
        train_windows,
        TARGET_FS,
        ODE_FS,
        logger,
        "train",
        sharded=True,
    )
    val_items = get_batches_cached(
        val_windows,
        TARGET_FS,
        ODE_FS,
        logger,
        "val",
        sharded=True,
    )

    all_patients = list(dict.fromkeys(
        TRAIN_PATIENTS + VAL_PATIENTS + TEST_PATIENTS
    ))
    patient_to_idx = {
        p: i for i, p in enumerate(all_patients)
    }

    train_pidx = torch.tensor(
        [patient_to_idx[train_windows[i]["patient_id"]]
         for i in range(len(train_items))],
        dtype=torch.long,
    )
    val_pidx = torch.tensor(
        [patient_to_idx[val_windows[i]["patient_id"]]
         for i in range(len(val_items))],
        dtype=torch.long,
    )

    # Streaming channel mean + covariance over the train view (no stacking):
    # mu (22,) and a 22x22 accumulator only.
    mu_sum = torch.zeros(N_CHANNELS, dtype=torch.float64)
    mu_n = 0
    for i in range(len(train_items)):
        y = train_items[i]["y"].to(torch.float64)
        mu_sum += y.sum(dim=(0, 1))
        mu_n += y.shape[0] * y.shape[1]
    mu = (mu_sum / mu_n).float()
    # PCA/SVD initialization ONLY for H (streamed SVD sample: immune to
    # singular/repeated-eigenvalue covariances from duplicated or
    # zero-filled channels, where eigh fails to converge).
    _svd_buf, _svd_max = [], 256
    _stride = max(1, len(train_items) // _svd_max)
    for i in range(0, len(train_items), _stride):
        _svd_buf.append(train_items[i]["y"].to(torch.float64) - mu.double().unsqueeze(-1))
        if len(_svd_buf) >= _svd_max:
            break
    _flat = torch.cat([w.reshape(-1, N_CHANNELS) for w in _svd_buf], dim=0)
    del _svd_buf
    _, _, _Vh = torch.linalg.svd(_flat.float(), full_matrices=False)
    del _flat
    init_H = _Vh[:STATE_DIM].T.contiguous()

    logger.log(f"train_windows={len(train_items)} val_windows={len(val_items)} "
               f"(streamed, peak RAM flat)")
    logger.log(
        f"forecast: prefix={PREFIX_STEPS} samples ({PREFIX_SEC}s), "
        f"horizon={HORIZON_STEPS} samples ({HORIZON_SEC}s)"
    )

    return (
        train_items,
        train_pidx,
        val_items,
        val_pidx,
        all_patients,
        patient_to_idx,
        mu,
        eigvecs,
    )


# -----------------------------------------------------------------------------
# 5. EXPERIMENTS (scientific models live in src/experiments/)
# -----------------------------------------------------------------------------
from src.experiments.common import (
    euler_rollout,
    estimate_boundary_state,
    forecast_loss,
    persistence_mse,
    residual_ratio,
)
from src.experiments.mechanistic import MechanisticODE
from src.experiments.pure_neural import NeuralODE
from src.experiments.hybrid import PureHybridODE
from src.experiments.hybrid_lyapunov import HybridLyapunovODE


def build_experiment(cfg: dict, init_H: torch.Tensor, mu: torch.Tensor,
                     patient_ids: list[str]):
    """Select the scientific model. main.py stays a runner; math lives in
    src/experiments/{mechanistic,pure_neural,hybrid,hybrid_lyapunov}.py."""
    name = str(cfg.get("experiment", {}).get("name", "hybrid"))
    model_cfg = cfg.get("model", {})
    hidden = int(model_cfg.get("residual_hidden", 128))
    layers = int(model_cfg.get("residual_layers", 2))
    act = str(model_cfg.get("residual_activation", "tanh"))
    skip = bool(model_cfg.get("residual_skip", False))
    res_kwargs = dict(residual_hidden=hidden, residual_layers=layers,
                      residual_activation=act, residual_skip=skip)
    if name == "mechanistic":
        return MechanisticODE(init_H, mu, patient_ids)
    if name == "pure_neural":
        return NeuralODE(init_H, mu, patient_ids, hidden=hidden,
                         n_layers=layers, activation=act, skip=skip)
    if name == "hybrid_lyapunov":
        return HybridLyapunovODE(
            init_H, mu, patient_ids,
            latent_dim=int(model_cfg.get("latent_dim", 5)), **res_kwargs)
    if name == "hybrid":
        return PureHybridODE(init_H, mu, patient_ids, **res_kwargs)
    raise ValueError(f"unknown experiment.name={name!r}")


# -----------------------------------------------------------------------------
# 6. MODEL EQUATIONS: see src/experiments/mechanistic.py (ReducedEpileptor)
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# 7. NEURAL RESIDUAL: see src/experiments/common.py (ResidualMLP)
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# 8. OBSERVATION: see src/experiments/common.py (ObservationOperator)
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# 9. HYBRID MODEL: see src/experiments/hybrid.py (PureHybridODE)
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# 10-13. ROLLOUT + STATE ESTIMATION: see src/experiments/common.py
#   (euler_rollout, ridge_initial_state, frozen_parameters,
#    estimate_boundary_state)
# -----------------------------------------------------------------------------


# (estimate_boundary_state moved to src/experiments/common.py)


# (forecast_loss, persistence_mse, residual_ratio live in
#  src/experiments/common.py)


# -----------------------------------------------------------------------------
# 16. TRAIN ONE EPOCH
# -----------------------------------------------------------------------------
def train_epoch(
    model,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
):
    model.train()

    total_sum = 0.0
    data_sum = 0.0
    reg_sum = 0.0
    n = 0

    for batch_y, pidx in loader:
        batch_y = batch_y.to(
            DEVICE,
            non_blocking=True,
        )
        pidx = pidx.to(
            DEVICE,
            non_blocking=True,
        )

        # Micro-batching: split a large batch into chunks that fit VRAM,
        # accumulate exact full-batch gradients (micro loss scaled by m/B).
        # Effective batch size is unchanged; only peak memory drops.
        Bfull = batch_y.shape[0]
        mbs = max(1, int(MICRO_BATCH)) if MICRO_BATCH else Bfull
        optimizer.zero_grad(set_to_none=True)
        mb_losses = []
        for s in range(0, Bfull, mbs):
            e = min(s + mbs, Bfull)
            scale = (e - s) / Bfull
            my = batch_y[s:e]
            mp = pidx[s:e]
            prefix = my[..., :PREFIX_STEPS]
            future = my[
                ...,
                PREFIX_STEPS:PREFIX_STEPS + HORIZON_STEPS,
            ]

            # IMPORTANT: x0 is inferred from PREFIX ONLY.
            x0 = estimate_boundary_state(
                model,
                prefix,
                mp,
                **EST,
            )

            # Future loss cannot backprop through the state estimator.
            future_traj = euler_rollout(
                model,
                x0,
                mp,
                HORIZON_STEPS,
                DT,
            )

            loss, data_loss, reg = forecast_loss(
                model,
                future_traj,
                future,
                LAMBDA_RESIDUAL,
                DT,
            )
            # Experimental extension (hybrid_lyapunov only; 0 otherwise).
            extra = model.extra_loss(
                future_traj,
                None,
                mp,
                LAMBDA_LYAPUNOV,
                LYAPUNOV_MARGIN,
            ) if hasattr(model, "extra_loss") else 0.0
            loss = (loss + extra) * scale

            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(
                    f"Non-finite training loss: {float(loss)}"
                )

            loss.backward()
            mb_losses.append((float(loss.detach()) / max(scale, 1e-12), e - s,
                              float(data_loss.detach()), float(reg.detach())))
            del x0, future_traj, prefix, future
        # No silent clipping and no silent bad-batch skipping.
        for p in model.parameters():
            if p.grad is not None and not bool(
                torch.isfinite(p.grad).all()
            ):
                raise FloatingPointError(
                    "Non-finite gradient encountered."
                )

        optimizer.step()

        bs = batch_y.shape[0]
        total_sum += sum(v * m for v, m, _, _ in mb_losses)
        data_sum += sum(d * m for _, m, d, _ in mb_losses)
        reg_sum += sum(r * m for _, m, _, r in mb_losses)
        n += bs
        del batch_y, pidx

    return {
        "total": total_sum / n,
        "data": data_sum / n,
        "residual_integral": reg_sum / n,
    }


# -----------------------------------------------------------------------------
# 17. VALIDATION
# -----------------------------------------------------------------------------
def validate(
    model,
    loader: DataLoader,
):
    model.eval()

    # State estimation needs gradients wrt x0, so temporarily leave model in
    # eval mode but call the estimator, which internally enables input grads.
    mse_sum = 0.0
    n = 0
    last_traj = None
    last_pidx = None

    for batch_y, pidx in loader:
        batch_y = batch_y.to(
            DEVICE,
            non_blocking=True,
        )
        pidx = pidx.to(
            DEVICE,
            non_blocking=True,
        )

        prefix = batch_y[..., :PREFIX_STEPS]
        future = batch_y[
            ...,
            PREFIX_STEPS:PREFIX_STEPS + HORIZON_STEPS,
        ]

        # Prefix-only state estimation contains its own input-gradient solve.
        x0 = estimate_boundary_state(
            model,
            prefix,
            pidx,
            **EST,
        )

        traj = euler_rollout(
            model,
            x0,
            pidx,
            HORIZON_STEPS,
            DT,
        )

        y_hat = model.observe(traj).permute(0, 2, 1)
        mse = torch.mean(
            (y_hat - future) ** 2,
            dim=(1, 2),
        )

        mse_sum += float(mse.sum())
        n += batch_y.shape[0]

        last_traj = traj
        last_pidx = pidx

    return mse_sum / n, last_traj, last_pidx


# -----------------------------------------------------------------------------
# 18. CHECKPOINT
# -----------------------------------------------------------------------------
def save_checkpoint(
    model,
    patient_to_idx: dict,
    best_val: float,
):
    CKPT_PATH.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model": model.state_dict(),
            "patient_to_idx": patient_to_idx,
            "best_val_mse": best_val,
            "config": {
                "target_fs": TARGET_FS,
                "ode_fs": ODE_FS,
                "prefix_sec": PREFIX_SEC,
                "horizon_sec": HORIZON_SEC,
                "state_dim": STATE_DIM,
                "batch_size": BATCH_SIZE,
                "lr": LR,
                "lambda_residual": LAMBDA_RESIDUAL,
                "state_inner_steps": STATE_INNER_STEPS,
                "state_lr": STATE_LR,
            },
        },
        CKPT_PATH,
    )


# -----------------------------------------------------------------------------
# 19. MAIN
# -----------------------------------------------------------------------------
def _flag(name: str) -> bool:
    return any(a == name for a in sys.argv[1:])

def _flag_value(name: str) -> str | None:
    args = sys.argv[1:]
    for i, a in enumerate(args):
        if a == name and i + 1 < len(args):
            return args[i + 1]
    return None


def main():
    global EPOCHS
    if (e := _flag_value("--epochs")) is not None:
        EPOCHS = int(e)
        logger.log(f"CLI override: epochs={EPOCHS}")
    resume = _flag("--resume")
    # --resume-from PATH: load weights/best from another checkpoint file
    # while saving to this run's own CKPT_PATH (never touches the source).
    _rf = _flag_value("--resume-from")
    RESUME_PATH = Path(_rf) if _rf is not None else CKPT_PATH
    logger.log(f"device={DEVICE}")
    if DEVICE.type == "cuda":
        logger.log(
            f"gpu={torch.cuda.get_device_name(0)} "
            f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB"
        )
    logger.log(f"torch={torch.__version__}")

    (
        train_items,
        train_pidx,
        val_items,
        val_pidx,
        patient_ids,
        patient_to_idx,
        mu,
        eigvecs,
    ) = load_data()

    # Fixed empirical mean + PCA basis from load_data (streamed, no stacking).
    # H remains trainable, but this avoids an arbitrary random observation
    # map at step zero. The PCA is not the state representation used later.
    init_H = eigvecs[:, -STATE_DIM:]

    logger.log(
        "observation model: y = mu + Hx, H=22x5 (110 params), "
        "PCA-initialized and trainable"
    )

    model = build_experiment(
        CFG,
        init_H=init_H,
        mu=mu,
        patient_ids=patient_ids,
    ).to(DEVICE)

    logger.log(
        f"experiment={EXPERIMENT_NAME}"
    )
    logger.log(
        f"trainable_params={sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )
    _mcfg = CFG.get("model", {})
    logger.log(
        f"mechanistic=5D ReducedEpileptor + residual="
        f"{_mcfg.get('residual_hidden', 128)}h/"
        f"{_mcfg.get('residual_layers', 2)}L-"
        f"{_mcfg.get('residual_activation', 'tanh')}"
        f"{'+skip' if _mcfg.get('residual_skip', False) else ''} MLP"
    )
    logger.log(
        f"residual_scale=1.0 lambda_R={LAMBDA_RESIDUAL}"
    )
    logger.log(
        f"forecast={PREFIX_SEC}s prefix -> {HORIZON_SEC}s horizon"
    )
    logger.log(
        "initial state: prefix-only nonlinear least-squares in R^5; "
        "NO encoder"
    )
    logger.log(
        "observation: fixed train mean + learned linear H; NO decoder"
    )

    class WindowDataset(torch.utils.data.Dataset):
        """Lazy (y, pidx) pairs over a ShardedWindows view (no stacking)."""

        def __init__(self, items, pidx):
            self.items = items
            self.pidx = pidx

        def __len__(self):
            return len(self.items)

        def __getitem__(self, i):
            return self.items[i]["y"], self.pidx[i]

    train_loader = DataLoader(
        WindowDataset(train_items, train_pidx),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=False,
    )

    val_loader = DataLoader(
        WindowDataset(val_items, val_pidx),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=False,
    )

    logger.log(
        f"batch_size={BATCH_SIZE} train_batches={len(train_loader)} val_batches={len(val_loader)}"
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
    )

    start_epoch = 0
    if resume and RESUME_PATH.exists():
        ckpt = torch.load(RESUME_PATH, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model"])
        logger.log(f"resumed weights from {RESUME_PATH} "
                   f"(ckpt best_val={ckpt.get('best_val_mse', float('nan')):.6f})")

    # Baseline immediately, using exactly the same future target.
    # Streamed over the val view (no stacking).
    with torch.no_grad():
        _ps, _pn = 0.0, 0
        for i in range(len(val_items)):
            vy = val_items[i]["y"]
            _ps += persistence_mse(vy[..., :PREFIX_STEPS],
                                   vy[..., PREFIX_STEPS:PREFIX_STEPS + HORIZON_STEPS])
            _pn += 1
        p_mse = _ps / max(_pn, 1)
    logger.log(f"validation persistence_mse={p_mse:.6f}")

    best_val = float("inf")
    if resume and RESUME_PATH.exists():
        try:
            best_val = float(torch.load(
                RESUME_PATH, map_location="cpu",
                weights_only=False).get("best_val_mse", float("inf")))
        except Exception:
            pass

    for epoch in range(start_epoch, start_epoch + EPOCHS):
        t0 = time.time()

        train_stats = train_epoch(
            model,
            train_loader,
            optimizer,
        )

        val_mse, last_traj, last_pidx = validate(
            model,
            val_loader,
        )

        elapsed = time.time() - t0

        logger.log(
            f"epoch {epoch:02d} | "
            f"train_total={train_stats['total']:.6f} | "
            f"train_forecast={train_stats['data']:.6f} | "
            f"train_R_integral={train_stats['residual_integral']:.6f} | "
            f"val={val_mse:.6f} | "
            f"time={elapsed:.1f}s"
        )

        if last_traj is not None and last_pidx is not None:
            nr, nm, ratio = residual_ratio(
                model,
                last_traj,
                last_pidx,
            )
            logger.log(
                f"dynamics | mean||g||={nr:.6f} "
                f"mean||f_mech||={nm:.6f} "
                f"ratio={ratio:.6f}"
            )

        if val_mse < best_val:
            best_val = val_mse
            save_checkpoint(
                model,
                patient_to_idx,
                best_val,
            )
            logger.log(
                f"NEW BEST -> val={best_val:.6f}"
            )

    logger.log("training complete")
    logger.log(f"best_val_mse={best_val:.6f}")
    logger.log(f"persistence_mse={p_mse:.6f}")
    logger.log(f"checkpoint={CKPT_PATH}")

    # Print learned train-patient parameters only once.
    model.eval()
    pp = model.patient
    logger.log("patient parameters:")
    for i, pid in enumerate(pp.patient_ids):
        logger.log(
            f"  {pid}: "
            f"x0={float(pp.x0[i]):.5f}, "
            f"I1={float(pp.I1[i]):.5f}, "
            f"I2={float(pp.I2[i]):.5f}"
        )


if __name__ == "__main__":
    try:
        main()
    finally:
        try:
            logger.close()
        except Exception:
            pass
