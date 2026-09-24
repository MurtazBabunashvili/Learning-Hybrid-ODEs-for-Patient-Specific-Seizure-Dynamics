"""Five publication figures for the hybrid forecast experiment.

1. Learning curves (train/val forecast MSE vs epoch, parsed from log).
2. True vs predicted 40 s rollout (prefix + autonomous forecast, val examples).
3. MSE by forecast-horizon bin (error accumulation along the rollout).
4. ||g||, ||f_mech||, ratio vs epoch (parsed from log dynamics lines).
5. Baseline comparison bars (persistence / mechanistic / neural / hybrid).

Usage: python scripts/visualize.py [--config config/hybrid.yaml] [--n-horizon 40]
Saves PNGs to artifacts/figures/ and prints the underlying numbers.
"""
from __future__ import annotations
import sys
import re
import argparse
import pickle
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data.loader import get_batches_cached
from src.evaluation.splits import sample_onset_relative
from src.experiments.common import (
    euler_rollout, estimate_boundary_state,
)
from src.experiments.hybrid import PureHybridODE

FIGDIR = ROOT / "artifacts" / "figures"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Tee:
    def __init__(self):
        pass

    def log(self, msg: str) -> None:
        print(msg, flush=True)


def parse_log(path: Path):
    epochs, train, val, g, f, ratio = [], [], [], [], [], []
    for line in open(path):
        m = re.search(r"epoch (\d+) \| train_total=[\d.]+ \| train_forecast=([\d.]+) \| "
                      r"train_R_integral=[\d.]+ \| val=([\d.]+)", line)
        if m:
            epochs.append(int(m.group(1)))
            train.append(float(m.group(2)))
            val.append(float(m.group(3)))
        m2 = re.search(r"mean\|\|g\|\|=([\d.]+)\s+mean\|\|f_mech\|\|=([\d.]+)\s+"
                       r"\|\|g\|\|/\|\|f_mech\|\|=([\d.]+)", line)
        if m2:
            g.append(float(m2.group(1)))
            f.append(float(m2.group(2)))
            ratio.append(float(m2.group(3)))
    return epochs, train, val, g, f, ratio


def load_model_and_val(cfg_name: str, ckpt_name: str, n_val: int):
    import yaml
    cfg = yaml.safe_load(open(ROOT / "config" / f"{cfg_name}.yaml"))
    ckpt = torch.load(ROOT / "artifacts" / "checkpoints" / ckpt_name,
                      map_location=DEVICE, weights_only=False)
    p2i = ckpt["patient_to_idx"]
    cohort = [p for p, _ in sorted(p2i.items(), key=lambda kv: kv[1])]
    data = cfg["data"]
    model = PureHybridODE(
        init_H=torch.zeros(data["n_channels"], data["state_dim"]),
        mu=torch.zeros(data["n_channels"]), patient_ids=cohort).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    splits = pickle.load(open(ROOT / "artifacts" / "splits" / "splits.pkl", "rb"))
    q = cfg["split"]["quotas"] if "quotas" in cfg.get("split", {}) else None
    quotas = q or {"interictal": 0.4, "preictal_far": 0.2, "preictal_near": 0.2,
                   "ictal": 0.1, "postictal": 0.1}
    wins = sample_onset_relative(splits["val"], cfg["split"]["val_patients"],
                                 n_val, quotas, int(cfg.get("seed", 0)) + 1)
    # NOTE: tag MUST be "val" (identical call as training) or the cache
    # lookup misses by filename and reloads full EDFs.
    batches = get_batches_cached(wins, float(data["target_fs"]),
                                 float(data["ode_fs"]), Tee(), "val",
                                 sharded=True)
    fs = float(data["ode_fs"])
    pre = int(round(float(data["prefix_sec"]) * fs))
    hor = int(round(float(data["horizon_sec"]) * fs))
    return model, p2i, batches, wins, fs, pre, hor, cfg


def forecast_one(model, p2i, b, pre, hor, fs):
    y = b["y"].unsqueeze(0).to(DEVICE)
    pidx = torch.tensor([p2i[b["patient_id"]]]).to(DEVICE)
    prefix = y[:, :, :pre]
    fut = y[:, :, pre:pre + hor]
    x0 = estimate_boundary_state(model, prefix, pidx, pre, 1.0 / fs)
    traj = euler_rollout(model, x0, pidx, hor, 1.0 / fs)
    y_hat = model.observe(traj).permute(0, 2, 1)
    return (y[0].cpu().numpy(), fut[0].cpu().numpy(), y_hat[0].detach().cpu().numpy(),
            b["kind"], b["recording_id"], b["start_sec"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="hybrid")
    ap.add_argument("--ckpt", default="hybrid.pt")
    ap.add_argument("--log", default="hybrid.log")
    ap.add_argument("--n-horizon", type=int, default=40)
    args = ap.parse_args()
    FIGDIR.mkdir(parents=True, exist_ok=True)

    # ---- Fig 1+4 data ----
    epochs, train, val, g, f, ratio = parse_log(
        ROOT / "artifacts" / "metrics" / args.log)
    print(f"parsed {len(epochs)} epochs from {args.log}", flush=True)

    fig, ax = plt.subplots()
    ax.plot(epochs, train, label="train forecast MSE")
    ax.plot(epochs, val, label="val forecast MSE")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE")
    ax.set_title("Learning curves (20s prefix -> 40s forecast)")
    ax.legend()
    fig.savefig(FIGDIR / "fig1_learning_curves.png", dpi=120)
    print("fig1 val: start=%.4f end=%.4f" % (val[0], val[-1]), flush=True)

    if g:
        fig, ax = plt.subplots()
        ax.plot(g, label="mean ||g||")
        ax.plot(f, label="mean ||f_mech||")
        ax.set_xlabel("epoch")
        ax.set_ylabel("norm")
        ax.set_title("Mechanistic vs residual magnitude")
        ax.legend()
        fig.savefig(FIGDIR / "fig4_dynamics.png", dpi=120)
        fig, ax = plt.subplots()
        ax.plot(ratio)
        ax.set_xlabel("epoch")
        ax.set_ylabel("||g|| / ||f_mech||")
        ax.set_title("Residual fraction")
        fig.savefig(FIGDIR / "fig4_ratio.png", dpi=120)
        print("fig4 ratio: start=%.4f end=%.4f" % (ratio[0], ratio[-1]), flush=True)

    # ---- model + val windows ----
    model, p2i, batches, wins, fs, pre, hor, cfg = load_model_and_val(
        args.config, args.ckpt, 120)

    # ---- Fig 2: true vs predicted rollouts (first of each kind) ----
    picks = []
    seen = {}
    for b in batches:
        if b["kind"] not in seen:
            seen[b["kind"]] = b
    picks = [seen[k] for k in ["ictal", "preictal", "interictal"] if k in seen]
    chs = [0, 5, 10, 15]
    fig, axes = plt.subplots(len(picks), 1, figsize=(12, 3 * len(picks)),
                             sharex=True)
    if len(picks) == 1:
        axes = [axes]
    full_true, full_pred = [], []
    for ax, b in zip(axes, picks):
        y, fut, yh, kind, rec, st = forecast_one(model, p2i, b, pre, hor, fs)
        full_true.append((y, kind))
        full_pred.append(yh)
        t_pre = np.arange(pre) / fs
        t_fut = pre / fs + np.arange(hor) / fs
        for c in chs:
            ax.plot(t_pre, y[c, :pre], color="k", lw=0.8, alpha=0.7)
            ax.plot(t_fut, y[c, pre:pre + hor], color="k", lw=1.2)
            ax.plot(t_fut, yh[c], color="r", lw=1.0, alpha=0.9)
        ax.axvline(pre / fs, color="b", ls="--", label="forecast start")
        ax.set_title(f"{rec}@{st:.0f}s {kind} (black=true, red=pred, ch {chs})")
        ax.set_xlabel("s")
    axes[0].legend()
    fig.savefig(FIGDIR / "fig2_rollouts.png", dpi=120)
    print(f"fig2: {len(picks)} examples", flush=True)

    # ---- Fig 3: MSE by horizon bin ----
    bins = [(0, 5), (5, 10), (10, 20), (20, 30), (30, 40)]
    acc = {b: [] for b in bins}
    cum_pairs = []  # (yh, fut) reused for Fig 6
    for b in batches[:args.n_horizon]:
        y, fut, yh, kind, rec, st = forecast_one(model, p2i, b, pre, hor, fs)
        cum_pairs.append((yh, fut))
        for lo, hi in bins:
            i0, i1 = int(lo * fs), int(hi * fs)
            acc[(lo, hi)].append(float(np.mean((yh[:, i0:i1] - fut[:, i0:i1]) ** 2)))
    labels = [f"{lo}-{hi}s" for lo, hi in bins]
    means = [float(np.mean(acc[b])) for b in bins]
    fig, ax = plt.subplots()
    ax.plot(labels, means, marker="o")
    ax.set_ylabel("MSE")
    ax.set_title("Forecast MSE by horizon bin")
    fig.savefig(FIGDIR / "fig3_horizon.png", dpi=120)
    for lab, m in zip(labels, means):
        print(f"fig3 {lab}: {m:.4f}", flush=True)

    # ---- Fig 6: cumulative MSE at 2/5/10/20/40 s horizons ----
    horizons = [2, 5, 10, 20, 40]
    cum_means = []
    for H in horizons:
        i1 = int(H * fs)
        vals = [float(np.mean((yh[:, :i1] - fut[:, :i1]) ** 2))
                for yh, fut in cum_pairs]
        cum_means.append(float(np.mean(vals)))
    fig, ax = plt.subplots()
    ax.plot(horizons, cum_means, marker="o")
    ax.set_xlabel("forecast horizon (s)")
    ax.set_ylabel("cumulative MSE over [0, H]")
    ax.set_title("Forecast error vs horizon (same checkpoint)")
    fig.savefig(FIGDIR / "fig6_horizon_cumulative.png", dpi=120)
    for H, m in zip(horizons, cum_means):
        print(f"fig6 MSE_{H}s: {m:.4f}", flush=True)

    # ---- Fig 5: baseline comparison ----
    pers, mech, neu = 1.560823, 1.068483, 0.800043
    hyb = float(np.mean([m for m in val[-5:]]))
    fig, ax = plt.subplots()
    ax.bar(["persistence\n(60s task)", "mechanistic\n(30s task)",
            "pure neural\n(30s task)", "hybrid\n(60s task)"],
           [pers, mech, neu, hyb])
    ax.set_ylabel("val MSE (note: different tasks, see labels)")
    ax.set_title("Baseline comparison (matched protocol where available)")
    fig.savefig(FIGDIR / "fig5_baselines.png", dpi=120)
    print(f"fig5 persistence={pers} mechanistic={mech} neural={neu} hybrid={hyb:.4f}",
          flush=True)
    print("figures in", FIGDIR, flush=True)


if __name__ == "__main__":
    main()
