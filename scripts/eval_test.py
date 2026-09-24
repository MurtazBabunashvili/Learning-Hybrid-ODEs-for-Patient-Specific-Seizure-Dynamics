"""Shared held-out TEST evaluator: identical protocol for every model.

Fixed test set: 300 windows, chb05/chb22/chb24, quotas 40/20/20/10/10,
seed 7 (== the windows behind hybrid's 1.051). Metrics: MSE/std/RMSE,
per-kind, per-patient, per-channel RMSE, mean Pearson r, persistence MSE.
Usage: python -u scripts/eval_test.py --ckpt <file> [--config <yaml>]
Writes artifacts/metrics/test_<stem>.json
"""
from __future__ import annotations
import sys
import json
import argparse
import pickle
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import yaml
import os

os.environ["MAIN_SILENT"] = "1"  # import main.py with no log side effects
# main.py parses sys.argv at import (config override); shield it so this
# script's own CLI (e.g. --ckpt x.pt --config y.yaml) can't leak through.
_argv_backup = sys.argv
sys.argv = [sys.argv[0]]
import main as M
sys.argv = _argv_backup

from src.data.loader import get_batches_cached
from src.evaluation.splits import sample_onset_relative
from src.experiments.common import euler_rollout, estimate_boundary_state

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
QUOTAS = {"interictal": 0.4, "preictal_far": 0.2, "preictal_near": 0.2,
          "ictal": 0.1, "postictal": 0.1}
N_TEST = 300
SEED_TEST = 7


class Tee:
    def log(self, msg: str) -> None:
        print(msg, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    ckpt = torch.load(ROOT / "artifacts" / "checkpoints" / args.ckpt,
                      map_location=DEVICE, weights_only=False)
    cfg = ckpt.get("config", {})
    if args.config:
        cfg = yaml.safe_load(open(ROOT / "config" / args.config))
    data = cfg.get("data", {})
    fs = float(data.get("ode_fs", 16.0))
    pre = int(round(float(data.get("prefix_sec", 20.0)) * fs))
    hor = int(round(float(data.get("horizon_sec", 40.0)) * fs))
    dt = 1.0 / fs
    se = cfg.get("state_estimation", {})
    est = dict(inner_steps=int(se.get("inner_steps", 3)),
               state_lr=float(se.get("lr", 0.02)),
               ridge=float(se.get("ridge", 1e-4)),
               clamp=float(se.get("clamp", 2.5)), state_dim=5)
    test_pats = cfg.get("split", {}).get("test_patients",
                                         ["chb05", "chb22", "chb24"])

    # Rebuild the exact model class from the checkpoint's experiment name.
    exp_name = cfg.get("experiment", {}).get("name", "hybrid")
    p2i = ckpt["patient_to_idx"]
    cohort = [p for p, _ in sorted(p2i.items(), key=lambda kv: kv[1])]
    model = M.build_experiment(cfg, torch.zeros(22, 5), torch.zeros(22),
                               cohort).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()

    splits = pickle.load(open(ROOT / "artifacts" / "splits" / "splits.pkl", "rb"))
    tw = sample_onset_relative(splits["test"], test_pats, N_TEST, QUOTAS,
                               SEED_TEST)
    tee = Tee()
    tb = get_batches_cached(tw, 64.0, fs, tee, "testrep")

    mses, maes, corrs, pats, kinds = [], [], [], [], []
    ch_mse = np.zeros(22)
    pers = []
    HSEC = [2, 5, 10, 20, 40]
    hcum = {h: [] for h in HSEC}
    examples = []  # (yc, hc, kind, rec, start) first of ictal/preictal/interictal
    seen_kinds = set()
    B = 32
    for s in range(0, len(tb), B):
        chunk = tb[s:s + B]
        y = torch.stack([b["y"] for b in chunk]).to(DEVICE)
        pidx = torch.tensor([p2i[b["patient_id"]] for b in chunk], device=DEVICE)
        prefix, fut = y[:, :, :pre], y[:, :, pre:pre + hor]
        x0 = estimate_boundary_state(model, prefix, pidx, pre, dt, **est)
        theta = {k: v for k, v in model.patient.batch(pidx).items()}
        yh = model.observe(euler_rollout(
            model, x0, pidx, hor, dt,
            theta_override=theta)).permute(0, 2, 1)
        e = ((yh - fut) ** 2).mean(dim=(1, 2)).detach().cpu().numpy()
        ae = (yh - fut).abs().mean(dim=(1, 2)).detach().cpu().numpy()
        yc, hc = fut.detach().cpu().numpy(), yh.detach().cpu().numpy()
        pm = ((prefix[:, :, -1:] .expand_as(fut) - fut) ** 2).mean(
            dim=(1, 2)).detach().cpu().numpy()
        for j, b in enumerate(chunk):
            mses.append(float(e[j]))
            maes.append(float(ae[j]))
            pers.append(float(pm[j]))
            pats.append(b["patient_id"])
            kinds.append(b["kind"])
            ch_mse += ((hc[j] - yc[j]) ** 2).mean(axis=1)
            cc = [float(np.corrcoef(yc[j, c], hc[j, c])[0, 1])
                  if np.std(hc[j, c]) > 1e-9 else 0.0 for c in range(22)]
            corrs.append(float(np.mean(cc)))
            for H in HSEC:
                i1 = min(int(H * fs), hor)
                hcum[H].append(float(np.mean((hc[j][:, :i1] - yc[j][:, :i1]) ** 2)))
            if b["kind"] not in seen_kinds and len(examples) < 3:
                seen_kinds.add(b["kind"])
                examples.append((yc[j], hc[j], b["kind"], b["recording_id"],
                                 b["start_sec"]))
    mses = np.array(mses)
    maes = np.array(maes)
    pats_arr = np.array(pats)
    kinds_arr = np.array(kinds)
    out = {
        "ckpt": args.ckpt, "experiment": exp_name, "n": len(mses),
        "mse_mean": float(mses.mean()), "mse_std": float(mses.std()),
        "rmse_mean": float(np.sqrt(mses.mean())),
        "mae_mean": float(maes.mean()), "mae_std": float(maes.std()),
        "corr_mean": float(np.mean(corrs)), "corr_std": float(np.std(corrs)),
        "persistence_mse": float(np.mean(pers)),
        "horizon_cumulative_mse": {f"{H}s": float(np.mean(hcum[H])) for H in HSEC},
        "per_patient": {p: {"mse": float(mses[pats_arr == p].mean()),
                            "mae": float(maes[pats_arr == p].mean()),
                            "n": int((pats_arr == p).sum())}
                        for p in sorted(set(pats))},
        "per_kind": {k: {"mse": float(mses[kinds_arr == k].mean()),
                         "mae": float(maes[kinds_arr == k].mean()),
                         "n": int((kinds_arr == k).sum())}
                     for k in sorted(set(kinds))},
        "per_channel_rmse": [float(v) for v in np.sqrt(ch_mse / len(tb))],
    }
    path = ROOT / "artifacts" / "metrics" / f"test_{Path(args.ckpt).stem}.json"
    json.dump(out, open(path, "w"), indent=1)
    print(f"TEST {args.ckpt}: mse={out['mse_mean']:.4f}±{out['mse_std']:.4f} "
          f"rmse={out['rmse_mean']:.4f} mae={out['mae_mean']:.4f}±{out['mae_std']:.4f} "
          f"corr={out['corr_mean']:+.4f} persist={out['persistence_mse']:.4f}", flush=True)
    print(f"  horizon={out['horizon_cumulative_mse']}", flush=True)
    print(f"  per_patient={out['per_patient']}", flush=True)
    print(f"  per_kind={out['per_kind']}", flush=True)
    print(f"wrote {path}", flush=True)

    # ---- figures ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figdir = ROOT / "artifacts" / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.ckpt).stem
    import re
    logname = {"hybrid_diverse.pt": "hybrid_diverse.log"}.get(args.ckpt)
    if logname and (ROOT / "artifacts" / "metrics" / logname).exists():
        ep, tr, va = [], [], []
        for line in open(ROOT / "artifacts" / "metrics" / logname):
            m = re.search(r"epoch (\d+) \| train_total=([\d.]+) .* val=([\d.]+)", line)
            if m:
                ep.append(int(m.group(1)))
                tr.append(float(m.group(2)))
                va.append(float(m.group(3)))
        if ep:
            fig, ax = plt.subplots()
            ax.plot(ep, tr, label="train")
            ax.plot(ep, va, label="val")
            ax.set_xlabel("epoch")
            ax.set_ylabel("forecast MSE")
            ax.set_title(f"Learning curves ({stem})")
            ax.legend()
            fig.savefig(figdir / f"fig1_{stem}.png", dpi=120)
    chs = [0, 5, 10, 15]
    fig, axes = plt.subplots(len(examples), 1, figsize=(12, 3 * len(examples)),
                             sharex=True)
    if len(examples) == 1:
        axes = [axes]
    for ax, (yc, hc, kind, rec, st) in zip(axes, examples):
        t = np.arange(hor) / fs
        for c in chs:
            ax.plot(t, yc[c], color="k", lw=1.0, alpha=0.8)
            ax.plot(t, hc[c], color="r", lw=1.0, alpha=0.8)
        ax.set_title(f"{rec}@{st:.0f}s {kind} (black=true, red=pred)")
        ax.set_xlabel("forecast s")
    axes[0].legend(["true", "pred"])
    fig.savefig(figdir / f"fig2_{stem}.png", dpi=120)
    fig, ax = plt.subplots()
    ax.plot(HSEC, [out["horizon_cumulative_mse"][f"{H}s"] for H in HSEC],
            marker="o")
    ax.set_xlabel("horizon (s)")
    ax.set_ylabel("cumulative MSE [0,H]")
    ax.set_title(f"Error vs horizon ({stem})")
    fig.savefig(figdir / f"fig6_{stem}.png", dpi=120)
    fig, ax = plt.subplots()
    ax.bar(list(out["per_patient"].keys()),
           [v["mse"] for v in out["per_patient"].values()])
    ax.set_ylabel("MSE")
    ax.set_title(f"Per-patient test MSE ({stem})")
    fig.savefig(figdir / f"fig9_{stem}.png", dpi=120)
    print(f"figures in {figdir}", flush=True)


if __name__ == "__main__":
    main()
