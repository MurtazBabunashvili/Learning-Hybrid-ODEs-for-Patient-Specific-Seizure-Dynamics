"""Lyapunov stability + seizure-onset hypothesis + held-out TEST evaluation.

Deck slides 7/9, first real test:
  V(x)  = (x-x*)' P (x-x*), P = I
  Vd(x) = 2 (x-x*)' P F(x)                      (analytic, per step)
  S(t)  = Vd / (V + eps)
  H1: S(t) -> 0 preictally; preictal S distribution != interictal
      (Mann-Whitney U + rank-biserial effect size, not just a plot).
  Plus: held-out TEST reporting (per-channel RMSE, per-channel Pearson r,
  per-patient MSE table) on chb05/chb22/chb24.

Usage: python -u scripts/lyapunov_onset.py [--ckpt hybrid_cont.pt]
Writes artifacts/metrics/lyapunov_onset.json + fig7/fig8/fig9.
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

from src.data.loader import get_batches_cached
from src.evaluation.splits import sample_onset_relative
from src.experiments.common import euler_rollout, estimate_boundary_state
from src.experiments.hybrid import PureHybridODE

FIGDIR = ROOT / "artifacts" / "figures"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPS = 1e-6


class Tee:
    def log(self, msg: str) -> None:
        print(msg, flush=True)


def load_field(ckpt_name: str):
    ckpt = torch.load(ROOT / "artifacts" / "checkpoints" / ckpt_name,
                      map_location=DEVICE, weights_only=False)
    p2i = ckpt["patient_to_idx"]
    cohort = [p for p, _ in sorted(p2i.items(), key=lambda kv: kv[1])]
    cfg = ckpt.get("config", {})
    data = cfg.get("data", {})
    model = PureHybridODE(
        init_H=torch.zeros(22, 5), mu=torch.zeros(22),
        patient_ids=cohort).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    fs = float(data.get("ode_fs", 16.0))
    pre = int(round(float(data.get("prefix_sec", 20.0)) * fs))
    hor = int(round(float(data.get("horizon_sec", 40.0)) * fs))
    se = cfg.get("state_estimation", {})
    est = dict(inner_steps=int(se.get("inner_steps", 3)),
               state_lr=float(se.get("lr", 0.02)),
               ridge=float(se.get("ridge", 1e-4)),
               clamp=float(se.get("clamp", 2.5)), state_dim=5)
    return model, p2i, cohort, cfg, fs, pre, hor, est


def find_equilibrium(model, theta, inits, steps=300, lr=0.05):
    """Minimize ||F(x)||^2 from several inits; return (x*, residual).

    NOTE: theta is detached once up front. The indexed patient-table lookup
    carries an IndexBackward node; reusing the same indexed tensors across
    backward passes raises 'backward through the graph a second time'.
    The search optimizes x only, so no graph through theta is needed.
    """
    theta = {k: v.detach() for k, v in theta.items()}
    best, best_r = None, float("inf")
    for x_init in inits:
        x = x_init.detach().clone().requires_grad_(True)
        opt = torch.optim.Adam([x], lr=lr)
        for _ in range(steps):
            opt.zero_grad()
            r = model.field_with_theta(x.unsqueeze(0), theta).squeeze(0)
            loss = (r ** 2).sum()
            loss.backward()
            opt.step()
        with torch.no_grad():
            r = model.field_with_theta(x.detach().unsqueeze(0), theta).squeeze(0)
            res = float(r.norm())
        if res < best_r:
            best_r, best = res, x.detach().clone()
    return best, best_r


def rollout_S_batch(model, p2i, batch, pre, hor, dt, est, xs_map):
    """Batched S(t): one estimate + one rollout for B windows.

    Returns list of S np [T] aligned to batch order. Per-sample theta from
    the patient table; per-sample x* from xs_map. Prefix = first `pre` steps;
    S covers the `hor`-step model rollout after the prefix end.
    """
    y = torch.stack([b["y"] for b in batch]).to(DEVICE)
    pidx = torch.tensor([p2i[b["patient_id"]] for b in batch], device=DEVICE)
    prefix = y[:, :, :pre]
    x0 = estimate_boundary_state(model, prefix, pidx, pre, dt, **est)
    theta = {k: v for k, v in model.patient.batch(pidx).items()}
    traj = euler_rollout(model, x0, pidx, hor, dt,
                         theta_override=theta)  # [B,T,5]
    xs = torch.stack([torch.tensor(xs_map[b["patient_id"]],
                                   device=DEVICE) for b in batch])  # [B,5]
    D = traj - xs.unsqueeze(1)
    V = (D ** 2).sum(dim=-1)
    thetaT = {k: v.unsqueeze(-1) for k, v in theta.items()}  # [B,1] vs [B,T,5]
    F = model.field_with_theta(traj, thetaT)
    Vd = 2.0 * (D * F).sum(dim=-1)
    S = Vd / (V + EPS)
    return [S[i].detach().cpu().numpy() for i in range(S.shape[0])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="hybrid_cont.pt")
    args = ap.parse_args()
    FIGDIR.mkdir(parents=True, exist_ok=True)
    tee = Tee()
    BATCH = 32

    model, p2i, cohort, cfg, fs, pre, hor, est = load_field(args.ckpt)
    dt = 1.0 / fs
    splits = pickle.load(open(ROOT / "artifacts" / "splits" / "splits.pkl", "rb"))
    manifest = {r["recording_id"]: r for r in
                (json.loads(l) for l in
                 open(ROOT / "artifacts" / "manifests" / "chbmit.jsonl"))}

    # ---------- equilibria per patient (test + val seizure patients) ----------
    theta_of = {}
    for pid in sorted(set(p2i)):
        theta_of[pid] = {k: v.to(DEVICE) for k, v in
                         model.patient.batch(
                             torch.tensor([p2i[pid]], device=DEVICE)).items()}
    equilib = {}
    for pid in ["chb05", "chb22", "chb24", "chb04", "chb19", "chb23"]:
        inits = [torch.randn(5, device=DEVICE) * s for s in (0.5, 1.0, 2.0)]
        xs, res = find_equilibrium(model, theta_of[pid], inits)
        equilib[pid] = {"x_star": [float(v) for v in xs], "residual": res}
        tee.log(f"equilibrium {pid}: residual={res:.4f} x*={equilib[pid]['x_star']}")

    # ---------- onset-centered windows: [-300,+60]s around each onset ----------
    # plus interictal reference windows far from any seizure (same patients).
    ana_wins = []
    inter_wins = []
    for pid in ["chb05", "chb22", "chb24", "chb04", "chb19", "chb23"]:
        for rec in manifest.values():
            if rec["patient_id"] != pid:
                continue
            dur = float(rec["duration_sec"])
            if not rec["seizures"]:
                # interictal reference: spread across the recording
                s = 60.0
                while s + 60.0 <= dur and len(
                        [w for w in inter_wins if w["recording_id"] == rec["recording_id"]]) < 4:
                    inter_wins.append({
                        "patient_id": pid, "recording_id": rec["recording_id"],
                        "path": rec["path"], "start_sec": s, "end_sec": s + 60.0,
                        "kind": "interictal_ref", "seizures": [], "onset": None,
                    })
                    s += (dur - 120.0) / 4
                continue
            for sz in rec["seizures"]:
                on = float(sz["onset_sec"])
                s = max(0.0, on - 300.0)
                while s + 60.0 <= min(dur, on + 120.0):
                    ana_wins.append({
                        "patient_id": pid, "recording_id": rec["recording_id"],
                        "path": rec["path"], "start_sec": s, "end_sec": s + 60.0,
                        "kind": "ana", "seizures": rec["seizures"], "onset": on,
                    })
                    s += 30.0
    ana_wins += inter_wins
    tee.log(f"onset-analysis windows: {len(ana_wins)}")
    ab = get_batches_cached(ana_wins, 64.0, fs, tee, "ana")

    # ---------- S(t) tagged by time-to-onset ----------
    pre_vals, inter_vals, ict_vals, traces = [], [], [], []
    xs_map = {pid: equilib[pid]["x_star"] for pid in equilib}
    BATCH = 32
    for s in range(0, len(ab), BATCH):
        chunk_b = ab[s:s + BATCH]
        chunk_w = ana_wins[s:s + BATCH]
        Slist = rollout_S_batch(model, p2i, chunk_b, pre, hor, dt, est, xs_map)
        tee.log(f"S(t): chunk {s // BATCH + 1}/{(len(ab) + BATCH - 1) // BATCH}")
        for S, b, w in zip(Slist, chunk_b, chunk_w):
            pid = w["patient_id"]
            if w["onset"] is None:
                tto = np.full(hor, -9999.0)  # interictal ref: far from onset
            else:
                tto = ((np.arange(hor) / fs)
                       + (w["start_sec"] + pre / fs - w["onset"]))
            traces.append((tto, S, w["kind"], pid))
            for tt, ss in zip(tto, S):
                if w["onset"] is None or tt < -300:
                    inter_vals.append(float(ss))
                elif tt < 0:
                    pre_vals.append(float(ss))
                else:
                    ict_vals.append(float(ss))  # ictal + postictal tail
    pre_vals = np.array(pre_vals)
    inter_vals = np.array(inter_vals)
    tee.log(f"S samples: preictal={len(pre_vals)} interictal={len(inter_vals)} "
            f"ictal={len(ict_vals)}")

    U, p = mannwhitneyu(pre_vals, inter_vals, alternative="two-sided")
    n1, n2 = len(pre_vals), len(inter_vals)
    r_rb = 1.0 - 2.0 * U / (n1 * n2)  # rank-biserial
    res = {
        "n_preictal": n1, "n_interictal": n2,
        "median_preictal": float(np.median(pre_vals)),
        "median_interictal": float(np.median(inter_vals)),
        "frac_neg_preictal": float((pre_vals < 0).mean()),
        "frac_neg_interictal": float((inter_vals < 0).mean()),
        "mannwhitney_U": float(U), "p_value": float(p),
        "rank_biserial_r": float(r_rb),
        "equilibria": equilib,
    }
    tee.log("ONSET TEST: " + json.dumps({k: v for k, v in res.items()
                                         if k != "equilibria"}, indent=1))

    # fig7: median S(t) aligned to onset
    grid = np.arange(-300, 61, 5)
    med, lo, hi = [], [], []
    tall = np.concatenate([t for t, _, _, _ in traces])
    sall = np.concatenate([s for _, s, _, _ in traces])
    for gc in grid:
        m = (tall >= gc - 2.5) & (tall < gc + 2.5)
        if m.sum() > 20:
            med.append(float(np.median(sall[m])))
            lo.append(float(np.percentile(sall[m], 25)))
            hi.append(float(np.percentile(sall[m], 75)))
        else:
            med.append(np.nan)
            lo.append(np.nan)
            hi.append(np.nan)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(grid, med, label="median S(t)")
    ax.fill_between(grid, lo, hi, alpha=0.3, label="IQR")
    ax.axvline(0, color="r", ls="--", label="onset")
    ax.set_xlabel("time to onset (s)")
    ax.set_ylabel("S(t)")
    ax.set_title("Lyapunov margin aligned to seizure onset")
    ax.legend()
    fig.savefig(FIGDIR / "fig7_Strace.png", dpi=120)

    # fig8: preictal vs interictal S distributions
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(np.clip(inter_vals, -5, 5), bins=60, alpha=0.5, label="interictal",
            density=True)
    ax.hist(np.clip(pre_vals, -5, 5), bins=60, alpha=0.5, label="preictal",
            density=True)
    ax.set_xlabel("S(t) (clipped)")
    ax.set_title(f"S distributions (p={p:.2e}, r_rb={r_rb:+.3f})")
    ax.legend()
    fig.savefig(FIGDIR / "fig8_Sdist.png", dpi=120)

    # ---------- held-out TEST reporting (forecast task, 20s->40s) ----------
    cfg_split = cfg.get("split", {})
    tw = sample_onset_relative(splits["test"],
                               cfg_split.get("test_patients",
                                             ["chb05", "chb22", "chb24"]),
                               300, {"interictal": 0.4, "preictal_far": 0.2,
                                     "preictal_near": 0.2, "ictal": 0.1,
                                     "postictal": 0.1}, 7)
    tb = get_batches_cached(tw, 64.0, fs, tee, "testrep")
    mses, corrs, pats, kinds = [], [], [], []
    ch_mse = np.zeros(22)
    ch_n = np.zeros(22)
    for s in range(0, len(tb), BATCH):
        chunk = tb[s:s + BATCH]
        y = torch.stack([b["y"] for b in chunk]).to(DEVICE)
        pidx = torch.tensor([p2i[b["patient_id"]] for b in chunk],
                            device=DEVICE)
        prefix = y[:, :, :pre]
        fut = y[:, :, pre:pre + hor]
        x0 = estimate_boundary_state(model, prefix, pidx, pre, dt, **est)
        theta = {k: v for k, v in model.patient.batch(pidx).items()}
        yh = model.observe(euler_rollout(
            model, x0, pidx, hor, dt,
            theta_override=theta)).permute(0, 2, 1)
        e = ((yh - fut) ** 2).mean(dim=(1, 2)).detach().cpu().numpy()
        yc = fut.detach().cpu().numpy()
        hc = yh.detach().cpu().numpy()
        for j, b in enumerate(chunk):
            mses.append(float(e[j]))
            pats.append(b["patient_id"])
            kinds.append(b["kind"])
            ch_mse += ((hc[j] - yc[j]) ** 2).mean(axis=1)
            ch_n += 1
            cc = [float(np.corrcoef(yc[j, c], hc[j, c])[0, 1])
                  if np.std(hc[j, c]) > 1e-9 else 0.0 for c in range(22)]
            corrs.append(float(np.mean(cc)))
        if (s // BATCH + 1) % 4 == 0:
            tee.log(f"test forecast: {min(s + BATCH, len(tb))}/{len(tb)}")
    mses = np.array(mses)
    res["test"] = {
        "n": len(mses), "mse_mean": float(mses.mean()),
        "mse_std": float(mses.std()), "rmse_mean": float(np.sqrt(mses.mean())),
        "corr_mean": float(np.mean(corrs)), "corr_std": float(np.std(corrs)),
        "per_patient": {p: {"n": int((np.array(pats) == p).sum()),
                            "mse": float(mses[np.array(pats) == p].mean())}
                        for p in sorted(set(pats))},
        "per_kind": {k: float(mses[np.array(kinds) == k].mean())
                     for k in sorted(set(kinds))},
        "per_channel_rmse": [float(v) for v in np.sqrt(ch_mse / ch_n)],
    }
    tee.log("TEST: " + json.dumps(res["test"], indent=1)[:2000])

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(22), res["test"]["per_channel_rmse"])
    ax.set_xlabel("channel")
    ax.set_ylabel("RMSE")
    ax.set_title("Held-out TEST per-channel RMSE (20s->40s forecast)")
    fig.savefig(FIGDIR / "fig9_test_channels.png", dpi=120)

    json.dump(res, open(ROOT / "artifacts" / "metrics" / "lyapunov_onset.json", "w"),
              indent=1)
    tee.log("wrote lyapunov_onset.json + fig7/fig8/fig9")


if __name__ == "__main__":
    main()
