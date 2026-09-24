"""Dataset breadth figure (no training, metadata only).

Reproduces the exact train/val/test selections and renders one composite
figure: summary table + phase bars + per-patient bars + windows-per-recording
+ windows-per-episode + spacing stats. Prints a one-line numeric check.
Usage: python -u scripts/dataset_summary.py
Saves artifacts/figures/fig0_dataset.png
"""
from __future__ import annotations
import sys
import json
import pickle
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.evaluation.splits import sample_targeted, sample_onset_relative

FIGDIR = ROOT / "artifacts" / "figures"
QUOTAS = {"interictal": 0.4, "preictal_far": 0.2, "preictal_near": 0.2,
          "ictal": 0.1, "postictal": 0.1}
TRAIN_P = ['chb01', 'chb02', 'chb03', 'chb06', 'chb07', 'chb08', 'chb09',
           'chb10', 'chb11', 'chb12', 'chb13', 'chb14', 'chb15', 'chb16',
           'chb17', 'chb18', 'chb20', 'chb21']
VAL_P = ['chb04', 'chb19', 'chb23']
TEST_P = ['chb05', 'chb22', 'chb24']


def phase_of(w):
    from src.evaluation.splits import window_phase
    return window_phase(w)


def episode_key(w):
    onsets = [float(s["onset_sec"]) for s in w.get("seizures", [])
              if float(s["onset_sec"]) >= 0]
    if not onsets or w["kind"] == "interictal":
        return None
    start = float(w["start_sec"])
    return (w["recording_id"], min(onsets, key=lambda on: abs(start - on)))


def stats_for(wins):
    pats = sorted({w["patient_id"] for w in wins})
    recs = defaultdict(list)
    eps = defaultdict(list)
    for w in wins:
        recs[w["recording_id"]].append(w)
        k = episode_key(w)
        if k is not None:
            eps[k].append(w)
    per_rec = sorted(len(v) for v in recs.values())
    mingaps = []
    for v in recs.values():
        ss = sorted(float(w["start_sec"]) for w in v)
        for a, b in zip(ss, ss[1:]):
            mingaps.append(b - a)
    return {
        "n": len(wins), "patients": pats,
        "recordings": len(recs), "episodes": len(eps),
        "phases": dict(Counter(phase_of(w) for w in wins)),
        "per_patient": {p: sum(1 for w in wins if w["patient_id"] == p)
                        for p in pats},
        "w_per_rec": per_rec,
        "w_per_ep": sorted(len(v) for v in eps.values()),
        "gaps": mingaps,
    }


def main():
    FIGDIR.mkdir(parents=True, exist_ok=True)
    manifest = [json.loads(l) for l in
                open(ROOT / "artifacts" / "manifests" / "chbmit.jsonl")]
    splits = pickle.load(open(ROOT / "artifacts" / "splits" / "splits.pkl", "rb"))
    train, _ = sample_targeted(manifest, TRAIN_P, 50000, seed=0)
    val = sample_onset_relative(splits["val"], VAL_P, 120, QUOTAS, 1)
    test = sample_onset_relative(splits["test"], TEST_P, 300, QUOTAS, 7)
    S = {"train": stats_for(train), "val": stats_for(val),
         "test": stats_for(test)}

    # patient-disjoint check
    assert not (set(S["train"]["patients"]) & set(S["test"]["patients"]))
    assert not (set(S["train"]["patients"]) & set(S["val"]["patients"]))

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.45, wspace=0.3)

    # A: summary table
    ax = fig.add_subplot(gs[0, :])
    ax.axis("off")
    rows = [["split", "patients", "recordings", "episodes", "windows",
             "inter", "pre", "ictal", "post"]]
    for name in ["train", "val", "test"]:
        s = S[name]
        ph = s["phases"]
        pre = ph.get("preictal", 0) + ph.get("preictal_far", 0) + ph.get("preictal_near", 0)
        rows.append([name, len(s["patients"]), s["recordings"], s["episodes"],
                     s["n"], ph.get("interictal", 0), pre,
                     ph.get("ictal", 0), ph.get("postictal", 0)])
    rows.append(["disjoint", "train∩val=∅ train∩test=∅", "", "", "", "", "", "", ""])
    t = ax.table(cellText=rows, loc="center", colWidths=[0.14] + [0.095] * 8)
    t.auto_set_font_size(False)
    t.set_fontsize(8)
    ax.set_title("Dataset breadth: 20s-prefix / 40s-forecast windows @16Hz")

    # B: phase bars per split
    ax = fig.add_subplot(gs[1, 0])
    labels = ["interictal", "preictal", "ictal", "postictal"]
    x = np.arange(len(labels))
    for i, name in enumerate(["train", "val", "test"]):
        s = S[name]
        ph = s["phases"]
        pre = ph.get("preictal", 0) + ph.get("preictal_far", 0) + ph.get("preictal_near", 0)
        ax.bar(x + (i - 1) * 0.25,
               [ph.get("interictal", 0), pre, ph.get("ictal", 0),
                ph.get("postictal", 0)], width=0.25, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(["inter", "pre", "ictal", "post"])
    ax.set_ylabel("windows (log)")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.set_title("Phase distribution per split")

    # C: per-patient train bars
    ax = fig.add_subplot(gs[1, 1])
    pp = S["train"]["per_patient"]
    ax.bar(range(len(pp)), [pp[p] for p in sorted(pp)], tick_label=sorted(pp))
    ax.tick_params(axis="x", labelsize=7, rotation=45)
    ax.set_ylabel("windows")
    ax.set_title(f"Train windows/patient (min/max "
                 f"{min(pp.values())}/{max(pp.values())})")

    # D: windows/recording + windows/episode + gaps
    ax = fig.add_subplot(gs[2, 0])
    ax.hist(S["train"]["w_per_rec"], bins=30, alpha=0.7, label="train w/rec")
    ax.hist(S["val"]["w_per_rec"], bins=15, alpha=0.7, label="val w/rec")
    ax.set_xlabel("windows per recording")
    ax.legend(fontsize=8)
    ax.set_title("Recording spread")
    ax = fig.add_subplot(gs[2, 1])
    we = S["train"]["w_per_ep"]
    ax.hist(we, bins=max(10, len(set(we))), alpha=0.7)
    ax.set_xlabel("windows per seizure episode (train)")
    med_gap = float(np.median(S["train"]["gaps"])) if S["train"]["gaps"] else float("nan")
    ax.set_title(f"Episode spread (median gap {med_gap:.0f}s)")

    fig.savefig(FIGDIR / "fig0_dataset.png", dpi=120)
    s = S["train"]
    print(f"train={s['n']} recs={s['recordings']} eps={s['episodes']} "
          f"pat_minmax={min(s['per_patient'].values())}/"
          f"{max(s['per_patient'].values())} "
          f"val={S['val']['n']} test={S['test']['n']} "
          f"disjoint=OK -> {FIGDIR / 'fig0_dataset.png'}", flush=True)


if __name__ == "__main__":
    main()
