"""Raw EEG panels: (a) interictal segment, (b) seizure-onset segment.

Same patient/recording (chb01_03, onset 2996 s), same channels, µV.
No captions baked in; panels labeled (a)/(b) only.
Usage: python -u scripts/fig_eeg.py
"""
import sys
sys.path.insert(0, '.')
from pathlib import Path
import numpy as np
import mne
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

mne.set_log_level("ERROR")
ROOT = Path(__file__).resolve().parents[1]

F = ROOT / "data" / "chb01" / "chb01_03.edf"
ONSET = 2996.0
CHS = ["FP1-F7", "F7-T7", "T7-P7", "P7-O1", "F3-C3", "C3-P3",
       "F4-C4", "C4-P4", "FZ-CZ", "CZ-PZ"]


def load_seg(t0, t1):
    raw = mne.io.read_raw_edf(str(F), preload=True, verbose=False)
    sf = float(raw.info["sfreq"])
    idx = [raw.ch_names.index(c) for c in CHS]
    d = raw.get_data(picks=idx)[:, int(t0 * sf):int(t1 * sf)] * 1e6
    return d, sf


def panel(ax, d, sf, t0, tag, onset=None):
    t = t0 + np.arange(d.shape[1]) / sf
    off = 0.0
    step = float(np.percentile(np.abs(d), 99)) * 1.6 + 1e-6
    ticks = []
    for c in range(d.shape[0]):
        ax.plot(t, d[c] + off, lw=0.6, color="k")
        ticks.append(off)
        off += step
    ax.set_yticks(ticks)
    ax.set_yticklabels(CHS)
    if onset is not None:
        ax.axvline(onset, color="r", lw=1.2)
    ax.set_xlim(t[0], t[-1])
    ax.set_xlabel("time (s)")
    ax.set_ylabel("µV (offset traces)")
    ax.set_title(f"({tag})", loc="left", fontsize=12)
    ax.tick_params(labelsize=7)


d_a, sf = load_seg(600.0, 620.0)          # interictal, far from onset
d_b, _ = load_seg(ONSET - 10.0, ONSET + 10.0)

fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 8), sharey=False)
panel(a1, d_a, sf, 600.0, "a")
panel(a2, d_b, sf, ONSET - 10.0, "b", onset=ONSET)
fig.tight_layout()
out = ROOT / "artifacts" / "figures" / "fig_eeg_raw.png"
fig.savefig(out, dpi=150)
print(f"wrote {out} ({len(CHS)} channels, {sf:.0f} Hz)", flush=True)
