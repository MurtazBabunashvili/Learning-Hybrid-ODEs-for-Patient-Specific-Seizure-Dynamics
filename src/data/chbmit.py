"""CHB-MIT dataset quirks: summaries, seizure files, channel handling.

Official notes respected here:
- most recordings have 23 EEG signals, some have extra (ECG);
- recording lengths differ substantially (chb04 ~4h vs ~1h);
- seizure onset/end annotations live in *-summary.txt (+ binary .seizures markers).
We parse *-summary.txt as the source of truth (the .seizures files are binary
duplicates) and normalise channels downstream.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from pathlib import Path
import mne


@dataclass
class SeizureEvent:
    onset_sec: float
    offset_sec: float


@dataclass
class Recording:
    patient_id: str
    recording_id: str
    edf_path: str
    duration_sec: float
    channels: list[str] = field(default_factory=list)
    seizures: list[SeizureEvent] = field(default_factory=list)

SUMMARY_RE = re.compile(
    r"File Name: (?P<fname>\S+).*?Number of Seizures in File: (?P<n>\d+)(?P<rest>.*?)(?=File Name:|\Z)",
    re.S,
)

# Canonical bipolar montage order seen in summaries (T8-P8 appears twice).
CANONICAL_23 = [
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2", "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
    "FZ-CZ", "CZ-PZ", "P7-T7", "T7-FT9", "FT9-FT10", "FT10-T8", "T8-P8",
]

NON_EEG = {"ECG", "EKG", "EMG"}


def parse_summary(summary_path: str | Path) -> dict[str, list[SeizureEvent]]:
    """Return {edf_filename: [SeizureEvent]} parsed from a *-summary.txt."""
    text = Path(summary_path).read_text()
    out: dict[str, list[SeizureEvent]] = {}
    for m in SUMMARY_RE.finditer(text):
        fname = m.group("fname")
        rest = m.group("rest")
        starts = [float(x) for x in re.findall(r"Seizure.*?Start Time: (\d+)", rest)]
        ends = [float(x) for x in re.findall(r"Seizure.*?End Time: (\d+)", rest)]
        out[fname] = [SeizureEvent(s, e) for s, e in zip(starts, ends)]
    return out


def dedup_channels(ch_names: list[str]) -> tuple[list[str], list[int]]:
    """Drop non-EEG, dedup repeated labels (T8-P8 x2 -> keep first).

    Returns (kept_names, kept_indices).
    """
    kept, idx = [], []
    seen = set()
    for i, ch in enumerate(ch_names):
        base = ch.split("-0")[0].split("-1")[0]  # undo MNE duplicate renaming
        if base in NON_EEG:
            continue
        if base in seen and base == "T8-P8":
            continue  # second T8-P8 copy
        if ch in seen:
            continue
        seen.add(base if base == "T8-P8" else ch)
        kept.append(base)
        idx.append(i)
    return kept, idx


# Fixed canonical montage: the 22 unique bipolar EEG channels of the
# standard CHB-MIT setup (T8-P8 listed twice in headers -> kept once).
# 655/686 files match fully; 28 miss the same 4 temporal-chain channels
# (zero-filled); 3 chb12 files use a disjoint montage (excluded, see below).
CANONICAL_22 = [
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2", "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
    "FZ-CZ", "CZ-PZ", "P7-T7", "T7-FT9", "FT9-FT10", "FT10-T8",
]

# Exclude recordings whose montage is disjoint from canonical (no signal to
# salvage, e.g. chb12_27/28/29 with CS2-reference montage).
MAX_MISSING_CHANNELS = 6


def canonicalize_indices(ch_names: list[str]) -> list[int]:
    """Map CANONICAL_22 -> EDF column index, or -1 if absent (zero-filled).

    First occurrence wins (handles duplicated T8-P8); junk/ECG/EKG/non-EEG
    labels never match canonical entries and are ignored.
    """
    base = [c.split("-0")[0].split("-1")[0] for c in ch_names]
    first: dict[str, int] = {}
    for i, b in enumerate(base):
        first.setdefault(b, i)
    return [first.get(c, -1) for c in CANONICAL_22]


def canonicalize_matrix(data: "np.ndarray", idx: list[int]) -> "np.ndarray":
    """Build (22, T) canonical matrix; missing rows are zeros."""
    import numpy as np
    out = np.zeros((len(CANONICAL_22), data.shape[1]), dtype=data.dtype)
    for r, i in enumerate(idx):
        if i >= 0:
            out[r] = data[i]
    return out


def scan_patient(patient_dir: str | Path) -> list[Recording]:
    """Build Recording list for one patient dir (no signal data loaded)."""
    patient_dir = Path(patient_dir)
    patient_id = patient_dir.name
    summary = patient_dir / f"{patient_id}-summary.txt"
    seizure_map = parse_summary(summary) if summary.exists() else {}
    recordings: list[Recording] = []
    for edf in sorted(patient_dir.glob("*.edf")):
        try:
            raw = mne.io.read_raw_edf(str(edf), preload=False, verbose=False)
            sfreq = float(raw.info["sfreq"])
            duration = float(raw.n_times) / sfreq if sfreq else 0.0
            ch_names = list(raw.ch_names)
        except Exception:
            duration, ch_names = 0.0, []
        seizures = seizure_map.get(edf.name, [])
        recordings.append(
            Recording(
                patient_id=patient_id,
                recording_id=edf.stem,
                edf_path=str(edf),
                duration_sec=duration,
                channels=ch_names,
                seizures=seizures,
            )
        )
    return recordings
