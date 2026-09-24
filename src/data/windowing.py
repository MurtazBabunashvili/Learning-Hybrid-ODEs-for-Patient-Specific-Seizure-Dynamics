"""Windowing: seizure-centered oversampling + strided interictal background."""
from __future__ import annotations


def label_window(
    start: float, end: float, seizures: list[dict],
    preictal_sec: float = 300.0, postictal_sec: float = 60.0,
) -> str:
    """interictal | preictal | ictal | postictal for window [start,end)."""
    for sz in seizures:
        on, off = float(sz["onset_sec"]), float(sz["offset_sec"])
        if start < off and end > on:
            return "ictal"
        if on - preictal_sec <= start < on and end <= on + (off - on):
            return "preictal"
        if off <= start < off + postictal_sec:
            return "postictal"
    return "interictal"


def build_windows_for_recording(
    recording: dict, window_sec: float = 30.0,
    stride_sec: float = 15.0, preictal_sec: float = 300.0,
    peri_onset_radius: float = 180.0,
) -> list[dict]:
    dur = float(recording["duration_sec"])
    seizures = recording.get("seizures", [])
    wins: list[dict] = []
    # 1) dense coverage around each seizure onset
    for sz in seizures:
        on = float(sz["onset_sec"])
        t = max(0.0, on - peri_onset_radius)
        end = min(dur - window_sec, on + peri_onset_radius)
        while t <= end:
            kind = label_window(t, t + window_sec, seizures, preictal_sec)
            wins.append(_mk(recording, t, window_sec, kind))
            t += stride_sec / 2.0  # denser near events
    # 2) strided background over whole recording
    t = 0.0
    while t + window_sec <= dur:
        kind = label_window(t, t + window_sec, seizures, preictal_sec)
        if kind == "interictal":
            wins.append(_mk(recording, t, window_sec, kind))
        t += stride_sec * 4.0  # sparse background (imbalance control)
    # dedup by start time
    seen, uniq = set(), []
    for w in sorted(wins, key=lambda w: w["start_sec"]):
        key = round(w["start_sec"], 1)
        if key not in seen:
            seen.add(key)
            uniq.append(w)
    return uniq


def _mk(rec: dict, start: float, window_sec: float, kind: str) -> dict:
    return {
        "patient_id": rec["patient_id"],
        "recording_id": rec["recording_id"],
        "path": rec["path"],
        "start_sec": start,
        "end_sec": start + window_sec,
        "kind": kind,
        "seizures": rec.get("seizures", []),
    }
