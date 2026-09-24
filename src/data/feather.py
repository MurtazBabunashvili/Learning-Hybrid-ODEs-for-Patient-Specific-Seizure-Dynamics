"""Feather (.ftr) seizure-dataset source: 256 Hz, 22 float EEG channels + ids.

Cloud layout: data/seizure_256Hz_dataset/chb01_03.ftr with columns
[22 bipolar channels..., series_id, p_id], values already in microvolts
(float16 on disk -> float32 in RAM). Annotations come from the
annotations/seizures_by_recording.json sidecar (built from CHB-MIT
summaries), keyed by recording id.
"""
from __future__ import annotations

SFREQ = 256.0


def header(path: str) -> dict:
    """Row count + columns without reading data."""
    import pyarrow.feather as feather
    n_rows = feather.read_table(str(path), columns=[]).num_rows
    try:
        import pyarrow.ipc as ipc
        with ipc.open_file(str(path)) as f:
            names = list(f.schema.names)
    except Exception:
        names = []
    return {"n_rows": n_rows, "columns": names,
            "duration_sec": n_rows / SFREQ, "sfreq": SFREQ}


def read_chunk(path: str, start_sec: float, end_sec: float,
               channels: list[str] | None = None):
    """Return (data [C, T] float32 microvolts, sfreq, channel names)."""
    import pandas as pd
    df = pd.read_feather(path)
    ch = channels or [c for c in df.columns if c not in ("series_id", "p_id")]
    sfreq = SFREQ
    i0 = max(0, int(start_sec * sfreq))
    i1 = min(len(df), int(end_sec * sfreq))
    data = df[ch].to_numpy(dtype="float32", copy=True).T  # [C, T], already µV
    return data[:, i0:i1], sfreq, list(ch)
