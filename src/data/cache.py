"""Disk cache for preprocessed window batches.

Keyed by window (path, start, end) list + preprocessing params, so retrains
skip EDF read + filter entirely. Tensors stored on CPU; moved to device
in the training step.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

PREPROCESS_VERSION = "v6"  # v6: NaN feather channels zero-filled at read

_DEFAULT_PREPROC: dict = {
    "l_freq": 0.5,
    "h_freq": 70.0,
    "notch": 60.0,
    "clip": 8.0,
    "normalize": True,
    "normalization": "global",
    "resample": "mne",
    "version": PREPROCESS_VERSION,
}


def cache_key(windows: list[dict], target_fs: float, ode_fs: float,
              preproc: dict | None = None) -> str:
    """Key windows + fs + preprocessing settings.

    preproc holds {l_freq, h_freq, notch, clip, normalize, normalization,
    resample, version}. When None (legacy 3-arg calls), today's defaults
    (global z-score, 0.5-70Hz, notch 60, clip 8, mne resample, v3) are used.
    """
    if preproc is None:
        eff = dict(_DEFAULT_PREPROC)
    else:
        eff = dict(_DEFAULT_PREPROC)
        eff.update(preproc)
        if "version" not in preproc:
            eff["version"] = PREPROCESS_VERSION
    payload = json.dumps(
        [[w["path"], w["start_sec"], w["end_sec"]] for w in windows]
        + [target_fs, ode_fs, eff],
        sort_keys=True,
    )
    return hashlib.sha1(payload.encode()).hexdigest()[:12]


def cache_file(cache_dir: str | Path, tag: str, key: str) -> Path:
    return Path(cache_dir) / f"{tag}_{key}.pt"


def save_cache(path: str | Path, batches: list[dict], meta: dict) -> None:
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"batches": batches, "meta": meta}, path)


def load_cache(path: str | Path):
    import torch
    path = Path(path)
    if not path.exists():
        return None
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return None
