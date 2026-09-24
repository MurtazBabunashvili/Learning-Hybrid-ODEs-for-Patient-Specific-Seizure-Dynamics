"""Preprocessing: EEG select -> filter -> resample -> normalize. Raw EDF untouched."""
from __future__ import annotations
import numpy as np


def bandpass_filter(
    data: np.ndarray, sfreq: float, l_freq: float = 0.5, h_freq: float = 70.0,
) -> np.ndarray:
    try:
        import mne
        return mne.filter.filter_data(
            data, sfreq, l_freq, h_freq, picks=None, fir_design="firwin", verbose=False
        )
    except Exception:
        return data  # fallback: no filtering if mne unavailable


def notch_filter(data: np.ndarray, sfreq: float, freq: float = 60.0) -> np.ndarray:
    try:
        import mne
        return mne.filter.notch_filter(data, sfreq, freqs=[freq], verbose=False)
    except Exception:
        return data


def resample(data: np.ndarray, sfreq: float, target_fs: float) -> tuple[np.ndarray, float]:
    if abs(target_fs - sfreq) < 1e-6:
        return data, sfreq
    try:
        import mne
        # mne expects [n_chan, n_times]
        out = mne.filter.resample(data, up=target_fs / sfreq, npad="auto", verbose=False)
        return out, float(target_fs)
    except Exception:
        step = max(1, int(round(sfreq / target_fs)))
        return data[:, ::step], float(sfreq / step)


def zscore_per_channel(data: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mu = data.mean(axis=1, keepdims=True)
    sd = data.std(axis=1, keepdims=True) + eps
    return (data - mu) / sd


def zscore_causal(full: np.ndarray, i0: int, i1: int, eps: float = 1e-6) -> np.ndarray:
    """Z-score slice full[:, i0:i1] using mean/std of data strictly before slice end.

    Stats are computed over full[:, :i1] (past + slice itself, no future beyond
    i1), so a huge future spike at >= i1 cannot affect the returned slice.
    """
    n_times = full.shape[1]
    i0c = max(0, int(i0))
    i1c = min(n_times, int(i1))
    if i1c <= 0 or i1c <= i0c and i1c == 0:
        return np.empty((full.shape[0], 0), dtype=np.float32)
    if i1c <= i0c:
        return np.empty((full.shape[0], max(0, i1c - i0c)), dtype=np.float32)
    past = full[:, :i1c]
    if past.shape[1] == 0:
        seg = full[:, i0c:i1c]
        return np.zeros_like(seg, dtype=np.float32)
    mu = past.mean(axis=1, keepdims=True)
    sd = past.std(axis=1, keepdims=True) + eps
    seg = full[:, i0c:i1c]
    return ((seg - mu) / sd).astype(np.float32)


def clip_outliers(data: np.ndarray, thresh: float = 8.0) -> np.ndarray:
    return np.clip(data, -thresh, thresh)


def decimate_aa(x: np.ndarray, step: int, fs_before: float, fs_after: float) -> np.ndarray:
    """Anti-alias decimation by integer step.

    Applies an FIR lowpass just below fs_after/2 via mne, then takes every
    step-th sample. Falls back to naive slicing if mne is missing or the
    cutoff is invalid.
    """
    step = int(step)
    if step <= 1:
        return x
    try:
        import mne
        h_freq = float(fs_after) / 2.0 * 0.95  # just below Nyquist of target
        if h_freq <= 0 or h_freq >= float(fs_before) / 2.0:
            return x[:, ::step]
        y = mne.filter.filter_data(
            x, float(fs_before), None, h_freq,
            fir_design="firwin", verbose=False,
        )
        return y[:, ::step]
    except Exception:
        return x[:, ::step]


def preprocess_recording(
    data_uv: np.ndarray, sfreq: float, params: dict,
) -> tuple[np.ndarray, float]:
    """Canonical per-recording preprocessing: bandpass->notch->resample->normalize->clip.

    Driven ENTIRELY by params dict keys:
      {target_fs, l_freq, h_freq, notch, clip, normalize: bool,
       normalization: "global"|"causal"}.
    Missing keys fall back to legacy defaults
    (target 64Hz, 0.5-70Hz, notch 60, clip 8, normalize True, global).

    Causal mode defers normalization: returns filtered/resampled (unnormalized)
    data so the caller can apply zscore_causal(full, i0, i1) per window slice
    (using only data strictly before the slice end) then clip. Global mode
    keeps the legacy whole-array z-score + clip.
    """
    p = params or {}
    target_fs = float(p.get("target_fs", 64.0))
    l_freq = p.get("l_freq", 0.5)
    h_freq = p.get("h_freq", 70.0)
    notch = p.get("notch", 60.0)
    clip = p.get("clip", 8.0)
    normalize = p.get("normalize", True)
    normalization = p.get("normalization", "global")

    x = data_uv
    if not (l_freq is None and h_freq is None):
        x = bandpass_filter(x, sfreq, l_freq, h_freq)
    if notch is not None and notch:
        try:
            x = notch_filter(x, sfreq, float(notch))
        except Exception:
            pass
    x, fs = resample(x, sfreq, target_fs)
    if normalize:
        if normalization == "causal":
            # Deferred: caller applies zscore_causal per slice + clip.
            pass
        else:
            x = zscore_per_channel(x)
            if clip is not None:
                x = clip_outliers(x, float(clip))
    return x.astype(np.float32), float(fs)


def preprocess_chunk(
    data_uv: np.ndarray, sfreq: float, target_fs: float = 64.0,
    l_freq: float = 0.5, h_freq: float = 70.0, notch: float = 60.0,
    clip: float = 8.0, normalize: bool = True,
    normalization: str = "global",
) -> tuple[np.ndarray, float]:
    params = {
        "target_fs": target_fs,
        "l_freq": l_freq,
        "h_freq": h_freq,
        "notch": notch,
        "clip": clip,
        "normalize": normalize,
        "normalization": normalization,
    }
    return preprocess_recording(data_uv, sfreq, params)
