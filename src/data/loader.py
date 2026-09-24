"""Cache-aware EDF -> tensor loading for main.py.

Moved verbatim from scripts/train.py (the only two functions the current
experiment uses). Dependency chain: loader -> {cache, chbmit, preprocessing}.
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import numpy as np
import os
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor

warnings.filterwarnings(
    "ignore", message=".*Channel names are not unique.*")

ROOT = Path(__file__).resolve().parents[2]

# Crop margin around a recording's window span: covers FIR filter transients
# so region-cropped output matches full-file filtering away from edges.
_MARGIN_SEC = 30.0

# Serializes log lines only; batch slots are disjoint per recording.
_LOG_LOCK = threading.Lock()


def _tsafe_log(tee, msg: str) -> None:
    with _LOG_LOCK:
        tee.log(msg)


def get_batches_cached(wins: list[dict], target_fs: float, ode_fs: float,
                       tee, tag: str, sharded: bool = False,
                       lru_files: int = 32):
    """Disk-cached wrapper around load_windows_grouped.

    sharded=False (legacy): returns the full list (fine for small selections).
    sharded=True: per-recording shard files + lazy LRU sequence -- peak RAM
    stays flat (~a few hundred MB) no matter how many windows.
    `tee` is any object with a .log(str) method.
    """
    from src.data.cache import cache_key, cache_file, save_cache, load_cache
    key = cache_key(wins, target_fs, ode_fs)
    if sharded:
        return get_sharded_cached(wins, target_fs, ode_fs, tee, tag, key,
                                  lru_files)
    cpath = cache_file(ROOT / "artifacts" / "windows" / "cache", tag, key)
    hit = load_cache(cpath)
    if hit is not None and len(hit["batches"]) == len(wins):
        tee.log(f"{tag}: CACHE HIT {cpath.name} "
                f"({len(wins)} windows, {cpath.stat().st_size / 1024**2:.1f}MB) "
                f"- no EDF loading")
        return hit["batches"]
    tee.log(f"{tag}: cache miss ({cpath.name}) - loading EDFs ...")
    batches = load_windows_grouped(wins, target_fs, ode_fs, tee, tag)
    save_cache(cpath, batches, {"key": key, "n": len(wins),
                                "target_fs": target_fs, "ode_fs": ode_fs})
    tee.log(f"{tag}: cached to {cpath.name} "
            f"({cpath.stat().st_size / 1024**2:.1f}MB)")
    return batches


class ShardedWindows:
    """Lazy indexable view over per-recording shard files.

    Only `lru_files` shard files are held in RAM (OrderedDict LRU).
    Supports len() and [i] -> window dict (CPU tensors), so it plugs
    into torch Datasets without ever materializing the full set.
    """

    def __init__(self, shard_dir, index: list, lru_files: int = 32):
        from collections import OrderedDict
        import torch as _t
        self._dir = Path(shard_dir)
        self._index = index  # [(shard_file, local_idx, meta_dict)]
        self._lru_files = max(1, int(lru_files))
        self._cache: OrderedDict = OrderedDict()
        self._torch = _t

    def __len__(self) -> int:
        return len(self._index)

    def _load_shard(self, shard_file: str):
        hit = self._cache.get(shard_file)
        if hit is not None:
            self._cache.move_to_end(shard_file)
            return hit
        data = self._torch.load(self._dir / shard_file, map_location="cpu",
                                weights_only=False)
        self._cache[shard_file] = data
        while len(self._cache) > self._lru_files:
            self._cache.popitem(last=False)
        return data

    def __getitem__(self, i: int) -> dict:
        shard_file, local_idx, meta = self._index[i]
        y, t = self._load_shard(shard_file)[local_idx]
        return {"y": y, "t": t, **meta}

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def ram_mb(self) -> float:
        total = 0
        for shard in self._cache.values():
            for y, _ in shard:
                total += y.nelement() * y.element_size()
        return total / 1024**2


def get_sharded_cached(wins: list[dict], target_fs: float, ode_fs: float,
                       tee, tag: str, key: str, lru_files: int = 32,
                       n_workers: int = 4):
    """Sharded variant: artifacts/windows/cache/{tag}_{key}.d/ sharded by
    recording (rec_XXXX.pt) + index.json. Reloads are instant metadata reads;
    window tensors stream from disk with an LRU file cache.

    The build streams per-recording shards straight to disk (bounded RAM:
    only `n_workers` recordings in flight, 4 by default) instead of
    accumulating all windows in RAM first.
    """
    import gc
    import json
    from collections import defaultdict
    import mne
    from concurrent.futures import ThreadPoolExecutor
    from src.data.chbmit import CANONICAL_22, canonicalize_indices, canonicalize_matrix
    from src.data.preprocessing import preprocess_chunk, decimate_aa

    dpath = ROOT / "artifacts" / "windows" / "cache" / f"{tag}_{key}.d"
    idx_path = dpath / "index.json"
    if idx_path.exists():
        index = json.loads(idx_path.read_text())
        tee.log(f"{tag}: SHARD CACHE HIT {dpath.name} "
                f"({len(index)} windows) - no EDF loading")
        return ShardedWindows(dpath, index, lru_files)
    tee.log(f"{tag}: shard cache miss ({dpath.name}) - loading EDFs ...")
    by_rec: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    for i, w in enumerate(wins):
        by_rec[w["path"]].append((i, w))
    total_recs = len(by_rec)
    _tsafe_log(tee, f"{tag}: {len(wins)} windows from {total_recs} recordings "
                    f"({n_workers} loader workers, streaming to disk)")
    dpath.mkdir(parents=True, exist_ok=True)
    index: list = [None] * len(wins)
    done = [0]
    step = max(1, int(round(target_fs / ode_fs)))

    def _one(args):
        r, path, items = args
        _tsafe_log(tee, f"{tag}: loading recording {r + 1}/{total_recs} "
                        f"{Path(path).name} ({len(items)} windows) ...")
        span_lo = max(0.0, min(float(w["start_sec"]) for _, w in items) - _MARGIN_SEC)
        span_hi = max(float(w["end_sec"]) for _, w in items) + _MARGIN_SEC
        if path.lower().endswith(".ftr"):
            # Feather seizure dataset: 256 Hz float16 µV, canonical columns.
            from src.data.feather import read_chunk as _ftr_chunk
            from src.data.chbmit import CANONICAL_22 as _C22
            data_uv, sfreq, names = _ftr_chunk(path, span_lo, span_hi)
            import numpy as _np
            if int((abs(data_uv).max(axis=1) == 0).sum()) > 6:
                raise ValueError(f"{path}: disjoint montage, excluded from cohort")
            raw = None
        else:
            raw = mne.io.read_raw_edf(path, preload=False, verbose=False)
            sfreq = float(raw.info["sfreq"])
            cidx = canonicalize_indices(list(raw.ch_names))
            if sum(i < 0 for i in cidx) > 6:
                raise ValueError(f"{path}: disjoint montage, excluded from cohort")
            dur = float(raw.n_times) / sfreq
            raw.crop(tmin=span_lo, tmax=min(span_hi, dur), include_tmax=False)
            raw.load_data(verbose=False)
            data_uv = canonicalize_matrix(raw.get_data(), cidx) * 1e6  # (22, T)
            del raw
        x_full, fs = preprocess_chunk(data_uv, sfreq, target_fs)
        del data_uv
        del raw  # noqa: defined in the EDF branch above; feather path skips it
        shard = []
        for gi, w in items:
            i0 = int((float(w["start_sec"]) - span_lo) * fs)
            i1 = int((float(w["end_sec"]) - span_lo) * fs)
            seg = x_full[:, i0:i1]
            xc = decimate_aa(seg, step, fs, ode_fs)
            t = np.arange(xc.shape[1]) / ode_fs
            shard.append((gi, torch.from_numpy(xc.copy()),
                          torch.from_numpy(t).float(), w))
        del x_full
        fname = f"rec_{r:04d}.pt"
        torch.save([(y, t) for _, y, t, _ in shard], dpath / fname)
        out = []
        for local, (gi, y, t, w) in enumerate(shard):
            out.append((gi, fname, local, {
                "kind": w["kind"], "patient_id": w["patient_id"],
                "recording_id": w["recording_id"],
                "start_sec": w["start_sec"]}))
        del shard
        return out

    import gc as _gc
    n_done = 0
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        for made in pool.map(
                _one, [(r, p, it) for r, (p, it) in enumerate(by_rec.items())]):
            for gi, fname, local, meta in made:
                index[gi] = [fname, local, meta]
                done[0] += 1
            n_done += 1
            if n_done % 25 == 0:
                _gc.collect()
            if n_done % 25 == 0 or done[0] == len(wins):
                _tsafe_log(tee, f"{tag}: [{done[0]}/{len(wins)} "
                                f"{100.0 * done[0] / len(wins):.1f}%] "
                                f"{n_done}/{total_recs} recordings")
    idx_path.write_text(json.dumps(index))
    tee.log(f"{tag}: sharded cache written to {dpath.name}")
    return ShardedWindows(dpath, index, lru_files)


def load_windows_grouped(wins: list[dict], target_fs: float, ode_fs: float,
                         tee, tag: str, n_workers: int | None = None) -> list[dict]:
    """Load each EDF once, preprocess once, slice all its windows.

    Canonical 22-channel montage (missing rows zero-filled; disjoint
    montages raise). Recordings load in parallel over a thread pool (EDF
    I/O + numpy FIR work releases the GIL; per-window slots are disjoint
    so results are identical to serial). Progress per window.
    """
    from collections import defaultdict
    import mne
    from concurrent.futures import ThreadPoolExecutor
    from src.data.chbmit import CANONICAL_22, canonicalize_indices, canonicalize_matrix
    from src.data.preprocessing import preprocess_chunk, decimate_aa

    if n_workers is None:
        import os as _os
        n_workers = max(1, (_os.cpu_count() or 4) - 1)

    by_rec: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    for i, w in enumerate(wins):
        by_rec[w["path"]].append((i, w))

    _tsafe_log(tee, f"{tag}: {len(wins)} windows from {len(by_rec)} recordings "
                    f"({n_workers} loader workers)")
    batches: list[dict | None] = [None] * len(wins)
    done = [0]
    total_recs = len(by_rec)

    def _one(args):
        r, path, items = args
        _tsafe_log(tee, f"{tag}: loading recording {r + 1}/{total_recs} "
                        f"{Path(path).name} ({len(items)} windows) ...")
        t0 = time.time()
        # Region-cropped read: only the spanned segment (+margin for FIR
        # edge effects) is loaded/filtered -- never the full (4h) file.
        span_lo = max(0.0, min(float(w["start_sec"]) for _, w in items) - _MARGIN_SEC)
        span_hi = max(float(w["end_sec"]) for _, w in items) + _MARGIN_SEC
        span_lo = max(0.0, min(float(w["start_sec"]) for _, w in items) - _MARGIN_SEC)
        span_hi = max(float(w["end_sec"]) for _, w in items) + _MARGIN_SEC
        if path.lower().endswith(".ftr"):
            # Feather seizure dataset: 256 Hz float16 µV, canonical columns.
            from src.data.feather import read_chunk as _ftr_chunk
            from src.data.chbmit import CANONICAL_22 as _C22
            data_uv, sfreq, names = _ftr_chunk(path, span_lo, span_hi)
            import numpy as _np
            if int((abs(data_uv).max(axis=1) == 0).sum()) > 6:
                raise ValueError(f"{path}: disjoint montage, excluded from cohort")
            dur = float("inf")
        else:
            raw = mne.io.read_raw_edf(path, preload=False, verbose=False)
            sfreq = float(raw.info["sfreq"])
            cidx = canonicalize_indices(list(raw.ch_names))
            if sum(i < 0 for i in cidx) > 6:
                raise ValueError(f"{path}: disjoint montage, excluded from cohort")
            dur = float(raw.n_times) / sfreq
            raw.crop(tmin=span_lo, tmax=min(span_hi, dur), include_tmax=False)
            raw.load_data(verbose=False)
            data_uv = canonicalize_matrix(raw.get_data(), cidx) * 1e6  # (22, T)
            del raw
        t1 = time.time()
        x_full, fs = preprocess_chunk(data_uv, sfreq, target_fs)
        del data_uv, raw
        step = max(1, int(round(target_fs / ode_fs)))
        made = []
        for gi, w in items:
            i0 = int((float(w["start_sec"]) - span_lo) * fs)
            i1 = int((float(w["end_sec"]) - span_lo) * fs)
            seg = x_full[:, i0:i1]
            # Anti-aliased decimation: low-pass below ode_fs/2 BEFORE downsampling.
            xc = decimate_aa(seg, step, fs, ode_fs)
            t = np.arange(xc.shape[1]) / ode_fs
            made.append((gi, {
                "y": torch.from_numpy(xc.copy()),
                "t": torch.from_numpy(t).float(),
                "kind": w["kind"],
                "patient_id": w["patient_id"],
                "recording_id": w["recording_id"],
                "start_sec": w["start_sec"],
            }, w))
        return made, time.time() - t0, time.time() - t1

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        for made, dt_tot, dt_filt in pool.map(
                _one, [(r, p, it) for r, (p, it) in enumerate(by_rec.items())]):
            for gi, b, w in made:
                batches[gi] = b
                done[0] += 1
                pct = 100.0 * done[0] / len(wins)
                _tsafe_log(tee, f"{tag}: [{done[0]}/{len(wins)} {pct:5.1f}%] "
                                f"{w['recording_id']}@{w['start_sec']:.0f}s kind={w['kind']} "
                                f"shape={b['y'].shape}")
    return batches  # type: ignore
