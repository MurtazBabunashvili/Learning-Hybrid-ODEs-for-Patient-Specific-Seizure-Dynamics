"""Feather source: header/chunk/manifest round-trip on EDF-derived data."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
ROOT = Path(__file__).resolve().parents[1]

from src.data.feather import header, read_chunk
from src.data.chbmit import CANONICAL_22


def _make_ftr(tmp_path):
    import mne
    mne.set_log_level("ERROR")
    raw = mne.io.read_raw_edf(
        str(ROOT / "data" / "chb01" / "chb01_03.edf"),
        preload=True, verbose=False)
    sf = float(raw.info["sfreq"])
    # canonical order, first occurrence wins (dedupe T8-P8 like pipeline)
    take = []
    used = set()
    for c in CANONICAL_22:
        for i, rc in enumerate(raw.ch_names):
            b = rc.split("-0")[0].split("-1")[0]
            if b == c and i not in used:
                take.append(i)
                used.add(i)
                break
    assert len(take) == 22
    d = raw.get_data(picks=take)[:, :int(60 * sf)] * 1e6
    df = pd.DataFrame(d.T.astype(np.float16), columns=CANONICAL_22)
    df["series_id"] = pd.Categorical(["chb01_03"] * len(df))
    df["p_id"] = ["chb01"] * len(df)
    p = tmp_path / "chb01_03.ftr"
    df.to_feather(p)
    return p


def test_feather_header_and_chunk(tmp_path):
    p = _make_ftr(tmp_path)
    h = header(str(p))
    assert h["n_rows"] == 60 * 256
    assert abs(h["duration_sec"] - 60.0) < 1e-6
    data, sf, names = read_chunk(str(p), 10.0, 20.0)
    assert sf == 256.0
    assert names == CANONICAL_22
    assert data.shape == (22, 2560)
    assert data.dtype == np.float32
    # value check against EDF ground truth
    import mne
    mne.set_log_level("ERROR")
    raw = mne.io.read_raw_edf(
        str(ROOT / "data" / "chb01" / "chb01_03.edf"),
        preload=True, verbose=False)
    assert abs(float(np.abs(data).max()) - 3000) < 3000  # sane µV range
    assert len(raw.ch_names) >= 22


def test_feather_nan_zero_filled(tmp_path):
    import pandas as pd
    from src.data.feather import read_chunk
    n = 2560
    rng = np.random.default_rng(1)
    cols = {c: rng.normal(0, 50, n).astype(np.float16) for c in CANONICAL_22}
    cols["T7-FT9"] = np.full(n, np.nan, dtype=np.float16)
    cols["series_id"] = pd.Categorical(["x"] * n)
    cols["p_id"] = ["chb99"] * n
    df = pd.DataFrame(cols)
    p = tmp_path / "nan.ftr"
    df.to_feather(p)
    data, sf, _ = read_chunk(str(p), 0.0, 10.0)
    assert bool(np.isfinite(data).all())
    assert (data[CANONICAL_22.index("T7-FT9")] == 0).all()


def test_feather_matches_edf_values(tmp_path):
    import mne
    mne.set_log_level("ERROR")
    p = _make_ftr(tmp_path)
    data, _, _ = read_chunk(str(p), 0.0, 5.0)
    raw = mne.io.read_raw_edf(
        str(ROOT / "data" / "chb01" / "chb01_03.edf"),
        preload=True, verbose=False)
    lut = {}
    used = set()
    for c in CANONICAL_22:
        for i, rc in enumerate(raw.ch_names):
            if rc.split("-0")[0].split("-1")[0] == c and i not in used:
                lut[c] = i
                used.add(i)
                break
    ref = raw.get_data(picks=[lut[c] for c in CANONICAL_22])[:, :1280] * 1e6
    assert np.allclose(data, ref.astype(np.float32), atol=2.0)  # float16 quant
