"""Canonical 22ch montage + preprocessing + manifest/split integrity."""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]


def test_canonical_22_fixed():
    import numpy as np
    from src.data.chbmit import CANONICAL_22, canonicalize_indices, canonicalize_matrix
    assert len(CANONICAL_22) == 22 and len(set(CANONICAL_22)) == 22
    names = ["FP1-F7", "ECG", "T8-P8", "T8-P8", "-", "FZ-CZ"]
    idx = canonicalize_indices(names)
    assert len(idx) == 22 and idx[0] == 0 and idx[14] == 2
    out = canonicalize_matrix(np.ones((len(names), 10)), idx)
    assert out.shape == (22, 10)


def test_manifest_and_holdout():
    rows = [json.loads(l) for l in
            open(ROOT / "artifacts" / "manifests" / "chbmit.jsonl")]
    assert len(rows) > 600
    assert all(r["n_eeg_channels"] == 22 for r in rows)
    import pickle
    splits = pickle.load(open(ROOT / "artifacts" / "splits" / "splits.pkl", "rb"))
    tr = {w["patient_id"] for w in splits["train"]}
    te = {w["patient_id"] for w in splits["test"]}
    assert not (tr & te), "patient holdout violated"
