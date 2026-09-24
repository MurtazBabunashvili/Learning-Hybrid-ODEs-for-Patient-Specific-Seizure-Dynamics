"""Windowing -> artifacts/windows/windows.json + splits (one-time data prep).

Settings mirror config/config.yaml data/split sections. Re-run only when
the manifest or sampling setup changes; training reads splits.pkl.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.manifest import load_manifest
from src.data.windowing import build_windows_for_recording
from src.evaluation.splits import patient_split

# Must match config/config.yaml split section.
TRAIN_PATIENTS = ["chb01", "chb02", "chb03", "chb06", "chb07", "chb08",
                  "chb09", "chb10", "chb11", "chb12", "chb13", "chb14",
                  "chb15", "chb16", "chb17", "chb18", "chb20", "chb21"]
VAL_PATIENTS = ["chb04", "chb19", "chb23"]
TEST_PATIENTS = ["chb05", "chb22", "chb24"]

WINDOW_SEC = 60.0
STRIDE_SEC = 15.0
PREICTAL_SEC = 300.0

root = Path(__file__).resolve().parents[1]
manifest_path = root / "artifacts" / "manifests" / "chbmit.jsonl"
if not manifest_path.exists():
    from src.data.manifest import build_manifest
    build_manifest(root / "data", manifest_path)
recs = load_manifest(manifest_path)
print(f"patients present: {sorted({r['patient_id'] for r in recs})}")

wins: list[dict] = []
for r in recs:
    wins += build_windows_for_recording(
        r, window_sec=WINDOW_SEC, stride_sec=STRIDE_SEC,
        preictal_sec=PREICTAL_SEC)
print(f"total windows: {len(wins)}")
from collections import Counter
print(Counter(w["kind"] for w in wins))

out = root / "artifacts" / "windows" / "windows.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(wins, indent=1))
splits = patient_split(wins, TRAIN_PATIENTS, VAL_PATIENTS, TEST_PATIENTS)
# patient-holdout guard
assert not (set(TRAIN_PATIENTS) & set(TEST_PATIENTS)), "train/test overlap!"
sout = root / "artifacts" / "splits" / "splits.json"
sout.parent.mkdir(parents=True, exist_ok=True)
sout.write_text(json.dumps({k: len(v) for k, v in splits.items()}, indent=1))
print({k: len(v) for k, v in splits.items()})
import pickle
with open(root / "artifacts" / "splits" / "splits.pkl", "wb") as f:
    pickle.dump(splits, f)
