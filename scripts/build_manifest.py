"""Build artifacts/manifests/chbmit.jsonl (source of truth).

Formats:
  edf     local data/chbXX/*.edf + *-summary.txt (default, unchanged)
  feather cloud data/seizure_256Hz_dataset/*.ftr (22ch float16 µV) +
          annotations/seizures_by_recording.json sidecar
Usage: python -u scripts/build_manifest.py [--format edf|feather] [--data DIR]
"""
from __future__ import annotations
import sys
import argparse
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.manifest import build_manifest

root = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--format", default="edf", choices=["edf", "feather"])
ap.add_argument("--data", default=None)
args = ap.parse_args()

if args.format == "edf":
    data_root = Path(args.data) if args.data else root / "data"
    rows = build_manifest(data_root, root / "artifacts" / "manifests" / "chbmit.jsonl")
else:
    from src.data.feather import header
    data_root = Path(args.data) if args.data else root / "data" / "seizure_256Hz_dataset"
    # Authoritative annotations live next to the feathers (198 rows, covers
    # every file incl. chb17a/chb17b/chb12_27 variants my EDF sidecar lacks).
    import csv as _csv
    import re as _re
    anno = {}
    with open(data_root.parent / "seizure_events.csv") as f:
        for row in _csv.DictReader(f):
            # series_id like chb01_03.edf (or chb02_16+.edf); chb17a_03 -> chb17
            rec_id = row["series_id"].removesuffix(".edf")
            anno.setdefault(rec_id, []).append(
                {"onset_sec": float(row["onset"]),
                 "offset_sec": float(row["offset"])})
    rows = []
    missing = []
    for ftr in sorted(data_root.glob("*.ftr")):
        rec_id = ftr.stem  # chb01_03
        m = _re.match(r"(chb\d+)", rec_id)
        patient = m.group(1) if m else rec_id.split("_")[0]
        h = header(str(ftr))
        if rec_id not in anno:
            missing.append(rec_id)
        rows.append({
            "patient_id": patient, "recording_id": rec_id,
            "path": str(ftr), "duration_sec": h["duration_sec"],
            "n_eeg_channels": 22,
            "seizures": anno.get(rec_id, []),
        })
    import json as _json
    out = root / "artifacts" / "manifests" / "chbmit.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for r in rows:
            f.write(_json.dumps(r) + "\n")
    print(f"no annotation for: {missing if missing else 'none'}")

n_sz = sum(len(r["seizures"]) for r in rows)
print(f"wrote {len(rows)} recordings, {n_sz} seizure events")
