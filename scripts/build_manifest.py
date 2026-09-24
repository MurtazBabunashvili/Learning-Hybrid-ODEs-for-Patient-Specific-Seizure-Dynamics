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
    anno = json.loads(open(root / "annotations" / "seizures_by_recording.json").read())
    rows = []
    missing = []
    for ftr in sorted(data_root.glob("*.ftr")):
        rec_id = ftr.stem  # chb01_03
        patient = rec_id.split("_")[0]
        h = header(str(ftr))
        if rec_id not in anno:
            missing.append(rec_id)
        rows.append({
            "patient_id": patient, "recording_id": rec_id,
            "path": str(ftr), "duration_sec": h["duration_sec"],
            "n_eeg_channels": 22,
            "seizures": [{"onset_sec": float(s["onset_sec"]),
                          "offset_sec": float(s["offset_sec"])}
                         for s in anno.get(rec_id, [])],
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
