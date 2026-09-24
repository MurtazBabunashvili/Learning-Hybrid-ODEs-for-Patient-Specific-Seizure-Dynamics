"""Master manifest: one JSONL row per recording. Rest of project reads this."""
from __future__ import annotations
import json
from pathlib import Path
from .chbmit import scan_patient, canonicalize_indices, MAX_MISSING_CHANNELS


def build_manifest(data_root: str | Path, out_path: str | Path) -> list[dict]:
    data_root = Path(data_root)
    rows: list[dict] = []
    skipped: list[str] = []
    for patient_dir in sorted(p for p in data_root.iterdir() if p.is_dir()):
        if not patient_dir.name.startswith("chb"):
            continue  # skip .idea, archives, etc.
        if not list(patient_dir.glob("*.edf")):
            continue
        for rec in scan_patient(patient_dir):
            idx = canonicalize_indices(rec.channels)
            n_missing = sum(i < 0 for i in idx)
            if n_missing > MAX_MISSING_CHANNELS:
                skipped.append(rec.recording_id)  # e.g. chb12_27-29, disjoint montage
                continue
            rows.append(
                {
                    "patient_id": rec.patient_id,
                    "recording_id": rec.recording_id,
                    "path": rec.edf_path,
                    "duration_sec": rec.duration_sec,
                    "n_eeg_channels": 22,
                    "n_missing_channels": n_missing,
                    "seizures": [
                        {"onset_sec": s.onset_sec, "offset_sec": s.offset_sec}
                        for s in rec.seizures
                    ],
                }
            )
    print(f"excluded disjoint-montage recordings: {skipped}")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return rows


def load_manifest(path: str | Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
