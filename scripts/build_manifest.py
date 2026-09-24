"""Build artifacts/manifests/chbmit.jsonl from local data/ (source of truth)."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.manifest import build_manifest

root = Path(__file__).resolve().parents[1]
rows = build_manifest(root / "data", root / "artifacts" / "manifests" / "chbmit.jsonl")
n_sz = sum(len(r["seizures"]) for r in rows)
print(f"wrote {len(rows)} recordings, {n_sz} seizure events")
