"""Folder-level analysis helpers shared by the CLI commands."""
from __future__ import annotations

import os
from pathlib import Path

from selecta.features.extractor import ExtractionError, extract
from selecta.store.db import FeatureStore

AUDIO_EXTS = {".mp3", ".wav", ".aiff", ".aif", ".m4a", ".flac", ".ogg"}


def iter_audio_files(folder: str):
    """Yield absolute paths of audio files under ``folder`` (sorted, recursive)."""
    for root, _dirs, files in os.walk(folder):
        for name in sorted(files):
            if Path(name).suffix.lower() in AUDIO_EXTS:
                yield os.path.join(root, name)


def analyze_folder(folder: str, store: FeatureStore, verbose: bool = True) -> tuple[int, int, int]:
    """Extract + cache features for new/changed files. Returns (analysed, skipped, failed)."""
    analysed = skipped = failed = 0
    for path in iter_audio_files(folder):
        if not store.needs_analysis(path):
            skipped += 1
            continue
        try:
            feats = extract(path)
            store.upsert(feats)
            analysed += 1
            if verbose:
                print(f"  + {Path(path).name}  |  {feats.bpm:.1f} BPM  {feats.key_camelot}  "
                      f"({len(feats.phrases)} phrases)")
        except (ExtractionError, Exception) as exc:  # noqa: BLE001 - keep batch resilient
            failed += 1
            if verbose:
                print(f"  ! {Path(path).name}  FAILED: {exc}")
    return analysed, skipped, failed
