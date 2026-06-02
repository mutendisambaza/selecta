"""Content hashing helpers for stored audio files."""

from __future__ import annotations

import hashlib

_CHUNK_SIZE = 1024 * 1024


def audio_hash(path: str) -> str:
    """Return a stable SHA-1 hash for the file contents at ``path``."""
    digest = hashlib.sha1()

    with open(path, "rb") as file_obj:
        while True:
            chunk = file_obj.read(_CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)

    return digest.hexdigest()
