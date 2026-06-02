from __future__ import annotations

import hashlib

import pytest

from selecta.store import audio_hash


def test_audio_hash_is_stable_for_same_content(tmp_path):
    audio_file = tmp_path / "track.wav"
    audio_file.write_bytes(b"selecta-test-audio" * 128)

    first_hash = audio_hash(str(audio_file))
    second_hash = audio_hash(str(audio_file))

    assert first_hash == second_hash
    assert first_hash == hashlib.sha1(audio_file.read_bytes()).hexdigest()


def test_audio_hash_changes_when_content_changes(tmp_path):
    first_file = tmp_path / "first.wav"
    second_file = tmp_path / "second.wav"
    first_file.write_bytes(b"alpha-content")
    second_file.write_bytes(b"beta-content")

    assert audio_hash(str(first_file)) != audio_hash(str(second_file))


def test_audio_hash_raises_for_missing_file(tmp_path):
    missing_file = tmp_path / "missing.wav"

    with pytest.raises(FileNotFoundError):
        audio_hash(str(missing_file))
