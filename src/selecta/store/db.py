"""SQLite-backed feature store."""

from __future__ import annotations

import json
import sqlite3

from selecta.features.types import TrackFeatures
from selecta.store.hashing import audio_hash


class FeatureStore:
    """Persist analyzed track features in SQLite."""

    def __init__(self, db_path: str):
        self._connection = sqlite3.connect(db_path)
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS tracks (
                path TEXT PRIMARY KEY,
                content_hash TEXT NOT NULL,
                features_json TEXT NOT NULL
            )
            """
        )
        self._connection.commit()

    def upsert(self, features: TrackFeatures) -> None:
        payload = json.dumps(features.to_dict())
        content_hash = audio_hash(features.path)
        self._connection.execute(
            """
            INSERT OR REPLACE INTO tracks (path, content_hash, features_json)
            VALUES (?, ?, ?)
            """,
            (features.path, content_hash, payload),
        )
        self._connection.commit()

    def get(self, path: str) -> TrackFeatures | None:
        cursor = self._connection.execute(
            "SELECT features_json FROM tracks WHERE path = ?",
            (path,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return TrackFeatures.from_dict(json.loads(row[0]))

    def all(self) -> list[TrackFeatures]:
        cursor = self._connection.execute(
            "SELECT features_json FROM tracks ORDER BY path"
        )
        return [TrackFeatures.from_dict(json.loads(row[0])) for row in cursor.fetchall()]

    def needs_analysis(self, path: str) -> bool:
        cursor = self._connection.execute(
            "SELECT content_hash FROM tracks WHERE path = ?",
            (path,),
        )
        row = cursor.fetchone()
        if row is None:
            return True

        try:
            return row[0] != audio_hash(path)
        except OSError:
            return True

    def close(self) -> None:
        self._connection.close()
