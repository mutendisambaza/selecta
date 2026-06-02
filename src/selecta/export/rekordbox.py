"""Rekordbox XML cue export."""

from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import quote
import xml.etree.ElementTree as ET

from selecta.features.types import PhraseRegion, TrackFeatures

_HOT_CUE_LABELS = ("intro", "drop", "breakdown", "outro")
_KIND = "MP3 File"


def _location_for_path(path: str) -> str:
    absolute_path = os.path.abspath(path)
    return "file://localhost" + quote(absolute_path)


def _position_mark_name(phrase: PhraseRegion) -> str:
    name = phrase.label.capitalize()
    if phrase.confidence < 0.5:
        return f"{name}?"
    return name


def _assign_hot_cues(phrases: tuple[PhraseRegion, ...]) -> dict[int, str]:
    assigned: dict[int, str] = {}
    next_hot_cue = 0

    for label in _HOT_CUE_LABELS:
        for index, phrase in enumerate(phrases):
            if phrase.label != label or index in assigned:
                continue
            if next_hot_cue > 7:
                return assigned
            assigned[index] = str(next_hot_cue)
            next_hot_cue += 1
            break

    return assigned


def export_cues(tracks: list[TrackFeatures], out_path: str) -> None:
    root = ET.Element("DJ_PLAYLISTS", Version="1.0.0")
    ET.SubElement(
        root,
        "PRODUCT",
        Name="rekordbox",
        Version="6.0.0",
        Company="AlphaTheta",
    )
    collection = ET.SubElement(root, "COLLECTION", Entries=str(len(tracks)))

    for track_id, track in enumerate(tracks, start=1):
        track_element = ET.SubElement(
            collection,
            "TRACK",
            TrackID=str(track_id),
            Name=track.title or "",
            Artist=track.artist or "",
            Location=_location_for_path(track.path),
            AverageBpm=str(track.bpm),
            Tonality=track.key_camelot,
            TotalTime=str(int(track.duration)),
            Kind=_KIND,
        )

        hot_cues = _assign_hot_cues(track.phrases)
        for index, phrase in enumerate(track.phrases):
            ET.SubElement(
                track_element,
                "POSITION_MARK",
                Name=_position_mark_name(phrase),
                Type="0",
                Start=str(phrase.start),
                Num=hot_cues.get(index, "-1"),
            )

    tree = ET.ElementTree(root)
    destination = Path(out_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tree.write(destination, encoding="utf-8", xml_declaration=True)
