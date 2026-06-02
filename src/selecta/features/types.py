"""Core feature data types for Selecta."""

from __future__ import annotations

from dataclasses import dataclass, field


_PHRASE_LABELS = {"intro", "build", "drop", "breakdown", "outro", "body"}


@dataclass(frozen=True)
class Beat:
    time: float
    is_downbeat: bool

    def to_dict(self) -> dict[str, object]:
        return {"time": self.time, "is_downbeat": self.is_downbeat}

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> Beat:
        return cls(time=float(data["time"]), is_downbeat=bool(data["is_downbeat"]))


@dataclass(frozen=True)
class PhraseRegion:
    start: float
    end: float
    bars: int
    label: str
    confidence: float

    def __post_init__(self) -> None:
        if self.label not in _PHRASE_LABELS:
            allowed = ", ".join(sorted(_PHRASE_LABELS))
            raise ValueError(f"label must be one of: {allowed}")

    def to_dict(self) -> dict[str, object]:
        return {
            "start": self.start,
            "end": self.end,
            "bars": self.bars,
            "label": self.label,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> PhraseRegion:
        return cls(
            start=float(data["start"]),
            end=float(data["end"]),
            bars=int(data["bars"]),
            label=str(data["label"]),
            confidence=float(data["confidence"]),
        )


@dataclass(frozen=True)
class TrackFeatures:
    path: str
    duration: float
    bpm: float
    bpm_confidence: float
    beats: tuple[Beat, ...] = field(default_factory=tuple)
    key_camelot: str = ""
    key_confidence: float = 0.0
    energy_curve: tuple[float, ...] = field(default_factory=tuple)
    intensity: float = 0.0
    spectral: dict[str, dict[str, float]] = field(default_factory=dict)
    phrases: tuple[PhraseRegion, ...] = field(default_factory=tuple)
    title: str | None = None
    artist: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "beats", tuple(self.beats))
        object.__setattr__(self, "energy_curve", tuple(self.energy_curve))
        object.__setattr__(self, "phrases", tuple(self.phrases))
        object.__setattr__(
            self,
            "spectral",
            {
                str(section): {str(name): float(value) for name, value in bands.items()}
                for section, bands in self.spectral.items()
            },
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "duration": self.duration,
            "bpm": self.bpm,
            "bpm_confidence": self.bpm_confidence,
            "beats": [beat.to_dict() for beat in self.beats],
            "key_camelot": self.key_camelot,
            "key_confidence": self.key_confidence,
            "energy_curve": list(self.energy_curve),
            "intensity": self.intensity,
            "spectral": {
                section: dict(bands) for section, bands in self.spectral.items()
            },
            "phrases": [phrase.to_dict() for phrase in self.phrases],
            "title": self.title,
            "artist": self.artist,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> TrackFeatures:
        beats = tuple(Beat.from_dict(item) for item in data["beats"])
        phrases = tuple(PhraseRegion.from_dict(item) for item in data["phrases"])
        spectral = {
            str(section): {str(name): float(value) for name, value in bands.items()}
            for section, bands in dict[str, dict[str, object]](data["spectral"]).items()
        }
        return cls(
            path=str(data["path"]),
            duration=float(data["duration"]),
            bpm=float(data["bpm"]),
            bpm_confidence=float(data["bpm_confidence"]),
            beats=beats,
            key_camelot=str(data["key_camelot"]),
            key_confidence=float(data["key_confidence"]),
            energy_curve=tuple(float(value) for value in data["energy_curve"]),
            intensity=float(data["intensity"]),
            spectral=spectral,
            phrases=phrases,
            title=None if data.get("title") is None else str(data["title"]),
            artist=None if data.get("artist") is None else str(data["artist"]),
        )
