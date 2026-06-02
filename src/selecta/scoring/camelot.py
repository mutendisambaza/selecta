"""Camelot wheel utilities for harmonic-mixing classification."""

from __future__ import annotations


def parse(code: str) -> tuple[int, str]:
    """Parse a Camelot code like ``8A`` into ``(8, "A")``."""
    if not isinstance(code, str):
        raise ValueError("Camelot code must be a string")

    value = code.strip().upper()
    if len(value) < 2:
        raise ValueError(f"Invalid Camelot code: {code!r}")

    letter = value[-1]
    number_part = value[:-1]

    if letter not in {"A", "B"} or not number_part.isdigit():
        raise ValueError(f"Invalid Camelot code: {code!r}")

    number = int(number_part)
    if not 1 <= number <= 12:
        raise ValueError(f"Invalid Camelot code: {code!r}")

    return number, letter


def relation(a: str, b: str) -> str:
    """Classify the harmonic relationship between two Camelot keys."""
    number_a, letter_a = parse(a)
    number_b, letter_b = parse(b)

    if (number_a, letter_a) == (number_b, letter_b):
        return "same"

    if number_a == number_b and letter_a != letter_b:
        return "relative"

    if letter_a == letter_b:
        delta = (number_b - number_a) % 12
        if delta in {1, 11}:
            return "adjacent"
        if delta == 7:
            return "energy_boost"

    return "distant"


def mood(a: str, b: str) -> str:
    """Classify the mood shift between two Camelot keys."""
    _, letter_a = parse(a)
    _, letter_b = parse(b)

    if letter_a == "A" and letter_b == "B":
        return "lift"
    if letter_a == "B" and letter_b == "A":
        return "darken"
    return "hold"
