import pytest

from selecta.scoring.camelot import mood, parse, relation


def test_parse_valid_code():
    assert parse("8A") == (8, "A")


def test_parse_invalid_code_raises():
    with pytest.raises(ValueError):
        parse("bogus")


def test_relation_same_and_hold():
    assert relation("8A", "8A") == "same"
    assert mood("8A", "8A") == "hold"


def test_relation_adjacent_wraps_on_wheel():
    assert relation("8A", "9A") == "adjacent"
    assert relation("8A", "7A") == "adjacent"
    assert relation("12A", "1A") == "adjacent"


def test_relation_relative_and_mood_shift():
    assert relation("8A", "8B") == "relative"
    assert mood("8A", "8B") == "lift"
    assert mood("8B", "8A") == "darken"


def test_relation_energy_boost():
    assert relation("8A", "3A") == "energy_boost"


def test_relation_distant():
    assert relation("8A", "2A") == "distant"
