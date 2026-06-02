import math

from selecta.features.phrasing import detect_phrases
from selecta.features.types import Beat


def test_detect_phrases_produces_ordered_bar_aligned_regions_with_core_labels():
    duration = 64.0
    beat_spacing = 0.5
    beat_count = int(duration / beat_spacing)
    beats = tuple(
        Beat(time=index * beat_spacing, is_downbeat=index % 4 == 0)
        for index in range(beat_count)
    )
    energy_curve = tuple(
        [0.12] * 64
        + [0.25 + (0.35 * index / 63.0) for index in range(64)]
        + [0.95] * 64
        + [0.22] * 64
    )

    phrases = detect_phrases(beats, energy_curve, duration)

    assert phrases
    assert phrases[0].label == "intro"
    assert any(region.label == "drop" for region in phrases)
    assert phrases[-1].label == "outro"

    allowed_labels = {"intro", "build", "drop", "breakdown", "outro", "body"}
    downbeat_times = [beat.time for beat in beats if beat.is_downbeat]

    assert all(region.label in allowed_labels for region in phrases)
    assert all(0.0 <= region.confidence <= 1.0 for region in phrases)
    assert math.isclose(phrases[0].start, 0.0, abs_tol=1e-6)
    assert math.isclose(phrases[-1].end, duration, abs_tol=1e-6)

    for index, region in enumerate(phrases):
        assert region.start < region.end
        assert any(math.isclose(region.start, time, abs_tol=1e-6) for time in downbeat_times)
        if index:
            previous = phrases[index - 1]
            assert math.isclose(previous.end, region.start, abs_tol=1e-6)
            assert previous.start < region.start
