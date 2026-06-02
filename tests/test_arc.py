from selecta.sequencing.arc import arc_target, load_profile


def test_arc_target_interpolates_and_clamps_simple_profile():
    points = [(0.0, 0.0), (1.0, 1.0)]

    assert arc_target(0.5, points) == 0.5
    assert arc_target(0.0, points) == 0.0
    assert arc_target(1.0, points) == 1.0
    assert arc_target(-0.2, points) == 0.0


def test_load_profile_experiential_has_expected_shape():
    points = load_profile("experiential")

    assert points == [
        (0.0, 0.15),
        (0.25, 0.45),
        (0.45, 0.35),
        (0.7, 0.95),
        (0.85, 0.7),
        (1.0, 0.3),
    ]
    assert arc_target(0.0, points) < 0.25
    assert arc_target(0.7, points) > 0.85
    assert arc_target(1.0, points) < 0.45


def test_experiential_profile_includes_mid_set_dip_before_peak():
    points = load_profile("experiential")

    p1 = arc_target(0.25, points)
    p2 = arc_target(0.45, points)
    p3 = arc_target(0.7, points)

    assert p2 < p1
    assert p3 > p2


def test_load_profile_raises_for_missing_profile():
    try:
        load_profile("missing_profile")
    except KeyError as exc:
        assert exc.args == ("missing_profile",)
    else:
        raise AssertionError("Expected KeyError for missing arc profile.")
