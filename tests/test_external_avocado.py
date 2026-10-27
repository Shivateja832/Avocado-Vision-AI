from __future__ import annotations

from validate_external_avocado import _source_stage, resolve_external_box


def test_external_source_stage_crosswalk_contains_only_supported_stages() -> None:
    assert _source_stage(4) == "unripe"
    assert _source_stage(5) == "ripe"
    assert _source_stage(6) == "ripe"
    assert _source_stage(7) == "overripe"
    assert _source_stage(0) is None


def test_landscape_box_uses_native_coordinates() -> None:
    assert resolve_external_box([100, 50, 200, 150], 640, 480) == (
        (100, 50, 300, 200),
        "native_landscape",
    )


def test_portrait_box_is_corrected_only_when_native_box_does_not_fit() -> None:
    assert resolve_external_box([264.714, 630.954, 2878.308, 1797.618], 3000, 4000) == (
        (198, 841, 2358, 3239),
        "portrait_aspect_corrected",
    )


def test_ambiguous_portrait_box_is_excluded() -> None:
    assert resolve_external_box([100, 100, 1000, 1000], 3000, 4000) is None


def test_invalid_portrait_box_is_excluded() -> None:
    assert resolve_external_box([3500, 3500, 2000, 2000], 3000, 4000) is None
