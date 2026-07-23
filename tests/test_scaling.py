"""Tests for public array-scaling helpers."""

from __future__ import annotations

import numpy as np
import pytest

from samstars.data import scale_to_01, scale_to_u8_per_channel


def test_scale_to_01_clips_and_replaces_nonfinite_values() -> None:
    values = np.array(
        [-5.0, 0.0, 5.0, 10.0, 15.0, np.nan, np.inf, -np.inf],
        dtype=np.float64,
    )

    result = scale_to_01(values, 0.0, 10.0)

    np.testing.assert_array_equal(
        result,
        np.array([0.0, 0.0, 0.5, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )
    assert result.dtype == np.float32


@pytest.mark.parametrize(("lo", "hi"), [(1.0, 1.0), (2.0, 1.0), (np.nan, 1.0)])
def test_scale_to_01_returns_zeros_for_invalid_bounds(lo: float, hi: float) -> None:
    result = scale_to_01(np.array([1.0, 2.0], dtype=np.float32), lo, hi)

    np.testing.assert_array_equal(result, np.zeros(2, dtype=np.float32))


def test_scale_to_u8_per_channel_scales_each_channel_independently() -> None:
    image = np.array(
        [
            [[0.0, -1.0], [5.0, 1.0]],
            [[10.0, 3.0], [20.0, np.nan]],
        ],
        dtype=np.float32,
    )

    result = scale_to_u8_per_channel(image, [(0.0, 10.0), (1.0, 3.0)])

    expected = np.array(
        [
            [[0, 0], [127, 0]],
            [[255, 255], [255, 0]],
        ],
        dtype=np.uint8,
    )
    np.testing.assert_array_equal(result, expected)
    assert result.dtype == np.uint8


def test_scale_to_u8_per_channel_validates_shape_and_bounds() -> None:
    with pytest.raises(ValueError, match="expects HWC input"):
        scale_to_u8_per_channel(np.zeros((2, 2)), [(0.0, 1.0)])

    with pytest.raises(ValueError, match="lo_hi length < channels"):
        scale_to_u8_per_channel(np.zeros((2, 2, 2)), [(0.0, 1.0)])
