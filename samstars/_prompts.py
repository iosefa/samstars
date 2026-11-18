"""Shared prompt defaults and validation without scientific dependencies."""

from numbers import Integral

DEFAULT_FG_EXTRA = 80


def validate_fg_extra(value: int) -> int:
    """Return a nonnegative integer count, preserving zero as seed-only prompting."""
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError("fg_extra must be a nonnegative integer (0 uses only the seed as foreground).")
    return int(value)
