"""Segmentation interfaces for samstars."""

from __future__ import annotations

__all__ = ["run_segmentation"]


def __getattr__(name: str):
    """Lazily expose the segmentation entry point."""
    if name == "run_segmentation":
        from .engine import run_segmentation

        return run_segmentation
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Return the public package surface for interactive use."""
    return sorted(__all__)
