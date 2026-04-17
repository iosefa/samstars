"""Training entrypoint modules for samstars."""

from __future__ import annotations

from importlib import import_module

__all__ = ["chips", "records", "sam_training", "stardist_training"]


def __getattr__(name: str):
    """Lazily expose training entrypoint modules."""
    if name in __all__:
        return import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Return the public package surface for interactive use."""
    return sorted(__all__)
