"""Public package surface for samstars."""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "SamStars",
    "SamStarsResult",
    "train",
    "segment",
    "data",
    "segmentation",
    "training",
    "features",
    "utils",
    "CostSurfaceWeights",
    "SeedParams",
    "build_cost_surface",
    "write_cost_surface",
    "detect_peaks",
    "build_sparse_cost_graph",
    "canonical_seeds",
]

_SUBMODULES = {"data", "segmentation", "training", "features", "utils"}
_API_EXPORTS = {"SamStars", "SamStarsResult", "train", "segment"}
_FEATURE_EXPORTS = {
    "CostSurfaceWeights",
    "SeedParams",
    "build_cost_surface",
    "write_cost_surface",
    "detect_peaks",
    "build_sparse_cost_graph",
    "canonical_seeds",
}


def __getattr__(name: str):
    """Lazily expose documented package areas and feature helpers."""
    if name in _SUBMODULES:
        return import_module(f"{__name__}.{name}")
    if name in _API_EXPORTS:
        api = import_module(f"{__name__}.api")
        return getattr(api, name)
    if name in _FEATURE_EXPORTS:
        features = import_module(f"{__name__}.features")
        return getattr(features, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Return the public package surface for interactive use."""
    return sorted(__all__)
