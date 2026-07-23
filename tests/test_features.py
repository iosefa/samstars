"""Tests for deterministic feature helpers."""

from __future__ import annotations

import numpy as np
import pytest
from rasterio.transform import Affine

from samstars.features import CostSurfaceWeights, build_sparse_cost_graph


def test_cost_surface_weights_require_a_unit_sum() -> None:
    weights = CostSurfaceWeights()

    assert weights.grad + weights.gap + weights.tex + weights.slic == pytest.approx(1.0)
    with pytest.raises(ValueError, match="Weights must sum to 1"):
        CostSurfaceWeights(grad=0.5, gap=0.5, tex=0.5, slic=0.0)


def test_sparse_cost_graph_is_symmetric_and_respects_merge_radius() -> None:
    graph = build_sparse_cost_graph(
        xs=np.array([0.0, 1.0, 4.0]),
        ys=np.array([0.0, 0.0, 0.0]),
        cost=np.zeros((2, 5), dtype=np.float32),
        tfm=Affine.identity(),
        weight=0.0,
        xy_thresh=0.5,
        merge_radius=1.5,
    )

    np.testing.assert_array_equal(
        graph.toarray(),
        np.array(
            [
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        ),
    )


def test_sparse_cost_graph_applies_sampled_cost_to_distance() -> None:
    kwargs = {
        "xs": np.array([0.0, 2.0]),
        "ys": np.array([0.0, 0.0]),
        "tfm": Affine.identity(),
        "weight": 1.0,
        "xy_thresh": 0.5,
        "merge_radius": 2.5,
        "samples": 3,
    }

    low_cost = build_sparse_cost_graph(cost=np.zeros((1, 3)), **kwargs)
    high_cost = build_sparse_cost_graph(cost=np.ones((1, 3)), **kwargs)

    assert low_cost.nnz == 2
    assert high_cost.nnz == 0
