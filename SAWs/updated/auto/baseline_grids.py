#!/usr/bin/env python3
"""Deterministic histogram grids for single-chain athermal baselines.

For N=30, 44, and 60, the historical Rg grid is extended on the same lattice
rather than replaced.  The original edges are retained verbatim inside the new
array.  ``min_compact_rg`` supplies a compact-cluster reference with a full-bin
lower margin; the hard sample-range assertion is the definitive guard against
silent histogram loss.

JS divergences on an extended grid are not numerically comparable with values
computed on a legacy grid until the extended mass is summed back onto that
legacy grid.  ``run_chain_length_transfer.py`` also defaults to a separate
100-bin grid, so bin-wise comparisons require explicit reconciliation.
"""

from __future__ import annotations

import itertools
import math
from pathlib import Path
from typing import Tuple

import numpy as np


LEGACY_BASELINE_FILES = {
    30: "RnBaseline30.npz",
    44: "RnBaseline.npz",
    60: "RnBaseline60.npz",
}


def rod_rg(n_beads: int) -> float:
    """Return the exact Rg of an N-site straight lattice rod."""
    if n_beads < 1:
        raise ValueError("n_beads must be positive")
    return math.sqrt((n_beads * n_beads - 1.0) / 12.0)


def _points_rg(points: np.ndarray) -> float:
    centered = points - points.mean(axis=0)
    return float(np.sqrt(np.square(centered).sum(axis=1).mean()))


def min_compact_rg(n_beads: int) -> float:
    """Return the smallest compact-cluster reference over cubic centre classes.

    For each of the eight integer/half-integer centre classes, take the N lattice
    sites nearest that centre and compute their Rg about their own centroid.
    Boundary-shell ties are resolved lexicographically for reproducibility.
    """
    if n_beads < 1:
        raise ValueError("n_beads must be positive")
    radius = max(2, int(math.ceil(n_beads ** (1.0 / 3.0))) + 2)
    while True:
        points = list(itertools.product(range(-radius, radius + 1), repeat=3))
        candidates = []
        touched_boundary = False
        for centre in itertools.product((0.0, 0.5), repeat=3):
            ordered = sorted(
                points,
                key=lambda point: (
                    sum((point[i] - centre[i]) ** 2 for i in range(3)),
                    point,
                ),
            )[:n_beads]
            if any(max(abs(value) for value in point) == radius for point in ordered):
                touched_boundary = True
                break
            candidates.append(_points_rg(np.asarray(ordered, dtype=np.float64)))
        if not touched_boundary:
            return min(candidates)
        radius += 2


def legacy_rg_grid(
    n_beads: int, search_dir: Path | str
) -> Tuple[np.ndarray, float, float]:
    """Load and validate the historical Rg edges for a project chain length."""
    if n_beads not in LEGACY_BASELINE_FILES:
        raise ValueError(f"no legacy Rg grid is registered for N={n_beads}")
    path = Path(search_dir) / LEGACY_BASELINE_FILES[n_beads]
    with np.load(path, allow_pickle=False) as data:
        if "rg_edges" not in data.files:
            raise ValueError(f"{path} does not contain rg_edges")
        edges = np.asarray(data["rg_edges"], dtype=np.float64)
    if edges.ndim != 1 or edges.size < 2 or not np.all(np.isfinite(edges)):
        raise ValueError(f"invalid rg_edges in {path}")
    differences = np.diff(edges)
    if np.any(differences <= 0.0):
        raise ValueError(f"rg_edges are not strictly increasing in {path}")
    width = float(np.median(differences))
    if not np.allclose(differences, width, rtol=1e-10, atol=1e-12):
        raise ValueError(f"legacy rg_edges are not uniformly spaced in {path}")
    return edges.copy(), width, float(edges[0])


def fixed_rg_edges(n_beads: int, search_dir: Path | str) -> np.ndarray:
    """Extend the legacy grid to cover compact-reference through rod Rg."""
    legacy, width, _ = legacy_rg_grid(n_beads, search_dir)
    lower_target = min_compact_rg(n_beads) - width
    upper_target = rod_rg(n_beads) + width
    n_left = max(0, int(math.ceil((legacy[0] - lower_target) / width - 1e-12)))
    n_right = max(0, int(math.ceil((upper_target - legacy[-1]) / width - 1e-12)))
    left = legacy[0] - width * np.arange(n_left, 0, -1, dtype=np.float64)
    right = legacy[-1] + width * np.arange(1, n_right + 1, dtype=np.float64)
    edges = np.concatenate((left, legacy, right))
    if edges[0] > lower_target + 1e-12 or edges[-1] < upper_target - 1e-12:
        raise RuntimeError("failed to extend the legacy Rg grid to its target range")
    return edges


def fixed_c_edges(m_min: int, m_cover: int) -> np.ndarray:
    """Return unit-width contact edges for the complete declared window."""
    if m_cover < m_min:
        raise ValueError("m_cover must be at least m_min")
    return np.arange(m_min - 0.5, m_cover + 1.5, 1.0, dtype=np.float64)


def assert_within_grid(
    values: np.ndarray, edges: np.ndarray, name: str
) -> int:
    """Raise if any finite sample would be dropped by NumPy histogramming."""
    values = np.asarray(values, dtype=np.float64)
    edges = np.asarray(edges, dtype=np.float64)
    if edges.ndim != 1 or edges.size < 2 or not np.all(np.isfinite(edges)):
        raise ValueError(f"{name} grid edges are invalid")
    if np.any(np.diff(edges) <= 0.0):
        raise ValueError(f"{name} grid edges are not strictly increasing")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} samples contain non-finite values")
    outside = (values < edges[0]) | (values > edges[-1])
    count = int(outside.sum())
    if count:
        offenders = values[outside]
        raise ValueError(
            f"{count} {name} samples fall outside [{edges[0]:.12g}, "
            f"{edges[-1]:.12g}]; offender range "
            f"[{offenders.min():.12g}, {offenders.max():.12g}]"
        )
    return 0
