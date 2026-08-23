#!/usr/bin/env python3
"""Derive lattice contact-support requirements from molecular REMD targets.

The REMD contact coordinate is continuous.  After subtracting the fitted
chain-length-specific contact offset, a lattice level ``m`` covers the interval
``[m - 0.5, m + 0.5]``.  Tail masses in this module therefore count target bins
whose shifted centres lie above ``m + 0.5``.

This module only measures target support.  It does not import the fitter or
alter the athermal lattice measure.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

import numpy as np


# Sourced from config30.json, config2.json, and config60.json, respectively.
CONTACT_OFFSETS = {30: 29, 44: 43, 60: 59}
REPORT_THRESHOLDS = (1e-2, 1e-3, 1e-4, 1e-5)


@dataclass(frozen=True)
class TargetContactSupport:
    """Normalized molecular target distributions on a shifted contact axis."""

    temps: np.ndarray
    m_axis: np.ndarray
    P: np.ndarray


def load_target_contact_support(
    npz_path: Path | str, offset: float
) -> TargetContactSupport:
    """Load and row-normalize ``ct_hists`` after shifting ``ct_centers``."""
    path = Path(npz_path)
    with np.load(path, allow_pickle=False) as data:
        missing = {"temps", "ct_centers", "ct_hists"} - set(data.files)
        if missing:
            raise ValueError(f"{path} is missing required fields: {sorted(missing)}")
        temps = np.asarray(data["temps"], dtype=np.float64)
        centres = np.asarray(data["ct_centers"], dtype=np.float64)
        histograms = np.asarray(data["ct_hists"], dtype=np.float64)

    if temps.ndim != 1 or centres.ndim != 1 or histograms.ndim != 2:
        raise ValueError("temps and ct_centers must be 1D and ct_hists must be 2D")
    if histograms.shape != (temps.size, centres.size):
        raise ValueError(
            "ct_hists shape must be (len(temps), len(ct_centers)); "
            f"got {histograms.shape}, {(temps.size, centres.size)}"
        )
    if not (
        np.all(np.isfinite(temps))
        and np.all(np.isfinite(centres))
        and np.all(np.isfinite(histograms))
    ):
        raise ValueError("target support arrays contain non-finite values")
    if np.any(histograms < 0.0):
        raise ValueError("ct_hists contains negative values")
    row_sums = histograms.sum(axis=1)
    if np.any(row_sums <= 0.0):
        bad = np.flatnonzero(row_sums <= 0.0)
        raise ValueError(f"ct_hists has zero-mass temperature rows: {bad.tolist()}")

    return TargetContactSupport(
        temps=temps.copy(),
        m_axis=centres - float(offset),
        P=histograms / row_sums[:, None],
    )


def restrict_to_window(
    support: TargetContactSupport, m_max: int
) -> TargetContactSupport:
    """Return the target renormalized to the lattice-representable window.

    Target mass above ``m_max`` sits at contact numbers the lattice cannot
    realise at all, so a tail measured against the full target charges the flat
    tier for a remainder no contact level can hold.  ``support_report``
    already reports that out-of-window mass separately, so counting it in the
    tail counts it twice.
    """
    keep = support.m_axis <= float(m_max) + 0.5
    if not np.any(keep):
        raise ValueError(f"the target has no mass at or below m_max={m_max}")
    restricted = support.P[:, keep]
    row_sums = restricted.sum(axis=1)
    if np.any(row_sums <= 0.0):
        bad = np.flatnonzero(row_sums <= 0.0)
        raise ValueError(
            f"restricting to m <= {m_max} leaves zero-mass temperature rows: "
            f"{bad.tolist()}"
        )
    return TargetContactSupport(
        temps=support.temps.copy(),
        m_axis=support.m_axis[keep].copy(),
        P=restricted / row_sums[:, None],
    )


def _tail_by_temperature(support: TargetContactSupport, m: int) -> np.ndarray:
    return support.P[:, support.m_axis > float(m) + 0.5].sum(axis=1)


def tail_mass_above(support: TargetContactSupport, m: int) -> float:
    """Return the largest target mass above lattice level ``m`` over temperature."""
    return float(_tail_by_temperature(support, m).max())


def flat_level(
    support: TargetContactSupport, threshold: float, m_max: int
) -> int:
    """Return the smallest level whose worst-temperature tail is below threshold.

    If no level through ``m_max`` meets the threshold, ``m_max`` is returned.
    Callers that need to distinguish a genuine crossing from this geometrical
    clamp should use ``support_report()['flat_threshold_reached']``.
    """
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must lie in (0, 1)")
    if m_max < 0:
        raise ValueError("m_max must be nonnegative")
    for m in range(m_max + 1):
        if tail_mass_above(support, m) < threshold:
            return m
    return int(m_max)


def support_report(
    support: TargetContactSupport,
    m_max: int,
    thresholds: Iterable[float] = REPORT_THRESHOLDS,
) -> Dict[str, Any]:
    """Summarize target tails within and beyond a geometric contact window."""
    threshold_values = tuple(float(value) for value in thresholds)
    if not threshold_values:
        raise ValueError("at least one threshold is required")
    levels = np.arange(m_max + 1, dtype=np.int64)
    tail_curve = np.asarray(
        [tail_mass_above(support, int(m)) for m in levels], dtype=np.float64
    )
    by_threshold = {
        threshold: flat_level(support, threshold, m_max)
        for threshold in threshold_values
    }
    reached = {
        threshold: bool(tail_curve[level] < threshold)
        for threshold, level in by_threshold.items()
    }
    unsupported = _tail_by_temperature(support, m_max)
    argmax = int(np.argmax(unsupported))
    return {
        "m_flat_by_threshold": by_threshold,
        "flat_threshold_reached": reached,
        "unsupported_mass_at_m_max": {
            "mean": float(unsupported.mean()),
            "max": float(unsupported[argmax]),
            "argmax_T": float(support.temps[argmax]),
            "argmax_index": argmax,
        },
        "tail_curve": {"m": levels, "tail_mass": tail_curve},
        "target_support_exceeds_m_max": bool(np.any(unsupported > 0.0)),
    }


def _default_target_path(n_beads: int) -> Path:
    return Path(__file__).resolve().with_name(
        f"remd_distributions_{n_beads}mer.npz"
    )


def _default_m_max(n_beads: int) -> int:
    # Lazy import avoids a module cycle when the sampler imports this helper.
    from single_chain_wang_landau import geometric_contact_maximum

    value = geometric_contact_maximum(n_beads)
    if value is None:
        raise ValueError(
            f"no verified geometric maximum is encoded for N={n_beads}; pass --m_max"
        )
    return int(value)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report molecular target contact tails on the lattice contact axis.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--N", type=int, required=True, help="number of lattice beads")
    parser.add_argument("--npz", type=Path, default=None, help="target REMD NPZ")
    parser.add_argument(
        "--contact_offset", type=float, default=None,
        help="override the config-derived molecular-to-lattice contact offset",
    )
    parser.add_argument("--m_max", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.N < 1:
        raise ValueError("--N must be positive")
    if args.contact_offset is None:
        if args.N not in CONTACT_OFFSETS:
            raise ValueError(
                f"no contact offset is encoded for N={args.N}; pass --contact_offset"
            )
        offset = float(CONTACT_OFFSETS[args.N])
    else:
        offset = float(args.contact_offset)
    m_max = _default_m_max(args.N) if args.m_max is None else int(args.m_max)
    path = _default_target_path(args.N) if args.npz is None else args.npz
    support = load_target_contact_support(path, offset)
    report = support_report(support, m_max)

    print(f"Target contact support: N={args.N}, offset={offset:g}, m_max={m_max}")
    print("threshold       m_flat    threshold reached")
    for threshold in REPORT_THRESHOLDS:
        level = report["m_flat_by_threshold"][threshold]
        status = "yes" if report["flat_threshold_reached"][threshold] else "no (clamped)"
        print(f"{threshold:9.0e} {level:12d}    {status}")
    unsupported = report["unsupported_mass_at_m_max"]
    print(
        "unsupported target mass at m_max: "
        f"mean={100.0 * unsupported['mean']:.6f}%  "
        f"max={100.0 * unsupported['max']:.6f}%  "
        f"T={unsupported['argmax_T']:g} K"
    )
    print(
        "target support exceeds geometric m_max: "
        f"{report['target_support_exceeds_m_max']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
