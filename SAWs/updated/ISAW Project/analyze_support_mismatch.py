#!/usr/bin/env python3
"""Quantify probability-mass support mismatch between a molecular target and a
lattice baseline.

Motivation
----------
The contact-model fitter reweights a fixed athermal baseline P0(m):

    P_model(m|T) ∝ P0(m) * exp[-b(T) * m]

Reweighting can move mass around among the contact values where P0(m) > 0, but
it can NEVER create mass at a contact value where P0(m) = 0 — even if that value
lies strictly between the smallest and largest baseline contacts (an "internal
gap").  If the molecular target distribution places probability mass outside the
baseline's *positive-probability* support, that mass is unreachable by any choice
of b(T) and bounds the achievable fit quality from below.

This script measures exactly how much target mass is unsupported, for every
contact offset and temperature, separating:

    * mass below the baseline geometric range (m < baseline min);
    * mass inside bins where the baseline probability is positive (reachable);
    * mass in internal unsupported gaps (inside the range but P0(m) <= threshold);
    * mass above the baseline geometric range (m > baseline max);
    * total unsupported mass = below + internal-gap + above.

It reports BOTH the geometric support (fraction inside [min, max]) and the
positive-probability support (fraction at bins with P0 > threshold), because the
two differ exactly by the internal-gap mass.

Histogram conventions
---------------------
The contact rebinning, baseline parsing, and JSON encoding are mirrored
*verbatim* from ``fit_lattice_contact_model_chat.py`` so that this diagnostic
sees exactly the same integer-contact grid and baseline P0(m) the fitter uses.
If those functions change in the fitter, update the copies here to match.  The
mapping is mass-conserving piecewise-constant density integration; support is
never classified by comparing bin centers alone, and the target is never
truncated and renormalized before its missing mass is computed.

Usage
-----
  python analyze_support_mismatch.py \\
      --target remd_distributions_44mer.npz \\
      --baseline single_uniform_chain2_athermal_dists_joint_N44_T1_seed42.npz \\
      --contact-offsets 43,40 \\
      --rg-scale 0.46320503312590167 \\
      --outdir support_diagnostics

  python analyze_support_mismatch.py --quick-test

Dependencies: NumPy (required); Matplotlib (optional, only for plots).
Python 3.8.8 compatible.  Strict JSON (no NaN / Infinity).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:  # plotting is optional (skipped with --no-plots or when unavailable)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - depends on local plotting stack
    plt = None


# ===========================================================================
# Histogram helpers — mirrored verbatim from fit_lattice_contact_model_chat.py
# (keep in sync with the fitter; do not change the math independently).
# ===========================================================================

def centers_to_edges(centers: np.ndarray) -> np.ndarray:
    centers = np.asarray(centers, dtype=float)
    if centers.size < 2:
        raise ValueError("Need at least 2 centers to infer bin width")
    d = float(np.mean(np.diff(centers)))
    edges = np.empty(centers.size + 1, dtype=float)
    edges[:-1] = centers - 0.5 * d
    edges[-1] = centers[-1] + 0.5 * d
    return edges


def pdf_to_mass(pdf: np.ndarray, centers: np.ndarray) -> Tuple[np.ndarray, float]:
    """Convert a pdf sampled on evenly-spaced centers to probability mass per bin."""
    centers = np.asarray(centers, dtype=float)
    pdf = np.asarray(pdf, dtype=float)
    d = float(np.mean(np.diff(centers)))
    mass = pdf * d
    s = np.sum(mass)
    if s > 0:
        mass = mass / s
    return mass, d


def rebin_pdf_mass_to_integer_bins(
    ct_centers: np.ndarray,
    ct_pdf_row: np.ndarray,
    m_min: Optional[int] = None,
    m_max: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Rebin a contact pdf defined on ct_centers onto integer m bins.

    Treats ct_pdf_row[j] as piecewise-constant density on [edges[j], edges[j+1]).
    Returns (m_vals, p_mass) on integer contacts.
    """
    ct_centers = np.asarray(ct_centers, dtype=float)
    ct_pdf_row = np.asarray(ct_pdf_row, dtype=float)

    edges = centers_to_edges(ct_centers)
    if m_min is None:
        m_min = int(np.floor(edges[0] + 0.5))
    if m_max is None:
        m_max = int(np.ceil(edges[-1] - 0.5))
    if m_max < m_min:
        m_max = m_min

    m_vals = np.arange(m_min, m_max + 1, dtype=int)
    int_edges = np.arange(m_min - 0.5, m_max + 1.5, 1.0, dtype=float)

    native_mass, _ = pdf_to_mass(ct_pdf_row, ct_centers)
    native_edges = edges
    native_w = np.diff(native_edges)
    dens = np.zeros_like(native_mass)
    mask = native_w > 0
    dens[mask] = native_mass[mask] / native_w[mask]

    p_int = np.zeros(m_vals.size, dtype=float)
    j = 0
    for i in range(m_vals.size):
        a, b = float(int_edges[i]), float(int_edges[i + 1])
        while j < len(dens) and native_edges[j + 1] <= a:
            j += 1
        jj = j
        while jj < len(dens) and native_edges[jj] < b:
            left = max(a, native_edges[jj])
            right = min(b, native_edges[jj + 1])
            if right > left:
                p_int[i] += dens[jj] * (right - left)
            jj += 1

    s = p_int.sum()
    if s > 0:
        p_int /= s
    return m_vals.astype(float), p_int


def rebin_mass_between_edges(
    source_edges: np.ndarray,
    source_mass: np.ndarray,
    target_edges: np.ndarray,
) -> np.ndarray:
    """Rebin probability mass from arbitrary source_edges onto target_edges.

    Treats source_mass[j] as mass spread uniformly across [source_edges[j],
    source_edges[j+1]) (piecewise-constant density) and integrates the density
    over each target bin.  Returns probability mass per target bin, normalized
    to sum to 1.
    """
    source_edges = np.asarray(source_edges, dtype=float)
    source_mass = np.asarray(source_mass, dtype=float)
    target_edges = np.asarray(target_edges, dtype=float)

    if source_edges.ndim != 1 or target_edges.ndim != 1:
        raise ValueError("source_edges and target_edges must be 1D")
    if source_mass.ndim != 1:
        raise ValueError("source_mass must be 1D")
    if len(source_edges) != len(source_mass) + 1:
        raise ValueError("len(source_edges) must equal len(source_mass) + 1")
    if np.any(np.diff(source_edges) <= 0):
        raise ValueError("source_edges must be strictly increasing")
    if np.any(np.diff(target_edges) <= 0):
        raise ValueError("target_edges must be strictly increasing")

    out = _overlap_masses(source_edges, source_mass, target_edges)
    s = out.sum()
    if s > 0:
        out /= s
    return out


def _overlap_masses(
    source_edges: np.ndarray,
    source_mass: np.ndarray,
    target_edges: np.ndarray,
) -> np.ndarray:
    """Mass-conserving overlap integration WITHOUT renormalization.

    Treats ``source_mass[j]`` as uniform density over [source_edges[j],
    source_edges[j+1]) and integrates that density over each target bin.  Unlike
    :func:`rebin_mass_between_edges`, the result is NOT renormalized, so mass that
    falls outside ``target_edges`` is genuinely dropped (and therefore countable
    as "missing").  The sum over a target grid that fully covers the source equals
    the total source mass.
    """
    source_edges = np.asarray(source_edges, dtype=float)
    source_mass = np.asarray(source_mass, dtype=float)
    target_edges = np.asarray(target_edges, dtype=float)

    widths = np.diff(source_edges)
    dens = np.zeros_like(source_mass, dtype=float)
    mask = widths > 0
    dens[mask] = source_mass[mask] / widths[mask]

    out = np.zeros(len(target_edges) - 1, dtype=float)
    j = 0
    for i in range(len(out)):
        a, b = float(target_edges[i]), float(target_edges[i + 1])
        while j < len(dens) and source_edges[j + 1] <= a:
            j += 1
        jj = j
        while jj < len(dens) and source_edges[jj] < b:
            left = max(a, source_edges[jj])
            right = min(b, source_edges[jj + 1])
            if right > left:
                out[i] += dens[jj] * (right - left)
            jj += 1
    return out


def _overlap_below_inside_above(
    source_edges: np.ndarray,
    source_mass: np.ndarray,
    lo: float,
    hi: float,
) -> Tuple[float, float, float]:
    """Split source mass into (below lo, inside [lo, hi], above hi).

    Exact, mass-conserving partition of a piecewise-constant density.  The three
    returned values sum to the total source mass; nothing is renormalized.
    """
    edges = np.asarray(source_edges, dtype=float)
    mass = np.asarray(source_mass, dtype=float)
    widths = np.diff(edges)
    dens = np.zeros_like(mass, dtype=float)
    m = widths > 0
    dens[m] = mass[m] / widths[m]

    below = inside = above = 0.0
    for j in range(len(mass)):
        e0, e1 = float(edges[j]), float(edges[j + 1])
        if e1 <= e0:
            continue
        dj = dens[j]
        below += dj * max(0.0, min(e1, lo) - e0)
        above += dj * max(0.0, e1 - max(e0, hi))
        inside += dj * max(0.0, min(e1, hi) - max(e0, lo))
    return below, inside, above


def _validated_integer_contacts(values, label: str) -> np.ndarray:
    raw = np.asarray(values, dtype=float)
    if raw.ndim != 1 or raw.size == 0:
        raise ValueError(f"{label} must be a non-empty 1D array")
    if not np.all(np.isfinite(raw)):
        raise ValueError(f"{label} contains non-finite values")
    rounded = np.rint(raw)
    if not np.allclose(raw, rounded, rtol=0.0, atol=1e-8):
        raise ValueError(f"{label} must contain integer-valued contacts")
    out = rounded.astype(int)
    if np.unique(out).size != out.size:
        raise ValueError(f"{label} contains duplicate contact values")
    return out


def build_baseline_mass_on_integer(
    m_centers_int: np.ndarray, baseline_npz: str
) -> np.ndarray:
    """Return baseline probability mass p0(m) on the integer m grid.

    Supports the four baseline formats accepted by the fitter, in the same
    precedence order:
      A. (c_vals, c_prob)            discrete contact probabilities
      B. (c_vals, Pc)                REMD-style; Pc averaged over temperatures
      C. (c_edges, crg_prob)         joint P0(m, Rg) marginalized over Rg
      D. (ct_centers, ct_hists)      contact pdf on arbitrary even bins
    """
    b = np.load(baseline_npz)
    m_centers_int = np.asarray(m_centers_int, dtype=float)
    m0 = int(round(m_centers_int.min()))
    m1 = int(round(m_centers_int.max()))
    m_vals = np.arange(m0, m1 + 1, dtype=int)

    # Case A: discrete contact values with per-value probabilities
    if "c_vals" in b.files and "c_prob" in b.files:
        c_vals = _validated_integer_contacts(b["c_vals"], "baseline c_vals")
        c_prob = np.asarray(b["c_prob"], dtype=float)
        if c_vals.ndim != 1 or c_prob.ndim != 1 or c_vals.size != c_prob.size:
            raise ValueError(
                "baseline c_vals and c_prob must be 1D arrays of equal length"
            )
        if not np.all(np.isfinite(c_prob)):
            raise ValueError("baseline c_prob contains non-finite values")
        if np.any(c_prob < -1e-12):
            raise ValueError("baseline c_prob contains negative probability mass")
        c_prob = np.clip(c_prob, 0.0, None)
        if c_prob.sum() <= 0:
            raise ValueError("baseline c_prob sums to 0")
        c_prob /= c_prob.sum()
        p0 = np.zeros(m_vals.size, dtype=float)
        for cv, pk in zip(c_vals, c_prob):
            if m0 <= cv <= m1:
                p0[cv - m0] += pk
        if p0.sum() <= 0:
            raise ValueError("Baseline has no mass on the requested integer m range.")
        return p0 / p0.sum()

    # Case B: REMD-style multi-temperature histogram (c_vals, Pc)
    if "c_vals" in b.files and "Pc" in b.files:
        c_vals = _validated_integer_contacts(b["c_vals"], "baseline c_vals")
        Pc = np.asarray(b["Pc"], dtype=float)
        if c_vals.ndim != 1 or Pc.ndim not in (1, 2):
            raise ValueError("baseline c_vals must be 1D and Pc must be 1D or 2D")
        if Pc.shape[-1] != c_vals.size:
            raise ValueError(
                f"baseline Pc last dimension {Pc.shape[-1]} must match "
                f"len(c_vals)={c_vals.size}"
            )
        if not np.all(np.isfinite(Pc) | np.isnan(Pc)):
            raise ValueError("baseline Pc contains invalid values")
        if Pc.ndim == 2:
            c_prob = np.nanmean(Pc, axis=0)
        else:
            c_prob = Pc.copy()
        c_prob = np.nan_to_num(c_prob, nan=0.0)
        if np.any(c_prob < -1e-12):
            raise ValueError("baseline Pc contains negative probability mass")
        c_prob = np.clip(c_prob, 0.0, None)
        if c_prob.sum() <= 0:
            raise ValueError("baseline Pc sums to 0 after averaging")
        c_prob /= c_prob.sum()
        p0 = np.zeros(m_vals.size, dtype=float)
        for cv, pk in zip(c_vals, c_prob):
            if m0 <= cv <= m1:
                p0[cv - m0] += pk
        if p0.sum() <= 0:
            raise ValueError("Baseline Pc has no mass on the requested integer m range.")
        return p0 / p0.sum()

    # Case C: joint baseline P0(m, Rg) — marginalize over Rg to get p0(m)
    if "c_edges" in b.files and "crg_prob" in b.files:
        c_edges = np.asarray(b["c_edges"], dtype=float)
        crg_prob = np.asarray(b["crg_prob"], dtype=float)
        if c_edges.ndim != 1:
            raise ValueError(f"baseline c_edges must be 1D, got shape {c_edges.shape}")
        if crg_prob.ndim != 2:
            raise ValueError(f"baseline crg_prob must be 2D, got shape {crg_prob.shape}")
        if len(c_edges) != crg_prob.shape[0] + 1:
            raise ValueError(
                f"c_edges length must equal crg_prob.shape[0] + 1: "
                f"len(c_edges)={len(c_edges)}, crg_prob.shape[0]={crg_prob.shape[0]}"
            )
        p_c = crg_prob.sum(axis=1)  # marginalize over Rg
        p_c = np.clip(p_c, 0.0, None)
        if p_c.sum() <= 0:
            raise ValueError("baseline crg_prob marginal over Rg sums to 0")
        p_c /= p_c.sum()
        int_edges = np.arange(m0 - 0.5, m1 + 1.5, 1.0, dtype=float)
        p0 = rebin_mass_between_edges(c_edges, p_c, int_edges)
        if p0.sum() <= 0:
            raise ValueError("Joint baseline has no mass on the requested integer m range.")
        return p0

    # Case D: baseline provided as contact pdf on arbitrary bins
    if "ct_centers" in b.files and "ct_hists" in b.files:
        ccent = np.asarray(b["ct_centers"], dtype=float)
        ch = np.asarray(b["ct_hists"], dtype=float)
        if ccent.ndim != 1 or ccent.size < 2 or not np.all(np.diff(ccent) > 0):
            raise ValueError("baseline ct_centers must be 1D and strictly increasing")
        if not np.allclose(np.diff(ccent), np.diff(ccent)[0], rtol=1e-6, atol=1e-10):
            raise ValueError("baseline ct_centers must be evenly spaced")
        if ch.ndim == 2:
            ch = ch[0]
        if ch.ndim != 1 or ch.size != ccent.size:
            raise ValueError(
                "baseline ct_hists must resolve to a 1D row matching ct_centers"
            )
        if not np.all(np.isfinite(ch)) or np.any(ch < 0) or ch.sum() <= 0:
            raise ValueError("baseline ct_hists row must be finite, nonnegative, and nonempty")
        _, p0 = rebin_pdf_mass_to_integer_bins(ccent, ch, m_min=m0, m_max=m1)
        return p0

    raise ValueError(
        "Unrecognized baseline format. Supported: "
        "(c_vals,c_prob), (c_vals,Pc), (c_edges,crg_prob), (ct_centers,ct_hists)."
    )


def _get_baseline_integer_range(baseline_npz: str) -> Tuple[int, int]:
    """Return (min, max) integer contact range of the baseline without building p0."""
    b = np.load(baseline_npz)
    if "c_vals" in b.files:
        c_vals = _validated_integer_contacts(b["c_vals"], "baseline c_vals")
        return int(c_vals.min()), int(c_vals.max())
    if "c_edges" in b.files:
        c_edges = np.asarray(b["c_edges"], dtype=float)
        if c_edges.ndim != 1:
            raise ValueError(f"baseline c_edges must be 1D, got shape {c_edges.shape}")
        if len(c_edges) < 2:
            raise ValueError(f"baseline c_edges must have at least 2 entries, got {len(c_edges)}")
        return (
            int(np.floor(c_edges[0] + 0.5)),
            int(np.ceil(c_edges[-1] - 0.5)),
        )
    if "ct_centers" in b.files:
        ccent = np.asarray(b["ct_centers"], dtype=float)
        edges = centers_to_edges(ccent)
        return int(np.floor(edges[0] + 0.5)), int(np.ceil(edges[-1] - 0.5))
    return 0, 0


class _NpEncoder(json.JSONEncoder):
    """JSON encoder converting numpy types to plain Python; non-finite -> None."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            value = float(obj)
            return value if np.isfinite(value) else None
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def _jsafe(x: Optional[float]) -> Optional[float]:
    """Map non-finite floats to None so strict JSON (allow_nan=False) succeeds."""
    if x is None:
        return None
    xf = float(x)
    return xf if np.isfinite(xf) else None


# ===========================================================================
# Core support classification
# ===========================================================================

def classify_contact_row(
    p_target_row: np.ndarray,
    m_grid: np.ndarray,
    p0: np.ndarray,
    bl_min: int,
    bl_max: int,
    threshold: float,
) -> Dict[str, float]:
    """Classify one temperature's target contact mass against baseline support.

    ``p_target_row`` and ``p0`` are probability mass on the same integer
    ``m_grid``.  Returns a dict of mass components that partition the target:
    below + inside_positive + internal_gap + above == sum(p_target_row).
    """
    m_grid = np.asarray(m_grid, dtype=float)
    p = np.asarray(p_target_row, dtype=float)
    p0 = np.asarray(p0, dtype=float)

    below_mask = m_grid < bl_min
    above_mask = m_grid > bl_max
    inside_mask = (~below_mask) & (~above_mask)
    positive_mask = inside_mask & (p0 > threshold)
    gap_mask = inside_mask & (p0 <= threshold)

    mass_below = float(p[below_mask].sum())
    mass_above = float(p[above_mask].sum())
    mass_inside_positive = float(p[positive_mask].sum())
    mass_internal_gap = float(p[gap_mask].sum())
    total_unsupported = mass_below + mass_internal_gap + mass_above
    geometric_support = float(p[inside_mask].sum())
    positive_support = mass_inside_positive
    mass_negative = float(p[m_grid < 0].sum())

    total = float(p.sum())
    if total > 0:
        mean_m = float((m_grid * p).sum() / total)
        var_m = float((p * (m_grid - mean_m) ** 2).sum() / total)
        std_m = float(np.sqrt(max(0.0, var_m)))
    else:
        mean_m = float("nan")
        std_m = float("nan")

    return {
        "mass_below": mass_below,
        "mass_inside_positive": mass_inside_positive,
        "mass_internal_gap": mass_internal_gap,
        "mass_above": mass_above,
        "total_unsupported": total_unsupported,
        "geometric_support": geometric_support,
        "positive_support": positive_support,
        "mass_negative": mass_negative,
        "target_mean_shifted": mean_m,
        "target_std_shifted": std_m,
    }


def build_target_on_grid(
    ct_centers_native: np.ndarray,
    ct_pdf_row: np.ndarray,
    g_min: int,
    g_max: int,
) -> np.ndarray:
    """Rebin a target contact pdf row onto the integer union grid [g_min, g_max].

    Uses the fitter's mass-conserving integer rebinning.  When the grid covers
    the full native support, normalization is a no-op and no mass is lost.
    """
    _, p = rebin_pdf_mass_to_integer_bins(
        ct_centers_native, ct_pdf_row, m_min=g_min, m_max=g_max
    )
    return p


# ===========================================================================
# Target / baseline loading
# ===========================================================================

def load_target(path: str) -> Dict[str, Any]:
    d = np.load(path)
    temp_key = "temps" if "temps" in d.files else "Ts" if "Ts" in d.files else None
    if temp_key is None:
        raise ValueError(
            f"target file {path!r} contains neither 'temps' nor 'Ts' (found: {list(d.files)})"
        )
    temps = np.asarray(d[temp_key], dtype=float)
    if "ct_centers" not in d.files or "ct_hists" not in d.files:
        raise ValueError(
            f"target file {path!r} must contain 'ct_centers' and 'ct_hists' "
            f"(found: {list(d.files)})"
        )
    ct_centers = np.asarray(d["ct_centers"], dtype=float)
    ct_hists = np.asarray(d["ct_hists"], dtype=float)

    if temps.ndim != 1:
        raise ValueError(f"temps must be 1D, got shape {temps.shape}")
    if ct_centers.ndim != 1 or ct_centers.size < 2:
        raise ValueError(f"ct_centers must be 1D with >= 2 entries, got {ct_centers.shape}")
    if ct_hists.ndim != 2:
        raise ValueError(f"ct_hists must be 2D (n_temps, n_bins), got {ct_hists.shape}")
    if ct_hists.shape[0] != temps.size:
        raise ValueError("ct_hists rows must match number of temperatures")
    if ct_hists.shape[1] != ct_centers.size:
        raise ValueError("ct_hists columns must match ct_centers")
    if not np.all(np.isfinite(temps)) or not np.all(temps > 0):
        raise ValueError("temps must be finite and positive")
    if not np.all(np.diff(temps) > 0):
        raise ValueError("temps must be strictly increasing")
    if not np.all(np.isfinite(ct_centers)) or not np.all(np.diff(ct_centers) > 0):
        raise ValueError("ct_centers must be finite and strictly increasing")
    steps = np.diff(ct_centers)
    if not np.allclose(steps, steps[0], rtol=1e-6, atol=1e-10):
        raise ValueError("ct_centers must be evenly spaced (interpreted as a density grid)")
    if not np.all(np.isfinite(ct_hists)) or np.any(ct_hists < 0):
        raise ValueError("ct_hists must be finite and nonnegative")
    if np.any(ct_hists.sum(axis=1) <= 0):
        raise ValueError("every ct_hists row must have positive total density")

    rg_centers_key = next((k for k in ("rg_centers", "Rg_centers") if k in d.files), None)
    rg_hists_key = next((k for k in ("rg_hists", "Rg_hists") if k in d.files), None)
    rg_centers = rg_hists = None
    if rg_centers_key is not None and rg_hists_key is not None:
        rg_centers = np.asarray(d[rg_centers_key], dtype=float)
        rg_hists = np.asarray(d[rg_hists_key], dtype=float)
        if rg_centers.ndim != 1 or rg_centers.size < 2:
            raise ValueError(f"{rg_centers_key} must be 1D with >= 2 entries")
        if rg_hists.ndim != 2 or rg_hists.shape[0] != temps.size or rg_hists.shape[1] != rg_centers.size:
            raise ValueError(f"{rg_hists_key} must be (n_temps, n_rg_bins) matching {rg_centers_key}")
        if not np.all(np.isfinite(rg_centers)) or not np.all(np.diff(rg_centers) > 0):
            raise ValueError(f"{rg_centers_key} must be finite and strictly increasing")
        rsteps = np.diff(rg_centers)
        if not np.allclose(rsteps, rsteps[0], rtol=1e-6, atol=1e-10):
            raise ValueError(f"{rg_centers_key} must be evenly spaced")
        if not np.all(np.isfinite(rg_hists)) or np.any(rg_hists < 0):
            raise ValueError(f"{rg_hists_key} must be finite and nonnegative")
        if np.any(rg_hists.sum(axis=1) <= 0):
            raise ValueError(f"every {rg_hists_key} row must have positive total density")

    return {
        "temps": temps,
        "ct_centers": ct_centers,
        "ct_hists": ct_hists,
        "rg_centers": rg_centers,
        "rg_hists": rg_hists,
        "temp_key": temp_key,
        "rg_centers_key": rg_centers_key,
        "rg_hists_key": rg_hists_key,
        "keys": list(d.files),
    }


def load_baseline_rg(path: str, rg_scale: float) -> Optional[Dict[str, Any]]:
    """Return scaled baseline Rg marginal from a joint baseline, else None."""
    b = np.load(path)
    if not all(k in b.files for k in ("c_edges", "rg_edges", "crg_prob")):
        return None
    rg_edges_lat = np.asarray(b["rg_edges"], dtype=float)
    crg_prob = np.asarray(b["crg_prob"], dtype=float)
    if rg_edges_lat.ndim != 1 or rg_edges_lat.size < 2:
        raise ValueError("baseline rg_edges must be 1D with >= 2 entries")
    if crg_prob.ndim != 2 or crg_prob.shape[1] != rg_edges_lat.size - 1:
        raise ValueError("baseline crg_prob columns must match rg_edges")
    if np.any(np.diff(rg_edges_lat) <= 0):
        raise ValueError("baseline rg_edges must be strictly increasing")
    p_rg = crg_prob.sum(axis=0)  # marginalize over contacts
    p_rg = np.clip(p_rg, 0.0, None)
    if p_rg.sum() <= 0:
        raise ValueError("baseline crg_prob marginal over contacts sums to 0")
    p_rg = p_rg / p_rg.sum()
    rg_edges_scaled = rg_scale * rg_edges_lat
    return {
        "rg_edges_lattice": rg_edges_lat,
        "rg_edges_scaled": rg_edges_scaled,
        "p_rg": p_rg,
    }


# ===========================================================================
# Per-offset and Rg analysis
# ===========================================================================

def analyze_offset(
    target: Dict[str, Any],
    baseline_path: str,
    offset: float,
    bl_min: int,
    bl_max: int,
    threshold: float,
) -> Dict[str, Any]:
    """Full contact support analysis for a single offset, over all temperatures."""
    temps = target["temps"]
    ct_centers_native = target["ct_centers"] - float(offset)
    ct_hists = target["ct_hists"]

    native_edges = centers_to_edges(ct_centers_native)
    t_min = int(np.floor(native_edges[0] + 0.5))
    t_max = int(np.ceil(native_edges[-1] - 0.5))
    g_min = min(t_min, bl_min)
    g_max = max(t_max, bl_max)
    m_grid = np.arange(g_min, g_max + 1, dtype=float)

    p0 = build_baseline_mass_on_integer(m_grid, baseline_path)

    n_t = temps.size
    p_target = np.zeros((n_t, m_grid.size), dtype=float)
    rows: List[Dict[str, float]] = []
    for i in range(n_t):
        p_target[i] = build_target_on_grid(ct_centers_native, ct_hists[i], g_min, g_max)
        rows.append(classify_contact_row(p_target[i], m_grid, p0, bl_min, bl_max, threshold))

    def col(name: str) -> np.ndarray:
        return np.array([r[name] for r in rows], dtype=float)

    components = [
        "mass_below",
        "mass_inside_positive",
        "mass_internal_gap",
        "mass_above",
        "total_unsupported",
        "geometric_support",
        "positive_support",
        "mass_negative",
        "target_mean_shifted",
        "target_std_shifted",
    ]
    per_temp = {name: col(name) for name in components}

    total_unsup = per_temp["total_unsupported"]
    neg = per_temp["mass_negative"]
    agg = {
        "mean_total_unsupported": float(np.mean(total_unsup)),
        "max_total_unsupported": float(np.max(total_unsup)),
        "min_total_unsupported": float(np.min(total_unsup)),
        "mean_mass_below": float(np.mean(per_temp["mass_below"])),
        "mean_mass_above": float(np.mean(per_temp["mass_above"])),
        "mean_mass_internal_gap": float(np.mean(per_temp["mass_internal_gap"])),
        "mean_positive_support": float(np.mean(per_temp["positive_support"])),
        "mean_geometric_support": float(np.mean(per_temp["geometric_support"])),
        "mean_negative_mass": float(np.mean(neg)),
        "max_negative_mass": float(np.max(neg)),
        "has_negative_support": bool(np.max(neg) > 1e-12),
    }

    # Mass-conservation invariant (each temperature partitions to 1.0).
    partition = (
        per_temp["mass_below"]
        + per_temp["mass_inside_positive"]
        + per_temp["mass_internal_gap"]
        + per_temp["mass_above"]
    )
    max_partition_err = float(np.max(np.abs(partition - 1.0)))

    return {
        "offset": float(offset),
        "t_min": int(t_min),
        "t_max": int(t_max),
        "g_min": int(g_min),
        "g_max": int(g_max),
        "m_grid": m_grid,
        "p0": p0,
        "p_target": p_target,
        "per_temp": per_temp,
        "aggregate": agg,
        "max_partition_err": max_partition_err,
    }


def analyze_rg(
    target: Dict[str, Any],
    baseline_rg: Dict[str, Any],
    threshold: float,
) -> Dict[str, Any]:
    """Rg support analysis (offset-independent), over all temperatures."""
    temps = target["temps"]
    rg_centers = target["rg_centers"]
    rg_hists = target["rg_hists"]
    rg_edges_scaled = baseline_rg["rg_edges_scaled"]
    p0_rg = baseline_rg["p_rg"]
    lo = float(rg_edges_scaled[0])
    hi = float(rg_edges_scaled[-1])

    target_edges = centers_to_edges(rg_centers)
    positive_bins = p0_rg > threshold

    n_t = temps.size
    below = np.zeros(n_t)
    inside = np.zeros(n_t)
    above = np.zeros(n_t)
    gap = np.zeros(n_t)
    inside_positive = np.zeros(n_t)
    total_unsup = np.zeros(n_t)
    rg_target_mass = np.zeros((n_t, p0_rg.size), dtype=float)

    for i in range(n_t):
        tmass, _ = pdf_to_mass(rg_hists[i], rg_centers)
        b, ins, a = _overlap_below_inside_above(target_edges, tmass, lo, hi)
        per_bin = _overlap_masses(target_edges, tmass, rg_edges_scaled)
        g = float(per_bin[~positive_bins].sum())
        below[i] = b
        inside[i] = ins
        above[i] = a
        gap[i] = g
        inside_positive[i] = ins - g
        total_unsup[i] = b + g + a
        rg_target_mass[i] = per_bin

    per_temp = {
        "rg_mass_below": below,
        "rg_mass_inside": inside,
        "rg_mass_inside_positive": inside_positive,
        "rg_mass_internal_gap": gap,
        "rg_mass_above": above,
        "rg_total_unsupported": total_unsup,
    }
    agg = {
        "mean_rg_total_unsupported": float(np.mean(total_unsup)),
        "max_rg_total_unsupported": float(np.max(total_unsup)),
        "mean_rg_below": float(np.mean(below)),
        "mean_rg_above": float(np.mean(above)),
        "mean_rg_internal_gap": float(np.mean(gap)),
        "mean_rg_inside": float(np.mean(inside)),
    }
    return {
        "rg_edges_scaled": rg_edges_scaled,
        "rg_edges_lattice": baseline_rg["rg_edges_lattice"],
        "p0_rg": p0_rg,
        "rg_range_scaled": [lo, hi],
        "rg_target_mass": rg_target_mass,
        "per_temp": per_temp,
        "aggregate": agg,
    }


# ===========================================================================
# Output writers
# ===========================================================================

def offset_label(offset: float) -> str:
    if float(offset).is_integer():
        return str(int(offset))
    return ("%g" % offset).replace("-", "m").replace(".", "p")


def write_csv(
    path: Path,
    target: Dict[str, Any],
    results: List[Dict[str, Any]],
    rg_result: Optional[Dict[str, Any]],
) -> None:
    temps = target["temps"]
    header = [
        "offset",
        "temp_index",
        "temperature",
        "mass_below",
        "mass_inside_positive",
        "mass_internal_gap",
        "mass_above",
        "total_unsupported",
        "geometric_support",
        "positive_support",
        "mass_negative",
        "target_mean_shifted",
        "target_std_shifted",
        "rg_mass_below",
        "rg_mass_inside",
        "rg_mass_internal_gap",
        "rg_mass_above",
        "rg_total_unsupported",
    ]

    def fmt(x: float) -> str:
        xf = float(x)
        return ("%.10g" % xf) if np.isfinite(xf) else ""

    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for res in results:
            pt = res["per_temp"]
            for i in range(temps.size):
                if rg_result is not None:
                    rp = rg_result["per_temp"]
                    rg_vals = [
                        fmt(rp["rg_mass_below"][i]),
                        fmt(rp["rg_mass_inside"][i]),
                        fmt(rp["rg_mass_internal_gap"][i]),
                        fmt(rp["rg_mass_above"][i]),
                        fmt(rp["rg_total_unsupported"][i]),
                    ]
                else:
                    rg_vals = ["", "", "", "", ""]
                w.writerow(
                    [
                        fmt(res["offset"]),
                        i,
                        fmt(temps[i]),
                        fmt(pt["mass_below"][i]),
                        fmt(pt["mass_inside_positive"][i]),
                        fmt(pt["mass_internal_gap"][i]),
                        fmt(pt["mass_above"][i]),
                        fmt(pt["total_unsupported"][i]),
                        fmt(pt["geometric_support"][i]),
                        fmt(pt["positive_support"][i]),
                        fmt(pt["mass_negative"][i]),
                        fmt(pt["target_mean_shifted"][i]),
                        fmt(pt["target_std_shifted"][i]),
                    ]
                    + rg_vals
                )


def select_best_offsets(
    results: List[Dict[str, Any]],
    rg_result: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    offsets = [r["offset"] for r in results]

    def argmin_by(key: str) -> float:
        vals = [r["aggregate"][key] for r in results]
        return offsets[int(np.argmin(vals))]

    best_mean = argmin_by("mean_total_unsupported")
    best_max = argmin_by("max_total_unsupported")

    no_neg_offsets = [
        r["offset"] for r in results if not r["aggregate"]["has_negative_support"]
    ]
    if no_neg_offsets:
        # Among offsets with no negative support, prefer least mean unsupported.
        sub = [r for r in results if not r["aggregate"]["has_negative_support"]]
        best_no_neg = sub[int(np.argmin([r["aggregate"]["mean_total_unsupported"] for r in sub]))]["offset"]
    else:
        best_no_neg = None

    rg_note = None
    best_rg = None
    if rg_result is not None:
        # Rg support is offset-independent; all offsets tie under this criterion.
        best_rg = offsets[0]
        rg_note = (
            "Rg support does not depend on the contact offset; all offsets are "
            "equivalent under this criterion."
        )

    return {
        "smallest_mean_unsupported_contact_mass": _jsafe(best_mean),
        "smallest_max_unsupported_contact_mass": _jsafe(best_max),
        "no_negative_shifted_contact_support": _jsafe(best_no_neg),
        "no_negative_support_candidates": [_jsafe(o) for o in no_neg_offsets],
        "best_rg_support": _jsafe(best_rg),
        "best_rg_support_note": rg_note,
    }


def write_summary_json(
    path: Path,
    args: argparse.Namespace,
    target: Dict[str, Any],
    bl_min: int,
    bl_max: int,
    p0_gap_ints: List[int],
    results: List[Dict[str, Any]],
    rg_result: Optional[Dict[str, Any]],
    best: Dict[str, Any],
) -> None:
    per_offset = []
    for r in results:
        agg = {k: _jsafe(v) if isinstance(v, float) else v for k, v in r["aggregate"].items()}
        per_offset.append(
            {
                "offset": _jsafe(r["offset"]),
                "target_integer_range": [r["t_min"], r["t_max"]],
                "union_grid_range": [r["g_min"], r["g_max"]],
                "max_partition_error": _jsafe(r["max_partition_err"]),
                "aggregate": agg,
            }
        )

    rg_block: Optional[Dict[str, Any]] = None
    if rg_result is not None:
        rg_agg = {k: _jsafe(v) for k, v in rg_result["aggregate"].items()}
        rg_block = {
            "rg_scale": _jsafe(args.rg_scale),
            "rg_range_scaled": [_jsafe(rg_result["rg_range_scaled"][0]), _jsafe(rg_result["rg_range_scaled"][1])],
            "offset_independent": True,
            "aggregate": rg_agg,
        }

    summary = {
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "target_path": str(args.target),
        "baseline_path": str(args.baseline),
        "target_keys": target["keys"],
        "n_temps": int(target["temps"].size),
        "temperature_range": [float(target["temps"].min()), float(target["temps"].max())],
        "contact_offsets": [_jsafe(o) for o in args.contact_offsets],
        "rg_scale": _jsafe(args.rg_scale),
        "positive_support_threshold": _jsafe(args.positive_support_threshold),
        "baseline_integer_range": [int(bl_min), int(bl_max)],
        "baseline_internal_gap_contacts": [int(g) for g in p0_gap_ints],
        "has_rg_analysis": rg_result is not None,
        "per_offset": per_offset,
        "rg": rg_block,
        "best_offsets": best,
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, cls=_NpEncoder, allow_nan=False)


def write_report_md(
    path: Path,
    args: argparse.Namespace,
    target: Dict[str, Any],
    bl_min: int,
    bl_max: int,
    p0_gap_ints: List[int],
    results: List[Dict[str, Any]],
    rg_result: Optional[Dict[str, Any]],
    best: Dict[str, Any],
) -> None:
    L: List[str] = []
    L.append("# Support-mismatch diagnostic report")
    L.append("")
    L.append(f"- Target: `{args.target}`")
    L.append(f"- Baseline: `{args.baseline}`")
    L.append(f"- Temperatures: {target['temps'].size} in "
             f"[{target['temps'].min():.4g}, {target['temps'].max():.4g}]")
    L.append(f"- Contact offsets: {', '.join(str(o) for o in args.contact_offsets)}")
    L.append(f"- Positive-support threshold: {args.positive_support_threshold:g}")
    L.append(f"- Baseline integer contact range: [{bl_min}, {bl_max}]")
    if p0_gap_ints:
        L.append(f"- Baseline internal gaps (P0 <= threshold inside range): "
                 f"{', '.join(str(g) for g in p0_gap_ints)}")
    else:
        L.append("- Baseline internal gaps: none")
    L.append("")
    L.append("Reweighting `P_model(m|T) ∝ P0(m)·exp[-b(T)·m]` cannot place mass at "
             "any contact where `P0(m)=0`. Target mass that is below the baseline "
             "range, above it, or in an internal gap is therefore unreachable by "
             "any `b(T)` and bounds the fit error from below.")
    L.append("")

    L.append("## Per-offset contact support (averaged over temperatures)")
    L.append("")
    L.append("| offset | mean unsup | max unsup | mean below | mean gap | mean above | "
             "mean pos-support | mean geom-support | max neg mass |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for r in results:
        a = r["aggregate"]
        L.append(
            "| {off:g} | {mu:.4g} | {mx:.4g} | {mb:.4g} | {mg:.4g} | {ma:.4g} | "
            "{ps:.4g} | {gs:.4g} | {nz:.4g} |".format(
                off=r["offset"],
                mu=a["mean_total_unsupported"],
                mx=a["max_total_unsupported"],
                mb=a["mean_mass_below"],
                mg=a["mean_mass_internal_gap"],
                ma=a["mean_mass_above"],
                ps=a["mean_positive_support"],
                gs=a["mean_geometric_support"],
                nz=a["max_negative_mass"],
            )
        )
    L.append("")
    L.append("`mean geom-support` − `mean pos-support` equals the internal-gap mass: "
             "geometric support counts everything inside [min, max]; positive support "
             "counts only bins with `P0 > threshold`.")
    L.append("")

    if rg_result is not None:
        ra = rg_result["aggregate"]
        lo, hi = rg_result["rg_range_scaled"]
        L.append("## Rg support (offset-independent)")
        L.append("")
        L.append(f"- Scaled baseline Rg range: [{lo:.4g}, {hi:.4g}]  "
                 f"(rg_scale = {args.rg_scale:g})")
        L.append(f"- Mean total unsupported Rg mass: {ra['mean_rg_total_unsupported']:.4g}")
        L.append(f"- Max total unsupported Rg mass: {ra['max_rg_total_unsupported']:.4g}")
        L.append(f"- Mean below / internal-gap / above: "
                 f"{ra['mean_rg_below']:.4g} / {ra['mean_rg_internal_gap']:.4g} / "
                 f"{ra['mean_rg_above']:.4g}")
        L.append("")
    else:
        L.append("## Rg support")
        L.append("")
        L.append("Rg analysis skipped (target Rg histograms and/or a joint baseline "
                 "were not both available).")
        L.append("")

    L.append("## Best offset by criterion (reported separately)")
    L.append("")
    L.append(f"- **Smallest mean unsupported contact mass:** "
             f"{_fmt_opt(best['smallest_mean_unsupported_contact_mass'])}")
    L.append(f"- **Smallest maximum unsupported contact mass:** "
             f"{_fmt_opt(best['smallest_max_unsupported_contact_mass'])}")
    if best["no_negative_shifted_contact_support"] is not None:
        L.append(f"- **No negative shifted-contact support:** "
                 f"{_fmt_opt(best['no_negative_shifted_contact_support'])} "
                 f"(candidates with zero negative mass: "
                 f"{', '.join(_fmt_opt(o) for o in best['no_negative_support_candidates'])})")
    else:
        L.append("- **No negative shifted-contact support:** none — every offset "
                 "places some target mass at negative shifted contacts.")
    if rg_result is not None:
        L.append(f"- **Best Rg support:** {_fmt_opt(best['best_rg_support'])} "
                 f"({best['best_rg_support_note']})")
    else:
        L.append("- **Best Rg support:** n/a (no Rg analysis).")
    L.append("")
    L.append("These criteria are intentionally not combined into a single score; "
             "choose the offset that matches your modeling priority.")
    L.append("")

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))


def _fmt_opt(x: Optional[float]) -> str:
    if x is None:
        return "none"
    return "%g" % float(x)


def write_npz(
    path: Path,
    target: Dict[str, Any],
    results: List[Dict[str, Any]],
    rg_result: Optional[Dict[str, Any]],
) -> None:
    out: Dict[str, Any] = {"temps": target["temps"]}
    for r in results:
        lab = offset_label(r["offset"])
        out[f"offset_{lab}_value"] = np.array(r["offset"], dtype=float)
        out[f"m_grid_off{lab}"] = r["m_grid"]
        out[f"p0_off{lab}"] = r["p0"]
        out[f"p_target_off{lab}"] = r["p_target"]
        for name, arr in r["per_temp"].items():
            out[f"{name}_off{lab}"] = arr
    if rg_result is not None:
        out["rg_edges_scaled"] = rg_result["rg_edges_scaled"]
        out["rg_edges_lattice"] = rg_result["rg_edges_lattice"]
        out["p0_rg"] = rg_result["p0_rg"]
        out["rg_target_mass"] = rg_result["rg_target_mass"]
        for name, arr in rg_result["per_temp"].items():
            out[name] = arr
    np.savez_compressed(path, **out)


# ===========================================================================
# Plots
# ===========================================================================

def make_plots(
    outdir: Path,
    target: Dict[str, Any],
    bl_min: int,
    bl_max: int,
    p0_gap_ints: List[int],
    results: List[Dict[str, Any]],
    rg_result: Optional[Dict[str, Any]],
) -> List[str]:
    if plt is None:
        raise RuntimeError(
            "matplotlib is required for plots; install it or pass --no-plots"
        )
    temps = target["temps"]
    saved: List[str] = []

    # 1. Unsupported contact mass vs T, one line per offset.
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for r in results:
        ax.plot(temps, r["per_temp"]["total_unsupported"], "-o", ms=3,
                label=f"offset {r['offset']:g}")
    ax.set_xlabel("T")
    ax.set_ylabel("total unsupported contact mass")
    ax.set_title("Unsupported contact mass vs temperature")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = outdir / "support_unsupported_vs_T.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(str(p))

    # 2. Stacked below/gap/inside/above for each offset.
    for r in results:
        pt = r["per_temp"]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.stackplot(
            temps,
            pt["mass_below"],
            pt["mass_internal_gap"],
            pt["mass_inside_positive"],
            pt["mass_above"],
            labels=["below", "internal gap", "inside (supported)", "above"],
            colors=["#d62728", "#ff7f0e", "#2ca02c", "#9467bd"],
        )
        ax.set_xlabel("T")
        ax.set_ylabel("contact mass fraction")
        ax.set_ylim(0, 1)
        ax.set_title(f"Contact mass partition (offset {r['offset']:g})")
        ax.legend(fontsize=8, loc="upper right")
        fig.tight_layout()
        p = outdir / f"support_stacked_contacts_off{offset_label(r['offset'])}.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(str(p))

    # 3. Target contact heatmap with baseline support boundaries.
    for r in results:
        fig, ax = plt.subplots(figsize=(8, 5))
        im = ax.pcolormesh(r["m_grid"], temps, r["p_target"], cmap="viridis", shading="auto")
        plt.colorbar(im, ax=ax, label="P_target(m)")
        ax.axvline(bl_min - 0.5, color="red", lw=1.5, ls="--", label="baseline range")
        ax.axvline(bl_max + 0.5, color="red", lw=1.5, ls="--")
        for g in p0_gap_ints:
            ax.axvline(g, color="white", lw=0.8, ls=":", alpha=0.7)
        ax.axvline(-0.5, color="cyan", lw=1.0, ls="-", alpha=0.6, label="m = 0")
        ax.set_xlabel("m (shifted integer contacts)")
        ax.set_ylabel("T")
        ax.set_title(f"Target contact distribution + baseline support (offset {r['offset']:g})")
        ax.legend(fontsize=8, loc="upper right")
        fig.tight_layout()
        p = outdir / f"support_contact_heatmap_off{offset_label(r['offset'])}.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(str(p))

    # 4. Max and mean unsupported mass by offset.
    offs = [r["offset"] for r in results]
    means = [r["aggregate"]["mean_total_unsupported"] for r in results]
    maxes = [r["aggregate"]["max_total_unsupported"] for r in results]
    x = np.arange(len(offs))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(x - 0.2, means, width=0.4, label="mean unsupported")
    ax.bar(x + 0.2, maxes, width=0.4, label="max unsupported")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{o:g}" for o in offs])
    ax.set_xlabel("contact offset")
    ax.set_ylabel("unsupported contact mass")
    ax.set_title("Mean and max unsupported contact mass by offset")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = outdir / "support_summary_by_offset.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(str(p))

    # 5. Rg unsupported mass vs T (if available).
    if rg_result is not None:
        rp = rg_result["per_temp"]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(temps, rp["rg_total_unsupported"], "-o", ms=3, color="k", label="total")
        ax.plot(temps, rp["rg_mass_below"], "-", color="#d62728", label="below")
        ax.plot(temps, rp["rg_mass_above"], "-", color="#9467bd", label="above")
        ax.plot(temps, rp["rg_mass_internal_gap"], "-", color="#ff7f0e", label="internal gap")
        ax.set_xlabel("T")
        ax.set_ylabel("unsupported Rg mass")
        ax.set_title("Unsupported Rg mass vs temperature")
        ax.legend(fontsize=8)
        fig.tight_layout()
        p = outdir / "support_rg_unsupported_vs_T.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(str(p))

    return saved


# ===========================================================================
# Quick-test (synthetic unit tests)
# ===========================================================================

def run_quick_test() -> int:
    failures: List[str] = []

    def check(cond: bool, msg: str) -> None:
        if cond:
            print(f"  PASS: {msg}")
        else:
            print(f"  FAIL: {msg}")
            failures.append(msg)

    print("Quick test 1: known below/inside/gap/above contact fractions")
    # baseline range [0,4] with internal gap at m=3 (P0(3)=0).
    m_grid = np.arange(-1, 6, dtype=float)  # -1..5
    p0 = np.zeros_like(m_grid)
    for cv, pk in [(0, 0.25), (1, 0.25), (2, 0.25), (4, 0.25)]:
        p0[int(cv) - (-1)] = pk
    bl_min, bl_max = 0, 4
    p_target = np.zeros_like(m_grid)
    # m: -1,0,1,2,3,4,5
    vals = {-1: 0.10, 0: 0.20, 1: 0.20, 2: 0.10, 3: 0.15, 4: 0.10, 5: 0.15}
    for k, v in vals.items():
        p_target[int(k) - (-1)] = v
    res = classify_contact_row(p_target, m_grid, p0, bl_min, bl_max, 0.0)
    check(abs(res["mass_below"] - 0.10) < 1e-12, "mass_below == 0.10")
    check(abs(res["mass_above"] - 0.15) < 1e-12, "mass_above == 0.15")
    check(abs(res["mass_internal_gap"] - 0.15) < 1e-12, "internal_gap (m=3) == 0.15")
    check(abs(res["mass_inside_positive"] - 0.60) < 1e-12, "inside_positive == 0.60")
    check(abs(res["total_unsupported"] - 0.40) < 1e-12, "total_unsupported == 0.40")
    check(abs(res["geometric_support"] - 0.75) < 1e-12, "geometric_support == 0.75")
    check(abs(res["positive_support"] - 0.60) < 1e-12, "positive_support == 0.60")
    check(abs(res["mass_negative"] - 0.10) < 1e-12, "mass_negative == 0.10")
    # mean = sum(m*p): -0.10+0+0.20+0.20+0.45+0.40+0.75 = 1.90
    check(abs(res["target_mean_shifted"] - 1.90) < 1e-12, "target_mean_shifted == 1.90")

    print("Quick test 2: exact mass conservation (random target/baseline)")
    rng = np.random.default_rng(0)
    for trial in range(200):
        n = int(rng.integers(5, 30))
        g0 = int(rng.integers(-5, 0))
        mg = np.arange(g0, g0 + n, dtype=float)
        pt = rng.random(n)
        pt /= pt.sum()
        p0r = rng.random(n)
        p0r[rng.random(n) < 0.3] = 0.0  # inject zero-prob gaps
        blo = int(mg[int(rng.integers(0, n))])
        bhi = int(mg[int(rng.integers(0, n))])
        if bhi < blo:
            blo, bhi = bhi, blo
        r = classify_contact_row(pt, mg, p0r, blo, bhi, 0.0)
        partition = (
            r["mass_below"]
            + r["mass_inside_positive"]
            + r["mass_internal_gap"]
            + r["mass_above"]
        )
        if abs(partition - 1.0) > 1e-12:
            check(False, f"conservation trial {trial}: partition={partition}")
            break
    else:
        check(True, "200 random partitions each sum to 1.0 within 1e-12")

    print("Quick test 3: rebin conservation + integer-center identity")
    centers = np.arange(0, 11, dtype=float)  # integer centers, spacing 1
    pdf = rng.random(centers.size)
    p = build_target_on_grid(centers, pdf, 0, 10)
    expected = pdf / pdf.sum()
    check(np.allclose(p, expected, atol=1e-12), "integer-center rebin is identity")
    check(abs(p.sum() - 1.0) < 1e-12, "rebinned mass sums to 1.0")
    # Non-integer fine grid, union covers full support: total mass preserved.
    fcent = np.linspace(2.3, 8.7, 50)
    fpdf = rng.random(fcent.size)
    fe = centers_to_edges(fcent)
    gmin = int(np.floor(fe[0] + 0.5))
    gmax = int(np.ceil(fe[-1] - 0.5))
    pf = build_target_on_grid(fcent, fpdf, gmin, gmax)
    check(abs(pf.sum() - 1.0) < 1e-12, "fine-grid rebin onto covering union sums to 1.0")

    print("Quick test 4: Rg overlap below/inside/above")
    edges = np.array([0.0, 1.0, 2.0, 3.0])
    mass = np.array([0.2, 0.5, 0.3])
    b, ins, a = _overlap_below_inside_above(edges, mass, 0.5, 2.5)
    check(abs(b - 0.1) < 1e-12, "Rg below == 0.10")
    check(abs(a - 0.15) < 1e-12, "Rg above == 0.15")
    check(abs(ins - 0.75) < 1e-12, "Rg inside == 0.75")
    check(abs((b + ins + a) - mass.sum()) < 1e-12, "Rg below+inside+above == total")
    # internal-gap detection via per-bin overlap
    per_bin = _overlap_masses(edges, mass, np.array([0.5, 1.5, 2.5]))
    check(abs(per_bin.sum() - 0.75) < 1e-12, "Rg per-bin inside overlap == 0.75")

    print()
    if failures:
        print(f"QUICK TEST FAILED: {len(failures)} assertion(s) failed.")
        return 1
    print("QUICK TEST PASSED: all assertions passed.")
    return 0


# ===========================================================================
# CLI / main
# ===========================================================================

def _parse_offsets(s: str) -> List[float]:
    out: List[float] = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        v = float(tok)
        if not np.isfinite(v):
            raise ValueError(f"contact offset {tok!r} is not finite")
        out.append(v)
    if not out:
        raise ValueError("--contact-offsets parsed to an empty list")
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Quantify probability-mass support mismatch between a "
        "molecular target NPZ and a lattice baseline NPZ.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--target", type=str, default=None,
                    help="Target (molecular/REMD) NPZ: temps/Ts, ct_centers, ct_hists, "
                         "optional rg_centers/Rg_centers, rg_hists/Rg_hists.")
    ap.add_argument("--baseline", type=str, default=None,
                    help="Lattice baseline NPZ (any fitter-accepted contact format; "
                         "joint c_edges/rg_edges/crg_prob enables Rg analysis).")
    ap.add_argument("--contact-offsets", type=str, default="43",
                    dest="contact_offsets_str", metavar="A,B,...",
                    help="Comma-separated contact offsets, e.g. 43,40.")
    ap.add_argument("--rg-scale", type=float, default=1.0, dest="rg_scale",
                    help="Lattice->observed Rg scale: Rg_obs = rg_scale * Rg_lattice.")
    ap.add_argument("--positive-support-threshold", type=float, default=0.0,
                    dest="positive_support_threshold",
                    help="Baseline bins with P0 <= this threshold count as unsupported.")
    ap.add_argument("--outdir", type=str, default="support_diagnostics",
                    help="Output directory (created if missing).")
    ap.add_argument("--no-plots", "--no_plots", action="store_true", dest="no_plots",
                    help="Skip all plot generation.")
    ap.add_argument("--quick-test", action="store_true", dest="quick_test",
                    help="Run synthetic unit tests and exit.")
    args = ap.parse_args(argv)

    if args.quick_test:
        return run_quick_test()

    if args.target is None or args.baseline is None:
        ap.error("--target and --baseline are required (unless --quick-test).")
    if not np.isfinite(args.rg_scale) or args.rg_scale <= 0:
        ap.error("--rg-scale must be finite and positive.")
    if not np.isfinite(args.positive_support_threshold) or args.positive_support_threshold < 0:
        ap.error("--positive-support-threshold must be finite and >= 0.")

    args.contact_offsets = _parse_offsets(args.contact_offsets_str)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    target = load_target(args.target)
    bl_min, bl_max = _get_baseline_integer_range(args.baseline)
    if bl_max < bl_min:
        raise ValueError(f"baseline integer range invalid: [{bl_min}, {bl_max}]")

    # Baseline P0 on its own range to report internal gaps (threshold-aware).
    bl_grid = np.arange(bl_min, bl_max + 1, dtype=float)
    p0_native = build_baseline_mass_on_integer(bl_grid, args.baseline)
    p0_gap_ints = [int(bl_grid[i]) for i in range(bl_grid.size)
                   if p0_native[i] <= args.positive_support_threshold]

    print("--- Support-mismatch diagnostic ---")
    print(f"Target:   {args.target}  (keys: {target['keys']})")
    print(f"Baseline: {args.baseline}")
    print(f"Temperatures: {target['temps'].size} in "
          f"[{target['temps'].min():.4g}, {target['temps'].max():.4g}]")
    print(f"Baseline integer contact range: [{bl_min}, {bl_max}]")
    print(f"Baseline internal gaps (P0 <= {args.positive_support_threshold:g}): "
          f"{p0_gap_ints if p0_gap_ints else 'none'}")
    print(f"Contact offsets: {args.contact_offsets}")

    results: List[Dict[str, Any]] = []
    for off in args.contact_offsets:
        r = analyze_offset(target, args.baseline, off, bl_min, bl_max,
                           args.positive_support_threshold)
        results.append(r)
        a = r["aggregate"]
        print(f"\noffset {off:g}: target integer range [{r['t_min']}, {r['t_max']}]"
              f"  union grid [{r['g_min']}, {r['g_max']}]")
        print(f"  mean total unsupported = {a['mean_total_unsupported']:.4g} "
              f"(max {a['max_total_unsupported']:.4g})")
        print(f"  mean below/gap/above = {a['mean_mass_below']:.4g} / "
              f"{a['mean_mass_internal_gap']:.4g} / {a['mean_mass_above']:.4g}")
        print(f"  mean positive-support = {a['mean_positive_support']:.4g}, "
              f"mean geometric-support = {a['mean_geometric_support']:.4g}")
        print(f"  negative shifted-contact mass: max = {a['max_negative_mass']:.4g} "
              f"({'PRESENT' if a['has_negative_support'] else 'none'})")
        print(f"  mass-conservation max partition error = {r['max_partition_err']:.2e}")

    # Rg analysis (if target Rg and joint baseline both available).
    rg_result: Optional[Dict[str, Any]] = None
    baseline_rg = None
    if target["rg_centers"] is not None:
        baseline_rg = load_baseline_rg(args.baseline, args.rg_scale)
    if target["rg_centers"] is not None and baseline_rg is not None:
        rg_result = analyze_rg(target, baseline_rg, args.positive_support_threshold)
        ra = rg_result["aggregate"]
        lo, hi = rg_result["rg_range_scaled"]
        print(f"\nRg support (offset-independent): scaled baseline range "
              f"[{lo:.4g}, {hi:.4g}]")
        print(f"  mean total unsupported Rg mass = {ra['mean_rg_total_unsupported']:.4g} "
              f"(max {ra['max_rg_total_unsupported']:.4g})")
        print(f"  mean below/gap/above = {ra['mean_rg_below']:.4g} / "
              f"{ra['mean_rg_internal_gap']:.4g} / {ra['mean_rg_above']:.4g}")
    else:
        print("\nRg support: skipped (need target Rg histograms AND a joint baseline).")

    best = select_best_offsets(results, rg_result)

    # Write outputs.
    csv_path = outdir / "support_by_temperature.csv"
    json_path = outdir / "support_summary.json"
    md_path = outdir / "support_report.md"
    npz_path = outdir / "support_rebinned.npz"

    write_csv(csv_path, target, results, rg_result)
    write_summary_json(json_path, args, target, bl_min, bl_max, p0_gap_ints,
                       results, rg_result, best)
    write_report_md(md_path, args, target, bl_min, bl_max, p0_gap_ints,
                    results, rg_result, best)
    write_npz(npz_path, target, results, rg_result)

    print(f"\nSaved: {csv_path}")
    print(f"Saved: {json_path}")
    print(f"Saved: {md_path}")
    print(f"Saved: {npz_path}")

    if not args.no_plots:
        saved = make_plots(outdir, target, bl_min, bl_max, p0_gap_ints,
                           results, rg_result)
        for s in saved:
            print(f"Saved: {s}")

    print("\nBest offset by criterion:")
    print(f"  smallest mean unsupported : {_fmt_opt(best['smallest_mean_unsupported_contact_mass'])}")
    print(f"  smallest max  unsupported : {_fmt_opt(best['smallest_max_unsupported_contact_mass'])}")
    print(f"  no negative support       : {_fmt_opt(best['no_negative_shifted_contact_support'])}")
    if rg_result is not None:
        print(f"  best Rg support           : {_fmt_opt(best['best_rg_support'])} (offset-independent)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
