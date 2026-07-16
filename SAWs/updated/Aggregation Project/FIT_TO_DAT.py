#!/usr/bin/env python3
"""Fit a lattice polymer contact model to REMD contact histograms.

Model
-----
P_model(m|T) ∝ P0(m) * exp[-b(T) * m]

where P0(m) is the athermal (T→∞) baseline from a SAW / lattice simulation
and b(T) is the reduced bias (contact coupling), chosen via --model.

Supported b(T) models
---------------------
hs            b(T) = h/T - s
                Enthalpy/entropy decomposition.  Tc = h/s.  DEFAULT.

tc_scale      b(T) = A * (Tc/T - 1)
                Parameterized directly by the transition temperature Tc.

hs_quadratic  b(T) = h/T - s + a2*x(T)^2,  x = (T-Tref)/Tscale
                Quadratic correction to hs; use when hs residuals are
                asymmetric about Tc.

poly2         b(T) = a0 + a1*x + a2*x^2,  x = (T-Tref)/Tscale
                Flexible polynomial; fit Tref/Tscale to center x on data.

poly3         b(T) = a0 + a1*x + a2*x^2 + a3*x^3,  x = (T-Tref)/Tscale
                Cubic polynomial.  WARNING: very flexible; can overfit when
                the number of temperatures is small relative to parameters.
                Always compare validation loss against a simpler model.

heat_capacity b(T) = [dh0 - T*ds0 + dCp*((T-T0) - T*ln(T/T0))] / T
                Gibbs free energy model with non-zero heat capacity of
                folding dCp.  At dCp=0 recovers the hs model.  Use --T0
                to set the reference temperature T0 (default: midpoint of
                temperature range).  dCp > 0 creates a cold-denaturation
                branch; dCp < 0 sharpens the transition.

Why validation loss matters
---------------------------
Fitting contact distributions at every temperature can overfit: the optimizer
finds parameters that memorize noise or ledges in the histograms rather than
the underlying thermodynamics.  Hold out a subset of temperatures with
--holdout-every or --holdout-indices and inspect the validation loss to detect
overfitting.  For poly3 especially, the validation loss often exceeds the
training loss substantially even on clean data.

How to run (quick start)
------------------------
  python fit_lattice_contact_model.py \\
      --remd remd_distributions_30mer.npz \\
      --baseline single_uniform_chain2_athermal_dists_joint_N30_T1_seed42.npz \\
      --contact_offset 29 --model hs --loss js --holdout-every 3 \\
      --outdir fits/my_run

Comparing models
----------------
Run with --outdir for each model.  Compare the train and validation losses
from train_validation_loss.csv (lower is better on the validation set).
Prefer a simpler model when its validation loss is similar to a richer one.

Rg prediction
-------------
If the baseline NPZ contains a joint P0(m, Rg) (keys c_edges, rg_edges,
crg_prob), P(Rg|T) is predicted for every temperature by marginalizing over
m with Boltzmann weights.  Use --fit-rg to include Rg in the objective
(requires observed rg_hists in the REMD file).

Scalar Rg(T) mode
-----------------
For experimental or molecular-simulation data that provide only a SCALAR Rg per
temperature (no histograms), use --rg-means-file.  The file has four
whitespace-separated columns: temperature (K), central Rg, lower Rg, upper Rg.
The lower/upper columns are treated as DESCRIPTIVE BOUNDS, never as standard
errors, unless you explicitly select --rg-mean-loss range_weighted.

This mode needs no REMD NPZ and no --contact_offset.  It requires a joint
baseline, predicts a scalar Rg by summarizing the reweighted P0(m, Rg), and fits
b(T) directly to Rg(T) with a scalar regression loss (--rg-mean-loss).  It never
fabricates contact targets or Rg histograms, and JS/KL divergence is never
applied to scalar values.

Units: --rg-scale keeps its existing meaning, Rg_observed = rg_scale * Rg_lattice.
For a 30-monomer chain with 0.345 nm between monomers, --rg-scale 0.345.

Primary (fixed-scale, physically motivated) analysis:

  python FIT_TO_DAT.py \\
      --baseline single_uniform_chain2_athermal_dists_joint_N30.npz \\
      --rg-means-file single.dat \\
      --rg-target-units observed \\
      --rg-scale 0.345 \\
      --rg-summary rms \\
      --rg-mean-loss mse \\
      --model tc_scale \\
      --holdout-every 3 \\
      --n_restarts 32 \\
      --rg-feasibility-scan \\
      --split-sensitivity \\
      --bootstrap 200 \\
      --uncertainty-diagnostics \\
      --outdir fits/paper_single_30mer

Diagnostic free-scale run (mapping sensitivity only):

  python FIT_TO_DAT.py \\
      --baseline single_uniform_chain2_athermal_dists_joint_N30.npz \\
      --rg-means-file single.dat \\
      --rg-target-units observed \\
      --rg-scale 0.345 \\
      --fit-rg-scale \\
      --rg-scale-min 0.25 \\
      --rg-scale-max 0.55 \\
      --rg-summary rms \\
      --rg-mean-loss mse \\
      --model tc_scale \\
      --outdir fits/paper_single_30mer_free_scale

The fixed-scale fit is the primary analysis: rg_scale = 0.345 nm per lattice unit
follows from the intermonomer distance, so it is an independent physical input
rather than something the data should choose.  The free-scale run is a
mapping-sensitivity DIAGNOSTIC: it asks how far rg_scale would have to move for
the lattice model to reproduce the data.  A fitted scale far from 0.345 does not
license reporting that scale as a result — it indicates the lattice-to-molecular
mapping, the chain length, or the baseline is mis-specified.

Always run --rg-feasibility-scan.  It scans the contact bias directly and reports
the range of scalar Rg the baseline can produce at ANY bias.  If observed Rg
values fall outside that reachable range, no b(T) can reproduce them and the fit
is not scientifically valid for those points, however small its residuals.

Transition descriptors are distinct and both are reported:
  T_bias_zero     temperature where b(T) = 0 (bias sign change)
  T_rg_max_slope  temperature maximizing -dRg/dT of the fitted finite-chain
                  curve; the primary finite-chain transition descriptor
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

try:  # plotting is optional when --no-plots is used
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - depends on local plotting stack
    plt = None

try:
    from scipy.optimize import minimize
except Exception:
    minimize = None


# ---------------------------------------------------------------------------
# Utility: histogram helpers
# ---------------------------------------------------------------------------

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


def rebin_pdf_to_mass(
    source_centers: np.ndarray,
    source_pdf: np.ndarray,
    target_edges: np.ndarray,
) -> np.ndarray:
    """Rebin a pdf (on evenly-spaced source_centers) onto arbitrary target_edges.

    Uses the same piecewise-constant density overlap-integration as
    rebin_pdf_mass_to_integer_bins.  Returns probability mass on each target
    bin, normalized to sum to 1.  Mass outside the source support is zero.
    """
    source_centers = np.asarray(source_centers, dtype=float)
    source_pdf = np.asarray(source_pdf, dtype=float)
    target_edges = np.asarray(target_edges, dtype=float)

    source_mass, _ = pdf_to_mass(source_pdf, source_centers)
    native_edges = centers_to_edges(source_centers)
    native_w = np.diff(native_edges)
    dens = np.zeros_like(source_mass)
    mask = native_w > 0
    dens[mask] = source_mass[mask] / native_w[mask]

    n_target = len(target_edges) - 1
    p_out = np.zeros(n_target, dtype=float)
    j = 0
    for i in range(n_target):
        a, b = float(target_edges[i]), float(target_edges[i + 1])
        while j < len(dens) and native_edges[j + 1] <= a:
            j += 1
        jj = j
        while jj < len(dens) and native_edges[jj] < b:
            left = max(a, native_edges[jj])
            right = min(b, native_edges[jj + 1])
            if right > left:
                p_out[i] += dens[jj] * (right - left)
            jj += 1

    s = p_out.sum()
    if s > 0:
        p_out /= s
    return p_out


def rebin_mass_between_edges(
    source_edges: np.ndarray,
    source_mass: np.ndarray,
    target_edges: np.ndarray,
) -> np.ndarray:
    """Rebin probability mass from arbitrary source_edges onto target_edges.

    Treats source_mass[j] as mass spread uniformly across [source_edges[j],
    source_edges[j+1]) (piecewise-constant density) and integrates the density
    over each target bin.  Returns probability mass per target bin, normalized
    to sum to 1.  Unlike rebin_pdf_to_mass, the source grid need not be
    evenly spaced.
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

    s = out.sum()
    if s > 0:
        out /= s
    return out


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
    """Return baseline probability mass p0(m) on the integer m grid."""
    b = np.load(baseline_npz)
    m_centers_int = np.asarray(m_centers_int, dtype=float)
    m0 = int(round(m_centers_int.min()))
    m1 = int(round(m_centers_int.max()))
    m_vals = np.arange(m0, m1 + 1, dtype=int)

    # Case A: discrete contact values with per-value probabilities (c_vals, c_prob)
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
    # Pc may be 2D (nT, maxC+1); average over rows to get the athermal baseline.
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
        p_c = crg_prob.sum(axis=1)          # marginalize over Rg
        p_c = np.clip(p_c, 0.0, None)
        if p_c.sum() <= 0:
            raise ValueError("baseline crg_prob marginal over Rg sums to 0")
        p_c /= p_c.sum()
        # p_c is probability mass per source contact bin, not a density.
        # Rebin it using the actual c_edges so non-unit or uneven source bins
        # remain mass-conserving.
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
        if not np.allclose(
            np.diff(ccent), np.diff(ccent)[0], rtol=1e-6, atol=1e-10
        ):
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


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def kl_div(p_obs: np.ndarray, p_mod: np.ndarray, eps: float = 1e-12) -> float:
    """KL divergence KL(p_obs || p_mod). Returns 0 when p_obs == p_mod."""
    p_obs = np.asarray(p_obs, dtype=float)
    p_mod = np.asarray(p_mod, dtype=float)
    p_obs = np.clip(p_obs, eps, 1.0)
    p_mod = np.clip(p_mod, eps, 1.0)
    p_obs = p_obs / p_obs.sum()
    p_mod = p_mod / p_mod.sum()
    return float(np.sum(p_obs * (np.log(p_obs) - np.log(p_mod))))


def js_div(p_obs: np.ndarray, p_mod: np.ndarray, eps: float = 1e-12) -> float:
    """Jensen-Shannon divergence (symmetric). Returns 0 when p_obs == p_mod."""
    p_obs = np.asarray(p_obs, dtype=float)
    p_mod = np.asarray(p_mod, dtype=float)
    p_obs = np.clip(p_obs, eps, 1.0)
    p_mod = np.clip(p_mod, eps, 1.0)
    p_obs = p_obs / p_obs.sum()
    p_mod = p_mod / p_mod.sum()
    m = np.clip(0.5 * (p_obs + p_mod), eps, 1.0)
    kl1 = float(np.sum(p_obs * (np.log(p_obs) - np.log(m))))
    kl2 = float(np.sum(p_mod * (np.log(p_mod) - np.log(m))))
    return 0.5 * (kl1 + kl2)


def _get_loss_fn(name: str) -> Callable[[np.ndarray, np.ndarray], float]:
    if name == "kl":
        return kl_div
    if name == "js":
        return js_div
    raise ValueError(f"Unknown loss {name!r}. Choose 'kl' or 'js'.")


def _rg_loss_sum(
    rg_mod_mass: np.ndarray,
    p_obs_rg: np.ndarray,
    loss_fn: Callable[[np.ndarray, np.ndarray], float],
) -> float:
    """Sum loss_fn(obs, mod) over temperatures for Rg distributions."""
    total = 0.0
    for i in range(len(rg_mod_mass)):
        total += loss_fn(p_obs_rg[i], rg_mod_mass[i])
    return total


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
# Each entry defines b(T) = the reduced bias (contact coupling).
# P_model(m|T) ∝ P0(m) * exp[-b(T) * m]
#
# raw_b_fn(params, T, Tref, Tscale) -> float
# derived_Tc(params) -> float | None  (or None if not meaningful for that model)

def _b_hs(params: np.ndarray, T: float, Tref: float, Tscale: float) -> float:
    return float(params[0]) / T - float(params[1])


def _b_tc_scale(params: np.ndarray, T: float, Tref: float, Tscale: float) -> float:
    return float(params[0]) * (float(params[1]) / T - 1.0)


def _b_hs_quadratic(params: np.ndarray, T: float, Tref: float, Tscale: float) -> float:
    x = (T - Tref) / Tscale
    return float(params[0]) / T - float(params[1]) + float(params[2]) * x * x


def _b_poly2(params: np.ndarray, T: float, Tref: float, Tscale: float) -> float:
    x = (T - Tref) / Tscale
    return float(params[0]) + float(params[1]) * x + float(params[2]) * x * x


def _b_poly3(params: np.ndarray, T: float, Tref: float, Tscale: float) -> float:
    x = (T - Tref) / Tscale
    return (
        float(params[0])
        + float(params[1]) * x
        + float(params[2]) * x * x
        + float(params[3]) * x * x * x
    )


def _b_heat_capacity(params: np.ndarray, T: float, Tref: float, Tscale: float) -> float:
    """b(T) = dg(T)/T, dg = dh0 - T*ds0 + dCp*((T-T0) - T*ln(T/T0)).

    Tref is used as T0 (set via --T0 or defaults to midpoint of T range).
    """
    T0 = Tref
    dh0, ds0, dCp = float(params[0]), float(params[1]), float(params[2])
    dg = dh0 - T * ds0 + dCp * ((T - T0) - T * np.log(T / T0))
    return dg / T


def _tc_hs(params: np.ndarray) -> Optional[float]:
    s = float(params[1])
    return float(params[0]) / s if abs(s) > 1e-15 else None


def _tc_tc_scale(params: np.ndarray) -> Optional[float]:
    return float(params[1])


MODEL_REGISTRY: Dict[str, Dict] = {
    "hs": {
        "param_names": ["h", "s"],
        "x0": [750.0, 2.8],
        "bounds": [(-2000.0, 2000.0), (-10.0, 10.0)],
        "raw_b_fn": _b_hs,
        "derived_Tc": _tc_hs,
        "description": "b(T) = h/T - s",
    },
    "tc_scale": {
        "param_names": ["A", "Tc"],
        "x0": [1.0, 300.0],
        "bounds": [(0.01, 200.0), (10.0, 5000.0)],
        "raw_b_fn": _b_tc_scale,
        "derived_Tc": _tc_tc_scale,
        "description": "b(T) = A*(Tc/T - 1)",
    },
    "hs_quadratic": {
        "param_names": ["h", "s", "a2"],
        "x0": [750.0, 2.8, 0.0],
        "bounds": [(-2000.0, 2000.0), (-10.0, 10.0), (-20.0, 20.0)],
        "raw_b_fn": _b_hs_quadratic,
        "derived_Tc": None,
        "description": "b(T) = h/T - s + a2*x(T)^2,  x = (T-Tref)/Tscale",
    },
    "poly2": {
        "param_names": ["a0", "a1", "a2"],
        "x0": [0.0, 0.0, 0.0],
        "bounds": [(-20.0, 20.0), (-20.0, 20.0), (-20.0, 20.0)],
        "raw_b_fn": _b_poly2,
        "derived_Tc": None,
        "description": "b(T) = a0 + a1*x(T) + a2*x(T)^2,  x = (T-Tref)/Tscale",
    },
    "poly3": {
        "param_names": ["a0", "a1", "a2", "a3"],
        "x0": [0.0, 0.0, 0.0, 0.0],
        "bounds": [(-20.0, 20.0), (-20.0, 20.0), (-20.0, 20.0), (-20.0, 20.0)],
        "raw_b_fn": _b_poly3,
        "derived_Tc": None,
        "description": (
            "b(T) = a0 + a1*x + a2*x^2 + a3*x^3,  x = (T-Tref)/Tscale  "
            "[WARNING: flexible — verify with validation loss to avoid overfitting]"
        ),
    },
    "heat_capacity": {
        "param_names": ["dh0", "ds0", "dCp"],
        "x0": [750.0, 2.8, 0.0],
        "bounds": [(-10000.0, 10000.0), (-50.0, 50.0), (-1000.0, 1000.0)],
        "raw_b_fn": _b_heat_capacity,
        "derived_Tc": None,
        "description": (
            "b(T) = [dh0 - T*ds0 + dCp*((T-T0) - T*ln(T/T0))] / T  "
            "(set T0 via --T0; defaults to midpoint of temperature range)"
        ),
    },
}


# Shared model-contract version.  Bump only when the model set, parameter names,
# or b(T) semantics change in a way that breaks cross-script compatibility.
MODEL_API_VERSION = 1


def get_model_contract() -> dict:
    """Return a callable-free description of the supported contact-bias models.

    Used to verify that this script and remd_uniform_chain_new.py agree on the
    model API version, model names, and parameter ordering.
    """
    return {
        "model_api_version": MODEL_API_VERSION,
        "models": {
            name: {
                "param_names": list(spec["param_names"]),
                "description": str(spec["description"]),
            }
            for name, spec in MODEL_REGISTRY.items()
        },
    }


def make_b_fn(
    model_name: str, Tref: float, Tscale: float
) -> Callable[[np.ndarray, float], float]:
    """Return b(params, T) with Tref and Tscale captured by closure."""
    raw = MODEL_REGISTRY[model_name]["raw_b_fn"]

    def b_fn(params: np.ndarray, T: float) -> float:
        return raw(params, T, Tref, Tscale)

    return b_fn


# ---------------------------------------------------------------------------
# Generic model probability
# ---------------------------------------------------------------------------

def model_contact_mass(
    p0_mass: np.ndarray,
    m_centers: np.ndarray,
    T: float,
    params: np.ndarray,
    b_fn: Callable[[np.ndarray, float], float],
) -> np.ndarray:
    """P_model(m|T) ∝ P0(m) * exp[-b(T)*m], with max-subtraction stabilization."""
    m_centers = np.asarray(m_centers, dtype=float)
    b = b_fn(params, float(T))
    x = -b * m_centers
    x = x - np.max(x)
    w = p0_mass * np.exp(x)
    Z = w.sum()
    if not np.isfinite(Z) or Z <= 0:
        return np.full_like(p0_mass, 1.0 / p0_mass.size)
    return w / Z


def objective(
    params: np.ndarray,
    temps: np.ndarray,
    m_centers: np.ndarray,
    p_obs_mass: np.ndarray,
    p0_mass: np.ndarray,
    b_fn: Callable[[np.ndarray, float], float],
    loss_fn: Callable[[np.ndarray, np.ndarray], float],
) -> float:
    """Sum of per-temperature contact loss over the provided temperatures."""
    total = 0.0
    for i, T in enumerate(temps):
        p_mod = model_contact_mass(p0_mass, m_centers, float(T), params, b_fn)
        total += loss_fn(p_obs_mass[i], p_mod)
    return total


def objective_combined(
    params: np.ndarray,
    train_temps: np.ndarray,
    m_centers: np.ndarray,
    p_obs_ct_train: np.ndarray,
    p0_mass: np.ndarray,
    crg_prob: np.ndarray,
    c_edges_joint: np.ndarray,
    p_obs_rg_train: np.ndarray,
    b_fn: Callable[[np.ndarray, float], float],
    loss_fn: Callable[[np.ndarray, np.ndarray], float],
    rg_weight: float,
) -> float:
    """Contact loss + rg_weight * Rg loss, summed over training temperatures."""
    m_joint = 0.5 * (c_edges_joint[:-1] + c_edges_joint[1:])
    total = 0.0
    for i, T in enumerate(train_temps):
        # contact term
        p_mod_ct = model_contact_mass(p0_mass, m_centers, float(T), params, b_fn)
        total += loss_fn(p_obs_ct_train[i], p_mod_ct)
        # Rg term: reweight joint baseline by exp[-b*m], marginalize over m
        b = b_fn(params, float(T))
        x = -b * m_joint
        x -= x.max()
        w_m = np.exp(x)
        rg_mass = (crg_prob.T * w_m).T.sum(axis=0)
        Z = rg_mass.sum()
        if Z > 0:
            rg_mass /= Z
        total += rg_weight * loss_fn(p_obs_rg_train[i], rg_mass)
    return total


# ---------------------------------------------------------------------------
# Rg prediction
# ---------------------------------------------------------------------------

def predict_rg_from_joint(
    crg_prob: np.ndarray,
    c_edges: np.ndarray,
    rg_edges: np.ndarray,
    temps: np.ndarray,
    params: np.ndarray,
    b_fn: Callable[[np.ndarray, float], float],
) -> Tuple[np.ndarray, np.ndarray]:
    """Predict P(Rg|T) for all temps by reweighting P0(m,Rg) in m.

    Returns (rg_centers, rg_mod_mass) where rg_mod_mass has shape (n_temps, n_rg).
    """
    c_edges = np.asarray(c_edges, dtype=float)
    rg_edges = np.asarray(rg_edges, dtype=float)
    crg_prob = np.asarray(crg_prob, dtype=float)

    m_centers = 0.5 * (c_edges[:-1] + c_edges[1:])
    rg_mass_T = np.zeros((temps.size, rg_edges.size - 1), dtype=float)

    for i, T in enumerate(temps):
        b = b_fn(params, float(T))
        x = -b * m_centers
        x = x - np.max(x)
        w_m = np.exp(x)
        rg_mass = (crg_prob.T * w_m).T.sum(axis=0)
        Z = rg_mass.sum()
        if Z > 0:
            rg_mass /= Z
        rg_mass_T[i] = rg_mass

    rg_centers = 0.5 * (rg_edges[:-1] + rg_edges[1:])
    return rg_centers, rg_mass_T


# ---------------------------------------------------------------------------
# Held-out split resolution
# ---------------------------------------------------------------------------

def _resolve_split_indices(
    n_temps: int,
    holdout_every: Optional[int],
    holdout_indices_str: Optional[str],
    train_indices_str: Optional[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (train_idx, val_idx) integer index arrays.

    Priority:
      1. --train-indices  → training set; validation = everything else
      2. --holdout-indices → validation set; training = everything else
      3. --holdout-every N → validation = [0, N, 2N, ...]; training = rest
      4. None             → train on all, no validation

    Raises ValueError for:
      - --holdout-every <= 0
      - indices outside [0, n_temps-1]
      - duplicate indices in any set
      - empty training set after split
    """
    if holdout_every is not None and holdout_every <= 0:
        raise ValueError(f"--holdout-every must be >= 1, got {holdout_every}")

    all_idx = np.arange(n_temps, dtype=int)

    def _parse_and_validate(s: str, flag: str) -> np.ndarray:
        raw = [int(x.strip()) for x in s.split(",") if x.strip()]
        if len(set(raw)) != len(raw):
            raise ValueError(f"{flag} contains duplicate indices: {raw}")
        arr = np.array(raw, dtype=int)
        if arr.size > 0 and (arr.min() < 0 or arr.max() >= n_temps):
            raise ValueError(
                f"{flag} has index out of range [0, {n_temps - 1}]: {raw}"
            )
        return arr

    if train_indices_str is not None:
        train_idx = _parse_and_validate(train_indices_str, "--train-indices")
        if train_idx.size == 0:
            raise ValueError("--train-indices is empty")
        val_idx = np.setdiff1d(all_idx, train_idx)
        return train_idx, val_idx

    if holdout_indices_str is not None:
        val_idx = _parse_and_validate(holdout_indices_str, "--holdout-indices")
        train_idx = np.setdiff1d(all_idx, val_idx)
        if train_idx.size == 0:
            raise ValueError("--holdout-indices left no training temperatures")
        return train_idx, val_idx

    if holdout_every is not None:
        val_idx = all_idx[::holdout_every]
        train_idx = np.setdiff1d(all_idx, val_idx)
        if train_idx.size == 0:
            raise ValueError(
                f"--holdout-every {holdout_every} left no training temperatures "
                f"(only {n_temps} temps total)"
            )
        return train_idx, val_idx

    return all_idx.copy(), np.array([], dtype=int)


# ---------------------------------------------------------------------------
# JSON serialization helper
# ---------------------------------------------------------------------------

class _NpEncoder(json.JSONEncoder):
    """JSON encoder that converts numpy scalars and arrays to plain Python types."""

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


def _finite_or_none(x: Any) -> Optional[float]:
    """Map a float to itself if finite, else None (for strict JSON)."""
    if x is None:
        return None
    xf = float(x)
    return xf if np.isfinite(xf) else None


# ---------------------------------------------------------------------------
# Reusable fitting primitives
# ---------------------------------------------------------------------------
# These wrap the optimization and scoring so a model can be fit repeatedly under
# different train/validation splits WITHOUT duplicating the underlying math
# (objective / objective_combined / model_contact_mass / predict_rg_from_joint
# remain the single source of truth).  The ordinary one-fit path in main() calls
# build_objective() and fit_one_split() too, so primary behavior is unchanged.

def build_objective(
    fit_rg: bool,
    train_temps: np.ndarray,
    m_centers: np.ndarray,
    p_obs_ct_train: np.ndarray,
    p0_mass: np.ndarray,
    b_fn: Callable[[np.ndarray, float], float],
    loss_fn: Callable[[np.ndarray, np.ndarray], float],
    *,
    crg_prob: Optional[np.ndarray] = None,
    c_edges_joint: Optional[np.ndarray] = None,
    p_obs_rg_train: Optional[np.ndarray] = None,
    rg_weight: float = 1.0,
) -> Tuple[Callable, Tuple]:
    """Return (objective_fn, args_tuple) for a given training split.

    Mirrors the selection used by the primary fit: combined contact+Rg objective
    when fit_rg is True, contact-only objective otherwise.
    """
    if fit_rg:
        obj_args = (
            train_temps, m_centers, p_obs_ct_train, p0_mass,
            crg_prob, c_edges_joint, p_obs_rg_train,
            b_fn, loss_fn, float(rg_weight),
        )
        return objective_combined, obj_args
    return objective, (train_temps, m_centers, p_obs_ct_train, p0_mass, b_fn, loss_fn)


def fit_restarts(
    obj_fn: Callable,
    obj_args: Tuple,
    x0s: List[np.ndarray],
    bounds: List[Tuple[float, float]],
    *,
    maxiter: int = 800,
):
    """Run L-BFGS-B from every restart.

    Returns (best_result, best_objective, restart_records) where restart_records
    is a per-restart list of dicts (success, objective, params, n_iter, message).
    Raises RuntimeError if no restart succeeds with a finite objective.
    Selection logic is identical to the primary fit.
    """
    if minimize is None:
        raise RuntimeError("scipy is required for fitting. Install scipy.")
    best = None
    best_val_obj = float("inf")
    failed_messages: List[str] = []
    restart_records: List[Dict[str, Any]] = []
    for ri, x0 in enumerate(x0s):
        res = minimize(
            obj_fn, x0,
            args=obj_args,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": maxiter},
        )
        restart_records.append({
            "restart_index": ri,
            "success": bool(res.success),
            "objective": float(res.fun) if np.isfinite(res.fun) else None,
            "n_iter": int(res.nit) if hasattr(res, "nit") else None,
            "message": str(res.message),
            "params": [float(v) for v in np.asarray(res.x, dtype=float)],
        })
        if not bool(res.success):
            failed_messages.append(str(res.message))
            continue
        if np.isfinite(res.fun) and res.fun < best_val_obj:
            best_val_obj = float(res.fun)
            best = res
    if best is None:
        detail = "; ".join(dict.fromkeys(failed_messages)) or "no finite objective"
        raise RuntimeError(f"All optimizer restarts failed: {detail}")
    return best, best_val_obj, restart_records


def fit_one_split(
    obj_fn: Callable,
    obj_args: Tuple,
    x0s: List[np.ndarray],
    bounds: List[Tuple[float, float]],
    *,
    maxiter: int = 800,
):
    """Run all restarts and return (best_result, best_objective).

    Thin wrapper over fit_restarts() so the primary fit, split sensitivity, and
    bootstrap all share one optimization pathway.
    """
    best, best_val_obj, _ = fit_restarts(
        obj_fn, obj_args, x0s, bounds, maxiter=maxiter
    )
    return best, best_val_obj


def per_temp_contact_losses(
    temps: np.ndarray,
    m_centers: np.ndarray,
    p_obs_mass: np.ndarray,
    p0_mass: np.ndarray,
    params: np.ndarray,
    b_fn: Callable[[np.ndarray, float], float],
    loss_fn: Callable[[np.ndarray, np.ndarray], float],
) -> np.ndarray:
    """Per-temperature contact loss; summing over a subset reproduces objective()."""
    out = np.empty(len(temps), dtype=float)
    for i, T in enumerate(temps):
        p_mod = model_contact_mass(p0_mass, m_centers, float(T), params, b_fn)
        out[i] = loss_fn(p_obs_mass[i], p_mod)
    return out


def per_temp_rg_losses(
    rg_mod_mass: np.ndarray,
    p_obs_rg_model_grid: np.ndarray,
    loss_fn: Callable[[np.ndarray, np.ndarray], float],
) -> np.ndarray:
    """Per-temperature Rg loss; summing over a subset reproduces _rg_loss_sum()."""
    n = len(rg_mod_mass)
    out = np.empty(n, dtype=float)
    for i in range(n):
        out[i] = loss_fn(p_obs_rg_model_grid[i], rg_mod_mass[i])
    return out


def count_boundary_hits(
    params: np.ndarray,
    bounds: List[Tuple[float, float]],
    param_names: List[str],
    *,
    atol: float = 1e-6,
    rtol: float = 1e-4,
) -> List[str]:
    """Return names of parameters resting on a (finite) optimization bound."""
    hits: List[str] = []
    for name, val, (lo, hi) in zip(param_names, params, bounds):
        span = abs(hi - lo)
        tol = atol + rtol * span
        if np.isfinite(lo) and abs(float(val) - lo) <= tol:
            hits.append(f"{name}@lo")
        elif np.isfinite(hi) and abs(hi - float(val)) <= tol:
            hits.append(f"{name}@hi")
    return hits


# ---------------------------------------------------------------------------
# Validation-split scheme construction
# ---------------------------------------------------------------------------

def _make_split(
    scheme: str, name: str, train_idx: np.ndarray, val_idx: np.ndarray, n: int
) -> Dict[str, Any]:
    """Validate and package one split; rejects empty train/val and bad indices."""
    train = np.unique(np.asarray(train_idx, dtype=int))
    val = np.unique(np.asarray(val_idx, dtype=int))
    if train.size == 0:
        raise ValueError(f"split {name!r}: empty training set")
    if val.size == 0:
        raise ValueError(f"split {name!r}: empty validation set")
    if np.intersect1d(train, val).size > 0:
        raise ValueError(f"split {name!r}: train/validation indices overlap")
    for arr, lbl in ((train, "train"), (val, "validation")):
        if arr.min() < 0 or arr.max() >= n:
            raise ValueError(f"split {name!r}: {lbl} index out of range [0, {n - 1}]")
    return {"scheme": scheme, "name": name, "train_idx": train, "val_idx": val}


def build_split_schemes(
    n: int,
    schemes: List[str],
    *,
    kfold_k: int,
    blocked_fraction: float,
    random_fraction: float,
    random_repeats: int,
    split_seed: int,
) -> List[Dict[str, Any]]:
    """Construct the requested built-in validation splits.

    Supported scheme names: every_third_phase, kfold, blocked_low, blocked_mid,
    blocked_high, random.
    """
    all_idx = np.arange(n, dtype=int)
    out: List[Dict[str, Any]] = []

    def blocked_size(frac: float) -> int:
        if not (0.0 < frac < 1.0):
            raise ValueError(f"fraction must be in (0, 1), got {frac}")
        b = int(np.floor(frac * n))
        b = max(1, min(b, n - 1))
        return b

    for s in schemes:
        if s == "every_third_phase":
            for phase in range(3):
                val = all_idx[all_idx % 3 == phase]
                train = np.setdiff1d(all_idx, val)
                out.append(_make_split(s, f"every_third_phase{phase}", train, val, n))
        elif s == "kfold":
            K = int(kfold_k)
            if K < 2:
                raise ValueError(f"kfold K must be >= 2, got {K}")
            if K > n:
                raise ValueError(f"kfold K={K} exceeds number of temperatures {n}")
            for j in range(K):
                val = all_idx[all_idx % K == j]
                train = np.setdiff1d(all_idx, val)
                out.append(_make_split(s, f"kfold{K}_fold{j}", train, val, n))
        elif s in ("blocked_low", "blocked_mid", "blocked_high"):
            b = blocked_size(blocked_fraction)
            if s == "blocked_low":
                val = all_idx[:b]
            elif s == "blocked_high":
                val = all_idx[n - b:]
            else:
                start = (n - b) // 2
                val = all_idx[start:start + b]
            train = np.setdiff1d(all_idx, val)
            out.append(_make_split(s, s, train, val, n))
        elif s == "random":
            rng = np.random.default_rng(split_seed)
            b = blocked_size(random_fraction)
            reps = int(random_repeats)
            if reps < 1:
                raise ValueError(f"random_repeats must be >= 1, got {reps}")
            for r in range(reps):
                val = np.sort(rng.choice(all_idx, size=b, replace=False))
                train = np.setdiff1d(all_idx, val)
                out.append(_make_split(s, f"random{r}", train, val, n))
        else:
            raise ValueError(
                f"unknown split scheme {s!r}. Choose from: every_third_phase, "
                "kfold, blocked_low, blocked_mid, blocked_high, random."
            )
    return out


def load_split_config_json(path: str, n: int) -> List[Dict[str, Any]]:
    """Load user-defined splits from JSON.

    Expected: a list of objects, each with a 'name' and either 'train_indices'
    (validation = the rest) or 'holdout_indices' (training = the rest).
    """
    with open(path, "r", encoding="utf-8") as fh:
        spec = json.load(fh)
    if not isinstance(spec, list) or not spec:
        raise ValueError("--split-config-json must contain a non-empty list of splits")
    all_idx = np.arange(n, dtype=int)
    out: List[Dict[str, Any]] = []
    for k, entry in enumerate(spec):
        if not isinstance(entry, dict):
            raise ValueError(f"split-config entry {k} is not an object")
        name = str(entry.get("name", f"custom{k}"))
        if "train_indices" in entry and entry["train_indices"] is not None:
            train = np.array([int(x) for x in entry["train_indices"]], dtype=int)
            val = np.setdiff1d(all_idx, train)
        elif "holdout_indices" in entry and entry["holdout_indices"] is not None:
            val = np.array([int(x) for x in entry["holdout_indices"]], dtype=int)
            train = np.setdiff1d(all_idx, val)
        else:
            raise ValueError(
                f"split-config entry {name!r} must define 'train_indices' or "
                "'holdout_indices'"
            )
        out.append(_make_split("custom", name, train, val, n))
    return out


def summarize_param_stability(
    records: List[Dict[str, Any]], keys: List[str]
) -> Dict[str, Dict[str, Optional[float]]]:
    """mean/std/min/max/range/cv for each key across split-fit records."""
    stats: Dict[str, Dict[str, Optional[float]]] = {}
    for key in keys:
        vals = np.array(
            [r[key] for r in records if key in r and np.isfinite(r[key])],
            dtype=float,
        )
        if vals.size == 0:
            stats[key] = {
                "mean": None, "std": None, "min": None, "max": None,
                "range": None, "cv": None, "n": 0,
            }
            continue
        mean = float(vals.mean())
        std = float(vals.std())
        vmin = float(vals.min())
        vmax = float(vals.max())
        cv = float(std / abs(mean)) if abs(mean) > 1e-15 else None
        stats[key] = {
            "mean": mean, "std": std, "min": vmin, "max": vmax,
            "range": float(vmax - vmin), "cv": cv, "n": int(vals.size),
        }
    return stats


# ---------------------------------------------------------------------------
# Validation-split sensitivity driver
# ---------------------------------------------------------------------------

def run_split_sensitivity(args: argparse.Namespace, ctx: Dict[str, Any]) -> None:
    """Fit the model under many validation splits and write supplementary outputs.

    Supplementary only: never touches the primary fit_summary.json / fit_results.npz
    / fit_params.csv / train_validation_loss.csv consumed by the suite.
    """
    temps = ctx["temps"]
    n = len(temps)
    m_centers = ctx["m_centers"]
    p_obs_mass = ctx["p_obs_mass"]
    p0_mass = ctx["p0_mass"]
    b_fn = ctx["b_fn"]
    loss_fn = ctx["loss_fn"]
    spec = ctx["spec"]
    param_names = ctx["param_names"]
    bounds = ctx["bounds"]
    x0s = ctx["x0s"]
    fit_rg = ctx["fit_rg"]
    rg_weight = float(ctx["rg_weight"])
    can_fit_rg = ctx["can_fit_rg"]
    crg_prob = ctx["crg_prob"]
    c_edges_joint = ctx["c_edges_joint"]
    rg_edges_model_lattice = ctx["rg_edges_model_lattice"]
    p_obs_rg_model_grid = ctx["p_obs_rg_model_grid"]
    outdir = ctx["outdir"]
    loss_name = ctx["loss_name"]
    derived_Tc_fn = spec["derived_Tc"]

    schemes = [s.strip() for s in args.split_schemes.split(",") if s.strip()]
    split_seed = args.split_seed if args.split_seed is not None else args.seed

    splits = build_split_schemes(
        n, schemes,
        kfold_k=args.split_kfold_k,
        blocked_fraction=args.split_blocked_fraction,
        random_fraction=args.split_random_fraction,
        random_repeats=args.split_random_repeats,
        split_seed=split_seed,
    )
    if args.split_config_json is not None:
        splits = splits + load_split_config_json(args.split_config_json, n)
    if not splits:
        raise ValueError("No validation splits were constructed.")

    print(f"\n=== Validation-split sensitivity ({len(splits)} splits) ===")
    print(f"  schemes: {schemes}"
          + (f" + custom({args.split_config_json})" if args.split_config_json else ""))
    print(f"  split_seed={split_seed}, kfold_k={args.split_kfold_k}, "
          f"blocked_fraction={args.split_blocked_fraction}, "
          f"random_fraction={args.split_random_fraction}, "
          f"random_repeats={args.split_random_repeats}")

    records: List[Dict[str, Any]] = []
    per_temp_rows: List[Dict[str, Any]] = []
    # Aggregate per-temperature held-out contact error across splits.
    held_ct_sum = np.zeros(n, dtype=float)
    held_ct_cnt = np.zeros(n, dtype=int)

    for sp in splits:
        train_idx = sp["train_idx"]
        val_idx = sp["val_idx"]
        has_val = val_idx.size > 0
        train_temps = temps[train_idx]
        p_obs_ct_train = p_obs_mass[train_idx]
        p_obs_rg_train = (
            p_obs_rg_model_grid[train_idx] if (fit_rg and can_fit_rg) else None
        )

        obj_fn, obj_args = build_objective(
            fit_rg, train_temps, m_centers, p_obs_ct_train, p0_mass, b_fn, loss_fn,
            crg_prob=crg_prob, c_edges_joint=c_edges_joint,
            p_obs_rg_train=p_obs_rg_train, rg_weight=rg_weight,
        )

        rec: Dict[str, Any] = {
            "scheme": sp["scheme"],
            "name": sp["name"],
            "n_train": int(train_idx.size),
            "n_val": int(val_idx.size),
            "train_indices": train_idx.tolist(),
            "val_indices": val_idx.tolist(),
            "train_temps": train_temps.tolist(),
            "val_temps": temps[val_idx].tolist(),
        }

        try:
            best, best_obj = fit_one_split(obj_fn, obj_args, x0s, bounds)
        except RuntimeError as exc:
            rec["optimization_success"] = False
            rec["optimization_message"] = str(exc)
            records.append(rec)
            print(f"  [{sp['name']}] FAILED: {exc}")
            continue

        params = best.x
        rec["optimization_success"] = bool(best.success)
        rec["optimization_message"] = str(best.message)
        rec["optimization_iterations"] = int(best.nit) if hasattr(best, "nit") else None
        rec["optimization_objective"] = float(best_obj)
        rec["boundary_hits"] = count_boundary_hits(params, bounds, param_names)
        for pname, pval in zip(param_names, params):
            rec[pname] = float(pval)
        if derived_Tc_fn is not None:
            tc = derived_Tc_fn(params)
            if tc is not None and np.isfinite(tc):
                rec["Tc"] = float(tc)

        # Per-temperature losses (all temps), then slice by split.
        ct_pt = per_temp_contact_losses(
            temps, m_centers, p_obs_mass, p0_mass, params, b_fn, loss_fn
        )
        rg_pt = None
        if can_fit_rg:
            _, rg_mod_mass = predict_rg_from_joint(
                crg_prob, c_edges_joint, rg_edges_model_lattice, temps, params, b_fn
            )
            rg_pt = per_temp_rg_losses(rg_mod_mass, p_obs_rg_model_grid, loss_fn)

        def _sum_mean(arr: np.ndarray, idx: np.ndarray):
            if idx.size == 0:
                return float("nan"), float("nan")
            s = float(arr[idx].sum())
            return s, s / idx.size

        ct_tr_sum, ct_tr_mean = _sum_mean(ct_pt, train_idx)
        ct_va_sum, ct_va_mean = _sum_mean(ct_pt, val_idx)
        ct_all_sum = float(ct_pt.sum())
        ct_all_mean = ct_all_sum / n
        rec.update(
            contact_train_loss_sum=ct_tr_sum, contact_train_loss_mean=ct_tr_mean,
            contact_val_loss_sum=ct_va_sum, contact_val_loss_mean=ct_va_mean,
            contact_all_loss_sum=ct_all_sum, contact_all_loss_mean=ct_all_mean,
        )

        if rg_pt is not None:
            rg_tr_sum, rg_tr_mean = _sum_mean(rg_pt, train_idx)
            rg_va_sum, rg_va_mean = _sum_mean(rg_pt, val_idx)
            rg_all_sum = float(rg_pt.sum())
            rec.update(
                rg_train_loss_sum=rg_tr_sum, rg_train_loss_mean=rg_tr_mean,
                rg_val_loss_sum=rg_va_sum, rg_val_loss_mean=rg_va_mean,
                rg_all_loss_sum=rg_all_sum, rg_all_loss_mean=rg_all_sum / n,
            )
            # Combined uses the configured rg_weight (objective_combined semantics).
            rec["combined_val_loss_sum"] = (
                ct_va_sum + rg_weight * rg_va_sum if has_val else float("nan")
            )
            rec["combined_val_loss_mean"] = (
                ct_va_mean + rg_weight * rg_va_mean if has_val else float("nan")
            )
        else:
            rec["combined_val_loss_sum"] = ct_va_sum
            rec["combined_val_loss_mean"] = ct_va_mean

        # Per-temperature provenance + aggregation of held-out contact error.
        val_set = set(val_idx.tolist())
        for i in range(n):
            in_val = i in val_set
            per_temp_rows.append({
                "split_name": sp["name"],
                "scheme": sp["scheme"],
                "temp_index": i,
                "temperature": float(temps[i]),
                "in_validation": in_val,
                "contact_loss": float(ct_pt[i]),
                "rg_loss": (float(rg_pt[i]) if rg_pt is not None else None),
            })
            if in_val:
                held_ct_sum[i] += float(ct_pt[i])
                held_ct_cnt[i] += 1

        print(f"  [{sp['name']}] train={train_idx.size} val={val_idx.size} "
              f"combined_val_mean={rec['combined_val_loss_mean']:.5g}"
              + (f" bounds={rec['boundary_hits']}" if rec.get("boundary_hits") else ""))

        records.append(rec)

    # -- write split_sensitivity.csv --
    sens_csv = outdir / "split_sensitivity.csv"
    base_cols = [
        "scheme", "name", "n_train", "n_val",
        "optimization_success", "optimization_iterations", "optimization_objective",
        "boundary_hits",
    ]
    param_cols = list(param_names)
    if any("Tc" in r for r in records):
        param_cols = param_cols + ["Tc"]
    loss_cols = [
        "contact_train_loss_sum", "contact_train_loss_mean",
        "contact_val_loss_sum", "contact_val_loss_mean",
        "contact_all_loss_sum", "contact_all_loss_mean",
        "rg_train_loss_sum", "rg_train_loss_mean",
        "rg_val_loss_sum", "rg_val_loss_mean",
        "rg_all_loss_sum", "rg_all_loss_mean",
        "combined_val_loss_sum", "combined_val_loss_mean",
    ]
    header = base_cols + param_cols + loss_cols

    def _csv_cell(v: Any) -> str:
        if v is None:
            return ""
        if isinstance(v, float):
            return "%.10g" % v if np.isfinite(v) else ""
        if isinstance(v, (list, tuple)):
            return ";".join(str(x) for x in v)
        return str(v)

    with open(sens_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for r in records:
            w.writerow([_csv_cell(r.get(c)) for c in header])
    print(f"Saved: {sens_csv}")

    # -- write split_sensitivity_per_temperature.csv --
    pt_csv = outdir / "split_sensitivity_per_temperature.csv"
    with open(pt_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["split_name", "scheme", "temp_index", "temperature",
                    "in_validation", "contact_loss", "rg_loss"])
        for row in per_temp_rows:
            w.writerow([
                row["split_name"], row["scheme"], row["temp_index"],
                _csv_cell(row["temperature"]), int(row["in_validation"]),
                _csv_cell(row["contact_loss"]), _csv_cell(row["rg_loss"]),
            ])
    print(f"Saved: {pt_csv}")

    # -- parameter stability across splits --
    ok_records = [r for r in records if r.get("optimization_success")]
    stab_keys = list(param_names) + (["Tc"] if any("Tc" in r for r in ok_records) else [])
    stability = summarize_param_stability(ok_records, stab_keys)

    stab_csv = outdir / "split_parameter_stability.csv"
    with open(stab_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["parameter", "mean", "std", "min", "max", "range", "cv", "n_splits"])
        for key in stab_keys:
            st = stability[key]
            w.writerow([
                key, _csv_cell(st["mean"]), _csv_cell(st["std"]),
                _csv_cell(st["min"]), _csv_cell(st["max"]), _csv_cell(st["range"]),
                _csv_cell(st["cv"]), st["n"],
            ])
    print(f"Saved: {stab_csv}")

    # -- aggregated per-temperature held-out contact error --
    held_mean = np.where(held_ct_cnt > 0, held_ct_sum / np.maximum(held_ct_cnt, 1), np.nan)

    # -- summary JSON (strict) --
    summary = {
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "model": ctx["model_name"],
        "loss": loss_name,
        "fit_rg": bool(fit_rg),
        "rg_weight": float(rg_weight),
        "can_fit_rg": bool(can_fit_rg),
        "n_temps": int(n),
        "schemes": schemes,
        "split_seed": int(split_seed),
        "kfold_k": int(args.split_kfold_k),
        "blocked_fraction": float(args.split_blocked_fraction),
        "random_fraction": float(args.split_random_fraction),
        "random_repeats": int(args.split_random_repeats),
        "n_splits": len(records),
        "n_splits_succeeded": len(ok_records),
        "comparison_metric": "mean-per-temperature loss (sums retained for provenance)",
        "splits": [
            {k: (_finite_or_none(v) if isinstance(v, float) else v)
             for k, v in r.items()}
            for r in records
        ],
        "parameter_stability": {
            key: {sk: _finite_or_none(sv) if isinstance(sv, float) else sv
                  for sk, sv in stability[key].items()}
            for key in stab_keys
        },
        "per_temperature_heldout_contact_error": {
            "temperature": temps.tolist(),
            "mean_contact_loss": [_finite_or_none(x) for x in held_mean.tolist()],
            "n_times_held_out": held_ct_cnt.tolist(),
        },
    }
    sens_json = outdir / "split_sensitivity_summary.json"
    with open(sens_json, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, cls=_NpEncoder, allow_nan=False)
    print(f"Saved: {sens_json}")

    # -- plots --
    if args.no_plots:
        return
    if plt is None:
        print("WARNING: matplotlib unavailable; skipping split-sensitivity plots.")
        return
    _plot_split_sensitivity(outdir, records, ok_records, param_names, stab_keys,
                            temps, held_mean, can_fit_rg)


def _plot_split_sensitivity(
    outdir: Path,
    records: List[Dict[str, Any]],
    ok_records: List[Dict[str, Any]],
    param_names: List[str],
    stab_keys: List[str],
    temps: np.ndarray,
    held_mean: np.ndarray,
    can_fit_rg: bool,
) -> None:
    names = [r["name"] for r in ok_records]
    x = np.arange(len(names))

    # 1. Validation contact / Rg / combined loss by split (mean per temp).
    fig, ax = plt.subplots(figsize=(max(7, 0.5 * len(names) + 3), 4.5))
    ct = [r.get("contact_val_loss_mean", np.nan) for r in ok_records]
    comb = [r.get("combined_val_loss_mean", np.nan) for r in ok_records]
    ax.plot(x, ct, "o-", label="contact val (mean/temp)")
    if can_fit_rg:
        rg = [r.get("rg_val_loss_mean", np.nan) for r in ok_records]
        ax.plot(x, rg, "s-", label="Rg val (mean/temp)")
    ax.plot(x, comb, "^-", label="combined val (mean/temp)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.set_ylabel("validation loss (mean per temp)")
    ax.set_title("Validation loss by split")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = outdir / "split_sensitivity_val_loss.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {p}")

    # 2. Parameter estimates by split.
    fig, ax = plt.subplots(figsize=(max(7, 0.5 * len(names) + 3), 4.5))
    for pname in param_names:
        ax.plot(x, [r.get(pname, np.nan) for r in ok_records], "o-", label=pname)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.set_ylabel("parameter value")
    ax.set_title("Parameter estimates by split")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = outdir / "split_sensitivity_params.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {p}")

    # 3. Parameter correlation heatmap across split fits.
    mat = np.array(
        [[r.get(k, np.nan) for k in stab_keys] for r in ok_records], dtype=float
    )
    if mat.shape[0] >= 2 and mat.shape[1] >= 2 and np.all(np.isfinite(mat)):
        corr = np.corrcoef(mat, rowvar=False)
        fig, ax = plt.subplots(figsize=(1.2 * len(stab_keys) + 2, 1.2 * len(stab_keys) + 2))
        im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_xticks(range(len(stab_keys)))
        ax.set_yticks(range(len(stab_keys)))
        ax.set_xticklabels(stab_keys, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(stab_keys, fontsize=8)
        for i in range(len(stab_keys)):
            for j in range(len(stab_keys)):
                ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center",
                        fontsize=7, color="black")
        plt.colorbar(im, ax=ax, label="correlation")
        ax.set_title("Parameter correlation across split fits")
        fig.tight_layout()
        p = outdir / "split_sensitivity_param_correlation.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {p}")

    # 4. Per-temperature held-out prediction error aggregated over splits.
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(temps, held_mean, "o-", ms=3)
    ax.set_xlabel("T")
    ax.set_ylabel("mean held-out contact loss")
    ax.set_title("Per-temperature held-out contact error (averaged over splits)")
    fig.tight_layout()
    p = outdir / "split_sensitivity_per_temperature_error.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {p}")


# ---------------------------------------------------------------------------
# Quick test (synthetic; split-sensitivity + determinism)
# ---------------------------------------------------------------------------

def run_quick_test() -> int:
    failures: List[str] = []

    def check(cond: bool, msg: str) -> None:
        if cond:
            print(f"  PASS: {msg}")
        else:
            print(f"  FAIL: {msg}")
            failures.append(msg)

    if minimize is None:
        print("FAIL: scipy is required for the quick test.")
        return 1

    # ---- synthetic data from a known hs ground truth ----
    rng = np.random.default_rng(0)
    temps = np.linspace(280.0, 360.0, 24)
    m_centers = np.arange(0, 26, dtype=float)
    # smooth, peaked athermal baseline
    p0_mass = np.exp(-0.5 * ((m_centers - 8.0) / 5.0) ** 2)
    p0_mass /= p0_mass.sum()
    h_true, s_true = 800.0, 2.6
    Tref, Tscale = float(temps.mean()), float(temps.max() - temps.min())
    b_hs = make_b_fn("hs", Tref, Tscale)
    p_obs_mass = np.zeros((temps.size, m_centers.size), dtype=float)
    for i, T in enumerate(temps):
        p = model_contact_mass(p0_mass, m_centers, float(T), np.array([h_true, s_true]), b_hs)
        p_obs_mass[i] = p
    loss_fn = _get_loss_fn("js")

    def make_x0s(model: str, seed: int) -> List[np.ndarray]:
        spec = MODEL_REGISTRY[model]
        bounds = spec["bounds"]
        r = np.random.default_rng(seed)
        x0s = [np.array(spec["x0"], dtype=float)]
        for _ in range(3):
            x0s.append(np.array([r.uniform(lo, hi) for lo, hi in bounds], dtype=float))
        return x0s

    def fit_model_on(model: str, train_idx: np.ndarray, seed: int) -> np.ndarray:
        spec = MODEL_REGISTRY[model]
        b_fn = make_b_fn(model, Tref, Tscale)
        obj_fn, obj_args = build_objective(
            False, temps[train_idx], m_centers, p_obs_mass[train_idx], p0_mass, b_fn, loss_fn
        )
        best, _ = fit_one_split(obj_fn, obj_args, make_x0s(model, seed), spec["bounds"])
        return best.x

    print("Quick test 1: deterministic fit under fixed seed")
    all_idx = np.arange(temps.size)
    p_a = fit_model_on("hs", all_idx, seed=123)
    p_b = fit_model_on("hs", all_idx, seed=123)
    check(np.allclose(p_a, p_b, rtol=0, atol=0), "identical hs params across two runs (seed=123)")

    print("Quick test 2: split schemes are valid and reject empties")
    splits = build_split_schemes(
        temps.size,
        ["every_third_phase", "kfold", "blocked_low", "blocked_mid", "blocked_high", "random"],
        kfold_k=4, blocked_fraction=0.25, random_fraction=0.25, random_repeats=3, split_seed=7,
    )
    check(len(splits) == 3 + 4 + 3 + 3, f"expected 13 splits, got {len(splits)}")
    ok = all(s["train_idx"].size > 0 and s["val_idx"].size > 0 for s in splits)
    check(ok, "every split has non-empty train and validation sets")
    try:
        _make_split("x", "bad", np.arange(temps.size), np.array([], dtype=int), temps.size)
        check(False, "empty validation should raise")
    except ValueError:
        check(True, "empty validation set is rejected")

    print("Quick test 3: deterministic random scheme under fixed split_seed")
    s1 = build_split_schemes(temps.size, ["random"], kfold_k=4, blocked_fraction=0.25,
                             random_fraction=0.25, random_repeats=3, split_seed=7)
    s2 = build_split_schemes(temps.size, ["random"], kfold_k=4, blocked_fraction=0.25,
                             random_fraction=0.25, random_repeats=3, split_seed=7)
    same = all(np.array_equal(a["val_idx"], b["val_idx"]) for a, b in zip(s1, s2))
    check(same, "random holdouts identical for identical split_seed")

    print("Quick test 4: flexible model (poly3) less stable than hs under blocked holdouts")
    blocked = build_split_schemes(temps.size, ["blocked_low", "blocked_mid", "blocked_high"],
                                  kfold_k=4, blocked_fraction=0.25, random_fraction=0.25,
                                  random_repeats=3, split_seed=7)

    def spread_of_b(model: str) -> float:
        b_fn = make_b_fn(model, Tref, Tscale)
        vals = []
        for sp in blocked:
            params = fit_model_on(model, sp["train_idx"], seed=123)
            vals.append(b_fn(params, float(temps.mean())))
        return float(np.std(vals))

    hs_spread = spread_of_b("hs")
    poly3_spread = spread_of_b("poly3")
    print(f"    b(Tmean) std across blocked holdouts: hs={hs_spread:.3e}, poly3={poly3_spread:.3e}")
    check(poly3_spread > hs_spread,
          "poly3 b(T) spread across blocked holdouts exceeds hs spread")

    print("Quick test 5: known parameter recovery (hs)")
    rec = fit_model_on("hs", all_idx, seed=123)
    check(abs(rec[0] - h_true) < 5.0 and abs(rec[1] - s_true) < 0.05,
          f"recovered hs params ~ ({h_true},{s_true}): got ({rec[0]:.3f},{rec[1]:.4f})")

    print("Quick test 6: strong-correlation identifiability flag")
    rcorr = np.random.default_rng(1)
    a = rcorr.normal(0, 1, size=200)
    mat = np.column_stack([a, 0.999 * a + 0.01 * rcorr.normal(0, 1, size=200),
                           rcorr.normal(0, 1, size=200)])
    corr = np.corrcoef(mat, rowvar=False)
    flags = correlation_flags(corr, ["p0", "p1", "p2"], 0.9)
    flagged = {(f["param_a"], f["param_b"]) for f in flags}
    check(("p0", "p1") in flagged, "p0~p1 (|r|>0.9) flagged as possibly non-identifiable")
    check(("p0", "p2") not in flagged and ("p1", "p2") not in flagged,
          "uncorrelated pairs not flagged")

    print("Quick test 7: failed-replicate accounting")
    # All restarts fail when the objective is non-finite everywhere.
    def _bad_obj(p, *rest):
        return float("inf")
    n_fail = 0
    try:
        fit_one_split(_bad_obj, (None,), make_x0s("hs", 0), MODEL_REGISTRY["hs"]["bounds"])
    except RuntimeError:
        n_fail = 1
    check(n_fail == 1, "fit_one_split raises when all restarts fail (counted as failure)")
    stats_empty = bootstrap_param_stats(np.array([]), 1.0, 0.95)
    check(stats_empty["n"] == 0 and stats_empty["mean"] is None,
          "bootstrap_param_stats handles zero successful replicates")

    print("Quick test 8: bound-hit detection")
    bnds = [(-10.0, 10.0), (0.0, 5.0)]
    pm = np.array([[10.0, 2.5], [9.9999999, 0.0], [3.0, 0.0]])  # col0 hits hi twice; col1 hits lo twice
    fr = param_bound_fractions(pm, bnds, ["x", "y"])
    check(abs(fr["x"]["at_upper"] - 2.0 / 3.0) < 1e-9, "x at upper bound in 2/3 of fits")
    check(abs(fr["y"]["at_lower"] - 2.0 / 3.0) < 1e-9, "y at lower bound in 2/3 of fits")
    hits = count_boundary_hits(np.array([10.0, 0.0]), bnds, ["x", "y"])
    check("x@hi" in hits and "y@lo" in hits, "count_boundary_hits identifies both bounds")

    print("Quick test 9: Pareto-dominance + knee detection")
    pts = np.array([[1.0, 4.0], [2.0, 2.0], [4.0, 1.0], [3.0, 3.0]])
    mask = pareto_efficient_mask(pts)
    check(mask.tolist() == [True, True, True, False],
          "(3,3) is Pareto-dominated by (2,2); others efficient")
    front = pts[mask]
    front = front[np.argsort(front[:, 0])]
    check(knee_index(front) == 1, "knee at the (2,2) elbow of the frontier")

    print("Quick test 10: increasing Rg weight trades contact fit for Rg fit")
    M = 18
    mc = np.arange(0, M, dtype=float)
    p0 = np.exp(-0.5 * ((mc - 7.0) / 4.0) ** 2); p0 /= p0.sum()
    t2 = np.linspace(280.0, 360.0, 8)
    Tref2, Tscale2 = float(t2.mean()), float(t2.max() - t2.min())
    bfn2 = make_b_fn("hs", Tref2, Tscale2)
    A = np.array([900.0, 3.0]); R = np.array([500.0, 1.5])  # conflicting truths
    obs_ct = np.array([model_contact_mass(p0, mc, float(T), A, bfn2) for T in t2])
    obs_rg = np.array([model_contact_mass(p0, mc, float(T), R, bfn2) for T in t2])
    c_edges2 = np.arange(-0.5, M + 0.5, 1.0)       # identity Rg = contact mapping
    rg_edges2 = np.arange(-0.5, M + 0.5, 1.0)
    crg2 = np.zeros((M, M)); crg2[np.arange(M), np.arange(M)] = p0
    lfn = _get_loss_fn("js")
    spec_hs = MODEL_REGISTRY["hs"]

    def fit_and_score(w: float):
        use_rg = w > 0
        of, oa = build_objective(
            use_rg, t2, mc, obs_ct, p0, bfn2, lfn,
            crg_prob=crg2, c_edges_joint=c_edges2, p_obs_rg_train=obs_rg, rg_weight=w,
        )
        best, _ = fit_one_split(of, oa, make_x0s("hs", 0), spec_hs["bounds"])
        pr = best.x
        ct = float(per_temp_contact_losses(t2, mc, obs_ct, p0, pr, bfn2, lfn).sum())
        _, rgm = predict_rg_from_joint(crg2, c_edges2, rg_edges2, t2, pr, bfn2)
        rg = float(per_temp_rg_losses(rgm, obs_rg, lfn).sum())
        return ct, rg

    c0, r0 = fit_and_score(0.0)
    c1, r1 = fit_and_score(6.0)
    print(f"    w=0: contact={c0:.4g} rg={r0:.4g}  |  w=6: contact={c1:.4g} rg={r1:.4g}")
    check(r1 < r0, "higher Rg weight reduces Rg loss")
    check(c1 > c0, "higher Rg weight worsens contact loss (genuine trade-off)")

    print("Quick test 11: contact-mode fit is unchanged (backward compatibility)")
    # Golden values measured from the pre-scalar-mode implementation (verified
    # bit-identical against fit_lattice_contact_model_2.py).  Any drift here means
    # the contact-histogram code path moved.
    rec_bc = fit_model_on("hs", all_idx, seed=123)
    check(abs(rec_bc[0] - 799.9966075094676) < 1e-9
          and abs(rec_bc[1] - 2.599989342750689) < 1e-12,
          f"contact-only hs fit reproduces the pre-change result to <1e-9: "
          f"got ({rec_bc[0]!r}, {rec_bc[1]!r})")
    obj_bc = objective(rec_bc, temps, m_centers, p_obs_mass, p0_mass, b_hs, loss_fn)
    check(abs(obj_bc - 3.4364269928372306e-11) < 1e-20,
          f"contact-only objective reproduces the pre-change value: {obj_bc:.6e}")

    failures.extend(_run_rg_scalar_tests(check))

    print()
    if failures:
        print(f"QUICK TEST FAILED: {len(failures)} assertion(s) failed.")
        return 1
    print("QUICK TEST PASSED: all assertions passed.")
    return 0


def _make_synthetic_joint(
    n_m: int = 20, n_rg: int = 40, *, coupling: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Synthetic joint baseline P0(m, Rg) with a tunable contact-Rg relationship.

    Rg falls linearly with m when ``coupling`` > 0 (compact states have more
    contacts); ``coupling`` = 0 makes contacts and Rg independent.
    Returns (crg_prob, c_edges, rg_edges) with lattice-unit Rg edges.
    """
    c_edges = np.arange(-0.5, n_m + 0.5, 1.0)
    m_centers = 0.5 * (c_edges[:-1] + c_edges[1:])
    rg_edges = np.linspace(1.0, 5.0, n_rg + 1)
    rg_centers = 0.5 * (rg_edges[:-1] + rg_edges[1:])

    p_m = np.exp(-0.5 * ((m_centers - 6.0) / 3.5) ** 2)
    p_m /= p_m.sum()
    crg = np.zeros((n_m, n_rg), dtype=float)
    for i, m in enumerate(m_centers):
        mu = 4.0 - coupling * 0.18 * m
        cond = np.exp(-0.5 * ((rg_centers - mu) / 0.35) ** 2)
        s = cond.sum()
        crg[i] = p_m[i] * (cond / s if s > 0 else np.ones(n_rg) / n_rg)
    crg /= crg.sum()
    return crg, c_edges, rg_edges


def _run_rg_scalar_tests(check: Callable[[bool, str], None]) -> List[str]:
    """Scalar-Rg(T) mode tests. Fast and deterministic; returns failure messages."""
    local_failures: List[str] = []

    def sub_check(cond: bool, msg: str) -> None:
        check(cond, msg)
        if not cond:
            local_failures.append(msg)

    import tempfile

    print("Scalar-Rg test 1: data loader validation")
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)

        good = tdp / "good.dat"
        good.write_text(
            "# T Rg lo hi\n240 1.34 1.25 1.51\n300 1.05 0.94 1.19\n360 0.71 0.69 0.72\n"
        )
        got = load_rg_mean_file(str(good))
        sub_check(
            got["temps"].size == 3
            and np.allclose(got["temps"], [240, 300, 360])
            and np.allclose(got["rg_target"], [1.34, 1.05, 0.71])
            and np.allclose(got["rg_lower"], [1.25, 0.94, 0.69])
            and np.allclose(got["rg_upper"], [1.51, 1.19, 0.72]),
            "valid 4-column file loads (comment header skipped)",
        )

        cases = [
            ("three_cols.dat", "240 1.34 1.25\n300 1.05 0.94\n360 0.71 0.69\n",
             "wrong column count is rejected"),
            ("nonmono.dat", "240 1.34 1.25 1.51\n360 0.71 0.69 0.72\n300 1.05 0.94 1.19\n",
             "non-monotonic temperatures are rejected"),
            ("dup.dat", "240 1.34 1.25 1.51\n240 1.05 0.94 1.19\n360 0.71 0.69 0.72\n",
             "duplicate temperatures are rejected"),
            ("lo_gt_mid.dat", "240 1.34 1.40 1.51\n300 1.05 0.94 1.19\n360 0.71 0.69 0.72\n",
             "lower > central is rejected"),
            ("mid_gt_hi.dat", "240 1.34 1.25 1.30\n300 1.05 0.94 1.19\n360 0.71 0.69 0.72\n",
             "central > upper is rejected"),
            ("nan.dat", "240 nan 1.25 1.51\n300 1.05 0.94 1.19\n360 0.71 0.69 0.72\n",
             "NaN is rejected"),
            ("short.dat", "240 1.34 1.25 1.51\n300 1.05 0.94 1.19\n",
             "fewer than 3 temperatures is rejected"),
            ("negT.dat", "-5 1.34 1.25 1.51\n300 1.05 0.94 1.19\n360 0.71 0.69 0.72\n",
             "non-positive temperature is rejected"),
        ]
        for fname, body, msg in cases:
            p = tdp / fname
            p.write_text(body)
            try:
                load_rg_mean_file(str(p))
                sub_check(False, msg)
            except ValueError:
                sub_check(True, msg)

    print("Scalar-Rg test 2: unit conversion applied exactly once")
    crg, c_edges, rg_edges = _make_synthetic_joint()
    t3 = np.array([280.0, 320.0, 360.0])
    bfn_t = make_b_fn("tc_scale", float(t3.mean()), float(t3.max() - t3.min()))
    pr = np.array([2.0, 320.0])
    lat, obs, mass = predict_rg_summary_from_joint(
        crg, c_edges, rg_edges, t3, pr, bfn_t,
        rg_scale=0.345, summary="rms", target_units="observed",
    )
    sub_check(np.allclose(obs, 0.345 * lat, rtol=0, atol=1e-14),
              "Rg_observed == 0.345 * Rg_lattice exactly")
    sub_check(np.allclose(mass.sum(axis=1), 1.0),
              "each predicted P(Rg|T) is normalized")
    sub_check(
        np.allclose(
            rg_pred_in_target_units(lat, obs, "observed"), obs, rtol=0, atol=0
        )
        and np.allclose(
            rg_pred_in_target_units(lat, obs, "lattice"), lat, rtol=0, atol=0
        ),
        "target-units selector returns observed vs lattice prediction",
    )
    # rg_scale must not enter the loss when targets are already in lattice units.
    o_a = objective_rg_scalar(
        pr, t3, lat, crg, c_edges, rg_edges, bfn_t,
        n_model_params=2, fixed_rg_scale=0.345, fit_rg_scale=False,
        rg_summary="rms", target_units="lattice", loss_name="mse",
    )
    o_b = objective_rg_scalar(
        pr, t3, lat, crg, c_edges, rg_edges, bfn_t,
        n_model_params=2, fixed_rg_scale=0.999, fit_rg_scale=False,
        rg_summary="rms", target_units="lattice", loss_name="mse",
    )
    sub_check(o_a == o_b == 0.0,
              "target_units=lattice ignores rg_scale inside the loss (and is exact)")
    o_obs = objective_rg_scalar(
        pr, t3, obs, crg, c_edges, rg_edges, bfn_t,
        n_model_params=2, fixed_rg_scale=0.345, fit_rg_scale=False,
        rg_summary="rms", target_units="observed", loss_name="mse",
    )
    sub_check(abs(o_obs) < 1e-24,
              "target_units=observed applies rg_scale exactly once inside the loss")

    print("Scalar-Rg test 3: mean and rms on a known two-bin distribution")
    two_edges = np.array([0.0, 2.0, 4.0])          # centers 1.0 and 3.0
    two_mass = np.array([[0.25, 0.75]])
    centers = 0.5 * (two_edges[:-1] + two_edges[1:])
    got_mean = rg_summary_from_mass(two_mass, centers, "mean")[0]
    got_rms = rg_summary_from_mass(two_mass, centers, "rms")[0]
    exp_mean = 0.25 * 1.0 + 0.75 * 3.0                       # 2.5
    exp_rms = float(np.sqrt(0.25 * 1.0 + 0.75 * 9.0))        # sqrt(7) ~ 2.6458
    sub_check(abs(got_mean - exp_mean) < 1e-15, f"mean == {exp_mean}: got {got_mean}")
    sub_check(abs(got_rms - exp_rms) < 1e-15, f"rms == sqrt(7): got {got_rms}")
    sub_check(got_rms > got_mean, "rms exceeds mean for a spread distribution")

    print("Scalar-Rg test 4: loss functions have exact known values")
    p_ = np.array([1.0, 2.0, 3.0])
    t_ = np.array([1.5, 1.0, 3.0])                 # residuals: -0.5, +1.0, 0.0
    sub_check(
        abs(rg_scalar_objective_value(p_, t_, "mse") - (0.25 + 1.0 + 0.0) / 3.0) < 1e-15,
        "mse == mean(r^2)",
    )
    sub_check(
        abs(rg_scalar_objective_value(p_, t_, "mae") - (0.5 + 1.0 + 0.0) / 3.0) < 1e-15,
        "mae == mean(|r|)",
    )
    # Huber, delta=0.75: |−0.5|<=0.75 -> 0.5*0.25=0.125 ; |1|>0.75 -> 0.75*(1-0.375)=0.46875
    exp_hub = (0.125 + 0.46875 + 0.0) / 3.0
    sub_check(
        abs(rg_scalar_objective_value(p_, t_, "huber", huber_delta=0.75) - exp_hub) < 1e-15,
        f"huber == {exp_hub:.6f} (quadratic below delta, linear above)",
    )
    lo_ = np.array([1.0, 0.5, 2.0])                # sigma^- = t-lo = 0.5, 0.5, 1.0
    hi_ = np.array([2.5, 3.0, 4.0])                # sigma^+ = hi-t = 1.0, 2.0, 1.0
    # r=-0.5 uses sigma^-=0.5 -> 1.0 ; r=+1.0 uses sigma^+=2.0 -> 0.25 ; r=0 -> 0
    exp_rw = (1.0 + 0.25 + 0.0) / 3.0
    sub_check(
        abs(rg_scalar_objective_value(
            p_, t_, "range_weighted", rg_lower=lo_, rg_upper=hi_, range_floor=1e-9
        ) - exp_rw) < 1e-15,
        f"range_weighted picks sigma^- for negative and sigma^+ for positive residuals "
        f"(== {exp_rw:.6f})",
    )
    # Range floor: a zero-width interval must not divide by zero.
    p2_ = np.array([1.5])
    t2_ = np.array([1.0])
    z_ = np.array([1.0])
    got_floor = rg_scalar_objective_value(
        p2_, t2_, "range_weighted", rg_lower=z_, rg_upper=z_, range_floor=0.1
    )
    sub_check(abs(got_floor - (0.5 / 0.1) ** 2) < 1e-12,
              f"range floor caps a zero-width interval: got {got_floor} == 25.0")
    sub_check(np.isfinite(got_floor), "range floor keeps the loss finite")

    print("Scalar-Rg test 5: synthetic parameter recovery (tc_scale)")
    crg_s, ce_s, re_s = _make_synthetic_joint(coupling=1.0)
    temps_s = np.linspace(270.0, 360.0, 12)
    Tref_s, Tscale_s = float(temps_s.mean()), float(temps_s.max() - temps_s.min())
    bfn_s = make_b_fn("tc_scale", Tref_s, Tscale_s)
    true_p = np.array([3.0, 315.0])
    true_scale = 0.345
    _, tgt_obs, _ = predict_rg_summary_from_joint(
        crg_s, ce_s, re_s, temps_s, true_p, bfn_s,
        rg_scale=true_scale, summary="rms", target_units="observed",
    )
    cfg_s = {
        "crg_prob": crg_s, "c_edges": ce_s, "rg_edges_lattice": re_s, "b_fn": bfn_s,
        "n_model_params": 2, "rg_scale": true_scale, "fit_rg_scale": False,
        "rg_summary": "rms", "target_units": "observed", "loss_name": "mse",
        "huber_delta": 0.05, "range_floor": 0.01,
    }
    of_s = _rg_scalar_objective_factory(cfg_s)
    ps_fixed = build_rg_scalar_param_spec(
        "tc_scale", fit_rg_scale=False, rg_scale=true_scale,
        rg_scale_min=0.25, rg_scale_max=0.55, n_restarts=6, seed=5,
    )
    dummy = np.zeros_like(tgt_obs)
    best_s, obj_s = fit_one_split(
        of_s, (temps_s, tgt_obs, dummy, dummy), ps_fixed["x0s"], ps_fixed["bounds"]
    )
    sub_check(abs(best_s.x[0] - 3.0) < 0.15 and abs(best_s.x[1] - 315.0) < 1.5,
              f"fixed-scale recovery of tc_scale (A=3, Tc=315): "
              f"got (A={best_s.x[0]:.4f}, Tc={best_s.x[1]:.3f}), obj={obj_s:.3e}")

    cfg_f = dict(cfg_s); cfg_f["fit_rg_scale"] = True
    of_f = _rg_scalar_objective_factory(cfg_f)
    ps_free = build_rg_scalar_param_spec(
        "tc_scale", fit_rg_scale=True, rg_scale=true_scale,
        rg_scale_min=0.25, rg_scale_max=0.55, n_restarts=10, seed=5,
    )
    sub_check(ps_free["param_names"] == ["A", "Tc", "rg_scale"],
              "fitted scale is appended last; registry order preserved")
    sub_check(ps_fixed["param_names"] == ["A", "Tc"],
              "fixed-scale mode keeps the registry parameter names untouched")
    sub_check(
        all(ps_free["bounds"][2][0] <= x[2] <= ps_free["bounds"][2][1]
            for x in ps_free["x0s"]),
        "every restart samples rg_scale inside its bounds",
    )
    sub_check(abs(ps_free["x0s"][0][2] - true_scale) < 1e-12,
              "first restart uses --rg-scale as the scale's initial guess")
    best_f, _ = fit_one_split(
        of_f, (temps_s, tgt_obs, dummy, dummy), ps_free["x0s"], ps_free["bounds"]
    )
    sub_check(abs(best_f.x[2] - true_scale) < 0.02,
              f"free-scale recovery of rg_scale=0.345: got {best_f.x[2]:.4f}")
    sub_check(abs(best_f.x[1] - 315.0) < 3.0,
              f"free-scale run still recovers Tc~315: got {best_f.x[1]:.3f}")

    print("Scalar-Rg test 6: split determinism under fixed seeds")
    a1 = build_rg_scalar_param_spec("tc_scale", fit_rg_scale=True, rg_scale=0.345,
                                    rg_scale_min=0.25, rg_scale_max=0.55,
                                    n_restarts=6, seed=11)
    a2 = build_rg_scalar_param_spec("tc_scale", fit_rg_scale=True, rg_scale=0.345,
                                    rg_scale_min=0.25, rg_scale_max=0.55,
                                    n_restarts=6, seed=11)
    sub_check(all(np.array_equal(x, y) for x, y in zip(a1["x0s"], a2["x0s"])),
              "identical restart initial guesses for identical seed")
    tr_i, va_i = _resolve_split_indices(temps_s.size, 3, None, None)
    r1, _ = fit_one_split(of_s, (temps_s[tr_i], tgt_obs[tr_i], dummy[tr_i], dummy[tr_i]),
                          ps_fixed["x0s"], ps_fixed["bounds"])
    r2, _ = fit_one_split(of_s, (temps_s[tr_i], tgt_obs[tr_i], dummy[tr_i], dummy[tr_i]),
                          ps_fixed["x0s"], ps_fixed["bounds"])
    sub_check(np.array_equal(r1.x, r2.x),
              "identical fit results for identical optimizer/split seeds")

    print("Scalar-Rg test 7: reachability diagnostics flag an unreachable target")
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        m_c = 0.5 * (ce_s[:-1] + ce_s[1:])
        r_c = 0.5 * (re_s[:-1] + re_s[1:])
        reach = np.array([
            joint_reweight_stats(crg_s, m_c, r_c, float(b), "rms")["pred_rg_lattice"]
            for b in np.linspace(-10, 10, 401)
        ])
        far = float(reach.max()) * 3.0                    # far above anything reachable
        bad_tgt_lat = np.array([far, far, far])
        fs = run_rg_feasibility_scan(
            crg_s, ce_s, re_s, bad_tgt_lat, 0.345 * bad_tgt_lat,
            rg_scale=0.345, summary="rms", bias_min=-10.0, bias_max=10.0,
            bias_points=401, outdir=tdp, make_plots=False,
        )
        sub_check(fs["target_within_reachable_range"] is False,
                  "unreachable target sets target_within_reachable_range=False")
        sub_check(fs["n_targets_outside_reachable"] == 3,
                  "all 3 unreachable targets are counted")
        sub_check(any("outside" in w for w in fs["warnings"]),
                  "unreachable target produces an explicit warning")
        sub_check((tdp / "rg_feasibility.csv").exists()
                  and (tdp / "rg_feasibility_summary.json").exists(),
                  "feasibility CSV and JSON are written")

        # A reachable target must NOT be flagged.
        ok_tgt_lat = np.array([float(np.median(reach))] * 3)
        fs_ok = run_rg_feasibility_scan(
            crg_s, ce_s, re_s, ok_tgt_lat, 0.345 * ok_tgt_lat,
            rg_scale=0.345, summary="rms", bias_min=-10.0, bias_max=10.0,
            bias_points=401, outdir=tdp, make_plots=False,
        )
        sub_check(fs_ok["target_within_reachable_range"] is True,
                  "a reachable target is not flagged as unreachable")

        print("Scalar-Rg test 8: mean-summary derivative identity d<Rg>/db = -Cov(Rg,m)")
        fs_mean = run_rg_feasibility_scan(
            crg_s, ce_s, re_s, ok_tgt_lat, 0.345 * ok_tgt_lat,
            rg_scale=0.345, summary="mean", bias_min=-4.0, bias_max=4.0,
            bias_points=401, outdir=tdp, make_plots=False,
        )
        dc = fs_mean["derivative_check"]
        sub_check(dc["checked"] is True and dc["agrees"] is True,
                  f"d<Rg>_b/db == -Cov_b(Rg,m) verified numerically "
                  f"(max rel diff {dc['max_abs_relative_difference']:.3g})")
        sub_check(fs["derivative_check"]["checked"] is False,
                  "the mean-only identity is NOT claimed for the rms summary")

        print("Scalar-Rg test 9: weak contact-Rg coupling is flagged")
        crg_w, ce_w, re_w = _make_synthetic_joint(coupling=0.0)
        fs_w = run_rg_feasibility_scan(
            crg_w, ce_w, re_w, np.array([3.0, 3.0]), np.array([1.035, 1.035]),
            rg_scale=0.345, summary="rms", bias_min=-10.0, bias_max=10.0,
            bias_points=201, outdir=tdp, make_plots=False,
        )
        sub_check(abs(fs_w["baseline_contact_rg_correlation"]) < 0.1,
                  f"independent contacts/Rg give ~0 correlation: "
                  f"{fs_w['baseline_contact_rg_correlation']:.3g}")
        sub_check(any("weak" in w.lower() for w in fs_w["warnings"]),
                  "weak contact-Rg coupling produces a warning")

        print("Scalar-Rg test 10: generated JSON is strict (no NaN/Infinity)")
        for name in ("rg_feasibility_summary.json",):
            txt = (tdp / name).read_text()
            parsed = json.loads(txt)          # raises if not valid JSON
            sub_check(isinstance(parsed, dict), f"{name} parses with the json module")
            sub_check(
                "NaN" not in txt and "Infinity" not in txt,
                f"{name} contains no NaN/Infinity tokens",
            )

    print("Scalar-Rg test 11: transition descriptors are distinct and correct")
    tm = rg_curve_transition_metrics(
        crg_s, ce_s, re_s, true_p, bfn_s, 270.0, 360.0,
        rg_scale=0.345, summary="rms", target_units="observed", n_grid=1001,
    )
    zc = bias_zero_crossings(bfn_s, true_p, 270.0, 360.0)
    sub_check(len(zc) == 1 and abs(zc[0] - 315.0) < 0.2,
              f"tc_scale b(T)=0 crossing found at Tc=315: got {zc}")
    sub_check(tm["grid"].size >= 1001, "dense Rg(T) grid has at least 1001 points")
    sub_check(tm["rg_max_negative_slope"] > 0.0,
              "the fitted curve has a genuine collapse (-dRg/dT > 0)")
    sub_check(270.0 <= tm["T_rg_max_slope"] <= 360.0,
              "T_rg_max_slope lies inside the observed interval")
    sub_check(abs(tm["T_rg_max_slope"] - zc[0]) > 1e-6,
              f"T_rg_max_slope ({tm['T_rg_max_slope']:.4g}) is distinct from "
              f"T_bias_zero ({zc[0]:.4g}) for a finite chain")

    return local_failures


# ---------------------------------------------------------------------------
# Parameter-uncertainty helpers
# ---------------------------------------------------------------------------

def _percentile_ci(arr: np.ndarray, confidence: float) -> Tuple[float, float]:
    """Two-sided percentile confidence interval at the given confidence level."""
    alpha = 1.0 - confidence
    lo = float(np.percentile(arr, 100.0 * alpha / 2.0))
    hi = float(np.percentile(arr, 100.0 * (1.0 - alpha / 2.0)))
    return lo, hi


def bootstrap_param_stats(
    values: np.ndarray, fitted: Optional[float], confidence: float
) -> Dict[str, Any]:
    """Summarize a bootstrap distribution for one parameter/derived quantity.

    Reports fitted value, bootstrap mean/median/std, bias vs the original fit,
    percentile CI, and coefficient of variation (when the mean is not ~0).
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    fitted_f = (
        float(fitted) if (fitted is not None and np.isfinite(float(fitted))) else None
    )
    if v.size == 0:
        return {
            "fitted": fitted_f, "mean": None, "median": None, "std": None,
            "bias": None, "ci_low": None, "ci_high": None,
            "confidence": float(confidence), "cv": None, "n": 0,
        }
    mean = float(v.mean())
    median = float(np.median(v))
    std = float(v.std())
    lo, hi = _percentile_ci(v, confidence)
    cv = float(std / abs(mean)) if abs(mean) > 1e-12 else None
    bias = float(mean - fitted_f) if fitted_f is not None else None
    return {
        "fitted": fitted_f, "mean": mean, "median": median, "std": std,
        "bias": bias, "ci_low": lo, "ci_high": hi,
        "confidence": float(confidence), "cv": cv, "n": int(v.size),
    }


def param_bound_fractions(
    param_matrix: np.ndarray,
    bounds: List[Tuple[float, float]],
    param_names: List[str],
    *,
    atol: float = 1e-6,
    rtol: float = 1e-4,
) -> Dict[str, Dict[str, float]]:
    """Fraction of bootstrap fits at/numerically near each parameter's bounds."""
    out: Dict[str, Dict[str, float]] = {}
    n = param_matrix.shape[0] if param_matrix.ndim == 2 else 0
    for j, (name, (lo, hi)) in enumerate(zip(param_names, bounds)):
        if n == 0:
            out[name] = {"at_lower": 0.0, "at_upper": 0.0, "at_any": 0.0}
            continue
        col = param_matrix[:, j]
        span = abs(hi - lo)
        tol = atol + rtol * span
        at_lo = np.isfinite(lo) & (np.abs(col - lo) <= tol)
        at_hi = np.isfinite(hi) & (np.abs(hi - col) <= tol)
        out[name] = {
            "at_lower": float(np.mean(at_lo)),
            "at_upper": float(np.mean(at_hi)),
            "at_any": float(np.mean(at_lo | at_hi)),
        }
    return out


def correlation_flags(
    corr: np.ndarray, names: List[str], threshold: float
) -> List[Dict[str, Any]]:
    """List off-diagonal parameter pairs with |correlation| >= threshold."""
    flags: List[Dict[str, Any]] = []
    k = len(names)
    if corr.ndim != 2 or corr.shape != (k, k):
        return flags
    for i in range(k):
        for j in range(i + 1, k):
            c = float(corr[i, j])
            if np.isfinite(c) and abs(c) >= threshold:
                flags.append({"param_a": names[i], "param_b": names[j],
                              "correlation": c})
    return flags


def numerical_hessian(
    f: Callable[[np.ndarray], float], x: np.ndarray, *, rel_step: float = 1e-4
) -> np.ndarray:
    """Central-difference Hessian of scalar f at x (objective curvature)."""
    x = np.asarray(x, dtype=float)
    n = x.size
    h = rel_step * np.maximum(np.abs(x), 1.0)
    f0 = float(f(x))
    H = np.zeros((n, n), dtype=float)
    for i in range(n):
        xi = x.copy(); xi[i] += h[i]
        xj = x.copy(); xj[i] -= h[i]
        H[i, i] = (float(f(xi)) - 2.0 * f0 + float(f(xj))) / (h[i] * h[i])
    for i in range(n):
        for j in range(i + 1, n):
            xpp = x.copy(); xpp[i] += h[i]; xpp[j] += h[j]
            xpm = x.copy(); xpm[i] += h[i]; xpm[j] -= h[j]
            xmp = x.copy(); xmp[i] -= h[i]; xmp[j] += h[j]
            xmm = x.copy(); xmm[i] -= h[i]; xmm[j] -= h[j]
            val = (float(f(xpp)) - float(f(xpm)) - float(f(xmp)) + float(f(xmm))) / (
                4.0 * h[i] * h[j]
            )
            H[i, j] = val
            H[j, i] = val
    return H


def hessian_diagnostics(H: np.ndarray) -> Dict[str, Any]:
    """Eigenvalues, condition number, and curvature length-scales of H.

    These are LOCAL OBJECTIVE-CURVATURE diagnostics, not formal standard errors:
    the JS/KL objective on normalized histograms is not a log-likelihood, so the
    inverse Hessian does not carry a covariance interpretation.
    """
    H = np.asarray(H, dtype=float)
    H = 0.5 * (H + H.T)
    try:
        eigvals = np.linalg.eigvalsh(H)
    except np.linalg.LinAlgError:
        eigvals = np.array([np.nan])
    finite = eigvals[np.isfinite(eigvals)]
    pos_def = bool(finite.size == eigvals.size and np.all(finite > 0))
    abs_eig = np.abs(finite)
    nz = abs_eig[abs_eig > 0]
    cond = float(nz.max() / nz.min()) if nz.size > 0 and nz.min() > 0 else None
    # Curvature length-scales: sqrt of diagonal of inverse Hessian when pos-def.
    curvature_scales: Optional[List[float]] = None
    if pos_def:
        try:
            inv = np.linalg.inv(H)
            diag = np.diag(inv)
            if np.all(diag > 0):
                curvature_scales = [float(np.sqrt(d)) for d in diag]
        except np.linalg.LinAlgError:
            curvature_scales = None
    return {
        "eigenvalues": [float(e) for e in eigvals],
        "condition_number": cond,
        "positive_definite": pos_def,
        "curvature_length_scales": curvature_scales,
        "note": (
            "Local objective-curvature diagnostics only. The JS/KL objective on "
            "normalized histograms is NOT a log-likelihood; inverse-Hessian "
            "quantities are curvature scales, not statistical standard errors."
        ),
    }


def restart_stability(
    restart_records: List[Dict[str, Any]], *, atol: float = 1e-8, rtol: float = 1e-6
) -> Dict[str, Any]:
    """Summarize whether random restarts converged to one or several minima."""
    ok = [r for r in restart_records if r.get("success") and r.get("objective") is not None]
    objs = np.array([r["objective"] for r in ok], dtype=float)
    n_total = len(restart_records)
    n_ok = len(ok)
    if n_ok == 0:
        return {"n_restarts": n_total, "n_success": 0, "best_objective": None,
                "distinct_minima": None, "n_distinct_objectives": 0,
                "max_param_spread": None}
    best = float(objs.min())
    tol = atol + rtol * abs(best)
    distinct = bool(np.any(objs - best > tol))
    # Count distinct objective levels (clustered by tol).
    sorted_objs = np.sort(objs)
    levels = 1
    for k in range(1, sorted_objs.size):
        if sorted_objs[k] - sorted_objs[k - 1] > tol:
            levels += 1
    # Max pairwise parameter spread among successful restarts.
    P = np.array([r["params"] for r in ok], dtype=float)
    spread = 0.0
    if P.shape[0] >= 2:
        for a in range(P.shape[0]):
            for b in range(a + 1, P.shape[0]):
                spread = max(spread, float(np.linalg.norm(P[a] - P[b])))
    return {
        "n_restarts": n_total, "n_success": n_ok, "best_objective": best,
        "distinct_minima": distinct, "n_distinct_objectives": int(levels),
        "max_param_spread": float(spread),
    }


def _predicted_means(
    temps: np.ndarray,
    m_centers: np.ndarray,
    p0_mass: np.ndarray,
    params: np.ndarray,
    b_fn: Callable[[np.ndarray, float], float],
) -> np.ndarray:
    """Predicted mean contacts at every temperature for one parameter set."""
    out = np.empty(temps.size, dtype=float)
    for i, T in enumerate(temps):
        p = model_contact_mass(p0_mass, m_centers, float(T), params, b_fn)
        out[i] = float((m_centers * p).sum())
    return out


# ---------------------------------------------------------------------------
# Bootstrap uncertainty driver (extends the existing temperature-resampling
# bootstrap; does not introduce a second pathway)
# ---------------------------------------------------------------------------

def run_bootstrap_uncertainty(args: argparse.Namespace, ctx: Dict[str, Any]) -> None:
    temps = ctx["temps"]
    m_centers = ctx["m_centers"]
    p_obs_mass = ctx["p_obs_mass"]
    p0_mass = ctx["p0_mass"]
    b_fn = ctx["b_fn"]
    loss_fn = ctx["loss_fn"]
    spec = ctx["spec"]
    param_names = ctx["param_names"]
    bounds = ctx["bounds"]
    x0s = ctx["x0s"]
    fit_rg = ctx["fit_rg"]
    rg_weight = float(ctx["rg_weight"])
    can_fit_rg = ctx["can_fit_rg"]
    crg_prob = ctx["crg_prob"]
    c_edges_joint = ctx["c_edges_joint"]
    rg_edges_model_lattice = ctx["rg_edges_model_lattice"]
    p_obs_rg_model_grid = ctx["p_obs_rg_model_grid"]
    outdir = ctx["outdir"]
    train_idx = ctx["train_idx"]
    val_idx = ctx["val_idx"]
    has_val = ctx["has_val"]
    train_temps = ctx["train_temps"]
    p_obs_ct_train = ctx["p_obs_ct_train"]
    p_obs_rg_train = ctx["p_obs_rg_train"]
    params_fit = ctx["params_fit"]
    Tc_derived = ctx["Tc_derived"]
    rg_scale = float(ctx["rg_scale"])
    confidence = float(args.bootstrap_confidence)
    derived_Tc_fn = spec["derived_Tc"]

    boot_csv_path = outdir / "bootstrap_params.csv"
    boot_json_path = outdir / "bootstrap_summary.json"

    bseed = args.bootstrap_seed if args.bootstrap_seed is not None else args.seed
    bstrap_rng = np.random.default_rng(bseed)
    n_train = len(train_idx)

    # observed mean contacts / Rg on the model grids (for band-vs-target plots)
    obs_mean_contacts = (m_centers[None, :] * p_obs_mass).sum(axis=1)
    rg_centers_model_lattice = (
        0.5 * (rg_edges_model_lattice[:-1] + rg_edges_model_lattice[1:])
        if (can_fit_rg and rg_edges_model_lattice is not None) else None
    )
    obs_mean_rg = None
    if can_fit_rg and rg_centers_model_lattice is not None:
        rg_centers_scaled = rg_scale * rg_centers_model_lattice
        obs_mean_rg = (rg_centers_scaled[None, :] * p_obs_rg_model_grid).sum(axis=1)

    boot_records: List[Dict[str, Any]] = []
    n_boot_failed = 0
    b_T_list: List[np.ndarray] = []
    meanC_list: List[np.ndarray] = []
    meanRg_list: List[np.ndarray] = []

    print(f"\nBootstrap ({args.bootstrap} replicates, method={args.bootstrap_method}, "
          f"seed={bseed}, confidence={confidence}):")
    print("  (empirical temperature-resampling uncertainty; contacts and Rg are "
          "resampled as paired observations)")

    for bi in range(args.bootstrap):
        # Resample training indices with replacement; contacts and Rg stay paired.
        local_idx = bstrap_rng.integers(0, n_train, size=n_train)
        boot_temps_b = train_temps[local_idx]
        boot_ct_b = p_obs_ct_train[local_idx]
        boot_rg_b = p_obs_rg_train[local_idx] if (fit_rg and can_fit_rg) else None

        obj_fn_b, obj_args_b = build_objective(
            fit_rg, boot_temps_b, m_centers, boot_ct_b, p0_mass, b_fn, loss_fn,
            crg_prob=crg_prob, c_edges_joint=c_edges_joint,
            p_obs_rg_train=boot_rg_b, rg_weight=rg_weight,
        )

        try:
            best_b, best_b_val = fit_one_split(obj_fn_b, obj_args_b, x0s, bounds)
        except (RuntimeError, Exception) as exc:  # noqa: B014 - count all failures
            print(f"  replicate {bi:4d}: FAILED ({exc})")
            n_boot_failed += 1
            continue

        params_b = best_b.x

        # Contact losses on ORIGINAL (non-resampled) temperature sets.
        ct_pt = per_temp_contact_losses(
            temps, m_centers, p_obs_mass, p0_mass, params_b, b_fn, loss_fn
        )
        train_loss_b = float(ct_pt[train_idx].sum())
        val_loss_b = float(ct_pt[val_idx].sum()) if has_val else float("nan")
        all_loss_b = float(ct_pt.sum())

        rg_train_b = rg_val_b = rg_all_b = float("nan")
        if can_fit_rg:
            _, rg_mod_b = predict_rg_from_joint(
                crg_prob, c_edges_joint, rg_edges_model_lattice, temps, params_b, b_fn
            )
            rg_pt = per_temp_rg_losses(rg_mod_b, p_obs_rg_model_grid, loss_fn)
            rg_train_b = float(rg_pt[train_idx].sum())
            rg_val_b = float(rg_pt[val_idx].sum()) if has_val else float("nan")
            rg_all_b = float(rg_pt.sum())
            # mean Rg prediction band
            meanRg_list.append(
                rg_scale * (rg_centers_model_lattice[None, :] * rg_mod_b).sum(axis=1)
            )

        # Combined losses use the configured rg_weight (objective_combined form).
        def _combined(ct, rg):
            return ct + rg_weight * rg if (can_fit_rg and np.isfinite(rg)) else ct
        comb_train_b = _combined(train_loss_b, rg_train_b)
        comb_val_b = _combined(val_loss_b, rg_val_b)
        comb_all_b = _combined(all_loss_b, rg_all_b)

        Tc_b: Optional[float] = None
        if derived_Tc_fn is not None:
            Tc_b = derived_Tc_fn(params_b)

        record: Dict[str, Any] = {"bootstrap_index": bi}
        for pname, pval in zip(param_names, params_b):
            record[pname] = float(pval)
        if Tc_b is not None and np.isfinite(Tc_b):
            record["Tc"] = float(Tc_b)
        record["train_loss"] = train_loss_b
        if has_val:
            record["validation_loss"] = val_loss_b
        record["all_loss"] = all_loss_b
        if can_fit_rg:
            record["rg_train_loss"] = rg_train_b
            if has_val:
                record["rg_validation_loss"] = rg_val_b
            record["rg_all_loss"] = rg_all_b
            record["combined_train_loss"] = comb_train_b
            if has_val:
                record["combined_validation_loss"] = comb_val_b
            record["combined_all_loss"] = comb_all_b
        record["objective"] = float(best_b_val)
        boot_records.append(record)

        # Prediction bands on the full temperature ladder.
        b_T_list.append(np.array([b_fn(params_b, float(T)) for T in temps]))
        meanC_list.append(_predicted_means(temps, m_centers, p0_mass, params_b, b_fn))

        interval = max(1, args.bootstrap // 5)
        if (bi + 1) % interval == 0 or bi == args.bootstrap - 1:
            print(f"  {bi + 1}/{args.bootstrap} done")

    n_boot_success = len(boot_records)
    if n_boot_success == 0:
        print(
            f"WARNING: All {args.bootstrap} bootstrap replicates failed. "
            f"bootstrap outputs not saved."
        )
        return
    if n_boot_failed > 0:
        print(f"  {n_boot_failed} replicate(s) failed and were excluded.")

    # ---- bootstrap_params.csv (extended with Rg/combined losses) ----
    boot_header = list(boot_records[0].keys())
    for rec in boot_records:  # union of keys preserving first-seen order
        for k in rec:
            if k not in boot_header:
                boot_header.append(k)
    with open(boot_csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(boot_header)
        for rec in boot_records:
            writer.writerow([rec.get(col, "") for col in boot_header])
    print(f"Saved: {boot_csv_path}")

    # ---- per-parameter / derived statistics ----
    fitted_vals = {pn: float(pv) for pn, pv in zip(param_names, params_fit)}
    param_matrix = np.array(
        [[r[pn] for pn in param_names] for r in boot_records], dtype=float
    )
    param_boot_stats: Dict[str, Dict[str, Any]] = {}
    for j, pn in enumerate(param_names):
        param_boot_stats[pn] = bootstrap_param_stats(
            param_matrix[:, j], fitted_vals[pn], confidence
        )
    derived_boot_stats: Dict[str, Dict[str, Any]] = {}
    if derived_Tc_fn is not None:
        tc_arr = np.array([r["Tc"] for r in boot_records if "Tc" in r], dtype=float)
        if tc_arr.size > 0:
            derived_boot_stats["Tc"] = bootstrap_param_stats(
                tc_arr, Tc_derived, confidence
            )

    bound_fracs = param_bound_fractions(param_matrix, bounds, param_names)
    frac_success = float(n_boot_success) / float(args.bootstrap)

    # ---- covariance / correlation matrices + identifiability flags ----
    if param_matrix.shape[0] >= 2 and len(param_names) >= 1:
        cov = np.atleast_2d(np.cov(param_matrix, rowvar=False))
        corr = np.atleast_2d(np.corrcoef(param_matrix, rowvar=False))
    else:
        cov = np.full((len(param_names), len(param_names)), np.nan)
        corr = np.full((len(param_names), len(param_names)), np.nan)
    corr_threshold = float(args.bootstrap_correlation_threshold)
    flags = correlation_flags(corr, param_names, corr_threshold)

    cov_csv = outdir / "bootstrap_covariance.csv"
    corr_csv = outdir / "bootstrap_correlation.csv"
    for path, mat in ((cov_csv, cov), (corr_csv, corr)):
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow([""] + list(param_names))
            for i, pn in enumerate(param_names):
                row = [pn]
                for jx in range(len(param_names)):
                    val = float(mat[i, jx]) if mat.ndim == 2 else float("nan")
                    row.append("%.10g" % val if np.isfinite(val) else "")
                w.writerow(row)
        print(f"Saved: {path}")

    # ---- loss distributions + CIs ----
    def _loss_dist(key: str) -> Optional[Dict[str, Any]]:
        vals = np.array([r[key] for r in boot_records if key in r], dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return None
        lo, hi = _percentile_ci(vals, confidence)
        return {"mean": float(vals.mean()), "median": float(np.median(vals)),
                "std": float(vals.std()), "ci_low": lo, "ci_high": hi,
                "n": int(vals.size)}
    loss_keys = ["train_loss", "validation_loss", "all_loss"]
    if can_fit_rg:
        loss_keys += ["rg_train_loss", "rg_validation_loss", "rg_all_loss",
                      "combined_train_loss", "combined_validation_loss",
                      "combined_all_loss"]
    loss_boot_stats = {k: _loss_dist(k) for k in loss_keys}
    loss_boot_stats = {k: v for k, v in loss_boot_stats.items() if v is not None}

    # ---- prediction bands (median + CI per temperature) ----
    def _band(stack: List[np.ndarray]):
        if not stack:
            return None
        A = np.vstack(stack)
        lo_pct = 100.0 * (1.0 - confidence) / 2.0
        hi_pct = 100.0 * (1.0 - (1.0 - confidence) / 2.0)
        return {
            "median": np.median(A, axis=0),
            "lo": np.percentile(A, lo_pct, axis=0),
            "hi": np.percentile(A, hi_pct, axis=0),
            "stack": A,
        }
    b_band = _band(b_T_list)
    c_band = _band(meanC_list)
    rg_band = _band(meanRg_list) if can_fit_rg else None

    # bootstrap_bands_by_temperature.csv (always written when bootstrap runs)
    bands_csv = outdir / "bootstrap_bands_by_temperature.csv"
    with open(bands_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        header = ["temperature",
                  "b_median", "b_lo", "b_hi",
                  "mean_contacts_median", "mean_contacts_lo", "mean_contacts_hi",
                  "mean_contacts_obs"]
        if rg_band is not None:
            header += ["mean_rg_median", "mean_rg_lo", "mean_rg_hi", "mean_rg_obs"]
        w.writerow(header)
        for i in range(temps.size):
            row = [
                "%.10g" % float(temps[i]),
                "%.10g" % b_band["median"][i], "%.10g" % b_band["lo"][i], "%.10g" % b_band["hi"][i],
                "%.10g" % c_band["median"][i], "%.10g" % c_band["lo"][i], "%.10g" % c_band["hi"][i],
                "%.10g" % float(obs_mean_contacts[i]),
            ]
            if rg_band is not None:
                row += [
                    "%.10g" % rg_band["median"][i], "%.10g" % rg_band["lo"][i],
                    "%.10g" % rg_band["hi"][i], "%.10g" % float(obs_mean_rg[i]),
                ]
            w.writerow(row)
    print(f"Saved: {bands_csv}")

    # bootstrap_prediction_bands.npz (full per-replicate arrays; gated by flag)
    if args.bootstrap_save_prediction_bands:
        bands_npz = outdir / "bootstrap_prediction_bands.npz"
        npz_kwargs: Dict[str, Any] = {
            "temps": temps,
            "confidence": np.array(confidence),
            "b_T_replicates": b_band["stack"],
            "b_T_median": b_band["median"], "b_T_lo": b_band["lo"], "b_T_hi": b_band["hi"],
            "mean_contacts_replicates": c_band["stack"],
            "mean_contacts_median": c_band["median"],
            "mean_contacts_lo": c_band["lo"], "mean_contacts_hi": c_band["hi"],
            "mean_contacts_obs": obs_mean_contacts,
            "param_matrix": param_matrix,
            "param_names": np.array(param_names),
        }
        if rg_band is not None:
            npz_kwargs.update(
                mean_rg_replicates=rg_band["stack"],
                mean_rg_median=rg_band["median"],
                mean_rg_lo=rg_band["lo"], mean_rg_hi=rg_band["hi"],
                mean_rg_obs=obs_mean_rg,
            )
        np.savez_compressed(bands_npz, **npz_kwargs)
        print(f"Saved: {bands_npz}")

    # ---- bootstrap_summary.json (extends the historical schema) ----
    def _stat_json(d: Optional[Dict[str, Any]]):
        if d is None:
            return None
        return {k: (_finite_or_none(v) if isinstance(v, float) else v)
                for k, v in d.items()}

    boot_summary: Dict[str, Any] = {
        # historical keys (kept for backward compatibility)
        "n_bootstrap": int(args.bootstrap),
        "n_success": int(n_boot_success),
        "n_failed": int(n_boot_failed),
        "bootstrap_seed": int(bseed),
        "model": ctx["model_name"],
        "loss": ctx["loss_name"],
        "fit_rg": bool(fit_rg),
        "rg_weight": float(rg_weight),
        "params": {pn: _stat_json(param_boot_stats[pn]) for pn in param_names},
        "derived": {k: _stat_json(v) for k, v in derived_boot_stats.items()},
        "losses": {k: _stat_json(v) for k, v in loss_boot_stats.items()},
        # new keys
        "uncertainty_kind": "empirical temperature-resampling bootstrap",
        "uncertainty_note": (
            "These intervals reflect sensitivity to which temperatures were "
            "sampled, not likelihood-based statistical error. The JS/KL objective "
            "on normalized histograms is not a log-likelihood and the histograms "
            "are not raw independent counts, so AIC/BIC and formal standard errors "
            "are intentionally not reported."
        ),
        "bootstrap_method": str(args.bootstrap_method),
        "confidence": confidence,
        "fraction_successful": frac_success,
        "param_bound_fractions": bound_fracs,
        "covariance": {
            "order": list(param_names),
            "matrix": [[_finite_or_none(float(cov[i, j])) for j in range(len(param_names))]
                       for i in range(len(param_names))] if cov.ndim == 2 else None,
        },
        "correlation": {
            "order": list(param_names),
            "matrix": [[_finite_or_none(float(corr[i, j])) for j in range(len(param_names))]
                       for i in range(len(param_names))] if corr.ndim == 2 else None,
            "threshold": corr_threshold,
            "flagged_pairs": flags,
            "possible_non_identifiability": bool(len(flags) > 0),
        },
    }
    with open(boot_json_path, "w", encoding="utf-8") as fh:
        json.dump(boot_summary, fh, indent=2, cls=_NpEncoder, allow_nan=False)
    print(f"Saved: {boot_json_path}")
    if flags:
        print(f"  NOTE: |correlation| >= {corr_threshold} for "
              + ", ".join(f"{f['param_a']}~{f['param_b']}={f['correlation']:.3f}"
                          for f in flags)
              + "  -> possible non-identifiability.")

    # ---- plots ----
    if args.no_plots:
        return
    if plt is None:
        print("WARNING: matplotlib unavailable; skipping bootstrap plots.")
        return
    _plot_bootstrap(outdir, param_names, param_matrix, fitted_vals, param_boot_stats,
                    corr, temps, b_band, c_band, rg_band, obs_mean_contacts,
                    obs_mean_rg, boot_records, has_val, confidence)


def _plot_bootstrap(outdir, param_names, param_matrix, fitted_vals, param_boot_stats,
                    corr, temps, b_band, c_band, rg_band, obs_mean_contacts,
                    obs_mean_rg, boot_records, has_val, confidence):
    # 1. Parameter marginal histograms with fitted value + CI.
    k = len(param_names)
    fig, axes = plt.subplots(1, k, figsize=(4 * k, 3.5), squeeze=False)
    for j, pn in enumerate(param_names):
        ax = axes[0, j]
        ax.hist(param_matrix[:, j], bins=20, color="#4477aa", alpha=0.8)
        st = param_boot_stats[pn]
        ax.axvline(fitted_vals[pn], color="k", lw=1.8, label="fit")
        if st["ci_low"] is not None:
            ax.axvline(st["ci_low"], color="r", ls="--", lw=1.2, label=f"{int(confidence*100)}% CI")
            ax.axvline(st["ci_high"], color="r", ls="--", lw=1.2)
        ax.set_title(pn)
        ax.legend(fontsize=7)
    fig.suptitle("Bootstrap parameter marginals (fit + CI)")
    fig.tight_layout()
    p = outdir / "bootstrap_param_hist.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {p}")

    # 2. Parameter correlation heatmap.
    if corr.ndim == 2 and corr.shape[0] >= 2 and np.all(np.isfinite(corr)):
        fig, ax = plt.subplots(figsize=(1.2 * k + 2, 1.2 * k + 2))
        im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_xticks(range(k)); ax.set_yticks(range(k))
        ax.set_xticklabels(param_names, rotation=45, ha="right")
        ax.set_yticklabels(param_names)
        for i in range(k):
            for j in range(k):
                ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=8)
        plt.colorbar(im, ax=ax, label="correlation")
        ax.set_title("Bootstrap parameter correlation")
        fig.tight_layout()
        p = outdir / "bootstrap_param_correlation.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {p}")

    # 3. b(T) median + confidence band.
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.fill_between(temps, b_band["lo"], b_band["hi"], color="#88aadd", alpha=0.4,
                    label=f"{int(confidence*100)}% band")
    ax.plot(temps, b_band["median"], "b-", lw=1.8, label="median")
    ax.axhline(0.0, color="gray", lw=0.8, ls="--")
    ax.set_xlabel("T"); ax.set_ylabel("b(T)")
    ax.set_title("Reduced bias b(T): bootstrap median and band")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = outdir / "bootstrap_bT_band.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {p}")

    # 4. Predicted mean contact (and Rg) bands vs target means.
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.fill_between(temps, c_band["lo"], c_band["hi"], color="#99cc99", alpha=0.4,
                    label="contacts band")
    ax.plot(temps, c_band["median"], "g-", lw=1.6, label="contacts median")
    ax.plot(temps, obs_mean_contacts, "k.", ms=5, label="target contacts")
    ax.set_xlabel("T"); ax.set_ylabel("mean contacts")
    ax.set_title("Predicted mean-contact band vs target")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = outdir / "bootstrap_mean_contacts_band.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {p}")

    if rg_band is not None:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.fill_between(temps, rg_band["lo"], rg_band["hi"], color="#ddaa88", alpha=0.4,
                        label="Rg band")
        ax.plot(temps, rg_band["median"], color="#cc6600", lw=1.6, label="Rg median")
        if obs_mean_rg is not None:
            ax.plot(temps, obs_mean_rg, "k.", ms=5, label="target Rg")
        ax.set_xlabel("T"); ax.set_ylabel("mean Rg (scaled units)")
        ax.set_title("Predicted mean-Rg band vs target")
        ax.legend(fontsize=8)
        fig.tight_layout()
        p = outdir / "bootstrap_mean_rg_band.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {p}")

    # 5. Validation-loss distribution.
    vkey = "combined_validation_loss" if any(
        "combined_validation_loss" in r for r in boot_records) else "validation_loss"
    vvals = np.array([r[vkey] for r in boot_records if vkey in r], dtype=float)
    vvals = vvals[np.isfinite(vvals)]
    if vvals.size > 0:
        fig, ax = plt.subplots(figsize=(6.5, 4))
        ax.hist(vvals, bins=20, color="#aa6688", alpha=0.85)
        lo, hi = _percentile_ci(vvals, confidence)
        ax.axvline(float(np.median(vvals)), color="k", lw=1.8, label="median")
        ax.axvline(lo, color="r", ls="--", lw=1.2, label=f"{int(confidence*100)}% CI")
        ax.axvline(hi, color="r", ls="--", lw=1.2)
        ax.set_xlabel(vkey)
        ax.set_title("Bootstrap validation-loss distribution")
        ax.legend(fontsize=8)
        fig.tight_layout()
        p = outdir / "bootstrap_val_loss_hist.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {p}")


# ---------------------------------------------------------------------------
# Local objective-curvature + restart-stability diagnostics
# ---------------------------------------------------------------------------

def run_uncertainty_diagnostics(args: argparse.Namespace, ctx: Dict[str, Any]) -> None:
    """Numerical-Hessian curvature + optimizer restart-stability diagnostics.

    These are NOT statistical standard errors (the JS/KL objective is not a log
    likelihood); they characterize the local objective geometry and the optimizer.
    """
    outdir = ctx["outdir"]
    param_names = ctx["param_names"]
    bounds = ctx["bounds"]
    x0s = ctx["x0s"]
    params_fit = np.asarray(ctx["params_fit"], dtype=float)

    obj_fn, obj_args = build_objective(
        ctx["fit_rg"], ctx["train_temps"], ctx["m_centers"], ctx["p_obs_ct_train"],
        ctx["p0_mass"], ctx["b_fn"], ctx["loss_fn"],
        crg_prob=ctx["crg_prob"], c_edges_joint=ctx["c_edges_joint"],
        p_obs_rg_train=ctx["p_obs_rg_train"], rg_weight=float(ctx["rg_weight"]),
    )

    print("\nUncertainty diagnostics (local curvature + restart stability):")

    # Restart stability on the primary training objective (deterministic x0s).
    _, _, restart_records = fit_restarts(obj_fn, obj_args, x0s, bounds)
    rs = restart_stability(restart_records)

    restart_csv = outdir / "restart_diagnostics.csv"
    with open(restart_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["restart_index", "success", "objective", "n_iter", "message"]
                   + list(param_names))
        for r in restart_records:
            w.writerow([
                r["restart_index"], int(r["success"]),
                ("%.10g" % r["objective"]) if r["objective"] is not None else "",
                r["n_iter"] if r["n_iter"] is not None else "",
                r["message"],
            ] + ["%.10g" % v for v in r["params"]])
    print(f"Saved: {restart_csv}")
    print(f"  restarts: {rs['n_success']}/{rs['n_restarts']} succeeded; "
          f"distinct local minima: {rs['distinct_minima']} "
          f"({rs['n_distinct_objectives']} objective level(s))")

    # Numerical Hessian (objective curvature) at the optimum.
    def f(p: np.ndarray) -> float:
        return float(obj_fn(p, *obj_args))
    H = numerical_hessian(f, params_fit)
    hdiag = hessian_diagnostics(H)
    print(f"  Hessian eigenvalues: "
          + ", ".join("%.4g" % e for e in hdiag["eigenvalues"])
          + (f"; condition number: {hdiag['condition_number']:.4g}"
             if hdiag["condition_number"] is not None else "; condition number: n/a"))

    diag_json = outdir / "uncertainty_diagnostics.json"
    payload = {
        "model": ctx["model_name"],
        "loss": ctx["loss_name"],
        "param_names": list(param_names),
        "params": [float(v) for v in params_fit],
        "hessian": {
            "matrix": [[_finite_or_none(float(H[i, j])) for j in range(H.shape[1])]
                       for i in range(H.shape[0])],
            "eigenvalues": [_finite_or_none(e) for e in hdiag["eigenvalues"]],
            "condition_number": _finite_or_none(hdiag["condition_number"]),
            "positive_definite": hdiag["positive_definite"],
            "curvature_length_scales": (
                [_finite_or_none(s) for s in hdiag["curvature_length_scales"]]
                if hdiag["curvature_length_scales"] is not None else None
            ),
            "note": hdiag["note"],
        },
        "restart_stability": rs,
    }
    with open(diag_json, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, cls=_NpEncoder, allow_nan=False)
    print(f"Saved: {diag_json}")


# ---------------------------------------------------------------------------
# Rg-weight sensitivity and Pareto analysis
# ---------------------------------------------------------------------------

def parse_weight_grid(grid_str: Optional[str], grid_file: Optional[str]) -> List[float]:
    """Parse weights from a comma string and/or a file (comma/space/newline)."""
    raw: List[str] = []
    if grid_file is not None:
        with open(grid_file, "r", encoding="utf-8") as fh:
            text = fh.read()
        raw += text.replace(",", " ").split()
    if grid_str is not None:
        raw += [t for t in grid_str.split(",")]
    weights: List[float] = []
    for tok in raw:
        tok = tok.strip()
        if not tok:
            continue
        w = float(tok)
        if not np.isfinite(w) or w < 0:
            raise ValueError(f"rg-weight grid value {tok!r} must be finite and >= 0")
        weights.append(w)
    # unique, ascending (dominance and plots are order-independent)
    uniq = sorted(set(weights))
    if not uniq:
        raise ValueError("rg-weight grid parsed to an empty list")
    return uniq


def pareto_efficient_mask(costs: np.ndarray) -> np.ndarray:
    """Boolean mask of Pareto-efficient rows (both columns minimized, lower better).

    A point is dominated if another point is <= in both objectives and strictly
    less in at least one.  Identical points are both kept (neither dominates).
    """
    costs = np.asarray(costs, dtype=float)
    n = costs.shape[0]
    eff = np.ones(n, dtype=bool)
    for i in range(n):
        if not np.all(np.isfinite(costs[i])):
            eff[i] = False
            continue
        for j in range(n):
            if j == i or not np.all(np.isfinite(costs[j])):
                continue
            if np.all(costs[j] <= costs[i]) and np.any(costs[j] < costs[i]):
                eff[i] = False
                break
    return eff


def knee_index(frontier: np.ndarray) -> Optional[int]:
    """Geometric knee (max perpendicular distance to the endpoint chord).

    HEURISTIC ONLY. frontier is (k, 2) sorted ascending by the first objective.
    Returns the row index of the knee, or None if fewer than 3 points.
    """
    frontier = np.asarray(frontier, dtype=float)
    k = frontier.shape[0]
    if k < 3:
        return None
    p0 = frontier[0]
    pe = frontier[-1]
    line = pe - p0
    L = float(np.hypot(line[0], line[1]))
    if L <= 0:
        return None
    dists = np.empty(k, dtype=float)
    for i in range(k):
        v = frontier[i] - p0
        dists[i] = abs(line[0] * v[1] - line[1] * v[0]) / L
    return int(np.argmax(dists))


def run_rg_weight_sensitivity(args: argparse.Namespace, ctx: Dict[str, Any]) -> None:
    """Sweep the Rg-loss weight, refit, and analyze the contact/Rg trade-off.

    Supplementary only: does not touch the standard fit outputs.  Requires Rg
    data (observed Rg histograms + joint baseline) so every weight can be scored
    on Rg even when the fit itself is contact-only (weight 0).
    """
    if not ctx["can_fit_rg"]:
        raise ValueError(
            "--rg-weight-grid requires Rg data (observed Rg histograms and a joint "
            "baseline with c_edges/rg_edges/crg_prob) so each weight can be scored "
            "on Rg. None available."
        )

    temps = ctx["temps"]
    n = temps.size
    m_centers = ctx["m_centers"]
    p_obs_mass = ctx["p_obs_mass"]
    p0_mass = ctx["p0_mass"]
    b_fn = ctx["b_fn"]
    loss_fn = ctx["loss_fn"]
    spec = ctx["spec"]
    param_names = ctx["param_names"]
    bounds = ctx["bounds"]
    x0s = ctx["x0s"]
    crg_prob = ctx["crg_prob"]
    c_edges_joint = ctx["c_edges_joint"]
    rg_edges_model_lattice = ctx["rg_edges_model_lattice"]
    p_obs_rg_model_grid = ctx["p_obs_rg_model_grid"]
    outdir = ctx["outdir"]
    train_idx = ctx["train_idx"]
    val_idx = ctx["val_idx"]
    has_val = ctx["has_val"]
    rg_scale = float(ctx["rg_scale"])
    derived_Tc_fn = spec["derived_Tc"]
    production_weight = float(ctx["rg_weight"])

    weights = parse_weight_grid(args.rg_weight_grid, args.rg_weight_grid_file)

    # Pareto/selection space: validation when available, else all-temperature.
    sel_idx = val_idx if has_val else np.arange(n)
    sel_name = "validation" if has_val else "all"

    obs_mean_contacts = (m_centers[None, :] * p_obs_mass).sum(axis=1)
    rg_centers_lat = 0.5 * (rg_edges_model_lattice[:-1] + rg_edges_model_lattice[1:])
    rg_centers_scaled = rg_scale * rg_centers_lat
    obs_mean_rg = (rg_centers_scaled[None, :] * p_obs_rg_model_grid).sum(axis=1)

    print(f"\n=== Rg-weight sensitivity ({len(weights)} weights) ===")
    print(f"  grid: {weights}")
    print(f"  production reference weight: {production_weight}")
    print(f"  Pareto space: {sel_name} contact-loss vs {sel_name} Rg-loss "
          "(weight 0 is a true contact-only fit, still scored on Rg)")

    records: List[Dict[str, Any]] = []
    per_temp_rows: List[Dict[str, Any]] = []
    bT_curves: List[np.ndarray] = []
    predC_curves: List[np.ndarray] = []
    predRg_curves: List[np.ndarray] = []
    param_path: List[np.ndarray] = []

    train_temps = temps[train_idx]
    p_obs_ct_train = p_obs_mass[train_idx]
    p_obs_rg_train = p_obs_rg_model_grid[train_idx]

    for w in weights:
        use_rg = w > 0.0
        obj_fn, obj_args = build_objective(
            use_rg, train_temps, m_centers, p_obs_ct_train, p0_mass, b_fn, loss_fn,
            crg_prob=crg_prob, c_edges_joint=c_edges_joint,
            p_obs_rg_train=p_obs_rg_train, rg_weight=w,
        )
        best, obj_val = fit_one_split(obj_fn, obj_args, x0s, bounds)
        params = best.x
        param_path.append(np.asarray(params, dtype=float))

        ct_pt = per_temp_contact_losses(
            temps, m_centers, p_obs_mass, p0_mass, params, b_fn, loss_fn
        )
        _, rg_mod = predict_rg_from_joint(
            crg_prob, c_edges_joint, rg_edges_model_lattice, temps, params, b_fn
        )
        rg_pt = per_temp_rg_losses(rg_mod, p_obs_rg_model_grid, loss_fn)

        def sm(arr: np.ndarray, idx: np.ndarray):
            if idx.size == 0:
                return float("nan"), float("nan")
            s = float(arr[idx].sum())
            return s, s / idx.size

        ct_tr_s, ct_tr_m = sm(ct_pt, train_idx)
        ct_va_s, ct_va_m = sm(ct_pt, val_idx)
        ct_al_s = float(ct_pt.sum()); ct_al_m = ct_al_s / n
        rg_tr_s, rg_tr_m = sm(rg_pt, train_idx)
        rg_va_s, rg_va_m = sm(rg_pt, val_idx)
        rg_al_s = float(rg_pt.sum()); rg_al_m = rg_al_s / n

        rec: Dict[str, Any] = {
            "rg_weight": float(w),
            "is_production": bool(abs(w - production_weight) <= 1e-12),
            "contact_only_fit": bool(not use_rg),
            "fit_objective": float(obj_val),
            "contact_train_sum": ct_tr_s, "contact_train_mean": ct_tr_m,
            "contact_val_sum": ct_va_s, "contact_val_mean": ct_va_m,
            "contact_all_sum": ct_al_s, "contact_all_mean": ct_al_m,
            "rg_train_sum": rg_tr_s, "rg_train_mean": rg_tr_m,
            "rg_val_sum": rg_va_s, "rg_val_mean": rg_va_m,
            "rg_all_sum": rg_al_s, "rg_all_mean": rg_al_m,
            # combined = contact + w * Rg at this weight (sums and means)
            "combined_train_sum": ct_tr_s + w * rg_tr_s,
            "combined_train_mean": ct_tr_m + w * rg_tr_m,
            "combined_val_sum": (ct_va_s + w * rg_va_s) if has_val else float("nan"),
            "combined_val_mean": (ct_va_m + w * rg_va_m) if has_val else float("nan"),
            "combined_all_sum": ct_al_s + w * rg_al_s,
            "combined_all_mean": ct_al_m + w * rg_al_m,
        }
        for pn, pv in zip(param_names, params):
            rec[pn] = float(pv)
        if derived_Tc_fn is not None:
            tc = derived_Tc_fn(params)
            if tc is not None and np.isfinite(tc):
                rec["Tc"] = float(tc)

        bT = np.array([b_fn(params, float(T)) for T in temps])
        predC = _predicted_means(temps, m_centers, p0_mass, params, b_fn)
        predRg = rg_scale * (rg_centers_lat[None, :] * rg_mod).sum(axis=1)
        bT_curves.append(bT)
        predC_curves.append(predC)
        predRg_curves.append(predRg)

        val_set = set(val_idx.tolist())
        for i in range(n):
            per_temp_rows.append({
                "rg_weight": float(w), "temp_index": i,
                "temperature": float(temps[i]),
                "in_validation": (i in val_set),
                "contact_loss": float(ct_pt[i]), "rg_loss": float(rg_pt[i]),
                "b_T": float(bT[i]),
                "pred_mean_contacts": float(predC[i]),
                "pred_mean_rg": float(predRg[i]),
            })

        records.append(rec)
        print(f"  w={w:<6g} contact_{sel_name}_mean="
              f"{rec['contact_'+('val' if has_val else 'all')+'_mean']:.5g} "
              f"rg_{sel_name}_mean="
              f"{rec['rg_'+('val' if has_val else 'all')+'_mean']:.5g}")

    # ---- Pareto frontier in (contact, Rg) selection space ----
    ckey = "contact_val_mean" if has_val else "contact_all_mean"
    rkey = "rg_val_mean" if has_val else "rg_all_mean"
    costs = np.array([[r[ckey], r[rkey]] for r in records], dtype=float)
    eff_mask = pareto_efficient_mask(costs)
    for r, e in zip(records, eff_mask):
        r["pareto_efficient"] = bool(e)

    eff_idx = np.where(eff_mask)[0]
    # frontier sorted ascending by contact loss
    eff_sorted = eff_idx[np.argsort(costs[eff_idx, 0])]
    frontier_pts = costs[eff_sorted]
    knee_pos = knee_index(frontier_pts)
    knee_weight = float(weights[eff_sorted[knee_pos]]) if knee_pos is not None else None
    for r in records:
        r["is_knee"] = False
    if knee_weight is not None:
        records[eff_sorted[knee_pos]]["is_knee"] = True

    frontier_weights = [float(weights[i]) for i in eff_sorted]
    dominated_weights = [float(weights[i]) for i in range(len(weights)) if not eff_mask[i]]

    print(f"  Pareto-efficient weights: {frontier_weights}")
    print(f"  Pareto-dominated weights: {dominated_weights}")
    if knee_weight is not None:
        print(f"  knee weight (heuristic): {knee_weight}")

    # ---- normalization diagnostics (optional; never rescales the objective) ----
    norm_diag = None
    if args.rg_weight_normalization_diagnostics:
        norm_diag = []
        for r in records:
            c = r[ckey]; rg = r[rkey]; w = r["rg_weight"]
            ratio = float(rg / c) if c > 0 else None
            wfrac = float(w * rg / (c + w * rg)) if (c + w * rg) > 0 else None
            norm_diag.append({
                "rg_weight": w,
                "contact_loss": c, "rg_loss": rg,
                "rg_over_contact": ratio,
                "weighted_rg": float(w * rg),
                "weighted_rg_fraction_of_objective": wfrac,
            })
        print("  normalization diagnostics (relative loss scales):")
        for d in norm_diag:
            print(f"    w={d['rg_weight']:<6g} rg/contact={d['rg_over_contact']}"
                  f"  weighted_rg_frac={d['weighted_rg_fraction_of_objective']}")

    # ---- write CSV outputs ----
    def _cell(v: Any) -> str:
        if v is None:
            return ""
        if isinstance(v, bool):
            return str(int(v))
        if isinstance(v, float):
            return "%.10g" % v if np.isfinite(v) else ""
        return str(v)

    sens_csv = outdir / "rg_weight_sensitivity.csv"
    base_cols = ["rg_weight", "is_production", "contact_only_fit", "pareto_efficient",
                 "is_knee", "fit_objective"]
    loss_cols = ["contact_train_sum", "contact_train_mean", "contact_val_sum",
                 "contact_val_mean", "contact_all_sum", "contact_all_mean",
                 "rg_train_sum", "rg_train_mean", "rg_val_sum", "rg_val_mean",
                 "rg_all_sum", "rg_all_mean", "combined_train_sum",
                 "combined_train_mean", "combined_val_sum", "combined_val_mean",
                 "combined_all_sum", "combined_all_mean"]
    pcols = list(param_names) + (["Tc"] if any("Tc" in r for r in records) else [])
    header = base_cols + pcols + loss_cols
    with open(sens_csv, "w", newline="", encoding="utf-8") as fh:
        w_ = csv.writer(fh)
        w_.writerow(header)
        for r in records:
            w_.writerow([_cell(r.get(c)) for c in header])
    print(f"Saved: {sens_csv}")

    pt_csv = outdir / "rg_weight_per_temperature.csv"
    with open(pt_csv, "w", newline="", encoding="utf-8") as fh:
        w_ = csv.writer(fh)
        w_.writerow(["rg_weight", "temp_index", "temperature", "in_validation",
                     "contact_loss", "rg_loss", "b_T", "pred_mean_contacts",
                     "pred_mean_rg"])
        for row in per_temp_rows:
            w_.writerow([_cell(row["rg_weight"]), row["temp_index"],
                         _cell(row["temperature"]), int(row["in_validation"]),
                         _cell(row["contact_loss"]), _cell(row["rg_loss"]),
                         _cell(row["b_T"]), _cell(row["pred_mean_contacts"]),
                         _cell(row["pred_mean_rg"])])
    print(f"Saved: {pt_csv}")

    path_csv = outdir / "rg_weight_parameter_path.csv"
    with open(path_csv, "w", newline="", encoding="utf-8") as fh:
        w_ = csv.writer(fh)
        head = ["rg_weight"] + list(param_names) + (["Tc"] if any("Tc" in r for r in records) else [])
        w_.writerow(head)
        for r in records:
            w_.writerow([_cell(r.get(c)) for c in head])
    print(f"Saved: {path_csv}")

    # ---- summary JSON (strict) ----
    summary = {
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "model": ctx["model_name"],
        "loss": ctx["loss_name"],
        "n_temps": int(n),
        "weight_grid": [float(w) for w in weights],
        "production_weight": production_weight,
        "production_weight_in_grid": bool(
            any(abs(w - production_weight) <= 1e-12 for w in weights)
        ),
        "pareto_space": sel_name,
        "pareto_note": (
            "Weights are NOT ranked by the raw combined objective (the weight "
            "defines that objective). Trade-off is assessed in "
            f"{sel_name} contact-loss vs {sel_name} Rg-loss space."
        ),
        "pareto_efficient_weights": frontier_weights,
        "pareto_dominated_weights": dominated_weights,
        "knee_weight_heuristic": knee_weight,
        "knee_note": "Geometric knee (max distance to endpoint chord); heuristic only.",
        "per_weight": [
            {k: (_finite_or_none(v) if isinstance(v, float) else v)
             for k, v in r.items()}
            for r in records
        ],
        "normalization_diagnostics": (
            [{k: (_finite_or_none(v) if isinstance(v, float) else v)
              for k, v in d.items()} for d in norm_diag]
            if norm_diag is not None else None
        ),
    }
    sens_json = outdir / "rg_weight_summary.json"
    with open(sens_json, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, cls=_NpEncoder, allow_nan=False)
    print(f"Saved: {sens_json}")

    # ---- plots ----
    if args.no_plots:
        return
    if plt is None:
        print("WARNING: matplotlib unavailable; skipping Rg-weight plots.")
        return
    _plot_rg_weight(outdir, weights, records, costs, eff_mask, eff_sorted, knee_pos,
                    ckey, rkey, sel_name, temps, np.array(bT_curves),
                    np.array(predC_curves), np.array(predRg_curves),
                    obs_mean_contacts, obs_mean_rg, param_names,
                    np.array(param_path), production_weight)


def _plot_rg_weight(outdir, weights, records, costs, eff_mask, eff_sorted, knee_pos,
                    ckey, rkey, sel_name, temps, bT_curves, predC_curves, predRg_curves,
                    obs_mean_contacts, obs_mean_rg, param_names, param_path,
                    production_weight):
    weights = np.asarray(weights, dtype=float)

    # 1. contact vs Rg with weight labels + Pareto frontier.
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.scatter(costs[:, 0], costs[:, 1], c="#4477aa", zorder=3)
    for i, w in enumerate(weights):
        ax.annotate(f"{w:g}", (costs[i, 0], costs[i, 1]), fontsize=7,
                    xytext=(3, 3), textcoords="offset points")
    ax.plot(costs[eff_sorted, 0], costs[eff_sorted, 1], "r-o", lw=1.5, ms=4,
            label="Pareto frontier", zorder=2)
    if knee_pos is not None:
        kp = eff_sorted[knee_pos]
        ax.scatter([costs[kp, 0]], [costs[kp, 1]], s=160, facecolors="none",
                   edgecolors="green", lw=2, label="knee (heuristic)", zorder=4)
    ax.set_xlabel(f"{sel_name} contact loss (mean/temp)")
    ax.set_ylabel(f"{sel_name} Rg loss (mean/temp)")
    ax.set_title("Contact vs Rg loss trade-off across Rg weights")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = outdir / "rg_weight_pareto.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Saved: {p}")

    # 2. each loss vs weight (symlog x so weight 0 is visible).
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(weights, [r[ckey] for r in records], "o-", label="contact")
    ax.plot(weights, [r[rkey] for r in records], "s-", label="Rg")
    comb_key = "combined_val_mean" if sel_name == "validation" else "combined_all_mean"
    ax.plot(weights, [r[comb_key] for r in records], "^-", label="combined")
    try:
        ax.set_xscale("symlog", linthresh=min([w for w in weights if w > 0] or [1.0]))
    except Exception:
        pass
    ax.set_xlabel("Rg weight (symlog)")
    ax.set_ylabel(f"{sel_name} loss (mean/temp)")
    ax.set_title("Losses vs Rg weight")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = outdir / "rg_weight_losses_vs_weight.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Saved: {p}")

    # 3. parameter values vs weight.
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for j, pn in enumerate(param_names):
        ax.plot(weights, param_path[:, j], "o-", label=pn)
    try:
        ax.set_xscale("symlog", linthresh=min([w for w in weights if w > 0] or [1.0]))
    except Exception:
        pass
    ax.set_xlabel("Rg weight (symlog)")
    ax.set_ylabel("parameter value")
    ax.set_title("Parameter path vs Rg weight")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = outdir / "rg_weight_param_path.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Saved: {p}")

    # 4. b(T) curves colored by weight.
    fig, ax = plt.subplots(figsize=(7, 4.5))
    cmap = plt.get_cmap("viridis")
    wmin, wmax = float(weights.min()), float(weights.max())
    for i, w in enumerate(weights):
        frac = (w - wmin) / (wmax - wmin) if wmax > wmin else 0.5
        ax.plot(temps, bT_curves[i], color=cmap(frac), lw=1.4, label=f"{w:g}")
    ax.axhline(0.0, color="gray", lw=0.8, ls="--")
    ax.set_xlabel("T"); ax.set_ylabel("b(T)")
    ax.set_title("Reduced bias b(T) colored by Rg weight")
    ax.legend(fontsize=7, title="rg_weight", ncol=2)
    fig.tight_layout()
    p = outdir / "rg_weight_bT.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Saved: {p}")

    # 5. target vs predicted mean contacts and Rg for selected frontier weights.
    sel = []
    for idx in (eff_sorted[0], eff_sorted[-1]):
        sel.append(int(idx))
    if knee_pos is not None:
        sel.append(int(eff_sorted[knee_pos]))
    prod_i = int(np.argmin(np.abs(weights - production_weight)))
    sel.append(prod_i)
    sel = sorted(set(sel))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(temps, obs_mean_contacts, "k.", ms=5, label="target")
    axes[1].plot(temps, obs_mean_rg, "k.", ms=5, label="target")
    for i in sel:
        lab = f"w={weights[i]:g}"
        axes[0].plot(temps, predC_curves[i], "-", lw=1.3, label=lab)
        axes[1].plot(temps, predRg_curves[i], "-", lw=1.3, label=lab)
    axes[0].set_xlabel("T"); axes[0].set_ylabel("mean contacts")
    axes[0].set_title("Target vs predicted mean contacts"); axes[0].legend(fontsize=7)
    axes[1].set_xlabel("T"); axes[1].set_ylabel("mean Rg (scaled)")
    axes[1].set_title("Target vs predicted mean Rg"); axes[1].legend(fontsize=7)
    fig.tight_layout()
    p = outdir / "rg_weight_pred_vs_target.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Saved: {p}")


# ---------------------------------------------------------------------------
# Scalar Rg(T) mode: data loading
# ---------------------------------------------------------------------------
# This mode fits the contact-bias model to a SCALAR Rg per temperature (e.g. from
# experiment or all-atom MD) instead of full contact/Rg histograms.  It never
# fabricates histograms from the scalar data: the prediction is a summary
# (mean or rms) of the reweighted joint baseline P0(m, Rg), and the objective is
# a plain scalar regression loss on Rg(T).

RG_TARGET_UNITS_CHOICES = ("observed", "lattice")
RG_SUMMARY_CHOICES = ("rms", "mean")
RG_MEAN_LOSS_CHOICES = ("mse", "mae", "huber", "range_weighted")


def load_rg_mean_file(path: str) -> Dict[str, np.ndarray]:
    """Load a scalar Rg(T) data file with four whitespace-separated columns.

    Columns (no header required; ``#`` comment lines are ignored):
        1. temperature (K)
        2. central Rg value
        3. lower Rg value
        4. upper Rg value

    The lower/upper columns are treated as DESCRIPTIVE BOUNDS, not standard
    errors and not confidence intervals.  They are only given a statistical
    role if the user explicitly selects ``--rg-mean-loss range_weighted``.

    Returns
    -------
    dict with keys ``temps``, ``rg_target``, ``rg_lower``, ``rg_upper``, each a
    1-D float array of equal length, in the file's own units.

    Raises
    ------
    ValueError
        If the file does not have exactly four numeric columns, has fewer than
        three temperatures, contains non-finite values, has non-positive or
        non-increasing (or duplicate) temperatures, has non-positive Rg values,
        or violates ``lower <= central <= upper`` on any row.
    """
    raw = np.loadtxt(path, dtype=float, ndmin=2)
    if raw.ndim != 2 or raw.shape[1] != 4:
        raise ValueError(
            f"{path!r}: expected exactly 4 numeric columns "
            f"(temperature, Rg, lower, upper), got shape {raw.shape}."
        )
    if raw.shape[0] < 3:
        raise ValueError(
            f"{path!r}: need at least 3 temperatures for a temperature-dependent "
            f"fit, got {raw.shape[0]}."
        )
    if not np.all(np.isfinite(raw)):
        bad = np.argwhere(~np.isfinite(raw))
        rows = sorted({int(r) for r, _ in bad})
        raise ValueError(
            f"{path!r}: non-finite value(s) on data row(s) {rows} (0-indexed)."
        )

    temps = raw[:, 0]
    rg_target = raw[:, 1]
    rg_lower = raw[:, 2]
    rg_upper = raw[:, 3]

    if np.any(temps <= 0.0):
        bad = np.argwhere(temps <= 0.0).ravel().tolist()
        raise ValueError(
            f"{path!r}: temperatures must be strictly positive; "
            f"violated on data row(s) {bad} (0-indexed)."
        )
    diffs = np.diff(temps)
    if np.any(diffs == 0.0):
        bad = (np.argwhere(diffs == 0.0).ravel() + 1).tolist()
        raise ValueError(
            f"{path!r}: duplicate temperature(s) at data row(s) {bad} (0-indexed). "
            f"Temperatures must be strictly increasing."
        )
    if np.any(diffs < 0.0):
        bad = (np.argwhere(diffs < 0.0).ravel() + 1).tolist()
        raise ValueError(
            f"{path!r}: temperatures must be strictly increasing; "
            f"decrease first occurs at data row(s) {bad} (0-indexed)."
        )
    for label, arr in (("central", rg_target), ("lower", rg_lower), ("upper", rg_upper)):
        if np.any(arr <= 0.0):
            bad = np.argwhere(arr <= 0.0).ravel().tolist()
            raise ValueError(
                f"{path!r}: {label} Rg values must be positive; "
                f"violated on data row(s) {bad} (0-indexed)."
            )
    bad_lo = np.argwhere(rg_lower > rg_target).ravel()
    if bad_lo.size:
        r = int(bad_lo[0])
        raise ValueError(
            f"{path!r}: requires lower <= central; data row {r} (0-indexed) has "
            f"lower={rg_lower[r]:.6g} > central={rg_target[r]:.6g}."
        )
    bad_hi = np.argwhere(rg_target > rg_upper).ravel()
    if bad_hi.size:
        r = int(bad_hi[0])
        raise ValueError(
            f"{path!r}: requires central <= upper; data row {r} (0-indexed) has "
            f"central={rg_target[r]:.6g} > upper={rg_upper[r]:.6g}."
        )

    return {
        "temps": np.ascontiguousarray(temps, dtype=float),
        "rg_target": np.ascontiguousarray(rg_target, dtype=float),
        "rg_lower": np.ascontiguousarray(rg_lower, dtype=float),
        "rg_upper": np.ascontiguousarray(rg_upper, dtype=float),
    }


# ---------------------------------------------------------------------------
# Scalar Rg(T) mode: prediction
# ---------------------------------------------------------------------------

def rg_summary_from_mass(
    rg_mass: np.ndarray, rg_centers: np.ndarray, summary: str
) -> np.ndarray:
    """Collapse P(Rg|T) rows to one scalar per temperature.

    ``mean``: sum_j r_j P(r_j|T).
    ``rms`` : sqrt(sum_j r_j^2 P(r_j|T)).

    ``rms`` matches the usual definition of Rg as the square root of an
    ensemble-averaged squared distance; the two differ whenever P(Rg|T) has
    non-zero width, so they are never substituted for one another.
    """
    if summary not in RG_SUMMARY_CHOICES:
        raise ValueError(
            f"Unknown Rg summary {summary!r}. Choose from {RG_SUMMARY_CHOICES}."
        )
    rg_mass = np.asarray(rg_mass, dtype=float)
    rg_centers = np.asarray(rg_centers, dtype=float)
    if summary == "mean":
        return np.sum(rg_mass * rg_centers[None, :], axis=1)
    return np.sqrt(np.sum(rg_mass * rg_centers[None, :] ** 2, axis=1))


def predict_rg_summary_from_joint(
    crg_prob: np.ndarray,
    c_edges: np.ndarray,
    rg_edges: np.ndarray,
    temps: np.ndarray,
    params: np.ndarray,
    b_fn: Callable[[np.ndarray, float], float],
    *,
    rg_scale: float,
    summary: str,
    target_units: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predict a scalar Rg per temperature from the reweighted joint baseline.

    Uses predict_rg_from_joint() as the single source of truth for the contact
    reweighting P(Rg|T) ∝ sum_m P0(m, Rg) exp[-b(T) m], then collapses each
    normalized distribution to a scalar via ``summary``.

    The lattice Rg grid is never rescaled and the probability masses are never
    scaled: ``rg_scale`` is applied exactly once, to the scalar result.

    Parameters
    ----------
    rg_scale
        Observed units per lattice unit: Rg_observed = rg_scale * Rg_lattice.
    summary
        ``mean`` or ``rms``.
    target_units
        Retained for interface symmetry with the objective; it does not change
        the returned arrays (both unit systems are always returned) but it is
        validated here so callers fail early on a bad value.

    Returns
    -------
    (rg_pred_lattice, rg_pred_observed, rg_mass)
        Scalar prediction in lattice units, the same scaled into observed units,
        and the full normalized P(Rg|T) mass of shape (n_temps, n_rg_bins).
    """
    if target_units not in RG_TARGET_UNITS_CHOICES:
        raise ValueError(
            f"Unknown target_units {target_units!r}. "
            f"Choose from {RG_TARGET_UNITS_CHOICES}."
        )
    if not np.isfinite(rg_scale) or rg_scale <= 0.0:
        raise ValueError(f"rg_scale must be finite and positive, got {rg_scale!r}")

    rg_centers_lattice, rg_mass = predict_rg_from_joint(
        crg_prob=crg_prob,
        c_edges=c_edges,
        rg_edges=rg_edges,
        temps=np.asarray(temps, dtype=float),
        params=params,
        b_fn=b_fn,
    )

    if not np.all(np.isfinite(rg_mass)):
        raise ValueError(
            "predicted P(Rg|T) contains non-finite probability mass; "
            "check the joint baseline and the b(T) parameters."
        )
    if np.any(rg_mass < 0.0):
        raise ValueError("predicted P(Rg|T) contains negative probability mass.")
    row_sums = rg_mass.sum(axis=1)
    if np.any(row_sums <= 0.0):
        bad = np.argwhere(row_sums <= 0.0).ravel().tolist()
        raise ValueError(
            f"predicted P(Rg|T) has zero total mass at temperature index/indices "
            f"{bad}; the reweighted joint baseline could not be normalized."
        )
    # predict_rg_from_joint normalizes each row already; renormalize defensively
    # so the summary is exact even if a row is only normalized to within rounding.
    rg_mass = rg_mass / row_sums[:, None]

    rg_pred_lattice = rg_summary_from_mass(rg_mass, rg_centers_lattice, summary)
    rg_pred_observed = float(rg_scale) * rg_pred_lattice
    return rg_pred_lattice, rg_pred_observed, rg_mass


def rg_pred_in_target_units(
    rg_pred_lattice: np.ndarray,
    rg_pred_observed: np.ndarray,
    target_units: str,
) -> np.ndarray:
    """Select the prediction that lives in the same units as the target data.

    ``observed`` compares rg_scale * Rg_lattice against the file values;
    ``lattice`` compares Rg_lattice directly and rg_scale never enters the loss.
    """
    if target_units == "observed":
        return rg_pred_observed
    if target_units == "lattice":
        return rg_pred_lattice
    raise ValueError(
        f"Unknown target_units {target_units!r}. Choose from {RG_TARGET_UNITS_CHOICES}."
    )


# ---------------------------------------------------------------------------
# Scalar Rg(T) mode: per-temperature losses
# ---------------------------------------------------------------------------
# One implementation, used by the objective, the train/validation scoring, the
# split-sensitivity sweep, the bootstrap, and the diagnostics.

def rg_scalar_per_temp_losses(
    rg_pred: np.ndarray,
    rg_target: np.ndarray,
    loss_name: str,
    *,
    rg_lower: Optional[np.ndarray] = None,
    rg_upper: Optional[np.ndarray] = None,
    huber_delta: float = 0.05,
    range_floor: float = 0.01,
) -> np.ndarray:
    """Per-temperature scalar Rg loss contributions, all in target units.

    ``mse``            (pred - target)^2
    ``mae``            |pred - target|
    ``huber``          0.5 r^2 for |r| <= delta, else delta (|r| - 0.5 delta)
    ``range_weighted`` (r / sigma_i^±)^2 with sigma_i^- = target - lower for a
                       negative residual and sigma_i^+ = upper - target for a
                       positive one, each floored at ``range_floor``.

    range_weighted is NOT a chi-squared statistic unless the supplied lower/upper
    columns happen to be genuine statistical uncertainties; by default they are
    treated as descriptive bounds only.

    The mean over the returned array is the objective, so summing a subset and
    dividing by its size reproduces the objective on that subset exactly.
    """
    if loss_name not in RG_MEAN_LOSS_CHOICES:
        raise ValueError(
            f"Unknown Rg scalar loss {loss_name!r}. Choose from {RG_MEAN_LOSS_CHOICES}."
        )
    pred = np.asarray(rg_pred, dtype=float)
    target = np.asarray(rg_target, dtype=float)
    if pred.shape != target.shape:
        raise ValueError(
            f"prediction/target shape mismatch: {pred.shape} vs {target.shape}"
        )
    resid = pred - target

    if loss_name == "mse":
        return resid ** 2
    if loss_name == "mae":
        return np.abs(resid)
    if loss_name == "huber":
        delta = float(huber_delta)
        if not np.isfinite(delta) or delta <= 0.0:
            raise ValueError(f"--rg-huber-delta must be finite and positive, got {delta!r}")
        a = np.abs(resid)
        return np.where(a <= delta, 0.5 * a ** 2, delta * (a - 0.5 * delta))

    # range_weighted
    if rg_lower is None or rg_upper is None:
        raise ValueError(
            "range_weighted loss requires the lower and upper columns of the "
            "scalar Rg data file."
        )
    floor = float(range_floor)
    if not np.isfinite(floor) or floor <= 0.0:
        raise ValueError(f"--rg-range-floor must be finite and positive, got {floor!r}")
    lower = np.asarray(rg_lower, dtype=float)
    upper = np.asarray(rg_upper, dtype=float)
    sigma_minus = np.maximum(target - lower, floor)
    sigma_plus = np.maximum(upper - target, floor)
    sigma = np.where(resid < 0.0, sigma_minus, sigma_plus)
    return (resid / sigma) ** 2


def rg_scalar_objective_value(
    rg_pred: np.ndarray,
    rg_target: np.ndarray,
    loss_name: str,
    *,
    rg_lower: Optional[np.ndarray] = None,
    rg_upper: Optional[np.ndarray] = None,
    huber_delta: float = 0.05,
    range_floor: float = 0.01,
) -> float:
    """Mean per-temperature scalar Rg loss (comparable across split sizes)."""
    per = rg_scalar_per_temp_losses(
        rg_pred, rg_target, loss_name,
        rg_lower=rg_lower, rg_upper=rg_upper,
        huber_delta=huber_delta, range_floor=range_floor,
    )
    if per.size == 0:
        return float("nan")
    return float(np.mean(per))


def rg_scalar_raw_metrics(
    rg_pred: np.ndarray, rg_target: np.ndarray
) -> Dict[str, Optional[float]]:
    """Unweighted, physically interpretable error metrics in target units.

    Reported alongside every objective (including range_weighted) so results stay
    interpretable regardless of which loss drove the optimization.
    """
    pred = np.asarray(rg_pred, dtype=float)
    target = np.asarray(rg_target, dtype=float)
    if pred.size == 0:
        return {"rmse": None, "mae": None, "max_abs_error": None, "bias": None}
    resid = pred - target
    return {
        "rmse": float(np.sqrt(np.mean(resid ** 2))),
        "mae": float(np.mean(np.abs(resid))),
        "max_abs_error": float(np.max(np.abs(resid))),
        "bias": float(np.mean(resid)),
    }


# ---------------------------------------------------------------------------
# Scalar Rg(T) mode: parameter vector and objective
# ---------------------------------------------------------------------------

def split_rg_scalar_params(
    optimization_params: np.ndarray, n_model_params: int, fit_rg_scale: bool,
    fixed_rg_scale: float,
) -> Tuple[np.ndarray, float]:
    """Split the optimization vector into (model params, rg_scale).

    The model registry's parameter ordering is never disturbed: the optional
    scale parameter is appended after all thermodynamic parameters.
    """
    p = np.asarray(optimization_params, dtype=float)
    model_params = p[:n_model_params]
    if fit_rg_scale:
        if p.size != n_model_params + 1:
            raise ValueError(
                f"expected {n_model_params + 1} optimization parameters "
                f"(model + rg_scale), got {p.size}"
            )
        return model_params, float(p[n_model_params])
    if p.size != n_model_params:
        raise ValueError(
            f"expected {n_model_params} optimization parameters, got {p.size}"
        )
    return model_params, float(fixed_rg_scale)


def objective_rg_scalar(
    optimization_params: np.ndarray,
    temps: np.ndarray,
    rg_target: np.ndarray,
    crg_prob: np.ndarray,
    c_edges: np.ndarray,
    rg_edges: np.ndarray,
    b_fn: Callable[[np.ndarray, float], float],
    *,
    n_model_params: int,
    fixed_rg_scale: float,
    fit_rg_scale: bool,
    rg_summary: str,
    target_units: str,
    loss_name: str,
    rg_lower: Optional[np.ndarray] = None,
    rg_upper: Optional[np.ndarray] = None,
    huber_delta: float = 0.05,
    range_floor: float = 0.01,
) -> float:
    """Mean-per-temperature scalar Rg(T) loss for one parameter vector.

    Returns the MEAN (not the sum) so train, validation, and all-temperature
    objectives are directly comparable across splits of different sizes.
    """
    model_params, scale = split_rg_scalar_params(
        optimization_params, n_model_params, fit_rg_scale, fixed_rg_scale
    )
    try:
        pred_lat, pred_obs, _ = predict_rg_summary_from_joint(
            crg_prob, c_edges, rg_edges, temps, model_params, b_fn,
            rg_scale=scale, summary=rg_summary, target_units=target_units,
        )
    except ValueError:
        # Degenerate reweighting at this trial point: report a large finite value
        # so L-BFGS-B can step away instead of failing the whole restart.
        return float(1e12)
    pred = rg_pred_in_target_units(pred_lat, pred_obs, target_units)
    return rg_scalar_objective_value(
        pred, rg_target, loss_name,
        rg_lower=rg_lower, rg_upper=rg_upper,
        huber_delta=huber_delta, range_floor=range_floor,
    )


def build_rg_scalar_param_spec(
    model_name: str,
    *,
    fit_rg_scale: bool,
    rg_scale: float,
    rg_scale_min: float,
    rg_scale_max: float,
    n_restarts: int,
    seed: int,
) -> Dict[str, Any]:
    """Assemble names, bounds, and restart initial guesses for scalar-Rg mode.

    The model registry's parameter names, order, and bounds are used verbatim;
    when ``fit_rg_scale`` is active an ``rg_scale`` parameter is appended last so
    thermodynamic parameters stay separable in every output.

    The first restart uses the registry default x0 with the fixed --rg-scale value
    (clipped into its bounds) as the scale's initial guess; the remaining restarts
    sample every parameter — including the scale — uniformly within its bounds.
    """
    spec = MODEL_REGISTRY[model_name]
    model_names: List[str] = list(spec["param_names"])
    model_bounds: List[Tuple[float, float]] = list(spec["bounds"])
    n_model_params = len(model_names)

    param_names = list(model_names)
    bounds = list(model_bounds)
    x0_default = list(np.asarray(spec["x0"], dtype=float))

    if fit_rg_scale:
        if not np.isfinite(rg_scale_min) or not np.isfinite(rg_scale_max):
            raise ValueError("--rg-scale-min/--rg-scale-max must be finite")
        if rg_scale_min <= 0.0 or rg_scale_max <= 0.0:
            raise ValueError("--rg-scale-min/--rg-scale-max must be positive")
        if not (rg_scale_min < rg_scale_max):
            raise ValueError(
                f"--rg-scale-min ({rg_scale_min}) must be strictly less than "
                f"--rg-scale-max ({rg_scale_max})"
            )
        param_names.append("rg_scale")
        bounds.append((float(rg_scale_min), float(rg_scale_max)))
        x0_default.append(float(np.clip(rg_scale, rg_scale_min, rg_scale_max)))

    rng = np.random.default_rng(seed)
    x0s = [np.array(x0_default, dtype=float)]
    for _ in range(max(0, int(n_restarts) - 1)):
        x0s.append(np.array([rng.uniform(lo, hi) for lo, hi in bounds], dtype=float))

    return {
        "param_names": param_names,
        "model_param_names": model_names,
        "bounds": bounds,
        "model_bounds": model_bounds,
        "x0s": x0s,
        "n_model_params": n_model_params,
    }


# ---------------------------------------------------------------------------
# Scalar Rg(T) mode: transition descriptors
# ---------------------------------------------------------------------------

def bias_zero_crossings(
    b_fn: Callable[[np.ndarray, float], float],
    params: np.ndarray,
    t_lo: float,
    t_hi: float,
    *,
    n_grid: int = 2001,
) -> List[float]:
    """Temperatures in [t_lo, t_hi] where b(T) = 0, found by sign change + bisection.

    This is the BIAS zero-crossing: the temperature at which the contact bias
    changes sign.  For a finite chain it is not generally the temperature at
    which Rg(T) changes fastest — see rg_curve_transition_metrics().
    """
    grid = np.linspace(float(t_lo), float(t_hi), int(n_grid))
    vals = np.array([b_fn(params, float(T)) for T in grid], dtype=float)
    out: List[float] = []
    for i in range(grid.size - 1):
        a, c = vals[i], vals[i + 1]
        if not (np.isfinite(a) and np.isfinite(c)):
            continue
        if a == 0.0:
            out.append(float(grid[i]))
            continue
        if a * c < 0.0:
            lo, hi = float(grid[i]), float(grid[i + 1])
            f_lo = a
            for _ in range(80):
                mid = 0.5 * (lo + hi)
                f_mid = float(b_fn(params, mid))
                if f_mid == 0.0:
                    break
                if f_lo * f_mid < 0.0:
                    hi = mid
                else:
                    lo, f_lo = mid, f_mid
            out.append(0.5 * (lo + hi))
    if vals.size and vals[-1] == 0.0:
        out.append(float(grid[-1]))
    # Deduplicate crossings that bisected to the same temperature.
    uniq: List[float] = []
    for t in out:
        if not any(abs(t - u) < 1e-6 * max(1.0, abs(u)) for u in uniq):
            uniq.append(float(t))
    return uniq


def rg_curve_transition_metrics(
    crg_prob: np.ndarray,
    c_edges: np.ndarray,
    rg_edges: np.ndarray,
    model_params: np.ndarray,
    b_fn: Callable[[np.ndarray, float], float],
    t_lo: float,
    t_hi: float,
    *,
    rg_scale: float,
    summary: str,
    target_units: str,
    n_grid: int = 1001,
) -> Dict[str, Any]:
    """Finite-chain transition descriptors of the fitted scalar Rg(T) curve.

    Evaluates Rg(T) on a dense grid spanning the observed temperature interval
    and reports where the collapse is steepest:

    ``T_rg_max_slope``        argmax_T of -dRg/dT
    ``rg_max_negative_slope`` the value of -dRg/dT there (target units per K)
    ``T_rg_half``             the temperature nearest the midpoint between the
                              fitted Rg at the low and high ends of the interval

    T_rg_max_slope is the primary finite-chain transition descriptor and is a
    DIFFERENT quantity from the bias zero-crossing b(T) = 0.
    """
    n_grid = max(int(n_grid), 1001)
    grid = np.linspace(float(t_lo), float(t_hi), n_grid)
    pred_lat, pred_obs, _ = predict_rg_summary_from_joint(
        crg_prob, c_edges, rg_edges, grid, model_params, b_fn,
        rg_scale=rg_scale, summary=summary, target_units=target_units,
    )
    curve = rg_pred_in_target_units(pred_lat, pred_obs, target_units)
    slope = np.gradient(curve, grid)
    neg_slope = -slope
    k = int(np.argmax(neg_slope))

    rg_start, rg_end = float(curve[0]), float(curve[-1])
    mid = 0.5 * (rg_start + rg_end)
    j = int(np.argmin(np.abs(curve - mid)))

    return {
        "T_rg_max_slope": float(grid[k]),
        "rg_max_negative_slope": float(neg_slope[k]),
        "T_rg_half": float(grid[j]),
        "rg_half_value": float(curve[j]),
        "rg_at_T_low": rg_start,
        "rg_at_T_high": rg_end,
        "grid": grid,
        "curve": curve,
        "curve_lattice": pred_lat,
        "curve_observed": pred_obs,
    }


# ---------------------------------------------------------------------------
# Scalar Rg(T) mode: bias→Rg feasibility / reachability diagnostics
# ---------------------------------------------------------------------------

def joint_reweight_stats(
    crg_prob: np.ndarray,
    m_centers: np.ndarray,
    rg_centers: np.ndarray,
    b: float,
    summary: str,
) -> Dict[str, float]:
    """Moments of the bias-reweighted joint P_b(m, Rg) ∝ P0(m, Rg) exp[-b m].

    All quantities are in lattice units.  Returns the scalar Rg summary, the mean
    contacts, and the contact-Rg covariance/correlation under P_b.
    """
    x = -float(b) * m_centers
    x = x - np.max(x)
    w = np.exp(x)
    joint = crg_prob * w[:, None]
    Z = joint.sum()
    if not np.isfinite(Z) or Z <= 0.0:
        raise ValueError(f"joint baseline reweighting is degenerate at b={b!r}")
    joint = joint / Z

    p_m = joint.sum(axis=1)
    p_r = joint.sum(axis=0)
    mean_m = float(np.sum(p_m * m_centers))
    mean_r = float(np.sum(p_r * rg_centers))
    e_mr = float(np.sum(joint * np.outer(m_centers, rg_centers)))
    cov = e_mr - mean_m * mean_r
    var_m = float(np.sum(p_m * (m_centers - mean_m) ** 2))
    var_r = float(np.sum(p_r * (rg_centers - mean_r) ** 2))
    denom = np.sqrt(var_m * var_r)
    corr = float(cov / denom) if denom > 0 else float("nan")

    if summary == "mean":
        rg_scalar = mean_r
    else:
        rg_scalar = float(np.sqrt(np.sum(p_r * rg_centers ** 2)))

    return {
        "pred_rg_lattice": rg_scalar,
        "mean_contacts": mean_m,
        "mean_rg_lattice": mean_r,
        "cov_contact_rg": float(cov),
        "corr_contact_rg": corr,
        "var_contacts": var_m,
        "var_rg": var_r,
    }


def run_rg_feasibility_scan(
    crg_prob: np.ndarray,
    c_edges: np.ndarray,
    rg_edges: np.ndarray,
    rg_target_lattice: np.ndarray,
    rg_target_observed: np.ndarray,
    *,
    rg_scale: float,
    summary: str,
    bias_min: float,
    bias_max: float,
    bias_points: int,
    outdir: Path,
    make_plots: bool,
) -> Dict[str, Any]:
    """Scan scalar contact bias b and report which Rg values are reachable.

    This is a pure diagnostic: it scans b directly and never touches the
    temperature model or the production fit.  It answers the question that
    decides whether a scalar-Rg fit is meaningful at all — can the joint baseline
    P0(m, Rg), reweighted by any bias in the scanned window, actually produce the
    observed Rg values?

    For ``summary == "mean"`` the exact identity d<Rg>_b/db = -Cov_b(Rg, m) holds
    in lattice units and is verified numerically.  For ``rms`` no such simple
    identity is claimed: only the numerical derivative and the moments are
    reported.

    Writes rg_feasibility.csv, rg_feasibility_summary.json, and (unless plots are
    disabled) rg_feasibility.png.  Returns the summary dict.
    """
    if not np.isfinite(bias_min) or not np.isfinite(bias_max):
        raise ValueError("--rg-bias-min/--rg-bias-max must be finite")
    if not (bias_min < bias_max):
        raise ValueError(
            f"--rg-bias-min ({bias_min}) must be strictly less than "
            f"--rg-bias-max ({bias_max})"
        )
    if int(bias_points) < 3:
        raise ValueError(f"--rg-bias-points must be >= 3, got {bias_points}")

    c_edges = np.asarray(c_edges, dtype=float)
    rg_edges = np.asarray(rg_edges, dtype=float)
    crg_prob = np.asarray(crg_prob, dtype=float)
    m_centers = 0.5 * (c_edges[:-1] + c_edges[1:])
    rg_centers = 0.5 * (rg_edges[:-1] + rg_edges[1:])

    bias_grid = np.linspace(float(bias_min), float(bias_max), int(bias_points))
    rows: List[Dict[str, float]] = []
    for b in bias_grid:
        st = joint_reweight_stats(crg_prob, m_centers, rg_centers, float(b), summary)
        st["bias"] = float(b)
        st["pred_rg_observed"] = float(rg_scale) * st["pred_rg_lattice"]
        rows.append(st)

    pred_lat = np.array([r["pred_rg_lattice"] for r in rows], dtype=float)
    pred_obs = np.array([r["pred_rg_observed"] for r in rows], dtype=float)
    mean_ct = np.array([r["mean_contacts"] for r in rows], dtype=float)
    cov_arr = np.array([r["cov_contact_rg"] for r in rows], dtype=float)
    corr_arr = np.array([r["corr_contact_rg"] for r in rows], dtype=float)
    mean_rg_lat = np.array([r["mean_rg_lattice"] for r in rows], dtype=float)
    d_rg_db = np.gradient(pred_lat, bias_grid)

    # Baseline (b = 0) statistics.
    base = joint_reweight_stats(crg_prob, m_centers, rg_centers, 0.0, summary)

    # Baseline Rg support: edges of the bins carrying non-zero mass.
    rg_marginal = crg_prob.sum(axis=0)
    nz = np.nonzero(rg_marginal > 0.0)[0]
    if nz.size == 0:
        raise ValueError("joint baseline has zero total Rg mass")
    baseline_rg_min = float(rg_edges[nz[0]])
    baseline_rg_max = float(rg_edges[nz[-1] + 1])

    reach_lo_lat, reach_hi_lat = float(pred_lat.min()), float(pred_lat.max())
    tgt_lo_lat, tgt_hi_lat = float(rg_target_lattice.min()), float(rg_target_lattice.max())

    warnings: List[str] = []

    # 1. Target outside the baseline Rg support.
    if tgt_lo_lat < baseline_rg_min or tgt_hi_lat > baseline_rg_max:
        warnings.append(
            f"Target Rg range [{tgt_lo_lat:.4g}, {tgt_hi_lat:.4g}] (lattice) is not "
            f"fully inside the baseline Rg support [{baseline_rg_min:.4g}, "
            f"{baseline_rg_max:.4g}] (lattice). The baseline cannot represent the "
            f"out-of-support values at any bias."
        )
    if not (baseline_rg_min < tgt_lo_lat or tgt_hi_lat < baseline_rg_max):
        pass  # overlap presence is checked by the caller via support_overlap below

    # 2. Target outside the reachable scalar range over the scanned bias window.
    n_unreach = int(
        np.sum((rg_target_lattice < reach_lo_lat) | (rg_target_lattice > reach_hi_lat))
    )
    target_within_reachable = bool(n_unreach == 0)
    if not target_within_reachable:
        warnings.append(
            f"{n_unreach} of {rg_target_lattice.size} target Rg value(s) fall outside "
            f"the reachable scalar-{summary} range [{reach_lo_lat:.4g}, {reach_hi_lat:.4g}] "
            f"(lattice) = [{reach_lo_lat * rg_scale:.4g}, {reach_hi_lat * rg_scale:.4g}] "
            f"(observed) over bias in [{bias_min:g}, {bias_max:g}]. "
            f"NO value of b(T) can reproduce those points: the fit cannot be "
            f"scientifically valid for them without changing the baseline, the "
            f"chain length, or rg_scale."
        )

    # 3. Weak contact-Rg coupling at the baseline.
    weak_corr_threshold = 0.1
    baseline_corr = base["corr_contact_rg"]
    if np.isfinite(baseline_corr) and abs(baseline_corr) < weak_corr_threshold:
        warnings.append(
            f"Baseline contact-Rg correlation is weak (|corr| = {abs(baseline_corr):.3g} "
            f"< {weak_corr_threshold}). Contacts barely constrain Rg, so the scalar-Rg "
            f"fit is close to non-identifiable."
        )

    # 4. Rg nearly insensitive to bias.
    rg_span_lat = reach_hi_lat - reach_lo_lat
    max_abs_dg = float(np.max(np.abs(d_rg_db))) if d_rg_db.size else 0.0
    insensitive_threshold = 1e-3
    if max_abs_dg < insensitive_threshold:
        warnings.append(
            f"Rg is nearly insensitive to contact bias (max |dRg/db| = {max_abs_dg:.3g} "
            f"< {insensitive_threshold} lattice units per unit bias) over the scanned "
            f"window; the bias cannot drive a collapse transition."
        )

    # 5. Target range reachable only at a bias-grid boundary.
    edge_frac = 0.02
    n_edge = max(1, int(np.ceil(edge_frac * bias_grid.size)))
    boundary_reached = False
    if target_within_reachable and rg_span_lat > 0:
        # Which bias index best reproduces each target?
        idx = np.array(
            [int(np.argmin(np.abs(pred_lat - t))) for t in rg_target_lattice], dtype=int
        )
        boundary_reached = bool(
            np.any(idx < n_edge) or np.any(idx >= bias_grid.size - n_edge)
        )
        if boundary_reached:
            warnings.append(
                f"At least one target Rg value is matched only at the edge of the "
                f"scanned bias window [{bias_min:g}, {bias_max:g}]. Widen the window "
                f"with --rg-bias-min/--rg-bias-max, or treat the implied bias as a "
                f"lower bound in magnitude."
            )

    # Theoretical derivative check (exact only for the 'mean' summary).
    deriv_check: Dict[str, Any] = {
        "identity": "d<Rg>_b/db = -Cov_b(Rg, m)  (lattice units)",
        "applies_to_summary": "mean",
        "checked": False,
        "max_abs_difference": None,
        "max_abs_relative_difference": None,
        "agrees": None,
        "note": (
            "For summary='rms' this identity does NOT hold: d/db sqrt(<Rg^2>_b) = "
            "-Cov_b(Rg^2, m) / (2 sqrt(<Rg^2>_b)). Only the numerical derivative and "
            "the reported moments are used for rms."
        ),
    }
    if summary == "mean":
        d_mean_db = np.gradient(mean_rg_lat, bias_grid)
        diff = np.abs(d_mean_db - (-cov_arr))
        # Ignore the one-sided endpoints, where np.gradient is only 1st order.
        interior = diff[1:-1] if diff.size > 2 else diff
        scale_ref = max(float(np.max(np.abs(cov_arr))), 1e-12)
        deriv_check.update({
            "checked": True,
            "max_abs_difference": float(np.max(interior)) if interior.size else None,
            "max_abs_relative_difference": (
                float(np.max(interior) / scale_ref) if interior.size else None
            ),
            "agrees": (
                bool(float(np.max(interior)) / scale_ref < 0.05) if interior.size else None
            ),
        })

    summary_dict: Dict[str, Any] = {
        "rg_summary": summary,
        "rg_scale": float(rg_scale),
        "bias_min": float(bias_min),
        "bias_max": float(bias_max),
        "bias_points": int(bias_points),
        "baseline_contact_rg_covariance": _finite_or_none(base["cov_contact_rg"]),
        "baseline_contact_rg_correlation": _finite_or_none(base["corr_contact_rg"]),
        "baseline_mean_contacts": _finite_or_none(base["mean_contacts"]),
        "baseline_rg_min": float(baseline_rg_min),
        "baseline_rg_max": float(baseline_rg_max),
        "baseline_rg_min_observed": float(baseline_rg_min * rg_scale),
        "baseline_rg_max_observed": float(baseline_rg_max * rg_scale),
        "baseline_rg_scalar_lattice": _finite_or_none(base["pred_rg_lattice"]),
        "baseline_rg_scalar_observed": _finite_or_none(base["pred_rg_lattice"] * rg_scale),
        "target_rg_min": float(tgt_lo_lat),
        "target_rg_max": float(tgt_hi_lat),
        "target_rg_min_observed": float(rg_target_observed.min()),
        "target_rg_max_observed": float(rg_target_observed.max()),
        "reachable_rg_min": float(reach_lo_lat),
        "reachable_rg_max": float(reach_hi_lat),
        "reachable_rg_min_observed": float(reach_lo_lat * rg_scale),
        "reachable_rg_max_observed": float(reach_hi_lat * rg_scale),
        "n_targets": int(rg_target_lattice.size),
        "n_targets_outside_reachable": n_unreach,
        "target_within_reachable_range": target_within_reachable,
        "target_reached_only_at_bias_boundary": bool(boundary_reached),
        "max_abs_d_rg_db": float(max_abs_dg),
        "derivative_check": deriv_check,
        "warnings": warnings,
        "units_note": (
            "Fields without an explicit _observed suffix are in LATTICE units. "
            "Observed units = rg_scale * lattice units."
        ),
        "definition_note": (
            "reachable_rg_* is the range of the scalar Rg summary over the scanned "
            "bias window, i.e. what P0(m,Rg) can produce for ANY constant bias b. "
            "A target outside it cannot be fit by any b(T)."
        ),
    }

    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / "rg_feasibility.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "bias", "pred_rg_lattice", "pred_rg_observed", "mean_contacts",
            "cov_contact_rg", "corr_contact_rg", "d_rg_db",
        ])
        for i, b in enumerate(bias_grid):
            writer.writerow([
                f"{b:.8g}", f"{pred_lat[i]:.8g}", f"{pred_obs[i]:.8g}",
                f"{mean_ct[i]:.8g}", f"{cov_arr[i]:.8g}",
                ("" if not np.isfinite(corr_arr[i]) else f"{corr_arr[i]:.8g}"),
                f"{d_rg_db[i]:.8g}",
            ])
    print(f"Saved: {csv_path}")

    json_path = outdir / "rg_feasibility_summary.json"
    with open(json_path, "w") as fh:
        json.dump(summary_dict, fh, indent=2, cls=_NpEncoder, allow_nan=False)
    print(f"Saved: {json_path}")

    if make_plots:
        if plt is None:
            raise RuntimeError(
                "matplotlib is required for plots; install it or use --no-plots"
            )
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(bias_grid, pred_obs, "k-", lw=1.8, label=f"predicted Rg ({summary})")
        ax.axhspan(
            float(rg_target_observed.min()), float(rg_target_observed.max()),
            color="tab:orange", alpha=0.2, label="target Rg range",
        )
        ax.axhline(float(rg_target_observed.min()), color="tab:orange", lw=0.8, ls="--")
        ax.axhline(float(rg_target_observed.max()), color="tab:orange", lw=0.8, ls="--")
        ax.axvline(0.0, color="gray", lw=0.8, ls=":")
        ax.set_xlabel("contact bias b  (P ∝ P0(m,Rg) exp[-b m])")
        ax.set_ylabel(f"scalar Rg, observed units (rg_scale={rg_scale:g})")
        status = "REACHABLE" if target_within_reachable else "NOT REACHABLE"
        ax.set_title(
            f"Bias→Rg reachability [{summary}] — target range {status}\n"
            f"reachable [{reach_lo_lat * rg_scale:.4g}, {reach_hi_lat * rg_scale:.4g}], "
            f"target [{float(rg_target_observed.min()):.4g}, "
            f"{float(rg_target_observed.max()):.4g}]"
        )
        ax.legend(fontsize=8)
        fig.tight_layout()
        p = outdir / "rg_feasibility.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {p}")

    return summary_dict


# ---------------------------------------------------------------------------
# Scalar Rg(T) mode: driver
# ---------------------------------------------------------------------------

def _rg_scalar_objective_factory(
    cfg: Dict[str, Any]
) -> Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray], float]:
    """Return an objective f(params, temps, target, lower, upper) for fit_restarts.

    Closes over the immutable fit configuration so the optimizer's ``args`` tuple
    carries only the per-split data, which is what the bootstrap and the split
    sweep vary.
    """
    def obj(
        p: np.ndarray,
        temps_: np.ndarray,
        target_: np.ndarray,
        lower_: np.ndarray,
        upper_: np.ndarray,
    ) -> float:
        return objective_rg_scalar(
            p, temps_, target_,
            cfg["crg_prob"], cfg["c_edges"], cfg["rg_edges_lattice"], cfg["b_fn"],
            n_model_params=cfg["n_model_params"],
            fixed_rg_scale=cfg["rg_scale"],
            fit_rg_scale=cfg["fit_rg_scale"],
            rg_summary=cfg["rg_summary"],
            target_units=cfg["target_units"],
            loss_name=cfg["loss_name"],
            rg_lower=lower_,
            rg_upper=upper_,
            huber_delta=cfg["huber_delta"],
            range_floor=cfg["range_floor"],
        )
    return obj


def _rg_scalar_predict(
    cfg: Dict[str, Any], optimization_params: np.ndarray, temps: np.ndarray
) -> Dict[str, Any]:
    """Predict scalar Rg (both unit systems) and P(Rg|T) for one parameter vector."""
    model_params, scale = split_rg_scalar_params(
        optimization_params, cfg["n_model_params"], cfg["fit_rg_scale"], cfg["rg_scale"]
    )
    pred_lat, pred_obs, mass = predict_rg_summary_from_joint(
        cfg["crg_prob"], cfg["c_edges"], cfg["rg_edges_lattice"], temps,
        model_params, cfg["b_fn"],
        rg_scale=scale, summary=cfg["rg_summary"], target_units=cfg["target_units"],
    )
    return {
        "model_params": model_params,
        "rg_scale": scale,
        "pred_lattice": pred_lat,
        "pred_observed": pred_obs,
        "pred_target_units": rg_pred_in_target_units(pred_lat, pred_obs, cfg["target_units"]),
        "mass": mass,
    }


def _rg_scalar_score(
    cfg: Dict[str, Any],
    optimization_params: np.ndarray,
    temps: np.ndarray,
    target: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> Dict[str, Optional[float]]:
    """Objective + raw metrics for one parameter vector on one set of temperatures."""
    if temps.size == 0:
        return {
            "objective_mean": None, "rmse": None, "mae": None,
            "max_abs_error": None, "bias": None, "n_temps": 0,
        }
    pred = _rg_scalar_predict(cfg, optimization_params, temps)["pred_target_units"]
    out: Dict[str, Optional[float]] = {
        "objective_mean": rg_scalar_objective_value(
            pred, target, cfg["loss_name"], rg_lower=lower, rg_upper=upper,
            huber_delta=cfg["huber_delta"], range_floor=cfg["range_floor"],
        ),
        "n_temps": int(temps.size),
    }
    out.update(rg_scalar_raw_metrics(pred, target))
    return out


def _rg_scalar_transitions(
    cfg: Dict[str, Any], optimization_params: np.ndarray, t_lo: float, t_hi: float
) -> Dict[str, Any]:
    """Bias zero-crossing(s) and fitted-curve transition descriptors."""
    model_params, scale = split_rg_scalar_params(
        optimization_params, cfg["n_model_params"], cfg["fit_rg_scale"], cfg["rg_scale"]
    )
    crossings = bias_zero_crossings(cfg["b_fn"], model_params, t_lo, t_hi)
    curve = rg_curve_transition_metrics(
        cfg["crg_prob"], cfg["c_edges"], cfg["rg_edges_lattice"],
        model_params, cfg["b_fn"], t_lo, t_hi,
        rg_scale=scale, summary=cfg["rg_summary"], target_units=cfg["target_units"],
        n_grid=cfg["dense_grid_points"],
    )
    derived_tc = None
    if cfg["spec"]["derived_Tc"] is not None:
        derived_tc = cfg["spec"]["derived_Tc"](model_params)
    return {
        "bias_zero_crossings": crossings,
        "T_bias_zero": (crossings[0] if crossings else None),
        "T_bias_zero_model_derived": _finite_or_none(derived_tc),
        **curve,
    }


def run_rg_scalar_mode(args: argparse.Namespace) -> None:
    """Fit the contact-bias model directly to scalar Rg(T) data.

    Never fabricates contact targets or Rg histograms: the objective is a scalar
    regression loss between the observed Rg(T) and the summary of the reweighted
    joint baseline P0(m, Rg).
    """
    if minimize is None:
        raise RuntimeError("scipy is required for fitting. Install scipy.")

    # ---- flag compatibility -------------------------------------------------
    if args.fit_rg:
        raise ValueError(
            "--rg-means-file fits SCALAR Rg(T) and is incompatible with --fit-rg, "
            "which fits full observed Rg histograms from a REMD NPZ. Choose one: "
            "drop --fit-rg for scalar mode, or drop --rg-means-file for histogram mode."
        )
    if args.rg_weight_grid is not None or args.rg_weight_grid_file is not None:
        raise ValueError(
            "--rg-weight-grid/--rg-weight-grid-file trade contact loss against Rg "
            "loss and have no meaning in scalar-Rg mode, whose objective is scalar "
            "Rg only."
        )
    if not np.isfinite(args.rg_huber_delta) or args.rg_huber_delta <= 0:
        raise ValueError("--rg-huber-delta must be finite and positive")
    if not np.isfinite(args.rg_range_floor) or args.rg_range_floor <= 0:
        raise ValueError("--rg-range-floor must be finite and positive")
    if args.fit_rg_scale and args.rg_target_units == "lattice":
        raise ValueError(
            "--fit-rg-scale with --rg-target-units lattice is not identifiable: "
            "rg_scale never enters the loss when the targets are already in lattice "
            "units, so the objective is flat in that parameter. Use "
            "--rg-target-units observed to fit the scale, or drop --fit-rg-scale."
        )

    # ---- output paths (same convention as the contact-histogram mode) -------
    if args.outdir is not None:
        outdir = Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        npz_path = outdir / "fit_results.npz"
        loss_csv_path = outdir / "train_validation_loss.csv"
        json_path = outdir / "fit_summary.json"
        params_csv_path = outdir / "fit_params.csv"
        plot_dir = outdir
    else:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        npz_path = out_path
        loss_csv_path = out_path.parent / "train_validation_loss.csv"
        json_path = out_path.parent / "fit_summary.json"
        params_csv_path = out_path.parent / "fit_params.csv"
        plot_dir = out_path.parent
    make_plots = not args.no_plots

    # ---- scalar target data -------------------------------------------------
    data = load_rg_mean_file(args.rg_means_file)
    temps = data["temps"]
    rg_target_input = data["rg_target"]
    rg_lower_input = data["rg_lower"]
    rg_upper_input = data["rg_upper"]
    n_temps = temps.size
    target_units = args.rg_target_units

    # ---- joint baseline (mandatory) ----------------------------------------
    b_data = np.load(args.baseline)
    missing = [k for k in ("c_edges", "rg_edges", "crg_prob") if k not in b_data.files]
    if missing:
        raise ValueError(
            f"scalar-Rg mode requires a JOINT baseline P0(m, Rg). Baseline "
            f"{args.baseline!r} is missing: {', '.join(missing)} "
            f"(found: {list(b_data.files)})."
        )
    crg_prob = np.asarray(b_data["crg_prob"], dtype=float)
    c_edges = np.asarray(b_data["c_edges"], dtype=float)
    rg_edges_lattice = np.asarray(b_data["rg_edges"], dtype=float)

    if c_edges.ndim != 1 or rg_edges_lattice.ndim != 1 or crg_prob.ndim != 2:
        raise ValueError(
            f"baseline shapes invalid: c_edges {c_edges.shape}, rg_edges "
            f"{rg_edges_lattice.shape}, crg_prob {crg_prob.shape}"
        )
    expected = (c_edges.size - 1, rg_edges_lattice.size - 1)
    if crg_prob.shape != expected:
        raise ValueError(
            f"baseline crg_prob shape must be (len(c_edges)-1, len(rg_edges)-1): "
            f"got {crg_prob.shape}, expected {expected}"
        )
    if not np.all(np.isfinite(crg_prob)) or np.any(crg_prob < 0.0):
        raise ValueError("baseline crg_prob must be finite and non-negative")
    if crg_prob.sum() <= 0.0:
        raise ValueError("baseline crg_prob must contain positive total mass")
    if np.any(np.diff(c_edges) <= 0) or np.any(np.diff(rg_edges_lattice) <= 0):
        raise ValueError("baseline c_edges and rg_edges must be strictly increasing")

    rg_centers_lattice = 0.5 * (rg_edges_lattice[:-1] + rg_edges_lattice[1:])
    rg_scale = float(args.rg_scale)

    # ---- targets in both unit systems (conversion applied exactly once) -----
    if target_units == "observed":
        rg_target_observed = rg_target_input.copy()
        rg_target_lattice = rg_target_input / rg_scale
    else:
        rg_target_lattice = rg_target_input.copy()
        rg_target_observed = rg_scale * rg_target_input

    # ---- model setup --------------------------------------------------------
    spec = MODEL_REGISTRY[args.model]
    Tmin, Tmax = float(temps.min()), float(temps.max())
    Tref = float(args.Tref) if args.Tref is not None else 0.5 * (Tmin + Tmax)
    Tscale = float(args.Tscale) if args.Tscale is not None else max(Tmax - Tmin, 1.0)
    if args.model == "heat_capacity" and args.T0 is not None:
        Tref = float(args.T0)
    if not np.isfinite(Tref):
        raise ValueError(f"Tref must be finite, got {Tref!r}")
    if not np.isfinite(Tscale) or Tscale <= 0.0:
        raise ValueError(f"Tscale must be finite and positive, got {Tscale!r}")
    if args.model == "heat_capacity" and Tref <= 0.0:
        raise ValueError(f"heat_capacity T0 must be positive, got {Tref!r}")
    b_fn = make_b_fn(args.model, Tref, Tscale)

    pspec = build_rg_scalar_param_spec(
        args.model,
        fit_rg_scale=bool(args.fit_rg_scale),
        rg_scale=rg_scale,
        rg_scale_min=float(args.rg_scale_min),
        rg_scale_max=float(args.rg_scale_max),
        n_restarts=int(args.n_restarts),
        seed=int(args.seed),
    )
    param_names: List[str] = pspec["param_names"]
    bounds: List[Tuple[float, float]] = pspec["bounds"]
    x0s: List[np.ndarray] = pspec["x0s"]

    cfg: Dict[str, Any] = {
        "crg_prob": crg_prob,
        "c_edges": c_edges,
        "rg_edges_lattice": rg_edges_lattice,
        "b_fn": b_fn,
        "spec": spec,
        "n_model_params": pspec["n_model_params"],
        "rg_scale": rg_scale,
        "fit_rg_scale": bool(args.fit_rg_scale),
        "rg_summary": args.rg_summary,
        "target_units": target_units,
        "loss_name": args.rg_mean_loss,
        "huber_delta": float(args.rg_huber_delta),
        "range_floor": float(args.rg_range_floor),
        "dense_grid_points": 1001,
    }
    obj_fn = _rg_scalar_objective_factory(cfg)

    # ---- report -------------------------------------------------------------
    print("--- Scalar Rg(T) fitting mode ---")
    print(f"Scalar Rg data: {args.rg_means_file}")
    print(f"  temperatures: {n_temps}  range [{Tmin:.4g}, {Tmax:.4g}] K")
    print(f"  target units: {target_units}")
    print(f"  target Rg (input units):   [{rg_target_input.min():.4g}, {rg_target_input.max():.4g}]")
    print(f"  target Rg (lattice units): [{rg_target_lattice.min():.4g}, {rg_target_lattice.max():.4g}]")
    print(f"  target Rg (observed units):[{rg_target_observed.min():.4g}, {rg_target_observed.max():.4g}]")
    print("  NOTE: the lower/upper columns are treated as DESCRIPTIVE BOUNDS, "
          "not standard errors.")
    print(f"Baseline NPZ: {args.baseline}")
    print(f"  keys: {list(b_data.files)}")
    print(f"  joint P0(m,Rg): available, shape {crg_prob.shape}")
    print(f"  Rg grid (lattice):  [{rg_edges_lattice.min():.4g}, {rg_edges_lattice.max():.4g}]")
    print(f"  Rg grid (observed): [{rg_edges_lattice.min() * rg_scale:.4g}, "
          f"{rg_edges_lattice.max() * rg_scale:.4g}]")
    print(f"Model : {args.model}  —  {spec['description']}")
    print(f"  Rg summary: {args.rg_summary}   objective: {args.rg_mean_loss} "
          f"(mean per temperature)")
    print(f"  rg_scale: {rg_scale:.8g} observed units per lattice unit"
          + ("  [FITTED as a free parameter]" if args.fit_rg_scale else "  [FIXED]"))
    print(f"  Parameters: {param_names}")

    # ---- feasibility scan (diagnostic only; never touches the fit) ----------
    feasibility: Optional[Dict[str, Any]] = None
    if args.rg_feasibility_scan:
        # Console output stays ASCII: cp1252 terminals cannot encode arrows.
        print("\n--- Bias-to-Rg feasibility scan (diagnostic; does not affect the fit) ---")
        feasibility = run_rg_feasibility_scan(
            crg_prob, c_edges, rg_edges_lattice,
            rg_target_lattice, rg_target_observed,
            rg_scale=rg_scale,
            summary=args.rg_summary,
            bias_min=float(args.rg_bias_min),
            bias_max=float(args.rg_bias_max),
            bias_points=int(args.rg_bias_points),
            outdir=plot_dir,
            make_plots=make_plots,
        )
        print(f"  baseline contact-Rg correlation: "
              f"{feasibility['baseline_contact_rg_correlation']}")
        print(f"  reachable scalar Rg (lattice):  "
              f"[{feasibility['reachable_rg_min']:.4g}, {feasibility['reachable_rg_max']:.4g}]")
        print(f"  reachable scalar Rg (observed): "
              f"[{feasibility['reachable_rg_min_observed']:.4g}, "
              f"{feasibility['reachable_rg_max_observed']:.4g}]")
        print(f"  target scalar Rg (observed):    "
              f"[{feasibility['target_rg_min_observed']:.4g}, "
              f"{feasibility['target_rg_max_observed']:.4g}]")
        for w in feasibility["warnings"]:
            print(f"  WARNING: {w}")
        if not feasibility["warnings"]:
            print("  No feasibility warnings.")

    # ---- train / validation split ------------------------------------------
    train_idx, val_idx = _resolve_split_indices(
        n_temps=n_temps,
        holdout_every=args.holdout_every,
        holdout_indices_str=args.holdout_indices,
        train_indices_str=args.train_indices,
    )
    if train_idx.size == 0:
        raise ValueError("Training set is empty. Adjust holdout options.")
    has_val = val_idx.size > 0
    print(f"\n  train: {train_idx.size} temps"
          + (f"  |  validation: {val_idx.size} temps" if has_val else "  |  no validation set"))

    train_args = (
        temps[train_idx], rg_target_input[train_idx],
        rg_lower_input[train_idx], rg_upper_input[train_idx],
    )

    # ---- primary fit (training temperatures only) --------------------------
    best, best_obj, restart_records = fit_restarts(obj_fn, train_args, x0s, bounds)
    params_all = np.asarray(best.x, dtype=float)
    model_params, fitted_scale = split_rg_scalar_params(
        params_all, cfg["n_model_params"], cfg["fit_rg_scale"], rg_scale
    )
    boundary_hits = count_boundary_hits(params_all, bounds, param_names)

    print("\nBest-fit parameters:")
    for name, val in zip(param_names, params_all):
        print(f"  {name} = {val:.6g}")
    if args.fit_rg_scale:
        print(f"  (rg_scale fitted within [{args.rg_scale_min:g}, {args.rg_scale_max:g}]; "
              f"fixed-scale input was {rg_scale:g})")
    print(f"Objective ({args.rg_mean_loss}, mean over {train_idx.size} train temps) "
          f"= {best_obj:.6g}")
    if boundary_hits:
        print(f"  WARNING: parameter(s) resting on a bound: {', '.join(boundary_hits)}")

    # ---- predictions at all temperatures -----------------------------------
    pred = _rg_scalar_predict(cfg, params_all, temps)
    rg_pred_lattice = pred["pred_lattice"]
    rg_pred_observed = pred["pred_observed"]
    rg_pred_target_units = pred["pred_target_units"]
    rg_mod_mass = pred["mass"]
    rg_residual_target_units = rg_pred_target_units - rg_target_input
    per_temp_obj = rg_scalar_per_temp_losses(
        rg_pred_target_units, rg_target_input, cfg["loss_name"],
        rg_lower=rg_lower_input, rg_upper=rg_upper_input,
        huber_delta=cfg["huber_delta"], range_floor=cfg["range_floor"],
    )
    inside_range = (rg_pred_target_units >= rg_lower_input) & (
        rg_pred_target_units <= rg_upper_input
    )
    b_T = np.array([b_fn(model_params, float(T)) for T in temps], dtype=float)

    # ---- metrics ------------------------------------------------------------
    train_metrics = _rg_scalar_score(
        cfg, params_all, temps[train_idx], rg_target_input[train_idx],
        rg_lower_input[train_idx], rg_upper_input[train_idx],
    )
    val_metrics = _rg_scalar_score(
        cfg, params_all, temps[val_idx], rg_target_input[val_idx],
        rg_lower_input[val_idx], rg_upper_input[val_idx],
    )
    all_metrics = _rg_scalar_score(
        cfg, params_all, temps, rg_target_input, rg_lower_input, rg_upper_input
    )

    print(f"\nScalar Rg loss ({args.rg_mean_loss}, mean per temperature; "
          f"raw metrics in {target_units} units):")
    for label, mt in (("train", train_metrics), ("validation", val_metrics),
                      ("all", all_metrics)):
        if mt["n_temps"] == 0:
            continue
        print(f"  {label:11s} ({mt['n_temps']:3d} temps): "
              f"objective={mt['objective_mean']:.6g}  rmse={mt['rmse']:.6g}  "
              f"mae={mt['mae']:.6g}  max|err|={mt['max_abs_error']:.6g}")

    # ---- transition descriptors --------------------------------------------
    trans = _rg_scalar_transitions(cfg, params_all, Tmin, Tmax)
    print("\nTransition descriptors:")
    if trans["bias_zero_crossings"]:
        print(f"  T_bias_zero (b(T)=0) within [{Tmin:.4g}, {Tmax:.4g}]: "
              + ", ".join(f"{t:.6g}" for t in trans["bias_zero_crossings"]))
    else:
        print(f"  T_bias_zero: no b(T)=0 crossing inside [{Tmin:.4g}, {Tmax:.4g}]")
    if trans["T_bias_zero_model_derived"] is not None:
        print(f"  Tc (model-derived bias zero, may lie outside the data range) = "
              f"{trans['T_bias_zero_model_derived']:.6g}")
    print(f"  T_rg_max_slope = {trans['T_rg_max_slope']:.6g} K   "
          f"(-dRg/dT = {trans['rg_max_negative_slope']:.6g} {target_units} units/K)")
    print(f"  T_rg_half      = {trans['T_rg_half']:.6g} K")
    print("  NOTE: T_bias_zero (bias sign change) and T_rg_max_slope (steepest "
          "finite-chain collapse) are distinct quantities.")

    # ---- assemble warnings --------------------------------------------------
    warnings_list: List[str] = []
    if feasibility is not None:
        warnings_list.extend(feasibility["warnings"])
    else:
        warnings_list.append(
            "Feasibility diagnostics were not run (--rg-feasibility-scan not given). "
            "Whether the joint baseline can reach the observed Rg range is UNVERIFIED."
        )
    if boundary_hits:
        warnings_list.append(
            f"Fitted parameter(s) rest on an optimization bound: {', '.join(boundary_hits)}. "
            f"The optimum may lie outside the allowed range."
        )
    n_outside = int(np.sum(~inside_range))
    if n_outside:
        warnings_list.append(
            f"{n_outside} of {n_temps} fitted predictions fall outside the supplied "
            f"lower/upper bounds. This is a descriptive comparison, NOT confidence-"
            f"interval coverage."
        )

    # ---- support diagnostics -----------------------------------------------
    rg_marg = crg_prob.sum(axis=0)
    nz = np.nonzero(rg_marg > 0.0)[0]
    baseline_rg_lo = float(rg_edges_lattice[nz[0]])
    baseline_rg_hi = float(rg_edges_lattice[nz[-1] + 1])
    target_in_support = bool(
        rg_target_lattice.min() >= baseline_rg_lo
        and rg_target_lattice.max() <= baseline_rg_hi
    )
    if not target_in_support:
        warnings_list.append(
            f"Target Rg range [{rg_target_lattice.min():.4g}, {rg_target_lattice.max():.4g}] "
            f"(lattice) is not contained in the baseline Rg support "
            f"[{baseline_rg_lo:.4g}, {baseline_rg_hi:.4g}] (lattice)."
        )
    support_diagnostics: Dict[str, Any] = {
        "baseline_rg_support_lattice": [baseline_rg_lo, baseline_rg_hi],
        "baseline_rg_support_observed": [
            baseline_rg_lo * rg_scale, baseline_rg_hi * rg_scale
        ],
        "target_rg_range_lattice": [
            float(rg_target_lattice.min()), float(rg_target_lattice.max())
        ],
        "target_rg_range_observed": [
            float(rg_target_observed.min()), float(rg_target_observed.max())
        ],
        "target_within_baseline_rg_support": target_in_support,
        "n_predictions_inside_input_range": int(np.sum(inside_range)),
        "n_temps": int(n_temps),
        "inside_input_range_note": (
            "inside_input_range compares the fitted prediction against the supplied "
            "lower/upper columns. It is NOT confidence-interval coverage."
        ),
    }

    # ---- fit_results.npz ----------------------------------------------------
    save_kwargs: Dict[str, Any] = dict(
        mode="rg_scalar",
        temps=temps,
        rg_target_input=rg_target_input,
        rg_lower_input=rg_lower_input,
        rg_upper_input=rg_upper_input,
        rg_target_units=str(target_units),
        rg_target_lattice=rg_target_lattice,
        rg_target_observed=rg_target_observed,
        rg_pred_lattice=rg_pred_lattice,
        rg_pred_observed=rg_pred_observed,
        rg_pred_target_units=rg_pred_target_units,
        rg_residual_target_units=rg_residual_target_units,
        rg_mod_mass=rg_mod_mass,
        rg_centers_lattice=rg_centers_lattice,
        rg_centers_observed=rg_scale * rg_centers_lattice,
        rg_edges_lattice=rg_edges_lattice,
        rg_edges_observed=rg_scale * rg_edges_lattice,
        rg_summary=str(args.rg_summary),
        rg_mean_loss=str(args.rg_mean_loss),
        rg_scale=float(fitted_scale),
        rg_scale_input=float(rg_scale),
        rg_scale_was_fitted=bool(args.fit_rg_scale),
        params=params_all,
        param_names=np.array(param_names),
        model_params=model_params,
        model_param_names=np.array(pspec["model_param_names"]),
        b_T=b_T,
        train_indices=train_idx,
        validation_indices=val_idx,
        T_rg_max_slope=float(trans["T_rg_max_slope"]),
        rg_max_negative_slope=float(trans["rg_max_negative_slope"]),
        T_rg_half=float(trans["T_rg_half"]),
        T_bias_zero=(
            float(trans["T_bias_zero"]) if trans["T_bias_zero"] is not None else np.nan
        ),
        bias_zero_crossings=np.array(trans["bias_zero_crossings"], dtype=float),
        model_name=str(args.model),
        baseline=str(args.baseline),
        rg_means_file=str(args.rg_means_file),
        Tref=float(Tref),
        Tscale=float(Tscale),
        per_temp_objective=per_temp_obj,
        inside_input_range=inside_range,
        objective_train=float(train_metrics["objective_mean"]),
        objective_all=float(all_metrics["objective_mean"]),
    )
    if has_val:
        save_kwargs["objective_validation"] = float(val_metrics["objective_mean"])
    np.savez_compressed(npz_path, **save_kwargs)
    print(f"\nSaved: {npz_path}")

    # ---- rg_fit_by_temperature.csv -----------------------------------------
    per_temp_csv = plot_dir / "rg_fit_by_temperature.csv"
    val_set = set(val_idx.tolist())
    with open(per_temp_csv, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "temp_index", "temperature", "split",
            "rg_target_input", "rg_lower_input", "rg_upper_input", "rg_target_units",
            "rg_target_lattice", "rg_target_observed",
            "rg_pred_lattice", "rg_pred_observed",
            "residual_target_units", "absolute_error_target_units",
            "squared_error_target_units", "objective_contribution",
            "inside_input_range", "b_T",
        ])
        for i in range(n_temps):
            r = float(rg_residual_target_units[i])
            writer.writerow([
                i, f"{temps[i]:.8g}",
                ("validation" if i in val_set else "train"),
                f"{rg_target_input[i]:.8g}", f"{rg_lower_input[i]:.8g}",
                f"{rg_upper_input[i]:.8g}", target_units,
                f"{rg_target_lattice[i]:.8g}", f"{rg_target_observed[i]:.8g}",
                f"{rg_pred_lattice[i]:.8g}", f"{rg_pred_observed[i]:.8g}",
                f"{r:.8g}", f"{abs(r):.8g}", f"{r * r:.8g}",
                f"{per_temp_obj[i]:.8g}",
                bool(inside_range[i]), f"{b_T[i]:.8g}",
            ])
    print(f"Saved: {per_temp_csv}")

    # ---- train_validation_loss.csv -----------------------------------------
    with open(loss_csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "split", "n_temps", "loss_name", "objective_mean",
            "rmse", "mae", "max_abs_error", "units",
        ])
        for label, mt in (("train", train_metrics), ("validation", val_metrics),
                          ("all", all_metrics)):
            if mt["n_temps"] == 0:
                continue
            writer.writerow([
                label, mt["n_temps"], args.rg_mean_loss,
                f"{mt['objective_mean']:.8g}", f"{mt['rmse']:.8g}",
                f"{mt['mae']:.8g}", f"{mt['max_abs_error']:.8g}", target_units,
            ])
    print(f"Saved: {loss_csv_path}")

    # ---- fit_params.csv -----------------------------------------------------
    param_descriptions: Dict[str, str] = {
        "h": "enthalpy-like coefficient of b(T) = h/T - s (reduced units x K)",
        "s": "entropy-like offset of b(T) = h/T - s (reduced units)",
        "A": "amplitude of b(T) = A*(Tc/T - 1) (reduced units)",
        "Tc": "bias zero-crossing temperature of b(T) = A*(Tc/T - 1) (K)",
        "a2": "quadratic coefficient in x(T) = (T-Tref)/Tscale (reduced units)",
        "a0": "constant term of b(T) polynomial (reduced units)",
        "a1": "linear coefficient in x(T) (reduced units)",
        "a3": "cubic coefficient in x(T) (reduced units)",
        "dh0": "enthalpy change at T0 (reduced units x K)",
        "ds0": "entropy change at T0 (reduced units)",
        "dCp": "heat-capacity change of folding (reduced units)",
        "rg_scale": "observed Rg units per lattice unit (Rg_obs = rg_scale*Rg_lattice)",
    }
    with open(params_csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "parameter", "value", "kind", "bound_low", "bound_high",
            "at_boundary", "units_or_description",
        ])
        for j, (name, val) in enumerate(zip(param_names, params_all)):
            lo, hi = bounds[j]
            kind = "scale" if name == "rg_scale" and args.fit_rg_scale else "thermodynamic"
            at_b = ""
            for hit in boundary_hits:
                if hit.split("@")[0] == name:
                    at_b = hit.split("@")[1]
            writer.writerow([
                name, f"{float(val):.8g}", kind, f"{lo:.8g}", f"{hi:.8g}",
                at_b, param_descriptions.get(name, ""),
            ])
        if not args.fit_rg_scale:
            writer.writerow([
                "rg_scale", f"{rg_scale:.8g}", "scale_fixed", "", "", "",
                param_descriptions["rg_scale"] + " [FIXED, not optimized]",
            ])
        if trans["T_bias_zero"] is not None:
            writer.writerow([
                "T_bias_zero", f"{trans['T_bias_zero']:.8g}", "derived", "", "", "",
                "temperature where b(T)=0 inside the observed range (K)",
            ])
        writer.writerow([
            "T_rg_max_slope", f"{trans['T_rg_max_slope']:.8g}", "derived", "", "", "",
            "temperature of steepest fitted collapse, argmax of -dRg/dT (K)",
        ])
        writer.writerow([
            "T_rg_half", f"{trans['T_rg_half']:.8g}", "derived", "", "", "",
            "temperature nearest the midpoint Rg between the interval ends (K)",
        ])
    print(f"Saved: {params_csv_path}")

    # ---- fit_summary.json ---------------------------------------------------
    def _metrics_json(mt: Dict[str, Optional[float]]) -> Dict[str, Any]:
        if mt["n_temps"] == 0:
            return {
                "n_temps": 0, "objective_mean": None, "rmse": None,
                "mae": None, "max_abs_error": None, "bias": None,
            }
        return {
            "n_temps": int(mt["n_temps"]),
            "objective_mean": _finite_or_none(mt["objective_mean"]),
            "rmse": _finite_or_none(mt["rmse"]),
            "mae": _finite_or_none(mt["mae"]),
            "max_abs_error": _finite_or_none(mt["max_abs_error"]),
            "bias": _finite_or_none(mt["bias"]),
        }

    summary_json: Dict[str, Any] = {
        "mode": "rg_scalar",
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "model_api_version": MODEL_API_VERSION,
        "input_file": str(args.rg_means_file),
        "baseline_file": str(args.baseline),
        "model": str(args.model),
        "model_description": str(spec["description"]),
        "target_units": target_units,
        "rg_summary": str(args.rg_summary),
        "rg_loss": str(args.rg_mean_loss),
        "rg_scale": float(fitted_scale),
        "rg_scale_input": float(rg_scale),
        "rg_scale_was_fitted": bool(args.fit_rg_scale),
        "rg_scale_bounds": (
            [float(args.rg_scale_min), float(args.rg_scale_max)]
            if args.fit_rg_scale else None
        ),
        "rg_huber_delta": float(args.rg_huber_delta) if args.rg_mean_loss == "huber" else None,
        "rg_range_floor": (
            float(args.rg_range_floor) if args.rg_mean_loss == "range_weighted" else None
        ),
        "parameters": {n: float(v) for n, v in zip(param_names, params_all)},
        "model_parameters": {
            n: float(v) for n, v in zip(pspec["model_param_names"], model_params)
        },
        "parameter_bounds": {
            n: [float(lo), float(hi)] for n, (lo, hi) in zip(param_names, bounds)
        },
        "boundary_hits": boundary_hits,
        "optimization": {
            "success": bool(best.success),
            "message": str(best.message),
            "iterations": int(best.nit) if hasattr(best, "nit") else None,
            "objective_value": _finite_or_none(best_obj),
            "n_restarts": int(args.n_restarts),
            "seed": int(args.seed),
            "restart_stability": restart_stability(restart_records),
        },
        "Tref": float(Tref),
        "Tscale": float(Tscale),
        "n_temps": int(n_temps),
        "n_train": int(train_idx.size),
        "n_val": int(val_idx.size),
        "temps_all": temps.tolist(),
        "train_indices": train_idx.tolist(),
        "validation_indices": val_idx.tolist(),
        "train_metrics": _metrics_json(train_metrics),
        "validation_metrics": _metrics_json(val_metrics),
        "all_metrics": _metrics_json(all_metrics),
        "transition_metrics": {
            "T_bias_zero": _finite_or_none(trans["T_bias_zero"]),
            "bias_zero_crossings": [float(t) for t in trans["bias_zero_crossings"]],
            "T_bias_zero_model_derived": trans["T_bias_zero_model_derived"],
            "T_rg_max_slope": _finite_or_none(trans["T_rg_max_slope"]),
            "rg_max_negative_slope": _finite_or_none(trans["rg_max_negative_slope"]),
            "T_rg_half": _finite_or_none(trans["T_rg_half"]),
            "rg_half_value": _finite_or_none(trans["rg_half_value"]),
            "dense_grid_points": int(cfg["dense_grid_points"]),
        },
        "support_diagnostics": support_diagnostics,
        "definitions": {
            "T_bias_zero": (
                "Temperature at which the contact bias b(T) changes sign (b(T)=0), "
                "located numerically inside the observed temperature interval. All "
                "crossings are reported when there are several. This is NOT "
                "necessarily where the finite-chain Rg(T) curve changes fastest."
            ),
            "T_bias_zero_model_derived": (
                "The model registry's closed-form Tc where one exists (e.g. Tc for "
                "tc_scale, h/s for hs). It may lie outside the observed temperature "
                "range. Kept for backward compatibility with contact-mode outputs."
            ),
            "T_rg_max_slope": (
                "Temperature maximizing -dRg/dT of the fitted scalar Rg(T) curve on a "
                "dense grid spanning the observed temperature interval. This is the "
                "primary finite-chain transition descriptor."
            ),
            "T_rg_half": (
                "Temperature on the dense grid whose fitted Rg is nearest the midpoint "
                "between the fitted Rg at the low and high ends of the observed "
                "temperature interval."
            ),
            "rg_summary": (
                "mean: sum_j r_j P(r_j|T).  rms: sqrt(sum_j r_j^2 P(r_j|T)). "
                f"Selected: {args.rg_summary}."
            ),
            "objective": (
                f"{args.rg_mean_loss}, averaged over temperatures (mean, not sum) so "
                f"train/validation/all values are comparable. Residuals are computed in "
                f"{target_units} units."
            ),
            "range_weighted_note": (
                "range_weighted uses the lower/upper columns as asymmetric scale "
                "factors. It is NOT a chi-squared statistic unless those columns are "
                "genuine statistical uncertainties."
            ),
            "rg_scale": (
                "Observed Rg units per lattice unit: Rg_observed = rg_scale * "
                "Rg_lattice. With --rg-target-units lattice it is used only for "
                "reporting and never enters the loss."
            ),
        },
        "warnings": warnings_list,
    }
    if feasibility is not None:
        summary_json["feasibility_diagnostics"] = feasibility
    if args.model == "heat_capacity":
        summary_json["T0"] = float(Tref)

    with open(json_path, "w") as fh:
        json.dump(summary_json, fh, indent=2, cls=_NpEncoder, allow_nan=False)
    print(f"Saved: {json_path}")

    # ---- supplementary analyses --------------------------------------------
    ctx: Dict[str, Any] = {
        "cfg": cfg, "obj_fn": obj_fn, "temps": temps,
        "rg_target_input": rg_target_input, "rg_lower_input": rg_lower_input,
        "rg_upper_input": rg_upper_input, "param_names": param_names,
        "bounds": bounds, "x0s": x0s, "train_idx": train_idx, "val_idx": val_idx,
        "has_val": has_val, "params_fit": params_all, "outdir": plot_dir,
        "Tmin": Tmin, "Tmax": Tmax, "trans": trans, "spec": spec,
        "restart_records": restart_records, "target_units": target_units,
        "make_plots": make_plots,
    }
    if args.bootstrap > 0:
        run_rg_scalar_bootstrap(args, ctx)
    if args.uncertainty_diagnostics:
        run_rg_scalar_uncertainty_diagnostics(args, ctx)
    if args.split_sensitivity:
        run_rg_scalar_split_sensitivity(args, ctx)

    # ---- plots --------------------------------------------------------------
    if make_plots:
        _plot_rg_scalar(args, ctx, trans, rg_pred_target_units, rg_mod_mass,
                        rg_centers_lattice, b_T, all_metrics)

    # ---- final verdict ------------------------------------------------------
    print("\n--- Scientific status ---")
    if feasibility is None:
        print("  UNVERIFIED: run --rg-feasibility-scan to establish whether the "
              "baseline can reach the observed Rg range.")
    elif feasibility["target_within_reachable_range"]:
        print("  Feasibility scan: the joint baseline CAN reach the observed Rg "
              "range over the scanned bias window.")
    else:
        print("  NOT VALID: the joint baseline CANNOT reach part of the observed Rg "
              "range at ANY bias. The fitted parameters are a constrained best "
              "effort, not a physically meaningful description of those points.")
    for w in warnings_list:
        print(f"  WARNING: {w}")


# ---------------------------------------------------------------------------
# Scalar Rg(T) mode: bootstrap
# ---------------------------------------------------------------------------

def run_rg_scalar_bootstrap(args: argparse.Namespace, ctx: Dict[str, Any]) -> None:
    """Empirical temperature-resampling bootstrap for scalar-Rg mode.

    Resamples TRAINING temperature rows with replacement, keeping (temperature,
    central, lower, upper) paired, refits with the same objective, and scores each
    replicate on the ORIGINAL (non-resampled) train, validation, and full sets.

    This is an empirical resampling bootstrap over a small temperature ladder, NOT
    a likelihood-based formal uncertainty: the scalar-Rg objective is not a
    log-likelihood, and the replicates are not independent measurements.
    """
    cfg = ctx["cfg"]
    obj_fn = ctx["obj_fn"]
    temps = ctx["temps"]
    target = ctx["rg_target_input"]
    lower = ctx["rg_lower_input"]
    upper = ctx["rg_upper_input"]
    train_idx, val_idx = ctx["train_idx"], ctx["val_idx"]
    has_val = ctx["has_val"]
    param_names: List[str] = ctx["param_names"]
    bounds = ctx["bounds"]
    x0s = ctx["x0s"]
    outdir: Path = ctx["outdir"]
    n_boot = int(args.bootstrap)
    conf = float(args.bootstrap_confidence)
    seed = int(args.bootstrap_seed) if args.bootstrap_seed is not None else int(args.seed)
    rng = np.random.default_rng(seed)

    print(f"\n--- Bootstrap ({n_boot} replicates, temperature resampling) ---")
    print("  NOTE: empirical temperature-resampling bootstrap, NOT a likelihood-based "
          "formal uncertainty estimate.")

    dense = np.linspace(ctx["Tmin"], ctx["Tmax"], 201)
    rows: List[Dict[str, Any]] = []
    params_list: List[np.ndarray] = []
    pred_ladder: List[np.ndarray] = []
    pred_dense: List[np.ndarray] = []
    n_failed = 0

    for r in range(n_boot):
        pick = rng.integers(0, train_idx.size, size=train_idx.size)
        bidx = train_idx[pick]
        bargs = (temps[bidx], target[bidx], lower[bidx], upper[bidx])
        try:
            best_b, _ = fit_one_split(obj_fn, bargs, x0s, bounds)
        except RuntimeError:
            # The existing bootstrap convention: count failed replicates instead of
            # aborting the whole run.
            n_failed += 1
            continue
        pb = np.asarray(best_b.x, dtype=float)
        params_list.append(pb)

        tr = _rg_scalar_score(cfg, pb, temps[train_idx], target[train_idx],
                              lower[train_idx], upper[train_idx])
        va = _rg_scalar_score(cfg, pb, temps[val_idx], target[val_idx],
                              lower[val_idx], upper[val_idx])
        al = _rg_scalar_score(cfg, pb, temps, target, lower, upper)
        tm = _rg_scalar_transitions(cfg, pb, ctx["Tmin"], ctx["Tmax"])

        pred_ladder.append(_rg_scalar_predict(cfg, pb, temps)["pred_target_units"])
        pred_dense.append(_rg_scalar_predict(cfg, pb, dense)["pred_target_units"])

        rec: Dict[str, Any] = {"replicate": r}
        for n, v in zip(param_names, pb):
            rec[n] = float(v)
        rec.update({
            "train_objective": tr["objective_mean"],
            "validation_objective": va["objective_mean"],
            "all_objective": al["objective_mean"],
            "train_rmse": tr["rmse"], "validation_rmse": va["rmse"],
            "all_rmse": al["rmse"],
            "T_bias_zero": tm["T_bias_zero"],
            "T_rg_max_slope": tm["T_rg_max_slope"],
        })
        rows.append(rec)

    n_ok = len(rows)
    print(f"  successful replicates: {n_ok}/{n_boot}"
          + (f"  ({n_failed} failed)" if n_failed else ""))
    if n_ok == 0:
        print("  WARNING: no bootstrap replicate converged; skipping bootstrap outputs.")
        return

    param_matrix = np.array(params_list, dtype=float)
    fitted = np.asarray(ctx["params_fit"], dtype=float)

    stats: Dict[str, Any] = {}
    for j, n in enumerate(param_names):
        stats[n] = bootstrap_param_stats(param_matrix[:, j], float(fitted[j]), conf)
    for key, fitted_val in (
        ("T_bias_zero", ctx["trans"]["T_bias_zero"]),
        ("T_rg_max_slope", ctx["trans"]["T_rg_max_slope"]),
        ("train_rmse", None), ("validation_rmse", None), ("all_rmse", None),
        ("train_objective", None), ("validation_objective", None),
        ("all_objective", None),
    ):
        vals = np.array(
            [r[key] for r in rows if r.get(key) is not None], dtype=float
        )
        stats[key] = bootstrap_param_stats(vals, fitted_val, conf)

    boot_csv = outdir / "rg_scalar_bootstrap.csv"
    header = ["replicate"] + list(param_names) + [
        "train_objective", "validation_objective", "all_objective",
        "train_rmse", "validation_rmse", "all_rmse",
        "T_bias_zero", "T_rg_max_slope",
    ]
    with open(boot_csv, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for rec in rows:
            writer.writerow([
                ("" if rec.get(c) is None else rec.get(c)) for c in header
            ])
    print(f"Saved: {boot_csv}")

    # Prediction bands on the original ladder and a dense grid.
    alpha = 1.0 - conf
    PL = np.array(pred_ladder, dtype=float)
    PD = np.array(pred_dense, dtype=float)
    lo_l = np.percentile(PL, 100.0 * alpha / 2.0, axis=0)
    hi_l = np.percentile(PL, 100.0 * (1.0 - alpha / 2.0), axis=0)
    md_l = np.median(PL, axis=0)
    lo_d = np.percentile(PD, 100.0 * alpha / 2.0, axis=0)
    hi_d = np.percentile(PD, 100.0 * (1.0 - alpha / 2.0), axis=0)
    md_d = np.median(PD, axis=0)

    bands_csv = outdir / "rg_scalar_bootstrap_bands.csv"
    with open(bands_csv, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "temperature", "rg_pred_median", "rg_pred_lo", "rg_pred_hi",
            "rg_target", "rg_lower", "rg_upper", "units",
        ])
        for i in range(temps.size):
            writer.writerow([
                f"{temps[i]:.8g}", f"{md_l[i]:.8g}", f"{lo_l[i]:.8g}", f"{hi_l[i]:.8g}",
                f"{target[i]:.8g}", f"{lower[i]:.8g}", f"{upper[i]:.8g}",
                ctx["target_units"],
            ])
    print(f"Saved: {bands_csv}")

    boot_json = outdir / "rg_scalar_bootstrap_summary.json"
    with open(boot_json, "w") as fh:
        json.dump({
            "mode": "rg_scalar",
            "n_requested": n_boot,
            "n_successful": n_ok,
            "n_failed": n_failed,
            "confidence": conf,
            "seed": seed,
            "method": "temperature",
            "statistics": stats,
            "bound_fractions": param_bound_fractions(param_matrix, bounds, param_names),
            "correlation_flags": (
                correlation_flags(
                    np.corrcoef(param_matrix, rowvar=False), param_names,
                    float(args.bootstrap_correlation_threshold),
                ) if param_matrix.shape[0] > 1 and len(param_names) > 1 else []
            ),
            "warning": (
                "Empirical temperature-resampling bootstrap over a small temperature "
                "ladder. NOT a likelihood-based formal uncertainty: the scalar-Rg "
                "objective is not a log-likelihood."
            ),
        }, fh, indent=2, cls=_NpEncoder, allow_nan=False)
    print(f"Saved: {boot_json}")

    if args.bootstrap_save_prediction_bands:
        p = outdir / "rg_scalar_bootstrap_prediction_bands.npz"
        np.savez_compressed(
            p, temps=temps, dense_temps=dense,
            pred_ladder=PL, pred_dense=PD, params=param_matrix,
            param_names=np.array(param_names), units=str(ctx["target_units"]),
        )
        print(f"Saved: {p}")

    if ctx["make_plots"]:
        if plt is None:
            raise RuntimeError("matplotlib is required for plots; use --no-plots")
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.fill_between(dense, lo_d, hi_d, color="tab:blue", alpha=0.25,
                        label=f"{conf:.0%} bootstrap band")
        ax.plot(dense, md_d, "-", color="tab:blue", lw=1.5, label="bootstrap median")
        yerr = np.vstack([target - lower, upper - target])
        ax.errorbar(temps, target, yerr=yerr, fmt="o", ms=4, color="black",
                    ecolor="gray", elinewidth=1, capsize=2, label="target (bounds)")
        ax.set_xlabel("T (K)")
        ax.set_ylabel(f"Rg ({ctx['target_units']} units)")
        ax.set_title("Bootstrap prediction band for scalar Rg(T)\n"
                     "empirical temperature resampling, not formal uncertainty")
        ax.legend(fontsize=8)
        fig.tight_layout()
        p = outdir / "rg_scalar_bootstrap_band.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {p}")


# ---------------------------------------------------------------------------
# Scalar Rg(T) mode: local curvature / restart diagnostics
# ---------------------------------------------------------------------------

def run_rg_scalar_uncertainty_diagnostics(
    args: argparse.Namespace, ctx: Dict[str, Any]
) -> None:
    """Numerical Hessian of the scalar objective + restart stability.

    Inverse-Hessian quantities are LOCAL OBJECTIVE-CURVATURE diagnostics, not
    formal covariance estimates: the scalar-Rg regression loss is not a
    log-likelihood with a calibrated noise model.
    """
    cfg = ctx["cfg"]
    obj_fn = ctx["obj_fn"]
    temps, target = ctx["temps"], ctx["rg_target_input"]
    lower, upper = ctx["rg_lower_input"], ctx["rg_upper_input"]
    train_idx = ctx["train_idx"]
    param_names: List[str] = ctx["param_names"]
    outdir: Path = ctx["outdir"]
    params_fit = np.asarray(ctx["params_fit"], dtype=float)

    print("\n--- Uncertainty diagnostics (local curvature + restarts) ---")
    print("  NOTE: inverse-Hessian quantities are local objective-curvature "
          "diagnostics, NOT statistical standard errors.")

    train_args = (
        temps[train_idx], target[train_idx], lower[train_idx], upper[train_idx]
    )

    def f(x: np.ndarray) -> float:
        return float(obj_fn(x, *train_args))

    H = numerical_hessian(f, params_fit)
    diag = hessian_diagnostics(H)
    diag["note"] = (
        "Local objective-curvature diagnostics only. The scalar-Rg regression "
        "objective is NOT a log-likelihood; inverse-Hessian quantities are "
        "curvature length-scales, not statistical standard errors."
    )
    stab = restart_stability(ctx["restart_records"])

    restart_csv = outdir / "rg_scalar_restart_diagnostics.csv"
    with open(restart_csv, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["restart_index", "success", "objective", "n_iter"]
                        + list(param_names) + ["message"])
        for rec in ctx["restart_records"]:
            writer.writerow(
                [rec["restart_index"], rec["success"],
                 ("" if rec["objective"] is None else f"{rec['objective']:.8g}"),
                 ("" if rec["n_iter"] is None else rec["n_iter"])]
                + [f"{v:.8g}" for v in rec["params"]]
                + [rec["message"]]
            )
    print(f"Saved: {restart_csv}")

    diag_json = outdir / "rg_scalar_uncertainty_diagnostics.json"
    with open(diag_json, "w") as fh:
        json.dump({
            "mode": "rg_scalar",
            "param_names": list(param_names),
            "params": [float(v) for v in params_fit],
            "rg_scale_was_fitted": bool(cfg["fit_rg_scale"]),
            "hessian": [[float(v) for v in row] for row in H],
            "hessian_diagnostics": diag,
            "restart_stability": stab,
        }, fh, indent=2, cls=_NpEncoder, allow_nan=False)
    print(f"Saved: {diag_json}")
    print(f"  restarts: {stab['n_success']}/{stab['n_restarts']} succeeded, "
          f"distinct minima: {stab['distinct_minima']}")


# ---------------------------------------------------------------------------
# Scalar Rg(T) mode: split sensitivity
# ---------------------------------------------------------------------------

def run_rg_scalar_split_sensitivity(
    args: argparse.Namespace, ctx: Dict[str, Any]
) -> None:
    """Refit scalar-Rg mode under many validation splits (supplementary outputs).

    Reuses the same built-in schemes as contact mode (every_third_phase, kfold,
    blocked_low/mid/high, random) plus --split-config-json custom splits.
    """
    cfg = ctx["cfg"]
    obj_fn = ctx["obj_fn"]
    temps, target = ctx["temps"], ctx["rg_target_input"]
    lower, upper = ctx["rg_lower_input"], ctx["rg_upper_input"]
    param_names: List[str] = ctx["param_names"]
    bounds, x0s = ctx["bounds"], ctx["x0s"]
    outdir: Path = ctx["outdir"]
    n = temps.size

    schemes = [s.strip() for s in args.split_schemes.split(",") if s.strip()]
    split_seed = int(args.split_seed) if args.split_seed is not None else int(args.seed)
    splits = build_split_schemes(
        n, schemes,
        kfold_k=int(args.split_kfold_k),
        blocked_fraction=float(args.split_blocked_fraction),
        random_fraction=float(args.split_random_fraction),
        random_repeats=int(args.split_random_repeats),
        split_seed=split_seed,
    )
    if args.split_config_json is not None:
        splits.extend(load_split_config_json(args.split_config_json, n))

    print(f"\n--- Split sensitivity ({len(splits)} splits) ---")
    records: List[Dict[str, Any]] = []
    for sp in splits:
        tr_i, va_i = sp["train_idx"], sp["val_idx"]
        sargs = (temps[tr_i], target[tr_i], lower[tr_i], upper[tr_i])
        try:
            best_s, obj_s = fit_one_split(obj_fn, sargs, x0s, bounds)
            ok = bool(best_s.success)
            ps = np.asarray(best_s.x, dtype=float)
        except RuntimeError:
            print(f"  {sp['name']:24s} : all restarts failed; skipped")
            continue
        tr_m = _rg_scalar_score(cfg, ps, temps[tr_i], target[tr_i], lower[tr_i], upper[tr_i])
        va_m = _rg_scalar_score(cfg, ps, temps[va_i], target[va_i], lower[va_i], upper[va_i])
        al_m = _rg_scalar_score(cfg, ps, temps, target, lower, upper)
        tm = _rg_scalar_transitions(cfg, ps, ctx["Tmin"], ctx["Tmax"])
        hits = count_boundary_hits(ps, bounds, param_names)

        rec: Dict[str, Any] = {"scheme": sp["scheme"], "split": sp["name"]}
        for nm, v in zip(param_names, ps):
            rec[nm] = float(v)
        rec.update({
            "train_objective": tr_m["objective_mean"],
            "validation_objective": va_m["objective_mean"],
            "train_rmse": tr_m["rmse"], "validation_rmse": va_m["rmse"],
            "all_rmse": al_m["rmse"],
            "train_mae": tr_m["mae"], "validation_mae": va_m["mae"],
            "T_bias_zero": tm["T_bias_zero"],
            "T_rg_max_slope": tm["T_rg_max_slope"],
            "boundary_hits": ";".join(hits),
            "optimizer_success": ok,
            "train_indices": " ".join(str(int(i)) for i in tr_i),
            "validation_indices": " ".join(str(int(i)) for i in va_i),
        })
        records.append(rec)
        print(f"  {sp['name']:24s} : train_rmse={tr_m['rmse']:.5g}  "
              f"val_rmse={va_m['rmse']:.5g}  T_rg_max_slope={tm['T_rg_max_slope']:.5g}")

    if not records:
        print("  WARNING: no split produced a successful fit; skipping outputs.")
        return

    header = ["scheme", "split"] + list(param_names) + [
        "train_objective", "validation_objective", "train_rmse", "validation_rmse",
        "all_rmse", "train_mae", "validation_mae", "T_bias_zero", "T_rg_max_slope",
        "boundary_hits", "optimizer_success", "train_indices", "validation_indices",
    ]
    sens_csv = outdir / "rg_scalar_split_sensitivity.csv"
    with open(sens_csv, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for rec in records:
            writer.writerow([("" if rec.get(c) is None else rec.get(c)) for c in header])
    print(f"Saved: {sens_csv}")

    keys = list(param_names) + ["T_rg_max_slope", "validation_rmse", "train_rmse"]
    stab = summarize_param_stability(
        [{k: v for k, v in r.items() if isinstance(v, (int, float))} for r in records],
        keys,
    )
    sens_json = outdir / "rg_scalar_split_sensitivity_summary.json"
    with open(sens_json, "w") as fh:
        json.dump({
            "mode": "rg_scalar",
            "n_splits": len(records),
            "schemes": schemes,
            "split_seed": split_seed,
            "stability": stab,
            "note": (
                "Supplementary analysis. Spread across splits indicates how strongly "
                "the fitted parameters depend on which temperatures are used."
            ),
        }, fh, indent=2, cls=_NpEncoder, allow_nan=False)
    print(f"Saved: {sens_json}")


# ---------------------------------------------------------------------------
# Scalar Rg(T) mode: plots
# ---------------------------------------------------------------------------

def _plot_rg_scalar(
    args: argparse.Namespace,
    ctx: Dict[str, Any],
    trans: Dict[str, Any],
    rg_pred_target_units: np.ndarray,
    rg_mod_mass: np.ndarray,
    rg_centers_lattice: np.ndarray,
    b_T: np.ndarray,
    all_metrics: Dict[str, Optional[float]],
) -> None:
    """Scalar-mode plots: fit, residuals, b(T), and predicted P(Rg|T) diagnostics."""
    if plt is None:
        raise RuntimeError("matplotlib is required for plots; install it or use --no-plots")
    outdir: Path = ctx["outdir"]
    temps = ctx["temps"]
    target = ctx["rg_target_input"]
    lower, upper = ctx["rg_lower_input"], ctx["rg_upper_input"]
    train_idx, val_idx = ctx["train_idx"], ctx["val_idx"]
    has_val = ctx["has_val"]
    units = ctx["target_units"]
    figs: List[Any] = []

    # --- 1. main fit ---
    fig, ax = plt.subplots(figsize=(7, 4.5))
    yerr = np.vstack([target - lower, upper - target])
    ax.errorbar(
        temps[train_idx], target[train_idx],
        yerr=yerr[:, train_idx], fmt="o", ms=5, color="black", ecolor="gray",
        elinewidth=1, capsize=2, label="target, train (lower/upper bounds)",
    )
    if has_val:
        ax.errorbar(
            temps[val_idx], target[val_idx],
            yerr=yerr[:, val_idx], fmt="s", ms=6, mfc="none", color="tab:red",
            ecolor="tab:red", elinewidth=1, capsize=2,
            label="target, validation (lower/upper bounds)",
        )
    ax.plot(trans["grid"], trans["curve"], "-", color="tab:blue", lw=1.8,
            label=f"fitted Rg(T) [{args.model}]")
    ax.axvline(trans["T_rg_max_slope"], color="tab:green", lw=1.2, ls="--",
               label=f"T_rg_max_slope = {trans['T_rg_max_slope']:.4g} K")
    ax.set_xlabel("T (K)")
    ax.set_ylabel(f"Rg ({units} units)")
    rmse = all_metrics["rmse"]
    ax.set_title(
        f"Scalar Rg(T) fit — model {args.model}, summary {args.rg_summary}, "
        f"loss {args.rg_mean_loss}\n"
        f"all-point RMSE = {rmse:.4g} {units} units   "
        f"(error bars are supplied bounds, NOT standard errors)"
    )
    ax.legend(fontsize=7)
    fig.tight_layout()
    p = outdir / "rg_scalar_fit.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    print(f"Saved: {p}")
    figs.append(fig)

    # --- 2. residuals ---
    resid = rg_pred_target_units - target
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axhline(0.0, color="gray", lw=1.0, ls="--")
    ax.plot(temps[train_idx], resid[train_idx], "o", ms=5, color="black", label="train")
    if has_val:
        ax.plot(temps[val_idx], resid[val_idx], "s", ms=6, mfc="none",
                color="tab:red", label="validation")
    ax.set_xlabel("T (K)")
    ax.set_ylabel(f"residual: fit − target ({units} units)")
    ax.set_title("Scalar Rg(T) residuals")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = outdir / "rg_scalar_residuals.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    print(f"Saved: {p}")
    figs.append(fig)

    # --- 3. b(T) ---
    fig, ax = plt.subplots(figsize=(7, 4))
    dense = trans["grid"]
    model_params, _ = split_rg_scalar_params(
        ctx["params_fit"], ctx["cfg"]["n_model_params"], ctx["cfg"]["fit_rg_scale"],
        ctx["cfg"]["rg_scale"],
    )
    b_dense = np.array([ctx["cfg"]["b_fn"](model_params, float(T)) for T in dense])
    ax.plot(dense, b_dense, "k-", lw=1.8)
    ax.axhline(0.0, color="gray", lw=0.8, ls="--")
    ax.plot(temps, b_T, "o", ms=4, color="tab:blue", label="data temperatures")
    for k, tz in enumerate(trans["bias_zero_crossings"]):
        ax.axvline(tz, color="tab:orange", lw=1.2, ls=":",
                   label=(f"T_bias_zero = {tz:.4g} K" if k == 0 else None))
    ax.set_xlabel("T (K)")
    ax.set_ylabel("b(T)")
    ax.set_title(f"Reduced contact bias  [{ctx['spec']['description']}]")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = outdir / "rg_scalar_bT.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    print(f"Saved: {p}")
    figs.append(fig)

    # --- 4. predicted P(Rg|T) at a deterministic set of temperatures ---
    pick = sorted({
        0,
        int(np.argmin(np.abs(temps - trans["T_rg_max_slope"]))),
        temps.size - 1,
    })
    rg_axis = (
        rg_centers_lattice if units == "lattice"
        else ctx["cfg"]["rg_scale"] * rg_centers_lattice
    )
    if ctx["cfg"]["fit_rg_scale"] and units == "observed":
        _, fitted_scale = split_rg_scalar_params(
            ctx["params_fit"], ctx["cfg"]["n_model_params"], True, ctx["cfg"]["rg_scale"]
        )
        rg_axis = fitted_scale * rg_centers_lattice
    fig, ax = plt.subplots(figsize=(7, 4.5))
    labels = {0: "lowest T", temps.size - 1: "highest T"}
    for i in pick:
        lab = labels.get(i, "near T_rg_max_slope")
        ax.plot(rg_axis, rg_mod_mass[i], "-", lw=1.6,
                label=f"T = {temps[i]:.4g} K ({lab})")
    ax.set_xlabel(f"Rg ({units} units)")
    ax.set_ylabel("P(Rg | T)")
    ax.set_title(
        "PREDICTED Rg distributions from the fitted model (diagnostic only)\n"
        "these are model predictions, NOT observed data"
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = outdir / "rg_scalar_predicted_distributions.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    print(f"Saved: {p}")
    figs.append(fig)

    if args.show_plots:
        plt.show()
    else:
        for f in figs:
            plt.close(f)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Fit lattice polymer contact model to REMD histograms.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # data
    ap.add_argument(
        "--remd", type=str, default="remd_distributions_44mer.npz",
        help="NPZ with temps, ct_centers, ct_hists (pdf). Optionally rg_centers, rg_hists.",
    )
    ap.add_argument(
        "--baseline", type=str,
        default="dists_44mer_long/single_uniform_chain2_athermal_dists_joint_N44_T1_seed42.npz",
        help="Baseline NPZ. For Rg prediction, must contain c_edges, rg_edges, crg_prob.",
    )
    ap.add_argument(
        "--contact_offset", type=float, default=43,
        help="Constant subtracted from ct_centers in the REMD file before binning.",
    )
    # output
    ap.add_argument(
        "--outdir", type=str, default=None,
        help=(
            "Output directory for all generated files (NPZ, JSON, CSV, plots). "
            "Created if it does not exist. When given, NPZ is saved as fit_results.npz."
        ),
    )
    ap.add_argument(
        "--out", type=str, default="fit_lattice_contact_model.npz",
        help="Output NPZ path (used when --outdir is not given).",
    )
    ap.add_argument(
        "--no-plots", "--no_plots", action="store_true", dest="no_plots",
        help="Skip all plot generation.",
    )
    ap.add_argument(
        "--show-plots", action="store_true", dest="show_plots",
        help="Show plots interactively in addition to saving them.",
    )

    # model selection
    ap.add_argument(
        "--model", type=str, default="hs",
        choices=list(MODEL_REGISTRY.keys()),
        help="Temperature-dependence model for b(T).",
    )
    ap.add_argument(
        "--Tref", type=float, default=None,
        help=(
            "Reference temperature for x(T) = (T-Tref)/Tscale. "
            "Default: midpoint of temperature range."
        ),
    )
    ap.add_argument(
        "--Tscale", type=float, default=None,
        help="Scale for x(T). Default: full temperature range (Tmax - Tmin).",
    )
    ap.add_argument(
        "--T0", type=float, default=None,
        help=(
            "Reference temperature T0 for the heat_capacity model. "
            "Affects dg(T) = dh0 - T*ds0 + dCp*((T-T0) - T*ln(T/T0)). "
            "Defaults to midpoint of temperature range (same default as --Tref). "
            "Ignored for other models."
        ),
    )

    # loss function
    ap.add_argument(
        "--loss", type=str, default="kl", choices=["kl", "js"],
        help="Divergence used as the fitting objective.",
    )

    # Rg fitting (mutually exclusive; default off so suite behavior is explicit)
    rg_group = ap.add_mutually_exclusive_group()
    rg_group.add_argument(
        "--fit-rg",
        action="store_true",
        dest="fit_rg",
        help="Include Rg loss in the optimization objective.",
    )
    rg_group.add_argument(
        "--no-fit-rg",
        "--no_fit_rg",
        action="store_false",
        dest="fit_rg",
        help="Use the contact-distribution objective only.",
    )
    ap.set_defaults(fit_rg=False)
    ap.add_argument(
        "--rg-weight", type=float, default=1.0, dest="rg_weight",
        help="Weight of Rg loss relative to contact loss when --fit-rg is active.",
    )
    ap.add_argument(
        "--rg-scale",
        type=float,
        default=0.46320503312590167,
        dest="rg_scale",
        help=(
            "Scale factor converting lattice Rg units into observed/molecular Rg units: "
            "Rg_observed_units = rg_scale * Rg_lattice."
        ),
    )

    # held-out validation
    ap.add_argument(
        "--holdout-every", type=int, default=None, metavar="N",
        help="Hold out every Nth temperature index (0-indexed: 0, N, 2N, …).",
    )
    ap.add_argument(
        "--holdout-indices", type=str, default=None, metavar="I,J,...",
        help="Comma-separated temperature indices to use as validation.",
    )
    ap.add_argument(
        "--train-indices", type=str, default=None, metavar="I,J,...",
        help="Comma-separated temperature indices to use for training (rest = validation).",
    )

    # optimization
    ap.add_argument("--n_restarts", type=int, default=8)
    ap.add_argument("--seed", type=int, default=123)

    # bootstrap
    ap.add_argument(
        "--bootstrap", type=int, default=0, metavar="N",
        help="Number of bootstrap replicates over training temperatures (0 = skip).",
    )
    ap.add_argument(
        "--bootstrap-seed", type=int, default=None, dest="bootstrap_seed",
        help="RNG seed for bootstrap resampling (defaults to --seed).",
    )
    ap.add_argument(
        "--bootstrap-method", type=str, default="temperature", dest="bootstrap_method",
        choices=["temperature"],
        help="Bootstrap resampling method. Currently only 'temperature' (empirical "
             "resampling of temperatures with replacement) is supported.",
    )
    ap.add_argument(
        "--bootstrap-confidence", type=float, default=0.95, dest="bootstrap_confidence",
        help="Confidence level for bootstrap percentile intervals (0 < c < 1).",
    )
    ap.add_argument(
        "--bootstrap-correlation-threshold", type=float, default=0.9,
        dest="bootstrap_correlation_threshold",
        help="|correlation| at or above this flags a parameter pair as possibly "
             "non-identifiable.",
    )
    ap.add_argument(
        "--bootstrap-save-prediction-bands", action="store_true",
        dest="bootstrap_save_prediction_bands",
        help="Also save full per-replicate prediction arrays to "
             "bootstrap_prediction_bands.npz.",
    )
    ap.add_argument(
        "--uncertainty-diagnostics", action="store_true", dest="uncertainty_diagnostics",
        help="Compute local objective-curvature (numerical Hessian) and optimizer "
             "restart-stability diagnostics (NOT statistical standard errors).",
    )

    # Rg-weight sensitivity / Pareto analysis (optional, supplementary)
    ap.add_argument(
        "--rg-weight-grid", type=str, default=None, dest="rg_weight_grid",
        help="Comma-separated Rg-loss weights to sweep, e.g. '0,0.1,0.25,0.5,1,2,4'. "
             "Enables the supplementary Rg-weight sensitivity/Pareto analysis. "
             "Weight 0 is a true contact-only fit (still scored on Rg).",
    )
    ap.add_argument(
        "--rg-weight-grid-file", type=str, default=None, dest="rg_weight_grid_file",
        help="File of Rg-loss weights (comma/space/newline separated); combined with "
             "--rg-weight-grid if both are given.",
    )
    ap.add_argument(
        "--rg-weight-normalization-diagnostics", action="store_true",
        dest="rg_weight_normalization_diagnostics",
        help="Report the relative numerical scale of contact vs Rg losses at each "
             "weight (does NOT rescale the production objective).",
    )

    # validation-split sensitivity (optional, supplementary outputs only)
    ap.add_argument(
        "--split-sensitivity", action="store_true", dest="split_sensitivity",
        help="After the primary fit, refit under many validation splits and write "
             "supplementary split_sensitivity.* files (does not change primary outputs).",
    )
    ap.add_argument(
        "--split-schemes", type=str, dest="split_schemes",
        default="every_third_phase,kfold,blocked_low,blocked_mid,blocked_high,random",
        help="Comma-separated built-in split schemes: every_third_phase, kfold, "
             "blocked_low, blocked_mid, blocked_high, random.",
    )
    ap.add_argument(
        "--split-config-json", type=str, default=None, dest="split_config_json",
        help="JSON file of explicit user-defined splits (list of objects with "
             "'name' and 'train_indices' or 'holdout_indices').",
    )
    ap.add_argument(
        "--split-seed", type=int, default=None, dest="split_seed",
        help="RNG seed for the random holdout scheme (defaults to --seed).",
    )
    ap.add_argument(
        "--split-kfold-k", type=int, default=5, dest="split_kfold_k",
        help="K for interleaved K-fold cross-validation.",
    )
    ap.add_argument(
        "--split-blocked-fraction", type=float, default=0.2, dest="split_blocked_fraction",
        help="Fraction of temperatures held out by the blocked schemes.",
    )
    ap.add_argument(
        "--split-random-fraction", type=float, default=0.2, dest="split_random_fraction",
        help="Fraction of temperatures held out per repeated-random split.",
    )
    ap.add_argument(
        "--split-random-repeats", type=int, default=5, dest="split_random_repeats",
        help="Number of repeated-random holdout splits.",
    )

    # ---- scalar Rg(T) mode -------------------------------------------------
    ap.add_argument(
        "--rg-means-file", type=str, default=None, dest="rg_means_file",
        metavar="PATH",
        help="Activate scalar-Rg(T) fitting mode. Whitespace-separated file with 4 "
             "columns: temperature, central Rg, lower Rg, upper Rg. Requires a joint "
             "baseline; --remd is not used.",
    )
    ap.add_argument(
        "--rg-target-units", type=str, default="observed", dest="rg_target_units",
        choices=list(RG_TARGET_UNITS_CHOICES),
        help="Units of the scalar Rg file. 'observed': same units as "
             "rg_scale*Rg_lattice (e.g. nm). 'lattice': already in lattice units "
             "(rg_scale then never enters the loss).",
    )
    ap.add_argument(
        "--rg-summary", type=str, default="rms", dest="rg_summary",
        choices=list(RG_SUMMARY_CHOICES),
        help="How the scalar Rg is computed from the predicted P(Rg|T). "
             "'rms' = sqrt(sum r^2 P(r)); 'mean' = sum r P(r).",
    )
    ap.add_argument(
        "--rg-mean-loss", type=str, default="mse", dest="rg_mean_loss",
        choices=list(RG_MEAN_LOSS_CHOICES),
        help="Scalar-Rg regression loss (does NOT use --loss, which is a "
             "distribution divergence).",
    )
    ap.add_argument(
        "--rg-huber-delta", type=float, default=0.05, dest="rg_huber_delta",
        help="Huber delta, in TARGET output units (e.g. nm when "
             "--rg-target-units observed).",
    )
    ap.add_argument(
        "--rg-range-floor", type=float, default=0.01, dest="rg_range_floor",
        help="Minimum asymmetric scale for range_weighted loss, in target units. "
             "Protects against zero-width lower/upper intervals.",
    )
    ap.add_argument(
        "--fit-rg-scale", action="store_true", dest="fit_rg_scale",
        help="DIAGNOSTIC: optimize rg_scale as a free parameter instead of fixing it "
             "at --rg-scale. A mapping-sensitivity check, not the primary analysis.",
    )
    ap.add_argument(
        "--rg-scale-min", type=float, default=0.25, dest="rg_scale_min",
        help="Lower bound on rg_scale when --fit-rg-scale is active.",
    )
    ap.add_argument(
        "--rg-scale-max", type=float, default=0.55, dest="rg_scale_max",
        help="Upper bound on rg_scale when --fit-rg-scale is active.",
    )
    ap.add_argument(
        "--rg-feasibility-scan", action="store_true", dest="rg_feasibility_scan",
        help="Scan contact bias b directly and report which scalar Rg values the "
             "joint baseline can reach. Diagnostic only; never affects the fit.",
    )
    ap.add_argument(
        "--rg-bias-min", type=float, default=-10.0, dest="rg_bias_min",
        help="Lowest contact bias b in the feasibility scan.",
    )
    ap.add_argument(
        "--rg-bias-max", type=float, default=10.0, dest="rg_bias_max",
        help="Highest contact bias b in the feasibility scan.",
    )
    ap.add_argument(
        "--rg-bias-points", type=int, default=401, dest="rg_bias_points",
        help="Number of bias grid points in the feasibility scan.",
    )

    ap.add_argument(
        "--quick-test", action="store_true", dest="quick_test",
        help="Run synthetic split-sensitivity/determinism unit tests and exit.",
    )
    args = ap.parse_args()

    if args.quick_test:
        raise SystemExit(run_quick_test())

    if not np.isfinite(args.rg_scale) or args.rg_scale <= 0:
        raise ValueError("--rg-scale must be finite and positive")
    if args.n_restarts < 1:
        raise ValueError("--n_restarts must be >= 1")
    if args.bootstrap < 0:
        raise ValueError("--bootstrap must be >= 0")
    if not (0.0 < args.bootstrap_confidence < 1.0):
        raise ValueError("--bootstrap-confidence must be in (0, 1)")
    if not (0.0 < args.bootstrap_correlation_threshold <= 1.0):
        raise ValueError("--bootstrap-correlation-threshold must be in (0, 1]")

    # Scalar-Rg(T) mode replaces the contact-histogram pipeline entirely: it needs
    # no REMD NPZ, no ct_centers/ct_hists, and no --contact_offset.  Dispatch before
    # any REMD loading so --remd's default path is never touched.
    if args.rg_means_file is not None:
        run_rg_scalar_mode(args)
        return

    if not np.isfinite(args.contact_offset):
        raise ValueError("--contact_offset must be finite")
    if not np.isfinite(args.rg_weight) or args.rg_weight < 0:
        raise ValueError("--rg-weight must be finite and >= 0")

    # -----------------------------------------------------------------------
    # Output directory setup
    # -----------------------------------------------------------------------
    if args.outdir is not None:
        outdir = Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        npz_path = outdir / "fit_results.npz"
        loss_csv_path = outdir / "train_validation_loss.csv"
        json_path = outdir / "fit_summary.json"
        params_csv_path = outdir / "fit_params.csv"
        plot_dir = outdir
    else:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        npz_path = out_path
        loss_csv_path = out_path.parent / "train_validation_loss.csv"
        json_path = out_path.parent / "fit_summary.json"
        params_csv_path = out_path.parent / "fit_params.csv"
        plot_dir = out_path.parent

    # -----------------------------------------------------------------------
    # Load raw inputs
    # -----------------------------------------------------------------------
    d = np.load(args.remd)
    temp_key = "temps" if "temps" in d.files else "Ts" if "Ts" in d.files else None
    if temp_key is None:
        raise ValueError(
            f"REMD file {args.remd!r} contains neither 'temps' nor 'Ts' "
            f"(found: {list(d.files)})"
        )
    temps = np.asarray(d[temp_key], dtype=float)
    ct_centers_raw = np.asarray(d["ct_centers"], dtype=float)        # before offset
    ct_centers_native = ct_centers_raw - float(args.contact_offset)  # after offset
    ct_pdf = np.asarray(d["ct_hists"], dtype=float)

    if temps.ndim != 1:
        raise ValueError(f"temps must be 1D, got shape {temps.shape}")
    if ct_centers_raw.ndim != 1 or ct_centers_raw.size < 2:
        raise ValueError(
            f"ct_centers must be 1D with at least 2 entries, got shape "
            f"{ct_centers_raw.shape}"
        )
    if ct_pdf.ndim != 2:
        raise ValueError(
            f"ct_hists must be 2D with shape (n_temps, n_contact_bins), got shape {ct_pdf.shape}"
        )
    if ct_pdf.shape[0] != len(temps):
        raise ValueError(
            f"ct_hists first dimension must match temps: "
            f"ct_hists.shape[0]={ct_pdf.shape[0]}, len(temps)={len(temps)}"
        )
    if ct_pdf.shape[1] != len(ct_centers_raw):
        raise ValueError(
            f"ct_hists second dimension must match ct_centers: "
            f"ct_hists.shape[1]={ct_pdf.shape[1]}, len(ct_centers)={len(ct_centers_raw)}"
        )
    if not np.all(np.isfinite(temps)):
        raise ValueError("temps contains non-finite values")
    if not np.all(temps > 0.0):
        raise ValueError("temps must contain only positive temperatures")
    if not np.all(np.diff(temps) > 0.0):
        raise ValueError("temps must be strictly increasing with no duplicates")
    if not np.all(np.isfinite(ct_centers_raw)):
        raise ValueError("ct_centers contains non-finite values")
    if not np.all(np.diff(ct_centers_raw) > 0.0):
        raise ValueError("ct_centers must be strictly increasing")
    ct_steps = np.diff(ct_centers_raw)
    if not np.allclose(ct_steps, ct_steps[0], rtol=1e-6, atol=1e-10):
        raise ValueError(
            "ct_centers must be evenly spaced because the fitter interprets "
            "ct_hists as piecewise-constant densities on that grid"
        )
    if not np.all(np.isfinite(ct_pdf)):
        raise ValueError("ct_hists contains non-finite values")
    if np.any(ct_pdf < 0.0):
        raise ValueError("ct_hists contains negative values")
    if np.any(ct_pdf.sum(axis=1) <= 0.0):
        raise ValueError("every ct_hists row must contain positive total density")

    # Detect observed Rg (try both capitalizations)
    _rg_centers_key = next(
        (k for k in ("rg_centers", "Rg_centers") if k in d.files), None
    )
    _rg_hists_key = next(
        (k for k in ("rg_hists", "Rg_hists") if k in d.files), None
    )
    has_obs_rg = _rg_centers_key is not None and _rg_hists_key is not None

    b_data = np.load(args.baseline)
    has_joint_baseline = all(k in b_data.files for k in ("c_edges", "rg_edges", "crg_prob"))

    # -----------------------------------------------------------------------
    # Input validation
    # -----------------------------------------------------------------------
    native_edges = centers_to_edges(ct_centers_native)
    m_min = int(np.floor(native_edges[0] + 0.5))
    m_max = int(np.ceil(native_edges[-1] - 0.5))
    remd_n = m_max - m_min + 1

    bl_min, bl_max = _get_baseline_integer_range(args.baseline)
    overlap_lo = max(m_min, bl_min)
    overlap_hi = min(m_max, bl_max)
    overlap_n = max(0, overlap_hi - overlap_lo + 1)
    overlap_pct = 100.0 * overlap_n / remd_n if remd_n > 0 else 0.0

    print("--- Input validation ---")
    print(f"REMD NPZ:    {args.remd}")
    print(f"  keys:        {list(d.files)}")
    print(f"  temps:       shape {temps.shape}")
    print(f"  ct_centers:  shape {ct_centers_raw.shape}"
          f"   range [{ct_centers_raw.min():.4g}, {ct_centers_raw.max():.4g}]")
    print(f"  ct_hists:    shape {ct_pdf.shape}")

    if has_obs_rg:
        _rg_c = np.asarray(d[_rg_centers_key], dtype=float)
        _rg_h = np.asarray(d[_rg_hists_key], dtype=float)
        print(f"  {_rg_centers_key}:  shape {_rg_c.shape}"
              f"   range [{_rg_c.min():.4g}, {_rg_c.max():.4g}]")
        print(f"  {_rg_hists_key}:    shape {_rg_h.shape}")
    else:
        _tried = "rg_centers / Rg_centers"
        print(f"  Observed Rg histograms: NOT FOUND (tried {_tried})")

    print(f"Baseline NPZ: {args.baseline}")
    print(f"  keys:        {list(b_data.files)}")
    print(f"  Joint P0(m,Rg): {'available (c_edges, rg_edges, crg_prob)' if has_joint_baseline else 'NOT FOUND'}")

    print(f"Contact offset: {args.contact_offset}")
    print(f"  Native range  (before offset):  [{ct_centers_raw.min():.4g}, {ct_centers_raw.max():.4g}]")
    print(f"  Shifted range (after offset):   [{ct_centers_native.min():.4g}, {ct_centers_native.max():.4g}]")
    print(f"  Integer m range (REMD):         [{m_min}, {m_max}]  ({remd_n} bins)")
    print(f"  Integer m range (baseline):     [{bl_min}, {bl_max}]")
    print(f"  Contact support overlap:        {overlap_n}/{remd_n} = {overlap_pct:.1f}%")

    if overlap_n == 0:
        raise ValueError(
            f"Zero contact support overlap: REMD shifted [{m_min}, {m_max}] vs "
            f"baseline [{bl_min}, {bl_max}]. Check --contact_offset."
        )
    if overlap_pct < 50.0:
        print(
            f"WARNING: contact support overlap is {overlap_pct:.1f}% (range coverage). "
            f"If most REMD mass falls in [{bl_min}, {bl_max}], fitting may still be valid."
        )

    # Rg grid overlap detection
    rg_grid_overlap_pct: Optional[float] = None
    if has_obs_rg and has_joint_baseline:
        _rg_c_arr = np.asarray(d[_rg_centers_key], dtype=float)
        # Baseline rg_edges are in lattice units; scale to observed/comparison units.
        _rg_edges_model_lattice = np.asarray(b_data["rg_edges"], dtype=float)
        _rg_edges_model = args.rg_scale * _rg_edges_model_lattice
        rg_obs_min = float(_rg_c_arr.min())
        rg_obs_max = float(_rg_c_arr.max())
        rg_model_min = float(_rg_edges_model.min())
        rg_model_max = float(_rg_edges_model.max())
        rg_ov_lo = max(rg_obs_min, rg_model_min)
        rg_ov_hi = min(rg_obs_max, rg_model_max)
        obs_range = rg_obs_max - rg_obs_min
        rg_grid_overlap_pct = (
            100.0 * max(0.0, rg_ov_hi - rg_ov_lo) / obs_range
            if obs_range > 0 else 0.0
        )
        print(f"  Rg scale:       {args.rg_scale:.8g} observed units per lattice unit")
        print(f"  Rg obs range:   [{rg_obs_min:.4g}, {rg_obs_max:.4g}]")
        print(f"  Rg model range (lattice): [{float(_rg_edges_model_lattice.min()):.4g}, {float(_rg_edges_model_lattice.max()):.4g}]")
        print(f"  Rg model range (scaled):  [{rg_model_min:.4g}, {rg_model_max:.4g}]")
        print(f"  Rg grid overlap: {rg_grid_overlap_pct:.1f}% of obs Rg range")
        if rg_grid_overlap_pct < 50.0:
            print(
                f"WARNING: Rg grid overlap is {rg_grid_overlap_pct:.1f}%. "
                f"The baseline Rg range does not cover most of the observed Rg range. "
                f"Rg scoring will only reflect the overlap region. "
                f"Rg fitting (--fit-rg) may not be meaningful."
            )
    print()

    # -----------------------------------------------------------------------
    # Load and process observed Rg (if available)
    # -----------------------------------------------------------------------
    rg_centers_obs: Optional[np.ndarray] = None
    rg_hists_obs: Optional[np.ndarray] = None
    p_obs_rg_native: Optional[np.ndarray] = None      # mass on obs grid
    p_obs_rg_model_grid: Optional[np.ndarray] = None  # mass rebinned to model grid
    rg_centers_model: Optional[np.ndarray] = None        # scaled (comparison units)
    rg_edges_model: Optional[np.ndarray] = None          # scaled (comparison units)
    rg_centers_model_lattice: Optional[np.ndarray] = None  # raw lattice units
    rg_edges_model_lattice: Optional[np.ndarray] = None    # raw lattice units
    crg_prob: Optional[np.ndarray] = None
    c_edges_joint: Optional[np.ndarray] = None

    if has_obs_rg:
        rg_centers_obs = np.asarray(d[_rg_centers_key], dtype=float)
        rg_hists_obs = np.asarray(d[_rg_hists_key], dtype=float)

        if rg_centers_obs.ndim != 1 or rg_centers_obs.size < 2:
            raise ValueError(
                f"{_rg_centers_key} must be 1D with at least 2 entries, "
                f"got shape {rg_centers_obs.shape}"
            )
        if rg_hists_obs.ndim != 2:
            raise ValueError(
                f"{_rg_hists_key} must be 2D with shape (n_temps, n_rg_bins), "
                f"got shape {rg_hists_obs.shape}"
            )
        if rg_hists_obs.shape[0] != len(temps):
            raise ValueError(
                f"{_rg_hists_key} first dimension must match temps: "
                f"{_rg_hists_key}.shape[0]={rg_hists_obs.shape[0]}, len(temps)={len(temps)}"
            )
        if rg_hists_obs.shape[1] != len(rg_centers_obs):
            raise ValueError(
                f"{_rg_hists_key} second dimension must match {_rg_centers_key}: "
                f"{_rg_hists_key}.shape[1]={rg_hists_obs.shape[1]}, "
                f"len({_rg_centers_key})={len(rg_centers_obs)}"
            )
        if not np.all(np.isfinite(rg_centers_obs)):
            raise ValueError(f"{_rg_centers_key} contains non-finite values")
        if not np.all(np.diff(rg_centers_obs) > 0.0):
            raise ValueError(f"{_rg_centers_key} must be strictly increasing")
        rg_steps = np.diff(rg_centers_obs)
        if not np.allclose(rg_steps, rg_steps[0], rtol=1e-6, atol=1e-10):
            raise ValueError(
                f"{_rg_centers_key} must be evenly spaced because observed Rg "
                "histograms are interpreted as densities"
            )
        if not np.all(np.isfinite(rg_hists_obs)):
            raise ValueError(f"{_rg_hists_key} contains non-finite values")
        if np.any(rg_hists_obs < 0.0):
            raise ValueError(f"{_rg_hists_key} contains negative values")
        if np.any(rg_hists_obs.sum(axis=1) <= 0.0):
            raise ValueError(f"every {_rg_hists_key} row must contain positive density")

        # Convert each obs PDF to probability mass on native grid
        p_obs_rg_native = np.array(
            [pdf_to_mass(rg_hists_obs[i], rg_centers_obs)[0] for i in range(len(temps))]
        )

    if has_joint_baseline:
        crg_prob = np.asarray(b_data["crg_prob"], dtype=float)
        c_edges_joint = np.asarray(b_data["c_edges"], dtype=float)
        # Raw lattice-unit Rg grid (crg_prob is indexed on these lattice bins).
        rg_edges_model_lattice = np.asarray(b_data["rg_edges"], dtype=float)
        rg_centers_model_lattice = 0.5 * (
            rg_edges_model_lattice[:-1] + rg_edges_model_lattice[1:]
        )
        # Comparison/output Rg grid in observed/molecular units.  Scaling the
        # axis does not change the probability mass in each bin.
        rg_edges_model = args.rg_scale * rg_edges_model_lattice
        rg_centers_model = args.rg_scale * rg_centers_model_lattice

        if c_edges_joint.ndim != 1:
            raise ValueError(f"baseline c_edges must be 1D, got shape {c_edges_joint.shape}")
        if rg_edges_model.ndim != 1:
            raise ValueError(f"baseline rg_edges must be 1D, got shape {rg_edges_model.shape}")
        if crg_prob.ndim != 2:
            raise ValueError(f"baseline crg_prob must be 2D, got shape {crg_prob.shape}")
        expected_crg_shape = (len(c_edges_joint) - 1, len(rg_edges_model) - 1)
        if crg_prob.shape != expected_crg_shape:
            raise ValueError(
                f"baseline crg_prob shape must be (len(c_edges)-1, len(rg_edges)-1): "
                f"got {crg_prob.shape}, expected {expected_crg_shape}"
            )
        if not np.all(np.isfinite(c_edges_joint)):
            raise ValueError("baseline c_edges contains non-finite values")
        if not np.all(np.isfinite(rg_edges_model)):
            raise ValueError("baseline rg_edges contains non-finite values")
        if not np.all(np.isfinite(crg_prob)):
            raise ValueError("baseline crg_prob contains non-finite values")
        if np.any(crg_prob < 0.0):
            raise ValueError("baseline crg_prob contains negative probability mass")
        if crg_prob.sum() <= 0.0:
            raise ValueError("baseline crg_prob must contain positive total mass")
        if np.any(np.diff(c_edges_joint) <= 0):
            raise ValueError("baseline c_edges must be strictly increasing")
        if np.any(np.diff(rg_edges_model) <= 0):
            raise ValueError("baseline rg_edges must be strictly increasing")
        c_widths = np.diff(c_edges_joint)
        if not np.allclose(c_widths, 1.0, rtol=1e-3, atol=1e-6):
            print(
                "WARNING: baseline c_edges do not appear to have unit-width contact bins. "
                f"Contact bin widths range from {c_widths.min():.6g} to {c_widths.max():.6g}. "
                "This may affect mapping the joint baseline P0(m,Rg) onto integer contact bins."
            )

    if has_obs_rg and has_joint_baseline:
        # Rebin observed Rg PDFs onto the model Rg grid for loss computation
        p_obs_rg_model_grid = np.array([
            rebin_pdf_to_mass(rg_centers_obs, rg_hists_obs[i], rg_edges_model)
            for i in range(len(temps))
        ])

    # Validate --fit-rg feasibility
    can_fit_rg = has_obs_rg and has_joint_baseline
    if args.fit_rg and not can_fit_rg:
        missing = []
        if not has_joint_baseline:
            missing.append("joint baseline (c_edges, rg_edges, crg_prob)")
        if not has_obs_rg:
            missing.append("observed Rg histograms (rg_centers, rg_hists)")
        raise ValueError(
            f"--fit-rg requested but required data is missing: {', '.join(missing)}"
        )

    # -----------------------------------------------------------------------
    # Rebin observed contacts onto integer bins
    # -----------------------------------------------------------------------
    m_centers, p_obs0 = rebin_pdf_mass_to_integer_bins(
        ct_centers_native, ct_pdf[0], m_min=m_min, m_max=m_max
    )
    p_obs_mass = np.zeros((ct_pdf.shape[0], m_centers.size), dtype=float)
    p_obs_mass[0] = p_obs0
    for i in range(1, ct_pdf.shape[0]):
        _, p_obs_mass[i] = rebin_pdf_mass_to_integer_bins(
            ct_centers_native, ct_pdf[i], m_min=m_min, m_max=m_max
        )

    # -----------------------------------------------------------------------
    # Baseline p0(m) on the same integer grid
    # -----------------------------------------------------------------------
    p0_mass = build_baseline_mass_on_integer(m_centers, args.baseline)

    # -----------------------------------------------------------------------
    # Model setup
    # -----------------------------------------------------------------------
    spec = MODEL_REGISTRY[args.model]
    param_names = spec["param_names"]
    bounds = spec["bounds"]

    Tmin, Tmax = float(temps.min()), float(temps.max())
    Tref = float(args.Tref) if args.Tref is not None else 0.5 * (Tmin + Tmax)
    Tscale = float(args.Tscale) if args.Tscale is not None else max(Tmax - Tmin, 1.0)
    # For heat_capacity, --T0 overrides --Tref to set the thermodynamic reference T0
    if args.model == "heat_capacity" and args.T0 is not None:
        Tref = float(args.T0)

    # Validate temperature metadata before constructing b(T) or writing any
    # output files. Invalid values previously failed only after optimization,
    # potentially leaving partial/stale outputs behind.
    if not np.isfinite(Tref):
        raise ValueError(f"Tref must be finite, got {Tref!r}")
    if not np.isfinite(Tscale) or Tscale <= 0.0:
        raise ValueError(f"Tscale must be finite and positive, got {Tscale!r}")
    if args.model == "heat_capacity" and Tref <= 0.0:
        raise ValueError(f"heat_capacity T0 must be positive, got {Tref!r}")

    b_fn = make_b_fn(args.model, Tref, Tscale)
    loss_fn = _get_loss_fn(args.loss)

    # -----------------------------------------------------------------------
    # Train / validation split
    # -----------------------------------------------------------------------
    train_idx, val_idx = _resolve_split_indices(
        n_temps=len(temps),
        holdout_every=args.holdout_every,
        holdout_indices_str=args.holdout_indices,
        train_indices_str=args.train_indices,
    )
    if len(train_idx) == 0:
        raise ValueError("Training set is empty. Adjust holdout options.")

    has_val = len(val_idx) > 0

    print(f"Model : {args.model}  —  {spec['description']}")
    if args.model in ("hs_quadratic", "poly2", "poly3"):
        print(f"  Tref={Tref:.4g},  Tscale={Tscale:.4g}")
    elif args.model == "heat_capacity":
        print(f"  T0={Tref:.4g}")
    print(f"  Parameters: {param_names}")
    print(f"  Temperature range: [{Tmin:.4g}, {Tmax:.4g}]  ({len(temps)} temps)")
    print(f"  Loss: {args.loss}  |  train: {len(train_idx)} temps"
          + (f"  |  validation: {len(val_idx)} temps" if has_val else ""))
    if args.fit_rg:
        print(f"  Rg fitting: ON  (rg_weight={args.rg_weight})")
    else:
        print(f"  Rg fitting: OFF  (contact-only objective)")

    # -----------------------------------------------------------------------
    # Optimization with random restarts
    # -----------------------------------------------------------------------
    if minimize is None:
        raise RuntimeError("scipy is required for fitting. Install scipy.")

    rng = np.random.default_rng(args.seed)
    x0_default = np.array(spec["x0"], dtype=float)
    x0s = [x0_default.copy()]
    for _ in range(max(0, args.n_restarts - 1)):
        x0 = np.array([rng.uniform(lo, hi) for lo, hi in bounds], dtype=float)
        x0s.append(x0)

    train_temps = temps[train_idx]
    p_obs_ct_train = p_obs_mass[train_idx]

    # Combined contact + Rg objective when fitting Rg; contact-only otherwise.
    # p_obs_rg_train is also reused by the bootstrap block below.
    p_obs_rg_train = p_obs_rg_model_grid[train_idx] if args.fit_rg else None  # type: ignore[index]
    obj_fn, obj_args = build_objective(
        args.fit_rg, train_temps, m_centers, p_obs_ct_train, p0_mass, b_fn, loss_fn,
        crg_prob=crg_prob, c_edges_joint=c_edges_joint,
        p_obs_rg_train=p_obs_rg_train, rg_weight=float(args.rg_weight),
    )

    best, best_val_obj = fit_one_split(obj_fn, obj_args, x0s, bounds)

    params_fit = best.x

    print("\nBest-fit parameters:")
    for name, val in zip(param_names, params_fit):
        print(f"  {name} = {val:.6g}")
    print(f"Objective ({args.loss}, {len(train_idx)} train temps) = {best_val_obj:.6g}")

    Tc_derived: Optional[float] = None
    if spec["derived_Tc"] is not None:
        Tc_derived = spec["derived_Tc"](params_fit)
        if Tc_derived is not None:
            print(f"  Derived Tc = {Tc_derived:.6g}")

    # -----------------------------------------------------------------------
    # Contact predictions (all temperatures)
    # -----------------------------------------------------------------------
    p_mod_mass = np.zeros_like(p_obs_mass)
    for i, T in enumerate(temps):
        p_mod_mass[i] = model_contact_mass(
            p0_mass, m_centers, float(T), params_fit, b_fn
        )

    # -----------------------------------------------------------------------
    # Post-fit contact loss breakdown
    # -----------------------------------------------------------------------
    train_loss = objective(
        params_fit, train_temps, m_centers, p_obs_ct_train, p0_mass, b_fn, loss_fn
    )
    val_loss = (
        objective(
            params_fit, temps[val_idx], m_centers, p_obs_mass[val_idx],
            p0_mass, b_fn, loss_fn,
        )
        if has_val else float("nan")
    )
    all_loss = objective(
        params_fit, temps, m_centers, p_obs_mass, p0_mass, b_fn, loss_fn
    )

    print(f"\nContact loss ({args.loss}):")
    print(f"  train      ({len(train_idx):3d} temps) : {train_loss:.6g}")
    if has_val:
        print(f"  validation ({len(val_idx):3d} temps) : {val_loss:.6g}")
    print(f"  all        ({len(temps):3d} temps) : {all_loss:.6g}")

    # -----------------------------------------------------------------------
    # Rg prediction (all temperatures, if joint baseline exists)
    # -----------------------------------------------------------------------
    rg_mod_mass: Optional[np.ndarray] = None
    if has_joint_baseline:
        # Predict on the raw lattice Rg bins (crg_prob is indexed by them),
        # then relabel the axis into scaled/comparison units.  rg_mod_mass is
        # probability mass and is unchanged by the axis scaling.
        rg_centers_lattice_pred, rg_mod_mass = predict_rg_from_joint(
            crg_prob=crg_prob,                # type: ignore[arg-type]
            c_edges=c_edges_joint,            # type: ignore[arg-type]
            rg_edges=rg_edges_model_lattice,  # type: ignore[arg-type]
            temps=temps,
            params=params_fit,
            b_fn=b_fn,
        )
        rg_centers_model = args.rg_scale * rg_centers_lattice_pred

    # -----------------------------------------------------------------------
    # Post-fit Rg loss breakdown (if obs Rg and joint baseline both available)
    # -----------------------------------------------------------------------
    rg_train_loss: float = float("nan")
    rg_val_loss: float = float("nan")
    rg_all_loss: float = float("nan")
    has_rg_scoring = can_fit_rg and rg_mod_mass is not None

    if has_rg_scoring:
        assert p_obs_rg_model_grid is not None
        rg_train_loss = _rg_loss_sum(
            rg_mod_mass[train_idx], p_obs_rg_model_grid[train_idx], loss_fn
        )
        rg_val_loss = (
            _rg_loss_sum(
                rg_mod_mass[val_idx], p_obs_rg_model_grid[val_idx], loss_fn
            )
            if has_val else float("nan")
        )
        rg_all_loss = _rg_loss_sum(rg_mod_mass, p_obs_rg_model_grid, loss_fn)

        print(f"\nRg loss ({args.loss}, on model Rg grid):")
        if rg_edges_model_lattice is not None and rg_edges_model is not None:
            print(f"  Rg scale: {args.rg_scale:.6g} observed units per lattice unit")
            print(
                f"  Rg model range, lattice units: "
                f"[{float(rg_edges_model_lattice.min()):.6g}, {float(rg_edges_model_lattice.max()):.6g}]"
            )
            print(
                f"  Rg model range, scaled units:  "
                f"[{float(rg_edges_model.min()):.6g}, {float(rg_edges_model.max()):.6g}]"
            )
        print(f"  train      ({len(train_idx):3d} temps) : {rg_train_loss:.6g}")
        if has_val:
            print(f"  validation ({len(val_idx):3d} temps) : {rg_val_loss:.6g}")
        print(f"  all        ({len(temps):3d} temps) : {rg_all_loss:.6g}")
        if rg_grid_overlap_pct is not None and rg_grid_overlap_pct < 50.0:
            print(
                f"  NOTE: Rg grid overlap is {rg_grid_overlap_pct:.1f}% — "
                f"losses reflect only the overlapping Rg region."
            )
    elif has_joint_baseline and not has_obs_rg:
        print("\nRg scoring: skipped (no observed Rg histograms).")
    elif has_obs_rg and not has_joint_baseline:
        print("\nRg scoring: skipped (no joint baseline P0(m,Rg)).")

    # -----------------------------------------------------------------------
    # Save fit_results.npz
    # -----------------------------------------------------------------------
    save_kwargs: Dict = dict(
        temps=temps,
        m_centers=m_centers,
        p_obs_mass=p_obs_mass,
        p_mod_mass=p_mod_mass,
        p0_mass=p0_mass,
        baseline=str(args.baseline),
        contact_offset=float(args.contact_offset),
        model_name=args.model,
        param_names=np.array(param_names),
        params=params_fit,
        Tref=Tref,
        Tscale=Tscale,
        loss_name=args.loss,
        train_indices=train_idx,
        validation_indices=val_idx,
        train_loss=train_loss,
        val_loss=val_loss,
        all_loss=all_loss,
        fit_rg=bool(args.fit_rg),
        rg_weight=float(args.rg_weight),
        rg_scale=float(args.rg_scale),
        rg_train_loss=rg_train_loss,
        rg_val_loss=rg_val_loss,
        rg_all_loss=rg_all_loss,
    )
    # Rg arrays
    if rg_centers_model is not None:
        # Canonical keys now carry observed/comparison units (scaled).
        save_kwargs["rg_centers"] = rg_centers_model
        save_kwargs["rg_centers0"] = rg_centers_model   # backward compat alias
        save_kwargs["rg_centers_scaled"] = rg_centers_model
    if rg_edges_model is not None:
        save_kwargs["rg_edges_scaled"] = rg_edges_model
    if rg_centers_model_lattice is not None:
        save_kwargs["rg_centers_lattice"] = rg_centers_model_lattice
    if rg_edges_model_lattice is not None:
        save_kwargs["rg_edges_lattice"] = rg_edges_model_lattice
    if rg_mod_mass is not None:
        save_kwargs["rg_mod_mass"] = rg_mod_mass
    if p_obs_rg_model_grid is not None:
        save_kwargs["rg_obs_mass"] = p_obs_rg_model_grid
    # backward-compat h/s keys for hs model
    if args.model == "hs":
        save_kwargs["h"] = float(params_fit[0])
        save_kwargs["s"] = float(params_fit[1])

    np.savez_compressed(npz_path, **save_kwargs)
    print(f"\nSaved: {npz_path}")

    # -----------------------------------------------------------------------
    # Save train_validation_loss.csv
    # -----------------------------------------------------------------------
    csv_rows: List[Tuple] = [
        ("train", len(train_idx), args.loss, f"{train_loss:.8g}"),
    ]
    if has_val:
        csv_rows.append(("validation", len(val_idx), args.loss, f"{val_loss:.8g}"))
    csv_rows.append(("all", len(temps), args.loss, f"{all_loss:.8g}"))
    if has_rg_scoring:
        rg_loss_name = f"rg_{args.loss}"
        csv_rows.append(("train_rg", len(train_idx), rg_loss_name, f"{rg_train_loss:.8g}"))
        if has_val:
            csv_rows.append(("validation_rg", len(val_idx), rg_loss_name, f"{rg_val_loss:.8g}"))
        csv_rows.append(("all_rg", len(temps), rg_loss_name, f"{rg_all_loss:.8g}"))

    with open(loss_csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["split", "n_temperatures", "loss_name", "contact_loss"])
        writer.writerows(csv_rows)
    print(f"Saved: {loss_csv_path}")

    # -----------------------------------------------------------------------
    # Save fit_params.csv
    # -----------------------------------------------------------------------
    params_rows: List[Tuple[str, float]] = [
        (name, float(val)) for name, val in zip(param_names, params_fit)
    ]
    if Tc_derived is not None:
        params_rows.append(("Tc", float(Tc_derived)))

    with open(params_csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["parameter", "value"])
        writer.writerows(params_rows)
    print(f"Saved: {params_csv_path}")

    # -----------------------------------------------------------------------
    # Save fit_summary.json
    # -----------------------------------------------------------------------
    derived_dict: Dict[str, Any] = {}
    if Tc_derived is not None:
        derived_dict["Tc"] = float(Tc_derived)

    # Validate temperature metadata before persisting it.
    if not np.isfinite(Tref):
        raise ValueError(f"Tref must be finite, got {Tref!r}")
    if not np.isfinite(Tscale) or Tscale <= 0.0:
        raise ValueError(f"Tscale must be finite and positive, got {Tscale!r}")
    if args.model == "heat_capacity" and (not np.isfinite(Tref) or Tref <= 0.0):
        raise ValueError(f"heat_capacity T0 must be positive, got {Tref!r}")

    metadata: Dict[str, Any] = {
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "model_api_version": MODEL_API_VERSION,
        "remd_path": str(args.remd),
        "baseline_path": str(args.baseline),
        "model": args.model,
        "model_description": spec["description"],
        "param_names": list(param_names),
        "params": {n: float(v) for n, v in zip(param_names, params_fit)},
        "derived": derived_dict,
        "loss": args.loss,
        "optimization_success": bool(best.success),
        "optimization_message": str(best.message),
        "optimization_iterations": int(best.nit) if hasattr(best, "nit") else None,
        "optimization_objective_value": float(best_val_obj),
        "optimization_objective_includes_rg": bool(args.fit_rg),
        "Tref": float(Tref),
        "Tscale": float(Tscale),
        "contact_offset": float(args.contact_offset),
        "n_temps": int(len(temps)),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "temps_all": temps.tolist(),
        "temps_train": temps[train_idx].tolist(),
        "temps_val": temps[val_idx].tolist() if has_val else [],
        "train_indices": train_idx.tolist(),
        "val_indices": val_idx.tolist(),
        "contact_range_native": [float(ct_centers_raw.min()), float(ct_centers_raw.max())],
        "contact_range_shifted": [float(ct_centers_native.min()), float(ct_centers_native.max())],
        "remd_integer_range": [int(m_min), int(m_max)],
        "baseline_integer_range": [int(bl_min), int(bl_max)],
        "support_overlap_n": int(overlap_n),
        "support_overlap_of_remd": int(remd_n),
        "support_overlap_pct": float(overlap_pct),
        "has_joint_baseline": bool(has_joint_baseline),
        "has_obs_rg": bool(has_obs_rg),
        "rg_grid_overlap_pct": rg_grid_overlap_pct,
        "fit_rg": bool(args.fit_rg),
        "rg_weight": float(args.rg_weight),
        "rg_scale": float(args.rg_scale),
        "train_loss": float(train_loss),
        "val_loss": None if not has_val else float(val_loss),
        "all_loss": float(all_loss),
        "rg_train_loss": None if not has_rg_scoring else float(rg_train_loss),
        "rg_val_loss": None if (not has_rg_scoring or not has_val) else float(rg_val_loss),
        "rg_all_loss": None if not has_rg_scoring else float(rg_all_loss),
    }
    # For heat_capacity, persist the thermodynamic reference temperature as T0
    # (stored in Tref by this fitter). Keep Tref for backward compatibility.
    if args.model == "heat_capacity":
        metadata["T0"] = float(Tref)

    if has_joint_baseline and rg_edges_model_lattice is not None and rg_edges_model is not None:
        metadata["rg_model_range_lattice"] = [
            float(rg_edges_model_lattice.min()), float(rg_edges_model_lattice.max())
        ]
        metadata["rg_model_range_scaled"] = [
            float(rg_edges_model.min()), float(rg_edges_model.max())
        ]

    with open(json_path, "w") as fh:
        json.dump(metadata, fh, indent=2, cls=_NpEncoder, allow_nan=False)
    print(f"Saved: {json_path}")

    # -----------------------------------------------------------------------
    # Bootstrap uncertainty + optional local-curvature/restart diagnostics.
    # Both reuse the shared optimization pathway (build_objective / fit_one_split /
    # fit_restarts) and write supplementary files only; the suite-consumed primary
    # outputs above are untouched.
    # -----------------------------------------------------------------------
    uncertainty_ctx: Dict[str, Any] = {
        "temps": temps, "m_centers": m_centers, "p_obs_mass": p_obs_mass,
        "p0_mass": p0_mass, "b_fn": b_fn, "loss_fn": loss_fn, "spec": spec,
        "param_names": param_names, "bounds": bounds, "x0s": x0s,
        "fit_rg": bool(args.fit_rg), "rg_weight": float(args.rg_weight),
        "can_fit_rg": can_fit_rg, "crg_prob": crg_prob,
        "c_edges_joint": c_edges_joint,
        "rg_edges_model_lattice": rg_edges_model_lattice,
        "p_obs_rg_model_grid": p_obs_rg_model_grid,
        "outdir": plot_dir, "loss_name": args.loss, "model_name": args.model,
        "train_idx": train_idx, "val_idx": val_idx, "has_val": has_val,
        "train_temps": train_temps, "p_obs_ct_train": p_obs_ct_train,
        "p_obs_rg_train": p_obs_rg_train, "params_fit": params_fit,
        "Tc_derived": Tc_derived, "rg_scale": float(args.rg_scale),
    }

    if args.bootstrap > 0:
        run_bootstrap_uncertainty(args, uncertainty_ctx)

    if args.uncertainty_diagnostics:
        run_uncertainty_diagnostics(args, uncertainty_ctx)

    if args.rg_weight_grid is not None or args.rg_weight_grid_file is not None:
        run_rg_weight_sensitivity(args, uncertainty_ctx)

    # -----------------------------------------------------------------------
    # Validation-split sensitivity (optional; supplementary outputs only).
    # Runs after all primary outputs are written and before primary plots, so it
    # executes even with --no-plots and never alters the files the suite reads.
    # -----------------------------------------------------------------------
    if args.split_sensitivity:
        ctx: Dict[str, Any] = {
            "temps": temps, "m_centers": m_centers, "p_obs_mass": p_obs_mass,
            "p0_mass": p0_mass, "b_fn": b_fn, "loss_fn": loss_fn, "spec": spec,
            "param_names": param_names, "bounds": bounds, "x0s": x0s,
            "fit_rg": bool(args.fit_rg), "rg_weight": float(args.rg_weight),
            "can_fit_rg": can_fit_rg, "crg_prob": crg_prob,
            "c_edges_joint": c_edges_joint,
            "rg_edges_model_lattice": rg_edges_model_lattice,
            "p_obs_rg_model_grid": p_obs_rg_model_grid,
            "outdir": plot_dir, "loss_name": args.loss, "model_name": args.model,
        }
        run_split_sensitivity(args, ctx)

    # -----------------------------------------------------------------------
    # Plots
    # -----------------------------------------------------------------------
    if args.no_plots:
        return
    if plt is None:
        raise RuntimeError(
            "matplotlib is required for plots; install it or use --no-plots"
        )

    val_set = set(val_idx.tolist())
    n_show = min(8, len(temps))
    show_idxs = np.linspace(0, len(temps) - 1, n_show).astype(int)
    cmap = plt.get_cmap("tab10")
    open_figs: List[plt.Figure] = []

    # --- 1. Mean contacts vs T ---
    mean_obs = (m_centers[None, :] * p_obs_mass).sum(axis=1)
    mean_mod = (m_centers[None, :] * p_mod_mass).sum(axis=1)

    fig1, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(temps, mean_obs, "o", ms=4, label="obs (all)")
    ax1.plot(temps, mean_mod, "-", lw=1.5, label="model")
    if has_val:
        ax1.plot(
            temps[val_idx], mean_obs[val_idx],
            "x", color="red", ms=7, mew=2, label="held-out obs",
        )
    ax1.set_xlabel("T")
    ax1.set_ylabel("mean contacts")
    title_suffix = " [+Rg fit]" if args.fit_rg else ""
    ax1.set_title(f"Mean contacts vs T  [{args.model}, {args.loss}]{title_suffix}")
    ax1.legend(fontsize=8)
    fig1.tight_layout()
    p1 = plot_dir / "mean_contacts_fit.png"
    fig1.savefig(p1, dpi=150, bbox_inches="tight")
    print(f"Saved: {p1}")
    open_figs.append(fig1)

    # --- 2. Contact distribution overlay (train solid, val dashed) ---
    from matplotlib.lines import Line2D

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    for k, i in enumerate(show_idxs):
        color = cmap(k % 10)
        ls = "--" if i in val_set else "-"
        ax2.step(m_centers, p_obs_mass[i], where="mid", color=color, alpha=0.25, lw=1.0, ls=ls)
        ax2.step(m_centers, p_mod_mass[i], where="mid", color=color, alpha=0.85, lw=1.8, ls=ls)
    ax2.set_xlabel("m (integer contacts)")
    ax2.set_ylabel("P(m)")
    ax2.set_title(
        f"Obs (faint) vs model (bold)  [{args.model}]\n"
        f"solid = train,  dashed = validation"
    )
    leg2 = [Line2D([0], [0], color="gray", lw=1.5, ls="-", label="train")]
    if has_val:
        leg2.append(Line2D([0], [0], color="gray", lw=1.5, ls="--", label="validation"))
    ax2.legend(handles=leg2, fontsize=8)
    fig2.tight_layout()
    p2 = plot_dir / "contact_distribution_overlay_train_val.png"
    fig2.savefig(p2, dpi=150, bbox_inches="tight")
    print(f"Saved: {p2}")
    open_figs.append(fig2)

    # --- 3. Reduced bias b(T) vs T ---
    b_vals = np.array([b_fn(params_fit, T) for T in temps])
    fig3, ax3 = plt.subplots(figsize=(6, 4))
    ax3.plot(temps, b_vals, "k-", lw=1.8)
    ax3.axhline(0.0, color="gray", lw=0.8, ls="--")
    if has_val:
        ax3.scatter(temps[val_idx], b_vals[val_idx], color="red", zorder=3, s=25, label="held-out")
        ax3.legend(fontsize=8)
    ax3.set_xlabel("T")
    ax3.set_ylabel("b(T)")
    ax3.set_title(f"Reduced bias  [{spec['description']}]")
    fig3.tight_layout()
    p3 = plot_dir / "reduced_bias_vs_T.png"
    fig3.savefig(p3, dpi=150, bbox_inches="tight")
    print(f"Saved: {p3}")
    open_figs.append(fig3)

    # --- 4. Contact residual heatmap ---
    residuals_ct = p_obs_mass - p_mod_mass   # shape (n_temps, n_m)
    fig_crhm, ax_crhm = plt.subplots(figsize=(9, 5))
    vmax_ct = float(np.abs(residuals_ct).max())
    vmax_ct = vmax_ct if vmax_ct > 0 else 1.0
    im_crhm = ax_crhm.pcolormesh(
        m_centers, temps, residuals_ct,
        cmap="RdBu_r", vmin=-vmax_ct, vmax=vmax_ct, shading="auto",
    )
    plt.colorbar(im_crhm, ax=ax_crhm, label="obs − model")
    ax_crhm.set_xlabel("m (integer contacts)")
    ax_crhm.set_ylabel("T")
    ax_crhm.set_title("Contact residual heatmap (obs − model)")
    if has_val:
        for vi in val_idx:
            ax_crhm.axhline(temps[vi], color="red", lw=0.5, alpha=0.4)
    fig_crhm.tight_layout()
    p_crhm = plot_dir / "contact_residual_heatmap.png"
    fig_crhm.savefig(p_crhm, dpi=150, bbox_inches="tight")
    print(f"Saved: {p_crhm}")
    open_figs.append(fig_crhm)

    # --- 6. Rg distribution overlay (if joint baseline) ---
    if rg_mod_mass is not None and rg_centers_model is not None:
        fig4, ax4 = plt.subplots(figsize=(8, 5))
        for k, i in enumerate(show_idxs):
            color = cmap(k % 10)
            ls = "--" if i in val_set else "-"
            # Observed (native grid), faint
            if p_obs_rg_native is not None and rg_centers_obs is not None:
                ax4.plot(
                    rg_centers_obs, p_obs_rg_native[i],
                    color=color, alpha=0.25, lw=1.0, ls=ls,
                )
            # Model (model grid), bold
            ax4.plot(
                rg_centers_model, rg_mod_mass[i],
                color=color, alpha=0.85, lw=1.8, ls=ls,
            )
        ax4.set_xlabel("Rg (observed units; lattice Rg scaled by --rg-scale)")
        ax4.set_ylabel("P(Rg)")
        obs_note = "obs (faint, native grid) vs " if has_obs_rg else ""
        ax4.set_title(
            f"{obs_note}predicted (bold, model grid)\n"
            f"solid = train,  dashed = validation"
        )
        leg4 = [Line2D([0], [0], color="gray", lw=1.5, ls="-", label="train")]
        if has_val:
            leg4.append(Line2D([0], [0], color="gray", lw=1.5, ls="--", label="validation"))
        ax4.legend(handles=leg4, fontsize=8)
        fig4.tight_layout()
        p4 = plot_dir / "rg_distribution_overlay.png"
        fig4.savefig(p4, dpi=150, bbox_inches="tight")
        print(f"Saved: {p4}")
        open_figs.append(fig4)

    # --- 7. Rg residual heatmap (if obs Rg available and joint baseline) ---
    if has_rg_scoring and rg_mod_mass is not None and rg_centers_obs is not None and p_obs_rg_native is not None:
        # Rebin model Rg mass (on scaled model edges) onto obs Rg grid for
        # residual computation.  rebin_mass_between_edges treats rg_mod_mass as
        # probability mass on rg_edges_model (scaled) and is robust to uneven grids.
        rg_edges_obs = centers_to_edges(rg_centers_obs)
        rg_mod_on_obs_grid = np.array([
            rebin_mass_between_edges(rg_edges_model, rg_mod_mass[i], rg_edges_obs)
            for i in range(len(temps))
        ])
        residuals = p_obs_rg_native - rg_mod_on_obs_grid  # shape: (n_temps, n_rg_obs)

        fig5, ax5 = plt.subplots(figsize=(8, 5))
        vmax = float(np.abs(residuals).max())
        vmax = vmax if vmax > 0 else 1.0
        im = ax5.pcolormesh(
            rg_centers_obs, temps, residuals,
            cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto",
        )
        plt.colorbar(im, ax=ax5, label="obs – model")
        ax5.set_xlabel("Rg")
        ax5.set_ylabel("T")
        ax5.set_title(f"Rg residual heatmap (obs − model, model rebinned to obs grid)")
        if has_val:
            for vi in val_idx:
                ax5.axhline(temps[vi], color="red", lw=0.5, alpha=0.4)
        fig5.tight_layout()
        p5 = plot_dir / "rg_residual_heatmap.png"
        fig5.savefig(p5, dpi=150, bbox_inches="tight")
        print(f"Saved: {p5}")
        open_figs.append(fig5)

    if args.show_plots:
        plt.show()
    else:
        for fig in open_figs:
            plt.close(fig)


if __name__ == "__main__":
    main()
