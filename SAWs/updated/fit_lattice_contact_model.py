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
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

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


def build_baseline_mass_on_integer(
    m_centers_int: np.ndarray, baseline_npz: str
) -> np.ndarray:
    """Return baseline probability mass p0(m) on the integer m grid."""
    b = np.load(baseline_npz)
    m_centers_int = np.asarray(m_centers_int, dtype=float)
    m0 = int(round(m_centers_int.min()))
    m1 = int(round(m_centers_int.max()))
    m_vals = np.arange(m0, m1 + 1, dtype=int)

    # Case A: discrete contact values (c_vals, c_prob)
    if "c_vals" in b.files and "c_prob" in b.files:
        c_vals = np.asarray(b["c_vals"], dtype=int)
        c_prob = np.asarray(b["c_prob"], dtype=float)
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

    # Case B: baseline provided as contact pdf on arbitrary bins
    if "ct_centers" in b.files and "ct_hists" in b.files:
        ccent = np.asarray(b["ct_centers"], dtype=float)
        ch = np.asarray(b["ct_hists"], dtype=float)
        if ch.ndim == 2:
            ch = ch[0]
        _, p0 = rebin_pdf_mass_to_integer_bins(ccent, ch, m_min=m0, m_max=m1)
        return p0

    raise ValueError(
        "Unrecognized baseline format. Expected (c_vals,c_prob) or (ct_centers,ct_hists)."
    )


def _get_baseline_integer_range(baseline_npz: str) -> Tuple[int, int]:
    """Return (min, max) integer contact range of the baseline without building p0."""
    b = np.load(baseline_npz)
    if "c_vals" in b.files:
        c_vals = np.asarray(b["c_vals"], dtype=int)
        return int(c_vals.min()), int(c_vals.max())
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
    """
    all_idx = np.arange(n_temps, dtype=int)

    if train_indices_str is not None:
        train_idx = np.array(
            [int(x.strip()) for x in train_indices_str.split(",")], dtype=int
        )
        val_idx = np.setdiff1d(all_idx, train_idx)
        return train_idx, val_idx

    if holdout_indices_str is not None:
        val_idx = np.array(
            [int(x.strip()) for x in holdout_indices_str.split(",")], dtype=int
        )
        train_idx = np.setdiff1d(all_idx, val_idx)
        return train_idx, val_idx

    if holdout_every is not None:
        val_idx = all_idx[::holdout_every]
        train_idx = np.setdiff1d(all_idx, val_idx)
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
            return None if np.isnan(obj) else float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


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
        default="single_uniform_chain2_athermal_dists_joint_N44_T1_seed42.npz",
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
    ap.add_argument("--no_plots", action="store_true", help="Skip all plot generation.")
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

    # Rg fitting
    ap.add_argument(
        "--fit-rg", action="store_true", dest="fit_rg",
        help=(
            "Include Rg loss in the optimization objective. Requires joint baseline "
            "P0(m,Rg) and observed Rg histograms."
        ),
    )
    ap.add_argument(
        "--rg-weight", type=float, default=1.0, dest="rg_weight",
        help="Weight of Rg loss relative to contact loss when --fit-rg is active.",
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
    args = ap.parse_args()

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
    temps = np.asarray(d["temps"], dtype=float)
    ct_centers_raw = np.asarray(d["ct_centers"], dtype=float)        # before offset
    ct_centers_native = ct_centers_raw - float(args.contact_offset)  # after offset
    ct_pdf = np.asarray(d["ct_hists"], dtype=float)

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
        _rg_edges_model = np.asarray(b_data["rg_edges"], dtype=float)
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
        print(f"  Rg obs range:   [{rg_obs_min:.4g}, {rg_obs_max:.4g}]")
        print(f"  Rg model range: [{rg_model_min:.4g}, {rg_model_max:.4g}]")
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
    rg_centers_model: Optional[np.ndarray] = None
    rg_edges_model: Optional[np.ndarray] = None
    crg_prob: Optional[np.ndarray] = None
    c_edges_joint: Optional[np.ndarray] = None

    if has_obs_rg:
        rg_centers_obs = np.asarray(d[_rg_centers_key], dtype=float)
        rg_hists_obs = np.asarray(d[_rg_hists_key], dtype=float)
        # Convert each obs PDF to probability mass on native grid
        p_obs_rg_native = np.array(
            [pdf_to_mass(rg_hists_obs[i], rg_centers_obs)[0] for i in range(len(temps))]
        )

    if has_joint_baseline:
        crg_prob = np.asarray(b_data["crg_prob"], dtype=float)
        c_edges_joint = np.asarray(b_data["c_edges"], dtype=float)
        rg_edges_model = np.asarray(b_data["rg_edges"], dtype=float)
        rg_centers_model = 0.5 * (rg_edges_model[:-1] + rg_edges_model[1:])

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

    if args.fit_rg:
        # Combined contact + Rg objective
        p_obs_rg_train = p_obs_rg_model_grid[train_idx]  # type: ignore[index]
        obj_fn = objective_combined
        obj_args = (
            train_temps, m_centers, p_obs_ct_train, p0_mass,
            crg_prob, c_edges_joint, p_obs_rg_train,
            b_fn, loss_fn, float(args.rg_weight),
        )
    else:
        # Contact-only objective (default, identical to pre-Phase-4 behavior)
        obj_fn = objective
        obj_args = (train_temps, m_centers, p_obs_ct_train, p0_mass, b_fn, loss_fn)

    best = None
    best_val_obj = float("inf")
    for x0 in x0s:
        res = minimize(
            obj_fn, x0,
            args=obj_args,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 800},
        )
        if res.fun < best_val_obj:
            best_val_obj = float(res.fun)
            best = res

    if best is None:
        raise RuntimeError("Fitting failed")

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
        _, rg_mod_mass = predict_rg_from_joint(
            crg_prob=crg_prob,      # type: ignore[arg-type]
            c_edges=c_edges_joint,  # type: ignore[arg-type]
            rg_edges=rg_edges_model,  # type: ignore[arg-type]
            temps=temps,
            params=params_fit,
            b_fn=b_fn,
        )

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
        rg_train_loss=rg_train_loss,
        rg_val_loss=rg_val_loss,
        rg_all_loss=rg_all_loss,
    )
    # Rg arrays
    if rg_centers_model is not None:
        save_kwargs["rg_centers"] = rg_centers_model
        save_kwargs["rg_centers0"] = rg_centers_model   # backward compat alias
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

    metadata: Dict[str, Any] = {
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "remd_path": str(args.remd),
        "baseline_path": str(args.baseline),
        "model": args.model,
        "model_description": spec["description"],
        "param_names": param_names,
        "params": {n: float(v) for n, v in zip(param_names, params_fit)},
        "derived": derived_dict,
        "loss": args.loss,
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
        "train_loss": float(train_loss),
        "val_loss": None if not has_val else float(val_loss),
        "all_loss": float(all_loss),
        "rg_train_loss": None if not has_rg_scoring else float(rg_train_loss),
        "rg_val_loss": None if (not has_rg_scoring or not has_val) else float(rg_val_loss),
        "rg_all_loss": None if not has_rg_scoring else float(rg_all_loss),
    }

    with open(json_path, "w") as fh:
        json.dump(metadata, fh, indent=2, cls=_NpEncoder)
    print(f"Saved: {json_path}")

    # -----------------------------------------------------------------------
    # Bootstrap uncertainty estimation
    # -----------------------------------------------------------------------
    boot_csv_path = plot_dir / "bootstrap_params.csv"
    boot_json_path = plot_dir / "bootstrap_summary.json"

    if args.bootstrap > 0:
        bseed = args.bootstrap_seed if args.bootstrap_seed is not None else args.seed
        bstrap_rng = np.random.default_rng(bseed)
        n_train = len(train_idx)

        boot_records: List[Dict[str, Any]] = []
        n_boot_failed = 0

        print(f"\nBootstrap ({args.bootstrap} replicates, seed={bseed}):")

        for bi in range(args.bootstrap):
            # Resample training indices with replacement
            local_idx = bstrap_rng.integers(0, n_train, size=n_train)
            boot_temps_b = train_temps[local_idx]
            boot_ct_b = p_obs_ct_train[local_idx]

            if args.fit_rg:
                boot_rg_b = p_obs_rg_train[local_idx]
                obj_args_b = (
                    boot_temps_b, m_centers, boot_ct_b, p0_mass,
                    crg_prob, c_edges_joint, boot_rg_b,
                    b_fn, loss_fn, float(args.rg_weight),
                )
                obj_fn_b = objective_combined
            else:
                obj_args_b = (boot_temps_b, m_centers, boot_ct_b, p0_mass, b_fn, loss_fn)
                obj_fn_b = objective

            best_b = None
            best_b_val = float("inf")
            try:
                for x0 in x0s:
                    res = minimize(
                        obj_fn_b, x0,
                        args=obj_args_b,
                        method="L-BFGS-B",
                        bounds=bounds,
                        options={"maxiter": 800},
                    )
                    if res.fun < best_b_val:
                        best_b_val = float(res.fun)
                        best_b = res
            except Exception as exc:
                print(f"  replicate {bi:4d}: FAILED ({exc})")
                n_boot_failed += 1
                continue

            if best_b is None:
                print(f"  replicate {bi:4d}: FAILED (no successful minimize)")
                n_boot_failed += 1
                continue

            params_b = best_b.x

            # Post-fit losses evaluated on the original (non-resampled) temperature sets
            train_loss_b = objective(
                params_b, train_temps, m_centers, p_obs_ct_train, p0_mass, b_fn, loss_fn
            )
            val_loss_b = (
                objective(
                    params_b, temps[val_idx], m_centers, p_obs_mass[val_idx],
                    p0_mass, b_fn, loss_fn,
                )
                if has_val else float("nan")
            )
            all_loss_b = objective(
                params_b, temps, m_centers, p_obs_mass, p0_mass, b_fn, loss_fn
            )

            Tc_b: Optional[float] = None
            if spec["derived_Tc"] is not None:
                Tc_b = spec["derived_Tc"](params_b)

            record: Dict[str, Any] = {"bootstrap_index": bi}
            for pname, pval in zip(param_names, params_b):
                record[pname] = float(pval)
            if Tc_b is not None:
                record["Tc"] = float(Tc_b)
            record["train_loss"] = float(train_loss_b)
            if has_val:
                record["validation_loss"] = float(val_loss_b)
            record["all_loss"] = float(all_loss_b)
            boot_records.append(record)

            interval = max(1, args.bootstrap // 5)
            if (bi + 1) % interval == 0 or bi == args.bootstrap - 1:
                print(f"  {bi + 1}/{args.bootstrap} done")

        n_boot_success = len(boot_records)

        if n_boot_success == 0:
            print(
                f"WARNING: All {args.bootstrap} bootstrap replicates failed. "
                f"bootstrap_params.csv and bootstrap_summary.json not saved."
            )
        else:
            if n_boot_failed > 0:
                print(f"  {n_boot_failed} replicate(s) failed and were excluded.")

            # CSV: header derived from first record (preserves column order)
            boot_header = list(boot_records[0].keys())
            with open(boot_csv_path, "w", newline="") as fh:
                writer = csv.writer(fh)
                writer.writerow(boot_header)
                for rec in boot_records:
                    writer.writerow([rec.get(col, "") for col in boot_header])
            print(f"Saved: {boot_csv_path}")

            # Summary statistics
            param_boot_stats: Dict[str, Dict[str, float]] = {}
            for pname in param_names:
                arr = np.array([r[pname] for r in boot_records], dtype=float)
                param_boot_stats[pname] = {"mean": float(arr.mean()), "std": float(arr.std())}

            derived_boot_stats: Dict[str, Dict[str, float]] = {}
            if spec["derived_Tc"] is not None:
                tc_arr = np.array(
                    [r["Tc"] for r in boot_records if "Tc" in r], dtype=float
                )
                if tc_arr.size > 0:
                    derived_boot_stats["Tc"] = {
                        "mean": float(tc_arr.mean()), "std": float(tc_arr.std())
                    }

            loss_boot_stats: Dict[str, Dict[str, float]] = {}
            for lkey in ("train_loss", "validation_loss", "all_loss"):
                if lkey in boot_records[0]:
                    lv = np.array(
                        [r[lkey] for r in boot_records if lkey in r], dtype=float
                    )
                    lv = lv[np.isfinite(lv)]
                    if lv.size > 0:
                        loss_boot_stats[lkey] = {
                            "mean": float(lv.mean()), "std": float(lv.std())
                        }

            boot_summary: Dict[str, Any] = {
                "n_bootstrap": int(args.bootstrap),
                "n_success": int(n_boot_success),
                "n_failed": int(n_boot_failed),
                "bootstrap_seed": int(bseed),
                "model": args.model,
                "loss": args.loss,
                "fit_rg": bool(args.fit_rg),
                "rg_weight": float(args.rg_weight),
                "params": param_boot_stats,
                "derived": derived_boot_stats,
                "losses": loss_boot_stats,
            }
            with open(boot_json_path, "w") as fh:
                json.dump(boot_summary, fh, indent=2, cls=_NpEncoder)
            print(f"Saved: {boot_json_path}")

    # -----------------------------------------------------------------------
    # Plots
    # -----------------------------------------------------------------------
    if args.no_plots:
        return

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

    # --- 4. Rg distribution overlay (if joint baseline) ---
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
        ax4.set_xlabel("Rg")
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

    # --- 5. Rg residual heatmap (if obs Rg available and joint baseline) ---
    if has_rg_scoring and rg_mod_mass is not None and rg_centers_obs is not None and p_obs_rg_native is not None:
        # Rebin model Rg mass onto obs Rg grid for residual computation
        rg_edges_obs = centers_to_edges(rg_centers_obs)
        rg_mod_on_obs_grid = np.array([
            rebin_pdf_to_mass(rg_centers_model, rg_mod_mass[i], rg_edges_obs)
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
