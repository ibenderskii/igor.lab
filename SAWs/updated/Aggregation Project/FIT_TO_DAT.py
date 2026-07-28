#!/usr/bin/env python3
"""Fit a lattice polymer contact model to REMD contact histograms.

Model
-----
P_model(m|T) ∝ P0(m) * exp[-u_contact(m, T; N)]

where P0(m) is the athermal (T→∞) baseline from a SAW / lattice simulation.
For every model below except the three nonlinear ones the contact potential is
linear in m,

    u_contact(m, T; N) = b(T) * m

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

Contact-quadratic models
------------------------
These two do NOT change b(T)'s meaning; they add a curvature term in the
CONTACT NUMBER m, normalized by the chain length N:

hs_m2_const   u_contact(m,T;N) = (h1/T - s1)*m + kappa2 * m^2/(2N)
                Temperature-independent contact-number curvature.
                Parameters: h1, s1, kappa2.

hs_m2_hs      u_contact(m,T;N) = (h1/T - s1)*m + (h2/T - s2) * m^2/(2N)
                Enthalpy/entropy decomposition of the curvature term.
                Parameters: h1, s1, h2, s2.

Both nest the hs model exactly: with the quadratic parameters at zero the
potential is identically (h1/T - s1)*m, which is why the first optimizer
restart starts there.  N is the number of beads; it is read from the baseline
(n_beads, else N) and may be supplied with --N when the baseline predates that
metadata.  hs_quadratic is unrelated: it is a quadratic in TEMPERATURE and
leaves the potential linear in m.

Saturating-cooperative model
----------------------------
This one is written in the contact FRACTION q = m/N (measured from m_ref = 0)
rather than in m, so its cooperative attraction saturates instead of growing
without bound:

saturating_cooperative_contact
              u_contact(m,T;N) = N * [ b(T)*q - A0*q^2/(1 + (q/q_sat)^2) ]
              b(T) = h_b/T - s_b,  q = m/N,  m_ref = 0
                Parameters: h_b, s_b, A0, q_sat.  A0 >= 0 and q_sat > 0 are
                both temperature-independent.  At q << q_sat the cooperative
                term is -N*A0*q^2, i.e. an ordinary quadratic attraction; at
                q >> q_sat it flattens to the constant -N*A0*q_sat^2, so the
                marginal slope du/dm returns to b(T).  A0 = 0 reproduces the hs
                potential exactly (bit for bit, not merely to tolerance), which
                is why the first optimizer restart starts there.  Needs N, for
                the q normalization rather than for an m^2/(2N) term: this model
                has no kappa(T), so no quadratic-coefficient plot is drawn for
                it and it gets a contact-potential plot instead.

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

Always run --rg-feasibility-scan.  It reports reachability at two strengths, which
must not be conflated:

  finite scan   the range of scalar Rg produced over the CONFIGURED bias interval
                [--rg-bias-min, --rg-bias-max].  A target outside it is
                unreachable WITHIN THE SCANNED INTERVAL.  Because the scan is
                finite it can never prove that no real b reproduces a target;
                widening the interval is always a legitimate response.
  asymptotic    the exact scalar-Rg range in the limits b -> +/-inf, computed in
                closed form from the minimum- and maximum-contact slices of the
                baseline.  A target outside THIS range cannot be reproduced by any
                real bias, given the baseline support.  This is the only
                reachability statement that licenses an all-b claim.

Both are scalar-summary statements about the fitted Rg(T) curve; neither is a
claim about full-distribution support, which is reported separately.

Both also presuppose a contact potential LINEAR in m: only then does a single
scalar b index the whole family of reweightings.  For every model NONLINEAR in m
-- the contact-quadratic pair and saturating_cooperative_contact alike -- the finite
one-dimensional scan and the b -> +/-inf endpoint reading are NOT applicable and
are not run.  What replaces them is model-independent and
holds for any contact-only reweighting: the support-overlap check, the rigorous
global outer bound derived from the contact-conditioned Rg values, and a
contact-slice conditional-Rg table written to
<prefix>_contact_slices.csv.  The fitted Rg(T) curve and T_rg_max_slope are
reported exactly as for the linear models.

Scientific status is structured per scale (nominal vs fitted), never a single
VALID/NOT VALID Boolean, and never follows from optimizer convergence.

Transition descriptors are distinct and both are reported:
  T_bias_zero     temperature where b(T) = 0 (bias sign change)
  T_rg_max_slope  temperature maximizing -dRg/dT of the fitted finite-chain
                  curve; the primary finite-chain transition descriptor.  It is
                  null when the curve shows no resolved collapse.
"""

from __future__ import annotations

import argparse
import csv
import json
# Aliased: several functions here build a local list literally named `warnings`
# (the diagnostic messages they emit), which would shadow a bare `import warnings`.
import warnings as _warnings
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

try:  # only used to refine tangential bias roots; the grid search still works
    from scipy.optimize import minimize_scalar
except Exception:
    minimize_scalar = None


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


# ---------------------------------------------------------------------------
# Baseline bending penalty (metadata only; never fitted)
# ---------------------------------------------------------------------------
# The bending penalty enters through the BASELINE distribution:
#     P_kappa,0(m, Rg) ~ P_0(m, Rg) * exp[-kappa_bend * n_bend]
# and is therefore already baked into the baseline NPZ produced by
# single_uniform_chain2_athermal_dists_joint.py.  The fitted model stays
#     P(m, Rg | T) ~ P_kappa,0(m, Rg) * exp[-b(T) * m]
# so kappa_bend must NOT be applied again during reweighting and is never
# optimized here.  The CLI value, when given, is a consistency check only.
BEND_DEFINITION = "90-degree turns; straight=0, right-angle turn=1"
KAPPA_BEND_TOL = 1e-9


def read_baseline_kappa_bend(b_data) -> float:
    """Return the bending penalty recorded in a baseline NPZ.

    Legacy baselines predate the bending penalty and carry no ``kappa_bend``
    key; they are athermal in the bending sense and read as 0.0.
    """
    if "kappa_bend" not in b_data.files:
        return 0.0
    kappa = float(np.asarray(b_data["kappa_bend"]).reshape(()))
    if not np.isfinite(kappa):
        raise ValueError(f"baseline kappa_bend must be finite, got {kappa!r}")
    if kappa < 0.0:
        raise ValueError(f"baseline kappa_bend must be >= 0, got {kappa!r}")
    return kappa


def resolve_kappa_bend(
    baseline_kappa: float,
    cli_kappa,
    baseline_path: str = "",
    tol: float = KAPPA_BEND_TOL,
) -> float:
    """Reconcile a CLI --kappa-bend with the value stored in the baseline.

    The baseline is authoritative.  When the CLI value is supplied it must match
    the baseline within ``tol``; a mismatch means the baseline does not encode
    the stiffness the caller believes it does, which would silently corrupt the
    fit, so it raises.
    """
    baseline_kappa = float(baseline_kappa)
    if cli_kappa is None:
        return baseline_kappa
    cli_kappa = float(cli_kappa)
    if not np.isfinite(cli_kappa):
        raise ValueError(f"--kappa-bend must be finite, got {cli_kappa!r}")
    if cli_kappa < 0.0:
        raise ValueError(f"--kappa-bend must be >= 0, got {cli_kappa!r}")
    if abs(cli_kappa - baseline_kappa) > tol:
        where = f" ({baseline_path})" if baseline_path else ""
        raise ValueError(
            f"--kappa-bend {cli_kappa!r} does not match the baseline"
            f"{where} kappa_bend {baseline_kappa!r} "
            f"(tolerance {tol:g}). The bending penalty is baked into the "
            "baseline distribution and is not refitted; regenerate the baseline "
            "with the intended --kappa-bend, or drop the CLI flag."
        )
    return baseline_kappa


# ---------------------------------------------------------------------------
# Baseline chain length (metadata only; never fitted)
# ---------------------------------------------------------------------------
# The nonlinear models normalize by the chain length N (number of beads): the
# contact-quadratic pair through kappa(T) * m^2 / (2N), the saturating-
# cooperative model through the contact fraction q = m/N.  N is a property of
# the simulated chain, so the baseline is authoritative; --N exists only for
# legacy baselines that predate the metadata.  Linear models never need it,
# which is why a missing chain length is not an error for them.

def read_baseline_chain_length(b_data) -> Optional[int]:
    """Return the chain length (number of beads) recorded in a baseline NPZ.

    ``n_beads`` is preferred; ``N`` is the older spelling of the same quantity.
    Returns None for legacy baselines that record neither.  Raises when the two
    keys disagree, because then the file does not identify one chain.
    """
    found: Dict[str, int] = {}
    for key in ("n_beads", "N"):
        if key not in b_data.files:
            continue
        raw = np.asarray(b_data[key]).reshape(())
        value = float(raw)
        if not np.isfinite(value):
            raise ValueError(f"baseline {key} must be finite, got {value!r}")
        if abs(value - round(value)) > 1e-9:
            raise ValueError(f"baseline {key} must be an integer, got {value!r}")
        if round(value) < 2:
            raise ValueError(f"baseline {key} must be >= 2, got {value!r}")
        found[key] = int(round(value))
    if len(found) == 2 and found["n_beads"] != found["N"]:
        raise ValueError(
            f"baseline records conflicting chain lengths: n_beads="
            f"{found['n_beads']} but N={found['N']}. Regenerate the baseline; the "
            f"chain-length normalization (m^2/(2N), q = m/N) is ambiguous "
            f"otherwise."
        )
    if "n_beads" in found:
        return found["n_beads"]
    return found.get("N")


def resolve_chain_length(
    baseline_n: Optional[int],
    cli_n: Optional[int],
    *,
    model_name: str,
    baseline_path: str = "",
) -> Optional[int]:
    """Reconcile the baseline chain length with an optional --N override.

    The baseline wins when both are present and they agree; a disagreement
    raises rather than silently rescaling the nonlinear term.  Models whose
    ``requires_chain_length`` flag is set fail when no chain length is available;
    legacy linear fits against a baseline without the metadata keep working.
    """
    if cli_n is not None:
        cli_n = int(cli_n)
        if cli_n < 2:
            raise ValueError(f"--N must be >= 2, got {cli_n}")
    if baseline_n is not None and cli_n is not None and baseline_n != cli_n:
        where = f" ({baseline_path})" if baseline_path else ""
        raise ValueError(
            f"--N {cli_n} conflicts with the chain length recorded in the "
            f"baseline{where}: {baseline_n}. The baseline is authoritative; drop "
            f"--N or point at the matching baseline."
        )
    resolved = baseline_n if baseline_n is not None else cli_n
    spec = MODEL_REGISTRY[model_name]
    if resolved is None and spec.get("requires_chain_length"):
        where = f" ({baseline_path})" if baseline_path else ""
        raise ValueError(
            f"model {model_name!r} normalizes its nonlinear contact term by the "
            f"chain length ({spec.get('potential_normalization')}), but the "
            f"baseline{where} records neither 'n_beads' nor 'N'. Pass --N with "
            f"the number of beads, or regenerate the baseline."
        )
    return resolved


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
# Each entry defines a reduced contact potential u_contact(m, T; N) used as
#
#     P_model(m|T) ∝ P0(m) * exp[-u_contact(m, T; N)]
#
# Three potential families are supported, selected by potential_kind:
#
#   "linear"                 u = b(T) * m
#   "contact_quadratic"      u = b(T) * m + kappa(T) * m^2/(2N)
#   "saturating_cooperative" u = N * [b(T)*q - A0*q^2/(1 + (q/q_sat)^2)],
#                            q = m/N
#
# b(T) is the reduced bias (linear contact coupling) and kappa(T) the quadratic
# coefficient, which is identically zero for every model whose potential_kind is
# "linear" -- there u_contact reduces to exactly b(T)*m.  kappa(T) is also zero
# for the saturating family, which has no m^2/(2N) term at all: its cooperative
# attraction is written in the contact fraction q and saturates in q, so it is
# NOT expressible through raw_q_fn / quadratic_normalization and does not use
# them.  The full potential is built by POTENTIAL_BUILDERS[potential_kind]; the
# b(T)/kappa(T) accessors below remain the coefficient-level API.
#
# raw_b_fn(params, T, Tref, Tscale) -> float
# raw_q_fn(params, T, Tref, Tscale) -> float   (0.0 unless contact_quadratic)
# derived_Tc(params) -> float | None  (or None if not meaningful for that model)
# potential_kind: "linear" | "contact_quadratic" | "saturating_cooperative"
# quadratic_normalization: QUADRATIC_NORMALIZATION for the contact_quadratic
#     models, None for every other kind (it names the m^2/(2N) convention
#     specifically, not "whatever this model divides by")
# potential_normalization: the generic label for what the chain length is used
#     for -- None when N is not needed, else QUADRATIC_NORMALIZATION or
#     Q_NORMALIZATION.  This is what user-facing messages should quote.
# potential_definition: the exact potential, as a string, recorded in the model
#     contract and in every fit output so a fitted parameter can never be
#     reinterpreted against a different convention
# m_ref: reference contact number the potential is measured from; 0 for every
#     model currently defined
# requires_chain_length: True when the normalization needs N

# Exact normalization of the contact-quadratic term.  Recorded in every output
# so a fitted kappa can never be reinterpreted against a different convention.
QUADRATIC_NORMALIZATION = "m^2/(2N)"

# Contact-fraction normalization used by the saturating-cooperative family.
Q_NORMALIZATION = "q = m/N"

# Reference contact number.  Every potential defined here is measured from m = 0,
# so q = (m - m_ref)/N = m/N; recorded explicitly rather than left implicit.
M_REF_DEFAULT = 0

SATURATING_COOPERATIVE_DEFINITION = (
    "u(m,T;N) = N*[b(T)*q - A0*q^2/(1 + (q/q_sat)^2)],  b(T) = h_b/T - s_b,  "
    "q = m/N,  m_ref = 0,  A0 >= 0 and q_sat > 0 (both temperature-independent)"
)


def _q_zero(params: np.ndarray, T: float, Tref: float, Tscale: float) -> float:
    """Quadratic coefficient of a purely linear contact potential: exactly 0."""
    return 0.0


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


def _q_hs_m2_const(params: np.ndarray, T: float, Tref: float, Tscale: float) -> float:
    """Temperature-independent contact-number curvature kappa2."""
    return float(params[2])


def _q_hs_m2_hs(params: np.ndarray, T: float, Tref: float, Tscale: float) -> float:
    """Enthalpy/entropy decomposition of the contact-number curvature."""
    return float(params[2]) / T - float(params[3])


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
    # --- contact-number-quadratic models -----------------------------------
    # These leave b(T) alone and add curvature in m.  Their quadratic parameters
    # default to zero so the first restart begins at the nested hs solution.
    "hs_m2_const": {
        "param_names": ["h1", "s1", "kappa2"],
        "x0": [750.0, 2.8, 0.0],
        "bounds": [(-2000.0, 2000.0), (-10.0, 10.0), (-50.0, 50.0)],
        "raw_b_fn": _b_hs,
        "raw_q_fn": _q_hs_m2_const,
        "derived_Tc": None,
        "potential_kind": "contact_quadratic",
        "quadratic_normalization": QUADRATIC_NORMALIZATION,
        "requires_chain_length": True,
        "description": (
            "u(m,T;N) = (h1/T - s1)*m + kappa2*m^2/(2N)  "
            "(temperature-independent contact-number curvature)"
        ),
    },
    "hs_m2_hs": {
        "param_names": ["h1", "s1", "h2", "s2"],
        "x0": [750.0, 2.8, 0.0, 0.0],
        "bounds": [
            (-2000.0, 2000.0), (-10.0, 10.0), (-2000.0, 2000.0), (-10.0, 10.0),
        ],
        "raw_b_fn": _b_hs,
        "raw_q_fn": _q_hs_m2_hs,
        "derived_Tc": None,
        "potential_kind": "contact_quadratic",
        "quadratic_normalization": QUADRATIC_NORMALIZATION,
        "requires_chain_length": True,
        "description": (
            "u(m,T;N) = (h1/T - s1)*m + (h2/T - s2)*m^2/(2N)  "
            "(enthalpy/entropy decomposition of the curvature)"
        ),
    },
    # --- saturating-cooperative model ---------------------------------------
    # Written in the contact fraction q = m/N rather than in m, so the
    # cooperative attraction saturates: it deepens the well at low contact
    # density but its marginal contribution dies off once q >> q_sat, and the
    # marginal slope du/dm returns to b(T).  A0 = 0 recovers hs exactly, so the
    # first restart begins at the nested linear solution.
    "saturating_cooperative_contact": {
        "param_names": ["h_b", "s_b", "A0", "q_sat"],
        "x0": [750.0, 2.8, 0.0, 0.35],
        "bounds": [
            (-2000.0, 2000.0), (-10.0, 10.0), (0.0, 20.0), (0.02, 2.0),
        ],
        "raw_b_fn": _b_hs,
        # No raw_q_fn: this potential has no m^2/(2N) term, so kappa(T) is
        # identically zero and the _q_zero default is the honest answer.
        "derived_Tc": None,
        "potential_kind": "saturating_cooperative",
        "quadratic_normalization": None,
        "potential_normalization": Q_NORMALIZATION,
        "potential_definition": SATURATING_COOPERATIVE_DEFINITION,
        "m_ref": M_REF_DEFAULT,
        "requires_chain_length": True,
        "description": (
            "u(m,T;N) = N*[(h_b/T - s_b)*q - A0*q^2/(1+(q/q_sat)^2)],  q = m/N  "
            "(saturating cooperative attraction; A0 >= 0, q_sat > 0)"
        ),
    },
}

# Linear models carry no curvature in m and need no chain length.  Filling the
# defaults here keeps the six historical entries textually unchanged.
for _spec in MODEL_REGISTRY.values():
    _spec.setdefault("raw_q_fn", _q_zero)
    _spec.setdefault("potential_kind", "linear")
    _spec.setdefault("quadratic_normalization", None)
    _spec.setdefault("requires_chain_length", False)
    # Every potential defined so far is measured from m = 0.
    _spec.setdefault("m_ref", M_REF_DEFAULT)
    # For the pre-existing models the description already IS the exact potential
    # definition, and m^2/(2N) is the only normalization that ever needed N.
    _spec.setdefault("potential_definition", _spec["description"])
    _spec.setdefault("potential_normalization", _spec["quadratic_normalization"])
del _spec


# Shared model-contract version.  Bump only when the model set, parameter names,
# or contact-potential semantics change in a way that breaks cross-script
# compatibility.
#   v2: registry gains raw_q_fn / potential_kind / quadratic_normalization and
#       the contact-number-quadratic models hs_m2_const and hs_m2_hs; the
#       reweighting weight is exp[-u_contact(m,T;N)] rather than exp[-b(T)*m]
#       (identical for every v1 model, all of which are potential_kind linear).
#   v3: potential_kind gains "saturating_cooperative" and the model
#       saturating_cooperative_contact; the full potential is built by
#       POTENTIAL_BUILDERS[potential_kind] instead of being hard-coded as
#       b*m + kappa*m^2/(2N), and the registry/contract gain
#       potential_definition, potential_normalization and m_ref.  Every v1 and
#       v2 model keeps its parameters, its potential, and its numerical output
#       bit for bit; the four pre-existing contract keys are unchanged, so a
#       v2-era comparator that reads only those keys still works.
MODEL_API_VERSION = 3


def get_model_contract() -> dict:
    """Return a callable-free description of the supported contact-bias models.

    Used to verify that this script and remd_uniform_chain_new.py agree on the
    model API version, model names, parameter ordering, and the exact potential
    each model defines.

    ``potential_definition`` is the authoritative statement of the potential;
    ``quadratic_normalization`` names the m^2/(2N) convention specifically and is
    None for every model that does not use it, while ``potential_normalization``
    is the generic label for whatever the chain length is used for.
    """
    return {
        "model_api_version": MODEL_API_VERSION,
        "models": {
            name: {
                "param_names": list(spec["param_names"]),
                "description": str(spec["description"]),
                "potential_kind": str(spec["potential_kind"]),
                "quadratic_normalization": spec["quadratic_normalization"],
                "potential_definition": str(spec["potential_definition"]),
                "potential_normalization": spec["potential_normalization"],
                "m_ref": int(spec["m_ref"]),
            }
            for name, spec in MODEL_REGISTRY.items()
        },
    }


# ---------------------------------------------------------------------------
# Contact-potential accessors
# ---------------------------------------------------------------------------
# reduced_bias/make_b_fn return the LINEAR coefficient only and are kept for
# backward compatibility; make_contact_u_fn returns the full potential used for
# reweighting.  For every linear model the two agree exactly: u = b(T)*m.

def reduced_bias(
    model_name: str, params: np.ndarray, T: float, Tref: float, Tscale: float
) -> float:
    """b(T), the linear contact coefficient of the selected model."""
    return float(MODEL_REGISTRY[model_name]["raw_b_fn"](params, float(T), Tref, Tscale))


def quadratic_bias(
    model_name: str, params: np.ndarray, T: float, Tref: float, Tscale: float
) -> float:
    """kappa(T), the coefficient of m^2/(2N); exactly 0 for linear models."""
    return float(MODEL_REGISTRY[model_name]["raw_q_fn"](params, float(T), Tref, Tscale))


def make_b_fn(
    model_name: str, Tref: float, Tscale: float
) -> Callable[[np.ndarray, float], float]:
    """Return b(params, T) with Tref and Tscale captured by closure."""
    raw = MODEL_REGISTRY[model_name]["raw_b_fn"]

    def b_fn(params: np.ndarray, T: float) -> float:
        return raw(params, T, Tref, Tscale)

    return b_fn


def make_q_fn(
    model_name: str, Tref: float, Tscale: float
) -> Callable[[np.ndarray, float], float]:
    """Return kappa(params, T) with Tref and Tscale captured by closure."""
    raw = MODEL_REGISTRY[model_name]["raw_q_fn"]

    def q_fn(params: np.ndarray, T: float) -> float:
        return raw(params, T, Tref, Tscale)

    return q_fn


def validate_chain_length(model_name: str, n_beads: Optional[int]) -> Optional[int]:
    """Return a usable chain length, or raise if the model requires one and it is absent."""
    spec = MODEL_REGISTRY[model_name]
    if not spec.get("requires_chain_length"):
        return None if n_beads is None else int(n_beads)
    if n_beads is None:
        raise ValueError(
            f"model {model_name!r} needs a chain length for its "
            f"{spec.get('potential_normalization') or 'normalization'} "
            f"normalization; none was resolved."
        )
    n = int(n_beads)
    if n < 2:
        raise ValueError(f"chain length must be >= 2, got {n_beads!r}")
    return n


def validated_saturating_params(params: np.ndarray) -> Tuple[float, float]:
    """Return (A0, q_sat) for the saturating-cooperative model, or raise.

    The potential is only defined for A0 >= 0 (an attraction, never a repulsion
    dressed up as one) and q_sat > 0 (a saturation scale; q_sat = 0 would divide
    by zero).  The registry bounds already confine the optimizer to that region,
    so this guards direct API use rather than the fit.
    """
    A0 = float(params[2])
    q_sat = float(params[3])
    if not np.isfinite(A0) or not np.isfinite(q_sat):
        raise ValueError(
            f"saturating_cooperative needs finite A0 and q_sat, got "
            f"A0={A0!r}, q_sat={q_sat!r}"
        )
    if A0 < 0.0:
        raise ValueError(f"saturating_cooperative requires A0 >= 0, got {A0!r}")
    if q_sat <= 0.0:
        raise ValueError(f"saturating_cooperative requires q_sat > 0, got {q_sat!r}")
    return A0, q_sat


# ---------------------------------------------------------------------------
# Potential builders
# ---------------------------------------------------------------------------
# One builder per potential_kind.  Each takes the registry spec plus the
# temperature reference/scale and the (already validated) chain length, and
# returns u_fn(params, T, m).  Adding a potential family means adding a builder
# and a registry entry -- no caller of u_fn needs to know which family it got.

def _build_u_linear(
    spec: Dict, Tref: float, Tscale: float, n: Optional[float]
) -> Callable[[np.ndarray, float, np.ndarray], np.ndarray]:
    """u = b(T) * m.  Needs no chain length."""
    raw_b = spec["raw_b_fn"]

    def u_fn_linear(params: np.ndarray, T: float, m: np.ndarray) -> np.ndarray:
        return raw_b(params, T, Tref, Tscale) * np.asarray(m, dtype=float)

    return u_fn_linear


def _build_u_contact_quadratic(
    spec: Dict, Tref: float, Tscale: float, n: Optional[float]
) -> Callable[[np.ndarray, float, np.ndarray], np.ndarray]:
    """u = b(T)*m + kappa(T)*m^2/(2N)."""
    raw_b = spec["raw_b_fn"]
    raw_q = spec["raw_q_fn"]

    def u_fn_quadratic(params: np.ndarray, T: float, m: np.ndarray) -> np.ndarray:
        m_arr = np.asarray(m, dtype=float)
        b = raw_b(params, T, Tref, Tscale)
        q = raw_q(params, T, Tref, Tscale)
        return b * m_arr + q * (m_arr * m_arr) / (2.0 * n)

    return u_fn_quadratic


def _build_u_saturating_cooperative(
    spec: Dict, Tref: float, Tscale: float, n: Optional[float]
) -> Callable[[np.ndarray, float, np.ndarray], np.ndarray]:
    """u = N*[b(T)*q - A0*q^2/(1 + (q/q_sat)^2)] with q = m/N and m_ref = 0.

    The linear part is evaluated as ``b(T)*m``, not as ``N*(b(T)*(m/N))``.  The
    two are the same number in exact arithmetic, but the round trip through m/N
    and back through *N is exactly the avoidable rounding: writing it as b(T)*m
    makes A0 = 0 reproduce the hs potential bit for bit, so this model nests the
    linear one exactly rather than approximately.  Only the cooperative term,
    which has no linear counterpart to agree with, is evaluated in q.
    """
    raw_b = spec["raw_b_fn"]

    def u_fn_saturating(params: np.ndarray, T: float, m: np.ndarray) -> np.ndarray:
        A0, q_sat = validated_saturating_params(params)
        m_arr = np.asarray(m, dtype=float)
        u_lin = raw_b(params, T, Tref, Tscale) * m_arr
        if A0 == 0.0:
            return u_lin
        q = m_arr / n
        r = q / q_sat
        return u_lin - n * A0 * (q * q) / (1.0 + r * r)

    return u_fn_saturating


POTENTIAL_BUILDERS: Dict[str, Callable] = {
    "linear": _build_u_linear,
    "contact_quadratic": _build_u_contact_quadratic,
    "saturating_cooperative": _build_u_saturating_cooperative,
}


def make_contact_u_fn(
    model_name: str,
    Tref: float,
    Tscale: float,
    n_beads: Optional[int] = None,
) -> Callable[[np.ndarray, float, np.ndarray], np.ndarray]:
    """Return u(params, T, m) with Tref, Tscale and N captured by closure.

    The chain length is validated once here rather than inside the reweighting
    loop, so a missing N for a model that needs one fails before any fitting
    starts.  Dispatch is by potential_kind through POTENTIAL_BUILDERS, so every
    reweighting path -- contacts, joint Rg, Rg prediction -- uses one potential.
    """
    spec = MODEL_REGISTRY[model_name]
    kind = str(spec["potential_kind"])
    try:
        build = POTENTIAL_BUILDERS[kind]
    except KeyError:
        raise ValueError(
            f"model {model_name!r} declares potential_kind {kind!r}, which has "
            f"no builder. Known kinds: {sorted(POTENTIAL_BUILDERS)}"
        ) from None
    n = validate_chain_length(model_name, n_beads)
    return build(spec, Tref, Tscale, None if n is None else float(n))


def reduced_contact_potential(
    m,
    T: float,
    model_name: str,
    params: np.ndarray,
    Tref: float,
    Tscale: float,
    n_beads: Optional[int] = None,
):
    """u_contact(m, T; N) for the selected model, elementwise in m.

    Linear models return exactly ``b(T) * m`` -- the same floating-point value
    the pre-quadratic code computed.  Contact-quadratic models add
    ``kappa(T) * m^2 / (2N)``, and the saturating-cooperative model subtracts
    ``N * A0 * q^2/(1 + (q/q_sat)^2)``; both vanish identically at zero curvature
    / zero amplitude, so both nest the linear potential exactly.
    """
    return make_contact_u_fn(model_name, Tref, Tscale, n_beads)(
        params, float(T), m
    )


# ---------------------------------------------------------------------------
# Generic model probability
# ---------------------------------------------------------------------------

def _stabilized_exponent(x: np.ndarray, support: np.ndarray) -> np.ndarray:
    """Shift x so its maximum OVER THE SUPPORTED BINS is 0; mask the rest.

    Bins with zero baseline mass contribute nothing to the partition function,
    so letting one of them set the stabilization constant would deflate every
    surviving weight -- harmless for a linear potential over a narrow m range,
    but able to underflow the whole distribution once u grows quadratically in
    m.  Unsupported bins are then set to -inf rather than left at a large
    positive exponent, which would otherwise overflow exp() and turn the
    0 * inf product into NaN.

    When every bin is supported this is exactly ``x - x.max()``, so linear
    models reproduce their previous weights bit for bit.  Falls back to the
    global maximum when nothing is supported.
    """
    x = np.asarray(x, dtype=float)
    if not np.any(support):
        return x - np.max(x)
    shifted = x - float(np.max(x[support]))
    if np.all(support):
        return shifted
    return np.where(support, shifted, -np.inf)


def model_contact_mass(
    p0_mass: np.ndarray,
    m_centers: np.ndarray,
    T: float,
    params: np.ndarray,
    u_fn: Callable[[np.ndarray, float, np.ndarray], np.ndarray],
) -> np.ndarray:
    """P_model(m|T) ∝ P0(m) * exp[-u_contact(m,T;N)], stabilized on the support."""
    m_centers = np.asarray(m_centers, dtype=float)
    x = _stabilized_exponent(
        -np.asarray(u_fn(params, float(T), m_centers), dtype=float),
        np.asarray(p0_mass, dtype=float) > 0.0,
    )
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
    u_fn: Callable[[np.ndarray, float, np.ndarray], np.ndarray],
    loss_fn: Callable[[np.ndarray, np.ndarray], float],
) -> float:
    """Sum of per-temperature contact loss over the provided temperatures."""
    total = 0.0
    for i, T in enumerate(temps):
        p_mod = model_contact_mass(p0_mass, m_centers, float(T), params, u_fn)
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
    u_fn: Callable[[np.ndarray, float, np.ndarray], np.ndarray],
    loss_fn: Callable[[np.ndarray, np.ndarray], float],
    rg_weight: float,
) -> float:
    """Contact loss + rg_weight * Rg loss, summed over training temperatures."""
    m_joint = 0.5 * (c_edges_joint[:-1] + c_edges_joint[1:])
    joint_support = np.asarray(crg_prob, dtype=float).sum(axis=1) > 0.0
    total = 0.0
    for i, T in enumerate(train_temps):
        # contact term
        p_mod_ct = model_contact_mass(p0_mass, m_centers, float(T), params, u_fn)
        total += loss_fn(p_obs_ct_train[i], p_mod_ct)
        # Rg term: reweight joint baseline by exp[-u(m,T;N)], marginalize over m
        x = _stabilized_exponent(
            -np.asarray(u_fn(params, float(T), m_joint), dtype=float), joint_support
        )
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
    u_fn: Callable[[np.ndarray, float, np.ndarray], np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """Predict P(Rg|T) for all temps by reweighting P0(m,Rg) in m.

    Returns (rg_centers, rg_mod_mass) where rg_mod_mass has shape (n_temps, n_rg).
    """
    c_edges = np.asarray(c_edges, dtype=float)
    rg_edges = np.asarray(rg_edges, dtype=float)
    crg_prob = np.asarray(crg_prob, dtype=float)

    m_centers = 0.5 * (c_edges[:-1] + c_edges[1:])
    joint_support = crg_prob.sum(axis=1) > 0.0
    rg_mass_T = np.zeros((temps.size, rg_edges.size - 1), dtype=float)

    for i, T in enumerate(temps):
        x = _stabilized_exponent(
            -np.asarray(u_fn(params, float(T), m_centers), dtype=float), joint_support
        )
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
    u_fn: Callable[[np.ndarray, float, np.ndarray], np.ndarray],
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
            u_fn, loss_fn, float(rg_weight),
        )
        return objective_combined, obj_args
    return objective, (train_temps, m_centers, p_obs_ct_train, p0_mass, u_fn, loss_fn)


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
    u_fn: Callable[[np.ndarray, float, np.ndarray], np.ndarray],
    loss_fn: Callable[[np.ndarray, np.ndarray], float],
) -> np.ndarray:
    """Per-temperature contact loss; summing over a subset reproduces objective()."""
    out = np.empty(len(temps), dtype=float)
    for i, T in enumerate(temps):
        p_mod = model_contact_mass(p0_mass, m_centers, float(T), params, u_fn)
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
    u_fn = ctx["u_fn"]
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
            fit_rg, train_temps, m_centers, p_obs_ct_train, p0_mass, u_fn, loss_fn,
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
            temps, m_centers, p_obs_mass, p0_mass, params, u_fn, loss_fn
        )
        rg_pt = None
        if can_fit_rg:
            _, rg_mod_mass = predict_rg_from_joint(
                crg_prob, c_edges_joint, rg_edges_model_lattice, temps, params, u_fn
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
    N_QUICK = 30                      # chain length for the contact-quadratic models
    Tref, Tscale = float(temps.mean()), float(temps.max() - temps.min())
    b_hs = make_b_fn("hs", Tref, Tscale)
    u_hs = make_contact_u_fn("hs", Tref, Tscale)
    p_obs_mass = np.zeros((temps.size, m_centers.size), dtype=float)
    for i, T in enumerate(temps):
        p = model_contact_mass(p0_mass, m_centers, float(T), np.array([h_true, s_true]), u_hs)
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
        u_fn = make_contact_u_fn(model, Tref, Tscale, n_beads=N_QUICK)
        obj_fn, obj_args = build_objective(
            False, temps[train_idx], m_centers, p_obs_mass[train_idx], p0_mass, u_fn, loss_fn
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
    ufn2 = make_contact_u_fn("hs", Tref2, Tscale2)
    A = np.array([900.0, 3.0]); R = np.array([500.0, 1.5])  # conflicting truths
    obs_ct = np.array([model_contact_mass(p0, mc, float(T), A, ufn2) for T in t2])
    obs_rg = np.array([model_contact_mass(p0, mc, float(T), R, ufn2) for T in t2])
    c_edges2 = np.arange(-0.5, M + 0.5, 1.0)       # identity Rg = contact mapping
    rg_edges2 = np.arange(-0.5, M + 0.5, 1.0)
    crg2 = np.zeros((M, M)); crg2[np.arange(M), np.arange(M)] = p0
    lfn = _get_loss_fn("js")
    spec_hs = MODEL_REGISTRY["hs"]

    def fit_and_score(w: float):
        use_rg = w > 0
        of, oa = build_objective(
            use_rg, t2, mc, obs_ct, p0, ufn2, lfn,
            crg_prob=crg2, c_edges_joint=c_edges2, p_obs_rg_train=obs_rg, rg_weight=w,
        )
        best, _ = fit_one_split(of, oa, make_x0s("hs", 0), spec_hs["bounds"])
        pr = best.x
        ct = float(per_temp_contact_losses(t2, mc, obs_ct, p0, pr, ufn2, lfn).sum())
        _, rgm = predict_rg_from_joint(crg2, c_edges2, rg_edges2, t2, pr, ufn2)
        rg = float(per_temp_rg_losses(rgm, obs_rg, lfn).sum())
        return ct, rg

    c0, r0 = fit_and_score(0.0)
    c1, r1 = fit_and_score(6.0)
    print(f"    w=0: contact={c0:.4g} rg={r0:.4g}  |  w=6: contact={c1:.4g} rg={r1:.4g}")
    check(r1 < r0, "higher Rg weight reduces Rg loss")
    check(c1 > c0, "higher Rg weight worsens contact loss (genuine trade-off)")

    print("Quick test 11: contact-mode fit is unchanged (backward compatibility)")
    # Tests the SCIENCE, not the optimizer build: that the contact-only path still
    # recovers the known synthetic truth and drives the objective to ~0. Exact
    # golden constants (1e-9..1e-20 on optimizer output) were removed because they
    # encode the arithmetic of one SciPy/BLAS build and break on another even when
    # the implementation is correct — a false alarm that hides real regressions.
    rec_bc = fit_model_on("hs", all_idx, seed=123)
    check(
        abs(rec_bc[0] - h_true) < 0.01 and abs(rec_bc[1] - s_true) < 1e-4,
        f"contact-only hs fit recovers the known synthetic parameters "
        f"(h={rec_bc[0]:.6g} vs {h_true}, s={rec_bc[1]:.6g} vs {s_true})",
    )
    obj_bc = objective(rec_bc, temps, m_centers, p_obs_mass, p0_mass, u_hs, loss_fn)
    check(
        obj_bc < 1e-8,
        f"contact-only synthetic objective remains negligible: {obj_bc:.6e}",
    )

    # Compare the OLD and NEW mathematical pathways directly, which is the claim
    # the golden constants were standing in for: model_contact_mass() is what the
    # contact objective consumes, and the joint path must reduce to it exactly when
    # the joint baseline is the contact baseline crossed with a single Rg bin.
    p0_single = p0_mass[:, None]                       # one Rg bin -> Rg carries no info
    ce_single = centers_to_edges(m_centers)
    re_single = np.array([0.0, 1.0])
    _, joint_rg_mass = predict_rg_from_joint(
        p0_single, ce_single, re_single, temps, rec_bc, u_hs
    )
    check(
        np.allclose(joint_rg_mass, 1.0),
        "degenerate single-Rg-bin joint reweighting normalizes to unit mass",
    )
    contact_direct = np.array(
        [model_contact_mass(p0_mass, m_centers, float(T), rec_bc, u_hs) for T in temps]
    )
    joint_contact = np.zeros_like(contact_direct)
    for i, T in enumerate(temps):
        w = np.exp(-b_hs(rec_bc, float(T)) * m_centers)
        w = w * p0_mass
        joint_contact[i] = w / w.sum()
    check(
        np.allclose(contact_direct, joint_contact, rtol=1e-12, atol=1e-15),
        "model_contact_mass agrees with explicit exp[-b m] reweighting of P0(m) "
        "(legacy linear equivalence)",
    )

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
    ufn_t = make_contact_u_fn("tc_scale", float(t3.mean()), float(t3.max() - t3.min()))
    pr = np.array([2.0, 320.0])
    lat, obs, mass = predict_rg_summary_from_joint(
        crg, c_edges, rg_edges, t3, pr, ufn_t,
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
        pr, t3, lat, crg, c_edges, rg_edges, ufn_t,
        n_model_params=2, fixed_rg_scale=0.345, fit_rg_scale=False,
        rg_summary="rms", target_units="lattice", loss_name="mse",
    )
    o_b = objective_rg_scalar(
        pr, t3, lat, crg, c_edges, rg_edges, ufn_t,
        n_model_params=2, fixed_rg_scale=0.999, fit_rg_scale=False,
        rg_summary="rms", target_units="lattice", loss_name="mse",
    )
    sub_check(o_a == o_b == 0.0,
              "target_units=lattice ignores rg_scale inside the loss (and is exact)")
    o_obs = objective_rg_scalar(
        pr, t3, obs, crg, c_edges, rg_edges, ufn_t,
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
    ufn_s = make_contact_u_fn("tc_scale", Tref_s, Tscale_s)
    true_p = np.array([3.0, 315.0])
    true_scale = 0.345
    _, tgt_obs, _ = predict_rg_summary_from_joint(
        crg_s, ce_s, re_s, temps_s, true_p, ufn_s,
        rg_scale=true_scale, summary="rms", target_units="observed",
    )
    cfg_s = {
        "crg_prob": crg_s, "c_edges": ce_s, "rg_edges_lattice": re_s,
        "b_fn": bfn_s, "q_fn": make_q_fn("tc_scale", Tref_s, Tscale_s),
        "u_fn": ufn_s, "n_beads": None,
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
        sub_check(bool(fs["warnings"]),
                  "unreachable target produces an explicit warning")
        # This target sits 3x above the whole baseline Rg support, so the strongest
        # correct objection is the support failure, not a scan-scoped one.
        sub_check(fs["reachability_status"] == "zero_support_overlap",
                  "a target far outside the baseline Rg support is reported as a "
                  "support failure, the strongest correct objection")
        sub_check(any("does not intersect" in w for w in fs["warnings"]),
                  "the support failure is stated explicitly")
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
        crg_s, ce_s, re_s, true_p, ufn_s, 270.0, 360.0,
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

    local_failures.extend(_run_rg_regression_tests(check, sub_check))
    return local_failures


def _scalar_args(**overrides: Any) -> argparse.Namespace:
    """A fully-defaulted scalar-mode Namespace built from the real CLI parser."""
    args = build_arg_parser().parse_args([])
    for k, v in overrides.items():
        if not hasattr(args, k):
            raise AttributeError(f"unknown CLI dest {k!r} in test override")
        setattr(args, k, v)
    return args


def _write_scalar_inputs(
    tdp: Path,
    temps: np.ndarray,
    targets: np.ndarray,
    crg: np.ndarray,
    c_edges: np.ndarray,
    rg_edges: np.ndarray,
    *,
    half_width: float = 0.02,
    name: str = "single.dat",
    baseline: str = "baseline.npz",
) -> Tuple[str, str]:
    """Write a scalar .dat and a joint baseline NPZ; return (dat_path, npz_path)."""
    dat = tdp / name
    lines = [
        f"{t:.8g} {v:.8g} {v * (1.0 - half_width):.8g} {v * (1.0 + half_width):.8g}"
        for t, v in zip(temps, targets)
    ]
    dat.write_text("# T Rg lo hi\n" + "\n".join(lines) + "\n")
    npz = tdp / baseline
    np.savez_compressed(npz, crg_prob=crg, c_edges=c_edges, rg_edges=rg_edges)
    return str(dat), str(npz)


def _run_rg_regression_tests(
    check: Callable[[bool, str], None],
    sub_check: Callable[[bool, str], None],
) -> List[str]:
    """Regression tests for the scale/feasibility/validity fixes."""
    import tempfile

    local_failures: List[str] = []

    def rcheck(cond: bool, msg: str) -> None:
        check(cond, msg)
        if not cond:
            local_failures.append(msg)

    crg_s, ce_s, re_s = _make_synthetic_joint(coupling=1.0)
    temps_s = np.linspace(270.0, 360.0, 12)
    Tref_s, Tscale_s = float(temps_s.mean()), float(temps_s.max() - temps_s.min())
    bfn_s = make_b_fn("tc_scale", Tref_s, Tscale_s)
    ufn_s = make_contact_u_fn("tc_scale", Tref_s, Tscale_s)
    true_p = np.array([3.0, 315.0])

    # ---------------------------------------------------------------- test G --
    print("Scalar-Rg test 12: joint baseline validation rejects malformed input")
    good_args = (ce_s, re_s, crg_s)
    try:
        validate_joint_baseline(*good_args)
        ok_valid = True
    except ValueError:
        ok_valid = False
    rcheck(ok_valid, "a valid joint baseline passes validation")

    def _rejects(c_e: np.ndarray, r_e: np.ndarray, p: np.ndarray, label: str) -> bool:
        try:
            validate_joint_baseline(c_e, r_e, p)
        except ValueError:
            return True
        return False

    ce_nan = ce_s.copy(); ce_nan[2] = np.nan
    re_nan = re_s.copy(); re_nan[3] = np.nan
    crg_nan = crg_s.copy(); crg_nan[1, 1] = np.nan
    crg_neg = crg_s.copy(); crg_neg[2, 2] = -0.5
    crg_zero = np.zeros_like(crg_s)
    ce_nonmono = ce_s.copy(); ce_nonmono[5] = ce_nonmono[3]
    re_nonmono = re_s.copy(); re_nonmono[5] = re_nonmono[3]

    rcheck(_rejects(ce_nan, re_s, crg_s, "c"), "NaN in c_edges is rejected")
    rcheck(_rejects(ce_s, re_nan, crg_s, "r"), "NaN in rg_edges is rejected")
    rcheck(_rejects(ce_s, re_s, crg_nan, "p"), "NaN in crg_prob is rejected")
    rcheck(_rejects(ce_s, re_s, crg_neg, "n"), "negative probability is rejected")
    rcheck(_rejects(ce_s, re_s, crg_zero, "z"), "zero total mass is rejected")
    rcheck(_rejects(ce_s, re_s, crg_s[:-1], "s"), "shape mismatch is rejected")
    rcheck(_rejects(ce_nonmono, re_s, crg_s, "m"), "non-monotonic c_edges is rejected")
    rcheck(_rejects(ce_s, re_nonmono, crg_s, "m"), "non-monotonic rg_edges is rejected")
    rcheck(
        _rejects(ce_s.reshape(-1, 1), re_s, crg_s, "d"), "2-D c_edges is rejected"
    )
    rcheck(_rejects(ce_s, re_s, crg_s.ravel(), "d"), "1-D crg_prob is rejected")

    # A NaN edge must not sneak through the np.diff(edges) <= 0 formulation.
    rcheck(
        not np.any(np.diff(ce_nan) <= 0),
        "np.diff(edges) <= 0 alone does NOT catch a NaN edge (why order matters)",
    )

    # Tiny roundoff negatives are tolerated, clipped, and reported.
    crg_tiny = crg_s.copy(); crg_tiny[0, 0] = -1e-16
    san, notes = sanitize_joint_baseline(ce_s, re_s, crg_tiny)
    rcheck(
        np.all(san >= 0.0) and abs(san.sum() - 1.0) < 1e-12 and len(notes) >= 1,
        "tiny negative roundoff is clipped, normalized, and reported",
    )

    # ---------------------------------------------------------------- test E --
    print("Scalar-Rg test 13: endpoint b -> +/-inf limits match the limiting slices")
    for summ in ("mean", "rms"):
        lim = endpoint_rg_limits(
            crg_s, ce_s, re_s, summary=summ, rg_scale=0.345
        )
        rg_centers_s = 0.5 * (re_s[:-1] + re_s[1:])
        cmarg = crg_s.sum(axis=1)
        nzb = np.flatnonzero(cmarg > 0.0)
        lo_bin, hi_bin = int(nzb[0]), int(nzb[-1])
        exp_pos = float(rg_summary_from_mass(
            (crg_s[lo_bin] / crg_s[lo_bin].sum())[None, :], rg_centers_s, summ)[0])
        exp_neg = float(rg_summary_from_mass(
            (crg_s[hi_bin] / crg_s[hi_bin].sum())[None, :], rg_centers_s, summ)[0])
        rcheck(
            abs(lim["rg_limit_b_pos_inf_lattice"] - exp_pos) < 1e-12,
            f"[{summ}] b->+inf limit equals the min-contact conditional slice summary",
        )
        rcheck(
            abs(lim["rg_limit_b_neg_inf_lattice"] - exp_neg) < 1e-12,
            f"[{summ}] b->-inf limit equals the max-contact conditional slice summary",
        )
        rcheck(
            lim["endpoint_limit_min_lattice"] == min(exp_pos, exp_neg)
            and lim["endpoint_limit_max_lattice"] == max(exp_pos, exp_neg),
            f"[{summ}] endpoint interval uses min()/max(), not an assumed ordering",
        )
        rcheck(
            abs(lim["rg_limit_b_pos_inf_observed"]
                - 0.345 * lim["rg_limit_b_pos_inf_lattice"]) < 1e-12,
            f"[{summ}] observed endpoint limit = rg_scale * lattice limit",
        )

    # A large finite bias must approach the endpoint limit.
    m_c = 0.5 * (ce_s[:-1] + ce_s[1:])
    r_c = 0.5 * (re_s[:-1] + re_s[1:])
    lim_rms = endpoint_rg_limits(crg_s, ce_s, re_s, summary="rms", rg_scale=1.0)
    big_pos = joint_reweight_stats(crg_s, m_c, r_c, 200.0, "rms")["pred_rg_lattice"]
    big_neg = joint_reweight_stats(crg_s, m_c, r_c, -200.0, "rms")["pred_rg_lattice"]
    rcheck(
        abs(big_pos - lim_rms["rg_limit_b_pos_inf_lattice"]) < 1e-6,
        "b=+200 numerically approaches the exact b->+inf limit",
    )
    rcheck(
        abs(big_neg - lim_rms["rg_limit_b_neg_inf_lattice"]) < 1e-6,
        "b=-200 numerically approaches the exact b->-inf limit",
    )
    # The ordering of the two limits flips with the sign of the contact-Rg coupling,
    # which is exactly why min()/max() is required.
    crg_anti, ce_a, re_a = _make_synthetic_joint(coupling=-1.0)
    lim_anti = endpoint_rg_limits(crg_anti, ce_a, re_a, summary="rms", rg_scale=1.0)
    rcheck(
        (lim_rms["rg_limit_b_pos_inf_lattice"] > lim_rms["rg_limit_b_neg_inf_lattice"])
        != (lim_anti["rg_limit_b_pos_inf_lattice"]
            > lim_anti["rg_limit_b_neg_inf_lattice"]),
        "limit ordering flips with the sign of the contact-Rg coupling",
    )

    # ---------------------------------------------------------------- test S --
    # joint_reweight_stats must stabilize the exponent over the SUPPORTED contact
    # bins, not the global maximum.  With zero-probability endpoint bins padded
    # onto the contact axis and a strong bias, an unsupported padded bin would set
    # the maximum and underflow every supported weight to 0 (Z == 0), raising a
    # false "degenerate" error.  The support-aware stabilization returns finite
    # statistics with the mass concentrated on the real (supported) bins.
    print("Scalar-Rg test 13e: joint_reweight_stats stabilizes over supported bins")
    pad = 6
    n_core = crg_s.shape[0]
    crg_pad = np.zeros((n_core + 2 * pad, crg_s.shape[1]), dtype=float)
    crg_pad[pad:pad + n_core] = crg_s          # zero-prob endpoint bins both sides
    ce_pad = np.arange(-0.5, crg_pad.shape[0] + 0.5, 1.0)
    m_pad = 0.5 * (ce_pad[:-1] + ce_pad[1:])
    r_c_pad = 0.5 * (re_s[:-1] + re_s[1:])
    strong_b = 200.0                           # strong positive bias favors low m
    # The unfixed stabilization (global max, set by the m=0 padded bin) underflows
    # every supported weight for this bias -> Z == 0 -> ValueError.
    st_pad = joint_reweight_stats(crg_pad, m_pad, r_c_pad, strong_b, "rms")
    rcheck(
        all(np.isfinite(v) for v in st_pad.values()),
        "joint_reweight_stats returns finite stats with padded zero-prob bins and "
        "a strong bias (no false 'degenerate' error)",
    )
    rcheck(
        m_pad[pad - 1] < st_pad["mean_contacts"] < m_pad[pad + n_core],
        "reweighted mean contacts lands inside the supported bins, not on a padded "
        "zero-mass bin",
    )
    # A fully supported baseline is untouched: support-aware stabilization reduces
    # to x - x.max() when every bin carries mass.
    st_full = joint_reweight_stats(crg_s, m_c, r_c, strong_b, "rms")
    rcheck(
        np.isfinite(st_full["pred_rg_lattice"]),
        "fully supported strong-bias reweighting is unchanged (still finite)",
    )
    # The deprecated alias must keep working and keep mirroring the endpoints.
    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        lim_dep = asymptotic_rg_limits(crg_s, ce_s, re_s, summary="rms", rg_scale=1.0)
    rcheck(
        any(issubclass(w.category, DeprecationWarning) for w in caught)
        and lim_dep["asymptotic_reachable_rg_min"]
        == lim_rms["endpoint_limit_min_lattice"]
        and lim_dep["deprecated"] is True,
        "asymptotic_rg_limits() still works, warns DeprecationWarning, mirrors endpoints",
    )

    # ---------------------------------------------------------------- test J --
    # §9: on a MONOTONIC baseline the endpoint limits must coincide with the
    # extrema of the global outer bound -- but that is a property of this
    # baseline, established by test, not an assumption about baselines generally
    # (test K below exhibits one where it fails).
    print("Scalar-Rg test 13b: monotonic baseline -> endpoints == global outer bound")
    for summ in ("mean", "rms"):
        for label, (cj, cej, rej) in (
            ("coupling=+1", (crg_s, ce_s, re_s)),
            ("coupling=-1", (crg_anti, ce_a, re_a)),
        ):
            ep_j = endpoint_rg_limits(cj, cej, rej, summary=summ, rg_scale=1.0)
            gob_j = global_rg_outer_bounds(cj, cej, rej, summary=summ, rg_scale=1.0)
            rcheck(
                ep_j["conditional_moment_monotonic"] is True,
                f"[{summ}, {label}] synthetic coupled baseline has a monotonic "
                f"conditional moment ({ep_j['conditional_moment_direction']})",
            )
            rcheck(
                abs(ep_j["endpoint_limit_min_lattice"]
                    - gob_j["global_outer_rg_min_lattice"]) < 1e-12
                and abs(ep_j["endpoint_limit_max_lattice"]
                        - gob_j["global_outer_rg_max_lattice"]) < 1e-12,
                f"[{summ}, {label}] monotonic -> endpoint limits ARE the outer-bound "
                f"extrema",
            )
            rcheck(
                ep_j["is_exact_extremal_bound"] is True,
                f"[{summ}, {label}] monotonic baseline sets is_exact_extremal_bound",
            )

    # -------------------------------------------------------------- test J2 --
    # BUGFIX 1: the exact global outer bound must include EVERY strictly-positive-
    # mass contact bin, with no probability threshold. A bin with mass 1e-16
    # dominates under strong exponential reweighting, so dropping it below a 1e-15
    # threshold would shrink the bound and manufacture a false impossibility.
    print("Scalar-Rg test 13d: outer bound includes every positive-mass contact bin")
    ce_tiny = np.array([-0.5, 0.5, 1.5])
    re_tiny = np.array([0.5, 1.5, 8.5, 9.5])          # centers 1.0, 5.0, 9.0
    crg_tiny = np.zeros((2, 3))
    crg_tiny[0, 0] = 1.0 - 1e-16                        # m=0 -> Rg=1, almost all mass
    crg_tiny[1, 2] = 1e-16                              # m=1 -> Rg=9, vanishing mass
    for summ in ("mean", "rms"):
        gob_tiny = global_rg_outer_bounds(
            crg_tiny, ce_tiny, re_tiny, summary=summ, rg_scale=1.0,
            rg_target_lattice=np.array([9.0]),
        )
        rcheck(
            abs(gob_tiny["global_outer_rg_min_lattice"] - 1.0) < 1e-9
            and abs(gob_tiny["global_outer_rg_max_lattice"] - 9.0) < 1e-9,
            f"[{summ}] outer bound is [1, 9] despite the 1e-16 bin "
            f"(got [{gob_tiny['global_outer_rg_min_lattice']:.4g}, "
            f"{gob_tiny['global_outer_rg_max_lattice']:.4g}])",
        )
        rcheck(
            gob_tiny["target_within_global_outer_bound"] is True,
            f"[{summ}] a target of 9 is NOT classified as impossible",
        )
        rcheck(
            gob_tiny["probability_tolerance"] == 0.0,
            f"[{summ}] the bound reports a zero probability threshold "
            f"(got {gob_tiny['probability_tolerance']!r})",
        )
        # The default 1e-15 argument must NOT be able to drop the bin either.
        gob_dflt = global_rg_outer_bounds(
            crg_tiny, ce_tiny, re_tiny, summary=summ, rg_scale=1.0,
            probability_tolerance=1e-15, rg_target_lattice=np.array([9.0]),
        )
        rcheck(
            abs(gob_dflt["global_outer_rg_max_lattice"] - 9.0) < 1e-9
            and gob_dflt["target_within_global_outer_bound"] is True,
            f"[{summ}] an explicit 1e-15 threshold still cannot drop the 1e-16 bin",
        )
    # And through the full scan: status must not be outside_global_outer_bound.
    with tempfile.TemporaryDirectory() as td_tiny:
        fs_tiny = run_rg_feasibility_scan(
            crg_tiny, ce_tiny, re_tiny,
            np.array([9.0, 9.0]), np.array([9.0, 9.0]),
            rg_scale=1.0, summary="mean", bias_min=-50.0, bias_max=50.0,
            bias_points=201, outdir=Path(td_tiny), make_plots=False,
        )
        rcheck(
            fs_tiny["global_outer_bound"]["target_within_global_outer_bound"] is True
            and fs_tiny["reachability_status"] != "outside_global_outer_bound",
            f"a target of 9 is not called all-b impossible via the scan "
            f"(status={fs_tiny['reachability_status']})",
        )

    # ---------------------------------------------------------------- test K --
    # §8: the counterexample that breaks the endpoint-only reasoning. Both b->+/-inf
    # limits are 1, yet b=0 gives ~8.84 (mean) / ~8.91 (rms). An implementation that
    # treats the endpoint interval as the all-b reachable range calls a target of 8.5
    # impossible when it is in fact reached at b=0.
    print("Scalar-Rg test 13c: nonmonotonic baseline -> no false all-b impossibility")
    ce_nm = np.array([-0.5, 0.5, 1.5, 2.5])
    re_nm = np.array([0.5, 1.5, 8.5, 9.5])          # centers 1.0, 5.0, 9.0
    crg_nm = np.zeros((3, 3))
    crg_nm[0, 0] = 0.01                              # m=0 -> Rg=1
    crg_nm[1, 2] = 0.98                              # m=1 -> Rg=9
    crg_nm[2, 0] = 0.01                              # m=2 -> Rg=1
    m_nm = 0.5 * (ce_nm[:-1] + ce_nm[1:])
    r_nm = 0.5 * (re_nm[:-1] + re_nm[1:])

    for summ, expect_at0 in (("mean", 8.84), ("rms", 8.910667)):
        ep_nm = endpoint_rg_limits(crg_nm, ce_nm, re_nm, summary=summ, rg_scale=1.0)
        gob_nm = global_rg_outer_bounds(
            crg_nm, ce_nm, re_nm, summary=summ, rg_scale=1.0
        )
        at0 = joint_reweight_stats(crg_nm, m_nm, r_nm, 0.0, summ)["pred_rg_lattice"]

        rcheck(
            abs(ep_nm["endpoint_limit_min_lattice"] - 1.0) < 1e-12
            and abs(ep_nm["endpoint_limit_max_lattice"] - 1.0) < 1e-12,
            f"[{summ}] both endpoint limits equal 1 on the counterexample",
        )
        rcheck(
            abs(at0 - expect_at0) < 1e-4,
            f"[{summ}] finite-b prediction at b=0 is {expect_at0:.4g}, not the "
            f"endpoint value (got {at0:.6g})",
        )
        rcheck(at0 > 8.0, f"[{summ}] b=0 prediction exceeds 8 while endpoints are 1")
        rcheck(
            ep_nm["conditional_moment_monotonic"] is False
            and ep_nm["conditional_moment_direction"] == "nonmonotonic",
            f"[{summ}] conditional moment is detected as nonmonotonic",
        )
        rcheck(
            ep_nm["is_exact_extremal_bound"] is False,
            f"[{summ}] nonmonotonic -> endpoints NOT advertised as extremal",
        )
        rcheck(
            abs(gob_nm["global_outer_rg_max_lattice"] - 9.0) < 1e-9,
            f"[{summ}] global outer max is ~9 (the m=1 conditional slice), got "
            f"{gob_nm['global_outer_rg_max_lattice']:.6g}",
        )
        rcheck(
            gob_nm["global_outer_rg_min_lattice"] <= at0
            <= gob_nm["global_outer_rg_max_lattice"],
            f"[{summ}] the true b=0 value lies INSIDE the global outer bound",
        )
        rcheck(
            gob_nm["is_exact_reachable_range"] is False
            and "does not prove" in gob_nm["note"],
            f"[{summ}] outer bound is labelled necessary-only, not exact",
        )

        # A target of ~8.5 is genuinely reachable near b=0. The scan must not call
        # it impossible, and validity must not report an all-b objection.
        tgt_nm = np.array([8.5, 8.5, 8.5])
        gob_t = global_rg_outer_bounds(
            crg_nm, ce_nm, re_nm, summary=summ, rg_scale=1.0,
            rg_target_lattice=tgt_nm,
        )
        rcheck(
            gob_t["target_within_global_outer_bound"] is True,
            f"[{summ}] target 8.5 is inside the global outer bound",
        )
        with tempfile.TemporaryDirectory() as td_nm:
            fs_nm = run_rg_feasibility_scan(
                crg_nm, ce_nm, re_nm, tgt_nm, tgt_nm,
                rg_scale=1.0, summary=summ, bias_min=-10.0, bias_max=10.0,
                bias_points=201, outdir=Path(td_nm), make_plots=False,
            )
            rcheck(
                fs_nm["reachability_status"] != "outside_global_outer_bound",
                f"[{summ}] target reachable at b=0 is NOT called all-b impossible "
                f"(status={fs_nm['reachability_status']})",
            )
            rcheck(
                fs_nm["target_within_reachable_range"] is True,
                f"[{summ}] target 8.5 is reached within a scan containing b=0",
            )
            rcheck(
                fs_nm["endpoint_limits"]["conditional_moment_monotonic"] is False
                and fs_nm["global_outer_bound"]["target_within_global_outer_bound"]
                is True,
                f"[{summ}] scan reports nonmonotonic moment and in-bound target",
            )
            val_nm = classify_scientific_validity(
                fs_nm,
                rg_support_overlap(tgt_nm, tgt_nm, 0.5, 9.5),
                is_fitted_scale=False,
            )
            rcheck(
                val_nm["status"] != "outside_global_outer_bound",
                f"[{summ}] validity does not place a b=0-reachable target outside "
                f"the all-b model range (status={val_nm['status']})",
            )
            rcheck(
                not any("cannot be reproduced by any real contact bias" in w
                        for w in fs_nm["warnings"]),
                f"[{summ}] no false all-b impossibility warning is emitted",
            )

    # ---------------------------------------------------------------- test H --
    print("Scalar-Rg test 14: a no-collapse curve reports no transition temperature")
    # coupling=0 makes Rg independent of contacts, so b(T) cannot move Rg at all.
    crg_flat, ce_f, re_f = _make_synthetic_joint(coupling=0.0)
    tm_flat = rg_curve_transition_metrics(
        crg_flat, ce_f, re_f, true_p, ufn_s, 270.0, 360.0,
        rg_scale=0.345, summary="rms", target_units="observed", n_grid=1001,
    )
    rcheck(tm_flat["collapse_detected"] is False,
           "flat Rg(T) curve sets collapse_detected=False")
    rcheck(tm_flat["T_rg_max_slope"] is None,
           "flat Rg(T) curve sets T_rg_max_slope=None")
    rcheck(tm_flat["slope_tolerance"] > 0.0, "slope_tolerance is reported and positive")

    # An EXPANDING curve (Rg grows with T) must also report no collapse: the
    # anti-coupled baseline with the same b(T) expands instead of collapsing.
    tm_exp = rg_curve_transition_metrics(
        crg_anti, ce_a, re_a, true_p, ufn_s, 270.0, 360.0,
        rg_scale=0.345, summary="rms", target_units="observed", n_grid=1001,
    )
    rcheck(
        tm_exp["curve"][-1] > tm_exp["curve"][0],
        "the anti-coupled synthetic curve genuinely expands with temperature",
    )
    rcheck(
        tm_exp["collapse_detected"] is False and tm_exp["T_rg_max_slope"] is None,
        "expanding Rg(T) curve reports no collapse and no T_rg_max_slope",
    )
    # The real collapse case must still be detected.
    tm_real = rg_curve_transition_metrics(
        crg_s, ce_s, re_s, true_p, ufn_s, 270.0, 360.0,
        rg_scale=0.345, summary="rms", target_units="observed", n_grid=1001,
    )
    rcheck(
        tm_real["collapse_detected"] is True and tm_real["T_rg_max_slope"] is not None,
        "a genuine collapse is still detected (no false negative)",
    )

    # ---------------------------------------------------------------- test D --
    print("Scalar-Rg test 15: finite-scan wording makes no unsupported all-b claim")
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        reach = np.array([
            joint_reweight_stats(crg_s, m_c, r_c, float(b), "rms")["pred_rg_lattice"]
            for b in np.linspace(-10, 10, 401)
        ])
        # Just outside the scanned range but still inside the global outer bound, so
        # only the finite-scan warning fires and its wording can be checked alone.
        lim_scan = endpoint_rg_limits(crg_s, ce_s, re_s, summary="rms", rg_scale=1.0)
        gob_scan = global_rg_outer_bounds(
            crg_s, ce_s, re_s, summary="rms", rg_scale=1.0
        )
        gap_lo = float(gob_scan["global_outer_rg_min_lattice"])
        just_out = float(reach.min()) - 0.25 * (float(reach.min()) - gap_lo)
        tgt_scan = np.array([just_out] * 3)
        fs_scan = run_rg_feasibility_scan(
            crg_s, ce_s, re_s, tgt_scan, 0.345 * tgt_scan,
            rg_scale=0.345, summary="rms", bias_min=-10.0, bias_max=10.0,
            bias_points=401, outdir=tdp, make_plots=False,
        )
        joined = " ".join(fs_scan["warnings"])
        # This baseline's Rg(b) is monotonic, so a target below the scanned minimum
        # is necessarily closest to a bias at the scan BOUNDARY: the honest verdict
        # is "inconclusive, widen it", not a settled miss. A settled
        # unreachable_within_scan needs an interior extremum -- checked below with
        # the nonmonotonic baseline.
        rcheck(
            fs_scan["reachability_status"] == "boundary_limited",
            f"a monotonic-curve target below the scanned range is boundary_limited "
            f"(got {fs_scan['reachability_status']})",
        )
        rcheck(
            "scanned bias interval" in joined,
            "the warning scopes itself to the scanned bias interval",
        )
        rcheck(
            "[-10, 10]" in joined,
            f"the warning states the actual interval: {joined[:0] or '[-10, 10]'}",
        )
        rcheck(
            "any bias" not in joined and "NO value of b(T)" not in joined,
            "no warning claims 'any bias' or 'NO value of b(T)' from a finite scan",
        )

        # A genuine settled unreachable_within_scan: the nonmonotonic baseline's
        # Rg(b) peaks at ~8.84 in the scan INTERIOR, so a target just above the peak
        # but below the outer bound (9) is missed away from the boundary.
        reach_nm = np.array([
            joint_reweight_stats(crg_nm, m_nm, r_nm, float(b), "mean")["pred_rg_lattice"]
            for b in np.linspace(-10, 10, 401)
        ])
        peak_nm = float(reach_nm.max())
        tgt_int = np.array([peak_nm + 0.25 * (9.0 - peak_nm)] * 3)
        fs_int = run_rg_feasibility_scan(
            crg_nm, ce_nm, re_nm, tgt_int, tgt_int,
            rg_scale=1.0, summary="mean", bias_min=-10.0, bias_max=10.0,
            bias_points=401, outdir=tdp, make_plots=False,
        )
        rcheck(
            fs_int["reachability_status"] == "unreachable_within_scan",
            f"a target missed at an INTERIOR extremum is unreachable_within_scan "
            f"(got {fs_int['reachability_status']})",
        )
        int_join = " ".join(fs_int["warnings"])
        rcheck(
            "were not reached within the scanned bias interval" in int_join,
            "the unreachable warning says 'not reached within the scanned bias "
            "interval'",
        )
        rcheck(
            "remain inside the rigorous global outer bound" in int_join
            and "does not prove impossibility" in int_join,
            "the unreachable warning states the target is still inside the outer "
            "bound and disclaims impossibility",
        )
        rcheck(
            not any("cannot be reproduced by any real contact bias" in w
                    for w in fs_int["warnings"]),
            "a scan-scoped miss inside the outer bound makes no all-b claim",
        )

        # BUGFIX 3: a target attained at BOTH a boundary bias and an interior bias
        # must be reachable_within_scan, not boundary_limited. The nonmonotonic
        # baseline's Rg(b) is symmetric about b=0, so on the asymmetric window
        # [-2, +10] the left-boundary value Rg(-2) equals the interior value
        # Rg(+2). An argmin-over-the-whole-grid boundary test picks the boundary
        # index and wrongly flags boundary_limited; the interior-envelope test
        # resolves it as reachable.
        b_left = -2.0
        tgt_val = float(
            joint_reweight_stats(crg_nm, m_nm, r_nm, b_left, "mean")["pred_rg_lattice"]
        )
        tgt_sym = np.array([tgt_val] * 3)
        fs_sym = run_rg_feasibility_scan(
            crg_nm, ce_nm, re_nm, tgt_sym, tgt_sym,
            rg_scale=1.0, summary="mean", bias_min=b_left, bias_max=10.0,
            bias_points=241, outdir=tdp, make_plots=False,
        )
        rcheck(
            fs_sym["reachability_status"] == "reachable_within_scan",
            f"a target attained at BOTH a boundary and an interior bias is "
            f"reachable_within_scan, not boundary_limited "
            f"(got {fs_sym['reachability_status']})",
        )
        rcheck(
            fs_sym["target_reached_only_at_bias_boundary"] is False,
            "such a target is not flagged as reached only at a bias boundary",
        )
        rcheck(
            not any("not conclusive" in w or "wider interval" in w
                    for w in fs_sym["warnings"]),
            "an interior-reachable target emits no boundary/inconclusive warning",
        )

        rcheck(
            fs_scan["reachability_status"] in (
                "unreachable_within_scan", "boundary_limited"
            ),
            f"finite-scan miss classified as a scan-scoped status: "
            f"{fs_scan['reachability_status']}",
        )
        rcheck(
            "global_outer_bound" in fs_scan["definition_note"].lower()
            and "never a proof" in fs_scan["reachability_status_note"].lower(),
            "definition_note disclaims proof-of-impossibility for the finite scan",
        )
        # §7: the finite-scan message and an all-b impossibility claim are mutually
        # exclusive conclusions and must never be printed about the same target.
        rcheck(
            not any("cannot be reproduced by any real contact bias" in w
                    for w in fs_scan["warnings"]),
            "a scan-scoped miss emits no all-b impossibility warning",
        )
        rcheck(
            sum(1 for w in fs_scan["warnings"]
                if "widen the bias interval" in w or "wider interval" in w) <= 1,
            "at most one inconclusive/widen-the-scan warning is emitted",
        )

        # A target outside the GLOBAL OUTER BOUND may make the stronger claim.
        # It is placed strictly INSIDE the baseline Rg support, between the outer
        # bound and the top of the support: support overlap therefore exists and
        # cannot pre-empt the verdict, isolating the all-b impossibility claim.
        # This is the scientifically interesting case -- the baseline has mass at
        # this Rg, yet no bias can move the SUMMARY there, because every biased
        # summary is a convex combination of the per-contact conditional moments.
        base_lo_k, base_hi_k = baseline_rg_support(crg_s, re_s)
        gob_hi_k = float(gob_scan["global_outer_rg_max_lattice"])
        far = gob_hi_k + 0.5 * (base_hi_k - gob_hi_k)
        rcheck(
            base_lo_k < far < base_hi_k and far > gob_hi_k,
            f"the out-of-bound test target {far:.4g} is inside the baseline support "
            f"[{base_lo_k:.4g}, {base_hi_k:.4g}] but above the outer bound "
            f"{gob_hi_k:.4g}",
        )
        tgt_far = np.array([far] * 3)
        fs_far = run_rg_feasibility_scan(
            crg_s, ce_s, re_s, tgt_far, 0.345 * tgt_far,
            rg_scale=0.345, summary="rms", bias_min=-10.0, bias_max=10.0,
            bias_points=401, outdir=tdp, make_plots=False,
        )
        gob_w = [w for w in fs_far["warnings"]
                 if "global scalar-Rg outer bound" in w]
        rcheck(
            bool(gob_w)
            and any("cannot be reproduced by any real contact bias b" in w
                    for w in gob_w),
            "an out-of-outer-bound target gets the stronger all-b impossibility claim",
        )
        rcheck(
            fs_far["global_outer_bound"]["target_within_global_outer_bound"] is False,
            "target_within_global_outer_bound=False for an impossible target",
        )
        rcheck(
            fs_far["reachability_status"] == "outside_global_outer_bound",
            f"an impossible target is classified outside_global_outer_bound, got "
            f"{fs_far['reachability_status']}",
        )
        # §7: the impossibility claim must not be accompanied by "widen the scan".
        rcheck(
            not any("widen the bias interval" in w or "wider interval" in w
                    for w in fs_far["warnings"]),
            "an all-b impossibility claim is NOT paired with a widen-the-scan warning",
        )
        rcheck(
            all("any real bias" not in w and "any real contact bias" not in w
                for w in fs_far["warnings"] if w not in gob_w
                and "support" not in w.lower()),
            "only the outer-bound warning speaks about all real bias",
        )

        # §6/§17.6: the scan's status and the validity status must never contradict.
        for fs_case in (fs_scan, fs_far):
            val_case = classify_scientific_validity(
                fs_case,
                rg_support_overlap(
                    np.array([fs_case["target_rg_min"]]),
                    np.array([fs_case["target_rg_max"]]),
                    fs_case["baseline_rg_min"], fs_case["baseline_rg_max"],
                ),
                is_fitted_scale=False,
            )
            consistent = {
                "zero_support_overlap": {"zero_support_overlap"},
                "outside_global_outer_bound": {"outside_global_outer_bound"},
                "boundary_limited": {"boundary_limited"},
                "unreachable_within_scan": {"outside_scanned_range"},
                "reachable_within_scan": {
                    "supported", "supported_as_mapping_diagnostic",
                    "weak_contact_rg_coupling",
                },
            }[fs_case["reachability_status"]]
            rcheck(
                val_case["status"] in consistent
                or val_case["status"] == "zero_support_overlap",
                f"validity status {val_case['status']!r} is consistent with "
                f"reachability_status {fs_case['reachability_status']!r}",
            )

    # -------------------------------------------- unreachability vs roundoff --
    print("Scalar-Rg test 15b: float noise never manufactures an unreachability claim")
    # Unreachability is the strongest claim this script makes, so it must not rest
    # on the last bit of a float. Converting a target observed -> lattice
    # (x -> s*x -> s*x/s) is inexact for roughly 7% of values, by up to 1 ulp.
    lo_r, hi_r = 2.0, 4.0
    one_ulp_out = np.array([np.nextafter(hi_r, np.inf), np.nextafter(lo_r, -np.inf)])
    rcheck(
        int(np.sum((one_ulp_out < lo_r) | (one_ulp_out > hi_r))) == 2,
        "a strict comparison DOES flag values 1 ulp outside the range (the artifact)",
    )
    rcheck(
        _count_outside_range(one_ulp_out, lo_r, hi_r) == 0,
        "the tolerance does not count values 1 ulp outside as unreachable",
    )
    # A real observed->lattice round-trip must never be counted as a miss.
    rng_rt = np.random.default_rng(0)
    xs = rng_rt.uniform(lo_r, hi_r, size=20000)
    for s_rt in (0.345, 0.3, 1.0 / 3.0):
        rt = (s_rt * xs) / s_rt
        rcheck(
            _count_outside_range(rt, float(xs.min()), float(xs.max())) == 0,
            f"observed->lattice round-trip at scale {s_rt:.6g} yields no false "
            f"unreachability",
        )
    # The tolerance must stay far below anything physically meaningful, so a real
    # miss is still caught.
    rcheck(
        _count_outside_range(np.array([hi_r * 1.000001]), lo_r, hi_r) == 1
        and _count_outside_range(np.array([hi_r * 2.0]), lo_r, hi_r) == 1,
        "a physically meaningful excursion is still counted as unreachable",
    )
    rcheck(
        RG_REACH_RTOL < 1e-6,
        f"the reachability tolerance ({RG_REACH_RTOL:g}) is far below any "
        f"experimental Rg precision",
    )
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        # A genuinely unreachable target must still be reported as such.
        crg0, ce0, re0 = _make_synthetic_joint(coupling=0.0)
        gob0 = global_rg_outer_bounds(crg0, ce0, re0, summary="rms", rg_scale=0.345)
        far0 = float(gob0["global_outer_rg_max_lattice"]) * 2.0
        fs0_far = run_rg_feasibility_scan(
            crg0, ce0, re0, np.array([far0] * 3), np.array([0.345 * far0] * 3),
            rg_scale=0.345, summary="rms", bias_min=-10.0, bias_max=10.0,
            bias_points=101, outdir=tdp, make_plots=False,
        )
        rcheck(
            fs0_far["n_targets_outside_reachable"] == 3
            and fs0_far["global_outer_bound"]["target_within_global_outer_bound"]
            is False,
            "the tolerance still detects a genuinely unreachable target",
        )
        # A zero-coupling baseline is non-identifiable and must say so regardless.
        rcheck(
            any("weak" in w.lower() for w in fs0_far["warnings"])
            and any("insensitive" in w.lower() for w in fs0_far["warnings"]),
            "a zero-coupling baseline still reports weak coupling and bias "
            "insensitivity",
        )

    # ---------------------------------------------------------------- test A --
    print("Scalar-Rg test 16: effective-scale consistency in free-scale mode")
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        # Truth generated at 0.5, but --rg-scale starts at 0.30, so the fitted
        # scale must move materially away from the initial value.
        true_scale = 0.50
        pred_lat_true, _, _ = predict_rg_summary_from_joint(
            crg_s, ce_s, re_s, temps_s, true_p, ufn_s,
            rg_scale=true_scale, summary="rms", target_units="observed",
        )
        tgt_true_obs = true_scale * pred_lat_true
        dat_p, npz_p = _write_scalar_inputs(
            tdp, temps_s, tgt_true_obs, crg_s, ce_s, re_s
        )
        out_a = tdp / "free"
        a = _scalar_args(
            rg_means_file=dat_p, baseline=npz_p, outdir=str(out_a),
            model="tc_scale", rg_summary="rms", rg_target_units="observed",
            rg_mean_loss="mse", rg_scale=0.30, fit_rg_scale=True,
            rg_scale_min=0.2, rg_scale_max=0.8, no_plots=True,
            rg_feasibility_scan=True, rg_bias_points=101, n_restarts=4, seed=7,
        )
        run_rg_scalar_mode(a)

        summ_a = json.loads((out_a / "fit_summary.json").read_text())
        # Copy the arrays out and close the handle: NpzFile keeps the file open,
        # and Windows refuses to remove the temp directory while it is.
        with np.load(out_a / "fit_results.npz", allow_pickle=False) as _z:
            z = {k: _z[k] for k in _z.files}
            z_files = list(_z.files)

        eff = float(summ_a["rg_scale_effective"])
        init = float(summ_a["rg_scale_initial"])
        fitv = summ_a["rg_scale_fitted"]
        rcheck(
            summ_a["rg_scale_was_fitted"] is True and fitv is not None,
            "free-scale run records rg_scale_was_fitted=True and a fitted scale",
        )
        rcheck(
            abs(eff - float(fitv)) < 1e-12 and abs(init - 0.30) < 1e-12,
            "rg_scale_effective == rg_scale_fitted and rg_scale_initial is preserved",
        )
        rcheck(
            abs(eff - init) > 0.05,
            f"the recovered scale differs materially from the initial "
            f"({eff:.4g} vs {init:.4g})",
        )
        rcheck(
            abs(eff - true_scale) < 1e-3,
            f"the fitted scale recovers the synthetic truth ({eff:.6g} vs {true_scale})",
        )
        # The four consistency identities required of the effective scale.
        rcheck(
            np.allclose(z["rg_pred_observed"], eff * z["rg_pred_lattice"], rtol=1e-12),
            "rg_pred_observed == rg_scale_effective * rg_pred_lattice",
        )
        rcheck(
            np.allclose(z["rg_centers_observed"], eff * z["rg_centers_lattice"],
                        rtol=1e-12),
            "rg_centers_observed == rg_scale_effective * rg_centers_lattice",
        )
        rcheck(
            np.allclose(z["rg_edges_observed"], eff * z["rg_edges_lattice"], rtol=1e-12),
            "rg_edges_observed == rg_scale_effective * rg_edges_lattice",
        )
        rcheck(
            np.allclose(z["rg_target_lattice"], z["rg_target_observed"] / eff,
                        rtol=1e-12),
            "rg_target_lattice == rg_target_observed / rg_scale_effective",
        )
        rcheck(
            np.allclose(z["rg_target_observed"], z["rg_target_input"], rtol=1e-12),
            "target_units=observed leaves rg_target_observed equal to the input",
        )
        # Nominal-scale fields must exist, be named _nominal, and differ.
        rcheck(
            np.allclose(z["rg_centers_observed_nominal"], init * z["rg_centers_lattice"],
                        rtol=1e-12),
            "rg_centers_observed_nominal uses rg_scale_initial",
        )
        rcheck(
            not np.allclose(z["rg_centers_observed_nominal"], z["rg_centers_observed"]),
            "nominal and effective observed centers are genuinely different arrays",
        )
        for key in ("rg_scale_initial", "rg_scale_fitted", "rg_scale_effective",
                    "rg_scale_was_fitted"):
            rcheck(key in z_files, f"NPZ carries {key}")
        rcheck(
            abs(float(z["rg_scale_effective"]) - eff) < 1e-12
            and abs(float(z["rg_scale_initial"]) - init) < 1e-12
            and bool(z["rg_scale_was_fitted"]) is True,
            "NPZ scale provenance matches the JSON summary",
        )

        # ------------------------------------------------------------ test B --
        print("Scalar-Rg test 17: nominal and fitted feasibility are separate results")
        fd = summ_a["feasibility_diagnostics"]
        rcheck(
            isinstance(fd, dict) and "nominal_scale" in fd and "fitted_scale" in fd,
            "feasibility_diagnostics has nominal_scale and fitted_scale sections",
        )
        rcheck(
            fd["nominal_scale"] is not None and fd["fitted_scale"] is not None,
            "free-scale mode populates BOTH feasibility sections",
        )
        rcheck(
            abs(fd["nominal_scale"]["rg_scale_used"] - init) < 1e-12
            and abs(fd["fitted_scale"]["rg_scale_used"] - eff) < 1e-12,
            "each feasibility section records the scale it actually used",
        )
        rcheck(
            fd["nominal_scale"]["rg_scale_used"] != fd["fitted_scale"]["rg_scale_used"],
            "the two feasibility summaries differ",
        )
        for fn in ("rg_feasibility_nominal.csv", "rg_feasibility_nominal_summary.json",
                   "rg_feasibility_fitted.csv", "rg_feasibility_fitted_summary.json"):
            rcheck((out_a / fn).exists(), f"free-scale mode writes {fn}")
        rcheck(
            not (out_a / "rg_feasibility_summary.json").exists(),
            "free-scale mode does not write the ambiguous unprefixed feasibility file",
        )
        # The nominal scan must not be clobbered by the fitted scan.
        nom_txt = json.loads((out_a / "rg_feasibility_nominal_summary.json").read_text())
        fit_txt = json.loads((out_a / "rg_feasibility_fitted_summary.json").read_text())
        rcheck(
            abs(nom_txt["rg_scale_used"] - init) < 1e-12
            and abs(fit_txt["rg_scale_used"] - eff) < 1e-12,
            "the fitted scan did not overwrite the nominal scan's file",
        )
        rcheck(
            nom_txt["scale_label"] == "nominal scale"
            and fit_txt["scale_label"] == "fitted scale",
            "each feasibility file labels its own scale",
        )

        # The fitted model must be classified by the FITTED result, not the nominal.
        sv = summ_a["scientific_validity"]
        rcheck(
            "fixed_or_nominal_scale" in sv and "fitted_scale" in sv,
            "scientific_validity is split into nominal and fitted branches",
        )
        rcheck(
            sv["fitted_scale"] is not None
            and sv["fitted_scale"]["status"] == "supported_as_mapping_diagnostic",
            f"a good fitted-scale run is a mapping diagnostic, not plain 'supported': "
            f"{sv['fitted_scale']['status'] if sv['fitted_scale'] else None}",
        )
        rcheck(
            sv["fitted_scale"]["within_global_outer_bound"] is True
            and sv["fitted_scale"]["reachable_within_scan"] is True,
            "the fitted-scale branch reports its own reachability, not the nominal's",
        )

        # ------------------------------------------------------------ test C --
        print("Scalar-Rg test 18: fit_summary.json stays consumable downstream")
        model_name = summ_a["model"]
        param_names_c = summ_a["param_names"]
        params_c = np.array([summ_a["params"][n] for n in param_names_c], dtype=float)
        Tref_c = summ_a["Tref"]
        Tscale_c = summ_a["Tscale"]
        rcheck(
            model_name == "tc_scale" and param_names_c == ["A", "Tc"],
            f"top-level model/param_names match the registry: {model_name}, "
            f"{param_names_c}",
        )
        rcheck(
            param_names_c == list(MODEL_REGISTRY[model_name]["param_names"]),
            "param_names is exactly the registry order for the model",
        )
        b_fn_c = make_b_fn(model_name, Tref_c, Tscale_c)
        u_fn_c = make_contact_u_fn(model_name, Tref_c, Tscale_c)
        val_c = b_fn_c(params_c, 300.0)
        rcheck(
            np.isfinite(val_c),
            f"make_b_fn accepts the reconstructed model vector: b(300)={val_c:.6g}",
        )
        rcheck(
            "rg_scale" not in summ_a["params"],
            "rg_scale is NOT inside summary['params']",
        )
        rcheck(
            len(summ_a["params"]) == len(MODEL_REGISTRY[model_name]["param_names"]),
            "params carries thermodynamic parameters only",
        )
        rcheck(
            "rg_scale" in summ_a["parameters"],
            "the fitted rg_scale is still reported, in 'parameters'",
        )
        # The prediction rebuilt from the summary alone must match the NPZ.
        pl_c, _, _ = predict_rg_summary_from_joint(
            crg_s, ce_s, re_s, z["temps"], params_c, u_fn_c,
            rg_scale=float(summ_a["rg_scale_effective"]), summary="rms",
            target_units="observed",
        )
        rcheck(
            np.allclose(pl_c, z["rg_pred_lattice"], rtol=1e-10),
            "the summary alone reproduces the saved prediction exactly",
        )
        # units section must not fabricate a physical unit.
        rcheck(
            summ_a["units"]["mapping"]
            == "Rg_observed = rg_scale_effective * Rg_lattice",
            "units.mapping states the effective-scale mapping",
        )
        rcheck(
            summ_a["units"]["target_input"] == "observed",
            "units.target_input records the target unit system",
        )
        # Strict JSON, no NaN/Infinity tokens anywhere.
        for fn in ("fit_summary.json", "rg_feasibility_nominal_summary.json",
                   "rg_feasibility_fitted_summary.json"):
            txt = (out_a / fn).read_text()
            json.loads(txt)
            rcheck(
                "NaN" not in txt and "Infinity" not in txt,
                f"{fn} contains no NaN/Infinity tokens",
            )

    # ---------------------------------------------------------- fixed-scale ---
    print("Scalar-Rg test 19: fixed-scale mode keeps its filenames and null fitted scale")
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        fixed_scale = 0.345
        pl_fx, _, _ = predict_rg_summary_from_joint(
            crg_s, ce_s, re_s, temps_s, true_p, ufn_s,
            rg_scale=fixed_scale, summary="rms", target_units="observed",
        )
        dat_p, npz_p = _write_scalar_inputs(
            tdp, temps_s, fixed_scale * pl_fx, crg_s, ce_s, re_s
        )
        out_f = tdp / "fixed"
        a = _scalar_args(
            rg_means_file=dat_p, baseline=npz_p, outdir=str(out_f),
            model="tc_scale", rg_summary="rms", rg_target_units="observed",
            rg_mean_loss="mse", rg_scale=fixed_scale, fit_rg_scale=False,
            no_plots=True, rg_feasibility_scan=True, rg_bias_points=101,
            n_restarts=4, seed=7,
        )
        run_rg_scalar_mode(a)
        summ_f = json.loads((out_f / "fit_summary.json").read_text())
        # Existing output filenames for fixed-scale scalar mode are unchanged.
        for fn in ("rg_feasibility.csv", "rg_feasibility_summary.json"):
            rcheck((out_f / fn).exists(),
                   f"fixed-scale mode keeps the historical filename {fn}")
        for fn in ("rg_feasibility_nominal_summary.json",
                   "rg_feasibility_fitted_summary.json"):
            rcheck(not (out_f / fn).exists(),
                   f"fixed-scale mode does not emit {fn}")
        rcheck(
            summ_f["rg_scale_fitted"] is None
            and summ_f["rg_scale_was_fitted"] is False
            and abs(summ_f["rg_scale_effective"] - fixed_scale) < 1e-12
            and abs(summ_f["rg_scale_initial"] - fixed_scale) < 1e-12,
            "fixed-scale mode: fitted is null, effective == initial == --rg-scale",
        )
        fd_f = summ_f["feasibility_diagnostics"]
        rcheck(
            fd_f["fitted_scale"] is None and fd_f["nominal_scale"] is not None,
            "fixed-scale mode: fitted_scale is null, nominal_scale is the result",
        )
        rcheck(
            summ_f["scientific_validity"]["fitted_scale"] is None,
            "fixed-scale mode: no fitted-scale validity branch",
        )
        rcheck(
            summ_f["scientific_validity"]["fixed_or_nominal_scale"]["status"]
            == "supported",
            f"a good fixed-scale run is 'supported': "
            f"{summ_f['scientific_validity']['fixed_or_nominal_scale']['status']}",
        )
        rcheck(
            "rg_scale" not in summ_f["params"]
            and summ_f["param_names"] == ["A", "Tc"],
            "fixed-scale summary also keeps params thermodynamic-only",
        )
        # rg_scale is a mapping constant: it must never reach b_fn.
        rcheck(
            len(summ_f["params"]) == 2
            and np.isfinite(make_b_fn("tc_scale", summ_f["Tref"], summ_f["Tscale"])(
                np.array([summ_f["params"]["A"], summ_f["params"]["Tc"]]), 300.0)),
            "fixed-scale summary reconstructs a working b_fn",
        )

    # ---------------------------------------------------------------- test F --
    print("Scalar-Rg test 20: zero Rg support overlap is handled per mode")
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        # Baseline Rg support is [1, 5] lattice. At rg_scale=0.345 a target of
        # ~40 observed units maps to ~116 lattice: entirely disjoint.
        disjoint_obs = np.full(temps_s.size, 40.0)
        dat_p, npz_p = _write_scalar_inputs(
            tdp, temps_s, disjoint_obs, crg_s, ce_s, re_s
        )
        a_fixed = _scalar_args(
            rg_means_file=dat_p, baseline=npz_p, outdir=str(tdp / "zs_fixed"),
            model="tc_scale", rg_summary="rms", rg_target_units="observed",
            rg_scale=0.345, fit_rg_scale=False, no_plots=True,
            rg_feasibility_scan=False, n_restarts=2, seed=3,
        )
        raised = ""
        try:
            run_rg_scalar_mode(a_fixed)
        except ValueError as exc:
            raised = str(exc)
        rcheck(
            ZERO_SUPPORT_OVERLAP_MESSAGE in raised,
            f"fixed-scale mode raises on zero support overlap: {raised[:60]!r}",
        )
        rcheck(
            not (tdp / "zs_fixed" / "fit_results.npz").exists(),
            "fixed-scale zero-overlap run aborts BEFORE optimization (no outputs)",
        )

        # Free-scale mode records the nominal failure but is allowed to proceed,
        # because moving the scale can restore overlap.
        out_z = tdp / "zs_free"
        a_free = _scalar_args(
            rg_means_file=dat_p, baseline=npz_p, outdir=str(out_z),
            model="tc_scale", rg_summary="rms", rg_target_units="observed",
            rg_scale=0.345, fit_rg_scale=True, rg_scale_min=5.0, rg_scale_max=30.0,
            no_plots=True, rg_feasibility_scan=True, rg_bias_points=51,
            n_restarts=3, seed=3,
        )
        ran = True
        try:
            run_rg_scalar_mode(a_free)
        except ValueError:
            ran = False
        rcheck(ran, "free-scale mode is not aborted by a nominal support failure")
        if ran:
            summ_z = json.loads((out_z / "fit_summary.json").read_text())
            sd = summ_z["support_diagnostics"]
            rcheck(
                sd["nominal_scale_support_overlap"]["zero_support_overlap"] is True,
                "the nominal-scale support failure is recorded, not hidden",
            )
            rcheck(
                summ_z["scientific_validity"]["fixed_or_nominal_scale"]["status"]
                == "zero_support_overlap",
                "nominal-scale validity is marked zero_support_overlap",
            )
            # rg_scale_max=30 lets the fitted scale restore overlap.
            restored = (
                sd["effective_scale_support_overlap"]["has_support_overlap"] is True
            )
            rcheck(
                restored,
                f"a fitted scale ({summ_z['rg_scale_effective']:.4g}) restores support "
                f"overlap, and the fitted branch is reassessed on its own terms",
            )
            rcheck(
                summ_z["scientific_validity"]["fitted_scale"]["support_overlap"]
                is restored,
                "fitted-scale diagnostics reassess support at the effective scale",
            )
            rcheck(
                summ_z["scientific_validity"]["fitted_scale"]["status"]
                != "zero_support_overlap",
                "the fitted result is NOT classified by the nominal support failure",
            )

    # --------------------------------------------------------------- test L --
    # §10: all four root topologies. The tangential root is placed OFF the
    # sampling grid on purpose: b(T) = ((T-Tref)/Tscale - 0.2)^2 has its root at
    # exactly T=324, where (324-315)/45 == 0.2 in exact float arithmetic, so
    # b(T) == 0.0 lands on the `a == 0.0` branch and even a pure sign-change
    # search "finds" it. That would make this test vacuous. Nudging the root off
    # the grid removes the coincidence and tests real tangency detection.
    print("Scalar-Rg test 21: zero crossings cover sign-change, exact, tangential, edge")
    Tref_r, Tscale_r = 315.0, 45.0
    t_lo_r, t_hi_r = 270.0, 360.0

    # (a) ordinary sign-changing root
    root_sign = 315.0
    fn_sign = lambda p, T: (T - root_sign) / Tscale_r
    got = bias_zero_crossings(fn_sign, np.array([0.0]), t_lo_r, t_hi_r)
    rcheck(
        len(got) == 1 and abs(got[0] - root_sign) < 1e-6,
        f"sign-changing root found at {root_sign}: got {got}",
    )

    # (b) exactly sampled zero that does NOT change sign is still a root
    #     (verified below by (c); here: an exactly sampled sign-changing zero)
    fn_exact = lambda p, T: (T - 324.0) / Tscale_r
    got = bias_zero_crossings(fn_exact, np.array([0.0]), t_lo_r, t_hi_r)
    rcheck(
        len(got) == 1 and abs(got[0] - 324.0) < 1e-9,
        f"exactly sampled zero found at 324.0: got {got}",
    )

    # (c) TANGENTIAL root, off-grid: touches zero, never changes sign
    r_tan = 0.2005
    root_tan = Tref_r + r_tan * Tscale_r          # 324.0225
    grid_r = np.linspace(t_lo_r, t_hi_r, 2001)
    rcheck(
        not np.any(np.abs(grid_r - root_tan) < 1e-12),
        f"the tangential test root {root_tan:g} is genuinely OFF the sampling grid "
        f"(otherwise this test would pass without tangency detection)",
    )
    fn_tan = lambda p, T: ((T - Tref_r) / Tscale_r - r_tan) ** 2
    vals_tan = np.array([fn_tan(None, float(T)) for T in grid_r])
    rcheck(
        np.all(vals_tan >= 0.0) and not np.any(vals_tan == 0.0),
        "the tangential b(T) never changes sign and is never exactly zero on the grid",
    )
    got_tan = bias_zero_crossings(fn_tan, np.array([0.0]), t_lo_r, t_hi_r)
    rcheck(
        len(got_tan) == 1 and abs(got_tan[0] - root_tan) < 1e-4,
        f"tangential root at {root_tan:g} is found: got {got_tan}",
    )
    rcheck(
        abs(fn_tan(None, got_tan[0])) <= 1e-8 if got_tan else False,
        "the tangential root actually satisfies |b(T_root)| <= tolerance",
    )

    # (d) root sitting on an interval endpoint
    fn_edge = lambda p, T: (T - t_hi_r) / Tscale_r
    got = bias_zero_crossings(fn_edge, np.array([0.0]), t_lo_r, t_hi_r)
    rcheck(
        len(got) == 1 and abs(got[0] - t_hi_r) < 1e-6,
        f"a root on the interval endpoint {t_hi_r:g} is found: got {got}",
    )
    fn_edge_lo = lambda p, T: (T - t_lo_r) / Tscale_r
    got = bias_zero_crossings(fn_edge_lo, np.array([0.0]), t_lo_r, t_hi_r)
    rcheck(
        len(got) == 1 and abs(got[0] - t_lo_r) < 1e-6,
        f"a root on the interval endpoint {t_lo_r:g} is found: got {got}",
    )

    # (e) a strictly positive b(T) with a non-zero minimum must NOT be a root
    fn_nonroot = lambda p, T: ((T - Tref_r) / Tscale_r - r_tan) ** 2 + 0.5
    got = bias_zero_crossings(fn_nonroot, np.array([0.0]), t_lo_r, t_hi_r)
    rcheck(
        got == [],
        f"a strictly positive b(T) yields NO root (minimum of |b| is not a zero): "
        f"got {got}",
    )
    # (e2) BUGFIX 2: a LARGE-DYNAMIC-RANGE b(T) whose minimum is 0.5 must yield no
    # root. A tolerance proportional to 1e-8*max|b| would be ~1 here (max|b|~1e8)
    # and wrongly accept the 0.5 minimum; the floating-point-scale tolerance plus
    # the final |b(T_root)| verification must reject it.
    fn_bigrange = lambda p, T: 1e8 * ((T - 315.0) / 45.0) ** 2 + 0.5
    got = bias_zero_crossings(fn_bigrange, np.array([0.0]), t_lo_r, t_hi_r)
    rcheck(
        got == [],
        f"a large-dynamic-range b(T) with minimum 0.5 yields NO root "
        f"(the 0.5 minimum is not a zero): got {got}",
    )
    # (f) two distinct tangential roots are both found and not deduplicated away
    fn_two = lambda p, T: ((T - 300.0) ** 2) * ((T - 340.0) ** 2) / 1e6
    got = bias_zero_crossings(fn_two, np.array([0.0]), t_lo_r, t_hi_r)
    rcheck(
        len(got) == 2
        and abs(got[0] - 300.0) < 1e-3 and abs(got[1] - 340.0) < 1e-3,
        f"two distinct tangential roots are both found: got {got}",
    )

    # --------------------------------------------------------------- test M --
    # §11: T_rg_half must be null for a flat curve and for an ambiguous one.
    print("Scalar-Rg test 22: T_rg_half is null for flat and ambiguous curves")
    tm_flat_h = rg_curve_transition_metrics(
        crg_flat, ce_f, re_f, true_p, ufn_s, 270.0, 360.0,
        rg_scale=0.345, summary="rms", target_units="observed", n_grid=1001,
    )
    rcheck(
        tm_flat_h["T_rg_half"] is None and tm_flat_h["rg_half_defined"] is False,
        f"a flat curve reports T_rg_half=None (got {tm_flat_h['T_rg_half']!r})",
    )
    rcheck(
        tm_flat_h["rg_half_ambiguous"] is False,
        "a flat curve is not 'ambiguous' -- it is undefined for a different reason",
    )
    rcheck(
        tm_flat_h["rg_curve_span"] <= tm_flat_h["rg_half_tolerance"],
        "the flat verdict is justified by span <= tolerance, not by eyeball",
    )
    # A genuine monotonic collapse must still yield exactly one crossing.
    rcheck(
        tm_real["rg_half_defined"] is True
        and tm_real["T_rg_half"] is not None
        and len(tm_real["T_rg_half_crossings"]) == 1,
        f"a genuine collapse still defines T_rg_half "
        f"(got {tm_real['T_rg_half']!r}, "
        f"{len(tm_real['T_rg_half_crossings'])} crossing(s))",
    )
    # And the crossing must be interpolated, not snapped to a grid point.
    grid_real = tm_real["grid"]
    on_grid_real = bool(
        np.any(np.abs(grid_real - tm_real["T_rg_half"]) < 1e-12)
    )
    rcheck(
        not on_grid_real
        or abs(tm_real["curve"][int(np.argmin(np.abs(grid_real - tm_real["T_rg_half"])))]
               - tm_real["rg_half_value"]) < 1e-9,
        "T_rg_half is an interpolated crossing, not the nearest sampled point",
    )
    # Ambiguity, from a REAL Rg(T) curve rather than a hand-drawn array. The
    # nonmonotonic baseline's Rg(b) is single-peaked at b=0; a poly2 b(T) with a
    # turning point sweeps ACROSS that peak twice, so Rg(T) is bimodal and crosses
    # its endpoint midpoint several times. (A unimodal curve always crosses the
    # endpoint average exactly once, which is why a turning-point b(T) is needed
    # to exhibit ambiguity at all.)
    ufn_poly2 = make_contact_u_fn("poly2", 315.0, 90.0)
    p_amb = np.array([-2.0, -6.0, 25.0 / 3.0])
    tm_amb = rg_curve_transition_metrics(
        crg_nm, ce_nm, re_nm, p_amb, ufn_poly2, 270.0, 360.0,
        rg_scale=1.0, summary="mean", target_units="lattice", n_grid=1001,
    )
    rcheck(
        len(tm_amb["T_rg_half_crossings"]) > 1,
        f"the bimodal Rg(T) curve genuinely crosses its endpoint midpoint "
        f"more than once (got {len(tm_amb['T_rg_half_crossings'])} crossings at "
        f"{[round(t, 2) for t in tm_amb['T_rg_half_crossings']]})",
    )
    rcheck(
        tm_amb["T_rg_half"] is None and tm_amb["rg_half_ambiguous"] is True
        and tm_amb["rg_half_defined"] is False,
        f"several midpoint crossings -> T_rg_half=None and rg_half_ambiguous=True "
        f"(got T_rg_half={tm_amb['T_rg_half']!r}, "
        f"ambiguous={tm_amb['rg_half_ambiguous']})",
    )
    rcheck(
        all(270.0 <= t <= 360.0 for t in tm_amb["T_rg_half_crossings"]),
        "every reported crossing lies inside the observed temperature interval",
    )

    # --------------------------------------------------------------- test N --
    # §12: the CSVs must be readable by ordinary tools with no special arguments.
    print("Scalar-Rg test 23: CSV outputs are rectangular with a real header row")
    try:
        import pandas as _pd
    except Exception:
        _pd = None
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        run_rg_feasibility_scan(
            crg_s, ce_s, re_s, np.array([1.4, 1.5]), np.array([0.48, 0.52]),
            rg_scale=0.345, summary="rms", bias_min=-5.0, bias_max=5.0,
            bias_points=11, outdir=tdp, make_plots=False,
        )
        csv_p = tdp / "rg_feasibility.csv"
        with open(csv_p, newline="") as fh:
            rows = list(csv.reader(fh))
        rcheck(
            rows[0][0] == "bias",
            f"row 0 IS the header, not a metadata banner (got {rows[0][0]!r})",
        )
        widths = {len(r) for r in rows}
        rcheck(
            len(widths) == 1,
            f"every CSV row has the same field count (widths seen: {sorted(widths)})",
        )
        with open(csv_p, newline="") as fh:
            recs = list(csv.DictReader(fh))
        rcheck(
            len(recs) == 11
            and "bias" in recs[0] and "pred_rg_lattice" in recs[0]
            and "rg_scale_used" in recs[0],
            "csv.DictReader recognizes the real data columns with no skiprows",
        )
        rcheck(
            all(abs(float(r["rg_scale_used"]) - 0.345) < 1e-12 for r in recs),
            "scale provenance rides along as a repeated column on every row",
        )
        if _pd is not None:
            df = _pd.read_csv(csv_p)
            rcheck(
                list(df.columns)[:2] == ["bias", "pred_rg_lattice"]
                and len(df) == 11
                and _pd.api.types.is_numeric_dtype(df["bias"]),
                f"pandas.read_csv() parses it with no skiprows/comment args and "
                f"keeps numeric dtypes (columns={list(df.columns)[:3]}, n={len(df)})",
            )

    # --------------------------------------------------------------- test O --
    # §13: the driver validates its own arguments, even when called directly.
    print("Scalar-Rg test 24: run_rg_scalar_mode() validates its own arguments")
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        # Argument validation runs before any file is opened, so a well-formed
        # baseline is enough; its scientific content is irrelevant here.
        pred_lat_v, _, _ = predict_rg_summary_from_joint(
            crg_s, ce_s, re_s, temps_s, true_p, ufn_s,
            rg_scale=0.345, summary="rms", target_units="observed",
        )
        dat_p, npz_p = _write_scalar_inputs(
            tdp, temps_s, 0.345 * pred_lat_v, crg_s, ce_s, re_s
        )
        bad_cases = [
            ("rg_scale=0", dict(rg_scale=0.0), "rg-scale"),
            ("rg_scale=-1", dict(rg_scale=-1.0), "rg-scale"),
            ("rg_scale=nan", dict(rg_scale=float("nan")), "rg-scale"),
            ("rg_bias_min>=max", dict(rg_bias_min=5.0, rg_bias_max=-5.0), "rg-bias-min"),
            ("rg_bias_points=2", dict(rg_bias_points=2), "rg-bias-points"),
            ("n_restarts=0", dict(n_restarts=0), "n_restarts"),
            ("bootstrap=-1", dict(bootstrap=-1), "bootstrap"),
            ("huber_delta=0", dict(rg_huber_delta=0.0), "rg-huber-delta"),
            ("range_floor=0", dict(rg_range_floor=0.0), "rg-range-floor"),
            (
                "scale_min>=scale_max while fitting",
                dict(fit_rg_scale=True, rg_scale_min=0.9, rg_scale_max=0.2),
                "rg-scale-min",
            ),
        ]
        for label, over, expect_token in bad_cases:
            outd = tdp / f"never_{abs(hash(label)) % 10000}"
            a_bad = _scalar_args(
                rg_means_file=str(dat_p), baseline=str(npz_p), outdir=str(outd),
                model="tc_scale", rg_summary="rms", rg_target_units="observed",
                no_plots=True, **over,
            )
            try:
                run_rg_scalar_mode(a_bad)
                rcheck(False, f"direct call with {label} must raise ValueError")
            except ValueError as exc:
                rcheck(
                    expect_token in str(exc),
                    f"direct call with {label} raises a clear ValueError naming "
                    f"{expect_token!r}: {str(exc)[:70]}",
                )
            except Exception as exc:  # pragma: no cover - failure path
                rcheck(
                    False,
                    f"direct call with {label} raised {type(exc).__name__}, not a "
                    f"clear ValueError: {str(exc)[:70]}",
                )
            rcheck(
                not outd.exists(),
                f"an invalid {label} call creates no output directory (no "
                f"half-written run is left behind)",
            )

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
    u_fn: Callable[[np.ndarray, float, np.ndarray], np.ndarray],
) -> np.ndarray:
    """Predicted mean contacts at every temperature for one parameter set."""
    out = np.empty(temps.size, dtype=float)
    for i, T in enumerate(temps):
        p = model_contact_mass(p0_mass, m_centers, float(T), params, u_fn)
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
    u_fn = ctx["u_fn"]
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
            fit_rg, boot_temps_b, m_centers, boot_ct_b, p0_mass, u_fn, loss_fn,
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
            temps, m_centers, p_obs_mass, p0_mass, params_b, u_fn, loss_fn
        )
        train_loss_b = float(ct_pt[train_idx].sum())
        val_loss_b = float(ct_pt[val_idx].sum()) if has_val else float("nan")
        all_loss_b = float(ct_pt.sum())

        rg_train_b = rg_val_b = rg_all_b = float("nan")
        if can_fit_rg:
            _, rg_mod_b = predict_rg_from_joint(
                crg_prob, c_edges_joint, rg_edges_model_lattice, temps, params_b, u_fn
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
        meanC_list.append(_predicted_means(temps, m_centers, p0_mass, params_b, u_fn))

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
        ctx["p0_mass"], ctx["u_fn"], ctx["loss_fn"],
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
    u_fn = ctx["u_fn"]
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
            use_rg, train_temps, m_centers, p_obs_ct_train, p0_mass, u_fn, loss_fn,
            crg_prob=crg_prob, c_edges_joint=c_edges_joint,
            p_obs_rg_train=p_obs_rg_train, rg_weight=w,
        )
        best, obj_val = fit_one_split(obj_fn, obj_args, x0s, bounds)
        params = best.x
        param_path.append(np.asarray(params, dtype=float))

        ct_pt = per_temp_contact_losses(
            temps, m_centers, p_obs_mass, p0_mass, params, u_fn, loss_fn
        )
        _, rg_mod = predict_rg_from_joint(
            crg_prob, c_edges_joint, rg_edges_model_lattice, temps, params, u_fn
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
        predC = _predicted_means(temps, m_centers, p0_mass, params, u_fn)
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
# Scalar Rg(T) mode: joint baseline validation
# ---------------------------------------------------------------------------

# Tiny negative entries are accepted as roundoff from whatever produced the
# baseline and are clipped by sanitize_joint_baseline(); anything more negative
# than this is a real defect and is rejected.
JOINT_BASELINE_NEG_TOL = -1e-15


def validate_joint_baseline(
    c_edges: np.ndarray,
    rg_edges: np.ndarray,
    crg_prob: np.ndarray,
    *,
    neg_tol: float = JOINT_BASELINE_NEG_TOL,
) -> None:
    """Validate the joint baseline P0(m, Rg) or raise ValueError.

    Checks dimensionality, shape consistency, finiteness, strict monotonicity of
    both edge vectors, non-negativity within ``neg_tol``, and positive total mass.

    Finiteness is checked BEFORE monotonicity on purpose: ``np.diff`` of an array
    containing NaN yields NaN, and ``NaN <= 0`` is False, so a NaN edge would
    silently pass a bare ``np.any(np.diff(edges) <= 0)`` test.
    """
    c_edges = np.asarray(c_edges, dtype=float)
    rg_edges = np.asarray(rg_edges, dtype=float)
    crg_prob = np.asarray(crg_prob, dtype=float)

    if c_edges.ndim != 1:
        raise ValueError(f"c_edges must be 1-D, got shape {c_edges.shape}")
    if rg_edges.ndim != 1:
        raise ValueError(f"rg_edges must be 1-D, got shape {rg_edges.shape}")
    if crg_prob.ndim != 2:
        raise ValueError(f"crg_prob must be 2-D, got shape {crg_prob.shape}")
    if c_edges.size < 2:
        raise ValueError(f"c_edges needs >= 2 entries (>= 1 bin), got {c_edges.size}")
    if rg_edges.size < 2:
        raise ValueError(f"rg_edges needs >= 2 entries (>= 1 bin), got {rg_edges.size}")

    expected = (c_edges.size - 1, rg_edges.size - 1)
    if crg_prob.shape != expected:
        raise ValueError(
            f"crg_prob shape must be (len(c_edges)-1, len(rg_edges)-1) = {expected}, "
            f"got {crg_prob.shape}"
        )

    # Finiteness first — see the docstring note about NaN and np.diff.
    if not np.all(np.isfinite(c_edges)):
        raise ValueError("c_edges contains non-finite value(s) (NaN or inf)")
    if not np.all(np.isfinite(rg_edges)):
        raise ValueError("rg_edges contains non-finite value(s) (NaN or inf)")
    if not np.all(np.isfinite(crg_prob)):
        raise ValueError("crg_prob contains non-finite value(s) (NaN or inf)")

    if np.any(np.diff(c_edges) <= 0.0):
        raise ValueError("c_edges must be strictly increasing")
    if np.any(np.diff(rg_edges) <= 0.0):
        raise ValueError("rg_edges must be strictly increasing")

    if np.any(crg_prob < neg_tol):
        worst = float(crg_prob.min())
        raise ValueError(
            f"crg_prob must be non-negative within tolerance {neg_tol:g}; "
            f"most negative entry is {worst:.6g}"
        )
    total = float(crg_prob.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError(f"crg_prob must carry positive total mass, got {total!r}")


def sanitize_joint_baseline(
    c_edges: np.ndarray,
    rg_edges: np.ndarray,
    crg_prob: np.ndarray,
    *,
    neg_tol: float = JOINT_BASELINE_NEG_TOL,
) -> Tuple[np.ndarray, List[str]]:
    """Validate, clip roundoff negatives, and normalize the joint baseline once.

    Returns (crg_prob_normalized, notes) where ``notes`` records any clipping or
    renormalization that was applied, so callers can surface it rather than
    silently changing the user's baseline.
    """
    validate_joint_baseline(c_edges, rg_edges, crg_prob, neg_tol=neg_tol)
    crg = np.asarray(crg_prob, dtype=float).copy()
    notes: List[str] = []

    if np.any(crg < 0.0):
        worst = float(crg.min())
        crg = np.clip(crg, 0.0, None)
        notes.append(
            f"Joint baseline contained negative roundoff (most negative {worst:.3g}, "
            f"within the {neg_tol:g} tolerance); clipped to zero."
        )
    total = float(crg.sum())
    if total <= 0.0:
        raise ValueError("crg_prob total mass is non-positive after clipping")
    if abs(total - 1.0) > 1e-9:
        crg = crg / total
        notes.append(
            f"Joint baseline total mass was {total:.10g}; normalized to 1."
        )
    return crg, notes


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
    u_fn: Callable[[np.ndarray, float, np.ndarray], np.ndarray],
    *,
    rg_scale: float,
    summary: str,
    target_units: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predict a scalar Rg per temperature from the reweighted joint baseline.

    Uses predict_rg_from_joint() as the single source of truth for the contact
    reweighting P(Rg|T) ∝ sum_m P0(m, Rg) exp[-u_contact(m,T;N)], then collapses each
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
        u_fn=u_fn,
    )

    if not np.all(np.isfinite(rg_mass)):
        raise ValueError(
            "predicted P(Rg|T) contains non-finite probability mass; "
            "check the joint baseline and the contact-potential parameters."
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
    u_fn: Callable[[np.ndarray, float, np.ndarray], np.ndarray],
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
            crg_prob, c_edges, rg_edges, temps, model_params, u_fn,
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
    root_tolerance: Optional[float] = None,
) -> List[float]:
    """Temperatures in [t_lo, t_hi] where b(T) = 0.

    This is the BIAS zero-crossing: the temperature at which the contact bias
    vanishes.  For a finite chain it is not generally the temperature at which
    Rg(T) changes fastest — see rg_curve_transition_metrics().

    Roots are found by two complementary passes, because a sign change is not the
    only way a function reaches zero:

      sign change   bracketed and refined by bisection (odd-multiplicity roots)
      tangency      local minima of |b(T)| refined by bounded minimization, then
                    accepted only if |b(T_root)| <= root_tolerance
                    (even-multiplicity roots, e.g. b(T) = ((T-Tref)/Tscale - r)^2,
                    which touches zero without ever changing sign and which a
                    sign-change search misses entirely)

    Exactly-sampled zeros and roots sitting on the interval endpoints are handled
    by both passes; results are deduplicated.
    """
    grid = np.linspace(float(t_lo), float(t_hi), int(n_grid))
    vals = np.array([b_fn(params, float(T)) for T in grid], dtype=float)

    if root_tolerance is None:
        # A floating-point-scale tolerance, NOT a fraction of the dynamic range: a
        # tolerance like 1e-8 * max|b| accepts a genuinely nonzero local minimum as
        # a "root" whenever b(T) has a large dynamic range (e.g. 1e8*(...)^2 + 0.5,
        # whose minimum 0.5 is far from zero yet below 1e-8*1e8). Scaling only by
        # eps keeps the acceptance band at the level of round-trip round-off.
        finite = vals[np.isfinite(vals)]
        b_ref = float(np.max(np.abs(finite))) if finite.size else 1.0
        scale = max(1.0, b_ref)
        root_tolerance = max(1e-10, 100.0 * float(np.finfo(float).eps) * scale)
    root_tolerance = float(root_tolerance)

    out: List[float] = []

    # --- pass 1: sign-changing roots (and exactly sampled zeros) --------------
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

    # --- pass 2: tangential roots (local minima of |b| that reach zero) -------
    absv = np.abs(vals)
    cand: List[int] = []
    for i in range(absv.size):
        if not np.isfinite(absv[i]):
            continue
        left = absv[i - 1] if i > 0 else np.inf
        right = absv[i + 1] if i < absv.size - 1 else np.inf
        # <= on one side so a flat-bottomed sampled minimum still registers.
        if absv[i] <= left and absv[i] <= right:
            cand.append(i)

    for i in cand:
        lo = float(grid[max(i - 1, 0)])
        hi = float(grid[min(i + 1, grid.size - 1)])
        t_root = float(grid[i])
        b_root = float(absv[i])
        if hi > lo and minimize_scalar is not None:
            try:
                res = minimize_scalar(
                    lambda T: abs(float(b_fn(params, float(T)))),
                    bounds=(lo, hi), method="bounded",
                    options={"xatol": 1e-12 * max(1.0, abs(hi))},
                )
                if res.success and np.isfinite(res.fun) and float(res.fun) <= b_root:
                    t_root, b_root = float(res.x), float(res.fun)
            except Exception:
                pass  # fall back to the sampled candidate
        if b_root <= root_tolerance:
            out.append(t_root)

    # Deduplicate roots found by more than one pass or bisected to the same point.
    uniq: List[float] = []
    for t in sorted(out):
        if not any(abs(t - u) < 1e-6 * max(1.0, abs(u)) for u in uniq):
            uniq.append(float(t))

    # Final verification: keep only points that genuinely satisfy b(T) ~ 0. A
    # bisection endpoint or a refined local minimum is a candidate, not a proof;
    # this discards any that do not actually reach zero to tolerance, so a large-
    # dynamic-range function with a strictly positive minimum yields no roots.
    verified = [
        t for t in uniq
        if np.isfinite(b_fn(params, t)) and abs(float(b_fn(params, t))) <= root_tolerance
    ]
    return verified


def rg_curve_transition_metrics(
    crg_prob: np.ndarray,
    c_edges: np.ndarray,
    rg_edges: np.ndarray,
    model_params: np.ndarray,
    u_fn: Callable[[np.ndarray, float, np.ndarray], np.ndarray],
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

    ``T_rg_max_slope``        argmax_T of -dRg/dT, or None when the curve shows no
                              resolved collapse (flat or expanding with T)
    ``rg_max_negative_slope`` the value of -dRg/dT there (target units per K)
    ``collapse_detected``     whether a collapse was resolved at all
    ``T_rg_half``             the temperature at which the curve crosses the
                              midpoint of its endpoint values, by interpolation --
                              but ONLY when exactly one such crossing exists
    ``T_rg_half_crossings``   every midpoint crossing found
    ``rg_half_defined``       whether T_rg_half is meaningful
    ``rg_half_ambiguous``     whether the midpoint is crossed more than once

    A curve that is flat or expands with temperature has no collapse temperature;
    reporting the argmax of numerical noise as one would be a fabricated
    transition, so T_rg_max_slope is None in that case.

    T_rg_half is subject to two separate failure modes, both of which return None
    rather than a fabricated number:

      flat curve       every temperature is equally close to the midpoint, so an
                       argmin over |curve - mid| returns the first grid point --
                       a meaningless answer that looks like a measurement.
      nonmonotonic     the midpoint can be crossed several times; no single one
                       of them is "the" half-height temperature.

    T_rg_max_slope is the primary finite-chain transition descriptor and is a
    DIFFERENT quantity from the bias zero-crossing b(T) = 0.
    """
    n_grid = max(int(n_grid), 1001)
    grid = np.linspace(float(t_lo), float(t_hi), n_grid)
    pred_lat, pred_obs, _ = predict_rg_summary_from_joint(
        crg_prob, c_edges, rg_edges, grid, model_params, u_fn,
        rg_scale=rg_scale, summary=summary, target_units=target_units,
    )
    curve = rg_pred_in_target_units(pred_lat, pred_obs, target_units)
    slope = np.gradient(curve, grid)
    neg_slope = -slope
    k = int(np.argmax(neg_slope))
    max_negative_slope = float(np.max(neg_slope))

    # Scale-aware floor: a slope below this is indistinguishable from the
    # numerical noise of the curve over this temperature interval.
    slope_tolerance = max(
        1e-10,
        1e-6 * max(float(np.ptp(curve)), 1.0) / max(float(t_hi) - float(t_lo), 1.0),
    )
    collapse_detected = bool(max_negative_slope > slope_tolerance)
    T_rg_max_slope: Optional[float] = float(grid[k]) if collapse_detected else None

    rg_start, rg_end = float(curve[0]), float(curve[-1])
    mid = 0.5 * (rg_start + rg_end)
    curve_span = abs(rg_end - rg_start)

    # Scale-aware floor: below this the endpoints are indistinguishable and the
    # "midpoint" is not a level the curve meaningfully crosses.
    half_tolerance = max(1e-12, 1e-6 * max(float(np.ptp(curve)), 1.0))
    rg_half_flat = bool(curve_span <= half_tolerance)

    # Locate midpoint crossings by linear interpolation between adjacent grid
    # points, never by snapping to the nearest sampled point.
    crossings: List[float] = []
    if not rg_half_flat:
        dev = curve - mid
        for i in range(dev.size - 1):
            a, c = float(dev[i]), float(dev[i + 1])
            if a == 0.0:
                crossings.append(float(grid[i]))
                continue
            if a * c < 0.0:
                frac = a / (a - c)  # a + frac*(c-a) == 0
                crossings.append(float(grid[i] + frac * (grid[i + 1] - grid[i])))
        if dev.size and float(dev[-1]) == 0.0:
            crossings.append(float(grid[-1]))
        uniq: List[float] = []
        for t in crossings:
            if not any(abs(t - u) <= 1e-9 * max(1.0, abs(u)) for u in uniq):
                uniq.append(t)
        crossings = uniq

    if rg_half_flat:
        T_rg_half: Optional[float] = None
        rg_half_value: Optional[float] = float(0.5 * (rg_start + rg_end))
        rg_half_defined = False
        rg_half_ambiguous = False
    elif len(crossings) == 1:
        T_rg_half = float(crossings[0])
        rg_half_value = float(mid)
        rg_half_defined = True
        rg_half_ambiguous = False
    else:
        # Zero crossings (endpoints differ but the midpoint level is never
        # attained on the grid) or several: no single half-height temperature.
        T_rg_half = None
        rg_half_value = float(mid)
        rg_half_defined = False
        rg_half_ambiguous = len(crossings) > 1

    return {
        "T_rg_max_slope": T_rg_max_slope,
        "rg_max_negative_slope": max_negative_slope,
        "collapse_detected": collapse_detected,
        "slope_tolerance": float(slope_tolerance),
        "T_rg_half": T_rg_half,
        "T_rg_half_crossings": [float(t) for t in crossings],
        "rg_half_value": rg_half_value,
        "rg_half_defined": rg_half_defined,
        "rg_half_ambiguous": rg_half_ambiguous,
        "rg_half_tolerance": float(half_tolerance),
        "rg_curve_span": float(curve_span),
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
    # Stabilize the exponent over the SUPPORTED contact bins only (rows carrying
    # baseline mass), matching the production reweighting paths.  Letting an
    # unsupported (zero-mass) bin set the maximum can underflow every supported
    # weight under a strong bias; those bins contribute nothing to Z anyway.
    support = np.asarray(crg_prob, dtype=float).sum(axis=1) > 0.0
    x = _stabilized_exponent(-float(b) * np.asarray(m_centers, dtype=float), support)
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


def _conditional_rg_by_contact(
    crg_prob: np.ndarray,
    c_edges: np.ndarray,
    rg_edges: np.ndarray,
    *,
    summary: str,
    probability_tolerance: float,
) -> Dict[str, Any]:
    """Per-contact-bin conditional Rg moments, for the bins carrying baseline mass.

    Returns, over the supported contact bins only (marginal mass strictly above
    ``probability_tolerance``):

    ``scalar``  the chosen scalar summary of the normalized P0(Rg | m_i):
                E[Rg | m_i] for 'mean', sqrt(E[Rg^2 | m_i]) for 'rms'
    ``moment``  the RAW moment that enters the convex combination linearly:
                E[Rg | m_i] for 'mean', E[Rg^2 | m_i] for 'rms'

    The distinction matters.  Only ``moment`` mixes linearly under reweighting, so
    only ``moment`` may be used for bounding and monotonicity arguments; ``scalar``
    is what the fit and the target are expressed in.
    """
    if summary not in RG_SUMMARY_CHOICES:
        raise ValueError(
            f"Unknown Rg summary {summary!r}. Choose from {RG_SUMMARY_CHOICES}."
        )
    validate_joint_baseline(c_edges, rg_edges, crg_prob)
    crg_prob = np.asarray(crg_prob, dtype=float)
    c_edges = np.asarray(c_edges, dtype=float)
    rg_edges = np.asarray(rg_edges, dtype=float)
    rg_centers = 0.5 * (rg_edges[:-1] + rg_edges[1:])
    m_centers = 0.5 * (c_edges[:-1] + c_edges[1:])

    contact_marginal = crg_prob.sum(axis=1)
    nonzero_contact_bins = np.flatnonzero(contact_marginal > probability_tolerance)
    if nonzero_contact_bins.size == 0:
        raise ValueError(
            "joint baseline has no contact bin with marginal mass above the "
            "probability tolerance; conditional Rg moments are undefined"
        )

    scalars: List[float] = []
    moments: List[float] = []
    for i in nonzero_contact_bins:
        row = crg_prob[int(i)]
        z = float(row.sum())
        if not np.isfinite(z) or z <= 0.0:
            raise ValueError(
                f"contact bin {int(i)} has non-positive conditional Rg mass"
            )
        cond = row / z
        scalars.append(
            float(rg_summary_from_mass(cond[None, :], rg_centers, summary)[0])
        )
        if summary == "mean":
            moments.append(float(np.sum(cond * rg_centers)))
        else:
            moments.append(float(np.sum(cond * rg_centers ** 2)))

    return {
        "supported_contact_bins": nonzero_contact_bins.astype(int),
        "supported_contact_centers": m_centers[nonzero_contact_bins],
        "scalar": np.asarray(scalars, dtype=float),
        "moment": np.asarray(moments, dtype=float),
        "contact_marginal": contact_marginal,
        "probability_tolerance": float(probability_tolerance),
    }


def conditional_moment_monotonicity(
    moment: np.ndarray, *, rel_tolerance: float = 1e-9
) -> Dict[str, Any]:
    """Is the conditional Rg moment monotonic in supported contact count?

    This is the exact condition under which the two endpoint (b -> +/-inf) limits
    are the extrema of the scalar over ALL real b.  The biased scalar moment is the
    convex combination sum_i w_i(b) mu_i with weights w_i(b) ∝ P0(m_i) exp[-b m_i].
    A convex combination of a monotonic sequence is bounded by its first and last
    terms, and the endpoint limits attain exactly those; if mu_i is NOT monotonic an
    interior contact bin can carry the extremum and the endpoint limits then bound
    nothing.

    Differences below a scale-aware tolerance count as flat, so that floating-point
    noise in a genuinely constant or monotonic sequence is not misread as structure.

    Returns ``monotonic`` and ``direction`` in
    {increasing, decreasing, constant, nonmonotonic}.
    """
    mu = np.asarray(moment, dtype=float)
    if mu.size == 0:
        raise ValueError("conditional moment array is empty")
    scale_ref = max(float(np.max(np.abs(mu))), 1.0)
    tol = float(rel_tolerance) * scale_ref
    diffs = np.diff(mu)
    up = bool(np.any(diffs > tol))
    down = bool(np.any(diffs < -tol))
    if up and down:
        direction = "nonmonotonic"
    elif up:
        direction = "increasing"
    elif down:
        direction = "decreasing"
    else:
        direction = "constant"
    return {
        "conditional_moment_monotonic": direction != "nonmonotonic",
        "conditional_moment_direction": direction,
        "conditional_moment_tolerance": tol,
        "conditional_moment_max_step": float(np.max(np.abs(diffs))) if diffs.size else 0.0,
    }


def global_rg_outer_bounds(
    crg_prob: np.ndarray,
    c_edges: np.ndarray,
    rg_edges: np.ndarray,
    *,
    summary: str,
    rg_scale: float,
    probability_tolerance: float = 1e-15,
    rg_target_lattice: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Rigorous outer bound on the scalar Rg over ALL real contact biases b.

    Contact-only reweighting factorizes: P_b(m, Rg) ∝ P0(m, Rg) exp[-b m] leaves
    every conditional P0(Rg | m_i) untouched and only re-weights the contact
    marginal.  So for every real b the biased raw moment is a convex combination
    over the supported contact bins,

        E_b[X] = sum_i w_i(b) mu_i,   w_i(b) ∝ P0(m_i) exp[-b m_i],  w_i >= 0,
                                      sum_i w_i = 1,

    with X = Rg (summary 'mean') or X = Rg^2 (summary 'rms').  A convex combination
    is bounded by the extreme values it mixes, hence for every real b:

        min_i mu_i <= E_b[X] <= max_i mu_i.

    For 'mean' that bounds the scalar directly.  For 'rms' it bounds the second
    moment, and since sqrt is monotone increasing,

        sqrt(min_i q_i) <= sqrt(E_b[Rg^2]) <= sqrt(max_i q_i),

    which is why taking min/max of the per-bin conditional SCALARS is equivalent
    and is what is reported here.

    This is a NECESSARY bound, not a sufficient one:

      * a target outside it is impossible for every real b
      * a target inside it is NOT thereby proven reachable -- the weights w_i(b)
        form a one-parameter exponential family, not the full simplex, so most
        convex combinations are never realized by any single b

    Unlike the two endpoint limits (see endpoint_rg_limits), this bound holds
    whether or not mu_i is monotonic in contact count, and is therefore the only
    quantity here that may license an all-real-b impossibility claim.

    The bound is computed over EVERY contact bin carrying strictly positive mass,
    with no probability threshold. Any positive-mass bin, however small, dominates
    the reweighted marginal under sufficiently strong exponential bias exp[-b m],
    so excluding it could shrink the bound and manufacture a false
    ``outside_global_outer_bound`` verdict. The ``probability_tolerance`` argument
    is therefore NOT used to select bins for the bound; it is retained only as
    provenance in the returned dict and must never gate scientific validity.
    """
    cond = _conditional_rg_by_contact(
        crg_prob, c_edges, rg_edges,
        summary=summary, probability_tolerance=0.0,
    )
    bins = cond["supported_contact_bins"]
    centers = cond["supported_contact_centers"]
    scalar = cond["scalar"]
    scale = float(rg_scale)

    k_min = int(np.argmin(scalar))
    k_max = int(np.argmax(scalar))
    lo = float(scalar[k_min])
    hi = float(scalar[k_max])

    mono = conditional_moment_monotonicity(cond["moment"])

    out: Dict[str, Any] = {
        "supported_contact_bins": bins,
        "supported_contact_centers": centers,
        "conditional_rg_scalar_by_contact_lattice": scalar,
        "conditional_rg_scalar_by_contact_observed": scalar * scale,
        "conditional_rg_moment_by_contact": cond["moment"],
        "conditional_moment_quantity": (
            "E[Rg | m]" if summary == "mean" else "E[Rg^2 | m]"
        ),
        "global_outer_rg_min_lattice": lo,
        "global_outer_rg_max_lattice": hi,
        "global_outer_rg_min_observed": lo * scale,
        "global_outer_rg_max_observed": hi * scale,
        "contact_bin_at_global_min": int(bins[k_min]),
        "contact_bin_at_global_max": int(bins[k_max]),
        "contact_value_at_global_min": float(centers[k_min]),
        "contact_value_at_global_max": float(centers[k_max]),
        "target_within_global_outer_bound": None,
        "n_targets_outside_global_outer_bound": None,
        # 0.0: the bound uses every strictly-positive-mass contact bin. The
        # incoming probability_tolerance is echoed separately for provenance but
        # does NOT gate the bound.
        "probability_tolerance": 0.0,
        "probability_tolerance_requested": float(probability_tolerance),
        "rg_scale_used": scale,
        "is_exact_reachable_range": False,
        "note": (
            "Necessary outer bound; being inside does not prove finite-b "
            "reachability."
        ),
        "interpretation": (
            "Rigorous outer bound on the scalar Rg summary over ALL real contact "
            "biases b, from the convex-combination structure of contact-only "
            "reweighting over the supported contact bins. A target outside this "
            "bound cannot be reproduced by any real b for this fixed joint "
            "baseline. A target inside it is NOT thereby reachable: the bound is "
            "necessary, not sufficient. It constrains the SCALAR SUMMARY only and "
            "is not a statement about full-distribution support."
        ),
    }
    out.update(mono)
    if rg_target_lattice is not None:
        n_out = _count_outside_range(rg_target_lattice, lo, hi)
        out["n_targets_outside_global_outer_bound"] = n_out
        out["target_within_global_outer_bound"] = bool(n_out == 0)
    return out


def endpoint_rg_limits(
    crg_prob: np.ndarray,
    c_edges: np.ndarray,
    rg_edges: np.ndarray,
    *,
    summary: str,
    rg_scale: float,
    tolerance: float = 0.0,
) -> Dict[str, Any]:
    """Exact scalar Rg limits of contact-only reweighting in the limits b -> +/-inf.

    Because P_b(m, Rg) ∝ P0(m, Rg) exp[-b m], the weight becomes infinitely peaked
    on one contact bin in each limit:

        b -> +inf  concentrates on the MINIMUM contact value carrying baseline mass
        b -> -inf  concentrates on the MAXIMUM contact value carrying baseline mass

    so each limiting scalar is the chosen summary of the normalized conditional
    P0(Rg | m) at that single bin.  Both limits are exact.

    These are ONLY two points of the b-trajectory, NOT its range.  They bound the
    finite-b curve exactly when the conditional moment is monotonic in supported
    contact count (then the trajectory is a convex combination of a monotonic
    sequence and cannot leave the interval its endpoints span).  When it is not
    monotonic an interior contact slice can carry a scalar outside BOTH endpoint
    values, and the endpoint interval bounds nothing:

        m=0: Rg=1, mass=0.01 | m=1: Rg=9, mass=0.98 | m=2: Rg=1, mass=0.01

    has both endpoint limits equal to 1 while b=0 gives mean Rg = 8.84.  Use
    global_rg_outer_bounds() for all-b impossibility tests; ``monotonicity_note``
    records whether these endpoints happen to coincide with that bound here.

    Which endpoint is the compact one depends on the sign of the contact-Rg
    relationship, so the interval is built with min()/max() rather than by assuming
    an ordering.
    """
    cond = _conditional_rg_by_contact(
        crg_prob, c_edges, rg_edges,
        summary=summary, probability_tolerance=tolerance,
    )
    bins = cond["supported_contact_bins"]
    centers = cond["supported_contact_centers"]
    scalar = cond["scalar"]
    scale = float(rg_scale)

    min_bin = int(bins[0])    # fewest contacts carrying mass
    max_bin = int(bins[-1])   # most contacts carrying mass
    rg_pos_inf = float(scalar[0])    # b -> +inf  -> fewest contacts
    rg_neg_inf = float(scalar[-1])   # b -> -inf  -> most contacts

    lo = min(rg_pos_inf, rg_neg_inf)
    hi = max(rg_pos_inf, rg_neg_inf)

    mono = conditional_moment_monotonicity(cond["moment"])
    monotonic = bool(mono["conditional_moment_monotonic"])

    return {
        "min_supported_contact_bin": min_bin,
        "max_supported_contact_bin": max_bin,
        "min_supported_contact_value": float(centers[0]),
        "max_supported_contact_value": float(centers[-1]),
        "contact_support_tolerance": float(tolerance),
        "rg_limit_b_pos_inf_lattice": rg_pos_inf,
        "rg_limit_b_neg_inf_lattice": rg_neg_inf,
        "rg_limit_b_pos_inf_observed": rg_pos_inf * scale,
        "rg_limit_b_neg_inf_observed": rg_neg_inf * scale,
        "endpoint_limit_min_lattice": lo,
        "endpoint_limit_max_lattice": hi,
        "endpoint_limit_min_observed": lo * scale,
        "endpoint_limit_max_observed": hi * scale,
        "conditional_moment_monotonic": monotonic,
        "conditional_moment_direction": mono["conditional_moment_direction"],
        "conditional_moment_tolerance": mono["conditional_moment_tolerance"],
        "is_exact_extremal_bound": monotonic,
        "rg_scale_used": scale,
        "monotonicity_note": (
            "The conditional moment is monotonic in supported contact count, so "
            "these two endpoint limits ARE the exact extrema of the scalar Rg over "
            "all real b. Values strictly between them are not thereby all attained."
            if monotonic else
            "The conditional moment is NOT monotonic in supported contact count, so "
            "these endpoint limits do NOT bound the finite-b trajectory: an interior "
            "contact slice can carry a scalar outside both. Use global_outer_bound "
            "for any all-b claim."
        ),
        "interpretation": (
            "Exact scalar-Rg values of this fixed joint baseline under contact-only "
            "reweighting in the limits b -> +/-inf, assuming the baseline support is "
            "exact. b -> +inf concentrates on the minimum-contact slice and b -> -inf "
            "on the maximum-contact slice. These are two points of the trajectory; "
            "they are its extrema only when the conditional moment is monotonic. This "
            "concerns the SCALAR SUMMARY only, not full-distribution support."
        ),
    }


def asymptotic_rg_limits(
    crg_prob: np.ndarray,
    c_edges: np.ndarray,
    rg_edges: np.ndarray,
    *,
    summary: str,
    rg_scale: float,
    tolerance: float = 0.0,
) -> Dict[str, Any]:
    """Deprecated alias for endpoint_rg_limits(); see that function.

    The former name and its ``asymptotic_reachable_rg_min/max`` keys asserted that
    the two b -> +/-inf endpoints were the all-b reachable range.  That is false for
    a non-monotonic conditional moment.  Retained only so existing callers keep
    working; the legacy keys are mirrored but must not drive scientific validity.
    Use endpoint_rg_limits() plus global_rg_outer_bounds() instead.
    """
    _warnings.warn(
        "asymptotic_rg_limits() is deprecated: the b -> +/-inf endpoints are not "
        "the all-b reachable range unless the conditional moment is monotonic. Use "
        "endpoint_rg_limits() for the endpoints and global_rg_outer_bounds() for a "
        "rigorous all-b bound.",
        DeprecationWarning,
        stacklevel=2,
    )
    out = endpoint_rg_limits(
        crg_prob, c_edges, rg_edges,
        summary=summary, rg_scale=rg_scale, tolerance=tolerance,
    )
    out["asymptotic_reachable_rg_min"] = out["endpoint_limit_min_lattice"]
    out["asymptotic_reachable_rg_max"] = out["endpoint_limit_max_lattice"]
    out["asymptotic_reachable_rg_min_observed"] = out["endpoint_limit_min_observed"]
    out["asymptotic_reachable_rg_max_observed"] = out["endpoint_limit_max_observed"]
    out["deprecated"] = True
    return out


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
    rg_lower_lattice: Optional[np.ndarray] = None,
    rg_upper_lattice: Optional[np.ndarray] = None,
    file_prefix: str = "rg_feasibility",
    scale_label: str = "fixed scale",
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

    The scan covers only the FINITE interval [bias_min, bias_max], so it can never
    prove that no real b reproduces a target.  Two further quantities are computed
    alongside it and reported separately, because they mean different things:

      endpoint_limits     the exact b -> +/-inf values (endpoint_rg_limits).  Two
                          POINTS of the trajectory; they bound it only when the
                          conditional moment is monotonic in contact count.
      global_outer_bound  a rigorous NECESSARY bound over all real b
                          (global_rg_outer_bounds).  The only quantity here that
                          may license an all-b impossibility claim.

    Writes <file_prefix>.csv, <file_prefix>_summary.json, and (unless plots are
    disabled) <file_prefix>.png.  ``scale_label`` names the mapping being probed
    ("fixed scale", "nominal scale", "fitted scale") and appears in the outputs so
    a nominal-scale and a fitted-scale scan can never be confused.

    Returns the summary dict.  Does not raise on zero support overlap: it reports
    ``reachability_status == "zero_support_overlap"`` and leaves the decision to
    abort to the caller, because a free-scale diagnostic run may legitimately
    continue past a nominal-scale support failure.
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
    if not np.isfinite(rg_scale) or rg_scale <= 0.0:
        raise ValueError(f"rg_scale must be finite and positive, got {rg_scale!r}")

    validate_joint_baseline(c_edges, rg_edges, crg_prob)
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

    # Exact b -> +/-inf endpoints. Two points of the trajectory, NOT its range:
    # they bound it only when the conditional moment is monotonic in contact count.
    endpoints = endpoint_rg_limits(
        crg_prob, c_edges, rg_edges, summary=summary, rg_scale=rg_scale
    )
    ep_pos = float(endpoints["rg_limit_b_pos_inf_lattice"])
    ep_neg = float(endpoints["rg_limit_b_neg_inf_lattice"])
    ep_lo = float(endpoints["endpoint_limit_min_lattice"])
    ep_hi = float(endpoints["endpoint_limit_max_lattice"])
    moment_monotonic = bool(endpoints["conditional_moment_monotonic"])

    # The rigorous all-b bound: the ONLY basis on which this function may claim a
    # target is unreachable by ANY real bias rather than merely by the scanned ones.
    outer = global_rg_outer_bounds(
        crg_prob, c_edges, rg_edges, summary=summary, rg_scale=rg_scale,
        rg_target_lattice=rg_target_lattice,
    )
    outer_lo = float(outer["global_outer_rg_min_lattice"])
    outer_hi = float(outer["global_outer_rg_max_lattice"])
    n_outside_outer = int(outer["n_targets_outside_global_outer_bound"])
    target_within_outer = bool(outer["target_within_global_outer_bound"])
    # When the moment is monotonic the endpoints coincide with the outer bound, so
    # the endpoint interval is then genuinely extremal (recorded, not re-derived).
    outer["is_exact_reachable_range"] = False
    outer["endpoints_coincide_with_outer_bound"] = bool(
        moment_monotonic
        and abs(ep_lo - outer_lo) <= 1e-9 * max(1.0, abs(outer_lo))
        and abs(ep_hi - outer_hi) <= 1e-9 * max(1.0, abs(outer_hi))
    )

    # Support overlap between the target bounds and the baseline Rg support,
    # via the same helper the driver uses. The full lower/upper range is
    # preferred; central values are the fallback when the caller has no bounds.
    if rg_lower_lattice is not None and rg_upper_lattice is not None:
        support_lo_arr, support_hi_arr = rg_lower_lattice, rg_upper_lattice
        support_basis = "target lower/upper bounds"
    else:
        support_lo_arr, support_hi_arr = rg_target_lattice, rg_target_lattice
        support_basis = "target central values"
    support_info = rg_support_overlap(
        support_lo_arr, support_hi_arr, baseline_rg_min, baseline_rg_max,
        basis=support_basis,
    )
    target_support_min = support_info["target_support_min_lattice"]
    target_support_max = support_info["target_support_max_lattice"]
    support_overlap_lo = support_info["support_overlap_lo"]
    support_overlap_hi = support_info["support_overlap_hi"]
    zero_support_overlap = support_info["zero_support_overlap"]

    # ---- classification ------------------------------------------------------
    # Computed BEFORE any warning is emitted, so that exactly one reachability
    # verdict exists and the warnings are a rendering of it. Deriving warnings
    # independently is what previously allowed "impossible for any real bias" and
    # "inconclusive, widen the scan" to be printed about the same target.
    n_unreach = _count_outside_range(rg_target_lattice, reach_lo_lat, reach_hi_lat)
    target_within_reachable = bool(n_unreach == 0)
    rg_span_lat = reach_hi_lat - reach_lo_lat
    max_abs_dg = float(np.max(np.abs(d_rg_db))) if d_rg_db.size else 0.0

    # Boundary handling. A target that is reachable at an INTERIOR scan point is
    # settled: because Rg(b) is continuous, any value inside the interior min/max
    # envelope is attained at some interior bias, so widening the scan cannot
    # overturn it. Only a target that is NOT resolved in the interior, yet is
    # approached at a scan boundary, makes the finite-scan verdict inconclusive.
    # Using argmin over the WHOLE grid (as before) could pick a boundary index for
    # a value that is also attained in the interior, wrongly flagging a reachable
    # target as boundary_limited.
    edge_frac = 0.02
    n_edge = max(1, int(np.ceil(edge_frac * bias_grid.size)))
    if bias_grid.size > 2 * n_edge:
        interior_pred = pred_lat[n_edge:bias_grid.size - n_edge]
    else:
        interior_pred = pred_lat  # too few points to define an interior band
    reach_lo_int = float(interior_pred.min())
    reach_hi_int = float(interior_pred.max())

    boundary_reached = False
    if rg_span_lat > 0:
        # Targets not resolvable within the interior envelope (same tolerance as
        # the full-range reachability test).
        int_tol = RG_REACH_RTOL * max(abs(reach_lo_int), abs(reach_hi_int), 1.0)
        unresolved_interior = (
            (rg_target_lattice < reach_lo_int - int_tol)
            | (rg_target_lattice > reach_hi_int + int_tol)
        )
        if np.any(unresolved_interior):
            # For those unresolved targets only, is the closest achievable
            # prediction at a boundary band (so a wider scan could resolve it)?
            idx = np.array(
                [int(np.argmin(np.abs(pred_lat - t))) for t in rg_target_lattice],
                dtype=int,
            )
            at_edge = (idx < n_edge) | (idx >= bias_grid.size - n_edge)
            boundary_reached = bool(np.any(unresolved_interior & at_edge))

    # THE authoritative status. classify_scientific_validity() consumes this rather
    # than re-deriving its own ordering from the raw flags.
    if zero_support_overlap:
        reachability_status = "zero_support_overlap"
    elif not target_within_outer:
        reachability_status = "outside_global_outer_bound"
    elif boundary_reached:
        reachability_status = "boundary_limited"
    elif not target_within_reachable:
        reachability_status = "unreachable_within_scan"
    else:
        reachability_status = "reachable_within_scan"

    # ---- warnings ------------------------------------------------------------
    warnings: List[str] = []

    # Out-of-support values are unreachable for every real bias regardless of the
    # status ladder below, so this is reported independently of it.
    if tgt_lo_lat < baseline_rg_min or tgt_hi_lat > baseline_rg_max:
        warnings.append(
            f"Target Rg range [{tgt_lo_lat:.4g}, {tgt_hi_lat:.4g}] (lattice) is not "
            f"fully inside the baseline Rg support [{baseline_rg_min:.4g}, "
            f"{baseline_rg_max:.4g}] (lattice). Reweighting rescales P0(m, Rg) "
            f"bin-by-bin and cannot create mass where the baseline has none, so the "
            f"out-of-support values are unreachable for every real bias. This is an "
            f"exact support statement, not a finite-scan conclusion."
        )

    # Exactly one reachability warning, selected by the authoritative status. These
    # conclusions are mutually exclusive and must never be emitted together.
    if reachability_status == "zero_support_overlap":
        warnings.append(
            f"ZERO Rg support overlap at this scale: the target support "
            f"[{target_support_min:.4g}, {target_support_max:.4g}] (lattice, from "
            f"{support_basis}) does not intersect the joint baseline Rg support "
            f"[{baseline_rg_min:.4g}, {baseline_rg_max:.4g}] (lattice). The scalar "
            f"fit is meaningless at this scale."
        )
    elif reachability_status == "outside_global_outer_bound":
        warnings.append(
            f"{n_outside_outer} of {rg_target_lattice.size} target Rg value(s) lie "
            f"outside the global scalar-Rg outer bound [{outer_lo:.4g}, "
            f"{outer_hi:.4g}] (lattice) = [{outer_lo * rg_scale:.4g}, "
            f"{outer_hi * rg_scale:.4g}] (observed) of this fixed joint baseline and "
            f"cannot be reproduced by any real contact bias b. The bound is the "
            f"min/max of the conditional scalar-{summary} Rg over the supported "
            f"contact bins; every biased prediction is a convex combination of those "
            f"conditional moments, so no real b can leave it. Changing this requires "
            f"a different baseline, chain length, or rg_scale -- not a wider scan."
        )
    elif reachability_status == "boundary_limited":
        warnings.append(
            f"The reachability conclusion is not conclusive because at least one "
            f"target is matched or approached at the boundary of the scanned bias "
            f"interval [{bias_min:g}, {bias_max:g}]. The target does lie inside the "
            f"rigorous global outer bound [{outer_lo:.4g}, {outer_hi:.4g}] (lattice), "
            f"so nothing here indicates impossibility. Repeat with a wider interval "
            f"using --rg-bias-min/--rg-bias-max, or treat the implied bias as a lower "
            f"bound in magnitude."
        )
    elif reachability_status == "unreachable_within_scan":
        warnings.append(
            f"{n_unreach} of {rg_target_lattice.size} target Rg value(s) were not "
            f"reached within the scanned bias interval [{bias_min:g}, {bias_max:g}], "
            f"whose reachable scalar-{summary} range is [{reach_lo_lat:.4g}, "
            f"{reach_hi_lat:.4g}] (lattice) = [{reach_lo_lat * rg_scale:.4g}, "
            f"{reach_hi_lat * rg_scale:.4g}] (observed). They remain inside the "
            f"rigorous global outer bound [{outer_lo:.4g}, {outer_hi:.4g}] (lattice), "
            f"so this finite scan does not prove impossibility; widen the bias "
            f"interval with --rg-bias-min/--rg-bias-max."
        )

    # Independent of reachability: these describe identifiability, not possibility.
    baseline_corr = base["corr_contact_rg"]
    if np.isfinite(baseline_corr) and abs(baseline_corr) < WEAK_CORR_THRESHOLD:
        warnings.append(
            f"Baseline contact-Rg correlation is weak (|corr| = {abs(baseline_corr):.3g} "
            f"< {WEAK_CORR_THRESHOLD}). Contacts barely constrain Rg, so the scalar-Rg "
            f"fit is close to non-identifiable."
        )

    insensitive_threshold = 1e-3
    if max_abs_dg < insensitive_threshold:
        warnings.append(
            f"Rg is nearly insensitive to contact bias (max |dRg/db| = {max_abs_dg:.3g} "
            f"< {insensitive_threshold} lattice units per unit bias) over the scanned "
            f"window; the bias cannot drive a collapse transition."
        )

    # Advisory: the endpoint limits are in the outputs, and for a non-monotonic
    # conditional moment they must not be read as the trajectory's range.
    if not moment_monotonic:
        warnings.append(
            f"The conditional {outer['conditional_moment_quantity']} is NOT monotonic "
            f"in contact count, so the b -> +/-inf endpoint limits "
            f"[{ep_lo:.4g}, {ep_hi:.4g}] (lattice) do NOT bound the finite-b "
            f"trajectory: an interior contact slice carries a scalar outside them. "
            f"All-b claims here use the global outer bound [{outer_lo:.4g}, "
            f"{outer_hi:.4g}] (lattice) instead."
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
        "rg_scale_used": float(rg_scale),
        "scale_label": str(scale_label),
        "file_prefix": str(file_prefix),
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
        "reachability_status": reachability_status,
        "max_abs_d_rg_db": float(max_abs_dg),
        "derivative_check": deriv_check,

        # --- the three distinct reachability concepts, never conflated ---------
        "finite_scan": {
            "bias_min": float(bias_min),
            "bias_max": float(bias_max),
            "bias_points": int(bias_points),
            "reachable_rg_min": float(reach_lo_lat),
            "reachable_rg_max": float(reach_hi_lat),
            "reachable_rg_min_observed": float(reach_lo_lat * rg_scale),
            "reachable_rg_max_observed": float(reach_hi_lat * rg_scale),
            "target_within_scan_range": target_within_reachable,
            "n_targets_outside_scan_range": n_unreach,
            "target_reached_only_at_bias_boundary": bool(boundary_reached),
            "note": (
                "Range of the scalar Rg summary over the SCANNED bias interval only. "
                "A target outside it was not reached within that interval; because "
                "the scan is finite this is NEVER proof that no real b reproduces it."
            ),
        },
        "endpoint_limits": {
            "b_pos_inf": float(ep_pos),
            "b_neg_inf": float(ep_neg),
            "b_pos_inf_observed": float(ep_pos * rg_scale),
            "b_neg_inf_observed": float(ep_neg * rg_scale),
            "endpoint_min": ep_lo,
            "endpoint_max": ep_hi,
            "endpoint_min_observed": float(ep_lo * rg_scale),
            "endpoint_max_observed": float(ep_hi * rg_scale),
            "conditional_moment_monotonic": moment_monotonic,
            "conditional_moment_direction": endpoints["conditional_moment_direction"],
            "is_exact_extremal_bound": moment_monotonic,
            "detail": endpoints,
            "note": (
                "Two exact points of the b-trajectory (b -> +/-inf), not its range. "
                "They are the extrema over all real b only when the conditional "
                "moment is monotonic in contact count (is_exact_extremal_bound). "
                "Even then, values strictly between them are not all necessarily "
                "attained."
            ),
        },
        "global_outer_bound": {
            "min": outer_lo,
            "max": outer_hi,
            "min_observed": float(outer_lo * rg_scale),
            "max_observed": float(outer_hi * rg_scale),
            "rg_min_lattice": outer_lo,
            "rg_max_lattice": outer_hi,
            "target_within_global_outer_bound": target_within_outer,
            "n_targets_outside_global_outer_bound": n_outside_outer,
            "is_exact_reachable_range": False,
            "detail": outer,
            "note": (
                "Necessary bound only; being inside does not prove reachability."
            ),
        },
        "support_overlap": support_info,
        "warnings": warnings,
        "units_note": (
            "Fields without an explicit _observed suffix are in LATTICE units. "
            "Observed units = rg_scale_used * lattice units."
        ),
        "definition_note": (
            "Three distinct concepts: finite_scan is what the scanned bias interval "
            "reached; endpoint_limits are the two exact b -> +/-inf points; "
            "global_outer_bound is a rigorous NECESSARY bound over all real b, from "
            "the convex-combination structure of contact-only reweighting. Only "
            "global_outer_bound may license an all-b impossibility claim, and only "
            "in the outward direction: outside it is impossible, inside it is not "
            "thereby reachable."
        ),
        "reachability_status_note": (
            "zero_support_overlap | outside_global_outer_bound | boundary_limited | "
            "unreachable_within_scan | reachable_within_scan, in that precedence. "
            "This is the single authoritative verdict; scientific_validity consumes "
            "it rather than re-deriving one. unreachable_within_scan is a statement "
            "about the scanned interval, never a proof of impossibility over all "
            "real b; only outside_global_outer_bound is such a proof."
        ),

        # --- deprecated: superseded by the three blocks above ------------------
        "deprecated_fields": {
            "deprecated": True,
            "note": (
                "asymptotic_reachable_rg_* and target_within_asymptotic_reachable_"
                "range asserted that the two b -> +/-inf endpoints were the all-b "
                "reachable range. That is FALSE when the conditional moment is not "
                "monotonic in contact count. Mirrored from endpoint_limits for "
                "backward compatibility only; they do not drive scientific validity. "
                "Use global_outer_bound for all-b claims."
            ),
            "asymptotic_reachable_rg_min": ep_lo,
            "asymptotic_reachable_rg_max": ep_hi,
            "asymptotic_reachable_rg_min_observed": float(ep_lo * rg_scale),
            "asymptotic_reachable_rg_max_observed": float(ep_hi * rg_scale),
            "target_within_asymptotic_reachable_range": bool(
                _count_outside_range(rg_target_lattice, ep_lo, ep_hi) == 0
            ),
        },
    }

    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / f"{file_prefix}.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        # Rectangular: row 0 IS the header and every row has the same width, so
        # pandas.read_csv(path) / csv.DictReader(fh) work with no skiprows= or
        # comment= argument. Scale provenance rides along as repeated columns
        # rather than a ragged banner row, so a nominal and a fitted scan can
        # still never be read as one table.
        writer.writerow([
            "bias", "pred_rg_lattice", "pred_rg_observed", "mean_contacts",
            "cov_contact_rg", "corr_contact_rg", "d_rg_db",
            "rg_scale_used", "scale_label", "rg_summary",
        ])
        for i, b in enumerate(bias_grid):
            writer.writerow([
                f"{b:.8g}", f"{pred_lat[i]:.8g}", f"{pred_obs[i]:.8g}",
                f"{mean_ct[i]:.8g}", f"{cov_arr[i]:.8g}",
                ("" if not np.isfinite(corr_arr[i]) else f"{corr_arr[i]:.8g}"),
                f"{d_rg_db[i]:.8g}",
                f"{rg_scale:.10g}", scale_label, summary,
            ])
    print(f"Saved: {csv_path}")

    json_path = outdir / f"{file_prefix}_summary.json"
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

        # Global outer bound: the only lines here that carry an all-b meaning.
        ax.axhline(outer_lo * rg_scale, color="tab:green", lw=1.2, ls="--",
                   label="global outer bound (necessary, all real b)")
        ax.axhline(outer_hi * rg_scale, color="tab:green", lw=1.2, ls="--")

        # Endpoint limits: two POINTS of the trajectory. Marked distinctly, and
        # never drawn as a spanning range when they do not bound it.
        ax.plot(bias_grid[-1], ep_neg * rg_scale, marker=">", ms=9,
                color="tab:purple", ls="none", label="endpoint limit b→−∞")
        ax.plot(bias_grid[0], ep_pos * rg_scale, marker="<", ms=9,
                color="tab:purple", ls="none", label="endpoint limit b→+∞")
        if moment_monotonic:
            # Only meaningful as an interval when the conditional moment is
            # monotonic; then it coincides with the outer bound.
            ax.axhline(ep_lo * rg_scale, color="tab:purple", lw=0.9, ls="-.",
                       alpha=0.7, label="endpoint limits (extremal: moment monotonic)")
            ax.axhline(ep_hi * rg_scale, color="tab:purple", lw=0.9, ls="-.",
                       alpha=0.7)

        ax.set_xlabel("contact bias b  (P ∝ P0(m,Rg) exp[-b m])")
        ax.set_ylabel(f"scalar Rg, observed units (rg_scale={rg_scale:g})")
        title = (
            f"Bias→Rg reachability [{summary}] — {scale_label}, "
            f"rg_scale={rg_scale:.6g}\n"
            f"status: {reachability_status} (scanned bias "
            f"[{bias_min:g}, {bias_max:g}])\n"
            f"scanned [{reach_lo_lat * rg_scale:.4g}, {reach_hi_lat * rg_scale:.4g}], "
            f"outer bound [{outer_lo * rg_scale:.4g}, {outer_hi * rg_scale:.4g}], "
            f"target [{float(rg_target_observed.min()):.4g}, "
            f"{float(rg_target_observed.max()):.4g}]"
        )
        if not moment_monotonic:
            title += (
                "\nEndpoint limits do not bound the finite-b trajectory; "
                "conditional moment is nonmonotonic."
            )
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=7)
        fig.tight_layout()
        p = outdir / f"{file_prefix}.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {p}")

    return summary_dict


def run_rg_contact_slice_diagnostic(
    crg_prob: np.ndarray,
    c_edges: np.ndarray,
    rg_edges: np.ndarray,
    rg_target_lattice: np.ndarray,
    rg_target_observed: np.ndarray,
    *,
    rg_scale: float,
    summary: str,
    outdir: Path,
    model_name: str,
    baseline_support: Dict[str, Any],
    file_prefix: str = "rg_feasibility",
    scale_label: str = "fixed scale",
    potential_kind: str = "contact_quadratic",
    potential_definition: str = "",
) -> Dict[str, Any]:
    """Model-independent reachability diagnostic for a NONLINEAR contact potential.

    run_rg_feasibility_scan() sweeps a single scalar bias b and reads the result
    as the reachable set of the model.  That reading is valid only when the
    contact potential is u = b*m, because only then does one number index the
    whole family of reweightings.  Every nonlinear potential -- the
    contact-quadratic

        u(m,T;N) = b(T)*m + kappa(T)*m^2/(2N)

    and the saturating-cooperative

        u(m,T;N) = N*[b(T)*q - A0*q^2/(1 + (q/q_sat)^2)],   q = m/N

    alike -- has a multi-parameter weight family, so a point on a
    one-dimensional b-scan is not a model prediction and the b -> +/-inf
    endpoint limits describe a limit the model never takes.  Running the scan
    anyway would produce numbers that look like feasibility statements but are
    not, so it is not run.

    What survives unchanged, because neither depends on the potential's form:

      support overlap     the target and baseline Rg supports either intersect
                          or they do not
      global outer bound  every contact-only reweighting -- ANY u(m), linear or
                          not -- leaves each conditional P0(Rg | m) untouched and
                          only re-weights the contact marginal, so the biased
                          scalar is a convex combination over contact slices and
                          is bounded by their extremes

    The per-contact-slice conditional Rg table underlying that bound is written
    out as the diagnostic to inspect in place of the bias scan.

    Writes ``<file_prefix>_summary.json`` and
    ``<file_prefix>_contact_slices.csv``.  It deliberately does NOT write
    ``<file_prefix>.csv`` or ``<file_prefix>.png``: those are the bias-scan
    artifacts, and an empty or fabricated one would invite exactly the
    interpretation this function exists to prevent.
    """
    outdir = Path(outdir)
    scale = float(rg_scale)
    rg_target_lattice = np.asarray(rg_target_lattice, dtype=float)
    rg_target_observed = np.asarray(rg_target_observed, dtype=float)

    outer = global_rg_outer_bounds(
        crg_prob, c_edges, rg_edges,
        summary=summary, rg_scale=scale, rg_target_lattice=rg_target_lattice,
    )

    definition = potential_definition or str(
        MODEL_REGISTRY[model_name]["potential_definition"]
    )
    not_applicable_reason = (
        f"Model {model_name!r} has a potential that is nonlinear in m "
        f"(potential_kind={potential_kind!r}): {definition}. The finite "
        f"one-dimensional scan over a single scalar bias b, and the reading of "
        f"its b -> +/-inf endpoints as limiting predictions, both presuppose a "
        f"potential linear in m. Neither is applicable here, so no bias scan was "
        f"run and no reachable-range or endpoint-limit numbers are reported. The "
        f"global outer bound below and the support-overlap check are unaffected: "
        f"they hold for ANY contact-only reweighting."
    )

    warnings: List[str] = []
    if baseline_support.get("zero_support_overlap"):
        warnings.append(
            ZERO_SUPPORT_OVERLAP_MESSAGE
            + f" Target support ["
            f"{baseline_support['target_support_min_lattice']:.4g}, "
            f"{baseline_support['target_support_max_lattice']:.4g}] (lattice) does "
            f"not intersect the baseline support ["
            f"{baseline_support['baseline_support_min_lattice']:.4g}, "
            f"{baseline_support['baseline_support_max_lattice']:.4g}] (lattice)."
        )
    if outer["target_within_global_outer_bound"] is False:
        warnings.append(
            f"{outer['n_targets_outside_global_outer_bound']} target Rg value(s) "
            f"lie outside the global outer bound ["
            f"{outer['global_outer_rg_min_observed']:.4g}, "
            f"{outer['global_outer_rg_max_observed']:.4g}] (observed units). Those "
            f"targets cannot be reproduced by ANY contact-only reweighting of this "
            f"baseline, nonlinear or not."
        )

    centers = np.asarray(outer["supported_contact_centers"], dtype=float)
    bins = np.asarray(outer["supported_contact_bins"], dtype=int)
    scalar_lat = np.asarray(outer["conditional_rg_scalar_by_contact_lattice"], dtype=float)
    moment = np.asarray(outer["conditional_rg_moment_by_contact"], dtype=float)
    # The bound is computed over every strictly-positive-mass contact bin, so the
    # marginal that indexes those bins is just the joint's contact marginal.
    marginal = np.asarray(crg_prob, dtype=float).sum(axis=1)

    slice_csv = outdir / f"{file_prefix}_contact_slices.csv"
    with open(slice_csv, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "contact_bin", "contact_value", "baseline_contact_mass",
            "conditional_rg_lattice", "conditional_rg_observed",
            "conditional_moment", "moment_quantity", "rg_scale_used",
        ])
        for k, i in enumerate(bins):
            writer.writerow([
                int(i), f"{centers[k]:.8g}", f"{marginal[int(i)]:.8g}",
                f"{scalar_lat[k]:.8g}", f"{scalar_lat[k] * scale:.8g}",
                f"{moment[k]:.8g}", outer["conditional_moment_quantity"],
                f"{scale:.10g}",
            ])
    print(f"Saved: {slice_csv}")

    summary_dict: Dict[str, Any] = {
        "diagnostic": "contact_slice_conditional_rg",
        "scale_label": scale_label,
        "model": model_name,
        "potential_kind": potential_kind,
        "potential_definition": definition,
        "quadratic_normalization": (
            MODEL_REGISTRY[model_name]["quadratic_normalization"]
        ),
        "potential_normalization": (
            MODEL_REGISTRY[model_name]["potential_normalization"]
        ),
        "m_ref": int(MODEL_REGISTRY[model_name]["m_ref"]),
        "bias_scan_applicable": False,
        "endpoint_limits_applicable": False,
        "not_applicable_reason": not_applicable_reason,
        "rg_scale_used": scale,
        "rg_summary": summary,
        "support": baseline_support,
        "global_outer_bound": {
            "min": _finite_or_none(outer["global_outer_rg_min_lattice"]),
            "max": _finite_or_none(outer["global_outer_rg_max_lattice"]),
            "min_observed": _finite_or_none(outer["global_outer_rg_min_observed"]),
            "max_observed": _finite_or_none(outer["global_outer_rg_max_observed"]),
            "contact_value_at_min": _finite_or_none(outer["contact_value_at_global_min"]),
            "contact_value_at_max": _finite_or_none(outer["contact_value_at_global_max"]),
            "target_within_global_outer_bound": outer["target_within_global_outer_bound"],
            "n_targets_outside_global_outer_bound": (
                outer["n_targets_outside_global_outer_bound"]
            ),
            "conditional_moment_direction": outer["conditional_moment_direction"],
            "conditional_moment_monotonic": outer["conditional_moment_monotonic"],
            "is_exact_reachable_range": False,
            "interpretation": outer["interpretation"],
        },
        "contact_slices": {
            "contact_bins": bins.tolist(),
            "contact_values": centers.tolist(),
            "conditional_rg_lattice": scalar_lat.tolist(),
            "conditional_rg_observed": (scalar_lat * scale).tolist(),
            "conditional_moment": moment.tolist(),
            "moment_quantity": outer["conditional_moment_quantity"],
            "csv": str(slice_csv.name),
        },
        "target_rg_min_observed": float(rg_target_observed.min()),
        "target_rg_max_observed": float(rg_target_observed.max()),
        "target_rg_min_lattice": float(rg_target_lattice.min()),
        "target_rg_max_lattice": float(rg_target_lattice.max()),
        "warnings": warnings,
    }

    json_path = outdir / f"{file_prefix}_summary.json"
    with open(json_path, "w") as fh:
        json.dump(summary_dict, fh, indent=2, cls=_NpEncoder, allow_nan=False)
    print(f"Saved: {json_path}")
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
            cfg["crg_prob"], cfg["c_edges"], cfg["rg_edges_lattice"], cfg["u_fn"],
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
        model_params, cfg["u_fn"],
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
        model_params, cfg["u_fn"], t_lo, t_hi,
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


def baseline_rg_support(
    crg_prob: np.ndarray, rg_edges: np.ndarray
) -> Tuple[float, float]:
    """Lattice-unit [min, max] Rg edges of the bins carrying baseline mass."""
    rg_marginal = np.asarray(crg_prob, dtype=float).sum(axis=0)
    nz = np.nonzero(rg_marginal > 0.0)[0]
    if nz.size == 0:
        raise ValueError("joint baseline has zero total Rg mass")
    rg_edges = np.asarray(rg_edges, dtype=float)
    return float(rg_edges[nz[0]]), float(rg_edges[nz[-1] + 1])


def rg_targets_at_scale(
    rg_target_input: np.ndarray,
    rg_lower_input: np.ndarray,
    rg_upper_input: np.ndarray,
    *,
    target_units: str,
    rg_scale: float,
) -> Dict[str, np.ndarray]:
    """Express the scalar targets in both unit systems for ONE given scale.

    The conversion is applied exactly once, in the direction fixed by
    ``target_units``:

        observed:  Rg_lattice  = Rg_observed / rg_scale
        lattice :  Rg_observed = rg_scale * Rg_lattice   (reporting only; the
                   supplied values are already lattice and are never divided)

    Callers pass rg_scale_initial for nominal diagnostics and rg_scale_effective
    for anything describing the fitted model, which is what keeps the two from
    being mixed.
    """
    if target_units not in RG_TARGET_UNITS_CHOICES:
        raise ValueError(
            f"Unknown target_units {target_units!r}. "
            f"Choose from {RG_TARGET_UNITS_CHOICES}."
        )
    if not np.isfinite(rg_scale) or rg_scale <= 0.0:
        raise ValueError(f"rg_scale must be finite and positive, got {rg_scale!r}")
    scale = float(rg_scale)
    if target_units == "observed":
        return {
            "target_lattice": rg_target_input / scale,
            "lower_lattice": rg_lower_input / scale,
            "upper_lattice": rg_upper_input / scale,
            "target_observed": rg_target_input.copy(),
            "lower_observed": rg_lower_input.copy(),
            "upper_observed": rg_upper_input.copy(),
        }
    return {
        "target_lattice": rg_target_input.copy(),
        "lower_lattice": rg_lower_input.copy(),
        "upper_lattice": rg_upper_input.copy(),
        "target_observed": scale * rg_target_input,
        "lower_observed": scale * rg_lower_input,
        "upper_observed": scale * rg_upper_input,
    }


def rg_support_overlap(
    rg_lower_lattice: np.ndarray,
    rg_upper_lattice: np.ndarray,
    baseline_support_min: float,
    baseline_support_max: float,
    *,
    basis: str = "target lower/upper bounds",
) -> Dict[str, Any]:
    """Overlap between the target Rg support and the joint baseline Rg support.

    The target support uses the full supplied lower-upper range, which is the
    most permissive honest reading of the data: if even that does not intersect
    the baseline support, no bias and no temperature model can help.

    Two intervals that merely touch at one point share no mass, so a positive-width
    target support requires ``hi > lo`` to count as overlapping.  A target support
    of zero width (identical bounds, or a caller falling back to central values) has
    no interior to compare, so it is tested by containment instead — otherwise every
    point target, including one sitting safely inside the baseline, would be
    misreported as disjoint.
    """
    target_support_min = float(np.min(np.asarray(rg_lower_lattice, dtype=float)))
    target_support_max = float(np.max(np.asarray(rg_upper_lattice, dtype=float)))
    b_lo, b_hi = float(baseline_support_min), float(baseline_support_max)
    lo = max(target_support_min, b_lo)
    hi = min(target_support_max, b_hi)

    if target_support_max <= target_support_min:
        zero = not (b_lo <= target_support_min <= b_hi)
        rule = "containment (zero-width target support)"
    else:
        zero = bool(hi <= lo)
        rule = "interval intersection with positive width"

    return {
        "target_support_min_lattice": target_support_min,
        "target_support_max_lattice": target_support_max,
        "target_support_basis": basis,
        "baseline_support_min_lattice": b_lo,
        "baseline_support_max_lattice": b_hi,
        "support_overlap_lo": lo,
        "support_overlap_hi": hi,
        "zero_support_overlap": bool(zero),
        "has_support_overlap": bool(not zero),
        "overlap_rule": rule,
    }


ZERO_SUPPORT_OVERLAP_MESSAGE = (
    "Zero Rg support overlap between the scalar target bounds and the joint "
    "baseline Rg support."
)

WEAK_CORR_THRESHOLD = 0.1

# Relative slack when testing a target against a reachable/asymptotic Rg range.
# Unreachability is the strongest claim this script makes, so it must not rest on
# the last bit of a float: the observed -> lattice conversion (x -> s*x -> s*x/s) is
# inexact for ~7% of values by up to 1 ulp (~2e-16 relative), which a strict
# inequality would report as a genuine miss. 1e-9 leaves ample headroom over that
# while staying orders of magnitude below any experimental Rg precision, so it can
# only ever make the verdict MORE conservative, never manufacture a claim.
RG_REACH_RTOL = 1e-9


def _count_outside_range(
    values: np.ndarray, lo: float, hi: float, *, rtol: float = RG_REACH_RTOL
) -> int:
    """Number of values outside [lo, hi], ignoring differences at rounding scale."""
    v = np.asarray(values, dtype=float)
    tol = rtol * max(abs(float(lo)), abs(float(hi)), 1.0)
    return int(np.sum((v < float(lo) - tol) | (v > float(hi) + tol)))


def observed_units_label(target_units: str) -> str:
    """A generic label for the observed unit system.

    The script has no CLI flag naming the physical unit, so the unit is whatever
    the data file uses. Hardcoding "nm" would be a fabricated claim — especially
    under --rg-target-units lattice, where no observed-unit data exists at all and
    the observed axis is purely derived through rg_scale_effective.
    """
    if target_units == "observed":
        return "observed units (as supplied in --rg-means-file)"
    return "observed units (derived via rg_scale_effective; input was lattice units)"


def classify_scientific_validity(
    feasibility: Optional[Dict[str, Any]],
    support: Dict[str, Any],
    *,
    is_fitted_scale: bool,
) -> Dict[str, Any]:
    """Structured validity for ONE scale, never a single overbroad Boolean.

    This CONSUMES the scan's authoritative ``reachability_status`` and maps it onto
    the scientific vocabulary; it does not re-derive an ordering from the raw flags.
    Two independent precedence ladders is exactly how the scan could report
    ``boundary_limited`` while validity reported ``outside_scanned_range`` for the
    same target.  The one mapping is:

      zero_support_overlap        -> zero_support_overlap
      outside_global_outer_bound  -> outside_global_outer_bound   (all-b impossible)
      boundary_limited            -> boundary_limited
      unreachable_within_scan     -> outside_scanned_range        (scan-scoped only)
      reachable_within_scan       -> weak_contact_rg_coupling, else supported

    A contact-quadratic model has no bias scan at all (see
    run_rg_contact_slice_diagnostic), so its ladder is shorter and stops at the
    only claim that survives without one:

      zero_support_overlap        -> zero_support_overlap
      outside_global_outer_bound  -> outside_global_outer_bound   (all-reweighting)
      otherwise                   -> unverified_no_bias_scan

    Weak coupling is only consulted once no reachability objection stands: it is a
    statement about identifiability, not about possibility.

    A clean fitted-scale result is reported as supported_as_mapping_diagnostic,
    never plain "supported": moving the scale to fit the data is a sensitivity
    probe, not an independent confirmation of the physically motivated mapping.
    """
    has_overlap = bool(support["has_support_overlap"])
    out: Dict[str, Any] = {
        "support_overlap": has_overlap,
        "reachable_within_scan": None,
        "within_global_outer_bound": None,
        "reachability_status": None,
        "conditional_moment_monotonic": None,
        "contact_rg_correlation": None,
        "status": None,
    }
    if feasibility is None:
        out["status"] = (
            "zero_support_overlap" if not has_overlap else "unverified_no_scan"
        )
        out["note"] = (
            "No feasibility scan was run (--rg-feasibility-scan not given); "
            "reachability is UNVERIFIED."
            if has_overlap else ZERO_SUPPORT_OVERLAP_MESSAGE
        )
        return out

    if feasibility.get("bias_scan_applicable") is False:
        # Nonlinear model: no bias scan was run, so there is no scan-scoped
        # reachability verdict to report. The global outer bound still applies
        # -- it holds for ANY contact-only reweighting -- and it is the only
        # impossibility claim available here. Absent it, the honest status is
        # "unverified", never "supported".
        within_bound = feasibility["global_outer_bound"][
            "target_within_global_outer_bound"
        ]
        out.update({
            "within_global_outer_bound": (
                None if within_bound is None else bool(within_bound)
            ),
            "reachability_status": "bias_scan_not_applicable",
            "conditional_moment_monotonic": bool(
                feasibility["global_outer_bound"]["conditional_moment_monotonic"]
            ),
        })
        if not has_overlap:
            out["status"] = "zero_support_overlap"
            out["note"] = ZERO_SUPPORT_OVERLAP_MESSAGE
        elif within_bound is False:
            out["status"] = "outside_global_outer_bound"
            out["note"] = (
                "Target Rg lies outside the global outer bound, which holds for "
                "ANY contact-only reweighting including this nonlinear one. "
                "This is an impossibility claim and does not depend on the "
                "bias scan."
            )
        else:
            out["status"] = "unverified_no_bias_scan"
            out["note"] = feasibility["not_applicable_reason"]
        out["status_note"] = (
            "The finite one-dimensional bias scan and its b -> +/-inf endpoint "
            "interpretation apply only to a contact potential linear in m, so no "
            "scan-scoped reachability verdict exists for this model. "
            "outside_global_outer_bound remains a valid all-reweighting "
            "impossibility claim; unverified_no_bias_scan means no objection was "
            "testable, NOT that the target is reachable."
        )
        return out

    scan_status = str(feasibility["reachability_status"])
    corr = feasibility.get("baseline_contact_rg_correlation")
    out.update({
        "reachable_within_scan": bool(feasibility["target_within_reachable_range"]),
        "within_global_outer_bound": bool(
            feasibility["global_outer_bound"]["target_within_global_outer_bound"]
        ),
        "reachability_status": scan_status,
        "conditional_moment_monotonic": bool(
            feasibility["endpoint_limits"]["conditional_moment_monotonic"]
        ),
        "contact_rg_correlation": corr,
    })

    weak_coupling = corr is not None and abs(float(corr)) < WEAK_CORR_THRESHOLD

    # The scan's status is authoritative. Support overlap is cross-checked because
    # the caller may have computed it from bounds the scan never saw; the two agree
    # by construction when the scan was given rg_lower/upper_lattice.
    if not has_overlap or scan_status == "zero_support_overlap":
        status = "zero_support_overlap"
    elif scan_status == "outside_global_outer_bound":
        status = "outside_global_outer_bound"
    elif scan_status == "boundary_limited":
        status = "boundary_limited"
    elif scan_status == "unreachable_within_scan":
        status = "outside_scanned_range"
    elif weak_coupling:
        status = "weak_contact_rg_coupling"
    else:
        status = "supported_as_mapping_diagnostic" if is_fitted_scale else "supported"
    out["status"] = status
    out["status_note"] = (
        "outside_global_outer_bound is the only status here that asserts "
        "impossibility for all real b. outside_scanned_range and boundary_limited "
        "are scoped to the scanned bias interval and are not impossibility claims. "
        "supported means no objection survived -- it is not proof of correctness, "
        "and optimizer convergence never contributes to it."
    )
    return out


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

    # The argument parser enforces most of this for CLI runs, but this driver is
    # also called directly (tests, notebooks, other scripts), where argparse never
    # runs. Validate here so a direct call with rg_scale=0 raises a clear error
    # instead of dividing by zero deep in the fit -- and so it raises BEFORE the
    # output directory below is created, leaving no half-written run behind.
    if not np.isfinite(args.rg_scale) or args.rg_scale <= 0.0:
        raise ValueError(
            f"--rg-scale must be finite and strictly positive, got {args.rg_scale!r}. "
            f"It is observed units per lattice unit and is divided by to map "
            f"observed targets into lattice units."
        )
    if not np.isfinite(args.rg_scale_min) or args.rg_scale_min <= 0.0:
        raise ValueError(
            f"--rg-scale-min must be finite and strictly positive, got "
            f"{args.rg_scale_min!r}"
        )
    if not np.isfinite(args.rg_scale_max) or args.rg_scale_max <= 0.0:
        raise ValueError(
            f"--rg-scale-max must be finite and strictly positive, got "
            f"{args.rg_scale_max!r}"
        )
    if args.fit_rg_scale and not (args.rg_scale_min < args.rg_scale_max):
        raise ValueError(
            f"--rg-scale-min ({args.rg_scale_min}) must be strictly less than "
            f"--rg-scale-max ({args.rg_scale_max}) when fitting the scale."
        )
    if not np.isfinite(args.rg_bias_min) or not np.isfinite(args.rg_bias_max):
        raise ValueError("--rg-bias-min/--rg-bias-max must be finite")
    if not (args.rg_bias_min < args.rg_bias_max):
        raise ValueError(
            f"--rg-bias-min ({args.rg_bias_min}) must be strictly less than "
            f"--rg-bias-max ({args.rg_bias_max})"
        )
    if int(args.rg_bias_points) < 3:
        raise ValueError(
            f"--rg-bias-points must be >= 3, got {args.rg_bias_points}"
        )
    if int(args.n_restarts) < 1:
        raise ValueError(f"--n_restarts must be >= 1, got {args.n_restarts}")
    if int(args.bootstrap) < 0:
        raise ValueError(f"--bootstrap must be >= 0, got {args.bootstrap}")

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

    # Single explicit gate: shapes, finiteness, monotonicity, non-negativity,
    # positive mass. Runs before any clipping so a NaN edge cannot slip through a
    # bare np.diff(edges) <= 0 test.
    crg_prob, baseline_notes = sanitize_joint_baseline(
        c_edges, rg_edges_lattice, crg_prob
    )
    for note in baseline_notes:
        print(f"  NOTE: {note}")

    rg_centers_lattice = 0.5 * (rg_edges_lattice[:-1] + rg_edges_lattice[1:])

    # Bending penalty and chain length are BASELINE metadata: the bending weight
    # is already inside P0(m, Rg) and is never re-applied or fitted here, and the
    # chain length only normalizes the nonlinear contact term.
    kappa_bend = resolve_kappa_bend(
        read_baseline_kappa_bend(b_data), args.kappa_bend, str(args.baseline)
    )
    bending_enabled = bool(kappa_bend != 0.0)
    fit_chain_length = resolve_chain_length(
        read_baseline_chain_length(b_data), args.chain_length,
        model_name=args.model, baseline_path=str(args.baseline),
    )
    potential_kind = str(MODEL_REGISTRY[args.model]["potential_kind"])
    quadratic_normalization = MODEL_REGISTRY[args.model]["quadratic_normalization"]
    potential_normalization = MODEL_REGISTRY[args.model]["potential_normalization"]
    potential_definition = str(MODEL_REGISTRY[args.model]["potential_definition"])
    m_ref = int(MODEL_REGISTRY[args.model]["m_ref"])

    # ---- the three scales -------------------------------------------------
    # rg_scale_initial   the physically motivated --rg-scale, never overwritten
    # rg_scale_fitted    the optimizer's value, only in --fit-rg-scale mode
    # rg_scale_effective what every FITTED-model quantity is mapped with
    # The effective scale is not known until after the fit; everything before it
    # is nominal-scale by construction and is named accordingly.
    rg_scale_initial = float(args.rg_scale)

    # ---- targets in both unit systems (conversion applied exactly once) -----
    # Nominal-scale views, used for the pre-fit diagnostics only. The
    # fitted-model views are recomputed with rg_scale_effective after the fit.
    if target_units == "observed":
        rg_target_observed_nominal = rg_target_input.copy()
        rg_target_lattice_nominal = rg_target_input / rg_scale_initial
        rg_lower_lattice_nominal = rg_lower_input / rg_scale_initial
        rg_upper_lattice_nominal = rg_upper_input / rg_scale_initial
    else:
        # Targets are already lattice values: never divide them by any scale.
        rg_target_lattice_nominal = rg_target_input.copy()
        rg_lower_lattice_nominal = rg_lower_input.copy()
        rg_upper_lattice_nominal = rg_upper_input.copy()
        rg_target_observed_nominal = rg_scale_initial * rg_target_input

    # ---- nominal-scale support overlap gate --------------------------------
    # A fixed-scale run is a production fit: zero overlap means the target and the
    # baseline describe disjoint Rg ranges, so the fit is meaningless and must not
    # proceed. A free-scale run is a diagnostic: the scale may yet move the target
    # into the baseline's support, so the nominal failure is recorded, not fatal,
    # and is rechecked at rg_scale_effective after the fit.
    baseline_rg_lo, baseline_rg_hi = baseline_rg_support(crg_prob, rg_edges_lattice)
    nominal_support = rg_support_overlap(
        rg_lower_lattice_nominal, rg_upper_lattice_nominal,
        baseline_rg_lo, baseline_rg_hi,
    )
    if nominal_support["zero_support_overlap"]:
        if not args.fit_rg_scale:
            raise ValueError(
                ZERO_SUPPORT_OVERLAP_MESSAGE
                + f" Target support [{nominal_support['target_support_min_lattice']:.4g}, "
                f"{nominal_support['target_support_max_lattice']:.4g}] (lattice) does "
                f"not intersect the baseline support [{baseline_rg_lo:.4g}, "
                f"{baseline_rg_hi:.4g}] (lattice) at rg_scale={rg_scale_initial:.6g}. "
                f"A fixed-scale production fit cannot proceed. Check --rg-scale, the "
                f"baseline, or the target units."
            )
        print(
            "\n  WARNING: zero Rg support overlap at the NOMINAL scale "
            f"(rg_scale={rg_scale_initial:.6g}). The nominal-scale mapping is INVALID. "
            "Continuing only because --fit-rg-scale may move the target into the "
            "baseline support; the check is repeated at the fitted scale."
        )

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
    q_fn = make_q_fn(args.model, Tref, Tscale)
    u_fn = make_contact_u_fn(args.model, Tref, Tscale, n_beads=fit_chain_length)

    pspec = build_rg_scalar_param_spec(
        args.model,
        fit_rg_scale=bool(args.fit_rg_scale),
        rg_scale=rg_scale_initial,
        rg_scale_min=float(args.rg_scale_min),
        rg_scale_max=float(args.rg_scale_max),
        n_restarts=int(args.n_restarts),
        seed=int(args.seed),
    )
    param_names: List[str] = pspec["param_names"]
    model_param_names: List[str] = pspec["model_param_names"]
    bounds: List[Tuple[float, float]] = pspec["bounds"]
    x0s: List[np.ndarray] = pspec["x0s"]

    cfg: Dict[str, Any] = {
        "crg_prob": crg_prob,
        "c_edges": c_edges,
        "rg_edges_lattice": rg_edges_lattice,
        "b_fn": b_fn,
        "q_fn": q_fn,
        "u_fn": u_fn,
        "n_beads": fit_chain_length,
        "spec": spec,
        "n_model_params": pspec["n_model_params"],
        # In fixed-scale mode this is the scale the objective uses. In free-scale
        # mode it is only the fallback for split_rg_scalar_params, which takes the
        # scale from the parameter vector instead.
        "rg_scale": rg_scale_initial,
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
    print(f"  target Rg (lattice, nominal scale): "
          f"[{rg_target_lattice_nominal.min():.4g}, {rg_target_lattice_nominal.max():.4g}]")
    print(f"  target Rg (observed, nominal scale):"
          f"[{rg_target_observed_nominal.min():.4g}, {rg_target_observed_nominal.max():.4g}]")
    print("  NOTE: the lower/upper columns are treated as DESCRIPTIVE BOUNDS, "
          "not standard errors.")
    print(f"Baseline NPZ: {args.baseline}")
    print(f"  keys: {list(b_data.files)}")
    print(f"  joint P0(m,Rg): available, shape {crg_prob.shape}")
    print(f"  kappa_bend: {kappa_bend:g} "
          f"({'bending enabled' if bending_enabled else 'no bending penalty'}; "
          f"{BEND_DEFINITION})")
    print(f"  chain length: "
          + ("not recorded" if fit_chain_length is None else f"{fit_chain_length}")
          + (f"  (used for {potential_normalization})"
             if potential_normalization else "  (not needed by this model)"))
    print(f"  potential:   [{potential_kind}]  {potential_definition}")
    print(f"               m_ref = {m_ref}")
    print(f"  Rg grid (lattice):  [{rg_edges_lattice.min():.4g}, {rg_edges_lattice.max():.4g}]")
    print(f"  Rg grid (observed, nominal scale): "
          f"[{rg_edges_lattice.min() * rg_scale_initial:.4g}, "
          f"{rg_edges_lattice.max() * rg_scale_initial:.4g}]")
    print(f"Model : {args.model}  —  {spec['description']}")
    print(f"  Rg summary: {args.rg_summary}   objective: {args.rg_mean_loss} "
          f"(mean per temperature)")
    print(f"  rg_scale (initial): {rg_scale_initial:.8g} observed units per lattice unit"
          + ("  [FITTED as a free parameter]" if args.fit_rg_scale else "  [FIXED]"))
    print(f"  Parameters: {param_names}"
          + ("  (rg_scale is appended last and is NOT a thermodynamic parameter)"
             if args.fit_rg_scale else ""))

    # ---- feasibility scan at the NOMINAL scale (diagnostic; never affects fit)
    # In fixed-scale mode this is the production feasibility result and keeps the
    # historical filenames. In free-scale mode it answers only "can the baseline
    # reproduce the data using the physically motivated input scale?", is named
    # *_nominal, and is never overwritten by the post-fit scan.
    def _report_scan(fs: Dict[str, Any], label: str) -> None:
        print(f"  [{label}] rg_scale used: {fs['rg_scale_used']:.8g}")
        print(f"  baseline contact-Rg correlation: "
              f"{fs['baseline_contact_rg_correlation']}")
        print(f"  reachable scalar Rg (lattice):  "
              f"[{fs['reachable_rg_min']:.4g}, {fs['reachable_rg_max']:.4g}]")
        print(f"  reachable scalar Rg (observed): "
              f"[{fs['reachable_rg_min_observed']:.4g}, "
              f"{fs['reachable_rg_max_observed']:.4g}]")
        ep = fs["endpoint_limits"]
        gob = fs["global_outer_bound"]
        print(f"  endpoint limits (lattice, b -> +inf / -inf): "
              f"{ep['b_pos_inf']:.4g} / {ep['b_neg_inf']:.4g}"
              f"   [conditional moment: {ep['conditional_moment_direction']}"
              + ("; extremal over all b]" if ep["is_exact_extremal_bound"]
                 else "; these do NOT bound the finite-b curve]"))
        print(f"  global outer bound (lattice, all real b): "
              f"[{gob['min']:.4g}, {gob['max']:.4g}]  "
              f"(necessary bound, not proof of reachability)")
        print(f"  target scalar Rg (observed):    "
              f"[{fs['target_rg_min_observed']:.4g}, "
              f"{fs['target_rg_max_observed']:.4g}]")
        print(f"  reachability_status: {fs['reachability_status']}")
        for w in fs["warnings"]:
            print(f"  WARNING: {w}")
        if not fs["warnings"]:
            print("  No feasibility warnings.")

    def _report_slices(fs: Dict[str, Any], label: str) -> None:
        print(f"  [{label}] rg_scale used: {fs['rg_scale_used']:.8g}")
        print(f"  NOT APPLICABLE: {fs['not_applicable_reason']}")
        gob = fs["global_outer_bound"]
        print(f"  global outer bound (lattice, ANY contact-only reweighting): "
              f"[{gob['min']:.4g}, {gob['max']:.4g}]  "
              f"(necessary bound, not proof of reachability)")
        print(f"  target scalar Rg (observed):    "
              f"[{fs['target_rg_min_observed']:.4g}, "
              f"{fs['target_rg_max_observed']:.4g}]")
        print(f"  contact-slice conditional Rg table: {fs['contact_slices']['csv']} "
              f"({len(fs['contact_slices']['contact_bins'])} supported contact bins)")
        for w in fs["warnings"]:
            print(f"  WARNING: {w}")
        if not fs["warnings"]:
            print("  No support-overlap or outer-bound warnings.")

    # The finite one-dimensional scan indexes the whole reweighting family by a
    # single scalar b, which describes a contact potential linear in m and
    # nothing else.  For any model nonlinear in m -- contact-quadratic or
    # saturating-cooperative -- the scan and its b -> +/-inf endpoint reading are
    # replaced by the model-independent slice diagnostic.
    bias_scan_applicable = potential_kind == "linear"

    nominal_prefix = "rg_feasibility_nominal" if args.fit_rg_scale else "rg_feasibility"
    nominal_label = "nominal scale" if args.fit_rg_scale else "fixed scale"

    feasibility_nominal: Optional[Dict[str, Any]] = None
    if args.rg_feasibility_scan and not bias_scan_applicable:
        print(f"\n--- Contact-slice Rg reachability [{nominal_label}] "
              f"(diagnostic; does not affect the fit) ---")
        feasibility_nominal = run_rg_contact_slice_diagnostic(
            crg_prob, c_edges, rg_edges_lattice,
            rg_target_lattice_nominal, rg_target_observed_nominal,
            rg_scale=rg_scale_initial,
            summary=args.rg_summary,
            outdir=plot_dir,
            model_name=args.model,
            baseline_support=nominal_support,
            file_prefix=nominal_prefix,
            scale_label=nominal_label,
            potential_kind=potential_kind,
            potential_definition=potential_definition,
        )
        _report_slices(feasibility_nominal, nominal_label)
    elif args.rg_feasibility_scan:
        # Console output stays ASCII: cp1252 terminals cannot encode arrows.
        print(f"\n--- Bias-to-Rg feasibility scan [{nominal_label}] "
              f"(diagnostic; does not affect the fit) ---")
        feasibility_nominal = run_rg_feasibility_scan(
            crg_prob, c_edges, rg_edges_lattice,
            rg_target_lattice_nominal, rg_target_observed_nominal,
            rg_scale=rg_scale_initial,
            summary=args.rg_summary,
            bias_min=float(args.rg_bias_min),
            bias_max=float(args.rg_bias_max),
            bias_points=int(args.rg_bias_points),
            outdir=plot_dir,
            make_plots=make_plots,
            rg_lower_lattice=rg_lower_lattice_nominal,
            rg_upper_lattice=rg_upper_lattice_nominal,
            file_prefix=nominal_prefix,
            scale_label=nominal_label,
        )
        _report_scan(feasibility_nominal, nominal_label)

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
    n_model_params = cfg["n_model_params"]
    model_params, _ = split_rg_scalar_params(
        params_all, n_model_params, cfg["fit_rg_scale"], rg_scale_initial
    )
    boundary_hits = count_boundary_hits(params_all, bounds, param_names)

    # ---- the three scales, resolved once -----------------------------------
    # Every fitted-model quantity below maps with rg_scale_effective. The initial
    # scale is kept verbatim for provenance and for the nominal diagnostics; it is
    # never overwritten. model_params carries thermodynamic parameters ONLY, so
    # b_fn never sees rg_scale.
    rg_scale_fitted = (
        float(params_all[n_model_params]) if args.fit_rg_scale else None
    )
    rg_scale_effective = (
        rg_scale_fitted if rg_scale_fitted is not None else rg_scale_initial
    )

    print("\nBest-fit parameters:")
    for name, val in zip(param_names, params_all):
        print(f"  {name} = {val:.6g}")
    if args.fit_rg_scale:
        print(f"  (rg_scale fitted within [{args.rg_scale_min:g}, {args.rg_scale_max:g}]; "
              f"initial input scale was {rg_scale_initial:g})")
    print(f"  rg_scale_initial   = {rg_scale_initial:.8g}")
    print(f"  rg_scale_fitted    = "
          + (f"{rg_scale_fitted:.8g}" if rg_scale_fitted is not None else "None (not fitted)"))
    print(f"  rg_scale_effective = {rg_scale_effective:.8g}  "
          f"(used for ALL fitted-model outputs)")
    print(f"Objective ({args.rg_mean_loss}, mean over {train_idx.size} train temps) "
          f"= {best_obj:.6g}")
    if boundary_hits:
        print(f"  WARNING: parameter(s) resting on a bound: {', '.join(boundary_hits)}")

    # ---- fitted-model target views (conversion applied exactly once) --------
    tgt_eff = rg_targets_at_scale(
        rg_target_input, rg_lower_input, rg_upper_input,
        target_units=target_units, rg_scale=rg_scale_effective,
    )
    rg_target_lattice = tgt_eff["target_lattice"]
    rg_target_observed = tgt_eff["target_observed"]
    rg_lower_lattice = tgt_eff["lower_lattice"]
    rg_upper_lattice = tgt_eff["upper_lattice"]
    rg_centers_observed = rg_scale_effective * rg_centers_lattice
    rg_edges_observed = rg_scale_effective * rg_edges_lattice

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
    q_T = np.array([q_fn(model_params, float(T)) for T in temps], dtype=float)

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

    # ---- feasibility scan at the FITTED scale ------------------------------
    # Answers a different question from the nominal scan: "can the fitted mapping
    # reproduce the data after allowing the scale to move?" It is a
    # mapping-sensitivity diagnostic, NOT the primary scientific test, and it never
    # overwrites the nominal scan's files.
    # Support overlap at the effective scale, needed by the fitted-scale
    # diagnostic below as well as by the warning that follows it.
    fitted_support = rg_support_overlap(
        rg_lower_lattice, rg_upper_lattice, baseline_rg_lo, baseline_rg_hi
    )

    feasibility_fitted: Optional[Dict[str, Any]] = None
    if args.rg_feasibility_scan and args.fit_rg_scale and not bias_scan_applicable:
        print("\n--- Contact-slice Rg reachability [fitted scale] "
              "(mapping-sensitivity diagnostic) ---")
        feasibility_fitted = run_rg_contact_slice_diagnostic(
            crg_prob, c_edges, rg_edges_lattice,
            rg_target_lattice, rg_target_observed,
            rg_scale=rg_scale_effective,
            summary=args.rg_summary,
            outdir=plot_dir,
            model_name=args.model,
            baseline_support=fitted_support,
            file_prefix="rg_feasibility_fitted",
            scale_label="fitted scale",
            potential_kind=potential_kind,
            potential_definition=potential_definition,
        )
        _report_slices(feasibility_fitted, "fitted scale")
    elif args.rg_feasibility_scan and args.fit_rg_scale:
        print("\n--- Bias-to-Rg feasibility scan [fitted scale] "
              "(mapping-sensitivity diagnostic) ---")
        feasibility_fitted = run_rg_feasibility_scan(
            crg_prob, c_edges, rg_edges_lattice,
            rg_target_lattice, rg_target_observed,
            rg_scale=rg_scale_effective,
            summary=args.rg_summary,
            bias_min=float(args.rg_bias_min),
            bias_max=float(args.rg_bias_max),
            bias_points=int(args.rg_bias_points),
            outdir=plot_dir,
            make_plots=make_plots,
            rg_lower_lattice=rg_lower_lattice,
            rg_upper_lattice=rg_upper_lattice,
            file_prefix="rg_feasibility_fitted",
            scale_label="fitted scale",
        )
        _report_scan(feasibility_fitted, "fitted scale")

    # ---- support overlap at the effective scale ----------------------------
    if args.fit_rg_scale and fitted_support["zero_support_overlap"]:
        print(
            "\n  WARNING: zero Rg support overlap remains at the FITTED scale "
            f"(rg_scale_effective={rg_scale_effective:.6g}). "
            + ZERO_SUPPORT_OVERLAP_MESSAGE
            + " The fitted result is marked INVALID."
        )

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
    no_collapse_warning = (
        "The fitted scalar Rg curve does not show a resolved temperature-driven "
        "collapse within the observed temperature interval."
    )
    if trans["collapse_detected"]:
        print(f"  T_rg_max_slope = {trans['T_rg_max_slope']:.6g} K   "
              f"(-dRg/dT = {trans['rg_max_negative_slope']:.6g} {target_units} units/K)")
    else:
        print(f"  T_rg_max_slope = None   (no resolved collapse: max -dRg/dT = "
              f"{trans['rg_max_negative_slope']:.6g} <= tolerance "
              f"{trans['slope_tolerance']:.3g} {target_units} units/K)")
    if trans["rg_half_defined"]:
        print(f"  T_rg_half      = {trans['T_rg_half']:.6g} K   "
              f"(single crossing of the endpoint midpoint "
              f"{trans['rg_half_value']:.6g})")
    elif trans["rg_half_ambiguous"]:
        print(f"  T_rg_half      = None   (AMBIGUOUS: the endpoint midpoint "
              f"{trans['rg_half_value']:.6g} is crossed "
              f"{len(trans['T_rg_half_crossings'])} times at "
              + ", ".join(f"{t:.6g}" for t in trans["T_rg_half_crossings"])
              + " K; no single half-height temperature exists)")
    else:
        print(f"  T_rg_half      = None   (curve endpoints differ by "
              f"{trans['rg_curve_span']:.3g} <= tolerance "
              f"{trans['rg_half_tolerance']:.3g} {target_units} units: the curve is "
              f"flat, so no half-height temperature is defined)")
    print("  NOTE: T_bias_zero (bias sign change) and T_rg_max_slope (steepest "
          "finite-chain collapse) are distinct quantities.")

    # ---- assemble warnings --------------------------------------------------
    warnings_list: List[str] = []
    if feasibility_nominal is not None:
        warnings_list.extend(
            f"[{nominal_label}] {w}" for w in feasibility_nominal["warnings"]
        )
    else:
        warnings_list.append(
            "Feasibility diagnostics were not run (--rg-feasibility-scan not given). "
            "Whether the joint baseline can reach the observed Rg range is UNVERIFIED."
        )
    if feasibility_fitted is not None:
        warnings_list.extend(
            f"[fitted scale] {w}" for w in feasibility_fitted["warnings"]
        )
    if not trans["collapse_detected"]:
        warnings_list.append(no_collapse_warning)
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

    # ---- support diagnostics (effective scale) -----------------------------
    target_in_support = bool(
        rg_target_lattice.min() >= baseline_rg_lo
        and rg_target_lattice.max() <= baseline_rg_hi
    )
    if not target_in_support:
        warnings_list.append(
            f"Target Rg range [{rg_target_lattice.min():.4g}, {rg_target_lattice.max():.4g}] "
            f"(lattice, at rg_scale_effective={rg_scale_effective:.6g}) is not contained "
            f"in the baseline Rg support [{baseline_rg_lo:.4g}, {baseline_rg_hi:.4g}] "
            f"(lattice)."
        )
    support_diagnostics: Dict[str, Any] = {
        "rg_scale_used": float(rg_scale_effective),
        "baseline_rg_support_lattice": [baseline_rg_lo, baseline_rg_hi],
        "baseline_rg_support_observed": [
            baseline_rg_lo * rg_scale_effective, baseline_rg_hi * rg_scale_effective
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
        "nominal_scale_support_overlap": nominal_support,
        "effective_scale_support_overlap": fitted_support,
        "inside_input_range_note": (
            "inside_input_range compares the fitted prediction against the supplied "
            "lower/upper columns. It is NOT confidence-interval coverage."
        ),
        "scale_note": (
            "All lattice/observed conversions in this block use rg_scale_effective. "
            "nominal_scale_support_overlap is the pre-fit check at rg_scale_initial."
        ),
    }

    # ---- reachability bounds at the effective scale -------------------------
    # Closed-form and cheap, so they are reported whether or not --rg-feasibility-
    # scan ran: the all-b outer bound is the single most important scientific check
    # here and must not be contingent on an optional flag.
    ep_eff = endpoint_rg_limits(
        crg_prob, c_edges, rg_edges_lattice,
        summary=args.rg_summary, rg_scale=rg_scale_effective,
    )
    gob_eff = global_rg_outer_bounds(
        crg_prob, c_edges, rg_edges_lattice,
        summary=args.rg_summary, rg_scale=rg_scale_effective,
        rg_target_lattice=rg_target_lattice,
    )
    endpoint_limits_json: Dict[str, Any] = {
        "rg_limit_b_pos_inf_lattice": _finite_or_none(
            ep_eff["rg_limit_b_pos_inf_lattice"]),
        "rg_limit_b_neg_inf_lattice": _finite_or_none(
            ep_eff["rg_limit_b_neg_inf_lattice"]),
        "rg_limit_b_pos_inf_observed": _finite_or_none(
            ep_eff["rg_limit_b_pos_inf_observed"]),
        "rg_limit_b_neg_inf_observed": _finite_or_none(
            ep_eff["rg_limit_b_neg_inf_observed"]),
        "endpoint_limit_min_lattice": _finite_or_none(
            ep_eff["endpoint_limit_min_lattice"]),
        "endpoint_limit_max_lattice": _finite_or_none(
            ep_eff["endpoint_limit_max_lattice"]),
        "conditional_moment_monotonic": bool(ep_eff["conditional_moment_monotonic"]),
        "conditional_moment_direction": str(ep_eff["conditional_moment_direction"]),
        "conditional_moment_quantity": str(gob_eff["conditional_moment_quantity"]),
        "is_exact_extremal_bound": bool(ep_eff["is_exact_extremal_bound"]),
        "rg_scale_used": float(rg_scale_effective),
        "note": str(ep_eff["monotonicity_note"]),
    }
    global_outer_bound_json: Dict[str, Any] = {
        "rg_min_lattice": _finite_or_none(gob_eff["global_outer_rg_min_lattice"]),
        "rg_max_lattice": _finite_or_none(gob_eff["global_outer_rg_max_lattice"]),
        "rg_min_observed": _finite_or_none(gob_eff["global_outer_rg_min_observed"]),
        "rg_max_observed": _finite_or_none(gob_eff["global_outer_rg_max_observed"]),
        "contact_bin_at_global_min": int(gob_eff["contact_bin_at_global_min"]),
        "contact_bin_at_global_max": int(gob_eff["contact_bin_at_global_max"]),
        "contact_value_at_global_min": _finite_or_none(
            gob_eff["contact_value_at_global_min"]),
        "contact_value_at_global_max": _finite_or_none(
            gob_eff["contact_value_at_global_max"]),
        "target_within_global_outer_bound": bool(
            gob_eff["target_within_global_outer_bound"]),
        "n_targets_outside_global_outer_bound": int(
            gob_eff["n_targets_outside_global_outer_bound"]),
        "is_exact_reachable_range": False,
        "rg_scale_used": float(rg_scale_effective),
        "note": (
            "Necessary outer bound; being inside does not prove finite-b "
            "reachability."
        ),
        "definition": (
            "min/max over supported contact bins of the conditional scalar-Rg "
            "summary of P0(Rg | m). Every contact-only-biased prediction is a convex "
            "combination of the per-bin conditional moments, so no real b can leave "
            "this interval."
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
        rg_centers_observed=rg_centers_observed,
        rg_edges_lattice=rg_edges_lattice,
        rg_edges_observed=rg_edges_observed,
        # Nominal-scale views are kept but must carry _nominal in the name so they
        # can never be mistaken for the fitted mapping.
        rg_centers_observed_nominal=rg_scale_initial * rg_centers_lattice,
        rg_edges_observed_nominal=rg_scale_initial * rg_edges_lattice,
        rg_target_lattice_nominal=rg_target_lattice_nominal,
        rg_target_observed_nominal=rg_target_observed_nominal,
        rg_summary=str(args.rg_summary),
        rg_mean_loss=str(args.rg_mean_loss),
        rg_scale=float(rg_scale_effective),
        rg_scale_input=float(rg_scale_initial),
        rg_scale_initial=float(rg_scale_initial),
        rg_scale_fitted=(
            float(rg_scale_fitted) if rg_scale_fitted is not None else np.nan
        ),
        rg_scale_effective=float(rg_scale_effective),
        rg_scale_was_fitted=bool(args.fit_rg_scale),
        params=params_all,
        param_names=np.array(param_names),
        model_params=model_params,
        model_param_names=np.array(model_param_names),
        b_T=b_T,
        q_T=q_T,
        potential_kind=potential_kind,
        quadratic_normalization=(
            "" if quadratic_normalization is None else str(quadratic_normalization)
        ),
        potential_normalization=(
            "" if potential_normalization is None else str(potential_normalization)
        ),
        potential_definition=potential_definition,
        m_ref=int(m_ref),
        fit_chain_length=(-1 if fit_chain_length is None else int(fit_chain_length)),
        kappa_bend=float(kappa_bend),
        bending_enabled=bool(bending_enabled),
        bend_definition=BEND_DEFINITION,
        # reduced_bias_by_temperature is kept as the linear-coefficient
        # compatibility alias of linear_coefficient_by_temperature.
        reduced_bias_by_temperature=b_T,
        linear_coefficient_by_temperature=b_T,
        quadratic_coefficient_by_temperature=q_T,
        train_indices=train_idx,
        validation_indices=val_idx,
        T_rg_max_slope=(
            float(trans["T_rg_max_slope"])
            if trans["T_rg_max_slope"] is not None else np.nan
        ),
        collapse_detected=bool(trans["collapse_detected"]),
        slope_tolerance=float(trans["slope_tolerance"]),
        rg_max_negative_slope=float(trans["rg_max_negative_slope"]),
        # NaN encodes "undefined", as for T_rg_max_slope above; the companion
        # flags say which of the two reasons applies.
        T_rg_half=(
            float(trans["T_rg_half"]) if trans["T_rg_half"] is not None else np.nan
        ),
        T_rg_half_crossings=np.array(trans["T_rg_half_crossings"], dtype=float),
        rg_half_defined=bool(trans["rg_half_defined"]),
        rg_half_ambiguous=bool(trans["rg_half_ambiguous"]),
        rg_half_tolerance=float(trans["rg_half_tolerance"]),
        rg_curve_span=float(trans["rg_curve_span"]),
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
    # The saturating-cooperative amplitude and saturation scale are also written
    # under their own names, so a reader never has to index into params by
    # position to recover the two parameters that define the nonlinearity.
    if potential_kind == "saturating_cooperative":
        save_kwargs["A0"] = float(model_params[2])
        save_kwargs["q_sat"] = float(model_params[3])
    np.savez_compressed(npz_path, **save_kwargs)
    print(f"\nSaved: {npz_path}")

    # ---- rg_fit_by_temperature.csv -----------------------------------------
    per_temp_csv = plot_dir / "rg_fit_by_temperature.csv"
    val_set = set(val_idx.tolist())
    with open(per_temp_csv, "w", newline="") as fh:
        writer = csv.writer(fh)
        # Rectangular: row 0 IS the header and every row has the same width, so
        # pandas.read_csv(path) / csv.DictReader(fh) work with no skiprows= or
        # comment= argument. Scale provenance rides along as repeated columns
        # rather than a ragged banner row: every lattice/observed column below uses
        # rg_scale_effective, so no column mixes nominal and fitted conversions.
        writer.writerow([
            "temp_index", "temperature", "split",
            "rg_target_input", "rg_lower_input", "rg_upper_input", "rg_target_units",
            "rg_target_lattice", "rg_target_observed",
            "rg_pred_lattice", "rg_pred_observed",
            "residual_target_units", "absolute_error_target_units",
            "squared_error_target_units", "objective_contribution",
            "inside_input_range", "b_T", "q_T",
            "rg_scale_effective", "rg_scale_initial", "rg_scale_was_fitted",
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
                bool(inside_range[i]), f"{b_T[i]:.8g}", f"{q_T[i]:.8g}",
                f"{rg_scale_effective:.10g}", f"{rg_scale_initial:.10g}",
                bool(args.fit_rg_scale),
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
                "rg_scale", f"{rg_scale_initial:.8g}", "scale_fixed", "", "", "",
                param_descriptions["rg_scale"] + " [FIXED, not optimized]",
            ])
        writer.writerow([
            "rg_scale_effective", f"{rg_scale_effective:.8g}", "scale_effective",
            "", "", "",
            "scale used for every fitted-model lattice/observed conversion",
        ])
        if trans["T_bias_zero"] is not None:
            writer.writerow([
                "T_bias_zero", f"{trans['T_bias_zero']:.8g}", "derived", "", "", "",
                "temperature where b(T)=0 inside the observed range (K)",
            ])
        writer.writerow([
            "T_rg_max_slope",
            ("" if trans["T_rg_max_slope"] is None else f"{trans['T_rg_max_slope']:.8g}"),
            "derived", "", "", "",
            ("temperature of steepest fitted collapse, argmax of -dRg/dT (K)"
             if trans["collapse_detected"]
             else "no resolved collapse detected within the observed interval"),
        ])
        writer.writerow([
            "T_rg_half",
            ("" if trans["T_rg_half"] is None else f"{trans['T_rg_half']:.8g}"),
            "derived", "", "", "",
            ("temperature crossing the midpoint Rg between the interval ends (K)"
             if trans["rg_half_defined"] else
             f"undefined: the endpoint midpoint is crossed "
             f"{len(trans['T_rg_half_crossings'])} times (ambiguous)"
             if trans["rg_half_ambiguous"] else
             "undefined: the fitted curve is flat between the interval ends"),
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

    observed_unit_label = observed_units_label(target_units)

    # The nominal/fixed-scale branch is the primary scientific test; the
    # fitted-scale branch is a mapping-sensitivity diagnostic. They are classified
    # separately so a fitted model is never judged by the nominal scan's verdict.
    scientific_validity: Dict[str, Any] = {
        "fixed_or_nominal_scale": classify_scientific_validity(
            feasibility_nominal, nominal_support, is_fitted_scale=False
        ),
        "fitted_scale": (
            classify_scientific_validity(
                feasibility_fitted, fitted_support, is_fitted_scale=True
            ) if args.fit_rg_scale else None
        ),
        "note": (
            "fixed_or_nominal_scale is the primary scientific test, evaluated at "
            "rg_scale_initial. fitted_scale is a mapping-sensitivity diagnostic at "
            "rg_scale_effective and is null in fixed-scale mode. Optimizer "
            "convergence is NOT evidence of validity: these statuses depend on "
            "support overlap, reachability, the asymptotic limits, and contact-Rg "
            "coupling."
        ),
    }

    summary_json: Dict[str, Any] = {
        "mode": "rg_scalar",
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "model_api_version": MODEL_API_VERSION,
        "input_file": str(args.rg_means_file),
        "baseline_file": str(args.baseline),
        "model": str(args.model),
        "model_description": str(spec["description"]),
        # --- backward-compatible contract for downstream lattice REMD scripts ---
        # load_fit_summary_json() in remd_uniform_chain_2_new.py requires top-level
        # model/params/Tref/Tscale and reads params by MODEL_REGISTRY name. These
        # are THERMODYNAMIC parameters only: rg_scale is a mapping constant, not a
        # b(T) parameter, and putting it here would corrupt the model vector.
        "param_names": list(model_param_names),
        "params": {
            name: float(value)
            for name, value in zip(model_param_names, model_params)
        },
        "target_units": target_units,
        "potential_kind": potential_kind,
        "quadratic_normalization": quadratic_normalization,
        "potential_normalization": potential_normalization,
        "potential_definition": potential_definition,
        "m_ref": int(m_ref),
        "fit_chain_length": (
            None if fit_chain_length is None else int(fit_chain_length)
        ),
        "kappa_bend": float(kappa_bend),
        "bending_enabled": bool(bending_enabled),
        "bend_definition": BEND_DEFINITION,
        "reduced_bias_by_temperature": b_T.tolist(),
        "linear_coefficient_by_temperature": b_T.tolist(),
        "quadratic_coefficient_by_temperature": q_T.tolist(),
        "coefficient_note": (
            "reduced_bias_by_temperature is kept as an alias of "
            "linear_coefficient_by_temperature. Neither coefficient's zero "
            "crossing is by itself the transition temperature; T_rg_max_slope is "
            "the primary finite-chain transition descriptor."
        ),
        "rg_summary": str(args.rg_summary),
        "rg_loss": str(args.rg_mean_loss),
        "rg_scale": float(rg_scale_effective),
        "rg_scale_initial": float(rg_scale_initial),
        "rg_scale_fitted": (
            float(rg_scale_fitted) if rg_scale_fitted is not None else None
        ),
        "rg_scale_effective": float(rg_scale_effective),
        "rg_scale_was_fitted": bool(args.fit_rg_scale),
        "rg_scale_input": float(rg_scale_initial),
        "rg_scale_bounds": (
            [float(args.rg_scale_min), float(args.rg_scale_max)]
            if args.fit_rg_scale else None
        ),
        "rg_huber_delta": float(args.rg_huber_delta) if args.rg_mean_loss == "huber" else None,
        "rg_range_floor": (
            float(args.rg_range_floor) if args.rg_mean_loss == "range_weighted" else None
        ),
        # Full optimization vector, INCLUDING rg_scale when it was fitted. Kept
        # separate from the top-level "params" contract above on purpose.
        "parameters": {n: float(v) for n, v in zip(param_names, params_all)},
        "model_parameters": {
            n: float(v) for n, v in zip(model_param_names, model_params)
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
            "T_rg_max_slope": (
                _finite_or_none(trans["T_rg_max_slope"])
                if trans["T_rg_max_slope"] is not None else None
            ),
            "collapse_detected": bool(trans["collapse_detected"]),
            "slope_tolerance": _finite_or_none(trans["slope_tolerance"]),
            "rg_max_negative_slope": _finite_or_none(trans["rg_max_negative_slope"]),
            "T_rg_half": _finite_or_none(trans["T_rg_half"]),
            "T_rg_half_crossings": [float(t) for t in trans["T_rg_half_crossings"]],
            "rg_half_value": _finite_or_none(trans["rg_half_value"]),
            "rg_half_defined": bool(trans["rg_half_defined"]),
            "rg_half_ambiguous": bool(trans["rg_half_ambiguous"]),
            "rg_half_tolerance": _finite_or_none(trans["rg_half_tolerance"]),
            "rg_curve_span": _finite_or_none(trans["rg_curve_span"]),
            "dense_grid_points": int(cfg["dense_grid_points"]),
            "collapse_note": (
                "T_rg_max_slope is null when max(-dRg/dT) does not exceed "
                "slope_tolerance, i.e. the fitted curve is flat or expands with "
                "temperature. No transition temperature is invented in that case."
            ),
            "half_note": (
                "T_rg_half is null unless the endpoint midpoint is crossed EXACTLY "
                "once. rg_half_defined=false with rg_half_ambiguous=true means the "
                "curve is nonmonotonic and crosses the midpoint several times (all "
                "listed in T_rg_half_crossings); with rg_half_ambiguous=false it "
                "means the curve is flat between its endpoints (span <= "
                "rg_half_tolerance), where every temperature is equally close to the "
                "midpoint and any single answer would be an artifact of the grid. "
                "Crossings are interpolated between grid points, not snapped to the "
                "nearest sampled temperature."
            ),
        },
        "support_diagnostics": support_diagnostics,
        "endpoint_limits": endpoint_limits_json,
        "global_outer_bound": global_outer_bound_json,
        "reachability_note": (
            "endpoint_limits are the two exact b -> +/-inf points of the trajectory; "
            "they are its extrema over all real b only when "
            "conditional_moment_monotonic is true. global_outer_bound is a rigorous "
            "NECESSARY bound over all real b: a target outside it is impossible for "
            "every real b, a target inside it is NOT thereby reachable. Both are "
            "evaluated at rg_scale_effective and constrain the scalar summary only."
        ),
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
                "Temperature at which the fitted Rg(T) curve crosses the midpoint "
                "between its values at the low and high ends of the observed "
                "temperature interval, located by linear interpolation between dense-"
                "grid points. Reported ONLY when exactly one such crossing exists; "
                "null for a flat curve (no defined midpoint level) or a nonmonotonic "
                "one (several crossings, see T_rg_half_crossings)."
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
            "rg_scale_initial": (
                "The --rg-scale value as supplied: the physically motivated input "
                "mapping. Never overwritten by the fit."
            ),
            "rg_scale_fitted": (
                "The scale the optimizer chose, or null when --fit-rg-scale was not "
                "given. It is NOT a thermodynamic parameter and never enters b(T)."
            ),
            "rg_scale_effective": (
                "The scale used for every fitted-model quantity in this run: the "
                "fitted scale when one exists, otherwise the initial scale."
            ),
            "params_vs_parameters": (
                "'params' carries thermodynamic parameters only and is the "
                "backward-compatible contract consumed by the lattice REMD scripts. "
                "'parameters' carries the full optimization vector, which includes "
                "rg_scale when it was fitted."
            ),
        },
        "units": {
            "target_input": target_units,
            "lattice_rg": "lattice bond units",
            "observed_rg": observed_unit_label,
            "mapping": "Rg_observed = rg_scale_effective * Rg_lattice",
            "note": (
                "Fields suffixed _nominal use rg_scale_initial; every other "
                "observed-unit field uses rg_scale_effective."
            ),
        },
        "warnings": warnings_list,
    }

    # Nominal and fitted feasibility are DIFFERENT questions and are reported as
    # such. For fixed-scale mode fitted_scale is null and nominal_scale is the
    # production result.
    summary_json["feasibility_diagnostics"] = {
        "nominal_scale": feasibility_nominal,
        "fitted_scale": feasibility_fitted,
        "note": (
            "nominal_scale answers: can the baseline reproduce the data using the "
            "physically motivated input scale? It is the primary scientific test. "
            "fitted_scale answers: can the fitted mapping reproduce the data after "
            "allowing the scale to move? It is a mapping-sensitivity diagnostic. In "
            "fixed-scale mode fitted_scale is null and nominal_scale is the "
            "production result."
        ),
    }
    summary_json["scientific_validity"] = scientific_validity
    if args.model == "heat_capacity":
        summary_json["T0"] = float(Tref)
    # The saturating-cooperative amplitude and saturation scale are also written
    # under their own names, so a reader never has to index into params by
    # position to recover the two parameters that define the nonlinearity.
    if potential_kind == "saturating_cooperative":
        summary_json["A0"] = float(model_params[2])
        summary_json["q_sat"] = float(model_params[3])

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
        "rg_scale_initial": rg_scale_initial,
        "rg_scale_fitted": rg_scale_fitted,
        "rg_scale_effective": rg_scale_effective,
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
                        rg_centers_lattice, b_T, all_metrics, q_T=q_T)

    # ---- final verdict ------------------------------------------------------
    # Structured, per-scale, and never reduced to VALID/NOT VALID. Convergence of
    # the optimizer is deliberately not part of this verdict.
    print("\n--- Scientific status ---")
    nv = scientific_validity["fixed_or_nominal_scale"]
    print(f"  nominal_scale_validity (PRIMARY scientific test, "
          f"rg_scale={rg_scale_initial:.6g}): {nv['status']}")
    print(f"    support_overlap={nv['support_overlap']}  "
          f"reachable_within_scan={nv['reachable_within_scan']}  "
          f"within_global_outer_bound={nv['within_global_outer_bound']}  "
          f"conditional_moment_monotonic={nv['conditional_moment_monotonic']}")
    fv = scientific_validity["fitted_scale"]
    if fv is None:
        print("  fitted_scale_validity: not applicable (fixed-scale mode; the "
              "nominal result above is the production result).")
    else:
        print(f"  fitted_scale_validity (mapping-sensitivity diagnostic, "
              f"rg_scale={rg_scale_effective:.6g}): {fv['status']}")
        print(f"    support_overlap={fv['support_overlap']}  "
              f"reachable_within_scan={fv['reachable_within_scan']}  "
              f"within_global_outer_bound={fv['within_global_outer_bound']}  "
              f"conditional_moment_monotonic={fv['conditional_moment_monotonic']}")
        print("    NOTE: a fitted-scale status describes the mapping's flexibility, "
              "not an independent confirmation of the physical scale.")
    print("  Only 'outside_global_outer_bound' asserts impossibility for all real "
          "b; scan-scoped statuses do not.")
    if feasibility_nominal is None:
        print("  UNVERIFIED: run --rg-feasibility-scan to establish whether the "
              "baseline can reach the observed Rg range.")
    print("  Optimizer convergence is NOT evidence of scientific validity.")
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
            # None when this replicate's curve shows no resolved collapse; the
            # stats below are computed only over replicates that detected one.
            "T_rg_max_slope": tm["T_rg_max_slope"],
            "collapse_detected": bool(tm["collapse_detected"]),
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

    n_collapse = int(sum(1 for r in rows if r.get("collapse_detected")))
    if n_collapse < n_ok:
        print(f"  NOTE: {n_ok - n_collapse}/{n_ok} replicate(s) show no resolved "
              f"collapse; T_rg_max_slope intervals use the {n_collapse} that do.")

    boot_csv = outdir / "rg_scalar_bootstrap.csv"
    header = ["replicate"] + list(param_names) + [
        "train_objective", "validation_objective", "all_objective",
        "train_rmse", "validation_rmse", "all_rmse",
        "T_bias_zero", "T_rg_max_slope", "collapse_detected",
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
            "n_replicates_with_collapse": n_collapse,
            "collapse_note": (
                "T_rg_max_slope statistics are computed only over replicates whose "
                "fitted curve showed a resolved collapse; replicates without one "
                "contribute no transition temperature rather than a spurious value."
            ),
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
        t_slope_str = (
            f"{tm['T_rg_max_slope']:.5g}" if tm["T_rg_max_slope"] is not None
            else "None (no collapse)"
        )
        print(f"  {sp['name']:24s} : train_rmse={tr_m['rmse']:.5g}  "
              f"val_rmse={va_m['rmse']:.5g}  T_rg_max_slope={t_slope_str}")

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
    *,
    q_T: Optional[np.ndarray] = None,
) -> None:
    """Scalar-mode plots: fit, residuals, b(T), and predicted P(Rg|T) diagnostics.

    ``q_T`` adds the contact-quadratic coefficient panel; it is omitted for the
    linear models, whose kappa(T) is identically zero.
    """
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

    # Every plotted mapping states which scale it uses. For a free-scale
    # production prediction the fitted scale appears in the title.
    rg_scale_effective: float = float(ctx["rg_scale_effective"])
    rg_scale_initial: float = float(ctx["rg_scale_initial"])
    if ctx["cfg"]["fit_rg_scale"]:
        scale_note = (
            f"mapping: FITTED scale rg_scale_effective = {rg_scale_effective:.6g} "
            f"(initial input scale was {rg_scale_initial:.6g})"
        )
    else:
        scale_note = f"mapping: FIXED scale rg_scale = {rg_scale_effective:.6g}"

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
    # No vertical transition line when no collapse was resolved: drawing one would
    # assert a transition the curve does not show.
    if trans["T_rg_max_slope"] is not None:
        ax.axvline(trans["T_rg_max_slope"], color="tab:green", lw=1.2, ls="--",
                   label=f"T_rg_max_slope = {trans['T_rg_max_slope']:.4g} K")
    else:
        ax.plot([], [], " ", label="no resolved collapse (T_rg_max_slope = None)")
    ax.set_xlabel("T (K)")
    ax.set_ylabel(f"Rg ({units} units)")
    rmse = all_metrics["rmse"]
    ax.set_title(
        f"Scalar Rg(T) fit — model {args.model}, summary {args.rg_summary}, "
        f"loss {args.rg_mean_loss}\n"
        f"{scale_note}\n"
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

    # --- 3b. kappa(T), the contact-quadratic coefficient ---
    # Drawn only when the model has one -- kappa(T) is the coefficient of
    # m^2/(2N), which only the contact_quadratic family contains. Its zero
    # crossing is deliberately NOT marked as a transition temperature:
    # T_rg_max_slope is that descriptor.
    if q_T is not None and str(ctx["cfg"]["spec"]["potential_kind"]) == "contact_quadratic":
        norm = ctx["cfg"]["spec"]["quadratic_normalization"]
        fig, ax = plt.subplots(figsize=(7, 4))
        q_dense = np.array(
            [ctx["cfg"]["q_fn"](model_params, float(T)) for T in dense]
        )
        ax.plot(dense, q_dense, "k-", lw=1.8)
        ax.axhline(0.0, color="gray", lw=0.8, ls="--")
        ax.plot(temps, q_T, "o", ms=4, color="tab:blue", label="data temperatures")
        ax.set_xlabel("T (K)")
        ax.set_ylabel(f"kappa(T)   [coefficient of {norm}]")
        ax.set_title(
            f"Contact-quadratic coefficient  [{args.model}, "
            f"N={ctx['cfg']['n_beads']}]\n"
            "kappa(T)=0 is not by itself the transition temperature"
        )
        ax.legend(fontsize=8)
        fig.tight_layout()
        p = outdir / "rg_scalar_qT.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        print(f"Saved: {p}")
        figs.append(fig)

    # --- 3c. Contact potential u(m,T) vs m (saturating-cooperative model) ---
    # The saturating model has no single coefficient that summarizes it, so the
    # potential itself is plotted against the linear reference b(T)*m. The gap
    # between the two IS the cooperative term; that it stops widening is the
    # saturation. Drawn over the baseline's own contact axis, since that is the
    # m range the reweighting actually sees in this mode.
    if str(ctx["cfg"]["spec"]["potential_kind"]) == "saturating_cooperative":
        c_edges_plot = np.asarray(ctx["cfg"]["c_edges"], dtype=float)
        m_axis = 0.5 * (c_edges_plot[:-1] + c_edges_plot[1:])
        u_plot_fn = ctx["cfg"]["u_fn"]
        pick_u = sorted({0, temps.size // 2, temps.size - 1})
        pick_u_labels = {0: "lowest T", temps.size - 1: "highest T"}
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for k, i in enumerate(pick_u):
            color = plt.cm.viridis(k / max(len(pick_u) - 1, 1))
            T_i = float(temps[i])
            lab = pick_u_labels.get(i, "mid T")
            ax.plot(
                m_axis, u_plot_fn(model_params, T_i, m_axis), "-", lw=1.8,
                color=color, label=f"u(m,T)   T = {T_i:.4g} K ({lab})",
            )
            ax.plot(
                m_axis, b_T[i] * m_axis, "--", lw=1.2, color=color, alpha=0.75,
                label=f"b(T)*m    T = {T_i:.4g} K",
            )
        ax.axhline(0.0, color="gray", lw=0.8, ls=":")
        ax.set_xlabel("m (contacts)")
        ax.set_ylabel("u(m,T)  (reduced units)")
        ax.set_title(
            f"Contact potential vs linear reference  [{args.model}, "
            f"N={ctx['cfg']['n_beads']}]\n"
            f"{ctx['cfg']['spec']['potential_definition']}",
            fontsize=8,
        )
        ax.legend(fontsize=7)
        fig.tight_layout()
        p = outdir / "rg_scalar_contact_potential.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        print(f"Saved: {p}")
        figs.append(fig)

    # --- 4. predicted P(Rg|T) at a deterministic set of temperatures ---
    picks = {0, temps.size - 1}
    if trans["T_rg_max_slope"] is not None:
        picks.add(int(np.argmin(np.abs(temps - trans["T_rg_max_slope"]))))
    else:
        picks.add(int(temps.size // 2))
    pick = sorted(picks)
    # One scale for the axis: the effective one. The nominal scale never appears
    # on a fitted-model plot.
    rg_axis = (
        rg_centers_lattice if units == "lattice"
        else rg_scale_effective * rg_centers_lattice
    )
    fig, ax = plt.subplots(figsize=(7, 4.5))
    labels = {
        0: "lowest T",
        temps.size - 1: "highest T",
    }
    mid_label = "near T_rg_max_slope" if trans["T_rg_max_slope"] is not None else "mid T"
    for i in pick:
        lab = labels.get(i, mid_label)
        ax.plot(rg_axis, rg_mod_mass[i], "-", lw=1.6,
                label=f"T = {temps[i]:.4g} K ({lab})")
    ax.set_xlabel(f"Rg ({units} units)")
    ax.set_ylabel("P(Rg | T)")
    ax.set_title(
        "PREDICTED Rg distributions from the fitted model (diagnostic only)\n"
        f"{scale_note}\n"
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

def build_arg_parser() -> argparse.ArgumentParser:
    """Construct the CLI parser.

    Split out of main() so the tests can build a fully-populated Namespace from
    the real defaults instead of hand-listing them, which would silently drift
    from the CLI. No argument name, default, or meaning is changed here.
    """
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
    ap.add_argument(
        "--N", dest="chain_length", type=int, default=None,
        help=(
            "Chain length (number of beads) used to normalize the "
            "contact-quadratic term as kappa*m^2/(2N). Fallback only: the "
            "baseline's n_beads (or N) wins and a mismatch is an error. Required "
            "by hs_m2_const/hs_m2_hs when the baseline predates that metadata."
        ),
    )
    ap.add_argument(
        "--kappa-bend", dest="kappa_bend", type=float, default=None,
        help=(
            "Consistency check only: assert the baseline was generated with this "
            "bending penalty. The penalty is already baked into the baseline "
            "distribution and is never fitted here. Default: take whatever the "
            "baseline records (legacy baselines count as 0)."
        ),
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
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()

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

    # Bending penalty: read from the baseline, optionally cross-checked against
    # the CLI. Never fitted, never re-applied during reweighting.
    kappa_bend = resolve_kappa_bend(
        read_baseline_kappa_bend(b_data), args.kappa_bend, str(args.baseline)
    )
    bending_enabled = bool(kappa_bend != 0.0)

    # Chain length: needed only by the nonlinear models, whose curvature term
    # (m^2/(2N)) or contact fraction (q = m/N) is normalized by it.
    fit_chain_length = resolve_chain_length(
        read_baseline_chain_length(b_data), args.chain_length,
        model_name=args.model, baseline_path=str(args.baseline),
    )
    potential_kind = str(MODEL_REGISTRY[args.model]["potential_kind"])
    quadratic_normalization = MODEL_REGISTRY[args.model]["quadratic_normalization"]
    potential_normalization = MODEL_REGISTRY[args.model]["potential_normalization"]
    potential_definition = str(MODEL_REGISTRY[args.model]["potential_definition"])
    m_ref = int(MODEL_REGISTRY[args.model]["m_ref"])

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
    print(f"  kappa_bend:  {kappa_bend:g} "
          f"({'bending enabled' if bending_enabled else 'no bending penalty'}; "
          f"{BEND_DEFINITION})")
    print(f"  chain length: "
          + ("not recorded" if fit_chain_length is None else f"{fit_chain_length}")
          + (f"  (used for {potential_normalization})"
             if potential_normalization else "  (not needed by this model)"))
    print(f"  potential:   [{potential_kind}]  {potential_definition}")
    print(f"               m_ref = {m_ref}")

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
    q_fn = make_q_fn(args.model, Tref, Tscale)
    # u_fn is what every reweighting path uses. For linear models it evaluates to
    # exactly b(T)*m, so their numerical results are unchanged.
    u_fn = make_contact_u_fn(args.model, Tref, Tscale, n_beads=fit_chain_length)
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
        args.fit_rg, train_temps, m_centers, p_obs_ct_train, p0_mass, u_fn, loss_fn,
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
            p0_mass, m_centers, float(T), params_fit, u_fn
        )

    # -----------------------------------------------------------------------
    # Post-fit contact loss breakdown
    # -----------------------------------------------------------------------
    train_loss = objective(
        params_fit, train_temps, m_centers, p_obs_ct_train, p0_mass, u_fn, loss_fn
    )
    val_loss = (
        objective(
            params_fit, temps[val_idx], m_centers, p_obs_mass[val_idx],
            p0_mass, u_fn, loss_fn,
        )
        if has_val else float("nan")
    )
    all_loss = objective(
        params_fit, temps, m_centers, p_obs_mass, p0_mass, u_fn, loss_fn
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
            u_fn=u_fn,
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
    # Per-temperature contact-potential coefficients
    # -----------------------------------------------------------------------
    # b(T) is the LINEAR coefficient and kappa(T) the coefficient of
    # m^2/(2N). Neither zero-crossing is by itself the transition temperature.
    linear_coefficient_by_temperature = np.array(
        [b_fn(params_fit, float(T)) for T in temps], dtype=float
    )
    quadratic_coefficient_by_temperature = np.array(
        [q_fn(params_fit, float(T)) for T in temps], dtype=float
    )

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
        kappa_bend=float(kappa_bend),
        bending_enabled=bool(bending_enabled),
        bend_definition=BEND_DEFINITION,
        potential_kind=potential_kind,
        quadratic_normalization=(
            "" if quadratic_normalization is None else str(quadratic_normalization)
        ),
        potential_normalization=(
            "" if potential_normalization is None else str(potential_normalization)
        ),
        potential_definition=potential_definition,
        m_ref=int(m_ref),
        fit_chain_length=(-1 if fit_chain_length is None else int(fit_chain_length)),
        # reduced_bias_by_temperature is retained as the linear-coefficient
        # compatibility field; the two explicit names below are the ones to read.
        reduced_bias_by_temperature=linear_coefficient_by_temperature,
        linear_coefficient_by_temperature=linear_coefficient_by_temperature,
        quadratic_coefficient_by_temperature=quadratic_coefficient_by_temperature,
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
    # The saturating-cooperative amplitude and saturation scale are also written
    # under their own names, so a reader never has to index into params by
    # position to recover the two parameters that define the nonlinearity.
    if potential_kind == "saturating_cooperative":
        save_kwargs["A0"] = float(params_fit[2])
        save_kwargs["q_sat"] = float(params_fit[3])

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
        "kappa_bend": float(kappa_bend),
        "bending_enabled": bool(bending_enabled),
        "bend_definition": BEND_DEFINITION,
        "potential_kind": potential_kind,
        "quadratic_normalization": quadratic_normalization,
        "potential_normalization": potential_normalization,
        "potential_definition": potential_definition,
        "m_ref": int(m_ref),
        "fit_chain_length": (
            None if fit_chain_length is None else int(fit_chain_length)
        ),
        "reduced_bias_by_temperature": linear_coefficient_by_temperature.tolist(),
        "linear_coefficient_by_temperature": linear_coefficient_by_temperature.tolist(),
        "quadratic_coefficient_by_temperature": (
            quadratic_coefficient_by_temperature.tolist()
        ),
        "coefficient_note": (
            "reduced_bias_by_temperature is kept as an alias of "
            "linear_coefficient_by_temperature. Neither coefficient's zero "
            "crossing is by itself the transition temperature. "
            "quadratic_coefficient_by_temperature is the coefficient of "
            "m^2/(2N) and is identically zero for every model whose "
            "potential_kind is not contact_quadratic; potential_definition is "
            "the authoritative statement of what was fitted."
        ),
        "rg_train_loss": None if not has_rg_scoring else float(rg_train_loss),
        "rg_val_loss": None if (not has_rg_scoring or not has_val) else float(rg_val_loss),
        "rg_all_loss": None if not has_rg_scoring else float(rg_all_loss),
    }
    # For heat_capacity, persist the thermodynamic reference temperature as T0
    # (stored in Tref by this fitter). Keep Tref for backward compatibility.
    if args.model == "heat_capacity":
        metadata["T0"] = float(Tref)

    # The saturating-cooperative amplitude and saturation scale are also written
    # under their own names, so a reader never has to index into params by
    # position to recover the two parameters that define the nonlinearity.
    if potential_kind == "saturating_cooperative":
        metadata["A0"] = float(params_fit[2])
        metadata["q_sat"] = float(params_fit[3])

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
        "p0_mass": p0_mass, "b_fn": b_fn, "u_fn": u_fn, "loss_fn": loss_fn, "spec": spec,
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
            "p0_mass": p0_mass, "b_fn": b_fn, "u_fn": u_fn, "loss_fn": loss_fn,
            "spec": spec,
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
    b_vals = linear_coefficient_by_temperature
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

    # --- 3b. Quadratic coefficient kappa(T) vs T (contact-quadratic models) ---
    # Only the contact_quadratic family HAS a kappa(T): it is the coefficient of
    # m^2/(2N). Other nonlinear kinds have none, and plotting their identically
    # zero kappa would imply a curvature term they do not contain.
    if potential_kind == "contact_quadratic":
        fig3q, ax3q = plt.subplots(figsize=(6, 4))
        ax3q.plot(temps, quadratic_coefficient_by_temperature, "k-", lw=1.8)
        ax3q.axhline(0.0, color="gray", lw=0.8, ls="--")
        if has_val:
            ax3q.scatter(
                temps[val_idx], quadratic_coefficient_by_temperature[val_idx],
                color="red", zorder=3, s=25, label="held-out",
            )
            ax3q.legend(fontsize=8)
        ax3q.set_xlabel("T")
        ax3q.set_ylabel(f"kappa(T)   [coefficient of {quadratic_normalization}]")
        ax3q.set_title(
            f"Contact-quadratic coefficient  [{args.model}, N={fit_chain_length}]\n"
            "kappa(T)=0 is not by itself the transition temperature"
        )
        fig3q.tight_layout()
        p3q = plot_dir / "quadratic_coefficient_vs_T.png"
        fig3q.savefig(p3q, dpi=150, bbox_inches="tight")
        print(f"Saved: {p3q}")
        open_figs.append(fig3q)

    # --- 3c. Contact potential u(m,T) vs m (saturating-cooperative model) ---
    # The saturating model has no single coefficient that summarizes it, so the
    # potential itself is plotted against the linear reference b(T)*m. The gap
    # between the two IS the cooperative term; that it stops widening is the
    # saturation. Temperatures are the deterministic low/middle/high triple, as
    # a set so a 1- or 2-temperature fit does not draw duplicates.
    if potential_kind == "saturating_cooperative":
        u_plot_fn = make_contact_u_fn(
            args.model, Tref, Tscale, n_beads=fit_chain_length
        )
        pick = sorted({0, len(temps) // 2, len(temps) - 1})
        pick_labels = {
            0: "lowest T", len(temps) - 1: "highest T",
        }
        fig3s, ax3s = plt.subplots(figsize=(6.5, 4.5))
        for k, i in enumerate(pick):
            color = plt.cm.viridis(k / max(len(pick) - 1, 1))
            T_i = float(temps[i])
            lab = pick_labels.get(i, "mid T")
            ax3s.plot(
                m_centers, u_plot_fn(params_fit, T_i, m_centers), "-", lw=1.8,
                color=color, label=f"u(m,T)   T = {T_i:.4g} ({lab})",
            )
            ax3s.plot(
                m_centers, linear_coefficient_by_temperature[i] * m_centers,
                "--", lw=1.2, color=color, alpha=0.75,
                label=f"b(T)*m    T = {T_i:.4g}",
            )
        ax3s.axhline(0.0, color="gray", lw=0.8, ls=":")
        ax3s.set_xlabel("m (integer contacts)")
        ax3s.set_ylabel("u(m,T)  (reduced units)")
        ax3s.set_title(
            f"Contact potential vs linear reference  [{args.model}, "
            f"N={fit_chain_length}]\n{potential_definition}",
            fontsize=8,
        )
        ax3s.legend(fontsize=7)
        fig3s.tight_layout()
        p3s = plot_dir / "contact_potential_vs_m.png"
        fig3s.savefig(p3s, dpi=150, bbox_inches="tight")
        print(f"Saved: {p3s}")
        open_figs.append(fig3s)

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
