#!/usr/bin/env python3
"""Fit the thermodynamic contact Hamiltonian to histogram data.

We assume a model of the form

    E(m;T) = m (h - T s)

where m is the (non-bonded) contact count (or HH contacts for HP), and
Boltzmann weight is

    exp(-beta E) = exp(-(h/T - s) m).

To predict P(m|T) you need the *athermal* density of states g(m), i.e.
P0(m) for a non-interacting (athermal) SAW. Then

    P_model(m|T;h,s) \\propto P0(m) * exp(-(h/T - s) m).

This script fits (h,s) by minimizing the summed KL divergence between
observed contact distributions and the model distributions.

If you only have P(m|T) at multiple temperatures but *no* athermal P0(m),
then (h,s) are not uniquely identifiable: only the combination kappa(T)
up to an additive constant is fixed. In that case, this script can still
estimate an effective h (see --fit_h_only).

Inputs
------
1) REMD/scan distributions: an .npz containing at least
   - temps: (nT,)
   - ct_centers: (nC,) bin centers for contacts
   - ct_hists: (nT,nC) contact pdf values (so sum(pdf)*dc \approx 1)
   Optionally:
   - rg_centers, rg_hists (used only for plotting, not for fitting unless
     you provide a joint baseline)

2) Baseline athermal contacts distribution: an .npz with either
   - c_vals, c_prob  (probability mass over integer contact counts)
   OR
   - ct_centers, ct_hists for a single condition that is truly athermal.

Example
-------
# 1) generate an athermal baseline (long run recommended)
python single_uniform_chain2_athermal_dists.py --N 30 --steps 2000000 --dist_dir dists --T 1
# take the produced DIST_FILE path as baseline

# 2) fit h and s
python fit_hs_to_remd_30mer.py \
  --remd remd_distributions_30mer.npz \
  --baseline dists/single_uniform_chain2_athermal_dists_N30_T1_seed42.npz \
  --out fit_hs_30mer.npz
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

try:
    from scipy.optimize import minimize
except Exception as e:
    minimize = None


def centers_to_edges(centers: np.ndarray) -> np.ndarray:
    """Assume uniform spacing, return bin edges."""
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


def build_baseline_mass(ct_centers: np.ndarray, baseline_npz: str) -> np.ndarray:
    """Map a baseline distribution onto the REMD contact bins."""
    b = np.load(baseline_npz)

    # Case A: discrete contact values
    if "c_vals" in b.files and "c_prob" in b.files:
        c_vals = np.asarray(b["c_vals"], dtype=float)
        c_prob = np.asarray(b["c_prob"], dtype=float)
        c_prob = np.clip(c_prob, 0.0, None)
        if c_prob.sum() <= 0:
            raise ValueError("baseline c_prob sums to 0")
        c_prob = c_prob / c_prob.sum()  # mass over c_vals

        edges = centers_to_edges(ct_centers)
        idx = np.digitize(c_vals, edges) - 1
        p0 = np.zeros(ct_centers.size, dtype=float)
        for k, pk in zip(idx, c_prob):
            if 0 <= k < p0.size:
                p0[k] += pk
        if p0.sum() <= 0:
            raise ValueError("Baseline mass ended up empty after binning. Check your bin ranges.")
        return p0 / p0.sum()

    # Case B: baseline provided as a pdf already on centers
    if "ct_centers" in b.files and "ct_hists" in b.files:
        ccent = np.asarray(b["ct_centers"], dtype=float)
        ch = np.asarray(b["ct_hists"], dtype=float)

        # if (nT,nC) choose the first row
        if ch.ndim == 2:
            ch = ch[0]

        # If the centers match, use directly
        if ccent.shape == ct_centers.shape and np.allclose(ccent, ct_centers):
            p0, _ = pdf_to_mass(ch, ct_centers)
            return p0

        # Otherwise interpolate pdf and then convert to mass
        ch_interp = np.interp(ct_centers, ccent, ch, left=0.0, right=0.0)
        p0, _ = pdf_to_mass(ch_interp, ct_centers)
        return p0

    raise ValueError(
        "Unrecognized baseline format. Expected (c_vals,c_prob) or (ct_centers,ct_hists)."
    )


def model_contact_mass(p0_mass: np.ndarray,
                       c_centers: np.ndarray,
                       T: float,
                       h: float,
                       s: float) -> np.ndarray:
    """Return model probability mass over the contact bins."""
    # exp(-(h/T - s) m)
    k = -(h / T - s)
    w = p0_mass * np.exp(k * c_centers)
    Z = w.sum()
    if not np.isfinite(Z) or Z <= 0:
        # numerical under/overflow: stabilize by subtracting max exponent
        x = k * c_centers
        x = x - np.max(x)
        w = p0_mass * np.exp(x)
        Z = w.sum()
    if Z <= 0:
        return np.full_like(p0_mass, 1.0 / p0_mass.size)
    return w / Z


def kl_div(p_obs: np.ndarray, p_mod: np.ndarray, eps: float = 1e-12) -> float:
    p_obs = np.asarray(p_obs, dtype=float)
    p_mod = np.asarray(p_mod, dtype=float)
    p_obs = np.clip(p_obs, eps, 1.0)
    p_mod = np.clip(p_mod, eps, 1.0)
    p_obs = p_obs / p_obs.sum()
    p_mod = p_mod / p_mod.sum()
    return float(np.sum(p_obs * (np.log(p_obs) - np.log(p_mod))))


def objective_hs(params: np.ndarray,
                 temps: np.ndarray,
                 c_centers: np.ndarray,
                 p_obs_mass: np.ndarray,
                 p0_mass: np.ndarray) -> float:
    h, s = float(params[0]), float(params[1])
    total = 0.0
    for i, T in enumerate(temps):
        p_mod = model_contact_mass(p0_mass, c_centers, float(T), h, s)
        total += kl_div(p_obs_mass[i], p_mod)
    return total


def fit_h_only_from_ratios(temps: np.ndarray,
                           c_centers: np.ndarray,
                           p_obs_mass: np.ndarray,
                           ref_idx: int = 0,
                           min_p: float = 1e-6) -> float:
    """Estimate h (only) from log-ratio slopes.

    Using
      log(P_i/P_ref) = h(1/T_ref - 1/T_i) * m + const
    so slope_i = h(1/T_ref - 1/T_i).

    Returns h_hat.
    """
    pref = p_obs_mass[ref_idx]
    Tref = float(temps[ref_idx])

    ms = []
    slopes = []
    weights = []

    for i, T in enumerate(temps):
        if i == ref_idx:
            continue
        p = p_obs_mass[i]
        mask = (p > min_p) & (pref > min_p)
        if mask.sum() < 5:
            continue
        m = c_centers[mask]
        y = np.log(p[mask]) - np.log(pref[mask])

        # weighted linear regression y = a + b m
        w = p[mask]
        W = w.sum()
        xm = (w * m).sum() / W
        ym = (w * y).sum() / W
        cov = (w * (m - xm) * (y - ym)).sum() / W
        var = (w * (m - xm) ** 2).sum() / W
        if var <= 0:
            continue
        b = cov / var

        denom = (1.0 / Tref - 1.0 / float(T))
        if abs(denom) < 1e-12:
            continue
        h_i = b / denom
        ms.append(float(T))
        slopes.append(float(h_i))
        weights.append(float(W))

    if not slopes:
        return float("nan")

    slopes = np.asarray(slopes)
    weights = np.asarray(weights)
    return float(np.sum(weights * slopes) / np.sum(weights))


def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--remd", type=str, default="remd_distributions_30mer.npz",
                    help="NPZ with temps, ct_centers, ct_hists (pdf)")
    ap.add_argument("--baseline", type=str, default= 'single_uniform_chain2_athermal_dists_N30_T1_seed42.npz',
                    help="NPZ baseline athermal contacts (c_vals,c_prob) or (ct_centers,ct_hists)")
    ap.add_argument("--contact_offset", type=float, default=29,
                    help="Constant to subtract from ct_centers in the REMD file (e.g., N-1 if contacts include bonded neighbors).")
    ap.add_argument("--fit_h_only", action="store_true",
                    help="Fit only h from log-ratio slopes (does not require baseline)")
    ap.add_argument("--ref_idx", type=int, default=0,
                    help="Reference temperature index for --fit_h_only")
    ap.add_argument("--out", type=str, default="fit_hs_30mer.npz",
                    help="Output NPZ path")
    ap.add_argument("--no_plots", action="store_true")

    # Optimization settings
    ap.add_argument("--h0", type=float, default=900.0)
    ap.add_argument("--s0", type=float, default=2)
    ap.add_argument("--h_bounds", type=float, nargs=2, default=[-2000.0, 2000.0])
    ap.add_argument("--s_bounds", type=float, nargs=2, default=[-5.0, 5.0])
    ap.add_argument("--n_restarts", type=int, default=8)
    ap.add_argument("--seed", type=int, default=123)

    args = ap.parse_args()

    d = np.load(args.remd)
    temps = np.asarray(d["temps"], dtype=float)
    c_centers = np.asarray(d["ct_centers"], dtype=float) - float(args.contact_offset)
    ct_pdf = np.asarray(d["ct_hists"], dtype=float)

    # normalize per-T pdf -> mass
    p_obs_mass = np.zeros_like(ct_pdf)
    for i in range(ct_pdf.shape[0]):
        p_obs_mass[i], _ = pdf_to_mass(ct_pdf[i], c_centers)

    if args.fit_h_only or args.baseline is None:
        h_hat = fit_h_only_from_ratios(temps, c_centers, p_obs_mass, ref_idx=args.ref_idx)
        print(f"h_hat (ratio fit, s not identifiable without baseline) = {h_hat:.6g}")
        np.savez_compressed(args.out, h=float(h_hat), s=float('nan'),
                            temps=temps, ct_centers=c_centers, ct_mass=p_obs_mass)
        if not args.no_plots:
            plt.figure(figsize=(6,4))
            plt.plot(temps, [np.sum(c_centers * p_obs_mass[i]) for i in range(len(temps))], 'o-')
            plt.xlabel('T (K)'); plt.ylabel('mean contacts')
            plt.title('Observed mean contacts')
            plt.tight_layout(); plt.show()
        return

    if minimize is None:
        raise RuntimeError("scipy is required for fitting. Install scipy or run with --fit_h_only.")

    p0_mass = build_baseline_mass(c_centers, args.baseline)

    rng = np.random.default_rng(args.seed)

    best = None
    best_val = float("inf")

    bounds = [tuple(args.h_bounds), tuple(args.s_bounds)]

    x0s = [np.array([args.h0, args.s0], dtype=float)]
    for _ in range(max(0, args.n_restarts - 1)):
        h = rng.uniform(args.h_bounds[0], args.h_bounds[1])
        s = rng.uniform(args.s_bounds[0], args.s_bounds[1])
        x0s.append(np.array([h, s], dtype=float))

    for x0 in x0s:
        res = minimize(
            objective_hs, x0,
            args=(temps, c_centers, p_obs_mass, p0_mass),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 500}
        )
        if res.fun < best_val:
            best_val = float(res.fun)
            best = res

    if best is None:
        raise RuntimeError("Fitting failed")

    h_fit, s_fit = float(best.x[0]), float(best.x[1])
    print(f"Best-fit: h = {h_fit:.6g}, s = {s_fit:.6g}")
    print(f"Objective (sum KL) = {best_val:.6g}")

    # model predictions
    p_mod_mass = np.zeros_like(p_obs_mass)
    for i, T in enumerate(temps):
        p_mod_mass[i] = model_contact_mass(p0_mass, c_centers, float(T), h_fit, s_fit)

    out_path = Path(args.out)
    np.savez_compressed(
        out_path,
        h=h_fit, s=s_fit,
        temps=temps,
        ct_centers=c_centers,
        p_obs_mass=p_obs_mass,
        p_mod_mass=p_mod_mass,
        p0_mass=p0_mass,
        baseline=str(args.baseline),
    )
    print(f"Saved: {out_path}")

    if args.no_plots:
        return

    # quick diagnostics
    mean_obs = (c_centers[None, :] * p_obs_mass).sum(axis=1)
    mean_mod = (c_centers[None, :] * p_mod_mass).sum(axis=1)

    plt.figure(figsize=(6,4))
    plt.plot(temps, mean_obs, 'o', label='obs')
    plt.plot(temps, mean_mod, '-', label='model')
    plt.xlabel('T (K)')
    plt.ylabel('mean contacts')
    plt.legend()
    plt.tight_layout()

    # overlay a handful of distributions
    fig, ax = plt.subplots(1, 1, figsize=(7,4))
    idxs = np.linspace(0, len(temps)-1, 8).astype(int)
    for i in idxs:
        ax.plot(c_centers, p_obs_mass[i], alpha=0.35)
        ax.plot(c_centers, p_mod_mass[i], alpha=0.9, lw=1.5)
    ax.set_xlabel('contacts'); ax.set_ylabel('P(contacts)')
    ax.set_title('Obs (faint) vs model (bold)')
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
