#!/usr/bin/env python3
"""Fit the thermodynamic contact Hamiltonian to histogram data and predict P(Rg|T).

Model
-----
We assume a contact-only thermodynamic model

    E(m;T) = m (h - T s)

so the Boltzmann weight is

    exp(-beta E) = exp(-(h/T - s) m),

with k_B = 1. Here m is the *non-bonded* contact count (or HH contacts in HP).

Given an athermal baseline that approximates the density of states over contacts,
P0(m) ~ g(m)/sum_m g(m), we predict

    P_model(m|T;h,s) ∝ P0(m) * exp(-(h/T - s) m).

NEW (joint baseline):
---------------------
If the athermal baseline also contains a joint histogram P0(m, Rg), saved as

    c_edges, rg_edges, crg_prob

(where crg_prob is a probability MASS over bins that sums to 1),
then we can also predict the Rg distribution by reweighting:

    P_model(Rg|T) ∝ Σ_m P0(m, Rg) * exp(-(h/T - s) m).

This lets you compare predicted P(Rg|T) to simulated P(Rg|T).

Inputs
------
1) REMD/scan distributions: an .npz containing at least
   - temps: (nT,)
   - ct_centers: (nC,) bin centers for contacts
   - ct_hists: (nT,nC) contact pdf values (sum(pdf)*dc ≈ 1)
   Optional:
   - rg_centers: (nR,) bin centers for Rg
   - rg_hists: (nT,nR) Rg pdf values

2) Baseline athermal distribution: an .npz with
   - (c_vals,c_prob) for P0(m)   [required for (h,s) fit]
   Optional (for P(Rg|T) prediction):
   - c_edges, rg_edges, crg_prob  for joint P0(m,Rg)

Notes on contact_offset
-----------------------
If your REMD "contact" definition includes the always-present bonded neighbours,
you may need to subtract (N-1) so that m matches the non-bonded definition used
by your lattice SAW code.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

try:
    from scipy.optimize import minimize
except Exception:
    minimize = None


# ------------------------- utilities -------------------------
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
    if centers.size < 2:
        # single bin
        mass = np.asarray(pdf, dtype=float).copy()
        s = mass.sum()
        if s > 0:
            mass /= s
        return mass, 1.0
    d = float(np.mean(np.diff(centers)))
    mass = pdf * d
    s = np.sum(mass)
    if s > 0:
        mass = mass / s
    return mass, d


def mass_to_pdf(mass: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Convert probability mass per bin to pdf per bin (piecewise constant)."""
    mass = np.asarray(mass, dtype=float)
    edges = np.asarray(edges, dtype=float)
    widths = np.diff(edges)
    widths = np.where(widths <= 0, 1.0, widths)
    pdf = mass / widths
    return pdf


# ------------------------- baseline builders -------------------------
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
        if ch.ndim == 2:
            ch = ch[0]

        if ccent.shape == ct_centers.shape and np.allclose(ccent, ct_centers):
            p0, _ = pdf_to_mass(ch, ct_centers)
            return p0

        ch_interp = np.interp(ct_centers, ccent, ch, left=0.0, right=0.0)
        p0, _ = pdf_to_mass(ch_interp, ct_centers)
        return p0

    raise ValueError(
        "Unrecognized baseline format. Expected (c_vals,c_prob) or (ct_centers,ct_hists)."
    )


def build_baseline_joint_mass(ct_centers: np.ndarray,
                              baseline_npz: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Return baseline joint probability mass P0(m_bin, Rg_bin) mapped to REMD contact bins.

    Returns (joint_mass, rg_edges). joint_mass has shape (nC, nR) and sums to 1.
    If the baseline file does not contain joint data, returns (None, None).
    """
    b = np.load(baseline_npz)
    need = {"c_edges", "rg_edges", "crg_prob"}
    if not need.issubset(set(b.files)):
        return None, None

    c_edges0 = np.asarray(b["c_edges"], dtype=float)
    rg_edges0 = np.asarray(b["rg_edges"], dtype=float)
    joint0 = np.asarray(b["crg_prob"], dtype=float)

    if joint0.ndim != 2:
        raise ValueError("baseline crg_prob must be 2D (nC0, nR0)")
    if c_edges0.size != joint0.shape[0] + 1:
        raise ValueError("baseline c_edges shape does not match crg_prob")
    if rg_edges0.size != joint0.shape[1] + 1:
        raise ValueError("baseline rg_edges shape does not match crg_prob")

    # Ensure normalized mass
    s0 = float(np.sum(joint0))
    if s0 <= 0 or not np.isfinite(s0):
        return None, None
    joint0 = joint0 / s0

    # Map contact bins from baseline into REMD bins
    edges_remd = centers_to_edges(ct_centers)
    c_cent0 = 0.5 * (c_edges0[:-1] + c_edges0[1:])

    idx = np.digitize(c_cent0, edges_remd) - 1  # baseline contact-bin -> remd-bin
    joint = np.zeros((ct_centers.size, joint0.shape[1]), dtype=float)
    for i0, k in enumerate(idx):
        if 0 <= k < joint.shape[0]:
            joint[k, :] += joint0[i0, :]

    sj = float(np.sum(joint))
    if sj <= 0:
        return None, None
    joint /= sj
    return joint, rg_edges0


# ------------------------- model -------------------------
def model_contact_mass(p0_mass: np.ndarray,
                       c_centers: np.ndarray,
                       T: float,
                       h: float,
                       s: float) -> np.ndarray:
    """Return model probability mass over the contact bins."""
    k = -(h / T - s)
    x = k * c_centers
    # stabilize exponentials
    x = x - np.max(x)
    w = p0_mass * np.exp(x)
    Z = w.sum()
    if Z <= 0 or not np.isfinite(Z):
        return np.full_like(p0_mass, 1.0 / p0_mass.size)
    return w / Z


def model_rg_mass_from_joint(joint0_mass: np.ndarray,
                             rg_edges0: np.ndarray,
                             c_centers: np.ndarray,
                             T: float,
                             h: float,
                             s: float) -> np.ndarray:
    """Predict P(Rg|T) mass using baseline joint P0(m_bin, Rg_bin).

    joint0_mass: shape (nC, nR), sums to 1 over (m,Rg)
    Returns rg_mass: shape (nR,), sums to 1.
    """
    k = -(h / T - s)
    x = k * c_centers
    x = x - np.max(x)  # stabilize
    w_m = np.exp(x)    # shape (nC,)

    # Weight each contact row and marginalize over m
    weighted = joint0_mass * w_m[:, None]
    rg_mass = weighted.sum(axis=0)

    Z = rg_mass.sum()
    if Z <= 0 or not np.isfinite(Z):
        return np.full(rg_mass.shape, 1.0 / rg_mass.size, dtype=float)
    return rg_mass / Z


# ------------------------- fitting objective -------------------------
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
    """Estimate h (only) from log-ratio slopes."""
    pref = p_obs_mass[ref_idx]
    Tref = float(temps[ref_idx])

    hs = []
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
        hs.append(float(h_i))
        weights.append(float(W))

    if not hs:
        return float("nan")

    hs = np.asarray(hs)
    weights = np.asarray(weights)
    return float(np.sum(weights * hs) / np.sum(weights))


# ------------------------- main -------------------------
def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--remd", type=str, default='remd_distributions_30mer.npz',
                    help="NPZ with temps, ct_centers, ct_hists (pdf); optional rg_centers, rg_hists")
    ap.add_argument("--baseline", type=str, required=False, default='single_uniform_chain2_athermal_dists_joint_N30_T1_seed42.npz',
                    help="NPZ baseline athermal contacts (c_vals,c_prob). Optional joint: (c_edges,rg_edges,crg_prob).")
    ap.add_argument("--contact_offset", type=float, default=29,
                    help="Constant to subtract from REMD ct_centers (e.g., N-1 if contacts include bonded neighbors).")
    ap.add_argument("--fit_h_only", action="store_true",
                    help="Fit only h from log-ratio slopes (does not require baseline)")

    ap.add_argument("--ref_idx", type=int, default=0,
                    help="Reference temperature index for --fit_h_only")
    ap.add_argument("--out", type=str, default="fit_hs_out.npz",
                    help="Output NPZ path")
    ap.add_argument("--no_plots", action="store_true")

    # Optimization settings
    ap.add_argument("--h0", type=float, default=900.0)
    ap.add_argument("--s0", type=float, default=2.0)
    ap.add_argument("--h_bounds", type=float, nargs=2, default=[-2000.0, 2000.0])
    ap.add_argument("--s_bounds", type=float, nargs=2, default=[-5.0, 5.0])
    ap.add_argument("--n_restarts", type=int, default=10)
    ap.add_argument("--seed", type=int, default=123)

    args = ap.parse_args()

    d = np.load(args.remd)
    temps = np.asarray(d["temps"], dtype=float)

    # contact distributions (observed)
    c_centers = np.asarray(d["ct_centers"], dtype=float) - float(args.contact_offset)
    ct_pdf = np.asarray(d["ct_hists"], dtype=float)

    p_obs_mass = np.zeros_like(ct_pdf)
    for i in range(ct_pdf.shape[0]):
        p_obs_mass[i], _ = pdf_to_mass(ct_pdf[i], c_centers)

    # optional observed Rg distributions
    rg_centers_obs = None
    rg_obs_mass = None
    if ("rg_centers" in d.files) and ("rg_hists" in d.files):
        rg_centers_obs = np.asarray(d["rg_centers"], dtype=float)
        rg_pdf = np.asarray(d["rg_hists"], dtype=float)
        rg_obs_mass = np.zeros_like(rg_pdf)
        for i in range(rg_pdf.shape[0]):
            rg_obs_mass[i], _ = pdf_to_mass(rg_pdf[i], rg_centers_obs)

    if args.fit_h_only or args.baseline is None:
        h_hat = fit_h_only_from_ratios(temps, c_centers, p_obs_mass, ref_idx=args.ref_idx)
        print(f"h_hat (ratio fit, s not identifiable without baseline) = {h_hat:.6g}")

        np.savez_compressed(
            args.out,
            h=float(h_hat), s=float("nan"),
            temps=temps,
            ct_centers=c_centers,
            p_obs_mass=p_obs_mass,
            rg_centers=rg_centers_obs if rg_centers_obs is not None else np.array([]),
            rg_obs_mass=rg_obs_mass if rg_obs_mass is not None else np.array([]),
        )

        if not args.no_plots:
            mean_obs = (c_centers[None, :] * p_obs_mass).sum(axis=1)
            plt.figure(figsize=(6, 4))
            plt.plot(temps, mean_obs, 'o-')
            plt.xlabel('T (K)')
            plt.ylabel('mean contacts')
            plt.title('Observed mean contacts')
            plt.tight_layout()
            plt.show()
        return

    if minimize is None:
        raise RuntimeError("scipy is required for fitting. Install scipy or run with --fit_h_only.")

    # baseline P0(m)
    p0_mass = build_baseline_mass(c_centers, args.baseline)

    # optional joint baseline for Rg prediction
    joint0_mass, rg_edges0 = build_baseline_joint_mass(c_centers, args.baseline)
    rg_centers0 = None
    if rg_edges0 is not None:
        rg_centers0 = 0.5 * (rg_edges0[:-1] + rg_edges0[1:])

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
        if float(res.fun) < best_val:
            best_val = float(res.fun)
            best = res

    if best is None:
        raise RuntimeError("Fitting failed")

    h_fit, s_fit = float(best.x[0]), float(best.x[1])
    print(f"Best-fit: h = {h_fit:.6g}, s = {s_fit:.6g}")
    print(f"Objective (sum KL) = {best_val:.6g}")

    # model predictions for contacts
    p_mod_mass = np.zeros_like(p_obs_mass)
    for i, T in enumerate(temps):
        p_mod_mass[i] = model_contact_mass(p0_mass, c_centers, float(T), h_fit, s_fit)

    # model predictions for Rg (if joint baseline exists)
    rg_mod_mass = None
    rg_mod_pdf_on_obs = None
    if joint0_mass is not None and rg_edges0 is not None:
        rg_mod_mass = np.zeros((len(temps), joint0_mass.shape[1]), dtype=float)
        for i, T in enumerate(temps):
            rg_mod_mass[i] = model_rg_mass_from_joint(joint0_mass, rg_edges0, c_centers, float(T), h_fit, s_fit)

        # If we also have observed Rg centers, build a pdf on those centers for overlay plotting
        if rg_centers_obs is not None and rg_centers0 is not None:
            rg_mod_pdf_on_obs = np.zeros((len(temps), rg_centers_obs.size), dtype=float)
            # baseline pdf is piecewise constant; we'll use bin-center pdf then interpolate
            widths0 = np.diff(rg_edges0)
            widths0 = np.where(widths0 <= 0, 1.0, widths0)
            for i in range(len(temps)):
                pdf0 = rg_mod_mass[i] / widths0
                pdf_interp = np.interp(rg_centers_obs, rg_centers0, pdf0, left=0.0, right=0.0)
                # renormalize to integrate ~1 over observed grid spacing
                if rg_centers_obs.size > 1:
                    dR = float(np.mean(np.diff(rg_centers_obs)))
                    area = float(np.sum(pdf_interp) * dR)
                    if area > 0:
                        pdf_interp /= area
                rg_mod_pdf_on_obs[i] = pdf_interp

    out_path = Path(args.out)
    np.savez_compressed(
        out_path,
        h=h_fit, s=s_fit,
        temps=temps,
        ct_centers=c_centers,
        p_obs_mass=p_obs_mass,
        p_mod_mass=p_mod_mass,
        p0_mass=p0_mass,
        # Rg outputs (may be empty)
        rg_edges0=rg_edges0 if rg_edges0 is not None else np.array([]),
        rg_centers0=rg_centers0 if rg_centers0 is not None else np.array([]),
        rg_mod_mass=rg_mod_mass if rg_mod_mass is not None else np.array([]),
        rg_centers_obs=rg_centers_obs if rg_centers_obs is not None else np.array([]),
        rg_obs_mass=rg_obs_mass if rg_obs_mass is not None else np.array([]),
        rg_mod_pdf_on_obs=rg_mod_pdf_on_obs if rg_mod_pdf_on_obs is not None else np.array([]),
        baseline=str(args.baseline),
        contact_offset=float(args.contact_offset),
    )
    print(f"Saved: {out_path}")

    if args.no_plots:
        return

    # ---------- plots ----------
    # Mean contacts vs T
    mean_obs = (c_centers[None, :] * p_obs_mass).sum(axis=1)
    mean_mod = (c_centers[None, :] * p_mod_mass).sum(axis=1)

    plt.figure(figsize=(6, 4))
    plt.plot(temps, mean_obs, 'o', label='obs')
    plt.plot(temps, mean_mod, '-', label='model')
    plt.xlabel('T (K)')
    plt.ylabel('mean contacts')
    plt.legend()
    plt.tight_layout()

    # Mean Rg vs T (only if we have observed and predicted)
    if rg_obs_mass is not None and rg_centers_obs is not None and rg_mod_pdf_on_obs is not None:
        mean_rg_obs = (rg_centers_obs[None, :] * rg_obs_mass).sum(axis=1)
        # predicted mean on baseline edges is more accurate; if available use that
        if rg_mod_mass is not None and rg_centers0 is not None:
            mean_rg_mod = (rg_centers0[None, :] * rg_mod_mass).sum(axis=1)
        else:
            mean_rg_mod = np.full_like(mean_rg_obs, np.nan)

        plt.figure(figsize=(6, 4))
        plt.plot(temps, mean_rg_obs, 'o', label='obs')
        plt.plot(temps, mean_rg_mod, '-', label='model')
        plt.xlabel('T (K)')
        plt.ylabel('mean $R_g$')
        plt.legend()
        plt.tight_layout()

    # Overlay a handful of contact distributions
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    idxs = np.linspace(0, len(temps) - 1, 8).astype(int)
    for i in idxs:
        ax.plot(c_centers, p_obs_mass[i], alpha=0.35)
        ax.plot(c_centers, p_mod_mass[i], alpha=0.9, lw=1.5)
    ax.set_xlabel('contacts')
    ax.set_ylabel('P(contacts) [mass]')
    ax.set_title('Contacts: obs (faint) vs model (bold)')
    plt.tight_layout()

    # Overlay a handful of Rg distributions if possible
    if rg_centers_obs is not None and rg_obs_mass is not None and rg_mod_pdf_on_obs is not None:
        fig, ax = plt.subplots(1, 1, figsize=(7, 4))
        for i in idxs:
            # observed is mass; convert to pdf for nicer overlay vs interpolated pdf
            # (mass->pdf on the observed grid using constant spacing approximation)
            if rg_centers_obs.size > 1:
                dR = float(np.mean(np.diff(rg_centers_obs)))
            else:
                dR = 1.0
            obs_pdf = rg_obs_mass[i] / dR
            ax.plot(rg_centers_obs, obs_pdf, alpha=0.35)
            ax.plot(rg_centers_obs, rg_mod_pdf_on_obs[i], alpha=0.9, lw=1.5)
        ax.set_xlabel('$R_g$')
        ax.set_ylabel('P($R_g$) [pdf]')
        ax.set_title('$R_g$: obs (faint) vs model (bold)')
        plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
