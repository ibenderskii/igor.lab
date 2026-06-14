#!/usr/bin/env python3
"""Fit the thermodynamic contact Hamiltonian to histogram data (with sane contact binning).

This version fixes the *spiky/discrete* look you were seeing by
*rebinnig the REMD contact PDFs onto integer contact bins* before fitting
and plotting.

Why this matters for your data
------------------------------
Your REMD file uses ~149 evenly spaced histogram bins for contacts with a
non-integer bin width (~0.537). Your SAW contact count m is an integer.
If you multiply an integer-m baseline by Boltzmann weights and then plot it
on the REMD centers, it looks like a comb (lots of empty bins) and the
reweighting can exaggerate that.

So here we do:
  1) Take the REMD *pdf on its native bins*
  2) Integrate it onto integer bins [m-0.5, m+0.5)
  3) Fit/plot on integer m

If the athermal baseline NPZ contains a joint histogram P0(m, Rg)
(keys: c_edges, rg_edges, crg_prob), we also predict P(Rg|T) by
reweighting the joint baseline.
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


def rebin_pdf_mass_to_integer_bins(ct_centers: np.ndarray,
                                  ct_pdf_row: np.ndarray,
                                  m_min: Optional[int] = None,
                                  m_max: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Rebin a contact pdf defined on ct_centers onto integer m bins.

    We treat ct_pdf_row[j] as a piecewise-constant density on the bin
    [edges[j], edges[j+1]) where edges come from centers_to_edges.

    Returns
    -------
    m_vals : (nM,) integer contact values
    p_mass : (nM,) probability mass for each integer m
    """
    ct_centers = np.asarray(ct_centers, dtype=float)
    ct_pdf_row = np.asarray(ct_pdf_row, dtype=float)

    edges = centers_to_edges(ct_centers)
    # choose integer range that covers essentially all support
    if m_min is None:
        m_min = int(np.floor(edges[0] + 0.5))
    if m_max is None:
        m_max = int(np.ceil(edges[-1] - 0.5))
    if m_max < m_min:
        m_max = m_min

    m_vals = np.arange(m_min, m_max + 1, dtype=int)
    int_edges = np.arange(m_min - 0.5, m_max + 1.5, 1.0, dtype=float)

    # mass in native bins
    native_mass, _ = pdf_to_mass(ct_pdf_row, ct_centers)
    native_edges = edges

    # piecewise-constant mass density within each native bin
    # density = mass / width
    native_w = np.diff(native_edges)
    dens = np.zeros_like(native_mass)
    mask = native_w > 0
    dens[mask] = native_mass[mask] / native_w[mask]

    # overlap integration
    p_int = np.zeros(m_vals.size, dtype=float)
    j = 0
    for i in range(m_vals.size):
        a, b = float(int_edges[i]), float(int_edges[i + 1])
        # advance j until native bin overlaps
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


def build_baseline_mass_on_integer(m_centers_int: np.ndarray, baseline_npz: str) -> np.ndarray:
    """Return baseline probability mass p0(m) on the integer m grid."""
    b = np.load(baseline_npz)
    m_centers_int = np.asarray(m_centers_int, dtype=float)
    m0 = int(round(m_centers_int.min()))
    m1 = int(round(m_centers_int.max()))
    m_vals = np.arange(m0, m1 + 1, dtype=int)

    # Case A: discrete contact values
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

    raise ValueError("Unrecognized baseline format. Expected (c_vals,c_prob) or (ct_centers,ct_hists).")


def model_contact_mass(p0_mass: np.ndarray,
                       m_centers: np.ndarray,
                       T: float,
                       h: float,
                       s: float) -> np.ndarray:
    """Return model probability mass over integer contact bins."""
    m_centers = np.asarray(m_centers, dtype=float)
    k = -(h / T - s)  # exp(k*m)

    # stabilize
    x = k * m_centers
    x = x - np.max(x)
    w = p0_mass * np.exp(x)
    Z = w.sum()
    if not np.isfinite(Z) or Z <= 0:
        return np.full_like(p0_mass, 1.0 / p0_mass.size)
    return w / Z


def kl_div(p_obs: np.ndarray, p_mod: np.ndarray, eps: float = 1e-12) -> float:
    p_obs = np.asarray(p_obs, dtype=float)
    p_mod = np.asarray(p_mod, dtype=float)
    p_obs = np.clip(p_obs, eps, 1.0)
    p_mod = np.clip(p_mod, eps, 1.0)
    p_obs /= p_obs.sum()
    p_mod /= p_mod.sum()
    return float(np.sum(p_obs * (np.log(p_obs) - np.log(p_mod))))


def objective_hs(params: np.ndarray,
                 temps: np.ndarray,
                 m_centers: np.ndarray,
                 p_obs_mass: np.ndarray,
                 p0_mass: np.ndarray) -> float:
    h, s = float(params[0]), float(params[1])
    total = 0.0
    for i, T in enumerate(temps):
        p_mod = model_contact_mass(p0_mass, m_centers, float(T), h, s)
        total += kl_div(p_obs_mass[i], p_mod)
    return total


def predict_rg_from_joint(crg_prob: np.ndarray,
                          c_edges: np.ndarray,
                          rg_edges: np.ndarray,
                          temps: np.ndarray,
                          h: float,
                          s: float) -> Tuple[np.ndarray, np.ndarray]:
    """Predict P(Rg|T) from baseline joint P0(m,Rg) by reweighting in m."""
    c_edges = np.asarray(c_edges, dtype=float)
    rg_edges = np.asarray(rg_edges, dtype=float)
    crg_prob = np.asarray(crg_prob, dtype=float)

    # m bin centers (integer-centered if you built them that way)
    m_centers = 0.5 * (c_edges[:-1] + c_edges[1:])

    rg_mass_T = np.zeros((temps.size, rg_edges.size - 1), dtype=float)
    for i, T in enumerate(temps):
        k = -(h / float(T) - s)
        x = k * m_centers
        x = x - np.max(x)
        w_m = np.exp(x)  # per-m weight
        weighted = (crg_prob.T * w_m).T  # broadcast over Rg
        rg_mass = weighted.sum(axis=0)
        Z = rg_mass.sum()
        if Z > 0:
            rg_mass /= Z
        rg_mass_T[i] = rg_mass
    rg_centers = 0.5 * (rg_edges[:-1] + rg_edges[1:])
    return rg_centers, rg_mass_T


def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--remd", type=str, default='remd_distributions_44mer.npz',
                    help="NPZ with temps, ct_centers, ct_hists (pdf). Optionally rg_centers, rg_hists.")
    ap.add_argument("--baseline", type=str, default='single_uniform_chain2_athermal_dists_joint_N44_T1_seed42.npz',
                    help="Baseline NPZ. For Rg prediction, must contain c_edges, rg_edges, crg_prob.")
    ap.add_argument("--contact_offset", type=float, default=43,
                    help="Constant to subtract from ct_centers in the REMD file.")
    ap.add_argument("--out", type=str, default="fit_hs_withRg_rebin.npz")
    ap.add_argument("--no_plots", action="store_true")

    # optimization
    ap.add_argument("--h0", type=float, default=750.0)
    ap.add_argument("--s0", type=float, default=2.8)
    ap.add_argument("--h_bounds", type=float, nargs=2, default=[-2000.0, 2000.0])
    ap.add_argument("--s_bounds", type=float, nargs=2, default=[-10.0, 10.0])
    ap.add_argument("--n_restarts", type=int, default=8)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    d = np.load(args.remd)
    temps = np.asarray(d["temps"], dtype=float)
    ct_centers_native = np.asarray(d["ct_centers"], dtype=float) - float(args.contact_offset)
    ct_pdf = np.asarray(d["ct_hists"], dtype=float)

    # Determine integer m range once (from native edges)
    native_edges = centers_to_edges(ct_centers_native)
    m_min = int(np.floor(native_edges[0] + 0.5))
    m_max = int(np.ceil(native_edges[-1] - 0.5))

    # rebin observed contacts onto integer bins
    m_centers, p_obs0 = rebin_pdf_mass_to_integer_bins(ct_centers_native, ct_pdf[0], m_min=m_min, m_max=m_max)
    p_obs_mass = np.zeros((ct_pdf.shape[0], m_centers.size), dtype=float)
    p_obs_mass[0] = p_obs0
    for i in range(1, ct_pdf.shape[0]):
        _, p_obs_mass[i] = rebin_pdf_mass_to_integer_bins(ct_centers_native, ct_pdf[i], m_min=m_min, m_max=m_max)

    # baseline p0(m) on the same integer bins
    p0_mass = build_baseline_mass_on_integer(m_centers, args.baseline)

    if minimize is None:
        raise RuntimeError("scipy is required for fitting. Install scipy.")

    rng = np.random.default_rng(args.seed)
    bounds = [tuple(args.h_bounds), tuple(args.s_bounds)]

    x0s = [np.array([args.h0, args.s0], dtype=float)]
    for _ in range(max(0, args.n_restarts - 1)):
        h = rng.uniform(args.h_bounds[0], args.h_bounds[1])
        s = rng.uniform(args.s_bounds[0], args.s_bounds[1])
        x0s.append(np.array([h, s], dtype=float))

    best = None
    best_val = float("inf")
    for x0 in x0s:
        res = minimize(
            objective_hs, x0,
            args=(temps, m_centers, p_obs_mass, p0_mass),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 800}
        )
        if res.fun < best_val:
            best_val = float(res.fun)
            best = res

    if best is None:
        raise RuntimeError("Fitting failed")

    h_fit, s_fit = float(best.x[0]), float(best.x[1])
    print(f"Best-fit: h = {h_fit:.6g}, s = {s_fit:.6g}")
    print(f"Objective (sum KL) = {best_val:.6g}")

    # contact predictions
    p_mod_mass = np.zeros_like(p_obs_mass)
    for i, T in enumerate(temps):
        p_mod_mass[i] = model_contact_mass(p0_mass, m_centers, float(T), h_fit, s_fit)

    # Rg prediction (optional)
    b = np.load(args.baseline)
    rg_centers0 = None
    rg_mod_mass = None
    if all(k in b.files for k in ("c_edges", "rg_edges", "crg_prob")):
        rg_centers0, rg_mod_mass = predict_rg_from_joint(
            crg_prob=b["crg_prob"],
            c_edges=b["c_edges"],
            rg_edges=b["rg_edges"],
            temps=temps,
            h=h_fit,
            s=s_fit,
        )

    out_path = Path(args.out)
    np.savez_compressed(
        out_path,
        h=h_fit, s=s_fit,
        temps=temps,
        m_centers=m_centers,
        p_obs_mass=p_obs_mass,
        p_mod_mass=p_mod_mass,
        p0_mass=p0_mass,
        baseline=str(args.baseline),
        rg_centers0=rg_centers0,
        rg_mod_mass=rg_mod_mass,
    )
    print(f"Saved: {out_path}")

    if args.no_plots:
        return

    # ---------- plots ----------
    mean_obs = (m_centers[None, :] * p_obs_mass).sum(axis=1)
    mean_mod = (m_centers[None, :] * p_mod_mass).sum(axis=1)

    plt.figure(figsize=(6, 4))
    plt.plot(temps, mean_obs, "o", label="obs")
    plt.plot(temps, mean_mod, "-", label="model")
    plt.xlabel("T")
    plt.ylabel("mean contacts")
    plt.legend()
    plt.tight_layout()

    # distribution overlay (step plots so the discreteness is honest)
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    idxs = np.linspace(0, len(temps) - 1, 8).astype(int)
    for i in idxs:
        ax.step(m_centers, p_obs_mass[i], where="mid", alpha=0.35)
        ax.step(m_centers, p_mod_mass[i], where="mid", alpha=0.9, lw=1.5)
    ax.set_xlabel("m (integer contacts)")
    ax.set_ylabel("P(m)")
    ax.set_title("Obs (faint) vs model (bold), rebinned to integer m")
    plt.tight_layout()

    if rg_centers0 is not None and rg_mod_mass is not None:
        fig2, ax2 = plt.subplots(1, 1, figsize=(7, 4))
        for i in idxs:
            ax2.plot(rg_centers0, rg_mod_mass[i], alpha=0.9, lw=1.5)
        ax2.set_xlabel("Rg")
        ax2.set_ylabel("P(Rg)")
        ax2.set_title("Predicted P(Rg|T) from joint baseline reweighting")
        plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
