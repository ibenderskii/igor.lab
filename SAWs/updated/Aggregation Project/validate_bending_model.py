#!/usr/bin/env python3
"""
Focused scientific validation of the fixed reduced bending penalty.

Runs short *contact-athermal* single-chain lattice Monte Carlo simulations over a
grid of bending penalties ``kappa_bend`` and reports how chain stiffness responds.
"Contact-athermal" means the contact reduced bias is identically zero (model
``hs`` with parameters ``(h, s) = (0, 0)`` gives ``b(T) = 0``), so the ONLY term
in the reduced potential is the bending penalty ``u = kappa_bend * n_bend`` and
the sampled measure is exp[-kappa_bend * n_bend(X)] over self-avoiding walks.
This isolates the bending physics that the samplers add on top of the contact
model.

For each ``kappa_bend`` and seed the script measures:

    * mean / std of the 90-degree-turn count and the bend fraction n_bend/(N-2)
    * straight-step fraction (1 - bend fraction)
    * tangent correlation C(s) = <u_i . u_{i+s}> for several contour separations
    * a persistence length from an early-lag exponential fit of C(s) (when valid)
    * mean radius of gyration and mean nonbonded contact count
    * the local-move acceptance rate
    * an ESS / integrated-autocorrelation-time diagnostic for the bend count,
      reusing :func:`remd_uniform_chain_2_new.integrated_autocorr_time`

An ideal *local* reference straight-step probability

    P_straight = 1 / (1 + 4 * exp(-kappa_bend))

is reported alongside the measurements.  It counts the five SAW continuations at a
vertex (one straight, four right-angle turns) weighted by exp(0) and exp(-kappa)
and is NOT an exact prediction for a self-avoiding finite chain -- it ignores
excluded volume and end effects.  It is a sanity anchor only.

Outputs (written to ``--outdir``):

    bending_validation_per_seed.csv
    bending_validation_summary.csv
    bending_validation_summary.json
    bending_validation_bend_fraction.png
    bending_validation_persistence_length.png
    bending_validation_rg.png
    bending_validation_contacts.png

Quick check:  python validate_bending_model.py --quick-test
Full example: python validate_bending_model.py --N 30 --steps 4000 --seeds 4
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import remd_uniform_chain_2_new as remd  # single-chain sampler + bending support
from lattice_bending import BEND_DEFINITION, count_bends

DEFAULT_KAPPA_GRID = (0.0, 1.0, 1.5, 1.75, 2.0, 2.5)
# Contour separations at which the tangent correlation C(s) is reported.
DEFAULT_CONTOUR_SEPARATIONS = (1, 2, 3, 4, 5)
# Athermal contact model: b(T) = h/T - s = 0 for (h, s) = (0, 0).
_ATHERMAL = dict(model_name="hs", params=[0.0, 0.0], Tref=300.0, Tscale=80.0,
                 T=300.0)


# ---------------------------------------------------------------------------
# Ideal local reference
# ---------------------------------------------------------------------------

def ideal_straight_probability(kappa_bend: float) -> float:
    """Ideal *local* straight-step probability 1 / (1 + 4 exp(-kappa_bend)).

    Five self-avoiding continuations at a vertex (one straight, four turns) with
    Boltzmann weights exp(0) and exp(-kappa_bend).  A local, excluded-volume-free
    reference -- NOT an exact prediction for a finite self-avoiding chain.
    """
    return 1.0 / (1.0 + 4.0 * math.exp(-float(kappa_bend)))


# ---------------------------------------------------------------------------
# Tangent correlation / persistence length
# ---------------------------------------------------------------------------

def tangent_correlation(bond_sums: dict, bond_counts: dict) -> dict:
    """Averaged C(s) from accumulated bond dot-product sums."""
    return {s: (bond_sums[s] / bond_counts[s] if bond_counts[s] else float("nan"))
            for s in bond_sums}


def persistence_length(cs: dict, max_lag: int) -> float:
    """Persistence length from an early-lag exponential fit of C(s).

    Fits ln C(s) = -s / Lp over the contiguous small-s lags (1..max_lag) for which
    C(s) > 0, requiring at least two such points and a decaying (negative-slope)
    fit.  Returns Lp in lattice (bond) units, or NaN when the fit is not valid.
    """
    xs, ys = [], []
    for s in range(1, int(max_lag) + 1):
        c = cs.get(s, float("nan"))
        if not (np.isfinite(c) and c > 0.0):
            break  # exponential fit only over the contiguous positive early lags
        xs.append(float(s))
        ys.append(math.log(c))
    if len(xs) < 2:
        return float("nan")
    slope, _ = np.polyfit(np.asarray(xs), np.asarray(ys), 1)
    if not np.isfinite(slope) or slope >= 0.0:
        return float("nan")
    return float(-1.0 / slope)


# ---------------------------------------------------------------------------
# One (kappa, seed) simulation
# ---------------------------------------------------------------------------

def run_single(N: int, kappa_bend: float, steps: int, burnin_frac: float,
               seed: int, separations) -> dict:
    """Sample one contact-athermal chain at fixed ``kappa_bend`` and seed.

    One "step" is one sweep of ``N`` local Metropolis moves; observables are
    recorded once per sweep, and the first ``burnin_frac`` fraction is discarded.
    """
    remd._seed_all(int(seed))
    state = remd.ChainState.initial_straight(
        N, _ATHERMAL["T"], _ATHERMAL["model_name"], _ATHERMAL["params"],
        _ATHERMAL["Tref"], _ATHERMAL["Tscale"])
    rep = remd.Replica(T=_ATHERMAL["T"], state=state)

    seps = [s for s in separations if 1 <= s <= N - 2]
    bond_sums = {s: 0.0 for s in seps}
    bond_counts = {s: 0 for s in seps}
    n_bend_samples, rg_samples, contact_samples = [], [], []

    burnin = int(math.floor(int(steps) * float(burnin_frac)))
    for step in range(int(steps)):
        remd.mc_sweep(rep, N, _ATHERMAL["model_name"], _ATHERMAL["params"],
                      _ATHERMAL["Tref"], _ATHERMAL["Tscale"], kappa_bend)
        if step < burnin:
            continue
        chain = np.asarray(rep.state.chain, dtype=np.int64)
        n_bend_samples.append(int(rep.state.n_bend))
        rg_samples.append(math.sqrt(remd.ico.radius_of_gyration_squared(rep.state.chain)))
        contact_samples.append(float(rep.state.m))
        bonds = np.diff(chain, axis=0).astype(np.float64)  # (N-1, 3) unit vectors
        for s in seps:
            dots = np.einsum("ij,ij->i", bonds[:-s], bonds[s:])
            bond_sums[s] += float(dots.sum())
            bond_counts[s] += int(dots.size)

    n_bend = np.asarray(n_bend_samples, dtype=float)
    denom = max(N - 2, 0)
    mean_bend = float(n_bend.mean()) if n_bend.size else float("nan")
    bend_fraction = mean_bend / denom if denom > 0 else float("nan")
    cs = tangent_correlation(bond_sums, bond_counts)
    ac = remd.integrated_autocorr_time(n_bend)

    return {
        "N": int(N),
        "kappa_bend": float(kappa_bend),
        "seed": int(seed),
        "n_samples": int(n_bend.size),
        "mean_bend": mean_bend,
        "std_bend": float(n_bend.std(ddof=0)) if n_bend.size else float("nan"),
        "bend_fraction": bend_fraction,
        "straight_fraction": (1.0 - bend_fraction
                              if math.isfinite(bend_fraction) else float("nan")),
        "P_straight_ideal": ideal_straight_probability(kappa_bend),
        "persistence_length": persistence_length(
            cs, max_lag=min(len(cs), max(2, N // 3))),
        "mean_Rg": float(np.mean(rg_samples)) if rg_samples else float("nan"),
        "mean_contacts": float(np.mean(contact_samples)) if contact_samples else float("nan"),
        "acceptance_rate": float(rep.state_changing_acceptance_rate),
        "ess_n_bend": float(ac["ess"]),
        "tau_int_n_bend": float(ac["tau_int"]),
        "tangent_correlation": {int(s): float(cs[s]) for s in cs},
    }


# ---------------------------------------------------------------------------
# Grid driver + aggregation
# ---------------------------------------------------------------------------

def run_grid(N, kappa_grid, steps, burnin_frac, seeds, separations) -> list:
    rows = []
    for kappa in kappa_grid:
        for seed in seeds:
            rows.append(run_single(N, kappa, steps, burnin_frac, seed, separations))
    return rows


def _agg(values) -> tuple:
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=float)
    if v.size == 0:
        return float("nan"), float("nan"), 0
    return float(v.mean()), float(v.std(ddof=0)), int(v.size)


def summarize(rows, kappa_grid, separations) -> list:
    seps = list(separations)
    summary = []
    for kappa in kappa_grid:
        group = [r for r in rows if r["kappa_bend"] == kappa]
        entry = {"kappa_bend": float(kappa), "n_seeds": len(group),
                 "P_straight_ideal": ideal_straight_probability(kappa)}
        for key in ("bend_fraction", "straight_fraction", "persistence_length",
                    "mean_Rg", "mean_contacts", "mean_bend", "acceptance_rate",
                    "ess_n_bend"):
            mean, std, n = _agg([r[key] for r in group])
            entry[f"{key}_mean"] = mean
            entry[f"{key}_std"] = std
            entry[f"{key}_n"] = n
        for s in seps:
            mean, _, _ = _agg([r["tangent_correlation"].get(s, float("nan"))
                               for r in group])
            entry[f"C_{s}_mean"] = mean
        summary.append(entry)
    return summary


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _finite(x):
    return x if (isinstance(x, (int, float)) and math.isfinite(x)) else (
        None if isinstance(x, float) else x)


def write_per_seed_csv(rows, path, separations) -> str:
    seps = list(separations)
    cols = ["N", "kappa_bend", "seed", "n_samples", "mean_bend", "std_bend",
            "bend_fraction", "straight_fraction", "P_straight_ideal",
            "persistence_length", "mean_Rg", "mean_contacts", "acceptance_rate",
            "ess_n_bend", "tau_int_n_bend"] + [f"C_{s}" for s in seps]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            base = [r.get(c, "") for c in cols[:15]]
            cvals = [r["tangent_correlation"].get(s, float("nan")) for s in seps]
            w.writerow(base + cvals)
    return str(path)


def write_summary_csv(summary, path, separations) -> str:
    seps = list(separations)
    cols = ["kappa_bend", "n_seeds", "P_straight_ideal",
            "bend_fraction_mean", "bend_fraction_std",
            "straight_fraction_mean", "straight_fraction_std",
            "persistence_length_mean", "persistence_length_std",
            "mean_Rg_mean", "mean_Rg_std", "mean_contacts_mean",
            "mean_contacts_std", "mean_bend_mean", "mean_bend_std",
            "acceptance_rate_mean", "ess_n_bend_mean"] + [f"C_{s}_mean" for s in seps]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for e in summary:
            w.writerow([e.get(c, "") for c in cols])
    return str(path)


def write_summary_json(summary, rows, meta, path) -> str:
    def clean(obj):
        if isinstance(obj, dict):
            return {str(k): clean(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [clean(v) for v in obj]
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating, float)):
            v = float(obj)
            return v if math.isfinite(v) else None
        return obj
    payload = {"metadata": meta, "summary": summary, "per_seed": rows}
    with open(path, "w") as f:
        json.dump(clean(payload), f, indent=2)
    return str(path)


def make_plots(summary, outdir) -> list:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("  [plot] matplotlib unavailable; skipping plots")
        return []
    kappa = [e["kappa_bend"] for e in summary]

    def plot(key, ylabel, fname, extra=None):
        fig, ax = plt.subplots(figsize=(5, 4))
        y = [e[f"{key}_mean"] for e in summary]
        yerr = [e.get(f"{key}_std", 0.0) for e in summary]
        ax.errorbar(kappa, y, yerr=yerr, fmt="o-", capsize=3, label="simulation")
        if extra is not None:
            ax.plot(kappa, extra, "s--", color="gray",
                    label="ideal local reference")
            ax.legend()
        ax.set_xlabel("kappa_bend")
        ax.set_ylabel(ylabel)
        fig.tight_layout()
        path = os.path.join(outdir, fname)
        fig.savefig(path, dpi=110)
        plt.close(fig)
        return path

    paths = [
        plot("bend_fraction", "mean bend fraction n_bend/(N-2)",
             "bending_validation_bend_fraction.png"),
        plot("persistence_length", "persistence length (lattice units)",
             "bending_validation_persistence_length.png"),
        plot("mean_Rg", "mean Rg (lattice units)", "bending_validation_rg.png"),
        plot("mean_contacts", "mean nonbonded contacts",
             "bending_validation_contacts.png"),
    ]
    # A companion straight-fraction plot with the ideal local reference overlay.
    paths.append(plot(
        "straight_fraction", "straight-step fraction",
        "bending_validation_straight_fraction.png",
        extra=[e["P_straight_ideal"] for e in summary]))
    for p in paths:
        print(f"Saved {p}")
    return paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_grid(text) -> list:
    return [float(v) for v in str(text).split(",") if v.strip() != ""]


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=("Validate the fixed reduced bending penalty on short "
                     "contact-athermal single-chain simulations."))
    ap.add_argument("--N", type=int, default=24, help="beads per chain")
    ap.add_argument("--kappa-grid", type=str,
                    default=",".join(str(k) for k in DEFAULT_KAPPA_GRID),
                    dest="kappa_grid",
                    help="comma-separated bending penalties")
    ap.add_argument("--steps", type=int, default=3000,
                    help="local-move sweeps per (kappa, seed) run (one sweep = N moves)")
    ap.add_argument("--burnin-frac", type=float, default=0.3, dest="burnin_frac")
    ap.add_argument("--seeds", type=int, default=3,
                    help="number of seeds (0..seeds-1) per kappa")
    ap.add_argument("--separations", type=str,
                    default=",".join(str(s) for s in DEFAULT_CONTOUR_SEPARATIONS),
                    help="comma-separated tangent-correlation contour separations")
    ap.add_argument("--outdir", type=str, default="bending_validation_out")
    ap.add_argument("--no-plots", action="store_true", dest="no_plots")
    ap.add_argument("--quick-test", action="store_true", dest="quick_test")
    return ap


def _validate(args) -> None:
    if args.N < 3:
        raise ValueError("--N must be >= 3")
    if args.steps < 1:
        raise ValueError("--steps must be >= 1")
    if not (0.0 <= args.burnin_frac < 1.0):
        raise ValueError("--burnin-frac must be in [0, 1)")
    if args.seeds < 1:
        raise ValueError("--seeds must be >= 1")
    grid = _parse_grid(args.kappa_grid)
    if not grid:
        raise ValueError("--kappa-grid must contain at least one value")
    for k in grid:
        if not math.isfinite(k) or k < 0.0:
            raise ValueError(f"kappa values must be finite and >= 0, got {k!r}")


def main(argv=None) -> None:
    args = build_arg_parser().parse_args(argv)
    if args.quick_test:
        run_quick_test()
        return
    _validate(args)

    kappa_grid = _parse_grid(args.kappa_grid)
    separations = [int(s) for s in _parse_grid(args.separations)]
    seeds = list(range(int(args.seeds)))
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Bending validation: N={args.N} steps={args.steps} "
          f"burnin={args.burnin_frac} seeds={seeds}")
    print(f"kappa grid: {kappa_grid}")
    print(f"Contact-athermal model (b(T)=0); {BEND_DEFINITION}")

    rows = run_grid(args.N, kappa_grid, args.steps, args.burnin_frac, seeds,
                    separations)
    summary = summarize(rows, kappa_grid, separations)

    meta = {
        "N": int(args.N), "steps": int(args.steps),
        "burnin_frac": float(args.burnin_frac), "seeds": seeds,
        "kappa_grid": kappa_grid, "separations": separations,
        "bend_definition": BEND_DEFINITION,
        "model": "hs (contact-athermal, b(T)=0)",
        "ideal_reference": "P_straight = 1/(1+4 exp(-kappa)) [local, not exact]",
    }
    per_seed_csv = write_per_seed_csv(
        rows, outdir / "bending_validation_per_seed.csv", separations)
    summary_csv = write_summary_csv(
        summary, outdir / "bending_validation_summary.csv", separations)
    summary_json = write_summary_json(
        summary, rows, meta, outdir / "bending_validation_summary.json")
    print(f"Saved {per_seed_csv}")
    print(f"Saved {summary_csv}")
    print(f"Saved {summary_json}")
    if not args.no_plots:
        make_plots(summary, str(outdir))

    print("\nkappa_bend  bend_fraction   Lp        mean_Rg   P_straight(ideal)")
    for e in summary:
        print(f"  {e['kappa_bend']:<8.3g} "
              f"{e['bend_fraction_mean']:<14.4f} "
              f"{e['persistence_length_mean']:<9.3f} "
              f"{e['mean_Rg_mean']:<9.3f} {e['P_straight_ideal']:.4f}")


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

def run_quick_test() -> None:
    print("bending-validation quick-test:")
    N, steps, burnin = 12, 250, 0.3
    seps = [1, 2, 3]
    grid = [0.0, 3.0]  # broad range: strongly straightens at the top

    rows = run_grid(N, grid, steps, burnin, [0, 1], seps)
    # (a) All reported quantities are finite where they must be.
    for r in rows:
        for key in ("mean_bend", "bend_fraction", "straight_fraction",
                    "mean_Rg", "mean_contacts", "acceptance_rate", "ess_n_bend"):
            assert math.isfinite(r[key]), f"{key} not finite for {r['kappa_bend']}"
        assert 0.0 <= r["bend_fraction"] <= 1.0
    print("  quick-test finite outputs: PASSED")

    # (b) Bend fraction decreases monotonically over the broad kappa range.
    summary = summarize(rows, grid, seps)
    bf = [e["bend_fraction_mean"] for e in summary]
    assert bf[0] > bf[-1], (
        f"bend fraction did not drop over kappa range: {bf}")
    # The strong-penalty persistence length must exceed the flexible one when both
    # fits are valid (a broad-range sanity check, not an exact value).
    lp = [e["persistence_length_mean"] for e in summary]
    if all(math.isfinite(x) for x in lp):
        assert lp[-1] >= lp[0], f"persistence length did not grow with kappa: {lp}"
    print(f"  quick-test monotonic bend reduction "
          f"(bf {bf[0]:.3f} -> {bf[-1]:.3f}): PASSED")

    # (c) Reproducibility: same seed + kappa reproduces the bend statistics.
    a = run_single(N, 1.5, steps, burnin, seed=7, separations=seps)
    b = run_single(N, 1.5, steps, burnin, seed=7, separations=seps)
    assert a["mean_bend"] == b["mean_bend"] and a["std_bend"] == b["std_bend"]
    assert a["tangent_correlation"] == b["tangent_correlation"]
    print("  quick-test reproducibility (fixed seed): PASSED")

    # (d) The cached bend count agrees with an independent full recount.
    remd._seed_all(3)
    state = remd.ChainState.initial_straight(
        N, _ATHERMAL["T"], _ATHERMAL["model_name"], _ATHERMAL["params"],
        _ATHERMAL["Tref"], _ATHERMAL["Tscale"])
    rep = remd.Replica(T=_ATHERMAL["T"], state=state)
    for _ in range(60):
        remd.mc_sweep(rep, N, _ATHERMAL["model_name"], _ATHERMAL["params"],
                      _ATHERMAL["Tref"], _ATHERMAL["Tscale"], 0.8)
    assert rep.state.n_bend == count_bends(rep.state.chain)
    print("  quick-test cached bend count vs recount: PASSED")
    print("bending-validation quick-test complete.")


if __name__ == "__main__":
    main()
