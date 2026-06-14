#!/usr/bin/env python3
"""
Serial replica exchange Monte Carlo for the lattice polymer model.

Hamiltonian:  H(C; T) = m(C) * (dh - T*ds)
Reduced potential:  u(C, T) = H(C, T) / T = m(C) * (dh/T - ds)

Because H depends explicitly on T, the standard REMD swap criterion must use
the reduced potential rather than a temperature-independent energy:

    log_accept = u(C_i,T_i) + u(C_j,T_j) - u(C_j,T_i) - u(C_i,T_j)

Accept if random() < exp(min(0, log_accept)).

Usage:
    python remd_uniform_chain.py --N 50 --nT 8 --Tmin 280 --Tmax 380 \\
        --steps-per-swap 500 --n-cycles 4000 --seed 42
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import math
import random
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from thermo_uniform_chain2_DBfix_dists_corrected import (
    ChainState,
    MOVE_FUNCS,
    contact_count,
    energy,
    radius_of_gyration,
)

# ---------------------------------------------------------------------------
# Reduced potential and swap criterion
# ---------------------------------------------------------------------------

def reduced_potential(m: float, T: float, dh: float, ds: float) -> float:
    """u(C, T) = H(C, T) / T = m * (dh/T - ds)."""
    return m * (dh / T - ds)


def swap_log_accept(
    m_i: float, m_j: float,
    T_i: float, T_j: float,
    dh: float, ds: float,
) -> float:
    """
    Log Metropolis ratio for swapping configs C_i (at T_i) and C_j (at T_j).

    log_accept = u(C_i,T_i) + u(C_j,T_j) - u(C_j,T_i) - u(C_i,T_j)

    Equivalent to (m_i - m_j) * dh * (1/T_i - 1/T_j), but written in the
    general form so the derivation remains transparent.
    """
    return (
        reduced_potential(m_i, T_i, dh, ds)
        + reduced_potential(m_j, T_j, dh, ds)
        - reduced_potential(m_j, T_i, dh, ds)
        - reduced_potential(m_i, T_j, dh, ds)
    )


# ---------------------------------------------------------------------------
# Replica dataclass
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class Replica:
    """One thermostat lane: fixed temperature, evolving chain configuration."""
    T:     float
    state: ChainState
    local_acc:  int = 0
    local_prop: int = 0
    E_traj:  list = dataclasses.field(default_factory=list)
    C_traj:  list = dataclasses.field(default_factory=list)
    Rg_traj: list = dataclasses.field(default_factory=list)

    @property
    def local_acc_rate(self) -> float:
        return self.local_acc / self.local_prop if self.local_prop else float("nan")


# ---------------------------------------------------------------------------
# MC sweep and swap
# ---------------------------------------------------------------------------

def mc_sweep(replica: Replica, steps: int, dh: float, ds: float) -> None:
    """Run `steps` local Metropolis moves on `replica` in-place."""
    T     = replica.T
    beta  = 1.0 / T
    state = replica.state

    for _ in range(steps):
        replica.local_prop += 1
        move = random.choice(MOVE_FUNCS)
        ok, chain_new, occ_new = move(state.chain, state.occ)
        if not ok:
            continue
        dE = energy(chain_new, occ_new, dh, ds, T) - state.E
        if dE <= 0 or random.random() < math.exp(-beta * dE):
            state.chain = chain_new
            state.occ   = occ_new
            state.E     = state.E + dE
            replica.local_acc += 1


def attempt_swap(
    rep_a: Replica, rep_b: Replica,
    dh: float, ds: float,
) -> bool:
    """
    Attempt a configuration swap between two adjacent replicas.

    Temperatures stay fixed; configurations (and energies) are exchanged.
    Returns True if accepted.
    """
    m_a = contact_count(rep_a.state.chain, rep_a.state.occ)
    m_b = contact_count(rep_b.state.chain, rep_b.state.occ)
    log_acc = swap_log_accept(m_a, m_b, rep_a.T, rep_b.T, dh, ds)

    accepted = log_acc >= 0 or random.random() < math.exp(log_acc)
    if accepted:
        rep_a.state.chain, rep_b.state.chain = rep_b.state.chain, rep_a.state.chain
        rep_a.state.occ,   rep_b.state.occ   = rep_b.state.occ,   rep_a.state.occ
        rep_a.state.E = energy(rep_a.state.chain, rep_a.state.occ, dh, ds, rep_a.T)
        rep_b.state.E = energy(rep_b.state.chain, rep_b.state.occ, dh, ds, rep_b.T)
    return accepted


# ---------------------------------------------------------------------------
# Parallel worker (must be top-level for pickling on Windows spawn)
# ---------------------------------------------------------------------------

def evolve_replica_worker(
    replica: "Replica",
    steps: int,
    dh: float,
    ds: float,
    seed: int,
) -> "Replica":
    """Deterministically seeded sweep of one replica. Top-level for pickling."""
    random.seed(seed)
    np.random.seed(seed)
    mc_sweep(replica, steps, dh, ds)
    return replica


# ---------------------------------------------------------------------------
# Main REMD loop
# ---------------------------------------------------------------------------

def run_remd(
    N: int,
    Ts: np.ndarray,
    steps_per_swap: int,
    n_cycles: int,
    dh: float,
    ds: float,
    seed: int | None = None,
    verbose: bool = True,
    n_workers: int = 1,
    timing: bool = False,
) -> tuple[list[Replica], np.ndarray, np.ndarray]:
    """
    Run serial REMD.

    Each cycle:
      1. Every replica runs `steps_per_swap` local Metropolis steps.
      2. Adjacent pairs attempt swaps with even/odd alternation.
      3. Observables are recorded for every replica.

    Returns:
        replicas   — list of Replica objects with full trajectories
        swap_props — (nT-1,) array: swap proposals per adjacent pair
        swap_accs  — (nT-1,) array: swap acceptances per adjacent pair
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    nT = len(Ts)
    replicas: list[Replica] = [
        Replica(T=float(T), state=ChainState.initial_straight(N, dh, ds, float(T)))
        for T in Ts
    ]

    swap_props = np.zeros(nT - 1, dtype=int)
    swap_accs  = np.zeros(nT - 1, dtype=int)

    report_every = max(1, n_cycles // 20)
    base_seed = seed if seed is not None else 0
    executor = (
        ProcessPoolExecutor(max_workers=min(n_workers, nT))
        if n_workers > 1 else None
    )
    t_sweep_total = t_swap_total = 0.0

    try:
        for cycle in range(n_cycles):
            t0 = time.perf_counter()

            # Local sweeps — parallel or serial
            if executor is not None:
                worker_seeds = [base_seed + 100_000 * cycle + k for k in range(nT)]
                futures = [
                    executor.submit(
                        evolve_replica_worker, replicas[k],
                        steps_per_swap, dh, ds, worker_seeds[k],
                    )
                    for k in range(nT)
                ]
                for k, fut in enumerate(futures):
                    replicas[k] = fut.result()
            else:
                for rep in replicas:
                    mc_sweep(rep, steps_per_swap, dh, ds)

            t1 = time.perf_counter()
            t_sweep_total += t1 - t0

            # Swap attempts: alternate even (0-1, 2-3, …) and odd (1-2, 3-4, …) pairs
            start = cycle % 2
            for k in range(start, nT - 1, 2):
                swap_props[k] += 1
                if attempt_swap(replicas[k], replicas[k + 1], dh, ds):
                    swap_accs[k] += 1

            t2 = time.perf_counter()
            t_swap_total += t2 - t1

            # Record observables
            for rep in replicas:
                rep.E_traj.append(rep.state.E)
                rep.C_traj.append(contact_count(rep.state.chain, rep.state.occ))
                rep.Rg_traj.append(radius_of_gyration(rep.state.chain))

            if verbose and (cycle + 1) % report_every == 0:
                rates = " ".join(
                    f"{swap_accs[k]/max(1,swap_props[k]):.2f}"
                    for k in range(nT - 1)
                )
                print(f"  cycle {cycle+1:>6}/{n_cycles}  swap rates: {rates}")

    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    if timing:
        total = t_sweep_total + t_swap_total
        print(
            f"Timing:  sweeps {t_sweep_total:.2f}s  |"
            f"  swaps {t_swap_total:.2f}s  |"
            f"  total {total:.2f}s"
        )

    return replicas, swap_props, swap_accs


# ---------------------------------------------------------------------------
# Post-processing: statistics and distributions
# ---------------------------------------------------------------------------

def compute_statistics(
    replicas: list[Replica],
    burnin_frac: float = 0.7,
) -> list[dict]:
    """Compute means and stds over the last (1-burnin_frac) of each trajectory."""
    results = []
    for rep in replicas:
        n = len(rep.E_traj)
        s = int(math.floor(n * burnin_frac))

        E_arr  = np.array(rep.E_traj[s:],  dtype=float)
        C_arr  = np.array(rep.C_traj[s:],  dtype=float)
        Rg_arr = np.array(rep.Rg_traj[s:], dtype=float)

        results.append({
            "T":        rep.T,
            "E_mean":   float(np.nanmean(E_arr)),
            "E_std":    float(np.nanstd(E_arr,  ddof=0)),
            "C_mean":   float(np.nanmean(C_arr)),
            "C_std":    float(np.nanstd(C_arr,  ddof=0)),
            "Rg_mean":  float(np.nanmean(Rg_arr)),
            "Rg_std":   float(np.nanstd(Rg_arr, ddof=0)),
            "local_acc_rate": rep.local_acc_rate,
        })
    return results


def build_distributions(
    replicas: list[Replica],
    rg_bins: int = 80,
    burnin_frac: float = 0.7,
) -> dict:
    """
    Build P(m|T) and P(Rg|T) from post-burnin replica trajectories.

    Output dict matches the format written by temp_scan2_overlay.py so the
    same replotting code can be used on both outputs.
    """
    nT = len(replicas)
    Ts = np.array([rep.T for rep in replicas], dtype=float)

    C_arrs:  list[np.ndarray] = []
    Rg_arrs: list[np.ndarray] = []
    for rep in replicas:
        n = len(rep.C_traj)
        s = int(math.floor(n * burnin_frac))
        C_arrs.append(np.array(rep.C_traj[s:],  dtype=float))
        Rg_arrs.append(np.array(rep.Rg_traj[s:], dtype=float))

    # Global contacts range
    maxC = max(
        (int(np.nanmax(a)) for a in C_arrs if a.size > 0),
        default=0,
    )

    # Global Rg range for common bin edges
    rg_all = np.concatenate([a[np.isfinite(a)] for a in Rg_arrs if a.size > 0])
    if rg_all.size > 0:
        rg_lo, rg_hi = float(rg_all.min()), float(rg_all.max())
        pad = 0.02 * (rg_hi - rg_lo) if rg_hi > rg_lo else 1e-9
    else:
        rg_lo, rg_hi, pad = 0.0, 1.0, 0.0
    rg_edges   = np.linspace(rg_lo - pad, rg_hi + pad, rg_bins + 1)
    rg_centers = 0.5 * (rg_edges[:-1] + rg_edges[1:])

    Pc  = np.full((nT, maxC + 1), np.nan, dtype=float)
    Prg = np.full((nT, rg_bins),  np.nan, dtype=float)

    for i, (C_arr, Rg_arr) in enumerate(zip(C_arrs, Rg_arrs)):
        if C_arr.size > 0:
            c_int = np.rint(C_arr).astype(int)
            row   = np.zeros(maxC + 1, dtype=float)
            valid = (c_int >= 0) & (c_int <= maxC)
            np.add.at(row, c_int[valid], 1)
            s = row.sum()
            if s > 0:
                Pc[i] = row / s

        rg = Rg_arr[np.isfinite(Rg_arr)]
        if rg.size > 0:
            counts, _ = np.histogram(rg, bins=rg_edges)
            s = counts.sum()
            if s > 0:
                Prg[i] = counts.astype(float) / s

    return {
        "Ts":         Ts,
        "c_vals":     np.arange(maxC + 1, dtype=int),
        "Pc":         Pc,
        "rg_edges":   rg_edges,
        "rg_centers": rg_centers,
        "Prg":        Prg,
    }


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def save_results_csv(results: list[dict], out_prefix: str) -> str:
    path = f"{out_prefix}_results.csv"
    keys = ["T", "E_mean", "E_std", "C_mean", "C_std", "Rg_mean", "Rg_std", "local_acc_rate"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(keys)
        for r in results:
            w.writerow([r.get(k, math.nan) for k in keys])
    print(f"Saved {path}")
    return path


def save_swap_csv(
    swap_props: np.ndarray,
    swap_accs:  np.ndarray,
    Ts: np.ndarray,
    out_prefix: str,
) -> str:
    path = f"{out_prefix}_swap_rates.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pair", "T_lo", "T_hi", "proposals", "acceptances", "rate"])
        for k in range(len(swap_props)):
            prop = int(swap_props[k])
            acc  = int(swap_accs[k])
            rate = acc / prop if prop > 0 else float("nan")
            w.writerow([k, float(Ts[k]), float(Ts[k + 1]), prop, acc, f"{rate:.4f}"])
    print(f"Saved {path}")
    return path


def save_distributions(dist: dict, out_prefix: str) -> str:
    path = f"{out_prefix}_distributions.npz"
    np.savez_compressed(path, **dist)
    print(f"Saved {path}")
    return path


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _make_colormap(Ts: np.ndarray):
    cmap = cm.coolwarm
    norm = mcolors.Normalize(vmin=float(np.nanmin(Ts)), vmax=float(np.nanmax(Ts)))
    sm   = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    return cmap, norm, sm


def plot_observables(results: list[dict], out_prefix: str) -> None:
    Ts      = np.array([r["T"]      for r in results])
    E_means = np.array([r["E_mean"] for r in results])
    E_errs  = np.array([r["E_std"]  for r in results])
    C_means = np.array([r["C_mean"] for r in results])
    C_errs  = np.array([r["C_std"]  for r in results])
    Rg_means = np.array([r["Rg_mean"] for r in results])
    Rg_errs  = np.array([r["Rg_std"]  for r in results])

    for (y, ye, ylabel, title, tag) in [
        (E_means,  E_errs,  "Energy E (mean ± std)",              "E vs T",        "E_vs_T"),
        (C_means,  C_errs,  "Contacts m (mean ± std)",            "Contacts vs T", "contacts_vs_T"),
        (Rg_means, Rg_errs, "Radius of gyration (mean ± std)",    "Rg vs T",       "Rg_vs_T"),
    ]:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.errorbar(Ts, y, yerr=ye, marker="o", linestyle="-", capsize=3)
        ax.set_xlabel("Temperature T")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        fig.tight_layout()
        out = f"{out_prefix}_{tag}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"Saved {out}")


def plot_distributions(dist: dict, out_prefix: str) -> None:
    Ts         = dist["Ts"]
    c_vals     = dist["c_vals"]
    Pc         = dist["Pc"]
    rg_centers = dist["rg_centers"]
    Prg        = dist["Prg"]

    cmap, norm, sm = _make_colormap(Ts)

    # Side-by-side overlay
    fig, (ax_rg, ax_c) = plt.subplots(1, 2, figsize=(14, 5))
    for i, T in enumerate(Ts):
        col = cmap(norm(T))
        if np.any(np.isfinite(Prg[i])):
            ax_rg.plot(rg_centers, Prg[i], color=col, alpha=0.7, linewidth=1.0)
        if np.any(np.isfinite(Pc[i])):
            ax_c.plot(c_vals, Pc[i], color=col, alpha=0.7, linewidth=1.0)
    ax_rg.set_xlabel("Rg");      ax_rg.set_ylabel("P(Rg)");      ax_rg.set_title("P(Rg) by temperature")
    ax_c.set_xlabel("Contacts"); ax_c.set_ylabel("P(m)");         ax_c.set_title("P(m) by temperature")
    fig.colorbar(sm, ax=ax_rg, label="T")
    fig.colorbar(sm, ax=ax_c,  label="T")
    fig.tight_layout()
    out = f"{out_prefix}_distributions_overlay.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Saved {out}")

    # P(Rg) alone
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for i, T in enumerate(Ts):
        if np.any(np.isfinite(Prg[i])):
            ax.plot(rg_centers, Prg[i], color=cmap(norm(T)), alpha=0.7, linewidth=1.0)
    ax.set_xlabel("Rg"); ax.set_ylabel("P(Rg)"); ax.set_title("P(Rg) colored by T")
    fig.colorbar(sm, ax=ax, label="T")
    fig.tight_layout()
    out = f"{out_prefix}_Prg_vs_T.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Saved {out}")

    # P(m) alone
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for i, T in enumerate(Ts):
        if np.any(np.isfinite(Pc[i])):
            ax.plot(c_vals, Pc[i], color=cmap(norm(T)), alpha=0.7, linewidth=1.0)
    ax.set_xlabel("Contacts"); ax.set_ylabel("P(m)"); ax.set_title("P(m) colored by T")
    fig.colorbar(sm, ax=ax, label="T")
    fig.tight_layout()
    out = f"{out_prefix}_Pc_vs_T.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

def run_quick_test() -> None:
    """Smoke-test: serial and 2-worker runs, check outputs and normalisation."""
    import math as _math
    import os
    import tempfile

    params = dict(
        N=20,
        Ts=np.linspace(300, 360, 4),
        steps_per_swap=50,
        n_cycles=20,
        dh=378.96,
        ds=1.39686,
        seed=7,
    )

    for n_workers in (1, 2):
        with tempfile.TemporaryDirectory() as tmpdir:
            prefix = os.path.join(tmpdir, f"qt_w{n_workers}")
            reps, sp, sa = run_remd(**params, n_workers=n_workers, verbose=False)

            save_swap_csv(sp, sa, params["Ts"], prefix)
            assert os.path.exists(f"{prefix}_swap_rates.csv"), "swap rates CSV missing"

            dist = build_distributions(reps, rg_bins=40, burnin_frac=0.5)
            for i, row in enumerate(dist["Pc"]):
                finite = row[np.isfinite(row)]
                if finite.size > 0:
                    s = float(finite.sum())
                    assert abs(s - 1.0) < 1e-6, f"Pc[{i}] not normalised: sum={s}"
            for i, row in enumerate(dist["Prg"]):
                finite = row[np.isfinite(row)]
                if finite.size > 0:
                    s = float(finite.sum())
                    assert abs(s - 1.0) < 1e-6, f"Prg[{i}] not normalised: sum={s}"

            stats = compute_statistics(reps, burnin_frac=0.5)
            for r in stats:
                assert not _math.isnan(r["E_mean"]),  f"NaN E_mean at T={r['T']}"
                assert not _math.isnan(r["C_mean"]),  f"NaN C_mean at T={r['T']}"
                assert not _math.isnan(r["Rg_mean"]), f"NaN Rg_mean at T={r['T']}"

        print(f"  quick-test n_workers={n_workers}: PASSED")
    print("quick-test complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Serial REMD for lattice polymer with T-dependent Hamiltonian.",
    )
    ap.add_argument("--N",              type=int,   default=300,    help="chain length")
    ap.add_argument("--Tmin",           type=float, default=280.0,  help="lowest temperature")
    ap.add_argument("--Tmax",           type=float, default=380.0,  help="highest temperature")
    ap.add_argument("--nT",             type=int,   default=8,      help="number of replicas")
    ap.add_argument("--steps-per-swap", type=int,   default=500,    help="local MC steps per replica per swap cycle")
    ap.add_argument("--n-cycles",       type=int,   default=4000,   help="number of swap cycles")
    ap.add_argument("--dh",             type=float, default=378.96, help="contact enthalpy (dh)")
    ap.add_argument("--ds",             type=float, default=1.39686,help="contact entropy (ds)")
    ap.add_argument("--seed",           type=int,   default=42,     help="RNG seed")
    ap.add_argument("--out-prefix",     type=str,   default="remd_out", help="prefix for all output files")
    ap.add_argument("--rg-bins",        type=int,   default=80,     help="bins for P(Rg) histograms")
    ap.add_argument("--burnin-frac",    type=float, default=0.7,    help="fraction of trajectory to discard as burnin")
    ap.add_argument("--n-workers",      type=int,   default=1,      help="parallel workers for local sweeps (1 = serial)")
    ap.add_argument("--timing",         action="store_true",        help="print sweep/swap/total wall times")
    ap.add_argument("--quick-test",     action="store_true",        help="run smoke-test (N=20, nT=4, 20 cycles) and exit")
    args = ap.parse_args()

    if args.quick_test:
        run_quick_test()
        return

    Ts = np.linspace(args.Tmin, args.Tmax, args.nT)
    total_steps = args.steps_per_swap * args.n_cycles

    print(
        f"REMD: {args.nT} replicas, T in [{args.Tmin}, {args.Tmax}], "
        f"{args.n_cycles} cycles x {args.steps_per_swap} steps = {total_steps} steps/replica"
    )

    replicas, swap_props, swap_accs = run_remd(
        N=args.N, Ts=Ts,
        steps_per_swap=args.steps_per_swap,
        n_cycles=args.n_cycles,
        dh=args.dh, ds=args.ds,
        seed=args.seed, verbose=True,
        n_workers=args.n_workers,
        timing=args.timing,
    )

    print("\nSwap acceptance rates by pair:")
    for k in range(len(swap_props)):
        rate = swap_accs[k] / max(1, swap_props[k])
        print(f"  T={Ts[k]:.1f} <-> T={Ts[k+1]:.1f}  {swap_accs[k]}/{swap_props[k]} = {rate:.3f}")

    results = compute_statistics(replicas, burnin_frac=args.burnin_frac)
    dist    = build_distributions(replicas, rg_bins=args.rg_bins, burnin_frac=args.burnin_frac)

    save_results_csv(results, args.out_prefix)
    save_swap_csv(swap_props, swap_accs, Ts, args.out_prefix)
    save_distributions(dist, args.out_prefix)
    plot_observables(results, args.out_prefix)
    plot_distributions(dist, args.out_prefix)


if __name__ == "__main__":
    main()
