#!/usr/bin/env python3
"""
validate_remd.py — Diagnostic validation for remd_uniform_chain.py.

Three simulations are run and compared:

  nswap  — REMD loop with swaps disabled: each replica evolves independently
            at its fixed temperature, sampling the canonical ensemble directly.
  remd   — REMD with swaps enabled (even/odd alternating pairs).
  singT  — Independent run_single_temperature calls at each T (scalar stats only).

Because nswap is mathematically identical to nT independent single-temperature
chains, P(m|T) and P(Rg|T) from nswap are used as the canonical reference when
comparing distribution shapes.  singT scalars (E_mean, C_mean, Rg_mean) provide
an additional independent check of the means.

Plots produced (all prefixed with --out-prefix):
  _Pm_nswap_vs_remd.png   P(m|T): no-swap vs REMD, side-by-side panels
  _PRg_nswap_vs_remd.png  P(Rg|T): same layout
  _means.png              E, C, Rg means vs T — nswap / remd / singT
  _swap_rates.png         Swap acceptance rate per adjacent pair
  _walker_traj.png        Temperature slot of each walker vs REMD cycle
  _round_trips.png        Extreme-to-extreme transitions per walker

Seeds:
  nswap  seed
  remd   seed + 10 000
  singT  seed + 20 000 + i  (i = replica index)
"""
from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from thermo_uniform_chain2_DBfix_dists_corrected import (
    ChainState,
    contact_count,
    radius_of_gyration,
    run_single_temperature,
)
from remd_uniform_chain import (
    Replica,
    attempt_swap,
    build_distributions,
    compute_statistics,
    mc_sweep,
)


# ---------------------------------------------------------------------------
# Simulation runners
# ---------------------------------------------------------------------------

def run_nswap(
    N: int,
    Ts: np.ndarray,
    steps_per_swap: int,
    n_cycles: int,
    dh: float,
    ds: float,
    seed: int | None = None,
    verbose: bool = False,
) -> list[Replica]:
    """
    Independent canonical MC at each T — no swap attempts.

    This is the no-swap control.  Each replica is statistically equivalent
    to an independent run_single_temperature call with the same total steps.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    replicas = [
        Replica(T=float(T), state=ChainState.initial_straight(N, dh, ds, float(T)))
        for T in Ts
    ]

    report_every = max(1, n_cycles // 10)
    for cycle in range(n_cycles):
        for rep in replicas:
            mc_sweep(rep, steps_per_swap, dh, ds)
            rep.E_traj.append(rep.state.E)
            rep.C_traj.append(contact_count(rep.state.chain, rep.state.occ))
            rep.Rg_traj.append(radius_of_gyration(rep.state.chain))

        if verbose and (cycle + 1) % report_every == 0:
            print(f"  [nswap] cycle {cycle + 1}/{n_cycles}")

    return replicas


def run_remd_tracked(
    N: int,
    Ts: np.ndarray,
    steps_per_swap: int,
    n_cycles: int,
    dh: float,
    ds: float,
    seed: int | None = None,
    verbose: bool = True,
) -> tuple[list[Replica], np.ndarray, np.ndarray, np.ndarray]:
    """
    REMD with swaps, tracking which walker (initial configuration) is in
    which temperature slot at every cycle.

    Returns
    -------
    replicas       : Replica list with full trajectories
    swap_props     : (nT-1,) proposals per adjacent pair
    swap_accs      : (nT-1,) acceptances per adjacent pair
    slot_of_walker : (n_cycles, nT) — slot index of walker w at cycle c
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    nT = len(Ts)
    replicas = [
        Replica(T=float(T), state=ChainState.initial_straight(N, dh, ds, float(T)))
        for T in Ts
    ]

    swap_props = np.zeros(nT - 1, dtype=int)
    swap_accs  = np.zeros(nT - 1, dtype=int)

    # walker_in_slot[slot] = walker id currently occupying that slot
    walker_in_slot = list(range(nT))
    # slot_of[walker] = slot currently occupied by that walker
    slot_of = list(range(nT))

    slot_of_walker = np.zeros((n_cycles, nT), dtype=int)
    report_every   = max(1, n_cycles // 20)

    for cycle in range(n_cycles):
        for rep in replicas:
            mc_sweep(rep, steps_per_swap, dh, ds)

        start = cycle % 2
        for k in range(start, nT - 1, 2):
            swap_props[k] += 1
            if attempt_swap(replicas[k], replicas[k + 1], dh, ds):
                swap_accs[k] += 1
                w_k  = walker_in_slot[k]
                w_k1 = walker_in_slot[k + 1]
                walker_in_slot[k], walker_in_slot[k + 1] = w_k1, w_k
                slot_of[w_k], slot_of[w_k1] = k + 1, k

        for rep in replicas:
            rep.E_traj.append(rep.state.E)
            rep.C_traj.append(contact_count(rep.state.chain, rep.state.occ))
            rep.Rg_traj.append(radius_of_gyration(rep.state.chain))

        for w in range(nT):
            slot_of_walker[cycle, w] = slot_of[w]

        if verbose and (cycle + 1) % report_every == 0:
            rates = " ".join(
                f"{swap_accs[k] / max(1, swap_props[k]):.2f}"
                for k in range(nT - 1)
            )
            print(f"  [remd]  cycle {cycle + 1:>6}/{n_cycles}  swap rates: {rates}")

    return replicas, swap_props, swap_accs, slot_of_walker


def run_single_T_all(
    N: int,
    Ts: np.ndarray,
    total_steps: int,
    dh: float,
    ds: float,
    base_seed: int | None,
    dist_dir: str = "validate_singT_dists",
    verbose: bool = False,
) -> list[dict]:
    """
    Independent run_single_temperature at each T.
    Returns a list of result dicts (scalars only — E_mean, C_mean, Rg_mean, …).
    """
    results = []
    for i, T in enumerate(Ts):
        seed = None if base_seed is None else base_seed + i
        if verbose:
            print(f"  [singT] T={T:.2f}  seed={seed}")
        r = run_single_temperature(
            N=N, steps=total_steps, T=float(T),
            dh=dh, ds=ds, seed=seed,
            dist_dir=dist_dir,
            rg_bins=80,
        )
        # Normalise key names to match compute_statistics output
        results.append({
            "T":       float(T),
            "E_mean":  r["E_mean"],  "E_std":  r["E_std"],
            "C_mean":  r["C_mean"],  "C_std":  r["C_std"],
            "Rg_mean": r["Rg_back_mean"], "Rg_std": r["Rg_back_std"],
        })
    return results


# ---------------------------------------------------------------------------
# Walker round-trip analysis
# ---------------------------------------------------------------------------

def count_extreme_transitions(slot_of_walker: np.ndarray) -> np.ndarray:
    """
    Count extreme-to-extreme temperature transitions per walker.

    A transition is counted each time a walker moves from the lowest slot
    (index 0) to the highest slot (index nT-1), or vice versa.  Intermediate
    slots are ignored.  One full round trip = 2 transitions.

    Parameters
    ----------
    slot_of_walker : (n_cycles, nT) array

    Returns
    -------
    transitions : (nT,) array of int — transition counts per walker
    """
    n_cycles, nT = slot_of_walker.shape
    lo, hi = 0, nT - 1
    transitions = np.zeros(nT, dtype=int)
    last_extreme = np.full(nT, -1, dtype=int)   # -1 = no extreme visited yet

    for cycle in range(n_cycles):
        for w in range(nT):
            slot = int(slot_of_walker[cycle, w])
            if slot == lo:
                if last_extreme[w] == hi:
                    transitions[w] += 1
                last_extreme[w] = lo
            elif slot == hi:
                if last_extreme[w] == lo:
                    transitions[w] += 1
                last_extreme[w] = hi

    return transitions


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _colormap(Ts: np.ndarray):
    cmap = cm.coolwarm
    norm = mcolors.Normalize(vmin=float(Ts.min()), vmax=float(Ts.max()))
    sm   = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    return cmap, norm, sm


def _label_Ts(Ts: np.ndarray) -> list[str]:
    return [f"{T:.1f}" for T in Ts]


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_Pm_side_by_side(
    dist_a: dict, label_a: str,
    dist_b: dict, label_b: str,
    out_path: str,
) -> None:
    """Side-by-side P(m|T) panels for two runs, colored by temperature."""
    Ts = dist_a["Ts"]
    cmap, norm, sm = _colormap(Ts)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    for ax, dist, label in zip(axes, [dist_a, dist_b], [label_a, label_b]):
        for i, T in enumerate(Ts):
            col = cmap(norm(T))
            y = dist["Pc"][i]
            if np.any(np.isfinite(y)):
                ax.plot(dist["c_vals"], y, color=col, alpha=0.7, lw=1.2)
        ax.set_xlabel("Contacts m")
        ax.set_ylabel("P(m)")
        ax.set_title(f"P(m|T) — {label}")
        fig.colorbar(sm, ax=ax, label="T")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_PRg_side_by_side(
    dist_a: dict, label_a: str,
    dist_b: dict, label_b: str,
    out_path: str,
) -> None:
    """Side-by-side P(Rg|T) panels for two runs, colored by temperature."""
    Ts = dist_a["Ts"]
    cmap, norm, sm = _colormap(Ts)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    for ax, dist, label in zip(axes, [dist_a, dist_b], [label_a, label_b]):
        for i, T in enumerate(Ts):
            col = cmap(norm(T))
            y = dist["Prg"][i]
            if np.any(np.isfinite(y)):
                ax.plot(dist["rg_centers"], y, color=col, alpha=0.7, lw=1.2)
        ax.set_xlabel("Rg")
        ax.set_ylabel("P(Rg)")
        ax.set_title(f"P(Rg|T) — {label}")
        fig.colorbar(sm, ax=ax, label="T")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_means(
    stats_nswap: list[dict],
    stats_remd:  list[dict],
    stats_singT: list[dict],
    out_path: str,
) -> None:
    """E_mean, C_mean, Rg_mean vs T for all three methods."""
    def _arr(stats, key):
        return np.array([r[key] for r in stats])

    Ts = _arr(stats_nswap, "T")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    for ax, ykey, errkey, ylabel in [
        (axes[0], "E_mean",  "E_std",  "Energy E"),
        (axes[1], "C_mean",  "C_std",  "Contacts m"),
        (axes[2], "Rg_mean", "Rg_std", "Radius of gyration"),
    ]:
        for stats, label, marker, ls in [
            (stats_nswap, "no-swap", "o", "-"),
            (stats_remd,  "REMD",    "s", "--"),
            (stats_singT, "singT",   "^", ":"),
        ]:
            y   = _arr(stats, ykey)
            err = _arr(stats, errkey)
            ax.errorbar(Ts, y, yerr=err, label=label,
                        marker=marker, linestyle=ls, capsize=3)
        ax.set_xlabel("T")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} vs T")
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_swap_rates(
    swap_props: np.ndarray,
    swap_accs:  np.ndarray,
    Ts: np.ndarray,
    out_path: str,
) -> None:
    """Horizontal bar chart of swap acceptance rates by adjacent pair."""
    nT   = len(Ts)
    npairs = nT - 1
    rates  = np.array([
        swap_accs[k] / max(1, swap_props[k]) for k in range(npairs)
    ])
    pair_labels = [f"T={Ts[k]:.1f} / T={Ts[k+1]:.1f}" for k in range(npairs)]

    fig, ax = plt.subplots(figsize=(7, max(3, npairs * 0.5 + 1)))
    y_pos = np.arange(npairs)
    bars = ax.barh(y_pos, rates, color="steelblue", alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(pair_labels, fontsize=8)
    ax.set_xlabel("Acceptance rate")
    ax.set_title("Swap acceptance rates by pair")
    ax.set_xlim(0, 1)
    ax.axvline(0.2, color="red",    linestyle="--", lw=1, alpha=0.6, label="0.2")
    ax.axvline(0.4, color="orange", linestyle="--", lw=1, alpha=0.6, label="0.4")
    ax.legend(fontsize=8, title="reference")
    for bar, rate in zip(bars, rates):
        ax.text(min(rate + 0.01, 0.95), bar.get_y() + bar.get_height() / 2,
                f"{rate:.2f}", va="center", ha="left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_walker_trajectories(
    slot_of_walker: np.ndarray,
    Ts: np.ndarray,
    burnin_frac: float,
    out_path: str,
) -> None:
    """
    Temperature of each walker vs REMD cycle.

    x-axis  : cycle index
    y-axis  : temperature (via Ts[slot])
    one line per walker (colored by initial walker index = initial temperature)
    A vertical dashed line marks the end of burnin.
    """
    n_cycles, nT = slot_of_walker.shape
    cmap = cm.tab10 if nT <= 10 else cm.nipy_spectral
    burnin_end = int(math.floor(n_cycles * burnin_frac))

    fig, ax = plt.subplots(figsize=(10, 4))
    cycles = np.arange(n_cycles)

    for w in range(nT):
        slots = slot_of_walker[:, w]
        temps = Ts[slots]
        color = cmap(w / max(1, nT - 1))
        ax.plot(cycles, temps, color=color, alpha=0.6, lw=0.8,
                label=f"w{w} (T0={Ts[w]:.0f})")

    ax.axvline(burnin_end, color="black", linestyle="--", lw=1.2, label="burnin end")
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Temperature")
    ax.set_title("Walker temperature trajectories (REMD)")
    ax.legend(fontsize=7, ncol=max(1, nT // 4), loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_round_trips(
    transitions: np.ndarray,
    Ts: np.ndarray,
    out_path: str,
) -> None:
    """
    Bar chart of extreme-to-extreme transitions per walker.
    Walker label = initial temperature.
    Note: 2 transitions = 1 complete round trip.
    """
    nT = len(Ts)
    labels = [f"{T:.1f}" for T in Ts]
    x = np.arange(nT)

    fig, ax = plt.subplots(figsize=(max(5, nT * 0.8), 4))
    ax.bar(x, transitions, color="steelblue", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Walker initial temperature")
    ax.set_ylabel("Extreme-to-extreme transitions")
    total = int(transitions.sum())
    ax.set_title(
        f"Replica round trips  (total transitions: {total},  "
        f"full round trips ~ {total // 2})"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Validation diagnostics comparing REMD to no-swap and single-T runs.",
    )
    ap.add_argument("--N",              type=int,   default=50,     help="chain length")
    ap.add_argument("--Tmin",           type=float, default=280.0)
    ap.add_argument("--Tmax",           type=float, default=380.0)
    ap.add_argument("--nT",             type=int,   default=6,      help="number of replicas")
    ap.add_argument("--steps-per-swap", type=int,   default=300,    help="local MC steps per cycle")
    ap.add_argument("--n-cycles",       type=int,   default=1000,   help="swap cycles")
    ap.add_argument("--dh",             type=float, default=378.96)
    ap.add_argument("--ds",             type=float, default=1.39686)
    ap.add_argument("--seed",           type=int,   default=42)
    ap.add_argument("--out-prefix",     type=str,   default="validate")
    ap.add_argument("--rg-bins",        type=int,   default=80)
    ap.add_argument("--burnin-frac",    type=float, default=0.7)
    ap.add_argument("--skip-singT",     action="store_true",
                    help="skip run_single_temperature calls (faster, omits means comparison)")
    args = ap.parse_args()

    Ts          = np.linspace(args.Tmin, args.Tmax, args.nT)
    total_steps = args.steps_per_swap * args.n_cycles
    seed_nswap  = args.seed
    seed_remd   = args.seed + 10_000
    seed_singT  = args.seed + 20_000

    print(
        f"Validation: N={args.N}, nT={args.nT}, T=[{args.Tmin},{args.Tmax}], "
        f"{args.n_cycles} cycles x {args.steps_per_swap} steps = {total_steps} steps/replica"
    )

    # ── 1. No-swap run ───────────────────────────────────────────────────────
    print(f"\n[1/3] No-swap run (seed={seed_nswap}) ...")
    replicas_nswap = run_nswap(
        N=args.N, Ts=Ts,
        steps_per_swap=args.steps_per_swap, n_cycles=args.n_cycles,
        dh=args.dh, ds=args.ds, seed=seed_nswap, verbose=True,
    )
    dist_nswap  = build_distributions(replicas_nswap, rg_bins=args.rg_bins, burnin_frac=args.burnin_frac)
    stats_nswap = compute_statistics(replicas_nswap, burnin_frac=args.burnin_frac)

    # ── 2. Full REMD run with walker tracking ────────────────────────────────
    print(f"\n[2/3] REMD run (seed={seed_remd}) ...")
    replicas_remd, swap_props, swap_accs, slot_of_walker = run_remd_tracked(
        N=args.N, Ts=Ts,
        steps_per_swap=args.steps_per_swap, n_cycles=args.n_cycles,
        dh=args.dh, ds=args.ds, seed=seed_remd, verbose=True,
    )
    dist_remd  = build_distributions(replicas_remd, rg_bins=args.rg_bins, burnin_frac=args.burnin_frac)
    stats_remd = compute_statistics(replicas_remd, burnin_frac=args.burnin_frac)

    # ── 3. Independent single-T runs ─────────────────────────────────────────
    if args.skip_singT:
        print("\n[3/3] Skipping single-T runs (--skip-singT).")
        stats_singT = [
            {"T": float(T), "E_mean": math.nan, "E_std": 0.0,
             "C_mean": math.nan, "C_std": 0.0,
             "Rg_mean": math.nan, "Rg_std": 0.0}
            for T in Ts
        ]
    else:
        singT_dist_dir = f"{args.out_prefix}_singT_dists"
        print(f"\n[3/3] Single-T runs (base_seed={seed_singT}, dist_dir={singT_dist_dir!r}) ...")
        stats_singT = run_single_T_all(
            N=args.N, Ts=Ts, total_steps=total_steps,
            dh=args.dh, ds=args.ds, base_seed=seed_singT,
            dist_dir=singT_dist_dir, verbose=True,
        )

    # ── Swap diagnostics ─────────────────────────────────────────────────────
    print("\nSwap acceptance rates:")
    for k in range(len(swap_props)):
        rate = swap_accs[k] / max(1, swap_props[k])
        print(f"  T={Ts[k]:.1f} <-> T={Ts[k+1]:.1f}  {swap_accs[k]}/{swap_props[k]} = {rate:.3f}")

    transitions = count_extreme_transitions(slot_of_walker)
    print(f"\nExtreme-to-extreme transitions per walker:")
    for w, (T, n) in enumerate(zip(Ts, transitions)):
        print(f"  walker {w} (T0={T:.1f}): {n} transitions (~{n // 2} full round trips)")

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\nGenerating plots ...")
    pfx = args.out_prefix

    plot_Pm_side_by_side(
        dist_nswap, "no-swap",
        dist_remd,  "REMD",
        out_path=f"{pfx}_Pm_nswap_vs_remd.png",
    )
    plot_PRg_side_by_side(
        dist_nswap, "no-swap",
        dist_remd,  "REMD",
        out_path=f"{pfx}_PRg_nswap_vs_remd.png",
    )
    plot_means(
        stats_nswap, stats_remd, stats_singT,
        out_path=f"{pfx}_means.png",
    )
    plot_swap_rates(
        swap_props, swap_accs, Ts,
        out_path=f"{pfx}_swap_rates.png",
    )
    plot_walker_trajectories(
        slot_of_walker, Ts, burnin_frac=args.burnin_frac,
        out_path=f"{pfx}_walker_traj.png",
    )
    plot_round_trips(
        transitions, Ts,
        out_path=f"{pfx}_round_trips.png",
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
