#!/usr/bin/env python3
"""
Self-contained replica exchange Monte Carlo for the lattice polymer model.

Hamiltonian:  H(C; T) = m(C) * (dh - T*ds)
Reduced potential:  u(C, T) = H(C, T) / T = m(C) * (dh/T - ds)

Because H depends explicitly on T, the standard REMD swap criterion must use
the reduced potential rather than a temperature-independent energy:

    log_accept = u(C_i,T_i) + u(C_j,T_j) - u(C_j,T_i) - u(C_i,T_j)

Accept if random() < exp(min(0, log_accept)).

Usage:
    python remd_uniform_chain.py --N 50 --nT 8 --Tmin 280 --Tmax 380 \\
        --steps-per-swap 500 --n-cycles 4000 --seed 42

Output NPZ keys (distributions file)
-------------------------------------
Canonical (existing):
    Ts          temperatures, shape (nT,)
    c_vals      integer contact counts [0, maxC], shape (maxC+1,)
    Pc          P(m|T) probability mass, shape (nT, maxC+1), rows sum to 1
    rg_edges    Rg histogram edges (output units = rg_scale*lattice), shape (rg_bins+1,)
    rg_centers  Rg bin centers (output units = rg_scale*lattice), shape (rg_bins,)
    Prg         P(Rg|T) probability mass, shape (nT, rg_bins), rows sum to 1

Rg scaling (rg_scale; default 1.0 = lattice units):
    rg_scale            scalar conversion factor, Rg_output = rg_scale*Rg_lattice
    rg_edges_lattice    raw lattice-unit Rg histogram edges, shape (rg_bins+1,)
    rg_centers_lattice  raw lattice-unit Rg bin centers, shape (rg_bins,)

Compatibility aliases for fit_lattice_contact_model.py:
    temps       = Ts
    ct_centers  = c_vals.astype(float)   (use --contact_offset 0 when fitting)
    ct_hists    = Pc                      (sum per row = 1; bin_width = 1)
    rg_hists    = Prg                     (sum per row = 1)
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import math
import random
import time
from concurrent.futures import ProcessPoolExecutor
from itertools import permutations, product
from pathlib import Path
from typing import List, Tuple

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Lattice polymer physics (self-contained, no external import required)
# ---------------------------------------------------------------------------

Vec = Tuple[int, int, int]
NN_VECS: List[Vec] = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]


def _add(a: Vec, b: Vec) -> Vec:
    return (a[0]+b[0], a[1]+b[1], a[2]+b[2])


def _sub(a: Vec, b: Vec) -> Vec:
    return (a[0]-b[0], a[1]-b[1], a[2]-b[2])


def _generate_cubic_rotations() -> list:
    """Return the 23 proper cubic rotations excluding the identity."""
    def _det3(M):
        (a,b,c),(d,e,f),(g,h,i) = M
        return a*(e*i-f*h) - b*(d*i-f*g) + c*(d*h-e*g)

    rots = []
    for perm in permutations([0, 1, 2]):
        for signs in product([-1, 1], repeat=3):
            M = [[0,0,0],[0,0,0],[0,0,0]]
            for r, c in enumerate(perm):
                M[r][c] = signs[r]
            Mt = (tuple(M[0]), tuple(M[1]), tuple(M[2]))
            if _det3(Mt) == 1:
                rots.append(Mt)
    I = ((1,0,0),(0,1,0),(0,0,1))
    return [M for M in rots if M != I]   # 23 matrices


ROT_MATS = _generate_cubic_rotations()


def _apply_rot(M, v: Vec) -> Vec:
    x, y, z = v
    return (
        M[0][0]*x + M[0][1]*y + M[0][2]*z,
        M[1][0]*x + M[1][1]*y + M[1][2]*z,
        M[2][0]*x + M[2][1]*y + M[2][2]*z,
    )


def energy(chain: List[Vec], occ: set, dh: float, ds: float, T: float) -> float:
    """H(C;T) = m(C)*(dh - T*ds).

    m(C) = number of unique non-bonded nearest-neighbour contacts.
    This energy depends explicitly on T; with Metropolis beta=1/T the sampled
    weight is exp(-H/T) = exp(-(dh/T - ds)*m).
    """
    cnt = 0
    N = len(chain)
    for idx, r in enumerate(chain):
        prev = chain[idx-1] if idx > 0   else None
        nxt  = chain[idx+1] if idx < N-1 else None
        for v in NN_VECS:
            nbr = _add(r, v)
            if nbr in occ and nbr not in (prev, nxt):
                cnt += 1
    return (0.5 * cnt) * (dh - T * ds)


def contact_count(chain: List[Vec], occ: set) -> float:
    """Number of unique non-bonded nearest-neighbour contacts."""
    m = 0
    N = len(chain)
    for i, r in enumerate(chain):
        prev = chain[i-1] if i > 0   else None
        nxt  = chain[i+1] if i < N-1 else None
        for v in NN_VECS:
            nbr = _add(r, v)
            if nbr in occ and nbr not in (prev, nxt):
                m += 1
    return 0.5 * m


def radius_of_gyration(chain: List[Vec]) -> float:
    r = np.array(chain, dtype=float)
    com = r.mean(axis=0)
    return math.sqrt(((r - com)**2).sum(axis=1).mean())


# --- MC move functions (must be top-level for pickling on Windows spawn) ---

def attempt_pivot(chain, occ) -> Tuple[bool, list, set]:
    """Global pivot move: rotate tail around a randomly chosen pivot monomer."""
    n = len(chain)
    i = random.randrange(1, n-1)
    head = chain[:i+1]
    tail = chain[i+1:]
    M = random.choice(ROT_MATS)
    new_occ = set(head)
    new_tail = []
    pivot = chain[i]
    for r in tail:
        r2 = _add(pivot, _apply_rot(M, _sub(r, pivot)))
        if r2 in new_occ:
            return False, chain, occ
        new_tail.append(r2)
        new_occ.add(r2)
    return True, head + new_tail, new_occ


def attempt_crankshaft(chain, occ) -> Tuple[bool, list, set]:
    """Local 90° kink flip (crankshaft move)."""
    n = len(chain)
    i = random.randrange(1, n-1)
    a, b, c = chain[i-1], chain[i], chain[i+1]
    u1 = _sub(b, a)
    u2 = _sub(c, b)
    if u1 not in NN_VECS or u2 not in NN_VECS:
        return False, chain, occ
    if u1 == u2 or u1 == (-u2[0], -u2[1], -u2[2]):
        return False, chain, occ
    if u1[0]*u2[0] + u1[1]*u2[1] + u1[2]*u2[2] != 0:
        return False, chain, occ
    b_new = _add(a, u2)
    if b_new in occ:
        return False, chain, occ
    if _sub(b_new, a) not in NN_VECS or _sub(c, b_new) not in NN_VECS:
        return False, chain, occ
    new_chain = chain.copy()
    new_chain[i] = b_new
    return True, new_chain, (occ - {b}) | {b_new}


def attempt_end_move(chain, occ) -> Tuple[bool, list, set]:
    """Symmetric end move: move one end to a random empty neighbor of its anchor."""
    n = len(chain)
    end = 0 if random.random() < 0.5 else n - 1
    anchor = 1 if end == 0 else n - 2

    old = chain[end]
    occ_without_old = occ - {old}

    v = random.choice(NN_VECS)
    r_new = _add(chain[anchor], v)

    if r_new == old:
        return False, chain, occ
    if r_new in occ_without_old:
        return False, chain, occ

    new_chain = chain.copy()
    new_chain[end] = r_new
    new_occ = occ_without_old | {r_new}
    return True, new_chain, new_occ


MOVE_FUNCS = [attempt_pivot, attempt_crankshaft, attempt_end_move]


@dataclasses.dataclass
class ChainState:
    """Mutable MC state: chain positions, occupied-site set, and current energy."""
    chain: List[Vec]
    occ:   set
    E:     float

    @classmethod
    def initial_straight(cls, N: int, dh: float, ds: float, T: float) -> "ChainState":
        chain = [(i, 0, 0) for i in range(N)]
        occ   = set(chain)
        return cls(chain=chain, occ=occ, E=energy(chain, occ, dh, ds, T))


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
# Parallel worker (top-level for pickling on Windows spawn)
# ---------------------------------------------------------------------------

def evolve_replica_worker(
    replica: "Replica",
    steps: int,
    dh: float,
    ds: float,
    seed: int,
) -> "Replica":
    """Deterministically seeded sweep of one replica."""
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
    Run REMD.

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

            start = cycle % 2
            for k in range(start, nT - 1, 2):
                swap_props[k] += 1
                if attempt_swap(replicas[k], replicas[k + 1], dh, ds):
                    swap_accs[k] += 1

            t2 = time.perf_counter()
            t_swap_total += t2 - t1

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
    rg_scale: float = 1.0,
) -> list[dict]:
    """Compute means and stds over the last (1-burnin_frac) of each trajectory.

    Rg is sampled in lattice units.  ``rg_scale`` converts it to output units:
    Rg_output = rg_scale * Rg_lattice.  Both the raw lattice values
    (Rg_mean_lattice, Rg_std_lattice) and the scaled output values
    (Rg_mean, Rg_std) are returned.  Physics is unaffected (rg_scale only
    rescales the reported Rg).
    """
    results = []
    for rep in replicas:
        n = len(rep.E_traj)
        s = int(math.floor(n * burnin_frac))

        E_arr  = np.array(rep.E_traj[s:],  dtype=float)
        C_arr  = np.array(rep.C_traj[s:],  dtype=float)
        Rg_arr = np.array(rep.Rg_traj[s:], dtype=float)

        Rg_mean_lattice = float(np.nanmean(Rg_arr))
        Rg_std_lattice  = float(np.nanstd(Rg_arr, ddof=0))

        results.append({
            "T":        rep.T,
            "E_mean":   float(np.nanmean(E_arr)),
            "E_std":    float(np.nanstd(E_arr,  ddof=0)),
            "C_mean":   float(np.nanmean(C_arr)),
            "C_std":    float(np.nanstd(C_arr,  ddof=0)),
            "Rg_mean_lattice": Rg_mean_lattice,
            "Rg_std_lattice":  Rg_std_lattice,
            "Rg_mean":  rg_scale * Rg_mean_lattice,
            "Rg_std":   rg_scale * Rg_std_lattice,
            "local_acc_rate": rep.local_acc_rate,
        })
    return results


def build_distributions(
    replicas: list[Replica],
    rg_bins: int = 80,
    burnin_frac: float = 0.7,
    rg_scale: float = 1.0,
) -> dict:
    """
    Build P(m|T) and P(Rg|T) from post-burnin replica trajectories.

    Rg is sampled and histogrammed in lattice units; ``rg_scale`` converts the
    saved Rg axis to output units (Rg_output = rg_scale * Rg_lattice).  The
    histogram is probability mass and is unchanged by scaling the axis.

    Returns a dict with the canonical keys (Ts, c_vals, Pc, rg_edges,
    rg_centers, Prg) carrying scaled/output Rg units, the raw lattice Rg grid
    (rg_edges_lattice, rg_centers_lattice), the scale metadata (rg_scale), and
    compatibility aliases for fit_lattice_contact_model.py (temps, ct_centers,
    ct_hists, rg_hists).
    """
    nT = len(replicas)
    Ts = np.array([rep.T for rep in replicas], dtype=float)

    C_arrs:        list[np.ndarray] = []
    Rg_arrs_lattice: list[np.ndarray] = []
    for rep in replicas:
        n = len(rep.C_traj)
        s = int(math.floor(n * burnin_frac))
        C_arrs.append(np.array(rep.C_traj[s:],  dtype=float))
        Rg_arrs_lattice.append(np.array(rep.Rg_traj[s:], dtype=float))

    maxC = max(
        (int(np.nanmax(a)) for a in C_arrs if a.size > 0),
        default=0,
    )

    # Build the Rg grid from the raw lattice Rg samples.
    rg_all = np.concatenate([a[np.isfinite(a)] for a in Rg_arrs_lattice if a.size > 0])
    if rg_all.size > 0:
        rg_lo, rg_hi = float(rg_all.min()), float(rg_all.max())
        pad = 0.02 * (rg_hi - rg_lo) if rg_hi > rg_lo else 1e-9
    else:
        rg_lo, rg_hi, pad = 0.0, 1.0, 0.0
    rg_edges_lattice   = np.linspace(rg_lo - pad, rg_hi + pad, rg_bins + 1)
    rg_centers_lattice = 0.5 * (rg_edges_lattice[:-1] + rg_edges_lattice[1:])

    Pc  = np.full((nT, maxC + 1), np.nan, dtype=float)
    Prg = np.full((nT, rg_bins),  np.nan, dtype=float)

    for i, (C_arr, Rg_arr) in enumerate(zip(C_arrs, Rg_arrs_lattice)):
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
            counts, _ = np.histogram(rg, bins=rg_edges_lattice)
            s = counts.sum()
            if s > 0:
                Prg[i] = counts.astype(float) / s

    # Scaled / output-unit Rg axis.  Probability mass (Prg) is unchanged.
    rg_edges_scaled   = rg_scale * rg_edges_lattice
    rg_centers_scaled = rg_scale * rg_centers_lattice

    c_vals = np.arange(maxC + 1, dtype=int)

    return {
        # Canonical keys (Rg axis in scaled/output units)
        "Ts":         Ts,
        "c_vals":     c_vals,
        "Pc":         Pc,
        "rg_edges":   rg_edges_scaled,
        "rg_centers": rg_centers_scaled,
        "Prg":        Prg,
        # Raw lattice-unit Rg grid
        "rg_edges_lattice":   rg_edges_lattice,
        "rg_centers_lattice": rg_centers_lattice,
        # Scale metadata
        "rg_scale":   float(rg_scale),
        # Aliases for fit_lattice_contact_model.py
        "temps":      Ts,
        "ct_centers": c_vals.astype(float),
        "ct_hists":   Pc,
        "rg_hists":   Prg,
    }


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def save_results_csv(results: list[dict], out_prefix: str) -> str:
    path = f"{out_prefix}_results.csv"
    keys = [
        "T", "E_mean", "E_std", "C_mean", "C_std",
        "Rg_mean", "Rg_std", "Rg_mean_lattice", "Rg_std_lattice",
        "local_acc_rate",
    ]
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
    """Save distributions NPZ with canonical keys and fitting-script aliases."""
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
        (E_means,  E_errs,  "Energy E (mean ± std)",           "E vs T",        "E_vs_T"),
        (C_means,  C_errs,  "Contacts m (mean ± std)",          "Contacts vs T", "contacts_vs_T"),
        (Rg_means, Rg_errs, "Radius of gyration Rg (output units)", "Rg vs T",       "Rg_vs_T"),
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

    fig, (ax_rg, ax_c) = plt.subplots(1, 2, figsize=(14, 5))
    for i, T in enumerate(Ts):
        col = cmap(norm(T))
        if np.any(np.isfinite(Prg[i])):
            ax_rg.plot(rg_centers, Prg[i], color=col, alpha=0.7, linewidth=1.0)
        if np.any(np.isfinite(Pc[i])):
            ax_c.plot(c_vals, Pc[i], color=col, alpha=0.7, linewidth=1.0)
    ax_rg.set_xlabel("Rg (output units)");      ax_rg.set_ylabel("P(Rg)");  ax_rg.set_title("P(Rg) by temperature")
    ax_c.set_xlabel("Contacts"); ax_c.set_ylabel("P(m)");    ax_c.set_title("P(m) by temperature")
    fig.colorbar(sm, ax=ax_rg, label="T")
    fig.colorbar(sm, ax=ax_c,  label="T")
    fig.tight_layout()
    out = f"{out_prefix}_distributions_overlay.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Saved {out}")

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for i, T in enumerate(Ts):
        if np.any(np.isfinite(Prg[i])):
            ax.plot(rg_centers, Prg[i], color=cmap(norm(T)), alpha=0.7, linewidth=1.0)
    ax.set_xlabel("Rg (output units)"); ax.set_ylabel("P(Rg)"); ax.set_title("P(Rg) colored by T")
    fig.colorbar(sm, ax=ax, label="T")
    fig.tight_layout()
    out = f"{out_prefix}_Prg_vs_T.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Saved {out}")

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
    """Smoke-test: serial and 2-worker runs; checks normalisation and outputs."""
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

            # Verify canonical keys present
            for key in ("Ts", "c_vals", "Pc", "rg_edges", "rg_centers", "Prg"):
                assert key in dist, f"Missing canonical key: {key}"

            # Verify compatibility alias keys present
            for key in ("temps", "ct_centers", "ct_hists", "rg_hists"):
                assert key in dist, f"Missing alias key: {key}"

            # Verify ct_centers == c_vals.astype(float)
            np.testing.assert_array_equal(dist["ct_centers"], dist["c_vals"].astype(float))

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

            # --- rg_scale != 1.0 mini-check ---
            dist_scaled = build_distributions(
                reps, rg_bins=40, burnin_frac=0.5, rg_scale=0.5
            )
            np.testing.assert_allclose(
                dist_scaled["rg_centers"], 0.5 * dist_scaled["rg_centers_lattice"]
            )
            np.testing.assert_allclose(
                dist_scaled["rg_edges"], 0.5 * dist_scaled["rg_edges_lattice"]
            )
            assert float(dist_scaled["rg_scale"]) == 0.5, "rg_scale metadata mismatch"
            # rg_hists rows still sum to 1 (probability mass unchanged by scaling)
            for i, row in enumerate(dist_scaled["rg_hists"]):
                finite = row[np.isfinite(row)]
                if finite.size > 0:
                    s = float(finite.sum())
                    assert abs(s - 1.0) < 1e-6, f"scaled rg_hists[{i}] not normalised: sum={s}"
            # contact aliases unchanged by rg_scale
            np.testing.assert_array_equal(dist_scaled["c_vals"], dist["c_vals"])
            np.testing.assert_array_equal(
                dist_scaled["ct_centers"], dist_scaled["c_vals"].astype(float)
            )
            for i in range(dist["Pc"].shape[0]):
                a, b = dist["Pc"][i], dist_scaled["Pc"][i]
                fa, fb = np.isfinite(a), np.isfinite(b)
                np.testing.assert_array_equal(fa, fb)
                np.testing.assert_allclose(a[fa], b[fb])

            stats = compute_statistics(reps, burnin_frac=0.5)
            for r in stats:
                assert not math.isnan(r["E_mean"]),  f"NaN E_mean at T={r['T']}"
                assert not math.isnan(r["C_mean"]),  f"NaN C_mean at T={r['T']}"
                assert not math.isnan(r["Rg_mean"]), f"NaN Rg_mean at T={r['T']}"

            # compute_statistics rg_scale consistency
            stats_scaled = compute_statistics(reps, burnin_frac=0.5, rg_scale=0.5)
            for r in stats_scaled:
                if not math.isnan(r["Rg_mean_lattice"]):
                    np.testing.assert_allclose(r["Rg_mean"], 0.5 * r["Rg_mean_lattice"])
                    np.testing.assert_allclose(r["Rg_std"], 0.5 * r["Rg_std_lattice"])

        # Sanity check: attempt_end_move accepts at least occasionally
        chain5 = [(0,0,0),(1,0,0),(2,0,0),(3,0,0),(4,0,0)]
        occ5 = set(chain5)
        n_accepted = sum(attempt_end_move(chain5, occ5)[0] for _ in range(200))
        assert n_accepted > 0, "attempt_end_move never accepted in 200 trials (bug?)"

        print(f"  quick-test n_workers={n_workers}: PASSED")
    print("quick-test complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Self-contained REMD for lattice polymer with T-dependent Hamiltonian.",
    )
    ap.add_argument("--N",              type=int,   default=44,     help="chain length")
    ap.add_argument("--Tmin",           type=float, default=230.0,   help="lowest temperature")
    ap.add_argument("--Tmax",           type=float, default=300.0,   help="highest temperature")
    ap.add_argument("--nT",             type=int,   default=64,       help="number of replicas")
    ap.add_argument("--steps-per-swap", type=int,   default=500,     help="local MC steps per replica per swap cycle")
    ap.add_argument("--n-cycles",       type=int,   default=4000,    help="number of swap cycles")
    ap.add_argument("--dh",             type=float, default=312.109,  help="contact enthalpy dh")
    ap.add_argument("--ds",             type=float, default=1.23278, help="contact entropy ds")
    ap.add_argument("--seed",           type=int,   default=42,      help="RNG seed")
    ap.add_argument("--out-prefix",     type=str,   default="remd_out", help="prefix for all output files")
    ap.add_argument("--rg-bins",        type=int,   default=64,      help="bins for P(Rg) histograms")
    ap.add_argument("--burnin-frac",    type=float, default=0.7,     help="fraction of trajectory to discard as burnin")
    ap.add_argument("--n-workers",      type=int,   default=4,       help="parallel workers for local sweeps (1 = serial)")
    ap.add_argument(
        "--rg-scale", type=float, default=1.0,
        help=(
            "Scale factor for reporting/outputting Rg. "
            "Rg_output = rg_scale * Rg_lattice. Default 1.0 preserves lattice units."
        ),
    )
    ap.add_argument("--timing",         action="store_true",         help="print sweep/swap/total wall times")
    ap.add_argument("--quick-test",     action="store_true",         help="run smoke-test and exit")
    args = ap.parse_args()

    if args.N < 3:
        raise ValueError("--N must be >= 3")
    if args.nT < 2:
        raise ValueError("--nT must be >= 2")
    if args.steps_per_swap < 1:
        raise ValueError("--steps-per-swap must be >= 1")
    if args.n_cycles < 1:
        raise ValueError("--n-cycles must be >= 1")
    if args.rg_bins < 1:
        raise ValueError("--rg-bins must be >= 1")
    if args.n_workers < 1:
        raise ValueError("--n-workers must be >= 1")
    if not (0.0 <= args.burnin_frac < 1.0):
        raise ValueError("--burnin-frac must be in [0, 1)")
    if args.Tmin <= 0 or args.Tmax <= 0:
        raise ValueError("Temperatures must be positive")
    if args.Tmax <= args.Tmin:
        raise ValueError("--Tmax must be greater than --Tmin")
    if args.rg_scale <= 0:
        raise ValueError("--rg-scale must be positive")

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

    results = compute_statistics(
        replicas, burnin_frac=args.burnin_frac, rg_scale=args.rg_scale
    )
    dist    = build_distributions(
        replicas, rg_bins=args.rg_bins, burnin_frac=args.burnin_frac,
        rg_scale=args.rg_scale,
    )

    save_results_csv(results, args.out_prefix)
    save_swap_csv(swap_props, swap_accs, Ts, args.out_prefix)
    save_distributions(dist, args.out_prefix)
    plot_observables(results, args.out_prefix)
    plot_distributions(dist, args.out_prefix)


if __name__ == "__main__":
    main()
