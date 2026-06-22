#!/usr/bin/env python3
"""
Self-contained replica exchange Monte Carlo for the lattice polymer model.

Sampling rule:  P(C|T) ∝ exp[-u(C,T)],  u(C,T) = m(C) * b(T)
Implied energy: H(C;T) = T * u(C,T) = m(C) * T * b(T)

where m(C) is the non-bonded contact count and b(T) is a model-specific
reduced contact bias.  The supported b(T) models mirror
fit_lattice_contact_model.py: hs, tc_scale, hs_quadratic, poly2, poly3,
heat_capacity (see MODEL_REGISTRY).  The default model is hs:

    b(T) = h/T - s

Loading fitted models
---------------------
The preferred way to reproduce a fitted contact-energy model is
--fit-summary-json, which loads the model name, parameters, Tref, Tscale (and
the effective heat_capacity T0) directly from fit_summary.json.  Alternatively
--fit-params-csv loads fit_params.csv and, when a companion fit_summary.json is
present in the same directory, automatically infers and validates the model,
Tref, Tscale, and heat_capacity T0 from it.

For backward compatibility, when --model hs is used without --params /
--fit-params-csv, the legacy --dh/--ds flags supply (h, s) = (dh, ds), so
H(C;T) = m(C)*(dh - T*ds) exactly as before.

Because u depends explicitly on T, the REMD swap criterion uses the reduced
potential rather than a temperature-independent energy:

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

Model metadata:
    model_name          name of the b(T) model used for sampling
    param_names         parameter names for the model, shape (n_params,)
    model_params        parameter values, shape (n_params,)
    Tref, Tscale        reference/scale for x=(T-Tref)/Tscale (polynomial models)

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
import json
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


# ---------------------------------------------------------------------------
# Reduced-bias model registry (self-contained; mirrors fit_lattice_contact_model.py)
# ---------------------------------------------------------------------------
# Each entry defines b(T) = the reduced contact bias.
# Sampling weight:  P(C|T) ∝ exp[-u(C,T)],  u(C,T) = m(C) * b(T)

def _b_hs(params, T, Tref, Tscale):
    return float(params[0]) / T - float(params[1])


def _b_tc_scale(params, T, Tref, Tscale):
    return float(params[0]) * (float(params[1]) / T - 1.0)


def _b_hs_quadratic(params, T, Tref, Tscale):
    x = (T - Tref) / Tscale
    return float(params[0]) / T - float(params[1]) + float(params[2]) * x * x


def _b_poly2(params, T, Tref, Tscale):
    x = (T - Tref) / Tscale
    return float(params[0]) + float(params[1]) * x + float(params[2]) * x * x


def _b_poly3(params, T, Tref, Tscale):
    x = (T - Tref) / Tscale
    return (
        float(params[0])
        + float(params[1]) * x
        + float(params[2]) * x * x
        + float(params[3]) * x * x * x
    )


def _b_heat_capacity(params, T, Tref, Tscale):
    T0 = Tref
    dh0, ds0, dCp = float(params[0]), float(params[1]), float(params[2])
    dg = dh0 - T * ds0 + dCp * ((T - T0) - T * math.log(T / T0))
    return dg / T


MODEL_REGISTRY = {
    "hs": {
        "param_names": ["h", "s"],
        "raw_b_fn": _b_hs,
        "description": "b(T) = h/T - s",
    },
    "tc_scale": {
        "param_names": ["A", "Tc"],
        "raw_b_fn": _b_tc_scale,
        "description": "b(T) = A*(Tc/T - 1)",
    },
    "hs_quadratic": {
        "param_names": ["h", "s", "a2"],
        "raw_b_fn": _b_hs_quadratic,
        "description": "b(T) = h/T - s + a2*x(T)^2",
    },
    "poly2": {
        "param_names": ["a0", "a1", "a2"],
        "raw_b_fn": _b_poly2,
        "description": "b(T) = a0 + a1*x + a2*x^2",
    },
    "poly3": {
        "param_names": ["a0", "a1", "a2", "a3"],
        "raw_b_fn": _b_poly3,
        "description": "b(T) = a0 + a1*x + a2*x^2 + a3*x^3",
    },
    "heat_capacity": {
        "param_names": ["dh0", "ds0", "dCp"],
        "raw_b_fn": _b_heat_capacity,
        "description": "b(T) = [dh0 - T*ds0 + dCp*((T-T0) - T*ln(T/T0))] / T",
    },
}


def reduced_bias(model_name, params, T, Tref, Tscale) -> float:
    """b(T) for the selected model."""
    return float(MODEL_REGISTRY[model_name]["raw_b_fn"](params, float(T), Tref, Tscale))


def reduced_potential(m, T, model_name, params, Tref, Tscale) -> float:
    """u(C,T) = m(C) * b(T)."""
    return float(m) * reduced_bias(model_name, params, float(T), Tref, Tscale)


def energy_from_contacts(m, T, model_name, params, Tref, Tscale) -> float:
    """Model-implied temperature-dependent energy H = T * u.

    For hs this is exactly H = m*(h - T*s), preserving old behavior.
    For other models it is the effective H corresponding to u = m*b(T).
    """
    return float(T) * reduced_potential(m, T, model_name, params, Tref, Tscale)


def energy(
    chain: List[Vec],
    occ: set,
    T: float,
    model_name: str,
    params,
    Tref: float,
    Tscale: float,
) -> float:
    """H(C;T) = m(C) * T * b(T), generic over the contact-bias model.

    m(C) = number of unique non-bonded nearest-neighbour contacts.  The sampled
    weight is exp(-u) = exp(-m*b(T)).  For model hs with params (h, s) this
    reduces to H = m*(h - T*s), i.e. the legacy --dh/--ds behavior.
    """
    m = contact_count(chain, occ)
    return energy_from_contacts(m, T, model_name, params, Tref, Tscale)


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
    def initial_straight(
        cls, N: int, T: float, model_name: str, params, Tref: float, Tscale: float
    ) -> "ChainState":
        chain = [(i, 0, 0) for i in range(N)]
        occ   = set(chain)
        return cls(
            chain=chain, occ=occ,
            E=energy(chain, occ, T, model_name, params, Tref, Tscale),
        )


# ---------------------------------------------------------------------------
# Swap criterion (reduced_potential/reduced_bias defined with the model registry)
# ---------------------------------------------------------------------------

def swap_log_accept(
    m_i: float, m_j: float,
    T_i: float, T_j: float,
    model_name: str, params,
    Tref: float, Tscale: float,
) -> float:
    """
    Log Metropolis ratio for swapping configs C_i (at T_i) and C_j (at T_j).

    log_accept = u(C_i,T_i) + u(C_j,T_j) - u(C_j,T_i) - u(C_i,T_j)
    """
    return (
        reduced_potential(m_i, T_i, model_name, params, Tref, Tscale)
        + reduced_potential(m_j, T_j, model_name, params, Tref, Tscale)
        - reduced_potential(m_j, T_i, model_name, params, Tref, Tscale)
        - reduced_potential(m_i, T_j, model_name, params, Tref, Tscale)
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

def mc_sweep(
    replica: Replica, steps: int,
    model_name: str, params, Tref: float, Tscale: float,
) -> None:
    """Run `steps` local Metropolis moves on `replica` in-place.

    Acceptance uses the generic reduced potential u = m*b(T):
        du = u_new - u_old,  accept if du <= 0 or rand < exp(-du).
    For model hs with params (h, s) this is identical to the legacy
    beta*dE criterion (du = beta*dE), preserving old sampling exactly.
    """
    T     = replica.T
    state = replica.state

    for _ in range(steps):
        replica.local_prop += 1
        move = random.choice(MOVE_FUNCS)
        ok, chain_new, occ_new = move(state.chain, state.occ)
        if not ok:
            continue
        m_old = contact_count(state.chain, state.occ)
        u_old = reduced_potential(m_old, T, model_name, params, Tref, Tscale)

        m_new = contact_count(chain_new, occ_new)
        u_new = reduced_potential(m_new, T, model_name, params, Tref, Tscale)

        du = u_new - u_old
        if du <= 0 or random.random() < math.exp(-du):
            state.chain = chain_new
            state.occ   = occ_new
            state.E     = energy_from_contacts(m_new, T, model_name, params, Tref, Tscale)
            replica.local_acc += 1


def attempt_swap(
    rep_a: Replica, rep_b: Replica,
    model_name: str, params, Tref: float, Tscale: float,
) -> bool:
    """
    Attempt a configuration swap between two adjacent replicas.

    Temperatures stay fixed; configurations (and energies) are exchanged.
    Returns True if accepted.
    """
    m_a = contact_count(rep_a.state.chain, rep_a.state.occ)
    m_b = contact_count(rep_b.state.chain, rep_b.state.occ)
    log_acc = swap_log_accept(
        m_a, m_b, rep_a.T, rep_b.T, model_name, params, Tref, Tscale
    )

    accepted = log_acc >= 0 or random.random() < math.exp(log_acc)
    if accepted:
        rep_a.state.chain, rep_b.state.chain = rep_b.state.chain, rep_a.state.chain
        rep_a.state.occ,   rep_b.state.occ   = rep_b.state.occ,   rep_a.state.occ
        rep_a.state.E = energy(
            rep_a.state.chain, rep_a.state.occ, rep_a.T, model_name, params, Tref, Tscale
        )
        rep_b.state.E = energy(
            rep_b.state.chain, rep_b.state.occ, rep_b.T, model_name, params, Tref, Tscale
        )
    return accepted


# ---------------------------------------------------------------------------
# Parallel worker (top-level for pickling on Windows spawn)
# ---------------------------------------------------------------------------

def evolve_replica_worker(
    replica: "Replica",
    steps: int,
    model_name: str,
    params: list,
    Tref: float,
    Tscale: float,
    seed: int,
) -> "Replica":
    """Deterministically seeded sweep of one replica."""
    random.seed(seed)
    np.random.seed(seed)
    mc_sweep(replica, steps, model_name, params, Tref, Tscale)
    return replica


# ---------------------------------------------------------------------------
# Main REMD loop
# ---------------------------------------------------------------------------

def run_remd(
    N: int,
    Ts: np.ndarray,
    steps_per_swap: int,
    n_cycles: int,
    model_name: str,
    params: list,
    Tref: float,
    Tscale: float,
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
        Replica(
            T=float(T),
            state=ChainState.initial_straight(
                N, float(T), model_name, params, Tref, Tscale
            ),
        )
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
                        steps_per_swap, model_name, params, Tref, Tscale,
                        worker_seeds[k],
                    )
                    for k in range(nT)
                ]
                for k, fut in enumerate(futures):
                    replicas[k] = fut.result()
            else:
                for rep in replicas:
                    mc_sweep(rep, steps_per_swap, model_name, params, Tref, Tscale)

            t1 = time.perf_counter()
            t_sweep_total += t1 - t0

            start = cycle % 2
            for k in range(start, nT - 1, 2):
                swap_props[k] += 1
                if attempt_swap(
                    replicas[k], replicas[k + 1], model_name, params, Tref, Tscale
                ):
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

    # --- hs model via legacy dh/ds (params = [h, s]) ---
    Ts_hs = np.linspace(300, 360, 4)
    hs_params = [378.96, 1.39686]
    Tref_hs = 0.5 * (float(Ts_hs.min()) + float(Ts_hs.max()))
    Tscale_hs = max(float(Ts_hs.max()) - float(Ts_hs.min()), 1.0)

    for n_workers in (1, 2):
        with tempfile.TemporaryDirectory() as tmpdir:
            prefix = os.path.join(tmpdir, f"qt_w{n_workers}")
            reps, sp, sa = run_remd(
                N=20, Ts=Ts_hs, steps_per_swap=50, n_cycles=20,
                model_name="hs", params=hs_params, Tref=Tref_hs, Tscale=Tscale_hs,
                seed=7, n_workers=n_workers, verbose=False,
            )

            save_swap_csv(sp, sa, Ts_hs, prefix)
            assert os.path.exists(f"{prefix}_swap_rates.csv"), "swap rates CSV missing"

            dist = build_distributions(reps, rg_bins=40, burnin_frac=0.5)
            attach_model_metadata(dist, "hs", ["h", "s"], hs_params, Tref_hs, Tscale_hs)

            # Verify model metadata keys
            for key in ("model_name", "param_names", "model_params", "Tref", "Tscale"):
                assert key in dist, f"Missing model metadata key: {key}"
            assert str(dist["model_name"]) == "hs"
            np.testing.assert_array_equal(dist["param_names"], np.array(["h", "s"]))
            np.testing.assert_allclose(dist["model_params"], np.array(hs_params))

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

    # --- tiny serial REMD run for every supported model ---
    def _check_model_run(model_name, model_params, Ts, Tref, Tscale, seed):
        param_names = MODEL_REGISTRY[model_name]["param_names"]

        # b(T) and the swap criterion must stay finite across the ladder.
        for T in Ts:
            assert math.isfinite(
                reduced_bias(model_name, model_params, float(T), Tref, Tscale)
            ), f"[{model_name}] non-finite reduced_bias at T={T}"
        for (Ti, Tj) in zip(Ts[:-1], Ts[1:]):
            la = swap_log_accept(
                3.0, 7.0, float(Ti), float(Tj),
                model_name, model_params, Tref, Tscale,
            )
            assert math.isfinite(la), f"[{model_name}] non-finite swap_log_accept"

        reps, _sp, _sa = run_remd(
            N=20, Ts=Ts, steps_per_swap=30, n_cycles=15,
            model_name=model_name, params=model_params, Tref=Tref, Tscale=Tscale,
            seed=seed, n_workers=1, verbose=False,
        )

        # Every recorded trajectory value must be finite.
        for rep in reps:
            assert all(math.isfinite(e) for e in rep.E_traj), f"[{model_name}] non-finite E"
            assert all(math.isfinite(c) for c in rep.C_traj), f"[{model_name}] non-finite C"
            assert all(math.isfinite(r) for r in rep.Rg_traj), f"[{model_name}] non-finite Rg"

        dist = build_distributions(reps, rg_bins=40, burnin_frac=0.5)
        attach_model_metadata(dist, model_name, param_names, model_params, Tref, Tscale)

        for key in ("model_name", "param_names", "model_params", "Tref", "Tscale"):
            assert key in dist, f"[{model_name}] missing metadata key: {key}"
        assert str(dist["model_name"]) == model_name
        np.testing.assert_array_equal(dist["param_names"], np.array(param_names))
        np.testing.assert_allclose(dist["model_params"], np.array(model_params, dtype=float))
        assert float(dist["Tref"]) == float(Tref)
        assert float(dist["Tscale"]) == float(Tscale)

        for i, row in enumerate(dist["Pc"]):
            finite = row[np.isfinite(row)]
            if finite.size > 0:
                assert abs(float(finite.sum()) - 1.0) < 1e-6, f"[{model_name}] Pc[{i}] not normalised"
        for i, row in enumerate(dist["Prg"]):
            finite = row[np.isfinite(row)]
            if finite.size > 0:
                assert abs(float(finite.sum()) - 1.0) < 1e-6, f"[{model_name}] Prg[{i}] not normalised"
        print(f"  quick-test model={model_name}: PASSED")

    model_cases = [
        {"model": "hs",            "params": [300.0, 1.0],            "Tref": 300.0, "Tscale": 80.0},
        {"model": "tc_scale",      "params": [1.0, 300.0],           "Tref": 300.0, "Tscale": 80.0},
        {"model": "hs_quadratic",  "params": [300.0, 1.0, 0.1],      "Tref": 300.0, "Tscale": 80.0},
        {"model": "poly2",         "params": [0.0, -0.5, 0.05],      "Tref": 300.0, "Tscale": 80.0},
        {"model": "poly3",         "params": [0.0, -0.5, 0.05, 0.01], "Tref": 300.0, "Tscale": 80.0},
        {"model": "heat_capacity", "params": [300.0, 1.0, 0.1],      "Tref": 300.0, "Tscale": 80.0},
    ]
    Ts_models = np.linspace(260.0, 340.0, 4)
    for k, case in enumerate(model_cases):
        _check_model_run(
            case["model"], case["params"], Ts_models,
            case["Tref"], case["Tscale"], seed=11 + k,
        )

    # -----------------------------------------------------------------------
    # fit_summary.json loading and resolve_model_params precedence tests
    # -----------------------------------------------------------------------
    def _make_args(**overrides):
        """Build a CLI-like namespace with the same defaults resolve uses."""
        defaults = dict(
            model=None, params=None, fit_params_csv=None, fit_summary_json=None,
            dh=312.109, ds=1.23278, Tref=None, Tscale=None, T0=None,
        )
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def _write_summary(dirpath, obj, name="fit_summary.json"):
        path = os.path.join(dirpath, name)
        with open(path, "w") as fh:
            json.dump(obj, fh)
        return path

    Ts_resolve = np.linspace(220.0, 320.0, 8)

    # Test 1: poly2 summary loading
    with tempfile.TemporaryDirectory() as tmp:
        p = _write_summary(tmp, {
            "model": "poly2",
            "param_names": ["a0", "a1", "a2"],
            "params": {"a0": 0.1, "a1": -0.5, "a2": 0.03},
            "Tref": 300.0,
            "Tscale": 80.0,
        })
        (model_name, params, param_names, Tref, Tscale,
         src, summ) = resolve_model_params(_make_args(fit_summary_json=p), Ts_resolve)
        assert model_name == "poly2", model_name
        assert param_names == ["a0", "a1", "a2"], param_names
        np.testing.assert_allclose(params, [0.1, -0.5, 0.03])
        assert Tref == 300.0 and Tscale == 80.0, (Tref, Tscale)
        assert src == "fit_summary_json", src
        assert summ == p, summ
    print("  quick-test fit_summary poly2 loading: PASSED")

    # Test 2: heat_capacity summary loading (Tref -> T0, and future T0 override)
    with tempfile.TemporaryDirectory() as tmp:
        p = _write_summary(tmp, {
            "model": "heat_capacity",
            "params": {"dh0": 500.0, "ds0": 1.5, "dCp": 0.2},
            "Tref": 300.0,
            "Tscale": 80.0,
        })
        _, _, _, Tref, _, _, _ = resolve_model_params(
            _make_args(fit_summary_json=p), Ts_resolve
        )
        assert Tref == 300.0, Tref

        p2 = _write_summary(tmp, {
            "model": "heat_capacity",
            "params": {"dh0": 500.0, "ds0": 1.5, "dCp": 0.2},
            "Tref": 300.0,
            "T0": 305.0,
            "Tscale": 80.0,
        }, name="fit_summary_t0.json")
        _, _, _, Tref2, _, _, _ = resolve_model_params(
            _make_args(fit_summary_json=p2), Ts_resolve
        )
        assert Tref2 == 305.0, Tref2
    print("  quick-test fit_summary heat_capacity T0: PASSED")

    # Test 3: parameter ordering independent of JSON insertion order
    with tempfile.TemporaryDirectory() as tmp:
        p = _write_summary(tmp, {
            "model": "poly3",
            "params": {"a2": 2.0, "a0": 0.0, "a3": 3.0, "a1": 1.0},
            "Tref": 300.0,
            "Tscale": 80.0,
        })
        _, params, param_names, _, _, _, _ = resolve_model_params(
            _make_args(fit_summary_json=p), Ts_resolve
        )
        assert param_names == ["a0", "a1", "a2", "a3"], param_names
        np.testing.assert_allclose(params, [0.0, 1.0, 2.0, 3.0])
    print("  quick-test fit_summary param ordering: PASSED")

    # Test 4: model mismatch between --model and summary
    with tempfile.TemporaryDirectory() as tmp:
        p = _write_summary(tmp, {
            "model": "poly2",
            "params": {"a0": 0.1, "a1": -0.5, "a2": 0.03},
            "Tref": 300.0, "Tscale": 80.0,
        })
        try:
            resolve_model_params(
                _make_args(fit_summary_json=p, model="poly3"), Ts_resolve
            )
            raise AssertionError("expected model mismatch error")
        except ValueError as e:
            assert "conflicts with model" in str(e), str(e)
    print("  quick-test fit_summary model mismatch: PASSED")

    # Test 5: conflicting Tref
    with tempfile.TemporaryDirectory() as tmp:
        p = _write_summary(tmp, {
            "model": "poly2",
            "params": {"a0": 0.1, "a1": -0.5, "a2": 0.03},
            "Tref": 300.0, "Tscale": 80.0,
        })
        try:
            resolve_model_params(
                _make_args(fit_summary_json=p, Tref=310.0), Ts_resolve
            )
            raise AssertionError("expected Tref conflict error")
        except ValueError as e:
            assert "Tref" in str(e), str(e)
        # Agreeing Tref within tolerance is accepted.
        _, _, _, Tref, _, _, _ = resolve_model_params(
            _make_args(fit_summary_json=p, Tref=300.0), Ts_resolve
        )
        assert Tref == 300.0
    print("  quick-test fit_summary Tref conflict: PASSED")

    # Test 6: ambiguous sources
    with tempfile.TemporaryDirectory() as tmp:
        p = _write_summary(tmp, {
            "model": "poly2",
            "params": {"a0": 0.1, "a1": -0.5, "a2": 0.03},
            "Tref": 300.0, "Tscale": 80.0,
        })
        for extra in (dict(params="0,0,0"), dict(fit_params_csv="x.csv")):
            try:
                resolve_model_params(
                    _make_args(fit_summary_json=p, **extra), Ts_resolve
                )
                raise AssertionError("expected ambiguous-source error")
            except ValueError as e:
                assert "only one model parameter source" in str(e), str(e)
    print("  quick-test ambiguous sources: PASSED")

    # Test 7: missing parameter (poly3 summary missing a3)
    with tempfile.TemporaryDirectory() as tmp:
        p = _write_summary(tmp, {
            "model": "poly3",
            "params": {"a0": 0.0, "a1": 1.0, "a2": 2.0},
            "Tref": 300.0, "Tscale": 80.0,
        })
        try:
            resolve_model_params(_make_args(fit_summary_json=p), Ts_resolve)
            raise AssertionError("expected missing-parameter error")
        except ValueError as e:
            assert "missing parameter" in str(e).lower(), str(e)
    print("  quick-test missing parameter: PASSED")

    # Test 8: old hs fallback (no summary, no csv, no params, no model)
    (model_name, params, param_names, Tref, Tscale,
     src, summ) = resolve_model_params(_make_args(), Ts_resolve)
    assert model_name == "hs", model_name
    np.testing.assert_allclose(params, [312.109, 1.23278])
    assert src == "legacy_dh_ds", src
    assert summ is None
    print("  quick-test legacy hs fallback: PASSED")

    # Companion-summary discovery for --fit-params-csv
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = os.path.join(tmp, "fit_params.csv")
        with open(csv_path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["parameter", "value"])
            for n, v in zip(["a0", "a1", "a2"], [0.1, -0.5, 0.03]):
                w.writerow([n, v])
        _write_summary(tmp, {
            "model": "poly2",
            "params": {"a0": 0.1, "a1": -0.5, "a2": 0.03},
            "Tref": 300.0, "Tscale": 80.0,
        })
        (model_name, params, _, Tref, Tscale,
         src, summ) = resolve_model_params(
            _make_args(fit_params_csv=csv_path, model="poly2"), Ts_resolve
        )
        assert src == "fit_params_csv_with_summary", src
        assert Tref == 300.0 and Tscale == 80.0, (Tref, Tscale)
        assert summ is not None
    print("  quick-test companion summary discovery: PASSED")

    # Test 9: small REMD run loaded from a summary file
    with tempfile.TemporaryDirectory() as tmp:
        p = _write_summary(tmp, {
            "model": "poly2",
            "param_names": ["a0", "a1", "a2"],
            "params": {"a0": 0.0, "a1": -0.5, "a2": 0.0},
            "Tref": 300.0,
            "Tscale": 80.0,
        })
        Ts_run = np.linspace(260, 340, 4)
        (model_name, params, param_names, Tref, Tscale,
         src, summ) = resolve_model_params(
            _make_args(fit_summary_json=p), Ts_run
        )
        reps, _sp, _sa = run_remd(
            N=20, Ts=Ts_run, steps_per_swap=50, n_cycles=20,
            model_name=model_name, params=params, Tref=Tref, Tscale=Tscale,
            seed=5, n_workers=1, verbose=False,
        )
        dist = build_distributions(reps, rg_bins=20, burnin_frac=0.5)
        attach_model_metadata(
            dist, model_name, param_names, params, Tref, Tscale,
            parameter_source=src, fit_summary_json=summ,
        )
        for i, row in enumerate(dist["Pc"]):
            finite = row[np.isfinite(row)]
            if finite.size > 0:
                assert abs(float(finite.sum()) - 1.0) < 1e-6, f"Pc[{i}] not normalised"
        for i, row in enumerate(dist["Prg"]):
            finite = row[np.isfinite(row)]
            if finite.size > 0:
                assert abs(float(finite.sum()) - 1.0) < 1e-6, f"Prg[{i}] not normalised"
        assert str(dist["model_name"]) == "poly2"
        np.testing.assert_array_equal(dist["param_names"], np.array(["a0", "a1", "a2"]))
        np.testing.assert_allclose(dist["model_params"], np.array([0.0, -0.5, 0.0]))
        assert float(dist["Tref"]) == 300.0
        assert float(dist["Tscale"]) == 80.0
        assert str(dist["parameter_source"]) == "fit_summary_json"
        assert str(dist["fit_summary_json"]) == p
    print("  quick-test REMD-from-summary: PASSED")

    # -----------------------------------------------------------------------
    # Part J: CSV companion-summary tests
    # -----------------------------------------------------------------------
    def _write_csv(dirpath, rows, name="fit_params.csv"):
        path = os.path.join(dirpath, name)
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["parameter", "value"])
            for n, v in rows:
                w.writerow([n, v])
        return path

    poly2_summary = {
        "model": "poly2",
        "params": {"a0": 0.1, "a1": -0.5, "a2": 0.03},
        "Tref": 300.0, "Tscale": 80.0,
    }
    poly2_rows = [("a0", 0.1), ("a1", -0.5), ("a2", 0.03)]

    # J1: infer poly2 from companion summary, no --model supplied
    with tempfile.TemporaryDirectory() as tmp:
        csvp = _write_csv(tmp, poly2_rows)
        _write_summary(tmp, poly2_summary)
        (model_name, params, _, Tref, Tscale, src, summ) = resolve_model_params(
            _make_args(fit_params_csv=csvp), Ts_resolve
        )
        assert model_name == "poly2", model_name
        assert src == "fit_params_csv_with_summary", src
        assert Tref == 300.0 and Tscale == 80.0, (Tref, Tscale)
        assert summ is not None
    print("  quick-test CSV infer poly2 from summary: PASSED")

    # J2: explicit matching model succeeds
    with tempfile.TemporaryDirectory() as tmp:
        csvp = _write_csv(tmp, poly2_rows)
        _write_summary(tmp, poly2_summary)
        (model_name, _, _, _, _, src, _) = resolve_model_params(
            _make_args(fit_params_csv=csvp, model="poly2"), Ts_resolve
        )
        assert model_name == "poly2" and src == "fit_params_csv_with_summary"
    print("  quick-test CSV explicit matching model: PASSED")

    # J3: explicit conflicting model raises
    with tempfile.TemporaryDirectory() as tmp:
        csvp = _write_csv(tmp, poly2_rows)
        _write_summary(tmp, poly2_summary)
        try:
            resolve_model_params(
                _make_args(fit_params_csv=csvp, model="poly3"), Ts_resolve
            )
            raise AssertionError("expected model conflict error")
        except ValueError as e:
            assert "conflicts with model" in str(e), str(e)
    print("  quick-test CSV conflicting model: PASSED")

    # J4: CSV/summary parameter mismatch raises
    with tempfile.TemporaryDirectory() as tmp:
        csvp = _write_csv(tmp, [("a0", 0.1), ("a1", -0.4), ("a2", 0.03)])
        _write_summary(tmp, poly2_summary)
        try:
            resolve_model_params(_make_args(fit_params_csv=csvp), Ts_resolve)
            raise AssertionError("expected CSV/summary param mismatch error")
        except ValueError as e:
            assert "do not match" in str(e), str(e)
    print("  quick-test CSV/summary param mismatch: PASSED")

    # J5: CLI Tref conflict raises
    with tempfile.TemporaryDirectory() as tmp:
        csvp = _write_csv(tmp, poly2_rows)
        _write_summary(tmp, poly2_summary)
        try:
            resolve_model_params(
                _make_args(fit_params_csv=csvp, Tref=310.0), Ts_resolve
            )
            raise AssertionError("expected Tref conflict error")
        except ValueError as e:
            assert "Tref" in str(e), str(e)
    print("  quick-test CSV CLI Tref conflict: PASSED")

    # J6: CLI Tscale conflict raises
    with tempfile.TemporaryDirectory() as tmp:
        csvp = _write_csv(tmp, poly2_rows)
        _write_summary(tmp, poly2_summary)
        try:
            resolve_model_params(
                _make_args(fit_params_csv=csvp, Tscale=100.0), Ts_resolve
            )
            raise AssertionError("expected Tscale conflict error")
        except ValueError as e:
            assert "Tscale" in str(e), str(e)
    print("  quick-test CSV CLI Tscale conflict: PASSED")

    # J7: heat_capacity T0 conflict raises
    with tempfile.TemporaryDirectory() as tmp:
        csvp = _write_csv(tmp, [("dh0", 500.0), ("ds0", 1.5), ("dCp", 0.2)])
        _write_summary(tmp, {
            "model": "heat_capacity",
            "params": {"dh0": 500.0, "ds0": 1.5, "dCp": 0.2},
            "Tref": 300.0, "Tscale": 80.0,
        })
        try:
            resolve_model_params(
                _make_args(fit_params_csv=csvp, T0=310.0), Ts_resolve
            )
            raise AssertionError("expected T0 conflict error")
        except ValueError as e:
            assert "T0" in str(e), str(e)
    print("  quick-test CSV heat_capacity T0 conflict: PASSED")

    # J8: no companion summary -> CSV-only path still works
    with tempfile.TemporaryDirectory() as tmp:
        csvp = _write_csv(tmp, poly2_rows)
        (model_name, _, _, Tref, Tscale, src, summ) = resolve_model_params(
            _make_args(fit_params_csv=csvp, model="poly2", Tref=300.0, Tscale=80.0),
            Ts_resolve,
        )
        assert model_name == "poly2"
        assert src == "fit_params_csv", src
        assert summ is None
        assert Tref == 300.0 and Tscale == 80.0
    print("  quick-test CSV-only (no companion): PASSED")

    # -----------------------------------------------------------------------
    # Part K: validation regression tests
    # -----------------------------------------------------------------------
    def _expect_value_error(thunk, needle, label):
        try:
            thunk()
            raise AssertionError(f"{label}: expected ValueError")
        except ValueError as e:
            assert needle in str(e), f"{label}: {e}"

    _expect_value_error(
        lambda: resolve_model_params(
            _make_args(model="poly2", params="0.1,nan,0.3"), Ts_resolve
        ),
        "finite float", "params nan",
    )
    _expect_value_error(
        lambda: resolve_model_params(
            _make_args(model="poly2", params="0.1,inf,0.3"), Ts_resolve
        ),
        "finite float", "params inf",
    )
    _expect_value_error(
        lambda: resolve_model_params(
            _make_args(model="poly2", params="0,-0.5,0.05", Tscale=0.0), Ts_resolve
        ),
        "Tscale", "Tscale=0",
    )
    _expect_value_error(
        lambda: resolve_model_params(
            _make_args(model="poly2", params="0,-0.5,0.05", Tscale=-1.0), Ts_resolve
        ),
        "Tscale", "Tscale=-1",
    )
    _expect_value_error(
        lambda: resolve_model_params(
            _make_args(model="poly2", params="0,-0.5,0.05", Tscale=float("nan")),
            Ts_resolve,
        ),
        "Tscale", "Tscale=nan",
    )
    _expect_value_error(
        lambda: resolve_model_params(
            _make_args(model="heat_capacity", params="500,1.5,0.2", T0=0.0),
            Ts_resolve,
        ),
        "T0", "heat_capacity T0=0",
    )
    _expect_value_error(
        lambda: resolve_model_params(
            _make_args(model="heat_capacity", params="500,1.5,0.2", T0=-5.0),
            Ts_resolve,
        ),
        "T0", "heat_capacity T0<0",
    )
    _expect_value_error(
        lambda: resolve_model_params(
            _make_args(model="poly2", params="0,-0.5,0.05", T0=300.0), Ts_resolve
        ),
        "--T0 is only valid", "T0 for poly2",
    )

    with tempfile.TemporaryDirectory() as tmp:
        dupcsv = _write_csv(
            tmp, [("a0", 0.1), ("a1", -0.5), ("a1", -0.6), ("a2", 0.03)],
            name="dup.csv",
        )
        _expect_value_error(
            lambda: load_fit_params_csv(dupcsv, "poly2"),
            "Duplicate", "duplicate CSV param",
        )
        nfcsv = _write_csv(
            tmp, [("a0", 0.1), ("a1", "nan"), ("a2", 0.03)], name="nf.csv",
        )
        _expect_value_error(
            lambda: load_fit_params_csv(nfcsv, "poly2"),
            "finite", "nonfinite CSV value",
        )
    print("  quick-test validation regressions: PASSED")

    print("quick-test complete.")


# ---------------------------------------------------------------------------
# Model parameter resolution
# ---------------------------------------------------------------------------

def validate_temperature_metadata(
    model_name: str,
    Tref: float,
    Tscale: float,
    source_description: str,
) -> None:
    """Validate resolved temperature metadata (finite Tref/Tscale, positive scale).

    For heat_capacity the reference temperature additionally serves as the
    thermodynamic T0 and must be positive.  ``source_description`` is used to
    make error messages point at the originating parameter source.
    """
    if not math.isfinite(Tref):
        raise ValueError(
            f"{source_description}: Tref/T0 must be finite, got {Tref!r}"
        )

    if not math.isfinite(Tscale):
        raise ValueError(
            f"{source_description}: Tscale must be finite, got {Tscale!r}"
        )

    if Tscale <= 0.0:
        raise ValueError(
            f"{source_description}: Tscale must be positive, got {Tscale!r}"
        )

    if model_name == "heat_capacity" and Tref <= 0.0:
        raise ValueError(
            f"{source_description}: heat_capacity T0 must be positive, "
            f"got {Tref!r}"
        )


def validate_model_params(
    model_name: str,
    params,
    source_description: str,
) -> list[float]:
    """Validate parameter count and finiteness; return floats in registry order.

    The parameter order always comes from MODEL_REGISTRY[model_name];
    ``params`` is expected to already be in that order.
    """
    param_names = MODEL_REGISTRY[model_name]["param_names"]

    if len(params) != len(param_names):
        raise ValueError(
            f"{source_description}: model {model_name!r} expects "
            f"{len(param_names)} parameters {param_names}, "
            f"but received {len(params)} values: {params}"
        )

    checked = []
    for name, value in zip(param_names, params):
        try:
            value_float = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{source_description}: parameter {name!r} could not be "
                f"converted to float: {value!r}"
            ) from exc

        if not math.isfinite(value_float):
            raise ValueError(
                f"{source_description}: parameter {name!r} must be finite, "
                f"got {value_float!r}"
            )

        checked.append(value_float)

    return checked


def validate_cli_against_summary(
    args,
    model_name: str,
    summary_Tref: float,
    summary_Tscale: float,
    summary_path: str,
) -> None:
    """Ensure explicit CLI Tref/Tscale/T0 agree with companion summary metadata.

    summary_Tref is the effective reference temperature (== T0 for the
    heat_capacity model).  Conflicting values are rejected rather than silently
    overridden.  --T0 is only meaningful for heat_capacity.
    """
    if args.Tscale is not None and not np.isclose(
        float(args.Tscale), summary_Tscale, rtol=1e-10, atol=1e-12
    ):
        raise ValueError(
            f"--Tscale {args.Tscale} conflicts with Tscale {summary_Tscale} "
            f"stored in {summary_path}"
        )

    if model_name == "heat_capacity":
        # --T0 and --Tref (if given) are both compared against the effective
        # reference temperature; they are not applied independently.
        if args.T0 is not None and not np.isclose(
            float(args.T0), summary_Tref, rtol=1e-10, atol=1e-12
        ):
            raise ValueError(
                f"--T0 {args.T0} conflicts with heat_capacity reference "
                f"temperature {summary_Tref} stored in {summary_path}"
            )
        if args.Tref is not None and not np.isclose(
            float(args.Tref), summary_Tref, rtol=1e-10, atol=1e-12
        ):
            raise ValueError(
                f"--Tref {args.Tref} conflicts with heat_capacity reference "
                f"temperature {summary_Tref} stored in {summary_path}"
            )
    else:
        if args.T0 is not None:
            raise ValueError("--T0 is only valid for the heat_capacity model")
        if args.Tref is not None and not np.isclose(
            float(args.Tref), summary_Tref, rtol=1e-10, atol=1e-12
        ):
            raise ValueError(
                f"--Tref {args.Tref} conflicts with Tref {summary_Tref} stored "
                f"in {summary_path}"
            )


def parse_params_string(params_str: str) -> list[float]:
    """Parse a comma-separated --params string into finite floats.

    Rejects an empty string and any value that is not a finite float (nan/inf),
    reporting the offending 1-based position.  The model-specific count check is
    performed separately via validate_model_params.
    """
    vals = [x.strip() for x in params_str.split(",") if x.strip()]
    if not vals:
        raise ValueError("--params was provided but no values were found")
    out: list[float] = []
    for i, x in enumerate(vals, start=1):
        try:
            v = float(x)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"--params value {i} is not a finite float: {x!r}"
            ) from exc
        if not math.isfinite(v):
            raise ValueError(f"--params value {i} is not a finite float: {x!r}")
        out.append(v)
    return out


def load_fit_params_csv(path: str, model_name: str) -> list[float]:
    import csv

    if not Path(path).exists():
        raise FileNotFoundError(f"fit_params.csv not found: {path}")

    needed = MODEL_REGISTRY[model_name]["param_names"]
    found: dict[str, float] = {}

    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        if (
            reader.fieldnames is None
            or "parameter" not in reader.fieldnames
            or "value" not in reader.fieldnames
        ):
            raise ValueError("fit_params.csv must contain columns: parameter,value")
        for row in reader:
            name = row["parameter"]
            if name not in needed:
                # Ignore derived/extra rows (e.g. Tc) that are not parameters.
                continue
            if name in found:
                raise ValueError(f"Duplicate parameter {name!r} in {path}")
            raw = row["value"]
            try:
                found[name] = float(raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Parameter {name!r} in {path} is not numeric: {raw!r}"
                ) from exc

    missing = [name for name in needed if name not in found]
    if missing:
        raise ValueError(
            f"Missing parameters for model {model_name!r} in {path}: {missing}"
        )

    # Registry order + finite-value validation.
    return validate_model_params(
        model_name, [found[name] for name in needed], f"fit_params.csv {path!r}"
    )


def load_fit_summary_json(path: str) -> dict:
    """Load and validate fit_summary.json from fit_lattice_contact_model.py.

    Returns a normalized dictionary carrying the model name, parameters in
    registry order, parameter names, Tref, Tscale, and the source path.  The
    parameter order is always taken from MODEL_REGISTRY (never from JSON
    insertion order).  For the heat_capacity model the thermodynamic reference
    temperature is taken from a future ``T0`` field if present, otherwise from
    ``Tref`` (where the fitter currently stores it).
    """
    with open(path) as fh:
        summary = json.load(fh)

    if not isinstance(summary, dict):
        raise ValueError(
            f"fit_summary.json {path!r} must contain a JSON object/dictionary "
            f"at the root."
        )

    required = ("model", "params", "Tref", "Tscale")
    missing_fields = [k for k in required if k not in summary]
    if missing_fields:
        raise ValueError(
            f"fit_summary.json {path!r} is missing required field(s): "
            f"{missing_fields}"
        )

    model_name = summary["model"]
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"fit_summary.json {path!r} specifies unknown model "
            f"{model_name!r}. Known models: {list(MODEL_REGISTRY.keys())}"
        )

    raw_params = summary["params"]
    if not isinstance(raw_params, dict):
        raise ValueError(
            f"fit_summary.json {path!r} field 'params' must be a dictionary "
            f"mapping parameter name -> value."
        )

    param_names = MODEL_REGISTRY[model_name]["param_names"]
    missing_params = [name for name in param_names if name not in raw_params]
    if missing_params:
        raise ValueError(
            f"fit_summary.json {path!r} is missing parameter(s) for model "
            f"{model_name!r}: {missing_params}. Expected: {param_names}"
        )

    # Registry order (NOT JSON insertion order) + finite-value validation.
    params = validate_model_params(
        model_name,
        [raw_params[name] for name in param_names],
        f"fit_summary.json {path!r}",
    )

    if model_name == "heat_capacity":
        Tref = float(summary.get("T0", summary["Tref"]))
    else:
        Tref = float(summary["Tref"])
    Tscale = float(summary["Tscale"])

    validate_temperature_metadata(
        model_name, Tref, Tscale, f"fit_summary.json {path!r}"
    )

    return {
        "model_name": model_name,
        "param_names": list(param_names),
        "params": params,
        "Tref": Tref,
        "Tscale": Tscale,
        "source_path": str(path),
    }


def resolve_model_params(
    args, Ts: np.ndarray
) -> tuple[str, list[float], list[str], float, float, str, str | None]:
    """Resolve the contact-bias model and its parameters.

    Precedence of explicit parameter sources (mutually exclusive):
        1. --fit-summary-json
        2. --fit-params-csv  (with optional companion fit_summary.json)
        3. --params
        4. legacy --dh/--ds (hs only)

    Returns (model_name, params, param_names, Tref, Tscale, parameter_source,
    fit_summary_json) where fit_summary_json is the summary path when one was
    used, else None.
    """
    # Reject ambiguous combinations of explicit sources.
    explicit = [
        ("--fit-summary-json", args.fit_summary_json is not None),
        ("--fit-params-csv", args.fit_params_csv is not None),
        ("--params", args.params is not None),
    ]
    n_explicit = sum(1 for _, present in explicit if present)
    if n_explicit > 1:
        raise ValueError(
            "Choose only one model parameter source: --fit-summary-json, "
            "--fit-params-csv, or --params."
        )

    # Defensive validation of the temperature ladder (for automated workflows).
    Ts = np.asarray(Ts, dtype=float)
    if Ts.ndim != 1:
        raise ValueError("Ts (temperature ladder) must be one-dimensional")
    if Ts.size == 0:
        raise ValueError("Ts (temperature ladder) must be nonempty")
    if not np.all(np.isfinite(Ts)):
        raise ValueError("Ts (temperature ladder) must contain only finite values")
    if not np.all(Ts > 0.0):
        raise ValueError("Ts (temperature ladder) must contain only positive temperatures")

    Tmin = float(np.min(Ts))
    Tmax = float(np.max(Ts))

    def _default_Tref_Tscale() -> tuple[float, float]:
        Tref = float(args.Tref) if args.Tref is not None else 0.5 * (Tmin + Tmax)
        Tscale = (
            float(args.Tscale) if args.Tscale is not None else max(Tmax - Tmin, 1.0)
        )
        return Tref, Tscale

    # --- Case A: --fit-summary-json supplied -------------------------------
    if args.fit_summary_json is not None:
        summary = load_fit_summary_json(args.fit_summary_json)
        model_name = summary["model_name"]
        param_names = summary["param_names"]
        params = summary["params"]
        Tref = summary["Tref"]
        Tscale = summary["Tscale"]

        if args.model is not None and args.model != model_name:
            raise ValueError(
                f"--model {args.model} conflicts with model {model_name} "
                f"stored in fit_summary.json"
            )

        # Any explicitly supplied Tref/Tscale/T0 must agree with the summary.
        validate_cli_against_summary(
            args, model_name, Tref, Tscale, summary["source_path"]
        )

        return (
            model_name, params, param_names, Tref, Tscale,
            "fit_summary_json", summary["source_path"],
        )

    # --- Case B: --fit-params-csv supplied ---------------------------------
    if args.fit_params_csv is not None:
        csv_path = Path(args.fit_params_csv)
        candidate_summary = csv_path.with_name("fit_summary.json")

        if candidate_summary.exists():
            # Companion summary is authoritative for model + temperature metadata.
            companion = load_fit_summary_json(str(candidate_summary))
            summary_model = companion["model_name"]
            if args.model is None:
                model_name = summary_model
            elif args.model != summary_model:
                raise ValueError(
                    f"--model {args.model} conflicts with model {summary_model} "
                    f"stored in companion fit_summary.json"
                )
            else:
                model_name = args.model

            param_names = MODEL_REGISTRY[model_name]["param_names"]
            csv_params = load_fit_params_csv(str(csv_path), model_name)

            if not np.allclose(
                np.asarray(csv_params, dtype=float),
                np.asarray(companion["params"], dtype=float),
                rtol=1e-10, atol=1e-12,
            ):
                raise ValueError(
                    f"CSV parameters {csv_params} do not match companion "
                    f"fit_summary.json parameters {companion['params']} "
                    f"({companion['source_path']})."
                )

            validate_cli_against_summary(
                args, model_name, companion["Tref"], companion["Tscale"],
                companion["source_path"],
            )
            Tref = companion["Tref"]
            Tscale = companion["Tscale"]
            validate_temperature_metadata(
                model_name, Tref, Tscale,
                f"companion fit_summary.json {companion['source_path']!r}",
            )
            print(f"Loaded companion metadata from {companion['source_path']}")
            return (
                model_name, csv_params, param_names, Tref, Tscale,
                "fit_params_csv_with_summary", companion["source_path"],
            )

        # No companion summary: preserve original CSV-only behavior.
        model_name = args.model if args.model is not None else "hs"
        if args.T0 is not None and model_name != "heat_capacity":
            raise ValueError(
                "--T0 is only valid when --model heat_capacity is used"
            )
        param_names = MODEL_REGISTRY[model_name]["param_names"]
        Tref, Tscale = _default_Tref_Tscale()
        if model_name == "heat_capacity" and args.T0 is not None:
            Tref = float(args.T0)
        csv_params = load_fit_params_csv(str(csv_path), model_name)
        validate_temperature_metadata(
            model_name, Tref, Tscale, "CSV-only pathway (CLI/default Tref/Tscale)"
        )
        return (
            model_name, csv_params, param_names, Tref, Tscale,
            "fit_params_csv", None,
        )

    # Resolve model for the remaining (non-CSV, non-summary) cases.
    model_name = args.model if args.model is not None else "hs"
    if args.T0 is not None and model_name != "heat_capacity":
        raise ValueError("--T0 is only valid when --model heat_capacity is used")
    param_names = MODEL_REGISTRY[model_name]["param_names"]
    Tref, Tscale = _default_Tref_Tscale()
    if model_name == "heat_capacity" and args.T0 is not None:
        Tref = float(args.T0)

    # --- Case C: --params supplied -----------------------------------------
    if args.params is not None:
        params = parse_params_string(args.params)
        parameter_source = "params_cli"
        source_desc = "--params"
    # --- Case D: no explicit source ----------------------------------------
    elif model_name == "hs":
        # Backward compatibility: old --dh/--ds flags define h and s.
        params = [args.dh, args.ds]
        parameter_source = "legacy_dh_ds"
        source_desc = "legacy --dh/--ds"
    else:
        raise ValueError(
            f"Model {model_name!r} requires one of --params, --fit-params-csv, "
            f"or --fit-summary-json. Expected parameters: {param_names}"
        )

    params = validate_model_params(model_name, params, source_desc)
    validate_temperature_metadata(
        model_name, Tref, Tscale, f"{source_desc} (CLI/default Tref/Tscale)"
    )

    return model_name, params, param_names, Tref, Tscale, parameter_source, None


def attach_model_metadata(
    dist: dict,
    model_name: str,
    param_names: list[str],
    model_params: list[float],
    Tref: float,
    Tscale: float,
    parameter_source: str | None = None,
    fit_summary_json: str | None = None,
) -> dict:
    """Inject contact-bias model metadata into a distributions dict in place.

    The existing metadata keys (model_name, param_names, model_params, Tref,
    Tscale) are preserved unchanged.  Optional provenance keys parameter_source
    and fit_summary_json are added when supplied.
    """
    dist["model_name"] = model_name
    dist["param_names"] = np.array(param_names)
    dist["model_params"] = np.array(model_params, dtype=float)
    dist["Tref"] = float(Tref)
    dist["Tscale"] = float(Tscale)
    if parameter_source is not None:
        dist["parameter_source"] = str(parameter_source)
    if fit_summary_json is not None:
        dist["fit_summary_json"] = str(fit_summary_json)
    return dist


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Self-contained REMD for lattice polymer with T-dependent Hamiltonian.",
    )
    ap.add_argument("--N",              type=int,   default=44,     help="chain length")
    ap.add_argument("--Tmin",           type=float, default=280.0,   help="lowest temperature")
    ap.add_argument("--Tmax",           type=float, default=360.0,   help="highest temperature")
    ap.add_argument("--nT",             type=int,   default=30,       help="number of replicas")
    ap.add_argument("--steps-per-swap", type=int,   default=500,     help="local MC steps per replica per swap cycle")
    ap.add_argument("--n-cycles",       type=int,   default=4000,    help="number of swap cycles")
    ap.add_argument("--dh",             type=float, default=312.109,  help="contact enthalpy dh; used as h when --model hs and --params/--fit-params-csv are not supplied")
    ap.add_argument("--ds",             type=float, default=1.23278, help="contact entropy ds; used as s when --model hs and --params/--fit-params-csv are not supplied")
    ap.add_argument("--seed",           type=int,   default=42,      help="RNG seed")
    ap.add_argument("--out-prefix",     type=str,   default="remd_out", help="prefix for all output files")
    ap.add_argument("--rg-bins",        type=int,   default=64,      help="bins for P(Rg) histograms")
    ap.add_argument("--burnin-frac",    type=float, default=0.7,     help="fraction of trajectory to discard as burnin")
    ap.add_argument("--n-workers",      type=int,   default=6,       help="parallel workers for local sweeps (1 = serial)")
    ap.add_argument(
        "--rg-scale", type=float, default=1.0,
        help=(
            "Scale factor for reporting/outputting Rg. "
            "Rg_output = rg_scale * Rg_lattice. Default 1.0 preserves lattice units."
        ),
    )
    ap.add_argument(
        "--model",
        type=str,
        default=None,
        choices=list(MODEL_REGISTRY.keys()),
        help=(
            "Contact-bias model b(T). If --fit-summary-json is supplied, the "
            "model is loaded from the summary (and --model, if given, must "
            "match it). Otherwise the default model is hs, which preserves the "
            "old --dh/--ds behavior."
        ),
    )
    ap.add_argument(
        "--fit-summary-json",
        type=str,
        default=None,
        dest="fit_summary_json",
        help=(
            "Load the fitted model, parameters, Tref, and Tscale directly from "
            "fit_summary.json produced by fit_lattice_contact_model.py. "
            "This is the preferred input for hs_quadratic, poly2, poly3, and "
            "heat_capacity models."
        ),
    )
    ap.add_argument(
        "--params",
        type=str,
        default=None,
        help=(
            "Comma-separated model parameters in the order expected by --model. "
            "Examples: hs='h,s', tc_scale='A,Tc', poly2='a0,a1,a2'. "
            "If omitted for hs, --dh and --ds are used."
        ),
    )
    ap.add_argument(
        "--fit-params-csv",
        type=str,
        default=None,
        dest="fit_params_csv",
        help=(
            "Load fitted parameters from fit_params.csv. If a companion "
            "fit_summary.json exists in the same directory, the model, Tref, "
            "Tscale, and heat-capacity T0 are inferred and validated automatically."
        ),
    )
    ap.add_argument(
        "--Tref",
        type=float,
        default=None,
        help=(
            "Reference temperature for polynomial models x=(T-Tref)/Tscale. "
            "Default: midpoint of the REMD temperature range."
        ),
    )
    ap.add_argument(
        "--Tscale",
        type=float,
        default=None,
        help="Scale for x=(T-Tref)/Tscale. Default: Tmax-Tmin.",
    )
    ap.add_argument(
        "--T0",
        type=float,
        default=None,
        help=(
            "Reference temperature T0 for heat_capacity model. "
            "Overrides Tref for heat_capacity only."
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

    (
        model_name, model_params, param_names, Tref, Tscale,
        parameter_source, fit_summary_json,
    ) = resolve_model_params(args, Ts)

    print(
        f"REMD: {args.nT} replicas, T in [{args.Tmin}, {args.Tmax}], "
        f"{args.n_cycles} cycles x {args.steps_per_swap} steps = {total_steps} steps/replica"
    )
    print(f"Model: {model_name} — {MODEL_REGISTRY[model_name]['description']}")
    print(f"Parameter source: {parameter_source}")
    if fit_summary_json is not None:
        print(f"Fit summary: {fit_summary_json}")
    print("Parameters:")
    for name, val in zip(param_names, model_params):
        print(f"  {name} = {val:.8g}")
    if model_name == "heat_capacity":
        print(f"T0 = {Tref:.8g}")
        print(f"Tscale = {Tscale:.8g}")
    else:
        print(f"Tref = {Tref:.8g}, Tscale = {Tscale:.8g}")
    if model_name == "hs" and abs(model_params[1]) > 1e-15:
        print(f"Derived Tc = {model_params[0] / model_params[1]:.8g}")
    elif model_name == "tc_scale":
        print(f"Tc = {model_params[1]:.8g}")

    replicas, swap_props, swap_accs = run_remd(
        N=args.N, Ts=Ts,
        steps_per_swap=args.steps_per_swap,
        n_cycles=args.n_cycles,
        model_name=model_name, params=model_params,
        Tref=Tref, Tscale=Tscale,
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
    attach_model_metadata(
        dist, model_name, param_names, model_params, Tref, Tscale,
        parameter_source=parameter_source, fit_summary_json=fit_summary_json,
    )

    save_results_csv(results, args.out_prefix)
    save_swap_csv(swap_props, swap_accs, Ts, args.out_prefix)
    save_distributions(dist, args.out_prefix)
    plot_observables(results, args.out_prefix)
    plot_distributions(dist, args.out_prefix)


if __name__ == "__main__":
    main()
