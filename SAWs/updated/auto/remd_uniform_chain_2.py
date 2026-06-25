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

import numpy as np

try:  # plotting is optional when --no-plots is used
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - depends on local plotting stack
    cm = None
    mcolors = None
    plt = None


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


# Shared model-contract version.  Bump only when the model set, parameter names,
# or b(T) semantics change in a way that breaks cross-script compatibility.
MODEL_API_VERSION = 1


def get_model_contract() -> dict:
    """Return a callable-free description of the supported contact-bias models.

    Used to verify that this script and fit_lattice_contact_model.py agree on the
    model API version, model names, and parameter ordering.
    """
    return {
        "model_api_version": MODEL_API_VERSION,
        "models": {
            name: {
                "param_names": list(spec["param_names"]),
                "description": str(spec["description"]),
            }
            for name, spec in MODEL_REGISTRY.items()
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


def _seed_all(seed: int) -> None:
    """Seed Python and NumPy RNGs without NumPy's unsigned-32-bit limitation."""
    random.seed(int(seed))
    np.random.seed(int(seed) % (2 ** 32))


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
    state: "ChainState",
    temperature: float,
    local_acc: int,
    local_prop: int,
    steps: int,
    model_name: str,
    params: list,
    Tref: float,
    Tscale: float,
    seed: int,
) -> tuple["ChainState", int, int]:
    """Deterministically sweep one lane without transferring trajectories.

    Only the mutable chain state and acceptance counters cross the process
    boundary. Sending the full Replica would also pickle its ever-growing
    trajectories every cycle, causing O(n_cycles^2) serialization overhead.
    """
    _seed_all(seed)
    replica = Replica(
        T=float(temperature), state=state,
        local_acc=int(local_acc), local_prop=int(local_prop),
    )
    mc_sweep(replica, steps, model_name, params, Tref, Tscale)
    return replica.state, replica.local_acc, replica.local_prop


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
    diagnostics: bool = False,
    diag_store: dict | None = None,
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
        _seed_all(seed)

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

    # Optional walker (configuration) identity tracking.  Lanes are temperature-
    # fixed; a "walker" is a configuration that migrates between lanes via swaps.
    # lane_walker[k] = walker currently in lane k; walker_lane[w] = lane of w.
    # Both are plain NumPy integer arrays (bounded memory, no Python objects).
    track_walkers = bool(diagnostics)
    if track_walkers:
        lane_walker = np.arange(nT, dtype=np.int64)
        walker_lane = np.arange(nT, dtype=np.int64)
        walker_temp_index = np.empty((n_cycles, nT), dtype=np.int32)
    else:
        walker_temp_index = None

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
                        evolve_replica_worker,
                        replicas[k].state, replicas[k].T,
                        replicas[k].local_acc, replicas[k].local_prop,
                        steps_per_swap, model_name, params, Tref, Tscale,
                        worker_seeds[k],
                    )
                    for k in range(nT)
                ]
                for k, fut in enumerate(futures):
                    state, local_acc, local_prop = fut.result()
                    replicas[k].state = state
                    replicas[k].local_acc = local_acc
                    replicas[k].local_prop = local_prop
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
                    if track_walkers:
                        wa = lane_walker[k]
                        wb = lane_walker[k + 1]
                        lane_walker[k] = wb
                        lane_walker[k + 1] = wa
                        walker_lane[wa] = k + 1
                        walker_lane[wb] = k

            t2 = time.perf_counter()
            t_swap_total += t2 - t1

            for rep in replicas:
                rep.E_traj.append(rep.state.E)
                rep.C_traj.append(contact_count(rep.state.chain, rep.state.occ))
                rep.Rg_traj.append(radius_of_gyration(rep.state.chain))

            if track_walkers:
                walker_temp_index[cycle] = walker_lane

            if verbose and (cycle + 1) % report_every == 0:
                rates = " ".join(
                    (
                        f"{swap_accs[k]/swap_props[k]:.2f}"
                        if swap_props[k] > 0 else "n/a"
                    )
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

    if track_walkers and diag_store is not None:
        diag_store["walker_temp_index"] = walker_temp_index

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
# Convergence and mixing diagnostics (optional; never alters canonical output)
# ---------------------------------------------------------------------------
# These diagnostics are computed in the main process from data the REMD loop
# already produces (per-lane post-burn-in observable trajectories and, when
# diagnostics are enabled, a walker temperature-index trajectory).  They do not
# touch the sampling rule, the swap criterion, the canonical distributions, or
# the multiprocessing path.

# Default thresholds for convergence/mixing warnings.  Every threshold is
# overridable from the CLI (and therefore from the suite config) and is recorded
# verbatim in run_diagnostics.json so a run is self-describing.
DEFAULT_DIAG_THRESHOLDS = {
    "min_round_trips": 1,        # warn if total low->high->low round trips < this
    "min_temp_coverage": 0.5,    # warn if any walker visits < this fraction of T
    "min_ess": 50.0,             # warn if any lane/observable ESS < this
    "max_drift": 1.0,            # warn if |early-late drift| / std > this
    "min_swap_rate": 0.05,       # warn if any adjacent swap rate < this
}
DEFAULT_DIAG_N_BLOCKS = 5


def _autocovariance(x: np.ndarray) -> np.ndarray:
    """Biased autocovariance (lags 0..n-1) via FFT; acov[0] is the variance."""
    x = np.asarray(x, dtype=float)
    n = x.size
    x = x - x.mean()
    size = 1
    while size < 2 * n:
        size *= 2
    f = np.fft.rfft(x, n=size)
    acov = np.fft.irfft(f * np.conjugate(f), n=size)[:n].real
    acov /= float(n)
    return acov


def integrated_autocorr_time(x) -> dict:
    """Integrated autocorrelation time via Geyer's initial-positive-sequence rule.

    Estimator (Geyer 1992): with normalized autocorrelations rho_k, form the
    paired sums Gamma_m = rho_{2m} + rho_{2m+1}.  The initial positive sequence
    truncates at the first m with Gamma_m <= 0, and

        tau_int = 2 * sum_{m=0..M} Gamma_m - 1,   ESS = n / tau_int.

    This is a documented, robust, monotone truncation that avoids summing noisy
    high-lag autocorrelations.  Returns a dict with tau_int, ess, n_samples and
    the method label.  A (near-)constant series is reported as tau_int = 1.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 2:
        return {"tau_int": float("nan"), "ess": float("nan"),
                "n_samples": int(n), "method": "insufficient_samples"}
    acov = _autocovariance(x)
    var = acov[0]
    if not np.isfinite(var) or var <= 0.0:
        return {"tau_int": 1.0, "ess": float(n), "n_samples": int(n),
                "method": "constant_series"}
    rho = acov / var
    gamma_sum = 0.0
    m = 0
    max_m = (n - 1) // 2
    while m <= max_m:
        idx = 2 * m
        g = rho[idx] + (rho[idx + 1] if (idx + 1) < n else 0.0)
        if g <= 0.0:
            break
        gamma_sum += g
        m += 1
    tau = 2.0 * gamma_sum - 1.0
    if not np.isfinite(tau) or tau < 1.0:
        tau = 1.0
    ess = n / tau
    if ess > n:
        ess = float(n)
    return {"tau_int": float(tau), "ess": float(ess), "n_samples": int(n),
            "method": "geyer_initial_positive_sequence"}


def early_late_drift(x) -> dict:
    """Compare the mean of the first half vs the second half of a series.

    Returns the raw drift (late - early) and the drift expressed in units of the
    series standard deviation (drift_in_std), which is the scale used for the
    configurable drift warning.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 2:
        return {"early_mean": float("nan"), "late_mean": float("nan"),
                "drift": float("nan"), "drift_in_std": float("nan")}
    half = n // 2
    early = float(x[:half].mean())
    late = float(x[half:].mean())
    drift = late - early
    sd = float(x.std(ddof=0))
    return {"early_mean": early, "late_mean": late, "drift": float(drift),
            "drift_in_std": float(drift / sd) if sd > 0.0 else 0.0}


def block_mean_stability(x, n_blocks: int) -> dict:
    """Stability of block means over `n_blocks` contiguous blocks.

    Reports the block means, their standard deviation, and the block-mean range
    normalized by the overall standard deviation (a scale-free stability index).
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    nb = max(1, int(n_blocks))
    if n < 1:
        return {"n_blocks": int(nb), "block_means": [],
                "block_mean_std": float("nan"),
                "block_mean_range_over_std": float("nan")}
    nb = min(nb, n)
    blocks = np.array_split(x, nb)
    bm = np.array([float(b.mean()) for b in blocks if b.size > 0], dtype=float)
    overall = float(x.std(ddof=0))
    bms = float(bm.std(ddof=0)) if bm.size > 1 else 0.0
    rng = float(bm.max() - bm.min()) if bm.size > 0 else 0.0
    return {"n_blocks": int(bm.size),
            "block_means": [float(v) for v in bm],
            "block_mean_std": bms,
            "block_mean_range_over_std": (rng / overall) if overall > 0.0 else 0.0}


def analyze_walker_trajectory(path, nT: int) -> dict:
    """Mixing diagnostics for one walker's temperature-index trajectory.

    `path` is the sequence of temperature indices (0 = lowest T, nT-1 = highest)
    occupied by a single walker across cycles.  Computes:
      - temperature-occupancy histogram and fraction of states visited;
      - first-passage times low->high and high->low;
      - complete L->H->L and H->L->H round-trip counts and durations.
    Round trips are counted on the sequence of extreme-temperature touches, so
    an alternating L,H,L,H,... record yields overlapping (shared-endpoint) trips
    in the conventional REMD sense.
    """
    path = np.asarray(path).astype(int)
    n = path.size
    occ = np.bincount(path, minlength=nT)[:nT] if nT > 0 else np.array([], dtype=int)
    visited = int(np.count_nonzero(occ))
    frac = (visited / float(nT)) if nT > 0 else 0.0

    bottom, top = 0, nT - 1
    events = []  # (cycle, 'L'|'H') at each *change* of extreme touched
    cur = None
    for c in range(n):
        v = int(path[c])
        if v == bottom:
            lab = "L"
        elif v == top:
            lab = "H"
        else:
            continue
        if lab != cur:
            events.append((c, lab))
            cur = lab

    lh, hl = [], []
    for i in range(len(events) - 1):
        a, b = events[i], events[i + 1]
        if a[1] == "L" and b[1] == "H":
            lh.append(b[0] - a[0])
        elif a[1] == "H" and b[1] == "L":
            hl.append(b[0] - a[0])

    dur_lhl, dur_hlh = [], []
    for i in range(len(events) - 2):
        a, b, c2 = events[i], events[i + 1], events[i + 2]
        if a[1] == "L" and b[1] == "H" and c2[1] == "L":
            dur_lhl.append(c2[0] - a[0])
        elif a[1] == "H" and b[1] == "L" and c2[1] == "H":
            dur_hlh.append(c2[0] - a[0])

    all_dur = dur_lhl + dur_hlh
    return {
        "occupancy": occ.astype(int),
        "n_visited": visited,
        "fraction_visited": float(frac),
        "first_passage_low_to_high": (int(lh[0]) if lh else None),
        "first_passage_high_to_low": (int(hl[0]) if hl else None),
        "n_round_trips_low": int(len(dur_lhl)),    # complete low->high->low
        "n_round_trips_high": int(len(dur_hlh)),   # complete high->low->high
        "n_round_trips": int(len(dur_lhl) + len(dur_hlh)),
        "round_trip_durations_low": [int(d) for d in dur_lhl],
        "round_trip_durations_high": [int(d) for d in dur_hlh],
        "mean_round_trip_duration": (float(np.mean(all_dur)) if all_dur else None),
    }


def _post_burnin_slice(n: int, burnin_frac: float) -> int:
    return int(math.floor(n * burnin_frac))


def compute_run_diagnostics(
    replicas: list,
    swap_props: np.ndarray,
    swap_accs: np.ndarray,
    walker_temp_index: np.ndarray,
    Ts: np.ndarray,
    burnin_frac: float,
    n_blocks: int,
    thresholds: dict,
    rg_scale: float = 1.0,
) -> dict:
    """Assemble per-lane convergence and per-walker mixing diagnostics + warnings.

    Lane convergence (autocorrelation/ESS/drift/block stability) is computed on
    the post-burn-in segment of each temperature lane (matching the canonical
    distributions).  Walker mixing (round trips, coverage) uses the full
    temperature-index trajectory, because burn-in is part of mixing.
    """
    nT = len(replicas)
    Ts = np.asarray(Ts, dtype=float)
    thr = dict(DEFAULT_DIAG_THRESHOLDS)
    thr.update(thresholds or {})

    lane_conv = []
    for i, rep in enumerate(replicas):
        n = len(rep.C_traj)
        s = _post_burnin_slice(n, burnin_frac)
        C = np.asarray(rep.C_traj[s:], dtype=float)
        E = np.asarray(rep.E_traj[s:], dtype=float)
        Rg = np.asarray(rep.Rg_traj[s:], dtype=float) * float(rg_scale)
        entry = {
            "temperature_index": int(i),
            "temperature": float(Ts[i]) if i < Ts.size else float("nan"),
            "n_post_burnin": int(C.size),
        }
        for name, arr in (("contacts", C), ("rg", Rg), ("energy", E)):
            ac = integrated_autocorr_time(arr)
            dr = early_late_drift(arr)
            bl = block_mean_stability(arr, n_blocks)
            entry[name] = {
                "mean": float(np.nanmean(arr)) if arr.size else float("nan"),
                "std": float(np.nanstd(arr, ddof=0)) if arr.size else float("nan"),
                "tau_int": ac["tau_int"],
                "ess": ac["ess"],
                "n_samples": ac["n_samples"],
                "acf_method": ac["method"],
                "drift": dr["drift"],
                "drift_in_std": dr["drift_in_std"],
                "early_mean": dr["early_mean"],
                "late_mean": dr["late_mean"],
                "block_mean_std": bl["block_mean_std"],
                "block_mean_range_over_std": bl["block_mean_range_over_std"],
                "block_means": bl["block_means"],
            }
        lane_conv.append(entry)

    wti = np.asarray(walker_temp_index)
    n_walkers = wti.shape[1] if wti.ndim == 2 else 0
    walker_diag = []
    for w in range(n_walkers):
        wd = analyze_walker_trajectory(wti[:, w], nT)
        wd["walker"] = int(w)
        walker_diag.append(wd)

    swap_rates = []
    for k in range(len(swap_props)):
        p = int(swap_props[k])
        a = int(swap_accs[k])
        swap_rates.append((a / p) if p > 0 else float("nan"))
    swap_rates_arr = np.array(swap_rates, dtype=float)
    finite_swap = swap_rates_arr[np.isfinite(swap_rates_arr)]

    # --- summary scalars used for warnings and cross-run aggregation ---
    def _obs_esss(name):
        return [lane[name]["ess"] for lane in lane_conv
                if np.isfinite(lane[name]["ess"])]

    def _obs_taus(name):
        return [lane[name]["tau_int"] for lane in lane_conv
                if np.isfinite(lane[name]["tau_int"])]

    ess_contacts = _obs_esss("contacts")
    drift_contacts = [abs(lane["contacts"]["drift_in_std"]) for lane in lane_conv
                      if np.isfinite(lane["contacts"]["drift_in_std"])]
    coverage = [w["fraction_visited"] for w in walker_diag]
    rt_low = [w["n_round_trips_low"] for w in walker_diag]
    total_round_trips = int(sum(rt_low))

    summary = {
        "n_temperatures": int(nT),
        "n_walkers": int(n_walkers),
        "total_round_trips_low": total_round_trips,
        "min_round_trips_per_walker": int(min(rt_low)) if rt_low else 0,
        "median_round_trips_per_walker": float(np.median(rt_low)) if rt_low else 0.0,
        "min_temp_coverage": float(min(coverage)) if coverage else float("nan"),
        "median_temp_coverage": float(np.median(coverage)) if coverage else float("nan"),
        "min_ess_contacts": float(min(ess_contacts)) if ess_contacts else float("nan"),
        "median_ess_contacts": float(np.median(ess_contacts)) if ess_contacts else float("nan"),
        "total_ess_contacts": float(sum(ess_contacts)) if ess_contacts else float("nan"),
        "max_autocorr_contacts": float(max(_obs_taus("contacts"))) if _obs_taus("contacts") else float("nan"),
        "max_autocorr_rg": float(max(_obs_taus("rg"))) if _obs_taus("rg") else float("nan"),
        "max_drift_contacts": float(max(drift_contacts)) if drift_contacts else float("nan"),
        "min_swap_rate": float(finite_swap.min()) if finite_swap.size else float("nan"),
        "median_swap_rate": float(np.median(finite_swap)) if finite_swap.size else float("nan"),
    }

    # --- threshold-driven warnings (structured + recorded) ---
    warnings = []

    def _warn(kind, message, value, threshold):
        warnings.append({"type": kind, "message": message,
                         "value": (None if value is None or (isinstance(value, float)
                                   and not math.isfinite(value)) else value),
                         "threshold": threshold})

    if total_round_trips < int(thr["min_round_trips"]):
        _warn("round_trips",
              f"total low->high->low round trips {total_round_trips} < "
              f"{thr['min_round_trips']}",
              total_round_trips, thr["min_round_trips"])
    if coverage and min(coverage) < float(thr["min_temp_coverage"]):
        _warn("temperature_coverage",
              f"minimum walker temperature coverage {min(coverage):.3f} < "
              f"{thr['min_temp_coverage']}",
              float(min(coverage)), thr["min_temp_coverage"])
    if ess_contacts and min(ess_contacts) < float(thr["min_ess"]):
        _warn("low_ess",
              f"minimum contact ESS {min(ess_contacts):.1f} < {thr['min_ess']}",
              float(min(ess_contacts)), thr["min_ess"])
    if drift_contacts and max(drift_contacts) > float(thr["max_drift"]):
        _warn("drift",
              f"maximum |contact drift|/std {max(drift_contacts):.3f} > "
              f"{thr['max_drift']}",
              float(max(drift_contacts)), thr["max_drift"])
    if finite_swap.size and float(finite_swap.min()) < float(thr["min_swap_rate"]):
        _warn("swap_rate",
              f"minimum adjacent swap rate {float(finite_swap.min()):.3f} < "
              f"{thr['min_swap_rate']}",
              float(finite_swap.min()), thr["min_swap_rate"])

    return {
        "thresholds": {k: thr[k] for k in DEFAULT_DIAG_THRESHOLDS},
        "n_blocks": int(n_blocks),
        "burnin_frac": float(burnin_frac),
        "summary": summary,
        "swap_rates": swap_rates,
        "lane_convergence": lane_conv,
        "walkers": walker_diag,
        "warnings": warnings,
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
# Saving: diagnostics (separate files; canonical outputs untouched)
# ---------------------------------------------------------------------------

def save_diagnostics_json(diagnostics: dict, out_prefix: str) -> str:
    path = f"{out_prefix}_diagnostics.json"
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(_json_safe(diagnostics), fh, indent=2, allow_nan=False)
    print(f"Saved {path}")
    return path


CONVERGENCE_CSV_COLUMNS = [
    "temperature_index", "temperature", "observable", "n_samples", "mean", "std",
    "tau_int", "ess", "drift", "drift_in_std", "block_mean_std",
    "block_mean_range_over_std", "acf_method",
]


def save_convergence_csv(diagnostics: dict, out_prefix: str) -> str:
    path = f"{out_prefix}_convergence.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(CONVERGENCE_CSV_COLUMNS)
        for lane in diagnostics["lane_convergence"]:
            for obs in ("contacts", "rg", "energy"):
                d = lane[obs]
                w.writerow([
                    lane["temperature_index"], lane["temperature"], obs,
                    d["n_samples"], d["mean"], d["std"], d["tau_int"], d["ess"],
                    d["drift"], d["drift_in_std"], d["block_mean_std"],
                    d["block_mean_range_over_std"], d["acf_method"],
                ])
    print(f"Saved {path}")
    return path


ROUND_TRIPS_CSV_COLUMNS = [
    "walker", "fraction_visited", "n_visited", "first_passage_low_to_high",
    "first_passage_high_to_low", "n_round_trips_low", "n_round_trips_high",
    "n_round_trips", "mean_round_trip_duration",
]


def save_round_trips_csv(diagnostics: dict, out_prefix: str) -> str:
    path = f"{out_prefix}_round_trips.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(ROUND_TRIPS_CSV_COLUMNS)
        for wd in diagnostics["walkers"]:
            w.writerow([
                wd["walker"], wd["fraction_visited"], wd["n_visited"],
                wd["first_passage_low_to_high"], wd["first_passage_high_to_low"],
                wd["n_round_trips_low"], wd["n_round_trips_high"],
                wd["n_round_trips"], wd["mean_round_trip_duration"],
            ])
    print(f"Saved {path}")
    return path


def save_walker_occupancy_csv(diagnostics: dict, out_prefix: str) -> str:
    path = f"{out_prefix}_walker_occupancy.csv"
    nT = int(diagnostics["summary"]["n_temperatures"])
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        header = ["walker", "fraction_visited"] + [
            f"occupancy_T{i}" for i in range(nT)
        ]
        w.writerow(header)
        for wd in diagnostics["walkers"]:
            occ = list(wd["occupancy"])
            occ = occ + [0] * (nT - len(occ))
            w.writerow([wd["walker"], wd["fraction_visited"]]
                       + [int(v) for v in occ[:nT]])
    print(f"Saved {path}")
    return path


def save_diagnostic_trajectories_npz(
    replicas: list,
    walker_temp_index: np.ndarray,
    Ts: np.ndarray,
    burnin_frac: float,
    out_prefix: str,
    rg_scale: float = 1.0,
) -> str:
    """Save compressed post-burn-in C/Rg/E and walker temperature-index traces.

    Arrays are stored as compact NumPy arrays (not Python objects):
        contacts_post  (nT, n_post)
        rg_post        (nT, n_post)   in output units (rg_scale applied)
        energy_post    (nT, n_post)
        walker_temp_index_post (n_post, n_walkers)
    """
    path = f"{out_prefix}_diagnostic_trajectories.npz"
    n_cycles = max((len(r.C_traj) for r in replicas), default=0)
    s = _post_burnin_slice(n_cycles, burnin_frac)
    C = np.array([np.asarray(r.C_traj[s:], dtype=np.float32) for r in replicas])
    E = np.array([np.asarray(r.E_traj[s:], dtype=np.float32) for r in replicas])
    Rg = np.array(
        [np.asarray(r.Rg_traj[s:], dtype=np.float32) * np.float32(rg_scale)
         for r in replicas]
    )
    wti = np.asarray(walker_temp_index)
    wti_post = wti[s:].astype(np.int16) if wti.ndim == 2 else wti
    np.savez_compressed(
        path,
        Ts=np.asarray(Ts, dtype=float),
        burnin_start_cycle=int(s),
        contacts_post=C,
        rg_post=Rg,
        energy_post=E,
        walker_temp_index_post=wti_post,
    )
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

    # -----------------------------------------------------------------------
    # Part 3 regressions: temperature loading, ladder validation, api-version
    # -----------------------------------------------------------------------
    # Exact temperature loading from 'temps'
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "from_temps.npz")
        np.savez(p, temps=np.array([260.0, 280.0, 300.0, 320.0]))
        Ts_loaded = load_temperatures_from_npz(p)
        np.testing.assert_allclose(Ts_loaded, [260.0, 280.0, 300.0, 320.0])
        # Exact temperature loading from 'Ts' (fallback key)
        p2 = os.path.join(tmp, "from_Ts.npz")
        np.savez(p2, Ts=np.array([250.0, 275.0, 305.0]))
        np.testing.assert_allclose(load_temperatures_from_npz(p2), [250.0, 275.0, 305.0])
        # Missing both keys -> error
        p3 = os.path.join(tmp, "no_temps.npz")
        np.savez(p3, foo=np.array([1.0, 2.0]))
        _expect_value_error(
            lambda: load_temperatures_from_npz(p3), "neither", "missing temps key"
        )
    print("  quick-test temps-from-npz (temps/Ts): PASSED")

    # Ladder validation: reject non-increasing, duplicates, too-short, nonpositive
    _expect_value_error(
        lambda: validate_temperature_ladder(np.array([300.0, 280.0, 320.0])),
        "increasing", "non-increasing ladder",
    )
    _expect_value_error(
        lambda: validate_temperature_ladder(np.array([300.0, 300.0])),
        "increasing", "duplicate ladder",
    )
    _expect_value_error(
        lambda: validate_temperature_ladder(np.array([300.0])),
        "at least two", "too-short ladder",
    )
    _expect_value_error(
        lambda: validate_temperature_ladder(np.array([-1.0, 300.0])),
        "positive", "nonpositive ladder",
    )
    np.testing.assert_allclose(
        validate_temperature_ladder(np.array([260.0, 280.0, 300.0])),
        [260.0, 280.0, 300.0],
    )
    print("  quick-test temperature-ladder validation: PASSED")

    # Summary model_api_version handling: absent OK, future version rejected
    with tempfile.TemporaryDirectory() as tmp:
        base = {
            "model": "poly2",
            "params": {"a0": 0.1, "a1": -0.5, "a2": 0.03},
            "Tref": 300.0, "Tscale": 80.0,
        }
        p_nover = _write_summary(tmp, base, name="no_ver.json")
        assert load_fit_summary_json(p_nover)["model_name"] == "poly2"
        p_curr = _write_summary(
            tmp, {**base, "model_api_version": MODEL_API_VERSION}, name="cur.json"
        )
        assert load_fit_summary_json(p_curr)["model_name"] == "poly2"
        p_future = _write_summary(
            tmp, {**base, "model_api_version": MODEL_API_VERSION + 1}, name="fut.json"
        )
        _expect_value_error(
            lambda: load_fit_summary_json(p_future), "newer", "future api version"
        )
    print("  quick-test summary api-version handling: PASSED")

    run_diagnostics_quick_test()

    print("quick-test complete.")


def run_diagnostics_quick_test() -> None:
    """Unit + smoke tests for the convergence/mixing diagnostics."""
    import os
    import tempfile

    rng = np.random.RandomState(0)

    # --- Integrated autocorrelation time on IID noise: tau ~ 1, ESS ~ n ----
    iid = rng.standard_normal(4000)
    ac = integrated_autocorr_time(iid)
    assert 0.5 < ac["tau_int"] < 2.0, ac
    assert ac["ess"] > 0.4 * iid.size, ac
    assert ac["method"] == "geyer_initial_positive_sequence"

    # --- AR(1): tau_int should approach (1+phi)/(1-phi) ---------------------
    phi = 0.8
    n = 60000
    x = np.empty(n)
    x[0] = 0.0
    noise = rng.standard_normal(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + noise[t]
    ac_ar = integrated_autocorr_time(x)
    tau_expected = (1.0 + phi) / (1.0 - phi)  # = 9.0
    assert abs(ac_ar["tau_int"] - tau_expected) < 0.25 * tau_expected, (
        ac_ar, tau_expected
    )
    assert ac_ar["ess"] < 0.5 * n, ac_ar  # correlated -> far fewer eff. samples

    # --- constant series is reported cleanly --------------------------------
    ac_const = integrated_autocorr_time(np.full(100, 3.0))
    assert ac_const["method"] == "constant_series"
    assert ac_const["tau_int"] == 1.0 and ac_const["ess"] == 100.0
    print("  diagnostics quick-test autocorr/ESS (IID, AR1, constant): PASSED")

    # --- drift / block stability --------------------------------------------
    drift_series = np.concatenate([np.zeros(500), np.ones(500)])
    dr = early_late_drift(drift_series)
    assert abs(dr["drift"] - 1.0) < 1e-9, dr
    assert dr["drift_in_std"] > 0.5, dr
    bl = block_mean_stability(np.arange(100, dtype=float), n_blocks=5)
    assert len(bl["block_means"]) == 5 and bl["block_mean_std"] > 0
    print("  diagnostics quick-test drift/block stability: PASSED")

    # --- walker analysis: known round trips ---------------------------------
    nT = 4
    # Build a path: bottom -> top -> bottom -> top -> bottom = 2 L->H->L trips.
    up = list(range(0, nT))            # 0,1,2,3
    down = list(range(nT - 1, -1, -1))  # 3,2,1,0
    path = np.array([0] + up[1:] + down[1:] + up[1:] + down[1:], dtype=int)
    wd = analyze_walker_trajectory(path, nT)
    assert wd["n_round_trips_low"] == 2, wd
    assert wd["n_round_trips_high"] == 1, wd
    assert wd["fraction_visited"] == 1.0, wd
    assert wd["first_passage_low_to_high"] is not None
    assert wd["mean_round_trip_duration"] is not None

    # --- walker analysis: no round trips (stuck near the bottom) ------------
    stuck = np.array([0, 0, 1, 0, 1, 1, 0, 0], dtype=int)  # never reaches top
    wd0 = analyze_walker_trajectory(stuck, nT)
    assert wd0["n_round_trips_low"] == 0 and wd0["n_round_trips"] == 0, wd0
    assert wd0["first_passage_low_to_high"] is None, wd0
    assert wd0["fraction_visited"] < 1.0, wd0
    print("  diagnostics quick-test walker round trips (known/none): PASSED")

    # --- deterministic 1-worker vs 2-worker diagnostics smoke ---------------
    Ts = np.linspace(300.0, 360.0, 5)
    hs_params = [378.96, 1.39686]
    Tref = 0.5 * (float(Ts.min()) + float(Ts.max()))
    Tscale = max(float(Ts.max()) - float(Ts.min()), 1.0)
    thresholds = dict(DEFAULT_DIAG_THRESHOLDS)
    for n_workers in (1, 2):
        diag_store: dict = {}
        reps, sp, sa = run_remd(
            N=18, Ts=Ts, steps_per_swap=40, n_cycles=60,
            model_name="hs", params=hs_params, Tref=Tref, Tscale=Tscale,
            seed=3, n_workers=n_workers, verbose=False,
            diagnostics=True, diag_store=diag_store,
        )
        wti = diag_store["walker_temp_index"]
        assert wti.shape == (60, len(Ts)), wti.shape
        # Each cycle's walker->lane mapping must be a permutation of 0..nT-1.
        for c in range(wti.shape[0]):
            assert sorted(wti[c].tolist()) == list(range(len(Ts))), c
        diag = compute_run_diagnostics(
            reps, sp, sa, wti, Ts, burnin_frac=0.5, n_blocks=4,
            thresholds=thresholds, rg_scale=1.0,
        )
        assert diag["summary"]["n_walkers"] == len(Ts)
        assert len(diag["lane_convergence"]) == len(Ts)
        assert len(diag["walkers"]) == len(Ts)
        # Occupancy histograms must sum to the number of cycles for each walker.
        for w in diag["walkers"]:
            assert int(np.sum(w["occupancy"])) == wti.shape[0]
        # Round-trip count is the conservation: total cycles spent at each lane.
        lane_totals = np.zeros(len(Ts), dtype=int)
        for w in diag["walkers"]:
            lane_totals += np.asarray(w["occupancy"], dtype=int)
        assert int(lane_totals.sum()) == wti.size

        # Saving the diagnostics files round-trips through JSON (no NaN/Inf).
        with tempfile.TemporaryDirectory() as tmp:
            prefix = os.path.join(tmp, "run")
            save_diagnostics_json(diag, prefix)
            save_convergence_csv(diag, prefix)
            save_round_trips_csv(diag, prefix)
            save_walker_occupancy_csv(diag, prefix)
            save_diagnostic_trajectories_npz(
                reps, wti, Ts, burnin_frac=0.5, out_prefix=prefix
            )
            for suffix in ("_diagnostics.json", "_convergence.csv",
                           "_round_trips.csv", "_walker_occupancy.csv",
                           "_diagnostic_trajectories.npz"):
                assert os.path.exists(prefix + suffix), suffix
            reloaded = json.loads(
                open(prefix + "_diagnostics.json", encoding="utf-8").read()
            )
            assert "summary" in reloaded and "warnings" in reloaded
        print(f"  diagnostics quick-test smoke n_workers={n_workers}: PASSED")


# ---------------------------------------------------------------------------
# Model parameter resolution
# ---------------------------------------------------------------------------

def validate_temperature_ladder(Ts: np.ndarray) -> np.ndarray:
    """Validate an explicit REMD temperature ladder.

    Requires a one-dimensional array of at least two finite, positive,
    strictly increasing temperatures with no duplicates.  Returns the array as
    float64.
    """
    Ts = np.asarray(Ts, dtype=float)
    if Ts.ndim != 1:
        raise ValueError("Temperature ladder must be one-dimensional")
    if Ts.size < 2:
        raise ValueError("Temperature ladder must have at least two entries")
    if not np.all(np.isfinite(Ts)):
        raise ValueError("Temperature ladder must contain only finite values")
    if not np.all(Ts > 0.0):
        raise ValueError("Temperature ladder must contain only positive temperatures")
    diffs = np.diff(Ts)
    if not np.all(diffs > 0.0):
        raise ValueError(
            "Temperature ladder must be strictly increasing with no duplicates"
        )
    return Ts


def load_temperatures_from_npz(path: str) -> np.ndarray:
    """Load an exact temperature ladder from an NPZ, trying 'temps' then 'Ts'."""
    with np.load(path) as data:
        for key in ("temps", "Ts"):
            if key in data:
                return validate_temperature_ladder(np.asarray(data[key], dtype=float))
        raise ValueError(
            f"{path!r} contains neither 'temps' nor 'Ts' temperature key "
            f"(found: {list(data.keys())})"
        )


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

    # Model-API-version handshake: absent is accepted for backward compatibility;
    # a version newer than we support is a hard error.
    api_version = summary.get("model_api_version", None)
    if api_version is not None:
        try:
            api_version_int = int(api_version)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"fit_summary.json {path!r} has non-integer "
                f"model_api_version: {api_version!r}"
            ) from exc
        if api_version_int > MODEL_API_VERSION:
            raise ValueError(
                f"fit_summary.json {path!r} model_api_version "
                f"{api_version_int} is newer than supported version "
                f"{MODEL_API_VERSION}. Update remd_uniform_chain_new.py."
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


def attach_run_metadata(
    dist: dict,
    *,
    seed: int,
    N: int,
    steps_per_swap: int,
    n_cycles: int,
    burnin_frac: float,
    n_workers: int,
) -> dict:
    """Inject simulation provenance into a distributions dict in place.

    Adds run/seed metadata and the model API version without removing or
    altering any existing canonical or model-metadata keys.
    """
    dist["seed"] = int(seed)
    dist["N"] = int(N)
    dist["steps_per_swap"] = int(steps_per_swap)
    dist["n_cycles"] = int(n_cycles)
    dist["burnin_frac"] = float(burnin_frac)
    dist["n_workers"] = int(n_workers)
    dist["model_api_version"] = int(MODEL_API_VERSION)
    return dist


def _json_safe(obj):
    """Recursively convert NumPy scalars/arrays to plain Python for json.dump."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_json_safe(v) for v in obj.tolist()]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        value = float(obj)
        return value if math.isfinite(value) else None
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def save_run_summary(summary: dict, path: str) -> str:
    """Write the REMD run-summary JSON (NumPy-safe)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(_json_safe(summary), fh, indent=2, allow_nan=False)
    print(f"Saved {path}")
    return path


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
    temp_group = ap.add_mutually_exclusive_group()
    temp_group.add_argument(
        "--temps-from-npz",
        type=str,
        default=None,
        dest="temps_from_npz",
        help=(
            "Load the exact REMD temperature ladder from an NPZ file. "
            "The loader checks keys 'temps' and then 'Ts'. "
            "When supplied, this overrides Tmin, Tmax, and nT."
        ),
    )
    temp_group.add_argument(
        "--temps",
        type=str,
        default=None,
        help="Explicit comma-separated temperature ladder.",
    )
    ap.add_argument(
        "--no-plots",
        action="store_true",
        dest="no_plots",
        help="Skip all plot generation (CSV and NPZ outputs are still written).",
    )
    ap.add_argument(
        "--run-summary-json",
        type=str,
        default=None,
        dest="run_summary_json",
        help=(
            "Optional path for the REMD run summary JSON. "
            "Default: <out-prefix>_run_summary.json."
        ),
    )
    ap.add_argument("--timing",         action="store_true",         help="print sweep/swap/total wall times")
    ap.add_argument("--quick-test",     action="store_true",         help="run smoke-test and exit")

    # --- Optional convergence/mixing diagnostics (off by default) -----------
    ap.add_argument(
        "--diagnostics", action="store_true", dest="diagnostics",
        help=(
            "Compute and save REMD convergence/mixing diagnostics "
            "(walker round trips, ESS/autocorrelation, drift, block stability). "
            "Off by default; canonical outputs are unchanged either way."
        ),
    )
    ap.add_argument(
        "--diagnostic-trajectories", action="store_true",
        dest="diagnostic_trajectories",
        help=(
            "Also save post-burn-in C/Rg/E and walker temperature-index traces "
            "to <out-prefix>_diagnostic_trajectories.npz (enables cross-seed "
            "Rhat downstream). Requires --diagnostics."
        ),
    )
    ap.add_argument("--diag-n-blocks", type=int, default=DEFAULT_DIAG_N_BLOCKS,
                    dest="diag_n_blocks",
                    help="number of blocks for block-mean stability")
    ap.add_argument("--diag-min-round-trips", type=int,
                    default=DEFAULT_DIAG_THRESHOLDS["min_round_trips"],
                    dest="diag_min_round_trips",
                    help="warn if total low->high->low round trips is below this")
    ap.add_argument("--diag-min-temp-coverage", type=float,
                    default=DEFAULT_DIAG_THRESHOLDS["min_temp_coverage"],
                    dest="diag_min_temp_coverage",
                    help="warn if any walker visits less than this fraction of T")
    ap.add_argument("--diag-min-ess", type=float,
                    default=DEFAULT_DIAG_THRESHOLDS["min_ess"],
                    dest="diag_min_ess",
                    help="warn if any lane/observable ESS is below this")
    ap.add_argument("--diag-max-drift", type=float,
                    default=DEFAULT_DIAG_THRESHOLDS["max_drift"],
                    dest="diag_max_drift",
                    help="warn if |early-late drift|/std exceeds this")
    ap.add_argument("--diag-min-swap-rate", type=float,
                    default=DEFAULT_DIAG_THRESHOLDS["min_swap_rate"],
                    dest="diag_min_swap_rate",
                    help="warn if any adjacent swap rate is below this")
    args = ap.parse_args()

    if args.N < 3:
        raise ValueError("--N must be >= 3")
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
    if not math.isfinite(args.rg_scale) or args.rg_scale <= 0:
        raise ValueError("--rg-scale must be finite and positive")
    if args.diagnostic_trajectories and not args.diagnostics:
        raise ValueError("--diagnostic-trajectories requires --diagnostics")
    if args.diagnostics and args.diag_n_blocks < 1:
        raise ValueError("--diag-n-blocks must be >= 1")

    if args.quick_test:
        run_quick_test()
        return

    # Standalone runs should be able to use a nested output prefix without
    # requiring the caller to create the directory first. The suite already
    # creates it, so this is harmless there.
    Path(args.out_prefix).parent.mkdir(parents=True, exist_ok=True)

    # Temperature ladder resolution: --temps-from-npz > --temps > linspace.
    if args.temps_from_npz is not None:
        Ts = load_temperatures_from_npz(args.temps_from_npz)
        temp_source = f"npz:{args.temps_from_npz}"
    elif args.temps is not None:
        Ts = validate_temperature_ladder(
            np.array(parse_params_string(args.temps), dtype=float)
        )
        temp_source = "cli:--temps"
    else:
        if args.nT < 2:
            raise ValueError("--nT must be >= 2")
        if not math.isfinite(args.Tmin) or not math.isfinite(args.Tmax):
            raise ValueError("Temperatures must be finite")
        if args.Tmin <= 0 or args.Tmax <= 0:
            raise ValueError("Temperatures must be positive")
        if args.Tmax <= args.Tmin:
            raise ValueError("--Tmax must be greater than --Tmin")
        Ts = np.linspace(args.Tmin, args.Tmax, args.nT)
        temp_source = "linspace"

    nT = len(Ts)
    Tmin_resolved, Tmax_resolved = float(Ts.min()), float(Ts.max())
    diffs = np.diff(Ts)
    temperature_uniform = bool(nT >= 2 and np.allclose(diffs, diffs[0]))
    total_steps = args.steps_per_swap * args.n_cycles
    print(
        f"Temperature ladder ({temp_source}): {nT} replicas, "
        f"min={Tmin_resolved:.6g}, max={Tmax_resolved:.6g}, "
        f"uniform={temperature_uniform}"
    )

    (
        model_name, model_params, param_names, Tref, Tscale,
        parameter_source, fit_summary_json,
    ) = resolve_model_params(args, Ts)

    print(
        f"REMD: {nT} replicas, T in [{Tmin_resolved:.6g}, {Tmax_resolved:.6g}], "
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

    diag_store: dict = {}
    t_run_start = time.perf_counter()
    replicas, swap_props, swap_accs = run_remd(
        N=args.N, Ts=Ts,
        steps_per_swap=args.steps_per_swap,
        n_cycles=args.n_cycles,
        model_name=model_name, params=model_params,
        Tref=Tref, Tscale=Tscale,
        seed=args.seed, verbose=True,
        n_workers=args.n_workers,
        timing=args.timing,
        diagnostics=args.diagnostics,
        diag_store=diag_store,
    )
    wall_time_seconds = time.perf_counter() - t_run_start

    print("\nSwap acceptance rates by pair:")
    swap_rates = []
    for k in range(len(swap_props)):
        rate = (
            float(swap_accs[k] / swap_props[k])
            if swap_props[k] > 0 else float("nan")
        )
        swap_rates.append(float(rate))
        rate_text = f"{rate:.3f}" if math.isfinite(rate) else "n/a"
        print(
            f"  T={Ts[k]:.1f} <-> T={Ts[k+1]:.1f}  "
            f"{swap_accs[k]}/{swap_props[k]} = {rate_text}"
        )

    results = compute_statistics(
        replicas, burnin_frac=args.burnin_frac, rg_scale=args.rg_scale
    )
    local_acceptance_rates = [float(r["local_acc_rate"]) for r in results]
    dist    = build_distributions(
        replicas, rg_bins=args.rg_bins, burnin_frac=args.burnin_frac,
        rg_scale=args.rg_scale,
    )
    attach_model_metadata(
        dist, model_name, param_names, model_params, Tref, Tscale,
        parameter_source=parameter_source, fit_summary_json=fit_summary_json,
    )
    attach_run_metadata(
        dist,
        seed=args.seed, N=args.N,
        steps_per_swap=args.steps_per_swap, n_cycles=args.n_cycles,
        burnin_frac=args.burnin_frac, n_workers=args.n_workers,
    )

    results_path = save_results_csv(results, args.out_prefix)
    swap_path = save_swap_csv(swap_props, swap_accs, Ts, args.out_prefix)
    dist_path = save_distributions(dist, args.out_prefix)
    output_files = {
        "results_csv": results_path,
        "swap_rates_csv": swap_path,
        "distributions_npz": dist_path,
    }

    # --- Optional diagnostics (computed/saved separately from canonical output) -
    diagnostics_result = None
    diagnostics_overhead_seconds = 0.0
    if args.diagnostics:
        t_diag = time.perf_counter()
        thresholds = {
            "min_round_trips": args.diag_min_round_trips,
            "min_temp_coverage": args.diag_min_temp_coverage,
            "min_ess": args.diag_min_ess,
            "max_drift": args.diag_max_drift,
            "min_swap_rate": args.diag_min_swap_rate,
        }
        walker_temp_index = diag_store.get("walker_temp_index")
        if walker_temp_index is None:
            raise RuntimeError(
                "diagnostics requested but walker trajectory was not recorded"
            )
        diagnostics_result = compute_run_diagnostics(
            replicas, swap_props, swap_accs, walker_temp_index, Ts,
            burnin_frac=args.burnin_frac, n_blocks=args.diag_n_blocks,
            thresholds=thresholds, rg_scale=args.rg_scale,
        )
        output_files["diagnostics_json"] = save_diagnostics_json(
            diagnostics_result, args.out_prefix
        )
        output_files["convergence_csv"] = save_convergence_csv(
            diagnostics_result, args.out_prefix
        )
        output_files["round_trips_csv"] = save_round_trips_csv(
            diagnostics_result, args.out_prefix
        )
        output_files["walker_occupancy_csv"] = save_walker_occupancy_csv(
            diagnostics_result, args.out_prefix
        )
        if args.diagnostic_trajectories:
            output_files["diagnostic_trajectories_npz"] = (
                save_diagnostic_trajectories_npz(
                    replicas, walker_temp_index, Ts,
                    burnin_frac=args.burnin_frac, out_prefix=args.out_prefix,
                    rg_scale=args.rg_scale,
                )
            )
        diagnostics_overhead_seconds = time.perf_counter() - t_diag
        for w in diagnostics_result["warnings"]:
            print(f"  [diagnostic warning] {w['message']}")
        print(
            f"Diagnostics: {len(diagnostics_result['warnings'])} warning(s); "
            f"computed in {diagnostics_overhead_seconds:.2f}s"
        )

    if not args.no_plots:
        if plt is None or cm is None or mcolors is None:
            raise RuntimeError(
                "matplotlib is required for plots; install it or use --no-plots"
            )
        plot_observables(results, args.out_prefix)
        plot_distributions(dist, args.out_prefix)

    swap_rates_arr = np.array(swap_rates, dtype=float) if swap_rates else np.array([])
    swap_rates_finite = swap_rates_arr[np.isfinite(swap_rates_arr)]
    run_summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
        "model_api_version": MODEL_API_VERSION,
        "model": model_name,
        "param_names": list(param_names),
        "params": [float(v) for v in model_params],
        "Tref": float(Tref),
        "Tscale": float(Tscale),
        "parameter_source": parameter_source,
        "fit_summary_json": fit_summary_json,
        "temperatures": [float(t) for t in Ts],
        "temperature_count": int(nT),
        "temperature_min": float(Tmin_resolved),
        "temperature_max": float(Tmax_resolved),
        "temperature_uniform": temperature_uniform,
        "temperature_source": temp_source,
        "N": int(args.N),
        "steps_per_swap": int(args.steps_per_swap),
        "n_cycles": int(args.n_cycles),
        "total_steps_per_replica": int(total_steps),
        "seed": int(args.seed),
        "n_workers": int(args.n_workers),
        "burnin_frac": float(args.burnin_frac),
        "rg_bins": int(args.rg_bins),
        "rg_scale": float(args.rg_scale),
        "wall_time_seconds": float(wall_time_seconds),
        "swap_rates": swap_rates,
        "swap_rate_min": float(swap_rates_finite.min()) if swap_rates_finite.size else None,
        "swap_rate_mean": float(swap_rates_finite.mean()) if swap_rates_finite.size else None,
        "swap_rate_median": float(np.median(swap_rates_finite)) if swap_rates_finite.size else None,
        "local_acceptance_rates": local_acceptance_rates,
        "output_files": output_files,
        "diagnostics_enabled": bool(args.diagnostics),
    }
    if diagnostics_result is not None:
        run_summary["diagnostics_overhead_seconds"] = float(
            diagnostics_overhead_seconds
        )
        run_summary["diagnostics_summary"] = diagnostics_result["summary"]
        run_summary["diagnostics_thresholds"] = diagnostics_result["thresholds"]
        run_summary["diagnostics_warnings"] = diagnostics_result["warnings"]
    if model_name == "heat_capacity":
        run_summary["T0"] = float(Tref)

    run_summary_path = (
        args.run_summary_json
        if args.run_summary_json is not None
        else f"{args.out_prefix}_run_summary.json"
    )
    output_files["run_summary_json"] = run_summary_path
    save_run_summary(run_summary, run_summary_path)


if __name__ == "__main__":
    main()
