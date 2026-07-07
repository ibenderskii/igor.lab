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
    python remd_uniform_chain_2_new.py --N 50 --nT 8 --Tmin 280 --Tmax 380 \\
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

Chain-length convention
-----------------------
``--N`` is the NUMBER OF BEADS (unchanged).  Every new output records both
``n_beads = N`` and ``n_steps = N - 1`` so downstream tools cannot make an
off-by-one error.  ``N`` is preserved in the NPZ for backward compatibility and
means ``n_beads``.

Canonical vs optional outputs
-----------------------------
Canonical (always written, backward compatible):
    <prefix>_results.csv, <prefix>_swap_rates.csv,
    <prefix>_distributions.npz, <prefix>_run_summary.json
Always written (new, additive — small):
    <prefix>_move_acceptance.csv   (per-lane, per-move-type accept breakdown)
Optional structural diagnostics (opt-in flags):
    <prefix>_configurations.h5            (--save-configurations)
    <prefix>_diagnostic_trajectories.npz (--diagnostics --diagnostic-trajectories;
                                          also carries structural traces)
    plus the existing diagnostics CSV/JSON under --diagnostics.

Structural observables (new; opt-in cadence)
--------------------------------------------
Scalar m, Rg, Rg^2, Ree^2 are recorded EVERY cycle.  Contact-map-derived
observables (the full contour-separation vector m_r, m_long, S_max, and the
largest-component fraction) are recorded every ``--structural-stride`` cycles
and use a SEPARATE trajectory length; the two cadences are never aligned by
index.  ``m_r`` is the authoritative contour-separation representation; the
short/medium/long bins are a convenience derived from it, and their definitions
are stored in the output metadata (``structural_bin_definitions``).  Full
pair-motif (disjoint/nested/interleaved/shared-endpoint) classification is
O(m^2) and runs OFFLINE in extract_contact_motif_features.py, never per cycle.

Sampling identities (do not conflate)
-------------------------------------
* Temperature lane: a fixed temperature slot (array index along the ladder).
* Walker identity:  a configuration that migrates between lanes via swaps.
* Independent seed: a SEPARATE REMD run.  Eight seeds = eight separate runs;
  ``--n-workers`` only parallelizes the lanes of ONE run and produces no
  independent seeds.
* Multiprocessing worker: an OS process evolving one lane's local sweeps for a
  cycle; it returns only the chain state and bounded counters.

Saved snapshots are correlated.  Use the integrated autocorrelation time / ESS
reported in the diagnostics (NOT the raw snapshot count) to estimate the number
of statistically independent configurations.

Contact susceptibility Var(m) and Cov(A, m) are model-independent diagnostics
and are generally valid.  The thermodynamic readings U = h<m> and
C_V = h^2 Var(m)/T^2 are PHYSICAL only under the constant-(h, s) model; they are
not labeled physical heat capacity for effective polynomial b(T) models.

Example commands (PowerShell)
-----------------------------
The canonical production script name is ``remd_uniform_chain_2_new.py``
(``remd_uniform_chain_2.py`` is a thin compatibility SHIM that re-exports this
module by object identity -- NOT a byte-identical copy -- kept for the existing
suite/test imports).

Quick test:
    python .\\remd_uniform_chain_2_new.py --quick-test

Short structural smoke test:
    python .\\remd_uniform_chain_2_new.py `
      --N 30 --Tmin 280 --Tmax 360 --nT 8 `
      --steps-per-swap 50 --n-cycles 100 --n-workers 2 `
      --diagnostics --diagnostic-trajectories `
      --structural-observables --structural-stride 1 `
      --save-configurations --snapshot-stride 5 `
      --out-prefix .\\test_outputs\\remd_structural_test

Offline feature extraction (reads the authoritative saved coordinates):
    python .\\extract_contact_motif_features.py `
      --input .\\test_outputs\\remd_structural_test_configurations.h5 `
      --output .\\test_outputs\\remd_structural_test_features.h5 --validate
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import math
import random
import time
import multiprocessing as _mp
from concurrent.futures import ProcessPoolExecutor
from itertools import permutations, product
from pathlib import Path
from typing import List, Tuple

import numpy as np

# Reusable, REMD-independent structural observables (contact maps, geometry,
# contact graph, pair motifs, contour-separation bins).  Co-located with this
# script; the script's own directory is on sys.path[0] when run directly, but
# add it explicitly so the module also imports cleanly when this file is
# imported from another working directory.
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import isaw_contact_observables as ico  # noqa: E402
import isaw_config_io as cio  # noqa: E402
from isaw_config_io import SnapshotWriter  # noqa: E402,F401  (run_remd annotation)

# Matplotlib is imported LAZILY inside the plotting functions only.  Importing
# it at module import time would initialize a GUI/Agg backend in every spawned
# worker process (the module is re-imported under the 'spawn' start method) and
# risks fork-related state on POSIX; keeping the REMD/worker import path free of
# matplotlib avoids that.  ``_import_matplotlib`` returns (cm, mcolors, plt) or
# (None, None, None) when matplotlib is unavailable.
def _import_matplotlib():
    try:
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors
        import matplotlib.pyplot as plt
        return cm, mcolors, plt
    except Exception:  # pragma: no cover - depends on local plotting stack
        return None, None, None


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

# Output-schema version for the distributions NPZ / run summary / snapshot files.
# Bump when the set of stored keys or their semantics change.
#   v2: independent fixed/scaled contour-bin schemes; move counters gain a
#       state_changing column (null moves excluded from acceptance);
#       diagnostics report tau_int_samples/tau_int_cycles; q stored alongside
#       authoritative K with overflow handled.
#   v3: authoritative resolved bin definitions threaded online->HDF5->extractor;
#       results CSV gains m_long_fixed_*/m_global_scaled_*/state_changing rate;
#       diagnostics gate structural warnings + dedup ESS types; configured vs
#       observed structural stride; definitions_version 1.1.0 (closed local bin).
SCHEMA_VERSION = 3


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
    """Lattice-unit Rg.  Implemented from R_g^2 for a single source of truth."""
    return math.sqrt(ico.radius_of_gyration_squared(chain))


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
MOVE_NAMES = ["pivot", "crankshaft", "end"]
# Counter columns per move type (move-resolved acceptance diagnostics).
#   proposed            : move type selected
#   geometrically_valid : proposal preserves lattice connectivity / self-avoidance
#   state_changing      : proposed chain actually differs from the current chain
#   metropolis_accepted : a STATE-CHANGING proposal accepted by Metropolis
# A null proposal (geometrically valid but identical to the current chain) is
# counted as proposed and geometrically_valid but NOT as state_changing and NOT
# as a meaningful accepted move.  This prevents null moves from inflating the
# acceptance rate and weakening move-freezing diagnostics.
MOVE_COUNTER_COLS = [
    "proposed", "geometrically_valid", "state_changing", "metropolis_accepted",
]
_N_MOVES = len(MOVE_FUNCS)
_N_MOVE_COLS = len(MOVE_COUNTER_COLS)


def new_move_counters() -> np.ndarray:
    """Zeroed (n_moves, 4) counter block: proposed/valid/state_changing/accepted."""
    return np.zeros((_N_MOVES, _N_MOVE_COLS), dtype=np.int64)


def _seed_all(seed: int) -> None:
    """Seed Python and NumPy RNGs without NumPy's unsigned-32-bit limitation."""
    random.seed(int(seed))
    np.random.seed(int(seed) % (2 ** 32))


def _parse_bool_env(name: str, default: bool = False) -> bool:
    """Parse a boolean environment variable.

    Values "0", "false", "no", "off", "" (case-insensitive) are False; "1",
    "true", "yes", "on" are True.  Any other value falls back to ``default``.
    """
    raw = _os.environ.get(name)
    if raw is None:
        return default
    v = raw.strip().lower()
    if v in ("0", "false", "no", "off", ""):
        return False
    if v in ("1", "true", "yes", "on"):
        return True
    return default


# Optional development cross-check: when ISAW_DEBUG_CONTACTS is truthy in the
# environment, mc_sweep asserts state.m == contact_count(...) after every
# accepted move.  Off by default so the expensive recount never runs in
# production.  Values like "0"/"false"/"no" explicitly disable it.
_DEBUG_CONTACTS = _parse_bool_env("ISAW_DEBUG_CONTACTS", False)


@dataclasses.dataclass
class ChainState:
    """Mutable MC state: chain positions, occupied-site set, energy, contacts.

    ``m`` is the cached non-bonded contact count, kept in sync with ``chain``/
    ``occ`` so the MC hot loop never recounts the *old* configuration.  The
    independent :func:`contact_count` remains available for validation.
    """
    chain: List[Vec]
    occ:   set
    E:     float
    m:     int

    @classmethod
    def initial_straight(
        cls, N: int, T: float, model_name: str, params, Tref: float, Tscale: float
    ) -> "ChainState":
        chain = [(i, 0, 0) for i in range(N)]
        occ   = set(chain)
        m = int(round(contact_count(chain, occ)))  # straight chain => 0
        return cls(
            chain=chain, occ=occ,
            E=energy_from_contacts(m, T, model_name, params, Tref, Tscale),
            m=m,
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
    """One thermostat lane: fixed temperature, evolving chain configuration.

    Trajectories recorded every cycle (backward compatible):
        E_traj, C_traj, Rg_traj
    Scalar structural trajectories recorded every cycle (cheap):
        Rg2_traj, Ree2_traj
    Contact-map-derived structural trajectories recorded every
    ``structural_stride`` cycles (see :func:`run_remd`):
        m_long_traj, Smax_traj, largest_component_fraction_traj,
        m_r_traj (full contour-separation vectors), structural_cycles
    ``move_counters`` is an (n_moves, 3) block of proposed/valid/accepted counts.
    """
    T:     float
    state: ChainState
    local_acc:  int = 0
    local_prop: int = 0
    E_traj:  list = dataclasses.field(default_factory=list)
    C_traj:  list = dataclasses.field(default_factory=list)
    Rg_traj: list = dataclasses.field(default_factory=list)
    # Scalar structural observables (every cycle).
    Rg2_traj:  list = dataclasses.field(default_factory=list)
    Ree2_traj: list = dataclasses.field(default_factory=list)
    # Contact-map-derived structural observables (every structural_stride cycle;
    # only populated when structural observables are explicitly enabled).
    # m_long_traj stores m_long_fixed (the fixed-scheme long count).
    m_long_traj: list = dataclasses.field(default_factory=list)
    m_global_scaled_traj: list = dataclasses.field(default_factory=list)
    Smax_traj:   list = dataclasses.field(default_factory=list)
    largest_component_fraction_traj: list = dataclasses.field(default_factory=list)
    # Full m_r vectors are retained ONLY when --save-m-r-trajectories is set.
    m_r_traj:    list = dataclasses.field(default_factory=list)
    structural_cycles: list = dataclasses.field(default_factory=list)
    # Move-resolved acceptance counters: shape (n_moves, 4).
    move_counters: np.ndarray = dataclasses.field(default_factory=new_move_counters)

    @property
    def local_acc_rate(self) -> float:
        # LEGACY metric: all accepted Metropolis outcomes / proposed (includes
        # accepted null moves).  Kept for backward compatibility.
        return self.local_acc / self.local_prop if self.local_prop else float("nan")

    @property
    def legacy_local_acceptance_rate(self) -> float:
        return self.local_acc_rate

    @property
    def state_changing_acceptance_rate(self) -> float:
        # State-changing accepted moves (move-counter column 3) / proposed.
        # Null moves are excluded; this is the freezing-diagnostic metric.
        if not self.local_prop:
            return float("nan")
        accepted_state_changing = int(np.asarray(self.move_counters)[:, 3].sum())
        return accepted_state_changing / self.local_prop


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
    counters = replica.move_counters

    for _ in range(steps):
        replica.local_prop += 1
        move_idx = random.randrange(_N_MOVES)
        counters[move_idx, 0] += 1                       # proposed
        ok, chain_new, occ_new = MOVE_FUNCS[move_idx](state.chain, state.occ)
        if not ok:
            continue
        counters[move_idx, 1] += 1                       # geometrically valid

        # A geometrically valid proposal that leaves every bead in place (e.g. a
        # pivot rotation about the tail's own axis) is a NULL move: it is tracked
        # but does not count as state-changing or as a meaningful accepted move.
        # The RNG call sequence below is unchanged by this bookkeeping.
        state_changing = chain_new != state.chain
        if state_changing:
            counters[move_idx, 2] += 1                   # state changing

        # state.m is the *old* contact count; recompute only the trial config.
        m_old = state.m
        u_old = reduced_potential(m_old, T, model_name, params, Tref, Tscale)

        m_new = int(round(contact_count(chain_new, occ_new)))
        u_new = reduced_potential(m_new, T, model_name, params, Tref, Tscale)

        du = u_new - u_old
        if du <= 0 or random.random() < math.exp(-du):
            state.chain = chain_new
            state.occ   = occ_new
            state.m     = m_new
            state.E     = energy_from_contacts(m_new, T, model_name, params, Tref, Tscale)
            replica.local_acc += 1
            if state_changing:
                counters[move_idx, 3] += 1               # accepted (state-changing)
            if _DEBUG_CONTACTS:
                assert state.m == int(round(contact_count(state.chain, state.occ))), (
                    "state.m out of sync with contact_count after accepted move"
                )


def attempt_swap(
    rep_a: Replica, rep_b: Replica,
    model_name: str, params, Tref: float, Tscale: float,
) -> bool:
    """
    Attempt a configuration swap between two adjacent replicas.

    Temperatures stay fixed; configurations (and energies) are exchanged.
    Returns True if accepted.
    """
    m_a = rep_a.state.m
    m_b = rep_b.state.m
    log_acc = swap_log_accept(
        m_a, m_b, rep_a.T, rep_b.T, model_name, params, Tref, Tscale
    )

    accepted = log_acc >= 0 or random.random() < math.exp(log_acc)
    if accepted:
        rep_a.state.chain, rep_b.state.chain = rep_b.state.chain, rep_a.state.chain
        rep_a.state.occ,   rep_b.state.occ   = rep_b.state.occ,   rep_a.state.occ
        rep_a.state.m,     rep_b.state.m     = rep_b.state.m,     rep_a.state.m
        # Recompute each lane's energy from the (now swapped) stored contact
        # number rather than recounting contacts.
        rep_a.state.E = energy_from_contacts(
            rep_a.state.m, rep_a.T, model_name, params, Tref, Tscale
        )
        rep_b.state.E = energy_from_contacts(
            rep_b.state.m, rep_b.T, model_name, params, Tref, Tscale
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
    move_counters: np.ndarray,
    steps: int,
    model_name: str,
    params: list,
    Tref: float,
    Tscale: float,
    seed: int,
) -> tuple["ChainState", int, int, np.ndarray]:
    """Deterministically sweep one lane without transferring trajectories.

    Only the mutable chain state and bounded acceptance counters (scalar
    accept/propose totals plus the small fixed-size move-counter block) cross
    the process boundary.  Sending the full Replica would also pickle its
    ever-growing trajectories every cycle, causing O(n_cycles^2) serialization
    overhead.
    """
    _seed_all(seed)
    replica = Replica(
        T=float(temperature), state=state,
        local_acc=int(local_acc), local_prop=int(local_prop),
        move_counters=np.asarray(move_counters, dtype=np.int64).copy(),
    )
    mc_sweep(replica, steps, model_name, params, Tref, Tscale)
    return (replica.state, replica.local_acc, replica.local_prop,
            replica.move_counters)


# ---------------------------------------------------------------------------
# Structural-observable recording (main process only; never in the MC hot loop)
# ---------------------------------------------------------------------------

def _record_scalar_observables(rep: Replica) -> tuple[float, float]:
    """Append per-cycle scalar observables; return (Rg2, Ree2) in lattice units.

    Records E, contacts (from the cached state.m), Rg, Rg^2, and Ree^2 every
    cycle.  These are cheap (O(N)) and recorded at full cadence so the canonical
    distributions and backward-compatible trajectories are unchanged.
    """
    rg2 = ico.radius_of_gyration_squared(rep.state.chain)
    ree2 = ico.end_to_end_distance_squared(rep.state.chain)
    rep.E_traj.append(rep.state.E)
    rep.C_traj.append(float(rep.state.m))
    rep.Rg_traj.append(math.sqrt(rg2))
    rep.Rg2_traj.append(rg2)
    rep.Ree2_traj.append(ree2)
    return rg2, ree2


def _record_structural_sample(
    rep: Replica, cycle: int, n_beads: int, save_m_r: bool = False,
    fixed_defs: dict | None = None, scaled_defs: dict | None = None,
) -> None:
    """Append contact-map-derived structural observables for one replica.

    Builds the contact map, verifies its count against the cached ``state.m``,
    and stores m_long_fixed, m_global_scaled, S_max, largest-component fraction,
    and the originating cycle index, using the EXACT resolved bin definitions
    (``fixed_defs``/``scaled_defs``) rather than module defaults.  The full m_r
    vector is retained in memory ONLY when ``save_m_r`` is True.  Raises if the
    recomputed count disagrees with ``state.m``.
    """
    cp, _seps = ico.build_contact_map(rep.state.chain)
    if cp.shape[0] != int(rep.state.m):
        raise RuntimeError(
            f"structural sample: contact-map count {cp.shape[0]} disagrees with "
            f"state.m={rep.state.m} (lane T={rep.T}, cycle={cycle})"
        )
    m_r = ico.contact_separation_counts(cp, n_beads)
    fixed = ico.bin_contact_separations_fixed(m_r, n_beads, fixed_defs)
    scaled = ico.bin_contact_separations_scaled(m_r, n_beads, scaled_defs)
    graph = ico.contact_graph_summary(cp, n_beads)
    rep.m_long_traj.append(int(fixed["m_long_fixed"]))
    rep.m_global_scaled_traj.append(int(scaled["m_global_scaled"]))
    rep.Smax_traj.append(int(graph["largest_component_vertices"]))
    rep.largest_component_fraction_traj.append(
        float(graph["largest_component_fraction_of_N"])
    )
    if save_m_r:
        rep.m_r_traj.append(m_r.astype(np.int32))
    rep.structural_cycles.append(int(cycle))


def _validate_stride(name: str, value: int) -> int:
    v = int(value)
    if v < 1:
        raise ValueError(f"{name} must be >= 1, got {value!r}")
    return v


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
    structural_observables: bool = False,
    structural_stride: int = 1,
    save_m_r: bool = False,
    bin_defs: dict | None = None,
    snapshot_writer: "SnapshotWriter | None" = None,
    snapshot_stride: int = 1,
    snapshot_start_cycle: int = 0,
) -> tuple[list[Replica], np.ndarray, np.ndarray]:
    """
    Run REMD.

    Each cycle:
      1. Every replica runs `steps_per_swap` local Metropolis steps.
      2. Adjacent pairs attempt swaps with even/odd alternation.
      3. Scalar observables (E, m, Rg, Rg^2, Ree^2) are recorded for every lane.
      4. Every ``structural_stride`` cycles, contact-map-derived structural
         observables (m_r, m_long, S_max, largest-component fraction) are
         recorded.  These use a separate trajectory length; missing cycles are
         not index-aligned with the scalar trajectories.
      5. When a ``snapshot_writer`` is supplied, coordinates and per-lane scalars
         are streamed to disk every ``snapshot_stride`` cycles, starting at
         ``snapshot_start_cycle``.

    The target ensemble, local Metropolis rule, and swap criterion are
    unchanged by structural recording or snapshotting; all of that work happens
    in the main process after the sweeps/swaps for a cycle complete.

    Returns:
        replicas   — list of Replica objects with full trajectories
        swap_props — (nT-1,) array: swap proposals per adjacent pair
        swap_accs  — (nT-1,) array: swap acceptances per adjacent pair
    """
    # Explicit seed=None behavior (serial and multiprocessing): draw one fresh
    # nondeterministic base seed, seed all RNGs with it, and use it as the base
    # for deterministic per-cycle/per-lane worker seeds.  This makes both modes
    # well-defined; previously seed=None left the global RNG unseeded.
    if seed is None:
        seed = random.SystemRandom().randrange(1, 2 ** 31)
    _seed_all(seed)

    structural_stride = _validate_stride("structural_stride", structural_stride)
    if save_m_r and not structural_observables:
        raise ValueError("save_m_r requires structural_observables to be enabled")
    save_configurations = snapshot_writer is not None
    if save_configurations:
        snapshot_stride = _validate_stride("snapshot_stride", snapshot_stride)
        if snapshot_start_cycle < 0:
            raise ValueError("snapshot_start_cycle must be >= 0")

    nT = len(Ts)
    n_beads = int(N)
    # Normalize + validate bin definitions ONCE at entry, even for direct
    # Python-API callers (not only via the CLI).  The result is a deep copy, so
    # the run never mutates module constants or the caller's dict, and the
    # definitions are validated semantically + as exhaustive partitions for N.
    # Phase 13: a direct API caller with bin_defs=None uses the RESOLVED schema
    # context (the JSON project definitions), NOT the isaw_contact_observables
    # compatibility constants -- so the online run and the CLI agree on one
    # authoritative definitions source.
    if bin_defs is None:
        import isaw_schema as _sch
        _ctx = _sch.active_definitions_context()
        # Thaw the recursively-frozen context bins to plain dicts before handing
        # them to normalize_bin_definitions (which deep-copies its inputs).
        _fixed_in = _sch._thaw(_ctx.fixed_bins)
        _scaled_in = _sch._thaw(_ctx.scaled_bins)
        _src = _sch.PROV_JSON
    elif isinstance(bin_defs, dict):
        _fixed_in = bin_defs.get("fixed")
        _scaled_in = bin_defs.get("scaled")
        _src = bin_defs.get("bin_definition_source", "explicit_caller_definitions")
    else:
        _fixed_in = _scaled_in = None
        _src = "compatibility_fallback"
    bin_defs = ico.normalize_bin_definitions(
        _fixed_in, _scaled_in, n_beads=n_beads, source=_src,
    )
    _fixed_defs = bin_defs["fixed"]
    _scaled_defs = bin_defs["scaled"]

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
    # Walker identity is tracked when *either* diagnostics or configuration
    # saving is active, because each saved snapshot must record its walker.
    track_walkers = bool(diagnostics) or save_configurations
    if track_walkers:
        lane_walker = np.arange(nT, dtype=np.int64)
        walker_lane = np.arange(nT, dtype=np.int64)
    # walker_temp_index (one row per cycle) is a diagnostics artifact only.
    if diagnostics:
        walker_temp_index = np.empty((n_cycles, nT), dtype=np.int32)
    else:
        walker_temp_index = None

    report_every = max(1, n_cycles // 20)
    base_seed = seed if seed is not None else 0
    # Use an explicit 'spawn' start method so worker startup is deterministic
    # and identical across platforms (Windows already defaults to spawn), and so
    # no forked parent state (e.g. an initialized plotting backend) leaks into
    # workers.  Worker seeds are supplied explicitly per cycle, so sampling is
    # unchanged by the start method.
    executor = (
        ProcessPoolExecutor(
            max_workers=min(n_workers, nT),
            mp_context=_mp.get_context("spawn"),
        )
        if n_workers > 1 else None
    )
    t_sweep_total = t_swap_total = t_struct_total = 0.0

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
                        replicas[k].move_counters,
                        steps_per_swap, model_name, params, Tref, Tscale,
                        worker_seeds[k],
                    )
                    for k in range(nT)
                ]
                for k, fut in enumerate(futures):
                    state, local_acc, local_prop, move_counters = fut.result()
                    replicas[k].state = state
                    replicas[k].local_acc = local_acc
                    replicas[k].local_prop = local_prop
                    replicas[k].move_counters = move_counters
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

            # --- observable recording (main process) ---------------------------
            do_structural = (
                structural_observables and (cycle % structural_stride) == 0
            )
            do_snapshot = (
                save_configurations
                and cycle >= snapshot_start_cycle
                and ((cycle - snapshot_start_cycle) % snapshot_stride) == 0
            )
            snap_coords = snap_contacts = snap_rg2 = snap_ree2 = None
            if do_snapshot:
                snap_coords = np.empty((nT, n_beads, 3), dtype=np.int64)
                snap_contacts = np.empty(nT, dtype=np.int64)
                snap_rg2 = np.empty(nT, dtype=np.float64)
                snap_ree2 = np.empty(nT, dtype=np.float64)

            for k, rep in enumerate(replicas):
                rg2, ree2 = _record_scalar_observables(rep)
                if do_structural:
                    _record_structural_sample(
                        rep, cycle, n_beads, save_m_r, _fixed_defs, _scaled_defs)
                if do_snapshot:
                    snap_coords[k] = np.asarray(rep.state.chain, dtype=np.int64)
                    snap_contacts[k] = int(rep.state.m)
                    snap_rg2[k] = rg2
                    snap_ree2[k] = ree2

            if do_snapshot:
                snapshot_writer.append(
                    cycle=cycle,
                    coordinates=snap_coords,
                    walker_id=lane_walker.astype(np.int64),
                    contacts=snap_contacts,
                    rg2_lattice=snap_rg2,
                    ree2_lattice=snap_ree2,
                )

            if diagnostics:
                walker_temp_index[cycle] = walker_lane

            t_struct_total += time.perf_counter() - t2

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
        total = t_sweep_total + t_swap_total + t_struct_total
        print(
            f"Timing:  sweeps {t_sweep_total:.2f}s  |"
            f"  swaps {t_swap_total:.2f}s  |"
            f"  structural {t_struct_total:.2f}s  |"
            f"  total {total:.2f}s"
        )

    if diagnostics and diag_store is not None:
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
    def _ms(arr):
        a = np.asarray(arr, dtype=float)
        if a.size == 0:
            return float("nan"), float("nan")
        return float(np.nanmean(a)), float(np.nanstd(a, ddof=0))

    results = []
    for rep in replicas:
        n = len(rep.E_traj)
        s = int(math.floor(n * burnin_frac))

        E_arr  = np.array(rep.E_traj[s:],  dtype=float)
        C_arr  = np.array(rep.C_traj[s:],  dtype=float)
        Rg_arr = np.array(rep.Rg_traj[s:], dtype=float)
        Rg2_arr  = np.array(rep.Rg2_traj[s:],  dtype=float)
        Ree2_arr = np.array(rep.Ree2_traj[s:], dtype=float)

        Rg_mean_lattice, Rg_std_lattice = _ms(Rg_arr)
        Rg2_mean_lat, Rg2_std_lat = _ms(Rg2_arr)
        Ree2_mean_lat, Ree2_std_lat = _ms(Ree2_arr)

        # Structural (contact-map) trajectories use their own burn-in slice
        # because they may be sampled at a coarser stride than the scalars.
        ns = len(rep.m_long_traj)
        ss = int(math.floor(ns * burnin_frac))
        m_long_mean, m_long_std = _ms(rep.m_long_traj[ss:])
        m_global_mean, m_global_std = _ms(rep.m_global_scaled_traj[ss:])
        smax_mean, smax_std = _ms(rep.Smax_traj[ss:])
        lcf_mean, lcf_std = _ms(rep.largest_component_fraction_traj[ss:])

        results.append({
            "T":        rep.T,
            "E_mean":   float(np.nanmean(E_arr)) if E_arr.size else float("nan"),
            "E_std":    float(np.nanstd(E_arr,  ddof=0)) if E_arr.size else float("nan"),
            "C_mean":   float(np.nanmean(C_arr)) if C_arr.size else float("nan"),
            "C_std":    float(np.nanstd(C_arr,  ddof=0)) if C_arr.size else float("nan"),
            "Rg_mean_lattice": Rg_mean_lattice,
            "Rg_std_lattice":  Rg_std_lattice,
            "Rg_mean":  rg_scale * Rg_mean_lattice,
            "Rg_std":   rg_scale * Rg_std_lattice,
            # R_g^2 / R_ee^2: squared lengths scale by rg_scale**2.
            "Rg2_mean_lattice": Rg2_mean_lat,
            "Rg2_std_lattice":  Rg2_std_lat,
            "Rg2_mean":  (rg_scale ** 2) * Rg2_mean_lat,
            "Rg2_std":   (rg_scale ** 2) * Rg2_std_lat,
            "Ree2_mean_lattice": Ree2_mean_lat,
            "Ree2_std_lattice":  Ree2_std_lat,
            "Ree2_mean":  (rg_scale ** 2) * Ree2_mean_lat,
            "Ree2_std":   (rg_scale ** 2) * Ree2_std_lat,
            # Contact/graph observables: never rescaled by rg_scale.
            # m_long_mean/std are the compatibility aliases for the FIXED scheme.
            "m_long_mean": m_long_mean,
            "m_long_std":  m_long_std,
            "m_long_fixed_mean": m_long_mean,
            "m_long_fixed_std":  m_long_std,
            "m_global_scaled_mean": m_global_mean,
            "m_global_scaled_std":  m_global_std,
            "Smax_mean":   smax_mean,
            "Smax_std":    smax_std,
            "largest_component_fraction_mean": lcf_mean,
            "largest_component_fraction_std":  lcf_std,
            "n_structural_samples": int(ns),
            # local_acc_rate is the LEGACY (null-inclusive) acceptance rate.
            "local_acc_rate": rep.local_acc_rate,
            "state_changing_acceptance_rate": rep.state_changing_acceptance_rate,
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
    "min_structural_samples": 20,  # warn if post-burn-in structural samples < this
    "min_state_changing_move_rate": 0.01,  # warn if state-changing accept rate < this
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
    structural_observables_enabled: bool = False,
) -> dict:
    """Assemble per-lane convergence and per-walker mixing diagnostics + warnings.

    When ``structural_observables_enabled`` is False, structural ESS/drift and
    insufficient-sample warnings are suppressed and the disabled state is
    recorded; per-cycle observables (contacts, Rg, Rg2, Ree2, energy) are always
    diagnosed.  Each observable emits at most one canonical ESS warning type
    (no duplicate ``low_ess`` + ``low_ess_contacts``).

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
        # Scalar structural observables share the per-cycle slice; squared
        # lengths scale by rg_scale**2.
        Rg2 = np.asarray(rep.Rg2_traj[s:], dtype=float) * float(rg_scale) ** 2
        Ree2 = np.asarray(rep.Ree2_traj[s:], dtype=float) * float(rg_scale) ** 2
        # Contact-map structural observables use their own (coarser) sample
        # count and cycle spacing; never rescaled by rg_scale.
        ns = len(rep.m_long_traj)
        ss = _post_burnin_slice(ns, burnin_frac)
        m_long = np.asarray(rep.m_long_traj[ss:], dtype=float)
        m_global = np.asarray(rep.m_global_scaled_traj[ss:], dtype=float)
        Smax = np.asarray(rep.Smax_traj[ss:], dtype=float)
        lcf = np.asarray(rep.largest_component_fraction_traj[ss:], dtype=float)
        struct_cycles = np.asarray(rep.structural_cycles[ss:], dtype=int)
        struct_spacing = (
            int(np.median(np.diff(struct_cycles)))
            if struct_cycles.size >= 2 else 1
        )
        # structural_burnin_start_index is a sample index into the structural
        # trajectory; structural_burnin_start_cycle is the originating REMD cycle.
        struct_start_cycle = (
            int(rep.structural_cycles[ss])
            if ns > 0 and ss < ns else None
        )
        entry = {
            "temperature_index": int(i),
            "temperature": float(Ts[i]) if i < Ts.size else float("nan"),
            "n_post_burnin": int(C.size),
            "n_post_burnin_structural": int(m_long.size),
            "structural_cycle_spacing": struct_spacing,
            "structural_burnin_start_index": int(ss),
            "structural_burnin_start_cycle": struct_start_cycle,
        }
        # spacing: per-cycle observables have cycle-spacing 1; structural
        # observables (sampled at structural_stride) use struct_spacing so that
        # tau_int_cycles is expressed in REMD cycles, not in sample steps.
        for name, arr, spacing in (
            ("contacts", C, 1), ("rg", Rg, 1), ("rg2", Rg2, 1),
            ("ree2", Ree2, 1), ("energy", E, 1),
            ("m_long_fixed", m_long, struct_spacing),
            ("m_global_scaled", m_global, struct_spacing),
            ("smax", Smax, struct_spacing),
            ("largest_component_fraction", lcf, struct_spacing),
        ):
            ac = integrated_autocorr_time(arr)
            dr = early_late_drift(arr)
            bl = block_mean_stability(arr, n_blocks)
            tau_samples = ac["tau_int"]
            tau_cycles = (
                tau_samples * float(spacing)
                if np.isfinite(tau_samples) else float("nan")
            )
            entry[name] = {
                "mean": float(np.nanmean(arr)) if arr.size else float("nan"),
                "std": float(np.nanstd(arr, ddof=0)) if arr.size else float("nan"),
                # tau_int kept as a compatibility alias (== tau_int_samples).
                "tau_int": tau_samples,
                "tau_int_samples": tau_samples,
                "tau_int_cycles": tau_cycles,
                "sample_cycle_spacing": int(spacing),
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
        # Backward-compatible alias: keep "m_long" pointing at m_long_fixed.
        entry["m_long"] = entry["m_long_fixed"]
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

    # Structural-observable ESS (separate stride; report per observable).
    # Canonical observable keys (no aliases): contact-map structural observables
    # are m_long_fixed / m_global_scaled / smax / largest_component_fraction.
    struct_obs = ("m_long_fixed", "m_global_scaled", "smax",
                  "largest_component_fraction")
    struct_min_ess = {}
    for name in struct_obs:
        vals = _obs_esss(name)
        struct_min_ess[name] = float(min(vals)) if vals else float("nan")
    # rg2/ree2 are per-cycle scalars (always available) -> reported separately.
    for name in ("rg2", "ree2"):
        vals = _obs_esss(name)
        struct_min_ess[name] = float(min(vals)) if vals else float("nan")
    key_struct = ("contacts", "rg2", "m_long_fixed", "smax")
    key_struct_ess = []
    for name in key_struct:
        vals = _obs_esss(name)
        if vals:
            key_struct_ess.append(min(vals))
    min_ess_key_structural = float(min(key_struct_ess)) if key_struct_ess else float("nan")
    n_struct_samples = [lane["n_post_burnin_structural"] for lane in lane_conv]
    min_struct_samples = int(min(n_struct_samples)) if n_struct_samples else 0
    struct_drift = [
        abs(lane["m_long_fixed"]["drift_in_std"]) for lane in lane_conv
        if np.isfinite(lane["m_long_fixed"]["drift_in_std"])
    ]

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
        # Structural-observable summary (separate sampling stride).
        "structural_observables_enabled": bool(structural_observables_enabled),
        "min_ess_rg2": struct_min_ess["rg2"],
        "min_ess_ree2": struct_min_ess["ree2"],
        "min_ess_m_long_fixed": struct_min_ess["m_long_fixed"],
        "min_ess_m_global_scaled": struct_min_ess["m_global_scaled"],
        "min_ess_smax": struct_min_ess["smax"],
        "min_ess_largest_component_fraction": struct_min_ess["largest_component_fraction"],
        "min_ess_key_structural": min_ess_key_structural,
        "min_structural_samples": min_struct_samples,
        "max_drift_m_long_fixed": float(max(struct_drift)) if struct_drift else float("nan"),
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

    # --- per-observable ESS warnings: exactly one canonical type each --------
    # Always-on per-cycle observables.
    always_on = (
        ("low_ess_contacts", "contacts", "contact"),
        ("low_ess_rg2", "rg2", "Rg^2"),
        ("low_ess_ree2", "ree2", "Ree^2"),
    )
    # Structural observables only diagnosed when explicitly enabled.
    structural_ess = (
        ("low_ess_m_long_fixed", "m_long_fixed", "m_long_fixed"),
        ("low_ess_m_global_scaled", "m_global_scaled", "m_global_scaled"),
        ("low_ess_smax", "smax", "S_max"),
        ("low_ess_largest_component_fraction", "largest_component_fraction",
         "largest-component fraction"),
    )
    ess_specs = list(always_on)
    if structural_observables_enabled:
        ess_specs += list(structural_ess)
    for warn_kind, obs_name, label in ess_specs:
        vals = _obs_esss(obs_name)
        if vals and min(vals) < float(thr["min_ess"]):
            _warn(warn_kind,
                  f"minimum {label} ESS {min(vals):.1f} < {thr['min_ess']}",
                  float(min(vals)), thr["min_ess"])

    if structural_observables_enabled:
        if struct_drift and max(struct_drift) > float(thr["max_drift"]):
            _warn("structural_drift",
                  f"maximum |m_long_fixed drift|/std {max(struct_drift):.3f} > "
                  f"{thr['max_drift']}",
                  float(max(struct_drift)), thr["max_drift"])
        min_struct_thr = int(thr.get("min_structural_samples",
                                     DEFAULT_DIAG_THRESHOLDS["min_structural_samples"]))
        if n_struct_samples and min_struct_samples < min_struct_thr:
            _warn("insufficient_structural_samples",
                  f"minimum post-burn-in structural samples {min_struct_samples} < "
                  f"{min_struct_thr}",
                  int(min_struct_samples), min_struct_thr)
    # Local-move freezing (state-changing acceptance, null moves excluded).
    for w in detect_local_move_freezing(
        replicas, Ts,
        float(thr.get("min_state_changing_move_rate",
                      DEFAULT_DIAG_THRESHOLDS["min_state_changing_move_rate"])),
    ):
        warnings.append(w)

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

def save_results_csv(
    results: list[dict], out_prefix: str,
    control_mode: str = "fitted_temperature",
) -> str:
    path = f"{out_prefix}_results.csv"
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    keys = [
        # Existing columns (unchanged order for backward compatibility).
        "T", "E_mean", "E_std", "C_mean", "C_std",
        "Rg_mean", "Rg_std", "Rg_mean_lattice", "Rg_std_lattice",
        "local_acc_rate",
        # New structural columns appended at the end.
        "Rg2_mean", "Rg2_std", "Rg2_mean_lattice", "Rg2_std_lattice",
        "Ree2_mean", "Ree2_std", "Ree2_mean_lattice", "Ree2_std_lattice",
        "m_long_mean", "m_long_std", "Smax_mean", "Smax_std",
        "largest_component_fraction_mean", "largest_component_fraction_std",
        "n_structural_samples",
        # Newest columns appended at the very end (compat: old readers ignore).
        # m_long_*_fixed are explicit aliases of the m_long_* (fixed-scheme)
        # columns above; m_global_scaled_* are new.
        "m_long_fixed_mean", "m_long_fixed_std",
        "m_global_scaled_mean", "m_global_scaled_std",
        "state_changing_acceptance_rate",
        # Control-parameter column (appended; old readers ignore).  In direct-K
        # mode this is the coupling K for the lane; in fitted-temperature mode it
        # is K(T) = -b(T).
        "K",
    ]
    # In direct-K mode temperature is not physically defined, so the "T" column
    # is written as NaN (least-disruptive: the column stays present, downstream
    # numeric readers get NaN) while the coupling lives in the "K" column.
    blank_temperature = control_mode == "direct_K"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(keys)
        for r in results:
            row = []
            for k in keys:
                if k == "T" and blank_temperature:
                    row.append(math.nan)
                else:
                    row.append(r.get(k, math.nan))
            w.writerow(row)
    print(f"Saved {path}")
    return path


def save_swap_csv(
    swap_props: np.ndarray,
    swap_accs:  np.ndarray,
    Ts: np.ndarray,
    out_prefix: str,
    control_mode: str = "fitted_temperature",
    K: np.ndarray | None = None,
) -> str:
    path = f"{out_prefix}_swap_rates.csv"
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    # K_lo/K_hi are appended so direct-K swap statistics remain analyzable by
    # coupling; in direct-K mode temperature is undefined so T_lo/T_hi are NaN.
    blank_temperature = control_mode == "direct_K"
    K = np.asarray(K, dtype=float) if K is not None else None
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pair", "T_lo", "T_hi", "proposals", "acceptances", "rate",
                    "K_lo", "K_hi"])
        for k in range(len(swap_props)):
            prop = int(swap_props[k])
            acc  = int(swap_accs[k])
            rate = acc / prop if prop > 0 else float("nan")
            t_lo = math.nan if blank_temperature else float(Ts[k])
            t_hi = math.nan if blank_temperature else float(Ts[k + 1])
            k_lo = float(K[k]) if K is not None else math.nan
            k_hi = float(K[k + 1]) if K is not None else math.nan
            w.writerow([k, t_lo, t_hi, prop, acc, f"{rate:.4f}", k_lo, k_hi])
    print(f"Saved {path}")
    return path


MOVE_ACCEPTANCE_CSV_COLUMNS = [
    # Existing leading columns (kept for backward compatibility), then new ones
    # appended.  metropolis_accepted now counts STATE-CHANGING accepted moves.
    "temperature_index", "temperature", "move_type",
    "proposed", "geometrically_valid", "metropolis_accepted",
    "a_geometry", "a_metropolis", "a_total",
    "state_changing", "a_state_changing",
]


def save_move_acceptance_csv(
    replicas: list[Replica], Ts: np.ndarray, out_prefix: str,
) -> str:
    """Per-lane, per-move-type acceptance breakdown (null moves excluded from
    state-changing acceptance).

    a_geometry        = geometrically_valid / proposed
    a_state_changing  = state_changing / proposed
    a_metropolis      = accepted / state_changing   (accepted == state-changing accepted)
    a_total           = accepted / proposed
    """
    path = f"{out_prefix}_move_acceptance.csv"
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(MOVE_ACCEPTANCE_CSV_COLUMNS)
        for i, rep in enumerate(replicas):
            counters = np.asarray(rep.move_counters, dtype=np.int64)
            for mi, mname in enumerate(MOVE_NAMES):
                prop = int(counters[mi, 0])
                valid = int(counters[mi, 1])
                state_changing = int(counters[mi, 2])
                acc = int(counters[mi, 3])
                a_geom = (valid / prop) if prop > 0 else float("nan")
                a_state = (state_changing / prop) if prop > 0 else float("nan")
                a_met = (acc / state_changing) if state_changing > 0 else float("nan")
                a_tot = (acc / prop) if prop > 0 else float("nan")
                w.writerow([
                    i, float(Ts[i]), mname, prop, valid, acc,
                    f"{a_geom:.6f}", f"{a_met:.6f}", f"{a_tot:.6f}",
                    state_changing, f"{a_state:.6f}",
                ])
    print(f"Saved {path}")
    return path


def detect_local_move_freezing(
    replicas: list[Replica], Ts: np.ndarray,
    min_state_changing_rate: float = 0.01,
) -> list[dict]:
    """Flag lanes where state-changing local moves are nearly frozen.

    Uses STATE-CHANGING accepted moves / proposed (null acceptances excluded).
    A frozen lane is not assumed to be in the collapsed phase; it is only a
    local-move-mixing warning.
    """
    warnings = []
    for i, rep in enumerate(replicas):
        counters = np.asarray(rep.move_counters, dtype=np.int64)
        prop = int(counters[:, 0].sum())
        acc = int(counters[:, 3].sum())   # state-changing accepted
        rate = (acc / prop) if prop > 0 else float("nan")
        if prop > 0 and rate < min_state_changing_rate:
            warnings.append({
                "type": "local_move_freezing",
                "temperature_index": int(i),
                "temperature": float(Ts[i]),
                "state_changing_acceptance_rate": float(rate),
                "threshold": float(min_state_changing_rate),
                "message": (
                    f"lane T={float(Ts[i]):.3g} state-changing move acceptance "
                    f"{rate:.4f} < {min_state_changing_rate} (local move freezing; "
                    f"not necessarily collapsed phase)"
                ),
            })
    return warnings


# Backward-compatible alias.
def detect_move_freezing(replicas, Ts, min_total_acc_rate: float = 0.01):
    return detect_local_move_freezing(replicas, Ts, min_total_acc_rate)


def save_distributions(dist: dict, out_prefix: str) -> str:
    """Save distributions NPZ with canonical keys and fitting-script aliases."""
    path = f"{out_prefix}_distributions.npz"
    Path(path).parent.mkdir(parents=True, exist_ok=True)
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
    "tau_int", "tau_int_samples", "tau_int_cycles", "sample_cycle_spacing",
    "ess", "drift", "drift_in_std", "block_mean_std",
    "block_mean_range_over_std", "acf_method",
]


def save_convergence_csv(diagnostics: dict, out_prefix: str) -> str:
    path = f"{out_prefix}_convergence.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(CONVERGENCE_CSV_COLUMNS)
        for lane in diagnostics["lane_convergence"]:
            for obs in ("contacts", "rg", "rg2", "ree2", "m_long_fixed",
                        "m_global_scaled", "smax",
                        "largest_component_fraction", "energy"):
                if obs not in lane:
                    continue
                d = lane[obs]
                w.writerow([
                    lane["temperature_index"], lane["temperature"], obs,
                    d["n_samples"], d["mean"], d["std"], d["tau_int"],
                    d.get("tau_int_samples", d["tau_int"]),
                    d.get("tau_int_cycles", d["tau_int"]),
                    d.get("sample_cycle_spacing", 1), d["ess"],
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
    configured_structural_stride: int | None = None,
    bin_definitions: dict | None = None,
    extra_metadata: dict | None = None,
) -> str:
    """Save compressed post-burn-in C/Rg/E and walker temperature-index traces.

    Arrays are stored as compact NumPy arrays (not Python objects):
        contacts_post  (nT, n_post)
        rg_post        (nT, n_post)   in output units (rg_scale applied)
        energy_post    (nT, n_post)
        rg2_post       (nT, n_post)   in output units (rg_scale**2 applied)
        ree2_post      (nT, n_post)   in output units (rg_scale**2 applied)
        walker_temp_index_post (n_post, n_walkers)

    Structural (coarser-stride) trajectories use a separate length n_struct:
        m_long_post                    (nT, n_struct)   (== m_long_fixed)
        m_global_scaled_post           (nT, n_struct)
        smax_post                      (nT, n_struct)
        largest_component_fraction_post (nT, n_struct)
        structural_sample_cycles       (n_struct,)   originating cycle indices
        structural_burnin_start_index  scalar (sample-index burn-in start)
        structural_burnin_start_cycle  scalar (originating REMD cycle, -1 if none)
        structural_stride              scalar
    When m_r trajectories were retained (--save-m-r-trajectories), the full
    contour-separation vectors are persisted (not merely kept in memory):
        m_r_post                       (nT, n_struct, n_beads)  compact int dtype
        m_r_index_definition           string
        fixed_bin_definitions          JSON string
        scaled_bin_definitions         JSON string

    The existing arrays are preserved; new arrays are added.  Scalar and
    structural arrays are NOT index-aligned because they use different strides.
    """
    path = f"{out_prefix}_diagnostic_trajectories.npz"
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    n_cycles = max((len(r.C_traj) for r in replicas), default=0)
    s = _post_burnin_slice(n_cycles, burnin_frac)
    C = np.array([np.asarray(r.C_traj[s:], dtype=np.float32) for r in replicas])
    E = np.array([np.asarray(r.E_traj[s:], dtype=np.float32) for r in replicas])
    Rg = np.array(
        [np.asarray(r.Rg_traj[s:], dtype=np.float32) * np.float32(rg_scale)
         for r in replicas]
    )
    rg2f = np.float32(rg_scale) ** 2
    Rg2 = np.array(
        [np.asarray(r.Rg2_traj[s:], dtype=np.float32) * rg2f for r in replicas]
    )
    Ree2 = np.array(
        [np.asarray(r.Ree2_traj[s:], dtype=np.float32) * rg2f for r in replicas]
    )
    # Structural trajectories (own stride / burn-in).
    ns = max((len(r.m_long_traj) for r in replicas), default=0)
    ss = _post_burnin_slice(ns, burnin_frac)
    m_long = np.array([np.asarray(r.m_long_traj[ss:], dtype=np.float32)
                       for r in replicas])
    m_global = np.array([np.asarray(r.m_global_scaled_traj[ss:], dtype=np.float32)
                         for r in replicas])
    smax = np.array([np.asarray(r.Smax_traj[ss:], dtype=np.float32)
                     for r in replicas])
    lcf = np.array([np.asarray(r.largest_component_fraction_traj[ss:],
                               dtype=np.float32) for r in replicas])
    struct_cycles = (
        np.asarray(replicas[0].structural_cycles[ss:], dtype=np.int64)
        if replicas else np.empty(0, dtype=np.int64)
    )
    struct_start_cycle = (
        int(replicas[0].structural_cycles[ss])
        if replicas and ss < len(replicas[0].structural_cycles) else -1
    )
    # Observed spacing is INFERRED from the saved sample cycles; it is -1 when
    # there are fewer than two structural samples (cannot be inferred).  The
    # configured stride is the value the run was launched with and is the
    # authoritative one; never infer the configured stride from observed gaps.
    observed_spacing = (
        int(np.median(np.diff(struct_cycles))) if struct_cycles.size >= 2 else -1
    )
    cfg_stride = (int(configured_structural_stride)
                  if configured_structural_stride is not None else -1)
    wti = np.asarray(walker_temp_index)
    wti_post = wti[s:].astype(np.int16) if wti.ndim == 2 else wti

    payload = dict(
        Ts=np.asarray(Ts, dtype=float),
        burnin_start_cycle=int(s),
        contacts_post=C,
        rg_post=Rg,
        energy_post=E,
        rg2_post=Rg2,
        ree2_post=Ree2,
        m_long_post=m_long,
        m_global_scaled_post=m_global,
        smax_post=smax,
        largest_component_fraction_post=lcf,
        structural_sample_cycles=struct_cycles,
        structural_burnin_start_index=int(ss),
        structural_burnin_start_cycle=int(struct_start_cycle),
        configured_structural_stride=int(cfg_stride),
        observed_structural_cycle_spacing=int(observed_spacing),
        # Compatibility key == the CONFIGURED stride (falls back to observed
        # only when the configured value was not supplied).
        structural_stride=int(cfg_stride if cfg_stride > 0 else observed_spacing),
        walker_temp_index_post=wti_post,
    )

    # Always record the EXACT resolved bin definitions used by this run (not the
    # current module defaults), because the NPZ carries m_long_post and
    # m_global_scaled_post which were binned with these definitions.
    if bin_definitions is not None:
        rec = bin_definitions
        fixed_d = rec.get("fixed", ico.FIXED_BIN_DEFINITIONS)
        scaled_d = rec.get("scaled", ico.SCALED_BIN_DEFINITIONS)
        defs_ver = rec.get("definitions_version", ico.DEFINITIONS_VERSION)
        src = rec.get("bin_definition_source", "unknown")
    else:
        fixed_d, scaled_d = ico.FIXED_BIN_DEFINITIONS, ico.SCALED_BIN_DEFINITIONS
        defs_ver, src = ico.DEFINITIONS_VERSION, "module_default"
    payload["definitions_version"] = str(defs_ver)
    payload["bin_definition_source"] = str(src)
    payload["fixed_bin_definitions"] = json.dumps(fixed_d)
    payload["scaled_bin_definitions"] = json.dumps(scaled_d)
    payload["structural_bin_definitions"] = json.dumps(
        bin_definitions if bin_definitions is not None
        else {"fixed": fixed_d, "scaled": scaled_d})

    # Persist full m_r vectors only if they were retained.
    if replicas and len(replicas[0].m_r_traj) > 0:
        n_beads = int(len(replicas[0].m_r_traj[0]))
        mr_max = 0
        for r in replicas:
            for v in r.m_r_traj[ss:]:
                mr_max = max(mr_max, int(np.asarray(v).max(initial=0)))
        mr_dtype = np.uint8 if mr_max <= 255 else (
            np.uint16 if mr_max <= 65535 else np.int32)
        m_r_post = np.array(
            [np.asarray(r.m_r_traj[ss:], dtype=mr_dtype) for r in replicas]
        )
        payload["m_r_post"] = m_r_post
        payload["m_r_index_definition"] = (
            "m_r_post[lane, sample, r] = number of contacts with contour "
            "separation r (0 <= r < n_beads); even r are zero; sum_r == m"
        )
        payload["n_beads"] = int(n_beads)

    # Additive control-mode metadata (e.g. direct-K vs fitted-temperature).  Keys
    # are stored verbatim; K_values is coerced to a float array for compactness.
    if extra_metadata:
        for key, value in extra_metadata.items():
            if key == "K_values":
                payload[key] = np.asarray(value, dtype=float)
            else:
                payload[key] = value

    np.savez_compressed(path, **payload)
    print(f"Saved {path}")
    return path


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _make_colormap(Ts: np.ndarray):
    cm, mcolors, _plt = _import_matplotlib()
    cmap = cm.coolwarm
    norm = mcolors.Normalize(vmin=float(np.nanmin(Ts)), vmax=float(np.nanmax(Ts)))
    sm   = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    return cmap, norm, sm


def plot_observables(results: list[dict], out_prefix: str) -> None:
    _cm, _mcolors, plt = _import_matplotlib()
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
    _cm, _mcolors, plt = _import_matplotlib()
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
    run_structural_quick_test()

    print("quick-test complete.")


def run_structural_quick_test() -> None:
    """Smoke + invariant tests for the structural-observable additions."""
    import os
    import tempfile

    # state.m stays in sync with contact_count; structural trajectories finite.
    Ts = np.linspace(300.0, 360.0, 5)
    hs_params = [378.96, 1.39686]
    Tref = 0.5 * (float(Ts.min()) + float(Ts.max()))
    Tscale = max(float(Ts.max()) - float(Ts.min()), 1.0)

    for n_workers in (1, 2):
        reps, _sp, _sa = run_remd(
            N=16, Ts=Ts, steps_per_swap=30, n_cycles=24,
            model_name="hs", params=hs_params, Tref=Tref, Tscale=Tscale,
            seed=9, n_workers=n_workers, verbose=False,
            structural_observables=True, structural_stride=2,
        )
        for rep in reps:
            assert rep.state.m == int(round(contact_count(rep.state.chain,
                                                          rep.state.occ)))
            # scalar trajectories every cycle, structural at stride 2.
            assert len(rep.C_traj) == 24
            assert len(rep.m_long_traj) == 12
            assert all(math.isfinite(v) for v in rep.Rg2_traj)
            assert all(math.isfinite(v) for v in rep.Ree2_traj)
            c = np.asarray(rep.move_counters)
            # accepted <= state_changing <= valid <= proposed
            assert np.all(c[:, 3] <= c[:, 2]) and np.all(c[:, 2] <= c[:, 1])
            assert np.all(c[:, 1] <= c[:, 0])
            # contact map count matches state.m for the final config.
            cp, _ = ico.build_contact_map(rep.state.chain)
            assert cp.shape[0] == rep.state.m
        stats = compute_statistics(reps, burnin_frac=0.5, rg_scale=2.0)
        for r in stats:
            # squared lengths scale by rg_scale**2
            if math.isfinite(r["Rg2_mean_lattice"]):
                np.testing.assert_allclose(r["Rg2_mean"],
                                           4.0 * r["Rg2_mean_lattice"])
        # move-acceptance CSV writes and round-trips.
        with tempfile.TemporaryDirectory() as tmp:
            prefix = os.path.join(tmp, "run")
            path = save_move_acceptance_csv(reps, Ts, prefix)
            assert os.path.exists(path)
    print("  structural quick-test (state.m, trajectories, scaling): PASSED")

    # Snapshot writer round-trip + contact-count agreement.
    if cio.h5py_available():
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "cfg.h5")
            writer = SnapshotWriter(
                path, n_beads=16, n_temperatures=len(Ts),
                metadata={"run_id": "qt",
                          "temperatures": [float(t) for t in Ts], "seed": 9},
                flush_interval=2,
            )
            reps, _, _ = run_remd(
                N=16, Ts=Ts, steps_per_swap=25, n_cycles=12,
                model_name="hs", params=hs_params, Tref=Tref, Tscale=Tscale,
                seed=9, n_workers=1, verbose=False,
                snapshot_writer=writer, snapshot_stride=3,
            )
            writer.close()
            import h5py as _h5
            with _h5.File(path, "r") as f:
                s = f["snapshots"]
                assert s["coordinates"].shape == (4, len(Ts), 16, 3)
                for si in range(s["coordinates"].shape[0]):
                    for k in range(len(Ts)):
                        coords = s["coordinates"][si, k].astype(np.int64)
                        cp, _ = ico.build_contact_map(coords)
                        assert cp.shape[0] == int(s["contacts"][si, k])
        print("  structural quick-test (HDF5 snapshot round-trip): PASSED")
    else:
        print("  structural quick-test (HDF5): SKIPPED (h5py unavailable)")


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
# Direct-K control mode (temperature-independent contact coupling)
# ---------------------------------------------------------------------------
# The direct-K ensemble is  P(C|K) ∝ exp[K m(C)], i.e. the reduced contact bias
# is b_i = -K_i (recall the sampler uses P(C) ∝ exp[-u], u = m*b).  This is
# realized by REUSING the existing ``poly2`` contact-bias model with the fixed
# coefficients (a0, a1, a2) = (0, -1, 0) and (Tref, Tscale) = (0, 1), so that
#     b(T) = a0 + a1*x + a2*x^2 = -x = -(T-0)/1 = -T .
# The per-lane "temperature" label then carries the coupling K directly, giving
#     b_i = b(K_i) = -K_i ,
# with NO new registry model (which would break the cross-script model contract
# in run_model_suite_2.py), NO change to the MC weight, and NO change to the
# replica-exchange criterion.  With this encoding the existing swap_log_accept
# reduces exactly to  (K_i - K_j)(m_j - m_i)  and the MC weight to exp(K_i * m).
DIRECT_K_MODEL_NAME = "poly2"
DIRECT_K_PARAMS = [0.0, -1.0, 0.0]
DIRECT_K_TREF = 0.0
DIRECT_K_TSCALE = 1.0


def parse_k_values(k_str: str) -> list[float]:
    """Parse/validate a comma-separated --K-values ladder; return sorted ascending.

    Rejects: an empty string, non-finite values (nan/inf), duplicate couplings,
    and fewer than two values.  Sorting into ascending order gives a
    deterministic ladder (adjacent lanes have the closest couplings, which is
    what the replica-exchange even/odd swap sweeps expect).
    """
    raw = [x.strip() for x in str(k_str).split(",") if x.strip()]
    if not raw:
        raise ValueError("--K-values was provided but no values were found")
    vals: list[float] = []
    for i, x in enumerate(raw, start=1):
        try:
            v = float(x)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"--K-values entry {i} is not a finite number: {x!r}"
            ) from exc
        if not math.isfinite(v):
            raise ValueError(f"--K-values entry {i} is not finite: {x!r}")
        vals.append(v)
    if len(vals) < 2:
        raise ValueError(
            f"--K-values needs at least two couplings, got {len(vals)}: {vals}"
        )
    ordered = sorted(vals)
    for a, b in zip(ordered[:-1], ordered[1:]):
        if a == b:
            raise ValueError(
                f"--K-values contains duplicate coupling {a!r}; couplings must "
                f"be distinct"
            )
    return ordered


def reject_direct_k_conflicts(args) -> None:
    """Raise ValueError if direct-K (--K-values) is combined with temperature/
    fitted-model arguments.

    Direct-K mode defines the contact coupling itself, so a temperature ladder,
    a fitted contact model, or explicit (Tref, Tscale, T0) are all meaningless
    and are rejected rather than silently ignored.
    """
    conflicts = []
    if getattr(args, "temps", None) is not None:
        conflicts.append("--temps")
    if getattr(args, "temps_from_npz", None) is not None:
        conflicts.append("--temps-from-npz")
    if getattr(args, "fit_summary_json", None) is not None:
        conflicts.append("--fit-summary-json")
    if getattr(args, "fit_params_csv", None) is not None:
        conflicts.append("--fit-params-csv")
    if getattr(args, "params", None) is not None:
        conflicts.append("--params")
    if getattr(args, "model", None) is not None:
        conflicts.append("--model")
    if getattr(args, "Tref", None) is not None:
        conflicts.append("--Tref")
    if getattr(args, "Tscale", None) is not None:
        conflicts.append("--Tscale")
    if getattr(args, "T0", None) is not None:
        conflicts.append("--T0")
    if conflicts:
        raise ValueError(
            "--K-values (direct-K mode) is mutually exclusive with temperature/"
            "fitted-model arguments: " + ", ".join(conflicts) + ". Direct-K mode "
            "sets the contact coupling directly (b_i = -K_i) and applies no "
            "temperature mapping."
        )


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
                f"{MODEL_API_VERSION}. Update remd_uniform_chain_2_new.py."
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
    # N is preserved for backward compatibility and means the number of beads.
    dist["N"] = int(N)
    dist["n_beads"] = int(N)
    dist["n_steps"] = int(N) - 1
    dist["steps_per_swap"] = int(steps_per_swap)
    dist["n_cycles"] = int(n_cycles)
    dist["burnin_frac"] = float(burnin_frac)
    dist["n_workers"] = int(n_workers)
    dist["model_api_version"] = int(MODEL_API_VERSION)
    dist["schema_version"] = int(SCHEMA_VERSION)
    return dist


def temperature_bias_arrays(
    Ts: np.ndarray, model_name: str, params, Tref: float, Tscale: float,
) -> dict:
    """Per-temperature reduced bias b(T), coupling K(T)=-b(T), weight q=exp(K).

    These are model-independent bookkeeping quantities used downstream; they do
    not assume the constant-(h, s) interpretation.
    """
    Ts = np.asarray(Ts, dtype=float)
    b = np.array([reduced_bias(model_name, params, float(T), Tref, Tscale)
                  for T in Ts], dtype=float)
    K = -b
    # K is authoritative.  q = exp(K) can overflow to +inf for large couplings;
    # keep the inf as a sentinel (serialized as null in JSON via _json_safe) and
    # warn rather than silently producing a misleading finite value.
    with np.errstate(over="ignore"):
        q = np.exp(K)
    if not np.all(np.isfinite(q)):
        n_overflow = int(np.count_nonzero(~np.isfinite(q)))
        print(
            f"  [warning] contact weight q=exp(K) overflowed for {n_overflow} "
            f"temperature(s); K is authoritative, q stored as inf/null."
        )
    return {
        "reduced_bias_by_temperature": b,
        "coupling_K_by_temperature": K,
        "contact_weight_q_by_temperature": q,
    }


def attach_structural_metadata(
    dist: dict,
    *,
    Ts: np.ndarray,
    model_name: str,
    params,
    Tref: float,
    Tscale: float,
    structural_stride: int,
    bin_defs: dict,
    save_configurations: bool,
    configuration_path: str | None,
    snapshot_stride: int,
    snapshot_start_cycle: int,
) -> dict:
    """Inject structural-analysis provenance into a distributions dict in place.

    All additive; existing canonical keys are untouched.  The bin definitions
    are stored as a JSON string (never a pickled object array).
    """
    bias = temperature_bias_arrays(Ts, model_name, params, Tref, Tscale)
    dist.update(bias)
    dist["structural_stride"] = int(structural_stride)
    dist["snapshot_stride"] = int(snapshot_stride)
    dist["snapshot_start_cycle"] = int(snapshot_start_cycle)
    dist["save_configurations"] = bool(save_configurations)
    dist["configuration_path"] = (
        str(configuration_path) if configuration_path else ""
    )
    dist["structural_bin_definitions"] = json.dumps(bin_defs)
    return dist


def _git_commit() -> str:
    """Best-effort current git commit hash, or 'unknown'."""
    try:
        import subprocess
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_os.path.dirname(_os.path.abspath(__file__)),
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip() or "unknown"
    except Exception:
        pass
    return "unknown"


def _file_sha256(path: str | None) -> str:
    """SHA-256 of a file, or 'unknown' when unavailable."""
    if not path:
        return "unknown"
    try:
        import hashlib
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return "unknown"


def _package_versions() -> dict:
    versions = {"numpy_version": np.__version__}
    try:
        import h5py as _h5
        versions["h5py_version"] = _h5.__version__
    except Exception:
        versions["h5py_version"] = "unknown"
    return versions


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
    ap.add_argument(
        "--K-values",
        type=str,
        default=None,
        dest="K_values",
        help=(
            "Direct-K control mode: comma-separated contact couplings K for the "
            "ladder, e.g. --K-values=-0.40,-0.32,...,0.35 (the '=' form is "
            "required so a leading negative value is not read as a flag). Samples "
            "P(C|K) proportional to exp[K*m(C)] with reduced bias b_i = -K_i and "
            "NO temperature mapping. Mutually exclusive with --temps / "
            "--fit-summary-json and the other fitted-model arguments."
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
    ap.add_argument("--diag-min-structural-samples", type=int,
                    default=DEFAULT_DIAG_THRESHOLDS["min_structural_samples"],
                    dest="diag_min_structural_samples",
                    help="warn if post-burn-in structural samples is below this")
    ap.add_argument("--diag-min-state-changing-move-rate", type=float,
                    default=DEFAULT_DIAG_THRESHOLDS["min_state_changing_move_rate"],
                    dest="diag_min_state_changing_move_rate",
                    help=("warn if a lane's state-changing local-move acceptance "
                          "rate (null moves excluded) is below this"))

    # --- Structural observables (opt-in; off by default) --------------------
    ap.add_argument(
        "--structural-observables", action="store_true",
        dest="structural_observables",
        help=(
            "Enable online contact-map-derived structural observables "
            "(m_long_fixed, m_global_scaled, S_max, largest-component fraction) "
            "recorded every --structural-stride cycles. OFF by default: ordinary "
            "REMD records only cheap scalars (m, Rg, Rg2, Ree2) every cycle and "
            "leaves full motif analysis to the offline extractor, which reads "
            "the authoritative saved coordinates."
        ),
    )
    ap.add_argument(
        "--structural-stride", type=int, default=1, dest="structural_stride",
        help=(
            "Record contact-map-derived structural observables every this many "
            "cycles (requires --structural-observables). A value of 0 disables "
            "structural observables. Default 1."
        ),
    )
    ap.add_argument(
        "--save-m-r-trajectories", action="store_true", dest="save_m_r_trajectories",
        help=(
            "Retain and persist the full per-sample m_r contour-separation "
            "vectors to the diagnostic-trajectories NPZ (requires "
            "--structural-observables and --diagnostic-trajectories). Off by "
            "default to avoid retaining large m_r histories in memory; the saved "
            "coordinates remain the authoritative source for offline motifs."
        ),
    )
    ap.add_argument(
        "--structural-bins-json", type=str, default=None,
        dest="structural_bins_json",
        help=(
            "Path to a JSON file overriding the fixed/scaled contour-separation "
            "bin definitions (keys: fixed.short_fixed.r_min/r_max, "
            "fixed.medium_fixed.r_min, fixed.long_threshold_fixed, "
            "scaled.local_max_ratio, scaled.meso_max_ratio)."
        ),
    )

    # --- Streaming coordinate snapshots (Phase 5; opt-in) -------------------
    ap.add_argument(
        "--save-configurations", action="store_true", dest="save_configurations",
        help=(
            "Stream per-cycle coordinate snapshots to a chunked HDF5 file "
            "(requires h5py). Off by default."
        ),
    )
    ap.add_argument(
        "--configuration-path", type=str, default=None, dest="configuration_path",
        help="Output path for snapshots. Default: <out-prefix>_configurations.h5",
    )
    ap.add_argument(
        "--snapshot-stride", type=int, default=1, dest="snapshot_stride",
        help="Save a coordinate snapshot every this many cycles. Default 1.",
    )
    ap.add_argument(
        "--snapshot-start-cycle", type=int, default=0, dest="snapshot_start_cycle",
        help="First cycle (0-based) eligible for snapshotting. Default 0.",
    )
    ap.add_argument(
        "--snapshot-flush-interval", type=int, default=50,
        dest="snapshot_flush_interval",
        help="Flush the HDF5 file every this many snapshots. Default 50.",
    )
    ap.add_argument(
        "--overwrite-configurations", action="store_true",
        dest="overwrite_configurations",
        help="Overwrite an existing configuration HDF5 file instead of failing.",
    )
    args = ap.parse_args()

    # Phase 11.3: refuse to begin a simulation when the authoritative JSON
    # definitions and the compatibility constants disagree.
    import isaw_schema as _sch
    _sch.check_definitions_consistency()

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
    if args.structural_stride < 0:
        raise ValueError("--structural-stride must be >= 0 (0 disables)")
    # --structural-stride 0 disables structural observables entirely.
    structural_observables = bool(args.structural_observables) and args.structural_stride >= 1
    structural_stride_eff = max(1, int(args.structural_stride))
    if args.save_m_r_trajectories:
        if not structural_observables:
            raise ValueError(
                "--save-m-r-trajectories requires --structural-observables and "
                "--structural-stride >= 1"
            )
        if not args.diagnostic_trajectories:
            raise ValueError(
                "--save-m-r-trajectories requires --diagnostic-trajectories "
                "(m_r vectors are persisted into that NPZ)"
            )
    if args.save_configurations:
        if args.snapshot_stride < 1:
            raise ValueError("--snapshot-stride must be >= 1")
        if args.snapshot_start_cycle < 0:
            raise ValueError("--snapshot-start-cycle must be >= 0")
        if args.snapshot_flush_interval < 1:
            raise ValueError("--snapshot-flush-interval must be >= 1")
        if args.snapshot_start_cycle >= args.n_cycles:
            raise ValueError(
                f"--snapshot-start-cycle ({args.snapshot_start_cycle}) >= "
                f"--n-cycles ({args.n_cycles}) would write an empty snapshot "
                f"file; lower the start cycle or omit --save-configurations."
            )
        if not cio.h5py_available():
            raise RuntimeError(
                "--save-configurations requires the 'h5py' package, which is "
                "not installed. Install h5py or drop --save-configurations."
            )
    elif args.configuration_path is not None:
        raise ValueError(
            "--configuration-path requires --save-configurations"
        )

    if args.quick_test:
        run_quick_test()
        return

    # Standalone runs should be able to use a nested output prefix without
    # requiring the caller to create the directory first. The suite already
    # creates it, so this is harmless there.
    Path(args.out_prefix).parent.mkdir(parents=True, exist_ok=True)

    # Two mutually exclusive control modes: direct-K (--K-values) sets the
    # contact coupling itself; otherwise the fitted-temperature ladder is
    # resolved (--temps-from-npz > --temps > linspace) with a fitted b(T) model.
    direct_k_mode = args.K_values is not None
    if direct_k_mode:
        reject_direct_k_conflicts(args)
        K_values = parse_k_values(args.K_values)
        # The lane label carries the coupling K directly; poly2(0,-1,0) with
        # (Tref, Tscale) = (0, 1) makes b(label) = -label = -K (see the direct-K
        # section above), so no new model and no hot-loop change are needed.
        Ts = np.asarray(K_values, dtype=float)
        temp_source = "cli:--K-values"
        model_name = DIRECT_K_MODEL_NAME
        model_params = list(DIRECT_K_PARAMS)
        param_names = MODEL_REGISTRY[model_name]["param_names"]
        Tref, Tscale = DIRECT_K_TREF, DIRECT_K_TSCALE
        parameter_source = "direct_K"
        fit_summary_json = None
    else:
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
        (
            model_name, model_params, param_names, Tref, Tscale,
            parameter_source, fit_summary_json,
        ) = resolve_model_params(args, Ts)

    nT = len(Ts)
    Tmin_resolved, Tmax_resolved = float(Ts.min()), float(Ts.max())
    diffs = np.diff(Ts)
    # In direct-K mode this measures K-ladder uniformity, not temperature.
    temperature_uniform = bool(nT >= 2 and np.allclose(diffs, diffs[0]))
    total_steps = args.steps_per_swap * args.n_cycles

    # Per-lane coupling K = -b(label); in direct-K mode this recovers the K
    # ladder exactly, in fitted mode it is K(T).  Single source for the CSV/K
    # columns and the control metadata below.
    K_by_lane = np.array(
        [-reduced_bias(model_name, model_params, float(T), Tref, Tscale)
         for T in Ts],
        dtype=float,
    )
    control_mode = "direct_K" if direct_k_mode else "fitted_temperature"
    control_metadata = {
        "control_mode": control_mode,
        "control_parameter": "K" if direct_k_mode else "temperature",
        "K_values": [float(k) for k in K_by_lane],
        "temperature_mapping_applied": bool(not direct_k_mode),
    }

    if direct_k_mode:
        print(
            f"Direct-K ladder ({temp_source}): {nT} lanes, "
            f"K in [{Tmin_resolved:.6g}, {Tmax_resolved:.6g}], "
            f"uniform={temperature_uniform}"
        )
        print(
            f"REMD (direct-K): {nT} lanes, "
            f"{args.n_cycles} cycles x {args.steps_per_swap} steps = "
            f"{total_steps} steps/lane"
        )
        print("Sampling: P(C|K) proportional to exp[K*m(C)];  b_i = -K_i "
              "(reused poly2 b(T) = -T with lane label = K; no temperature "
              "mapping)")
    else:
        print(
            f"Temperature ladder ({temp_source}): {nT} replicas, "
            f"min={Tmin_resolved:.6g}, max={Tmax_resolved:.6g}, "
            f"uniform={temperature_uniform}"
        )
        print(
            f"REMD: {nT} replicas, T in [{Tmin_resolved:.6g}, "
            f"{Tmax_resolved:.6g}], {args.n_cycles} cycles x "
            f"{args.steps_per_swap} steps = {total_steps} steps/replica"
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

    # Contour-separation bin definitions (two independent schemes; optional
    # JSON override of either).  Both the requested override and the resolved
    # definitions are recorded in the run summary for provenance.
    requested_bin_definitions = None
    # Phase 11.2: default runtime bin definitions come from the authoritative
    # project-definitions JSON (via isaw_schema), not the compatibility fallback
    # constants in isaw_contact_observables.  A consistency check at CLI entry
    # (below, in the ladder-resolution block) refuses to run when they diverge.
    import isaw_schema as _sch
    fixed_defs = _sch.get_fixed_bin_definitions()
    scaled_defs = _sch.get_scaled_bin_definitions()
    if args.structural_bins_json is not None:
        with open(args.structural_bins_json, encoding="utf-8") as fh:
            override = json.load(fh)
        requested_bin_definitions = override
        if "fixed" in override:
            fixed_defs.update(override["fixed"])
        if "scaled" in override:
            scaled_defs.update(override["scaled"])
    bin_check = ico.validate_bin_definitions(int(args.N), fixed_defs, scaled_defs)
    for msg in bin_check["warnings"]:
        print(f"  [bin warning] {msg}")
    bin_defs = ico.project_bin_definitions(int(args.N))
    bin_defs["fixed"] = fixed_defs
    bin_defs["scaled"] = scaled_defs
    # Phase 13: the default bins are JSON-sourced (via isaw_schema), so label
    # them accordingly -- NOT "module_default" (which would misattribute the
    # authoritative JSON definitions to the compatibility constants).
    bin_defs["bin_definition_source"] = (
        "cli_override" if requested_bin_definitions is not None
        else _sch.PROV_JSON)

    # Optional streaming coordinate-snapshot writer (opt-in).
    snapshot_writer = None
    configuration_path = None
    if args.save_configurations:
        configuration_path = (
            args.configuration_path
            if args.configuration_path is not None
            else f"{args.out_prefix}_configurations.h5"
        )
        import socket as _socket
        snap_meta = {
            "schema_version": int(cio.SNAPSHOT_SCHEMA_VERSION),
            "run_id": Path(args.out_prefix).name,
            "n_beads": int(args.N),
            "n_steps": int(args.N) - 1,
            "seed": int(args.seed),
            "model_name": model_name,
            "param_names": list(param_names),
            "model_params": [float(v) for v in model_params],
            "Tref": float(Tref),
            "Tscale": float(Tscale),
            "temperatures": [float(t) for t in Ts],
            "temperature_source": temp_source,
            "rg_scale": float(args.rg_scale),
            "steps_per_swap": int(args.steps_per_swap),
            "n_cycles": int(args.n_cycles),
            "burnin_frac": float(args.burnin_frac),
            "n_workers": int(args.n_workers),
            "structural_observables_enabled": bool(structural_observables),
            "structural_stride": int(structural_stride_eff),
            "snapshot_stride": int(args.snapshot_stride),
            "snapshot_start_cycle": int(args.snapshot_start_cycle),
            "snapshot_flush_interval": int(args.snapshot_flush_interval),
            "fixed_bin_definitions": json.dumps(fixed_defs),
            "scaled_bin_definitions": json.dumps(scaled_defs),
            "structural_bin_definitions": bin_defs,
            "command_line": " ".join(_sys.argv),
            "python_version": _sys.version.split()[0],
            "hostname": _socket.gethostname(),
            "git_commit": _git_commit(),
            "start_time": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
            "end_time": "unknown",
            "input_fit_summary_path": str(fit_summary_json) if fit_summary_json else "unknown",
            "input_fit_summary_sha256": _file_sha256(fit_summary_json),
        }
        snap_meta.update(_package_versions())
        snap_meta.update(
            {k: [float(v) for v in arr] for k, arr in
             temperature_bias_arrays(Ts, model_name, model_params,
                                     Tref, Tscale).items()}
        )
        # Unambiguous control-mode tag (direct-K vs fitted-temperature).
        snap_meta.update(control_metadata)
        snapshot_writer = SnapshotWriter(
            configuration_path,
            n_beads=int(args.N), n_temperatures=int(nT),
            metadata=snap_meta,
            flush_interval=int(args.snapshot_flush_interval),
            overwrite=bool(args.overwrite_configurations),
        )

    diag_store: dict = {}
    t_run_start = time.perf_counter()
    try:
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
            structural_observables=structural_observables,
            structural_stride=structural_stride_eff,
            save_m_r=bool(args.save_m_r_trajectories),
            bin_defs=bin_defs,
            snapshot_writer=snapshot_writer,
            snapshot_stride=int(args.snapshot_stride),
            snapshot_start_cycle=int(args.snapshot_start_cycle),
        )
        # Mark the snapshot file complete ONLY after run_remd returns normally.
        if snapshot_writer is not None:
            snapshot_writer.update_metadata({
                "end_time": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
            })
            snapshot_writer.mark_complete()
    finally:
        if snapshot_writer is not None:
            snapshot_writer.close()
    wall_time_seconds = time.perf_counter() - t_run_start
    if snapshot_writer is not None:
        print(
            f"Saved {configuration_path} "
            f"({snapshot_writer.n_snapshots} snapshots)"
        )

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
    legacy_local_acceptance_rates = list(local_acceptance_rates)
    state_changing_acceptance_rates = [
        float(r["state_changing_acceptance_rate"]) for r in results]
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
    attach_structural_metadata(
        dist,
        Ts=Ts, model_name=model_name, params=model_params,
        Tref=Tref, Tscale=Tscale,
        structural_stride=int(structural_stride_eff),
        bin_defs=bin_defs,
        save_configurations=bool(args.save_configurations),
        configuration_path=configuration_path,
        snapshot_stride=int(args.snapshot_stride),
        snapshot_start_cycle=int(args.snapshot_start_cycle),
    )
    dist["structural_observables_enabled"] = bool(structural_observables)
    # Control-mode tags so a direct-K distributions NPZ is self-identifying.
    dist["control_mode"] = control_metadata["control_mode"]
    dist["control_parameter"] = control_metadata["control_parameter"]
    dist["K_values"] = np.asarray(K_by_lane, dtype=float)
    dist["temperature_mapping_applied"] = bool(
        control_metadata["temperature_mapping_applied"])

    # Per-lane coupling K for the CSV "K" column (K(T) in fitted mode, the
    # coupling itself in direct-K mode).
    for i, r in enumerate(results):
        r["K"] = float(K_by_lane[i])

    results_path = save_results_csv(
        results, args.out_prefix, control_mode=control_mode)
    swap_path = save_swap_csv(
        swap_props, swap_accs, Ts, args.out_prefix,
        control_mode=control_mode, K=K_by_lane)
    move_acc_path = save_move_acceptance_csv(replicas, Ts, args.out_prefix)
    dist_path = save_distributions(dist, args.out_prefix)
    output_files = {
        "results_csv": results_path,
        "swap_rates_csv": swap_path,
        "move_acceptance_csv": move_acc_path,
        "distributions_npz": dist_path,
    }
    if args.save_configurations:
        output_files["configuration_hdf5"] = configuration_path
    # Warn about local-move freezing (state-changing acceptance; null moves
    # excluded). A frozen lane is not assumed to be collapsed.
    for w in detect_local_move_freezing(
        replicas, Ts, float(args.diag_min_state_changing_move_rate)
    ):
        print(f"  [move warning] {w['message']}")

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
            "min_structural_samples": args.diag_min_structural_samples,
            "min_state_changing_move_rate": args.diag_min_state_changing_move_rate,
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
            structural_observables_enabled=structural_observables,
        )
        # Tag the diagnostics with the control mode so a direct-K diagnostics
        # JSON is self-identifying (lane "temperature" fields carry K).
        diagnostics_result.update(control_metadata)
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
            traj_path = save_diagnostic_trajectories_npz(
                replicas, walker_temp_index, Ts,
                burnin_frac=args.burnin_frac, out_prefix=args.out_prefix,
                rg_scale=args.rg_scale,
                configured_structural_stride=(
                    structural_stride_eff if structural_observables else None),
                bin_definitions=bin_defs,
                extra_metadata=control_metadata,
            )
            output_files["diagnostic_trajectories_npz"] = traj_path
            # The diagnostic-trajectories NPZ also carries the post-burn-in
            # structural (m_long/S_max/largest-component-fraction) traces.
            output_files["structural_trajectories_npz"] = traj_path
        diagnostics_overhead_seconds = time.perf_counter() - t_diag
        for w in diagnostics_result["warnings"]:
            print(f"  [diagnostic warning] {w['message']}")
        print(
            f"Diagnostics: {len(diagnostics_result['warnings'])} warning(s); "
            f"computed in {diagnostics_overhead_seconds:.2f}s"
        )

    if not args.no_plots:
        _cm, _mcolors, _plt = _import_matplotlib()
        if _plt is None or _cm is None or _mcolors is None:
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
        # Control-mode tags: direct_K sets b_i = -K_i with no temperature
        # mapping; fitted_temperature uses the fitted b(T) model.
        "control_mode": control_metadata["control_mode"],
        "control_parameter": control_metadata["control_parameter"],
        "K_values": control_metadata["K_values"],
        "temperature_mapping_applied": control_metadata[
            "temperature_mapping_applied"],
        "direct_k_mode": bool(direct_k_mode),
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
        "n_beads": int(args.N),
        "n_steps": int(args.N) - 1,
        "schema_version": int(SCHEMA_VERSION),
        "steps_per_swap": int(args.steps_per_swap),
        "n_cycles": int(args.n_cycles),
        "total_steps_per_replica": int(total_steps),
        "seed": int(args.seed),
        "n_workers": int(args.n_workers),
        "burnin_frac": float(args.burnin_frac),
        "rg_bins": int(args.rg_bins),
        "rg_scale": float(args.rg_scale),
        "structural_observables_enabled": bool(structural_observables),
        "structural_stride": int(structural_stride_eff),
        "save_m_r_trajectories": bool(args.save_m_r_trajectories),
        "structural_bin_definitions": bin_defs,
        "requested_bin_definitions": requested_bin_definitions,
        "resolved_bin_definitions": {"fixed": fixed_defs, "scaled": scaled_defs,
                                     "definitions_version": ico.DEFINITIONS_VERSION},
        "git_commit": _git_commit(),
        "diag_min_state_changing_move_rate": float(args.diag_min_state_changing_move_rate),
        "save_configurations": bool(args.save_configurations),
        "configuration_path": configuration_path,
        "snapshot_stride": int(args.snapshot_stride),
        "snapshot_start_cycle": int(args.snapshot_start_cycle),
        "reduced_bias_by_temperature": dist["reduced_bias_by_temperature"].tolist(),
        "coupling_K_by_temperature": dist["coupling_K_by_temperature"].tolist(),
        "contact_weight_q_by_temperature": dist["contact_weight_q_by_temperature"].tolist(),
        "wall_time_seconds": float(wall_time_seconds),
        "swap_rates": swap_rates,
        "swap_rate_min": float(swap_rates_finite.min()) if swap_rates_finite.size else None,
        "swap_rate_mean": float(swap_rates_finite.mean()) if swap_rates_finite.size else None,
        "swap_rate_median": float(np.median(swap_rates_finite)) if swap_rates_finite.size else None,
        "local_acceptance_rates": local_acceptance_rates,
        "legacy_local_acceptance_rates": legacy_local_acceptance_rates,
        "state_changing_acceptance_rates": state_changing_acceptance_rates,
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
