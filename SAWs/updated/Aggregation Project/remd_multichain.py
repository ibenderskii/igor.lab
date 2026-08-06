#!/usr/bin/env python3
"""
Replica-exchange Monte Carlo for the multi-chain lattice aggregation model.

Simulates M identical connected self-avoiding chains (N beads each) in a
periodic cubic box of side L, reusing the validated single-chain contact free
energy from ``remd_uniform_chain_2_new.py`` WITHOUT refitting.  The reduced
For the ``hs`` and ``saturating_cooperative_contact`` models, the reduced
potential is the selected single-chain Hamiltonian applied to every contact:

    u(X, T) = lambda_contact * u_contact(m_total(X), T; M*N)
              + kappa_bend * n_bend_total(X)
    m_total = m_intra + m_inter

where ``lambda_contact = lambda_intra = lambda_inter``.  Unequal lambda values
are rejected for these two models because they would make the Hamiltonian depend
on whether a contact is intra- or interchain.  The contact-quadratic models keep
their existing per-chain-intrachain plus linear-interchain definition.

where ``u_contact(m, T; N)`` is the generic single-chain contact potential from
the model API, in whichever of the three families the fitted model belongs to:

    linear                  u = b(T)*m
    contact_quadratic       u = b(T)*m + kappa(T)*m^2/(2N)
    saturating_cooperative  u = N*[b(T)*q - A0*q^2/(1 + (q/q_sat)^2)],  q = m/N

where ``b(T)`` is the model's own linear coefficient (``h_b/T - s_b`` for the
hs family, which includes ``saturating_cooperative_contact``) and ``m_ref = 0``
for every model, so ``q = m/N`` exactly.  In the multichain saturation model,
``m`` is the globally unique total contact count and ``N`` in that formula is the
total bead count ``M*N``.  Thus intrachain and interchain contacts enter only
through their sum and receive the same fitted cooperative Hamiltonian.  The
single molecular-to-lattice contact offset is NOT applied.  Setting both lambda
values to zero disables contact interactions but does not disable bending when
``kappa_bend != 0``.

Sampling weight P(X|T) ∝ exp(-u).  The REMD swap uses the generalized full-state
reduced-potential rule (evaluated through the shared state-aware potential):

    log_accept = u(X_i,T_i) + u(X_j,T_j) - u(X_j,T_i) - u(X_i,T_j)

Sweep convention:
    1 local sweep       = M * N local move proposals
    1 translation sweep = M whole-chain translation proposals
    1 reptation sweep   = M whole-chain reptation (slithering-snake) proposals
    1 rotation sweep    = M whole-chain rigid-rotation proposals
Each REMD cycle performs local_sweeps_per_swap local sweeps, then
translation_sweeps_per_swap translation sweeps, then reptation_sweeps_per_swap
reptation sweeps, then rotation_sweeps_per_swap rotation sweeps, then one
even/odd swap stage.

Quick test:  python remd_multichain.py --quick-test
"""
from __future__ import annotations

import argparse
import dataclasses
import math
import multiprocessing as _mp
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import remd_uniform_chain_2_new as remd  # model registry + fitted-model loading
import multichain_state as mcs
import multichain_contacts as mcc
import multichain_moves as mvs
import multichain_observables as obs
import multichain_config_io as mcio
import multichain_diagnostics as mcd
from lattice_bending import BEND_DEFINITION
from multichain_state import MultiChainState, ContactCounts

SCHEMA_VERSION = mcio.MULTICHAIN_OUTPUT_SCHEMA_VERSION
MODEL_API_VERSION = remd.MODEL_API_VERSION

# The exact label-blind potential used by hs and saturating_cooperative_contact.
# The two retained lambda arguments must be equal and therefore form one common
# contact scale.  Contact-quadratic models retain the older split definition.
MULTICHAIN_POTENTIAL_DEFINITION = (
    "u(X,T) = lambda_contact * u_contact(m_total, T; M*N) "
    "+ kappa_bend * n_bend_total;  m_total = m_intra + m_inter;  "
    "lambda_contact = lambda_intra = lambda_inter;  intra- and interchain "
    "contacts are energetically indistinguishable")

SPLIT_MULTICHAIN_POTENTIAL_DEFINITION = (
    "u(X,T) = lambda_intra * sum_alpha u_contact(m_intra_alpha, T; N) "
    "+ lambda_inter * b(T) * m_inter + kappa_bend * n_bend_total;  "
    "u_contact is evaluated per chain and the interchain term is linear")

LABEL_BLIND_CONTACT_MODELS = frozenset(("hs", "saturating_cooperative_contact"))

# Scope retained by the contact-quadratic models.  The saturation model reports
# ``all_contacts_global`` through :func:`_contact_scope` instead.
NONLINEAR_CONTACT_SCOPE = "intra_per_chain"


def _uses_label_blind_contacts(model_name: str) -> bool:
    return str(model_name) in LABEL_BLIND_CONTACT_MODELS


def _label_blind_contact_scale(lambda_intra: float, lambda_inter: float) -> float:
    """Return the common contact scale, rejecting intra/inter discrimination."""
    li = float(lambda_intra)
    lin = float(lambda_inter)
    if li != lin:
        raise ValueError(
            "hs and saturating_cooperative_contact require "
            "lambda_intra == lambda_inter so all contacts use one Hamiltonian")
    return li


def _multichain_potential_definition(model_name: str) -> str:
    if _uses_label_blind_contacts(model_name):
        return MULTICHAIN_POTENTIAL_DEFINITION
    return SPLIT_MULTICHAIN_POTENTIAL_DEFINITION


def _contact_scope(model_name: str) -> str:
    return "all_contacts_global" if _uses_label_blind_contacts(model_name) \
        else NONLINEAR_CONTACT_SCOPE


def _interchain_contact_model(model_name: str) -> str:
    return "same_single_chain_potential" if _uses_label_blind_contacts(model_name) \
        else "linear_coefficient_only"


def _import_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except Exception:  # pragma: no cover - depends on local plotting stack
        return None


# ---------------------------------------------------------------------------
# Reduced potential and swap rule (both contact counts)
# ---------------------------------------------------------------------------

def _require_linear_for_aggregate(model_name: str, where: str) -> None:
    """Reject EVERY nonlinear model for the aggregate-count compatibility helpers.

    The contact-quadratic family needs the per-chain allocation, while the
    saturating-cooperative family needs the total bead count M*N for q.  Neither
    is available from ``ContactCounts`` alone, so these helpers remain linear-only.
    """
    kind = str(remd.MODEL_REGISTRY[model_name]["potential_kind"])
    if kind != "linear":
        raise NotImplementedError(
            f"{where} received model {model_name!r}, which is nonlinear in m "
            f"(potential_kind={kind!r}); the aggregate ContactCounts carries only "
            f"contact totals without the state information needed by this "
            f"model. Use the state-aware potential "
            f"(reduced_potential_state / reduced_contact_potential_state)."
        )


def reduced_potential_counts(
    counts: ContactCounts, temperature: float, model_name: str, params,
    Tref: float, Tscale: float, lambda_intra: float, lambda_inter: float,
) -> float:
    """u = reduced_bias(T) * (lambda_intra * m_intra + lambda_inter * m_inter).

    Aggregate-count compatibility helper for LINEAR models only (every nonlinear
    model is rejected because aggregate totals cannot supply the per-chain
    allocation that ``sum_alpha u_contact(m_alpha)`` depends on).
    """
    _require_linear_for_aggregate(model_name, "reduced_potential_counts")
    b = remd.reduced_bias(model_name, params, float(temperature), Tref, Tscale)
    if _uses_label_blind_contacts(model_name):
        scale = _label_blind_contact_scale(lambda_intra, lambda_inter)
        return scale * float(b) * (int(counts.intra) + int(counts.inter))
    return float(b) * (
        float(lambda_intra) * int(counts.intra)
        + float(lambda_inter) * int(counts.inter))


def reduced_contact_potential_state(
    state: MultiChainState, temperature: float, model_name: str, params,
    Tref: float, Tscale: float, lambda_intra: float, lambda_inter: float,
) -> float:
    """State-aware contacts-only reduced potential.

    For hs and saturating_cooperative_contact, apply the selected single-chain
    Hamiltonian once to ``m_total = m_intra + m_inter``, using ``M*N`` beads for
    any intensive normalization.  The contact classification is therefore absent
    from the energy.  Other registered models retain the existing split form:

        lambda_intra * sum_alpha u_contact(m_intra_alpha, T; N)
        + lambda_inter * b(T) * m_inter
    """
    spec = remd.MODEL_REGISTRY[model_name]
    if _uses_label_blind_contacts(model_name):
        scale = _label_blind_contact_scale(lambda_intra, lambda_inter)
        m_total = int(state.counts.intra) + int(state.counts.inter)
        total_beads = int(state.n_chains) * int(state.chain_length)
        return scale * remd.reduced_contact_potential(
            m_total, float(temperature), model_name, params, Tref, Tscale,
            total_beads)
    b = remd.reduced_bias(model_name, params, float(temperature), Tref, Tscale)
    if spec["potential_kind"] == "linear":
        # Aggregate form: identical to the historical contacts-only potential.
        return float(b) * (
            float(lambda_intra) * int(state.counts.intra)
            + float(lambda_inter) * int(state.counts.inter))
    N = state.chain_length
    intra_u = 0.0
    for m_alpha in state.intra_contacts_by_chain:
        intra_u += remd.reduced_contact_potential(
            int(m_alpha), float(temperature), model_name, params, Tref, Tscale, N)
    return (float(lambda_intra) * intra_u
            + float(lambda_inter) * float(b) * int(state.counts.inter))


def reduced_potential_state(
    state: MultiChainState, temperature: float, model_name: str, params,
    Tref: float, Tscale: float, lambda_intra: float, lambda_inter: float,
    kappa_bend: float = 0.0,
) -> float:
    """Full state-aware reduced potential: contacts + kappa_bend * n_bend_total.

    ``u(state, T) = reduced_contact_potential_state(state, T, ...) + kappa_bend *
    n_bend_total``.  This is the AUTHORITATIVE reduced potential used by the
    production sampler for observables and (through :func:`swap_log_accept_state`)
    the REMD swaps.  With ``kappa_bend == 0`` it is exactly the contacts-only
    potential; for a linear model it additionally reduces to
    :func:`reduced_potential_bending_counts` bit-for-bit.
    """
    return reduced_contact_potential_state(
        state, temperature, model_name, params, Tref, Tscale,
        lambda_intra, lambda_inter) + float(kappa_bend) * int(state.n_bend)


def reduced_potential_bending_counts(
    counts: ContactCounts, temperature: float, model_name: str, params,
    Tref: float, Tscale: float, lambda_intra: float, lambda_inter: float,
    kappa_bend: float = 0.0, n_bend: int = 0,
) -> float:
    """Full reduced potential u(X,T) = contact term + kappa_bend * n_bend_total.

    Adds the optional fixed (temperature-independent) bending penalty to the
    contact reduced potential.  With ``kappa_bend == 0`` this is exactly
    :func:`reduced_potential_counts`, so the contacts-only behaviour is unchanged.
    ``n_bend`` is the total 90-degree-turn count summed over all chains.

    Aggregate-count compatibility helper for LINEAR models only (inherits the
    nonlinear-model rejection from :func:`reduced_potential_counts`); the
    per-chain :func:`reduced_potential_state` is authoritative for every
    nonlinear model.
    """
    return reduced_potential_counts(
        counts, temperature, model_name, params, Tref, Tscale,
        lambda_intra, lambda_inter) + float(kappa_bend) * int(n_bend)


def swap_log_accept_counts(
    counts_i: ContactCounts, counts_j: ContactCounts, T_i: float, T_j: float,
    model_name: str, params, Tref: float, Tscale: float,
    lambda_intra: float, lambda_inter: float,
) -> float:
    """Generalized swap log-acceptance using BOTH intra/inter counts.

    log_accept = u(C_i,T_i) + u(C_j,T_j) - u(C_j,T_i) - u(C_i,T_j).

    The optional fixed bending penalty kappa_bend * n_bend is the SAME reduced
    (temperature-independent) parameter in every lane, so its contribution is
    kappa_bend * (n_i + n_j - n_j - n_i) = 0: it cancels exactly and never enters
    the swap criterion.  The swap therefore stays contacts-only for any kappa_bend.

    Aggregate-count compatibility helper for LINEAR models only (inherits the
    nonlinear-model rejection from :func:`reduced_potential_counts`); the
    production swap uses the per-chain :func:`swap_log_accept_state`.
    """
    def u(c, T):
        return reduced_potential_counts(
            c, T, model_name, params, Tref, Tscale, lambda_intra, lambda_inter)
    return (u(counts_i, T_i) + u(counts_j, T_j)
            - u(counts_j, T_i) - u(counts_i, T_j))


def swap_log_accept_state(
    state_i: MultiChainState, state_j: MultiChainState, T_i: float, T_j: float,
    model_name: str, params, Tref: float, Tscale: float,
    lambda_intra: float, lambda_inter: float,
) -> float:
    """Generalized full-state swap log-acceptance.

    log_accept = u(X_i,T_i) + u(X_j,T_j) - u(X_j,T_i) - u(X_i,T_j)

    evaluated through the shared state-aware contacts-only potential
    :func:`reduced_contact_potential_state`, so whatever nonlinear structure the
    model carries is evaluated with its authoritative scope: global total
    contacts for saturation and per-chain intrachain contacts for the quadratic
    family.  The fixed reduced bending penalty is
    temperature-independent and identical within each state, so it cancels
    exactly (kappa_bend * (n_i + n_j - n_j - n_i) = 0) and is omitted -- there is
    a single swap rule for all three potential families.  For linear models every
    term is bit-for-bit identical to :func:`swap_log_accept_counts`.
    """
    def u(state, T):
        return reduced_contact_potential_state(
            state, T, model_name, params, Tref, Tscale, lambda_intra, lambda_inter)
    return (u(state_i, T_i) + u(state_j, T_j)
            - u(state_j, T_i) - u(state_i, T_j))


def metropolis_accept_delta(
    delta_intra: int, delta_inter: int, reduced_bias_value: float,
    lambda_intra: float, lambda_inter: float, random_value: float,
    kappa_bend: float = 0.0, delta_bends: int = 0,
) -> bool:
    """Metropolis decision for a proposed contact-count (and bend-count) change.

    ``delta_u = b(T) * (lambda_intra * delta_intra + lambda_inter * delta_inter)
    + kappa_bend * delta_bends`` where ``b(T)`` is the (precomputed) reduced
    contact bias and ``kappa_bend * delta_bends`` is the optional fixed bending
    penalty change.  Accept when ``delta_u <= 0`` (favorable / neutral) or
    ``random_value < exp(-delta_u)``.  ``random_value`` is supplied by the caller
    so the decision is pure and directly testable; the caller draws it only for
    unfavorable moves.  ``kappa_bend`` and ``delta_bends`` default to 0, so
    contacts-only callers keep the previous behaviour exactly.
    """
    delta_u = float(reduced_bias_value) * (
        float(lambda_intra) * int(delta_intra)
        + float(lambda_inter) * int(delta_inter)) \
        + float(kappa_bend) * int(delta_bends)
    if delta_u <= 0.0:
        return True
    return float(random_value) < math.exp(-delta_u)


def attempt_swap(
    replica_a: "MultiReplica", replica_b: "MultiReplica",
    model_name: str, params, Tref: float, Tscale: float,
    lambda_intra: float, lambda_inter: float, rng: random.Random,
) -> bool:
    """Attempt a generalized REMD state exchange between two lanes.

    Computes the generalized swap log-acceptance from both cached contact counts
    at the two (fixed) lane temperatures, applies the Metropolis decision, and on
    acceptance exchanges the complete :class:`MultiChainState` objects.
    Temperatures stay pinned to their lanes; walker bookkeeping is the caller's
    responsibility.  Returns ``True`` iff the states were exchanged.

    The favorable branch (``log_accept >= 0``) short-circuits WITHOUT drawing a
    random number, matching the sampler's lazy-draw convention so run
    reproducibility (serial vs multiprocessing) is preserved.

    The generalized rule is evaluated through the shared state-aware potential
    :func:`swap_log_accept_state`, so it is correct for all three potential
    families (for linear models it is bit-for-bit identical to the historical
    aggregate-count swap).
    """
    log_accept = swap_log_accept_state(
        replica_a.state, replica_b.state, replica_a.T, replica_b.T,
        model_name, params, Tref, Tscale, lambda_intra, lambda_inter)
    if log_accept >= 0.0 or rng.random() < math.exp(log_accept):
        replica_a.state, replica_b.state = replica_b.state, replica_a.state
        return True
    return False


# ---------------------------------------------------------------------------
# Local MC sweep (one lane)
# ---------------------------------------------------------------------------

def mc_sweep(
    state: MultiChainState, counters: np.ndarray, temperature: float,
    model_name: str, params, Tref: float, Tscale: float,
    lambda_intra: float, lambda_inter: float,
    n_local: int, n_translation: int, rng: random.Random,
    n_reptation: int = 1, n_rotation: int = 1,
    debug_contacts: bool = False, kappa_bend: float = 0.0,
) -> None:
    """Run one lane's local + translation + reptation + rotation proposals in place.

    For hs and saturating_cooperative_contact, the contact contribution is the
    exact difference ``lambda_contact * [u_contact(m_total + d_total; M*N) -
    u_contact(m_total; M*N)]``, where ``d_total = d_intra + d_inter``.  Other
    models retain their existing per-chain intrachain and linear interchain
    scoring.  Accept if ``du <= 0`` or ``random() < exp(-du)``.  The fixed bending
    penalty
    ``kappa_bend * d_bends`` defaults off (``kappa_bend == 0``): the bend delta is
    still cached, but it contributes exactly 0 to ``du`` and draws no random
    number, so kappa_bend=0 runs are bit-for-bit identical to the contacts-only
    sampler.  Move counters are the shared (n_moves, 4) block: proposed /
    geometrically_valid / state_changing / metropolis_accepted (null proposals
    never count as state-changing).
    """
    # Environment-controlled contact debugging works even without the CLI flag.
    debug_contacts = bool(debug_contacts or mcc.DEBUG_CONTACTS)
    b = remd.reduced_bias(model_name, params, float(temperature), Tref, Tscale)
    li = float(lambda_intra)
    lin = float(lambda_inter)
    kb = float(kappa_bend)
    label_blind = _uses_label_blind_contacts(model_name)
    contact_scale = _label_blind_contact_scale(li, lin) if label_blind else 0.0
    # Runtime chain length for the per-chain normalization (m^2/(2N) for the
    # quadratic family, q = m/N for the saturating one); invariant under every
    # move, so it is read once per sweep.  Every model that is nonlinear in m
    # scores the intra term from the moved chain's cached m_alpha; linear models
    # keep the exact historical b*delta arithmetic (bit-for-bit).
    N = state.chain_length
    total_beads = int(state.n_chains) * int(N)
    nonlinear = remd.MODEL_REGISTRY[model_name]["potential_kind"] != "linear"

    def _attempt(prop):
        idx = mvs.MOVE_INDEX[prop.move_type]
        counters[idx, 0] += 1  # proposed
        if not prop.ok:
            return
        counters[idx, 1] += 1  # geometrically valid
        if prop.state_changing:
            counters[idx, 2] += 1  # state changing
        d_intra, d_inter = mvs.proposal_delta(state, prop)
        d_bends = mvs.proposal_delta_bends(state, prop)
        if label_blind:
            m_old = int(state.counts.intra) + int(state.counts.inter)
            m_new = m_old + int(d_intra) + int(d_inter)
            u_old = remd.reduced_contact_potential(
                m_old, temperature, model_name, params, Tref, Tscale,
                total_beads)
            u_new = remd.reduced_contact_potential(
                m_new, temperature, model_name, params, Tref, Tscale,
                total_beads)
            du = contact_scale * (u_new - u_old) + kb * int(d_bends)
            accept = (du <= 0.0) or (rng.random() < math.exp(-du))
        elif nonlinear:
            # Nonlinear intra term: evaluate the full per-chain contact potential
            # at the moved chain's OWN old and new intrachain count (never
            # approximate it as b*delta, and never evaluate it at the total
            # intrachain count).  Only the moved chain's m_alpha can change, so
            # every other chain's contribution cancels out of the difference.
            # The interchain term stays linear (b(T)*d_inter).
            alpha = int(prop.chain)
            m_old = int(state.intra_contacts_by_chain[alpha])
            m_new = m_old + int(d_intra)
            u_old = remd.reduced_contact_potential(
                m_old, temperature, model_name, params, Tref, Tscale, N)
            u_new = remd.reduced_contact_potential(
                m_new, temperature, model_name, params, Tref, Tscale, N)
            du = (li * (u_new - u_old) + lin * b * int(d_inter)
                  + kb * int(d_bends))
            # Draw a random number only for an unfavorable move (du > 0).
            accept = (du <= 0.0) or (rng.random() < math.exp(-du))
        else:
            du = b * (li * d_intra + lin * d_inter) + kb * d_bends
            # Draw a random number only for an unfavorable move (du > 0); the pure
            # helper then makes the identical decision on that draw.
            accept = (du <= 0.0) or metropolis_accept_delta(
                d_intra, d_inter, b, li, lin, rng.random(), kb, d_bends)
        if accept:
            mvs.apply_proposal(state, prop, (d_intra, d_inter), delta_bends=d_bends)
            if prop.state_changing:
                counters[idx, 3] += 1  # accepted (state-changing)
            if debug_contacts:
                mcc.assert_counts_match(state, f"after {prop.move_type}")
                mcs.assert_bends_match(state, f"after {prop.move_type}")

    for _ in range(int(n_local)):
        _attempt(mvs.propose_local(state, rng))
    for _ in range(int(n_translation)):
        _attempt(mvs.propose_translation(state, rng))
    for _ in range(int(n_reptation)):
        _attempt(mvs.propose_reptation(state, rng))
    for _ in range(int(n_rotation)):
        _attempt(mvs.propose_chain_rotation(state, rng))


# ---------------------------------------------------------------------------
# Replica
# ---------------------------------------------------------------------------

@dataclass
class MultiReplica:
    """One temperature lane: fixed T, evolving multi-chain state, trajectories."""
    T: float
    state: MultiChainState
    move_counters: np.ndarray = field(default_factory=mvs.new_move_counters)
    u_traj: list = field(default_factory=list)
    eeff_traj: list = field(default_factory=list)
    m_intra_traj: list = field(default_factory=list)
    m_inter_traj: list = field(default_factory=list)
    f_inter_traj: list = field(default_factory=list)
    rg_traj: list = field(default_factory=list)          # mean per-chain Rg (lattice)
    rg2_traj: list = field(default_factory=list)         # mean per-chain Rg^2 (lattice)
    per_chain_rg_traj: list = field(default_factory=list)  # full (M,) Rg vector/cycle (lattice)
    std_chain_rg_traj: list = field(default_factory=list)  # std of per-chain Rg (lattice)
    lcs_traj: list = field(default_factory=list)         # largest cluster size
    lcf_traj: list = field(default_factory=list)         # largest cluster fraction
    n_clusters_traj: list = field(default_factory=list)  # number of chain clusters
    dc_lattice_traj: list = field(default_factory=list)  # degree of cohesion Dc (lattice)
    dc_over_L_traj: list = field(default_factory=list)   # dimensionless Dc / L
    n_bend_traj: list = field(default_factory=list)      # cached total 90-degree turns/cycle
    # Independent-initialization provenance (recorded per lane; Change 2).
    init_seed: int = 0
    init_m_intra: int = 0
    init_m_inter: int = 0
    init_mean_chain_rg_lattice: float = 0.0  # lattice units (rg_scale applied downstream)
    init_largest_cluster_size: int = 0

    def local_acceptance_rate(self) -> float:
        c = np.asarray(self.move_counters, dtype=np.int64)
        prop = int(c[:3, 0].sum())   # pivot, crankshaft, end
        acc = int(c[:3, 3].sum())
        return acc / prop if prop else float("nan")

    def translation_acceptance_rate(self) -> float:
        c = np.asarray(self.move_counters, dtype=np.int64)
        idx = mvs.MOVE_INDEX["chain_translation"]
        prop = int(c[idx, 0])
        acc = int(c[idx, 3])
        return acc / prop if prop else float("nan")

    def reptation_acceptance_rate(self) -> float:
        c = np.asarray(self.move_counters, dtype=np.int64)
        idx = mvs.MOVE_INDEX["reptation"]
        prop = int(c[idx, 0])
        acc = int(c[idx, 3])
        return acc / prop if prop else float("nan")

    def rotation_acceptance_rate(self) -> float:
        c = np.asarray(self.move_counters, dtype=np.int64)
        idx = mvs.MOVE_INDEX["rotation"]
        prop = int(c[idx, 0])
        acc = int(c[idx, 3])
        return acc / prop if prop else float("nan")


# ---------------------------------------------------------------------------
# Worker (top-level, pickleable under Windows 'spawn')
# ---------------------------------------------------------------------------

def evolve_lane_worker(
    coords_unwrapped: np.ndarray, counts_tuple: Tuple[int, int], n_bend: int,
    intra_by_chain: np.ndarray,
    box_size: int, temperature: float, move_counters: np.ndarray,
    n_local: int, n_translation: int, n_reptation: int, n_rotation: int,
    model_name: str, params, Tref: float, Tscale: float,
    lambda_intra: float, lambda_inter: float, kappa_bend: float, seed: int,
    debug_contacts: bool,
) -> Tuple[np.ndarray, Tuple[int, int], np.ndarray, int, np.ndarray]:
    """Evolve one lane for a cycle without transferring trajectories.

    The occupancy map is rebuilt from coordinates inside the worker, so only the
    bounded coordinate array, the two cached counts, the cached bend count, the
    per-chain intrachain cache, and the small move-counter block cross the process
    boundary (no growing trajectory lists, no shared mutable occupancy).  The bend
    count and the per-chain contact cache are carried exactly like the aggregate
    counts so serial and multiprocessing runs stay bit-identical.
    """
    coords = np.asarray(coords_unwrapped, dtype=np.int64).copy()
    site_owner = mcs.build_site_owner(coords, box_size)
    state = MultiChainState(
        coords_unwrapped=coords, site_owner=site_owner,
        counts=ContactCounts.from_tuple(counts_tuple), box_size=int(box_size),
        n_bend=int(n_bend),
        intra_contacts_by_chain=np.asarray(intra_by_chain, dtype=np.int64).copy())
    counters = np.asarray(move_counters, dtype=np.int64).copy()
    rng = random.Random(int(seed))
    mc_sweep(state, counters, temperature, model_name, params, Tref, Tscale,
             lambda_intra, lambda_inter, n_local, n_translation, rng,
             n_reptation=n_reptation, n_rotation=n_rotation,
             debug_contacts=debug_contacts, kappa_bend=kappa_bend)
    return (state.coords_unwrapped, state.counts.as_tuple(), counters,
            int(state.n_bend), state.intra_contacts_by_chain)


def _lane_seed(base_seed: int, cycle: int, lane: int) -> int:
    """Deterministic per-lane, per-cycle seed (base, cycle, temperature lane)."""
    return int(base_seed) + 100_003 * int(cycle) + 131 * int(lane)


# Stride for independent per-lane INITIALIZATION seeds.  Chosen large and
# distinct from the per-cycle stride (100_003) and lane stride (131) used by
# ``_lane_seed`` so an initialization seed never coincides with any per-cycle
# sweep seed, and initialization RNG (numpy RandomState) is a different family
# from the sweep RNG (random.Random) besides.
INITIALIZATION_SEED_STRIDE = 1_000_003


def _init_lane_seed(base_seed: int, lane: int) -> int:
    """Deterministic per-lane initialization seed (independent starting states)."""
    return int(base_seed) + INITIALIZATION_SEED_STRIDE * int(lane)


def _build_independent_replicas(
    Ts: np.ndarray, M: int, N: int, L: int, base_seed: int,
    cluster_contact_threshold: int,
) -> List[MultiReplica]:
    """One independently initialized :class:`MultiReplica` per temperature lane.

    Each lane is grown from its own deterministic seed so REMD walkers do not all
    start from a single shared configuration.  Per-lane initial observables are
    recorded on the replica for the run summary.  This runs in the main process
    only, so serial and multiprocessing runs share identical starting states.
    """
    replicas: List[MultiReplica] = []
    for lane, T in enumerate(Ts):
        init_seed = _init_lane_seed(base_seed, lane)
        state = mcs.initialize_dispersed_state(M, N, L, seed=init_seed)
        clu = obs.cluster_summary(state, cluster_contact_threshold)
        rep = MultiReplica(T=float(T), state=state)
        rep.init_seed = int(init_seed)
        rep.init_m_intra = int(state.counts.intra)
        rep.init_m_inter = int(state.counts.inter)
        rep.init_mean_chain_rg_lattice = float(mcs.per_chain_rg(state).mean())
        rep.init_largest_cluster_size = int(clu["largest_cluster_size"])
        replicas.append(rep)
    return replicas


# ---------------------------------------------------------------------------
# Main REMD loop
# ---------------------------------------------------------------------------

def run_remd_multichain(
    *, n_chains: int, chain_length: int, box_size: int, Ts: np.ndarray,
    local_sweeps_per_swap: int, translation_sweeps_per_swap: int, n_cycles: int,
    reptation_sweeps_per_swap: int = 1, rotation_sweeps_per_swap: int = 1,
    model_name: str, params, Tref: float, Tscale: float,
    lambda_intra: float = 1.0, lambda_inter: float = 1.0,
    kappa_bend: float = 0.0,
    cluster_contact_threshold: int = 1,
    seed: int | None = None, n_workers: int = 1, verbose: bool = True,
    debug_contacts: bool = False,
    snapshot_writer=None, snapshot_stride: int = 1, snapshot_start_cycle: int = 0,
) -> Tuple[List[MultiReplica], np.ndarray, np.ndarray, np.ndarray]:
    """Run multi-chain REMD.

    ``kappa_bend`` is the optional fixed reduced bending penalty applied to the
    total 90-degree-turn count (``kappa_bend == 0`` reproduces the contacts-only
    sampler exactly).  Returns ``(replicas, swap_props, swap_accs,
    walker_lane_history)`` where the last is ``(n_cycles, nT)`` giving each
    walker's lane index per cycle (for round-trip diagnostics).
    """
    if seed is None:
        seed = random.SystemRandom().randrange(1, 2 ** 31)
    base_seed = int(seed)
    kappa_bend = float(kappa_bend)
    if not math.isfinite(kappa_bend) or kappa_bend < 0.0:
        raise ValueError(f"kappa_bend must be finite and >= 0, got {kappa_bend!r}")
    if _uses_label_blind_contacts(model_name):
        _label_blind_contact_scale(lambda_intra, lambda_inter)
    # Environment-controlled contact debugging works even without the CLI flag.
    debug_contacts = bool(debug_contacts or mcc.DEBUG_CONTACTS)

    M = int(n_chains)
    N = int(chain_length)
    L = int(box_size)
    nT = len(Ts)
    n_local = int(local_sweeps_per_swap) * M * N
    n_translation = int(translation_sweeps_per_swap) * M
    n_reptation = int(reptation_sweeps_per_swap) * M
    n_rotation = int(rotation_sweeps_per_swap) * M

    # Each REMD lane begins from its OWN independently generated dispersed state
    # (deterministic per-lane seeds); no shared starting configuration.
    replicas = _build_independent_replicas(
        Ts, M, N, L, base_seed, cluster_contact_threshold)

    swap_props = np.zeros(max(nT - 1, 0), dtype=np.int64)
    swap_accs = np.zeros(max(nT - 1, 0), dtype=np.int64)
    lane_walker = np.arange(nT, dtype=np.int64)   # walker currently in lane k
    walker_lane = np.arange(nT, dtype=np.int64)   # lane of walker w
    walker_lane_history = np.empty((n_cycles, nT), dtype=np.int32)

    save_configurations = snapshot_writer is not None
    report_every = max(1, n_cycles // 10)

    executor = (
        ProcessPoolExecutor(max_workers=min(n_workers, nT),
                            mp_context=_mp.get_context("spawn"))
        if n_workers > 1 else None)

    try:
        for cycle in range(n_cycles):
            if executor is not None:
                futures = [
                    executor.submit(
                        evolve_lane_worker,
                        replicas[k].state.coords_unwrapped,
                        replicas[k].state.counts.as_tuple(),
                        int(replicas[k].state.n_bend),
                        replicas[k].state.intra_contacts_by_chain,
                        L, replicas[k].T, replicas[k].move_counters,
                        n_local, n_translation, n_reptation, n_rotation,
                        model_name, params, Tref, Tscale,
                        lambda_intra, lambda_inter, kappa_bend,
                        _lane_seed(base_seed, cycle, k), debug_contacts)
                    for k in range(nT)]
                for k, fut in enumerate(futures):
                    coords, counts_tuple, counters, n_bend, intra_by_chain = \
                        fut.result()
                    rep = replicas[k]
                    rep.state = MultiChainState(
                        coords_unwrapped=np.asarray(coords, dtype=np.int64),
                        site_owner=mcs.build_site_owner(coords, L),
                        counts=ContactCounts.from_tuple(counts_tuple), box_size=L,
                        n_bend=int(n_bend),
                        intra_contacts_by_chain=np.asarray(
                            intra_by_chain, dtype=np.int64))
                    rep.move_counters = counters
            else:
                for k, rep in enumerate(replicas):
                    rng = random.Random(_lane_seed(base_seed, cycle, k))
                    mc_sweep(rep.state, rep.move_counters, rep.T, model_name,
                             params, Tref, Tscale, lambda_intra, lambda_inter,
                             n_local, n_translation, rng,
                             n_reptation=n_reptation, n_rotation=n_rotation,
                             debug_contacts=debug_contacts, kappa_bend=kappa_bend)

            # Even/odd adjacent swaps (temperatures fixed to lanes).  The actual
            # production swap operation is attempt_swap; the deterministic
            # per-(cycle, pair) RNG keeps serial and multiprocessing runs
            # identical (swaps always run in the main process).
            start = cycle % 2
            for k in range(start, nT - 1, 2):
                swap_props[k] += 1
                swap_rng = random.Random(_lane_seed(base_seed, cycle, 1000 + k))
                if attempt_swap(replicas[k], replicas[k + 1], model_name, params,
                                Tref, Tscale, lambda_intra, lambda_inter,
                                swap_rng):
                    swap_accs[k] += 1
                    wa, wb = lane_walker[k], lane_walker[k + 1]
                    lane_walker[k], lane_walker[k + 1] = wb, wa
                    walker_lane[wa], walker_lane[wb] = k + 1, k
            walker_lane_history[cycle] = walker_lane

            # Observable recording (main process only).
            do_snapshot = (
                save_configurations and cycle >= snapshot_start_cycle
                and ((cycle - snapshot_start_cycle) % snapshot_stride) == 0)
            if do_snapshot:
                snap_coords = np.empty((nT, M, N, 3), dtype=np.int64)
                snap_mi = np.empty(nT, dtype=np.int64)
                snap_me = np.empty(nT, dtype=np.int64)
                snap_rg2 = np.empty(nT, dtype=np.float64)
                snap_lcs = np.empty(nT, dtype=np.int64)

            for k, rep in enumerate(replicas):
                cyc = obs.cycle_observables(rep.state, cluster_contact_threshold)
                u = reduced_potential_state(
                    rep.state, rep.T, model_name, params, Tref, Tscale,
                    lambda_intra, lambda_inter, kappa_bend)
                rep.u_traj.append(u)
                rep.eeff_traj.append(rep.T * u)
                rep.m_intra_traj.append(cyc["m_intra"])
                rep.m_inter_traj.append(cyc["m_inter"])
                rep.f_inter_traj.append(cyc["f_inter"])
                rep.rg_traj.append(cyc["mean_chain_rg_lattice"])
                rep.rg2_traj.append(cyc["mean_chain_rg2_lattice"])
                rep.per_chain_rg_traj.append(
                    np.asarray(cyc["per_chain_rg_lattice"], dtype=np.float64))
                rep.std_chain_rg_traj.append(cyc["std_chain_rg_lattice"])
                rep.lcs_traj.append(cyc["largest_cluster_size"])
                rep.lcf_traj.append(cyc["largest_cluster_fraction"])
                rep.n_clusters_traj.append(cyc["n_clusters"])
                rep.dc_lattice_traj.append(cyc["dc_lattice"])
                rep.dc_over_L_traj.append(cyc["dc_over_L"])
                # Cached total bend count (O(1) read); additive supplementary
                # observable, never alters sampling.
                rep.n_bend_traj.append(int(rep.state.n_bend))
                if do_snapshot:
                    snap_coords[k] = rep.state.coords_unwrapped
                    snap_mi[k] = cyc["m_intra"]
                    snap_me[k] = cyc["m_inter"]
                    snap_rg2[k] = cyc["mean_chain_rg2_lattice"]
                    snap_lcs[k] = cyc["largest_cluster_size"]

            if do_snapshot:
                snapshot_writer.append(
                    cycle=cycle, coordinates=snap_coords,
                    walker_id=lane_walker.astype(np.int64),
                    m_intra=snap_mi, m_inter=snap_me,
                    mean_chain_rg2=snap_rg2, largest_cluster_size=snap_lcs)

            if verbose and (cycle + 1) % report_every == 0:
                rates = " ".join(
                    f"{swap_accs[k] / swap_props[k]:.2f}" if swap_props[k] else "n/a"
                    for k in range(nT - 1))
                print(f"  cycle {cycle + 1:>5}/{n_cycles}  swap: {rates}")
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    return replicas, swap_props, swap_accs, walker_lane_history


# ---------------------------------------------------------------------------
# Post-processing helpers
# ---------------------------------------------------------------------------

def _traj_dicts(replicas: List[MultiReplica]) -> List[dict]:
    return [{
        "u": rep.u_traj, "effective_energy": rep.eeff_traj,
        "m_intra": rep.m_intra_traj, "m_inter": rep.m_inter_traj,
        "f_inter": rep.f_inter_traj,
        "mean_chain_rg_lattice": rep.rg_traj,
        "mean_chain_rg2_lattice": rep.rg2_traj,
        "std_chain_rg_lattice": rep.std_chain_rg_traj,
        "per_chain_rg_lattice": rep.per_chain_rg_traj,
        "largest_cluster_size": rep.lcs_traj,
        "largest_cluster_fraction": rep.lcf_traj,
        "n_clusters": rep.n_clusters_traj,
        "dc_lattice": rep.dc_lattice_traj,
        "dc_over_L": rep.dc_over_L_traj,
        # M carried per lane so contact normalization uses an explicit chain
        # count (never inferred from cluster sizes).
        "n_chains": rep.state.n_chains,
    } for rep in replicas]


def _bend_fraction_by_temp(replicas, M: int, N: int, burnin_frac: float
                           ) -> np.ndarray:
    """Post-burn-in mean bend fraction n_bend_total / (M * (N - 2)) per lane.

    ``n_bend_total`` is the total 90-degree-turn count summed over all M chains,
    and each chain has ``N - 2`` interior angle centers, so the maximally-bent
    reference is ``M * (N - 2)``.  Additive summary of the cached bend-count
    trajectory; NaN for lanes with no post-burn-in samples and, for every lane,
    when the chains are too short to bend (N <= 2).
    """
    denom = int(M) * (int(N) - 2)
    out = np.full(len(replicas), np.nan, dtype=float)
    if denom <= 0:
        return out
    for i, rep in enumerate(replicas):
        arr = np.asarray(rep.n_bend_traj, dtype=float)
        s = int(math.floor(arr.size * float(burnin_frac)))
        post = arr[s:]
        if post.size:
            out[i] = float(post.mean()) / denom
    return out


def summarize_results(replicas, Ts, burnin_frac, rg_scale):
    traj = _traj_dicts(replicas)
    n_chains = replicas[0].state.n_chains if replicas else None
    results = obs.compute_statistics(traj, Ts, burnin_frac=burnin_frac,
                                     rg_scale=rg_scale, n_chains=n_chains)
    for rep, r in zip(replicas, results):
        r["local_acceptance_rate"] = rep.local_acceptance_rate()
        r["translation_acceptance_rate"] = rep.translation_acceptance_rate()
        r["reptation_acceptance_rate"] = rep.reptation_acceptance_rate()
        r["rotation_acceptance_rate"] = rep.rotation_acceptance_rate()
    return results


def build_run_distributions(replicas, Ts, *, burnin_frac, rg_bins, rg_scale):
    dist = obs.build_distributions(
        _traj_dicts(replicas), Ts, burnin_frac=burnin_frac, rg_bins=rg_bins,
        rg_scale=rg_scale)
    # Additive bending trajectories/observables (never touch the canonical
    # contact/Rg distributions).  ``n_bends`` is the per-lane per-cycle total
    # 90-degree-turn count; ``bend_fraction`` is its post-burn-in mean divided by
    # the maximally-bent reference M*(N-2) (NaN for N <= 2).
    M = replicas[0].state.n_chains if replicas else 0
    N = replicas[0].state.chain_length if replicas else 0
    dist["n_bends"] = np.array(
        [np.asarray(r.n_bend_traj, dtype=np.int64) for r in replicas],
        dtype=np.int64) if replicas else np.zeros((0, 0), dtype=np.int64)
    dist["bend_fraction"] = _bend_fraction_by_temp(replicas, M, N, burnin_frac)
    return dist


def attach_metadata(dist: dict, *, M, N, L, Ts, seed, model_name, param_names,
                    model_params, Tref, Tscale, lambda_intra, lambda_inter,
                    local_sweeps_per_swap, translation_sweeps_per_swap, n_cycles,
                    burnin_frac, cluster_contact_threshold, parameter_source,
                    fit_summary_json, reptation_sweeps_per_swap=1,
                    rotation_sweeps_per_swap=1, kappa_bend=0.0,
                    fit_chain_length=None) -> dict:
    """Inject full model + run provenance into a distributions/summary dict."""
    dist["schema_version"] = int(SCHEMA_VERSION)
    dist["model_api_version"] = int(MODEL_API_VERSION)
    dist["model_name"] = str(model_name)
    dist["param_names"] = np.array(list(param_names))
    dist["model_params"] = np.array([float(v) for v in model_params], dtype=float)
    dist["Tref"] = float(Tref)
    dist["Tscale"] = float(Tscale)
    dist["lambda_intra"] = float(lambda_intra)
    dist["lambda_inter"] = float(lambda_inter)
    # Contact-potential contract.  hs and saturating_cooperative_contact use one
    # label-blind total-contact Hamiltonian; the other models retain the split
    # per-chain-intra / linear-inter definition.
    # potential_definition is the single-chain u_contact actually sampled and
    # multichain_potential_definition is the assembled multi-chain potential;
    # potential_normalization says what the chain length was used for ("" when it
    # was not needed) and m_ref is the contact-number reference (0 everywhere).
    spec = remd.MODEL_REGISTRY[model_name]
    dist["potential_kind"] = str(spec["potential_kind"])
    dist["quadratic_contact_scope"] = _contact_scope(model_name)
    dist["nonlinear_contact_scope"] = _contact_scope(model_name)
    dist["interchain_contact_model"] = _interchain_contact_model(model_name)
    dist["quadratic_normalization"] = (
        "m_chain^2/(2*N)" if spec["potential_kind"] == "contact_quadratic" else "")
    dist["potential_definition"] = str(spec["potential_definition"])
    dist["potential_normalization"] = str(spec["potential_normalization"] or "")
    dist["m_ref"] = int(spec["m_ref"])
    dist["multichain_potential_definition"] = \
        _multichain_potential_definition(model_name)
    if str(spec["potential_kind"]) == "saturating_cooperative":
        # Named saturating parameters alongside the packed model_params vector,
        # mirroring the single-chain sampler's convention.
        _A0, _q_sat = remd.validated_saturating_params(model_params)
        dist["A0"] = float(_A0)
        dist["q_sat"] = float(_q_sat)
    dist["runtime_chain_length"] = int(N)
    dist["fit_chain_length"] = int(-1 if fit_chain_length is None
                                   else fit_chain_length)
    # Additive bending-penalty provenance (metadata only; the trajectories
    # ``n_bends`` / ``bend_fraction`` are attached in build_run_distributions).
    dist["kappa_bend"] = float(kappa_bend)
    dist["bending_enabled"] = bool(float(kappa_bend) != 0.0)
    dist["bend_definition"] = BEND_DEFINITION
    dist["M"] = int(M)
    dist["N"] = int(N)
    dist["L"] = int(L)
    dist["n_beads"] = int(M * N)
    dist["volume_fraction"] = float(M * N) / float(L ** 3)
    dist["seed"] = int(seed)
    dist["temperatures"] = np.asarray(Ts, dtype=float)
    dist["local_sweeps_per_swap"] = int(local_sweeps_per_swap)
    dist["translation_sweeps_per_swap"] = int(translation_sweeps_per_swap)
    dist["reptation_sweeps_per_swap"] = int(reptation_sweeps_per_swap)
    dist["rotation_sweeps_per_swap"] = int(rotation_sweeps_per_swap)
    dist["local_proposals_per_cycle"] = int(local_sweeps_per_swap) * M * N
    dist["translation_proposals_per_cycle"] = int(translation_sweeps_per_swap) * M
    dist["reptation_proposals_per_cycle"] = int(reptation_sweeps_per_swap) * M
    dist["rotation_proposals_per_cycle"] = int(rotation_sweeps_per_swap) * M
    dist["n_cycles"] = int(n_cycles)
    dist["burnin_frac"] = float(burnin_frac)
    dist["cluster_contact_threshold"] = int(cluster_contact_threshold)
    dist["parameter_source"] = str(parameter_source)
    dist["fit_summary_json"] = str(fit_summary_json) if fit_summary_json else ""
    # Per-temperature reduced bias (model-independent bookkeeping).  The potential
    # is no longer described by a single b(T) curve once the model is nonlinear in
    # m, so the linear coefficient b(T) and the quadratic coefficient kappa(T)
    # (coefficient of m^2/(2N); exactly 0 for linear models AND for the
    # saturating-cooperative model, which has no m^2/(2N) term at all -- read
    # potential_definition, A0 and q_sat for that one) are stored separately.
    # reduced_bias_by_temperature is retained (== the linear coefficient) for
    # backward compatibility.
    linear_coeff = np.array(
        [remd.reduced_bias(model_name, model_params, float(T), Tref, Tscale)
         for T in Ts], dtype=float)
    quadratic_coeff = np.array(
        [remd.quadratic_bias(model_name, model_params, float(T), Tref, Tscale)
         for T in Ts], dtype=float)
    dist["reduced_bias_by_temperature"] = linear_coeff
    dist["linear_coefficient_by_temperature"] = linear_coeff
    dist["quadratic_coefficient_by_temperature"] = quadratic_coeff
    return dist


# ---------------------------------------------------------------------------
# Plotting (lazy; minimal smoke plots)
# ---------------------------------------------------------------------------

def make_plots(results, dist, out_prefix: str) -> None:
    plt = _import_matplotlib()
    if plt is None:
        print("  [plot] matplotlib unavailable; skipping plots")
        return
    Ts = [r["T"] for r in results]

    def col(key):
        return [r.get(key, float("nan")) for r in results]

    # Error bars are standard errors of the mean (std/sqrt(ESS), autocorrelation
    # corrected) -- NOT the *_std ensemble fluctuation widths, which measure the
    # physical spread of the distribution rather than the uncertainty on its mean.
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    axes[0].errorbar(Ts, col("Rg_mean"), yerr=col("Rg_sem"), fmt="o-", capsize=3)
    axes[0].set_xlabel("T"); axes[0].set_ylabel("mean per-chain Rg")
    axes[1].errorbar(Ts, col("m_intra_mean"), yerr=col("m_intra_sem"),
                     fmt="o-", capsize=3, label="m_intra")
    axes[1].errorbar(Ts, col("m_inter_mean"), yerr=col("m_inter_sem"),
                     fmt="s-", capsize=3, label="m_inter")
    axes[1].set_xlabel("T"); axes[1].set_ylabel("mean contacts"); axes[1].legend()
    axes[2].errorbar(Ts, col("largest_cluster_fraction_mean"),
                     yerr=col("largest_cluster_fraction_sem"), fmt="o-", capsize=3)
    axes[2].set_xlabel("T"); axes[2].set_ylabel("largest cluster fraction")
    axes[3].errorbar(Ts, col("Dc_over_L_mean"), yerr=col("Dc_over_L_sem"),
                     fmt="o-", capsize=3)
    # Uncorrelated uniform chain COMs give Dc/L = 0.4803; aggregation falls below it.
    axes[3].axhline(0.4803, ls="--", color="gray", label="uncorrelated")
    axes[3].set_xlabel("T"); axes[3].set_ylabel("Dc / L"); axes[3].legend()
    fig.tight_layout()
    path = f"{out_prefix}_observables.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def _resolve_temperatures(args) -> Tuple[np.ndarray, str]:
    if args.temps_from_npz is not None:
        return remd.load_temperatures_from_npz(args.temps_from_npz), \
            f"npz:{args.temps_from_npz}"
    if args.temps is not None:
        Ts = remd.validate_temperature_ladder(
            np.array(remd.parse_params_string(args.temps), dtype=float))
        return Ts, "cli:--temps"
    if args.nT < 2:
        raise ValueError("--nT must be >= 2")
    if args.Tmax <= args.Tmin:
        raise ValueError("--Tmax must be greater than --Tmin")
    return np.linspace(args.Tmin, args.Tmax, args.nT), "linspace"


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Replica-exchange MC for the multi-chain lattice aggregation model.")
    ap.add_argument("--n-chains", type=int, default=4, dest="n_chains")
    ap.add_argument("--N", type=int, default=20, help="beads per chain")
    ap.add_argument("--box-size", type=int, default=16, dest="box_size")
    ap.add_argument("--local-sweeps-per-swap", type=int, default=1,
                    dest="local_sweeps_per_swap")
    ap.add_argument("--translation-sweeps-per-swap", type=int, default=1,
                    dest="translation_sweeps_per_swap")
    ap.add_argument("--reptation-sweeps-per-swap", type=int, default=1,
                    dest="reptation_sweeps_per_swap")
    ap.add_argument("--rotation-sweeps-per-swap", type=int, default=1,
                    dest="rotation_sweeps_per_swap")
    ap.add_argument("--n-cycles", type=int, default=200, dest="n_cycles")
    ap.add_argument("--lambda-intra", type=float, default=1.0, dest="lambda_intra")
    ap.add_argument("--lambda-inter", type=float, default=1.0, dest="lambda_inter")
    ap.add_argument("--kappa-bend", type=float, default=None, dest="kappa_bend",
                    help="fixed reduced bending penalty: u_bend = kappa_bend * "
                         "n_bend_total (n_bend_total = total 90-degree turns over "
                         "all chains). Loaded from the fit summary when present "
                         "(a supplied value must match it); kappa_bend > 0 enables "
                         "bending, 0 (default) disables it.")
    ap.add_argument("--cluster-contact-threshold", type=int, default=1,
                    dest="cluster_contact_threshold")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-workers", type=int, default=1, dest="n_workers")
    ap.add_argument("--out-prefix", type=str, default="remd_multichain_out",
                    dest="out_prefix")
    ap.add_argument("--rg-scale", type=float, default=1.0, dest="rg_scale")
    ap.add_argument("--rg-bins", type=int, default=40, dest="rg_bins")
    ap.add_argument("--burnin-frac", type=float, default=0.5, dest="burnin_frac")
    # Model / parameter sources (reuse the single-chain resolution semantics).
    ap.add_argument("--model", type=str, default=None,
                    choices=list(remd.MODEL_REGISTRY.keys()))
    ap.add_argument("--fit-summary-json", type=str, default=None,
                    dest="fit_summary_json",
                    help="preferred production parameter source")
    ap.add_argument("--fit-params-csv", type=str, default=None,
                    dest="fit_params_csv")
    ap.add_argument("--params", type=str, default=None)
    ap.add_argument("--Tref", type=float, default=None)
    ap.add_argument("--Tscale", type=float, default=None)
    ap.add_argument("--T0", type=float, default=None)
    ap.add_argument("--dh", type=float, default=312.109,
                    help="legacy hs enthalpy (used only for --model hs default)")
    ap.add_argument("--ds", type=float, default=1.23278,
                    help="legacy hs entropy (used only for --model hs default)")
    # Temperatures.
    ap.add_argument("--Tmin", type=float, default=300.0)
    ap.add_argument("--Tmax", type=float, default=360.0)
    ap.add_argument("--nT", type=int, default=8)
    tg = ap.add_mutually_exclusive_group()
    tg.add_argument("--temps-from-npz", type=str, default=None,
                    dest="temps_from_npz")
    tg.add_argument("--temps", type=str, default=None)
    # Optional outputs.
    ap.add_argument("--diagnostics", action="store_true")
    ap.add_argument("--diagnostic-trajectories", action="store_true",
                    dest="diagnostic_trajectories")
    ap.add_argument("--save-configurations", action="store_true",
                    dest="save_configurations")
    ap.add_argument("--snapshot-stride", type=int, default=1,
                    dest="snapshot_stride")
    ap.add_argument("--configuration-path", type=str, default=None,
                    dest="configuration_path")
    ap.add_argument("--overwrite-configurations", action="store_true",
                    dest="overwrite_configurations")
    ap.add_argument("--debug-contacts", action="store_true", dest="debug_contacts",
                    help="assert cached counts == full oracle after every accepted move")
    ap.add_argument("--no-plots", action="store_true", dest="no_plots")
    ap.add_argument("--quick-test", action="store_true", dest="quick_test")
    return ap


def _validate_cli(args) -> None:
    if args.n_chains < 1:
        raise ValueError("--n-chains must be >= 1")
    if args.N < 3:
        raise ValueError("--N must be >= 3")
    if args.box_size < mcs.MIN_BOX_SIZE:
        raise ValueError(f"--box-size must be >= {mcs.MIN_BOX_SIZE}")
    if args.n_chains * args.N > args.box_size ** 3:
        raise ValueError(
            f"volume fraction > 1: M*N={args.n_chains * args.N} > "
            f"L^3={args.box_size ** 3}")
    if (args.local_sweeps_per_swap < 0 or args.translation_sweeps_per_swap < 0
            or args.reptation_sweeps_per_swap < 0
            or args.rotation_sweeps_per_swap < 0):
        raise ValueError("sweeps-per-swap must be >= 0")
    if (args.local_sweeps_per_swap + args.translation_sweeps_per_swap
            + args.reptation_sweeps_per_swap + args.rotation_sweeps_per_swap) < 1:
        raise ValueError(
            "need at least one local, translation, reptation, or rotation sweep "
            "per swap")
    if args.n_cycles < 1:
        raise ValueError("--n-cycles must be >= 1")
    if not (0.0 <= args.burnin_frac < 1.0):
        raise ValueError("--burnin-frac must be in [0, 1)")
    if args.n_workers < 1:
        raise ValueError("--n-workers must be >= 1")
    if not math.isfinite(args.rg_scale) or args.rg_scale <= 0:
        raise ValueError("--rg-scale must be finite and positive")
    if args.rg_bins < 1:
        raise ValueError("--rg-bins must be >= 1")
    if args.cluster_contact_threshold < 1:
        raise ValueError("--cluster-contact-threshold must be >= 1")
    for name in ("lambda_intra", "lambda_inter"):
        v = getattr(args, name)
        if not math.isfinite(v):
            raise ValueError(f"--{name.replace('_', '-')} must be finite")
    # Snapshot / diagnostic flag consistency (reject before initialization).
    if args.save_configurations and args.snapshot_stride < 1:
        raise ValueError("--snapshot-stride must be >= 1 when saving configurations")
    if args.diagnostic_trajectories and not args.diagnostics:
        raise ValueError("--diagnostic-trajectories requires --diagnostics")
    if args.configuration_path is not None and not args.save_configurations:
        raise ValueError("--configuration-path requires --save-configurations")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv=None) -> None:
    args = build_arg_parser().parse_args(argv)
    if args.quick_test:
        run_quick_test()
        return

    _validate_cli(args)
    Ts, temp_source = _resolve_temperatures(args)
    # Final guard: the resolved ladder must be 1-D, >= 2 entries, finite,
    # positive, and strictly increasing regardless of how it was produced.
    Ts = remd.validate_temperature_ladder(Ts)
    (model_name, model_params, param_names, Tref, Tscale,
     parameter_source, fit_summary_json) = remd.resolve_model_params(args, Ts)

    if _uses_label_blind_contacts(model_name):
        _label_blind_contact_scale(args.lambda_intra, args.lambda_inter)

    # Validate the fitted single-chain normalization.  The label-blind saturation
    # Hamiltonian uses M*N total beads at evaluation time, while N remains the
    # physical chain length and the fit-transfer provenance value.
    remd.validate_chain_length(model_name, args.N)

    # Resolve the fixed reduced bending penalty AFTER the model, reusing the
    # single-chain rules: a fit summary is authoritative (re-read for kappa_bend,
    # already validated in resolve_model_params) and cross-checks any CLI value;
    # without a summary the CLI value applies (default 0.0).  kappa_bend > 0
    # enables bending; there is no separate flag.  The fit summary also carries
    # ``fit_chain_length`` for provenance; chain-length transfer is permitted (the
    # sampler always uses the RUNTIME N in m^2/(2N)) with a note on mismatch.
    fit_chain_length = None
    if fit_summary_json:
        _summary = remd.load_fit_summary_json(fit_summary_json)
        summary_kappa = _summary["kappa_bend"]
        fit_chain_length = _summary["fit_chain_length"]
        kappa_bend = remd.resolve_kappa_bend(
            args.kappa_bend, summary_kappa, True, source=fit_summary_json)
    else:
        kappa_bend = remd.resolve_kappa_bend(args.kappa_bend, None, False)
    bending_enabled = bool(kappa_bend != 0.0)

    M, N, L = args.n_chains, args.N, args.box_size
    _spec = remd.MODEL_REGISTRY[model_name]
    if (_spec["potential_kind"] != "linear"
            and fit_chain_length is not None and int(fit_chain_length) != int(N)):
        print(f"Note: model {model_name} (nonlinear in m) fit at chain length "
              f"{int(fit_chain_length)} is being run at N={int(N)}; the "
              f"{_spec['potential_normalization'] or 'normalization'} uses the "
              f"runtime N (chain-length transfer).")
    phi = M * N / L ** 3
    print(f"Multi-chain REMD: M={M} N={N} L={L} phi={phi:.4f}, {len(Ts)} lanes "
          f"T in [{Ts.min():.4g}, {Ts.max():.4g}] ({temp_source})")
    print(f"Model: {model_name} — {_spec['description']}")
    print(f"Single-chain contact potential: {_spec['potential_definition']}")
    print(f"Multichain potential: {_multichain_potential_definition(model_name)}")
    print(f"m_ref = {int(_spec['m_ref'])}")
    print(f"Parameter source: {parameter_source}"
          + (f" ({fit_summary_json})" if fit_summary_json else ""))
    print(f"lambda_intra={args.lambda_intra} lambda_inter={args.lambda_inter} "
          f"cluster_contact_threshold={args.cluster_contact_threshold}")
    print(f"Bending penalty: kappa_bend = {kappa_bend:g} "
          f"({'enabled' if bending_enabled else 'disabled'}; {BEND_DEFINITION})")

    Path(args.out_prefix).parent.mkdir(parents=True, exist_ok=True)

    snapshot_writer = None
    configuration_path = None
    if args.save_configurations:
        if not mcio.h5py_available():
            raise RuntimeError("--save-configurations requires h5py")
        configuration_path = (args.configuration_path
                              or f"{args.out_prefix}_configurations.h5")
        snap_meta = _snapshot_metadata(
            args, Ts, model_name, param_names, model_params, Tref, Tscale,
            temp_source, phi, fit_summary_json, kappa_bend, bending_enabled,
            fit_chain_length)
        snapshot_writer = mcio.MultiChainSnapshotWriter(
            configuration_path, n_chains=M, chain_length=N,
            n_temperatures=len(Ts), metadata=snap_meta,
            overwrite=args.overwrite_configurations)

    t0 = time.perf_counter()
    try:
        replicas, swap_props, swap_accs, walker_hist = run_remd_multichain(
            n_chains=M, chain_length=N, box_size=L, Ts=Ts,
            local_sweeps_per_swap=args.local_sweeps_per_swap,
            translation_sweeps_per_swap=args.translation_sweeps_per_swap,
            reptation_sweeps_per_swap=args.reptation_sweeps_per_swap,
            rotation_sweeps_per_swap=args.rotation_sweeps_per_swap,
            n_cycles=args.n_cycles, model_name=model_name, params=model_params,
            Tref=Tref, Tscale=Tscale,
            lambda_intra=args.lambda_intra, lambda_inter=args.lambda_inter,
            kappa_bend=kappa_bend,
            cluster_contact_threshold=args.cluster_contact_threshold,
            seed=args.seed, n_workers=args.n_workers, verbose=True,
            debug_contacts=args.debug_contacts,
            snapshot_writer=snapshot_writer, snapshot_stride=args.snapshot_stride)
        if snapshot_writer is not None:
            snapshot_writer.update_metadata({
                "end_time": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())})
            snapshot_writer.mark_complete()
    finally:
        if snapshot_writer is not None:
            snapshot_writer.close()
    wall = time.perf_counter() - t0

    results = summarize_results(replicas, Ts, args.burnin_frac, args.rg_scale)
    dist = build_run_distributions(replicas, Ts, burnin_frac=args.burnin_frac,
                                   rg_bins=args.rg_bins, rg_scale=args.rg_scale)
    attach_metadata(
        dist, M=M, N=N, L=L, Ts=Ts, seed=args.seed, model_name=model_name,
        param_names=param_names, model_params=model_params, Tref=Tref,
        Tscale=Tscale, lambda_intra=args.lambda_intra,
        lambda_inter=args.lambda_inter,
        local_sweeps_per_swap=args.local_sweeps_per_swap,
        translation_sweeps_per_swap=args.translation_sweeps_per_swap,
        reptation_sweeps_per_swap=args.reptation_sweeps_per_swap,
        rotation_sweeps_per_swap=args.rotation_sweeps_per_swap,
        n_cycles=args.n_cycles, burnin_frac=args.burnin_frac,
        cluster_contact_threshold=args.cluster_contact_threshold,
        parameter_source=parameter_source, fit_summary_json=fit_summary_json,
        kappa_bend=kappa_bend, fit_chain_length=fit_chain_length)

    out_files = {
        "results_csv": mcio.save_results_csv(results, args.out_prefix),
        "swap_rates_csv": mcio.save_swap_csv(swap_props, swap_accs, Ts, args.out_prefix),
        "move_acceptance_csv": mcio.save_move_acceptance_csv(
            [r.move_counters for r in replicas], mvs.MOVE_TYPES, Ts, args.out_prefix),
        "distributions_npz": mcio.save_distributions_npz(dist, args.out_prefix),
    }
    if configuration_path is not None:
        out_files["configuration_hdf5"] = configuration_path
        print(f"Saved {configuration_path} ({snapshot_writer.n_snapshots} snapshots)")

    # Convergence / mixing diagnostics (reuse the validated single-chain
    # primitives via the multi-chain adapter).
    diagnostics = None
    if args.diagnostics:
        diagnostics = mcd.compute_multichain_diagnostics(
            replicas, swap_props, swap_accs, walker_hist, Ts,
            burnin_frac=args.burnin_frac, rg_scale=args.rg_scale)
        diag_files = mcd.save_all_diagnostics(
            diagnostics, replicas, walker_hist, Ts, out_prefix=args.out_prefix,
            burnin_frac=args.burnin_frac, rg_scale=args.rg_scale,
            save_trajectories=args.diagnostic_trajectories)
        out_files.update(diag_files)
        nwarn = len(diagnostics["warnings"])
        print(f"Diagnostics: {nwarn} warning(s)"
              + (":" if nwarn else "."))
        for w in diagnostics["warnings"]:
            print(f"  [warn:{w['type']}] {w['message']}")

    if not args.no_plots:
        make_plots(results, dist, args.out_prefix)

    run_summary = _run_summary(
        args, Ts, temp_source, model_name, param_names, model_params, Tref,
        Tscale, parameter_source, fit_summary_json, phi, swap_props, swap_accs,
        results, wall, out_files, replicas, diagnostics,
        configuration_path is not None, kappa_bend, bending_enabled,
        fit_chain_length)
    summary_path = f"{args.out_prefix}_run_summary.json"
    out_files["run_summary_json"] = summary_path
    mcio.save_run_summary(run_summary, summary_path)


def _snapshot_metadata(args, Ts, model_name, param_names, model_params, Tref,
                       Tscale, temp_source, phi, fit_summary_json,
                       kappa_bend=0.0, bending_enabled=False,
                       fit_chain_length=None) -> dict:
    spec = remd.MODEL_REGISTRY[model_name]
    meta = {
        "schema_version": int(mcio.MULTICHAIN_SNAPSHOT_SCHEMA_VERSION),
        "run_id": Path(args.out_prefix).name,
        "M": int(args.n_chains), "N": int(args.N), "L": int(args.box_size),
        "volume_fraction": float(phi),
        "lambda_intra": float(args.lambda_intra),
        "lambda_inter": float(args.lambda_inter),
        # Additive bending-penalty provenance (does not change coordinate storage).
        "kappa_bend": float(kappa_bend),
        "bending_enabled": bool(bending_enabled),
        "bend_definition": BEND_DEFINITION,
        # Contact-potential contract: the Hamiltonian this snapshot file was
        # produced under, stated in full rather than implied by the model name.
        "potential_kind": str(spec["potential_kind"]),
        "quadratic_contact_scope": _contact_scope(model_name),
        "nonlinear_contact_scope": _contact_scope(model_name),
        "interchain_contact_model": _interchain_contact_model(model_name),
        "quadratic_normalization": (
            "m_chain^2/(2*N)"
            if spec["potential_kind"] == "contact_quadratic" else "null"),
        "potential_definition": str(spec["potential_definition"]),
        "potential_normalization": (
            str(spec["potential_normalization"])
            if spec["potential_normalization"] is not None else "null"),
        "m_ref": int(spec["m_ref"]),
        "multichain_potential_definition":
            _multichain_potential_definition(model_name),
        "runtime_chain_length": int(args.N),
        "fit_chain_length": (int(fit_chain_length)
                             if fit_chain_length is not None else "null"),
        "cluster_contact_threshold": int(args.cluster_contact_threshold),
        "model_name": model_name, "param_names": list(param_names),
        "model_params": [float(v) for v in model_params],
        "Tref": float(Tref), "Tscale": float(Tscale),
        "temperatures": [float(t) for t in Ts], "temperature_source": temp_source,
        "seed": int(args.seed),
        "local_sweeps_per_swap": int(args.local_sweeps_per_swap),
        "translation_sweeps_per_swap": int(args.translation_sweeps_per_swap),
        "reptation_sweeps_per_swap": int(args.reptation_sweeps_per_swap),
        "rotation_sweeps_per_swap": int(args.rotation_sweeps_per_swap),
        "n_cycles": int(args.n_cycles),
        "snapshot_stride": int(args.snapshot_stride),
        "fit_summary_json": str(fit_summary_json) if fit_summary_json else "null",
        "git_commit": remd._git_commit(),
        "start_time": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
        "end_time": "unknown",
    }
    if str(spec["potential_kind"]) == "saturating_cooperative":
        _A0, _q_sat = remd.validated_saturating_params(model_params)
        meta["A0"] = float(_A0)
        meta["q_sat"] = float(_q_sat)
    return meta


def _initial_states_metadata(replicas, rg_scale: float = 1.0) -> list:
    """Per-lane independent-initialization provenance (seed + initial observables).

    ``init_mean_chain_rg`` is a length: the lattice value is stored explicitly
    and the scaled output value is ``rg_scale * init_mean_chain_rg_lattice``.
    """
    return [{
        "temperature_index": int(k),
        "temperature": float(rep.T),
        "init_seed": int(rep.init_seed),
        "init_m_intra": int(rep.init_m_intra),
        "init_m_inter": int(rep.init_m_inter),
        "init_mean_chain_rg_lattice": float(rep.init_mean_chain_rg_lattice),
        "init_mean_chain_rg": float(rg_scale) * float(rep.init_mean_chain_rg_lattice),
        "init_largest_cluster_size": int(rep.init_largest_cluster_size),
    } for k, rep in enumerate(replicas)]


def _run_summary(args, Ts, temp_source, model_name, param_names, model_params,
                 Tref, Tscale, parameter_source, fit_summary_json, phi,
                 swap_props, swap_accs, results, wall, out_files, replicas,
                 diagnostics, snapshots_saved, kappa_bend=0.0,
                 bending_enabled=False, fit_chain_length=None) -> dict:
    swap_rates = [float(swap_accs[k] / swap_props[k]) if swap_props[k] else float("nan")
                  for k in range(len(swap_props))]
    M = int(args.n_chains)
    N = int(args.N)
    bend_fraction = _bend_fraction_by_temp(replicas, M, N, float(args.burnin_frac))
    spec = remd.MODEL_REGISTRY[model_name]
    linear_coeff = [remd.reduced_bias(model_name, model_params, float(T), Tref, Tscale)
                    for T in Ts]
    quadratic_coeff = [remd.quadratic_bias(model_name, model_params, float(T), Tref, Tscale)
                       for T in Ts]
    summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
        "schema_version": int(SCHEMA_VERSION),
        "multichain_output_schema_version": int(mcio.MULTICHAIN_OUTPUT_SCHEMA_VERSION),
        "snapshot_schema_version": int(mcio.MULTICHAIN_SNAPSHOT_SCHEMA_VERSION),
        "model_api_version": int(MODEL_API_VERSION),
        "model": model_name, "param_names": list(param_names),
        "params": [float(v) for v in model_params],
        "Tref": float(Tref), "Tscale": float(Tscale),
        "parameter_source": parameter_source, "fit_summary_json": fit_summary_json,
        "M": int(args.n_chains), "N": int(args.N), "L": int(args.box_size),
        "n_beads": int(args.n_chains * args.N), "volume_fraction": float(phi),
        "lambda_intra": float(args.lambda_intra),
        "lambda_inter": float(args.lambda_inter),
        # Optional fixed reduced bending penalty (additive; kappa_bend == 0
        # reproduces the contacts-only run exactly).  bend_fraction is the
        # post-burn-in mean n_bend_total/(M*(N-2)) per temperature lane.
        "kappa_bend": float(kappa_bend),
        "bending_enabled": bool(bending_enabled),
        "bend_definition": BEND_DEFINITION,
        "bend_fraction": [float(v) for v in bend_fraction],
        # Contact-potential contract.  The runtime N and fit chain length remain
        # chain-level provenance; for a label-blind nonlinear model the exact
        # M*N normalization is stated in multichain_potential_definition.  The
        # potential is no longer one b(T) curve once the model is nonlinear in m,
        # so the linear coefficient b(T) and quadratic coefficient kappa(T) (0 for
        # linear models and for the saturating-cooperative model, which has no
        # m^2/(2N) term) are listed separately; potential_definition states the
        # single-chain u_contact and multichain_potential_definition the assembled
        # multi-chain potential.
        "potential_kind": str(spec["potential_kind"]),
        "quadratic_contact_scope": _contact_scope(model_name),
        "nonlinear_contact_scope": _contact_scope(model_name),
        "interchain_contact_model": _interchain_contact_model(model_name),
        "quadratic_normalization": (
            "m_chain^2/(2*N)"
            if spec["potential_kind"] == "contact_quadratic" else None),
        "potential_definition": str(spec["potential_definition"]),
        "potential_normalization": spec["potential_normalization"],
        "m_ref": int(spec["m_ref"]),
        "multichain_potential_definition":
            _multichain_potential_definition(model_name),
        "runtime_chain_length": int(N),
        "fit_chain_length": (int(fit_chain_length)
                             if fit_chain_length is not None else None),
        "linear_coefficient_by_temperature": [float(v) for v in linear_coeff],
        "quadratic_coefficient_by_temperature": [float(v) for v in quadratic_coeff],
        "cluster_contact_threshold": int(args.cluster_contact_threshold),
        "temperatures": [float(t) for t in Ts], "temperature_count": int(len(Ts)),
        "temperature_source": temp_source,
        "local_sweeps_per_swap": int(args.local_sweeps_per_swap),
        "translation_sweeps_per_swap": int(args.translation_sweeps_per_swap),
        "reptation_sweeps_per_swap": int(args.reptation_sweeps_per_swap),
        "rotation_sweeps_per_swap": int(args.rotation_sweeps_per_swap),
        "local_proposals_per_cycle": int(args.local_sweeps_per_swap) * args.n_chains * args.N,
        "translation_proposals_per_cycle": int(args.translation_sweeps_per_swap) * args.n_chains,
        "reptation_proposals_per_cycle": int(args.reptation_sweeps_per_swap) * args.n_chains,
        "rotation_proposals_per_cycle": int(args.rotation_sweeps_per_swap) * args.n_chains,
        "n_cycles": int(args.n_cycles), "seed": int(args.seed),
        "n_workers": int(args.n_workers), "burnin_frac": float(args.burnin_frac),
        "rg_scale": float(args.rg_scale),
        # Independent-initialization provenance (Change 2).
        "initialization_seed_stride": int(INITIALIZATION_SEED_STRIDE),
        "independent_initialization": True,
        "initialization_seeds": [int(rep.init_seed) for rep in replicas],
        "initial_states": _initial_states_metadata(replicas, float(args.rg_scale)),
        # Representation provenance (Change 1 / Change 9).
        "coordinate_canonicalization_enabled": True,
        "snapshot_coordinate_dtype": "int32",
        "snapshots_saved": bool(snapshots_saved),
        # effective_energy is a TEMPERATURE-DEPENDENT effective energy, T * u,
        # NOT a temperature-independent microscopic energy.
        "effective_energy_definition": "effective_energy = T * u (temperature-dependent)",
        "swap_rates": swap_rates,
        "local_acceptance_rates": [r["local_acceptance_rate"] for r in results],
        "translation_acceptance_rates": [r["translation_acceptance_rate"] for r in results],
        "reptation_acceptance_rates": [r["reptation_acceptance_rate"] for r in results],
        "rotation_acceptance_rates": [r["rotation_acceptance_rate"] for r in results],
        "diagnostics_enabled": bool(diagnostics is not None),
        "wall_time_seconds": float(wall),
        "git_commit": remd._git_commit(),
        "output_files": out_files,
    }
    if str(spec["potential_kind"]) == "saturating_cooperative":
        # Named saturating parameters alongside the packed params list.
        _A0, _q_sat = remd.validated_saturating_params(model_params)
        summary["A0"] = float(_A0)
        summary["q_sat"] = float(_q_sat)
    if diagnostics is not None:
        summary["diagnostic_thresholds"] = diagnostics["thresholds"]
        summary["diagnostic_summary"] = diagnostics["summary"]
        summary["diagnostic_warnings"] = diagnostics["warnings"]
        summary["diagnostic_trajectories_saved"] = bool(
            args.diagnostic_trajectories)
    return summary


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

def _qt_unit_invariants() -> None:
    state = mcs.initialize_dispersed_state(3, 8, 12, seed=5)
    mcs.validate_state(state)
    # Athermal reduced potential is identically zero.
    u = reduced_potential_counts(state.counts, 300.0, "hs", [300.0, 1.0],
                                 300.0, 80.0, 0.0, 0.0)
    assert u == 0.0
    print("  quick-test unit invariants: PASSED")


def _qt_m1_regression() -> None:
    rng = np.random.RandomState(99)
    for _ in range(10):
        saw = mcs.generate_saw(18, rng)
        state = mcs.make_state(np.stack([saw]), 60)
        c = mcc.full_contact_counts(state)
        chain = [tuple(int(v) for v in row) for row in saw]
        m_ref = int(round(remd.contact_count(chain, set(chain))))
        assert c.inter == 0 and c.intra == m_ref
        # Reduced potential matches the single-chain u = m*b(T) at lambda_intra=1.
        for T in (300.0, 340.0):
            u_mc = reduced_potential_counts(c, T, "hs", [400.0, 1.3], 320.0, 80.0,
                                            1.0, 1.0)
            u_ref = remd.reduced_potential(m_ref, T, "hs", [400.0, 1.3], 320.0, 80.0)
            assert abs(u_mc - u_ref) < 1e-9
    print("  quick-test M=1 regression vs single-chain: PASSED")


def _qt_lambda_modes() -> None:
    # Build a state with both intra and inter contacts.
    hp = np.array([(0, 0, 0), (1, 0, 0), (2, 0, 0), (2, 1, 0), (1, 1, 0), (0, 1, 0)],
                  dtype=np.int64)
    state = mcs.make_state(np.stack([hp, hp + np.array([0, 0, 1])]), 20)
    c = state.counts
    args_common = ("hs", [400.0, 1.3], 320.0, 80.0)
    b = remd.reduced_bias(*(("hs", [400.0, 1.3], 340.0, 320.0, 80.0)))
    # (0,0): athermal -> u == 0 for any counts.
    assert reduced_potential_counts(c, 340.0, *args_common, 0.0, 0.0) == 0.0
    # Unequal scales are rejected because they distinguish contact classes.
    for li, lin in ((1.0, 0.0), (0.0, 1.0), (0.5, 0.25)):
        try:
            reduced_potential_counts(c, 340.0, *args_common, li, lin)
        except ValueError:
            pass
        else:  # pragma: no cover - the guard is the point of this check
            raise AssertionError("hs accepted unequal contact scales")
    # (1,1): all contacts use the same Hamiltonian.
    u11 = reduced_potential_counts(c, 340.0, *args_common, 1.0, 1.0)
    assert abs(u11 - b * (c.intra + c.inter)) < 1e-9
    print("  quick-test label-blind contact scale: PASSED")


def _qt_generalized_swap() -> None:
    # Manual generalized swap with UNEQUAL intra/inter counts.
    ci = ContactCounts(3, 5)
    cj = ContactCounts(7, 2)
    Ti, Tj = 300.0, 350.0
    mp = ("poly2", [0.1, -0.4, 0.2], 320.0, 80.0)
    li, lin = 1.0, 0.6

    def u(c, T):
        return reduced_potential_counts(c, T, *mp, li, lin)
    expect = u(ci, Ti) + u(cj, Tj) - u(cj, Ti) - u(ci, Tj)
    got = swap_log_accept_counts(ci, cj, Ti, Tj, *mp, li, lin)
    assert abs(got - expect) < 1e-12
    print("  quick-test generalized swap (unequal intra/inter): PASSED")


def _qt_bending() -> None:
    # (1) Full reduced potential adds kappa_bend * n_bend; kappa=0 collapses onto
    # the contacts-only reduced potential exactly.
    c = ContactCounts(4, 2)
    mp = ("hs", [400.0, 1.3], 320.0, 80.0)
    u_full = reduced_potential_bending_counts(c, 330.0, *mp, 1.0, 1.0,
                                              kappa_bend=0.5, n_bend=6)
    u_contacts = reduced_potential_counts(c, 330.0, *mp, 1.0, 1.0)
    assert abs(u_full - (u_contacts + 0.5 * 6)) < 1e-12
    assert reduced_potential_bending_counts(
        c, 330.0, *mp, 1.0, 1.0, kappa_bend=0.0, n_bend=6) == u_contacts

    # (2) Metropolis helper includes the bending contribution.  Favorable bends
    # (delta_bends < 0) always accept without a draw; an unfavorable pure bend
    # change is probabilistic.
    assert metropolis_accept_delta(0, 0, 1.0, 1.0, 1.0, 0.99,
                                   kappa_bend=0.5, delta_bends=-2) is True
    assert metropolis_accept_delta(0, 0, 1.0, 1.0, 1.0, 0.90,
                                   kappa_bend=1.0, delta_bends=1) is False  # du=1
    assert metropolis_accept_delta(0, 0, 1.0, 1.0, 1.0, 0.10,
                                   kappa_bend=1.0, delta_bends=1) is True

    # (3) Total bend count equals the sum of per-chain counts, and periodic chains
    # crossing the box boundary are counted from contiguous unwrapped coordinates.
    from lattice_bending import count_bends
    state = mcs.initialize_dispersed_state(3, 8, 12, seed=5)
    total = mcs.total_bend_count(state.coords_unwrapped)
    per_chain = sum(count_bends(state.coords_unwrapped[k])
                    for k in range(state.n_chains))
    assert total == per_chain == state.n_bend

    # (4) The fixed bending term cancels exactly from the REMD swap for any kappa
    # and any (different) bend counts.
    ci, cj = ContactCounts(3, 5), ContactCounts(7, 2)
    base = swap_log_accept_counts(ci, cj, 300.0, 350.0, *mp, 1.0, 1.0)
    for kap in (0.0, 0.7, 2.0):
        for ni, nj in ((0, 5), (4, 4), (6, 1)):
            def u(cc, T, n):
                return reduced_potential_bending_counts(
                    cc, T, *mp, 1.0, 1.0, kappa_bend=kap, n_bend=n)
            la = (u(ci, 300.0, ni) + u(cj, 350.0, nj)
                  - u(cj, 300.0, nj) - u(ci, 350.0, ni))
            assert abs(la - base) < 1e-9

    # (5) A positive penalty straightens the chains: mean bend fraction drops.
    long_kw = dict(n_chains=3, chain_length=8, box_size=12,
                   Ts=np.linspace(320, 340, 3), local_sweeps_per_swap=3,
                   translation_sweeps_per_swap=1, n_cycles=160, model_name="hs",
                   params=[0.0, 0.0], Tref=330.0, Tscale=40.0, seed=321,
                   n_workers=1, verbose=False)
    reps0, *_ = run_remd_multichain(kappa_bend=0.0, **long_kw)
    repsK, *_ = run_remd_multichain(kappa_bend=1.5, **long_kw)
    bf0 = np.nanmean(_bend_fraction_by_temp(reps0, 3, 8, 0.5))
    bfK = np.nanmean(_bend_fraction_by_temp(repsK, 3, 8, 0.5))
    assert bfK < bf0, (
        f"positive kappa did not reduce bend fraction: {bfK:.3f} !< {bf0:.3f}")
    print(f"  quick-test bending (potential/Metropolis/total/swap-cancel/"
          f"straightens {bfK:.3f}<{bf0:.3f}): PASSED")


def _qt_saturating_cooperative() -> None:
    """Check the label-blind all-contact saturation Hamiltonian."""
    model = "saturating_cooperative_contact"
    p = [700.0, 2.4, 5.0, 0.15]
    p0 = [700.0, 2.4, 0.0, 0.15]
    Tref, Tscale = 320.0, 80.0
    T = 335.0

    def u_sat(m, n_beads, temperature=T):
        b = 700.0 / T - 2.4
        if temperature != T:
            b = 700.0 / float(temperature) - 2.4
        q = float(m) / float(n_beads)
        r = q / 0.15
        return float(n_beads) * (b * q - 5.0 * q * q / (1.0 + r * r))

    # The exact single-chain formula is evaluated at the total contact count and
    # total bead count.
    state = mcs.initialize_dispersed_state(3, 8, 12, seed=6)
    m_total = int(state.counts.intra) + int(state.counts.inter)
    total_beads = state.n_chains * state.chain_length
    got = reduced_contact_potential_state(
        state, T, model, p, Tref, Tscale, 1.0, 1.0)
    assert abs(got - u_sat(m_total, total_beads)) < 1e-9

    # Reclassifying contacts without changing their total cannot change u.
    state_a = state.copy()
    state_b = state.copy()
    state_a.counts = ContactCounts(2, 5)
    state_b.counts = ContactCounts(5, 2)
    ua = reduced_contact_potential_state(
        state_a, T, model, p, Tref, Tscale, 1.0, 1.0)
    ub = reduced_contact_potential_state(
        state_b, T, model, p, Tref, Tscale, 1.0, 1.0)
    assert ua == ub

    # Unequal contact scales are invalid; equal zero disables contacts only.
    for li, lin in ((1.0, 0.0), (0.0, 1.0), (0.5, 0.25)):
        try:
            reduced_contact_potential_state(
                state, T, model, p, Tref, Tscale, li, lin)
        except ValueError:
            pass
        else:  # pragma: no cover
            raise AssertionError("saturation model accepted unequal contact scales")
    assert reduced_contact_potential_state(
        state, T, model, p, Tref, Tscale, 0.0, 0.0) == 0.0
    u_bend = reduced_potential_state(state, T, model, p, Tref, Tscale, 0.0, 0.0,
                                     kappa_bend=0.5)
    assert abs(u_bend - 0.5 * state.n_bend) < 1e-12

    # Every proposal delta equals the exact change in the total-contact potential.
    st = mcs.initialize_dispersed_state(3, 8, 12, seed=23)
    rng = random.Random(11)
    kb = 0.4
    n_checked = 0
    for _ in range(600):
        prop = mvs.propose_local(st, rng)
        if not prop.ok:
            continue
        d_intra, d_inter = mvs.proposal_delta(st, prop)
        d_bends = mvs.proposal_delta_bends(st, prop)
        m_old = int(st.counts.intra) + int(st.counts.inter)
        d_total = int(d_intra) + int(d_inter)
        u_before = reduced_potential_state(st, T, model, p, Tref, Tscale,
                                           1.0, 1.0, kappa_bend=kb)
        du = (u_sat(m_old + d_total, st.n_chains * st.chain_length)
              - u_sat(m_old, st.n_chains * st.chain_length)
              + kb * int(d_bends))
        mvs.apply_proposal(st, prop, (d_intra, d_inter), delta_bends=d_bends)
        u_after = reduced_potential_state(st, T, model, p, Tref, Tscale,
                                          1.0, 1.0, kappa_bend=kb)
        assert abs((u_after - u_before) - du) < 1e-9
        n_checked += 1
    assert n_checked > 50

    # The generalized swap uses the same total-contact Hamiltonian.
    sa = mcs.initialize_dispersed_state(3, 8, 12, seed=21)
    sb = mcs.initialize_dispersed_state(3, 8, 12, seed=22)
    Ti, Tj = 305.0, 350.0

    def u_state(s, t):
        return reduced_contact_potential_state(s, t, model, p, Tref, Tscale,
                                               1.0, 1.0)
    manual = (u_state(sa, Ti) + u_state(sb, Tj)
              - u_state(sb, Ti) - u_state(sa, Tj))
    got = swap_log_accept_state(sa, sb, Ti, Tj, model, p, Tref, Tscale, 1.0, 1.0)
    assert abs(got - manual) < 1e-9

    # M = 1 reproduces the original single-chain Hamiltonian exactly.
    saw_rng = np.random.RandomState(4)
    for _ in range(4):
        one = mcs.make_state(np.stack([mcs.generate_saw(18, saw_rng)]), 60)
        assert int(one.counts.inter) == 0
        for kap in (0.0, 0.8):
            u_mc = reduced_potential_state(one, T, model, p, Tref, Tscale,
                                           1.0, 1.0, kappa_bend=kap)
            u_ref = remd.reduced_potential_bending(
                int(one.counts.intra), T, model, p, Tref, Tscale, kappa_bend=kap,
                n_bend=one.n_bend, n_beads=one.chain_length)
            assert abs(u_mc - u_ref) < 1e-9

    # A0 = 0 nests the label-blind hs model.
    u_lin = reduced_contact_potential_state(
        state, T, "hs", [700.0, 2.4], Tref, Tscale, 1.0, 1.0)
    u_a0 = reduced_contact_potential_state(
        state, T, model, p0, Tref, Tscale, 1.0, 1.0)
    assert abs(u_a0 - u_lin) < 1e-9

    # Bending disabled / enabled, and serial vs multiprocessing determinism.
    run_kw = dict(n_chains=2, chain_length=8, box_size=10,
                  Ts=np.linspace(305, 350, 4), local_sweeps_per_swap=1,
                  translation_sweeps_per_swap=1, n_cycles=10, model_name=model,
                  params=p, Tref=Tref, Tscale=Tscale, lambda_intra=1.0,
                  lambda_inter=1.0, seed=13, verbose=False, debug_contacts=True)
    for kap in (0.0, 0.5):
        reps1, sp1, sa1, wh1 = run_remd_multichain(kappa_bend=kap, n_workers=1,
                                                   **run_kw)
        reps2, sp2, sa2, wh2 = run_remd_multichain(kappa_bend=kap, n_workers=2,
                                                   **run_kw)
        assert np.array_equal(sp1, sp2) and np.array_equal(sa1, sa2)
        assert np.array_equal(wh1, wh2)
        for ra, rb in zip(reps1, reps2):
            assert np.array_equal(ra.state.coords_unwrapped,
                                  rb.state.coords_unwrapped)
            assert np.array_equal(ra.state.intra_contacts_by_chain,
                                  rb.state.intra_contacts_by_chain)
            assert ra.state.n_bend == rb.state.n_bend
            assert ra.u_traj == rb.u_traj
            mcs.validate_state(ra.state)
        assert all(math.isfinite(x) for r in reps1 for x in r.u_traj)

    # Aggregate ContactCounts helpers lack M*N and must refuse this model.
    for fn, kw in ((reduced_potential_counts, {}),
                   (reduced_potential_bending_counts,
                    dict(kappa_bend=0.3, n_bend=5))):
        try:
            fn(state.counts, T, model, p, Tref, Tscale, 1.0, 1.0, **kw)
        except NotImplementedError as exc:
            assert "nonlinear" in str(exc)
        else:  # pragma: no cover - the guard is the point of the test
            raise AssertionError(f"{fn.__name__} accepted a nonlinear model")

    print("  quick-test saturating-cooperative all-contact Hamiltonian "
          "(formula/label-blind/move-delta/swap/M=1/A0=0/determinism): PASSED")


def _qt_short_run(n_workers, tmp, tag, lambda_intra=1.0, lambda_inter=1.0,
                  debug=True, kappa_bend=0.0):
    import os as _os
    Ts = np.linspace(305, 350, 4)
    reps, sp, sa, wh = run_remd_multichain(
        n_chains=2, chain_length=8, box_size=10, Ts=Ts,
        local_sweeps_per_swap=1, translation_sweeps_per_swap=1, n_cycles=12,
        model_name="hs", params=[400.0, 1.3], Tref=320.0, Tscale=80.0,
        lambda_intra=lambda_intra, lambda_inter=lambda_inter,
        kappa_bend=kappa_bend,
        seed=13, n_workers=n_workers, verbose=False, debug_contacts=debug)
    # Walker identities always a permutation.
    assert wh.shape == (12, len(Ts))
    for row in wh:
        assert sorted(row.tolist()) == list(range(len(Ts)))
    # States remain valid; cached counts match the oracle.
    for rep in reps:
        mcs.validate_state(rep.state)
    return reps, sp, sa, wh, Ts


def _qt_serial_vs_workers(tmp) -> None:
    # Determinism must hold both with bending off and with a nonzero penalty.
    for kappa in (0.0, 0.6):
        r1 = _qt_short_run(1, tmp, "serial", kappa_bend=kappa)
        r2 = _qt_short_run(2, tmp, "workers", kappa_bend=kappa)
        # Deterministic per-lane seeding => identical results across worker counts.
        for a, b in zip(r1[0], r2[0]):
            assert a.state.counts.as_tuple() == b.state.counts.as_tuple(), (
                "serial and 2-worker runs diverged despite identical seeds")
            assert np.array_equal(a.state.coords_unwrapped, b.state.coords_unwrapped)
            assert a.state.n_bend == b.state.n_bend
            assert a.n_bend_traj == b.n_bend_traj
    print("  quick-test serial vs 2-worker determinism (kappa 0 and 0.6): PASSED")


def _qt_output_roundtrip(tmp) -> None:
    import os as _os
    reps, sp, sa, wh, Ts = _qt_short_run(1, tmp, "io")
    prefix = _os.path.join(tmp, "qt_mc")
    results = summarize_results(reps, Ts, burnin_frac=0.4, rg_scale=1.0)
    for rep, r in zip(reps, results):
        r["local_acceptance_rate"] = rep.local_acceptance_rate()
        r["translation_acceptance_rate"] = rep.translation_acceptance_rate()
    dist = build_run_distributions(reps, Ts, burnin_frac=0.4, rg_bins=20,
                                   rg_scale=1.0)
    attach_metadata(
        dist, M=2, N=8, L=10, Ts=Ts, seed=13, model_name="hs",
        param_names=["h", "s"], model_params=[400.0, 1.3], Tref=320.0,
        Tscale=80.0, lambda_intra=1.0, lambda_inter=1.0,
        local_sweeps_per_swap=1, translation_sweeps_per_swap=1, n_cycles=12,
        burnin_frac=0.4, cluster_contact_threshold=1,
        parameter_source="params_cli", fit_summary_json=None)
    mcio.save_results_csv(results, prefix)
    mcio.save_swap_csv(sp, sa, Ts, prefix)
    mcio.save_move_acceptance_csv([r.move_counters for r in reps], mvs.MOVE_TYPES,
                                  Ts, prefix)
    npz_path = mcio.save_distributions_npz(dist, prefix)
    mcio.save_run_summary({"schema_version": SCHEMA_VERSION, "M": 2}, f"{prefix}_run_summary.json")

    with np.load(npz_path, allow_pickle=True) as data:
        for key in ("Ts", "m_intra_vals", "P_m_intra", "m_inter_vals",
                    "P_m_inter", "rg_edges", "rg_centers", "P_mean_chain_rg",
                    "fmax_edges", "fmax_centers", "P_fmax", "M", "N", "L",
                    "lambda_intra", "lambda_inter", "model_params",
                    "volume_fraction", "seed", "temperatures", "schema_version"):
            assert key in data, f"distributions NPZ missing metadata key {key}"
        assert int(data["M"]) == 2 and int(data["N"]) == 8 and int(data["L"]) == 10
        assert float(data["lambda_intra"]) == 1.0
        for name in ("P_m_intra", "P_m_inter", "P_mean_chain_rg", "P_fmax"):
            for row in data[name]:
                finite = row[np.isfinite(row)]
                if finite.size:
                    assert abs(float(finite.sum()) - 1.0) < 1e-6, f"{name} not normalized"

    # HDF5 snapshot round-trip (explicit chain identity).
    if mcio.h5py_available():
        import h5py
        h5_path = _os.path.join(tmp, "qt_snap.h5")
        writer = mcio.MultiChainSnapshotWriter(
            h5_path, n_chains=2, chain_length=8, n_temperatures=len(Ts),
            metadata={"M": 2, "N": 8, "L": 10, "lambda_intra": 1.0,
                      "lambda_inter": 1.0, "model_params": [400.0, 1.3],
                      "temperatures": [float(t) for t in Ts]})
        for cyc in range(3):
            coords = np.stack([r.state.coords_unwrapped for r in reps])
            writer.append(
                cycle=cyc, coordinates=coords,
                walker_id=np.arange(len(Ts)),
                m_intra=np.array([r.state.counts.intra for r in reps]),
                m_inter=np.array([r.state.counts.inter for r in reps]),
                mean_chain_rg2=np.array(
                    [obs.chain_geometry_summary(r.state)["mean_chain_rg2_lattice"]
                     for r in reps]),
                largest_cluster_size=np.array([1 for _ in reps]))
        writer.mark_complete()
        writer.close()
        with h5py.File(h5_path, "r") as f:
            assert f["snapshots"].attrs["status"] == "complete"
            assert f["snapshots"].attrs["coordinate_dims"] == \
                "snapshot,temperature_lane,chain,monomer,xyz"
            coords = f["snapshots"]["coordinates"]
            assert coords.shape == (3, len(Ts), 2, 8, 3)
            assert int(f["metadata"].attrs["M"]) == 2
        print("  quick-test output + HDF5 snapshot round-trip: PASSED")
    else:
        print("  quick-test output round-trip (HDF5 skipped): PASSED")


def _qt_strong_attraction_smoke() -> None:
    # A strong interchain attraction should increase m_inter and the largest
    # cluster fraction relative to the athermal control (broad statistical check).
    # Contacts are FAVORED when the reduced bias b(T) < 0 (coupling K = -b > 0);
    # hs params (h=0, s=4) give a constant strongly-attractive b(T) = -4.
    Ts = np.linspace(320, 340, 3)
    common = dict(n_chains=4, chain_length=6, box_size=8, Ts=Ts,
                  local_sweeps_per_swap=2, translation_sweeps_per_swap=3,
                  n_cycles=120, model_name="hs", params=[0.0, 4.0], Tref=330.0,
                  Tscale=40.0, seed=7, n_workers=1, verbose=False)
    reps_ath, *_ = run_remd_multichain(lambda_intra=0.0, lambda_inter=0.0, **common)
    reps_att, *_ = run_remd_multichain(lambda_intra=1.0, lambda_inter=1.0, **common)

    def mean_tail(reps, attr):
        vals = []
        for rep in reps:
            arr = np.asarray(getattr(rep, attr), dtype=float)
            vals.append(arr[len(arr) // 2:].mean())
        return float(np.mean(vals))
    m_inter_ath = mean_tail(reps_ath, "m_inter_traj")
    m_inter_att = mean_tail(reps_att, "m_inter_traj")
    fmax_ath = mean_tail(reps_ath, "lcf_traj")
    fmax_att = mean_tail(reps_att, "lcf_traj")
    assert m_inter_att >= m_inter_ath, (
        f"strong attraction did not raise m_inter ({m_inter_att} < {m_inter_ath})")
    assert fmax_att >= fmax_ath, (
        f"strong attraction did not raise largest-cluster fraction "
        f"({fmax_att} < {fmax_ath})")
    print("  quick-test strong-attraction smoke (m_inter/fmax up): PASSED")


def _qt_independent_init() -> None:
    Ts = np.linspace(305, 350, 4)
    reps = _build_independent_replicas(Ts, 3, 6, 12, base_seed=1234,
                                       cluster_contact_threshold=1)
    assert not np.array_equal(reps[0].state.coords_unwrapped,
                              reps[1].state.coords_unwrapped), \
        "independent lanes must not share one starting configuration"
    for lane, rep in enumerate(reps):
        assert rep.init_seed == 1234 + INITIALIZATION_SEED_STRIDE * lane
    print("  quick-test independent per-lane initialization: PASSED")


def _qt_canonicalization() -> None:
    state = mcs.initialize_dispersed_state(2, 6, 10, seed=7)
    counts0 = state.counts.as_tuple()
    rg0 = mcs.per_chain_rg2(state).copy()
    wrapped0 = state.wrapped().copy()
    L = state.box_size
    state.coords_unwrapped[1] += np.array([2 * L, -3 * L, L], dtype=np.int64)
    mcs.canonicalize_chain_coordinates(state, 1)
    b0 = state.coords_unwrapped[1, 0]
    assert np.all((b0 >= 0) & (b0 < L))
    assert state.counts.as_tuple() == counts0
    assert np.array_equal(state.wrapped(), wrapped0)
    np.testing.assert_allclose(mcs.per_chain_rg2(state), rg0)
    mcs.validate_state(state)
    print("  quick-test coordinate canonicalization invariance: PASSED")


class _ScriptRandom:
    """Minimal deterministic RNG stub for exact hand-constructed move tests.

    ``randrange`` and ``random`` pop scripted values from independent queues so a
    proposer's draws are fully controlled (and independent of the CPython PRNG
    implementation).  Only the methods the proposers call are provided.
    """

    def __init__(self, ints, floats=()):
        self._ints = list(ints)
        self._floats = list(floats)

    def randrange(self, n):  # noqa: ARG002 - n is intentionally ignored
        return self._ints.pop(0)

    def random(self):
        return self._floats.pop(0)


def _qt_reptation_regression() -> None:
    # (a) Exact hand-constructed coordinates, head removal, v = (0, 1, 0).
    chain = np.array([(2, 2, 2), (3, 2, 2), (4, 2, 2), (5, 2, 2)], dtype=np.int64)
    v_head = mvs.NN6.index((0, 1, 0))
    state = mcs.make_state(np.stack([chain]), 12)
    prop = mvs.propose_reptation(state, _ScriptRandom([0, v_head], [0.1]))
    assert prop.ok and prop.state_changing
    assert prop.moved == {0: (3, 2, 2), 1: (4, 2, 2), 2: (5, 2, 2), 3: (5, 3, 2)}
    # delta_contacts agrees with a full recount before/after applying the move.
    d = mvs.proposal_delta(state, prop)
    mvs.apply_proposal(state, prop, d)
    mcc.assert_counts_match(state, "reptation head")
    mcs.validate_state(state)

    # (b) Exact tail removal, v = (0, 0, 1), on a fresh chain.
    state2 = mcs.make_state(np.stack([chain]), 12)
    v_tail = mvs.NN6.index((0, 0, 1))
    prop2 = mvs.propose_reptation(state2, _ScriptRandom([0, v_tail], [0.9]))
    assert prop2.ok and prop2.moved == {
        0: (2, 2, 3), 1: (2, 2, 2), 2: (3, 2, 2), 3: (4, 2, 2)}
    d2 = mvs.proposal_delta(state2, prop2)
    mvs.apply_proposal(state2, prop2, d2)
    mcc.assert_counts_match(state2, "reptation tail")
    mcs.validate_state(state2)

    # (c) Many reptation-only athermal moves keep a multi-chain state valid and
    # the cached counts consistent (delta vs full recount, via debug_contacts).
    ms = mcs.initialize_dispersed_state(3, 8, 12, seed=21)
    counters = mvs.new_move_counters()
    mc_sweep(ms, counters, 330.0, "hs", [400.0, 1.3], 320.0, 80.0,
             lambda_intra=0.0, lambda_inter=0.0,
             n_local=0, n_translation=0, rng=random.Random(3),
             n_reptation=600, n_rotation=0, debug_contacts=True)
    mcs.validate_state(ms)
    idx = mvs.MOVE_INDEX["reptation"]
    assert counters[idx, 0] == 600 and counters[idx, 3] > 0
    print("  quick-test reptation regression (exact coords + delta + validity): PASSED")


def _qt_rotation_regression() -> None:
    # (a) Exact hand-constructed coordinates: pivot bead 0, 90-degree rotation
    # about z (a proper cubic rotation guaranteed to be in the 24-element pool).
    Rz90 = ((0, -1, 0), (1, 0, 0), (0, 0, 1))
    idx_rot = remd.ROT_MATS.index(Rz90)
    chain = np.array([(2, 2, 2), (3, 2, 2), (4, 2, 2)], dtype=np.int64)
    state = mcs.make_state(np.stack([chain]), 12)
    prop = mvs.propose_chain_rotation(state, _ScriptRandom([0, 0, idx_rot]))
    assert prop.ok and prop.state_changing
    # Bead 0 (pivot) excluded; beads 1,2 rotate about (2,2,2).
    assert prop.moved == {1: (2, 3, 2), 2: (2, 4, 2)}
    d = mvs.proposal_delta(state, prop)
    mvs.apply_proposal(state, prop, d)
    mcc.assert_counts_match(state, "rotation")
    mcs.validate_state(state)

    # (b) Identity rotation (the 24th pool element) moves nothing.
    state_id = mcs.make_state(np.stack([chain]), 12)
    prop_id = mvs.propose_chain_rotation(
        state_id, _ScriptRandom([0, 1, len(remd.ROT_MATS)]))
    assert prop_id.ok and not prop_id.state_changing

    # (c) Rigid rotation preserves each chain's Rg^2 EXACTLY: many rotation-only
    # athermal sweeps leave per-chain Rg^2 unchanged, and the state stays valid.
    ms = mcs.initialize_dispersed_state(3, 8, 14, seed=31)
    rg2_before = mcs.per_chain_rg2(ms).copy()
    counters = mvs.new_move_counters()
    mc_sweep(ms, counters, 330.0, "hs", [400.0, 1.3], 320.0, 80.0,
             lambda_intra=0.0, lambda_inter=0.0,
             n_local=0, n_translation=0, rng=random.Random(4),
             n_reptation=0, n_rotation=600, debug_contacts=True)
    mcs.validate_state(ms)
    np.testing.assert_allclose(mcs.per_chain_rg2(ms), rg2_before)
    idx = mvs.MOVE_INDEX["rotation"]
    assert counters[idx, 0] == 600
    print("  quick-test rotation regression (exact coords + Rg^2 invariance): PASSED")


def _qt_athermal_move_baseline() -> None:
    # Detailed-balance smoke: reptation-only sampling at an athermal potential
    # must reproduce the self-avoiding-walk Rg^2 baseline within statistical
    # error (reptation is symmetric and preserves the uniform SAW measure).
    N, L = 8, 30
    rng_np = np.random.RandomState(123)
    saw_rg2 = np.array(
        [mcs.chain_radius_of_gyration_squared(
            mcs.make_state(np.stack([mcs.generate_saw(N, rng_np)]), L), 0)
         for _ in range(400)])
    baseline = float(saw_rg2.mean())

    state = mcs.make_state(np.stack([mcs.generate_saw(N, np.random.RandomState(7))]), L)
    counters = mvs.new_move_counters()
    samples = []
    for cyc in range(120):
        mc_sweep(state, counters, 330.0, "hs", [400.0, 1.3], 320.0, 80.0,
                 lambda_intra=0.0, lambda_inter=0.0,
                 n_local=0, n_translation=0, rng=random.Random(1000 + cyc),
                 n_reptation=40, n_rotation=0)
        if cyc >= 40:  # discard burn-in
            samples.append(mcs.chain_radius_of_gyration_squared(state, 0))
    sampled = float(np.mean(samples))
    rel = abs(sampled - baseline) / baseline
    assert rel < 0.30, (
        f"reptation athermal Rg^2 {sampled:.3f} deviates {rel:.2%} from SAW "
        f"baseline {baseline:.3f} (>30%): detailed balance / ergodicity suspect")
    print(f"  quick-test athermal reptation matches SAW Rg^2 baseline "
          f"(rel={rel:.2%}): PASSED")


def _qt_dc_over_L() -> None:
    # (a) Hand-built COMs, no boundary crossing: two 2-bead chains whose COMs are
    # (0, 0.5, 0) and (3, 0.5, 0) in a box of L = 10 -> Dc = 3, Dc/L = 0.3.
    L = 10
    chain_a = np.array([(0, 0, 0), (0, 1, 0)], dtype=np.int64)
    chain_near = np.array([(3, 0, 0), (3, 1, 0)], dtype=np.int64)
    dc = obs.mean_pair_com_distance_over_L(
        mcs.make_state(np.stack([chain_a, chain_near]), L))
    assert abs(dc - 0.3) < 1e-12

    # (b) Minimum-image check: the second chain sits at the FAR boundary, COM
    # (9, 0.5, 0).  The raw separation is 9, but the nearest periodic image is 1.
    # A missing min-image term would report 0.9 here.
    chain_far = np.array([(9, 0, 0), (9, 1, 0)], dtype=np.int64)
    dc_far = obs.mean_pair_com_distance_over_L(
        mcs.make_state(np.stack([chain_a, chain_far]), L))
    assert abs(dc_far - 0.1) < 1e-12, (
        f"Dc/L={dc_far} != 0.1: minimum-image convention not applied to the "
        f"COM separation")

    # (c) The same physical configuration with the far chain shifted one box in
    # -x: its unwrapped COM is now (-1, 0.5, 0), OUTSIDE the primary box.  Dc is
    # unchanged, confirming COMs are never wrapped and only the separation is
    # min-imaged.
    dc_out = obs.mean_pair_com_distance_over_L(
        mcs.make_state(np.stack([chain_a, chain_far - np.array([L, 0, 0])]), L))
    assert abs(dc_out - 0.1) < 1e-12

    # (d) M = 1: no interchain pairs -> NaN.
    solo = mcs.make_state(np.stack([mcs.generate_saw(8, np.random.RandomState(3))]), 20)
    assert math.isnan(obs.mean_pair_com_distance_over_L(solo))

    # (e) cycle_observables agrees with a direct recomputation, and the lattice
    # form is exactly L * (Dc/L).
    state = mcs.initialize_dispersed_state(4, 6, 12, seed=17)
    cyc = obs.cycle_observables(state, 1)
    direct = obs.mean_pair_com_distance_over_L(state)
    assert abs(cyc["dc_over_L"] - direct) < 1e-12, "dc_over_L mismatch"
    assert abs(cyc["dc_lattice"] - direct * 12.0) < 1e-12, "dc_lattice mismatch"

    # (f) Smoke: for dispersed athermal states Dc/L is O(1) and roughly box-size
    # insensitive (Dc itself scales with L, which is why the paper reports Dc/L).
    # The min-image distance is bounded by L*sqrt(3)/2, so Dc/L < 0.867 always.
    def mean_dc_over_L(box):
        return float(np.mean([
            obs.mean_pair_com_distance_over_L(
                mcs.initialize_dispersed_state(6, 6, box, seed=100 + s))
            for s in range(12)]))
    d12, d16 = mean_dc_over_L(12), mean_dc_over_L(16)
    for box, d in ((12, d12), (16, d16)):
        assert 0.25 < d < 0.75, f"dispersed Dc/L={d:.3f} at L={box} is not O(1)"
    assert abs(d12 - d16) < 0.15, (
        f"dispersed Dc/L varies strongly with box size ({d12:.3f} vs {d16:.3f})")
    print(f"  quick-test Dc/L (min-image, M=1 NaN, "
          f"dispersed Dc/L~{0.5 * (d12 + d16):.2f}): PASSED")


def _qt_diagnostics(tmp) -> None:
    import os as _os
    reps, sp, sa, wh, Ts = _qt_short_run(1, tmp, "diag")
    diag = mcd.compute_multichain_diagnostics(
        reps, sp, sa, wh, Ts, burnin_frac=0.5, rg_scale=1.0)
    prefix = _os.path.join(tmp, "qt_diag")
    files = mcd.save_all_diagnostics(
        diag, reps, wh, Ts, out_prefix=prefix, burnin_frac=0.5, rg_scale=1.0,
        save_trajectories=True)
    for key in ("diagnostics_json", "convergence_csv", "round_trips_csv",
                "walker_occupancy_csv", "diagnostic_trajectories_npz"):
        assert _os.path.exists(files[key]), f"diagnostics file {key} missing"
    with np.load(files["diagnostic_trajectories_npz"]) as data:
        for key in ("u_post", "m_intra_post", "m_inter_post",
                    "per_chain_rg_post", "walker_temp_index_post"):
            assert key in data and data[key].dtype != object
    print("  quick-test diagnostics files + trajectories: PASSED")


def run_quick_test() -> None:
    import tempfile
    print("multichain quick-test:")
    _qt_unit_invariants()
    _qt_m1_regression()
    _qt_lambda_modes()
    _qt_generalized_swap()
    _qt_saturating_cooperative()
    _qt_bending()
    _qt_independent_init()
    _qt_canonicalization()
    _qt_reptation_regression()
    _qt_rotation_regression()
    _qt_dc_over_L()
    with tempfile.TemporaryDirectory() as tmp:
        _qt_serial_vs_workers(tmp)
        _qt_output_roundtrip(tmp)
        _qt_diagnostics(tmp)
    _qt_athermal_move_baseline()
    _qt_strong_attraction_smoke()
    print("quick-test complete.")


if __name__ == "__main__":
    main()
