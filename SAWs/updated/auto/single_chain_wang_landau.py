#!/usr/bin/env python3
"""Wang-Landau baseline sampler for one self-avoiding lattice chain.

The physical target remains the uniform (athermal) measure over self-avoiding
conformations.  Wang-Landau adaptation is used only to learn a contact-number
bias.  The adaptive trajectory is discarded.  Independent production chains
then run with the frozen weight ``W(m) = exp(-log_g_hat[m])`` and every recorded
sample is reweighted by ``1/W(m) = exp(log_g_hat[m])``.

This separation matters: even an imperfect ``log_g_hat`` gives a consistent
athermal estimator when the frozen-weight production simulation is equilibrated
and covers the requested contact window.  A better estimate improves mixing and
statistical efficiency; self-normalized finite-sample estimates are not claimed
to be exactly unbiased.

Contact levels are split into a flat tier, a coverage-only tier, and an excluded
tier.  The flatness effort is concentrated where the molecular target has mass,
while every included level still has a minimum-coverage requirement.  Relaxing
flatness changes efficiency, not the limiting reweighted target, provided the
fixed-weight chain equilibrates and covers the declared window.

The optional Belardinelli-Pereyra schedule uses cumulative Monte Carlo time
``t = attempted_moves / included_levels``.  The time origin is never reset at a
refinement stage.  This is essential to the asymptotic ``1/t`` schedule.

Four move families are used: pivot rotations, corner flips, end moves, and
Lesh-Mitzenmacher-Whitesides pull moves.  Only the pull moves can relocate a
bead whose neighbours are all occupied, which is what makes the compact end of
the contact window re-reachable after the bias has been built up.  They are not
a symmetric proposal, and the move set is not closed under inversion, so they
carry an explicit Hastings ratio and irreversible proposals are rejected; see
``attempt_pull_move`` for the derivation and for why it is not optional.

Project probes measured about 50.8k, 38.1k, and 29.9k attempted moves per
second per learning core for N=30, 44, and 60.  Those figures predate pull
moves, which cost an extra catalog build per proposal and are substantially
slower per attempted move at the default ``--pull_move_weight``; the achieved
rate is printed in the progress line and stored as
``wl_learning_steps_per_second``.  Learning is single-process while production
is parallel, so the default final modification factor is 1e-4 and the learning
caps are deliberately generous.  This changes the quality of the bias estimate,
not the limiting target of the frozen-weight reweighting step.

The output preserves the distribution fields used by
``single_uniform_chain2_athermal_dists_joint.py``:

    c_vals, c_prob, c_edges, rg_edges, rg_prob, crg_prob

and its principal run metadata.  Additive ``wl_*`` fields record the learned
bias, flatness stages, range coverage, round trips, and importance-sampling
diagnostics.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
import tempfile
import time
import warnings
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import permutations, product
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from baseline_grids import (
    LEGACY_BASELINE_FILES,
    assert_within_grid,
    fixed_c_edges,
    fixed_rg_edges,
    legacy_rg_grid,
    min_compact_rg,
    rod_rg,
)
from target_support import (
    CONTACT_OFFSETS,
    flat_level,
    load_target_contact_support,
    restrict_to_window,
    support_report,
    tail_mass_above,
)


Vec = Tuple[int, int, int]
Matrix = Tuple[Tuple[int, int, int], Tuple[int, int, int], Tuple[int, int, int]]
NN_VECS: Tuple[Vec, ...] = (
    (1, 0, 0), (-1, 0, 0), (0, 1, 0),
    (0, -1, 0), (0, 0, 1), (0, 0, -1),
)
# The four unit vectors perpendicular to each lattice direction.  Pull-move
# enumeration needs this set once per candidate anchor, so it is tabulated
# rather than recomputed as six dot products every time.
PERPENDICULAR: Dict[Vec, Tuple[Vec, ...]] = {
    bond: tuple(
        vector for vector in NN_VECS
        if bond[0] * vector[0] + bond[1] * vector[1] + bond[2] * vector[2] == 0
    )
    for bond in NN_VECS
}
BEND_DEFINITION = "number of 90-degree turns among the N-2 internal vertices"
RAW_SAMPLES_WARNING = (
    "These arrays are systematic importance resamples with duplicates; do not "
    "use them for variance or error-bar estimation. Use c_blocked_stderr."
)
# Cubic-lattice self-avoiding-walk connective constant; sets the analytic scale
# N*ln(mu) of the learned log-density across a full contact window.
SAW_CONNECTIVE_CONSTANT = 4.684
TIER_EXCLUDED = np.int8(0)
TIER_COVERAGE = np.int8(1)
TIER_FLAT = np.int8(2)
WL_STAGE_DTYPE = np.dtype(
    [
        ("log_f", np.float64),
        ("steps", np.int64),
        ("wall_seconds", np.float64),
        ("min_over_mean", np.float64),
        ("min_visits_tier2", np.int64),
        ("min_visits_tier1", np.int64),
        ("accepted", np.int64),
        ("round_trips", np.int64),
        ("highest_m", np.int64),
        ("slowest_levels", np.int64, (3,)),
    ]
)


def add(a: Vec, b: Vec) -> Vec:
    return a[0] + b[0], a[1] + b[1], a[2] + b[2]


def sub(a: Vec, b: Vec) -> Vec:
    return a[0] - b[0], a[1] - b[1], a[2] - b[2]


def _det3(matrix: Matrix) -> int:
    (a, b, c), (d, e, f), (g, h, i) = matrix
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)


def generate_cubic_rotations() -> Tuple[Matrix, ...]:
    identity: Matrix = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    rotations: List[Matrix] = []
    for perm in permutations((0, 1, 2)):
        for signs in product((-1, 1), repeat=3):
            raw = [[0, 0, 0] for _ in range(3)]
            for row, col in enumerate(perm):
                raw[row][col] = signs[row]
            matrix: Matrix = (tuple(raw[0]), tuple(raw[1]), tuple(raw[2]))
            if _det3(matrix) == 1 and matrix != identity:
                rotations.append(matrix)
    if len(rotations) != 23:
        raise RuntimeError(f"expected 23 non-identity cubic rotations, got {len(rotations)}")
    return tuple(rotations)


ROT_MATS = generate_cubic_rotations()


def apply_rot(matrix: Matrix, vector: Vec) -> Vec:
    x, y, z = vector
    return (
        matrix[0][0] * x + matrix[0][1] * y + matrix[0][2] * z,
        matrix[1][0] * x + matrix[1][1] * y + matrix[1][2] * z,
        matrix[2][0] * x + matrix[2][1] * y + matrix[2][2] * z,
    )


def contact_count(chain: Sequence[Vec], occupied: Set[Vec]) -> int:
    """Return the unique non-bonded nearest-neighbour contact count."""
    count = 0
    n_beads = len(chain)
    for i, site in enumerate(chain):
        previous = chain[i - 1] if i > 0 else None
        following = chain[i + 1] if i + 1 < n_beads else None
        for direction in NN_VECS:
            neighbour = add(site, direction)
            if neighbour in occupied and neighbour != previous and neighbour != following:
                count += 1
    if count % 2:
        raise RuntimeError("contact double-count is odd; chain bookkeeping is inconsistent")
    return count // 2


def coordination_histogram(
    chain: Sequence[Vec], occupied: Set[Vec]
) -> np.ndarray:
    """Per-bead nonbonded nearest-neighbour degree histogram for k=0..6.

    Bonded predecessor/successor sites are excluded exactly as in
    :func:`contact_count`.  The weighted degree sum is therefore twice the
    unique nonbonded contact count.
    """
    histogram = np.zeros(7, dtype=np.int64)
    n_beads = len(chain)
    for i, site in enumerate(chain):
        previous = chain[i - 1] if i > 0 else None
        following = chain[i + 1] if i + 1 < n_beads else None
        degree = 0
        for direction in NN_VECS:
            neighbour = add(site, direction)
            if (
                neighbour in occupied
                and neighbour != previous
                and neighbour != following
            ):
                degree += 1
        histogram[degree] += 1
    degree_sum = int(np.dot(np.arange(7, dtype=np.int64), histogram))
    if degree_sum % 2:
        raise RuntimeError("coordination degree sum is odd; chain is inconsistent")
    return histogram


def contact_delta_from_occupancy(old_occupied: Set[Vec], new_occupied: Set[Vec]) -> int:
    """Return the contact change using only sites added to or removed from occupancy.

    For a connected N-bead chain, ``m`` equals the number of occupied nearest-
    neighbour lattice edges minus the fixed ``N-1`` bonded edges.  We can
    therefore update ``m`` from the occupancy symmetric difference without
    rescanning the whole chain.  This makes corner-flip and end proposals O(1)
    while retaining an exact update for pivots, and it is exact for the
    arbitrary multi-bead symmetric differences a pull move produces.
    """
    removed = old_occupied - new_occupied
    added = new_occupied - old_occupied

    def incident_edges(changed: Set[Vec], occupied: Set[Vec]) -> int:
        edges = 0
        for site in changed:
            for direction in NN_VECS:
                neighbour = add(site, direction)
                if neighbour not in occupied:
                    continue
                # An edge joining two changed sites is otherwise counted twice.
                if neighbour in changed and neighbour < site:
                    continue
                edges += 1
        return edges

    return incident_edges(added, new_occupied) - incident_edges(removed, old_occupied)


def radius_of_gyration(chain: Sequence[Vec]) -> float:
    coordinates = np.asarray(chain, dtype=np.float64)
    centered = coordinates - coordinates.mean(axis=0)
    return float(np.sqrt(np.square(centered).sum(axis=1).mean()))


def count_bends(chain: Sequence[Vec]) -> int:
    bends = 0
    for i in range(1, len(chain) - 1):
        first = sub(chain[i], chain[i - 1])
        second = sub(chain[i + 1], chain[i])
        if first[0] * second[0] + first[1] * second[1] + first[2] * second[2] == 0:
            bends += 1
    return bends


# Every move function returns ``(valid, chain, occupied, log_q_ratio)`` with
#
#     log_q_ratio = log q(X'->X) - log q(X->X')
#
# the Hastings correction its own proposal kernel requires.  The three local
# families below are self-inverse or closed under inversion at fixed proposal
# probability, so their ratio is exactly zero; pull moves are not, and compute
# theirs by enumeration.  Keeping the term in the shared signature is what stops
# a future move family from silently inheriting a symmetry it does not have.
def attempt_pivot(
    chain: List[Vec], occupied: Set[Vec], rng: random.Random
) -> Tuple[bool, List[Vec], Set[Vec], float]:
    pivot_index = rng.randrange(1, len(chain) - 1)
    head = chain[: pivot_index + 1]
    pivot = chain[pivot_index]
    matrix = rng.choice(ROT_MATS)
    new_occupied = set(head)
    new_tail: List[Vec] = []
    for site in chain[pivot_index + 1 :]:
        moved = add(pivot, apply_rot(matrix, sub(site, pivot)))
        if moved in new_occupied:
            return False, chain, occupied, 0.0
        new_tail.append(moved)
        new_occupied.add(moved)
    # ROT_MATS is the full set of 23 non-identity proper cubic rotations, which
    # is closed under inversion, so the reverse rotation about the same index is
    # drawn with the same probability.
    return True, head + new_tail, new_occupied, 0.0


def attempt_corner_flip(
    chain: List[Vec], occupied: Set[Vec], rng: random.Random
) -> Tuple[bool, List[Vec], Set[Vec], float]:
    """Move one bead across a 90-degree corner (a kink jump).

    This is not a crankshaft: a crankshaft rotates a two-bead segment about the
    axis through its neighbours.  Exactly one bead moves here.
    """
    index = rng.randrange(1, len(chain) - 1)
    previous, current, following = chain[index - 1], chain[index], chain[index + 1]
    first = sub(current, previous)
    second = sub(following, current)
    dot = first[0] * second[0] + first[1] * second[1] + first[2] * second[2]
    if first not in NN_VECS or second not in NN_VECS or dot != 0:
        return False, chain, occupied, 0.0
    replacement = add(previous, second)
    if replacement in occupied:
        return False, chain, occupied, 0.0
    new_chain = chain.copy()
    new_chain[index] = replacement
    # Self-inverse: flipping the same index back reverses it, and the index is
    # drawn uniformly either way.
    return True, new_chain, (occupied - {current}) | {replacement}, 0.0


def attempt_end_move(
    chain: List[Vec], occupied: Set[Vec], rng: random.Random
) -> Tuple[bool, List[Vec], Set[Vec], float]:
    end = 0 if rng.random() < 0.5 else len(chain) - 1
    anchor = 1 if end == 0 else len(chain) - 2
    old_site = chain[end]
    occupied_without_old = occupied - {old_site}
    replacement = add(chain[anchor], rng.choice(NN_VECS))
    if replacement == old_site or replacement in occupied_without_old:
        return False, chain, occupied, 0.0
    new_chain = chain.copy()
    new_chain[end] = replacement
    # Symmetric: the same end and the same anchor propose the old site back with
    # probability 1/6, the probability that placed the new one.
    return True, new_chain, occupied_without_old | {replacement}, 0.0


MOVE_FUNCS = (attempt_pivot, attempt_corner_flip, attempt_end_move)

# A pull-move outcome, listing only the beads whose positions change, as
# ``((index, new_site), ...)`` sorted by index.  Descriptors rather than rebuilt
# chains keep enumeration O(pull length) per candidate instead of O(N).
PullOutcome = Tuple[Tuple[int, Vec], ...]


def _pull_move_catalog(
    chain: Sequence[Vec], occupied: Set[Vec]
) -> Dict[PullOutcome, None]:
    """Return every distinct Lesh-Mitzenmacher-Whitesides pull move from a state.

    A pull move is anchored at bead ``i`` and propagates a vacancy along the
    backbone in one index direction, so it can relocate a bead whose own
    neighbours are all occupied.  That is what pivots and corner flips cannot do,
    and why the compact region of the contact window is otherwise hard to
    re-enter once the Wang-Landau bias has been built up.

    With ``toward`` the index direction the vacancy propagates and
    ``nbr = i - toward`` the bead on the opposite side of the anchor:

    * ``b = chain[nbr] - chain[i]`` is the anchor bond and ``v`` ranges over the
      four unit vectors perpendicular to it;
    * ``L = chain[nbr] + v`` must be unoccupied, which makes ``L`` a
      face-diagonal neighbour of ``chain[i]`` and ``(chain[i], chain[nbr], L, C)``
      a unit square;
    * ``C = chain[i] + v``.

    Three cases produce a move.  ``i`` terminal on the propagation side: bead
    ``i`` alone moves to ``L``.  ``C == chain[i + toward]``: bead ``i`` alone
    moves to ``L``, which is degenerate with a corner flip but is still a legal
    member of this catalog -- dropping it would change the catalog size and hence
    every proposal ratio taken from it.  ``C`` unoccupied: bead ``i`` moves to
    ``L``, bead ``i + toward`` moves to ``C``, and the vacancy propagates until
    the chain reconnects.

    The catalog is deduplicated, because distinct ``(i, toward, v)`` triples can
    occasionally yield the same conformation and ``q = 1/len(catalog)`` is only
    exact once each reachable outcome appears exactly once.

    This catalog is *not* closed under inversion.  See ``attempt_pull_move`` for
    what that means and how it is handled; it is a property of the move set, not
    of this implementation.
    """
    n_beads = len(chain)
    outcomes: Dict[PullOutcome, None] = {}
    for anchor in range(n_beads):
        for toward in (-1, 1):
            neighbour = anchor - toward
            if not 0 <= neighbour < n_beads:
                continue
            bond = sub(chain[neighbour], chain[anchor])
            perpendicular = PERPENDICULAR.get(bond)
            if perpendicular is None:
                continue
            for offset in perpendicular:
                pulled = add(chain[neighbour], offset)
                if pulled in occupied:
                    continue
                corner = add(chain[anchor], offset)
                follower = anchor + toward
                if not 0 <= follower < n_beads:
                    # Case A: the anchor is the terminal bead on the propagation
                    # side, so nothing behind it needs to move.
                    outcomes[((anchor, pulled),)] = None
                    continue
                if corner == chain[follower]:
                    # Case B: the square is already closed by the follower.
                    outcomes[((anchor, pulled),)] = None
                    continue
                if corner in occupied:
                    continue
                # Case C: pull the anchor and its follower, then propagate the
                # vacancy backwards until the chain reconnects.  ``vacated``
                # tracks the sites released so far: a newly assigned position is
                # legal only if it is free in the current state or has just been
                # released.  This is enforced here rather than assumed, so that
                # self-avoidance is an invariant of the catalog; a candidate that
                # violates it is dropped instead of emitted.
                moves: List[Tuple[int, Vec]] = [(anchor, pulled), (follower, corner)]
                vacated = {chain[anchor], chain[follower]}
                assigned = {pulled, corner}
                broken = False
                index = anchor + 2 * toward
                while 0 <= index < n_beads:
                    previous_new = moves[-1][1]
                    if sub(chain[index], previous_new) in NN_VECS:
                        break
                    replacement = chain[index - 2 * toward]
                    if replacement in assigned or (
                        replacement in occupied and replacement not in vacated
                    ):
                        broken = True
                        break
                    moves.append((index, replacement))
                    vacated.add(chain[index])
                    assigned.add(replacement)
                    index += toward
                if broken:
                    continue
                outcomes[tuple(sorted(moves))] = None
    return outcomes


def enumerate_pull_moves(
    chain: Sequence[Vec], occupied: Set[Vec]
) -> List[PullOutcome]:
    """Return the deduplicated pull-move catalog of a state as a list.

    Order is the enumeration order and is deterministic.  See
    ``_pull_move_catalog`` for the geometry.
    """
    return list(_pull_move_catalog(chain, occupied))


def reverse_pull_outcome(
    chain: Sequence[Vec], outcome: PullOutcome
) -> PullOutcome:
    """Return the descriptor that undoes ``outcome`` applied to ``chain``.

    The two states differ on exactly the indices ``outcome`` names, so the
    inverse is those same indices carrying their original positions.
    """
    return tuple(sorted((index, chain[index]) for index, _ in outcome))


def apply_pull_move(
    chain: Sequence[Vec], occupied: Set[Vec], outcome: PullOutcome
) -> Tuple[List[Vec], Set[Vec]]:
    """Apply a pull-move descriptor, returning fresh chain and occupancy sets."""
    new_chain = list(chain)
    new_occupied = set(occupied)
    # Every vacated site is released before any new site is claimed.  Interleaving
    # the two would erase a bead that moved into a site another bead just left.
    for index, _ in outcome:
        new_occupied.discard(chain[index])
    for index, site in outcome:
        new_chain[index] = site
        new_occupied.add(site)
    return new_chain, new_occupied


def attempt_pull_move(
    chain: List[Vec], occupied: Set[Vec], rng: random.Random
) -> Tuple[bool, List[Vec], Set[Vec], float]:
    """Propose a uniformly chosen pull move and return its Hastings ratio.

    The proposal is uniform over the forward catalog, so ``q(X->X') = 1/n_f``
    and ``q(X'->X) = 1/n_r`` with ``n_f`` and ``n_r`` the deduplicated catalog
    sizes at ``X`` and ``X'``.  Therefore

        q(X'->X) / q(X->X') = (1/n_r) / (1/n_f) = n_f / n_r

    and the returned term is ``log(n_f) - log(n_r)``.  The sign is easy to
    invert: the *forward* count is in the numerator.

    That formula presumes the reverse move exists at all, and for pull moves on
    a rectangular lattice it does not always exist.  Lesh, Mitzenmacher and
    Whitesides claimed the move set was reversible, but the proof is wrong:
    Gyorffy, Zavodszky and Szilagyi (arXiv:1210.0495, J. Comput. Chem. 2013)
    showed some pull moves have no inverse pull move, "which leads to biases in
    the parameters estimated from the simulations".  Measured here, single-bead
    outcomes always invert, while multi-bead outcomes invert only about 60% of
    the time -- so this is a property of the move set, not of this code.

    The cheapest correct repair is to reject the proposal outright whenever the
    inverse is absent.  Let ``R`` be the set of ordered pairs ``(X, X')`` with
    ``X'`` in the catalog of ``X`` *and* ``X`` in the catalog of ``X'``.  ``R``
    is symmetric by construction.  Proposing uniformly from the full catalog and
    rejecting every proposal outside ``R`` gives, for ``X != X'``,

        pi(X) T(X->X') = 1[(X,X') in R] * min(pi(X)/n_f, pi(X')/n_r)

    which is symmetric under exchanging ``X`` and ``X'``, so detailed balance
    holds exactly.  The rejected proposals are ordinary self-loops and cost only
    efficiency.  Note ``n_f`` and ``n_r`` remain the *full* catalog sizes: the
    irreversible members are still proposable, just never accepted, so removing
    them from the counts would break the very balance this restores.

    This costs nothing extra -- the reverse catalog is already built to obtain
    ``n_r``, so the repair is one membership test.  Gyorffy et al. instead extend
    the move set until it is closed under inversion, which wastes fewer
    proposals but is a larger change than this sampler needs.

    No accounting across move types is needed.  A mixture of kernels that each
    satisfy detailed balance with respect to pi also satisfies detailed balance
    with respect to pi, provided the mixture weights do not depend on the state.
    Each move type therefore needs only its own correct ratio, and we never sum
    ``q`` over the several move types that could produce the same ``X'``.
    """
    catalog_forward = _pull_move_catalog(chain, occupied)
    if not catalog_forward:
        return False, chain, occupied, 0.0
    forward_count = len(catalog_forward)
    outcome = list(catalog_forward)[rng.randrange(forward_count)]
    proposed_chain, proposed_occupied = apply_pull_move(chain, occupied, outcome)
    catalog_reverse = _pull_move_catalog(proposed_chain, proposed_occupied)
    if reverse_pull_outcome(chain, outcome) not in catalog_reverse:
        # Irreversible proposal: no inverse pull move exists, so accepting it at
        # any ratio would violate detailed balance.  Reject as a self-loop.
        return False, chain, occupied, 0.0
    log_q_ratio = math.log(forward_count) - math.log(len(catalog_reverse))
    return True, proposed_chain, proposed_occupied, log_q_ratio


def validate_chain(chain: Sequence[Vec]) -> None:
    if len(set(chain)) != len(chain):
        raise RuntimeError("chain is not self-avoiding")
    for first, second in zip(chain[:-1], chain[1:]):
        if sub(second, first) not in NN_VECS:
            raise RuntimeError("chain contains a non-unit lattice bond")


def metropolis_step(
    chain: List[Vec],
    occupied: Set[Vec],
    contact: int,
    log_g: np.ndarray,
    m_min: int,
    m_max: int,
    rng: random.Random,
    pull_move_weight: float = 0.0,
) -> Tuple[List[Vec], Set[Vec], int, bool, bool]:
    """Take one fixed-bias proposal.

    Returns ``chain, occupied, contact, geometrically_valid, accepted``.  Each
    move type carries its own proposal ratio, returned as ``log_q_ratio`` and
    added to the log acceptance: the three local families are symmetric and
    return zero, while pull moves return the Hastings term their asymmetric
    catalog requires.

    ``pull_move_weight`` is the probability of proposing a pull move, with the
    remainder split evenly among the three local families.  It must not depend
    on the state -- not on the contact number, the chain density, or anything
    else -- or the mixture weight fails to cancel in the ratio above and leaves
    an uncorrected factor in ``q``.
    """
    # Short-circuit rather than compare against a drawn variate: at weight 0.0
    # this consumes no random number, so the stream is bit-identical to a build
    # with no pull moves at all.
    if pull_move_weight > 0.0 and rng.random() < pull_move_weight:
        move = attempt_pull_move
    else:
        move = rng.choice(MOVE_FUNCS)
    valid, proposed_chain, proposed_occupied, log_q_ratio = move(chain, occupied, rng)
    if not valid:
        return chain, occupied, contact, False, False
    proposed_contact = contact + contact_delta_from_occupancy(occupied, proposed_occupied)
    if proposed_contact < m_min or proposed_contact > m_max:
        return chain, occupied, contact, True, False
    log_acceptance = float(
        log_g[contact - m_min] - log_g[proposed_contact - m_min]
    ) + log_q_ratio
    if log_acceptance >= 0.0 or rng.random() < math.exp(log_acceptance):
        return proposed_chain, proposed_occupied, proposed_contact, True, True
    return chain, occupied, contact, True, False


class RoundTripCounter:
    """Count completed low -> high -> low traversals of a contact window."""

    def __init__(self, low: int, high: int) -> None:
        self.low = int(low)
        self.high = int(high)
        self.phase = 0
        self.round_trips = 0

    def observe(self, contact: int) -> None:
        if self.low == self.high:
            return
        if self.phase == 0 and contact <= self.low:
            self.phase = 1
        elif self.phase == 1 and contact >= self.high:
            self.phase = 2
        elif self.phase == 2 and contact <= self.low:
            self.round_trips += 1
            self.phase = 1


def _save_checkpoint(
    path: Path,
    *,
    n_beads: int,
    m_min: int,
    m_max: int,
    chain: Sequence[Vec],
    log_g: np.ndarray,
    next_log_f: float,
    stages_completed: int,
    attempted_steps: int,
    accepted_moves: int,
    round_trips: int,
    seed: int,
    tier: np.ndarray,
    schedule: str,
    pull_move_weight: float,
    one_over_t_mode: bool,
    one_over_t_trigger_reason: str,
    one_over_t_round_trips: int,
    stall_relaxed: bool,
    stall_relaxed_step: int,
    visits_since_start: np.ndarray,
    visits_since_one_over_t: np.ndarray,
    visits_since_stall: np.ndarray,
    stage_records: np.ndarray,
    learning_wall_seconds: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(
            temporary,
            checkpoint_version=np.array(2, dtype=np.int64),
            N=np.array(n_beads, dtype=np.int64),
            m_min=np.array(m_min, dtype=np.int64),
            m_max=np.array(m_max, dtype=np.int64),
            chain=np.asarray(chain, dtype=np.int64),
            log_g=np.asarray(log_g, dtype=np.float64),
            next_log_f=np.array(next_log_f, dtype=np.float64),
            stages_completed=np.array(stages_completed, dtype=np.int64),
            attempted_steps=np.array(attempted_steps, dtype=np.int64),
            accepted_moves=np.array(accepted_moves, dtype=np.int64),
            round_trips=np.array(round_trips, dtype=np.int64),
            original_seed=np.array(seed, dtype=np.int64),
            wl_tier=np.asarray(tier, dtype=np.int8),
            wl_schedule=np.array(schedule),
            wl_pull_move_weight=np.array(pull_move_weight, dtype=np.float64),
            one_over_t_mode=np.array(one_over_t_mode, dtype=bool),
            wl_one_over_t_trigger=np.array(one_over_t_trigger_reason),
            wl_one_over_t_round_trips=np.array(
                one_over_t_round_trips, dtype=np.int64
            ),
            wl_stall_relaxed=np.array(stall_relaxed, dtype=bool),
            wl_stall_relaxed_step=np.array(stall_relaxed_step, dtype=np.int64),
            wl_visits_since_start=np.asarray(visits_since_start, dtype=np.int64),
            wl_visits_since_one_over_t=np.asarray(
                visits_since_one_over_t, dtype=np.int64
            ),
            wl_visits_since_stall=np.asarray(visits_since_stall, dtype=np.int64),
            wl_stage_records=np.asarray(stage_records, dtype=WL_STAGE_DTYPE),
            learning_wall_seconds=np.array(learning_wall_seconds, dtype=np.float64),
        )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _rng_state_to_arrays(state: Tuple[Any, ...]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split ``random.Random.getstate()`` into pickle-free numeric arrays.

    The state is ``(version, tuple of 625 ints, gauss_next)``.  Checkpoints load
    with ``allow_pickle=False``, so the tuple is stored as ``uint32`` and the
    optional ``gauss_next`` as a float with NaN standing for ``None``.
    """
    version, keys, gauss_next = state
    return (
        np.array(version, dtype=np.int64),
        np.asarray(keys, dtype=np.uint32),
        np.array(math.nan if gauss_next is None else gauss_next, dtype=np.float64),
    )


def _rng_state_from_arrays(version: Any, keys: Any, gauss_next: Any) -> Tuple[Any, ...]:
    """Rebuild a ``random.Random`` state from ``_rng_state_to_arrays`` output."""
    gauss_value = float(gauss_next)
    return (
        int(version),
        tuple(int(value) for value in np.asarray(keys, dtype=np.uint32)),
        None if math.isnan(gauss_value) else gauss_value,
    )


def production_checkpoint_path(stem: Path, worker_id: int) -> Path:
    """Return the per-worker production checkpoint file for a checkpoint stem."""
    stem = Path(stem)
    return stem.with_name(f"{stem.stem}_prod_w{worker_id}.npz")


def _save_production_checkpoint(
    path: Path,
    *,
    worker_id: int,
    n_beads: int,
    m_min: int,
    m_max: int,
    log_g: np.ndarray,
    tier: np.ndarray,
    chain: Sequence[Vec],
    contact: int,
    steps_done: int,
    accepted: int,
    geometrically_valid: int,
    contacts: Sequence[int],
    radii: Sequence[float],
    bends: Sequence[int],
    tracker: RoundTripCounter,
    rng_state: Tuple[Any, ...],
    coordination_histograms: Optional[Sequence[Sequence[int]]] = None,
) -> None:
    """Write one worker's production state atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rng_version, rng_keys, rng_gauss = _rng_state_to_arrays(rng_state)
    coordination_array = np.asarray(
        [] if coordination_histograms is None else coordination_histograms,
        dtype=np.int64,
    )
    if coordination_array.size == 0:
        coordination_array = np.empty((0, 7), dtype=np.int64)
    if coordination_array.ndim != 2 or coordination_array.shape[1] != 7:
        raise ValueError("checkpoint coordination histograms must have shape (S, 7)")
    if coordination_array.shape[0] != len(contacts):
        raise ValueError(
            "checkpoint coordination histogram count must match contact samples"
        )
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(
            temporary,
            checkpoint_version=np.array(2, dtype=np.int64),
            worker_id=np.array(worker_id, dtype=np.int64),
            N=np.array(n_beads, dtype=np.int64),
            m_min=np.array(m_min, dtype=np.int64),
            m_max=np.array(m_max, dtype=np.int64),
            log_g=np.asarray(log_g, dtype=np.float64),
            wl_tier=np.asarray(tier, dtype=np.int8),
            chain=np.asarray(chain, dtype=np.int64),
            contact=np.array(contact, dtype=np.int64),
            steps_done=np.array(steps_done, dtype=np.int64),
            accepted_moves=np.array(accepted, dtype=np.int64),
            geometrically_valid_moves=np.array(geometrically_valid, dtype=np.int64),
            contact_samples=np.asarray(contacts, dtype=np.int64),
            rg_samples=np.asarray(radii, dtype=np.float64),
            bend_samples=np.asarray(bends, dtype=np.int64),
            coordination_histogram_samples=coordination_array,
            round_trip_phase=np.array(tracker.phase, dtype=np.int64),
            round_trips=np.array(tracker.round_trips, dtype=np.int64),
            rng_version=rng_version,
            rng_keys=rng_keys,
            rng_gauss_next=rng_gauss,
        )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_production_checkpoint(
    path: Path,
    *,
    n_beads: int,
    m_min: int,
    m_max: int,
    log_g: np.ndarray,
    tier: np.ndarray,
) -> Dict[str, Any]:
    """Load one worker's production state, refusing any run mismatch.

    A resumed chain restarts its random stream from the saved state, so it is
    statistically valid but not a bitwise continuation, exactly as for a resumed
    WL stage.
    """
    with np.load(path, allow_pickle=False) as saved:
        checkpoint_version = int(saved["checkpoint_version"])
        if checkpoint_version not in (1, 2):
            raise ValueError(
                "unsupported production checkpoint version "
                f"{checkpoint_version}; supported versions are 1 and 2"
            )
        saved_window = (int(saved["N"]), int(saved["m_min"]), int(saved["m_max"]))
        if saved_window != (n_beads, m_min, m_max):
            raise ValueError(
                "production checkpoint N/contact window does not match the "
                f"requested run: saved={saved_window}, "
                f"requested={(n_beads, m_min, m_max)}"
            )
        if not np.array_equal(np.asarray(saved["log_g"], dtype=np.float64), log_g):
            raise ValueError(
                "production checkpoint log_g does not match the frozen weights of "
                "the requested run"
            )
        if not np.array_equal(np.asarray(saved["wl_tier"], dtype=np.int8), tier):
            raise ValueError(
                "production checkpoint contact tiers do not match the requested run"
            )
        chain = [tuple(map(int, row)) for row in np.asarray(saved["chain"])]
        saved_contact = int(saved["contact"])
        contacts = [int(value) for value in np.asarray(saved["contact_samples"])]
        if "coordination_histogram_samples" in saved.files:
            coordination_histograms = np.asarray(
                saved["coordination_histogram_samples"], dtype=np.int64
            )
        elif contacts:
            raise ValueError(
                "production checkpoint predates local coordination observables "
                "and already contains samples; restart production from the frozen "
                "Wang-Landau weights"
            )
        else:
            coordination_histograms = np.empty((0, 7), dtype=np.int64)
        if coordination_histograms.shape != (len(contacts), 7):
            raise ValueError(
                "production checkpoint coordination samples do not match contact samples"
            )
        state = {
            "chain": chain,
            "contact": saved_contact,
            "steps_done": int(saved["steps_done"]),
            "accepted": int(saved["accepted_moves"]),
            "geometrically_valid": int(saved["geometrically_valid_moves"]),
            "contacts": contacts,
            "radii": [float(value) for value in np.asarray(saved["rg_samples"])],
            "bends": [int(value) for value in np.asarray(saved["bend_samples"])],
            "coordination_histograms": [
                row.copy() for row in coordination_histograms
            ],
            "checkpoint_version": checkpoint_version,
            "round_trip_phase": int(saved["round_trip_phase"]),
            "round_trips": int(saved["round_trips"]),
            "rng_state": _rng_state_from_arrays(
                saved["rng_version"], saved["rng_keys"], saved["rng_gauss_next"]
            ),
        }
    validate_chain(chain)
    if contact_count(chain, set(chain)) != saved_contact:
        raise ValueError(
            "production checkpoint contact count disagrees with its stored chain"
        )
    return state


GEOMETRIC_CONTACT_MAXIMA = {
    # Exact maxima for the chain lengths used by this project.  Each is attained
    # by a compact cubic-lattice shape with a Hamiltonian path.
    30: 30,  # 2 x 3 x 5 cuboid
    44: 50,  # 3 x 3 x 5 cuboid with one corner removed
    60: 74,  # 3 x 4 x 5 cuboid
}


def contact_upper_bound(n_beads: int) -> int:
    """Return a rigorous, not necessarily attainable, contact upper bound.

    N sites induce at most 3N - 3N^(2/3) lattice edges and N-1 of them are
    bonded, so m <= 2N + 1 - 3N^(2/3).  Integer lattice geometry and the
    Hamiltonian-path constraint can make the exact maximum smaller.
    """
    return math.floor(2 * n_beads + 1 - 3 * n_beads ** (2.0 / 3.0))


def geometric_contact_maximum(n_beads: int) -> Optional[int]:
    """Return the verified exact contact maximum, when one is encoded."""
    return GEOMETRIC_CONTACT_MAXIMA.get(n_beads)


# Optimal (maximally compact) bounding boxes, from an exact MILP over induced
# edges plus a Hamiltonian-path check.  Volume may exceed N; the snake is
# truncated to N sites and the resulting contact count is asserted, never
# assumed.
COMPACT_SEED_BOXES = {30: (2, 3, 5), 44: (3, 3, 5), 60: (3, 4, 5)}


def _boustrophedon(dims: Tuple[int, int, int]) -> List[Vec]:
    """Hamiltonian path through a box, as a list of lattice sites.

    Requires an odd extent in the middle axis position, otherwise the
    layer-to-layer step is not a nearest-neighbour move.  Every optimal box
    here contains a 5, so an odd extent can always be permuted into place.
    """
    odd = [t for t in range(3) if dims[t] % 2 == 1]
    if not odd:
        raise ValueError(f"no odd extent in box {dims}; snake is not a path")
    mid = odd[0]
    rest = [t for t in range(3) if t != mid]
    perm = (rest[0], mid, rest[1])
    a, b, c = dims[perm[0]], dims[perm[1]], dims[perm[2]]
    path: List[Vec] = []
    for i in range(a):
        for jj in range(b):
            j = jj if i % 2 == 0 else b - 1 - jj
            for kk in range(c):
                k = kk if (i + jj) % 2 == 0 else c - 1 - kk
                cell = [0, 0, 0]
                cell[perm[0]] = i
                cell[perm[1]] = j
                cell[perm[2]] = k
                path.append(tuple(cell))
    return path


def compact_seed_chain(n_beads: int) -> List[Vec]:
    """Return a SAW of n_beads with exactly the geometric maximum contacts."""
    box = COMPACT_SEED_BOXES.get(n_beads)
    if box is None:
        raise ValueError(
            f"no verified compact box encoded for N={n_beads}; supply one only "
            "after an independent exact-maximum verification"
        )
    if box[0] * box[1] * box[2] < n_beads:
        raise ValueError(f"box {box} cannot hold {n_beads} beads")
    chain = _boustrophedon(box)[:n_beads]
    validate_chain(chain)
    contacts = contact_count(chain, set(chain))
    expected = geometric_contact_maximum(n_beads)
    if expected is not None and contacts != expected:
        raise RuntimeError(
            f"compact seed for N={n_beads} has m={contacts}, expected the "
            f"verified geometric maximum {expected}"
        )
    return chain


def report_contact_upper_bound(args: argparse.Namespace) -> int:
    """Print the analytical upper bound and any verified exact maximum."""
    upper_bound = contact_upper_bound(args.N)
    exact_maximum = geometric_contact_maximum(args.N)
    print(f"rigorous contact upper bound for N={args.N}: {upper_bound}")
    if exact_maximum is None:
        print("verified exact geometric m_max: not encoded for this chain length")
    else:
        print(f"verified exact geometric m_max: {exact_maximum}")
    return 0


def make_contact_tiers(
    m_min: int,
    m_max: int,
    m_flat: int,
    m_cover: int,
    excluded_contact_levels: Sequence[int] = (),
) -> np.ndarray:
    """Build and validate the three-tier contact window."""
    if not m_min <= m_flat <= m_cover <= m_max:
        raise ValueError(
            "contact tiers require m_min <= m_flat <= m_cover <= m_max; "
            f"got {(m_min, m_flat, m_cover, m_max)}"
        )
    window = np.arange(m_min, m_max + 1, dtype=np.int64)
    tier = np.full(window.size, TIER_EXCLUDED, dtype=np.int8)
    tier[window <= m_cover] = TIER_COVERAGE
    tier[window <= m_flat] = TIER_FLAT
    excluded = np.asarray(excluded_contact_levels, dtype=np.int64)
    if excluded.size:
        if np.unique(excluded).size != excluded.size:
            raise ValueError("--excluded_contact_levels contains duplicates")
        invalid = excluded[(excluded <= m_min) | (excluded >= m_cover)]
        if invalid.size:
            raise ValueError(
                "--excluded_contact_levels may contain only verified internal "
                f"gaps inside (m_min, m_cover); invalid values: {invalid.tolist()}"
            )
        tier[excluded - m_min] = TIER_EXCLUDED
    if not np.any(tier == TIER_FLAT):
        raise ValueError("the contact window contains no tier-2 levels")
    included = np.flatnonzero(tier > TIER_EXCLUDED)
    if included.size == 0 or included[0] != 0 or included[-1] != m_cover - m_min:
        raise ValueError("the included contact window must retain both endpoints")
    return tier


def _validate_tier(tier: np.ndarray, expected_size: int) -> np.ndarray:
    tier = np.asarray(tier, dtype=np.int8)
    if tier.shape != (expected_size,):
        raise ValueError("tier array shape does not match the contact window")
    if not np.all(np.isin(tier, (TIER_EXCLUDED, TIER_COVERAGE, TIER_FLAT))):
        raise ValueError("tier array may contain only 0, 1, and 2")
    if not np.any(tier == TIER_FLAT):
        raise ValueError("tier array contains no flat levels")
    return tier


def _resume_histogram(saved: Any, key: str, expected_size: int) -> np.ndarray:
    """Load a checkpointed visit histogram, validating it like ``wl_tier``."""
    if key not in saved:
        return np.zeros(expected_size, dtype=np.int64)
    histogram = np.asarray(saved[key], dtype=np.int64)
    if histogram.shape != (expected_size,):
        raise ValueError(
            f"checkpoint {key} shape does not match the requested contact "
            f"window: saved={histogram.shape}, requested={(expected_size,)}"
        )
    if np.any(histogram < 0):
        raise ValueError(f"checkpoint {key} contains negative visit counts")
    return histogram


def _included_limits(tier: np.ndarray, m_min: int) -> Tuple[int, int]:
    included = np.flatnonzero(tier > TIER_EXCLUDED)
    if included.size == 0:
        raise ValueError("tier array excludes every contact level")
    return int(included[0] + m_min), int(included[-1] + m_min)


def _slowest_levels(histogram: np.ndarray, tier: np.ndarray, m_min: int) -> np.ndarray:
    included = np.flatnonzero(tier > TIER_EXCLUDED)
    order = included[np.argsort(histogram[included], kind="stable")]
    levels = np.full(3, -1, dtype=np.int64)
    count = min(3, order.size)
    levels[:count] = order[:count] + m_min
    return levels


def _report_log_g_spread(
    log_g: np.ndarray, included: np.ndarray, n_beads: int, progress: bool
) -> Tuple[float, float]:
    """Report the learned bias amplitude against its analytic scale.

    ``log g(m)`` spans roughly ``N ln(mu)`` over the whole window, with ``mu``
    the cubic-lattice self-avoiding-walk connective constant.  A spread far
    above that is dominated by accumulation noise rather than by the density of
    states, which is the signature of a bias that was frozen before the window
    was properly covered.  This is advisory: the production support and round
    trip gates remain the hard checks.
    """
    values = log_g[included]
    spread = float(values.max() - values.min())
    reference = float(n_beads * math.log(SAW_CONNECTIVE_CONSTANT))
    if progress:
        print(
            f"learned log_g spread over included levels: {spread:.3g} "
            f"(analytic scale N*ln(mu)={reference:.3g})",
            flush=True,
        )
    if spread > 10.0 * reference:
        print(
            f"WARNING: learned log_g spread {spread:.6g} exceeds ten times the "
            f"analytic scale {reference:.6g}; the bias is likely dominated by "
            "accumulation noise rather than the density of states. Inspect "
            "wl_visits_since_one_over_t and wl_visits_since_stall against "
            "wl_visits_since_start before trusting the frozen weights.",
            flush=True,
        )
    return spread, reference


def _coverage_satisfied(
    range_covered: bool,
    min_flat: int,
    min_coverage: int,
    tier: np.ndarray,
    min_visits: int,
    min_cover_visits: int,
) -> bool:
    """Return whether a visit histogram meets the coverage requirement."""
    return bool(
        range_covered
        and min_flat >= min_visits
        and (
            not np.any(tier == TIER_COVERAGE)
            or min_coverage >= min_cover_visits
        )
    )


def _report_cap_reconciliation(
    *,
    stage_steps: int,
    elapsed: float,
    max_steps: int,
    max_seconds: float,
    elapsed_total: float,
    progress: bool,
) -> None:
    """Report which of the step and time caps actually binds this run.

    Neither cap is altered.  A step cap far beyond what the time cap permits is
    dead code, and silently leaving it in place hides which limit stops the run.
    """
    if elapsed <= 0.0 or not math.isfinite(max_seconds):
        return
    rate = stage_steps / elapsed
    remaining_seconds = max(max_seconds - elapsed_total, 0.0)
    time_implied_steps = rate * remaining_seconds
    binding = "--wl_max_steps" if max_steps <= time_implied_steps else "--wl_max_seconds"
    if progress:
        print(
            f"[WL] measured rate={rate:.0f} steps/s; --wl_max_seconds permits "
            f"about {time_implied_steps:.3g} further steps versus "
            f"--wl_max_steps={max_steps:.3g}; binding cap is {binding}",
            flush=True,
        )
    if max_steps > 2.0 * time_implied_steps:
        warnings.warn(
            f"--wl_max_steps={max_steps:.3g} is unreachable within "
            f"--wl_max_seconds={max_seconds:.6g}: at the measured "
            f"{rate:.0f} steps/s the time cap permits only about "
            f"{time_implied_steps:.3g} further steps. --wl_max_seconds is the "
            "operative limit; neither value has been changed.",
            RuntimeWarning,
            stacklevel=2,
        )


PRODUCTION_PROBE_STEPS = 20_000


def _report_production_budget(
    *,
    chain: Sequence[Vec],
    log_g: np.ndarray,
    m_min: int,
    tier: np.ndarray,
    steps_per_worker: int,
    n_workers: int,
    pull_move_weight: float,
    seed: int,
    max_seconds: float,
    probe_steps: int = PRODUCTION_PROBE_STEPS,
) -> float:
    """Measure the frozen-weight step rate and project the production budget.

    The learning phase reports its binding cap through
    ``_report_cap_reconciliation``; production had no equivalent, so a wall
    clock far beyond the partition limit was only discoverable after the run had
    already been killed.  This runs a short throwaway burst on the frozen
    ``log_g`` and refuses to launch when the projection exceeds
    ``--production_max_seconds``.  Returns the measured steps per second.
    """
    _, included_high = _included_limits(tier, m_min)
    probe_chain = [tuple(site) for site in chain]
    occupied = set(probe_chain)
    contact = contact_count(probe_chain, occupied)
    rng = random.Random(seed)
    probe = max(1, min(probe_steps, steps_per_worker))
    t0 = time.time()
    for _ in range(probe):
        probe_chain, occupied, contact, _, _ = metropolis_step(
            probe_chain, occupied, contact, log_g, m_min, included_high, rng,
            pull_move_weight,
        )
    elapsed = time.time() - t0
    if elapsed <= 0.0:
        return math.inf
    rate = probe / elapsed
    per_worker_seconds = steps_per_worker / rate
    core_seconds = n_workers * per_worker_seconds
    print(
        f"[production] measured rate={rate:.0f} steps/s over {probe} probe "
        f"steps; --steps_per_worker={steps_per_worker:.3g} projects about "
        f"{per_worker_seconds / 3600.0:.1f} h per worker "
        f"({per_worker_seconds:.3g} s), {core_seconds / 3600.0:.1f} core-hours "
        f"over {n_workers} parallel workers",
        flush=True,
    )
    if per_worker_seconds > max_seconds:
        fitting = int(max_seconds * rate)
        raise RuntimeError(
            f"projected per-worker production wall clock "
            f"{per_worker_seconds:.3g} s ({per_worker_seconds / 3600.0:.1f} h) "
            f"exceeds --production_max_seconds={max_seconds:.6g} "
            f"({max_seconds / 3600.0:.1f} h): at the measured {rate:.0f} steps/s "
            f"only --steps_per_worker={fitting} would fit. Neither value has "
            "been changed; production was not launched. Use "
            "--production_checkpoint to run the full budget across several "
            "invocations instead of shortening it"
        )
    return rate


def one_over_t_trigger(
    log_f: float, inverse_time: float, stage_steps: int, stall_steps: int
) -> Tuple[bool, str]:
    """Decide whether to enter the asymptotic ``1/t`` phase.

    Two independent conditions enter the phase.  The stall trigger fires when a
    refinement stage has run ``stall_steps`` attempted moves without completing;
    this is the operative one, because a stage that cannot reach coverage is
    exactly the situation the ``1/t`` schedule exists to escape.  The
    Belardinelli-Pereyra trigger fires when ``log_f`` has fallen to or below
    ``1/t``, after which the halving schedule would drive the modification
    factor below the level where the error stops decreasing.

    The stall is tested first so that its reason is reported when both hold.
    Returns ``(fire, reason)`` with reason in
    ``{"", "stall", "belardinelli_pereyra"}``.
    """
    if stage_steps >= stall_steps:
        return True, "stall"
    if log_f <= inverse_time:
        return True, "belardinelli_pereyra"
    return False, ""


def _mode_label(schedule: str, one_over_t_mode: bool, stall_relaxed: bool) -> str:
    """Name the refinement law currently in force, for logs and failures."""
    if one_over_t_mode:
        return "1/t"
    if stall_relaxed:
        return f"{schedule}/stall_relaxed"
    return schedule


def _visited_flatness_ratio(histogram: np.ndarray, tier: np.ndarray) -> float:
    """Minimum-to-mean visit ratio over the tier-2 levels actually visited.

    A stall-relaxed stage cannot use the ordinary ratio: one level the chain
    never reaches drives ``min`` to zero and pins the ratio at zero forever.
    Restricting the ratio to visited levels still requires the reachable part
    of the window to be flat, while the unvisited levels remain governed by the
    cumulative coverage counts, which are never relaxed.
    """
    flat_counts = histogram[tier == TIER_FLAT]
    visited = flat_counts[flat_counts > 0]
    if visited.size == 0:
        return 0.0
    mean_visited = float(visited.mean())
    return float(visited.min()) / mean_visited if mean_visited else 0.0


def _stage_visit_statistics(
    histogram: np.ndarray, tier: np.ndarray
) -> Tuple[float, int, int, bool]:
    flat_counts = histogram[tier == TIER_FLAT]
    mean_flat = float(flat_counts.mean())
    min_flat = int(flat_counts.min())
    ratio = min_flat / mean_flat if mean_flat else 0.0
    coverage_counts = histogram[tier == TIER_COVERAGE]
    min_coverage = int(coverage_counts.min()) if coverage_counts.size else 0
    included = np.flatnonzero(tier > TIER_EXCLUDED)
    endpoints_covered = bool(
        histogram[included[0]] > 0 and histogram[included[-1]] > 0
    )
    return ratio, min_flat, min_coverage, endpoints_covered


def learn_log_density(
    *,
    n_beads: int,
    m_min: int,
    m_max: int,
    seed: int,
    initial_log_f: float,
    final_log_f: float,
    flatness: float,
    min_visits: int,
    check_every: int,
    max_steps: int,
    min_cover_visits: int = 50,
    max_seconds: float = math.inf,
    max_seconds_scope: str = "cumulative",
    max_steps_per_stage: int = 1_000_000_000,
    schedule: str = "halving",
    stage_stall_steps: int = 50_000_000,
    pull_move_weight: float = 0.0,
    checkpoint_every_seconds: float = 1800.0,
    init: str = "rod",
    tier: Optional[np.ndarray] = None,
    checkpoint: Optional[Path] = None,
    resume_checkpoint: Optional[Path] = None,
    progress: bool = True,
) -> Dict[str, Any]:
    """Learn a contact density estimate over a tiered contact window.

    The halving schedule uses the tier-2 flatness ratio and minimum visit counts
    in both included tiers.  The Belardinelli-Pereyra schedule uses cumulative
    time per included level and drops the flatness-ratio requirement during its
    pre-asymptotic coverage stages and continuous ``1/t`` phase.

    Once the ``1/t`` phase begins there are no further stages, so per-stage
    coverage is no longer meaningful.  Coverage is then judged against a visit
    histogram that is reset at the moment the phase begins, which requires the
    frozen bias to have been exercised over the whole included window at its
    final resolution rather than once per refinement stage.

    The two triggers that can end the halving phase are not interchangeable.
    The Belardinelli-Pereyra crossing is rate-neutral by construction -- at the
    crossing ``log_f`` and ``1/t`` are equal -- so adopting ``1/t`` as the
    refinement factor there is continuous, and it enters the ``1/t`` phase.  A
    stall carries no such guarantee: when it fires, ``1/t`` is typically orders
    of magnitude below ``log_f``, so adopting it would collapse the refinement
    factor and freeze a barely-learned bias.  The stall therefore does *not*
    enter the ``1/t`` phase and does not touch ``log_f``.  It only relaxes how
    a stage is permitted to advance, and halving continues normally with the
    Belardinelli-Pereyra trigger still armed.
    """
    if schedule not in {"halving", "one_over_t"}:
        raise ValueError("schedule must be 'halving' or 'one_over_t'")
    if max_seconds_scope not in {"cumulative", "per_invocation"}:
        raise ValueError(
            "max_seconds_scope must be 'cumulative' or 'per_invocation'"
        )
    if min(
        min_visits, min_cover_visits, check_every, max_steps,
        max_steps_per_stage, stage_stall_steps,
    ) < 1:
        raise ValueError("WL visit/check/step controls must be positive")
    if max_seconds <= 0.0 or checkpoint_every_seconds <= 0.0:
        raise ValueError("WL time controls must be positive")
    if not 0.0 <= pull_move_weight <= 1.0:
        raise ValueError("pull_move_weight must lie in [0, 1]")
    if init not in {"rod", "compact"}:
        raise ValueError("init must be 'rod' or 'compact'")

    if resume_checkpoint is None:
        chain: List[Vec] = (
            compact_seed_chain(n_beads) if init == "compact"
            else [(i, 0, 0) for i in range(n_beads)]
        )
        log_g = np.zeros(m_max - m_min + 1, dtype=np.float64)
        log_f = float(initial_log_f)
        stages_completed = 0
        total_steps = 0
        accepted = 0
        previous_round_trips = 0
        one_over_t_mode = False
        one_over_t_reason = ""
        stall_relaxed = False
        stall_relaxed_step = 0
        stage_records: List[Any] = []
        historical_wall = 0.0
        saved_tier = None
        visits_since_start = np.zeros(m_max - m_min + 1, dtype=np.int64)
        visits_since_one_over_t = np.zeros(m_max - m_min + 1, dtype=np.int64)
        visits_since_stall = np.zeros(m_max - m_min + 1, dtype=np.int64)
        one_over_t_trips_before = 0
    else:
        with np.load(resume_checkpoint, allow_pickle=False) as saved:
            saved_n = int(saved["N"])
            saved_min = int(saved["m_min"])
            saved_max = int(saved["m_max"])
            if (saved_n, saved_min, saved_max) != (n_beads, m_min, m_max):
                raise ValueError(
                    "checkpoint N/contact window does not match the requested run: "
                    f"saved={(saved_n, saved_min, saved_max)}, "
                    f"requested={(n_beads, m_min, m_max)}"
                )
            chain = [tuple(map(int, row)) for row in np.asarray(saved["chain"])]
            log_g = np.asarray(saved["log_g"], dtype=np.float64).copy()
            log_f = float(saved["next_log_f"])
            stages_completed = int(saved["stages_completed"])
            total_steps = int(saved["attempted_steps"])
            accepted = int(saved["accepted_moves"])
            previous_round_trips = int(saved["round_trips"])
            one_over_t_mode = bool(saved["one_over_t_mode"]) if "one_over_t_mode" in saved else False
            one_over_t_reason = (
                str(saved["wl_one_over_t_trigger"])
                if "wl_one_over_t_trigger" in saved else ""
            )
            historical_wall = (
                float(saved["learning_wall_seconds"])
                if "learning_wall_seconds" in saved else 0.0
            )
            stall_relaxed = (
                bool(saved["wl_stall_relaxed"])
                if "wl_stall_relaxed" in saved else False
            )
            stall_relaxed_step = (
                int(saved["wl_stall_relaxed_step"])
                if "wl_stall_relaxed_step" in saved else 0
            )
            visits_since_start = _resume_histogram(
                saved, "wl_visits_since_start", log_g.size
            )
            visits_since_one_over_t = _resume_histogram(
                saved, "wl_visits_since_one_over_t", log_g.size
            )
            visits_since_stall = _resume_histogram(
                saved, "wl_visits_since_stall", log_g.size
            )
            one_over_t_trips_before = (
                int(saved["wl_one_over_t_round_trips"])
                if "wl_one_over_t_round_trips" in saved else 0
            )
            stage_records = (
                list(np.asarray(saved["wl_stage_records"], dtype=WL_STAGE_DTYPE))
                if "wl_stage_records" in saved else []
            )
            saved_tier = (
                np.asarray(saved["wl_tier"], dtype=np.int8)
                if "wl_tier" in saved else None
            )
            if "wl_schedule" in saved:
                saved_schedule = str(saved["wl_schedule"])
                if saved_schedule != schedule:
                    raise ValueError(
                        "checkpoint WL schedule does not match the requested run: "
                        f"saved={saved_schedule}, requested={schedule}"
                    )
            if "wl_pull_move_weight" in saved:
                saved_pull_weight = float(saved["wl_pull_move_weight"])
                if saved_pull_weight != pull_move_weight:
                    raise ValueError(
                        "checkpoint pull-move weight does not match the requested "
                        f"run: saved={saved_pull_weight}, "
                        f"requested={pull_move_weight}"
                    )
        validate_chain(chain)

    if tier is None:
        tier = np.full(log_g.size, TIER_FLAT, dtype=np.int8)
    tier = _validate_tier(tier, log_g.size)
    if saved_tier is not None and not np.array_equal(saved_tier, tier):
        raise ValueError("checkpoint contact tiers do not match the requested run")
    included = tier > TIER_EXCLUDED
    included_count = int(included.sum())
    included_low, included_high = _included_limits(tier, m_min)

    occupied = set(chain)
    contact = contact_count(chain, occupied)
    if not (m_min <= contact <= included_high):
        # The compact seed sits at the exact geometric maximum by construction,
        # so it lands outside the window whenever the operator has narrowed the
        # ceiling below it -- a lowered --m_max, or a declared truncation.
        detail = (
            " The compact initializer starts at the geometric maximum, which is "
            f"above the included ceiling {included_high} here; use --wl_init rod "
            "for a narrowed window."
            if init == "compact" and resume_checkpoint is None else ""
        )
        raise ValueError(
            "initial/checkpoint chain has "
            f"m={contact}, outside [{m_min}, {included_high}]." + detail
        )
    if tier[contact - m_min] == TIER_EXCLUDED:
        raise ValueError(f"initial/checkpoint chain is at excluded level m={contact}")
    rng = random.Random(seed + 1_000_003 * stages_completed)
    tracker = RoundTripCounter(included_low, included_high)
    tracker.observe(contact)
    t0 = time.time()
    last_checkpoint_time = t0
    # Round trips completed in the 1/t phase are counted from the moment it is
    # entered; a resumed 1/t run carries its earlier total in
    # ``one_over_t_trips_before``.
    entry_round_trips = tracker.round_trips
    reported_cap_reconciliation = False

    def checkpoint_now(current_log_f: float) -> None:
        if checkpoint is None:
            return
        _save_checkpoint(
            checkpoint,
            n_beads=n_beads,
            m_min=m_min,
            m_max=m_max,
            chain=chain,
            log_g=log_g,
            next_log_f=current_log_f,
            stages_completed=stages_completed,
            attempted_steps=total_steps,
            accepted_moves=accepted,
            round_trips=previous_round_trips + tracker.round_trips,
            seed=seed,
            tier=tier,
            schedule=schedule,
            pull_move_weight=pull_move_weight,
            one_over_t_mode=one_over_t_mode,
            one_over_t_trigger_reason=one_over_t_reason,
            one_over_t_round_trips=(
                one_over_t_trips_before + (tracker.round_trips - entry_round_trips)
                if one_over_t_mode else 0
            ),
            stall_relaxed=stall_relaxed,
            stall_relaxed_step=stall_relaxed_step,
            visits_since_start=visits_since_start,
            visits_since_one_over_t=visits_since_one_over_t,
            visits_since_stall=visits_since_stall,
            stage_records=np.asarray(stage_records, dtype=WL_STAGE_DTYPE),
            learning_wall_seconds=historical_wall + (time.time() - t0),
        )

    def cap_failure(
        cap_name: str,
        suggestion: str,
        histogram: np.ndarray,
        stage_steps: int,
        stage_start: float,
        highest_m: int,
        highest_first_step: int,
    ) -> RuntimeError:
        checkpoint_now(log_f)
        elapsed = time.time() - stage_start
        rate = stage_steps / elapsed if elapsed > 0.0 else float("nan")
        slow = np.flatnonzero(included)[
            np.argsort(histogram[included], kind="stable")[:5]
        ]
        slow_text = [
            {
                "m": int(index + m_min),
                "count": int(histogram[index]),
                "tier": int(tier[index]),
            }
            for index in slow
        ]
        per_invocation = time.time() - t0
        return RuntimeError(
            f"Wang-Landau learning reached {cap_name} at stage "
            f"{stages_completed + 1}, log_f={log_f:.6g}. "
            f"Slowest levels: {slow_text}. Highest m reached this stage: "
            f"{highest_m} (first reached at stage step {highest_first_step}). "
            f"Stage elapsed={elapsed:.1f}s, rate={rate:.1f} steps/s. "
            f"Elapsed cumulative={historical_wall + per_invocation:.1f}s "
            f"(this invocation={per_invocation:.1f}s, "
            f"prior invocations={historical_wall:.1f}s); "
            f"--wl_max_seconds_scope={max_seconds_scope}. "
            f"mode={_mode_label(schedule, one_over_t_mode, stall_relaxed)}. "
            f"{suggestion}"
        )

    finished = False
    while not finished and (one_over_t_mode or log_f > final_log_f):
        histogram = np.zeros_like(log_g, dtype=np.int64)
        stage_steps = 0
        stage_accepted = 0
        stage_start = time.time()
        stage_log_f = float(log_f)
        stage_trip_start = tracker.round_trips
        highest_m = int(contact)
        highest_first_step = 0
        while True:
            per_invocation_elapsed = time.time() - t0
            # A resumed run keeps spending the same budget it started with.
            # Without the historical term every resume restarts the clock and
            # total learning time is unbounded.
            elapsed_total = (
                historical_wall + per_invocation_elapsed
                if max_seconds_scope == "cumulative" else per_invocation_elapsed
            )
            if elapsed_total >= max_seconds:
                raise cap_failure(
                    "--wl_max_seconds",
                    "Increase --wl_max_seconds or resume from the checkpoint.",
                    histogram, stage_steps, stage_start, highest_m,
                    highest_first_step,
                )
            overall_remaining = max_steps - total_steps
            # The 1/t phase is continuous, so there are no stages for
            # --wl_max_steps_per_stage to bound.
            stage_remaining = (
                overall_remaining if one_over_t_mode
                else max_steps_per_stage - stage_steps
            )
            block = min(check_every, overall_remaining, stage_remaining)
            if block <= 0:
                if overall_remaining <= 0:
                    raise cap_failure(
                        "--wl_max_steps",
                        "Increase --wl_max_steps or resume from the checkpoint.",
                        histogram, stage_steps, stage_start, highest_m,
                        highest_first_step,
                    )
                raise cap_failure(
                    "--wl_max_steps_per_stage",
                    "Increase --wl_max_steps_per_stage after checking the slow levels.",
                    histogram, stage_steps, stage_start, highest_m,
                    highest_first_step,
                )
            for _ in range(block):
                chain, occupied, contact, _, was_accepted = metropolis_step(
                    chain, occupied, contact, log_g, m_min, included_high, rng,
                    pull_move_weight,
                )
                total_steps += 1
                stage_steps += 1
                if was_accepted:
                    accepted += 1
                    stage_accepted += 1
                index = contact - m_min
                if tier[index] == TIER_EXCLUDED:
                    raise RuntimeError(
                        f"encountered excluded contact level m={contact}; remove it "
                        "from --excluded_contact_levels and restart learning"
                    )
                if contact > highest_m:
                    highest_m = int(contact)
                    highest_first_step = int(stage_steps)
                if one_over_t_mode:
                    # A running minimum, not a plain assignment: 1/t is
                    # recomputed every step, so assigning it directly would let
                    # log_f climb back up whenever 1/t still exceeds it.
                    log_f = min(log_f, included_count / float(total_steps))
                    visits_since_one_over_t[index] += 1
                if stall_relaxed:
                    visits_since_stall[index] += 1
                log_g[index] += log_f
                histogram[index] += 1
                visits_since_start[index] += 1
                tracker.observe(contact)

            now = time.time()
            if checkpoint is not None and now - last_checkpoint_time >= checkpoint_every_seconds:
                checkpoint_now(log_f)
                last_checkpoint_time = now

            if not reported_cap_reconciliation:
                reported_cap_reconciliation = True
                _report_cap_reconciliation(
                    stage_steps=stage_steps,
                    elapsed=now - stage_start,
                    max_steps=max_steps,
                    max_seconds=max_seconds,
                    elapsed_total=elapsed_total,
                    progress=progress,
                )

            ratio, min_flat, min_coverage, range_covered = _stage_visit_statistics(
                histogram, tier
            )
            coverage_ok = _coverage_satisfied(
                range_covered, min_flat, min_coverage, tier,
                min_visits, min_cover_visits,
            )
            stage_complete = coverage_ok and (
                schedule == "one_over_t" or ratio >= flatness
            )
            if progress:
                # Pull moves cost an extra catalog build per proposal, so the
                # achieved rate is printed to keep that cost visible rather than
                # buried in the wall-clock cap.
                stage_elapsed = now - stage_start
                stage_rate = stage_steps / stage_elapsed if stage_elapsed > 0 else 0.0
                print(
                    f"[WL stage {stages_completed + 1}] log_f={log_f:.6g} "
                    f"steps={stage_steps} min/mean={ratio:.3f} "
                    f"min_tier2={min_flat} min_tier1={min_coverage} "
                    f"range={range_covered} highest_m={highest_m} "
                    f"round_trips={previous_round_trips + tracker.round_trips} "
                    f"steps_per_s={stage_rate:.0f} "
                    f"mode={_mode_label(schedule, one_over_t_mode, stall_relaxed)}",
                    flush=True,
                )

            # The switch is evaluated on every check block, not only when a
            # stage completes.  A stage that never completes is precisely the
            # stall the 1/t schedule exists to escape, so gating the switch on
            # stage completion made the option unreachable.
            if schedule == "one_over_t" and not one_over_t_mode:
                inverse_time = included_count / float(total_steps)
                # The stall can only fire once; after it does, its relaxation is
                # already in force.  Suppressing it by zeroing the step count --
                # rather than by discarding the trigger's answer -- keeps the
                # Belardinelli-Pereyra crossing armed, which the trigger would
                # otherwise never report while the stall condition also holds.
                fire, reason = one_over_t_trigger(
                    log_f,
                    inverse_time,
                    0 if stall_relaxed else stage_steps,
                    stage_stall_steps,
                )
                uncovered = (
                    np.flatnonzero(included & (histogram == 0)) + m_min
                ).tolist()
                if fire and reason == "belardinelli_pereyra":
                    # Rate-neutral by construction: log_f and 1/t are equal at
                    # the crossing, so adopting 1/t changes nothing abruptly.
                    one_over_t_mode = True
                    one_over_t_reason = reason
                    log_f = min(log_f, inverse_time)
                    # Coverage restarts here: the frozen bias must be exercised
                    # over the whole window at its final resolution, not merely
                    # at some coarser log_f earlier in the run.
                    visits_since_one_over_t[:] = 0
                    entry_round_trips = tracker.round_trips
                    if progress:
                        print(
                            f"[WL] entering 1/t mode via {reason} trigger at stage "
                            f"{stages_completed + 1}: stage_steps={stage_steps}, "
                            f"log_f={log_f:.6g}, 1/t={inverse_time:.6g}, "
                            f"levels still uncovered this stage: {uncovered}",
                            flush=True,
                        )
                    break
                if fire and reason == "stall":
                    # A stall carries no rate-neutrality guarantee.  When it
                    # fires, 1/t is typically orders of magnitude below log_f --
                    # measured at N=44, a stall at 3M steps gives 1/t = 1.7e-5
                    # against log_f = 1.0, already below final_log_f -- so
                    # adopting it would end learning after 3M steps of
                    # adaptation and freeze a bias the chain had barely begun to
                    # build.  Keep log_f, keep halving; relax only how a stage
                    # is permitted to advance.
                    stall_relaxed = True
                    stall_relaxed_step = total_steps
                    visits_since_stall[:] = 0
                    if progress:
                        print(
                            f"[WL] stall relaxation engaged at stage "
                            f"{stages_completed + 1}: stage_steps={stage_steps}, "
                            f"log_f={log_f:.6g} (unchanged; 1/t={inverse_time:.6g} "
                            "is NOT adopted), halving continues, coverage now "
                            "judged cumulatively from this step. Levels still "
                            f"uncovered this stage: {uncovered}",
                            flush=True,
                        )
                    # Deliberately no break: log_f is unchanged, so the stage
                    # continues and the relaxed criterion is applied below.

            if one_over_t_mode:
                log_f = min(log_f, included_count / float(total_steps))
                trips_since_entry = one_over_t_trips_before + (
                    tracker.round_trips - entry_round_trips
                )
                (
                    ratio, min_flat, min_coverage, range_covered
                ) = _stage_visit_statistics(visits_since_one_over_t, tier)
                coverage_ok = _coverage_satisfied(
                    range_covered, min_flat, min_coverage, tier,
                    min_visits, min_cover_visits,
                )
                if log_f > final_log_f or not coverage_ok or trips_since_entry < 1:
                    continue
                stage_records.append(
                    (
                        stage_log_f,
                        stage_steps,
                        time.time() - stage_start,
                        ratio,
                        min_flat,
                        min_coverage,
                        stage_accepted,
                        tracker.round_trips - stage_trip_start,
                        highest_m,
                        _slowest_levels(histogram, tier, m_min),
                    )
                )
                stages_completed += 1
                log_g -= log_g[0]
                finished = True
                checkpoint_now(log_f)
                break

            # Stall relaxation widens stage advancement; it never narrows it, so
            # a stage that satisfies the ordinary criterion still advances at
            # once.  The relaxed path costs a full stall budget per stage, and
            # its coverage counts -- unlike its flatness ratio -- are the
            # unrelaxed ones, merely accumulated since the stall rather than
            # since the start of the stage.  A level the chain genuinely cannot
            # reach therefore still blocks every stage and still fails loudly.
            if stall_relaxed and not stage_complete and stage_steps >= stage_stall_steps:
                (
                    _, stall_min_flat, stall_min_coverage, stall_range_covered
                ) = _stage_visit_statistics(visits_since_stall, tier)
                stage_complete = _coverage_satisfied(
                    stall_range_covered, stall_min_flat, stall_min_coverage,
                    tier, min_visits, min_cover_visits,
                ) and _visited_flatness_ratio(histogram, tier) >= flatness

            if not stage_complete:
                continue
            stage_records.append(
                (
                    stage_log_f,
                    stage_steps,
                    time.time() - stage_start,
                    ratio,
                    min_flat,
                    min_coverage,
                    stage_accepted,
                    tracker.round_trips - stage_trip_start,
                    highest_m,
                    _slowest_levels(histogram, tier, m_min),
                )
            )
            stages_completed += 1
            log_g -= log_g[0]
            log_f = 0.5 * log_f
            checkpoint_now(log_f)
            last_checkpoint_time = time.time()
            break

    log_g -= log_g[0]
    spread, spread_reference = _report_log_g_spread(
        log_g, included, n_beads, progress
    )
    learning_wall = historical_wall + (time.time() - t0)
    return {
        "log_g": log_g,
        "chain": chain,
        "attempted_steps": total_steps,
        "accepted_moves": accepted,
        "stages_completed": stages_completed,
        "stage_records": np.asarray(stage_records, dtype=WL_STAGE_DTYPE),
        "round_trips": previous_round_trips + tracker.round_trips,
        "wall_time": learning_wall,
        # Cumulative across resumes, as both terms are, so the ratio stays a fair
        # average rather than a snapshot of the final invocation.
        "steps_per_second": (total_steps / learning_wall) if learning_wall > 0 else 0.0,
        "pull_move_weight": float(pull_move_weight),
        "next_log_f": log_f,
        "tier": tier,
        "active": included,
        "one_over_t_mode": one_over_t_mode,
        "one_over_t_trigger": one_over_t_reason,
        "one_over_t_round_trips": (
            one_over_t_trips_before + (tracker.round_trips - entry_round_trips)
            if one_over_t_mode else 0
        ),
        "stall_relaxed": stall_relaxed,
        "stall_relaxed_step": stall_relaxed_step,
        "visits_since_start": visits_since_start,
        "visits_since_one_over_t": visits_since_one_over_t,
        "visits_since_stall": visits_since_stall,
        "log_g_spread": spread,
        "log_g_spread_reference": spread_reference,
    }


def run_production_chain(
    worker_id: int,
    seed: int,
    initial_chain: Sequence[Vec],
    log_g: np.ndarray,
    m_min: int,
    m_max: int,
    steps: int,
    burnin: float,
    sample_every: int,
    progress: bool = True,
    tier: Optional[np.ndarray] = None,
    pull_move_weight: float = 0.0,
    checkpoint_path: Optional[Path] = None,
    checkpoint_every_seconds: float = 1800.0,
    resume_path: Optional[Path] = None,
) -> Dict[str, Any]:
    if checkpoint_every_seconds <= 0.0:
        raise ValueError("checkpoint_every_seconds must be positive")
    if tier is None:
        tier = np.full(log_g.size, TIER_FLAT, dtype=np.int8)
    if log_g.shape != (m_max - m_min + 1,):
        raise ValueError("log_g shape does not match [m_min, m_max]")
    tier = _validate_tier(tier, m_max - m_min + 1)
    included_low, included_high = _included_limits(tier, m_min)

    n_beads = len(initial_chain)
    rng = random.Random(seed)
    chain = [tuple(site) for site in initial_chain]
    steps_done = 0
    accepted = 0
    geometrically_valid = 0
    contacts: List[int] = []
    radii: List[float] = []
    bends: List[int] = []
    coordination_histograms: List[np.ndarray] = []
    tracker = RoundTripCounter(included_low, included_high)

    if resume_path is not None:
        state = _load_production_checkpoint(
            resume_path,
            n_beads=n_beads,
            m_min=m_min,
            m_max=m_max,
            log_g=log_g,
            tier=tier,
        )
        chain = state["chain"]
        steps_done = state["steps_done"]
        accepted = state["accepted"]
        geometrically_valid = state["geometrically_valid"]
        contacts = state["contacts"]
        radii = state["radii"]
        bends = state["bends"]
        coordination_histograms = state["coordination_histograms"]
        tracker.phase = state["round_trip_phase"]
        tracker.round_trips = state["round_trips"]
        rng.setstate(state["rng_state"])

    occupied = set(chain)
    contact = contact_count(chain, occupied)
    if tier[contact - m_min] == TIER_EXCLUDED:
        raise RuntimeError(
            f"production worker {worker_id} started in excluded contact level "
            f"m={contact}"
        )
    # Burn-in is measured against the total step budget, so a resumed worker
    # does not re-burn and discard samples it has already earned.
    burn_steps = int(round(burnin * steps))
    t0 = time.time()
    last_checkpoint_time = t0
    progress_mark = max(1, steps // 10)

    def checkpoint_now(step: int) -> None:
        if checkpoint_path is None:
            return
        _save_production_checkpoint(
            checkpoint_path,
            worker_id=worker_id,
            n_beads=n_beads,
            m_min=m_min,
            m_max=m_max,
            log_g=log_g,
            tier=tier,
            chain=chain,
            contact=contact,
            steps_done=step,
            accepted=accepted,
            geometrically_valid=geometrically_valid,
            contacts=contacts,
            radii=radii,
            bends=bends,
            tracker=tracker,
            rng_state=rng.getstate(),
            coordination_histograms=coordination_histograms,
        )

    for step in range(steps_done + 1, steps + 1):
        chain, occupied, contact, valid, was_accepted = metropolis_step(
            chain, occupied, contact, log_g, m_min, included_high, rng,
            pull_move_weight,
        )
        if tier[contact - m_min] == TIER_EXCLUDED:
            raise RuntimeError(
                f"production worker {worker_id} encountered excluded contact level "
                f"m={contact}; remove it from --excluded_contact_levels"
            )
        if valid:
            geometrically_valid += 1
        if was_accepted:
            accepted += 1
        if step > burn_steps:
            tracker.observe(contact)
            if (step - burn_steps) % sample_every == 0:
                contacts.append(contact)
                radii.append(radius_of_gyration(chain))
                bends.append(count_bends(chain))
                histogram = coordination_histogram(chain, occupied)
                if int(np.dot(np.arange(7), histogram)) != 2 * contact:
                    raise RuntimeError(
                        "coordination histogram disagrees with production contact count"
                    )
                coordination_histograms.append(histogram)
        if progress and step % progress_mark == 0:
            print(
                f"[production worker {worker_id} seed={seed}] "
                f"{100.0 * step / steps:5.1f}% ({step}/{steps}) "
                f"accepted={accepted} samples={len(contacts)} "
                f"round_trips={tracker.round_trips}",
                flush=True,
            )
        if (
            checkpoint_path is not None
            and time.time() - last_checkpoint_time >= checkpoint_every_seconds
        ):
            checkpoint_now(step)
            last_checkpoint_time = time.time()
    checkpoint_now(steps)

    return {
        "worker_id": worker_id,
        "seed": seed,
        "contact_samples": np.asarray(contacts, dtype=np.int64),
        "rg_samples": np.asarray(radii, dtype=np.float64),
        "bend_samples": np.asarray(bends, dtype=np.int64),
        "coordination_histogram_samples": np.asarray(
            coordination_histograms, dtype=np.int64
        ).reshape((-1, 7)),
        "accepted_moves": accepted,
        "geometrically_valid_moves": geometrically_valid,
        "attempted_moves": steps,
        "round_trips": tracker.round_trips,
        "wall_time": time.time() - t0,
    }


def normalized_importance_weights(contacts: np.ndarray, log_g: np.ndarray, m_min: int) -> np.ndarray:
    log_weights = log_g[contacts - m_min]
    shift = float(np.max(log_weights))
    weights = np.exp(log_weights - shift)
    total = float(weights.sum())
    if not math.isfinite(total) or total <= 0.0:
        raise RuntimeError("importance weights are non-finite or sum to zero")
    return weights / total


def effective_sample_size(normalized_weights: np.ndarray) -> float:
    return float(1.0 / np.square(normalized_weights).sum())


def systematic_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    count = weights.size
    positions = (rng.random() + np.arange(count)) / count
    cumulative = np.cumsum(weights)
    cumulative[-1] = 1.0
    return np.searchsorted(cumulative, positions, side="right")


def blocked_contact_stderr(
    worker_contacts: Sequence[np.ndarray],
    log_g: np.ndarray,
    m_min: int,
    m_cover: int,
    n_blocks: int,
) -> Tuple[np.ndarray, int]:
    """Estimate the standard error of P0(m) from per-worker batch means.

    Each non-overlapping block is reweighted and self-normalized independently.
    Blocks are pooled across independent workers, and their between-block sample
    variance is divided by the number of blocks.  This accounts for within-chain
    autocorrelation when blocks are long relative to the correlation time; it
    does not include uncertainty in the learned bias itself.
    """
    if n_blocks < 2:
        raise ValueError("n_blocks must be at least 2")
    n_levels = m_cover - m_min + 1
    estimates: List[np.ndarray] = []
    for worker_id, values in enumerate(worker_contacts):
        contacts = np.asarray(values, dtype=np.int64)
        if contacts.size < 2:
            raise ValueError(
                f"production worker {worker_id} has fewer than two recorded samples"
            )
        if np.any((contacts < m_min) | (contacts > m_cover)):
            raise ValueError(
                f"production worker {worker_id} has contacts outside the declared window"
            )
        for block in np.array_split(contacts, min(n_blocks, contacts.size)):
            weights = normalized_importance_weights(block, log_g, m_min)
            estimates.append(
                np.bincount(
                    block - m_min, weights=weights, minlength=n_levels
                )[:n_levels]
            )
    batches = np.asarray(estimates, dtype=np.float64)
    if batches.shape[0] < 2:
        raise ValueError("batch-means error estimation requires at least two blocks")
    stderr = batches.std(axis=0, ddof=1) / math.sqrt(batches.shape[0])
    return stderr, int(batches.shape[0])


LOCAL_COORD_SCHEMA_VERSION = 1


def build_local_coordination_statistics(
    coordination_histograms: np.ndarray,
    contacts: np.ndarray,
    radii: np.ndarray,
    weights: np.ndarray,
    c_prob: np.ndarray,
    m_min: int,
    n_beads: int,
    rg_edges: np.ndarray,
    include_rg_joint: bool,
) -> Dict[str, Any]:
    """Compress production samples into exact local-coordination fit statistics."""
    hist_raw = np.asarray(coordination_histograms)
    if hist_raw.shape != (contacts.size, 7):
        raise RuntimeError(
            "coordination histogram samples must have shape (n_samples, 7)"
        )
    if not np.all(np.isfinite(hist_raw)) or not np.all(hist_raw == np.rint(hist_raw)):
        raise RuntimeError("coordination histogram samples must be finite integers")
    hist = np.rint(hist_raw).astype(np.int64)
    if np.any(hist < 0) or np.any(hist.sum(axis=1) != int(n_beads)):
        raise RuntimeError(
            "coordination histogram samples must be nonnegative and sum to N"
        )
    degree_sum = hist @ np.arange(7, dtype=np.int64)
    if np.any(degree_sum % 2) or not np.array_equal(degree_sum, 2 * contacts):
        raise RuntimeError(
            "coordination histograms violate sum_k k*h_k = 2*m"
        )
    unique_hist, state_index = np.unique(hist, axis=0, return_inverse=True)
    state_mass = np.bincount(
        state_index, weights=weights, minlength=unique_hist.shape[0]
    ).astype(np.float64)
    state_mass /= state_mass.sum()
    state_contacts = (
        unique_hist @ np.arange(7, dtype=np.int64)
    ) // 2
    contact_marginal = np.bincount(
        state_contacts - m_min, weights=state_mass, minlength=c_prob.size
    )[:c_prob.size]
    contact_error = float(np.max(np.abs(contact_marginal - c_prob)))
    if contact_error > 1e-12:
        raise RuntimeError(
            "local coordination state masses do not reproduce P0(m): "
            f"maximum error {contact_error:.3e}"
        )
    out: Dict[str, Any] = {
        "local_coord_schema_version": np.array(
            LOCAL_COORD_SCHEMA_VERSION, dtype=np.int64
        ),
        "local_coord_degree_values": np.arange(7, dtype=np.int64),
        "local_coord_histograms": unique_hist,
        "local_coord_contact_counts": state_contacts.astype(np.int64),
        "local_coord_state_mass": state_mass,
        "local_coord_contact_marginal_error": contact_error,
    }
    if include_rg_joint:
        rg_edges = np.asarray(rg_edges, dtype=np.float64)
        n_rg = rg_edges.size - 1
        rg_index = np.searchsorted(rg_edges, radii, side="right") - 1
        # numpy.histogram includes a value exactly equal to the last edge in the
        # final bin; mirror that convention in the sparse representation.
        rg_index[rg_index == n_rg] = n_rg - 1
        if np.any(rg_index < 0) or np.any(rg_index >= n_rg):
            raise RuntimeError("Rg sample fell outside the sparse local joint grid")
        pair_code = state_index.astype(np.int64) * n_rg + rg_index.astype(np.int64)
        unique_code, pair_inverse = np.unique(pair_code, return_inverse=True)
        joint_mass = np.bincount(
            pair_inverse, weights=weights, minlength=unique_code.size
        ).astype(np.float64)
        joint_mass /= joint_mass.sum()
        rg_state_index = unique_code // n_rg
        rg_bin_index = unique_code % n_rg
        state_marginal = np.bincount(
            rg_state_index, weights=joint_mass, minlength=unique_hist.shape[0]
        )
        rg_marginal = np.bincount(
            rg_bin_index, weights=joint_mass, minlength=n_rg
        )
        expected_rg, _ = np.histogram(radii, bins=rg_edges, weights=weights)
        expected_rg = expected_rg.astype(np.float64)
        expected_rg /= expected_rg.sum()
        state_error = float(np.max(np.abs(state_marginal - state_mass)))
        rg_error = float(np.max(np.abs(rg_marginal - expected_rg)))
        if state_error > 1e-12 or rg_error > 1e-12:
            raise RuntimeError(
                "sparse local coordination/Rg joint lost marginal mass: "
                f"state error={state_error:.3e}, Rg error={rg_error:.3e}"
            )
        out.update(
            local_coord_rg_state_index=rg_state_index.astype(np.int64),
            local_coord_rg_bin_index=rg_bin_index.astype(np.int64),
            local_coord_rg_joint_mass=joint_mass,
            local_coord_state_marginal_error=state_error,
            local_coord_rg_marginal_error=rg_error,
        )
    return out


def build_distributions(
    contacts: np.ndarray,
    radii: np.ndarray,
    bends: np.ndarray,
    log_g: np.ndarray,
    m_min: int,
    rg_bins: int,
    no_joint: bool,
    *,
    n_beads: Optional[int] = None,
    m_cover: Optional[int] = None,
    rg_edges: Optional[np.ndarray] = None,
    grid_source: Optional[str] = None,
    grid_search_dir: Optional[Path] = None,
    coordination_histograms: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    if contacts.size == 0 or contacts.size != radii.size or contacts.size != bends.size:
        raise RuntimeError("production sample arrays are empty or have inconsistent lengths")
    if not np.all(np.isfinite(radii)):
        raise RuntimeError("non-finite Rg production samples")
    if m_cover is None:
        m_cover = int(contacts.max())
    if m_cover < m_min:
        raise ValueError("m_cover must be at least m_min")
    weights = normalized_importance_weights(contacts, log_g, m_min)
    c_edges = fixed_c_edges(m_min, m_cover)
    c_vals = np.arange(m_min, m_cover + 1, dtype=np.int64)
    assert_within_grid(contacts, c_edges, "contact")
    c_prob = np.bincount(
        contacts - m_min, weights=weights, minlength=c_vals.size
    )[:c_vals.size].astype(np.float64)
    production_c_counts = np.bincount(
        contacts - m_min, minlength=c_vals.size
    )[:c_vals.size].astype(np.int64)
    if not math.isclose(float(c_prob.sum()), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError("contact histogram lost probability mass")
    c_prob /= c_prob.sum()
    # Raw visits from the biased fixed-weight production run.  These are not
    # athermal counts, but because every sample in one contact level carries the
    # same importance weight, they are what sets the statistical error of
    # P0(m) and of P0(Rg | m) in that level.  n_samples * c_prob does not:
    # in the compact tail it is orders of magnitude smaller than the number of
    # samples actually backing the bin, which would report the best-sampled
    # levels as the worst.
    # Superseded by c_blocked_stderr for error estimation.  This remains only a
    # raw coverage indicator and ignores autocorrelation and reweighting.
    c_naive_count_error = np.full(c_vals.size, np.inf, dtype=np.float64)
    visited = production_c_counts > 0
    c_naive_count_error[visited] = 1.0 / np.sqrt(
        production_c_counts[visited].astype(np.float64)
    )

    legacy_edges = np.array([], dtype=np.float64)
    legacy_source = ""
    search_dir = Path(__file__).resolve().parent if grid_search_dir is None else grid_search_dir
    if rg_edges is None and n_beads in LEGACY_BASELINE_FILES:
        legacy_edges, width, _ = legacy_rg_grid(int(n_beads), search_dir)
        rg_edges = fixed_rg_edges(int(n_beads), search_dir)
        resolved_grid_source = "legacy_extended"
        legacy_source = LEGACY_BASELINE_FILES[int(n_beads)]
    elif rg_edges is None:
        if n_beads is None:
            raise ValueError("n_beads is required when rg_edges is not supplied")
        rg_min, rg_max = float(radii.min()), float(radii.max())
        pad = 1e-9 if rg_max <= rg_min else 0.02 * (rg_max - rg_min)
        upper = max(rg_max + pad, rod_rg(int(n_beads)))
        rg_edges = np.linspace(rg_min - pad, upper, rg_bins + 1)
        width = float(np.median(np.diff(rg_edges)))
        resolved_grid_source = "adaptive"
        warnings.warn(
            f"N={n_beads} has no registered legacy Rg grid; using a run-dependent "
            "adaptive grid padded to the rod maximum",
            RuntimeWarning,
            stacklevel=2,
        )
    else:
        rg_edges = np.asarray(rg_edges, dtype=np.float64)
        width = float(np.median(np.diff(rg_edges)))
        resolved_grid_source = grid_source or "provided"
    rg_edges = np.asarray(rg_edges, dtype=np.float64)
    rg_out_of_range_count = assert_within_grid(radii, rg_edges, "Rg")
    rg_prob, _ = np.histogram(radii, bins=rg_edges, weights=weights)
    rg_prob = rg_prob.astype(np.float64)
    if not math.isclose(float(rg_prob.sum()), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError("Rg histogram lost probability mass")
    rg_prob /= rg_prob.sum()

    if no_joint:
        crg_prob = np.array([[]], dtype=np.float64)
    else:
        assert_within_grid(contacts, c_edges, "joint contact")
        assert_within_grid(radii, rg_edges, "joint Rg")
        crg_prob, _, _ = np.histogram2d(
            contacts.astype(np.float64), radii, bins=(c_edges, rg_edges), weights=weights
        )
        crg_prob = crg_prob.astype(np.float64)
        if not math.isclose(float(crg_prob.sum()), 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError("joint histogram lost probability mass")
        crg_prob /= crg_prob.sum()

    bend_vals = np.unique(bends)
    bend_mass = np.array([weights[bends == value].sum() for value in bend_vals])
    bend_mass /= bend_mass.sum()
    mean_bends = float(np.dot(weights, bends))

    if no_joint:
        marginal_m_error = float("nan")
        marginal_rg_error = float("nan")
    else:
        marginal_m_error = float(np.max(np.abs(crg_prob.sum(axis=1) - c_prob)))
        marginal_rg_error = float(np.max(np.abs(crg_prob.sum(axis=0) - rg_prob)))

    result = {
        "weights": weights,
        "c_vals": c_vals,
        "c_prob": c_prob,
        "production_c_counts": production_c_counts,
        "c_naive_count_error": c_naive_count_error,
        "c_edges": c_edges,
        "rg_edges": rg_edges,
        "rg_grid_source": resolved_grid_source,
        "legacy_rg_edges": legacy_edges,
        "legacy_grid_source": legacy_source,
        "rg_grid_width": width,
        "rg_out_of_range_count": rg_out_of_range_count,
        "rg_prob": rg_prob,
        "crg_prob": crg_prob,
        "bend_vals": bend_vals,
        "bend_prob": bend_mass,
        "mean_bends": mean_bends,
        "marginal_m_error": marginal_m_error,
        "marginal_rg_error": marginal_rg_error,
    }
    if coordination_histograms is not None:
        if n_beads is None:
            raise ValueError(
                "n_beads is required when coordination_histograms are supplied"
            )
        result.update(build_local_coordination_statistics(
            coordination_histograms,
            contacts,
            radii,
            weights,
            c_prob,
            m_min,
            int(n_beads),
            rg_edges,
            include_rg_joint=not no_joint,
        ))
    return result


def enumerate_rooted_saws(n_beads: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Enumerate short rooted SAWs for exact validation; not used in production."""
    chain: List[Vec] = [(0, 0, 0)]
    occupied: Set[Vec] = {chain[0]}
    contacts: List[int] = []
    radii: List[float] = []

    def visit() -> None:
        if len(chain) == n_beads:
            contacts.append(contact_count(chain, occupied))
            radii.append(radius_of_gyration(chain))
            return
        for direction in NN_VECS:
            site = add(chain[-1], direction)
            if site in occupied:
                continue
            chain.append(site)
            occupied.add(site)
            visit()
            occupied.remove(site)
            chain.pop()

    visit()
    c = np.asarray(contacts, dtype=np.int64)
    r = np.asarray(radii, dtype=np.float64)
    values, counts = np.unique(c, return_counts=True)
    return values, counts / counts.sum(), r


def _probe_sweep(
    *,
    chain: List[Vec],
    occupied: Set[Vec],
    contact: int,
    log_g: np.ndarray,
    m_min: int,
    m_max: int,
    tier: np.ndarray,
    rng: random.Random,
    steps: int,
    pull_move_weight: float,
    log_f: float,
    flatness: float,
    min_visits: int,
    min_cover_visits: int,
    check_every: int,
) -> Dict[str, Any]:
    """Run a fixed-``log_f`` Wang-Landau sweep and record when levels are first hit.

    This deliberately does not go through ``learn_log_density``: the probe needs
    per-level first-hit steps and a frozen refinement factor, and neither is a
    concern of the production learner.  The stage logic, schedules, triggers and
    budgets there are untouched.
    """
    log_g = np.asarray(log_g, dtype=np.float64).copy()
    included = tier > TIER_EXCLUDED
    histogram = np.zeros(log_g.size, dtype=np.int64)
    first_hit = np.full(log_g.size, -1, dtype=np.int64)
    highest_m = contact
    covered_step = -1
    stage_step = -1
    step = 0
    start = time.time()
    while step < steps:
        for _ in range(min(check_every, steps - step)):
            chain, occupied, contact, _, _ = metropolis_step(
                chain, occupied, contact, log_g, m_min, m_max, rng, pull_move_weight
            )
            step += 1
            index = contact - m_min
            log_g[index] += log_f
            histogram[index] += 1
            if first_hit[index] < 0:
                first_hit[index] = step
            if contact > highest_m:
                highest_m = contact
            if covered_step < 0 and not np.any(included & (histogram == 0)):
                covered_step = step
        # Touching every level once is a far weaker bar than a stage actually
        # has to clear, and it is not the bar that fails: a starving stage has
        # visited the top of the window and still cannot accumulate the minimum
        # counts there.  So the stage criterion itself is the headline number.
        ratio, min_flat, min_coverage, range_covered = _stage_visit_statistics(
            histogram, tier
        )
        if stage_step < 0 and _coverage_satisfied(
            range_covered, min_flat, min_coverage, tier, min_visits, min_cover_visits
        ) and ratio >= flatness:
            stage_step = step
            break
    elapsed = time.time() - start
    return {
        "first_hit": first_hit,
        "histogram": histogram,
        "highest_m": highest_m,
        "covered_step": covered_step,
        "stage_step": stage_step,
        "min_visits_in_window": int(histogram[included].min()) if included.any() else 0,
        "steps_run": step,
        "wall_seconds": elapsed,
        "steps_per_second": (step / elapsed) if elapsed > 0 else 0.0,
    }


def _probe_warm_up(
    *,
    n_beads: int,
    m_min: int,
    m_max: int,
    tier: np.ndarray,
    rng: random.Random,
    flatness: float,
    min_visits: int,
    min_cover_visits: int,
    max_steps: int,
    check_every: int,
) -> Dict[str, Any]:
    """Drive a fixed-``log_f`` chain until real stage-1 completion criteria hold.

    Terminating on first coverage would be the wrong state: coverage happens far
    earlier and leaves ``log_g`` only mildly inflated, which is not the state
    that starves.  The criterion here is the one ``learn_log_density`` itself
    uses to complete a stage -- the tier-2 flatness ratio together with the
    per-tier minimum visit counts -- reusing ``_stage_visit_statistics`` and
    ``_coverage_satisfied`` rather than restating it.
    """
    chain: List[Vec] = [(i, 0, 0) for i in range(n_beads)]
    occupied = set(chain)
    contact = contact_count(chain, occupied)
    log_g = np.zeros(m_max - m_min + 1, dtype=np.float64)
    histogram = np.zeros(log_g.size, dtype=np.int64)
    steps = 0
    start = time.time()
    while steps < max_steps:
        for _ in range(min(check_every, max_steps - steps)):
            chain, occupied, contact, _, _ = metropolis_step(
                chain, occupied, contact, log_g, m_min, m_max, rng, 0.0
            )
            index = contact - m_min
            log_g[index] += 1.0
            histogram[index] += 1
            steps += 1
        ratio, min_flat, min_coverage, range_covered = _stage_visit_statistics(
            histogram, tier
        )
        complete = _coverage_satisfied(
            range_covered, min_flat, min_coverage, tier, min_visits, min_cover_visits
        ) and ratio >= flatness
        print(
            f"[probe warm-up] steps={steps} min/mean={ratio:.3f} "
            f"min_tier2={min_flat} min_tier1={min_coverage} "
            f"range={range_covered} rate={steps / max(time.time() - start, 1e-9):.0f}/s",
            flush=True,
        )
        if complete:
            break
    return {
        "chain": chain,
        "occupied": occupied,
        "contact": contact,
        "log_g": log_g,
        "steps": steps,
        "completed": bool(complete),
    }


def run_pull_move_probe(args: argparse.Namespace) -> int:
    """Measure whether pull moves restore re-reachability of the compact window.

    Reaching the top of the window once is not the question -- the existing
    sampler already does that in its first stages.  The question is whether the
    window can be swept *again* after ``log_g`` has been built up, which is where
    an N=44 run starves.  So the probe first drives a chain to a genuine
    stage-1 completion, then replays the same warmed state twice with and
    without pull moves.
    """
    validate_common_args(args)
    resolve_m_max(args)
    m_min, m_max = int(args.m_min), int(args.m_max)
    tier = np.full(m_max - m_min + 1, TIER_FLAT, dtype=np.int8)
    print(
        f"=== pull-move probe: N={args.N} window [{m_min}, {m_max}] "
        f"weight={args.pull_move_weight} ===",
        flush=True,
    )

    rng = random.Random(args.wl_seed)
    if args.pull_move_probe_log_g:
        source = Path(args.pull_move_probe_log_g)
        with np.load(source, allow_pickle=False) as saved:
            if int(saved["N"]) != args.N:
                raise ValueError(
                    f"checkpoint N={int(saved['N'])} does not match --N={args.N}"
                )
            log_g = np.asarray(saved["log_g"], dtype=np.float64).copy()
            chain = [tuple(map(int, row)) for row in np.asarray(saved["chain"])]
        validate_chain(chain)
        occupied = set(chain)
        contact = contact_count(chain, occupied)
        warm_steps = 0
        print(f"loaded built-up log_g from {source}", flush=True)
    else:
        warm = _probe_warm_up(
            n_beads=args.N, m_min=m_min, m_max=m_max, tier=tier, rng=rng,
            flatness=args.wl_flatness, min_visits=args.wl_min_visits,
            min_cover_visits=args.wl_min_cover_visits,
            max_steps=args.pull_move_probe_steps, check_every=args.wl_check_every,
        )
        if not warm["completed"]:
            print(
                "WARNING: warm-up hit its step budget before satisfying the "
                "stage-1 criterion, so log_g is less built up than a real "
                "stage-1 exit. Raise --pull_move_probe_steps.",
                flush=True,
            )
        chain, occupied = warm["chain"], warm["occupied"]
        contact, log_g = warm["contact"], warm["log_g"]
        warm_steps = warm["steps"]

    spread, reference = _report_log_g_spread(
        log_g, tier > TIER_EXCLUDED, int(args.N), True
    )
    if spread < 10.0 * reference:
        print(
            "\n*** WARNING: the probe has NOT reproduced the failure state. ***\n"
            f"    log_g spread {spread:.6g} is below ten times the analytic "
            f"scale {reference:.6g}. The starving N=44 stage-3 state has a "
            "spread far above that; a bias this mild is not the regime pull "
            "moves are meant to rescue, so whatever the arms below report is "
            "NOT evidence either way. Warm up longer before drawing a "
            "conclusion.\n",
            flush=True,
        )

    # Both arms replay one identical warmed state, including the random stream.
    # Paired replay is far more sensitive than two independent runs, and the
    # warm-up is paid once.
    snapshot_state = rng.getstate()
    arms: Dict[float, Dict[str, Any]] = {}
    for weight in (0.0, float(args.pull_move_weight)):
        arm_rng = random.Random()
        arm_rng.setstate(snapshot_state)
        arms[weight] = _probe_sweep(
            chain=list(chain), occupied=set(occupied), contact=contact,
            log_g=log_g, m_min=m_min, m_max=m_max, tier=tier, rng=arm_rng,
            steps=args.pull_move_probe_steps, pull_move_weight=weight,
            log_f=1.0, flatness=args.wl_flatness,
            min_visits=args.wl_min_visits,
            min_cover_visits=args.wl_min_cover_visits,
            check_every=args.wl_check_every,
        )

    top_levels = list(range(max(m_min, m_max - 4), m_max + 1))
    print(f"\nwarm-up steps: {warm_steps}", flush=True)
    for weight, result in arms.items():
        label = "without pull moves" if weight == 0.0 else f"pull_move_weight={weight}"
        covered = result["covered_step"]
        staged = result["stage_step"]
        print(
            f"\n[{label}]\n"
            f"  steps run        : {result['steps_run']}\n"
            f"  steps/s          : {result['steps_per_second']:.0f}\n"
            f"  wall seconds     : {result['wall_seconds']:.1f}\n"
            f"  highest m        : {result['highest_m']}\n"
            f"  first coverage   : "
            + (f"step {covered}" if covered > 0 else "NOT COVERED") + "\n"
            f"  stage criterion  : "
            + (f"met at step {staged}" if staged > 0 else "NOT MET") + "\n"
            f"  min visits/level : {result['min_visits_in_window']} "
            f"(need {args.wl_min_visits})",
            flush=True,
        )
        for level in top_levels:
            step = int(result["first_hit"][level - m_min])
            print(
                f"    m={level:<3d} first hit: "
                + (f"step {step}" if step > 0 else "never"),
                flush=True,
            )

    baseline, treated = arms[0.0], arms[float(args.pull_move_weight)]
    print("\n=== verdict ===", flush=True)
    if float(args.pull_move_weight) == 0.0:
        print("both arms ran at weight 0.0; nothing to compare.", flush=True)
        return 0
    # The stage criterion is the bar that a starving stage fails, so it decides
    # the verdict; first coverage is reported alongside but is much weaker.
    treated_stage, baseline_stage = treated["stage_step"], baseline["stage_step"]
    if treated_stage > 0 and baseline_stage <= 0:
        print(
            f"pull moves met the stage criterion in {treated_stage} steps; the "
            "pull-free arm did not meet it at all within the budget.",
            flush=True,
        )
    elif treated_stage > 0 and baseline_stage > 0:
        print(
            f"both arms met the stage criterion: {treated_stage} steps with pull "
            f"moves, {baseline_stage} without.",
            flush=True,
        )
        if treated_stage < baseline_stage:
            print(
                f"pull moves needed {baseline_stage / treated_stage:.1f}x fewer "
                f"steps, but cost {treated['wall_seconds']:.1f}s against "
                f"{baseline['wall_seconds']:.1f}s of wall clock. The step count "
                "is the mixing measure; the wall clock is what a run actually "
                "spends.",
                flush=True,
            )
        else:
            print(
                "pull moves were no faster in steps. Neither arm starved on "
                "this state, so it does not discriminate: the state is not the "
                "one that fails, whatever its log_g spread.",
                flush=True,
            )
    else:
        print(
            f"neither arm met the stage criterion within "
            f"{args.pull_move_probe_steps} steps "
            f"(min visits/level {treated['min_visits_in_window']} with pull "
            f"moves versus {baseline['min_visits_in_window']} without, "
            f"need {args.wl_min_visits}). At --pull_move_weight "
            f"{args.pull_move_weight} this is insufficient on the measured "
            "state. Reported as measured; the default has NOT been tuned to "
            "make this pass.",
            flush=True,
        )
    if baseline["steps_per_second"] > 0:
        print(
            f"throughput cost: {treated['steps_per_second']:.0f} vs "
            f"{baseline['steps_per_second']:.0f} steps/s "
            f"({baseline['steps_per_second'] / max(treated['steps_per_second'], 1e-9):.2f}x "
            "slower per attempted move).",
            flush=True,
        )
    return 0


def run_self_test() -> int:
    print("Self-test: exact enumeration and reweighted WL production")
    n_beads = 6
    exact_values, exact_prob, exact_radii = enumerate_rooted_saws(n_beads)
    if int(exact_prob.size) < 2:
        print("FAIL: exact test chain does not span multiple contact levels")
        return 1
    m_min, m_max = int(exact_values.min()), int(exact_values.max())
    learned = learn_log_density(
        n_beads=n_beads,
        m_min=m_min,
        m_max=m_max,
        seed=1701,
        initial_log_f=1.0,
        final_log_f=1e-4,
        flatness=0.75,
        min_visits=100,
        check_every=1000,
        max_steps=1_000_000,
        progress=False,
    )
    results = [
        run_production_chain(
            worker_id=i,
            seed=9000 + i,
            initial_chain=learned["chain"],
            log_g=learned["log_g"],
            m_min=m_min,
            m_max=m_max,
            steps=250_000,
            burnin=0.1,
            sample_every=5,
            progress=False,
        )
        for i in range(2)
    ]
    contacts = np.concatenate([result["contact_samples"] for result in results])
    radii = np.concatenate([result["rg_samples"] for result in results])
    bends = np.concatenate([result["bend_samples"] for result in results])
    coordination_histograms = np.concatenate([
        result["coordination_histogram_samples"] for result in results
    ], axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        built = build_distributions(
            contacts,
            radii,
            bends,
            learned["log_g"],
            m_min,
            24,
            False,
            n_beads=n_beads,
            m_cover=m_max,
            coordination_histograms=coordination_histograms,
        )
    observed = np.zeros(m_max - m_min + 1)
    observed[built["c_vals"] - m_min] = built["c_prob"]
    exact = np.zeros_like(observed)
    exact[exact_values - m_min] = exact_prob
    total_variation = 0.5 * float(np.abs(observed - exact).sum())
    weighted_mean_rg = float(np.dot(built["weights"], radii))
    exact_mean_rg = float(exact_radii.mean())
    relative_rg_error = abs(weighted_mean_rg - exact_mean_rg) / exact_mean_rg
    checks = {
        "23 proper non-identity rotations": len(ROT_MATS) == 23,
        "exact N=6 rooted SAW count is 3534": exact_radii.size == 3534,
        "contact total variation < 0.03": total_variation < 0.03,
        "mean Rg relative error < 0.02": relative_rg_error < 0.02,
        "P(m) normalized": abs(float(built["c_prob"].sum()) - 1.0) < 1e-12,
        "P(Rg) normalized": abs(float(built["rg_prob"].sum()) - 1.0) < 1e-12,
        "joint normalized": abs(float(built["crg_prob"].sum()) - 1.0) < 1e-12,
        "joint contact marginal consistent": built["marginal_m_error"] < 1e-12,
        "joint Rg marginal consistent": built["marginal_rg_error"] < 1e-12,
    }
    delta_rng = random.Random(771)
    delta_chain: List[Vec] = [(i, 0, 0) for i in range(12)]
    delta_occupied = set(delta_chain)
    delta_contact = 0
    delta_updates_agree = True
    for _ in range(5000):
        valid, candidate, candidate_occupied, _ = delta_rng.choice(MOVE_FUNCS)(
            delta_chain, delta_occupied, delta_rng
        )
        if not valid:
            continue
        candidate_contact = delta_contact + contact_delta_from_occupancy(
            delta_occupied, candidate_occupied
        )
        if candidate_contact != contact_count(candidate, candidate_occupied):
            delta_updates_agree = False
            break
        delta_chain, delta_occupied, delta_contact = (
            candidate, candidate_occupied, candidate_contact
        )
    checks["incremental contact updates match full recounts"] = delta_updates_agree

    # Regression guard for a verified gapped contact spectrum.  Exact
    # enumeration shows that the 8-bead SAW can reach m=5 but not m=4.
    gap_values, _, _ = enumerate_rooted_saws(8)
    gap_active = np.isin(np.arange(6), gap_values)
    gap_tier = np.where(gap_active, TIER_FLAT, TIER_EXCLUDED).astype(np.int8)
    checks["exact N=8 support has only the internal gap m=4"] = (
        not bool(gap_active[4]) and bool(gap_active[5]) and bool(gap_active[:4].all())
    )
    try:
        learn_log_density(
            n_beads=8, m_min=0, m_max=5, seed=606, initial_log_f=1.0,
            final_log_f=0.01, flatness=0.8, min_visits=100, check_every=2000,
            max_steps=2_000_000, tier=gap_tier, progress=False,
        )
        gap_learn_ok = True
    except RuntimeError:
        gap_learn_ok = False
    checks["learning converges on a gapped window"] = gap_learn_ok

    skips: List[str] = []
    auto_dir = Path(__file__).resolve().parent

    target_expectations = {
        30: (29, 30, 20, 25),
        44: (43, 50, 39, 49),
        60: (59, 74, 66, 74),
    }
    target_paths = {
        n: auto_dir / f"remd_distributions_{n}mer.npz"
        for n in target_expectations
    }
    if all(path.exists() for path in target_paths.values()):
        target_levels_ok = True
        for n, (offset, maximum, expected_1e2, expected_1e3) in target_expectations.items():
            support = load_target_contact_support(target_paths[n], offset)
            target_levels_ok &= flat_level(support, 1e-2, maximum) == expected_1e2
            target_levels_ok &= flat_level(support, 1e-3, maximum) == expected_1e3
        checks["target-derived tier levels match N=30/44/60 regressions"] = target_levels_ok
    else:
        skips.append("target-derived tier levels (REMD NPZ files absent)")

    tiered_tvd = float("inf")
    try:
        tiered_window = make_contact_tiers(0, 2, 0, 2)
        tiered_learned = learn_log_density(
            n_beads=6,
            m_min=0,
            m_max=2,
            seed=1711,
            initial_log_f=1.0,
            final_log_f=1e-3,
            flatness=0.8,
            min_visits=100,
            min_cover_visits=5,
            check_every=500,
            max_steps=1_000_000,
            tier=tiered_window,
            progress=False,
        )
        tiered_results = [
            run_production_chain(
                i, 9100 + i, tiered_learned["chain"], tiered_learned["log_g"],
                0, 2, 150_000, 0.1, 5, False, tiered_window,
            )
            for i in range(2)
        ]
        tiered_contacts = np.concatenate(
            [result["contact_samples"] for result in tiered_results]
        )
        tiered_radii = np.concatenate(
            [result["rg_samples"] for result in tiered_results]
        )
        tiered_bends = np.concatenate(
            [result["bend_samples"] for result in tiered_results]
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            tiered_built = build_distributions(
                tiered_contacts,
                tiered_radii,
                tiered_bends,
                tiered_learned["log_g"],
                0,
                24,
                False,
                n_beads=6,
                m_cover=2,
            )
        tiered_exact = np.zeros(3, dtype=np.float64)
        tiered_exact[exact_values] = exact_prob
        tiered_tvd = 0.5 * float(
            np.abs(tiered_built["c_prob"] - tiered_exact).sum()
        )
        tiered_ok = tiered_tvd < 0.03
    except (RuntimeError, ValueError):
        tiered_ok = False
    checks["coverage-only tier preserves exact N=6 P(m)"] = tiered_ok

    legacy_paths = [
        auto_dir / LEGACY_BASELINE_FILES[n] for n in (30, 44, 60)
    ]
    if all(path.exists() for path in legacy_paths):
        subset_ok = True
        for n in (30, 44, 60):
            legacy, _, _ = legacy_rg_grid(n, auto_dir)
            emitted = fixed_rg_edges(n, auto_dir)
            matches = np.flatnonzero(np.isclose(emitted, legacy[0], atol=1e-12))
            subset_ok &= bool(matches.size)
            if matches.size:
                start = int(matches[0])
                subset_ok &= bool(np.allclose(
                    emitted[start:start + legacy.size], legacy,
                    rtol=0.0, atol=1e-12,
                ))
            subset_ok &= emitted[0] <= min_compact_rg(n)
            subset_ok &= emitted[-1] >= rod_rg(n)
        checks["legacy Rg grids are exact subsets of extended grids"] = subset_ok
    else:
        skips.append("legacy Rg grid subset checks (baseline NPZ files absent)")

    try:
        assert_within_grid(np.array([-0.1]), np.array([0.0, 1.0]), "test")
        guard_raised = False
    except ValueError:
        guard_raised = True
    checks["out-of-range histogram guard raises"] = guard_raised
    checks["normal distribution build has zero Rg drops"] = (
        built["rg_out_of_range_count"] == 0
    )

    missing_tier = make_contact_tiers(0, 2, 2, 2)
    missing_result = run_production_chain(
        0,
        8181,
        [(i, 0, 0) for i in range(6)],
        np.array([0.0, 0.0, 1000.0]),
        0,
        2,
        2000,
        0.0,
        1,
        False,
        missing_tier,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        missing_built = build_distributions(
            missing_result["contact_samples"],
            missing_result["rg_samples"],
            missing_result["bend_samples"],
            np.array([0.0, 0.0, 1000.0]),
            0,
            12,
            False,
            n_beads=6,
            m_cover=2,
        )
    checks["declared contact axis survives a missed high level"] = (
        np.array_equal(missing_built["c_vals"], np.arange(3))
        and np.array_equal(missing_built["c_edges"], np.arange(-0.5, 3.5))
        and missing_built["c_prob"][2] == 0.0
    )

    blocked_stderr, _ = blocked_contact_stderr(
        [result["contact_samples"] for result in results],
        learned["log_g"],
        m_min,
        m_max,
        10,
    )
    resampled_indices = systematic_resample(
        built["weights"], np.random.default_rng(12345)
    )
    unique_fraction = np.unique(resampled_indices).size / resampled_indices.size
    with tempfile.TemporaryDirectory() as temporary:
        short_path = Path(temporary) / "self_test_baseline.npz"
        np.savez_compressed(
            short_path,
            c_vals=built["c_vals"],
            c_prob=built["c_prob"],
            c_edges=built["c_edges"],
            rg_edges=built["rg_edges"],
            rg_prob=built["rg_prob"],
            crg_prob=built["crg_prob"],
            production_c_counts=built["production_c_counts"],
            c_blocked_stderr=blocked_stderr,
            c_samples_resampled=contacts[resampled_indices],
            rg_samples_resampled=radii[resampled_indices],
            bend_samples_resampled=bends[resampled_indices],
            raw_samples_resampled=True,
            raw_samples_unique_fraction=unique_fraction,
            raw_samples_warning=RAW_SAMPLES_WARNING,
        )
        with np.load(short_path, allow_pickle=False) as emitted:
            provenance_ok = (
                "c_samples" not in emitted.files
                and "c_samples_resampled" in emitted.files
                and float(emitted["raw_samples_unique_fraction"]) < 1.0
                and emitted["c_blocked_stderr"].shape == emitted["c_vals"].shape
                and np.all(
                    emitted["c_blocked_stderr"][emitted["production_c_counts"] > 1]
                    > 0.0
                )
            )
        checks["resampled-array provenance and blocked errors are explicit"] = provenance_ok

        matplotlib_dir = Path(temporary) / "matplotlib"
        matplotlib_dir.mkdir()
        os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_dir))
        try:
            from fit_lattice_contact_model_2 import build_baseline_mass_on_integer
        except ImportError:
            skips.append("fitter baseline round trip (fitter import unavailable)")
        else:
            try:
                round_trip = build_baseline_mass_on_integer(
                    np.arange(m_min, m_max + 1), short_path
                )
                fitter_ok = (
                    round_trip.shape == (m_max - m_min + 1,)
                    and abs(float(round_trip.sum()) - 1.0) < 1e-12
                    and np.array_equal(built["c_vals"], np.arange(m_min, m_max + 1))
                )
            except Exception:
                fitter_ok = False
            checks["fitter loads the declared zero-preserving contact support"] = fitter_ok

    print(f"  exact P(m)       = {dict(zip(exact_values.tolist(), exact_prob.tolist()))}")
    print(f"  estimated P(m)   = {dict(zip(built['c_vals'].tolist(), built['c_prob'].tolist()))}")
    print(f"  contact TVD      = {total_variation:.6g}")
    print(f"  tiered contact TVD = {tiered_tvd:.6g}")
    print(f"  mean Rg rel.err  = {relative_rg_error:.6g}")
    for description, passed in checks.items():
        print(f"  {'PASS' if passed else 'FAIL'}: {description}")
    for description in skips:
        print(f"  SKIP: {description}")
    if all(checks.values()):
        print("SELF-TEST PASSED")
        return 0
    print("SELF-TEST FAILED")
    return 1


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate a single-chain athermal baseline consistently using "
        "Wang-Landau contact flattening plus fixed-weight production reweighting.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--N", type=int, default=44, help="number of beads")
    parser.add_argument(
        "--m_max", type=int, default=None,
        help="exact geometric contact maximum; N=30, 44, and 60 use verified values automatically",
    )
    parser.add_argument(
        "--m_min", type=int, default=0,
        help="lowest contact level; currently must be 0 so the straight initializer is in-window",
    )
    parser.add_argument(
        "--m_flat", type=int, default=None,
        help="last tier-2 level; otherwise derived from the molecular target",
    )
    parser.add_argument(
        "--m_cover", type=int, default=None,
        help="last included level; defaults to the exact geometric maximum",
    )
    parser.add_argument(
        "--flat_tail_threshold", type=float, default=1e-3,
        help="worst-temperature target tail allowed above the tier-2 boundary",
    )
    parser.add_argument(
        "--cover_tail_threshold", type=float, default=0.0,
        help="target-tail threshold for explicit truncation; 0 keeps the full window",
    )
    parser.add_argument(
        "--flat_tail_scope", choices=("full", "in_window"), default="full",
        help="target used when deriving m_flat from --flat_tail_threshold. "
             "'full' measures the tail against the whole molecular target, "
             "including mass above m_max that no lattice level can represent. "
             "'in_window' renormalises to m <= m_max first, so the flat tier is "
             "not made maximally strict by a remainder it cannot satisfy. This "
             "changes tier boundaries and therefore needs a deliberate "
             "sign-off; --cover_tail_threshold is unaffected, since declared "
             "truncation must report omitted mass against the full target",
    )
    parser.add_argument(
        "--target_npz", type=str, default=None,
        help="molecular REMD target used to derive contact tiers",
    )
    parser.add_argument("--T", type=float, default=1.0, help="metadata only; target is athermal")
    parser.add_argument("--eps", type=float, default=0.0, help="metadata only; target is athermal")
    parser.add_argument("--dist_dir", type=str, default="dists", help="output directory")
    parser.add_argument("--rg_bins", type=int, default=60, help="number of Rg bins")
    parser.add_argument("--no_joint", action="store_true", help="omit joint P(m,Rg)")
    parser.add_argument("--base_seed", type=int, default=42, help="production seed base")
    parser.add_argument("--wl_seed", type=int, default=1729, help="WL learning seed")
    parser.add_argument("--n_workers", type=int, default=12, help="fixed-weight production workers")
    parser.add_argument(
        "--steps_per_worker", type=int, default=400_000_000,
        help="fixed-weight production attempts per worker",
    )
    parser.add_argument("--burnin", type=float, default=0.3, help="production burn-in fraction")
    parser.add_argument("--sample_every", type=int, default=500, help="production sampling interval")
    parser.add_argument("--wl_initial_log_f", type=float, default=1.0)
    parser.add_argument("--wl_final_log_f", type=float, default=1e-4)
    parser.add_argument("--wl_flatness", type=float, default=0.8)
    parser.add_argument("--wl_min_visits", type=int, default=1000)
    parser.add_argument("--wl_min_cover_visits", type=int, default=50)
    parser.add_argument("--wl_check_every", type=int, default=100_000)
    parser.add_argument("--wl_max_steps", type=int, default=10_000_000_000)
    parser.add_argument("--wl_max_seconds", type=float, default=21_600.0)
    parser.add_argument(
        "--wl_max_seconds_scope", choices=("cumulative", "per_invocation"),
        default="cumulative",
        help="whether --wl_max_seconds bounds total learning time across "
             "resumes or only the current invocation",
    )
    parser.add_argument("--wl_max_steps_per_stage", type=int, default=1_000_000_000)
    parser.add_argument(
        "--wl_schedule", choices=("halving", "one_over_t"), default="halving",
        help="density-refinement schedule",
    )
    parser.add_argument(
        "--wl_stage_stall_steps", type=int, default=50_000_000,
        help="attempted moves in one incomplete stage after which the "
             "one_over_t schedule relaxes stage advancement: coverage is then "
             "judged cumulatively and flatness only over visited tier-2 "
             "levels. log_f is not reduced and halving continues; only the "
             "Belardinelli-Pereyra crossing enters the 1/t phase",
    )
    parser.add_argument(
        "--pull_move_weight", type=float, default=0.25,
        help="probability of proposing a Lesh-Mitzenmacher-Whitesides pull move; "
             "the remainder is split evenly among pivot, corner-flip and end "
             "moves. Pull moves relocate a bead whose neighbours are all "
             "occupied, which the local moves cannot do, at the cost of an "
             "extra catalog build per proposal. 0.0 reproduces the move mix "
             "used before pull moves existed, bit for bit",
    )
    parser.add_argument(
        "--wl_init", choices=("rod", "compact"), default="rod",
        help="initial learning conformation. 'rod' is the straight chain. "
             "'compact' seeds a boustrophedon snake through the verified "
             "optimal bounding box, which starts at the exact geometric m_max "
             "and removes the search problem of reaching the compact end from "
             "the rod. Recommended for N=44 and N=60; it does not affect the "
             "resume path, which never re-seeds",
    )
    parser.add_argument(
        "--checkpoint_every_seconds", type=float, default=1800.0,
        help="checkpoint timer within a WL stage",
    )
    parser.add_argument(
        "--show_contact_upper_bound", action="store_true",
        help="print the rigorous analytical upper bound and verified exact m_max, then exit",
    )
    parser.add_argument(
        "--excluded_contact_levels", type=int, nargs="*", default=[],
        help="verified unreachable internal contact levels to omit from flatness "
             "and production-support checks",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="write a restartable NPZ after every completed WL refinement stage",
    )
    parser.add_argument(
        "--resume_checkpoint", type=str, default=None,
        help="resume DOS learning from a checkpoint (histogram and RNG stream restart)",
    )
    parser.add_argument(
        "--production_checkpoint", type=str, default=None,
        help="checkpoint stem for the fixed-weight production phase; worker i "
             "writes {stem}_prod_w{i}.npz periodically",
    )
    parser.add_argument(
        "--resume_production_checkpoint", type=str, default=None,
        help="resume every production worker from {stem}_prod_w{i}.npz "
             "(sample arrays and RNG state restart from the saved state)",
    )
    parser.add_argument(
        "--production_max_seconds", type=float, default=math.inf,
        help="refuse to launch production when the measured-throughput "
             "projection of the per-worker wall clock exceeds this; the "
             "default never refuses",
    )
    parser.add_argument(
        "--min_production_round_trips", type=int, default=1,
        help="minimum summed low-high-low round trips; zero disables the hard check",
    )
    parser.add_argument(
        "--min_production_samples_per_level", type=int, default=1,
        help="minimum recorded fixed-weight samples in every required contact level",
    )
    parser.add_argument(
        "--save_raw_samples", dest="save_raw_samples", action="store_true", default=True,
        help="store an athermal importance-resampled c/Rg/bend sample set",
    )
    parser.add_argument("--no_save_raw_samples", dest="save_raw_samples", action="store_false")
    parser.add_argument(
        "--legacy_sample_aliases", action="store_true",
        help="also write deprecated c_samples/rg_samples/bend_samples aliases",
    )
    parser.add_argument(
        "--n_blocks", type=int, default=20,
        help="non-overlapping batch-means blocks per production worker",
    )
    parser.add_argument("--self-test", action="store_true", help="run exact small-chain validation")
    parser.add_argument(
        "--pull_move_probe", action="store_true",
        help="measure re-reachability of the compact window with and without "
             "pull moves from one warmed log_g, then exit. Not a unit test: it "
             "is far too slow for CI. Use --wl_schedule halving",
    )
    parser.add_argument(
        "--pull_move_probe_steps", type=int, default=10_000_000,
        help="step budget for the probe warm-up and for each measured arm",
    )
    parser.add_argument(
        "--pull_move_probe_log_g", type=str, default=None,
        help="optional WL checkpoint supplying a real built-up log_g and chain, "
             "skipping the probe's own warm-up",
    )
    return parser.parse_args(argv)


def validate_common_args(args: argparse.Namespace) -> None:
    if args.N < 3:
        raise ValueError("--N must be at least 3")
    if args.m_min != 0:
        raise ValueError("this implementation requires --m_min=0")


def resolve_m_max(args: argparse.Namespace) -> None:
    verified_maximum = geometric_contact_maximum(args.N)
    if verified_maximum is not None:
        if args.m_max is not None and args.m_max != verified_maximum:
            raise ValueError(
                f"--m_max={args.m_max} conflicts with the verified exact geometric "
                f"maximum {verified_maximum} for N={args.N}"
            )
        args.m_max = verified_maximum
    elif args.m_max is None:
        raise ValueError(
            "--m_max is required for chain lengths without an encoded exact "
            "geometric maximum"
        )


def resolve_contact_tiers(args: argparse.Namespace) -> Dict[str, Any]:
    """Resolve target-derived tier boundaries and truncation diagnostics."""
    default_target = Path(__file__).resolve().with_name(
        f"remd_distributions_{args.N}mer.npz"
    )
    target_path = Path(args.target_npz) if args.target_npz else default_target
    explicit_target = args.target_npz is not None
    target = None
    target_report = None
    if target_path.exists() and args.N in CONTACT_OFFSETS:
        target = load_target_contact_support(target_path, CONTACT_OFFSETS[args.N])
        target_report = support_report(target, args.m_max)
    elif explicit_target and not target_path.exists():
        raise FileNotFoundError(f"target REMD NPZ does not exist: {target_path}")

    # The flat tier answers "where does the target still have mass a lattice
    # level can hold?".  Under 'in_window' the target is renormalised to
    # m <= m_max first, so the unsatisfiable remainder above the geometric
    # maximum -- already reported by support_report -- does not also drive
    # m_flat upward.
    flat_target = target
    if target is not None and args.flat_tail_scope == "in_window":
        flat_target = restrict_to_window(target, args.m_max)

    if args.m_flat is not None:
        m_flat = int(args.m_flat)
    elif flat_target is not None:
        m_flat = flat_level(flat_target, args.flat_tail_threshold, args.m_max)
        clamp_tail = tail_mass_above(flat_target, m_flat)
        if not clamp_tail < args.flat_tail_threshold:
            print(
                "WARNING: the requested flat-tail threshold is not reached before "
                f"the geometric maximum; m_flat is clamped to {args.m_max}. The "
                f"worst-temperature {args.flat_tail_scope} tail above m={m_flat} "
                f"is {clamp_tail:.6g} against --flat_tail_threshold="
                f"{args.flat_tail_threshold:.6g}.",
                flush=True,
            )
    else:
        m_flat = int(args.m_max)
        print(
            "WARNING: no usable target REMD NPZ was found and --m_flat was not "
            "provided; flattening the full geometric contact window.",
            flush=True,
        )

    if args.m_cover is not None:
        m_cover = int(args.m_cover)
    elif args.cover_tail_threshold == 0.0:
        m_cover = int(args.m_max)
    elif target is not None:
        m_cover = flat_level(target, args.cover_tail_threshold, args.m_max)
        if not tail_mass_above(target, m_cover) < args.cover_tail_threshold:
            print(
                "WARNING: the requested cover-tail threshold is not reached before "
                f"the geometric maximum; m_cover is clamped to {args.m_max}.",
                flush=True,
            )
    else:
        m_cover = int(args.m_max)
        print(
            "WARNING: no usable target REMD NPZ was found; m_cover defaults to "
            "the geometric maximum.",
            flush=True,
        )

    tier = make_contact_tiers(
        args.m_min,
        args.m_max,
        m_flat,
        m_cover,
        args.excluded_contact_levels,
    )
    declared_truncation = m_cover < args.m_max
    truncation_mass = (
        tail_mass_above(target, m_cover)
        if declared_truncation and target is not None else float("nan")
    )
    if declared_truncation:
        target_text = (
            f"{100.0 * truncation_mass:.6f}% worst-temperature target mass"
            if math.isfinite(truncation_mass) else "unknown target mass"
        )
        print("=" * 72, flush=True)
        print(
            f"DECLARED TRUNCATION: levels {m_cover + 1}..{args.m_max} are excluded; "
            f"omitted {target_text}.",
            flush=True,
        )
        print(
            "The omitted athermal mass is not estimable from this conditional run.",
            flush=True,
        )
        print("=" * 72, flush=True)

    args.m_flat = m_flat
    args.m_cover = m_cover
    return {
        "tier": tier,
        "target_path": target_path,
        "target": target,
        "target_report": target_report,
        "declared_truncation": declared_truncation,
        "declared_truncation_target_mass": truncation_mass,
    }


def validate_args(args: argparse.Namespace) -> None:
    validate_common_args(args)
    resolve_m_max(args)
    if args.m_max < args.m_min:
        raise ValueError("--m_max must be nonnegative")
    rigorous_bound = contact_upper_bound(args.N)
    if args.m_max > rigorous_bound:
        raise ValueError(
            f"--m_max={args.m_max} exceeds the cubic-lattice contact upper bound "
            f"{rigorous_bound} for N={args.N}"
        )
    if args.n_workers < 1 or args.steps_per_worker < 1:
        raise ValueError("--n_workers and --steps_per_worker must be positive")
    if args.rg_bins < 1 or args.sample_every < 1:
        raise ValueError("--rg_bins and --sample_every must be positive")
    if not 0.0 <= args.burnin < 1.0:
        raise ValueError("--burnin must lie in [0,1)")
    expected = (args.steps_per_worker - int(round(args.burnin * args.steps_per_worker))) // args.sample_every
    if expected < 1:
        raise ValueError("production settings yield zero samples per worker")
    if not (args.wl_initial_log_f > args.wl_final_log_f > 0.0):
        raise ValueError("require --wl_initial_log_f > --wl_final_log_f > 0")
    if not 0.0 < args.wl_flatness <= 1.0:
        raise ValueError("--wl_flatness must lie in (0,1]")
    if not 0.0 <= args.pull_move_weight <= 1.0:
        raise ValueError("--pull_move_weight must lie in [0,1]")
    if min(
        args.wl_min_visits,
        args.wl_min_cover_visits,
        args.wl_check_every,
        args.wl_max_steps,
        args.wl_max_steps_per_stage,
        args.wl_stage_stall_steps,
    ) < 1:
        raise ValueError("WL visit/check/step controls must be positive")
    if args.wl_max_seconds <= 0.0 or args.checkpoint_every_seconds <= 0.0:
        raise ValueError("WL time controls must be positive")
    if args.production_max_seconds <= 0.0:
        raise ValueError("--production_max_seconds must be positive")
    if not 0.0 < args.flat_tail_threshold < 1.0:
        raise ValueError("--flat_tail_threshold must lie in (0,1)")
    if not 0.0 <= args.cover_tail_threshold < 1.0:
        raise ValueError("--cover_tail_threshold must lie in [0,1)")
    excluded = list(args.excluded_contact_levels)
    if len(set(excluded)) != len(excluded):
        raise ValueError("--excluded_contact_levels contains duplicates")
    if args.min_production_round_trips < 0:
        raise ValueError("--min_production_round_trips must be nonnegative")
    if args.min_production_samples_per_level < 1:
        raise ValueError("--min_production_samples_per_level must be positive")
    if args.n_blocks < 2:
        raise ValueError("--n_blocks must be at least 2")
    if args.legacy_sample_aliases and not args.save_raw_samples:
        raise ValueError("--legacy_sample_aliases requires raw-sample saving")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        return run_self_test()
    if args.pull_move_probe:
        return run_pull_move_probe(args)
    if args.show_contact_upper_bound:
        validate_common_args(args)
        return report_contact_upper_bound(args)
    validate_args(args)
    checkpoint = Path(args.checkpoint) if args.checkpoint else None
    resume = Path(args.resume_checkpoint) if args.resume_checkpoint else None

    window = np.arange(args.m_min, args.m_max + 1, dtype=np.int64)
    tier_info = resolve_contact_tiers(args)
    tier = np.asarray(tier_info["tier"], dtype=np.int8)
    active = tier > TIER_EXCLUDED
    coverage_text = (
        f"{args.m_flat + 1}..{args.m_cover}"
        if args.m_cover > args.m_flat else "none"
    )
    excluded_text = (
        f"{args.m_cover + 1}..{args.m_max}"
        if args.m_cover < args.m_max else "none"
    )
    print(
        f"Contact tiers: flat={args.m_min}..{args.m_flat}, "
        f"coverage={coverage_text}, excluded_above={excluded_text}",
        flush=True,
    )
    if args.excluded_contact_levels:
        print(
            "Verified internal contact gaps excluded from convergence checks: "
            f"{sorted(args.excluded_contact_levels)}",
            flush=True,
        )

    print("=== Wang-Landau learning (adaptive samples are discarded) ===", flush=True)
    learned = learn_log_density(
        n_beads=args.N,
        m_min=args.m_min,
        m_max=args.m_max,
        seed=args.wl_seed,
        initial_log_f=args.wl_initial_log_f,
        final_log_f=args.wl_final_log_f,
        flatness=args.wl_flatness,
        min_visits=args.wl_min_visits,
        min_cover_visits=args.wl_min_cover_visits,
        check_every=args.wl_check_every,
        max_steps=args.wl_max_steps,
        max_seconds=args.wl_max_seconds,
        max_seconds_scope=args.wl_max_seconds_scope,
        max_steps_per_stage=args.wl_max_steps_per_stage,
        schedule=args.wl_schedule,
        stage_stall_steps=args.wl_stage_stall_steps,
        pull_move_weight=args.pull_move_weight,
        checkpoint_every_seconds=args.checkpoint_every_seconds,
        init=args.wl_init,
        tier=tier,
        checkpoint=checkpoint,
        resume_checkpoint=resume,
    )
    print(
        f"WL complete: stages={learned['stages_completed']} "
        f"steps={learned['attempted_steps']} round_trips={learned['round_trips']} "
        f"wall={learned['wall_time']:.1f}s "
        f"mode={_mode_label(args.wl_schedule, learned['one_over_t_mode'], learned['stall_relaxed'])}"
        + (
            f" (entered via {learned['one_over_t_trigger']} trigger, "
            f"{learned['one_over_t_round_trips']} round trips since)"
            if learned["one_over_t_mode"] else ""
        )
        + (
            f" (stall relaxation engaged at step {learned['stall_relaxed_step']}; "
            "log_f was not reduced and halving continued)"
            if learned["stall_relaxed"] else ""
        ),
        flush=True,
    )

    worker_seeds = [args.base_seed + i for i in range(args.n_workers)]
    print("=== Frozen-weight production ===", flush=True)
    production_steps_per_second = _report_production_budget(
        chain=learned["chain"],
        log_g=learned["log_g"],
        m_min=args.m_min,
        tier=tier,
        steps_per_worker=args.steps_per_worker,
        n_workers=args.n_workers,
        pull_move_weight=args.pull_move_weight,
        seed=args.base_seed + 20_000_003,
        max_seconds=args.production_max_seconds,
    )
    production_checkpoint = (
        Path(args.production_checkpoint) if args.production_checkpoint else None
    )
    production_resume = (
        Path(args.resume_production_checkpoint)
        if args.resume_production_checkpoint else None
    )
    t0 = time.time()
    results: List[Dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.n_workers) as executor:
        futures = [
            executor.submit(
                run_production_chain,
                i,
                worker_seeds[i],
                learned["chain"],
                learned["log_g"],
                args.m_min,
                args.m_max,
                args.steps_per_worker,
                args.burnin,
                args.sample_every,
                True,
                tier,
                args.pull_move_weight,
                (
                    production_checkpoint_path(production_checkpoint, i)
                    if production_checkpoint is not None else None
                ),
                args.checkpoint_every_seconds,
                (
                    production_checkpoint_path(production_resume, i)
                    if production_resume is not None else None
                ),
            )
            for i in range(args.n_workers)
        ]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(
                f"production worker {result['worker_id']} complete: "
                f"samples={result['contact_samples'].size} "
                f"round_trips={result['round_trips']}",
                flush=True,
            )
    results.sort(key=lambda result: result["worker_id"])

    contacts = np.concatenate([result["contact_samples"] for result in results])
    radii = np.concatenate([result["rg_samples"] for result in results])
    bends = np.concatenate([result["bend_samples"] for result in results])
    coordination_histograms = np.concatenate([
        result["coordination_histogram_samples"] for result in results
    ], axis=0)
    production_counts_full = np.bincount(
        contacts - args.m_min, minlength=window.size
    )[:window.size]
    encountered_excluded = window[(tier == TIER_EXCLUDED) & (production_counts_full > 0)]
    if encountered_excluded.size:
        raise RuntimeError(
            "fixed-weight production encountered contact levels listed as excluded: "
            f"{encountered_excluded.tolist()}; output was not written"
        )
    tier2_minimum = max(args.wl_min_visits, args.min_production_samples_per_level)
    tier1_minimum = max(
        args.wl_min_cover_visits, args.min_production_samples_per_level
    )
    deficient_tier2 = window[
        (tier == TIER_FLAT) & (production_counts_full < tier2_minimum)
    ]
    deficient_tier1 = window[
        (tier == TIER_COVERAGE) & (production_counts_full < tier1_minimum)
    ]
    if deficient_tier2.size or deficient_tier1.size:
        raise RuntimeError(
            "fixed-weight production did not adequately sample the tiered contact "
            f"window: tier 2 deficient={deficient_tier2.tolist()} "
            f"(minimum {tier2_minimum}), tier 1 deficient="
            f"{deficient_tier1.tolist()} (minimum {tier1_minimum}); "
            "output was not written"
        )
    built = build_distributions(
        contacts,
        radii,
        bends,
        learned["log_g"],
        args.m_min,
        args.rg_bins,
        args.no_joint,
        n_beads=args.N,
        m_cover=args.m_cover,
        coordination_histograms=coordination_histograms,
    )
    c_blocked_stderr, blocked_batch_count = blocked_contact_stderr(
        [result["contact_samples"] for result in results],
        learned["log_g"],
        args.m_min,
        args.m_cover,
        args.n_blocks,
    )
    built["c_blocked_stderr"] = c_blocked_stderr
    weights = built.pop("weights")
    ess = effective_sample_size(weights)
    total_round_trips = sum(result["round_trips"] for result in results)
    if total_round_trips < args.min_production_round_trips:
        raise RuntimeError(
            f"fixed-weight production made {total_round_trips} contact round trips, "
            f"below required {args.min_production_round_trips}; output was not written"
        )

    production_max_contact = int(contacts.max())

    accepted_per_worker = np.asarray([r["accepted_moves"] for r in results], dtype=np.int64)
    valid_per_worker = np.asarray([r["geometrically_valid_moves"] for r in results], dtype=np.int64)
    attempted_per_worker = np.asarray([r["attempted_moves"] for r in results], dtype=np.int64)
    samples_per_worker = np.asarray([r["contact_samples"].size for r in results], dtype=np.int64)
    acceptance_per_worker = accepted_per_worker / attempted_per_worker
    total_attempted = int(attempted_per_worker.sum())
    total_accepted = int(accepted_per_worker.sum())

    per_worker_m_mean: List[float] = []
    per_worker_rg_mean: List[float] = []
    offset = 0
    for count in samples_per_worker:
        sl = slice(offset, offset + int(count))
        local_weights = weights[sl] / weights[sl].sum()
        per_worker_m_mean.append(float(np.dot(local_weights, contacts[sl])))
        per_worker_rg_mean.append(float(np.dot(local_weights, radii[sl])))
        offset += int(count)

    dist_dir = Path(args.dist_dir)
    dist_dir.mkdir(parents=True, exist_ok=True)
    output_path = dist_dir / (
        f"{Path(__file__).stem}_N{args.N}_workers{args.n_workers}"
        f"_steps{args.steps_per_worker}_seed{args.base_seed}.npz"
    )
    save: Dict[str, Any] = {
        key: value for key, value in built.items()
        if key not in {"marginal_m_error", "marginal_rg_error"}
    }
    save.update(
        N=int(args.N), n_beads=int(args.N), n_steps=int(args.N) - 1,
        T=float(args.T), eps=float(args.eps),
        n_workers=int(args.n_workers), steps_per_worker=int(args.steps_per_worker),
        total_attempted_steps=total_attempted, base_seed=int(args.base_seed),
        worker_seeds=np.asarray(worker_seeds, dtype=np.int64),
        burnin=float(args.burnin), sample_every=int(args.sample_every),
        n_samples=int(contacts.size), accepted_moves_per_worker=accepted_per_worker,
        acceptance_ratios_per_worker=acceptance_per_worker,
        combined_acceptance_ratio=float(total_accepted / total_attempted),
        rg_bins=int(np.asarray(built["rg_edges"]).size - 1),
        requested_rg_bins=int(args.rg_bins), samples_per_worker=samples_per_worker,
        per_worker_m_mean=np.asarray(per_worker_m_mean),
        per_worker_rg_mean=np.asarray(per_worker_rg_mean),
        kappa_bend=0.0, bending_enabled=False, bend_definition=BEND_DEFINITION,
        sampler="wang_landau_fixed_weight_reweighted",
        raw_samples_resampled=True,
        raw_samples_available=bool(args.save_raw_samples),
        raw_samples_warning=RAW_SAMPLES_WARNING,
        raw_samples_unique_fraction=float("nan"),
        n_blocks_per_worker=int(args.n_blocks),
        c_blocked_batch_count=int(blocked_batch_count),
        wl_m_values=np.arange(args.m_min, args.m_max + 1, dtype=np.int64),
        wl_log_g=np.asarray(learned["log_g"], dtype=np.float64),
        wl_seed=int(args.wl_seed), wl_initial_log_f=float(args.wl_initial_log_f),
        wl_final_log_f=float(args.wl_final_log_f),
        wl_flatness=float(args.wl_flatness), wl_min_visits=int(args.wl_min_visits),
        wl_min_cover_visits=int(args.wl_min_cover_visits),
        wl_max_steps=int(args.wl_max_steps),
        wl_max_seconds=float(args.wl_max_seconds),
        wl_max_steps_per_stage=int(args.wl_max_steps_per_stage),
        wl_schedule=str(args.wl_schedule),
        wl_init=str(args.wl_init),
        wl_pull_move_weight=float(args.pull_move_weight),
        checkpoint_every_seconds=float(args.checkpoint_every_seconds),
        wl_learning_steps=int(learned["attempted_steps"]),
        wl_learning_accepted_moves=int(learned["accepted_moves"]),
        wl_learning_round_trips=int(learned["round_trips"]),
        wl_learning_wall_seconds=float(learned["wall_time"]),
        wl_learning_steps_per_second=float(learned["steps_per_second"]),
        wl_one_over_t_mode=bool(learned["one_over_t_mode"]),
        wl_one_over_t_trigger=str(learned["one_over_t_trigger"]),
        wl_one_over_t_round_trips=int(learned["one_over_t_round_trips"]),
        wl_stage_stall_steps=int(args.wl_stage_stall_steps),
        wl_max_seconds_scope=str(args.wl_max_seconds_scope),
        wl_stall_relaxed=bool(learned["stall_relaxed"]),
        wl_stall_relaxed_step=int(learned["stall_relaxed_step"]),
        wl_visits_since_start=np.asarray(
            learned["visits_since_start"], dtype=np.int64
        ),
        wl_visits_since_one_over_t=np.asarray(
            learned["visits_since_one_over_t"], dtype=np.int64
        ),
        wl_visits_since_stall=np.asarray(
            learned["visits_since_stall"], dtype=np.int64
        ),
        wl_log_g_spread=float(learned["log_g_spread"]),
        wl_log_g_spread_reference=float(learned["log_g_spread_reference"]),
        wl_stages_completed=int(learned["stages_completed"]),
        wl_stage_records=np.asarray(learned["stage_records"]),
        production_steps_per_second=float(production_steps_per_second),
        production_max_seconds=float(args.production_max_seconds),
        production_resumed=bool(args.resume_production_checkpoint),
        production_geometrically_valid_per_worker=valid_per_worker,
        production_round_trips_per_worker=np.asarray(
            [result["round_trips"] for result in results], dtype=np.int64
        ),
        importance_effective_sample_size=float(ess),
        importance_effective_fraction=float(ess / contacts.size),
        n_samples_effective=float(ess),
        production_max_contact=production_max_contact,
        excluded_contact_levels=np.asarray(
            sorted(args.excluded_contact_levels), dtype=np.int64
        ),
        min_production_samples_per_level=int(args.min_production_samples_per_level),
        production_samples_per_level=production_counts_full.astype(np.int64),
        wl_tier=tier,
        wl_active_levels=np.asarray(learned["active"], dtype=bool),
        m_flat=int(args.m_flat), m_cover=int(args.m_cover),
        flat_tail_threshold=float(args.flat_tail_threshold),
        flat_tail_scope=str(args.flat_tail_scope),
        cover_tail_threshold=float(args.cover_tail_threshold),
        target_support_npz=str(tier_info["target_path"]),
        declared_truncation=bool(tier_info["declared_truncation"]),
        declared_truncation_target_mass=float(
            tier_info["declared_truncation_target_mass"]
        ),
        joint_contact_marginal_error=float(built["marginal_m_error"]),
        joint_rg_marginal_error=float(built["marginal_rg_error"]),
    )
    if args.save_raw_samples:
        resampled_indices = systematic_resample(
            weights, np.random.default_rng(args.base_seed + 10_000_019)
        )
        save["raw_samples_unique_fraction"] = float(
            np.unique(resampled_indices).size / resampled_indices.size
        )
        save["c_samples_resampled"] = contacts[resampled_indices]
        save["rg_samples_resampled"] = radii[resampled_indices]
        save["bend_samples_resampled"] = bends[resampled_indices]
        if args.legacy_sample_aliases:
            save["c_samples"] = save["c_samples_resampled"]
            save["rg_samples"] = save["rg_samples_resampled"]
            save["bend_samples"] = save["bend_samples_resampled"]
    np.savez_compressed(output_path, **save)

    print("\n=== Reweighted athermal result ===")
    print(f"P(m) sum                         : {built['c_prob'].sum():.12g}")
    print(f"P(Rg) sum                        : {built['rg_prob'].sum():.12g}")
    print(f"joint sum                        : {built['crg_prob'].sum():.12g}")
    print(f"joint -> P(m) max error          : {built['marginal_m_error']:.3e}")
    print(f"joint -> P(Rg) max error         : {built['marginal_rg_error']:.3e}")
    print(f"importance ESS                   : {ess:.1f}/{contacts.size}")
    print(
        "min production visits per included level: "
        f"{int(production_counts_full[active].min())}"
    )
    print(
        "largest naive count diagnostic  : "
        f"{float(built['c_naive_count_error'].max()):.4f} "
        f"at m={int(built['c_vals'][int(np.argmax(built['c_naive_count_error']))])}"
    )
    print(f"highest sampled contact level    : {production_max_contact} (window {args.m_max})")
    print(f"production round trips           : {total_round_trips}")
    print(f"combined production acceptance   : {total_accepted / total_attempted:.4f}")
    print(f"production wall time             : {time.time() - t0:.1f}s")
    print(f"DIST_FILE = {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
