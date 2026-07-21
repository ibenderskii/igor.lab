#!/usr/bin/env python3
"""
Multi-chain Monte Carlo moves (Stage 2).

Implements the minimum move set for the aggregation model, generalizing the
single-chain moves in ``remd_uniform_chain_2_new.py`` to M chains sharing one
periodic cubic box with global excluded volume:

    * ``end``              selected-chain end move
    * ``crankshaft``       selected-chain corner/kink flip
    * ``pivot``            selected-chain tail pivot (proper cubic rotation)
    * ``chain_translation``whole-chain nearest-neighbour translation
    * ``reptation``        whole-chain slithering-snake (grow one end)
    * ``rotation``         whole-chain rigid rotation about one of its own beads

Every proposal preserves linear connectivity (unit bonds in UNWRAPPED
coordinates), self-avoidance, and interchain excluded volume (checked against
the global wrapped occupancy).  A proposal returns the set of moved global bead
IDs and their new positions -- enough to update contacts with the O(moved)
delta algorithm in :mod:`multichain_contacts`.

Moves are model-agnostic (they know nothing about temperature or the contact
bias).  Metropolis acceptance using the reduced potential lives in the sampler
(:mod:`remd_multichain`).  A ``random.Random`` instance is passed in so the
caller controls determinism (no hidden global RNG state).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Tuple

import numpy as np

import remd_uniform_chain_2_new as _remd  # ROT_MATS (proper cubic rotations)
from lattice_bending import delta_bends as _delta_bends
from multichain_state import (
    NN6, MultiChainState, wrap_coordinate, canonicalize_chain_coordinates)
from multichain_contacts import delta_contacts, apply_moved_beads

Site = Tuple[int, int, int]

# Counter layout shared with the sampler / CSV writer.  The three local moves
# occupy indices 0-2 and ``chain_translation`` index 3; the two large-scale moves
# below are appended AFTER them so ``MultiReplica.local_acceptance_rate``'s
# ``[:3]`` slice keeps meaning pivot/crankshaft/end.
MOVE_TYPES = ("pivot", "crankshaft", "end", "chain_translation",
              "reptation", "rotation")
MOVE_INDEX = {name: i for i, name in enumerate(MOVE_TYPES)}
LOCAL_MOVE_TYPES = ("pivot", "crankshaft", "end")
MOVE_COUNTER_COLS = ("proposed", "geometrically_valid", "state_changing",
                     "metropolis_accepted")
_N_MOVES = len(MOVE_TYPES)
_N_COLS = len(MOVE_COUNTER_COLS)


def new_move_counters() -> np.ndarray:
    """Zeroed (_N_MOVES, 4) counter block: proposed/valid/state_changing/accepted."""
    return np.zeros((_N_MOVES, _N_COLS), dtype=np.int64)


@dataclass
class MoveProposal:
    """A proposed move and the information needed to score/apply it.

    ``moved`` maps each moved global ID to its NEW unwrapped coordinate;
    ``new_sites`` maps the same IDs to their NEW wrapped lattice site.  When
    ``ok`` is False the move is geometrically invalid (connectivity, self-
    avoidance, or interchain excluded volume) and must be discarded.
    """
    move_type: str
    chain: int
    ok: bool
    state_changing: bool
    moved: Dict[int, Tuple[int, int, int]] = field(default_factory=dict)
    new_sites: Dict[int, Site] = field(default_factory=dict)


_INVALID = None


def _check_validity(
    state: MultiChainState, moved: Dict[int, Tuple[int, int, int]],
) -> Optional[Dict[int, Site]]:
    """Return new wrapped sites if ``moved`` is geometrically valid, else None.

    Valid means: the moved beads' new wrapped sites are mutually distinct AND
    each is either empty or currently occupied by one of the moved beads (its
    own vacated site).  A new site occupied by a STATIONARY bead (same chain or
    another chain) fails excluded volume.
    """
    L = int(state.box_size)
    moved_ids = set(moved)
    new_sites: Dict[int, Site] = {}
    seen: set = set()
    for gid, pos in moved.items():
        s = (int(pos[0]) % L, int(pos[1]) % L, int(pos[2]) % L)
        if s in seen:
            return _INVALID  # two moved beads collide on one site
        seen.add(s)
        new_sites[gid] = s
    for gid, s in new_sites.items():
        owner = state.site_owner.get(s)
        if owner is not None and int(owner) not in moved_ids:
            return _INVALID  # collides with a stationary bead
    return new_sites


# ---------------------------------------------------------------------------
# Individual proposals
# ---------------------------------------------------------------------------

def propose_end(state: MultiChainState, rng: random.Random) -> MoveProposal:
    """Move one end bead of a random chain to a neighbour of its anchor."""
    N = state.chain_length
    c = rng.randrange(state.n_chains)
    end = 0 if rng.random() < 0.5 else N - 1
    anchor = 1 if end == 0 else N - 2
    gid = c * N + end
    anchor_pos = state.coords_unwrapped[c, anchor]
    v = NN6[rng.randrange(6)]
    new_pos = (int(anchor_pos[0] + v[0]), int(anchor_pos[1] + v[1]),
               int(anchor_pos[2] + v[2]))
    old_pos = tuple(int(x) for x in state.coords_unwrapped[c, end])
    moved = {gid: new_pos}
    new_sites = _check_validity(state, moved)
    if new_sites is _INVALID:
        return MoveProposal("end", c, ok=False, state_changing=False)
    return MoveProposal(
        "end", c, ok=True, state_changing=(new_pos != old_pos),
        moved=moved, new_sites=new_sites)


def propose_crankshaft(state: MultiChainState, rng: random.Random) -> MoveProposal:
    """Corner/kink flip of one interior bead of a random chain."""
    N = state.chain_length
    c = rng.randrange(state.n_chains)
    if N < 3:
        return MoveProposal("crankshaft", c, ok=False, state_changing=False)
    i = rng.randrange(1, N - 1)
    a = state.coords_unwrapped[c, i - 1].astype(np.int64)
    b = state.coords_unwrapped[c, i].astype(np.int64)
    d = state.coords_unwrapped[c, i + 1].astype(np.int64)
    u1 = tuple(int(x) for x in (b - a))
    u2 = tuple(int(x) for x in (d - b))
    # Require a right-angle corner (perpendicular unit bonds).
    if u1 == u2 or u1 == (-u2[0], -u2[1], -u2[2]):
        return MoveProposal("crankshaft", c, ok=False, state_changing=False)
    if u1[0] * u2[0] + u1[1] * u2[1] + u1[2] * u2[2] != 0:
        return MoveProposal("crankshaft", c, ok=False, state_changing=False)
    b_new = (int(a[0] + u2[0]), int(a[1] + u2[1]), int(a[2] + u2[2]))
    gid = c * N + i
    old_pos = tuple(int(x) for x in b)
    moved = {gid: b_new}
    new_sites = _check_validity(state, moved)
    if new_sites is _INVALID:
        return MoveProposal("crankshaft", c, ok=False, state_changing=False)
    return MoveProposal(
        "crankshaft", c, ok=True, state_changing=(b_new != old_pos),
        moved=moved, new_sites=new_sites)


def propose_pivot(state: MultiChainState, rng: random.Random) -> MoveProposal:
    """Rotate the tail of a random chain about a random pivot bead."""
    N = state.chain_length
    c = rng.randrange(state.n_chains)
    if N < 3:
        return MoveProposal("pivot", c, ok=False, state_changing=False)
    i = rng.randrange(1, N - 1)
    pivot = state.coords_unwrapped[c, i].astype(np.int64)
    R = np.asarray(rng.choice(_remd.ROT_MATS), dtype=np.int64)
    tail = state.coords_unwrapped[c, i + 1:].astype(np.int64)
    rel = tail - pivot
    rotated = (rel @ R.T) + pivot  # integer rotation about the pivot
    moved: Dict[int, Tuple[int, int, int]] = {}
    changed = False
    for k in range(rotated.shape[0]):
        mono = i + 1 + k
        gid = c * N + mono
        new_pos = (int(rotated[k, 0]), int(rotated[k, 1]), int(rotated[k, 2]))
        old_pos = tuple(int(x) for x in state.coords_unwrapped[c, mono])
        moved[gid] = new_pos
        if new_pos != old_pos:
            changed = True
    new_sites = _check_validity(state, moved)
    if new_sites is _INVALID:
        return MoveProposal("pivot", c, ok=False, state_changing=False)
    return MoveProposal(
        "pivot", c, ok=True, state_changing=changed,
        moved=moved, new_sites=new_sites)


def propose_translation(state: MultiChainState, rng: random.Random) -> MoveProposal:
    """Translate a whole random chain by one of the six lattice unit vectors.

    Symmetric proposal (direction and its opposite are equiprobable).  The
    translating chain's own current sites are ignored in the collision test, so
    a bead may move onto a site vacated by another bead of the same chain.
    Intrachain contacts are preserved exactly; only interchain contacts change.
    """
    N = state.chain_length
    c = rng.randrange(state.n_chains)
    d = NN6[rng.randrange(6)]
    moved: Dict[int, Tuple[int, int, int]] = {}
    for mono in range(N):
        gid = c * N + mono
        p = state.coords_unwrapped[c, mono]
        moved[gid] = (int(p[0] + d[0]), int(p[1] + d[1]), int(p[2] + d[2]))
    new_sites = _check_validity(state, moved)
    if new_sites is _INVALID:
        return MoveProposal("chain_translation", c, ok=False, state_changing=False)
    # A whole-chain translation by a unit vector always changes positions.
    return MoveProposal(
        "chain_translation", c, ok=True, state_changing=True,
        moved=moved, new_sites=new_sites)


def propose_reptation(state: MultiChainState, rng: random.Random) -> MoveProposal:
    """Slithering-snake (reptation) move on a whole random chain.

    One end of chain ``c`` is REMOVED and a new bead is grown at the opposite
    end in a uniformly drawn nearest-neighbour direction; every interior bead
    inherits its neighbour's old position, so all N beads move.  Removing the
    head (index 0): ``new_r[i] = old_r[i+1]`` for ``i = 0..N-2`` and
    ``new_r[N-1] = old_r[N-1] + v``; removing the tail (index N-1):
    ``new_r[i] = old_r[i-1]`` for ``i = 1..N-1`` and ``new_r[0] = old_r[0] + v``,
    with ``v`` uniform over :data:`NN6`.

    Detailed balance: this proposal is SYMMETRIC (no Metropolis-Hastings
    correction).  The reverse of "remove head, grow tail in direction ``v``" is
    "remove tail, grow head in direction ``w = -(r_1 - r_0)``", and ``w`` is
    itself one of the six uniformly drawn NN6 directions available to the reverse
    proposal.  Because the end choice (2 options) and the direction choice (6
    options, including both ``v`` and ``-v``) are each drawn with uniform,
    geometry-independent probability regardless of whether the move turns out
    valid, the forward and reverse proposal probabilities are equal.  The sampler
    therefore accepts with plain ``delta_u <= 0 or rand < exp(-delta_u)`` -- no
    extra acceptance factor.

    All N beads go through the SAME :func:`_check_validity` used by the other
    proposers, which rejects duplicate new sites among the moved set and any new
    site occupied by a bead NOT in the moved set (i.e. another chain).  This is an
    O(N) move (same cost class as :func:`propose_translation`).
    """
    N = state.chain_length
    c = rng.randrange(state.n_chains)
    if N < 2:
        return MoveProposal("reptation", c, ok=False, state_changing=False)
    coords = state.coords_unwrapped[c].astype(np.int64)
    v = NN6[rng.randrange(6)]
    remove_head = rng.random() < 0.5
    new_coords = np.empty((N, 3), dtype=np.int64)
    if remove_head:
        new_coords[:N - 1] = coords[1:]
        new_coords[N - 1] = coords[N - 1] + np.asarray(v, dtype=np.int64)
    else:
        new_coords[1:] = coords[:N - 1]
        new_coords[0] = coords[0] + np.asarray(v, dtype=np.int64)
    moved: Dict[int, Tuple[int, int, int]] = {}
    changed = False
    for mono in range(N):
        gid = c * N + mono
        new_pos = (int(new_coords[mono, 0]), int(new_coords[mono, 1]),
                   int(new_coords[mono, 2]))
        old_pos = (int(coords[mono, 0]), int(coords[mono, 1]), int(coords[mono, 2]))
        moved[gid] = new_pos
        if new_pos != old_pos:
            changed = True
    new_sites = _check_validity(state, moved)
    if new_sites is _INVALID:
        return MoveProposal("reptation", c, ok=False, state_changing=False)
    return MoveProposal(
        "reptation", c, ok=True, state_changing=changed,
        moved=moved, new_sites=new_sites)


def propose_chain_rotation(state: MultiChainState, rng: random.Random) -> MoveProposal:
    """Rotate a WHOLE random chain rigidly about one of its own bead positions.

    Unlike :func:`propose_pivot` (which re-folds only the tail past a pivot), this
    reorients the entire chain: a pivot bead index ``j`` is chosen uniformly in
    ``[0, N-1]`` and stays fixed, while every other bead ``i`` maps to
    ``pivot + R @ (r_i - pivot)`` for a proper cubic rotation ``R``.  ``R`` is
    drawn from the SAME 24-element pool as :func:`multichain_state._random_cubic_rotation`
    (the 23 proper rotations in ``remd_uniform_chain_2_new.ROT_MATS`` plus the
    identity); that helper is bound to a NumPy RNG, so the identical 24-way draw
    is inlined here for the ``random.Random`` the proposers receive.

    Detailed balance: SYMMETRIC (no Metropolis-Hastings correction).  The reverse
    move uses the same chain ``c``, the same pivot bead index ``j`` (its position
    and identity are unchanged since it never moves), and rotation ``R^{-1}``,
    which is itself a proper cubic rotation and therefore in the same 24-element
    pool with the same 1/24 draw probability; forward and reverse proposal
    probabilities are equal.

    A rigid rotation preserves all pairwise distances, so the rotated chain is
    automatically self-avoiding and connectivity-preserving (bond vectors map
    NN6 -> NN6); only collisions with OTHER chains' stationary beads can
    invalidate it, and :func:`_check_validity` already checks exactly that.  The
    pivot bead is excluded from ``moved`` (it does not move), mirroring
    :func:`propose_pivot`.  This is an O(N) move.
    """
    N = state.chain_length
    c = rng.randrange(state.n_chains)
    if N < 2:
        return MoveProposal("rotation", c, ok=False, state_changing=False)
    j = rng.randrange(N)
    # Same 24-way draw as multichain_state._random_cubic_rotation (23 proper
    # rotations + identity), inlined for the random.Random RNG.
    idx = rng.randrange(len(_remd.ROT_MATS) + 1)
    if idx == len(_remd.ROT_MATS):
        R = np.eye(3, dtype=np.int64)
    else:
        R = np.asarray(_remd.ROT_MATS[idx], dtype=np.int64)
    coords = state.coords_unwrapped[c].astype(np.int64)
    pivot = coords[j]
    rotated = (coords - pivot) @ R.T + pivot  # integer rotation about the pivot
    moved: Dict[int, Tuple[int, int, int]] = {}
    changed = False
    for mono in range(N):
        if mono == j:
            continue
        gid = c * N + mono
        new_pos = (int(rotated[mono, 0]), int(rotated[mono, 1]),
                   int(rotated[mono, 2]))
        old_pos = (int(coords[mono, 0]), int(coords[mono, 1]), int(coords[mono, 2]))
        moved[gid] = new_pos
        if new_pos != old_pos:
            changed = True
    new_sites = _check_validity(state, moved)
    if new_sites is _INVALID:
        return MoveProposal("rotation", c, ok=False, state_changing=False)
    return MoveProposal(
        "rotation", c, ok=True, state_changing=changed,
        moved=moved, new_sites=new_sites)


LOCAL_PROPOSERS: Tuple[Callable[[MultiChainState, random.Random], MoveProposal], ...] = (
    propose_pivot, propose_crankshaft, propose_end,
)


def propose_local(state: MultiChainState, rng: random.Random) -> MoveProposal:
    """Pick one of the three local moves uniformly and propose it."""
    return LOCAL_PROPOSERS[rng.randrange(len(LOCAL_PROPOSERS))](state, rng)


# ---------------------------------------------------------------------------
# Delta / apply helpers (used by the sampler and by tests)
# ---------------------------------------------------------------------------

def proposal_delta(state: MultiChainState, proposal: MoveProposal
                   ) -> Tuple[int, int]:
    """(delta_intra, delta_inter) for a geometrically valid proposal."""
    moved_ids = list(proposal.moved.keys())
    return delta_contacts(state, moved_ids, proposal.new_sites)


def proposal_delta_bends(state: MultiChainState, proposal: MoveProposal) -> int:
    """Change in the total 90-degree-turn count for a geometrically valid proposal.

    Whole-chain translation and rigid whole-chain rotation are isometries of the
    moved chain (a translation preserves every bond vector; a rigid cubic rotation
    maps NN6 bonds to NN6 bonds and preserves every interior angle), so they never
    change any bend and return exactly 0 without inspecting geometry.  Local bead
    moves and reptation can change bends only WITHIN the single moved chain, so the
    delta is evaluated from that one chain's old vs new UNWRAPPED coordinates over
    the moved-bead angle centers (via :func:`lattice_bending.delta_bends`) -- never
    a full recount over all chains.
    """
    if not proposal.ok:
        return 0
    if proposal.move_type in ("chain_translation", "rotation"):
        return 0
    N = state.chain_length
    if N < 3 or not proposal.moved:
        return 0
    c = int(proposal.chain)
    old = state.coords_unwrapped[c]
    new = np.array(old, dtype=np.int64, copy=True)
    changed = []
    for gid, pos in proposal.moved.items():
        i = int(gid) % N
        new[i, 0], new[i, 1], new[i, 2] = int(pos[0]), int(pos[1]), int(pos[2])
        changed.append(i)
    return int(_delta_bends(old, new, changed))


def apply_proposal(state: MultiChainState, proposal: MoveProposal,
                   delta: Tuple[int, int], delta_bends: int = None) -> None:
    """Apply an ACCEPTED proposal's new positions and update caches in place.

    After committing the moved beads, the proposal's chain is re-based into the
    primary periodic box (see :func:`canonicalize_chain_coordinates`) so that a
    long sequence of accepted whole-chain translations cannot drive unwrapped
    coordinates off to infinity.  Canonicalization runs only here, on the
    accepted state -- never on trial coordinates before the contact-delta
    evaluation -- and preserves wrapped sites, occupancy, contacts, Rg, and (being
    a whole-chain box-multiple shift) the bend count too.

    ``delta_bends`` is the change in the cached total 90-degree-turn count.  The
    sampler passes the value it already computed for the Metropolis decision; when
    omitted (direct callers / tests) it is recomputed from the pre-move state, so
    ``state.n_bend`` stays in sync however the proposal is applied.
    """
    if delta_bends is None:
        delta_bends = proposal_delta_bends(state, proposal)
    apply_moved_beads(state, proposal.moved, delta)
    canonicalize_chain_coordinates(state, proposal.chain)
    state.n_bend = int(state.n_bend) + int(delta_bends)
