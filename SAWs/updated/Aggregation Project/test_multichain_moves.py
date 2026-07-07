#!/usr/bin/env python3
"""pytest suite for multichain_moves (Stage 2: Monte Carlo moves).

Run:  python -m pytest test_multichain_moves.py -q
"""
import os
import random
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import multichain_state as mcs
import multichain_contacts as mcc
import multichain_moves as mvs


def straight_chain(n, axis=0, start=(0, 0, 0)):
    coords = np.tile(np.asarray(start, dtype=np.int64), (n, 1))
    coords[:, axis] += np.arange(n, dtype=np.int64)
    return coords


def chain_com(state, c):
    return state.coords_unwrapped[c].mean(axis=0)


# ---------------------------------------------------------------------------
# Per-move invariants: connectivity, occupancy, delta agreement
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("proposer,name", [
    (mvs.propose_end, "end"),
    (mvs.propose_crankshaft, "crankshaft"),
    (mvs.propose_pivot, "pivot"),
    (mvs.propose_translation, "chain_translation"),
])
def test_move_preserves_invariants_and_delta(proposer, name):
    rng = random.Random(1234)
    state = mcs.initialize_dispersed_state(3, 8, 12, seed=17)
    n_valid = 0
    for _ in range(2000):
        prop = proposer(state, rng)
        assert prop.move_type == name
        if not prop.ok:
            continue
        n_valid += 1
        before = state.counts.copy()
        d = mvs.proposal_delta(state, prop)
        mvs.apply_proposal(state, prop, d)
        # Connectivity preserved (unit bonds in unwrapped coordinates).
        mcs.validate_connectivity(state.coords_unwrapped)
        # Occupancy + cached counts consistent with a full recount.
        mcs.validate_state(state)
        recount = mcc.full_contact_counts(state)
        assert state.counts.as_tuple() == (recount.intra, recount.inter)
        assert d[0] == recount.intra - before.intra
        assert d[1] == recount.inter - before.inter
    assert n_valid > 0, f"{name} never produced a valid move"


def test_null_move_not_state_changing():
    # An end move whose random neighbour lands back on the bead's own site is a
    # null move: geometrically valid but not state-changing.
    rng = random.Random(0)
    state = mcs.make_state(np.stack([straight_chain(5)]), 30)
    seen_null = False
    for _ in range(500):
        prop = mvs.propose_end(state, rng)
        if prop.ok and not prop.state_changing:
            seen_null = True
            # The moved bead's new position equals its old position.
            gid = next(iter(prop.moved))
            c, i = state.chain_of(gid), state.monomer_of(gid)
            assert tuple(prop.moved[gid]) == tuple(
                int(x) for x in state.coords_unwrapped[c, i])
            break
    assert seen_null, "expected at least one null end move on a straight chain"


# ---------------------------------------------------------------------------
# Whole-chain translation specifics
# ---------------------------------------------------------------------------

def test_translation_preserves_intrachain_contacts():
    # A hairpin (m_intra=2) plus a spectator chain; translate the hairpin and
    # confirm intrachain contacts never change.
    hairpin = np.array(
        [(0, 0, 0), (1, 0, 0), (2, 0, 0), (2, 1, 0), (1, 1, 0), (0, 1, 0)],
        dtype=np.int64)
    spectator = straight_chain(6, axis=0, start=(0, 6, 0))
    state = mcs.make_state(np.stack([hairpin, spectator]), 20)
    intra0 = state.counts.intra
    rng = random.Random(3)
    n_trans = 0
    for _ in range(500):
        prop = mvs.propose_translation(state, rng)
        if not prop.ok:
            continue
        d = mvs.proposal_delta(state, prop)
        assert d[0] == 0, "translation changed intrachain contacts"
        mvs.apply_proposal(state, prop, d)
        assert state.counts.intra == intra0
        n_trans += 1
    assert n_trans > 0


def test_translation_interchain_updates_correctly():
    # Two parallel chains (5 interchain contacts). Translating one apart in +z
    # removes all interchain contacts.
    c0 = straight_chain(5, axis=0, start=(0, 0, 0))
    c1 = straight_chain(5, axis=0, start=(0, 1, 0))
    state = mcs.make_state(np.stack([c0, c1]), 20)
    assert state.counts.inter == 5
    # Build the specific +z translation of chain 0.
    N = state.chain_length
    d = (0, 0, 1)
    moved = {0 * N + m: tuple(int(state.coords_unwrapped[0, m, k] + d[k])
                              for k in range(3)) for m in range(N)}
    new_sites = mvs._check_validity(state, moved)
    assert new_sites is not None
    prop = mvs.MoveProposal("chain_translation", 0, ok=True, state_changing=True,
                            moved=moved, new_sites=new_sites)
    delta = mvs.proposal_delta(state, prop)
    mvs.apply_proposal(state, prop, delta)
    assert state.counts.inter == 0
    assert state.counts.intra == 0
    mcs.validate_state(state)


def test_translation_through_periodic_boundary():
    # Chain against the +x boundary; translate +x so it wraps to x=0.
    L = 6
    c0 = straight_chain(3, axis=1, start=(L - 1, 0, 0))  # along y at x=L-1
    c1 = straight_chain(3, axis=1, start=(2, 0, 0))
    state = mcs.make_state(np.stack([c0, c1]), L)
    N = state.chain_length
    d = (1, 0, 0)
    moved = {0 * N + m: tuple(int(state.coords_unwrapped[0, m, k] + d[k])
                              for k in range(3)) for m in range(N)}
    new_sites = mvs._check_validity(state, moved)
    assert new_sites is not None
    prop = mvs.MoveProposal("chain_translation", 0, ok=True, state_changing=True,
                            moved=moved, new_sites=new_sites)
    delta = mvs.proposal_delta(state, prop)
    mvs.apply_proposal(state, prop, delta)
    mcs.validate_state(state)
    # Chain 0 now occupies wrapped x=0.
    w = state.wrapped()[0]
    assert np.all(w[:, 0] == 0)


def test_translation_collision_rejected():
    # Two adjacent parallel chains; translating chain 0 by +y lands on chain 1.
    c0 = straight_chain(4, axis=0, start=(0, 0, 0))
    c1 = straight_chain(4, axis=0, start=(0, 1, 0))
    state = mcs.make_state(np.stack([c0, c1]), 20)
    N = state.chain_length
    into = {0 * N + m: tuple(int(state.coords_unwrapped[0, m, k] + (0, 1, 0)[k])
                             for k in range(3)) for m in range(N)}
    assert mvs._check_validity(state, into) is None  # collides with chain 1
    away = {0 * N + m: tuple(int(state.coords_unwrapped[0, m, k] + (0, 0, 1)[k])
                             for k in range(3)) for m in range(N)}
    assert mvs._check_validity(state, away) is not None  # free direction


def test_translation_reverse_symmetry():
    # Translating by d then by -d returns to the original configuration.
    state = mcs.make_state(
        np.stack([straight_chain(4, start=(2, 2, 2)),
                  straight_chain(4, start=(2, 6, 2))]), 20)
    coords0 = state.coords_unwrapped.copy()
    N = state.chain_length
    for d, dinv in [((1, 0, 0), (-1, 0, 0))]:
        fwd = {m: tuple(int(state.coords_unwrapped[0, m, k] + d[k])
                        for k in range(3)) for m in range(N)}
        ns = mvs._check_validity(state, fwd)
        assert ns is not None
        p = mvs.MoveProposal("chain_translation", 0, True, True, fwd, ns)
        mvs.apply_proposal(state, p, mvs.proposal_delta(state, p))
        rev = {m: tuple(int(state.coords_unwrapped[0, m, k] + dinv[k])
                        for k in range(3)) for m in range(N)}
        ns2 = mvs._check_validity(state, rev)
        assert ns2 is not None
        p2 = mvs.MoveProposal("chain_translation", 0, True, True, rev, ns2)
        mvs.apply_proposal(state, p2, mvs.proposal_delta(state, p2))
    assert np.array_equal(state.coords_unwrapped, coords0)


def test_translation_canonicalization_keeps_anchor_in_box():
    # Accepting a long stream of whole-chain translations must NOT let unwrapped
    # coordinates run away: apply_proposal canonicalizes the moved chain, so every
    # chain's anchor bead 0 stays inside the primary box after every accepted move.
    rng = random.Random(5)
    state = mcs.initialize_dispersed_state(2, 6, 12, seed=9)
    L = state.box_size
    n_moves = 0
    for _ in range(5000):
        prop = mvs.propose_translation(state, rng)
        if not prop.ok:
            continue
        mvs.apply_proposal(state, prop, mvs.proposal_delta(state, prop))
        n_moves += 1
        for c in range(state.n_chains):
            b0 = state.coords_unwrapped[c, 0]
            assert np.all((b0 >= 0) & (b0 < L)), (
                f"chain {c} anchor {tuple(int(v) for v in b0)} left [0,{L})^3")
    assert n_moves > 100
    mcs.validate_state(state)


def test_translation_allows_com_diffusion_athermal():
    # Accept every valid translation (athermal limit): a chain's center of mass
    # must move away from its start over many attempts.
    rng = random.Random(7)
    state = mcs.initialize_dispersed_state(2, 6, 16, seed=3)
    com0 = chain_com(state, 0).copy()
    moved_any = False
    for _ in range(3000):
        prop = mvs.propose_translation(state, rng)
        if prop.chain != 0 or not prop.ok:
            continue
        mvs.apply_proposal(state, prop, mvs.proposal_delta(state, prop))
        moved_any = True
    com1 = chain_com(state, 0)
    assert moved_any
    assert not np.allclose(com0, com1), "chain 0 COM did not diffuse"


# ---------------------------------------------------------------------------
# Stress test: cached counts must track the oracle over many moves (debug)
# ---------------------------------------------------------------------------

def test_stress_delta_matches_oracle_debug():
    rng = random.Random(2718)
    state = mcs.initialize_dispersed_state(2, 6, 6, seed=1)  # small, dense-ish
    n_attempts = 0
    n_accepted = 0
    target = 100_000
    while n_attempts < target:
        n_attempts += 1
        if rng.random() < 0.5:
            prop = mvs.propose_local(state, rng)
        else:
            prop = mvs.propose_translation(state, rng)
        if not prop.ok or not prop.state_changing:
            continue
        d = mvs.proposal_delta(state, prop)
        mvs.apply_proposal(state, prop, d)
        n_accepted += 1
        # Check cached counts against the full oracle after EVERY accepted move.
        recount = mcc.full_contact_counts(state)
        assert state.counts.as_tuple() == (recount.intra, recount.inter), (
            f"cache/oracle mismatch after {n_accepted} accepted moves")
    assert n_accepted > 100, "stress test accepted implausibly few moves"


if __name__ == "__main__":
    sys.exit(pytest.main([os.path.abspath(__file__), "-q"]))
