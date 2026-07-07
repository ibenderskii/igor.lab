#!/usr/bin/env python3
"""pytest suite for multichain_contacts (Stage 1: contact oracle + deltas).

Run:  python -m pytest test_multichain_contacts.py -q
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import multichain_state as mcs
import multichain_contacts as mcc
import remd_uniform_chain_2_new as remd


def straight_chain(n, axis=0, start=(0, 0, 0)):
    coords = np.tile(np.asarray(start, dtype=np.int64), (n, 1))
    coords[:, axis] += np.arange(n, dtype=np.int64)
    return coords


HAIRPIN6 = np.array(
    [(0, 0, 0), (1, 0, 0), (2, 0, 0), (2, 1, 0), (1, 1, 0), (0, 1, 0)],
    dtype=np.int64)


def brute_force_counts(state):
    """Independent O((MN)^2) min-image contact oracle for cross-checking."""
    coords = state.coords_unwrapped
    M, N, _ = coords.shape
    L = int(state.box_size)
    w = np.mod(coords, L).reshape(-1, 3)
    intra = 0
    inter = 0
    n = M * N
    for a in range(n):
        for b in range(a + 1, n):
            d = 0
            for k in range(3):
                dd = abs(int(w[a, k]) - int(w[b, k])) % L
                d += min(dd, L - dd)
            if d != 1:
                continue
            ca, cb = a // N, b // N
            if ca == cb:
                if abs((a % N) - (b % N)) == 1:
                    continue
                intra += 1
            else:
                inter += 1
    return intra, inter


# ---------------------------------------------------------------------------
# Full oracle: hand-designed configurations
# ---------------------------------------------------------------------------

def test_two_separated_chains_no_contacts():
    coords = np.stack([
        straight_chain(4, axis=0, start=(0, 0, 0)),
        straight_chain(4, axis=0, start=(0, 8, 0)),
    ])
    state = mcs.make_state(coords, 30)
    c = mcc.full_contact_counts(state)
    assert c.intra == 0 and c.inter == 0


def test_one_interchain_contact():
    # Chain A along x (len 2); chain B along y (len 2) touching A at one bead.
    A = np.array([(0, 0, 0), (1, 0, 0)], dtype=np.int64)
    B = np.array([(0, 1, 0), (0, 2, 0)], dtype=np.int64)
    state = mcs.make_state(np.stack([A, B]), 20)
    c = mcc.full_contact_counts(state)
    assert c.intra == 0
    assert c.inter == 1
    assert (c.intra, c.inter) == brute_force_counts(state)


def test_multiple_parallel_interchain_contacts():
    # Two parallel straight chains offset by 1 in y: N interchain contacts.
    N = 5
    A = straight_chain(N, axis=0, start=(0, 0, 0))
    B = straight_chain(N, axis=0, start=(0, 1, 0))
    state = mcs.make_state(np.stack([A, B]), 20)
    c = mcc.full_contact_counts(state)
    assert c.intra == 0
    assert c.inter == N
    assert (c.intra, c.inter) == brute_force_counts(state)


def test_known_intrachain_nonbonded_contact():
    # Hairpin: contacts (0,5) r=5 and (1,4) r=3 -> m_intra = 2.
    state = mcs.make_state(np.stack([HAIRPIN6]), 20)
    c = mcc.full_contact_counts(state)
    assert c.intra == 2
    assert c.inter == 0


def test_bonded_neighbors_excluded():
    # A straight chain: every occupied nearest neighbour is a covalent bond.
    state = mcs.make_state(np.stack([straight_chain(10)]), 30)
    c = mcc.full_contact_counts(state)
    assert c.intra == 0 and c.inter == 0


def test_contact_across_periodic_boundary():
    # Chain A at x=0, chain B at x=L-1: they touch across the x-boundary.
    L = 6
    A = np.array([(0, 0, 0), (0, 1, 0)], dtype=np.int64)
    B = np.array([(L - 1, 0, 0), (L - 1, 0, 1)], dtype=np.int64)
    state = mcs.make_state(np.stack([A, B]), L)
    c = mcc.full_contact_counts(state)
    assert c.inter == 1  # only A0-(B0) across the boundary
    assert c.intra == 0
    assert (c.intra, c.inter) == brute_force_counts(state)


def test_no_double_counting_against_bruteforce():
    # Dense-ish random state; oracle must equal the independent brute force.
    for seed in range(6):
        state = mcs.initialize_dispersed_state(4, 8, 10, seed=seed)
        c = mcc.full_contact_counts(state)
        assert (c.intra, c.inter) == brute_force_counts(state)


def test_simultaneous_intra_and_inter():
    # Two identical hairpins stacked in z (chains must share length N).
    # Each hairpin has m_intra = 2 (contacts (0,5) and (1,4)); stacking them one
    # lattice step apart in z creates one interchain contact per bead pair (6).
    A = HAIRPIN6
    B = HAIRPIN6 + np.array([0, 0, 1], dtype=np.int64)
    state = mcs.make_state(np.stack([A, B]), 20)
    c = mcc.full_contact_counts(state)
    assert c.intra == 4  # 2 per hairpin
    assert c.inter == 6  # one per stacked bead pair
    assert (c.intra, c.inter) == brute_force_counts(state)


# ---------------------------------------------------------------------------
# M = 1 agreement with the original single-chain contact_count
# ---------------------------------------------------------------------------

def test_m1_matches_original_contact_count():
    rng = np.random.RandomState(2024)
    for _ in range(20):
        saw = mcs.generate_saw(15, rng)
        state = mcs.make_state(np.stack([saw]), 60)  # big box: no image contact
        c = mcc.full_contact_counts(state)
        chain = [tuple(int(v) for v in row) for row in saw]
        m_ref = int(round(remd.contact_count(chain, set(chain))))
        assert c.inter == 0
        assert c.intra == m_ref


# ---------------------------------------------------------------------------
# Delta contacts agree with full recount
# ---------------------------------------------------------------------------

def test_delta_matches_recount_single_bead():
    # Move one end bead and compare delta to the full-recount difference.
    A = straight_chain(6, axis=0, start=(0, 0, 0))
    B = straight_chain(6, axis=0, start=(0, 2, 0))
    state = mcs.make_state(np.stack([A, B]), 20)
    before = mcc.full_contact_counts(state)

    # Move chain 0's last bead (global id 5) from (5,0,0) to (5,1,0).
    gid = 5
    new_unwrapped = (5, 1, 0)
    new_site = mcs.wrap_coordinate(new_unwrapped, state.box_size)
    d_intra, d_inter = mcc.delta_contacts(state, [gid], {gid: new_site})
    mcc.apply_moved_beads(state, {gid: new_unwrapped}, (d_intra, d_inter))

    after = mcc.full_contact_counts(state)
    assert state.counts.as_tuple() == (after.intra, after.inter)
    assert d_intra == after.intra - before.intra
    assert d_inter == after.inter - before.inter


def test_delta_matches_recount_random_moves():
    # Random single-bead relocations; delta must always track the full recount.
    rng = np.random.RandomState(11)
    state = mcs.initialize_dispersed_state(3, 8, 12, seed=5)
    for _ in range(300):
        c = rng.randint(state.n_chains)
        # Move an end bead to a random empty neighbour of its anchor.
        end = 0 if rng.rand() < 0.5 else state.chain_length - 1
        anchor = 1 if end == 0 else state.chain_length - 2
        gid = c * state.chain_length + end
        d = mcs.NN6[rng.randint(6)]
        new_unwrapped = tuple(int(state.coords_unwrapped[c, anchor, k] + d[k])
                              for k in range(3))
        new_site = mcs.wrap_coordinate(new_unwrapped, state.box_size)
        old_site = mcc._wrapped_site(state, gid)
        if new_site == old_site:
            continue
        # Reject if occupied by another bead (excluded volume).
        owner = state.site_owner.get(new_site)
        if owner is not None and owner != gid:
            continue
        before = state.counts.copy()
        d_intra, d_inter = mcc.delta_contacts(state, [gid], {gid: new_site})
        mcc.apply_moved_beads(state, {gid: new_unwrapped}, (d_intra, d_inter))
        recount = mcc.full_contact_counts(state)
        assert state.counts.as_tuple() == (recount.intra, recount.inter), (
            f"delta desync: cached {state.counts.as_tuple()} vs "
            f"recount {(recount.intra, recount.inter)} (before {before.as_tuple()})")


def test_interchain_pair_counts():
    # Chain 0 parallel to chain 1 (5 contacts); chain 2 far away.
    N = 5
    c0 = straight_chain(N, axis=0, start=(0, 0, 0))
    c1 = straight_chain(N, axis=0, start=(0, 1, 0))
    c2 = straight_chain(N, axis=0, start=(0, 10, 0))
    state = mcs.make_state(np.stack([c0, c1, c2]), 30)
    pairs = mcc.interchain_pair_counts(state)
    assert pairs == {(0, 1): 5}


if __name__ == "__main__":
    sys.exit(pytest.main([os.path.abspath(__file__), "-q"]))
