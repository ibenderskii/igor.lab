#!/usr/bin/env python3
"""pytest suite for multichain_state (Stage 1: state and geometry).

Run:  python -m pytest test_multichain_state.py -q
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import multichain_state as mcs


# ---------------------------------------------------------------------------
# Hand-built conformations
# ---------------------------------------------------------------------------

def straight_chain(n, axis=0, start=(0, 0, 0)):
    coords = np.tile(np.asarray(start, dtype=np.int64), (n, 1))
    coords[:, axis] += np.arange(n, dtype=np.int64)
    return coords


HAIRPIN6 = np.array(
    [(0, 0, 0), (1, 0, 0), (2, 0, 0), (2, 1, 0), (1, 1, 0), (0, 1, 0)],
    dtype=np.int64)


# ---------------------------------------------------------------------------
# Wrapping / occupancy
# ---------------------------------------------------------------------------

def test_wrap_coordinate_negative_and_positive():
    assert mcs.wrap_coordinate((-1, 0, 5), 5) == (4, 0, 0)
    assert mcs.wrap_coordinate((7, -3, 10), 5) == (2, 2, 0)
    assert mcs.wrap_coordinate((0, 0, 0), 3) == (0, 0, 0)


def test_wrapped_coordinates_array():
    coords = np.array([[[-1, 0, 6], [3, 3, 3]]], dtype=np.int64)
    w = mcs.wrapped_coordinates(coords, 3)
    assert w.shape == coords.shape
    assert np.array_equal(w, np.array([[[2, 0, 0], [0, 0, 0]]]))
    assert w.min() >= 0 and w.max() < 3


def test_build_site_owner_ids_and_bijection():
    # Two length-3 chains, well separated in a big box.
    coords = np.stack([
        straight_chain(3, axis=0, start=(0, 0, 0)),
        straight_chain(3, axis=0, start=(0, 5, 0)),
    ])
    so = mcs.build_site_owner(coords, 20)
    assert len(so) == 6
    # global_id = chain*N + monomer
    assert so[(0, 0, 0)] == 0
    assert so[(2, 0, 0)] == 2
    assert so[(0, 5, 0)] == 3
    assert sorted(so.values()) == list(range(6))


def test_build_site_owner_rejects_double_occupancy():
    # Two chains sharing a wrapped site.
    coords = np.stack([
        straight_chain(3, axis=0, start=(0, 0, 0)),
        straight_chain(3, axis=0, start=(0, 0, 0)),
    ])
    with pytest.raises(mcs.MultiChainStateError):
        mcs.build_site_owner(coords, 20)


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------

def test_validate_connectivity_ok_and_broken():
    coords = np.stack([straight_chain(5)])
    mcs.validate_connectivity(coords)  # no raise
    broken = coords.copy()
    broken[0, 2] += np.array([0, 3, 0])  # snap a bond
    with pytest.raises(mcs.MultiChainStateError):
        mcs.validate_connectivity(broken)


def test_validate_unique_occupancy_detects_wrapped_overlap():
    # Straight length-4 chain in a box of side 3 wraps onto itself: bead 3 at
    # x=3 wraps to x=0 == bead 0.
    coords = np.stack([straight_chain(4)])
    with pytest.raises(mcs.MultiChainStateError):
        mcs.validate_unique_occupancy(coords, 3)


def test_box_size_below_three_rejected():
    coords = np.stack([straight_chain(3)])
    with pytest.raises(mcs.MultiChainStateError):
        mcs.make_state(coords, 2)


def test_fractional_coordinates_rejected():
    coords = np.stack([straight_chain(3)]).astype(float)
    coords[0, 1, 0] += 0.5
    with pytest.raises(mcs.MultiChainStateError):
        mcs._validate_coords_array(coords)


def test_validate_state_full_and_contacts():
    coords = np.stack([HAIRPIN6])
    state = mcs.make_state(coords, 20)
    mcs.validate_state(state)  # includes cached-count recount
    # Corrupt the cached counts -> validate_state must catch it.
    state.counts.intra += 5
    with pytest.raises(mcs.MultiChainStateError):
        mcs.validate_state(state)


def test_validate_state_rejects_desynced_site_owner():
    coords = np.stack([straight_chain(4)])
    state = mcs.make_state(coords, 20)
    # Remove an entry so site_owner disagrees with coordinates.
    some_site = next(iter(state.site_owner))
    del state.site_owner[some_site]
    with pytest.raises(mcs.MultiChainStateError):
        mcs.validate_state(state, check_contacts=False)


# ---------------------------------------------------------------------------
# Geometry (unwrapped is authoritative; correct across boundaries)
# ---------------------------------------------------------------------------

def test_per_chain_rg_straight_matches_analytic():
    # Rg^2 of a straight N-mer along an axis is (N^2 - 1)/12.
    coords = np.stack([straight_chain(5)])
    state = mcs.make_state(coords, 50)
    rg2 = mcs.per_chain_rg2(state)
    assert rg2.shape == (1,)
    assert abs(rg2[0] - (5 ** 2 - 1) / 12.0) < 1e-9


def test_rg_correct_for_boundary_crossing_chain():
    # A straight 3-mer growing in -x crosses the periodic boundary when wrapped
    # (wrapped sites 0,5,4 in L=6) but its UNWRAPPED Rg must equal the ordinary
    # straight 3-mer value (N^2-1)/12 = 8/12 = 2/3.
    coords = np.array([[[0, 0, 0], [-1, 0, 0], [-2, 0, 0]]], dtype=np.int64)
    state = mcs.make_state(coords, 6)
    mcs.validate_state(state)  # wrapped occupancy is valid (no overlap)
    rg2 = mcs.per_chain_rg2(state)[0]
    assert abs(rg2 - (3 ** 2 - 1) / 12.0) < 1e-9
    # Wrapped sites confirm the chain crosses the boundary.
    w = state.wrapped()[0]
    sites = {tuple(int(v) for v in row) for row in w}
    assert (5, 0, 0) in sites and (4, 0, 0) in sites


# ---------------------------------------------------------------------------
# Dispersed initialization
# ---------------------------------------------------------------------------

def test_initialize_dispersed_state_valid():
    state = mcs.initialize_dispersed_state(
        n_chains=4, chain_length=8, box_size=12, seed=123)
    assert state.n_chains == 4
    assert state.chain_length == 8
    assert state.n_beads == 32
    mcs.validate_state(state)
    # Volume fraction phi = M N / L^3.
    assert abs(state.volume_fraction - 32.0 / 12 ** 3) < 1e-12


def test_initialize_dispersed_state_deterministic_from_seed():
    a = mcs.initialize_dispersed_state(3, 6, 12, seed=7)
    b = mcs.initialize_dispersed_state(3, 6, 12, seed=7)
    assert np.array_equal(a.coords_unwrapped, b.coords_unwrapped)
    assert a.counts.as_tuple() == b.counts.as_tuple()
    # A different seed generally differs.
    c = mcs.initialize_dispersed_state(3, 6, 12, seed=8)
    assert not np.array_equal(a.coords_unwrapped, c.coords_unwrapped)


def test_initialize_dispersed_state_not_all_straight():
    # A dispersed state should not place every chain as an identical straight
    # line; at least one chain should bend (have a nonzero turn).
    state = mcs.initialize_dispersed_state(4, 10, 14, seed=42)
    bent = 0
    for c in range(state.n_chains):
        diffs = np.diff(state.coords_unwrapped[c], axis=0)
        # A straight chain has all identical bond directions.
        if not np.all(diffs == diffs[0]):
            bent += 1
    assert bent >= 1


def test_initialize_dispersed_state_no_double_occupancy():
    state = mcs.initialize_dispersed_state(6, 6, 10, seed=99)
    w = state.wrapped().reshape(-1, 3)
    uniq = np.unique(w, axis=0)
    assert uniq.shape[0] == w.shape[0]


def test_initialize_rejects_overfull_box():
    with pytest.raises(mcs.MultiChainStateError):
        mcs.initialize_dispersed_state(n_chains=30, chain_length=30, box_size=3,
                                       seed=1)


# ---------------------------------------------------------------------------
# Unwrapped-coordinate canonicalization (Change 1)
# ---------------------------------------------------------------------------

def test_canonicalize_chain_coordinates_preserves_invariants():
    # A whole-box integer shift of one chain leaves the wrapped configuration
    # (and hence occupancy, contacts, Rg, reduced potential) unchanged;
    # canonicalization must undo the drift without disturbing any of them.
    state = mcs.initialize_dispersed_state(3, 6, 10, seed=21)
    L = state.box_size
    c = 1

    wrapped_before = state.wrapped().copy()
    site_owner_before = dict(state.site_owner)
    counts_before = state.counts.as_tuple()
    rg2_before = mcs.per_chain_rg2(state).copy()

    # Drive chain c far out of the primary box by whole box multiples per axis.
    state.coords_unwrapped[c] += np.array([3 * L, -2 * L, 4 * L], dtype=np.int64)
    # A whole-box shift changes nothing wrapped.
    assert np.array_equal(state.wrapped(), wrapped_before)

    mcs.canonicalize_chain_coordinates(state, c)

    # Bead 0 of the canonicalized chain lies in [0, L)^3.
    b0 = state.coords_unwrapped[c, 0]
    assert np.all((b0 >= 0) & (b0 < L))
    # Wrapped coords, occupancy map, cached counts, and per-chain Rg all preserved.
    assert np.array_equal(state.wrapped(), wrapped_before)
    assert dict(state.site_owner) == site_owner_before
    assert state.counts.as_tuple() == counts_before
    np.testing.assert_allclose(mcs.per_chain_rg2(state), rg2_before)
    # Connectivity and the full cached-vs-oracle contact recount still hold.
    mcs.validate_connectivity(state.coords_unwrapped)
    mcs.validate_state(state)


def test_canonicalize_noop_when_already_in_box():
    state = mcs.initialize_dispersed_state(2, 5, 12, seed=3)
    coords_before = state.coords_unwrapped.copy()
    for c in range(state.n_chains):
        # initialize_dispersed_state already places bead 0 inside the box.
        mcs.canonicalize_chain_coordinates(state, c)
    assert np.array_equal(state.coords_unwrapped, coords_before)


if __name__ == "__main__":
    sys.exit(pytest.main([os.path.abspath(__file__), "-q"]))
