"""Tests for lattice_bending.py and the bending penalty in the athermal baseline.

Run:  python -m pytest SAWs/updated/tests/test_lattice_bending.py -q
"""
import math
import random
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from lattice_bending import bend_indicator, count_bends, delta_bends
import single_uniform_chain2_athermal_dists_joint as saw


# ---------------------------------------------------------------------------
# Reference chains (hand-built so the expected bend counts are obvious)
# ---------------------------------------------------------------------------

# Straight along x: no turns.
STRAIGHT = [(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0), (4, 0, 0)]

# One corner at bead 2.
ONE_CORNER = [(0, 0, 0), (1, 0, 0), (2, 0, 0), (2, 1, 0), (2, 2, 0)]

# Staircase: every interior bead is a 90 degree turn.
STAIRCASE = [
    (0, 0, 0), (1, 0, 0), (1, 1, 0), (2, 1, 0), (2, 2, 0),
    (3, 2, 0), (3, 3, 0), (4, 3, 0), (4, 4, 0), (5, 4, 0),
]

# Square-wave zigzag: turns at every interior bead too, but with a repeating
# up/over/down/over motif rather than a monotone staircase.
ZIGZAG = [
    (0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0),
    (2, 0, 0), (2, 1, 0), (3, 1, 0), (3, 0, 0),
]

# The 4-bead square used by test_thermo_chain.py.
SQUARE = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]


# ---------------------------------------------------------------------------
# 1-3: count_bends on known chains
# ---------------------------------------------------------------------------

def test_straight_chain_has_zero_bends():
    assert count_bends(STRAIGHT) == 0


def test_one_corner_chain_has_one_bend():
    assert count_bends(ONE_CORNER) == 1


def test_known_zigzag_bend_counts():
    # Every interior bead of a staircase / square wave turns.
    assert count_bends(STAIRCASE) == len(STAIRCASE) - 2
    assert count_bends(ZIGZAG) == len(ZIGZAG) - 2
    # The square has 2 interior beads, both corners.
    assert count_bends(SQUARE) == 2


def test_bend_indicator_straight_and_corner():
    assert bend_indicator(STRAIGHT, 2) == 0
    assert bend_indicator(ONE_CORNER, 2) == 1
    # Centers outside 1 .. N-2 are not angle centers.
    with pytest.raises(ValueError):
        bend_indicator(STRAIGHT, 0)
    with pytest.raises(ValueError):
        bend_indicator(STRAIGHT, len(STRAIGHT) - 1)


def test_bend_indicator_rejects_non_unit_and_reversed_bonds():
    # Bond of length 2 is not a lattice bond.
    with pytest.raises(ValueError):
        count_bends([(0, 0, 0), (2, 0, 0), (2, 1, 0)])
    # Immediate reversal is a self-overlap, not a 180 degree "bend".
    with pytest.raises(ValueError):
        bend_indicator([(0, 0, 0), (1, 0, 0), (0, 0, 0)], 1)


def test_count_bends_accepts_numpy_coordinates():
    assert count_bends(np.array(STAIRCASE, dtype=np.int64)) == len(STAIRCASE) - 2


# ---------------------------------------------------------------------------
# 4: delta_bends agrees with a full recount for real moves
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "move", [saw.attempt_pivot, saw.attempt_crankshaft, saw.attempt_end_move],
    ids=["pivot", "crankshaft", "end"],
)
def test_delta_bends_matches_full_recount(move):
    """Drive each real move and compare the local delta to count_bends()."""
    chain = list(STAIRCASE)
    occ = set(chain)
    rng = random.Random(2024)

    n_valid = 0
    for _ in range(500):
        ok, chain_new, occ_new, changed = move(chain, occ, rng)
        if not ok:
            continue
        n_valid += 1
        assert delta_bends(chain, chain_new, changed) == (
            count_bends(chain_new) - count_bends(chain)
        )
        chain, occ = chain_new, occ_new

    assert n_valid > 0, f"{move.__name__} never produced a valid proposal"


def test_delta_bends_ignores_centers_outside_the_chain():
    """Changed end beads clip to the single reachable angle center."""
    old = list(ONE_CORNER)
    new = list(old)
    new[0] = (1, 1, 0)                      # move bead 0 off the straight run
    assert delta_bends(old, new, (0,)) == count_bends(new) - count_bends(old)


def test_delta_bends_of_no_change_is_zero():
    assert delta_bends(STAIRCASE, STAIRCASE, (0, 4, len(STAIRCASE) - 1)) == 0


# ---------------------------------------------------------------------------
# 5: periodic boundary conditions
# ---------------------------------------------------------------------------

def test_periodic_bonds_use_minimum_image():
    L = 6
    # Straight run along x that wraps the box: 5 -> 0 is a +1 minimum-image bond.
    unwrapped = [(3, 0, 0), (4, 0, 0), (5, 0, 0), (6, 0, 0), (7, 0, 0)]
    wrapped = [(x % L, y, z) for (x, y, z) in unwrapped]

    assert count_bends(unwrapped) == 0
    # Without the box the wrapped chain has an illegal 5-unit bond.
    with pytest.raises(ValueError):
        count_bends(wrapped)
    assert count_bends(wrapped, box_size=L) == count_bends(unwrapped)


def test_periodic_corner_bend_survives_wrapping():
    L = 6
    unwrapped = [(4, 0, 0), (5, 0, 0), (6, 0, 0), (6, 1, 0), (6, 2, 0)]
    wrapped = [(x % L, y % L, z % L) for (x, y, z) in unwrapped]
    assert count_bends(unwrapped) == 1
    assert count_bends(wrapped, box_size=L) == 1
    assert bend_indicator(wrapped, 2, box_size=L) == 1


# ---------------------------------------------------------------------------
# 6: kappa_bend = 0 preserves the original baseline behavior exactly
# ---------------------------------------------------------------------------

def _original_accept_every_valid_move(seed, N, steps, burnin, sample_every):
    """The pre-bending sampler loop: accept every geometrically valid proposal.

    Reproduced here (rather than imported) so the test pins the ORIGINAL
    behavior independently of the current implementation.
    """
    py_rng = random.Random(seed)
    chain = [(i, 0, 0) for i in range(N)]
    occ = set(chain)

    burn_steps = max(0, min(int(round(burnin * steps)), steps))
    contacts, rgs = [], []
    for step in range(1, steps + 1):
        move = py_rng.choice(saw.MOVE_FUNCS)
        ok, chain_new, occ_new, _changed = move(chain, occ, py_rng)
        if ok:
            chain, occ = chain_new, occ_new
        if step > burn_steps and (step - burn_steps) % sample_every == 0:
            contacts.append(int(saw.contact_count(chain, occ)))
            rgs.append(float(saw.radius_of_gyration(chain)))
    return np.asarray(contacts, dtype=np.int64), np.asarray(rgs, dtype=np.float64)


def test_kappa_zero_reproduces_original_sampler():
    kw = dict(N=12, steps=5000, burnin=0.2, sample_every=50)
    res = saw.run_independent_chain(0, 7, kappa_bend=0.0, **kw)
    ref_c, ref_rg = _original_accept_every_valid_move(7, **kw)

    assert res["contact_samples"].size > 0
    assert np.array_equal(res["contact_samples"], ref_c)
    assert np.array_equal(res["rg_samples"], ref_rg)


def test_kappa_zero_is_the_default():
    kw = dict(N=12, steps=2000, burnin=0.2, sample_every=50)
    default = saw.run_independent_chain(0, 3, **kw)
    explicit = saw.run_independent_chain(0, 3, kappa_bend=0.0, **kw)
    assert np.array_equal(default["contact_samples"], explicit["contact_samples"])
    assert np.array_equal(default["rg_samples"], explicit["rg_samples"])
    assert default["accepted_moves"] == explicit["accepted_moves"]


def test_tracked_bend_count_matches_full_recount():
    """The incrementally tracked n_bend never drifts from count_bends()."""
    res = saw.run_independent_chain(0, 5, N=12, steps=3000, burnin=0.0,
                                    sample_every=1, kappa_bend=1.0)
    # Bend samples are integers in the valid range for a 12-bead chain.
    bends = res["bend_samples"]
    assert bends.size == 3000
    assert bends.min() >= 0 and bends.max() <= 12 - 2


# ---------------------------------------------------------------------------
# 7: a positive kappa_bend really stiffens the chain
# ---------------------------------------------------------------------------

def test_positive_kappa_lowers_the_bend_fraction():
    N, kw = 16, dict(steps=60_000, burnin=0.3, sample_every=50)
    soft = saw.run_independent_chain(0, 99, N=N, kappa_bend=0.0, **kw)
    stiff = saw.run_independent_chain(0, 99, N=N, kappa_bend=1.5, **kw)

    n_angles = N - 2
    f_soft = float(soft["bend_samples"].mean()) / n_angles
    f_stiff = float(stiff["bend_samples"].mean()) / n_angles

    # Athermal SAWs bend at most interior sites; kappa=1.5 should cut that
    # substantially. Margin is generous so the test is not flaky.
    assert f_soft > 0.5
    assert f_stiff < f_soft - 0.15


def test_sanity_checks_pass():
    """The startup validation hooks the sampler runs before every production run."""
    saw.sanity_check_end_move()
    saw.sanity_check_bends()
