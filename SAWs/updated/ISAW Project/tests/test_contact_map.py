"""Contact-coordinate and contact-map validation tests (Phases 1-2)."""
import numpy as np
import pytest

import isaw_contact_observables as ico
from conftest import HAIRPIN6, STRAIGHT10, random_saw


# --- coordinate validation (Phase 1) --------------------------------------

def test_fractional_coordinates_rejected():
    with pytest.raises(ico.ContactMapError):
        ico.normalize_lattice_coordinates([(0.5, 0, 0), (1, 0, 0)])


def test_nan_and_inf_rejected():
    for bad in (np.nan, np.inf, -np.inf):
        with pytest.raises(ico.ContactMapError):
            ico.normalize_lattice_coordinates([(0, 0, 0), (bad, 0, 0)])


def test_duplicate_sites_rejected():
    with pytest.raises(ico.ContactMapError):
        ico.normalize_lattice_coordinates([(0, 0, 0), (1, 0, 0), (1, 0, 0)])


def test_wrong_shape_rejected():
    with pytest.raises(ico.ContactMapError):
        ico.normalize_lattice_coordinates([[0, 0], [1, 0]])
    with pytest.raises(ico.ContactMapError):
        ico.normalize_lattice_coordinates(np.zeros((0, 3)))


def test_broken_backbone_rejected():
    with pytest.raises(ico.ContactMapError):
        ico.normalize_lattice_coordinates([(0, 0, 0), (5, 0, 0)])


def test_valid_int_array_and_tuple_list_accepted():
    arr = ico.normalize_lattice_coordinates(HAIRPIN6)
    assert arr.dtype.kind == "i"
    arr2 = ico.normalize_lattice_coordinates(np.asarray(HAIRPIN6, dtype=np.int32))
    assert np.array_equal(arr, arr2)


def test_light_mode_allows_non_saw_geometry():
    # Two coincident points: rejected with SAW, allowed in light geometry mode.
    pts = [(0, 0, 0), (0, 0, 0)]
    with pytest.raises(ico.ContactMapError):
        ico.normalize_lattice_coordinates(pts)
    out = ico.normalize_lattice_coordinates(
        pts, require_self_avoiding=False, require_backbone_bonds=False)
    assert out.shape == (2, 3)


# --- contact-map tests (Phase 2) ------------------------------------------

def test_straight_chain_zero_contacts():
    cp, seps = ico.build_contact_map(STRAIGHT10)
    assert cp.shape == (0, 2) and seps.shape == (0,)


def test_handbuilt_known_contacts():
    cp, seps = ico.build_contact_map(HAIRPIN6)
    assert [tuple(p) for p in cp] == [(0, 5), (1, 4)]
    assert sorted(seps.tolist()) == [3, 5]


def test_hash_matches_bruteforce():
    for seed in range(30):
        chain = random_saw(18, seed)
        if chain is None:
            continue
        a, _ = ico.build_contact_map(chain)
        b, _ = ico.build_contact_map_bruteforce(chain)
        assert np.array_equal(a, b), seed


def test_incomplete_map_fails_strict():
    cp, seps = ico.build_contact_map(HAIRPIN6)
    with pytest.raises(ico.ContactMapError):
        ico.validate_contact_map(HAIRPIN6, cp[:1])


def test_extra_contacts_fail():
    with pytest.raises(ico.ContactMapError):
        ico.validate_contact_map(HAIRPIN6, np.array([[0, 5], [1, 4], [0, 3]]))


def test_incorrect_separations_fail():
    cp, _ = ico.build_contact_map(HAIRPIN6)
    with pytest.raises(ico.ContactMapError):
        ico.validate_contact_map(HAIRPIN6, cp, separations=np.array([9, 9]))


def test_out_of_bounds_indices_fail():
    with pytest.raises(ico.ContactMapError):
        ico.validate_contact_map(HAIRPIN6, np.array([[0, 99]]))


def test_duplicate_pairs_fail():
    with pytest.raises(ico.ContactMapError):
        ico.validate_contact_map(HAIRPIN6, np.array([[0, 5], [0, 5]]))


def test_mr_sums_to_m_and_odd():
    for seed in range(25):
        chain = random_saw(24, seed)
        if chain is None:
            continue
        cp, seps = ico.build_contact_map(chain)
        m_r = ico.contact_separation_counts(cp, len(chain))
        assert int(m_r.sum()) == cp.shape[0]
        assert np.all(m_r[0::2] == 0)             # even entries zero
        if seps.size:
            assert np.all((seps % 2) == 1)        # all odd


def test_contact_separation_counts_rejects_bad_n():
    cp, _ = ico.build_contact_map(HAIRPIN6)
    with pytest.raises(ico.ContactMapError):
        ico.contact_separation_counts(cp, 0)


def test_validate_full_map_ok():
    cp, seps = ico.build_contact_map(HAIRPIN6)
    info = ico.validate_contact_map(HAIRPIN6, cp, seps, expected_contact_count=2)
    assert info["ok"] and info["contact_count"] == 2
