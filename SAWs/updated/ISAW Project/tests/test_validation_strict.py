"""P3/P4/P13: strict pair/separation/m_r validation and overflow guards."""
import numpy as np
import pytest

import isaw_contact_observables as ico
from conftest import HAIRPIN6


# --- contact-pair shape (P3) ----------------------------------------------

@pytest.mark.parametrize("bad", [
    np.array([0, 5, 1, 4]),            # (4,)
    np.array([[0, 5, 1, 4]]),          # (1, 4)
    np.zeros((2, 2, 1)),               # 3-D
    np.zeros((0, 4)),                  # empty but wrong trailing dim
])
def test_wrong_pair_shape_rejected(bad):
    with pytest.raises(ico.ContactMapError):
        ico.validate_contact_map(HAIRPIN6, bad)


def test_object_pair_array_rejected():
    with pytest.raises(ico.ContactMapError):
        ico.validate_contact_map(HAIRPIN6, np.array([[0, 5], [1]], dtype=object))


def test_fractional_pair_index_rejected():
    with pytest.raises(ico.ContactMapError):
        ico.validate_contact_map(HAIRPIN6, np.array([[0.5, 5.0], [1.0, 4.0]]))


def test_empty_pairs_ok():
    info = ico.validate_contact_map([(0, 0, 0), (1, 0, 0), (2, 0, 0)],
                                    np.empty((0, 2)))
    assert info["contact_count"] == 0


# --- separation validation (P3) -------------------------------------------

def test_fractional_separation_rejected():
    cp, _ = ico.build_contact_map(HAIRPIN6)
    with pytest.raises(ico.ContactMapError):
        ico.validate_contact_map(HAIRPIN6, cp, separations=np.array([5.9, 3.0]))


def test_nan_separation_rejected():
    cp, _ = ico.build_contact_map(HAIRPIN6)
    with pytest.raises(ico.ContactMapError):
        ico.validate_contact_map(HAIRPIN6, cp, separations=np.array([np.nan, 3.0]))


def test_wrong_separation_shape_rejected():
    cp, _ = ico.build_contact_map(HAIRPIN6)
    with pytest.raises(ico.ContactMapError):
        ico.validate_contact_map(HAIRPIN6, cp, separations=np.array([[5], [3]]))


# --- m_r validation (P4) --------------------------------------------------

def test_valid_m_r_accepted():
    m_r = np.array([0, 0, 0, 1, 0, 1], dtype=np.int64)
    out = ico.validate_m_r(m_r, n_beads=6)
    assert out.dtype.kind == "i" and out.tolist() == [0, 0, 0, 1, 0, 1]


def test_zero_contact_m_r_accepted():
    assert ico.validate_m_r(np.zeros(10, dtype=np.int64)).sum() == 0


@pytest.mark.parametrize("bad,kw", [
    (np.array([-1, 0, 0, 1, 0, 1]), {}),         # negative
    (np.array([0, 0, 0, 1.5, 0, 1]), {}),         # fractional
    (np.array([np.nan, 0, 0, 1, 0, 1]), {}),      # NaN
    (np.array([0, 0, 2, 0, 0, 0]), {}),           # even-r nonzero
    (np.array([0, 0, 0, 1]), {"n_beads": 6}),     # wrong length
    (np.array([[0, 0, 0, 1, 0, 1]]), {}),         # 2-D
])
def test_invalid_m_r_rejected(bad, kw):
    with pytest.raises(ico.ContactMapError):
        ico.validate_m_r(bad, **kw)


def test_even_r_allowed_when_parity_disabled():
    # r=4 is even (parity-forbidden) but >= MIN_CONTOUR_SEPARATION; allowed only
    # when cubic parity is not required.
    m_r = np.array([0, 0, 0, 0, 2, 0, 0, 0])
    with pytest.raises(ico.ContactMapError):
        ico.validate_m_r(m_r)                      # parity enforced by default
    out = ico.validate_m_r(m_r, require_cubic_parity=False)
    assert out[4] == 2


# --- unsigned/overflow coordinate guard (P13) -----------------------------

def test_uint64_overflow_rejected():
    bad = np.array([[np.uint64(2 ** 63 + 1), 0, 0], [1, 0, 0]], dtype=np.uint64)
    with pytest.raises(ico.ContactMapError):
        ico.normalize_lattice_coordinates(bad)


def test_large_but_valid_uint_accepted():
    arr = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.uint32)
    out = ico.normalize_lattice_coordinates(arr)
    assert out.dtype == np.int64
