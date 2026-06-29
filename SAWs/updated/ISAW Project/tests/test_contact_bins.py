"""Independent fixed/scaled contour-bin tests (Phase 3)."""
import numpy as np
import pytest

import isaw_contact_observables as ico
from conftest import random_saw


@pytest.mark.parametrize("n", [30, 44])
def test_fixed_scheme_exhaustive(n):
    chk = ico.validate_bin_definitions(n)
    valid = chk["valid_separations"]
    fm = chk["fixed_membership"]
    covered = sorted(fm["short_fixed"] + fm["medium_fixed"] + fm["long_fixed"])
    assert covered == valid
    # non-overlapping
    assert (len(fm["short_fixed"]) + len(fm["medium_fixed"])
            + len(fm["long_fixed"])) == len(valid)


@pytest.mark.parametrize("n", [30, 44])
def test_scaled_scheme_exhaustive(n):
    chk = ico.validate_bin_definitions(n)
    valid = chk["valid_separations"]
    sm = chk["scaled_membership"]
    covered = sorted(sm["local_scaled"] + sm["mesoscopic_scaled"]
                     + sm["global_scaled"])
    assert covered == valid


@pytest.mark.parametrize("n", [30, 44])
def test_no_unintended_empty_fixed(n):
    chk = ico.validate_bin_definitions(n)
    # The fixed scheme is designed to be non-empty for both pilot lengths.
    for name in ("short_fixed", "medium_fixed", "long_fixed"):
        assert chk["fixed_membership"][name], (n, name)


def test_scaled_local_empty_warns_at_n30():
    chk = ico.validate_bin_definitions(30)
    assert any("local_scaled" in w for w in chk["warnings"])


@pytest.mark.parametrize("n", [30, 44])
def test_fixed_and_scaled_totals_equal_m(n):
    for seed in range(8):
        chain = random_saw(n, 100 + seed)
        if chain is None:
            continue
        cp, _ = ico.build_contact_map(chain)
        m_r = ico.contact_separation_counts(cp, n)
        fixed = ico.bin_contact_separations_fixed(m_r, n)
        scaled = ico.bin_contact_separations_scaled(m_r, n)
        assert sum(fixed.values()) == cp.shape[0]
        assert sum(scaled.values()) == cp.shape[0]


def test_boundary_separations_classified_as_documented():
    # Fixed: r=9 short, r=11 medium, r=13 medium, r=15 long.
    assert ico.assign_fixed_bin(9) == "short_fixed"
    assert ico.assign_fixed_bin(11) == "medium_fixed"
    assert ico.assign_fixed_bin(13) == "medium_fixed"
    assert ico.assign_fixed_bin(15) == "long_fixed"
    # Scaled at N=30: r=3 -> r/N=0.10 -> mesoscopic (not local); r=11 -> global.
    assert ico.assign_scaled_bin(3, 30) == "mesoscopic_scaled"
    assert ico.assign_scaled_bin(9, 30) == "mesoscopic_scaled"
    assert ico.assign_scaled_bin(11, 30) == "global_scaled"
    # Scaled at N=44: r=3 -> local.
    assert ico.assign_scaled_bin(3, 44) == "local_scaled"
