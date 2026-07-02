"""Phase 2: fixed-bin gap handling and scaled boundary-mode validation."""
import pytest

import isaw_contact_observables as ico


def test_gap_fixed_definition_rejected():
    # short 3-5, medium_min 11 leaves valid odd separations 7,9 unassigned.
    bad = {"scheme": "fixed", "short_fixed": {"r_min": 3, "r_max": 5},
           "medium_fixed": {"r_min": 11}, "long_threshold_fixed": 15}
    with pytest.raises(ico.ContactMapError) as e:
        ico.validate_fixed_bin_semantics(bad, 30)
    assert "gap" in str(e.value)


def test_assign_fixed_raises_on_gap_separation():
    bad = {"scheme": "fixed", "short_fixed": {"r_min": 3, "r_max": 5},
           "medium_fixed": {"r_min": 11}, "long_threshold_fixed": 15}
    # r=7 is in the undeclared gap.
    with pytest.raises(ico.ContactMapError):
        ico.assign_fixed_bin(7, bad)
    # boundary values still classify without error
    assert ico.assign_fixed_bin(3, bad) == "short_fixed"
    assert ico.assign_fixed_bin(11, bad) == "medium_fixed"
    assert ico.assign_fixed_bin(15, bad) == "long_fixed"


def test_contiguous_fixed_definition_accepted():
    ok = {"scheme": "fixed", "short_fixed": {"r_min": 3, "r_max": 9},
          "medium_fixed": {"r_min": 11}, "long_threshold_fixed": 13}
    ico.validate_fixed_bin_semantics(ok, 30)   # no raise


@pytest.mark.parametrize("mode", ["open", "half_open", "OPEN", "left"])
def test_unsupported_scaled_boundary_rejected(mode):
    bad = {"local_max_ratio": 0.10, "meso_max_ratio": 0.33, "local_boundary": mode}
    with pytest.raises(ico.ContactMapError) as e:
        ico.validate_scaled_bin_semantics(bad)
    assert "local_boundary" in str(e.value)


def test_closed_scaled_boundary_accepted():
    ok = {"local_max_ratio": 0.10, "meso_max_ratio": 0.33, "local_boundary": "closed"}
    ico.validate_scaled_bin_semantics(ok)   # no raise


def test_missing_local_boundary_defaults_closed():
    ok = {"local_max_ratio": 0.10, "meso_max_ratio": 0.33}
    ico.validate_scaled_bin_semantics(ok)   # no raise (defaults to closed)
