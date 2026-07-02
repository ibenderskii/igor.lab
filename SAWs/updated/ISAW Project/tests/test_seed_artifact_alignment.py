"""Phase 5/6/14: seed-artifact audit + cross-seed alignment enforcement."""
import copy
import os
import tempfile

import pytest

import run_structural_regime_pilot as cal


# --- seed-artifact audit ----------------------------------------------------

def _companions(tmp, seed, *, make=True, drop=()):
    prefix = os.path.join(tmp, f"s{seed}")
    cfg = f"{prefix}_configurations.h5"
    feat = f"{prefix}_features.h5"
    comp = cal._companion_paths(prefix, cfg, feat)
    if make:
        for name, path in comp.items():
            if name in drop:
                continue
            open(path, "w").close()
    return comp


def test_audit_all_complete_is_assessable():
    with tempfile.TemporaryDirectory() as tmp:
        sc = {1: _companions(tmp, 1), 2: _companions(tmp, 2)}
        audit = cal.audit_seed_artifacts(sc, feature_validator=lambda p: True)
        assert audit["assessable"] is True
        assert audit["complete_seeds"] == [1, 2]
        assert not audit["missing_seeds"] and not audit["invalid_seeds"]


def test_audit_one_seed_missing_trajectory_not_assessable():
    with tempfile.TemporaryDirectory() as tmp:
        sc = {1: _companions(tmp, 1),
              2: _companions(tmp, 2, drop=("diagnostic_trajectories_npz",))}
        audit = cal.audit_seed_artifacts(sc, feature_validator=lambda p: True)
        assert audit["assessable"] is False
        assert 2 in audit["missing_seeds"]
        assert "diagnostic_trajectories_npz" in \
            audit["artifact_failures_by_seed"][2]["missing_artifacts"]


def test_audit_invalid_feature_not_assessable():
    with tempfile.TemporaryDirectory() as tmp:
        sc = {1: _companions(tmp, 1), 2: _companions(tmp, 2)}
        audit = cal.audit_seed_artifacts(
            sc, feature_validator=lambda p: "s2" not in str(p))
        assert audit["assessable"] is False
        assert 2 in audit["invalid_seeds"]


def test_audit_one_seed_is_not_assessable():
    with tempfile.TemporaryDirectory() as tmp:
        sc = {1: _companions(tmp, 1)}
        audit = cal.audit_seed_artifacts(sc, feature_validator=lambda p: True)
        assert audit["all_requested_complete"] is True
        assert audit["assessable"] is False       # < 2 seeds


# --- cross-seed alignment ---------------------------------------------------

def _record(**over):
    base = dict(N=30, temperatures=[300.0, 320.0, 340.0],
                K=[-0.1, 0.0, 0.1], model_name="hs",
                model_params=[330.5, 1.28], Tref=320.0, Tscale=80.0,
                definitions_version="1.1.0",
                trajectory_obs=["contacts_post", "rg2_post"],
                trajectory_shapes={"contacts_post": (3, 100),
                                   "rg2_post": (3, 100)})
    base.update(over)
    return base


def test_aligned_seeds_pass():
    out = cal.require_seed_alignment([_record(), _record()])
    assert out["aligned"] is True and out["n_temperatures"] == 3


@pytest.mark.parametrize("bad", [
    _record(temperatures=[300.0, 320.0]),                       # missing a lane
    _record(temperatures=[301.0, 320.0, 340.0]),                # shifted temp
    _record(temperatures=[340.0, 320.0, 300.0]),                # reversed lanes
    _record(K=[-0.1, 0.0, 0.2]),                                # different K
    _record(model_params=[331.0, 1.28]),                        # different params
    _record(trajectory_shapes={"contacts_post": (3, 50),        # shorter arrays
                               "rg2_post": (3, 50)}),
    _record(definitions_version="9.9.9"),                       # different defs
])
def test_misaligned_seed_rejected(bad):
    with pytest.raises(cal.SeedAlignmentError):
        cal.require_seed_alignment([_record(), bad])


def test_alignment_is_seed_order_invariant():
    recs = [_record(), _record(K=[-0.1, 0.0, 0.2])]
    with pytest.raises(cal.SeedAlignmentError):
        cal.require_seed_alignment(recs)
    with pytest.raises(cal.SeedAlignmentError):
        cal.require_seed_alignment(list(reversed(recs)))
    good = [_record(), _record()]
    assert cal.require_seed_alignment(good)["aligned"] is True
    assert cal.require_seed_alignment(list(reversed(good)))["aligned"] is True
