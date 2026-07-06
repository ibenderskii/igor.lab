"""Phase 5: the one shared certificate-trust function.

``validate_feature_certificate`` must reject an obsolete validator revision, a
wrong stage fingerprint, a wrong source/feature/definitions hash, a wrong
definitions version, a certificate for another feature path, and a missing
certificate -- an old certificate is never trusted merely because its feature
hash still matches.
"""
import json
import os
import shutil
import tempfile

import numpy as np
import pytest

import isaw_contact_observables as ico
import isaw_config_io as cio
import extract_contact_motif_features as ext
import run_structural_regime_pilot as cal

pytestmark = pytest.mark.skipif(not cio.h5py_available(), reason="h5py missing")

HP = [(0, 0, 0), (1, 0, 0), (2, 0, 0), (2, 1, 0), (1, 1, 0), (0, 1, 0)]
FP = "deadbeef" * 8


def _build(tmp):
    cp, _ = ico.build_contact_map(HP)
    m = cp.shape[0]
    rg2 = ico.radius_of_gyration_squared(HP)
    ree2 = ico.end_to_end_distance_squared(HP)
    cfg = os.path.join(tmp, "cfg.h5")
    w = cio.SnapshotWriter(cfg, n_beads=6, n_temperatures=2,
                           metadata={"run_id": "c", "seed": 1,
                                     "temperatures": [300.0, 340.0],
                                     "model_name": "hs", "param_names": ["h", "s"],
                                     "model_params": [330.5, 1.28],
                                     "Tref": 320.0, "Tscale": 80.0})
    w.append(cycle=0, coordinates=np.stack([np.asarray(HP, dtype=np.int64)] * 2),
             walker_id=np.array([0, 1]), contacts=np.full(2, m),
             rg2_lattice=np.full(2, rg2), ree2_lattice=np.full(2, ree2))
    w.mark_complete()
    w.close()
    feat = os.path.join(tmp, "feat.h5")
    ext.extract(cfg, feat, validate=True, overwrite=True)
    cal._embed_fingerprint_feature(feat, FP, {"N": 6})
    cal._deep_validate_and_certify(feat, stage_fingerprint=FP,
                                   stage_fingerprint_fields={"N": 6},
                                   source_config=cfg)
    return feat, cfg


def _expect(**over):
    defs_sha, defs_ver, feat_ver = cal._current_certificate_expectations()
    d = dict(expected_stage_fingerprint=FP,
             expected_validator_revision=cal.VALIDATOR_REVISION,
             expected_definitions_sha256=defs_sha,
             expected_definitions_version=defs_ver,
             expected_feature_schema_version=feat_ver)
    d.update(over)
    return d


def test_valid_certificate_trusted():
    with tempfile.TemporaryDirectory() as tmp:
        feat, cfg = _build(tmp)
        ok, reason = cal.validate_feature_certificate(
            feat, source_configuration_path=cfg, **_expect())
        assert ok, reason


def test_obsolete_validator_revision_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        feat, cfg = _build(tmp)
        ok, _ = cal.validate_feature_certificate(
            feat, source_configuration_path=cfg,
            **_expect(expected_validator_revision=cal.VALIDATOR_REVISION + 1))
        assert not ok


def test_wrong_stage_fingerprint_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        feat, cfg = _build(tmp)
        ok, _ = cal.validate_feature_certificate(
            feat, source_configuration_path=cfg,
            **_expect(expected_stage_fingerprint="0" * 64))
        assert not ok


def test_wrong_definitions_hash_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        feat, cfg = _build(tmp)
        ok, _ = cal.validate_feature_certificate(
            feat, source_configuration_path=cfg,
            **_expect(expected_definitions_sha256="beef" * 16))
        assert not ok


def test_wrong_definitions_version_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        feat, cfg = _build(tmp)
        ok, _ = cal.validate_feature_certificate(
            feat, source_configuration_path=cfg,
            **_expect(expected_definitions_version="9.9.9"))
        assert not ok


def test_source_hash_change_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        feat, cfg = _build(tmp)
        import h5py
        with h5py.File(cfg, "r+") as f:
            f.attrs["_tamper"] = 1
        ok, _ = cal.validate_feature_certificate(
            feat, source_configuration_path=cfg, **_expect())
        assert not ok


def test_feature_hash_change_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        feat, cfg = _build(tmp)
        import h5py
        with h5py.File(feat, "r+") as f:
            f.attrs["_tamper"] = 1
        ok, _ = cal.validate_feature_certificate(
            feat, source_configuration_path=cfg, **_expect())
        assert not ok


def test_certificate_for_other_feature_path_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        feat, cfg = _build(tmp)
        feat2 = os.path.join(tmp, "other.h5")
        shutil.copy(feat, feat2)
        shutil.copy(cal._certificate_path(feat), cal._certificate_path(feat2))
        ok, reason = cal.validate_feature_certificate(
            feat2, source_configuration_path=cfg, **_expect())
        assert not ok and "feature path" in reason


def test_missing_certificate_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        feat, cfg = _build(tmp)
        os.remove(cal._certificate_path(feat))
        ok, _ = cal.validate_feature_certificate(
            feat, source_configuration_path=cfg, **_expect())
        assert not ok


def test_missing_source_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        feat, cfg = _build(tmp)
        os.remove(cfg)
        ok, _ = cal.validate_feature_certificate(
            feat, source_configuration_path=cfg, **_expect())
        assert not ok
