"""Phase 4/14: resume-stage fingerprint proves a feature file belongs to the
exact requested calibration stage.
"""
import json
import os
import tempfile

import numpy as np
import pytest

import isaw_contact_observables as ico
import isaw_config_io as cio
import extract_contact_motif_features as ext
import run_structural_regime_pilot as cal

pytestmark = pytest.mark.skipif(not cio.h5py_available(), reason="h5py missing")

HAIRPIN6 = [(0, 0, 0), (1, 0, 0), (2, 0, 0), (2, 1, 0), (1, 1, 0), (0, 1, 0)]
INFO = {"model_name": "hs", "param_names": ["h", "s"], "params": [330.5, 1.28],
        "Tref": 320.0, "Tscale": 80.0}


def _fields(tmp, **over):
    fit = os.path.join(tmp, "fit.json")
    with open(fit, "w") as fh:
        json.dump({"model": "hs", "params": {"h": 330.5, "s": 1.28},
                   "Tref": 320.0, "Tscale": 80.0}, fh)
    base = dict(N=30, seed=1, ladder=[300.0, 320.0, 340.0],
                K_ladder=[-0.1, 0.0, 0.1], info=INFO, fit_summary_path=fit,
                n_cycles=400, steps_per_swap=60, n_workers=1,
                structural_stride=5, snapshot_stride=5, burnin_frac=0.5)
    base.update(over)
    return cal.build_stage_fingerprint_fields(**base)


def test_fingerprint_is_deterministic():
    with tempfile.TemporaryDirectory() as tmp:
        f1 = _fields(tmp)
        f2 = _fields(tmp)
        assert cal._stage_fingerprint(f1) == cal._stage_fingerprint(f2)


@pytest.mark.parametrize("over", [
    {"n_cycles": 401},
    {"ladder": [300.0, 320.0, 341.0]},
    {"K_ladder": [-0.1, 0.0, 0.2]},
    {"steps_per_swap": 61},
    {"structural_stride": 6},
    {"snapshot_stride": 6},
    {"burnin_frac": 0.4},
    {"seed": 2},
    {"n_workers": 2},
])
def test_changed_field_changes_fingerprint(over):
    with tempfile.TemporaryDirectory() as tmp:
        base = cal._stage_fingerprint(_fields(tmp))
        changed = cal._stage_fingerprint(_fields(tmp, **over))
        assert base != changed


def test_changed_fit_summary_changes_fingerprint():
    with tempfile.TemporaryDirectory() as tmp:
        f = _fields(tmp)
        base = cal._stage_fingerprint(f)
        # rewrite the fit-summary file the fingerprint hashed -> hash changes
        with open(f["fit_summary_path"], "w") as fh:
            json.dump({"model": "hs", "params": {"h": 331.0, "s": 1.28},
                       "Tref": 320.0, "Tscale": 80.0}, fh)
        changed = cal._stage_fingerprint(_fields_same_path(f))
        assert base != changed


def _fields_same_path(f):
    # recompute using the same fit path so only its on-disk hash differs
    return cal.build_stage_fingerprint_fields(
        N=f["N"], seed=f["seed"], ladder=f["temperature_ladder"],
        K_ladder=f["K_ladder"], info=INFO, fit_summary_path=f["fit_summary_path"],
        n_cycles=f["n_cycles"], steps_per_swap=f["steps_per_swap"],
        n_workers=f["n_workers"], structural_stride=f["structural_stride"],
        snapshot_stride=f["snapshot_stride"], burnin_frac=f["burnin_frac"])


# --- stage_reusable end-to-end with a real feature file + certificate --------

def _make_stage(tmp):
    nb, nT = len(HAIRPIN6), 2
    cp, _ = ico.build_contact_map(HAIRPIN6)
    m = cp.shape[0]
    rg2 = ico.radius_of_gyration_squared(HAIRPIN6)
    ree2 = ico.end_to_end_distance_squared(HAIRPIN6)
    meta = {"run_id": "cf", "seed": 1, "temperatures": [300.0, 350.0],
            "model_name": "hs", "param_names": ["h", "s"],
            "model_params": [647.7, 1.874], "Tref": 320.0, "Tscale": 80.0}
    cfg = os.path.join(tmp, "cfg.h5")
    w = cio.SnapshotWriter(cfg, n_beads=nb, n_temperatures=nT, metadata=meta)
    for c in range(3):
        w.append(cycle=c,
                 coordinates=np.stack([np.asarray(HAIRPIN6, dtype=np.int64)] * nT),
                 walker_id=np.array([0, 1] if c % 2 == 0 else [1, 0]),
                 contacts=np.full(nT, m), rg2_lattice=np.full(nT, rg2),
                 ree2_lattice=np.full(nT, ree2))
    w.mark_complete()
    w.close()
    feat = os.path.join(tmp, "feat.h5")
    ext.extract(cfg, feat, validate=True, output_format="hdf5", overwrite=True)
    return cfg, feat


def _companions(tmp, cfg, feat):
    # create empty placeholder companion files so presence checks pass
    prefix = os.path.join(tmp, "calib")
    comp = cal._companion_paths(prefix, cfg, feat)
    for name, path in comp.items():
        if not os.path.exists(path):
            open(path, "w").close()
    return comp


def test_stage_reusable_accepts_matching_and_rejects_mismatch():
    with tempfile.TemporaryDirectory() as tmp:
        cfg, feat = _make_stage(tmp)
        comp = _companions(tmp, cfg, feat)
        fp = "fp-abc"
        cal._deep_validate_and_certify(feat, stage_fingerprint=fp,
                                       source_config=cfg)
        ok, reason = cal.stage_reusable(feat, cfg, fp, comp)
        assert ok is True, reason
        # wrong fingerprint -> not reusable
        ok2, reason2 = cal.stage_reusable(feat, cfg, "fp-different", comp)
        assert ok2 is False and "fingerprint" in reason2


def test_stage_reusable_rejects_missing_companion():
    with tempfile.TemporaryDirectory() as tmp:
        cfg, feat = _make_stage(tmp)
        comp = _companions(tmp, cfg, feat)
        fp = "fp-abc"
        cal._deep_validate_and_certify(feat, stage_fingerprint=fp,
                                       source_config=cfg)
        os.remove(comp["diagnostic_trajectories_npz"])
        ok, reason = cal.stage_reusable(feat, cfg, fp, comp)
        assert ok is False and "companion" in reason


def test_stage_reusable_rejects_changed_source_hash():
    with tempfile.TemporaryDirectory() as tmp:
        cfg, feat = _make_stage(tmp)
        comp = _companions(tmp, cfg, feat)
        fp = "fp-abc"
        cal._deep_validate_and_certify(feat, stage_fingerprint=fp,
                                       source_config=cfg)
        # mutate the source configuration so its hash no longer matches the cert
        with open(cfg, "ab") as fh:
            fh.write(b"\x00")
        ok, reason = cal.stage_reusable(feat, cfg, fp, comp)
        assert ok is False and "source configuration hash" in reason
