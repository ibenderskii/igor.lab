"""Part 3: every artifact must EQUAL the expected fitted-model stage record.

Each test faithfully updates the artifact-manifest hash after corrupting a file,
so a passing hash check can never mask a semantic (model/parameter/ladder/seed/N)
mismatch.
"""
import csv
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

import isaw_contact_observables as ico
import isaw_config_io as cio
import extract_contact_motif_features as ext
import remd_uniform_chain_2_new as remd
import run_structural_regime_pilot as cal

pytestmark = pytest.mark.skipif(not cio.h5py_available(), reason="h5py missing")

HP = [(0, 0, 0), (1, 0, 0), (2, 0, 0), (2, 1, 0), (1, 1, 0), (0, 1, 0)]
N, SEED = 6, 5
LADDER = [300.0, 340.0]
FP = "modelrec" * 8
MODEL = {"model_name": "hs", "param_names": ["h", "s"], "params": [330.5, 1.28],
         "Tref": 320.0, "Tscale": 80.0}


def _csvw(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)


def _write_distributions(prefix, *, model_name=None, model_params=None,
                         param_names=None, Tref=None, Tscale=None, Ts=None):
    nT = len(LADDER)
    np.savez(prefix + "_distributions.npz",
             Ts=np.asarray(LADDER if Ts is None else Ts),
             temps=np.asarray(LADDER if Ts is None else Ts),
             Pc=np.zeros((nT, 5)), Prg=np.zeros((nT, 7)),
             model_name=np.array(model_name or MODEL["model_name"]),
             param_names=np.array(param_names or MODEL["param_names"]),
             model_params=np.array(model_params or MODEL["params"], float),
             Tref=np.array(MODEL["Tref"] if Tref is None else Tref),
             Tscale=np.array(MODEL["Tscale"] if Tscale is None else Tscale),
             n_beads=np.array(N))


def _write_run_summary(prefix, *, seed=SEED, n=N):
    Path(prefix + "_run_summary.json").write_text(json.dumps({
        "N": n, "n_beads": n, "seed": seed, "temperatures": LADDER,
        "model": MODEL["model_name"], "param_names": MODEL["param_names"],
        "params": MODEL["params"], "Tref": MODEL["Tref"],
        "Tscale": MODEL["Tscale"], "burnin_frac": 0.5, "n_cycles": 400,
        "steps_per_swap": 60, "snapshot_stride": 5}), encoding="utf-8")


def _make_run(tmp):
    prefix = os.path.join(tmp, "calib_N6_s5")
    cfg = prefix + "_configurations.h5"
    feat = prefix + "_features.h5"
    nT = len(LADDER)
    cp, _ = ico.build_contact_map(HP)
    m = cp.shape[0]
    rg2 = ico.radius_of_gyration_squared(HP)
    ree2 = ico.end_to_end_distance_squared(HP)
    w = cio.SnapshotWriter(cfg, n_beads=N, n_temperatures=nT,
                           metadata={"run_id": "r6", "seed": SEED,
                                     "temperatures": LADDER,
                                     "model_name": MODEL["model_name"],
                                     "param_names": MODEL["param_names"],
                                     "model_params": MODEL["params"],
                                     "Tref": MODEL["Tref"],
                                     "Tscale": MODEL["Tscale"]})
    for c in range(2):
        w.append(cycle=c,
                 coordinates=np.stack([np.asarray(HP, dtype=np.int64)] * nT),
                 walker_id=np.array([0, 1] if c % 2 == 0 else [1, 0]),
                 contacts=np.full(nT, m), rg2_lattice=np.full(nT, rg2),
                 ree2_lattice=np.full(nT, ree2))
    w.mark_complete()
    w.close()
    cal._embed_fingerprint_hdf5(cfg, FP, {"model_name": MODEL["model_name"]})
    ext.extract(cfg, feat, validate=True, overwrite=True)
    cal._embed_fingerprint_feature(feat, FP, {"model_name": MODEL["model_name"]})
    cal._deep_validate_and_certify(feat, stage_fingerprint=FP,
                                   stage_fingerprint_fields={"m": MODEL["model_name"]},
                                   source_config=cfg)
    _csvw(prefix + "_results.csv", ["T", "C_mean", "Rg2_mean_lattice"],
          [[LADDER[i], 3.0, rg2] for i in range(nT)])
    _csvw(prefix + "_swap_rates.csv",
          ["pair", "T_lo", "T_hi", "proposals", "acceptances", "rate"],
          [[0, LADDER[0], LADDER[1], 10, 5, "0.5"]])
    ma = [[i, LADDER[i], mn, 10, 8, 6] for i in range(nT) for mn in remd.MOVE_NAMES]
    _csvw(prefix + "_move_acceptance.csv",
          ["temperature_index", "temperature", "move_type",
           "proposed", "geometrically_valid", "metropolis_accepted"], ma)
    Path(prefix + "_diagnostics.json").write_text(json.dumps(
        {"temperatures": LADDER, "n_temperatures": nT, "burnin_frac": 0.5}),
        encoding="utf-8")
    arrs = {k: np.zeros((nT, 8), np.float32) for k in
            ("contacts_post", "rg2_post", "m_long_post", "m_global_scaled_post",
             "smax_post", "largest_component_fraction_post")}
    np.savez(prefix + "_diagnostic_trajectories.npz", Ts=np.asarray(LADDER),
             stage_fingerprint=np.array(FP), **arrs)
    _write_distributions(prefix)
    _write_run_summary(prefix)
    return prefix, cfg, feat


def _record():
    info = {"model_name": MODEL["model_name"], "param_names": MODEL["param_names"],
            "params": MODEL["params"], "Tref": MODEL["Tref"],
            "Tscale": MODEL["Tscale"]}
    return cal.build_expected_stage_record(
        N=N, seed=SEED, run_id=None, ladder=LADDER, K_ladder=[0.0, 0.0], info=info,
        fit_summary_sha256=None, burnin_frac=0.5, n_cycles=400, steps_per_swap=60,
        structural_stride=5, snapshot_stride=5, stage_fingerprint=FP)


def _validate(prefix, cfg, feat):
    paths = cal._full_companion_paths(prefix, cfg, feat)
    manp = prefix + "_artifact_manifest.json"
    cal.build_artifact_manifest(paths, manp)
    return cal.validate_artifact_manifest(
        manp, N=N, seed=SEED, ladder=LADDER, source_config=cfg,
        stage_fingerprint=FP, run_id="r6", model_record=_record())


def test_clean_run_with_model_record_validates():
    with tempfile.TemporaryDirectory() as tmp:
        prefix, cfg, feat = _make_run(tmp)
        ok, report = _validate(prefix, cfg, feat)
        assert ok, report


def test_changed_model_name_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        prefix, cfg, feat = _make_run(tmp)
        _write_distributions(prefix, model_name="poly2")   # hash rebuilt in _validate
        ok, report = _validate(prefix, cfg, feat)
        assert not ok and "semantic" in report["distributions_npz"]


def test_changed_parameter_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        prefix, cfg, feat = _make_run(tmp)
        _write_distributions(prefix, model_params=[999.0, 1.28])
        ok, report = _validate(prefix, cfg, feat)
        assert not ok and "semantic" in report["distributions_npz"]


def test_changed_tref_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        prefix, cfg, feat = _make_run(tmp)
        _write_distributions(prefix, Tref=300.0)
        ok, report = _validate(prefix, cfg, feat)
        assert not ok and "semantic" in report["distributions_npz"]


def test_changed_tscale_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        prefix, cfg, feat = _make_run(tmp)
        _write_distributions(prefix, Tscale=70.0)
        ok, report = _validate(prefix, cfg, feat)
        assert not ok and "semantic" in report["distributions_npz"]


def test_changed_temperature_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        prefix, cfg, feat = _make_run(tmp)
        _write_distributions(prefix, Ts=[300.0, 999.0])
        ok, report = _validate(prefix, cfg, feat)
        assert not ok and "semantic" in report["distributions_npz"]


def test_changed_seed_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        prefix, cfg, feat = _make_run(tmp)
        _write_run_summary(prefix, seed=SEED + 1)
        ok, report = _validate(prefix, cfg, feat)
        assert not ok and "semantic" in report["run_summary_json"]


def test_changed_N_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        prefix, cfg, feat = _make_run(tmp)
        _write_run_summary(prefix, n=N + 1)
        ok, report = _validate(prefix, cfg, feat)
        assert not ok and "semantic" in report["run_summary_json"]
