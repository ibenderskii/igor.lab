"""Phase 6: artifact-manifest hash + SEMANTIC validation of every companion.

Each artifact is corrupted independently; a hash mismatch OR a semantic
violation must invalidate that artifact and the whole manifest.
"""
import csv
import json
import os
import tempfile

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
FP = "abc123" * 10 + "abcd"   # 64 hex chars
FIELDS = {"N": N, "seed": SEED}


def _write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)


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
                                     "temperatures": LADDER, "model_name": "hs",
                                     "param_names": ["h", "s"],
                                     "model_params": [330.5, 1.28],
                                     "Tref": 320.0, "Tscale": 80.0})
    for c in range(2):
        w.append(cycle=c,
                 coordinates=np.stack([np.asarray(HP, dtype=np.int64)] * nT),
                 walker_id=np.array([0, 1] if c % 2 == 0 else [1, 0]),
                 contacts=np.full(nT, m), rg2_lattice=np.full(nT, rg2),
                 ree2_lattice=np.full(nT, ree2))
    w.mark_complete()
    w.close()
    cal._embed_fingerprint_hdf5(cfg, FP, FIELDS)
    ext.extract(cfg, feat, validate=True, overwrite=True)
    cal._embed_fingerprint_feature(feat, FP, FIELDS)
    cal._deep_validate_and_certify(feat, stage_fingerprint=FP,
                                   stage_fingerprint_fields=FIELDS,
                                   source_config=cfg)
    # results / swap / move-acceptance CSVs
    _write_csv(prefix + "_results.csv",
               ["T", "C_mean", "Rg2_mean_lattice"],
               [[LADDER[i], 3.0, rg2] for i in range(nT)])
    _write_csv(prefix + "_swap_rates.csv",
               ["pair", "T_lo", "T_hi", "proposals", "acceptances", "rate"],
               [[0, LADDER[0], LADDER[1], 10, 5, "0.5000"]])
    ma_rows = []
    for i in range(nT):
        for mname in remd.MOVE_NAMES:
            ma_rows.append([i, LADDER[i], mname, 10, 8, 6])
    _write_csv(prefix + "_move_acceptance.csv",
               ["temperature_index", "temperature", "move_type",
                "proposed", "geometrically_valid", "metropolis_accepted"],
               ma_rows)
    # diagnostics json
    Path = __import__("pathlib").Path
    Path(prefix + "_diagnostics.json").write_text(json.dumps({
        "temperatures": LADDER, "n_temperatures": nT, "burnin_frac": 0.5}),
        encoding="utf-8")
    # diagnostic trajectories npz
    arrs = {k: np.zeros((nT, 8), np.float32) for k in
            ("contacts_post", "rg2_post", "m_long_post", "m_global_scaled_post",
             "smax_post", "largest_component_fraction_post")}
    np.savez(prefix + "_diagnostic_trajectories.npz", Ts=np.asarray(LADDER),
             stage_fingerprint=np.array(FP), **arrs)
    # distributions npz
    np.savez(prefix + "_distributions.npz", Ts=np.asarray(LADDER),
             temps=np.asarray(LADDER), Pc=np.zeros((nT, 5)), Prg=np.zeros((nT, 7)),
             model_name=np.array("hs"), n_beads=np.array(N))
    # run summary json
    Path(prefix + "_run_summary.json").write_text(json.dumps({
        "N": N, "n_beads": N, "seed": SEED, "temperatures": LADDER,
        "model": "hs", "param_names": ["h", "s"], "params": [330.5, 1.28]}),
        encoding="utf-8")
    return prefix, cfg, feat


def _validate(prefix, cfg, feat):
    paths = cal._full_companion_paths(prefix, cfg, feat)
    manp = prefix + "_artifact_manifest.json"
    cal.build_artifact_manifest(paths, manp)
    return cal.validate_artifact_manifest(
        manp, N=N, seed=SEED, ladder=LADDER, source_config=cfg,
        stage_fingerprint=FP, run_id="r6")


def test_clean_manifest_validates():
    with tempfile.TemporaryDirectory() as tmp:
        prefix, cfg, feat = _make_run(tmp)
        ok, report = _validate(prefix, cfg, feat)
        assert ok, report
        assert all(v == "ok" for v in report.values()), report


def test_hash_mismatch_detected():
    with tempfile.TemporaryDirectory() as tmp:
        prefix, cfg, feat = _make_run(tmp)
        # build manifest first, THEN tamper the file so its hash changes
        paths = cal._full_companion_paths(prefix, cfg, feat)
        manp = prefix + "_artifact_manifest.json"
        cal.build_artifact_manifest(paths, manp)
        with open(prefix + "_results.csv", "a", encoding="utf-8") as f:
            f.write("\n# tamper\n")
        ok, report = cal.validate_artifact_manifest(
            manp, N=N, seed=SEED, ladder=LADDER, source_config=cfg,
            stage_fingerprint=FP, run_id="r6")
        assert not ok and report["results_csv"] == "hash mismatch"


def test_results_csv_wrong_temperature_fails():
    with tempfile.TemporaryDirectory() as tmp:
        prefix, cfg, feat = _make_run(tmp)
        _write_csv(prefix + "_results.csv", ["T", "C_mean", "Rg2_mean_lattice"],
                   [[LADDER[0] + 9.0, 3.0, 1.0], [LADDER[1], 3.0, 1.0]])
        ok, report = _validate(prefix, cfg, feat)
        assert not ok and "results_csv" in report and "semantic" in report["results_csv"]


def test_swap_rates_wrong_pair_count_fails():
    with tempfile.TemporaryDirectory() as tmp:
        prefix, cfg, feat = _make_run(tmp)
        _write_csv(prefix + "_swap_rates.csv",
                   ["pair", "T_lo", "T_hi", "proposals", "acceptances", "rate"],
                   [[0, LADDER[0], LADDER[1], 10, 5, "0.5"],
                    [1, LADDER[1], 999.0, 10, 5, "0.5"]])
        ok, report = _validate(prefix, cfg, feat)
        assert not ok and "semantic" in report["swap_rates_csv"]


def test_move_acceptance_missing_move_type_fails():
    with tempfile.TemporaryDirectory() as tmp:
        prefix, cfg, feat = _make_run(tmp)
        rows = []
        for i in range(len(LADDER)):
            for mname in list(remd.MOVE_NAMES)[:-1]:  # drop one move type
                rows.append([i, LADDER[i], mname, 10, 8, 6])
        _write_csv(prefix + "_move_acceptance.csv",
                   ["temperature_index", "temperature", "move_type",
                    "proposed", "geometrically_valid", "metropolis_accepted"], rows)
        ok, report = _validate(prefix, cfg, feat)
        assert not ok and "semantic" in report["move_acceptance_csv"]


def test_distributions_wrong_lane_count_fails():
    with tempfile.TemporaryDirectory() as tmp:
        prefix, cfg, feat = _make_run(tmp)
        np.savez(prefix + "_distributions.npz", Ts=np.asarray(LADDER),
                 temps=np.asarray(LADDER), Pc=np.zeros((1, 5)),
                 Prg=np.zeros((2, 7)), model_name=np.array("hs"),
                 n_beads=np.array(N))
        ok, report = _validate(prefix, cfg, feat)
        assert not ok and "semantic" in report["distributions_npz"]


def test_diag_trajectories_wrong_lane_count_fails():
    with tempfile.TemporaryDirectory() as tmp:
        prefix, cfg, feat = _make_run(tmp)
        arrs = {k: np.zeros((1, 8), np.float32) for k in
                ("contacts_post", "rg2_post", "m_long_post",
                 "m_global_scaled_post", "smax_post",
                 "largest_component_fraction_post")}
        np.savez(prefix + "_diagnostic_trajectories.npz",
                 Ts=np.asarray(LADDER), **arrs)
        ok, report = _validate(prefix, cfg, feat)
        assert not ok and "semantic" in report["diagnostic_trajectories_npz"]


def test_run_summary_seed_mismatch_fails():
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmp:
        prefix, cfg, feat = _make_run(tmp)
        Path(prefix + "_run_summary.json").write_text(json.dumps({
            "N": N, "seed": SEED + 1, "temperatures": LADDER, "model": "hs"}),
            encoding="utf-8")
        ok, report = _validate(prefix, cfg, feat)
        assert not ok and "semantic" in report["run_summary_json"]


def test_diagnostics_missing_burnin_fails():
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmp:
        prefix, cfg, feat = _make_run(tmp)
        Path(prefix + "_diagnostics.json").write_text(json.dumps({
            "temperatures": LADDER, "n_temperatures": len(LADDER)}),
            encoding="utf-8")
        ok, report = _validate(prefix, cfg, feat)
        assert not ok and "semantic" in report["diagnostics_json"]


def test_config_fingerprint_mismatch_fails():
    with tempfile.TemporaryDirectory() as tmp:
        prefix, cfg, feat = _make_run(tmp)
        cal._embed_fingerprint_hdf5(cfg, "0" * 64, FIELDS)
        ok, report = _validate(prefix, cfg, feat)
        assert not ok and "semantic" in report["configuration_h5"]
