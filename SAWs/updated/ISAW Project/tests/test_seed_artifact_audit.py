"""Phase 7: seed-artifact audit over the FULL artifact set with expectations.

A seed is assessable only when every required artifact exists AND its artifact
manifest + certificate validate for exactly the expected stage.  Missing or
corrupted artifacts mark a seed invalid and the analysis not assessable.
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
N = 6
LADDER = [300.0, 340.0]


def _csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)


def _build_seed(tmp, seed):
    prefix = os.path.join(tmp, f"calib_N6_s{seed}")
    cfg = prefix + "_configurations.h5"
    feat = prefix + "_features.h5"
    fp = f"seed{seed}".ljust(64, "0")
    fields = {"N": N, "seed": seed}
    nT = len(LADDER)
    cp, _ = ico.build_contact_map(HP)
    m = cp.shape[0]
    rg2 = ico.radius_of_gyration_squared(HP)
    ree2 = ico.end_to_end_distance_squared(HP)
    w = cio.SnapshotWriter(cfg, n_beads=N, n_temperatures=nT,
                           metadata={"run_id": f"r{seed}", "seed": seed,
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
    cal._embed_fingerprint_hdf5(cfg, fp, fields)
    ext.extract(cfg, feat, validate=True, overwrite=True)
    cal._embed_fingerprint_feature(feat, fp, fields)
    cal._deep_validate_and_certify(feat, stage_fingerprint=fp,
                                   stage_fingerprint_fields=fields,
                                   source_config=cfg)
    _csv(prefix + "_results.csv", ["T", "C_mean", "Rg2_mean_lattice"],
         [[LADDER[i], 3.0, rg2] for i in range(nT)])
    _csv(prefix + "_swap_rates.csv",
         ["pair", "T_lo", "T_hi", "proposals", "acceptances", "rate"],
         [[0, LADDER[0], LADDER[1], 10, 5, "0.5"]])
    ma = [[i, LADDER[i], mn, 10, 8, 6] for i in range(nT) for mn in remd.MOVE_NAMES]
    _csv(prefix + "_move_acceptance.csv",
         ["temperature_index", "temperature", "move_type",
          "proposed", "geometrically_valid", "metropolis_accepted"], ma)
    Path(prefix + "_diagnostics.json").write_text(json.dumps(
        {"temperatures": LADDER, "n_temperatures": nT, "burnin_frac": 0.5}),
        encoding="utf-8")
    arrs = {k: np.zeros((nT, 8), np.float32) for k in
            ("contacts_post", "rg2_post", "m_long_post", "m_global_scaled_post",
             "smax_post", "largest_component_fraction_post")}
    np.savez(prefix + "_diagnostic_trajectories.npz", Ts=np.asarray(LADDER),
             stage_fingerprint=np.array(fp), **arrs)
    np.savez(prefix + "_distributions.npz", Ts=np.asarray(LADDER),
             temps=np.asarray(LADDER), Pc=np.zeros((nT, 5)), Prg=np.zeros((nT, 7)),
             model_name=np.array("hs"), n_beads=np.array(N))
    Path(prefix + "_run_summary.json").write_text(json.dumps(
        {"N": N, "seed": seed, "temperatures": LADDER, "model": "hs"}),
        encoding="utf-8")
    comp = cal._companion_paths(prefix, cfg, feat)
    cal.build_artifact_manifest(cal._full_companion_paths(prefix, cfg, feat),
                                comp["artifact_manifest_json"])
    exp = {"N": N, "seed": seed, "ladder": LADDER, "source_config": cfg,
           "stage_fingerprint": fp, "run_id": None}
    return comp, exp


def test_all_seeds_complete_assessable():
    with tempfile.TemporaryDirectory() as tmp:
        comps, exps = {}, {}
        for s in (1, 2):
            comps[s], exps[s] = _build_seed(tmp, s)
        audit = cal.audit_seed_artifacts(comps, expectations=exps)
        assert audit["assessable"] is True
        assert audit["complete_seeds"] == [1, 2]


def test_missing_artifact_not_assessable():
    with tempfile.TemporaryDirectory() as tmp:
        comps, exps = {}, {}
        for s in (1, 2):
            comps[s], exps[s] = _build_seed(tmp, s)
        os.remove(comps[2]["diagnostic_trajectories_npz"])
        audit = cal.audit_seed_artifacts(comps, expectations=exps)
        assert audit["assessable"] is False and 2 in audit["missing_seeds"]


def test_corrupted_artifact_marks_seed_invalid():
    with tempfile.TemporaryDirectory() as tmp:
        comps, exps = {}, {}
        for s in (1, 2):
            comps[s], exps[s] = _build_seed(tmp, s)
        # semantic corruption of seed 2's results CSV, then rebuild its manifest
        prefix = os.path.join(tmp, "calib_N6_s2")
        _csv(prefix + "_results.csv", ["T", "C_mean", "Rg2_mean_lattice"],
             [[999.0, 3.0, 1.0], [LADDER[1], 3.0, 1.0]])
        cal.build_artifact_manifest(
            cal._full_companion_paths(prefix, comps[2]["configuration_h5"],
                                      comps[2]["feature_h5"]),
            comps[2]["artifact_manifest_json"])
        audit = cal.audit_seed_artifacts(comps, expectations=exps)
        assert audit["assessable"] is False and 2 in audit["invalid_seeds"]
        assert "artifact_validation" in audit["artifact_failures_by_seed"][2]


def test_single_seed_not_assessable():
    with tempfile.TemporaryDirectory() as tmp:
        comp, exp = _build_seed(tmp, 1)
        audit = cal.audit_seed_artifacts({1: comp}, expectations={1: exp})
        assert audit["all_requested_complete"] is True
        assert audit["assessable"] is False
