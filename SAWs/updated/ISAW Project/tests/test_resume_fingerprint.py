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


# --- stage_reusable end-to-end (FULL artifact validation, Part 2) ------------
# stage_reusable must require full artifact-manifest hash + semantic validation,
# not merely a trusted certificate and present filenames.

import csv as _csv

FP = "fp-" + "abc123" * 10   # >0-length deterministic fingerprint
LADDER = [300.0, 350.0]
MODEL = {"model_name": "hs", "param_names": ["h", "s"], "params": [647.7, 1.874],
         "Tref": 320.0, "Tscale": 80.0}


def _wcsv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = _csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)


def _build_full_stage(tmp):
    import remd_uniform_chain_2_new as remd
    from pathlib import Path
    nb, nT = len(HAIRPIN6), len(LADDER)
    cp, _ = ico.build_contact_map(HAIRPIN6)
    m = cp.shape[0]
    rg2 = ico.radius_of_gyration_squared(HAIRPIN6)
    ree2 = ico.end_to_end_distance_squared(HAIRPIN6)
    prefix = os.path.join(tmp, "calib_N6_s1")
    cfg = prefix + "_configurations.h5"
    feat = prefix + "_features.h5"
    meta = {"run_id": "cf", "seed": 1, "temperatures": LADDER}
    meta.update({"model_name": MODEL["model_name"],
                 "param_names": MODEL["param_names"],
                 "model_params": MODEL["params"],
                 "Tref": MODEL["Tref"], "Tscale": MODEL["Tscale"]})
    w = cio.SnapshotWriter(cfg, n_beads=nb, n_temperatures=nT, metadata=meta)
    for c in range(3):
        w.append(cycle=c,
                 coordinates=np.stack([np.asarray(HAIRPIN6, dtype=np.int64)] * nT),
                 walker_id=np.array([0, 1] if c % 2 == 0 else [1, 0]),
                 contacts=np.full(nT, m), rg2_lattice=np.full(nT, rg2),
                 ree2_lattice=np.full(nT, ree2))
    w.mark_complete()
    w.close()
    fields = {"model_name": MODEL["model_name"], "seed": 1}
    cal._embed_fingerprint_hdf5(cfg, FP, fields)
    ext.extract(cfg, feat, validate=True, output_format="hdf5", overwrite=True)
    cal._embed_fingerprint_feature(feat, FP, fields)
    cal._deep_validate_and_certify(feat, stage_fingerprint=FP,
                                   stage_fingerprint_fields=fields,
                                   source_config=cfg)
    _wcsv(prefix + "_results.csv", ["T", "C_mean", "Rg2_mean_lattice"],
          [[LADDER[i], 3.0, rg2] for i in range(nT)])
    _wcsv(prefix + "_swap_rates.csv",
          ["pair", "T_lo", "T_hi", "proposals", "acceptances", "rate"],
          [[0, LADDER[0], LADDER[1], 10, 5, "0.5"]])
    ma = [[i, LADDER[i], mn, 10, 8, 6] for i in range(nT) for mn in remd.MOVE_NAMES]
    _wcsv(prefix + "_move_acceptance.csv",
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
    np.savez(prefix + "_distributions.npz", Ts=np.asarray(LADDER),
             temps=np.asarray(LADDER), Pc=np.zeros((nT, 5)), Prg=np.zeros((nT, 7)),
             model_name=np.array(MODEL["model_name"]),
             param_names=np.array(MODEL["param_names"]),
             model_params=np.array(MODEL["params"], float),
             Tref=np.array(MODEL["Tref"]), Tscale=np.array(MODEL["Tscale"]),
             n_beads=np.array(nb))
    Path(prefix + "_run_summary.json").write_text(json.dumps({
        "N": nb, "n_beads": nb, "seed": 1, "temperatures": LADDER,
        "model": MODEL["model_name"], "param_names": MODEL["param_names"],
        "params": MODEL["params"], "Tref": MODEL["Tref"],
        "Tscale": MODEL["Tscale"], "burnin_frac": 0.5, "n_cycles": 400,
        "steps_per_swap": 60, "snapshot_stride": 5}), encoding="utf-8")
    comp = cal._companion_paths(prefix, cfg, feat)
    cal.build_artifact_manifest(cal._full_companion_paths(prefix, cfg, feat),
                                comp["artifact_manifest_json"])
    info = {"model_name": MODEL["model_name"], "param_names": MODEL["param_names"],
            "params": MODEL["params"], "Tref": MODEL["Tref"],
            "Tscale": MODEL["Tscale"]}
    record = cal.build_expected_stage_record(
        N=nb, seed=1, run_id=None, ladder=LADDER, K_ladder=[0.0, 0.0], info=info,
        fit_summary_sha256=None, burnin_frac=0.5, n_cycles=400, steps_per_swap=60,
        structural_stride=5, snapshot_stride=5, stage_fingerprint=FP)
    return prefix, cfg, feat, comp, record


def _reusable(cfg, feat, comp, record, **over):
    kw = dict(feature_path=feat, source_config=cfg,
              artifact_manifest_path=comp["artifact_manifest_json"],
              expected_N=6, expected_seed=1, expected_ladder=LADDER,
              expected_stage_fingerprint=FP, expected_run_id=None,
              expected_model_record=record, sampler_controls={},
              definitions_context=cal.sch.active_definitions_context(),
              companions=comp)
    kw.update(over)
    return cal.stage_reusable(**kw)


def test_stage_reusable_accepts_matching_and_rejects_mismatch():
    with tempfile.TemporaryDirectory() as tmp:
        prefix, cfg, feat, comp, record = _build_full_stage(tmp)
        ok, reason = _reusable(cfg, feat, comp, record)
        assert ok is True, reason
        ok2, reason2 = _reusable(cfg, feat, comp, record,
                                 expected_stage_fingerprint="fp-different")
        assert ok2 is False and "fingerprint" in reason2


def test_stage_reusable_rejects_missing_companion():
    with tempfile.TemporaryDirectory() as tmp:
        prefix, cfg, feat, comp, record = _build_full_stage(tmp)
        os.remove(comp["diagnostic_trajectories_npz"])
        ok, reason = _reusable(cfg, feat, comp, record)
        assert ok is False and "companion" in reason


def test_stage_reusable_rejects_changed_source_hash():
    with tempfile.TemporaryDirectory() as tmp:
        prefix, cfg, feat, comp, record = _build_full_stage(tmp)
        with open(cfg, "ab") as fh:
            fh.write(b"\x00")
        ok, reason = _reusable(cfg, feat, comp, record)
        assert ok is False and "source configuration hash" in reason


def test_stage_reusable_rejects_corrupted_artifact_hash():
    # File changed WITHOUT updating the artifact-manifest hash -> hash check fails.
    with tempfile.TemporaryDirectory() as tmp:
        prefix, cfg, feat, comp, record = _build_full_stage(tmp)
        with open(prefix + "_results.csv", "a", encoding="utf-8") as f:
            f.write("\n# tamper\n")
        ok, reason = _reusable(cfg, feat, comp, record)
        assert ok is False and "artifact validation failed" in reason


def test_stage_reusable_rejects_semantic_corruption_with_updated_hash():
    # File changed AND manifest hash updated -> semantic validation must fail.
    with tempfile.TemporaryDirectory() as tmp:
        prefix, cfg, feat, comp, record = _build_full_stage(tmp)
        _wcsv(prefix + "_results.csv", ["T", "C_mean", "Rg2_mean_lattice"],
              [[999.0, 3.0, 1.0], [LADDER[1], 3.0, 1.0]])
        cal.build_artifact_manifest(
            cal._full_companion_paths(prefix, cfg, feat),
            comp["artifact_manifest_json"])
        ok, reason = _reusable(cfg, feat, comp, record)
        assert ok is False and "artifact validation failed" in reason
