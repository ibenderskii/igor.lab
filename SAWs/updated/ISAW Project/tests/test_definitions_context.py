"""Phase 8: one deeply-immutable resolved definitions context.

The context cannot be mutated even one level deep, resolution honors the
documented precedence (env override / explicit path, unrelated CWD ignored), and
the stage fingerprint is driven by the context (not a mix of frozen globals).
"""
import json
import os

import pytest

import isaw_schema as sch
import run_structural_regime_pilot as cal

INFO = {"model_name": "hs", "param_names": ["h", "s"], "params": [330.5, 1.28],
        "Tref": 320.0, "Tscale": 80.0}


def _fp_kwargs(fit):
    return dict(N=30, seed=1, ladder=[300.0, 320.0, 340.0],
                K_ladder=[0.1, 0.2, 0.3], info=INFO, fit_summary_path=fit,
                n_cycles=100, steps_per_swap=8, n_workers=1,
                structural_stride=2, snapshot_stride=5, burnin_frac=0.5)


def test_context_is_recursively_immutable():
    ctx = sch.CONTEXT
    with pytest.raises(TypeError):
        ctx.fixed_bins["long_threshold_fixed"] = 99
    with pytest.raises(TypeError):
        ctx.scaled_bins["local_max_ratio"] = 0.5
    with pytest.raises(TypeError):
        ctx.record["definitions_version"] = "9.9.9"
    # nested mapping is frozen too
    with pytest.raises(TypeError):
        ctx.record["fixed_contour_bins"]["long_threshold_fixed"] = 1


def test_bin_defs_returns_mutable_copy():
    rec = sch.CONTEXT.bin_defs(30)
    rec["fixed"]["long_threshold_fixed"] = 999   # thawed copy, safe to edit
    assert int(sch.CONTEXT.fixed_bins["long_threshold_fixed"]) != 999


def test_explicit_path_and_unrelated_cwd_ignored(tmp_path, monkeypatch):
    # An unrelated project_definitions.json in the CWD must be ignored.
    (tmp_path / "project_definitions.json").write_text("{}", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(sch.DEFINITIONS_ENV_VAR, raising=False)
    ctx = sch.active_definitions_context()
    assert ctx.resolved_path == sch.RESOLVED_DEFINITIONS_PATH
    assert int(ctx.fixed_bins["long_threshold_fixed"]) == 15


def test_env_override_changes_context(tmp_path, monkeypatch):
    alt = tmp_path / "alt.json"
    d = json.loads(sch.RESOLVED_DEFINITIONS_PATH.read_text(encoding="utf-8"))
    d["fixed_contour_bins"]["long_threshold_fixed"] = 13
    alt.write_text(json.dumps(d), encoding="utf-8")
    monkeypatch.setenv(sch.DEFINITIONS_ENV_VAR, str(alt))
    ctx = sch.active_definitions_context()
    assert ctx.resolved_path == alt.resolve()
    assert int(ctx.fixed_bins["long_threshold_fixed"]) == 13
    assert ctx.sha256 == sch._sha256_file(alt)


def test_fingerprint_uses_context(tmp_path):
    fit = tmp_path / "fit.json"
    fit.write_text(json.dumps({"model": "hs", "params": [1.0]}), encoding="utf-8")
    base = _fp_kwargs(str(fit))
    f_default = cal._stage_fingerprint(
        cal.build_stage_fingerprint_fields(**base))
    # An alternate context (different definitions hash/version) changes the fp.
    alt = tmp_path / "alt.json"
    d = json.loads(sch.RESOLVED_DEFINITIONS_PATH.read_text(encoding="utf-8"))
    d["definitions_version"] = "9.9.9"
    alt.write_text(json.dumps(d), encoding="utf-8")
    alt_ctx = sch.resolve_definitions_context(str(alt))
    f_alt = cal._stage_fingerprint(
        cal.build_stage_fingerprint_fields(**base, context=alt_ctx))
    assert f_alt != f_default
    # Same context -> same fingerprint.
    assert cal._stage_fingerprint(
        cal.build_stage_fingerprint_fields(**base, context=sch.CONTEXT)) == f_default


def test_fingerprint_definitions_fields_match_context():
    fields = cal.build_stage_fingerprint_fields(
        **_fp_kwargs("x.json"), context=sch.CONTEXT)
    assert fields["definitions_path"] == str(sch.CONTEXT.resolved_path)
    assert fields["definitions_sha256"] == sch.CONTEXT.sha256
    assert fields["definitions_version"] == sch.CONTEXT.definitions_version
    assert fields["feature_schema_version"] == int(
        sch.CONTEXT.schema_versions["feature_schema_version"])
