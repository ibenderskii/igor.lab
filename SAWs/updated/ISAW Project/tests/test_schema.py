"""Part 14: schema/definitions authority, resolution, and consistency."""
import copy
import json
from pathlib import Path

import pytest

import isaw_schema as sch
import isaw_contact_observables as ico


def _alt_defs(long_threshold=13):
    """A valid alternate definitions record with a different fixed threshold."""
    d = json.loads(sch.RESOLVED_DEFINITIONS_PATH.read_text(encoding="utf-8"))
    d["fixed_contour_bins"]["long_threshold_fixed"] = long_threshold
    return d


# --- resolution precedence --------------------------------------------------

def test_path_precedence_explicit_over_env(tmp_path, monkeypatch):
    a = tmp_path / "a.json"; b = tmp_path / "b.json"
    a.write_text(sch.RESOLVED_DEFINITIONS_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    b.write_text(sch.RESOLVED_DEFINITIONS_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setenv(sch.DEFINITIONS_ENV_VAR, str(b))
    assert sch.resolve_definitions_path(str(a)) == a.resolve()   # explicit wins


def test_environment_override(tmp_path, monkeypatch):
    env = tmp_path / "env.json"
    env.write_text(sch.RESOLVED_DEFINITIONS_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setenv(sch.DEFINITIONS_ENV_VAR, str(env))
    assert sch.resolve_definitions_path() == env.resolve()


def test_root_fallback(monkeypatch):
    monkeypatch.delenv(sch.DEFINITIONS_ENV_VAR, raising=False)
    p = sch.resolve_definitions_path()
    assert p.name == "project_definitions.json" and p.exists()


def test_unrelated_cwd_file_ignored(tmp_path, monkeypatch):
    # A project_definitions.json in the CWD must NOT be picked up (resolution is
    # relative to the module directory, not the process CWD).
    monkeypatch.delenv(sch.DEFINITIONS_ENV_VAR, raising=False)
    (tmp_path / "project_definitions.json").write_text("{}", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    p = sch.resolve_definitions_path()
    assert p == sch.RESOLVED_DEFINITIONS_PATH.resolve()
    assert p.parent != tmp_path


# --- alternate definitions used at runtime ----------------------------------

def test_alternate_definitions_used_at_runtime(tmp_path, monkeypatch):
    alt = tmp_path / "alt.json"
    alt.write_text(json.dumps(_alt_defs(13)), encoding="utf-8")
    monkeypatch.setenv(sch.DEFINITIONS_ENV_VAR, str(alt))
    # accessors reflect the alternate JSON
    fixed = sch.get_fixed_bin_definitions()
    assert int(fixed["long_threshold_fixed"]) == 13
    scaled = sch.get_scaled_bin_definitions()
    # runtime normalization uses those values (valid partition for N=30)
    rec = ico.normalize_bin_definitions(fixed, scaled, n_beads=30)
    assert int(rec["fixed"]["long_threshold_fixed"]) == 13
    assert str(sch.active_definitions_path()) == str(alt.resolve())


def test_default_accessors_match_committed_json():
    fixed = sch.get_fixed_bin_definitions()
    assert int(fixed["long_threshold_fixed"]) == \
        int(ico.FIXED_BIN_DEFINITIONS["long_threshold_fixed"])
    scaled = sch.get_scaled_bin_definitions()
    assert float(scaled["local_max_ratio"]) == \
        float(ico.SCALED_BIN_DEFINITIONS["local_max_ratio"])


# --- consistency ------------------------------------------------------------

def test_consistency_passes_on_committed():
    sch.check_definitions_consistency()


def test_complete_consistency_failure_divergent(monkeypatch):
    broken = copy.deepcopy(sch._DEFS)
    broken["fixed_contour_bins"]["long_threshold_fixed"] = 999   # diverges from code
    monkeypatch.setattr(sch, "_DEFS", broken)
    with pytest.raises(sch.SchemaError):
        sch.check_definitions_consistency()


def test_complete_consistency_failure_incomplete(monkeypatch):
    broken = copy.deepcopy(sch._DEFS)
    del broken["contact_definition"]                             # missing record
    monkeypatch.setattr(sch, "_DEFS", broken)
    with pytest.raises(sch.SchemaError):
        sch.check_definitions_consistency()


def test_output_schema_versions_accessor():
    ov = sch.output_schema_versions()
    assert ov["feature_schema_version"] == sch.FEATURE_SCHEMA_VERSION
