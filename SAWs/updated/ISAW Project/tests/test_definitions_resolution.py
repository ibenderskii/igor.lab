"""Phase 1: definitions-file resolution, completeness, doc/feature-dict artifacts."""
import json
import os
from pathlib import Path

import pytest

import isaw_schema as sch
import extract_contact_motif_features as ext


def test_root_definitions_resolves_by_default(monkeypatch):
    monkeypatch.delenv(sch.DEFINITIONS_ENV_VAR, raising=False)
    p = sch.resolve_definitions_path()
    assert p.name == "project_definitions.json"
    assert p.exists()


def test_explicit_path_takes_precedence(tmp_path, monkeypatch):
    monkeypatch.delenv(sch.DEFINITIONS_ENV_VAR, raising=False)
    custom = tmp_path / "custom_defs.json"
    custom.write_text(sch.RESOLVED_DEFINITIONS_PATH.read_text(encoding="utf-8"),
                      encoding="utf-8")
    assert sch.resolve_definitions_path(str(custom)) == custom.resolve()


def test_env_var_precedence(tmp_path, monkeypatch):
    custom = tmp_path / "env_defs.json"
    custom.write_text(sch.RESOLVED_DEFINITIONS_PATH.read_text(encoding="utf-8"),
                      encoding="utf-8")
    monkeypatch.setenv(sch.DEFINITIONS_ENV_VAR, str(custom))
    assert sch.resolve_definitions_path() == custom.resolve()


def test_missing_definitions_lists_attempts(tmp_path, monkeypatch):
    # The repo-root file always exists, so force an all-miss scenario by
    # pointing every candidate at nonexistent paths.
    cands = [tmp_path / "a.json", tmp_path / "b.json"]
    monkeypatch.setattr(sch, "_candidate_definition_paths", lambda explicit=None: cands)
    with pytest.raises(sch.SchemaError) as e:
        sch.resolve_definitions_path()
    msg = str(e.value)
    assert "a.json" in msg and "b.json" in msg


def test_explicit_missing_falls_back_to_root(monkeypatch):
    # An explicit path that does not exist falls through to the root file
    # (backward compatibility), never silently loading a CWD file.
    monkeypatch.delenv(sch.DEFINITIONS_ENV_VAR, raising=False)
    p = sch.resolve_definitions_path("/no/such/definitions.json")
    assert p.name == "project_definitions.json" and p.exists()


def test_load_records_resolved_path():
    defs = sch.load_project_definitions()
    assert Path(defs["_resolved_path"]).exists()


def test_consistency_full_record():
    sch.check_definitions_consistency()   # raises on any incompleteness/divergence


def test_output_schema_versions_match_code():
    ov = sch.output_schema_versions()
    import isaw_config_io as cio
    import remd_uniform_chain_2_new as remd
    assert ov["feature_schema_version"] == ext.FEATURE_SCHEMA_VERSION
    assert ov["snapshot_schema_version"] == cio.SNAPSHOT_SCHEMA_VERSION
    assert ov["distributions_schema_version"] == remd.SCHEMA_VERSION
    assert ov["model_api_version"] == remd.MODEL_API_VERSION


def test_feature_dictionary_matches_committed_file():
    p = Path(ext.__file__).resolve().parent / "docs" / "feature_dictionary.json"
    assert p.exists(), "run `python -c 'import extract...; write_feature_dictionary()'`"
    committed = json.loads(p.read_text(encoding="utf-8"))
    assert committed == ext.build_feature_dictionary()
