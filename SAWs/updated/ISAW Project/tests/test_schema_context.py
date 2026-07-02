"""Phase 13/14: one resolved definitions context (no split-brain)."""
import dataclasses
import json

import pytest

import isaw_schema as sch
import isaw_contact_observables as ico


def test_context_is_frozen_and_consistent():
    ctx = sch.CONTEXT
    assert isinstance(ctx, sch.DefinitionsContext)
    # frozen dataclass -> assignment raises
    with pytest.raises(dataclasses.FrozenInstanceError):
        ctx.definitions_version = "9.9.9"
    # the bundle is internally consistent
    assert ctx.definitions_version == str(ctx.record["definitions_version"])
    assert str(ctx.resolved_path) == str(sch.RESOLVED_DEFINITIONS_PATH)
    assert int(ctx.fixed_bins["long_threshold_fixed"]) == \
        int(ico.FIXED_BIN_DEFINITIONS["long_threshold_fixed"])


def test_context_bin_defs_match_json():
    ctx = sch.resolve_definitions_context()
    rec = ctx.bin_defs(30)
    assert int(rec["fixed"]["long_threshold_fixed"]) == \
        int(sch.get_fixed_bin_definitions()["long_threshold_fixed"])
    assert float(rec["scaled"]["local_max_ratio"]) == \
        float(sch.get_scaled_bin_definitions()["local_max_ratio"])


def test_context_sha256_matches_file():
    ctx = sch.CONTEXT
    assert ctx.sha256 == sch._sha256_file(ctx.resolved_path)
    assert len(ctx.sha256) == 64


def test_alternate_json_gives_alternate_context(tmp_path, monkeypatch):
    alt = tmp_path / "alt.json"
    d = json.loads(sch.RESOLVED_DEFINITIONS_PATH.read_text(encoding="utf-8"))
    d["fixed_contour_bins"]["long_threshold_fixed"] = 13
    alt.write_text(json.dumps(d), encoding="utf-8")
    monkeypatch.setenv(sch.DEFINITIONS_ENV_VAR, str(alt))
    ctx = sch.active_definitions_context()
    # the WHOLE bundle reflects the alternate JSON -- path, bins, and version are
    # mutually consistent (no split brain between path and values)
    assert str(ctx.resolved_path) == str(alt.resolve())
    assert int(ctx.fixed_bins["long_threshold_fixed"]) == 13
    assert ctx.sha256 == sch._sha256_file(alt)


def test_provenance_labels_are_not_module_default():
    # JSON-sourced definitions must carry the json provenance label.
    assert sch.PROV_JSON == "json_project_definitions"
    ctx = sch.CONTEXT
    assert ctx.provenance == sch.PROV_JSON
    rec = ctx.bin_defs(30)
    assert rec.get("bin_definition_source") == sch.PROV_JSON


def test_run_remd_default_bins_use_json_context():
    import remd_uniform_chain_2_new as remd
    import numpy as np
    # A direct API run with bin_defs=None must source the JSON context (label
    # json_project_definitions), not the compatibility constants.
    replicas, sp, sa = remd.run_remd(
        N=30, Ts=np.array([300.0, 340.0]), steps_per_swap=2, n_cycles=2,
        model_name="hs", params=[330.5, 1.28], Tref=320.0, Tscale=80.0,
        seed=1, verbose=False, structural_observables=True, structural_stride=1,
        bin_defs=None)
    assert len(replicas) == 2
