"""Part 13/14: generated docs are deterministic and cover every output field."""
import json
from pathlib import Path

import isaw_schema as sch
import extract_contact_motif_features as ext
import run_structural_regime_pilot as cal

DOCS = Path(ext.__file__).resolve().parent / "docs"


def test_definitions_md_matches_committed():
    committed = (DOCS / "definitions.md").read_text(encoding="utf-8")
    assert committed == sch.render_definitions_md()


def test_artifact_provenance_md_matches_committed():
    committed = (DOCS / "artifact_provenance.md").read_text(encoding="utf-8")
    assert committed == cal.render_artifact_provenance_md()


def test_artifact_provenance_regeneration_is_deterministic():
    assert cal.render_artifact_provenance_md() == cal.render_artifact_provenance_md()


def test_feature_dictionary_matches_committed():
    committed = json.loads((DOCS / "feature_dictionary.json").read_text(encoding="utf-8"))
    assert committed == ext.build_feature_dictionary()


def test_every_output_field_has_exactly_one_entry():
    fd = ext.build_feature_dictionary()
    names = [x["name"] for x in fd["fields"]]
    assert len(names) == len(set(names)), "duplicate dictionary entries"
    output_fields = (set(ext.FEATURE_COLUMNS)
                     | {"m_r", "contact_pairs/pairs", "contact_pairs/offsets"})
    for f in output_fields:
        assert names.count(f) == 1, f"{f} not documented exactly once"


def test_no_entry_points_to_a_nonexistent_field():
    fd = ext.build_feature_dictionary()
    known = (set(ext.FEATURE_COLUMNS)
             | {"m_r", "contact_pairs/pairs", "contact_pairs/offsets"}
             | set(fd["provenance_metadata_fields"]))
    for x in fd["fields"]:
        assert x["name"] in known, f"dictionary entry {x['name']} has no field"


def test_every_entry_has_all_required_keys():
    fd = ext.build_feature_dictionary()
    required = {"name", "dtype", "units", "mathematical_definition",
                "source_representation", "cadence", "nullable",
                "schema_version_introduced", "validation_identity"}
    for x in fd["fields"]:
        assert required <= set(x), f"{x['name']} missing keys {required - set(x)}"
        # No placeholder definition mechanically derived from the field name.
        assert x["mathematical_definition"] and \
            x["mathematical_definition"] != x["name"].replace("_", " ")
