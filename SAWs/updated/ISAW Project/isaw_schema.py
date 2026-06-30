#!/usr/bin/env python3
"""Shared authoritative schema / definitions layer (computational plan P00).

Loads ``project_definitions.json`` (the single source of frozen definitions for
the PNIPAM ISAW contact-motif analysis) and exposes deep-copied,
immutable-by-convention definitions plus validators used across the pipeline.

Repository note
---------------
The repository is currently a flat layout (no ``src/isaw`` package).  This
module provides the shared definitions/validation surface that the plan calls
``src/isaw/schema.py`` without an uncontrolled reorganization; a thin
``src.isaw.schema`` compatibility import can be added later if the repo is
packaged.

Key conventions (also frozen in the JSON)
-----------------------------------------
* N = n_beads ; n_steps = N - 1
* b(T) = reduced contact bias ; K(T) = -b(T) ; q(T) = exp(K(T))
* P(C|T) ∝ exp[K(T) m(C)]  -> HIGHER K FAVORS MORE CONTACTS
* PNIPAM is LCST: collapse is expected as temperature INCREASES over the fitted
  range (do not assume lower T is more collapsed).
"""
from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path

import numpy as np

import isaw_contact_observables as ico

_HERE = Path(__file__).resolve().parent
PROJECT_DEFINITIONS_PATH = _HERE / "project_definitions.json"

PRIMARY_KEY = ("run_id", "seed", "snapshot_index", "temperature_index")


class SchemaError(ValueError):
    """Raised when definitions, metadata, or feature tables fail validation."""


# ---------------------------------------------------------------------------
# Load definitions
# ---------------------------------------------------------------------------

def load_project_definitions(path: str | os.PathLike | None = None) -> dict:
    """Load and minimally validate the project definitions JSON (deep-copied)."""
    p = Path(path) if path is not None else PROJECT_DEFINITIONS_PATH
    if not p.exists():
        raise SchemaError(f"project definitions file not found: {p}")
    with open(p, encoding="utf-8") as fh:
        defs = json.load(fh)
    for key in ("project_name", "schema_version", "definitions_version",
                "fixed_contour_bins", "scaled_contour_bins",
                "primary_key_definition", "required_metadata_fields"):
        if key not in defs:
            raise SchemaError(f"project definitions missing required key {key!r}")
    return copy.deepcopy(defs)


_DEFS = load_project_definitions()

PROJECT_SCHEMA_VERSION = int(_DEFS["schema_version"])
DEFINITIONS_VERSION = str(_DEFS["definitions_version"])
_VERSIONS = _DEFS.get("output_schema_versions", {})
FEATURE_SCHEMA_VERSION = int(_VERSIONS.get("feature_schema_version", 4))
SNAPSHOT_SCHEMA_VERSION = int(_VERSIONS.get("snapshot_schema_version", 3))
DIAGNOSTIC_TRAJECTORY_SCHEMA_VERSION = int(
    _VERSIONS.get("diagnostic_trajectory_schema_version", 2))
CALIBRATION_REPORT_SCHEMA_VERSION = int(
    _VERSIONS.get("calibration_report_schema_version", 1))
MODEL_API_VERSION = int(_VERSIONS.get("model_api_version", 1))

REQUIRED_METADATA_FIELDS = tuple(_DEFS["required_metadata_fields"])


# ---------------------------------------------------------------------------
# Definition accessors (deep copies)
# ---------------------------------------------------------------------------

def get_fixed_bin_definitions() -> dict:
    return copy.deepcopy(ico.FIXED_BIN_DEFINITIONS)


def get_scaled_bin_definitions() -> dict:
    return copy.deepcopy(ico.SCALED_BIN_DEFINITIONS)


def project_definitions() -> dict:
    return copy.deepcopy(_DEFS)


def validate_bin_definitions(n_beads, fixed_defs=None, scaled_defs=None):
    return ico.validate_bin_definitions(n_beads, fixed_defs, scaled_defs)


# ---------------------------------------------------------------------------
# Code <-> JSON consistency
# ---------------------------------------------------------------------------

def check_definitions_consistency() -> None:
    """Raise SchemaError if code-level definitions diverge from the JSON."""
    jf = _DEFS["fixed_contour_bins"]
    cf = ico.FIXED_BIN_DEFINITIONS
    if (int(jf["short_fixed"]["r_min"]) != int(cf["short_fixed"]["r_min"])
            or int(jf["short_fixed"]["r_max"]) != int(cf["short_fixed"]["r_max"])
            or int(jf["medium_fixed"]["r_min"]) != int(cf["medium_fixed"]["r_min"])
            or int(jf["long_threshold_fixed"]) != int(cf["long_threshold_fixed"])):
        raise SchemaError("fixed bin definitions diverge between JSON and code")
    js = _DEFS["scaled_contour_bins"]
    cs = ico.SCALED_BIN_DEFINITIONS
    if (float(js["local_max_ratio"]) != float(cs["local_max_ratio"])
            or float(js["meso_max_ratio"]) != float(cs["meso_max_ratio"])):
        raise SchemaError("scaled bin definitions diverge between JSON and code")
    if str(_DEFS["definitions_version"]) != str(ico.DEFINITIONS_VERSION):
        raise SchemaError(
            f"definitions_version diverges: JSON {_DEFS['definitions_version']} "
            f"!= code {ico.DEFINITIONS_VERSION}")
    if list(_DEFS["primary_key_definition"]) != list(PRIMARY_KEY):
        raise SchemaError("primary_key_definition diverges between JSON and code")


# ---------------------------------------------------------------------------
# Metadata / primary-key validation
# ---------------------------------------------------------------------------

def validate_run_metadata(meta: dict) -> None:
    """Ensure required run-provenance metadata fields are present and sane."""
    missing = [k for k in REQUIRED_METADATA_FIELDS if k not in meta]
    if missing:
        raise SchemaError(f"run metadata missing required field(s): {missing}")
    nb = int(meta["n_beads"])
    if "n_steps" in meta and int(meta["n_steps"]) != nb - 1:
        raise SchemaError(
            f"n_steps {meta['n_steps']} != n_beads-1 ({nb - 1})")
    temps = np.asarray(meta["temperatures"], dtype=float).ravel()
    if temps.size == 0 or not np.all(np.isfinite(temps)):
        raise SchemaError("temperatures must be a nonempty finite array")


def validate_feature_primary_keys(columns: dict) -> None:
    """Validate that the primary key (run_id, seed, snapshot_index,
    temperature_index) is unique across all feature rows.

    ``columns`` maps each primary-key name to a 1-D sequence of equal length.
    """
    for key in PRIMARY_KEY:
        if key not in columns:
            raise SchemaError(f"feature table missing primary-key column {key!r}")
    lengths = {len(np.asarray(columns[k]).ravel()) for k in PRIMARY_KEY}
    if len(lengths) != 1:
        raise SchemaError(f"primary-key columns have unequal lengths: {lengths}")
    n = lengths.pop()
    keys = list(zip(*[
        [str(v) for v in np.asarray(columns[k]).ravel().tolist()]
        for k in PRIMARY_KEY]))
    if len(set(keys)) != n:
        raise SchemaError(
            f"feature primary key is not unique: {n - len(set(keys))} "
            f"duplicate row(s)")


# ---------------------------------------------------------------------------
# Thermodynamic helpers (K is authoritative; q may overflow)
# ---------------------------------------------------------------------------

def reduced_potential_from_K(m, K) -> float:
    """u(C,T) = -m K(T)."""
    return float(-float(m) * float(K))


def q_from_K(K) -> float:
    """q = exp(K); +inf on overflow (K authoritative)."""
    with np.errstate(over="ignore"):
        q = math.exp(K) if abs(K) < 700 else (math.inf if K > 0 else 0.0)
    return float(q)


# ---------------------------------------------------------------------------
# JSON-safe conversion
# ---------------------------------------------------------------------------

def render_definitions_md(defs: dict | None = None) -> str:
    """Render a deterministic markdown summary of the frozen definitions.

    ``docs/definitions.md`` is generated from this; a test re-renders and
    compares so the doc can never silently diverge from the JSON.
    """
    d = defs if defs is not None else _DEFS
    f = d["fixed_contour_bins"]
    s = d["scaled_contour_bins"]
    lines = [
        f"# {d['project_name']} — frozen definitions",
        "",
        "> Generated from `project_definitions.json` by "
        "`isaw_schema.render_definitions_md()`. Do not edit by hand; edit the "
        "JSON and regenerate (`python isaw_schema.py`).",
        "",
        f"- schema_version: {d['schema_version']}",
        f"- definitions_version: {d['definitions_version']}",
        f"- polymer_system: {d['polymer_system']}",
        f"- transition_type: {d['transition_type']}",
        "",
        "## Chain-length convention",
        "- N = n_beads; n_steps = N - 1.",
        "",
        "## Reduced bias, K, q",
        f"- {d['reduced_bias_definition']}",
        f"- {d['K_definition']}",
        f"- {d['q_definition']}",
        f"- {d['reduced_potential_definition']}",
        f"- {d['effective_energy_definition']}",
        "",
        "## PNIPAM LCST expectations (verify against the fitted model; do not hardcode)",
    ] + [f"- {x}" for x in d["pnipam_lcst_expectations"]["over_fitted_range"]] + [
        f"- primary contact-favoring coordinate: "
        f"{d['pnipam_lcst_expectations']['primary_contact_favoring_coordinate']}",
        "",
        "## Fixed contour bins",
        f"- short_fixed: {f['short_fixed']['r_min']} <= r <= {f['short_fixed']['r_max']}",
        f"- medium_fixed: {f['medium_fixed']['r_min']} <= r < long_threshold_fixed",
        f"- long_fixed: r >= {f['long_threshold_fixed']}",
        f"- constraints: {f['constraints']}",
        "",
        "## Scaled contour bins",
        f"- {s['boundary_inclusion']}",
        f"- constraints: {s['constraints']}",
        "",
        "## Contact / augmented graph",
        f"- contact graph: {d['contact_graph_definition']['edge_conservation']}",
        f"- augmented graph identities: "
        f"{d['augmented_graph_definition']['identities_connected_chain']}",
        "",
        "## Primary key",
        f"- {tuple(d['primary_key_definition'])}",
        "",
        "## Required run metadata fields",
        f"- {', '.join(d['required_metadata_fields'])}",
        "",
    ]
    return "\n".join(lines)


def json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [json_safe(v) for v in obj.tolist()]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


DOCS_DEFINITIONS_PATH = _HERE / "docs" / "definitions.md"


def write_definitions_md() -> str:
    DOCS_DEFINITIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    text = render_definitions_md()
    DOCS_DEFINITIONS_PATH.write_text(text, encoding="utf-8")
    return str(DOCS_DEFINITIONS_PATH)


if __name__ == "__main__":
    check_definitions_consistency()
    print("Wrote", write_definitions_md())
