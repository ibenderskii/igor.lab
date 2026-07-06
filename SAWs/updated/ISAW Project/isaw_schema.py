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
import hashlib
import json
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

import numpy as np

import isaw_contact_observables as ico

_HERE = Path(__file__).resolve().parent
# Backward-compatible name: the repository-root definitions file.
PROJECT_DEFINITIONS_PATH = _HERE / "project_definitions.json"
# Environment variable that overrides the search (2nd precedence).
DEFINITIONS_ENV_VAR = "ISAW_PROJECT_DEFINITIONS"

PRIMARY_KEY = ("run_id", "seed", "snapshot_index", "temperature_index")


class SchemaError(ValueError):
    """Raised when definitions, metadata, or feature tables fail validation."""


# ---------------------------------------------------------------------------
# Definitions-file resolution (Phase 1.1)
# ---------------------------------------------------------------------------
# One explicit precedence, resolved relative to this module's directory (the
# repository root in the current flat layout), NOT the process CWD -- so an
# unrelated ``project_definitions.json`` in the working directory is never
# silently loaded:
#     1. an explicit path argument;
#     2. the ISAW_PROJECT_DEFINITIONS environment variable;
#     3. configs/project_definitions.json (the packaged location);
#     4. project_definitions.json at the repository root (backward compatible).

def _candidate_definition_paths(explicit=None) -> list[Path]:
    cands: list[Path] = []
    if explicit is not None:
        p = Path(explicit)
        cands.append(p if p.is_absolute() else (_HERE / p))
    env = os.environ.get(DEFINITIONS_ENV_VAR)
    if env:
        p = Path(env)
        cands.append(p if p.is_absolute() else (_HERE / p))
    cands.append(_HERE / "configs" / "project_definitions.json")
    cands.append(_HERE / "project_definitions.json")
    return cands


def resolve_definitions_path(explicit=None) -> Path:
    """Return the first existing definitions file by the documented precedence.

    Raises :class:`SchemaError` listing every attempted path if none exists.
    """
    attempted = _candidate_definition_paths(explicit)
    for p in attempted:
        if p.exists():
            return p.resolve()
    raise SchemaError(
        "no project definitions file found; attempted (in order): "
        + "; ".join(str(p) for p in attempted)
        + f" (set {DEFINITIONS_ENV_VAR} or place the file at one of these paths)"
    )


# ---------------------------------------------------------------------------
# Load definitions
# ---------------------------------------------------------------------------

def load_project_definitions(path: str | os.PathLike | None = None) -> dict:
    """Load and minimally validate the project definitions JSON (deep-copied).

    ``path`` (if given) is honored first; otherwise the documented resolution
    precedence applies.  The resolved absolute path is stored on the returned
    record under ``_resolved_path`` so manifests can record provenance.
    """
    p = resolve_definitions_path(path)
    with open(p, encoding="utf-8") as fh:
        defs = json.load(fh)
    for key in ("project_name", "schema_version", "definitions_version",
                "fixed_contour_bins", "scaled_contour_bins",
                "primary_key_definition", "required_metadata_fields"):
        if key not in defs:
            raise SchemaError(f"project definitions missing required key {key!r}")
    out = copy.deepcopy(defs)
    out["_resolved_path"] = str(p)
    return out


_DEFS = load_project_definitions()
RESOLVED_DEFINITIONS_PATH = Path(_DEFS["_resolved_path"])

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
# Definition accessors (Phase 11: authoritative, derived from the JSON)
# ---------------------------------------------------------------------------
# The runtime bin definitions are DERIVED FROM THE JSON, not copied from the
# hardcoded ``isaw_contact_observables`` constants (those remain only as a
# compatibility fallback that ``check_definitions_consistency`` cross-checks).
# The accessors re-resolve the active definitions file on each call so an
# ``ISAW_PROJECT_DEFINITIONS`` override (or an explicit path) is honored at
# runtime, not frozen at import time.

def _fixed_from_json(jf: dict) -> dict:
    """Map the JSON ``fixed_contour_bins`` record to the ico runtime shape."""
    thr = int(jf["long_threshold_fixed"])
    smin = int(jf["short_fixed"]["r_min"])
    smax = int(jf["short_fixed"]["r_max"])
    mmin = int(jf["medium_fixed"]["r_min"])
    return {
        "scheme": "fixed",
        "short_fixed": {"r_min": smin, "r_max": smax},
        "medium_fixed": {"r_min": mmin},
        "long_threshold_fixed": thr,
        "description": (
            f"short_fixed {smin}<=r<={smax}; medium_fixed {mmin}<=r<"
            f"long_threshold_fixed; long_fixed r>=long_threshold_fixed"),
    }


def _scaled_from_json(js: dict) -> dict:
    """Map the JSON ``scaled_contour_bins`` record to the ico runtime shape."""
    local = float(js["local_max_ratio"])
    meso = float(js["meso_max_ratio"])
    boundary = str(js.get("local_boundary", "closed"))
    return {
        "scheme": "scaled",
        "local_max_ratio": local,
        "meso_max_ratio": meso,
        "local_boundary": boundary,
        "description": (
            f"local_scaled r/N<={local}; mesoscopic_scaled {local}<r/N<{meso}; "
            f"global_scaled r/N>={meso}"),
    }


def get_fixed_bin_definitions() -> dict:
    """Authoritative fixed-bin definitions derived from the active JSON."""
    return _fixed_from_json(load_project_definitions()["fixed_contour_bins"])


def get_scaled_bin_definitions() -> dict:
    """Authoritative scaled-bin definitions derived from the active JSON."""
    return _scaled_from_json(load_project_definitions()["scaled_contour_bins"])


def active_definitions_path() -> Path:
    """The definitions file currently in effect (honors runtime overrides)."""
    return resolve_definitions_path()


# ---------------------------------------------------------------------------
# One resolved definitions context (Phase 13)
# ---------------------------------------------------------------------------
# Provenance labels for bin definitions used across the pipeline.  JSON-sourced
# definitions are NEVER labeled "module_default".
PROV_JSON = "json_project_definitions"
PROV_INPUT_HISTORICAL = "input_file_historical_definitions"
PROV_EXPLICIT_CALLER = "explicit_caller_definitions"
PROV_COMPAT_FALLBACK = "compatibility_fallback"


def _freeze(obj):
    """Recursively convert an object into a read-only structure.

    Dicts become :class:`types.MappingProxyType` (item assignment raises
    ``TypeError``) and lists/tuples become tuples, so a resolved
    :class:`DefinitionsContext` cannot be mutated even one level deep -- a frozen
    dataclass holding plain dicts is NOT sufficient.
    """
    if isinstance(obj, Mapping):
        return MappingProxyType({str(k): _freeze(v) for k, v in obj.items()})
    if isinstance(obj, (list, tuple)):
        return tuple(_freeze(v) for v in obj)
    return obj


def _thaw(obj):
    """Recursively convert a frozen structure back into mutable dict/list."""
    if isinstance(obj, Mapping):
        return {k: _thaw(v) for k, v in obj.items()}
    if isinstance(obj, tuple):
        return [_thaw(v) for v in obj]
    return obj


@dataclass(frozen=True)
class DefinitionsContext:
    """Deeply immutable bundle of everything derived from ONE definitions file.

    Resolving a context once (at process startup, or explicitly for a CLI) avoids
    a split-brain process in which some values are re-read from an alternate JSON
    while version/path globals stay frozen: a context always carries a mutually
    consistent record, path, hash, version, bins, and schema versions.  All
    mapping/sequence fields are recursively frozen (``MappingProxyType`` /
    tuples), so no consumer can mutate them.
    """
    record: MappingProxyType
    resolved_path: Path
    sha256: str
    definitions_version: str
    fixed_bins: MappingProxyType
    scaled_bins: MappingProxyType
    schema_versions: MappingProxyType
    provenance: str = PROV_JSON

    def bin_defs(self, n_beads: int) -> dict:
        """Normalized (validated) bin-definition record for N (thawed copy)."""
        rec = ico.normalize_bin_definitions(
            _thaw(self.fixed_bins), _thaw(self.scaled_bins),
            n_beads=int(n_beads), source=self.provenance)
        return rec


def _sha256_file(path: str | os.PathLike) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_definitions_context(path: str | os.PathLike | None = None
                                ) -> DefinitionsContext:
    """Resolve a complete, immutable definitions context from ONE JSON file.

    Honors the documented resolution precedence (explicit path, then the
    ``ISAW_PROJECT_DEFINITIONS`` env var, then the packaged/root locations).
    """
    rec = load_project_definitions(path)
    resolved = Path(rec["_resolved_path"])
    return DefinitionsContext(
        record=_freeze(rec),
        resolved_path=resolved,
        sha256=_sha256_file(resolved),
        definitions_version=str(rec["definitions_version"]),
        fixed_bins=_freeze(_fixed_from_json(rec["fixed_contour_bins"])),
        scaled_bins=_freeze(_scaled_from_json(rec["scaled_contour_bins"])),
        schema_versions=_freeze(rec.get("output_schema_versions", {})),
        provenance=PROV_JSON,
    )


# Resolved once at import (process startup).  CLI entry points that must honor a
# runtime override call ``resolve_definitions_context()`` again to get a fresh,
# internally consistent bundle rather than mixing frozen globals with re-read
# values.
CONTEXT = resolve_definitions_context()


def active_definitions_context() -> DefinitionsContext:
    """Fresh context honoring any runtime ``ISAW_PROJECT_DEFINITIONS`` override."""
    return resolve_definitions_context()


def project_definitions() -> dict:
    return copy.deepcopy(_DEFS)


def validate_bin_definitions(n_beads, fixed_defs=None, scaled_defs=None):
    return ico.validate_bin_definitions(n_beads, fixed_defs, scaled_defs)


# ---------------------------------------------------------------------------
# Code <-> JSON consistency
# ---------------------------------------------------------------------------

# Complete declarative structure the definitions record MUST provide.  Each
# entry is (key, subkeys-that-must-be-present).  Validated without any heavy
# dependency (Pydantic is not assumed available).
_REQUIRED_RECORDS = {
    "chain_length_convention": ("N", "n_beads", "n_steps"),
    "contact_definition": ("contact_indicator", "total_contact_number",
                           "minimum_contour_separation", "parity_constraint"),
    "pnipam_lcst_expectations": ("over_fitted_range",
                                 "primary_contact_favoring_coordinate",
                                 "expected_signs_high_minus_low_K"),
    "fixed_contour_bins": ("short_fixed", "medium_fixed", "long_threshold_fixed",
                           "labels"),
    "scaled_contour_bins": ("local_max_ratio", "meso_max_ratio",
                            "local_boundary", "labels"),
    "pair_motif_taxonomy": ("shared_endpoint", "disjoint", "nested",
                            "interleaved", "exhaustiveness"),
    "contact_graph_definition": ("vertices", "edges", "cycle_rank",
                                 "edge_conservation"),
    "augmented_graph_definition": ("vertices", "edges",
                                   "identities_connected_chain"),
    "thermodynamic_feature_columns": ("b_T", "K_T", "q_T",
                                      "reduced_potential_u", "effective_energy_H"),
    "output_schema_versions": ("feature_schema_version", "snapshot_schema_version",
                               "distributions_schema_version",
                               "diagnostic_trajectory_schema_version",
                               "calibration_report_schema_version",
                               "model_api_version"),
}

_REQUIRED_TOP_LEVEL_STRINGS = (
    "reduced_bias_definition", "K_definition", "q_definition",
    "reduced_potential_definition", "effective_energy_definition",
    "polymer_system", "transition_type",
)


def _check_record_shape(defs: dict) -> None:
    for key, subkeys in _REQUIRED_RECORDS.items():
        if key not in defs:
            raise SchemaError(f"definitions missing required record {key!r}")
        rec = defs[key]
        if not isinstance(rec, dict):
            raise SchemaError(f"definitions record {key!r} must be an object")
        for sk in subkeys:
            if sk not in rec:
                raise SchemaError(
                    f"definitions record {key!r} missing subkey {sk!r}")
    for key in _REQUIRED_TOP_LEVEL_STRINGS:
        if not isinstance(defs.get(key), str) or not defs[key]:
            raise SchemaError(f"definitions missing/empty string field {key!r}")
    if str(defs["polymer_system"]).upper() != "PNIPAM":
        raise SchemaError("polymer_system must be PNIPAM")
    if str(defs["transition_type"]).upper() != "LCST":
        raise SchemaError("transition_type must be LCST")


def check_definitions_consistency() -> None:
    """Raise SchemaError if the definitions record is incomplete OR code-level
    constants diverge from the JSON.

    Validates the complete nested record (chain-length convention, contact
    definition, minimum separation, parity, both bin schemes with boundary
    inclusion, pair taxonomy, graph identities, primary key, required metadata,
    output schema versions, PNIPAM/LCST, and the K/q/u/H definitions) and then
    cross-checks the code constants that mirror them.  The loaded record is
    never mutated.
    """
    _check_record_shape(_DEFS)

    # -- fixed bins ---------------------------------------------------------
    jf = _DEFS["fixed_contour_bins"]
    cf = ico.FIXED_BIN_DEFINITIONS
    if (int(jf["short_fixed"]["r_min"]) != int(cf["short_fixed"]["r_min"])
            or int(jf["short_fixed"]["r_max"]) != int(cf["short_fixed"]["r_max"])
            or int(jf["medium_fixed"]["r_min"]) != int(cf["medium_fixed"]["r_min"])
            or int(jf["long_threshold_fixed"]) != int(cf["long_threshold_fixed"])):
        raise SchemaError("fixed bin definitions diverge between JSON and code")

    # -- scaled bins (incl. boundary inclusion) -----------------------------
    js = _DEFS["scaled_contour_bins"]
    cs = ico.SCALED_BIN_DEFINITIONS
    if (float(js["local_max_ratio"]) != float(cs["local_max_ratio"])
            or float(js["meso_max_ratio"]) != float(cs["meso_max_ratio"])):
        raise SchemaError("scaled bin definitions diverge between JSON and code")
    if str(js["local_boundary"]) != str(cs.get("local_boundary")):
        raise SchemaError("scaled local_boundary diverges between JSON and code")

    # -- minimum separation / parity ----------------------------------------
    if int(_DEFS["contact_definition"]["minimum_contour_separation"]) != \
            int(ico.MIN_CONTOUR_SEPARATION):
        raise SchemaError("minimum_contour_separation diverges between JSON/code")
    if any(r % 2 == 0 for r in ico._valid_separations(64)):
        raise SchemaError("code valid separations include even r (parity broken)")

    # -- versions -----------------------------------------------------------
    if str(_DEFS["definitions_version"]) != str(ico.DEFINITIONS_VERSION):
        raise SchemaError(
            f"definitions_version diverges: JSON {_DEFS['definitions_version']} "
            f"!= code {ico.DEFINITIONS_VERSION}")
    if list(_DEFS["primary_key_definition"]) != list(PRIMARY_KEY):
        raise SchemaError("primary_key_definition diverges between JSON and code")

    # -- output schema versions cross-checked against the code modules ------
    # (lazy imports so this module has no import cycle with the pipeline.)
    ov = _DEFS["output_schema_versions"]
    import isaw_config_io as _cio
    import extract_contact_motif_features as _ext
    import remd_uniform_chain_2_new as _remd
    code_versions = {
        "feature_schema_version": int(_ext.FEATURE_SCHEMA_VERSION),
        "snapshot_schema_version": int(_cio.SNAPSHOT_SCHEMA_VERSION),
        "distributions_schema_version": int(_remd.SCHEMA_VERSION),
        "model_api_version": int(_remd.MODEL_API_VERSION),
    }
    for k, cv in code_versions.items():
        if int(ov[k]) != cv:
            raise SchemaError(
                f"{k} diverges: JSON {ov[k]} != code {cv}")

    # -- thermodynamic identities obeyed by the helpers ---------------------
    for K in (-1.3, 0.0, 0.75):
        if abs(reduced_potential_from_K(4, K) - (-4.0 * K)) > 1e-12:
            raise SchemaError("reduced_potential_from_K violates u = -m K")
    if abs(q_from_K(0.5) - math.exp(0.5)) > 1e-12:
        raise SchemaError("q_from_K violates q = exp(K)")


def output_schema_versions() -> dict:
    """Deep copy of the frozen output-schema version block."""
    return copy.deepcopy(_DEFS["output_schema_versions"])


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
