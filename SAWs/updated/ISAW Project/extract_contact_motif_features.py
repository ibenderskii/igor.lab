#!/usr/bin/env python3
"""
Offline topology-resolved feature extraction for ISAW REMD coordinate snapshots.

Reads a coordinate HDF5 file written by ``remd_uniform_chain_2_new.py
--save-configurations`` and produces one validated feature row per
(snapshot, temperature lane) configuration *without rerunning REMD*.  Expensive
O(m^2) pair-motif classification lives here, not in the Monte Carlo hot loop.

Streaming + correctness model
-----------------------------
* Memory scales with ``--chunk-size``, not total rows: each input chunk is read,
  validated, feature-extracted, and appended to extensible output datasets (or
  written as one Parquet record batch).  No full in-memory feature table.
* Only input rows with index ``< committed_rows`` are read.
* The input ``status`` must be ``complete`` unless ``--allow-interrupted``.
* Coordinates are passed to the strict lattice validator WITHOUT a pre-cast, so
  fractional / NaN / inf / out-of-range coordinates are rejected before any
  integer truncation could hide them.
* The bin definitions stored IN THE INPUT FILE are used (and validated); the
  extractor never silently substitutes module defaults when the input carries
  explicit definitions.
* The output file carries ``committed_feature_rows`` and a ``status``
  (running -> complete / interrupted) so an interrupted extraction is readable
  up to the last committed batch.

Output
------
HDF5 (default):
    /features/scalars/<column>     extensible 1-D datasets
    /features/sample_index/<col>   extensible 1-D datasets
    /features/m_r                  (n_rows, n_beads), uint16
    /metadata                      attrs (manifest, columns, status, committed)
Parquet (``--format parquet``, needs pyarrow): one record batch per chunk, with
``m_r_<r>`` columns for the odd separations.
A companion ``<output>.manifest.json`` records provenance, row counts, feature
names, and validation discrepancy counts (all zero for a clean extraction).

Usage
-----
    python extract_contact_motif_features.py --input run_configurations.h5 \
        --output run_features.h5 --validate

    python extract_contact_motif_features.py --quick-test
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import socket
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import isaw_contact_observables as ico  # noqa: E402
import isaw_config_io as cio  # noqa: E402
from isaw_schema import PRIMARY_KEY  # noqa: E402

try:
    import h5py
    _HAVE_H5PY = True
except Exception:  # pragma: no cover
    h5py = None
    _HAVE_H5PY = False

try:
    import pyarrow  # noqa: F401
    import pyarrow.parquet  # noqa: F401
    _HAVE_PYARROW = True
except Exception:  # pragma: no cover
    _HAVE_PYARROW = False

FEATURE_SCHEMA_VERSION = 4   # +contact_pairs, thermodynamic columns, augmented graph

# Fixed, documented m_r dtype: counts per separation are bounded by N (< 2^16),
# so uint16 is always safe and the dataset dtype never needs widening.
_M_R_DTYPE = np.uint16

# Numerical tolerance comparing stored vs recomputed Rg^2 / Ree^2.
_GEOM_ATOL = 1e-6
_GEOM_RTOL = 1e-9

INDEX_COLUMNS = [
    "run_id", "seed", "snapshot_index", "cycle",
    "temperature_index", "temperature", "walker_id",
]
SCALAR_COLUMNS = [
    "n_beads", "n_steps", "m",
    "Rg2_lattice", "Ree2_lattice",
    "asphericity", "gyration_lambda_1", "gyration_lambda_2", "gyration_lambda_3",
    "mean_contact_separation", "max_contact_separation",
    "m_short_fixed", "m_medium_fixed", "m_long_fixed",
    "m_local_scaled", "m_mesoscopic_scaled", "m_global_scaled",
    "pair_shared_endpoint", "pair_disjoint", "pair_nested", "pair_interleaved",
    "pair_total",
    "contact_vertices", "contact_graph_components",
    "contact_graph_edges", "sum_component_edges",
    "largest_component_vertices", "largest_component_edges",
    "largest_component_fraction_of_N",
    "largest_component_fraction_of_contact_vertices",
    "mean_degree_nonisolated", "degree_variance_nonisolated",
    "contact_graph_cycle_rank", "number_of_multiedge_components",
    # Augmented backbone+contact graph (C3).
    "augmented_graph_vertices", "augmented_graph_edges",
    "augmented_graph_components", "augmented_graph_cycle_rank",
    "augmented_mean_degree", "augmented_degree_variance",
    "augmented_largest_component_vertices",
    # Thermodynamic columns (C2); K is authoritative, q may be inf.
    "b_T", "K_T", "q_T", "reduced_potential_u", "effective_energy_H",
]
FEATURE_COLUMNS = INDEX_COLUMNS + SCALAR_COLUMNS

_STRING_COLS = {"run_id"}
_FLOAT_COLS = {
    "temperature", "Rg2_lattice", "Ree2_lattice", "asphericity",
    "gyration_lambda_1", "gyration_lambda_2", "gyration_lambda_3",
    "mean_contact_separation", "max_contact_separation",
    "largest_component_fraction_of_N",
    "largest_component_fraction_of_contact_vertices",
    "mean_degree_nonisolated", "degree_variance_nonisolated",
    "augmented_mean_degree", "augmented_degree_variance",
    "b_T", "K_T", "q_T", "reduced_potential_u", "effective_energy_H",
}


def _col_dtype(name):
    if name in _STRING_COLS:
        return "str"
    if name in _FLOAT_COLS:
        return "float"
    return "int"


DISCREPANCY_KEYS = (
    "contact_count", "rg2", "ree2", "odd_separation", "m_r_sum",
    "pair_total", "graph_edge_total", "walker_permutation",
)


class ExtractionError(RuntimeError):
    pass


_INT64_MAX = int(np.iinfo(np.int64).max)
_INT64_MIN = int(np.iinfo(np.int64).min)


def require_exact_integer_array(values, *, field_name, row=None):
    """Return an int64 array ONLY after every value is verified exactly integral.

    A stored integer dataset that was recreated as floating point (a corruption
    or a lossy round-trip) must never be silently truncated by ``astype(int)``.
    This helper rejects, raising :class:`ExtractionError` naming the field (and
    row when given):

    * object / ragged arrays;
    * complex values;
    * NaN and +/- infinity;
    * fractional values (values that are not exactly integral);
    * values outside the signed int64 range.

    The int64 array is returned only when all checks pass.
    """
    loc = f"[row {row}] " if row is not None else ""
    try:
        arr = np.asarray(values)
    except (TypeError, ValueError) as exc:
        raise ExtractionError(
            f"{loc}{field_name} could not be interpreted as an array") from exc
    if arr.dtype == object:
        raise ExtractionError(
            f"{loc}{field_name} must be a numeric array, not object/ragged")
    kind = arr.dtype.kind
    if kind == "b":
        return arr.astype(np.int64)
    if kind == "c":
        raise ExtractionError(f"{loc}{field_name} must be real, not complex")
    if kind in ("i", "u"):
        if kind == "u" and arr.size and int(arr.max()) > _INT64_MAX:
            raise ExtractionError(
                f"{loc}{field_name} value exceeds the signed int64 range")
        return arr.astype(np.int64)
    if kind == "f":
        if not np.all(np.isfinite(arr)):
            raise ExtractionError(
                f"{loc}{field_name} contains NaN or infinity (not an exact integer)")
        rounded = np.rint(arr)
        if not np.all(rounded == arr):
            raise ExtractionError(
                f"{loc}{field_name} has fractional values (not exact integers)")
        if arr.size:
            # Compare on true integers (float64 cannot represent int64 edges).
            if int(round(float(arr.max()))) > _INT64_MAX \
                    or int(round(float(arr.min()))) < _INT64_MIN:
                raise ExtractionError(
                    f"{loc}{field_name} is outside the signed int64 range")
        return rounded.astype(np.int64)
    raise ExtractionError(
        f"{loc}{field_name} has unsupported dtype {arr.dtype!r}")


def _project_definitions_path() -> str:
    """Resolved project-definitions path (for manifest provenance)."""
    try:
        import isaw_schema as _sch
        return str(_sch.RESOLVED_DEFINITIONS_PATH)
    except Exception:
        return "unknown"


def _safe_q(K: float) -> float:
    """q = exp(K); +inf on overflow (K is authoritative and stored separately)."""
    try:
        if K > 700:
            return float("inf")
        if K < -700:
            return 0.0
        return float(math.exp(K))
    except OverflowError:
        return float("inf")


def _model_bias_evaluator(meta: dict):
    """Build b(T) from the input file's model metadata, or None if insufficient.

    Reuses the canonical reduced_bias model registry so the thermodynamic
    columns match the sampler exactly.
    """
    name = meta.get("model_name")
    if isinstance(name, bytes):
        name = name.decode()
    if not name:
        return None, "missing model_name"
    params = meta.get("model_params")
    if params is None:
        return None, "missing model_params"
    params = [float(v) for v in np.asarray(params).ravel().tolist()]
    Tref = meta.get("Tref")
    Tscale = meta.get("Tscale")
    if Tref is None or Tscale is None:
        return None, "missing Tref/Tscale"
    import remd_uniform_chain_2_new as _remd
    if str(name) not in _remd.MODEL_REGISTRY:
        return None, f"unknown model {name!r}"

    def b_of_T(T):
        return float(_remd.reduced_bias(str(name), params, float(T),
                                        float(Tref), float(Tscale)))
    return b_of_T, "ok"


# ---------------------------------------------------------------------------
# Per-configuration feature computation + validation
# ---------------------------------------------------------------------------

def compute_features_for_config(
    coordinates,
    *,
    n_beads: int,
    fixed_defs: dict | None = None,
    scaled_defs: dict | None = None,
    stored_contacts: int | None = None,
    stored_rg2: float | None = None,
    stored_ree2: float | None = None,
    temperature: float | None = None,
    b_T: float | None = None,
    validate: bool = False,
    discrepancies: dict | None = None,
    locator: str = "",
) -> dict:
    """Compute the full topology-resolved feature dict for one conformation.

    ``coordinates`` is passed straight to :func:`ico.build_contact_map`, which
    validates strictly (no pre-cast): fractional/NaN/inf/out-of-range inputs are
    rejected here.  Bin definitions are the EXACT resolved definitions from the
    input file (``fixed_defs``/``scaled_defs``).
    """
    cp, seps = ico.build_contact_map(coordinates)
    m = int(cp.shape[0])

    def _fail(key: str, msg: str):
        if discrepancies is not None:
            discrepancies[key] = discrepancies.get(key, 0) + 1
        raise ExtractionError(f"{locator}{msg}")

    if stored_contacts is not None and m != int(stored_contacts):
        _fail("contact_count",
              f"recomputed contact count {m} != stored {int(stored_contacts)}")

    rg2 = ico.radius_of_gyration_squared(coordinates)
    ree2 = ico.end_to_end_distance_squared(coordinates)
    if stored_rg2 is not None and not math.isclose(
        rg2, float(stored_rg2), rel_tol=_GEOM_RTOL, abs_tol=_GEOM_ATOL
    ):
        _fail("rg2", f"recomputed Rg2 {rg2} != stored {float(stored_rg2)}")
    if stored_ree2 is not None and not math.isclose(
        ree2, float(stored_ree2), rel_tol=_GEOM_RTOL, abs_tol=_GEOM_ATOL
    ):
        _fail("ree2", f"recomputed Ree2 {ree2} != stored {float(stored_ree2)}")

    if m > 0 and not np.all((seps % 2) == 1):
        _fail("odd_separation", "non-odd contour separation present")

    m_r = ico.contact_separation_counts(cp, n_beads)
    ico.validate_m_r(m_r, n_beads=n_beads)
    if int(m_r.sum()) != m:
        _fail("m_r_sum", f"sum(m_r)={int(m_r.sum())} != m={m}")

    fixed = ico.bin_contact_separations_fixed(m_r, n_beads, fixed_defs)
    scaled = ico.bin_contact_separations_scaled(m_r, n_beads, scaled_defs)
    sep_summary = ico.contact_separation_summary(seps)
    motifs = ico.count_pair_motifs(cp)
    if motifs["pair_total"] != m * (m - 1) // 2:
        _fail("pair_total", f"pair total {motifs['pair_total']} != C({m},2)")
    graph = ico.contact_graph_summary(cp, n_beads)

    # Independent graph edge-total checks (not the cycle-rank identity used to
    # build the result): edges == m and sum_component_edges == m.
    if graph["contact_graph_edges"] != m or graph["sum_component_edges"] != m:
        _fail("graph_edge_total",
              f"graph edges {graph['contact_graph_edges']}/"
              f"{graph['sum_component_edges']} != m={m}")
    if graph["contact_graph_cycle_rank"] != (
        m - graph["contact_vertices"] + graph["contact_graph_components"]
    ):
        _fail("graph_edge_total", "graph cycle-rank identity violated")

    aug = ico.augmented_graph_summary(cp, n_beads)
    # Augmented-graph identity for a connected open chain: edges == N-1+m,
    # components == 1, cycle_rank == m.
    if (aug["augmented_graph_edges"] != (int(n_beads) - 1 + m)
            or aug["augmented_graph_components"] != 1
            or aug["augmented_graph_cycle_rank"] != m):
        _fail("graph_edge_total",
              f"augmented-graph identity violated: edges="
              f"{aug['augmented_graph_edges']} components="
              f"{aug['augmented_graph_components']} cyc="
              f"{aug['augmented_graph_cycle_rank']} (N={n_beads}, m={m})")

    if validate:
        ico.validate_contact_map(coordinates, cp, seps,
                                 expected_contact_count=m, strict=True)

    lam = ico.gyration_eigenvalues(coordinates)  # ascending
    asph = float(lam[2] - 0.5 * (lam[0] + lam[1]))

    # Thermodynamic columns (C2): K is authoritative; q may overflow to inf.
    if b_T is None:
        b_val = K_val = q_val = u_val = H_val = float("nan")
    else:
        b_val = float(b_T)
        K_val = -b_val
        q_val = _safe_q(K_val)
        u_val = float(m) * b_val            # u = m b = -m K
        H_val = (float(temperature) * u_val
                 if temperature is not None else float("nan"))

    feat = {
        "n_beads": int(n_beads),
        "n_steps": int(n_beads) - 1,
        "m": m,
        "Rg2_lattice": float(rg2),
        "Ree2_lattice": float(ree2),
        "asphericity": asph,
        "gyration_lambda_1": float(lam[0]),
        "gyration_lambda_2": float(lam[1]),
        "gyration_lambda_3": float(lam[2]),
        "mean_contact_separation": sep_summary["mean_contact_separation"],
        "max_contact_separation": sep_summary["max_contact_separation"],
        "m_short_fixed": fixed["m_short_fixed"],
        "m_medium_fixed": fixed["m_medium_fixed"],
        "m_long_fixed": fixed["m_long_fixed"],
        "m_local_scaled": scaled["m_local_scaled"],
        "m_mesoscopic_scaled": scaled["m_mesoscopic_scaled"],
        "m_global_scaled": scaled["m_global_scaled"],
        "pair_shared_endpoint": motifs["pair_shared_endpoint"],
        "pair_disjoint": motifs["pair_disjoint"],
        "pair_nested": motifs["pair_nested"],
        "pair_interleaved": motifs["pair_interleaved"],
        "pair_total": motifs["pair_total"],
        "b_T": b_val, "K_T": K_val, "q_T": q_val,
        "reduced_potential_u": u_val, "effective_energy_H": H_val,
    }
    feat.update(graph)
    feat.update(aug)
    feat["_m_r"] = m_r
    feat["_pairs"] = cp        # (m, 2) int64, lexicographically sorted
    return feat


# ---------------------------------------------------------------------------
# HDF5 reading helpers
# ---------------------------------------------------------------------------

def _read_metadata(f) -> dict:
    meta = {}
    g = f["metadata"]
    for k, v in g.attrs.items():
        meta[k] = v
    for k in g:
        meta[k] = g[k][()]
    return meta


def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit() -> str:
    try:
        import subprocess
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip() or "unknown"
    except Exception:
        pass
    return "unknown"


def _resolve_input_bin_defs(meta, n_beads):
    """Use the bin definitions stored in the input file (validated).

    Returns ``(fixed_defs, scaled_defs, source, stored_version)`` where
    ``source`` is 'input_file' when explicit definitions were present, else
    'module_default'; ``stored_version`` is the definitions_version recorded IN
    THE INPUT FILE (historical) when available, else the current code version.

    The stored (historical) definitions and their version are preserved exactly
    -- they are never relabeled with the installed code's ``DEFINITIONS_VERSION``
    -- so a feature file extracted from an older run is validated against the
    definitions it was actually produced with.
    """
    def _load(key):
        raw = meta.get(key)
        if raw is None:
            return None
        if isinstance(raw, bytes):
            raw = raw.decode()
        if isinstance(raw, str):
            try:
                return json.loads(raw)
            except (ValueError, TypeError):
                return None
        if isinstance(raw, dict):
            return dict(raw)
        return None

    combined = _load("structural_bin_definitions")
    fixed = _load("fixed_bin_definitions")
    scaled = _load("scaled_bin_definitions")
    if (fixed is None or scaled is None) and isinstance(combined, dict):
        fixed = fixed or combined.get("fixed")
        scaled = scaled or combined.get("scaled")

    # Historical definitions_version: prefer the combined record, then a
    # top-level metadata attribute, else the current code version.
    stored_version = None
    if isinstance(combined, dict):
        stored_version = combined.get("definitions_version")
    if stored_version is None:
        dv = meta.get("definitions_version")
        if isinstance(dv, bytes):
            dv = dv.decode()
        stored_version = dv
    if stored_version is None:
        stored_version = ico.DEFINITIONS_VERSION

    if fixed is not None and scaled is not None:
        ico.validate_bin_definitions(int(n_beads), fixed, scaled)
        return fixed, scaled, "input_file", str(stored_version)
    # No explicit definitions in the file -> module defaults (recorded).
    return (dict(ico.FIXED_BIN_DEFINITIONS), dict(ico.SCALED_BIN_DEFINITIONS),
            "module_default", ico.DEFINITIONS_VERSION)


# ---------------------------------------------------------------------------
# Streaming output writers
# ---------------------------------------------------------------------------

class _StreamHDF5Writer:
    """Append-per-chunk extensible HDF5 feature writer with commit/status."""

    def __init__(self, path, *, n_beads, run_id):
        self.path = path
        self.n_beads = int(n_beads)
        self.committed = 0
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.f = h5py.File(path, "w")
        self.f.attrs["feature_schema_version"] = FEATURE_SCHEMA_VERSION
        self.f.attrs["run_id"] = run_id
        self.f.attrs["n_beads"] = int(n_beads)
        self.f.attrs["status"] = "running"
        self.feats = self.f.create_group("features")
        self.scalars = self.feats.create_group("scalars")
        self.idxg = self.feats.create_group("sample_index")
        chunk = 256
        self._dsets = {}
        for name in INDEX_COLUMNS:
            self._dsets[name] = self._make_col(self.idxg, name, chunk)
        for name in SCALAR_COLUMNS:
            self._dsets[name] = self._make_col(self.scalars, name, chunk)
        self.mr = self.feats.create_dataset(
            "m_r", shape=(0, self.n_beads), maxshape=(None, self.n_beads),
            dtype=_M_R_DTYPE, chunks=(chunk, self.n_beads),
            compression="gzip", compression_opts=4)
        # Ragged contact-pair storage (C1): pairs + offsets (no Python objects).
        cpg = self.feats.create_group("contact_pairs")
        self.cp_pairs = cpg.create_dataset(
            "pairs", shape=(0, 2), maxshape=(None, 2), dtype="int64",
            chunks=(chunk, 2), compression="gzip", compression_opts=4)
        self.cp_offsets = cpg.create_dataset(
            "offsets", shape=(1,), maxshape=(None,), dtype="int64",
            chunks=(chunk + 1,))
        self.cp_offsets[0] = 0
        self._n_pairs = 0
        self.feats.attrs["committed_feature_rows"] = 0
        self.f.flush()

    def _make_col(self, group, name, chunk):
        kind = _col_dtype(name)
        if kind == "str":
            dt = h5py.string_dtype()
        elif kind == "float":
            dt = "float64"
        else:
            dt = "int64"
        return group.create_dataset(name, shape=(0,), maxshape=(None,),
                                    dtype=dt, chunks=(chunk,))

    def append_batch(self, rows, m_r_block, pairs_block):
        k = len(rows)
        if k == 0:
            return
        new = self.committed + k
        for name in FEATURE_COLUMNS:
            ds = self._dsets[name]
            kind = _col_dtype(name)
            if kind == "str":
                vals = np.asarray([str(r.get(name, "")) for r in rows],
                                  dtype=object)
            elif kind == "float":
                vals = np.asarray(
                    [np.nan if r.get(name) is None else float(r.get(name))
                     for r in rows], dtype=np.float64)
            else:
                vals = np.asarray([int(r.get(name, 0)) for r in rows],
                                  dtype=np.int64)
            ds.resize((new,))
            ds[self.committed:new] = vals
        self.mr.resize((new, self.n_beads))
        self.mr[self.committed:new] = np.asarray(m_r_block, dtype=_M_R_DTYPE)
        # Append contact pairs + monotone offsets (one offset per appended row).
        flat = (np.concatenate([np.asarray(p, dtype=np.int64).reshape(-1, 2)
                                for p in pairs_block], axis=0)
                if pairs_block else np.zeros((0, 2), dtype=np.int64))
        if flat.shape[0]:
            self.cp_pairs.resize((self._n_pairs + flat.shape[0], 2))
            self.cp_pairs[self._n_pairs:self._n_pairs + flat.shape[0]] = flat
        running = self._n_pairs
        new_offsets = np.empty(k, dtype=np.int64)
        for i, p in enumerate(pairs_block):
            running += int(np.asarray(p).reshape(-1, 2).shape[0])
            new_offsets[i] = running
        self.cp_offsets.resize((new + 1,))
        self.cp_offsets[self.committed + 1:new + 1] = new_offsets
        self._n_pairs = running
        # 1. flush row data, 2. update commit marker, 3. flush marker.
        self.f.flush()
        self.committed = new
        self.feats.attrs["committed_feature_rows"] = int(self.committed)
        self.f.flush()

    def finalize(self, *, status, manifest):
        self.f.attrs["status"] = status
        self.feats.attrs["committed_feature_rows"] = int(self.committed)
        self.feats["contact_pairs"].attrs["total_contacts"] = int(self._n_pairs)
        meta = self.f.create_group("metadata")
        meta.attrs["manifest"] = json.dumps(manifest, default=str)
        meta.attrs["columns"] = json.dumps(FEATURE_COLUMNS)
        meta.attrs["primary_key"] = json.dumps(list(PRIMARY_KEY))
        self.f.flush()
        self.f.close()


class _StreamParquetWriter:
    """One Parquet record batch per chunk (no full in-memory table).

    EXPERIMENTAL format.  Parquet cannot offer the append-safe committed-row
    semantics of HDF5, so the file is written to a temporary path and atomically
    renamed onto the final path ONLY on success; on failure the temp file is
    removed (no partial file at the final path).  Schema key/value metadata
    embeds provenance; the companion ``.manifest.json`` is authoritative.
    Contact-pair ragged arrays are NOT written to Parquet (HDF5 is authoritative
    for those); m_r is exported as m_r_<r> columns.
    """

    def __init__(self, path, *, n_beads, run_id, definitions, provenance=None):
        import pyarrow as pa
        self.pa = pa
        self.path = path
        self.tmp_path = str(path) + ".tmp"
        self.run_id = run_id
        self.committed = 0
        self.mr_cols = [r for r in range(int(n_beads))
                        if r % 2 == 1 and r >= ico.MIN_CONTOUR_SEPARATION]
        fields = []
        for name in FEATURE_COLUMNS:
            kind = _col_dtype(name)
            t = (pa.string() if kind == "str"
                 else pa.float64() if kind == "float" else pa.int64())
            fields.append(pa.field(name, t))
        for r in self.mr_cols:
            fields.append(pa.field(f"m_r_{r}", pa.int64()))
        # Phase 12: preserve the HISTORICAL definitions version this file was
        # produced with (never relabel with the installed code version), plus the
        # definitions source/path and input schema version.  Parquet is an
        # explicitly EXPERIMENTAL scalar/tabular export -- it omits the ragged
        # contact-pair arrays, so HDF5 remains authoritative.
        prov = provenance or {}
        kv = {
            b"feature_schema_version": str(FEATURE_SCHEMA_VERSION).encode(),
            b"run_id": str(run_id).encode(),
            b"n_beads": str(int(n_beads)).encode(),
            b"definitions_version": str(
                prov.get("definitions_version", ico.DEFINITIONS_VERSION)).encode(),
            b"code_definitions_version": str(ico.DEFINITIONS_VERSION).encode(),
            b"uses_current_definitions": str(
                bool(prov.get("uses_current_definitions", True))).encode(),
            b"bin_definitions_source": str(
                prov.get("bin_definitions_source", "module_default")).encode(),
            b"project_definitions_path": str(
                prov.get("project_definitions_path", "unknown")).encode(),
            b"input_schema_version": str(
                prov.get("input_schema_version", "unknown")).encode(),
            b"fixed_bin_definitions": json.dumps(definitions[0]).encode(),
            b"scaled_bin_definitions": json.dumps(definitions[1]).encode(),
            b"primary_key": json.dumps(list(PRIMARY_KEY)).encode(),
            b"format": b"parquet",
            b"experimental": b"true",
            b"authoritative_representation": (
                b"HDF5 (Parquet is a scalar/tabular export without ragged "
                b"contact pairs)"),
            b"status_semantics": (
                b"file present at final path => complete; committed_feature_rows"
                b" == parquet num_rows; manifest is authoritative"),
        }
        self.schema = pa.schema(fields, metadata=kv)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        import pyarrow.parquet as pq
        self.writer = pq.ParquetWriter(self.tmp_path, self.schema)

    def append_batch(self, rows, m_r_block, pairs_block):
        if not rows:
            return
        pa = self.pa
        arrays = []
        for name in FEATURE_COLUMNS:
            kind = _col_dtype(name)
            if kind == "str":
                arrays.append(pa.array([str(r.get(name, "")) for r in rows],
                                       type=pa.string()))
            elif kind == "float":
                arrays.append(pa.array(
                    [None if r.get(name) is None else float(r.get(name))
                     for r in rows], type=pa.float64()))
            else:
                arrays.append(pa.array([int(r.get(name, 0)) for r in rows],
                                       type=pa.int64()))
        m_r_block = np.asarray(m_r_block)
        for r in self.mr_cols:
            arrays.append(pa.array([int(m_r_block[i, r]) for i in range(len(rows))],
                                   type=pa.int64()))
        batch = pa.record_batch(arrays, schema=self.schema)
        self.writer.write_batch(batch)
        self.committed += len(rows)

    def finalize(self, *, status, manifest):
        self.writer.close()
        if status == "complete":
            os.replace(self.tmp_path, self.path)   # atomic on same filesystem
        else:
            try:
                os.remove(self.tmp_path)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Extraction driver
# ---------------------------------------------------------------------------

def extract(
    input_path: str,
    output_path: str,
    *,
    chunk_size: int = 256,
    overwrite: bool = False,
    validate: bool = False,
    output_format: str | None = None,
    allow_interrupted: bool = False,
) -> dict:
    """Extract features for every committed (snapshot, lane) config (streaming)."""
    if not _HAVE_H5PY:
        raise ExtractionError("reading snapshots requires h5py (not installed)")
    if not Path(input_path).exists():
        raise ExtractionError(f"input file not found: {input_path}")
    out = Path(output_path)
    if out.exists() and not overwrite:
        raise ExtractionError(
            f"output {output_path!r} exists; pass --overwrite to replace it"
        )

    fmt = (output_format or "hdf5").lower()
    if fmt not in ("hdf5", "parquet"):
        raise ExtractionError(f"unknown output format {fmt!r}")
    if fmt == "parquet" and not _HAVE_PYARROW:
        raise ExtractionError(
            "Parquet output requested but pyarrow is not available; install "
            "pyarrow or use --format hdf5 (the default)."
        )

    input_sha = _file_sha256(input_path)
    discrepancies = {k: 0 for k in DISCREPANCY_KEYS}

    f = h5py.File(input_path, "r")
    writer = None
    completed = False
    try:
        meta = _read_metadata(f)
        if "snapshots" not in f:
            raise ExtractionError("input has no /snapshots group")
        snap = f["snapshots"]
        status = str(snap.attrs.get("status", f.attrs.get("status", "unknown")))
        if status != cio.STATUS_COMPLETE and not allow_interrupted:
            raise ExtractionError(
                f"input status is {status!r}, not 'complete'. Pass "
                f"--allow-interrupted to extract the committed rows anyway."
            )

        coords_ds = snap["coordinates"]
        n_alloc, nT, n_beads, three = coords_ds.shape
        if three != 3:
            raise ExtractionError("coordinates last dim != 3")
        n_committed = cio.committed_rows(snap)
        if n_committed > n_alloc:
            raise ExtractionError(f"committed_rows {n_committed} > allocated {n_alloc}")

        cycle_ds = snap["cycle"]
        walker_ds = snap["walker_id"]
        contacts_ds = snap["contacts"]
        rg2_ds = snap["rg2_lattice"]
        ree2_ds = snap["ree2_lattice"]

        run_id = str(meta.get("run_id", Path(input_path).stem))
        seed = int(meta.get("seed", -1))
        temps = np.asarray(meta.get("temperatures", np.full(nT, np.nan)), dtype=float)
        schema_version = int(meta.get("schema_version", 0))
        # Phase 2: copy the authoritative model record from the source snapshot
        # metadata so the feature file carries an independent provenance copy.
        _model_name = meta.get("model_name")
        if isinstance(_model_name, bytes):
            _model_name = _model_name.decode()
        _param_names = meta.get("param_names")
        if _param_names is not None:
            _param_names = [str(v) for v in np.asarray(_param_names).ravel().tolist()]
        _model_params = meta.get("model_params")
        if _model_params is not None:
            _model_params = [float(v) for v in np.asarray(_model_params).ravel().tolist()]
        _Tref = meta.get("Tref")
        _Tscale = meta.get("Tscale")
        model_record = {
            "model_name": (None if _model_name is None else str(_model_name)),
            "param_names": _param_names,
            "model_params": _model_params,
            "Tref": (None if _Tref is None else float(_Tref)),
            "Tscale": (None if _Tscale is None else float(_Tscale)),
        }
        (fixed_defs, scaled_defs, defs_source,
         stored_defs_version) = _resolve_input_bin_defs(meta, n_beads)
        uses_current_definitions = (
            str(stored_defs_version) == str(ico.DEFINITIONS_VERSION))
        b_of_T, thermo_status = _model_bias_evaluator(meta)
        if validate and thermo_status != "ok":
            raise ExtractionError(
                f"input lacks sufficient model metadata for thermodynamic "
                f"columns ({thermo_status}); cannot run a validated extraction. "
                f"Re-run the sampler so model_name/model_params/Tref/Tscale are "
                f"saved, or extract without --validate.")

        if fmt == "hdf5":
            writer = _StreamHDF5Writer(output_path, n_beads=int(n_beads), run_id=run_id)
        else:
            writer = _StreamParquetWriter(
                output_path, n_beads=int(n_beads), run_id=run_id,
                definitions=(fixed_defs, scaled_defs),
                provenance={
                    "definitions_version": str(stored_defs_version),
                    "uses_current_definitions": bool(uses_current_definitions),
                    "bin_definitions_source": defs_source,
                    "project_definitions_path": _project_definitions_path(),
                    "input_schema_version": schema_version,
                })

        mr_max = 0
        sample_index = 0
        for s0 in range(0, n_committed, chunk_size):
            s1 = min(s0 + chunk_size, n_committed)
            coords_block = coords_ds[s0:s1]
            cyc_block = cycle_ds[s0:s1]
            walk_block = walker_ds[s0:s1]
            cont_block = contacts_ds[s0:s1]
            rg2_block = rg2_ds[s0:s1]
            ree2_block = ree2_ds[s0:s1]

            rows_chunk = []
            m_r_chunk = []
            pairs_chunk = []
            for si in range(s1 - s0):
                snap_idx = s0 + si
                cyc = int(cyc_block[si])
                w = np.asarray(walk_block[si], dtype=np.int64)
                if not np.array_equal(np.sort(w), np.arange(nT, dtype=np.int64)):
                    discrepancies["walker_permutation"] += 1
                    raise ExtractionError(
                        f"[snapshot {snap_idx} cycle {cyc}] walker_id not a "
                        f"permutation: {w.tolist()}"
                    )
                for k in range(nT):
                    locator = f"[snapshot {snap_idx} cycle {cyc} lane {k}] "
                    T_k = float(temps[k]) if k < temps.size else math.nan
                    b_k = (b_of_T(T_k) if (b_of_T is not None
                                           and math.isfinite(T_k)) else None)
                    # Pass the raw HDF5 slice straight to the strict validator --
                    # NO pre-cast to int64.
                    feat = compute_features_for_config(
                        coords_block[si, k],
                        n_beads=int(n_beads),
                        fixed_defs=fixed_defs, scaled_defs=scaled_defs,
                        stored_contacts=int(cont_block[si, k]),
                        stored_rg2=float(rg2_block[si, k]),
                        stored_ree2=float(ree2_block[si, k]),
                        temperature=T_k, b_T=b_k,
                        validate=validate,
                        discrepancies=discrepancies,
                        locator=locator,
                    )
                    m_r = feat.pop("_m_r")
                    pairs = feat.pop("_pairs")
                    mr_max = max(mr_max, int(m_r.max(initial=0)))
                    row = {
                        "run_id": run_id, "seed": seed,
                        "snapshot_index": snap_idx, "cycle": cyc,
                        "temperature_index": int(k),
                        "temperature": T_k,
                        "walker_id": int(w[k]),
                    }
                    row.update(feat)
                    rows_chunk.append(row)
                    m_r_chunk.append(np.asarray(m_r, dtype=_M_R_DTYPE))
                    pairs_chunk.append(np.asarray(pairs, dtype=np.int64).reshape(-1, 2))
                    sample_index += 1
            writer.append_batch(
                rows_chunk,
                (np.asarray(m_r_chunk, dtype=_M_R_DTYPE) if m_r_chunk
                 else np.zeros((0, int(n_beads)), _M_R_DTYPE)),
                pairs_chunk)

        n_rows = writer.committed
        manifest = {
            "input_path": str(input_path),
            "input_sha256": input_sha,
            "input_schema_version": schema_version,
            "input_status": status,
            "output_path": str(output_path),
            "output_format": fmt,
            "output_status": "complete",
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "row_count": int(n_rows),
            "committed_feature_rows": int(n_rows),
            "n_beads": int(n_beads),
            "temperature_count": int(nT),
            # Phase 2: authoritative temperature ladder + model record + source
            # provenance, copied from the source snapshot metadata (unambiguous
            # JSON, not lossy string formatting).
            "temperatures": [float(t) for t in temps.tolist()],
            "model_record": model_record,
            "source_configuration_path": str(input_path),
            "source_configuration_sha256": input_sha,
            "source_snapshot_schema_version": int(schema_version),
            "committed_rows": int(n_committed),
            "allocated_rows": int(n_alloc),
            "extracted_snapshot_rows": int(n_committed),
            "feature_names": list(FEATURE_COLUMNS),
            "m_r_dtype": str(np.dtype(_M_R_DTYPE)),
            "m_r_max_observed": int(mr_max),
            "m_r_representation": (
                "dense contour-separation histogram length n_beads; "
                "m_r[r] = #contacts with separation r; even r are zero; sum_r == m"
            ),
            # definitions_version is the HISTORICAL version this file was
            # produced with (authoritative for validating THIS file); the
            # installed code version is recorded separately.
            "definitions_version": str(stored_defs_version),
            "code_definitions_version": ico.DEFINITIONS_VERSION,
            "uses_current_definitions": bool(uses_current_definitions),
            "bin_definitions_source": defs_source,
            "fixed_bin_definitions": fixed_defs,
            "scaled_bin_definitions": scaled_defs,
            "project_definitions_path": _project_definitions_path(),
            "primary_key": list(PRIMARY_KEY),
            "contact_pairs_representation": (
                "HDF5 /features/contact_pairs/pairs (total_contacts,2) + "
                "offsets (n_rows+1,); row i pairs = pairs[offsets[i]:offsets[i+1]]"
            ),
            "thermodynamic_columns": ["b_T", "K_T", "q_T",
                                      "reduced_potential_u", "effective_energy_H"],
            "thermodynamic_status": thermo_status,
            "thermodynamic_note": (
                "K_T is authoritative; q_T may be inf on overflow; "
                "effective_energy_H is model-implied, not necessarily physical "
                "for polynomial effective models"),
            "augmented_graph_columns": [
                "augmented_graph_vertices", "augmented_graph_edges",
                "augmented_graph_components", "augmented_graph_cycle_rank",
                "augmented_mean_degree", "augmented_degree_variance",
                "augmented_largest_component_vertices"],
            "parquet_experimental": bool(fmt == "parquet"),
            "validation_enabled": bool(validate),
            "validation_discrepancy_counts": discrepancies,
            "creation_time": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
            "hostname": socket.gethostname(),
            "git_commit": _git_commit(),
            "software_versions": {
                "python": sys.version.split()[0],
                "numpy": np.__version__,
                "h5py": getattr(h5py, "__version__", "unknown"),
                "pyarrow": (pyarrow.__version__ if _HAVE_PYARROW else "unavailable"),
            },
        }
        writer.finalize(status="complete", manifest=manifest)
        completed = True
    except Exception:
        if writer is not None and not completed:
            try:
                writer.finalize(status="interrupted",
                                manifest={"output_status": "interrupted"})
            except Exception:
                pass
        raise
    finally:
        f.close()

    manifest_path = str(output_path) + ".manifest.json"
    Path(manifest_path).parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    manifest["manifest_path"] = manifest_path
    print(f"Saved {output_path} ({manifest['row_count']} rows) and {manifest_path}")
    return manifest


# ---------------------------------------------------------------------------
# Feature dictionary + comprehensive schema validation (C4)
# ---------------------------------------------------------------------------

# Complete, explicit dictionary metadata for EVERY output field.  Tuple layout:
#   (units, mathematical_definition, source_representation, nullable,
#    schema_version_introduced, validation_identity)
# No entry is auto-generated from the field name; every column is documented.
_FEATURE_META = {
    # --- index / provenance ---
    "run_id": ("none", "run identifier string", "run metadata", False, 1,
               "constant within a run"),
    "seed": ("none", "master RNG seed", "run metadata", False, 1,
             "constant within a run"),
    "snapshot_index": ("none", "committed snapshot row index in the input file",
                       "sample index", False, 1,
                       "0 <= snapshot_index < committed_rows; rows sorted ascending"),
    "cycle": ("cycles", "originating REMD cycle number", "sample index", False, 1,
              "strictly increasing across distinct snapshots"),
    "temperature_index": ("none", "temperature-lane index k", "sample index",
                          False, 1, "0 <= k < n_temperatures"),
    "temperature": ("K", "lane temperature T_k", "run metadata (ladder)", False, 1,
                    "constant for a given temperature_index"),
    "walker_id": ("none", "walker occupying the lane at this snapshot",
                  "sample index", False, 1, "permutation of 0..nT-1 per snapshot"),
    # --- chain-length convention ---
    "n_beads": ("beads", "chain length N", "config", False, 1, "== file n_beads"),
    "n_steps": ("bonds", "N - 1", "config", False, 1, "== n_beads - 1"),
    # --- contact map ---
    "m": ("contacts", "m = #{(i,j): j-i>1, |r_i-r_j|=1}", "contact_map", False, 1,
          "== len(row contact pairs) == sum_r m_r"),
    # --- geometry ---
    "Rg2_lattice": ("lattice^2", "R_g^2 = (1/N) sum_i |r_i - r_cm|^2", "geometry",
                    False, 1, "== stored rg2_lattice within tol"),
    "Ree2_lattice": ("lattice^2", "R_ee^2 = |r_{N-1} - r_0|^2", "geometry", False, 1,
                     "== stored ree2_lattice within tol"),
    "asphericity": ("lattice^2", "lam3 - 0.5 (lam1 + lam2)", "gyration tensor",
                    False, 1, "lam ascending eigenvalues; >= 0"),
    "gyration_lambda_1": ("lattice^2", "smallest gyration-tensor eigenvalue",
                          "gyration tensor", False, 1, "lam1 <= lam2 <= lam3"),
    "gyration_lambda_2": ("lattice^2", "middle gyration-tensor eigenvalue",
                          "gyration tensor", False, 1, "lam1 <= lam2 <= lam3"),
    "gyration_lambda_3": ("lattice^2", "largest gyration-tensor eigenvalue",
                          "gyration tensor", False, 1,
                          "lam1+lam2+lam3 == Rg2 within tol"),
    "mean_contact_separation": ("beads", "mean r = j-i over contacts",
                                "contact_map", True, 1, "None when m == 0"),
    "max_contact_separation": ("beads", "max r = j-i over contacts", "contact_map",
                               True, 1, "None when m == 0"),
    # --- fixed contour bins ---
    "m_short_fixed": ("contacts", "#contacts with short_min<=r<=short_max",
                      "m_r + fixed bins", False, 1,
                      "m_short+m_medium+m_long == m"),
    "m_medium_fixed": ("contacts", "#contacts with medium_min<=r<long_threshold",
                       "m_r + fixed bins", False, 1, "part of fixed partition"),
    "m_long_fixed": ("contacts", "#contacts with r>=long_threshold",
                     "m_r + fixed bins", False, 1, "part of fixed partition"),
    # --- scaled contour bins ---
    "m_local_scaled": ("contacts", "#contacts with r/N<=local_max_ratio",
                       "m_r + scaled bins", False, 1,
                       "m_local+m_meso+m_global == m"),
    "m_mesoscopic_scaled": ("contacts",
                            "#contacts with local_max_ratio<r/N<meso_max_ratio",
                            "m_r + scaled bins", False, 1, "part of scaled partition"),
    "m_global_scaled": ("contacts", "#contacts with r/N>=meso_max_ratio",
                        "m_r + scaled bins", False, 1, "part of scaled partition"),
    # --- pair motifs ---
    "pair_shared_endpoint": ("pairs", "#contact-pairs sharing >=1 bead index",
                             "contact_pairs", False, 1, "sum of 4 classes == C(m,2)"),
    "pair_disjoint": ("pairs", "#contact-pairs i<j<k<l (or swapped)",
                      "contact_pairs", False, 1, "part of C(m,2) partition"),
    "pair_nested": ("pairs", "#contact-pairs i<k<l<j (or swapped)",
                    "contact_pairs", False, 1, "part of C(m,2) partition"),
    "pair_interleaved": ("pairs", "#contact-pairs i<k<j<l (or swapped)",
                         "contact_pairs", False, 1, "part of C(m,2) partition"),
    "pair_total": ("pairs", "C(m,2) total contact-pair count", "contact_pairs",
                   False, 1, "== m(m-1)/2 == sum of 4 motif classes"),
    # --- contact graph (backbone edges excluded) ---
    "contact_vertices": ("vertices", "#bead indices in >=1 nonbonded contact",
                         "contact graph", False, 1, "V of contact graph"),
    "contact_graph_components": ("components", "#connected components (contacts)",
                                 "contact graph", False, 1, "C of contact graph"),
    "contact_graph_edges": ("edges", "#contact edges", "contact graph", False, 1,
                            "== m"),
    "sum_component_edges": ("edges", "sum of per-component edge counts",
                            "contact graph", False, 1, "== m"),
    "largest_component_vertices": ("vertices", "S_max: vertices in largest component",
                                   "contact graph", False, 1,
                                   "tie-break by smallest min vertex index"),
    "largest_component_edges": ("edges", "edges in the largest component",
                                "contact graph", False, 1, "<= m"),
    "largest_component_fraction_of_N": ("none", "S_max / N", "contact graph", False,
                                        1, "in [0,1]"),
    "largest_component_fraction_of_contact_vertices": (
        "none", "S_max / contact_vertices", "contact graph", False, 1,
        "in [0,1]; 0 when no contacts"),
    "mean_degree_nonisolated": ("none", "mean degree over contact vertices",
                                "contact graph", False, 1, ">= 0"),
    "degree_variance_nonisolated": ("none", "degree variance over contact vertices",
                                    "contact graph", False, 1, ">= 0"),
    "contact_graph_cycle_rank": ("none", "E - V + C", "contact graph", False, 1,
                                 "== m - contact_vertices + components"),
    "number_of_multiedge_components": ("components",
                                       "#components with >=2 edges", "contact graph",
                                       False, 1, ">= 0"),
    # --- augmented backbone+contact graph ---
    "augmented_graph_vertices": ("vertices", "all N beads", "augmented graph", False,
                                 4, "== N"),
    "augmented_graph_edges": ("edges", "(N-1) backbone + m contact edges",
                              "augmented graph", False, 4, "== N-1+m"),
    "augmented_graph_components": ("components", "connected components",
                                   "augmented graph", False, 4,
                                   "== 1 for a connected chain"),
    "augmented_graph_cycle_rank": ("none", "E - V + C", "augmented graph", False, 4,
                                   "== m for a connected chain"),
    "augmented_mean_degree": ("none", "mean degree over all N vertices",
                              "augmented graph", False, 4, ">= 0"),
    "augmented_degree_variance": ("none", "degree variance over all N vertices",
                                  "augmented graph", False, 4, ">= 0"),
    "augmented_largest_component_vertices": ("vertices",
                                             "largest augmented component size",
                                             "augmented graph", False, 4, "== N"),
    # --- thermodynamic (K authoritative; q may overflow) ---
    "b_T": ("kT", "reduced contact bias b(T) from the fitted model",
            "model(T)", True, 4, "== -K_T"),
    "K_T": ("kT", "K(T) = -b(T) (authoritative)", "model(T)", True, 4,
            "higher K favors more contacts"),
    "q_T": ("none", "q(T) = exp(K(T))", "model(T)", True, 4,
            "== exp(K) when finite; inf on overflow"),
    "reduced_potential_u": ("kT", "u = m b(T) = -m K(T)", "m + model(T)", True, 4,
                            "== -m K_T"),
    "effective_energy_H": ("energy", "H = T u (model-implied)", "T + u", True, 4,
                           "== T * reduced_potential_u"),
}


def build_feature_dictionary() -> dict:
    """Machine-readable feature dictionary for every output field.

    Each entry carries name, dtype, units, mathematical_definition,
    source_representation, cadence, nullable, schema_version_introduced, and a
    validation_identity.  Every ``FEATURE_COLUMNS`` entry plus ``m_r`` and
    ``contact_pairs`` is covered exactly once (a test enforces the bijection).
    """
    fields = []
    for name in FEATURE_COLUMNS:
        kind = _col_dtype(name)
        dtype = {"str": "string", "float": "float64", "int": "int64"}[kind]
        if name not in _FEATURE_META:
            raise ExtractionError(f"feature dictionary missing entry for {name!r}")
        units, mdef, srep, nullable, ver, ident = _FEATURE_META[name]
        fields.append({
            "name": name, "dtype": dtype, "units": units,
            "mathematical_definition": mdef,
            "source_representation": srep,
            "cadence": "per (snapshot, lane) configuration",
            "nullable": bool(nullable),
            "schema_version_introduced": int(ver),
            "validation_identity": ident,
        })
    fields.append({
        "name": "m_r", "dtype": str(np.dtype(_M_R_DTYPE)), "units": "contacts",
        "mathematical_definition": "m_r[r] = #contacts with contour separation r",
        "source_representation": "dense histogram length n_beads",
        "cadence": "per (snapshot, lane) configuration",
        "nullable": False, "schema_version_introduced": 1,
        "validation_identity": "sum_r m_r == m; even-r entries zero; width n_beads"})
    # Ragged contact pairs are stored as TWO physical datasets; document each.
    fields.append({
        "name": "contact_pairs/pairs", "dtype": "int64", "units": "bead-index pairs",
        "mathematical_definition": "concatenated sorted (i,j) contact pairs "
                                   "(j-i>1, r odd) across all rows",
        "source_representation": "dataset pairs (total_contacts, 2)",
        "cadence": "per (snapshot, lane) configuration (ragged)",
        "nullable": False, "schema_version_introduced": 4,
        "validation_identity": "row i = pairs[offsets[i]:offsets[i+1]]; "
                               "lexicographically sorted; no duplicates"})
    fields.append({
        "name": "contact_pairs/offsets", "dtype": "int64", "units": "index",
        "mathematical_definition": "row-start offsets into pairs (CSR-style)",
        "source_representation": "dataset offsets (n_rows+1,)",
        "cadence": "per feature file",
        "nullable": False, "schema_version_introduced": 4,
        "validation_identity": "offsets[0]==0; nondecreasing; offsets[-1]==len(pairs); "
                               "diff(offsets)==m"})
    # Provenance metadata fields (recorded in the manifest, not per-row columns).
    prov = {
        "definitions_version": (
            "historical bin-definitions version this file was produced with",
            "string", "manifest.definitions_version",
            "preserved exactly; never relabeled with the code version"),
        "code_definitions_version": (
            "installed code bin-definitions version at extraction time",
            "string", "manifest.code_definitions_version",
            "== isaw_contact_observables.DEFINITIONS_VERSION"),
        "project_definitions_path": (
            "resolved project_definitions.json path used at extraction",
            "string", "manifest.project_definitions_path",
            "existing file path (or 'unknown')"),
        "primary_key": (
            "primary-key column tuple",
            "json", "metadata.primary_key",
            "unique across all rows"),
        # Phase 2/15: authoritative temperature ladder + model record + source
        # provenance copied from the source snapshot metadata.
        "temperatures": (
            "authoritative temperature ladder (per lane index)",
            "json", "manifest.temperatures",
            "len == temperature_count; each row temperature matches its index"),
        "temperature_count": (
            "number of temperature lanes",
            "int", "manifest.temperature_count",
            "== len(temperatures); each snapshot has exactly this many rows"),
        "model_record": (
            "fitted-model metadata (model_name/param_names/model_params/Tref/Tscale)",
            "json", "manifest.model_record",
            "matches the source configuration model record when available"),
        "source_configuration_path": (
            "path of the source coordinate snapshot file",
            "string", "manifest.source_configuration_path",
            "hashed and re-verified by the validator when present"),
        "source_configuration_sha256": (
            "SHA-256 of the source coordinate snapshot file",
            "string", "manifest.source_configuration_sha256",
            "matches the actual source file hash when available"),
        "source_snapshot_schema_version": (
            "snapshot schema version of the source file",
            "int", "manifest.source_snapshot_schema_version",
            ">= 1"),
        # Phase 4/15: calibration-stage fingerprint + validation certificate.
        "stage_fingerprint": (
            "SHA-256 of the canonical calibration-stage fingerprint fields",
            "string", "validation certificate.stage_fingerprint",
            "resume reuses a stage only on an exact fingerprint match"),
        "validation_certificate": (
            "companion <feature>.validation.json (feature_path, feature_sha256, "
            "source_configuration_path/sha256, stage_fingerprint, "
            "validator_revision, validation_timestamp, definitions_version, "
            "validation_result)",
            "json", "sidecar validation certificate",
            "feature_sha256 matches the file; validation_result == passed"),
        "feature_schema_version_attr": (
            "feature schema version file attribute",
            "int", "file attribute feature_schema_version",
            "1..current; equals the manifest feature_schema_version"),
    }
    for name, (mdef, dt, srep, ident) in prov.items():
        fields.append({
            "name": name, "dtype": dt, "units": "none",
            "mathematical_definition": mdef,
            "source_representation": srep,
            "cadence": "per feature file (provenance metadata)",
            "nullable": False, "schema_version_introduced": 1,
            "validation_identity": ident})
    return {
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "definitions_version": ico.DEFINITIONS_VERSION,
        "primary_key": list(PRIMARY_KEY),
        "provenance_metadata_fields": list(prov),
        "fields": fields,
    }


def write_feature_dictionary(path=None) -> str:
    p = Path(path) if path is not None else (
        Path(__file__).resolve().parent / "docs" / "feature_dictionary.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(build_feature_dictionary(), fh, indent=2)
    return str(p)


def validate_feature_file_hdf5(path: str, *, deep: bool = True) -> dict:
    """Comprehensive structural/identity validation of a feature HDF5 file.

    Independently RECONSTRUCTS every derived representation from the stored
    contact pairs and requires exact agreement with the stored scalar columns --
    the scalar columns are never trusted merely because the same extractor wrote
    them.  On any failure it raises :class:`ExtractionError` identifying the row
    index and the violated invariant.

    The file is validated against ITS OWN stored (possibly historical)
    definitions record; it is never required to equal the currently installed
    code version.  ``deep=False`` skips the per-row reconstruction (aggregate
    checks only) for very large files.
    """
    import isaw_schema as sch
    with h5py.File(path, "r") as f:
        if str(f.attrs.get("status")) != "complete":
            raise ExtractionError("feature file status != complete")
        feats = f["features"]
        scalars = feats["scalars"]
        idxg = feats["sample_index"]
        S = {name: scalars[name][()] for name in scalars}
        I = {name: idxg[name][()] for name in idxg}
        # -- Phase 1: EXACT integer validation before any int cast ----------
        # Every stored integer dataset must be exactly integral and in int64
        # range; a dataset silently recreated as float with a fractional / NaN /
        # inf / out-of-range value is rejected here, never truncated.
        for name in scalars:
            if _col_dtype(name) == "int":
                S[name] = require_exact_integer_array(S[name], field_name=name)
        for name in idxg:
            if _col_dtype(name) == "int":
                I[name] = require_exact_integer_array(I[name], field_name=name)
        m = S["m"].astype(np.int64)
        n_rows = int(m.shape[0])
        committed = int(feats.attrs.get("committed_feature_rows", -1))
        if committed != n_rows:
            raise ExtractionError(
                f"committed_feature_rows {committed} != row count {n_rows}")
        for label, grp in (("scalar", S), ("index", I)):
            for name, arr in grp.items():
                if arr.shape[0] != n_rows:
                    raise ExtractionError(f"{label} dataset {name} length mismatch")
        n_beads = int(require_exact_integer_array(
            np.asarray(f.attrs["n_beads"]).reshape(1), field_name="n_beads")[0])

        # -- n_steps / chain-length convention ------------------------------
        if not np.array_equal(S["n_beads"].astype(np.int64),
                              np.full(n_rows, n_beads, np.int64)):
            raise ExtractionError("n_beads column != file n_beads")
        if not np.array_equal(S["n_steps"].astype(np.int64),
                              np.full(n_rows, n_beads - 1, np.int64)):
            raise ExtractionError("n_steps != n_beads - 1 for some row")

        # -- m_r block (exact integer; never truncate a float recreation) ---
        m_r = require_exact_integer_array(feats["m_r"][()], field_name="m_r")
        if m_r.shape != (n_rows, n_beads):
            raise ExtractionError(f"m_r shape {m_r.shape} != {(n_rows, n_beads)}")
        if not np.array_equal(m_r.sum(axis=1).astype(np.int64), m):
            raise ExtractionError("sum_r m_r != m for some row")

        # -- contact-pair offsets (aggregate) -------------------------------
        pairs = require_exact_integer_array(
            feats["contact_pairs/pairs"][()], field_name="contact_pairs/pairs")
        offsets = require_exact_integer_array(
            feats["contact_pairs/offsets"][()], field_name="contact_pairs/offsets")
        if offsets.shape[0] != n_rows + 1:
            raise ExtractionError("offsets length != n_rows+1")
        if int(offsets[0]) != 0:
            raise ExtractionError("offsets[0] != 0")
        if not np.all(np.diff(offsets) >= 0):
            raise ExtractionError("offsets not nondecreasing")
        if int(offsets[-1]) != pairs.shape[0]:
            raise ExtractionError("offsets[-1] != len(pairs)")
        if not np.array_equal(np.diff(offsets), m):
            bad = int(np.argmax(np.diff(offsets) != m))
            raise ExtractionError(f"[row {bad}] pair count != m")

        # -- stored (historical) definitions: validate the file against its OWN
        #    record; never require equality with the installed code version ---
        man = json.loads(f["metadata"].attrs["manifest"])
        if int(man.get("row_count", -1)) != n_rows:
            raise ExtractionError("manifest row_count mismatch")
        fixed_defs = man.get("fixed_bin_definitions") or dict(ico.FIXED_BIN_DEFINITIONS)
        scaled_defs = man.get("scaled_bin_definitions") or dict(ico.SCALED_BIN_DEFINITIONS)
        # Validate the scaled scheme always; validate the fixed partition only
        # when it is dimensionally applicable (long_threshold < n_beads).  For a
        # degenerate tiny chain the fixed scheme cannot partition, but the
        # per-row bin recomputation below still catches any tampering.
        try:
            ico.validate_scaled_bin_semantics(scaled_defs)
            if int(fixed_defs.get("long_threshold_fixed", n_beads)) < n_beads:
                ico.validate_bin_definitions(n_beads, fixed_defs, scaled_defs)
        except ico.ContactMapError as exc:
            raise ExtractionError(f"stored bin definitions invalid: {exc}") from exc
        stored_ver = str(man.get("definitions_version", ico.DEFINITIONS_VERSION))
        uses_current = (stored_ver == str(ico.DEFINITIONS_VERSION))

        # -- feature-schema version support + file/manifest agreement (Phase 3)
        file_ver = int(f.attrs.get("feature_schema_version", -1))
        if not (1 <= file_ver <= FEATURE_SCHEMA_VERSION):
            raise ExtractionError(
                f"unsupported feature_schema_version {file_ver} "
                f"(supported 1..{FEATURE_SCHEMA_VERSION})")
        man_ver = int(man.get("feature_schema_version", file_ver))
        if man_ver != file_ver:
            raise ExtractionError(
                f"manifest/file feature_schema_version mismatch "
                f"({man_ver} != {file_ver})")

        # -- per-file constant run_id / seed --------------------------------
        def _as_str(v):
            return v.decode() if isinstance(v, bytes) else str(v)
        run_ids = np.asarray([_as_str(v) for v in I["run_id"].tolist()])
        file_run_id = _as_str(f.attrs.get("run_id", run_ids[0] if run_ids.size else ""))
        if run_ids.size and not np.all(run_ids == file_run_id):
            raise ExtractionError("run_id is not constant across all feature rows")
        seed_col = I["seed"].astype(np.int64)
        if seed_col.size and int(seed_col.min()) != int(seed_col.max()):
            raise ExtractionError("seed is not constant across all feature rows")

        # -- authoritative temperature ladder (Phase 2) ---------------------
        ti = I["temperature_index"].astype(np.int64)
        tv = I["temperature"].astype(float)
        if ti.size and (int(ti.min()) < 0):
            raise ExtractionError("negative temperature_index")
        auth_temps = man.get("temperatures")
        nT_man = int(man.get("temperature_count",
                             len(np.unique(ti)) if ti.size else 0))
        if auth_temps is not None:
            auth = np.asarray(auth_temps, dtype=float).ravel()
            if int(auth.size) != nT_man:
                raise ExtractionError(
                    f"temperature_count {nT_man} != len(authoritative "
                    f"temperatures) {auth.size}")
            if not np.all(np.isfinite(auth)):
                raise ExtractionError("authoritative temperature ladder has a "
                                      "nonfinite entry")
        else:
            auth = None
        # Every stored feature-row temperature must be finite; each temperature
        # index must sit in [0, temperature_count); each index maps to ONE
        # consistent temperature that matches the authoritative ladder entry.
        if tv.size and not np.all(np.isfinite(tv)):
            bad = int(np.nonzero(~np.isfinite(tv))[0][0])
            raise ExtractionError(f"[row {bad}] nonfinite feature-row temperature")
        ladder = {}
        for r in range(n_rows):
            k = int(ti[r])
            if k < 0 or k >= nT_man:
                raise ExtractionError(
                    f"[row {r}] temperature_index {k} outside [0,{nT_man})")
            if k in ladder:
                if abs(ladder[k] - float(tv[r])) > 1e-9:
                    raise ExtractionError(
                        f"[row {r}] temperature {tv[r]} != ladder[{k}]={ladder[k]}")
            else:
                ladder[k] = float(tv[r])
            if auth is not None and abs(float(tv[r]) - float(auth[k])) > 1e-6:
                raise ExtractionError(
                    f"[row {r}] temperature {float(tv[r])} != authoritative "
                    f"ladder[{k}]={float(auth[k])}")

        # -- per-snapshot invariants (Phase 3) ------------------------------
        # Each snapshot must contain every temperature index and every walker id
        # exactly once, share one cycle/run_id/seed, and have exactly
        # temperature_count rows.  Cycles must strictly increase across snapshots.
        snap_idx = I["snapshot_index"].astype(np.int64)
        cyc = I["cycle"].astype(np.int64)
        walker = I["walker_id"].astype(np.int64)
        order = list(zip(snap_idx.tolist(), ti.tolist()))
        if order != sorted(order):
            raise ExtractionError(
                "rows not in (snapshot_index, temperature_index) order")
        uniq_snap = np.unique(snap_idx)
        if uniq_snap.size and not np.array_equal(
                uniq_snap, np.arange(int(uniq_snap[0]),
                                     int(uniq_snap[0]) + uniq_snap.size)):
            raise ExtractionError("snapshot_index values are not contiguous")
        expected_lane = np.arange(nT_man, dtype=np.int64)
        last_cycle = None
        for s in uniq_snap.tolist():
            rows_s = np.nonzero(snap_idx == s)[0]
            if rows_s.size != nT_man:
                raise ExtractionError(
                    f"[snapshot {s}] has {rows_s.size} rows != temperature_count "
                    f"{nT_man}")
            lane_s = np.sort(ti[rows_s])
            if not np.array_equal(lane_s, expected_lane):
                raise ExtractionError(
                    f"[snapshot {s}] temperature_index not a permutation of "
                    f"0..{nT_man - 1}")
            walk_s = np.sort(walker[rows_s])
            if not np.array_equal(walk_s, expected_lane):
                raise ExtractionError(
                    f"[snapshot {s}] walker_id not a permutation of "
                    f"0..{nT_man - 1}")
            cyc_s = cyc[rows_s]
            if int(cyc_s.min()) != int(cyc_s.max()):
                raise ExtractionError(
                    f"[snapshot {s}] cycle not constant across lanes")
            if run_ids.size and not np.all(run_ids[rows_s] == file_run_id):
                raise ExtractionError(f"[snapshot {s}] run_id not constant")
            if not np.all(seed_col[rows_s] == seed_col[rows_s][0]):
                raise ExtractionError(f"[snapshot {s}] seed not constant")
            this_cycle = int(cyc_s[0])
            if last_cycle is not None and this_cycle <= last_cycle:
                raise ExtractionError(
                    f"[snapshot {s}] cycle {this_cycle} not strictly greater "
                    f"than previous snapshot cycle {last_cycle}")
            last_cycle = this_cycle

        # -- source-configuration + model-record provenance (Phase 2) -------
        src_path = man.get("source_configuration_path")
        src_sha = man.get("source_configuration_sha256")
        if src_path and src_sha and Path(src_path).exists():
            if _file_sha256(src_path) != str(src_sha):
                raise ExtractionError(
                    "source_configuration_sha256 does not match the actual "
                    "source configuration file")
            try:
                with h5py.File(src_path, "r") as _sf:
                    src_meta = _read_metadata(_sf)
                for key, fv in (man.get("model_record") or {}).items():
                    sv = src_meta.get(key)
                    if isinstance(sv, bytes):
                        sv = sv.decode()
                    if key in ("param_names",) and sv is not None:
                        sv = [str(x) for x in np.asarray(sv).ravel().tolist()]
                    if key in ("model_params",) and sv is not None:
                        sv = [float(x) for x in np.asarray(sv).ravel().tolist()]
                        if fv is not None and (len(sv) != len(fv) or any(
                                abs(a - b) > 1e-9 for a, b in zip(sv, fv))):
                            raise ExtractionError(
                                "feature-file model_params disagree with the "
                                "source configuration")
                        continue
                    if key in ("Tref", "Tscale") and sv is not None and fv is not None:
                        if abs(float(sv) - float(fv)) > 1e-9:
                            raise ExtractionError(
                                f"feature-file {key} disagrees with source config")
                        continue
                    if sv is not None and fv is not None and str(sv) != str(fv):
                        raise ExtractionError(
                            f"feature-file model record {key} disagrees with "
                            f"source configuration")
            except OSError:
                pass  # unreadable source is treated as "unavailable", not a fail

        # -- primary-key uniqueness -----------------------------------------
        try:
            sch.validate_feature_primary_keys({k: I[k] for k in PRIMARY_KEY})
        except sch.SchemaError as exc:
            raise ExtractionError(f"primary-key validation failed: {exc}") from exc

        # -- aggregate graph / thermo identities ----------------------------
        if not (np.array_equal(S["contact_graph_edges"].astype(np.int64), m)
                and np.array_equal(S["sum_component_edges"].astype(np.int64), m)):
            raise ExtractionError("contact graph edge totals != m")
        if not np.array_equal(S["augmented_graph_edges"].astype(np.int64),
                              (n_beads - 1 + m)):
            raise ExtractionError("augmented edges != N-1+m")
        if not np.array_equal(S["augmented_graph_cycle_rank"].astype(np.int64), m):
            raise ExtractionError("augmented cycle rank != m")
        # -- thermodynamic identities (K authoritative; NO nonfinite substitution)
        # (Part 1.1) A thermodynamic file must carry finite b_T, K_T, u for every
        # row; q_T may only overflow to +inf / underflow to 0 in the documented
        # ranges (NaN and arbitrary inf are rejected); effective_energy_H must be
        # finite whenever T and u are finite.  Every failure identifies its row.
        b = S["b_T"][()].astype(float); K = S["K_T"][()].astype(float)
        q = S["q_T"][()].astype(float)
        u = S["reduced_potential_u"][()].astype(float)
        H = S["effective_energy_H"][()].astype(float)
        mf = m.astype(float)
        thermo_status = str(man.get("thermodynamic_status", "ok"))
        has_thermo = bool(np.isfinite(K).any())
        if thermo_status == "ok" and not has_thermo:
            raise ExtractionError(
                "manifest declares thermodynamic_status 'ok' but every K_T is "
                "nonfinite (thermodynamic columns were wholesale corrupted)")
        if has_thermo:
            for arr, nm in ((b, "b_T"), (K, "K_T"), (u, "reduced_potential_u")):
                bad = np.nonzero(~np.isfinite(arr))[0]
                if bad.size:
                    raise ExtractionError(
                        f"[row {int(bad[0])}] nonfinite {nm} in a thermodynamic "
                        f"file (NaN/inf substitution rejected)")
            bad = np.nonzero(np.abs(b - (-K)) > 1e-9)[0]
            if bad.size:
                raise ExtractionError(f"[row {int(bad[0])}] b_T != -K_T")
            tgt = -(mf * K)
            bad = np.nonzero(np.abs(u - tgt) > (1e-6 * np.maximum(1.0, np.abs(tgt)) + 1e-9))[0]
            if bad.size:
                raise ExtractionError(
                    f"[row {int(bad[0])}] reduced_potential_u != -m K_T")
            # q_T = exp(K): finite in-range, +inf on overflow, 0 on underflow.
            nanq = np.nonzero(np.isnan(q))[0]
            if nanq.size:
                raise ExtractionError(f"[row {int(nanq[0])}] q_T is NaN")
            in_range = (K >= -700.0) & (K <= 700.0)
            over = K > 700.0
            under = K < -700.0
            expK = np.exp(np.clip(K, -700.0, 700.0))
            bad = np.nonzero(in_range & (~np.isfinite(q)
                   | (np.abs(q - expK) > 1e-6 * np.maximum(1.0, expK) + 1e-9)))[0]
            if bad.size:
                raise ExtractionError(
                    f"[row {int(bad[0])}] q_T != exp(K_T) (finite range)")
            bad = np.nonzero(over & (q != np.inf))[0]
            if bad.size:
                raise ExtractionError(
                    f"[row {int(bad[0])}] q_T must be +inf on overflow "
                    f"(K_T={float(K[bad[0]]):.3g})")
            bad = np.nonzero(under & (q != 0.0))[0]
            if bad.size:
                raise ExtractionError(
                    f"[row {int(bad[0])}] q_T must underflow to 0 "
                    f"(K_T={float(K[bad[0]]):.3g})")
            # H = T u; reject arbitrary inf/NaN when T and u are finite.
            Tfin = np.isfinite(tv) & np.isfinite(u)
            bad = np.nonzero(Tfin & ~np.isfinite(H))[0]
            if bad.size:
                raise ExtractionError(
                    f"[row {int(bad[0])}] effective_energy_H nonfinite though "
                    f"T and u are finite")
            tgtH = tv * u
            bad = np.nonzero(Tfin & (np.abs(H - tgtH)
                   > 1e-6 * np.maximum(1.0, np.abs(tgtH)) + 1e-9))[0]
            if bad.size:
                raise ExtractionError(
                    f"[row {int(bad[0])}] effective_energy_H != T u")

        # -- geometric identities (Part 1.3) --------------------------------
        lam1 = S["gyration_lambda_1"][()].astype(float)
        lam2 = S["gyration_lambda_2"][()].astype(float)
        lam3 = S["gyration_lambda_3"][()].astype(float)
        rg2 = S["Rg2_lattice"][()].astype(float)
        asph = S["asphericity"][()].astype(float)
        gtol = 1e-6
        bad = np.nonzero((lam1 > lam2 + gtol) | (lam2 > lam3 + gtol))[0]
        if bad.size:
            raise ExtractionError(
                f"[row {int(bad[0])}] gyration eigenvalues not ascending")
        bad = np.nonzero(lam1 < -gtol)[0]
        if bad.size:
            raise ExtractionError(
                f"[row {int(bad[0])}] materially negative gyration eigenvalue")
        lam_sum = lam1 + lam2 + lam3
        bad = np.nonzero(np.abs(lam_sum - rg2) > gtol + 1e-6 * np.abs(rg2))[0]
        if bad.size:
            raise ExtractionError(
                f"[row {int(bad[0])}] lambda_1+lambda_2+lambda_3 != Rg2")
        exp_asph = lam3 - 0.5 * (lam1 + lam2)
        bad = np.nonzero(np.abs(asph - exp_asph) > gtol + 1e-6 * np.abs(exp_asph))[0]
        if bad.size:
            raise ExtractionError(
                f"[row {int(bad[0])}] asphericity != lambda_3-(lambda_1+lambda_2)/2")

        # -- per-row reconstruction from stored pairs (deep) ----------------
        if deep:
            for r in range(n_rows):
                lo, hi = int(offsets[r]), int(offsets[r + 1])
                p = pairs[lo:hi].astype(np.int64).reshape(-1, 2)
                mr = int(m[r])
                if p.shape[0] != mr:
                    raise ExtractionError(f"[row {r}] pair count {p.shape[0]} != m {mr}")
                if mr:
                    i, j = p[:, 0], p[:, 1]
                    if int(i.min()) < 0 or int(j.max()) >= n_beads:
                        raise ExtractionError(f"[row {r}] pair index out of bounds")
                    if not np.all(i < j):
                        raise ExtractionError(f"[row {r}] pair violates i<j")
                    if not np.all((j - i) > 1):
                        raise ExtractionError(f"[row {r}] bonded pair (j-i<=1)")
                    if not np.all(((j - i) % 2) == 1):
                        raise ExtractionError(f"[row {r}] even contour separation")
                    keys = list(zip(i.tolist(), j.tolist()))
                    if len(set(keys)) != mr:
                        raise ExtractionError(f"[row {r}] duplicate contact pair")
                    if keys != sorted(keys):
                        raise ExtractionError(f"[row {r}] pairs not lexicographically sorted")
                # stored m_r must satisfy cubic parity (even-r entries zero) and
                # the r<3 zero convention (Part 1.5), then equal the pairs.
                try:
                    ico.validate_m_r(m_r[r], n_beads=n_beads)
                except ico.ContactMapError as exc:
                    raise ExtractionError(f"[row {r}] stored m_r invalid: {exc}") from exc
                # reconstruct m_r
                mr_rec = ico.contact_separation_counts(p, n_beads)
                if not np.array_equal(mr_rec.astype(np.int64),
                                      m_r[r].astype(np.int64)):
                    raise ExtractionError(f"[row {r}] reconstructed m_r != stored m_r")
                # recompute pair motifs
                motifs = ico.count_pair_motifs(p)
                for key in ("pair_shared_endpoint", "pair_disjoint",
                            "pair_nested", "pair_interleaved", "pair_total"):
                    if int(S[key][r]) != int(motifs[key]):
                        raise ExtractionError(
                            f"[row {r}] {key} {int(S[key][r])} != recomputed "
                            f"{int(motifs[key])}")
                if int(motifs["pair_total"]) != mr * (mr - 1) // 2:
                    raise ExtractionError(f"[row {r}] pair_total != C(m,2)")
                # recompute graph summaries (integer fields exact)
                g = ico.contact_graph_summary(p, n_beads)
                for key in ("contact_vertices", "contact_graph_components",
                            "contact_graph_edges", "sum_component_edges",
                            "largest_component_vertices", "largest_component_edges",
                            "contact_graph_cycle_rank", "number_of_multiedge_components"):
                    if int(S[key][r]) != int(g[key]):
                        raise ExtractionError(
                            f"[row {r}] graph field {key} mismatch")
                # contact-graph floating fields (Part 1.2), absolute+relative tol
                for key in ("largest_component_fraction_of_N",
                            "largest_component_fraction_of_contact_vertices",
                            "mean_degree_nonisolated", "degree_variance_nonisolated"):
                    got = float(S[key][r]); exp = float(g[key])
                    if abs(got - exp) > 1e-9 + 1e-9 * abs(exp):
                        raise ExtractionError(
                            f"[row {r}] graph float field {key} {got} != "
                            f"recomputed {exp}")
                ag = ico.augmented_graph_summary(p, n_beads)
                for key in ("augmented_graph_vertices", "augmented_graph_edges",
                            "augmented_graph_components", "augmented_graph_cycle_rank",
                            "augmented_largest_component_vertices"):
                    if int(S[key][r]) != int(ag[key]):
                        raise ExtractionError(
                            f"[row {r}] augmented graph field {key} mismatch")
                for key in ("augmented_mean_degree", "augmented_degree_variance"):
                    got = float(S[key][r]); exp = float(ag[key])
                    if abs(got - exp) > 1e-9 + 1e-9 * abs(exp):
                        raise ExtractionError(
                            f"[row {r}] augmented graph float field {key} {got} "
                            f"!= recomputed {exp}")
                # separation summaries recomputed from stored pairs (Part 1.4)
                sm = float(S["mean_contact_separation"][r])
                sx = float(S["max_contact_separation"][r])
                if mr == 0:
                    if not (math.isnan(sm) and math.isnan(sx)):
                        raise ExtractionError(
                            f"[row {r}] zero-contact separation summary must be "
                            f"null/NaN; got mean={sm}, max={sx}")
                else:
                    seps = (p[:, 1] - p[:, 0]).astype(float)
                    if abs(sm - float(seps.mean())) > 1e-6:
                        raise ExtractionError(
                            f"[row {r}] mean_contact_separation {sm} != "
                            f"recomputed {float(seps.mean())}")
                    if int(sx) != int(seps.max()):
                        raise ExtractionError(
                            f"[row {r}] max_contact_separation {int(sx)} != "
                            f"recomputed {int(seps.max())}")
                # recompute contour bins from stored definitions
                fb = ico.bin_contact_separations_fixed(m_r[r], n_beads, fixed_defs)
                sb = ico.bin_contact_separations_scaled(m_r[r], n_beads, scaled_defs)
                for key, val in {**fb, **sb}.items():
                    if int(S[key][r]) != int(val):
                        raise ExtractionError(
                            f"[row {r}] contour bin {key} mismatch")
    return {"row_count": n_rows, "n_beads": n_beads,
            "definitions_version": stored_ver,
            "uses_current_definitions": bool(uses_current), "ok": True}


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

def run_quick_test() -> None:
    import tempfile

    if not _HAVE_H5PY:
        raise ExtractionError("quick-test requires h5py")
    chain = [(0, 0, 0), (1, 0, 0), (2, 0, 0), (2, 1, 0), (1, 1, 0), (0, 1, 0)]
    n_beads = len(chain)
    cp, _ = ico.build_contact_map(chain)
    expected_m = cp.shape[0]
    rg2 = ico.radius_of_gyration_squared(chain)
    ree2 = ico.end_to_end_distance_squared(chain)

    with tempfile.TemporaryDirectory() as tmp:
        inp = os.path.join(tmp, "snap.h5")
        nT = 2
        w = cio.SnapshotWriter(inp, n_beads=n_beads, n_temperatures=nT,
                               metadata={"run_id": "qt", "seed": 1,
                                         "temperatures": [300.0, 320.0],
                                         "model_name": "hs",
                                         "param_names": ["h", "s"],
                                         "model_params": [378.96, 1.39686],
                                         "Tref": 320.0, "Tscale": 80.0})
        for c in range(5):
            coords = np.stack([np.asarray(chain, dtype=np.int64)] * nT)
            w.append(cycle=c, coordinates=coords,
                     walker_id=np.array([0, 1] if c % 2 == 0 else [1, 0]),
                     contacts=np.full(nT, expected_m),
                     rg2_lattice=np.full(nT, rg2),
                     ree2_lattice=np.full(nT, ree2))
        w.mark_complete()
        w.close()

        formats = ["hdf5"] + (["parquet"] if _HAVE_PYARROW else [])
        for fmt in formats:
            outp = os.path.join(tmp, f"feat.{fmt}")
            # chunk_size 2 forces multiple streamed batches over 5 snapshots.
            info = extract(inp, outp, validate=True, output_format=fmt,
                           overwrite=True, chunk_size=2)
            assert info["row_count"] == 10, info
            assert all(v == 0 for v in info["validation_discrepancy_counts"].values()), info
            print(f"  extractor quick-test format={fmt}: PASSED")

        # Reject interrupted status without the flag.
        inp2 = os.path.join(tmp, "snap_interrupted.h5")
        w2 = cio.SnapshotWriter(inp2, n_beads=n_beads, n_temperatures=nT,
                                metadata={"run_id": "qt2"})
        w2.append(cycle=0,
                  coordinates=np.stack([np.asarray(chain, dtype=np.int64)] * nT),
                  walker_id=np.array([0, 1]),
                  contacts=np.full(nT, expected_m),
                  rg2_lattice=np.full(nT, rg2), ree2_lattice=np.full(nT, ree2))
        w2.close()
        try:
            extract(inp2, os.path.join(tmp, "f.h5"), output_format="hdf5",
                    overwrite=True)
            raise AssertionError("expected interrupted-status rejection")
        except ExtractionError as e:
            assert "complete" in str(e), str(e)
        info_i = extract(inp2, os.path.join(tmp, "f.h5"), output_format="hdf5",
                         overwrite=True, allow_interrupted=True)
        assert info_i["row_count"] == 2, info_i
        print("  extractor quick-test interrupted-handling: PASSED")
    print("extractor quick-test complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__,
    )
    ap.add_argument("--input", type=str, default=None,
                    help="input coordinate HDF5 file (from --save-configurations)")
    ap.add_argument("--output", type=str, default=None,
                    help="output feature table (.h5 default, or .parquet)")
    ap.add_argument("--chunk-size", type=int, default=256,
                    help="snapshots read/written per streamed chunk")
    ap.add_argument("--overwrite", action="store_true",
                    help="overwrite an existing output file")
    ap.add_argument("--validate", action="store_true",
                    help="run the full strict contact-map validation per config")
    ap.add_argument("--allow-interrupted", action="store_true",
                    dest="allow_interrupted",
                    help="extract committed rows even if status != complete")
    ap.add_argument("--format", type=str, default="hdf5",
                    choices=["hdf5", "parquet"],
                    help="output format (default hdf5; parquet needs pyarrow)")
    ap.add_argument("--experimental-parquet", action="store_true",
                    dest="experimental_parquet",
                    help="permit the EXPERIMENTAL Parquet export (scalar/tabular "
                         "only; HDF5 remains authoritative)")
    ap.add_argument("--quick-test", action="store_true",
                    help="run the self-contained extractor smoke test and exit")
    args = ap.parse_args()

    # Phase 11.3: refuse to run when JSON definitions and compatibility
    # constants disagree (skipped for the self-contained --quick-test).
    if not args.quick_test:
        import isaw_schema as _sch
        _sch.check_definitions_consistency()

    if args.quick_test:
        run_quick_test()
        return
    if args.format == "parquet" and not args.experimental_parquet:
        ap.error("Parquet is an EXPERIMENTAL scalar/tabular export (no ragged "
                 "contact pairs; HDF5 is authoritative). Pass "
                 "--experimental-parquet to use it in production.")
    if not args.input or not args.output:
        ap.error("--input and --output are required (or use --quick-test)")
    if args.chunk_size < 1:
        ap.error("--chunk-size must be >= 1")

    info = extract(
        args.input, args.output,
        chunk_size=args.chunk_size, overwrite=args.overwrite,
        validate=args.validate, output_format=args.format,
        allow_interrupted=args.allow_interrupted,
    )
    print(json.dumps({k: info[k] for k in
                      ("row_count", "n_beads", "temperature_count",
                       "validation_discrepancy_counts", "output_format",
                       "bin_definitions_source")},
                     indent=2))


if __name__ == "__main__":
    main()
