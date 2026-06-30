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

    Returns ``(fixed_defs, scaled_defs, source)`` where ``source`` is
    'input_file' when explicit definitions were present, else 'module_default'.
    Never silently substitutes module defaults when the input carries explicit
    definitions.
    """
    def _load(key):
        raw = meta.get(key)
        if raw is None:
            return None
        if isinstance(raw, bytes):
            raw = raw.decode()
        if isinstance(raw, str):
            return json.loads(raw)
        if isinstance(raw, dict):
            return dict(raw)
        return None

    fixed = _load("fixed_bin_definitions")
    scaled = _load("scaled_bin_definitions")
    if fixed is None or scaled is None:
        # Fall back to the structural_bin_definitions combined record if present.
        combined = _load("structural_bin_definitions")
        if isinstance(combined, dict):
            fixed = fixed or combined.get("fixed")
            scaled = scaled or combined.get("scaled")

    if fixed is not None and scaled is not None:
        ico.validate_bin_definitions(int(n_beads), fixed, scaled)
        return fixed, scaled, "input_file"
    # No explicit definitions in the file -> module defaults (recorded).
    return (dict(ico.FIXED_BIN_DEFINITIONS), dict(ico.SCALED_BIN_DEFINITIONS),
            "module_default")


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

    def __init__(self, path, *, n_beads, run_id, definitions):
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
        kv = {
            b"feature_schema_version": str(FEATURE_SCHEMA_VERSION).encode(),
            b"run_id": str(run_id).encode(),
            b"n_beads": str(int(n_beads)).encode(),
            b"definitions_version": str(ico.DEFINITIONS_VERSION).encode(),
            b"fixed_bin_definitions": json.dumps(definitions[0]).encode(),
            b"scaled_bin_definitions": json.dumps(definitions[1]).encode(),
            b"primary_key": json.dumps(list(PRIMARY_KEY)).encode(),
            b"format": b"parquet",
            b"experimental": b"true",
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
        fixed_defs, scaled_defs, defs_source = _resolve_input_bin_defs(meta, n_beads)
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
            writer = _StreamParquetWriter(output_path, n_beads=int(n_beads),
                                          run_id=run_id,
                                          definitions=(fixed_defs, scaled_defs))

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
            "definitions_version": ico.DEFINITIONS_VERSION,
            "bin_definitions_source": defs_source,
            "fixed_bin_definitions": fixed_defs,
            "scaled_bin_definitions": scaled_defs,
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

_FEATURE_META = {
    "run_id": ("string", "none", "run identifier", "metadata", False, 1),
    "seed": ("int", "none", "RNG seed", "metadata", False, 1),
    "snapshot_index": ("int", "none", "committed snapshot row index", "index", False, 1),
    "cycle": ("int", "cycles", "originating REMD cycle", "index", False, 1),
    "temperature_index": ("int", "none", "lane index", "index", False, 1),
    "temperature": ("float", "K", "lane temperature", "metadata", False, 1),
    "walker_id": ("int", "none", "walker occupying the lane", "index", False, 1),
    "n_beads": ("int", "beads", "chain length N", "config", False, 1),
    "n_steps": ("int", "bonds", "N-1", "config", False, 1),
    "m": ("int", "contacts", "total nonbonded contacts", "contact_map", False, 1),
    "Rg2_lattice": ("float", "lattice^2", "radius of gyration squared", "geometry", False, 1),
    "Ree2_lattice": ("float", "lattice^2", "end-to-end distance squared", "geometry", False, 1),
    "asphericity": ("float", "lattice^2", "lam3-0.5(lam1+lam2)", "geometry", False, 1),
    "mean_contact_separation": ("float", "beads", "mean r among contacts", "contact_map", True, 1),
    "max_contact_separation": ("float", "beads", "max r among contacts", "contact_map", True, 1),
    "b_T": ("float", "kT", "reduced contact bias b(T)", "thermo", True, 4),
    "K_T": ("float", "kT", "K(T)=-b(T) (authoritative)", "thermo", True, 4),
    "q_T": ("float", "none", "exp(K); inf on overflow", "thermo", True, 4),
    "reduced_potential_u": ("float", "kT", "u=-m K", "thermo", True, 4),
    "effective_energy_H": ("float", "energy", "H=T u (model-implied)", "thermo", True, 4),
}


def build_feature_dictionary() -> dict:
    """Machine-readable feature dictionary for every column."""
    fields = []
    for name in FEATURE_COLUMNS:
        kind = _col_dtype(name)
        dtype = {"str": "string", "float": "float64", "int": "int64"}[kind]
        units, _u2, definition, source, nullable, ver = _FEATURE_META.get(
            name, (dtype, "various", name.replace("_", " "),
                   "contact_map/graph", False, 1))
        if name in _FEATURE_META:
            units = _FEATURE_META[name][1]
        fields.append({
            "name": name, "dtype": dtype, "units": units,
            "definition": definition, "source": source,
            "cadence": "per (snapshot, lane) configuration",
            "nullable": bool(nullable),
            "schema_version_introduced": int(ver),
        })
    fields.append({
        "name": "m_r", "dtype": str(np.dtype(_M_R_DTYPE)), "units": "contacts",
        "definition": "dense contour-separation histogram length n_beads; "
                      "m_r[r]=#contacts with separation r; sum_r==m",
        "source": "contact_map", "cadence": "per configuration",
        "nullable": False, "schema_version_introduced": 1})
    fields.append({
        "name": "contact_pairs", "dtype": "ragged int64 (pairs+offsets)",
        "units": "bead-index pairs",
        "definition": "row i pairs = pairs[offsets[i]:offsets[i+1]]; "
                      "lexicographically sorted; row pair count == m",
        "source": "contact_map", "cadence": "per configuration",
        "nullable": False, "schema_version_introduced": 4})
    return {
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "definitions_version": ico.DEFINITIONS_VERSION,
        "primary_key": list(PRIMARY_KEY),
        "fields": fields,
    }


def write_feature_dictionary(path=None) -> str:
    p = Path(path) if path is not None else (
        Path(__file__).resolve().parent / "docs" / "feature_dictionary.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(build_feature_dictionary(), fh, indent=2)
    return str(p)


def validate_feature_file_hdf5(path: str) -> dict:
    """Comprehensive structural/identity validation of a feature HDF5 file.

    Raises :class:`ExtractionError` on any failure; returns a summary dict.
    """
    import isaw_schema as sch
    with h5py.File(path, "r") as f:
        if str(f.attrs.get("status")) != "complete":
            raise ExtractionError("feature file status != complete")
        feats = f["features"]
        scalars = feats["scalars"]
        idxg = feats["sample_index"]
        m = scalars["m"][()]
        n_rows = int(m.shape[0])
        committed = int(feats.attrs.get("committed_feature_rows", -1))
        if committed != n_rows:
            raise ExtractionError(
                f"committed_feature_rows {committed} != row count {n_rows}")
        for grp, label in ((scalars, "scalar"), (idxg, "index")):
            for name in grp:
                if grp[name].shape[0] != n_rows:
                    raise ExtractionError(f"{label} dataset {name} length mismatch")
        n_beads = int(f.attrs["n_beads"])
        m_r = feats["m_r"][()]
        if m_r.shape != (n_rows, n_beads):
            raise ExtractionError(f"m_r shape {m_r.shape} != {(n_rows, n_beads)}")
        if not np.array_equal(m_r.sum(axis=1).astype(np.int64), m.astype(np.int64)):
            raise ExtractionError("sum_r m_r != m for some row")
        # contact-pair offsets / counts
        pairs = feats["contact_pairs/pairs"][()]
        offsets = feats["contact_pairs/offsets"][()]
        if offsets.shape[0] != n_rows + 1:
            raise ExtractionError("offsets length != n_rows+1")
        if int(offsets[0]) != 0:
            raise ExtractionError("offsets[0] != 0")
        if not np.all(np.diff(offsets) >= 0):
            raise ExtractionError("offsets not nondecreasing")
        if int(offsets[-1]) != pairs.shape[0]:
            raise ExtractionError("offsets[-1] != len(pairs)")
        row_pair_counts = np.diff(offsets)
        if not np.array_equal(row_pair_counts.astype(np.int64), m.astype(np.int64)):
            raise ExtractionError("row pair count != m")
        if pairs.shape[0]:
            if int(pairs.min()) < 0 or int(pairs.max()) >= n_beads:
                raise ExtractionError("contact pair index out of bounds")
            if not np.all(pairs[:, 0] < pairs[:, 1]):
                raise ExtractionError("contact pair violates i<j")
        # graph identities
        cge = scalars["contact_graph_edges"][()]
        sce = scalars["sum_component_edges"][()]
        if not (np.array_equal(cge.astype(np.int64), m.astype(np.int64))
                and np.array_equal(sce.astype(np.int64), m.astype(np.int64))):
            raise ExtractionError("contact graph edge totals != m")
        aug_e = scalars["augmented_graph_edges"][()]
        aug_cyc = scalars["augmented_graph_cycle_rank"][()]
        if not np.array_equal(aug_e.astype(np.int64),
                              (n_beads - 1 + m).astype(np.int64)):
            raise ExtractionError("augmented edges != N-1+m")
        if not np.array_equal(aug_cyc.astype(np.int64), m.astype(np.int64)):
            raise ExtractionError("augmented cycle rank != m")
        # thermodynamic identity: u == -m K (where finite)
        K = scalars["K_T"][()]
        u = scalars["reduced_potential_u"][()]
        finite = np.isfinite(K) & np.isfinite(u)
        if finite.any():
            if not np.allclose(u[finite], -(m.astype(float)[finite]) * K[finite],
                               atol=1e-6, rtol=1e-9):
                raise ExtractionError("reduced_potential_u != -m K")
        # primary key uniqueness
        cols = {k: idxg[k][()] for k in PRIMARY_KEY}
        sch.validate_feature_primary_keys(cols)
        man = json.loads(f["metadata"].attrs["manifest"])
        if int(man.get("row_count", -1)) != n_rows:
            raise ExtractionError("manifest row_count mismatch")
        if str(man.get("definitions_version")) != str(ico.DEFINITIONS_VERSION):
            raise ExtractionError("definitions_version mismatch")
    return {"row_count": n_rows, "n_beads": n_beads, "ok": True}


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
    ap.add_argument("--quick-test", action="store_true",
                    help="run the self-contained extractor smoke test and exit")
    args = ap.parse_args()

    if args.quick_test:
        run_quick_test()
        return
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
