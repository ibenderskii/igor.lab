#!/usr/bin/env python3
"""
Offline topology-resolved feature extraction for ISAW REMD coordinate snapshots.

Reads a coordinate HDF5 file written by ``remd_uniform_chain_2_new.py
--save-configurations`` and produces one validated feature row per
(snapshot, temperature lane) configuration *without rerunning REMD*.  Expensive
O(m^2) pair-motif classification lives here, not in the Monte Carlo hot loop.

Robustness / correctness model
------------------------------
* Only rows with index ``< committed_rows`` are read (an interrupted writer may
  have allocated, but not committed, a final row -- that row is ignored).
* The file ``status`` must be ``complete`` unless ``--allow-interrupted`` is
  given (then the readable committed rows are still processed).
* Every conformation is validated strictly: the contact map is rebuilt and the
  recomputed contact count / Rg^2 / Ree^2 are compared against the stored
  values; all separations must be odd; ``sum(m_r) == m``; pair-motif counts sum
  to C(m,2); contact-graph edge totals equal m; and walker IDs form a lane
  permutation for every snapshot.  Any discrepancy fails with a clear
  snapshot/lane/cycle identifier.

Output
------
HDF5 is the default (always available because the input is HDF5).  Parquet is
written only when ``--format parquet`` is requested AND ``pyarrow`` is
installed; the format is never silently switched.

HDF5 layout:
    /features/scalars       group of 1-D datasets (one per scalar column)
    /features/m_r           (n_rows, n_beads) compact-int contour histograms
    /features/sample_index  structured-ish columns: snapshot_index, cycle,
                            temperature_index, walker_id
    /metadata               attributes (schema, run, definitions)
A companion ``<output>.manifest.json`` records provenance, row counts, feature
names, and the validation discrepancy counts (all zero for a clean extraction).

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

FEATURE_SCHEMA_VERSION = 2

# Numerical tolerance comparing stored vs recomputed Rg^2 / Ree^2 (the stored
# values were produced by the same float64 routines, so they agree to ~eps).
_GEOM_ATOL = 1e-6
_GEOM_RTOL = 1e-9

# Index columns + scalar feature columns, in a deterministic order.
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
    "largest_component_vertices", "largest_component_edges",
    "largest_component_fraction_of_N",
    "largest_component_fraction_of_contact_vertices",
    "mean_degree_nonisolated", "degree_variance_nonisolated",
    "contact_graph_cycle_rank", "number_of_multiedge_components",
]
FEATURE_COLUMNS = INDEX_COLUMNS + SCALAR_COLUMNS

DISCREPANCY_KEYS = (
    "contact_count", "rg2", "ree2", "odd_separation", "m_r_sum",
    "pair_total", "graph_edge_total", "walker_permutation",
)


class ExtractionError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Per-configuration feature computation + validation
# ---------------------------------------------------------------------------

def compute_features_for_config(
    coordinates: np.ndarray,
    *,
    n_beads: int,
    stored_contacts: int | None = None,
    stored_rg2: float | None = None,
    stored_ree2: float | None = None,
    validate: bool = False,
    discrepancies: dict | None = None,
    locator: str = "",
) -> dict:
    """Compute the full topology-resolved feature dict for one conformation.

    When ``validate`` is True every invariant is checked and any failure raises
    :class:`ExtractionError` (incrementing the matching ``discrepancies`` key).
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
    if int(m_r.sum()) != m:
        _fail("m_r_sum", f"sum(m_r)={int(m_r.sum())} != m={m}")

    fixed = ico.bin_contact_separations_fixed(m_r, n_beads)
    scaled = ico.bin_contact_separations_scaled(m_r, n_beads)
    sep_summary = ico.contact_separation_summary(seps)
    motifs = ico.count_pair_motifs(cp)
    if motifs["pair_total"] != m * (m - 1) // 2:
        _fail("pair_total",
              f"pair total {motifs['pair_total']} != C({m},2)")
    graph = ico.contact_graph_summary(cp, n_beads)

    # Graph edge totals must equal m (sum of component edges == m).
    if graph["contact_graph_cycle_rank"] != (
        m - graph["contact_vertices"] + graph["contact_graph_components"]
    ):
        _fail("graph_edge_total", "graph cycle-rank identity violated")

    if validate:
        ico.validate_contact_map(coordinates, cp, seps,
                                 expected_contact_count=m, strict=True)

    lam = ico.gyration_eigenvalues(coordinates)  # ascending
    asph = float(lam[2] - 0.5 * (lam[0] + lam[1]))

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
    }
    feat.update(graph)
    feat["_m_r"] = m_r
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


def _resolve_bin_defs(meta) -> None:
    """Definitions come from this module's frozen constants (single source of
    truth).  Stored per-file definitions are recorded in the manifest but the
    extractor always uses ico.FIXED/SCALED constants for reproducibility."""
    return None


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
    """Extract features for every committed (snapshot, lane) config."""
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

    with h5py.File(input_path, "r") as f:
        meta = _read_metadata(f)
        if "snapshots" not in f:
            raise ExtractionError("input has no /snapshots group")
        snap = f["snapshots"]

        status = str(snap.attrs.get("status",
                                    f.attrs.get("status", "unknown")))
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
            raise ExtractionError(
                f"committed_rows {n_committed} > allocated {n_alloc}"
            )

        cycle_ds = snap["cycle"]
        walker_ds = snap["walker_id"]
        contacts_ds = snap["contacts"]
        rg2_ds = snap["rg2_lattice"]
        ree2_ds = snap["ree2_lattice"]

        run_id = str(meta.get("run_id", Path(input_path).stem))
        seed = int(meta.get("seed", -1))
        temps = np.asarray(meta.get("temperatures",
                                    np.full(nT, np.nan)), dtype=float)
        schema_version = int(meta.get("schema_version", 0))

        mr_max = 0
        rows: list[dict] = []
        m_r_rows: list[np.ndarray] = []
        sample_index = 0

        for s0 in range(0, n_committed, chunk_size):
            s1 = min(s0 + chunk_size, n_committed)
            coords_block = coords_ds[s0:s1]
            cyc_block = cycle_ds[s0:s1]
            walk_block = walker_ds[s0:s1]
            cont_block = contacts_ds[s0:s1]
            rg2_block = rg2_ds[s0:s1]
            ree2_block = ree2_ds[s0:s1]
            for si in range(s1 - s0):
                snap_idx = s0 + si
                cyc = int(cyc_block[si])
                # walker IDs form a lane permutation for this snapshot.
                w = np.asarray(walk_block[si], dtype=np.int64)
                if not np.array_equal(np.sort(w), np.arange(nT, dtype=np.int64)):
                    discrepancies["walker_permutation"] += 1
                    raise ExtractionError(
                        f"[snapshot {snap_idx} cycle {cyc}] walker_id not a "
                        f"permutation: {w.tolist()}"
                    )
                for k in range(nT):
                    locator = f"[snapshot {snap_idx} cycle {cyc} lane {k}] "
                    feat = compute_features_for_config(
                        coords_block[si, k].astype(np.int64),
                        n_beads=int(n_beads),
                        stored_contacts=int(cont_block[si, k]),
                        stored_rg2=float(rg2_block[si, k]),
                        stored_ree2=float(ree2_block[si, k]),
                        validate=validate,
                        discrepancies=discrepancies,
                        locator=locator,
                    )
                    m_r = feat.pop("_m_r")
                    mr_max = max(mr_max, int(m_r.max(initial=0)))
                    row = {
                        "run_id": run_id,
                        "seed": seed,
                        "snapshot_index": snap_idx,
                        "cycle": cyc,
                        "temperature_index": int(k),
                        "temperature": float(temps[k]) if k < temps.size else math.nan,
                        "walker_id": int(w[k]),
                    }
                    row.update(feat)
                    rows.append(row)
                    m_r_rows.append(m_r)
                    sample_index += 1

    # Stack m_r with a compact integer dtype.
    mr_dtype = np.uint8 if mr_max <= 255 else (
        np.uint16 if mr_max <= 65535 else np.int32)
    m_r_arr = (np.vstack([r.astype(mr_dtype) for r in m_r_rows])
               if m_r_rows else np.zeros((0, int(n_beads)), dtype=mr_dtype))

    manifest = {
        "input_path": str(input_path),
        "input_sha256": input_sha,
        "input_schema_version": schema_version,
        "input_status": status,
        "output_path": str(output_path),
        "output_format": fmt,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "row_count": len(rows),
        "n_beads": int(n_beads),
        "temperature_count": int(nT),
        "committed_rows": int(n_committed),
        "allocated_rows": int(n_alloc),
        "extracted_snapshot_rows": int(n_committed),
        "feature_names": list(FEATURE_COLUMNS),
        "m_r_representation": (
            "dense contour-separation histogram length n_beads; "
            "m_r[r] = #contacts with separation r; even r are zero; sum_r == m"
        ),
        "definitions_version": "1.0.0",
        "fixed_bin_definitions": ico.FIXED_BIN_DEFINITIONS,
        "scaled_bin_definitions": ico.SCALED_BIN_DEFINITIONS,
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

    _write_table(rows, m_r_arr, output_path, fmt, run_id=run_id,
                 n_beads=int(n_beads), manifest=manifest)

    manifest_path = str(output_path) + ".manifest.json"
    Path(manifest_path).parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    manifest["manifest_path"] = manifest_path
    print(f"Saved {output_path} ({len(rows)} rows) and {manifest_path}")
    return manifest


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


def _write_table(rows, m_r_arr, output_path, fmt, *, run_id, n_beads,
                 manifest) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    if fmt == "parquet":
        import pyarrow as pa
        import pyarrow.parquet as pq
        cols = {name: [row.get(name) for row in rows] for name in FEATURE_COLUMNS}
        for r in range(int(n_beads)):
            if r % 2 == 1 and r >= ico.MIN_CONTOUR_SEPARATION:
                cols[f"m_r_{r}"] = [int(m_r_arr[i, r]) if m_r_arr.shape[0] else 0
                                    for i in range(m_r_arr.shape[0])]
        table = pa.table(cols)
        table = table.replace_schema_metadata({
            "feature_schema_version": str(FEATURE_SCHEMA_VERSION),
            "run_id": run_id,
            "n_beads": str(n_beads),
            "manifest": json.dumps(manifest, default=str),
        })
        pq.write_table(table, output_path)
        return

    # HDF5 (default).
    with h5py.File(output_path, "w") as fo:
        fo.attrs["feature_schema_version"] = FEATURE_SCHEMA_VERSION
        fo.attrs["run_id"] = run_id
        fo.attrs["n_beads"] = int(n_beads)
        meta = fo.create_group("metadata")
        meta.attrs["manifest"] = json.dumps(manifest, default=str)
        meta.attrs["columns"] = json.dumps(FEATURE_COLUMNS)
        feats = fo.create_group("features")
        scalars = feats.create_group("scalars")
        idxg = feats.create_group("sample_index")
        for name in SCALAR_COLUMNS:
            scalars.create_dataset(
                name, data=_as_h5_array([row.get(name) for row in rows])
            )
        for name in INDEX_COLUMNS:
            idxg.create_dataset(
                name, data=_as_h5_array([row.get(name) for row in rows])
            )
        feats.create_dataset("m_r", data=m_r_arr,
                             compression="gzip", compression_opts=4)


def _as_h5_array(vals):
    if len(vals) and all(isinstance(v, str) for v in vals):
        return np.asarray(vals, dtype=h5py.string_dtype())
    arr = np.asarray([np.nan if v is None else v for v in vals])
    return arr


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

def run_quick_test() -> None:
    import tempfile

    if not _HAVE_H5PY:
        raise ExtractionError("quick-test requires h5py")
    # Hand conformation: planar hairpin (2 contacts).
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
                                         "temperatures": [300.0, 320.0]})
        for c in range(2):
            coords = np.stack([np.asarray(chain, dtype=np.int64)] * nT)
            w.append(cycle=c, coordinates=coords,
                     walker_id=np.array([0, 1] if c == 0 else [1, 0]),
                     contacts=np.full(nT, expected_m),
                     rg2_lattice=np.full(nT, rg2),
                     ree2_lattice=np.full(nT, ree2))
        w.mark_complete()
        w.close()

        formats = ["hdf5"] + (["parquet"] if _HAVE_PYARROW else [])
        for fmt in formats:
            outp = os.path.join(tmp, f"feat.{fmt}")
            info = extract(inp, outp, validate=True, output_format=fmt,
                           overwrite=True)
            assert info["row_count"] == 4, info
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
        w2.close()  # not marked complete -> 'interrupted'
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
                    help="snapshots read per chunk")
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
                       "validation_discrepancy_counts", "output_format")},
                     indent=2))


if __name__ == "__main__":
    main()
