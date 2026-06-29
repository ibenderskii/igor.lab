#!/usr/bin/env python3
"""
Offline topology-resolved feature extraction for ISAW REMD coordinate snapshots.

Reads a coordinate HDF5 file written by ``remd_uniform_chain_2.py
--save-configurations`` and produces one feature row per (snapshot, temperature
lane) configuration *without rerunning REMD*.  Expensive O(m^2) pair-motif
classification lives here, not in the Monte Carlo hot loop.

Output is Parquet when ``pyarrow`` is available (default), otherwise an HDF5
table with the same columns and a documented schema.

Each row contains the scalar thermodynamic/geometric observables, contour-class
counts (m_short/m_medium/m_long and the full m_r vector as ``m_r_3``,
``m_r_5`` ... columns), pair-contact topology counts, and contact-graph
statistics.  See ``FEATURE_COLUMNS`` for the full list.

Usage
-----
    python extract_contact_motif_features.py \
        --input run_configurations.h5 \
        --output run_features.parquet \
        --validate

    python extract_contact_motif_features.py --quick-test
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import isaw_contact_observables as ico  # noqa: E402

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

FEATURE_SCHEMA_VERSION = 1

# Scalar (non-m_r) feature columns, in a deterministic order.
FEATURE_COLUMNS = [
    "run_id", "seed", "cycle", "temperature_index", "temperature", "walker_id",
    "sample_index", "n_beads", "n_steps", "m",
    "Rg2_lattice", "Ree2_lattice",
    "mean_contact_separation", "max_contact_separation",
    "m_short", "m_medium", "m_long",
    "pair_shared_endpoint", "pair_disjoint", "pair_nested", "pair_interleaved",
    "contact_vertices", "contact_graph_components",
    "largest_component_vertices", "largest_component_edges",
    "largest_component_fraction_of_N",
    "largest_component_fraction_of_contact_vertices",
    "mean_degree_nonisolated", "degree_variance_nonisolated",
    "contact_graph_cycle_rank", "number_of_multiedge_components",
]


class ExtractionError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Per-configuration feature computation
# ---------------------------------------------------------------------------

def compute_features_for_config(
    coordinates: np.ndarray,
    *,
    n_beads: int,
    bin_defs: dict,
    stored_contacts: int | None = None,
    validate: bool = False,
) -> dict:
    """Compute the full topology-resolved feature dict for one conformation."""
    cp, seps = ico.build_contact_map(coordinates)
    m = int(cp.shape[0])

    if stored_contacts is not None and m != int(stored_contacts):
        raise ExtractionError(
            f"recomputed contact count {m} disagrees with stored value "
            f"{int(stored_contacts)}"
        )
    if validate:
        ico.validate_contact_map(coordinates, cp, seps,
                                 expected_contact_count=m)

    m_r = ico.contact_separation_counts(cp, n_beads)
    binned = ico.bin_contact_separations(m_r, n_beads, bin_defs)
    sep_summary = ico.contact_separation_summary(seps)
    motifs = ico.count_pair_motifs(cp)
    graph = ico.contact_graph_summary(cp, n_beads)

    feat = {
        "n_beads": int(n_beads),
        "n_steps": int(n_beads) - 1,
        "m": m,
        "Rg2_lattice": ico.radius_of_gyration_squared(coordinates),
        "Ree2_lattice": ico.end_to_end_distance_squared(coordinates),
        "mean_contact_separation": sep_summary["mean_contact_separation"],
        "max_contact_separation": sep_summary["max_contact_separation"],
        "m_short": binned["m_short"],
        "m_medium": binned["m_medium"],
        "m_long": binned["m_long"],
        "pair_shared_endpoint": motifs["pair_shared_endpoint"],
        "pair_disjoint": motifs["pair_disjoint"],
        "pair_nested": motifs["pair_nested"],
        "pair_interleaved": motifs["pair_interleaved"],
    }
    feat.update(graph)
    feat["_m_r"] = m_r  # carried separately; expanded into m_r_<r> columns later
    return feat


# ---------------------------------------------------------------------------
# HDF5 reading / chunked iteration
# ---------------------------------------------------------------------------

def _read_metadata(f) -> dict:
    meta = {}
    g = f["metadata"]
    for k, v in g.attrs.items():
        meta[k] = v
    for k in g:  # datasets (e.g. temperatures, model_params)
        meta[k] = g[k][()]
    return meta


def extract(
    input_path: str,
    output_path: str,
    *,
    chunk_size: int = 256,
    overwrite: bool = False,
    validate: bool = False,
    output_format: str | None = None,
) -> dict:
    """Extract features for every (snapshot, lane) config to Parquet/HDF5."""
    if not _HAVE_H5PY:
        raise ExtractionError("reading snapshots requires h5py (not installed)")
    if not Path(input_path).exists():
        raise ExtractionError(f"input file not found: {input_path}")
    out = Path(output_path)
    if out.exists() and not overwrite:
        raise ExtractionError(
            f"output {output_path!r} exists; pass --overwrite to replace it"
        )

    fmt = output_format or ("parquet" if _HAVE_PYARROW else "hdf5")
    if fmt == "parquet" and not _HAVE_PYARROW:
        raise ExtractionError("Parquet output requested but pyarrow not available")

    with h5py.File(input_path, "r") as f:
        meta = _read_metadata(f)
        snap = f["snapshots"]
        coords_ds = snap["coordinates"]
        n_snap, nT, n_beads, _ = coords_ds.shape
        cycle_ds = snap["cycle"]
        walker_ds = snap["walker_id"]
        contacts_ds = snap["contacts"]

        run_id = str(meta.get("run_id", Path(input_path).stem))
        seed = int(meta.get("seed", -1))
        temps = np.asarray(meta.get("temperatures",
                                    np.full(nT, np.nan)), dtype=float)
        bin_defs_raw = meta.get("structural_bin_definitions", None)
        if bin_defs_raw is not None:
            bin_defs = json.loads(bin_defs_raw) if isinstance(
                bin_defs_raw, str) else dict(bin_defs_raw)
        else:
            bin_defs = ico.default_bin_definitions(int(n_beads))

        # Determine the m_r column set (odd r in [3, n_beads-1]).
        mr_columns = [r for r in range(3, int(n_beads)) if r % 2 == 1]

        rows: list[dict] = []
        sample_index = 0
        for s0 in range(0, n_snap, chunk_size):
            s1 = min(s0 + chunk_size, n_snap)
            coords_block = coords_ds[s0:s1]
            cyc_block = cycle_ds[s0:s1]
            walk_block = walker_ds[s0:s1]
            cont_block = contacts_ds[s0:s1]
            for si in range(s1 - s0):
                for k in range(nT):
                    feat = compute_features_for_config(
                        coords_block[si, k].astype(np.int64),
                        n_beads=int(n_beads), bin_defs=bin_defs,
                        stored_contacts=int(cont_block[si, k]),
                        validate=validate,
                    )
                    m_r = feat.pop("_m_r")
                    row = {
                        "run_id": run_id,
                        "seed": seed,
                        "cycle": int(cyc_block[si]),
                        "temperature_index": int(k),
                        "temperature": float(temps[k]) if k < temps.size else math.nan,
                        "walker_id": int(walk_block[si, k]),
                        "sample_index": sample_index,
                    }
                    row.update(feat)
                    for r in mr_columns:
                        row[f"m_r_{r}"] = int(m_r[r]) if r < len(m_r) else 0
                    rows.append(row)
                    sample_index += 1

        column_order = list(FEATURE_COLUMNS) + [f"m_r_{r}" for r in mr_columns]
        _write_table(rows, column_order, output_path, fmt, run_id=run_id,
                     n_beads=int(n_beads), bin_defs=bin_defs)

    return {
        "input": str(input_path),
        "output": str(output_path),
        "format": fmt,
        "n_rows": len(rows),
        "n_snapshots": int(n_snap),
        "n_temperatures": int(nT),
        "n_beads": int(n_beads),
        "mr_columns": mr_columns,
    }


def _write_table(rows, column_order, output_path, fmt, *, run_id, n_beads,
                 bin_defs) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    if fmt == "parquet":
        import pyarrow as pa
        import pyarrow.parquet as pq
        cols = {name: [row.get(name) for row in rows] for name in column_order}
        table = pa.table(cols)
        table = table.replace_schema_metadata({
            "feature_schema_version": str(FEATURE_SCHEMA_VERSION),
            "run_id": run_id,
            "n_beads": str(n_beads),
            "structural_bin_definitions": json.dumps(bin_defs),
        })
        pq.write_table(table, output_path)
    elif fmt == "hdf5":
        import h5py as _h5
        with _h5.File(output_path, "w") as fo:
            fo.attrs["feature_schema_version"] = FEATURE_SCHEMA_VERSION
            fo.attrs["run_id"] = run_id
            fo.attrs["n_beads"] = int(n_beads)
            fo.attrs["structural_bin_definitions"] = json.dumps(bin_defs)
            fo.attrs["columns"] = json.dumps(column_order)
            g = fo.create_group("features")
            for name in column_order:
                vals = [row.get(name) for row in rows]
                arr = _as_h5_array(vals)
                g.create_dataset(name, data=arr)
    else:
        raise ExtractionError(f"unknown output format {fmt!r}")


def _as_h5_array(vals):
    if all(isinstance(v, str) for v in vals):
        return np.asarray(vals, dtype=h5py.string_dtype())
    arr = np.asarray([np.nan if v is None else v for v in vals])
    return arr


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

def run_quick_test() -> None:
    import tempfile

    # Build a tiny synthetic snapshot file with a hand conformation.
    if not _HAVE_H5PY:
        raise ExtractionError("quick-test requires h5py")
    chain = [(0, 0, 0), (1, 0, 0), (2, 0, 0), (2, 1, 0), (1, 1, 0), (0, 1, 0)]
    n_beads = len(chain)
    bin_defs = ico.default_bin_definitions(n_beads)
    cp, _ = ico.build_contact_map(chain)
    expected_m = cp.shape[0]

    with tempfile.TemporaryDirectory() as tmp:
        inp = os.path.join(tmp, "snap.h5")
        with h5py.File(inp, "w") as f:
            g = f.create_group("metadata")
            g.attrs["run_id"] = "qt"
            g.attrs["seed"] = 1
            g.attrs["structural_bin_definitions"] = json.dumps(bin_defs)
            g.create_dataset("temperatures", data=np.array([300.0, 320.0]))
            sg = f.create_group("snapshots")
            coords = np.zeros((2, 2, n_beads, 3), dtype=np.int16)
            for si in range(2):
                for k in range(2):
                    coords[si, k] = np.asarray(chain, dtype=np.int16)
            sg.create_dataset("coordinates", data=coords)
            sg.create_dataset("cycle", data=np.array([0, 5], dtype=np.int64))
            sg.create_dataset("walker_id",
                              data=np.array([[0, 1], [1, 0]], dtype=np.int64))
            sg.create_dataset("contacts",
                              data=np.full((2, 2), expected_m, dtype=np.int64))
            sg.create_dataset("rg2_lattice", data=np.zeros((2, 2)))
            sg.create_dataset("ree2_lattice", data=np.ones((2, 2)))

        # Parquet path if available, else HDF5.
        for fmt in (["parquet"] if _HAVE_PYARROW else []) + ["hdf5"]:
            outp = os.path.join(tmp, f"feat.{fmt}")
            info = extract(inp, outp, validate=True, output_format=fmt,
                           overwrite=True)
            assert info["n_rows"] == 4, info
            if fmt == "parquet":
                import pyarrow.parquet as pq
                t = pq.read_table(outp)
                d = t.to_pydict()
                assert all(v == expected_m for v in d["m"]), d["m"]
                # pair-motif invariant: sum == C(m,2)
                for i in range(len(d["m"])):
                    mm = d["m"][i]
                    tot = (d["pair_shared_endpoint"][i] + d["pair_disjoint"][i]
                           + d["pair_nested"][i] + d["pair_interleaved"][i])
                    assert tot == mm * (mm - 1) // 2, (i, tot, mm)
                # m_r columns sum to m
                mr_cols = [c for c in t.column_names if c.startswith("m_r_")]
                for i in range(len(d["m"])):
                    s = sum(d[c][i] for c in mr_cols)
                    assert s == d["m"][i], (i, s, d["m"][i])
            print(f"  extractor quick-test format={fmt}: PASSED")
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
                    help="output feature table (.parquet or .h5)")
    ap.add_argument("--chunk-size", type=int, default=256,
                    help="snapshots read per chunk")
    ap.add_argument("--overwrite", action="store_true",
                    help="overwrite an existing output file")
    ap.add_argument("--validate", action="store_true",
                    help="fully validate every contact map (slower)")
    ap.add_argument("--format", type=str, default=None,
                    choices=["parquet", "hdf5"],
                    help="output format; default parquet if pyarrow available")
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
    )
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
