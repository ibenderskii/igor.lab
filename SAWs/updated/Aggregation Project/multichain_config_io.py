#!/usr/bin/env python3
"""
Multi-chain output writers and coordinate-snapshot streaming (Stage 2).

Two responsibilities:

1. :class:`MultiChainSnapshotWriter` -- a chunked HDF5 writer whose coordinate
   dataset has EXPLICIT chain identity, dimensions::

       (snapshot, temperature_lane, chain, monomer, xyz)

   The single-chain HDF5 schema ``(snapshot, lane, bead, xyz)`` is deliberately
   NOT reused for multi-chain coordinates.  The writer keeps the transactional
   ``committed_rows`` / ``status`` semantics of ``isaw_config_io`` so an
   interrupted file is always readable up to ``committed_rows`` and is only
   marked ``complete`` on an explicit :meth:`mark_complete`.

2. Canonical text/array outputs for a run: results CSV, swap-rate CSV, move-
   acceptance CSV, distributions NPZ, and the run-summary JSON.

This module has no sampler/model imports, so it stays acyclic.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

try:  # h5py only required when configuration saving is requested
    import h5py
    _HAVE_H5PY = True
except Exception:  # pragma: no cover - import guard
    h5py = None
    _HAVE_H5PY = False

MULTICHAIN_SNAPSHOT_SCHEMA_VERSION = 1
MULTICHAIN_OUTPUT_SCHEMA_VERSION = 1

_INT16_MIN, _INT16_MAX = -32768, 32767
_INT16_SAFE = 30000

STATUS_RUNNING = "running"
STATUS_COMPLETE = "complete"
STATUS_INTERRUPTED = "interrupted"


def h5py_available() -> bool:
    return _HAVE_H5PY


class SnapshotWriterError(RuntimeError):
    pass


def _json_safe(obj):
    """Recursively convert NumPy/inf to plain JSON-safe Python."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_json_safe(v) for v in obj.tolist()]
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


# ---------------------------------------------------------------------------
# Snapshot writer
# ---------------------------------------------------------------------------

def _choose_coord_dtype(coords: np.ndarray) -> str:
    amax = int(np.abs(coords).max()) if coords.size else 0
    return "int16" if amax <= _INT16_SAFE else "int32"


def committed_rows(group) -> int:
    if "committed_rows" in group.attrs:
        return int(group.attrs["committed_rows"])
    if "n_snapshots" in group.attrs:
        return int(group.attrs["n_snapshots"])
    return int(group["cycle"].shape[0]) if "cycle" in group else 0


class MultiChainSnapshotWriter:
    """Append-only chunked HDF5 writer for multi-chain coordinate snapshots."""

    def __init__(
        self,
        path: str,
        *,
        n_chains: int,
        chain_length: int,
        n_temperatures: int,
        metadata: dict,
        coord_dtype: str = "int32",
        flush_interval: int = 50,
        overwrite: bool = False,
    ) -> None:
        # Production snapshots default to int32.  The writer allocates the
        # coordinate dtype once (from the first snapshot) and cannot safely widen
        # it later, so it never auto-selects int16: an early small-coordinate
        # snapshot must not lock the dataset to a width that a later, larger
        # coordinate would overflow.  int16 remains available for explicit,
        # size-bounded use (tests / backward compatibility); "auto" is retained
        # for callers that opt in to first-snapshot width selection.
        if not _HAVE_H5PY:
            raise SnapshotWriterError(
                "Configuration saving requires the 'h5py' package, which is not "
                "available. Install h5py or omit --save-configurations.")
        if coord_dtype not in ("auto", "int16", "int32"):
            raise SnapshotWriterError(f"invalid coord_dtype {coord_dtype!r}")
        self.path = str(path)
        self.M = int(n_chains)
        self.N = int(chain_length)
        self.nT = int(n_temperatures)
        self.flush_interval = max(1, int(flush_interval))
        self._coord_dtype = None if coord_dtype == "auto" else str(coord_dtype)
        self._n_written = 0
        self._n_allocated = 0
        self._last_cycle = None
        self._completed = False
        self._datasets_created = False

        p = Path(self.path)
        if p.exists() and not overwrite:
            raise SnapshotWriterError(
                f"configuration file {self.path!r} already exists; refusing to "
                f"overwrite (pass overwrite=True).")
        p.parent.mkdir(parents=True, exist_ok=True)

        self._f = h5py.File(self.path, "w")
        self._write_metadata(metadata)
        self._snap = self._f.create_group("snapshots")
        self._snap.attrs["committed_rows"] = 0
        self._snap.attrs["n_snapshots"] = 0
        self._snap.attrs["status"] = STATUS_RUNNING
        self._snap.attrs["schema_version"] = MULTICHAIN_SNAPSHOT_SCHEMA_VERSION
        self._snap.attrs["coordinate_dims"] = "snapshot,temperature_lane,chain,monomer,xyz"
        self._f.attrs["status"] = STATUS_RUNNING
        self._f.flush()

    def _write_metadata(self, metadata: dict) -> None:
        meta = self._f.create_group("metadata")
        meta.attrs["schema_version"] = MULTICHAIN_SNAPSHOT_SCHEMA_VERSION
        for key, value in metadata.items():
            if value is None:
                meta.attrs[key] = "null"
            elif isinstance(value, dict):
                meta.attrs[key] = json.dumps(_json_safe(value))
            elif isinstance(value, (list, tuple, np.ndarray)):
                arr = np.asarray(value)
                if arr.dtype.kind in ("U", "S", "O"):
                    meta.create_dataset(
                        key, data=np.asarray([str(v) for v in value],
                                             dtype=h5py.string_dtype()))
                else:
                    meta.create_dataset(key, data=arr)
            elif isinstance(value, (bool, np.bool_)):
                meta.attrs[key] = bool(value)
            elif isinstance(value, (int, np.integer)):
                meta.attrs[key] = int(value)
            elif isinstance(value, (float, np.floating)):
                meta.attrs[key] = float(value)
            else:
                meta.attrs[key] = str(value)

    def _create_datasets(self, coord_dtype: str) -> None:
        nT, M, N = self.nT, self.M, self.N
        g = self._snap
        chunk_rows = min(self.flush_interval, 64)
        g.create_dataset("cycle", shape=(0,), maxshape=(None,), dtype="int64",
                         chunks=(chunk_rows,))
        g.create_dataset(
            "coordinates", shape=(0, nT, M, N, 3), maxshape=(None, nT, M, N, 3),
            dtype=coord_dtype, chunks=(chunk_rows, nT, M, N, 3),
            compression="gzip", compression_opts=4)
        for name in ("walker_id", "m_intra", "m_inter", "largest_cluster_size"):
            g.create_dataset(name, shape=(0, nT), maxshape=(None, nT),
                             dtype="int64", chunks=(chunk_rows, nT))
        g.create_dataset("mean_chain_rg2", shape=(0, nT), maxshape=(None, nT),
                         dtype="float64", chunks=(chunk_rows, nT))
        g.attrs["coordinate_dtype"] = coord_dtype
        self._datasets_created = True

    def append(
        self, *, cycle: int, coordinates: np.ndarray, walker_id: np.ndarray,
        m_intra: np.ndarray, m_inter: np.ndarray, mean_chain_rg2: np.ndarray,
        largest_cluster_size: np.ndarray,
    ) -> None:
        nT, M, N = self.nT, self.M, self.N
        coords = np.asarray(coordinates)
        if coords.shape != (nT, M, N, 3):
            raise SnapshotWriterError(
                f"coordinates shape {coords.shape} != {(nT, M, N, 3)}")
        cycle_i = int(cycle)
        if self._last_cycle is not None and cycle_i <= self._last_cycle:
            raise SnapshotWriterError(
                f"cycle numbers must strictly increase; got {cycle_i} after "
                f"{self._last_cycle}")
        w = np.asarray(walker_id).astype(np.int64)
        if w.shape != (nT,) or not np.array_equal(
                np.sort(w), np.arange(nT, dtype=np.int64)):
            raise SnapshotWriterError("walker_id must be a permutation of 0..nT-1")

        def _vec(a, name, nonneg=True):
            arr = np.asarray(a)
            if arr.shape != (nT,):
                raise SnapshotWriterError(f"{name} shape {arr.shape} != {(nT,)}")
            return arr

        mi = _vec(m_intra, "m_intra").astype(np.int64)
        me = _vec(m_inter, "m_inter").astype(np.int64)
        lcs = _vec(largest_cluster_size, "largest_cluster_size").astype(np.int64)
        rg2 = _vec(mean_chain_rg2, "mean_chain_rg2").astype(np.float64)
        if np.any(mi < 0) or np.any(me < 0) or np.any(lcs < 0):
            raise SnapshotWriterError("contact/cluster counts must be nonnegative")
        if not np.all(np.isfinite(rg2)) or np.any(rg2 < 0):
            raise SnapshotWriterError("mean_chain_rg2 must be finite and nonnegative")

        if not self._datasets_created:
            dtype = self._coord_dtype or _choose_coord_dtype(coords)
            self._create_datasets(dtype)
        dtype = self._snap.attrs["coordinate_dtype"]
        lo, hi = (_INT16_MIN, _INT16_MAX) if dtype == "int16" else (
            int(np.iinfo(np.int32).min), int(np.iinfo(np.int32).max))
        if int(coords.min()) < lo or int(coords.max()) > hi:
            raise SnapshotWriterError(
                f"coordinate out of range for dtype {dtype}; recreate with int32")

        row = self._n_allocated
        new = row + 1
        for name, value, shape in (
            ("cycle", np.int64(cycle_i), (new,)),
            ("coordinates", coords.astype(dtype), (new, nT, M, N, 3)),
            ("walker_id", w, (new, nT)),
            ("m_intra", mi, (new, nT)),
            ("m_inter", me, (new, nT)),
            ("largest_cluster_size", lcs, (new, nT)),
            ("mean_chain_rg2", rg2, (new, nT)),
        ):
            ds = self._snap[name]
            ds.resize(shape)
            ds[row] = value
        self._n_allocated = new
        self._f.flush()
        self._n_written = new
        self._last_cycle = cycle_i
        self._snap.attrs["committed_rows"] = int(self._n_written)
        self._snap.attrs["n_snapshots"] = int(self._n_written)
        self._f.flush()

    @property
    def n_snapshots(self) -> int:
        return self._n_written

    def update_metadata(self, extra: dict) -> None:
        if self._f is None:
            raise SnapshotWriterError("writer is closed")
        meta = self._f["metadata"]
        for key, value in extra.items():
            if isinstance(value, (int, np.integer)):
                meta.attrs[key] = int(value)
            elif isinstance(value, (float, np.floating)):
                meta.attrs[key] = float(value)
            else:
                meta.attrs[key] = str(value)
        self._f.flush()

    def mark_complete(self) -> None:
        self._completed = True

    def close(self) -> None:
        if self._f is not None:
            status = STATUS_COMPLETE if self._completed else STATUS_INTERRUPTED
            self._snap.attrs["committed_rows"] = int(self._n_written)
            self._snap.attrs["n_snapshots"] = int(self._n_written)
            self._snap.attrs["status"] = status
            self._f.attrs["status"] = status
            self._f.flush()
            self._f.close()
            self._f = None

    def __enter__(self) -> "MultiChainSnapshotWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Canonical text/array outputs
# ---------------------------------------------------------------------------

RESULTS_CSV_COLUMNS = [
    "T", "u_mean", "u_std", "effective_energy_mean",
    "m_intra_mean", "m_intra_std", "m_inter_mean", "m_inter_std", "m_total_mean",
    # Intensive per-chain contact normalizations (analysis only; raw columns
    # above remain authoritative).  "pairs" = globally-unique interchain pairs / M;
    # "incidences" = 2 * pairs / M (contact endpoints per chain).
    "m_intra_per_chain_mean", "m_intra_per_chain_std",
    "m_inter_pairs_per_chain_mean", "m_inter_pairs_per_chain_std",
    "m_inter_incidences_per_chain_mean", "m_inter_incidences_per_chain_std",
    "m_total_pairs_per_chain_mean",
    "Rg_mean", "Rg_std", "Rg_mean_lattice", "Rg_std_lattice",
    "Rg2_mean", "Rg2_mean_lattice",
    "f_inter_mean",
    "largest_cluster_size_mean", "largest_cluster_fraction_mean",
    # Per-chain Rg heterogeneity and chain-cluster count (chain-collapse vs
    # mixed expanded/collapsed discrimination).  std_chain_rg_mean is a length
    # (rg_scale applied); std_chain_rg_mean_lattice is the raw-lattice value.
    "std_chain_rg_mean", "std_chain_rg_mean_lattice", "n_clusters_mean",
    "local_acceptance_rate", "translation_acceptance_rate",
]


def save_results_csv(results: list, out_prefix: str) -> str:
    path = f"{out_prefix}_results.csv"
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(RESULTS_CSV_COLUMNS)
        for r in results:
            w.writerow([r.get(k, math.nan) for k in RESULTS_CSV_COLUMNS])
    print(f"Saved {path}")
    return path


def save_swap_csv(swap_props, swap_accs, Ts, out_prefix: str) -> str:
    path = f"{out_prefix}_swap_rates.csv"
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pair", "T_lo", "T_hi", "proposals", "acceptances", "rate"])
        for k in range(len(swap_props)):
            prop = int(swap_props[k])
            acc = int(swap_accs[k])
            rate = acc / prop if prop > 0 else float("nan")
            w.writerow([k, float(Ts[k]), float(Ts[k + 1]), prop, acc, f"{rate:.4f}"])
    print(f"Saved {path}")
    return path


MOVE_ACCEPTANCE_CSV_COLUMNS = [
    "temperature_index", "temperature", "move_type",
    "proposed", "geometrically_valid", "state_changing", "metropolis_accepted",
    "a_geometry", "a_state_changing", "a_metropolis", "a_total",
]


def save_move_acceptance_csv(move_counters, move_names, Ts, out_prefix: str) -> str:
    """``move_counters`` is a list (per lane) of (n_moves, 4) integer arrays."""
    path = f"{out_prefix}_move_acceptance.csv"
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(MOVE_ACCEPTANCE_CSV_COLUMNS)
        for i, counters in enumerate(move_counters):
            counters = np.asarray(counters, dtype=np.int64)
            for mi, mname in enumerate(move_names):
                prop = int(counters[mi, 0])
                valid = int(counters[mi, 1])
                sc = int(counters[mi, 2])
                acc = int(counters[mi, 3])
                a_geom = (valid / prop) if prop else float("nan")
                a_sc = (sc / prop) if prop else float("nan")
                a_met = (acc / sc) if sc else float("nan")
                a_tot = (acc / prop) if prop else float("nan")
                w.writerow([i, float(Ts[i]), mname, prop, valid, sc, acc,
                            f"{a_geom:.6f}", f"{a_sc:.6f}", f"{a_met:.6f}",
                            f"{a_tot:.6f}"])
    print(f"Saved {path}")
    return path


def save_distributions_npz(dist: dict, out_prefix: str) -> str:
    path = f"{out_prefix}_distributions.npz"
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **dist)
    print(f"Saved {path}")
    return path


def save_run_summary(summary: dict, path: str) -> str:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(_json_safe(summary), fh, indent=2, allow_nan=False)
    print(f"Saved {path}")
    return path
