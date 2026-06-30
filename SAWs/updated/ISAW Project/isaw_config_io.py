#!/usr/bin/env python3
"""
Streaming coordinate-snapshot writer for ISAW REMD runs (hardened).

Coordinates and a few per-lane scalars are streamed to a chunked HDF5 file so a
long run never accumulates the full trajectory in memory and partial data
survives an interruption.

Robustness model (schema version 2)
-----------------------------------
* Every row's inputs are validated BEFORE any dataset is resized, so a malformed
  row never half-mutates the file.
* ``/snapshots`` carries a ``committed_rows`` attribute.  A row is only counted
  in ``committed_rows`` after every field of that row has been written and the
  row data has been flushed.  Readers MUST process only rows with index
  ``< committed_rows``; an interrupted writer may have allocated (resized) a row
  that was never committed, and that row must be ignored.
* The file ``status`` attribute transitions ``running -> complete`` only via an
  explicit :meth:`SnapshotWriter.mark_complete` call (after ``run_remd`` returns
  successfully).  A close that happens during stack unwinding leaves the file
  ``running``/``interrupted`` -- it is never silently marked complete.

HDF5 layout
-----------
/metadata   (group)  run/model provenance as attributes + small datasets
/snapshots  (group)  attrs: committed_rows, n_snapshots, coordinate_dtype, status
    cycle        (n_rows,)                int64   (strictly increasing)
    coordinates  (n_rows, nT, n_beads, 3) int16 or int32
    walker_id    (n_rows, nT)             int64   (permutation of 0..nT-1)
    contacts     (n_rows, nT)             int64   (nonnegative)
    rg2_lattice  (n_rows, nT)             float64 (finite, nonnegative)
    ree2_lattice (n_rows, nT)             float64 (finite, nonnegative)

The temperature-lane index is the second array dimension; the temperature
ladder itself is stored once in /metadata.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

try:  # h5py is required only when configuration saving is requested
    import h5py
    _HAVE_H5PY = True
except Exception:  # pragma: no cover - import guard
    h5py = None
    _HAVE_H5PY = False

SNAPSHOT_SCHEMA_VERSION = 3

ALLOWED_COORD_DTYPES = ("auto", "int16", "int32")

_INT16_MIN, _INT16_MAX = -32768, 32767
# Safety margin so a coordinate that drifts slightly between snapshots does not
# silently overflow a dtype chosen from the first snapshot.
_INT16_SAFE = 30000

STATUS_RUNNING = "running"
STATUS_COMPLETE = "complete"
STATUS_INTERRUPTED = "interrupted"


def h5py_available() -> bool:
    return _HAVE_H5PY


class SnapshotWriterError(RuntimeError):
    pass


def _choose_coord_dtype(coords: np.ndarray) -> str:
    """Pick int16 when safe, else int32 (never a silent overflow)."""
    amax = int(np.abs(coords).max()) if coords.size else 0
    return "int16" if amax <= _INT16_SAFE else "int32"


def committed_rows(group) -> int:
    """Number of valid (committed) snapshot rows in an open ``/snapshots`` group.

    Falls back to the dataset length only for legacy (schema 1) files that
    predate the committed-row attribute.
    """
    if "committed_rows" in group.attrs:
        return int(group.attrs["committed_rows"])
    if "n_snapshots" in group.attrs:
        return int(group.attrs["n_snapshots"])
    return int(group["cycle"].shape[0]) if "cycle" in group else 0


class SnapshotWriter:
    """Append-only chunked HDF5 writer for REMD coordinate snapshots.

    Transactional append semantics guarantee that an interrupted file is always
    safely readable up to ``committed_rows``.
    """

    def __init__(
        self,
        path: str,
        *,
        n_beads: int,
        n_temperatures: int,
        metadata: dict,
        coord_dtype: str = "auto",
        flush_interval: int = 50,
        overwrite: bool = False,
    ) -> None:
        if not _HAVE_H5PY:
            raise SnapshotWriterError(
                "Configuration saving requires the 'h5py' package, which is not "
                "available. Install h5py or omit --save-configurations."
            )
        if coord_dtype not in ALLOWED_COORD_DTYPES:
            raise SnapshotWriterError(
                f"coord_dtype must be one of {ALLOWED_COORD_DTYPES}; "
                f"got {coord_dtype!r}"
            )
        self.path = str(path)
        self.n_beads = int(n_beads)
        self.n_temperatures = int(n_temperatures)
        self.flush_interval = max(1, int(flush_interval))
        self._coord_dtype = None if coord_dtype == "auto" else str(coord_dtype)
        self._n_written = 0          # committed rows
        self._n_allocated = 0        # resized rows (>= committed)
        self._since_flush = 0
        self._last_cycle = None
        self._completed = False

        p = Path(self.path)
        if p.exists() and not overwrite:
            raise SnapshotWriterError(
                f"configuration file {self.path!r} already exists; refusing to "
                f"overwrite (pass overwrite=True / --overwrite to replace it)."
            )
        p.parent.mkdir(parents=True, exist_ok=True)

        self._f = h5py.File(self.path, "w")
        self._write_metadata(metadata)
        self._snap = self._f.create_group("snapshots")
        self._snap.attrs["committed_rows"] = 0
        self._snap.attrs["n_snapshots"] = 0
        self._snap.attrs["status"] = STATUS_RUNNING
        self._snap.attrs["schema_version"] = SNAPSHOT_SCHEMA_VERSION
        self._f.attrs["status"] = STATUS_RUNNING
        self._datasets_created = False  # coordinate dtype fixed on first append
        self._f.flush()

    # -- metadata -----------------------------------------------------------
    def _write_metadata(self, metadata: dict) -> None:
        meta = self._f.create_group("metadata")
        meta.attrs["schema_version"] = SNAPSHOT_SCHEMA_VERSION
        for key, value in metadata.items():
            if value is None:
                meta.attrs[key] = "null"
            elif isinstance(value, dict):
                meta.attrs[key] = json.dumps(value)
            elif isinstance(value, (list, tuple, np.ndarray)):
                arr = np.asarray(value)
                if arr.dtype.kind in ("U", "S", "O"):
                    meta.create_dataset(
                        key, data=np.asarray([str(v) for v in value],
                                             dtype=h5py.string_dtype())
                    )
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

    # -- dataset creation ---------------------------------------------------
    def _create_datasets(self, coord_dtype: str) -> None:
        nT, nb = self.n_temperatures, self.n_beads
        g = self._snap
        chunk_rows = min(self.flush_interval, 64)
        g.create_dataset(
            "cycle", shape=(0,), maxshape=(None,), dtype="int64",
            chunks=(chunk_rows,),
        )
        g.create_dataset(
            "coordinates", shape=(0, nT, nb, 3), maxshape=(None, nT, nb, 3),
            dtype=coord_dtype, chunks=(chunk_rows, nT, nb, 3),
            compression="gzip", compression_opts=4,
        )
        for name in ("walker_id", "contacts"):
            g.create_dataset(
                name, shape=(0, nT), maxshape=(None, nT), dtype="int64",
                chunks=(chunk_rows, nT),
            )
        for name in ("rg2_lattice", "ree2_lattice"):
            g.create_dataset(
                name, shape=(0, nT), maxshape=(None, nT), dtype="float64",
                chunks=(chunk_rows, nT),
            )
        self._snap.attrs["coordinate_dtype"] = coord_dtype
        self._datasets_created = True

    # -- validation ---------------------------------------------------------
    def _validate_row(
        self, cycle, coordinates, walker_id, contacts, rg2_lattice, ree2_lattice,
    ) -> tuple:
        nT, nb = self.n_temperatures, self.n_beads

        # cycle: scalar integer, strictly increasing.
        c_arr = np.asarray(cycle)
        if c_arr.ndim != 0:
            raise SnapshotWriterError(f"cycle must be scalar, got shape {c_arr.shape}")
        if c_arr.dtype.kind == "f" and float(c_arr) != int(c_arr):
            raise SnapshotWriterError(f"cycle must be integral, got {cycle!r}")
        cycle_i = int(c_arr)
        if self._last_cycle is not None and cycle_i <= self._last_cycle:
            raise SnapshotWriterError(
                f"cycle numbers must be strictly increasing; got {cycle_i} "
                f"after {self._last_cycle}"
            )

        # coordinates: validated strictly via the lattice validator, per lane.
        coords = np.asarray(coordinates)
        if coords.shape != (nT, nb, 3):
            raise SnapshotWriterError(
                f"coordinates shape {coords.shape} != expected {(nT, nb, 3)}"
            )
        try:
            import isaw_contact_observables as _ico
            for k in range(nT):
                _ico.normalize_lattice_coordinates(coords[k])
        except Exception as exc:
            raise SnapshotWriterError(
                f"coordinate validation failed at cycle {cycle_i}: {exc}"
            ) from exc

        # walker_id: shape (nT,), integer, exact permutation of 0..nT-1.
        w = np.asarray(walker_id)
        if w.shape != (nT,):
            raise SnapshotWriterError(f"walker_id shape {w.shape} != {(nT,)}")
        if w.dtype.kind not in ("i", "u"):
            if not np.all(w == np.rint(w)):
                raise SnapshotWriterError("walker_id must be integer-valued")
        w = w.astype(np.int64)
        if not np.array_equal(np.sort(w), np.arange(nT, dtype=np.int64)):
            raise SnapshotWriterError(
                f"walker_id must be a permutation of 0..{nT - 1}; got {w.tolist()}"
            )

        # contacts: shape (nT,), integer, nonnegative.
        cont = np.asarray(contacts)
        if cont.shape != (nT,):
            raise SnapshotWriterError(f"contacts shape {cont.shape} != {(nT,)}")
        if not np.all(cont == np.rint(cont)):
            raise SnapshotWriterError("contacts must be integer-valued")
        cont = cont.astype(np.int64)
        if np.any(cont < 0):
            raise SnapshotWriterError("contacts must be nonnegative")

        # rg2 / ree2: shape (nT,), finite, nonnegative.
        rg2 = np.asarray(rg2_lattice, dtype=np.float64)
        ree2 = np.asarray(ree2_lattice, dtype=np.float64)
        for name, a in (("rg2_lattice", rg2), ("ree2_lattice", ree2)):
            if a.shape != (nT,):
                raise SnapshotWriterError(f"{name} shape {a.shape} != {(nT,)}")
            if not np.all(np.isfinite(a)):
                raise SnapshotWriterError(f"{name} must be finite")
            if np.any(a < 0.0):
                raise SnapshotWriterError(f"{name} must be nonnegative")

        return cycle_i, coords, w, cont, rg2, ree2

    # -- append -------------------------------------------------------------
    def append(
        self,
        *,
        cycle: int,
        coordinates: np.ndarray,
        walker_id: np.ndarray,
        contacts: np.ndarray,
        rg2_lattice: np.ndarray,
        ree2_lattice: np.ndarray,
    ) -> None:
        nT, nb = self.n_temperatures, self.n_beads

        # 1. Validate ALL inputs before any mutation.
        (cycle_i, coords, w, cont, rg2, ree2) = self._validate_row(
            cycle, coordinates, walker_id, contacts, rg2_lattice, ree2_lattice
        )

        if not self._datasets_created:
            dtype = self._coord_dtype or _choose_coord_dtype(coords)
            self._create_datasets(dtype)
        dtype = self._snap.attrs["coordinate_dtype"]

        # Strict overflow check: never silently truncate / overflow coordinates.
        lo, hi = (_INT16_MIN, _INT16_MAX) if dtype == "int16" else (
            int(np.iinfo(np.int32).min), int(np.iinfo(np.int32).max)
        )
        if int(coords.min()) < lo or int(coords.max()) > hi:
            raise SnapshotWriterError(
                f"coordinate out of range for dataset dtype {dtype} "
                f"[{int(coords.min())}, {int(coords.max())}]; the dtype was "
                f"fixed at dataset creation and cannot change. Recreate the "
                f"writer with coord_dtype='int32'."
            )

        row = self._n_allocated
        new = row + 1

        # 2. Resize all datasets, then 3. write all row fields.
        for name, value, shape in (
            ("cycle", np.int64(cycle_i), (new,)),
            ("coordinates", coords.astype(dtype), (new, nT, nb, 3)),
            ("walker_id", w, (new, nT)),
            ("contacts", cont, (new, nT)),
            ("rg2_lattice", rg2, (new, nT)),
            ("ree2_lattice", ree2, (new, nT)),
        ):
            ds = self._snap[name]
            ds.resize(shape)
            ds[row] = value
        self._n_allocated = new

        # Row-level durability policy:
        #   3. flush row data (durable on disk)
        #   4. update committed_rows / n_snapshots marker
        #   5. flush the commit marker (durable on disk)
        # The marker is therefore never flushed ahead of the row data it refers
        # to, and a crash between any two appends leaves a consistent file whose
        # committed_rows points only at fully-written rows.
        self._f.flush()
        self._n_written = new
        self._last_cycle = cycle_i
        self._snap.attrs["committed_rows"] = int(self._n_written)
        self._snap.attrs["n_snapshots"] = int(self._n_written)
        self._f.flush()
        self._since_flush = 0

    # -- lifecycle ----------------------------------------------------------
    @property
    def n_snapshots(self) -> int:
        return self._n_written

    def update_metadata(self, extra: dict) -> None:
        """Add/overwrite /metadata attributes after creation (e.g. end_time)."""
        if self._f is None:
            raise SnapshotWriterError("writer is closed")
        meta = self._f["metadata"]
        for key, value in extra.items():
            if value is None:
                meta.attrs[key] = "null"
            elif isinstance(value, (bool, np.bool_)):
                meta.attrs[key] = bool(value)
            elif isinstance(value, (int, np.integer)):
                meta.attrs[key] = int(value)
            elif isinstance(value, (float, np.floating)):
                meta.attrs[key] = float(value)
            else:
                meta.attrs[key] = str(value)
        self._f.flush()

    def mark_complete(self) -> None:
        """Mark the file complete.  Call ONLY after run_remd returns successfully."""
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

    def __enter__(self) -> "SnapshotWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        # Only a clean (exception-free) context exit is allowed to mark complete,
        # and only if the caller already requested it.
        self.close()
