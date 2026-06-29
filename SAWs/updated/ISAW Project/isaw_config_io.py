#!/usr/bin/env python3
"""
Streaming coordinate-snapshot writer for ISAW REMD runs.

Coordinates and a few per-lane scalars are streamed to a chunked HDF5 file so a
long run never accumulates the full trajectory in memory and partial data
survives an interruption (the file is flushed periodically).

HDF5 layout
-----------
/metadata   (group)  run/model provenance as attributes + small datasets
/snapshots  (group)
    cycle        (n_snapshots,)                int64
    coordinates  (n_snapshots, nT, n_beads, 3) int16 or int32
    walker_id    (n_snapshots, nT)             int64
    contacts     (n_snapshots, nT)             int64
    rg2_lattice  (n_snapshots, nT)             float64
    ree2_lattice (n_snapshots, nT)             float64

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

SNAPSHOT_SCHEMA_VERSION = 1

_INT16_MIN, _INT16_MAX = -32768, 32767
# Safety margin so a coordinate that drifts slightly between snapshots does not
# silently overflow a dtype chosen from the first snapshot.
_INT16_SAFE = 30000


def h5py_available() -> bool:
    return _HAVE_H5PY


class SnapshotWriterError(RuntimeError):
    pass


def _choose_coord_dtype(coords: np.ndarray) -> str:
    """Pick int16 when safe, else int32 (never a silent overflow)."""
    amax = int(np.abs(coords).max()) if coords.size else 0
    return "int16" if amax <= _INT16_SAFE else "int32"


class SnapshotWriter:
    """Append-only chunked HDF5 writer for REMD coordinate snapshots."""

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
        self.path = str(path)
        self.n_beads = int(n_beads)
        self.n_temperatures = int(n_temperatures)
        self.flush_interval = max(1, int(flush_interval))
        self._coord_dtype = None if coord_dtype == "auto" else str(coord_dtype)
        self._n_written = 0
        self._since_flush = 0

        p = Path(self.path)
        if p.exists() and not overwrite:
            raise SnapshotWriterError(
                f"configuration file {self.path!r} already exists; refusing to "
                f"overwrite (pass overwrite=True / --overwrite to replace it). "
                f"Appending to an existing file is only safe when metadata and "
                f"array dimensions match exactly."
            )
        p.parent.mkdir(parents=True, exist_ok=True)

        self._f = h5py.File(self.path, "w")
        self._write_metadata(metadata)
        self._snap = self._f.create_group("snapshots")
        self._datasets_created = False  # coordinate dtype fixed on first append

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
        coords = np.asarray(coordinates)
        if coords.shape != (nT, nb, 3):
            raise SnapshotWriterError(
                f"coordinates shape {coords.shape} != expected {(nT, nb, 3)}"
            )

        if not self._datasets_created:
            dtype = self._coord_dtype or _choose_coord_dtype(coords)
            self._create_datasets(dtype)
        dtype = self._snap.attrs["coordinate_dtype"]

        # Strict overflow check: never silently truncate coordinates.
        lo, hi = (_INT16_MIN, _INT16_MAX) if dtype == "int16" else (
            np.iinfo(np.int32).min, np.iinfo(np.int32).max
        )
        if coords.min() < lo or coords.max() > hi:
            raise SnapshotWriterError(
                f"coordinate out of range for dtype {dtype} "
                f"[{coords.min()}, {coords.max()}]; recreate the writer with "
                f"coord_dtype='int32'."
            )

        row = self._n_written
        new = row + 1
        for name, value, shape in (
            ("cycle", np.int64(cycle), (new,)),
            ("coordinates", coords.astype(dtype), (new, nT, nb, 3)),
            ("walker_id", np.asarray(walker_id, dtype=np.int64), (new, nT)),
            ("contacts", np.asarray(contacts, dtype=np.int64), (new, nT)),
            ("rg2_lattice", np.asarray(rg2_lattice, dtype=np.float64), (new, nT)),
            ("ree2_lattice", np.asarray(ree2_lattice, dtype=np.float64), (new, nT)),
        ):
            ds = self._snap[name]
            ds.resize(shape)
            ds[row] = value

        self._n_written = new
        self._since_flush += 1
        if self._since_flush >= self.flush_interval:
            self._f.flush()
            self._since_flush = 0

    # -- lifecycle ----------------------------------------------------------
    @property
    def n_snapshots(self) -> int:
        return self._n_written

    def close(self) -> None:
        if self._f is not None:
            self._snap.attrs["n_snapshots"] = int(self._n_written)
            self._f.flush()
            self._f.close()
            self._f = None

    def __enter__(self) -> "SnapshotWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
