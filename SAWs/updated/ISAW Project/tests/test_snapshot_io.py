"""Hardened HDF5 snapshot-writer tests (Phase 5)."""
import os
import tempfile

import numpy as np
import pytest

import isaw_config_io as cio

pytestmark = pytest.mark.skipif(not cio.h5py_available(), reason="h5py missing")

NB, NT = 6, 2


def _lane(shift=0):
    return np.array([(i + shift, 0, 0) for i in range(NB)], dtype=np.int64)


def _coords():
    return np.stack([_lane(), _lane()])


def _writer(tmp, **kw):
    path = os.path.join(tmp, "c.h5")
    return path, cio.SnapshotWriter(
        path, n_beads=NB, n_temperatures=NT,
        metadata={"run_id": "t", "seed": 1, "temperatures": [300.0, 320.0]},
        **kw)


def _ok_append(w, cycle):
    w.append(cycle=cycle, coordinates=_coords(),
             walker_id=np.array([0, 1] if cycle % 2 == 0 else [1, 0]),
             contacts=np.zeros(NT), rg2_lattice=np.ones(NT),
             ree2_lattice=np.ones(NT))


def test_invalid_dtype_name_fails():
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(cio.SnapshotWriterError):
            cio.SnapshotWriter(os.path.join(tmp, "x.h5"), n_beads=NB,
                               n_temperatures=NT, metadata={},
                               coord_dtype="float32")


def test_float_coordinates_fail():
    with tempfile.TemporaryDirectory() as tmp:
        _, w = _writer(tmp)
        coords = _coords().astype(float)
        coords[0, 0, 0] = 0.5
        with pytest.raises(cio.SnapshotWriterError):
            w.append(cycle=0, coordinates=coords, walker_id=np.array([0, 1]),
                     contacts=np.zeros(NT), rg2_lattice=np.ones(NT),
                     ree2_lattice=np.ones(NT))
        w.close()


def test_wrong_scalar_shape_fails_before_resize():
    with tempfile.TemporaryDirectory() as tmp:
        _, w = _writer(tmp)
        with pytest.raises(cio.SnapshotWriterError):
            w.append(cycle=0, coordinates=_coords(),
                     walker_id=np.array([0, 1, 2]),  # wrong length
                     contacts=np.zeros(NT), rg2_lattice=np.ones(NT),
                     ree2_lattice=np.ones(NT))
        assert w.n_snapshots == 0
        w.close()


def test_invalid_walker_permutation_fails():
    with tempfile.TemporaryDirectory() as tmp:
        _, w = _writer(tmp)
        with pytest.raises(cio.SnapshotWriterError):
            w.append(cycle=0, coordinates=_coords(),
                     walker_id=np.array([0, 0]),
                     contacts=np.zeros(NT), rg2_lattice=np.ones(NT),
                     ree2_lattice=np.ones(NT))
        w.close()


def test_nonmonotonic_cycles_fail():
    with tempfile.TemporaryDirectory() as tmp:
        _, w = _writer(tmp)
        _ok_append(w, 5)
        with pytest.raises(cio.SnapshotWriterError):
            _ok_append(w, 5)
        w.close()


def test_overflow_fails():
    with tempfile.TemporaryDirectory() as tmp:
        _, w = _writer(tmp, coord_dtype="int16")
        big = _coords()
        big[0, 0, 0] = 40000  # > int16 max
        with pytest.raises(cio.SnapshotWriterError):
            w.append(cycle=0, coordinates=big, walker_id=np.array([0, 1]),
                     contacts=np.zeros(NT), rg2_lattice=np.ones(NT),
                     ree2_lattice=np.ones(NT))
        w.close()


def test_committed_rows_and_status_transitions():
    import h5py
    with tempfile.TemporaryDirectory() as tmp:
        path, w = _writer(tmp)
        for c in range(3):
            _ok_append(w, c)
        # before close, the file is 'running' with committed_rows == 3
        assert w.n_snapshots == 3
        w.mark_complete()
        w.close()
        with h5py.File(path, "r") as f:
            s = f["snapshots"]
            assert int(s.attrs["committed_rows"]) == 3
            assert str(s.attrs["status"]) == cio.STATUS_COMPLETE
            assert cio.committed_rows(s) == 3


def test_interrupted_status_when_not_marked():
    import h5py
    with tempfile.TemporaryDirectory() as tmp:
        path, w = _writer(tmp)
        _ok_append(w, 0)
        w.close()  # never marked complete
        with h5py.File(path, "r") as f:
            assert str(f["snapshots"].attrs["status"]) == cio.STATUS_INTERRUPTED


def test_failed_append_does_not_commit_row():
    import h5py
    with tempfile.TemporaryDirectory() as tmp:
        path, w = _writer(tmp)
        _ok_append(w, 0)
        with pytest.raises(cio.SnapshotWriterError):
            w.append(cycle=1, coordinates=_coords(), walker_id=np.array([0, 0]),
                     contacts=np.zeros(NT), rg2_lattice=np.ones(NT),
                     ree2_lattice=np.ones(NT))
        w.mark_complete()
        w.close()
        with h5py.File(path, "r") as f:
            s = f["snapshots"]
            assert int(s.attrs["committed_rows"]) == 1


def test_commit_marker_survives_reopen():
    import h5py
    with tempfile.TemporaryDirectory() as tmp:
        path, w = _writer(tmp)
        for c in range(4):
            _ok_append(w, c)
            # After every append the committed count is durable on disk: open a
            # SECOND read-only handle and confirm the marker is already there.
            with h5py.File(path, "r") as rf:
                assert cio.committed_rows(rf["snapshots"]) == c + 1
        w.mark_complete()
        w.close()


def test_refuses_overwrite():
    with tempfile.TemporaryDirectory() as tmp:
        path, w = _writer(tmp)
        _ok_append(w, 0)
        w.close()
        with pytest.raises(cio.SnapshotWriterError):
            cio.SnapshotWriter(path, n_beads=NB, n_temperatures=NT, metadata={})
