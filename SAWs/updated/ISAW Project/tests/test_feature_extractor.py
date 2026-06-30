"""Offline feature-extractor tests (Phase 9)."""
import json
import os
import tempfile

import numpy as np
import pytest

import isaw_config_io as cio
import isaw_contact_observables as ico
import extract_contact_motif_features as ext

pytestmark = pytest.mark.skipif(not cio.h5py_available(), reason="h5py missing")

HAIRPIN6 = [(0, 0, 0), (1, 0, 0), (2, 0, 0), (2, 1, 0), (1, 1, 0), (0, 1, 0)]


def _make_snapshot(tmp, *, contacts=None, rg2=None, ree2=None, complete=True,
                   n_rows=2):
    nb, nT = 6, 2
    cp, _ = ico.build_contact_map(HAIRPIN6)
    m = cp.shape[0]
    rg2v = ico.radius_of_gyration_squared(HAIRPIN6) if rg2 is None else rg2
    ree2v = ico.end_to_end_distance_squared(HAIRPIN6) if ree2 is None else ree2
    mv = m if contacts is None else contacts
    path = os.path.join(tmp, "snap.h5")
    w = cio.SnapshotWriter(path, n_beads=nb, n_temperatures=nT,
                           metadata={"run_id": "qt", "seed": 1,
                                     "temperatures": [300.0, 320.0]})
    for c in range(n_rows):
        coords = np.stack([np.asarray(HAIRPIN6, dtype=np.int64)] * nT)
        w.append(cycle=c, coordinates=coords,
                 walker_id=np.array([0, 1] if c % 2 == 0 else [1, 0]),
                 contacts=np.full(nT, mv),
                 rg2_lattice=np.full(nT, rg2v),
                 ree2_lattice=np.full(nT, ree2v))
    if complete:
        w.mark_complete()
    w.close()
    return path


def test_tiny_valid_extracts():
    with tempfile.TemporaryDirectory() as tmp:
        inp = _make_snapshot(tmp)
        out = os.path.join(tmp, "feat.h5")
        info = ext.extract(inp, out, validate=True, output_format="hdf5")
        assert info["row_count"] == 4
        assert all(v == 0 for v in info["validation_discrepancy_counts"].values())
        assert os.path.exists(out + ".manifest.json")


def test_contact_mismatch_fails():
    with tempfile.TemporaryDirectory() as tmp:
        inp = _make_snapshot(tmp, contacts=99)
        with pytest.raises(ext.ExtractionError):
            ext.extract(inp, os.path.join(tmp, "f.h5"), output_format="hdf5")


def test_rg2_mismatch_fails():
    with tempfile.TemporaryDirectory() as tmp:
        inp = _make_snapshot(tmp, rg2=123.456)
        with pytest.raises(ext.ExtractionError):
            ext.extract(inp, os.path.join(tmp, "f.h5"), output_format="hdf5")


def test_ree2_mismatch_fails():
    with tempfile.TemporaryDirectory() as tmp:
        inp = _make_snapshot(tmp, ree2=987.0)
        with pytest.raises(ext.ExtractionError):
            ext.extract(inp, os.path.join(tmp, "f.h5"), output_format="hdf5")


def test_incomplete_rows_ignored():
    import h5py
    with tempfile.TemporaryDirectory() as tmp:
        inp = _make_snapshot(tmp, complete=True, n_rows=2)
        # Forge an allocated-but-uncommitted extra row.
        with h5py.File(inp, "r+") as f:
            s = f["snapshots"]
            n = int(s.attrs["committed_rows"])
            for name in ("cycle", "coordinates", "walker_id", "contacts",
                         "rg2_lattice", "ree2_lattice"):
                ds = s[name]
                ds.resize(n + 1, axis=0)  # uncommitted row of zeros
        out = os.path.join(tmp, "feat.h5")
        info = ext.extract(inp, out, output_format="hdf5")
        # the zero (invalid) row must be ignored -> still 2 snapshots * 2 lanes
        assert info["row_count"] == 4
        assert info["extracted_snapshot_rows"] == 2


def test_interrupted_requires_flag():
    with tempfile.TemporaryDirectory() as tmp:
        inp = _make_snapshot(tmp, complete=False)
        with pytest.raises(ext.ExtractionError):
            ext.extract(inp, os.path.join(tmp, "f.h5"), output_format="hdf5")
        info = ext.extract(inp, os.path.join(tmp, "f.h5"), output_format="hdf5",
                           allow_interrupted=True)
        assert info["row_count"] == 4


def test_deterministic_ordering_and_m_r():
    import h5py
    with tempfile.TemporaryDirectory() as tmp:
        inp = _make_snapshot(tmp)
        out = os.path.join(tmp, "feat.h5")
        ext.extract(inp, out, output_format="hdf5")
        with h5py.File(out, "r") as f:
            idx = f["features/sample_index"]
            snap = idx["snapshot_index"][()]
            tk = idx["temperature_index"][()]
            # rows ordered by (snapshot, lane)
            order = list(zip(snap.tolist(), tk.tolist()))
            assert order == sorted(order)
            m_r = f["features/m_r"][()]
            m = f["features/scalars/m"][()]
            for i in range(m_r.shape[0]):
                assert int(m_r[i].sum()) == int(m[i])
                # compare to direct calculation
                cp, _ = ico.build_contact_map(HAIRPIN6)
                direct = ico.contact_separation_counts(cp, 6)
                assert np.array_equal(m_r[i].astype(np.int64), direct)


def _make_float_coord_snapshot(tmp):
    """Hand-build a status=complete HDF5 with FRACTIONAL float coordinates."""
    import h5py
    nb, nT = 6, 1
    path = os.path.join(tmp, "floatsnap.h5")
    with h5py.File(path, "w") as f:
        m = f.create_group("metadata")
        m.attrs["run_id"] = "fc"
        m.attrs["seed"] = 0
        m.attrs["schema_version"] = 3
        m.create_dataset("temperatures", data=np.array([300.0]))
        s = f.create_group("snapshots")
        coords = np.zeros((1, nT, nb, 3), dtype=np.float64)
        coords[0, 0] = np.array([(i, 0, 0) for i in range(nb)], dtype=np.float64)
        coords[0, 0, 2, 0] = 2.5    # fractional -> must be rejected
        s.create_dataset("coordinates", data=coords)
        s.create_dataset("cycle", data=np.array([0], dtype=np.int64))
        s.create_dataset("walker_id", data=np.zeros((1, nT), dtype=np.int64))
        s.create_dataset("contacts", data=np.zeros((1, nT), dtype=np.int64))
        s.create_dataset("rg2_lattice", data=np.zeros((1, nT)))
        s.create_dataset("ree2_lattice", data=np.zeros((1, nT)))
        s.attrs["committed_rows"] = 1
        s.attrs["status"] = "complete"
    return path


def test_fractional_hdf5_coordinates_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        inp = _make_float_coord_snapshot(tmp)
        with pytest.raises(ico.ContactMapError):
            # No pre-cast can hide the fractional coordinate: strict validation
            # fires inside compute_features_for_config -> build_contact_map.
            ext.extract(inp, os.path.join(tmp, "f.h5"), output_format="hdf5",
                        validate=True)


def test_multi_chunk_streaming_consistent():
    import h5py
    with tempfile.TemporaryDirectory() as tmp:
        inp = _make_snapshot(tmp, n_rows=5)
        out = os.path.join(tmp, "feat.h5")
        info = ext.extract(inp, out, output_format="hdf5", chunk_size=2,
                           validate=True)
        assert info["row_count"] == 10   # 5 snapshots * 2 lanes
        with h5py.File(out, "r") as f:
            assert str(f.attrs["status"]) == "complete"
            assert int(f["features"].attrs["committed_feature_rows"]) == 10
            m = f["features/scalars/m"][()]
            m_r = f["features/m_r"][()]
            assert m_r.shape == (10, 6)
            assert np.array_equal(m_r.sum(axis=1).astype(np.int64),
                                  m.astype(np.int64))
            snap = f["features/sample_index/snapshot_index"][()]
            ti = f["features/sample_index/temperature_index"][()]
            order = list(zip(snap.tolist(), ti.tolist()))
            assert order == sorted(order)
            # all datasets share the row count
            for name in f["features/scalars"]:
                assert f["features/scalars/" + name].shape[0] == 10


def test_parquet_multi_chunk():
    if not ext._HAVE_PYARROW:
        pytest.skip("pyarrow not installed")
    import pyarrow.parquet as pq
    with tempfile.TemporaryDirectory() as tmp:
        inp = _make_snapshot(tmp, n_rows=5)
        out = os.path.join(tmp, "feat.parquet")
        ext.extract(inp, out, output_format="parquet", chunk_size=2)
        t = pq.read_table(out)
        assert t.num_rows == 10
        mr_cols = [c for c in t.column_names if c.startswith("m_r_")]
        d = t.to_pydict()
        for i in range(t.num_rows):
            assert sum(d[c][i] for c in mr_cols) == d["m"][i]


def test_manifest_discrepancies_zero():
    with tempfile.TemporaryDirectory() as tmp:
        inp = _make_snapshot(tmp)
        out = os.path.join(tmp, "feat.h5")
        ext.extract(inp, out, validate=True, output_format="hdf5")
        with open(out + ".manifest.json") as fh:
            man = json.load(fh)
        assert all(v == 0 for v in man["validation_discrepancy_counts"].values())
        assert man["feature_schema_version"] == ext.FEATURE_SCHEMA_VERSION
