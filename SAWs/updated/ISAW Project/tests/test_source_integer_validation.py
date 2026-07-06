"""Phase 1/2: exact integer validation of RAW source snapshot integers.

The extractor must reject a source configuration file whose integer datasets or
integer metadata are silently fractional / NaN / inf / out-of-int64 BEFORE any
feature row is written -- never truncated by ``int(4.5) == 4``.
"""
import os
import shutil
import tempfile

import numpy as np
import pytest

import isaw_contact_observables as ico
import isaw_config_io as cio
import extract_contact_motif_features as ext

pytestmark = pytest.mark.skipif(not cio.h5py_available(), reason="h5py missing")

HAIRPIN6 = [(0, 0, 0), (1, 0, 0), (2, 0, 0), (2, 1, 0), (1, 1, 0), (0, 1, 0)]
MODEL_META = {"model_name": "hs", "param_names": ["h", "s"],
              "model_params": [647.7, 1.874], "Tref": 320.0, "Tscale": 80.0}


def _make_source(tmp):
    nb, nT = len(HAIRPIN6), 2
    cp, _ = ico.build_contact_map(HAIRPIN6)
    m = cp.shape[0]
    rg2 = ico.radius_of_gyration_squared(HAIRPIN6)
    ree2 = ico.end_to_end_distance_squared(HAIRPIN6)
    meta = {"run_id": "src", "seed": 7, "temperatures": [300.0, 350.0]}
    meta.update(MODEL_META)
    inp = os.path.join(tmp, "snap.h5")
    w = cio.SnapshotWriter(inp, n_beads=nb, n_temperatures=nT, metadata=meta)
    for c in range(3):
        w.append(cycle=c,
                 coordinates=np.stack([np.asarray(HAIRPIN6, dtype=np.int64)] * nT),
                 walker_id=np.array([0, 1] if c % 2 == 0 else [1, 0]),
                 contacts=np.full(nT, m), rg2_lattice=np.full(nT, rg2),
                 ree2_lattice=np.full(nT, ree2))
    w.mark_complete()
    w.close()
    return inp


def _poison_dataset(src, dset, idx, value):
    dst = src + ".bad.h5"
    shutil.copy(src, dst)
    import h5py
    with h5py.File(dst, "r+") as f:
        arr = f[dset][()].astype(np.float64)
        arr[idx] = value
        attrs = dict(f[dset].attrs)
        del f[dset]
        d = f.create_dataset(dset, data=arr)
        for k, v in attrs.items():
            d.attrs[k] = v
    return dst


def _poison_attr(src, group, attr, value):
    dst = src + ".bad.h5"
    shutil.copy(src, dst)
    import h5py
    with h5py.File(dst, "r+") as f:
        f[group].attrs[attr] = value
    return dst


# --- scalar helper unit tests -----------------------------------------------

@pytest.mark.parametrize("bad", [4.5, 96.5, float("nan"), float("inf"), 2.0 ** 63])
def test_scalar_helper_rejects_bad(bad):
    with pytest.raises(ext.ExtractionError):
        ext.require_exact_integer_scalar(bad, field_name="x")


def test_scalar_helper_rejects_multielement():
    with pytest.raises(ext.ExtractionError):
        ext.require_exact_integer_scalar(np.array([1, 2]), field_name="x")


def test_scalar_helper_accepts_integer():
    assert ext.require_exact_integer_scalar(4.0, field_name="x") == 4
    assert ext.require_exact_integer_scalar(np.int64(9), field_name="x") == 9


# --- source DATASET corruptions (rejected before feature rows written) ------

DATASETS = {
    "fractional_cycle": ("snapshots/cycle", 0, 0.25),
    "fractional_walker_id": ("snapshots/walker_id", (0, 0), 0.5),
    "fractional_contacts": ("snapshots/contacts", (0, 0), 1.5),
    "nan_walker_id": ("snapshots/walker_id", (0, 0), float("nan")),
    "inf_cycle": ("snapshots/cycle", 0, float("inf")),
    "oversize_contacts": ("snapshots/contacts", (0, 0), 2.0 ** 63),
}


@pytest.mark.parametrize("name", list(DATASETS))
def test_source_dataset_integer_rejected(name):
    with tempfile.TemporaryDirectory() as tmp:
        src = _make_source(tmp)
        dset, idx, value = DATASETS[name]
        bad = _poison_dataset(src, dset, idx, value)
        out = os.path.join(tmp, "feat.h5")
        with pytest.raises(ext.ExtractionError):
            ext.extract(bad, out, validate=True, output_format="hdf5",
                        overwrite=True)
        # No completed feature file may be left at the output path.
        if os.path.exists(out):
            import h5py
            with h5py.File(out, "r") as f:
                assert str(f.attrs.get("status")) != "complete"


# --- source METADATA / attribute corruptions --------------------------------

def test_fractional_seed_metadata_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        src = _make_source(tmp)
        bad = _poison_attr(src, "metadata", "seed", 7.5)
        with pytest.raises(ext.ExtractionError):
            ext.extract(bad, os.path.join(tmp, "f.h5"), validate=True,
                        overwrite=True)


def test_fractional_schema_version_metadata_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        src = _make_source(tmp)
        bad = _poison_attr(src, "metadata", "schema_version", 3.5)
        with pytest.raises(ext.ExtractionError):
            ext.extract(bad, os.path.join(tmp, "f.h5"), validate=True,
                        overwrite=True)


def test_fractional_committed_rows_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        src = _make_source(tmp)
        bad = _poison_attr(src, "snapshots", "committed_rows", 2.5)
        with pytest.raises(ext.ExtractionError):
            ext.extract(bad, os.path.join(tmp, "f.h5"), validate=True,
                        overwrite=True)


def test_fractional_n_beads_metadata_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        src = _make_source(tmp)
        bad = _poison_attr(src, "metadata", "n_beads", 6.5)
        with pytest.raises(ext.ExtractionError):
            ext.extract(bad, os.path.join(tmp, "f.h5"), validate=True,
                        overwrite=True)


def test_clean_source_extracts():
    with tempfile.TemporaryDirectory() as tmp:
        src = _make_source(tmp)
        out = os.path.join(tmp, "feat.h5")
        info = ext.extract(src, out, validate=True, output_format="hdf5",
                           overwrite=True)
        assert info["row_count"] == 6
        assert all(v == 0 for v in info["validation_discrepancy_counts"].values())
