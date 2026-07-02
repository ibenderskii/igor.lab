"""Phase 1/14: exact integer validation of stored feature datasets.

Every integer dataset that is silently recreated as floating point with a
fractional / NaN / inf / out-of-int64 value must be REJECTED before any integer
cast (never truncated).
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


def _make_feature_file(tmp):
    nb, nT = len(HAIRPIN6), 2
    cp, _ = ico.build_contact_map(HAIRPIN6)
    m = cp.shape[0]
    rg2 = ico.radius_of_gyration_squared(HAIRPIN6)
    ree2 = ico.end_to_end_distance_squared(HAIRPIN6)
    meta = {"run_id": "cf", "seed": 1, "temperatures": [300.0, 350.0]}
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
    out = os.path.join(tmp, "feat.h5")
    ext.extract(inp, out, validate=True, output_format="hdf5", overwrite=True)
    return out


def _recreate_float(path, dset, idx, value):
    """Rewrite an integer dataset as float64 with one poisoned element."""
    dst = path + ".bad.h5"
    shutil.copy(path, dst)
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


# --- helper unit tests ------------------------------------------------------

@pytest.mark.parametrize("bad", [0.25, 1.5, float("nan"), float("inf"), 2.0 ** 63])
def test_helper_rejects_bad_float(bad):
    with pytest.raises(ext.ExtractionError):
        ext.require_exact_integer_array(np.array([1.0, bad, 3.0]),
                                        field_name="x")


def test_helper_rejects_complex_and_object():
    with pytest.raises(ext.ExtractionError):
        ext.require_exact_integer_array(np.array([1 + 2j]), field_name="x")
    with pytest.raises(ext.ExtractionError):
        ext.require_exact_integer_array(np.array([1, [2, 3]], dtype=object),
                                        field_name="x")


def test_helper_accepts_exact_integers():
    out = ext.require_exact_integer_array(np.array([1.0, 2.0, 3.0]),
                                          field_name="x")
    assert out.dtype == np.int64 and out.tolist() == [1, 2, 3]


# --- adversarial file corruptions -------------------------------------------

FRACTIONAL = {
    "m_scalar": ("features/scalars/m", 0, 0.25),
    "m_r": ("features/m_r", (0, 3), 1.5),
    "pair_index": ("features/contact_pairs/pairs", (0, 1), 2.5),
    "offset": ("features/contact_pairs/offsets", 1, 0.5),
    "temperature_index": ("features/sample_index/temperature_index", 0, 0.5),
    "walker_id": ("features/sample_index/walker_id", 0, 1.5),
    "cycle": ("features/sample_index/cycle", 0, 0.25),
}
NONFINITE = {
    "m_nan": ("features/scalars/m", 0, float("nan")),
    "m_inf": ("features/scalars/m", 0, float("inf")),
    "walker_inf": ("features/sample_index/walker_id", 0, float("inf")),
}
OUT_OF_RANGE = {
    "m_2p63": ("features/scalars/m", 0, 2.0 ** 63),
}


@pytest.mark.parametrize("name", list(FRACTIONAL))
def test_fractional_integer_field_rejected(name):
    with tempfile.TemporaryDirectory() as tmp:
        out = _make_feature_file(tmp)
        dset, idx, value = FRACTIONAL[name]
        bad = _recreate_float(out, dset, idx, value)
        with pytest.raises(ext.ExtractionError):
            ext.validate_feature_file_hdf5(bad, deep=True)


@pytest.mark.parametrize("name", list(NONFINITE))
def test_nonfinite_integer_field_rejected(name):
    with tempfile.TemporaryDirectory() as tmp:
        out = _make_feature_file(tmp)
        dset, idx, value = NONFINITE[name]
        bad = _recreate_float(out, dset, idx, value)
        with pytest.raises(ext.ExtractionError):
            ext.validate_feature_file_hdf5(bad, deep=True)


@pytest.mark.parametrize("name", list(OUT_OF_RANGE))
def test_out_of_int64_range_rejected(name):
    with tempfile.TemporaryDirectory() as tmp:
        out = _make_feature_file(tmp)
        dset, idx, value = OUT_OF_RANGE[name]
        bad = _recreate_float(out, dset, idx, value)
        with pytest.raises(ext.ExtractionError):
            ext.validate_feature_file_hdf5(bad, deep=True)


def test_clean_file_still_validates():
    with tempfile.TemporaryDirectory() as tmp:
        out = _make_feature_file(tmp)
        info = ext.validate_feature_file_hdf5(out, deep=True)
        assert info["ok"] is True
