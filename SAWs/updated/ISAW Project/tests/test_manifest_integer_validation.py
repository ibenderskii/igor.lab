"""Phase 2: exact integer validation of feature-file schema/manifest metadata.

Fractional schema versions, row counts, temperature counts, and n_beads in the
file attributes or the embedded manifest must be rejected (never ``int(4.5)``).
"""
import json
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
    meta = {"run_id": "mf", "seed": 1, "temperatures": [300.0, 350.0]}
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


def _set_file_attr(path, attr, value):
    dst = path + ".bad.h5"
    shutil.copy(path, dst)
    import h5py
    with h5py.File(dst, "r+") as f:
        f.attrs[attr] = value
    return dst


def _patch_manifest(path, **updates):
    dst = path + ".bad.h5"
    shutil.copy(path, dst)
    import h5py
    with h5py.File(dst, "r+") as f:
        man = json.loads(f["metadata"].attrs["manifest"])
        man.update(updates)
        f["metadata"].attrs["manifest"] = json.dumps(man, default=str)
    return dst


def test_file_feature_schema_version_fractional_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        out = _make_feature_file(tmp)
        bad = _set_file_attr(out, "feature_schema_version", 4.5)
        with pytest.raises(ext.ExtractionError):
            ext.validate_feature_file_hdf5(bad, deep=True)


@pytest.mark.parametrize("field,value", [
    ("feature_schema_version", 4.5),
    ("row_count", 6.5),
    ("temperature_count", 2.5),
    ("n_beads", 6.5),
    ("committed_feature_rows", 6.5),
])
def test_manifest_fractional_field_rejected(field, value):
    with tempfile.TemporaryDirectory() as tmp:
        out = _make_feature_file(tmp)
        bad = _patch_manifest(out, **{field: value})
        with pytest.raises(ext.ExtractionError):
            ext.validate_feature_file_hdf5(bad, deep=True)


@pytest.mark.parametrize("field", ["row_count", "temperature_count"])
def test_manifest_nonfinite_field_rejected(field):
    with tempfile.TemporaryDirectory() as tmp:
        out = _make_feature_file(tmp)
        bad = _patch_manifest(out, **{field: float("inf")})
        with pytest.raises(ext.ExtractionError):
            ext.validate_feature_file_hdf5(bad, deep=True)


def test_clean_file_validates():
    with tempfile.TemporaryDirectory() as tmp:
        out = _make_feature_file(tmp)
        assert ext.validate_feature_file_hdf5(out, deep=True)["ok"] is True
