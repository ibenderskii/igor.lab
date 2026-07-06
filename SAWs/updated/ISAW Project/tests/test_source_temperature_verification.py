"""Phase 3: independent source-configuration verification (tri-state status).

A COHERENT feature-only rewrite (manifest + rows changed together) must fail
whenever the independent source configuration is available, and strict
``require_source`` validation must fail when the source is missing or mismatched.
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


def _make(tmp, temps=(300.0, 350.0)):
    nb, nT = len(HAIRPIN6), 2
    cp, _ = ico.build_contact_map(HAIRPIN6)
    m = cp.shape[0]
    rg2 = ico.radius_of_gyration_squared(HAIRPIN6)
    ree2 = ico.end_to_end_distance_squared(HAIRPIN6)
    meta = {"run_id": "sv", "seed": 3, "temperatures": list(temps)}
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
    return inp, out


def _rewrite_temperature(feat, new_temps):
    """Coherently rewrite the temperature column + manifest ladder together."""
    import h5py
    with h5py.File(feat, "r+") as f:
        tcol = f["features/sample_index/temperature"][()]
        ti = f["features/sample_index/temperature_index"][()].astype(int)
        newcol = np.array([new_temps[int(k)] for k in ti], dtype=float)
        f["features/sample_index/temperature"][...] = newcol
        man = json.loads(f["metadata"].attrs["manifest"])
        man["temperatures"] = list(new_temps)
        f["metadata"].attrs["manifest"] = json.dumps(man, default=str)


def test_clean_file_is_fully_source_verified():
    with tempfile.TemporaryDirectory() as tmp:
        _, out = _make(tmp)
        info = ext.validate_feature_file_hdf5(out, deep=True, require_source=True)
        assert info["source_verification_status"] == "fully_source_verified"


def test_coherent_temperature_rewrite_fails_against_source():
    with tempfile.TemporaryDirectory() as tmp:
        _, out = _make(tmp, temps=(300.0, 350.0))
        # Shift both the rows AND the manifest coherently; the source still says
        # 300/350, so verification must fail.
        _rewrite_temperature(out, [305.0, 355.0])
        with pytest.raises(ext.ExtractionError):
            ext.validate_feature_file_hdf5(out, deep=True, require_source=True)
        with pytest.raises(ext.ExtractionError):
            ext.validate_feature_file_hdf5(out, deep=True)  # source present => fail


def test_source_hash_mismatch_fails():
    with tempfile.TemporaryDirectory() as tmp:
        src, out = _make(tmp)
        # Mutate the source file's bytes so its hash no longer matches the manifest.
        import h5py
        with h5py.File(src, "r+") as f:
            f["metadata"].attrs["run_id"] = "sv"  # touch attrs -> different bytes
            f.attrs["_tamper"] = 1
        with pytest.raises(ext.ExtractionError):
            ext.validate_feature_file_hdf5(out, deep=True, require_source=True)


def test_missing_source_is_failure_in_strict_mode():
    with tempfile.TemporaryDirectory() as tmp:
        src, out = _make(tmp)
        os.remove(src)
        # strict: missing source is a hard failure
        with pytest.raises(ext.ExtractionError):
            ext.validate_feature_file_hdf5(out, deep=True, require_source=True)
        # weak: allowed, but only internally_consistent_only
        info = ext.validate_feature_file_hdf5(out, deep=True)
        assert info["source_verification_status"] == "internally_consistent_only"


def test_coherent_model_param_rewrite_fails_against_source():
    with tempfile.TemporaryDirectory() as tmp:
        src, out = _make(tmp)
        import h5py
        with h5py.File(out, "r+") as f:
            man = json.loads(f["metadata"].attrs["manifest"])
            man["model_record"]["model_params"] = [999.9, 9.99]
            f["metadata"].attrs["manifest"] = json.dumps(man, default=str)
        with pytest.raises(ext.ExtractionError):
            ext.validate_feature_file_hdf5(out, deep=True, require_source=True)
