"""Phase 2/3/14: authoritative temperature ladder + per-snapshot / run-level
sample-index integrity.

Every corruption of the temperature ladder, walker permutation, cycle grouping,
run_id, seed, or schema version must be REJECTED by the deep validator.
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


def _make_feature_file(tmp, n_snap=4):
    nb, nT = len(HAIRPIN6), 2
    cp, _ = ico.build_contact_map(HAIRPIN6)
    m = cp.shape[0]
    rg2 = ico.radius_of_gyration_squared(HAIRPIN6)
    ree2 = ico.end_to_end_distance_squared(HAIRPIN6)
    meta = {"run_id": "cf", "seed": 7, "temperatures": [300.0, 350.0]}
    meta.update(MODEL_META)
    inp = os.path.join(tmp, "snap.h5")
    w = cio.SnapshotWriter(inp, n_beads=nb, n_temperatures=nT, metadata=meta)
    for c in range(n_snap):
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


def _corrupt(out, fn):
    dst = out + ".bad.h5"
    shutil.copy(out, dst)
    import h5py
    with h5py.File(dst, "r+") as f:
        fn(f)
    return dst


def _set(f, path, idx, value):
    d = f[path]; arr = d[()]; arr[idx] = value; d[...] = arr


def _set_manifest(f, mutate):
    man = json.loads(f["metadata"].attrs["manifest"])
    mutate(man)
    f["metadata"].attrs["manifest"] = json.dumps(man, default=str)


# --- temperature integrity --------------------------------------------------

def _all_temps_nan(f):
    d = f["features/sample_index/temperature"]
    arr = d[()].astype(np.float64); arr[:] = np.nan; d[...] = arr


def _shift_one_lane(f):
    # shift temperature of lane 0 rows only (breaks the authoritative ladder)
    ti = f["features/sample_index/temperature_index"][()]
    d = f["features/sample_index/temperature"]; arr = d[()]
    arr[ti == 0] += 5.0; d[...] = arr


def _shift_coherently(f):
    # shift ALL temperatures and the manifest ladder together -> still breaks,
    # because H = T*u then disagrees per row (thermo identity) OR the model
    # record temperature no longer matches; here we only shift feature temps.
    d = f["features/sample_index/temperature"]; arr = d[()]; arr += 5.0; d[...] = arr


def _permute_index(f):
    d = f["features/sample_index/temperature_index"]
    arr = d[()]
    # swap the two lanes of snapshot 0 -> index no longer matches its temperature
    arr[0], arr[1] = arr[1], arr[0]
    d[...] = arr


TEMP_CORRUPT = {
    "all_temps_nan": _all_temps_nan,
    "one_lane_shifted": _shift_one_lane,
    "all_temps_shifted": _shift_coherently,
    "temperature_index_permuted": _permute_index,
}


@pytest.mark.parametrize("name", list(TEMP_CORRUPT))
def test_temperature_corruption_rejected(name):
    with tempfile.TemporaryDirectory() as tmp:
        out = _make_feature_file(tmp)
        bad = _corrupt(out, TEMP_CORRUPT[name])
        with pytest.raises(ext.ExtractionError):
            ext.validate_feature_file_hdf5(bad, deep=True)


def test_temperature_count_mismatch_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        out = _make_feature_file(tmp)
        bad = _corrupt(out, lambda f: _set_manifest(
            f, lambda man: man.__setitem__("temperatures", [300.0, 350.0, 400.0])))
        with pytest.raises(ext.ExtractionError):
            ext.validate_feature_file_hdf5(bad, deep=True)


# --- walker / cycle / run_id / seed -----------------------------------------

def _dup_walker(f):
    d = f["features/sample_index/walker_id"]; arr = d[()]
    arr[0] = arr[1]      # both lanes of snapshot 0 claim the same walker
    d[...] = arr


def _walker_out_of_range(f):
    _set(f, "features/sample_index/walker_id", 0, 99)


def _cycle_differs_within_snapshot(f):
    _set(f, "features/sample_index/cycle", 1, 999)   # lane 1 of snapshot 0


def _decreasing_cycle(f):
    d = f["features/sample_index/cycle"]; arr = d[()]
    arr[2] = arr[3] = -5    # snapshot 1 cycle < snapshot 0 cycle
    d[...] = arr


def _run_id_changed(f):
    d = f["features/sample_index/run_id"]; arr = d[()]
    arr[0] = b"other" if isinstance(arr[0], bytes) else "other"
    d[...] = arr


def _seed_changed(f):
    _set(f, "features/sample_index/seed", 0, 999)


INDEX_CORRUPT = {
    "walker_duplication": _dup_walker,
    "walker_out_of_range": _walker_out_of_range,
    "cycle_differs_within_snapshot": _cycle_differs_within_snapshot,
    "decreasing_cycle": _decreasing_cycle,
    "run_id_changed": _run_id_changed,
    "seed_changed": _seed_changed,
}


@pytest.mark.parametrize("name", list(INDEX_CORRUPT))
def test_index_corruption_rejected(name):
    with tempfile.TemporaryDirectory() as tmp:
        out = _make_feature_file(tmp)
        bad = _corrupt(out, INDEX_CORRUPT[name])
        with pytest.raises(ext.ExtractionError):
            ext.validate_feature_file_hdf5(bad, deep=True)


def test_unsupported_schema_version_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        out = _make_feature_file(tmp)
        bad = _corrupt(out, lambda f: f.attrs.__setitem__(
            "feature_schema_version", 999))
        with pytest.raises(ext.ExtractionError):
            ext.validate_feature_file_hdf5(bad, deep=True)


def test_manifest_schema_mismatch_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        out = _make_feature_file(tmp)
        bad = _corrupt(out, lambda f: _set_manifest(
            f, lambda man: man.__setitem__("feature_schema_version", 2)))
        with pytest.raises(ext.ExtractionError):
            ext.validate_feature_file_hdf5(bad, deep=True)


def test_source_config_hash_mismatch_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        out = _make_feature_file(tmp)
        bad = _corrupt(out, lambda f: _set_manifest(
            f, lambda man: man.__setitem__(
                "source_configuration_sha256", "deadbeef")))
        with pytest.raises(ext.ExtractionError):
            ext.validate_feature_file_hdf5(bad, deep=True)


def test_clean_file_validates_with_authoritative_provenance():
    with tempfile.TemporaryDirectory() as tmp:
        out = _make_feature_file(tmp)
        info = ext.validate_feature_file_hdf5(out, deep=True)
        assert info["ok"] is True
