"""Phase 3/4/5: historical-definition provenance, hardened feature-file
validator (adversarial corruption), and feature-dictionary bijection."""
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


def _make_feature_file(tmp, chain=HAIRPIN6, extra_meta=None):
    nb, nT = len(chain), 2
    cp, _ = ico.build_contact_map(chain)
    m = cp.shape[0]
    rg2 = ico.radius_of_gyration_squared(chain)
    ree2 = ico.end_to_end_distance_squared(chain)
    meta = {"run_id": "cf", "seed": 1, "temperatures": [300.0, 350.0]}
    meta.update(MODEL_META)
    if extra_meta:
        meta.update(extra_meta)
    inp = os.path.join(tmp, "snap.h5")
    w = cio.SnapshotWriter(inp, n_beads=nb, n_temperatures=nT, metadata=meta)
    for c in range(3):
        w.append(cycle=c,
                 coordinates=np.stack([np.asarray(chain, dtype=np.int64)] * nT),
                 walker_id=np.array([0, 1] if c % 2 == 0 else [1, 0]),
                 contacts=np.full(nT, m), rg2_lattice=np.full(nT, rg2),
                 ree2_lattice=np.full(nT, ree2))
    w.mark_complete()
    w.close()
    out = os.path.join(tmp, "feat.h5")
    ext.extract(inp, out, validate=True, output_format="hdf5", overwrite=True)
    return out


def test_clean_feature_file_validates():
    with tempfile.TemporaryDirectory() as tmp:
        out = _make_feature_file(tmp)
        info = ext.validate_feature_file_hdf5(out)
        assert info["ok"] and info["uses_current_definitions"] is True


# --- Phase 3: historical definition provenance -----------------------------

def test_historical_definitions_preserved():
    import h5py
    straight = [(i, 0, 0) for i in range(16)]   # valid SAW, m=0
    historical = {
        "definitions_version": "1.0.0", "n_beads": 16, "min_separation": 3,
        "fixed": dict(ico.FIXED_BIN_DEFINITIONS),
        # older scaled record without an explicit local_boundary key
        "scaled": {"scheme": "scaled", "local_max_ratio": 0.10,
                   "meso_max_ratio": 0.33},
        "bin_definition_source": "historical_run",
    }
    with tempfile.TemporaryDirectory() as tmp:
        out = _make_feature_file(
            tmp, chain=straight,
            extra_meta={"structural_bin_definitions": historical})
        with h5py.File(out, "r") as f:
            man = json.loads(f["metadata"].attrs["manifest"])
        assert man["definitions_version"] == "1.0.0"          # not relabeled
        assert man["code_definitions_version"] == ico.DEFINITIONS_VERSION
        assert man["uses_current_definitions"] is False
        # validator accepts a historical file and reports it as non-current
        info = ext.validate_feature_file_hdf5(out)
        assert info["definitions_version"] == "1.0.0"
        assert info["uses_current_definitions"] is False


# --- Phase 4: adversarial corruption ---------------------------------------

def _corrupt(out, fn):
    """Copy the feature file and apply an in-place mutation, return new path."""
    dst = out + ".bad.h5"
    shutil.copy(out, dst)
    import h5py
    with h5py.File(dst, "r+") as f:
        fn(f)
    return dst


def _set(f, path, idx, value):
    d = f[path]
    arr = d[()]
    arr[idx] = value
    d[...] = arr


CORRUPTIONS = {
    "bonded_pair": lambda f: _set(f, "features/contact_pairs/pairs", 0, [0, 1]),
    "duplicate_pair": lambda f: _set(f, "features/contact_pairs/pairs", 1,
                                     f["features/contact_pairs/pairs"][0]),
    "unsorted_pairs": lambda f: _swap_row0_pairs(f),
    "even_separation": lambda f: _set(f, "features/contact_pairs/pairs", 0, [0, 4]),
    "bad_offset": lambda f: _set(f, "features/contact_pairs/offsets", 1,
                                 int(f["features/contact_pairs/offsets"][1]) + 1),
    "m_inconsistent": lambda f: _set(f, "features/scalars/m", 0,
                                     int(f["features/scalars/m"][0]) + 5),
    "m_r_inconsistent": lambda f: _shift_mr(f),
    "bad_motif": lambda f: _set(f, "features/scalars/pair_nested", 0,
                                int(f["features/scalars/pair_nested"][0]) + 1),
    "bad_graph_stat": lambda f: _set(f, "features/scalars/largest_component_vertices",
                                     0, 99),
    "bad_graph_float": lambda f: _set(
        f, "features/scalars/largest_component_fraction_of_N", 0, 0.99),
    "bad_augmented_float": lambda f: _set(
        f, "features/scalars/augmented_mean_degree", 0,
        float(f["features/scalars/augmented_mean_degree"][0]) + 1.0),
    "bad_eigenvalue": lambda f: _set(f, "features/scalars/gyration_lambda_1", 0, -5.0),
    "bad_asphericity": lambda f: _set(
        f, "features/scalars/asphericity", 0,
        float(f["features/scalars/asphericity"][0]) + 5.0),
    "bad_mean_separation": lambda f: _set(
        f, "features/scalars/mean_contact_separation", 0,
        float(f["features/scalars/mean_contact_separation"][0]) + 3.0),
    "bad_max_separation": lambda f: _set(
        f, "features/scalars/max_contact_separation", 0,
        float(f["features/scalars/max_contact_separation"][0]) + 3.0),
    "bad_b": lambda f: _set(f, "features/scalars/b_T", 0, 12345.0),
    "bad_K": lambda f: _set(f, "features/scalars/K_T", 0, 12345.0),
    "bad_q_finite": lambda f: _set(f, "features/scalars/q_T", 0, 12345.0),
    "q_nan": lambda f: _set(f, "features/scalars/q_T", 0, float("nan")),
    "q_invalid_inf": lambda f: _set(f, "features/scalars/q_T", 0, float("inf")),
    "bad_u": lambda f: _set(f, "features/scalars/reduced_potential_u", 0, 12345.0),
    "bad_H": lambda f: _set(f, "features/scalars/effective_energy_H", 0, 12345.0),
    "H_nan": lambda f: _set(f, "features/scalars/effective_energy_H", 0, float("nan")),
    "H_invalid_inf": lambda f: _set(f, "features/scalars/effective_energy_H", 0,
                                    float("inf")),
    "duplicate_primary_key": lambda f: _dup_pk(f),
    "incorrect_temperature_value": lambda f: _set(
        f, "features/sample_index/temperature", 0,
        float(f["features/sample_index/temperature"][0]) + 5.0),
    "incorrect_temperature_index": lambda f: _set(
        f, "features/sample_index/temperature_index", 0,
        int(f["features/sample_index/temperature_index"][0]) + 1),
    "bad_historical_metadata": lambda f: _tamper_manifest_defs(f),
}


def _swap_row0_pairs(f):
    d = f["features/contact_pairs/pairs"]
    arr = d[()]
    arr[[0, 1]] = arr[[1, 0]]   # reverse the two pairs of row 0 -> unsorted
    d[...] = arr


def _shift_mr(f):
    d = f["features/m_r"]
    arr = d[()]
    # move one count from r=5 to r=3 in row 0 (sum preserved, parity preserved)
    arr[0, 5] -= 1
    arr[0, 3] += 1
    d[...] = arr


def _dup_pk(f):
    # make row 1 collide with row 0 on the full primary key AND ladder
    for col in ("temperature_index",):
        _set(f, "features/sample_index/" + col, 1,
             int(f["features/sample_index/" + col][0]))
    _set(f, "features/sample_index/temperature", 1,
         float(f["features/sample_index/temperature"][0]))
    _set(f, "features/sample_index/snapshot_index", 1,
         int(f["features/sample_index/snapshot_index"][0]))


def _tamper_manifest_defs(f):
    man = json.loads(f["metadata"].attrs["manifest"])
    man["scaled_bin_definitions"]["meso_max_ratio"] = 0.05   # < local -> invalid
    f["metadata"].attrs["manifest"] = json.dumps(man, default=str)


@pytest.mark.parametrize("name", list(CORRUPTIONS))
def test_corruption_rejected(name):
    with tempfile.TemporaryDirectory() as tmp:
        out = _make_feature_file(tmp)
        bad = _corrupt(out, CORRUPTIONS[name])
        with pytest.raises(ext.ExtractionError):
            ext.validate_feature_file_hdf5(bad)


def test_corruption_identifies_row():
    with tempfile.TemporaryDirectory() as tmp:
        out = _make_feature_file(tmp)
        bad = _corrupt(out, CORRUPTIONS["bonded_pair"])
        with pytest.raises(ext.ExtractionError) as e:
            ext.validate_feature_file_hdf5(bad)
        assert "row 0" in str(e.value)


# --- Phase 5: feature dictionary bijection ---------------------------------

def test_feature_dictionary_bijection():
    fd = ext.build_feature_dictionary()
    names = [x["name"] for x in fd["fields"]]
    assert len(names) == len(set(names)), "duplicate dictionary entries"
    expected = (set(ext.FEATURE_COLUMNS)
                | {"m_r", "contact_pairs/pairs", "contact_pairs/offsets"}
                | set(fd["provenance_metadata_fields"]))
    assert set(names) == expected
    # no placeholder definitions: every field has a non-empty, non-name definition
    for x in fd["fields"]:
        assert x["mathematical_definition"] and \
            x["mathematical_definition"] != x["name"].replace("_", " ")
        assert "validation_identity" in x and x["validation_identity"]
        assert "source_representation" in x
