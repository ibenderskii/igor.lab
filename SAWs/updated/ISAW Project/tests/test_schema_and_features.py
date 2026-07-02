"""Part B/C tests: shared schema, semantic bins, deepcopy, augmented graph,
thermodynamic columns, contact-pair persistence, feature dictionary."""
import json
import os
import tempfile

import numpy as np
import pytest

import isaw_contact_observables as ico
import isaw_schema as sch
import isaw_config_io as cio
import extract_contact_motif_features as ext

pytestmark_h5 = pytest.mark.skipif(not cio.h5py_available(), reason="h5py missing")

HAIRPIN6 = [(0, 0, 0), (1, 0, 0), (2, 0, 0), (2, 1, 0), (1, 1, 0), (0, 1, 0)]


# --- Part B: schema ---------------------------------------------------------

def test_code_json_definitions_consistent():
    sch.check_definitions_consistency()


def test_primary_key_matches_json():
    defs = sch.project_definitions()
    assert list(defs["primary_key_definition"]) == list(sch.PRIMARY_KEY)


def test_primary_key_uniqueness_validator():
    cols = {"run_id": ["r"] * 4, "seed": [1, 1, 1, 1],
            "snapshot_index": [0, 0, 1, 1], "temperature_index": [0, 1, 0, 1]}
    sch.validate_feature_primary_keys(cols)
    cols_bad = dict(cols, temperature_index=[0, 1, 0, 0])
    with pytest.raises(sch.SchemaError):
        sch.validate_feature_primary_keys(cols_bad)


def test_run_metadata_validator():
    good = {"run_id": "r", "seed": 1, "n_beads": 30, "n_steps": 29,
            "model_name": "hs", "param_names": ["h", "s"], "model_params": [1, 2],
            "Tref": 320, "Tscale": 80, "temperatures": [300, 320]}
    sch.validate_run_metadata(good)
    with pytest.raises(sch.SchemaError):
        sch.validate_run_metadata(dict(good, n_steps=99))
    with pytest.raises(sch.SchemaError):
        bad = dict(good); del bad["model_params"]
        sch.validate_run_metadata(bad)


def test_definitions_md_matches_json():
    # docs/definitions.md must equal the rendered version (verify, don't drift).
    rendered = sch.render_definitions_md()
    assert sch.DOCS_DEFINITIONS_PATH.exists(), "run `python isaw_schema.py`"
    assert sch.DOCS_DEFINITIONS_PATH.read_text(encoding="utf-8") == rendered


# --- Part A: semantic bin validation + deepcopy -----------------------------

@pytest.mark.parametrize("bad", [
    {"short_fixed": {"r_min": 3, "r_max": 9}, "medium_fixed": {"r_min": 8},
     "long_threshold_fixed": 15},                       # short_max !< medium_min
    {"short_fixed": {"r_min": 3, "r_max": 9}, "medium_fixed": {"r_min": 11},
     "long_threshold_fixed": 9},                        # medium_min !< long_thr
    {"short_fixed": {"r_min": 3, "r_max": 9}, "medium_fixed": {"r_min": 11},
     "long_threshold_fixed": 11.5},                     # fractional boundary
    {"short_fixed": {"r_min": 1, "r_max": 9}, "medium_fixed": {"r_min": 11},
     "long_threshold_fixed": 15},                       # short_min < 3
])
def test_fixed_semantics_rejects(bad):
    with pytest.raises(ico.ContactMapError):
        ico.validate_fixed_bin_semantics(bad, 30)


@pytest.mark.parametrize("bad", [
    {"local_max_ratio": -0.1, "meso_max_ratio": 0.3},   # negative
    {"local_max_ratio": 0.4, "meso_max_ratio": 0.2},    # reversed
    {"local_max_ratio": 0.1, "meso_max_ratio": 1.5},    # > 1
    {"local_max_ratio": 0.1},                            # missing key
    {"local_max_ratio": float("nan"), "meso_max_ratio": 0.3},  # nonfinite
])
def test_scaled_semantics_rejects(bad):
    with pytest.raises(ico.ContactMapError):
        ico.validate_scaled_bin_semantics(bad)


def test_resolved_definitions_are_deepcopies():
    rec = ico.project_bin_definitions(30)
    rec["fixed"]["short_fixed"]["r_min"] = 999
    rec["scaled"]["local_max_ratio"] = 999
    assert ico.FIXED_BIN_DEFINITIONS["short_fixed"]["r_min"] == 3
    assert ico.SCALED_BIN_DEFINITIONS["local_max_ratio"] == 0.10
    rec2 = ico.normalize_bin_definitions(n_beads=30)
    rec2["fixed"]["long_threshold_fixed"] = -5
    assert ico.FIXED_BIN_DEFINITIONS["long_threshold_fixed"] == 15


# --- Part C3: augmented graph ----------------------------------------------

def test_augmented_graph_identities():
    cp, _ = ico.build_contact_map(HAIRPIN6)
    m = cp.shape[0]
    g = ico.augmented_graph_summary(cp, 6)
    assert g["augmented_graph_edges"] == 6 - 1 + m
    assert g["augmented_graph_components"] == 1
    assert g["augmented_graph_cycle_rank"] == m
    # straight chain (zero contacts): edges N-1, cycle rank 0
    g0 = ico.augmented_graph_summary(np.empty((0, 2), dtype=np.int64), 8)
    assert g0["augmented_graph_edges"] == 7
    assert g0["augmented_graph_cycle_rank"] == 0


# --- Part C2/C1: thermo columns + contact pairs (via a real extraction) -----

def _snapshot(tmp):
    nb, nT = 6, 2
    cp, _ = ico.build_contact_map(HAIRPIN6)
    m = cp.shape[0]
    rg2 = ico.radius_of_gyration_squared(HAIRPIN6)
    ree2 = ico.end_to_end_distance_squared(HAIRPIN6)
    path = os.path.join(tmp, "snap.h5")
    w = cio.SnapshotWriter(path, n_beads=nb, n_temperatures=nT,
                           metadata={"run_id": "qt", "seed": 1,
                                     "temperatures": [300.0, 350.0],
                                     "model_name": "hs", "param_names": ["h", "s"],
                                     "model_params": [647.7, 1.874],
                                     "Tref": 320.0, "Tscale": 80.0})
    for c in range(3):
        w.append(cycle=c,
                 coordinates=np.stack([np.asarray(HAIRPIN6, dtype=np.int64)] * nT),
                 walker_id=np.array([0, 1] if c % 2 == 0 else [1, 0]),
                 contacts=np.full(nT, m), rg2_lattice=np.full(nT, rg2),
                 ree2_lattice=np.full(nT, ree2))
    w.mark_complete()
    w.close()
    return path, m


@pytestmark_h5
def test_thermo_and_pairs_and_schema_validator():
    import h5py
    import remd_uniform_chain_2_new as remd
    with tempfile.TemporaryDirectory() as tmp:
        inp, m = _snapshot(tmp)
        out = os.path.join(tmp, "feat.h5")
        info = ext.extract(inp, out, validate=True, output_format="hdf5")
        assert info["thermodynamic_status"] == "ok"
        # comprehensive schema validator passes
        ext.validate_feature_file_hdf5(out)
        with h5py.File(out, "r") as f:
            K = f["features/scalars/K_T"][()]
            b = f["features/scalars/b_T"][()]
            u = f["features/scalars/reduced_potential_u"][()]
            mm = f["features/scalars/m"][()]
            T = f["features/sample_index/temperature"][()]
            # K = -b ; u = -m K ; b matches the model exactly
            assert np.allclose(K, -b)
            assert np.allclose(u, -mm.astype(float) * K)
            for i in range(len(T)):
                exp_b = remd.reduced_bias("hs", [647.7, 1.874], float(T[i]),
                                          320.0, 80.0)
                assert abs(b[i] - exp_b) < 1e-9
            # contact pairs reconstruct per row, count == m
            pairs = f["features/contact_pairs/pairs"][()]
            offs = f["features/contact_pairs/offsets"][()]
            assert int(offs[0]) == 0 and int(offs[-1]) == pairs.shape[0]
            for i in range(len(mm)):
                row_pairs = pairs[offs[i]:offs[i + 1]]
                assert row_pairs.shape[0] == int(mm[i])


def test_feature_dictionary_covers_all_columns():
    fd = ext.build_feature_dictionary()
    names = {x["name"] for x in fd["fields"]}
    for col in ext.FEATURE_COLUMNS:
        assert col in names, col
    assert "m_r" in names
    assert "contact_pairs/pairs" in names and "contact_pairs/offsets" in names
    assert fd["feature_schema_version"] == ext.FEATURE_SCHEMA_VERSION


def test_q_overflow_safe():
    assert ext._safe_q(1000.0) == float("inf")
    assert ext._safe_q(-1000.0) == 0.0
    assert abs(ext._safe_q(0.5) - np.exp(0.5)) < 1e-9
