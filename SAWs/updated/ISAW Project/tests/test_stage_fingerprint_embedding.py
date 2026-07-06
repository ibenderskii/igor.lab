"""Phase 4: stage-fingerprint sensitivity + embedding into core artifacts.

Changing ANY canonical fingerprint field changes the SHA-256; the embedding
helpers write the same fingerprint (and its field record) into the configuration
HDF5, feature manifest + companion, and diagnostic-trajectory NPZ.
"""
import json
import os
import tempfile

import numpy as np
import pytest

import run_structural_regime_pilot as pilot

h5py = pytest.importorskip("h5py")

INFO = {"model_name": "hs", "param_names": ["h", "s"], "params": [330.5, 1.28],
        "Tref": 320.0, "Tscale": 80.0}


def _base(tmp):
    fit = os.path.join(tmp, "fit.json")
    with open(fit, "w") as fh:
        json.dump({"model_name": "hs", "params": [330.5, 1.28]}, fh)
    return dict(N=30, seed=1, ladder=[300.0, 320.0, 340.0],
                K_ladder=[0.1, 0.2, 0.3], info=INFO, fit_summary_path=fit,
                n_cycles=100, steps_per_swap=8, n_workers=1,
                structural_stride=2, snapshot_stride=5, burnin_frac=0.5)


def _fp(kw):
    return pilot._stage_fingerprint(pilot.build_stage_fingerprint_fields(**kw))


def test_each_field_change_changes_fingerprint():
    with tempfile.TemporaryDirectory() as tmp:
        base = _base(tmp)
        f0 = _fp(base)
        mutations = [
            ("N", 31), ("seed", 2), ("n_cycles", 101), ("steps_per_swap", 9),
            ("n_workers", 2), ("structural_stride", 3), ("snapshot_stride", 6),
            ("burnin_frac", 0.6),
            ("ladder", [300.0, 321.0, 340.0]),
            ("K_ladder", [0.1, 0.25, 0.3]),
        ]
        for key, val in mutations:
            kw = dict(base)
            kw[key] = val
            assert _fp(kw) != f0, f"fingerprint unchanged when {key} changed"
        # model record mutations
        for mut in ({"model_name": "poly"}, {"params": [331.0, 1.28]},
                    {"param_names": ["a", "b"]}, {"Tref": 321.0},
                    {"Tscale": 81.0}):
            kw = dict(base)
            kw["info"] = {**INFO, **mut}
            assert _fp(kw) != f0, f"fingerprint unchanged for info {mut}"
        # fit-summary content change -> different file hash -> different fp
        kw = dict(base)
        with open(kw["fit_summary_path"], "w") as fh:
            json.dump({"model_name": "hs", "params": [330.5, 1.29]}, fh)
        assert _fp(kw) != f0


def test_same_inputs_same_fingerprint():
    with tempfile.TemporaryDirectory() as tmp:
        base = _base(tmp)
        assert _fp(base) == _fp(dict(base))


def test_embed_into_hdf5_and_npz_and_feature_consistent():
    with tempfile.TemporaryDirectory() as tmp:
        fields = pilot.build_stage_fingerprint_fields(**_base(tmp))
        fp = pilot._stage_fingerprint(fields)

        # configuration-like HDF5 with a metadata group
        cfg = os.path.join(tmp, "cfg.h5")
        with h5py.File(cfg, "w") as f:
            f.create_group("metadata")
        pilot._embed_fingerprint_hdf5(cfg, fp, fields)
        with h5py.File(cfg, "r") as f:
            assert f.attrs["stage_fingerprint"] == fp
            assert f["metadata"].attrs["stage_fingerprint"] == fp
            rec = json.loads(f["metadata"].attrs["stage_fingerprint_fields"])
            assert rec["N"] == fields["N"] and rec["seed"] == fields["seed"]

        # diagnostic-trajectory NPZ
        npz = os.path.join(tmp, "diag.npz")
        np.savez(npz, Ts=np.array([300.0, 340.0]), x=np.arange(5))
        pilot._embed_fingerprint_npz(npz, fp, fields)
        with np.load(npz, allow_pickle=True) as z:
            assert str(z["stage_fingerprint"]) == fp
            assert np.array_equal(z["x"], np.arange(5))  # payload preserved
            rec2 = json.loads(str(z["stage_fingerprint_fields"]))
            assert rec2["K_ladder"] == fields["K_ladder"]


def test_embed_into_feature_manifest_and_companion():
    ext = pytest.importorskip("extract_contact_motif_features")
    cio = pytest.importorskip("isaw_config_io")
    ico = pytest.importorskip("isaw_contact_observables")
    if not cio.h5py_available():
        pytest.skip("h5py missing")
    hp = [(0, 0, 0), (1, 0, 0), (2, 0, 0), (2, 1, 0), (1, 1, 0), (0, 1, 0)]
    with tempfile.TemporaryDirectory() as tmp:
        cp, _ = ico.build_contact_map(hp)
        m = cp.shape[0]
        rg2 = ico.radius_of_gyration_squared(hp)
        ree2 = ico.end_to_end_distance_squared(hp)
        inp = os.path.join(tmp, "snap.h5")
        w = cio.SnapshotWriter(inp, n_beads=6, n_temperatures=2,
                               metadata={"run_id": "fp", "seed": 1,
                                         "temperatures": [300.0, 340.0],
                                         "model_name": "hs",
                                         "param_names": ["h", "s"],
                                         "model_params": [330.5, 1.28],
                                         "Tref": 320.0, "Tscale": 80.0})
        w.append(cycle=0,
                 coordinates=np.stack([np.asarray(hp, dtype=np.int64)] * 2),
                 walker_id=np.array([0, 1]), contacts=np.full(2, m),
                 rg2_lattice=np.full(2, rg2), ree2_lattice=np.full(2, ree2))
        w.mark_complete()
        w.close()
        feat = os.path.join(tmp, "feat.h5")
        ext.extract(inp, feat, validate=True, overwrite=True)
        fields = pilot.build_stage_fingerprint_fields(**_base(tmp))
        fp = pilot._stage_fingerprint(fields)
        pilot._embed_fingerprint_feature(feat, fp, fields)
        with h5py.File(feat, "r") as f:
            man = json.loads(f["metadata"].attrs["manifest"])
            assert man["stage_fingerprint"] == fp
            assert f.attrs["stage_fingerprint"] == fp
        comp = json.loads(open(feat + ".manifest.json").read())
        assert comp["stage_fingerprint"] == fp
        assert comp["stage_fingerprint_fields"]["N"] == fields["N"]
