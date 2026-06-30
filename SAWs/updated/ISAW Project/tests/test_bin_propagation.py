"""P2: custom bin definitions are authoritative online AND offline (stored)."""
import json
import os
import tempfile

import numpy as np
import pytest

import isaw_config_io as cio
import isaw_contact_observables as ico
import extract_contact_motif_features as ext

pytestmark = pytest.mark.skipif(not cio.h5py_available(), reason="h5py missing")


# A deterministic 14-bead hairpin with contacts at r = 3,5,7,9,11,13 (m=6).
# With the default long_threshold_fixed=15, m_long_fixed=0; with a semantically
# valid custom threshold of 13, r=13 becomes long so m_long_fixed=1.
HAIRPIN14 = (
    [(x, 0, 0) for x in range(7)]            # beads 0..6 along +x
    + [(x, 1, 0) for x in range(6, -1, -1)]  # beads 7..13 back along +x at y=1
)
# Custom fixed definitions: valid (3<=3<=9<11<13<14) and reclassify r=13 as long.
CUSTOM_FIXED = {**ico.FIXED_BIN_DEFINITIONS, "long_threshold_fixed": 13}


def test_custom_threshold_changes_fixed_long_counts():
    n = len(HAIRPIN14)
    cp, seps = ico.build_contact_map(HAIRPIN14)
    assert sorted(seps.tolist()) == [3, 5, 7, 9, 11, 13]
    m_r = ico.contact_separation_counts(cp, n)
    default_long = ico.bin_contact_separations_fixed(m_r, n)["m_long_fixed"]
    custom_long = ico.bin_contact_separations_fixed(m_r, n, CUSTOM_FIXED)["m_long_fixed"]
    assert default_long == 0 and custom_long == 1


def _write_snapshot(tmp, chain, *, fixed_defs, scaled_defs, store_defs=True):
    n = len(chain)
    nT = 2
    cp, _ = ico.build_contact_map(chain)
    m = cp.shape[0]
    rg2 = ico.radius_of_gyration_squared(chain)
    ree2 = ico.end_to_end_distance_squared(chain)
    meta = {"run_id": "binprop", "seed": 1, "temperatures": [300.0, 320.0]}
    if store_defs:
        meta["fixed_bin_definitions"] = json.dumps(fixed_defs)
        meta["scaled_bin_definitions"] = json.dumps(scaled_defs)
    path = os.path.join(tmp, "snap.h5")
    w = cio.SnapshotWriter(path, n_beads=n, n_temperatures=nT, metadata=meta)
    coords = np.stack([np.asarray(chain, dtype=np.int64)] * nT)
    w.append(cycle=0, coordinates=coords, walker_id=np.array([0, 1]),
             contacts=np.full(nT, m), rg2_lattice=np.full(nT, rg2),
             ree2_lattice=np.full(nT, ree2))
    w.mark_complete()
    w.close()
    return path


def _feature_m_long(feat_path):
    import h5py
    with h5py.File(feat_path, "r") as f:
        return int(f["features/scalars/m_long_fixed"][0]), \
               json.loads(f["metadata"].attrs["manifest"])["bin_definitions_source"]


def test_extractor_uses_stored_custom_definitions():
    n = len(HAIRPIN14)
    with tempfile.TemporaryDirectory() as tmp:
        inp = _write_snapshot(tmp, HAIRPIN14, fixed_defs=CUSTOM_FIXED,
                              scaled_defs=ico.SCALED_BIN_DEFINITIONS)
        out = os.path.join(tmp, "feat.h5")
        ext.extract(inp, out, output_format="hdf5", overwrite=True)
        m_long, source = _feature_m_long(out)
        assert source == "input_file"
        assert m_long == 1     # custom threshold 13 classifies r=13 as long


def test_extractor_defaults_when_no_stored_definitions():
    with tempfile.TemporaryDirectory() as tmp:
        inp = _write_snapshot(tmp, HAIRPIN14, fixed_defs=ico.FIXED_BIN_DEFINITIONS,
                              scaled_defs=ico.SCALED_BIN_DEFINITIONS,
                              store_defs=False)
        out = os.path.join(tmp, "feat.h5")
        ext.extract(inp, out, output_format="hdf5", overwrite=True)
        m_long, source = _feature_m_long(out)
        assert source == "module_default"
        assert m_long == 0     # default threshold 15 -> r=13 is medium, not long


def test_online_run_threads_custom_definitions():
    # Two identical-seed runs differing only in the SCALED meso boundary must
    # produce different m_global_scaled trajectories (definitions are
    # authoritative and threaded into the online structural calculation).
    import remd_uniform_chain_2_new as remd
    Ts = np.linspace(300, 360, 4)
    common = dict(N=16, Ts=Ts, steps_per_swap=30, n_cycles=40,
                  model_name="hs", params=[647.7, 1.874], Tref=320.0,
                  Tscale=80.0, seed=7, n_workers=1, verbose=False,
                  structural_observables=True, structural_stride=1)
    default_defs = ico.project_bin_definitions(16)
    custom_defs = ico.project_bin_definitions(16)
    # meso 0.20 -> global catches r/N>=0.20 (r>=4 -> r=5,7,...) vs default 0.33.
    custom_defs["scaled"] = {**ico.SCALED_BIN_DEFINITIONS, "meso_max_ratio": 0.20}
    reps_d, _, _ = remd.run_remd(bin_defs=default_defs, **common)
    reps_c, _, _ = remd.run_remd(bin_defs=custom_defs, **common)
    sum_d = sum(sum(r.m_global_scaled_traj) for r in reps_d)
    sum_c = sum(sum(r.m_global_scaled_traj) for r in reps_c)
    assert sum_c >= sum_d
    assert sum_c != sum_d
