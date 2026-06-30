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


def _chain_with_longish_contact():
    """A SAW whose contact map has a separation r >= 5 (so a custom long
    threshold actually changes the fixed-long count)."""
    from conftest import random_saw
    for seed in range(300):
        chain = random_saw(24, seed)
        if chain is None:
            continue
        cp, seps = ico.build_contact_map(chain)
        if seps.size and int(seps.max()) >= 5:
            return chain
    raise AssertionError("no suitable chain found")


def test_custom_threshold_changes_fixed_long_counts():
    chain = _chain_with_longish_contact()
    n = len(chain)
    cp, _ = ico.build_contact_map(chain)
    m_r = ico.contact_separation_counts(cp, n)
    default_long = ico.bin_contact_separations_fixed(m_r, n)["m_long_fixed"]
    custom = {**ico.FIXED_BIN_DEFINITIONS, "long_threshold_fixed": 5}
    custom_long = ico.bin_contact_separations_fixed(m_r, n, custom)["m_long_fixed"]
    assert custom_long >= default_long
    assert custom_long != default_long   # the chain was chosen to make it differ


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
    chain = _chain_with_longish_contact()
    n = len(chain)
    cp, _ = ico.build_contact_map(chain)
    m_r = ico.contact_separation_counts(cp, n)
    custom = {**ico.FIXED_BIN_DEFINITIONS, "long_threshold_fixed": 5}
    expected_custom = ico.bin_contact_separations_fixed(m_r, n, custom)["m_long_fixed"]
    expected_default = ico.bin_contact_separations_fixed(m_r, n)["m_long_fixed"]

    with tempfile.TemporaryDirectory() as tmp:
        inp = _write_snapshot(tmp, chain, fixed_defs=custom,
                              scaled_defs=ico.SCALED_BIN_DEFINITIONS)
        out = os.path.join(tmp, "feat.h5")
        ext.extract(inp, out, output_format="hdf5", overwrite=True)
        m_long, source = _feature_m_long(out)
        assert source == "input_file"
        assert m_long == expected_custom != expected_default


def test_extractor_defaults_when_no_stored_definitions():
    chain = _chain_with_longish_contact()
    n = len(chain)
    cp, _ = ico.build_contact_map(chain)
    m_r = ico.contact_separation_counts(cp, n)
    expected_default = ico.bin_contact_separations_fixed(m_r, n)["m_long_fixed"]
    with tempfile.TemporaryDirectory() as tmp:
        inp = _write_snapshot(tmp, chain, fixed_defs=ico.FIXED_BIN_DEFINITIONS,
                              scaled_defs=ico.SCALED_BIN_DEFINITIONS,
                              store_defs=False)
        out = os.path.join(tmp, "feat.h5")
        ext.extract(inp, out, output_format="hdf5", overwrite=True)
        m_long, source = _feature_m_long(out)
        assert source == "module_default"
        assert m_long == expected_default


def test_online_run_threads_custom_definitions():
    # Two identical-seed runs differing only in the fixed long threshold must
    # produce different m_long_fixed trajectories (definitions are authoritative).
    import remd_uniform_chain_2_new as remd
    Ts = np.linspace(300, 360, 4)
    common = dict(N=16, Ts=Ts, steps_per_swap=30, n_cycles=30,
                  model_name="hs", params=[378.96, 1.39686], Tref=330.0,
                  Tscale=80.0, seed=7, n_workers=1, verbose=False,
                  structural_observables=True, structural_stride=1)
    default_defs = ico.project_bin_definitions(16)
    custom_defs = ico.project_bin_definitions(16)
    custom_defs["fixed"] = {**ico.FIXED_BIN_DEFINITIONS, "long_threshold_fixed": 5}
    reps_d, _, _ = remd.run_remd(bin_defs=default_defs, **common)
    reps_c, _, _ = remd.run_remd(bin_defs=custom_defs, **common)
    sum_d = sum(sum(r.m_long_traj) for r in reps_d)
    sum_c = sum(sum(r.m_long_traj) for r in reps_c)
    assert sum_c >= sum_d
    assert sum_c != sum_d
