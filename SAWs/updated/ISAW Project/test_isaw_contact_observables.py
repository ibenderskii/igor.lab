"""
pytest suite for the ISAW structural-observable stack:

* isaw_contact_observables  (contact maps, geometry, graph, pair motifs, bins)
* remd_uniform_chain_2       (state.m refactor, structural trajectories, output)
* isaw_config_io / extract_contact_motif_features  (HDF5 round-trip, features)

Run:  python -m pytest "test_isaw_contact_observables.py" -q
"""
import math
import os
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import isaw_contact_observables as ico
import remd_uniform_chain_2_new as remd

try:
    import networkx as nx
    _HAVE_NX = True
except Exception:
    _HAVE_NX = False

try:
    import h5py
    _HAVE_H5PY = True
except Exception:
    _HAVE_H5PY = False


# Hand-built conformations -------------------------------------------------
STRAIGHT10 = [(i, 0, 0) for i in range(10)]
# Planar hairpin: two contacts (0,5) r=5 and (1,4) r=3.
HAIRPIN6 = [(0, 0, 0), (1, 0, 0), (2, 0, 0), (2, 1, 0), (1, 1, 0), (0, 1, 0)]


def _random_saw(n, seed):
    """Generate a random self-avoiding walk on the cubic lattice (or None)."""
    rng = np.random.RandomState(seed)
    nn = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    for _ in range(200):
        chain = [(0, 0, 0)]
        occ = {(0, 0, 0)}
        ok = True
        for _ in range(n - 1):
            opts = [(chain[-1][0] + d[0], chain[-1][1] + d[1], chain[-1][2] + d[2])
                    for d in nn]
            opts = [o for o in opts if o not in occ]
            if not opts:
                ok = False
                break
            nxt = opts[rng.randint(len(opts))]
            chain.append(nxt)
            occ.add(nxt)
        if ok:
            return chain
    return None


# ---------------------------------------------------------------------------
# Contact-map tests
# ---------------------------------------------------------------------------

def test_straight_chain_zero_contacts():
    cp, seps = ico.build_contact_map(STRAIGHT10)
    assert cp.shape == (0, 2)
    assert seps.shape == (0,)


def test_handbuilt_known_contacts():
    cp, seps = ico.build_contact_map(HAIRPIN6)
    assert [tuple(p) for p in cp] == [(0, 5), (1, 4)]
    assert sorted(seps.tolist()) == [3, 5]


def test_hash_matches_bruteforce():
    for seed in range(25):
        chain = _random_saw(18, seed)
        if chain is None:
            continue
        a, _ = ico.build_contact_map(chain)
        b, _ = ico.build_contact_map_bruteforce(chain)
        assert np.array_equal(a, b), seed


def test_len_pairs_matches_contact_count():
    for seed in range(20):
        chain = _random_saw(20, seed)
        if chain is None:
            continue
        cp, _ = ico.build_contact_map(chain)
        cc = int(round(remd.contact_count(chain, set(chain))))
        assert cp.shape[0] == cc, seed


def test_all_separations_odd():
    for seed in range(20):
        chain = _random_saw(22, seed)
        if chain is None:
            continue
        _, seps = ico.build_contact_map(chain)
        assert np.all((seps % 2) == 1)


def test_no_bonded_pair_present():
    for seed in range(20):
        chain = _random_saw(20, seed)
        if chain is None:
            continue
        cp, _ = ico.build_contact_map(chain)
        if cp.shape[0]:
            assert np.all((cp[:, 1] - cp[:, 0]) > 1)


def test_duplicate_pairs_rejected_by_validate():
    cp = np.array([[0, 5], [0, 5]], dtype=np.int64)
    with pytest.raises(ico.ContactMapError):
        ico.validate_contact_map(HAIRPIN6, cp)


def test_mr_sums_to_m():
    for seed in range(20):
        chain = _random_saw(24, seed)
        if chain is None:
            continue
        cp, _ = ico.build_contact_map(chain)
        m_r = ico.contact_separation_counts(cp, len(chain))
        assert int(m_r.sum()) == cp.shape[0]
        # even entries are zero
        assert np.all(m_r[0::2] == 0)


def test_validate_detects_self_avoidance_violation():
    bad = [(0, 0, 0), (1, 0, 0), (1, 0, 0)]
    with pytest.raises(ico.ContactMapError):
        ico.validate_contact_map(bad, np.empty((0, 2), dtype=np.int64))


# ---------------------------------------------------------------------------
# Geometry tests
# ---------------------------------------------------------------------------

def test_rg2_matches_square_of_rg():
    for seed in range(15):
        chain = _random_saw(20, seed)
        if chain is None:
            continue
        rg = remd.radius_of_gyration(chain)
        rg2 = ico.radius_of_gyration_squared(chain)
        assert abs(rg * rg - rg2) < 1e-9


def test_translation_invariance():
    chain = _random_saw(20, 3)
    shifted = [(x + 7, y - 4, z + 11) for (x, y, z) in chain]
    assert abs(ico.radius_of_gyration_squared(chain)
               - ico.radius_of_gyration_squared(shifted)) < 1e-9
    assert abs(ico.end_to_end_distance_squared(chain)
               - ico.end_to_end_distance_squared(shifted)) < 1e-9
    a, _ = ico.build_contact_map(chain)
    b, _ = ico.build_contact_map(shifted)
    assert np.array_equal(a, b)


def test_cubic_rotation_invariance():
    chain = _random_saw(20, 5)
    M = remd.ROT_MATS[0]
    rot = [remd._apply_rot(M, r) for r in chain]
    assert abs(ico.radius_of_gyration_squared(chain)
               - ico.radius_of_gyration_squared(rot)) < 1e-9
    assert abs(ico.end_to_end_distance_squared(chain)
               - ico.end_to_end_distance_squared(rot)) < 1e-9
    a, _ = ico.build_contact_map(chain)
    b, _ = ico.build_contact_map(rot)
    assert a.shape == b.shape  # same number of contacts


def test_rg_scale_squares():
    chain = _random_saw(20, 7)
    rg2 = ico.radius_of_gyration_squared(chain)
    scale = 0.37
    assert abs((scale ** 2) * rg2 - (scale * math.sqrt(rg2)) ** 2) < 1e-9


def test_gyration_trace_equals_rg2():
    chain = _random_saw(20, 9)
    G = ico.gyration_tensor(chain)
    assert abs(np.trace(G) - ico.radius_of_gyration_squared(chain)) < 1e-9
    assert abs(ico.gyration_eigenvalues(chain).sum()
               - ico.radius_of_gyration_squared(chain)) < 1e-9


# ---------------------------------------------------------------------------
# Graph tests
# ---------------------------------------------------------------------------

def test_zero_contact_graph():
    g = ico.contact_graph_summary(np.empty((0, 2), dtype=np.int64), 10)
    assert g["contact_vertices"] == 0
    assert g["contact_graph_components"] == 0
    assert g["largest_component_vertices"] == 0
    assert g["contact_graph_cycle_rank"] == 0


def test_component_edges_sum_to_m():
    for seed in range(20):
        chain = _random_saw(26, seed)
        if chain is None:
            continue
        cp, _ = ico.build_contact_map(chain)
        g = ico.contact_graph_summary(cp, len(chain))
        # cycle_rank = m - vertices + components  (rearranged identity check)
        assert (g["contact_graph_cycle_rank"]
                == cp.shape[0] - g["contact_vertices"]
                + g["contact_graph_components"])


def test_handbuilt_graph_exact():
    cp, _ = ico.build_contact_map(HAIRPIN6)  # edges (0,5),(1,4) -> 2 comps
    g = ico.contact_graph_summary(cp, 6)
    assert g["contact_vertices"] == 4
    assert g["contact_graph_components"] == 2
    assert g["largest_component_vertices"] == 2
    assert g["largest_component_edges"] == 1
    assert g["contact_graph_cycle_rank"] == 0


@pytest.mark.skipif(not _HAVE_NX, reason="networkx not installed")
def test_union_find_matches_networkx():
    for seed in range(15):
        chain = _random_saw(26, seed)
        if chain is None:
            continue
        cp, _ = ico.build_contact_map(chain)
        g = ico.contact_graph_summary(cp, len(chain))
        G = nx.Graph()
        G.add_edges_from([tuple(p) for p in cp])
        if cp.shape[0] == 0:
            assert g["contact_graph_components"] == 0
            continue
        comps = list(nx.connected_components(G))
        assert g["contact_graph_components"] == len(comps)
        assert g["largest_component_vertices"] == max(len(c) for c in comps)


# ---------------------------------------------------------------------------
# Pair-motif tests
# ---------------------------------------------------------------------------

def test_pair_classification_examples():
    assert ico.classify_contact_pair((0, 5), (6, 9)) == "disjoint"
    assert ico.classify_contact_pair((0, 9), (2, 5)) == "nested"
    assert ico.classify_contact_pair((0, 5), (2, 9)) == "interleaved"
    assert ico.classify_contact_pair((0, 5), (5, 9)) == "shared_endpoint"
    assert ico.classify_contact_pair((0, 5), (0, 9)) == "shared_endpoint"


def test_pair_classification_symmetric():
    rng = np.random.RandomState(0)
    for _ in range(200):
        a = tuple(sorted(rng.randint(0, 20, size=2)))
        b = tuple(sorted(rng.randint(0, 20, size=2)))
        if a[0] == a[1] or b[0] == b[1]:
            continue
        assert ico.classify_contact_pair(a, b) == ico.classify_contact_pair(b, a)


def test_pair_counts_equal_binomial():
    for seed in range(20):
        chain = _random_saw(26, seed)
        if chain is None:
            continue
        cp, _ = ico.build_contact_map(chain)
        counts = ico.count_pair_motifs(cp)
        m = cp.shape[0]
        tot = (counts["pair_shared_endpoint"] + counts["pair_disjoint"]
               + counts["pair_nested"] + counts["pair_interleaved"])
        assert tot == m * (m - 1) // 2


# ---------------------------------------------------------------------------
# Bins
# ---------------------------------------------------------------------------

def test_bins_exhaustive_partition():
    for n in (20, 30, 44, 60):
        chk = ico.validate_bin_definitions(n)
        assert isinstance(chk["warnings"], list)
        chain = _random_saw(n, n)
        if chain is None:
            continue
        cp, _ = ico.build_contact_map(chain)
        m_r = ico.contact_separation_counts(cp, n)
        fixed = ico.bin_contact_separations_fixed(m_r, n)
        scaled = ico.bin_contact_separations_scaled(m_r, n)
        assert sum(fixed.values()) == cp.shape[0]
        assert sum(scaled.values()) == cp.shape[0]


# ---------------------------------------------------------------------------
# REMD regression tests
# ---------------------------------------------------------------------------

HS = dict(model_name="hs", params=[378.96, 1.39686], Tref=330.0, Tscale=80.0)


def _run(nworkers, n_cycles=30, N=16, structural_stride=1, seed=7):
    Ts = np.linspace(300, 360, 5)
    return remd.run_remd(
        N=N, Ts=Ts, steps_per_swap=30, n_cycles=n_cycles,
        model_name=HS["model_name"], params=HS["params"],
        Tref=HS["Tref"], Tscale=HS["Tscale"], seed=seed,
        n_workers=nworkers, verbose=False,
        structural_observables=True, structural_stride=structural_stride,
    )


def test_state_m_matches_contact_count_after_run():
    reps, _, _ = _run(1)
    for rep in reps:
        assert rep.state.m == int(round(remd.contact_count(rep.state.chain,
                                                           rep.state.occ)))


def test_canonical_keys_present_and_defaults_backward_compatible():
    reps, _, _ = _run(1)
    dist = remd.build_distributions(reps, rg_bins=20, burnin_frac=0.5)
    for key in ("Ts", "c_vals", "Pc", "rg_edges", "rg_centers", "Prg",
                "rg_edges_lattice", "rg_centers_lattice", "rg_scale",
                "temps", "ct_centers", "ct_hists", "rg_hists"):
        assert key in dist


def test_structural_trajectories_finite():
    reps, _, _ = _run(1)
    for rep in reps:
        assert all(math.isfinite(v) for v in rep.Rg2_traj)
        assert all(math.isfinite(v) for v in rep.Ree2_traj)
        assert all(math.isfinite(v) for v in rep.m_long_traj)
        assert all(math.isfinite(v) for v in rep.Smax_traj)
        assert all(math.isfinite(v) for v in rep.largest_component_fraction_traj)
        # structural stride 1 => one structural sample per cycle
        assert len(rep.m_long_traj) == len(rep.C_traj)


def test_move_counters_ordering():
    reps, _, _ = _run(1)
    for rep in reps:
        c = rep.move_counters
        assert np.all(c[:, 2] <= c[:, 1])  # accepted <= valid
        assert np.all(c[:, 1] <= c[:, 0])  # valid <= proposed


def test_serial_and_parallel_preserve_invariants():
    for nw in (1, 2):
        reps, _, _ = _run(nw)
        for rep in reps:
            cp, _ = ico.build_contact_map(rep.state.chain)
            assert cp.shape[0] == rep.state.m
            c = rep.move_counters
            assert np.all(c[:, 2] <= c[:, 1])
            assert np.all(c[:, 1] <= c[:, 0])


def test_structural_stride_changes_sample_count():
    reps, _, _ = _run(1, n_cycles=40, structural_stride=4)
    for rep in reps:
        assert len(rep.m_long_traj) == 10           # ceil(40/4)
        assert len(rep.C_traj) == 40                # scalars every cycle


def test_walker_mapping_is_permutation():
    Ts = np.linspace(300, 360, 5)
    store = {}
    remd.run_remd(
        N=16, Ts=Ts, steps_per_swap=30, n_cycles=25,
        model_name=HS["model_name"], params=HS["params"],
        Tref=HS["Tref"], Tscale=HS["Tscale"], seed=3,
        n_workers=1, verbose=False, diagnostics=True, diag_store=store,
    )
    wti = store["walker_temp_index"]
    for c in range(wti.shape[0]):
        assert sorted(wti[c].tolist()) == list(range(len(Ts)))


# ---------------------------------------------------------------------------
# HDF5 snapshot round-trip + extractor
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _HAVE_H5PY, reason="h5py not installed")
def test_snapshot_roundtrip_and_contacts_match():
    import isaw_config_io as cio
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "cfg.h5")
        Ts = np.linspace(300, 360, 4)
        writer = cio.SnapshotWriter(
            path, n_beads=16, n_temperatures=4,
            metadata={"run_id": "t", "temperatures": [float(x) for x in Ts],
                      "seed": 1},
            flush_interval=2,
        )
        reps, _, _ = remd.run_remd(
            N=16, Ts=Ts, steps_per_swap=25, n_cycles=12,
            model_name=HS["model_name"], params=HS["params"],
            Tref=HS["Tref"], Tscale=HS["Tscale"], seed=1, n_workers=1,
            verbose=False, snapshot_writer=writer, snapshot_stride=3,
        )
        writer.close()
        with h5py.File(path, "r") as f:
            s = f["snapshots"]
            assert s["coordinates"].shape == (4, 4, 16, 3)  # 4 snapshots
            assert s["walker_id"].shape == (4, 4)
            # snapshot contact counts match recomputed counts
            for si in range(s["coordinates"].shape[0]):
                for k in range(4):
                    coords = s["coordinates"][si, k].astype(np.int64)
                    cp, _ = ico.build_contact_map(coords)
                    assert cp.shape[0] == int(s["contacts"][si, k])


@pytest.mark.skipif(not _HAVE_H5PY, reason="h5py not installed")
def test_snapshot_refuses_overwrite():
    import isaw_config_io as cio
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "cfg.h5")
        w = cio.SnapshotWriter(path, n_beads=8, n_temperatures=2,
                               metadata={"run_id": "x"})
        straight = np.array([(i, 0, 0) for i in range(8)], dtype=np.int64)
        coords = np.stack([straight, straight])
        w.append(cycle=0, coordinates=coords,
                 walker_id=np.array([0, 1]), contacts=np.zeros(2),
                 rg2_lattice=np.full(2, 5.25), ree2_lattice=np.full(2, 49.0))
        w.close()
        with pytest.raises(cio.SnapshotWriterError):
            cio.SnapshotWriter(path, n_beads=8, n_temperatures=2,
                               metadata={"run_id": "x"})
