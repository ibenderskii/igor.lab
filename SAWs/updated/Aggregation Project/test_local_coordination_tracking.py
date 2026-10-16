"""Per-bead (k_intra, k_inter) degree cache and coordination observables.

Covers the cached local-cooperativity hot path against the unchanged oracles,
the exhaustive single-chain cross-check against remd_uniform_chain_2_new,
bit-identity of every pre-existing output against golden runs of the code
before this change, and the new coordination outputs and their invariants.
"""
from __future__ import annotations

import csv
import json
import math
import os
import random
import subprocess
import sys
import tempfile

import numpy as np
import pytest

import multichain_config_io as mcio
import multichain_contacts as mcc
import multichain_moves as mvs
import multichain_observables as obs
import multichain_state as mcs
import remd_multichain as rm
import remd_uniform_chain_2_new as remd

HERE = os.path.dirname(os.path.abspath(__file__))
LOCAL = "local_coordination_saturation"
P_LOCAL = [700.0, 2.3, 1.2, 0.4]
Q_SAT = 0.4
TREF, TSCALE = 320.0, 80.0
WEIGHTED = (1.0, 0.25)
PROPOSERS = (mvs.propose_pivot, mvs.propose_crankshaft, mvs.propose_end,
             mvs.propose_translation, mvs.propose_reptation,
             mvs.propose_chain_rotation)
# Dense small boxes (wraparound neighbours, every chain crosses the boundary)
# plus one dilute box.
BOXES = [(2, 6, 3, 1), (4, 6, 4, 2), (3, 5, 4, 3), (4, 8, 5, 4), (3, 10, 8, 5)]


def _weighted_S(state, weights):
    """Independent oracle: sum_i g(w_intra*k_intra_i + w_inter*k_inter_i)."""
    return sum(mcc.cooperative_g(weights[0] * a + weights[1] * e, Q_SAT)
               for a, e in mcc.contact_degrees_split(state).tolist())


def _run(model, params, n_workers=1, **kw):
    base = dict(n_chains=3, chain_length=8, box_size=8,
                Ts=np.linspace(305.0, 350.0, 4), local_sweeps_per_swap=1,
                translation_sweeps_per_swap=1, n_cycles=16, model_name=model,
                params=params, Tref=TREF, Tscale=TSCALE, seed=31,
                verbose=False, n_workers=n_workers)
    base.update(kw)
    return rm.run_remd_multichain(**base)


# ---------------------------------------------------------------------------
# 1. Cache vs oracle
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("weights", [rm.DEFAULT_DEGREE_WEIGHTS, WEIGHTED])
@pytest.mark.parametrize("M,N,L,seed", BOXES)
def test_cache_matches_oracle_after_every_accepted_move(M, N, L, seed, weights):
    state = mcs.initialize_dispersed_state(M, N, L, seed=seed, track_degrees=True)
    counters = mvs.new_move_counters()
    crossed = False
    for sweep in range(60):
        # debug_contacts asserts cache == oracle after EVERY accepted move.
        rm.mc_sweep(state, counters, 340.0, LOCAL, P_LOCAL, TREF, TSCALE, 1.0,
                    1.0, M * N, M, random.Random(1000 * seed + sweep),
                    n_reptation=M, n_rotation=M, debug_contacts=True,
                    degree_weights=weights)
        crossed |= bool(np.any((state.coords_unwrapped < 0)
                               | (state.coords_unwrapped >= L)))
    assert np.all(counters[:, 3] > 0), counters[:, 3]  # all six move types
    assert crossed
    mcs.validate_state(state)
    u = rm.reduced_contact_potential_state(
        state, 340.0, LOCAL, P_LOCAL, TREF, TSCALE, 1.0, 1.0,
        cooperativity="local", degree_weights=weights)
    b = remd.reduced_bias(LOCAL, P_LOCAL, 340.0, TREF, TSCALE)
    assert u == pytest.approx(b * state.counts.total
                              - P_LOCAL[2] * _weighted_S(state, weights),
                              abs=1e-10)


@pytest.mark.parametrize("M,N,L,seed", BOXES)
def test_cached_delta_matches_the_oracles_for_every_proposal(M, N, L, seed):
    state = mcs.initialize_dispersed_state(M, N, L, seed=seed, track_degrees=True)
    rng = random.Random(seed)
    n_checked = 0
    for i in range(1200):
        prop = PROPOSERS[i % 6](state, rng)
        if not prop.ok:
            continue
        changes = mcc.trial_degree_changes(state, prop.moved, prop.new_sites)
        delta = mvs.proposal_delta(state, prop)
        assert mcc.contact_delta_from_degree_changes(state, changes) == delta
        # Default weights: bit-identical to the unchanged oracle delta.
        assert (mcc.delta_cooperative_sum_cached(state, changes, Q_SAT)
                == mcc.delta_cooperative_sum(state, prop.moved, prop.new_sites,
                                             Q_SAT))
        d_w = mcc.delta_cooperative_sum_cached(state, changes, Q_SAT, WEIGHTED)
        s_before = _weighted_S(state, WEIGHTED)
        # Athermal: apply every valid proposal so the state keeps moving.
        mvs.apply_proposal(state, prop, delta, degree_changes=changes)
        assert _weighted_S(state, WEIGHTED) - s_before == pytest.approx(
            d_w, abs=1e-12)
        mcc.assert_counts_match(state, prop.move_type)
        n_checked += 1
    assert n_checked > 300


def test_apply_without_changes_recomputes_the_cache():
    state = mcs.initialize_dispersed_state(4, 6, 4, seed=8, track_degrees=True)
    rng = random.Random(2)
    for i in range(600):
        prop = PROPOSERS[i % 6](state, rng)
        if prop.ok:
            mvs.apply_proposal(state, prop, mvs.proposal_delta(state, prop))
    mcc.assert_counts_match(state)


def test_full_potential_reads_the_cache_bit_identically():
    state = mcs.initialize_dispersed_state(4, 8, 6, seed=9)
    assert state.degrees is None
    cached = state.copy()
    mcc.attach_degree_cache(cached)
    for T in (305.0, 335.0):
        u_oracle = rm.reduced_contact_potential_state(
            state, T, LOCAL, P_LOCAL, TREF, TSCALE, 1.0, 1.0)
        assert rm.reduced_contact_potential_state(
            cached, T, LOCAL, P_LOCAL, TREF, TSCALE, 1.0, 1.0) == u_oracle
        b = remd.reduced_bias(LOCAL, P_LOCAL, T, TREF, TSCALE)
        assert u_oracle == (b * state.counts.total
                            - P_LOCAL[2] * mcc.cooperative_sum(state, Q_SAT))


def test_degree_weights_are_validated():
    with pytest.raises(ValueError):
        rm._validate_degree_weights((1.0, -0.5), LOCAL, "local")
    with pytest.raises(ValueError):
        rm._validate_degree_weights((1.0, float("nan")), LOCAL, "local")
    with pytest.raises(ValueError):  # weights act on the local term only
        rm._validate_degree_weights((1.0, 0.5), "hs", None)
    with pytest.raises(ValueError):
        rm._validate_degree_weights((1.0, 0.5),
                                    "saturating_cooperative_contact", "global")
    assert rm._validate_degree_weights(
        (1, 0.5), "saturating_cooperative_contact", "local") == (1.0, 0.5)


# ---------------------------------------------------------------------------
# 2. Exhaustive single-chain cross-check (all 16,926 six-step SAWs)
# ---------------------------------------------------------------------------

def _all_saws(n_steps):
    walks = []

    def grow(walk, occ):
        if len(walk) == n_steps + 1:
            walks.append(list(walk))
            return
        x, y, z = walk[-1]
        for dx, dy, dz in mcs.NN6:
            nxt = (x + dx, y + dy, z + dz)
            if nxt not in occ:
                walk.append(nxt)
                occ.add(nxt)
                grow(walk, occ)
                occ.discard(nxt)
                walk.pop()

    grow([(0, 0, 0)], {(0, 0, 0)})
    return walks


def test_exhaustive_single_chain_cross_script_check():
    walks = _all_saws(6)
    assert len(walks) == 16926
    N = 7
    L = 2 * N + 1
    temps = (300.0, 320.0, 345.0)
    max_diff = 0.0
    for walk in walks:
        state = mcs.make_state(np.array([walk], dtype=np.int64), L,
                               track_degrees=True)
        m, hist = remd.contact_count_and_coordination_histogram(walk, set(walk))
        assert state.counts.intra == m and state.counts.inter == 0
        assert np.all(state.degrees[:, 1] == 0)
        assert np.bincount(state.degrees[:, 0], minlength=7).tolist() == list(hist)
        assert mcc.degree_histogram(state) == tuple(hist)
        for T in temps:
            u_mc = rm.reduced_contact_potential_state(
                state, T, LOCAL, P_LOCAL, TREF, TSCALE, 1.0, 1.0)
            u_ref = remd.reduced_contact_potential_state(
                m, hist, T, LOCAL, P_LOCAL, TREF, TSCALE, n_beads=N)
            max_diff = max(max_diff, abs(u_mc - u_ref))
    print(f"\nmax |u_multichain - u_single_chain| over 16926 SAWs = {max_diff:.3e}")
    assert max_diff <= 1e-12


# ---------------------------------------------------------------------------
# 3. Bit-identity against golden outputs of the code before this change
# ---------------------------------------------------------------------------

GOLDEN = os.environ.get("AGG_GOLDEN_DIR",
                        os.path.join(tempfile.gettempdir(), "agg_golden"))
GOLDEN_COMMON = [
    "--n-chains", "4", "--N", "12", "--box-size", "14", "--Tmin", "305",
    "--Tmax", "350", "--nT", "4", "--n-cycles", "40", "--seed", "2026",
    "--save-configurations", "--diagnostics", "--diagnostic-trajectories",
    "--debug-contacts", "--no-plots"]
_P = ["--Tref", "320", "--Tscale", "80"]
GOLDEN_MODELS = {
    "local_A0pos": ["--fit-summary-json", "fit_summary_local_A0pos.json"],
    "local_A0zero": ["--fit-summary-json", "fit_summary_local_A0zero.json"],
    "hs": ["--model", "hs", "--params", "700,2.3"] + _P,
    "hs_m2_hs": ["--model", "hs_m2_hs", "--params", "700,2.3,50,0.1"] + _P,
    "satcoop_global": ["--model", "saturating_cooperative_contact", "--params",
                       "700,2.3,1.2,0.4", "--cooperativity", "global"] + _P,
    "satcoop_local": ["--model", "saturating_cooperative_contact", "--params",
                      "700,2.3,1.2,0.4", "--cooperativity", "local"] + _P,
}
# Provenance that legitimately differs between runs, plus the deliberately
# bumped schema-version stamps (output 4 -> 5, snapshot 1 -> 2).
VOLATILE = {"timestamp", "wall_time_seconds", "git_commit", "output_files",
            "fit_summary_json", "n_workers", "run_id", "start_time", "end_time",
            "schema_version", "multichain_output_schema_version",
            "snapshot_schema_version"}


def _json_contains(old, new):
    """Every old value is unchanged in new; dicts may gain keys, lists may grow."""
    if isinstance(old, dict):
        return isinstance(new, dict) and all(
            k in new and _json_contains(v, new[k]) for k, v in old.items()
            if k not in VOLATILE)
    if isinstance(old, list):
        return (isinstance(new, list) and len(old) <= len(new)
                and all(_json_contains(a, b) for a, b in zip(old, new)))
    return old == new


def _assert_preexisting_outputs_identical(old_dir, new_dir):
    import h5py

    for name in ("run_distributions.npz", "run_diagnostic_trajectories.npz"):
        with np.load(os.path.join(old_dir, name), allow_pickle=True) as a, \
                np.load(os.path.join(new_dir, name), allow_pickle=True) as b:
            for k in a.files:
                if k in VOLATILE:
                    continue
                assert k in b.files, (name, k)
                assert np.array_equal(a[k], b[k], equal_nan=a[k].dtype.kind == "f"), (name, k)
    for name in sorted(f for f in os.listdir(old_dir) if f.endswith(".csv")):
        with open(os.path.join(old_dir, name), newline="") as fa, \
                open(os.path.join(new_dir, name), newline="") as fb:
            ra, rb = list(csv.reader(fa)), list(csv.reader(fb))
        width = len(ra[0])
        new_rows = iter(rb)  # old rows appear in order; new columns appended
        for row in ra:
            assert any(row == r[:width] for r in new_rows), (name, row[:3])
    for name in ("run_run_summary.json", "run_diagnostics.json"):
        with open(os.path.join(old_dir, name)) as fa, \
                open(os.path.join(new_dir, name)) as fb:
            assert _json_contains(json.load(fa), json.load(fb)), name
    with h5py.File(os.path.join(old_dir, "run_configurations.h5"), "r") as fa, \
            h5py.File(os.path.join(new_dir, "run_configurations.h5"), "r") as fb:
        for group in ("snapshots", "metadata"):
            for k in fa[group]:
                assert np.array_equal(fa[group][k][()], fb[group][k][()]), (group, k)
            for k, v in fa[group].attrs.items():
                if k not in VOLATILE:
                    assert np.array_equal(np.asarray(v),
                                          np.asarray(fb[group].attrs[k])), (group, k)


@pytest.mark.skipif(not os.path.isdir(GOLDEN),
                    reason=f"golden outputs not found at {GOLDEN} "
                           f"(set AGG_GOLDEN_DIR)")
@pytest.mark.parametrize("mode", ["serial", "w2"])
@pytest.mark.parametrize("tag", sorted(GOLDEN_MODELS))
def test_preexisting_outputs_bit_identical_to_golden(tag, mode, tmp_path):
    golden = os.path.join(GOLDEN, tag, mode)
    if not os.path.isdir(golden):
        pytest.skip(f"missing golden run {golden}")
    args = [a if not a.startswith("fit_summary_") else
            os.path.join(GOLDEN, "inputs", a) for a in GOLDEN_MODELS[tag]]
    proc = subprocess.run(
        [sys.executable, os.path.join(HERE, "remd_multichain.py"), *GOLDEN_COMMON,
         *args, "--n-workers", "2" if mode == "w2" else "1",
         "--out-prefix", str(tmp_path / "run")],
        cwd=HERE, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr[-2000:]
    _assert_preexisting_outputs_identical(golden, str(tmp_path))


# ---------------------------------------------------------------------------
# 4-5, 7. Run-level identities
# ---------------------------------------------------------------------------

def test_local_A0_zero_reproduces_hs_bit_for_bit():
    reps_l, sp_l, sa_l, wh_l = _run(LOCAL, [700.0, 2.3, 0.0, 0.4])
    reps_h, sp_h, sa_h, wh_h = _run("hs", [700.0, 2.3])
    assert np.array_equal(sp_l, sp_h) and np.array_equal(sa_l, sa_h)
    assert np.array_equal(wh_l, wh_h)
    for a, b in zip(reps_l, reps_h):
        assert a.m_intra_traj == b.m_intra_traj
        assert a.m_inter_traj == b.m_inter_traj
        assert a.u_traj == b.u_traj
        assert np.array_equal(a.move_counters, b.move_counters)
        assert a.degree_hist_traj and all(
            np.array_equal(x, y) for x, y in zip(a.degree_hist_traj,
                                                 b.degree_hist_traj))


def _assert_replicas_identical(r1, r2):
    for a, b in zip(r1, r2):
        assert np.array_equal(a.state.coords_unwrapped, b.state.coords_unwrapped)
        assert np.array_equal(a.state.degrees, b.state.degrees)
        assert np.array_equal(a.state.degree_joint, b.state.degree_joint)
        for attr in ("u_traj", "m_intra_traj", "m_inter_traj", "coop_sum_traj",
                     "u_linear_traj", "u_coop_traj", "coord_n_post"):
            assert getattr(a, attr) == getattr(b, attr), attr
        assert np.array_equal(np.array(a.degree_hist_traj),
                              np.array(b.degree_hist_traj))
        for attr in ("coord_joint", "coord_k_intra_by_position",
                     "coord_k_inter_by_position"):
            assert np.array_equal(getattr(a, attr), getattr(b, attr)), attr


@pytest.mark.parametrize("weights", [rm.DEFAULT_DEGREE_WEIGHTS, WEIGHTED])
def test_serial_and_two_workers_are_bit_identical(weights):
    r1, sp1, sa1, wh1 = _run(LOCAL, P_LOCAL, n_workers=1, degree_weights=weights,
                             n_cycles=8)
    r2, sp2, sa2, wh2 = _run(LOCAL, P_LOCAL, n_workers=2, degree_weights=weights,
                             n_cycles=8)
    assert np.array_equal(sa1, sa2) and np.array_equal(wh1, wh2)
    _assert_replicas_identical(r1, r2)


@pytest.mark.parametrize("model,params,coop", [
    ("hs", [700.0, 2.3], None),
    ("hs_m2_hs", [700.0, 2.3, 50.0, 0.1], None),
    ("saturating_cooperative_contact", P_LOCAL, "global"),
])
def test_non_local_runs_never_build_the_degree_cache(model, params, coop,
                                                     monkeypatch):
    def forbidden(*_a, **_k):
        raise AssertionError("degree cache used by a non-local Hamiltonian")

    for name in ("attach_degree_cache", "trial_degree_changes",
                 "apply_degree_changes"):
        monkeypatch.setattr(mcc, name, forbidden)
    reps, *_ = _run(model, params, cooperativity=coop, debug_contacts=True)
    for rep in reps:
        assert rep.state.degrees is None and rep.state.degree_joint is None
        assert len(rep.degree_hist_traj) == 16 and rep.coop_sum_traj == []


def test_weights_change_the_local_hamiltonian_only_through_g():
    state = mcs.initialize_dispersed_state(4, 8, 5, seed=4, track_degrees=True)
    assert state.counts.inter > 0
    u = {w: rm.reduced_contact_potential_state(
        state, 330.0, LOCAL, P_LOCAL, TREF, TSCALE, 1.0, 1.0,
        degree_weights=w) for w in (rm.DEFAULT_DEGREE_WEIGHTS, (1.0, 0.0))}
    b = remd.reduced_bias(LOCAL, P_LOCAL, 330.0, TREF, TSCALE)
    intra_only = sum(mcc.cooperative_g(a, Q_SAT)
                     for a in state.degrees[:, 0].tolist())
    assert u[(1.0, 0.0)] == pytest.approx(
        b * state.counts.total - P_LOCAL[2] * intra_only, abs=1e-12)
    assert u[(1.0, 0.0)] != u[rm.DEFAULT_DEGREE_WEIGHTS]


# ---------------------------------------------------------------------------
# Coordination observables: invariants on real runs, outputs, snapshots
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def local_run():
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "snap.h5")
    writer = mcio.MultiChainSnapshotWriter(
        path, n_chains=3, chain_length=8, n_temperatures=4,
        metadata={"M": 3, "N": 8, "L": 6})
    out = _run(LOCAL, P_LOCAL, box_size=6, n_cycles=30, kappa_bend=0.3,
               snapshot_writer=writer, degree_weights=WEIGHTED, burnin_frac=0.4)
    writer.mark_complete()
    writer.close()
    return out, path


def test_coordination_invariants_on_a_real_run(local_run):
    (reps, *_), _ = local_run
    M, N = 3, 8
    start = obs._post_burnin(30, 0.4)
    for rep in reps:
        H = np.array(rep.degree_hist_traj)
        m_total = np.array(rep.m_intra_traj) + np.array(rep.m_inter_traj)
        assert np.all(H.sum(axis=1) == M * N)
        assert np.array_equal(H @ np.arange(7), 2 * m_total)
        assert np.all(H[:, 6] == 0)
        assert rep.coord_start_cycle == start and rep.coord_n_post == 30 - start
        a = np.arange(7)
        assert int(a @ rep.coord_joint.sum(axis=1)) == 2 * sum(rep.m_intra_traj[start:])
        assert int(a @ rep.coord_joint.sum(axis=0)) == 2 * sum(rep.m_inter_traj[start:])
        mean_k = (rep.coord_k_intra_by_position + rep.coord_k_inter_by_position) \
            / (M * rep.coord_n_post)
        assert mean_k.sum() * M == pytest.approx(2 * m_total[start:].mean(),
                                                 rel=1e-12)
        # u = u_linear + u_coop + kappa_bend * n_bend
        resid = (np.array(rep.u_traj) - 0.3 * np.array(rep.n_bend_traj)
                 - np.array(rep.u_linear_traj) - np.array(rep.u_coop_traj))
        assert np.max(np.abs(resid)) < 1e-12


def test_per_bead_degree_bounds_on_snapshots(local_run):
    _, path = local_run
    deg = obs.snapshot_degrees(path)  # (n, nT, M*N, 2)
    k = deg.sum(axis=-1).reshape(deg.shape[0], deg.shape[1], 3, 8)
    assert k[..., 1:-1].max() <= 4  # interior beads: two bonded neighbours
    assert k[..., [0, -1]].max() <= 5  # end beads: one bonded neighbour


def test_snapshot_v2_round_trip_and_rebuilt_degrees(local_run):
    import h5py

    (reps, *_), path = local_run
    with h5py.File(path, "r") as f:
        assert int(f["metadata"].attrs["schema_version"]) == 2
        hist = f["snapshots"]["degree_histogram"][()]
        S = f["snapshots"]["cooperative_sum"][()]
    assert hist.shape == (30, 4, 7) and S.shape == (30, 4)
    for k, rep in enumerate(reps):
        assert np.array_equal(hist[:, k], np.array(rep.degree_hist_traj))
        assert S[:, k].tolist() == rep.coop_sum_traj
    deg = obs.snapshot_degrees(path)
    rebuilt = np.array([[np.bincount(d.sum(axis=1), minlength=7) for d in row]
                        for row in deg])
    assert np.array_equal(rebuilt, hist)
    joints = [[mcs.degree_joint_from_degrees(d) for d in row] for row in deg]
    assert [[mcc.cooperative_sum_from_joint(j, Q_SAT, WEIGHTED) for j in row]
            for row in joints] == S.tolist()


def test_snapshot_v1_files_still_read(tmp_path):
    import h5py

    states = [mcs.initialize_dispersed_state(2, 6, 7, seed=s) for s in (1, 2)]
    path = str(tmp_path / "v1.h5")
    writer = mcio.MultiChainSnapshotWriter(
        path, n_chains=2, chain_length=6, n_temperatures=2,
        metadata={"M": 2, "N": 6, "L": 7})
    writer.append(cycle=0, coordinates=np.stack([s.coords_unwrapped for s in states]),
                  walker_id=np.arange(2),
                  m_intra=np.array([s.counts.intra for s in states]),
                  m_inter=np.array([s.counts.inter for s in states]),
                  mean_chain_rg2=np.ones(2), largest_cluster_size=np.ones(2))
    writer.mark_complete()
    writer.close()
    with h5py.File(path, "r+") as f:  # stamp it exactly as a v1 file
        f["metadata"].attrs["schema_version"] = 1
        f["snapshots"].attrs["schema_version"] = 1
        assert "degree_histogram" not in f["snapshots"]
    deg = obs.snapshot_degrees(path)
    for k, s in enumerate(states):
        assert np.array_equal(deg[0, k], mcc.contact_degrees_split(s))


def test_distribution_keys_and_csv_columns(local_run, tmp_path):
    (reps, *_), _ = local_run
    Ts = [r.T for r in reps]
    dist = rm.build_run_distributions(reps, Ts, burnin_frac=0.4, rg_bins=10,
                                      rg_scale=1.0)
    nT, N = 4, 8
    assert dist["coord_degree_values"].tolist() == list(range(7))
    assert dist["coord_degree_hist"].shape == (nT, 30, 7)
    assert dist["coord_P_degree"].shape == (nT, 7)
    assert np.allclose(dist["coord_P_degree"].sum(axis=1), 1.0)
    assert np.allclose(dist["coord_P_joint"].sum(axis=(1, 2)), 1.0)
    assert dist["coord_joint_counts"].dtype == np.int64
    for key in ("coord_mean_degree_by_position",
                "coord_mean_degree_intra_by_position",
                "coord_mean_degree_inter_by_position"):
        assert dist[key].shape == (nT, N)
    assert np.allclose(dist["coord_mean_degree_by_position"],
                       dist["coord_mean_degree_intra_by_position"]
                       + dist["coord_mean_degree_inter_by_position"])
    assert dist["coord_S_mean"].shape == (nT,)
    results = rm.summarize_results(reps, Ts, 0.4, 1.0)
    mcio.save_results_csv(results, str(tmp_path / "r"))
    with open(tmp_path / "r_results.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert all(float(r["S_mean"]) > 0 for r in rows)
    assert all(math.isclose(sum(float(r[f"frac_k{k}"]) for k in range(7)), 1.0)
               for r in rows)

    reps_hs, *_ = _run("hs", [700.0, 2.3], n_cycles=6)
    res_hs = rm.summarize_results(reps_hs, Ts, 0.5, 1.0)
    mcio.save_results_csv(res_hs, str(tmp_path / "h"))
    with open(tmp_path / "h_results.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert all(r["S_mean"] == "" and r["u_coop_mean"] == "" for r in rows)
    assert all(r["degree_mean"] != "" for r in rows)
    assert "coord_S_mean" not in rm.build_run_distributions(
        reps_hs, Ts, burnin_frac=0.5, rg_bins=10, rg_scale=1.0)
