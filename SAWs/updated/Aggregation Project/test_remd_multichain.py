#!/usr/bin/env python3
"""pytest suite for remd_multichain (Stage 2: sampler, swap, REMD).

Run:  python -m pytest test_remd_multichain.py -q
"""
import math
import os
import random
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import multichain_state as mcs
import multichain_contacts as mcc
import multichain_moves as mvs
import multichain_diagnostics as mcd
import remd_uniform_chain_2_new as remd
import remd_multichain as rmc
from multichain_state import ContactCounts

MP = ("hs", [400.0, 1.3], 320.0, 80.0)  # model_name, params, Tref, Tscale


def straight_chain(n, axis=0, start=(0, 0, 0)):
    coords = np.tile(np.asarray(start, dtype=np.int64), (n, 1))
    coords[:, axis] += np.arange(n, dtype=np.int64)
    return coords


class _StubRng:
    """Deterministic rng stub: ``random()`` always returns a fixed value."""

    def __init__(self, value):
        self.value = float(value)
        self.calls = 0

    def random(self):
        self.calls += 1
        return self.value


def _two_parallel(N=5, L=20):
    """Two parallel straight chains offset by 1 in y -> m_inter = N, m_intra = 0."""
    c0 = straight_chain(N, axis=0, start=(0, 0, 0))
    c1 = straight_chain(N, axis=0, start=(0, 1, 0))
    return mcs.make_state(np.stack([c0, c1]), L)


def _two_separated(N=5, L=20):
    """Two well-separated straight chains -> m_inter = 0, m_intra = 0."""
    c0 = straight_chain(N, axis=0, start=(0, 0, 0))
    c1 = straight_chain(N, axis=0, start=(0, 6, 0))
    return mcs.make_state(np.stack([c0, c1]), L)


# ---------------------------------------------------------------------------
# Reduced potential: four lambda control modes (deterministic energy tests)
# ---------------------------------------------------------------------------

def test_reduced_potential_full_transferability():
    c = ContactCounts(3, 5)
    b = remd.reduced_bias(*MP[:2], 330.0, *MP[2:])
    u = rmc.reduced_potential_counts(c, 330.0, *MP, 1.0, 1.0)
    assert abs(u - b * (3 + 5)) < 1e-12


def test_athermal_all_configs_equal_potential():
    for c in (ContactCounts(0, 0), ContactCounts(3, 5), ContactCounts(9, 1)):
        assert rmc.reduced_potential_counts(c, 330.0, *MP, 0.0, 0.0) == 0.0


def test_independent_collapse_inter_irrelevant():
    # (1, 0): configs with equal intra but different inter have equal potential.
    c1 = ContactCounts(4, 1)
    c2 = ContactCounts(4, 9)
    u1 = rmc.reduced_potential_counts(c1, 330.0, *MP, 1.0, 0.0)
    u2 = rmc.reduced_potential_counts(c2, 330.0, *MP, 1.0, 0.0)
    assert u1 == u2


def test_interchain_only_intra_irrelevant():
    # (0, 1): configs with equal inter but different intra have equal potential.
    c1 = ContactCounts(2, 5)
    c2 = ContactCounts(8, 5)
    u1 = rmc.reduced_potential_counts(c1, 330.0, *MP, 0.0, 1.0)
    u2 = rmc.reduced_potential_counts(c2, 330.0, *MP, 0.0, 1.0)
    assert u1 == u2


def test_m1_lambda_intra_matches_single_chain():
    rng = np.random.RandomState(3)
    saw = mcs.generate_saw(16, rng)
    state = mcs.make_state(np.stack([saw]), 60)
    m = int(state.counts.intra)
    for T in (300.0, 345.0):
        u_mc = rmc.reduced_potential_counts(state.counts, T, *MP, 1.0, 1.0)
        u_ref = remd.reduced_potential(m, T, *MP)
        assert abs(u_mc - u_ref) < 1e-9


# ---------------------------------------------------------------------------
# Generalized swap rule with UNEQUAL intra/inter counts
# ---------------------------------------------------------------------------

def test_generalized_swap_manual_unequal_counts():
    ci = ContactCounts(3, 7)
    cj = ContactCounts(8, 1)
    Ti, Tj = 300.0, 350.0
    li, lin = 1.0, 0.5

    def u(c, T):
        return rmc.reduced_potential_counts(c, T, *MP, li, lin)
    expect = u(ci, Ti) + u(cj, Tj) - u(cj, Ti) - u(ci, Tj)
    got = rmc.swap_log_accept_counts(ci, cj, Ti, Tj, *MP, li, lin)
    assert abs(got - expect) < 1e-12


def test_swap_uses_both_counts_when_lambdas_differ():
    # Two configs with equal TOTAL but different split must give a different
    # swap log-acceptance when lambda_intra != lambda_inter (proves both counts
    # are used, not just the total).
    ci = ContactCounts(6, 0)  # total 6
    cj = ContactCounts(0, 6)  # total 6, different split
    la_equal = rmc.swap_log_accept_counts(ci, cj, 300.0, 350.0, *MP, 1.0, 1.0)
    la_split = rmc.swap_log_accept_counts(ci, cj, 300.0, 350.0, *MP, 1.0, 0.0)
    assert not math.isclose(la_equal, la_split)


# ---------------------------------------------------------------------------
# State exchange preserves counts and occupancy
# ---------------------------------------------------------------------------

def test_swap_state_exchange_preserves_counts_and_occupancy():
    sa = mcs.initialize_dispersed_state(2, 6, 10, seed=1)
    sb = mcs.initialize_dispersed_state(2, 6, 10, seed=2)
    ca, cb = sa.counts.as_tuple(), sb.counts.as_tuple()
    ra = rmc.MultiReplica(T=300.0, state=sa)
    rb = rmc.MultiReplica(T=320.0, state=sb)
    ra.state, rb.state = rb.state, ra.state  # the swap operation
    assert ra.state.counts.as_tuple() == cb
    assert rb.state.counts.as_tuple() == ca
    mcs.validate_state(ra.state)
    mcs.validate_state(rb.state)


# ---------------------------------------------------------------------------
# Short REMD runs: finiteness, walker permutation, worker equivalence
# ---------------------------------------------------------------------------

def _short_run(n_workers=1, lam=(1.0, 1.0), debug=True, seed=13):
    Ts = np.linspace(305, 350, 4)
    return rmc.run_remd_multichain(
        n_chains=2, chain_length=8, box_size=10, Ts=Ts,
        local_sweeps_per_swap=1, translation_sweeps_per_swap=1, n_cycles=15,
        model_name="hs", params=[400.0, 1.3], Tref=320.0, Tscale=80.0,
        lambda_intra=lam[0], lambda_inter=lam[1],
        seed=seed, n_workers=n_workers, verbose=False, debug_contacts=debug), Ts


def test_short_run_produces_finite_trajectories():
    (reps, sp, sa, wh), Ts = _short_run()
    for rep in reps:
        assert len(rep.u_traj) == 15
        assert all(math.isfinite(x) for x in rep.u_traj)
        assert all(math.isfinite(x) for x in rep.rg_traj)
        assert all(x >= 0 for x in rep.m_intra_traj)
        assert all(x >= 0 for x in rep.m_inter_traj)
        mcs.validate_state(rep.state)


def test_walker_identities_remain_permutation():
    (reps, sp, sa, wh), Ts = _short_run()
    assert wh.shape == (15, len(Ts))
    for row in wh:
        assert sorted(row.tolist()) == list(range(len(Ts)))


def test_serial_and_two_workers_identical():
    (r1, *_), _ = _short_run(n_workers=1)
    (r2, *_), _ = _short_run(n_workers=2)
    for a, b in zip(r1, r2):
        assert a.state.counts.as_tuple() == b.state.counts.as_tuple()
        assert np.array_equal(a.state.coords_unwrapped, b.state.coords_unwrapped)


def test_serial_and_two_workers_full_equivalence():
    # Every recorded quantity must match bit-for-bit between serial and 2-worker
    # runs at the same seed: coordinates, counts, all scalar and per-chain
    # trajectories, move counters, swap proposals/acceptances, walker histories.
    (r1, sp1, sa1, wh1), _ = _short_run(n_workers=1)
    (r2, sp2, sa2, wh2), _ = _short_run(n_workers=2)
    assert np.array_equal(sp1, sp2)
    assert np.array_equal(sa1, sa2)
    assert np.array_equal(wh1, wh2)
    for a, b in zip(r1, r2):
        assert np.array_equal(a.state.coords_unwrapped, b.state.coords_unwrapped)
        assert a.state.counts.as_tuple() == b.state.counts.as_tuple()
        assert np.array_equal(a.move_counters, b.move_counters)
        for attr in ("u_traj", "m_intra_traj", "m_inter_traj", "rg_traj",
                     "std_chain_rg_traj", "lcs_traj", "lcf_traj",
                     "n_clusters_traj"):
            assert np.allclose(np.asarray(getattr(a, attr), dtype=float),
                               np.asarray(getattr(b, attr), dtype=float)), attr
        # Per-chain Rg trajectories (list of (M,) arrays) match cycle-by-cycle.
        assert len(a.per_chain_rg_traj) == len(b.per_chain_rg_traj)
        for va, vb in zip(a.per_chain_rg_traj, b.per_chain_rg_traj):
            assert np.array_equal(va, vb)


# ---------------------------------------------------------------------------
# Deterministic Metropolis acceptance helper (Change 6)
# ---------------------------------------------------------------------------

def test_metropolis_accept_delta_four_cases():
    b = 1.0  # positive reduced bias
    # delta_u < 0 -> always accept (adding a favorable contact: delta_intra = -1).
    assert rmc.metropolis_accept_delta(-1, 0, b, 1.0, 1.0, 0.999) is True
    # delta_u = 0 -> accept.
    assert rmc.metropolis_accept_delta(0, 0, b, 1.0, 1.0, 0.999) is True
    # delta_u > 0 and r < exp(-delta_u) -> accept.  delta_u = 1 -> exp(-1)=0.3679.
    assert rmc.metropolis_accept_delta(1, 0, b, 1.0, 1.0, 0.10) is True
    # delta_u > 0 and r > exp(-delta_u) -> reject.
    assert rmc.metropolis_accept_delta(1, 0, b, 1.0, 1.0, 0.90) is False


def test_metropolis_accept_delta_lambda_modes():
    b = 1.0
    # (0,0): athermal -> delta_u = 0 regardless of counts -> always accept.
    assert rmc.metropolis_accept_delta(5, -3, b, 0.0, 0.0, 0.999) is True
    # (1,0): only intra matters.  delta_inter is ignored.
    assert rmc.metropolis_accept_delta(1, -9, b, 1.0, 0.0, 0.90) is False  # du=+1
    assert rmc.metropolis_accept_delta(-1, 9, b, 1.0, 0.0, 0.999) is True  # du=-1
    # (0,1): only inter matters.
    assert rmc.metropolis_accept_delta(-9, 1, b, 0.0, 1.0, 0.90) is False  # du=+1
    assert rmc.metropolis_accept_delta(9, -1, b, 0.0, 1.0, 0.999) is True  # du=-1
    # (1,1): both contribute (net favorable).
    assert rmc.metropolis_accept_delta(-1, -1, b, 1.0, 1.0, 0.999) is True


# ---------------------------------------------------------------------------
# The actual production swap function (Change 5)
# ---------------------------------------------------------------------------

def test_attempt_swap_guaranteed_acceptance_athermal():
    # Athermal (0,0): log_accept == 0 -> accept WITHOUT drawing a random number.
    a = rmc.MultiReplica(T=305.0, state=_two_separated())
    b = rmc.MultiReplica(T=350.0, state=_two_parallel())
    sa, sb = a.state, b.state
    stub = _StubRng(1.0)  # would force rejection if it were ever consulted
    accepted = rmc.attempt_swap(a, b, *MP, 0.0, 0.0, stub)
    assert accepted is True
    assert stub.calls == 0            # favorable branch short-circuits
    assert a.state is sb and b.state is sa  # states exchanged


def test_attempt_swap_deterministic_reject_and_accept():
    # Cold lane holds the LOW-contact state, hot lane the HIGH-contact state, so
    # swapping is unfavorable (log_accept < 0) under interchain attraction (0,1).
    def fresh():
        a = rmc.MultiReplica(T=305.0, state=_two_separated())
        b = rmc.MultiReplica(T=350.0, state=_two_parallel())
        return a, b

    a, b = fresh()
    la = rmc.swap_log_accept_counts(a.state.counts, b.state.counts, a.T, b.T,
                                    *MP, 0.0, 1.0)
    assert la < 0.0, "test setup expects an unfavorable swap"
    thresh = math.exp(la)

    # r > exp(la) -> reject, no state change.
    a, b = fresh()
    sa, sb = a.state, b.state
    assert rmc.attempt_swap(a, b, *MP, 0.0, 1.0, _StubRng(thresh + 0.05)) is False
    assert a.state is sa and b.state is sb

    # r < exp(la) -> accept, states exchanged; occupancy/counts stay valid.
    a, b = fresh()
    sa, sb = a.state, b.state
    ca, cb = sa.counts.as_tuple(), sb.counts.as_tuple()
    assert rmc.attempt_swap(a, b, *MP, 0.0, 1.0, _StubRng(max(thresh - 0.05, 0.0))) is True
    assert a.state is sb and b.state is sa
    assert a.state.counts.as_tuple() == cb and b.state.counts.as_tuple() == ca
    mcs.validate_state(a.state)
    mcs.validate_state(b.state)


def test_attempt_swap_uses_both_counts_on_split():
    # Two states with equal TOTAL contacts but different intra/inter split must
    # produce different swap behaviour when lambda_intra != lambda_inter.
    hairpin = np.array(
        [(0, 0, 0), (1, 0, 0), (2, 0, 0), (2, 1, 0), (1, 1, 0), (0, 1, 0)],
        dtype=np.int64)
    s_intra = mcs.make_state(np.stack([hairpin,
                                       straight_chain(6, start=(0, 6, 0))]), 20)
    s_inter = _two_parallel(N=6)
    a = rmc.MultiReplica(T=305.0, state=s_intra)
    b = rmc.MultiReplica(T=350.0, state=s_inter)
    la_equal = rmc.swap_log_accept_counts(a.state.counts, b.state.counts,
                                          a.T, b.T, *MP, 1.0, 1.0)
    la_split = rmc.swap_log_accept_counts(a.state.counts, b.state.counts,
                                          a.T, b.T, *MP, 1.0, 0.0)
    assert not math.isclose(la_equal, la_split)


# ---------------------------------------------------------------------------
# Independent per-lane initialization (Change 2)
# ---------------------------------------------------------------------------

def test_independent_initialization_distinct_and_deterministic():
    Ts = np.linspace(305, 350, 4)
    reps = rmc._build_independent_replicas(Ts, 3, 6, 12, base_seed=1234,
                                           cluster_contact_threshold=1)
    # Different lanes get different configurations for a fixed base seed.
    assert not np.array_equal(reps[0].state.coords_unwrapped,
                              reps[1].state.coords_unwrapped)
    # Initialization seeds are recorded and follow the documented stride.
    for lane, rep in enumerate(reps):
        assert rep.init_seed == 1234 + rmc.INITIALIZATION_SEED_STRIDE * lane
        assert rep.init_m_intra == rep.state.counts.intra
        assert rep.init_m_inter == rep.state.counts.inter
    # Deterministic across repeated construction.
    reps2 = rmc._build_independent_replicas(Ts, 3, 6, 12, base_seed=1234,
                                            cluster_contact_threshold=1)
    for a, b in zip(reps, reps2):
        assert np.array_equal(a.state.coords_unwrapped, b.state.coords_unwrapped)
        assert a.init_seed == b.init_seed


def test_independent_initialization_serial_vs_workers():
    # Serial and multiprocessing runs must begin from identical independent
    # starting states (initialization happens in the main process only).
    (r1, *_), _ = _short_run(n_workers=1)
    (r2, *_), _ = _short_run(n_workers=2)
    for a, b in zip(r1, r2):
        assert a.init_seed == b.init_seed
        assert a.init_m_intra == b.init_m_intra
        assert a.init_m_inter == b.init_m_inter
    # Lanes started from genuinely different configurations.
    seeds = [rep.init_seed for rep in r1]
    assert len(set(seeds)) == len(seeds)


# ---------------------------------------------------------------------------
# Contact-debug environment variable (Change 7)
# ---------------------------------------------------------------------------

def test_debug_contacts_env_var(monkeypatch):
    # With MULTICHAIN_DEBUG_CONTACTS set, mc_sweep asserts after every accepted
    # move even though debug_contacts=False is passed.  A corrupted cache is then
    # caught (proving the assertion actually runs).
    monkeypatch.setattr(mcc, "DEBUG_CONTACTS", True)
    state = mcs.initialize_dispersed_state(2, 6, 10, seed=2)
    state.counts.inter += 3  # corrupt the cache
    counters = mvs.new_move_counters()
    rng = random.Random(0)
    with pytest.raises(AssertionError):
        rmc.mc_sweep(state, counters, 330.0, "hs", [0.0, 4.0], 330.0, 40.0,
                     lambda_intra=0.0, lambda_inter=1.0,
                     n_local=200, n_translation=50, rng=rng, debug_contacts=False)


# ---------------------------------------------------------------------------
# Convergence diagnostics (Change 4)
# ---------------------------------------------------------------------------

def test_ess_iid_high_correlated_low():
    # Synthetic ESS: an IID series keeps ~all samples; a random walk (strongly
    # autocorrelated) keeps far fewer.  These are the exact primitives the
    # multi-chain diagnostics adapter reuses.
    rng = np.random.RandomState(0)
    n = 4000
    iid = rng.rand(n)
    walk = np.cumsum(rng.rand(n) - 0.5)
    ess_iid = remd.integrated_autocorr_time(iid)["ess"]
    ess_walk = remd.integrated_autocorr_time(walk)["ess"]
    assert ess_iid > 0.4 * n
    assert ess_walk < 0.1 * ess_iid


def test_known_early_late_drift():
    x = np.concatenate([np.zeros(200), np.ones(200)])
    dr = remd.early_late_drift(x)
    assert abs(dr["drift"] - 1.0) < 1e-9


def test_known_round_trips_primitive():
    path = [0, 1, 2, 1, 0, 1, 2, 1, 0]     # L,H,L,H,L on the extremes
    wd = remd.analyze_walker_trajectory(path, 3)
    assert wd["n_round_trips_low"] == 2
    assert wd["fraction_visited"] == 1.0


def _fake_diag_replica(T, n, M=2, seed=1):
    state = mcs.initialize_dispersed_state(M, 4, 8, seed=seed)
    rep = rmc.MultiReplica(T=float(T), state=state)
    rep.u_traj = list(np.linspace(0.0, 1.0, n))
    rep.m_intra_traj = [2] * n
    rep.m_inter_traj = [1] * n
    rep.f_inter_traj = [0.3] * n
    rep.rg_traj = [1.0] * n
    rep.rg2_traj = [1.0] * n
    rep.per_chain_rg_traj = [np.ones(M) for _ in range(n)]
    rep.std_chain_rg_traj = [0.0] * n
    rep.lcs_traj = [1] * n
    rep.lcf_traj = [0.5] * n
    rep.n_clusters_traj = [2] * n
    return rep


def test_diagnostics_coverage_and_round_trip_warnings():
    # Valid permutation history in which one walker never reaches the top lane:
    # coverage and round-trip warnings must both fire.
    nT, n = 3, 8
    Ts = np.linspace(305, 350, nT)
    reps = [_fake_diag_replica(T, n, seed=i + 1) for i, T in enumerate(Ts)]
    rows = [[2, 1, 0], [1, 2, 0]]
    wti = np.array([rows[c % 2] for c in range(n)], dtype=np.int64)
    diag = mcd.compute_multichain_diagnostics(
        reps, np.zeros(nT - 1, dtype=np.int64), np.zeros(nT - 1, dtype=np.int64),
        wti, Ts, burnin_frac=0.0)
    kinds = {w["type"] for w in diag["warnings"]}
    assert "temperature_coverage" in kinds
    assert "round_trips" in kinds
    # Walker 2 (column 2) never occupies the top lane.
    w2 = next(w for w in diag["walkers"] if w["walker"] == 2)
    assert w2["fraction_visited"] < 1.0
    assert diag["summary"]["total_round_trips_low"] == 0


def test_diagnostics_files_created_and_reloadable(tmp_path):
    (reps, sp, sa, wh), Ts = _short_run()
    diag = mcd.compute_multichain_diagnostics(
        reps, sp, sa, wh, Ts, burnin_frac=0.5, rg_scale=1.0)
    prefix = str(tmp_path / "diag")
    files = mcd.save_all_diagnostics(
        diag, reps, wh, Ts, out_prefix=prefix, burnin_frac=0.5, rg_scale=1.0,
        save_trajectories=True)
    import json as _json
    with open(files["diagnostics_json"]) as f:
        dj = _json.load(f)
    assert "lane_convergence" in dj and "warnings" in dj and "summary" in dj
    for key in ("convergence_csv", "round_trips_csv", "walker_occupancy_csv"):
        assert os.path.exists(files[key])
    # Diagnostic-trajectory NPZ: required arrays present, NONE object-typed.
    required = ("Ts", "burnin_start_cycle", "u_post", "m_intra_post",
                "m_inter_post", "mean_chain_rg_post", "mean_chain_rg_lattice_post",
                "per_chain_rg_post", "per_chain_rg_lattice_post",
                "std_chain_rg_post", "n_clusters_post",
                "largest_cluster_size_post", "largest_cluster_fraction_post",
                "walker_temp_index_post")
    with np.load(files["diagnostic_trajectories_npz"]) as data:
        nT = len(Ts)
        for key in required:
            assert key in data, f"missing trajectory array {key}"
            assert data[key].dtype != object, f"{key} is an object array"
        assert data["u_post"].shape[0] == nT
        assert data["per_chain_rg_post"].ndim == 3
        assert data["per_chain_rg_post"].shape[0] == nT


def test_diagnostic_trajectories_requires_diagnostics_cli():
    with pytest.raises(ValueError):
        args = rmc.build_arg_parser().parse_args(
            ["--diagnostic-trajectories", "--n-cycles", "2"])
        rmc._validate_cli(args)


def test_configuration_path_requires_save_configurations_cli():
    with pytest.raises(ValueError):
        args = rmc.build_arg_parser().parse_args(
            ["--configuration-path", "x.h5"])
        rmc._validate_cli(args)


def test_cli_rejects_nonincreasing_and_small_box():
    ap = rmc.build_arg_parser()
    with pytest.raises(ValueError):
        rmc._validate_cli(ap.parse_args(["--box-size", "2"]))
    with pytest.raises(ValueError):
        rmc._validate_cli(ap.parse_args(["--rg-bins", "0"]))
    with pytest.raises(ValueError):
        rmc._validate_cli(ap.parse_args(
            ["--save-configurations", "--snapshot-stride", "0"]))


# ---------------------------------------------------------------------------
# std_chain_rg scaling + init Rg unit labels (Part 1)
# ---------------------------------------------------------------------------

def test_std_chain_rg_scaling_in_diagnostic_npz(tmp_path):
    (reps, sp, sa, wh), Ts = _short_run()
    rg_scale = 0.5
    path = mcd.save_diagnostic_trajectories_npz(
        reps, wh, Ts, burnin_frac=0.5, out_prefix=str(tmp_path / "d"),
        rg_scale=rg_scale)
    with np.load(path) as d:
        assert "std_chain_rg_lattice_post" in d and "std_chain_rg_post" in d
        assert d["std_chain_rg_lattice_post"].dtype != object
        assert d["std_chain_rg_post"].dtype != object
        # std_chain_rg_post == rg_scale * std_chain_rg_lattice_post exactly.
        np.testing.assert_allclose(
            d["std_chain_rg_post"],
            (d["std_chain_rg_lattice_post"] * np.float32(rg_scale)).astype(np.float32))
        # mean/per-chain Rg scaling unchanged and consistent.
        np.testing.assert_allclose(
            d["mean_chain_rg_post"],
            (d["mean_chain_rg_lattice_post"] * np.float32(rg_scale)).astype(np.float32))


def test_init_mean_chain_rg_units():
    Ts = np.linspace(305, 350, 3)
    reps = rmc._build_independent_replicas(Ts, 3, 6, 12, base_seed=5,
                                           cluster_contact_threshold=1)
    rg_scale = 0.5
    meta = rmc._initial_states_metadata(reps, rg_scale)
    for rec, rep in zip(meta, reps):
        assert "init_mean_chain_rg_lattice" in rec and "init_mean_chain_rg" in rec
        assert abs(rec["init_mean_chain_rg_lattice"]
                   - rep.init_mean_chain_rg_lattice) < 1e-12
        assert abs(rec["init_mean_chain_rg"]
                   - rg_scale * rep.init_mean_chain_rg_lattice) < 1e-12
    # rg_scale=1 keeps the two equal.
    meta1 = rmc._initial_states_metadata(reps, 1.0)
    for rec in meta1:
        assert rec["init_mean_chain_rg"] == rec["init_mean_chain_rg_lattice"]


# ---------------------------------------------------------------------------
# Contact normalization arrays in the diagnostic NPZ (Part 2)
# ---------------------------------------------------------------------------

def test_normalized_contacts_in_diagnostic_npz(tmp_path):
    (reps, sp, sa, wh), Ts = _short_run()          # M = 2
    M = reps[0].state.n_chains
    path = mcd.save_diagnostic_trajectories_npz(
        reps, wh, Ts, burnin_frac=0.5, out_prefix=str(tmp_path / "n"),
        rg_scale=1.0)
    with np.load(path) as d:
        for k in ("m_intra_per_chain_post", "m_inter_pairs_per_chain_post",
                  "m_inter_incidences_per_chain_post", "m_total_pairs_per_chain_post"):
            assert k in d and d[k].dtype != object
        mi = d["m_intra_post"].astype(np.float64)
        me = d["m_inter_post"].astype(np.float64)
        np.testing.assert_allclose(d["m_intra_per_chain_post"], mi / M)
        np.testing.assert_allclose(d["m_inter_pairs_per_chain_post"], me / M)
        np.testing.assert_allclose(d["m_inter_incidences_per_chain_post"], 2.0 * me / M)
        np.testing.assert_allclose(d["m_total_pairs_per_chain_post"], (mi + me) / M)
        assert int(d["n_chains"]) == M


def test_rg_scale_does_not_affect_contacts_or_normalized(tmp_path):
    # Contacts (raw and normalized) must be identical for two rg_scale values;
    # only length axes (std/mean Rg) scale.
    (reps, sp, sa, wh), Ts = _short_run()
    p1 = mcd.save_diagnostic_trajectories_npz(
        reps, wh, Ts, burnin_frac=0.5, out_prefix=str(tmp_path / "s1"), rg_scale=1.0)
    p2 = mcd.save_diagnostic_trajectories_npz(
        reps, wh, Ts, burnin_frac=0.5, out_prefix=str(tmp_path / "s2"), rg_scale=0.25)
    with np.load(p1) as d1, np.load(p2) as d2:
        for k in ("m_intra_post", "m_inter_post", "m_intra_per_chain_post",
                  "m_inter_pairs_per_chain_post", "m_inter_incidences_per_chain_post",
                  "m_total_pairs_per_chain_post", "largest_cluster_fraction_post",
                  "n_clusters_post", "std_chain_rg_lattice_post"):
            assert np.array_equal(d1[k], d2[k]), f"{k} changed with rg_scale"
        # Scaled length differs by exactly the rg_scale ratio.
        np.testing.assert_allclose(
            d2["std_chain_rg_post"] * np.float32(4.0), d1["std_chain_rg_post"],
            rtol=1e-6, atol=1e-6)


def test_diagnostics_include_normalized_convergence_no_extra_warnings():
    # Normalized series get ESS/drift entries but do NOT add warning types beyond
    # the raw central observables.
    (reps, sp, sa, wh), Ts = _short_run()
    diag = mcd.compute_multichain_diagnostics(
        reps, sp, sa, wh, Ts, burnin_frac=0.5, rg_scale=1.0)
    lane0 = diag["lane_convergence"][0]
    assert "m_intra_per_chain" in lane0 and "m_inter_pairs_per_chain" in lane0
    # Every warning references a raw/central observable name, never a per_chain one.
    for w in diag["warnings"]:
        assert "per_chain" not in w["type"]


# ---------------------------------------------------------------------------
# Snapshot writer: dtype, dimensions, overflow, status (Change 1 + Change 5)
# ---------------------------------------------------------------------------

import multichain_config_io as mcio  # noqa: E402


def _snap_writer(path, nT, M, N, **kw):
    return mcio.MultiChainSnapshotWriter(
        path, n_chains=M, chain_length=N, n_temperatures=nT,
        metadata={"M": M, "N": N}, **kw)


def _append_zeros(w, nT, M, N, cycle=0, coords=None):
    if coords is None:
        coords = np.zeros((nT, M, N, 3), dtype=np.int64)
    w.append(cycle=cycle, coordinates=coords, walker_id=np.arange(nT),
             m_intra=np.zeros(nT, dtype=np.int64),
             m_inter=np.zeros(nT, dtype=np.int64),
             mean_chain_rg2=np.ones(nT), largest_cluster_size=np.ones(nT, dtype=np.int64))


@pytest.mark.skipif(not mcio.h5py_available(), reason="h5py unavailable")
def test_snapshot_default_dtype_int32_and_dims(tmp_path):
    import h5py
    nT, M, N = 2, 2, 4
    path = str(tmp_path / "snap.h5")
    w = _snap_writer(path, nT, M, N)
    _append_zeros(w, nT, M, N)
    w.mark_complete()
    w.close()
    with h5py.File(path, "r") as f:
        snap = f["snapshots"]
        assert snap.attrs["coordinate_dtype"] == "int32"
        assert snap["coordinates"].dtype == np.int32
        assert snap["coordinates"].shape == (1, nT, M, N, 3)
        assert snap.attrs["coordinate_dims"] == \
            "snapshot,temperature_lane,chain,monomer,xyz"
        assert snap.attrs["status"] == "complete"


@pytest.mark.skipif(not mcio.h5py_available(), reason="h5py unavailable")
def test_snapshot_large_coordinates_no_overflow(tmp_path):
    import h5py
    nT, M, N = 1, 1, 4
    path = str(tmp_path / "big.h5")
    big = 40000  # exceeds int16 max (32767) but fits int32
    coords = np.zeros((nT, M, N, 3), dtype=np.int64)
    coords[0, 0, :, 0] = big + np.arange(N)
    w = _snap_writer(path, nT, M, N)
    _append_zeros(w, nT, M, N, coords=coords)
    w.mark_complete()
    w.close()
    with h5py.File(path, "r") as f:
        stored = f["snapshots"]["coordinates"][:]
        assert stored.dtype == np.int32
        assert int(stored.max()) >= big


@pytest.mark.skipif(not mcio.h5py_available(), reason="h5py unavailable")
def test_snapshot_int16_overflow_raises(tmp_path):
    nT, M, N = 1, 1, 4
    path = str(tmp_path / "ovf.h5")
    coords = np.zeros((nT, M, N, 3), dtype=np.int64)
    coords[0, 0, 0, 0] = 40000
    w = _snap_writer(path, nT, M, N, coord_dtype="int16")
    with pytest.raises(mcio.SnapshotWriterError):
        _append_zeros(w, nT, M, N, coords=coords)
    w.close()


@pytest.mark.skipif(not mcio.h5py_available(), reason="h5py unavailable")
def test_snapshot_interrupted_status(tmp_path):
    import h5py
    nT, M, N = 2, 1, 4
    path = str(tmp_path / "interrupted.h5")
    w = _snap_writer(path, nT, M, N)
    _append_zeros(w, nT, M, N)
    w.close()  # no mark_complete -> interrupted
    with h5py.File(path, "r") as f:
        assert f["snapshots"].attrs["status"] == "interrupted"
        assert int(f["snapshots"].attrs["committed_rows"]) == 1


# ---------------------------------------------------------------------------
# Statistical limits at the sweep level
# ---------------------------------------------------------------------------

def test_independent_collapse_translations_all_accepted():
    # Under (1, 0), a whole-chain translation has du = b * lambda_intra * 0 = 0,
    # so every valid state-changing translation is accepted.
    state = mcs.initialize_dispersed_state(3, 8, 12, seed=4)
    counters = mvs.new_move_counters()
    rng = random.Random(0)
    rmc.mc_sweep(state, counters, 330.0, "hs", [400.0, 1.3], 320.0, 80.0,
                 lambda_intra=1.0, lambda_inter=0.0,
                 n_local=0, n_translation=500, rng=rng,
                 n_reptation=0, n_rotation=0, debug_contacts=True)
    idx = mvs.MOVE_INDEX["chain_translation"]
    # accepted (col 3) == state-changing valid (col 2): none rejected by Metropolis.
    assert counters[idx, 3] == counters[idx, 2]
    assert counters[idx, 2] > 0


def test_athermal_all_valid_state_changing_moves_accepted():
    # Under (0, 0), du = 0 for every move, so all valid state-changing moves pass.
    state = mcs.initialize_dispersed_state(3, 8, 12, seed=6)
    counters = mvs.new_move_counters()
    rng = random.Random(1)
    rmc.mc_sweep(state, counters, 330.0, "hs", [400.0, 1.3], 320.0, 80.0,
                 lambda_intra=0.0, lambda_inter=0.0,
                 n_local=800, n_translation=200, rng=rng,
                 n_reptation=200, n_rotation=200, debug_contacts=True)
    assert np.array_equal(counters[:, 3], counters[:, 2])
    assert counters[:, 2].sum() > 0


def test_strong_interchain_attraction_raises_inter():
    # Strong attraction (b < 0, favoring contacts) should yield more interchain
    # contacts than the athermal control on a short run (broad sanity check).
    Ts = np.linspace(320, 340, 3)
    common = dict(n_chains=4, chain_length=6, box_size=8, Ts=Ts,
                  local_sweeps_per_swap=2, translation_sweeps_per_swap=3,
                  n_cycles=120, model_name="hs", params=[0.0, 4.0], Tref=330.0,
                  Tscale=40.0, seed=7, n_workers=1, verbose=False)
    reps_ath, *_ = rmc.run_remd_multichain(lambda_intra=0.0, lambda_inter=0.0, **common)
    reps_att, *_ = rmc.run_remd_multichain(lambda_intra=0.0, lambda_inter=1.0, **common)

    def tail(reps, attr):
        return float(np.mean([np.asarray(getattr(r, attr))[60:].mean() for r in reps]))
    assert tail(reps_att, "m_inter_traj") >= tail(reps_ath, "m_inter_traj")
    assert tail(reps_att, "lcf_traj") >= tail(reps_ath, "lcf_traj")


if __name__ == "__main__":
    sys.exit(pytest.main([os.path.abspath(__file__), "-q"]))
