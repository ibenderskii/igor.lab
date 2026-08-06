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
from lattice_bending import BEND_DEFINITION, count_bends, count_bends_multichain

MP = ("hs", [400.0, 1.3], 320.0, 80.0)  # model_name, params, Tref, Tscale
SPLIT_MP = ("poly2", [0.1, -0.4, 0.2], 320.0, 80.0)


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


@pytest.mark.parametrize("lambda_intra,lambda_inter", [
    (1.0, 1.0), (1.0, 0.0), (0.0, 1.0), (0.5, 0.25),
])
def test_hs_supports_independent_contact_scales(lambda_intra, lambda_inter):
    # hs is LINEAR in the contact count, so the intra/inter split carries
    # through exactly and the two scales are independent.  (1,0) and (0,1) are
    # the collapse-only and association-only pilot controls; forcing
    # lambda_intra == lambda_inter would make them unreachable.
    c = ContactCounts(4, 3)
    b = remd.reduced_bias(*MP[:2], 330.0, *MP[2:])
    u = rmc.reduced_potential_counts(c, 330.0, *MP, lambda_intra, lambda_inter)
    assert u == b * (lambda_intra * c.intra + lambda_inter * c.inter)


@pytest.mark.parametrize("lambda_intra,lambda_inter", [
    (1.0, 0.0), (0.0, 1.0), (0.5, 0.25),
])
def test_saturating_rejects_unequal_contact_scales(lambda_intra, lambda_inter):
    # Only the model that is NONLINEAR in m needs one common contact scale.
    state = mcs.initialize_dispersed_state(3, 8, 12, seed=4)
    with pytest.raises(ValueError, match="lambda_intra == lambda_inter"):
        rmc.reduced_contact_potential_state(
            state, 330.0, SAT, SAT_P, QTREF, QTSCALE,
            lambda_intra, lambda_inter)


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
        return rmc.reduced_potential_counts(c, T, *SPLIT_MP, li, lin)
    expect = u(ci, Ti) + u(cj, Tj) - u(cj, Ti) - u(ci, Tj)
    got = rmc.swap_log_accept_counts(ci, cj, Ti, Tj, *SPLIT_MP, li, lin)
    assert abs(got - expect) < 1e-12


def test_hs_swap_is_invariant_to_contact_classification():
    ci = ContactCounts(6, 0)  # total 6
    cj = ContactCounts(0, 6)  # total 6, different split
    got = rmc.swap_log_accept_counts(ci, cj, 300.0, 350.0, *MP, 1.0, 1.0)
    assert got == 0.0


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
                                    *SPLIT_MP, 0.0, 1.0)
    assert la < 0.0, "test setup expects an unfavorable swap"
    thresh = math.exp(la)

    # r > exp(la) -> reject, no state change.
    a, b = fresh()
    sa, sb = a.state, b.state
    assert rmc.attempt_swap(
        a, b, *SPLIT_MP, 0.0, 1.0, _StubRng(thresh + 0.05)) is False
    assert a.state is sa and b.state is sb

    # r < exp(la) -> accept, states exchanged; occupancy/counts stay valid.
    a, b = fresh()
    sa, sb = a.state, b.state
    ca, cb = sa.counts.as_tuple(), sb.counts.as_tuple()
    assert rmc.attempt_swap(
        a, b, *SPLIT_MP, 0.0, 1.0,
        _StubRng(max(thresh - 0.05, 0.0))) is True
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
                                          a.T, b.T, *SPLIT_MP, 1.0, 1.0)
    la_split = rmc.swap_log_accept_counts(a.state.counts, b.state.counts,
                                          a.T, b.T, *SPLIT_MP, 1.0, 0.0)
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
                     lambda_intra=0.0, lambda_inter=0.0,
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
    reps_att, *_ = rmc.run_remd_multichain(lambda_intra=1.0, lambda_inter=1.0, **common)

    def tail(reps, attr):
        return float(np.mean([np.asarray(getattr(r, attr))[60:].mean() for r in reps]))
    assert tail(reps_att, "m_inter_traj") >= tail(reps_ath, "m_inter_traj")
    assert tail(reps_att, "lcf_traj") >= tail(reps_ath, "lcf_traj")


# ---------------------------------------------------------------------------
# Optional fixed reduced bending penalty  u += kappa_bend * n_bend_total
# ---------------------------------------------------------------------------

def _run_kappa(kappa_bend, n_workers=1, seed=13, debug=True, n_cycles=15):
    Ts = np.linspace(305, 350, 4)
    return rmc.run_remd_multichain(
        n_chains=2, chain_length=8, box_size=10, Ts=Ts,
        local_sweeps_per_swap=1, translation_sweeps_per_swap=1, n_cycles=n_cycles,
        model_name="hs", params=[400.0, 1.3], Tref=320.0, Tscale=80.0,
        lambda_intra=1.0, lambda_inter=1.0, kappa_bend=kappa_bend,
        seed=seed, n_workers=n_workers, verbose=False, debug_contacts=debug), Ts


def test_bend_total_equals_sum_of_per_chain_counts():
    # (1) The cached / recounted total equals the sum of independent per-chain
    # counts, for several dispersed multi-chain states.
    for seed in (1, 5, 9, 17):
        state = mcs.initialize_dispersed_state(4, 8, 12, seed=seed)
        per_chain = [count_bends(state.coords_unwrapped[c])
                     for c in range(state.n_chains)]
        total = mcs.total_bend_count(state.coords_unwrapped)
        assert total == sum(per_chain)
        assert total == count_bends_multichain(state.coords_unwrapped)
        assert state.n_bend == total  # cache initialized from the full count


def test_bend_count_periodic_boundary_crossing():
    # (2) An L-shaped chain with exactly one 90-degree turn, placed so it wraps
    # across the periodic boundary in x, must still count as one bend because the
    # UNWRAPPED coordinates stay contiguous.
    L = 8
    lshape = np.array([(6, 0, 0), (7, 0, 0), (8, 0, 0), (8, 1, 0), (8, 2, 0)],
                      dtype=np.int64)  # x=8 wraps to 0; turn at index 2
    partner = straight_chain(5, axis=1, start=(3, 0, 3))
    crossing = mcs.make_state(np.stack([lshape, partner]), L)
    assert crossing.n_bend == 1
    assert mcs.total_bend_count(crossing.coords_unwrapped) == 1
    # Same shape entirely inside the box must give the identical count.
    inside = mcs.make_state(
        np.stack([lshape - np.array([4, 0, 0]), partner]), L)
    assert inside.n_bend == crossing.n_bend == 1
    # A straight partner chain contributes no bends.
    assert count_bends(crossing.coords_unwrapped[1]) == 0


def test_local_and_reptation_bend_deltas_match_full_recount():
    # (3) proposal_delta_bends for local and reptation moves equals the change in
    # the full recount, and the cached n_bend stays exact after applying.
    for proposer in (mvs.propose_local, mvs.propose_reptation):
        state = mcs.initialize_dispersed_state(3, 10, 12, seed=23)
        rng = random.Random(7)
        n_changing = 0
        for _ in range(1500):
            prop = proposer(state, rng)
            if not prop.ok:
                continue
            d_bends = mvs.proposal_delta_bends(state, prop)
            n_before = mcs.total_bend_count(state.coords_unwrapped)
            delta = mvs.proposal_delta(state, prop)
            mvs.apply_proposal(state, prop, delta, delta_bends=d_bends)
            n_after = mcs.total_bend_count(state.coords_unwrapped)
            assert d_bends == n_after - n_before
            assert state.n_bend == n_after
            if d_bends != 0:
                n_changing += 1
        mcs.validate_state(state)
        assert n_changing > 0, f"{proposer.__name__} never changed a bend"


def test_translation_and_rotation_have_zero_bend_delta():
    # (4) Whole-chain translation and rigid whole-chain rotation are isometries of
    # the moved chain: proposal_delta_bends is exactly 0 and the count is unchanged.
    for proposer in (mvs.propose_translation, mvs.propose_chain_rotation):
        state = mcs.initialize_dispersed_state(3, 8, 12, seed=31)
        rng = random.Random(3)
        n_applied = 0
        for _ in range(1200):
            prop = proposer(state, rng)
            if not prop.ok:
                continue
            assert mvs.proposal_delta_bends(state, prop) == 0
            n_before = state.n_bend
            delta = mvs.proposal_delta(state, prop)
            mvs.apply_proposal(state, prop, delta)  # recomputes delta_bends -> 0
            assert state.n_bend == n_before
            assert mcs.total_bend_count(state.coords_unwrapped) == n_before
            n_applied += 1
        assert n_applied > 0, f"{proposer.__name__} produced no valid moves"


def test_cached_bend_count_correct_after_long_debug_run():
    # (5) A long mixed-move athermal sweep in debug mode keeps the cached bend
    # count in sync with the full recount (mc_sweep asserts it every accepted
    # move; validate_state re-checks it at the end).
    state = mcs.initialize_dispersed_state(3, 8, 12, seed=42)
    counters = mvs.new_move_counters()
    rng = random.Random(2024)
    rmc.mc_sweep(state, counters, 330.0, "hs", [0.0, 0.0], 330.0, 40.0,
                 lambda_intra=0.0, lambda_inter=0.0,
                 n_local=1500, n_translation=300, rng=rng,
                 n_reptation=300, n_rotation=300, debug_contacts=True,
                 kappa_bend=0.7)
    mcs.validate_state(state)
    assert state.n_bend == mcs.total_bend_count(state.coords_unwrapped)


def test_metropolis_helper_includes_bending():
    # (6) The Metropolis helper folds kappa_bend * delta_bends into delta_u.
    b, li, lin = 1.0, 1.0, 1.0
    # Pure bend change, no contact change.  Favorable (du<0) always accepts w/o draw.
    assert rmc.metropolis_accept_delta(0, 0, b, li, lin, 0.999,
                                       kappa_bend=0.5, delta_bends=-3) is True
    # Neutral (du=0) always accepts.
    assert rmc.metropolis_accept_delta(0, 0, b, li, lin, 0.999,
                                       kappa_bend=0.5, delta_bends=0) is True
    # Unfavorable du = kappa*delta_bends = 1.0 -> exp(-1)=0.3679 threshold.
    assert rmc.metropolis_accept_delta(0, 0, b, li, lin, 0.10,
                                       kappa_bend=1.0, delta_bends=1) is True
    assert rmc.metropolis_accept_delta(0, 0, b, li, lin, 0.90,
                                       kappa_bend=1.0, delta_bends=1) is False
    # Contact and bend contributions add: d_intra=+1 (du_c=1) and delta_bends=-1
    # with kappa=1 exactly cancel -> du=0 -> accept without consulting the draw.
    stub = _StubRng(0.0)
    assert rmc.metropolis_accept_delta(1, 0, b, li, lin, stub.random(),
                                       kappa_bend=1.0, delta_bends=-1) is True
    # Full reduced potential helper is consistent with the contact-only one.
    c = ContactCounts(4, 2)
    u_c = rmc.reduced_potential_counts(c, 330.0, *MP, 1.0, 1.0)
    u_b = rmc.reduced_potential_bending_counts(c, 330.0, *MP, 1.0, 1.0,
                                               kappa_bend=0.5, n_bend=6)
    assert abs(u_b - (u_c + 0.5 * 6)) < 1e-12


def test_kappa_zero_preserves_sampling_exactly():
    # (7) With kappa_bend = 0 the bend term is multiplied by zero, so the sampler
    # must be completely insensitive to the bend delta.  Monkeypatching
    # proposal_delta_bends to return a huge constant leaves every coordinate,
    # count, move counter and contact trajectory unchanged (only n_bend, which is
    # not consulted for sampling, diverges) -- proving kappa=0 reproduces the
    # original contacts-only output bit-for-bit.
    (base, *_), _ = _run_kappa(0.0, debug=False)
    real = mvs.proposal_delta_bends
    try:
        mvs.proposal_delta_bends = lambda state, proposal: 1000
        (patched, *_), _ = _run_kappa(0.0, debug=False)
    finally:
        mvs.proposal_delta_bends = real
    for a, b in zip(base, patched):
        assert np.array_equal(a.state.coords_unwrapped, b.state.coords_unwrapped)
        assert a.state.counts.as_tuple() == b.state.counts.as_tuple()
        assert np.array_equal(a.move_counters, b.move_counters)
        assert np.allclose(a.u_traj, b.u_traj)
        assert a.m_intra_traj == b.m_intra_traj
        assert a.m_inter_traj == b.m_inter_traj
    # And a plain kappa=0 run is reproducible for a fixed seed.
    (r1, *_), _ = _run_kappa(0.0)
    (r2, *_), _ = _run_kappa(0.0)
    for a, b in zip(r1, r2):
        assert np.array_equal(a.state.coords_unwrapped, b.state.coords_unwrapped)
        assert a.n_bend_traj == b.n_bend_traj


def test_fixed_kappa_cancels_from_swap_rule():
    # (8) The swap log-acceptance is invariant to bend counts at fixed kappa: the
    # temperature-independent penalty cancels exactly.  The production swap helper
    # is contacts-only; the full-potential form reproduces it for any kappa and
    # any (different) bend counts.
    ci, cj = ContactCounts(3, 7), ContactCounts(8, 1)
    Ti, Tj = 300.0, 350.0
    contact_only = rmc.swap_log_accept_counts(ci, cj, Ti, Tj, *MP, 1.0, 1.0)
    for kappa in (0.0, 0.4, 1.7):
        for ni, nj in ((0, 6), (5, 5), (6, 1)):
            def u(cc, T, n):
                return rmc.reduced_potential_bending_counts(
                    cc, T, *MP, 1.0, 1.0, kappa_bend=kappa, n_bend=n)
            full = (u(ci, Ti, ni) + u(cj, Tj, nj)
                    - u(cj, Ti, nj) - u(ci, Tj, ni))
            assert abs(full - contact_only) < 1e-9


def test_m1_full_potential_matches_single_chain_with_bending():
    # (9) For M = 1 the multichain full reduced potential (contacts + bending)
    # equals the single-chain reduced_potential_bending for the same coordinates,
    # contact count, bend count, temperature and kappa.
    rng = np.random.RandomState(11)
    for _ in range(8):
        saw = mcs.generate_saw(18, rng)
        state = mcs.make_state(np.stack([saw]), 60)
        m = int(state.counts.intra)
        assert state.counts.inter == 0
        n_bend = state.n_bend
        assert n_bend == count_bends(saw)
        for T in (300.0, 342.0):
            for kappa in (0.0, 0.8):
                u_mc = rmc.reduced_potential_bending_counts(
                    state.counts, T, *MP, 1.0, 1.0, kappa_bend=kappa, n_bend=n_bend)
                u_ref = remd.reduced_potential_bending(
                    m, T, *MP, kappa_bend=kappa, n_bend=n_bend)
                assert abs(u_mc - u_ref) < 1e-9


def test_serial_and_two_workers_identical_with_nonzero_kappa():
    # (10) Deterministic per-lane seeding keeps serial and multiprocessing runs
    # bit-identical even with the bending penalty enabled (kappa is threaded to
    # every worker and the bend count is carried across the process boundary).
    (r1, sp1, sa1, wh1), _ = _run_kappa(0.6, n_workers=1)
    (r2, sp2, sa2, wh2), _ = _run_kappa(0.6, n_workers=2)
    assert np.array_equal(sp1, sp2) and np.array_equal(sa1, sa2)
    assert np.array_equal(wh1, wh2)
    for a, b in zip(r1, r2):
        assert np.array_equal(a.state.coords_unwrapped, b.state.coords_unwrapped)
        assert a.state.counts.as_tuple() == b.state.counts.as_tuple()
        assert a.state.n_bend == b.state.n_bend
        assert a.n_bend_traj == b.n_bend_traj
        assert np.array_equal(a.move_counters, b.move_counters)


def test_positive_kappa_reduces_mean_bend_fraction():
    # (11) A positive penalty straightens the chains: the post-burn-in mean bend
    # fraction drops relative to the athermal (kappa=0) control in a short run.
    common = dict(n_chains=3, chain_length=8, box_size=12,
                  Ts=np.linspace(320, 340, 3), local_sweeps_per_swap=3,
                  translation_sweeps_per_swap=1, n_cycles=180, model_name="hs",
                  params=[0.0, 0.0], Tref=330.0, Tscale=40.0, seed=321,
                  n_workers=1, verbose=False)
    reps0, *_ = rmc.run_remd_multichain(kappa_bend=0.0, **common)
    repsK, *_ = rmc.run_remd_multichain(kappa_bend=1.5, **common)
    bf0 = np.nanmean(rmc._bend_fraction_by_temp(reps0, 3, 8, 0.5))
    bfK = np.nanmean(rmc._bend_fraction_by_temp(repsK, 3, 8, 0.5))
    assert 0.0 <= bfK < bf0 <= 1.0, f"kappa did not reduce bends: {bfK} !< {bf0}"


def test_kappa_metadata_propagates_into_summary_and_distributions(tmp_path):
    # (12) The CLI kappa value flows through resolution into both the run-summary
    # JSON and the distributions NPZ (metadata + additive trajectories).
    prefix = str(tmp_path / "kb")
    rmc.main([
        "--n-chains", "2", "--N", "6", "--box-size", "8", "--n-cycles", "8",
        "--nT", "3", "--Tmin", "320", "--Tmax", "340", "--model", "hs",
        "--params", "0,4", "--Tref", "330", "--Tscale", "40",
        "--kappa-bend", "0.5", "--seed", "1", "--no-plots",
        "--out-prefix", prefix])
    import json as _json
    with open(f"{prefix}_run_summary.json") as f:
        summary = _json.load(f)
    assert summary["kappa_bend"] == 0.5
    assert summary["bending_enabled"] is True
    assert summary["bend_definition"] == BEND_DEFINITION
    assert len(summary["bend_fraction"]) == 3
    with np.load(f"{prefix}_distributions.npz", allow_pickle=True) as d:
        assert float(d["kappa_bend"]) == 0.5
        assert bool(d["bending_enabled"]) is True
        assert str(d["bend_definition"]) == BEND_DEFINITION
        assert "n_bends" in d and d["n_bends"].shape == (3, 8)
        assert "bend_fraction" in d and d["bend_fraction"].shape == (3,)


def test_recorded_u_and_effective_energy_include_bending():
    # (13) Regression: the recorded reduced potential u (and the effective energy
    # T*u) must be the FULL potential u_contact + kappa_bend * n_bend, not the
    # contact-only term.  With lambda_intra = lambda_inter = 0 the contact reduced
    # potential is identically zero, so every recorded u must equal
    # kappa_bend * n_bend (nonzero because dispersed chains bend) and the effective
    # energy must be T * u.  Before the fix u_traj used the contacts-only helper
    # and every recorded u would be 0 here.
    kappa_bend = 0.7
    Ts = np.linspace(305, 350, 4)
    reps, *_ = rmc.run_remd_multichain(
        n_chains=2, chain_length=8, box_size=10, Ts=Ts,
        local_sweeps_per_swap=1, translation_sweeps_per_swap=1, n_cycles=15,
        model_name="hs", params=[400.0, 1.3], Tref=320.0, Tscale=80.0,
        lambda_intra=0.0, lambda_inter=0.0, kappa_bend=kappa_bend,
        seed=13, n_workers=1, verbose=False)
    saw_nonzero_bend = False
    for rep in reps:
        assert len(rep.u_traj) == len(rep.eeff_traj) == len(rep.n_bend_traj) == 15
        for u, eeff, n_bend in zip(rep.u_traj, rep.eeff_traj, rep.n_bend_traj):
            assert abs(u - kappa_bend * n_bend) < 1e-12
            assert abs(eeff - rep.T * u) < 1e-12
            if n_bend > 0:
                saw_nonzero_bend = True
    assert saw_nonzero_bend, "test never exercised a nonzero bending contribution"


# ---------------------------------------------------------------------------
# Contact-quadratic models: per-chain intrachain m^2/(2N), linear interchain
# ---------------------------------------------------------------------------

CONST_P = [700.0, 2.4, 0.9]        # hs_m2_const: h1, s1, kappa2
CONST_P0 = [700.0, 2.4, 0.0]       # hs_m2_const with zero curvature
HSHS_P = [700.0, 2.4, 300.0, 0.8]  # hs_m2_hs: h1, s1, h2, s2
HSHS_P0 = [700.0, 2.4, 0.0, 0.0]   # hs_m2_hs with zero curvature
QTREF, QTSCALE = 320.0, 80.0


def test_per_chain_cache_equals_full_recount():
    # (1) The cached per-chain intrachain vector equals a full per-chain recount
    # for several dispersed states, and stays exact after many accepted moves.
    for seed in (1, 5, 9, 17):
        state = mcs.initialize_dispersed_state(4, 8, 12, seed=seed)
        recount = mcc.full_intra_contacts_by_chain_state(state)
        assert np.array_equal(state.intra_contacts_by_chain, recount)
    # After a long mixed athermal sweep the per-chain cache still matches.
    state = mcs.initialize_dispersed_state(3, 8, 12, seed=42)
    counters = mvs.new_move_counters()
    rmc.mc_sweep(state, counters, 330.0, "hs", [0.0, 0.0], 330.0, 40.0,
                 lambda_intra=0.0, lambda_inter=0.0,
                 n_local=1500, n_translation=300, rng=random.Random(7),
                 n_reptation=300, n_rotation=300, debug_contacts=True)
    assert np.array_equal(state.intra_contacts_by_chain,
                          mcc.full_intra_contacts_by_chain_state(state))


def test_per_chain_sum_equals_total_intra():
    # (2) sum(intra_contacts_by_chain) == counts.intra for dispersed states and
    # after moves (validate_state also enforces this invariant).
    for seed in (2, 8, 14):
        state = mcs.initialize_dispersed_state(4, 8, 12, seed=seed)
        assert int(state.intra_contacts_by_chain.sum()) == int(state.counts.intra)
        counters = mvs.new_move_counters()
        rmc.mc_sweep(state, counters, 330.0, "hs_m2_const", CONST_P, QTREF, QTSCALE,
                     lambda_intra=1.0, lambda_inter=1.0,
                     n_local=400, n_translation=80, rng=random.Random(seed),
                     n_reptation=80, n_rotation=80, debug_contacts=True)
        assert int(state.intra_contacts_by_chain.sum()) == int(state.counts.intra)
        mcs.validate_state(state)


def test_hs_state_potential_equals_split_aggregate_formula():
    state = mcs.initialize_dispersed_state(3, 8, 12, seed=7)
    for T in (300.0, 330.0, 345.0):
        for li, lin in ((1.0, 1.0), (0.0, 0.0), (0.7, 0.7),
                        (1.0, 0.0), (0.0, 1.0), (0.5, 0.25)):
            u_state = rmc.reduced_contact_potential_state(
                state, T, *MP, li, lin)
            u_agg = rmc.reduced_potential_counts(state.counts, T, *MP, li, lin)
            assert u_state == u_agg
            # And the full (bending) potential matches too.
            u_full = rmc.reduced_potential_state(state, T, *MP, li, lin,
                                                 kappa_bend=0.5)
            u_full_agg = rmc.reduced_potential_bending_counts(
                state.counts, T, *MP, li, lin, kappa_bend=0.5,
                n_bend=state.n_bend)
            assert u_full == u_full_agg


def test_aggregate_helpers_reject_contact_quadratic():
    # The aggregate-ContactCounts compatibility helpers carry only the total
    # intrachain count and cannot determine sum_alpha u_contact(m_alpha), so they
    # must reject the contact-quadratic models loudly.
    c = ContactCounts(4, 2)
    for model, p in (("hs_m2_const", CONST_P), ("hs_m2_hs", HSHS_P)):
        with pytest.raises(NotImplementedError, match="nonlinear in m"):
            rmc.reduced_potential_counts(c, 330.0, model, p, QTREF, QTSCALE, 1.0, 1.0)
        with pytest.raises(NotImplementedError, match="nonlinear in m"):
            rmc.reduced_potential_bending_counts(
                c, 330.0, model, p, QTREF, QTSCALE, 1.0, 1.0,
                kappa_bend=0.3, n_bend=5)
        with pytest.raises(NotImplementedError, match="nonlinear in m"):
            rmc.swap_log_accept_counts(c, c, 300.0, 350.0, model, p, QTREF,
                                       QTSCALE, 1.0, 1.0)


def test_m1_full_potential_matches_single_chain_quadratic():
    # (4) For M = 1 the multichain full reduced potential (per-chain contacts +
    # bending) equals the single-chain reduced_potential_bending for both new
    # models, with bending off and on.
    rng = np.random.RandomState(4)
    for _ in range(6):
        saw = mcs.generate_saw(18, rng)
        state = mcs.make_state(np.stack([saw]), 60)
        m = int(state.counts.intra)
        assert state.counts.inter == 0
        for model, p in (("hs_m2_const", CONST_P), ("hs_m2_hs", HSHS_P)):
            for T in (300.0, 342.0):
                for kappa in (0.0, 0.8):
                    u_mc = rmc.reduced_potential_state(
                        state, T, model, p, QTREF, QTSCALE, 1.0, 1.0,
                        kappa_bend=kappa)
                    u_ref = remd.reduced_potential_bending(
                        m, T, model, p, QTREF, QTSCALE, kappa_bend=kappa,
                        n_bend=state.n_bend, n_beads=state.chain_length)
                    assert abs(u_mc - u_ref) < 1e-9


def test_two_separated_chains_additive_intrachain_energy():
    # (5) Two well-separated chains (m_inter = 0) have an intrachain contact
    # potential equal to the sum of each chain's single-chain contact potential.
    N, L = 8, 40
    rng = np.random.RandomState(23)
    A = mcs.generate_saw(N, rng)
    B = mcs.generate_saw(N, rng) + np.array([0, 20, 0], dtype=np.int64)
    state = mcs.make_state(np.stack([A, B]), L)
    assert int(state.counts.inter) == 0
    for model, p in (("hs_m2_const", CONST_P), ("hs_m2_hs", HSHS_P)):
        for T in (305.0, 348.0):
            u_total = rmc.reduced_contact_potential_state(
                state, T, model, p, QTREF, QTSCALE, 1.0, 0.0)
            u_a = remd.reduced_contact_potential(
                int(state.intra_contacts_by_chain[0]), T, model, p, QTREF,
                QTSCALE, N)
            u_b = remd.reduced_contact_potential(
                int(state.intra_contacts_by_chain[1]), T, model, p, QTREF,
                QTSCALE, N)
            assert abs(u_total - (u_a + u_b)) < 1e-9


def test_interchain_only_change_leaves_quadratic_intra_unchanged():
    # (6) A whole-chain translation changes only interchain contacts (intrachain
    # counts are preserved), so the per-chain quadratic intra potential is
    # unchanged while the linear interchain term absorbs the whole delta.
    state = mcs.initialize_dispersed_state(3, 8, 12, seed=17)
    model, p, T = "hs_m2_const", CONST_P, 330.0

    def intra_u(st):
        N = st.chain_length
        return sum(remd.reduced_contact_potential(int(m), T, model, p, QTREF,
                                                  QTSCALE, N)
                   for m in st.intra_contacts_by_chain)
    before_intra_u = intra_u(state)
    before_ibc = state.intra_contacts_by_chain.copy()
    rng = random.Random(3)
    n_trans = 0
    for _ in range(400):
        prop = mvs.propose_translation(state, rng)
        if not prop.ok:
            continue
        d_intra, d_inter = mvs.proposal_delta(state, prop)
        assert d_intra == 0, "translation must not change intrachain contacts"
        mvs.apply_proposal(state, prop, (d_intra, d_inter))
        n_trans += 1
    assert n_trans > 0
    assert np.array_equal(state.intra_contacts_by_chain, before_ibc)
    assert abs(intra_u(state) - before_intra_u) < 1e-12


def test_rigid_translation_preserves_intra_and_bending():
    # (7) A rigid whole-chain translation preserves both the per-chain intrachain
    # counts and the total bend count (isometry): the intra + bending
    # contributions to the potential are unchanged.
    state = mcs.initialize_dispersed_state(3, 8, 12, seed=31)
    ibc0 = state.intra_contacts_by_chain.copy()
    n_bend0 = state.n_bend
    rng = random.Random(5)
    n_applied = 0
    for _ in range(500):
        prop = mvs.propose_translation(state, rng)
        if not prop.ok:
            continue
        assert mvs.proposal_delta_bends(state, prop) == 0
        d_intra, d_inter = mvs.proposal_delta(state, prop)
        assert d_intra == 0
        mvs.apply_proposal(state, prop, (d_intra, d_inter))
        n_applied += 1
    assert n_applied > 0
    assert np.array_equal(state.intra_contacts_by_chain, ibc0)
    assert state.n_bend == n_bend0


def test_quadratic_lambda_control_modes():
    # (8) The four lambda modes select the intended contributions for a
    # contact-quadratic model.  Build a state with both intra and inter contacts.
    state = mcs.initialize_dispersed_state(4, 8, 12, seed=6)
    model, p, T = "hs_m2_hs", HSHS_P, 335.0
    N = state.chain_length
    b = remd.reduced_bias(model, p, T, QTREF, QTSCALE)
    intra_u = sum(remd.reduced_contact_potential(int(m), T, model, p, QTREF,
                                                 QTSCALE, N)
                  for m in state.intra_contacts_by_chain)
    inter = int(state.counts.inter)
    assert inter > 0 and int(state.counts.intra) > 0
    # (0,0): no contacts.
    assert rmc.reduced_contact_potential_state(
        state, T, model, p, QTREF, QTSCALE, 0.0, 0.0) == 0.0
    # (1,0): per-chain quadratic intra only.
    u10 = rmc.reduced_contact_potential_state(state, T, model, p, QTREF, QTSCALE,
                                              1.0, 0.0)
    assert abs(u10 - intra_u) < 1e-9
    # (0,1): linear interchain only.
    u01 = rmc.reduced_contact_potential_state(state, T, model, p, QTREF, QTSCALE,
                                              0.0, 1.0)
    assert abs(u01 - b * inter) < 1e-9
    # (1,1): both.
    u11 = rmc.reduced_contact_potential_state(state, T, model, p, QTREF, QTSCALE,
                                              1.0, 1.0)
    assert abs(u11 - (intra_u + b * inter)) < 1e-9
    # (0,0) with bending stays nonzero when kappa_bend != 0 (contacts off does
    # NOT disable bending).
    u_bend = rmc.reduced_potential_state(state, T, model, p, QTREF, QTSCALE,
                                         0.0, 0.0, kappa_bend=0.5)
    assert abs(u_bend - 0.5 * state.n_bend) < 1e-12


def test_full_state_swap_matches_manual_four_potential_quadratic():
    # (9) The generalized full-state swap equals a manual four-potential
    # calculation for the contact-quadratic models, and the fixed bending penalty
    # cancels (it never enters the swap).
    sa = mcs.initialize_dispersed_state(3, 8, 12, seed=21)
    sb = mcs.initialize_dispersed_state(3, 8, 12, seed=22)
    Ti, Tj = 305.0, 350.0
    for model, p in (("hs_m2_const", CONST_P), ("hs_m2_hs", HSHS_P)):
        def u(state, T):
            return rmc.reduced_contact_potential_state(
                state, T, model, p, QTREF, QTSCALE, 1.0, 0.6)
        manual = u(sa, Ti) + u(sb, Tj) - u(sb, Ti) - u(sa, Tj)
        got = rmc.swap_log_accept_state(sa, sb, Ti, Tj, model, p, QTREF, QTSCALE,
                                        1.0, 0.6)
        assert abs(got - manual) < 1e-9
        # attempt_swap uses exactly this log-acceptance (favorable -> no draw).
        ra = rmc.MultiReplica(T=Ti, state=sa.copy())
        rb = rmc.MultiReplica(T=Tj, state=sb.copy())
        if manual >= 0.0:
            stub = _StubRng(1.0)
            assert rmc.attempt_swap(ra, rb, model, p, QTREF, QTSCALE, 1.0, 0.6,
                                    stub) is True
            assert stub.calls == 0


def _run_quadratic(model, p, n_workers, kappa=0.0, seed=13, n_cycles=12):
    Ts = np.linspace(305, 350, 4)
    return rmc.run_remd_multichain(
        n_chains=2, chain_length=8, box_size=10, Ts=Ts,
        local_sweeps_per_swap=1, translation_sweeps_per_swap=1, n_cycles=n_cycles,
        model_name=model, params=p, Tref=QTREF, Tscale=QTSCALE,
        lambda_intra=1.0, lambda_inter=1.0, kappa_bend=kappa,
        seed=seed, n_workers=n_workers, verbose=False, debug_contacts=True), Ts


def test_serial_vs_workers_determinism_quadratic():
    # (10) Serial and multiprocessing runs are bit-identical for the
    # contact-quadratic models (with bending off and on), including the per-chain
    # cache carried across the process boundary.
    for model, p in (("hs_m2_const", CONST_P), ("hs_m2_hs", HSHS_P)):
        for kappa in (0.0, 0.5):
            (r1, sp1, sa1, wh1), _ = _run_quadratic(model, p, 1, kappa)
            (r2, sp2, sa2, wh2), _ = _run_quadratic(model, p, 2, kappa)
            assert np.array_equal(sp1, sp2) and np.array_equal(sa1, sa2)
            assert np.array_equal(wh1, wh2)
            for a, b in zip(r1, r2):
                assert np.array_equal(a.state.coords_unwrapped,
                                      b.state.coords_unwrapped)
                assert a.state.counts.as_tuple() == b.state.counts.as_tuple()
                assert np.array_equal(a.state.intra_contacts_by_chain,
                                      b.state.intra_contacts_by_chain)
                assert a.state.n_bend == b.state.n_bend
                assert np.allclose(a.u_traj, b.u_traj)
                assert np.array_equal(a.move_counters, b.move_counters)
                mcs.validate_state(a.state)


def test_hs_m2_const_zero_curvature_matches_hs():
    # hs_m2_const(kappa2=0) and hs_m2_hs(h2=s2=0) reduce to hs: equal potential
    # for the same state, temperature and lambdas.
    state = mcs.initialize_dispersed_state(3, 8, 12, seed=11)
    for T in (300.0, 340.0):
        for li, lin in ((1.0, 1.0), (0.0, 0.0), (0.7, 0.7)):
            u_hs = rmc.reduced_contact_potential_state(
                state, T, "hs", [700.0, 2.4], QTREF, QTSCALE, li, lin)
            u_c0 = rmc.reduced_contact_potential_state(
                state, T, "hs_m2_const", CONST_P0, QTREF, QTSCALE, li, lin)
            u_h0 = rmc.reduced_contact_potential_state(
                state, T, "hs_m2_hs", HSHS_P0, QTREF, QTSCALE, li, lin)
            assert abs(u_hs - u_c0) < 1e-9
            assert abs(u_hs - u_h0) < 1e-9


def test_m1_and_m2_smoke_runs_each_new_model():
    # (13) Short M=1 and M=2 multichain runs for each new model complete with
    # valid states, finite trajectories, and (debug) exact caches throughout.
    for model, p in (("hs_m2_const", CONST_P), ("hs_m2_hs", HSHS_P)):
        for M in (1, 2):
            Ts = np.linspace(305, 350, 3)
            reps, *_ = rmc.run_remd_multichain(
                n_chains=M, chain_length=(18 if M == 1 else 8),
                box_size=(60 if M == 1 else 12), Ts=Ts,
                local_sweeps_per_swap=1, translation_sweeps_per_swap=1,
                n_cycles=8, model_name=model, params=p, Tref=QTREF,
                Tscale=QTSCALE, lambda_intra=1.0, lambda_inter=1.0,
                kappa_bend=0.3, seed=5, n_workers=1, verbose=False,
                debug_contacts=True)
            for rep in reps:
                mcs.validate_state(rep.state)
                assert all(math.isfinite(x) for x in rep.u_traj)
                assert int(rep.state.intra_contacts_by_chain.sum()) == \
                    int(rep.state.counts.intra)


def _write_fit_summary(path, model, params_dict, *, kappa_bend=None,
                       fit_chain_length=None):
    import json as _json
    summary = {
        "model_api_version": remd.MODEL_API_VERSION,
        "model": model,
        "param_names": list(params_dict.keys()),
        "params": params_dict,
        "Tref": QTREF,
        "Tscale": QTSCALE,
    }
    if kappa_bend is not None:
        summary["kappa_bend"] = kappa_bend
    if fit_chain_length is not None:
        summary["fit_chain_length"] = fit_chain_length
    with open(path, "w") as fh:
        _json.dump(summary, fh)
    return path


def test_fit_summary_roundtrip_drives_multichain_quadratic(tmp_path):
    # (12) A fit_summary.json for each new model drives a multichain run through
    # --fit-summary-json, including fit_chain_length transfer to a different
    # runtime N.  The recorded metadata reflects the runtime N and fit length.
    import json as _json
    cases = [
        ("hs_m2_const", {"h1": 700.0, "s1": 2.4, "kappa2": 0.9}, 30),
        ("hs_m2_hs", {"h1": 700.0, "s1": 2.4, "h2": 300.0, "s2": 0.8}, 30),
    ]
    for model, params_dict, fit_N in cases:
        summary_path = _write_fit_summary(
            tmp_path / f"{model}_summary.json", model, params_dict,
            kappa_bend=0.4, fit_chain_length=fit_N)
        # The single-chain loader accepts the quadratic summary + fit length.
        loaded = remd.load_fit_summary_json(str(summary_path))
        assert loaded["model_name"] == model
        assert loaded["fit_chain_length"] == fit_N
        assert loaded["kappa_bend"] == 0.4
        # Drive a short multichain run at a DIFFERENT runtime N (8 != 30).
        prefix = str(tmp_path / f"{model}_run")
        rmc.main([
            "--n-chains", "2", "--N", "8", "--box-size", "12",
            "--n-cycles", "6", "--nT", "3", "--Tmin", "310", "--Tmax", "345",
            "--fit-summary-json", str(summary_path),
            "--seed", "1", "--no-plots", "--out-prefix", prefix])
        with open(f"{prefix}_run_summary.json") as fh:
            s = _json.load(fh)
        assert s["model"] == model
        assert s["potential_kind"] == "contact_quadratic"
        assert s["quadratic_contact_scope"] == "intra_per_chain"
        assert s["interchain_contact_model"] == "linear_coefficient_only"
        assert s["quadratic_normalization"] == "m_chain^2/(2*N)"
        assert s["runtime_chain_length"] == 8
        assert s["fit_chain_length"] == fit_N
        assert s["kappa_bend"] == 0.4
        with np.load(f"{prefix}_distributions.npz", allow_pickle=False) as d:
            assert str(d["potential_kind"]) == "contact_quadratic"
            assert str(d["quadratic_normalization"]) == "m_chain^2/(2*N)"
            assert int(d["runtime_chain_length"]) == 8
            assert int(d["fit_chain_length"]) == fit_N
            assert np.any(d["quadratic_coefficient_by_temperature"] != 0.0)
            assert np.array_equal(d["reduced_bias_by_temperature"],
                                  d["linear_coefficient_by_temperature"])


def test_output_roundtrip_quadratic_metadata(tmp_path):
    # (11) The distributions NPZ round-trips (pickle-free) with the full
    # contact-quadratic contract metadata for a directly-parameterized run.
    prefix = str(tmp_path / "q")
    rmc.main([
        "--n-chains", "2", "--N", "8", "--box-size", "12", "--n-cycles", "6",
        "--nT", "3", "--Tmin", "310", "--Tmax", "345",
        "--model", "hs_m2_const", "--params", "700,2.4,0.9",
        "--Tref", "320", "--Tscale", "80", "--kappa-bend", "0.3",
        "--seed", "2", "--no-plots", "--out-prefix", prefix])
    # allow_pickle=False proves no None/object arrays leaked into the NPZ.
    with np.load(f"{prefix}_distributions.npz", allow_pickle=False) as d:
        for key in ("potential_kind", "quadratic_contact_scope",
                    "interchain_contact_model", "quadratic_normalization",
                    "runtime_chain_length", "fit_chain_length",
                    "linear_coefficient_by_temperature",
                    "quadratic_coefficient_by_temperature",
                    "reduced_bias_by_temperature", "kappa_bend",
                    "bending_enabled", "n_bends", "bend_fraction"):
            assert key in d, f"missing NPZ key {key}"
        assert int(d["fit_chain_length"]) == -1  # not from a fit summary
        assert int(d["runtime_chain_length"]) == 8


# ---------------------------------------------------------------------------
# saturating_cooperative_contact: one Hamiltonian over all contacts
#
#   u(X,T) = lambda_contact * u_sat(m_intra + m_inter, T; M*N)
#            + kappa_bend * n_bend_total
#   u_sat(m,T;N) = N*[b(T)*(m/N) - A0*(m/N)^2/(1 + ((m/N)/q_sat)^2)]
#   b(T)         = h_b/T - s_b
#
# q_sat is deliberately small enough (0.15) that the saturation is visible at the
# handful of contacts an N = 8 test chain can make.
# ---------------------------------------------------------------------------

SAT = "saturating_cooperative_contact"
SAT_P = [700.0, 2.4, 5.0, 0.15]    # h_b, s_b, A0, q_sat
SAT_P0 = [700.0, 2.4, 0.0, 0.15]   # A0 = 0 -> the linear hs potential
HS_P = [700.0, 2.4]                # the same h_b, s_b through the hs model


def _u_sat_direct(m, T, N, params=SAT_P):
    """The specified potential, transcribed literally (no shared helpers)."""
    h_b, s_b, A0, q_sat = (float(v) for v in params)
    b = h_b / float(T) - s_b
    q = float(m) / float(N)
    return float(N) * (b * q - A0 * q * q / (1.0 + (q / q_sat) ** 2))


# Three N = 6 conformations with known intrachain contact counts, used to build
# states whose TOTAL intrachain count is equal but whose per-chain allocation
# differs.  P: 3x2 rectangle (2 contacts).  Q: square + tail (1 contact).
# S: straight (0 contacts).
_CHAIN_P = np.array([(0, 0, 0), (1, 0, 0), (2, 0, 0), (2, 1, 0), (1, 1, 0),
                     (0, 1, 0)], dtype=np.int64)
_CHAIN_Q = np.array([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 2, 0),
                     (0, 3, 0)], dtype=np.int64)
_CHAIN_S = np.array([(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0), (4, 0, 0),
                     (5, 0, 0)], dtype=np.int64)
_FAR = np.array([0, 10, 0], dtype=np.int64)


def test_saturating_uses_single_hamiltonian_on_all_contacts():
    for seed in (6, 17, 31):
        state = mcs.initialize_dispersed_state(3, 8, 12, seed=seed)
        total_beads = state.n_chains * state.chain_length
        m_total = int(state.counts.intra) + int(state.counts.inter)
        for T in (305.0, 335.0, 350.0):
            direct = _u_sat_direct(m_total, T, total_beads)
            got = rmc.reduced_contact_potential_state(
                state, T, SAT, SAT_P, QTREF, QTSCALE, 1.0, 1.0)
            assert abs(got - direct) < 1e-9


def test_saturating_equal_total_different_allocation_is_identical():
    st_a = mcs.make_state(np.stack([_CHAIN_P, _CHAIN_S + _FAR]), 30)
    st_b = mcs.make_state(np.stack([_CHAIN_Q, _CHAIN_Q + _FAR]), 30)
    assert sorted(st_a.intra_contacts_by_chain.tolist()) == [0, 2]
    assert sorted(st_b.intra_contacts_by_chain.tolist()) == [1, 1]
    assert int(st_a.counts.intra) == int(st_b.counts.intra) == 2
    assert int(st_a.counts.inter) == int(st_b.counts.inter) == 0
    N = st_a.chain_length
    for T in (305.0, 335.0):
        ua = rmc.reduced_contact_potential_state(
            st_a, T, SAT, SAT_P, QTREF, QTSCALE, 1.0, 1.0)
        ub = rmc.reduced_contact_potential_state(
            st_b, T, SAT, SAT_P, QTREF, QTSCALE, 1.0, 1.0)
        assert ua == ub
        assert abs(ua - _u_sat_direct(2, T, 2 * N)) < 1e-9


def test_saturating_is_invariant_to_intra_inter_reclassification():
    state = mcs.initialize_dispersed_state(4, 8, 12, seed=6)
    a = state.copy()
    b = state.copy()
    a.counts = ContactCounts(2, 5)
    b.counts = ContactCounts(5, 2)
    for T in (305.0, 335.0):
        ua = rmc.reduced_contact_potential_state(
            a, T, SAT, SAT_P, QTREF, QTSCALE, 1.0, 1.0)
        ub = rmc.reduced_contact_potential_state(
            b, T, SAT, SAT_P, QTREF, QTSCALE, 1.0, 1.0)
        assert ua == ub


def test_saturating_lambda_control_modes():
    state = mcs.initialize_dispersed_state(4, 8, 12, seed=6)
    T = 335.0
    for li, lin in ((1.0, 0.0), (0.0, 1.0), (0.5, 0.25)):
        with pytest.raises(ValueError, match="lambda_intra == lambda_inter"):
            rmc.reduced_contact_potential_state(
                state, T, SAT, SAT_P, QTREF, QTSCALE, li, lin)
    assert rmc.reduced_contact_potential_state(
        state, T, SAT, SAT_P, QTREF, QTSCALE, 0.0, 0.0) == 0.0
    u_bend = rmc.reduced_potential_state(state, T, SAT, SAT_P, QTREF, QTSCALE,
                                         0.0, 0.0, kappa_bend=0.5)
    assert abs(u_bend - 0.5 * state.n_bend) < 1e-12
    assert u_bend != 0.0
    # and it is added on top of the contact terms, never folded into them.
    u_full = rmc.reduced_potential_state(state, T, SAT, SAT_P, QTREF, QTSCALE,
                                         1.0, 1.0, kappa_bend=0.5)
    u_contacts = rmc.reduced_contact_potential_state(
        state, T, SAT, SAT_P, QTREF, QTSCALE, 1.0, 1.0)
    assert abs(u_full - (u_contacts + 0.5 * state.n_bend)) < 1e-9


def test_saturating_move_delta_uses_full_potential():
    # Every proposal delta uses m_total and M*N, regardless of contact class.
    state = mcs.initialize_dispersed_state(3, 8, 12, seed=23)
    T, kb = 335.0, 0.4
    rng = random.Random(11)
    n_checked = 0
    proposers = (mvs.propose_local, mvs.propose_translation, mvs.propose_reptation,
                 mvs.propose_chain_rotation)
    for step in range(800):
        prop = proposers[step % len(proposers)](state, rng)
        if not prop.ok:
            continue
        d_intra, d_inter = mvs.proposal_delta(state, prop)
        d_bends = mvs.proposal_delta_bends(state, prop)
        m_old = int(state.counts.intra) + int(state.counts.inter)
        d_total = int(d_intra) + int(d_inter)
        total_beads = state.n_chains * state.chain_length
        u_before = rmc.reduced_potential_state(
            state, T, SAT, SAT_P, QTREF, QTSCALE, 1.0, 1.0, kappa_bend=kb)
        du = (remd.reduced_contact_potential(m_old + d_total, T, SAT, SAT_P,
                                             QTREF, QTSCALE, total_beads)
              - remd.reduced_contact_potential(m_old, T, SAT, SAT_P, QTREF,
                                               QTSCALE, total_beads)
              + kb * int(d_bends))
        mvs.apply_proposal(state, prop, (d_intra, d_inter), delta_bends=d_bends)
        u_after = rmc.reduced_potential_state(
            state, T, SAT, SAT_P, QTREF, QTSCALE, 1.0, 1.0, kappa_bend=kb)
        assert abs((u_after - u_before) - du) < 1e-9
        n_checked += 1
    assert n_checked > 50
    mcs.validate_state(state)
    assert np.array_equal(state.intra_contacts_by_chain,
                          mcc.full_intra_contacts_by_chain_state(state))


def test_saturating_sweep_acceptance_uses_full_potential():
    # (7) The sampler itself (not just the helpers) scores moves with the full
    # potential: with debug_contacts on, a long sweep keeps every cache exact, and
    # a strongly cooperative parameterization visibly differs from the linear
    # model run from the same seed.
    def sweep(model, p):
        state = mcs.initialize_dispersed_state(3, 8, 12, seed=42)
        counters = mvs.new_move_counters()
        rmc.mc_sweep(state, counters, 335.0, model, p, QTREF, QTSCALE,
                     lambda_intra=1.0, lambda_inter=1.0, n_local=600,
                     n_translation=120, rng=random.Random(9), n_reptation=120,
                     n_rotation=120, debug_contacts=True, kappa_bend=0.3)
        mcs.validate_state(state)
        return state, counters
    st_sat, _ = sweep(SAT, SAT_P)
    st_lin, _ = sweep("hs", HS_P)
    assert np.array_equal(st_sat.intra_contacts_by_chain,
                          mcc.full_intra_contacts_by_chain_state(st_sat))
    assert not np.array_equal(st_sat.coords_unwrapped, st_lin.coords_unwrapped), (
        "a strong cooperative attraction sampled the same trajectory as the "
        "linear model; the sweep is probably ignoring A0")


def test_saturating_full_state_swap_matches_manual_four_potential():
    # The generalized swap uses the same global total-contact potential.
    sa = mcs.initialize_dispersed_state(3, 8, 12, seed=21)
    sb = mcs.initialize_dispersed_state(3, 8, 12, seed=22)
    Ti, Tj = 305.0, 350.0

    def u(state, T):
        return rmc.reduced_contact_potential_state(
            state, T, SAT, SAT_P, QTREF, QTSCALE, 1.0, 1.0)
    manual = u(sa, Ti) + u(sb, Tj) - u(sb, Ti) - u(sa, Tj)
    got = rmc.swap_log_accept_state(sa, sb, Ti, Tj, SAT, SAT_P, QTREF, QTSCALE,
                                    1.0, 1.0)
    assert abs(got - manual) < 1e-9
    def u_literal(state, T):
        m_total = int(state.counts.intra) + int(state.counts.inter)
        return _u_sat_direct(m_total, T, state.n_chains * state.chain_length)
    literal = (u_literal(sa, Ti) + u_literal(sb, Tj)
               - u_literal(sb, Ti) - u_literal(sa, Tj))
    assert abs(got - literal) < 1e-9
    # The bending penalty is temperature-independent and cancels exactly.
    for kap in (0.0, 0.7, 2.0):
        def u_bend(state, T):
            return rmc.reduced_potential_state(
                state, T, SAT, SAT_P, QTREF, QTSCALE, 1.0, 1.0, kappa_bend=kap)
        la = (u_bend(sa, Ti) + u_bend(sb, Tj)
              - u_bend(sb, Ti) - u_bend(sa, Tj))
        assert abs(la - manual) < 1e-9
    # attempt_swap uses exactly this log-acceptance (favorable -> no draw).
    ra = rmc.MultiReplica(T=Ti, state=sa.copy())
    rb = rmc.MultiReplica(T=Tj, state=sb.copy())
    if manual >= 0.0:
        stub = _StubRng(1.0)
        assert rmc.attempt_swap(ra, rb, SAT, SAT_P, QTREF, QTSCALE, 1.0, 1.0,
                                stub) is True
        assert stub.calls == 0


def test_saturating_m1_matches_single_chain_potential():
    # (9) For M = 1 the multichain full potential equals the single-chain full
    # potential, with bending off and on.
    rng = np.random.RandomState(4)
    for _ in range(6):
        state = mcs.make_state(np.stack([mcs.generate_saw(18, rng)]), 60)
        assert int(state.counts.inter) == 0
        m = int(state.counts.intra)
        for T in (300.0, 342.0):
            for kappa in (0.0, 0.8):
                u_mc = rmc.reduced_potential_state(
                    state, T, SAT, SAT_P, QTREF, QTSCALE, 1.0, 1.0,
                    kappa_bend=kappa)
                u_ref = remd.reduced_potential_bending(
                    m, T, SAT, SAT_P, QTREF, QTSCALE, kappa_bend=kappa,
                    n_bend=state.n_bend, n_beads=state.chain_length)
                assert abs(u_mc - u_ref) < 1e-9
                assert abs(u_mc - (_u_sat_direct(m, T, state.chain_length)
                                   + kappa * state.n_bend)) < 1e-9


def test_saturating_a0_zero_matches_linear_multichain():
    # A0 = 0 nests the label-blind hs multichain model.
    state = mcs.initialize_dispersed_state(3, 8, 12, seed=11)
    for T in (300.0, 340.0):
        for li, lin in ((1.0, 1.0), (0.0, 0.0), (0.7, 0.7)):
            u_hs = rmc.reduced_contact_potential_state(
                state, T, "hs", HS_P, QTREF, QTSCALE, li, lin)
            u_a0 = rmc.reduced_contact_potential_state(
                state, T, SAT, SAT_P0, QTREF, QTSCALE, li, lin)
            assert abs(u_a0 - u_hs) < 1e-9

    def run(model, p, kappa):
        return rmc.run_remd_multichain(
            n_chains=2, chain_length=8, box_size=10, Ts=np.linspace(305, 350, 4),
            local_sweeps_per_swap=1, translation_sweeps_per_swap=1, n_cycles=15,
            model_name=model, params=p, Tref=QTREF, Tscale=QTSCALE,
            lambda_intra=1.0, lambda_inter=1.0, kappa_bend=kappa, seed=13,
            n_workers=1, verbose=False, debug_contacts=True)
    for kappa in (0.0, 0.3):
        reps_sat, sp_s, sa_s, wh_s = run(SAT, SAT_P0, kappa)
        # hs_m2_const with zero curvature takes the SAME nonlinear code path and
        # evaluates u_contact to the same bits, so this equality is exact by
        # construction rather than by luck.
        reps_q0, sp_q, sa_q, wh_q = run("hs_m2_const", CONST_P0, kappa)
        # hs takes the historical linear-aggregate path; it reassociates
        # b*m_new - b*m_old as b*delta_m, so agreement here is a value-level
        # regression rather than an arithmetic identity.
        reps_hs, sp_h, sa_h, wh_h = run("hs", HS_P, kappa)
        for reps_ref, sp_r, sa_r, wh_r, exact in (
                (reps_q0, sp_q, sa_q, wh_q, True),
                (reps_hs, sp_h, sa_h, wh_h, False)):
            assert np.array_equal(sp_s, sp_r) and np.array_equal(sa_s, sa_r)
            assert np.array_equal(wh_s, wh_r)
            for a, b in zip(reps_sat, reps_ref):
                assert np.array_equal(a.state.coords_unwrapped,
                                      b.state.coords_unwrapped)
                assert np.array_equal(a.state.intra_contacts_by_chain,
                                      b.state.intra_contacts_by_chain)
                assert a.state.n_bend == b.state.n_bend
                assert np.array_equal(a.move_counters, b.move_counters)
                if exact:
                    assert a.u_traj == b.u_traj
                else:
                    assert np.allclose(a.u_traj, b.u_traj, rtol=0, atol=1e-9)


def _run_saturating(n_workers, kappa=0.0, seed=13, n_cycles=12, params=SAT_P):
    Ts = np.linspace(305, 350, 4)
    return rmc.run_remd_multichain(
        n_chains=2, chain_length=8, box_size=10, Ts=Ts,
        local_sweeps_per_swap=1, translation_sweeps_per_swap=1,
        n_cycles=n_cycles, model_name=SAT, params=params, Tref=QTREF,
        Tscale=QTSCALE, lambda_intra=1.0, lambda_inter=1.0, kappa_bend=kappa,
        seed=seed, n_workers=n_workers, verbose=False, debug_contacts=True)


def test_saturating_bending_enabled_and_disabled():
    # (11) Runs complete with bending disabled and enabled; the recorded u
    # includes the bending term; and a positive penalty straightens the chains.
    reps0, *_ = _run_saturating(1, kappa=0.0)
    repsK, *_ = _run_saturating(1, kappa=0.6)
    for reps in (reps0, repsK):
        for rep in reps:
            mcs.validate_state(rep.state)
            assert all(math.isfinite(x) for x in rep.u_traj)
    # The recorded u is the full potential: contacts + kappa_bend * n_bend.
    rep = repsK[0]
    u_last = rmc.reduced_potential_state(
        rep.state, rep.T, SAT, SAT_P, QTREF, QTSCALE, 1.0, 1.0, kappa_bend=0.6)
    u_contacts = rmc.reduced_contact_potential_state(
        rep.state, rep.T, SAT, SAT_P, QTREF, QTSCALE, 1.0, 1.0)
    assert abs(u_last - (u_contacts + 0.6 * rep.state.n_bend)) < 1e-9
    assert abs(rep.eeff_traj[-1] - rep.T * rep.u_traj[-1]) < 1e-9
    # A strong penalty reduces the bend fraction (longer run, athermal contacts
    # so only the bending term differs).
    long_kw = dict(n_chains=3, chain_length=8, box_size=12,
                   Ts=np.linspace(320, 340, 3), local_sweeps_per_swap=3,
                   translation_sweeps_per_swap=1, n_cycles=140, model_name=SAT,
                   params=[0.0, 0.0, 0.0, 0.15], Tref=330.0, Tscale=40.0,
                   seed=321, n_workers=1, verbose=False)
    r0, *_ = rmc.run_remd_multichain(kappa_bend=0.0, **long_kw)
    rK, *_ = rmc.run_remd_multichain(kappa_bend=1.5, **long_kw)
    bf0 = np.nanmean(rmc._bend_fraction_by_temp(r0, 3, 8, 0.5))
    bfK = np.nanmean(rmc._bend_fraction_by_temp(rK, 3, 8, 0.5))
    assert bfK < bf0


def test_saturating_serial_vs_workers_determinism():
    # (12) Serial and multiprocessing runs are bit-identical, with bending off
    # and on, including the per-chain cache carried across the process boundary.
    for kappa in (0.0, 0.5):
        r1, sp1, sa1, wh1 = _run_saturating(1, kappa)
        r2, sp2, sa2, wh2 = _run_saturating(2, kappa)
        assert np.array_equal(sp1, sp2) and np.array_equal(sa1, sa2)
        assert np.array_equal(wh1, wh2)
        for a, b in zip(r1, r2):
            assert np.array_equal(a.state.coords_unwrapped,
                                  b.state.coords_unwrapped)
            assert a.state.counts.as_tuple() == b.state.counts.as_tuple()
            assert np.array_equal(a.state.intra_contacts_by_chain,
                                  b.state.intra_contacts_by_chain)
            assert a.state.n_bend == b.state.n_bend
            assert a.u_traj == b.u_traj
            assert np.array_equal(a.move_counters, b.move_counters)
            mcs.validate_state(a.state)


def test_aggregate_helpers_reject_saturating_cooperative():
    # (13) The aggregate-ContactCounts helpers must refuse the saturating model
    # too: the total intrachain count cannot determine the sum of the per-chain
    # nonlinear potentials.
    c = ContactCounts(4, 2)
    with pytest.raises(NotImplementedError, match="nonlinear in m"):
        rmc.reduced_potential_counts(c, 330.0, SAT, SAT_P, QTREF, QTSCALE,
                                     1.0, 1.0)
    with pytest.raises(NotImplementedError, match="nonlinear in m"):
        rmc.reduced_potential_bending_counts(
            c, 330.0, SAT, SAT_P, QTREF, QTSCALE, 1.0, 1.0, kappa_bend=0.3,
            n_bend=5)
    with pytest.raises(NotImplementedError, match="nonlinear in m"):
        rmc.swap_log_accept_counts(c, c, 300.0, 350.0, SAT, SAT_P, QTREF,
                                   QTSCALE, 1.0, 1.0)
    # Even A0 = 0 is refused: the guard keys on the model's potential_kind, not
    # on a parameter value that happens to linearize it.
    with pytest.raises(NotImplementedError, match="nonlinear in m"):
        rmc.reduced_potential_counts(c, 330.0, SAT, SAT_P0, QTREF, QTSCALE,
                                     1.0, 1.0)
    # The linear models still work through the aggregate helpers.
    assert math.isfinite(rmc.reduced_potential_counts(
        c, 330.0, "hs", HS_P, QTREF, QTSCALE, 1.0, 1.0))


def test_saturating_fit_summary_roundtrip_and_output_metadata(tmp_path):
    # (14) A saturating fit_summary.json drives a multichain run end to end, and
    # the run summary + distributions NPZ record the potential that was sampled.
    import json as _json
    fit_N = 30
    summary_path = _write_fit_summary(
        tmp_path / "sat_summary.json", SAT,
        {"h_b": 700.0, "s_b": 2.4, "A0": 5.0, "q_sat": 0.15},
        kappa_bend=0.4, fit_chain_length=fit_N)
    loaded = remd.load_fit_summary_json(str(summary_path))
    assert loaded["model_name"] == SAT
    assert loaded["fit_chain_length"] == fit_N
    assert loaded["kappa_bend"] == 0.4
    assert [float(v) for v in loaded["params"]] == SAT_P

    prefix = str(tmp_path / "sat_run")
    rmc.main([
        "--n-chains", "2", "--N", "8", "--box-size", "12",
        "--n-cycles", "6", "--nT", "3", "--Tmin", "310", "--Tmax", "345",
        "--fit-summary-json", str(summary_path),
        "--seed", "1", "--no-plots", "--out-prefix", prefix])
    with open(f"{prefix}_run_summary.json") as fh:
        s = _json.load(fh)
    assert s["model"] == SAT
    assert s["model_api_version"] == remd.MODEL_API_VERSION
    assert s["potential_kind"] == "saturating_cooperative"
    assert s["potential_definition"] == remd.SATURATING_COOPERATIVE_DEFINITION
    assert s["potential_normalization"] == remd.Q_NORMALIZATION == "q = m/N"
    assert s["m_ref"] == 0
    assert s["multichain_potential_definition"] == \
        rmc.MULTICHAIN_POTENTIAL_DEFINITION
    assert s["nonlinear_contact_scope"] == "all_contacts_global"
    assert s["quadratic_contact_scope"] == "all_contacts_global"
    assert s["interchain_contact_model"] == "same_single_chain_potential"
    # No m^2/(2N) term exists in this family, so the quadratic fields stay empty.
    assert s["quadratic_normalization"] is None
    assert all(v == 0.0 for v in s["quadratic_coefficient_by_temperature"])
    assert s["A0"] == 5.0 and s["q_sat"] == 0.15
    assert s["params"] == SAT_P
    assert s["runtime_chain_length"] == 8 and s["fit_chain_length"] == fit_N
    assert s["kappa_bend"] == 0.4 and s["bending_enabled"] is True

    # allow_pickle=False proves no None/object arrays leaked into the NPZ.
    with np.load(f"{prefix}_distributions.npz", allow_pickle=False) as d:
        assert str(d["potential_kind"]) == "saturating_cooperative"
        assert str(d["potential_definition"]) == \
            remd.SATURATING_COOPERATIVE_DEFINITION
        assert str(d["potential_normalization"]) == "q = m/N"
        assert int(d["m_ref"]) == 0
        assert str(d["multichain_potential_definition"]) == \
            rmc.MULTICHAIN_POTENTIAL_DEFINITION
        assert str(d["nonlinear_contact_scope"]) == "all_contacts_global"
        assert str(d["interchain_contact_model"]) == \
            "same_single_chain_potential"
        assert str(d["quadratic_normalization"]) == ""
        assert float(d["A0"]) == 5.0 and float(d["q_sat"]) == 0.15
        assert int(d["runtime_chain_length"]) == 8
        assert int(d["fit_chain_length"]) == fit_N
        assert np.all(d["quadratic_coefficient_by_temperature"] == 0.0)
        assert np.array_equal(d["reduced_bias_by_temperature"],
                              d["linear_coefficient_by_temperature"])
        assert float(d["kappa_bend"]) == 0.4


def test_saturating_invalid_parameters_rejected():
    # (15) The domain constraints A0 >= 0 and q_sat > 0 are enforced when the
    # parameters are resolved, not silently at the first potential evaluation.
    for bad in ([700.0, 2.4, -1.0, 0.15], [700.0, 2.4, 5.0, 0.0],
                [700.0, 2.4, 5.0, -0.15]):
        with pytest.raises(ValueError):
            remd.validate_model_params(SAT, bad, "test")
        with pytest.raises(ValueError):
            rmc.reduced_contact_potential_state(
                mcs.initialize_dispersed_state(2, 8, 12, seed=3), 330.0, SAT,
                bad, QTREF, QTSCALE, 1.0, 1.0)


def test_quadratic_metadata_unchanged_for_legacy_models(tmp_path):
    # (16) The generalized metadata is purely additive: a contact-quadratic run
    # still records exactly the legacy contract values.
    import json as _json
    prefix = str(tmp_path / "legacy")
    rmc.main([
        "--n-chains", "2", "--N", "8", "--box-size", "12", "--n-cycles", "6",
        "--nT", "3", "--Tmin", "310", "--Tmax", "345",
        "--model", "hs_m2_const", "--params", "700,2.4,0.9",
        "--Tref", "320", "--Tscale", "80",
        "--seed", "2", "--no-plots", "--out-prefix", prefix])
    with open(f"{prefix}_run_summary.json") as fh:
        s = _json.load(fh)
    assert s["potential_kind"] == "contact_quadratic"
    assert s["quadratic_normalization"] == "m_chain^2/(2*N)"
    assert s["quadratic_contact_scope"] == "intra_per_chain"
    assert s["potential_normalization"] == "m^2/(2N)"
    assert s["m_ref"] == 0
    assert "A0" not in s and "q_sat" not in s
    with np.load(f"{prefix}_distributions.npz", allow_pickle=False) as d:
        assert str(d["quadratic_normalization"]) == "m_chain^2/(2*N)"
        assert "A0" not in d and "q_sat" not in d
    # And a linear run keeps the empty-string sentinel.
    prefix2 = str(tmp_path / "lin")
    rmc.main([
        "--n-chains", "2", "--N", "8", "--box-size", "12", "--n-cycles", "6",
        "--nT", "3", "--Tmin", "310", "--Tmax", "345",
        "--model", "hs", "--params", "700,2.4", "--Tref", "320",
        "--Tscale", "80", "--seed", "2", "--no-plots", "--out-prefix", prefix2])
    with np.load(f"{prefix2}_distributions.npz", allow_pickle=False) as d:
        assert str(d["quadratic_normalization"]) == ""
        assert str(d["potential_normalization"]) == ""


if __name__ == "__main__":
    sys.exit(pytest.main([os.path.abspath(__file__), "-q"]))
