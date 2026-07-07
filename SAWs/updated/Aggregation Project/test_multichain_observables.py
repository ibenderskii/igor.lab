#!/usr/bin/env python3
"""pytest suite for multichain_observables (Stage 2).

Run:  python -m pytest test_multichain_observables.py -q
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import multichain_state as mcs
import multichain_contacts as mcc
import multichain_observables as obs


def straight_chain(n, axis=0, start=(0, 0, 0)):
    coords = np.tile(np.asarray(start, dtype=np.int64), (n, 1))
    coords[:, axis] += np.arange(n, dtype=np.int64)
    return coords


def test_f_inter_zero_convention():
    assert obs.f_inter(mcc.ContactCounts(0, 0)) == 0.0
    assert obs.f_inter(mcc.ContactCounts(3, 1)) == 0.25
    assert obs.f_inter(mcc.ContactCounts(0, 4)) == 1.0


def test_chain_geometry_summary_straight():
    coords = np.stack([straight_chain(5), straight_chain(5, start=(0, 5, 0))])
    state = mcs.make_state(coords, 40)
    g = obs.chain_geometry_summary(state)
    # Both chains identical straight 5-mers: Rg^2 = (25-1)/12 = 2.
    assert abs(g["mean_chain_rg2_lattice"] - 2.0) < 1e-9
    assert abs(g["mean_chain_rg_lattice"] - np.sqrt(2.0)) < 1e-9
    assert abs(g["std_chain_rg_lattice"]) < 1e-9  # identical chains
    assert g["per_chain_rg_lattice"].shape == (2,)


def test_cluster_summary_threshold():
    # chains 0,1 parallel (5 interchain contacts); chain 2 isolated.
    c0 = straight_chain(5, axis=0, start=(0, 0, 0))
    c1 = straight_chain(5, axis=0, start=(0, 1, 0))
    c2 = straight_chain(5, axis=0, start=(0, 10, 0))
    state = mcs.make_state(np.stack([c0, c1, c2]), 30)

    s1 = obs.cluster_summary(state, cluster_contact_threshold=1)
    assert s1["largest_cluster_size"] == 2
    assert s1["n_clusters"] == 2
    assert abs(s1["largest_cluster_fraction"] - 2.0 / 3.0) < 1e-12

    # Threshold above the shared-contact count breaks the edge.
    s6 = obs.cluster_summary(state, cluster_contact_threshold=6)
    assert s6["largest_cluster_size"] == 1
    assert s6["n_clusters"] == 3
    assert abs(s6["largest_cluster_fraction"] - 1.0 / 3.0) < 1e-12


def test_cluster_summary_all_aggregated():
    # Three chains stacked in y, each adjacent to the next -> one cluster.
    chains = [straight_chain(5, axis=0, start=(0, k, 0)) for k in range(3)]
    state = mcs.make_state(np.stack(chains), 30)
    s = obs.cluster_summary(state, 1)
    assert s["largest_cluster_size"] == 3
    assert s["n_clusters"] == 1
    assert s["largest_cluster_fraction"] == 1.0


def test_cluster_threshold_validation():
    state = mcs.make_state(np.stack([straight_chain(4), straight_chain(4, start=(0, 5, 0))]), 20)
    with pytest.raises(ValueError):
        obs.cluster_summary(state, 0)


def _fake_traj(nT=3, n_cycles=40, seed=0):
    rng = np.random.RandomState(seed)
    traj = []
    for _ in range(nT):
        n = n_cycles
        traj.append({
            "u": rng.rand(n) - 0.5,
            "effective_energy": rng.rand(n),
            "m_intra": rng.randint(0, 5, n),
            "m_inter": rng.randint(0, 4, n),
            "f_inter": rng.rand(n),
            "mean_chain_rg_lattice": 1.0 + rng.rand(n),
            "mean_chain_rg2_lattice": 1.0 + rng.rand(n),
            "largest_cluster_size": rng.randint(1, 4, n),
            "largest_cluster_fraction": rng.rand(n),
        })
    return traj


def test_compute_statistics_keys_and_finite():
    traj = _fake_traj()
    temps = [300.0, 320.0, 340.0]
    res = obs.compute_statistics(traj, temps, burnin_frac=0.5, rg_scale=0.5)
    assert len(res) == 3
    for r in res:
        for k in ("T", "u_mean", "effective_energy_mean", "m_intra_mean",
                  "m_inter_mean", "m_total_mean", "Rg_mean", "Rg2_mean",
                  "f_inter_mean", "largest_cluster_size_mean",
                  "largest_cluster_fraction_mean"):
            assert k in r
        # rg_scale applied to output Rg.
        assert abs(r["Rg_mean"] - 0.5 * r["Rg_mean_lattice"]) < 1e-9
        assert abs(r["m_total_mean"] - (r["m_intra_mean"] + r["m_inter_mean"])) < 1e-9


def test_build_distributions_normalized_and_keys():
    traj = _fake_traj()
    temps = [300.0, 320.0, 340.0]
    d = obs.build_distributions(traj, temps, burnin_frac=0.5, rg_bins=20,
                                fmax_bins=20, rg_scale=1.0)
    for key in ("Ts", "m_intra_vals", "P_m_intra", "m_inter_vals", "P_m_inter",
                "rg_edges", "rg_centers", "P_mean_chain_rg",
                "fmax_edges", "fmax_centers", "P_fmax"):
        assert key in d, f"missing distributions key {key}"
    for name in ("P_m_intra", "P_m_inter", "P_mean_chain_rg", "P_fmax"):
        for row in d[name]:
            finite = row[np.isfinite(row)]
            if finite.size:
                assert abs(float(finite.sum()) - 1.0) < 1e-9, f"{name} not normalized"


def test_build_distributions_rg_scale_axis():
    traj = _fake_traj()
    temps = [300.0, 320.0, 340.0]
    d = obs.build_distributions(traj, temps, rg_scale=0.5)
    np.testing.assert_allclose(d["rg_edges"], 0.5 * d["rg_edges_lattice"])
    np.testing.assert_allclose(d["rg_centers"], 0.5 * d["rg_centers_lattice"])


# ---------------------------------------------------------------------------
# Per-chain structural observables (Change 3)
# ---------------------------------------------------------------------------

def _fake_traj_with_chains(nT=3, n_cycles=40, M=4, seed=1):
    """Fake trajectory that also carries per-chain Rg / std / n_clusters / M."""
    rng = np.random.RandomState(seed)
    traj = []
    for _ in range(nT):
        n = n_cycles
        per_chain = [1.0 + rng.rand(M) for _ in range(n)]
        traj.append({
            "u": rng.rand(n) - 0.5,
            "effective_energy": rng.rand(n),
            "m_intra": rng.randint(0, 5, n),
            "m_inter": rng.randint(0, 4, n),
            "f_inter": rng.rand(n),
            "mean_chain_rg_lattice": np.array([v.mean() for v in per_chain]),
            "mean_chain_rg2_lattice": 1.0 + rng.rand(n),
            "std_chain_rg_lattice": np.array([v.std(ddof=0) for v in per_chain]),
            "per_chain_rg_lattice": per_chain,
            "largest_cluster_size": rng.randint(1, M + 1, n),
            "largest_cluster_fraction": rng.rand(n),
            "n_clusters": rng.randint(1, M + 1, n),
            "n_chains": M,
        })
    return traj


def test_both_rg_distributions_present_and_normalized():
    traj = _fake_traj_with_chains()
    temps = [300.0, 320.0, 340.0]
    d = obs.build_distributions(traj, temps, burnin_frac=0.5, rg_bins=25,
                                rg_scale=1.0)
    # Both distributions saved and semantically distinct keys present.
    for key in ("P_mean_chain_rg", "P_chain_rg", "chain_rg_edges",
                "chain_rg_centers", "chain_rg_edges_lattice",
                "chain_rg_centers_lattice"):
        assert key in d, f"missing distribution key {key}"
    for name in ("P_mean_chain_rg", "P_chain_rg", "P_m_intra", "P_m_inter",
                 "P_fmax"):
        for row in d[name]:
            finite = row[np.isfinite(row)]
            if finite.size:
                assert abs(float(finite.sum()) - 1.0) < 1e-9, f"{name} not normalized"


def test_chain_rg_axis_scales_only_length():
    traj = _fake_traj_with_chains()
    temps = [300.0, 320.0, 340.0]
    d1 = obs.build_distributions(traj, temps, rg_scale=1.0)
    d2 = obs.build_distributions(traj, temps, rg_scale=0.5)
    # Length axes scale; probability mass (the histogram itself) is unchanged.
    np.testing.assert_allclose(d2["chain_rg_edges"], 0.5 * d2["chain_rg_edges_lattice"])
    np.testing.assert_allclose(d2["chain_rg_centers"], 0.5 * d2["chain_rg_centers_lattice"])
    np.testing.assert_allclose(d1["P_chain_rg"], d2["P_chain_rg"], equal_nan=True)
    np.testing.assert_allclose(d1["chain_rg_edges_lattice"],
                               d2["chain_rg_edges_lattice"])


def test_compute_statistics_has_heterogeneity_columns():
    traj = _fake_traj_with_chains()
    temps = [300.0, 320.0, 340.0]
    res = obs.compute_statistics(traj, temps, burnin_frac=0.5, rg_scale=2.0)
    for r in res:
        for k in ("std_chain_rg_mean", "std_chain_rg_mean_lattice",
                  "n_clusters_mean"):
            assert k in r
        # std_chain_rg is a length: rg_scale applies.
        assert abs(r["std_chain_rg_mean"] - 2.0 * r["std_chain_rg_mean_lattice"]) < 1e-9
        assert r["n_clusters_mean"] >= 1.0


def test_compute_statistics_tolerates_missing_chain_keys():
    # A bare fake trajectory (no per-chain keys) must not raise; new columns NaN.
    traj = _fake_traj()
    res = obs.compute_statistics(traj, [300.0, 320.0, 340.0], burnin_frac=0.5)
    for r in res:
        assert np.isnan(r["std_chain_rg_mean"])
        assert np.isnan(r["n_clusters_mean"])
        assert np.isnan(r["m_intra_per_chain_mean"])  # no M -> NaN


# ---------------------------------------------------------------------------
# Contact normalization by number of chains (Part 2)
# ---------------------------------------------------------------------------

def test_per_chain_contacts_known_state():
    # M=4, m_intra=8, m_inter=12.
    d = obs.per_chain_contacts(8, 12, 4)
    assert d["m_intra_per_chain"] == 2.0
    assert d["m_inter_pairs_per_chain"] == 3.0
    assert d["m_inter_incidences_per_chain"] == 6.0
    assert d["m_total_pairs_per_chain"] == 5.0


def test_cycle_observables_normalized_relations():
    # For any real state the normalized fields equal raw/M cycle-by-cycle.
    for M in (2, 4):
        state = mcs.initialize_dispersed_state(M, 6, 12, seed=3 + M)
        cyc = obs.cycle_observables(state)
        mi, me = cyc["m_intra"], cyc["m_inter"]
        assert cyc["m_intra_per_chain"] == mi / M
        assert cyc["m_inter_pairs_per_chain"] == me / M
        assert cyc["m_inter_incidences_per_chain"] == 2.0 * me / M
        assert cyc["m_total_pairs_per_chain"] == (mi + me) / M


def test_compute_statistics_normalized_means_two_M():
    temps = [300.0, 320.0, 340.0]
    for M in (2, 5):
        traj = _fake_traj_with_chains(M=M, seed=M)
        res = obs.compute_statistics(traj, temps, burnin_frac=0.5)
        for r in res:
            assert abs(r["m_intra_per_chain_mean"] - r["m_intra_mean"] / M) < 1e-12
            assert abs(r["m_intra_per_chain_std"] - r["m_intra_std"] / M) < 1e-12
            assert abs(r["m_inter_pairs_per_chain_mean"] - r["m_inter_mean"] / M) < 1e-12
            assert abs(r["m_inter_pairs_per_chain_std"] - r["m_inter_std"] / M) < 1e-12
            assert abs(r["m_inter_incidences_per_chain_mean"]
                       - 2.0 * r["m_inter_mean"] / M) < 1e-12
            assert abs(r["m_total_pairs_per_chain_mean"] - r["m_total_mean"] / M) < 1e-12


def test_compute_statistics_explicit_n_chains_arg():
    # M supplied via the argument (not the lane dict) is honoured.
    traj = _fake_traj()  # no per-lane n_chains
    res = obs.compute_statistics(traj, [300.0, 320.0, 340.0], burnin_frac=0.5,
                                 n_chains=3)
    for r in res:
        assert abs(r["m_intra_per_chain_mean"] - r["m_intra_mean"] / 3.0) < 1e-12


def test_build_distributions_normalized_axes_reuse_P():
    traj = _fake_traj_with_chains(M=4)
    d = obs.build_distributions(traj, [300.0, 320.0, 340.0])
    np.testing.assert_allclose(d["m_intra_per_chain_vals"], d["m_intra_vals"] / 4.0)
    np.testing.assert_allclose(d["m_inter_pairs_per_chain_vals"], d["m_inter_vals"] / 4.0)
    np.testing.assert_allclose(d["m_inter_incidences_per_chain_vals"],
                               2.0 * d["m_inter_vals"] / 4.0)
    # Probability matrices are reused, not recomputed.
    assert d["P_m_intra_per_chain"] is d["P_m_intra"] or np.array_equal(
        d["P_m_intra_per_chain"], d["P_m_intra"])
    assert np.array_equal(d["P_m_inter_pairs_per_chain"], d["P_m_inter"])
    assert np.array_equal(d["P_m_inter_incidences_per_chain"], d["P_m_inter"])


def test_normalized_contacts_invariant_to_rg_scale():
    traj = _fake_traj_with_chains(M=4)
    temps = [300.0, 320.0, 340.0]
    r1 = obs.compute_statistics(traj, temps, rg_scale=1.0)
    r2 = obs.compute_statistics(traj, temps, rg_scale=0.5)
    for a, b in zip(r1, r2):
        for k in ("m_intra_mean", "m_inter_mean", "m_total_mean",
                  "m_intra_per_chain_mean", "m_inter_pairs_per_chain_mean",
                  "m_inter_incidences_per_chain_mean", "m_total_pairs_per_chain_mean"):
            assert a[k] == b[k], f"{k} changed with rg_scale"


if __name__ == "__main__":
    sys.exit(pytest.main([os.path.abspath(__file__), "-q"]))
