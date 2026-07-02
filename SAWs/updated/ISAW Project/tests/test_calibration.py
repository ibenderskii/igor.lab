"""Calibration analysis helpers (Parts 3-10): deterministic unit tests."""
import numpy as np
import pytest

import run_structural_regime_pilot as cal


# --- branch detection -------------------------------------------------------

def _kinfo(T, K):
    return {"T": np.asarray(T, float), "K": np.asarray(K, float)}


def test_monotonic_increasing_single_branch():
    T = np.linspace(280, 360, 50)
    K = 2.0 - 100.0 / T
    assert len(cal.monotonic_branches(T, K)) == 1
    br = cal.select_branch(_kinfo(T, K), "auto")
    assert br["K_direction"] == "increasing" and br["K_increases_with_T"]


def test_monotonic_decreasing_single_branch():
    T = np.linspace(280, 360, 50)
    K = -(2.0 - 100.0 / T)
    br = cal.select_branch(_kinfo(T, K), "auto")
    assert br["K_direction"] == "decreasing" and not br["K_increases_with_T"]


def test_nonmonotonic_low_high_branches():
    T = np.linspace(0, 10, 101)
    K = -(T - 5.0) ** 2
    brs = cal.monotonic_branches(T, K)
    assert len(brs) == 2
    low = cal.select_branch(_kinfo(T, K), "low")
    high = cal.select_branch(_kinfo(T, K), "high")
    assert low["branch_index"] == 0 and high["branch_index"] == 1
    assert low["K_direction"] == "increasing"
    assert high["K_direction"] == "decreasing"


def test_auto_fails_on_ambiguous():
    T = np.linspace(0, 10, 101)
    K = -(T - 5.0) ** 2
    with pytest.raises(SystemExit):
        cal.select_branch(_kinfo(T, K), "auto")
    br = cal.select_branch(_kinfo(T, K), "auto", physical_branch=1)
    assert br["branch_index"] == 1


# --- K ladder (Part 9) ------------------------------------------------------

def test_ladder_exact_size_and_endpoints():
    T = np.linspace(280, 360, 200)
    K = 2.0 - 200.0 / T
    info = cal.build_k_ladder(T, K, 8)
    assert info["unique_count"] == 8 and len(info["temperatures"]) == 8
    assert abs(info["temperatures"][0] - 280.0) < 1e-6
    assert abs(info["temperatures"][-1] - 360.0) < 1e-6
    assert info["exact_count"] and info["endpoints_included"]


def test_ladder_increasing_K_recomputed_from_model():
    # k_of_T recomputes K by CALLING the model, not interpolating an array.
    T = np.linspace(280, 360, 200)
    K = 2.0 - 200.0 / T
    model = lambda t: 2.0 - 200.0 / t     # noqa: E731
    info = cal.build_k_ladder(T, K, 6, k_of_T=model)
    for t, k in zip(info["temperatures"], info["K_values"]):
        assert abs(k - model(t)) < 1e-9


def test_ladder_decreasing_K_inversion():
    # Decreasing K(T): the inversion must not pass a descending x to interp.
    T = np.linspace(280, 360, 200)
    K = -(2.0 - 200.0 / T)                # decreasing in T
    model = lambda t: -(2.0 - 200.0 / t)  # noqa: E731
    info = cal.build_k_ladder(T, K, 7, k_of_T=model)
    temps = np.asarray(info["temperatures"])
    assert info["K_direction"] == "decreasing"
    assert np.all(np.diff(temps) > 0)                 # temperatures ascending
    assert abs(temps[0] - 280.0) < 1e-6 and abs(temps[-1] - 360.0) < 1e-6
    for t, k in zip(temps, info["K_values"]):
        assert abs(k - model(t)) < 1e-9               # K matches the model


def test_ladder_nonmonotonic_low_and_high_branch_ladders():
    T = np.linspace(0.0, 10.0, 201)
    K = -(T - 5.0) ** 2                    # up then down
    low = cal.select_branch(_kinfo(T, K), "low")
    high = cal.select_branch(_kinfo(T, K), "high")
    li = cal.build_k_ladder(low["Ts"], low["K"], 5)
    hi = cal.build_k_ladder(high["Ts"], high["K"], 5)
    assert li["K_direction"] == "increasing"
    assert hi["K_direction"] == "decreasing"
    assert np.all(np.diff(li["temperatures"]) > 0)
    assert np.all(np.diff(hi["temperatures"]) > 0)


def test_ladder_spacing_enforcement_flags():
    T = np.linspace(300, 300.5, 200)      # tiny interval
    K = 2.0 - 200.0 / T
    info = cal.build_k_ladder(T, K, 8, min_dT=1.0)
    assert info["unique_count"] == 8
    assert info["min_dT_ok"] is False
    assert info["spacing_ok"] is False
    assert info["recommendation"]


# --- statistics (Part 4/7) --------------------------------------------------

def test_spearman_with_ties():
    x = [1, 2, 2, 3, 4]; y = [10, 20, 20, 30, 40]
    assert abs(cal.spearman(x, y) - 1.0) < 1e-9
    x2 = [1, 2, 2, 3]; y2 = [1, 3, 2, 4]
    r = cal.spearman(x2, y2)
    try:
        from scipy.stats import spearmanr
        assert abs(r - spearmanr(x2, y2).correlation) < 1e-9
    except Exception:
        assert -1 <= r <= 1


def test_ci_excludes_zero():
    assert cal.ci_excludes_zero(0.5, 2.0)
    assert cal.ci_excludes_zero(-2.0, -0.5)
    assert not cal.ci_excludes_zero(-0.5, 0.5)


def test_block_bootstrap_positive_series():
    x = np.full(100, 3.0) + np.linspace(-0.01, 0.01, 100)
    m, lo, hi = cal.block_bootstrap_mean(x, n_blocks=10)
    assert lo <= m <= hi and lo > 0


def _make_seed(shift, rng, n=200, nT=4):
    """Synthetic per-lane trajectories with a strong low->high-K trend."""
    K = np.array([-0.4, -0.2, 0.0, 0.2])
    base_c = np.array([2.0, 4.0, 8.0, 16.0]) + shift
    base_r = np.array([20.0, 14.0, 8.0, 3.0]) + shift
    contacts = np.array([base_c[k] + rng.normal(0, 0.05, n) for k in range(nT)])
    rg2 = np.array([base_r[k] + rng.normal(0, 0.05, n) for k in range(nT)])
    mg = np.array([(0.1 * base_c[k]) + rng.normal(0, 0.01, n) for k in range(nT)])
    sm = np.array([base_c[k] + rng.normal(0, 0.05, n) for k in range(nT)])
    return {"K": K, "contacts": contacts, "rg2": rg2, "m_global": mg, "smax": sm}


def test_hierarchical_bootstrap_signs_and_fields():
    rng = np.random.RandomState(0)
    seed_data = [_make_seed(0.0, rng), _make_seed(0.3, rng)]
    hb = cal.hierarchical_bootstrap(seed_data, n_boot=500, block_len=20)
    for q in ("delta_contacts", "delta_rg2", "slope_contacts", "slope_rg2",
              "slope_m_global", "slope_smax"):
        e = hb[q]
        assert set(e) >= {"estimate", "bootstrap_mean", "ci95", "n_seeds",
                          "block_length", "n_bootstrap_replicates"}
        assert e["n_seeds"] == 2 and e["n_bootstrap_replicates"] == 500
    # PNIPAM support: Δ<m> CI>0 and Δ<Rg2> CI<0.
    assert hb["delta_contacts"]["ci95"][0] > 0
    assert hb["delta_rg2"]["ci95"][1] < 0


def test_hierarchical_bootstrap_deterministic():
    rng1 = np.random.RandomState(1); rng2 = np.random.RandomState(1)
    sd1 = [_make_seed(0.0, rng1), _make_seed(0.3, rng1)]
    sd2 = [_make_seed(0.0, rng2), _make_seed(0.3, rng2)]
    a = cal.hierarchical_bootstrap(sd1, n_boot=200, block_len=10, seed=7)
    b = cal.hierarchical_bootstrap(sd2, n_boot=200, block_len=10, seed=7)
    assert a["delta_contacts"]["ci95"] == b["delta_contacts"]["ci95"]


# --- regime classification + seed aggregation (Part 5) ----------------------

def test_regime_classification_spanning():
    K = np.linspace(-0.5, 0.5, 6)
    m = np.linspace(1, 20, 6); rg2 = np.linspace(20, 2, 6)
    within = np.full(6, 1.0)
    out = cal.classify_regimes(K, m, rg2, within_lane_std=within, min_effect_size=2.0)
    assert "swollen" in out["labels"] and "collapsed" in out["labels"]
    assert out["distinct_regimes_resolved"] is True


def test_regime_not_resolved_small_effect():
    K = np.linspace(-0.5, 0.5, 6)
    m = np.linspace(5, 5.2, 6); rg2 = np.linspace(5, 4.9, 6)
    within = np.full(6, 5.0)
    out = cal.classify_regimes(K, m, rg2, within_lane_std=within, min_effect_size=2.0)
    assert out["distinct_regimes_resolved"] is False


def _canned_rows(shift):
    T = [280.0, 300.0, 320.0, 340.0]
    cm = [2.0, 6.0, 12.0, 20.0]
    rg = [20.0, 14.0, 8.0, 3.0]
    return [{"T": T[k], "C_mean": cm[k] + shift, "Rg2_mean": rg[k],
             "m_global_scaled_mean": 0.1 * cm[k], "Smax_mean": cm[k],
             "largest_component_fraction_mean": 0.05 * cm[k], "C_std": 1.0}
            for k in range(4)]


def test_regime_aggregation_seed_order_invariant(monkeypatch):
    canned = {"s1": _canned_rows(0.0), "s2": _canned_rows(0.5)}
    monkeypatch.setattr(cal, "_read_results", lambda p: canned[p])
    monkeypatch.setattr(cal, "_K_of", lambda info, t: float(t))   # K = T
    fwd = cal.aggregate_lane_observables(["s1", "s2"], info=None, n_beads=30)
    rev = cal.aggregate_lane_observables(["s2", "s1"], info=None, n_beads=30)
    lab_f = cal.classify_regimes(fwd["K"], fwd["aggregate"]["m"],
                                 fwd["aggregate"]["rg2"],
                                 within_lane_std=fwd["within_lane_std"])["labels"]
    lab_r = cal.classify_regimes(rev["K"], rev["aggregate"]["m"],
                                 rev["aggregate"]["rg2"],
                                 within_lane_std=rev["within_lane_std"])["labels"]
    assert lab_f == lab_r
    assert np.allclose(fwd["aggregate"]["m"], rev["aggregate"]["m"])


# --- high-contact tail + support boundary (Part 7) --------------------------

def test_tail_threshold_and_unavailable_boundary():
    c_vals = np.arange(10)
    Pc = np.zeros((2, 10))
    Pc[0, :3] = [0.5, 0.3, 0.2]
    Pc[1, 7:] = [0.5, 0.3, 0.2]
    t = cal.high_contact_tail(Pc, c_vals, ess_by_lane=[50, 50],
                              n_by_lane=[100, 100], tau_by_lane=[1, 1])
    assert 0 <= t["tail_threshold"] <= 9
    assert t["maximum_observed_m"] == 9
    # No external support reference -> boundary is unavailable, NOT inferred.
    assert t["support_boundary_source"] == "unavailable"
    assert t["maximum_is_boundary_limited"] is None


def test_tail_boundary_limited_with_external_support():
    c_vals = np.arange(10)
    Pc = np.zeros((2, 10)); Pc[0, :3] = [0.5, 0.3, 0.2]; Pc[1, 7:] = [0.5, 0.3, 0.2]
    t = cal.high_contact_tail(Pc, c_vals, [50, 50], [100, 100], [1, 1],
                              support_maximum=9, support_source="baseline")
    assert t["support_boundary_source"] == "baseline"
    assert t["maximum_is_boundary_limited"] is True
    t2 = cal.high_contact_tail(Pc, c_vals, [50, 50], [100, 100], [1, 1],
                               support_maximum=50, support_source="external")
    assert t2["maximum_is_boundary_limited"] is False


# --- production estimator (Part 8) ------------------------------------------

def test_production_requirement_per_regime():
    regime_map = {"swollen": [0, 1], "crossover": [2], "collapsed": [3, 4]}
    tau = [2.0, 2.0, 5.0, 30.0, 30.0]
    out = cal.production_requirement(regime_map, tau, post_burnin_frac=0.5,
                                     snapshot_stride=5, n_seeds=4, target_ess=5000)
    assert out["limiting_regime"] == "collapsed"
    assert out["per_regime"]["collapsed"]["required_production_cycles"] \
        >= out["per_regime"]["swollen"]["required_production_cycles"]
    assert out["every_regime_reaches_target"] is True


def test_production_infeasible_when_tau_missing():
    regime_map = {"swollen": [0], "crossover": [], "collapsed": [1]}
    out = cal.production_requirement(regime_map, [2.0, 3.0], 0.5, 5, 2)
    assert out["every_regime_reaches_target"] is False


def test_production_snapshot_stride_changes_recommendation():
    regime_map = {"swollen": [0], "crossover": [1], "collapsed": [2]}
    tau = [2.0, 2.0, 2.0]     # 2*tau = 4
    o1 = cal.production_requirement(regime_map, tau, 0.5, 1, 4)
    o10 = cal.production_requirement(regime_map, tau, 0.5, 10, 4)
    o100 = cal.production_requirement(regime_map, tau, 0.5, 100, 4)
    p1 = o1["recommended_production_cycles_per_seed"]
    p10 = o10["recommended_production_cycles_per_seed"]
    p100 = o100["recommended_production_cycles_per_seed"]
    # stride > 2*tau makes saving coarser -> more cycles required.
    assert p1 <= p10 < p100
    # raw saved-config counts change with the stride.
    r1 = o1["per_regime"]["swollen"]["expected_raw_saved_configs_per_seed"]
    r100 = o100["per_regime"]["swollen"]["expected_raw_saved_configs_per_seed"]
    assert r1 != r100


# --- sampling gates + band gate (Part 10) -----------------------------------

def test_gates_pass_and_fail():
    good = {"min_ess": 500, "min_round_trips": 10, "min_temp_coverage": 0.9,
            "min_adjacent_overlap": 0.5, "min_worst_seed_adjacent_overlap": 0.3,
            "min_swap_rate": 0.4, "max_swap_rate": 0.6,
            "max_drift_in_std": 0.2, "min_state_changing_acceptance": 0.1,
            "tail_probability": 0.05, "raw_tail_count_pooled": 100.0,
            "seed_relative_spread": 0.1}
    g = cal.evaluate_sampling_gates(good)
    assert g["_all_passed"] is True
    bad = dict(good, min_ess=5)
    g2 = cal.evaluate_sampling_gates(bad)
    assert g2["_all_passed"] is False and g2["min_ess"]["passed"] is False


def test_swap_rate_is_a_band_gate():
    metrics = {"min_swap_rate": 0.05, "max_swap_rate": 0.95}
    g = cal.evaluate_sampling_gates(metrics)["swap_rate"]
    # records BOTH bounds (Part 10)
    for key in ("minimum_value", "minimum_threshold", "maximum_value",
                "maximum_threshold", "passed"):
        assert key in g
    assert g["passed"] is False       # 0.95 > max threshold 0.90
