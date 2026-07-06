"""Phase 11/12/14: production estimator responds to seeds/stride/burn-in/tau,
identifies the limiting observable, and reports provisional when unstable.
"""
import numpy as np

import run_structural_regime_pilot as cal


def _regimes():
    return {"swollen": [0, 1], "crossover": [2, 3], "collapsed": [4, 5]}


TAU = [2.0, 2.0, 3.0, 3.0, 5.0, 5.0]


def test_production_seeds_change_result():
    r4 = cal.production_requirement(_regimes(), TAU, 0.5, 1, n_seeds=4)
    r8 = cal.production_requirement(_regimes(), TAU, 0.5, 1, n_seeds=8)
    # more production seeds -> fewer required cycles per seed
    assert r8["recommended_production_cycles_per_seed"] < \
        r4["recommended_production_cycles_per_seed"]


def test_snapshot_stride_changes_result():
    r1 = cal.production_requirement(_regimes(), TAU, 0.5, 1, n_seeds=4)
    r10 = cal.production_requirement(_regimes(), TAU, 0.5, 10, n_seeds=4)
    r100 = cal.production_requirement(_regimes(), TAU, 0.5, 100, n_seeds=4)
    p1 = r1["recommended_production_cycles_per_seed"]
    p10 = r10["recommended_production_cycles_per_seed"]
    p100 = r100["recommended_production_cycles_per_seed"]
    # Delta_eff = max(stride, tau_int_cycles); non-decreasing in stride and
    # strictly larger once stride exceeds tau (max tau 5).
    assert p1 <= p10 < p100


def test_burnin_fraction_changes_result():
    hi = cal.production_requirement(_regimes(), TAU, 0.8, 1, n_seeds=4)
    lo = cal.production_requirement(_regimes(), TAU, 0.2, 1, n_seeds=4)
    # a smaller usable (post-burn-in) fraction needs MORE cycles
    assert lo["recommended_production_cycles_per_seed"] > \
        hi["recommended_production_cycles_per_seed"]


def test_tau_changes_result():
    slow = cal.production_requirement(_regimes(), [4, 4, 6, 6, 10, 10], 0.5, 1,
                                      n_seeds=4)
    fast = cal.production_requirement(_regimes(), [1, 1, 1, 1, 1, 1], 0.5, 1,
                                      n_seeds=4)
    assert slow["recommended_production_cycles_per_seed"] > \
        fast["recommended_production_cycles_per_seed"]


def test_limiting_observable_identified():
    obs_by_lane = ["contacts", "contacts", "rg2", "rg2", "smax", "smax"]
    r = cal.production_requirement(_regimes(), TAU, 0.5, 1, n_seeds=4,
                                   limiting_observable_by_lane=obs_by_lane)
    # collapsed regime's slowest lane (tau 5) is a smax lane
    assert r["per_regime"]["collapsed"]["limiting_observable"] == "smax"


def test_limiting_lane_identified():
    # lane 5 has the max tau in the collapsed regime {4,5}
    tau = [2.0, 2.0, 3.0, 3.0, 4.0, 9.0]
    r = cal.production_requirement(_regimes(), tau, 0.5, 1, n_seeds=4)
    assert r["per_regime"]["collapsed"]["limiting_lane"] == 5
    assert r["per_regime"]["collapsed"]["limiting_tau_int_cycles"] == 9.0


def test_stride_changes_raw_and_effective_counts():
    r1 = cal.production_requirement(_regimes(), TAU, 0.5, 1, n_seeds=4,
                                    target_ess=10)
    r100 = cal.production_requirement(_regimes(), TAU, 0.5, 100, n_seeds=4,
                                      target_ess=10)
    a = r1["per_regime"]["collapsed"]
    b = r100["per_regime"]["collapsed"]
    # coarser stride -> larger effective spacing -> more required cycles, and the
    # raw saved-per-lane count reflects the stride.
    assert b["required_production_cycles"] > a["required_production_cycles"]
    assert "expected_raw_saved_configs_per_lane_per_seed" in a
    assert "expected_effective_configs_per_lane_per_seed" in a


def test_regime_lane_count_changes_result():
    one = cal.production_requirement({"r": [0]}, TAU, 0.5, 1, n_seeds=4,
                                     target_ess=10)
    many = cal.production_requirement({"r": [0, 1, 2, 3]}, TAU, 0.5, 1, n_seeds=4,
                                      target_ess=10)
    # more lanes in a regime -> fewer cycles needed to reach the same pooled ESS
    assert many["recommended_production_cycles_per_seed"] < \
        one["recommended_production_cycles_per_seed"]


def test_unstable_regimes_are_provisional():
    stable = cal.production_requirement(_regimes(), TAU, 0.5, 1, n_seeds=8,
                                        target_ess=1, regime_labels_stable=True)
    unstable = cal.production_requirement(_regimes(), TAU, 0.5, 1, n_seeds=8,
                                          target_ess=1, regime_labels_stable=False)
    assert stable["status"] == "definitive"
    assert unstable["status"] == "provisional"
    for rr in unstable["per_regime"].values():
        assert rr["status"] == "provisional"


def test_number_of_seeds_recorded_per_regime():
    r = cal.production_requirement(_regimes(), TAU, 0.5, 1, n_seeds=6)
    for rr in r["per_regime"].values():
        assert rr["number_of_seeds"] == 6
