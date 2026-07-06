"""Part 8/12: production status is 'definitive' ONLY for a fully-passed,
multi-seed scientific calibration; every other case is 'provisional'."""
import run_structural_regime_pilot as cal


REGIME_MAP = {"swollen": [0], "crossover": [1], "collapsed": [2]}
TAU = [2.0, 2.0, 2.0]


def _base_production():
    # A production dict with a real numeric recommendation.
    return cal.production_requirement(
        REGIME_MAP, TAU, post_burnin_frac=0.5, snapshot_stride=1, n_seeds=8,
        target_ess=100, regime_labels_stable=True)


def _finalize(**over):
    conds = dict(
        smoke_test=False, n_complete_calibration_seeds=2,
        n_requested_calibration_seeds=2, all_requested_seeds_valid=True,
        seeds_aligned=True, statistical_support_status="supported",
        regime_labels_stable=True, sampling_gates_passed=True,
        calibration_gate_passed=True)
    conds.update(over)
    return cal.finalize_production_status(_base_production(), **conds)


def test_fully_passed_is_definitive():
    p = _finalize()
    assert p["status"] == "definitive"
    assert p["status_reasons"] == []
    assert p["numeric_recommendation_is_provisional"] is False
    # numeric recommendation still present
    assert p["recommended_production_cycles_per_seed"] > 0


def test_one_seed_is_provisional():
    p = _finalize(n_complete_calibration_seeds=1, n_requested_calibration_seeds=1,
                  statistical_support_status="not_assessable")
    assert p["status"] == "provisional"
    # a single seed is never "stable merely because it agrees with itself"
    assert p["regime_label_stability_status"] == "not_assessable"
    assert p["recommended_production_cycles_per_seed"] > 0  # still available


def test_two_requested_one_incomplete_is_provisional():
    p = _finalize(n_complete_calibration_seeds=1, n_requested_calibration_seeds=2,
                  all_requested_seeds_valid=False,
                  statistical_support_status="not_assessable")
    assert p["status"] == "provisional"
    assert p["n_complete_calibration_seeds"] == 1


def test_two_complete_but_misaligned_is_provisional():
    p = _finalize(seeds_aligned=False, calibration_gate_passed=False)
    assert p["status"] == "provisional"
    assert any("aligned" in r for r in p["status_reasons"])


def test_two_aligned_failed_sampling_gate_is_provisional():
    p = _finalize(sampling_gates_passed=False, calibration_gate_passed=False)
    assert p["status"] == "provisional"
    assert any("sampling gates" in r for r in p["status_reasons"])


def test_two_aligned_unstable_regimes_is_provisional():
    p = _finalize(regime_labels_stable=False, calibration_gate_passed=False)
    assert p["status"] == "provisional"
    assert p["regime_label_stability_status"] == "unstable"


def test_smoke_mode_is_provisional():
    p = _finalize(smoke_test=True, calibration_gate_passed=False)
    assert p["status"] == "provisional"
    assert any("smoke" in r for r in p["status_reasons"])


def test_per_regime_status_forced_provisional_when_overall_provisional():
    p = _finalize(smoke_test=True, calibration_gate_passed=False)
    for rr in p["per_regime"].values():
        assert rr["status"] == "provisional"
