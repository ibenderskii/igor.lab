"""Phase 9: the strong tail gate is authoritative in the scientific gate.

The overall gate must fail when the pooled effective count, seed coverage,
minimum per-seed tail, or the common-threshold accounting identity fails -- not
just when a probability is low.
"""
import run_structural_regime_pilot as cal


def _tail(**over):
    base = dict(raw_tail_count_pooled=100.0, effective_tail_count_pooled=50.0,
                raw_tail_count_per_seed=[40.0, 60.0], n_seeds=2, seed_coverage=2,
                pooled_raw_equals_seed_sum=True)
    base.update(over)
    return base


def _gates_with(tail):
    metrics = {"min_ess": 500, "min_round_trips": 10, "min_temp_coverage": 0.9,
               "min_adjacent_overlap": 0.5, "min_worst_seed_adjacent_overlap": 0.3,
               "min_swap_rate": 0.4, "max_swap_rate": 0.6, "max_drift_in_std": 0.2,
               "min_state_changing_acceptance": 0.1, "seed_relative_spread": 0.1,
               "tail_gate": cal.evaluate_tail_gate(tail)}
    return cal.evaluate_sampling_gates(metrics)


def test_all_criteria_pass():
    g = _gates_with(_tail())
    assert g["high_contact_tail_support"]["passed"] is True
    assert g["_all_passed"] is True


def test_effective_count_fail_fails_gate():
    # raw count passes but effective count is below threshold
    g = _gates_with(_tail(effective_tail_count_pooled=1.0))
    assert g["high_contact_tail_support"]["passed"] is False
    assert g["_all_passed"] is False


def test_one_seed_zero_tail_fails_gate():
    # pooled support fine, but one seed has zero tail samples
    g = _gates_with(_tail(raw_tail_count_per_seed=[100.0, 0.0], seed_coverage=1))
    assert g["high_contact_tail_support"]["passed"] is False


def test_seed_coverage_fail_fails_gate():
    g = _gates_with(_tail(raw_tail_count_per_seed=[100.0, 0.0, 0.0, 0.0],
                          n_seeds=4, seed_coverage=1))
    assert g["high_contact_tail_support"]["passed"] is False


def test_accounting_identity_fail_fails_gate():
    g = _gates_with(_tail(pooled_raw_equals_seed_sum=False))
    assert g["high_contact_tail_support"]["passed"] is False
    assert g["_all_passed"] is False


def test_evaluate_tail_gate_reports_accounting():
    tg = cal.evaluate_tail_gate(_tail(pooled_raw_equals_seed_sum=False))
    assert tg["pooled_raw_equals_seed_sum"] is False and tg["passed"] is False
