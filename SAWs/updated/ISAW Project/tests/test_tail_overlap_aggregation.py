"""Phase 9/10/14: one common tail threshold + pooled/worst-seed overlap gates."""
import numpy as np
import pytest

import run_structural_regime_pilot as cal


def _pc(counts_row):
    c = np.asarray(counts_row, float)
    s = c.sum()
    return c / s if s > 0 else c


# --- one common tail threshold + exact pooled==sum accounting ----------------

def test_pooled_tail_equals_sum_of_seed_tails():
    c_vals = np.arange(6, dtype=float)
    # two seeds, two lanes each; raw counts vary between seeds
    seed0 = np.array([_pc([10, 5, 3, 2, 1, 0]), _pc([1, 2, 3, 4, 5, 5])])
    seed1 = np.array([_pc([8, 6, 4, 2, 0, 0]), _pc([0, 1, 2, 3, 6, 8])])
    counts = [[100.0, 200.0], [120.0, 180.0]]
    ess = [[50.0, 80.0], [60.0, 70.0]]
    out = cal.common_tail_statistics([seed0, seed1], counts, ess, c_vals)
    assert out["pooled_raw_equals_seed_sum"] is True
    assert abs(out["raw_tail_count_pooled"]
               - sum(out["raw_tail_count_per_seed"])) < 1e-6


def test_one_common_threshold_used_for_all_seeds():
    c_vals = np.arange(6, dtype=float)
    seed0 = np.array([_pc([10, 5, 3, 2, 1, 0])])
    seed1 = np.array([_pc([0, 0, 1, 2, 3, 10])])   # very different shape
    out = cal.common_tail_statistics([seed0, seed1], [[100.0], [100.0]],
                                     [[50.0], [50.0]], c_vals)
    # exactly one threshold, applied to both seeds
    assert isinstance(out["tail_threshold"], float)
    assert out["threshold_source"] == "pooled_sampled_percentile"
    assert len(out["raw_tail_count_per_seed"]) == 2


def test_user_supplied_threshold_source():
    c_vals = np.arange(6, dtype=float)
    seed0 = np.array([_pc([10, 5, 3, 2, 1, 0])])
    out = cal.common_tail_statistics([seed0], [[100.0]], [[50.0]], c_vals,
                                     threshold=4.0)
    assert out["tail_threshold"] == 4.0
    assert out["threshold_source"] == "user_supplied"


def test_support_boundary_unavailable_by_default():
    c_vals = np.arange(6, dtype=float)
    seed0 = np.array([_pc([10, 5, 3, 2, 1, 0])])
    out = cal.common_tail_statistics([seed0], [[100.0]], [[50.0]], c_vals)
    assert out["support_boundary_source"] == "unavailable"
    assert out["maximum_is_boundary_limited"] is None


def test_support_boundary_external_reference():
    c_vals = np.arange(6, dtype=float)
    seed0 = np.array([_pc([0, 0, 0, 0, 1, 10])])   # max observed m == 5
    out = cal.common_tail_statistics([seed0], [[100.0]], [[50.0]], c_vals,
                                     support_maximum=5.0, support_source="exact")
    assert out["support_boundary_source"] == "exact"
    assert out["maximum_is_boundary_limited"] is True


# --- tail gate --------------------------------------------------------------

def test_tail_gate_requires_pooled_and_per_seed_support():
    good = {"raw_tail_count_pooled": 100.0, "effective_tail_count_pooled": 20.0,
            "raw_tail_count_per_seed": [40.0, 60.0], "n_seeds": 2,
            "seed_coverage": 2}
    assert cal.evaluate_tail_gate(good)["passed"] is True
    # one seed with no tail support -> fails per-seed requirement
    bad = dict(good, raw_tail_count_per_seed=[0.0, 100.0], seed_coverage=1)
    assert cal.evaluate_tail_gate(bad)["passed"] is False


# --- pooled vs worst-seed overlap gates -------------------------------------

def test_pooled_overlap_can_hide_worst_seed():
    # pooled overlap healthy but one seed has a poorly connected gap
    metrics = {"min_adjacent_overlap": 0.5,
               "min_worst_seed_adjacent_overlap": 0.02}
    g = cal.evaluate_sampling_gates(metrics)
    assert g["min_adjacent_overlap"]["passed"] is True
    assert g["worst_seed_adjacent_overlap"]["passed"] is False


def test_both_overlap_gates_pass_when_healthy():
    metrics = {"min_adjacent_overlap": 0.5,
               "min_worst_seed_adjacent_overlap": 0.4}
    g = cal.evaluate_sampling_gates(metrics)
    assert g["min_adjacent_overlap"]["passed"] is True
    assert g["worst_seed_adjacent_overlap"]["passed"] is True
