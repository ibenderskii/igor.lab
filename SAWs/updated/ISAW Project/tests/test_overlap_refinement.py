"""Phase 10: ladder refinement uses pooled, worst-seed, and disagreement overlap.

A lane is deleted as redundant only when pooled AND worst-seed overlap are high;
a poor worst-seed gap gets an insert even if pooled overlap is fine; high seed
disagreement is flagged; endpoints are preserved.
"""
import run_structural_regime_pilot as cal


def _gap(pooled, per_seed):
    return {"overlap_per_seed": list(per_seed),
            "pooled_count_overlap": pooled,
            "minimum_overlap": min(per_seed),
            "median_overlap": sorted(per_seed)[len(per_seed) // 2]}


def _causes(recs):
    return {r["cause"] for r in recs}


def test_healthy_pooled_and_worst_no_action():
    temps = [300.0, 320.0, 340.0]
    gaps = [_gap(0.5, [0.5, 0.55]), _gap(0.5, [0.52, 0.48])]
    out = cal.refine_ladder(temps, gaps)
    assert out["recommendations"] == []
    assert out["refined_temperatures"] == temps


def test_poor_worst_seed_inserts_even_if_pooled_ok():
    temps = [300.0, 320.0, 340.0]
    # pooled overlap looks fine (0.5) but one seed has near-zero connectivity
    gaps = [_gap(0.5, [0.9, 0.02]), _gap(0.5, [0.5, 0.5])]
    out = cal.refine_ladder(temps, gaps)
    assert "poor worst-seed overlap" in _causes(out["recommendations"])
    assert 310.0 in out["refined_temperatures"]  # inserted midpoint


def test_poor_pooled_inserts():
    temps = [300.0, 340.0]
    gaps = [_gap(0.05, [0.05, 0.05])]
    out = cal.refine_ladder(temps, gaps)
    assert "poor pooled overlap" in _causes(out["recommendations"])
    assert 320.0 in out["refined_temperatures"]


def test_high_disagreement_flagged():
    temps = [300.0, 340.0]
    gaps = [_gap(0.5, [0.95, 0.4])]   # spread 0.55 >= 0.25
    out = cal.refine_ladder(temps, gaps)
    assert "high seed disagreement" in _causes(out["recommendations"])


def test_redundant_only_when_pooled_and_worst_high():
    temps = [300.0, 320.0, 340.0]
    # pooled high but worst-seed low => NOT deleted (insert instead)
    gaps = [_gap(0.95, [0.95, 0.05]), _gap(0.5, [0.5, 0.5])]
    out = cal.refine_ladder(temps, gaps)
    actions = {r["action"] for r in out["recommendations"]}
    assert "delete_redundant_lane" not in actions
    assert 320.0 in out["refined_temperatures"]


def test_redundant_pooled_and_worst_high_deletes_interior():
    temps = [300.0, 320.0, 340.0]
    # gap 0 (300-320) redundant across all seeds -> drops interior lane 320
    gaps = [_gap(0.95, [0.95, 0.96]), _gap(0.5, [0.5, 0.5])]
    out = cal.refine_ladder(temps, gaps)
    actions = {r["action"] for r in out["recommendations"]}
    assert "delete_redundant_lane" in actions
    assert 320.0 not in out["refined_temperatures"]
    # endpoints survive
    assert 300.0 in out["refined_temperatures"]
    assert 340.0 in out["refined_temperatures"]


def test_endpoint_preserved_even_if_redundant():
    temps = [300.0, 320.0]
    # the only interior candidate is endpoint 320 -> preserved, not dropped
    gaps = [_gap(0.98, [0.98, 0.99])]
    out = cal.refine_ladder(temps, gaps)
    actions = {r["action"] for r in out["recommendations"]}
    assert "preserve_endpoint" in actions
    assert 320.0 in out["refined_temperatures"]
    assert 300.0 in out["refined_temperatures"]


def test_legacy_float_list_still_supported():
    out = cal.refine_ladder([300.0, 320.0, 340.0], [0.05, 0.5])
    assert 310.0 in out["refined_temperatures"]
