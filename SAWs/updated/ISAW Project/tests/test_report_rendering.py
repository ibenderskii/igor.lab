"""Part 1: gate rendering is robust to structured gates (no KeyError) and run
finalization is transactional (a failed report leaves status=failed)."""
import json
import tempfile
from pathlib import Path

import pytest

import run_structural_regime_pilot as cal


def _tail_gate():
    # The strong tail gate carries NONE of value/threshold/minimum_value.
    return cal.evaluate_tail_gate({
        "raw_tail_count_pooled": 100.0, "effective_tail_count_pooled": 50.0,
        "raw_tail_count_per_seed": [40.0, 60.0], "n_seeds": 2,
        "seed_coverage": 2, "pooled_raw_equals_seed_sum": True})


def test_format_gate_handles_strong_tail_gate():
    s = cal._format_gate(_tail_gate())
    assert "raw_pooled" in s and "accounting_ok" in s     # no KeyError


def test_format_gate_handles_scalar_and_band():
    scalar = cal._gate(True, 0.5, 0.1, "x")
    band = cal._band_gate(0.4, 0.1, 0.6, 0.9, "y")
    assert "value=" in cal._format_gate(scalar)
    assert "min=" in cal._format_gate(band) and "max=" in cal._format_gate(band)


def test_format_gate_handles_unknown_structured_gate():
    weird = {"passed": True, "raw_tail_count_pooled_missing": 1,
             "some_new_field": [1, 2, 3], "nested": {"a": 1}}
    # unknown shape without value/threshold/minimum_value/raw_tail_count_pooled
    weird2 = {"passed": False, "foo": 1, "bar": {"b": 2}}
    s = cal._format_gate(weird2)
    assert "foo" in s and "bar" in s                       # deterministic JSON


def test_console_and_md_render_structured_tail_gate():
    # Mimic evaluate_sampling_gates output containing the strong tail gate.
    gates = {"high_contact_tail_support": _tail_gate(),
             "min_ess": cal._gate(True, 500, 200, "ess"),
             "swap_rate": cal._band_gate(0.4, 0.1, 0.6, 0.9, "swap"),
             "_all_passed": True}
    # console-style loop (as in main) must not raise
    for name, gr in gates.items():
        if name.startswith("_"):
            continue
        line = f"gate {name}: passed={gr.get('passed')} " + cal._format_gate(gr)
        assert isinstance(line, str)
    # MD-style rendering (as in _write_report_md) must not raise either
    md = [f"- {k}: passed={v.get('passed')} " + cal._format_gate(v)
          for k, v in gates.items() if not k.startswith("_")]
    assert len(md) == 3


def _minimal_report():
    tg = _tail_gate()
    return {
        "N": 30, "model": "hs", "run_id": "r1",
        "scientifically_validated": False,
        "fitted_temperature_interval": [300.0, 360.0],
        "branch_selection": {"branch_index": 0},
        "scientific_conclusions": {"calibration_gate_passed": False},
        "trends": {"statistical_support_status": "not_assessable",
                   "endpoint_delta_m_point": 1.0, "endpoint_delta_Rg2_point": -1.0,
                   "hierarchical_bootstrap": None},
        "regimes": {"labels": ["swollen", "collapsed"], "rule": "r"},
        "sampling_gates": {"high_contact_tail_support": tg,
                           "min_ess": cal._gate(True, 500, 200, "ess"),
                           "_all_passed": False},
        "recommended_production": {
            "status": "provisional", "status_reasons": ["smoke"],
            "recommended_production_cycles_per_seed": 123,
            "limiting_regime": "collapsed", "every_regime_reaches_target": False},
    }


def test_write_report_md_does_not_raise_on_structured_gate():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "report.md"
        cal._write_report_md(p, _minimal_report())     # must not raise KeyError
        text = p.read_text(encoding="utf-8")
        assert "high_contact_tail_support" in text
        assert "raw_pooled" in text


def test_finalize_run_marks_complete_when_reports_written():
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        manifest = {"status": "running", "run_id": "r"}
        report = _minimal_report()
        cal.finalize_run(out, manifest, report, report["recommended_production"])
        man = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
        assert man["status"] == "complete"
        assert (out / "calibration_report.json").exists()
        assert (out / "calibration_report.md").exists()
        assert (out / "recommended_production_config.json").exists()


def test_finalize_run_leaves_status_failed_on_report_exception(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        manifest = {"status": "running", "run_id": "r"}
        report = _minimal_report()

        def _boom(path, rep):
            raise RuntimeError("markdown boom")
        monkeypatch.setattr(cal, "_write_report_md", _boom)
        with pytest.raises(RuntimeError):
            cal.finalize_run(out, manifest, report,
                             report["recommended_production"])
        man = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
        assert man["status"] == "failed"
        assert man["failure_stage"] == "report_generation"
        assert man["failure_exception_type"] == "RuntimeError"
        assert "boom" in man["failure_exception_message"]
