"""Part 5: aggregate_lane_observables must reject unequal / misaligned seed grids
instead of silently truncating to the shortest common length."""
import numpy as np
import pytest

import run_structural_regime_pilot as cal


BASE = [280.0, 300.0, 320.0, 340.0]


def _rows(temps, shift=0.0, drop_col=None):
    out = []
    for i, t in enumerate(temps):
        r = {"T": float(t), "C_mean": 2.0 + shift + i, "Rg2_mean": 20.0 - i,
             "m_global_scaled_mean": 0.2 * i, "Smax_mean": 1.0 * i,
             "largest_component_fraction_mean": 0.05 * i, "C_std": 1.0}
        if drop_col is not None:
            r.pop(drop_col, None)
        out.append(r)
    return out


@pytest.fixture
def patched(monkeypatch):
    store = {}
    monkeypatch.setattr(cal, "_read_results", lambda p: store[p])
    monkeypatch.setattr(cal, "_K_of", lambda info, t: float(t))
    return store


def test_valid_two_seeds_aggregates(patched):
    patched["s1"] = _rows(BASE, 0.0)
    patched["s2"] = _rows(BASE, 0.5)
    out = cal.aggregate_lane_observables(["s1", "s2"], info=None, n_beads=30)
    assert out["n_temperatures"] == 4 and out["n_seeds"] == 2


def test_seed_order_invariant_for_valid_inputs(patched):
    patched["s1"] = _rows(BASE, 0.0)
    patched["s2"] = _rows(BASE, 0.5)
    fwd = cal.aggregate_lane_observables(["s1", "s2"], info=None, n_beads=30)
    rev = cal.aggregate_lane_observables(["s2", "s1"], info=None, n_beads=30)
    assert np.allclose(fwd["aggregate"]["m"], rev["aggregate"]["m"])
    assert np.allclose(fwd["temperatures"], rev["temperatures"])


def test_missing_last_row_rejected(patched):
    patched["s1"] = _rows(BASE, 0.0)
    patched["s2"] = _rows(BASE[:-1], 0.5)          # one lane short
    with pytest.raises(cal.SeedAlignmentError):
        cal.aggregate_lane_observables(["s1", "s2"], info=None, n_beads=30)


def test_extra_row_rejected(patched):
    patched["s1"] = _rows(BASE, 0.0)
    patched["s2"] = _rows(BASE + [360.0], 0.5)     # one extra lane
    with pytest.raises(cal.SeedAlignmentError):
        cal.aggregate_lane_observables(["s1", "s2"], info=None, n_beads=30)


def test_shifted_temperature_rejected(patched):
    patched["s1"] = _rows(BASE, 0.0)
    patched["s2"] = _rows([280.0, 300.0, 320.0, 341.0], 0.5)  # one shifted lane
    with pytest.raises(cal.SeedAlignmentError):
        cal.aggregate_lane_observables(["s1", "s2"], info=None, n_beads=30)


def test_duplicate_temperature_rejected(patched):
    patched["s1"] = _rows(BASE, 0.0)
    patched["s2"] = _rows([280.0, 300.0, 300.0, 340.0], 0.5)  # duplicate lane
    with pytest.raises(cal.SeedAlignmentError):
        cal.aggregate_lane_observables(["s1", "s2"], info=None, n_beads=30)


def test_missing_observable_column_rejected(patched):
    patched["s1"] = _rows(BASE, 0.0)
    patched["s2"] = _rows(BASE, 0.5, drop_col="C_mean")
    with pytest.raises(cal.SeedAlignmentError):
        cal.aggregate_lane_observables(["s1", "s2"], info=None, n_beads=30)
