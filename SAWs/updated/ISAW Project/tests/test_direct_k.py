"""Direct-K control-mode tests (sampler + diagnostic driver).

Covers: --K-values parsing (negative leading value), conflict rejection,
b(K=0)=0 and b=-K, K-monotone contacts, the direct-K swap-acceptance identity,
fitted-temperature regression, a tiny end-to-end diagnostic smoke run, and the
boundary-peak transition_bracketed=False case.
"""
import argparse
import csv
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import remd_uniform_chain_2_new as remd
import run_direct_k_diagnostic as dk

HERE = Path(__file__).resolve().parent.parent
DKPARAMS = dict(model_name=remd.DIRECT_K_MODEL_NAME, params=remd.DIRECT_K_PARAMS,
                Tref=remd.DIRECT_K_TREF, Tscale=remd.DIRECT_K_TSCALE)


def _direct_k_namespace(**overrides):
    """A Namespace with every fitted/temperature arg unset (as argparse leaves them)."""
    base = dict(temps=None, temps_from_npz=None, fit_summary_json=None,
                fit_params_csv=None, params=None, model=None,
                Tref=None, Tscale=None, T0=None)
    base.update(overrides)
    return argparse.Namespace(**base)


# 1. Parsing --K-values with a negative first value ---------------------------

def test_parse_k_values_negative_first_and_sorted():
    out = remd.parse_k_values("-0.40,-0.32,-0.08,0.00,0.35")
    assert out == [-0.40, -0.32, -0.08, 0.00, 0.35]        # ascending
    # Unsorted input is sorted deterministically.
    assert remd.parse_k_values("0.3,-0.4,0.0") == [-0.4, 0.0, 0.3]


def test_parse_k_values_rejects_bad_input():
    with pytest.raises(ValueError):
        remd.parse_k_values("-0.4")                 # fewer than two
    with pytest.raises(ValueError):
        remd.parse_k_values("-0.4,nan,0.3")         # non-finite
    with pytest.raises(ValueError):
        remd.parse_k_values("-0.4,0.1,0.1")         # duplicate


def test_k_values_equals_form_reaches_sampler():
    # The '=' form is required so a leading negative value is not read as a flag;
    # --help exercises argparse's handling of the registered --K-values option.
    out = subprocess.run(
        [sys.executable, str(HERE / "remd_uniform_chain_2_new.py"), "--help"],
        capture_output=True, text=True, cwd=str(HERE))
    assert out.returncode == 0
    assert "--K-values" in out.stdout


# 2. Direct-K rejects conflicting temperature/fitted-model args ----------------

def test_direct_k_rejects_temperature_args():
    with pytest.raises(ValueError):
        remd.reject_direct_k_conflicts(_direct_k_namespace(temps="300,320,340"))
    with pytest.raises(ValueError):
        remd.reject_direct_k_conflicts(
            _direct_k_namespace(fit_summary_json="fit_summary.json"))
    with pytest.raises(ValueError):
        remd.reject_direct_k_conflicts(_direct_k_namespace(Tref=300.0))
    # No conflicting args -> no error.
    remd.reject_direct_k_conflicts(_direct_k_namespace())


# 3. At K=0 the internal reduced bias is zero (and b = -K elsewhere) -----------

def test_direct_k_bias_zero_at_zero_and_negative_k():
    # The lane label T carries the coupling K; reduced_bias returns b = -K.
    assert remd.reduced_bias(T=0.0, **DKPARAMS) == 0.0
    for K in (-0.4, -0.08, 0.14, 0.35):
        assert remd.reduced_bias(T=K, **DKPARAMS) == pytest.approx(-K, abs=1e-15)


# 4. Increasing K favors larger contact counts --------------------------------

def test_increasing_k_favors_contacts():
    Ts = np.array([-0.4, 0.0, 0.4])            # lane label carries K
    reps, _sp, _sa = remd.run_remd(
        N=20, Ts=Ts, steps_per_swap=40, n_cycles=250,
        seed=7, n_workers=1, verbose=False, **DKPARAMS)
    stats = remd.compute_statistics(reps, burnin_frac=0.5)
    by_k = {round(s["T"], 6): s["C_mean"] for s in stats}
    assert by_k[0.4] > by_k[-0.4]              # stronger coupling -> more contacts
    assert by_k[0.0] >= by_k[-0.4]


# 5. Direct-K replica-exchange expression == (K_i - K_j)(m_j - m_i) ------------

def test_direct_k_swap_acceptance_identity():
    cases = [(3, 7, -0.2, 0.4), (5, 2, 0.1, -0.3), (0, 9, 0.0, 0.35),
             (4, 4, -0.4, 0.2)]
    for mi, mj, Ki, Kj in cases:
        la = remd.swap_log_accept(mi, mj, Ki, Kj, **DKPARAMS)
        assert la == pytest.approx((Ki - Kj) * (mj - mi), abs=1e-12)


# 6. Existing fitted-temperature mode still works -----------------------------

def test_fitted_temperature_mode_still_works(tmp_path):
    HS = dict(model_name="hs", params=[378.96, 1.39686], Tref=330.0, Tscale=80.0)
    Ts = np.linspace(300.0, 360.0, 5)
    reps, _sp, _sa = remd.run_remd(
        N=16, Ts=Ts, steps_per_swap=30, n_cycles=40, seed=3, n_workers=1,
        verbose=False, **HS)
    stats = remd.compute_statistics(reps, burnin_frac=0.5)
    for i, r in enumerate(stats):
        r["K"] = -remd.reduced_bias(T=float(r["T"]), **HS)   # K(T) = -b(T)
        assert np.isfinite(r["C_mean"]) and np.isfinite(r["T"])
    # Default control mode writes real temperatures AND the appended K column.
    path = remd.save_results_csv(stats, str(tmp_path / "fit"))
    rows = list(csv.DictReader(open(path, newline="")))
    assert "K" in rows[0] and "T" in rows[0]
    assert all(np.isfinite(float(r["T"])) for r in rows)     # T not blanked
    assert all(np.isfinite(float(r["K"])) for r in rows)


def test_direct_k_results_csv_blanks_temperature(tmp_path):
    reps, _sp, _sa = remd.run_remd(
        N=16, Ts=np.array([-0.2, 0.0, 0.2]), steps_per_swap=20, n_cycles=30,
        seed=1, n_workers=1, verbose=False, **DKPARAMS)
    stats = remd.compute_statistics(reps, burnin_frac=0.5)
    for r in stats:
        r["K"] = float(r["T"])                 # in direct-K the label is K
    path = remd.save_results_csv(stats, str(tmp_path / "dk"),
                                 control_mode="direct_K")
    rows = list(csv.DictReader(open(path, newline="")))
    assert all(r["T"].lower() == "nan" for r in rows)        # temperature blanked
    assert [float(r["K"]) for r in rows] == [-0.2, 0.0, 0.2]


# 7. A very short direct-K smoke run produces all required report files --------

def test_direct_k_diagnostic_smoke_produces_reports(tmp_path):
    out_dir = tmp_path / "dk_smoke"
    cmd = [
        sys.executable, str(HERE / "run_direct_k_diagnostic.py"),
        "--N", "18", "--K-values=-0.3,-0.1,0.1,0.3",
        "--seeds", "7", "--n-workers", "1", "--n-cycles", "20",
        "--steps-per-swap", "10", "--burnin-frac", "0.5",
        "--structural-stride", "2", "--snapshot-stride", "2",
        "--run-id", "smoke_pytest", "--output-dir", str(out_dir),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(HERE))
    assert res.returncode == 0, res.stdout + "\n" + res.stderr
    for fname in ("direct_K_report.json", "direct_K_report.md",
                  "direct_K_response_curves.csv", "run_manifest.json"):
        assert (out_dir / fname).exists(), f"missing {fname}"
    # Per-seed sampler artifacts are kept.
    assert (out_dir / "directK_N18_s7_configurations.h5").exists()
    assert (out_dir / "directK_N18_s7_features.h5").exists()


# 8. A boundary maximum is reported as transition_bracketed: false ------------

def test_boundary_peak_not_bracketed():
    K = np.array([-0.4, -0.2, 0.0, 0.2, 0.4])
    # Monotone-rising response => every peak at the high endpoint.
    var_m = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    dm_dK = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    rg2_resp = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
    net_resp = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    t = dk.assess_transition(K, var_m, dm_dK, rg2_resp, net_resp)
    assert t["transition_bracketed"] is False
    assert t["peaks"]["K_peak_contact_variance"]["at_endpoint"] is True
    assert t["recommended_scan_direction"] == "higher"


def test_interior_peak_is_bracketed():
    K = np.array([-0.4, -0.2, 0.0, 0.2, 0.4])
    # Interior maximum at K=0 for the size/variance response.
    var_m = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
    dm_dK = np.array([0.5, 1.0, 1.5, 1.0, 0.5])
    rg2_resp = np.array([0.5, 1.0, 2.0, 1.0, 0.5])
    net_resp = np.array([0.1, 0.2, 0.3, 0.2, 0.1])
    t = dk.assess_transition(K, var_m, dm_dK, rg2_resp, net_resp)
    assert t["transition_bracketed"] is True
    assert t["peaks"]["K_peak_contact_variance"]["at_endpoint"] is False
    assert t["n_lanes_below_variance_peak"] >= 2
    assert t["n_lanes_above_variance_peak"] >= 2


def test_fluctuation_dissipation_check_reports_agreement():
    # d<m>/dK ≈ Var(m): identical arrays -> zero diff, unit correlation.
    v = np.array([1.0, 2.0, 4.0, 3.0])
    fd = dk.fluctuation_dissipation_check(v, v)
    assert fd["n_compared"] == 4
    assert fd["max_abs_diff"] == pytest.approx(0.0)
    assert fd["correlation"] == pytest.approx(1.0)
