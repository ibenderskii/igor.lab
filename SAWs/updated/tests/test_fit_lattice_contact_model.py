"""Tests for fit_lattice_contact_model.py."""
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).parent.parent))

from fit_lattice_contact_model import (
    centers_to_edges,
    pdf_to_mass,
    rebin_pdf_mass_to_integer_bins,
    rebin_pdf_to_mass,
    kl_div,
    js_div,
    model_contact_mass,
    objective,
    make_b_fn,
    predict_rg_from_joint,
    MODEL_REGISTRY,
)


# ---------------------------------------------------------------------------
# Rebinning
# ---------------------------------------------------------------------------

def test_rebin_integer_bins_preserves_mass():
    """Rebinning a smooth pdf onto integer bins preserves total probability mass."""
    centers = np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
    pdf = np.array([0.05, 0.15, 0.35, 0.28, 0.12, 0.05])
    pdf = pdf / pdf.sum()
    _, p_int = rebin_pdf_mass_to_integer_bins(centers, pdf)
    assert abs(p_int.sum() - 1.0) < 1e-10


def test_rebin_integer_bins_non_unit_offset():
    """Rebinning works when the offset shifts centers to non-zero integers."""
    # Simulate a contact distribution with offset applied (centers near 0–16)
    centers = np.linspace(0.1, 16.9, 80)
    pdf = np.exp(-0.5 * ((centers - 8.0) / 3.0) ** 2)
    pdf = pdf / pdf.sum()
    m_vals, p_int = rebin_pdf_mass_to_integer_bins(centers, pdf, m_min=0, m_max=16)
    assert m_vals.shape == p_int.shape
    assert abs(p_int.sum() - 1.0) < 1e-10


def test_rebin_pdf_to_mass_normalizes():
    """General rebinning normalizes output to sum 1 when any overlap exists."""
    centers = np.linspace(0.0, 5.0, 100)
    pdf = np.exp(-0.5 * ((centers - 2.5) / 0.8) ** 2)
    pdf = pdf / pdf.sum()
    target_edges = np.linspace(1.0, 4.0, 7)   # 6 bins, within source support
    p_out = rebin_pdf_to_mass(centers, pdf, target_edges)
    assert p_out.shape == (6,)
    assert abs(p_out.sum() - 1.0) < 1e-10


def test_rebin_pdf_to_mass_no_overlap_returns_zeros():
    """When target range is completely outside source, output is all zeros."""
    centers = np.linspace(0.0, 2.0, 20)
    pdf = np.ones(20) / 20.0
    target_edges = np.linspace(5.0, 10.0, 6)  # no overlap with [0, 2]
    p_out = rebin_pdf_to_mass(centers, pdf, target_edges)
    assert p_out.sum() == 0.0


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def test_kl_div_zero_when_equal():
    """KL(p || p) should be zero."""
    p = np.array([0.1, 0.4, 0.3, 0.2])
    assert kl_div(p, p) < 1e-10


def test_kl_div_positive():
    """KL divergence should be strictly positive when p != q."""
    p = np.array([0.1, 0.4, 0.3, 0.2])
    q = np.array([0.3, 0.2, 0.3, 0.2])
    assert kl_div(p, q) > 0.0


def test_js_div_zero_when_equal():
    """JS(p, p) should be zero."""
    p = np.array([0.1, 0.4, 0.3, 0.2])
    assert js_div(p, p) < 1e-10


def test_js_div_symmetric():
    """JS divergence must be symmetric."""
    p = np.array([0.1, 0.4, 0.3, 0.2])
    q = np.array([0.3, 0.2, 0.3, 0.2])
    assert abs(js_div(p, q) - js_div(q, p)) < 1e-10


def test_js_div_bounded():
    """JS divergence should be in [0, log(2)] for any p, q."""
    rng = np.random.default_rng(0)
    for _ in range(10):
        p = rng.dirichlet(np.ones(5))
        q = rng.dirichlet(np.ones(5))
        val = js_div(p, q)
        assert 0.0 <= val <= np.log(2) + 1e-10


# ---------------------------------------------------------------------------
# Model probability
# ---------------------------------------------------------------------------

def test_model_contact_mass_sums_to_one():
    """P_model(m|T) must sum to 1 across all T and all b(T) signs."""
    m_centers = np.arange(0, 12, dtype=float)
    p0_mass = np.ones(12) / 12.0
    b_fn = make_b_fn("hs", Tref=300.0, Tscale=100.0)
    params = np.array([500.0, 2.0])
    for T in [200.0, 250.0, 300.0, 350.0, 400.0]:
        p = model_contact_mass(p0_mass, m_centers, T, params, b_fn)
        assert abs(p.sum() - 1.0) < 1e-10, f"Not normalized at T={T}"


def test_model_contact_mass_extreme_b():
    """model_contact_mass should remain stable for extreme b values."""
    m_centers = np.arange(0, 8, dtype=float)
    p0_mass = np.ones(8) / 8.0
    b_fn = make_b_fn("hs", Tref=300.0, Tscale=100.0)
    for T in [1.0, 1e6]:   # extreme temperatures → very large |b|
        p = model_contact_mass(p0_mass, m_centers, T, np.array([5000.0, 0.0]), b_fn)
        assert np.all(np.isfinite(p))
        assert abs(p.sum() - 1.0) < 1e-10


# ---------------------------------------------------------------------------
# Synthetic parameter recovery
# ---------------------------------------------------------------------------

def _synthetic_contact_data(model_name, true_params, temps, m_centers=None, p0_mass=None):
    """Build exact contact distributions from a known model (no noise)."""
    if m_centers is None:
        m_centers = np.arange(0, 14, dtype=float)
    if p0_mass is None:
        p0_mass = np.exp(-0.12 * m_centers)
        p0_mass /= p0_mass.sum()
    Tref = 0.5 * (temps.min() + temps.max())
    Tscale = max(temps.max() - temps.min(), 1.0)
    b_fn = make_b_fn(model_name, Tref, Tscale)
    p_obs = np.array([
        model_contact_mass(p0_mass, m_centers, float(T), true_params, b_fn)
        for T in temps
    ])
    return m_centers, p0_mass, b_fn, p_obs


def test_hs_parameter_recovery():
    """Fitting noiseless hs data must recover Tc = h/s to within 2%."""
    h_true, s_true = 500.0, 2.0
    Tc_true = h_true / s_true  # 250 K
    true_params = np.array([h_true, s_true])
    temps = np.linspace(200.0, 310.0, 12)

    m_centers, p0_mass, b_fn, p_obs = _synthetic_contact_data("hs", true_params, temps)
    spec = MODEL_REGISTRY["hs"]

    best, best_val = None, float("inf")
    for x0 in [np.array([h_true, s_true]),
               np.array([350.0, 1.4]),
               np.array([700.0, 2.8])]:
        res = minimize(
            objective, x0,
            args=(temps, m_centers, p_obs, p0_mass, b_fn, kl_div),
            method="L-BFGS-B", bounds=spec["bounds"],
            options={"maxiter": 600},
        )
        if res.fun < best_val:
            best_val, best = res.fun, res

    Tc_fit = best.x[0] / best.x[1]
    assert abs(Tc_fit - Tc_true) / Tc_true < 0.02, (
        f"hs recovery: Tc_fit={Tc_fit:.3f}, Tc_true={Tc_true:.3f}"
    )


def test_tc_scale_parameter_recovery():
    """Fitting noiseless tc_scale data must recover Tc to within 2%."""
    A_true, Tc_true = 1.5, 275.0
    true_params = np.array([A_true, Tc_true])
    temps = np.linspace(230.0, 340.0, 12)

    m_centers, p0_mass, b_fn, p_obs = _synthetic_contact_data("tc_scale", true_params, temps)
    spec = MODEL_REGISTRY["tc_scale"]

    best, best_val = None, float("inf")
    for x0 in [np.array([A_true, Tc_true]),
               np.array([0.8, 250.0]),
               np.array([2.5, 310.0])]:
        res = minimize(
            objective, x0,
            args=(temps, m_centers, p_obs, p0_mass, b_fn, kl_div),
            method="L-BFGS-B", bounds=spec["bounds"],
            options={"maxiter": 600},
        )
        if res.fun < best_val:
            best_val, best = res.fun, res

    Tc_fit = best.x[1]
    assert abs(Tc_fit - Tc_true) / Tc_true < 0.02, (
        f"tc_scale recovery: Tc_fit={Tc_fit:.3f}, Tc_true={Tc_true:.3f}"
    )


# ---------------------------------------------------------------------------
# Rg prediction
# ---------------------------------------------------------------------------

def test_rg_prediction_normalized():
    """predict_rg_from_joint must return rows that sum to 1 for any T."""
    rng = np.random.default_rng(42)
    n_m, n_rg = 5, 6
    crg_prob = rng.random((n_m, n_rg))
    crg_prob /= crg_prob.sum()

    c_edges = np.arange(-0.5, n_m + 0.5, 1.0)
    rg_edges = np.linspace(1.0, 7.0, n_rg + 1)
    temps = np.array([240.0, 270.0, 300.0, 330.0])
    b_fn = make_b_fn("hs", Tref=285.0, Tscale=90.0)
    params = np.array([500.0, 2.0])

    rg_centers, rg_mod = predict_rg_from_joint(
        crg_prob, c_edges, rg_edges, temps, params, b_fn
    )
    assert rg_mod.shape == (len(temps), n_rg)
    assert rg_centers.shape == (n_rg,)
    for i in range(len(temps)):
        assert abs(rg_mod[i].sum() - 1.0) < 1e-10, f"Row {i} not normalized"


def test_rg_prediction_changes_with_T():
    """Rg predictions must differ across temperatures (b(T) must shift the marginal)."""
    rng = np.random.default_rng(7)
    crg_prob = rng.random((6, 8))
    crg_prob /= crg_prob.sum()
    c_edges = np.arange(-0.5, 6.5, 1.0)
    rg_edges = np.linspace(1.0, 5.0, 9)
    temps = np.array([250.0, 350.0])
    b_fn = make_b_fn("hs", Tref=300.0, Tscale=100.0)
    params = np.array([800.0, 2.5])  # large h → strong T-dependence

    _, rg_mod = predict_rg_from_joint(crg_prob, c_edges, rg_edges, temps, params, b_fn)
    # The two rows should not be identical
    assert not np.allclose(rg_mod[0], rg_mod[1])


# ---------------------------------------------------------------------------
# New models: poly3 and heat_capacity
# ---------------------------------------------------------------------------

def test_poly3_in_registry():
    """poly3 must be present in the model registry with 4 parameters."""
    assert "poly3" in MODEL_REGISTRY
    spec = MODEL_REGISTRY["poly3"]
    assert len(spec["param_names"]) == 4
    assert len(spec["x0"]) == 4
    assert len(spec["bounds"]) == 4


def test_heat_capacity_in_registry():
    """heat_capacity must be present with params dh0, ds0, dCp."""
    assert "heat_capacity" in MODEL_REGISTRY
    spec = MODEL_REGISTRY["heat_capacity"]
    assert spec["param_names"] == ["dh0", "ds0", "dCp"]


def test_poly3_sums_to_one():
    """model_contact_mass with poly3 must produce normalized distributions."""
    m_centers = np.arange(0, 10, dtype=float)
    p0_mass = np.ones(10) / 10.0
    b_fn = make_b_fn("poly3", Tref=300.0, Tscale=80.0)
    params = np.array([0.5, -1.0, 0.2, 0.05])
    for T in [260.0, 300.0, 340.0]:
        p = model_contact_mass(p0_mass, m_centers, T, params, b_fn)
        assert abs(p.sum() - 1.0) < 1e-10


def test_heat_capacity_sums_to_one():
    """model_contact_mass with heat_capacity must produce normalized distributions."""
    m_centers = np.arange(0, 10, dtype=float)
    p0_mass = np.ones(10) / 10.0
    b_fn = make_b_fn("heat_capacity", Tref=300.0, Tscale=80.0)
    params = np.array([600.0, 2.0, 5.0])   # dh0, ds0, dCp
    for T in [260.0, 300.0, 340.0]:
        p = model_contact_mass(p0_mass, m_centers, T, params, b_fn)
        assert np.all(np.isfinite(p))
        assert abs(p.sum() - 1.0) < 1e-10


def test_heat_capacity_dcp_zero_matches_hs():
    """heat_capacity with dCp=0 must match hs model exactly."""
    m_centers = np.arange(0, 10, dtype=float)
    p0_mass = np.exp(-0.15 * m_centers)
    p0_mass /= p0_mass.sum()
    T0 = 300.0
    h, s = 600.0, 2.2
    # hs: b = h/T - s
    b_fn_hs = make_b_fn("hs", Tref=T0, Tscale=100.0)
    # heat_capacity with dh0=h, ds0=s, dCp=0: b = (h - T*s)/T = h/T - s  ✓
    b_fn_hc = make_b_fn("heat_capacity", Tref=T0, Tscale=100.0)
    params_hs = np.array([h, s])
    params_hc = np.array([h, s, 0.0])
    for T in [250.0, 300.0, 350.0]:
        p_hs = model_contact_mass(p0_mass, m_centers, T, params_hs, b_fn_hs)
        p_hc = model_contact_mass(p0_mass, m_centers, T, params_hc, b_fn_hc)
        np.testing.assert_allclose(p_hs, p_hc, atol=1e-12,
                                   err_msg=f"hs vs heat_capacity mismatch at T={T}")
