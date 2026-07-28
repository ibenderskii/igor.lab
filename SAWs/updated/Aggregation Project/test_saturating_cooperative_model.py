#!/usr/bin/env python3
"""pytest suite for the saturating_cooperative contact model.

The claim under test is narrow and specific.  With the contact fraction
q = m/N measured from m_ref = 0,

    b(T)     = h_b/T - s_b
    u(m,T;N) = N * [ b(T)*q - A0*q^2 / (1 + (q/q_sat)^2) ]

must (a) equal that formula exactly, (b) reduce to the hs potential BIT FOR BIT
at A0 = 0, (c) behave as -N*A0*q^2 at low q, (d) saturate at high q so that the
marginal slope du/dm returns to b(T), (e) be extensive at fixed q, (f) refuse
A0 < 0 and q_sat <= 0, (g) be recoverable from data it generated, and (h) be
declared identically by every fitter copy.  Requirement (i) is that none of the
pre-existing models moved a single bit while this one was added.

Run:  python -m pytest "test_saturating_cooperative_model.py" -q
"""
import hashlib
import importlib.util
import json
import os
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import fit_lattice_contact_model_2 as fit
import FIT_TO_DAT as dat

HERE = os.path.dirname(os.path.abspath(__file__))
UPDATED = os.path.dirname(HERE)
FITTER = os.path.join(HERE, "fit_lattice_contact_model_2.py")

MODEL = "saturating_cooperative"
LEGACY_MODELS = ("hs", "tc_scale", "hs_quadratic", "poly2", "poly3", "heat_capacity")
QUADRATIC_MODELS = ("hs_m2_const", "hs_m2_hs")
TREF, TSCALE = 320.0, 80.0
TEMPS = np.linspace(280.0, 360.0, 9)
N_BEADS = 30

# The three copies of the fitter that must declare an identical model contract.
FITTER_COPIES = {
    "isaw": os.path.join(UPDATED, "ISAW Project", "fit_lattice_contact_model_2.py"),
    "aggregation": os.path.join(
        UPDATED, "Aggregation Project", "fit_lattice_contact_model_2.py"
    ),
    "auto": os.path.join(UPDATED, "auto", "fit_lattice_contact_model_2.py"),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _u(params, T, m, *, n_beads=N_BEADS, model=MODEL, mod=fit):
    """The model's own potential, through the public builder API."""
    return mod.make_contact_u_fn(model, TREF, TSCALE, n_beads=n_beads)(
        np.asarray(params, dtype=float), float(T), np.asarray(m, dtype=float)
    )


def _u_reference(params, T, m, n_beads):
    """The definition, transcribed straight from the docstring above.

    Deliberately written as N*(b*q - ...) rather than as b*m + ..., so it is an
    independent statement of the formula and not a copy of the implementation.
    """
    h_b, s_b, A0, q_sat = (float(v) for v in params)
    b = h_b / float(T) - s_b
    q = np.asarray(m, dtype=float) / float(n_beads)
    return float(n_beads) * (b * q - A0 * q ** 2 / (1.0 + (q / q_sat) ** 2))


def _p0(m_centers):
    p = np.exp(-0.5 * ((m_centers - 8.0) / 4.0) ** 2)
    return p / p.sum()


def _joint(n_m=20, n_rg=24):
    """Joint baseline P0(m, Rg) whose Rg shrinks as contacts grow."""
    c_edges = np.arange(-0.5, n_m + 0.5, 1.0)
    m_centers = 0.5 * (c_edges[:-1] + c_edges[1:])
    rg_edges = np.linspace(1.0, 5.0, n_rg + 1)
    rg_centers = 0.5 * (rg_edges[:-1] + rg_edges[1:])
    p_m = _p0(m_centers)
    crg = np.zeros((n_m, n_rg))
    for i, m in enumerate(m_centers):
        mu = 4.0 - 0.15 * m
        row = np.exp(-0.5 * ((rg_centers - mu) / 0.4) ** 2)
        crg[i] = p_m[i] * row / row.sum()
    return crg / crg.sum(), c_edges, rg_edges


def _write_baseline(path, *, n_beads=N_BEADS):
    crg, c_edges, rg_edges = _joint()
    np.savez_compressed(
        path,
        c_vals=np.arange(crg.shape[0], dtype=np.int64),
        c_prob=crg.sum(axis=1),
        c_edges=c_edges,
        rg_edges=rg_edges,
        rg_prob=crg.sum(axis=0),
        crg_prob=crg,
        N=np.int64(n_beads),
    )
    return crg, c_edges, rg_edges


def _write_remd(path, crg, c_edges, params, model, n_beads=N_BEADS):
    m_centers = 0.5 * (c_edges[:-1] + c_edges[1:])
    p0 = crg.sum(axis=1)
    p0 = p0 / p0.sum()
    u_fn = fit.make_contact_u_fn(model, TREF, TSCALE, n_beads=n_beads)
    hists = np.array([
        fit.model_contact_mass(p0, m_centers, float(T), params, u_fn) for T in TEMPS
    ])
    np.savez_compressed(path, temps=TEMPS, ct_centers=m_centers, ct_hists=hists)
    return m_centers, p0, hists


# ---------------------------------------------------------------------------
# 1. The exact formula
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mod", [fit, dat], ids=["fitter", "fit_to_dat"])
@pytest.mark.parametrize("n_beads", [12, 30, 64])
@pytest.mark.parametrize(
    "A0,q_sat", [(0.5, 0.10), (3.0, 0.35), (12.0, 1.5), (20.0, 0.02)]
)
def test_potential_matches_the_definition_exactly(mod, n_beads, A0, q_sat):
    params = [750.0, 2.8, A0, q_sat]
    m = np.arange(0.0, float(n_beads) + 1.0)
    for T in (280.0, 320.0, 360.0):
        got = _u(params, T, m, n_beads=n_beads, mod=mod)
        want = _u_reference(params, T, m, n_beads)
        np.testing.assert_allclose(got, want, rtol=1e-13, atol=1e-12)


def test_scalar_and_array_inputs_agree():
    params = np.array([750.0, 2.8, 3.0, 0.35])
    m = np.arange(0.0, 21.0)
    arr = _u(params, 315.0, m)
    each = np.array([float(_u(params, 315.0, mi)) for mi in m])
    np.testing.assert_allclose(arr, each, rtol=0, atol=0)


def test_reduced_contact_potential_agrees_with_make_contact_u_fn():
    params = np.array([750.0, 2.8, 4.0, 0.25])
    m = np.arange(0.0, 31.0)
    direct = fit.reduced_contact_potential(
        m, 300.0, MODEL, params, TREF, TSCALE, n_beads=N_BEADS
    )
    built = _u(params, 300.0, m)
    assert np.array_equal(direct, built)


# ---------------------------------------------------------------------------
# 2. A0 = 0 nests hs exactly
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mod", [fit, dat], ids=["fitter", "fit_to_dat"])
@pytest.mark.parametrize("q_sat", [0.02, 0.35, 2.0])
def test_zero_amplitude_nests_hs_bit_for_bit(mod, q_sat):
    """A0 = 0 must give the SAME floating-point value as hs, not merely a close one.

    array_equal, not allclose: the point of evaluating the linear part as
    b(T)*m rather than N*(b(T)*(m/N)) is that no rounding is introduced, so the
    saturating model nests the linear one exactly.
    """
    m = np.arange(0.0, 31.0)
    hs_p = np.array([812.5, 2.6125])
    sat_p = np.array([812.5, 2.6125, 0.0, q_sat])
    u_hs = mod.make_contact_u_fn("hs", TREF, TSCALE)
    u_sat = mod.make_contact_u_fn(MODEL, TREF, TSCALE, n_beads=N_BEADS)
    for T in (280.0, 297.3, 320.0, 341.7, 360.0):
        assert np.array_equal(u_sat(sat_p, T, m), u_hs(hs_p, T, m))
        b = mod.reduced_bias(MODEL, sat_p, T, TREF, TSCALE)
        assert np.array_equal(u_sat(sat_p, T, m), b * m)


def test_default_first_restart_starts_at_the_nested_hs_solution():
    x0 = fit.MODEL_REGISTRY[MODEL]["x0"]
    assert x0[2] == 0.0, "A0 must default to zero so restart 0 is the hs solution"
    assert x0 == [750.0, 2.8, 0.0, 0.35]
    assert np.array_equal(
        _u(x0, 320.0, np.arange(0.0, 31.0)),
        fit.make_contact_u_fn("hs", TREF, TSCALE)(
            np.array(x0[:2]), 320.0, np.arange(0.0, 31.0)
        ),
    )


def test_zero_amplitude_leaves_the_predicted_distribution_identical():
    """Nesting has to survive the reweighting, not just the potential."""
    m = np.arange(0.0, 20.0)
    p0 = _p0(m)
    hs_p, sat_p = np.array([780.0, 2.7]), np.array([780.0, 2.7, 0.0, 0.35])
    u_hs = fit.make_contact_u_fn("hs", TREF, TSCALE)
    u_sat = fit.make_contact_u_fn(MODEL, TREF, TSCALE, n_beads=N_BEADS)
    for T in (285.0, 320.0, 355.0):
        assert np.array_equal(
            fit.model_contact_mass(p0, m, T, sat_p, u_sat),
            fit.model_contact_mass(p0, m, T, hs_p, u_hs),
        )


# ---------------------------------------------------------------------------
# 3. Low-q expansion
# ---------------------------------------------------------------------------

def test_low_q_expansion_is_quadratic_with_quartic_residual():
    """For q << q_sat the cooperative term is -N*A0*q^2 + O(q^4).

    1/(1+x^2) = 1 - x^2 + O(x^4), so the leading correction to -N*A0*q^2 is
    +N*A0*q^4/q_sat^2.  Halving m must therefore cut the residual by ~16.
    """
    N, A0, q_sat, T = 200, 4.0, 0.50, 310.0
    params = np.array([750.0, 2.8, A0, q_sat])
    b = fit.reduced_bias(MODEL, params, T, TREF, TSCALE)

    def residual(m):
        q = m / N
        pure_quadratic = b * m - N * A0 * q ** 2
        return abs(float(_u(params, T, m, n_beads=N)) - pure_quadratic)

    m0 = 0.02 * q_sat * N          # q/q_sat = 0.02: deep in the low-q regime
    r0, r1, r2 = residual(m0), residual(m0 / 2.0), residual(m0 / 4.0)
    assert r0 > 0.0
    # Each halving of m cuts the quartic residual by 2^4 = 16.
    assert r0 / r1 == pytest.approx(16.0, rel=0.02)
    assert r1 / r2 == pytest.approx(16.0, rel=0.02)

    # And the quartic prediction itself is right, not just its scaling.
    q0 = m0 / N
    assert residual(m0) == pytest.approx(N * A0 * q0 ** 4 / q_sat ** 2, rel=1e-3)


# ---------------------------------------------------------------------------
# 4. High-q saturation and the marginal slope
# ---------------------------------------------------------------------------

def test_cooperative_term_saturates_at_a_finite_plateau():
    """As q/q_sat -> inf the cooperative term tends to the constant -N*A0*q_sat^2."""
    N, A0, q_sat, T = 40, 2.5, 0.20, 305.0
    params = np.array([750.0, 2.8, A0, q_sat])
    b = fit.reduced_bias(MODEL, params, T, TREF, TSCALE)
    plateau = N * A0 * q_sat ** 2

    ratios = np.array([1.0, 10.0, 100.0, 1000.0, 10000.0])
    m = ratios * q_sat * N
    gap = b * m - _u(params, T, m, n_beads=N)     # the cooperative depth

    assert np.all(np.diff(gap) > 0.0), "the gap must approach the plateau from below"
    assert np.all(gap < plateau), "the gap must never exceed its own limit"
    assert float(gap[-1]) == pytest.approx(plateau, rel=1e-7)


def test_marginal_slope_returns_to_b_at_high_q():
    """du/dm -> b(T) once q >> q_sat: the attraction stops adding anything.

    Measured as the secant over [m, 2m] rather than a central difference with a
    fixed step.  Both are the same limit, but the secant differences two u
    values of order b*m, so there is no cancellation and the test stays honest
    out to q/q_sat = 1e4, where a fixed-step central difference would be
    measuring nothing but floating-point noise.
    """
    N, A0, q_sat, T = 40, 2.5, 0.20, 305.0
    params = np.array([750.0, 2.8, A0, q_sat])
    b = fit.reduced_bias(MODEL, params, T, TREF, TSCALE)

    def secant(m):
        return float(
            _u(params, T, 2.0 * m, n_beads=N) - _u(params, T, m, n_beads=N)
        ) / m

    ratios = (10.0, 100.0, 1000.0, 10000.0)
    errs = [abs(secant(r * q_sat * N) - b) for r in ratios]
    assert all(e2 < e1 for e1, e2 in zip(errs, errs[1:])), errs
    # The deviation falls off as 1/r^3, so each decade in r buys three of error.
    for e1, e2 in zip(errs, errs[1:]):
        assert e1 / e2 == pytest.approx(1000.0, rel=0.05)
    # In reduced units, where |b| is of order 0.34, the residual slope error at
    # q/q_sat = 1e4 is twelve orders of magnitude down.
    assert errs[-1] < 1e-12

    # At low q the slope is genuinely NOT b(T) -- otherwise the test above would
    # be satisfied by a model with no cooperative term at all.
    assert abs(secant(0.5 * q_sat * N) - b) > 0.5 * abs(b)


def test_pointwise_marginal_slope_matches_the_analytic_derivative():
    """du/dm = b(T) - 2*A0*q/(1 + (q/q_sat)^2)^2, checked where it is well conditioned."""
    N, A0, q_sat, T = 40, 2.5, 0.20, 305.0
    params = np.array([750.0, 2.8, A0, q_sat])
    b = fit.reduced_bias(MODEL, params, T, TREF, TSCALE)
    for r in (0.25, 1.0, 4.0):
        m = r * q_sat * N
        h = 1e-6 * m
        got = float(
            _u(params, T, m + h, n_beads=N) - _u(params, T, m - h, n_beads=N)
        ) / (2.0 * h)
        q = m / N
        want = b - 2.0 * A0 * q / (1.0 + (q / q_sat) ** 2) ** 2
        assert got == pytest.approx(want, rel=1e-6)


# ---------------------------------------------------------------------------
# 5. Extensivity at fixed contact fraction
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("q", [0.05, 0.25, 0.5, 0.9])
def test_potential_per_bead_is_chain_length_independent_at_fixed_q(q):
    """u/N depends on (q, T) only: the whole point of writing the model in q."""
    params = np.array([750.0, 2.8, 3.0, 0.35])
    T = 312.0
    per_bead = [
        float(_u(params, T, q * N, n_beads=N)) / N for N in (20, 40, 80, 160)
    ]
    for value in per_bead[1:]:
        assert value == pytest.approx(per_bead[0], rel=1e-12)


def test_the_same_m_at_different_N_does_NOT_collapse():
    """Guards the test above: N enters through q, so fixing m instead must differ."""
    params = np.array([750.0, 2.8, 3.0, 0.35])
    values = [float(_u(params, 312.0, 8.0, n_beads=N)) for N in (20, 40, 80)]
    assert len(set(values)) == 3, values


# ---------------------------------------------------------------------------
# 6. Invalid parameters
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mod", [fit, dat], ids=["fitter", "fit_to_dat"])
@pytest.mark.parametrize(
    "params,match",
    [
        ([750.0, 2.8, -1e-12, 0.35], "A0 >= 0"),
        ([750.0, 2.8, -5.0, 0.35], "A0 >= 0"),
        ([750.0, 2.8, 3.0, 0.0], "q_sat > 0"),
        ([750.0, 2.8, 3.0, -0.35], "q_sat > 0"),
        ([750.0, 2.8, np.nan, 0.35], "finite"),
        ([750.0, 2.8, 3.0, np.inf], "finite"),
    ],
)
def test_invalid_amplitude_or_saturation_scale_is_refused(mod, params, match):
    u_fn = mod.make_contact_u_fn(MODEL, TREF, TSCALE, n_beads=N_BEADS)
    with pytest.raises(ValueError, match=match):
        u_fn(np.array(params), 320.0, np.arange(0.0, 10.0))
    with pytest.raises(ValueError, match=match):
        mod.reduced_contact_potential(
            5.0, 320.0, MODEL, np.array(params), TREF, TSCALE, n_beads=N_BEADS
        )


@pytest.mark.parametrize("mod", [fit, dat], ids=["fitter", "fit_to_dat"])
def test_registry_bounds_enforce_the_valid_region(mod):
    """The optimizer must never be able to reach an invalid (A0, q_sat)."""
    bounds = mod.MODEL_REGISTRY[MODEL]["bounds"]
    assert bounds == [
        (-2000.0, 2000.0), (-10.0, 10.0), (0.0, 20.0), (0.02, 2.0),
    ]
    (a_lo, a_hi), (q_lo, q_hi) = bounds[2], bounds[3]
    assert a_lo >= 0.0 and a_hi > a_lo          # A0 >= 0 everywhere in range
    assert q_lo > 0.0 and q_hi > q_lo           # q_sat > 0 everywhere in range
    # Every corner of the box is an evaluable potential.
    for A0 in (a_lo, a_hi):
        for q_sat in (q_lo, q_hi):
            u = _u([750.0, 2.8, A0, q_sat], 320.0, np.arange(0.0, 31.0), mod=mod)
            assert np.all(np.isfinite(u))


def test_the_model_requires_a_chain_length():
    assert fit.MODEL_REGISTRY[MODEL]["requires_chain_length"] is True
    with pytest.raises(ValueError, match="chain length"):
        fit.make_contact_u_fn(MODEL, TREF, TSCALE, n_beads=None)
    with pytest.raises(ValueError, match="chain length"):
        fit.resolve_chain_length(None, None, model_name=MODEL)
    # and the message names the q normalization, not m^2/(2N)
    with pytest.raises(ValueError, match=r"q = m/N"):
        fit.validate_chain_length(MODEL, None)


# ---------------------------------------------------------------------------
# 7. Synthetic-data recovery
# ---------------------------------------------------------------------------

def test_recovers_known_parameters_from_data_it_generated():
    m_centers = np.arange(0.0, 26.0)
    p0_mass = _p0(m_centers)
    true = np.array([820.0, 2.65, 6.0, 0.30])
    u_true = fit.make_contact_u_fn(MODEL, TREF, TSCALE, n_beads=N_BEADS)
    p_obs = np.array([
        fit.model_contact_mass(p0_mass, m_centers, float(T), true, u_true)
        for T in TEMPS
    ])
    loss_fn = fit._get_loss_fn("js")
    obj, obj_args = fit.build_objective(
        False, TEMPS, m_centers, p_obs, p0_mass, u_true, loss_fn
    )
    rng = np.random.default_rng(7)
    bounds = fit.MODEL_REGISTRY[MODEL]["bounds"]
    x0s = [np.array(fit.MODEL_REGISTRY[MODEL]["x0"], dtype=float)]
    x0s += [
        np.array([rng.uniform(lo, hi) for lo, hi in bounds]) for _ in range(9)
    ]
    best, _ = fit.fit_one_split(obj, obj_args, x0s, bounds)

    assert best.x[0] == pytest.approx(true[0], rel=0.05)     # h_b
    assert best.x[1] == pytest.approx(true[1], rel=0.05)     # s_b
    assert best.x[2] == pytest.approx(true[2], rel=0.10)     # A0
    assert best.x[3] == pytest.approx(true[3], rel=0.10)     # q_sat
    # and the recovered potential reproduces the generating one over the support
    np.testing.assert_allclose(
        _u(best.x, 320.0, m_centers), _u(true, 320.0, m_centers),
        rtol=0, atol=1e-2,
    )


def test_a_saturating_target_is_not_reproducible_by_the_linear_model():
    """Otherwise 'recovery' would prove nothing: hs must fit this data worse."""
    m_centers = np.arange(0.0, 26.0)
    p0_mass = _p0(m_centers)
    true = np.array([820.0, 2.65, 6.0, 0.30])
    u_true = fit.make_contact_u_fn(MODEL, TREF, TSCALE, n_beads=N_BEADS)
    p_obs = np.array([
        fit.model_contact_mass(p0_mass, m_centers, float(T), true, u_true)
        for T in TEMPS
    ])
    loss_fn = fit._get_loss_fn("js")

    def best_loss(model, n_beads):
        u_fn = fit.make_contact_u_fn(model, TREF, TSCALE, n_beads=n_beads)
        obj, obj_args = fit.build_objective(
            False, TEMPS, m_centers, p_obs, p0_mass, u_fn, loss_fn
        )
        rng = np.random.default_rng(11)
        bounds = fit.MODEL_REGISTRY[model]["bounds"]
        x0s = [np.array(fit.MODEL_REGISTRY[model]["x0"], dtype=float)]
        x0s += [
            np.array([rng.uniform(lo, hi) for lo, hi in bounds]) for _ in range(9)
        ]
        return float(fit.fit_one_split(obj, obj_args, x0s, bounds)[0].fun)

    sat_loss = best_loss(MODEL, N_BEADS)
    hs_loss = best_loss("hs", None)
    assert sat_loss < hs_loss
    assert sat_loss < 1e-8


# ---------------------------------------------------------------------------
# 8. Model contracts are identical across every fitter copy
# ---------------------------------------------------------------------------

def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_the_three_fitter_copies_are_byte_identical():
    digests = {}
    for name, path in FITTER_COPIES.items():
        assert os.path.exists(path), path
        with open(path, "rb") as fh:
            digests[name] = hashlib.md5(fh.read()).hexdigest()
    assert len(set(digests.values())) == 1, digests


def test_model_contract_is_identical_across_every_fitter_copy():
    contracts = {
        name: _load_module(f"_satcoop_fitter_{name}", path).get_model_contract()
        for name, path in FITTER_COPIES.items()
    }
    contracts["fit_to_dat"] = dat.get_model_contract()

    reference = contracts["isaw"]
    for name, contract in contracts.items():
        assert contract == reference, name

    assert reference["model_api_version"] == 3
    assert MODEL in reference["models"]


def test_the_contract_records_the_potential_definition_m_ref_and_q_normalization():
    entry = fit.get_model_contract()["models"][MODEL]
    assert entry["param_names"] == ["h_b", "s_b", "A0", "q_sat"]
    assert entry["potential_kind"] == "saturating_cooperative"
    assert entry["m_ref"] == 0
    assert entry["potential_normalization"] == "q = m/N"
    # The saturating model is NOT forced into the m^2/(2N) convention.
    assert entry["quadratic_normalization"] is None

    definition = entry["potential_definition"]
    for fragment in (
        "N*[b(T)*q - A0*q^2/(1 + (q/q_sat)^2)]",
        "b(T) = h_b/T - s_b",
        "q = m/N",
        "m_ref = 0",
        "A0 >= 0",
        "q_sat > 0",
    ):
        assert fragment in definition, (fragment, definition)


def test_every_model_carries_the_new_contract_keys():
    contract = fit.get_model_contract()["models"]
    for name, entry in contract.items():
        assert entry["m_ref"] == 0, name
        assert isinstance(entry["potential_definition"], str) and entry[
            "potential_definition"
        ], name
    for name in LEGACY_MODELS:
        assert contract[name]["potential_normalization"] is None, name
    for name in QUADRATIC_MODELS:
        assert contract[name]["potential_normalization"] == "m^2/(2N)", name
        assert contract[name]["quadratic_normalization"] == "m^2/(2N)", name


def test_generic_potential_api_covers_all_three_kinds():
    """The builder table is the extension point; no kind is special-cased away."""
    for mod in (fit, dat):
        assert set(mod.POTENTIAL_BUILDERS) == {
            "linear", "contact_quadratic", "saturating_cooperative",
        }
        kinds = {
            str(spec["potential_kind"]) for spec in mod.MODEL_REGISTRY.values()
        }
        assert kinds <= set(mod.POTENTIAL_BUILDERS), kinds


# ---------------------------------------------------------------------------
# 9. Pre-existing models are untouched
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model", LEGACY_MODELS)
def test_legacy_models_still_return_exactly_b_times_m(model):
    m = np.arange(0.0, 31.0)
    params = np.array(fit.MODEL_REGISTRY[model]["x0"], dtype=float)
    for T in (285.0, 320.0, 355.0):
        u = fit.reduced_contact_potential(
            m, T, model, params, TREF, TSCALE, n_beads=N_BEADS
        )
        b = fit.reduced_bias(model, params, T, TREF, TSCALE)
        assert np.array_equal(u, b * m)
        assert fit.quadratic_bias(model, params, T, TREF, TSCALE) == 0.0
        assert fit.MODEL_REGISTRY[model]["potential_kind"] == "linear"


@pytest.mark.parametrize("model", QUADRATIC_MODELS)
def test_contact_quadratic_models_still_use_m_squared_over_2N(model):
    m = np.arange(0.0, 31.0)
    params = np.array(fit.MODEL_REGISTRY[model]["x0"], dtype=float)
    params[2] = 0.7                                   # a nonzero curvature
    u_fn = fit.make_contact_u_fn(model, TREF, TSCALE, n_beads=N_BEADS)
    for T in (285.0, 320.0, 355.0):
        b = fit.reduced_bias(model, params, T, TREF, TSCALE)
        kappa = fit.quadratic_bias(model, params, T, TREF, TSCALE)
        want = b * m + kappa * (m * m) / (2.0 * N_BEADS)
        assert np.array_equal(u_fn(params, T, m), want)


def test_saturating_model_reports_no_quadratic_coefficient():
    """kappa(T) is the coefficient of m^2/(2N); this model has no such term."""
    params = np.array([750.0, 2.8, 6.0, 0.3])
    for T in (285.0, 320.0, 355.0):
        assert fit.quadratic_bias(MODEL, params, T, TREF, TSCALE) == 0.0
    assert fit.MODEL_REGISTRY[MODEL]["raw_q_fn"] is fit._q_zero


# ---------------------------------------------------------------------------
# 10. End-to-end: the fit outputs record the contract
# ---------------------------------------------------------------------------

def test_fit_summary_and_npz_record_the_saturating_contract(tmp_path):
    crg, c_edges, _ = _write_baseline(tmp_path / "base.npz")
    _write_remd(
        tmp_path / "remd.npz", crg, c_edges,
        np.array([800.0, 2.7, 5.0, 0.30]), MODEL,
    )
    out = tmp_path / "run"
    subprocess.run(
        [sys.executable, FITTER,
         "--remd", str(tmp_path / "remd.npz"),
         "--baseline", str(tmp_path / "base.npz"),
         "--contact_offset", "0", "--model", MODEL,
         "--loss", "js", "--n_restarts", "4", "--no-plots",
         "--outdir", str(out)],
        check=True, capture_output=True, text=True,
    )

    summary = json.loads((out / "fit_summary.json").read_text())
    assert summary["model_api_version"] == 3
    assert summary["model"] == MODEL
    assert summary["potential_kind"] == "saturating_cooperative"
    assert summary["potential_normalization"] == "q = m/N"
    assert summary["quadratic_normalization"] is None
    assert summary["m_ref"] == 0
    assert summary["fit_chain_length"] == N_BEADS
    assert list(summary["param_names"]) == ["h_b", "s_b", "A0", "q_sat"]
    assert "q = m/N" in summary["potential_definition"]
    assert "m_ref = 0" in summary["potential_definition"]
    # A0 and q_sat are recorded by name, not only by position in params
    assert summary["A0"] == pytest.approx(summary["params"]["A0"], rel=1e-12)
    assert summary["q_sat"] == pytest.approx(summary["params"]["q_sat"], rel=1e-12)
    assert summary["A0"] >= 0.0 and summary["q_sat"] > 0.0
    # this model has no kappa(T), so the recorded series is identically zero
    assert all(v == 0.0 for v in summary["quadratic_coefficient_by_temperature"])

    z = np.load(out / "fit_results.npz", allow_pickle=True)
    assert str(z["potential_kind"]) == "saturating_cooperative"
    assert str(z["potential_normalization"]) == "q = m/N"
    assert str(z["quadratic_normalization"]) == ""      # npz sentinel for None
    assert "q = m/N" in str(z["potential_definition"])
    assert int(z["m_ref"]) == 0
    assert int(z["fit_chain_length"]) == N_BEADS
    assert float(z["A0"]) == pytest.approx(summary["A0"], rel=1e-12)
    assert float(z["q_sat"]) == pytest.approx(summary["q_sat"], rel=1e-12)


def test_the_saturating_model_gets_a_potential_plot_and_no_kappa_plot(tmp_path):
    crg, c_edges, _ = _write_baseline(tmp_path / "base.npz")
    _write_remd(
        tmp_path / "remd.npz", crg, c_edges,
        np.array([800.0, 2.7, 5.0, 0.30]), MODEL,
    )
    out = tmp_path / "run"
    subprocess.run(
        [sys.executable, FITTER,
         "--remd", str(tmp_path / "remd.npz"),
         "--baseline", str(tmp_path / "base.npz"),
         "--contact_offset", "0", "--model", MODEL,
         "--loss", "js", "--n_restarts", "2",
         "--outdir", str(out)],
        check=True, capture_output=True, text=True,
    )
    assert (out / "contact_potential_vs_m.png").exists()
    assert not (out / "quadratic_coefficient_vs_T.png").exists()


def test_a_missing_chain_length_fails_before_any_fitting(tmp_path):
    crg, c_edges, rg_edges = _joint()
    np.savez_compressed(
        tmp_path / "base.npz",
        c_vals=np.arange(crg.shape[0], dtype=np.int64),
        c_prob=crg.sum(axis=1), c_edges=c_edges,
        rg_edges=rg_edges, rg_prob=crg.sum(axis=0), crg_prob=crg,
    )                                              # no n_beads, no N
    _write_remd(
        tmp_path / "remd.npz", crg, c_edges,
        np.array([800.0, 2.7, 5.0, 0.30]), MODEL,
    )
    out = tmp_path / "run"
    proc = subprocess.run(
        [sys.executable, FITTER,
         "--remd", str(tmp_path / "remd.npz"),
         "--baseline", str(tmp_path / "base.npz"),
         "--contact_offset", "0", "--model", MODEL,
         "--loss", "js", "--n_restarts", "2", "--no-plots",
         "--outdir", str(out)],
        capture_output=True, text=True,
    )
    assert proc.returncode != 0
    combined = proc.stdout + proc.stderr
    assert "chain length" in combined
    assert "q = m/N" in combined
    assert not (out / "fit_summary.json").exists()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
