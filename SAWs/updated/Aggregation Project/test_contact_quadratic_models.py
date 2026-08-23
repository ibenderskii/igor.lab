#!/usr/bin/env python3
"""pytest suite for the contact-number-quadratic models hs_m2_const / hs_m2_hs.

The claim under test is narrow and specific:

    u_contact(m, T; N) = b(T)*m + kappa(T) * m^2/(2N)

must (a) reduce EXACTLY to the historical b(T)*m for every pre-existing model,
(b) nest the hs solution exactly at zero curvature, (c) use precisely the
m^2/(2N) normalization, and (d) be the single potential every reweighting path
uses -- contacts, joint Rg, and the scalar-Rg mode alike.

Run:  python -m pytest "test_contact_quadratic_models.py" -q
"""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import fit_lattice_contact_model_2 as fit
import FIT_TO_DAT as dat
import remd_uniform_chain_2_new as remd

HERE = os.path.dirname(os.path.abspath(__file__))
FITTER = os.path.join(HERE, "fit_lattice_contact_model_2.py")
DAT = os.path.join(HERE, "FIT_TO_DAT.py")

LEGACY_MODELS = ("hs", "tc_scale", "hs_quadratic", "poly2", "poly3", "heat_capacity")
NEW_MODELS = ("hs_m2_const", "hs_m2_hs")
TREF, TSCALE = 320.0, 80.0
TEMPS = np.linspace(280.0, 360.0, 9)
N_BEADS = 30


# ---------------------------------------------------------------------------
# Synthetic inputs
# ---------------------------------------------------------------------------

def _p0(m_centers):
    p = np.exp(-0.5 * ((m_centers - 8.0) / 4.0) ** 2)
    return p / p.sum()


def _joint(n_m=20, n_rg=24, coupling=1.0):
    """Joint baseline P0(m, Rg) whose Rg shrinks as contacts grow."""
    c_edges = np.arange(-0.5, n_m + 0.5, 1.0)
    m_centers = 0.5 * (c_edges[:-1] + c_edges[1:])
    rg_edges = np.linspace(1.0, 5.0, n_rg + 1)
    rg_centers = 0.5 * (rg_edges[:-1] + rg_edges[1:])
    p_m = _p0(m_centers)
    crg = np.zeros((n_m, n_rg))
    for i, m in enumerate(m_centers):
        mu = 4.0 - coupling * 0.15 * m
        row = np.exp(-0.5 * ((rg_centers - mu) / 0.4) ** 2)
        crg[i] = p_m[i] * row / row.sum()
    return crg / crg.sum(), c_edges, rg_edges


def _write_baseline(path, *, n_beads=N_BEADS, kappa_bend=None, chain_key="N"):
    """Joint baseline NPZ; omit the chain-length key to emulate a legacy file."""
    crg, c_edges, rg_edges = _joint()
    kwargs = dict(
        c_vals=np.arange(crg.shape[0], dtype=np.int64),
        c_prob=crg.sum(axis=1),
        c_edges=c_edges,
        rg_edges=rg_edges,
        rg_prob=crg.sum(axis=0),
        crg_prob=crg,
    )
    if chain_key is not None:
        kwargs[chain_key] = np.int64(n_beads)
    if kappa_bend is not None:
        kwargs["kappa_bend"] = np.float64(kappa_bend)
        kwargs["bending_enabled"] = np.bool_(kappa_bend != 0.0)
    np.savez_compressed(path, **kwargs)
    return crg, c_edges, rg_edges


def _write_remd(path, crg, c_edges, params, model, n_beads=N_BEADS):
    """REMD-style contact histograms generated from a known model."""
    m_centers = 0.5 * (c_edges[:-1] + c_edges[1:])
    p0 = crg.sum(axis=1)
    p0 = p0 / p0.sum()
    u_fn = fit.make_contact_u_fn(model, TREF, TSCALE, n_beads=n_beads)
    hists = np.array([
        fit.model_contact_mass(p0, m_centers, float(T), params, u_fn) for T in TEMPS
    ])
    np.savez_compressed(
        path, temps=TEMPS, ct_centers=m_centers, ct_hists=hists,
    )
    return m_centers, p0, hists


# ---------------------------------------------------------------------------
# 1. Legacy model equivalence
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mod", [fit, dat], ids=["fitter", "fit_to_dat"])
@pytest.mark.parametrize("model", LEGACY_MODELS)
def test_legacy_models_have_zero_curvature_and_linear_potential(mod, model):
    """Every pre-existing model keeps u == b(T)*m to the last bit."""
    spec = mod.MODEL_REGISTRY[model]
    assert spec["potential_kind"] == "linear"
    assert spec["quadratic_normalization"] is None
    assert spec["requires_chain_length"] is False

    params = np.array(spec["x0"], dtype=float)
    m = np.arange(0.0, 20.0)
    for T in TEMPS:
        assert mod.quadratic_bias(model, params, float(T), TREF, TSCALE) == 0.0
        b = mod.reduced_bias(model, params, float(T), TREF, TSCALE)
        u = mod.reduced_contact_potential(
            m, float(T), model, params, TREF, TSCALE, n_beads=N_BEADS
        )
        assert np.array_equal(u, b * m)


@pytest.mark.parametrize("model", LEGACY_MODELS)
def test_legacy_reweighting_matches_explicit_exp_minus_b_m(model):
    """model_contact_mass reproduces the historical exp[-b(T)*m] formula."""
    m = np.arange(0.0, 20.0)
    p0 = _p0(m)
    spec = fit.MODEL_REGISTRY[model]
    params = np.array(spec["x0"], dtype=float)
    b_fn = fit.make_b_fn(model, TREF, TSCALE)
    u_fn = fit.make_contact_u_fn(model, TREF, TSCALE)
    for T in TEMPS:
        got = fit.model_contact_mass(p0, m, float(T), params, u_fn)
        w = p0 * np.exp(-b_fn(params, float(T)) * m)
        assert np.allclose(got, w / w.sum(), rtol=1e-13, atol=1e-15)


# ---------------------------------------------------------------------------
# 2. Zero-curvature nesting
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mod", [fit, dat], ids=["fitter", "fit_to_dat"])
@pytest.mark.parametrize(
    "model, params",
    [("hs_m2_const", [700.0, 2.4, 0.0]), ("hs_m2_hs", [700.0, 2.4, 0.0, 0.0])],
)
def test_zero_curvature_nests_hs_exactly(mod, model, params):
    m = np.arange(0.0, 20.0)
    u_new = mod.make_contact_u_fn(model, TREF, TSCALE, n_beads=N_BEADS)
    u_hs = mod.make_contact_u_fn("hs", TREF, TSCALE)
    hs_params = np.array(params[:2], dtype=float)
    for T in TEMPS:
        assert np.array_equal(
            u_new(np.array(params, dtype=float), float(T), m),
            u_hs(hs_params, float(T), m),
        )


def test_default_first_restart_starts_at_the_nested_hs_solution():
    """The zero-curvature start is what makes the nesting reachable by the optimizer."""
    for model in NEW_MODELS:
        x0 = fit.MODEL_REGISTRY[model]["x0"]
        names = fit.MODEL_REGISTRY[model]["param_names"]
        assert names[:2] == ["h1", "s1"]
        assert all(v == 0.0 for v in x0[2:]), (model, x0)
        assert x0[:2] == fit.MODEL_REGISTRY["hs"]["x0"]


# ---------------------------------------------------------------------------
# 3. Exact m^2/(2N) normalization
# ---------------------------------------------------------------------------

def test_quadratic_term_is_exactly_m_squared_over_2N():
    m = np.arange(0.0, 20.0)
    kappa = 0.75
    u = fit.make_contact_u_fn("hs_m2_const", TREF, TSCALE, n_beads=N_BEADS)
    base = np.array([700.0, 2.4, 0.0])
    with_k = np.array([700.0, 2.4, kappa])
    for T in TEMPS:
        delta = u(with_k, float(T), m) - u(base, float(T), m)
        assert np.allclose(delta, kappa * m ** 2 / (2.0 * N_BEADS), rtol=0, atol=1e-13)


def test_hs_m2_hs_quadratic_coefficient_is_h2_over_T_minus_s2():
    h2, s2 = 400.0, 1.1
    params = np.array([700.0, 2.4, h2, s2])
    q_fn = fit.make_q_fn("hs_m2_hs", TREF, TSCALE)
    for T in TEMPS:
        assert q_fn(params, float(T)) == pytest.approx(h2 / T - s2, rel=1e-14)


def test_normalization_scales_inversely_with_chain_length():
    m = np.arange(0.0, 20.0)
    params = np.array([700.0, 2.4, 1.0])
    u30 = fit.make_contact_u_fn("hs_m2_const", TREF, TSCALE, n_beads=30)
    u60 = fit.make_contact_u_fn("hs_m2_const", TREF, TSCALE, n_beads=60)
    lin = fit.make_contact_u_fn("hs", TREF, TSCALE)(np.array([700.0, 2.4]), 320.0, m)
    d30 = u30(params, 320.0, m) - lin
    d60 = u60(params, 320.0, m) - lin
    assert np.allclose(d30, 2.0 * d60, rtol=1e-13, atol=1e-15)


# ---------------------------------------------------------------------------
# 4. Vectorized / scalar equality
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model", LEGACY_MODELS + NEW_MODELS)
def test_vectorized_potential_matches_scalar_elementwise(model):
    spec = fit.MODEL_REGISTRY[model]
    params = np.array(spec["x0"], dtype=float)
    if model == "hs_m2_const":
        params[2] = 0.6
    elif model == "hs_m2_hs":
        params[2], params[3] = 350.0, 0.9
    m = np.arange(0.0, 20.0)
    u_fn = fit.make_contact_u_fn(model, TREF, TSCALE, n_beads=N_BEADS)
    for T in TEMPS:
        vec = u_fn(params, float(T), m)
        scalars = np.array([
            float(u_fn(params, float(T), float(mi))) for mi in m
        ])
        assert np.array_equal(vec, scalars)
        # the module-level accessor must agree with the closure
        assert np.array_equal(
            vec,
            fit.reduced_contact_potential(
                m, float(T), model, params, TREF, TSCALE, n_beads=N_BEADS
            ),
        )


# ---------------------------------------------------------------------------
# 5. Synthetic recovery for both new models
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "model, truth, atol",
    [
        ("hs_m2_const", [780.0, 2.7, 0.8], [8.0, 0.05, 0.03]),
        ("hs_m2_hs", [780.0, 2.7, 300.0, 0.6], [20.0, 0.1, 30.0, 0.1]),
    ],
)
def test_synthetic_recovery(model, truth, atol):
    """Data generated from known parameters is recovered by the fit."""
    m = np.arange(0.0, 24.0)
    p0 = _p0(m)
    truth = np.array(truth, dtype=float)
    u_fn = fit.make_contact_u_fn(model, TREF, TSCALE, n_beads=N_BEADS)
    obs = np.array([
        fit.model_contact_mass(p0, m, float(T), truth, u_fn) for T in TEMPS
    ])
    loss_fn = fit._get_loss_fn("js")
    obj_fn, obj_args = fit.build_objective(False, TEMPS, m, obs, p0, u_fn, loss_fn)

    spec = fit.MODEL_REGISTRY[model]
    rng = np.random.default_rng(11)
    x0s = [np.array(spec["x0"], dtype=float)]
    x0s += [
        np.array([rng.uniform(lo, hi) for lo, hi in spec["bounds"]])
        for _ in range(7)
    ]
    best, obj = fit.fit_one_split(obj_fn, obj_args, x0s, spec["bounds"])
    assert obj < 1e-8, f"objective did not reach the synthetic truth: {obj:.3e}"
    for name, got, want, tol in zip(spec["param_names"], best.x, truth, atol):
        assert abs(got - want) < tol, f"{model}.{name}: {got:.6g} vs {want:.6g}"


def test_recovered_zero_curvature_data_is_fit_by_the_nested_model():
    """hs-generated data is reproduced by hs_m2_const with kappa2 ~ 0."""
    m = np.arange(0.0, 24.0)
    p0 = _p0(m)
    truth_hs = np.array([780.0, 2.7])
    u_hs = fit.make_contact_u_fn("hs", TREF, TSCALE)
    obs = np.array([
        fit.model_contact_mass(p0, m, float(T), truth_hs, u_hs) for T in TEMPS
    ])
    loss_fn = fit._get_loss_fn("js")
    spec = fit.MODEL_REGISTRY["hs_m2_const"]
    u_q = fit.make_contact_u_fn("hs_m2_const", TREF, TSCALE, n_beads=N_BEADS)
    obj_fn, obj_args = fit.build_objective(False, TEMPS, m, obs, p0, u_q, loss_fn)
    rng = np.random.default_rng(3)
    x0s = [np.array(spec["x0"], dtype=float)]
    x0s += [
        np.array([rng.uniform(lo, hi) for lo, hi in spec["bounds"]])
        for _ in range(7)
    ]
    best, obj = fit.fit_one_split(obj_fn, obj_args, x0s, spec["bounds"])
    assert obj < 1e-9, f"nested model did not reach the hs truth: {obj:.3e}"
    assert abs(best.x[2]) < 1e-2, f"kappa2 should collapse to 0, got {best.x[2]:.3e}"
    assert best.x[0] == pytest.approx(truth_hs[0], abs=5.0)
    assert best.x[1] == pytest.approx(truth_hs[1], abs=0.05)


# ---------------------------------------------------------------------------
# 6. Contact and joint-Rg weight consistency
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model, params", [
    ("hs", [780.0, 2.7]),
    ("hs_m2_const", [780.0, 2.7, 0.8]),
    ("hs_m2_hs", [780.0, 2.7, 300.0, 0.6]),
])
def test_joint_rg_reweighting_uses_the_same_contact_weights(model, params):
    """P(Rg|T) marginalizes the SAME exp[-u] weights the contact path uses."""
    crg, c_edges, rg_edges = _joint()
    m_centers = 0.5 * (c_edges[:-1] + c_edges[1:])
    p0 = crg.sum(axis=1)
    p0 = p0 / p0.sum()
    params = np.array(params, dtype=float)
    u_fn = fit.make_contact_u_fn(model, TREF, TSCALE, n_beads=N_BEADS)

    _, rg_mass = fit.predict_rg_from_joint(
        crg, c_edges, rg_edges, TEMPS, params, u_fn
    )
    for i, T in enumerate(TEMPS):
        p_m = fit.model_contact_mass(p0, m_centers, float(T), params, u_fn)
        # Marginalizing the joint with the contact weights must reproduce the
        # same P(m|T), and the resulting P(Rg|T) must be that mixture.
        cond = crg / crg.sum(axis=1, keepdims=True)
        expected = (cond * p_m[:, None]).sum(axis=0)
        assert np.allclose(rg_mass[i], expected, rtol=1e-11, atol=1e-14)
        assert rg_mass[i].sum() == pytest.approx(1.0, rel=1e-12)


def test_combined_objective_and_predict_agree_on_the_rg_term():
    """objective_combined's inline Rg reweighting matches predict_rg_from_joint."""
    crg, c_edges, rg_edges = _joint()
    m_centers = 0.5 * (c_edges[:-1] + c_edges[1:])
    p0 = crg.sum(axis=1)
    p0 = p0 / p0.sum()
    params = np.array([780.0, 2.7, 0.8])
    u_fn = fit.make_contact_u_fn("hs_m2_const", TREF, TSCALE, n_beads=N_BEADS)
    loss_fn = fit._get_loss_fn("js")

    _, rg_mass = fit.predict_rg_from_joint(
        crg, c_edges, rg_edges, TEMPS, params, u_fn
    )
    obs_ct = np.array([
        fit.model_contact_mass(p0, m_centers, float(T), params, u_fn) for T in TEMPS
    ])
    # With the model's own predictions as targets the combined objective is the
    # contact loss (0) plus the Rg loss against rg_mass (also 0).
    total = fit.objective_combined(
        params, TEMPS, m_centers, obs_ct, p0, crg, c_edges, rg_mass,
        u_fn, loss_fn, 1.0,
    )
    assert total < 1e-18


def test_unsupported_bins_do_not_set_the_stabilization_maximum():
    """A p0 == 0 bin must not deflate (or overflow) the surviving weights."""
    m = np.arange(0.0, 40.0)
    p0 = np.zeros_like(m)
    p0[:8] = _p0(m[:8])                       # support only at small m
    params = np.array([780.0, 2.7, -6.0])     # strong negative curvature
    u_fn = fit.make_contact_u_fn("hs_m2_const", TREF, TSCALE, n_beads=N_BEADS)
    with np.errstate(over="raise", invalid="raise"):
        p = fit.model_contact_mass(p0, m, 300.0, params, u_fn)
    assert np.all(np.isfinite(p))
    assert p.sum() == pytest.approx(1.0, rel=1e-12)
    assert np.all(p[8:] == 0.0)
    # And it is the true conditional distribution on the support, not the
    # uniform fallback the unstabilized version would have produced.
    w = p0[:8] * np.exp(-(u_fn(params, 300.0, m[:8])))
    assert np.allclose(p[:8], w / w.sum(), rtol=1e-12, atol=1e-15)


# ---------------------------------------------------------------------------
# 7. Chain-length resolution
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mod", [fit, dat], ids=["fitter", "fit_to_dat"])
def test_chain_length_prefers_n_beads_then_N(mod, tmp_path):
    p = tmp_path / "b_nbeads.npz"
    np.savez(p, n_beads=np.int64(44), N=np.int64(44))
    assert mod.read_baseline_chain_length(np.load(p)) == 44

    p2 = tmp_path / "b_N.npz"
    np.savez(p2, N=np.int64(30))
    assert mod.read_baseline_chain_length(np.load(p2)) == 30

    p3 = tmp_path / "b_legacy.npz"
    np.savez(p3, c_vals=np.arange(3))
    assert mod.read_baseline_chain_length(np.load(p3)) is None


@pytest.mark.parametrize("mod", [fit, dat], ids=["fitter", "fit_to_dat"])
def test_conflicting_chain_lengths_are_rejected(mod, tmp_path):
    p = tmp_path / "conflict.npz"
    np.savez(p, n_beads=np.int64(30), N=np.int64(44))
    with pytest.raises(ValueError, match="conflicting chain lengths"):
        mod.read_baseline_chain_length(np.load(p))

    with pytest.raises(ValueError, match="conflicts with"):
        mod.resolve_chain_length(30, 44, model_name="hs_m2_const")


@pytest.mark.parametrize("mod", [fit, dat], ids=["fitter", "fit_to_dat"])
def test_cli_N_is_a_fallback_and_only_the_new_models_require_it(mod):
    # baseline wins, agreeing CLI value is fine
    assert mod.resolve_chain_length(30, 30, model_name="hs_m2_const") == 30
    # fallback when the baseline has none
    assert mod.resolve_chain_length(None, 30, model_name="hs_m2_const") == 30
    # legacy linear fit against a baseline with no chain length still works
    assert mod.resolve_chain_length(None, None, model_name="hs") is None
    # the quadratic models refuse to proceed without one
    with pytest.raises(ValueError, match="chain length"):
        mod.resolve_chain_length(None, None, model_name="hs_m2_hs")
    with pytest.raises(ValueError, match="chain length"):
        mod.make_contact_u_fn("hs_m2_hs", TREF, TSCALE, n_beads=None)


# ---------------------------------------------------------------------------
# 8. Fit-summary loading and model-contract parity
# ---------------------------------------------------------------------------

def test_model_contract_parity_across_all_three_modules():
    contracts = {
        "fitter": fit.get_model_contract(),
        "fit_to_dat": dat.get_model_contract(),
        "remd": remd.get_model_contract(),
    }
    versions = {k: c["model_api_version"] for k, c in contracts.items()}
    # Strict parity: fitters and sampler must be on the SAME model API version.
    # A sampler left behind would happily read a fit summary and sample some
    # other potential, which is precisely what this handshake exists to prevent.
    assert (
        versions["fitter"]
        == versions["fit_to_dat"]
        == versions["remd"]
        == fit.MODEL_API_VERSION
    ), versions

    names = {k: set(c["models"]) for k, c in contracts.items()}
    # Strict parity again: all three must know exactly the same model set.
    assert names["fitter"] == names["fit_to_dat"] == names["remd"], {
        k: sorted(v) for k, v in names.items()
    }
    assert {"hs_m2_const", "hs_m2_hs", "saturating_cooperative_contact"} <= names["remd"]

    for model in sorted(names["fitter"]):
        entries = [c["models"][model] for c in contracts.values()]
        for key in ("param_names", "potential_kind", "quadratic_normalization",
                    "potential_normalization", "m_ref"):
            values = [e[key] for e in entries]
            assert all(v == values[0] for v in values), (model, key, values)
        # potential_definition is back-filled from each module's own description
        # for the models that predate the field, and those descriptions have
        # always been worded differently in the sampler than in the fitters (the
        # fitters' carry CLI hints) while describing the same potential -- which
        # is checked numerically elsewhere. Where a model declares the field
        # explicitly it is authoritative, and then every module must state it
        # identically and none may fall back to its description.
        defs = [e["potential_definition"] for e in entries]
        declared = [d for d, e in zip(defs, entries) if d != e["description"]]
        if declared:
            assert len(declared) == len(entries), (model, defs)
            assert all(d == declared[0] for d in declared), (model, defs)

    assert contracts["fitter"]["models"]["hs_m2_const"]["param_names"] == [
        "h1", "s1", "kappa2"
    ]
    assert contracts["fitter"]["models"]["hs_m2_hs"]["param_names"] == [
        "h1", "s1", "h2", "s2"
    ]
    for model in NEW_MODELS:
        e = contracts["fitter"]["models"][model]
        assert e["potential_kind"] == "contact_quadratic"
        assert e["quadratic_normalization"] == "m^2/(2N)"


def test_saturating_cooperative_contact_is_in_every_contract():
    """The new model must be declared identically by the fitters AND the sampler.

    Present in the fitters only would mean a fit summary the sampler cannot
    reproduce; present with different fields would mean the two disagree about
    what the fitted parameters mean.
    """
    model = "saturating_cooperative_contact"
    entries = {
        "fitter": fit.get_model_contract()["models"],
        "fit_to_dat": dat.get_model_contract()["models"],
        "remd": remd.get_model_contract()["models"],
    }
    for who, models in entries.items():
        assert model in models, who
        e = models[model]
        assert e["param_names"] == ["h_b", "s_b", "A0", "q_sat"], who
        assert e["potential_kind"] == "saturating_cooperative", who
        # It is NOT an m^2/(2N) model: the curvature normalization stays None and
        # the chain length serves the contact-fraction normalization instead.
        assert e["quadratic_normalization"] is None, who
        assert e["potential_normalization"] == "q = m/N", who
        assert e["m_ref"] == 0, who
        assert "A0*q^2/(1 + (q/q_sat)^2)" in e["potential_definition"], who


def test_current_fit_summary_loads_and_future_version_is_rejected(tmp_path):
    """End-to-end version gate on a summary the fitter actually wrote.

    A summary written by the current fitter must load in the sampler unchanged;
    a summary one version newer must be refused rather than silently sampled
    against whatever this sampler happens to implement.
    """
    crg, c_edges, rg_edges = _write_baseline(tmp_path / "base.npz")
    _write_remd(
        tmp_path / "remd.npz", crg, c_edges,
        np.array([780.0, 2.7, 1.2, 0.4]), "saturating_cooperative_contact",
    )
    out = tmp_path / "run"
    subprocess.run(
        [sys.executable, FITTER,
         "--remd", str(tmp_path / "remd.npz"),
         "--baseline", str(tmp_path / "base.npz"),
         "--contact_offset", "0", "--model", "saturating_cooperative_contact",
         "--loss", "js", "--n_restarts", "3", "--no-plots",
         "--outdir", str(out)],
        check=True, capture_output=True, text=True,
    )
    summary_path = out / "fit_summary.json"
    summary = json.loads(summary_path.read_text())
    assert summary["model_api_version"] == fit.MODEL_API_VERSION == remd.MODEL_API_VERSION

    loaded = remd.load_fit_summary_json(str(summary_path))
    assert loaded["model_name"] == "saturating_cooperative_contact"
    assert loaded["param_names"] == ["h_b", "s_b", "A0", "q_sat"]
    assert loaded["fit_chain_length"] == N_BEADS
    # the loaded parameters are the ones the fitter reported, in registry order
    np.testing.assert_allclose(
        loaded["params"], [summary["params"][n] for n in loaded["param_names"]]
    )
    # and they are usable: the sampler evaluates the same potential the fitter did
    u_fit = fit.make_contact_u_fn(
        loaded["model_name"], loaded["Tref"], loaded["Tscale"], n_beads=N_BEADS)
    for T in (280.0, 320.0, 360.0):
        for m in (0, 3, 11):
            assert abs(
                float(u_fit(np.asarray(loaded["params"]), T, float(m)))
                - remd.reduced_contact_potential(
                    m, T, loaded["model_name"], loaded["params"],
                    loaded["Tref"], loaded["Tscale"], N_BEADS)
            ) < 1e-12, (T, m)

    future = tmp_path / "future_summary.json"
    future.write_text(json.dumps({
        **summary, "model_api_version": fit.MODEL_API_VERSION + 1
    }))
    with pytest.raises(ValueError, match="newer"):
        remd.load_fit_summary_json(str(future))


def test_generic_interface_present_in_every_module():
    for mod in (fit, dat, remd):
        for name in ("reduced_bias", "make_b_fn", "quadratic_bias", "make_q_fn",
                     "reduced_contact_potential", "make_contact_u_fn"):
            assert callable(getattr(mod, name)), (mod.__name__, name)


def test_remd_sampler_samples_contact_quadratic_models():
    """The single-chain sampler now samples the full u = b*m + kappa*m^2/(2N).

    The single-chain sampler no longer refuses contact-quadratic models; it
    evaluates the generic potential through ``reduced_contact_potential`` at the
    runtime chain length.  Zero curvature must still nest hs bit-for-bit, and a
    nonzero curvature must change the acceptance criterion (proving the m^2 term
    is not silently dropped).  The ``require_linear_contact_potential`` guard is
    retained (linear-only samplers such as the multi-chain sampler import it) but
    is no longer invoked by this sampler.
    """
    N = 24
    Tref, Tscale = TREF, TSCALE
    T = 330.0
    hs_p = np.array([700.0, 2.4])
    # Zero curvature nests hs exactly for both new models.
    for model, p in (("hs_m2_const", [700.0, 2.4, 0.0]),
                     ("hs_m2_hs", [700.0, 2.4, 0.0, 0.0])):
        for m in (0.0, 5.0, 12.0):
            assert remd.reduced_contact_potential(
                m, T, model, np.array(p), Tref, Tscale, N
            ) == remd.reduced_contact_potential(m, T, "hs", hs_p, Tref, Tscale, N)
    # Nonzero curvature adds exactly kappa(T) * m^2 / (2N).
    kappa = 0.75
    p = np.array([700.0, 2.4, kappa])
    for m in (3.0, 8.0, 15.0):
        base = remd.reduced_contact_potential(m, T, "hs", hs_p, Tref, Tscale, N)
        got = remd.reduced_contact_potential(m, T, "hs_m2_const", p, Tref, Tscale, N)
        assert got == pytest.approx(base + kappa * m ** 2 / (2.0 * N), rel=1e-13)

    # The linear-only guard is retained for samplers that still need it (the
    # multi-chain sampler imports it): it passes linear models and refuses the
    # contact-quadratic ones.
    for model in LEGACY_MODELS:
        remd.require_linear_contact_potential(model)   # no raise
    for model in NEW_MODELS:
        with pytest.raises(NotImplementedError, match="nonlinear in m"):
            remd.require_linear_contact_potential(model)


def test_fit_summary_records_the_new_contract_fields(tmp_path):
    crg, c_edges, rg_edges = _write_baseline(tmp_path / "base.npz")
    _write_remd(
        tmp_path / "remd.npz", crg, c_edges, np.array([780.0, 2.7, 0.5]),
        "hs_m2_const",
    )
    out = tmp_path / "run"
    subprocess.run(
        [sys.executable, FITTER,
         "--remd", str(tmp_path / "remd.npz"),
         "--baseline", str(tmp_path / "base.npz"),
         "--contact_offset", "0", "--model", "hs_m2_const",
         "--loss", "js", "--n_restarts", "3", "--no-plots",
         "--outdir", str(out)],
        check=True, capture_output=True, text=True,
    )
    summary = json.loads((out / "fit_summary.json").read_text())
    assert summary["model_api_version"] == fit.MODEL_API_VERSION
    assert summary["potential_kind"] == "contact_quadratic"
    assert summary["quadratic_normalization"] == "m^2/(2N)"
    assert summary["fit_chain_length"] == N_BEADS
    assert list(summary["param_names"]) == ["h1", "s1", "kappa2"]
    # compatibility alias + the two explicit coefficient series
    n = summary["n_temps"]
    assert summary["reduced_bias_by_temperature"] == \
        summary["linear_coefficient_by_temperature"]
    assert len(summary["quadratic_coefficient_by_temperature"]) == n
    assert all(
        v == pytest.approx(summary["params"]["kappa2"], rel=1e-12)
        for v in summary["quadratic_coefficient_by_temperature"]
    )

    z = np.load(out / "fit_results.npz", allow_pickle=True)
    assert str(z["potential_kind"]) == "contact_quadratic"
    assert str(z["quadratic_normalization"]) == "m^2/(2N)"
    assert int(z["fit_chain_length"]) == N_BEADS
    assert np.array_equal(
        z["reduced_bias_by_temperature"], z["linear_coefficient_by_temperature"]
    )


def test_legacy_summary_keeps_linear_metadata(tmp_path):
    """A legacy baseline with no chain length still fits with a linear model."""
    crg, c_edges, rg_edges = _write_baseline(tmp_path / "base.npz", chain_key=None)
    _write_remd(tmp_path / "remd.npz", crg, c_edges, np.array([780.0, 2.7]), "hs")
    out = tmp_path / "run"
    subprocess.run(
        [sys.executable, FITTER,
         "--remd", str(tmp_path / "remd.npz"),
         "--baseline", str(tmp_path / "base.npz"),
         "--contact_offset", "0", "--model", "hs",
         "--loss", "js", "--n_restarts", "2", "--no-plots",
         "--outdir", str(out)],
        check=True, capture_output=True, text=True,
    )
    summary = json.loads((out / "fit_summary.json").read_text())
    assert summary["potential_kind"] == "linear"
    assert summary["quadratic_normalization"] is None
    assert summary["fit_chain_length"] is None
    assert all(v == 0.0 for v in summary["quadratic_coefficient_by_temperature"])


def test_quadratic_model_without_chain_length_fails_clearly(tmp_path):
    crg, c_edges, rg_edges = _write_baseline(tmp_path / "base.npz", chain_key=None)
    _write_remd(tmp_path / "remd.npz", crg, c_edges, np.array([780.0, 2.7]), "hs")
    res = subprocess.run(
        [sys.executable, FITTER,
         "--remd", str(tmp_path / "remd.npz"),
         "--baseline", str(tmp_path / "base.npz"),
         "--contact_offset", "0", "--model", "hs_m2_const",
         "--no-plots", "--outdir", str(tmp_path / "run")],
        capture_output=True, text=True,
    )
    assert res.returncode != 0
    assert "n_beads" in res.stderr and "--N" in res.stderr


# ---------------------------------------------------------------------------
# 9. Bending-enabled baseline compatibility
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mod", [fit, dat], ids=["fitter", "fit_to_dat"])
def test_bending_metadata_is_read_not_refitted(mod, tmp_path):
    p = tmp_path / "bend.npz"
    np.savez(p, kappa_bend=np.float64(0.35), N=np.int64(30))
    assert mod.read_baseline_kappa_bend(np.load(p)) == pytest.approx(0.35)
    # legacy baseline without the key reads as athermal in the bending sense
    p2 = tmp_path / "legacy.npz"
    np.savez(p2, N=np.int64(30))
    assert mod.read_baseline_kappa_bend(np.load(p2)) == 0.0
    # the baseline/CLI consistency rule still bites
    assert mod.resolve_kappa_bend(0.35, 0.35) == pytest.approx(0.35)
    with pytest.raises(ValueError, match="does not match the baseline"):
        mod.resolve_kappa_bend(0.35, 0.20)
    # kappa_bend is never a fitted parameter of any model
    for spec in mod.MODEL_REGISTRY.values():
        assert "kappa_bend" not in spec["param_names"]


def test_bending_enabled_baseline_fits_a_quadratic_model(tmp_path):
    """A bending-enabled baseline works unchanged: the penalty is already in P0."""
    crg, c_edges, rg_edges = _write_baseline(tmp_path / "base.npz", kappa_bend=0.4)
    _write_remd(
        tmp_path / "remd.npz", crg, c_edges, np.array([780.0, 2.7, 0.5]),
        "hs_m2_const",
    )
    out = tmp_path / "run"
    subprocess.run(
        [sys.executable, FITTER,
         "--remd", str(tmp_path / "remd.npz"),
         "--baseline", str(tmp_path / "base.npz"),
         "--contact_offset", "0", "--model", "hs_m2_const",
         "--kappa-bend", "0.4",
         "--loss", "js", "--n_restarts", "3", "--no-plots",
         "--outdir", str(out)],
        check=True, capture_output=True, text=True,
    )
    summary = json.loads((out / "fit_summary.json").read_text())
    assert summary["kappa_bend"] == pytest.approx(0.4)
    assert summary["bending_enabled"] is True
    assert summary["bend_definition"] == fit.BEND_DEFINITION
    assert summary["potential_kind"] == "contact_quadratic"
    assert summary["fit_chain_length"] == N_BEADS
    # no joint histogram over contacts AND bends is required
    assert "n_bend" not in summary


def test_bending_cli_mismatch_is_rejected_by_the_fitter(tmp_path):
    crg, c_edges, rg_edges = _write_baseline(tmp_path / "base.npz", kappa_bend=0.4)
    _write_remd(tmp_path / "remd.npz", crg, c_edges, np.array([780.0, 2.7]), "hs")
    res = subprocess.run(
        [sys.executable, FITTER,
         "--remd", str(tmp_path / "remd.npz"),
         "--baseline", str(tmp_path / "base.npz"),
         "--contact_offset", "0", "--model", "hs",
         "--kappa-bend", "0.1", "--no-plots",
         "--outdir", str(tmp_path / "run")],
        capture_output=True, text=True,
    )
    assert res.returncode != 0
    assert "does not match the baseline" in res.stderr


# ---------------------------------------------------------------------------
# 10. Nonlinear scalar-Rg feasibility dispatch
# ---------------------------------------------------------------------------

def _scalar_data_file(path, crg, c_edges, rg_edges, model, params, rg_scale=0.345):
    u_fn = dat.make_contact_u_fn(model, TREF, TSCALE, n_beads=N_BEADS)
    _, obs, _ = dat.predict_rg_summary_from_joint(
        crg, c_edges, rg_edges, TEMPS, np.array(params, dtype=float), u_fn,
        rg_scale=rg_scale, summary="rms", target_units="observed",
    )
    with open(path, "w") as fh:
        for T, r in zip(TEMPS, obs):
            fh.write(f"{T:.6f} {r:.6f} {r * 0.97:.6f} {r * 1.03:.6f}\n")
    return obs


def _run_scalar(tmp_path, model, params, outdir_name):
    crg, c_edges, rg_edges = _write_baseline(tmp_path / f"base_{outdir_name}.npz")
    dat_file = tmp_path / f"rg_{outdir_name}.dat"
    _scalar_data_file(dat_file, crg, c_edges, rg_edges, model, params)
    out = tmp_path / outdir_name
    res = subprocess.run(
        [sys.executable, DAT,
         "--baseline", str(tmp_path / f"base_{outdir_name}.npz"),
         "--rg-means-file", str(dat_file),
         "--rg-target-units", "observed", "--rg-scale", "0.345",
         "--rg-summary", "rms", "--rg-mean-loss", "mse",
         "--model", model, "--n_restarts", "3",
         "--rg-feasibility-scan", "--no-plots",
         "--outdir", str(out)],
        capture_output=True, text=True,
    )
    assert res.returncode == 0, res.stderr[-3000:]
    return out, res.stdout


def test_linear_model_keeps_the_bias_scan_and_its_filenames(tmp_path):
    out, stdout = _run_scalar(tmp_path, "hs", [780.0, 2.7], "lin")
    assert (out / "rg_feasibility.csv").exists()
    assert (out / "rg_feasibility_summary.json").exists()
    assert not (out / "rg_feasibility_contact_slices.csv").exists()
    fs = json.loads((out / "rg_feasibility_summary.json").read_text())
    assert "reachable_rg_min" in fs and "endpoint_limits" in fs
    assert "Bias-to-Rg feasibility scan" in stdout


def test_nonlinear_model_replaces_the_scan_with_the_slice_diagnostic(tmp_path):
    out, stdout = _run_scalar(
        tmp_path, "hs_m2_const", [780.0, 2.7, 0.5], "quad"
    )
    # the one-dimensional bias scan and its plot are NOT produced
    assert not (out / "rg_feasibility.csv").exists()
    assert not (out / "rg_feasibility.png").exists()
    # the model-independent diagnostic is
    slices = out / "rg_feasibility_contact_slices.csv"
    assert slices.exists()
    header = slices.read_text().splitlines()[0].split(",")
    assert header[:5] == [
        "contact_bin", "contact_value", "baseline_contact_mass",
        "conditional_rg_lattice", "conditional_rg_observed",
    ]
    assert len(slices.read_text().splitlines()) > 2

    fs = json.loads((out / "rg_feasibility_summary.json").read_text())
    assert fs["bias_scan_applicable"] is False
    assert fs["endpoint_limits_applicable"] is False
    assert "reachable_rg_min" not in fs
    assert "endpoint_limits" not in fs
    reason = fs["not_applicable_reason"]
    assert "one-dimensional" in reason and "linear in m" in reason
    # the rigorous all-b bound and the support check survive
    assert fs["global_outer_bound"]["min"] is not None
    assert fs["global_outer_bound"]["is_exact_reachable_range"] is False
    assert "zero_support_overlap" in fs["support"]
    assert "NOT APPLICABLE" in stdout

    # the fitted Rg curve and T_rg_max_slope are still reported
    summary = json.loads((out / "fit_summary.json").read_text())
    tm = summary["transition_metrics"]
    assert "T_rg_max_slope" in tm and tm["collapse_detected"] in (True, False)
    assert summary["potential_kind"] == "contact_quadratic"
    assert summary["fit_chain_length"] == N_BEADS
    assert summary["quadratic_normalization"] == "m^2/(2N)"
    assert len(summary["quadratic_coefficient_by_temperature"]) == len(TEMPS)
    z = np.load(out / "fit_results.npz", allow_pickle=True)
    assert np.array_equal(z["reduced_bias_by_temperature"], z["b_T"])
    assert z["q_T"].shape == (len(TEMPS),)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
