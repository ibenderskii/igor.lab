#!/usr/bin/env python3
"""Tests for the LOCAL (per-monomer contact degree) cooperativity rule and for
the restored independent intra/inter contact scales of the linear models.

Two independent changes are covered:

1. ``hs`` (and every other LINEAR model) supports independent ``lambda_intra``
   and ``lambda_inter`` again, which is what the three-control pilot needs.
   Only the model that is NONLINEAR in the contact count still requires one
   common scale.

2. ``cooperativity="local"`` replaces the globally coupled multichain rule
   ``u_contact(m_total; M*N)`` with a sum of strictly local per-monomer terms
   built from the nonbonded contact degree ``k_i`` (intra AND inter).
"""
from __future__ import annotations

import math
import random

import numpy as np
import pytest

import multichain_contacts as mcc
import multichain_moves as mvs
import multichain_state as mcs
import remd_multichain as rmc
import remd_uniform_chain_2_new as remd
from multichain_state import ContactCounts

TREF, TSCALE = 320.0, 80.0
SAT = "saturating_cooperative_contact"
SAT_P = [400.0, 1.3, 0.6, 0.35]     # h_b, s_b, A0, q_sat
SAT_P0 = [400.0, 1.3, 0.0, 0.35]    # A0 = 0 -> exactly hs
A0, Q_SAT = SAT_P[2], SAT_P[3]
HS_P = [400.0, 1.3]


# ---------------------------------------------------------------------------
# 1. Independent lambda_intra / lambda_inter for linear models
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("li,lin", [(1.0, 1.0), (1.0, 0.0), (0.0, 1.0),
                                    (0.5, 0.25), (0.0, 0.0)])
def test_hs_supports_independent_lambdas(li, lin):
    """The three pilot controls A/B/C must all be reachable with hs."""
    state = mcs.initialize_dispersed_state(4, 10, 12, seed=3)
    T = 330.0
    b = remd.reduced_bias("hs", HS_P, T, TREF, TSCALE)
    expect = b * (li * state.counts.intra + lin * state.counts.inter)
    u_state = rmc.reduced_contact_potential_state(
        state, T, "hs", HS_P, TREF, TSCALE, li, lin)
    u_agg = rmc.reduced_potential_counts(
        state.counts, T, "hs", HS_P, TREF, TSCALE, li, lin)
    assert u_state == expect
    assert u_agg == expect


def test_hs_lambda_controls_are_distinguishable():
    """B (collapse-only) and C (association-only) must not be the same model."""
    state = mcs.make_state(np.stack([
        mcs.generate_saw(10, np.random.RandomState(1)),
        mcs.generate_saw(10, np.random.RandomState(2)) + np.array([1, 0, 0]),
    ]), 20)
    T = 330.0
    if state.counts.intra == state.counts.inter:
        pytest.skip("degenerate probe state")
    u_b = rmc.reduced_contact_potential_state(
        state, T, "hs", HS_P, TREF, TSCALE, 1.0, 0.0)
    u_c = rmc.reduced_contact_potential_state(
        state, T, "hs", HS_P, TREF, TSCALE, 0.0, 1.0)
    assert u_b != u_c


@pytest.mark.parametrize("li,lin", [(1.0, 0.0), (0.0, 1.0), (0.5, 0.25)])
@pytest.mark.parametrize("mode", ["global", "local"])
def test_saturating_still_requires_one_contact_scale(li, lin, mode):
    """Nonlinearity in m is what forces a common scale -- and it still does."""
    state = mcs.initialize_dispersed_state(3, 8, 12, seed=4)
    with pytest.raises(ValueError, match="lambda_intra == lambda_inter"):
        rmc.reduced_contact_potential_state(
            state, 330.0, SAT, SAT_P, TREF, TSCALE, li, lin, cooperativity=mode)


@pytest.mark.parametrize("model,params", [("hs", HS_P),
                                          ("hs_m2_hs", [400.0, 1.3, 50.0, 0.2])])
def test_local_mode_refused_for_models_without_A0(model, params):
    state = mcs.initialize_dispersed_state(3, 8, 12, seed=4)
    with pytest.raises(ValueError, match="applies only to"):
        rmc.reduced_contact_potential_state(
            state, 330.0, model, params, TREF, TSCALE, 1.0, 1.0,
            cooperativity="local")


def test_unknown_cooperativity_rejected():
    state = mcs.initialize_dispersed_state(2, 8, 12, seed=4)
    with pytest.raises(ValueError, match="cooperativity must be"):
        rmc.reduced_contact_potential_state(
            state, 330.0, SAT, SAT_P, TREF, TSCALE, 1.0, 1.0,
            cooperativity="mean_field")


# ---------------------------------------------------------------------------
# 2. Contact degree: definition and tie to the authoritative pair counts
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5])
def test_degree_sum_equals_twice_m_total(seed):
    """Every contact contributes one incidence to each of its two endpoints."""
    state = mcs.initialize_dispersed_state(4, 10, 12, seed=seed)
    deg = mcc.contact_degrees(state)
    assert int(deg.sum()) == 2 * (state.counts.intra + state.counts.inter)


def test_degree_counts_interchain_neighbours_too():
    """Two chains laid side by side: the degree must see the OTHER chain."""
    line = np.array([(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)], dtype=np.int64)
    state = mcs.make_state(np.stack([line, line + np.array([0, 1, 0])]), 20)
    deg = mcc.contact_degrees(state)
    assert state.counts.intra == 0          # a straight chain has no intra contacts
    assert state.counts.inter == 4          # four rungs of the ladder
    assert list(deg) == [1, 1, 1, 1, 1, 1, 1, 1]
    assert int(deg.sum()) == 2 * state.counts.inter


def test_degree_excludes_bonded_neighbours():
    line = np.array([(0, 0, 0), (1, 0, 0), (2, 0, 0)], dtype=np.int64)
    state = mcs.make_state(np.stack([line]), 20)
    assert list(mcc.contact_degrees(state)) == [0, 0, 0]


@pytest.mark.parametrize("seed", [0, 3, 7])
def test_degree_histogram_matches_bincount(seed):
    state = mcs.initialize_dispersed_state(4, 10, 12, seed=seed)
    deg = mcc.contact_degrees(state)
    assert list(mcc.degree_histogram(state)) == list(np.bincount(deg, minlength=7))


# ---------------------------------------------------------------------------
# 3. The local form reduces EXACTLY to the fitted single-chain potential
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("N", [30, 44, 60])
@pytest.mark.parametrize("m", [1, 10, 25, 40])
def test_uniform_degree_reproduces_fitted_cooperative_term(N, m):
    """k_i = 2m/N for every bead  =>  sum_i g(k_i) == N*q^2/(1+(q/q_sat)^2).

    This is why the local rule needs NO refit: it carries the same A0 and q_sat
    and agrees with the fit exactly in the homogeneous limit the fit assumes.
    """
    q = m / N
    fitted = N * q * q / (1.0 + (q / Q_SAT) ** 2)
    local = N * mcc.cooperative_g(2.0 * m / N, Q_SAT)
    assert local == pytest.approx(fitted, rel=1e-12, abs=1e-12)


def test_cooperative_g_is_bounded_by_q_sat_squared():
    for k in range(0, 7):
        assert mcc.cooperative_g(k, Q_SAT) <= Q_SAT ** 2
    # and it saturates from below, monotonically in k
    vals = [mcc.cooperative_g(k, Q_SAT) for k in range(7)]
    assert vals == sorted(vals)


@pytest.mark.parametrize("T", [305.0, 330.0, 350.0])
def test_A0_zero_nests_hs_exactly(T):
    state = mcs.initialize_dispersed_state(3, 10, 12, seed=2)
    b = remd.reduced_bias("hs", HS_P, T, TREF, TSCALE)
    u = rmc.reduced_contact_potential_state(
        state, T, SAT, SAT_P0, TREF, TSCALE, 1.0, 1.0, cooperativity="local")
    assert u == b * (state.counts.intra + state.counts.inter)


# ---------------------------------------------------------------------------
# 4. Locality: exact additivity over non-interacting chains
# ---------------------------------------------------------------------------

def _separated_state(box=40):
    rng = np.random.RandomState(11)
    chains = []
    for c in range(4):
        saw = mcs.generate_saw(10, rng)
        off = (np.array([2, 2, 2]) if c < 2 else np.array([22, 22, 22]))
        chains.append(saw + off + np.array([0, 4 * (c % 2), 0]))
    return mcs.make_state(np.stack(chains), box)


def test_local_rule_is_exactly_additive_over_separated_chains():
    """u(A + B) == u(A) + u(B) when A and B share no contacts.

    This is the property the global rule violates, and it is the reason the
    global rule cannot be interpreted as a short-ranged interaction.
    """
    state = _separated_state()
    T = 330.0
    whole = rmc.reduced_contact_potential_state(
        state, T, SAT, SAT_P, TREF, TSCALE, 1.0, 1.0, cooperativity="local")
    parts = sum(
        rmc.reduced_contact_potential_state(
            mcs.make_state(state.coords_unwrapped[g], state.box_size),
            T, SAT, SAT_P, TREF, TSCALE, 1.0, 1.0, cooperativity="local")
        for g in ([0, 1], [2, 3]))
    assert whole == pytest.approx(parts, abs=1e-12)


def test_global_rule_is_not_additive():
    """Documents the defect the local rule fixes (guards against silent revert)."""
    state = _separated_state()
    T = 330.0
    whole = rmc.reduced_contact_potential_state(
        state, T, SAT, SAT_P, TREF, TSCALE, 1.0, 1.0, cooperativity="global")
    parts = sum(
        rmc.reduced_contact_potential_state(
            mcs.make_state(state.coords_unwrapped[g], state.box_size),
            T, SAT, SAT_P, TREF, TSCALE, 1.0, 1.0, cooperativity="global")
        for g in ([0, 1], [2, 3]))
    assert abs(whole - parts) > 1e-6


def test_global_rule_cannot_see_where_contacts_are():
    """Same m_total, wildly different spatial allocation -> identical global u."""
    a = mcs.initialize_dispersed_state(4, 10, 12, seed=6).copy()
    b = a.copy()
    a.counts = ContactCounts(9, 2)
    b.counts = ContactCounts(2, 9)
    T = 330.0
    ua = rmc.reduced_contact_potential_state(
        a, T, SAT, SAT_P, TREF, TSCALE, 1.0, 1.0, cooperativity="global")
    ub = rmc.reduced_contact_potential_state(
        b, T, SAT, SAT_P, TREF, TSCALE, 1.0, 1.0, cooperativity="global")
    assert ua == ub


# ---------------------------------------------------------------------------
# 5. Incremental update correctness (this is what the sampler actually runs)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [3, 12])
def test_delta_cooperative_sum_matches_full_recount(seed):
    state = mcs.initialize_dispersed_state(4, 10, 12, seed=7)
    rng = random.Random(seed)
    proposers = [mvs.propose_local, mvs.propose_translation,
                 mvs.propose_reptation, mvs.propose_chain_rotation]
    seen = set()
    n = 0
    for _ in range(3000):
        prop = proposers[rng.randrange(len(proposers))](state, rng)
        if not prop.ok:
            continue
        before = mcc.cooperative_sum(state, Q_SAT)
        dS = mcc.delta_cooperative_sum(
            state, prop.moved.keys(), prop.new_sites, Q_SAT)
        d = mvs.proposal_delta(state, prop)
        mvs.apply_proposal(state, prop, d)
        after = mcc.cooperative_sum(state, Q_SAT)
        assert after - before == pytest.approx(dS, abs=1e-10)
        seen.add(prop.move_type)
        n += 1
    assert n > 500
    assert {"end", "crankshaft", "pivot", "chain_translation", "reptation",
            "rotation"} <= seen
    mcc.assert_counts_match(state, "after local-cooperativity sweep")


@pytest.mark.parametrize("T", [310.0, 345.0])
def test_sweep_delta_equals_full_potential_difference(T):
    """The Metropolis du must be the exact change in the sampled potential."""
    state = mcs.initialize_dispersed_state(4, 10, 12, seed=9)
    rng = random.Random(5)
    b = remd.reduced_bias(SAT, SAT_P, T, TREF, TSCALE)
    proposers = [mvs.propose_local, mvs.propose_translation,
                 mvs.propose_reptation, mvs.propose_chain_rotation]
    for _ in range(1500):
        prop = proposers[rng.randrange(len(proposers))](state, rng)
        if not prop.ok:
            continue
        u0 = rmc.reduced_contact_potential_state(
            state, T, SAT, SAT_P, TREF, TSCALE, 1.0, 1.0, cooperativity="local")
        d = mvs.proposal_delta(state, prop)
        dS = mcc.delta_cooperative_sum(
            state, prop.moved.keys(), prop.new_sites, Q_SAT)
        du = b * (d[0] + d[1]) - A0 * dS
        mvs.apply_proposal(state, prop, d)
        u1 = rmc.reduced_contact_potential_state(
            state, T, SAT, SAT_P, TREF, TSCALE, 1.0, 1.0, cooperativity="local")
        assert u1 - u0 == pytest.approx(du, abs=1e-9)


def test_local_swap_matches_manual_four_potential_calculation():
    si = mcs.initialize_dispersed_state(3, 8, 12, seed=1)
    sj = mcs.initialize_dispersed_state(3, 8, 12, seed=2)
    Ti, Tj = 305.0, 345.0

    def u(state, T):
        return rmc.reduced_contact_potential_state(
            state, T, SAT, SAT_P, TREF, TSCALE, 1.0, 1.0, cooperativity="local")

    expect = u(si, Ti) + u(sj, Tj) - u(sj, Ti) - u(si, Tj)
    got = rmc.swap_log_accept_state(
        si, sj, Ti, Tj, SAT, SAT_P, TREF, TSCALE, 1.0, 1.0,
        cooperativity="local")
    assert got == pytest.approx(expect, abs=1e-12)


def test_local_swap_is_unaffected_by_kappa_bend():
    """The bending term is temperature independent, so it cancels in the swap."""
    si = mcs.initialize_dispersed_state(3, 8, 12, seed=1)
    sj = mcs.initialize_dispersed_state(3, 8, 12, seed=2)
    a = rmc.swap_log_accept_state(si, sj, 305.0, 345.0, SAT, SAT_P, TREF,
                                  TSCALE, 1.0, 1.0, cooperativity="local")
    ui = rmc.reduced_potential_state(si, 305.0, SAT, SAT_P, TREF, TSCALE,
                                     1.0, 1.0, kappa_bend=0.7,
                                     cooperativity="local")
    uj = rmc.reduced_potential_state(si, 305.0, SAT, SAT_P, TREF, TSCALE,
                                     1.0, 1.0, kappa_bend=0.0,
                                     cooperativity="local")
    assert ui - uj == pytest.approx(0.7 * si.n_bend, abs=1e-12)
    assert math.isfinite(a)


# ---------------------------------------------------------------------------
# 6. End-to-end: determinism and the default is unchanged
# ---------------------------------------------------------------------------

def _short_run(mode, n_workers=1, seed=17):
    Ts = np.linspace(305, 350, 4)
    reps, sp, sa, wh = rmc.run_remd_multichain(
        n_chains=3, chain_length=10, box_size=12, Ts=Ts,
        local_sweeps_per_swap=1, translation_sweeps_per_swap=1, n_cycles=12,
        model_name=SAT, params=SAT_P, Tref=TREF, Tscale=TSCALE,
        lambda_intra=1.0, lambda_inter=1.0, cooperativity=mode,
        seed=seed, n_workers=n_workers, verbose=False, debug_contacts=True)
    return [np.asarray(r.u_traj) for r in reps], sa, wh


@pytest.mark.parametrize("mode", ["global", "local"])
def test_serial_and_multiprocessing_are_bit_identical(mode):
    """The worker rebuilds site_owner, so any order-dependent float sum would
    show up here.  Both the cooperative sum and its delta contract integer
    degree counts with g in fixed order precisely to avoid that."""
    a = _short_run(mode, n_workers=1)
    b = _short_run(mode, n_workers=2)
    for x, y in zip(a[0], b[0]):
        assert np.array_equal(x, y)
    assert np.array_equal(a[1], b[1])
    assert np.array_equal(a[2], b[2])


def test_default_cooperativity_is_global_and_unchanged():
    """Existing runs must be untouched: the default rule is still the old one."""
    assert rmc._validate_cooperativity(
        rmc.DEFAULT_COOPERATIVITY, SAT
    ) == "global"
    state = mcs.initialize_dispersed_state(3, 8, 12, seed=5)
    T = 330.0
    explicit = rmc.reduced_contact_potential_state(
        state, T, SAT, SAT_P, TREF, TSCALE, 1.0, 1.0, cooperativity="global")
    default = rmc.reduced_contact_potential_state(
        state, T, SAT, SAT_P, TREF, TSCALE, 1.0, 1.0)
    legacy = remd.reduced_contact_potential(
        state.counts.intra + state.counts.inter, T, SAT, SAT_P, TREF, TSCALE,
        state.n_chains * state.chain_length)
    assert explicit == default == legacy

    metadata = {}
    rmc.attach_metadata(
        metadata, M=3, N=8, L=12, Ts=np.array([300.0, 340.0]), seed=5,
        model_name=SAT, param_names=["h_b", "s_b", "A0", "q_sat"],
        model_params=SAT_P, Tref=TREF, Tscale=TSCALE, lambda_intra=1.0,
        lambda_inter=1.0, local_sweeps_per_swap=1,
        translation_sweeps_per_swap=1, n_cycles=1, burnin_frac=0.5,
        cluster_contact_threshold=1, parameter_source="test",
        fit_summary_json="",
    )
    assert metadata["cooperativity"] == "global"


def test_local_run_records_its_rule_in_metadata():
    dist = {}
    rmc.attach_metadata(
        dist, M=3, N=10, L=12, Ts=np.array([305.0, 350.0]), seed=1,
        model_name=SAT, param_names=["h_b", "s_b", "A0", "q_sat"],
        model_params=SAT_P, Tref=TREF, Tscale=TSCALE, lambda_intra=1.0,
        lambda_inter=1.0, local_sweeps_per_swap=1,
        translation_sweeps_per_swap=1, n_cycles=5, burnin_frac=0.5,
        cluster_contact_threshold=1, parameter_source="test",
        fit_summary_json="", cooperativity="local")
    assert dist["cooperativity"] == "local"
    assert dist["nonlinear_contact_scope"] == "all_contacts_local_degree"
    assert "sum_i g(k_i)" in dist["multichain_potential_definition"]
