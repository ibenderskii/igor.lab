"""The Numba multichain lane sweep must reproduce the Python sweep bit for bit.

Every comparison covers coordinates, occupancy, intra/inter and per-chain
contact counts, bend count, degree cache + joint table, move counters and the
lane RNG state (multichain_kernels.snapshot + rng.getstate()).
"""
from __future__ import annotations

import json
import random

import numpy as np
import pytest

import remd_multichain as rm
import multichain_contacts as mcc
import multichain_kernels as mk

pytestmark = pytest.mark.skipif(not mk.NUMBA_AVAILABLE, reason="numba not installed")

LCS = [173.7070586551577, -0.7663649589655334, 2.018635667565305, 1.8024265951628209]
CONFIGS = {
    "hs": ("hs", [558.879803466925, 2.0099937029926584], None, (1.0, 1.0), 1.0, 1.0),
    "hs_lambdas": ("hs", [558.879803466925, 2.0099937029926584], None, (1.0, 1.0), 1.0, 0.7),
    "hs_quadratic": ("hs_quadratic", [538.88498, 1.960151, 0.290520], None, (1.0, 1.0), 1.0, 0.8),
    "sat_global": ("saturating_cooperative_contact", [200.0, 0.0, 1.0, 1.5], None, (1.0, 1.0), 1.0, 1.0),
    "sat_local": ("saturating_cooperative_contact", [200.0, 0.0, 1.0, 1.5], "local", (1.0, 1.0), 1.0, 1.0),
    "lcs": ("local_coordination_saturation", LCS, None, (1.0, 1.0), 1.0, 1.0),
    "lcs_weights": ("local_coordination_saturation", LCS, None, (1.0, 0.5), 1.0, 1.0),
}


def _kw(cfg, M, N, kb):
    model, params, coop, w, li, lin = CONFIGS[cfg]
    return dict(model_name=model, params=params, Tref=320.0, Tscale=80.0, lambda_intra=li,
                lambda_inter=lin, n_local=M * N, n_translation=M, n_reptation=M, n_rotation=M,
                kappa_bend=kb, cooperativity=coop, degree_weights=w)


def _lockstep(cfg, M, N, L, T, kb, warm=1, sweeps=2, seed=3):
    kw = _kw(cfg, M, N, kb)
    track = rm._tracks_degrees(kw["model_name"], kw["cooperativity"])
    rep = rm._build_independent_replicas(np.array([T]), M, N, L, seed, 1, track_degrees=track)[0]
    for k in range(warm):
        rm.mc_sweep(rep.state, rep.move_counters, T, rng=random.Random(k), kernel="python", **kw)
    sa, sb = rep.state.copy(), rep.state.copy()
    ca, cb = rep.move_counters.copy(), rep.move_counters.copy()
    for k in range(sweeps):
        ra, rb = random.Random(100 + k), random.Random(100 + k)
        rm.mc_sweep(sa, ca, T, rng=ra, kernel="python", **kw)
        rm.mc_sweep(sb, cb, T, rng=rb, kernel="numba", **kw)
        assert mk.snapshot(sa, ca) == mk.snapshot(sb, cb), f"{cfg} state differs at sweep {k}"
        assert ra.getstate() == rb.getstate(), f"{cfg} RNG differs at sweep {k}"
    return sa


@pytest.mark.parametrize("cfg", sorted(CONFIGS))
@pytest.mark.parametrize("kb", [0.0, 0.5])
def test_lockstep_small_systems(cfg, kb):
    for T in (290.0, 370.0):
        _lockstep(cfg, 4, 12, 8, T, kb)


@pytest.mark.parametrize("cfg", ["hs", "lcs"])
def test_lockstep_production_size(cfg):
    _lockstep(cfg, 16, 60, 22, 300.0, 0.0, warm=1, sweeps=1)


@pytest.mark.parametrize("M,N,L", [(3, 6, 3), (2, 12, 4), (5, 3, 5), (4, 2, 4)])
def test_lockstep_tight_boxes_and_short_chains(M, N, L):
    # chains that wrap around tiny periodic boxes, N = 3 (no reptation-born
    # bends) and N = 2 (pivot/crankshaft always invalid)
    for cfg in ("hs", "lcs"):
        try:
            _lockstep(cfg, M, N, L, 300.0, 0.3, warm=1, sweeps=2, seed=M + N + L)
        except Exception as exc:  # pragma: no cover - placement may be impossible
            if "place" in str(exc).lower():
                pytest.skip(f"cannot place M={M} N={N} in L={L}")
            raise


def test_integer_observables_match_oracles():
    for (M, N, L, T) in ((16, 60, 19, 290.0), (3, 12, 4, 300.0), (4, 20, 6, 280.0)):
        kw = _kw("hs", M, N, 0.0)
        rep = rm._build_independent_replicas(np.array([T]), M, N, L, 5, 1)[0]
        for k in range(6):
            rm.mc_sweep(rep.state, rep.move_counters, T, rng=random.Random(k), kernel="numba", **kw)
            a, b = mcc.contact_degrees_split(rep.state), mk.contact_degrees_split(rep.state)
            assert a.dtype == b.dtype and np.array_equal(a, b)
            assert mcc.interchain_pair_counts(rep.state) == mk.interchain_pair_counts(rep.state)


def test_run_remd_multichain_kernels_agree():
    common = dict(n_chains=3, chain_length=12, box_size=8, Ts=np.array([290.0, 330.0, 370.0]),
                  local_sweeps_per_swap=1, translation_sweeps_per_swap=1, n_cycles=6,
                  model_name="local_coordination_saturation", params=LCS, Tref=320.0,
                  Tscale=80.0, kappa_bend=0.2, seed=9, verbose=False)

    def run(**extra):
        reps, sp, sa, wh = rm.run_remd_multichain(**common, **extra)
        return ([(r.u_traj, r.m_intra_traj, r.m_inter_traj, r.rg_traj, r.lcs_traj,
                  r.state.coords_unwrapped.tolist(), r.move_counters.tolist()) for r in reps],
                sp.tolist(), sa.tolist(), wh.tolist())

    ref = run(kernel="python")
    assert run(kernel="numba") == ref
    assert run(kernel="numba", n_workers=2) == ref
    assert rm.LAST_SAMPLER_INFO["kernel"] == "numba"


def test_resolution_and_fallback():
    kind, info = rm.resolve_sampler_kernel(
        "auto", model_name="hs", params=CONFIGS["hs"][1], Tref=320.0, Tscale=80.0, M=3, N=10,
        L=7, Ts=[290.0, 350.0], lambda_intra=1.0, lambda_inter=1.0, kappa_bend=0.0,
        cooperativity=None, degree_weights=(1.0, 1.0))
    assert kind == "numba" and info["selfcheck"]["ok"]
    saved = list(rm._MCK_MODULE)
    try:
        rm._MCK_MODULE[:] = [None]
        kind, info = rm.resolve_sampler_kernel(
            "auto", model_name="hs", params=CONFIGS["hs"][1], Tref=320.0, Tscale=80.0, M=3,
            N=10, L=7, Ts=[300.0], lambda_intra=1.0, lambda_inter=1.0, kappa_bend=0.0,
            cooperativity=None, degree_weights=(1.0, 1.0))
        assert kind == "python"
        with pytest.raises(RuntimeError):
            rm.resolve_sampler_kernel(
                "numba", model_name="hs", params=CONFIGS["hs"][1], Tref=320.0, Tscale=80.0,
                M=3, N=10, L=7, Ts=[300.0], lambda_intra=1.0, lambda_inter=1.0,
                kappa_bend=0.0, cooperativity=None, degree_weights=(1.0, 1.0))
    finally:
        rm._MCK_MODULE[:] = saved


# --- --kernel auto must fall back to Python when Numba fails to compile ------

if mk.NUMBA_AVAILABLE:
    from numba import njit as _njit
    from numba.core.errors import TypingError

    @_njit
    def _uncompilable(*args):
        # A genuine Numba compilation failure (array + str): raises TypingError
        # on the first call, the way an unsupported numba/numpy combination would.
        return args[0] + "x"


def _resolve(requested, **over):
    kw = dict(model_name="hs", params=CONFIGS["hs"][1], Tref=320.0, Tscale=80.0, M=3, N=10,
              L=7, Ts=[290.0, 350.0], lambda_intra=1.0, lambda_inter=1.0, kappa_bend=0.0,
              cooperativity=None, degree_weights=(1.0, 1.0))
    kw.update(over)
    return rm.resolve_sampler_kernel(requested, **kw)


@pytest.mark.parametrize("target,where", [("_sweep", "lane sweep"),
                                          ("_pair_matrix", "observables"),
                                          ("_degrees_split", "observables")])
def test_auto_falls_back_on_compilation_error(monkeypatch, target, where):
    monkeypatch.setattr(mk, target, _uncompilable)
    with pytest.warns(RuntimeWarning, match="numba self-check failed"):
        kind, info = _resolve("auto")
    assert kind == "python"
    check = info["selfcheck"]
    assert not check["ok"] and check["error_type"] == "TypingError"
    assert where in check["reason"] and "TypingError" in info["reason"]
    assert "_exception" not in check and "\x1b" not in check["error"]
    json.dumps(info)  # stored in run_summary.json


@pytest.mark.parametrize("target", ["_sweep", "_pair_matrix", "_degrees_split"])
def test_explicit_numba_raises_with_cause(monkeypatch, target):
    monkeypatch.setattr(mk, target, _uncompilable)
    with pytest.raises(RuntimeError, match="TypingError") as ei:
        _resolve("numba")
    assert "--kernel python" in str(ei.value)
    assert isinstance(ei.value.__cause__, TypingError)


def test_auto_falls_back_when_observables_differ(monkeypatch):
    monkeypatch.setattr(mk, "contact_degrees_split",
                        lambda st: mcc.contact_degrees_split(st) + 1)
    with pytest.warns(RuntimeWarning):
        kind, info = _resolve("auto")
    assert kind == "python" and "observables differ" in info["reason"]
    assert info["selfcheck"]["temperatures"][-1]["observables_identical"] is False


def test_selfcheck_records_observable_check():
    kind, info = _resolve("auto", model_name="local_coordination_saturation", params=LCS,
                          kappa_bend=0.3)
    assert kind == "numba"
    temps = info["selfcheck"]["temperatures"]
    assert temps and all(t["identical"] and t["observables_identical"] for t in temps)


def test_python_reference_errors_propagate(monkeypatch):
    real = rm.mc_sweep

    def sweep(*a, kernel="python", **k):
        if kernel == "python":
            raise ValueError("python reference failure")
        return real(*a, kernel=kernel, **k)

    monkeypatch.setattr(rm, "mc_sweep", sweep)
    with pytest.raises(ValueError, match="python reference failure"):
        _resolve("auto")


@pytest.mark.parametrize("target", ["_sweep", "_pair_matrix"])
def test_run_remd_multichain_auto_with_broken_numba_matches_python(monkeypatch, target):
    common = dict(n_chains=3, chain_length=10, box_size=7, Ts=np.array([290.0, 350.0]),
                  local_sweeps_per_swap=1, translation_sweeps_per_swap=1, n_cycles=4,
                  model_name="hs", params=CONFIGS["hs"][1], Tref=320.0, Tscale=80.0,
                  kappa_bend=0.2, seed=2, verbose=False)

    def run(**extra):
        reps, sp, sa, wh = rm.run_remd_multichain(**common, **extra)
        return ([(r.u_traj, r.m_intra_traj, r.m_inter_traj, r.rg_traj, r.lcs_traj,
                  r.state.coords_unwrapped.tolist(), r.move_counters.tolist()) for r in reps],
                sp.tolist(), sa.tolist(), wh.tolist())

    ref = run(kernel="python")
    monkeypatch.setattr(mk, target, _uncompilable)
    with pytest.warns(RuntimeWarning):
        got = run(kernel="auto")
    assert got == ref
    assert rm.LAST_SAMPLER_INFO["kernel"] == "python"
    assert rm.LAST_SAMPLER_INFO["selfcheck"]["error_type"] == "TypingError"
