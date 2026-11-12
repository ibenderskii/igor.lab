"""The Numba multichain lane sweep must reproduce the Python sweep bit for bit.

Every comparison covers coordinates, occupancy, intra/inter and per-chain
contact counts, bend count, degree cache + joint table, move counters and the
lane RNG state (multichain_kernels.snapshot + rng.getstate()).
"""
from __future__ import annotations

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
