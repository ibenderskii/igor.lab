"""The Numba sweep must reproduce the Python sweep bit for bit.

Every check compares the complete replica state (chain, occupancy, contact
count, coordination histogram, bend count, energy, move and acceptance
counters) AND the Python ``random`` state after the sweep.
"""
from __future__ import annotations

import copy
import json
import random
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

AUTO_DIR = Path(__file__).resolve().parents[1]
if str(AUTO_DIR) not in sys.path:
    sys.path.insert(0, str(AUTO_DIR))

import remd_uniform_chain_2_new as R  # noqa: E402
import isaw_kernels as K  # noqa: E402

MODELS = {
    "hs": [415.59484, 1.582392],
    "tc_scale": [2.0, 330.0],
    "hs_quadratic": [431.48082, 1.650975, 0.224333],
    "poly2": [-0.302489, -0.340247, 0.308173],
    "poly3": [-0.303747, -0.310642, 0.322663, -0.197050],
    "heat_capacity": [416.76477, 1.605548, -7.370982],
    "hs_m2_const": [415.0, 1.58, 0.3],
    "hs_m2_hs": [415.0, 1.58, 10.0, 0.02],
    "saturating_cooperative_contact": [200.0, 0.0, 1.0, 1.5],
    "local_coordination_saturation": [173.7070586551577, -0.7663649589655334,
                                      2.018635667565305, 1.8024265951628209],
}


def _snap(rep):
    s = rep.state
    return (tuple(s.chain), frozenset(s.occ), s.m, s.E, s.n_bend,
            None if s.coordination_hist is None else tuple(s.coordination_hist),
            rep.local_acc, rep.local_prop, rep.move_counters.tobytes())


def _pair(model, params, N, T, kb, steps, seed=7, warm=1500):
    random.seed(seed)
    st = R.ChainState.initial_straight(N, T, model, params, 320.0, 80.0)
    rep = R.Replica(T=T, state=st)
    R.mc_sweep(rep, warm, model, params, 320.0, 80.0, kb, N)
    a, b = copy.deepcopy(rep), copy.deepcopy(rep)
    s0 = random.getstate()
    R.mc_sweep(a, steps, model, params, 320.0, 80.0, kb, N)
    sa = random.getstate()
    random.setstate(s0)
    K.mc_sweep_numba(R, b, steps, model, params, 320.0, 80.0, kb, N)
    sb = random.getstate()
    return _snap(a), sa, _snap(b), sb


@unittest.skipUnless(K.NUMBA_AVAILABLE, "numba not installed")
class NumbaKernelTests(unittest.TestCase):
    def test_rng_stream_matches_cpython(self):
        for seed in (0, 1, 2024, 123456789012345):
            rng = random.Random(seed)
            mt, st = K.mt_from_state(rng.getstate())
            for i in range(20000):
                c = i % 4
                if c == 0:
                    want, got = rng.randrange(3), K._randbelow(mt, st, 3)
                elif c == 1:
                    want, got = rng.randrange(1, 59), 1 + K._randbelow(mt, st, 58)
                elif c == 2:
                    want, got = rng.randrange(23), K._randbelow(mt, st, 23)
                else:
                    want, got = rng.random(), K._rand_random(mt, st)
                self.assertEqual(want, got, f"seed {seed}, draw {i}")
            self.assertEqual(K.state_from_mt(mt, st, rng.getstate()), rng.getstate())

    def test_all_models_lockstep(self):
        for model, params in MODELS.items():
            for N in (12, 30):
                for T, kb in ((280.0, 0.0), (360.0, 0.6)):
                    a, sa, b, sb = _pair(model, params, N, T, kb, 2500)
                    self.assertEqual(a, b, f"{model} N={N} T={T} kappa={kb}: state differs")
                    self.assertEqual(sa, sb, f"{model} N={N} T={T} kappa={kb}: RNG differs")

    def test_n60_scalar_and_local(self):
        for model in ("hs", "saturating_cooperative_contact", "local_coordination_saturation"):
            a, sa, b, sb = _pair(model, MODELS[model], 60, 320.0, 0.0, 3000)
            self.assertEqual((a, sa), (b, sb), model)

    def test_local_cache_not_shared_across_chain_lengths(self):
        # regression: histogram keys use base N+1, so caches are per (q_sat, N)
        model, params = "local_coordination_saturation", MODELS["local_coordination_saturation"]
        for N in (20, 31, 20):
            a, sa, b, sb = _pair(model, params, N, 300.0, 0.0, 1500, seed=N)
            self.assertEqual((a, sa), (b, sb), f"N={N}")

    def test_tight_grid(self):
        # N = 7 -> 2N + 2 = 16 = LB exactly: the smallest grid the aliasing bound allows
        self.assertEqual(1 << K._grid_bits(7), 16)
        for T in (280.0, 360.0):
            a, sa, b, sb = _pair("hs", MODELS["hs"], 7, T, 0.0, 5000, seed=3)
            self.assertEqual((a, sa), (b, sb))

    def test_selfcheck_and_resolution(self):
        info = K.selfcheck(R, "hs", MODELS["hs"], 320.0, 80.0, 0.0, 20, [280.0, 320.0, 360.0],
                           steps=500, warm=300)
        self.assertTrue(info["ok"], info)
        random.seed(99)
        before = random.getstate()
        kernel, kinfo = R.resolve_sampler_kernel("auto", "hs", MODELS["hs"], 320.0, 80.0, 0.0,
                                                 20, np.array([280.0, 360.0]))
        self.assertEqual(kernel, "numba")
        self.assertEqual(random.getstate(), before, "self-check must restore the global RNG")
        self.assertEqual(R.resolve_sampler_kernel("python", "hs", MODELS["hs"], 320.0, 80.0,
                                                  0.0, 20, [300.0])[0], "python")

    def test_run_remd_kernels_and_schemes_agree(self):
        kw = dict(N=16, Ts=np.array([290.0, 320.0, 350.0]), steps_per_swap=150, n_cycles=6,
                  model_name="local_coordination_saturation",
                  params=MODELS["local_coordination_saturation"], Tref=320.0, Tscale=80.0,
                  seed=5, verbose=False, kappa_bend=0.3)

        def run(**extra):
            reps, sp, sa = R.run_remd(**kw, **extra)
            return [r.E_traj + r.C_traj + r.Rg_traj for r in reps], sp.tolist(), sa.tolist()

        py_serial = run(kernel="python")
        self.assertEqual(py_serial, run(kernel="numba"))
        py_lane = run(kernel="python", rng_scheme="per-lane")
        self.assertEqual(py_lane, run(kernel="numba", rng_scheme="per-lane"))
        self.assertEqual(py_lane, run(kernel="numba", n_workers=2))


class FallbackTests(unittest.TestCase):
    def test_auto_falls_back_without_numba(self):
        saved = K.NUMBA_AVAILABLE
        try:
            K.NUMBA_AVAILABLE = False
            kernel, info = R.resolve_sampler_kernel("auto", "hs", MODELS["hs"], 320.0, 80.0,
                                                    0.0, 20, [300.0])
            self.assertEqual(kernel, "python")
            with self.assertRaises(RuntimeError):
                R.resolve_sampler_kernel("numba", "hs", MODELS["hs"], 320.0, 80.0, 0.0, 20, [300.0])
        finally:
            K.NUMBA_AVAILABLE = saved


if K.NUMBA_AVAILABLE:
    from numba import njit as _njit
    from numba.core.errors import TypingError

    @_njit
    def _uncompilable(*args):
        # A genuine Numba compilation failure (array + str): raises TypingError
        # on the first call, the way an unsupported numba/numpy combination would.
        return args[0] + "x"


@unittest.skipUnless(K.NUMBA_AVAILABLE, "numba not installed")
class CompileFailureFallbackTests(unittest.TestCase):
    """--kernel auto must fall back to Python when the Numba sweep fails to compile."""

    ARGS = ("hs", MODELS["hs"], 320.0, 80.0, 0.0, 20, [280.0, 360.0])

    def test_auto_falls_back_on_compilation_error(self):
        random.seed(11)
        before = random.getstate()
        with mock.patch.object(K, "_sweep", _uncompilable):
            with self.assertWarns(RuntimeWarning):
                kernel, info = R.resolve_sampler_kernel("auto", *self.ARGS)
        self.assertEqual(kernel, "python")
        self.assertIn("TypingError", info["reason"])
        check = info["selfcheck"]
        self.assertFalse(check["ok"])
        self.assertEqual(check["error_type"], "TypingError")
        self.assertNotIn("_exception", check)
        self.assertNotIn("\x1b", check["error"])
        json.dumps(info)  # stored in run_summary.json
        self.assertEqual(random.getstate(), before, "self-check must restore the global RNG")

    def test_explicit_numba_raises_with_cause(self):
        with mock.patch.object(K, "_sweep", _uncompilable):
            with self.assertRaises(RuntimeError) as cm:
                R.resolve_sampler_kernel("numba", *self.ARGS)
        self.assertIn("TypingError", str(cm.exception))
        self.assertIn("--kernel python", str(cm.exception))
        self.assertIsInstance(cm.exception.__cause__, TypingError)

    def test_python_reference_errors_propagate(self):
        def broken(*a, **k):
            raise ValueError("python reference failure")
        with mock.patch.object(R, "mc_sweep", broken):
            with self.assertRaisesRegex(ValueError, "python reference failure"):
                R.resolve_sampler_kernel("auto", *self.ARGS)

    def test_run_remd_auto_with_broken_numba_matches_python(self):
        kw = dict(N=16, Ts=np.array([290.0, 330.0]), steps_per_swap=100, n_cycles=4,
                  model_name="hs", params=MODELS["hs"], Tref=320.0, Tscale=80.0,
                  seed=4, verbose=False, kappa_bend=0.2)

        def run(**extra):
            reps, sp, sa = R.run_remd(**kw, **extra)
            return [r.E_traj + r.C_traj + r.Rg_traj for r in reps], sp.tolist(), sa.tolist()

        ref = run(kernel="python")
        with mock.patch.object(K, "_sweep", _uncompilable):
            with self.assertWarns(RuntimeWarning):
                got = run(kernel="auto")
        self.assertEqual(got, ref)


if __name__ == "__main__":
    unittest.main()
