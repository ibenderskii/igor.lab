"""Self-tests for the validation harness: it must detect planted differences.

Run:  python -m pytest SAWs/updated/validation -q      (or: python test_validation_harness.py)
"""
from __future__ import annotations

import json
import random
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import compare_outputs as co  # noqa: E402
import lockstep as ls  # noqa: E402
import make_inputs as mi  # noqa: E402


def _diffs(a, b, extra_ignores=(), normalize=()):
    rep = co.compare_trees(Path(a), Path(b), list(co.DEFAULT_IGNORE_KEYS) + list(extra_ignores),
                           list(normalize))
    return rep


class CompareOutputsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.a, self.b = self.tmp / "a", self.tmp / "b"
        self.a.mkdir()
        x = np.linspace(0.0, 1.0, 7)
        np.savez(self.a / "d.npz", x=x, k=np.arange(5), nanv=np.array([np.nan, 1.0]),
                 path=np.array(str(self.a / "conf.h5")), timing=np.array([1.5]))
        (self.a / "s.json").write_text(json.dumps(
            {"value": 0.1, "timestamp": "t0", "nested": {"wall_time_seconds": 3.0, "p": [1, 2]},
             "execution": {"workers": 1, "mode": "serial"}, "out": str(self.a)}))
        (self.a / "r.csv").write_text("T,C_mean,elapsed_s\n280,1.25,0.3\n300,1.5,0.4\n")
        shutil.copytree(self.a, self.b)

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def _rewrite_npz(self, **changes):
        with np.load(self.b / "d.npz", allow_pickle=True) as d:
            arrays = {k: d[k] for k in d.files}
        arrays.update(changes)
        np.savez(self.b / "d.npz", **arrays)

    def test_identical(self):
        self.assertEqual(_diffs(self.a, self.b).diffs, {})

    def test_one_ulp_detected(self):
        with np.load(self.a / "d.npz") as d:
            x = d["x"].copy()
        x[3] = np.nextafter(x[3], 2.0)
        self._rewrite_npz(x=x)
        self.assertIn("d.npz", _diffs(self.a, self.b).diffs)

    def test_signed_zero_detected(self):
        with np.load(self.a / "d.npz") as d:
            x = d["x"].copy()
        x[0] = -0.0
        self._rewrite_npz(x=x)
        self.assertIn("d.npz", _diffs(self.a, self.b).diffs)

    def test_nan_equal_and_dtype_strict(self):
        self.assertEqual(_diffs(self.a, self.b).diffs, {})
        self._rewrite_npz(k=np.arange(5, dtype=np.int32))
        self.assertIn("d.npz", _diffs(self.a, self.b).diffs)

    def test_timing_ignored_but_science_detected(self):
        s = json.loads((self.b / "s.json").read_text())
        s["timestamp"] = "t1"
        s["nested"]["wall_time_seconds"] = 9.0
        (self.b / "s.json").write_text(json.dumps(s))
        self._rewrite_npz(timing=np.array([2.5]))
        rep = _diffs(self.a, self.b)
        self.assertEqual(rep.diffs, {})
        self.assertIn("s.json", rep.ignored)
        s["nested"]["p"] = [1, 3]
        (self.b / "s.json").write_text(json.dumps(s))
        self.assertIn("s.json", _diffs(self.a, self.b).diffs)

    def test_csv_cell_and_ignored_column(self):
        (self.b / "r.csv").write_text("T,C_mean,elapsed_s\n280,1.25,9.9\n300,1.5,0.4\n")
        self.assertEqual(_diffs(self.a, self.b).diffs, {})
        (self.b / "r.csv").write_text("T,C_mean,elapsed_s\n280,1.2500000000000002,0.3\n300,1.5,0.4\n")
        self.assertIn("r.csv", _diffs(self.a, self.b).diffs)

    def test_paths_normalized_across_widths(self):
        long_b = self.tmp / "b_longer_name"
        self.b.rename(long_b)
        self.b = long_b
        # outputs that embed their own (longer) root path
        self._rewrite_npz(path=np.array(str(long_b / "conf.h5")))
        s = json.loads((long_b / "s.json").read_text())
        s["out"] = str(long_b)
        (long_b / "s.json").write_text(json.dumps(s))
        self.assertTrue(_diffs(self.a, long_b).diffs)          # raw: differ
        rep = co.compare_trees(self.a, long_b, co.DEFAULT_IGNORE_KEYS,
                               [(str(self.a), "<R>"), (str(long_b), "<R>")])
        self.assertEqual(rep.diffs, {})                         # normalized: equal

    def test_extra_and_missing_files(self):
        (self.b / "extra.json").write_text("{}")
        self.assertIn("extra.json", _diffs(self.a, self.b).diffs)

    def test_full_path_ignore(self):
        s = json.loads((self.b / "s.json").read_text())
        s["execution"] = {"workers": 2, "mode": "fork"}
        (self.b / "s.json").write_text(json.dumps(s))
        self.assertIn("s.json", _diffs(self.a, self.b).diffs)
        self.assertEqual(_diffs(self.a, self.b, extra_ignores=["execution"]).diffs, {})

    def test_hdf5(self):
        try:
            import h5py
        except ImportError:
            self.skipTest("h5py not available")
        for root, val in ((self.a, 1.0), (self.b, 1.0)):
            with h5py.File(root / "c.h5", "w") as f:
                f.attrs["start_time"] = "x" if root is self.a else "y"
                f.create_dataset("g/coords", data=np.arange(6.0).reshape(2, 3) * val)
        self.assertEqual(_diffs(self.a, self.b).diffs, {})
        with h5py.File(self.b / "c.h5", "a") as f:
            f["g/coords"][0, 0] = 1e-300
        self.assertIn("c.h5", _diffs(self.a, self.b).diffs)


class MakeInputsTests(unittest.TestCase):
    def test_exact_saw_counts(self):
        # c_n for the simple cubic lattice: c_5 = 3534, c_9 = 1853886; first bond fixed -> /6
        m, h, r = mi.enumerate_saws(6)
        self.assertEqual(m.size, 3534 // 6)
        base, info = mi.build_baseline(7, 8)
        self.assertEqual(info["n_walks"], 16926 // 6)
        hist = base["local_coord_histograms"]
        self.assertTrue(np.all(hist.sum(axis=1) == 7))
        self.assertTrue(np.array_equal(hist @ np.arange(7), 2 * base["local_coord_contact_counts"]))
        self.assertAlmostEqual(float(base["local_coord_state_mass"].sum()), 1.0, places=12)
        self.assertTrue(np.allclose(np.bincount(base["local_coord_contact_counts"],
                                                weights=base["local_coord_state_mass"]),
                                    base["c_prob"], atol=1e-14))


class LockstepTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(HERE.parent / "auto"))
        import remd_uniform_chain_2_new as R  # noqa: E402
        cls.R = R
        cls.args = ("hs", [415.59484, 1.582392], 320.0, 80.0)

    def _start(self):
        return ls.single_chain_start(self.R, 20, 320.0, *self.args, seed=3, burn_steps=2000)

    def test_identical_kernels(self):
        k = ls.single_chain_python_kernel(self.R, *self.args, n_beads=20)
        res = ls.run_lockstep(self._start(), k, k, ls.single_chain_snapshot, 4, 500)
        self.assertTrue(res.ok, str(res))

    def test_planted_rng_skew_detected(self):
        k = ls.single_chain_python_kernel(self.R, *self.args, n_beads=20)

        def skewed(state, rng, steps):
            rng.random()            # one extra draw: every later decision shifts
            k(state, rng, steps)

        res = ls.run_lockstep(self._start(), k, skewed, ls.single_chain_snapshot, 4, 500)
        self.assertFalse(res.ok)
        self.assertEqual(res.first_divergent_chunk, 0)

    def test_global_rng_untouched(self):
        random.seed(12345)
        before = random.getstate()
        k = ls.single_chain_python_kernel(self.R, *self.args, n_beads=20)
        ls.run_lockstep(self._start(), k, k, ls.single_chain_snapshot, 1, 200)
        self.assertEqual(random.getstate(), before)


if __name__ == "__main__":
    unittest.main()
