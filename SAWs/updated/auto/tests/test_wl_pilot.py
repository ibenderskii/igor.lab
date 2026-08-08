from __future__ import annotations

import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


AUTO_DIR = Path(__file__).resolve().parents[1]
if str(AUTO_DIR) not in sys.path:
    sys.path.insert(0, str(AUTO_DIR))

from run_wl_pilot import (  # noqa: E402
    AcceptanceFailure,
    _dry_run,
    _rebin_mass,
    _upper_tail_metrics,
    _validate_wl_output,
    parse_args,
)
from single_chain_wang_landau import WL_STAGE_DTYPE  # noqa: E402


class WangLandauPilotTests(unittest.TestCase):
    def test_requested_chains_are_put_in_gate_order(self) -> None:
        args = parse_args(["--dry_run", "--chains", "60", "30"])
        self.assertEqual(args.chains, [30, 60])

    def test_upper_tail_regression(self) -> None:
        metrics = _upper_tail_metrics(
            AUTO_DIR / "remd_distributions_44mer.npz", 43, 33
        )
        self.assertAlmostEqual(metrics["mean"], 0.017307, delta=3e-5)
        self.assertAlmostEqual(metrics["max"], 0.047380, delta=3e-5)

    def test_piecewise_rebin_preserves_mass(self) -> None:
        rebinned = _rebin_mass(
            np.array([0.0, 1.0, 2.0]),
            np.array([0.25, 0.75]),
            np.array([0.0, 0.5, 1.5, 2.0]),
        )
        self.assertAlmostEqual(float(rebinned.sum()), 1.0, places=15)
        np.testing.assert_allclose(rebinned, [0.125, 0.5, 0.375])

    def test_output_gates_and_pull_move_trigger(self) -> None:
        stage = np.zeros(1, dtype=WL_STAGE_DTYPE)
        stage["highest_m"] = 1
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "baseline.npz"
            np.savez_compressed(
                path,
                wl_tier=np.array([2, 1], dtype=np.int8),
                production_samples_per_level=np.array([1000, 50]),
                wl_min_visits=1000,
                wl_min_cover_visits=50,
                production_round_trips_per_worker=np.array([1]),
                rg_out_of_range_count=0,
                joint_contact_marginal_error=1e-14,
                joint_rg_marginal_error=1e-14,
                m_cover=1,
                wl_stage_records=stage,
                production_max_contact=1,
                importance_effective_sample_size=500.0,
                wl_learning_steps=100,
                wl_learning_wall_seconds=1.0,
            )
            summary = _validate_wl_output(path)
            self.assertTrue(all(summary["gates"].values()))
            self.assertEqual(summary["learning_steps_per_second"], 100.0)
            self.assertEqual(summary["production_round_trips_per_worker"].tolist(), [1])
            self.assertEqual(summary["stages"][0]["highest_m"], 1)

            stage["highest_m"] = 0
            np.savez_compressed(
                path,
                wl_tier=np.array([2, 1], dtype=np.int8),
                production_samples_per_level=np.array([1000, 50]),
                wl_min_visits=1000,
                wl_min_cover_visits=50,
                production_round_trips_per_worker=np.array([1]),
                rg_out_of_range_count=0,
                joint_contact_marginal_error=1e-14,
                joint_rg_marginal_error=1e-14,
                m_cover=1,
                wl_stage_records=stage,
                production_max_contact=1,
                importance_effective_sample_size=500.0,
                wl_learning_steps=100,
                wl_learning_wall_seconds=1.0,
            )
            with self.assertRaisesRegex(AcceptanceFailure, "evidence gate for W8"):
                _validate_wl_output(path)

    def test_dry_run_reports_both_caps_and_flags_the_conflict(self) -> None:
        """The step and time caps must be reported separately.

        The single "estimated WL wall cap" line silently added production time
        to the learning budget, reporting 8.92 h for N=44 while
        --wl_max_seconds was 6.0 h, and hid the fact that --wl_max_steps is an
        order of magnitude beyond anything the time cap permits.
        """
        with tempfile.TemporaryDirectory() as temporary:
            args = parse_args([
                "--dry_run", "--chains", "44", "--models", "hs",
                "--skip_bootstrap", "--outdir", str(Path(temporary) / "out"),
            ])
            captured = io.StringIO()
            with contextlib.redirect_stdout(captured):
                self.assertEqual(_dry_run(args), 0)
            text = captured.getvalue()

        self.assertIn("WL step cap: --wl_max_steps=1e+10 = 72.91 h", text)
        self.assertIn("WL time cap: --wl_max_seconds=21600 = 6.00 h", text)
        self.assertIn("binding WL cap: --wl_max_seconds", text)
        self.assertIn("CONFLICT", text)
        self.assertIn("12.2x", text)
        # The misleading conflated figure must be gone.
        self.assertNotIn("estimated WL wall cap", text)

    def test_dry_run_creates_no_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            outdir = Path(temporary) / "not_created"
            args = parse_args([
                "--dry_run", "--chains", "30", "--models", "hs",
                "--skip_bootstrap", "--outdir", str(outdir),
            ])
            with contextlib.redirect_stdout(io.StringIO()):
                result = _dry_run(args)
            self.assertEqual(result, 0)
            self.assertFalse(outdir.exists())


if __name__ == "__main__":
    unittest.main()
