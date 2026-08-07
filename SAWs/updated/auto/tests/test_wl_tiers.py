from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


AUTO_DIR = Path(__file__).resolve().parents[1]
if str(AUTO_DIR) not in sys.path:
    sys.path.insert(0, str(AUTO_DIR))

from single_chain_wang_landau import (  # noqa: E402
    TIER_COVERAGE,
    TIER_EXCLUDED,
    TIER_FLAT,
    WL_STAGE_DTYPE,
    learn_log_density,
    make_contact_tiers,
    parse_args,
    resolve_contact_tiers,
    validate_args,
)


class WangLandauTierTests(unittest.TestCase):
    def test_target_derived_default_tiers(self) -> None:
        args = parse_args(["--N", "44"])
        validate_args(args)
        resolved = resolve_contact_tiers(args)
        tier = resolved["tier"]
        self.assertEqual(args.m_flat, 49)
        self.assertEqual(args.m_cover, 50)
        self.assertTrue(np.all(tier[:50] == TIER_FLAT))
        self.assertEqual(tier[50], TIER_COVERAGE)
        self.assertFalse(resolved["declared_truncation"])

    def test_tier_builder_preserves_verified_internal_gap(self) -> None:
        tier = make_contact_tiers(0, 5, 3, 5, [4])
        np.testing.assert_array_equal(
            tier,
            [TIER_FLAT, TIER_FLAT, TIER_FLAT, TIER_FLAT,
             TIER_EXCLUDED, TIER_COVERAGE],
        )

    def test_one_over_t_uses_cumulative_time_and_structured_records(self) -> None:
        tier = make_contact_tiers(0, 2, 1, 2)
        result = learn_log_density(
            n_beads=6,
            m_min=0,
            m_max=2,
            seed=4,
            initial_log_f=0.02,
            final_log_f=0.005,
            flatness=0.9,
            min_visits=1,
            min_cover_visits=1,
            check_every=200,
            max_steps=100_000,
            max_seconds=30.0,
            max_steps_per_stage=100_000,
            schedule="one_over_t",
            checkpoint_every_seconds=1000.0,
            tier=tier,
            progress=False,
        )
        self.assertTrue(result["one_over_t_mode"])
        self.assertLessEqual(result["next_log_f"], 0.005)
        self.assertEqual(result["stage_records"].dtype, WL_STAGE_DTYPE)
        expected = np.count_nonzero(tier) / result["attempted_steps"]
        self.assertGreaterEqual(result["attempted_steps"], 600)
        self.assertAlmostEqual(result["next_log_f"], expected, places=15)

    def test_stage_cap_reports_slowest_levels_and_checkpoints(self) -> None:
        tier = make_contact_tiers(0, 2, 1, 2)
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "wl_checkpoint.npz"
            with self.assertRaisesRegex(RuntimeError, "Slowest levels"):
                learn_log_density(
                    n_beads=6,
                    m_min=0,
                    m_max=2,
                    seed=7,
                    initial_log_f=1.0,
                    final_log_f=0.1,
                    flatness=0.8,
                    min_visits=1000,
                    min_cover_visits=1000,
                    check_every=10,
                    max_steps=100,
                    max_seconds=30.0,
                    max_steps_per_stage=20,
                    schedule="halving",
                    checkpoint_every_seconds=1e-9,
                    tier=tier,
                    checkpoint=checkpoint,
                    progress=False,
                )
            self.assertTrue(checkpoint.exists())
            with np.load(checkpoint, allow_pickle=False) as saved:
                self.assertEqual(int(saved["checkpoint_version"]), 2)
                np.testing.assert_array_equal(saved["wl_tier"], tier)
                self.assertEqual(saved["wl_stage_records"].dtype, WL_STAGE_DTYPE)


if __name__ == "__main__":
    unittest.main()
