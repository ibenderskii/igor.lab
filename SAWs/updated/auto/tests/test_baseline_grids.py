from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np


AUTO_DIR = Path(__file__).resolve().parents[1]
if str(AUTO_DIR) not in sys.path:
    sys.path.insert(0, str(AUTO_DIR))

from baseline_grids import (  # noqa: E402
    assert_within_grid,
    fixed_c_edges,
    fixed_rg_edges,
    legacy_rg_grid,
    min_compact_rg,
    rod_rg,
)
from single_chain_wang_landau import (  # noqa: E402
    blocked_contact_stderr,
    build_distributions,
)


class BaselineGridTests(unittest.TestCase):
    def test_measured_geometric_ranges_and_legacy_subset(self) -> None:
        expected_min = {30: 1.4787, 44: 1.6824, 60: 1.8753}
        for n_beads in (30, 44, 60):
            with self.subTest(n_beads=n_beads):
                compact = min_compact_rg(n_beads)
                self.assertAlmostEqual(compact, expected_min[n_beads], places=4)
                self.assertAlmostEqual(
                    rod_rg(n_beads),
                    math.sqrt((n_beads * n_beads - 1.0) / 12.0),
                    places=15,
                )
                legacy, _, _ = legacy_rg_grid(n_beads, AUTO_DIR)
                emitted = fixed_rg_edges(n_beads, AUTO_DIR)
                start = np.flatnonzero(np.isclose(emitted, legacy[0], atol=1e-12))[0]
                np.testing.assert_allclose(
                    emitted[start:start + legacy.size],
                    legacy,
                    rtol=0.0,
                    atol=1e-12,
                )
                self.assertLessEqual(emitted[0], compact)
                self.assertGreaterEqual(emitted[-1], rod_rg(n_beads))

    def test_contact_edges_span_declared_window(self) -> None:
        np.testing.assert_array_equal(fixed_c_edges(0, 5), np.arange(-0.5, 6.5))

    def test_out_of_range_guard(self) -> None:
        self.assertEqual(
            assert_within_grid(np.array([0.0, 0.5, 1.0]), np.array([0.0, 1.0]), "x"),
            0,
        )
        with self.assertRaisesRegex(ValueError, "samples fall outside"):
            assert_within_grid(np.array([-0.01, 0.5]), np.array([0.0, 1.0]), "x")

    def test_distribution_keeps_zero_mass_contact_levels(self) -> None:
        contacts = np.array([0, 0, 2, 2], dtype=np.int64)
        radii = np.array([2.0, 2.1, 2.2, 2.3])
        bends = np.array([1, 2, 1, 2], dtype=np.int64)
        built = build_distributions(
            contacts,
            radii,
            bends,
            np.zeros(51),
            0,
            60,
            False,
            n_beads=44,
            m_cover=4,
            grid_search_dir=AUTO_DIR,
        )
        np.testing.assert_array_equal(built["c_vals"], np.arange(5))
        np.testing.assert_array_equal(built["c_edges"], np.arange(-0.5, 5.5))
        np.testing.assert_array_equal(built["c_prob"][[1, 3, 4]], 0.0)
        self.assertEqual(built["rg_grid_source"], "legacy_extended")
        self.assertEqual(built["rg_out_of_range_count"], 0)
        self.assertLess(built["marginal_m_error"], 1e-12)
        self.assertLess(built["marginal_rg_error"], 1e-12)

    def test_distribution_rejects_truncated_rg_grid(self) -> None:
        with self.assertRaisesRegex(ValueError, "Rg samples fall outside"):
            build_distributions(
                np.array([0, 1]),
                np.array([1.0, 2.0]),
                np.array([0, 1]),
                np.zeros(2),
                0,
                2,
                False,
                n_beads=6,
                m_cover=1,
                rg_edges=np.array([1.1, 2.1]),
                grid_source="test",
            )

    def test_batch_means_stderr_has_declared_support(self) -> None:
        stderr, n_batches = blocked_contact_stderr(
            [
                np.array([0, 0, 1, 1, 2, 2, 2, 1]),
                np.array([0, 1, 0, 2, 1, 2, 0, 2]),
            ],
            np.array([0.0, 0.2, 0.5]),
            0,
            2,
            2,
        )
        self.assertEqual(stderr.shape, (3,))
        self.assertEqual(n_batches, 4)
        self.assertTrue(np.all(stderr > 0.0))


if __name__ == "__main__":
    unittest.main()
