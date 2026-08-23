from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


AUTO_DIR = Path(__file__).resolve().parents[1]
if str(AUTO_DIR) not in sys.path:
    sys.path.insert(0, str(AUTO_DIR))

from single_chain_wang_landau import (  # noqa: E402
    build_distributions,
    contact_count,
    coordination_histogram,
    make_contact_tiers,
    run_production_chain,
)


class CoordinationHistogramTests(unittest.TestCase):
    def test_known_single_contact_chain(self) -> None:
        chain = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
        occupied = set(chain)
        self.assertEqual(contact_count(chain, occupied), 1)
        np.testing.assert_array_equal(
            coordination_histogram(chain, occupied),
            np.array([2, 2, 0, 0, 0, 0, 0]),
        )

    def test_production_samples_and_sparse_marginals_are_consistent(self) -> None:
        n_beads = 8
        m_max = 5
        result = run_production_chain(
            worker_id=0,
            seed=731,
            initial_chain=[(i, 0, 0) for i in range(n_beads)],
            log_g=np.zeros(m_max + 1),
            m_min=0,
            m_max=m_max,
            steps=2_000,
            burnin=0.0,
            sample_every=20,
            progress=False,
            tier=make_contact_tiers(0, m_max, m_max, m_max),
            pull_move_weight=0.25,
        )
        contacts = result["contact_samples"]
        radii = result["rg_samples"]
        bends = result["bend_samples"]
        hist = result["coordination_histogram_samples"]
        self.assertEqual(hist.shape, (contacts.size, 7))
        np.testing.assert_array_equal(hist.sum(axis=1), n_beads)
        np.testing.assert_array_equal(hist @ np.arange(7), 2 * contacts)

        built = build_distributions(
            contacts,
            radii,
            bends,
            np.zeros(m_max + 1),
            0,
            12,
            False,
            n_beads=n_beads,
            m_cover=m_max,
            rg_edges=np.linspace(0.0, 3.0, 13),
            grid_source="unit_test",
            coordination_histograms=hist,
        )
        self.assertEqual(int(built["local_coord_schema_version"]), 1)
        self.assertLessEqual(built["local_coord_contact_marginal_error"], 1e-12)
        self.assertLessEqual(built["local_coord_state_marginal_error"], 1e-12)
        self.assertLessEqual(built["local_coord_rg_marginal_error"], 1e-12)
        self.assertAlmostEqual(float(built["local_coord_state_mass"].sum()), 1.0)
        self.assertAlmostEqual(float(built["local_coord_rg_joint_mass"].sum()), 1.0)

    def test_malformed_degree_identity_is_rejected(self) -> None:
        contacts = np.array([1], dtype=np.int64)
        with self.assertRaisesRegex(RuntimeError, "sum_k"):
            build_distributions(
                contacts,
                np.array([1.0]),
                np.array([0]),
                np.zeros(2),
                0,
                2,
                True,
                n_beads=4,
                m_cover=1,
                rg_edges=np.array([0.0, 2.0]),
                grid_source="unit_test",
                coordination_histograms=np.array([[4, 0, 0, 0, 0, 0, 0]]),
            )


if __name__ == "__main__":
    unittest.main()
