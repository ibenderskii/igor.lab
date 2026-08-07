from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


AUTO_DIR = Path(__file__).resolve().parents[1]
if str(AUTO_DIR) not in sys.path:
    sys.path.insert(0, str(AUTO_DIR))

from target_support import (  # noqa: E402
    flat_level,
    load_target_contact_support,
    support_report,
)


CASES = {
    30: {
        "offset": 29,
        "m_max": 30,
        "levels": {1e-2: 20, 1e-3: 25, 1e-4: 28},
        "unsupported": (0.0, 0.0),
        "exceeds": False,
    },
    44: {
        "offset": 43,
        "m_max": 50,
        "levels": {1e-2: 39, 1e-3: 49, 1e-4: 50},
        "unsupported": (0.000025, 0.000730),
        "exceeds": True,
    },
    60: {
        "offset": 59,
        "m_max": 74,
        "levels": {1e-2: 66, 1e-3: 74, 1e-4: 74},
        "unsupported": (0.000147, 0.001229),
        "exceeds": True,
    },
}


class TargetSupportTests(unittest.TestCase):
    def test_real_target_support_regression(self) -> None:
        for n_beads, case in CASES.items():
            with self.subTest(n_beads=n_beads):
                support = load_target_contact_support(
                    AUTO_DIR / f"remd_distributions_{n_beads}mer.npz",
                    case["offset"],
                )
                report = support_report(support, case["m_max"])

                for threshold, expected in case["levels"].items():
                    self.assertEqual(
                        flat_level(support, threshold, case["m_max"]), expected
                    )
                    self.assertEqual(
                        report["m_flat_by_threshold"][threshold], expected
                    )

                expected_mean, expected_max = case["unsupported"]
                observed = report["unsupported_mass_at_m_max"]
                if expected_mean == 0.0:
                    self.assertEqual(observed["mean"], 0.0)
                    self.assertEqual(observed["max"], 0.0)
                else:
                    self.assertAlmostEqual(
                        observed["mean"], expected_mean,
                        delta=0.03 * expected_mean,
                    )
                    self.assertAlmostEqual(
                        observed["max"], expected_max,
                        delta=0.03 * expected_max,
                    )
                self.assertIs(
                    report["target_support_exceeds_m_max"], case["exceeds"]
                )

    def test_rows_are_normalized(self) -> None:
        support = load_target_contact_support(
            AUTO_DIR / "remd_distributions_30mer.npz", 29
        )
        np.testing.assert_allclose(
            support.P.sum(axis=1), 1.0, rtol=0.0, atol=1e-14
        )

    def test_report_marks_geometric_clamping(self) -> None:
        support = load_target_contact_support(
            AUTO_DIR / "remd_distributions_44mer.npz", 43
        )
        report = support_report(support, 50)
        self.assertEqual(report["m_flat_by_threshold"][1e-4], 50)
        self.assertIs(report["flat_threshold_reached"][1e-4], False)


if __name__ == "__main__":
    unittest.main()
