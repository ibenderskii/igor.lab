from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


AUTO_DIR = Path(__file__).resolve().parents[1]
if str(AUTO_DIR) not in sys.path:
    sys.path.insert(0, str(AUTO_DIR))

from single_chain_wang_landau import (  # noqa: E402
    COMPACT_SEED_BOXES,
    NN_VECS,
    compact_seed_chain,
    contact_count,
    geometric_contact_maximum,
    learn_log_density,
    make_contact_tiers,
    sub,
    validate_chain,
)


class CompactSeedTests(unittest.TestCase):
    """The compact initializer must start at the exact geometric maximum.

    That is the whole point of it: from the rod, N=60 learning plateaus below
    the top of its window and stage 1 never completes.  A seed that merely
    looks compact would move the plateau rather than remove it, so each seed is
    checked against the independently verified maximum, not against itself.
    """

    def test_seed_is_a_maximally_compact_saw(self) -> None:
        for n_beads in (30, 44, 60):
            with self.subTest(N=n_beads):
                chain = compact_seed_chain(n_beads)
                self.assertEqual(len(chain), n_beads)
                self.assertEqual(
                    len(set(chain)), n_beads, "compact seed revisits a site"
                )
                validate_chain(chain)
                for first, second in zip(chain[:-1], chain[1:]):
                    self.assertIn(
                        sub(second, first), NN_VECS,
                        "consecutive seed beads are not a unit lattice step",
                    )
                self.assertEqual(
                    contact_count(chain, set(chain)),
                    geometric_contact_maximum(n_beads),
                    "compact seed does not attain the verified geometric maximum",
                )

    def test_seed_fits_inside_its_declared_box(self) -> None:
        for n_beads, box in COMPACT_SEED_BOXES.items():
            with self.subTest(N=n_beads):
                sites = np.asarray(compact_seed_chain(n_beads), dtype=np.int64)
                extent = sites.max(axis=0) - sites.min(axis=0) + 1
                self.assertTrue(
                    np.all(extent <= np.asarray(box)),
                    f"seed for N={n_beads} escapes its declared box {box}",
                )

    def test_unencoded_chain_length_is_refused(self) -> None:
        # No box is claimed for N=31, and one must never be guessed: an
        # unverified box would seed the learner outside the true maximum.
        with self.assertRaises(ValueError):
            compact_seed_chain(31)

    def _highest_m_on_a_starved_stage(self, init: str) -> str:
        """Run a stage far too short to complete and return its failure text.

        The message reports the highest contact level the stage reached, which
        is exactly the quantity P1 exists to change.  Deliberately starving the
        stage keeps this to a fraction of a second; a completed N=30 stage costs
        about ten.
        """
        tier = make_contact_tiers(0, 30, 30, 30)
        with self.assertRaises(RuntimeError) as raised:
            learn_log_density(
                n_beads=30, m_min=0, m_max=30, seed=99, initial_log_f=1.0,
                final_log_f=0.5, flatness=0.8, min_visits=1000,
                min_cover_visits=1, check_every=1_000, max_steps=5_000,
                tier=tier, progress=False, init=init,
            )
        return str(raised.exception)

    def test_compact_initializer_starts_at_the_top_of_the_window(self) -> None:
        self.assertIn(
            "Highest m reached this stage: 30 (first reached at stage step 0)",
            self._highest_m_on_a_starved_stage("compact"),
            "the compact seed did not start the learner at the geometric maximum",
        )
        self.assertNotIn(
            "Highest m reached this stage: 30",
            self._highest_m_on_a_starved_stage("rod"),
            "the rod start reached m_max on a starved stage, so this test no "
            "longer discriminates between the two initializers",
        )

    def test_seed_above_a_lowered_ceiling_is_refused(self) -> None:
        # Reachable with a user-supplied --m_max below the geometric maximum, or
        # with a declared truncation.  Starting out of window must fail loudly,
        # and must say which initializer put the chain there.
        with self.assertRaisesRegex(ValueError, "compact initializer"):
            learn_log_density(
                n_beads=30, m_min=0, m_max=25, seed=99, initial_log_f=1.0,
                final_log_f=0.5, flatness=0.8, min_visits=1, check_every=100,
                max_steps=1_000, tier=make_contact_tiers(0, 25, 25, 25),
                progress=False, init="compact",
            )

    def test_unknown_initializer_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            learn_log_density(
                n_beads=6, m_min=0, m_max=2, seed=1, initial_log_f=1.0,
                final_log_f=0.5, flatness=0.5, min_visits=1, check_every=100,
                max_steps=100, progress=False, init="spiral",
            )


if __name__ == "__main__":
    unittest.main()
