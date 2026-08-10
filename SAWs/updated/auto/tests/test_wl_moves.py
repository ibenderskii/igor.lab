from __future__ import annotations

import math
import random
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


AUTO_DIR = Path(__file__).resolve().parents[1]
if str(AUTO_DIR) not in sys.path:
    sys.path.insert(0, str(AUTO_DIR))

import single_chain_wang_landau as wl  # noqa: E402
from single_chain_wang_landau import (  # noqa: E402
    TIER_EXCLUDED,
    TIER_FLAT,
    attempt_pull_move,
    contact_count,
    contact_delta_from_occupancy,
    enumerate_pull_moves,
    enumerate_rooted_saws,
    learn_log_density,
    reverse_pull_outcome,
    run_production_chain,
    validate_chain,
)


def _rod(n_beads: int):
    chain = [(i, 0, 0) for i in range(n_beads)]
    return chain, set(chain)


def _descriptor(old_chain, new_chain):
    """Descriptor of the move that carries ``old_chain`` to ``new_chain``."""
    return tuple(
        (index, new_chain[index])
        for index in range(len(old_chain))
        if new_chain[index] != old_chain[index]
    )


class PullMoveReversibilityTests(unittest.TestCase):
    """The invariant the pull-move Hastings ratio rests on.

    Lesh-Mitzenmacher-Whitesides pull moves are *not* fully reversible on a
    rectangular lattice: Gyorffy, Zavodszky and Szilagyi (arXiv:1210.0495)
    showed the original reversibility proof is wrong, and this codebase
    reproduces that -- single-bead outcomes always invert, multi-bead outcomes
    only about 60% of the time.  ``attempt_pull_move`` therefore rejects any
    proposal whose inverse is absent, which is what makes its detailed-balance
    argument valid.

    This is the primary test: if the accepted/rejected split does not track
    inverse-presence exactly, the proposal ratio is meaningless.
    """

    def test_accepted_pull_moves_always_have_their_inverse(self) -> None:
        total = irreversible = 0
        for n_beads in (8, 12, 20):
            rng = random.Random(90210 + n_beads)
            chain, occupied = _rod(n_beads)
            for _ in range(3400):
                catalog = enumerate_pull_moves(chain, occupied)
                if catalog:
                    total += 1
                    valid, new_chain, new_occupied, log_q = attempt_pull_move(
                        chain, occupied, rng
                    )
                    if valid:
                        outcome = _descriptor(chain, new_chain)
                        self.assertIn(
                            outcome, set(catalog),
                            "accepted outcome is not a member of the forward catalog",
                        )
                        reverse_catalog = enumerate_pull_moves(new_chain, new_occupied)
                        self.assertIn(
                            reverse_pull_outcome(chain, outcome),
                            set(reverse_catalog),
                            "accepted pull move has no inverse; detailed balance "
                            "would be violated",
                        )
                        # The ratio must be exactly the two catalog sizes.
                        self.assertAlmostEqual(
                            log_q,
                            math.log(len(catalog)) - math.log(len(reverse_catalog)),
                            places=12,
                        )
                        chain, occupied = new_chain, new_occupied
                    else:
                        irreversible += 1
                other = rng.choice(wl.MOVE_FUNCS)(chain, occupied, rng)
                if other[0]:
                    chain, occupied = other[1], other[2]

        self.assertGreater(total, 9000, "too few pull proposals to be conclusive")
        # Recorded so a future move-set change that accidentally fixes or worsens
        # reversibility is visible rather than silent.  Measured 35-55%.
        self.assertGreater(
            irreversible, 0,
            "no irreversible proposals seen at all; the rejection path is "
            "untested and may have been removed",
        )
        self.assertLess(irreversible, total, "every proposal was rejected")


class PullMoveDetailedBalanceTests(unittest.TestCase):
    """Frozen-weight sampling must reproduce an exactly enumerable distribution.

    With a fixed non-flat ``log_g`` the invariant distribution of the
    fixed-weight chain is ``P(m) proportional to g(m) exp(-log_g[m])``, and
    ``g(m)`` is known exactly for N=6 from the 3,534 rooted SAWs.  This is the
    end-to-end check that the pull-move proposal ratio is right.

    Without the inverse-presence rejection in ``attempt_pull_move`` this test
    measures TVD = 0.219 rather than 0.002 -- P(0) comes out at 0.49 against an
    exact 0.712 -- so it genuinely discriminates.
    """

    LOG_G = np.array([0.0, 1.7, -0.9])
    STEPS = 400_000

    def _exact(self) -> np.ndarray:
        values, prob, _ = enumerate_rooted_saws(6)
        counts = np.zeros(3)
        counts[values] = prob
        exact = counts * np.exp(-self.LOG_G)
        return exact / exact.sum()

    def _sampled(self, pull_move_weight: float) -> np.ndarray:
        result = run_production_chain(
            worker_id=0,
            seed=20260809,
            initial_chain=[(i, 0, 0) for i in range(6)],
            log_g=self.LOG_G,
            m_min=0,
            m_max=2,
            steps=self.STEPS,
            burnin=0.1,
            sample_every=5,
            progress=False,
            pull_move_weight=pull_move_weight,
        )
        counts = np.bincount(result["contact_samples"], minlength=3).astype(float)
        return counts / counts.sum()

    def test_pull_moves_preserve_the_frozen_weight_distribution(self) -> None:
        exact = self._exact()
        # 0.5 rather than 1.0 so the test does not depend on the pull-move kernel
        # alone being irreducible on a six-bead chain.
        observed = self._sampled(0.5)
        tvd = 0.5 * float(np.abs(observed - exact).sum())
        self.assertLess(
            tvd, 0.02,
            f"pull-move chain does not target the frozen-weight distribution: "
            f"TVD={tvd:.4f}, observed={observed}, exact={exact}",
        )

        # Control: the same chain with pull moves switched off.  If this arm
        # also drifted, the fault would be in the harness, not the pull moves.
        control = self._sampled(0.0)
        control_tvd = 0.5 * float(np.abs(control - exact).sum())
        self.assertLess(
            control_tvd, 0.02, f"control chain is biased: TVD={control_tvd:.4f}"
        )


class PullMoveChainValidityTests(unittest.TestCase):
    """Every accepted pull move must leave a legal chain and an exact contact count.

    ``contact_delta_from_occupancy`` is documented to handle an arbitrary
    symmetric difference between occupancy sets.  A pull move is the first
    caller that actually produces a multi-bead difference, so this verifies that
    claim rather than assuming it.
    """

    def test_validity_and_incremental_contact_delta(self) -> None:
        n_beads = 20
        rng = random.Random(5150)
        chain, occupied = _rod(n_beads)
        contact = contact_count(chain, occupied)
        accepted = 0
        multi_bead = 0
        while accepted < 20_000:
            valid, new_chain, new_occupied, _ = attempt_pull_move(chain, occupied, rng)
            if not valid:
                other = rng.choice(wl.MOVE_FUNCS)(chain, occupied, rng)
                if other[0]:
                    chain, occupied = other[1], other[2]
                    contact = contact_count(chain, occupied)
                continue
            validate_chain(new_chain)
            self.assertEqual(
                new_occupied, set(new_chain),
                "occupancy set disagrees with the chain it should mirror",
            )
            self.assertEqual(len(new_chain), n_beads)
            delta = contact_delta_from_occupancy(occupied, new_occupied)
            self.assertEqual(
                contact + delta,
                contact_count(new_chain, new_occupied),
                "incremental contact delta disagrees with a full recount",
            )
            if len(_descriptor(chain, new_chain)) > 2:
                multi_bead += 1
            contact += delta
            chain, occupied = new_chain, new_occupied
            accepted += 1
        self.assertGreater(
            multi_bead, 0,
            "no multi-bead pulls were exercised, so the multi-site occupancy "
            "difference this test exists to check was never produced",
        )


class PullMoveWeightZeroTests(unittest.TestCase):
    """Weight 0.0 must reproduce the pre-pull-move sampler exactly.

    Two assertions, because the golden array is only a fingerprint.  The first
    states the property directly; the second pins the RNG stream so an unrelated
    change to move selection cannot slip through unnoticed.
    """

    # Regenerate with, from SAWs/updated/auto:
    #
    #   python -c "
    #   import numpy as np, single_chain_wang_landau as wl
    #   gap_values, _, _ = wl.enumerate_rooted_saws(8)
    #   gap_tier = np.where(np.isin(np.arange(6), gap_values), wl.TIER_FLAT,
    #                       wl.TIER_EXCLUDED).astype(np.int8)
    #   r = wl.learn_log_density(n_beads=8, m_min=0, m_max=5, seed=606,
    #       initial_log_f=1.0, final_log_f=0.01, flatness=0.8, min_visits=100,
    #       check_every=2000, max_steps=2_000_000, tier=gap_tier, progress=False,
    #       pull_move_weight=0.0)
    #   print([repr(float(x)) for x in r['log_g']])"
    #
    # Generated at b557cb7, and verified byte-identical at 0240b9a -- the commit
    # before pull moves existed.  N=8 has a verified internal gap at m=4, which
    # is why that level is excluded and why its entry is a large negative number.
    GOLDEN_LOG_G = np.array([
        0.0, -0.6875, -1.40625, -2.40625, -808.046875, -4.484375,
    ])

    def _gapped_tier(self) -> np.ndarray:
        gap_values, _, _ = enumerate_rooted_saws(8)
        return np.where(
            np.isin(np.arange(6), gap_values), TIER_FLAT, TIER_EXCLUDED
        ).astype(np.int8)

    def _learn(self, pull_move_weight: float):
        return learn_log_density(
            n_beads=8, m_min=0, m_max=5, seed=606, initial_log_f=1.0,
            final_log_f=0.01, flatness=0.8, min_visits=100, check_every=2000,
            max_steps=2_000_000, tier=self._gapped_tier(), progress=False,
            pull_move_weight=pull_move_weight,
        )

    def test_weight_zero_is_a_no_op(self) -> None:
        calls = []
        real = wl.attempt_pull_move

        def counting(*args, **kwargs):
            calls.append(1)
            return real(*args, **kwargs)

        # The property: no pull move is even proposed at weight 0.0.
        with mock.patch.object(wl, "attempt_pull_move", counting):
            learned = self._learn(0.0)
        self.assertEqual(
            len(calls), 0,
            "attempt_pull_move was called despite pull_move_weight=0.0; the "
            "move-selection short circuit is broken",
        )
        # The shim must be able to observe a real call, or the assertion above
        # would pass for the wrong reason.  N=8 is small, so this is cheap.
        with mock.patch.object(wl, "attempt_pull_move", counting):
            learn_log_density(
                n_beads=8, m_min=0, m_max=5, seed=606, initial_log_f=1.0,
                final_log_f=0.5, flatness=0.8, min_visits=10, check_every=500,
                max_steps=2_000, tier=self._gapped_tier(), progress=False,
                pull_move_weight=1.0,
            )
        self.assertGreater(len(calls), 0, "the counting shim never intercepts")

        # The fingerprint: the exact RNG stream, not merely the same move mix.
        np.testing.assert_array_equal(
            learned["log_g"],
            self.GOLDEN_LOG_G,
            err_msg=(
                "this test pins the RNG stream of learn_log_density; if you "
                "intentionally changed move selection or rng consumption, "
                "regenerate with the command in the comment above GOLDEN_LOG_G "
                "and note why in the commit message"
            ),
        )


if __name__ == "__main__":
    unittest.main()
