from __future__ import annotations

import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock

import numpy as np


AUTO_DIR = Path(__file__).resolve().parents[1]
if str(AUTO_DIR) not in sys.path:
    sys.path.insert(0, str(AUTO_DIR))

import single_chain_wang_landau as wl  # noqa: E402
from single_chain_wang_landau import (  # noqa: E402
    TIER_COVERAGE,
    TIER_EXCLUDED,
    TIER_FLAT,
    WL_STAGE_DTYPE,
    _save_checkpoint,
    build_distributions,
    enumerate_rooted_saws,
    learn_log_density,
    make_contact_tiers,
    one_over_t_trigger,
    parse_args,
    resolve_contact_tiers,
    run_production_chain,
    validate_args,
)


def _quiet(function, *args, **kwargs):
    """Run ``function`` with warnings suppressed, for calls not under test."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return function(*args, **kwargs)


# Shared N=6 learning controls.  The exact rooted six-bead SAW spectrum is
# m in {0, 1, 2}, so the whole window is reachable and a stall can only come
# from the visit requirement, never from geometry.
N6_LEARNING = dict(
    n_beads=6,
    m_min=0,
    m_max=2,
    seed=1701,
    initial_log_f=1.0,
    final_log_f=1e-4,
    flatness=0.75,
    min_visits=5000,
    min_cover_visits=1,
    check_every=1000,
    max_steps=1_000_000,
    max_seconds=120.0,
    checkpoint_every_seconds=1e9,
    progress=False,
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
            check_every=100,
            max_steps=100_000,
            max_seconds=30.0,
            max_steps_per_stage=100_000,
            schedule="one_over_t",
            # Only the Belardinelli-Pereyra crossing enters the 1/t phase, so
            # the stall is disabled here.  At these tiny modification factors
            # the crossing happens on the first check block: log_f=0.02 is
            # already below 1/t = 3/100.  The assertions exercise the 1/t
            # machinery itself, not the trigger that reached it.
            stage_stall_steps=10**9,
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


class OneOverTTriggerTests(unittest.TestCase):
    """The switch decision is a pure function, tested from a value table."""

    def test_trigger_table(self) -> None:
        cases = [
            # (log_f, inverse_time, stage_steps, stall_steps, fire, reason)
            # Neither: the stage is young and log_f is still far above 1/t.
            (1.0, 1e-4, 100, 10_000, False, ""),
            # Stall only: log_f is nowhere near 1/t, but the stage is stuck.
            (1.0, 1e-4, 10_000, 10_000, True, "stall"),
            (1.0, 1e-4, 10_001, 10_000, True, "stall"),
            # Belardinelli-Pereyra only: log_f has fallen to or below 1/t.
            (1e-5, 3e-5, 100, 10_000, True, "belardinelli_pereyra"),
            # Equality counts as a crossing.
            (3e-5, 3e-5, 100, 10_000, True, "belardinelli_pereyra"),
            # Both hold; the stall is reported because it is checked first.
            (1e-5, 3e-5, 10_000, 10_000, True, "stall"),
            # inverse_time far above log_f, early in a wide window.  The
            # trigger fires; the running minimum in learn_log_density is what
            # stops log_f from climbing.
            (1.0, 5.1, 10, 10_000, True, "belardinelli_pereyra"),
            # Just short of the stall, and log_f still above 1/t.
            (1e-3, 9.9e-4, 9_999, 10_000, False, ""),
        ]
        for log_f, inverse_time, stage_steps, stall_steps, fire, reason in cases:
            with self.subTest(log_f=log_f, inverse_time=inverse_time,
                              stage_steps=stage_steps):
                self.assertEqual(
                    one_over_t_trigger(
                        log_f, inverse_time, stage_steps, stall_steps
                    ),
                    (fire, reason),
                )


class OneOverTActivationTests(unittest.TestCase):
    def test_stall_relaxes_stages_without_entering_one_over_t(self) -> None:
        """A stage that cannot finish must escape -- by relaxing, not freezing.

        This is the regression guard for the switch having been gated on stage
        completion, which made --wl_schedule one_over_t inert: the same run
        finishes no stage at all without the fix.  The escape is the relaxed
        stage-advancement criterion; the 1/t rate law is *not* adopted, because
        a stall gives no reason to believe log_f and 1/t are comparable.
        """
        tier = make_contact_tiers(0, 2, 2, 2)
        result = _quiet(
            learn_log_density,
            schedule="one_over_t",
            stage_stall_steps=3_000,
            max_steps_per_stage=10_000_000,
            tier=tier,
            **N6_LEARNING,
        )
        self.assertTrue(result["stall_relaxed"])
        self.assertFalse(result["one_over_t_mode"])
        self.assertEqual(result["one_over_t_trigger"], "")
        self.assertGreater(result["stall_relaxed_step"], 0)
        self.assertLessEqual(result["next_log_f"], N6_LEARNING["final_log_f"])
        # Advancement is judged from the histogram reset at the stall, so it
        # must show the unrelaxed per-level counts.
        self.assertGreaterEqual(
            int(result["visits_since_stall"].min()), N6_LEARNING["min_visits"]
        )

    def test_stall_entry_does_not_reduce_log_f(self) -> None:
        """The stall must never adopt 1/t as the refinement factor.

        Measured at N=44: the stall fired at 3M steps where 1/t was 1.7e-5,
        already below final_log_f, so clamping log_f to it ended learning after
        3M steps of adaptation and froze a bias the chain had barely begun to
        build.  The condition is reproduced here in miniature -- the stall
        budget is below ``min_visits``, so the stage provably cannot finish
        first, and 1/t at the stall is an order of magnitude under final_log_f.
        """
        tier = make_contact_tiers(0, 2, 2, 2)
        samples: list = []
        entries: list = []
        original_save = wl._save_checkpoint
        original_trigger = wl.one_over_t_trigger

        def save_spy(path, **kwargs):
            samples.append(
                (int(kwargs["attempted_steps"]), float(kwargs["next_log_f"]))
            )
            return original_save(path, **kwargs)

        def trigger_spy(log_f, inverse_time, stage_steps, stall_steps):
            fire, reason = original_trigger(
                log_f, inverse_time, stage_steps, stall_steps
            )
            if fire:
                entries.append((reason, log_f, inverse_time))
            return fire, reason

        controls = dict(N6_LEARNING)
        controls.pop("checkpoint_every_seconds")
        controls["final_log_f"] = 1e-2
        with tempfile.TemporaryDirectory() as temporary:
            with mock.patch.object(wl, "_save_checkpoint", save_spy), \
                    mock.patch.object(wl, "one_over_t_trigger", trigger_spy):
                result = _quiet(
                    learn_log_density,
                    schedule="one_over_t",
                    stage_stall_steps=3_000,
                    max_steps_per_stage=10_000_000,
                    # Sample log_f on every check block, not only per stage.
                    checkpoint_every_seconds=1e-9,
                    checkpoint=Path(temporary) / "stall_entry.npz",
                    tier=tier,
                    **controls,
                )

        self.assertTrue(result["stall_relaxed"])
        self.assertFalse(result["one_over_t_mode"])

        reason, log_f_at_entry, inverse_time_at_entry = entries[0]
        self.assertEqual(reason, "stall")
        # The exact condition that made the old behaviour fatal.
        self.assertLess(inverse_time_at_entry, controls["final_log_f"])
        self.assertGreater(log_f_at_entry / inverse_time_at_entry, 100.0)

        # log_f is untouched at entry: across the whole run it either holds or
        # halves.  The pre-patch code divided it by more than 100 right here.
        for (_, earlier), (_, later) in zip(samples, samples[1:]):
            self.assertIn(round(later / earlier, 12), (1.0, 0.5))

        # And learning continues afterwards rather than freezing.
        after = [
            value for steps, value in samples
            if steps >= result["stall_relaxed_step"]
        ]
        self.assertLess(min(after), max(after))

    def test_belardinelli_pereyra_activates_one_over_t(self) -> None:
        """With the stall disabled, log_f <= 1/t is what enters the phase."""
        tier = make_contact_tiers(0, 2, 2, 2)
        controls = dict(N6_LEARNING)
        controls.update(
            final_log_f=1e-5, min_visits=100, check_every=5000, max_steps=5_000_000
        )
        result = _quiet(
            learn_log_density,
            schedule="one_over_t",
            stage_stall_steps=10**9,
            max_steps_per_stage=10_000_000,
            tier=tier,
            **controls,
        )
        self.assertTrue(result["one_over_t_mode"])
        # Assert on the reason, never on a stage index: which stage crosses
        # depends on the stage lengths this run happened to take.
        self.assertEqual(result["one_over_t_trigger"], "belardinelli_pereyra")

    def test_log_f_is_monotone_non_increasing(self) -> None:
        """1/t is recomputed every step, so log_f must be a running minimum."""
        tier = make_contact_tiers(0, 2, 2, 2)
        controls = dict(N6_LEARNING)
        controls.pop("checkpoint_every_seconds")
        controls.update(
            final_log_f=1e-5, min_visits=100, check_every=5000,
            max_steps=5_000_000,
        )
        observed: list = []
        original = wl._save_checkpoint

        def spy(path, **kwargs):
            observed.append(float(kwargs["next_log_f"]))
            return original(path, **kwargs)

        with tempfile.TemporaryDirectory() as temporary:
            with mock.patch.object(wl, "_save_checkpoint", spy):
                result = _quiet(
                    learn_log_density,
                    schedule="one_over_t",
                    stage_stall_steps=10**9,
                    max_steps_per_stage=10_000_000,
                    # Checkpoint on every check block so log_f is sampled
                    # throughout the run rather than only at the end.
                    checkpoint_every_seconds=1e-9,
                    checkpoint=Path(temporary) / "monotone.npz",
                    tier=tier,
                    **controls,
                )
        # The running minimum only matters once 1/t drives log_f.
        self.assertTrue(result["one_over_t_mode"])
        self.assertGreater(len(observed), 5)
        for earlier, later in zip(observed, observed[1:]):
            self.assertLessEqual(later, earlier)


class StallEscapeTests(unittest.TestCase):
    def test_relaxation_escapes_a_budget_halving_cannot_finish(self) -> None:
        """At an identical budget, halving dies where the relaxation completes.

        Halving re-earns the full per-level visit requirement in every one of
        its 14 stages.  The relaxation earns it once, cumulatively, and then
        only requires each stage to spend its stall budget -- 55k attempted
        moves against halving's 224k for the same 14 stages and the same final
        log_f.  The global cap here sits between the two.

        The per-stage step cap is deliberately left generous: stall relaxation
        widens the stage-advancement criterion and nothing else, so
        --wl_max_steps_per_stage still binds exactly as it does under halving.
        """
        tier = make_contact_tiers(0, 2, 2, 2)
        controls = dict(N6_LEARNING)
        controls["max_steps"] = 100_000
        shared = dict(max_steps_per_stage=10_000_000, tier=tier, **controls)

        with self.assertRaises(RuntimeError) as caught:
            _quiet(learn_log_density, schedule="halving", **shared)
        self.assertIn("--wl_max_steps", str(caught.exception))

        escaped = _quiet(
            learn_log_density,
            schedule="one_over_t",
            stage_stall_steps=3_000,
            **shared,
        )
        self.assertTrue(escaped["stall_relaxed"])
        self.assertFalse(escaped["one_over_t_mode"])
        self.assertLessEqual(escaped["next_log_f"], controls["final_log_f"])
        self.assertLess(escaped["attempted_steps"], controls["max_steps"])

    def test_unreachable_level_still_fails_loudly(self) -> None:
        """A geometric gap must fail in both schedules, never silently pass.

        Exact enumeration shows the 8-bead SAW reaches m=5 but not m=4, so an
        all-flat window can never be covered.  The stall relaxation widens the
        flatness ratio to visited levels only, but never the per-level coverage
        counts, so it cannot manufacture coverage the run never achieved; it
        still fails, and it still names the starved level.
        """
        tier = make_contact_tiers(0, 5, 5, 5)
        shared = dict(
            n_beads=8, m_min=0, m_max=5, seed=606, initial_log_f=1.0,
            final_log_f=0.01, flatness=0.8, min_visits=100, min_cover_visits=1,
            check_every=2000, max_steps=200_000, max_seconds=120.0,
            max_steps_per_stage=10_000_000, checkpoint_every_seconds=1e9,
            tier=tier, progress=False,
        )
        with self.assertRaises(RuntimeError) as halving:
            _quiet(learn_log_density, schedule="halving", **shared)
        self.assertIn("'m': 4", str(halving.exception))

        with self.assertRaises(RuntimeError) as one_over_t:
            _quiet(
                learn_log_density,
                schedule="one_over_t",
                stage_stall_steps=3_000,
                **shared,
            )
        message = str(one_over_t.exception)
        self.assertIn("'m': 4", message)
        # The relaxation did engage; it simply cannot cover an impossible level.
        self.assertIn("mode=one_over_t/stall_relaxed", message)


class OneOverTEstimatorTests(unittest.TestCase):
    def test_reweighting_still_matches_exact_enumeration(self) -> None:
        """The schedule must not disturb the frozen-weight estimator.

        The reweighted P(m) is consistent for any frozen log_g, so neither
        escape route may disturb it: both must still reproduce the exact
        3534-walk six-bead enumeration.
        """
        tier = make_contact_tiers(0, 2, 2, 2)
        exact_values, exact_prob, exact_radii = enumerate_rooted_saws(6)
        self.assertEqual(exact_radii.size, 3534)

        belardinelli = dict(N6_LEARNING)
        belardinelli.update(
            final_log_f=1e-5, min_visits=100, check_every=5000,
            max_steps=5_000_000,
        )
        routes = {
            "stall_relaxed": (dict(N6_LEARNING), 3_000, "stall_relaxed"),
            "belardinelli_pereyra": (belardinelli, 10**9, "one_over_t_mode"),
        }
        for name, (controls, stall_steps, expected_flag) in routes.items():
            with self.subTest(route=name):
                learned = _quiet(
                    learn_log_density,
                    schedule="one_over_t",
                    stage_stall_steps=stall_steps,
                    max_steps_per_stage=10_000_000,
                    tier=tier,
                    **controls,
                )
                self.assertTrue(learned[expected_flag])

                results = [
                    run_production_chain(
                        worker_id=i, seed=9000 + i,
                        initial_chain=learned["chain"],
                        log_g=learned["log_g"], m_min=0, m_max=2,
                        steps=250_000, burnin=0.1, sample_every=5,
                        progress=False, tier=learned["tier"],
                    )
                    for i in range(2)
                ]
                contacts = np.concatenate(
                    [r["contact_samples"] for r in results]
                )
                radii = np.concatenate([r["rg_samples"] for r in results])
                bends = np.concatenate([r["bend_samples"] for r in results])
                built = _quiet(
                    build_distributions, contacts, radii, bends,
                    learned["log_g"], 0, 24, False, n_beads=6, m_cover=2,
                )
                observed = np.zeros(3)
                observed[built["c_vals"]] = built["c_prob"]
                exact = np.zeros(3)
                exact[exact_values] = exact_prob
                total_variation = 0.5 * float(np.abs(observed - exact).sum())
                self.assertLess(total_variation, 0.03)


class WallClockCapTests(unittest.TestCase):
    def _checkpoint_with_history(self, path: Path, seconds: float) -> None:
        tier = make_contact_tiers(0, 2, 2, 2)
        _save_checkpoint(
            path,
            n_beads=6, m_min=0, m_max=2,
            chain=[(i, 0, 0) for i in range(6)],
            log_g=np.zeros(3), next_log_f=1.0, stages_completed=0,
            attempted_steps=0, accepted_moves=0, round_trips=0, seed=5,
            tier=tier, schedule="halving", pull_move_weight=0.0,
            one_over_t_mode=False,
            one_over_t_trigger_reason="", one_over_t_round_trips=0,
            stall_relaxed=False, stall_relaxed_step=0,
            visits_since_start=np.zeros(3, dtype=np.int64),
            visits_since_one_over_t=np.zeros(3, dtype=np.int64),
            visits_since_stall=np.zeros(3, dtype=np.int64),
            stage_records=np.zeros(0, dtype=WL_STAGE_DTYPE),
            learning_wall_seconds=seconds,
        )

    def test_cap_is_cumulative_across_resumes_by_default(self) -> None:
        """Without the historical term every resume restarts the budget."""
        tier = make_contact_tiers(0, 2, 2, 2)
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "resume.npz"
            self._checkpoint_with_history(checkpoint, 99_999.0)
            controls = dict(
                n_beads=6, m_min=0, m_max=2, seed=5, initial_log_f=1.0,
                final_log_f=0.25, flatness=0.5, min_visits=10,
                min_cover_visits=1, check_every=500, max_steps=200_000,
                max_seconds=30.0, max_steps_per_stage=10_000_000,
                checkpoint_every_seconds=1e9, tier=tier,
                resume_checkpoint=checkpoint, progress=False,
            )
            with self.assertRaises(RuntimeError) as caught:
                _quiet(
                    learn_log_density,
                    max_seconds_scope="cumulative",
                    **controls,
                )
            self.assertIn("--wl_max_seconds", str(caught.exception))
            self.assertIn("prior invocations=99999.0s", str(caught.exception))

            # The old per-invocation behaviour stays reachable on purpose.
            result = _quiet(
                learn_log_density,
                max_seconds_scope="per_invocation",
                **controls,
            )
            self.assertLessEqual(result["next_log_f"], 0.25)

    def test_unreachable_step_cap_warns(self) -> None:
        """A step cap far beyond the time cap is dead code, and must say so."""
        tier = make_contact_tiers(0, 2, 2, 2)
        with self.assertWarns(RuntimeWarning) as caught:
            learn_log_density(
                n_beads=6, m_min=0, m_max=2, seed=5, initial_log_f=1.0,
                final_log_f=0.25, flatness=0.5, min_visits=10,
                min_cover_visits=1, check_every=500,
                max_steps=10**12, max_seconds=0.5,
                max_steps_per_stage=10**12, checkpoint_every_seconds=1e9,
                tier=tier, progress=False,
            )
        message = str(caught.warning)
        self.assertIn("--wl_max_steps", message)
        self.assertIn("unreachable", message)
        self.assertIn("--wl_max_seconds is the operative limit", message)


if __name__ == "__main__":
    unittest.main()
