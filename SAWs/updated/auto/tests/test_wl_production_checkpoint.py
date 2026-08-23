from __future__ import annotations

import random
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


AUTO_DIR = Path(__file__).resolve().parents[1]
if str(AUTO_DIR) not in sys.path:
    sys.path.insert(0, str(AUTO_DIR))

from single_chain_wang_landau import (  # noqa: E402
    RoundTripCounter,
    _rng_state_from_arrays,
    _rng_state_to_arrays,
    _save_production_checkpoint,
    contact_count,
    make_contact_tiers,
    production_checkpoint_path,
    run_production_chain,
)


M_MAX = 5
N_BEADS = 8


def _rod():
    return [(i, 0, 0) for i in range(N_BEADS)]


def _run(**overrides):
    settings = dict(
        worker_id=0,
        seed=4242,
        initial_chain=_rod(),
        log_g=np.linspace(0.0, 1.5, M_MAX + 1),
        m_min=0,
        m_max=M_MAX,
        steps=2_000,
        burnin=0.0,
        sample_every=50,
        progress=False,
        tier=make_contact_tiers(0, M_MAX, M_MAX, M_MAX),
        pull_move_weight=0.25,
    )
    settings.update(overrides)
    return run_production_chain(**settings)


def _rewrite_checkpoint(path: Path, *, version: int, drop=()) -> None:
    """Rewrite a test checkpoint without pickle or private zip manipulation."""
    with np.load(path, allow_pickle=False) as saved:
        payload = {name: saved[name] for name in saved.files if name not in drop}
    payload["checkpoint_version"] = np.array(version, dtype=np.int64)
    np.savez_compressed(path, **payload)


class RngStateRoundTripTests(unittest.TestCase):
    """``random.Random`` state must survive an ``allow_pickle=False`` NPZ.

    The state is a 625-int tuple plus an optional cached normal variate, none of
    which numpy stores natively, so the split and the rebuild are checked
    against the only thing that matters: the next numbers out of the generator.
    """

    def _assert_round_trip(self, source: random.Random) -> None:
        version, keys, gauss_next = _rng_state_to_arrays(source.getstate())
        self.assertEqual(keys.dtype, np.uint32)
        self.assertEqual(keys.shape, (625,))
        restored = random.Random()
        restored.setstate(_rng_state_from_arrays(version, keys, gauss_next))
        self.assertEqual(
            [source.random() for _ in range(20)],
            [restored.random() for _ in range(20)],
        )

    def test_plain_state_round_trips(self) -> None:
        rng = random.Random(17)
        for _ in range(500):
            rng.random()
        self._assert_round_trip(rng)

    def test_cached_normal_variate_round_trips(self) -> None:
        # ``gauss`` caches a second variate, so ``gauss_next`` is not None here
        # and must not be silently dropped by the NaN encoding.
        rng = random.Random(17)
        rng.gauss(0.0, 1.0)
        self._assert_round_trip(rng)


class ProductionResumeTests(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.stem = Path(self._directory.name) / "run"
        self.addCleanup(self._directory.cleanup)

    def test_per_worker_paths_are_distinct(self) -> None:
        self.assertEqual(
            production_checkpoint_path(self.stem, 3).name, "run_prod_w3.npz"
        )
        self.assertNotEqual(
            production_checkpoint_path(self.stem, 0),
            production_checkpoint_path(self.stem, 1),
        )

    def test_resume_continues_the_same_chain_exactly(self) -> None:
        """A resumed worker reproduces the uninterrupted run step for step.

        The full generator state is checkpointed at a step boundary, so the
        continuation is exact rather than merely statistically equivalent.  The
        samples, acceptance count and round-trip phase all carry across.
        """
        path = production_checkpoint_path(self.stem, 0)
        whole = _run()
        _run(steps=1_000, checkpoint_path=path)
        resumed = _run(resume_path=path)

        np.testing.assert_array_equal(
            resumed["contact_samples"], whole["contact_samples"]
        )
        np.testing.assert_array_equal(resumed["rg_samples"], whole["rg_samples"])
        np.testing.assert_array_equal(resumed["bend_samples"], whole["bend_samples"])
        np.testing.assert_array_equal(
            resumed["coordination_histogram_samples"],
            whole["coordination_histogram_samples"],
        )
        self.assertEqual(resumed["accepted_moves"], whole["accepted_moves"])
        self.assertEqual(
            resumed["geometrically_valid_moves"], whole["geometrically_valid_moves"]
        )
        self.assertEqual(resumed["round_trips"], whole["round_trips"])
        self.assertEqual(resumed["attempted_moves"], whole["attempted_moves"])

    def test_burn_in_is_measured_against_the_total_step_budget(self) -> None:
        """A resumed worker must not re-burn and throw away good samples.

        The checkpoint sits at step 1000 of a 2000-step budget with a 0.5 burn-in
        fraction, so burn-in is already over.  Counting it per invocation would
        discard the first 500 resumed steps and yield five samples instead of ten.
        """
        path = production_checkpoint_path(self.stem, 0)
        chain = _rod()
        _save_production_checkpoint(
            path,
            worker_id=0,
            n_beads=N_BEADS,
            m_min=0,
            m_max=M_MAX,
            log_g=np.linspace(0.0, 1.5, M_MAX + 1),
            tier=make_contact_tiers(0, M_MAX, M_MAX, M_MAX),
            chain=chain,
            contact=contact_count(chain, set(chain)),
            steps_done=1_000,
            accepted=0,
            geometrically_valid=0,
            contacts=[],
            radii=[],
            bends=[],
            tracker=RoundTripCounter(0, M_MAX),
            rng_state=random.Random(1).getstate(),
        )
        resumed = _run(steps=2_000, burnin=0.5, sample_every=100, resume_path=path)
        self.assertEqual(resumed["contact_samples"].size, 10)

    def test_resume_refuses_a_different_frozen_bias(self) -> None:
        path = production_checkpoint_path(self.stem, 0)
        _run(steps=200, checkpoint_path=path)
        with self.assertRaisesRegex(ValueError, "log_g"):
            _run(log_g=np.zeros(M_MAX + 1), resume_path=path)

    def test_resume_refuses_a_different_contact_window(self) -> None:
        path = production_checkpoint_path(self.stem, 0)
        _run(steps=200, checkpoint_path=path)
        with self.assertRaisesRegex(ValueError, "contact window"):
            _run(
                m_max=M_MAX - 1,
                log_g=np.linspace(0.0, 1.5, M_MAX + 1)[:-1],
                tier=make_contact_tiers(0, M_MAX - 1, M_MAX - 1, M_MAX - 1),
                resume_path=path,
            )

    def test_resume_refuses_v1_checkpoint_that_already_has_samples(self) -> None:
        path = production_checkpoint_path(self.stem, 0)
        _run(steps=200, sample_every=20, checkpoint_path=path)
        _rewrite_checkpoint(
            path, version=1, drop=("coordination_histogram_samples",)
        )
        with self.assertRaisesRegex(ValueError, "predates local coordination"):
            _run(resume_path=path)

    def test_resume_refuses_unknown_future_checkpoint_version(self) -> None:
        path = production_checkpoint_path(self.stem, 0)
        _run(steps=200, checkpoint_path=path)
        _rewrite_checkpoint(path, version=3)
        with self.assertRaisesRegex(ValueError, "unsupported.*version"):
            _run(resume_path=path)


if __name__ == "__main__":
    unittest.main()
