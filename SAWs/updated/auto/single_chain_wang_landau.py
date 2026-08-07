#!/usr/bin/env python3
"""Wang-Landau baseline sampler for one self-avoiding lattice chain.

The physical target remains the uniform (athermal) measure over self-avoiding
conformations.  Wang-Landau adaptation is used only to learn a contact-number
bias.  The adaptive trajectory is discarded.  Independent production chains
then run with the frozen weight ``W(m) = exp(-log_g_hat[m])`` and every recorded
sample is reweighted by ``1/W(m) = exp(log_g_hat[m])``.

This separation matters: even an imperfect ``log_g_hat`` gives a consistent
athermal estimator when the frozen-weight production simulation is equilibrated
and covers the requested contact window.  A better estimate improves mixing and
statistical efficiency; self-normalized finite-sample estimates are not claimed
to be exactly unbiased.

Contact levels are split into a flat tier, a coverage-only tier, and an excluded
tier.  The flatness effort is concentrated where the molecular target has mass,
while every included level still has a minimum-coverage requirement.  Relaxing
flatness changes efficiency, not the limiting reweighted target, provided the
fixed-weight chain equilibrates and covers the declared window.

The optional Belardinelli-Pereyra schedule uses cumulative Monte Carlo time
``t = attempted_moves / included_levels``.  The time origin is never reset at a
refinement stage.  This is essential to the asymptotic ``1/t`` schedule.

Project probes measured about 50.8k, 38.1k, and 29.9k attempted moves per
second per learning core for N=30, 44, and 60.  Learning is single-process while
production is parallel, so the default final modification factor is 1e-4 and
the learning caps are deliberately generous.  This changes the quality of the
bias estimate, not the limiting target of the frozen-weight reweighting step.

The output preserves the distribution fields used by
``single_uniform_chain2_athermal_dists_joint.py``:

    c_vals, c_prob, c_edges, rg_edges, rg_prob, crg_prob

and its principal run metadata.  Additive ``wl_*`` fields record the learned
bias, flatness stages, range coverage, round trips, and importance-sampling
diagnostics.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import tempfile
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import permutations, product
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from baseline_grids import (
    LEGACY_BASELINE_FILES,
    assert_within_grid,
    fixed_c_edges,
    fixed_rg_edges,
    legacy_rg_grid,
    rod_rg,
)
from target_support import (
    CONTACT_OFFSETS,
    flat_level,
    load_target_contact_support,
    support_report,
    tail_mass_above,
)


Vec = Tuple[int, int, int]
Matrix = Tuple[Tuple[int, int, int], Tuple[int, int, int], Tuple[int, int, int]]
NN_VECS: Tuple[Vec, ...] = (
    (1, 0, 0), (-1, 0, 0), (0, 1, 0),
    (0, -1, 0), (0, 0, 1), (0, 0, -1),
)
BEND_DEFINITION = "number of 90-degree turns among the N-2 internal vertices"
RAW_SAMPLES_WARNING = (
    "These arrays are systematic importance resamples with duplicates; do not "
    "use them for variance or error-bar estimation. Use c_blocked_stderr."
)
TIER_EXCLUDED = np.int8(0)
TIER_COVERAGE = np.int8(1)
TIER_FLAT = np.int8(2)
WL_STAGE_DTYPE = np.dtype(
    [
        ("log_f", np.float64),
        ("steps", np.int64),
        ("wall_seconds", np.float64),
        ("min_over_mean", np.float64),
        ("min_visits_tier2", np.int64),
        ("min_visits_tier1", np.int64),
        ("accepted", np.int64),
        ("round_trips", np.int64),
        ("highest_m", np.int64),
        ("slowest_levels", np.int64, (3,)),
    ]
)


def add(a: Vec, b: Vec) -> Vec:
    return a[0] + b[0], a[1] + b[1], a[2] + b[2]


def sub(a: Vec, b: Vec) -> Vec:
    return a[0] - b[0], a[1] - b[1], a[2] - b[2]


def _det3(matrix: Matrix) -> int:
    (a, b, c), (d, e, f), (g, h, i) = matrix
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)


def generate_cubic_rotations() -> Tuple[Matrix, ...]:
    identity: Matrix = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    rotations: List[Matrix] = []
    for perm in permutations((0, 1, 2)):
        for signs in product((-1, 1), repeat=3):
            raw = [[0, 0, 0] for _ in range(3)]
            for row, col in enumerate(perm):
                raw[row][col] = signs[row]
            matrix: Matrix = (tuple(raw[0]), tuple(raw[1]), tuple(raw[2]))
            if _det3(matrix) == 1 and matrix != identity:
                rotations.append(matrix)
    if len(rotations) != 23:
        raise RuntimeError(f"expected 23 non-identity cubic rotations, got {len(rotations)}")
    return tuple(rotations)


ROT_MATS = generate_cubic_rotations()


def apply_rot(matrix: Matrix, vector: Vec) -> Vec:
    x, y, z = vector
    return (
        matrix[0][0] * x + matrix[0][1] * y + matrix[0][2] * z,
        matrix[1][0] * x + matrix[1][1] * y + matrix[1][2] * z,
        matrix[2][0] * x + matrix[2][1] * y + matrix[2][2] * z,
    )


def contact_count(chain: Sequence[Vec], occupied: Set[Vec]) -> int:
    """Return the unique non-bonded nearest-neighbour contact count."""
    count = 0
    n_beads = len(chain)
    for i, site in enumerate(chain):
        previous = chain[i - 1] if i > 0 else None
        following = chain[i + 1] if i + 1 < n_beads else None
        for direction in NN_VECS:
            neighbour = add(site, direction)
            if neighbour in occupied and neighbour != previous and neighbour != following:
                count += 1
    if count % 2:
        raise RuntimeError("contact double-count is odd; chain bookkeeping is inconsistent")
    return count // 2


def contact_delta_from_occupancy(old_occupied: Set[Vec], new_occupied: Set[Vec]) -> int:
    """Return the contact change using only sites added to or removed from occupancy.

    For a connected N-bead chain, ``m`` equals the number of occupied nearest-
    neighbour lattice edges minus the fixed ``N-1`` bonded edges.  We can
    therefore update ``m`` from the occupancy symmetric difference without
    rescanning the whole chain.  This makes crankshaft and end proposals O(1)
    while retaining an exact update for pivots.
    """
    removed = old_occupied - new_occupied
    added = new_occupied - old_occupied

    def incident_edges(changed: Set[Vec], occupied: Set[Vec]) -> int:
        edges = 0
        for site in changed:
            for direction in NN_VECS:
                neighbour = add(site, direction)
                if neighbour not in occupied:
                    continue
                # An edge joining two changed sites is otherwise counted twice.
                if neighbour in changed and neighbour < site:
                    continue
                edges += 1
        return edges

    return incident_edges(added, new_occupied) - incident_edges(removed, old_occupied)


def radius_of_gyration(chain: Sequence[Vec]) -> float:
    coordinates = np.asarray(chain, dtype=np.float64)
    centered = coordinates - coordinates.mean(axis=0)
    return float(np.sqrt(np.square(centered).sum(axis=1).mean()))


def count_bends(chain: Sequence[Vec]) -> int:
    bends = 0
    for i in range(1, len(chain) - 1):
        first = sub(chain[i], chain[i - 1])
        second = sub(chain[i + 1], chain[i])
        if first[0] * second[0] + first[1] * second[1] + first[2] * second[2] == 0:
            bends += 1
    return bends


def attempt_pivot(
    chain: List[Vec], occupied: Set[Vec], rng: random.Random
) -> Tuple[bool, List[Vec], Set[Vec]]:
    pivot_index = rng.randrange(1, len(chain) - 1)
    head = chain[: pivot_index + 1]
    pivot = chain[pivot_index]
    matrix = rng.choice(ROT_MATS)
    new_occupied = set(head)
    new_tail: List[Vec] = []
    for site in chain[pivot_index + 1 :]:
        moved = add(pivot, apply_rot(matrix, sub(site, pivot)))
        if moved in new_occupied:
            return False, chain, occupied
        new_tail.append(moved)
        new_occupied.add(moved)
    return True, head + new_tail, new_occupied


def attempt_crankshaft(
    chain: List[Vec], occupied: Set[Vec], rng: random.Random
) -> Tuple[bool, List[Vec], Set[Vec]]:
    index = rng.randrange(1, len(chain) - 1)
    previous, current, following = chain[index - 1], chain[index], chain[index + 1]
    first = sub(current, previous)
    second = sub(following, current)
    dot = first[0] * second[0] + first[1] * second[1] + first[2] * second[2]
    if first not in NN_VECS or second not in NN_VECS or dot != 0:
        return False, chain, occupied
    replacement = add(previous, second)
    if replacement in occupied:
        return False, chain, occupied
    new_chain = chain.copy()
    new_chain[index] = replacement
    return True, new_chain, (occupied - {current}) | {replacement}


def attempt_end_move(
    chain: List[Vec], occupied: Set[Vec], rng: random.Random
) -> Tuple[bool, List[Vec], Set[Vec]]:
    end = 0 if rng.random() < 0.5 else len(chain) - 1
    anchor = 1 if end == 0 else len(chain) - 2
    old_site = chain[end]
    occupied_without_old = occupied - {old_site}
    replacement = add(chain[anchor], rng.choice(NN_VECS))
    if replacement == old_site or replacement in occupied_without_old:
        return False, chain, occupied
    new_chain = chain.copy()
    new_chain[end] = replacement
    return True, new_chain, occupied_without_old | {replacement}


MOVE_FUNCS = (attempt_pivot, attempt_crankshaft, attempt_end_move)


def validate_chain(chain: Sequence[Vec]) -> None:
    if len(set(chain)) != len(chain):
        raise RuntimeError("chain is not self-avoiding")
    for first, second in zip(chain[:-1], chain[1:]):
        if sub(second, first) not in NN_VECS:
            raise RuntimeError("chain contains a non-unit lattice bond")


def metropolis_step(
    chain: List[Vec],
    occupied: Set[Vec],
    contact: int,
    log_g: np.ndarray,
    m_min: int,
    m_max: int,
    rng: random.Random,
) -> Tuple[List[Vec], Set[Vec], int, bool, bool]:
    """Take one fixed-bias proposal.

    Returns ``chain, occupied, contact, geometrically_valid, accepted``.  The
    proposal kernels are symmetric, so no Hastings proposal-ratio term appears.
    """
    move = rng.choice(MOVE_FUNCS)
    valid, proposed_chain, proposed_occupied = move(chain, occupied, rng)
    if not valid:
        return chain, occupied, contact, False, False
    proposed_contact = contact + contact_delta_from_occupancy(occupied, proposed_occupied)
    if proposed_contact < m_min or proposed_contact > m_max:
        return chain, occupied, contact, True, False
    log_acceptance = float(log_g[contact - m_min] - log_g[proposed_contact - m_min])
    if log_acceptance >= 0.0 or rng.random() < math.exp(log_acceptance):
        return proposed_chain, proposed_occupied, proposed_contact, True, True
    return chain, occupied, contact, True, False


class RoundTripCounter:
    """Count completed low -> high -> low traversals of a contact window."""

    def __init__(self, low: int, high: int) -> None:
        self.low = int(low)
        self.high = int(high)
        self.phase = 0
        self.round_trips = 0

    def observe(self, contact: int) -> None:
        if self.low == self.high:
            return
        if self.phase == 0 and contact <= self.low:
            self.phase = 1
        elif self.phase == 1 and contact >= self.high:
            self.phase = 2
        elif self.phase == 2 and contact <= self.low:
            self.round_trips += 1
            self.phase = 1


def _save_checkpoint(
    path: Path,
    *,
    n_beads: int,
    m_min: int,
    m_max: int,
    chain: Sequence[Vec],
    log_g: np.ndarray,
    next_log_f: float,
    stages_completed: int,
    attempted_steps: int,
    accepted_moves: int,
    round_trips: int,
    seed: int,
    tier: np.ndarray,
    schedule: str,
    one_over_t_mode: bool,
    stage_records: np.ndarray,
    learning_wall_seconds: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(
            temporary,
            checkpoint_version=np.array(2, dtype=np.int64),
            N=np.array(n_beads, dtype=np.int64),
            m_min=np.array(m_min, dtype=np.int64),
            m_max=np.array(m_max, dtype=np.int64),
            chain=np.asarray(chain, dtype=np.int64),
            log_g=np.asarray(log_g, dtype=np.float64),
            next_log_f=np.array(next_log_f, dtype=np.float64),
            stages_completed=np.array(stages_completed, dtype=np.int64),
            attempted_steps=np.array(attempted_steps, dtype=np.int64),
            accepted_moves=np.array(accepted_moves, dtype=np.int64),
            round_trips=np.array(round_trips, dtype=np.int64),
            original_seed=np.array(seed, dtype=np.int64),
            wl_tier=np.asarray(tier, dtype=np.int8),
            wl_schedule=np.array(schedule),
            one_over_t_mode=np.array(one_over_t_mode, dtype=bool),
            wl_stage_records=np.asarray(stage_records, dtype=WL_STAGE_DTYPE),
            learning_wall_seconds=np.array(learning_wall_seconds, dtype=np.float64),
        )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


GEOMETRIC_CONTACT_MAXIMA = {
    # Exact maxima for the chain lengths used by this project.  Each is attained
    # by a compact cubic-lattice shape with a Hamiltonian path.
    30: 30,  # 2 x 3 x 5 cuboid
    44: 50,  # 3 x 3 x 5 cuboid with one corner removed
    60: 74,  # 3 x 4 x 5 cuboid
}


def contact_upper_bound(n_beads: int) -> int:
    """Return a rigorous, not necessarily attainable, contact upper bound.

    N sites induce at most 3N - 3N^(2/3) lattice edges and N-1 of them are
    bonded, so m <= 2N + 1 - 3N^(2/3).  Integer lattice geometry and the
    Hamiltonian-path constraint can make the exact maximum smaller.
    """
    return math.floor(2 * n_beads + 1 - 3 * n_beads ** (2.0 / 3.0))


def geometric_contact_maximum(n_beads: int) -> Optional[int]:
    """Return the verified exact contact maximum, when one is encoded."""
    return GEOMETRIC_CONTACT_MAXIMA.get(n_beads)


def report_contact_upper_bound(args: argparse.Namespace) -> int:
    """Print the analytical upper bound and any verified exact maximum."""
    upper_bound = contact_upper_bound(args.N)
    exact_maximum = geometric_contact_maximum(args.N)
    print(f"rigorous contact upper bound for N={args.N}: {upper_bound}")
    if exact_maximum is None:
        print("verified exact geometric m_max: not encoded for this chain length")
    else:
        print(f"verified exact geometric m_max: {exact_maximum}")
    return 0


def make_contact_tiers(
    m_min: int,
    m_max: int,
    m_flat: int,
    m_cover: int,
    excluded_contact_levels: Sequence[int] = (),
) -> np.ndarray:
    """Build and validate the three-tier contact window."""
    if not m_min <= m_flat <= m_cover <= m_max:
        raise ValueError(
            "contact tiers require m_min <= m_flat <= m_cover <= m_max; "
            f"got {(m_min, m_flat, m_cover, m_max)}"
        )
    window = np.arange(m_min, m_max + 1, dtype=np.int64)
    tier = np.full(window.size, TIER_EXCLUDED, dtype=np.int8)
    tier[window <= m_cover] = TIER_COVERAGE
    tier[window <= m_flat] = TIER_FLAT
    excluded = np.asarray(excluded_contact_levels, dtype=np.int64)
    if excluded.size:
        if np.unique(excluded).size != excluded.size:
            raise ValueError("--excluded_contact_levels contains duplicates")
        invalid = excluded[(excluded <= m_min) | (excluded >= m_cover)]
        if invalid.size:
            raise ValueError(
                "--excluded_contact_levels may contain only verified internal "
                f"gaps inside (m_min, m_cover); invalid values: {invalid.tolist()}"
            )
        tier[excluded - m_min] = TIER_EXCLUDED
    if not np.any(tier == TIER_FLAT):
        raise ValueError("the contact window contains no tier-2 levels")
    included = np.flatnonzero(tier > TIER_EXCLUDED)
    if included.size == 0 or included[0] != 0 or included[-1] != m_cover - m_min:
        raise ValueError("the included contact window must retain both endpoints")
    return tier


def _validate_tier(tier: np.ndarray, expected_size: int) -> np.ndarray:
    tier = np.asarray(tier, dtype=np.int8)
    if tier.shape != (expected_size,):
        raise ValueError("tier array shape does not match the contact window")
    if not np.all(np.isin(tier, (TIER_EXCLUDED, TIER_COVERAGE, TIER_FLAT))):
        raise ValueError("tier array may contain only 0, 1, and 2")
    if not np.any(tier == TIER_FLAT):
        raise ValueError("tier array contains no flat levels")
    return tier


def _included_limits(tier: np.ndarray, m_min: int) -> Tuple[int, int]:
    included = np.flatnonzero(tier > TIER_EXCLUDED)
    if included.size == 0:
        raise ValueError("tier array excludes every contact level")
    return int(included[0] + m_min), int(included[-1] + m_min)


def _slowest_levels(histogram: np.ndarray, tier: np.ndarray, m_min: int) -> np.ndarray:
    included = np.flatnonzero(tier > TIER_EXCLUDED)
    order = included[np.argsort(histogram[included], kind="stable")]
    levels = np.full(3, -1, dtype=np.int64)
    count = min(3, order.size)
    levels[:count] = order[:count] + m_min
    return levels


def _stage_visit_statistics(
    histogram: np.ndarray, tier: np.ndarray
) -> Tuple[float, int, int, bool]:
    flat_counts = histogram[tier == TIER_FLAT]
    mean_flat = float(flat_counts.mean())
    min_flat = int(flat_counts.min())
    ratio = min_flat / mean_flat if mean_flat else 0.0
    coverage_counts = histogram[tier == TIER_COVERAGE]
    min_coverage = int(coverage_counts.min()) if coverage_counts.size else 0
    included = np.flatnonzero(tier > TIER_EXCLUDED)
    endpoints_covered = bool(
        histogram[included[0]] > 0 and histogram[included[-1]] > 0
    )
    return ratio, min_flat, min_coverage, endpoints_covered


def learn_log_density(
    *,
    n_beads: int,
    m_min: int,
    m_max: int,
    seed: int,
    initial_log_f: float,
    final_log_f: float,
    flatness: float,
    min_visits: int,
    check_every: int,
    max_steps: int,
    min_cover_visits: int = 50,
    max_seconds: float = math.inf,
    max_steps_per_stage: int = 1_000_000_000,
    schedule: str = "halving",
    checkpoint_every_seconds: float = 1800.0,
    tier: Optional[np.ndarray] = None,
    checkpoint: Optional[Path] = None,
    resume_checkpoint: Optional[Path] = None,
    progress: bool = True,
) -> Dict[str, Any]:
    """Learn a contact density estimate over a tiered contact window.

    The halving schedule uses the tier-2 flatness ratio and minimum visit counts
    in both included tiers.  The Belardinelli-Pereyra schedule uses cumulative
    time per included level and drops the flatness-ratio requirement during its
    pre-asymptotic coverage stages and continuous ``1/t`` phase.
    """
    if schedule not in {"halving", "one_over_t"}:
        raise ValueError("schedule must be 'halving' or 'one_over_t'")
    if min(min_visits, min_cover_visits, check_every, max_steps, max_steps_per_stage) < 1:
        raise ValueError("WL visit/check/step controls must be positive")
    if max_seconds <= 0.0 or checkpoint_every_seconds <= 0.0:
        raise ValueError("WL time controls must be positive")

    if resume_checkpoint is None:
        chain: List[Vec] = [(i, 0, 0) for i in range(n_beads)]
        log_g = np.zeros(m_max - m_min + 1, dtype=np.float64)
        log_f = float(initial_log_f)
        stages_completed = 0
        total_steps = 0
        accepted = 0
        previous_round_trips = 0
        one_over_t_mode = False
        stage_records: List[Any] = []
        historical_wall = 0.0
        saved_tier = None
    else:
        with np.load(resume_checkpoint, allow_pickle=False) as saved:
            saved_n = int(saved["N"])
            saved_min = int(saved["m_min"])
            saved_max = int(saved["m_max"])
            if (saved_n, saved_min, saved_max) != (n_beads, m_min, m_max):
                raise ValueError(
                    "checkpoint N/contact window does not match the requested run: "
                    f"saved={(saved_n, saved_min, saved_max)}, "
                    f"requested={(n_beads, m_min, m_max)}"
                )
            chain = [tuple(map(int, row)) for row in np.asarray(saved["chain"])]
            log_g = np.asarray(saved["log_g"], dtype=np.float64).copy()
            log_f = float(saved["next_log_f"])
            stages_completed = int(saved["stages_completed"])
            total_steps = int(saved["attempted_steps"])
            accepted = int(saved["accepted_moves"])
            previous_round_trips = int(saved["round_trips"])
            one_over_t_mode = bool(saved["one_over_t_mode"]) if "one_over_t_mode" in saved else False
            historical_wall = (
                float(saved["learning_wall_seconds"])
                if "learning_wall_seconds" in saved else 0.0
            )
            stage_records = (
                list(np.asarray(saved["wl_stage_records"], dtype=WL_STAGE_DTYPE))
                if "wl_stage_records" in saved else []
            )
            saved_tier = (
                np.asarray(saved["wl_tier"], dtype=np.int8)
                if "wl_tier" in saved else None
            )
            if "wl_schedule" in saved:
                saved_schedule = str(saved["wl_schedule"])
                if saved_schedule != schedule:
                    raise ValueError(
                        "checkpoint WL schedule does not match the requested run: "
                        f"saved={saved_schedule}, requested={schedule}"
                    )
        validate_chain(chain)

    if tier is None:
        tier = np.full(log_g.size, TIER_FLAT, dtype=np.int8)
    tier = _validate_tier(tier, log_g.size)
    if saved_tier is not None and not np.array_equal(saved_tier, tier):
        raise ValueError("checkpoint contact tiers do not match the requested run")
    included = tier > TIER_EXCLUDED
    included_count = int(included.sum())
    included_low, included_high = _included_limits(tier, m_min)

    occupied = set(chain)
    contact = contact_count(chain, occupied)
    if not (m_min <= contact <= included_high):
        raise ValueError(
            "initial/checkpoint chain has "
            f"m={contact}, outside [{m_min}, {included_high}]"
        )
    if tier[contact - m_min] == TIER_EXCLUDED:
        raise ValueError(f"initial/checkpoint chain is at excluded level m={contact}")
    rng = random.Random(seed + 1_000_003 * stages_completed)
    tracker = RoundTripCounter(included_low, included_high)
    tracker.observe(contact)
    t0 = time.time()
    last_checkpoint_time = t0

    def checkpoint_now(current_log_f: float) -> None:
        if checkpoint is None:
            return
        _save_checkpoint(
            checkpoint,
            n_beads=n_beads,
            m_min=m_min,
            m_max=m_max,
            chain=chain,
            log_g=log_g,
            next_log_f=current_log_f,
            stages_completed=stages_completed,
            attempted_steps=total_steps,
            accepted_moves=accepted,
            round_trips=previous_round_trips + tracker.round_trips,
            seed=seed,
            tier=tier,
            schedule=schedule,
            one_over_t_mode=one_over_t_mode,
            stage_records=np.asarray(stage_records, dtype=WL_STAGE_DTYPE),
            learning_wall_seconds=historical_wall + (time.time() - t0),
        )

    def cap_failure(
        cap_name: str,
        suggestion: str,
        histogram: np.ndarray,
        stage_steps: int,
        stage_start: float,
        highest_m: int,
        highest_first_step: int,
    ) -> RuntimeError:
        checkpoint_now(log_f)
        elapsed = time.time() - stage_start
        rate = stage_steps / elapsed if elapsed > 0.0 else float("nan")
        slow = np.flatnonzero(included)[
            np.argsort(histogram[included], kind="stable")[:5]
        ]
        slow_text = [
            {
                "m": int(index + m_min),
                "count": int(histogram[index]),
                "tier": int(tier[index]),
            }
            for index in slow
        ]
        return RuntimeError(
            f"Wang-Landau learning reached {cap_name} at stage "
            f"{stages_completed + 1}, log_f={log_f:.6g}. "
            f"Slowest levels: {slow_text}. Highest m reached this stage: "
            f"{highest_m} (first reached at stage step {highest_first_step}). "
            f"Stage elapsed={elapsed:.1f}s, rate={rate:.1f} steps/s. {suggestion}"
        )

    finished = False
    while not finished and (one_over_t_mode or log_f > final_log_f):
        histogram = np.zeros_like(log_g, dtype=np.int64)
        stage_steps = 0
        stage_accepted = 0
        stage_start = time.time()
        stage_log_f = float(log_f)
        stage_trip_start = tracker.round_trips
        highest_m = int(contact)
        highest_first_step = 0
        while True:
            elapsed_total = time.time() - t0
            if elapsed_total >= max_seconds:
                raise cap_failure(
                    "--wl_max_seconds",
                    "Increase --wl_max_seconds or resume from the checkpoint.",
                    histogram, stage_steps, stage_start, highest_m,
                    highest_first_step,
                )
            overall_remaining = max_steps - total_steps
            stage_remaining = max_steps_per_stage - stage_steps
            block = min(check_every, overall_remaining, stage_remaining)
            if block <= 0:
                if overall_remaining <= 0:
                    raise cap_failure(
                        "--wl_max_steps",
                        "Increase --wl_max_steps or resume from the checkpoint.",
                        histogram, stage_steps, stage_start, highest_m,
                        highest_first_step,
                    )
                raise cap_failure(
                    "--wl_max_steps_per_stage",
                    "Increase --wl_max_steps_per_stage after checking the slow levels.",
                    histogram, stage_steps, stage_start, highest_m,
                    highest_first_step,
                )
            for _ in range(block):
                chain, occupied, contact, _, was_accepted = metropolis_step(
                    chain, occupied, contact, log_g, m_min, included_high, rng
                )
                total_steps += 1
                stage_steps += 1
                if was_accepted:
                    accepted += 1
                    stage_accepted += 1
                index = contact - m_min
                if tier[index] == TIER_EXCLUDED:
                    raise RuntimeError(
                        f"encountered excluded contact level m={contact}; remove it "
                        "from --excluded_contact_levels and restart learning"
                    )
                if contact > highest_m:
                    highest_m = int(contact)
                    highest_first_step = int(stage_steps)
                if one_over_t_mode:
                    log_f = included_count / float(total_steps)
                log_g[index] += log_f
                histogram[index] += 1
                tracker.observe(contact)

            now = time.time()
            if checkpoint is not None and now - last_checkpoint_time >= checkpoint_every_seconds:
                checkpoint_now(log_f)
                last_checkpoint_time = now

            ratio, min_flat, min_coverage, range_covered = _stage_visit_statistics(
                histogram, tier
            )
            coverage_ok = (
                range_covered
                and min_flat >= min_visits
                and (
                    not np.any(tier == TIER_COVERAGE)
                    or min_coverage >= min_cover_visits
                )
            )
            stage_complete = coverage_ok and (
                schedule == "one_over_t" or ratio >= flatness
            )
            if progress:
                print(
                    f"[WL stage {stages_completed + 1}] log_f={log_f:.6g} "
                    f"steps={stage_steps} min/mean={ratio:.3f} "
                    f"min_tier2={min_flat} min_tier1={min_coverage} "
                    f"range={range_covered} highest_m={highest_m} "
                    f"round_trips={previous_round_trips + tracker.round_trips} "
                    f"mode={'1/t' if one_over_t_mode else schedule}",
                    flush=True,
                )

            if one_over_t_mode:
                log_f = included_count / float(total_steps)
                if log_f > final_log_f or not coverage_ok:
                    continue
                stage_records.append(
                    (
                        stage_log_f,
                        stage_steps,
                        time.time() - stage_start,
                        ratio,
                        min_flat,
                        min_coverage,
                        stage_accepted,
                        tracker.round_trips - stage_trip_start,
                        highest_m,
                        _slowest_levels(histogram, tier, m_min),
                    )
                )
                stages_completed += 1
                log_g -= log_g[0]
                finished = True
                checkpoint_now(log_f)
                break

            if not stage_complete:
                continue
            stage_records.append(
                (
                    stage_log_f,
                    stage_steps,
                    time.time() - stage_start,
                    ratio,
                    min_flat,
                    min_coverage,
                    stage_accepted,
                    tracker.round_trips - stage_trip_start,
                    highest_m,
                    _slowest_levels(histogram, tier, m_min),
                )
            )
            stages_completed += 1
            next_log_f = 0.5 * log_f
            log_g -= log_g[0]
            if schedule == "one_over_t":
                inverse_time = included_count / float(total_steps)
                # The refinement factor must remain monotone at the switch.
                if next_log_f <= inverse_time <= log_f:
                    one_over_t_mode = True
                    log_f = inverse_time
                else:
                    log_f = next_log_f
            else:
                log_f = next_log_f
            checkpoint_now(log_f)
            last_checkpoint_time = time.time()
            break

    log_g -= log_g[0]
    return {
        "log_g": log_g,
        "chain": chain,
        "attempted_steps": total_steps,
        "accepted_moves": accepted,
        "stages_completed": stages_completed,
        "stage_records": np.asarray(stage_records, dtype=WL_STAGE_DTYPE),
        "round_trips": previous_round_trips + tracker.round_trips,
        "wall_time": historical_wall + (time.time() - t0),
        "next_log_f": log_f,
        "tier": tier,
        "active": included,
        "one_over_t_mode": one_over_t_mode,
    }


def run_production_chain(
    worker_id: int,
    seed: int,
    initial_chain: Sequence[Vec],
    log_g: np.ndarray,
    m_min: int,
    m_max: int,
    steps: int,
    burnin: float,
    sample_every: int,
    progress: bool = True,
    tier: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    rng = random.Random(seed)
    chain = [tuple(site) for site in initial_chain]
    occupied = set(chain)
    contact = contact_count(chain, occupied)
    if tier is None:
        tier = np.full(log_g.size, TIER_FLAT, dtype=np.int8)
    if log_g.shape != (m_max - m_min + 1,):
        raise ValueError("log_g shape does not match [m_min, m_max]")
    tier = _validate_tier(tier, m_max - m_min + 1)
    included_low, included_high = _included_limits(tier, m_min)
    if tier[contact - m_min] == TIER_EXCLUDED:
        raise RuntimeError(
            f"production worker {worker_id} started in excluded contact level "
            f"m={contact}"
        )
    burn_steps = int(round(burnin * steps))
    contacts: List[int] = []
    radii: List[float] = []
    bends: List[int] = []
    accepted = 0
    geometrically_valid = 0
    tracker = RoundTripCounter(included_low, included_high)
    t0 = time.time()
    progress_mark = max(1, steps // 10)

    for step in range(1, steps + 1):
        chain, occupied, contact, valid, was_accepted = metropolis_step(
            chain, occupied, contact, log_g, m_min, included_high, rng
        )
        if tier[contact - m_min] == TIER_EXCLUDED:
            raise RuntimeError(
                f"production worker {worker_id} encountered excluded contact level "
                f"m={contact}; remove it from --excluded_contact_levels"
            )
        if valid:
            geometrically_valid += 1
        if was_accepted:
            accepted += 1
        if step > burn_steps:
            tracker.observe(contact)
            if (step - burn_steps) % sample_every == 0:
                contacts.append(contact)
                radii.append(radius_of_gyration(chain))
                bends.append(count_bends(chain))
        if progress and step % progress_mark == 0:
            print(
                f"[production worker {worker_id} seed={seed}] "
                f"{100.0 * step / steps:5.1f}% ({step}/{steps}) "
                f"accepted={accepted} samples={len(contacts)} "
                f"round_trips={tracker.round_trips}",
                flush=True,
            )

    return {
        "worker_id": worker_id,
        "seed": seed,
        "contact_samples": np.asarray(contacts, dtype=np.int64),
        "rg_samples": np.asarray(radii, dtype=np.float64),
        "bend_samples": np.asarray(bends, dtype=np.int64),
        "accepted_moves": accepted,
        "geometrically_valid_moves": geometrically_valid,
        "attempted_moves": steps,
        "round_trips": tracker.round_trips,
        "wall_time": time.time() - t0,
    }


def normalized_importance_weights(contacts: np.ndarray, log_g: np.ndarray, m_min: int) -> np.ndarray:
    log_weights = log_g[contacts - m_min]
    shift = float(np.max(log_weights))
    weights = np.exp(log_weights - shift)
    total = float(weights.sum())
    if not math.isfinite(total) or total <= 0.0:
        raise RuntimeError("importance weights are non-finite or sum to zero")
    return weights / total


def effective_sample_size(normalized_weights: np.ndarray) -> float:
    return float(1.0 / np.square(normalized_weights).sum())


def systematic_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    count = weights.size
    positions = (rng.random() + np.arange(count)) / count
    cumulative = np.cumsum(weights)
    cumulative[-1] = 1.0
    return np.searchsorted(cumulative, positions, side="right")


def blocked_contact_stderr(
    worker_contacts: Sequence[np.ndarray],
    log_g: np.ndarray,
    m_min: int,
    m_cover: int,
    n_blocks: int,
) -> Tuple[np.ndarray, int]:
    """Estimate the standard error of P0(m) from per-worker batch means.

    Each non-overlapping block is reweighted and self-normalized independently.
    Blocks are pooled across independent workers, and their between-block sample
    variance is divided by the number of blocks.  This accounts for within-chain
    autocorrelation when blocks are long relative to the correlation time; it
    does not include uncertainty in the learned bias itself.
    """
    if n_blocks < 2:
        raise ValueError("n_blocks must be at least 2")
    n_levels = m_cover - m_min + 1
    estimates: List[np.ndarray] = []
    for worker_id, values in enumerate(worker_contacts):
        contacts = np.asarray(values, dtype=np.int64)
        if contacts.size < 2:
            raise ValueError(
                f"production worker {worker_id} has fewer than two recorded samples"
            )
        if np.any((contacts < m_min) | (contacts > m_cover)):
            raise ValueError(
                f"production worker {worker_id} has contacts outside the declared window"
            )
        for block in np.array_split(contacts, min(n_blocks, contacts.size)):
            weights = normalized_importance_weights(block, log_g, m_min)
            estimates.append(
                np.bincount(
                    block - m_min, weights=weights, minlength=n_levels
                )[:n_levels]
            )
    batches = np.asarray(estimates, dtype=np.float64)
    if batches.shape[0] < 2:
        raise ValueError("batch-means error estimation requires at least two blocks")
    stderr = batches.std(axis=0, ddof=1) / math.sqrt(batches.shape[0])
    return stderr, int(batches.shape[0])


def build_distributions(
    contacts: np.ndarray,
    radii: np.ndarray,
    bends: np.ndarray,
    log_g: np.ndarray,
    m_min: int,
    rg_bins: int,
    no_joint: bool,
    *,
    n_beads: Optional[int] = None,
    m_cover: Optional[int] = None,
    rg_edges: Optional[np.ndarray] = None,
    grid_source: Optional[str] = None,
    grid_search_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    if contacts.size == 0 or contacts.size != radii.size or contacts.size != bends.size:
        raise RuntimeError("production sample arrays are empty or have inconsistent lengths")
    if not np.all(np.isfinite(radii)):
        raise RuntimeError("non-finite Rg production samples")
    if m_cover is None:
        m_cover = int(contacts.max())
    if m_cover < m_min:
        raise ValueError("m_cover must be at least m_min")
    weights = normalized_importance_weights(contacts, log_g, m_min)
    c_edges = fixed_c_edges(m_min, m_cover)
    c_vals = np.arange(m_min, m_cover + 1, dtype=np.int64)
    assert_within_grid(contacts, c_edges, "contact")
    c_prob = np.bincount(
        contacts - m_min, weights=weights, minlength=c_vals.size
    )[:c_vals.size].astype(np.float64)
    production_c_counts = np.bincount(
        contacts - m_min, minlength=c_vals.size
    )[:c_vals.size].astype(np.int64)
    if not math.isclose(float(c_prob.sum()), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError("contact histogram lost probability mass")
    c_prob /= c_prob.sum()
    # Raw visits from the biased fixed-weight production run.  These are not
    # athermal counts, but because every sample in one contact level carries the
    # same importance weight, they are what sets the statistical error of
    # P0(m) and of P0(Rg | m) in that level.  n_samples * c_prob does not:
    # in the compact tail it is orders of magnitude smaller than the number of
    # samples actually backing the bin, which would report the best-sampled
    # levels as the worst.
    # Superseded by c_blocked_stderr for error estimation.  This remains only a
    # raw coverage indicator and ignores autocorrelation and reweighting.
    c_naive_count_error = np.full(c_vals.size, np.inf, dtype=np.float64)
    visited = production_c_counts > 0
    c_naive_count_error[visited] = 1.0 / np.sqrt(
        production_c_counts[visited].astype(np.float64)
    )

    legacy_edges = np.array([], dtype=np.float64)
    legacy_source = ""
    search_dir = Path(__file__).resolve().parent if grid_search_dir is None else grid_search_dir
    if rg_edges is None and n_beads in LEGACY_BASELINE_FILES:
        legacy_edges, width, _ = legacy_rg_grid(int(n_beads), search_dir)
        rg_edges = fixed_rg_edges(int(n_beads), search_dir)
        resolved_grid_source = "legacy_extended"
        legacy_source = LEGACY_BASELINE_FILES[int(n_beads)]
    elif rg_edges is None:
        if n_beads is None:
            raise ValueError("n_beads is required when rg_edges is not supplied")
        rg_min, rg_max = float(radii.min()), float(radii.max())
        pad = 1e-9 if rg_max <= rg_min else 0.02 * (rg_max - rg_min)
        upper = max(rg_max + pad, rod_rg(int(n_beads)))
        rg_edges = np.linspace(rg_min - pad, upper, rg_bins + 1)
        width = float(np.median(np.diff(rg_edges)))
        resolved_grid_source = "adaptive"
        warnings.warn(
            f"N={n_beads} has no registered legacy Rg grid; using a run-dependent "
            "adaptive grid padded to the rod maximum",
            RuntimeWarning,
            stacklevel=2,
        )
    else:
        rg_edges = np.asarray(rg_edges, dtype=np.float64)
        width = float(np.median(np.diff(rg_edges)))
        resolved_grid_source = grid_source or "provided"
    rg_edges = np.asarray(rg_edges, dtype=np.float64)
    rg_out_of_range_count = assert_within_grid(radii, rg_edges, "Rg")
    rg_prob, _ = np.histogram(radii, bins=rg_edges, weights=weights)
    rg_prob = rg_prob.astype(np.float64)
    if not math.isclose(float(rg_prob.sum()), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError("Rg histogram lost probability mass")
    rg_prob /= rg_prob.sum()

    if no_joint:
        crg_prob = np.array([[]], dtype=np.float64)
    else:
        assert_within_grid(contacts, c_edges, "joint contact")
        assert_within_grid(radii, rg_edges, "joint Rg")
        crg_prob, _, _ = np.histogram2d(
            contacts.astype(np.float64), radii, bins=(c_edges, rg_edges), weights=weights
        )
        crg_prob = crg_prob.astype(np.float64)
        if not math.isclose(float(crg_prob.sum()), 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError("joint histogram lost probability mass")
        crg_prob /= crg_prob.sum()

    bend_vals = np.unique(bends)
    bend_mass = np.array([weights[bends == value].sum() for value in bend_vals])
    bend_mass /= bend_mass.sum()
    mean_bends = float(np.dot(weights, bends))

    if no_joint:
        marginal_m_error = float("nan")
        marginal_rg_error = float("nan")
    else:
        marginal_m_error = float(np.max(np.abs(crg_prob.sum(axis=1) - c_prob)))
        marginal_rg_error = float(np.max(np.abs(crg_prob.sum(axis=0) - rg_prob)))

    return {
        "weights": weights,
        "c_vals": c_vals,
        "c_prob": c_prob,
        "production_c_counts": production_c_counts,
        "c_naive_count_error": c_naive_count_error,
        "c_edges": c_edges,
        "rg_edges": rg_edges,
        "rg_grid_source": resolved_grid_source,
        "legacy_rg_edges": legacy_edges,
        "legacy_grid_source": legacy_source,
        "rg_grid_width": width,
        "rg_out_of_range_count": rg_out_of_range_count,
        "rg_prob": rg_prob,
        "crg_prob": crg_prob,
        "bend_vals": bend_vals,
        "bend_prob": bend_mass,
        "mean_bends": mean_bends,
        "marginal_m_error": marginal_m_error,
        "marginal_rg_error": marginal_rg_error,
    }


def enumerate_rooted_saws(n_beads: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Enumerate short rooted SAWs for exact validation; not used in production."""
    chain: List[Vec] = [(0, 0, 0)]
    occupied: Set[Vec] = {chain[0]}
    contacts: List[int] = []
    radii: List[float] = []

    def visit() -> None:
        if len(chain) == n_beads:
            contacts.append(contact_count(chain, occupied))
            radii.append(radius_of_gyration(chain))
            return
        for direction in NN_VECS:
            site = add(chain[-1], direction)
            if site in occupied:
                continue
            chain.append(site)
            occupied.add(site)
            visit()
            occupied.remove(site)
            chain.pop()

    visit()
    c = np.asarray(contacts, dtype=np.int64)
    r = np.asarray(radii, dtype=np.float64)
    values, counts = np.unique(c, return_counts=True)
    return values, counts / counts.sum(), r


def run_self_test() -> int:
    print("Self-test: exact enumeration and reweighted WL production")
    n_beads = 6
    exact_values, exact_prob, exact_radii = enumerate_rooted_saws(n_beads)
    if int(exact_prob.size) < 2:
        print("FAIL: exact test chain does not span multiple contact levels")
        return 1
    m_min, m_max = int(exact_values.min()), int(exact_values.max())
    learned = learn_log_density(
        n_beads=n_beads,
        m_min=m_min,
        m_max=m_max,
        seed=1701,
        initial_log_f=1.0,
        final_log_f=1e-4,
        flatness=0.75,
        min_visits=100,
        check_every=1000,
        max_steps=1_000_000,
        progress=False,
    )
    results = [
        run_production_chain(
            worker_id=i,
            seed=9000 + i,
            initial_chain=learned["chain"],
            log_g=learned["log_g"],
            m_min=m_min,
            m_max=m_max,
            steps=250_000,
            burnin=0.1,
            sample_every=5,
            progress=False,
        )
        for i in range(2)
    ]
    contacts = np.concatenate([result["contact_samples"] for result in results])
    radii = np.concatenate([result["rg_samples"] for result in results])
    bends = np.concatenate([result["bend_samples"] for result in results])
    built = build_distributions(
        contacts,
        radii,
        bends,
        learned["log_g"],
        m_min,
        24,
        False,
        n_beads=n_beads,
        m_cover=m_max,
    )
    observed = np.zeros(m_max - m_min + 1)
    observed[built["c_vals"] - m_min] = built["c_prob"]
    exact = np.zeros_like(observed)
    exact[exact_values - m_min] = exact_prob
    total_variation = 0.5 * float(np.abs(observed - exact).sum())
    weighted_mean_rg = float(np.dot(built["weights"], radii))
    exact_mean_rg = float(exact_radii.mean())
    relative_rg_error = abs(weighted_mean_rg - exact_mean_rg) / exact_mean_rg
    checks = {
        "23 proper non-identity rotations": len(ROT_MATS) == 23,
        "exact N=6 rooted SAW count is 3534": exact_radii.size == 3534,
        "contact total variation < 0.03": total_variation < 0.03,
        "mean Rg relative error < 0.02": relative_rg_error < 0.02,
        "P(m) normalized": abs(float(built["c_prob"].sum()) - 1.0) < 1e-12,
        "P(Rg) normalized": abs(float(built["rg_prob"].sum()) - 1.0) < 1e-12,
        "joint normalized": abs(float(built["crg_prob"].sum()) - 1.0) < 1e-12,
        "joint contact marginal consistent": built["marginal_m_error"] < 1e-12,
        "joint Rg marginal consistent": built["marginal_rg_error"] < 1e-12,
    }
    delta_rng = random.Random(771)
    delta_chain: List[Vec] = [(i, 0, 0) for i in range(12)]
    delta_occupied = set(delta_chain)
    delta_contact = 0
    delta_updates_agree = True
    for _ in range(5000):
        valid, candidate, candidate_occupied = delta_rng.choice(MOVE_FUNCS)(
            delta_chain, delta_occupied, delta_rng
        )
        if not valid:
            continue
        candidate_contact = delta_contact + contact_delta_from_occupancy(
            delta_occupied, candidate_occupied
        )
        if candidate_contact != contact_count(candidate, candidate_occupied):
            delta_updates_agree = False
            break
        delta_chain, delta_occupied, delta_contact = (
            candidate, candidate_occupied, candidate_contact
        )
    checks["incremental contact updates match full recounts"] = delta_updates_agree

    # Regression guard for a verified gapped contact spectrum.  Exact
    # enumeration shows that the 8-bead SAW can reach m=5 but not m=4.
    gap_values, _, _ = enumerate_rooted_saws(8)
    gap_active = np.isin(np.arange(6), gap_values)
    gap_tier = np.where(gap_active, TIER_FLAT, TIER_EXCLUDED).astype(np.int8)
    checks["exact N=8 support has only the internal gap m=4"] = (
        not bool(gap_active[4]) and bool(gap_active[5]) and bool(gap_active[:4].all())
    )
    try:
        learn_log_density(
            n_beads=8, m_min=0, m_max=5, seed=606, initial_log_f=1.0,
            final_log_f=0.01, flatness=0.8, min_visits=100, check_every=2000,
            max_steps=2_000_000, tier=gap_tier, progress=False,
        )
        gap_learn_ok = True
    except RuntimeError:
        gap_learn_ok = False
    checks["learning converges on a gapped window"] = gap_learn_ok
    print(f"  exact P(m)       = {dict(zip(exact_values.tolist(), exact_prob.tolist()))}")
    print(f"  estimated P(m)   = {dict(zip(built['c_vals'].tolist(), built['c_prob'].tolist()))}")
    print(f"  contact TVD      = {total_variation:.6g}")
    print(f"  mean Rg rel.err  = {relative_rg_error:.6g}")
    for description, passed in checks.items():
        print(f"  {'PASS' if passed else 'FAIL'}: {description}")
    if all(checks.values()):
        print("SELF-TEST PASSED")
        return 0
    print("SELF-TEST FAILED")
    return 1


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate a single-chain athermal baseline consistently using "
        "Wang-Landau contact flattening plus fixed-weight production reweighting.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--N", type=int, default=44, help="number of beads")
    parser.add_argument(
        "--m_max", type=int, default=None,
        help="exact geometric contact maximum; N=30, 44, and 60 use verified values automatically",
    )
    parser.add_argument(
        "--m_min", type=int, default=0,
        help="lowest contact level; currently must be 0 so the straight initializer is in-window",
    )
    parser.add_argument(
        "--m_flat", type=int, default=None,
        help="last tier-2 level; otherwise derived from the molecular target",
    )
    parser.add_argument(
        "--m_cover", type=int, default=None,
        help="last included level; defaults to the exact geometric maximum",
    )
    parser.add_argument(
        "--flat_tail_threshold", type=float, default=1e-3,
        help="worst-temperature target tail allowed above the tier-2 boundary",
    )
    parser.add_argument(
        "--cover_tail_threshold", type=float, default=0.0,
        help="target-tail threshold for explicit truncation; 0 keeps the full window",
    )
    parser.add_argument(
        "--target_npz", type=str, default=None,
        help="molecular REMD target used to derive contact tiers",
    )
    parser.add_argument("--T", type=float, default=1.0, help="metadata only; target is athermal")
    parser.add_argument("--eps", type=float, default=0.0, help="metadata only; target is athermal")
    parser.add_argument("--dist_dir", type=str, default="dists", help="output directory")
    parser.add_argument("--rg_bins", type=int, default=60, help="number of Rg bins")
    parser.add_argument("--no_joint", action="store_true", help="omit joint P(m,Rg)")
    parser.add_argument("--base_seed", type=int, default=42, help="production seed base")
    parser.add_argument("--wl_seed", type=int, default=1729, help="WL learning seed")
    parser.add_argument("--n_workers", type=int, default=12, help="fixed-weight production workers")
    parser.add_argument(
        "--steps_per_worker", type=int, default=400_000_000,
        help="fixed-weight production attempts per worker",
    )
    parser.add_argument("--burnin", type=float, default=0.3, help="production burn-in fraction")
    parser.add_argument("--sample_every", type=int, default=500, help="production sampling interval")
    parser.add_argument("--wl_initial_log_f", type=float, default=1.0)
    parser.add_argument("--wl_final_log_f", type=float, default=1e-4)
    parser.add_argument("--wl_flatness", type=float, default=0.8)
    parser.add_argument("--wl_min_visits", type=int, default=1000)
    parser.add_argument("--wl_min_cover_visits", type=int, default=50)
    parser.add_argument("--wl_check_every", type=int, default=100_000)
    parser.add_argument("--wl_max_steps", type=int, default=10_000_000_000)
    parser.add_argument("--wl_max_seconds", type=float, default=21_600.0)
    parser.add_argument("--wl_max_steps_per_stage", type=int, default=1_000_000_000)
    parser.add_argument(
        "--wl_schedule", choices=("halving", "one_over_t"), default="halving",
        help="density-refinement schedule",
    )
    parser.add_argument(
        "--checkpoint_every_seconds", type=float, default=1800.0,
        help="checkpoint timer within a WL stage",
    )
    parser.add_argument(
        "--show_contact_upper_bound", action="store_true",
        help="print the rigorous analytical upper bound and verified exact m_max, then exit",
    )
    parser.add_argument(
        "--excluded_contact_levels", type=int, nargs="*", default=[],
        help="verified unreachable internal contact levels to omit from flatness "
             "and production-support checks",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="write a restartable NPZ after every completed WL refinement stage",
    )
    parser.add_argument(
        "--resume_checkpoint", type=str, default=None,
        help="resume DOS learning from a checkpoint (histogram and RNG stream restart)",
    )
    parser.add_argument(
        "--min_production_round_trips", type=int, default=1,
        help="minimum summed low-high-low round trips; zero disables the hard check",
    )
    parser.add_argument(
        "--min_production_samples_per_level", type=int, default=1,
        help="minimum recorded fixed-weight samples in every required contact level",
    )
    parser.add_argument(
        "--save_raw_samples", dest="save_raw_samples", action="store_true", default=True,
        help="store an athermal importance-resampled c/Rg/bend sample set",
    )
    parser.add_argument("--no_save_raw_samples", dest="save_raw_samples", action="store_false")
    parser.add_argument(
        "--legacy_sample_aliases", action="store_true",
        help="also write deprecated c_samples/rg_samples/bend_samples aliases",
    )
    parser.add_argument(
        "--n_blocks", type=int, default=20,
        help="non-overlapping batch-means blocks per production worker",
    )
    parser.add_argument("--self-test", action="store_true", help="run exact small-chain validation")
    return parser.parse_args(argv)


def validate_common_args(args: argparse.Namespace) -> None:
    if args.N < 3:
        raise ValueError("--N must be at least 3")
    if args.m_min != 0:
        raise ValueError("this implementation requires --m_min=0")


def resolve_m_max(args: argparse.Namespace) -> None:
    verified_maximum = geometric_contact_maximum(args.N)
    if verified_maximum is not None:
        if args.m_max is not None and args.m_max != verified_maximum:
            raise ValueError(
                f"--m_max={args.m_max} conflicts with the verified exact geometric "
                f"maximum {verified_maximum} for N={args.N}"
            )
        args.m_max = verified_maximum
    elif args.m_max is None:
        raise ValueError(
            "--m_max is required for chain lengths without an encoded exact "
            "geometric maximum"
        )


def resolve_contact_tiers(args: argparse.Namespace) -> Dict[str, Any]:
    """Resolve target-derived tier boundaries and truncation diagnostics."""
    default_target = Path(__file__).resolve().with_name(
        f"remd_distributions_{args.N}mer.npz"
    )
    target_path = Path(args.target_npz) if args.target_npz else default_target
    explicit_target = args.target_npz is not None
    target = None
    target_report = None
    if target_path.exists() and args.N in CONTACT_OFFSETS:
        target = load_target_contact_support(target_path, CONTACT_OFFSETS[args.N])
        target_report = support_report(target, args.m_max)
    elif explicit_target and not target_path.exists():
        raise FileNotFoundError(f"target REMD NPZ does not exist: {target_path}")

    if args.m_flat is not None:
        m_flat = int(args.m_flat)
    elif target is not None:
        m_flat = flat_level(target, args.flat_tail_threshold, args.m_max)
        if not tail_mass_above(target, m_flat) < args.flat_tail_threshold:
            print(
                "WARNING: the requested flat-tail threshold is not reached before "
                f"the geometric maximum; m_flat is clamped to {args.m_max}.",
                flush=True,
            )
    else:
        m_flat = int(args.m_max)
        print(
            "WARNING: no usable target REMD NPZ was found and --m_flat was not "
            "provided; flattening the full geometric contact window.",
            flush=True,
        )

    if args.m_cover is not None:
        m_cover = int(args.m_cover)
    elif args.cover_tail_threshold == 0.0:
        m_cover = int(args.m_max)
    elif target is not None:
        m_cover = flat_level(target, args.cover_tail_threshold, args.m_max)
        if not tail_mass_above(target, m_cover) < args.cover_tail_threshold:
            print(
                "WARNING: the requested cover-tail threshold is not reached before "
                f"the geometric maximum; m_cover is clamped to {args.m_max}.",
                flush=True,
            )
    else:
        m_cover = int(args.m_max)
        print(
            "WARNING: no usable target REMD NPZ was found; m_cover defaults to "
            "the geometric maximum.",
            flush=True,
        )

    tier = make_contact_tiers(
        args.m_min,
        args.m_max,
        m_flat,
        m_cover,
        args.excluded_contact_levels,
    )
    declared_truncation = m_cover < args.m_max
    truncation_mass = (
        tail_mass_above(target, m_cover)
        if declared_truncation and target is not None else float("nan")
    )
    if declared_truncation:
        target_text = (
            f"{100.0 * truncation_mass:.6f}% worst-temperature target mass"
            if math.isfinite(truncation_mass) else "unknown target mass"
        )
        print("=" * 72, flush=True)
        print(
            f"DECLARED TRUNCATION: levels {m_cover + 1}..{args.m_max} are excluded; "
            f"omitted {target_text}.",
            flush=True,
        )
        print(
            "The omitted athermal mass is not estimable from this conditional run.",
            flush=True,
        )
        print("=" * 72, flush=True)

    args.m_flat = m_flat
    args.m_cover = m_cover
    return {
        "tier": tier,
        "target_path": target_path,
        "target": target,
        "target_report": target_report,
        "declared_truncation": declared_truncation,
        "declared_truncation_target_mass": truncation_mass,
    }


def validate_args(args: argparse.Namespace) -> None:
    validate_common_args(args)
    resolve_m_max(args)
    if args.m_max < args.m_min:
        raise ValueError("--m_max must be nonnegative")
    rigorous_bound = contact_upper_bound(args.N)
    if args.m_max > rigorous_bound:
        raise ValueError(
            f"--m_max={args.m_max} exceeds the cubic-lattice contact upper bound "
            f"{rigorous_bound} for N={args.N}"
        )
    if args.n_workers < 1 or args.steps_per_worker < 1:
        raise ValueError("--n_workers and --steps_per_worker must be positive")
    if args.rg_bins < 1 or args.sample_every < 1:
        raise ValueError("--rg_bins and --sample_every must be positive")
    if not 0.0 <= args.burnin < 1.0:
        raise ValueError("--burnin must lie in [0,1)")
    expected = (args.steps_per_worker - int(round(args.burnin * args.steps_per_worker))) // args.sample_every
    if expected < 1:
        raise ValueError("production settings yield zero samples per worker")
    if not (args.wl_initial_log_f > args.wl_final_log_f > 0.0):
        raise ValueError("require --wl_initial_log_f > --wl_final_log_f > 0")
    if not 0.0 < args.wl_flatness <= 1.0:
        raise ValueError("--wl_flatness must lie in (0,1]")
    if min(
        args.wl_min_visits,
        args.wl_min_cover_visits,
        args.wl_check_every,
        args.wl_max_steps,
        args.wl_max_steps_per_stage,
    ) < 1:
        raise ValueError("WL visit/check/step controls must be positive")
    if args.wl_max_seconds <= 0.0 or args.checkpoint_every_seconds <= 0.0:
        raise ValueError("WL time controls must be positive")
    if not 0.0 < args.flat_tail_threshold < 1.0:
        raise ValueError("--flat_tail_threshold must lie in (0,1)")
    if not 0.0 <= args.cover_tail_threshold < 1.0:
        raise ValueError("--cover_tail_threshold must lie in [0,1)")
    excluded = list(args.excluded_contact_levels)
    if len(set(excluded)) != len(excluded):
        raise ValueError("--excluded_contact_levels contains duplicates")
    if args.min_production_round_trips < 0:
        raise ValueError("--min_production_round_trips must be nonnegative")
    if args.min_production_samples_per_level < 1:
        raise ValueError("--min_production_samples_per_level must be positive")
    if args.n_blocks < 2:
        raise ValueError("--n_blocks must be at least 2")
    if args.legacy_sample_aliases and not args.save_raw_samples:
        raise ValueError("--legacy_sample_aliases requires raw-sample saving")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        return run_self_test()
    if args.show_contact_upper_bound:
        validate_common_args(args)
        return report_contact_upper_bound(args)
    validate_args(args)
    checkpoint = Path(args.checkpoint) if args.checkpoint else None
    resume = Path(args.resume_checkpoint) if args.resume_checkpoint else None

    window = np.arange(args.m_min, args.m_max + 1, dtype=np.int64)
    tier_info = resolve_contact_tiers(args)
    tier = np.asarray(tier_info["tier"], dtype=np.int8)
    active = tier > TIER_EXCLUDED
    coverage_text = (
        f"{args.m_flat + 1}..{args.m_cover}"
        if args.m_cover > args.m_flat else "none"
    )
    excluded_text = (
        f"{args.m_cover + 1}..{args.m_max}"
        if args.m_cover < args.m_max else "none"
    )
    print(
        f"Contact tiers: flat={args.m_min}..{args.m_flat}, "
        f"coverage={coverage_text}, excluded_above={excluded_text}",
        flush=True,
    )
    if args.excluded_contact_levels:
        print(
            "Verified internal contact gaps excluded from convergence checks: "
            f"{sorted(args.excluded_contact_levels)}",
            flush=True,
        )

    print("=== Wang-Landau learning (adaptive samples are discarded) ===", flush=True)
    learned = learn_log_density(
        n_beads=args.N,
        m_min=args.m_min,
        m_max=args.m_max,
        seed=args.wl_seed,
        initial_log_f=args.wl_initial_log_f,
        final_log_f=args.wl_final_log_f,
        flatness=args.wl_flatness,
        min_visits=args.wl_min_visits,
        min_cover_visits=args.wl_min_cover_visits,
        check_every=args.wl_check_every,
        max_steps=args.wl_max_steps,
        max_seconds=args.wl_max_seconds,
        max_steps_per_stage=args.wl_max_steps_per_stage,
        schedule=args.wl_schedule,
        checkpoint_every_seconds=args.checkpoint_every_seconds,
        tier=tier,
        checkpoint=checkpoint,
        resume_checkpoint=resume,
    )
    print(
        f"WL complete: stages={learned['stages_completed']} "
        f"steps={learned['attempted_steps']} round_trips={learned['round_trips']} "
        f"wall={learned['wall_time']:.1f}s",
        flush=True,
    )

    worker_seeds = [args.base_seed + i for i in range(args.n_workers)]
    print("=== Frozen-weight production ===", flush=True)
    t0 = time.time()
    results: List[Dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.n_workers) as executor:
        futures = [
            executor.submit(
                run_production_chain,
                i,
                worker_seeds[i],
                learned["chain"],
                learned["log_g"],
                args.m_min,
                args.m_max,
                args.steps_per_worker,
                args.burnin,
                args.sample_every,
                True,
                tier,
            )
            for i in range(args.n_workers)
        ]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(
                f"production worker {result['worker_id']} complete: "
                f"samples={result['contact_samples'].size} "
                f"round_trips={result['round_trips']}",
                flush=True,
            )
    results.sort(key=lambda result: result["worker_id"])

    contacts = np.concatenate([result["contact_samples"] for result in results])
    radii = np.concatenate([result["rg_samples"] for result in results])
    bends = np.concatenate([result["bend_samples"] for result in results])
    production_counts_full = np.bincount(
        contacts - args.m_min, minlength=window.size
    )[:window.size]
    encountered_excluded = window[(tier == TIER_EXCLUDED) & (production_counts_full > 0)]
    if encountered_excluded.size:
        raise RuntimeError(
            "fixed-weight production encountered contact levels listed as excluded: "
            f"{encountered_excluded.tolist()}; output was not written"
        )
    tier2_minimum = max(args.wl_min_visits, args.min_production_samples_per_level)
    tier1_minimum = max(
        args.wl_min_cover_visits, args.min_production_samples_per_level
    )
    deficient_tier2 = window[
        (tier == TIER_FLAT) & (production_counts_full < tier2_minimum)
    ]
    deficient_tier1 = window[
        (tier == TIER_COVERAGE) & (production_counts_full < tier1_minimum)
    ]
    if deficient_tier2.size or deficient_tier1.size:
        raise RuntimeError(
            "fixed-weight production did not adequately sample the tiered contact "
            f"window: tier 2 deficient={deficient_tier2.tolist()} "
            f"(minimum {tier2_minimum}), tier 1 deficient="
            f"{deficient_tier1.tolist()} (minimum {tier1_minimum}); "
            "output was not written"
        )
    built = build_distributions(
        contacts,
        radii,
        bends,
        learned["log_g"],
        args.m_min,
        args.rg_bins,
        args.no_joint,
        n_beads=args.N,
        m_cover=args.m_cover,
    )
    c_blocked_stderr, blocked_batch_count = blocked_contact_stderr(
        [result["contact_samples"] for result in results],
        learned["log_g"],
        args.m_min,
        args.m_cover,
        args.n_blocks,
    )
    built["c_blocked_stderr"] = c_blocked_stderr
    weights = built.pop("weights")
    ess = effective_sample_size(weights)
    total_round_trips = sum(result["round_trips"] for result in results)
    if total_round_trips < args.min_production_round_trips:
        raise RuntimeError(
            f"fixed-weight production made {total_round_trips} contact round trips, "
            f"below required {args.min_production_round_trips}; output was not written"
        )

    production_max_contact = int(contacts.max())

    accepted_per_worker = np.asarray([r["accepted_moves"] for r in results], dtype=np.int64)
    valid_per_worker = np.asarray([r["geometrically_valid_moves"] for r in results], dtype=np.int64)
    attempted_per_worker = np.asarray([r["attempted_moves"] for r in results], dtype=np.int64)
    samples_per_worker = np.asarray([r["contact_samples"].size for r in results], dtype=np.int64)
    acceptance_per_worker = accepted_per_worker / attempted_per_worker
    total_attempted = int(attempted_per_worker.sum())
    total_accepted = int(accepted_per_worker.sum())

    per_worker_m_mean: List[float] = []
    per_worker_rg_mean: List[float] = []
    offset = 0
    for count in samples_per_worker:
        sl = slice(offset, offset + int(count))
        local_weights = weights[sl] / weights[sl].sum()
        per_worker_m_mean.append(float(np.dot(local_weights, contacts[sl])))
        per_worker_rg_mean.append(float(np.dot(local_weights, radii[sl])))
        offset += int(count)

    dist_dir = Path(args.dist_dir)
    dist_dir.mkdir(parents=True, exist_ok=True)
    output_path = dist_dir / (
        f"{Path(__file__).stem}_N{args.N}_workers{args.n_workers}"
        f"_steps{args.steps_per_worker}_seed{args.base_seed}.npz"
    )
    save: Dict[str, Any] = {
        key: value for key, value in built.items()
        if key not in {"marginal_m_error", "marginal_rg_error"}
    }
    save.update(
        N=int(args.N), T=float(args.T), eps=float(args.eps),
        n_workers=int(args.n_workers), steps_per_worker=int(args.steps_per_worker),
        total_attempted_steps=total_attempted, base_seed=int(args.base_seed),
        worker_seeds=np.asarray(worker_seeds, dtype=np.int64),
        burnin=float(args.burnin), sample_every=int(args.sample_every),
        n_samples=int(contacts.size), accepted_moves_per_worker=accepted_per_worker,
        acceptance_ratios_per_worker=acceptance_per_worker,
        combined_acceptance_ratio=float(total_accepted / total_attempted),
        rg_bins=int(np.asarray(built["rg_edges"]).size - 1),
        requested_rg_bins=int(args.rg_bins), samples_per_worker=samples_per_worker,
        per_worker_m_mean=np.asarray(per_worker_m_mean),
        per_worker_rg_mean=np.asarray(per_worker_rg_mean),
        kappa_bend=0.0, bending_enabled=False, bend_definition=BEND_DEFINITION,
        sampler="wang_landau_fixed_weight_reweighted",
        raw_samples_resampled=True,
        raw_samples_available=bool(args.save_raw_samples),
        raw_samples_warning=RAW_SAMPLES_WARNING,
        raw_samples_unique_fraction=float("nan"),
        n_blocks_per_worker=int(args.n_blocks),
        c_blocked_batch_count=int(blocked_batch_count),
        wl_m_values=np.arange(args.m_min, args.m_max + 1, dtype=np.int64),
        wl_log_g=np.asarray(learned["log_g"], dtype=np.float64),
        wl_seed=int(args.wl_seed), wl_initial_log_f=float(args.wl_initial_log_f),
        wl_final_log_f=float(args.wl_final_log_f),
        wl_flatness=float(args.wl_flatness), wl_min_visits=int(args.wl_min_visits),
        wl_min_cover_visits=int(args.wl_min_cover_visits),
        wl_max_steps=int(args.wl_max_steps),
        wl_max_seconds=float(args.wl_max_seconds),
        wl_max_steps_per_stage=int(args.wl_max_steps_per_stage),
        wl_schedule=str(args.wl_schedule),
        checkpoint_every_seconds=float(args.checkpoint_every_seconds),
        wl_learning_steps=int(learned["attempted_steps"]),
        wl_learning_accepted_moves=int(learned["accepted_moves"]),
        wl_learning_round_trips=int(learned["round_trips"]),
        wl_learning_wall_seconds=float(learned["wall_time"]),
        wl_one_over_t_mode=bool(learned["one_over_t_mode"]),
        wl_stages_completed=int(learned["stages_completed"]),
        wl_stage_records=np.asarray(learned["stage_records"]),
        production_geometrically_valid_per_worker=valid_per_worker,
        production_round_trips_per_worker=np.asarray(
            [result["round_trips"] for result in results], dtype=np.int64
        ),
        importance_effective_sample_size=float(ess),
        importance_effective_fraction=float(ess / contacts.size),
        n_samples_effective=float(ess),
        production_max_contact=production_max_contact,
        excluded_contact_levels=np.asarray(
            sorted(args.excluded_contact_levels), dtype=np.int64
        ),
        min_production_samples_per_level=int(args.min_production_samples_per_level),
        production_samples_per_level=production_counts_full.astype(np.int64),
        wl_tier=tier,
        wl_active_levels=np.asarray(learned["active"], dtype=bool),
        m_flat=int(args.m_flat), m_cover=int(args.m_cover),
        flat_tail_threshold=float(args.flat_tail_threshold),
        cover_tail_threshold=float(args.cover_tail_threshold),
        target_support_npz=str(tier_info["target_path"]),
        declared_truncation=bool(tier_info["declared_truncation"]),
        declared_truncation_target_mass=float(
            tier_info["declared_truncation_target_mass"]
        ),
        joint_contact_marginal_error=float(built["marginal_m_error"]),
        joint_rg_marginal_error=float(built["marginal_rg_error"]),
    )
    if args.save_raw_samples:
        resampled_indices = systematic_resample(
            weights, np.random.default_rng(args.base_seed + 10_000_019)
        )
        save["raw_samples_unique_fraction"] = float(
            np.unique(resampled_indices).size / resampled_indices.size
        )
        save["c_samples_resampled"] = contacts[resampled_indices]
        save["rg_samples_resampled"] = radii[resampled_indices]
        save["bend_samples_resampled"] = bends[resampled_indices]
        if args.legacy_sample_aliases:
            save["c_samples"] = save["c_samples_resampled"]
            save["rg_samples"] = save["rg_samples_resampled"]
            save["bend_samples"] = save["bend_samples_resampled"]
    np.savez_compressed(output_path, **save)

    print("\n=== Reweighted athermal result ===")
    print(f"P(m) sum                         : {built['c_prob'].sum():.12g}")
    print(f"P(Rg) sum                        : {built['rg_prob'].sum():.12g}")
    print(f"joint sum                        : {built['crg_prob'].sum():.12g}")
    print(f"joint -> P(m) max error          : {built['marginal_m_error']:.3e}")
    print(f"joint -> P(Rg) max error         : {built['marginal_rg_error']:.3e}")
    print(f"importance ESS                   : {ess:.1f}/{contacts.size}")
    print(
        "min production visits per included level: "
        f"{int(production_counts_full[active].min())}"
    )
    print(
        "largest naive count diagnostic  : "
        f"{float(built['c_naive_count_error'].max()):.4f} "
        f"at m={int(built['c_vals'][int(np.argmax(built['c_naive_count_error']))])}"
    )
    print(f"highest sampled contact level    : {production_max_contact} (window {args.m_max})")
    print(f"production round trips           : {total_round_trips}")
    print(f"combined production acceptance   : {total_accepted / total_attempted:.4f}")
    print(f"production wall time             : {time.time() - t0:.1f}s")
    print(f"DIST_FILE = {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
