#!/usr/bin/env python3
"""Wang-Landau baseline sampler for one self-avoiding lattice chain.

The physical target remains the uniform (athermal) measure over self-avoiding
conformations.  Wang-Landau adaptation is used only to learn a contact-number
bias.  The adaptive trajectory is discarded.  Independent production chains
then run with the frozen weight ``W(m) = exp(-log_g_hat[m])`` and every recorded
sample is reweighted by ``1/W(m) = exp(log_g_hat[m])``.

This separation matters: even an imperfect ``log_g_hat`` gives a consistent
athermal estimator when the frozen-weight production simulation is equilibrated
and covers the requested contact window.  A better estimate merely improves
mixing and statistical efficiency.

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
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import permutations, product
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np


Vec = Tuple[int, int, int]
Matrix = Tuple[Tuple[int, int, int], Tuple[int, int, int], Tuple[int, int, int]]
NN_VECS: Tuple[Vec, ...] = (
    (1, 0, 0), (-1, 0, 0), (0, 1, 0),
    (0, -1, 0), (0, 0, 1), (0, 0, -1),
)
BEND_DEFINITION = "number of 90-degree turns among the N-2 internal vertices"


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
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(
            temporary,
            checkpoint_version=np.array(1, dtype=np.int64),
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
        )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


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
    checkpoint: Optional[Path] = None,
    resume_checkpoint: Optional[Path] = None,
    progress: bool = True,
) -> Dict[str, Any]:
    """Learn a contact density estimate with the original WL refinement rule."""
    if resume_checkpoint is None:
        chain: List[Vec] = [(i, 0, 0) for i in range(n_beads)]
        log_g = np.zeros(m_max - m_min + 1, dtype=np.float64)
        log_f = float(initial_log_f)
        stages_completed = 0
        total_steps = 0
        accepted = 0
        previous_round_trips = 0
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
        validate_chain(chain)

    occupied = set(chain)
    contact = contact_count(chain, occupied)
    if not (m_min <= contact <= m_max):
        raise ValueError(
            f"initial/checkpoint chain has m={contact}, outside [{m_min}, {m_max}]"
        )
    rng = random.Random(seed + 1_000_003 * stages_completed)
    tracker = RoundTripCounter(m_min, m_max)
    tracker.observe(contact)
    stage_records: List[Tuple[float, int, float, int, int]] = []
    t0 = time.time()

    while log_f > final_log_f:
        histogram = np.zeros_like(log_g, dtype=np.int64)
        stage_steps = 0
        stage_accepted = 0
        while True:
            block = min(check_every, max_steps - total_steps)
            if block <= 0:
                missing = np.flatnonzero(histogram < min_visits) + m_min
                raise RuntimeError(
                    "Wang-Landau learning exhausted --wl_max_steps before "
                    f"convergence at log_f={log_f:.3g}. Contact levels below "
                    f"--wl_min_visits in the current stage: {missing.tolist()}. "
                    "Increase the limit, lower --m_max only if the omitted tail is "
                    "scientifically irrelevant, or verify that every requested "
                    "integer contact level is geometrically reachable."
                )
            for _ in range(block):
                chain, occupied, contact, _, was_accepted = metropolis_step(
                    chain, occupied, contact, log_g, m_min, m_max, rng
                )
                total_steps += 1
                stage_steps += 1
                if was_accepted:
                    accepted += 1
                    stage_accepted += 1
                index = contact - m_min
                log_g[index] += log_f
                histogram[index] += 1
                tracker.observe(contact)

            mean_count = float(histogram.mean())
            minimum_count = int(histogram.min())
            ratio = minimum_count / mean_count if mean_count else 0.0
            range_covered = bool(histogram[0] > 0 and histogram[-1] > 0)
            is_flat = (
                range_covered
                and minimum_count >= min_visits
                and ratio >= flatness
            )
            if progress:
                print(
                    f"[WL stage {stages_completed + 1}] log_f={log_f:.6g} "
                    f"steps={stage_steps} min/mean={ratio:.3f} "
                    f"min_visits={minimum_count} range={range_covered} "
                    f"round_trips={previous_round_trips + tracker.round_trips}",
                    flush=True,
                )
            if is_flat:
                stage_records.append(
                    (log_f, stage_steps, ratio, minimum_count, stage_accepted)
                )
                stages_completed += 1
                log_f *= 0.5
                # The DOS is defined only up to an additive constant.  Centering
                # every stage avoids loss of precision in long learning runs.
                log_g -= log_g[0]
                if checkpoint is not None:
                    _save_checkpoint(
                        checkpoint,
                        n_beads=n_beads,
                        m_min=m_min,
                        m_max=m_max,
                        chain=chain,
                        log_g=log_g,
                        next_log_f=log_f,
                        stages_completed=stages_completed,
                        attempted_steps=total_steps,
                        accepted_moves=accepted,
                        round_trips=previous_round_trips + tracker.round_trips,
                        seed=seed,
                    )
                break

    log_g -= log_g[0]
    return {
        "log_g": log_g,
        "chain": chain,
        "attempted_steps": total_steps,
        "accepted_moves": accepted,
        "stages_completed": stages_completed,
        "stage_records": np.asarray(stage_records, dtype=np.float64).reshape((-1, 5)),
        "round_trips": previous_round_trips + tracker.round_trips,
        "wall_time": time.time() - t0,
        "next_log_f": log_f,
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
) -> Dict[str, Any]:
    rng = random.Random(seed)
    chain = [tuple(site) for site in initial_chain]
    occupied = set(chain)
    contact = contact_count(chain, occupied)
    burn_steps = int(round(burnin * steps))
    contacts: List[int] = []
    radii: List[float] = []
    bends: List[int] = []
    accepted = 0
    geometrically_valid = 0
    tracker = RoundTripCounter(m_min, m_max)
    t0 = time.time()
    progress_mark = max(1, steps // 10)

    for step in range(1, steps + 1):
        chain, occupied, contact, valid, was_accepted = metropolis_step(
            chain, occupied, contact, log_g, m_min, m_max, rng
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


def build_distributions(
    contacts: np.ndarray,
    radii: np.ndarray,
    bends: np.ndarray,
    log_g: np.ndarray,
    m_min: int,
    rg_bins: int,
    no_joint: bool,
) -> Dict[str, Any]:
    if contacts.size == 0 or contacts.size != radii.size or contacts.size != bends.size:
        raise RuntimeError("production sample arrays are empty or have inconsistent lengths")
    if not np.all(np.isfinite(radii)):
        raise RuntimeError("non-finite Rg production samples")
    weights = normalized_importance_weights(contacts, log_g, m_min)
    c_min = int(contacts.min())
    c_max = int(contacts.max())
    c_edges = np.arange(c_min - 0.5, c_max + 1.5, 1.0, dtype=np.float64)
    c_full = np.arange(c_min, c_max + 1, dtype=np.int64)
    c_mass = np.bincount(contacts - c_min, weights=weights, minlength=c_full.size)
    positive = c_mass > 0.0
    c_vals = c_full[positive]
    c_prob = c_mass[positive]
    c_prob /= c_prob.sum()

    rg_min, rg_max = float(radii.min()), float(radii.max())
    pad = 1e-9 if rg_max <= rg_min else 0.02 * (rg_max - rg_min)
    rg_edges = np.linspace(rg_min - pad, rg_max + pad, rg_bins + 1)
    rg_prob, _ = np.histogram(radii, bins=rg_edges, weights=weights)
    rg_prob = rg_prob.astype(np.float64)
    rg_prob /= rg_prob.sum()

    if no_joint:
        crg_prob = np.array([[]], dtype=np.float64)
    else:
        crg_prob, _, _ = np.histogram2d(
            contacts.astype(np.float64), radii, bins=(c_edges, rg_edges), weights=weights
        )
        crg_prob = crg_prob.astype(np.float64)
        crg_prob /= crg_prob.sum()

    bend_vals = np.unique(bends)
    bend_mass = np.array([weights[bends == value].sum() for value in bend_vals])
    bend_mass /= bend_mass.sum()
    mean_bends = float(np.dot(weights, bends))

    if no_joint:
        marginal_m_error = float("nan")
        marginal_rg_error = float("nan")
    else:
        marginal_m_error = float(np.max(np.abs(crg_prob.sum(axis=1) - c_mass)))
        marginal_rg_error = float(np.max(np.abs(crg_prob.sum(axis=0) - rg_prob)))

    return {
        "weights": weights,
        "c_vals": c_vals,
        "c_prob": c_prob,
        "c_edges": c_edges,
        "rg_edges": rg_edges,
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
    built = build_distributions(contacts, radii, bends, learned["log_g"], m_min, 24, False)
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
        description="Estimate an unbiased single-chain athermal baseline using "
        "Wang-Landau contact flattening plus fixed-weight production reweighting.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--N", type=int, default=44, help="number of beads")
    parser.add_argument(
        "--m_max", type=int, default=None,
        help="highest contact level in the sampled window; required outside --self-test",
    )
    parser.add_argument(
        "--m_min", type=int, default=0,
        help="lowest contact level; currently must be 0 so the straight initializer is in-window",
    )
    parser.add_argument("--T", type=float, default=1.0, help="metadata only; target is athermal")
    parser.add_argument("--eps", type=float, default=0.0, help="metadata only; target is athermal")
    parser.add_argument("--dist_dir", type=str, default="dists", help="output directory")
    parser.add_argument("--rg_bins", type=int, default=60, help="number of Rg bins")
    parser.add_argument("--no_joint", action="store_true", help="omit joint P(m,Rg)")
    parser.add_argument("--base_seed", type=int, default=42, help="production seed base")
    parser.add_argument("--wl_seed", type=int, default=1729, help="WL learning seed")
    parser.add_argument("--n_workers", type=int, default=8, help="fixed-weight production workers")
    parser.add_argument(
        "--steps_per_worker", type=int, default=20_000_000,
        help="fixed-weight production attempts per worker",
    )
    parser.add_argument("--burnin", type=float, default=0.3, help="production burn-in fraction")
    parser.add_argument("--sample_every", type=int, default=1000, help="production sampling interval")
    parser.add_argument("--wl_initial_log_f", type=float, default=1.0)
    parser.add_argument("--wl_final_log_f", type=float, default=1e-6)
    parser.add_argument("--wl_flatness", type=float, default=0.8)
    parser.add_argument("--wl_min_visits", type=int, default=1000)
    parser.add_argument("--wl_check_every", type=int, default=100_000)
    parser.add_argument("--wl_max_steps", type=int, default=500_000_000)
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
        "--save_raw_samples", dest="save_raw_samples", action="store_true", default=True,
        help="store an athermal importance-resampled c/Rg/bend sample set",
    )
    parser.add_argument("--no_save_raw_samples", dest="save_raw_samples", action="store_false")
    parser.add_argument("--self-test", action="store_true", help="run exact small-chain validation")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if args.N < 3:
        raise ValueError("--N must be at least 3")
    if args.m_max is None:
        raise ValueError("--m_max is required; choose it from the contact support you need")
    if args.m_min != 0 or args.m_max < args.m_min:
        raise ValueError("this implementation requires --m_min=0 and --m_max >= 0")
    rigorous_bound = 2 * args.N + 1
    if args.m_max > rigorous_bound:
        raise ValueError(
            f"--m_max={args.m_max} exceeds the cubic-lattice degree bound {rigorous_bound}"
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
    if min(args.wl_min_visits, args.wl_check_every, args.wl_max_steps) < 1:
        raise ValueError("WL visit/check/step controls must be positive")
    if args.min_production_round_trips < 0:
        raise ValueError("--min_production_round_trips must be nonnegative")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        return run_self_test()
    validate_args(args)
    checkpoint = Path(args.checkpoint) if args.checkpoint else None
    resume = Path(args.resume_checkpoint) if args.resume_checkpoint else None

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
        check_every=args.wl_check_every,
        max_steps=args.wl_max_steps,
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
    built = build_distributions(
        contacts, radii, bends, learned["log_g"], args.m_min, args.rg_bins, args.no_joint
    )
    weights = built.pop("weights")
    ess = effective_sample_size(weights)
    total_round_trips = sum(result["round_trips"] for result in results)
    if total_round_trips < args.min_production_round_trips:
        raise RuntimeError(
            f"fixed-weight production made {total_round_trips} contact round trips, "
            f"below required {args.min_production_round_trips}; output was not written"
        )

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
        rg_bins=int(args.rg_bins), samples_per_worker=samples_per_worker,
        per_worker_m_mean=np.asarray(per_worker_m_mean),
        per_worker_rg_mean=np.asarray(per_worker_rg_mean),
        kappa_bend=0.0, bending_enabled=False, bend_definition=BEND_DEFINITION,
        sampler="wang_landau_fixed_weight_reweighted",
        raw_samples_resampled=bool(args.save_raw_samples),
        wl_m_values=np.arange(args.m_min, args.m_max + 1, dtype=np.int64),
        wl_log_g=np.asarray(learned["log_g"], dtype=np.float64),
        wl_seed=int(args.wl_seed), wl_initial_log_f=float(args.wl_initial_log_f),
        wl_final_log_f=float(args.wl_final_log_f),
        wl_flatness=float(args.wl_flatness), wl_min_visits=int(args.wl_min_visits),
        wl_learning_steps=int(learned["attempted_steps"]),
        wl_learning_accepted_moves=int(learned["accepted_moves"]),
        wl_learning_round_trips=int(learned["round_trips"]),
        wl_stages_completed=int(learned["stages_completed"]),
        wl_stage_records=np.asarray(learned["stage_records"]),
        production_geometrically_valid_per_worker=valid_per_worker,
        production_round_trips_per_worker=np.asarray(
            [result["round_trips"] for result in results], dtype=np.int64
        ),
        importance_effective_sample_size=float(ess),
        importance_effective_fraction=float(ess / contacts.size),
        joint_contact_marginal_error=float(built["marginal_m_error"]),
        joint_rg_marginal_error=float(built["marginal_rg_error"]),
    )
    if args.save_raw_samples:
        indices = systematic_resample(weights, np.random.default_rng(args.base_seed + 10_000_019))
        save["c_samples"] = contacts[indices]
        save["rg_samples"] = radii[indices]
        save["bend_samples"] = bends[indices]
    np.savez_compressed(output_path, **save)

    print("\n=== Unbiased athermal result ===")
    print(f"P(m) sum                         : {built['c_prob'].sum():.12g}")
    print(f"P(Rg) sum                        : {built['rg_prob'].sum():.12g}")
    print(f"joint sum                        : {built['crg_prob'].sum():.12g}")
    print(f"joint -> P(m) max error          : {built['marginal_m_error']:.3e}")
    print(f"joint -> P(Rg) max error         : {built['marginal_rg_error']:.3e}")
    print(f"importance ESS                   : {ess:.1f}/{contacts.size}")
    print(f"production round trips           : {total_round_trips}")
    print(f"combined production acceptance   : {total_accepted / total_attempted:.4f}")
    print(f"production wall time             : {time.time() - t0:.1f}s")
    print(f"DIST_FILE = {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
