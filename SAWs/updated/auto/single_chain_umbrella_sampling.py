#!/usr/bin/env python3
"""Replica-exchange umbrella sampler for an athermal lattice polymer.

The physical target is the uniform measure over self-avoiding walks.  The only
energy used while sampling is an artificial, dimensionless harmonic bias in the
non-bonded contact count ``m``::

    B_j(m) = 0.5 * k * (m - m0_j)**2

The default ``k=0.30`` and center spacing ``3`` were selected from the existing
N=30, 44, and 60 athermal contact distributions to give roughly 25 percent
adjacent replica-exchange acceptance.  Centers continue until the last center
is at or above the geometric contact maximum; a center just beyond the maximum
is intentional because it pulls probability toward the compact endpoint.

Each umbrella samples ``exp[-B_j(m)]`` times the athermal density.  Adjacent
umbrellas exchange configurations, and the production histograms are combined
with discrete WHAM.  WHAM determines the otherwise unknown normalization of
each window before assigning every recorded sample the athermal weight

    w(m) = 1 / sum_j n_j * exp[f_j - B_j(m)].

Consequently, ``c_prob``, ``rg_prob``, and ``crg_prob`` in the output NPZ have
the umbrella bias removed.  The biased trajectories are never exposed under
the legacy ``*_samples`` names.  Optional raw arrays are systematic importance
resamples and are explicitly labelled as such.

The move geometry and Hastings-corrected pull moves are imported from
``single_chain_wang_landau.py`` so both enhanced samplers use exactly the same
Markov kernel.  Multiple windows run concurrently with a persistent process
pool.  ``--n_processes`` may be smaller than the number of windows; windows are
then time-shared without changing the Markov chain or exchange equations.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
import tempfile
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from single_chain_wang_landau import (
    BEND_DEFINITION,
    MOVE_FUNCS,
    attempt_pull_move,
    build_distributions,
    compact_seed_chain,
    contact_count,
    contact_delta_from_occupancy,
    contact_upper_bound,
    coordination_histogram,
    count_bends,
    effective_sample_size,
    enumerate_rooted_saws,
    geometric_contact_maximum,
    radius_of_gyration,
    systematic_resample,
    validate_chain,
)


Vec = Tuple[int, int, int]
DEFAULT_UMBRELLA_K = 0.30
DEFAULT_WINDOW_SPACING = 3
CHECKPOINT_VERSION = 1
RAW_SAMPLES_WARNING = (
    "These arrays are systematic importance resamples with duplicates; do not "
    "use them for variance or error-bar estimation. Use c_blocked_stderr."
)


def logsumexp(values: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    """NumPy-only log-sum-exp with well-defined all-minus-infinity slices."""
    array = np.asarray(values, dtype=np.float64)
    maximum = np.max(array, axis=axis, keepdims=True)
    finite = np.isfinite(maximum)
    with np.errstate(invalid="ignore"):
        shifted = np.where(finite, array - maximum, -np.inf)
    total = np.sum(np.exp(shifted), axis=axis, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(finite, maximum + np.log(total), -np.inf)
    if axis is None:
        return np.asarray(result.squeeze())
    return np.squeeze(result, axis=axis)


def make_window_centers(m_min: int, m_max: int, spacing: int) -> np.ndarray:
    """Return integer centers from m_min through the first center >= m_max."""
    if m_max < m_min:
        raise ValueError("m_max must be at least m_min")
    if spacing < 1:
        raise ValueError("window spacing must be positive")
    count = int(math.ceil((m_max - m_min) / spacing)) + 1
    return m_min + spacing * np.arange(count, dtype=np.int64)


def harmonic_bias_matrix(
    centers: np.ndarray, contact_values: np.ndarray, umbrella_k: float
) -> np.ndarray:
    """Return B[j,m] = k/2 (m-m0_j)^2 in reduced, dimensionless units."""
    centers = np.asarray(centers, dtype=np.float64)
    contacts = np.asarray(contact_values, dtype=np.float64)
    if centers.ndim != 1 or contacts.ndim != 1:
        raise ValueError("centers and contact_values must be one-dimensional")
    if not math.isfinite(umbrella_k) or umbrella_k <= 0.0:
        raise ValueError("umbrella_k must be finite and positive")
    return 0.5 * umbrella_k * np.square(contacts[None, :] - centers[:, None])


def solve_wham(
    window_histograms: np.ndarray,
    bias_matrix: np.ndarray,
    *,
    tolerance: float = 1e-12,
    max_iterations: int = 100_000,
    initial_free_energies: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Solve the discrete WHAM equations for the unbiased contact probability.

    ``free_energies[j]`` is ``-log Z_j`` when the unbiased contact probability
    is normalized.  Zero-count contact levels retain exactly zero probability;
    callers decide whether such levels are allowed before writing a baseline.
    """
    histogram = np.asarray(window_histograms, dtype=np.float64)
    bias = np.asarray(bias_matrix, dtype=np.float64)
    if histogram.ndim != 2 or histogram.shape != bias.shape:
        raise ValueError("window_histograms and bias_matrix must have equal 2D shapes")
    if np.any(~np.isfinite(histogram)) or np.any(histogram < 0.0):
        raise ValueError("window_histograms must be finite and nonnegative")
    if np.any(~np.isfinite(bias)):
        raise ValueError("bias_matrix must be finite")
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("WHAM tolerance must be finite and positive")
    if max_iterations < 1:
        raise ValueError("WHAM max_iterations must be positive")

    samples_per_window = histogram.sum(axis=1)
    if np.any(samples_per_window <= 0.0):
        empty = np.flatnonzero(samples_per_window <= 0.0).tolist()
        raise ValueError(f"WHAM windows contain no samples: {empty}")
    pooled = histogram.sum(axis=0)
    observed = pooled > 0.0
    if not np.any(observed):
        raise ValueError("WHAM received no samples")

    n_windows = histogram.shape[0]
    if initial_free_energies is None:
        free_energies = np.zeros(n_windows, dtype=np.float64)
    else:
        free_energies = np.asarray(initial_free_energies, dtype=np.float64).copy()
        if free_energies.shape != (n_windows,) or np.any(~np.isfinite(free_energies)):
            raise ValueError("initial_free_energies has the wrong shape or non-finite values")

    log_samples = np.log(samples_per_window)
    converged = False
    delta = math.inf
    log_probability = np.full(pooled.size, -np.inf, dtype=np.float64)
    log_denominator = np.zeros(pooled.size, dtype=np.float64)

    for iteration in range(1, max_iterations + 1):
        log_denominator = logsumexp(
            log_samples[:, None] + free_energies[:, None] - bias,
            axis=0,
        )
        log_probability.fill(-np.inf)
        log_probability[observed] = (
            np.log(pooled[observed]) - log_denominator[observed]
        )
        log_probability -= float(logsumexp(log_probability))
        updated = -logsumexp(log_probability[None, :] - bias, axis=1)
        delta = float(np.max(np.abs(updated - free_energies)))
        free_energies = updated
        if delta < tolerance:
            converged = True
            break

    if not converged:
        raise RuntimeError(
            f"WHAM failed to converge in {max_iterations} iterations; "
            f"last max |delta f|={delta:.3e}"
        )

    # Recompute once at the converged offsets so returned weights and P(m) use
    # the same final iterate rather than values from the preceding one.
    log_denominator = logsumexp(
        log_samples[:, None] + free_energies[:, None] - bias,
        axis=0,
    )
    log_probability.fill(-np.inf)
    log_probability[observed] = np.log(pooled[observed]) - log_denominator[observed]
    log_probability -= float(logsumexp(log_probability))
    probability = np.exp(log_probability)
    normalized_window_probability = np.exp(
        log_probability[None, :] - bias + free_energies[:, None]
    )
    normalization_error = float(
        np.max(np.abs(normalized_window_probability.sum(axis=1) - 1.0))
    )
    if normalization_error > max(1e-10, 100.0 * tolerance):
        raise RuntimeError(
            "WHAM window probabilities are not normalized: maximum error "
            f"{normalization_error:.3e}"
        )

    return {
        "probability": probability,
        "log_probability": log_probability,
        "free_energies": free_energies,
        "log_denominator": log_denominator,
        "log_sample_weight_by_contact": -log_denominator,
        "window_probability": normalized_window_probability,
        "samples_per_window": samples_per_window,
        "iterations": iteration,
        "converged": converged,
        "max_delta": delta,
        "normalization_error": normalization_error,
    }


def empirical_adjacent_overlap(window_histograms: np.ndarray) -> np.ndarray:
    histogram = np.asarray(window_histograms, dtype=np.float64)
    totals = histogram.sum(axis=1)
    if histogram.ndim != 2 or np.any(totals <= 0.0):
        raise ValueError("every histogram row must contain samples")
    probabilities = histogram / totals[:, None]
    return np.minimum(probabilities[:-1], probabilities[1:]).sum(axis=1)


def predicted_adjacent_diagnostics(
    window_probability: np.ndarray, bias_matrix: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Return WHAM-predicted overlap coefficients and exchange acceptance."""
    probability = np.asarray(window_probability, dtype=np.float64)
    bias = np.asarray(bias_matrix, dtype=np.float64)
    if probability.shape != bias.shape:
        raise ValueError("window_probability and bias_matrix shapes differ")
    n_pairs = probability.shape[0] - 1
    overlaps = np.empty(n_pairs, dtype=np.float64)
    exchange = np.empty(n_pairs, dtype=np.float64)
    for left in range(n_pairs):
        right = left + 1
        p_left = probability[left]
        p_right = probability[right]
        overlaps[left] = np.minimum(p_left, p_right).sum()
        # Rows are m sampled in the left window, columns are n sampled in the
        # right.  This is old reduced bias minus swapped reduced bias.
        log_acceptance = (
            bias[left, :, None]
            + bias[right, None, :]
            - bias[left, None, :]
            - bias[right, :, None]
        )
        acceptance = np.exp(np.minimum(0.0, log_acceptance))
        exchange[left] = float(
            np.sum(p_left[:, None] * p_right[None, :] * acceptance)
        )
    return overlaps, exchange


def _attempt_biased_move(
    chain: List[Vec],
    occupied: Set[Vec],
    contact: int,
    center: float,
    umbrella_k: float,
    m_min: int,
    m_max: int,
    rng: random.Random,
    pull_move_weight: float,
) -> Tuple[List[Vec], Set[Vec], int, bool, bool]:
    """Take one Metropolis-Hastings step under one harmonic umbrella."""
    if pull_move_weight > 0.0 and rng.random() < pull_move_weight:
        move = attempt_pull_move
    else:
        move = rng.choice(MOVE_FUNCS)
    valid, proposed_chain, proposed_occupied, log_q_ratio = move(
        chain, occupied, rng
    )
    if not valid:
        return chain, occupied, contact, False, False
    proposed_contact = contact + contact_delta_from_occupancy(
        occupied, proposed_occupied
    )
    if proposed_contact < m_min or proposed_contact > m_max:
        return chain, occupied, contact, True, False
    old_bias = 0.5 * umbrella_k * (contact - center) ** 2
    new_bias = 0.5 * umbrella_k * (proposed_contact - center) ** 2
    log_acceptance = old_bias - new_bias + log_q_ratio
    if log_acceptance >= 0.0 or rng.random() < math.exp(log_acceptance):
        return proposed_chain, proposed_occupied, proposed_contact, True, True
    return chain, occupied, contact, True, False


def advance_walker(
    state: Dict[str, Any],
    window_index: int,
    center: float,
    umbrella_k: float,
    block_steps: int,
    burn_steps: int,
    sample_every: int,
    m_min: int,
    m_max: int,
    pull_move_weight: float,
) -> Dict[str, Any]:
    """Advance one walker between exchange attempts; safe for worker processes."""
    walker_id = int(state["walker_id"])
    chain = [tuple(map(int, site)) for site in state["chain"]]
    occupied = set(chain)
    contact = int(state["contact"])
    if contact_count(chain, occupied) != contact:
        raise RuntimeError(f"walker {walker_id} contact bookkeeping is inconsistent")
    rng = random.Random()
    rng.setstate(state["rng_state"])
    steps_done = int(state["steps_done"])
    accepted = int(state["accepted_moves"])
    geometrically_valid = int(state["geometrically_valid_moves"])

    contacts: List[int] = []
    radii: List[float] = []
    bends: List[int] = []
    coordination: List[np.ndarray] = []
    sample_steps: List[int] = []

    for local_step in range(1, block_steps + 1):
        step = steps_done + local_step
        chain, occupied, contact, valid, was_accepted = _attempt_biased_move(
            chain,
            occupied,
            contact,
            center,
            umbrella_k,
            m_min,
            m_max,
            rng,
            pull_move_weight,
        )
        if valid:
            geometrically_valid += 1
        if was_accepted:
            accepted += 1
        if step > burn_steps and (step - burn_steps) % sample_every == 0:
            histogram = coordination_histogram(chain, occupied)
            if int(np.dot(np.arange(7, dtype=np.int64), histogram)) != 2 * contact:
                raise RuntimeError(
                    f"walker {walker_id} coordination histogram disagrees with m"
                )
            contacts.append(contact)
            radii.append(radius_of_gyration(chain))
            bends.append(count_bends(chain))
            coordination.append(histogram)
            sample_steps.append(step)

    if contact_count(chain, occupied) != contact:
        raise RuntimeError(f"walker {walker_id} contact drift after block")
    return {
        "walker_id": walker_id,
        "seed": int(state["seed"]),
        "chain": chain,
        "contact": contact,
        "rng_state": rng.getstate(),
        "steps_done": steps_done + block_steps,
        "accepted_moves": accepted,
        "geometrically_valid_moves": geometrically_valid,
        "contact_samples": np.asarray(contacts, dtype=np.int64),
        "rg_samples": np.asarray(radii, dtype=np.float64),
        "bend_samples": np.asarray(bends, dtype=np.int64),
        "coordination_histogram_samples": np.asarray(
            coordination, dtype=np.int64
        ).reshape((-1, 7)),
        "sample_steps": np.asarray(sample_steps, dtype=np.int64),
        "sample_windows": np.full(len(contacts), window_index, dtype=np.int64),
        "sample_walkers": np.full(len(contacts), walker_id, dtype=np.int64),
    }


SAMPLE_KEYS = (
    "contact_samples",
    "rg_samples",
    "bend_samples",
    "coordination_histogram_samples",
    "sample_steps",
    "sample_windows",
    "sample_walkers",
)


def empty_sample_store() -> Dict[str, List[np.ndarray]]:
    return {key: [] for key in SAMPLE_KEYS}


def append_samples(
    store: Dict[str, List[np.ndarray]], result: Dict[str, Any]
) -> None:
    for key in SAMPLE_KEYS:
        values = np.asarray(result[key])
        if key == "coordination_histogram_samples":
            values = values.reshape((-1, 7))
        if values.shape[0] > 0:
            store[key].append(values)


def concatenate_samples(store: Dict[str, List[np.ndarray]]) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    integer_keys = {
        "contact_samples", "bend_samples", "sample_steps",
        "sample_windows", "sample_walkers",
    }
    for key in SAMPLE_KEYS:
        chunks = store[key]
        if chunks:
            out[key] = np.concatenate(chunks, axis=0)
        elif key == "coordination_histogram_samples":
            out[key] = np.empty((0, 7), dtype=np.int64)
        elif key in integer_keys:
            out[key] = np.array([], dtype=np.int64)
        else:
            out[key] = np.array([], dtype=np.float64)
    sizes = {key: values.shape[0] for key, values in out.items()}
    if len(set(sizes.values())) != 1:
        raise RuntimeError(f"sample arrays have inconsistent lengths: {sizes}")
    return out


def _rng_states_to_arrays(
    states: Sequence[Tuple[Any, ...]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    versions: List[int] = []
    keys: List[np.ndarray] = []
    gaussian: List[float] = []
    for state in states:
        versions.append(int(state[0]))
        key = np.asarray(state[1], dtype=np.uint32)
        if key.ndim != 1:
            raise ValueError("random state key must be one-dimensional")
        keys.append(key)
        gaussian.append(float("nan") if state[2] is None else float(state[2]))
    if len({key.size for key in keys}) != 1:
        raise ValueError("random states have inconsistent key lengths")
    return (
        np.asarray(versions, dtype=np.int64),
        np.stack(keys, axis=0),
        np.asarray(gaussian, dtype=np.float64),
    )


def _rng_states_from_arrays(
    versions: np.ndarray, keys: np.ndarray, gaussian: np.ndarray
) -> List[Tuple[Any, ...]]:
    versions = np.asarray(versions, dtype=np.int64)
    keys = np.asarray(keys, dtype=np.uint32)
    gaussian = np.asarray(gaussian, dtype=np.float64)
    if keys.ndim != 2 or versions.shape != (keys.shape[0],) or gaussian.shape != versions.shape:
        raise ValueError("checkpoint random-state arrays have inconsistent shapes")
    states: List[Tuple[Any, ...]] = []
    for version, key, gauss in zip(versions, keys, gaussian):
        states.append(
            (
                int(version),
                tuple(int(value) for value in key),
                None if math.isnan(float(gauss)) else float(gauss),
            )
        )
    return states


def _atomic_savez(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=path.stem + "_", suffix=".npz", dir=path.parent, delete=False
        ) as handle:
            temporary_path = Path(handle.name)
        np.savez_compressed(temporary_path, **payload)
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def save_checkpoint(
    path: Path,
    *,
    args: argparse.Namespace,
    centers: np.ndarray,
    burn_steps: int,
    states: Sequence[Dict[str, Any]],
    window_by_walker: np.ndarray,
    sample_store: Dict[str, List[np.ndarray]],
    exchange_rng: random.Random,
    exchange_round: int,
    swap_attempts: np.ndarray,
    swap_accepts: np.ndarray,
    round_trip_phase: np.ndarray,
    round_trips: np.ndarray,
    initialization: np.ndarray,
    elapsed_seconds: float,
) -> None:
    samples = concatenate_samples(sample_store)
    rng_versions, rng_keys, rng_gaussian = _rng_states_to_arrays(
        [state["rng_state"] for state in states]
    )
    exchange_versions, exchange_keys, exchange_gaussian = _rng_states_to_arrays(
        [exchange_rng.getstate()]
    )
    payload: Dict[str, Any] = {
        "checkpoint_version": np.array(CHECKPOINT_VERSION, dtype=np.int64),
        "N": np.array(args.N, dtype=np.int64),
        "m_min": np.array(args.m_min, dtype=np.int64),
        "m_max": np.array(args.m_max, dtype=np.int64),
        "umbrella_k": np.array(args.umbrella_k, dtype=np.float64),
        "window_spacing": np.array(args.window_spacing, dtype=np.int64),
        "window_centers": np.asarray(centers, dtype=np.int64),
        "base_seed": np.array(args.base_seed, dtype=np.int64),
        "sample_every": np.array(args.sample_every, dtype=np.int64),
        "exchange_every": np.array(args.exchange_every, dtype=np.int64),
        "pull_move_weight": np.array(args.pull_move_weight, dtype=np.float64),
        "burn_steps": np.array(burn_steps, dtype=np.int64),
        "target_steps_per_window": np.array(
            args.steps_per_window, dtype=np.int64
        ),
        "walker_ids": np.asarray(
            [state["walker_id"] for state in states], dtype=np.int64
        ),
        "worker_seeds": np.asarray(
            [state["seed"] for state in states], dtype=np.int64
        ),
        "chains": np.asarray([state["chain"] for state in states], dtype=np.int64),
        "current_contacts": np.asarray(
            [state["contact"] for state in states], dtype=np.int64
        ),
        "steps_done": np.asarray(
            [state["steps_done"] for state in states], dtype=np.int64
        ),
        "accepted_moves": np.asarray(
            [state["accepted_moves"] for state in states], dtype=np.int64
        ),
        "geometrically_valid_moves": np.asarray(
            [state["geometrically_valid_moves"] for state in states],
            dtype=np.int64,
        ),
        "walker_rng_versions": rng_versions,
        "walker_rng_keys": rng_keys,
        "walker_rng_gaussian": rng_gaussian,
        "exchange_rng_versions": exchange_versions,
        "exchange_rng_keys": exchange_keys,
        "exchange_rng_gaussian": exchange_gaussian,
        "window_by_walker": np.asarray(window_by_walker, dtype=np.int64),
        "exchange_round": np.array(exchange_round, dtype=np.int64),
        "swap_attempts": np.asarray(swap_attempts, dtype=np.int64),
        "swap_accepts": np.asarray(swap_accepts, dtype=np.int64),
        "round_trip_phase": np.asarray(round_trip_phase, dtype=np.int8),
        "round_trips": np.asarray(round_trips, dtype=np.int64),
        "initialization": np.asarray(initialization, dtype="U16"),
        "elapsed_seconds": np.array(elapsed_seconds, dtype=np.float64),
    }
    payload.update(samples)
    _atomic_savez(path, payload)


def load_checkpoint(
    path: Path, args: argparse.Namespace, centers: np.ndarray
) -> Dict[str, Any]:
    with np.load(path, allow_pickle=False) as saved:
        if int(saved["checkpoint_version"]) != CHECKPOINT_VERSION:
            raise ValueError(
                f"unsupported checkpoint version {int(saved['checkpoint_version'])}"
            )
        exact_checks = {
            "N": args.N,
            "m_min": args.m_min,
            "m_max": args.m_max,
            "window_spacing": args.window_spacing,
            "base_seed": args.base_seed,
            "sample_every": args.sample_every,
            "exchange_every": args.exchange_every,
        }
        for key, expected in exact_checks.items():
            actual = int(saved[key])
            if actual != int(expected):
                raise ValueError(
                    f"checkpoint {key}={actual} does not match requested {expected}"
                )
        float_checks = {
            "umbrella_k": args.umbrella_k,
            "pull_move_weight": args.pull_move_weight,
        }
        for key, expected in float_checks.items():
            actual = float(saved[key])
            if not math.isclose(actual, float(expected), rel_tol=0.0, abs_tol=1e-15):
                raise ValueError(
                    f"checkpoint {key}={actual} does not match requested {expected}"
                )
        saved_centers = np.asarray(saved["window_centers"], dtype=np.int64)
        if not np.array_equal(saved_centers, centers):
            raise ValueError("checkpoint umbrella centers do not match this run")

        chains = np.asarray(saved["chains"], dtype=np.int64)
        n_walkers = centers.size
        if chains.shape != (n_walkers, args.N, 3):
            raise ValueError("checkpoint chain array has the wrong shape")
        rng_states = _rng_states_from_arrays(
            saved["walker_rng_versions"],
            saved["walker_rng_keys"],
            saved["walker_rng_gaussian"],
        )
        ids = np.asarray(saved["walker_ids"], dtype=np.int64)
        seeds = np.asarray(saved["worker_seeds"], dtype=np.int64)
        contacts = np.asarray(saved["current_contacts"], dtype=np.int64)
        steps_done = np.asarray(saved["steps_done"], dtype=np.int64)
        accepted = np.asarray(saved["accepted_moves"], dtype=np.int64)
        valid = np.asarray(saved["geometrically_valid_moves"], dtype=np.int64)
        if not all(array.shape == (n_walkers,) for array in (
            ids, seeds, contacts, steps_done, accepted, valid
        )):
            raise ValueError("checkpoint walker metadata has the wrong shape")
        if not np.array_equal(ids, np.arange(n_walkers)):
            raise ValueError("checkpoint walker IDs are not canonical")
        if len(set(steps_done.tolist())) != 1:
            raise ValueError("checkpoint walkers are at different step counts")
        if int(steps_done[0]) > args.steps_per_window:
            raise ValueError(
                "checkpoint is beyond requested --steps_per_window; increase the target"
            )

        states: List[Dict[str, Any]] = []
        for i in range(n_walkers):
            chain = [tuple(map(int, site)) for site in chains[i]]
            validate_chain(chain)
            if contact_count(chain, set(chain)) != int(contacts[i]):
                raise ValueError(f"checkpoint contact mismatch for walker {i}")
            states.append(
                {
                    "walker_id": i,
                    "seed": int(seeds[i]),
                    "chain": chain,
                    "contact": int(contacts[i]),
                    "rng_state": rng_states[i],
                    "steps_done": int(steps_done[i]),
                    "accepted_moves": int(accepted[i]),
                    "geometrically_valid_moves": int(valid[i]),
                }
            )

        sample_store = empty_sample_store()
        sample_sizes: List[int] = []
        for key in SAMPLE_KEYS:
            values = np.asarray(saved[key]).copy()
            if key == "coordination_histogram_samples":
                values = values.astype(np.int64).reshape((-1, 7))
            elif key == "rg_samples":
                values = values.astype(np.float64)
            else:
                values = values.astype(np.int64)
            sample_sizes.append(values.shape[0])
            if values.shape[0]:
                sample_store[key].append(values)
        if len(set(sample_sizes)) != 1:
            raise ValueError("checkpoint sample arrays have inconsistent lengths")

        exchange_state = _rng_states_from_arrays(
            saved["exchange_rng_versions"],
            saved["exchange_rng_keys"],
            saved["exchange_rng_gaussian"],
        )[0]
        return {
            "states": states,
            "window_by_walker": np.asarray(
                saved["window_by_walker"], dtype=np.int64
            ).copy(),
            "sample_store": sample_store,
            "exchange_rng_state": exchange_state,
            "exchange_round": int(saved["exchange_round"]),
            "swap_attempts": np.asarray(saved["swap_attempts"], dtype=np.int64).copy(),
            "swap_accepts": np.asarray(saved["swap_accepts"], dtype=np.int64).copy(),
            "round_trip_phase": np.asarray(
                saved["round_trip_phase"], dtype=np.int8
            ).copy(),
            "round_trips": np.asarray(saved["round_trips"], dtype=np.int64).copy(),
            "initialization": np.asarray(saved["initialization"], dtype="U16").copy(),
            "burn_steps": int(saved["burn_steps"]),
            "saved_target_steps": int(saved["target_steps_per_window"]),
            "elapsed_seconds": float(saved["elapsed_seconds"]),
        }


def _observe_window_round_trips(
    window_by_walker: np.ndarray,
    phases: np.ndarray,
    round_trips: np.ndarray,
) -> None:
    high = window_by_walker.size - 1
    if high <= 0:
        return
    for walker, window in enumerate(window_by_walker):
        if phases[walker] == 0 and window == 0:
            phases[walker] = 1
        elif phases[walker] == 1 and window == high:
            phases[walker] = 2
        elif phases[walker] == 2 and window == 0:
            round_trips[walker] += 1
            phases[walker] = 1


def _initial_walkers(
    args: argparse.Namespace, centers: np.ndarray
) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    rod: List[Vec] = [(i, 0, 0) for i in range(args.N)]
    validate_chain(rod)
    compact: Optional[List[Vec]] = None
    if args.init in ("auto", "compact"):
        try:
            compact = compact_seed_chain(args.N)
        except ValueError:
            if args.init == "compact":
                raise

    states: List[Dict[str, Any]] = []
    initialization: List[str] = []
    midpoint = 0.5 * (args.m_min + args.m_max)
    for walker, center in enumerate(centers):
        use_compact = args.init == "compact" or (
            args.init == "auto" and compact is not None and center > midpoint
        )
        chain = list(compact if use_compact else rod)
        occupied = set(chain)
        seed = args.base_seed + walker
        rng = random.Random(seed)
        states.append(
            {
                "walker_id": walker,
                "seed": seed,
                "chain": chain,
                "contact": contact_count(chain, occupied),
                "rng_state": rng.getstate(),
                "steps_done": 0,
                "accepted_moves": 0,
                "geometrically_valid_moves": 0,
            }
        )
        initialization.append("compact" if use_compact else "rod")
    return states, np.asarray(initialization, dtype="U16")


def _attempt_replica_exchanges(
    states: Sequence[Dict[str, Any]],
    window_by_walker: np.ndarray,
    centers: np.ndarray,
    umbrella_k: float,
    parity: int,
    rng: random.Random,
    attempts: np.ndarray,
    accepts: np.ndarray,
) -> None:
    n_windows = centers.size
    walker_at_window = np.empty(n_windows, dtype=np.int64)
    walker_at_window[window_by_walker] = np.arange(n_windows, dtype=np.int64)
    for left in range(parity, n_windows - 1, 2):
        right = left + 1
        left_walker = int(walker_at_window[left])
        right_walker = int(walker_at_window[right])
        left_contact = int(states[left_walker]["contact"])
        right_contact = int(states[right_walker]["contact"])
        # For equal k and adjacent centers this reduces exactly to
        # k*(m0_right-m0_left)*(m_left-m_right).
        log_acceptance = umbrella_k * (
            float(centers[right]) - float(centers[left])
        ) * (left_contact - right_contact)
        attempts[left] += 1
        if log_acceptance >= 0.0 or rng.random() < math.exp(log_acceptance):
            window_by_walker[left_walker] = right
            window_by_walker[right_walker] = left
            accepts[left] += 1


def resolve_process_count(requested: Optional[int], n_windows: int) -> int:
    if requested is not None:
        return min(requested, n_windows)
    allocation = os.environ.get("SLURM_CPUS_PER_TASK")
    if allocation is not None:
        try:
            available = int(allocation)
        except ValueError:
            warnings.warn(
                f"ignoring invalid SLURM_CPUS_PER_TASK={allocation!r}",
                RuntimeWarning,
                stacklevel=2,
            )
        else:
            if available > 0:
                return min(available, n_windows)
    return min(os.cpu_count() or 1, n_windows)


def run_replica_exchange(
    args: argparse.Namespace,
    centers: np.ndarray,
    *,
    progress: bool = True,
) -> Dict[str, Any]:
    """Run or resume all umbrellas and return the complete biased sample set."""
    n_windows = centers.size
    requested_burn_steps = int(round(args.burnin * args.steps_per_window))
    checkpoint_path = (
        Path(args.checkpoint)
        if args.checkpoint
        else (Path(args.resume_checkpoint) if args.resume_checkpoint else None)
    )

    if args.resume_checkpoint:
        loaded = load_checkpoint(Path(args.resume_checkpoint), args, centers)
        states = loaded["states"]
        window_by_walker = loaded["window_by_walker"]
        sample_store = loaded["sample_store"]
        exchange_rng = random.Random()
        exchange_rng.setstate(loaded["exchange_rng_state"])
        exchange_round = loaded["exchange_round"]
        swap_attempts = loaded["swap_attempts"]
        swap_accepts = loaded["swap_accepts"]
        round_trip_phase = loaded["round_trip_phase"]
        round_trips = loaded["round_trips"]
        initialization = loaded["initialization"]
        burn_steps = loaded["burn_steps"]
        previous_elapsed = loaded["elapsed_seconds"]
        if requested_burn_steps != burn_steps:
            print(
                "Resume note: retaining checkpoint burn-in boundary "
                f"{burn_steps} instead of newly requested {requested_burn_steps} steps.",
                flush=True,
            )
        if args.steps_per_window != loaded["saved_target_steps"]:
            print(
                "Resume note: extending target from "
                f"{loaded['saved_target_steps']} to {args.steps_per_window} "
                "steps per window.",
                flush=True,
            )
    else:
        states, initialization = _initial_walkers(args, centers)
        window_by_walker = np.arange(n_windows, dtype=np.int64)
        sample_store = empty_sample_store()
        exchange_rng = random.Random(args.base_seed + 10_000_019)
        exchange_round = 0
        swap_attempts = np.zeros(n_windows - 1, dtype=np.int64)
        swap_accepts = np.zeros(n_windows - 1, dtype=np.int64)
        round_trip_phase = np.zeros(n_windows, dtype=np.int8)
        round_trips = np.zeros(n_windows, dtype=np.int64)
        burn_steps = requested_burn_steps
        previous_elapsed = 0.0
        _observe_window_round_trips(
            window_by_walker, round_trip_phase, round_trips
        )

    if sorted(window_by_walker.tolist()) != list(range(n_windows)):
        raise RuntimeError("window labels are not a permutation")
    current_steps = {int(state["steps_done"]) for state in states}
    if len(current_steps) != 1:
        raise RuntimeError("walkers are not synchronized at an exchange boundary")
    starting_step = current_steps.pop()
    n_processes = resolve_process_count(args.n_processes, n_windows)
    if progress:
        print(
            f"Umbrellas={n_windows} centers={centers.tolist()} k={args.umbrella_k:g} "
            f"processes={n_processes}",
            flush=True,
        )
        if n_processes < n_windows:
            print(
                f"{n_windows} logical windows will time-share {n_processes} processes.",
                flush=True,
            )
        if starting_step:
            print(
                f"Resuming {n_windows} walkers at step {starting_step}; "
                f"samples retained={concatenate_samples(sample_store)['contact_samples'].size}.",
                flush=True,
            )

    invocation_start = time.time()
    last_checkpoint_time = invocation_start
    progress_stride = max(1, args.steps_per_window // 20)
    last_progress_bucket = starting_step // progress_stride
    stopped_for_time = False
    clean_state_keys = (
        "walker_id", "seed", "chain", "contact", "rng_state", "steps_done",
        "accepted_moves", "geometrically_valid_moves",
    )

    executor: Optional[ProcessPoolExecutor] = None
    try:
        if n_processes > 1 and starting_step < args.steps_per_window:
            executor = ProcessPoolExecutor(max_workers=n_processes)
        while starting_step < args.steps_per_window:
            block_steps = min(
                args.exchange_every, args.steps_per_window - starting_step
            )
            if executor is None:
                results = [
                    advance_walker(
                        states[walker],
                        int(window_by_walker[walker]),
                        float(centers[window_by_walker[walker]]),
                        args.umbrella_k,
                        block_steps,
                        burn_steps,
                        args.sample_every,
                        args.m_min,
                        args.m_max,
                        args.pull_move_weight,
                    )
                    for walker in range(n_windows)
                ]
            else:
                futures = [
                    executor.submit(
                        advance_walker,
                        states[walker],
                        int(window_by_walker[walker]),
                        float(centers[window_by_walker[walker]]),
                        args.umbrella_k,
                        block_steps,
                        burn_steps,
                        args.sample_every,
                        args.m_min,
                        args.m_max,
                        args.pull_move_weight,
                    )
                    for walker in range(n_windows)
                ]
                results = [future.result() for future in futures]
            results.sort(key=lambda result: int(result["walker_id"]))
            for result in results:
                append_samples(sample_store, result)
                walker = int(result["walker_id"])
                states[walker] = {key: result[key] for key in clean_state_keys}

            _attempt_replica_exchanges(
                states,
                window_by_walker,
                centers,
                args.umbrella_k,
                exchange_round % 2,
                exchange_rng,
                swap_attempts,
                swap_accepts,
            )
            exchange_round += 1
            _observe_window_round_trips(
                window_by_walker, round_trip_phase, round_trips
            )
            starting_step += block_steps
            now = time.time()
            elapsed = previous_elapsed + now - invocation_start

            bucket = starting_step // progress_stride
            if progress and (
                bucket > last_progress_bucket or starting_step == args.steps_per_window
            ):
                accepted = sum(int(state["accepted_moves"]) for state in states)
                attempts = n_windows * starting_step
                attempted_pairs = swap_attempts > 0
                minimum_swap = (
                    float(np.min(swap_accepts[attempted_pairs] / swap_attempts[attempted_pairs]))
                    if np.any(attempted_pairs) else float("nan")
                )
                print(
                    f"{100.0 * starting_step / args.steps_per_window:5.1f}% "
                    f"step={starting_step}/{args.steps_per_window} "
                    f"local_acc={accepted / max(attempts, 1):.3f} "
                    f"min_swap={minimum_swap:.3f} "
                    f"round_trips={int(round_trips.sum())} "
                    f"elapsed={elapsed:.1f}s",
                    flush=True,
                )
                last_progress_bucket = bucket

            if (
                checkpoint_path is not None
                and now - last_checkpoint_time >= args.checkpoint_every_seconds
            ):
                save_checkpoint(
                    checkpoint_path,
                    args=args,
                    centers=centers,
                    burn_steps=burn_steps,
                    states=states,
                    window_by_walker=window_by_walker,
                    sample_store=sample_store,
                    exchange_rng=exchange_rng,
                    exchange_round=exchange_round,
                    swap_attempts=swap_attempts,
                    swap_accepts=swap_accepts,
                    round_trip_phase=round_trip_phase,
                    round_trips=round_trips,
                    initialization=initialization,
                    elapsed_seconds=elapsed,
                )
                last_checkpoint_time = now
                if progress:
                    print(f"Checkpoint written: {checkpoint_path}", flush=True)

            invocation_elapsed = now - invocation_start
            if (
                math.isfinite(args.max_wall_seconds)
                and invocation_elapsed
                >= args.max_wall_seconds - args.shutdown_margin_seconds
            ):
                stopped_for_time = True
                break
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

    elapsed = previous_elapsed + time.time() - invocation_start
    if checkpoint_path is not None:
        save_checkpoint(
            checkpoint_path,
            args=args,
            centers=centers,
            burn_steps=burn_steps,
            states=states,
            window_by_walker=window_by_walker,
            sample_store=sample_store,
            exchange_rng=exchange_rng,
            exchange_round=exchange_round,
            swap_attempts=swap_attempts,
            swap_accepts=swap_accepts,
            round_trip_phase=round_trip_phase,
            round_trips=round_trips,
            initialization=initialization,
            elapsed_seconds=elapsed,
        )

    samples = concatenate_samples(sample_store)
    complete = all(
        int(state["steps_done"]) >= args.steps_per_window for state in states
    )
    if stopped_for_time and progress:
        print(
            f"Stopping before the wall-time limit at step {starting_step}; "
            f"resume with --resume_checkpoint {checkpoint_path}.",
            flush=True,
        )
    return {
        "states": states,
        "samples": samples,
        "window_by_walker": window_by_walker,
        "swap_attempts": swap_attempts,
        "swap_accepts": swap_accepts,
        "round_trip_phase": round_trip_phase,
        "round_trips": round_trips,
        "initialization": initialization,
        "exchange_rounds": exchange_round,
        "burn_steps": burn_steps,
        "elapsed_seconds": elapsed,
        "n_processes": n_processes,
        "complete": complete,
    }


def build_window_histograms(
    contacts: np.ndarray,
    sample_windows: np.ndarray,
    n_windows: int,
    m_min: int,
    m_max: int,
) -> np.ndarray:
    contacts = np.asarray(contacts, dtype=np.int64)
    windows = np.asarray(sample_windows, dtype=np.int64)
    if contacts.shape != windows.shape:
        raise ValueError("contacts and sample_windows shapes differ")
    if np.any((contacts < m_min) | (contacts > m_max)):
        raise ValueError("contact sample lies outside the declared range")
    if np.any((windows < 0) | (windows >= n_windows)):
        raise ValueError("sample contains an invalid umbrella index")
    histogram = np.zeros((n_windows, m_max - m_min + 1), dtype=np.int64)
    np.add.at(histogram, (windows, contacts - m_min), 1)
    return histogram


def blocked_wham_contact_stderr(
    contacts: np.ndarray,
    sample_windows: np.ndarray,
    sample_steps: np.ndarray,
    n_windows: int,
    m_min: int,
    m_max: int,
    bias_matrix: np.ndarray,
    n_blocks: int,
    tolerance: float,
    max_iterations: int,
    initial_free_energies: np.ndarray,
) -> Tuple[np.ndarray, int, np.ndarray]:
    """Re-solve WHAM in contiguous time blocks and return batch-mean errors."""
    contacts = np.asarray(contacts, dtype=np.int64)
    windows = np.asarray(sample_windows, dtype=np.int64)
    steps = np.asarray(sample_steps, dtype=np.int64)
    if not (contacts.shape == windows.shape == steps.shape):
        raise ValueError("blocked WHAM sample arrays have inconsistent shapes")
    unique_steps = np.unique(steps)
    block_count = min(n_blocks, unique_steps.size)
    if block_count < 2:
        raise ValueError("blocked WHAM requires at least two sampled time points")
    estimates: List[np.ndarray] = []
    for step_block in np.array_split(unique_steps, block_count):
        mask = (steps >= step_block[0]) & (steps <= step_block[-1])
        histogram = build_window_histograms(
            contacts[mask], windows[mask], n_windows, m_min, m_max
        )
        solved = solve_wham(
            histogram,
            bias_matrix,
            tolerance=tolerance,
            max_iterations=max_iterations,
            initial_free_energies=initial_free_energies,
        )
        estimates.append(solved["probability"])
    block_estimates = np.asarray(estimates, dtype=np.float64)
    stderr = block_estimates.std(axis=0, ddof=1) / math.sqrt(block_count)
    return stderr, block_count, block_estimates


def analyse_samples(
    args: argparse.Namespace,
    centers: np.ndarray,
    simulation: Dict[str, Any],
    *,
    enforce_checks: bool = True,
) -> Dict[str, Any]:
    samples = simulation["samples"]
    contacts = np.asarray(samples["contact_samples"], dtype=np.int64)
    radii = np.asarray(samples["rg_samples"], dtype=np.float64)
    bends = np.asarray(samples["bend_samples"], dtype=np.int64)
    windows = np.asarray(samples["sample_windows"], dtype=np.int64)
    sample_steps = np.asarray(samples["sample_steps"], dtype=np.int64)
    coordination = np.asarray(
        samples["coordination_histogram_samples"], dtype=np.int64
    ).reshape((-1, 7))
    if contacts.size == 0:
        raise RuntimeError("no post-burn-in umbrella samples were recorded")

    contact_values = np.arange(args.m_min, args.m_max + 1, dtype=np.int64)
    bias = harmonic_bias_matrix(centers, contact_values, args.umbrella_k)
    window_histograms = build_window_histograms(
        contacts,
        windows,
        centers.size,
        args.m_min,
        args.m_max,
    )
    wham = solve_wham(
        window_histograms,
        bias,
        tolerance=args.wham_tolerance,
        max_iterations=args.wham_max_iterations,
    )
    empirical_overlap = empirical_adjacent_overlap(window_histograms)
    predicted_overlap, predicted_swap = predicted_adjacent_diagnostics(
        wham["window_probability"], bias
    )
    swap_attempts = np.asarray(simulation["swap_attempts"], dtype=np.int64)
    swap_accepts = np.asarray(simulation["swap_accepts"], dtype=np.int64)
    swap_rate = np.full(swap_attempts.shape, np.nan, dtype=np.float64)
    attempted_pairs = swap_attempts > 0
    swap_rate[attempted_pairs] = (
        swap_accepts[attempted_pairs] / swap_attempts[attempted_pairs]
    )

    pooled_counts = window_histograms.sum(axis=0)
    required = np.ones(contact_values.size, dtype=bool)
    if args.excluded_contact_levels:
        required[
            np.asarray(args.excluded_contact_levels, dtype=np.int64) - args.m_min
        ] = False
    failures: List[str] = []
    if args.min_samples_per_level > 0:
        deficient = contact_values[
            required & (pooled_counts < args.min_samples_per_level)
        ]
        if deficient.size:
            failures.append(
                "contact levels below --min_samples_per_level="
                f"{args.min_samples_per_level}: {deficient.tolist()}"
            )
    deficient_overlap = np.flatnonzero(
        empirical_overlap < args.min_adjacent_overlap
    )
    if deficient_overlap.size:
        pairs = [
            (int(centers[i]), int(centers[i + 1]), float(empirical_overlap[i]))
            for i in deficient_overlap
        ]
        failures.append(
            "adjacent empirical overlap below --min_adjacent_overlap="
            f"{args.min_adjacent_overlap:g}: {pairs}"
        )
    if np.any(~attempted_pairs):
        failures.append(
            f"adjacent pairs with no exchange attempts: "
            f"{np.flatnonzero(~attempted_pairs).tolist()}"
        )
    deficient_swap = np.flatnonzero(
        attempted_pairs & (swap_rate < args.min_swap_acceptance)
    )
    if deficient_swap.size:
        pairs = [
            (int(centers[i]), int(centers[i + 1]), float(swap_rate[i]))
            for i in deficient_swap
        ]
        failures.append(
            "adjacent exchange acceptance below --min_swap_acceptance="
            f"{args.min_swap_acceptance:g}: {pairs}"
        )
    total_round_trips = int(np.asarray(simulation["round_trips"]).sum())
    if total_round_trips < args.min_round_trips:
        failures.append(
            f"only {total_round_trips} walker round trips; "
            f"--min_round_trips={args.min_round_trips}"
        )
    if failures and enforce_checks:
        joined = "\n  - ".join(failures)
        raise RuntimeError(
            "umbrella production diagnostics failed; the athermal output was not "
            f"written:\n  - {joined}\nThe complete samples remain in the checkpoint."
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        built = build_distributions(
            contacts,
            radii,
            bends,
            wham["log_sample_weight_by_contact"],
            args.m_min,
            args.rg_bins,
            args.no_joint,
            n_beads=args.N,
            m_cover=args.m_max,
            grid_search_dir=Path(__file__).resolve().parent,
            coordination_histograms=coordination,
        )
    wham_histogram_error = float(
        np.max(np.abs(built["c_prob"] - wham["probability"]))
    )
    if wham_histogram_error > 1e-11:
        raise RuntimeError(
            "sample-weight histogram does not reproduce the WHAM contact "
            f"probability: max error {wham_histogram_error:.3e}"
        )

    blocked_stderr, blocked_count, block_estimates = blocked_wham_contact_stderr(
        contacts,
        windows,
        sample_steps,
        centers.size,
        args.m_min,
        args.m_max,
        bias,
        args.n_blocks,
        args.wham_tolerance,
        args.wham_max_iterations,
        wham["free_energies"],
    )
    weights = np.asarray(built["weights"], dtype=np.float64)
    ess = effective_sample_size(weights)
    window_samples = window_histograms.sum(axis=1)
    per_window_m_mean = (
        window_histograms @ contact_values.astype(np.float64) / window_samples
    )

    walker_ids = np.asarray(samples["sample_walkers"], dtype=np.int64)
    per_walker_sample_count = np.bincount(
        walker_ids, minlength=centers.size
    ).astype(np.int64)
    per_walker_m_mean = np.full(centers.size, np.nan, dtype=np.float64)
    per_walker_rg_mean = np.full(centers.size, np.nan, dtype=np.float64)
    for walker in range(centers.size):
        mask = walker_ids == walker
        if np.any(mask):
            per_walker_m_mean[walker] = float(contacts[mask].mean())
            per_walker_rg_mean[walker] = float(radii[mask].mean())

    return {
        "built": built,
        "weights": weights,
        "wham": wham,
        "bias_matrix": bias,
        "window_histograms": window_histograms,
        "empirical_overlap": empirical_overlap,
        "predicted_overlap": predicted_overlap,
        "predicted_swap_acceptance": predicted_swap,
        "swap_rate": swap_rate,
        "c_blocked_stderr": blocked_stderr,
        "c_block_estimates": block_estimates,
        "blocked_count": blocked_count,
        "importance_ess": ess,
        "per_window_m_mean": per_window_m_mean,
        "per_walker_sample_count": per_walker_sample_count,
        "per_walker_m_mean": per_walker_m_mean,
        "per_walker_rg_mean": per_walker_rg_mean,
        "wham_histogram_error": wham_histogram_error,
        "diagnostic_failures": failures,
        "total_round_trips": total_round_trips,
    }


def write_output(
    args: argparse.Namespace,
    centers: np.ndarray,
    simulation: Dict[str, Any],
    analysis: Dict[str, Any],
) -> Path:
    samples = simulation["samples"]
    contacts = np.asarray(samples["contact_samples"], dtype=np.int64)
    radii = np.asarray(samples["rg_samples"], dtype=np.float64)
    bends = np.asarray(samples["bend_samples"], dtype=np.int64)
    built = analysis["built"]
    wham = analysis["wham"]
    states = simulation["states"]
    accepted = np.asarray(
        [state["accepted_moves"] for state in states], dtype=np.int64
    )
    geometrically_valid = np.asarray(
        [state["geometrically_valid_moves"] for state in states], dtype=np.int64
    )
    attempted = np.asarray(
        [state["steps_done"] for state in states], dtype=np.int64
    )
    acceptance = accepted / attempted

    if args.output:
        output_path = Path(args.output)
    else:
        output_dir = Path(args.dist_dir)
        output_path = output_dir / (
            f"{Path(__file__).stem}_N{args.N}_windows{centers.size}"
            f"_steps{args.steps_per_window}_seed{args.base_seed}.npz"
        )

    save: Dict[str, Any] = {
        key: value
        for key, value in built.items()
        if key not in {"weights", "marginal_m_error", "marginal_rg_error"}
    }
    save.update(
        N=np.array(args.N, dtype=np.int64),
        n_beads=np.array(args.N, dtype=np.int64),
        n_steps=np.array(args.N - 1, dtype=np.int64),
        T=np.array(1.0, dtype=np.float64),
        eps=np.array(0.0, dtype=np.float64),
        sampler=np.array("replica_exchange_umbrella_wham"),
        athermal_target=np.array(True),
        bias_removed=np.array(True),
        bias_convention=np.array("B_j(m)=0.5*k*(m-m0_j)^2; sampling weight exp(-B_j)"),
        n_workers=np.array(centers.size, dtype=np.int64),
        n_processes=np.array(simulation["n_processes"], dtype=np.int64),
        steps_per_window=np.array(args.steps_per_window, dtype=np.int64),
        steps_per_worker=np.array(args.steps_per_window, dtype=np.int64),
        total_attempted_steps=np.array(int(attempted.sum()), dtype=np.int64),
        base_seed=np.array(args.base_seed, dtype=np.int64),
        worker_seeds=np.asarray(
            [state["seed"] for state in states], dtype=np.int64
        ),
        burnin=np.array(
            simulation["burn_steps"] / args.steps_per_window, dtype=np.float64
        ),
        burn_steps=np.array(simulation["burn_steps"], dtype=np.int64),
        sample_every=np.array(args.sample_every, dtype=np.int64),
        exchange_every=np.array(args.exchange_every, dtype=np.int64),
        n_samples=np.array(contacts.size, dtype=np.int64),
        accepted_moves_per_worker=accepted,
        geometrically_valid_moves_per_worker=geometrically_valid,
        acceptance_ratios_per_worker=acceptance,
        combined_acceptance_ratio=np.array(
            accepted.sum() / attempted.sum(), dtype=np.float64
        ),
        samples_per_worker=analysis["per_walker_sample_count"],
        per_worker_m_mean=analysis["per_walker_m_mean"],
        per_worker_rg_mean=analysis["per_walker_rg_mean"],
        kappa_bend=np.array(0.0, dtype=np.float64),
        bending_enabled=np.array(False),
        bend_definition=np.array(BEND_DEFINITION),
        umbrella_k=np.array(args.umbrella_k, dtype=np.float64),
        umbrella_window_spacing=np.array(args.window_spacing, dtype=np.int64),
        umbrella_window_centers=np.asarray(centers, dtype=np.int64),
        umbrella_bias_matrix=analysis["bias_matrix"],
        umbrella_initialization=np.asarray(simulation["initialization"], dtype="U16"),
        umbrella_window_sample_counts=np.asarray(
            wham["samples_per_window"], dtype=np.int64
        ),
        umbrella_window_contact_histograms=analysis["window_histograms"],
        umbrella_per_window_m_mean=analysis["per_window_m_mean"],
        umbrella_empirical_adjacent_overlap=analysis["empirical_overlap"],
        umbrella_wham_predicted_adjacent_overlap=analysis["predicted_overlap"],
        umbrella_swap_attempts=np.asarray(
            simulation["swap_attempts"], dtype=np.int64
        ),
        umbrella_swap_accepts=np.asarray(
            simulation["swap_accepts"], dtype=np.int64
        ),
        umbrella_swap_acceptance=analysis["swap_rate"],
        umbrella_wham_predicted_swap_acceptance=analysis[
            "predicted_swap_acceptance"
        ],
        umbrella_exchange_rounds=np.array(
            simulation["exchange_rounds"], dtype=np.int64
        ),
        umbrella_walker_round_trips=np.asarray(
            simulation["round_trips"], dtype=np.int64
        ),
        umbrella_total_round_trips=np.array(
            analysis["total_round_trips"], dtype=np.int64
        ),
        umbrella_pull_move_weight=np.array(
            args.pull_move_weight, dtype=np.float64
        ),
        wham_free_energies=np.asarray(
            wham["free_energies"], dtype=np.float64
        ),
        wham_log_sample_weight_by_contact=np.asarray(
            wham["log_sample_weight_by_contact"], dtype=np.float64
        ),
        wham_iterations=np.array(wham["iterations"], dtype=np.int64),
        wham_converged=np.array(wham["converged"]),
        wham_tolerance=np.array(args.wham_tolerance, dtype=np.float64),
        wham_max_delta=np.array(wham["max_delta"], dtype=np.float64),
        wham_window_normalization_error=np.array(
            wham["normalization_error"], dtype=np.float64
        ),
        wham_contact_histogram_error=np.array(
            analysis["wham_histogram_error"], dtype=np.float64
        ),
        importance_effective_sample_size=np.array(
            analysis["importance_ess"], dtype=np.float64
        ),
        importance_effective_fraction=np.array(
            analysis["importance_ess"] / contacts.size, dtype=np.float64
        ),
        n_samples_effective=np.array(
            analysis["importance_ess"], dtype=np.float64
        ),
        c_blocked_stderr=analysis["c_blocked_stderr"],
        c_block_estimates=analysis["c_block_estimates"],
        c_blocked_batch_count=np.array(
            analysis["blocked_count"], dtype=np.int64
        ),
        n_blocks=np.array(args.n_blocks, dtype=np.int64),
        min_samples_per_level=np.array(
            args.min_samples_per_level, dtype=np.int64
        ),
        min_adjacent_overlap=np.array(
            args.min_adjacent_overlap, dtype=np.float64
        ),
        min_swap_acceptance=np.array(
            args.min_swap_acceptance, dtype=np.float64
        ),
        min_round_trips=np.array(args.min_round_trips, dtype=np.int64),
        excluded_contact_levels=np.asarray(
            sorted(args.excluded_contact_levels), dtype=np.int64
        ),
        joint_contact_marginal_error=np.array(
            built["marginal_m_error"], dtype=np.float64
        ),
        joint_rg_marginal_error=np.array(
            built["marginal_rg_error"], dtype=np.float64
        ),
        total_wall_seconds=np.array(
            simulation["elapsed_seconds"], dtype=np.float64
        ),
        raw_samples_resampled=np.array(bool(args.save_raw_samples)),
        raw_samples_available=np.array(bool(args.save_raw_samples)),
        raw_samples_warning=np.array(RAW_SAMPLES_WARNING),
        raw_samples_unique_fraction=np.array(float("nan"), dtype=np.float64),
    )
    if args.save_raw_samples:
        resampled_indices = systematic_resample(
            analysis["weights"],
            np.random.default_rng(args.base_seed + 20_000_033),
        )
        save["raw_samples_unique_fraction"] = np.array(
            np.unique(resampled_indices).size / resampled_indices.size,
            dtype=np.float64,
        )
        save["c_samples_resampled"] = contacts[resampled_indices]
        save["rg_samples_resampled"] = radii[resampled_indices]
        save["bend_samples_resampled"] = bends[resampled_indices]
        if args.legacy_sample_aliases:
            save["c_samples"] = save["c_samples_resampled"]
            save["rg_samples"] = save["rg_samples_resampled"]
            save["bend_samples"] = save["bend_samples_resampled"]
    _atomic_savez(output_path, save)
    return output_path


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate the uniform athermal single-chain baseline with "
            "replica-exchange harmonic umbrellas in contact number and WHAM."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--N", type=int, default=44, help="number of beads")
    parser.add_argument(
        "--m_min", type=int, default=0,
        help="lowest contact level; must be zero for a full athermal baseline",
    )
    parser.add_argument(
        "--m_max", type=int, default=None,
        help="geometric contact maximum; known values for N=30,44,60 are automatic",
    )
    parser.add_argument(
        "--umbrella_k", type=float, default=DEFAULT_UMBRELLA_K,
        help="dimensionless harmonic force constant in B=k/2*(m-m0)^2",
    )
    parser.add_argument(
        "--window_spacing", type=int, default=DEFAULT_WINDOW_SPACING,
        help="integer spacing between harmonic centers",
    )
    parser.add_argument(
        "--steps_per_window", type=int, default=20_000_000,
        help="attempted local moves for every logical umbrella walker",
    )
    parser.add_argument(
        "--burnin", type=float, default=0.30,
        help="fraction of steps discarded before production sampling",
    )
    parser.add_argument(
        "--sample_every", type=int, default=500,
        help="record every this many attempted local moves after burn-in",
    )
    parser.add_argument(
        "--exchange_every", type=int, default=1_000,
        help="attempt alternating adjacent replica exchanges at this interval",
    )
    parser.add_argument(
        "--pull_move_weight", type=float, default=0.25,
        help="probability of the Hastings-corrected pull move; remaining weight "
             "is split among pivot, corner-flip, and end moves",
    )
    parser.add_argument(
        "--init", choices=("auto", "rod", "compact"), default="auto",
        help="initial conformation; auto uses rods below and compact seeds above midrange",
    )
    parser.add_argument(
        "--n_processes", type=int, default=None,
        help="worker processes; defaults to SLURM_CPUS_PER_TASK or local CPU count, "
             "capped at the number of windows",
    )
    parser.add_argument("--base_seed", type=int, default=42)
    parser.add_argument("--rg_bins", type=int, default=60)
    parser.add_argument("--no_joint", action="store_true", help="omit P(m,Rg)")
    parser.add_argument("--dist_dir", type=str, default="dists")
    parser.add_argument(
        "--output", type=str, default=None,
        help="exact output NPZ path; otherwise construct a name in --dist_dir",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="restartable umbrella checkpoint NPZ written atomically",
    )
    parser.add_argument(
        "--resume_checkpoint", type=str, default=None,
        help="resume chains, RNG streams, swaps, and samples from this checkpoint",
    )
    parser.add_argument(
        "--checkpoint_every_seconds", type=float, default=1_800.0,
        help="periodic checkpoint interval",
    )
    parser.add_argument(
        "--max_wall_seconds", type=float, default=math.inf,
        help="gracefully checkpoint and stop this invocation before this wall time",
    )
    parser.add_argument(
        "--shutdown_margin_seconds", type=float, default=120.0,
        help="time reserved to write the final checkpoint before max wall time",
    )
    parser.add_argument("--wham_tolerance", type=float, default=1e-12)
    parser.add_argument("--wham_max_iterations", type=int, default=100_000)
    parser.add_argument(
        "--n_blocks", type=int, default=20,
        help="contiguous time blocks that independently re-solve WHAM for errors",
    )
    parser.add_argument(
        "--min_samples_per_level", type=int, default=1_000,
        help="minimum pooled biased samples at each required contact level; zero disables",
    )
    parser.add_argument(
        "--min_adjacent_overlap", type=float, default=0.10,
        help="minimum empirical overlap coefficient for every adjacent window pair",
    )
    parser.add_argument(
        "--min_swap_acceptance", type=float, default=0.10,
        help="minimum observed exchange acceptance for every adjacent pair",
    )
    parser.add_argument(
        "--min_round_trips", type=int, default=1,
        help="minimum total low-window to high-window to low-window walker trips",
    )
    parser.add_argument(
        "--excluded_contact_levels", type=int, nargs="*", default=[],
        help="independently verified unreachable levels exempted from coverage checks",
    )
    parser.add_argument(
        "--save_raw_samples", dest="save_raw_samples", action="store_true", default=True,
        help="save an athermal systematic importance-resample",
    )
    parser.add_argument(
        "--no_save_raw_samples", dest="save_raw_samples", action="store_false"
    )
    parser.add_argument(
        "--legacy_sample_aliases", action="store_true",
        help="also expose resampled arrays under deprecated c_samples/rg_samples names",
    )
    parser.add_argument(
        "--show_window_design", action="store_true",
        help="print resolved contact range, centers, and harmonic width, then exit",
    )
    parser.add_argument(
        "--self-test", action="store_true",
        help="run exact-estimator and short end-to-end validation, then exit",
    )
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if args.N < 3:
        raise ValueError("--N must be at least 3")
    if args.m_min != 0:
        raise ValueError("--m_min must be zero for a full athermal baseline")
    verified = geometric_contact_maximum(args.N)
    if verified is not None:
        if args.m_max is not None and args.m_max != verified:
            raise ValueError(
                f"--m_max={args.m_max} conflicts with verified maximum {verified} "
                f"for N={args.N}"
            )
        args.m_max = verified
    elif args.m_max is None:
        raise ValueError(
            "--m_max is required when no independently verified exact maximum is encoded"
        )
    if args.m_max < args.m_min:
        raise ValueError("--m_max must be at least --m_min")
    upper = contact_upper_bound(args.N)
    if args.m_max > upper:
        raise ValueError(
            f"--m_max={args.m_max} exceeds rigorous cubic-lattice upper bound "
            f"{upper} for N={args.N}"
        )
    if not math.isfinite(args.umbrella_k) or args.umbrella_k <= 0.0:
        raise ValueError("--umbrella_k must be finite and positive")
    if args.window_spacing < 1:
        raise ValueError("--window_spacing must be positive")
    if args.steps_per_window < 1:
        raise ValueError("--steps_per_window must be positive")
    if not 0.0 <= args.burnin < 1.0:
        raise ValueError("--burnin must lie in [0,1)")
    if args.sample_every < 1 or args.exchange_every < 1:
        raise ValueError("sample and exchange intervals must be positive")
    burn_steps = int(round(args.burnin * args.steps_per_window))
    if (args.steps_per_window - burn_steps) // args.sample_every < 1:
        raise ValueError("run settings yield zero post-burn-in samples per window")
    if not 0.0 <= args.pull_move_weight <= 1.0:
        raise ValueError("--pull_move_weight must lie in [0,1]")
    if args.n_processes is not None and args.n_processes < 1:
        raise ValueError("--n_processes must be positive")
    if args.rg_bins < 1:
        raise ValueError("--rg_bins must be positive")
    if args.checkpoint_every_seconds <= 0.0:
        raise ValueError("--checkpoint_every_seconds must be positive")
    if args.max_wall_seconds <= 0.0:
        raise ValueError("--max_wall_seconds must be positive")
    if args.shutdown_margin_seconds < 0.0:
        raise ValueError("--shutdown_margin_seconds must be nonnegative")
    if math.isfinite(args.max_wall_seconds):
        if args.max_wall_seconds <= args.shutdown_margin_seconds:
            raise ValueError("max wall time must exceed the shutdown margin")
        if not (args.checkpoint or args.resume_checkpoint):
            raise ValueError("finite --max_wall_seconds requires a checkpoint path")
    if not math.isfinite(args.wham_tolerance) or args.wham_tolerance <= 0.0:
        raise ValueError("--wham_tolerance must be finite and positive")
    if args.wham_max_iterations < 1:
        raise ValueError("--wham_max_iterations must be positive")
    if args.n_blocks < 2:
        raise ValueError("--n_blocks must be at least 2")
    if args.min_samples_per_level < 0:
        raise ValueError("--min_samples_per_level must be nonnegative")
    if not 0.0 <= args.min_adjacent_overlap <= 1.0:
        raise ValueError("--min_adjacent_overlap must lie in [0,1]")
    if not 0.0 <= args.min_swap_acceptance <= 1.0:
        raise ValueError("--min_swap_acceptance must lie in [0,1]")
    if args.min_round_trips < 0:
        raise ValueError("--min_round_trips must be nonnegative")
    excluded = sorted(args.excluded_contact_levels)
    if len(set(excluded)) != len(excluded):
        raise ValueError("--excluded_contact_levels contains duplicates")
    invalid = [level for level in excluded if not args.m_min <= level <= args.m_max]
    if invalid:
        raise ValueError(f"excluded contact levels outside the range: {invalid}")
    if args.legacy_sample_aliases and not args.save_raw_samples:
        raise ValueError("--legacy_sample_aliases requires --save_raw_samples")
    if args.resume_checkpoint and not Path(args.resume_checkpoint).exists():
        raise FileNotFoundError(args.resume_checkpoint)


def print_window_design(args: argparse.Namespace, centers: np.ndarray) -> None:
    sigma = 1.0 / math.sqrt(args.umbrella_k)
    adjacent_penalty = 0.5 * args.umbrella_k * args.window_spacing ** 2
    print("=== umbrella design ===", flush=True)
    print(f"N                              : {args.N}", flush=True)
    print(f"physical contact range         : {args.m_min}..{args.m_max}", flush=True)
    print(f"harmonic centers               : {centers.tolist()}", flush=True)
    print(f"number of windows              : {centers.size}", flush=True)
    print(f"dimensionless k                : {args.umbrella_k:g}", flush=True)
    print(f"Gaussian width 1/sqrt(k)       : {sigma:.4f} contacts", flush=True)
    print(
        f"bias at one center spacing     : {adjacent_penalty:.4f}", flush=True
    )
    if centers[-1] > args.m_max:
        print(
            f"upper center {int(centers[-1])} lies beyond m_max intentionally; "
            "it concentrates the endpoint window at the compact limit.",
            flush=True,
        )


def run_self_test() -> int:
    print("Self-test: WHAM algebra and a short exact N=6 umbrella simulation")
    checks: Dict[str, bool] = {}

    contact_values = np.arange(3, dtype=np.int64)
    centers = make_window_centers(0, 2, DEFAULT_WINDOW_SPACING)
    bias = harmonic_bias_matrix(centers, contact_values, DEFAULT_UMBRELLA_K)
    exact_probability = np.asarray([0.68, 0.24, 0.08], dtype=np.float64)
    partition = np.sum(exact_probability[None, :] * np.exp(-bias), axis=1)
    exact_window_probability = (
        exact_probability[None, :] * np.exp(-bias) / partition[:, None]
    )
    synthetic_histograms = 100_000.0 * exact_window_probability
    synthetic = solve_wham(
        synthetic_histograms, bias, tolerance=1e-14, max_iterations=100_000
    )
    synthetic_error = float(
        np.max(np.abs(synthetic["probability"] - exact_probability))
    )
    checks["WHAM exactly removes known harmonic biases"] = synthetic_error < 1e-11

    x, y = 1, 2
    full_swap = bias[0, x] + bias[1, y] - bias[0, y] - bias[1, x]
    simple_swap = DEFAULT_UMBRELLA_K * (centers[1] - centers[0]) * (x - y)
    checks["replica-exchange exponent has the correct sign"] = bool(
        abs(float(full_swap - simple_swap)) < 1e-14 and simple_swap < 0.0
    )

    exact_values, exact_contact_probability, exact_radii = enumerate_rooted_saws(6)
    test_args = parse_args([])
    test_args.N = 6
    test_args.m_min = 0
    test_args.m_max = int(exact_values.max())
    test_args.umbrella_k = DEFAULT_UMBRELLA_K
    test_args.window_spacing = DEFAULT_WINDOW_SPACING
    test_args.steps_per_window = 160_000
    test_args.burnin = 0.20
    test_args.sample_every = 10
    test_args.exchange_every = 50
    test_args.pull_move_weight = 0.0
    test_args.init = "rod"
    test_args.n_processes = 1
    test_args.base_seed = 8128
    test_args.rg_bins = 24
    test_args.no_joint = False
    test_args.checkpoint = None
    test_args.resume_checkpoint = None
    test_args.max_wall_seconds = math.inf
    test_args.n_blocks = 8
    test_args.min_samples_per_level = 0
    test_args.min_adjacent_overlap = 0.0
    test_args.min_swap_acceptance = 0.0
    test_args.min_round_trips = 0
    validate_args(test_args)
    test_centers = make_window_centers(
        test_args.m_min, test_args.m_max, test_args.window_spacing
    )
    simulation = run_replica_exchange(test_args, test_centers, progress=False)
    analysis = analyse_samples(
        test_args, test_centers, simulation, enforce_checks=False
    )
    estimated = analysis["built"]["c_prob"]
    exact_full = np.zeros(test_args.m_max + 1, dtype=np.float64)
    exact_full[exact_values] = exact_contact_probability
    total_variation = 0.5 * float(np.abs(estimated - exact_full).sum())
    estimated_mean_rg = float(
        np.dot(analysis["weights"], simulation["samples"]["rg_samples"])
    )
    exact_mean_rg = float(exact_radii.mean())
    relative_rg_error = abs(estimated_mean_rg - exact_mean_rg) / exact_mean_rg
    checks["short REUS/WHAM run recovers exact P(m)"] = total_variation < 0.035
    checks["short REUS/WHAM run recovers exact mean Rg"] = relative_rg_error < 0.035
    checks["joint distribution preserves both marginals"] = bool(
        analysis["built"]["marginal_m_error"] < 1e-12
        and analysis["built"]["marginal_rg_error"] < 1e-12
    )
    checks["every umbrella contributes equally scheduled samples"] = bool(
        len(set(analysis["wham"]["samples_per_window"].tolist())) == 1
    )

    print(f"  synthetic WHAM max error : {synthetic_error:.3e}")
    print(f"  exact P(m)               : {exact_full.tolist()}")
    print(f"  estimated P(m)           : {estimated.tolist()}")
    print(f"  contact TVD              : {total_variation:.6g}")
    print(f"  mean Rg relative error   : {relative_rg_error:.6g}")
    print(f"  empirical overlaps       : {analysis['empirical_overlap'].tolist()}")
    print(f"  exchange acceptance      : {analysis['swap_rate'].tolist()}")
    for description, passed in checks.items():
        print(f"  {'PASS' if passed else 'FAIL'}: {description}")
    if all(checks.values()):
        print("SELF-TEST PASSED")
        return 0
    print("SELF-TEST FAILED")
    return 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        return run_self_test()
    validate_args(args)
    centers = make_window_centers(args.m_min, args.m_max, args.window_spacing)
    print_window_design(args, centers)
    if args.show_window_design:
        return 0

    expected_per_window = (
        args.steps_per_window - int(round(args.burnin * args.steps_per_window))
    ) // args.sample_every
    print(
        f"Expected production samples: {expected_per_window} per window, "
        f"{expected_per_window * centers.size} total.",
        flush=True,
    )
    simulation = run_replica_exchange(args, centers)
    if not simulation["complete"]:
        print("Run segment checkpointed successfully; no partial baseline was written.")
        return 0
    analysis = analyse_samples(args, centers, simulation)
    output_path = write_output(args, centers, simulation, analysis)

    built = analysis["built"]
    pooled_counts = analysis["window_histograms"].sum(axis=0)
    print("\n=== unbiased athermal result ===", flush=True)
    print(f"P(m) sum                         : {built['c_prob'].sum():.12g}", flush=True)
    print(f"P(Rg) sum                        : {built['rg_prob'].sum():.12g}", flush=True)
    if not args.no_joint:
        print(f"joint sum                        : {built['crg_prob'].sum():.12g}", flush=True)
        print(
            f"joint -> P(m) max error          : {built['marginal_m_error']:.3e}",
            flush=True,
        )
        print(
            f"joint -> P(Rg) max error         : {built['marginal_rg_error']:.3e}",
            flush=True,
        )
    print(
        f"WHAM iterations / final delta    : {analysis['wham']['iterations']} / "
        f"{analysis['wham']['max_delta']:.3e}",
        flush=True,
    )
    print(
        f"importance ESS                   : {analysis['importance_ess']:.1f}/"
        f"{simulation['samples']['contact_samples'].size}",
        flush=True,
    )
    print(
        f"pooled visits per contact level  : {int(pooled_counts.min())}.."
        f"{int(pooled_counts.max())}",
        flush=True,
    )
    print(
        f"empirical adjacent overlap       : "
        f"{float(analysis['empirical_overlap'].min()):.3f}.."
        f"{float(analysis['empirical_overlap'].max()):.3f}",
        flush=True,
    )
    print(
        f"observed adjacent swap acceptance: "
        f"{float(np.nanmin(analysis['swap_rate'])):.3f}.."
        f"{float(np.nanmax(analysis['swap_rate'])):.3f}",
        flush=True,
    )
    print(
        f"walker low-high-low round trips  : {analysis['total_round_trips']}",
        flush=True,
    )
    print(f"total wall time                  : {simulation['elapsed_seconds']:.1f}s", flush=True)
    print(f"DIST_FILE = {output_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
