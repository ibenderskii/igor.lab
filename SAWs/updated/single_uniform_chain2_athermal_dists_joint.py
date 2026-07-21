#!/usr/bin/env python3
"""
Athermal (non-interacting) self-avoiding walk (SAW) on a 3D cubic lattice.

- Uses pivot / crankshaft / end moves.
- Accepts every geometrically valid move (no energetics).
- Measures the *non-bonded* nearest-neighbour contact count m and R_g.
- Outputs equilibrium distributions:
    * P(m) as (c_vals, c_prob) over integer m
    * P(Rg) as (rg_edges, rg_prob)
    * joint P(m, Rg) as a 2D histogram (c_edges, rg_edges, crg_prob)

The joint distribution is useful as an athermal baseline to predict
P(Rg | T) for thermodynamic models whose energy depends only on m via
reweighting by exp[-(h/T - s) m].

Parallel sampling
-----------------
This script runs ``--n_workers`` statistically independent Markov chains in
parallel (one per CPU core via ``ProcessPoolExecutor``).  Each worker uses a
deterministic per-worker seed ``base_seed + k`` and its own private simulation
state (chain, occupancy set, Python RNG, NumPy RNG).  After all workers finish,
the parent process pools their post-burn-in samples and builds a single combined
baseline NPZ file.  Workers never touch shared mutable state and never write the
output file -- only the parent writes the combined result.
"""

from __future__ import annotations
import argparse
import os as _os
import random
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from itertools import permutations, product
from typing import Tuple, List, Set, Dict, Any

import numpy as np

from lattice_bending import BEND_DEFINITION, count_bends, delta_bends

Vec = Tuple[int, int, int]
NN_VECS: List[Vec] = [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]

def add(a: Vec, b: Vec) -> Vec:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])

def sub(a: Vec, b: Vec) -> Vec:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])

# ------------------------- rotations (proper cubic rotations) -------------------------
def _det3(M: Tuple[Tuple[int,int,int], Tuple[int,int,int], Tuple[int,int,int]]) -> int:
    (a,b,c), (d,e,f), (g,h,i) = M
    return a*(e*i - f*h) - b*(d*i - f*g) + c*(d*h - e*g)

_IDENTITY: Tuple[Tuple[int,int,int], Tuple[int,int,int], Tuple[int,int,int]] = (
    (1, 0, 0), (0, 1, 0), (0, 0, 1)
)

def generate_cubic_rotations() -> List[Tuple[Tuple[int,int,int], Tuple[int,int,int], Tuple[int,int,int]]]:
    """All 23 *non-identity* orientation-preserving signed-permutation 3x3 matrices.

    The identity matrix is excluded so that a pivot proposal which leaves the
    chain unchanged is never proposed (and hence never miscounted as an accepted
    move).
    """
    rots = []
    for perm in permutations([0, 1, 2]):
        for signs in product([-1, 1], repeat=3):
            M = [[0,0,0],[0,0,0],[0,0,0]]
            for r, c in enumerate(perm):
                M[r][c] = signs[r]
            Mt = (tuple(M[0]), tuple(M[1]), tuple(M[2]))
            if _det3(Mt) == 1 and Mt != _IDENTITY:
                rots.append(Mt)
    return rots  # 23 non-identity rotations total

ROT_MATS = generate_cubic_rotations()

def apply_rot(M, v: Vec) -> Vec:
    x, y, z = v
    return (
        M[0][0]*x + M[0][1]*y + M[0][2]*z,
        M[1][0]*x + M[1][1]*y + M[1][2]*z,
        M[2][0]*x + M[2][1]*y + M[2][2]*z,
    )

# ------------------------- observables -------------------------
def contact_count(chain: List[Vec], occ: Set[Vec]) -> int:
    """Unique non-bonded nearest-neighbour contacts (integer)."""
    m = 0
    N = len(chain)
    for i, r in enumerate(chain):
        prev = chain[i-1] if i > 0   else None
        nxt  = chain[i+1] if i < N-1 else None
        for v in NN_VECS:
            nbr = add(r, v)
            if nbr in occ and nbr not in (prev, nxt):
                m += 1
    return m // 2  # each contact seen from both endpoints

def radius_of_gyration(chain: List[Vec]) -> float:
    r = np.array(chain, dtype=float)
    com = r.mean(axis=0)
    return float(math.sqrt(((r - com)**2).sum(axis=1).mean()))

# ------------------------- moves (geometric validity; energetics in the caller) -------------------------
# NOTE: all move functions take an explicit ``rng`` (a random.Random instance)
# rather than calling the global ``random`` module, so that parallel workers
# stay statistically independent and reproducible.
#
# Each move returns ``(ok, new_chain, new_occ, changed)`` where ``changed`` lists
# the bead indices whose bend bookkeeping must be revisited; it is fed straight to
# ``delta_bends``.  Rejected proposals return an empty ``changed``.
def attempt_pivot(chain: List[Vec], occ: Set[Vec], rng: random.Random):
    n = len(chain)
    i = rng.randrange(1, n-1)  # pivot monomer (not ends)

    head = chain[:i+1]
    tail = chain[i+1:]
    pivot = chain[i]

    M = rng.choice(ROT_MATS)

    new_tail: List[Vec] = []
    new_occ: Set[Vec] = set(head)

    for r in tail:
        dr = sub(r, pivot)
        r2 = add(pivot, apply_rot(M, dr))
        if r2 in new_occ:
            return False, chain, occ, ()
        new_tail.append(r2)
        new_occ.add(r2)

    # A pivot rotates the whole tail RIGIDLY: every bond from index i onwards is
    # rotated by the same matrix M, so all interior tail angles are preserved and
    # only the angle at center i can change.  Reporting bead i+1 (the first bead
    # that actually moved) makes delta_bends inspect centers {i, i+1, i+2}, which
    # covers center i exactly; the other two contribute 0.  This keeps the pivot's
    # bend bookkeeping O(1) instead of O(N).  sanity_check_bends() and
    # SAW_DEBUG_BENDS both verify this invariant.
    return True, head + new_tail, new_occ, (i + 1,)

def attempt_crankshaft(chain: List[Vec], occ: Set[Vec], rng: random.Random):
    """Local kink flip (crankshaft). Works only for perfect 90° kinks."""
    n = len(chain)
    i = rng.randrange(1, n-1)
    a, b, c = chain[i-1], chain[i], chain[i+1]

    u1 = sub(b, a)
    u2 = sub(c, b)

    if u1 not in NN_VECS or u2 not in NN_VECS:
        return False, chain, occ, ()

    # reject straight segments (parallel or anti-parallel)
    if u1 == u2 or u1 == (-u2[0], -u2[1], -u2[2]):
        return False, chain, occ, ()

    # require perfect 90° kink
    if (u1[0]*u2[0] + u1[1]*u2[1] + u1[2]*u2[2]) != 0:
        return False, chain, occ, ()

    b_new = add(a, u2)
    if b_new in occ:
        return False, chain, occ, ()

    # bonds must remain unit length
    if sub(b_new, a) not in NN_VECS or sub(c, b_new) not in NN_VECS:
        return False, chain, occ, ()

    new_chain = chain.copy()
    new_chain[i] = b_new
    new_occ = (occ - {b}) | {b_new}
    return True, new_chain, new_occ, (i,)

def attempt_end_move(chain: List[Vec], occ: Set[Vec], rng: random.Random):
    """Symmetric end move: move one end to a random empty neighbor of its anchor.

    The proposed new endpoint position is chosen relative to the *anchor*
    (the monomer adjacent to the end), not the old endpoint, and the old
    endpoint is removed from the occupancy set before testing the proposal.
    """
    n = len(chain)
    end = 0 if rng.random() < 0.5 else n - 1
    anchor = 1 if end == 0 else n - 2

    old = chain[end]
    occ_without_old = occ - {old}

    v = rng.choice(NN_VECS)
    r_new = add(chain[anchor], v)

    if r_new == old:
        return False, chain, occ, ()
    if r_new in occ_without_old:
        return False, chain, occ, ()

    new_chain = chain.copy()
    new_chain[end] = r_new
    new_occ = occ_without_old | {r_new}
    return True, new_chain, new_occ, (end,)


def sanity_check_end_move() -> None:
    """Verify that the end move can accept from a simple straight chain."""
    chain = [(0,0,0), (1,0,0), (2,0,0), (3,0,0), (4,0,0)]
    occ = set(chain)
    rng = random.Random(0)
    n_accept = sum(attempt_end_move(chain, occ, rng)[0] for _ in range(200))
    if n_accept <= 0:
        raise RuntimeError("attempt_end_move never accepted in 200 trials")

MOVE_FUNCS = [attempt_pivot, attempt_crankshaft, attempt_end_move]


def sanity_check_bends() -> None:
    """Verify the incremental bend delta against a full recount for every move.

    Cheap enough to run unconditionally at startup: it drives a few hundred
    proposals of each move type on a short chain and asserts that
    ``delta_bends(old, new, changed)`` matches ``count_bends(new) - count_bends(old)``.
    This is what guards the O(1) pivot shortcut (see :func:`attempt_pivot`).
    """
    # Staircase start: every interior site is a 90-degree kink, so the crankshaft
    # move (which only fires on perfect kinks) actually gets exercised.
    staircase: List[Vec] = [
        (0,0,0), (1,0,0), (1,1,0), (2,1,0), (2,2,0),
        (3,2,0), (3,3,0), (4,3,0), (4,4,0), (5,4,0),
    ]
    for move in MOVE_FUNCS:
        chain: List[Vec] = list(staircase)
        occ: Set[Vec] = set(chain)
        rng = random.Random(12345)
        n_valid = 0
        for _ in range(400):
            ok, chain_new, occ_new, changed = move(chain, occ, rng)
            if not ok:
                continue
            n_valid += 1
            dn = delta_bends(chain, chain_new, changed)
            dn_full = count_bends(chain_new) - count_bends(chain)
            if dn != dn_full:
                raise RuntimeError(
                    f"{move.__name__}: delta_bends={dn} disagrees with full "
                    f"recount delta {dn_full} (changed={changed})"
                )
            chain, occ = chain_new, occ_new
        if n_valid == 0:
            raise RuntimeError(f"{move.__name__} never produced a valid proposal")


def _parse_bool_env(name: str, default: bool = False) -> bool:
    """Parse a boolean environment variable (same convention as the REMD scripts)."""
    raw = _os.environ.get(name)
    if raw is None:
        return default
    v = raw.strip().lower()
    if v in ("0", "false", "no", "off", ""):
        return False
    if v in ("1", "true", "yes", "on"):
        return True
    return default


# Optional development cross-check: when SAW_DEBUG_BENDS is truthy in the
# environment, the sampler asserts that the incrementally tracked bend count
# equals count_bends(chain) after every accepted move.  Off by default so the
# expensive O(N) recount never runs in production.
_DEBUG_BENDS = _parse_bool_env("SAW_DEBUG_BENDS", False)

# ------------------------- per-worker simulation -------------------------
def run_independent_chain(
    worker_id: int,
    seed: int,
    N: int,
    steps: int,
    burnin: float,
    sample_every: int,
    kappa_bend: float = 0.0,
) -> Dict[str, Any]:
    """Run one complete, statistically independent SAW Markov chain.

    Every piece of mutable state (chain, occupancy set, Python RNG, NumPy RNG,
    sample buffers) is created locally so that this function is safe to run in a
    separate process with no shared state.  Sampling is keyed off *attempted*
    Monte Carlo moves, not accepted ones.

    The chain is contact-athermal but carries the fixed reduced bending penalty
    u_bend = kappa_bend * n_bend, so it samples P(X) ~ exp[-kappa_bend*n_bend(X)].
    At ``kappa_bend == 0`` every valid proposal has du == 0.0 and takes the
    ``du <= 0`` branch, so no acceptance random number is drawn and the RNG
    stream (and hence the sampled distributions) is identical to the original
    accept-every-valid-move behaviour.

    Returns a dict with the worker id, its seed, the post-burn-in contact, Rg and
    bend samples (as NumPy arrays), and accepted / attempted move counts.
    """
    # Explicit per-worker RNGs (no reliance on module-level global RNG state).
    py_rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)  # initialized for reproducibility / future use
    _ = np_rng  # currently unused by the moves; kept for forward compatibility

    # initial straight chain + occupancy (private to this worker)
    chain: List[Vec] = [(i, 0, 0) for i in range(N)]
    occ: Set[Vec] = set(chain)

    burn_steps = int(round(burnin * steps))
    burn_steps = max(0, min(burn_steps, steps))

    acc = 0
    n_bend = 0  # straight start chain has no bends
    C_samples: List[int] = []
    Rg_samples: List[float] = []
    B_samples: List[int] = []

    kappa_bend = float(kappa_bend)

    # ~10% progress reporting (prefixed + flushed so HPC logs stay readable)
    progress_mark = max(1, steps // 10)
    t0 = time.time()

    for step in range(1, steps + 1):
        move = py_rng.choice(MOVE_FUNCS)
        ok, chain_new, occ_new, changed = move(chain, occ, py_rng)
        if ok:
            dn = delta_bends(chain, chain_new, changed)
            du = kappa_bend * dn
            if du <= 0.0 or py_rng.random() < math.exp(-du):
                chain, occ = chain_new, occ_new
                n_bend += dn
                acc += 1
                if _DEBUG_BENDS:
                    assert n_bend == count_bends(chain), (
                        "tracked n_bend out of sync with count_bends after "
                        "accepted move"
                    )

        if step > burn_steps and (step - burn_steps) % sample_every == 0:
            C_samples.append(int(contact_count(chain, occ)))
            Rg_samples.append(float(radius_of_gyration(chain)))
            B_samples.append(int(n_bend))

        if step % progress_mark == 0:
            pct = 100.0 * step / steps
            print(
                f"[worker {worker_id} seed={seed}] {pct:5.1f}% "
                f"({step}/{steps}) acc={acc} samples={len(C_samples)} "
                f"elapsed={time.time()-t0:6.1f}s",
                flush=True,
            )

    return {
        "worker_id": int(worker_id),
        "seed": int(seed),
        "contact_samples": np.asarray(C_samples, dtype=np.int64),
        "rg_samples": np.asarray(Rg_samples, dtype=np.float64),
        "bend_samples": np.asarray(B_samples, dtype=np.int64),
        "accepted_moves": int(acc),
        "attempted_moves": int(steps),
        "wall_time": float(time.time() - t0),
    }

# ------------------------- main -------------------------
def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--N", type=int, default=44, help="chain length")
    ap.add_argument("--T", type=float, default=1.0, help="temperature (metadata-only; athermal model)")
    ap.add_argument("--eps", type=float, default=0.0, help="contact energy (metadata-only; athermal model)")
    ap.add_argument("--dist_dir", type=str, default="dists", help="where to write equilibrium distribution .npz")
    ap.add_argument("--rg_bins", type=int, default=60, help="number of histogram bins for P(Rg)")
    ap.add_argument("--burnin", type=float, default=0.3, help="burn-in fraction of steps (0..1)")
    ap.add_argument("--sample_every", type=int, default=1000, help="sample every this many attempted moves after burn-in")
    ap.add_argument("--no_joint", action="store_true", help="skip writing the joint P(m,Rg) histogram")
    ap.add_argument("--kappa-bend", dest="kappa_bend", type=float, default=0.0,
                    help="fixed reduced bending penalty: u_bend = kappa_bend * n_bend "
                         "(n_bend = number of 90-degree turns). 0 = original athermal SAW.")
    # parallel / reproducibility controls
    ap.add_argument("--n_workers", type=int, default=8, help="number of independent chains to run in parallel")
    ap.add_argument("--base_seed", type=int, default=42, help="base seed; worker k uses base_seed + k")
    ap.add_argument("--steps_per_worker", type=int, default=20_000_000,
                    help="attempted Monte Carlo moves performed by each worker")
    ap.add_argument("--save_raw_samples", dest="save_raw_samples", action="store_true", default=True,
                    help="store pooled raw c_samples/rg_samples in the NPZ (default: on)")
    ap.add_argument("--no_save_raw_samples", dest="save_raw_samples", action="store_false",
                    help="disable storing pooled raw samples in the NPZ")
    args = ap.parse_args()

    if args.N < 3:
        raise ValueError("--N must be >= 3")
    if args.steps_per_worker < 1:
        raise ValueError("--steps_per_worker must be >= 1")
    if args.n_workers < 1:
        raise ValueError("--n_workers must be >= 1")
    if args.rg_bins < 1:
        raise ValueError("--rg_bins must be >= 1")
    if not (0.0 <= args.burnin < 1.0):
        raise ValueError("--burnin must be in [0, 1)")
    if args.sample_every < 1:
        raise ValueError("--sample_every must be >= 1")
    if not math.isfinite(args.kappa_bend):
        raise ValueError(f"--kappa-bend must be finite, got {args.kappa_bend!r}")
    if args.kappa_bend < 0.0:
        raise ValueError(f"--kappa-bend must be >= 0, got {args.kappa_bend!r}")

    sanity_check_end_move()
    sanity_check_bends()

    # deterministic, unique per-worker seeds
    worker_seeds = [int(args.base_seed) + k for k in range(args.n_workers)]
    if len(set(worker_seeds)) != len(worker_seeds):
        raise ValueError("worker seeds are not unique (check base_seed / n_workers)")

    # expected number of samples per worker (sampling is on attempted steps)
    burn_steps = max(0, min(int(round(args.burnin * args.steps_per_worker)), args.steps_per_worker))
    post_burn = args.steps_per_worker - burn_steps
    expected_samples = post_burn // args.sample_every
    print(f"N={args.N}  n_workers={args.n_workers}  steps_per_worker={args.steps_per_worker}", flush=True)
    print(f"base_seed={args.base_seed}  worker_seeds={worker_seeds}", flush=True)
    print(f"burnin={args.burnin} ({burn_steps} steps)  sample_every={args.sample_every}", flush=True)
    print(f"kappa_bend={args.kappa_bend:g} ({'ON' if args.kappa_bend != 0.0 else 'off'}; "
          f"{BEND_DEFINITION})", flush=True)
    print(f"Expected samples per worker: {expected_samples}  "
          f"(total ~{expected_samples * args.n_workers})", flush=True)
    if expected_samples <= 0:
        raise ValueError(
            "Chosen settings produce zero post-burn-in samples per worker: "
            f"post_burn={post_burn}, sample_every={args.sample_every}. "
            "Increase --steps_per_worker, lower --burnin, or lower --sample_every."
        )

    # ---- run independent chains in parallel ----
    results: List[Dict[str, Any]] = []
    t_start = time.time()
    n_done = 0
    with ProcessPoolExecutor(max_workers=args.n_workers) as ex:
        futures = {
            ex.submit(
                run_independent_chain,
                k, worker_seeds[k], args.N, args.steps_per_worker,
                args.burnin, args.sample_every, args.kappa_bend,
            ): k
            for k in range(args.n_workers)
        }
        for fut in as_completed(futures):
            res = fut.result()
            results.append(res)
            n_done += 1
            att = res["attempted_moves"]
            accr = res["accepted_moves"] / att if att else float("nan")
            print(
                f"Worker {n_done}/{args.n_workers} completed: "
                f"id={res['worker_id']}, seed={res['seed']}, "
                f"samples={res['contact_samples'].size}, "
                f"acceptance={accr:.3f}, wall={res['wall_time']:.1f}s",
                flush=True,
            )

    # deterministic ordering for reproducible concatenation
    results.sort(key=lambda r: r["worker_id"])

    # ---- pool post-burn-in samples ----
    C_all = np.concatenate([r["contact_samples"] for r in results]) if results else np.array([], dtype=np.int64)
    Rg_all = np.concatenate([r["rg_samples"] for r in results]) if results else np.array([], dtype=np.float64)
    B_all = np.concatenate([r["bend_samples"] for r in results]) if results else np.array([], dtype=np.int64)

    accepted_per_worker = np.array([r["accepted_moves"] for r in results], dtype=np.int64)
    attempted_per_worker = np.array([r["attempted_moves"] for r in results], dtype=np.int64)
    acceptance_per_worker = accepted_per_worker / attempted_per_worker
    total_attempted = int(attempted_per_worker.sum())
    total_accepted = int(accepted_per_worker.sum())
    combined_acceptance = total_accepted / total_attempted if total_attempted else float("nan")

    samples_per_worker = np.array([r["contact_samples"].size for r in results], dtype=np.int64)
    total_samples = int(samples_per_worker.sum())
    if total_samples != int(C_all.size):
        raise RuntimeError(
            f"pooled sample count {C_all.size} != sum of worker counts {total_samples}"
        )
    if total_samples == 0:
        raise RuntimeError("no post-burn-in samples were collected by any worker")

    C_arr = C_all.astype(np.int64)
    Rg_arr = Rg_all.astype(np.float64)
    B_arr = B_all.astype(np.int64)

    # bend summary over the samples already being accumulated (same shape as P(m))
    bend_vals, bend_counts = np.unique(B_arr, return_counts=True)
    bend_prob = bend_counts.astype(float) / bend_counts.sum()
    mean_bends = float(B_arr.mean())

    # ---- build pooled distributions from raw samples ----
    # P(m): exact over observed integer contact counts
    c_vals, c_counts = np.unique(C_arr, return_counts=True)
    c_prob = c_counts.astype(float) / c_counts.sum()

    # P(Rg): histogram on a single common set of edges from all pooled samples
    rg = Rg_arr[np.isfinite(Rg_arr)]
    rg_min = float(rg.min()); rg_max = float(rg.max())
    pad = 1e-9 if rg_max <= rg_min else 0.02 * (rg_max - rg_min)
    rg_edges = np.linspace(rg_min - pad, rg_max + pad, int(args.rg_bins) + 1)
    rg_counts, _ = np.histogram(rg, bins=rg_edges)
    rg_prob = rg_counts.astype(float)
    if rg_prob.sum() > 0:
        rg_prob /= rg_prob.sum()

    # joint P(m, Rg): integer-centered contact edges spanning the full range
    c_min = int(C_arr.min())
    c_max = int(C_arr.max())
    c_edges = np.arange(c_min - 0.5, c_max + 1.5, 1.0, dtype=float)
    if not args.no_joint:
        crg_counts, _, _ = np.histogram2d(
            C_arr.astype(float), rg, bins=[c_edges, rg_edges]
        )
        crg_prob = crg_counts / crg_counts.sum() if crg_counts.sum() > 0 else crg_counts
    else:
        crg_prob = np.array([[]], dtype=float)

    # ---- marginal consistency checks (on a common full integer contact grid) ----
    full_c = np.arange(c_min, c_max + 1)
    # c_prob mapped onto the full integer grid
    c_prob_full = np.zeros(full_c.size, dtype=float)
    c_prob_full[c_vals - c_min] = c_prob

    if not args.no_joint and crg_prob.size > 0:
        joint_over_rg = crg_prob.sum(axis=1)   # P(m) implied by the joint, full grid
        joint_over_c = crg_prob.sum(axis=0)    # P(Rg) implied by the joint
        marg_m_err = float(np.max(np.abs(joint_over_rg - c_prob_full)))
        marg_rg_err = float(np.max(np.abs(joint_over_c - rg_prob)))
    else:
        marg_m_err = float("nan")
        marg_rg_err = float("nan")

    # per-worker means (to spot non-equilibrated / outlier chains)
    per_worker_m_mean = np.array(
        [float(r["contact_samples"].mean()) if r["contact_samples"].size else float("nan")
         for r in results]
    )
    per_worker_rg_mean = np.array(
        [float(r["rg_samples"].mean()) if r["rg_samples"].size else float("nan")
         for r in results]
    )

    # ---- write the single combined NPZ (parent only) ----
    dist_dir = Path(args.dist_dir)
    dist_dir.mkdir(parents=True, exist_ok=True)
    # The kappa tag is appended ONLY for a nonzero penalty so that default
    # (kappa_bend = 0) filenames stay byte-identical to the existing artifacts,
    # while a stiff baseline can never silently overwrite an athermal one.
    kappa_tag = "" if args.kappa_bend == 0.0 else f"_kappa{args.kappa_bend:g}"
    dist_file = dist_dir / (
        f"{Path(__file__).stem}_N{args.N}_workers{args.n_workers}"
        f"_steps{args.steps_per_worker}_seed{args.base_seed}{kappa_tag}.npz"
    )

    save_kwargs: Dict[str, Any] = dict(
        # distributions (downstream-fitting field names preserved)
        c_vals=c_vals, c_prob=c_prob,
        c_edges=c_edges,
        rg_edges=rg_edges, rg_prob=rg_prob,
        crg_prob=crg_prob,
        # metadata
        N=int(args.N), T=float(args.T), eps=float(args.eps),
        n_workers=int(args.n_workers),
        steps_per_worker=int(args.steps_per_worker),
        total_attempted_steps=int(total_attempted),
        base_seed=int(args.base_seed),
        worker_seeds=np.asarray(worker_seeds, dtype=np.int64),
        burnin=float(args.burnin), sample_every=int(args.sample_every),
        n_samples=int(total_samples),
        accepted_moves_per_worker=accepted_per_worker,
        acceptance_ratios_per_worker=acceptance_per_worker,
        combined_acceptance_ratio=float(combined_acceptance),
        rg_bins=int(args.rg_bins),
        samples_per_worker=samples_per_worker,
        per_worker_m_mean=per_worker_m_mean,
        per_worker_rg_mean=per_worker_rg_mean,
        # bending metadata (additive; legacy readers ignore these)
        kappa_bend=float(args.kappa_bend),
        bending_enabled=bool(args.kappa_bend != 0.0),
        bend_definition=BEND_DEFINITION,
        bend_vals=bend_vals, bend_prob=bend_prob,
        mean_bends=mean_bends,
    )
    if args.save_raw_samples:
        save_kwargs["c_samples"] = C_arr
        save_kwargs["rg_samples"] = Rg_arr
        save_kwargs["bend_samples"] = B_arr

    np.savez_compressed(dist_file, **save_kwargs)

    # ---- diagnostics ----
    print("\n================ per-worker summary ================", flush=True)
    print(f"{'id':>3} {'seed':>6} {'attempted':>12} {'accepted':>12} "
          f"{'acc_ratio':>10} {'samples':>9} {'<m>':>8} {'<Rg>':>8}", flush=True)
    for r in results:
        wid = r["worker_id"]
        att = r["attempted_moves"]; ac = r["accepted_moves"]
        ns = r["contact_samples"].size
        print(f"{wid:>3} {r['seed']:>6} {att:>12} {ac:>12} "
              f"{ac/att:>10.4f} {ns:>9} "
              f"{per_worker_m_mean[wid]:>8.3f} {per_worker_rg_mean[wid]:>8.3f}",
              flush=True)

    print("\n================ combined statistics ================", flush=True)
    print(f"total attempted moves      : {total_attempted}", flush=True)
    print(f"total accepted moves       : {total_accepted}", flush=True)
    print(f"combined acceptance ratio  : {combined_acceptance:.4f}", flush=True)
    print(f"total samples              : {total_samples}", flush=True)
    print(f"m  : mean={C_arr.mean():.4f}  std={C_arr.std(ddof=0):.4f}", flush=True)
    print(f"Rg : mean={Rg_arr.mean():.4f}  std={Rg_arr.std(ddof=0):.4f}", flush=True)
    print(f"n_bend : mean={mean_bends:.4f}  fraction={mean_bends/max(1, args.N-2):.4f} "
          f"(kappa_bend={args.kappa_bend:g})", flush=True)
    print(f"contact range              : [{c_min}, {c_max}]", flush=True)
    print(f"Rg range                   : [{rg_min:.4f}, {rg_max:.4f}]", flush=True)
    print(f"c_prob sum                 : {float(c_prob.sum()):.10g}", flush=True)
    print(f"rg_prob sum                : {float(rg_prob.sum()):.10g}", flush=True)
    print(f"crg_prob sum               : {float(crg_prob.sum()):.10g}", flush=True)
    print(f"joint->P(m) marginal error : {marg_m_err:.3e}", flush=True)
    print(f"joint->P(Rg) marginal error: {marg_rg_err:.3e}", flush=True)
    print(f"total wall time            : {time.time()-t_start:.1f}s", flush=True)
    print(f"DIST_FILE = {dist_file}", flush=True)

if __name__ == "__main__":
    main()
