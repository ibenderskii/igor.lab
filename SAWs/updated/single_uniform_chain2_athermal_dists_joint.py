#!/usr/bin/env python3
"""
Athermal (non-interacting) self-avoiding walk (SAW) on a 3D cubic lattice.

- Uses pivot / crankshaft / end moves.
- Accepts every geometrically valid move (no energetics).
- Measures the *non-bonded* nearest-neighbour contact count m and R_g.
- Outputs equilibrium distributions:
    * P(m) as (c_vals, c_prob) over integer m
    * P(Rg) as (rg_edges, rg_prob)
    * NEW: joint P(m, Rg) as a 2D histogram (c_edges, rg_edges, crg_prob)

The joint distribution is useful as an athermal baseline to predict
P(Rg | T) for thermodynamic models whose energy depends only on m via
reweighting by exp[-(h/T - s) m].
"""

from __future__ import annotations
import argparse
import random
import math
from pathlib import Path
from itertools import permutations, product
from typing import Tuple, List, Set

import numpy as np

Vec = Tuple[int, int, int]
NN_VECS: List[Vec] = [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]

def add(a: Vec, b: Vec) -> Vec:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])

def sub(a: Vec, b: Vec) -> Vec:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])

# ------------------------- rotations (24 proper cubic rotations) -------------------------
def _det3(M: Tuple[Tuple[int,int,int], Tuple[int,int,int], Tuple[int,int,int]]) -> int:
    (a,b,c), (d,e,f), (g,h,i) = M
    return a*(e*i - f*h) - b*(d*i - f*g) + c*(d*h - e*g)

def generate_cubic_rotations() -> List[Tuple[Tuple[int,int,int], Tuple[int,int,int], Tuple[int,int,int]]]:
    """All 24 orientation-preserving signed-permutation 3x3 matrices."""
    rots = []
    for perm in permutations([0, 1, 2]):
        for signs in product([-1, 1], repeat=3):
            M = [[0,0,0],[0,0,0],[0,0,0]]
            for r, c in enumerate(perm):
                M[r][c] = signs[r]
            Mt = (tuple(M[0]), tuple(M[1]), tuple(M[2]))
            if _det3(Mt) == 1:
                rots.append(Mt)
    return rots  # 24 rotations total

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

# ------------------------- moves (athermal; accept if geometrically valid) -------------------------
def attempt_pivot(chain: List[Vec], occ: Set[Vec]):
    n = len(chain)
    i = random.randrange(1, n-1)  # pivot monomer (not ends)

    head = chain[:i+1]
    tail = chain[i+1:]
    pivot = chain[i]

    M = random.choice(ROT_MATS)

    new_tail: List[Vec] = []
    new_occ: Set[Vec] = set(head)

    for r in tail:
        dr = sub(r, pivot)
        r2 = add(pivot, apply_rot(M, dr))
        if r2 in new_occ:
            return False, chain, occ
        new_tail.append(r2)
        new_occ.add(r2)

    return True, head + new_tail, new_occ

def attempt_crankshaft(chain: List[Vec], occ: Set[Vec]):
    """Local kink flip (crankshaft). Works only for perfect 90° kinks."""
    n = len(chain)
    i = random.randrange(1, n-1)
    a, b, c = chain[i-1], chain[i], chain[i+1]

    u1 = sub(b, a)
    u2 = sub(c, b)

    if u1 not in NN_VECS or u2 not in NN_VECS:
        return False, chain, occ

    # reject straight segments (parallel or anti-parallel)
    if u1 == u2 or u1 == (-u2[0], -u2[1], -u2[2]):
        return False, chain, occ

    # require perfect 90° kink
    if (u1[0]*u2[0] + u1[1]*u2[1] + u1[2]*u2[2]) != 0:
        return False, chain, occ

    b_new = add(a, u2)
    if b_new in occ:
        return False, chain, occ

    # bonds must remain unit length
    if sub(b_new, a) not in NN_VECS or sub(c, b_new) not in NN_VECS:
        return False, chain, occ

    new_chain = chain.copy()
    new_chain[i] = b_new
    new_occ = (occ - {b}) | {b_new}
    return True, new_chain, new_occ

def attempt_end_move(chain: List[Vec], occ: Set[Vec]):
    """Symmetric end move: move one end to a random empty neighbor of its anchor."""
    n = len(chain)
    end = 0 if random.random() < 0.5 else n - 1
    anchor = 1 if end == 0 else n - 2

    old = chain[end]
    occ_without_old = occ - {old}

    v = random.choice(NN_VECS)
    r_new = add(chain[anchor], v)

    if r_new == old:
        return False, chain, occ
    if r_new in occ_without_old:
        return False, chain, occ

    new_chain = chain.copy()
    new_chain[end] = r_new
    new_occ = occ_without_old | {r_new}
    return True, new_chain, new_occ


def sanity_check_end_move() -> None:
    """Verify that the end move can accept from a simple straight chain."""
    chain = [(0,0,0), (1,0,0), (2,0,0), (3,0,0), (4,0,0)]
    occ = set(chain)
    n_accept = sum(attempt_end_move(chain, occ)[0] for _ in range(200))
    if n_accept <= 0:
        raise RuntimeError("attempt_end_move never accepted in 200 trials")

MOVE_FUNCS = [attempt_pivot, attempt_crankshaft, attempt_end_move]

# ------------------------- main -------------------------
def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--N", type=int, default=44, help="chain length")
    ap.add_argument("--steps", type=int, default=10_000_000, help="total MC attempted moves")
    ap.add_argument("--T", type=float, default=1.0, help="temperature (ignored; kept for temp_scan compatibility)")
    ap.add_argument("--eps", type=float, default=0.0, help="contact energy (ignored; kept for temp_scan compatibility)")
    ap.add_argument("--dist_dir", type=str, default="dists", help="where to write equilibrium distribution .npz")
    ap.add_argument("--rg_bins", type=int, default=60, help="number of histogram bins for P(Rg)")
    ap.add_argument("--burnin", type=float, default=0.3, help="burn-in fraction of steps (0..1)")
    ap.add_argument("--sample_every", type=int, default=1000, help="sample every this many steps after burn-in")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed")
    ap.add_argument("--no_joint", action="store_true", help="skip writing the joint P(m,Rg) histogram")
    args = ap.parse_args()

    if args.N < 3:
        raise ValueError("--N must be >= 3")
    if args.steps < 1:
        raise ValueError("--steps must be >= 1")
    if args.rg_bins < 1:
        raise ValueError("--rg_bins must be >= 1")
    if not (0.0 <= args.burnin < 1.0):
        raise ValueError("--burnin must be in [0, 1)")
    if args.sample_every < 1:
        raise ValueError("--sample_every must be >= 1")

    random.seed(args.seed)
    np.random.seed(args.seed)

    sanity_check_end_move()

    # initial straight chain
    chain: List[Vec] = [(i, 0, 0) for i in range(args.N)]
    occ: Set[Vec] = set(chain)

    burn_steps = int(round(args.burnin * args.steps))
    burn_steps = max(0, min(burn_steps, args.steps))

    acc = 0

    # store samples for equilibrium distributions (post burn-in)
    C_samples: List[int] = []
    Rg_samples: List[float] = []

    for step in range(1, args.steps + 1):
        move = random.choice(MOVE_FUNCS)
        ok, chain_new, occ_new = move(chain, occ)
        if ok:
            chain, occ = chain_new, occ_new
            acc += 1

        if step > burn_steps and (step - burn_steps) % args.sample_every == 0:
            C_samples.append(int(contact_count(chain, occ)))
            Rg_samples.append(float(radius_of_gyration(chain)))

    print(f"Acceptance ratio: {acc / args.steps:.3f}")
    print(f"Samples collected: {len(C_samples)} (after burn-in)")

    # Scalars in the same format as the interacting scripts (temp_scan regexes)
    # (E is identically zero in the athermal model)
    E_mean = 0.0
    E_std = 0.0

    C_arr = np.asarray(C_samples, dtype=int)
    Rg_arr = np.asarray(Rg_samples, dtype=float)

    C_mean = float(np.nanmean(C_arr)) if C_arr.size else float("nan")
    C_std  = float(np.nanstd(C_arr.astype(float), ddof=0)) if C_arr.size else float("nan")
    Rg_mean = float(np.nanmean(Rg_arr)) if Rg_arr.size else float("nan")
    Rg_std  = float(np.nanstd(Rg_arr, ddof=0)) if Rg_arr.size else float("nan")

    print(f"E_mean = {E_mean:.3f} ± {E_std:.3f}")
    print(f"Rg_back_mean = {Rg_mean:.3f} ± {Rg_std:.3f}")
    print(f"C_mean = {C_mean:.3f} ± {C_std:.3f}")

    # --- equilibrium distributions for temp_scan ---
    dist_dir = Path(args.dist_dir)
    dist_dir.mkdir(parents=True, exist_ok=True)

    # P(m): exact over integer contact count
    if C_arr.size == 0:
        c_vals = np.array([0], dtype=int)
        c_prob = np.array([1.0], dtype=float)
    else:
        c_vals, c_counts = np.unique(C_arr, return_counts=True)
        c_prob = c_counts.astype(float) / c_counts.sum()

    # P(Rg): histogram
    rg = Rg_arr[np.isfinite(Rg_arr)]
    if rg.size == 0:
        rg_edges = np.linspace(0.0, 1.0, int(args.rg_bins) + 1)
        rg_prob = np.zeros(int(args.rg_bins), dtype=float)
        rg_prob[0] = 1.0
    else:
        rg_min = float(rg.min()); rg_max = float(rg.max())
        pad = 1e-9 if rg_max <= rg_min else 0.02*(rg_max - rg_min)
        rg_edges = np.linspace(rg_min - pad, rg_max + pad, int(args.rg_bins) + 1)
        rg_counts, _ = np.histogram(rg, bins=rg_edges)
        rg_prob = rg_counts.astype(float)
        sprob = rg_prob.sum()
        if sprob > 0:
            rg_prob /= sprob

    # NEW: joint P(m, Rg) as a 2D histogram (m bins are integer-centered)
    if (not args.no_joint) and (C_arr.size > 0) and (rg.size > 0):
        c_min = int(C_arr.min())
        c_max = int(C_arr.max())
        # integer bins centered on ...,-0.5,0.5,1.5,...
        c_edges = np.arange(c_min - 0.5, c_max + 1.5, 1.0, dtype=float)
        # note histogram2d expects x then y; we'll treat x=m, y=Rg
        crg_counts, _, _ = np.histogram2d(C_arr.astype(float), rg, bins=[c_edges, rg_edges])
        crg_prob = crg_counts / crg_counts.sum() if crg_counts.sum() > 0 else crg_counts
    else:
        c_edges = np.array([], dtype=float)
        crg_prob = np.array([[]], dtype=float)

    seed_tag = args.seed if args.seed is not None else "na"
    dist_file = dist_dir / f"{Path(__file__).stem}_N{args.N}_T{args.T:.6g}_seed{seed_tag}.npz"
    np.savez_compressed(
        dist_file,
        # existing fields (temp_scan-compatible)
        c_vals=c_vals, c_prob=c_prob,
        rg_edges=rg_edges, rg_prob=rg_prob,
        # new joint fields
        c_edges=c_edges, crg_prob=crg_prob,
        # metadata
        T=float(args.T), N=int(args.N), steps=int(args.steps),
        seed=seed_tag, eps=float(getattr(args, "eps", 0.0)),
        burnin=float(args.burnin), sample_every=int(args.sample_every),
        n_samples=int(C_arr.size),
    )
    print(f"DIST_FILE = {dist_file}")
    print(f"c_prob sum = {float(c_prob.sum()):.6g}")
    print(f"rg_prob sum = {float(rg_prob.sum()):.6g}")
    if crg_prob.size > 0:
        print(f"crg_prob shape = {crg_prob.shape}")
        print(f"crg_prob sum = {float(crg_prob.sum()):.6g}")

if __name__ == "__main__":
    main()
