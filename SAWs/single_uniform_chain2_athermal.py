#!/usr/bin/env python3
"""
Athermal (non-interacting) self-avoiding walk (SAW) on a 3D cubic lattice.

- Uses pivot / crankshaft / end moves.
- Accepts every geometrically valid move (no energetics).
- Measures the *non-bonded* nearest-neighbour contact count m
  and reports mean + variance after an initial burn-in.

This is a simplified variant of your attractive-contact Metropolis code:
we keep the same contact definition, but contacts do NOT affect acceptance.
"""

from __future__ import annotations
import argparse
import random
import math
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
    # 24 rotations total
    return rots

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

# ------------------------- moves (all athermal, accept if geometrically valid) -------------------------
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
    """
    Local kink flip (crankshaft):
    Works only when the local triplet (a,b,c) forms a 90° kink.
    """
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
    """
    Symmetric end move: pick a random end, then pick a random lattice direction.
    If it yields an empty site that preserves the end bond, accept; else reject.
    """
    n = len(chain)
    end = 0 if random.random() < 0.5 else n - 1
    anchor = 1 if end == 0 else n - 2

    v = random.choice(NN_VECS)
    r_new = add(chain[end], v)

    # must be empty
    if r_new in occ:
        return False, chain, occ

    # must maintain unit bond to anchor
    if sub(r_new, chain[anchor]) not in NN_VECS:
        return False, chain, occ

    new_chain = chain.copy()
    old = chain[end]
    new_chain[end] = r_new
    new_occ = (occ - {old}) | {r_new}
    return True, new_chain, new_occ

MOVE_FUNCS = [attempt_pivot, attempt_crankshaft, attempt_end_move]

# ------------------------- streaming mean/variance (Welford) -------------------------
class RunningStats:
    def __init__(self) -> None:
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0

    def update(self, x: float) -> None:
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2

    def variance_population(self) -> float:
        return self.M2 / self.n if self.n > 0 else float("nan")

    def variance_sample(self) -> float:
        return self.M2 / (self.n - 1) if self.n > 1 else float("nan")

# ------------------------- main -------------------------
def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--N", type=int, default=200, help="chain length")
    ap.add_argument("--steps", type=int, default=2000000, help="total MC attempted moves")
    ap.add_argument("--burnin", type=float, default=0.3, help="burn-in fraction of steps (0..1)")
    ap.add_argument("--sample_every", type=int, default=1000, help="sample every this many steps after burn-in")
    ap.add_argument("--seed", type=int, default=131, help="RNG seed")
    ap.add_argument("--track_rg", action="store_true", help="also measure Rg mean/var (slower)")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # initial straight chain
    chain: List[Vec] = [(i, 0, 0) for i in range(args.N)]
    occ: Set[Vec] = set(chain)

    burn_steps = int(round(args.burnin * args.steps))
    burn_steps = max(0, min(burn_steps, args.steps))

    acc = 0

    Cstats = RunningStats()
    Rgstats = RunningStats() if args.track_rg else None

    for step in range(1, args.steps + 1):
        move = random.choice(MOVE_FUNCS)
        ok, chain_new, occ_new = move(chain, occ)
        if ok:
            chain, occ = chain_new, occ_new
            acc += 1

        if step > burn_steps and (step - burn_steps) % args.sample_every == 0:
            c = contact_count(chain, occ)
            Cstats.update(float(c))
            if Rgstats is not None:
                Rgstats.update(radius_of_gyration(chain))

    print(f"Acceptance ratio (geometric): {acc / args.steps:.3f}")
    print(f"Samples collected: {Cstats.n} (after burn-in)")

    C_mean = Cstats.mean
    C_var_pop = Cstats.variance_population()
    C_var_samp = Cstats.variance_sample()
    print(f"C_mean = {C_mean:.6f}")
    print(f"C_var(pop) = {C_var_pop:.6f}")
    print(f"C_var(sample) = {C_var_samp:.6f}")
    print(f"C_std(pop) = {math.sqrt(C_var_pop):.6f}")

    if Rgstats is not None:
        print(f"Rg_mean = {Rgstats.mean:.6f}")
        print(f"Rg_var(pop) = {Rgstats.variance_population():.6f}")

if __name__ == "__main__":
    main()
