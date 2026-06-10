#!/usr/bin/env python3
"""
HP lattice model (3D cubic) with self-avoiding polymer and HH contact energy.

Goal (for now):
- sample configurations and find low-energy ones (energy minimization via simulated annealing)
- visualize/save a few distinct best conformations

Energy convention:
E = -eps * (# of unique non-bonded nearest-neighbour H-H contacts)

Notes:
- If your sequence has only ONE H (e.g., N=12 and H_period=12 -> only index 0 is H),
  then HH contacts are impossible, so every conformation has E=0 and "minimization" is trivial.
  For nontrivial folding, use >=2 Hs (e.g., --seq HPPPPPPPPPPH or --H_period 6).
"""

from __future__ import annotations
import argparse
import random
import math
from itertools import permutations, product
from typing import Tuple, List, Set, Dict

import numpy as np
import matplotlib.pyplot as plt

Vec = Tuple[int, int, int]
NN_VECS: List[Vec] = [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]

def add(a: Vec, b: Vec) -> Vec:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])

def sub(a: Vec, b: Vec) -> Vec:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])

# ------------------------- 24 proper cubic rotations -------------------------
def _det3(M: Tuple[Tuple[int,int,int], Tuple[int,int,int], Tuple[int,int,int]]) -> int:
    (a,b,c), (d,e,f), (g,h,i) = M
    return a*(e*i - f*h) - b*(d*i - f*g) + c*(d*h - e*g)

def generate_cubic_rotations() -> List[Tuple[Tuple[int,int,int], Tuple[int,int,int], Tuple[int,int,int]]]:
    rots = []
    for perm in permutations([0, 1, 2]):
        for signs in product([-1, 1], repeat=3):
            M = [[0,0,0],[0,0,0],[0,0,0]]
            for r, c in enumerate(perm):
                M[r][c] = signs[r]
            Mt = (tuple(M[0]), tuple(M[1]), tuple(M[2]))
            if _det3(Mt) == 1:
                rots.append(Mt)
    return rots  # 24

ROT_MATS = generate_cubic_rotations()

def apply_rot(M, v: Vec) -> Vec:
    x, y, z = v
    return (
        M[0][0]*x + M[0][1]*y + M[0][2]*z,
        M[1][0]*x + M[1][1]*y + M[1][2]*z,
        M[2][0]*x + M[2][1]*y + M[2][2]*z,
    )

# ------------------------- HP sequence helpers -------------------------
def make_sequence(N: int, seq: str | None, H_period: int | None) -> str:
    if seq is not None:
        s = seq.strip().upper()
        if any(ch not in ("H","P") for ch in s):
            raise ValueError("Sequence must contain only H and P")
        if len(s) != N:
            raise ValueError(f"--seq length {len(s)} does not match N={N}")
        return s

    if H_period is None:
        H_period = 12

    s = ["P"] * N
    for i in range(0, N, H_period):
        s[i] = "H"
    return "".join(s)

# ------------------------- energy + observables -------------------------
def hh_contacts(chain: List[Vec], occ: Set[Vec], seq: str) -> int:
    """Unique non-bonded nearest-neighbour HH contacts."""
    N = len(chain)
    m = 0
    pos_to_idx: Dict[Vec, int] = {pos: i for i, pos in enumerate(chain)}

    for i, r in enumerate(chain):
        if seq[i] != "H":
            continue
        prev = chain[i-1] if i > 0 else None
        nxt  = chain[i+1] if i < N-1 else None

        for v in NN_VECS:
            nbr = add(r, v)
            if nbr in occ and nbr not in (prev, nxt):
                j = pos_to_idx.get(nbr, None)
                if j is None:
                    continue
                if seq[j] == "H":
                    m += 1
    return m // 2  # counted from both ends

def energy(chain: List[Vec], occ: Set[Vec], seq: str, eps: float) -> float:
    return -eps * float(hh_contacts(chain, occ, seq))

def total_contacts(chain: List[Vec], occ: Set[Vec]) -> int:
    """Unique non-bonded NN contacts, regardless of bead type."""
    m = 0
    N = len(chain)
    for i, r in enumerate(chain):
        prev = chain[i-1] if i > 0   else None
        nxt  = chain[i+1] if i < N-1 else None
        for v in NN_VECS:
            nbr = add(r, v)
            if nbr in occ and nbr not in (prev, nxt):
                m += 1
    return m // 2

# ------------------------- moves -------------------------
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
    n = len(chain)
    i = random.randrange(1, n-1)
    a, b, c = chain[i-1], chain[i], chain[i+1]
    u1 = sub(b, a)
    u2 = sub(c, b)

    if u1 not in NN_VECS or u2 not in NN_VECS:
        return False, chain, occ

    if u1 == u2 or u1 == (-u2[0], -u2[1], -u2[2]):
        return False, chain, occ

    # 90-degree kink
    if (u1[0]*u2[0] + u1[1]*u2[1] + u1[2]*u2[2]) != 0:
        return False, chain, occ

    b_new = add(a, u2)
    if b_new in occ:
        return False, chain, occ

    if sub(b_new, a) not in NN_VECS or sub(c, b_new) not in NN_VECS:
        return False, chain, occ

    new_chain = chain.copy()
    new_chain[i] = b_new
    new_occ = (occ - {b}) | {b_new}
    return True, new_chain, new_occ

def attempt_end_move(chain: List[Vec], occ: Set[Vec]):
    """
    Symmetric end move: choose end, choose a random lattice direction,
    accept if (i) empty and (ii) maintains unit bond to anchor.
    """
    n = len(chain)
    end = 0 if random.random() < 0.5 else n - 1
    anchor = 1 if end == 0 else n - 2

    v = random.choice(NN_VECS)
    r_new = add(chain[end], v)
    if r_new in occ:
        return False, chain, occ
    if sub(r_new, chain[anchor]) not in NN_VECS:
        return False, chain, occ

    new_chain = chain.copy()
    old = chain[end]
    new_chain[end] = r_new
    new_occ = (occ - {old}) | {r_new}
    return True, new_chain, new_occ

MOVE_FUNCS = [attempt_pivot, attempt_crankshaft, attempt_end_move]

# ------------------------- bookkeeping for "distinct best" -------------------------
def canonical_key(chain: List[Vec]) -> Tuple[Vec, ...]:
    """
    Translation-invariant key (no rotational canonicalization, just shift).
    Good enough for keeping a handful of distinct minima from repeated anneals.
    """
    xs = [p[0] for p in chain]; ys = [p[1] for p in chain]; zs = [p[2] for p in chain]
    shift = (min(xs), min(ys), min(zs))
    shifted = [(p[0]-shift[0], p[1]-shift[1], p[2]-shift[2]) for p in chain]
    return tuple(shifted)

def keep_best(pool: List[Tuple[float, Tuple[Vec, ...]]], E: float, key: Tuple[Vec, ...], k: int) -> None:
    # If already present, keep the lower energy version (should be same E anyway)
    for idx, (Ei, ki) in enumerate(pool):
        if ki == key:
            if E < Ei:
                pool[idx] = (E, key)
            return
    pool.append((E, key))
    pool.sort(key=lambda x: x[0])
    del pool[k:]

# ------------------------- visualization -------------------------
def plot_chain(chain: List[Vec], seq: str, title: str = "", save: str | None = None, show: bool = True) -> None:
    r = np.asarray(chain, dtype=float)
    hp = np.array([1 if ch == "H" else 0 for ch in seq], dtype=float)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")

    # line
    ax.plot(r[:,0], r[:,1], r[:,2], marker="o", linewidth=1)

    # overlay beads with a colormap (H=1, P=0)
    sc = ax.scatter(r[:,0], r[:,1], r[:,2], c=hp, cmap="viridis", s=60)
    cb = fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.1)
    cb.set_ticks([0, 1])
    cb.set_ticklabels(["P", "H"])

    mins = r.min(axis=0) - 1
    maxs = r.max(axis=0) + 1
    ax.set_xlim(mins[0], maxs[0])
    ax.set_ylim(mins[1], maxs[1])
    ax.set_zlim(mins[2], maxs[2])
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_title(title)

    fig.tight_layout()
    if save is not None:
        fig.savefig(save, dpi=200)
    if show:
        plt.show()
    plt.close(fig)

# ------------------------- MC driver -------------------------
def metropolis_accept(dE: float, beta: float) -> bool:
    if dE <= 0:
        return True
    return random.random() < math.exp(-beta * dE)

def anneal_one_run(N: int, seq: str, eps: float, T_hi: float, T_lo: float,
                   n_temps: int, steps_per_T: int) -> Tuple[List[Vec], Set[Vec], float]:
    # start straight
    chain: List[Vec] = [(i, 0, 0) for i in range(N)]
    occ: Set[Vec] = set(chain)
    E = energy(chain, occ, seq, eps)

    # geometric schedule from T_hi down to T_lo
    if n_temps <= 1:
        temps = [T_lo]
    else:
        ratio = (T_lo / T_hi) ** (1.0 / (n_temps - 1))
        temps = [T_hi * (ratio ** i) for i in range(n_temps)]

    for T in temps:
        beta = 1.0 / T
        for _ in range(steps_per_T):
            move = random.choice(MOVE_FUNCS)
            ok, chain_new, occ_new = move(chain, occ)
            if not ok:
                continue
            E_new = energy(chain_new, occ_new, seq, eps)
            dE = E_new - E
            if metropolis_accept(dE, beta):
                chain, occ, E = chain_new, occ_new, E_new

    return chain, occ, E

def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--N", type=int, default=36, help="chain length")
    ap.add_argument("--seq", type=str, default=None, help="explicit HP sequence (length N), e.g. HPPHPPHPPHPP")
    ap.add_argument("--H_period", type=int, default=2, help="place an H every H_period beads (0, H_period, 2*H_period, ...)")
    ap.add_argument("--eps", type=float, default=1.0, help="HH contact strength")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed")

    # annealing controls
    ap.add_argument("--restarts", type=int, default=50, help="independent annealing runs")
    ap.add_argument("--T_hi", type=float, default=5.0, help="starting temperature")
    ap.add_argument("--T_lo", type=float, default=0.2, help="ending temperature")
    ap.add_argument("--n_temps", type=int, default=25, help="number of temperatures in schedule")
    ap.add_argument("--steps_per_T", type=int, default=500, help="MC attempts per temperature")

    # output / viz
    ap.add_argument("--n_best", type=int, default=8, help="keep this many distinct best conformations")
    ap.add_argument("--save_dir", type=str, default=None, help="if set, save PNGs of best structures here")
    ap.add_argument("--show", action="store_true", default=True, help="display the best structure at the end")

    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    seq = make_sequence(args.N, args.seq, args.H_period)
    nH = seq.count("H")

    print(f"Sequence (N={args.N}, H={nH}): {seq}")
    if nH < 2:
        print("Warning: sequence has <2 H beads, so HH contacts cannot form; all energies will be 0.")

    best_pool: List[Tuple[float, Tuple[Vec, ...]]] = []
    best_chain: List[Vec] | None = None
    best_occ: Set[Vec] | None = None
    best_E = float("inf")

    for r in range(args.restarts):
        chain, occ, E = anneal_one_run(
            N=args.N, seq=seq, eps=args.eps,
            T_hi=args.T_hi, T_lo=args.T_lo,
            n_temps=args.n_temps, steps_per_T=args.steps_per_T
        )
        key = canonical_key(chain)
        keep_best(best_pool, E, key, args.n_best)

        if E < best_E:
            best_E = E
            best_chain, best_occ = chain, occ

        if (r + 1) % max(1, args.restarts // 10) == 0:
            print(f"restart {r+1}/{args.restarts}: current best E = {best_E:.1f}")

    print("\nBest distinct conformations found (energy, HH_contacts, total_contacts):")
    for idx, (E, key) in enumerate(best_pool, start=1):
        chain = list(key)
        occ = set(chain)
        hh = hh_contacts(chain, occ, seq)
        tc = total_contacts(chain, occ)
        print(f"  {idx:2d}) E={E:6.1f}   HH={hh:2d}   C_total={tc:2d}")

    if best_chain is None or best_occ is None:
        raise RuntimeError("No chain produced; this should not happen.")

    # save best structures
    if args.save_dir is not None:
        import os
        os.makedirs(args.save_dir, exist_ok=True)
        for idx, (E, key) in enumerate(best_pool, start=1):
            chain = list(key)
            occ = set(chain)
            hh = hh_contacts(chain, occ, seq)
            out = os.path.join(args.save_dir, f"best_{idx:02d}_E{int(E)}_HH{hh}.png")
            plot_chain(chain, seq, title=f"best {idx}: E={E:.0f}, HH={hh}", save=out, show=False)
        print(f"Saved {len(best_pool)} PNGs to {args.save_dir}")

    if args.show:
        hh = hh_contacts(best_chain, best_occ, seq)
        plot_chain(best_chain, seq, title=f"Best: E={best_E:.0f}, HH={hh}", save=None, show=True)

if __name__ == "__main__":
    main()
