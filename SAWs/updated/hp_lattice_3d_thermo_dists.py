#!/usr/bin/env python3
"""
HP lattice model (3D cubic) with a self-avoiding polymer and HH-only contacts.

This version matches the output contract of thermo_uniform_chain2_DBfix_dists_corrected.py:
- Metropolis MC at fixed temperature T
- reports E_mean, Rg_back_mean, C_mean (here C = HH contacts), and C_var
- writes equilibrium distributions P(C) and P(Rg) to a .npz and prints:
      DIST_FILE = <path>

Hamiltonian (effective free energy):
    H(C; T) = N_HH(C) * (dh - T * ds)
where N_HH is the number of unique non-bonded nearest-neighbour H-H contacts.

Notes:
- If your sequence has <2 H beads, HH contacts cannot form and C will remain 0.
"""

from __future__ import annotations

import argparse
import random
import math
from pathlib import Path
from itertools import permutations, product
from typing import Tuple, List, Set, Dict

import numpy as np

Vec = Tuple[int, int, int]
NN_VECS: List[Vec] = [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]

def add(a: Vec, b: Vec) -> Vec:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])

def sub(a: Vec, b: Vec) -> Vec:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])

# ------------------------- 24 proper cubic rotations (exclude identity) -------------------------
def _det3(M: Tuple[Tuple[int,int,int], Tuple[int,int,int], Tuple[int,int,int]]) -> int:
    (a,b,c), (d,e,f), (g,h,i) = M
    return a*(e*i - f*h) - b*(d*i - f*g) + c*(d*h - e*g)

def generate_cubic_rotations_no_identity() -> List[Tuple[Tuple[int,int,int], Tuple[int,int,int], Tuple[int,int,int]]]:
    rots: List[Tuple[Tuple[int,int,int], Tuple[int,int,int], Tuple[int,int,int]]] = []
    for perm in permutations([0, 1, 2]):
        for signs in product([-1, 1], repeat=3):
            M = [[0,0,0],[0,0,0],[0,0,0]]
            for r, c in enumerate(perm):
                M[r][c] = signs[r]
            Mt = (tuple(M[0]), tuple(M[1]), tuple(M[2]))
            if _det3(Mt) == 1:
                rots.append(Mt)
    I = ((1,0,0),(0,1,0),(0,0,1))
    return [M for M in rots if M != I]

ROT_MATS = generate_cubic_rotations_no_identity()

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

# ------------------------- observables -------------------------
def radius_of_gyration(chain: List[Vec]) -> float:
    r = np.asarray(chain, dtype=float)
    com = r.mean(axis=0)
    return float(np.sqrt(((r - com) ** 2).sum(axis=1).mean()))

def hh_contacts(chain: List[Vec], occ: Set[Vec], seq: str) -> int:
    """Unique non-bonded nearest-neighbour HH contacts."""
    # map positions -> monomer index so we can check bead types
    pos_to_idx: Dict[Vec, int] = {pos: i for i, pos in enumerate(chain)}
    m = 0
    N = len(chain)

    for i, r in enumerate(chain):
        if seq[i] != "H":
            continue
        prev = chain[i-1] if i > 0 else None
        nxt  = chain[i+1] if i < N-1 else None

        for v in NN_VECS:
            nbr = add(r, v)
            if nbr in occ and nbr not in (prev, nxt):
                j = pos_to_idx.get(nbr)
                if j is not None and seq[j] == "H":
                    m += 1

    return m // 2  # counted from both ends

def energy(chain: List[Vec], occ: Set[Vec], seq: str, dh: float, ds: float, T: float) -> float:
    """Effective free-energy Hamiltonian: H = N_HH * (dh - T ds)."""
    n_hh = float(hh_contacts(chain, occ, seq))
    return n_hh * (dh - T * ds)

# ------------------------- moves (same move set as your SAW scripts) -------------------------
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
    Symmetric end move (no Hastings correction needed):
    - choose an end with prob 1/2
    - choose a lattice direction with prob 1/6
    - accept if target site empty AND maintains unit bond to anchor
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

def log_schedule(total_steps: int, n_frames: int) -> np.ndarray:
    xs = np.geomspace(1, total_steps, n_frames).astype(int)
    return np.unique(xs)

def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # polymer + thermodynamics
    ap.add_argument("--N", type=int, default=200, help="chain length")
    ap.add_argument("--seq", type=str, default=None, help="explicit HP sequence (length N), e.g. HPPHPPHPPHPP")
    ap.add_argument("--H_period", type=int, default=3, help="place an H every H_period beads (0, H_period, 2*H_period, ...)")
    ap.add_argument("--dh", type=float, default=2.0, help="per-HH-contact enthalpy term Δh")
    ap.add_argument("--ds", type=float, default=1.0, help="per-HH-contact entropy term Δs")
    ap.add_argument("--T", type=float, default=3, help="temperature (k_B=1)")
    ap.add_argument("--steps", type=int, default=500000, help="Monte-Carlo steps")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed")

    # distribution output (for temp_scan)
    ap.add_argument("--dist_dir", type=str, default="dists", help="where to write equilibrium distribution .npz")
    ap.add_argument("--rg_bins", type=int, default=80, help="number of histogram bins for P(Rg)")

    # bookkeeping / sampling cadence
    ap.add_argument("--record_points", type=int, default=2000, help="number of points to record for time series / equilibrium stats")
    args = ap.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    seq = make_sequence(args.N, args.seq, args.H_period)
    nH = seq.count("H")
    if nH < 2:
        print("Warning: sequence has <2 H beads, so HH contacts cannot form; C will stay ~0.")

    # initial straight chain
    chain: List[Vec] = [(i, 0, 0) for i in range(args.N)]
    occ: Set[Vec] = set(chain)

    E = energy(chain, occ, seq, args.dh, args.ds, args.T)
    beta = 1.0 / args.T

    acc = 0
    record_interval = max(1, args.steps // int(args.record_points))
    saved_steps: List[int] = []
    E_traj: List[float] = []
    Rg_traj: List[float] = []
    C_traj: List[float] = []  # HH contacts

    # (optional) keep a few log-spaced snapshots if you later want to animate/debug
    # frames_to_save = set(log_schedule(args.steps, 2000))
    # frames: List[np.ndarray] = []

    for step in range(1, args.steps + 1):
        if step % record_interval == 0:
            saved_steps.append(step)
            E_traj.append(E)
            Rg_traj.append(radius_of_gyration(chain))
            C_traj.append(float(hh_contacts(chain, occ, seq)))

        move = random.choice(MOVE_FUNCS)
        ok, chain_new, occ_new = move(chain, occ)
        if not ok:
            continue

        E_new = energy(chain_new, occ_new, seq, args.dh, args.ds, args.T)
        dE = E_new - E
        if dE <= 0.0 or random.random() < math.exp(-beta * dE):
            chain, occ, E = chain_new, occ_new, E_new
            acc += 1

        # if step in frames_to_save:
        #     frames.append(np.asarray(chain, dtype=np.float64))

    print(f"Acceptance ratio: {acc/args.steps:.3f}")

    # ---------- equilibrium averages over last 30% of recorded points ----------
    if len(saved_steps) > 0:
        n = len(saved_steps)
        start_idx = int(math.floor(n * 0.7))

        E_slice = np.asarray(E_traj[start_idx:], dtype=float)
        Rg_slice = np.asarray(Rg_traj[start_idx:], dtype=float)
        C_slice = np.asarray(C_traj[start_idx:], dtype=float)

        E_mean = float(np.nanmean(E_slice))
        E_std  = float(np.nanstd(E_slice, ddof=0))

        Rg_mean = float(np.nanmean(Rg_slice))
        Rg_std  = float(np.nanstd(Rg_slice, ddof=0))

        C_mean = float(np.nanmean(C_slice))
        C_std  = float(np.nanstd(C_slice, ddof=0))
        C_var_pop = float(np.nanvar(C_slice, ddof=0))
        C_var_samp = float(np.nanvar(C_slice, ddof=1)) if C_slice.size > 1 else float("nan")

        # ---------- write equilibrium distributions ----------
        try:
            dist_dir = Path(args.dist_dir)
            dist_dir.mkdir(parents=True, exist_ok=True)

            # Contacts distribution P(C) where C is HH contacts (integer)
            c_int = np.rint(C_slice).astype(int)
            c_vals, c_counts = np.unique(c_int, return_counts=True)
            c_prob = c_counts.astype(float) / max(1, c_counts.sum())

            # Rg distribution P(Rg)
            rg = Rg_slice[np.isfinite(Rg_slice)]
            if rg.size == 0:
                rg_edges = np.linspace(0.0, 1.0, int(args.rg_bins) + 1)
                rg_prob = np.zeros(int(args.rg_bins), dtype=float)
            else:
                rg_min = float(rg.min())
                rg_max = float(rg.max())
                pad = 1e-9 if rg_max <= rg_min else 0.02 * (rg_max - rg_min)
                rg_edges = np.linspace(rg_min - pad, rg_max + pad, int(args.rg_bins) + 1)
                rg_counts, _ = np.histogram(rg, bins=rg_edges)
                rg_prob = rg_counts.astype(float)
                sprob = rg_prob.sum()
                if sprob > 0:
                    rg_prob /= sprob

            seed_tag = args.seed if args.seed is not None else "na"
            dist_file = dist_dir / f"{Path(__file__).stem}_N{args.N}_T{args.T:.6g}_seed{seed_tag}.npz"
            np.savez_compressed(
                dist_file,
                c_vals=c_vals,
                c_prob=c_prob,
                rg_edges=rg_edges,
                rg_prob=rg_prob,
                T=float(args.T),
                N=int(args.N),
                steps=int(args.steps),
                seed=seed_tag,
                dh=float(args.dh),
                ds=float(args.ds),
                seq=str(seq),
                n_samples=int(rg.size),
            )
            print(f"DIST_FILE = {dist_file}")
        except Exception:
            # temp_scan will still work for scalar observables
            pass

        # instrumented scalar output expected by temp_scan2_updated.py
        print(f"E_mean = {E_mean:.3f} ± {E_std:.3f}")
        print(f"Rg_back_mean = {Rg_mean:.3f} ± {Rg_std:.3f}")
        print(f"C_mean = {C_mean:.3f} ± {C_std:.3f}")
        print(f"C_var(pop) = {C_var_pop:.3f}")
        print(f"C_var(sample) = {C_var_samp:.3f}")
    else:
        # fallback (should be rare if steps is large)
        C0 = float(hh_contacts(chain, occ, seq))
        Rg0 = radius_of_gyration(chain)
        print(f"E_mean = {E:.3f} ± {0.000:.3f}")
        print(f"Rg_back_mean = {Rg0:.3f} ± {0.000:.3f}")
        print(f"C_mean = {C0:.3f} ± {0.000:.3f}")
        print(f"C_var(pop) = {0.000:.3f}")
        print(f"C_var(sample) = nan")

if __name__ == "__main__":
    main()
