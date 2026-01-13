#!/usr/bin/env python3
"""
MC polymer collapse
3D lattice, nearest-neighbour contacts,
Metropolis with pivot / crank-shaft / end-flip moves.

PNIPAM-inspired LCST toy Hamiltonian:

    m       = # non-bonded nearest-neighbour contacts
    n_kink  = # 90° kinks along the backbone

    E(T) = -eps(T) * m
           + eta(T) * n_kink
           - 0.5 * J(T) * m**2

Temperature-dependent couplings via a smooth switch:

    s(T)   = 0.5 * [1 + tanh((T - Tc)/width)]   in [0,1]

LCST mapping (coil at low T, globule at high T):
  - Contacts turn on at high T:      eps(T) = eps_high * s(T)
  - Backbone stiffness at low T:     eta(T) = eta_low  * (1 - s(T))
  - Cooperativity turns on at high T: J(T)  = J_high  * s(T)

Low T:   eps ~ 0, eta ~ eta_low,  J ~ 0   -> stiff, extended chain
High T:  eps ~ eps_high, eta ~ 0,  J ~ J_high -> attractive + cooperative collapse
"""

from __future__ import annotations
import argparse, random, math
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

Vec = Tuple[int, int, int]

# unit lattice vectors
NN_VECS: List[Vec] = [
    (1,0,0), (-1,0,0),
    (0,1,0), (0,-1,0),
    (0,0,1), (0,0,-1)
]

def add(a:Vec, b:Vec) -> Vec:
    return (a[0]+b[0], a[1]+b[1], a[2]+b[2])

def sub(a:Vec, b:Vec) -> Vec:
    return (a[0]-b[0], a[1]-b[1], a[2]-b[2])

# 90° rotations for pivot moves
ROT_X = lambda v: ( v[0],  -v[2],  v[1])
ROT_Y = lambda v: ( v[2],   v[1], -v[0])
ROT_Z = lambda v: (-v[1],   v[0],  v[2])
ROTATIONS = [ ROT_X, ROT_Y, ROT_Z ]

# ----------------------------------------------------------------------
# Smooth LCST switch and couplings
# ----------------------------------------------------------------------

def lcst_switch(T: float, Tc: float, width: float) -> float:
    """s(T) in [0,1]: 0 at low T, 1 at high T."""
    if width <= 0:
        return 0.0 if T < Tc else 1.0
    x = (T - Tc) / width
    return 0.5 * (1.0 + math.tanh(x))

# ----------------------------------------------------------------------
# Energy with contact, kink, and cooperative terms
# ----------------------------------------------------------------------

def count_contacts_and_kinks(chain: List[Vec], occ: set[Vec]) -> tuple[int, int]:
    """Return (m, n_kink)."""
    N = len(chain)

    # count non-bonded NN contacts
    contact_pairs = 0
    for i, r in enumerate(chain):
        prev = chain[i-1] if i > 0   else None
        nxt  = chain[i+1] if i < N-1 else None
        for v in NN_VECS:
            nbr = add(r, v)
            if nbr in occ and nbr not in (prev, nxt):
                contact_pairs += 1
    m = contact_pairs // 2  # each contact counted twice

    # count kinks (orthogonal consecutive bonds)
    n_kink = 0
    for i in range(1, N-1):
        u1 = sub(chain[i],   chain[i-1])
        u2 = sub(chain[i+1], chain[i])
        if u1 not in NN_VECS or u2 not in NN_VECS:
            continue
        if (u1[0]*u2[0] + u1[1]*u2[1] + u1[2]*u2[2]) == 0:
            n_kink += 1

    return m, n_kink

def energy(chain: List[Vec],
           occ: set[Vec],
           eps_T: float,
           eta_T: float = 0.0,
           J_T: float   = 0.0) -> float:
    """
    E = -eps_T * m + eta_T * n_kink - 0.5 * J_T * m^2
    """
    m, n_kink = count_contacts_and_kinks(chain, occ)
    return (-eps_T * m) + (eta_T * n_kink) - (0.5 * J_T * (m*m))

# ----------------------------------------------------------------------

def radius_of_gyration(chain:List[Vec]) -> float:
    r = np.array(chain, dtype=float)
    com = r.mean(axis=0)
    return math.sqrt(((r - com)**2).sum(axis=1).mean())

# ----------------------------------------------------------------------
# MC moves
# ----------------------------------------------------------------------

def attempt_pivot(chain, occ) -> Tuple[bool, List[Vec], set[Vec]]:
    n = len(chain)
    if n <= 3:
        return False, chain, occ
    i = random.randrange(1, n-1)
    head = chain[:i+1]
    tail = chain[i+1:]

    rot = random.choice(ROTATIONS)
    pivot = chain[i]

    new_tail = []
    new_occ = set(head)
    for r in tail:
        dr = sub(r, pivot)
        r2 = add(pivot, rot(dr))
        if r2 in new_occ:
            return False, chain, occ
        new_tail.append(r2)
        new_occ.add(r2)

    return True, head + new_tail, new_occ

def attempt_crankshaft(chain, occ):
    n = len(chain)
    if n <= 3:
        return False, chain, occ
    i = random.randrange(1, n-1)
    a, b, c = chain[i-1], chain[i], chain[i+1]

    u1 = sub(b, a)
    u2 = sub(c, b)

    if u1 not in NN_VECS or u2 not in NN_VECS:
        return False, chain, occ
    if u1 == u2 or u1 == (-u2[0], -u2[1], -u2[2]):
        return False, chain, occ
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

def attempt_end_move(chain, occ) -> Tuple[bool, List[Vec], set[Vec]]:
    n = len(chain)
    if n <= 1:
        return False, chain, occ
    end = 0 if random.random() < 0.5 else n-1
    anchor = 1 if end == 0 else n-2

    candidates = []
    for v in NN_VECS:
        r_new = add(chain[end], v)
        if r_new in occ:
            continue
        if sub(r_new, chain[anchor]) in NN_VECS:
            candidates.append(r_new)

    if not candidates:
        return False, chain, occ

    r_new = random.choice(candidates)
    new_chain = chain.copy()
    old = chain[end]
    new_chain[end] = r_new
    new_occ = (occ - {old}) | {r_new}
    return True, new_chain, new_occ

MOVE_FUNCS = [attempt_pivot, attempt_crankshaft, attempt_end_move]

# ----------------------------------------------------------------------

def sample_trajectory(chain, snap_list):
    snap_list.append(np.asarray(chain, dtype=np.float64))

def render_movie(snaps, out="polymer.mp4", fps=30, stride=1):
    traj = np.asarray(snaps, dtype=float)[::stride]
    if traj.size == 0:
        return
    F = traj.shape[0]
    mins = traj.reshape(-1,3).min(0) - 2
    maxs = traj.reshape(-1,3).max(0) + 2

    fig = plt.figure(figsize=(5,5))
    ax  = fig.add_subplot(111, projection='3d')
    (line,) = ax.plot([], [], [], '-o', ms=3)
    ax.set_xlim(mins[0], maxs[0])
    ax.set_ylim(mins[1], maxs[1])
    ax.set_zlim(mins[2], maxs[2])  # fixed
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')

    def init():
        line.set_data([], [])
        line.set_3d_properties([])
        return (line,)

    def update(f):
        r = traj[f]
        line.set_data(r[:,0], r[:,1])
        line.set_3d_properties(r[:,2])
        ax.set_title(f"frame {f+1}/{F}")
        return (line,)

    ani = animation.FuncAnimation(fig, update, init_func=init, frames=F,
                                  interval=1000/fps, blit=False)
    try:
        ani.save(out, writer=animation.FFMpegWriter(fps=fps, bitrate=1800))
    except Exception:
        ani.save(out.replace(".mp4", ".gif"), writer="pillow", fps=fps)
    plt.close(fig)

def log_schedule(total_steps, n_frames):
    xs = np.geomspace(1, total_steps, n_frames).astype(int)
    return np.unique(xs)

# ----------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--N',        type=int,   default=44,    help='chain length')
    ap.add_argument('--steps',    type=int,   default=50000, help='Monte-Carlo steps')

    ap.add_argument('--T',        type=float, default=0.5,   help='temperature (k_B=1)')
    ap.add_argument('--Tc',       type=float, default=2.0,   help='LCST crossover temperature')
    ap.add_argument('--width',    type=float, default=0.1,   help='width of LCST crossover in T')

    ap.add_argument('--eps',      type=float, default=1.0,
                    help='high-T contact strength eps_high (low-T approx 0)')
    ap.add_argument('--eta',      type=float, default=1.0,
                    help='low-T kink penalty eta_low (high-T approx 0)')
    ap.add_argument('--coop',     type=float, default=0.5,
                    help='high-T cooperative amplitude J_high (low-T approx 0)')

    ap.add_argument('--seed',     type=int,   default=42,    help='RNG seed')
    args = ap.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    # Temperature-dependent couplings
    sT     = lcst_switch(args.T, args.Tc, args.width)

    eps_T  = args.eps  * sT          # turns on at high T
    eta_T  = args.eta  * (1.0 - sT)  # turns off at high T (stiff at low T)
    J_T    = args.coop * sT          # turns on at high T

    print(f"T = {args.T:.3f}, Tc = {args.Tc:.3f}, width = {args.width:.3f}")
    print(f"s(T) = {sT:.3f}")
    print(f"eps(T) = {eps_T:.3f} (high-T attraction)")
    print(f"eta(T) = {eta_T:.3f} (low-T stiffness)")
    print(f"J(T)   = {J_T:.3f} (high-T cooperativity)")

    # initial chain
    chain: List[Vec] = [(i,0,0) for i in range(args.N)]
    occ = set(chain)

    E = energy(chain, occ, eps_T, eta_T, J_T)
    beta = 1.0 / args.T

    acc = 0
    snapshots = []
    sample_every = 1000

    record_interval = max(1, args.steps // 2000)
    saved_steps, E_traj2, Rg_traj2 = [], [], []
    frames_to_save = set(log_schedule(args.steps, 2000))
    frames = []

    for step in range(1, args.steps + 1):
        if step % record_interval == 0:
            saved_steps.append(step)
            E_traj2.append(E)
            Rg_traj2.append(radius_of_gyration(chain))

        move = random.choice(MOVE_FUNCS)
        ok, chain_new, occ_new = move(chain, occ)
        if not ok:
            continue

        dE = energy(chain_new, occ_new, eps_T, eta_T, J_T) - E
        if dE <= 0 or random.random() < math.exp(-beta * dE):
            chain, occ, E = chain_new, occ_new, E + dE
            acc += 1

        if step % sample_every == 0:
            sample_trajectory(chain, snapshots)

        if step in frames_to_save:
            sample_trajectory(chain, frames)

    print(f"Acceptance ratio: {acc/args.steps:.3f}")

    # averages over last 30% of recorded samples
    if len(saved_steps) > 0:
        n = len(saved_steps)
        start_idx = int(math.floor(n * 0.7))
        if start_idx < 0:
            start_idx = 0

        E_slice = np.array(E_traj2[start_idx:]) if len(E_traj2) > start_idx else np.array([E])
        Rg_slice = np.array(Rg_traj2[start_idx:]) if len(Rg_traj2) > start_idx else np.array([radius_of_gyration(chain)])

        E_mean = float(np.nanmean(E_slice))
        E_std  = float(np.nanstd(E_slice, ddof=0))
        Rg_mean = float(np.nanmean(Rg_slice))
        Rg_std  = float(np.nanstd(Rg_slice, ddof=0))

        # placeholders to keep temp_scan compatibility
        Rg_full_mean = 0.0
        Rg_full_std  = 0.0
        HH_mean = 0.0
        HH_std  = 0.0

        print(f"E_mean = {E_mean:.3f} ± {E_std:.3f}")
        print(f"Rg_back_mean = {Rg_mean:.3f} ± {Rg_std:.3f}")
        print(f"Rg_full_mean = {Rg_full_mean:.3f} ± {Rg_full_std:.3f}")
        print(f"HH_mean = {HH_mean:.3f} ± {HH_std:.3f}")

    # optional movie
    if len(frames) > 0:
        try:
            render_movie(frames, out="polymer.mp4", fps=24, stride=1)
        except Exception:
            pass

if __name__ == "__main__":
    main()
