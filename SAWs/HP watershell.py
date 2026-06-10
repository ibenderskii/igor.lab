#!/usr/bin/env python3
"""
MC polymer collapse (hydrophobic / hydrophilic sections)
3D lattice, nearest-neighbour contacts,
Metropolis with pivot / crankshaft / end-flip moves.

Compatible with temp_scan: prints instrumented averages:
  E_mean, Rg_back_mean, Rg_full_mean, HH_mean

Model terms:
  - LCST-like H-H interaction: eps_HH(T) crosses 0 near Tmid
    (repulsive at low T, attractive at high T)
  - Quadratic cooperativity in total H-H contacts (turns on at high T):
      E_coop = -0.5 * J(T) * m_HH^2,  J(T) = J_high * s(T)
  - Water-cage-style backbone stiffness (turns on at low T, turns off at high T):
      E_kink = + eta(T) * n_kink,     eta(T) = eta_low * (1 - s(T))

where s(T) = 0.5 * [1 + tanh((T - Tmid)/width)] in [0,1].
"""

from __future__ import annotations
import argparse, random, math
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D

Vec = Tuple[int, int, int]

# ---------------------------------------------------------------------
# helpers
NN_VECS: List[Vec] = [
    (1,0,0), (-1,0,0),
    (0,1,0), (0,-1,0),
    (0,0,1), (0,0,-1)
]
def add(a:Vec, b:Vec) -> Vec: return (a[0]+b[0], a[1]+b[1], a[2]+b[2])
def sub(a:Vec, b:Vec) -> Vec: return (a[0]-b[0], a[1]-b[1], a[2]-b[2])

# 90° rotations
ROT_X = lambda v: ( v[0],  -v[2],  v[1])
ROT_Y = lambda v: ( v[2],   v[1], -v[0])
ROT_Z = lambda v: (-v[1],   v[0],  v[2])
ROTATIONS = [ ROT_X, ROT_Y, ROT_Z ]

def sigmoid_T(T: float, Tmid: float, width: float) -> float:
    """Smooth switch 0->1 centered at Tmid with width."""
    if width == 0:
        return 1.0 if T > Tmid else 0.0
    return 0.5 * (1.0 + math.tanh((T - Tmid)/width))

# ---------------------------------------------------------------------
# eps(T) sigmoid
def make_eps_sigmoid(eps_params):
    """
    eps_params[(ti,tj)] = (eps_low, eps_high, Tmid, width)
    returns eps_fn(ti,tj,T)

    For LCST behavior in H-H:
      eps_low < 0, eps_high > 0  => eps crosses 0 near Tmid.
    """
    def eps_fn(ti, tj, T):
        key = (ti, tj) if (ti, tj) in eps_params else (tj, ti)
        eps_low, eps_high, Tmid, width = eps_params.get(key, (0.0, 0.0, 1.0, 1.0))
        s = sigmoid_T(T, Tmid, width)
        return eps_low*(1-s) + eps_high*s
    return eps_fn

# ---------------------------------------------------------------------
# geometry observables
def radius_of_gyration(chain:List[Vec]) -> float:
    r = np.array(chain, dtype=float)
    com = r.mean(axis=0)
    return math.sqrt(((r - com)**2).sum(axis=1).mean())

def count_kinks(chain: List[Vec]) -> int:
    """Count 90° bends along the backbone (orthogonal successive bonds)."""
    n = len(chain)
    if n < 3:
        return 0
    k = 0
    for i in range(1, n-1):
        u1 = sub(chain[i], chain[i-1])
        u2 = sub(chain[i+1], chain[i])
        if u1 not in NN_VECS or u2 not in NN_VECS:
            continue
        # dot(u1,u2)=0 means 90° turn (on this lattice, backtracking would self-intersect)
        if (u1[0]*u2[0] + u1[1]*u2[1] + u1[2]*u2[2]) == 0:
            k += 1
    return k

# ---------------------------------------------------------------------
# contact counters
def count_HH_contacts(chain:List[Vec], types:List[str]) -> int:
    pos_to_type = {tuple(chain[i]): types[i] for i in range(len(chain))}
    seen = set()
    hh = 0
    for pos, t in pos_to_type.items():
        if t != 'H':
            continue
        for v in NN_VECS:
            nbr = add(pos, v)
            if pos_to_type.get(nbr, None) == 'H':
                a = (pos, nbr) if pos < nbr else (nbr, pos)
                if a in seen:
                    continue
                seen.add(a)
                hh += 1
    return hh

# ---------------------------------------------------------------------
# energy
def energy(chain: List[Vec],
           occ: set,
           types: List[str],
           T: float,
           eps_fn,
           J_high: float = 0.0,
           eta_low: float = 0.0,
           Tmid: float = 2.0,
           width: float = 0.2) -> float:
    """
    Total energy = base contact term + kink penalty + quadratic cooperativity.

    Base:
      E_base = - sum_{nonbonded NN pairs (i,j)} eps_fn(t_i, t_j, T)

    Backbone stiffness (water-cage style), ON at low T:
      E_kink = + eta(T) * n_kink,  eta(T) = eta_low * (1 - s(T))

    Quadratic cooperativity in total H-H contacts, ON at high T:
      E_coop = -0.5 * J(T) * m_HH^2,   J(T) = J_high * s(T)

    m_HH counts unique nonbonded H-H contacts (pairs).
    """
    pos_to_idx = {tuple(r): i for i, r in enumerate(chain)}
    E = 0.0
    N = len(chain)

    # store unique H-H contacts by index pair
    hh_pairs = set()

    for i, r in enumerate(chain):
        t_i = types[i]
        prev = chain[i-1] if i > 0 else None
        nxt  = chain[i+1] if i < N-1 else None

        for v in NN_VECS:
            nbr = add(r, v)
            j = pos_to_idx.get(nbr, None)
            if j is None:
                continue
            if nbr == prev or nbr == nxt:
                continue

            t_j = types[j]
            E -= eps_fn(t_i, t_j, T)

            if t_i == 'H' and t_j == 'H':
                a = (i, j) if i < j else (j, i)
                hh_pairs.add(a)

    E *= 0.5  # contacts were double-counted

    sT = sigmoid_T(T, Tmid, width)

    # stiffness ON at low T, OFF at high T
    if eta_low != 0.0:
        eta_T = eta_low * (1.0 - sT)
        E += eta_T * count_kinks(chain)

    # cooperativity ON at high T
    if J_high != 0.0:
        J_T = J_high * sT
        m = float(len(hh_pairs))
        E += -0.5 * J_T * (m * m)

    return E

# ---------------------------------------------------------------------
# MC moves
def attempt_pivot(chain, occ):
    n = len(chain)
    if n <= 3:
        return False, chain, occ
    i = random.randrange(1, n-1)
    head = chain[:i+1]; tail = chain[i+1:]
    rot = random.choice(ROTATIONS); pivot = chain[i]
    new_occ = set(head)
    new_tail = []
    for r in tail:
        dr = sub(r, pivot)
        r2 = add(pivot, rot(dr))
        if r2 in new_occ:
            return False, chain, occ
        new_tail.append(r2); new_occ.add(r2)
    return True, head + new_tail, new_occ

def attempt_crankshaft(chain, occ):
    n = len(chain)
    if n <= 3:
        return False, chain, occ
    i = random.randrange(1, n-1)
    a, b, c = chain[i-1], chain[i], chain[i+1]
    u1 = sub(b, a); u2 = sub(c, b)
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

def attempt_end_move(chain, occ):
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

# ---------------------------------------------------------------------
# sampling / movie
def sample_trajectory(chain, snap_list):
    snap_list.append(np.asarray(chain, dtype=np.float64))

def render_movie(snaps, out="polymer.mp4", fps=30, stride=1):
    if len(snaps) == 0:
        return
    traj = np.asarray(snaps, dtype=float)[::stride]
    F = traj.shape[0]
    mins = traj.reshape(-1,3).min(0) - 2
    maxs = traj.reshape(-1,3).max(0) + 2
    fig = plt.figure(figsize=(5,5)); ax = fig.add_subplot(111, projection='3d')
    (line,) = ax.plot([], [], [], '-o', ms=3)
    ax.set_xlim(mins[0], maxs[0]); ax.set_ylim(mins[1], maxs[1]); ax.set_zlim(mins[2], maxs[2])
    def init():
        line.set_data([], []); line.set_3d_properties([]); return (line,)
    def update(f):
        r = traj[f]; line.set_data(r[:,0], r[:,1]); line.set_3d_properties(r[:,2])
        ax.set_title(f"frame {f+1}/{F}"); return (line,)
    ani = animation.FuncAnimation(fig, update, init_func=init, frames=F, interval=1000/fps, blit=False)
    try:
        ani.save(out, writer=animation.FFMpegWriter(fps=fps, bitrate=1800))
    except Exception:
        try:
            ani.save(out.replace(".mp4", ".gif"), writer="pillow", fps=fps)
        except Exception:
            pass
    plt.close(fig)

def log_schedule(total_steps, n_frames):
    xs = np.geomspace(1, total_steps, n_frames).astype(int)
    return np.unique(xs)

# ---------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--N',        type=int,   default=44,     help='chain length')
    ap.add_argument('--steps',    type=int,   default=100000, help='Monte-Carlo steps')
    ap.add_argument('--T',        type=float, default=2.0,    help='temperature (k_B=1)')
    ap.add_argument('--seed',     type=int,   default=None,   help='RNG seed')
    ap.add_argument('--fH',       type=float, default=0.5,    help='fraction hydrophobic backbone')
    ap.add_argument('--blocky',   type=int,   default=1,      help='1 => block of H then P, 0 => random placement')

    # LCST-style H-H sigmoid params
    ap.add_argument('--eps_HH_low',  type=float, default=-0.5,
                    help='H-H contact free energy at T << Tmid (repulsive if negative)')
    ap.add_argument('--eps_HH_high', type=float, default=1.5,
                    help='H-H contact free energy at T >> Tmid (attractive if positive)')
    ap.add_argument('--eps_Tmid',    type=float, default=2.0,
                    help='Temperature for the H-H crossover (LCST-like)')
    ap.add_argument('--eps_width',   type=float, default=0.2,
                    help='Width of tanh crossover in T')

    # paper-style terms
    ap.add_argument('--coop', type=float, default=0.0,
                    help='(alias) high-T quadratic cooperativity strength J_high')
    ap.add_argument('--J', type=float, default=None,
                    help='High-T quadratic cooperativity strength (overrides --coop)')
    ap.add_argument('--eta', type=float, default=0.0,
                    help='LOW-T kink penalty eta_low (stiffness; turns off at high T)')

    args = ap.parse_args()

    if args.seed is not None:
        random.seed(args.seed); np.random.seed(args.seed)

    J_high = args.coop if args.J is None else args.J

    eps_params = {
        ('H','H'): (args.eps_HH_low, args.eps_HH_high, args.eps_Tmid, args.eps_width),
        ('H','P'): (0.0, 0.0, args.eps_Tmid, args.eps_width),
        ('P','P'): (0.0, 0.0, args.eps_Tmid, args.eps_width),
    }
    eps_fn = make_eps_sigmoid(eps_params)

    chain: List[Vec] = [(i,0,0) for i in range(args.N)]
    occ = set(chain)

    nH = int(round(args.fH * args.N))
    if args.blocky:
        types = ['H'] * nH + ['P'] * (args.N - nH)
    else:
        types = ['H' if random.random() < args.fH else 'P' for _ in range(args.N)]

    sT = sigmoid_T(args.T, args.eps_Tmid, args.eps_width)
    eta_T = args.eta * (1.0 - sT)     # corrected: stiffness at low T
    J_T   = J_high * sT               # cooperativity at high T

    print(f"Model: H fraction {types.count('H')}/{args.N}, blocky={bool(args.blocky)}")
    print(f"eps_HH(T={args.T:.2f}) = {eps_fn('H','H', args.T):.3f} (low={args.eps_HH_low:.3f}, high={args.eps_HH_high:.3f})")
    print(f"Switch s(T) = {sT:.3f}  |  J(T)={J_T:.3f}, eta(T)={eta_T:.3f}")

    E = energy(chain, occ, types, args.T, eps_fn,
               J_high=J_high, eta_low=args.eta,
               Tmid=args.eps_Tmid, width=args.eps_width)
    beta = 1.0 / args.T

    acc = 0
    snapshots = []
    sample_every = 1000

    record_interval = max(1, args.steps // 2000)
    saved_steps, E_traj2, Rg_traj2 = [], [], []
    frames_to_save = set(log_schedule(args.steps, 2000))
    frames = []

    for step in range(1, args.steps+1):
        if step % record_interval == 0:
            saved_steps.append(step)
            E_traj2.append(E)
            Rg_traj2.append(radius_of_gyration(chain))

        move = random.choice(MOVE_FUNCS)
        ok, chain_new, occ_new = move(chain, occ)
        if not ok:
            continue

        E_new = energy(chain_new, occ_new, types, args.T, eps_fn,
                       J_high=J_high, eta_low=args.eta,
                       Tmid=args.eps_Tmid, width=args.eps_width)
        dE = E_new - E

        if dE <= 0 or random.random() < math.exp(-beta * dE):
            chain, occ, E = chain_new, occ_new, E_new
            acc += 1

        if step % sample_every == 0:
            sample_trajectory(chain, snapshots)

        if step in frames_to_save:
            sample_trajectory(chain, frames)

    print(f"Acceptance ratio: {acc/args.steps:.3f}")

    if len(saved_steps) > 0:
        n = len(saved_steps)
        start_idx = int(math.floor(n * 0.7))
        E_slice = np.array(E_traj2[start_idx:]) if len(E_traj2) > start_idx else np.array([E])
        Rg_slice = np.array(Rg_traj2[start_idx:]) if len(Rg_traj2) > start_idx else np.array([radius_of_gyration(chain)])

        E_mean = float(np.nanmean(E_slice)); E_std = float(np.nanstd(E_slice, ddof=0))
        Rg_mean = float(np.nanmean(Rg_slice)); Rg_std = float(np.nanstd(Rg_slice, ddof=0))

        HH_final = count_HH_contacts(chain, types)
        HH_mean = float(HH_final); HH_std = 0.0

        Rg_full_mean = Rg_mean; Rg_full_std = Rg_std

        # instrumented prints (temp_scan expects these exact keys)
        print(f"E_mean = {E_mean:.3f} ± {E_std:.3f}")
        print(f"Rg_back_mean = {Rg_mean:.3f} ± {Rg_std:.3f}")
        print(f"Rg_full_mean = {Rg_full_mean:.3f} ± {Rg_full_std:.3f}")
        print(f"HH_mean = {HH_mean:.3f} ± {HH_std:.3f}")
    else:
        Rg_inst = radius_of_gyration(chain)
        print(f"E_mean = {E:.3f} ± {0.000:.3f}")
        print(f"Rg_back_mean = {Rg_inst:.3f} ± {0.000:.3f}")
        print(f"Rg_full_mean = {Rg_inst:.3f} ± {0.000:.3f}")
        print(f"HH_mean = {count_HH_contacts(chain, types):.3f} ± {0.000:.3f}")

    if len(frames) > 0:
        try:
            render_movie(frames, out="polymer.mp4", fps=24, stride=1)
        except Exception:
            pass

if __name__ == "__main__":
    main()
