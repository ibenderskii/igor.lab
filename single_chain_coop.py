#!/usr/bin/env python3 

"""
MC polymer collapse
3d lattice, nearest-neighbour contacts,
Metropolis with pivot / crank-shaft / end-flip moves.
Now with optional cooperativity between non-bonded contacts
and an LCST-like temperature-dependent contact energy.

Energy:
    E = -eps_eff(T) * (# non-bonded contacts)
        - coop * sum_i C(c_i, 2)

where eps_eff(T) = eps0 * (T - Tc).
Below Tc, contacts are unfavorable; above Tc, they are favorable.
"""

from __future__ import annotations
import argparse, random, math
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
from scipy.optimize import curve_fit
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D 

Vec = Tuple[int, int, int]

# helper functions for Vec
NN_VECS: List[Vec] = [
    (1,0,0), (-1,0,0),
    (0,1,0), (0,-1,0),
    (0,0,1), (0,0,-1)
]

def add(a:Vec, b:Vec) -> Vec:
    return (a[0]+b[0], a[1]+b[1], a[2]+b[2])

def sub(a:Vec, b:Vec) -> Vec:
    return (a[0]-b[0], a[1]-b[1], a[2]-b[2])

# 90° rotations
ROT_X = lambda v: ( v[0],  -v[2],  v[1])
ROT_Y = lambda v: ( v[2],   v[1], -v[0])
ROT_Z = lambda v: (-v[1],   v[0],  v[2])
ROTATIONS = [ ROT_X, ROT_Y, ROT_Z ]


def energy(chain: List[Vec],
           occ: set[Vec],
           eps_eff: float,
           coop: float = 0.0) -> float:
    """
    Return energy with non-bonded contacts and optional cooperativity.

    Base term:
        E_contact = -eps_eff * (# non-bonded nearest-neighbour contacts)

    Cooperative term (if coop > 0):
        For each monomer i with c_i non-bonded NN contacts,
        add an extra -coop * C(c_i, 2) = -coop * c_i*(c_i-1)/2.
        This rewards monomers that are part of multiple contacts.

    NOTE: eps_eff here is already the temperature-dependent effective
    contact strength (for LCST we set eps_eff = eps0 * (T - Tc) in main()).
    """
    N = len(chain)
    contact_E = 0.0
    local_contacts: List[int] = []

    for i, r in enumerate(chain):
        prev = chain[i-1] if i > 0   else None
        nxt  = chain[i+1] if i < N-1 else None

        c_i = 0  # number of non-bonded contacts of monomer i
        for v in NN_VECS:
            nbr = add(r, v)
            if nbr in occ and nbr not in (prev, nxt):
                contact_E -= eps_eff
                c_i += 1
        local_contacts.append(c_i)

    # each contact counted twice (once from each end)
    E = 0.5 * contact_E

    # cooperative contribution: sum over monomers of C(c_i, 2)
    if coop != 0.0:
        coop_E = 0.0
        for c in local_contacts:
            if c >= 2:
                coop_E -= coop * (c * (c - 1) / 2.0)
        E += coop_E

    return E


def radius_of_gyration(chain:List[Vec]) -> float:
    r = np.array(chain, dtype=float)
    com = r.mean(axis=0)
    return math.sqrt(((r - com)**2).sum(axis=1).mean())


def attempt_pivot(chain, occ) -> Tuple[bool, List[Vec], set[Vec]]:
    n = len(chain)
    i = random.randrange(1, n-1)         # pivot monomer (not ends)
    tail = chain[i+1:]
    head = chain[:i+1]

    rot = random.choice(ROTATIONS)
    new_tail = []
    new_occ  = set(head)
    pivot = chain[i]
    for r in tail:
        dr = sub(r, pivot)
        r2 = add(pivot, rot(dr))
        if r2 in new_occ:                # self-intersection
            return False, chain, occ
        new_tail.append(r2)
        new_occ.add(r2)
    new_chain = head + new_tail
    return True, new_chain, new_occ


def attempt_crankshaft(chain, occ):
    """
    Local 'kink flip' crankshaft move:
    Works only when the local triplet (a,b,c) forms a 90° kink.
    """
    n = len(chain)
    i = random.randrange(1, n-1)        # b = chain[i], needs neighbors on both sides
    a, b, c = chain[i-1], chain[i], chain[i+1]

    u1 = sub(b, a)   # bond a->b
    u2 = sub(c, b)   # bond b->c

    # both bonds must be unit lattice vectors
    if u1 not in NN_VECS or u2 not in NN_VECS:
        return False, chain, occ

    # reject straight segments (parallel or antiparallel)
    if u1 == u2 or u1 == (-u2[0], -u2[1], -u2[2]):
        return False, chain, occ

    # we only accept perfect 90° kinks
    if (u1[0]*u2[0] + u1[1]*u2[1] + u1[2]*u2[2]) != 0:
        return False, chain, occ

    # proposed new middle monomer (180° flip of the kink)
    b_new = add(a, u2)

    # must be empty
    if b_new in occ:
        return False, chain, occ

    # bonds must remain unit length
    if sub(b_new, a) not in NN_VECS or sub(c, b_new) not in NN_VECS:
        return False, chain, occ

    # commit
    new_chain = chain.copy()
    new_chain[i] = b_new
    new_occ = (occ - {b}) | {b_new}
    return True, new_chain, new_occ


def attempt_end_move(chain, occ) -> Tuple[bool, List[Vec], set[Vec]]:
    n = len(chain)
    end = 0 if random.random() < 0.5 else n-1
    anchor = 1 if end == 0 else n-2
    new_site_candidates = [
        add(chain[end], v) for v in NN_VECS
        if add(chain[end], v) not in occ
        and sub(add(chain[end], v), chain[anchor]) in NN_VECS
    ]
    if not new_site_candidates:
        return False, chain, occ
    r_new = random.choice(new_site_candidates)
    new_chain = chain.copy()
    new_chain[end] = r_new
    new_occ = (occ - { chain[end] }) | { r_new }
    return True, new_chain, new_occ


MOVE_FUNCS = [attempt_pivot, attempt_crankshaft, attempt_end_move]


def sample_trajectory(chain, snap_list):
    """Append a snapshot (N,3) float64 array to snap_list."""
    snap_list.append(np.asarray(chain, dtype=np.float64))


def diffusion_vs_radius(trajectory: np.ndarray,
                        r_bins: np.ndarray,
                        tau_frames: int,
                        dt_frame: float):
    
    if tau_frames >= trajectory.shape[0]:
        raise ValueError("tau_frames larger than trajectory length")

    # displacement over lag tau
    disp = trajectory[tau_frames:] - trajectory[:-tau_frames]        # (F-tau, N, 3)
    msd  = np.sum(disp*disp, axis=2)                                 # (F-tau, N)

    # r at start of each pair
    r0   = np.linalg.norm(trajectory[:-tau_frames], axis=2)          # (F-tau, N)

    # flatten to 1-D
    r_flat  = r0.ravel()
    msd_flat = msd.ravel()

    bin_idx = np.digitize(r_flat, r_bins) - 1                        # 0 … n_bins-1
    n_bins  = len(r_bins) - 1
    D       = np.full(n_bins, np.nan)
    counts  = np.zeros(n_bins, dtype=np.int64)

    for k in range(n_bins):
        mask = bin_idx == k
        if mask.any():
            counts[k] = mask.sum()
            D[k]      = msd_flat[mask].mean() / (6.0 * dt_frame * tau_frames)

    r_centers = 0.5*(r_bins[:-1] + r_bins[1:])
    return r_centers, D, counts  


def tanh_step(r, D_core, D_shell, r_c, w):
    return D_shell - 0.5*(D_shell - D_core)*(1 - np.tanh((r - r_c)/w))


def render_movie(snaps, out="polymer.mp4", fps=30, stride=1):
    """snaps: list of (N,3) arrays collected during the run."""
    traj = np.asarray(snaps, dtype=float)[::stride]          # (F, N, 3)
    if traj.size == 0:
        return
    F = traj.shape[0]
    mins = traj.reshape(-1,3).min(0) - 2
    maxs = traj.reshape(-1,3).max(0) + 2

    fig = plt.figure(figsize=(5,5))
    ax  = fig.add_subplot(111, projection='3d')
    (line,) = ax.plot([], [], [], '-o', ms=3)
    ax.set_xlim(mins[0], maxs[0]); ax.set_ylim(mins[1], maxs[1]); ax.set_zlim(mins[2], maxs[2])
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
    # indices in [1..total_steps] roughly logarithmically spaced
    xs = np.geomspace(1, total_steps, n_frames).astype(int)
    return np.unique(xs)


# ----------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--N',        type=int,   default=44,    help='chain length')
    ap.add_argument('--steps',    type=int,   default=50000, help='Monte-Carlo steps')
    ap.add_argument('--eps',      type=float, default=1.0,
                    help='LCST contact amplitude eps0 (eps_eff = eps0*(T - Tc))')
    ap.add_argument('--Tc',       type=float, default=1.5,
                    help='LCST crossover temperature Tc')
    ap.add_argument('--T',        type=float, default=0.5,
                    help='temperature (k_B=1)')
    ap.add_argument('--seed',     type=int,   default=42,    help='RNG seed')
    ap.add_argument('--coop',     type=float, default=0.5,
                    help='cooperativity strength for non-bonded contacts')
    args = ap.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    # LCST-style effective contact strength:
    #   eps_eff(T) = eps0 * (T - Tc)
    # Below Tc: eps_eff < 0 -> contacts penalized.
    # Above Tc: eps_eff > 0 -> contacts favorable.
    eps_eff = args.eps * (args.T - args.Tc)

    print(f"T = {args.T:.3f}, Tc = {args.Tc:.3f}, eps0 = {args.eps:.3f}, "
          f"eps_eff(T) = {eps_eff:.3f}, coop = {args.coop:.3f}")

    # initial chain
    chain: List[Vec] = [(i,0,0) for i in range(args.N)]
    occ = set(chain)
    E = energy(chain, occ, eps_eff, args.coop)
    beta = 1.0/args.T

    acc = 0
    snapshots = []
    sample_every = 1000          # MC steps between saved frames
    dt_frame     = sample_every  # one attempted move per monomer 

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
        dE = energy(chain_new, occ_new, eps_eff, args.coop) - E
        if dE <= 0 or random.random() < math.exp(-beta*dE):
            chain, occ, E = chain_new, occ_new, E + dE
            acc += 1

        if step % sample_every == 0:
            sample_trajectory(chain, snapshots)

        if step in frames_to_save:
            sample_trajectory(chain, frames)

    trajectory = np.stack(snapshots) if snapshots else np.empty((0, args.N, 3))

    print(f"Acceptance ratio: {acc/args.steps:.3f}")

    # Compute averages over last 30% of recorded samples (if available)
    if len(saved_steps) > 0:
        n = len(saved_steps)
        start_idx = int(math.floor(n * 0.7))
        if start_idx < 0:
            start_idx = 0
        # slices
        E_slice = np.array(E_traj2[start_idx:]) if len(E_traj2) > start_idx else np.array([E])
        Rg_slice = np.array(Rg_traj2[start_idx:]) if len(Rg_traj2) > start_idx else np.array([radius_of_gyration(chain)])

        E_mean = float(np.nanmean(E_slice))
        E_std  = float(np.nanstd(E_slice, ddof=0))
        Rg_mean = float(np.nanmean(Rg_slice))
        Rg_std  = float(np.nanstd(Rg_slice, ddof=0))

        # placeholders for models that do not provide these quantities
        Rg_full_mean = 0.0
        Rg_full_std  = 0.0
        HH_mean = 0.0
        HH_std  = 0.0

        # Print in instrumented format expected by temp_scan.py
        print(f"E_mean = {E_mean:.3f} ± {E_std:.3f}")
        print(f"Rg_back_mean = {Rg_mean:.3f} ± {Rg_std:.3f}")
        print(f"Rg_full_mean = {Rg_full_mean:.3f} ± {Rg_full_std:.3f}")
        print(f"HH_mean = {HH_mean:.3f} ± {HH_std:.3f}")
    else:
        # fallback to instantaneous prints if no recorded samples
        print(f"⟨E⟩ = {np.mean(E_traj2) if E_traj2 else E:.2f}   ⟨R_g⟩ = {np.mean(Rg_traj2) if Rg_traj2 else radius_of_gyration(chain):.2f}")
        print(f"E_mean = {E:.3f} ± {0.000:.3f}")
        print(f"Rg_back_mean = {radius_of_gyration(chain):.3f} ± {0.000:.3f}")
        # placeholders
        print(f"Rg_full_mean = {0.000:.3f} ± {0.000:.3f}")
        print(f"HH_mean = {0.000:.3f} ± {0.000:.3f}")


if __name__ == "__main__":
    main()
