#!/usr/bin/env python3

"""
SAW on 3d cubic lattice, attractive nearest-neighbour contacts,
Metropolis MC  with canonical pivot / crank-shaft / end-flip moves.
"""
from __future__ import annotations
import argparse, dataclasses, random, math
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, List
from scipy.optimize import curve_fit
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
Vec = Tuple[int, int, int]

#helper functions for Vec
NN_VECS: List[Vec] = [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]

def add(a:Vec, b:Vec) -> Vec: return (a[0]+b[0], a[1]+b[1], a[2]+b[2])
def sub(a:Vec, b:Vec) -> Vec: return (a[0]-b[0], a[1]-b[1], a[2]-b[2])

# 24 proper cubic rotations (orientation-preserving signed permutations)
from itertools import permutations, product

def _det3(M):
    (a,b,c), (d,e,f), (g,h,i) = M
    return a*(e*i - f*h) - b*(d*i - f*g) + c*(d*h - e*g)

def _generate_cubic_rotations():
    rots = []
    for perm in permutations([0, 1, 2]):
        for signs in product([-1, 1], repeat=3):
            M = [[0,0,0],[0,0,0],[0,0,0]]
            for r, c in enumerate(perm):
                M[r][c] = signs[r]
            Mt = (tuple(M[0]), tuple(M[1]), tuple(M[2]))
            if _det3(Mt) == 1:
                rots.append(Mt)
    # drop identity to avoid wasting proposals
    I = ((1,0,0),(0,1,0),(0,0,1))
    rots = [M for M in rots if M != I]
    return rots  # 23 matrices

ROT_MATS = _generate_cubic_rotations()

def apply_rot(M, v: Vec) -> Vec:
    x, y, z = v
    return (
        M[0][0]*x + M[0][1]*y + M[0][2]*z,
        M[1][0]*x + M[1][1]*y + M[1][2]*z,
        M[2][0]*x + M[2][1]*y + M[2][2]*z,
    )
def energy(chain:List[Vec], occ:set[Vec], dh:float, ds:float, T:float) -> float:
    """Effective free-energy of a configuration.

    Implements:  H = N_nb (Δh - T Δs)
    where N_nb is the number of *unique* non-bonded nearest-neighbour contacts.

    Note: this depends explicitly on T; with Metropolis using β = 1/T, the sampled
    weight is exp(-H/T) = exp(-(Δh/T - Δs) N_nb).
    """
    cnt = 0
    N = len(chain)
    for idx, r in enumerate(chain):
        prev = chain[idx-1] if idx > 0   else None
        nxt  = chain[idx+1] if idx < N-1 else None
        for v in NN_VECS:
            nbr = add(r, v)
            if nbr in occ and nbr not in (prev, nxt):
                cnt += 1
    n_nb = 0.5 * cnt  # each contact seen from both endpoints
    return n_nb * (dh - T * ds)

def contact_count(chain: List[Vec], occ: set[Vec]) -> float:
    """Number of *unique* non-bonded nearest-neighbour contacts.

    We scan each monomer and count occupied nearest neighbours that are
    not its bonded neighbours; divide by 2 because each contact is seen
    from both endpoints.
    """
    m = 0
    N = len(chain)
    for i, r in enumerate(chain):
        prev = chain[i-1] if i > 0   else None
        nxt  = chain[i+1] if i < N-1 else None
        for v in NN_VECS:
            nbr = add(r, v)
            if nbr in occ and nbr not in (prev, nxt):
                m += 1
    return 0.5 * m


def radius_of_gyration(chain:List[Vec]) -> float:
    r = np.array(chain, dtype=float)
    com = r.mean(axis=0)
    return math.sqrt(((r - com)**2).sum(axis=1).mean())

def attempt_pivot(chain, occ) -> Tuple[bool, List[Vec], set[Vec]]:
    n = len(chain)
    i = random.randrange(1, n-1)         # pivot monomer (not ends)
    tail = chain[i+1:]
    head = chain[:i+1]

    M = random.choice(ROT_MATS)
    new_tail = []
    new_occ  = set(head)
    pivot = chain[i]
    for r in tail:
        dr = sub(r, pivot)
        r2 = add(pivot, apply_rot(M, dr))
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
    """
    Symmetric end move (good for detailed balance without a Hastings correction):
    - choose an end with prob 1/2
    - choose a lattice direction with prob 1/6
    - accept if the target site is empty AND the end bond remains length 1
    """
    n = len(chain)
    end = 0 if random.random() < 0.5 else n-1
    anchor = 1 if end == 0 else n-2

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


@dataclasses.dataclass
class ChainState:
    """Mutable MC state: chain positions, occupied-site set, and current energy."""
    chain: List[Vec]
    occ:   set
    E:     float

    @classmethod
    def initial_straight(cls, N: int, dh: float, ds: float, T: float) -> "ChainState":
        chain = [(i, 0, 0) for i in range(N)]
        occ   = set(chain)
        return cls(chain=chain, occ=occ, E=energy(chain, occ, dh, ds, T))


# ------------------------------------------------------------------
def sample_trajectory(chain, snap_list):
    """Append a snapshot (N,3) float64 array to snap_list."""
    snap_list.append(np.asarray(chain, dtype=np.float64))

# ------------------------------------------------------------------
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
def run_single_temperature(
    N: int,
    steps: int,
    T: float,
    dh: float,
    ds: float,
    seed: int | None = None,
    dist_dir: str = "dists",
    rg_bins: int = 80,
) -> dict:
    """
    Run the MC simulation at one temperature and return structured results.

    Returns a dict with keys:
        E_mean, E_std, Rg_back_mean, Rg_back_std,
        C_mean, C_std, C_var_pop, C_var_samp,
        dist_file (str path or None), acceptance_ratio.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    state = ChainState.initial_straight(N, dh, ds, T)
    beta  = 1.0 / T

    acc           = 0
    snapshots: list = []
    sample_every  = 1500
    record_interval = max(1, steps // 2000)
    saved_steps: list = []
    E_traj: list = []
    Rg_traj: list = []
    C_traj: list  = []
    frames_to_save = set(log_schedule(steps, 2000))
    frames: list  = []

    for step in range(1, steps + 1):
        if step % record_interval == 0:
            saved_steps.append(step)
            E_traj.append(state.E)
            Rg_traj.append(radius_of_gyration(state.chain))
            C_traj.append(contact_count(state.chain, state.occ))

        move = random.choice(MOVE_FUNCS)
        ok, chain_new, occ_new = move(state.chain, state.occ)
        if not ok:
            continue
        dE = energy(chain_new, occ_new, dh, ds, T) - state.E
        if dE <= 0 or random.random() < math.exp(-beta * dE):
            state.chain, state.occ, state.E = chain_new, occ_new, state.E + dE
            acc += 1

        if step % sample_every == 0:
            sample_trajectory(state.chain, snapshots)
        if step in frames_to_save:
            sample_trajectory(state.chain, frames)

    dist_file: str | None = None

    if saved_steps:
        n = len(saved_steps)
        start_idx = int(math.floor(n * 0.7))

        E_slice  = np.array(E_traj[start_idx:])  if len(E_traj)  > start_idx else np.array([state.E])
        Rg_slice = np.array(Rg_traj[start_idx:]) if len(Rg_traj) > start_idx else np.array([radius_of_gyration(state.chain)])
        C_slice  = np.array(C_traj[start_idx:])  if len(C_traj)  > start_idx else np.array([contact_count(state.chain, state.occ)])

        E_mean   = float(np.nanmean(E_slice))
        E_std    = float(np.nanstd(E_slice,  ddof=0))
        Rg_mean  = float(np.nanmean(Rg_slice))
        Rg_std   = float(np.nanstd(Rg_slice, ddof=0))
        C_mean   = float(np.nanmean(C_slice))
        C_std    = float(np.nanstd(C_slice,  ddof=0))
        C_var_pop  = float(np.nanvar(C_slice, ddof=0))
        C_var_samp = float(np.nanvar(C_slice, ddof=1)) if C_slice.size > 1 else float("nan")

        try:
            dist_dir_path = Path(dist_dir)
            dist_dir_path.mkdir(parents=True, exist_ok=True)

            c_int = np.rint(C_slice.astype(float)).astype(int)
            c_vals, c_counts = np.unique(c_int, return_counts=True)
            c_prob = c_counts.astype(float) / max(1, c_counts.sum())

            rg = Rg_slice.astype(float)
            rg = rg[np.isfinite(rg)]
            if rg.size == 0:
                rg_edges = np.linspace(0.0, 1.0, int(rg_bins) + 1)
                rg_prob  = np.zeros(int(rg_bins), dtype=float)
            else:
                rg_min, rg_max = float(rg.min()), float(rg.max())
                pad = 1e-9 if rg_max <= rg_min else 0.02 * (rg_max - rg_min)
                rg_edges = np.linspace(rg_min - pad, rg_max + pad, int(rg_bins) + 1)
                rg_counts, _ = np.histogram(rg, bins=rg_edges)
                rg_prob = rg_counts.astype(float)
                if rg_prob.sum() > 0:
                    rg_prob /= rg_prob.sum()

            seed_tag = seed if seed is not None else "na"
            out_path = dist_dir_path / f"{Path(__file__).stem}_N{N}_T{T:.6g}_seed{seed_tag}.npz"
            np.savez_compressed(
                out_path,
                c_vals=c_vals, c_prob=c_prob,
                rg_edges=rg_edges, rg_prob=rg_prob,
                T=float(T), N=int(N), steps=int(steps), seed=seed_tag,
                dh=float(dh), ds=float(ds), n_samples=int(rg.size),
            )
            dist_file = str(out_path)
        except Exception:
            pass
    else:
        C0  = float(contact_count(state.chain, state.occ))
        Rg0 = float(radius_of_gyration(state.chain))
        E_mean   = state.E;  E_std    = 0.0
        Rg_mean  = Rg0;      Rg_std   = 0.0
        C_mean   = C0;       C_std    = 0.0
        C_var_pop  = 0.0;    C_var_samp = float("nan")

    return {
        "E_mean": E_mean,         "E_std": E_std,
        "Rg_back_mean": Rg_mean,  "Rg_back_std": Rg_std,
        "C_mean": C_mean,         "C_std": C_std,
        "C_var_pop": C_var_pop,   "C_var_samp": C_var_samp,
        "dist_file": dist_file,
        "acceptance_ratio": acc / steps,
    }


# ----------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--N',        type=int,   default=300,    help='chain length')
    ap.add_argument('--steps',    type=int,   default=500000, help='Monte-Carlo steps')
    ap.add_argument('--dh',      type=float, default=378.96,   help='contact energy ')
    ap.add_argument('--ds',      type=float, default=1.39686,   help='contact energy ')
    ap.add_argument('--T',        type=float, default=360,   help='temperature (k_B=1)')
    ap.add_argument('--seed',     type=int,   default=42,  help='RNG seed')
    ap.add_argument('--dist_dir', type=str, default='dists', help='where to write equilibrium distribution .npz')
    ap.add_argument('--rg_bins', type=int, default=80, help='number of histogram bins for P(Rg)')
    args = ap.parse_args()

    r = run_single_temperature(
        N=args.N, steps=args.steps, T=args.T,
        dh=args.dh, ds=args.ds, seed=args.seed,
        dist_dir=args.dist_dir, rg_bins=args.rg_bins,
    )
    print(f"Acceptance ratio: {r['acceptance_ratio']:.3f}")
    if r["dist_file"]:
        print(f"DIST_FILE = {r['dist_file']}")
    print(f"E_mean = {r['E_mean']:.3f} ± {r['E_std']:.3f}")
    print(f"Rg_back_mean = {r['Rg_back_mean']:.3f} ± {r['Rg_back_std']:.3f}")
    print(f"C_mean = {r['C_mean']:.3f} ± {r['C_std']:.3f}")
    print(f"C_var(pop) = {r['C_var_pop']:.3f}")
    print(f"C_var(sample) = {r['C_var_samp']:.3f}")

# RUn
if __name__ == "__main__":
    main()