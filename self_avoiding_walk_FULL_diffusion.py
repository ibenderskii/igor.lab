#!/usr/bin/env python3
#test
print ("ur mom")

"""
MC polymer collapse
3d lattice, attractive nearest-neighbour contacts,
Metropolis with pivot / crank-shaft / end-flip moves.
"""
from __future__ import annotations
import argparse, random, math
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
from scipy.optimize import curve_fit
Vec = Tuple[int, int, int]

#helper functions for Vec
NN_VECS: List[Vec] = [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]

def add(a:Vec, b:Vec) -> Vec: return (a[0]+b[0], a[1]+b[1], a[2]+b[2])
def sub(a:Vec, b:Vec) -> Vec: return (a[0]-b[0], a[1]-b[1], a[2]-b[2])

# 90° rotations
ROT_X = lambda v: ( v[0],  -v[2],  v[1])
ROT_Y = lambda v: ( v[2],   v[1], -v[0])
ROT_Z = lambda v: (-v[1],   v[0],  v[2])
ROTATIONS = [ ROT_X, ROT_Y, ROT_Z ]

def energy(chain:List[Vec], occ:set[Vec], eps:float) -> float:
    """Return −ε × (# non-bonded contacts)."""
    E = 0.0
    N=len(chain)
    for i, r in enumerate(chain):
        prev = chain[i-1] if i > 0       else None
        next = chain[i+1] if i < N-1     else None
        for v in NN_VECS:
            nbr = add(r, v)
            if nbr in occ and nbr not in (prev, next):
                E -= eps
    return 0.5*E        # each contact counted twice


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
    new_site_candidates = [add(chain[end], v) for v in NN_VECS
                           if add(chain[end], v) not in occ
                           and sub(add(chain[end], v), chain[anchor]) in NN_VECS]
    if not new_site_candidates:
        return False, chain, occ
    r_new = random.choice(new_site_candidates)
    new_chain = chain.copy()
    new_chain[end] = r_new
    new_occ = (occ - { chain[end] }) | { r_new }
    # keep bond length 1 for the new end bond 
    return True, new_chain, new_occ

MOVE_FUNCS = [attempt_pivot, attempt_crankshaft, attempt_end_move]
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
# ----------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--N',        type=int,   default=100,    help='chain length')
    ap.add_argument('--steps',    type=int,   default=1000000, help='Monte-Carlo steps')
    ap.add_argument('--eps',      type=float, default=1.0,   help='contact energy ε')
    ap.add_argument('--T',        type=float, default=3.0,   help='temperature (k_B=1)')
    ap.add_argument('--seed',     type=int,   default=None,  help='RNG seed')
    args = ap.parse_args()

    if args.seed is not None:
        random.seed(args.seed); np.random.seed(args.seed)

    #  initial chain-
    chain: List[Vec] = [(i,0,0) for i in range(args.N)]
    occ = set(chain)
    E = energy(chain, occ, args.eps)
    beta = 1.0/args.T

    acc = 0
    snapshots = []
    sample_every = 1000          # MC steps between saved frames
    dt_frame     = sample_every  # one attempted move per monomer 

    record_interval = max(1, args.steps // 2000)
    saved_steps, E_traj2, Rg_traj2 = [], [], []



    for step in range(1, args.steps+1):
        if step % record_interval == 0:
            saved_steps.append(step)
            E_traj2.append(E)
            Rg_traj2.append(radius_of_gyration(chain))

        move = random.choice(MOVE_FUNCS)
        ok, chain_new, occ_new = move(chain, occ)
        if not ok:
            continue
        dE = energy(chain_new, occ_new, args.eps) - E
        if dE <= 0 or random.random() < math.exp(-beta*dE):
            chain, occ, E = chain_new, occ_new, E+dE
            acc += 1

    
        if step % sample_every == 0:
            sample_trajectory(chain, snapshots)

    trajectory = np.stack(snapshots)

    print(f"Acceptance ratio: {acc/args.steps:.3f}")
    print(f"⟨E⟩ = {np.mean(E_traj2):.2f}   ⟨R_g⟩ = {np.mean(Rg_traj2):.2f}")
   





    # plotting
    plt.plot(saved_steps, Rg_traj2, label = 'Rg raw' )
    plt.title('Rg vs steps')
    plt.show()
    
    fig, ax = plt.subplots(1,2, figsize=(10,4))
    ax[0].plot(E_traj2); ax[0].set_xlabel('MC block'); ax[0].set_ylabel('E')
    ax[1].plot(Rg_traj2); ax[1].set_xlabel('MC block'); ax[1].set_ylabel(r'$R_g$')
    plt.tight_layout(); plt.show()
   
    #curve fitting:
    burn = int(0.30 * trajectory.shape[0])  
    traj_eq = trajectory[burn:]
    traj_eq = traj_eq - traj_eq.mean(axis=1, keepdims=True)
    dt_frame = sample_every   
    
    Rmax   = np.linalg.norm(traj_eq[-1], axis=1).max()   
    r_bins = np.linspace(0, Rmax, 21)     # 20 radial bins
    tau    = 5                            # evaluate MSD after 5 saved frames


    r_mid, D_r, hits = diffusion_vs_radius(traj_eq, r_bins, tau, dt_frame)
    # keep only well-populated bins
    mask = (hits > 50) & np.isfinite(D_r)
    if mask.sum() < 4:
        raise ValueError("Not enough bins with counts>50 to fit 4 parameters. Run longer or merge bins.")
    r_fit = r_mid[mask]
    D_fit = D_r[mask]
    w_fit = hits[mask]    
    print(mask.sum(), "bins with enough hits")
    print(traj_eq.shape[0] , "traj frames")

    p0 = (D_fit.min(), D_fit.max(), r_fit[np.argmin(np.abs(D_fit-np.median(D_fit)))], 0.1*Rmax)
    sigma = 1/np.sqrt(np.maximum(w_fit, 2))
    popt, pcov = curve_fit(tanh_step, r_fit, D_fit, p0=p0, sigma=sigma, absolute_sigma=True, bounds=([0, 0, 0, 1e-6], [np.inf, np.inf, r_bins[-1], r_bins[-1]]))
    D_core, D_shell, r_c, w = popt
    print(f"Fitted parameters: D_core={D_core:.3f}, D_shell={D_shell:.3f}, r_c={r_c:.3f}, w={w:.3f}")
    r_plot = np.linspace(0, Rmax, 400)
    #plot fit and data
    plt.figure(figsize=(6,4))
    plt.plot(r_mid, D_r, 'o', alpha=0.4, label='raw (post-burn)')
    plt.plot(r_fit, D_fit, 'o', label='fit bins')
    plt.plot(r_plot, tanh_step(r_plot, *popt), '-', label='tanh fit')
    plt.xlabel('r'); plt.ylabel('D(r)'); plt.legend(); plt.tight_layout(); plt.show()

# RUn
if __name__ == "__main__":
    main()
