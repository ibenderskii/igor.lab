# polymer_hp_grafts.py
from __future__ import annotations
import argparse, random, math
from matplotlib import animation
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from typing import Tuple, List, Dict
Vec = Tuple[int, int, int]

# ---------- basic geometry helpers ----------
NN_VECS: List[Vec] = [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]
def add(a:Vec, b:Vec) -> Vec: return (a[0]+b[0], a[1]+b[1], a[2]+b[2])
def sub(a:Vec, b:Vec) -> Vec: return (a[0]-b[0], a[1]-b[1], a[2]-b[2])

# 90° rotations about origin
ROT_X = lambda v: ( v[0],  -v[2],  v[1])
ROT_Y = lambda v: ( v[2],   v[1], -v[0])
ROT_Z = lambda v: (-v[1],   v[0],  v[2])
ROTATIONS = [ROT_X, ROT_Y, ROT_Z]


def render_movie(snaps, out="polymer.mp4", fps=30, stride=1):
    """snaps: list of (N,3) arrays (backbone-only recommended) collected during the run."""
    if len(snaps) == 0:
        print("render_movie: no frames to render.")
        return
    traj = np.asarray(snaps, dtype=float)[::stride]          # (F, N, 3)
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
        print(f"Saved movie to {out}")
    except Exception:
        fallback = out.replace(".mp4", ".gif")
        try:
            ani.save(fallback, writer="pillow", fps=fps)
            print(f"Saved movie to {fallback}")
        except Exception as e:
            print("render_movie: failed to save movie:", e)
    plt.close(fig)

def log_schedule(total_steps, n_frames):
    # indices in [1..total_steps] roughly logarithmically spaced
    xs = np.geomspace(1, total_steps, n_frames).astype(int)
    return np.unique(xs)

# helper: moving average
def moving_average(x, w):
    if w <= 1:
        return x
    arr = np.asarray(x, dtype=float)
    pad = w//2
    padded = np.pad(arr, (pad, pad), mode='edge')
    kernel = np.ones(w)/w
    return np.convolve(padded, kernel, mode='valid')

# helper: count non-bonded nearest-neighbour contacts for a given chain snapshot
def count_contacts_from_chain(chain, occ=None):
    """
    Return number of non-bonded nearest-neighbour contacts (each contact counted once).
    chain: list/array of backbone coordinates (Vec)
    occ: optional set of occupied coordinates (if you have grafts include them)
    """
    index_map = {tuple(r): i for i, r in enumerate(chain)}
    contacts = 0
    seen = set()
    N = len(chain)
    for i, r in enumerate(chain):
        for v in NN_VECS:
            nbr = add(r, v)
            j = index_map.get(nbr, None)
            if j is None:
                continue
            # exclude bonded neighbours
            if j in (i-1, i+1):
                continue
            a = (i, j) if i < j else (j, i)
            if a in seen:
                continue
            seen.add(a)
            contacts += 1
    return contacts

# ---------- epsilon function factory (type- and T-dependent) ----------
def make_eps_func(eps_params):
    """
    eps_params: dict mapping (type_i,type_j) -> (base, alpha)
    returns eps_fn(type_i, type_j, T) -> float (positive means attraction magnitude)
    """
    def eps_fn(ti, tj, T):
        a, b = eps_params[(ti, tj)]
        val = a + b * T
        return val
    return eps_fn

# ---------- energy counting (backbone + grafts) ----------
def energy(chain:List[Vec],
           side_grafts:Dict[int, Vec],
           types:List[str],
           T:float,
           eps_fn) -> float:
    """
    Return total energy (negative for attractions).
    """
    # build position -> id mapping; id is ('b',i) or ('s',i)
    pos_to_id = {}
    for i, r in enumerate(chain):
        pos_to_id[r] = ('b', i)
    for i, rg in side_grafts.items():
        if rg in pos_to_id:
            # this should never happen in a valid conformation
            raise RuntimeError(f"collision detected: graft at {rg} overlaps existing bead")
        pos_to_id[rg] = ('s', i)

    E = 0.0
    for pos, id_ in list(pos_to_id.items()):
        if id_[0] == 'b':
            i = id_[1]
            t_i = types[i]
            bonded = set()
            if i > 0:
                bonded.add(('b', i-1))
            if i < len(chain)-1:
                bonded.add(('b', i+1))
            if i in side_grafts:
                bonded.add(('s', i))
        else:
            i = id_[1]
            t_i = types[i]
            bonded = {('b', i)}

        for v in NN_VECS:
            nbr = add(pos, v)
            other = pos_to_id.get(nbr, None)
            if other is None:
                continue
            if other in bonded:
                continue
            if other[0] == 'b':
                j = other[1]
                t_j = types[j]
            else:
                j = other[1]; t_j = types[j]
            E -= eps_fn(t_i, t_j, T)
    return 0.5 * E  # contacts counted twice

# ---------- Rg unchanged ----------
def radius_of_gyration(chain:List[Vec]) -> float:
    r = np.array(chain, dtype=float)
    com = r.mean(axis=0)
    return math.sqrt(((r - com)**2).sum(axis=1).mean())

# ---------- Moves (carry grafts) ----------
def attempt_pivot(chain:List[Vec], occ:set, side_grafts:Dict[int,Vec]):
    n = len(chain)
    if n <= 3:
        return False, chain, occ, side_grafts
    i = random.randrange(1, n-1)
    head = chain[:i+1]
    tail = chain[i+1:]
    rot = random.choice(ROTATIONS)
    pivot = chain[i]

    new_occ = set(head)
    for parent in range(0, i+1):
        if parent in side_grafts:
            new_occ.add(side_grafts[parent])

    new_chain = list(head)
    tail_idx = list(range(i+1, n))
    for old_pos in tail:
        dr = sub(old_pos, pivot)
        new_pos = add(pivot, rot(dr))
        if new_pos in new_occ:
            return False, chain, occ, side_grafts
        new_chain.append(new_pos)
        new_occ.add(new_pos)

    new_side = dict(side_grafts)
    for parent in tail_idx:
        if parent in side_grafts:
            old_parent_pos = chain[parent]
            old_graft_pos  = side_grafts[parent]
            rel = sub(old_graft_pos, old_parent_pos)
            new_parent_pos = new_chain[parent]
            new_rel = rot(rel)
            new_graft_pos = add(new_parent_pos, new_rel)
            if new_graft_pos in new_occ:
                return False, chain, occ, side_grafts
            new_side[parent] = new_graft_pos
            new_occ.add(new_graft_pos)

    return True, new_chain, new_occ, new_side

def attempt_crankshaft(chain:List[Vec], occ:set, side_grafts:Dict[int,Vec]):
    n = len(chain)
    if n <= 3:
        return False, chain, occ, side_grafts
    i = random.randrange(1, n-1)
    a, b, c = chain[i-1], chain[i], chain[i+1]
    u1 = sub(b, a)
    u2 = sub(c, b)
    if u1 not in NN_VECS or u2 not in NN_VECS:
        return False, chain, occ, side_grafts
    if u1 == u2 or u1 == (-u2[0], -u2[1], -u2[2]):
        return False, chain, occ, side_grafts
    if (u1[0]*u2[0] + u1[1]*u2[1] + u1[2]*u2[2]) != 0:
        return False, chain, occ, side_grafts

    b_new = add(a, u2)
    occ_excluding_b = set(occ)
    occ_excluding_b.discard(b)
    old_g = side_grafts.get(i, None)
    if old_g is not None:
        occ_excluding_b.discard(old_g)

    if b_new in occ_excluding_b:
        return False, chain, occ, side_grafts
    if sub(b_new, a) not in NN_VECS or sub(c, b_new) not in NN_VECS:
        return False, chain, occ, side_grafts

    new_side = dict(side_grafts)
    if i in side_grafts:
        old_g = side_grafts[i]
        delta = sub(b_new, b)
        new_g = add(old_g, delta)
        if new_g in occ_excluding_b:
            return False, chain, occ, side_grafts
        new_side[i] = new_g

    new_chain = chain.copy()
    new_chain[i] = b_new
    new_occ = (occ - {b})
    if i in side_grafts:
        new_occ = (new_occ - { side_grafts[i] })
        new_occ.add(new_side[i])
    new_occ.add(b_new)
    return True, new_chain, new_occ, new_side

def attempt_end_move(chain:List[Vec], occ:set, side_grafts:Dict[int,Vec]):
    n = len(chain)
    if n <= 1:
        return False, chain, occ, side_grafts
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
        return False, chain, occ, side_grafts
    r_new = random.choice(candidates)
    new_chain = chain.copy()
    old_end_pos = chain[end]
    new_chain[end] = r_new
    new_occ = (occ - { old_end_pos }) | { r_new }
    new_side = side_grafts.copy()
    if end in side_grafts:
        old_g = side_grafts[end]
        delta = sub(r_new, old_end_pos)
        new_g = add(old_g, delta)
        if new_g in new_occ:
            return False, chain, occ, side_grafts
        new_side[end] = new_g
        new_occ = (new_occ - { old_g }) | { new_g }
    return True, new_chain, new_occ, new_side

MOVE_FUNCS = [attempt_pivot, attempt_crankshaft, attempt_end_move]

# ---------- sampling helper ----------
def sample_trajectory(chain, side_grafts, include_grafts: bool = False):
    """Return snapshot array: backbone coords (N,3). If include_grafts=True, return backbone then graft coords."""
    backbone = np.asarray(chain, dtype=float)
    grafts = []
    for i in sorted(side_grafts.keys()):
        grafts.append(np.asarray(side_grafts[i], dtype=float))
    if include_grafts and grafts:
        return np.vstack([backbone, np.vstack(grafts)])
    else:
        return backbone

# ---------- sanity check helper (optional) ----------
def verify_no_overlap(chain: List[Vec], side_grafts: Dict[int, Vec]) -> None:
    pos_map = {}
    for i, p in enumerate(chain):
        if p in pos_map:
            raise RuntimeError(f"Overlap detected: backbone {i} at {p} collides with {pos_map[p]}")
        pos_map[p] = ('b', i)
    for parent, g in side_grafts.items():
        if g in pos_map:
            raise RuntimeError(f"Overlap detected: graft of parent {parent} at {g} collides with {pos_map[g]}")
        pos_map[g] = ('s', parent)

# ---------- main / CLI ----------
def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--N',        type=int,   default=44,    help='backbone chain length')
    ap.add_argument('--steps',    type=int,   default=500000, help='MC steps')
    ap.add_argument('--T',        type=float, default=2.0,   help='temperature (k_B=1)')
    ap.add_argument('--seed',     type=int,   default=None,  help='RNG seed')
    ap.add_argument('--fH',       type=float, default=0.4,   help='fraction hydrophobic backbone')
    ap.add_argument('--f_graft',  type=float, default=0.2,   help='fraction of backbone sites with one graft')
    ap.add_argument('--eps_HH_base', type=float, default=-2.0)
    ap.add_argument('--alpha_HH',    type=float, default=2.0)
    ap.add_argument('--eps_HP_base', type=float, default=0.1)
    ap.add_argument('--alpha_HP',    type=float, default=0.0)
    ap.add_argument('--eps_PP_base', type=float, default=0.1)
    ap.add_argument('--alpha_PP',    type=float, default=0.0)
    args = ap.parse_args()

    if args.seed is not None:
        random.seed(args.seed); np.random.seed(args.seed)

    eps_params = {
        ('H','H'): (args.eps_HH_base, args.alpha_HH),
        ('H','P'): (args.eps_HP_base, args.alpha_HP),
        ('P','H'): (args.eps_HP_base, args.alpha_HP),
        ('P','P'): (args.eps_PP_base, args.alpha_PP),
    }
    eps_fn = make_eps_func(eps_params)

    chain: List[Vec] = [(i,0,0) for i in range(args.N)]
    occ = set(chain)

    types = ['H' if random.random() < args.fH else 'P' for _ in range(args.N)]

    n_grafts = int(round(args.f_graft * args.N))
    candidate_parents = list(range(1, args.N-1))
    random.shuffle(candidate_parents)
    graft_parents = sorted(candidate_parents[:n_grafts])

    side_grafts: Dict[int, Vec] = {}
    for p in graft_parents:
        base = chain[p]
        placed = False
        for v in [(0,0,1),(0,0,-1),(0,1,0),(0,-1,0),(1,0,0),(-1,0,0)]:
            candidate = add(base, v)
            if candidate not in occ:
                side_grafts[p] = candidate
                occ.add(candidate)
                placed = True
                break
        if not placed:
            print(f"Warning: couldn't place graft at parent {p} (skipping).")

    # optional sanity check (uncomment to enable)
    # verify_no_overlap(chain, side_grafts)

    E = energy(chain, side_grafts, types, args.T, eps_fn)
    beta = 1.0 / args.T

    print(f"N={args.N}, grafts={len(side_grafts)}, H={types.count('H')}, T={args.T}")
    acc = 0
    snapshots = []
    sample_every = 1000
    dt_frame     = sample_every

    record_interval = max(1, args.steps // 2000)
    saved_steps, E_traj2, Rg_traj2 = [], [], []
    frames_to_save = set(log_schedule(args.steps, 2000))
    frames = []
    frame_steps = []

    for step in range(1, args.steps+1):
        if step % record_interval == 0:
            saved_steps.append(step)
            E_traj2.append(E)
            Rg_traj2.append(radius_of_gyration(chain))

        move = random.choice(MOVE_FUNCS)
        ok, chain_new, occ_new, side_new = move(chain, occ, side_grafts)
        if not ok:
            continue
        dE = energy(chain_new, side_new, types, args.T, eps_fn) - E
        if dE <= 0 or random.random() < math.exp(-beta*dE):
            chain, occ, side_grafts, E = chain_new, occ_new, side_new, E + dE
            acc += 1

        if step % sample_every == 0:
            # store a full snapshot (backbone + grafts) for analysis
            snapshots.append(sample_trajectory(chain, side_grafts, include_grafts=True))

        if step in frames_to_save:
            # store backbone-only frames for the movie (keeps render_movie simple)
            frames.append(sample_trajectory(chain, side_grafts, include_grafts=False))
            frame_steps.append(step)

    print(f"Acceptance ratio: {acc/args.steps:.3f}")
    print(f"Final E = {E:.3f}, Rg = {radius_of_gyration(chain):.3f}")

    # 1) movie from log-sampled frames
    if len(frames) > 0:
        print(f"Rendering movie from {len(frames)} log-sampled frames -> polymer_logmovie.mp4")
        try:
            render_movie(frames, out="polymer_logmovie.mp4", fps=24, stride=1)
        except Exception as e:
            print("Movie render failed (ffmpeg?) — trying GIF fallback.")
            try:
                render_movie(frames, out="polymer_logmovie.gif", fps=12, stride=1)
            except Exception as e2:
                print("Both mp4/gif save failed:", e2)
    else:
        print("No log-sampled frames found in `frames`. Increase the log_schedule frames or check frames_to_save.")

    # 2) Rg vs MC steps plot
    if len(saved_steps) and len(Rg_traj2):
        steps_arr = np.array(saved_steps)
        Rg_arr    = np.array(Rg_traj2)
        Rg_smooth = moving_average(Rg_arr, w=21)

        plt.figure(figsize=(7,3.5))
        plt.plot(steps_arr, Rg_arr, '.', alpha=0.35, label='sampled Rg')
        plt.plot(steps_arr, Rg_smooth, '-', color='k', lw=1.5, label='moving avg')

        if len(frames) > 0 and len(frame_steps) == len(frames):
            for s in frame_steps:
                plt.axvline(s, color='tab:gray', alpha=0.2, lw=1)

        plt.xlabel('MC step')
        plt.ylabel(r'$R_g$')
        plt.legend()
        plt.tight_layout()
        plt.savefig("Rg_vs_steps.png", dpi=160)
        plt.close()
    else:
        print("No saved_steps / Rg_traj2 found — ensure you recorded values in the run (record_interval etc.).")

    # 3) Energy time series
    if len(saved_steps) and len(E_traj2):
        plt.figure(figsize=(7,3))
        plt.plot(saved_steps, E_traj2, '-', alpha=0.7)
        plt.xlabel('MC step'); plt.ylabel('Energy')
        plt.tight_layout()
        plt.savefig("Energy_vs_steps.png", dpi=160)
        plt.close()

if __name__ == "__main__":
    main()
