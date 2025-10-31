# polymer_hp_grafts_with_dimer_sidechains_instrumented.py
from __future__ import annotations
import argparse, random, math
from matplotlib import animation
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from typing import Tuple, List, Dict, Set

Vec = Tuple[int, int, int]

# ---------- basic geometry helpers ----------
NN_VECS: List[Vec] = [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]
def add(a:Vec, b:Vec) -> Vec: return (a[0]+b[0], a[1]+b[1], a[2]+b[2])
def sub(a:Vec, b:Vec) -> Vec: return (a[0]-b[0], a[1]-b[1], a[2]-b[2])

# 90° rotations about origin (right-handed)
ROT_X = lambda v: ( v[0],  -v[2],  v[1])
ROT_Y = lambda v: ( v[2],   v[1], -v[0])
ROT_Z = lambda v: (-v[1],   v[0],  v[2])
ROTATIONS = [ROT_X, ROT_Y, ROT_Z]

# ---------- visualization utilities (unchanged) ----------
def render_movie(snaps, out="polymer.mp4", fps=30, stride=1):
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
    xs = np.geomspace(1, total_steps, n_frames).astype(int)
    return np.unique(xs)

def moving_average(x, w):
    if w <= 1:
        return x
    arr = np.asarray(x, dtype=float)
    pad = w//2
    padded = np.pad(arr, (pad, pad), mode='edge')
    kernel = np.ones(w)/w
    return np.convolve(padded, kernel, mode='valid')

# ---------- contact counting (backbone only) ----------
def count_contacts_from_chain(chain:List[Vec], occ: Set[Vec] = None):
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
            if j in (i-1, i+1):
                continue
            a = (i, j) if i < j else (j, i)
            if a in seen:
                continue
            seen.add(a)
            contacts += 1
    return contacts

# ---------- epsilon function factory (sigmoid) ----------
def make_eps_sigmoid(eps_params):
    """
    eps_params[(ti,tj)] = (eps_low, eps_high, Tmid, width)
    returns eps_fn(ti,tj,T)
    """
    def eps_fn(ti, tj, T):
        eps_low, eps_high, Tmid, width = eps_params[(ti,tj)]
        # protect against width=0
        if width == 0:
            s = 1.0 if T > Tmid else 0.0
        else:
            s = 0.5 * (1.0 + math.tanh((T - Tmid)/width))
        return eps_low*(1-s) + eps_high*s
    return eps_fn

# ---------- energy counting (backbone + multi-unit grafts) ----------
def energy(chain:List[Vec],
           side_grafts:Dict[int, List[Vec]],
           types:List[str],
           graft_types:Dict[int, List[str]],
           T:float,
           eps_fn) -> float:
    """
    Return total energy (negative for attractions): this function subtracts eps_fn from E
    for each non-bonded nearest-neighbour pair (so eps>0 -> attraction lowers energy).
    side_grafts: parent_index -> list of bead positions (innermost first).
    graft_types: parent_index -> list of types (same length as side_grafts[parent]).
    """
    pos_to_id = {}
    for i, r in enumerate(chain):
        pos_to_id[r] = ('b', i)
    for parent, glist in side_grafts.items():
        for k, pos in enumerate(glist):
            pos_to_id[pos] = ('s', parent, k)

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
                bonded.add(('s', i, 0))
        else:
            _, parent, k = id_
            t_i = graft_types[parent][k]
            bonded = set()
            if k == 0:
                bonded.add(('b', parent))
                if len(side_grafts[parent]) > 1:
                    bonded.add(('s', parent, 1))
            else:
                bonded.add(('s', parent, k-1))
                if k+1 < len(side_grafts[parent]):
                    bonded.add(('s', parent, k+1))

        for v in NN_VECS:
            nbr = add(pos, v)
            other = pos_to_id.get(nbr, None)
            if other is None:
                continue
            if other in bonded:
                continue
            # determine neighbor type
            if other[0] == 'b':
                t_j = types[ other[1] ]
            else:
                t_j = graft_types[ other[1] ][ other[2] ]
            E -= eps_fn(t_i, t_j, T)
    return 0.5 * E  # pairs counted twice

# ---------- radius of gyration (backbone and full system) ----------
def radius_of_gyration(chain:List[Vec]) -> float:
    r = np.array(chain, dtype=float)
    com = r.mean(axis=0)
    return math.sqrt(((r - com)**2).sum(axis=1).mean())

def radius_of_gyration_full(chain:List[Vec], side_grafts:Dict[int,List[Vec]]) -> float:
    arrs = [np.asarray(chain, dtype=float)]
    for p in sorted(side_grafts.keys()):
        for pos in side_grafts[p]:
            arrs.append(np.asarray(pos, dtype=float))
    r = np.vstack(arrs)
    com = r.mean(axis=0)
    return math.sqrt(((r - com)**2).sum(axis=1).mean())

# ---------- H-H contact counter including grafts ----------
def count_HH_contacts(chain, side_grafts, types, graft_types):
    pos_to_type = {}
    for i, p in enumerate(chain):
        pos_to_type[p] = types[i]
    for parent, glist in side_grafts.items():
        for k, p in enumerate(glist):
            pos_to_type[p] = graft_types[parent][k]

    seen = set()
    hh_contacts = 0
    for pos, t in pos_to_type.items():
        if t != 'H':
            continue
        for v in NN_VECS:
            nbr = add(pos, v)
            t2 = pos_to_type.get(nbr, None)
            if t2 != 'H':
                continue
            a = (pos, nbr) if pos < nbr else (nbr, pos)
            if a in seen:
                continue
            # exclude bonded neighbours when both are backbone adjacent indices is ignored here
            seen.add(a)
            hh_contacts += 1
    return hh_contacts

def eps_summary(eps_fn, Ts=[0.5, 1.0, 2.0, 3.0, 4.0]):
    print("eps summary:")
    for T in Ts:
        print(f"  eps_HH(T={T}) = {eps_fn('H','H',T):.4f}")

# ---------- Moves (carry multi-unit grafts) ----------
def attempt_pivot(chain:List[Vec], occ:set, side_grafts:Dict[int,List[Vec]]):
    n = len(chain)
    if n <= 3:
        return False, chain, occ, side_grafts
    i = random.randrange(1, n-1)
    head = chain[:i+1]
    tail = chain[i+1:]
    rot = random.choice(ROTATIONS)
    pivot = chain[i]

    new_occ = set(head)
    # include grafts attached to head parents
    for parent in range(0, i+1):
        if parent in side_grafts:
            for gpos in side_grafts[parent]:
                new_occ.add(gpos)

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
    # update grafts for parents in tail (they moved/rotated rigidly with parent)
    for parent in tail_idx:
        if parent in side_grafts:
            old_parent_pos = chain[parent]
            old_glist = side_grafts[parent]
            new_parent_pos = new_chain[parent]
            new_glist = []
            for old_graft_pos in old_glist:
                rel = sub(old_graft_pos, old_parent_pos)
                new_rel = rot(rel)
                new_graft_pos = add(new_parent_pos, new_rel)
                if new_graft_pos in new_occ:
                    return False, chain, occ, side_grafts
                new_glist.append(new_graft_pos)
                new_occ.add(new_graft_pos)
            new_side[parent] = new_glist

    return True, new_chain, new_occ, new_side

def attempt_crankshaft(chain:List[Vec], occ:set, side_grafts:Dict[int,List[Vec]]):
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
        for pos in old_g:
            occ_excluding_b.discard(pos)

    if b_new in occ_excluding_b:
        return False, chain, occ, side_grafts
    if sub(b_new, a) not in NN_VECS or sub(c, b_new) not in NN_VECS:
        return False, chain, occ, side_grafts

    new_side = dict(side_grafts)
    if i in side_grafts:
        old_glist = side_grafts[i]
        delta = sub(b_new, b)
        new_glist = []
        for old_g in old_glist:
            new_g = add(old_g, delta)
            if new_g in occ_excluding_b:
                return False, chain, occ, side_grafts
            new_glist.append(new_g)
        new_side[i] = new_glist

    new_chain = chain.copy()
    new_chain[i] = b_new
    new_occ = (occ - {b})
    if i in side_grafts:
        for pos in side_grafts[i]:
            new_occ.discard(pos)
        for pos in new_side[i]:
            new_occ.add(pos)
    new_occ.add(b_new)
    return True, new_chain, new_occ, new_side

def attempt_end_move(chain:List[Vec], occ:set, side_grafts:Dict[int,List[Vec]]):
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
    new_side = {k:list(v) for k,v in side_grafts.items()}
    if end in side_grafts:
        old_glist = side_grafts[end]
        delta = sub(r_new, old_end_pos)
        new_glist = []
        for old_g in old_glist:
            new_g = add(old_g, delta)
            if new_g in new_occ:
                return False, chain, occ, side_grafts
            new_glist.append(new_g)
        # update occ: remove old graft positions, add new ones
        for old_g in old_glist:
            new_occ.discard(old_g)
        for new_g in new_glist:
            new_occ.add(new_g)
        new_side[end] = new_glist
    return True, new_chain, new_occ, new_side

# ---------- Local graft move: move/rotate outer bead of a graft chain ----------
def attempt_graft_local(chain:List[Vec], occ:set, side_grafts:Dict[int,List[Vec]]):
    if not side_grafts:
        return False, chain, occ, side_grafts
    parent = random.choice(list(side_grafts.keys()))
    glist = side_grafts[parent]
    # we will try to move the OUTERMOST graft bead (index = len(glist)-1)
    k = len(glist) - 1
    if k < 0:
        return False, chain, occ, side_grafts
    neighbor_pos = glist[k-1] if k > 0 else chain[parent]
    candidates = []
    for v in NN_VECS:
        new_pos = add(neighbor_pos, v)
        # must stay unoccupied (or be the current bead itself)
        if new_pos in occ and new_pos not in glist:
            continue
        # ensure bond length to neighbor remains 1
        if sub(new_pos, neighbor_pos) not in NN_VECS:
            continue
        # also avoid overlapping with parent or other graft beads in this chain (unless it's itself)
        if new_pos == chain[parent]:
            continue
        candidates.append(new_pos)
    if not candidates:
        return False, chain, occ, side_grafts
    new_pos = random.choice(candidates)
    new_side = {k:list(v) for k,v in side_grafts.items()}
    new_glist = list(glist)
    old_pos = new_glist[k]
    if new_pos == old_pos:
        return False, chain, occ, side_grafts
    # quick collision check: other occupied positions excluding the moving bead
    occ_excluding = set(occ)
    occ_excluding.discard(old_pos)
    if new_pos in occ_excluding:
        return False, chain, occ, side_grafts
    new_glist[k] = new_pos
    new_side[parent] = new_glist
    new_occ = set(occ)
    new_occ.discard(old_pos)
    new_occ.add(new_pos)
    return True, chain, new_occ, new_side

MOVE_FUNCS = [attempt_pivot, attempt_crankshaft, attempt_end_move, attempt_graft_local]

# ---------- sampling helper ----------
def sample_trajectory(chain, side_grafts, include_grafts: bool = False):
    backbone = np.asarray(chain, dtype=float)
    grafts = []
    for i in sorted(side_grafts.keys()):
        for pos in side_grafts[i]:
            grafts.append(np.asarray(pos, dtype=float))
    if include_grafts and grafts:
        return np.vstack([backbone, np.vstack(grafts)])
    else:
        return backbone

# ---------- sanity check helper ----------
def verify_no_overlap(chain: List[Vec], side_grafts: Dict[int, List[Vec]]) -> None:
    pos_map = {}
    for i, p in enumerate(chain):
        if p in pos_map:
            raise RuntimeError(f"Overlap detected: backbone {i} at {p} collides with {pos_map[p]}")
        pos_map[p] = ('b', i)
    for parent, glist in side_grafts.items():
        for g in glist:
            if g in pos_map:
                raise RuntimeError(f"Overlap detected: graft of parent {parent} at {g} collides with {pos_map[g]}")
            pos_map[g] = ('s', parent)

# ---------- main / CLI ----------
def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--N',        type=int,   default=44,    help='backbone chain length')
    ap.add_argument('--steps',    type=int,   default=50000, help='MC steps')
    ap.add_argument('--T',        type=float, default=3,   help='temperature (k_B=1)')
    ap.add_argument('--seed',     type=int,   default=42,  help='RNG seed')
    ap.add_argument('--fH',       type=float, default=0.6,   help='fraction hydrophobic backbone')
    ap.add_argument('--f_graft',  type=float, default=0.5,   help='fraction of backbone sites with one graft (excluding ends)')
    ap.add_argument('--graft_len',type=int,   default=2,     help='length of side graft chains (>=1)')
    # NOTE: eps params are now handled below (sigmoid)
    args = ap.parse_args()

    if args.seed is not None:
        random.seed(args.seed); np.random.seed(args.seed)

    # ---- stronger defaults for quick testing ----
    eps_params = {
        ('H','H'): (0.05, 3.5,  2.0, 0.2),   # eps_low, eps_high, Tmid, width
        ('H','P'): (0.0,  0.0,  2.0, 0.1),
        ('P','H'): (0.0,  0.0,  2.0, 0.1),
        ('P','P'): (0.0,  0.0,  2.0, 0.1),
    }
    eps_fn = make_eps_sigmoid(eps_params)

    # backbone initial straight line
    chain: List[Vec] = [(i,0,0) for i in range(args.N)]
    occ = set(chain)

    # backbone monomer types
    types = ['H' if random.random() < args.fH else 'P' for _ in range(args.N)]

    # choose graft parent indices (avoid ends)
    n_grafts = int(round(args.f_graft * args.N))
    candidate_parents = list(range(1, args.N-1))
    random.shuffle(candidate_parents)
    graft_parents = sorted(candidate_parents[:n_grafts])

    # side_grafts: parent -> list of positions (innermost first)
    side_grafts: Dict[int, List[Vec]] = {}
    graft_types: Dict[int, List[str]] = {}
    GRAFT_LEN = max(1, int(args.graft_len))

    for p in graft_parents:
        base = chain[p]
        placed = False
        # try each direction for the first bead; attempt to place the rest either colinear or orthogonal
        for v in NN_VECS:
            pos1 = add(base, v)
            if pos1 in occ:
                continue
            glist = [pos1]
            occ_ok = True
            # attempt to place further beads continuing in direction v; if blocked, try perpendicular options
            curr = pos1
            for k in range(1, GRAFT_LEN):
                cand = add(curr, v)   # prefer straight continuation
                if cand in occ:
                    # try perpendicular options around curr
                    found = False
                    for v2 in NN_VECS:
                        if v2 == tuple(-x for x in v) or v2 == v:
                            continue
                        cand2 = add(curr, v2)
                        if cand2 not in occ:
                            cand = cand2
                            found = True
                            break
                    if not found:
                        occ_ok = False
                        break
                glist.append(cand)
                curr = cand
            if not occ_ok:
                continue
            # success: assign graft positions and types (example: inner hydrophilic 'P', outer hydrophobic 'H')
            side_grafts[p] = glist
            # choose graft monomer types: inner = P, outer = H for PNIPAM-like pattern
            gt = []
            for k in range(GRAFT_LEN):
                if k == 0:
                    gt.append('P')
                elif k == GRAFT_LEN-1:
                    gt.append('H')
                else:
                    gt.append('P')
            graft_types[p] = gt
            for pos in glist:
                occ.add(pos)
            placed = True
            break
        if not placed:
            print(f"Warning: couldn't place graft (len={GRAFT_LEN}) at parent {p} (skipping).")

    # optional check
    # verify_no_overlap(chain, side_grafts)

    # initial diagnostics
    print(f"Initial T = {args.T}")
    eps_summary(eps_fn, Ts=[0.5,1.0,2.0,3.0,4.0])
    hh_init = count_HH_contacts(chain, side_grafts, types, graft_types)
    print(f"Initial HH nonbonded contacts = {hh_init}")
    print(f"Backbone H fraction = {types.count('H')}/{len(types)}; grafts placed = {len(side_grafts)}")

    E = energy(chain, side_grafts, types, graft_types, args.T, eps_fn)
    beta = 1.0 / args.T

    print(f"N={args.N}, grafts={len(side_grafts)}, H(backbone)={types.count('H')}, T={args.T}")
    acc = 0
    snapshots = []
    sample_every = 1000

    record_interval = max(1, args.steps // 2000)
    saved_steps, E_traj2, Rg_traj2 = [], [], []
    Rg_full_traj, HH_traj = [], []
    frames_to_save = set(log_schedule(args.steps, 2000))
    frames = []
    frame_steps = []

    for step in range(1, args.steps+1):
        if step % record_interval == 0:
            saved_steps.append(step)
            E_traj2.append(E)
            Rg_back = radius_of_gyration(chain)
            Rg_full = radius_of_gyration_full(chain, side_grafts)
            Rg_traj2.append(Rg_back)
            Rg_full_traj.append(Rg_full)
            HH_traj.append(count_HH_contacts(chain, side_grafts, types, graft_types))

        move = random.choice(MOVE_FUNCS)
        ok, chain_new, occ_new, side_new = move(chain, occ, side_grafts)
        if not ok:
            continue
        dE = energy(chain_new, side_new, types, graft_types, args.T, eps_fn) - E
        if dE <= 0 or random.random() < math.exp(-beta*dE):
            chain, occ, side_grafts, E = chain_new, occ_new, side_new, E + dE
            acc += 1

        if step % sample_every == 0:
            snapshots.append(sample_trajectory(chain, side_grafts, include_grafts=True))

        if step in frames_to_save:
            frames.append(sample_trajectory(chain, side_grafts, include_grafts=False))
            frame_steps.append(step)

    hh_final = count_HH_contacts(chain, side_grafts, types, graft_types)

    # Compute averages over last 30% of recorded samples (if available)
    avg_msg = ""
    if len(saved_steps) > 0:
        n = len(saved_steps)
        start_idx = int(math.floor(n * 0.7))
        if start_idx < 0:
            start_idx = 0
        # slice arrays
        E_slice = np.array(E_traj2[start_idx:]) if len(E_traj2) > start_idx else np.array([E])
        Rg_back_slice = np.array(Rg_traj2[start_idx:]) if len(Rg_traj2) > start_idx else np.array([radius_of_gyration(chain)])
        Rg_full_slice = np.array(Rg_full_traj[start_idx:]) if len(Rg_full_traj) > start_idx else np.array([radius_of_gyration_full(chain, side_grafts)])
        HH_slice = np.array(HH_traj[start_idx:]) if len(HH_traj) > start_idx else np.array([hh_final])

        E_mean, E_std = float(np.nanmean(E_slice)), float(np.nanstd(E_slice, ddof=0))
        Rg_back_mean, Rg_back_std = float(np.nanmean(Rg_back_slice)), float(np.nanstd(Rg_back_slice, ddof=0))
        Rg_full_mean, Rg_full_std = float(np.nanmean(Rg_full_slice)), float(np.nanstd(Rg_full_slice, ddof=0))
        HH_mean, HH_std = float(np.nanmean(HH_slice)), float(np.nanstd(HH_slice, ddof=0))

        avg_msg += f"\nAverages over last 30% of recorded samples (indices {start_idx}..{n-1}):\n"
        avg_msg += f"  E_mean = {E_mean:.3f} ± {E_std:.3f}\n"
        avg_msg += f"  Rg_back_mean = {Rg_back_mean:.3f} ± {Rg_back_std:.3f}\n"
        avg_msg += f"  Rg_full_mean = {Rg_full_mean:.3f} ± {Rg_full_std:.3f}\n"
        avg_msg += f"  HH_mean = {HH_mean:.3f} ± {HH_std:.3f}\n"
    else:
        avg_msg += "\nNo recorded samples to average over; using instantaneous final values.\n"

    print(f"\nRun summary:")
    eps_summary(eps_fn, Ts=[args.T])
    print(f"Initial HH contacts = {hh_init}; Final HH contacts = {hh_final}")
    print(f"Acceptance ratio: {acc/args.steps:.3f}")
    # Print averaged values (replace single final Rg/E prints with averaged values as requested)
    if avg_msg:
        print(avg_msg)
    else:
        print(f"Final E = {E:.3f}, Rg_backbone = {radius_of_gyration(chain):.3f}, Rg_full = {radius_of_gyration_full(chain, side_grafts):.3f}")

    # save diagnostic time series plots
    if len(saved_steps):
        steps_arr = np.array(saved_steps)
        Rg_back_arr = np.array(Rg_traj2)
        Rg_full_arr = np.array(Rg_full_traj)
        HH_arr = np.array(HH_traj)

        plt.figure(figsize=(6,3))
        plt.plot(steps_arr, Rg_back_arr, '.', alpha=0.5, label='R_g (backbone)')
        plt.plot(steps_arr, Rg_full_arr, '.', alpha=0.5, label='R_g (full)')
        plt.xlabel('MC step'); plt.ylabel('R_g')
        plt.legend(); plt.tight_layout()
        plt.savefig("Rg_vs_steps_full.png", dpi=160)
        plt.close()
        print("Saved Rg_vs_steps_full.png")

        plt.figure(figsize=(6,3))
        plt.plot(steps_arr, HH_arr, '-', alpha=0.7)
        plt.xlabel('MC step'); plt.ylabel('HH nonbonded contacts')
        plt.tight_layout()
        plt.savefig("HH_contacts_vs_steps.png", dpi=160)
        plt.close()
        print("Saved HH_contacts_vs_steps.png")

    # 1) movie from log-sampled frames
    if len(frames) > 0:
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

    # Energy time series (existing plot)
    if len(saved_steps) and len(E_traj2):
        plt.figure(figsize=(7,3))
        plt.plot(saved_steps, E_traj2, '-', alpha=0.7)
        plt.xlabel('MC step'); plt.ylabel('Energy')
        plt.tight_layout()
        plt.savefig("Energy_vs_steps.png", dpi=160)
        plt.close()
        print("Saved Energy_vs_steps.png")

if __name__ == "__main__":
    main()
