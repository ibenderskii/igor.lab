# polymer_hp_grafts.py
from __future__ import annotations
import argparse, random, math
import numpy as np
import matplotlib.pyplot as plt
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

# ---------- epsilon function factory (type- and T-dependent) ----------
def make_eps_func(eps_params):
    """
    eps_params: dict mapping (type_i,type_j) -> (base, alpha)
    returns eps_fn(type_i, type_j, T) -> positive float (attraction magnitude)
    """
    def eps_fn(ti, tj, T):
        a, b = eps_params[(ti, tj)]
        val = a + b * T
        # allow small negatives if user wants repulsion: keep as-is
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
    chain: backbone coords indexed 0..N-1
    side_grafts: mapping parent_index -> graft_coord
    types: backbone types list length N (e.g. 'H' or 'P'); graft inherits parent's type
    eps_fn: function(ti,tj,T) -> float (positive means attraction magnitude)
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
    # iterate beads
    for pos, id_ in list(pos_to_id.items()):
        if id_[0] == 'b':
            i = id_[1]
            t_i = types[i]
            # bonded backbone neighbors
            bonded = set()
            if i > 0:
                bonded.add(('b', i-1))
            if i < len(chain)-1:
                bonded.add(('b', i+1))
            # backbone-graft bonded neighbor if present
            if i in side_grafts:
                bonded.add(('s', i))
        else:
            # graft bead
            i = id_[1]  # parent index
            t_i = types[i]
            # bonded only to parent backbone
            bonded = {('b', i)}

        # check six NN positions
        for v in NN_VECS:
            nbr = add(pos, v)
            other = pos_to_id.get(nbr, None)
            if other is None:
                continue
            # exclude bonded pairs
            if other in bonded:
                continue
            # compute interaction
            if other[0] == 'b':
                j = other[1]
                t_j = types[j]
            else:
                # graft -> its parent j
                j = other[1]; t_j = types[j]
            # subtract eps (attractive lowers energy)
            E -= eps_fn(t_i, t_j, T)
    return 0.5 * E  # contacts counted twice

# ---------- Rg unchanged ----------
def radius_of_gyration(chain:List[Vec]) -> float:
    r = np.array(chain, dtype=float)
    com = r.mean(axis=0)
    return math.sqrt(((r - com)**2).sum(axis=1).mean())

# ---------- Moves: now carry grafts through moves ----------
def attempt_pivot(chain:List[Vec], occ:set, side_grafts:Dict[int,Vec]):
    n = len(chain)
    if n <= 3:
        return False, chain, occ, side_grafts
    i = random.randrange(1, n-1)         # pivot monomer (not ends)
    head = chain[:i+1]
    tail = chain[i+1:]
    rot = random.choice(ROTATIONS)
    pivot = chain[i]

    # Build initial new_occ with head backbone positions AND any grafts attached to the head
    new_occ = set(head)
    for parent in range(0, i+1):
        if parent in side_grafts:
            new_occ.add(side_grafts[parent])

    new_chain = list(head)  # start with head

    # Rotate tail backbone positions; check collision against new_occ (which includes head grafts)
    tail_idx = list(range(i+1, n))
    for old_pos in tail:
        dr = sub(old_pos, pivot)
        new_pos = add(pivot, rot(dr))
        if new_pos in new_occ:
            return False, chain, occ, side_grafts
        new_chain.append(new_pos)
        new_occ.add(new_pos)

    # Now rotate grafts attached to tail parents and check collisions against new_occ (which has heads + rotated tail backbone)
    new_side = dict(side_grafts)  # copy
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

    # head grafts untouched (already included in new_occ)
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

    b_new = add(a, u2)  # proposed new middle
    # Check that b_new doesn't collide with beads that are not being vacated
    # We'll allow landing on the old b position (it will be vacated) but not on other beads.
    occ_excluding_b = set(occ)
    occ_excluding_b.discard(b)
    # also remove the old graft of the middle bead from the exclusion set (it will be moved too)
    old_g = side_grafts.get(i, None)
    if old_g is not None:
        occ_excluding_b.discard(old_g)

    if b_new in occ_excluding_b:
        return False, chain, occ, side_grafts
    if sub(b_new, a) not in NN_VECS or sub(c, b_new) not in NN_VECS:
        return False, chain, occ, side_grafts

    new_side = dict(side_grafts)
    # if middle had graft, propose new graft position (translate by delta)
    if i in side_grafts:
        old_g = side_grafts[i]
        delta = sub(b_new, b)
        new_g = add(old_g, delta)
        # check new_g against occupancy excluding the beads that are being removed (b and old_g)
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
    """Move an end bead (0 or n-1) to a neighboring empty site such that bond to anchor remains unit.
       If moved end has a graft, translate graft by same displacement."""
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
        # ensure new bond to anchor is unit
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
    # if end had graft, translate graft by delta
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
def sample_trajectory(chain, side_grafts, snap_list):
    """Append snapshot of backbone and grafts (array shape (N+ng,3)) for visualization or analysis."""
    # create ordered list: backbone then graft beads (in parent index order)
    backbone = np.asarray(chain, dtype=float)
    grafts = []
    graft_parents = []
    for i in sorted(side_grafts.keys()):
        grafts.append(np.asarray(side_grafts[i], dtype=float))
        graft_parents.append(i)
    if grafts:
        return np.vstack([backbone, np.vstack(grafts)])
    else:
        return backbone

# ---------- main / CLI ----------
def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--N',        type=int,   default=44,    help='backbone chain length')
    ap.add_argument('--steps',    type=int,   default=500000, help='MC steps')
    ap.add_argument('--T',        type=float, default=1.5,   help='temperature (k_B=1)')
    ap.add_argument('--seed',     type=int,   default=None,  help='RNG seed')
    ap.add_argument('--fH',       type=float, default=0.4,   help='fraction hydrophobic backbone')
    ap.add_argument('--f_graft',  type=float, default=0.2,   help='fraction of backbone sites with one graft')
    # eps params (simple linear-in-T)
    ap.add_argument('--eps_HH_base', type=float, default=0.0)
    ap.add_argument('--alpha_HH',    type=float, default=1.0)
    ap.add_argument('--eps_HP_base', type=float, default=0.0)
    ap.add_argument('--alpha_HP',    type=float, default=0.0)
    ap.add_argument('--eps_PP_base', type=float, default=0.0)
    ap.add_argument('--alpha_PP',    type=float, default=0.0)
    args = ap.parse_args()

    if args.seed is not None:
        random.seed(args.seed); np.random.seed(args.seed)

    # epsilon params dict
    eps_params = {
        ('H','H'): (args.eps_HH_base, args.alpha_HH),
        ('H','P'): (args.eps_HP_base, args.alpha_HP),
        ('P','H'): (args.eps_HP_base, args.alpha_HP),
        ('P','P'): (args.eps_PP_base, args.alpha_PP),
    }
    eps_fn = make_eps_func(eps_params)

    # initial straight backbone along x
    chain: List[Vec] = [(i,0,0) for i in range(args.N)]
    occ = set(chain)

    # assign backbone types
    types = ['H' if random.random() < args.fH else 'P' for _ in range(args.N)]

    # choose graft parents (avoid ends if you prefer)
    n_grafts = int(round(args.f_graft * args.N))
    candidate_parents = list(range(1, args.N-1))  # avoid grafting on terminal beads for simplicity
    random.shuffle(candidate_parents)
    graft_parents = sorted(candidate_parents[:n_grafts])

    # initial graft placement: put graft at parent + (0,0,1) or another free NN if occupied
    side_grafts: Dict[int, Vec] = {}
    for p in graft_parents:
        base = chain[p]
        # choose first available NN slot (try +z, -z, +y, -y, +x, -x)
        placed = False
        for v in [(0,0,1),(0,0,-1),(0,1,0),(0,-1,0),(1,0,0),(-1,0,0)]:
            candidate = add(base, v)
            if candidate not in occ:
                side_grafts[p] = candidate
                occ.add(candidate)
                placed = True
                break
        if not placed:
            # extremely unlikely for a straight chain, but fail-safe: skip this graft
            print(f"Warning: couldn't place graft at parent {p} (skipping).")
    # graft types inherit parent's type
    # (if you want different graft types you can build a separate dict here)

    # initial energy
    E = energy(chain, side_grafts, types, args.T, eps_fn)
    beta = 1.0 / args.T

    print(f"N={args.N}, grafts={len(side_grafts)}, H={types.count('H')}, T={args.T}")
    acc = 0
    snapshots = []
    sample_every = 1000

    for step in range(1, args.steps+1):
        move = random.choice(MOVE_FUNCS)
        ok, chain_new, occ_new, side_new = move(chain, occ, side_grafts)
        if not ok:
            continue
        dE = energy(chain_new, side_new, types, args.T, eps_fn) - E
        if dE <= 0 or random.random() < math.exp(-beta*dE):
            chain, occ, side_grafts, E = chain_new, occ_new, side_new, E + dE
            acc += 1
        if step % sample_every == 0:
            snapshots.append(sample_trajectory(chain, side_grafts, None))

    print(f"Acceptance ratio: {acc/args.steps:.3f}")
    print(f"Final E = {E:.3f}, Rg = {radius_of_gyration(chain):.3f}")
    # You can now save/analyze snapshots (list of arrays backbone+grafts) or extend analysis below.

if __name__ == "__main__":
    main()

