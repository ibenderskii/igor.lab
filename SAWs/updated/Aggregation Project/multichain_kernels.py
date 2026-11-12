"""Numba kernel for one lane of the multichain sweep, bit-identical to Python.

``mc_sweep_numba`` runs the complete per-lane sweep of
``remd_multichain.mc_sweep`` -- n_local local moves (pivot / crankshaft / end),
then n_translation whole-chain translations, n_reptation reptations and
n_rotation rigid rotations -- in compiled code.  From the same state and the
same ``random.Random`` state it produces the same state and counters, and
leaves the RNG in the same state as the Python sweep.  The state covers
coordinates, occupancy, intra/inter and per-chain contact counts, bend count,
and the per-bead degree cache and joint table.  It does so for every
Hamiltonian branch of the sampler:

    path 0  linear models:       du = b*(l_i*d_intra + l_e*d_inter) + kb*d_bend
    path 1  contact-quadratic:   per-chain intrachain u(m_alpha) + linear inter
    path 2  label-blind global:  u_contact(m_total; M*N)  (saturating, global)
    path 3  label-blind local:   b*d_total - A0*delta_S from the degree cache

How bit-identity is kept (see isaw_kernels for the RNG and potential rules):

* Draws, in the order of multichain_moves: randrange(3) for the local move
  type, then randrange(M), then the move's own draws (pivot: randrange(1, N-1),
  choice(ROT_MATS); crankshaft: randrange(1, N-1); end: random(),
  randrange(6); translation: randrange(6); reptation: randrange(6), random();
  rotation: randrange(N), randrange(24)), and one random() only when du > 0.
* Validity is _check_validity's: new wrapped sites mutually distinct and each
  empty or owned by a moved bead.  Contact deltas count every contact incident
  to a moved bead before and after (pairs of two moved beads once), with
  bonded same-chain neighbours excluded, exactly like _incident_counts.  Local
  degree changes and delta_S follow trial_degree_changes /
  delta_cooperative_sum_cached, with the cooperative tables g computed by the
  Python functions.  Accepted moves update the caches and re-base the chain
  with canonicalize_chain_coordinates.
* Floating-point expressions repeat the sampler's operand order.

The Python occupancy dict ``site_owner`` is rebuilt from the coordinates after
the sweep.  The sampler never depends on its iteration order: the
multiprocessing path already rebuilds it every cycle.
"""
from __future__ import annotations

import math
import random
from typing import Any, Dict

import numpy as np

import isaw_kernels as _ik
from isaw_kernels import NUMBA_AVAILABLE, NUMBA_VERSION, njit  # noqa: F401

_genrand_uint32 = _ik._genrand_uint32
_rand_random = _ik._rand_random
_randbelow = _ik._randbelow
ROT = _ik.ROT
NNV = _ik.NNV

PATH_LINEAR, PATH_QUADRATIC, PATH_GLOBAL, PATH_LOCAL = 0, 1, 2, 3


@njit(cache=True, inline="always")
def _wrap(x, L):
    """x % L (Python floor semantics) without a division in the common range."""
    if x >= 0:
        if x < L:
            return x
        if x < 2 * L:
            return x - L
    elif x >= -L:
        return x + L
    return x % L


@njit(cache=True, inline="always")
def _site(x, y, z, L):
    return (_wrap(x, L) * L + _wrap(y, L)) * L + _wrap(z, L)


@njit(cache=True, inline="always")
def _bend(px, py, pz, cx, cy, cz, nx, ny, nz):
    if (cx - px) * (nx - cx) + (cy - py) * (ny - cy) + (cz - pz) * (nz - cz) == 1:
        return 0
    return 1


@njit(cache=True)
def _chain_bends(P, N):
    """Total 90-degree turns of one chain given as an (N, 3) array."""
    t = 0
    for c in range(1, N - 1):
        t += _bend(P[c - 1, 0], P[c - 1, 1], P[c - 1, 2], P[c, 0], P[c, 1], P[c, 2],
                   P[c + 1, 0], P[c + 1, 1], P[c + 1, 2])
    return t


@njit(cache=True, inline="always")
def _trial_occ(s, grid, tstamp, tocc, vstamp, stamp):
    """Occupant of wrapped site s in the trial configuration (-1 if empty)."""
    if tstamp[s] == stamp:
        return tocc[s]
    if vstamp[s] == stamp:
        return -1
    return grid[s] - 1


@njit(cache=True)
def _build_grid(coords, L, grid):
    M = coords.shape[0]
    N = coords.shape[1]
    for c in range(M):
        for i in range(N):
            s = _site(coords[c, i, 0], coords[c, i, 1], coords[c, i, 2], L)
            if grid[s] != 0:
                raise ValueError("double occupancy in multichain state")
            grid[s] = c * N + i + 1


@njit(cache=True)
def _sweep(coords, grid, L, mt, st, n_local, n_trans, n_rep, n_rot,
           path, kind, b, kappa, A0, q_sat, n_f, li, lin, cs, kb,
           counts, intra_by_chain, nbend_arr, counters,
           deg, joint, gtab, equal_w,
           tstamp, tocc, vstamp, mstamp, mv_g, mv_new, mv_os, mv_ns,
           aff, amark, chg, newp, oldp, mpos):
    M = coords.shape[0]
    N = coords.shape[1]
    stamp = 0
    n_total = n_local + n_trans + n_rep + n_rot
    for att in range(n_total):
        # ---------------- proposal ----------------------------------------
        if att < n_local:
            mtype = _randbelow(mt, st, 3)          # 0 pivot, 1 crankshaft, 2 end
        elif att < n_local + n_trans:
            mtype = 3
        elif att < n_local + n_trans + n_rep:
            mtype = 4
        else:
            mtype = 5
        counters[mtype, 0] += 1
        c = _randbelow(mt, st, M)
        nmv = 0
        ok = True
        changed = False
        piv = -1
        if mtype == 0:
            if N < 3:
                continue
            piv = 1 + _randbelow(mt, st, N - 2)
            r = _randbelow(mt, st, 23)
            px = coords[c, piv, 0]
            py = coords[c, piv, 1]
            pz = coords[c, piv, 2]
            for mono in range(piv + 1, N):
                rx = coords[c, mono, 0] - px
                ry = coords[c, mono, 1] - py
                rz = coords[c, mono, 2] - pz
                mv_g[nmv] = c * N + mono
                mv_new[nmv, 0] = px + ROT[r, 0, 0] * rx + ROT[r, 0, 1] * ry + ROT[r, 0, 2] * rz
                mv_new[nmv, 1] = py + ROT[r, 1, 0] * rx + ROT[r, 1, 1] * ry + ROT[r, 1, 2] * rz
                mv_new[nmv, 2] = pz + ROT[r, 2, 0] * rx + ROT[r, 2, 1] * ry + ROT[r, 2, 2] * rz
                if (mv_new[nmv, 0] != coords[c, mono, 0] or mv_new[nmv, 1] != coords[c, mono, 1]
                        or mv_new[nmv, 2] != coords[c, mono, 2]):
                    changed = True
                nmv += 1
        elif mtype == 1:
            if N < 3:
                continue
            i = 1 + _randbelow(mt, st, N - 2)
            u1x = coords[c, i, 0] - coords[c, i - 1, 0]
            u1y = coords[c, i, 1] - coords[c, i - 1, 1]
            u1z = coords[c, i, 2] - coords[c, i - 1, 2]
            u2x = coords[c, i + 1, 0] - coords[c, i, 0]
            u2y = coords[c, i + 1, 1] - coords[c, i, 1]
            u2z = coords[c, i + 1, 2] - coords[c, i, 2]
            if (u1x == u2x and u1y == u2y and u1z == u2z) or (
                    u1x == -u2x and u1y == -u2y and u1z == -u2z):
                continue
            if u1x * u2x + u1y * u2y + u1z * u2z != 0:
                continue
            mv_g[0] = c * N + i
            mv_new[0, 0] = coords[c, i - 1, 0] + u2x
            mv_new[0, 1] = coords[c, i - 1, 1] + u2y
            mv_new[0, 2] = coords[c, i - 1, 2] + u2z
            changed = (mv_new[0, 0] != coords[c, i, 0] or mv_new[0, 1] != coords[c, i, 1]
                       or mv_new[0, 2] != coords[c, i, 2])
            nmv = 1
        elif mtype == 2:
            end = 0 if _rand_random(mt, st) < 0.5 else N - 1
            anchor = 1 if end == 0 else N - 2
            v = _randbelow(mt, st, 6)
            mv_g[0] = c * N + end
            mv_new[0, 0] = coords[c, anchor, 0] + NNV[v, 0]
            mv_new[0, 1] = coords[c, anchor, 1] + NNV[v, 1]
            mv_new[0, 2] = coords[c, anchor, 2] + NNV[v, 2]
            changed = (mv_new[0, 0] != coords[c, end, 0] or mv_new[0, 1] != coords[c, end, 1]
                       or mv_new[0, 2] != coords[c, end, 2])
            nmv = 1
        elif mtype == 3:
            v = _randbelow(mt, st, 6)
            for mono in range(N):
                mv_g[nmv] = c * N + mono
                mv_new[nmv, 0] = coords[c, mono, 0] + NNV[v, 0]
                mv_new[nmv, 1] = coords[c, mono, 1] + NNV[v, 1]
                mv_new[nmv, 2] = coords[c, mono, 2] + NNV[v, 2]
                nmv += 1
            changed = True
        elif mtype == 4:
            if N < 2:
                continue
            v = _randbelow(mt, st, 6)
            remove_head = _rand_random(mt, st) < 0.5
            for mono in range(N):
                mv_g[nmv] = c * N + mono
                if remove_head:
                    if mono < N - 1:
                        src = mono + 1
                        for k in range(3):
                            mv_new[nmv, k] = coords[c, src, k]
                    else:
                        for k in range(3):
                            mv_new[nmv, k] = coords[c, N - 1, k] + NNV[v, k]
                else:
                    if mono > 0:
                        src = mono - 1
                        for k in range(3):
                            mv_new[nmv, k] = coords[c, src, k]
                    else:
                        for k in range(3):
                            mv_new[nmv, k] = coords[c, 0, k] + NNV[v, k]
                if (mv_new[nmv, 0] != coords[c, mono, 0] or mv_new[nmv, 1] != coords[c, mono, 1]
                        or mv_new[nmv, 2] != coords[c, mono, 2]):
                    changed = True
                nmv += 1
        else:
            if N < 2:
                continue
            j = _randbelow(mt, st, N)
            idx = _randbelow(mt, st, 24)
            px = coords[c, j, 0]
            py = coords[c, j, 1]
            pz = coords[c, j, 2]
            for mono in range(N):
                if mono == j:
                    continue
                rx = coords[c, mono, 0] - px
                ry = coords[c, mono, 1] - py
                rz = coords[c, mono, 2] - pz
                mv_g[nmv] = c * N + mono
                if idx == 23:
                    mv_new[nmv, 0] = px + rx
                    mv_new[nmv, 1] = py + ry
                    mv_new[nmv, 2] = pz + rz
                else:
                    mv_new[nmv, 0] = px + ROT[idx, 0, 0] * rx + ROT[idx, 0, 1] * ry + ROT[idx, 0, 2] * rz
                    mv_new[nmv, 1] = py + ROT[idx, 1, 0] * rx + ROT[idx, 1, 1] * ry + ROT[idx, 1, 2] * rz
                    mv_new[nmv, 2] = pz + ROT[idx, 2, 0] * rx + ROT[idx, 2, 1] * ry + ROT[idx, 2, 2] * rz
                if (mv_new[nmv, 0] != coords[c, mono, 0] or mv_new[nmv, 1] != coords[c, mono, 1]
                        or mv_new[nmv, 2] != coords[c, mono, 2]):
                    changed = True
                nmv += 1

        # ---------------- validity (_check_validity) ----------------------
        stamp += 1
        for t in range(nmv):
            g = mv_g[t]
            mstamp[g] = stamp
            mpos[g] = t
            mono = g % N
            mv_os[t] = _site(coords[c, mono, 0], coords[c, mono, 1], coords[c, mono, 2], L)
            s = _site(mv_new[t, 0], mv_new[t, 1], mv_new[t, 2], L)
            mv_ns[t] = s
            if tstamp[s] == stamp:
                ok = False
            tstamp[s] = stamp
            tocc[s] = g
        if ok:
            for t in range(nmv):
                o = grid[mv_ns[t]] - 1
                if o >= 0 and mstamp[o] != stamp:
                    ok = False
                    break
        if not ok:
            continue
        counters[mtype, 1] += 1
        if changed:
            counters[mtype, 2] += 1
        for t in range(nmv):
            vstamp[mv_os[t]] = stamp

        # ---------------- contact delta (_incident_counts before/after) ----
        bi = 0
        be = 0
        ai = 0
        ae = 0
        for t in range(nmv):
            g = mv_g[t]
            mono = g % N
            ox = coords[c, mono, 0]
            oy = coords[c, mono, 1]
            oz = coords[c, mono, 2]
            nx = mv_new[t, 0]
            ny = mv_new[t, 1]
            nz = mv_new[t, 2]
            for v in range(6):
                o = grid[_site(ox + NNV[v, 0], oy + NNV[v, 1], oz + NNV[v, 2], L)] - 1
                if o >= 0 and o != g and not (mstamp[o] == stamp and o < g):
                    if o // N == c:
                        dm = mono - o % N
                        if dm != 1 and dm != -1:
                            bi += 1
                    else:
                        be += 1
                o = _trial_occ(_site(nx + NNV[v, 0], ny + NNV[v, 1], nz + NNV[v, 2], L),
                               grid, tstamp, tocc, vstamp, stamp)
                if o >= 0 and o != g and not (mstamp[o] == stamp and o < g):
                    if o // N == c:
                        dm = mono - o % N
                        if dm != 1 and dm != -1:
                            ai += 1
                    else:
                        ae += 1
        di = ai - bi
        de = ae - be

        # ---------------- bends (proposal_delta_bends) ---------------------
        dn = 0
        if N >= 3 and (mtype == 0 or mtype == 1 or mtype == 2 or mtype == 4):
            if mtype == 4:
                for t in range(N):
                    for k in range(3):
                        oldp[t, k] = coords[c, t, k]
                        newp[t, k] = mv_new[t, k]
                dn = _chain_bends(newp, N) - _chain_bends(oldp, N)
            elif mtype == 0:
                a = piv - 1
                dn = (_bend(coords[c, a, 0], coords[c, a, 1], coords[c, a, 2],
                            coords[c, piv, 0], coords[c, piv, 1], coords[c, piv, 2],
                            mv_new[0, 0], mv_new[0, 1], mv_new[0, 2])
                      - _bend(coords[c, a, 0], coords[c, a, 1], coords[c, a, 2],
                              coords[c, piv, 0], coords[c, piv, 1], coords[c, piv, 2],
                              coords[c, piv + 1, 0], coords[c, piv + 1, 1], coords[c, piv + 1, 2]))
            else:
                kmov = mv_g[0] % N
                lo = kmov - 1 if kmov - 1 >= 1 else 1
                hi = kmov + 1 if kmov + 1 <= N - 2 else N - 2
                for cc in range(lo, hi + 1):
                    dn -= _bend(coords[c, cc - 1, 0], coords[c, cc - 1, 1], coords[c, cc - 1, 2],
                                coords[c, cc, 0], coords[c, cc, 1], coords[c, cc, 2],
                                coords[c, cc + 1, 0], coords[c, cc + 1, 1], coords[c, cc + 1, 2])
                    for q in range(3):
                        b3 = cc - 1 + q
                        for k in range(3):
                            if b3 == kmov:
                                newp[q, k] = mv_new[0, k]
                            else:
                                newp[q, k] = coords[c, b3, k]
                    dn += _bend(newp[0, 0], newp[0, 1], newp[0, 2], newp[1, 0], newp[1, 1],
                                newp[1, 2], newp[2, 0], newp[2, 1], newp[2, 2])

        # ---------------- energy change --------------------------------------
        nchg = 0
        if path == 3:
            if deg is not None:
                # trial_degree_changes: moved beads + occupants next to old sites
                # (current occupancy) + occupants next to new sites (trial).
                na = 0
                for t in range(nmv):
                    g = mv_g[t]
                    if amark[g] != stamp:
                        amark[g] = stamp
                        aff[na] = g
                        na += 1
                for t in range(nmv):
                    g = mv_g[t]
                    mono = g % N
                    for v in range(6):
                        o = grid[_site(coords[c, mono, 0] + NNV[v, 0], coords[c, mono, 1] + NNV[v, 1],
                                       coords[c, mono, 2] + NNV[v, 2], L)] - 1
                        if o >= 0 and amark[o] != stamp:
                            amark[o] = stamp
                            aff[na] = o
                            na += 1
                        o = _trial_occ(_site(mv_new[t, 0] + NNV[v, 0], mv_new[t, 1] + NNV[v, 1],
                                             mv_new[t, 2] + NNV[v, 2], L),
                                       grid, tstamp, tocc, vstamp, stamp)
                        if o >= 0 and amark[o] != stamp:
                            amark[o] = stamp
                            aff[na] = o
                            na += 1
                for t in range(na):
                    g = aff[t]
                    cg = g // N
                    mg = g % N
                    sx = 0
                    sy = 0
                    sz = 0
                    if mstamp[g] == stamp:
                        u = mpos[g]
                        sx = mv_new[u, 0]
                        sy = mv_new[u, 1]
                        sz = mv_new[u, 2]
                    else:
                        sx = coords[cg, mg, 0]
                        sy = coords[cg, mg, 1]
                        sz = coords[cg, mg, 2]
                    ka = 0
                    ke = 0
                    for v in range(6):
                        o = _trial_occ(_site(sx + NNV[v, 0], sy + NNV[v, 1], sz + NNV[v, 2], L),
                                       grid, tstamp, tocc, vstamp, stamp)
                        if o < 0 or o == g:
                            continue
                        if o // N == cg:
                            dmm = mg - o % N
                            if dmm == 1 or dmm == -1:
                                continue
                            ka += 1
                        else:
                            ke += 1
                    if ka != deg[g, 0] or ke != deg[g, 1]:
                        chg[nchg, 0] = g
                        chg[nchg, 1] = ka
                        chg[nchg, 2] = ke
                        nchg += 1
                da = 0
                dee = 0
                for t in range(nchg):
                    g = chg[t, 0]
                    da += chg[t, 1] - deg[g, 0]
                    dee += chg[t, 2] - deg[g, 1]
                di = da // 2
                de = dee // 2
                d_total = di + de
                if A0 == 0.0:
                    d_contact = b * float(d_total)
                else:
                    dn49 = np.zeros(49, dtype=np.int64)
                    for t in range(nchg):
                        g = chg[t, 0]
                        dn49[deg[g, 0] * 7 + deg[g, 1]] -= 1
                        dn49[chg[t, 1] * 7 + chg[t, 2]] += 1
                    d_S = 0.0
                    if equal_w:
                        nk = np.zeros(7, dtype=np.int64)
                        for a2 in range(7):
                            for e2 in range(7 - a2):
                                nk[a2 + e2] += dn49[a2 * 7 + e2]
                        for k in range(7):
                            if nk[k] != 0:
                                d_S += gtab[k] * float(nk[k])
                    else:
                        for k in range(49):
                            if dn49[k] != 0:
                                d_S += gtab[k] * float(dn49[k])
                    d_contact = b * float(d_total) - A0 * d_S
                du = cs * d_contact + kb * float(dn)
            else:
                du = 0.0
        elif path == 2:
            m_old = counts[0] + counts[1]
            m_new = m_old + di + de
            du = (cs * (_ik._u_scalar(kind, b, kappa, A0, q_sat, n_f, m_new)
                        - _ik._u_scalar(kind, b, kappa, A0, q_sat, n_f, m_old))
                  + kb * float(dn))
        elif path == 1:
            m_old = intra_by_chain[c]
            m_new = m_old + di
            du = (li * (_ik._u_scalar(kind, b, kappa, A0, q_sat, n_f, m_new)
                        - _ik._u_scalar(kind, b, kappa, A0, q_sat, n_f, m_old))
                  + lin * b * float(de) + kb * float(dn))
        else:
            du = b * (li * float(di) + lin * float(de)) + kb * float(dn)

        if not (du <= 0.0 or _rand_random(mt, st) < math.exp(-du)):
            continue

        # ---------------- apply (apply_proposal) -----------------------------
        for t in range(nmv):
            grid[mv_os[t]] = 0
        for t in range(nmv):
            g = mv_g[t]
            mono = g % N
            coords[c, mono, 0] = mv_new[t, 0]
            coords[c, mono, 1] = mv_new[t, 1]
            coords[c, mono, 2] = mv_new[t, 2]
            grid[mv_ns[t]] = g + 1
        counts[0] += di
        counts[1] += de
        intra_by_chain[c] += di
        if path == 3:
            if deg is not None:
                for t in range(nchg):
                    g = chg[t, 0]
                    joint[deg[g, 0], deg[g, 1]] -= 1
                    joint[chg[t, 1], chg[t, 2]] += 1
                    deg[g, 0] = chg[t, 1]
                    deg[g, 1] = chg[t, 2]
        # canonicalize_chain_coordinates
        sx0 = coords[c, 0, 0] // L
        sy0 = coords[c, 0, 1] // L
        sz0 = coords[c, 0, 2] // L
        if sx0 != 0 or sy0 != 0 or sz0 != 0:
            for mono in range(N):
                coords[c, mono, 0] -= sx0 * L
                coords[c, mono, 1] -= sy0 * L
                coords[c, mono, 2] -= sz0 * L
        nbend_arr[0] += dn
        if changed:
            counters[mtype, 3] += 1


def mc_sweep_numba(state, counters, rng: random.Random, *, n_local: int, n_translation: int,
                   n_reptation: int, n_rotation: int, path: int, kind: int, b: float,
                   kappa: float, A0: float, q_sat: float, n_f: float, li: float, lin: float,
                   cs: float, kb: float, gtab=None, equal_w: bool = True) -> None:
    """Run one lane's sweep in place on ``state`` / ``counters`` with ``rng``.

    The Hamiltonian coefficients are resolved by remd_multichain.mc_sweep (the
    Python reference) and passed through, so both kernels see the same numbers.
    """
    if not NUMBA_AVAILABLE:
        raise RuntimeError("numba is not available; use the python kernel")
    coords = state.coords_unwrapped
    if coords.dtype != np.int64 or not coords.flags.c_contiguous:
        coords = np.ascontiguousarray(coords, dtype=np.int64)
        state.coords_unwrapped = coords
    M, N = int(coords.shape[0]), int(coords.shape[1])
    L = int(state.box_size)
    if counters.dtype != np.int64 or not counters.flags.c_contiguous or counters.shape != (6, 4):
        raise ValueError("move counters must be a C-contiguous (6, 4) int64 array")
    ibc = state.intra_contacts_by_chain
    if ibc.dtype != np.int64 or not ibc.flags.c_contiguous:
        ibc = np.ascontiguousarray(ibc, dtype=np.int64)
        state.intra_contacts_by_chain = ibc
    deg = joint = None
    if path == PATH_LOCAL:
        deg = state.degrees
        joint = state.degree_joint
        if deg is None or joint is None:
            raise RuntimeError("local cooperativity requires the per-bead degree cache")
        if gtab is None:
            raise ValueError("local cooperativity requires the g table")
    gt = np.ascontiguousarray(np.asarray(gtab if gtab is not None else [0.0], dtype=np.float64))
    nb = M * N
    V = L * L * L
    grid = np.zeros(V, dtype=np.int64)
    _build_grid(coords, L, grid)
    counts = np.array([int(state.counts.intra), int(state.counts.inter)], dtype=np.int64)
    nbend = np.array([int(state.n_bend)], dtype=np.int64)
    rs = rng.getstate()
    mt, st = _ik.mt_from_state(rs)
    _sweep(coords, grid, L, mt, st, int(n_local), int(n_translation), int(n_reptation),
           int(n_rotation), int(path), int(kind), float(b), float(kappa), float(A0),
           float(q_sat), float(n_f), float(li), float(lin), float(cs), float(kb),
           counts, ibc, nbend, counters, deg, joint, gt, bool(equal_w),
           np.zeros(V, dtype=np.int64), np.zeros(V, dtype=np.int64),
           np.zeros(V, dtype=np.int64), np.zeros(nb, dtype=np.int64),
           np.zeros(N, dtype=np.int64), np.zeros((N, 3), dtype=np.int64),
           np.zeros(N, dtype=np.int64), np.zeros(N, dtype=np.int64),
           np.zeros(nb, dtype=np.int64), np.zeros(nb, dtype=np.int64),
           np.zeros((nb, 3), dtype=np.int64), np.zeros((max(N, 3), 3), dtype=np.int64),
           np.zeros((max(N, 3), 3), dtype=np.int64), np.zeros(nb, dtype=np.int64))
    rng.setstate(_ik.state_from_mt(mt, st, rs))
    state.counts.intra = int(counts[0])
    state.counts.inter = int(counts[1])
    state.n_bend = int(nbend[0])
    wrapped = np.mod(coords, L).reshape(-1, 3).tolist()
    state.site_owner = {tuple(s): g for g, s in enumerate(wrapped)}


def check_move_tables(mvs, remd) -> None:
    """The compiled ROT / NN tables must equal the Python ones, in order."""
    if not (np.array_equal(np.array(remd.ROT_MATS, dtype=np.int64), ROT)
            and np.array_equal(np.array(mvs.NN6, dtype=np.int64), NNV)):
        raise RuntimeError("multichain_kernels move tables differ from ROT_MATS / NN6")


def snapshot(state, counters) -> tuple:
    """Everything a sweep can change, for lockstep comparison."""
    return (state.coords_unwrapped.tobytes(), state.coords_unwrapped.shape,
            tuple(sorted(state.site_owner.items())), state.counts.as_tuple(),
            int(state.n_bend), state.intra_contacts_by_chain.tobytes(),
            None if state.degrees is None else state.degrees.tobytes(),
            None if state.degree_joint is None else state.degree_joint.tobytes(),
            counters.tobytes())


def selfcheck(rm, model_name: str, params, Tref: float, Tscale: float, M: int, N: int,
              L: int, temps, lambda_intra: float, lambda_inter: float, kappa_bend: float,
              cooperativity, degree_weights, n_local_sweeps: int = 1,
              n_translation_sweeps: int = 1, n_reptation_sweeps: int = 1,
              n_rotation_sweeps: int = 1, sweeps: int = 2, seed: int = 20261102) -> Dict[str, Any]:
    """Python vs Numba lane sweeps in lockstep on THIS machine.

    For the lowest, middle and highest temperature: build an independent lane
    state, advance it with ``sweeps`` Python sweeps, then run ``sweeps`` more
    sweeps with each kernel from copies of the same state and RNG state, and
    require identical state, counters and RNG state.
    """
    import copy
    info: Dict[str, Any] = {"ok": False, "sweeps": int(sweeps), "temperatures": []}
    if not NUMBA_AVAILABLE:
        info["reason"] = "numba not importable"
        return info
    temps = [float(t) for t in temps]
    picks = sorted({temps[0], temps[len(temps) // 2], temps[-1]})
    track = rm._tracks_degrees(model_name, cooperativity)
    kw = dict(model_name=model_name, params=params, Tref=Tref, Tscale=Tscale,
              lambda_intra=lambda_intra, lambda_inter=lambda_inter,
              n_local=int(n_local_sweeps) * M * N, n_translation=int(n_translation_sweeps) * M,
              n_reptation=int(n_reptation_sweeps) * M, n_rotation=int(n_rotation_sweeps) * M,
              kappa_bend=kappa_bend, cooperativity=cooperativity, degree_weights=degree_weights)
    for T in picks:
        rep = rm._build_independent_replicas(np.array([T]), M, N, L, seed, 1,
                                             track_degrees=track)[0]
        st = rep.state
        ctr = rep.move_counters
        for k in range(int(sweeps)):
            rm.mc_sweep(st, ctr, T, rng=random.Random(seed + k), kernel="python", **kw)
        sa, sb = st.copy(), st.copy()
        ca, cb = ctr.copy(), ctr.copy()
        same = True
        for k in range(int(sweeps)):
            ra, rb = random.Random(seed + 100 + k), random.Random(seed + 100 + k)
            rm.mc_sweep(sa, ca, T, rng=ra, kernel="python", **kw)
            rm.mc_sweep(sb, cb, T, rng=rb, kernel="numba", **kw)
            same = same and (snapshot(sa, ca) == snapshot(sb, cb)) and ra.getstate() == rb.getstate()
        info["temperatures"].append({"T": T, "identical": bool(same)})
        if not same:
            info["reason"] = f"python and numba lane sweeps differ at T={T}"
            return info
    info["ok"] = True
    return info


# ---------------------------------------------------------------------------
# Integer per-cycle observables (exact; replace the O(M*N) Python oracles)
# ---------------------------------------------------------------------------

@njit(cache=True)
def _degrees_split(coords, grid, L, out):
    """contact_degrees_split: (k_intra, k_inter) of every bead, global-ID order."""
    M = coords.shape[0]
    N = coords.shape[1]
    for c in range(M):
        for i in range(N):
            g = c * N + i
            ka = 0
            ke = 0
            for v in range(6):
                o = grid[_site(coords[c, i, 0] + NNV[v, 0], coords[c, i, 1] + NNV[v, 1],
                               coords[c, i, 2] + NNV[v, 2], L)] - 1
                if o < 0 or o == g:
                    continue
                if o // N == c:
                    d = i - o % N
                    if d == 1 or d == -1:
                        continue
                    ka += 1
                else:
                    ke += 1
            out[g, 0] = ka
            out[g, 1] = ke


@njit(cache=True)
def _pair_matrix(coords, grid, L, out):
    """interchain_pair_counts as an (M, M) upper-triangular count matrix."""
    M = coords.shape[0]
    N = coords.shape[1]
    for c in range(M):
        for i in range(N):
            g = c * N + i
            for v in range(6):
                o = grid[_site(coords[c, i, 0] + NNV[v, 0], coords[c, i, 1] + NNV[v, 1],
                               coords[c, i, 2] + NNV[v, 2], L)] - 1
                if o <= g:
                    continue
                cj = o // N
                if cj == c:
                    continue
                if c < cj:
                    out[c, cj] += 1
                else:
                    out[cj, c] += 1


def _state_grid(state):
    coords = np.ascontiguousarray(state.coords_unwrapped, dtype=np.int64)
    L = int(state.box_size)
    grid = np.zeros(L * L * L, dtype=np.int64)
    _build_grid(coords, L, grid)
    return coords, L, grid


def contact_degrees_split(state) -> np.ndarray:
    """Same array as multichain_contacts.contact_degrees_split (exact integers)."""
    coords, L, grid = _state_grid(state)
    out = np.zeros((coords.shape[0] * coords.shape[1], 2), dtype=np.int64)
    _degrees_split(coords, grid, L, out)
    return out


def interchain_pair_counts(state) -> Dict:
    """Same mapping as multichain_contacts.interchain_pair_counts.

    Keys are inserted in sorted order rather than discovery order;
    cluster_summary only uses the set of pairs and their counts.
    """
    coords, L, grid = _state_grid(state)
    M = int(coords.shape[0])
    mat = np.zeros((M, M), dtype=np.int64)
    _pair_matrix(coords, grid, L, mat)
    out = {}
    for a, b in zip(*np.nonzero(mat)):
        out[(int(a), int(b))] = int(mat[a, b])
    return out
