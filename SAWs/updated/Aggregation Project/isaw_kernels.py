"""Numba kernel for the single-chain ISAW local-move sweep, bit-identical to Python.

``mc_sweep_numba`` is a drop-in replacement for
``remd_uniform_chain_2_new.mc_sweep``: from the same state and the same
Python ``random`` state it produces the same chain, contact count, coordination
histogram, bend count, energy, move counters and acceptance counters, and leaves
``random`` in the same state -- for every potential kind the sampler supports
(linear, contact_quadratic, saturating_cooperative,
local_coordination_saturation) and any kappa_bend.  It is faster because the
moves, the contact/degree bookkeeping and the Metropolis test run in compiled
code; the arithmetic is unchanged.  How bit-identity is kept:

* Random numbers: CPython's Mersenne Twister (MT19937), ``random()`` and the
  ``getrandbits``-rejection ``_randbelow`` behind ``randrange``/``choice`` are
  re-implemented exactly; the 624-word state is read from ``random.getstate()``
  before the sweep and written back with ``random.setstate()`` after it.
* Draw order: one ``randrange(3)`` per step, then exactly the draws of the
  chosen move (pivot: randrange(1, N-1), choice(ROT_MATS); crankshaft:
  randrange(1, N-1); end: random(), choice(NN_VECS)), then one ``random()``
  only when du > 0 -- the same calls in the same order as the Python moves.
* Potentials: the same IEEE operations in the same order as
  ``_u_linear_value`` / ``_u_contact_quadratic_value`` /
  ``_u_saturating_cooperative_value`` (no fast-math).  b(T), kappa(T), A0 and
  q_sat come from the Python functions.  The local-coordination sum
  ``local_coordination_sum`` uses ``np.dot`` (BLAS), whose summation order is
  library-specific, so it is NOT re-implemented: each distinct histogram's
  value is computed once by the Python function itself (an object-mode
  call) and cached.
* Contacts and degrees are updated incrementally (exact integers), instead of
  being recounted over the whole chain after every move.

Occupancy uses a periodic hash grid of side LB = next power of two >= 2N + 2.
Every lookup compares positions less than LB apart along each axis, so no two
distinct sites alias.  The grid is a per-process scratch array; only the
chain's own N cells are set and cleared per sweep.

Without numba importing, ``NUMBA_AVAILABLE`` is False and callers fall back to
the Python kernel.  Python >= 3.8; numba >= 0.53.
"""
from __future__ import annotations

import math
import random
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:  # pragma: no cover - availability depends on the environment
    import numba as _nb
    from numba import njit
    from numba import types as _nbt
    from numba.typed import Dict as _NbDict
    NUMBA_AVAILABLE = True
    NUMBA_VERSION = str(_nb.__version__)
except Exception:  # pragma: no cover
    NUMBA_AVAILABLE = False
    NUMBA_VERSION = None

    def njit(*args, **kwargs):  # minimal stand-in so the module still imports
        if args and callable(args[0]):
            return args[0]
        return lambda f: f

KIND_CODES = {
    "linear": 0,
    "contact_quadratic": 1,
    "saturating_cooperative": 2,
    "local_coordination_saturation": 3,
}

def _cubic_rotations():
    """The 23 proper cubic rotations except the identity, in the order of
    remd_uniform_chain_2_new.ROT_MATS (same construction; checked at run time)."""
    from itertools import permutations, product
    rots = []
    for perm in permutations([0, 1, 2]):
        for signs in product([-1, 1], repeat=3):
            M = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
            for r, c in enumerate(perm):
                M[r][c] = signs[r]
            (a, b, c), (d, e, f), (g, h, i) = M
            if a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g) == 1:
                rots.append(M)
    return [M for M in rots if M != [[1, 0, 0], [0, 1, 0], [0, 0, 1]]]


ROT = np.array(_cubic_rotations(), dtype=np.int64)                      # (23, 3, 3)
NNV = np.array([(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)],
               dtype=np.int64)                                          # NN_VECS order

# Python implementation of the local-coordination sum, installed by the caller
# (remd_uniform_chain_2_new.local_coordination_sum).  Called from object mode.
_COOP_SUM_IMPL = [None]


def _py_coop_sum(hist, q_sat):
    return float(_COOP_SUM_IMPL[0](hist, q_sat))


# ---------------------------------------------------------------------------
# CPython-identical MT19937
# ---------------------------------------------------------------------------

@njit(cache=True, inline="always")
def _genrand_uint32(mt, st):
    if st[0] >= 624:
        for kk in range(624):
            y = (mt[kk] & np.uint64(0x80000000)) | (mt[(kk + 1) % 624] & np.uint64(0x7fffffff))
            v = mt[(kk + 397) % 624] ^ (y >> np.uint64(1))
            if y & np.uint64(1):
                v = v ^ np.uint64(0x9908b0df)
            mt[kk] = v
        st[0] = 0
    y = mt[st[0]]
    st[0] += 1
    y = y ^ (y >> np.uint64(11))
    y = y ^ ((y << np.uint64(7)) & np.uint64(0x9d2c5680))
    y = y ^ ((y << np.uint64(15)) & np.uint64(0xefc60000))
    y = y ^ (y >> np.uint64(18))
    return y & np.uint64(0xffffffff)


@njit(cache=True, inline="always")
def _rand_random(mt, st):
    """random.random()"""
    a = _genrand_uint32(mt, st) >> np.uint64(5)
    b = _genrand_uint32(mt, st) >> np.uint64(6)
    return (float(a) * 67108864.0 + float(b)) * (1.0 / 9007199254740992.0)


@njit(cache=True, inline="always")
def _randbelow(mt, st, n):
    """random._randbelow_with_getrandbits(n) for 0 < n < 2**32"""
    k = 0
    t = n
    while t:
        k += 1
        t >>= 1
    shift = np.uint64(32 - k)
    r = int(_genrand_uint32(mt, st) >> shift)
    while r >= n:
        r = int(_genrand_uint32(mt, st) >> shift)
    return r


def mt_from_state(state) -> Tuple[np.ndarray, np.ndarray]:
    """(mt, st) arrays from ``random.getstate()``."""
    internal = state[1]
    return (np.array(internal[:624], dtype=np.uint64),
            np.array([internal[624]], dtype=np.int64))


def state_from_mt(mt: np.ndarray, st: np.ndarray, template) -> tuple:
    """``random.setstate``-compatible tuple (keeps version and gauss_next)."""
    return (template[0], tuple(mt.tolist()) + (int(st[0]),), template[2])


# ---------------------------------------------------------------------------
# Lattice helpers
# ---------------------------------------------------------------------------

@njit(cache=True, inline="always")
def _cell(x, y, z, lbits, mask):
    return ((x & mask) << (2 * lbits)) | ((y & mask) << lbits) | (z & mask)


@njit(cache=True)
def _grid_fill(grid, chain, lbits, value_offset):
    """value_offset=1: write bead index + 1; value_offset=0: clear the chain's cells."""
    mask = (1 << lbits) - 1
    for k in range(chain.shape[0]):
        c = _cell(chain[k, 0], chain[k, 1], chain[k, 2], lbits, mask)
        grid[c] = (k + 1) if value_offset else 0


@njit(cache=True)
def _degrees(grid, chain, lbits, deg):
    """Nonbonded nearest-neighbour degree of every bead; returns the contact count."""
    mask = (1 << lbits) - 1
    n = chain.shape[0]
    total = 0
    for k in range(n):
        d = 0
        for dx, dy, dz in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
            o = grid[_cell(chain[k, 0] + dx, chain[k, 1] + dy, chain[k, 2] + dz, lbits, mask)] - 1
            if o >= 0 and o != k - 1 and o != k + 1 and o != k:
                d += 1
        deg[k] = d
        total += d
    return total // 2


@njit(cache=True, inline="always")
def _bend(px, py, pz, cx, cy, cz, nx, ny, nz):
    """lattice_bending.bend_indicator for unit bonds (reversal impossible in a SAW)."""
    if (cx - px) * (nx - cx) + (cy - py) * (ny - cy) + (cz - pz) * (nz - cz) == 1:
        return 0
    return 1


@njit(cache=True)
def _coop_sum(hist, s_cache, key_base, q_sat):
    key = 0
    for k in range(6, -1, -1):
        key = key * key_base + hist[k]
    if key in s_cache:
        return s_cache[key]
    h = hist.copy()
    with _nb.objmode(s="float64"):
        s = _py_coop_sum(h, q_sat)
    s_cache[key] = s
    return s


@njit(cache=True, inline="always")
def _u_scalar(kind, b, kappa, A0, q_sat, n_f, m):
    """_u_linear_value / _u_contact_quadratic_value / _u_saturating_cooperative_value."""
    fm = float(m)
    if kind == 0:
        return b * fm
    if kind == 1:
        return b * fm + kappa * (fm * fm) / (2.0 * n_f)
    u_lin = b * fm
    if A0 == 0.0:
        return u_lin
    q = fm / n_f
    r = q / q_sat
    return u_lin - n_f * A0 * (q * q) / (1.0 + r * r)


@njit(cache=True)
def _u_local(b, A0, q_sat, m, hist, s_cache, key_base):
    """reduced_contact_potential_state for local_coordination_saturation."""
    u_lin = b * float(m)
    if A0 == 0.0:
        return u_lin
    return u_lin - A0 * _coop_sum(hist, s_cache, key_base, q_sat)


# ---------------------------------------------------------------------------
# The sweep
# ---------------------------------------------------------------------------

@njit(cache=True)
def _sweep(chain, grid, lbits, mt, st, steps, kind, T, b, kappa, A0, q_sat, n_f, kb,
           scal, counters, newt, deg, hist, hist_new, ddeg, mark, touched,
           s_cache, key_base):
    """Run ``steps`` local Metropolis steps in place.

    ``s_cache`` is the local-coordination sum cache for the local model and
    None otherwise; numba compiles a separate specialization for None and
    prunes every degree-tracking branch from it (the tracking code, even when
    not executed, roughly halves the speed of the scalar kernels).
    scal = [m, n_bend, local_acc, local_prop] (int64, updated in place).
    Returns (E of the last accepted move, whether any move was accepted).
    """
    n = chain.shape[0]
    mask = (1 << lbits) - 1
    m = scal[0]
    nbend = scal[1]
    lacc = scal[2]
    lprop = scal[3]
    E_last = 0.0
    any_acc = False
    if s_cache is not None:
        u_cur = _u_local(b, A0, q_sat, m, hist, s_cache, key_base)
    else:
        u_cur = _u_scalar(kind, b, kappa, A0, q_sat, n_f, m)
    for _step in range(steps):
        lprop += 1
        mi = _randbelow(mt, st, 3)
        counters[mi, 0] += 1
        ntouch = 0
        dm = 0
        dn = 0
        k = -1
        nx = 0
        ny = 0
        nz = 0
        bx = 0
        by = 0
        bz = 0
        piv = -1
        state_changing = True
        if mi == 0:
            # ---- pivot: rotate the tail i+1.. about bead i ----------------
            i = 1 + _randbelow(mt, st, n - 2)
            r = _randbelow(mt, st, 23)
            px = chain[i, 0]
            py = chain[i, 1]
            pz = chain[i, 2]
            ok = True
            changed = False
            for j in range(i + 1, n):
                rx = chain[j, 0] - px
                ry = chain[j, 1] - py
                rz = chain[j, 2] - pz
                x = px + ROT[r, 0, 0] * rx + ROT[r, 0, 1] * ry + ROT[r, 0, 2] * rz
                y = py + ROT[r, 1, 0] * rx + ROT[r, 1, 1] * ry + ROT[r, 1, 2] * rz
                z = pz + ROT[r, 2, 0] * rx + ROT[r, 2, 1] * ry + ROT[r, 2, 2] * rz
                o = grid[_cell(x, y, z, lbits, mask)] - 1
                if o >= 0 and o <= i:
                    ok = False
                    break
                newt[j, 0] = x
                newt[j, 1] = y
                newt[j, 2] = z
                if x != chain[j, 0] or y != chain[j, 1] or z != chain[j, 2]:
                    changed = True
            if not ok:
                continue
            counters[0, 1] += 1
            state_changing = changed
            if changed:
                counters[0, 2] += 1
            piv = i
            # contacts between the head (0..i) and the moved tail; the bond
            # (i, i+1) is excluded on both sides.  Tail-tail contacts are
            # preserved by the rigid rotation.
            for j in range(i + 1, n):
                for v in range(6):
                    o = grid[_cell(newt[j, 0] + NNV[v, 0], newt[j, 1] + NNV[v, 1],
                                   newt[j, 2] + NNV[v, 2], lbits, mask)] - 1
                    if o >= 0 and o <= i and not (j == i + 1 and o == i):
                        dm += 1
                        if s_cache is not None:
                            for x in (j, o):
                                if mark[x] == 0:
                                    mark[x] = 1
                                    touched[ntouch] = x
                                    ntouch += 1
                                ddeg[x] += 1
                    o = grid[_cell(chain[j, 0] + NNV[v, 0], chain[j, 1] + NNV[v, 1],
                                   chain[j, 2] + NNV[v, 2], lbits, mask)] - 1
                    if o >= 0 and o <= i and not (j == i + 1 and o == i):
                        dm -= 1
                        if s_cache is not None:
                            for x in (j, o):
                                if mark[x] == 0:
                                    mark[x] = 1
                                    touched[ntouch] = x
                                    ntouch += 1
                                ddeg[x] -= 1
            # only the angle at bead i can change (delta_bends centers i, i+1, i+2)
            a = i - 1
            dn = (_bend(chain[a, 0], chain[a, 1], chain[a, 2], px, py, pz,
                        newt[i + 1, 0], newt[i + 1, 1], newt[i + 1, 2])
                  - _bend(chain[a, 0], chain[a, 1], chain[a, 2], px, py, pz,
                          chain[i + 1, 0], chain[i + 1, 1], chain[i + 1, 2]))
        else:
            if mi == 1:
                # ---- crankshaft: flip the corner bead k --------------------
                k = 1 + _randbelow(mt, st, n - 2)
                ax = chain[k - 1, 0]
                ay = chain[k - 1, 1]
                az = chain[k - 1, 2]
                bx = chain[k, 0]
                by = chain[k, 1]
                bz = chain[k, 2]
                u2x = chain[k + 1, 0] - bx
                u2y = chain[k + 1, 1] - by
                u2z = chain[k + 1, 2] - bz
                if (bx - ax) * u2x + (by - ay) * u2y + (bz - az) * u2z != 0:
                    continue
                nx = ax + u2x
                ny = ay + u2y
                nz = az + u2z
                if grid[_cell(nx, ny, nz, lbits, mask)] != 0:
                    continue
                lo_c = k - 1 if k - 1 >= 1 else 1
                hi_c = k + 1 if k + 1 <= n - 2 else n - 2
            else:
                # ---- end move ----------------------------------------------
                end = 0 if _rand_random(mt, st) < 0.5 else n - 1
                anc = 1 if end == 0 else n - 2
                v0 = _randbelow(mt, st, 6)
                nx = chain[anc, 0] + NNV[v0, 0]
                ny = chain[anc, 1] + NNV[v0, 1]
                nz = chain[anc, 2] + NNV[v0, 2]
                k = end
                bx = chain[k, 0]
                by = chain[k, 1]
                bz = chain[k, 2]
                if nx == bx and ny == by and nz == bz:
                    continue
                if grid[_cell(nx, ny, nz, lbits, mask)] != 0:
                    continue
                lo_c = 1 if end == 0 else n - 2
                hi_c = lo_c
            counters[mi, 1] += 1
            counters[mi, 2] += 1   # the bead always moves
            for v in range(6):
                o = grid[_cell(nx + NNV[v, 0], ny + NNV[v, 1], nz + NNV[v, 2], lbits, mask)] - 1
                if o >= 0 and o != k - 1 and o != k and o != k + 1:
                    dm += 1
                    if s_cache is not None:
                        for x in (k, o):
                            if mark[x] == 0:
                                mark[x] = 1
                                touched[ntouch] = x
                                ntouch += 1
                            ddeg[x] += 1
                o = grid[_cell(bx + NNV[v, 0], by + NNV[v, 1], bz + NNV[v, 2], lbits, mask)] - 1
                if o >= 0 and o != k - 1 and o != k and o != k + 1:
                    dm -= 1
                    if s_cache is not None:
                        for x in (k, o):
                            if mark[x] == 0:
                                mark[x] = 1
                                touched[ntouch] = x
                                ntouch += 1
                            ddeg[x] -= 1
            for c in range(lo_c, hi_c + 1):
                dn -= _bend(chain[c - 1, 0], chain[c - 1, 1], chain[c - 1, 2],
                            chain[c, 0], chain[c, 1], chain[c, 2],
                            chain[c + 1, 0], chain[c + 1, 1], chain[c + 1, 2])
                if c - 1 == k:
                    q0x, q0y, q0z = nx, ny, nz
                else:
                    q0x, q0y, q0z = chain[c - 1, 0], chain[c - 1, 1], chain[c - 1, 2]
                if c == k:
                    q1x, q1y, q1z = nx, ny, nz
                else:
                    q1x, q1y, q1z = chain[c, 0], chain[c, 1], chain[c, 2]
                if c + 1 == k:
                    q2x, q2y, q2z = nx, ny, nz
                else:
                    q2x, q2y, q2z = chain[c + 1, 0], chain[c + 1, 1], chain[c + 1, 2]
                dn += _bend(q0x, q0y, q0z, q1x, q1y, q1z, q2x, q2y, q2z)

        m_new = m + dm
        if s_cache is not None:
            for t in range(7):
                hist_new[t] = hist[t]
            for t in range(ntouch):
                x = touched[t]
                d = ddeg[x]
                if d != 0:
                    kn = deg[x] + d
                    if kn < 0 or kn > 6:
                        raise ValueError("coordination degree left [0, 6]")
                    hist_new[deg[x]] -= 1
                    hist_new[kn] += 1
        if s_cache is not None:
            u_new = _u_local(b, A0, q_sat, m_new, hist_new, s_cache, key_base)
        else:
            u_new = _u_scalar(kind, b, kappa, A0, q_sat, n_f, m_new)
        du = u_new - u_cur + kb * dn
        if du <= 0.0 or _rand_random(mt, st) < math.exp(-du):
            if mi == 0:
                for j in range(piv + 1, n):
                    grid[_cell(chain[j, 0], chain[j, 1], chain[j, 2], lbits, mask)] = 0
                for j in range(piv + 1, n):
                    chain[j, 0] = newt[j, 0]
                    chain[j, 1] = newt[j, 1]
                    chain[j, 2] = newt[j, 2]
                    grid[_cell(chain[j, 0], chain[j, 1], chain[j, 2], lbits, mask)] = j + 1
            else:
                grid[_cell(bx, by, bz, lbits, mask)] = 0
                chain[k, 0] = nx
                chain[k, 1] = ny
                chain[k, 2] = nz
                grid[_cell(nx, ny, nz, lbits, mask)] = k + 1
            m = m_new
            nbend += dn
            u_cur = u_new
            E_last = T * u_new
            any_acc = True
            lacc += 1
            if state_changing:
                counters[mi, 3] += 1
            if s_cache is not None:
                for t in range(ntouch):
                    x = touched[t]
                    deg[x] += ddeg[x]
                for t in range(7):
                    hist[t] = hist_new[t]
        if s_cache is not None:
            for t in range(ntouch):
                x = touched[t]
                ddeg[x] = 0
                mark[x] = 0
    scal[0] = m
    scal[1] = nbend
    scal[2] = lacc
    scal[3] = lprop
    return E_last, any_acc


# ---------------------------------------------------------------------------
# Python-facing wrapper
# ---------------------------------------------------------------------------

_GRIDS: Dict[int, np.ndarray] = {}
_S_CACHES: Dict[Tuple[float, int], Any] = {}
_CONSTS: Dict[str, np.ndarray] = {}


def _grid_bits(n_beads: int) -> int:
    lbits = 1
    while (1 << lbits) < 2 * int(n_beads) + 2:
        lbits += 1
    return lbits


def _scratch_grid(lbits: int) -> np.ndarray:
    g = _GRIDS.get(lbits)
    if g is None:
        g = np.zeros(1 << (3 * lbits), dtype=np.int32)
        _GRIDS[lbits] = g
    return g


def _s_cache(q_sat: float, n_beads: int):
    """Per-(q_sat, N) cache of local_coordination_sum.  Keys encode a histogram in
    base N + 1, so caches for different chain lengths must not be shared."""
    key = (float(q_sat), int(n_beads))
    c = _S_CACHES.get(key)
    if c is None:
        c = _NbDict.empty(key_type=_nbt.int64, value_type=_nbt.float64)
        _S_CACHES[key] = c
    return c


def _check_move_tables(R) -> None:
    """The compiled ROT/NNV tables must equal the sampler's, entry for entry and in
    order (random.choice indexes them)."""
    if "checked" not in _CONSTS:
        if not (np.array_equal(np.array(R.ROT_MATS, dtype=np.int64), ROT)
                and np.array_equal(np.array(R.NN_VECS, dtype=np.int64), NNV)):
            raise RuntimeError("isaw_kernels move tables differ from the sampler's ROT_MATS/NN_VECS")
        _CONSTS["checked"] = np.ones(1)


def mc_sweep_numba(R, replica, steps: int, model_name: str, params, Tref: float,
                   Tscale: float, kappa_bend: float = 0.0, n_beads=None) -> None:
    """Bit-identical replacement for ``R.mc_sweep`` (R = remd_uniform_chain_2_new).

    Consumes and advances the module-level ``random`` state exactly as the
    Python sweep does, and updates ``replica`` in place.
    """
    if not NUMBA_AVAILABLE:
        raise RuntimeError("numba is not available; use the python kernel")
    state = replica.state
    T = float(replica.T)
    N = int(n_beads) if n_beads is not None else len(state.chain)
    spec = R.MODEL_REGISTRY[model_name]
    kind = KIND_CODES[str(spec["potential_kind"])]
    configuration_dependent = bool(spec["configuration_dependent"])
    b = R.reduced_bias(model_name, params, T, Tref, Tscale)
    kappa = 0.0
    A0, q_sat = 0.0, 1.0
    n_f = float(N)
    if kind == 1:
        kappa = R.quadratic_bias(model_name, params, T, Tref, Tscale)
        n_f = float(R.validate_chain_length(model_name, N))
    elif kind == 2:
        A0, q_sat = R.validated_saturating_params(params)
        n_f = float(R.validate_chain_length(model_name, N))
    elif kind == 3:
        A0, q_sat = R.validated_saturating_params(params)
    kb = float(kappa_bend)

    chain = np.array(state.chain, dtype=np.int64).reshape(-1, 3)
    if chain.shape[0] != N:
        raise ValueError(f"chain has {chain.shape[0]} beads, expected {N}")
    lbits = _grid_bits(N)
    grid = _scratch_grid(lbits)
    _check_move_tables(R)
    deg = np.zeros(N, dtype=np.int64)
    hist = np.zeros(7, dtype=np.int64)
    track = kind == 3
    if track and (N + 1) ** 7 >= 2 ** 63:
        raise ValueError(f"numba kernel supports N <= 501 for {model_name}; got N={N}")
    if configuration_dependent:
        if state.coordination_hist is None:
            raise RuntimeError("configuration-dependent ChainState is missing coordination_hist")
        hist[:] = np.asarray(state.coordination_hist, dtype=np.int64)
        _COOP_SUM_IMPL[0] = R.local_coordination_sum
    s_cache = _s_cache(q_sat, N)
    scal = np.array([int(state.m), int(state.n_bend), int(replica.local_acc),
                     int(replica.local_prop)], dtype=np.int64)
    counters = replica.move_counters
    if counters.dtype != np.int64 or not counters.flags.c_contiguous or counters.shape != (3, 4):
        raise ValueError("move_counters must be a C-contiguous (3, 4) int64 array")
    rs = random.getstate()
    mt, st = mt_from_state(rs)
    _grid_fill(grid, chain, lbits, 1)
    try:
        m0 = _degrees(grid, chain, lbits, deg)
        if m0 != int(state.m):
            raise RuntimeError(f"state.m={state.m} disagrees with the chain ({m0} contacts)")
        if track:
            h0 = np.bincount(deg, minlength=7)[:7]
            if not np.array_equal(h0, hist):
                raise RuntimeError("coordination_hist disagrees with the chain")
        E_last, any_acc = _sweep(
            chain, grid, lbits, mt, st, int(steps), kind, T, float(b), float(kappa),
            float(A0), float(q_sat), n_f, kb, scal, counters,
            np.zeros_like(chain), deg, hist, np.zeros(7, dtype=np.int64),
            np.zeros(N, dtype=np.int64), np.zeros(N, dtype=np.int8),
            np.zeros(2 * N, dtype=np.int64), (s_cache if track else None), int(N + 1))
    finally:
        _grid_fill(grid, chain, lbits, 0)
    random.setstate(state_from_mt(mt, st, rs))

    new_chain = [tuple(row) for row in chain.tolist()]
    state.chain = new_chain
    state.occ = set(new_chain)
    state.m = int(scal[0])
    state.n_bend = int(scal[1])
    if configuration_dependent:
        state.coordination_hist = tuple(hist.tolist())
    if any_acc:
        state.E = float(E_last)
    replica.local_acc = int(scal[2])
    replica.local_prop = int(scal[3])


# ---------------------------------------------------------------------------
# Start-up self-check: Python vs Numba in lockstep on this machine
# ---------------------------------------------------------------------------

def _snapshot(rep) -> tuple:
    s = rep.state
    return (tuple(s.chain), frozenset(s.occ), s.m, s.E, s.n_bend,
            None if s.coordination_hist is None else tuple(s.coordination_hist),
            rep.local_acc, rep.local_prop, rep.move_counters.tobytes())


def selfcheck(R, model_name: str, params, Tref: float, Tscale: float, kappa_bend: float,
              N: int, temps, steps: int = 2000, warm: int = 1000, seed: int = 20261102) -> Dict[str, Any]:
    """Compare the Python and Numba sweeps from identical states on THIS machine.

    For the lowest, middle and highest temperature: grow a chain with the Python
    sweep (``warm`` steps), then run ``steps`` more steps with each kernel from
    copies of the same state and RNG state, and require identical state,
    counters and RNG state.  The global ``random`` state is restored.
    """
    import copy
    info: Dict[str, Any] = {"ok": False, "steps": int(steps), "temperatures": []}
    if not NUMBA_AVAILABLE:
        info["reason"] = "numba not importable"
        return info
    temps = [float(t) for t in temps]
    picks = sorted({temps[0], temps[len(temps) // 2], temps[-1]})
    saved = random.getstate()
    try:
        for T in picks:
            random.seed(seed)
            st0 = R.ChainState.initial_straight(int(N), T, model_name, params, Tref, Tscale)
            rep0 = R.Replica(T=T, state=st0)
            R.mc_sweep(rep0, int(warm), model_name, params, Tref, Tscale, kappa_bend, int(N))
            a, b = copy.deepcopy(rep0), copy.deepcopy(rep0)
            s0 = random.getstate()
            R.mc_sweep(a, int(steps), model_name, params, Tref, Tscale, kappa_bend, int(N))
            sa = random.getstate()
            random.setstate(s0)
            mc_sweep_numba(R, b, int(steps), model_name, params, Tref, Tscale, kappa_bend, int(N))
            sb = random.getstate()
            same = (_snapshot(a) == _snapshot(b)) and (sa == sb)
            info["temperatures"].append({"T": T, "identical": bool(same)})
            if not same:
                info["reason"] = f"python and numba sweeps differ at T={T}"
                return info
        info["ok"] = True
        return info
    finally:
        random.setstate(saved)
