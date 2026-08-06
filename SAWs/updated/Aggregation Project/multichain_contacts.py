#!/usr/bin/env python3
"""
Multi-chain lattice contact counting (Stage 1).

Provides the authoritative full contact-counting oracle and an O(moved) local
delta-contact algorithm for the multi-chain lattice-polymer model.  Both paths
count RAW lattice contacts and classify them as intrachain (same chain,
nonbonded) or interchain (different chains):

    * examine the six wrapped nearest-neighbour sites of every occupied bead;
    * retrieve the neighbouring bead owner from ``site_owner``;
    * canonicalize the bead pair as ``(min_id, max_id)`` and count it exactly
      once;
    * classify the pair as intra- or interchain;
    * exclude covalently bonded same-chain neighbours (|monomer_i - monomer_j|
      == 1).

The single molecular-to-lattice contact offset is NOT applied.  Counts are
integers throughout.  This mirrors the single-chain ``contact_count`` in
``remd_uniform_chain_2_new.py`` but returns the intra/inter split needed for the
aggregation reduced potential.
"""
from __future__ import annotations

import os
from typing import Dict, Iterable, Tuple

import numpy as np

from multichain_state import (
    NN6,
    ContactCounts,
    MultiChainState,
)

Site = Tuple[int, int, int]


def _parse_bool_env(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    v = raw.strip().lower()
    if v in ("0", "false", "no", "off", ""):
        return False
    if v in ("1", "true", "yes", "on"):
        return True
    return default


# Optional development cross-check: when MULTICHAIN_DEBUG_CONTACTS is truthy the
# MC sweep asserts cached counts == full_contact_counts after every accepted
# move (analogous to ISAW_DEBUG_CONTACTS in the single-chain reference).
DEBUG_CONTACTS = _parse_bool_env("MULTICHAIN_DEBUG_CONTACTS", False)


# ---------------------------------------------------------------------------
# Full oracle
# ---------------------------------------------------------------------------

def full_contact_counts_from_map(
    coords_unwrapped: np.ndarray, site_owner: Dict[Site, int], box_size: int,
) -> ContactCounts:
    """Authoritative contact counts from an occupancy map.

    ``coords_unwrapped`` supplies the chain length (its second axis); the actual
    geometry used is the wrapped ``site_owner`` map.  Every occupied bead probes
    its six wrapped nearest neighbours; each unordered pair is counted once
    (``g2 > g``), classified intra/inter, and bonded same-chain neighbours are
    excluded.
    """
    N = int(coords_unwrapped.shape[1])
    L = int(box_size)
    intra = 0
    inter = 0
    for site, gid in site_owner.items():
        gid = int(gid)
        ci = gid // N
        mi = gid % N
        sx, sy, sz = site
        for dx, dy, dz in NN6:
            s2 = ((sx + dx) % L, (sy + dy) % L, (sz + dz) % L)
            g2 = site_owner.get(s2)
            if g2 is None:
                continue
            g2 = int(g2)
            if g2 <= gid:
                continue  # count each unordered pair exactly once
            if ci == (g2 // N):
                if abs(mi - (g2 % N)) == 1:
                    continue  # covalently bonded: excluded from intrachain
                intra += 1
            else:
                inter += 1
    return ContactCounts(int(intra), int(inter))


def full_contact_counts(state: MultiChainState) -> ContactCounts:
    """Authoritative contact counts for a :class:`MultiChainState`."""
    return full_contact_counts_from_map(
        state.coords_unwrapped, state.site_owner, state.box_size)


def full_contacts_split(
    coords_unwrapped: np.ndarray, site_owner: Dict[Site, int], box_size: int,
) -> Tuple[np.ndarray, int]:
    """Authoritative (intra_by_chain, inter_total) from an occupancy map.

    ``intra_by_chain`` is an ``(M,)`` int64 array whose entry ``alpha`` is the
    number of unique nonbonded nearest-neighbour intrachain contacts on chain
    ``alpha``; ``inter_total`` is the total interchain contact count.  By
    construction ``int(intra_by_chain.sum()) == full_contact_counts_from_map(
    ...).intra`` and the returned ``inter_total`` equals its ``.inter`` -- this
    is the single per-chain-resolved pass that seeds
    :attr:`MultiChainState.intra_contacts_by_chain`.
    """
    M = int(coords_unwrapped.shape[0])
    N = int(coords_unwrapped.shape[1])
    L = int(box_size)
    intra_by_chain = np.zeros(M, dtype=np.int64)
    inter = 0
    for site, gid in site_owner.items():
        gid = int(gid)
        ci = gid // N
        mi = gid % N
        sx, sy, sz = site
        for dx, dy, dz in NN6:
            s2 = ((sx + dx) % L, (sy + dy) % L, (sz + dz) % L)
            g2 = site_owner.get(s2)
            if g2 is None:
                continue
            g2 = int(g2)
            if g2 <= gid:
                continue  # count each unordered pair exactly once
            if ci == (g2 // N):
                if abs(mi - (g2 % N)) == 1:
                    continue  # covalently bonded: excluded from intrachain
                intra_by_chain[ci] += 1
            else:
                inter += 1
    return intra_by_chain, int(inter)


def full_intra_contacts_by_chain(
    coords_unwrapped: np.ndarray, site_owner: Dict[Site, int], box_size: int,
) -> np.ndarray:
    """Per-chain intrachain contact counts ``(M,)`` (see :func:`full_contacts_split`)."""
    return full_contacts_split(coords_unwrapped, site_owner, box_size)[0]


def full_intra_contacts_by_chain_state(state: MultiChainState) -> np.ndarray:
    """Per-chain intrachain contact counts for a :class:`MultiChainState`."""
    return full_intra_contacts_by_chain(
        state.coords_unwrapped, state.site_owner, state.box_size)


# ---------------------------------------------------------------------------
# Local delta-contact algorithm (production path)
# ---------------------------------------------------------------------------

def _incident_counts(
    occ: Dict[Site, int],
    moved_sites: Dict[int, Site],
    N: int,
    L: int,
) -> Tuple[int, int]:
    """(intra, inter) contacts incident to the moved beads within ``occ``.

    A contact is "incident" if at least one endpoint is a moved bead.  Each
    canonical unordered pair is counted once even when both endpoints are moved
    beads (deduplicated via ``seen``), so a shared contact between two moved
    beads is not double counted.  Bonded same-chain neighbours are excluded.
    """
    seen: set = set()
    intra = 0
    inter = 0
    for gid, (sx, sy, sz) in moved_sites.items():
        gid = int(gid)
        ci = gid // N
        mi = gid % N
        for dx, dy, dz in NN6:
            s2 = ((sx + dx) % L, (sy + dy) % L, (sz + dz) % L)
            g2 = occ.get(s2)
            if g2 is None:
                continue
            g2 = int(g2)
            if g2 == gid:
                continue
            pair = (gid, g2) if gid < g2 else (g2, gid)
            if pair in seen:
                continue
            seen.add(pair)
            if ci == (g2 // N):
                if abs(mi - (g2 % N)) == 1:
                    continue  # bonded excluded
                intra += 1
            else:
                inter += 1
    return intra, inter


def _wrapped_site(state: MultiChainState, global_id: int) -> Site:
    N = state.chain_length
    L = int(state.box_size)
    c = int(global_id) // N
    i = int(global_id) % N
    x, y, z = state.coords_unwrapped[c, i]
    return (int(x) % L, int(y) % L, int(z) % L)


def delta_contacts(
    state: MultiChainState,
    moved_ids: Iterable[int],
    new_sites: Dict[int, Site],
) -> Tuple[int, int]:
    """Change in (intra, inter) contacts if ``moved_ids`` move to ``new_sites``.

    ``new_sites`` maps each moved global ID to its NEW wrapped site.  The move
    is assumed geometrically valid (new sites mutually distinct and disjoint
    from stationary beads); the caller validates this first.  Returns
    ``(delta_intra, delta_inter) = counts(after) - counts(before)``.

    Only contacts incident to a moved bead can change, so this is O(len(moved))
    rather than O(M*N).
    """
    N = state.chain_length
    L = int(state.box_size)
    moved_ids = [int(g) for g in moved_ids]

    old_sites: Dict[int, Site] = {g: _wrapped_site(state, g) for g in moved_ids}

    before_intra, before_inter = _incident_counts(state.site_owner, old_sites, N, L)

    # Trial occupancy: vacate all old moved sites, then place all new sites.
    occ_new = dict(state.site_owner)
    for g in moved_ids:
        occ_new.pop(old_sites[g], None)
    for g in moved_ids:
        occ_new[new_sites[g]] = g

    after_intra, after_inter = _incident_counts(occ_new, new_sites, N, L)

    return (after_intra - before_intra, after_inter - before_inter)


def apply_moved_beads(
    state: MultiChainState,
    moved: Dict[int, Tuple[int, int, int]],
    delta: Tuple[int, int],
) -> None:
    """Apply new UNWRAPPED positions for moved beads and update caches in place.

    ``moved`` maps each moved global ID to its NEW unwrapped coordinate.
    Updates ``coords_unwrapped``, ``site_owner`` (removing old wrapped sites and
    inserting new ones), the cached contact counts by ``delta``, and the
    per-chain intrachain cache ``intra_contacts_by_chain``.

    Every proposal moves beads of a SINGLE chain, so the whole intrachain delta
    ``delta[0]`` is attributed to that chain (inferred from the moved beads'
    global IDs).  The invariant ``intra_contacts_by_chain.sum() == counts.intra``
    is therefore preserved by construction.
    """
    N = state.chain_length
    L = int(state.box_size)
    # Remove old wrapped sites first (so a bead moving onto another moved bead's
    # vacated site does not clobber it).
    for g in moved:
        state.site_owner.pop(_wrapped_site(state, g), None)
    moved_chain = None
    for g, new_pos in moved.items():
        c = int(g) // N
        if moved_chain is None:
            moved_chain = c
        i = int(g) % N
        state.coords_unwrapped[c, i, 0] = int(new_pos[0])
        state.coords_unwrapped[c, i, 1] = int(new_pos[1])
        state.coords_unwrapped[c, i, 2] = int(new_pos[2])
        site = (int(new_pos[0]) % L, int(new_pos[1]) % L, int(new_pos[2]) % L)
        state.site_owner[site] = int(g)
    state.counts.intra = int(state.counts.intra) + int(delta[0])
    state.counts.inter = int(state.counts.inter) + int(delta[1])
    if moved_chain is not None:
        state.intra_contacts_by_chain[moved_chain] = (
            int(state.intra_contacts_by_chain[moved_chain]) + int(delta[0]))


# ---------------------------------------------------------------------------
# Per-monomer contact degree and LOCAL cooperative sum
# ---------------------------------------------------------------------------
# The contact degree k_i of bead i is the number of its six wrapped nearest
# neighbour sites occupied by a bead that is NOT one of its covalent neighbours.
# Both intrachain and interchain neighbours count, so an interchain neighbour
# raises the same degree that an intrachain neighbour does and the two boost each
# other cooperatively.  The degree is tied to the authoritative pair counts by
#
#     sum_i k_i == 2 * (m_intra + m_inter)
#
# because every counted contact contributes one incidence to each of its two
# endpoints.  This identity is asserted in the debug path.
#
# The local cooperative sum is
#
#     S(X; q_sat) = sum_i g(k_i),   g(k) = kappa^2 / (1 + (kappa/q_sat)^2),
#                                   kappa = k / 2,
#
# an intensive per-bead contact density kappa (contacts per monomer, the local
# analogue of q = m/N).  The factor 1/2 is fixed, not free: when every bead has
# the same degree k_i = 2m/N, kappa = q and S = N*q^2/(1 + (q/q_sat)^2), so
# A0 * S reproduces the fitted single-chain cooperative term EXACTLY with the
# same A0 and q_sat.  S is a sum of terms that each depend only on one bead and
# its six neighbours, so it is strictly local and exactly additive over
# non-interacting subsystems.


def contact_degrees(state: MultiChainState) -> np.ndarray:
    """Nonbonded contact degree k_i of every bead, shape ``(M*N,)`` (oracle).

    Indexed by global bead ID.  Counts occupied wrapped nearest neighbours that
    are not covalently bonded, without regard to chain identity, so intra- and
    interchain neighbours contribute equally.  Satisfies
    ``contact_degrees(state).sum() == 2 * (counts.intra + counts.inter)``.
    """
    N = state.chain_length
    L = int(state.box_size)
    deg = np.zeros(int(state.n_chains) * int(N), dtype=np.int64)
    for site, gid in state.site_owner.items():
        gid = int(gid)
        deg[gid] = _degree_at(state.site_owner, gid, site, N, L)
    return deg


def _degree_at(occ: Dict[Site, int], gid: int, site: Site, N: int, L: int) -> int:
    """Nonbonded contact degree of bead ``gid`` sitting at ``site`` within ``occ``."""
    ci = gid // N
    mi = gid % N
    sx, sy, sz = site
    k = 0
    for dx, dy, dz in NN6:
        g2 = occ.get(((sx + dx) % L, (sy + dy) % L, (sz + dz) % L))
        if g2 is None:
            continue
        g2 = int(g2)
        if g2 == gid:
            continue
        if g2 // N == ci and abs(mi - (g2 % N)) == 1:
            continue  # covalently bonded: never a contact
        k += 1
    return k


def cooperative_g(k, q_sat: float) -> float:
    """``g(k) = kappa^2 / (1 + (kappa/q_sat)^2)`` with ``kappa = k/2``.

    Bounded above by ``q_sat**2``, so the cooperative energy per bead saturates
    exactly as the single-chain model's does.  ``q_sat > 0`` is the caller's
    responsibility (validated once when the model parameters are resolved).
    """
    kappa = 0.5 * float(k)
    r = kappa / float(q_sat)
    return (kappa * kappa) / (1.0 + r * r)


def _g_table(q_sat: float) -> Tuple[float, ...]:
    """``g(k)`` for the only degrees a cubic-lattice chain bead can have (0..6)."""
    return tuple(cooperative_g(k, q_sat) for k in range(7))


def degree_histogram(state: MultiChainState) -> Tuple[int, ...]:
    """Counts of beads at each possible contact degree 0..6 (oracle; O(M*N)).

    Order independent by construction: only integer counts are accumulated, so
    the result cannot depend on the iteration order of ``site_owner``.  This
    matters because the multiprocessing path REBUILDS ``site_owner`` from
    coordinates while the serial path updates it incrementally, giving the two
    dicts different iteration orders for the same physical state.
    """
    N = state.chain_length
    L = int(state.box_size)
    hist = [0] * 7
    for site, gid in state.site_owner.items():
        hist[_degree_at(state.site_owner, int(gid), site, N, L)] += 1
    return tuple(hist)


def cooperative_sum(state: MultiChainState, q_sat: float) -> float:
    """``S = sum_i g(k_i)`` over all beads (oracle; O(M*N)).

    Evaluated as ``sum_k g(k) * n_k`` over the seven possible degrees in fixed
    ascending order, so the floating-point result is bit-identical for any
    ``site_owner`` iteration order (serial vs multiprocessing).
    """
    g = _g_table(q_sat)
    hist = degree_histogram(state)
    total = 0.0
    for k in range(7):
        total += g[k] * hist[k]
    return float(total)


def _affected_beads(
    occ: Dict[Site, int], sites: Dict[int, Site], N: int, L: int,
) -> set:
    """Global IDs whose degree can change: the moved beads and their neighbours."""
    out = set(int(g) for g in sites)
    for s in sites.values():
        sx, sy, sz = s
        for dx, dy, dz in NN6:
            g2 = occ.get(((sx + dx) % L, (sy + dy) % L, (sz + dz) % L))
            if g2 is not None:
                out.add(int(g2))
    return out


def delta_cooperative_sum(
    state: MultiChainState,
    moved_ids: Iterable[int],
    new_sites: Dict[int, Site],
    q_sat: float,
) -> float:
    """Change in ``S`` if ``moved_ids`` move to ``new_sites`` (O(len(moved))).

    A bead's degree can change only if it moved or if one of its six neighbour
    sites gained or lost an occupant, so only the union of the moved beads and
    the neighbours of their OLD and NEW sites is re-evaluated.  Returns
    ``S(after) - S(before)``; the move is assumed geometrically valid (the caller
    validates it first), exactly as for :func:`delta_contacts`.
    """
    g = _g_table(q_sat)
    N = state.chain_length
    L = int(state.box_size)
    moved_ids = [int(x) for x in moved_ids]
    old_sites: Dict[int, Site] = {x: _wrapped_site(state, x) for x in moved_ids}

    occ_new = dict(state.site_owner)
    for x in moved_ids:
        occ_new.pop(old_sites[x], None)
    for x in moved_ids:
        occ_new[new_sites[x]] = x

    affected = _affected_beads(state.site_owner, old_sites, N, L)
    affected |= _affected_beads(occ_new, new_sites, N, L)

    # Accumulate INTEGER per-degree count changes, then contract with g in fixed
    # ascending degree order.  Only integers are summed over the (unordered) set,
    # so the float result cannot depend on set iteration order -- which is what
    # keeps serial and multiprocessing runs bit-identical.
    dn = [0] * 7
    for gid in affected:
        home = _wrapped_site(state, gid)
        dn[_degree_at(state.site_owner, gid, old_sites.get(gid, home), N, L)] -= 1
        dn[_degree_at(occ_new, gid, new_sites.get(gid, home), N, L)] += 1

    delta = 0.0
    for k in range(7):
        if dn[k]:
            delta += g[k] * dn[k]
    return float(delta)


# ---------------------------------------------------------------------------
# Interchain pair aggregation (for the chain-cluster graph)
# ---------------------------------------------------------------------------

def interchain_pair_counts(state: MultiChainState) -> Dict[Tuple[int, int], int]:
    """Number of interchain contacts per unordered chain pair.

    Returns a dict mapping ``(chain_a, chain_b)`` with ``chain_a < chain_b`` to
    the count of interchain nearest-neighbour contacts between those two chains.
    Only chain pairs with at least one interchain contact appear.
    """
    N = state.chain_length
    L = int(state.box_size)
    out: Dict[Tuple[int, int], int] = {}
    for site, gid in state.site_owner.items():
        gid = int(gid)
        ci = gid // N
        sx, sy, sz = site
        for dx, dy, dz in NN6:
            s2 = ((sx + dx) % L, (sy + dy) % L, (sz + dz) % L)
            g2 = state.site_owner.get(s2)
            if g2 is None:
                continue
            g2 = int(g2)
            if g2 <= gid:
                continue
            cj = g2 // N
            if ci == cj:
                continue
            key = (ci, cj) if ci < cj else (cj, ci)
            out[key] = out.get(key, 0) + 1
    return out


# ---------------------------------------------------------------------------
# Debug assertion
# ---------------------------------------------------------------------------

def assert_counts_match(state: MultiChainState, context: str = "") -> None:
    """Assert cached counts equal the full oracle (used under DEBUG_CONTACTS).

    Also verifies the per-chain intrachain cache: it must equal a full per-chain
    recount element-by-element (which implies ``sum == counts.intra``).
    """
    intra_by_chain, inter = full_contacts_split(
        state.coords_unwrapped, state.site_owner, state.box_size)
    recount_intra = int(intra_by_chain.sum())
    if (int(state.counts.intra) != recount_intra
            or int(state.counts.inter) != int(inter)):
        raise AssertionError(
            f"cached counts (intra={state.counts.intra}, "
            f"inter={state.counts.inter}) != full recount "
            f"(intra={recount_intra}, inter={inter}){' ' + context if context else ''}")
    cached = np.asarray(state.intra_contacts_by_chain, dtype=np.int64)
    if not np.array_equal(cached, intra_by_chain):
        raise AssertionError(
            f"cached intra_contacts_by_chain {cached.tolist()} != full per-chain "
            f"recount {intra_by_chain.tolist()}{' ' + context if context else ''}")
    # Degree/pair-count identity: every contact contributes one incidence to each
    # of its two endpoints, so sum_i k_i == 2 * m_total for ANY valid state.
    deg_sum = int(contact_degrees(state).sum())
    if deg_sum != 2 * (recount_intra + int(inter)):
        raise AssertionError(
            f"sum of contact degrees {deg_sum} != 2*m_total "
            f"{2 * (recount_intra + int(inter))}"
            f"{' ' + context if context else ''}")
