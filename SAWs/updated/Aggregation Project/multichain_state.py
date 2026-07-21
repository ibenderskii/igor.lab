#!/usr/bin/env python3
"""
Multi-chain lattice-polymer state and geometry (Stage 1).

This module holds the authoritative state representation for M identical
connected self-avoiding chains, each with N beads, in a periodic cubic lattice
box of side length L.  It is the multi-chain analogue of the ``ChainState`` /
occupancy machinery in ``remd_uniform_chain_2_new.py`` (the validated
single-chain reference), generalized to:

* **Unwrapped** integer coordinates of shape ``(M, N, 3)`` as the AUTHORITATIVE
  representation for chain connectivity and per-chain geometry (radius of
  gyration).  A chain may extend beyond ``[0, L)``; it is never wrapped for
  geometry, so Rg is correct even when a chain crosses a periodic boundary.
* **Wrapped** coordinates ``coord % L`` used for occupancy, excluded-volume
  checks, contact counting, and interchain collisions.
* An occupancy mapping ``site_owner`` from a wrapped lattice site ``(x, y, z)``
  in ``[0, L)^3`` to a UNIQUE global bead ID (not merely a set), so that a
  neighbor lookup can classify a contact as intrachain or interchain.

Global bead-ID convention (frozen)::

    global_id = chain_index * N + monomer_index
    chain_index = global_id // N ;  monomer_index = global_id % N

Scientific conventions match ``aggregation_model_spec_v1.json``.  The single
molecular-to-lattice contact offset used in the single-chain fit is NOT applied
here; raw lattice geometry is authoritative.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Tuple

import numpy as np

import isaw_contact_observables as ico  # radius_of_gyration_squared (unwrapped)
from lattice_bending import BEND_DEFINITION, count_bends_multichain

# Six nearest-neighbour displacement vectors on the simple-cubic lattice.
NN6: Tuple[Tuple[int, int, int], ...] = (
    (1, 0, 0), (-1, 0, 0),
    (0, 1, 0), (0, -1, 0),
    (0, 0, 1), (0, 0, -1),
)

# Minimum periodic box size.  L < 3 makes nearest-neighbour wrapping ambiguous
# (a bead's +d and -d neighbour sites coincide mod L), so such boxes are
# rejected everywhere in this package.
MIN_BOX_SIZE = 3


class MultiChainStateError(ValueError):
    """Raised when a multi-chain state or its geometry fails validation."""


class PlacementError(RuntimeError):
    """Raised when a dispersed initial configuration cannot be placed."""


# ---------------------------------------------------------------------------
# Contact counts
# ---------------------------------------------------------------------------

@dataclass
class ContactCounts:
    """Raw lattice contact counts (authoritative).

    ``intra`` = unique nonbonded nearest-neighbour contacts between beads on the
    same chain; ``inter`` = unique nearest-neighbour contacts between beads on
    different chains.  Both are counted exactly once per unordered global-ID
    pair.  Integer throughout.
    """
    intra: int
    inter: int

    @property
    def total(self) -> int:
        return int(self.intra) + int(self.inter)

    def copy(self) -> "ContactCounts":
        return ContactCounts(int(self.intra), int(self.inter))

    def as_tuple(self) -> Tuple[int, int]:
        return (int(self.intra), int(self.inter))

    @classmethod
    def from_tuple(cls, t) -> "ContactCounts":
        return cls(int(t[0]), int(t[1]))


# ---------------------------------------------------------------------------
# Coordinate wrapping / occupancy
# ---------------------------------------------------------------------------

def wrap_coordinate(coord, box_size: int) -> Tuple[int, int, int]:
    """Wrap a single integer 3-vector into ``[0, box_size)^3``.

    Uses Python's floor-modulo so negative coordinates wrap correctly.
    """
    L = int(box_size)
    x, y, z = int(coord[0]), int(coord[1]), int(coord[2])
    return (x % L, y % L, z % L)


def wrapped_coordinates(coords_unwrapped: np.ndarray, box_size: int) -> np.ndarray:
    """Return wrapped coordinates ``coords_unwrapped % L`` (same shape, int64).

    Works for any array whose last axis is length 3.  Python/NumPy modulo of a
    negative integer already yields a value in ``[0, L)``.
    """
    L = int(box_size)
    arr = np.asarray(coords_unwrapped)
    if arr.shape[-1] != 3:
        raise MultiChainStateError(
            f"coordinates last axis must be length 3; got shape {arr.shape}")
    return np.mod(arr.astype(np.int64), L)


def build_site_owner(coords_unwrapped: np.ndarray, box_size: int) -> dict:
    """Build the wrapped-site -> global-ID occupancy map.

    Raises :class:`MultiChainStateError` on any double occupancy (two beads on
    the same wrapped site), because the map must be a bijection between occupied
    beads and occupied sites for the contact classifier to be well defined.
    """
    coords = _validate_coords_array(coords_unwrapped)
    M, N, _ = coords.shape
    L = int(box_size)
    _check_box_size(L)
    wrapped = np.mod(coords, L)
    site_owner: dict = {}
    for c in range(M):
        for i in range(N):
            gid = c * N + i
            site = (int(wrapped[c, i, 0]), int(wrapped[c, i, 1]),
                    int(wrapped[c, i, 2]))
            other = site_owner.get(site)
            if other is not None:
                raise MultiChainStateError(
                    f"double occupancy at wrapped site {site}: beads "
                    f"{other} and {gid} (self-avoidance / excluded volume "
                    f"violated)")
            site_owner[site] = gid
    return site_owner


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

@dataclass
class MultiChainState:
    """Mutable multi-chain Monte Carlo state.

    Attributes
    ----------
    coords_unwrapped : np.ndarray
        Authoritative integer coordinates, shape ``(M, N, 3)``.
    site_owner : dict
        Wrapped-site ``(x, y, z)`` -> global bead ID.  Kept in sync with
        ``coords_unwrapped``; the single source of occupancy truth.
    counts : ContactCounts
        Cached raw lattice contact counts (intra, inter).
    box_size : int
        Periodic cubic box side length L.
    n_bend : int
        Cached total number of 90-degree turns summed over ALL chains (the
        optional fixed bending penalty depends on it).  Maintained incrementally
        by the moves exactly like ``counts``; defaults to 0 so states built by
        older code or a straight chain keep their previous behaviour.
    """
    coords_unwrapped: np.ndarray
    site_owner: dict
    counts: ContactCounts
    box_size: int
    n_bend: int = 0

    # -- shape helpers ------------------------------------------------------
    @property
    def n_chains(self) -> int:
        return int(self.coords_unwrapped.shape[0])

    @property
    def chain_length(self) -> int:
        return int(self.coords_unwrapped.shape[1])

    @property
    def n_beads(self) -> int:
        return self.n_chains * self.chain_length

    @property
    def volume_fraction(self) -> float:
        L = int(self.box_size)
        return float(self.n_beads) / float(L ** 3)

    def wrapped(self) -> np.ndarray:
        """Wrapped coordinates (M, N, 3) int64."""
        return np.mod(self.coords_unwrapped.astype(np.int64), int(self.box_size))

    def chain_of(self, global_id: int) -> int:
        return int(global_id) // self.chain_length

    def monomer_of(self, global_id: int) -> int:
        return int(global_id) % self.chain_length

    def copy(self) -> "MultiChainState":
        """Deep-ish copy safe for independent MC evolution (Windows spawn safe)."""
        return MultiChainState(
            coords_unwrapped=self.coords_unwrapped.copy(),
            site_owner=dict(self.site_owner),
            counts=self.counts.copy(),
            box_size=int(self.box_size),
            n_bend=int(self.n_bend),
        )


def make_state(coords_unwrapped: np.ndarray, box_size: int) -> MultiChainState:
    """Construct a fully consistent state from unwrapped coordinates.

    Builds the occupancy map (raising on double occupancy) and computes the
    authoritative contact counts via the full oracle.  This is the canonical
    constructor used by tests and initializers.
    """
    coords = _validate_coords_array(coords_unwrapped)
    L = int(box_size)
    _check_box_size(L)
    site_owner = build_site_owner(coords, L)
    # Local import breaks the module-load cycle (contacts imports this module).
    from multichain_contacts import full_contact_counts_from_map
    counts = full_contact_counts_from_map(coords, site_owner, L)
    return MultiChainState(
        coords_unwrapped=coords, site_owner=site_owner, counts=counts,
        box_size=L, n_bend=total_bend_count(coords),
    )


def total_bend_count(coords_unwrapped: np.ndarray) -> int:
    """Total 90-degree turns over all chains (authoritative full recount).

    Uses the UNWRAPPED coordinates, where every chain is a contiguous unit-bond
    walk, so a chain crossing a periodic boundary is still counted correctly
    (its unwrapped bonds never wrap).  This is the multi-chain analogue of
    ``count_bends`` and the initializer for :attr:`MultiChainState.n_bend`.
    """
    coords = _validate_coords_array(coords_unwrapped)
    return int(count_bends_multichain(coords))


def assert_bends_match(state: MultiChainState, context: str = "") -> None:
    """Assert the cached ``n_bend`` equals a full recount (used under debug mode)."""
    recount = total_bend_count(state.coords_unwrapped)
    if int(state.n_bend) != int(recount):
        raise AssertionError(
            f"cached n_bend={state.n_bend} != full recount {recount}"
            f"{' ' + context if context else ''}")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _check_box_size(box_size: int) -> int:
    L = int(box_size)
    if L < MIN_BOX_SIZE:
        raise MultiChainStateError(
            f"box_size L={L} < {MIN_BOX_SIZE}: periodic boxes smaller than "
            f"{MIN_BOX_SIZE} create ambiguous nearest-neighbour wrapping and "
            f"are rejected")
    return L


def _validate_coords_array(coords_unwrapped) -> np.ndarray:
    """Return an int64 (M, N, 3) coordinate array or raise."""
    arr = np.asarray(coords_unwrapped)
    if arr.dtype.kind == "f":
        if not np.all(np.isfinite(arr)):
            raise MultiChainStateError("coordinates contain NaN or infinity")
        rounded = np.rint(arr)
        if not np.all(rounded == arr):
            raise MultiChainStateError(
                "coordinates are not exactly integral (fractional lattice "
                "coordinates are rejected, not truncated)")
        arr = rounded
    elif arr.dtype.kind not in ("i", "u", "b"):
        raise MultiChainStateError(
            f"coordinates have unsupported dtype {arr.dtype!r}")
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise MultiChainStateError(
            f"coordinates must have shape (M, N, 3); got shape {arr.shape}")
    if arr.shape[0] < 1 or arr.shape[1] < 2:
        raise MultiChainStateError(
            f"need at least 1 chain of length >= 2; got shape {arr.shape}")
    return arr.astype(np.int64)


def validate_connectivity(coords_unwrapped: np.ndarray) -> None:
    """Verify every covalent bond is exactly one lattice step (unwrapped).

    Uses UNWRAPPED coordinates: consecutive bonded beads in a chain must differ
    by unit Manhattan distance.  Raises :class:`MultiChainStateError` otherwise.
    """
    coords = _validate_coords_array(coords_unwrapped)
    M, N, _ = coords.shape
    for c in range(M):
        d = np.abs(np.diff(coords[c], axis=0)).sum(axis=1)
        bad = np.nonzero(d != 1)[0]
        if bad.size:
            i = int(bad[0])
            raise MultiChainStateError(
                f"chain {c} bond {i}-{i + 1} has Manhattan length {int(d[i])} "
                f"!= 1 (chain is not a connected lattice walk in unwrapped "
                f"coordinates)")


def validate_unique_occupancy(coords_unwrapped: np.ndarray, box_size: int) -> None:
    """Verify no two beads occupy the same WRAPPED lattice site."""
    coords = _validate_coords_array(coords_unwrapped)
    L = _check_box_size(box_size)
    wrapped = np.mod(coords, L).reshape(-1, 3)
    uniq = np.unique(wrapped, axis=0)
    if uniq.shape[0] != wrapped.shape[0]:
        n_dup = wrapped.shape[0] - uniq.shape[0]
        raise MultiChainStateError(
            f"excluded-volume violated: {n_dup} wrapped site(s) occupied by "
            f"more than one bead")


def validate_state(state: MultiChainState, *, check_contacts: bool = True) -> None:
    """Validate a complete :class:`MultiChainState`.

    Verifies:
      * correct array shape and integer coordinates;
      * every covalent bond is one lattice step in unwrapped coordinates;
      * no two beads occupy the same wrapped site;
      * ``site_owner`` agrees EXACTLY with the coordinates (same keys, same
        global IDs, and a bijection);
      * the periodic box size is >= 3;
      * (when ``check_contacts``) the cached contact counts match a full
        contact recount.

    Raises :class:`MultiChainStateError` on the first violation.
    """
    coords = _validate_coords_array(state.coords_unwrapped)
    L = _check_box_size(state.box_size)
    M, N, _ = coords.shape

    validate_connectivity(coords)
    validate_unique_occupancy(coords, L)

    # site_owner must agree exactly with the coordinates.
    expected = build_site_owner(coords, L)
    if len(state.site_owner) != len(expected):
        raise MultiChainStateError(
            f"site_owner has {len(state.site_owner)} entries; coordinates imply "
            f"{len(expected)} occupied sites")
    for site, gid in expected.items():
        got = state.site_owner.get(site)
        if got is None:
            raise MultiChainStateError(
                f"site_owner missing occupied site {site} (bead {gid})")
        if int(got) != int(gid):
            raise MultiChainStateError(
                f"site_owner disagrees at {site}: has bead {got}, coordinates "
                f"imply bead {gid}")
    # global IDs must be a permutation of 0..M*N-1 (bijection guard).
    ids = sorted(int(v) for v in state.site_owner.values())
    if ids != list(range(M * N)):
        raise MultiChainStateError(
            "site_owner global IDs are not a permutation of 0..M*N-1")

    if check_contacts:
        from multichain_contacts import full_contact_counts
        recount = full_contact_counts(state)
        if (int(state.counts.intra) != int(recount.intra)
                or int(state.counts.inter) != int(recount.inter)):
            raise MultiChainStateError(
                f"cached contact counts (intra={state.counts.intra}, "
                f"inter={state.counts.inter}) disagree with full recount "
                f"(intra={recount.intra}, inter={recount.inter})")
        # The cached bending count is maintained the same way as the contacts and
        # is verified against a full recount here.  It tracks pure geometry, so it
        # is checked for every run regardless of the (sampling-only) kappa_bend.
        recount_bends = total_bend_count(coords)
        if int(state.n_bend) != int(recount_bends):
            raise MultiChainStateError(
                f"cached bend count n_bend={state.n_bend} disagrees with full "
                f"recount {recount_bends}")


# ---------------------------------------------------------------------------
# Per-chain geometry (unwrapped coordinates are authoritative)
# ---------------------------------------------------------------------------

def chain_radius_of_gyration_squared(state: MultiChainState, chain_index: int
                                     ) -> float:
    """Rg^2 (lattice units) of one chain from UNWRAPPED coordinates."""
    coords = state.coords_unwrapped[int(chain_index)]
    return float(ico.radius_of_gyration_squared(coords))


def per_chain_rg2(state: MultiChainState) -> np.ndarray:
    """Rg^2 for every chain, shape (M,), from unwrapped coordinates."""
    M = state.n_chains
    out = np.empty(M, dtype=np.float64)
    for c in range(M):
        out[c] = ico.radius_of_gyration_squared(state.coords_unwrapped[c])
    return out


def per_chain_rg(state: MultiChainState) -> np.ndarray:
    """Rg for every chain, shape (M,), from unwrapped coordinates."""
    return np.sqrt(per_chain_rg2(state))


# ---------------------------------------------------------------------------
# Unwrapped-coordinate canonicalization
# ---------------------------------------------------------------------------

def canonicalize_chain_coordinates(state: MultiChainState, chain_index: int
                                   ) -> None:
    """Shift one chain by integer box multiples so bead 0 lies in the primary box.

    A whole-chain translation lets the UNWRAPPED coordinates of a chain perform
    an unrestricted random walk even though the physical WRAPPED configuration
    stays inside the periodic box.  This rebases the chain by an integer multiple
    of the box length per axis so that bead 0 lies in ``[0, L)^3`` again, keeping
    unwrapped magnitudes bounded (so int32 snapshots never overflow) without
    changing any physical quantity.

    Because the shift is a whole multiple of ``box_size`` applied to the entire
    chain, it preserves wrapped coordinates, occupancy, ``site_owner``, chain
    connectivity, all contact counts, per-chain radius of gyration, and the
    reduced potential.  ``site_owner`` therefore needs no rebuild.
    """
    c = int(chain_index)
    L = int(state.box_size)
    shift_boxes = np.floor_divide(state.coords_unwrapped[c, 0], L)
    if np.any(shift_boxes):
        state.coords_unwrapped[c] -= (shift_boxes * L).astype(
            state.coords_unwrapped.dtype)


# ---------------------------------------------------------------------------
# Self-avoiding-walk generation (unbounded) + dispersed initialization
# ---------------------------------------------------------------------------

def generate_saw(chain_length: int, rng: np.random.RandomState,
                 max_attempts: int = 400) -> np.ndarray:
    """Generate one unbounded self-avoiding walk of ``chain_length`` beads.

    Returns an ``(N, 3)`` int64 array starting at the origin.  Raises
    :class:`PlacementError` if a SAW cannot be grown within ``max_attempts``.
    The walk is unbounded (not wrapped); wrapping happens only later during
    placement/occupancy checks.
    """
    N = int(chain_length)
    if N < 2:
        raise MultiChainStateError("chain_length must be >= 2")
    for _ in range(int(max_attempts)):
        chain = [(0, 0, 0)]
        occ = {(0, 0, 0)}
        ok = True
        for _ in range(N - 1):
            x, y, z = chain[-1]
            opts = [(x + dx, y + dy, z + dz) for dx, dy, dz in NN6]
            opts = [o for o in opts if o not in occ]
            if not opts:
                ok = False
                break
            nxt = opts[rng.randint(len(opts))]
            chain.append(nxt)
            occ.add(nxt)
        if ok:
            return np.asarray(chain, dtype=np.int64)
    raise PlacementError(
        f"failed to grow a self-avoiding walk of length {N} in {max_attempts} "
        f"attempts")


def _random_cubic_rotation(rng: np.random.RandomState) -> np.ndarray:
    """A random proper cubic rotation matrix (includes the identity)."""
    # Reuse the reference implementation's 23 proper rotations (excludes I).
    import remd_uniform_chain_2_new as _remd
    mats = _remd.ROT_MATS
    idx = rng.randint(len(mats) + 1)  # +1 slot for identity
    if idx == len(mats):
        return np.eye(3, dtype=np.int64)
    return np.asarray(mats[idx], dtype=np.int64)


def initialize_dispersed_state(
    n_chains: int,
    chain_length: int,
    box_size: int,
    seed: int,
    max_placement_attempts: int = 2000,
) -> MultiChainState:
    """Deterministically build a dispersed multi-chain state from a seed.

    Each chain is grown as an unbounded self-avoiding walk, given a random
    proper cubic rotation and a random integer translation, then wrapped ONLY
    for the occupancy check.  A placement is accepted if the chain's own wrapped
    sites are unique AND disjoint from previously placed chains; otherwise it is
    retried up to ``max_placement_attempts`` per chain.  Straight chains are
    never used (a straight chain longer than the box can overlap its periodic
    image), so this is safe for dilute and moderately concentrated pilots.

    Raises :class:`PlacementError` if any chain cannot be placed.
    """
    M = int(n_chains)
    N = int(chain_length)
    L = _check_box_size(box_size)
    if M < 1:
        raise MultiChainStateError("n_chains must be >= 1")
    if N < 2:
        raise MultiChainStateError("chain_length must be >= 2")
    if M * N > L ** 3:
        raise MultiChainStateError(
            f"volume fraction > 1: M*N={M * N} beads cannot fit in L^3={L ** 3} "
            f"sites (phi={M * N / L ** 3:.3f})")

    rng = np.random.RandomState(int(seed) % (2 ** 32))
    coords = np.zeros((M, N, 3), dtype=np.int64)
    occupied: dict = {}  # wrapped site -> global id (occupancy so far)

    for c in range(M):
        placed = False
        for _ in range(int(max_placement_attempts)):
            saw = generate_saw(N, rng)
            R = _random_cubic_rotation(rng)
            rotated = saw @ R.T  # apply rotation to each bead (integer)
            shift = np.array(
                [rng.randint(L), rng.randint(L), rng.randint(L)], dtype=np.int64)
            candidate = rotated + shift  # unwrapped
            wrapped = np.mod(candidate, L)
            sites = [(int(a), int(b), int(cc)) for a, b, cc in wrapped]
            # Chain must not self-overlap when wrapped, nor hit existing beads.
            if len(set(sites)) != N:
                continue
            if any(s in occupied for s in sites):
                continue
            # Accept.
            coords[c] = candidate
            for i, s in enumerate(sites):
                occupied[s] = c * N + i
            placed = True
            break
        if not placed:
            raise PlacementError(
                f"could not place chain {c} of {M} (N={N}, L={L}, "
                f"phi={M * N / L ** 3:.3f}) within {max_placement_attempts} "
                f"attempts; lower the volume fraction or raise "
                f"max_placement_attempts")

    return make_state(coords, L)
