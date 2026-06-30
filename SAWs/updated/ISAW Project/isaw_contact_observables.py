#!/usr/bin/env python3
"""
Reusable, REMD-independent contact and structural observables for the 3D
interacting self-avoiding walk (ISAW) on the simple-cubic lattice.

This module contains only *pure* functions that operate on bead coordinates or
contact-pair arrays.  Nothing here imports REMD internals, so the same code is
used by the production sampler (online structural trajectories) and by the
offline feature extractor (full pair-motif analysis).

Scientific definitions (frozen)
-------------------------------
A nonbonded contact is an unordered pair (i, j) with

    * i < j
    * j - i > 1                      (nonbonded: backbone neighbours excluded)
    * Manhattan distance |r_i - r_j| == 1   (nearest neighbour on the lattice)

Each contact is counted exactly once.  On the simple-cubic lattice every valid
nonbonded nearest-neighbour contact has *odd* contour separation r = j - i.

    N       = number of beads (chain length)
    m       = total number of contacts
    m_r     = number of contacts with contour separation r  (sum_r m_r == m;
              m_r is the AUTHORITATIVE contour-separation representation)
    R_g^2   = (1/N) sum_i |r_i - r_cm|^2        (primary compaction observable)
    R_ee^2  = |r_{N-1} - r_0|^2

Contour-separation classification (two INDEPENDENT schemes)
-----------------------------------------------------------
The *fixed* scheme (short/medium/long_fixed) uses absolute contour separations
and answers "how local is this contact in monomers".  The *scaled* scheme
(local/mesoscopic/global_scaled) uses r/N and answers "how global is this
contact relative to the chain".  Each scheme is independently exhaustive and
non-overlapping over the valid (odd, r>=3) separations; the two schemes may and
do overlap with each other because they answer different questions.  ``m_r``
remains authoritative; the binned counts are convenience aggregates.

Contact graph (backbone edges excluded)
---------------------------------------
Vertices are bead indices that participate in at least one nonbonded contact;
edges are the contact pairs.  ``contact_graph_summary`` returns component,
degree, and cycle-rank statistics using a union-find implementation (no
NetworkX dependency).  ``S_max`` is the number of vertices in the largest
connected component.
"""
from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

Vec = Tuple[int, int, int]

# Six nearest-neighbour displacement vectors on the simple-cubic lattice.
_NN6: Tuple[Vec, ...] = (
    (1, 0, 0), (-1, 0, 0),
    (0, 1, 0), (0, -1, 0),
    (0, 0, 1), (0, 0, -1),
)

MIN_CONTOUR_SEPARATION = 3

# Bumped to 1.1.0 when the scaled-bin local boundary became closed (r/N<=0.10),
# making local_scaled non-empty for N=30.  Stored in every definition record so
# old files keep their historical definitions/version.
DEFINITIONS_VERSION = "1.1.0"


class ContactMapError(ValueError):
    """Raised when coordinates or a contact map fail validation."""


# ---------------------------------------------------------------------------
# Phase 1: strict coordinate normalization (single source of truth)
# ---------------------------------------------------------------------------

def normalize_lattice_coordinates(
    coordinates,
    *,
    require_self_avoiding: bool = True,
    require_backbone_bonds: bool = True,
) -> np.ndarray:
    """Return a validated signed-integer coordinate array of shape (N, 3).

    Rejects (raising :class:`ContactMapError` with a message identifying the
    failed condition):

    * wrong shape (not exactly ``(N, 3)``) or an empty array;
    * non-finite values (NaN, +/-inf);
    * fractional coordinates (values that are not exactly integral) -- they are
      rejected, never truncated;
    * duplicate occupied sites          (when ``require_self_avoiding``);
    * non-unit backbone bonds           (when ``require_backbone_bonds``).

    The lighter geometric callers pass both ``require_*`` flags False; they
    still get shape/finiteness/integrality checking but skip the lattice
    self-avoidance and connectivity requirements that are mathematically
    unnecessary for pure geometry.
    """
    try:
        arr = np.asarray(coordinates)
    except (TypeError, ValueError) as exc:
        # Ragged / inhomogeneous input (e.g. rows of differing length).
        raise ContactMapError(
            "coordinates must be a homogeneous (N, 3) array; got ragged/"
            "inhomogeneous input"
        ) from exc

    # Object / ragged input (e.g. list of mixed-length tuples) -> float coerce.
    if arr.dtype == object:
        try:
            arr = np.asarray(coordinates, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ContactMapError(
                "coordinates could not be interpreted as a numeric (N, 3) array"
            ) from exc

    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ContactMapError(
            f"coordinates must have shape (N, 3); got shape {arr.shape}"
        )
    n = arr.shape[0]
    if n == 0:
        raise ContactMapError("coordinates array is empty (N = 0)")

    if arr.dtype.kind in ("i", "u"):
        # Guard unsigned/over-wide integers that would silently wrap when cast
        # to int64 (e.g. np.uint64(2**63 + 1) -> negative).
        if arr.size:
            int64_max = int(np.iinfo(np.int64).max)
            int64_min = int(np.iinfo(np.int64).min)
            amax = int(arr.max())
            amin = int(arr.min())
            if amax > int64_max or amin < int64_min:
                raise ContactMapError(
                    f"coordinate value out of int64 range "
                    f"[{amin}, {amax}]; refusing to wrap on cast"
                )
        ints = arr.astype(np.int64)
    elif arr.dtype.kind in ("f", "c"):
        if arr.dtype.kind == "c":
            raise ContactMapError("coordinates must be real, not complex")
        if not np.all(np.isfinite(arr)):
            raise ContactMapError("coordinates contain NaN or infinity")
        rounded = np.rint(arr)
        if not np.all(rounded == arr):
            raise ContactMapError(
                "coordinates are not exactly integral; fractional lattice "
                "coordinates are rejected (not truncated)"
            )
        ints = rounded.astype(np.int64)
    elif arr.dtype.kind == "b":
        ints = arr.astype(np.int64)
    else:
        raise ContactMapError(
            f"coordinates have unsupported dtype {arr.dtype!r}"
        )

    if require_self_avoiding:
        uniq = np.unique(ints, axis=0)
        if uniq.shape[0] != n:
            raise ContactMapError(
                f"self-avoidance violated: {n - uniq.shape[0]} duplicate "
                f"occupied site(s)"
            )

    if require_backbone_bonds and n >= 2:
        bond = np.abs(np.diff(ints, axis=0)).sum(axis=1)
        bad = np.nonzero(bond != 1)[0]
        if bad.size:
            i = int(bad[0])
            raise ContactMapError(
                f"backbone bond {i}-{i + 1} has Manhattan length "
                f"{int(bond[i])} != 1 (chain is not a connected lattice walk)"
            )

    return ints


def _geometry_array(coordinates) -> np.ndarray:
    """Float (N, 3) array for pure-geometry callers (no SAW/bond requirement)."""
    ints = normalize_lattice_coordinates(
        coordinates, require_self_avoiding=False, require_backbone_bonds=False
    )
    return ints.astype(np.float64)


# ---------------------------------------------------------------------------
# Contact-map construction (O(N) via coordinate hashing)
# ---------------------------------------------------------------------------

def build_contact_map(
    coordinates: Sequence[Vec],
) -> tuple[np.ndarray, np.ndarray]:
    """Extract the nonbonded contact map from a single conformation.

    Returns
    -------
    contact_pairs : np.ndarray, shape (m, 2), dtype int64, rows (i, j) with i<j
    separations   : np.ndarray, shape (m,),  dtype int64, r = j - i

    Coordinates are validated strictly (self-avoiding, unit backbone bonds,
    integral) via :func:`normalize_lattice_coordinates`.  The construction maps
    each occupied coordinate to its bead index and probes the six
    nearest-neighbour sites, so it is O(N) rather than O(N^2).
    """
    ints = normalize_lattice_coordinates(coordinates)
    n = ints.shape[0]
    coords = [(int(x), int(y), int(z)) for x, y, z in ints]

    index_of: dict[Vec, int] = {}
    for i, r in enumerate(coords):
        index_of[r] = i  # self-avoiding walks have unique sites

    pairs: list[tuple[int, int]] = []
    for i, (x, y, z) in enumerate(coords):
        for dx, dy, dz in _NN6:
            nbr = (x + dx, y + dy, z + dz)
            j = index_of.get(nbr)
            if j is None:
                continue
            if j <= i:
                continue          # count each unordered pair once (j > i)
            if j - i <= 1:
                continue          # nonbonded only (exclude backbone neighbour)
            pairs.append((i, j))

    if not pairs:
        return (np.empty((0, 2), dtype=np.int64),
                np.empty((0,), dtype=np.int64))

    pairs.sort()  # lexicographic; pairs are already unique by construction
    arr = np.array(pairs, dtype=np.int64)
    seps = arr[:, 1] - arr[:, 0]
    return arr, seps


def build_contact_map_bruteforce(
    coordinates: Sequence[Vec],
) -> tuple[np.ndarray, np.ndarray]:
    """Reference O(N^2) contact-map construction (for tests/validation)."""
    ints = normalize_lattice_coordinates(coordinates)
    arr = ints.astype(np.int64)
    n = arr.shape[0]
    pairs: list[tuple[int, int]] = []
    for i in range(n):
        for j in range(i + 2, n):  # j - i > 1
            if int(np.abs(arr[i] - arr[j]).sum()) == 1:
                pairs.append((i, j))
    if not pairs:
        return (np.empty((0, 2), dtype=np.int64),
                np.empty((0,), dtype=np.int64))
    out = np.array(sorted(pairs), dtype=np.int64)
    return out, out[:, 1] - out[:, 0]


# ---------------------------------------------------------------------------
# Phase 2: complete contact-map validation
# ---------------------------------------------------------------------------

def _exact_int_array(values, *, what: str) -> np.ndarray:
    """Coerce to int64 only if numeric, finite, and exactly integral.

    Rejects object/ragged arrays, NaN/inf, and fractional values rather than
    truncating via ``astype``.
    """
    try:
        arr = np.asarray(values)
    except (TypeError, ValueError) as exc:
        raise ContactMapError(f"{what} could not be interpreted as an array") from exc
    if arr.dtype == object:
        raise ContactMapError(f"{what} must be a numeric array, not object/ragged")
    if arr.dtype.kind == "b":
        return arr.astype(np.int64)
    if arr.dtype.kind in ("i", "u"):
        if arr.dtype.kind == "u" and arr.size and int(arr.max()) > int(np.iinfo(np.int64).max):
            raise ContactMapError(f"{what} value out of int64 range")
        return arr.astype(np.int64)
    if arr.dtype.kind in ("f", "c"):
        if arr.dtype.kind == "c":
            raise ContactMapError(f"{what} must be real, not complex")
        if not np.all(np.isfinite(arr)):
            raise ContactMapError(f"{what} contains NaN or infinity")
        rounded = np.rint(arr)
        if not np.all(rounded == arr):
            raise ContactMapError(f"{what} has fractional values (not exact integers)")
        return rounded.astype(np.int64)
    raise ContactMapError(f"{what} has unsupported dtype {arr.dtype!r}")


def _canonical_pairs(contact_pairs) -> np.ndarray:
    """Validate and return contact pairs as an exact int64 (m, 2) array.

    Requires exactly 2-D input with a trailing dimension of size 2.  An empty
    array is accepted only if it can represent zero pairs (shape (0, 2) or an
    empty 1-D array).  No silent reshape of arbitrary arrays.
    """
    try:
        raw = np.asarray(contact_pairs)
    except (TypeError, ValueError) as exc:
        raise ContactMapError("contact pairs could not be interpreted as an array") from exc
    if raw.dtype == object:
        raise ContactMapError("contact pairs must be numeric, not object/ragged")
    if raw.size == 0:
        # Accept empty 1-D or (0, 2); reject other empty shapes like (0, 4).
        if raw.ndim == 1 or (raw.ndim == 2 and raw.shape[1] == 2):
            return np.empty((0, 2), dtype=np.int64)
        raise ContactMapError(
            f"empty contact-pair array has shape {raw.shape}; expected (0, 2)"
        )
    if raw.ndim != 2 or raw.shape[1] != 2:
        raise ContactMapError(
            f"contact pairs must have shape (m, 2); got shape {raw.shape}"
        )
    return _exact_int_array(raw, what="contact pairs")


def validate_contact_map(
    coordinates: Sequence[Vec],
    contact_pairs: np.ndarray,
    separations: np.ndarray | None = None,
    expected_contact_count: int | None = None,
    *,
    strict: bool = True,
) -> dict:
    """Validate that ``contact_pairs`` is the *complete* contact map of ``coordinates``.

    In strict mode (default) this:

    1. normalizes coordinates strictly;
    2. canonicalizes supplied pairs to integer ``(i, j)`` rows;
    3. requires all pair indices in ``[0, n_beads)``;
    4. requires ``i < j``; 5. requires ``j - i > 1``;
    6. requires unit Manhattan distance; 7. rejects duplicate pairs;
    8. sorts to a canonical (lexicographic) representation;
    9. computes the complete reference map with
       :func:`build_contact_map_bruteforce`;
    10. requires exact set equality of supplied and reference pairs;
    11. if ``separations`` is supplied, requires shape ``(m,)`` and exact
        equality to ``pairs[:, 1] - pairs[:, 0]``;
    12. requires all separations odd;
    13. if ``expected_contact_count`` is supplied, requires equality.

    The lighter ``strict=False`` mode checks per-pair structural validity
    (steps 2-7, 12) without proving completeness against the brute-force map --
    use it only for performance-sensitive internal re-checks where the map was
    just produced by :func:`build_contact_map`.

    Returns structured validation metadata; raises :class:`ContactMapError` on
    the first hard violation.
    """
    ints = normalize_lattice_coordinates(coordinates)
    n = ints.shape[0]
    pairs = _canonical_pairs(contact_pairs)
    m = pairs.shape[0]

    if m > 0:
        i = pairs[:, 0]
        j = pairs[:, 1]
        if i.min() < 0 or j.max() >= n:
            raise ContactMapError(
                f"contact pair index out of bounds for n_beads={n}"
            )
        if not np.all(i < j):
            raise ContactMapError("contact pair violates i < j")
        if not np.all((j - i) > 1):
            raise ContactMapError("contact pair violates j - i > 1 (bonded pair)")
        dist = np.abs(ints[i] - ints[j]).sum(axis=1)
        if not np.all(dist == 1):
            raise ContactMapError("contact pair Manhattan distance != 1")
        seen = {(int(a), int(b)) for a, b in pairs}
        if len(seen) != m:
            raise ContactMapError("duplicate contact pair(s) present")

    # Canonical lexicographic order.
    if m > 0:
        order = np.lexsort((pairs[:, 1], pairs[:, 0]))
        pairs_sorted = pairs[order]
    else:
        pairs_sorted = pairs

    seps_self = (pairs_sorted[:, 1] - pairs_sorted[:, 0]) if m else np.empty(0, np.int64)
    if m > 0 and not np.all((seps_self % 2) == 1):
        raise ContactMapError("non-odd contour separation on cubic lattice")

    if separations is not None:
        raw_sep = np.asarray(separations)
        if raw_sep.ndim != 1:
            raise ContactMapError(
                f"separations must be 1-D; got shape {raw_sep.shape}"
            )
        if raw_sep.shape[0] != m:
            raise ContactMapError(
                f"separations shape {raw_sep.shape} inconsistent with "
                f"{m} contact pair(s)"
            )
        # Exact-int validation (rejects 5.9, NaN, inf) -- never truncate.
        sep_arr = _exact_int_array(raw_sep, what="separations")
        # Compare against the *as-supplied* pair order, not the sorted one.
        seps_supplied = (pairs[:, 1] - pairs[:, 0]) if m else np.empty(0, np.int64)
        if not np.array_equal(sep_arr, seps_supplied):
            raise ContactMapError(
                "supplied separations != pairs[:, 1] - pairs[:, 0]"
            )

    if strict:
        ref, _ = build_contact_map_bruteforce(ints)
        ref_set = {(int(a), int(b)) for a, b in ref}
        sup_set = {(int(a), int(b)) for a, b in pairs_sorted}
        if ref_set != sup_set:
            missing = sorted(ref_set - sup_set)
            extra = sorted(sup_set - ref_set)
            raise ContactMapError(
                "supplied contact map is not the complete map implied by the "
                f"coordinates: missing {missing[:5]}"
                f"{'...' if len(missing) > 5 else ''}, "
                f"extra {extra[:5]}{'...' if len(extra) > 5 else ''}"
            )

    if expected_contact_count is not None and m != int(expected_contact_count):
        raise ContactMapError(
            f"contact count mismatch: map has {m}, expected "
            f"{int(expected_contact_count)}"
        )

    return {
        "ok": True,
        "n_beads": int(n),
        "contact_count": int(m),
        "strict": bool(strict),
        "all_separations_odd": True,
    }


def contact_separation_counts(
    contact_pairs: np.ndarray,
    n_beads: int,
) -> np.ndarray:
    """Dense contour-separation histogram m_r, length ``n_beads``.

    ``m_r[r]`` is the number of contacts with separation r = j - i.  Even
    entries are zero on the cubic lattice.  Validates ``n_beads >= 1``, pair
    bounds, positive separations, ``r < n_beads``, and ``sum(m_r) == m``.
    """
    n_beads = int(n_beads)
    if n_beads < 1:
        raise ContactMapError(f"n_beads must be >= 1, got {n_beads}")
    m_r = np.zeros(n_beads, dtype=np.int64)
    pairs = _canonical_pairs(contact_pairs)
    m = pairs.shape[0]
    if m == 0:
        return m_r
    i = pairs[:, 0]
    j = pairs[:, 1]
    if i.min() < 0 or j.max() >= n_beads:
        raise ContactMapError(
            f"contact pair index out of bounds for n_beads={n_beads}"
        )
    seps = j - i
    if seps.min() <= 0:
        raise ContactMapError("contact separations must be strictly positive")
    if seps.max() >= n_beads:
        raise ContactMapError(
            f"contact separation {int(seps.max())} >= n_beads {n_beads}"
        )
    counts = np.bincount(seps, minlength=n_beads)
    m_r[: counts.shape[0]] = counts[:n_beads]
    if int(m_r.sum()) != m:
        raise ContactMapError(
            f"m_r sum {int(m_r.sum())} != number of contacts {m}"
        )
    return m_r


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def radius_of_gyration_squared(coordinates: Sequence[Vec]) -> float:
    """R_g^2 = (1/N) sum_i |r_i - r_cm|^2 (lattice units)."""
    r = _geometry_array(coordinates)
    cm = r.mean(axis=0)
    return float(((r - cm) ** 2).sum(axis=1).mean())


def end_to_end_distance_squared(coordinates: Sequence[Vec]) -> float:
    """R_ee^2 = |r_{N-1} - r_0|^2 (lattice units)."""
    r = _geometry_array(coordinates)
    d = r[-1] - r[0]
    return float((d * d).sum())


def gyration_tensor(coordinates: Sequence[Vec]) -> np.ndarray:
    """Symmetric 3x3 gyration tensor; trace equals R_g^2."""
    r = _geometry_array(coordinates)
    d = r - r.mean(axis=0)
    return (d.T @ d) / float(d.shape[0])


def gyration_eigenvalues(coordinates: Sequence[Vec]) -> np.ndarray:
    """Ascending eigenvalues of the gyration tensor (lambda_1<=...<=lambda_3)."""
    return np.linalg.eigvalsh(gyration_tensor(coordinates))


def asphericity(coordinates: Sequence[Vec]) -> float:
    """Asphericity b = lam3 - 0.5*(lam1+lam2) (>= 0; 0 for spherical)."""
    lam = gyration_eigenvalues(coordinates)
    return float(lam[2] - 0.5 * (lam[0] + lam[1]))


# ---------------------------------------------------------------------------
# Contact graph (union-find; no NetworkX in the hot loop)
# ---------------------------------------------------------------------------

def _zero_contact_graph_summary(n_beads: int) -> dict:
    return {
        "contact_vertices": 0,
        "contact_graph_components": 0,
        "contact_graph_edges": 0,
        "sum_component_edges": 0,
        "largest_component_vertices": 0,
        "largest_component_edges": 0,
        "largest_component_fraction_of_N": 0.0,
        "largest_component_fraction_of_contact_vertices": 0.0,
        "mean_degree_nonisolated": 0.0,
        "degree_variance_nonisolated": 0.0,
        "contact_graph_cycle_rank": 0,
        "number_of_multiedge_components": 0,
    }


def contact_graph_summary(contact_pairs: np.ndarray, n_beads: int) -> dict:
    """Compact connected-component / degree / cycle-rank statistics.

    The contact graph excludes backbone edges.  S_max ==
    ``largest_component_vertices``.  The zero-contact case is handled
    explicitly (all counts zero, fractions 0.0).  Largest-component
    tie-breaking is deterministic: ties on vertex count are broken by the
    smallest minimum vertex index in the component.
    """
    n_beads = int(n_beads)
    pairs = _canonical_pairs(contact_pairs)
    m = pairs.shape[0]
    if m == 0:
        return _zero_contact_graph_summary(n_beads)

    verts = np.unique(pairs)
    n_vert = int(verts.shape[0])

    parent = {int(v): int(v) for v in verts}

    def find(x: int) -> int:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:  # path compression
            parent[x], x = root, parent[x]
        return root

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    degree: dict[int, int] = {int(v): 0 for v in verts}
    for a, b in pairs:
        a = int(a); b = int(b)
        union(a, b)
        degree[a] += 1
        degree[b] += 1

    comp_vertices: dict[int, int] = {}
    comp_edges: dict[int, int] = {}
    comp_min_vertex: dict[int, int] = {}
    for v in verts:
        root = find(int(v))
        comp_vertices[root] = comp_vertices.get(root, 0) + 1
        comp_min_vertex[root] = min(comp_min_vertex.get(root, int(v)), int(v))
    for a, b in pairs:
        root = find(int(a))
        comp_edges[root] = comp_edges.get(root, 0) + 1

    n_components = len(comp_vertices)

    # Largest component by vertex count; deterministic tie-break by smallest
    # minimum vertex index.
    best_root = min(
        comp_vertices,
        key=lambda r: (-comp_vertices[r], comp_min_vertex[r]),
    )
    s_max = int(comp_vertices[best_root])
    s_max_edges = int(comp_edges.get(best_root, 0))

    deg_vals = np.array([degree[int(v)] for v in verts], dtype=float)
    mean_degree = float(deg_vals.mean())
    degree_var = float(deg_vals.var())

    cycle_rank = int(m - n_vert + n_components)
    n_multiedge = int(sum(1 for e in comp_edges.values() if e >= 2))
    # Independent edge totals (not derived from the cycle-rank identity), so the
    # consumer can cross-check edges == m and sum_component_edges == m without
    # the check being tautological.
    sum_comp_edges = int(sum(comp_edges.values()))

    return {
        "contact_vertices": n_vert,
        "contact_graph_components": int(n_components),
        "contact_graph_edges": int(m),
        "sum_component_edges": sum_comp_edges,
        "largest_component_vertices": s_max,
        "largest_component_edges": s_max_edges,
        "largest_component_fraction_of_N": (
            float(s_max) / float(n_beads) if n_beads > 0 else 0.0
        ),
        "largest_component_fraction_of_contact_vertices": (
            float(s_max) / float(n_vert) if n_vert > 0 else 0.0
        ),
        "mean_degree_nonisolated": mean_degree,
        "degree_variance_nonisolated": degree_var,
        "contact_graph_cycle_rank": cycle_rank,
        "number_of_multiedge_components": n_multiedge,
    }


# ---------------------------------------------------------------------------
# Phase 3: two INDEPENDENT contour-separation classification schemes
# ---------------------------------------------------------------------------
# Fixed scheme (absolute contour separation, monomers):
#     short_fixed  : 3  <= r <= 9
#     medium_fixed : 11 <= r <  long_threshold_fixed
#     long_fixed   : r  >= long_threshold_fixed
# The fixed long threshold is a fixed odd separation (default 15) chosen so that
# every class is non-empty for the pilot lengths N=30 and N=44 (the previous
# floor(N/3) threshold produced an empty medium class at N=30).
#
# Scaled scheme (relative contour separation r/N):
#     local_scaled       : r/N <  0.10
#     mesoscopic_scaled  : 0.10 <= r/N < 0.33
#     global_scaled      : r/N >= 0.33
#
# The two schemes are independent and may overlap with each other.  ``m_r``
# stays authoritative; both schemes are exhaustive and non-overlapping over the
# valid (odd, r>=3) separations within their own definition.

FIXED_BIN_DEFINITIONS = {
    "scheme": "fixed",
    "short_fixed": {"r_min": 3, "r_max": 9},
    "medium_fixed": {"r_min": 11},
    "long_threshold_fixed": 15,
    "description": (
        "short_fixed 3<=r<=9; medium_fixed 11<=r<long_threshold_fixed; "
        "long_fixed r>=long_threshold_fixed"
    ),
}

# Boundary inclusion (canonical, definitions_version 1.1.0):
#     local_scaled      : r/N <= local_max_ratio          (CLOSED upper bound)
#     mesoscopic_scaled : local_max_ratio < r/N < meso_max_ratio
#     global_scaled     : r/N >= meso_max_ratio           (CLOSED lower bound)
# The local upper bound is closed so that r=3 at N=30 (r/N = 0.10 exactly) falls
# in local_scaled rather than leaving the local class empty.
SCALED_BIN_DEFINITIONS = {
    "scheme": "scaled",
    "local_max_ratio": 0.10,
    "meso_max_ratio": 0.33,
    "local_boundary": "closed",   # r/N == local_max_ratio -> local_scaled
    "description": (
        "local_scaled r/N<=0.10; mesoscopic_scaled 0.10<r/N<0.33; "
        "global_scaled r/N>=0.33"
    ),
}

FIXED_BIN_LABELS = ("m_short_fixed", "m_medium_fixed", "m_long_fixed")
SCALED_BIN_LABELS = ("m_local_scaled", "m_mesoscopic_scaled", "m_global_scaled")


def _fixed_defs(defs: dict | None) -> dict:
    return FIXED_BIN_DEFINITIONS if defs is None else defs


def _scaled_defs(defs: dict | None) -> dict:
    return SCALED_BIN_DEFINITIONS if defs is None else defs


def assign_fixed_bin(r: int, defs: dict | None = None) -> str:
    """Return 'short_fixed' | 'medium_fixed' | 'long_fixed' for separation r."""
    defs = _fixed_defs(defs)
    r = int(r)
    thr = int(defs["long_threshold_fixed"])
    short = defs["short_fixed"]
    if r >= thr:
        return "long_fixed"
    if int(short["r_min"]) <= r <= int(short["r_max"]):
        return "short_fixed"
    return "medium_fixed"


def assign_scaled_bin(r: int, n_beads: int, defs: dict | None = None) -> str:
    """Return 'local_scaled' | 'mesoscopic_scaled' | 'global_scaled'.

    Boundary inclusion (definitions_version 1.1.0): local upper bound is closed
    (r/N <= local_max_ratio -> local_scaled); global lower bound is closed
    (r/N >= meso_max_ratio -> global_scaled).
    """
    defs = _scaled_defs(defs)
    ratio = float(r) / float(n_beads)
    if ratio <= float(defs["local_max_ratio"]):
        return "local_scaled"
    if ratio < float(defs["meso_max_ratio"]):
        return "mesoscopic_scaled"
    return "global_scaled"


def _valid_separations(n_beads: int) -> list[int]:
    return [r for r in range(MIN_CONTOUR_SEPARATION, int(n_beads)) if r % 2 == 1]


def validate_m_r(
    m_r,
    *,
    n_beads: int | None = None,
    require_cubic_parity: bool = True,
) -> np.ndarray:
    """Strictly validate an authoritative contour-separation vector m_r.

    Checks: 1-D, numeric, finite, exact integers, nonnegative; length equals
    ``n_beads`` when supplied; entries at r < 3 are zero; even-r entries are
    zero (cubic-lattice parity) when ``require_cubic_parity``.  Never truncates;
    returns a signed int64 array.
    """
    raw = np.asarray(m_r)
    if raw.dtype == object:
        raise ContactMapError("m_r must be a numeric array, not object/ragged")
    if raw.ndim != 1:
        raise ContactMapError(f"m_r must be 1-D; got shape {raw.shape}")
    arr = _exact_int_array(raw, what="m_r")
    if np.any(arr < 0):
        raise ContactMapError("m_r entries must be nonnegative")
    if n_beads is not None and arr.shape[0] != int(n_beads):
        raise ContactMapError(
            f"m_r length {arr.shape[0]} != n_beads {int(n_beads)}"
        )
    below = np.nonzero(arr[:MIN_CONTOUR_SEPARATION])[0]
    if below.size:
        raise ContactMapError(
            f"m_r has nonzero count(s) at invalid separation r<{MIN_CONTOUR_SEPARATION}"
        )
    if require_cubic_parity and np.any(arr[0::2] != 0):
        raise ContactMapError(
            "m_r has nonzero count(s) at even separation (forbidden on cubic lattice)"
        )
    return arr


def _check_m_r(m_r: np.ndarray) -> np.ndarray:
    # Backward-compatible internal helper -> strict validator (parity enforced).
    return validate_m_r(m_r)


def bin_contact_separations_fixed(
    m_r: np.ndarray, n_beads: int | None = None, defs: dict | None = None,
) -> dict:
    """Aggregate m_r into the fixed scheme (short/medium/long_fixed)."""
    m_r = _check_m_r(m_r)
    counts = {"m_short_fixed": 0, "m_medium_fixed": 0, "m_long_fixed": 0}
    for r in range(len(m_r)):
        c = int(m_r[r])
        if c == 0:
            continue
        counts["m_" + assign_fixed_bin(r, defs)] += c
    total = sum(counts.values())
    if total != int(m_r.sum()):
        raise ContactMapError(
            f"fixed-binned total {total} != m_r sum {int(m_r.sum())}"
        )
    return {k: int(v) for k, v in counts.items()}


def bin_contact_separations_scaled(
    m_r: np.ndarray, n_beads: int, defs: dict | None = None,
) -> dict:
    """Aggregate m_r into the scaled scheme (local/mesoscopic/global_scaled)."""
    m_r = _check_m_r(m_r)
    counts = {"m_local_scaled": 0, "m_mesoscopic_scaled": 0, "m_global_scaled": 0}
    for r in range(len(m_r)):
        c = int(m_r[r])
        if c == 0:
            continue
        counts["m_" + assign_scaled_bin(r, n_beads, defs)] += c
    total = sum(counts.values())
    if total != int(m_r.sum()):
        raise ContactMapError(
            f"scaled-binned total {total} != m_r sum {int(m_r.sum())}"
        )
    return {k: int(v) for k, v in counts.items()}


def validate_bin_definitions(
    n_beads: int,
    fixed_defs: dict | None = None,
    scaled_defs: dict | None = None,
) -> dict:
    """Check both schemes are exhaustive, non-overlapping, and non-pathological.

    Each scheme must partition the valid (odd, r>=3) separations exactly once.
    Empty categories for the selected N are reported as warnings (never
    silently produced).  Raises on overlap or gaps within a scheme.
    """
    valid_r = _valid_separations(n_beads)

    fixed_membership = {"short_fixed": [], "medium_fixed": [], "long_fixed": []}
    for r in valid_r:
        fixed_membership[assign_fixed_bin(r, fixed_defs)].append(r)
    covered = sorted(sum(fixed_membership.values(), []))
    if covered != valid_r:
        raise ContactMapError(
            f"fixed scheme is not an exhaustive non-overlapping partition of "
            f"{valid_r}; covered {covered}"
        )

    scaled_membership = {
        "local_scaled": [], "mesoscopic_scaled": [], "global_scaled": [],
    }
    for r in valid_r:
        scaled_membership[assign_scaled_bin(r, n_beads, scaled_defs)].append(r)
    covered_s = sorted(sum(scaled_membership.values(), []))
    if covered_s != valid_r:
        raise ContactMapError(
            f"scaled scheme is not an exhaustive non-overlapping partition of "
            f"{valid_r}; covered {covered_s}"
        )

    warnings = []
    for name, members in fixed_membership.items():
        if not members:
            warnings.append(
                f"fixed category {name!r} is empty for n_beads={n_beads}"
            )
    for name, members in scaled_membership.items():
        if not members:
            warnings.append(
                f"scaled category {name!r} is empty for n_beads={n_beads}"
            )

    return {
        "valid_separations": valid_r,
        "fixed_membership": fixed_membership,
        "scaled_membership": scaled_membership,
        "warnings": warnings,
    }


def project_bin_definitions(n_beads: int) -> dict:
    """Serializable record of both bin schemes for a given N (for metadata)."""
    return {
        "definitions_version": DEFINITIONS_VERSION,
        "n_beads": int(n_beads),
        "min_separation": MIN_CONTOUR_SEPARATION,
        "fixed": dict(FIXED_BIN_DEFINITIONS),
        "scaled": dict(SCALED_BIN_DEFINITIONS),
    }


# ---- Backward-compatibility shims (legacy short/medium/long API) -----------
# The previous hybrid scheme combined a fixed short class with an N-scaled long
# class into one exclusive classification (which produced an empty medium class
# at N=30).  These shims preserve the old call signatures used by older callers
# and map onto the new FIXED scheme, which is exhaustive for all N.

def default_bin_definitions(n_beads: int) -> dict:
    """Deprecated: returns the combined fixed+scaled definition record."""
    return project_bin_definitions(n_beads)


def bin_contact_separations(m_r, n_beads, bin_defs: dict | None = None) -> dict:
    """Deprecated shim: returns m_short/m_medium/m_long from the FIXED scheme."""
    fixed = bin_contact_separations_fixed(m_r, n_beads)
    return {
        "m_short": fixed["m_short_fixed"],
        "m_medium": fixed["m_medium_fixed"],
        "m_long": fixed["m_long_fixed"],
    }


# ---------------------------------------------------------------------------
# Pair-contact topology (offline; O(m^2) is acceptable)
# ---------------------------------------------------------------------------

PAIR_MOTIF_LABELS = ("shared_endpoint", "disjoint", "nested", "interleaved")


def classify_contact_pair(
    contact_a: Sequence[int],
    contact_b: Sequence[int],
) -> str:
    """Exclusive topology class of two contacts.

    Priority: ``shared_endpoint`` first (any shared bead index); otherwise, with
    four distinct endpoints, one of ``disjoint``, ``nested``, ``interleaved``.
    Contacts are canonicalized (i<j) before classification, so the result is
    symmetric under swapping the two contacts.
    """
    i, j = sorted((int(contact_a[0]), int(contact_a[1])))
    k, l = sorted((int(contact_b[0]), int(contact_b[1])))

    if len({i, j, k, l}) < 4:
        return "shared_endpoint"

    if j < k or l < i:
        return "disjoint"          # i<j<k<l or k<l<i<j
    if (i < k and l < j) or (k < i and j < l):
        return "nested"            # i<k<l<j or k<i<j<l
    return "interleaved"           # i<k<j<l or k<i<l<j


def count_pair_motifs(contact_pairs: np.ndarray) -> dict:
    """Count exclusive pair-topology classes over all C(m,2) contact pairs.

    Returns counts keyed ``pair_shared_endpoint``, ``pair_disjoint``,
    ``pair_nested``, ``pair_interleaved`` plus ``pair_total``.  Their sum
    (excluding ``pair_total``) equals C(m,2).
    """
    pairs = _canonical_pairs(contact_pairs)
    m = pairs.shape[0]
    counts = {label: 0 for label in PAIR_MOTIF_LABELS}
    for a in range(m):
        ia, ja = int(pairs[a, 0]), int(pairs[a, 1])
        for b in range(a + 1, m):
            label = classify_contact_pair((ia, ja), (pairs[b, 0], pairs[b, 1]))
            counts[label] += 1
    out = {f"pair_{label}": int(counts[label]) for label in PAIR_MOTIF_LABELS}
    n_total = sum(counts.values())
    expected = m * (m - 1) // 2
    if n_total != expected:
        raise ContactMapError(
            f"pair-motif total {n_total} != C(m,2)={expected}"
        )
    out["pair_total"] = int(n_total)
    return out


def contact_separation_summary(separations: np.ndarray) -> dict:
    """Mean / max contour separation among contacts (None when m == 0)."""
    seps = np.asarray(separations, dtype=float).ravel()
    if seps.size == 0:
        return {"mean_contact_separation": None, "max_contact_separation": None}
    return {
        "mean_contact_separation": float(seps.mean()),
        "max_contact_separation": int(seps.max()),
    }
