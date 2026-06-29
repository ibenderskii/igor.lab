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

    m       = total number of contacts
    m_r     = number of contacts with contour separation r  (sum_r m_r == m)
    R_g^2   = (1/N) sum_i |r_i - r_cm|^2        (primary compaction observable)
    R_ee^2  = |r_{N-1} - r_0|^2

Contact graph (backbone edges excluded)
---------------------------------------
Vertices are bead indices that participate in at least one nonbonded contact;
edges are the contact pairs.  ``contact_graph_summary`` returns component,
degree, and cycle-rank statistics using a union-find implementation (no
NetworkX dependency).  ``S_max`` is the number of vertices in the largest
connected component.
"""
from __future__ import annotations

import math
from typing import Sequence, Tuple

import numpy as np

Vec = Tuple[int, int, int]

# Six nearest-neighbour displacement vectors on the simple-cubic lattice.
_NN6: Tuple[Vec, ...] = (
    (1, 0, 0), (-1, 0, 0),
    (0, 1, 0), (0, -1, 0),
    (0, 0, 1), (0, 0, -1),
)


class ContactMapError(ValueError):
    """Raised when a contact map fails validation."""


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def _as_int_tuples(coordinates: Sequence[Vec]) -> list[Vec]:
    """Return coordinates as a list of integer 3-tuples (no copy of dtype)."""
    out: list[Vec] = []
    for r in coordinates:
        out.append((int(r[0]), int(r[1]), int(r[2])))
    return out


def _as_float_array(coordinates: Sequence[Vec]) -> np.ndarray:
    arr = np.asarray(coordinates, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(
            f"coordinates must have shape (N, 3); got {arr.shape}"
        )
    return arr


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

    The construction maps each occupied coordinate to its bead index and probes
    the six nearest-neighbour sites, so it is O(N) rather than O(N^2).
    """
    coords = _as_int_tuples(coordinates)
    n = len(coords)

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
    arr = _as_float_array(coordinates)
    n = arr.shape[0]
    pairs: list[tuple[int, int]] = []
    for i in range(n):
        for j in range(i + 2, n):  # j - i > 1
            if np.abs(arr[i] - arr[j]).sum() == 1.0:
                pairs.append((i, j))
    if not pairs:
        return (np.empty((0, 2), dtype=np.int64),
                np.empty((0,), dtype=np.int64))
    out = np.array(sorted(pairs), dtype=np.int64)
    return out, out[:, 1] - out[:, 0]


def validate_contact_map(
    coordinates: Sequence[Vec],
    contact_pairs: np.ndarray,
    separations: np.ndarray | None = None,
    expected_contact_count: int | None = None,
) -> dict:
    """Validate a contact map against its coordinates.

    Checks self-avoidance, backbone bond adjacency, i<j, j-i>1, unit Manhattan
    distance, no duplicate pairs, odd contour separation, and (optionally) the
    expected contact count.  Returns a dict with ``ok`` and diagnostic fields;
    raises :class:`ContactMapError` on the first hard violation.
    """
    arr = _as_float_array(coordinates)
    n = arr.shape[0]
    pairs = np.asarray(contact_pairs, dtype=np.int64).reshape(-1, 2)

    # Self-avoidance: all coordinates distinct.
    uniq = {tuple(int(v) for v in r) for r in arr}
    if len(uniq) != n:
        raise ContactMapError(
            f"self-avoidance violated: {n - len(uniq)} duplicated site(s)"
        )

    # Backbone bond adjacency: consecutive beads at unit Manhattan distance.
    if n >= 2:
        bond = np.abs(np.diff(arr, axis=0)).sum(axis=1)
        if not np.all(bond == 1.0):
            bad = int(np.argmax(bond != 1.0))
            raise ContactMapError(
                f"backbone bond {bad}-{bad + 1} is not unit length "
                f"(|step| = {bond[bad]})"
            )

    m = pairs.shape[0]
    if m > 0:
        i = pairs[:, 0]
        j = pairs[:, 1]
        if not np.all(i < j):
            raise ContactMapError("contact pair violates i < j")
        if not np.all((j - i) > 1):
            raise ContactMapError("contact pair violates j - i > 1 (bonded pair)")
        dist = np.abs(arr[i] - arr[j]).sum(axis=1)
        if not np.all(dist == 1.0):
            raise ContactMapError("contact pair Manhattan distance != 1")
        # No duplicate pairs.
        seen = {(int(a), int(b)) for a, b in pairs}
        if len(seen) != m:
            raise ContactMapError("duplicate contact pair(s) present")
        # Odd contour separation.
        seps = (j - i) if separations is None else np.asarray(separations)
        if not np.all((seps % 2) == 1):
            raise ContactMapError("non-odd contour separation on cubic lattice")

    if expected_contact_count is not None and m != int(expected_contact_count):
        raise ContactMapError(
            f"contact count mismatch: map has {m}, expected "
            f"{int(expected_contact_count)}"
        )

    return {
        "ok": True,
        "n_beads": int(n),
        "contact_count": int(m),
        "all_separations_odd": True,
    }


def contact_separation_counts(
    contact_pairs: np.ndarray,
    n_beads: int,
) -> np.ndarray:
    """Dense contour-separation histogram m_r, length ``n_beads``.

    ``m_r[r]`` is the number of contacts with separation r = j - i.  Even
    entries are zero on the cubic lattice.  Satisfies ``m_r.sum() == m``.
    """
    n_beads = int(n_beads)
    m_r = np.zeros(n_beads, dtype=np.int64)
    pairs = np.asarray(contact_pairs, dtype=np.int64).reshape(-1, 2)
    if pairs.shape[0] == 0:
        return m_r
    seps = pairs[:, 1] - pairs[:, 0]
    counts = np.bincount(seps, minlength=n_beads)
    m_r[: counts.shape[0]] = counts[:n_beads]
    if int(m_r.sum()) != pairs.shape[0]:
        raise ContactMapError(
            f"m_r sum {int(m_r.sum())} != number of contacts {pairs.shape[0]}"
        )
    return m_r


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def radius_of_gyration_squared(coordinates: Sequence[Vec]) -> float:
    """R_g^2 = (1/N) sum_i |r_i - r_cm|^2 (lattice units)."""
    r = _as_float_array(coordinates)
    cm = r.mean(axis=0)
    return float(((r - cm) ** 2).sum(axis=1).mean())


def end_to_end_distance_squared(coordinates: Sequence[Vec]) -> float:
    """R_ee^2 = |r_{N-1} - r_0|^2 (lattice units)."""
    r = _as_float_array(coordinates)
    d = r[-1] - r[0]
    return float((d * d).sum())


def gyration_tensor(coordinates: Sequence[Vec]) -> np.ndarray:
    """Symmetric 3x3 gyration tensor; trace equals R_g^2."""
    r = _as_float_array(coordinates)
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
    explicitly (all counts zero, fractions 0.0).
    """
    n_beads = int(n_beads)
    pairs = np.asarray(contact_pairs, dtype=np.int64).reshape(-1, 2)
    m = pairs.shape[0]
    if m == 0:
        return _zero_contact_graph_summary(n_beads)

    # Vertices that participate in at least one contact.
    verts = np.unique(pairs)
    n_vert = int(verts.shape[0])

    # Union-find over participating vertices.
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

    # Aggregate per component.
    comp_vertices: dict[int, int] = {}
    comp_edges: dict[int, int] = {}
    for v in verts:
        root = find(int(v))
        comp_vertices[root] = comp_vertices.get(root, 0) + 1
    for a, b in pairs:
        root = find(int(a))
        comp_edges[root] = comp_edges.get(root, 0) + 1

    n_components = len(comp_vertices)

    # Largest component selected by vertex count (ties -> first encountered).
    best_root = max(comp_vertices, key=lambda r: comp_vertices[r])
    s_max = int(comp_vertices[best_root])
    s_max_edges = int(comp_edges.get(best_root, 0))

    deg_vals = np.array([degree[int(v)] for v in verts], dtype=float)
    mean_degree = float(deg_vals.mean())
    degree_var = float(deg_vals.var())

    # Cycle rank of a forest-or-graph: edges - vertices + components.
    cycle_rank = int(m - n_vert + n_components)

    n_multiedge = int(sum(1 for e in comp_edges.values() if e >= 2))

    return {
        "contact_vertices": n_vert,
        "contact_graph_components": int(n_components),
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
# Contour-separation bins
# ---------------------------------------------------------------------------

MIN_CONTOUR_SEPARATION = 3


def default_bin_definitions(n_beads: int) -> dict:
    """Default short/medium/long contour-separation bins for a pilot run.

    short : 3 <= r <= 9
    long  : r / n_beads >= 1/3
    medium: all remaining valid (odd, r >= 3) separations.

    The definitions are returned as plain numbers/strings so they can be
    serialized to JSON (never pickled).
    """
    return {
        "scheme": "fixed_short_scaled_long",
        "min_separation": MIN_CONTOUR_SEPARATION,
        "short": {"r_min": 3, "r_max": 9},
        "long": {"r_over_n_min": 1.0 / 3.0},
        "medium": "remaining valid separations not in short or long",
        "n_beads": int(n_beads),
    }


def _long_threshold(n_beads: int, bin_defs: dict) -> float:
    return float(bin_defs["long"]["r_over_n_min"]) * float(n_beads)


def assign_bin(r: int, n_beads: int, bin_defs: dict) -> str:
    """Return 'short', 'medium', or 'long' for separation r (priority: long, short)."""
    r = int(r)
    long_thr = _long_threshold(n_beads, bin_defs)
    if r >= long_thr:
        return "long"
    if bin_defs["short"]["r_min"] <= r <= bin_defs["short"]["r_max"]:
        return "short"
    return "medium"


def bin_contact_separations(
    m_r: np.ndarray,
    n_beads: int,
    bin_defs: dict,
) -> dict:
    """Aggregate a dense m_r vector into short/medium/long counts.

    Returns ``{"m_short", "m_medium", "m_long"}``.  Categories are exhaustive
    and non-overlapping over valid (odd, r >= 3) separations; the totals sum to
    ``m_r.sum()``.
    """
    m_r = np.asarray(m_r, dtype=np.int64)
    counts = {"short": 0, "medium": 0, "long": 0}
    for r in range(len(m_r)):
        c = int(m_r[r])
        if c == 0:
            continue
        if r < MIN_CONTOUR_SEPARATION:
            # No valid contact below the minimum separation should appear.
            raise ContactMapError(
                f"m_r has {c} contact(s) at invalid separation r={r}"
            )
        counts[assign_bin(r, n_beads, bin_defs)] += c
    total = counts["short"] + counts["medium"] + counts["long"]
    if total != int(m_r.sum()):
        raise ContactMapError(
            f"binned total {total} != m_r sum {int(m_r.sum())}"
        )
    return {
        "m_short": int(counts["short"]),
        "m_medium": int(counts["medium"]),
        "m_long": int(counts["long"]),
    }


def validate_bin_definitions(n_beads: int, bin_defs: dict) -> dict:
    """Check bins are exhaustive, non-overlapping, and non-pathological.

    Returns a dict with per-category valid-separation membership and a list of
    warnings (e.g. empty categories for small N).  Raises on overlap or gaps.
    """
    valid_r = [r for r in range(MIN_CONTOUR_SEPARATION, n_beads) if r % 2 == 1]
    membership = {"short": [], "medium": [], "long": []}
    for r in valid_r:
        membership[assign_bin(r, n_beads, bin_defs)].append(r)
    covered = sorted(membership["short"] + membership["medium"]
                     + membership["long"])
    if covered != valid_r:
        raise ContactMapError(
            "bin definitions are not an exhaustive non-overlapping partition "
            f"of valid separations {valid_r}; covered {covered}"
        )
    warnings = [
        f"bin category {name!r} is empty for n_beads={n_beads}"
        for name in ("short", "medium", "long") if not membership[name]
    ]
    return {"valid_separations": valid_r, "membership": membership,
            "warnings": warnings}


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

    # Four distinct endpoints: compare the two contour intervals [i,j], [k,l].
    if j < k or l < i:
        return "disjoint"          # i<j<k<l or k<l<i<j
    if (i < k and l < j) or (k < i and j < l):
        return "nested"            # i<k<l<j or k<i<j<l
    return "interleaved"           # i<k<j<l or k<i<l<j


def count_pair_motifs(contact_pairs: np.ndarray) -> dict:
    """Count exclusive pair-topology classes over all C(m,2) contact pairs.

    Returns counts keyed ``pair_shared_endpoint``, ``pair_disjoint``,
    ``pair_nested``, ``pair_interleaved``.  Their sum equals C(m,2).
    """
    pairs = np.asarray(contact_pairs, dtype=np.int64).reshape(-1, 2)
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
