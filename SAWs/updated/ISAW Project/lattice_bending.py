#!/usr/bin/env python3
"""Bend counting for self-avoiding walks on the 3D cubic lattice.

A *bend* is a 90-degree turn between two consecutive lattice bonds.  A straight
continuation contributes 0, a right-angle turn contributes 1.  The reduced
(dimensionless) bending penalty used by the samplers is

    u_bend = kappa_bend * n_bend

so that the athermal baseline samples P(X) proportional to exp[-kappa_bend * n_bend(X)].

Only plain functions live here: there is deliberately no class hierarchy and no
generalized angle-potential framework.  :func:`count_bends` is the authoritative
full recount used by tests and by the debug cross-checks in the samplers;
:func:`delta_bends` is the O(1)-per-local-move incremental version used in the
Monte Carlo hot loops.

Coordinates may be a list of integer 3-tuples or an (N, 3) integer array; both
are indexed the same way.  When ``box_size`` is given the bond vectors are taken
under the minimum-image convention, so chains stored in wrapped periodic
coordinates are handled correctly.
"""

from __future__ import annotations

BEND_DEFINITION = "90-degree turns; straight=0, right-angle turn=1"


def _bond(coords, i: int, j: int, box_size=None):
    """Return the validated unit bond vector from bead ``i`` to bead ``j``.

    Raises ValueError unless the (minimum-image) difference is a unit cubic
    lattice vector, i.e. exactly one component is +-1 and the other two are 0.
    """
    a = coords[i]
    b = coords[j]
    d = [int(b[0]) - int(a[0]), int(b[1]) - int(a[1]), int(b[2]) - int(a[2])]

    if box_size is not None:
        L = int(box_size)
        if L < 1:
            raise ValueError(f"box_size must be >= 1, got {box_size!r}")
        for k in range(3):
            # Integer minimum image: fold into (-L/2, L/2].
            v = d[k] % L
            if 2 * v > L:
                v -= L
            d[k] = v

    if abs(d[0]) + abs(d[1]) + abs(d[2]) != 1:
        raise ValueError(
            f"bond {i}->{j} is not a unit cubic-lattice vector: {tuple(d)} "
            f"(box_size={box_size!r})"
        )
    return (d[0], d[1], d[2])


def bend_indicator(coords, center_index: int, box_size=None) -> int:
    """Return 1 if the chain turns 90 degrees at ``center_index``, else 0.

    ``center_index`` must be a valid angle center, i.e. 1 <= center_index <= N-2.
    An immediate reversal (the second bond exactly undoing the first) is a
    self-overlap and cannot occur in a self-avoiding walk; it raises rather than
    being silently scored.
    """
    n = len(coords)
    c = int(center_index)
    if not (1 <= c <= n - 2):
        raise ValueError(
            f"center_index {c} is not a valid angle center for a chain of "
            f"length {n} (expected 1 .. {n - 2})"
        )

    u1 = _bond(coords, c - 1, c, box_size)
    u2 = _bond(coords, c, c + 1, box_size)
    dot = u1[0] * u2[0] + u1[1] * u2[1] + u1[2] * u2[2]

    if dot == 1:
        return 0        # straight continuation
    if dot == 0:
        return 1        # right-angle turn
    raise ValueError(
        f"immediate reversal at center {c}: bonds {u1} and {u2} overlap, "
        "which violates self-avoidance"
    )


def count_bends(coords, box_size=None) -> int:
    """Total number of 90-degree turns in the chain (authoritative recount)."""
    n = len(coords)
    if n < 3:
        return 0
    return sum(bend_indicator(coords, c, box_size) for c in range(1, n - 1))


def delta_bends(old_coords, new_coords, changed_indices, box_size=None) -> int:
    """Change in bend count caused by moving the beads in ``changed_indices``.

    Only the angle centers at ``i-1``, ``i`` and ``i+1`` for each changed bead
    ``i`` can have a different bend indicator, so only that union (clipped to the
    valid centers ``1 .. N-2``) is inspected.
    """
    n = len(new_coords)
    if n != len(old_coords):
        raise ValueError(
            f"old_coords and new_coords must have the same length, got "
            f"{len(old_coords)} and {n}"
        )
    if n < 3:
        return 0

    centers = set()
    for i in changed_indices:
        i = int(i)
        if not (0 <= i < n):
            raise ValueError(f"changed index {i} out of range for chain length {n}")
        for c in (i - 1, i, i + 1):
            if 1 <= c <= n - 2:
                centers.add(c)

    return sum(
        bend_indicator(new_coords, c, box_size) - bend_indicator(old_coords, c, box_size)
        for c in centers
    )


def count_bends_multichain(coords_unwrapped, box_size=None) -> int:
    """Total 90-degree turns summed over every chain of an (M, N, 3) array.

    ``coords_unwrapped`` is iterated over its leading axis, so each element is one
    chain's ``(N, 3)`` coordinates.  This is the authoritative multi-chain recount
    used by the state initializer, the debug cross-checks, and the tests.  The
    multi-chain samplers keep each chain contiguous in UNWRAPPED coordinates (bonds
    are exact unit vectors even when the wrapped configuration crosses a periodic
    boundary), so the default ``box_size=None`` is correct there; the minimum-image
    option is retained for callers that pass wrapped coordinates.
    """
    return int(sum(count_bends(chain, box_size) for chain in coords_unwrapped))
