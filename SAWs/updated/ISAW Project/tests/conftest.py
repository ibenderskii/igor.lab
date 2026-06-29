"""Shared pytest fixtures/helpers for the ISAW structural-analysis suite."""
import os
import sys

import numpy as np
import pytest

# Make the project modules importable when running from tests/.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

_NN = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]


def random_saw(n, seed):
    """A random self-avoiding walk on the cubic lattice, or None on failure."""
    rng = np.random.RandomState(seed)
    for _ in range(400):
        chain = [(0, 0, 0)]
        occ = {(0, 0, 0)}
        ok = True
        for _ in range(n - 1):
            opts = [(chain[-1][0] + d[0], chain[-1][1] + d[1], chain[-1][2] + d[2])
                    for d in _NN]
            opts = [o for o in opts if o not in occ]
            if not opts:
                ok = False
                break
            nxt = opts[rng.randint(len(opts))]
            chain.append(nxt)
            occ.add(nxt)
        if ok:
            return chain
    return None


@pytest.fixture
def saw():
    return random_saw


# Planar hairpin with two contacts (0,5) r=5 and (1,4) r=3.
HAIRPIN6 = [(0, 0, 0), (1, 0, 0), (2, 0, 0), (2, 1, 0), (1, 1, 0), (0, 1, 0)]
STRAIGHT10 = [(i, 0, 0) for i in range(10)]


@pytest.fixture
def hairpin6():
    return list(HAIRPIN6)
