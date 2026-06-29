"""Geometry tests: invariance, gyration tensor, scaling (Phase 1 callers)."""
import math

import numpy as np

import isaw_contact_observables as ico
import remd_uniform_chain_2_new as remd
from conftest import random_saw


def test_translation_invariance():
    chain = random_saw(20, 3)
    shifted = [(x + 7, y - 4, z + 11) for (x, y, z) in chain]
    assert abs(ico.radius_of_gyration_squared(chain)
               - ico.radius_of_gyration_squared(shifted)) < 1e-9
    assert abs(ico.end_to_end_distance_squared(chain)
               - ico.end_to_end_distance_squared(shifted)) < 1e-9


def test_cubic_rotation_invariance():
    chain = random_saw(20, 5)
    for M in remd.ROT_MATS[:5]:
        rot = [remd._apply_rot(M, r) for r in chain]
        assert abs(ico.radius_of_gyration_squared(chain)
                   - ico.radius_of_gyration_squared(rot)) < 1e-9
        assert abs(ico.end_to_end_distance_squared(chain)
                   - ico.end_to_end_distance_squared(rot)) < 1e-9


def test_trace_equals_rg2():
    chain = random_saw(20, 9)
    G = ico.gyration_tensor(chain)
    assert abs(np.trace(G) - ico.radius_of_gyration_squared(chain)) < 1e-9


def test_eigenvalues_nonnegative():
    for seed in range(10):
        chain = random_saw(20, seed)
        if chain is None:
            continue
        lam = ico.gyration_eigenvalues(chain)
        assert np.all(lam >= -1e-9)
        assert abs(lam.sum() - ico.radius_of_gyration_squared(chain)) < 1e-9


def test_rg_and_ree_scaling_uses_square():
    chain = random_saw(20, 7)
    rg2 = ico.radius_of_gyration_squared(chain)
    ree2 = ico.end_to_end_distance_squared(chain)
    scale = 0.37
    assert abs((scale ** 2) * rg2 - (scale * math.sqrt(rg2)) ** 2) < 1e-9
    assert (scale ** 2) * ree2 >= 0.0
