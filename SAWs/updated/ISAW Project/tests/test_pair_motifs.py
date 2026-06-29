"""Pair-contact topology tests (offline motif classification)."""
import numpy as np

import isaw_contact_observables as ico
from conftest import random_saw


def test_classification_examples():
    assert ico.classify_contact_pair((0, 5), (6, 9)) == "disjoint"
    assert ico.classify_contact_pair((0, 9), (2, 5)) == "nested"
    assert ico.classify_contact_pair((0, 5), (2, 9)) == "interleaved"
    assert ico.classify_contact_pair((0, 5), (5, 9)) == "shared_endpoint"
    assert ico.classify_contact_pair((0, 5), (0, 9)) == "shared_endpoint"
    # reversed orderings
    assert ico.classify_contact_pair((6, 9), (0, 5)) == "disjoint"
    assert ico.classify_contact_pair((2, 5), (0, 9)) == "nested"


def test_symmetry_under_ordering():
    rng = np.random.RandomState(0)
    for _ in range(300):
        a = tuple(sorted(rng.randint(0, 25, size=2)))
        b = tuple(sorted(rng.randint(0, 25, size=2)))
        if a[0] == a[1] or b[0] == b[1]:
            continue
        assert ico.classify_contact_pair(a, b) == ico.classify_contact_pair(b, a)


def test_pair_counts_sum_to_binomial():
    for seed in range(20):
        chain = random_saw(26, seed)
        if chain is None:
            continue
        cp, _ = ico.build_contact_map(chain)
        counts = ico.count_pair_motifs(cp)
        m = cp.shape[0]
        tot = (counts["pair_shared_endpoint"] + counts["pair_disjoint"]
               + counts["pair_nested"] + counts["pair_interleaved"])
        assert tot == m * (m - 1) // 2
        assert counts["pair_total"] == tot
