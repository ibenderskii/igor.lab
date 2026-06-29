"""Contact-graph (union-find) tests."""
import numpy as np
import pytest

import isaw_contact_observables as ico
from conftest import HAIRPIN6, random_saw

try:
    import networkx as nx
    _HAVE_NX = True
except Exception:
    _HAVE_NX = False


def test_zero_contact_behavior():
    g = ico.contact_graph_summary(np.empty((0, 2), dtype=np.int64), 10)
    assert g["contact_vertices"] == 0
    assert g["contact_graph_components"] == 0
    assert g["largest_component_vertices"] == 0
    assert g["contact_graph_cycle_rank"] == 0


def test_handbuilt_graph_exact():
    cp, _ = ico.build_contact_map(HAIRPIN6)  # edges (0,5),(1,4) -> 2 comps
    g = ico.contact_graph_summary(cp, 6)
    assert g["contact_vertices"] == 4
    assert g["contact_graph_components"] == 2
    assert g["largest_component_vertices"] == 2
    assert g["largest_component_edges"] == 1
    assert g["contact_graph_cycle_rank"] == 0


def test_edge_totals_equal_m():
    for seed in range(20):
        chain = random_saw(26, seed)
        if chain is None:
            continue
        cp, _ = ico.build_contact_map(chain)
        g = ico.contact_graph_summary(cp, len(chain))
        # cycle_rank identity == sum of component edges == m
        assert (g["contact_graph_cycle_rank"]
                == cp.shape[0] - g["contact_vertices"]
                + g["contact_graph_components"])


def test_cycle_rank_triangle():
    # Triangle 0-3, 0-5, 3-5 (all separations >1): 3 edges, 3 verts, 1 comp.
    cp = np.array([[0, 3], [0, 5], [3, 5]], dtype=np.int64)
    g = ico.contact_graph_summary(cp, 8)
    assert g["contact_graph_components"] == 1
    assert g["largest_component_vertices"] == 3
    assert g["contact_graph_cycle_rank"] == 1


def test_tie_break_deterministic():
    # Two equal-size components: {0,3} and {5,8}. Largest by tie-break has the
    # smaller minimum vertex (0).
    cp = np.array([[0, 3], [5, 8]], dtype=np.int64)
    g1 = ico.contact_graph_summary(cp, 10)
    g2 = ico.contact_graph_summary(cp[::-1].copy(), 10)
    assert g1["largest_component_vertices"] == g2["largest_component_vertices"]
    assert g1["largest_component_edges"] == g2["largest_component_edges"]


@pytest.mark.skipif(not _HAVE_NX, reason="networkx not installed")
def test_matches_networkx():
    for seed in range(15):
        chain = random_saw(26, seed)
        if chain is None:
            continue
        cp, _ = ico.build_contact_map(chain)
        g = ico.contact_graph_summary(cp, len(chain))
        G = nx.Graph()
        G.add_edges_from([tuple(p) for p in cp])
        if cp.shape[0] == 0:
            assert g["contact_graph_components"] == 0
            continue
        comps = list(nx.connected_components(G))
        assert g["contact_graph_components"] == len(comps)
        assert g["largest_component_vertices"] == max(len(c) for c in comps)
