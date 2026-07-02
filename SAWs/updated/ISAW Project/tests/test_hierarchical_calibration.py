"""Phase 7/14: hierarchical statistical assessability."""
import numpy as np
import pytest

import run_structural_regime_pilot as cal


def _seed(shift=0.0, n_lanes=4, n_post=200, seed=0):
    """Synthetic per-seed trajectories with a genuine low->high-K collapse."""
    rng = np.random.RandomState(seed)
    K = np.linspace(-0.5, 0.5, n_lanes)
    contacts = np.array([5.0 + 8.0 * (k + 0.5) + shift + rng.randn(n_post) * 0.3
                         for k in K])
    rg2 = np.array([20.0 - 12.0 * (k + 0.5) + shift + rng.randn(n_post) * 0.3
                    for k in K])
    mg = np.array([1.0 + 3.0 * (k + 0.5) + rng.randn(n_post) * 0.1 for k in K])
    sm = np.array([2.0 + 5.0 * (k + 0.5) + rng.randn(n_post) * 0.1 for k in K])
    lcf = np.array([0.2 + 0.6 * (k + 0.5) + rng.randn(n_post) * 0.02 for k in K])
    return {"K": K, "contacts": contacts, "rg2": rg2, "m_global": mg,
            "smax": sm, "lcf": lcf}


def test_bootstrap_reports_all_observables_and_fields():
    hb = cal.hierarchical_bootstrap([_seed(seed=1), _seed(shift=0.4, seed=2)],
                                    n_boot=300, n_requested_seeds=2)
    for q in ("delta_contacts", "delta_rg2", "slope_contacts", "slope_rg2",
              "slope_m_global", "slope_smax", "slope_lcf"):
        assert q in hb
        e = hb[q]
        for f in ("estimate", "bootstrap_mean", "ci95", "n_seeds",
                  "n_complete_seeds", "n_requested_seeds", "block_length",
                  "n_bootstrap_replicates", "bootstrap_seed"):
            assert f in e


def test_support_requires_correct_signs():
    hb = cal.hierarchical_bootstrap([_seed(seed=1), _seed(shift=0.3, seed=2)],
                                    n_boot=500, n_requested_seeds=2)
    dm = hb["delta_contacts"]["ci95"]
    dr = hb["delta_rg2"]["ci95"]
    assert dm[0] > 0            # Δ<m> CI excludes 0 from above
    assert dr[1] < 0            # Δ<Rg2> CI excludes 0 from below


def test_bootstrap_is_deterministic():
    a = cal.hierarchical_bootstrap([_seed(seed=1), _seed(seed=2)], n_boot=200)
    b = cal.hierarchical_bootstrap([_seed(seed=1), _seed(seed=2)], n_boot=200)
    assert a["delta_contacts"]["ci95"] == b["delta_contacts"]["ci95"]


# --- assessability rules via the audit --------------------------------------

def _audit(n_requested, n_complete):
    sc = {}
    import tempfile, os
    tmp = tempfile.mkdtemp()
    for s in range(n_requested):
        prefix = os.path.join(tmp, f"s{s}")
        comp = cal._companion_paths(prefix, f"{prefix}_cfg.h5", f"{prefix}_feat.h5")
        if s < n_complete:
            for p in comp.values():
                open(p, "w").close()
        sc[s] = comp
    return cal.audit_seed_artifacts(sc, feature_validator=lambda p: True)


def test_one_seed_not_assessable():
    assert _audit(1, 1)["assessable"] is False


def test_two_complete_seeds_assessable():
    assert _audit(2, 2)["assessable"] is True


def test_two_requested_one_incomplete_not_assessable():
    a = _audit(2, 1)
    assert a["assessable"] is False and a["all_requested_complete"] is False
