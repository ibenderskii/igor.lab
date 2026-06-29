"""REMD invariants and move-counter (null-move) regression tests (Phases 4,6)."""
import math

import numpy as np
import pytest

import isaw_contact_observables as ico
import remd_uniform_chain_2_new as remd

HS = dict(model_name="hs", params=[378.96, 1.39686], Tref=330.0, Tscale=80.0)


def _run(nworkers, n_cycles=30, N=16, seed=7, **kw):
    Ts = np.linspace(300, 360, 5)
    return remd.run_remd(
        N=N, Ts=Ts, steps_per_swap=30, n_cycles=n_cycles,
        model_name=HS["model_name"], params=HS["params"],
        Tref=HS["Tref"], Tscale=HS["Tscale"], seed=seed,
        n_workers=nworkers, verbose=False, **kw)


def test_state_m_matches_contact_count():
    reps, _, _ = _run(1)
    for rep in reps:
        assert rep.state.m == int(round(
            remd.contact_count(rep.state.chain, rep.state.occ)))


def test_canonical_keys_preserved():
    reps, _, _ = _run(1)
    dist = remd.build_distributions(reps, rg_bins=20, burnin_frac=0.5)
    for key in ("Ts", "c_vals", "Pc", "rg_edges", "rg_centers", "Prg",
                "rg_edges_lattice", "rg_centers_lattice", "rg_scale",
                "temps", "ct_centers", "ct_hists", "rg_hists"):
        assert key in dist
    # model metadata is attached separately and preserved.
    remd.attach_model_metadata(dist, HS["model_name"], ["h", "s"],
                               HS["params"], HS["Tref"], HS["Tscale"])
    for key in ("model_name", "param_names", "model_params", "Tref", "Tscale"):
        assert key in dist


def test_default_does_not_retain_structural():
    # Phase 4: ordinary REMD does not build/retain structural trajectories.
    reps, _, _ = _run(1)
    for rep in reps:
        assert len(rep.m_long_traj) == 0
        assert len(rep.m_r_traj) == 0
        # scalar trajectories are still recorded every cycle
        assert len(rep.C_traj) == 30
        assert all(math.isfinite(v) for v in rep.Rg2_traj)


def test_structural_observables_opt_in():
    reps, _, _ = _run(1, n_cycles=40, structural_observables=True,
                      structural_stride=4)
    for rep in reps:
        assert len(rep.m_long_traj) == 10           # ceil(40/4)
        assert len(rep.m_global_scaled_traj) == 10
        assert len(rep.C_traj) == 40


def test_move_counter_ordering():
    reps, _, _ = _run(1, structural_observables=True)
    for rep in reps:
        c = np.asarray(rep.move_counters)
        # accepted <= state_changing <= geometrically_valid <= proposed
        assert np.all(c[:, 3] <= c[:, 2])
        assert np.all(c[:, 2] <= c[:, 1])
        assert np.all(c[:, 1] <= c[:, 0])


def test_null_moves_not_state_changing():
    # A straight chain admits pivots about its own axis that are geometrically
    # valid but leave the chain unchanged (null moves).  mc_sweep must count
    # them as valid but not as state-changing.
    import random
    random.seed(0)
    chain = [(i, 0, 0) for i in range(8)]
    rep = remd.Replica(T=1000.0, state=remd.ChainState.initial_straight(
        8, 1000.0, "hs", [0.0, 0.0], 1.0, 1.0))
    remd.mc_sweep(rep, 400, "hs", [0.0, 0.0], 1.0, 1.0)
    c = np.asarray(rep.move_counters)
    valid = int(c[:, 1].sum())
    state_changing = int(c[:, 2].sum())
    accepted = int(c[:, 3].sum())
    # there must have been at least one valid-but-not-state-changing (null) move
    assert state_changing < valid
    assert accepted <= state_changing


def test_serial_parallel_invariants():
    for nw in (1, 2):
        reps, _, _ = _run(nw, structural_observables=True)
        for rep in reps:
            cp, _ = ico.build_contact_map(rep.state.chain)
            assert cp.shape[0] == rep.state.m


def test_walker_mapping_is_permutation():
    Ts = np.linspace(300, 360, 5)
    store = {}
    remd.run_remd(N=16, Ts=Ts, steps_per_swap=30, n_cycles=25,
                  model_name=HS["model_name"], params=HS["params"],
                  Tref=HS["Tref"], Tscale=HS["Tscale"], seed=3,
                  n_workers=1, verbose=False, diagnostics=True, diag_store=store)
    wti = store["walker_temp_index"]
    for c in range(wti.shape[0]):
        assert sorted(wti[c].tolist()) == list(range(len(Ts)))


def test_move_csv_ratios(tmp_path):
    reps, _, _ = _run(1, structural_observables=True)
    Ts = np.linspace(300, 360, 5)
    path = remd.save_move_acceptance_csv(reps, Ts, str(tmp_path / "run"))
    import csv
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        prop = int(r["proposed"])
        if prop == 0:
            continue
        sc = int(r["state_changing"])
        acc = int(r["metropolis_accepted"])
        assert acc <= sc <= int(r["geometrically_valid"]) <= prop
        if sc > 0:
            assert abs(float(r["a_metropolis"]) - acc / sc) < 1e-6
        assert abs(float(r["a_total"]) - acc / prop) < 1e-6
        assert abs(float(r["a_state_changing"]) - sc / prop) < 1e-6
