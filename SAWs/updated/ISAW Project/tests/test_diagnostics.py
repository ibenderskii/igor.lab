"""Convergence/mixing diagnostics tests (Phase 7)."""
import numpy as np

import remd_uniform_chain_2_new as remd

HS = dict(model_name="hs", params=[378.96, 1.39686], Tref=330.0, Tscale=80.0)


def test_autocorr_iid_and_ar1():
    rng = np.random.RandomState(0)
    iid = rng.standard_normal(4000)
    ac = remd.integrated_autocorr_time(iid)
    assert 0.5 < ac["tau_int"] < 2.0
    phi, n = 0.8, 40000
    x = np.empty(n)
    x[0] = 0.0
    noise = rng.standard_normal(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + noise[t]
    ac_ar = remd.integrated_autocorr_time(x)
    tau_expected = (1.0 + phi) / (1.0 - phi)
    assert abs(ac_ar["tau_int"] - tau_expected) < 0.3 * tau_expected


def test_tau_int_cycles_uses_spacing():
    Ts = np.linspace(300, 360, 4)
    store = {}
    reps, sp, sa = remd.run_remd(
        N=16, Ts=Ts, steps_per_swap=30, n_cycles=60,
        model_name=HS["model_name"], params=HS["params"],
        Tref=HS["Tref"], Tscale=HS["Tscale"], seed=3, n_workers=1,
        verbose=False, diagnostics=True, diag_store=store,
        structural_observables=True, structural_stride=3)
    diag = remd.compute_run_diagnostics(
        reps, sp, sa, store["walker_temp_index"], Ts,
        burnin_frac=0.5, n_blocks=4, thresholds=remd.DEFAULT_DIAG_THRESHOLDS)
    lane = diag["lane_convergence"][0]
    # per-cycle observable: spacing 1, tau_int_cycles == tau_int_samples
    c = lane["contacts"]
    assert c["sample_cycle_spacing"] == 1
    assert c["tau_int_cycles"] == c["tau_int_samples"]
    # structural observable: spacing == structural stride (3)
    s = lane["m_long_fixed"]
    assert s["sample_cycle_spacing"] == 3
    if np.isfinite(s["tau_int_samples"]):
        assert abs(s["tau_int_cycles"] - 3 * s["tau_int_samples"]) < 1e-9
    # burn-in metadata: index vs cycle distinguished
    assert "structural_burnin_start_index" in lane
    assert "structural_burnin_start_cycle" in lane


def test_empty_structural_arrays_safe():
    # structural observables OFF -> diagnostics must not crash.
    Ts = np.linspace(300, 360, 4)
    store = {}
    reps, sp, sa = remd.run_remd(
        N=16, Ts=Ts, steps_per_swap=30, n_cycles=40,
        model_name=HS["model_name"], params=HS["params"],
        Tref=HS["Tref"], Tscale=HS["Tscale"], seed=1, n_workers=1,
        verbose=False, diagnostics=True, diag_store=store)
    diag = remd.compute_run_diagnostics(
        reps, sp, sa, store["walker_temp_index"], Ts,
        burnin_frac=0.5, n_blocks=4, thresholds=remd.DEFAULT_DIAG_THRESHOLDS)
    assert diag["summary"]["n_temperatures"] == 4
    lane = diag["lane_convergence"][0]
    assert lane["n_post_burnin_structural"] == 0


def test_local_move_freezing_uses_state_changing():
    Ts = np.linspace(300, 360, 3)
    reps, _, _ = remd.run_remd(
        N=16, Ts=Ts, steps_per_swap=20, n_cycles=20,
        model_name=HS["model_name"], params=HS["params"],
        Tref=HS["Tref"], Tscale=HS["Tscale"], seed=2, n_workers=1, verbose=False)
    # A very high threshold forces a warning of the renamed type.
    warns = remd.detect_local_move_freezing(reps, Ts, min_state_changing_rate=2.0)
    assert warns and all(w["type"] == "local_move_freezing" for w in warns)
