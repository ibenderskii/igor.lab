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


def _diag(structural, enabled_flag, n_cycles=40, stride=5):
    Ts = np.linspace(300, 360, 4)
    store = {}
    reps, sp, sa = remd.run_remd(
        N=16, Ts=Ts, steps_per_swap=30, n_cycles=n_cycles,
        model_name=HS["model_name"], params=HS["params"],
        Tref=HS["Tref"], Tscale=HS["Tscale"], seed=4, n_workers=1,
        verbose=False, diagnostics=True, diag_store=store,
        structural_observables=structural, structural_stride=stride)
    # Force ESS warnings with an impossibly high min_ess threshold.
    thr = dict(remd.DEFAULT_DIAG_THRESHOLDS)
    thr["min_ess"] = 1e9
    return remd.compute_run_diagnostics(
        reps, sp, sa, store["walker_temp_index"], Ts,
        burnin_frac=0.5, n_blocks=4, thresholds=thr,
        structural_observables_enabled=enabled_flag)


def test_no_structural_warnings_when_disabled():
    diag = _diag(structural=False, enabled_flag=False)
    types = [w["type"] for w in diag["warnings"]]
    for t in ("low_ess_m_long_fixed", "low_ess_m_global_scaled", "low_ess_smax",
              "low_ess_largest_component_fraction", "structural_drift",
              "insufficient_structural_samples"):
        assert t not in types, t
    assert diag["summary"]["structural_observables_enabled"] is False


def test_no_duplicate_ess_warning_types():
    diag = _diag(structural=True, enabled_flag=True)
    types = [w["type"] for w in diag["warnings"]]
    # Each ESS warning type appears at most once; the old generic "low_ess" is gone.
    assert "low_ess" not in types
    assert len(types) == len(set(types)), types
    # The canonical contact ESS warning is present (min_ess forced very high).
    assert "low_ess_contacts" in types


def test_global_scaled_diagnostics_included():
    diag = _diag(structural=True, enabled_flag=True)
    assert "min_ess_m_global_scaled" in diag["summary"]
    assert "m_global_scaled" in diag["lane_convergence"][0]


def test_diagnostic_npz_bin_provenance(tmp_path):
    import json
    import isaw_contact_observables as ico
    Ts = np.linspace(300, 360, 4)
    store = {}
    custom = ico.project_bin_definitions(16)
    custom["scaled"] = {**ico.SCALED_BIN_DEFINITIONS, "meso_max_ratio": 0.25}
    custom["bin_definition_source"] = "cli_override"
    reps, _, _ = remd.run_remd(
        N=16, Ts=Ts, steps_per_swap=30, n_cycles=20,
        model_name=HS["model_name"], params=HS["params"],
        Tref=HS["Tref"], Tscale=HS["Tscale"], seed=1, n_workers=1,
        verbose=False, diagnostics=True, diag_store=store,
        structural_observables=True, structural_stride=5, bin_defs=custom)
    path = remd.save_diagnostic_trajectories_npz(
        reps, store["walker_temp_index"], Ts, burnin_frac=0.5,
        out_prefix=str(tmp_path / "run"), configured_structural_stride=5,
        bin_definitions=custom)
    d = np.load(path)
    # bin provenance saved even though m_r_post was NOT requested
    assert "m_r_post" not in d
    assert str(d["bin_definition_source"]) == "cli_override"
    scaled = json.loads(str(d["scaled_bin_definitions"]))
    assert scaled["meso_max_ratio"] == 0.25      # the EXACT run definitions
    assert str(d["definitions_version"]) == ico.DEFINITIONS_VERSION


def test_one_sample_structural_stride(tmp_path):
    # A burn-in leaving a single structural sample -> observed spacing sentinel
    # -1, but the configured stride is preserved.
    Ts = np.linspace(300, 360, 3)
    store = {}
    reps, _, _ = remd.run_remd(
        N=16, Ts=Ts, steps_per_swap=20, n_cycles=12,
        model_name=HS["model_name"], params=HS["params"],
        Tref=HS["Tref"], Tscale=HS["Tscale"], seed=1, n_workers=1,
        verbose=False, diagnostics=True, diag_store=store,
        structural_observables=True, structural_stride=5)
    # burnin_frac high so only ~1 structural sample remains post-burn-in
    path = remd.save_diagnostic_trajectories_npz(
        reps, store["walker_temp_index"], Ts, burnin_frac=0.6,
        out_prefix=str(tmp_path / "one"), configured_structural_stride=5)
    d = np.load(path)
    assert int(d["configured_structural_stride"]) == 5
    assert int(d["observed_structural_cycle_spacing"]) in (-1, 5)
    assert int(d["structural_stride"]) == 5      # compat == configured


def test_local_move_freezing_uses_state_changing():
    Ts = np.linspace(300, 360, 3)
    reps, _, _ = remd.run_remd(
        N=16, Ts=Ts, steps_per_swap=20, n_cycles=20,
        model_name=HS["model_name"], params=HS["params"],
        Tref=HS["Tref"], Tscale=HS["Tscale"], seed=2, n_workers=1, verbose=False)
    # A very high threshold forces a warning of the renamed type.
    warns = remd.detect_local_move_freezing(reps, Ts, min_state_changing_rate=2.0)
    assert warns and all(w["type"] == "local_move_freezing" for w in warns)
