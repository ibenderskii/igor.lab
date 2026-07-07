#!/usr/bin/env python3
"""Direct-K finite-size crossover diagnostic for the lattice polymer.

Runs the REMD sampler in direct-K mode (P(C|K) proportional to exp[K m(C)],
reduced bias b_i = -K_i, NO temperature mapping) over an explicit K ladder for
several independent seeds, saves the standard per-seed sampler outputs and
coordinate snapshots, runs the existing offline feature extractor, then
summarizes the structural response versus K and estimates the finite-N
pseudotransition region.

This is a *diagnostic* driver: unlike run_structural_regime_pilot.py it does not
map K to a PNIPAM temperature, does not run scientific gates, and does not plan a
production campaign.  It answers a single question -- where, in K, does an N-bead
lattice polymer cross over from coil to globule -- and whether the scanned K
range brackets that crossover.

The per-seed sampler outputs (results.csv, distributions.npz, diagnostics.json,
diagnostic_trajectories.npz, configurations.h5, features.h5) are the normal
direct-K sampler artifacts and are kept.  The driver adds four summary files:

    direct_K_report.json          machine-readable summary + transition estimate
    direct_K_report.md            concise human-readable report
    direct_K_response_curves.csv  per-lane observables and dO/dK response curves
    run_manifest.json             seeds, controls, artifact paths, commands

Example (broad N=30 scan)
-------------------------
    python run_direct_k_diagnostic.py \
        --N 30 \
        --K-values=-0.40,-0.32,-0.24,-0.16,-0.08,0.00,0.08,0.14,0.20,0.25,0.30,0.35 \
        --seeds 101 202 \
        --n-workers 8 \
        --n-cycles 5000 \
        --steps-per-swap 60 \
        --burnin-frac 0.5 \
        --structural-stride 5 \
        --snapshot-stride 5 \
        --run-id N30_direct_K_broad_v1 \
        --output-dir direct_K_outputs

The pure analysis helpers (finite_difference, find_peak, assess_transition,
fluctuation_dissipation_check) are unit-tested in tests/test_direct_k.py.
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import remd_uniform_chain_2_new as remd  # noqa: E402

PY = sys.executable
REMD = str(HERE / "remd_uniform_chain_2_new.py")
EXTRACT = str(HERE / "extract_contact_motif_features.py")

# Adequacy thresholds for the diagnostic (documented; not scientific gates).
DEFAULT_MIN_ESS = 200.0
DEFAULT_MIN_ROUND_TRIPS = 5
DEFAULT_MIN_SWAP_RATE = 0.10
DEFAULT_MAX_SWAP_RATE = 0.90
DEFAULT_MIN_ADJ_OVERLAP = 0.20


# ---------------------------------------------------------------------------
# Pure analysis helpers (unit-tested)
# ---------------------------------------------------------------------------

def finite_difference(K, y) -> np.ndarray:
    """dy/dK evaluated at each K by a central/one-sided finite difference.

    Uses numpy.gradient, which is exact for the endpoints (one-sided) and
    interior points (central) and correctly handles a non-uniform K ladder.
    Non-finite y entries propagate as NaN in the derivative.
    """
    K = np.asarray(K, float)
    y = np.asarray(y, float)
    if K.size < 2:
        return np.full(K.shape, np.nan)
    return np.gradient(y, K)


def find_peak(K, values) -> dict:
    """Locate the maximum of ``values`` over the K ladder.

    Returns the peak coupling, its lane index, the peak value, and whether the
    peak sits at a ladder endpoint (index 0 or n-1).  All-NaN input yields a
    null peak.
    """
    K = np.asarray(K, float)
    v = np.asarray(values, float)
    finite = np.isfinite(v)
    if not np.any(finite):
        return {"K_peak": None, "index": None, "value": None,
                "at_endpoint": None}
    idx = int(np.nanargmax(np.where(finite, v, -np.inf)))
    at_end = bool(idx == 0 or idx == K.size - 1)
    return {"K_peak": float(K[idx]), "index": idx, "value": float(v[idx]),
            "at_endpoint": at_end}


def fluctuation_dissipation_check(dm_dK, var_m) -> dict:
    """Compare d<m>/dK with Var(m) lane-by-lane (fluctuation-response identity).

    For P(C|K) proportional to exp[K m], d<m>/dK = Var(m) exactly; finite
    sampling and the finite-difference derivative make this only approximate, so
    this is REPORTED as a diagnostic, never asserted.
    """
    a = np.asarray(dm_dK, float)
    b = np.asarray(var_m, float)
    valid = np.isfinite(a) & np.isfinite(b)
    if not np.any(valid):
        return {"n_compared": 0, "max_abs_diff": None, "mean_abs_diff": None,
                "correlation": None, "per_lane_abs_diff": []}
    diff = np.abs(a - b)
    corr = (float(np.corrcoef(a[valid], b[valid])[0, 1])
            if int(valid.sum()) >= 2
            and np.std(a[valid]) > 0 and np.std(b[valid]) > 0 else None)
    return {
        "n_compared": int(valid.sum()),
        "max_abs_diff": float(np.nanmax(diff[valid])),
        "mean_abs_diff": float(np.nanmean(diff[valid])),
        "correlation": corr,
        "per_lane_abs_diff": [None if not f else float(d)
                              for f, d in zip(valid, diff)],
        "note": ("d<m>/dK == Var(m) exactly for exp[K m]; deviations here are "
                 "finite-sampling and finite-difference error."),
    }


def assess_transition(K, var_m, dm_dK, rg2_response, network_response) -> dict:
    """Estimate the finite-N pseudotransition region from the response curves.

    ``rg2_response`` is the SIZE response -d<Rg^2>/dK and ``network_response`` is
    d<S_max/N>/dK; both are passed already sign-oriented so that their MAXIMUM
    marks the strongest response.  Reports the four peak locations, a consensus
    estimate (mean of finite interior peaks), whether the transition is
    bracketed by the scan, and a recommendation for the next scan.
    """
    K = np.asarray(K, float)
    peaks = {
        "K_peak_contact_variance": find_peak(K, var_m),
        "K_peak_contact_derivative": find_peak(K, dm_dK),
        "K_peak_Rg2_response": find_peak(K, rg2_response),
        "K_peak_network_response": find_peak(K, network_response),
    }
    # Consensus: mean of the finite peak positions that are NOT at an endpoint.
    interior = [p["K_peak"] for p in peaks.values()
                if p["K_peak"] is not None and p["at_endpoint"] is False]
    consensus = float(np.mean(interior)) if interior else None

    var_peak = peaks["K_peak_contact_variance"]
    rg2_peak = peaks["K_peak_Rg2_response"]
    # The contact-variance peak is the primary crossover estimate; count how many
    # ladder points lie strictly below/above it.
    n_below = n_above = 0
    if var_peak["K_peak"] is not None:
        n_below = int(np.sum(K < var_peak["K_peak"]))
        n_above = int(np.sum(K > var_peak["K_peak"]))

    transition_bracketed = bool(
        var_peak["at_endpoint"] is False
        and rg2_peak["at_endpoint"] is False
        and n_below >= 2
        and n_above >= 2
    )

    # Next-scan recommendation.
    if var_peak["index"] is None:
        recommendation = "no finite contact-variance peak; check the sampler run"
        extend = "unknown"
    elif var_peak["index"] == 0:
        recommendation = (
            f"contact-variance peak is at the lowest K={var_peak['K_peak']:.3g}; "
            f"extend the K range LOWER")
        extend = "lower"
    elif var_peak["index"] == K.size - 1:
        recommendation = (
            f"contact-variance peak is at the highest K={var_peak['K_peak']:.3g}; "
            f"extend the K range HIGHER")
        extend = "higher"
    else:
        recommendation = (
            f"interior peak near K={var_peak['K_peak']:.3g}; "
            f"refine the ladder around the interior peak")
        extend = "refine_interior"

    return {
        "peaks": peaks,
        "consensus_K_transition": consensus,
        "n_interior_peaks": len(interior),
        "n_lanes_below_variance_peak": n_below,
        "n_lanes_above_variance_peak": n_above,
        "transition_bracketed": transition_bracketed,
        "recommended_scan_direction": extend,
        "recommendation": recommendation,
    }


def bhattacharyya(p, q) -> float:
    """Bhattacharyya overlap coefficient of two (unnormalized) histograms."""
    p = np.asarray(p, float); q = np.asarray(q, float)
    p = np.where(np.isfinite(p), p, 0.0)
    q = np.where(np.isfinite(q), q, 0.0)
    sp, sq = p.sum(), q.sum()
    if sp <= 0 or sq <= 0:
        return 0.0
    return float(np.sqrt((p / sp) * (q / sq)).sum())


# ---------------------------------------------------------------------------
# Per-seed sampler + extractor
# ---------------------------------------------------------------------------

def seed_prefix(out_dir: Path, N: int, seed: int) -> Path:
    return out_dir / f"directK_N{N}_s{seed}"


def run_seed(out_dir, N, k_csv, seed, n_cycles, steps_per_swap, burnin_frac,
             structural_stride, snapshot_stride, n_workers,
             resume=False, commands=None) -> str:
    """Run one direct-K REMD seed and extract features; return the out-prefix.

    On ``resume`` an existing, complete seed (features.h5 + results.csv +
    run_summary.json present) is reused without rerunning.
    """
    prefix = seed_prefix(out_dir, N, seed)
    cfg = f"{prefix}_configurations.h5"
    feat = out_dir / f"directK_N{N}_s{seed}_features.h5"
    if resume and Path(feat).exists() and Path(f"{prefix}_results.csv").exists() \
            and Path(f"{prefix}_run_summary.json").exists():
        print(f"RESUME: seed {seed} reused ({feat})")
        return str(prefix)
    cmd = [
        PY, REMD, "--N", str(N), f"--K-values={k_csv}",
        "--steps-per-swap", str(steps_per_swap), "--n-cycles", str(n_cycles),
        "--n-workers", str(n_workers), "--seed", str(seed),
        "--burnin-frac", str(float(burnin_frac)),
        "--structural-observables", "--structural-stride", str(structural_stride),
        "--diagnostics", "--diagnostic-trajectories",
        "--save-configurations", "--snapshot-stride", str(snapshot_stride),
        "--overwrite-configurations", "--no-plots",
        "--out-prefix", str(prefix),
    ]
    if commands is not None:
        commands.append(" ".join(cmd))
    print("RUN:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(HERE))
    ecmd = [PY, EXTRACT, "--input", cfg, "--output", str(feat),
            "--validate", "--overwrite"]
    if commands is not None:
        commands.append(" ".join(ecmd))
    print("RUN:", " ".join(ecmd))
    subprocess.run(ecmd, check=True, cwd=str(HERE))
    return str(prefix)


# ---------------------------------------------------------------------------
# Reading per-seed artifacts
# ---------------------------------------------------------------------------

# results.csv column -> summary key.
_RESULT_COLS = {
    "C_mean": "mean_contacts",
    "C_std": "contacts_std",
    "Rg2_mean": "mean_rg2",
    "Ree2_mean": "mean_ree2",
    "m_long_fixed_mean": "mean_m_long",
    "m_global_scaled_mean": "mean_m_global",
    "Smax_mean": "mean_smax",
    "largest_component_fraction_mean": "mean_lcf",
}


def read_seed_results(prefix) -> dict:
    """Read a seed's results.csv into per-lane arrays keyed by ascending K."""
    rows = list(csv.DictReader(open(f"{prefix}_results.csv", newline="")))
    K = np.array([float(r["K"]) for r in rows], float)
    order = np.argsort(K)
    out = {"K": K[order]}
    for col, key in _RESULT_COLS.items():
        out[key] = np.array([float(rows[i][col]) for i in order], float)
    return out


def read_seed_diagnostics(prefix) -> dict:
    """Read a seed's diagnostics.json for per-lane tau/ESS and swap/round-trips."""
    with open(f"{prefix}_diagnostics.json") as fh:
        d = json.load(fh)
    lanes = d.get("lane_convergence", [])
    lanes = sorted(lanes, key=lambda x: int(x["temperature_index"]))
    tau = np.array([float(l["contacts"]["tau_int"]) for l in lanes], float)
    ess = np.array([float(l["contacts"]["ess"]) for l in lanes], float)
    swap = np.array([float(x) for x in d.get("swap_rates", [])], float)
    summ = d.get("summary", {})
    return {
        "tau_int_contacts": tau,
        "ess_contacts": ess,
        "swap_rates": swap,
        "total_round_trips_low": int(summ.get("total_round_trips_low", 0)),
        "min_temp_coverage": float(summ.get("min_temp_coverage", float("nan"))),
    }


def read_seed_overlap(prefix) -> np.ndarray:
    """Adjacent-lane P(m) Bhattacharyya overlap from a seed's distributions.npz."""
    with np.load(f"{prefix}_distributions.npz", allow_pickle=True) as dd:
        Pc = np.asarray(dd["Pc"], float)
    nT = Pc.shape[0]
    return np.array([bhattacharyya(Pc[i], Pc[i + 1]) for i in range(nT - 1)],
                    float)


# ---------------------------------------------------------------------------
# Cross-seed aggregation + response curves
# ---------------------------------------------------------------------------

def aggregate(prefixes, K_ladder) -> dict:
    """Average per-lane observables/diagnostics across seeds (index-aligned)."""
    K = np.asarray(K_ladder, float)
    res = [read_seed_results(p) for p in prefixes]
    diag = [read_seed_diagnostics(p) for p in prefixes]
    overlaps = [read_seed_overlap(p) for p in prefixes]
    for r in res:
        if r["K"].shape != K.shape or not np.allclose(r["K"], K, atol=1e-9):
            raise ValueError(f"seed K ladder {r['K']} != requested {K}")

    def _mean(key, source):
        return np.nanmean(np.vstack([s[key] for s in source]), axis=0)

    agg = {"K": K}
    for key in _RESULT_COLS.values():
        agg[key] = _mean(key, res)
    # Var(m) = mean over seeds of the within-lane contact variance (std^2).
    agg["var_contacts"] = np.nanmean(
        np.vstack([r["contacts_std"] ** 2 for r in res]), axis=0)
    agg["tau_int_contacts"] = _mean("tau_int_contacts", diag)
    agg["ess_contacts"] = _mean("ess_contacts", diag)
    # mean_smax_over_N is derived from the real N in response_curves().
    agg["adjacent_overlap"] = np.nanmean(np.vstack(overlaps), axis=0)
    agg["swap_rates"] = np.nanmean(np.vstack([d["swap_rates"] for d in diag]),
                                   axis=0)
    agg["round_trips_low_per_seed"] = [int(d["total_round_trips_low"])
                                       for d in diag]
    agg["min_temp_coverage_per_seed"] = [float(d["min_temp_coverage"])
                                         for d in diag]
    return agg


def response_curves(agg, N) -> dict:
    """Finite-difference response curves d<m>/dK, -d<Rg2>/dK, d<Smax/N>/dK."""
    K = agg["K"]
    smax_over_N = agg["mean_smax"] / float(N)
    dm_dK = finite_difference(K, agg["mean_contacts"])
    drg2_dK = finite_difference(K, agg["mean_rg2"])
    dsmax_dK = finite_difference(K, smax_over_N)
    return {
        "smax_over_N": smax_over_N,
        "dm_dK": dm_dK,
        "neg_drg2_dK": -drg2_dK,          # SIZE response (peaks at strongest)
        "dsmax_over_N_dK": dsmax_dK,      # NETWORK response
    }


# ---------------------------------------------------------------------------
# Report assembly + rendering
# ---------------------------------------------------------------------------

def _adequacy(agg, thresholds) -> dict:
    ess = agg["ess_contacts"][np.isfinite(agg["ess_contacts"])]
    swap = agg["swap_rates"][np.isfinite(agg["swap_rates"])]
    overlap = agg["adjacent_overlap"][np.isfinite(agg["adjacent_overlap"])]
    min_ess = float(ess.min()) if ess.size else float("nan")
    min_swap = float(swap.min()) if swap.size else float("nan")
    max_swap = float(swap.max()) if swap.size else float("nan")
    min_overlap = float(overlap.min()) if overlap.size else float("nan")
    min_round_trips = (int(min(agg["round_trips_low_per_seed"]))
                       if agg["round_trips_low_per_seed"] else 0)
    ess_ok = bool(np.isfinite(min_ess) and min_ess >= thresholds["min_ess"])
    rt_ok = bool(min_round_trips >= thresholds["min_round_trips"])
    swap_ok = bool(np.isfinite(min_swap)
                   and min_swap >= thresholds["min_swap_rate"]
                   and max_swap <= thresholds["max_swap_rate"])
    overlap_ok = bool(np.isfinite(min_overlap)
                      and min_overlap >= thresholds["min_adjacent_overlap"])
    return {
        "min_ess_contacts": min_ess,
        "min_round_trips_low": min_round_trips,
        "min_swap_rate": min_swap,
        "max_swap_rate": max_swap,
        "min_adjacent_overlap": min_overlap,
        "ess_adequate": ess_ok,
        "round_trips_adequate": rt_ok,
        "swap_rates_adequate": swap_ok,
        "overlap_adequate": overlap_ok,
        "all_adequate": bool(ess_ok and rt_ok and swap_ok and overlap_ok),
        "thresholds": thresholds,
    }


def build_report(run_id, N, K_ladder, seeds, controls, agg, curves,
                 transition, fd_check, adequacy, prefixes) -> dict:
    K = [float(x) for x in agg["K"]]

    def _lst(a):
        return [None if not np.isfinite(v) else float(v)
                for v in np.asarray(a, float)]

    per_lane = []
    for i, k in enumerate(K):
        per_lane.append({
            "K": k,
            "mean_contacts": _lst(agg["mean_contacts"])[i],
            "var_contacts": _lst(agg["var_contacts"])[i],
            "mean_rg2": _lst(agg["mean_rg2"])[i],
            "mean_ree2": _lst(agg["mean_ree2"])[i],
            "mean_m_long": _lst(agg["mean_m_long"])[i],
            "mean_m_global": _lst(agg["mean_m_global"])[i],
            "mean_smax": _lst(agg["mean_smax"])[i],
            "mean_smax_over_N": _lst(curves["smax_over_N"])[i],
            "mean_largest_component_fraction": _lst(agg["mean_lcf"])[i],
            "tau_int_contacts": _lst(agg["tau_int_contacts"])[i],
            "ess_contacts": _lst(agg["ess_contacts"])[i],
            "dm_dK": _lst(curves["dm_dK"])[i],
            "neg_drg2_dK": _lst(curves["neg_drg2_dK"])[i],
            "dsmax_over_N_dK": _lst(curves["dsmax_over_N_dK"])[i],
        })
    return {
        "schema": "direct_K_diagnostic_v1",
        "run_id": run_id,
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "control_mode": "direct_K",
        "control_parameter": "K",
        "temperature_mapping_applied": False,
        "N": int(N),
        "K_values": [float(x) for x in K_ladder],
        "seeds": [int(s) for s in seeds],
        "n_seeds": len(seeds),
        "sampler_controls": controls,
        "run_completed": True,
        "per_lane": per_lane,
        "adjacent_overlap": _lst(agg["adjacent_overlap"]),
        "swap_rates": _lst(agg["swap_rates"]),
        "round_trips_low_per_seed": agg["round_trips_low_per_seed"],
        "response_curves": {
            "dm_dK": _lst(curves["dm_dK"]),
            "neg_drg2_dK": _lst(curves["neg_drg2_dK"]),
            "dsmax_over_N_dK": _lst(curves["dsmax_over_N_dK"]),
        },
        "fluctuation_dissipation_check": fd_check,
        "transition": transition,
        "sampling_adequacy": adequacy,
        "seed_prefixes": [str(p) for p in prefixes],
    }


def write_response_curves_csv(path, agg, curves) -> None:
    cols = ["K", "mean_contacts", "var_contacts", "mean_rg2", "mean_ree2",
            "mean_m_long", "mean_m_global", "mean_smax", "mean_smax_over_N",
            "mean_largest_component_fraction", "tau_int_contacts",
            "ess_contacts", "dm_dK", "neg_drg2_dK", "dsmax_over_N_dK"]
    src = {
        "K": agg["K"], "mean_contacts": agg["mean_contacts"],
        "var_contacts": agg["var_contacts"], "mean_rg2": agg["mean_rg2"],
        "mean_ree2": agg["mean_ree2"], "mean_m_long": agg["mean_m_long"],
        "mean_m_global": agg["mean_m_global"], "mean_smax": agg["mean_smax"],
        "mean_smax_over_N": curves["smax_over_N"],
        "mean_largest_component_fraction": agg["mean_lcf"],
        "tau_int_contacts": agg["tau_int_contacts"],
        "ess_contacts": agg["ess_contacts"], "dm_dK": curves["dm_dK"],
        "neg_drg2_dK": curves["neg_drg2_dK"],
        "dsmax_over_N_dK": curves["dsmax_over_N_dK"],
    }
    n = len(agg["K"])
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for i in range(n):
            w.writerow([f"{float(src[c][i]):.8g}" for c in cols])


def render_markdown(report) -> str:
    t = report["transition"]
    adq = report["sampling_adequacy"]
    fd = report["fluctuation_dissipation_check"]
    peaks = t["peaks"]

    def _pk(name):
        p = peaks[name]
        if p["K_peak"] is None:
            return "n/a"
        tag = " (endpoint)" if p["at_endpoint"] else ""
        return f"K={p['K_peak']:.4g}{tag}"

    var_peak = peaks["K_peak_contact_variance"]
    rg2_peak = peaks["K_peak_Rg2_response"]
    agree = "yes" if (var_peak["K_peak"] is not None
                      and rg2_peak["K_peak"] is not None
                      and abs(var_peak["K_peak"] - rg2_peak["K_peak"])
                      <= 0.5 * (max(report["K_values"]) - min(report["K_values"]))
                      / max(1, len(report["K_values"]) - 1) * 2) else "partial/no"
    consensus = ("n/a" if t["consensus_K_transition"] is None
                 else f"{t['consensus_K_transition']:.4g}")
    lines = [
        f"# Direct-K crossover diagnostic — {report['run_id']}",
        "",
        f"- **N (beads):** {report['N']}",
        f"- **K ladder:** {', '.join(f'{k:.3g}' for k in report['K_values'])}",
        f"- **Seeds:** {', '.join(str(s) for s in report['seeds'])} "
        f"({report['n_seeds']} independent)",
        f"- **Control mode:** direct_K (P(C|K) ∝ exp[K·m]; b = −K; "
        f"no temperature mapping)",
        "",
        "## Answers",
        "",
        f"1. **Did the run complete?** "
        f"{'Yes' if report['run_completed'] else 'No'} "
        f"({report['n_seeds']} seed(s) sampled and extracted).",
        f"2. **Was the transition bracketed?** "
        f"{'YES' if t['transition_bracketed'] else 'NO'} "
        f"({t['n_lanes_below_variance_peak']} lanes below and "
        f"{t['n_lanes_above_variance_peak']} above the contact-variance peak).",
        f"3. **Largest contact fluctuation (Var(m)) at:** {_pk('K_peak_contact_variance')}.",
        f"4. **Strongest size response (−d⟨Rg²⟩/dK) at:** {_pk('K_peak_Rg2_response')}.",
        f"5. **Do observables agree?** {agree}. Peaks: "
        f"contact-variance {_pk('K_peak_contact_variance')}, "
        f"d⟨m⟩/dK {_pk('K_peak_contact_derivative')}, "
        f"size {_pk('K_peak_Rg2_response')}, "
        f"network {_pk('K_peak_network_response')}; "
        f"consensus K ≈ {consensus}.",
        f"6. **Are ESS, swap rates, and round trips adequate?** "
        f"{'Yes' if adq['all_adequate'] else 'No'} — "
        f"min ESS(contacts) {adq['min_ess_contacts']:.1f} "
        f"(≥{adq['thresholds']['min_ess']:.0f}? "
        f"{'ok' if adq['ess_adequate'] else 'LOW'}), "
        f"swap∈[{adq['min_swap_rate']:.2f},{adq['max_swap_rate']:.2f}] "
        f"({'ok' if adq['swap_rates_adequate'] else 'out-of-band'}), "
        f"min round trips {adq['min_round_trips_low']} "
        f"({'ok' if adq['round_trips_adequate'] else 'LOW'}), "
        f"min adjacent overlap {adq['min_adjacent_overlap']:.2f} "
        f"({'ok' if adq['overlap_adequate'] else 'LOW'}).",
        f"7. **Next scan:** {report['transition']['recommendation']}.",
        "",
        "## Fluctuation–response check",
        "",
        f"d⟨m⟩/dK should equal Var(m). Compared {fd['n_compared']} lanes: "
        f"max |Δ| = "
        f"{'n/a' if fd['max_abs_diff'] is None else f'{fd['max_abs_diff']:.3g}'}, "
        f"mean |Δ| = "
        f"{'n/a' if fd['mean_abs_diff'] is None else f'{fd['mean_abs_diff']:.3g}'}, "
        f"correlation = "
        f"{'n/a' if fd['correlation'] is None else f'{fd['correlation']:.3f}'}.",
        "",
        "## Peak estimates",
        "",
        "| Estimator | K peak | at endpoint |",
        "| --- | --- | --- |",
    ]
    for name in ("K_peak_contact_variance", "K_peak_contact_derivative",
                 "K_peak_Rg2_response", "K_peak_network_response"):
        p = peaks[name]
        kp = "n/a" if p["K_peak"] is None else f"{p['K_peak']:.4g}"
        ae = "" if p["at_endpoint"] is None else ("yes" if p["at_endpoint"]
                                                  else "no")
        lines.append(f"| {name} | {kp} | {ae} |")
    lines += [
        "",
        f"Consensus K (mean of interior peaks): **{consensus}**.",
        "",
        "See `direct_K_report.json` and `direct_K_response_curves.csv` for the "
        "full per-lane numbers.",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Direct-K coil-to-globule crossover diagnostic.")
    ap.add_argument("--N", type=int, required=True, help="number of beads")
    ap.add_argument("--K-values", dest="K_values", required=True,
                    help="comma-separated K ladder, e.g. --K-values=-0.4,...,0.35")
    ap.add_argument("--seeds", type=int, nargs="+", default=[101, 202],
                    help="independent REMD seeds (run sequentially)")
    ap.add_argument("--n-workers", type=int, default=8,
                    help="lane-worker processes within each seed")
    ap.add_argument("--n-cycles", type=int, default=5000)
    ap.add_argument("--steps-per-swap", type=int, default=60)
    ap.add_argument("--burnin-frac", type=float, default=0.5)
    ap.add_argument("--structural-stride", type=int, default=5)
    ap.add_argument("--snapshot-stride", type=int, default=5)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--resume", action="store_true",
                    help="reuse complete per-seed artifacts instead of rerunning")
    ap.add_argument("--overwrite", action="store_true",
                    help="allow writing into an existing output directory")
    ap.add_argument("--min-ess", type=float, default=DEFAULT_MIN_ESS)
    ap.add_argument("--min-round-trips", type=int, default=DEFAULT_MIN_ROUND_TRIPS)
    ap.add_argument("--min-swap-rate", type=float, default=DEFAULT_MIN_SWAP_RATE)
    ap.add_argument("--max-swap-rate", type=float, default=DEFAULT_MAX_SWAP_RATE)
    ap.add_argument("--min-adjacent-overlap", type=float,
                    default=DEFAULT_MIN_ADJ_OVERLAP)
    args = ap.parse_args()

    K_ladder = remd.parse_k_values(args.K_values)   # sorted ascending; validated
    n_lanes = len(K_ladder)
    if not (1 <= args.n_workers <= n_lanes):
        raise SystemExit(
            f"--n-workers must satisfy 1 <= n_workers <= number of K lanes "
            f"({n_lanes}); got {args.n_workers}.")
    if args.N < 3:
        raise SystemExit("--N must be >= 3")
    if len(set(args.seeds)) != len(args.seeds):
        raise SystemExit("--seeds must be distinct")

    run_id = args.run_id or f"directK_N{args.N}"
    out_dir = Path(args.output_dir or f"direct_K_{run_id}")
    if out_dir.exists() and not (args.overwrite or args.resume):
        raise SystemExit(
            f"output directory {out_dir} exists; pass --overwrite or --resume.")
    out_dir.mkdir(parents=True, exist_ok=True)

    k_csv = ",".join(f"{k:.6g}" for k in K_ladder)
    controls = {
        "n_cycles": int(args.n_cycles),
        "steps_per_swap": int(args.steps_per_swap),
        "burnin_frac": float(args.burnin_frac),
        "structural_stride": int(args.structural_stride),
        "snapshot_stride": int(args.snapshot_stride),
        "n_workers": int(args.n_workers),
    }
    thresholds = {
        "min_ess": float(args.min_ess),
        "min_round_trips": int(args.min_round_trips),
        "min_swap_rate": float(args.min_swap_rate),
        "max_swap_rate": float(args.max_swap_rate),
        "min_adjacent_overlap": float(args.min_adjacent_overlap),
    }

    commands: list[str] = []
    prefixes = []
    for seed in args.seeds:            # sequential, as the calibration driver
        prefixes.append(run_seed(
            out_dir, args.N, k_csv, seed,
            n_cycles=args.n_cycles, steps_per_swap=args.steps_per_swap,
            burnin_frac=args.burnin_frac,
            structural_stride=args.structural_stride,
            snapshot_stride=args.snapshot_stride, n_workers=args.n_workers,
            resume=args.resume, commands=commands))

    agg = aggregate(prefixes, K_ladder)
    curves = response_curves(agg, args.N)
    fd_check = fluctuation_dissipation_check(curves["dm_dK"], agg["var_contacts"])
    transition = assess_transition(
        agg["K"], agg["var_contacts"], curves["dm_dK"],
        curves["neg_drg2_dK"], curves["dsmax_over_N_dK"])
    adequacy = _adequacy(agg, thresholds)

    report = build_report(run_id, args.N, K_ladder, args.seeds, controls,
                          agg, curves, transition, fd_check, adequacy, prefixes)

    report_json = out_dir / "direct_K_report.json"
    report_md = out_dir / "direct_K_report.md"
    curves_csv = out_dir / "direct_K_response_curves.csv"
    manifest_json = out_dir / "run_manifest.json"

    with open(report_json, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    with open(report_md, "w", encoding="utf-8") as fh:
        fh.write(render_markdown(report))
    write_response_curves_csv(curves_csv, agg, curves)
    manifest = {
        "run_id": run_id,
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "N": int(args.N),
        "K_values": [float(k) for k in K_ladder],
        "seeds": [int(s) for s in args.seeds],
        "sampler_controls": controls,
        "adequacy_thresholds": thresholds,
        "output_dir": str(out_dir),
        "seed_prefixes": [str(p) for p in prefixes],
        "reports": {
            "report_json": str(report_json),
            "report_md": str(report_md),
            "response_curves_csv": str(curves_csv),
        },
        "commands": commands,
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "git_commit": remd._git_commit(),
    }
    with open(manifest_json, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"\nSaved {report_json}")
    print(f"Saved {report_md}")
    print(f"Saved {curves_csv}")
    print(f"Saved {manifest_json}")
    print(f"\nTransition bracketed: {transition['transition_bracketed']}")
    print(f"Recommendation: {transition['recommendation']}")


if __name__ == "__main__":
    main()
