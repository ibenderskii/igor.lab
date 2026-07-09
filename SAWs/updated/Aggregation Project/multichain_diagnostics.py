#!/usr/bin/env python3
"""
Convergence and mixing diagnostics for multi-chain REMD (Stage 2 correction).

This is a thin ADAPTER around the validated single-chain diagnostic primitives
in ``remd_uniform_chain_2_new.py``.  The numerical algorithms (Geyer initial-
positive-sequence integrated autocorrelation time / ESS, early-vs-late drift,
block-mean stability, per-walker round-trip analysis) are REUSED verbatim; this
module only:

* selects the multi-chain per-lane trajectories to diagnose
  (``u``, ``m_intra``, ``m_inter``, ``mean_chain_rg``, ``n_clusters``,
  ``largest_cluster_size``, ``largest_cluster_fraction``);
* adds multi-chain-specific warnings, including a whole-chain translation
  acceptance floor (``min_translation_acceptance_rate``);
* writes the diagnostic output files.

The scientifically central diagnostics are ``m_intra`` (single-chain collapse),
``m_inter`` (microscopic interchain association) and ``largest_cluster_fraction``
(collective aggregation).
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

import remd_uniform_chain_2_new as remd
from multichain_config_io import _json_safe

# Reuse the reference thresholds and add whole-chain move acceptance floors.
DEFAULT_MC_DIAG_THRESHOLDS = dict(remd.DEFAULT_DIAG_THRESHOLDS)
DEFAULT_MC_DIAG_THRESHOLDS["min_translation_acceptance_rate"] = 0.01
DEFAULT_MC_DIAG_THRESHOLDS["min_reptation_acceptance_rate"] = 0.01
DEFAULT_MC_DIAG_THRESHOLDS["min_rotation_acceptance_rate"] = 0.01
DEFAULT_MC_DIAG_N_BLOCKS = remd.DEFAULT_DIAG_N_BLOCKS

# Per-lane per-cycle observables diagnosed (all cycle-spacing 1).  mean_chain_rg
# is a length and is rescaled by rg_scale; every other observable is a raw count
# or fraction and is never rescaled.
DIAGNOSED_OBSERVABLES = (
    "u", "m_intra", "m_inter", "mean_chain_rg", "n_clusters",
    "largest_cluster_size", "largest_cluster_fraction",
    # Intensive per-chain normalizations (linear transforms of the raw counts;
    # included for cross-M comparison of ESS/drift/means).
    "m_intra_per_chain", "m_inter_pairs_per_chain",
)
# Observables whose ESS / drift trigger convergence WARNINGS.  Kept to the raw
# extensive counts so normalized series (deterministic transforms at fixed M) do
# not emit duplicate warnings.
CENTRAL_OBSERVABLES = ("m_intra", "m_inter", "largest_cluster_fraction")


def _post_burnin_slice(n: int, burnin_frac: float) -> int:
    return int(math.floor(int(n) * float(burnin_frac)))


def _lane_series(rep, rg_scale: float) -> Dict[str, np.ndarray]:
    """Full (pre-slice) per-cycle series for one replica in diagnosed units.

    Contacts are never rescaled by ``rg_scale``; the per-chain contact series are
    exact ``/M`` transforms of the raw counts (M = number of chains).
    """
    m_intra = np.asarray(rep.m_intra_traj, dtype=float)
    m_inter = np.asarray(rep.m_inter_traj, dtype=float)
    M = float(rep.state.n_chains) if rep.state.n_chains else float("nan")
    return {
        "u": np.asarray(rep.u_traj, dtype=float),
        "m_intra": m_intra,
        "m_inter": m_inter,
        "mean_chain_rg": np.asarray(rep.rg_traj, dtype=float) * float(rg_scale),
        "n_clusters": np.asarray(rep.n_clusters_traj, dtype=float),
        "largest_cluster_size": np.asarray(rep.lcs_traj, dtype=float),
        "largest_cluster_fraction": np.asarray(rep.lcf_traj, dtype=float),
        "m_intra_per_chain": m_intra / M,
        "m_inter_pairs_per_chain": m_inter / M,
    }


def compute_multichain_diagnostics(
    replicas: list, swap_props: np.ndarray, swap_accs: np.ndarray,
    walker_temp_index: np.ndarray, Ts: np.ndarray, *,
    burnin_frac: float, n_blocks: int = DEFAULT_MC_DIAG_N_BLOCKS,
    thresholds: dict | None = None, rg_scale: float = 1.0,
) -> dict:
    """Assemble per-lane convergence + per-walker mixing diagnostics and warnings.

    Lane convergence uses the post-burn-in segment of each lane; walker mixing
    uses the full temperature-index history (burn-in is part of mixing).
    """
    nT = len(replicas)
    Ts = np.asarray(Ts, dtype=float)
    thr = dict(DEFAULT_MC_DIAG_THRESHOLDS)
    thr.update(thresholds or {})

    # --- per-lane convergence -------------------------------------------------
    lane_conv: List[dict] = []
    for i, rep in enumerate(replicas):
        series = _lane_series(rep, rg_scale)
        n = len(series["u"])
        s = _post_burnin_slice(n, burnin_frac)
        entry = {
            "temperature_index": int(i),
            "temperature": float(Ts[i]) if i < Ts.size else float("nan"),
            "n_post_burnin": int(max(n - s, 0)),
            "local_acceptance_rate": float(rep.local_acceptance_rate()),
            "translation_acceptance_rate": float(rep.translation_acceptance_rate()),
            "reptation_acceptance_rate": float(rep.reptation_acceptance_rate()),
            "rotation_acceptance_rate": float(rep.rotation_acceptance_rate()),
        }
        for name in DIAGNOSED_OBSERVABLES:
            arr = series[name][s:]
            ac = remd.integrated_autocorr_time(arr)
            dr = remd.early_late_drift(arr)
            bl = remd.block_mean_stability(arr, n_blocks)
            entry[name] = {
                "mean": float(np.nanmean(arr)) if arr.size else float("nan"),
                "std": float(np.nanstd(arr, ddof=0)) if arr.size else float("nan"),
                "tau_int": ac["tau_int"],
                "ess": ac["ess"],
                "n_samples": ac["n_samples"],
                "acf_method": ac["method"],
                "drift": dr["drift"],
                "drift_in_std": dr["drift_in_std"],
                "early_mean": dr["early_mean"],
                "late_mean": dr["late_mean"],
                "block_mean_std": bl["block_mean_std"],
                "block_mean_range_over_std": bl["block_mean_range_over_std"],
                "block_means": bl["block_means"],
            }
        lane_conv.append(entry)

    # --- per-walker mixing ----------------------------------------------------
    wti = np.asarray(walker_temp_index)
    n_walkers = wti.shape[1] if wti.ndim == 2 else 0
    walker_diag = []
    for w in range(n_walkers):
        wd = remd.analyze_walker_trajectory(wti[:, w], nT)
        wd["walker"] = int(w)
        walker_diag.append(wd)

    swap_rates = [
        (int(swap_accs[k]) / int(swap_props[k])) if int(swap_props[k]) > 0
        else float("nan") for k in range(len(swap_props))]
    swap_arr = np.array(swap_rates, dtype=float)
    finite_swap = swap_arr[np.isfinite(swap_arr)]

    def _obs_ess(name):
        return [lane[name]["ess"] for lane in lane_conv
                if np.isfinite(lane[name]["ess"])]

    def _obs_drift(name):
        return [abs(lane[name]["drift_in_std"]) for lane in lane_conv
                if np.isfinite(lane[name]["drift_in_std"])]

    coverage = [w["fraction_visited"] for w in walker_diag]
    rt_low = [w["n_round_trips_low"] for w in walker_diag]
    total_round_trips = int(sum(rt_low))
    local_rates = [lane["local_acceptance_rate"] for lane in lane_conv]
    trans_rates = [lane["translation_acceptance_rate"] for lane in lane_conv]
    rept_rates = [lane["reptation_acceptance_rate"] for lane in lane_conv]
    rot_rates = [lane["rotation_acceptance_rate"] for lane in lane_conv]

    def _finite_min(vals):
        v = [x for x in vals if x is not None and np.isfinite(x)]
        return float(min(v)) if v else float("nan")

    summary = {
        "n_temperatures": int(nT),
        "n_walkers": int(n_walkers),
        "total_round_trips_low": total_round_trips,
        "min_round_trips_per_walker": int(min(rt_low)) if rt_low else 0,
        "median_round_trips_per_walker": float(np.median(rt_low)) if rt_low else 0.0,
        "min_temp_coverage": float(min(coverage)) if coverage else float("nan"),
        "median_temp_coverage": float(np.median(coverage)) if coverage else float("nan"),
        "min_swap_rate": float(finite_swap.min()) if finite_swap.size else float("nan"),
        "median_swap_rate": float(np.median(finite_swap)) if finite_swap.size else float("nan"),
        "min_local_acceptance_rate": _finite_min(local_rates),
        "min_translation_acceptance_rate": _finite_min(trans_rates),
        "min_reptation_acceptance_rate": _finite_min(rept_rates),
        "min_rotation_acceptance_rate": _finite_min(rot_rates),
    }
    for name in DIAGNOSED_OBSERVABLES:
        ess = _obs_ess(name)
        drift = _obs_drift(name)
        summary[f"min_ess_{name}"] = float(min(ess)) if ess else float("nan")
        summary[f"median_ess_{name}"] = float(np.median(ess)) if ess else float("nan")
        summary[f"max_drift_{name}"] = float(max(drift)) if drift else float("nan")

    # --- structured warnings --------------------------------------------------
    warnings: List[dict] = []

    def _warn(kind, message, value, threshold, **extra):
        w = {"type": kind, "message": message,
             "value": (None if value is None or (isinstance(value, float)
                       and not math.isfinite(value)) else value),
             "threshold": threshold}
        w.update(extra)
        warnings.append(w)

    if total_round_trips < int(thr["min_round_trips"]):
        _warn("round_trips",
              f"total low->high->low round trips {total_round_trips} < "
              f"{thr['min_round_trips']}",
              total_round_trips, thr["min_round_trips"])
    if coverage and min(coverage) < float(thr["min_temp_coverage"]):
        _warn("temperature_coverage",
              f"minimum walker temperature coverage {min(coverage):.3f} < "
              f"{thr['min_temp_coverage']}",
              float(min(coverage)), thr["min_temp_coverage"])
    if finite_swap.size and float(finite_swap.min()) < float(thr["min_swap_rate"]):
        _warn("swap_rate",
              f"minimum adjacent swap rate {float(finite_swap.min()):.3f} < "
              f"{thr['min_swap_rate']}",
              float(finite_swap.min()), thr["min_swap_rate"])
    for name in CENTRAL_OBSERVABLES:
        ess = _obs_ess(name)
        if ess and min(ess) < float(thr["min_ess"]):
            _warn(f"low_ess_{name}",
                  f"minimum {name} ESS {min(ess):.1f} < {thr['min_ess']}",
                  float(min(ess)), thr["min_ess"])
        drift = _obs_drift(name)
        if drift and max(drift) > float(thr["max_drift"]):
            _warn(f"drift_{name}",
                  f"maximum |{name} drift|/std {max(drift):.3f} > {thr['max_drift']}",
                  float(max(drift)), thr["max_drift"])
    # Near-frozen local moves (state-changing acceptance) and translations.
    for i, lane in enumerate(lane_conv):
        lr = lane["local_acceptance_rate"]
        if np.isfinite(lr) and lr < float(thr["min_state_changing_move_rate"]):
            _warn("local_move_freezing",
                  f"lane T={lane['temperature']:.3g} state-changing local move "
                  f"acceptance {lr:.4f} < {thr['min_state_changing_move_rate']}",
                  float(lr), thr["min_state_changing_move_rate"],
                  temperature_index=i, temperature=lane["temperature"])
        tr = lane["translation_acceptance_rate"]
        if np.isfinite(tr) and tr < float(thr["min_translation_acceptance_rate"]):
            _warn("translation_freezing",
                  f"lane T={lane['temperature']:.3g} whole-chain translation "
                  f"acceptance {tr:.4f} < {thr['min_translation_acceptance_rate']}",
                  float(tr), thr["min_translation_acceptance_rate"],
                  temperature_index=i, temperature=lane["temperature"])
        rp = lane["reptation_acceptance_rate"]
        if np.isfinite(rp) and rp < float(thr["min_reptation_acceptance_rate"]):
            _warn("reptation_freezing",
                  f"lane T={lane['temperature']:.3g} whole-chain reptation "
                  f"acceptance {rp:.4f} < {thr['min_reptation_acceptance_rate']}",
                  float(rp), thr["min_reptation_acceptance_rate"],
                  temperature_index=i, temperature=lane["temperature"])
        ro = lane["rotation_acceptance_rate"]
        if np.isfinite(ro) and ro < float(thr["min_rotation_acceptance_rate"]):
            _warn("rotation_freezing",
                  f"lane T={lane['temperature']:.3g} whole-chain rotation "
                  f"acceptance {ro:.4f} < {thr['min_rotation_acceptance_rate']}",
                  float(ro), thr["min_rotation_acceptance_rate"],
                  temperature_index=i, temperature=lane["temperature"])

    return {
        "thresholds": {k: thr[k] for k in DEFAULT_MC_DIAG_THRESHOLDS},
        "n_blocks": int(n_blocks),
        "burnin_frac": float(burnin_frac),
        "rg_scale": float(rg_scale),
        "diagnosed_observables": list(DIAGNOSED_OBSERVABLES),
        "central_observables": list(CENTRAL_OBSERVABLES),
        "summary": summary,
        "swap_rates": swap_rates,
        "lane_convergence": lane_conv,
        "walkers": walker_diag,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def save_diagnostics_json(diagnostics: dict, out_prefix: str) -> str:
    path = f"{out_prefix}_diagnostics.json"
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(_json_safe(diagnostics), fh, indent=2, allow_nan=False)
    print(f"Saved {path}")
    return path


CONVERGENCE_CSV_COLUMNS = [
    "temperature_index", "temperature", "observable", "n_samples", "mean", "std",
    "tau_int", "ess", "drift", "drift_in_std", "block_mean_std",
    "block_mean_range_over_std", "acf_method",
]


def save_convergence_csv(diagnostics: dict, out_prefix: str) -> str:
    path = f"{out_prefix}_convergence.csv"
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(CONVERGENCE_CSV_COLUMNS)
        for lane in diagnostics["lane_convergence"]:
            for name in DIAGNOSED_OBSERVABLES:
                d = lane[name]
                w.writerow([
                    lane["temperature_index"], lane["temperature"], name,
                    d["n_samples"], d["mean"], d["std"], d["tau_int"], d["ess"],
                    d["drift"], d["drift_in_std"], d["block_mean_std"],
                    d["block_mean_range_over_std"], d["acf_method"],
                ])
    print(f"Saved {path}")
    return path


ROUND_TRIPS_CSV_COLUMNS = [
    "walker", "fraction_visited", "n_visited", "first_passage_low_to_high",
    "first_passage_high_to_low", "n_round_trips_low", "n_round_trips_high",
    "n_round_trips", "mean_round_trip_duration",
]


def save_round_trips_csv(diagnostics: dict, out_prefix: str) -> str:
    path = f"{out_prefix}_round_trips.csv"
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(ROUND_TRIPS_CSV_COLUMNS)
        for wd in diagnostics["walkers"]:
            w.writerow([
                wd["walker"], wd["fraction_visited"], wd["n_visited"],
                wd["first_passage_low_to_high"], wd["first_passage_high_to_low"],
                wd["n_round_trips_low"], wd["n_round_trips_high"],
                wd["n_round_trips"], wd["mean_round_trip_duration"],
            ])
    print(f"Saved {path}")
    return path


def save_walker_occupancy_csv(diagnostics: dict, out_prefix: str) -> str:
    path = f"{out_prefix}_walker_occupancy.csv"
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    nT = int(diagnostics["summary"]["n_temperatures"])
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["walker", "fraction_visited"]
                   + [f"occupancy_T{i}" for i in range(nT)])
        for wd in diagnostics["walkers"]:
            occ = list(wd["occupancy"])
            occ = occ + [0] * (nT - len(occ))
            w.writerow([wd["walker"], wd["fraction_visited"]]
                       + [int(v) for v in occ[:nT]])
    print(f"Saved {path}")
    return path


def save_diagnostic_trajectories_npz(
    replicas: list, walker_temp_index: np.ndarray, Ts: np.ndarray, *,
    burnin_frac: float, out_prefix: str, rg_scale: float = 1.0,
) -> str:
    """Save compact post-burn-in per-lane trajectories (no object arrays).

    Scalar lane arrays are ``(nT, n_post)``; the per-chain array is
    ``(nT, n_post, M)``; the walker temperature-index trace is
    ``(n_post, n_walkers)``.  Lengths are uniform across lanes because every lane
    runs the same number of cycles, so the arrays are strictly rectangular.
    """
    path = f"{out_prefix}_diagnostic_trajectories.npz"
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    n_cycles = max((len(r.u_traj) for r in replicas), default=0)
    s = _post_burnin_slice(n_cycles, burnin_frac)
    rg = np.float32(rg_scale)

    def stack_scalar(attr, dtype=np.float32, scale=1.0):
        return np.array(
            [np.asarray(getattr(r, attr)[s:], dtype=dtype) * dtype(scale)
             for r in replicas], dtype=dtype)

    per_chain_lat = np.array(
        [np.asarray(r.per_chain_rg_traj[s:], dtype=np.float32) for r in replicas],
        dtype=np.float32)  # (nT, n_post, M)

    wti = np.asarray(walker_temp_index)
    wti_post = (wti[s:].astype(np.int16) if wti.ndim == 2
                else np.empty((0, 0), dtype=np.int16))

    # Raw contact arrays (extensive; also the base for the intensive forms).
    m_intra_post = stack_scalar("m_intra_traj", dtype=np.int32)
    m_inter_post = stack_scalar("m_inter_traj", dtype=np.int32)
    # std of per-chain Rg is a LENGTH: store lattice (unscaled) and scaled forms
    # under unambiguous names (previously std_chain_rg_post held lattice units).
    std_rg_lattice = stack_scalar("std_chain_rg_traj")
    std_rg_scaled = (std_rg_lattice * rg).astype(np.float32)
    # Intensive per-chain contacts, derived EXACTLY from the raw arrays and M.
    M = int(replicas[0].state.n_chains) if replicas else 0
    Mf = np.float32(M) if M > 0 else np.float32("nan")

    arrays = {
        "Ts": np.asarray(Ts, dtype=np.float64),
        "burnin_start_cycle": np.int64(s),
        "n_chains": np.int64(M),
        "u_post": stack_scalar("u_traj"),
        "m_intra_post": m_intra_post,
        "m_inter_post": m_inter_post,
        "m_intra_per_chain_post": m_intra_post.astype(np.float32) / Mf,
        "m_inter_pairs_per_chain_post": m_inter_post.astype(np.float32) / Mf,
        "m_inter_incidences_per_chain_post":
            (2.0 * m_inter_post.astype(np.float32)) / Mf,
        "m_total_pairs_per_chain_post":
            (m_intra_post + m_inter_post).astype(np.float32) / Mf,
        "mean_chain_rg_lattice_post": stack_scalar("rg_traj"),
        "mean_chain_rg_post": stack_scalar("rg_traj", scale=float(rg_scale)),
        "per_chain_rg_lattice_post": per_chain_lat,
        "per_chain_rg_post": (per_chain_lat * rg).astype(np.float32),
        "std_chain_rg_lattice_post": std_rg_lattice,
        "std_chain_rg_post": std_rg_scaled,
        "n_clusters_post": stack_scalar("n_clusters_traj", dtype=np.int32),
        "largest_cluster_size_post": stack_scalar("lcs_traj", dtype=np.int32),
        "largest_cluster_fraction_post": stack_scalar("lcf_traj"),
        "walker_temp_index_post": wti_post,
    }
    np.savez_compressed(path, **arrays)
    print(f"Saved {path}")
    return path


def save_all_diagnostics(
    diagnostics: dict, replicas: list, walker_temp_index: np.ndarray,
    Ts: np.ndarray, *, out_prefix: str, burnin_frac: float, rg_scale: float,
    save_trajectories: bool,
) -> Dict[str, str]:
    """Write the diagnostics JSON + CSVs (+ optional trajectory NPZ); return map."""
    files = {
        "diagnostics_json": save_diagnostics_json(diagnostics, out_prefix),
        "convergence_csv": save_convergence_csv(diagnostics, out_prefix),
        "round_trips_csv": save_round_trips_csv(diagnostics, out_prefix),
        "walker_occupancy_csv": save_walker_occupancy_csv(diagnostics, out_prefix),
    }
    if save_trajectories:
        files["diagnostic_trajectories_npz"] = save_diagnostic_trajectories_npz(
            replicas, walker_temp_index, Ts, burnin_frac=burnin_frac,
            out_prefix=out_prefix, rg_scale=rg_scale)
    return files
