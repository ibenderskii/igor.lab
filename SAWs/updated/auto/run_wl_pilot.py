#!/usr/bin/env python3
"""Run the gated Wang-Landau baseline acceptance sequence.

Chain lengths are always processed in the scientific validation order 30, 44,
60.  A failed gate stops the sequence.  The runner does not lower ``m_cover``
when the top of a window is not reached; it recommends the separately reviewed
pull-move work instead.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from target_support import CONTACT_OFFSETS, load_target_contact_support


AUTO_DIR = Path(__file__).resolve().parent
CHAIN_ORDER = (30, 44, 60)
CHAIN_INFO: Dict[int, Dict[str, Any]] = {
    30: {
        "config": "config30.json",
        "target": "remd_distributions_30mer.npz",
        "legacy": "RnBaseline30.npz",
        "m_max": 30,
        "throughput": 50_800.0,
        "reference_before": (0.000089, 0.000369),
        "reference_after": (0.0, 0.0),
    },
    44: {
        "config": "config2.json",
        "target": "remd_distributions_44mer.npz",
        "legacy": "RnBaseline.npz",
        "m_max": 50,
        "throughput": 38_100.0,
        "reference_before": (0.017307, 0.047380),
        "reference_after": (0.000025, 0.000730),
    },
    60: {
        "config": "config60.json",
        "target": "remd_distributions_60mer.npz",
        "legacy": "RnBaseline60.npz",
        "m_max": 74,
        "throughput": 29_900.0,
        "reference_before": (0.057199, 0.159548),
        "reference_after": (0.000147, 0.001229),
    },
}


class AcceptanceFailure(RuntimeError):
    pass


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _command_text(command: Sequence[str]) -> str:
    return shlex.join(str(item) for item in command)


def _run_command(command: Sequence[str], log_path: Path) -> float:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    environment = os.environ.copy()
    environment["MPLCONFIGDIR"] = str(log_path.parent / "matplotlib")
    Path(environment["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"$ {_command_text(command)}\n\n")
        log.flush()
        completed = subprocess.run(
            list(command),
            cwd=AUTO_DIR,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    elapsed = time.time() - started
    if completed.returncode:
        raise AcceptanceFailure(
            f"command failed with exit code {completed.returncode}; see {log_path}"
        )
    return elapsed


def _wl_command(args: argparse.Namespace, n_beads: int, output_dir: Path) -> List[str]:
    return [
        args.python,
        str(AUTO_DIR / "single_chain_wang_landau.py"),
        "--N", str(n_beads),
        "--dist_dir", str(output_dir),
        "--n_workers", str(args.n_workers),
        "--steps_per_worker", str(args.steps_per_worker),
        "--base_seed", str(args.base_seed),
        "--wl_seed", str(args.wl_seed),
        "--burnin", str(args.burnin),
        "--sample_every", str(args.sample_every),
        "--wl_final_log_f", str(args.wl_final_log_f),
        "--wl_flatness", str(args.wl_flatness),
        "--wl_min_visits", str(args.wl_min_visits),
        "--wl_min_cover_visits", str(args.wl_min_cover_visits),
        "--wl_check_every", str(args.wl_check_every),
        "--wl_max_steps", str(args.wl_max_steps),
        "--wl_max_seconds", str(args.wl_max_seconds),
        "--wl_max_seconds_scope", args.wl_max_seconds_scope,
        "--wl_max_steps_per_stage", str(args.wl_max_steps_per_stage),
        "--wl_schedule", args.wl_schedule,
        "--wl_stage_stall_steps", str(args.wl_stage_stall_steps),
        "--checkpoint_every_seconds", str(args.checkpoint_every_seconds),
        "--checkpoint", str(output_dir / "learning_checkpoint.npz"),
        "--n_blocks", str(args.n_blocks),
    ]


def _wl_output_path(args: argparse.Namespace, n_beads: int, output_dir: Path) -> Path:
    return output_dir / (
        f"single_chain_wang_landau_N{n_beads}_workers{args.n_workers}"
        f"_steps{args.steps_per_worker}_seed{args.base_seed}.npz"
    )


def _validate_wl_output(path: Path) -> Dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        required = {
            "wl_tier", "production_samples_per_level", "wl_min_visits",
            "wl_min_cover_visits", "production_round_trips_per_worker",
            "rg_out_of_range_count", "joint_contact_marginal_error",
            "joint_rg_marginal_error", "m_cover", "wl_stage_records",
            "production_max_contact", "importance_effective_sample_size",
            "wl_learning_steps", "wl_learning_wall_seconds",
        }
        missing = required - set(data.files)
        if missing:
            raise AcceptanceFailure(
                f"{path} is missing acceptance fields: {sorted(missing)}"
            )
        tier = np.asarray(data["wl_tier"], dtype=np.int8)
        counts = np.asarray(data["production_samples_per_level"], dtype=np.int64)
        flat_min = int(data["wl_min_visits"])
        cover_min = int(data["wl_min_cover_visits"])
        deficient_flat = np.flatnonzero((tier == 2) & (counts < flat_min))
        deficient_cover = np.flatnonzero((tier == 1) & (counts < cover_min))
        round_trips_per_worker = np.asarray(
            data["production_round_trips_per_worker"], dtype=np.int64
        )
        round_trips = int(round_trips_per_worker.sum())
        rg_drops = int(data["rg_out_of_range_count"])
        joint_m = float(data["joint_contact_marginal_error"])
        joint_rg = float(data["joint_rg_marginal_error"])
        m_cover = int(data["m_cover"])
        stage_records = np.asarray(data["wl_stage_records"])
        stages = [
            {name: _jsonable(record[name]) for name in stage_records.dtype.names or ()}
            for record in stage_records
        ]
        highest_learning = (
            int(stage_records["highest_m"].max()) if stage_records.size else -1
        )
        gates = {
            "tier_coverage": not deficient_flat.size and not deficient_cover.size,
            "production_round_trip": round_trips >= 1,
            "rg_no_drops": rg_drops == 0,
            "joint_marginals": joint_m < 1e-12 and joint_rg < 1e-12,
            "learning_reached_top": highest_learning >= m_cover,
        }
        summary = {
            "path": path,
            "gates": gates,
            "deficient_tier2": deficient_flat,
            "deficient_tier1": deficient_cover,
            "production_round_trips": round_trips,
            "production_round_trips_per_worker": round_trips_per_worker,
            "rg_out_of_range_count": rg_drops,
            "joint_contact_marginal_error": joint_m,
            "joint_rg_marginal_error": joint_rg,
            "m_cover": m_cover,
            "highest_learning_contact": highest_learning,
            "highest_production_contact": int(data["production_max_contact"]),
            "learning_steps": int(data["wl_learning_steps"]),
            "learning_wall_seconds": float(data["wl_learning_wall_seconds"]),
            "learning_steps_per_second": (
                float(data["wl_learning_steps"])
                / float(data["wl_learning_wall_seconds"])
                if float(data["wl_learning_wall_seconds"]) > 0.0
                else None
            ),
            "stages": stages,
            "importance_ess": float(data["importance_effective_sample_size"]),
            "production_counts": counts,
        }
    if not all(gates.values()):
        if not gates["learning_reached_top"]:
            raise AcceptanceFailure(
                f"learning reached m={highest_learning}, below m_cover={m_cover}; "
                "do not lower m_cover silently. This is the evidence gate for W8 pull moves."
            )
        raise AcceptanceFailure(f"Wang-Landau acceptance gates failed: {gates}")
    return summary


def _direct_batch_stderr(
    samples: np.ndarray,
    c_vals: np.ndarray,
    per_worker: np.ndarray,
    n_blocks: int,
) -> np.ndarray:
    estimates: List[np.ndarray] = []
    offset = 0
    for count in per_worker:
        worker = samples[offset:offset + int(count)]
        offset += int(count)
        for block in np.array_split(worker, min(n_blocks, worker.size)):
            estimates.append(
                np.array([(block == level).mean() for level in c_vals], dtype=float)
            )
    batches = np.asarray(estimates)
    return batches.std(axis=0, ddof=1) / math.sqrt(batches.shape[0])


def _cross_validate(
    legacy_path: Path,
    wl_path: Path,
    outdir: Path,
    min_samples: int,
    n_blocks: int,
) -> Dict[str, Any]:
    with np.load(legacy_path, allow_pickle=False) as legacy, np.load(
        wl_path, allow_pickle=False
    ) as wl:
        if "c_samples" not in legacy.files:
            raise AcceptanceFailure(
                f"legacy baseline {legacy_path} lacks direct c_samples for batch errors"
            )
        legacy_vals = np.asarray(legacy["c_vals"], dtype=np.int64)
        legacy_prob = np.asarray(legacy["c_prob"], dtype=float)
        legacy_samples = np.asarray(legacy["c_samples"], dtype=np.int64)
        legacy_counts = np.array(
            [(legacy_samples == level).sum() for level in legacy_vals],
            dtype=np.int64,
        )
        legacy_se = _direct_batch_stderr(
            legacy_samples,
            legacy_vals,
            np.asarray(legacy["samples_per_worker"], dtype=np.int64),
            n_blocks,
        )
        wl_vals = np.asarray(wl["c_vals"], dtype=np.int64)
        wl_prob = np.asarray(wl["c_prob"], dtype=float)
        wl_counts = np.asarray(wl["production_c_counts"], dtype=np.int64)
        wl_se = np.asarray(wl["c_blocked_stderr"], dtype=float)

    legacy_index = {int(level): i for i, level in enumerate(legacy_vals)}
    wl_index = {int(level): i for i, level in enumerate(wl_vals)}
    rows: List[Dict[str, Any]] = []
    failed_levels: List[int] = []
    for level in sorted(set(legacy_index) & set(wl_index)):
        old_i, new_i = legacy_index[level], wl_index[level]
        if legacy_counts[old_i] < min_samples or wl_counts[new_i] < min_samples:
            continue
        old_p, new_p = float(legacy_prob[old_i]), float(wl_prob[new_i])
        if old_p <= 0.0 or new_p <= 0.0:
            continue
        ratio = new_p / old_p
        ratio_se = ratio * math.sqrt(
            (float(wl_se[new_i]) / new_p) ** 2
            + (float(legacy_se[old_i]) / old_p) ** 2
        )
        z = abs(ratio - 1.0) / ratio_se if ratio_se > 0.0 else float("inf")
        passed = z <= 3.0
        if not passed:
            failed_levels.append(level)
        rows.append(
            {
                "m": level,
                "p_wl": new_p,
                "p_direct": old_p,
                "ratio": ratio,
                "ratio_stderr": ratio_se,
                "z_from_one": z,
                "wl_samples": int(wl_counts[new_i]),
                "direct_samples": int(legacy_counts[old_i]),
                "passed_3sigma": passed,
            }
        )
    if not rows:
        raise AcceptanceFailure("no contact levels had adequate cross-validation statistics")

    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / "bulk_cross_validation.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        x = np.array([row["m"] for row in rows])
        y = np.array([row["ratio"] for row in rows])
        e = np.array([row["ratio_stderr"] for row in rows])
        figure, axis = plt.subplots(figsize=(7, 4))
        axis.errorbar(x, y, yerr=e, fmt="o", capsize=2)
        axis.axhline(1.0, color="black", linewidth=1)
        axis.set(xlabel="contact level m", ylabel="P0_WL / P0_direct")
        figure.tight_layout()
        figure.savefig(outdir / "bulk_cross_validation.png", dpi=180)
        plt.close(figure)
    except ImportError:
        pass
    summary = {
        "adequate_levels": len(rows),
        "failed_levels": failed_levels,
        "passed": not failed_levels,
        "table": csv_path,
    }
    if failed_levels:
        raise AcceptanceFailure(
            "bulk cross-validation exceeded three combined standard errors at "
            f"contact levels {failed_levels}"
        )
    return summary


def _upper_tail_metrics(target_path: Path, offset: int, ceiling: int) -> Dict[str, float]:
    support = load_target_contact_support(target_path, offset)
    per_temperature = support.P[:, support.m_axis > ceiling + 0.5].sum(axis=1)
    return {
        "mean": float(per_temperature.mean()),
        "max": float(per_temperature.max()),
        "argmax_T": float(support.temps[int(np.argmax(per_temperature))]),
    }


def _support_command(
    args: argparse.Namespace,
    target: Path,
    baseline: Path,
    offset: int,
    rg_scale: float,
    outdir: Path,
) -> List[str]:
    return [
        args.python,
        str(AUTO_DIR / "analyze_support_mismatch.py"),
        "--target", str(target),
        "--baseline", str(baseline),
        "--contact-offsets", str(offset),
        "--rg-scale", str(rg_scale),
        "--outdir", str(outdir),
        "--no-plots",
    ]


def _fit_command(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    n_beads: int,
    model: str,
    target: Path,
    baseline: Path,
    outdir: Path,
) -> List[str]:
    fit = config["fit"]
    baseline_config = config["baselines"][0]
    command = [
        args.python,
        str(AUTO_DIR / config["fit_script"]),
        "--remd", str(target),
        "--baseline", str(baseline),
        "--contact_offset", str(baseline_config["contact_offset"]),
        "--N", str(n_beads),
        "--model", model,
        "--loss", str(fit["loss"]),
        "--rg-weight", str(fit["rg_weight"]),
        "--rg-scale", str(baseline_config["rg_scale"]),
        "--n_restarts", str(fit["n_restarts"]),
        "--seed", str(fit["seed"]),
        "--outdir", str(outdir),
    ]
    command.append("--fit-rg" if fit.get("fit_rg", False) else "--no-fit-rg")
    if not fit.get("plots", True):
        command.append("--no-plots")
    for key, flag in (
        ("holdout_every", "--holdout-every"),
        ("holdout_indices", "--holdout-indices"),
        ("train_indices", "--train-indices"),
    ):
        value = fit.get(key)
        if value is not None:
            if isinstance(value, list):
                value = ",".join(map(str, value))
            command.extend((flag, str(value)))
    bootstrap = fit.get("bootstrap", {})
    if bootstrap.get("enabled", False) and not args.skip_bootstrap:
        command.extend(("--bootstrap", str(bootstrap["replicates"])))
        command.extend(("--bootstrap-seed", str(bootstrap["seed"])))
        command.extend(("--bootstrap-method", str(bootstrap["method"])))
        command.extend(("--bootstrap-confidence", str(bootstrap["confidence"])))
        command.extend((
            "--bootstrap-correlation-threshold",
            str(bootstrap["correlation_threshold"]),
        ))
        if bootstrap.get("save_prediction_bands", False):
            command.append("--bootstrap-save-prediction-bands")
    if fit.get("uncertainty_diagnostics", {}).get("enabled", False):
        command.append("--uncertainty-diagnostics")
    return command


def _centers_to_edges(centers: np.ndarray) -> np.ndarray:
    centers = np.asarray(centers, dtype=float)
    midpoints = 0.5 * (centers[:-1] + centers[1:])
    return np.concatenate((
        [centers[0] - (midpoints[0] - centers[0])],
        midpoints,
        [centers[-1] + (centers[-1] - midpoints[-1])],
    ))


def _rebin_mass(old_edges: np.ndarray, mass: np.ndarray, new_edges: np.ndarray) -> np.ndarray:
    result = np.zeros(new_edges.size - 1, dtype=float)
    for i, value in enumerate(np.asarray(mass, dtype=float)):
        left, right = old_edges[i], old_edges[i + 1]
        if value == 0.0 or right <= left:
            continue
        first = max(0, int(np.searchsorted(new_edges, left, side="right") - 1))
        last = min(result.size - 1, int(np.searchsorted(new_edges, right, side="left")))
        for j in range(first, last + 1):
            overlap = max(0.0, min(right, new_edges[j + 1]) - max(left, new_edges[j]))
            if overlap:
                result[j] += value * overlap / (right - left)
    total = result.sum()
    if total > 0.0:
        result /= total
    return result


def _js(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    first /= first.sum()
    second /= second.sum()
    middle = 0.5 * (first + second)
    keep_first = first > 0.0
    keep_second = second > 0.0
    return 0.5 * (
        float(np.sum(first[keep_first] * np.log(first[keep_first] / middle[keep_first])))
        + float(np.sum(second[keep_second] * np.log(second[keep_second] / middle[keep_second])))
    )


def _align_contact(values: np.ndarray, rows: np.ndarray, grid: np.ndarray) -> np.ndarray:
    aligned = np.zeros((rows.shape[0], grid.size), dtype=float)
    lookup = {int(value): i for i, value in enumerate(grid)}
    for source, value in enumerate(values):
        aligned[:, lookup[int(round(float(value)))]] += rows[:, source]
    return aligned


def _compare_fit_pair(
    legacy_dir: Path, new_dir: Path, target_path: Path
) -> Dict[str, Any]:
    legacy_summary = json.loads((legacy_dir / "fit_summary.json").read_text())
    new_summary = json.loads((new_dir / "fit_summary.json").read_text())
    with np.load(legacy_dir / "fit_results.npz", allow_pickle=False) as old, np.load(
        new_dir / "fit_results.npz", allow_pickle=False
    ) as new:
        temps = np.asarray(old["temps"], dtype=float)
        if not np.array_equal(temps, np.asarray(new["temps"], dtype=float)):
            raise AcceptanceFailure("legacy/new fit temperatures differ")
        contact_grid = np.union1d(old["m_centers"], new["m_centers"]).astype(int)
        old_obs = _align_contact(old["m_centers"], old["p_obs_mass"], contact_grid)
        old_mod = _align_contact(old["m_centers"], old["p_mod_mass"], contact_grid)
        new_obs = _align_contact(new["m_centers"], new["p_obs_mass"], contact_grid)
        new_mod = _align_contact(new["m_centers"], new["p_mod_mass"], contact_grid)
        if not np.allclose(old_obs, new_obs, rtol=0.0, atol=2e-12):
            raise AcceptanceFailure(
                "legacy/new fits did not use the same observed contact distribution"
            )
        target_contact = old_obs
        contact_js_old = np.array([_js(a, b) for a, b in zip(target_contact, old_mod)])
        contact_js_new = np.array([_js(a, b) for a, b in zip(target_contact, new_mod)])
        contact_tvd_old = 0.5 * np.abs(target_contact - old_mod).sum(axis=1)
        contact_tvd_new = 0.5 * np.abs(target_contact - new_mod).sum(axis=1)

        rg_metrics: Dict[str, Any] = {}
        if all(key in old.files and key in new.files for key in ("rg_mod_mass", "rg_edges_scaled")):
            with np.load(target_path, allow_pickle=False) as target:
                target_edges = _centers_to_edges(np.asarray(target["rg_centers"], dtype=float))
                target_hist = np.asarray(target["rg_hists"], dtype=float)
            common_edges = np.unique(np.concatenate((
                target_edges,
                np.asarray(old["rg_edges_scaled"], dtype=float),
                np.asarray(new["rg_edges_scaled"], dtype=float),
            )))
            widths = np.diff(target_edges)
            target_mass = target_hist * widths[None, :]
            target_mass /= target_mass.sum(axis=1, keepdims=True)
            target_common = np.asarray([
                _rebin_mass(target_edges, row, common_edges) for row in target_mass
            ])
            old_common = np.asarray([
                _rebin_mass(np.asarray(old["rg_edges_scaled"]), row, common_edges)
                for row in old["rg_mod_mass"]
            ])
            new_common = np.asarray([
                _rebin_mass(np.asarray(new["rg_edges_scaled"]), row, common_edges)
                for row in new["rg_mod_mass"]
            ])
            rg_js_old = np.array([_js(a, b) for a, b in zip(target_common, old_common)])
            rg_js_new = np.array([_js(a, b) for a, b in zip(target_common, new_common)])
            rg_tvd_old = 0.5 * np.abs(target_common - old_common).sum(axis=1)
            rg_tvd_new = 0.5 * np.abs(target_common - new_common).sum(axis=1)
            rg_metrics = {
                "mean_js_legacy": float(rg_js_old.mean()),
                "mean_js_new": float(rg_js_new.mean()),
                "mean_tvd_legacy": float(rg_tvd_old.mean()),
                "mean_tvd_new": float(rg_tvd_new.mean()),
                "js_by_temperature_legacy": rg_js_old,
                "js_by_temperature_new": rg_js_new,
                "tvd_by_temperature_legacy": rg_tvd_old,
                "tvd_by_temperature_new": rg_tvd_new,
            }

        old_b = np.asarray(old["linear_coefficient_by_temperature"], dtype=float)
        new_b = np.asarray(new["linear_coefficient_by_temperature"], dtype=float)

    old_params = legacy_summary["params"]
    new_params = new_summary["params"]
    return {
        "model": legacy_summary["model"],
        "parameters_legacy": old_params,
        "parameters_new": new_params,
        "parameter_changes": {
            key: float(new_params[key] - old_params[key]) for key in old_params
        },
        "temperatures": temps,
        "b_T_legacy": old_b,
        "b_T_new": new_b,
        "b_T_change": new_b - old_b,
        "b_T_rms_change": float(np.sqrt(np.mean(np.square(new_b - old_b)))),
        "b_T_max_abs_change": float(np.max(np.abs(new_b - old_b))),
        "contact_mean_js_legacy": float(contact_js_old.mean()),
        "contact_mean_js_new": float(contact_js_new.mean()),
        "contact_mean_tvd_legacy": float(contact_tvd_old.mean()),
        "contact_mean_tvd_new": float(contact_tvd_new.mean()),
        "contact_js_by_temperature_legacy": contact_js_old,
        "contact_js_by_temperature_new": contact_js_new,
        "contact_tvd_by_temperature_legacy": contact_tvd_old,
        "contact_tvd_by_temperature_new": contact_tvd_new,
        "rg_common_grid": rg_metrics,
    }


def _write_report(summary: Mapping[str, Any], path: Path) -> None:
    lines = ["# Wang-Landau baseline acceptance report", ""]
    if summary.get("failure"):
        lines.extend((f"**Acceptance failed:** {summary['failure']}", ""))
    for n_text, result in summary.get("chains", {}).items():
        lines.extend((f"## N={n_text}", ""))
        wl = result.get("wl", {})
        lines.append(
            f"- WL learning: {wl.get('learning_steps', 'n/a')} steps, "
            f"{wl.get('learning_wall_seconds', 'n/a')} s, "
            f"{wl.get('learning_steps_per_second', 'n/a')} steps/s"
        )
        lines.append(
            f"- Highest contact: learning {wl.get('highest_learning_contact', 'n/a')}, "
            f"production {wl.get('highest_production_contact', 'n/a')}, "
            f"declared cover {wl.get('m_cover', 'n/a')}"
        )
        cross = result.get("cross_validation", {})
        lines.append(
            f"- Bulk cross-validation: {cross.get('adequate_levels', 0)} levels, "
            f"failed levels {cross.get('failed_levels', [])}"
        )
        support = result.get("upper_tail_support", {})
        if support:
            lines.append(
                "- Unsupported upper-tail target mass, mean / worst T: "
                f"{100 * support['before']['mean']:.6f}% / "
                f"{100 * support['before']['max']:.6f}% before; "
                f"{100 * support['after']['mean']:.6f}% / "
                f"{100 * support['after']['max']:.6f}% after"
            )
        fits = result.get("fits", [])
        if fits:
            lines.extend(("", "| Model | mean contact JS old | mean contact JS new | max |Δb(T)| |", "|---|---:|---:|---:|"))
            for fit in fits:
                lines.append(
                    f"| {fit['model']} | {fit['contact_mean_js_legacy']:.6g} | "
                    f"{fit['contact_mean_js_new']:.6g} | {fit['b_T_max_abs_change']:.6g} |"
                )
            for fit in fits:
                lines.extend(("", f"### {fit['model']} by temperature", ""))
                lines.extend((
                    "| T | b old | b new | contact JS old | contact JS new |",
                    "|---:|---:|---:|---:|---:|",
                ))
                for i, temperature in enumerate(fit["temperatures"]):
                    lines.append(
                        f"| {temperature:.6g} | {fit['b_T_legacy'][i]:.6g} | "
                        f"{fit['b_T_new'][i]:.6g} | "
                        f"{fit['contact_js_by_temperature_legacy'][i]:.6g} | "
                        f"{fit['contact_js_by_temperature_new'][i]:.6g} |"
                    )
        lines.append("")
    lines.extend((
        "Residual upper-tail mass for N=44 and N=60 is geometrically irreducible: "
        "the shifted molecular coordinate extends beyond the exact lattice maxima.",
        "",
    ))
    path.write_text("\n".join(lines), encoding="utf-8")


def _dry_run(args: argparse.Namespace) -> int:
    total_seconds = 0.0
    conflicts: List[int] = []
    for n_beads in args.chains:
        info = CHAIN_INFO[n_beads]
        chain_dir = args.outdir / f"N{n_beads}"
        wl_dir = chain_dir / "wl"
        wl_command = _wl_command(args, n_beads, wl_dir)
        print(f"N={n_beads}")
        print(f"  {_command_text(wl_command)}")
        throughput = info["throughput"]
        # The two WL caps are reported separately.  Summing learning and
        # production into one number labelled a "WL wall cap" hid the fact that
        # the step cap is unreachable within the time cap.
        step_cap_seconds = args.wl_max_steps / throughput
        time_implied_steps = args.wl_max_seconds * throughput
        binding = (
            "--wl_max_steps" if args.wl_max_steps <= time_implied_steps
            else "--wl_max_seconds"
        )
        learning = min(args.wl_max_seconds, step_cap_seconds)
        production = args.steps_per_worker / throughput
        total_seconds += learning + production
        print(
            f"  throughput: {throughput / 1000.0:.1f}k attempted moves/s/core"
        )
        print(
            f"  WL step cap: --wl_max_steps={args.wl_max_steps:.3g} "
            f"= {step_cap_seconds / 3600.0:.2f} h"
        )
        print(
            f"  WL time cap: --wl_max_seconds={args.wl_max_seconds:.6g} "
            f"= {args.wl_max_seconds / 3600.0:.2f} h "
            f"= {time_implied_steps:.3g} steps"
        )
        print(f"  binding WL cap: {binding} ({learning / 3600.0:.2f} h)")
        if args.wl_max_steps > 2.0 * time_implied_steps:
            conflicts.append(n_beads)
            print(
                f"  CONFLICT: --wl_max_steps is {args.wl_max_steps / time_implied_steps:.1f}x "
                "beyond what --wl_max_seconds permits, so it can never bind. "
                "--wl_max_seconds is the operative limit."
            )
        print(f"  estimated production wall: {production / 3600.0:.2f} h")
        config = json.loads((AUTO_DIR / info["config"]).read_text())
        target = AUTO_DIR / info["target"]
        legacy = AUTO_DIR / info["legacy"]
        wl_output = _wl_output_path(args, n_beads, wl_dir)
        for label, baseline in (("legacy", legacy), ("new", wl_output)):
            support = _support_command(
                args, target, baseline, CONTACT_OFFSETS[n_beads],
                float(config["baselines"][0]["rg_scale"]),
                chain_dir / "support" / label,
            )
            print(f"  {_command_text(support)}")
        models = args.models or config["models"]
        for model in models:
            for label, baseline in (("legacy", legacy), ("new", wl_output)):
                command = _fit_command(
                    args, config, n_beads, model, target, baseline,
                    chain_dir / "fits" / model / label,
                )
                print(f"  {_command_text(command)}")
    print(
        "Estimated learning-plus-production wall across requested chains: "
        f"{total_seconds / 3600.0:.2f} h"
    )
    if conflicts:
        print(
            "Step and time caps conflict for N="
            f"{', '.join(str(n) for n in conflicts)}; --wl_max_steps is "
            "unreachable there and --wl_max_seconds is what stops the run."
        )
    print("Fit and bootstrap time is not included in this estimate.")
    return 0


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run gated N=30 -> 44 -> 60 Wang-Landau baseline acceptance.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--chains", type=int, nargs="+", default=list(CHAIN_ORDER))
    parser.add_argument("--outdir", type=Path, default=Path("wl_pilot"))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry_run", "--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--skip_bootstrap", action="store_true")
    parser.add_argument("--n_workers", type=int, default=12)
    parser.add_argument("--steps_per_worker", type=int, default=400_000_000)
    parser.add_argument("--base_seed", type=int, default=42)
    parser.add_argument("--wl_seed", type=int, default=1729)
    parser.add_argument("--burnin", type=float, default=0.3)
    parser.add_argument("--sample_every", type=int, default=500)
    parser.add_argument("--wl_final_log_f", type=float, default=1e-4)
    parser.add_argument("--wl_flatness", type=float, default=0.8)
    parser.add_argument("--wl_min_visits", type=int, default=1000)
    parser.add_argument("--wl_min_cover_visits", type=int, default=50)
    parser.add_argument("--wl_check_every", type=int, default=100_000)
    parser.add_argument("--wl_max_steps", type=int, default=10_000_000_000)
    parser.add_argument("--wl_max_seconds", type=float, default=21_600.0)
    parser.add_argument(
        "--wl_max_seconds_scope", choices=("cumulative", "per_invocation"),
        default="cumulative",
    )
    parser.add_argument("--wl_max_steps_per_stage", type=int, default=1_000_000_000)
    parser.add_argument("--wl_schedule", choices=("halving", "one_over_t"), default="halving")
    parser.add_argument("--wl_stage_stall_steps", type=int, default=50_000_000)
    parser.add_argument("--checkpoint_every_seconds", type=float, default=1800.0)
    parser.add_argument("--n_blocks", type=int, default=20)
    parser.add_argument("--min_crossval_samples", type=int, default=100)
    args = parser.parse_args(argv)
    requested = list(args.chains)
    if len(set(requested)) != len(requested) or any(n not in CHAIN_INFO for n in requested):
        parser.error("--chains must be unique values drawn from 30, 44, 60")
    args.chains = [n for n in CHAIN_ORDER if n in requested]
    if args.n_workers < 1 or args.steps_per_worker < 1 or args.n_blocks < 2:
        parser.error("worker, step, and block controls must be positive (n_blocks >= 2)")
    args.outdir = args.outdir.resolve()
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.dry_run:
        return _dry_run(args)

    args.outdir.mkdir(parents=True, exist_ok=True)
    summary: Dict[str, Any] = {"chain_order": args.chains, "chains": {}}
    summary_path = args.outdir / "acceptance_summary.json"
    report_path = args.outdir / "acceptance_report.md"
    try:
        for n_beads in args.chains:
            info = CHAIN_INFO[n_beads]
            chain_dir = args.outdir / f"N{n_beads}"
            wl_dir = chain_dir / "wl"
            wl_dir.mkdir(parents=True, exist_ok=True)
            wl_path = _wl_output_path(args, n_beads, wl_dir)
            command = _wl_command(args, n_beads, wl_dir)
            if args.force or not wl_path.exists():
                try:
                    _run_command(command, chain_dir / "wl.log")
                except AcceptanceFailure as error:
                    raise AcceptanceFailure(
                        f"N={n_beads} Wang-Landau run failed. Inspect the log for "
                        "the highest reached contact. If it is below m_cover, this "
                        f"triggers W8 rather than truncation. {error}"
                    ) from error
            wl_summary = _validate_wl_output(wl_path)

            legacy_path = AUTO_DIR / info["legacy"]
            cross = _cross_validate(
                legacy_path, wl_path, chain_dir / "cross_validation",
                args.min_crossval_samples, args.n_blocks,
            )
            target_path = AUTO_DIR / info["target"]
            with np.load(legacy_path, allow_pickle=False) as legacy_data:
                legacy_ceiling = int(np.asarray(legacy_data["c_vals"]).max())
            upper_support = {
                "before": _upper_tail_metrics(
                    target_path, CONTACT_OFFSETS[n_beads], legacy_ceiling
                ),
                "after": _upper_tail_metrics(
                    target_path, CONTACT_OFFSETS[n_beads], info["m_max"]
                ),
                "reference_before": info["reference_before"],
                "reference_after": info["reference_after"],
            }

            config = json.loads((AUTO_DIR / info["config"]).read_text())
            rg_scale = float(config["baselines"][0]["rg_scale"])
            for label, baseline in (("legacy", legacy_path), ("new", wl_path)):
                support_dir = chain_dir / "support" / label
                _run_command(
                    _support_command(
                        args, target_path, baseline, CONTACT_OFFSETS[n_beads],
                        rg_scale, support_dir,
                    ),
                    chain_dir / f"support_{label}.log",
                )

            fit_comparisons = []
            models = args.models or config["models"]
            for model in models:
                legacy_fit = chain_dir / "fits" / model / "legacy"
                new_fit = chain_dir / "fits" / model / "new"
                for label, baseline, fit_dir in (
                    ("legacy", legacy_path, legacy_fit),
                    ("new", wl_path, new_fit),
                ):
                    if args.force or not (fit_dir / "fit_summary.json").exists():
                        _run_command(
                            _fit_command(
                                args, config, n_beads, model, target_path,
                                baseline, fit_dir,
                            ),
                            chain_dir / f"fit_{model}_{label}.log",
                        )
                fit_comparisons.append(
                    _compare_fit_pair(legacy_fit, new_fit, target_path)
                )

            summary["chains"][str(n_beads)] = {
                "wl": wl_summary,
                "cross_validation": cross,
                "upper_tail_support": upper_support,
                "fits": fit_comparisons,
            }
            summary_path.write_text(
                json.dumps(_jsonable(summary), indent=2), encoding="utf-8"
            )
            _write_report(_jsonable(summary), report_path)
    except AcceptanceFailure as error:
        summary["failure"] = str(error)
        summary_path.write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8")
        _write_report(_jsonable(summary), report_path)
        print(f"ACCEPTANCE FAILED: {error}", file=sys.stderr)
        return 1

    print(f"Acceptance report: {report_path}")
    print(f"Machine summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
