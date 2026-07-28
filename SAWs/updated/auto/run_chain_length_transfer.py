#!/usr/bin/env python3
"""Run and compare chain-length transferability tests for fitted models.

Each fit summary is evaluated with the same runtime chain length, temperature
ladder, and seed list.  The script calls ``remd_uniform_chain_2_new.py`` for
every fit/seed pair, aggregates the resulting observables across seeds, and
writes comparison CSV, JSON, and PNG files.

Example
-------
Evaluate N=44 using parameterizations fitted at N=30, 44, and 60:

    python run_chain_length_transfer.py \
      --N 44 \
      --fit-summary N30=/path/to/N30/fit_summary.json \
      --fit-summary N44=/path/to/N44/fit_summary.json \
      --fit-summary N60=/path/to/N60/fit_summary.json \
      --target-npz remd_distributions_44mer.npz \
      --seeds 1,2,3,4,5 \
      --n-workers 16 \
      --outdir transferability_N44

``--target-npz`` supplies both the common temperature ladder and optional
reference curves.  Target contacts are shifted by ``N - 1`` by default, which
matches the fitting convention; use ``--target-contact-offset`` to override it.

Rg scaling
----------
By default, each run uses the ``rg_scale`` stored in its fit summary.  Pass one
common ``--rg-scale`` to isolate contact-potential transfer from differences in
the fitted Rg mapping.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class FitSpec:
    label: str
    slug: str
    path: Path
    fit_chain_length: int | None
    model: str
    rg_scale: float
    summary: dict


def _slugify(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip()).strip("._-")
    if not slug:
        raise ValueError(f"Could not make a safe output name from label {text!r}")
    return slug


def _infer_chain_length(text: str) -> int | None:
    for pattern in (
        r"(?i)(?:^|[^A-Za-z0-9])N[_-]?(\d+)(?:[^0-9]|$)",
        r"(?i)(?:^|[^0-9])(\d+)[_-]?mer(?:[^A-Za-z0-9]|$)",
        r"^\s*(\d+)\s*$",
    ):
        match = re.search(pattern, text)
        if match:
            value = int(match.group(1))
            return value if value >= 2 else None
    return None


def _load_fit_spec(raw: str, common_rg_scale: float | None) -> FitSpec:
    if "=" in raw:
        label, path_text = raw.split("=", 1)
        label = label.strip()
    else:
        label, path_text = "", raw

    path = Path(path_text).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Fit summary does not exist: {path}")
    with path.open(encoding="utf-8") as handle:
        summary = json.load(handle)
    if not isinstance(summary, dict):
        raise ValueError(f"Fit summary must contain a JSON object: {path}")

    model = summary.get("model")
    if not isinstance(model, str) or not model:
        raise ValueError(f"Fit summary is missing a valid 'model' field: {path}")
    if not isinstance(summary.get("params"), dict):
        raise ValueError(f"Fit summary is missing a parameter dictionary: {path}")

    raw_n = summary.get("fit_chain_length")
    fit_n: int | None = None
    if raw_n is not None:
        fit_n = int(raw_n)
        if fit_n < 0:
            fit_n = None
        elif fit_n < 2:
            raise ValueError(f"Invalid fit_chain_length={fit_n} in {path}")

    if not label:
        if fit_n is None:
            fit_n = _infer_chain_length(str(path))
        if fit_n is None:
            raise ValueError(
                f"{path} does not record fit_chain_length. Supply a labeled "
                f"argument such as --fit-summary N30={path}"
            )
        label = f"N{fit_n}"

    if fit_n is None:
        fit_n = _infer_chain_length(label)

    if common_rg_scale is None:
        rg_scale = float(summary.get("rg_scale", 1.0))
    else:
        rg_scale = float(common_rg_scale)
    if not math.isfinite(rg_scale) or rg_scale <= 0.0:
        raise ValueError(f"Invalid rg_scale={rg_scale!r} for {path}")

    return FitSpec(
        label=label,
        slug=_slugify(label),
        path=path,
        fit_chain_length=fit_n,
        model=model,
        rg_scale=rg_scale,
        summary=summary,
    )


def _parse_int_list(text: str, flag: str) -> list[int]:
    try:
        values = [int(part.strip()) for part in text.split(",") if part.strip()]
    except ValueError as exc:
        raise ValueError(f"{flag} must be a comma-separated integer list") from exc
    if not values:
        raise ValueError(f"{flag} must contain at least one integer")
    if len(set(values)) != len(values):
        raise ValueError(f"{flag} contains duplicate values: {values}")
    if any(value < 0 for value in values):
        raise ValueError(f"{flag} values must be nonnegative: {values}")
    return values


def _validate_temperatures(values: Iterable[float]) -> np.ndarray:
    temps = np.asarray(list(values), dtype=float)
    if temps.ndim != 1 or temps.size < 2:
        raise ValueError("The temperature ladder must contain at least two values")
    if not np.all(np.isfinite(temps)) or not np.all(temps > 0.0):
        raise ValueError("Temperatures must be finite and positive")
    if not np.all(np.diff(temps) > 0.0):
        raise ValueError("Temperatures must be strictly increasing")
    return temps


def _temperatures_from_npz(path: Path) -> np.ndarray:
    with np.load(path) as data:
        for key in ("temps", "Ts"):
            if key in data:
                return _validate_temperatures(data[key])
        raise ValueError(
            f"{path} has neither a 'temps' nor a 'Ts' temperature array"
        )


def _resolve_temperatures(args: argparse.Namespace) -> np.ndarray:
    if args.target_npz is not None:
        return _temperatures_from_npz(args.target_npz)
    if args.temps is not None:
        return _validate_temperatures(
            float(part.strip()) for part in args.temps.split(",") if part.strip()
        )
    if args.nT < 2:
        raise ValueError("--nT must be at least 2")
    if not (args.Tmin > 0.0 and args.Tmax > args.Tmin):
        raise ValueError("Require 0 < --Tmin < --Tmax")
    return _validate_temperatures(np.linspace(args.Tmin, args.Tmax, args.nT))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(payload), handle, indent=2, allow_nan=False)
        handle.write("\n")


def _run_paths(outdir: Path, spec: FitSpec, seed: int) -> dict[str, Path]:
    run_dir = outdir / f"fit_{spec.slug}" / f"seed_{seed}"
    prefix = run_dir / "run"
    return {
        "dir": run_dir,
        "prefix": prefix,
        "results": Path(f"{prefix}_results.csv"),
        "distributions": Path(f"{prefix}_distributions.npz"),
        "summary": Path(f"{prefix}_run_summary.json"),
        "stdout": run_dir / "stdout.log",
        "stderr": run_dir / "stderr.log",
        "command": run_dir / "command.txt",
        "inputs": run_dir / "run_inputs.json",
    }


def _is_complete(paths: dict[str, Path]) -> bool:
    return all(paths[key].is_file() for key in ("results", "distributions", "summary"))


def _build_command(
    args: argparse.Namespace,
    spec: FitSpec,
    seed: int,
    ladder_path: Path,
    paths: dict[str, Path],
) -> list[str]:
    command = [
        sys.executable,
        str(args.remd_script),
        "--N",
        str(args.N),
        "--temps-from-npz",
        str(ladder_path),
        "--steps-per-swap",
        str(args.steps_per_swap),
        "--n-cycles",
        str(args.n_cycles),
        "--seed",
        str(seed),
        "--n-workers",
        str(args.n_workers),
        "--burnin-frac",
        str(args.burnin_frac),
        "--rg-bins",
        str(args.rg_bins),
        "--rg-scale",
        str(spec.rg_scale),
        "--fit-summary-json",
        str(spec.path),
        "--out-prefix",
        str(paths["prefix"]),
        "--no-plots",
    ]
    if args.diagnostics:
        command.append("--diagnostics")
    if args.timing:
        command.append("--timing")
    return command


def _run_simulations(
    args: argparse.Namespace,
    specs: list[FitSpec],
    seeds: list[int],
    ladder_path: Path,
) -> list[dict]:
    statuses: list[dict] = []
    total = len(specs) * len(seeds)
    ordinal = 0
    for spec in specs:
        for seed in seeds:
            ordinal += 1
            paths = _run_paths(args.outdir, spec, seed)
            command = _build_command(args, spec, seed, ladder_path, paths)
            status = {
                "label": spec.label,
                "fit_chain_length": spec.fit_chain_length,
                "seed": seed,
                "command": command,
                "results_csv": paths["results"],
                "distributions_npz": paths["distributions"],
                "run_summary_json": paths["summary"],
            }
            run_inputs = {
                "command": command,
                "fit_summary_sha256": _sha256(spec.path),
            }

            if args.dry_run:
                status["status"] = "dry_run"
                print(f"[{ordinal}/{total}] {shlex.join(command)}")
                statuses.append(status)
                continue

            paths["dir"].mkdir(parents=True, exist_ok=True)
            if _is_complete(paths) and not args.force:
                existing_inputs = None
                if paths["inputs"].is_file():
                    with paths["inputs"].open(encoding="utf-8") as handle:
                        existing_inputs = json.load(handle)
                if existing_inputs == run_inputs:
                    status["status"] = "reused"
                    print(
                        f"[{ordinal}/{total}] Reusing {spec.label}, seed {seed}: "
                        f"{paths['dir']}"
                    )
                    statuses.append(status)
                    continue
                print(
                    f"[{ordinal}/{total}] Inputs changed for {spec.label}, seed "
                    f"{seed}; rerunning stale output"
                )

            paths["command"].write_text(shlex.join(command) + "\n", encoding="utf-8")
            _write_json(paths["inputs"], run_inputs)
            print(f"[{ordinal}/{total}] Running {spec.label}, seed {seed}")
            started = time.perf_counter()
            with paths["stdout"].open("w", encoding="utf-8") as stdout_handle:
                with paths["stderr"].open("w", encoding="utf-8") as stderr_handle:
                    completed = subprocess.run(
                        command,
                        cwd=args.remd_script.parent,
                        stdout=stdout_handle,
                        stderr=stderr_handle,
                        check=False,
                    )
            status["wall_time_seconds"] = time.perf_counter() - started
            status["returncode"] = completed.returncode
            if completed.returncode == 0 and _is_complete(paths):
                status["status"] = "completed"
            else:
                status["status"] = "failed"
                message = (
                    f"REMD failed for {spec.label}, seed {seed}; see "
                    f"{paths['stderr']}"
                )
                statuses.append(status)
                if not args.continue_on_error:
                    raise RuntimeError(message)
                print(f"WARNING: {message}", file=sys.stderr)
                continue
            statuses.append(status)
    return statuses


def _validate_existing_runs(
    args: argparse.Namespace,
    specs: list[FitSpec],
    seeds: list[int],
    ladder_path: Path,
) -> None:
    """Require analyze-only inputs to match the current requested experiment."""
    problems = []
    for spec in specs:
        for seed in seeds:
            paths = _run_paths(args.outdir, spec, seed)
            expected = {
                "command": _build_command(args, spec, seed, ladder_path, paths),
                "fit_summary_sha256": _sha256(spec.path),
            }
            if not _is_complete(paths):
                problems.append(f"{spec.label}, seed {seed}: incomplete outputs")
                continue
            if not paths["inputs"].is_file():
                problems.append(f"{spec.label}, seed {seed}: missing run_inputs.json")
                continue
            with paths["inputs"].open(encoding="utf-8") as handle:
                recorded = json.load(handle)
            if recorded != expected:
                problems.append(f"{spec.label}, seed {seed}: inputs have changed")
    if problems:
        detail = "\n  - ".join(problems)
        raise RuntimeError(
            "--analyze-only cannot use stale or incomplete runs:\n  - " + detail
        )


def _read_results(path: Path) -> dict[str, np.ndarray]:
    columns = {"T": [], "C_mean": [], "C_std": [], "Rg_mean": [], "Rg_std": []}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = [name for name in columns if name not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path} is missing columns: {missing}")
        for row in reader:
            for name in columns:
                columns[name].append(float(row[name]))
    return {name: np.asarray(values, dtype=float) for name, values in columns.items()}


def _sample_summary(values: list[float]) -> tuple[int, float, float, float]:
    finite = np.asarray([value for value in values if math.isfinite(value)], dtype=float)
    if finite.size == 0:
        return 0, math.nan, math.nan, math.nan
    mean = float(np.mean(finite))
    if finite.size < 2:
        return int(finite.size), mean, math.nan, math.nan
    sd = float(np.std(finite, ddof=1))
    return int(finite.size), mean, sd, sd / math.sqrt(float(finite.size))


def _center_widths(centers: np.ndarray) -> np.ndarray:
    centers = np.asarray(centers, dtype=float)
    if centers.ndim != 1 or centers.size < 2 or not np.all(np.diff(centers) > 0.0):
        raise ValueError("Histogram centers must be a strictly increasing 1D array")
    edges = np.empty(centers.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
    edges[0] = centers[0] - 0.5 * (centers[1] - centers[0])
    edges[-1] = centers[-1] + 0.5 * (centers[-1] - centers[-2])
    return np.diff(edges)


def _histogram_means(centers: np.ndarray, histograms: np.ndarray) -> np.ndarray:
    centers = np.asarray(centers, dtype=float)
    histograms = np.asarray(histograms, dtype=float)
    if histograms.ndim != 2 or histograms.shape[1] != centers.size:
        raise ValueError(
            f"Histogram shape {histograms.shape} is incompatible with "
            f"{centers.size} centers"
        )
    widths = _center_widths(centers)
    means = np.full(histograms.shape[0], np.nan, dtype=float)
    for index, row in enumerate(histograms):
        mass = np.where(np.isfinite(row) & (row >= 0.0), row, 0.0) * widths
        total = float(np.sum(mass))
        if total > 0.0:
            means[index] = float(np.sum(mass * centers) / total)
    return means


def _load_target(path: Path, contact_offset: float) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        required = ("ct_centers", "ct_hists", "rg_centers", "rg_hists")
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(f"Target NPZ {path} is missing keys: {missing}")
        temperatures = None
        for key in ("temps", "Ts"):
            if key in data:
                temperatures = _validate_temperatures(data[key])
                break
        if temperatures is None:
            raise ValueError(f"Target NPZ {path} has no temperature array")
        contact_centers = np.asarray(data["ct_centers"], dtype=float) - contact_offset
        rg_centers = np.asarray(data["rg_centers"], dtype=float)
        contact_mean = _histogram_means(contact_centers, data["ct_hists"])
        rg_mean = _histogram_means(rg_centers, data["rg_hists"])
    return {
        "T": temperatures,
        "C_mean": contact_mean,
        "Rg_mean": rg_mean,
    }


def _aggregate_runs(
    args: argparse.Namespace,
    specs: list[FitSpec],
    seeds: list[int],
    temperatures: np.ndarray,
    target: dict[str, np.ndarray] | None,
    successful_runs: set[tuple[str, int]] | None = None,
) -> tuple[list[dict], dict[str, dict[str, np.ndarray]]]:
    rows: list[dict] = []
    curves: dict[str, dict[str, np.ndarray]] = {}
    for spec in specs:
        runs = []
        for seed in seeds:
            if (
                successful_runs is not None
                and (spec.label, seed) not in successful_runs
            ):
                continue
            paths = _run_paths(args.outdir, spec, seed)
            if paths["results"].is_file():
                result = _read_results(paths["results"])
                if result["T"].shape != temperatures.shape or not np.allclose(
                    result["T"], temperatures, rtol=0.0, atol=1e-9
                ):
                    raise ValueError(
                        f"Temperature mismatch in {paths['results']}; all fits "
                        "must use the same ladder"
                    )
                runs.append((seed, result))
        if not runs:
            raise RuntimeError(f"No successful result files found for {spec.label}")

        curve = {
            "T": temperatures.copy(),
            "C_mean": np.full(temperatures.size, np.nan),
            "C_seed_sd": np.full(temperatures.size, np.nan),
            "C_seed_sem": np.full(temperatures.size, np.nan),
            "Rg_mean": np.full(temperatures.size, np.nan),
            "Rg_seed_sd": np.full(temperatures.size, np.nan),
            "Rg_seed_sem": np.full(temperatures.size, np.nan),
        }
        for index, temperature in enumerate(temperatures):
            c_values = [float(run["C_mean"][index]) for _, run in runs]
            rg_values = [float(run["Rg_mean"][index]) for _, run in runs]
            c_n, c_mean, c_sd, c_sem = _sample_summary(c_values)
            rg_n, rg_mean, rg_sd, rg_sem = _sample_summary(rg_values)
            curve["C_mean"][index] = c_mean
            curve["C_seed_sd"][index] = c_sd
            curve["C_seed_sem"][index] = c_sem
            curve["Rg_mean"][index] = rg_mean
            curve["Rg_seed_sd"][index] = rg_sd
            curve["Rg_seed_sem"][index] = rg_sem
            row = {
                "evaluation_N": args.N,
                "parameterization": spec.label,
                "fit_chain_length": spec.fit_chain_length,
                "native_fit": spec.fit_chain_length == args.N,
                "model": spec.model,
                "rg_scale": spec.rg_scale,
                "temperature": float(temperature),
                "n_successful_seeds": min(c_n, rg_n),
                "contacts_mean": c_mean,
                "contacts_seed_sd": c_sd,
                "contacts_seed_sem": c_sem,
                "rg_mean": rg_mean,
                "rg_seed_sd": rg_sd,
                "rg_seed_sem": rg_sem,
            }
            if target is not None:
                row["target_contacts_mean"] = float(target["C_mean"][index])
                row["target_rg_mean"] = float(target["Rg_mean"][index])
                row["contacts_error"] = c_mean - row["target_contacts_mean"]
                row["rg_error"] = rg_mean - row["target_rg_mean"]
            rows.append(row)
        curves[spec.label] = curve
    return rows, curves


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        ""
                        if isinstance(value, float) and not math.isfinite(value)
                        else value
                    )
                    for key, value in row.items()
                }
            )


def _rmse(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.sqrt(np.mean(finite**2))) if finite.size else math.nan


def _mae(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.mean(np.abs(finite))) if finite.size else math.nan


def _comparison_tables(
    args: argparse.Namespace,
    specs: list[FitSpec],
    curves: dict[str, dict[str, np.ndarray]],
    target: dict[str, np.ndarray] | None,
) -> tuple[list[dict], list[dict], list[dict]]:
    metrics = []
    if target is not None:
        for spec in specs:
            curve = curves[spec.label]
            c_error = curve["C_mean"] - target["C_mean"]
            rg_error = curve["Rg_mean"] - target["Rg_mean"]
            metrics.append(
                {
                    "evaluation_N": args.N,
                    "parameterization": spec.label,
                    "fit_chain_length": spec.fit_chain_length,
                    "native_fit": spec.fit_chain_length == args.N,
                    "model": spec.model,
                    "contacts_rmse": _rmse(c_error),
                    "contacts_mae": _mae(c_error),
                    "contacts_max_abs_error": float(np.nanmax(np.abs(c_error))),
                    "rg_rmse": _rmse(rg_error),
                    "rg_mae": _mae(rg_error),
                    "rg_max_abs_error": float(np.nanmax(np.abs(rg_error))),
                }
            )

    pairwise_rows = []
    pairwise_metrics = []
    for first, second in itertools.combinations(specs, 2):
        curve_a = curves[first.label]
        curve_b = curves[second.label]
        delta_c = curve_a["C_mean"] - curve_b["C_mean"]
        delta_rg = curve_a["Rg_mean"] - curve_b["Rg_mean"]
        for index, temperature in enumerate(curve_a["T"]):
            pairwise_rows.append(
                {
                    "evaluation_N": args.N,
                    "parameterization_a": first.label,
                    "fit_chain_length_a": first.fit_chain_length,
                    "parameterization_b": second.label,
                    "fit_chain_length_b": second.fit_chain_length,
                    "temperature": float(temperature),
                    "contacts_a_minus_b": float(delta_c[index]),
                    "abs_contacts_difference": float(abs(delta_c[index])),
                    "rg_a_minus_b": float(delta_rg[index]),
                    "abs_rg_difference": float(abs(delta_rg[index])),
                }
            )
        pairwise_metrics.append(
            {
                "evaluation_N": args.N,
                "parameterization_a": first.label,
                "fit_chain_length_a": first.fit_chain_length,
                "parameterization_b": second.label,
                "fit_chain_length_b": second.fit_chain_length,
                "contacts_difference_rmse": _rmse(delta_c),
                "contacts_difference_max_abs": float(np.nanmax(np.abs(delta_c))),
                "rg_difference_rmse": _rmse(delta_rg),
                "rg_difference_max_abs": float(np.nanmax(np.abs(delta_rg))),
            }
        )
    return metrics, pairwise_rows, pairwise_metrics


def _plot_observables(
    path: Path,
    args: argparse.Namespace,
    specs: list[FitSpec],
    curves: dict[str, dict[str, np.ndarray]],
    target: dict[str, np.ndarray] | None,
    reference_label: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharex="col")
    ax_rg, ax_c = axes[0]
    ax_drg, ax_dc = axes[1]
    reference = curves[reference_label]
    colors = plt.get_cmap("tab10")

    for index, spec in enumerate(specs):
        curve = curves[spec.label]
        color = colors(index % 10)
        native = spec.fit_chain_length == args.N
        line_width = 2.7 if native else 1.8
        marker = "o" if native else None
        ax_rg.plot(
            curve["T"],
            curve["Rg_mean"],
            color=color,
            linewidth=line_width,
            marker=marker,
            markersize=3,
            label=spec.label,
        )
        ax_c.plot(
            curve["T"],
            curve["C_mean"],
            color=color,
            linewidth=line_width,
            marker=marker,
            markersize=3,
            label=spec.label,
        )
        if np.any(np.isfinite(curve["Rg_seed_sd"])):
            ax_rg.fill_between(
                curve["T"],
                curve["Rg_mean"] - curve["Rg_seed_sd"],
                curve["Rg_mean"] + curve["Rg_seed_sd"],
                color=color,
                alpha=0.14,
                linewidth=0,
            )
        if np.any(np.isfinite(curve["C_seed_sd"])):
            ax_c.fill_between(
                curve["T"],
                curve["C_mean"] - curve["C_seed_sd"],
                curve["C_mean"] + curve["C_seed_sd"],
                color=color,
                alpha=0.14,
                linewidth=0,
            )
        ax_drg.plot(
            curve["T"],
            curve["Rg_mean"] - reference["Rg_mean"],
            color=color,
            linewidth=line_width,
        )
        ax_dc.plot(
            curve["T"],
            curve["C_mean"] - reference["C_mean"],
            color=color,
            linewidth=line_width,
        )

    if target is not None:
        ax_rg.plot(
            target["T"],
            target["Rg_mean"],
            color="black",
            linestyle="--",
            linewidth=2.0,
            label="target data",
        )
        ax_c.plot(
            target["T"],
            target["C_mean"],
            color="black",
            linestyle="--",
            linewidth=2.0,
            label="target data",
        )

    ax_rg.set_ylabel("mean Rg")
    ax_c.set_ylabel("mean nonbonded contacts")
    ax_drg.set_ylabel(f"ΔRg vs {reference_label}")
    ax_dc.set_ylabel(f"Δcontacts vs {reference_label}")
    ax_drg.set_xlabel("temperature")
    ax_dc.set_xlabel("temperature")
    ax_rg.set_title(f"Rg transfer at evaluation N={args.N}")
    ax_c.set_title(f"Contact transfer at evaluation N={args.N}")
    ax_drg.axhline(0.0, color="0.4", linewidth=1.0)
    ax_dc.axhline(0.0, color="0.4", linewidth=1.0)
    ax_rg.legend(frameon=False)
    ax_c.legend(frameon=False)
    for axis in axes.flat:
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_target_errors(path: Path, metrics: list[dict]) -> None:
    if not metrics:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [
        (
            f"{row['parameterization']}\nfit N={row['fit_chain_length']}"
            if row["fit_chain_length"] is not None
            else row["parameterization"]
        )
        for row in metrics
    ]
    x = np.arange(len(metrics))
    figure, (ax_c, ax_rg) = plt.subplots(1, 2, figsize=(11, 4.5))
    colors = ["tab:blue" if row["native_fit"] else "tab:orange" for row in metrics]
    ax_c.bar(x, [row["contacts_rmse"] for row in metrics], color=colors)
    ax_rg.bar(x, [row["rg_rmse"] for row in metrics], color=colors)
    for axis, ylabel in (
        (ax_c, "contact RMSE vs target"),
        (ax_rg, "Rg RMSE vs target"),
    ):
        axis.set_xticks(x, labels, rotation=25, ha="right")
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle("Transfer error across the common temperature ladder")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_coefficients(
    path: Path,
    args: argparse.Namespace,
    specs: list[FitSpec],
    seeds: list[int],
) -> None:
    coefficient_curves = []
    for spec in specs:
        distribution_path = None
        for seed in seeds:
            candidate = _run_paths(args.outdir, spec, seed)["distributions"]
            if candidate.is_file():
                distribution_path = candidate
                break
        if distribution_path is None:
            continue
        with np.load(distribution_path) as data:
            required = (
                "Ts",
                "linear_coefficient_by_temperature",
                "quadratic_coefficient_by_temperature",
            )
            if any(key not in data for key in required):
                continue
            coefficient_curves.append(
                (
                    spec,
                    np.asarray(data["Ts"], dtype=float),
                    np.asarray(data["linear_coefficient_by_temperature"], dtype=float),
                    np.asarray(data["quadratic_coefficient_by_temperature"], dtype=float),
                )
            )
    if not coefficient_curves:
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, (ax_linear, ax_quadratic) = plt.subplots(1, 2, figsize=(11, 4.5))
    for spec, temperatures, linear, quadratic in coefficient_curves:
        ax_linear.plot(temperatures, linear, linewidth=2.0, label=spec.label)
        ax_quadratic.plot(temperatures, quadratic, linewidth=2.0, label=spec.label)
    ax_linear.set_ylabel("linear contact coefficient")
    ax_quadratic.set_ylabel("quadratic contact coefficient")
    for axis in (ax_linear, ax_quadratic):
        axis.set_xlabel("temperature")
        axis.grid(alpha=0.2)
        axis.legend(frameon=False)
    figure.suptitle(f"Parameterizations used at runtime N={args.N}")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _choose_reference(
    requested: str | None, specs: list[FitSpec], evaluation_n: int
) -> str:
    labels = [spec.label for spec in specs]
    if requested is not None:
        if requested not in labels:
            raise ValueError(f"--reference must be one of: {labels}")
        return requested
    for spec in specs:
        if spec.fit_chain_length == evaluation_n:
            return spec.label
    return specs[0].label


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Run one fixed-N REMD comparison with several fit-summary "
            "parameterizations."
        ),
    )
    parser.add_argument("--N", type=int, required=True, help="runtime bead count")
    parser.add_argument(
        "--fit-summary",
        action="append",
        required=True,
        metavar="[LABEL=]PATH",
        help=(
            "Fit summary to evaluate; repeat for every training chain length. "
            "A label such as N30= is required for legacy summaries that do not "
            "record fit_chain_length."
        ),
    )
    parser.add_argument(
        "--target-contact-offset",
        type=float,
        default=None,
        help="Value subtracted from target ct_centers; default N-1.",
    )
    parser.add_argument(
        "--reference",
        default=None,
        help="Label used for difference plots; defaults to the native fit, if present.",
    )
    parser.add_argument(
        "--rg-scale",
        type=float,
        default=None,
        help=(
            "Common Rg conversion for every run. By default, each fit summary's "
            "own rg_scale is used."
        ),
    )
    parser.add_argument("--seeds", default="1,2,3", help="paired REMD seed list")
    parser.add_argument("--steps-per-swap", type=int, default=1000)
    parser.add_argument("--n-cycles", type=int, default=5000)
    parser.add_argument("--n-workers", type=int, default=1)
    parser.add_argument("--burnin-frac", type=float, default=0.7)
    parser.add_argument("--rg-bins", type=int, default=100)
    parser.add_argument(
        "--remd-script",
        type=Path,
        default=Path(__file__).resolve().with_name("remd_uniform_chain_2_new.py"),
    )
    parser.add_argument("--outdir", type=Path, default=None)

    temperature_group = parser.add_mutually_exclusive_group()
    temperature_group.add_argument(
        "--target-npz",
        type=Path,
        default=None,
        help=(
            "Evaluation-chain data NPZ. Supplies the exact temperature ladder "
            "and target mean-contact/Rg curves."
        ),
    )
    temperature_group.add_argument(
        "--temps", default=None, help="comma-separated ladder when no target NPZ is used"
    )
    temperature_group.add_argument(
        "--nT",
        type=int,
        default=64,
        help="uniform ladder size when no target NPZ or --temps is used",
    )
    parser.add_argument("--Tmin", type=float, default=280.0)
    parser.add_argument("--Tmax", type=float, default=360.0)

    parser.add_argument(
        "--diagnostics", action="store_true", help="enable sampler diagnostics"
    )
    parser.add_argument("--timing", action="store_true")
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="skip simulations and rebuild comparison outputs from existing runs",
    )
    parser.add_argument(
        "--force", action="store_true", help="rerun complete fit/seed directories"
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="analyze successful runs even if another fit/seed run fails",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate inputs and print commands without running or writing outputs",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    if args.N < 2:
        parser.error("--N must be at least 2")
    if args.steps_per_swap < 1 or args.n_cycles < 1:
        parser.error("--steps-per-swap and --n-cycles must be positive")
    if args.n_workers < 1 or args.rg_bins < 2:
        parser.error("--n-workers must be positive and --rg-bins must be at least 2")
    if not (0.0 <= args.burnin_frac < 1.0):
        parser.error("--burnin-frac must be in [0, 1)")
    if args.rg_scale is not None and (
        not math.isfinite(args.rg_scale) or args.rg_scale <= 0.0
    ):
        parser.error("--rg-scale must be finite and positive")

    args.remd_script = args.remd_script.expanduser().resolve()
    if not args.remd_script.is_file():
        parser.error(f"REMD script not found: {args.remd_script}")
    args.target_npz = (
        args.target_npz.expanduser().resolve() if args.target_npz is not None else None
    )
    if args.target_npz is not None and not args.target_npz.is_file():
        parser.error(f"Target NPZ not found: {args.target_npz}")
    if args.target_npz is None and args.target_contact_offset is not None:
        parser.error("--target-contact-offset requires --target-npz")
    args.outdir = (
        args.outdir.expanduser().resolve()
        if args.outdir is not None
        else (Path.cwd() / f"chain_length_transfer_N{args.N}").resolve()
    )

    seeds = _parse_int_list(args.seeds, "--seeds")
    specs = [_load_fit_spec(raw, args.rg_scale) for raw in args.fit_summary]
    labels = [spec.label for spec in specs]
    slugs = [spec.slug for spec in specs]
    if len(set(labels)) != len(labels):
        parser.error(f"Fit-summary labels must be unique: {labels}")
    if len(set(slugs)) != len(slugs):
        parser.error(f"Fit-summary labels map to duplicate output names: {slugs}")
    if len(specs) < 2:
        parser.error("Provide at least two --fit-summary arguments to compare")
    models = sorted({spec.model for spec in specs})
    if len(models) > 1:
        print(
            "WARNING: fit summaries use different model families "
            f"{models}; results compare both model form and chain-length transfer.",
            file=sys.stderr,
        )
    scales = sorted({round(spec.rg_scale, 14) for spec in specs})
    if args.rg_scale is None and len(scales) > 1:
        print(
            "WARNING: fit summaries use different rg_scale values. Pass one "
            "common --rg-scale to isolate contact-potential transfer.",
            file=sys.stderr,
        )

    temperatures = _resolve_temperatures(args)
    reference_label = _choose_reference(args.reference, specs, args.N)
    contact_offset = (
        float(args.N - 1)
        if args.target_contact_offset is None
        else float(args.target_contact_offset)
    )
    if not math.isfinite(contact_offset):
        parser.error("--target-contact-offset must be finite")

    ladder_path = args.outdir / "temperature_ladder.npz"
    manifest = {
        "evaluation_N": args.N,
        "n_beads": args.N,
        "n_steps": args.N - 1,
        "reference_parameterization": reference_label,
        "seeds": seeds,
        "temperatures": temperatures,
        "target_npz": args.target_npz,
        "target_contact_offset": contact_offset if args.target_npz is not None else None,
        "remd_script": args.remd_script,
        "steps_per_swap": args.steps_per_swap,
        "n_cycles": args.n_cycles,
        "n_workers": args.n_workers,
        "burnin_frac": args.burnin_frac,
        "rg_bins": args.rg_bins,
        "common_rg_scale_override": args.rg_scale,
        "fit_summaries": [
            {
                "label": spec.label,
                "path": spec.path,
                "sha256": _sha256(spec.path),
                "fit_chain_length": spec.fit_chain_length,
                "native_fit": spec.fit_chain_length == args.N,
                "model": spec.model,
                "rg_scale_used": spec.rg_scale,
            }
            for spec in specs
        ],
    }

    if args.dry_run:
        _run_simulations(args, specs, seeds, ladder_path)
        return 0

    args.outdir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(ladder_path, temps=temperatures, Ts=temperatures)
    _write_json(args.outdir / "comparison_manifest.json", manifest)

    statuses = []
    if not args.analyze_only:
        statuses = _run_simulations(args, specs, seeds, ladder_path)
    else:
        _validate_existing_runs(args, specs, seeds, ladder_path)

    target = None
    if args.target_npz is not None:
        target = _load_target(args.target_npz, contact_offset)
        if target["T"].shape != temperatures.shape or not np.allclose(
            target["T"], temperatures, rtol=0.0, atol=1e-9
        ):
            raise ValueError("Target-data temperatures do not match the run ladder")

    successful_runs = None
    if statuses:
        successful_runs = {
            (status["label"], int(status["seed"]))
            for status in statuses
            if status["status"] in {"completed", "reused"}
        }
    summary_rows, curves = _aggregate_runs(
        args, specs, seeds, temperatures, target, successful_runs
    )
    metrics, pairwise_rows, pairwise_metrics = _comparison_tables(
        args, specs, curves, target
    )

    summary_csv = args.outdir / "summary_by_temperature.csv"
    metrics_csv = args.outdir / "target_error_metrics.csv"
    pairwise_csv = args.outdir / "pairwise_temperature_differences.csv"
    pairwise_metrics_csv = args.outdir / "pairwise_difference_metrics.csv"
    observables_plot = args.outdir / "transferability_observables.png"
    errors_plot = args.outdir / "transferability_target_errors.png"
    coefficients_plot = args.outdir / "parameterization_coefficients.png"
    report_path = args.outdir / "transferability_report.json"

    _write_csv(summary_csv, summary_rows)
    if metrics:
        _write_csv(metrics_csv, metrics)
    _write_csv(pairwise_csv, pairwise_rows)
    _write_csv(pairwise_metrics_csv, pairwise_metrics)
    _plot_observables(
        observables_plot, args, specs, curves, target, reference_label
    )
    _plot_target_errors(errors_plot, metrics)
    _plot_coefficients(coefficients_plot, args, specs, seeds)

    report = {
        **manifest,
        "run_statuses": statuses,
        "target_error_metrics": metrics,
        "pairwise_difference_metrics": pairwise_metrics,
        "outputs": {
            "summary_by_temperature_csv": summary_csv,
            "target_error_metrics_csv": metrics_csv if metrics else None,
            "pairwise_temperature_differences_csv": pairwise_csv,
            "pairwise_difference_metrics_csv": pairwise_metrics_csv,
            "transferability_observables_png": observables_plot,
            "transferability_target_errors_png": errors_plot if metrics else None,
            "parameterization_coefficients_png": (
                coefficients_plot if coefficients_plot.is_file() else None
            ),
        },
    }
    _write_json(report_path, report)

    print(f"Completed transferability comparison at runtime N={args.N}")
    print(f"Reference parameterization: {reference_label}")
    print(f"Results: {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
