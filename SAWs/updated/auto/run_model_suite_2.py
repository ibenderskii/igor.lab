#!/usr/bin/env python3
"""
run_model_suite.py — subprocess orchestrator for the lattice contact-model suite.

For every baseline x every selected contact-bias model this script:
  1. fits the model to the target REMD distributions (fit_lattice_contact_model_2.py),
  2. loads the resulting fit_summary.json,
  3. runs one or more lattice REMD replicates (remd_uniform_chain_2.py),
  4. compares simulated outputs to the target with a standardized metric,
  5. writes a per-baseline comparison table and a provenance manifest.

This is an orchestrator: it never re-implements the fitter or the REMD engine.
All work happens through list-form subprocess calls (no shell=True), using
sys.executable, so it runs on Windows.

This file implements config + preflight (model-contract and numeric b(T)
equality, target/baseline validation), fit + REMD execution with strict output
validation, resume/force/continue-on-error, standardized contact/Rg comparison
with mass-conserving grid alignment, per-temperature tables, plots, report.md,
the provenance manifest, and a self-contained --quick-test.

Usage:
    python run_model_suite.py --config model_suite_config.json
    python run_model_suite.py --config model_suite_config.json --dry-run
    python run_model_suite.py --config model_suite_config.json --resume
    python run_model_suite.py --config model_suite_config.json --force
    python run_model_suite.py --config model_suite_config.json --continue-on-error
    python run_model_suite.py --quick-test
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

try:  # optional: enables Wasserstein distance in the comparison
    from scipy.stats import wasserstein_distance as _wasserstein
except Exception:  # pragma: no cover - scipy may be absent
    _wasserstein = None

# Completion-file sets (Part 15).
FIT_COMPLETION_FILES = (
    "fit_summary.json",
    "fit_results.npz",
    "fit_params.csv",
    "train_validation_loss.csv",
)
# Completion outputs required by each newer fitter analysis when enabled.
# Filenames verified directly against fit_lattice_contact_model_2.py drivers.
# Plot files are deliberately excluded (they only exist when plotting is on and
# are never treated as completion markers).
FIT_BOOTSTRAP_FILES = (
    "bootstrap_summary.json",
    "bootstrap_params.csv",
    "bootstrap_covariance.csv",
    "bootstrap_correlation.csv",
    "bootstrap_bands_by_temperature.csv",
)
FIT_BOOTSTRAP_PREDICTION_BANDS_FILE = "bootstrap_prediction_bands.npz"
FIT_UNCERTAINTY_FILES = (
    "uncertainty_diagnostics.json",
    "restart_diagnostics.csv",
)
FIT_SPLIT_FILES = (
    "split_sensitivity.csv",
    "split_sensitivity_per_temperature.csv",
    "split_parameter_stability.csv",
    "split_sensitivity_summary.json",
)
FIT_RGW_FILES = (
    "rg_weight_sensitivity.csv",
    "rg_weight_per_temperature.csv",
    "rg_weight_parameter_path.csv",
    "rg_weight_summary.json",
)


def expected_fit_outputs(fit_cfg: dict) -> tuple[str, ...]:
    """Completion files the fitter must produce for the given fit configuration.

    Always includes the primary fit files. When an optional analysis is enabled
    its documented completion outputs are appended so post-run validation,
    resume, stale detection, force cleanup, and the manifest all agree on the
    full output set. Plot files are never required.
    """
    files: list[str] = list(FIT_COMPLETION_FILES)
    bs = fit_cfg.get("bootstrap") or {}
    if bs.get("enabled"):
        files += list(FIT_BOOTSTRAP_FILES)
        if bs.get("save_prediction_bands"):
            files.append(FIT_BOOTSTRAP_PREDICTION_BANDS_FILE)
    if (fit_cfg.get("uncertainty_diagnostics") or {}).get("enabled"):
        files += list(FIT_UNCERTAINTY_FILES)
    if (fit_cfg.get("split_sensitivity") or {}).get("enabled"):
        files += list(FIT_SPLIT_FILES)
    if (fit_cfg.get("rg_weight_sensitivity") or {}).get("enabled"):
        files += list(FIT_RGW_FILES)
    return tuple(files)
REMD_COMPLETION_FILES = (
    "run_results.csv",
    "run_swap_rates.csv",
    "run_distributions.npz",
    "run_run_summary.json",
)
# Additional completion files required only when REMD diagnostics are enabled.
# Kept separate so older runs and the resume fingerprint are unaffected when
# diagnostics are off.
REMD_DIAGNOSTIC_FILES = (
    "run_diagnostics.json",
    "run_convergence.csv",
    "run_round_trips.csv",
    "run_walker_occupancy.csv",
)
REMD_DIAGNOSTIC_TRAJECTORY_FILE = "run_diagnostic_trajectories.npz"

# Numeric b(T) cross-check grid (Part 6).
BFN_CHECK_T = (250.0, 300.0, 350.0)
BFN_CHECK_TREF = 300.0
BFN_CHECK_TSCALE = 100.0


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

def _json_safe(obj):
    """Recursively convert NumPy/Path objects to plain Python for json.dump."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_json_safe(v) for v in obj.tolist()]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        return value if math.isfinite(value) else None
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def sanitize_name(name: str) -> str:
    """Make a string safe to use as a directory name."""
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", str(name)).strip("._")
    if not safe:
        raise ValueError(f"Name {name!r} sanitizes to empty string")
    return safe


_SHA256_CACHE: dict[tuple[str, int, int], str] = {}


def sha256_file(path: str | Path) -> str:
    p = Path(path)
    st = p.stat()
    key = (str(p.resolve()), int(st.st_size), int(st.st_mtime_ns))
    cached = _SHA256_CACHE.get(key)
    if cached is not None:
        return cached
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    digest = h.hexdigest()
    _SHA256_CACHE[key] = digest
    return digest


def now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


class Logger:
    """Tiny tee logger: prints and appends to pipeline.log."""

    def __init__(self, path: Path | None):
        self.path = path

    def __call__(self, msg: str) -> None:
        print(msg)
        if self.path is not None:
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(f"{now_iso()}  {msg}\n")


# ---------------------------------------------------------------------------
# Configuration (Part 5)
# ---------------------------------------------------------------------------

REQUIRED_TOP = ("target_remd", "fit_script", "remd_script", "output_root",
                "models", "baselines")
SUPPORTED_MODELS = ("hs", "tc_scale", "hs_quadratic", "poly2", "poly3",
                    "heat_capacity")

# Supported values for the fitter's newer analyses (verified against
# fit_lattice_contact_model_2.py: build_split_schemes() and the --bootstrap-method
# choices). Kept here so the orchestrator can validate config without importing
# the fitter.
SUPPORTED_SPLIT_SCHEMES = (
    "every_third_phase", "kfold", "blocked_low", "blocked_mid", "blocked_high",
    "random",
)
SUPPORTED_BOOTSTRAP_METHODS = ("temperature",)
# Parameter names that are "extra" terms whose interval containing zero is a
# meaningful identifiability signal (a2/a3 polynomial corrections, dCp heat
# capacity). Used only for reporting, never for fitting.
EXTRA_TERM_PARAMS = ("a2", "a3", "dCp")

DEFAULT_FIT = {
    "loss": "js", "fit_rg": False, "rg_weight": 1.0, "holdout_every": None,
    "holdout_indices": None, "train_indices": None, "n_restarts": 8,
    "seed": 123, "bootstrap": 0, "bootstrap_seed": None, "plots": False,
}
# Defaults for the newer fitter analyses (all backward compatible: disabled).
# Mirrors the fitter's own argparse defaults so a bare {"enabled": true} block
# produces the same behavior as running the fitter with just the enabling flag.
DEFAULT_FIT_BOOTSTRAP = {
    "enabled": None,            # None -> derive from replicates > 0
    "replicates": 0,
    "seed": None,              # None -> fitter falls back to fit.seed
    "method": "temperature",
    "confidence": 0.95,
    "correlation_threshold": 0.9,
    "save_prediction_bands": False,
}
DEFAULT_FIT_UNCERTAINTY = {"enabled": False}
DEFAULT_FIT_SPLIT = {
    "enabled": False,
    "schemes": list(SUPPORTED_SPLIT_SCHEMES),
    "config_json": None,
    "seed": None,              # None -> fitter falls back to fit.seed
    "kfold_k": 5,
    "blocked_fraction": 0.2,
    "random_fraction": 0.2,
    "random_repeats": 5,
}
DEFAULT_FIT_RGW = {
    "enabled": False,
    "weights": None,
    "normalization_diagnostics": False,
}
DEFAULT_REMD = {
    "N": None, "steps_per_swap": 1000, "n_cycles": 5000, "rg_bins": 100,
    "burnin_frac": 0.7, "n_workers": 1, "seeds": [1], "plots": False,
    "timing": False, "diagnostics": None,
}
# Diagnostics sub-config defaults (backward compatible: disabled).
DEFAULT_REMD_DIAGNOSTICS = {
    "enabled": False,
    "trajectories": False,   # save post-burn-in traces (needed for cross-seed Rhat)
    "n_blocks": 5,
    "min_round_trips": 1,
    "min_temp_coverage": 0.5,
    "min_ess": 50.0,
    "max_drift": 1.0,
    "min_swap_rate": 0.05,
    "rhat_threshold": 1.1,   # cross-seed convergence flag (suite-side)
}
DEFAULT_COMPARISON = {
    "include_rg": False, "rg_weight": 1.0, "temperature_tolerance": 1e-10,
    "make_plots": False, "statistics": None,
}
# Paired model-comparison statistics defaults (backward compatible: disabled).
DEFAULT_COMPARISON_STATISTICS = {
    "enabled": False,
    "alpha": 0.05,
    "bootstrap_replicates": 10000,
    "seed": 12345,
    "practical_equivalence_epsilon": 0.001,
    "multiple_testing": "holm",
}


def load_config(path: str) -> dict:
    with open(path) as fh:
        cfg = json.load(fh)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config {path!r} must be a JSON object")
    return cfg


def resolve_config_paths(cfg: dict, config_path: str) -> dict:
    """Resolve relative file paths against the configuration file directory."""
    base = Path(config_path).expanduser().resolve().parent

    def _resolve(value) -> str:
        p = Path(value).expanduser()
        return str(p if p.is_absolute() else (base / p).resolve())

    cfg = dict(cfg)
    for key in ("target_remd", "fit_script", "remd_script", "output_root"):
        cfg[key] = _resolve(cfg[key])
    new_baselines = []
    for b in cfg["baselines"]:
        nb = {**b, "path": _resolve(b["path"])}
        # Resolve the optional custom split-sensitivity JSON relative to the
        # config file so it can be hashed into the fit job signature.
        fit = nb.get("fit")
        if isinstance(fit, dict):
            ss = fit.get("split_sensitivity")
            if isinstance(ss, dict) and ss.get("config_json"):
                fit = dict(fit)
                ss = dict(ss)
                ss["config_json"] = _resolve(ss["config_json"])
                fit["split_sensitivity"] = ss
                nb["fit"] = fit
        new_baselines.append(nb)
    cfg["baselines"] = new_baselines
    return cfg


def _merge(defaults: dict, override: dict | None) -> dict:
    out = dict(defaults)
    if override:
        for k, v in override.items():
            out[k] = v
    return out


def _as_float_list(value, label: str, name: str) -> list[float]:
    """Coerce a list or comma string of numbers to a list of floats."""
    if value is None:
        return []
    if isinstance(value, str):
        items = [t.strip() for t in value.split(",") if t.strip()]
    elif isinstance(value, (list, tuple)):
        items = list(value)
    else:
        raise ValueError(
            f"Baseline {name!r}: {label} must be a list or comma-separated string"
        )
    out: list[float] = []
    for it in items:
        try:
            out.append(float(it))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Baseline {name!r}: {label} contains a non-numeric value {it!r}"
            ) from exc
    return out


def _as_str_list(value, label: str, name: str) -> list[str]:
    """Coerce a list or comma string into a list of non-empty strings."""
    if value is None:
        return []
    if isinstance(value, str):
        items = [t.strip() for t in value.split(",") if t.strip()]
    elif isinstance(value, (list, tuple)):
        items = [str(t).strip() for t in value if str(t).strip()]
    else:
        raise ValueError(
            f"Baseline {name!r}: {label} must be a list or comma-separated string"
        )
    return items


def _normalize_fit_analyses(fit: dict, name: str) -> dict:
    """Normalize + validate the newer fitter-analysis config blocks.

    Produces canonical nested dicts at fit['bootstrap'], fit['uncertainty_diagnostics'],
    fit['split_sensitivity'], and fit['rg_weight_sensitivity'] while preserving
    backward compatibility with the legacy flat keys.

    Precedence rule for bootstrap: if fit['bootstrap'] is a JSON object it defines
    the bootstrap analysis (and the legacy flat fit['bootstrap_seed'] is folded in
    only when the object omits 'seed'); if it is a scalar (legacy form) it is the
    replicate count, fit['bootstrap_seed'] is the seed, and the analysis is enabled
    iff replicates > 0. File-dependent checks (custom split JSON existence and the
    Rg-weight data requirements) are performed later in run_suite preflight where
    the resolved paths and NPZ contents are available.
    """
    fit = dict(fit)

    # ---- bootstrap -------------------------------------------------------
    raw_bs = fit.get("bootstrap")
    legacy_seed = fit.get("bootstrap_seed")
    if isinstance(raw_bs, dict):
        bs = _merge(DEFAULT_FIT_BOOTSTRAP, raw_bs)
        seed = bs.get("seed", legacy_seed)
        if seed is None:
            seed = legacy_seed
        enabled_field = raw_bs.get("enabled", None)
    else:
        bs = dict(DEFAULT_FIT_BOOTSTRAP)
        bs["replicates"] = raw_bs if raw_bs is not None else 0
        seed = legacy_seed
        enabled_field = None
    try:
        replicates = int(bs.get("replicates", 0) or 0)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Baseline {name!r}: fit.bootstrap.replicates must be an integer"
        ) from exc
    if replicates < 0:
        raise ValueError(f"Baseline {name!r}: fit.bootstrap.replicates must be >= 0")
    enabled = bool(enabled_field) if enabled_field is not None else (replicates > 0)
    method = str(bs.get("method", "temperature"))
    try:
        confidence = float(bs.get("confidence", 0.95))
        corr_thr = float(bs.get("correlation_threshold", 0.9))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Baseline {name!r}: fit.bootstrap.confidence and "
            f"correlation_threshold must be numbers"
        ) from exc
    save_bands = bool(bs.get("save_prediction_bands", False))
    if seed is not None:
        try:
            seed = int(seed)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Baseline {name!r}: fit.bootstrap.seed must be an integer"
            ) from exc
    if enabled:
        if replicates < 1:
            raise ValueError(
                f"Baseline {name!r}: fit.bootstrap.replicates must be >= 1 when "
                f"bootstrap is enabled"
            )
        if not (0.0 < confidence < 1.0):
            raise ValueError(
                f"Baseline {name!r}: fit.bootstrap.confidence must be in (0, 1)"
            )
        if not (0.0 < corr_thr <= 1.0):
            raise ValueError(
                f"Baseline {name!r}: fit.bootstrap.correlation_threshold must be "
                f"in (0, 1]"
            )
        if method not in SUPPORTED_BOOTSTRAP_METHODS:
            raise ValueError(
                f"Baseline {name!r}: fit.bootstrap.method {method!r} unsupported; "
                f"choose from {list(SUPPORTED_BOOTSTRAP_METHODS)}"
            )
    fit["bootstrap"] = {
        "enabled": bool(enabled), "replicates": replicates, "seed": seed,
        "method": method, "confidence": confidence,
        "correlation_threshold": corr_thr,
        "save_prediction_bands": save_bands,
    }
    fit.pop("bootstrap_seed", None)

    # ---- uncertainty diagnostics ----------------------------------------
    raw_unc = fit.get("uncertainty_diagnostics")
    if isinstance(raw_unc, dict):
        unc_enabled = bool(raw_unc.get("enabled", False))
    else:
        unc_enabled = bool(raw_unc)
    fit["uncertainty_diagnostics"] = {"enabled": unc_enabled}

    # ---- validation-split sensitivity -----------------------------------
    raw_ss = fit.get("split_sensitivity")
    ss = _merge(DEFAULT_FIT_SPLIT, raw_ss if isinstance(raw_ss, dict) else None)
    if not isinstance(raw_ss, dict) and raw_ss is not None:
        ss["enabled"] = bool(raw_ss)
    ss_enabled = bool(ss.get("enabled", False))
    schemes = _as_str_list(ss.get("schemes"), "fit.split_sensitivity.schemes", name)
    if not schemes:
        schemes = list(SUPPORTED_SPLIT_SCHEMES)
    try:
        kfold_k = int(ss.get("kfold_k", 5))
        random_repeats = int(ss.get("random_repeats", 5))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Baseline {name!r}: fit.split_sensitivity.kfold_k and random_repeats "
            f"must be integers"
        ) from exc
    try:
        blocked_fraction = float(ss.get("blocked_fraction", 0.2))
        random_fraction = float(ss.get("random_fraction", 0.2))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Baseline {name!r}: fit.split_sensitivity.blocked_fraction and "
            f"random_fraction must be numbers"
        ) from exc
    ss_seed = ss.get("seed")
    if ss_seed is not None:
        try:
            ss_seed = int(ss_seed)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Baseline {name!r}: fit.split_sensitivity.seed must be an integer"
            ) from exc
    config_json = ss.get("config_json")
    if config_json is not None and not isinstance(config_json, str):
        raise ValueError(
            f"Baseline {name!r}: fit.split_sensitivity.config_json must be a path string"
        )
    if ss_enabled:
        if not schemes:
            raise ValueError(
                f"Baseline {name!r}: fit.split_sensitivity.schemes is empty"
            )
        bad = [s for s in schemes if s not in SUPPORTED_SPLIT_SCHEMES]
        if bad:
            raise ValueError(
                f"Baseline {name!r}: unsupported split scheme(s) {bad}; choose "
                f"from {list(SUPPORTED_SPLIT_SCHEMES)}"
            )
        if kfold_k < 2:
            raise ValueError(
                f"Baseline {name!r}: fit.split_sensitivity.kfold_k must be >= 2"
            )
        if not (0.0 < blocked_fraction < 1.0):
            raise ValueError(
                f"Baseline {name!r}: fit.split_sensitivity.blocked_fraction must "
                f"be in (0, 1)"
            )
        if not (0.0 < random_fraction < 1.0):
            raise ValueError(
                f"Baseline {name!r}: fit.split_sensitivity.random_fraction must "
                f"be in (0, 1)"
            )
        if random_repeats < 1:
            raise ValueError(
                f"Baseline {name!r}: fit.split_sensitivity.random_repeats must be >= 1"
            )
    fit["split_sensitivity"] = {
        "enabled": ss_enabled, "schemes": schemes, "config_json": config_json,
        "seed": ss_seed, "kfold_k": kfold_k, "blocked_fraction": blocked_fraction,
        "random_fraction": random_fraction, "random_repeats": random_repeats,
    }

    # ---- Rg-weight sensitivity ------------------------------------------
    raw_rgw = fit.get("rg_weight_sensitivity")
    rgw = _merge(DEFAULT_FIT_RGW, raw_rgw if isinstance(raw_rgw, dict) else None)
    if not isinstance(raw_rgw, dict) and raw_rgw is not None:
        rgw["enabled"] = bool(raw_rgw)
    rgw_enabled = bool(rgw.get("enabled", False))
    weights = _as_float_list(rgw.get("weights"),
                             "fit.rg_weight_sensitivity.weights", name)
    norm_diag = bool(rgw.get("normalization_diagnostics", False))
    if rgw_enabled:
        if not weights:
            raise ValueError(
                f"Baseline {name!r}: fit.rg_weight_sensitivity.weights must be a "
                f"non-empty list when enabled"
            )
        for w in weights:
            if not np.isfinite(w) or w < 0:
                raise ValueError(
                    f"Baseline {name!r}: fit.rg_weight_sensitivity.weights must "
                    f"all be finite and >= 0 (got {w!r})"
                )
        if not fit.get("fit_rg", False):
            raise ValueError(
                f"Baseline {name!r}: fit.rg_weight_sensitivity requires fit.fit_rg=true "
                f"(and a joint baseline with target Rg data)"
            )
    fit["rg_weight_sensitivity"] = {
        "enabled": rgw_enabled,
        "weights": sorted(set(weights)) if weights else [],
        "normalization_diagnostics": norm_diag,
    }
    return fit


def validate_config(cfg: dict) -> dict:
    """Validate required fields and normalize per-baseline overrides."""
    missing = [k for k in REQUIRED_TOP if k not in cfg]
    if missing:
        raise ValueError(f"Config missing required fields: {missing}")

    if not isinstance(cfg["models"], (list, tuple)):
        raise ValueError("Config 'models' must be a list of model names")
    models = list(cfg["models"])
    if not models:
        raise ValueError("Config 'models' must be non-empty")
    if any(not isinstance(m, str) or not m for m in models):
        raise ValueError("Every entry in config 'models' must be a non-empty string")
    if len(set(models)) != len(models):
        raise ValueError("Config 'models' contains duplicates")
    baselines = cfg["baselines"]
    if not isinstance(baselines, list) or not baselines:
        raise ValueError("Config 'baselines' must be a non-empty list")

    fit_base = _merge(DEFAULT_FIT, cfg.get("fit"))
    remd_base = _merge(DEFAULT_REMD, cfg.get("remd"))
    cmp_base = _merge(DEFAULT_COMPARISON, cfg.get("comparison"))

    seen_names = set()
    norm_baselines = []
    for b in baselines:
        if not isinstance(b, dict):
            raise ValueError(f"Each baseline must be a JSON object, got: {b!r}")
        if "name" not in b or "path" not in b:
            raise ValueError(f"Each baseline needs 'name' and 'path': {b}")
        name = sanitize_name(b["name"])
        if name in seen_names:
            raise ValueError(f"Duplicate baseline name after sanitizing: {name!r}")
        seen_names.add(name)
        fit = _merge(fit_base, b.get("fit"))
        remd = _merge(remd_base, b.get("remd"))
        comp = _merge(cmp_base, b.get("comparison"))
        if remd.get("N") is None:
            raise ValueError(f"Baseline {name!r}: remd.N is required")
        if "contact_offset" not in b:
            raise ValueError(f"Baseline {name!r}: 'contact_offset' is required")
        if fit.get("loss") not in ("js", "kl"):
            raise ValueError(f"Baseline {name!r}: fit.loss must be 'js' or 'kl'")
        try:
            n_restarts = int(fit.get("n_restarts", 0))
            fit_seed = int(fit.get("seed", 123))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Baseline {name!r}: fit.n_restarts and fit.seed must be integers"
            ) from exc
        if n_restarts < 1:
            raise ValueError(f"Baseline {name!r}: fit.n_restarts must be >= 1")
        fit["n_restarts"] = n_restarts
        fit["seed"] = fit_seed
        if fit.get("holdout_every") is not None:
            fit["holdout_every"] = int(fit["holdout_every"])
            if fit["holdout_every"] < 1:
                raise ValueError(
                    f"Baseline {name!r}: fit.holdout_every must be >= 1"
                )
        fit_rg_weight = float(fit.get("rg_weight", 1.0))
        if not np.isfinite(fit_rg_weight) or fit_rg_weight < 0:
            raise ValueError(f"Baseline {name!r}: fit.rg_weight must be finite and >= 0")
        fit["rg_weight"] = fit_rg_weight
        # Normalize and validate the newer fitter analyses (bootstrap,
        # uncertainty diagnostics, split sensitivity, Rg-weight sensitivity).
        fit = _normalize_fit_analyses(fit, name)

        for field in ("N", "steps_per_swap", "n_cycles", "rg_bins", "n_workers"):
            try:
                value = int(remd[field])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Baseline {name!r}: remd.{field} must be an integer"
                ) from exc
            lower = 3 if field == "N" else 1
            if value < lower:
                raise ValueError(
                    f"Baseline {name!r}: remd.{field} must be >= {lower}"
                )
            remd[field] = value
        burnin = float(remd.get("burnin_frac", 0.7))
        if not np.isfinite(burnin) or not (0.0 <= burnin < 1.0):
            raise ValueError(
                f"Baseline {name!r}: remd.burnin_frac must be finite and in [0, 1)"
            )
        remd["burnin_frac"] = burnin
        seeds = remd.get("seeds")
        if not isinstance(seeds, (list, tuple)) or not seeds:
            raise ValueError(f"Baseline {name!r}: remd.seeds must be a non-empty list")
        try:
            seeds = [int(s) for s in seeds]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Baseline {name!r}: every REMD seed must be an integer") from exc
        if len(set(seeds)) != len(seeds):
            raise ValueError(f"Baseline {name!r}: remd.seeds contains duplicates")
        remd["seeds"] = seeds

        diag = _merge(DEFAULT_REMD_DIAGNOSTICS, remd.get("diagnostics"))
        diag["enabled"] = bool(diag.get("enabled", False))
        diag["trajectories"] = bool(diag.get("trajectories", False))
        try:
            diag["n_blocks"] = int(diag["n_blocks"])
            diag["min_round_trips"] = int(diag["min_round_trips"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Baseline {name!r}: remd.diagnostics.n_blocks and "
                f"min_round_trips must be integers"
            ) from exc
        if diag["n_blocks"] < 1:
            raise ValueError(f"Baseline {name!r}: remd.diagnostics.n_blocks must be >= 1")
        if diag["min_round_trips"] < 0:
            raise ValueError(
                f"Baseline {name!r}: remd.diagnostics.min_round_trips must be >= 0"
            )
        for fkey in ("min_temp_coverage", "min_ess", "max_drift",
                     "min_swap_rate", "rhat_threshold"):
            fv = float(diag[fkey])
            if not np.isfinite(fv) or fv < 0:
                raise ValueError(
                    f"Baseline {name!r}: remd.diagnostics.{fkey} must be finite and >= 0"
                )
            diag[fkey] = fv
        remd["diagnostics"] = diag

        contact_offset = float(b["contact_offset"])
        rg_scale = float(b.get("rg_scale", 1.0))
        if not np.isfinite(contact_offset):
            raise ValueError(f"Baseline {name!r}: contact_offset must be finite")
        if not np.isfinite(rg_scale) or rg_scale <= 0:
            raise ValueError(f"Baseline {name!r}: rg_scale must be finite and positive")
        temp_tol = float(comp.get("temperature_tolerance", 1e-10))
        rg_cmp_weight = float(comp.get("rg_weight", 1.0))
        if not np.isfinite(temp_tol) or temp_tol < 0:
            raise ValueError(
                f"Baseline {name!r}: comparison.temperature_tolerance must be finite and >= 0"
            )
        if not np.isfinite(rg_cmp_weight) or rg_cmp_weight < 0:
            raise ValueError(
                f"Baseline {name!r}: comparison.rg_weight must be finite and >= 0"
            )
        comp["temperature_tolerance"] = temp_tol
        comp["rg_weight"] = rg_cmp_weight

        stats = _merge(DEFAULT_COMPARISON_STATISTICS, comp.get("statistics"))
        stats["enabled"] = bool(stats.get("enabled", False))
        try:
            stats["bootstrap_replicates"] = int(stats["bootstrap_replicates"])
            stats["seed"] = int(stats["seed"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Baseline {name!r}: comparison.statistics.bootstrap_replicates "
                f"and seed must be integers"
            ) from exc
        if stats["bootstrap_replicates"] < 1:
            raise ValueError(
                f"Baseline {name!r}: comparison.statistics.bootstrap_replicates "
                f"must be >= 1"
            )
        for fkey in ("alpha", "practical_equivalence_epsilon"):
            fv = float(stats[fkey])
            if not np.isfinite(fv) or fv < 0:
                raise ValueError(
                    f"Baseline {name!r}: comparison.statistics.{fkey} must be "
                    f"finite and >= 0"
                )
            stats[fkey] = fv
        if not (0.0 < stats["alpha"] < 1.0):
            raise ValueError(
                f"Baseline {name!r}: comparison.statistics.alpha must be in (0, 1)"
            )
        if str(stats["multiple_testing"]).lower() not in ("holm", "none"):
            raise ValueError(
                f"Baseline {name!r}: comparison.statistics.multiple_testing must "
                f"be 'holm' or 'none'"
            )
        stats["multiple_testing"] = str(stats["multiple_testing"]).lower()
        comp["statistics"] = stats
        # Exactly-one split option (None allowed = no split).
        n_split = sum(
            1 for k in ("train_indices", "holdout_indices", "holdout_every")
            if fit.get(k) is not None
        )
        if n_split > 1:
            raise ValueError(
                f"Baseline {name!r}: choose at most one of train_indices, "
                f"holdout_indices, holdout_every"
            )
        norm_baselines.append({
            "name": name,
            "raw_name": b["name"],
            "path": b["path"],
            "contact_offset": contact_offset,
            "rg_scale": rg_scale,
            "fit": fit,
            "remd": remd,
            "comparison": comp,
        })

    cfg = dict(cfg)
    cfg["models"] = models
    cfg["baselines"] = norm_baselines
    return cfg


# ---------------------------------------------------------------------------
# Module import + model-contract preflight (Part 6)
# ---------------------------------------------------------------------------

def import_module_from_path(mod_name: str, path: str):
    """Import a module by file path; register in sys.modules before exec."""
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create import spec for {path!r}")
    module = importlib.util.module_from_spec(spec)
    # Insert before exec_module so dataclasses / pickling resolve the module.
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


def check_model_contracts(fit_mod, remd_mod, log: Logger) -> dict:
    """Compare API version, model names, param_names, and numeric b(T)."""
    fit_contract = fit_mod.get_model_contract()
    remd_contract = remd_mod.get_model_contract()

    if fit_contract["model_api_version"] != remd_contract["model_api_version"]:
        raise ValueError(
            f"MODEL_API_VERSION mismatch: fitter="
            f"{fit_contract['model_api_version']} remd="
            f"{remd_contract['model_api_version']}"
        )

    fit_models = set(fit_contract["models"])
    remd_models = set(remd_contract["models"])
    if fit_models != remd_models:
        raise ValueError(
            f"Model-name sets differ: fitter-only={fit_models - remd_models}, "
            f"remd-only={remd_models - fit_models}"
        )

    for name in sorted(fit_models):
        fn = fit_contract["models"][name]["param_names"]
        rn = remd_contract["models"][name]["param_names"]
        if list(fn) != list(rn):
            raise ValueError(
                f"param_names differ for {name!r}: fitter={fn} remd={rn}"
            )
        # potential_kind and quadratic_normalization define WHAT is being
        # reweighted, so a disagreement is as fatal as a parameter-order one:
        # the two scripts would be exponentiating different potentials.
        for key in ("potential_kind", "quadratic_normalization"):
            fv = fit_contract["models"][name][key]
            rv = remd_contract["models"][name][key]
            if fv != rv:
                raise ValueError(
                    f"{key} differs for {name!r}: fitter={fv!r} remd={rv!r}"
                )

    # Numeric b(T) and kappa(T) equality using the fitter's x0 for each model.
    fit_reg = fit_mod.MODEL_REGISTRY
    remd_reg = remd_mod.MODEL_REGISTRY
    for name in sorted(fit_models):
        x0 = list(fit_reg[name]["x0"])
        for coeff, key in (("b", "raw_b_fn"), ("kappa", "raw_q_fn")):
            ff = fit_reg[name][key]
            rf = remd_reg[name][key]
            for T in BFN_CHECK_T:
                vf = float(ff(x0, T, BFN_CHECK_TREF, BFN_CHECK_TSCALE))
                vr = float(rf(x0, T, BFN_CHECK_TREF, BFN_CHECK_TSCALE))
                if not np.isclose(vf, vr, rtol=1e-12, atol=1e-12):
                    raise ValueError(
                        f"{coeff}(T) mismatch for {name!r} at T={T}: "
                        f"fitter={vf!r} remd={vr!r}"
                    )
    n_quadratic = sum(
        1 for name in fit_models
        if fit_contract["models"][name]["potential_kind"] != "linear"
    )
    log(f"Preflight: model contract OK (api v{fit_contract['model_api_version']}, "
        f"{len(fit_models)} models, {n_quadratic} contact-quadratic, "
        f"numeric b(T) and kappa(T) equal).")
    return fit_contract


# ---------------------------------------------------------------------------
# Target / baseline validation (Part 6)
# ---------------------------------------------------------------------------

def _npz_keys(path: str) -> set:
    with np.load(path, allow_pickle=True) as d:
        return set(d.files)


def validate_target_npz(path: str) -> np.ndarray:
    """Validate the target NPZ and return its exact temperature ladder."""
    if not Path(path).exists():
        raise FileNotFoundError(f"target_remd not found: {path}")
    keys = _npz_keys(path)
    if not ({"temps", "Ts"} & keys):
        raise ValueError(f"target {path!r} lacks a temperature key (temps/Ts)")
    if "ct_centers" not in keys or "ct_hists" not in keys:
        raise ValueError(f"target {path!r} must contain ct_centers and ct_hists")
    with np.load(path) as d:
        temps = np.asarray(d["temps" if "temps" in d else "Ts"], dtype=float)
        ct_centers = np.asarray(d["ct_centers"], dtype=float)
        ct_hists = np.asarray(d["ct_hists"], dtype=float)
    if temps.ndim != 1 or temps.size < 2:
        raise ValueError(f"target {path!r} temperature ladder must be 1D with >=2 values")
    if not np.all(np.isfinite(temps)) or not np.all(temps > 0):
        raise ValueError(f"target {path!r} temperatures must be finite and positive")
    if not np.all(np.diff(temps) > 0):
        raise ValueError(
            f"target {path!r} temperatures must be strictly increasing with no duplicates"
        )
    if ct_centers.ndim != 1 or ct_centers.size < 2:
        raise ValueError(f"target {path!r} ct_centers must be 1D with >=2 values")
    if not np.all(np.isfinite(ct_centers)) or not np.all(np.diff(ct_centers) > 0):
        raise ValueError(f"target {path!r} ct_centers must be finite and increasing")
    if ct_hists.shape != (temps.size, ct_centers.size):
        raise ValueError(
            f"target {path!r} ct_hists shape {ct_hists.shape} does not match "
            f"({temps.size}, {ct_centers.size})"
        )
    if not np.all(np.isfinite(ct_hists)) or np.any(ct_hists < 0):
        raise ValueError(f"target {path!r} ct_hists must be finite and nonnegative")
    if np.any(ct_hists.sum(axis=1) <= 0):
        raise ValueError(f"target {path!r} every ct_hists row must contain positive mass")
    return temps


def target_has_rg(path: str) -> bool:
    keys = _npz_keys(path)
    return bool(({"rg_centers", "Rg_centers"} & keys)
                and ({"rg_hists", "Rg_hists"} & keys))


def baseline_keyset(path: str) -> str:
    """Return which supported baseline key-set a file provides."""
    keys = _npz_keys(path)
    if {"c_vals", "c_prob"} <= keys:
        return "c_vals,c_prob"
    if {"c_vals", "Pc"} <= keys:
        return "c_vals,Pc"
    if {"c_edges", "rg_edges", "crg_prob"} <= keys:
        return "c_edges,rg_edges,crg_prob"
    if {"ct_centers", "ct_hists"} <= keys:
        return "ct_centers,ct_hists"
    raise ValueError(
        f"baseline {path!r} has none of the supported key-sets "
        f"(found: {sorted(keys)})"
    )


def baseline_is_joint(path: str) -> bool:
    return {"c_edges", "rg_edges", "crg_prob"} <= _npz_keys(path)


# ---------------------------------------------------------------------------
# Standardized comparison helpers (Part 11)
# ---------------------------------------------------------------------------

def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence (base 2, in [0, 1]) between two mass vectors."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    ps, qs = p.sum(), q.sum()
    if ps <= 0 or qs <= 0:
        return float("nan")
    p = p / ps
    q = q / qs
    m = 0.5 * (p + q)

    def _kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def union_int_grid(*int_arrays) -> np.ndarray:
    """Sorted integer grid spanning the union of all provided integer supports."""
    vals = set()
    for arr in int_arrays:
        for v in np.asarray(arr).ravel():
            vals.add(int(round(float(v))))
    if not vals:
        return np.array([], dtype=int)
    return np.arange(min(vals), max(vals) + 1, dtype=int)


def align_contact_row(c_vals, row, grid: np.ndarray) -> np.ndarray:
    """Map a contact mass row onto a common integer grid by zero-padding.

    Mass outside the grid would be dropped, but callers build `grid` from the
    union so that never happens (no truncate-and-renormalize).
    """
    out = np.zeros(len(grid), dtype=float)
    index = {int(v): k for k, v in enumerate(grid)}
    for v, p in zip(np.asarray(c_vals).ravel(), np.asarray(row).ravel()):
        if not np.isfinite(p):
            continue
        iv = int(round(float(v)))
        if iv in index:
            out[index[iv]] += float(p)
    return out


def rebin_mass_piecewise(src_edges, src_mass, dst_edges) -> np.ndarray:
    """Rebin probability mass to a new grid by piecewise-constant overlap.

    Distributes each source bin's mass to destination bins in proportion to
    their overlap length.  Conserves total mass provided dst spans src support.
    """
    src_edges = np.asarray(src_edges, dtype=float)
    src_mass = np.asarray(src_mass, dtype=float)
    dst_edges = np.asarray(dst_edges, dtype=float)
    dst = np.zeros(len(dst_edges) - 1, dtype=float)
    for i in range(len(src_mass)):
        m = src_mass[i]
        if not np.isfinite(m) or m == 0.0:
            continue
        lo, hi = src_edges[i], src_edges[i + 1]
        width = hi - lo
        if width <= 0:
            continue
        for j in range(len(dst)):
            ov = min(hi, dst_edges[j + 1]) - max(lo, dst_edges[j])
            if ov > 0:
                dst[j] += m * ov / width
    return dst


# ---------------------------------------------------------------------------
# Subprocess execution + provenance (Part 10)
# ---------------------------------------------------------------------------

def run_subprocess(command: list[str], log_path: Path) -> dict:
    """Run a list-form command, tee output to log_path, return provenance."""
    start = time.perf_counter()
    started_at = now_iso()
    with open(log_path, "w", encoding="utf-8") as log_file:
        log_file.write(f"# {started_at}\n# {' '.join(command)}\n\n")
        log_file.flush()
        proc = subprocess.run(
            command,
            check=False,
            shell=False,
            text=True,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
    elapsed = time.perf_counter() - start
    return {
        "command": list(command),
        "returncode": int(proc.returncode),
        "started_at": started_at,
        "ended_at": now_iso(),
        "elapsed_seconds": float(elapsed),
        "log": str(log_path),
    }


def make_job_signature(kind: str, command: list[str], inputs: dict[str, str]) -> dict:
    """Fingerprint a job by exact command plus hashes of material inputs."""
    hashes = {}
    for label, path in inputs.items():
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Cannot fingerprint missing {label}: {path}")
        hashes[label] = {"path": str(p.resolve()), "sha256": sha256_file(p)}
    return {
        "kind": kind,
        "command": [str(x) for x in command],
        "inputs": hashes,
    }


def signature_matches(path: Path, expected: dict) -> bool:
    try:
        actual = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return actual == expected


def write_signature(path: Path, signature: dict) -> None:
    path.write_text(
        json.dumps(_json_safe(signature), indent=2, allow_nan=False),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Fit job (Part 8)
# ---------------------------------------------------------------------------

def build_fit_command(cfg, baseline, model, fit_dir: Path) -> list[str]:
    fit = baseline["fit"]
    bootstrap = fit.get("bootstrap") or {}
    cmd = [
        sys.executable, cfg["fit_script"],
        "--remd", cfg["target_remd"],
        "--baseline", baseline["path"],
        "--contact_offset", str(baseline["contact_offset"]),
        "--model", model,
        "--loss", str(fit["loss"]),
        "--outdir", str(fit_dir),
        "--rg-scale", str(baseline["rg_scale"]),
        "--rg-weight", str(fit["rg_weight"]),
        "--n_restarts", str(fit["n_restarts"]),
        "--seed", str(fit["seed"]),
        # --bootstrap is always forwarded (0 disables in the fitter) so the
        # legacy contract is preserved; the extra bootstrap flags below are only
        # appended when the analysis is actually enabled.
        "--bootstrap", str(int(bootstrap.get("replicates", 0) or 0)),
    ]
    # Exactly one split option, never null.
    if fit.get("train_indices") is not None:
        cmd += ["--train-indices", _csv_indices(fit["train_indices"])]
    elif fit.get("holdout_indices") is not None:
        cmd += ["--holdout-indices", _csv_indices(fit["holdout_indices"])]
    elif fit.get("holdout_every") is not None:
        cmd += ["--holdout-every", str(int(fit["holdout_every"]))]
    cmd += ["--fit-rg"] if fit["fit_rg"] else ["--no-fit-rg"]
    if not fit.get("plots", False):
        cmd += ["--no-plots"]

    # ---- bootstrap uncertainty / identifiability -------------------------
    if bootstrap.get("enabled"):
        cmd += ["--bootstrap-method", str(bootstrap.get("method", "temperature"))]
        cmd += ["--bootstrap-confidence", str(bootstrap.get("confidence", 0.95))]
        cmd += ["--bootstrap-correlation-threshold",
                str(bootstrap.get("correlation_threshold", 0.9))]
        if bootstrap.get("seed") is not None:
            cmd += ["--bootstrap-seed", str(int(bootstrap["seed"]))]
        if bootstrap.get("save_prediction_bands"):
            cmd += ["--bootstrap-save-prediction-bands"]

    # ---- optimizer-restart / local-curvature diagnostics -----------------
    if (fit.get("uncertainty_diagnostics") or {}).get("enabled"):
        cmd += ["--uncertainty-diagnostics"]

    # ---- validation-split sensitivity ------------------------------------
    ss = fit.get("split_sensitivity") or {}
    if ss.get("enabled"):
        cmd += ["--split-sensitivity"]
        schemes = ss.get("schemes") or list(SUPPORTED_SPLIT_SCHEMES)
        cmd += ["--split-schemes", ",".join(str(s) for s in schemes)]
        cmd += ["--split-kfold-k", str(int(ss.get("kfold_k", 5)))]
        cmd += ["--split-blocked-fraction", str(ss.get("blocked_fraction", 0.2))]
        cmd += ["--split-random-fraction", str(ss.get("random_fraction", 0.2))]
        cmd += ["--split-random-repeats", str(int(ss.get("random_repeats", 5)))]
        if ss.get("seed") is not None:
            cmd += ["--split-seed", str(int(ss["seed"]))]
        if ss.get("config_json"):
            cmd += ["--split-config-json", str(ss["config_json"])]

    # ---- Rg-weight sensitivity / Pareto analysis -------------------------
    rgw = fit.get("rg_weight_sensitivity") or {}
    if rgw.get("enabled"):
        weights = rgw.get("weights") or []
        cmd += ["--rg-weight-grid",
                ",".join(_num_str(w) for w in weights)]
        if rgw.get("normalization_diagnostics"):
            cmd += ["--rg-weight-normalization-diagnostics"]
    return cmd


def _num_str(value) -> str:
    """Render a number compactly without a trailing '.0' for whole values."""
    f = float(value)
    if f.is_integer():
        return str(int(f))
    return repr(f)


def _csv_indices(value) -> str:
    if isinstance(value, (list, tuple)):
        return ",".join(str(int(v)) for v in value)
    return str(value)


def validate_fit_outputs(fit_dir: Path, model: str, fit_cfg: dict | None = None) -> dict:
    """Validate fit completion; return a status dict (status in ok/failed).

    The required output set is feature-aware: every enabled analysis must have
    produced its documented completion files (an enabled analysis whose outputs
    are missing fails the model rather than being silently ignored).
    """
    required = (expected_fit_outputs(fit_cfg) if fit_cfg is not None
                else FIT_COMPLETION_FILES)
    for fname in required:
        if not (fit_dir / fname).exists():
            return {"status": "failed", "error": f"missing {fname}"}
    try:
        summary = json.loads((fit_dir / "fit_summary.json").read_text())
    except Exception as exc:
        return {"status": "failed", "error": f"invalid fit_summary.json: {exc}"}
    if summary.get("model") != model:
        return {"status": "failed",
                "error": f"summary model {summary.get('model')} != {model}"}
    param_names = summary.get("param_names")
    params = summary.get("params", {})
    if not isinstance(param_names, list) or not param_names:
        return {"status": "failed", "error": "missing/invalid param_names"}
    if not isinstance(params, dict) or set(params) != set(param_names):
        return {"status": "failed", "error": "params do not match param_names"}
    if not all(np.isfinite(float(params[n])) for n in param_names):
        return {"status": "failed", "error": "non-finite fit parameters"}
    try:
        Tref, Tscale = float(summary["Tref"]), float(summary["Tscale"])
    except (KeyError, TypeError, ValueError) as exc:
        return {"status": "failed", "error": f"invalid Tref/Tscale: {exc}"}
    if not np.isfinite(Tref) or not np.isfinite(Tscale) or Tscale <= 0:
        return {"status": "failed", "error": "invalid Tref/Tscale"}
    if summary.get("optimization_success") is False:
        return {"status": "failed", "error": "optimizer reported failure"}
    try:
        with np.load(fit_dir / "fit_results.npz", allow_pickle=True) as d:
            npz_model = str(d["model_name"])
            npz_names = [str(x) for x in np.asarray(d["param_names"]).tolist()]
            npz_params = np.asarray(d["params"], dtype=float)
        expected = np.array([params[n] for n in param_names], dtype=float)
        if npz_model != model or npz_names != param_names:
            return {"status": "failed", "error": "fit_results metadata mismatch"}
        if npz_params.shape != expected.shape or not np.allclose(
            npz_params, expected, rtol=1e-10, atol=1e-12
        ):
            return {"status": "failed", "error": "fit_results params mismatch"}
    except Exception as exc:
        return {"status": "failed", "error": f"invalid fit_results.npz: {exc}"}
    return {"status": "ok", "summary": summary}


# ---------------------------------------------------------------------------
# REMD job (Part 9)
# ---------------------------------------------------------------------------

def build_remd_command(cfg, baseline, fit_dir: Path, seed: int,
                       seed_dir: Path) -> list[str]:
    remd = baseline["remd"]
    cmd = [
        sys.executable, cfg["remd_script"],
        "--N", str(int(remd["N"])),
        "--temps-from-npz", cfg["target_remd"],
        "--steps-per-swap", str(int(remd["steps_per_swap"])),
        "--n-cycles", str(int(remd["n_cycles"])),
        "--fit-summary-json", str(fit_dir / "fit_summary.json"),
        "--seed", str(int(seed)),
        "--out-prefix", str(seed_dir / "run"),
        "--rg-bins", str(int(remd["rg_bins"])),
        "--burnin-frac", str(remd["burnin_frac"]),
        "--rg-scale", str(baseline["rg_scale"]),
        "--n-workers", str(int(remd["n_workers"])),
    ]
    if remd.get("timing", False):
        cmd += ["--timing"]
    if not remd.get("plots", False):
        cmd += ["--no-plots"]
    diag = remd.get("diagnostics") or {}
    if diag.get("enabled", False):
        cmd += [
            "--diagnostics",
            "--diag-n-blocks", str(int(diag["n_blocks"])),
            "--diag-min-round-trips", str(int(diag["min_round_trips"])),
            "--diag-min-temp-coverage", str(diag["min_temp_coverage"]),
            "--diag-min-ess", str(diag["min_ess"]),
            "--diag-max-drift", str(diag["max_drift"]),
            "--diag-min-swap-rate", str(diag["min_swap_rate"]),
        ]
        if diag.get("trajectories", False):
            cmd += ["--diagnostic-trajectories"]
    return cmd


def validate_remd_outputs(seed_dir: Path, fit_summary: dict,
                          target_temps: np.ndarray, tol: float,
                          diag_cfg: dict | None = None) -> dict:
    for fname in REMD_COMPLETION_FILES:
        if not (seed_dir / fname).exists():
            return {"status": "failed", "error": f"missing {fname}"}
    diag_cfg = diag_cfg or {}
    if diag_cfg.get("enabled", False):
        for fname in REMD_DIAGNOSTIC_FILES:
            if not (seed_dir / fname).exists():
                return {"status": "failed",
                        "error": f"missing diagnostic output {fname}"}
        try:
            diag = json.loads((seed_dir / "run_diagnostics.json").read_text())
        except Exception as exc:
            return {"status": "failed",
                    "error": f"invalid run_diagnostics.json: {exc}"}
        if not isinstance(diag, dict) or "summary" not in diag \
                or "warnings" not in diag:
            return {"status": "failed",
                    "error": "run_diagnostics.json missing summary/warnings"}
        if diag_cfg.get("trajectories", False) and not (
            seed_dir / REMD_DIAGNOSTIC_TRAJECTORY_FILE
        ).exists():
            return {"status": "failed",
                    "error": f"missing {REMD_DIAGNOSTIC_TRAJECTORY_FILE}"}
    try:
        run_summary = json.loads((seed_dir / "run_run_summary.json").read_text())
    except Exception as exc:
        return {"status": "failed", "error": f"invalid run_run_summary.json: {exc}"}
    if not isinstance(run_summary, dict) or run_summary.get("model") != fit_summary["model"]:
        return {"status": "failed", "error": "run summary model mismatch"}
    with np.load(seed_dir / "run_distributions.npz", allow_pickle=True) as d:
        model_name = str(d["model_name"])
        if model_name != fit_summary["model"]:
            return {"status": "failed",
                    "error": f"model_name {model_name} != {fit_summary['model']}"}
        params = np.asarray(d["model_params"], dtype=float)
        summ_params = np.array(
            [fit_summary["params"][n] for n in fit_summary["param_names"]],
            dtype=float,
        )
        if not np.allclose(params, summ_params, rtol=1e-8, atol=1e-10):
            return {"status": "failed", "error": "params != fit_summary"}
        summary_ref = (
            fit_summary.get("T0", fit_summary["Tref"])
            if fit_summary.get("model") == "heat_capacity"
            else fit_summary["Tref"]
        )
        if not np.isclose(float(d["Tref"]), float(summary_ref),
                          rtol=1e-10, atol=1e-12):
            return {"status": "failed", "error": "Tref != fit_summary"}
        if not np.isclose(float(d["Tscale"]), float(fit_summary["Tscale"]),
                          rtol=1e-10, atol=1e-12):
            return {"status": "failed", "error": "Tscale != fit_summary"}
        temps = np.asarray(d["temps" if "temps" in d else "Ts"], dtype=float)
        if temps.shape != target_temps.shape or not np.allclose(
            temps, target_temps, rtol=0.0, atol=tol
        ):
            return {"status": "failed", "error": "temps != target"}
        c_vals = np.asarray(d["c_vals"], dtype=float)
        Pc = np.asarray(d["Pc"], dtype=float)
        rg_edges = np.asarray(d["rg_edges"], dtype=float)
        rg_centers = np.asarray(d["rg_centers"], dtype=float)
        Prg = np.asarray(d["Prg"], dtype=float)
    if (
        c_vals.ndim != 1
        or not np.all(np.isfinite(c_vals))
        or not np.allclose(c_vals, np.rint(c_vals), rtol=0.0, atol=1e-10)
        or np.unique(np.rint(c_vals).astype(int)).size != c_vals.size
        or (c_vals.size > 1 and not np.all(np.diff(c_vals) > 0))
    ):
        return {"status": "failed", "error": "invalid c_vals support"}
    if Pc.shape != (target_temps.size, c_vals.size):
        return {"status": "failed", "error": f"Pc has invalid shape {Pc.shape}"}
    if (
        rg_centers.ndim != 1
        or rg_edges.ndim != 1
        or rg_edges.size != rg_centers.size + 1
        or not np.all(np.isfinite(rg_centers))
        or (rg_centers.size > 1 and not np.all(np.diff(rg_centers) > 0))
        or not np.all(np.isfinite(rg_edges))
        or not np.all(np.diff(rg_edges) > 0)
        or Prg.shape != (target_temps.size, rg_centers.size)
    ):
        return {"status": "failed", "error": "invalid Rg grid/distribution shape"}
    for tag, arr in (("Pc", Pc), ("Prg", Prg)):
        if arr.ndim != 2 or arr.shape[0] != target_temps.size:
            return {"status": "failed", "error": f"{tag} has invalid shape {arr.shape}"}
        for i, row in enumerate(arr):
            if not np.all(np.isfinite(row)):
                return {"status": "failed", "error": f"{tag}[{i}] contains non-finite mass"}
            finite = row[np.isfinite(row)]
            if finite.size == 0:
                return {"status": "failed", "error": f"{tag}[{i}] has no finite mass"}
            if np.any(finite < 0):
                return {"status": "failed", "error": f"{tag}[{i}] has negative mass"}
            if abs(float(finite.sum()) - 1.0) > 1e-6:
                return {"status": "failed",
                        "error": f"{tag}[{i}] not normalized"}
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Comparison (Part 11): A=target-vs-fit, B=target-vs-REMD, C=REMD-vs-fit
# ---------------------------------------------------------------------------

def _safe_mean(values):
    vals = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def _safe_std(values):
    vals = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.std(vals)) if vals else float("nan")


def _safe_min(values):
    vals = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.min(vals)) if vals else float("nan")


def _safe_max(values):
    vals = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.max(vals)) if vals else float("nan")


def centers_to_edges(centers: np.ndarray) -> np.ndarray:
    """Midpoint bin edges for monotone centers (ends extrapolated)."""
    centers = np.asarray(centers, dtype=float)
    if centers.size == 1:
        return np.array([centers[0] - 0.5, centers[0] + 0.5])
    mids = 0.5 * (centers[:-1] + centers[1:])
    first = centers[0] - (mids[0] - centers[0])
    last = centers[-1] + (centers[-1] - mids[-1])
    return np.concatenate([[first], mids, [last]])


def pdf_to_mass(pdf_row: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Convert a density row on `centers` to probability mass (sums to 1)."""
    widths = np.diff(centers_to_edges(centers))
    mass = np.asarray(pdf_row, dtype=float) * widths
    s = mass.sum()
    return mass / s if s > 0 else mass


def _common_rg_grid(edge_arrays, target_width: float) -> np.ndarray:
    """Uniform Rg edges (target spacing) spanning the union of all supports."""
    lo = min(float(e[0]) for e in edge_arrays)
    hi = max(float(e[-1]) for e in edge_arrays)
    w = float(target_width)
    if not np.isfinite(w) or w <= 0:
        w = (hi - lo) / 50.0 if hi > lo else 1.0
    n = int(np.ceil((hi - lo) / w)) + 1
    edges = lo + w * np.arange(n + 1)
    if edges[-1] < hi:  # guarantee coverage of the union
        edges = np.append(edges, edges[-1] + w)
    return edges


def _split_indices(fit_results_path: Path, n_temps: int):
    with np.load(fit_results_path, allow_pickle=True) as d:
        train = (np.asarray(d["train_indices"], dtype=int)
                 if "train_indices" in d else np.arange(n_temps))
        val = (np.asarray(d["validation_indices"], dtype=int)
               if "validation_indices" in d else np.array([], dtype=int))
    return train, val


def _wass(a_centers, a_mass, b_mass):
    """1-D Wasserstein between two mass vectors on a shared support."""
    if _wasserstein is None:
        return float("nan")
    sa, sb = np.sum(a_mass), np.sum(b_mass)
    if sa <= 0 or sb <= 0:
        return float("nan")
    return float(_wasserstein(a_centers, a_centers,
                              np.asarray(a_mass) / sa, np.asarray(b_mass) / sb))


def compare_seed(fit_dir: Path, seed_dir: Path, target_path: str,
                 include_rg: bool) -> dict:
    """Per-temperature A/B/C JS and means for contacts and (optionally) Rg."""
    with np.load(fit_dir / "fit_results.npz", allow_pickle=True) as f:
        m_centers = np.round(np.asarray(f["m_centers"], dtype=float)).astype(int)
        p_obs = np.asarray(f["p_obs_mass"], dtype=float)   # target on shifted grid
        p_mod = np.asarray(f["p_mod_mass"], dtype=float)   # analytic fit prediction
        rg_fit_centers = (np.asarray(f["rg_centers"], dtype=float)
                          if "rg_centers" in f else None)
        rg_mod = (np.asarray(f["rg_mod_mass"], dtype=float)
                  if "rg_mod_mass" in f else None)
    with np.load(seed_dir / "run_distributions.npz", allow_pickle=True) as d:
        c_vals = np.asarray(d["c_vals"], dtype=float)
        Pc = np.asarray(d["Pc"], dtype=float)
        rg_remd_centers = np.asarray(d["rg_centers"], dtype=float)
        rg_remd_edges = np.asarray(d["rg_edges"], dtype=float)
        Prg = np.asarray(d["Prg"], dtype=float)

    n_temps = p_obs.shape[0]
    if (
        p_obs.ndim != 2 or p_mod.shape != p_obs.shape
        or p_obs.shape[1] != m_centers.size
    ):
        raise ValueError(
            f"Fit contact arrays have incompatible shapes: p_obs={p_obs.shape}, "
            f"p_mod={p_mod.shape}, m_centers={m_centers.shape}"
        )
    for label, arr in (("p_obs_mass", p_obs), ("p_mod_mass", p_mod)):
        if not np.all(np.isfinite(arr)) or np.any(arr < 0):
            raise ValueError(f"{label} must be finite and nonnegative")
        if np.any(np.abs(arr.sum(axis=1) - 1.0) > 1e-6):
            raise ValueError(f"{label} rows are not normalized")
    if c_vals.ndim != 1 or Pc.shape != (n_temps, c_vals.size):
        raise ValueError(
            f"REMD contact arrays have incompatible shapes: c_vals={c_vals.shape}, "
            f"Pc={Pc.shape}, n_temps={n_temps}"
        )
    grid = union_int_grid(m_centers, c_vals)
    out = {
        "n_temps": n_temps,
        "contact_js_tf": np.full(n_temps, np.nan),
        "contact_js_tr": np.full(n_temps, np.nan),
        "contact_js_fr": np.full(n_temps, np.nan),
        "target_mean_contacts": np.full(n_temps, np.nan),
        "fit_mean_contacts": np.full(n_temps, np.nan),
        "remd_mean_contacts": np.full(n_temps, np.nan),
        "contact_wass_tr": np.full(n_temps, np.nan),
        "rg_js_tf": np.full(n_temps, np.nan),
        "rg_js_tr": np.full(n_temps, np.nan),
        "rg_js_fr": np.full(n_temps, np.nan),
        "target_mean_rg": np.full(n_temps, np.nan),
        "fit_mean_rg": np.full(n_temps, np.nan),
        "remd_mean_rg": np.full(n_temps, np.nan),
        "rg_wass_tr": np.full(n_temps, np.nan),
        "rg_available": False,
    }

    gridf = grid.astype(float)
    for t in range(n_temps):
        tgt = align_contact_row(m_centers, p_obs[t], grid)
        fit = align_contact_row(m_centers, p_mod[t], grid)
        rem = align_contact_row(c_vals, Pc[t], grid)
        out["contact_js_tf"][t] = js_divergence(tgt, fit)
        out["contact_js_tr"][t] = js_divergence(tgt, rem)
        out["contact_js_fr"][t] = js_divergence(fit, rem)
        if tgt.sum() > 0:
            out["target_mean_contacts"][t] = float(np.sum(gridf * tgt / tgt.sum()))
        if fit.sum() > 0:
            out["fit_mean_contacts"][t] = float(np.sum(gridf * fit / fit.sum()))
        if rem.sum() > 0:
            out["remd_mean_contacts"][t] = float(np.sum(gridf * rem / rem.sum()))
        out["contact_wass_tr"][t] = _wass(gridf, tgt, rem)

    # Rg block (Part 11.3). Target-vs-REMD Rg can be scored even when the
    # baseline is contact-only and therefore the analytic fitter has no Rg
    # prediction. Fit-related Rg columns remain NaN in that case.
    rg_ok = include_rg and target_has_rg(target_path)
    if rg_ok:
        with np.load(target_path, allow_pickle=True) as tnpz:
            rc_key = "rg_centers" if "rg_centers" in tnpz else "Rg_centers"
            rh_key = "rg_hists" if "rg_hists" in tnpz else "Rg_hists"
            tg_centers = np.asarray(tnpz[rc_key], dtype=float)
            tg_hists = np.asarray(tnpz[rh_key], dtype=float)
        if tg_centers.ndim != 1 or tg_hists.shape != (n_temps, tg_centers.size):
            raise ValueError(
                f"Target Rg arrays have incompatible shapes: centers={tg_centers.shape}, "
                f"hists={tg_hists.shape}, n_temps={n_temps}"
            )
        if not np.all(np.isfinite(tg_centers)) or not np.all(np.diff(tg_centers) > 0):
            raise ValueError("Target Rg centers must be finite and strictly increasing")
        if not np.all(np.isfinite(tg_hists)) or np.any(tg_hists < 0):
            raise ValueError("Target Rg histograms must be finite and nonnegative")
        if np.any(tg_hists.sum(axis=1) <= 0):
            raise ValueError("Every target Rg histogram row must contain positive mass")
        if Prg.shape[0] != n_temps or Prg.shape[1] != rg_remd_centers.size:
            raise ValueError(
                f"REMD Rg arrays have incompatible shapes: centers={rg_remd_centers.shape}, "
                f"Prg={Prg.shape}, n_temps={n_temps}"
            )
        if (
            rg_remd_edges.ndim != 1
            or rg_remd_edges.size != rg_remd_centers.size + 1
            or not np.all(np.isfinite(rg_remd_edges))
            or not np.all(np.diff(rg_remd_edges) > 0)
        ):
            raise ValueError("REMD Rg edges are invalid or inconsistent with centers")
        tg_edges = centers_to_edges(tg_centers)
        if (rg_fit_centers is None) != (rg_mod is None):
            raise ValueError("Fit results contain incomplete Rg prediction arrays")
        if rg_fit_centers is not None and rg_mod is not None:
            if (
                rg_fit_centers.ndim != 1
                or rg_mod.shape != (n_temps, rg_fit_centers.size)
                or not np.all(np.isfinite(rg_fit_centers))
                or not np.all(np.diff(rg_fit_centers) > 0)
                or not np.all(np.isfinite(rg_mod))
                or np.any(rg_mod < 0)
            ):
                raise ValueError("Fit Rg prediction arrays are invalid")
            fit_rg_available = True
        else:
            fit_rg_available = False
        edge_arrays = [tg_edges, rg_remd_edges]
        f_edges = None
        if fit_rg_available:
            f_edges = centers_to_edges(rg_fit_centers)
            edge_arrays.append(f_edges)
        common_edges = _common_rg_grid(edge_arrays, np.mean(np.diff(tg_edges)))
        common_centers = 0.5 * (common_edges[:-1] + common_edges[1:])
        out["rg_available"] = True
        for t in range(n_temps):
            tg_m = rebin_mass_piecewise(
                tg_edges, pdf_to_mass(tg_hists[t], tg_centers), common_edges
            )
            r_m = rebin_mass_piecewise(rg_remd_edges, Prg[t], common_edges)
            f_m = (
                rebin_mass_piecewise(f_edges, rg_mod[t], common_edges)
                if fit_rg_available else None
            )
            # Rebinned rows must remain (numerically) normalized.
            vectors = [("target", tg_m), ("remd", r_m)]
            if f_m is not None:
                vectors.append(("fit", f_m))
            for nm, vec in vectors:
                s = vec.sum()
                if s <= 0 or abs(s - 1.0) >= 1e-6:
                    raise ValueError(f"Rg rebin for {nm} did not conserve mass: sum={s}")
            out["rg_js_tr"][t] = js_divergence(tg_m, r_m)
            if f_m is not None:
                out["rg_js_tf"][t] = js_divergence(tg_m, f_m)
                out["rg_js_fr"][t] = js_divergence(f_m, r_m)
            if tg_m.sum() > 0:
                out["target_mean_rg"][t] = float(np.sum(common_centers * tg_m))
            if f_m is not None and f_m.sum() > 0:
                out["fit_mean_rg"][t] = float(np.sum(common_centers * f_m))
            if r_m.sum() > 0:
                out["remd_mean_rg"][t] = float(np.sum(common_centers * r_m))
            out["rg_wass_tr"][t] = _wass(common_centers, tg_m, r_m)
    return out


def _idx_mean(arr, idx):
    if idx is None or len(idx) == 0:
        return float("nan")
    return _safe_mean(np.asarray(arr)[idx])


def _rmse(a, b):
    diff = np.asarray(a) - np.asarray(b)
    diff = diff[np.isfinite(diff)]
    return float(np.sqrt(np.mean(diff ** 2))) if diff.size else float("nan")


def _mae(a, b):
    diff = np.asarray(a) - np.asarray(b)
    diff = diff[np.isfinite(diff)]
    return float(np.mean(np.abs(diff))) if diff.size else float("nan")


PER_TEMP_COLUMNS = [
    "baseline", "model", "seed", "temperature_index", "temperature", "split",
    "target_vs_fit_contact_js", "target_vs_remd_contact_js",
    "fit_vs_remd_contact_js", "target_vs_fit_rg_js", "target_vs_remd_rg_js",
    "fit_vs_remd_rg_js", "target_mean_contacts", "fit_mean_contacts",
    "remd_mean_contacts", "target_mean_rg", "fit_mean_rg", "remd_mean_rg",
]


# ---------------------------------------------------------------------------
# Cross-seed convergence diagnostics (Part 11): ESS/autocorr/round-trips/Rhat
# ---------------------------------------------------------------------------

try:  # optional: exact inverse-normal CDF for rank-normalized Rhat
    from scipy.special import ndtri as _ndtri
except Exception:  # pragma: no cover - scipy may be absent
    _ndtri = None


def _inv_normal_cdf(p):
    """Inverse standard-normal CDF (Acklam approximation; NumPy-only fallback)."""
    if _ndtri is not None:
        return _ndtri(p)
    p = np.asarray(p, dtype=float)
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    x = np.empty_like(p)
    lo = p < plow
    hi = p > phigh
    mid = (~lo) & (~hi)
    if np.any(lo):
        q = np.sqrt(-2 * np.log(p[lo]))
        x[lo] = (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if np.any(hi):
        q = np.sqrt(-2 * np.log(1 - p[hi]))
        x[hi] = -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                 ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if np.any(mid):
        q = p[mid] - 0.5
        r = q * q
        x[mid] = (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
                 (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    return x


def _rankdata_average(a):
    """Average-tie ranks (1..n), NumPy-only (scipy.stats.rankdata equivalent)."""
    a = np.asarray(a, dtype=float)
    n = a.size
    sorter = np.argsort(a, kind="mergesort")
    inv = np.empty(n, dtype=int)
    inv[sorter] = np.arange(n)
    a_sorted = a[sorter]
    obs = np.concatenate(([True], a_sorted[1:] != a_sorted[:-1]))
    dense = np.cumsum(obs)[inv]
    count = np.concatenate((np.nonzero(obs)[0], [n]))
    return 0.5 * (count[dense] + count[dense - 1] + 1)


def split_rhat(chains) -> float:
    """Standard split-Rhat for a list of 1-D chains (NumPy-only).

    Each chain is split into two halves to detect within-chain non-stationarity.
    Returns NaN when there is insufficient data.
    """
    chains = [np.asarray(c, dtype=float) for c in chains]
    chains = [c[np.isfinite(c)] for c in chains if c.size >= 4]
    if len(chains) < 2:
        return float("nan")
    split = []
    for c in chains:
        h = c.size // 2
        if h < 2:
            continue
        split.append(c[:h])
        split.append(c[h:2 * h])
    if len(split) < 2:
        return float("nan")
    N = min(s.size for s in split)
    split = [s[:N] for s in split]
    means = np.array([s.mean() for s in split], dtype=float)
    vars = np.array([s.var(ddof=1) for s in split], dtype=float)
    W = float(vars.mean())
    if W <= 0.0:
        return 1.0 if float(means.var(ddof=1)) <= 0.0 else float("inf")
    B = N * float(means.var(ddof=1))
    var_hat = (N - 1) / N * W + B / N
    return float(np.sqrt(var_hat / W))


def rank_normalized_split_rhat(chains) -> float:
    """Rank-normalized split-Rhat (Vehtari et al. 2021), NumPy-only.

    Pools all draws, replaces them with normal scores of their average ranks
    (Blom transform), then computes split-Rhat.  Robust to heavy tails and is
    invariant to monotone reparameterization.
    """
    chains = [np.asarray(c, dtype=float) for c in chains]
    chains = [c[np.isfinite(c)] for c in chains if c.size >= 4]
    if len(chains) < 2:
        return float("nan")
    sizes = [c.size for c in chains]
    pooled = np.concatenate(chains)
    n = pooled.size
    ranks = _rankdata_average(pooled)
    z = _inv_normal_cdf((ranks - 3.0 / 8.0) / (n - 0.25))
    # Re-split z back into the original chains.
    out, start = [], 0
    for sz in sizes:
        out.append(z[start:start + sz])
        start += sz
    return split_rhat(out)


def _load_seed_diagnostics(seed_dir: Path) -> dict:
    path = seed_dir / "run_diagnostics.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _load_seed_traj(seed_dir: Path):
    path = seed_dir / REMD_DIAGNOSTIC_TRAJECTORY_FILE
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=True) as d:
            return {
                "contacts_post": np.asarray(d["contacts_post"], dtype=float),
                "rg_post": np.asarray(d["rg_post"], dtype=float),
                "energy_post": np.asarray(d["energy_post"], dtype=float),
            }
    except Exception:
        return None


DIAG_PER_TEMP_COLUMNS = [
    "baseline", "model", "temperature_index", "temperature", "n_seeds",
    "ess_contacts_min", "ess_contacts_median", "ess_contacts_total",
    "tau_contacts_max", "ess_rg_min", "ess_rg_median", "tau_rg_max",
    "drift_contacts_max", "seed_mean_contacts_std", "seed_mean_rg_std",
    "rhat_contacts", "rhat_rg",
]


def aggregate_seed_diagnostics(rec, baseline, target_temps):
    """Aggregate per-seed REMD diagnostics for one model into row + per-T rows.

    Returns (row_additions, per_temp_diag_rows).  Only ok seeds contribute.
    Cross-seed Rhat is computed when trajectory NPZs are present for >= 2 seeds.
    """
    diag_cfg = baseline["remd"].get("diagnostics") or {}
    raw = baseline["raw_name"]
    n_temps = int(target_temps.size)

    seed_diags, seed_trajs = [], []
    for srec in rec.get("seeds", {}).values():
        if srec.get("status") != "ok":
            continue
        sd = _load_seed_diagnostics(Path(srec["seed_dir"]))
        if sd:
            seed_diags.append(sd)
            seed_trajs.append(_load_seed_traj(Path(srec["seed_dir"])))

    add = {"convergence_status": "unknown"}
    per_temp = []
    if not seed_diags:
        return add, per_temp

    # Per-temperature collection across seeds.
    rhat_thr = float(diag_cfg.get("rhat_threshold", 1.1))
    all_ess_c, all_tau_c, all_drift_c, all_ess_rg, all_tau_rg = [], [], [], [], []
    all_rhat_c, all_rhat_rg = [], []
    total_ess_c = 0.0
    cov_min, rt_totals, rt_min_walker = [], [], []
    for sd in seed_diags:
        summ = sd.get("summary", {})
        cov_min.append(_f(summ.get("min_temp_coverage")))
        rt_totals.append(_f(summ.get("total_round_trips_low")))
        rt_min_walker.append(_f(summ.get("min_round_trips_per_walker")))

    for t in range(n_temps):
        ess_c, tau_c, drift_c, ess_rg, tau_rg = [], [], [], [], []
        mean_c, mean_rg = [], []
        for sd in seed_diags:
            lanes = sd.get("lane_convergence", [])
            if t >= len(lanes):
                continue
            lane = lanes[t]
            c = lane.get("contacts", {})
            r = lane.get("rg", {})
            ess_c.append(_f(c.get("ess")))
            tau_c.append(_f(c.get("tau_int")))
            drift_c.append(abs(_f(c.get("drift_in_std"))))
            ess_rg.append(_f(r.get("ess")))
            tau_rg.append(_f(r.get("tau_int")))
            mean_c.append(_f(c.get("mean")))
            mean_rg.append(_f(r.get("mean")))

        # Cross-seed Rhat from trajectories at this temperature.
        rhat_c = rhat_rg = float("nan")
        traj_chains_c = [tr["contacts_post"][t] for tr in seed_trajs
                         if tr is not None and t < tr["contacts_post"].shape[0]]
        traj_chains_rg = [tr["rg_post"][t] for tr in seed_trajs
                          if tr is not None and t < tr["rg_post"].shape[0]]
        if len(traj_chains_c) >= 2:
            rhat_c = rank_normalized_split_rhat(traj_chains_c)
        if len(traj_chains_rg) >= 2:
            rhat_rg = rank_normalized_split_rhat(traj_chains_rg)

        ess_c_f = [v for v in ess_c if np.isfinite(v)]
        ess_rg_f = [v for v in ess_rg if np.isfinite(v)]
        per_temp.append({
            "baseline": raw, "model": rec.get("model"),
            "temperature_index": t,
            "temperature": float(target_temps[t]),
            "n_seeds": len(seed_diags),
            "ess_contacts_min": _safe_min(ess_c),
            "ess_contacts_median": _safe_mean(ess_c) if not ess_c_f else float(np.median(ess_c_f)),
            "ess_contacts_total": float(sum(ess_c_f)) if ess_c_f else float("nan"),
            "tau_contacts_max": _safe_max(tau_c),
            "ess_rg_min": _safe_min(ess_rg),
            "ess_rg_median": float(np.median(ess_rg_f)) if ess_rg_f else float("nan"),
            "tau_rg_max": _safe_max(tau_rg),
            "drift_contacts_max": _safe_max(drift_c),
            "seed_mean_contacts_std": _safe_std(mean_c),
            "seed_mean_rg_std": _safe_std(mean_rg),
            "rhat_contacts": rhat_c,
            "rhat_rg": rhat_rg,
        })
        all_ess_c += ess_c_f
        all_tau_c += [v for v in tau_c if np.isfinite(v)]
        all_drift_c += [v for v in drift_c if np.isfinite(v)]
        all_ess_rg += ess_rg_f
        all_tau_rg += [v for v in tau_rg if np.isfinite(v)]
        if np.isfinite(rhat_c):
            all_rhat_c.append(rhat_c)
        if np.isfinite(rhat_rg):
            all_rhat_rg.append(rhat_rg)
        total_ess_c += float(sum(ess_c_f))

    add["diag_n_seeds"] = len(seed_diags)
    add["diag_min_ess_contacts"] = _safe_min(all_ess_c)
    add["diag_median_ess_contacts"] = float(np.median(all_ess_c)) if all_ess_c else float("nan")
    add["diag_total_ess_contacts"] = float(total_ess_c) if all_ess_c else float("nan")
    add["diag_max_autocorr_contacts"] = _safe_max(all_tau_c)
    add["diag_max_autocorr_rg"] = _safe_max(all_tau_rg)
    add["diag_min_ess_rg"] = _safe_min(all_ess_rg)
    add["diag_total_round_trips"] = float(sum(v for v in rt_totals if np.isfinite(v)))
    add["diag_min_round_trips_total"] = _safe_min(rt_totals)
    add["diag_min_round_trips_per_walker"] = _safe_min(rt_min_walker)
    add["diag_min_temp_coverage"] = _safe_min(cov_min)
    add["diag_max_drift_contacts"] = _safe_max(all_drift_c)
    add["diag_seed_mean_contacts_dispersion"] = _safe_mean(
        [p["seed_mean_contacts_std"] for p in per_temp]
    )
    add["diag_seed_mean_rg_dispersion"] = _safe_mean(
        [p["seed_mean_rg_std"] for p in per_temp]
    )
    add["diag_max_rhat_contacts"] = _safe_max(all_rhat_c)
    add["diag_max_rhat_rg"] = _safe_max(all_rhat_rg)

    # Convergence-qualified status (does NOT change numeric ranks).
    reasons = []
    if np.isfinite(_f(add["diag_min_round_trips_total"])) and \
            _f(add["diag_min_round_trips_total"]) < int(diag_cfg.get("min_round_trips", 1)):
        reasons.append("round_trips")
    if np.isfinite(_f(add["diag_min_ess_contacts"])) and \
            _f(add["diag_min_ess_contacts"]) < float(diag_cfg.get("min_ess", 50.0)):
        reasons.append("low_ess")
    if np.isfinite(_f(add["diag_min_temp_coverage"])) and \
            _f(add["diag_min_temp_coverage"]) < float(diag_cfg.get("min_temp_coverage", 0.5)):
        reasons.append("temp_coverage")
    if np.isfinite(_f(add["diag_max_drift_contacts"])) and \
            _f(add["diag_max_drift_contacts"]) > float(diag_cfg.get("max_drift", 1.0)):
        reasons.append("drift")
    if np.isfinite(_f(add["diag_max_rhat_contacts"])) and \
            _f(add["diag_max_rhat_contacts"]) > rhat_thr:
        reasons.append("rhat")
    add["convergence_status"] = "unreliable" if reasons else "reliable"
    add["convergence_flags"] = ";".join(reasons)
    return add, per_temp


def build_comparison_rows(suite_state, log: Logger):
    """Return (aggregated rows, per-temperature rows) for all completed jobs."""
    cfg = suite_state["config"]
    target_path = cfg["target_remd"]
    target_temps = validate_target_npz(target_path)
    rows, per_temp_rows, per_temp_diag_rows = [], [], []

    for baseline in cfg["baselines"]:
        bname = baseline["name"]
        include_rg = bool(baseline["comparison"].get("include_rg", False))
        rg_weight = float(baseline["comparison"].get("rg_weight", 1.0))
        baseline_rows = []
        for model in cfg["models"]:
            rec = suite_state["models"].get((bname, model), {})
            fit_dir = Path(rec.get("fit_dir", ""))
            row = {
                "baseline": baseline["raw_name"], "model": model,
                "fit_status": rec.get("status", "unknown"),
                "status": rec.get("status", "unknown"),
                "n_parameters": float("nan"),
                "n_successful_seeds": 0, "n_failed_seeds": 0,
            }
            summary = rec.get("summary")
            if summary is not None:
                row["n_parameters"] = len(summary.get("param_names", []))
                row["fit_train_contact_loss"] = summary.get("train_loss")
                row["fit_validation_contact_loss"] = summary.get("val_loss")
                row["fit_all_contact_loss"] = summary.get("all_loss")
                row["fit_train_rg_loss"] = summary.get("rg_train_loss")
                row["fit_validation_rg_loss"] = summary.get("rg_val_loss")
                row["fit_all_rg_loss"] = summary.get("rg_all_loss")
                row["fit_optimization_objective"] = summary.get(
                    "optimization_objective_value"
                )

            # Per-seed scalar accumulators.
            acc = {k: [] for k in (
                "ctr_train", "ctr_val", "ctr_all", "rgtr_val", "rgtr_all",
                "comb_val", "comb_all", "cfr_all", "rgfr_all",
                "mc_rmse", "mc_mae", "mr_rmse", "mr_mae",
                "swap_min", "swap_med", "local",
            )}
            train_idx = val_idx = np.array([], dtype=int)
            if summary is not None and (fit_dir / "fit_results.npz").exists():
                with np.load(fit_dir / "fit_results.npz", allow_pickle=True) as f:
                    n_temps = int(np.asarray(f["p_obs_mass"]).shape[0])
                train_idx, val_idx = _split_indices(
                    fit_dir / "fit_results.npz", n_temps
                )
                val_set, train_set = set(val_idx.tolist()), set(train_idx.tolist())
                for seed, srec in rec.get("seeds", {}).items():
                    if srec.get("status") != "ok":
                        row["n_failed_seeds"] += 1
                        continue
                    row["n_successful_seeds"] += 1
                    seed_dir = Path(srec["seed_dir"])
                    cs = compare_seed(fit_dir, seed_dir, target_path, include_rg)
                    nT = cs["n_temps"]

                    # Combined per-temperature (B): contact + w*rg (if available).
                    comb = np.array(cs["contact_js_tr"], dtype=float)
                    if cs["rg_available"]:
                        comb = comb + rg_weight * np.nan_to_num(
                            cs["rg_js_tr"], nan=0.0
                        )

                    # Scalar score per (model, seed) for paired statistics:
                    # validation combined JS when a split exists, else all-T.
                    seed_score = (_idx_mean(comb, val_idx) if val_idx.size
                                  else _safe_mean(comb))
                    rec.setdefault("seed_scores", {})[seed] = float(seed_score)

                    acc["ctr_train"].append(_idx_mean(cs["contact_js_tr"], train_idx))
                    acc["ctr_val"].append(_idx_mean(cs["contact_js_tr"], val_idx))
                    acc["ctr_all"].append(_safe_mean(cs["contact_js_tr"]))
                    acc["rgtr_val"].append(_idx_mean(cs["rg_js_tr"], val_idx))
                    acc["rgtr_all"].append(_safe_mean(cs["rg_js_tr"]))
                    acc["comb_val"].append(_idx_mean(comb, val_idx))
                    acc["comb_all"].append(_safe_mean(comb))
                    acc["cfr_all"].append(_safe_mean(cs["contact_js_fr"]))
                    acc["rgfr_all"].append(_safe_mean(cs["rg_js_fr"]))
                    acc["mc_rmse"].append(
                        _rmse(cs["target_mean_contacts"], cs["remd_mean_contacts"])
                    )
                    acc["mc_mae"].append(
                        _mae(cs["target_mean_contacts"], cs["remd_mean_contacts"])
                    )
                    acc["mr_rmse"].append(
                        _rmse(cs["target_mean_rg"], cs["remd_mean_rg"])
                    )
                    acc["mr_mae"].append(
                        _mae(cs["target_mean_rg"], cs["remd_mean_rg"])
                    )
                    rs = srec.get("run_summary", {})
                    acc["swap_min"].append(rs.get("swap_rate_min"))
                    acc["swap_med"].append(rs.get("swap_rate_median"))
                    lar = rs.get("local_acceptance_rates") or []
                    acc["local"].append(_safe_mean(lar) if lar else None)

                    # Per-temperature rows.
                    for t in range(nT):
                        split = ("validation" if t in val_set
                                 else "train" if t in train_set else "all")
                        per_temp_rows.append({
                            "baseline": baseline["raw_name"], "model": model,
                            "seed": seed, "temperature_index": t,
                            "temperature": float(target_temps[t])
                            if t < target_temps.size else float("nan"),
                            "split": split,
                            "target_vs_fit_contact_js": cs["contact_js_tf"][t],
                            "target_vs_remd_contact_js": cs["contact_js_tr"][t],
                            "fit_vs_remd_contact_js": cs["contact_js_fr"][t],
                            "target_vs_fit_rg_js": cs["rg_js_tf"][t],
                            "target_vs_remd_rg_js": cs["rg_js_tr"][t],
                            "fit_vs_remd_rg_js": cs["rg_js_fr"][t],
                            "target_mean_contacts": cs["target_mean_contacts"][t],
                            "fit_mean_contacts": cs["fit_mean_contacts"][t],
                            "remd_mean_contacts": cs["remd_mean_contacts"][t],
                            "target_mean_rg": cs["target_mean_rg"][t],
                            "fit_mean_rg": cs["fit_mean_rg"][t],
                            "remd_mean_rg": cs["remd_mean_rg"][t],
                        })
            else:
                row["n_failed_seeds"] = len(rec.get("seeds", {}))

            row["has_validation"] = bool(val_idx.size)
            # Aggregate seed-level scalars (mean/std; min/max kept in JSON).
            row["remd_target_train_contact_js_mean"] = _safe_mean(acc["ctr_train"])
            row["remd_target_train_contact_js_std"] = _safe_std(acc["ctr_train"])
            row["remd_target_validation_contact_js_mean"] = _safe_mean(acc["ctr_val"])
            row["remd_target_validation_contact_js_std"] = _safe_std(acc["ctr_val"])
            row["remd_target_all_contact_js_mean"] = _safe_mean(acc["ctr_all"])
            row["remd_target_all_contact_js_std"] = _safe_std(acc["ctr_all"])
            row["remd_target_validation_rg_js_mean"] = _safe_mean(acc["rgtr_val"])
            row["remd_target_validation_rg_js_std"] = _safe_std(acc["rgtr_val"])
            row["remd_target_all_rg_js_mean"] = _safe_mean(acc["rgtr_all"])
            row["remd_target_all_rg_js_std"] = _safe_std(acc["rgtr_all"])
            row["remd_target_validation_combined_js_mean"] = _safe_mean(acc["comb_val"])
            row["remd_target_validation_combined_js_std"] = _safe_std(acc["comb_val"])
            row["remd_target_all_combined_js_mean"] = _safe_mean(acc["comb_all"])
            row["remd_target_all_combined_js_std"] = _safe_std(acc["comb_all"])
            row["remd_fit_contact_js_mean"] = _safe_mean(acc["cfr_all"])
            row["remd_fit_rg_js_mean"] = _safe_mean(acc["rgfr_all"])
            row["mean_contacts_rmse_mean"] = _safe_mean(acc["mc_rmse"])
            row["mean_contacts_rmse_std"] = _safe_std(acc["mc_rmse"])
            row["mean_contacts_mae_mean"] = _safe_mean(acc["mc_mae"])
            row["mean_rg_rmse_mean"] = _safe_mean(acc["mr_rmse"])
            row["mean_rg_rmse_std"] = _safe_std(acc["mr_rmse"])
            row["mean_rg_mae_mean"] = _safe_mean(acc["mr_mae"])
            row["swap_rate_min_mean"] = _safe_mean(acc["swap_min"])
            row["swap_rate_median_mean"] = _safe_mean(acc["swap_med"])
            row["local_acceptance_mean"] = _safe_mean(acc["local"])
            if row["fit_status"] == "ok":
                if row["n_successful_seeds"] == 0:
                    row["status"] = "remd_failed"
                elif row["n_failed_seeds"] > 0:
                    row["status"] = "partial"
                else:
                    row["status"] = "ok"

            # Cross-seed convergence diagnostics (only when enabled).
            if (baseline["remd"].get("diagnostics") or {}).get("enabled", False) \
                    and row["fit_status"] == "ok":
                rec["model"] = model
                diag_add, diag_pt = aggregate_seed_diagnostics(
                    rec, baseline, target_temps
                )
                row.update(diag_add)
                per_temp_diag_rows.extend(diag_pt)
            baseline_rows.append(row)

        _assign_ranks(baseline_rows, log)
        rows.extend(baseline_rows)
    return rows, per_temp_rows, per_temp_diag_rows


def _assign_ranks(rows: list[dict], log: Logger) -> None:
    """Fit rank by val (else all) fit loss; sim rank by val (else all) combined JS.

    Within a tie tolerance, prefer the simpler (fewer-parameter) model with a
    textual note — the numeric rank order is not silently altered.
    """
    def _fit_key(r):
        v = r.get("fit_validation_contact_loss")
        if v is None or not np.isfinite(_f(v)):
            v = r.get("fit_all_contact_loss")
        return _f(v)

    def _sim_key(r):
        if r.get("has_validation"):
            v = r.get("remd_target_validation_combined_js_mean")
        else:
            v = r.get("remd_target_all_combined_js_mean")
        return _f(v)

    fit_ok = [r for r in rows if r.get("fit_status") == "ok"]
    for i, r in enumerate(sorted(fit_ok, key=_fit_key), start=1):
        r["fit_rank"] = i
    sim_ranked = sorted(
        [r for r in fit_ok if r.get("n_successful_seeds", 0) > 0
         and np.isfinite(_sim_key(r))],
        key=_sim_key,
    )
    for i, r in enumerate(sim_ranked, start=1):
        r["simulation_rank"] = i
    for r in rows:
        r.setdefault("fit_rank", None)
        r.setdefault("simulation_rank", None)
        r.setdefault("simulation_rank_note", "")

    # Simpler-model tie note (does not change numeric ranks).
    if len(sim_ranked) >= 2:
        best = sim_ranked[0]
        for r in sim_ranked[1:]:
            if (abs(_sim_key(r) - _sim_key(best)) <= 1e-3
                    and r.get("n_parameters", np.inf) < best.get("n_parameters", np.inf)):
                r["simulation_rank_note"] = (
                    f"within ~1e-3 of rank-1 with fewer parameters "
                    f"({r['n_parameters']} vs {best['n_parameters']}); "
                    f"consider preferring on parsimony"
                )


def _f(v) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return float("inf")
    return x if np.isfinite(x) else float("inf")


COMPARISON_COLUMNS = [
    "baseline", "model", "n_parameters", "n_successful_seeds", "n_failed_seeds",
    "fit_train_contact_loss", "fit_validation_contact_loss",
    "fit_all_contact_loss", "fit_train_rg_loss", "fit_validation_rg_loss",
    "fit_all_rg_loss", "fit_optimization_objective",
    "remd_target_train_contact_js_mean", "remd_target_train_contact_js_std",
    "remd_target_validation_contact_js_mean",
    "remd_target_validation_contact_js_std",
    "remd_target_all_contact_js_mean", "remd_target_all_contact_js_std",
    "remd_target_validation_rg_js_mean", "remd_target_validation_rg_js_std",
    "remd_target_all_rg_js_mean", "remd_target_all_rg_js_std",
    "remd_target_validation_combined_js_mean",
    "remd_target_validation_combined_js_std",
    "remd_target_all_combined_js_mean", "remd_target_all_combined_js_std",
    "remd_fit_contact_js_mean", "remd_fit_rg_js_mean",
    "mean_contacts_rmse_mean", "mean_contacts_rmse_std",
    "mean_contacts_mae_mean",
    "mean_rg_rmse_mean", "mean_rg_rmse_std", "mean_rg_mae_mean",
    "swap_rate_min_mean", "swap_rate_median_mean", "local_acceptance_mean",
    "diag_n_seeds", "diag_min_ess_contacts", "diag_median_ess_contacts",
    "diag_total_ess_contacts", "diag_min_ess_rg",
    "diag_max_autocorr_contacts", "diag_max_autocorr_rg",
    "diag_total_round_trips", "diag_min_round_trips_total",
    "diag_min_round_trips_per_walker", "diag_min_temp_coverage",
    "diag_max_drift_contacts", "diag_seed_mean_contacts_dispersion",
    "diag_seed_mean_rg_dispersion", "diag_max_rhat_contacts",
    "diag_max_rhat_rg", "convergence_status", "convergence_flags",
    # Fit-robustness scalar findings (detailed data lives in the supplementary
    # JSON/CSV outputs and comparison/fit_robustness_summary.json).
    "bootstrap_enabled", "boot_requested_replicates", "boot_successful_replicates",
    "boot_failed_replicates", "boot_confidence", "boot_widest_relative_ci",
    "boot_max_abs_param_correlation", "boot_n_identifiability_warnings",
    "boot_max_bound_hit_fraction", "boot_extra_terms_ci_include_zero", "boot_status",
    "split_enabled", "split_n_attempted", "split_n_succeeded",
    "split_mean_heldout_loss", "split_range_heldout_loss", "split_worst_blocked_low",
    "split_worst_blocked_mid", "split_worst_blocked_high", "split_max_param_cv",
    "split_boundary_warnings", "split_stability_status",
    "rgw_enabled", "rgw_tested_weights", "rgw_production_weight",
    "rgw_pareto_weights", "rgw_knee_weight", "rgw_contact_loss_range",
    "rgw_rg_loss_range", "rgw_weight_sensitive", "rgw_status",
    "unc_enabled", "unc_restart_objective_spread", "unc_n_distinct_minima",
    "unc_condition_number", "unc_positive_definite", "unc_identifiability_status",
    "has_validation", "fit_rank", "simulation_rank", "simulation_rank_note",
    "fit_status", "status",
]


def write_comparison(rows, per_temp_rows, comparison_dir: Path,
                     log: Logger, per_temp_diag_rows=None) -> None:
    comparison_dir.mkdir(parents=True, exist_ok=True)
    csv_path = comparison_dir / "model_comparison.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(COMPARISON_COLUMNS)
        for r in rows:
            w.writerow([_csv_safe_value(r.get(c, "")) for c in COMPARISON_COLUMNS])
    json_path = comparison_dir / "model_comparison.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(_json_safe(rows), fh, indent=2, allow_nan=False)

    pt_csv = comparison_dir / "per_temperature_metrics.csv"
    with open(pt_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(PER_TEMP_COLUMNS)
        for r in per_temp_rows:
            w.writerow([_csv_safe_value(r.get(c, "")) for c in PER_TEMP_COLUMNS])
    pt_json = comparison_dir / "per_temperature_metrics.json"
    with open(pt_json, "w", encoding="utf-8") as fh:
        json.dump(_json_safe(per_temp_rows), fh, indent=2, allow_nan=False)
    log(f"Wrote {csv_path}, {json_path}, {pt_csv}, {pt_json}")

    if per_temp_diag_rows:
        diag_csv = comparison_dir / "per_temperature_diagnostics.csv"
        with open(diag_csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(DIAG_PER_TEMP_COLUMNS)
            for r in per_temp_diag_rows:
                w.writerow([_csv_safe_value(r.get(c, ""))
                            for c in DIAG_PER_TEMP_COLUMNS])
        diag_json = comparison_dir / "per_temperature_diagnostics.json"
        with open(diag_json, "w", encoding="utf-8") as fh:
            json.dump(_json_safe(per_temp_diag_rows), fh, indent=2,
                      allow_nan=False)
        log(f"Wrote {diag_csv}, {diag_json}")


def _csv_safe_value(value):
    if value is None:
        return ""
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return ""
    return value


# ---------------------------------------------------------------------------
# Fit-robustness aggregation (parsers for the fitter's supplementary summaries)
# ---------------------------------------------------------------------------
# The fitter remains the single source of truth: these helpers only READ the
# JSON/CSV it wrote and distill a few scalar findings. They tolerate a disabled
# analysis, a missing optional field, NaN-like nulls, failed bootstrap
# replicates, and models for which a derived quantity (Tc) is undefined.

def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _num(v):
    """Float of v if finite, else None (tolerates None / null / NaN strings)."""
    if v is None:
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _max_abs_offdiag(matrix) -> float | None:
    if not isinstance(matrix, list):
        return None
    best = None
    for i, row in enumerate(matrix):
        if not isinstance(row, list):
            continue
        for j, v in enumerate(row):
            if i == j:
                continue
            a = _num(v)
            if a is None:
                continue
            a = abs(a)
            if best is None or a > best:
                best = a
    return best


def parse_bootstrap_summary(fit_dir: Path, fit_cfg: dict) -> dict:
    """Distill bootstrap_summary.json into concise robustness fields."""
    bs = (fit_cfg.get("bootstrap") or {})
    if not bs.get("enabled"):
        return {"enabled": False}
    path = fit_dir / "bootstrap_summary.json"
    out: dict = {"enabled": True, "source": "bootstrap_summary.json"}
    data = _read_json(path)
    if not isinstance(data, dict):
        out["status"] = "missing"
        out["error"] = "bootstrap_summary.json missing or unreadable"
        return out
    out["requested_replicates"] = data.get("n_bootstrap")
    out["successful_replicates"] = data.get("n_success")
    out["failed_replicates"] = data.get("n_failed")
    out["confidence"] = _num(data.get("confidence"))
    params = data.get("params") or {}
    ci_by_param: dict = {}
    widest_rel = None
    for pn, st in params.items():
        if not isinstance(st, dict):
            continue
        lo, hi, fitted = _num(st.get("ci_low")), _num(st.get("ci_high")), _num(st.get("fitted"))
        ci_by_param[pn] = [lo, hi]
        if lo is not None and hi is not None and fitted is not None and abs(fitted) > 1e-12:
            rel = abs(hi - lo) / abs(fitted)
            if widest_rel is None or rel > widest_rel:
                widest_rel = rel
    out["param_confidence_intervals"] = ci_by_param
    out["widest_relative_ci"] = widest_rel
    corr = data.get("correlation") or {}
    out["strongest_abs_param_correlation"] = _max_abs_offdiag(corr.get("matrix"))
    flagged = corr.get("flagged_pairs") or []
    out["n_identifiability_warnings"] = len(flagged) if isinstance(flagged, list) else 0
    out["identifiability_pairs"] = [
        f"{f.get('param_a')}~{f.get('param_b')}={_fmt(_num(f.get('correlation')))}"
        for f in flagged if isinstance(f, dict)
    ]
    bound = data.get("param_bound_fractions") or {}
    max_bound = None
    for pn, fr in bound.items():
        if isinstance(fr, dict):
            v = _num(fr.get("at_any"))
            if v is not None and (max_bound is None or v > max_bound):
                max_bound = v
    out["max_bound_hit_fraction"] = max_bound
    # Extra terms (a2/a3/dCp) whose CI brackets zero -> term may be unnecessary.
    incl_zero = []
    for pn in EXTRA_TERM_PARAMS:
        if pn in ci_by_param:
            lo, hi = ci_by_param[pn]
            if lo is not None and hi is not None and lo <= 0.0 <= hi:
                incl_zero.append(pn)
    out["extra_terms_ci_include_zero"] = incl_zero
    # Derived Tc interval (only when defined for this model).
    derived = data.get("derived") or {}
    if isinstance(derived.get("Tc"), dict):
        out["Tc_ci"] = [_num(derived["Tc"].get("ci_low")), _num(derived["Tc"].get("ci_high"))]
    failed = out.get("failed_replicates") or 0
    warn = (out["n_identifiability_warnings"] > 0
            or (max_bound is not None and max_bound >= 0.5)
            or bool(incl_zero))
    if not data.get("n_success"):
        out["status"] = "no_successful_replicates"
    elif warn or failed:
        out["status"] = "warn"
    else:
        out["status"] = "ok"
    return out


def parse_uncertainty_diagnostics(fit_dir: Path, fit_cfg: dict) -> dict:
    """Distill uncertainty_diagnostics.json + restart_diagnostics.csv."""
    if not (fit_cfg.get("uncertainty_diagnostics") or {}).get("enabled"):
        return {"enabled": False}
    out: dict = {"enabled": True, "source": "uncertainty_diagnostics.json"}
    data = _read_json(fit_dir / "uncertainty_diagnostics.json")
    if not isinstance(data, dict):
        out["status"] = "missing"
        out["error"] = "uncertainty_diagnostics.json missing or unreadable"
        return out
    hess = data.get("hessian") or {}
    out["condition_number"] = _num(hess.get("condition_number"))
    out["positive_definite"] = bool(hess.get("positive_definite")) \
        if hess.get("positive_definite") is not None else None
    rs = data.get("restart_stability") or {}
    out["n_restarts"] = rs.get("n_restarts")
    out["n_successful_restarts"] = rs.get("n_success")
    out["n_distinct_minima"] = rs.get("n_distinct_objectives")
    out["distinct_minima"] = rs.get("distinct_minima")
    out["max_param_spread"] = _num(rs.get("max_param_spread"))
    # Restart objective spread (max - min over successful restarts).
    spread = None
    objs: list[float] = []
    rpath = fit_dir / "restart_diagnostics.csv"
    if rpath.exists():
        try:
            with open(rpath, newline="", encoding="utf-8") as fh:
                for r in csv.DictReader(fh):
                    if str(r.get("success")) in ("1", "True", "true"):
                        v = _num(r.get("objective"))
                        if v is not None:
                            objs.append(v)
        except Exception:
            objs = []
    if objs:
        spread = float(max(objs) - min(objs))
    out["restart_objective_spread"] = spread
    # The fitter does not export boundary hits for this analysis; report n/a.
    out["boundary_warnings"] = None
    cond = out["condition_number"]
    ill_conditioned = cond is not None and cond > 1e8
    multi = bool(out.get("distinct_minima"))
    not_posdef = out["positive_definite"] is False
    out["identifiability_status"] = (
        "warn" if (ill_conditioned or multi or not_posdef) else "ok"
    )
    return out


def parse_split_sensitivity(fit_dir: Path, fit_cfg: dict) -> dict:
    """Distill split_sensitivity_summary.json into concise stability fields."""
    if not (fit_cfg.get("split_sensitivity") or {}).get("enabled"):
        return {"enabled": False}
    out: dict = {"enabled": True, "source": "split_sensitivity_summary.json"}
    data = _read_json(fit_dir / "split_sensitivity_summary.json")
    if not isinstance(data, dict):
        out["status"] = "missing"
        out["error"] = "split_sensitivity_summary.json missing or unreadable"
        return out
    out["n_attempted"] = data.get("n_splits")
    out["n_succeeded"] = data.get("n_splits_succeeded")
    splits = data.get("splits") or []
    held = []
    worst = {"blocked_low": None, "blocked_mid": None, "blocked_high": None}
    n_boundary = 0
    for sp in splits:
        if not isinstance(sp, dict):
            continue
        if not sp.get("optimization_success", True):
            continue
        cv = _num(sp.get("combined_val_loss_mean"))
        if cv is not None:
            held.append(cv)
        scheme = sp.get("scheme")
        if scheme in worst:
            cur = worst[scheme]
            if cv is not None and (cur is None or cv > cur):
                worst[scheme] = cv
        bh = sp.get("boundary_hits")
        if bh:
            n_boundary += 1
    out["mean_heldout_combined_loss"] = float(np.mean(held)) if held else None
    out["range_heldout_combined_loss"] = (
        float(max(held) - min(held)) if len(held) >= 2 else (0.0 if held else None)
    )
    out["worst_blocked_low"] = worst["blocked_low"]
    out["worst_blocked_mid"] = worst["blocked_mid"]
    out["worst_blocked_high"] = worst["blocked_high"]
    stab = data.get("parameter_stability") or {}
    max_cv = None
    cv_by_param: dict = {}
    for pn, st in stab.items():
        if isinstance(st, dict):
            v = _num(st.get("cv"))
            cv_by_param[pn] = v
            if v is not None and (max_cv is None or v > max_cv):
                max_cv = v
    out["param_cv"] = cv_by_param
    out["max_param_cv"] = max_cv
    out["boundary_hit_warnings"] = n_boundary
    # Overall split-stability status: parameters that wander a lot across splits
    # (CV) or boundary hits indicate sensitivity.
    if max_cv is None:
        status = "unknown"
    elif max_cv > 0.5 or n_boundary > 0:
        status = "sensitive"
    elif max_cv > 0.2:
        status = "moderate"
    else:
        status = "stable"
    out["stability_status"] = status
    return out


def parse_rg_weight_sensitivity(fit_dir: Path, fit_cfg: dict) -> dict:
    """Distill rg_weight_summary.json into concise Pareto/tradeoff fields."""
    if not (fit_cfg.get("rg_weight_sensitivity") or {}).get("enabled"):
        return {"enabled": False}
    out: dict = {"enabled": True, "source": "rg_weight_summary.json"}
    data = _read_json(fit_dir / "rg_weight_summary.json")
    if not isinstance(data, dict):
        out["status"] = "missing"
        out["error"] = "rg_weight_summary.json missing or unreadable"
        return out
    out["tested_weights"] = data.get("weight_grid") or []
    out["production_weight"] = _num(data.get("production_weight"))
    out["pareto_efficient_weights"] = data.get("pareto_efficient_weights") or []
    out["knee_weight"] = _num(data.get("knee_weight_heuristic"))
    space = data.get("pareto_space") or "all"
    ckey = "contact_val_mean" if space == "validation" else "contact_all_mean"
    rkey = "rg_val_mean" if space == "validation" else "rg_all_mean"
    per = data.get("per_weight") or []
    cvals, rvals = [], []
    param_paths: dict = {}
    for r in per:
        if not isinstance(r, dict):
            continue
        c, rg = _num(r.get(ckey)), _num(r.get(rkey))
        if c is not None:
            cvals.append(c)
        if rg is not None:
            rvals.append(rg)
        for pn, pv in r.items():
            if pn in ("rg_weight", "is_production", "contact_only_fit",
                      "pareto_efficient", "is_knee"):
                continue
            val = _num(pv)
            if val is not None and pn in _RGW_PARAM_KEYS:
                param_paths.setdefault(pn, []).append(val)
    out["contact_loss_range"] = (
        float(max(cvals) - min(cvals)) if len(cvals) >= 2 else (0.0 if cvals else None)
    )
    out["rg_loss_range"] = (
        float(max(rvals) - min(rvals)) if len(rvals) >= 2 else (0.0 if rvals else None)
    )
    # Strong weight-sensitivity if any parameter's relative spread across weights
    # is large, or the production weight is Pareto-dominated.
    max_rel = None
    for pn, vals in param_paths.items():
        if len(vals) >= 2:
            mean = float(np.mean(vals))
            if abs(mean) > 1e-12:
                rel = (max(vals) - min(vals)) / abs(mean)
                if max_rel is None or rel > max_rel:
                    max_rel = rel
    prod = out["production_weight"]
    eff = out["pareto_efficient_weights"]
    prod_efficient = (prod is not None and isinstance(eff, list)
                      and any(abs(prod - _num(w)) <= 1e-9 for w in eff
                              if _num(w) is not None))
    out["production_weight_pareto_efficient"] = bool(prod_efficient)
    out["max_param_relative_spread"] = max_rel
    sensitive = (max_rel is not None and max_rel > 0.5) or not prod_efficient
    out["weight_sensitive"] = bool(sensitive)
    out["status"] = "sensitive" if sensitive else "stable"
    return out


# Parameter names across all models (used to recognize parameter columns in the
# Rg-weight per-weight records without misreading loss columns as parameters).
_RGW_PARAM_KEYS = {"h", "s", "A", "Tc", "a0", "a1", "a2", "a3", "dh0", "ds0", "dCp"}


FIT_ROBUSTNESS_COLUMNS = [
    "baseline", "model",
    "bootstrap_enabled", "boot_requested_replicates", "boot_successful_replicates",
    "boot_failed_replicates", "boot_confidence", "boot_widest_relative_ci",
    "boot_max_abs_param_correlation", "boot_n_identifiability_warnings",
    "boot_max_bound_hit_fraction", "boot_extra_terms_ci_include_zero", "boot_status",
    "split_enabled", "split_n_attempted", "split_n_succeeded",
    "split_mean_heldout_loss", "split_range_heldout_loss", "split_worst_blocked_low",
    "split_worst_blocked_mid", "split_worst_blocked_high", "split_max_param_cv",
    "split_boundary_warnings", "split_stability_status",
    "rgw_enabled", "rgw_tested_weights", "rgw_production_weight",
    "rgw_pareto_weights", "rgw_knee_weight", "rgw_contact_loss_range",
    "rgw_rg_loss_range", "rgw_weight_sensitive", "rgw_status",
    "unc_enabled", "unc_restart_objective_spread", "unc_n_distinct_minima",
    "unc_condition_number", "unc_positive_definite", "unc_identifiability_status",
    "bootstrap_source", "split_source", "rg_weight_source", "uncertainty_source",
]


def _join_list(values) -> str:
    if not values:
        return ""
    return ";".join(_num_str(v) if isinstance(v, (int, float)) else str(v)
                    for v in values)


def collect_fit_robustness(suite_state, rows, log: Logger):
    """Parse every model's supplementary summaries; merge concise fields into the
    comparison rows and return a structured record + flat per-model rows.

    Mutates `rows` in place to add the fit-robustness scalar columns. Detailed
    per-parameter data stays in the JSON/CSV summaries; only the most useful
    scalars land in model_comparison.csv.
    """
    cfg = suite_state["config"]
    fit_by_name = {b["name"]: b for b in cfg["baselines"]}
    row_by_key = {(r["baseline"], r["model"]): r for r in rows}

    structured: dict = {}
    flat_rows: list[dict] = []
    any_enabled = False

    for baseline in cfg["baselines"]:
        bname = baseline["name"]
        raw = baseline["raw_name"]
        fit_cfg = baseline["fit"]
        structured[raw] = {}
        for model in cfg["models"]:
            rec = suite_state["models"].get((bname, model), {})
            fit_dir = Path(rec.get("fit_dir", ""))
            # Only parse when the fit itself succeeded; otherwise the supplementary
            # files are absent or stale and the row already reflects the failure.
            fit_ok = rec.get("status") == "ok" or rec.get("summary") is not None
            boot = parse_bootstrap_summary(fit_dir, fit_cfg) if fit_ok else {"enabled": bool((fit_cfg.get("bootstrap") or {}).get("enabled"))}
            unc = parse_uncertainty_diagnostics(fit_dir, fit_cfg) if fit_ok else {"enabled": bool((fit_cfg.get("uncertainty_diagnostics") or {}).get("enabled"))}
            split = parse_split_sensitivity(fit_dir, fit_cfg) if fit_ok else {"enabled": bool((fit_cfg.get("split_sensitivity") or {}).get("enabled"))}
            rgw = parse_rg_weight_sensitivity(fit_dir, fit_cfg) if fit_ok else {"enabled": bool((fit_cfg.get("rg_weight_sensitivity") or {}).get("enabled"))}
            if any(d.get("enabled") for d in (boot, unc, split, rgw)):
                any_enabled = True
            structured[raw][model] = {
                "model": model, "bootstrap": boot, "uncertainty_diagnostics": unc,
                "split_sensitivity": split, "rg_weight_sensitivity": rgw,
            }

            row = row_by_key.get((raw, model))
            scalars = {
                "bootstrap_enabled": bool(boot.get("enabled")),
                "boot_requested_replicates": boot.get("requested_replicates"),
                "boot_successful_replicates": boot.get("successful_replicates"),
                "boot_failed_replicates": boot.get("failed_replicates"),
                "boot_confidence": boot.get("confidence"),
                "boot_widest_relative_ci": boot.get("widest_relative_ci"),
                "boot_max_abs_param_correlation": boot.get("strongest_abs_param_correlation"),
                "boot_n_identifiability_warnings": boot.get("n_identifiability_warnings"),
                "boot_max_bound_hit_fraction": boot.get("max_bound_hit_fraction"),
                "boot_extra_terms_ci_include_zero": _join_list(boot.get("extra_terms_ci_include_zero")),
                "boot_status": boot.get("status"),
                "split_enabled": bool(split.get("enabled")),
                "split_n_attempted": split.get("n_attempted"),
                "split_n_succeeded": split.get("n_succeeded"),
                "split_mean_heldout_loss": split.get("mean_heldout_combined_loss"),
                "split_range_heldout_loss": split.get("range_heldout_combined_loss"),
                "split_worst_blocked_low": split.get("worst_blocked_low"),
                "split_worst_blocked_mid": split.get("worst_blocked_mid"),
                "split_worst_blocked_high": split.get("worst_blocked_high"),
                "split_max_param_cv": split.get("max_param_cv"),
                "split_boundary_warnings": split.get("boundary_hit_warnings"),
                "split_stability_status": split.get("stability_status"),
                "rgw_enabled": bool(rgw.get("enabled")),
                "rgw_tested_weights": _join_list(rgw.get("tested_weights")),
                "rgw_production_weight": rgw.get("production_weight"),
                "rgw_pareto_weights": _join_list(rgw.get("pareto_efficient_weights")),
                "rgw_knee_weight": rgw.get("knee_weight"),
                "rgw_contact_loss_range": rgw.get("contact_loss_range"),
                "rgw_rg_loss_range": rgw.get("rg_loss_range"),
                "rgw_weight_sensitive": rgw.get("weight_sensitive"),
                "rgw_status": rgw.get("status"),
                "unc_enabled": bool(unc.get("enabled")),
                "unc_restart_objective_spread": unc.get("restart_objective_spread"),
                "unc_n_distinct_minima": unc.get("n_distinct_minima"),
                "unc_condition_number": unc.get("condition_number"),
                "unc_positive_definite": unc.get("positive_definite"),
                "unc_identifiability_status": unc.get("identifiability_status"),
            }
            if row is not None:
                row.update(scalars)
            flat = {"baseline": raw, "model": model}
            flat.update(scalars)
            flat["bootstrap_source"] = boot.get("source", "")
            flat["split_source"] = split.get("source", "")
            flat["rg_weight_source"] = rgw.get("source", "")
            flat["uncertainty_source"] = unc.get("source", "")
            flat_rows.append(flat)

    return {
        "any_enabled": any_enabled,
        "structured": structured,
        "flat_rows": flat_rows,
    }


def write_fit_robustness(robustness, comparison_dir: Path, log: Logger) -> None:
    """Write the consolidated machine-readable robustness summary + flat CSV."""
    comparison_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "generated": now_iso(),
        "any_analysis_enabled": robustness["any_enabled"],
        "baselines": robustness["structured"],
    }
    json_path = comparison_dir / "fit_robustness_summary.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(_json_safe(summary), fh, indent=2, allow_nan=False)
    csv_path = comparison_dir / "fit_robustness.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(FIT_ROBUSTNESS_COLUMNS)
        for r in robustness["flat_rows"]:
            w.writerow([_csv_safe_value(r.get(c, "")) for c in FIT_ROBUSTNESS_COLUMNS])
    log(f"Wrote {json_path}, {csv_path}")


# ---------------------------------------------------------------------------
# Cross-baseline ranking gating (Part 13)
# ---------------------------------------------------------------------------

def cross_baseline_global_ranking(suite_state, rows):
    """Return a global ranking only if all baselines are comparable, else None.

    Baselines must share contact_offset, Rg unit convention (rg_scale), and
    comparison metric configuration (include_rg, rg_weight).
    """
    baselines = suite_state["config"]["baselines"]
    if len(baselines) < 2:
        return None
    sig0 = _baseline_signature(baselines[0])
    if any(_baseline_signature(b) != sig0 for b in baselines[1:]):
        return None
    scored = []
    for r in rows:
        if r.get("fit_status") != "ok" or r.get("n_successful_seeds", 0) <= 0:
            continue
        v = (r.get("remd_target_validation_combined_js_mean")
             if r.get("has_validation")
             else r.get("remd_target_all_combined_js_mean"))
        if v is not None and np.isfinite(_f(v)):
            scored.append((r["baseline"], r["model"], float(v)))
    scored.sort(key=lambda x: x[2])
    return [{"rank": i, "baseline": b, "model": m, "combined_js": v}
            for i, (b, m, v) in enumerate(scored, start=1)]


def _baseline_signature(b) -> tuple:
    comp = b["comparison"]
    return (
        float(b["contact_offset"]), float(b["rg_scale"]),
        bool(comp.get("include_rg", False)), float(comp.get("rg_weight", 1.0)),
    )


# ---------------------------------------------------------------------------
# Paired model-comparison statistics (Part 12)
# ---------------------------------------------------------------------------
# All inference here is paired across the (few) independent seeds. With ~3 seeds
# the design has very low inferential power, so effect sizes, uncertainty, and
# practical equivalence are reported alongside (de-emphasized) p-values.

PAIRWISE_COLUMNS = [
    "baseline", "model_a", "model_b", "n_paired", "mean_delta", "median_delta",
    "std_delta", "se_delta", "ci_low", "ci_high", "p_sign_flip", "p_holm",
    "p_method", "effect_size_dz", "effect_size_reliable", "frac_a_won",
    "frac_b_won", "frac_tie", "prob_a_better", "prob_practical_equivalent",
    "favored",
]
SEED_SCORE_COLUMNS = ["baseline", "model", "seed", "score", "score_scope"]
RANK_STABILITY_COLUMNS = [
    "baseline", "model", "n_parameters", "n_seeds", "mean_rank", "median_rank",
    "std_rank", "prob_rank1", "n_seed_wins", "mean_score", "seed_ranks",
]


def _holm_adjust(pvals):
    """Holm-Bonferroni step-down adjustment; preserves input order."""
    m = len(pvals)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: (float("inf") if pvals[i] is None
                                            or not np.isfinite(pvals[i])
                                            else pvals[i]))
    adj = [None] * m
    running = 0.0
    for rank, idx in enumerate(order):
        p = pvals[idx]
        if p is None or not np.isfinite(p):
            adj[idx] = None
            continue
        val = (m - rank) * p
        running = max(running, val)
        adj[idx] = float(min(1.0, running))
    return adj


def _sign_flip_p(deltas):
    """Exact two-sided paired sign-flip permutation p-value (statistic = mean).

    Enumerates all 2^n sign assignments when feasible (n <= 22); returns the
    fraction with |mean(signed delta)| >= |observed mean|.  NaN for n == 0.
    """
    import itertools
    d = np.asarray([x for x in deltas if np.isfinite(x)], dtype=float)
    n = d.size
    if n == 0:
        return float("nan"), "none"
    obs = abs(float(d.mean()))
    tol = 1e-12
    if n <= 22:
        count = 0
        total = 0
        for signs in itertools.product((1.0, -1.0), repeat=n):
            stat = abs(float(np.dot(signs, d) / n))
            total += 1
            if stat >= obs - tol:
                count += 1
        return float(count) / float(total), "exact_sign_flip"
    # Fallback for large n: deterministic sampled sign flips.
    rng = np.random.RandomState(0)
    reps = 20000
    flips = rng.choice((1.0, -1.0), size=(reps, n))
    stats = np.abs((flips * d).mean(axis=1))
    return float(np.mean(stats >= obs - tol)), "sampled_sign_flip"


def _paired_bootstrap_means(deltas, n_rep, seed):
    """Deterministic bootstrap of the mean paired difference (resample seeds)."""
    d = np.asarray([x for x in deltas if np.isfinite(x)], dtype=float)
    n = d.size
    if n == 0:
        return np.array([], dtype=float)
    rng = np.random.RandomState(int(seed))
    idx = rng.randint(0, n, size=(int(n_rep), n))
    return d[idx].mean(axis=1)


def compute_pair_stats(scores_a, scores_b, model_a, model_b, seeds_common,
                       n_rep, seed, eps, alpha):
    """Paired statistics for delta = score_A - score_B over common seeds."""
    deltas = np.array([scores_a[s] - scores_b[s] for s in seeds_common],
                      dtype=float)
    deltas = deltas[np.isfinite(deltas)]
    n = deltas.size
    out = {
        "model_a": model_a, "model_b": model_b, "n_paired": int(n),
        "mean_delta": float("nan"), "median_delta": float("nan"),
        "std_delta": float("nan"), "se_delta": float("nan"),
        "ci_low": float("nan"), "ci_high": float("nan"),
        "p_sign_flip": float("nan"), "p_method": "none",
        "effect_size_dz": float("nan"), "effect_size_reliable": False,
        "frac_a_won": float("nan"), "frac_b_won": float("nan"),
        "frac_tie": float("nan"), "prob_a_better": float("nan"),
        "prob_practical_equivalent": float("nan"), "favored": "n/a",
    }
    if n == 0:
        return out
    mean_d = float(deltas.mean())
    std_d = float(deltas.std(ddof=1)) if n > 1 else 0.0
    out["mean_delta"] = mean_d
    out["median_delta"] = float(np.median(deltas))
    out["std_delta"] = std_d
    out["se_delta"] = (std_d / math.sqrt(n)) if n > 1 else float("nan")
    tol = 1e-12
    out["frac_a_won"] = float(np.mean(deltas < -tol))   # A better => delta<0
    out["frac_b_won"] = float(np.mean(deltas > tol))
    out["frac_tie"] = float(np.mean(np.abs(deltas) <= tol))
    p, method = _sign_flip_p(deltas)
    out["p_sign_flip"] = p
    out["p_method"] = method
    if std_d > 0:
        out["effect_size_dz"] = mean_d / std_d
    elif mean_d == 0:
        out["effect_size_dz"] = 0.0
    else:
        out["effect_size_dz"] = float("inf") if mean_d > 0 else float("-inf")
    out["effect_size_reliable"] = bool(n >= 5)
    boots = _paired_bootstrap_means(deltas, n_rep, seed)
    if boots.size:
        out["ci_low"] = float(np.percentile(boots, 100.0 * alpha / 2.0))
        out["ci_high"] = float(np.percentile(boots, 100.0 * (1.0 - alpha / 2.0)))
        out["prob_a_better"] = float(np.mean(boots < 0.0))
        out["prob_practical_equivalent"] = float(np.mean(np.abs(boots) < eps))
    if mean_d < -tol:
        out["favored"] = model_a
    elif mean_d > tol:
        out["favored"] = model_b
    else:
        out["favored"] = "tie"
    return out


def compute_rank_stability(model_scores, models, n_rep, seed):
    """Per-seed ranks, summary, and paired-bootstrap rank-1 probability.

    `model_scores` maps model -> {seed: score}.  Per-seed ranks use all models
    present at that seed (average ranks for ties); the bootstrap rank-1
    probability uses the seeds common to every included model.
    """
    seeds_by_model = {m: set(model_scores.get(m, {})) for m in models}
    all_seeds = sorted(set().union(*seeds_by_model.values())) if models else []

    # Per-seed ranks (1 = best/lowest score).
    seed_ranks = {m: {} for m in models}
    seed_wins = {m: 0 for m in models}
    for s in all_seeds:
        present = [m for m in models if s in model_scores.get(m, {})]
        if not present:
            continue
        vals = np.array([model_scores[m][s] for m in present], dtype=float)
        ranks = _rankdata_average(vals)
        best = float(vals.min())
        for m, r, v in zip(present, ranks, vals):
            seed_ranks[m][s] = float(r)
            if abs(v - best) <= 1e-12:
                seed_wins[m] += 1

    common = [s for s in all_seeds
              if all(s in model_scores.get(m, {}) for m in models)]
    prob_rank1 = {m: float("nan") for m in models}
    if len(common) >= 1 and len(models) >= 2:
        rng = np.random.RandomState(int(seed) + 7919)
        mat = np.array([[model_scores[m][s] for s in common] for m in models],
                       dtype=float)
        wins = np.zeros(len(models), dtype=float)
        nrep = int(n_rep)
        for _ in range(nrep):
            pick = rng.randint(0, len(common), size=len(common))
            means = mat[:, pick].mean(axis=1)
            mn = means.min()
            winners = np.where(np.abs(means - mn) <= 1e-12)[0]
            wins[winners] += 1.0 / winners.size
        prob_rank1 = {m: float(wins[i] / nrep) for i, m in enumerate(models)}

    rows = []
    for m in models:
        rks = [seed_ranks[m][s] for s in sorted(seed_ranks[m])]
        sc = list(model_scores.get(m, {}).values())
        rows.append({
            "model": m,
            "n_seeds": len(rks),
            "mean_rank": float(np.mean(rks)) if rks else float("nan"),
            "median_rank": float(np.median(rks)) if rks else float("nan"),
            "std_rank": float(np.std(rks, ddof=0)) if rks else float("nan"),
            "prob_rank1": prob_rank1[m],
            "n_seed_wins": int(seed_wins[m]),
            "mean_score": float(np.mean(sc)) if sc else float("nan"),
            "seed_ranks": ";".join(
                f"{s}:{seed_ranks[m][s]:.3g}" for s in sorted(seed_ranks[m])
            ),
        })
    return rows


def _parsimony_recommendation(models, model_scores, pair_lookup, n_params, eps):
    """Recommend the simplest model indistinguishable from the best (1-SE/PE)."""
    means = {m: (np.mean(list(model_scores[m].values()))
                 if model_scores.get(m) else float("inf")) for m in models}
    ranked = sorted(models, key=lambda m: means[m])
    if not ranked:
        return {"best_model": None, "recommended_model": None, "reason": "no models"}
    best = ranked[0]
    candidates = [best]
    for m in models:
        if m == best:
            continue
        key = (best, m) if (best, m) in pair_lookup else (m, best)
        ps = pair_lookup.get(key)
        if not ps or not ps.get("n_paired"):
            continue
        mean_d = abs(_f(ps.get("mean_delta")))
        se = _f(ps.get("se_delta"))
        ci_low, ci_high = _f(ps.get("ci_low")), _f(ps.get("ci_high"))
        ci_includes_0 = (np.isfinite(ci_low) and np.isfinite(ci_high)
                         and ci_low <= 0.0 <= ci_high)
        within_1se = np.isfinite(se) and mean_d <= se
        practically_equiv = mean_d < eps
        if ci_includes_0 or within_1se or practically_equiv:
            candidates.append(m)
    rec = min(
        candidates,
        key=lambda m: (n_params.get(m, float("inf")), means[m]),
    )
    reason = ("best model is already simplest among indistinguishable set"
              if rec == best else
              f"{rec} is statistically/practically indistinguishable from the "
              f"best model {best} but has fewer parameters")
    return {"best_model": best, "recommended_model": rec, "reason": reason}


def run_pairwise_statistics(suite_state, rows, comparison_dir: Path, log: Logger):
    """Compute paired model-comparison statistics; write CSV/JSON; return summary.

    Reuses the per-seed scalar scores collected during comparison (no REMD
    rerun).  Returns a per-baseline result structure for plotting and the report.
    """
    cfg = suite_state["config"]
    pairwise_rows, seed_rows, rank_rows = [], [], []
    summary = {}

    for baseline in cfg["baselines"]:
        stats_cfg = baseline["comparison"].get("statistics") or {}
        if not stats_cfg.get("enabled", False):
            continue
        bname, raw = baseline["name"], baseline["raw_name"]
        n_rep = int(stats_cfg["bootstrap_replicates"])
        seed = int(stats_cfg["seed"])
        eps = float(stats_cfg["practical_equivalence_epsilon"])
        alpha = float(stats_cfg["alpha"])
        do_holm = stats_cfg.get("multiple_testing", "holm") == "holm"

        # Gather per-(model, seed) scores for fit-ok models with scores.
        model_scores = {}
        n_params = {}
        for model in cfg["models"]:
            rec = suite_state["models"].get((bname, model), {})
            if rec.get("status") not in ("ok", "partial") and \
                    rec.get("summary") is None:
                continue
            ss = {int(s): float(v) for s, v in rec.get("seed_scores", {}).items()
                  if np.isfinite(_f(v))}
            if ss:
                model_scores[model] = ss
                summ = rec.get("summary") or {}
                n_params[model] = len(summ.get("param_names", []))
        models = [m for m in cfg["models"] if m in model_scores]
        score_scope = ("validation" if any(r.get("has_validation")
                       for r in rows if r["baseline"] == raw) else "all")

        for model in models:
            for s in sorted(model_scores[model]):
                seed_rows.append({
                    "baseline": raw, "model": model, "seed": s,
                    "score": model_scores[model][s], "score_scope": score_scope,
                })

        # Pairwise comparisons (i < j in configured model order).
        base_pairs = []
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                a, b = models[i], models[j]
                common = sorted(set(model_scores[a]) & set(model_scores[b]))
                ps = compute_pair_stats(
                    model_scores[a], model_scores[b], a, b, common,
                    n_rep, seed, eps, alpha,
                )
                ps["baseline"] = raw
                base_pairs.append(ps)
        if do_holm:
            adj = _holm_adjust([p["p_sign_flip"] for p in base_pairs])
        else:
            adj = [p["p_sign_flip"] for p in base_pairs]
        for p, a in zip(base_pairs, adj):
            p["p_holm"] = a
        pairwise_rows.extend(base_pairs)

        # Rank stability.
        brank = compute_rank_stability(model_scores, models, n_rep, seed)
        for r in brank:
            r["baseline"] = raw
            r["n_parameters"] = n_params.get(r["model"], 0)
        rank_rows.extend(brank)

        # Parsimony + interpretation summary.
        pair_lookup = {(p["model_a"], p["model_b"]): p for p in base_pairs}
        pars = _parsimony_recommendation(models, model_scores, pair_lookup,
                                         n_params, eps)
        n_seeds_min = min((len(model_scores[m]) for m in models), default=0)
        raw_winner = min(models, key=lambda m: np.mean(
            list(model_scores[m].values()))) if models else None
        supported = [
            {"model_a": p["model_a"], "model_b": p["model_b"],
             "p_holm": p["p_holm"], "mean_delta": p["mean_delta"]}
            for p in base_pairs
            if p["p_holm"] is not None and np.isfinite(p["p_holm"])
            and p["p_holm"] < alpha
        ]
        equivalent = [
            {"model_a": p["model_a"], "model_b": p["model_b"],
             "prob_practical_equivalent": p["prob_practical_equivalent"],
             "mean_delta": p["mean_delta"]}
            for p in base_pairs
            if np.isfinite(_f(p["prob_practical_equivalent"]))
            and _f(p["prob_practical_equivalent"]) >= 0.5
        ]
        summary[raw] = {
            "settings": {"alpha": alpha, "bootstrap_replicates": n_rep,
                         "seed": seed, "practical_equivalence_epsilon": eps,
                         "multiple_testing": stats_cfg.get("multiple_testing")},
            "n_models": len(models), "min_paired_seeds": int(n_seeds_min),
            "low_power": bool(n_seeds_min <= 3),
            "raw_winner": raw_winner,
            "parsimony": pars,
            "statistically_supported_differences": supported,
            "practical_equivalences": equivalent,
        }

    if not summary:
        return None

    comparison_dir.mkdir(parents=True, exist_ok=True)
    _write_rows_csv(comparison_dir / "pairwise_model_comparison.csv",
                    PAIRWISE_COLUMNS, pairwise_rows)
    with open(comparison_dir / "pairwise_model_comparison.json", "w",
              encoding="utf-8") as fh:
        json.dump(_json_safe(pairwise_rows), fh, indent=2, allow_nan=False)
    _write_rows_csv(comparison_dir / "seed_level_model_scores.csv",
                    SEED_SCORE_COLUMNS, seed_rows)
    _write_rows_csv(comparison_dir / "model_rank_stability.csv",
                    RANK_STABILITY_COLUMNS, rank_rows)
    with open(comparison_dir / "model_statistics_summary.json", "w",
              encoding="utf-8") as fh:
        json.dump(_json_safe(summary), fh, indent=2, allow_nan=False)
    log(f"Wrote pairwise statistics ({len(pairwise_rows)} pairs) to "
        f"{comparison_dir}")
    return {"summary": summary, "pairwise_rows": pairwise_rows,
            "seed_rows": seed_rows, "rank_rows": rank_rows}


def _write_rows_csv(path: Path, columns, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(columns)
        for r in rows:
            w.writerow([_csv_safe_value(r.get(c, "")) for c in columns])


# ---------------------------------------------------------------------------
# Plots (Part 14)
# ---------------------------------------------------------------------------

def make_plots(suite_state, rows, per_temp_rows, remd_mod, comparison_dir: Path,
               log: Logger, per_temp_diag_rows=None) -> None:
    """Generate per-baseline diagnostic plots (best-effort; never fatal)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        log(f"Plots skipped (matplotlib unavailable): {exc}")
        return

    plots_dir = comparison_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    target_temps = validate_target_npz(suite_state["config"]["target_remd"])

    for baseline in suite_state["config"]["baselines"]:
        bname = baseline["name"]
        raw = baseline["raw_name"]
        brows = [
            r for r in rows
            if r["baseline"] == raw and r.get("fit_status") == "ok"
        ]
        if not brows:
            continue
        models = [r["model"] for r in brows]
        pts = [p for p in per_temp_rows if p["baseline"] == raw]

        def _save(fig, tag):
            out = plots_dir / f"{bname}__{tag}.png"
            fig.tight_layout()
            fig.savefig(out, dpi=140)
            plt.close(fig)

        # 1. Analytic fit validation score by model.
        _bar(plt, models,
             [_pick(r, "fit_validation_contact_loss", "fit_all_contact_loss")
              for r in brows],
             "fit validation contact loss", f"{raw}: analytic fit score",
             _save, "1_fit_validation_score")

        # 2. REMD->target validation score by model with seed error bars.
        fig, ax = plt.subplots(figsize=(7, 4))
        y = [r.get("remd_target_validation_combined_js_mean")
             if r.get("has_validation")
             else r.get("remd_target_all_combined_js_mean") for r in brows]
        yerr = [r.get("remd_target_validation_combined_js_std")
                if r.get("has_validation")
                else r.get("remd_target_all_combined_js_std") for r in brows]
        ax.bar(models, [_plot_value(v) for v in y], yerr=[_nan0(v) for v in yerr],
               capsize=4, color="#4477aa")
        ax.set_ylabel("REMD→target combined JS"); ax.set_title(
            f"{raw}: simulation score (validation when available)")
        ax.tick_params(axis="x", rotation=30)
        _save(fig, "2_remd_validation_score")

        # 3. Mean contacts vs T: target + each model REMD mean.
        _mean_vs_T(plt, pts, target_temps, "remd_mean_contacts",
                   "target_mean_contacts", "mean contacts",
                   f"{raw}: mean contacts vs T", _save, "3_mean_contacts_vs_T")

        # 4. Mean Rg vs T (when available).
        if any(np.isfinite(_f(p.get("remd_mean_rg"))) for p in pts):
            _mean_vs_T(plt, pts, target_temps, "remd_mean_rg", "target_mean_rg",
                       "mean Rg", f"{raw}: mean Rg vs T", _save, "4_mean_rg_vs_T")

        # 5. Per-temperature contact JS curves (target vs REMD).
        _js_vs_T(plt, pts, target_temps, "target_vs_remd_contact_js",
                 "contact JS (target vs REMD)",
                 f"{raw}: per-T contact JS", _save, "5_contact_js_vs_T")

        # 6. Per-temperature Rg JS curves (when available).
        if any(np.isfinite(_f(p.get("target_vs_remd_rg_js"))) for p in pts):
            _js_vs_T(plt, pts, target_temps, "target_vs_remd_rg_js",
                     "Rg JS (target vs REMD)", f"{raw}: per-T Rg JS",
                     _save, "6_rg_js_vs_T")

        # 7. Reduced-bias b(T) curves from fitted summaries.
        _bofT(plt, suite_state, bname, models, remd_mod, target_temps,
              f"{raw}: fitted b(T)", _save, "7_reduced_bias_bT")

        # 8. Swap-acceptance summary (mean swap rate by model).
        _bar(plt, models, [r.get("swap_rate_median_mean") for r in brows],
             "median swap rate", f"{raw}: swap acceptance",
             _save, "8_swap_acceptance")

        # 9. Analytic-fit vs REMD convergence (REMD↔fit contact JS).
        _bar(plt, models, [r.get("remd_fit_contact_js_mean") for r in brows],
             "REMD↔fit contact JS", f"{raw}: REMD reproduces analytic fit?",
             _save, "9_remd_vs_fit_convergence")

        # --- Convergence-diagnostics plots (only when diagnostics ran) ------
        dpts = [p for p in (per_temp_diag_rows or []) if p["baseline"] == raw]
        if dpts:
            # 10. Swap rate vs adjacent-pair temperature (per model).
            _swap_rate_vs_T(plt, suite_state, bname, models, _save,
                            f"{raw}: swap rate vs T", "10_swap_rate_vs_T")
            # 11. ESS (contacts) vs temperature.
            _diag_vs_T(plt, dpts, "ess_contacts_min", "min contact ESS (seeds)",
                       f"{raw}: ESS vs T", _save, "11_ess_vs_T")
            # 12. Integrated autocorrelation time (contacts) vs temperature.
            _diag_vs_T(plt, dpts, "tau_contacts_max",
                       "max contact autocorr time (seeds)",
                       f"{raw}: autocorr time vs T", _save, "12_autocorr_vs_T")
            # 13. Walker round trips by model (total low->high->low).
            _bar(plt, models, [r.get("diag_total_round_trips") for r in brows],
                 "total round trips (all seeds)", f"{raw}: walker round trips",
                 _save, "13_walker_round_trips")
            # 14. Cross-seed rank-normalized split-Rhat (contacts) vs T.
            if any(np.isfinite(_f(p.get("rhat_contacts"))) for p in dpts):
                _diag_vs_T(plt, dpts, "rhat_contacts",
                           "cross-seed split-Rhat (contacts)",
                           f"{raw}: cross-seed Rhat vs T", _save,
                           "14_cross_seed_rhat", hline=1.1)
    log(f"Wrote plots to {plots_dir}")


def _diag_vs_T(plt, dpts, key, ylabel, title, save, tag, hline=None):
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    models = sorted({p["model"] for p in dpts})
    for i, m in enumerate(models):
        s = {}
        for p in dpts:
            if p["model"] != m:
                continue
            v = _f(p.get(key))
            if np.isfinite(v):
                s[float(p["temperature"])] = v
        if s:
            xs = sorted(s)
            ax.plot(xs, [s[x] for x in xs], marker=".",
                    color=_model_color(plt, i, len(models)), label=m)
    if hline is not None:
        ax.axhline(hline, color="gray", lw=0.8, ls="--")
    ax.set_xlabel("temperature"); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.legend(fontsize=8, ncol=2)
    save(fig, tag)


def _swap_rate_vs_T(plt, suite_state, bname, models, save, title, tag):
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for i, model in enumerate(models):
        rec = suite_state["models"].get((bname, model), {})
        rates_by_pair = {}
        for srec in rec.get("seeds", {}).values():
            if srec.get("status") != "ok":
                continue
            rs = (srec.get("run_summary") or {}).get("swap_rates") or []
            for k, v in enumerate(rs):
                fv = _f(v)
                if np.isfinite(fv):
                    rates_by_pair.setdefault(k, []).append(fv)
        if rates_by_pair:
            ks = sorted(rates_by_pair)
            ax.plot(ks, [float(np.mean(rates_by_pair[k])) for k in ks],
                    marker=".", color=_model_color(plt, i, len(models)),
                    label=model)
    ax.axhline(0.05, color="gray", lw=0.8, ls="--")
    ax.set_xlabel("adjacent pair index (low->high T)")
    ax.set_ylabel("mean swap rate (seeds)")
    ax.set_title(title)
    ax.legend(fontsize=8, ncol=2)
    save(fig, tag)


def _nan0(v):
    x = _f(v)
    return 0.0 if not np.isfinite(x) else x


def _plot_value(v):
    x = _f(v)
    return x if np.isfinite(x) else np.nan


def _pick(r, *keys):
    for k in keys:
        v = r.get(k)
        if v is not None and np.isfinite(_f(v)):
            return _f(v)
    return float("nan")


def _bar(plt, models, values, ylabel, title, save, tag):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(models, [_plot_value(v) for v in values], color="#66a61e")
    ax.set_ylabel(ylabel); ax.set_title(title)
    ax.tick_params(axis="x", rotation=30)
    save(fig, tag)


def _model_color(plt, i, n):
    import matplotlib.cm as cm
    return cm.tab10(i % 10)


def _mean_vs_T(plt, pts, temps, remd_key, target_key, ylabel, title, save, tag):
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    models = sorted({p["model"] for p in pts})
    # Target (seed-independent): take from any model's rows.
    tgt = _series_by_T(pts, models[0], target_key) if models else {}
    if tgt:
        xs = sorted(tgt)
        ax.plot(xs, [tgt[x] for x in xs], "k--", lw=2, label="target")
    for i, m in enumerate(models):
        s = _series_by_T(pts, m, remd_key)
        if s:
            xs = sorted(s)
            ax.plot(xs, [s[x] for x in xs], marker="o", ms=3,
                    color=_model_color(plt, i, len(models)), label=m)
    ax.set_xlabel("temperature"); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.legend(fontsize=8, ncol=2)
    save(fig, tag)


def _js_vs_T(plt, pts, temps, key, ylabel, title, save, tag):
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    models = sorted({p["model"] for p in pts})
    for i, m in enumerate(models):
        s = _series_by_T(pts, m, key)
        if s:
            xs = sorted(s)
            ax.plot(xs, [s[x] for x in xs], marker=".", color=_model_color(plt, i, len(models)), label=m)
    ax.set_xlabel("temperature"); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.legend(fontsize=8, ncol=2)
    save(fig, tag)


def _series_by_T(pts, model, key):
    """Mean value of `key` per temperature for `model` (averaged over seeds)."""
    by_t = {}
    for p in pts:
        if p["model"] != model:
            continue
        v = _f(p.get(key))
        if np.isfinite(v):
            by_t.setdefault(float(p["temperature"]), []).append(v)
    return {t: float(np.mean(vs)) for t, vs in by_t.items()}


def _bofT(plt, suite_state, bname, models, remd_mod, temps, title, save, tag):
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    reg = remd_mod.MODEL_REGISTRY
    for i, model in enumerate(models):
        rec = suite_state["models"].get((bname, model), {})
        summ = rec.get("summary")
        if not summ:
            continue
        params = [summ["params"][n] for n in summ["param_names"]]
        Tref = float(
            summ.get("T0", summ["Tref"])
            if model == "heat_capacity" else summ["Tref"]
        )
        Tscale = float(summ["Tscale"])
        raw = reg[model]["raw_b_fn"]
        b = [float(raw(params, float(T), Tref, Tscale)) for T in temps]
        ax.plot(temps, b, marker=".", color=_model_color(plt, i, len(models)),
                label=model)
    ax.axhline(0.0, color="gray", lw=0.8)
    ax.set_xlabel("temperature"); ax.set_ylabel("b(T)"); ax.set_title(title)
    ax.legend(fontsize=8, ncol=2)
    save(fig, tag)


def make_statistics_plots(stats_result, comparison_dir: Path, log: Logger):
    """Plots for paired model statistics (best-effort; never fatal)."""
    if not stats_result:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        log(f"Statistics plots skipped (matplotlib unavailable): {exc}")
        return
    plots_dir = comparison_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    pairwise = stats_result["pairwise_rows"]
    seed_rows = stats_result["seed_rows"]
    rank_rows = stats_result["rank_rows"]

    for raw in sorted({p["baseline"] for p in pairwise}):
        prs = [p for p in pairwise if p["baseline"] == raw]
        models = sorted({p["model_a"] for p in prs} | {p["model_b"] for p in prs})
        if len(models) < 2:
            continue
        idx = {m: i for i, m in enumerate(models)}

        def _matrix(key, antisym):
            M = np.full((len(models), len(models)), np.nan)
            for p in prs:
                i, j = idx[p["model_a"]], idx[p["model_b"]]
                v = _f(p.get(key))
                M[i, j] = v
                if antisym:
                    M[j, i] = -v
                else:
                    M[j, i] = v
            return M

        # 1. Mean-difference heatmap (delta = row - col; negative favors row).
        Md = _matrix("mean_delta", antisym=True)
        _heatmap(plt, Md, models, f"{raw}: mean paired delta (row-col)",
                 "delta (negative favors row model)", plots_dir,
                 f"{raw}__stats_mean_delta_heatmap.png", cmap="coolwarm",
                 center0=True)
        # 2. Practical-equivalence probability heatmap.
        Mp = _matrix("prob_practical_equivalent", antisym=False)
        _heatmap(plt, Mp, models, f"{raw}: P(practical equivalence)",
                 "probability", plots_dir,
                 f"{raw}__stats_practical_equivalence_heatmap.png",
                 cmap="viridis", center0=False)

        # 3. Per-seed model rank plot.
        rrows = [r for r in rank_rows if r["baseline"] == raw]
        srows = [s for s in seed_rows if s["baseline"] == raw]
        seeds = sorted({s["seed"] for s in srows})
        if seeds:
            fig, ax = plt.subplots(figsize=(7.5, 4.5))
            for i, m in enumerate(models):
                pts = {}
                for r in rrows:
                    if r["model"] != m:
                        continue
                    for tok in str(r.get("seed_ranks", "")).split(";"):
                        if ":" in tok:
                            sk, rk = tok.split(":")
                            pts[int(sk)] = float(rk)
                xs = [s for s in seeds if s in pts]
                if xs:
                    ax.plot(xs, [pts[s] for s in xs], marker="o",
                            color=_model_color(plt, i, len(models)), label=m)
            ax.invert_yaxis()
            ax.set_xlabel("seed"); ax.set_ylabel("rank (1 = best)")
            ax.set_title(f"{raw}: per-seed model rank")
            ax.legend(fontsize=8, ncol=2)
            fig.tight_layout()
            fig.savefig(plots_dir / f"{raw}__stats_per_seed_rank.png", dpi=140)
            plt.close(fig)

            # 4. Paired line plot: score per model, one line per seed.
            fig, ax = plt.subplots(figsize=(7.5, 4.5))
            by_seed = {}
            for s in srows:
                by_seed.setdefault(s["seed"], {})[s["model"]] = _f(s["score"])
            xpos = np.arange(len(models))
            for k, sd in enumerate(seeds):
                ys = [by_seed.get(sd, {}).get(m, np.nan) for m in models]
                ax.plot(xpos, ys, marker="o", alpha=0.8,
                        color=_model_color(plt, k, len(seeds)),
                        label=f"seed {sd}")
            ax.set_xticks(xpos); ax.set_xticklabels(models, rotation=30)
            ax.set_ylabel("combined JS score"); ax.set_title(
                f"{raw}: paired per-seed scores")
            ax.legend(fontsize=8, ncol=2)
            fig.tight_layout()
            fig.savefig(plots_dir / f"{raw}__stats_paired_scores.png", dpi=140)
            plt.close(fig)
    log(f"Wrote statistics plots to {plots_dir}")


def _heatmap(plt, M, labels, title, cbar_label, plots_dir, fname, cmap,
             center0):
    fig, ax = plt.subplots(figsize=(1.4 + 0.7 * len(labels),
                                    1.2 + 0.7 * len(labels)))
    kw = {"cmap": cmap}
    if center0:
        vmax = np.nanmax(np.abs(M)) if np.any(np.isfinite(M)) else 1.0
        kw.update({"vmin": -vmax, "vmax": vmax})
    im = ax.imshow(M, **kw)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45,
                                                          ha="right", fontsize=7)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=7)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.2g}", ha="center", va="center",
                        fontsize=6, color="black")
    ax.set_title(title, fontsize=9)
    fig.colorbar(im, ax=ax, label=cbar_label, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(plots_dir / fname, dpi=140)
    plt.close(fig)


def _statistics_report_lines(raw, bstat, stats_result, ok_rows):
    """Restrained statistical-interpretation block for one baseline."""
    pars = bstat.get("parsimony", {})
    n_seeds = bstat.get("min_paired_seeds", 0)
    lines = ["### Statistical model comparison (paired across seeds)", ""]
    if bstat.get("low_power"):
        lines += [
            f"_Only {n_seeds} independent seed(s): inferential power is very "
            f"low. Treat p-values as weak evidence; prefer effect sizes, "
            f"bootstrap uncertainty, and practical equivalence below._", "",
        ]
    lines += [f"- **Raw winner (lowest mean combined JS):** "
              f"{bstat.get('raw_winner')}"]

    supported = bstat.get("statistically_supported_differences", [])
    if supported:
        txt = ", ".join(
            f"{d['model_a']} vs {d['model_b']} (Holm p={_fmt(d['p_holm'])})"
            for d in supported)
        lines += [f"- **Statistically supported differences "
                  f"(Holm p < alpha):** {txt}"]
    else:
        lines += ["- **Statistically supported differences:** none survive "
                  "multiple-testing correction at this sample size"]

    equiv = bstat.get("practical_equivalences", [])
    if equiv:
        txt = ", ".join(
            f"{d['model_a']}~{d['model_b']} "
            f"(P={_fmt(d['prob_practical_equivalent'])})" for d in equiv)
        lines += [f"- **Practically equivalent pairs (P|delta|<eps >= 0.5):** "
                  f"{txt}"]
    else:
        lines += ["- **Practically equivalent pairs:** none"]

    rec = pars.get("recommended_model")
    best = pars.get("best_model")
    lines += [f"- **Parsimonious recommendation:** {rec} "
              f"({pars.get('reason', '')})"]

    # Convergence-qualified recommendation when Priority-11 diagnostics exist.
    conv_by_model = {r["model"]: r.get("convergence_status")
                     for r in ok_rows
                     if r.get("convergence_status") not in (None, "unknown")}
    if conv_by_model:
        rec_status = conv_by_model.get(rec)
        if rec_status == "unreliable":
            reliable = [m for m, s in conv_by_model.items() if s == "reliable"]
            lines += [f"- **Convergence-qualified recommendation:** the "
                      f"parsimonious pick {rec} is convergence-flagged as "
                      f"UNRELIABLE; prefer a convergence-reliable model "
                      f"({', '.join(reliable) if reliable else 'none available'})."]
        else:
            lines += [f"- **Convergence-qualified recommendation:** {rec} "
                      f"passes convergence diagnostics (status="
                      f"{rec_status or 'n/a'})."]
    lines += [""]
    return lines


# ---------------------------------------------------------------------------
# Report (Part 14)
# ---------------------------------------------------------------------------

def write_report(suite_state, rows, global_ranking, comparison_dir: Path,
                 log: Logger, stats_result=None, robustness=None) -> None:
    cfg = suite_state["config"]
    lines = ["# Model suite report", "",
             f"Generated: {now_iso()}", "",
             "## Configuration", "",
             f"- Target REMD: `{cfg['target_remd']}`",
             f"- Models: {', '.join(cfg['models'])}",
             f"- Baselines: {', '.join(b['raw_name'] for b in cfg['baselines'])}",
             f"- Output root: `{cfg['output_root']}`", ""]

    # Jobs summary.
    n_fit_ok = sum(1 for r in rows if r.get("fit_status") == "ok")
    n_fit_fail = sum(1 for r in rows if r.get("fit_status") != "ok")
    seed_ok = sum(r.get("n_successful_seeds", 0) for r in rows)
    seed_fail = sum(r.get("n_failed_seeds", 0) for r in rows)
    lines += ["## Jobs", "",
              f"- Fits succeeded: {n_fit_ok}; fits failed/incomplete: {n_fit_fail}",
              f"- REMD seeds succeeded: {seed_ok}; failed: {seed_fail}", ""]

    for baseline in cfg["baselines"]:
        raw = baseline["raw_name"]
        brows = [r for r in rows if r["baseline"] == raw]
        ok = [r for r in brows if r.get("fit_status") == "ok"]
        lines += [f"## Baseline: {raw}", ""]
        if not ok:
            lines += ["_No successful models._", ""]
            continue

        fit_sorted = sorted(
            [r for r in ok if r.get("fit_rank")], key=lambda r: r["fit_rank"])
        sim_sorted = sorted(
            [r for r in ok if r.get("simulation_rank")],
            key=lambda r: r["simulation_rank"])

        if sim_sorted:
            w = sim_sorted[0]
            sim_metric = (
                w.get("remd_target_validation_combined_js_mean")
                if w.get("has_validation")
                else w.get("remd_target_all_combined_js_mean")
            )
            sim_scope = "validation" if w.get("has_validation") else "all-temperature"
            lines += [f"- **Simulation winner:** {w['model']} "
                      f"({sim_scope} combined JS {_fmt(sim_metric)})"]
        if fit_sorted:
            fw = fit_sorted[0]
            fit_metric = _pick(
                fw, "fit_validation_contact_loss", "fit_all_contact_loss"
            )
            fit_scope = "validation" if fw.get("has_validation") else "all-temperature"
            lines += [f"- **Analytic-fit winner:** {fw['model']} "
                      f"({fit_scope} contact loss {_fmt(fit_metric)})"]
        lines += [""]

        lines += ["### Analytic fit ranking", "",
                  "| rank | model | val contact loss | all contact loss |",
                  "|---|---|---|---|"]
        for r in fit_sorted:
            lines.append(
                f"| {r['fit_rank']} | {r['model']} | "
                f"{_fmt(r.get('fit_validation_contact_loss'))} | "
                f"{_fmt(r.get('fit_all_contact_loss'))} |")
        lines += [""]

        lines += ["### REMD simulation ranking", "",
                  "| rank | model | combined JS (mean±std) | contact JS | Rg JS | note |",
                  "|---|---|---|---|---|---|"]
        for r in sim_sorted:
            if r.get("has_validation"):
                combined_mean = r.get("remd_target_validation_combined_js_mean")
                combined_std = r.get("remd_target_validation_combined_js_std")
                contact_mean = r.get("remd_target_validation_contact_js_mean")
                rg_mean = r.get("remd_target_validation_rg_js_mean")
            else:
                combined_mean = r.get("remd_target_all_combined_js_mean")
                combined_std = r.get("remd_target_all_combined_js_std")
                contact_mean = r.get("remd_target_all_contact_js_mean")
                rg_mean = r.get("remd_target_all_rg_js_mean")
            lines.append(
                f"| {r['simulation_rank']} | {r['model']} | "
                f"{_fmt(combined_mean)}±{_fmt(combined_std)} | "
                f"{_fmt(contact_mean)} | {_fmt(rg_mean)} | "
                f"{r.get('simulation_rank_note', '')} |")
        lines += [""]

        # Diagnostics.
        over = [r["model"] for r in ok if _overfit(r)]
        nonrepro = [r["model"] for r in ok
                    if np.isfinite(_f(r.get("remd_fit_contact_js_mean")))
                    and _f(r.get("remd_fit_contact_js_mean")) > 0.05]
        lowswap = [r["model"] for r in ok
                   if np.isfinite(_f(r.get("swap_rate_min_mean")))
                   and _f(r.get("swap_rate_min_mean")) < 0.05]
        lines += ["### Diagnostics", "",
                  f"- Possible overfitting (val ≫ train fit loss): "
                  f"{', '.join(over) if over else 'none'}",
                  f"- REMD does not reproduce analytic prediction "
                  f"(REMD↔fit contact JS > 0.05): "
                  f"{', '.join(nonrepro) if nonrepro else 'none'}",
                  f"- Convergence warnings (min swap rate < 0.05): "
                  f"{', '.join(lowswap) if lowswap else 'none'}", ""]

        # Convergence diagnostics (only when REMD diagnostics were enabled).
        diag_rows = [r for r in ok if r.get("convergence_status") not in
                     (None, "unknown")]
        if diag_rows:
            unreliable = [r["model"] for r in diag_rows
                          if r.get("convergence_status") == "unreliable"]
            lines += [
                "### Convergence diagnostics", "",
                "_Raw ranks above are unchanged; the status column below "
                "qualifies whether a low JS is trustworthy. A model flagged "
                "**unreliable** failed one or more convergence thresholds and "
                "its score should not be trusted regardless of rank._", "",
                "| model | status | min ESS | max τ | round trips | "
                "min coverage | max drift | max Rhat | flags |",
                "|---|---|---|---|---|---|---|---|---|",
            ]
            for r in diag_rows:
                status = r.get("convergence_status", "unknown")
                badge = ("UNRELIABLE" if status == "unreliable"
                         else "ok" if status == "reliable" else status)
                lines.append(
                    f"| {r['model']} | {badge} | "
                    f"{_fmt(r.get('diag_min_ess_contacts'))} | "
                    f"{_fmt(r.get('diag_max_autocorr_contacts'))} | "
                    f"{_fmt(r.get('diag_total_round_trips'))} | "
                    f"{_fmt(r.get('diag_min_temp_coverage'))} | "
                    f"{_fmt(r.get('diag_max_drift_contacts'))} | "
                    f"{_fmt(r.get('diag_max_rhat_contacts'))} | "
                    f"{r.get('convergence_flags', '')} |"
                )
            lines += [""]
            if unreliable:
                lines += [f"- **Unreliable (convergence-flagged):** "
                          f"{', '.join(unreliable)}", ""]

        # Paired statistical model comparison (only when enabled).
        bstat = (stats_result or {}).get("summary", {}).get(raw) \
            if stats_result else None
        if bstat:
            lines += _statistics_report_lines(raw, bstat, stats_result, ok)

    # Fit robustness and parameter identifiability (only when any analysis ran).
    if robustness and robustness.get("any_enabled"):
        lines += _fit_robustness_report_lines(cfg, rows, robustness)

    if global_ranking:
        lines += ["## Global cross-baseline ranking", "",
                  "_All baselines share contact_offset, Rg units, and metric config._",
                  "", "| rank | baseline | model | combined JS |", "|---|---|---|---|"]
        for g in global_ranking[:20]:
            lines.append(f"| {g['rank']} | {g['baseline']} | {g['model']} | "
                         f"{_fmt(g['combined_js'])} |")
        lines += [""]
    else:
        lines += ["## Global cross-baseline ranking", "",
                  "_Not produced: baselines differ in offset/Rg-units/metric config, "
                  "or only one baseline. Per-baseline rankings above apply._", ""]

    lines += ["## Outputs", "",
              f"- `{comparison_dir / 'model_comparison.csv'}`",
              f"- `{comparison_dir / 'per_temperature_metrics.csv'}`",
              f"- `{comparison_dir / 'plots'}`",
              f"- `{Path(cfg['output_root']) / 'manifest.json'}`"]
    if stats_result:
        lines += [
            f"- `{comparison_dir / 'pairwise_model_comparison.csv'}`",
            f"- `{comparison_dir / 'seed_level_model_scores.csv'}`",
            f"- `{comparison_dir / 'model_rank_stability.csv'}`",
            f"- `{comparison_dir / 'model_statistics_summary.json'}`",
        ]
    if robustness and robustness.get("any_enabled"):
        lines += [
            f"- `{comparison_dir / 'fit_robustness_summary.json'}`",
            f"- `{comparison_dir / 'fit_robustness.csv'}`",
        ]
    lines += [""]

    path = comparison_dir / "report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    log(f"Wrote {path}")


def _fit_robustness_report_lines(cfg, rows, robustness) -> list[str]:
    """Build the fit-robustness report section (restrained language).

    The REMD simulation ranking remains the primary predictive ranking; these
    fitter diagnostics only qualify it. Detailed per-parameter numbers stay in
    the supplementary JSON/CSV outputs.
    """
    structured = robustness.get("structured", {})
    lines = ["## Fit robustness and parameter identifiability", "",
             "_Fitter-side diagnostics. These qualify but never replace the REMD "
             "simulation ranking. Bootstrap intervals are empirical "
             "temperature-resampling ranges, not likelihood standard errors._", ""]
    row_by_key = {(r["baseline"], r["model"]): r for r in rows}

    for baseline in cfg["baselines"]:
        raw = baseline["raw_name"]
        per_model = structured.get(raw, {})
        if not any(
            d.get("bootstrap", {}).get("enabled")
            or d.get("split_sensitivity", {}).get("enabled")
            or d.get("rg_weight_sensitivity", {}).get("enabled")
            or d.get("uncertainty_diagnostics", {}).get("enabled")
            for d in per_model.values()
        ):
            continue
        lines += [f"### Baseline: {raw}", "",
                  "| model | bootstrap success | widest rel CI | max |corr| | "
                  "split stability | Rg-weight | optimizer/identifiability |",
                  "|---|---|---|---|---|---|---|"]
        for model in cfg["models"]:
            d = per_model.get(model)
            if not d:
                continue
            boot = d.get("bootstrap", {})
            split = d.get("split_sensitivity", {})
            rgw = d.get("rg_weight_sensitivity", {})
            unc = d.get("uncertainty_diagnostics", {})
            if boot.get("enabled"):
                ns = boot.get("successful_replicates")
                nr = boot.get("requested_replicates")
                boot_cell = f"{ns}/{nr}" if ns is not None and nr is not None else "n/a"
            else:
                boot_cell = "off"
            rel_ci = boot.get("widest_relative_ci")
            ci_cell = (f"{rel_ci:.2g}" if isinstance(rel_ci, (int, float))
                       and np.isfinite(rel_ci) else
                       ("warn" if boot.get("enabled") and rel_ci is None else "n/a"))
            corr = boot.get("strongest_abs_param_correlation")
            corr_cell = _fmt(corr) if boot.get("enabled") else "off"
            split_cell = split.get("stability_status", "off") if split.get("enabled") else "off"
            rgw_cell = rgw.get("status", "off") if rgw.get("enabled") else "off"
            unc_cell = unc.get("identifiability_status", "off") if unc.get("enabled") else "off"
            lines.append(
                f"| {model} | {boot_cell} | {ci_cell} | {corr_cell} | "
                f"{split_cell} | {rgw_cell} | {unc_cell} |"
            )
        lines += [""]

        # --- Bootstrap uncertainty subsection ---
        boot_models = {m: d["bootstrap"] for m, d in per_model.items()
                       if d.get("bootstrap", {}).get("enabled")}
        if boot_models:
            lines += ["#### Bootstrap uncertainty", ""]
            for model, b in boot_models.items():
                if b.get("status") == "no_successful_replicates":
                    lines.append(f"- **{model}**: all bootstrap replicates failed; "
                                 "uncertainty undetermined.")
                    continue
                bits = []
                incl = b.get("extra_terms_ci_include_zero") or []
                if incl:
                    bits.append(f"CI brackets zero for {', '.join(incl)} "
                                "(term may be unnecessary)")
                pairs = b.get("identifiability_pairs") or []
                if pairs:
                    bits.append("strong parameter correlation(s): " + ", ".join(pairs)
                                + " (possible non-identifiability)")
                mbf = b.get("max_bound_hit_fraction")
                if isinstance(mbf, (int, float)) and np.isfinite(mbf) and mbf >= 0.5:
                    bits.append(f"a parameter hits an optimization bound in "
                                f"{mbf:.0%} of replicates")
                failed = b.get("failed_replicates") or 0
                if failed:
                    bits.append(f"{failed} replicate(s) failed")
                msg = "; ".join(bits) if bits else (
                    "parameters appear well constrained; no intervals bracket "
                    "zero and no strong correlations flagged")
                lines.append(f"- **{model}**: {msg}.")
            lines += [""]

        # --- Validation-split sensitivity subsection ---
        split_models = {m: d["split_sensitivity"] for m, d in per_model.items()
                        if d.get("split_sensitivity", {}).get("enabled")}
        if split_models:
            lines += ["#### Validation-split sensitivity", ""]
            for model, s in split_models.items():
                status = s.get("stability_status", "unknown")
                na, nok = s.get("n_attempted"), s.get("n_succeeded")
                cv = s.get("max_param_cv")
                cv_txt = (f"max parameter CV {cv:.2g}" if isinstance(cv, (int, float))
                          and np.isfinite(cv) else "parameter CV n/a")
                bw = s.get("boundary_hit_warnings") or 0
                extra = f"; {bw} split(s) hit a bound" if bw else ""
                lines.append(
                    f"- **{model}**: {status} across {nok}/{na} successful splits "
                    f"({cv_txt}{extra}); held-out combined loss range "
                    f"{_fmt(s.get('range_heldout_combined_loss'))}.")
            lines += [
                "",
                "_Estimates are considered stable when parameter estimates and "
                "ranking change little across interpolation, blocked-low/-mid/-high, "
                "k-fold, and random holdouts that were enabled._", "",
            ]

        # --- Rg-weight sensitivity subsection ---
        rgw_models = {m: d["rg_weight_sensitivity"] for m, d in per_model.items()
                      if d.get("rg_weight_sensitivity", {}).get("enabled")}
        if rgw_models:
            lines += ["#### Rg-weight sensitivity", ""]
            for model, r in rgw_models.items():
                prod = r.get("production_weight")
                eff = r.get("production_weight_pareto_efficient")
                knee = r.get("knee_weight")
                near = ("lies on the Pareto frontier" if eff else
                        "is Pareto-dominated (a different weight improves both losses)")
                knee_txt = (f"; heuristic knee at weight {_num_str(knee)}"
                            if isinstance(knee, (int, float)) and np.isfinite(knee)
                            else "")
                concl = ("conclusions change materially with the weight"
                         if r.get("weight_sensitive")
                         else "conclusions are robust to the weight")
                lines.append(
                    f"- **{model}**: production weight {_num_str(prod) if prod is not None else 'n/a'} "
                    f"{near}{knee_txt}; {concl}.")
            lines += [""]

    # --- Recommendation qualified by robustness ---
    lines += _robustness_recommendation_lines(cfg, rows)
    return lines


def _robustness_recommendation_lines(cfg, rows) -> list[str]:
    """Robustness-qualified recommendation that preserves the REMD ranking.

    Decision hierarchy: (1) flag unreliable REMD convergence, (2) find models
    practically equivalent to the best REMD score, (3) prefer split-stable ones,
    (4) prefer those without identifiability/boundary warnings, (5) prefer the
    simpler model when predictive differences are negligible.
    """
    lines = ["### Recommendation qualified by robustness", ""]
    for baseline in cfg["baselines"]:
        raw = baseline["raw_name"]
        brows = [r for r in rows if r["baseline"] == raw
                 and r.get("simulation_rank") is not None]
        if not brows:
            continue
        brows.sort(key=lambda r: r["simulation_rank"])
        best = brows[0]

        def _sim(r):
            return _f(r.get("remd_target_validation_combined_js_mean")
                      if r.get("has_validation")
                      else r.get("remd_target_all_combined_js_mean"))

        best_score = _sim(best)
        # Models practically equivalent to the best REMD score (primary ranking).
        equiv = [r for r in brows
                 if np.isfinite(_sim(r)) and np.isfinite(best_score)
                 and abs(_sim(r) - best_score) <= 1e-3]
        if not equiv:
            equiv = [best]

        def _split_stable(r):
            return r.get("split_stability_status") in (None, "stable", "moderate")

        def _no_warn(r):
            if r.get("unc_identifiability_status") == "warn":
                return False
            if r.get("boot_status") == "warn":
                return False
            mbf = _f(r.get("boot_max_bound_hit_fraction"))
            if np.isfinite(mbf) and mbf >= 0.5:
                return False
            return True

        unreliable = best.get("convergence_status") == "unreliable"
        # Apply the hierarchy among the practically-equivalent set.
        pool = [r for r in equiv if r.get("convergence_status") != "unreliable"] or equiv
        stable = [r for r in pool if _split_stable(r)] or pool
        clean = [r for r in stable if _no_warn(r)] or stable
        rec = min(clean, key=lambda r: (_f(r.get("n_parameters")), _sim(r)))

        note = []
        if unreliable:
            note.append("the top REMD model is convergence-flagged UNRELIABLE; treat "
                        "its score with caution")
        if len(equiv) > 1:
            note.append(f"{len(equiv)} model(s) are within ~1e-3 of the best REMD score")
        if rec["model"] != best["model"]:
            note.append(f"among them, **{rec['model']}** is preferred on stability/"
                        "identifiability/parsimony")
        suffix = (" (" + "; ".join(note) + ")") if note else ""
        lines.append(
            f"- **{raw}:** primary REMD pick is **{best['model']}** "
            f"(combined JS {_fmt(best_score)}); robustness-qualified recommendation: "
            f"**{rec['model']}**{suffix}.")
    lines += [""]
    return lines


def _overfit(r) -> bool:
    tr = _f(r.get("fit_train_contact_loss"))
    va = _f(r.get("fit_validation_contact_loss"))
    return np.isfinite(tr) and np.isfinite(va) and tr > 0 and va > 2.0 * tr


def _fmt(v) -> str:
    x = _f(v)
    return f"{x:.4g}" if np.isfinite(x) else "n/a"


# ---------------------------------------------------------------------------
# Manifest (Part 10)
# ---------------------------------------------------------------------------

def write_manifest(suite_state, config_path: str, log: Logger) -> None:
    cfg = suite_state["config"]
    output_root = Path(cfg["output_root"])
    hashes = {}
    for label, path in (
        ("fit_script", cfg["fit_script"]),
        ("remd_script", cfg["remd_script"]),
        ("target_remd", cfg["target_remd"]),
        ("config", config_path),
    ):
        if path and Path(path).exists():
            hashes[label] = sha256_file(path)
    for b in cfg["baselines"]:
        if Path(b["path"]).exists():
            hashes[f"baseline:{b['name']}"] = sha256_file(b["path"])

    fit_by_name = {b["name"]: b["fit"] for b in cfg["baselines"]}
    manifest = {
        "timestamp": now_iso(),
        "python": sys.executable,
        "resolved_config": cfg,
        "hashes": hashes,
        "jobs": suite_state["jobs"],
        "models": {
            f"{bn}/{mn}": {
                "status": rec.get("status"),
                "error": rec.get("error"),
                "fit_dir": rec.get("fit_dir"),
                # Feature-aware required outputs for this model's fit, so the
                # manifest records exactly which files were validated.
                "expected_fit_outputs": list(
                    expected_fit_outputs(fit_by_name.get(bn, {}))
                ),
                "seeds": {
                    str(s): {"status": sr.get("status"),
                             "error": sr.get("error"),
                             "seed_dir": sr.get("seed_dir")}
                    for s, sr in rec.get("seeds", {}).items()
                },
            }
            for (bn, mn), rec in suite_state["models"].items()
        },
    }
    path = output_root / "manifest.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(_json_safe(manifest), fh, indent=2, allow_nan=False)
    log(f"Wrote {path}")


# ---------------------------------------------------------------------------
# Orchestration (Parts 7-10, 15-16)
# ---------------------------------------------------------------------------

class SuiteError(RuntimeError):
    pass


def run_suite(config_path: str, args) -> dict:
    cfg = resolve_config_paths(validate_config(load_config(config_path)), config_path)
    output_root = Path(cfg["output_root"])
    if not args.dry_run:
        output_root.mkdir(parents=True, exist_ok=True)
    log = Logger(output_root / "pipeline.log" if not args.dry_run else None)
    log(f"=== model suite start ({now_iso()}) ===")
    log(f"config={config_path} dry_run={args.dry_run} resume={args.resume} "
        f"force={args.force} continue_on_error={args.continue_on_error}")

    # Preflight (Part 6).
    for label, path in (("fit_script", cfg["fit_script"]),
                        ("remd_script", cfg["remd_script"])):
        if not Path(path).exists():
            raise SuiteError(f"{label} not found: {path}")
        rc = subprocess.run(
            [sys.executable, "-m", "py_compile", path],
            check=False, shell=False, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        if rc.returncode != 0:
            raise SuiteError(f"py_compile failed for {path}:\n{rc.stdout}")
    fit_mod = import_module_from_path("_suite_fit_mod", cfg["fit_script"])
    remd_mod = import_module_from_path("_suite_remd_mod", cfg["remd_script"])
    model_contract = check_model_contracts(fit_mod, remd_mod, log)
    available_models = set(model_contract["models"])
    unknown_models = [m for m in cfg["models"] if m not in available_models]
    if unknown_models:
        raise SuiteError(
            f"Config requests unsupported models {unknown_models}; "
            f"available models are {sorted(available_models)}"
        )

    target_temps = validate_target_npz(cfg["target_remd"])
    log(f"Preflight: target OK ({target_temps.size} temperatures).")
    for b in cfg["baselines"]:
        if not Path(b["path"]).exists():
            raise SuiteError(f"baseline not found: {b['path']}")
        ks = baseline_keyset(b["path"])
        if b["comparison"].get("include_rg", False) and not target_has_rg(
            cfg["target_remd"]
        ):
            raise SuiteError(
                f"comparison.include_rg=true but target lacks Rg histograms "
                f"({cfg['target_remd']})"
            )
        if b["fit"]["fit_rg"]:
            if not target_has_rg(cfg["target_remd"]):
                raise SuiteError(
                    f"fit_rg=true but target lacks Rg histograms ({cfg['target_remd']})"
                )
            if not baseline_is_joint(b["path"]):
                raise SuiteError(
                    f"fit_rg=true requires a joint baseline (c_edges,rg_edges,"
                    f"crg_prob) for {b['name']!r}; found key-set {ks}"
                )
        # Custom split-sensitivity JSON must exist (resolved relative to config).
        ss = b["fit"].get("split_sensitivity") or {}
        if ss.get("enabled") and ss.get("config_json"):
            if not Path(ss["config_json"]).exists():
                raise SuiteError(
                    f"baseline {b['name']!r}: split_sensitivity.config_json not "
                    f"found: {ss['config_json']}"
                )
        # Rg-weight sensitivity requires fit_rg, target Rg data, and a joint baseline.
        rgw = b["fit"].get("rg_weight_sensitivity") or {}
        if rgw.get("enabled"):
            if not b["fit"].get("fit_rg", False):
                raise SuiteError(
                    f"baseline {b['name']!r}: rg_weight_sensitivity requires "
                    f"fit.fit_rg=true"
                )
            if not target_has_rg(cfg["target_remd"]):
                raise SuiteError(
                    f"baseline {b['name']!r}: rg_weight_sensitivity requires target "
                    f"Rg histograms, but {cfg['target_remd']} has none"
                )
            if not baseline_is_joint(b["path"]):
                raise SuiteError(
                    f"baseline {b['name']!r}: rg_weight_sensitivity requires a joint "
                    f"baseline (c_edges,rg_edges,crg_prob); found key-set {ks}"
                )
        log(f"Preflight: baseline {b['name']!r} OK (key-set: {ks}).")

    suite_state = {"config": cfg, "models": {}, "jobs": []}

    # Job matrix (Parts 7-9).
    for baseline in cfg["baselines"]:
        bname = baseline["name"]
        for model in cfg["models"]:
            key = (bname, model)
            model_dir = output_root / bname / model
            fit_dir = model_dir / "fit"
            rec = {"fit_dir": str(fit_dir), "seeds": {}, "status": "pending"}
            suite_state["models"][key] = rec

            fit_cmd = build_fit_command(cfg, baseline, model, fit_dir)
            if args.dry_run:
                log(f"[dry-run] FIT {bname}/{model}: {' '.join(fit_cmd)}")
            else:
                ok = _do_fit(cfg, baseline, model, fit_dir, fit_cmd,
                             rec, suite_state, args, log)
                if not ok:
                    rec["status"] = "fit_failed"
                    if not args.continue_on_error:
                        raise SuiteError(f"fit failed for {bname}/{model}")
                    continue
                rec["status"] = "ok"

            # REMD per seed.
            for seed in baseline["remd"]["seeds"]:
                seed_dir = model_dir / "remd" / f"seed_{seed}"
                srec = {"seed_dir": str(seed_dir), "status": "pending"}
                rec["seeds"][seed] = srec
                remd_cmd = build_remd_command(cfg, baseline, fit_dir, seed,
                                              seed_dir)
                if args.dry_run:
                    log(f"[dry-run] REMD {bname}/{model} seed={seed}: "
                        f"{' '.join(remd_cmd)}")
                    continue
                ok = _do_remd(cfg, baseline, fit_dir, seed, seed_dir, remd_cmd,
                              rec, srec, suite_state, args, log, target_temps)
                if not ok and not args.continue_on_error:
                    raise SuiteError(
                        f"REMD failed for {bname}/{model} seed={seed}"
                    )

    if args.dry_run:
        log("=== dry-run complete (no jobs launched, no outputs written) ===")
        return suite_state

    # Comparison (Part 11 subset) + manifest (Part 10).
    rows, per_temp_rows, per_temp_diag_rows = build_comparison_rows(
        suite_state, log
    )
    comparison_dir = output_root / "comparison"
    # Parse the fitter's supplementary summaries and merge concise robustness
    # findings into the comparison rows BEFORE writing model_comparison.csv.
    robustness = collect_fit_robustness(suite_state, rows, log)
    suite_state["fit_robustness"] = robustness
    write_comparison(rows, per_temp_rows, comparison_dir, log,
                     per_temp_diag_rows=per_temp_diag_rows)
    write_fit_robustness(robustness, comparison_dir, log)
    global_ranking = cross_baseline_global_ranking(suite_state, rows)
    suite_state["global_ranking"] = global_ranking

    # Paired model-comparison statistics (optional; reuses per-seed scores).
    stats_result = run_pairwise_statistics(suite_state, rows, comparison_dir, log)

    make_any_plots = any(
        b["comparison"].get("make_plots", False)
        for b in cfg["baselines"]
    )
    if make_any_plots:
        make_plots(suite_state, rows, per_temp_rows, remd_mod, comparison_dir,
                   log, per_temp_diag_rows=per_temp_diag_rows)
        make_statistics_plots(stats_result, comparison_dir, log)
    write_report(suite_state, rows, global_ranking, comparison_dir, log,
                 stats_result=stats_result, robustness=robustness)
    write_manifest(suite_state, config_path, log)
    log("=== model suite complete ===")
    return suite_state


def _completion_ok(directory: Path, files) -> bool:
    return all((directory / f).exists() for f in files)


def _remove_completion_files(directory: Path, files) -> None:
    """Remove prior completion markers before launching a fresh subprocess.

    Without this, a failed rerun could be incorrectly reported as successful
    because valid-looking files from an older run were still present.
    """
    for name in files:
        path = directory / name
        if path.exists():
            path.unlink()


def _do_fit(cfg, baseline, model, fit_dir, fit_cmd, rec, suite_state,
            args, log) -> bool:
    fit_dir.mkdir(parents=True, exist_ok=True)
    fit_cfg = baseline["fit"]
    completion_files = expected_fit_outputs(fit_cfg)
    signature_path = fit_dir / "suite_job.json"
    # Material inputs hashed into the fit fingerprint. A custom split-sensitivity
    # JSON is an additional input so changing its contents reruns the fit.
    sig_inputs = {
        "fit_script": cfg["fit_script"],
        "target_remd": cfg["target_remd"],
        "baseline": baseline["path"],
    }
    ss = fit_cfg.get("split_sensitivity") or {}
    if ss.get("enabled") and ss.get("config_json"):
        sig_inputs["split_config_json"] = ss["config_json"]
    signature = make_job_signature("fit", fit_cmd, sig_inputs)
    # Resume: skip only if complete AND valid.
    if (
        args.resume and not args.force
        and _completion_ok(fit_dir, completion_files)
        and signature_matches(signature_path, signature)
    ):
        v = validate_fit_outputs(fit_dir, model, fit_cfg)
        if v["status"] == "ok":
            rec["summary"] = v["summary"]
            log(f"Resume: fit {baseline['name']}/{model} already complete.")
            suite_state["jobs"].append({
                "kind": "fit", "baseline": baseline["name"], "model": model,
                "status": "skipped_resume",
            })
            return True
    elif args.resume and not args.force and _completion_ok(fit_dir, completion_files):
        log(f"Resume: fit {baseline['name']}/{model} fingerprint changed; rerunning.")
    _remove_completion_files(fit_dir, completion_files)
    if signature_path.exists():
        signature_path.unlink()
    prov = run_subprocess(fit_cmd, fit_dir / "stdout.log")
    v = (
        validate_fit_outputs(fit_dir, model, fit_cfg)
        if prov["returncode"] == 0
        else {"status": "failed", "error": f"subprocess exited with code {prov['returncode']}"}
    )
    prov.update({"kind": "fit", "baseline": baseline["name"], "model": model,
                 "status": v["status"], "error": v.get("error")})
    suite_state["jobs"].append(prov)
    if v["status"] != "ok":
        rec["error"] = v.get("error")
        log(f"FIT FAILED {baseline['name']}/{model}: {v.get('error')} "
            f"(rc={prov['returncode']})")
        return False
    rec["summary"] = v["summary"]
    write_signature(signature_path, signature)
    log(f"FIT ok {baseline['name']}/{model} "
        f"({prov['elapsed_seconds']:.1f}s)")
    return True


def _do_remd(cfg, baseline, fit_dir, seed, seed_dir, remd_cmd, rec, srec,
             suite_state, args, log, target_temps) -> bool:
    seed_dir.mkdir(parents=True, exist_ok=True)
    summary = rec["summary"]
    tol = float(baseline["comparison"]["temperature_tolerance"])
    diag_cfg = baseline["remd"].get("diagnostics") or {}
    signature_path = seed_dir / "suite_job.json"
    signature = make_job_signature(
        "remd", remd_cmd,
        {
            "remd_script": cfg["remd_script"],
            "target_remd": cfg["target_remd"],
            "fit_summary": str(fit_dir / "fit_summary.json"),
        },
    )
    if (
        args.resume and not args.force
        and _completion_ok(seed_dir, REMD_COMPLETION_FILES)
        and signature_matches(signature_path, signature)
    ):
        v = validate_remd_outputs(seed_dir, summary, target_temps, tol, diag_cfg)
        if v["status"] == "ok":
            srec["status"] = "ok"
            srec["run_summary"] = _load_run_summary(seed_dir)
            log(f"Resume: REMD {baseline['name']}/{summary['model']} "
                f"seed={seed} already complete.")
            suite_state["jobs"].append({
                "kind": "remd", "baseline": baseline["name"],
                "model": summary["model"], "seed": seed,
                "status": "skipped_resume",
            })
            return True
    elif args.resume and not args.force and _completion_ok(seed_dir, REMD_COMPLETION_FILES):
        log(
            f"Resume: REMD {baseline['name']}/{summary['model']} seed={seed} "
            "fingerprint changed; rerunning."
        )
    _remove_completion_files(seed_dir, REMD_COMPLETION_FILES)
    if signature_path.exists():
        signature_path.unlink()
    prov = run_subprocess(remd_cmd, seed_dir / "stdout.log")
    v = (
        validate_remd_outputs(seed_dir, summary, target_temps, tol, diag_cfg)
        if prov["returncode"] == 0
        else {"status": "failed", "error": f"subprocess exited with code {prov['returncode']}"}
    )
    prov.update({"kind": "remd", "baseline": baseline["name"],
                 "model": summary["model"], "seed": seed,
                 "status": v["status"], "error": v.get("error")})
    suite_state["jobs"].append(prov)
    srec["status"] = v["status"]
    srec["error"] = v.get("error")
    if v["status"] != "ok":
        log(f"REMD FAILED {baseline['name']}/{summary['model']} seed={seed}: "
            f"{v.get('error')} (rc={prov['returncode']})")
        return False
    srec["run_summary"] = _load_run_summary(seed_dir)
    write_signature(signature_path, signature)
    log(f"REMD ok {baseline['name']}/{summary['model']} seed={seed} "
        f"({prov['elapsed_seconds']:.1f}s)")
    return True


def _load_run_summary(seed_dir: Path) -> dict:
    path = seed_dir / "run_run_summary.json"
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            return {}
    return {}


# ---------------------------------------------------------------------------
# Quick test (Part 17)
# ---------------------------------------------------------------------------

def _write_synthetic_target(path: Path, temps, m_max=8, n_rg=12, h=300.0, s=1.0):
    """Tiny synthetic joint target with contacts and Rg.

    P(m|T) ∝ P0(m) exp(-m*(h/T - s)); Rg shifts to smaller values at lower T
    (more contacts -> more compact).  Returns (p0, m, rg_centers).
    """
    m = np.arange(0, m_max + 1, dtype=float)
    p0 = np.exp(-0.5 * ((m - m_max / 2.0) / 1.5) ** 2)
    p0 /= p0.sum()
    rg_centers = np.linspace(1.0, 3.0, n_rg)
    rg_widths = np.diff(centers_to_edges(rg_centers))

    ct_hists, rg_hists = [], []
    for T in temps:
        b = h / T - s
        w = p0 * np.exp(-m * b)
        w = w / w.sum()
        ct_hists.append(w)
        mean_m = float(np.sum(m * w))
        rg_mu = 3.0 - 1.5 * (mean_m / m_max)        # compact when many contacts
        rg_mass = np.exp(-0.5 * ((rg_centers - rg_mu) / 0.35) ** 2)
        rg_mass /= rg_mass.sum()
        rg_hists.append(rg_mass / rg_widths)        # store as density (pdf)
    np.savez(
        path,
        temps=np.asarray(temps, dtype=float),
        ct_centers=m,            # contact_offset 0 -> native == shifted
        ct_hists=np.array(ct_hists),
        rg_centers=rg_centers,
        rg_hists=np.array(rg_hists),
    )
    return p0, m, rg_centers


def _write_synthetic_joint_baseline(path: Path, p0, m, rg_centers):
    """Joint athermal baseline P0(m, Rg): keys c_edges, rg_edges, crg_prob."""
    c_edges = centers_to_edges(m)
    rg_edges = centers_to_edges(rg_centers)
    m_max = float(m[-1])
    crg = np.zeros((len(m), len(rg_centers)), dtype=float)
    for i, mi in enumerate(m):
        rg_mu = 3.0 - 1.5 * (mi / m_max)
        prof = np.exp(-0.5 * ((rg_centers - rg_mu) / 0.4) ** 2)
        prof /= prof.sum()
        crg[i] = p0[i] * prof
    crg /= crg.sum()
    np.savez(path, c_edges=c_edges, rg_edges=rg_edges, crg_prob=crg,
             c_vals=np.asarray(m, dtype=int), c_prob=np.asarray(p0, dtype=float))


def _pairwise_statistics_unit_tests() -> None:
    """Synthetic paired-statistics scenarios (clear winner, tie, equivalence,
    missing/failed seed, single paired seed)."""
    n_rep, seed, eps, alpha = 2000, 42, 0.01, 0.05

    # 1. Clear winner: A strictly below B at every seed.
    a = {1: 0.10, 2: 0.11, 3: 0.09}
    b = {1: 0.30, 2: 0.31, 3: 0.29}
    common = sorted(set(a) & set(b))
    ps = compute_pair_stats(a, b, "A", "B", common, n_rep, seed, eps, alpha)
    assert ps["n_paired"] == 3 and ps["mean_delta"] < 0
    assert ps["favored"] == "A" and ps["frac_a_won"] == 1.0
    assert ps["prob_a_better"] > 0.99
    assert ps["ci_high"] < 0  # whole CI favors A
    assert ps["p_method"] == "exact_sign_flip"

    # 2. Exact tie: identical scores -> delta 0, p = 1, all ties.
    t = {1: 0.2, 2: 0.2, 3: 0.2}
    pst = compute_pair_stats(t, dict(t), "A", "B", [1, 2, 3], n_rep, seed,
                             eps, alpha)
    assert pst["mean_delta"] == 0.0 and pst["favored"] == "tie"
    assert pst["frac_tie"] == 1.0 and abs(pst["p_sign_flip"] - 1.0) < 1e-12
    assert abs(pst["prob_practical_equivalent"] - 1.0) < 1e-12

    # 3. Practical equivalence: constant tiny gap below epsilon.
    a3 = {1: 0.100, 2: 0.100, 3: 0.100}
    b3 = {1: 0.100 + eps / 2, 2: 0.100 + eps / 2, 3: 0.100 + eps / 2}
    ps3 = compute_pair_stats(a3, b3, "A", "B", [1, 2, 3], n_rep, seed, eps, alpha)
    assert abs(ps3["mean_delta"]) < eps
    assert ps3["prob_practical_equivalent"] > 0.99

    # 4. Missing seed in one model: pairing only on common seeds.
    am = {1: 0.1, 2: 0.2, 3: 0.3}
    bm = {1: 0.15, 3: 0.35}            # seed 2 absent
    common_m = sorted(set(am) & set(bm))
    psm = compute_pair_stats(am, bm, "A", "B", common_m, n_rep, seed, eps, alpha)
    assert psm["n_paired"] == 2, psm["n_paired"]

    # 5. Failed seed: excluded upstream -> behaves like a missing seed here.
    #    (seed 2 score never recorded for B because that REMD seed failed.)
    assert sorted(set(am) & set(bm)) == [1, 3]

    # 6. Only one paired seed: degenerate but must not crash.
    a1 = {1: 0.10}
    b1 = {1: 0.20}
    ps1 = compute_pair_stats(a1, b1, "A", "B", [1], n_rep, seed, eps, alpha)
    assert ps1["n_paired"] == 1 and ps1["favored"] == "A"
    assert not np.isfinite(ps1["se_delta"])      # SE undefined for n=1
    assert ps1["effect_size_reliable"] is False
    assert abs(ps1["p_sign_flip"] - 1.0) < 1e-12  # n=1 -> p=1

    # Holm monotonicity / clamping.
    adj = _holm_adjust([0.01, 0.04, 0.20])
    assert adj[0] <= adj[1] <= adj[2] and all(x <= 1.0 for x in adj)

    # Rank stability: A is best at every seed -> mean rank 1, high prob_rank1.
    ms = {"A": a, "B": b, "C": {1: 0.5, 2: 0.5, 3: 0.5}}
    rk = compute_rank_stability(ms, ["A", "B", "C"], n_rep, seed)
    rk_by = {r["model"]: r for r in rk}
    assert rk_by["A"]["mean_rank"] == 1.0 and rk_by["A"]["n_seed_wins"] == 3
    assert rk_by["A"]["prob_rank1"] > 0.99

    # Deterministic bootstrap: identical inputs -> identical CI.
    c1 = compute_pair_stats(a, b, "A", "B", common, n_rep, seed, eps, alpha)
    c2 = compute_pair_stats(a, b, "A", "B", common, n_rep, seed, eps, alpha)
    assert c1["ci_low"] == c2["ci_low"] and c1["ci_high"] == c2["ci_high"]
    print("  suite quick-test pairwise-statistics scenarios: PASSED")


def _fit_robustness_unit_tests() -> None:
    """Synthetic coverage for config normalization/validation, command building,
    feature-aware outputs, robustness parsers, signatures, and rank invariance."""
    import tempfile

    def _min_cfg(fit):
        return {
            "target_remd": "t.npz", "fit_script": "f.py", "remd_script": "r.py",
            "output_root": "o", "models": ["hs"],
            "baselines": [{"name": "b", "path": "b.npz", "contact_offset": 0,
                           "fit": fit, "remd": {"N": 4}}],
        }

    def _norm(fit_over):
        cfg = validate_config(_min_cfg(fit_over))
        return cfg["baselines"][0]["fit"]

    # 1. Legacy flat config with all new features off still works and disables them.
    legacy = _norm({"bootstrap": 0, "bootstrap_seed": 7})
    assert legacy["bootstrap"]["enabled"] is False, legacy["bootstrap"]
    assert legacy["bootstrap"]["replicates"] == 0
    assert legacy["bootstrap"]["seed"] == 7  # flat seed folded in
    assert legacy["uncertainty_diagnostics"]["enabled"] is False
    assert legacy["split_sensitivity"]["enabled"] is False
    assert legacy["rg_weight_sensitivity"]["enabled"] is False
    assert expected_fit_outputs(legacy) == FIT_COMPLETION_FILES

    # Legacy scalar bootstrap > 0 enables the analysis (derived enabled).
    legacy_on = _norm({"bootstrap": 50})
    assert legacy_on["bootstrap"]["enabled"] is True
    assert legacy_on["bootstrap"]["replicates"] == 50

    # 2. Nested full-feature config normalizes and validates.
    full = _norm({
        "fit_rg": True, "rg_weight": 0.5,
        "bootstrap": {"enabled": True, "replicates": 5, "seed": 3,
                      "confidence": 0.9, "correlation_threshold": 0.8,
                      "save_prediction_bands": True},
        "uncertainty_diagnostics": {"enabled": True},
        "split_sensitivity": {"enabled": True, "schemes": ["blocked_low", "kfold"],
                              "kfold_k": 3, "seed": 9},
        "rg_weight_sensitivity": {"enabled": True, "weights": [0, 0.5, 1.0],
                                  "normalization_diagnostics": True},
    })
    assert full["bootstrap"]["enabled"] and full["bootstrap"]["replicates"] == 5
    assert full["rg_weight_sensitivity"]["weights"] == [0.0, 0.5, 1.0]
    out = expected_fit_outputs(full)
    for f in (FIT_BOOTSTRAP_FILES + FIT_UNCERTAINTY_FILES + FIT_SPLIT_FILES
              + FIT_RGW_FILES + (FIT_BOOTSTRAP_PREDICTION_BANDS_FILE,)):
        assert f in out, f
    print("  suite quick-test config normalization + expected outputs: PASSED")

    # 3. Validation rejects bad settings (baseline-specific messages).
    def _expect_fail(fit_over, needle):
        try:
            _norm(fit_over)
        except ValueError as exc:
            assert needle in str(exc), (needle, str(exc))
            return
        raise AssertionError(f"expected failure for {fit_over}")

    _expect_fail({"bootstrap": {"enabled": True, "replicates": 0}}, "replicates must be >= 1")
    _expect_fail({"bootstrap": {"enabled": True, "replicates": 5, "confidence": 1.5}},
                 "confidence must be in (0, 1)")
    _expect_fail({"bootstrap": {"enabled": True, "replicates": 5,
                                "correlation_threshold": 0.0}}, "correlation_threshold must be")
    _expect_fail({"split_sensitivity": {"enabled": True, "schemes": ["nope"]}},
                 "unsupported split scheme")
    _expect_fail({"split_sensitivity": {"enabled": True, "kfold_k": 1}}, "kfold_k must be >= 2")
    _expect_fail({"split_sensitivity": {"enabled": True, "random_repeats": 0}},
                 "random_repeats must be >= 1")
    _expect_fail({"split_sensitivity": {"enabled": True, "blocked_fraction": 1.5}},
                 "blocked_fraction must be in (0, 1)")
    _expect_fail({"rg_weight_sensitivity": {"enabled": True, "weights": [-1.0]},
                  "fit_rg": True}, "must all be finite and >= 0")
    _expect_fail({"rg_weight_sensitivity": {"enabled": True, "weights": [0, 1]},
                  "fit_rg": False}, "requires fit.fit_rg=true")
    print("  suite quick-test strict validation rejections: PASSED")

    # 4. build_fit_command forwards exactly the enabled flags.
    cfg = {"fit_script": "f.py", "target_remd": "t.npz"}
    baseline = {"path": "b.npz", "contact_offset": 0, "rg_scale": 1.0, "fit": full}
    cmd = build_fit_command(cfg, baseline, "hs", Path("out"))
    joined = " ".join(cmd)
    for flag in ("--bootstrap 5", "--bootstrap-method temperature",
                 "--bootstrap-confidence 0.9", "--bootstrap-correlation-threshold 0.8",
                 "--bootstrap-seed 3", "--bootstrap-save-prediction-bands",
                 "--uncertainty-diagnostics", "--split-sensitivity",
                 "--split-schemes blocked_low,kfold", "--split-kfold-k 3",
                 "--split-seed 9", "--rg-weight-grid 0,0.5,1",
                 "--rg-weight-normalization-diagnostics"):
        assert flag in joined, flag
    # Disabled analyses append nothing extra.
    cmd_off = " ".join(build_fit_command(cfg, {**baseline, "fit": legacy}, "hs", Path("out")))
    for flag in ("--uncertainty-diagnostics", "--split-sensitivity",
                 "--rg-weight-grid", "--bootstrap-method"):
        assert flag not in cmd_off, flag
    assert "--bootstrap 0" in cmd_off  # legacy contract preserved
    print("  suite quick-test build_fit_command flag forwarding: PASSED")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        # 5. Enabled analysis with a missing required file fails validation.
        fdir = tmp / "fit"
        fdir.mkdir()
        for fn in FIT_COMPLETION_FILES:
            (fdir / fn).write_text("{}" if fn.endswith(".json") else "x")
        # Make fit_summary.json minimally valid so only the bootstrap file is missing.
        (fdir / "fit_summary.json").write_text(json.dumps({
            "model": "hs", "param_names": ["h", "s"], "params": {"h": 1.0, "s": 1.0},
            "Tref": 300.0, "Tscale": 100.0, "optimization_success": True,
        }))
        v = validate_fit_outputs(fdir, "hs", full)
        assert v["status"] == "failed" and "bootstrap_summary.json" in v["error"], v
        v_off = validate_fit_outputs(fdir, "hs", legacy)
        # With analyses off the primary files suffice (npz check will still fail on
        # the dummy, but the bootstrap file is not required).
        assert "bootstrap_summary.json" not in v_off.get("error", "")
        print("  suite quick-test missing-enabled-output rejection: PASSED")

        # 6. Bootstrap parser tolerates nulls + failed replicates + undefined Tc.
        bdir = tmp / "boot"
        bdir.mkdir()
        (bdir / "bootstrap_summary.json").write_text(json.dumps({
            "n_bootstrap": 10, "n_success": 7, "n_failed": 3, "confidence": 0.95,
            "params": {"h": {"ci_low": None, "ci_high": None, "fitted": 1.0},
                       "a2": {"ci_low": -1.0, "ci_high": 1.0, "fitted": 0.0}},
            "derived": {},  # Tc undefined for this model
            "correlation": {"matrix": [[1.0, None], [None, 1.0]],
                            "flagged_pairs": []},
            "param_bound_fractions": {"h": {"at_any": 0.0}, "a2": {"at_any": 0.6}},
        }))
        bp = parse_bootstrap_summary(bdir, {"bootstrap": {"enabled": True}})
        assert bp["successful_replicates"] == 7 and bp["failed_replicates"] == 3
        assert "a2" in bp["extra_terms_ci_include_zero"]
        assert bp["max_bound_hit_fraction"] == 0.6
        assert "Tc_ci" not in bp  # undefined Tc not invented
        assert bp["status"] == "warn"
        # Disabled / missing tolerated.
        assert parse_bootstrap_summary(bdir, {"bootstrap": {"enabled": False}}) == {"enabled": False}
        miss = parse_bootstrap_summary(tmp / "nope", {"bootstrap": {"enabled": True}})
        assert miss["status"] == "missing"
        print("  suite quick-test bootstrap parser tolerance: PASSED")

        # 7. Custom split JSON content is part of the fit signature.
        sj = tmp / "splits.json"
        sj.write_text(json.dumps([{"name": "c", "holdout_indices": [0]}]))
        cmd_sig = ["python", "f.py", "--x"]
        s1 = make_job_signature("fit", cmd_sig,
                                {"a": str(sj)})
        sj.write_text(json.dumps([{"name": "c", "holdout_indices": [1]}]))
        s2 = make_job_signature("fit", cmd_sig, {"a": str(sj)})
        assert s1 != s2, "split JSON content must change the fit signature"
        print("  suite quick-test split-JSON signature sensitivity: PASSED")

    # 8. Robustness scalar fields must not change the primary REMD numeric ranking.
    base_rows = [
        {"baseline": "B", "model": "hs", "fit_status": "ok", "n_successful_seeds": 2,
         "n_parameters": 2, "has_validation": False,
         "fit_all_contact_loss": 0.10, "remd_target_all_combined_js_mean": 0.20},
        {"baseline": "B", "model": "poly3", "fit_status": "ok", "n_successful_seeds": 2,
         "n_parameters": 4, "has_validation": False,
         "fit_all_contact_loss": 0.05, "remd_target_all_combined_js_mean": 0.10},
    ]
    import copy
    rows_plain = copy.deepcopy(base_rows)
    rows_robust = copy.deepcopy(base_rows)
    for r in rows_robust:  # add unrelated robustness fields
        r.update({"boot_status": "warn", "split_stability_status": "sensitive",
                  "rgw_status": "stable", "unc_identifiability_status": "warn"})
    log = Logger(None)
    _assign_ranks(rows_plain, log)
    _assign_ranks(rows_robust, log)
    assert ([r["simulation_rank"] for r in rows_plain]
            == [r["simulation_rank"] for r in rows_robust]), "ranking changed!"
    assert rows_plain[1]["simulation_rank"] == 1  # poly3 has lower JS -> rank 1
    print("  suite quick-test REMD ranking invariance to robustness fields: PASSED")


def run_quick_test() -> None:
    """End-to-end tiny suite run on synthetic data, plus unit checks."""
    import tempfile

    # --- unit checks (fast, no subprocess) ---
    # Contact mass padding without truncation.
    grid = union_int_grid([0, 1, 2], [2, 3, 4])
    assert list(grid) == [0, 1, 2, 3, 4], list(grid)
    row = align_contact_row([2, 3, 4], [0.2, 0.3, 0.5], grid)
    assert abs(row.sum() - 1.0) < 1e-12 and row[0] == 0 and row[1] == 0
    # JS divergence basic properties.
    assert abs(js_divergence([1, 0, 0], [1, 0, 0])) < 1e-12
    assert js_divergence([1, 0], [0, 1]) > 0.99
    # Rg rebinning conserves total probability when grid covers support.
    src_edges = np.array([0.0, 1.0, 2.0, 3.0])
    src_mass = np.array([0.2, 0.5, 0.3])
    dst_edges = np.linspace(0.0, 3.0, 7)   # finer grid spanning the support
    dst_mass = rebin_mass_piecewise(src_edges, src_mass, dst_edges)
    assert abs(dst_mass.sum() - src_mass.sum()) < 1e-12, dst_mass.sum()
    # pdf_to_mass round-trips to unit mass; common-grid covers the union.
    centers = np.linspace(1.0, 3.0, 5)
    assert abs(pdf_to_mass(np.ones_like(centers), centers).sum() - 1.0) < 1e-12
    ce = _common_rg_grid([np.array([0.5, 2.5]), np.array([1.0, 4.0])], 0.5)
    assert ce[0] <= 0.5 and ce[-1] >= 4.0
    # A failed seed must not be scored as zero.
    assert math.isnan(_safe_mean([None, float("nan")]))

    # Rg target-vs-REMD scoring must work with a contact-only analytic fit.
    with tempfile.TemporaryDirectory() as unit_tmp:
        unit_tmp = Path(unit_tmp)
        fit_dir = unit_tmp / "fit"
        seed_dir = unit_tmp / "seed"
        fit_dir.mkdir(); seed_dir.mkdir()
        np.savez(
            fit_dir / "fit_results.npz",
            m_centers=np.array([0.0, 1.0]),
            p_obs_mass=np.array([[0.7, 0.3], [0.4, 0.6]]),
            p_mod_mass=np.array([[0.65, 0.35], [0.45, 0.55]]),
        )
        np.savez(
            seed_dir / "run_distributions.npz",
            c_vals=np.array([0, 1]),
            Pc=np.array([[0.6, 0.4], [0.5, 0.5]]),
            rg_centers=np.array([1.0, 2.0]),
            rg_edges=np.array([0.5, 1.5, 2.5]),
            Prg=np.array([[0.8, 0.2], [0.3, 0.7]]),
        )
        target = unit_tmp / "target.npz"
        np.savez(
            target,
            temps=np.array([280.0, 300.0]),
            ct_centers=np.array([0.0, 1.0]),
            ct_hists=np.array([[0.7, 0.3], [0.4, 0.6]]),
            rg_centers=np.array([1.0, 2.0]),
            rg_hists=np.array([[0.75, 0.25], [0.35, 0.65]]),
        )
        cmp = compare_seed(fit_dir, seed_dir, str(target), include_rg=True)
        assert cmp["rg_available"]
        assert np.all(np.isfinite(cmp["rg_js_tr"]))
        assert np.all(np.isnan(cmp["rg_js_tf"]))

        # Job fingerprints change when material input content changes.
        inp = unit_tmp / "input.dat"
        inp.write_text("a")
        sig1 = make_job_signature("test", ["cmd", "--x"], {"input": str(inp)})
        sig_path = unit_tmp / "sig.json"
        write_signature(sig_path, sig1)
        assert signature_matches(sig_path, sig1)
        inp.write_text("changed")
        sig2 = make_job_signature("test", ["cmd", "--x"], {"input": str(inp)})
        assert sig1 != sig2

    # Cross-seed Rhat: identical/IID chains -> ~1; divergent chains -> >> 1.
    rng = np.random.RandomState(7)
    iid_chains = [rng.standard_normal(500) for _ in range(4)]
    rhat_iid = rank_normalized_split_rhat(iid_chains)
    assert np.isfinite(rhat_iid) and rhat_iid < 1.05, rhat_iid
    shifted = [rng.standard_normal(500) + off for off in (0.0, 5.0, 10.0, 15.0)]
    rhat_bad = rank_normalized_split_rhat(shifted)
    assert rhat_bad > 1.2, rhat_bad
    # Plain split-Rhat agrees on the easy IID case.
    assert split_rhat(iid_chains) < 1.1
    # Average-rank ties match a hand-computed example.
    np.testing.assert_allclose(
        _rankdata_average([10.0, 10.0, 20.0]), [1.5, 1.5, 3.0]
    )

    _pairwise_statistics_unit_tests()
    _fit_robustness_unit_tests()
    print("  suite quick-test unit checks: PASSED")

    here = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        temps = [260.0, 280.0, 300.0, 320.0]
        target = tmp / "target.npz"
        baseline = tmp / "baseline.npz"
        p0, m, rg_centers = _write_synthetic_target(target, temps)
        # Exercise the fitter/suite fallback to the alternate temperature key.
        with np.load(target) as td:
            np.savez(
                tmp / "target_Ts.npz",
                Ts=td["temps"], ct_centers=td["ct_centers"],
                ct_hists=td["ct_hists"], rg_centers=td["rg_centers"],
                rg_hists=td["rg_hists"],
            )
        target = tmp / "target_Ts.npz"
        _write_synthetic_joint_baseline(baseline, p0, m, rg_centers)

        config = {
            "target_remd": str(target),
            "fit_script": str(here / "fit_lattice_contact_model_2.py"),
            "remd_script": str(here / "remd_uniform_chain_2.py"),
            "output_root": str(tmp / "out"),
            "models": list(SUPPORTED_MODELS),
            "baselines": [{
                "name": "synthetic",
                "path": str(baseline),
                "contact_offset": 0,
                "rg_scale": 1.0,
            }],
            "fit": {
                "loss": "js", "fit_rg": False, "n_restarts": 1, "seed": 1,
                "holdout_every": 2, "plots": False,
            },
            "remd": {
                "N": 12, "steps_per_swap": 20, "n_cycles": 16, "rg_bins": 16,
                "burnin_frac": 0.5, "n_workers": 1, "seeds": [1, 2],
                "plots": False,
                "diagnostics": {
                    "enabled": True, "trajectories": True, "n_blocks": 3,
                    "min_round_trips": 1, "min_temp_coverage": 0.5,
                    "min_ess": 5.0, "max_drift": 2.0, "min_swap_rate": 0.01,
                },
            },
            "comparison": {
                "include_rg": True, "rg_weight": 0.25, "make_plots": True,
                "statistics": {
                    "enabled": True, "alpha": 0.05,
                    "bootstrap_replicates": 500, "seed": 12345,
                    "practical_equivalence_epsilon": 0.01,
                    "multiple_testing": "holm",
                },
            },
        }
        config_path = tmp / "config.json"
        config_path.write_text(json.dumps(config, indent=2))

        args = argparse.Namespace(
            config=str(config_path), dry_run=False, resume=False, force=False,
            continue_on_error=True, quick_test=True,
        )
        run_suite(str(config_path), args)

        out = Path(config["output_root"])
        for rel in ("manifest.json", "comparison/model_comparison.csv",
                    "comparison/model_comparison.json",
                    "comparison/per_temperature_metrics.csv",
                    "comparison/report.md"):
            assert (out / rel).exists(), f"missing {rel}"
        plots = list((out / "comparison" / "plots").glob("*.png"))
        assert plots, "no plots produced"

        rows = json.loads((out / "comparison/model_comparison.json").read_text())
        models_seen = {r["model"] for r in rows}
        assert models_seen == set(SUPPORTED_MODELS), models_seen

        # The Rg comparison path must have produced finite Rg JS somewhere.
        pt = (out / "comparison/per_temperature_metrics.csv").read_text()
        assert "target_vs_remd_rg_js" in pt
        rg_finite = any(
            np.isfinite(_f(r.get("remd_target_validation_rg_js_mean")))
            or np.isfinite(_f(r.get("remd_fit_rg_js_mean")))
            for r in rows
        )
        assert rg_finite, "no finite Rg JS computed (Rg path not exercised)"

        # Report mentions every model.
        report = (out / "comparison/report.md").read_text(encoding="utf-8")
        for model in SUPPORTED_MODELS:
            assert model in report, f"report missing {model}"

        # Every successful REMD run's distributions must be normalized.
        n_ok = 0
        for model in SUPPORTED_MODELS:
            dnpz = out / "synthetic" / model / "remd" / "seed_1" / "run_distributions.npz"
            if dnpz.exists():
                with np.load(dnpz, allow_pickle=True) as d:
                    for arr_name in ("Pc", "Prg"):
                        arr = np.asarray(d[arr_name], dtype=float)
                        for r in arr:
                            fin = r[np.isfinite(r)]
                            if fin.size:
                                assert abs(float(fin.sum()) - 1.0) < 1e-6
                n_ok += 1
        assert n_ok == len(SUPPORTED_MODELS), (
            f"only {n_ok}/{len(SUPPORTED_MODELS)} REMD model runs succeeded"
        )
        print(f"  suite quick-test end-to-end ({n_ok}/6 models simulated, "
              f"Rg+plots+report): PASSED")

        # --- Diagnostics outputs (diagnostics enabled in the config above) --
        for model in SUPPORTED_MODELS:
            sd = out / "synthetic" / model / "remd" / "seed_1"
            if not (sd / "run_distributions.npz").exists():
                continue
            for fn in ("run_diagnostics.json", "run_convergence.csv",
                       "run_round_trips.csv", "run_walker_occupancy.csv",
                       "run_diagnostic_trajectories.npz"):
                assert (sd / fn).exists(), f"missing per-seed diagnostic {fn} ({model})"
        # Suite-level cross-seed diagnostics CSV + columns + report section.
        diag_csv = out / "comparison" / "per_temperature_diagnostics.csv"
        assert diag_csv.exists(), "missing per_temperature_diagnostics.csv"
        diag_text = diag_csv.read_text()
        assert "rhat_contacts" in diag_text and "ess_contacts_min" in diag_text
        cmp_text = (out / "comparison/model_comparison.csv").read_text()
        for col in ("convergence_status", "diag_min_ess_contacts",
                    "diag_total_round_trips", "diag_max_rhat_contacts"):
            assert col in cmp_text, f"comparison CSV missing column {col}"
        assert "convergence_status" in {k for r in rows for k in r}, \
            "rows lack convergence_status"
        # A cross-seed Rhat must have been computed for at least one (model, T).
        assert any(np.isfinite(_f(r.get("diag_max_rhat_contacts"))) for r in rows), \
            "no cross-seed Rhat computed"
        assert "Convergence diagnostics" in report, \
            "report missing convergence diagnostics section"
        # Diagnostic plots present.
        dplots = list((out / "comparison" / "plots").glob("*_ess_vs_T.png"))
        assert dplots, "missing ESS-vs-T diagnostic plot"
        print("  suite quick-test diagnostics (per-seed files, cross-seed "
              "Rhat, columns, report, plots): PASSED")

        # --- Paired statistics outputs (statistics enabled in config above) -
        comp = out / "comparison"
        for fn in ("pairwise_model_comparison.csv",
                   "pairwise_model_comparison.json",
                   "seed_level_model_scores.csv",
                   "model_rank_stability.csv",
                   "model_statistics_summary.json"):
            assert (comp / fn).exists(), f"missing statistics output {fn}"
        pw = json.loads((comp / "pairwise_model_comparison.json").read_text())
        assert pw, "no pairwise comparisons produced"
        # Every pair must carry the required scalar fields and a Holm p-value.
        for p in pw:
            for k in ("n_paired", "mean_delta", "ci_low", "ci_high",
                      "p_sign_flip", "p_holm", "prob_a_better",
                      "prob_practical_equivalent", "favored"):
                assert k in p, f"pairwise row missing {k}"
        # n_pairs == C(n_models, 2) for the single synthetic baseline.
        n_models = len({p["model_a"] for p in pw} | {p["model_b"] for p in pw})
        assert len(pw) == n_models * (n_models - 1) // 2, len(pw)
        pw_csv = (comp / "pairwise_model_comparison.csv").read_text()
        for col in ("effect_size_dz", "prob_practical_equivalent", "p_holm"):
            assert col in pw_csv, f"pairwise CSV missing column {col}"
        rk_csv = (comp / "model_rank_stability.csv").read_text()
        assert "prob_rank1" in rk_csv and "n_seed_wins" in rk_csv
        ssum = json.loads((comp / "model_statistics_summary.json").read_text())
        assert ssum, "empty statistics summary"
        bkey = next(iter(ssum))
        assert "parsimony" in ssum[bkey] and "raw_winner" in ssum[bkey]
        assert ssum[bkey]["low_power"] is True  # only 2-3 seeds
        assert "Statistical model comparison" in report, \
            "report missing statistics section"
        splots = list((comp / "plots").glob("*stats_mean_delta_heatmap.png"))
        assert splots, "missing pairwise mean-delta heatmap"
        print("  suite quick-test paired statistics (5 files, columns, report, "
              "plots): PASSED")

    _robustness_end_to_end_quick_test(here)
    print("suite quick-test complete.")


def _robustness_end_to_end_quick_test(here: Path) -> None:
    """Tiny end-to-end run with all newer fitter analyses enabled.

    Exercises feature-aware output validation, supplementary-summary parsing,
    the robustness machine-readable outputs, and the new report section without
    perturbing the primary end-to-end test above. Kept small (few temps, 1
    restart, tiny REMD) so it stays fast.
    """
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        temps = [260.0, 280.0, 300.0, 320.0, 340.0, 360.0]
        target = tmp / "target.npz"
        baseline = tmp / "baseline.npz"
        p0, m, rg_centers = _write_synthetic_target(target, temps)
        _write_synthetic_joint_baseline(baseline, p0, m, rg_centers)

        config = {
            "target_remd": str(target),
            "fit_script": str(here / "fit_lattice_contact_model_2.py"),
            "remd_script": str(here / "remd_uniform_chain_2.py"),
            "output_root": str(tmp / "out"),
            "models": ["hs", "hs_quadratic"],
            "baselines": [{
                "name": "synthetic", "path": str(baseline),
                "contact_offset": 0, "rg_scale": 1.0,
            }],
            "fit": {
                "loss": "js", "fit_rg": True, "rg_weight": 0.5, "n_restarts": 1,
                "seed": 1, "holdout_every": 2, "plots": False,
                "bootstrap": {"enabled": True, "replicates": 4, "seed": 1,
                              "confidence": 0.9, "save_prediction_bands": True},
                "uncertainty_diagnostics": {"enabled": True},
                "split_sensitivity": {"enabled": True,
                                      "schemes": ["blocked_low", "blocked_high", "kfold"],
                                      "kfold_k": 2, "seed": 1},
                "rg_weight_sensitivity": {"enabled": True, "weights": [0, 0.5, 1.0],
                                          "normalization_diagnostics": True},
            },
            "remd": {
                "N": 12, "steps_per_swap": 20, "n_cycles": 16, "rg_bins": 16,
                "burnin_frac": 0.5, "n_workers": 1, "seeds": [1, 2], "plots": False,
            },
            "comparison": {"include_rg": True, "rg_weight": 0.25, "make_plots": False},
        }
        config_path = tmp / "config.json"
        config_path.write_text(json.dumps(config, indent=2))
        args = argparse.Namespace(
            config=str(config_path), dry_run=False, resume=False, force=False,
            continue_on_error=True, quick_test=True,
        )
        run_suite(str(config_path), args)

        out = Path(config["output_root"])
        # Per-model supplementary outputs present and validated.
        for model in ("hs", "hs_quadratic"):
            fdir = out / "synthetic" / model / "fit"
            for fn in (FIT_BOOTSTRAP_FILES + FIT_UNCERTAINTY_FILES
                       + FIT_SPLIT_FILES + FIT_RGW_FILES
                       + (FIT_BOOTSTRAP_PREDICTION_BANDS_FILE,)):
                assert (fdir / fn).exists(), f"missing {fn} for {model}"

        comp = out / "comparison"
        assert (comp / "fit_robustness_summary.json").exists()
        assert (comp / "fit_robustness.csv").exists()
        rsum = json.loads((comp / "fit_robustness_summary.json").read_text())
        assert rsum["any_analysis_enabled"] is True
        bb = rsum["baselines"]["synthetic"]["hs"]["bootstrap"]
        assert bb["enabled"] and bb.get("successful_replicates") is not None

        cmp_csv = (comp / "model_comparison.csv").read_text()
        for col in ("boot_status", "split_stability_status", "rgw_status",
                    "unc_identifiability_status", "boot_successful_replicates"):
            assert col in cmp_csv, f"comparison CSV missing {col}"

        report = (comp / "report.md").read_text(encoding="utf-8")
        for needle in ("## Fit robustness and parameter identifiability",
                       "#### Bootstrap uncertainty",
                       "#### Validation-split sensitivity",
                       "#### Rg-weight sensitivity",
                       "### Recommendation qualified by robustness"):
            assert needle in report, f"report missing {needle!r}"

        # Manifest records the feature-aware expected outputs.
        manifest = json.loads((out / "manifest.json").read_text())
        ent = manifest["models"]["synthetic/hs"]["expected_fit_outputs"]
        assert "bootstrap_summary.json" in ent and "rg_weight_summary.json" in ent

        # Resume must skip everything when nothing changed (feature-aware files OK).
        args_resume = argparse.Namespace(
            config=str(config_path), dry_run=False, resume=True, force=False,
            continue_on_error=True, quick_test=True,
        )
        state = run_suite(str(config_path), args_resume)
        skipped = [j for j in state["jobs"] if j.get("status") == "skipped_resume"]
        assert skipped, "resume skipped nothing despite unchanged feature-aware outputs"
        print("  suite quick-test robustness end-to-end (outputs, parsing, report, "
              "resume): PASSED")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Orchestrate fit + REMD + comparison across models/baselines.",
    )
    ap.add_argument("--config", type=str, default=None,
                    help="Path to the suite configuration JSON.")
    ap.add_argument("--dry-run", action="store_true", dest="dry_run",
                    help="Preflight + print commands; launch nothing, write no outputs.")
    ap.add_argument("--resume", action="store_true",
                    help="Skip only jobs whose outputs exist and pass validation.")
    ap.add_argument("--force", action="store_true",
                    help="Rerun all jobs, overwriting outputs under output_root.")
    ap.add_argument("--continue-on-error", action="store_true",
                    dest="continue_on_error",
                    help="Record failures and continue independent jobs.")
    ap.add_argument("--quick-test", action="store_true", dest="quick_test",
                    help="Run a tiny synthetic end-to-end suite and exit.")
    args = ap.parse_args()

    if args.quick_test:
        run_quick_test()
        return
    if not args.config:
        ap.error("--config is required (or use --quick-test)")
    run_suite(args.config, args)


if __name__ == "__main__":
    main()
