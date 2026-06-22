#!/usr/bin/env python3
"""
run_model_suite.py — subprocess orchestrator for the lattice contact-model suite.

For every baseline x every selected contact-bias model this script:
  1. fits the model to the target REMD distributions (fit_lattice_contact_model.py),
  2. loads the resulting fit_summary.json,
  3. runs one or more lattice REMD replicates (remd_uniform_chain_new.py),
  4. compares simulated outputs to the target with a standardized metric,
  5. writes a per-baseline comparison table and a provenance manifest.

This is an orchestrator: it never re-implements the fitter or the REMD engine.
All work happens through list-form subprocess calls (no shell=True), using
sys.executable, so it runs on Windows.

Stage 1 (this file) implements: config + preflight (model-contract and numeric
b(T) equality, target/baseline validation), fit + REMD execution with strict
output validation, resume/force/continue-on-error, a standardized contact
Jensen-Shannon comparison with mass-conserving grid alignment, the comparison
tables, the manifest, and a self-contained --quick-test.  Per-temperature
tables, the full plot set, and report.md are deferred to Stage 2.

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
REMD_COMPLETION_FILES = (
    "run_results.csv",
    "run_swap_rates.csv",
    "run_distributions.npz",
    "run_run_summary.json",
)

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
        return float(obj)
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


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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

DEFAULT_FIT = {
    "loss": "js", "fit_rg": False, "rg_weight": 1.0, "holdout_every": None,
    "holdout_indices": None, "train_indices": None, "n_restarts": 8,
    "seed": 123, "bootstrap": 0, "bootstrap_seed": None, "plots": False,
}
DEFAULT_REMD = {
    "N": None, "steps_per_swap": 1000, "n_cycles": 5000, "rg_bins": 100,
    "burnin_frac": 0.7, "n_workers": 1, "seeds": [1], "plots": False,
    "timing": False,
}
DEFAULT_COMPARISON = {
    "include_rg": False, "rg_weight": 1.0, "temperature_tolerance": 1e-10,
    "make_plots": False,
}


def load_config(path: str) -> dict:
    with open(path) as fh:
        cfg = json.load(fh)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config {path!r} must be a JSON object")
    return cfg


def _merge(defaults: dict, override: dict | None) -> dict:
    out = dict(defaults)
    if override:
        for k, v in override.items():
            out[k] = v
    return out


def validate_config(cfg: dict) -> dict:
    """Validate required fields and normalize per-baseline overrides."""
    missing = [k for k in REQUIRED_TOP if k not in cfg]
    if missing:
        raise ValueError(f"Config missing required fields: {missing}")

    models = list(cfg["models"])
    if not models:
        raise ValueError("Config 'models' must be non-empty")
    bad = [m for m in models if m not in SUPPORTED_MODELS]
    if bad:
        raise ValueError(f"Unsupported models {bad}; supported: {SUPPORTED_MODELS}")

    baselines = cfg["baselines"]
    if not isinstance(baselines, list) or not baselines:
        raise ValueError("Config 'baselines' must be a non-empty list")

    fit_base = _merge(DEFAULT_FIT, cfg.get("fit"))
    remd_base = _merge(DEFAULT_REMD, cfg.get("remd"))
    cmp_base = _merge(DEFAULT_COMPARISON, cfg.get("comparison"))

    seen_names = set()
    norm_baselines = []
    for b in baselines:
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
            "contact_offset": b["contact_offset"],
            "rg_scale": b.get("rg_scale", 1.0),
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

    # Numeric b(T) equality using the fitter's x0 for each model.
    fit_reg = fit_mod.MODEL_REGISTRY
    remd_reg = remd_mod.MODEL_REGISTRY
    for name in sorted(fit_models):
        x0 = list(fit_reg[name]["x0"])
        fb = fit_reg[name]["raw_b_fn"]
        rb = remd_reg[name]["raw_b_fn"]
        for T in BFN_CHECK_T:
            vf = float(fb(x0, T, BFN_CHECK_TREF, BFN_CHECK_TSCALE))
            vr = float(rb(x0, T, BFN_CHECK_TREF, BFN_CHECK_TSCALE))
            if not np.isclose(vf, vr, rtol=1e-12, atol=1e-12):
                raise ValueError(
                    f"b(T) mismatch for {name!r} at T={T}: fitter={vf!r} "
                    f"remd={vr!r}"
                )
    log(f"Preflight: model contract OK (api v{fit_contract['model_api_version']}, "
        f"{len(fit_models)} models, numeric b(T) equal).")
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
    if temps.ndim != 1 or temps.size < 2:
        raise ValueError(f"target {path!r} temperature ladder is invalid")
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


# ---------------------------------------------------------------------------
# Fit job (Part 8)
# ---------------------------------------------------------------------------

def build_fit_command(cfg, baseline, model, fit_dir: Path) -> list[str]:
    fit = baseline["fit"]
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
        "--bootstrap", str(fit["bootstrap"]),
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
    if fit.get("bootstrap_seed") is not None:
        cmd += ["--bootstrap-seed", str(int(fit["bootstrap_seed"]))]
    return cmd


def _csv_indices(value) -> str:
    if isinstance(value, (list, tuple)):
        return ",".join(str(int(v)) for v in value)
    return str(value)


def validate_fit_outputs(fit_dir: Path, model: str) -> dict:
    """Validate fit completion; return a status dict (status in ok/failed)."""
    for fname in FIT_COMPLETION_FILES:
        if not (fit_dir / fname).exists():
            return {"status": "failed", "error": f"missing {fname}"}
    summary = json.loads((fit_dir / "fit_summary.json").read_text())
    if summary.get("model") != model:
        return {"status": "failed",
                "error": f"summary model {summary.get('model')} != {model}"}
    params = summary.get("params", {})
    if not all(np.isfinite(float(v)) for v in params.values()):
        return {"status": "failed", "error": "non-finite fit parameters"}
    Tref, Tscale = float(summary["Tref"]), float(summary["Tscale"])
    if not np.isfinite(Tref) or not np.isfinite(Tscale) or Tscale <= 0:
        return {"status": "failed", "error": "invalid Tref/Tscale"}
    if summary.get("optimization_success") is False:
        return {"status": "failed", "error": "optimizer reported failure"}
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
    return cmd


def validate_remd_outputs(seed_dir: Path, fit_summary: dict,
                          target_temps: np.ndarray, tol: float) -> dict:
    for fname in REMD_COMPLETION_FILES:
        if not (seed_dir / fname).exists():
            return {"status": "failed", "error": f"missing {fname}"}
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
        if not np.isclose(float(d["Tref"]), float(fit_summary["Tref"]),
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
        Pc = np.asarray(d["Pc"], dtype=float)
        Prg = np.asarray(d["Prg"], dtype=float)
    for tag, arr in (("Pc", Pc), ("Prg", Prg)):
        for i, row in enumerate(arr):
            finite = row[np.isfinite(row)]
            if finite.size and abs(float(finite.sum()) - 1.0) > 1e-6:
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

    # Rg block (Part 11.3): native target Rg + analytic fit Rg + REMD Rg, rebinned
    # onto one mass-conserving common grid that spans the union of all supports.
    rg_ok = (include_rg and rg_fit_centers is not None and rg_mod is not None
             and target_has_rg(target_path))
    if rg_ok:
        with np.load(target_path, allow_pickle=True) as tnpz:
            rc_key = "rg_centers" if "rg_centers" in tnpz else "Rg_centers"
            rh_key = "rg_hists" if "rg_hists" in tnpz else "Rg_hists"
            tg_centers = np.asarray(tnpz[rc_key], dtype=float)
            tg_hists = np.asarray(tnpz[rh_key], dtype=float)
        tg_edges = centers_to_edges(tg_centers)
        f_edges = centers_to_edges(rg_fit_centers)
        common_edges = _common_rg_grid(
            [tg_edges, f_edges, rg_remd_edges], np.mean(np.diff(tg_edges))
        )
        common_centers = 0.5 * (common_edges[:-1] + common_edges[1:])
        out["rg_available"] = True
        for t in range(n_temps):
            tg_m = rebin_mass_piecewise(
                tg_edges, pdf_to_mass(tg_hists[t], tg_centers), common_edges
            )
            f_m = rebin_mass_piecewise(f_edges, rg_mod[t], common_edges)
            r_m = rebin_mass_piecewise(rg_remd_edges, Prg[t], common_edges)
            # Rebinned rows must remain (numerically) normalized.
            for nm, vec in (("tg", tg_m), ("fit", f_m), ("remd", r_m)):
                s = vec.sum()
                assert s == 0 or abs(s - 1.0) < 1e-6, f"rg rebin {nm} sum={s}"
            out["rg_js_tf"][t] = js_divergence(tg_m, f_m)
            out["rg_js_tr"][t] = js_divergence(tg_m, r_m)
            out["rg_js_fr"][t] = js_divergence(f_m, r_m)
            if tg_m.sum() > 0:
                out["target_mean_rg"][t] = float(np.sum(common_centers * tg_m))
            if f_m.sum() > 0:
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


def build_comparison_rows(suite_state, log: Logger):
    """Return (aggregated rows, per-temperature rows) for all completed jobs."""
    cfg = suite_state["config"]
    target_path = cfg["target_remd"]
    target_temps = validate_target_npz(target_path)
    rows, per_temp_rows = [], []

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
            row["remd_target_validation_combined_js_mean"] = _safe_mean(acc["comb_val"])
            row["remd_target_validation_combined_js_std"] = _safe_std(acc["comb_val"])
            row["remd_target_all_combined_js_mean"] = _safe_mean(acc["comb_all"])
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
            baseline_rows.append(row)

        _assign_ranks(baseline_rows, log)
        rows.extend(baseline_rows)
    return rows, per_temp_rows


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

    ok = [r for r in rows if r["status"] == "ok"]
    for i, r in enumerate(sorted(ok, key=_fit_key), start=1):
        r["fit_rank"] = i
    sim_ranked = sorted([r for r in ok if np.isfinite(_sim_key(r))], key=_sim_key)
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
    "remd_target_validation_combined_js_mean",
    "remd_target_validation_combined_js_std",
    "remd_fit_contact_js_mean", "remd_fit_rg_js_mean",
    "mean_contacts_rmse_mean", "mean_contacts_rmse_std",
    "mean_rg_rmse_mean", "mean_rg_rmse_std",
    "swap_rate_min_mean", "swap_rate_median_mean", "local_acceptance_mean",
    "fit_rank", "simulation_rank", "status",
]


def write_comparison(rows, per_temp_rows, comparison_dir: Path,
                     log: Logger) -> None:
    comparison_dir.mkdir(parents=True, exist_ok=True)
    csv_path = comparison_dir / "model_comparison.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(COMPARISON_COLUMNS)
        for r in rows:
            w.writerow([r.get(c, "") for c in COMPARISON_COLUMNS])
    json_path = comparison_dir / "model_comparison.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(_json_safe(rows), fh, indent=2)

    pt_csv = comparison_dir / "per_temperature_metrics.csv"
    with open(pt_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(PER_TEMP_COLUMNS)
        for r in per_temp_rows:
            w.writerow([r.get(c, "") for c in PER_TEMP_COLUMNS])
    pt_json = comparison_dir / "per_temperature_metrics.json"
    with open(pt_json, "w", encoding="utf-8") as fh:
        json.dump(_json_safe(per_temp_rows), fh, indent=2)
    log(f"Wrote {csv_path}, {json_path}, {pt_csv}, {pt_json}")


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
        if r.get("status") != "ok":
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
# Plots (Part 14)
# ---------------------------------------------------------------------------

def make_plots(suite_state, rows, per_temp_rows, remd_mod, comparison_dir: Path,
               log: Logger) -> None:
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
        brows = [r for r in rows if r["baseline"] == raw and r["status"] == "ok"]
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
        yerr = [r.get("remd_target_validation_combined_js_std") or 0.0 for r in brows]
        ax.bar(models, [_nan0(v) for v in y], yerr=[_nan0(v) for v in yerr],
               capsize=4, color="#4477aa")
        ax.set_ylabel("REMD→target combined JS"); ax.set_title(
            f"{raw}: simulation score (validation)")
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
    log(f"Wrote plots to {plots_dir}")


def _nan0(v):
    x = _f(v)
    return 0.0 if not np.isfinite(x) else x


def _pick(r, *keys):
    for k in keys:
        v = r.get(k)
        if v is not None and np.isfinite(_f(v)):
            return _f(v)
    return float("nan")


def _bar(plt, models, values, ylabel, title, save, tag):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(models, [_nan0(v) for v in values], color="#66a61e")
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
        Tref, Tscale = float(summ["Tref"]), float(summ["Tscale"])
        raw = reg[model]["raw_b_fn"]
        b = [float(raw(params, float(T), Tref, Tscale)) for T in temps]
        ax.plot(temps, b, marker=".", color=_model_color(plt, i, len(models)),
                label=model)
    ax.axhline(0.0, color="gray", lw=0.8)
    ax.set_xlabel("temperature"); ax.set_ylabel("b(T)"); ax.set_title(title)
    ax.legend(fontsize=8, ncol=2)
    save(fig, tag)


# ---------------------------------------------------------------------------
# Report (Part 14)
# ---------------------------------------------------------------------------

def write_report(suite_state, rows, global_ranking, comparison_dir: Path,
                 log: Logger) -> None:
    cfg = suite_state["config"]
    lines = ["# Model suite report", "",
             f"Generated: {now_iso()}", "",
             "## Configuration", "",
             f"- Target REMD: `{cfg['target_remd']}`",
             f"- Models: {', '.join(cfg['models'])}",
             f"- Baselines: {', '.join(b['raw_name'] for b in cfg['baselines'])}",
             f"- Output root: `{cfg['output_root']}`", ""]

    # Jobs summary.
    n_fit_ok = sum(1 for r in rows if r["status"] == "ok")
    n_fit_fail = sum(1 for r in rows if r["status"] != "ok")
    seed_ok = sum(r.get("n_successful_seeds", 0) for r in rows)
    seed_fail = sum(r.get("n_failed_seeds", 0) for r in rows)
    lines += ["## Jobs", "",
              f"- Fits succeeded: {n_fit_ok}; fits failed/incomplete: {n_fit_fail}",
              f"- REMD seeds succeeded: {seed_ok}; failed: {seed_fail}", ""]

    for baseline in cfg["baselines"]:
        raw = baseline["raw_name"]
        brows = [r for r in rows if r["baseline"] == raw]
        ok = [r for r in brows if r["status"] == "ok"]
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
            lines += [f"- **Simulation winner:** {w['model']} "
                      f"(combined JS {_fmt(w.get('remd_target_validation_combined_js_mean'))})"]
        if fit_sorted:
            fw = fit_sorted[0]
            lines += [f"- **Analytic-fit winner:** {fw['model']} "
                      f"(val contact loss {_fmt(fw.get('fit_validation_contact_loss'))})"]
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
            lines.append(
                f"| {r['simulation_rank']} | {r['model']} | "
                f"{_fmt(r.get('remd_target_validation_combined_js_mean'))}±"
                f"{_fmt(r.get('remd_target_validation_combined_js_std'))} | "
                f"{_fmt(r.get('remd_target_validation_contact_js_mean'))} | "
                f"{_fmt(r.get('remd_target_validation_rg_js_mean'))} | "
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
              f"- `{Path(cfg['output_root']) / 'manifest.json'}`", ""]

    path = comparison_dir / "report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    log(f"Wrote {path}")


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
        json.dump(_json_safe(manifest), fh, indent=2)
    log(f"Wrote {path}")


# ---------------------------------------------------------------------------
# Orchestration (Parts 7-10, 15-16)
# ---------------------------------------------------------------------------

class SuiteError(RuntimeError):
    pass


def run_suite(config_path: str, args) -> dict:
    cfg = validate_config(load_config(config_path))
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
    check_model_contracts(fit_mod, remd_mod, log)

    target_temps = validate_target_npz(cfg["target_remd"])
    log(f"Preflight: target OK ({target_temps.size} temperatures).")
    for b in cfg["baselines"]:
        if not Path(b["path"]).exists():
            raise SuiteError(f"baseline not found: {b['path']}")
        ks = baseline_keyset(b["path"])
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
    rows, per_temp_rows = build_comparison_rows(suite_state, log)
    comparison_dir = output_root / "comparison"
    write_comparison(rows, per_temp_rows, comparison_dir, log)
    global_ranking = cross_baseline_global_ranking(suite_state, rows)
    suite_state["global_ranking"] = global_ranking
    make_any_plots = any(
        b["comparison"].get("make_plots", False)
        for b in cfg["baselines"]
    )
    if make_any_plots:
        make_plots(suite_state, rows, per_temp_rows, remd_mod, comparison_dir, log)
    write_report(suite_state, rows, global_ranking, comparison_dir, log)
    write_manifest(suite_state, config_path, log)
    log("=== model suite complete ===")
    return suite_state


def _completion_ok(directory: Path, files) -> bool:
    return all((directory / f).exists() for f in files)


def _do_fit(cfg, baseline, model, fit_dir, fit_cmd, rec, suite_state,
            args, log) -> bool:
    fit_dir.mkdir(parents=True, exist_ok=True)
    # Resume: skip only if complete AND valid.
    if args.resume and not args.force and _completion_ok(fit_dir, FIT_COMPLETION_FILES):
        v = validate_fit_outputs(fit_dir, model)
        if v["status"] == "ok":
            rec["summary"] = v["summary"]
            log(f"Resume: fit {baseline['name']}/{model} already complete.")
            suite_state["jobs"].append({
                "kind": "fit", "baseline": baseline["name"], "model": model,
                "status": "skipped_resume",
            })
            return True
    prov = run_subprocess(fit_cmd, fit_dir / "stdout.log")
    v = validate_fit_outputs(fit_dir, model)
    prov.update({"kind": "fit", "baseline": baseline["name"], "model": model,
                 "status": v["status"], "error": v.get("error")})
    suite_state["jobs"].append(prov)
    if v["status"] != "ok":
        rec["error"] = v.get("error")
        log(f"FIT FAILED {baseline['name']}/{model}: {v.get('error')} "
            f"(rc={prov['returncode']})")
        return False
    rec["summary"] = v["summary"]
    log(f"FIT ok {baseline['name']}/{model} "
        f"({prov['elapsed_seconds']:.1f}s)")
    return True


def _do_remd(cfg, baseline, fit_dir, seed, seed_dir, remd_cmd, rec, srec,
             suite_state, args, log, target_temps) -> bool:
    seed_dir.mkdir(parents=True, exist_ok=True)
    summary = rec["summary"]
    tol = float(baseline["comparison"]["temperature_tolerance"])
    if args.resume and not args.force and _completion_ok(seed_dir, REMD_COMPLETION_FILES):
        v = validate_remd_outputs(seed_dir, summary, target_temps, tol)
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
    prov = run_subprocess(remd_cmd, seed_dir / "stdout.log")
    v = validate_remd_outputs(seed_dir, summary, target_temps, tol)
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
    print("  suite quick-test unit checks: PASSED")

    here = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        temps = [260.0, 280.0, 300.0, 320.0]
        target = tmp / "target.npz"
        baseline = tmp / "baseline.npz"
        p0, m, rg_centers = _write_synthetic_target(target, temps)
        _write_synthetic_joint_baseline(baseline, p0, m, rg_centers)

        config = {
            "target_remd": str(target),
            "fit_script": str(here / "fit_lattice_contact_model.py"),
            "remd_script": str(here / "remd_uniform_chain_new.py"),
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
                "N": 12, "steps_per_swap": 20, "n_cycles": 8, "rg_bins": 16,
                "burnin_frac": 0.5, "n_workers": 1, "seeds": [1], "plots": False,
            },
            "comparison": {"include_rg": True, "rg_weight": 0.25,
                           "make_plots": True},
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
        report = (out / "comparison/report.md").read_text()
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
        assert n_ok >= 1, "no successful REMD runs in quick-test"
        print(f"  suite quick-test end-to-end ({n_ok}/6 models simulated, "
              f"Rg+plots+report): PASSED")

    print("suite quick-test complete.")


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
