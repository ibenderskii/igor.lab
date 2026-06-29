#!/usr/bin/env python3
"""Diagnose the lattice->observed Rg mapping and its temperature dependence.

Phase A diagnostic (no production changes).  The production workflow uses a single
multiplicative scalar, ``Rg_observed = rg_scale * Rg_lattice``.  This tool tests
whether that scalar is stable across temperature, and whether a richer mapping
(affine or temperature-linear scale) is warranted by held-out evidence.

What it does
------------
For each supplied ``fit_summary.json`` it reconstructs the predicted *lattice*
P(Rg|T) from the joint baseline P0(m,Rg) and the fitted b(T) (the exact same
physics as the cdfitter, imported from fit_lattice_contact_model_chat).  Then:

  * finds the per-temperature optimal multiplicative scale s(T) that minimizes
    JS divergence to the observed P(Rg|T) after mass-conserving rebinning onto a
    common grid;
  * fits and compares mappings on HELD-OUT temperatures (blocked + interleaved
    splits) so a smooth temperature-dependent mapping is not rewarded merely for
    interpolation:
        - constant multiplicative : Rg_obs = s0 * Rg_lat
        - affine                  : Rg_obs = a  * Rg_lat + c
        - temperature-linear scale: Rg_obs = (s0 + s1*x(T)) * Rg_lat
        - temperature-linear affine (optional, --include-tlinear-affine)
  * bootstraps over temperatures to put confidence intervals on the mapping
    coefficients and tests whether the slope s1 is distinguishable from zero,
    reporting effect size and held-out validation-loss improvement, not only a
    yes/no.

Scientific guardrail
--------------------
A temperature-DEPENDENT geometric scale is physically suspicious: a genuine
lattice->physical unit conversion should be constant.  A nonconstant s(T) more
likely signals model misspecification (e.g. the contact model not capturing the
true compaction).  This tool therefore PREFERS the constant mapping and only
recommends a richer mapping when the held-out improvement is both stable across
splits and practically meaningful.

Resampling assumptions
---------------------
Uncertainty on the mapping coefficients comes from resampling TEMPERATURES with
replacement (temperatures are the independent replicate units here).  Per-T s(T)
error bars are a LOCAL JS-curvature half-width, not a sampling interval.  Bin
(count) resampling is NOT performed: the target provides normalized Rg densities,
not raw independent counts, so treating bins as multinomial counts would be
unjustified (see Phase 7 guardrail).

Dependencies: NumPy, SciPy, optional Matplotlib.  Python 3.8.8 compatible.
Strict JSON (no NaN / Infinity).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    from scipy.optimize import minimize, minimize_scalar
except Exception:  # pragma: no cover
    minimize = None
    minimize_scalar = None

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

# Reuse the fitter's physics so b(T), the joint-baseline Rg prediction, and the
# JS divergence are byte-for-byte identical to production.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
try:
    from fit_lattice_contact_model_2 import (
        MODEL_REGISTRY,
        build_split_schemes,
        centers_to_edges,
        js_div,
        make_b_fn,
        pdf_to_mass,
        predict_rg_from_joint,
        _NpEncoder,
    )
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "analyze_rg_mapping.py must sit next to fit_lattice_contact_model_chat.py "
        f"to reuse its Rg physics. Import failed: {exc}"
    )


# ===========================================================================
# Mass-conserving helpers
# ===========================================================================

def overlap_nonorm(
    source_edges: np.ndarray, source_mass: np.ndarray, target_edges: np.ndarray
) -> np.ndarray:
    """Piecewise-constant overlap integration WITHOUT renormalization.

    Mass outside ``target_edges`` is dropped (so a target grid that covers the
    source preserves total mass).  Used to land both distributions on one grid.

    Exact and fully vectorized: a piecewise-constant density has a piecewise-LINEAR
    CDF whose nodes are (source_edges, cumulative mass).  The mass in any target
    bin [a, b] is CDF(b) - CDF(a), and np.interp evaluates that CDF exactly
    (linear within each source bin), clamping to [0, total] outside the source
    range.  This replaces an O(n*m) Python double loop with C-level interpolation.
    """
    se = np.asarray(source_edges, dtype=float)
    sm = np.asarray(source_mass, dtype=float)
    te = np.asarray(target_edges, dtype=float)
    cdf = np.empty(se.size, dtype=float)
    cdf[0] = 0.0
    np.cumsum(sm, out=cdf[1:])
    F = np.interp(te, se, cdf)  # clamps to cdf[0]=0 (below) and cdf[-1]=total (above)
    return np.diff(F)


def js_on_common_grid(
    obs_mass: np.ndarray,
    obs_edges: np.ndarray,
    pred_mass: np.ndarray,
    pred_edges: np.ndarray,
) -> float:
    """JS divergence between observed and mapped-predicted Rg on a common grid.

    The common grid is the observed bin edges plus two sentinel bins capturing any
    mass that the mapping pushes below/above the observed range, so misalignment
    is penalized rather than silently renormalized away.
    """
    lo = min(float(obs_edges[0]), float(pred_edges[0])) - 1e-9
    hi = max(float(obs_edges[-1]), float(pred_edges[-1])) + 1e-9
    common = np.concatenate([[lo], np.asarray(obs_edges, dtype=float), [hi]])
    obs_c = overlap_nonorm(obs_edges, obs_mass, common)
    pred_c = overlap_nonorm(pred_edges, pred_mass, common)
    return js_div(obs_c, pred_c)


# ===========================================================================
# Mapping registry
# ===========================================================================
# Each mapping turns lattice Rg bin edges into observed-unit edges given params
# and x(T).  Returns mapped edges; the predicted mass per bin is unchanged.

def _map_multiplicative(p, x, edges):
    return p[0] * edges


def _map_affine(p, x, edges):
    return p[0] * edges + p[1]


def _map_tlinear_scale(p, x, edges):
    return (p[0] + p[1] * x) * edges


def _map_tlinear_affine(p, x, edges):
    return (p[0] + p[1] * x) * edges + (p[2] + p[3] * x)


MAPPINGS: Dict[str, Dict[str, Any]] = {
    "multiplicative": {"n": 1, "apply": _map_multiplicative,
                       "param_names": ["s0"]},
    "affine": {"n": 2, "apply": _map_affine, "param_names": ["a", "c"]},
    "tlinear_scale": {"n": 2, "apply": _map_tlinear_scale,
                      "param_names": ["s0", "s1"]},
    "tlinear_affine": {"n": 4, "apply": _map_tlinear_affine,
                       "param_names": ["a0", "a1", "c0", "c1"]},
}


def _x0_for(kind: str, scale_guess: float) -> np.ndarray:
    if kind == "multiplicative":
        return np.array([scale_guess])
    if kind == "affine":
        return np.array([scale_guess, 0.0])
    if kind == "tlinear_scale":
        return np.array([scale_guess, 0.0])
    if kind == "tlinear_affine":
        return np.array([scale_guess, 0.0, 0.0, 0.0])
    raise ValueError(f"unknown mapping {kind!r}")


def _finite_or_none(x: Any) -> Optional[float]:
    if x is None:
        return None
    xf = float(x)
    return xf if np.isfinite(xf) else None


def _sanitize_label(label: str) -> str:
    """Make a label safe for filenames / NPZ keys on all platforms.

    Notably strips ':' which is illegal on Windows (NTFS alternate data streams).
    """
    out = label
    for ch in (":", "/", "\\", " ", "*", "?", '"', "<", ">", "|"):
        out = out.replace(ch, "_")
    return out


# ===========================================================================
# Core scoring
# ===========================================================================

def mapping_total_js(
    kind: str,
    params: np.ndarray,
    idx: np.ndarray,
    xT: np.ndarray,
    obs_mass: np.ndarray,
    obs_edges: np.ndarray,
    pred_mass: np.ndarray,
    rg_edges_lat: np.ndarray,
) -> float:
    """Sum of per-temperature JS over the given temperature indices."""
    apply = MAPPINGS[kind]["apply"]
    total = 0.0
    for i in idx:
        me = apply(params, float(xT[i]), rg_edges_lat)
        if not np.all(np.diff(me) > 0):
            return 1e6
        total += js_on_common_grid(obs_mass[i], obs_edges, pred_mass[i], me)
    return total


def fit_mapping(
    kind: str,
    train_idx: np.ndarray,
    xT: np.ndarray,
    obs_mass: np.ndarray,
    obs_edges: np.ndarray,
    pred_mass: np.ndarray,
    rg_edges_lat: np.ndarray,
    scale_guess: float,
) -> Tuple[np.ndarray, float]:
    """Fit a mapping by minimizing train-set JS (deterministic Nelder-Mead)."""
    if minimize is None:
        raise RuntimeError("scipy is required.")
    x0 = _x0_for(kind, scale_guess)

    def obj(p):
        return mapping_total_js(kind, p, train_idx, xT, obs_mass, obs_edges,
                                pred_mass, rg_edges_lat)

    best_p = None
    best_f = float("inf")
    starts = [x0]
    if kind != "multiplicative":
        x0b = x0.copy(); x0b[0] *= 1.1
        starts.append(x0b)
    for s0 in starts:
        res = minimize(obj, s0, method="Nelder-Mead",
                       options={"xatol": 1e-6, "fatol": 1e-9, "maxiter": 1200})
        if np.isfinite(res.fun) and res.fun < best_f:
            best_f = float(res.fun)
            best_p = np.asarray(res.x, dtype=float)
    if best_p is None:
        raise RuntimeError(f"mapping fit failed for {kind!r}")
    return best_p, best_f


def per_temperature_optimal_scale(
    xT: np.ndarray,
    obs_mass: np.ndarray,
    obs_edges: np.ndarray,
    pred_mass: np.ndarray,
    rg_edges_lat: np.ndarray,
    s_lo: float,
    s_hi: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-T optimal multiplicative s, its JS at the optimum, and a local
    JS-curvature half-width (heuristic, not a sampling interval)."""
    n = len(obs_mass)
    s_opt = np.empty(n, dtype=float)
    js_opt = np.empty(n, dtype=float)
    sigma = np.empty(n, dtype=float)
    # Coarse grid to bracket the (often narrow) JS minimum before refining.  A
    # plain bounded-Brent search rails to the boundary because JS is flat (≈ln2)
    # over the non-overlapping region, so the bracket-then-refine approach is
    # required for robustness.
    grid = np.linspace(s_lo, s_hi, 161)
    for i in range(n):
        def f(s):
            me = s * rg_edges_lat
            return js_on_common_grid(obs_mass[i], obs_edges, pred_mass[i], me)
        gvals = np.array([f(s) for s in grid])
        k = int(np.argmin(gvals))
        a = grid[max(0, k - 1)]
        b = grid[min(grid.size - 1, k + 1)]
        s = float(grid[k])
        jmin = float(gvals[k])
        if b > a:
            res = minimize_scalar(f, bounds=(a, b), method="bounded",
                                  options={"xatol": 1e-9})
            if np.isfinite(res.fun) and res.fun <= jmin:
                s = float(res.x)
                jmin = float(res.fun)
        s_opt[i] = s
        js_opt[i] = jmin
        # local curvature half-width: s-range over which JS rises by delta
        h = max(1e-4, 1e-3 * s)
        f0 = f(s)
        fp = f(s + h)
        fm = f(s - h)
        curv = (fp - 2.0 * f0 + fm) / (h * h)
        delta = 1e-3
        sigma[i] = float(np.sqrt(2.0 * delta / curv)) if curv > 0 else float("nan")
    return s_opt, js_opt, sigma


# ===========================================================================
# fit_summary -> predicted lattice P(Rg|T)
# ===========================================================================

def load_fit_summary(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        s = json.load(fh)
    for k in ("model", "param_names", "params", "Tref", "Tscale"):
        if k not in s:
            raise ValueError(f"fit_summary {path!r} missing required key {k!r}")
    if s["model"] not in MODEL_REGISTRY:
        raise ValueError(f"fit_summary {path!r} has unknown model {s['model']!r}")
    return s


def predicted_lattice_rg(
    summary: Dict[str, Any],
    temps: np.ndarray,
    crg_prob: np.ndarray,
    c_edges: np.ndarray,
    rg_edges_lat: np.ndarray,
) -> np.ndarray:
    """Reconstruct predicted lattice P(Rg|T) (mass per lattice bin) for all T."""
    model = summary["model"]
    pnames = list(summary["param_names"])
    params = np.array([float(summary["params"][p]) for p in pnames], dtype=float)
    b_fn = make_b_fn(model, float(summary["Tref"]), float(summary["Tscale"]))
    _, rg_mass = predict_rg_from_joint(crg_prob, c_edges, rg_edges_lat, temps,
                                       params, b_fn)
    return rg_mass


# ===========================================================================
# Bootstrap over temperatures
# ===========================================================================

def bootstrap_mappings(
    kinds: List[str],
    base_idx: np.ndarray,
    xT: np.ndarray,
    obs_mass: np.ndarray,
    obs_edges: np.ndarray,
    pred_mass: np.ndarray,
    rg_edges_lat: np.ndarray,
    scale_guess: float,
    n_boot: int,
    seed: int,
) -> Dict[str, np.ndarray]:
    """Resample temperatures (with replacement) and refit each mapping.

    Returns {kind: array (n_success, n_params)} of bootstrap coefficients.
    """
    rng = np.random.default_rng(seed)
    out: Dict[str, List[np.ndarray]] = {k: [] for k in kinds}
    n = base_idx.size
    for _ in range(n_boot):
        samp = base_idx[rng.integers(0, n, size=n)]
        for k in kinds:
            try:
                p, _ = fit_mapping(k, samp, xT, obs_mass, obs_edges, pred_mass,
                                   rg_edges_lat, scale_guess)
                out[k].append(p)
            except Exception:
                continue
    return {k: (np.array(v) if v else np.zeros((0, MAPPINGS[k]["n"]))) for k, v in out.items()}


def _ci(arr: np.ndarray, confidence: float = 0.95) -> Tuple[Optional[float], Optional[float]]:
    if arr.size == 0:
        return None, None
    a = 1.0 - confidence
    return (float(np.percentile(arr, 100 * a / 2)),
            float(np.percentile(arr, 100 * (1 - a / 2))))


# ===========================================================================
# Per-model analysis
# ===========================================================================

def analyze_one_model(
    label: str,
    summary: Dict[str, Any],
    temps: np.ndarray,
    obs_mass: np.ndarray,
    obs_edges: np.ndarray,
    obs_centers: np.ndarray,
    crg_prob: np.ndarray,
    c_edges: np.ndarray,
    rg_edges_lat: np.ndarray,
    splits: List[Dict[str, Any]],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    n = temps.size
    rg_centers_lat = 0.5 * (rg_edges_lat[:-1] + rg_edges_lat[1:])

    pred_mass = predicted_lattice_rg(summary, temps, crg_prob, c_edges, rg_edges_lat)

    # x(T) for the mapping (independent of the model's b(T) Tref/Tscale).
    Tref = float(temps.mean())
    Tscale = float(temps.max() - temps.min()) or 1.0
    xT = (temps - Tref) / Tscale
    x_span = float(xT.max() - xT.min())

    # mean Rg
    obs_mean_rg = (obs_centers[None, :] * obs_mass).sum(axis=1)
    pred_mean_lat = (rg_centers_lat[None, :] * pred_mass).sum(axis=1)
    scale_guess = float(np.mean(obs_mean_rg) / np.mean(pred_mean_lat))

    # per-T optimal scale
    s_opt, js_opt, s_sigma = per_temperature_optimal_scale(
        xT, obs_mass, obs_edges, pred_mass, rg_edges_lat,
        args.scale_lo, args.scale_hi
    )

    poor_overlap = bool(np.median(js_opt) > args.poor_overlap_js)

    kinds = ["multiplicative", "affine", "tlinear_scale"]
    if args.include_tlinear_affine:
        kinds.append("tlinear_affine")

    all_idx = np.arange(n)

    # All-temperature point estimates (for coefficient reporting).
    all_fits: Dict[str, Dict[str, Any]] = {}
    for k in kinds:
        p, f = fit_mapping(k, all_idx, xT, obs_mass, obs_edges, pred_mass,
                           rg_edges_lat, scale_guess)
        all_fits[k] = {"params": p, "js_sum": f, "js_mean": f / n}

    # Bootstrap coefficients over all temperatures.
    boot = bootstrap_mappings(kinds, all_idx, xT, obs_mass, obs_edges, pred_mass,
                              rg_edges_lat, scale_guess, args.bootstrap, args.seed)
    boot_ci: Dict[str, Dict[str, Any]] = {}
    for k in kinds:
        pn = MAPPINGS[k]["param_names"]
        arr = boot[k]
        cis = {}
        for j, name in enumerate(pn):
            col = arr[:, j] if arr.size else np.array([])
            lo, hi = _ci(col, args.confidence)
            cis[name] = {
                "median": float(np.median(col)) if col.size else None,
                "ci_low": lo, "ci_high": hi,
                "n": int(col.size),
            }
        boot_ci[k] = cis

    # Slope s1 test (temperature-linear scale).
    s1_boot = boot["tlinear_scale"][:, 1] if boot["tlinear_scale"].size else np.array([])
    s1_lo, s1_hi = _ci(s1_boot, args.confidence)
    s0_med = (np.median(boot["tlinear_scale"][:, 0])
              if boot["tlinear_scale"].size else float("nan"))
    s1_med = float(np.median(s1_boot)) if s1_boot.size else float("nan")
    slope_excludes_zero = bool(
        s1_lo is not None and s1_hi is not None and (s1_lo > 0 or s1_hi < 0)
    )
    # effect size: relative change of scale across the temperature range.
    effect_size = (abs(s1_med) * x_span / abs(s0_med)
                   if np.isfinite(s1_med) and abs(s0_med) > 1e-12 else float("nan"))

    # Held-out split comparison (blocked + interleaved).
    split_rows: List[Dict[str, Any]] = []
    per_split_valmean: Dict[str, List[float]] = {k: [] for k in kinds}
    for sp in splits:
        tr, va = sp["train_idx"], sp["val_idx"]
        for k in kinds:
            p, train_js = fit_mapping(k, tr, xT, obs_mass, obs_edges, pred_mass,
                                      rg_edges_lat, scale_guess)
            val_js = mapping_total_js(k, p, va, xT, obs_mass, obs_edges,
                                      pred_mass, rg_edges_lat)
            all_js = mapping_total_js(k, p, all_idx, xT, obs_mass, obs_edges,
                                      pred_mass, rg_edges_lat)
            row = {
                "model": label, "scheme": sp["scheme"], "split": sp["name"],
                "mapping": k, "n_params": MAPPINGS[k]["n"],
                "n_train": int(tr.size), "n_val": int(va.size),
                "train_js_sum": train_js, "train_js_mean": train_js / tr.size,
                "val_js_sum": val_js,
                "val_js_mean": val_js / va.size if va.size else float("nan"),
                "all_js_sum": all_js, "all_js_mean": all_js / n,
            }
            for nm, pv in zip(MAPPINGS[k]["param_names"], p):
                row["param_" + nm] = float(pv)
            split_rows.append(row)
            if va.size:
                per_split_valmean[k].append(val_js / va.size)

    # Held-out improvement of richer mappings vs constant (mean over splits).
    const_val = np.array(per_split_valmean["multiplicative"], dtype=float)
    improvements: Dict[str, Dict[str, Any]] = {}
    for k in kinds:
        if k == "multiplicative":
            continue
        kv = np.array(per_split_valmean[k], dtype=float)
        if const_val.size and kv.size == const_val.size:
            delta = const_val - kv  # positive => richer mapping better
            rel = delta / np.where(const_val > 0, const_val, np.nan)
            improvements[k] = {
                "val_js_mean_improvement_per_split": [float(x) for x in delta],
                "mean_improvement": float(np.mean(delta)),
                "min_improvement": float(np.min(delta)),
                "mean_relative_improvement": float(np.nanmean(rel)),
                "improves_all_splits": bool(np.all(delta > 0)),
            }

    # ---- verdict (guardrail: prefer constant) ----
    tl = improvements.get("tlinear_scale", {})
    stable_and_meaningful = bool(
        slope_excludes_zero
        and tl.get("improves_all_splits", False)
        and tl.get("mean_relative_improvement", 0.0) >= args.meaningful_rel_improvement
    )
    recommendation = (
        "temperature-linear scale" if stable_and_meaningful else "constant multiplicative"
    )

    return {
        "label": label,
        "model": summary["model"],
        "Tref_mapping": Tref, "Tscale_mapping": Tscale,
        "scale_guess": scale_guess,
        "poor_overlap": poor_overlap,
        "median_js_at_optimal_scale": float(np.median(js_opt)),
        "temps": temps, "xT": xT,
        "s_opt": s_opt, "js_opt": js_opt, "s_sigma": s_sigma,
        "obs_mean_rg": obs_mean_rg, "pred_mean_lat": pred_mean_lat,
        "pred_mass": pred_mass, "rg_centers_lat": rg_centers_lat,
        "all_fits": all_fits, "boot_ci": boot_ci, "kinds": kinds,
        "s1_median": s1_med, "s1_ci": [s1_lo, s1_hi],
        "slope_excludes_zero": slope_excludes_zero,
        "effect_size_rel_scale_change": effect_size,
        "improvements": improvements,
        "split_rows": split_rows,
        "stable_and_meaningful": stable_and_meaningful,
        "recommendation": recommendation,
    }


# ===========================================================================
# Output writers
# ===========================================================================

def _cell(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return str(int(v))
    if isinstance(v, float):
        return "%.10g" % v if np.isfinite(v) else ""
    return str(v)


def write_outputs(outdir: Path, results: List[Dict[str, Any]],
                  args: argparse.Namespace, production_scalar: float) -> None:
    # rg_scale_by_temperature.csv
    p = outdir / "rg_scale_by_temperature.csv"
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "temp_index", "temperature", "x_T", "s_opt",
                    "s_local_sigma", "js_at_opt", "obs_mean_rg", "pred_mean_lat",
                    "production_scalar"])
        for r in results:
            for i in range(r["temps"].size):
                w.writerow([r["label"], i, _cell(float(r["temps"][i])),
                            _cell(float(r["xT"][i])), _cell(float(r["s_opt"][i])),
                            _cell(float(r["s_sigma"][i])), _cell(float(r["js_opt"][i])),
                            _cell(float(r["obs_mean_rg"][i])),
                            _cell(float(r["pred_mean_lat"][i])),
                            _cell(production_scalar)])
    print(f"Saved: {p}")

    # rg_mapping_model_comparison.csv
    p = outdir / "rg_mapping_model_comparison.csv"
    extra_cols: List[str] = []
    for r in results:
        for row in r["split_rows"]:
            for key in row:
                if key.startswith("param_") and key not in extra_cols:
                    extra_cols.append(key)
    base = ["model", "scheme", "split", "mapping", "n_params", "n_train", "n_val",
            "train_js_sum", "train_js_mean", "val_js_sum", "val_js_mean",
            "all_js_sum", "all_js_mean"]
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(base + extra_cols)
        for r in results:
            for row in r["split_rows"]:
                w.writerow([_cell(row.get(c)) for c in base]
                           + [_cell(row.get(c)) for c in extra_cols])
    print(f"Saved: {p}")

    # rg_mapping_predictions.npz
    p = outdir / "rg_mapping_predictions.npz"
    npz: Dict[str, Any] = {"production_scalar": np.array(production_scalar)}
    for r in results:
        lab = _sanitize_label(r["label"])
        npz[f"{lab}__temps"] = r["temps"]
        npz[f"{lab}__s_opt"] = r["s_opt"]
        npz[f"{lab}__s_sigma"] = r["s_sigma"]
        npz[f"{lab}__js_opt"] = r["js_opt"]
        npz[f"{lab}__obs_mean_rg"] = r["obs_mean_rg"]
        npz[f"{lab}__pred_mean_lat"] = r["pred_mean_lat"]
        npz[f"{lab}__pred_mass"] = r["pred_mass"]
        npz[f"{lab}__rg_centers_lat"] = r["rg_centers_lat"]
    np.savez_compressed(p, **npz)
    print(f"Saved: {p}")

    # rg_mapping_summary.json
    p = outdir / "rg_mapping_summary.json"
    summary = {
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "production_scalar": float(production_scalar),
        "bootstrap": int(args.bootstrap),
        "seed": int(args.seed),
        "confidence": float(args.confidence),
        "split_schemes": args.split_schemes,
        "meaningful_rel_improvement_threshold": float(args.meaningful_rel_improvement),
        "guardrail": (
            "A temperature-dependent geometric scale may indicate model "
            "misspecification rather than a real unit conversion. The constant "
            "multiplicative mapping is preferred unless a richer mapping improves "
            "held-out Rg JS on every split and by a practically meaningful margin."
        ),
        "resampling_assumptions": (
            "Coefficient uncertainty from resampling temperatures with replacement "
            "(temperatures are the replicate units). Per-T s(T) error bars are a "
            "local JS-curvature half-width, not a sampling interval. Histogram-bin "
            "resampling not performed: target Rg is a normalized density, not raw "
            "counts."
        ),
        "models": [],
    }
    for r in results:
        summary["models"].append({
            "label": r["label"], "model": r["model"],
            "Tref_mapping": _finite_or_none(r["Tref_mapping"]),
            "Tscale_mapping": _finite_or_none(r["Tscale_mapping"]),
            "scale_guess": _finite_or_none(r["scale_guess"]),
            "poor_overlap": r["poor_overlap"],
            "median_js_at_optimal_scale": _finite_or_none(r["median_js_at_optimal_scale"]),
            "s_opt_mean": _finite_or_none(float(np.mean(r["s_opt"]))),
            "s_opt_std": _finite_or_none(float(np.std(r["s_opt"]))),
            "s_opt_min": _finite_or_none(float(np.min(r["s_opt"]))),
            "s_opt_max": _finite_or_none(float(np.max(r["s_opt"]))),
            "all_temp_fits": {
                k: {"params": {nm: _finite_or_none(float(pv))
                               for nm, pv in zip(MAPPINGS[k]["param_names"],
                                                 r["all_fits"][k]["params"])},
                    "js_mean": _finite_or_none(r["all_fits"][k]["js_mean"])}
                for k in r["kinds"]
            },
            "bootstrap_ci": {
                k: {nm: {kk: _finite_or_none(vv) if isinstance(vv, float) else vv
                         for kk, vv in r["boot_ci"][k][nm].items()}
                    for nm in MAPPINGS[k]["param_names"]}
                for k in r["kinds"]
            },
            "slope_s1": {
                "median": _finite_or_none(r["s1_median"]),
                "ci": [_finite_or_none(r["s1_ci"][0]), _finite_or_none(r["s1_ci"][1])],
                "distinguishable_from_zero": r["slope_excludes_zero"],
                "effect_size_rel_scale_change": _finite_or_none(r["effect_size_rel_scale_change"]),
            },
            "held_out_improvement_vs_constant": {
                k: {kk: (vv if isinstance(vv, (bool, list)) else _finite_or_none(vv))
                    for kk, vv in r["improvements"][k].items()}
                for k in r["improvements"]
            },
            "stable_and_meaningful": r["stable_and_meaningful"],
            "recommendation": r["recommendation"],
        })
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, cls=_NpEncoder, allow_nan=False)
    print(f"Saved: {p}")

    # rg_mapping_report.md
    write_report(outdir, results, args, production_scalar)


def write_report(outdir: Path, results: List[Dict[str, Any]],
                 args: argparse.Namespace, production_scalar: float) -> None:
    L: List[str] = []
    L.append("# Rg mapping diagnostic report")
    L.append("")
    L.append(f"- Production scalar mapping: `Rg_obs = {production_scalar:.12g} * Rg_lat`")
    L.append(f"- Bootstrap reps: {args.bootstrap}, seed {args.seed}, "
             f"confidence {args.confidence}")
    L.append(f"- Split schemes (blocked + interleaved): {args.split_schemes}")
    L.append("")
    L.append("> **Guardrail.** A temperature-dependent geometric scale is "
             "physically suspicious — a genuine unit conversion is constant. A "
             "nonconstant s(T) more likely reflects contact-model misspecification "
             "than a real conversion. The constant mapping is preferred unless a "
             "richer mapping improves held-out Rg JS on *every* split and by a "
             "practically meaningful margin.")
    L.append("")
    for r in results:
        L.append(f"## {r['label']}  (model: {r['model']})")
        L.append("")
        s_mean = float(np.mean(r["s_opt"]))
        s_std = float(np.std(r["s_opt"]))
        s_min = float(np.min(r["s_opt"]))
        s_max = float(np.max(r["s_opt"]))
        L.append(f"- Per-temperature optimal scale s(T): mean **{s_mean:.5g}**, "
                 f"std {s_std:.3g}, range [{s_min:.5g}, {s_max:.5g}]")
        L.append(f"- Production scalar {production_scalar:.6g} vs s(T) mean "
                 f"{s_mean:.6g}: relative offset "
                 f"{100.0*(s_mean-production_scalar)/production_scalar:+.1f}%")
        L.append(f"- Median JS at optimal scale: {np.median(r['js_opt']):.4g}"
                 + ("  **(POOR OVERLAP — interpret with caution)**" if r["poor_overlap"] else ""))
        s1lo, s1hi = r["s1_ci"]
        L.append(f"- Temperature-linear slope s1 = {r['s1_median']:.4g} "
                 f"(CI [{_fmt(s1lo)}, {_fmt(s1hi)}]); distinguishable from zero: "
                 f"**{r['slope_excludes_zero']}**; relative scale change across T "
                 f"range ~ {r['effect_size_rel_scale_change']:.2%}")
        for k in ("affine", "tlinear_scale", "tlinear_affine"):
            if k in r["improvements"]:
                imp = r["improvements"][k]
                L.append(f"- Held-out improvement, {k} vs constant: mean "
                         f"{imp['mean_improvement']:+.3g} JS/temp "
                         f"({imp['mean_relative_improvement']:+.1%}), "
                         f"improves all splits: {imp['improves_all_splits']}")
        L.append("")
        L.append(f"**Recommendation: {r['recommendation']}.**")
        if r["recommendation"] == "constant multiplicative":
            L.append("")
            L.append("The temperature-linear scale does not clear the stability + "
                     "meaningfulness bar, so the constant mapping is retained.")
        L.append("")
    # overall verdict on the production scalar
    L.append("## Verdict on the production scalar")
    L.append("")
    any_nonconst = any(r["stable_and_meaningful"] for r in results)
    s_means = [float(np.mean(r["s_opt"])) for r in results]
    L.append(f"- Production scalar: `{production_scalar:.12g}`")
    L.append(f"- Mean optimal s(T) across models: "
             f"{np.mean(s_means):.6g} (per-model means: "
             f"{', '.join('%.5g' % s for s in s_means)})")
    if any_nonconst:
        L.append("- **At least one model shows stable, meaningful evidence for a "
                 "temperature-dependent scale.** Phase B integration may be "
                 "justified, but treat a nonconstant scale as a possible "
                 "misspecification signal, not a confirmed unit conversion.")
    else:
        L.append("- **No model shows stable, meaningful evidence for a "
                 "temperature-dependent or affine mapping.** The constant scalar "
                 "is adequate; Phase B integration is NOT warranted at this time.")
    L.append("")
    p = outdir / "rg_mapping_report.md"
    with open(p, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))
    print(f"Saved: {p}")


def _fmt(x: Optional[float]) -> str:
    return "n/a" if x is None else "%.4g" % x


# ===========================================================================
# Plots
# ===========================================================================

def make_plots(outdir: Path, results: List[Dict[str, Any]],
               production_scalar: float) -> None:
    if plt is None:
        print("WARNING: matplotlib unavailable; skipping plots.")
        return
    for r in results:
        lab = _sanitize_label(r["label"])
        temps = r["temps"]; xT = r["xT"]

        # 1. optimal s(T) with uncertainty + constant/linear fits
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.errorbar(temps, r["s_opt"], yerr=r["s_sigma"], fmt="o", ms=4,
                    capsize=2, label="s(T) optimal (local-curvature bars)")
        ax.axhline(production_scalar, color="k", ls="--", lw=1.2,
                   label=f"production {production_scalar:.4g}")
        s0 = r["all_fits"]["multiplicative"]["params"][0]
        ax.axhline(s0, color="b", ls="-", lw=1.2, label=f"constant fit s0={s0:.4g}")
        if "tlinear_scale" in r["all_fits"]:
            tp = r["all_fits"]["tlinear_scale"]["params"]
            ax.plot(temps, tp[0] + tp[1] * xT, "r-", lw=1.4,
                    label=f"linear fit s0+s1·x (s1={tp[1]:.3g})")
        ax.set_xlabel("T"); ax.set_ylabel("optimal multiplicative scale s")
        ax.set_title(f"Optimal Rg scale vs temperature\n{r['label']}")
        ax.legend(fontsize=8)
        fig.tight_layout()
        _save(fig, outdir / f"rg_scale_vs_T_{lab}.png")

        # 2. observed vs mapped mean Rg (constant mapping)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(temps, r["obs_mean_rg"], "k.", ms=5, label="observed")
        ax.plot(temps, production_scalar * r["pred_mean_lat"], "-",
                label="mapped (production scalar)")
        ax.plot(temps, s0 * r["pred_mean_lat"], "--", label="mapped (constant fit)")
        ax.set_xlabel("T"); ax.set_ylabel("mean Rg")
        ax.set_title(f"Observed vs mapped mean Rg\n{r['label']}")
        ax.legend(fontsize=8)
        fig.tight_layout()
        _save(fig, outdir / f"rg_mean_obs_vs_mapped_{lab}.png")

        # 3. Rg JS vs T for each mapping
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(temps, r["js_opt"], "o-", ms=3, label="per-T optimal s")
        for k in r["kinds"]:
            p = r["all_fits"][k]["params"]
            apply = MAPPINGS[k]["apply"]
            jsv = []
            for i in range(temps.size):
                me = apply(p, float(xT[i]), _rg_edges_from_centers(r["rg_centers_lat"]))
                jsv.append(js_on_common_grid(
                    _obs_mass_for(r, i), _obs_edges_for(r), r["pred_mass"][i], me))
            ax.plot(temps, jsv, "-", lw=1.2, label=k)
        ax.set_xlabel("T"); ax.set_ylabel("Rg JS divergence")
        ax.set_title(f"Rg JS vs temperature by mapping\n{r['label']}")
        ax.legend(fontsize=8)
        fig.tight_layout()
        _save(fig, outdir / f"rg_js_vs_T_{lab}.png")

        # 4. residuals (s(T) - constant fit) vs T
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.axhline(0.0, color="gray", lw=0.8)
        ax.plot(temps, r["s_opt"] - s0, "o-", ms=4)
        ax.set_xlabel("T"); ax.set_ylabel("s(T) - constant s0")
        ax.set_title(f"Scale residuals vs temperature\n{r['label']}")
        fig.tight_layout()
        _save(fig, outdir / f"rg_scale_residuals_{lab}.png")

        # 5. affine intercept diagnostic
        if "affine" in r["boot_ci"]:
            c_ci = r["boot_ci"]["affine"]["c"]
            fig, ax = plt.subplots(figsize=(6, 4))
            cmed = c_ci["median"]; clo = c_ci["ci_low"]; chi = c_ci["ci_high"]
            ax.axvline(0.0, color="k", ls="--", lw=1.2, label="zero intercept")
            if cmed is not None:
                ax.axvline(cmed, color="b", lw=1.6, label=f"median c={cmed:.3g}")
            if clo is not None:
                ax.axvspan(clo, chi, color="b", alpha=0.2,
                           label="bootstrap CI")
            ax.set_xlabel("affine intercept c (observed Rg units)")
            ax.set_title(f"Affine intercept diagnostic\n{r['label']}")
            ax.legend(fontsize=8)
            fig.tight_layout()
            _save(fig, outdir / f"rg_affine_intercept_{lab}.png")


def _rg_edges_from_centers(centers: np.ndarray) -> np.ndarray:
    return centers_to_edges(centers)


# plot helpers stash obs arrays on the result dict lazily
def _obs_mass_for(r: Dict[str, Any], i: int) -> np.ndarray:
    return r["_obs_mass"][i]


def _obs_edges_for(r: Dict[str, Any]) -> np.ndarray:
    return r["_obs_edges"]


def _save(fig, path: Path) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ===========================================================================
# Loading target / baseline
# ===========================================================================

def load_target_rg(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    d = np.load(path)
    tkey = "temps" if "temps" in d.files else "Ts" if "Ts" in d.files else None
    if tkey is None:
        raise ValueError(f"target {path!r} has neither 'temps' nor 'Ts'")
    rc_key = next((k for k in ("rg_centers", "Rg_centers") if k in d.files), None)
    rh_key = next((k for k in ("rg_hists", "Rg_hists") if k in d.files), None)
    if rc_key is None or rh_key is None:
        raise ValueError(
            f"target {path!r} lacks observed Rg histograms (rg_centers/rg_hists); "
            "Rg mapping analysis is impossible without observed Rg."
        )
    temps = np.asarray(d[tkey], dtype=float)
    rg_centers = np.asarray(d[rc_key], dtype=float)
    rg_hists = np.asarray(d[rh_key], dtype=float)
    if rg_hists.shape != (temps.size, rg_centers.size):
        raise ValueError("target rg_hists shape must be (n_temps, n_rg_bins)")
    obs_edges = centers_to_edges(rg_centers)
    obs_mass = np.array([pdf_to_mass(rg_hists[i], rg_centers)[0]
                         for i in range(temps.size)])
    return temps, rg_centers, obs_edges, obs_mass


def load_joint_baseline(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    b = np.load(path)
    for k in ("c_edges", "rg_edges", "crg_prob"):
        if k not in b.files:
            raise ValueError(
                f"baseline {path!r} is not a joint baseline (needs c_edges, "
                f"rg_edges, crg_prob); Rg mapping analysis requires the joint P0(m,Rg)."
            )
    return (np.asarray(b["crg_prob"], dtype=float),
            np.asarray(b["c_edges"], dtype=float),
            np.asarray(b["rg_edges"], dtype=float))


# ===========================================================================
# Quick test
# ===========================================================================

def run_quick_test() -> int:
    failures: List[str] = []

    def check(cond, msg):
        if cond:
            print(f"  PASS: {msg}")
        else:
            print(f"  FAIL: {msg}")
            failures.append(msg)

    # synthetic lattice predicted Rg: gaussian per temperature, fixed shape
    rng = np.random.default_rng(0)
    n_t = 12
    temps = np.linspace(280.0, 360.0, n_t)
    Tref = float(temps.mean()); Tscale = float(temps.max() - temps.min())
    xT = (temps - Tref) / Tscale
    rg_edges_lat = np.linspace(2.0, 7.0, 81)
    rg_centers_lat = 0.5 * (rg_edges_lat[:-1] + rg_edges_lat[1:])
    # predicted lattice mass: gaussian centered at 4.5, slight T drift
    pred_mass = np.zeros((n_t, rg_centers_lat.size))
    for i in range(n_t):
        mu = 4.5 + 0.3 * xT[i]
        g = np.exp(-0.5 * ((rg_centers_lat - mu) / 0.5) ** 2)
        pred_mass[i] = g / g.sum()

    def build_obs(map_fn):
        """Make observed mass = predicted lattice transformed by map_fn(edges,x)."""
        obs_centers = np.linspace(0.5, 4.0, 140)
        obs_edges = centers_to_edges(obs_centers)
        obs_mass = np.zeros((n_t, obs_centers.size))
        for i in range(n_t):
            me = map_fn(rg_edges_lat, xT[i])
            obs_mass[i] = overlap_nonorm(me, pred_mass[i], obs_edges)
            s = obs_mass[i].sum()
            if s > 0:
                obs_mass[i] /= s
        return obs_centers, obs_edges, obs_mass

    s_true = 0.6
    print("Quick test 1: constant scale recovery")
    oc, oe, om = build_obs(lambda e, x: s_true * e)
    p, _ = fit_mapping("multiplicative", np.arange(n_t), xT, om, oe, pred_mass,
                       rg_edges_lat, 0.7)
    check(abs(p[0] - s_true) < 0.01, f"recovered s0~{s_true}: got {p[0]:.4f}")
    pt = fit_mapping("tlinear_scale", np.arange(n_t), xT, om, oe, pred_mass,
                     rg_edges_lat, 0.7)[0]
    check(abs(pt[1]) < 0.02, f"tlinear slope ~0 for constant data: s1={pt[1]:.4f}")

    print("Quick test 2: known temperature-dependent scale recovery")
    s0t, s1t = 0.6, 0.15
    oc, oe, om = build_obs(lambda e, x: (s0t + s1t * x) * e)
    pt = fit_mapping("tlinear_scale", np.arange(n_t), xT, om, oe, pred_mass,
                     rg_edges_lat, 0.7)[0]
    check(abs(pt[0] - s0t) < 0.02 and abs(pt[1] - s1t) < 0.03,
          f"recovered (s0,s1)~({s0t},{s1t}): got ({pt[0]:.3f},{pt[1]:.3f})")
    js_const = fit_mapping("multiplicative", np.arange(n_t), xT, om, oe, pred_mass,
                           rg_edges_lat, 0.7)[1]
    js_tlin = mapping_total_js("tlinear_scale", pt, np.arange(n_t), xT, om, oe,
                               pred_mass, rg_edges_lat)
    check(js_tlin < js_const, "temperature-linear beats constant on T-dependent data")

    print("Quick test 3: affine offset recovery")
    a_t, c_t = 0.5, 0.3
    oc, oe, om = build_obs(lambda e, x: a_t * e + c_t)
    pa = fit_mapping("affine", np.arange(n_t), xT, om, oe, pred_mass,
                     rg_edges_lat, 0.6)[0]
    check(abs(pa[0] - a_t) < 0.03 and abs(pa[1] - c_t) < 0.05,
          f"recovered (a,c)~({a_t},{c_t}): got ({pa[0]:.3f},{pa[1]:.3f})")
    js_mult = fit_mapping("multiplicative", np.arange(n_t), xT, om, oe, pred_mass,
                          rg_edges_lat, 0.6)[1]
    js_aff = mapping_total_js("affine", pa, np.arange(n_t), xT, om, oe, pred_mass,
                              rg_edges_lat)
    check(js_aff < js_mult, "affine beats pure multiplicative on offset data")

    print("Quick test 4: no-overlap failure is detected (high residual JS)")
    # observed far below any reachable scaled prediction within bounds
    obs_centers = np.linspace(0.01, 0.05, 60)
    obs_edges = centers_to_edges(obs_centers)
    obs_mass = np.zeros((n_t, obs_centers.size))
    obs_mass[:, obs_centers.size // 2] = 1.0
    s_opt, js_opt, _ = per_temperature_optimal_scale(
        xT, obs_mass, obs_edges, pred_mass, rg_edges_lat, 0.05, 5.0)
    check(np.median(js_opt) > 0.5,
          f"disjoint Rg ranges leave high JS (median {np.median(js_opt):.3f}) -> flagged")

    print("Quick test 5: scalar/affine/tlinear transformations are exact")
    e = np.array([1.0, 2.0, 3.0])
    check(np.allclose(MAPPINGS["multiplicative"]["apply"]([2.0], 0.3, e), 2.0 * e),
          "multiplicative apply exact")
    check(np.allclose(MAPPINGS["affine"]["apply"]([2.0, 0.5], 0.3, e), 2.0 * e + 0.5),
          "affine apply exact")
    check(np.allclose(MAPPINGS["tlinear_scale"]["apply"]([2.0, 1.0], 0.5, e),
                      (2.0 + 0.5) * e), "tlinear_scale apply exact")

    print()
    if failures:
        print(f"QUICK TEST FAILED: {len(failures)} assertion(s) failed.")
        return 1
    print("QUICK TEST PASSED: all assertions passed.")
    return 0


# ===========================================================================
# Main
# ===========================================================================

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Diagnose the lattice->observed Rg mapping and its temperature "
        "dependence (Phase A; no production changes).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--target", type=str, default=None,
                    help="Target NPZ with temps, rg_centers/Rg_centers, rg_hists/Rg_hists.")
    ap.add_argument("--baseline", type=str, default=None,
                    help="Joint baseline NPZ with c_edges, rg_edges, crg_prob.")
    ap.add_argument("--fit-summary", type=str, nargs="+", default=None,
                    dest="fit_summary",
                    help="One or more fit_summary.json files (one per fitted model).")
    ap.add_argument("--split-schemes", type=str, dest="split_schemes",
                    default="blocked_low,blocked_mid,blocked_high,every_third_phase",
                    help="Blocked + interleaved validation schemes for held-out comparison.")
    ap.add_argument("--kfold-k", type=int, default=5, dest="kfold_k")
    ap.add_argument("--blocked-fraction", type=float, default=0.25,
                    dest="blocked_fraction")
    ap.add_argument("--outdir", type=str, default="rg_mapping_diagnostics")
    ap.add_argument("--bootstrap", type=int, default=200,
                    help="Bootstrap repetitions (resampling temperatures).")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--confidence", type=float, default=0.95)
    ap.add_argument("--scale-lo", type=float, default=0.05, dest="scale_lo")
    ap.add_argument("--scale-hi", type=float, default=5.0, dest="scale_hi")
    ap.add_argument("--poor-overlap-js", type=float, default=0.5, dest="poor_overlap_js",
                    help="Median per-T JS above this flags poor Rg overlap.")
    ap.add_argument("--meaningful-rel-improvement", type=float, default=0.05,
                    dest="meaningful_rel_improvement",
                    help="Minimum mean relative held-out JS improvement for a richer "
                         "mapping to be recommended over the constant scalar.")
    ap.add_argument("--include-tlinear-affine", action="store_true",
                    dest="include_tlinear_affine",
                    help="Also fit the temperature-linear affine mapping.")
    ap.add_argument("--production-scalar", type=float, default=0.46320503312590167,
                    dest="production_scalar",
                    help="Reference production rg_scale to compare against.")
    ap.add_argument("--no-plots", action="store_true", dest="no_plots")
    ap.add_argument("--quick-test", action="store_true", dest="quick_test")
    args = ap.parse_args(argv)

    if args.quick_test:
        return run_quick_test()

    if minimize is None or minimize_scalar is None:
        raise RuntimeError("scipy is required.")
    if args.target is None or args.baseline is None or not args.fit_summary:
        ap.error("--target, --baseline, and --fit-summary are required "
                 "(unless --quick-test).")
    if not (0.0 < args.confidence < 1.0):
        ap.error("--confidence must be in (0, 1).")
    if args.bootstrap < 0:
        ap.error("--bootstrap must be >= 0.")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    temps, rg_centers, obs_edges, obs_mass = load_target_rg(args.target)
    crg_prob, c_edges, rg_edges_lat = load_joint_baseline(args.baseline)
    obs_centers = rg_centers

    schemes = [s.strip() for s in args.split_schemes.split(",") if s.strip()]
    splits = build_split_schemes(
        temps.size, schemes, kfold_k=args.kfold_k,
        blocked_fraction=args.blocked_fraction, random_fraction=0.25,
        random_repeats=1, split_seed=args.seed,
    )

    print("--- Rg mapping diagnostic (Phase A) ---")
    print(f"Target:   {args.target}  ({temps.size} temps)")
    print(f"Baseline: {args.baseline}")
    print(f"Production scalar: {args.production_scalar}")
    print(f"Splits: {[sp['name'] for sp in splits]}")

    results: List[Dict[str, Any]] = []
    for fs_path in args.fit_summary:
        summary = load_fit_summary(fs_path)
        label = f"{summary['model']}:{Path(fs_path).parent.name or Path(fs_path).stem}"
        print(f"\nAnalyzing {label}  ({fs_path})")
        r = analyze_one_model(label, summary, temps, obs_mass, obs_edges, obs_centers,
                              crg_prob, c_edges, rg_edges_lat, splits, args)
        # stash obs arrays for plotting closures
        r["_obs_mass"] = obs_mass
        r["_obs_edges"] = obs_edges
        results.append(r)
        print(f"  s(T) mean={np.mean(r['s_opt']):.5g} std={np.std(r['s_opt']):.3g}  "
              f"slope s1={r['s1_median']:.4g} (CI {_fmt(r['s1_ci'][0])},{_fmt(r['s1_ci'][1])})  "
              f"-> {r['recommendation']}")

    write_outputs(outdir, results, args, args.production_scalar)
    if not args.no_plots:
        make_plots(outdir, results, args.production_scalar)

    print("\nDone.")
    for r in results:
        print(f"  {r['label']}: recommend {r['recommendation']} "
              f"(s1 distinguishable from 0: {r['slope_excludes_zero']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
