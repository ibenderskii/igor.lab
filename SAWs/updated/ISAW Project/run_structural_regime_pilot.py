#!/usr/bin/env python3
"""PNIPAM-aware structural CALIBRATION pilot (LCST-correct; K-based).

Builds a temperature ladder from a FITTED contact model, evaluates K(T) = -b(T)
densely over the fitted interval, selects a monotonic K branch, verifies whether
the model is consistent with PNIPAM LCST behavior (collapse as T INCREASES
because K increases with T), runs short structural REMD with full saving,
extracts validated features, and assesses collapse trends against K using
endpoint effect sizes, tie-corrected rank correlation, slopes, and block + seed
bootstrap.  It then evaluates structured sampling/mixing gates and recommends
production settings per structural regime WITHOUT launching production.

PNIPAM convention
-----------------
P(C|T) ∝ exp[K(T) m(C)] with K(T) = -b(T); higher K favors more contacts.  For
the fitted LCST branch we EXPECT, from low-K to high-K: <m> up, <Rg^2> down,
global/long contacts up, connected structure up.  These are verified, never
hardcoded.

Scientific vs smoke mode
------------------------
* ``--smoke-test`` runs a tiny, fast software check; it is NEVER labeled
  scientifically validated and always exits 0.
* A scientific run requires >= 2 independent seeds and PASSES only when every
  gate passes.  It exits NONZERO on a failed scientific gate unless
  ``--allow-failed-calibration`` is given.

This script does NOT launch the eight-seed production campaign.

Pure analysis helpers (``evaluate_K_grid``, ``monotonic_branches``,
``select_branch``, ``build_k_ladder``, ``spearman``, ``block_bootstrap_mean``,
``classify_regimes``, ``high_contact_tail``, ``production_requirement``,
``evaluate_sampling_gates``) are unit-tested in ``tests/test_calibration.py``.
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import remd_uniform_chain_2_new as remd
import isaw_schema as sch
import isaw_config_io as cio
import extract_contact_motif_features as ext

PY = sys.executable
REMD = str(HERE / "remd_uniform_chain_2_new.py")
EXTRACT = str(HERE / "extract_contact_motif_features.py")

CALIBRATION_REPORT_SCHEMA_VERSION = sch.CALIBRATION_REPORT_SCHEMA_VERSION
TARGET_EFF_PER_REGIME = 5000   # computational-plan target (per regime, per N)
REGIME_NAMES = ("swollen", "crossover", "collapsed")

# Default scientific gate thresholds (documented; overridable).
DEFAULT_GATE_THRESHOLDS = {
    "min_ess": 200.0,
    "min_round_trips": 5.0,
    "min_temp_coverage": 0.5,
    "min_adjacent_overlap": 0.20,
    "min_swap_rate": 0.10,
    "max_swap_rate": 0.90,
    "max_drift_in_std": 1.0,
    "min_state_changing_acceptance": 0.01,
    "min_tail_probability": 1e-3,
    "max_seed_relative_spread": 0.5,
    "min_regime_effect_size": 2.0,
    # Phase 9 tail gate (pooled + per-seed support, not just a probability).
    "min_raw_pooled_tail_count": 20.0,
    "min_effective_pooled_tail_count": 5.0,
    "min_seed_tail_fraction": 0.5,
    "min_per_seed_tail_count": 1.0,
    # Phase 10 worst-seed adjacent overlap (isolated poor connectivity).
    "min_worst_seed_adjacent_overlap": 0.10,
    # Phase 8 seed-label agreement for regime stability.
    "min_regime_label_agreement": 1.0,
}


# ---------------------------------------------------------------------------
# Fitted model + K(T) analysis (Phase 6.1, 6.2)
# ---------------------------------------------------------------------------

def infer_fit_chain_length(raw: dict):
    """Infer N from a fit summary (contact_offset == N-1) or return None."""
    if "n_beads" in raw:
        return int(raw["n_beads"])
    if "N" in raw:
        return int(raw["N"])
    if "contact_offset" in raw and raw["contact_offset"] is not None:
        return int(round(float(raw["contact_offset"]))) + 1
    return None


def load_fitted_model(summary_path, cli_tmin, cli_tmax, requested_N,
                      allow_n_mismatch=False):
    info = remd.load_fit_summary_json(summary_path)
    with open(summary_path) as fh:
        raw = json.load(fh)
    fit_N = infer_fit_chain_length(raw)
    n_note = None
    if fit_N is not None and int(fit_N) != int(requested_N):
        n_note = (f"requested N={requested_N} != fit-summary N={fit_N}")
        if not allow_n_mismatch:
            raise SystemExit(
                n_note + "; pass --allow-n-mismatch to override (recorded).")
    elif fit_N is not None:
        n_note = f"fit-summary N={fit_N} matches requested N"
    rng = None
    if cli_tmin is not None and cli_tmax is not None:
        rng = (float(cli_tmin), float(cli_tmax))
    elif "fitted_temperature_range" in raw:
        rng = (float(min(raw["fitted_temperature_range"])),
               float(max(raw["fitted_temperature_range"])))
    elif "temps_all" in raw and raw["temps_all"]:
        ta = [float(t) for t in raw["temps_all"]]
        rng = (min(ta), max(ta))
    if rng is None:
        raise SystemExit(
            "fit summary has no explicit temperature interval; supply "
            "--t-min and --t-max (no silent fallback).")
    return info, rng, {"fit_N": fit_N, "n_note": n_note}


def evaluate_K_grid(info, T_lo, T_hi, n_dense=400):
    Ts = np.linspace(T_lo, T_hi, n_dense)
    b = np.array([remd.reduced_bias(info["model_name"], info["params"],
                                    float(T), info["Tref"], info["Tscale"])
                  for T in Ts])
    K = -b
    dKdT = np.gradient(K, Ts)
    inc = bool(np.all(np.diff(K) > 0))
    dec = bool(np.all(np.diff(K) < 0))
    zero_cross = None
    sgn = np.sign(b)
    idx = np.where(np.diff(sgn) != 0)[0]
    if idx.size:
        i = int(idx[0])
        zero_cross = float(np.interp(0.0, [b[i + 1], b[i]], [Ts[i + 1], Ts[i]])
                           if b[i + 1] != b[i] else Ts[i])
    return {"T": Ts, "b": b, "K": K, "dKdT": dKdT,
            "K_increases_with_T": inc, "monotonic": bool(inc or dec),
            "zero_crossing_T": zero_cross}


# Backwards-compatible alias used by older callers/tests.
analyze_K = evaluate_K_grid


def monotonic_branches(T, K):
    """Return index ranges (i0, i1) of maximal monotonic runs of K over T.

    Flat segments (dK == 0) are absorbed into the neighbouring run.
    """
    K = np.asarray(K, float)
    n = K.shape[0]
    if n < 2:
        return [(0, max(0, n - 1))]
    sign = np.sign(np.diff(K))
    branches = []
    start = 0
    cur = 0.0
    for i, sg in enumerate(sign):
        if sg == 0:
            continue
        if cur == 0.0:
            cur = sg
        elif sg != cur:
            branches.append((start, i))
            start = i
            cur = sg
    branches.append((start, n - 1))
    return branches


def select_branch(kinfo, branch="auto", physical_branch=None):
    """Select a monotonic K branch and report its bounds and direction.

    ``branch`` is 'low' (lowest-T branch), 'high' (highest-T branch), or 'auto'
    (the full interval if monotonic, else fail unless ``physical_branch`` picks
    one explicitly).
    """
    T = np.asarray(kinfo["T"], float)
    K = np.asarray(kinfo["K"], float)
    brs = monotonic_branches(T, K)
    if branch == "auto":
        if len(brs) == 1:
            idx = 0
        elif physical_branch is not None:
            idx = int(physical_branch)
        else:
            raise SystemExit(
                "K(T) is non-monotonic over the interval and no physical branch "
                "is marked; rerun with --branch low or --branch high.")
    elif branch == "low":
        idx = 0
    elif branch == "high":
        idx = len(brs) - 1
    else:
        raise SystemExit(f"unknown branch {branch!r}")
    i0, i1 = brs[idx]
    Tb, Kb = T[i0:i1 + 1], K[i0:i1 + 1]
    direction = ("increasing" if Kb[-1] > Kb[0]
                 else "decreasing" if Kb[-1] < Kb[0] else "flat")
    return {
        "branch_index": int(idx),
        "n_branches": int(len(brs)),
        "branch_temperature_bounds": [float(Tb[0]), float(Tb[-1])],
        "branch_K_bounds": [float(Kb[0]), float(Kb[-1])],
        "K_direction": direction,
        "K_increases_with_T": bool(direction == "increasing"),
        "Ts": Tb, "K": Kb,
    }


# ---------------------------------------------------------------------------
# K ladder (Phase 6.3) + overlap refinement (Phase 6.4)
# ---------------------------------------------------------------------------

def build_k_ladder(Ts, K, n_points, min_dT=0.25, min_dK=1e-4, round_to=4,
                   k_of_T=None):
    """Ladder approximately uniform in K, mapped to T (Part 9).

    Works for BOTH increasing and decreasing K(T): all interpolation is done on
    a strictly T-ascending coordinate, so a descending x-array is never passed
    to ``np.interp``.  After selecting temperatures the K values are recomputed
    by calling the actual fitted model (``k_of_T``) when supplied, rather than
    re-interpolating a possibly misordered array.  Spacing constraints are
    reported (``spacing_ok``, ``exact_count``, ``endpoints_included``) with a
    recommendation when they cannot be met.
    """
    Ts = np.asarray(Ts, float)
    K = np.asarray(K, float)
    n_points = int(n_points)
    if n_points < 2:
        raise ValueError("n_points must be >= 2")
    # Strictly T-ascending working arrays (drop duplicate temperatures).
    to = np.argsort(Ts)
    Tt, Kt = Ts[to], K[to]
    keep = np.concatenate([[True], np.diff(Tt) > 0])
    Tt, Kt = Tt[keep], Kt[keep]
    K0, K1 = float(Kt[0]), float(Kt[-1])
    K_levels = np.linspace(K0, K1, n_points)
    # Invert K -> T.  np.interp needs an ASCENDING xp, so flip when K decreases.
    if K1 >= K0:
        temps = np.interp(K_levels, Kt, Tt)
    else:
        temps = np.interp(K_levels[::-1], Kt[::-1], Tt[::-1])[::-1]
    temps[0], temps[-1] = float(Tt[0]), float(Tt[-1])
    temps = np.sort(temps)
    temps = np.round(temps, round_to)
    step = 10.0 ** (-round_to)
    for i in range(1, temps.size):
        if temps[i] <= temps[i - 1]:
            temps[i] = round(temps[i - 1] + step, round_to)
    # Recompute K at the chosen temperatures via the ACTUAL model when given;
    # else interpolate on the strictly T-ascending arrays (never K-ordered).
    if k_of_T is not None:
        K_at = np.array([float(k_of_T(float(t))) for t in temps])
    else:
        K_at = np.interp(temps, Tt, Kt)
    dK = np.abs(np.diff(K_at))
    dT = np.abs(np.diff(temps))
    unique_count = int(np.unique(temps).size)
    min_dT_ok = bool(dT.size and dT.min() >= min_dT)
    min_dK_ok = bool(dK.size and dK.min() >= min_dK)
    exact_count = bool(unique_count == n_points)
    endpoints_included = bool(abs(temps[0] - Tt[0]) <= step
                              and abs(temps[-1] - Tt[-1]) <= step)
    spacing_ok = bool(min_dT_ok and min_dK_ok and exact_count and endpoints_included)
    recommendation = None
    if not spacing_ok:
        recs = []
        if not exact_count:
            recs.append(f"cannot place {n_points} unique temperatures in the "
                        f"branch interval; reduce --n-temperatures")
        if not min_dT_ok:
            recs.append(f"minimum dT {float(dT.min()) if dT.size else float('nan'):.4g}"
                        f" < min_dT {min_dT}; reduce --n-temperatures or relax min_dT")
        if not min_dK_ok:
            recs.append(f"minimum dK {float(dK.min()) if dK.size else float('nan'):.4g}"
                        f" < min_dK {min_dK}; reduce --n-temperatures or relax min_dK")
        recommendation = "; ".join(recs)
    return {
        "temperatures": temps,
        "K_values": [float(x) for x in K_at],
        "achieved_dK": [float(x) for x in dK],
        "achieved_dT": [float(x) for x in dT],
        "min_dK": float(dK.min()) if dK.size else float("nan"),
        "min_dT": float(dT.min()) if dT.size else float("nan"),
        "min_dK_ok": min_dK_ok,
        "min_dT_ok": min_dT_ok,
        "unique_count": unique_count,
        "requested_count": n_points,
        "exact_count": exact_count,
        "endpoints_included": endpoints_included,
        "spacing_ok": spacing_ok,
        "K_direction": ("increasing" if K1 > K0 else
                        "decreasing" if K1 < K0 else "flat"),
        "recommendation": recommendation,
    }


def _normalize_gap_records(adjacent_gaps):
    """Coerce adjacent-gap records into a structured pooled/worst/median/spread
    form (accepts either the structured aggregate dicts or a legacy list of
    pooled floats)."""
    out = []
    for g in adjacent_gaps:
        if isinstance(g, (int, float, np.floating)):
            v = float(g)
            out.append({"pooled_overlap": v, "min_seed_overlap": v,
                        "median_seed_overlap": v, "overlap_spread": 0.0,
                        "per_seed": [v]})
            continue
        per = [float(x) for x in g.get("overlap_per_seed", [])]
        pooled = float(g.get("pooled_count_overlap",
                             g.get("pooled_overlap",
                                   np.median(per) if per else float("nan"))))
        mn = float(g.get("minimum_overlap",
                         g.get("min_seed_overlap", min(per) if per else pooled)))
        med = float(g.get("median_overlap",
                          g.get("median_seed_overlap",
                                np.median(per) if per else pooled)))
        spread = float((max(per) - min(per)) if per else 0.0)
        out.append({"pooled_overlap": pooled, "min_seed_overlap": mn,
                    "median_seed_overlap": med, "overlap_spread": spread,
                    "per_seed": per})
    return out


def refine_ladder(temps, adjacent_gaps, low_overlap=0.15, high_overlap=0.90,
                  max_disagreement=0.25):
    """Recommend ladder edits from a STRUCTURED adjacent-gap overlap record (Phase 10).

    Each gap carries pooled overlap, minimum/median per-seed overlap, the
    across-seed spread, and the per-seed values.  Every recommendation states its
    cause: ``poor pooled overlap``, ``poor worst-seed overlap``, ``high seed
    disagreement``, or ``redundant overlap across all seeds``.

    A lane is deleted as redundant ONLY when both the pooled AND the worst-seed
    overlap are high, so a seed with poor connectivity is never ignored because
    the pooled overlap looks acceptable; a gap with poor worst-seed overlap gets
    an insert even if pooled overlap is fine.  Ladder ENDPOINTS are preserved
    (never deleted) unless explicitly recorded.  Does NOT rerun anything.
    """
    temps = [float(t) for t in temps]
    gaps = _normalize_gap_records(adjacent_gaps)
    endpoints = {temps[0], temps[-1]} if temps else set()
    recs = []
    drop = set()
    inserts = []
    for a, g in enumerate(gaps):
        lo, hi = temps[a], temps[a + 1]
        pooled = g["pooled_overlap"]
        worst = g["min_seed_overlap"]
        spread = g["overlap_spread"]
        base = {"gap": [lo, hi], "gap_index": a, "pooled_overlap": pooled,
                "worst_seed_overlap": worst, "median_seed_overlap":
                g["median_seed_overlap"], "overlap_spread": spread,
                "per_seed_overlap": g["per_seed"]}
        if spread >= max_disagreement:
            recs.append({**base, "action": "resample_or_investigate",
                         "cause": "high seed disagreement",
                         "detail": f"per-seed overlap spread {spread:.3f} "
                                   f">= {max_disagreement}"})
        if pooled <= low_overlap or worst <= low_overlap:
            cause = ("poor pooled overlap" if pooled <= low_overlap
                     else "poor worst-seed overlap")
            mid = round(0.5 * (lo + hi), 4)
            recs.append({**base, "action": "insert_lane",
                         "insert_temperature": mid, "cause": cause,
                         "detail": f"pooled {pooled:.3f} / worst {worst:.3f} "
                                   f"<= {low_overlap}"})
            inserts.append(mid)
        elif pooled >= high_overlap and worst >= high_overlap:
            target = hi
            if target in endpoints:
                recs.append({**base, "action": "preserve_endpoint",
                             "cause": "redundant overlap across all seeds",
                             "detail": f"lane {target} is a ladder endpoint; "
                                       f"preserved despite pooled {pooled:.3f} "
                                       f"/ worst {worst:.3f} >= {high_overlap}"})
            else:
                recs.append({**base, "action": "delete_redundant_lane",
                             "cause": "redundant overlap across all seeds",
                             "detail": f"pooled {pooled:.3f} AND worst "
                                       f"{worst:.3f} >= {high_overlap}"})
                drop.add(target)
    refined = sorted(set([t for t in temps if t not in drop] + inserts))
    # Endpoints must survive refinement.
    for e in endpoints:
        if e not in refined:
            refined = sorted(set(refined + [e]))
    return {"recommendations": recs, "refined_temperatures": refined,
            "overlap_metric": "Bhattacharyya coefficient",
            "adjacent_gaps": gaps,
            "endpoints_preserved": sorted(endpoints)}


# ---------------------------------------------------------------------------
# Statistics (Phase 7)
# ---------------------------------------------------------------------------

def spearman(x, y):
    """Tie-corrected Spearman rank correlation (SciPy if available)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    if x.size < 3:
        return float("nan")
    try:
        from scipy.stats import spearmanr
        r = spearmanr(x, y).correlation
        return float(r)
    except Exception:
        def _avg_rank(a):
            order = np.argsort(a, kind="mergesort")
            ranks = np.empty(a.size, float)
            ranks[order] = np.arange(a.size, dtype=float)
            # average ties
            sa = a[order]
            i = 0
            while i < a.size:
                j = i
                while j + 1 < a.size and sa[j + 1] == sa[i]:
                    j += 1
                if j > i:
                    ranks[order[i:j + 1]] = (i + j) / 2.0
                i = j + 1
            return ranks
        rx, ry = _avg_rank(x), _avg_rank(y)
        rx -= rx.mean(); ry -= ry.mean()
        denom = np.sqrt((rx * rx).sum() * (ry * ry).sum())
        return float((rx * ry).sum() / denom) if denom > 0 else float("nan")


def seed_bootstrap(vals, n=4000, seed=0):
    """Hierarchical (across-seed) bootstrap CI of the mean."""
    vals = np.asarray(vals, float)
    if vals.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    if vals.size == 1:
        return (float(vals[0]), float(vals[0]), float(vals[0]))
    rng = np.random.RandomState(seed)
    bs = vals[rng.randint(0, vals.size, size=(n, vals.size))].mean(axis=1)
    return (float(vals.mean()), float(np.percentile(bs, 2.5)),
            float(np.percentile(bs, 97.5)))


def block_bootstrap_mean(x, n_blocks=10, n=2000, seed=0):
    """Moving-block bootstrap CI of the mean of a correlated series."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    if x.size < n_blocks or n_blocks < 2:
        m = float(x.mean())
        return (m, m, m)
    L = max(1, x.size // n_blocks)
    starts_max = x.size - L
    rng = np.random.RandomState(seed)
    n_needed = int(np.ceil(x.size / L))
    means = np.empty(n)
    for b in range(n):
        s = rng.randint(0, starts_max + 1, size=n_needed)
        sample = np.concatenate([x[i:i + L] for i in s])[:x.size]
        means[b] = sample.mean()
    return (float(x.mean()), float(np.percentile(means, 2.5)),
            float(np.percentile(means, 97.5)))


def ci_excludes_zero(lo, hi):
    return bool((lo > 0 and hi > 0) or (lo < 0 and hi < 0))


def _block_resample_mean(x, block_len, rng):
    """Mean of one moving-block bootstrap resample of a correlated 1-D series."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = x.size
    if n == 0:
        return float("nan")
    L = max(1, min(int(block_len), n))
    n_blocks = int(np.ceil(n / L))
    if n > L:
        starts = rng.randint(0, n - L + 1, size=n_blocks)
    else:
        starts = np.zeros(n_blocks, dtype=int)
    sample = np.concatenate([x[s:s + L] for s in starts])[:n]
    return float(sample.mean())


def hierarchical_bootstrap(seed_data, n_boot=2000, block_len=20, seed=20260701,
                           n_requested_seeds=None):
    """True two-level (seed x block) bootstrap of endpoint deltas and K-slopes (Part 4).

    ``seed_data`` is a list (one entry per independent seed) of dicts with keys
    ``K`` (nT,), and per-lane trajectory matrices ``contacts``/``rg2``
    (nT, n_post) and ``m_global``/``smax`` (nT, n_struct).  Each replicate: (1)
    resamples SEEDS with replacement; (2) within each selected seed resamples
    trajectory BLOCKS per lane; (3) recomputes lane means; (4) recomputes the
    high-K-minus-low-K endpoint deltas and the slopes vs K.  Point estimates use
    the full data.  Deterministic ``seed`` makes the CIs reproducible.
    """
    S = len(seed_data)
    if S == 0:
        return None
    K = np.asarray(seed_data[0]["K"], float)
    order = np.argsort(K)
    lo, hi = int(order[0]), int(order[-1])
    # Only bootstrap observables actually present (synthetic callers may omit
    # the largest-component fraction; real seed data carries it).
    keys = tuple(k for k in ("contacts", "rg2", "m_global", "smax", "lcf")
                 if k in seed_data[0])

    def lane_means(sd, key, rng=None):
        arr = np.asarray(sd[key], float)
        nT = arr.shape[0]
        if rng is None:
            return np.array([float(np.nanmean(arr[k])) for k in range(nT)])
        return np.array([_block_resample_mean(arr[k], block_len, rng)
                         for k in range(nT)])

    def agg(selected, rng=None):
        out = {}
        idx = selected if selected is not None else range(S)
        for key in keys:
            lm = np.mean([lane_means(seed_data[i], key, rng) for i in idx], axis=0)
            out[key] = lm
        return out

    def _slope(K_, y):
        return float(np.polyfit(K_, y, 1)[0]) if y.size >= 2 else float("nan")

    a0 = agg(None)
    est = {
        "delta_contacts": float(a0["contacts"][hi] - a0["contacts"][lo]),
        "delta_rg2": float(a0["rg2"][hi] - a0["rg2"][lo]),
    }
    for k in keys:
        est[f"slope_{k}"] = _slope(K, a0[k])
    rng = np.random.RandomState(int(seed))
    draws = {q: np.empty(n_boot) for q in est}
    for b in range(n_boot):
        sel = rng.randint(0, S, size=S)
        a = agg(sel, rng)
        draws["delta_contacts"][b] = a["contacts"][hi] - a["contacts"][lo]
        draws["delta_rg2"][b] = a["rg2"][hi] - a["rg2"][lo]
        for k in keys:
            draws[f"slope_{k}"][b] = _slope(K, a[k])
    n_req = int(n_requested_seeds) if n_requested_seeds is not None else int(S)
    out = {}
    for q, arr in draws.items():
        arr = arr[np.isfinite(arr)]
        if arr.size:
            ci = [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))]
            bm = float(arr.mean())
        else:
            ci, bm = [float("nan"), float("nan")], float("nan")
        out[q] = {"estimate": est[q], "bootstrap_mean": bm, "ci95": ci,
                  "n_seeds": int(S), "n_complete_seeds": int(S),
                  "n_requested_seeds": n_req,
                  "block_length": int(block_len),
                  "n_bootstrap_replicates": int(n_boot),
                  "bootstrap_seed": int(seed)}
    out["endpoint_lanes"] = {"low_K_lane": lo, "high_K_lane": hi}
    return out


# ---------------------------------------------------------------------------
# Regime classification (Phase 7.4)
# ---------------------------------------------------------------------------

def _norm(a, invert=False):
    a = np.asarray(a, float)
    lo, hi = np.nanmin(a), np.nanmax(a)
    if not np.isfinite(lo) or hi <= lo:
        return np.zeros_like(a)
    z = (a - lo) / (hi - lo)
    return 1.0 - z if invert else z


def classify_regimes(K, m_mean, rg2_mean, global_frac=None, smax_frac=None,
                     within_lane_std=None, min_effect_size=2.0):
    """Classify each lane as swollen/crossover/collapsed from multiple signals.

    Rule (documented in the report): build a per-lane collapse coordinate in
    [0,1] as the mean of available normalized signals -- normalized mean
    contacts, inverted normalized Rg^2, normalized global-contact fraction,
    normalized S_max/N -- where higher means more collapsed.  Terciles of the
    coordinate label swollen (<1/3), crossover (<2/3), collapsed (>=2/3).
    ``distinct_regimes_resolved`` additionally requires the endpoint contact
    effect size (|Δ<m>|/pooled within-lane std) to exceed ``min_effect_size``
    and at least one lane in each of the swollen and collapsed classes.
    """
    signals = [_norm(m_mean), _norm(rg2_mean, invert=True)]
    if global_frac is not None:
        signals.append(_norm(global_frac))
    if smax_frac is not None:
        signals.append(_norm(smax_frac))
    coord = np.mean(np.vstack(signals), axis=0)
    labels = np.where(coord < 1 / 3, "swollen",
                      np.where(coord < 2 / 3, "crossover", "collapsed"))
    m = np.asarray(m_mean, float)
    order = np.argsort(K)
    dm = float(m[order][-1] - m[order][0])
    pooled = (float(np.nanmean(within_lane_std))
              if within_lane_std is not None
              and np.any(np.isfinite(within_lane_std)) else float("nan"))
    effect = abs(dm) / pooled if (pooled and np.isfinite(pooled) and pooled > 0) \
        else float("inf") if dm != 0 else 0.0
    has_both = ("swollen" in labels) and ("collapsed" in labels)
    resolved = bool(has_both and effect >= min_effect_size)
    return {
        "labels": [str(x) for x in labels],
        "collapse_coordinate": [float(x) for x in coord],
        "endpoint_effect_size_contacts": float(effect),
        "distinct_regimes_resolved": resolved,
        "rule": ("mean of normalized [m, 1-Rg2, global_frac, Smax/N]; terciles "
                 "label regimes; resolved requires effect size >= "
                 f"{min_effect_size} and >=1 swollen and >=1 collapsed lane"),
    }


def aggregate_lane_observables(prefixes, info, n_beads):
    """Seed-aligned per-lane aggregation of the regime observables (Part 5).

    Lanes are aligned by ascending temperature (deterministic, seed-order
    invariant).  Returns per-lane mean across seeds, between-seed std, pooled
    within-seed std (from C_std), and the per-seed matrices for each of: mean
    contacts, Rg^2, global-contact fraction, S_max/N, largest-component
    fraction.  ``K`` is recomputed from the aligned temperatures.
    """
    cols = {
        "m": "C_mean", "rg2": "Rg2_mean",
        "global_frac": "m_global_scaled_mean",
        "smax": "Smax_mean",
        "largest_component_fraction": "largest_component_fraction_mean",
    }
    per_seed_rows = [sorted(_read_results(p), key=lambda r: float(r["T"]))
                     for p in prefixes]
    nT = min(len(r) for r in per_seed_rows)
    per_seed_rows = [rows[:nT] for rows in per_seed_rows]
    T = np.array([float(per_seed_rows[0][k]["T"]) for k in range(nT)])
    K = np.array([_K_of(info, t) for t in T])
    S = len(per_seed_rows)
    aggregate, between_std, per_seed = {}, {}, {}
    for name, col in cols.items():
        M = np.array([[float(rows[k][col]) for k in range(nT)]
                      for rows in per_seed_rows])          # (S, nT)
        aggregate[name] = M.mean(axis=0)
        between_std[name] = (M.std(axis=0, ddof=1) if S > 1 else np.zeros(nT))
        per_seed[name] = M
    cstd = np.array([[float(rows[k].get("C_std", 0.0)) for k in range(nT)]
                     for rows in per_seed_rows])
    aggregate["smax_over_N"] = aggregate["smax"] / max(1, int(n_beads))
    return {
        "n_temperatures": nT, "n_seeds": S,
        "temperatures": T, "K": K,
        "aggregate": aggregate, "between_seed_std": between_std,
        "per_seed": per_seed, "within_lane_std": cstd.mean(axis=0),
    }


# ---------------------------------------------------------------------------
# High-contact tail (Phase 9)
# ---------------------------------------------------------------------------

def high_contact_tail(Pc, c_vals, ess_by_lane, n_by_lane, tau_by_lane,
                      threshold=None, percentile=90.0,
                      support_maximum=None, support_source=None):
    """High-contact-tail diagnostics from the sampled distribution.

    The tail threshold is the pooled-sampled ``percentile`` (default 90th) of
    the contact count, NOT a percentile of the integer support grid.  Effective
    tail counts use per-lane ESS.

    Support-boundary truncation (Part 7) is decided ONLY against a genuine
    external reference (an exact known maximum contact count for N, an externally
    supplied maximum, a baseline maximum, or a fixed pre-sampling histogram
    bound).  When none is available, ``support_boundary_source`` is
    ``"unavailable"`` and ``maximum_is_boundary_limited`` is ``None`` -- it is
    NEVER inferred from the observed maximum against a dynamically generated
    histogram edge (which would make it nearly always True).
    """
    Pc = np.asarray(Pc, float)
    c_vals = np.asarray(c_vals, float)
    nT = Pc.shape[0]
    weights = np.nan_to_num(Pc).sum(axis=1)
    pooled = np.nan_to_num(Pc).sum(axis=0)
    ps = pooled.sum()
    pooled = pooled / ps if ps > 0 else pooled
    if threshold is None:
        cdf = np.cumsum(pooled)
        k = int(np.searchsorted(cdf, percentile / 100.0))
        threshold = float(c_vals[min(k, c_vals.size - 1)])
    tail_mask = c_vals >= threshold
    per_lane = []
    raw_total = 0.0
    eff_total = 0.0
    max_obs = 0.0
    for i in range(nT):
        p = np.nan_to_num(Pc[i])
        tp = float(p[tail_mask].sum())
        n_i = float(n_by_lane[i]) if i < len(n_by_lane) else 0.0
        e_i = float(ess_by_lane[i]) if i < len(ess_by_lane) else 0.0
        raw = tp * n_i
        eff = tp * e_i
        raw_total += raw
        eff_total += eff
        nz = np.nonzero(p > 0)[0]
        if nz.size:
            max_obs = max(max_obs, float(c_vals[nz[-1]]))
        per_lane.append({"temperature_index": i, "tail_probability": tp,
                         "raw_tail_count": raw, "effective_tail_count": eff})
    # Genuine support reference (Part 7): use an externally supplied maximum;
    # otherwise the boundary is UNAVAILABLE and truncation is NOT inferred from
    # the observed maximum against a dynamic histogram edge.
    if support_maximum is not None and np.isfinite(support_maximum):
        boundary_value = float(support_maximum)
        boundary_src = support_source or "external"
        boundary_limited = bool(max_obs >= boundary_value)
        distance = float(boundary_value - max_obs)
    else:
        boundary_value = None
        boundary_src = "unavailable"
        boundary_limited = None
        distance = None
    return {
        "tail_threshold": float(threshold),
        "tail_threshold_rule": f"pooled sampled {percentile:.0f}th percentile",
        "raw_tail_count": float(raw_total),
        "tail_probability": float(np.average(
            [pl["tail_probability"] for pl in per_lane],
            weights=weights) if weights.sum() > 0 else 0.0),
        "effective_tail_count": float(eff_total),
        "per_seed_tail_count": None,   # filled by caller across seeds
        "per_lane": per_lane,
        "maximum_observed_m": float(max_obs),
        "highest_histogram_edge": (float(c_vals[-1]) if c_vals.size else None),
        "support_boundary_source": boundary_src,
        "support_boundary_value": boundary_value,
        "distance_to_boundary": distance,
        "maximum_is_boundary_limited": boundary_limited,
    }


def common_tail_statistics(per_seed_Pc, per_seed_counts, per_seed_ess, c_vals,
                           threshold=None, threshold_source=None, percentile=90.0,
                           support_maximum=None, support_source=None):
    """ONE common high-contact tail threshold applied to every seed (Phase 9).

    Every seed's normalized histogram is converted back to RAW counts using the
    actual post-burn-in sample counts, the counts are pooled across all seeds and
    lanes, and a single tail threshold is chosen (user-supplied, or the pooled
    sampled ``percentile``).  Every seed's and the pool's tail statistics use that
    same threshold, so ``raw_tail_count_pooled == sum(raw_tail_count_per_seed)``
    exactly.
    """
    S = len(per_seed_Pc)
    Pc = np.nan_to_num(np.asarray(per_seed_Pc, float))          # (S, nT, W)
    counts = np.asarray(per_seed_counts, float)                 # (S, nT)
    ess = np.asarray(per_seed_ess, float)                       # (S, nT)
    c_vals = np.asarray(c_vals, float)
    raw = Pc * counts[:, :, None]                               # (S, nT, W)
    pooled_counts = raw.sum(axis=(0, 1))                        # (W,)
    if threshold is None:
        tot = float(pooled_counts.sum())
        if tot > 0:
            cdf = np.cumsum(pooled_counts) / tot
            k = int(np.searchsorted(cdf, percentile / 100.0))
            threshold = float(c_vals[min(k, c_vals.size - 1)])
            src = threshold_source or "pooled_sampled_percentile"
        else:
            threshold = float(c_vals[-1]) if c_vals.size else 0.0
            src = threshold_source or "empty_distribution"
    else:
        threshold = float(threshold)
        src = threshold_source or "user_supplied"
    mask = c_vals >= threshold
    nT = counts.shape[1] if counts.ndim == 2 else 0
    raw_per_seed, eff_per_seed, prob_per_seed, max_per_seed = [], [], [], []
    for s in range(S):
        raw_per_seed.append(float(raw[s][:, mask].sum()))
        tp_lane = np.zeros(nT)
        for lane in range(nT):
            n = counts[s, lane]
            tp_lane[lane] = (raw[s, lane][mask].sum() / n) if n > 0 else 0.0
        eff_per_seed.append(float((tp_lane * ess[s]).sum()))
        w = counts[s]
        prob_per_seed.append(
            float(np.average(tp_lane, weights=w)) if w.sum() > 0 else 0.0)
        nz = np.nonzero(np.nan_to_num(Pc[s]).sum(axis=0) > 0)[0]
        max_per_seed.append(float(c_vals[nz[-1]]) if nz.size else 0.0)
    raw_pooled = float(pooled_counts[mask].sum())
    eff_pooled = float(np.sum(eff_per_seed))
    seeds_with_tail = int(sum(1 for r in raw_per_seed if r > 0))
    # Support-boundary truncation only against a genuine external reference.
    if support_maximum is not None and np.isfinite(support_maximum):
        boundary_value = float(support_maximum)
        boundary_src = support_source or "external"
        max_pooled = max(max_per_seed) if max_per_seed else 0.0
        boundary_limited = bool(max_pooled >= boundary_value)
    else:
        boundary_value = None
        boundary_src = "unavailable"
        boundary_limited = None
    return {
        "tail_threshold": float(threshold),
        "threshold_source": src,
        "threshold_percentile": (float(percentile)
                                 if src == "pooled_sampled_percentile" else None),
        "raw_tail_count_per_seed": raw_per_seed,
        "effective_tail_count_per_seed": eff_per_seed,
        "tail_probability_per_seed": prob_per_seed,
        "raw_tail_count_pooled": raw_pooled,
        "effective_tail_count_pooled": eff_pooled,
        "tail_probability_pooled": (raw_pooled / float(pooled_counts.sum())
                                    if pooled_counts.sum() > 0 else 0.0),
        "seed_coverage": seeds_with_tail,
        "n_seeds": int(S),
        "maximum_observed_m_per_seed": max_per_seed,
        "maximum_observed_m_pooled": (max(max_per_seed) if max_per_seed
                                      else float("nan")),
        "support_boundary_source": boundary_src,
        "support_boundary_value": boundary_value,
        "maximum_is_boundary_limited": boundary_limited,
        "pooled_raw_equals_seed_sum": bool(
            abs(raw_pooled - float(np.sum(raw_per_seed))) <= 1e-6),
    }


def evaluate_tail_gate(tail, thresholds=None):
    """Strengthened high-contact-tail gate (Phase 9): pooled raw + effective
    counts, seed coverage, and per-seed support -- not just a probability."""
    t = dict(DEFAULT_GATE_THRESHOLDS)
    if thresholds:
        t.update(thresholds)
    raw_pooled = float(tail.get("raw_tail_count_pooled", 0.0))
    eff_pooled = float(tail.get("effective_tail_count_pooled", 0.0))
    per_seed = list(tail.get("raw_tail_count_per_seed", []))
    n_seeds = max(1, int(tail.get("n_seeds", len(per_seed) or 1)))
    seed_cov = int(tail.get("seed_coverage", 0))
    min_per_seed = float(min(per_seed)) if per_seed else 0.0
    # The common-threshold accounting identity (pooled raw == sum of per-seed raw)
    # is part of the gate: if it is broken the tail statistics are inconsistent.
    accounting_ok = bool(tail.get("pooled_raw_equals_seed_sum", True))
    passed = (raw_pooled >= t["min_raw_pooled_tail_count"]
              and eff_pooled >= t["min_effective_pooled_tail_count"]
              and (seed_cov / n_seeds) >= t["min_seed_tail_fraction"]
              and min_per_seed >= t["min_per_seed_tail_count"]
              and accounting_ok)
    return {
        "passed": bool(passed),
        "raw_tail_count_pooled": raw_pooled,
        "min_raw_pooled_tail_count": t["min_raw_pooled_tail_count"],
        "effective_tail_count_pooled": eff_pooled,
        "min_effective_pooled_tail_count": t["min_effective_pooled_tail_count"],
        "seed_tail_fraction": seed_cov / n_seeds,
        "min_seed_tail_fraction": t["min_seed_tail_fraction"],
        "min_per_seed_tail_count": min_per_seed,
        "min_per_seed_tail_count_threshold": t["min_per_seed_tail_count"],
        "pooled_raw_equals_seed_sum": accounting_ok,
        "explanation": "high-contact tail has adequate pooled + per-seed support "
                       "and consistent common-threshold accounting",
    }


# ---------------------------------------------------------------------------
# Per-regime production estimator (Phase 10)
# ---------------------------------------------------------------------------

def production_requirement(regime_to_lanes, tau_by_lane, post_burnin_frac,
                           snapshot_stride, n_seeds, target_ess=TARGET_EFF_PER_REGIME,
                           limiting_observable_by_lane=None,
                           regime_labels_stable=True):
    """Snapshot-stride-aware required production cycles per structural regime (Part 8).

    Autocorrelation convention (reconciled, Phase 11)
    -------------------------------------------------
    ``tau_int_cycles`` is the integrated autocorrelation time in CYCLE units and
    the diagnostics define ``ESS = n / tau_int``.  Converting to SNAPSHOT-indexed
    units, the saved autocorrelation time is ``tau_saved = tau_int_cycles /
    snapshot_stride`` and the raw saved count is ``n_saved = post_burnin_cycles /
    snapshot_stride``, so ``ESS ~= n_saved / max(1, tau_saved)``.  Written back in
    cycle units this is exactly::

        Delta_eff = max(snapshot_stride, tau_int_cycles)
        ESS_per_lane_per_seed = post_burnin_cycles / Delta_eff

    with NO extra factor of two -- the effective spacing is simply the larger of
    the save stride (coarse saving is already independent) and the autocorrelation
    time (fine saving is autocorrelation-limited), consistent with the
    ``ESS = n / tau_int`` used by the diagnostics.  Required cycles solve
    ``L*n_seeds*(f*P/Delta_eff) >= target_ess``; the recommended length is the
    MAXIMUM required length over all regimes (and, via tau being the max over the
    key observables, over key observables).  ``limiting_observable`` names the
    observable at the regime's slowest lane.
    """
    f = float(post_burnin_frac)
    stride = max(1, int(snapshot_stride))
    per_regime = {}
    required_P = {}
    for regime, lanes in regime_to_lanes.items():
        taus = [float(tau_by_lane[i]) for i in lanes
                if i < len(tau_by_lane) and np.isfinite(tau_by_lane[i])
                and tau_by_lane[i] > 0]
        L = len(lanes)
        limiting_obs = None
        limiting_lane = None
        if taus:
            limiting_lane = max(
                (i for i in lanes if i < len(tau_by_lane)
                 and np.isfinite(tau_by_lane[i]) and tau_by_lane[i] > 0),
                key=lambda i: tau_by_lane[i], default=None)
            if (limiting_observable_by_lane is not None
                    and limiting_lane is not None
                    and limiting_lane < len(limiting_observable_by_lane)):
                limiting_obs = limiting_observable_by_lane[limiting_lane]
        if not taus or L == 0 or f <= 0:
            per_regime[regime] = {
                "lane_indices": list(lanes), "number_of_lanes": L,
                "number_of_seeds": int(n_seeds), "snapshot_stride": stride,
                "post_burnin_frac": f, "limiting_observable": limiting_obs,
                "limiting_lane": limiting_lane, "limiting_tau_int_cycles": None,
                "slowest_tau_int_cycles": None, "effective_sample_spacing_cycles": None,
                "required_production_cycles": None, "reaches_target": False,
                "limiting_reason": "no valid tau / lanes"}
            required_P[regime] = float("inf")
            continue
        tau = max(taus)
        delta_eff = max(float(stride), float(tau))
        P = int(np.ceil(target_ess * delta_eff / max(1e-9, (L * n_seeds * f))))
        required_P[regime] = P
        per_regime[regime] = {
            "lane_indices": list(lanes), "number_of_lanes": L,
            "number_of_seeds": int(n_seeds), "snapshot_stride": stride,
            "post_burnin_frac": f, "limiting_observable": limiting_obs,
            "limiting_lane": limiting_lane,
            "limiting_tau_int_cycles": float(tau),
            "slowest_tau_int_cycles": float(tau),
            "effective_sample_spacing_cycles": float(delta_eff),
            "required_production_cycles": int(P),
        }
    P_rec = int(max([v for v in required_P.values() if np.isfinite(v)] or [0]))
    all_reach = True
    for regime, lanes in regime_to_lanes.items():
        rr = per_regime[regime]
        tau = rr.get("slowest_tau_int_cycles")
        L = rr["number_of_lanes"]
        if tau and L:
            delta_eff = rr["effective_sample_spacing_cycles"]
            raw_per_lane_per_seed = int(np.floor(f * P_rec / stride))
            eff_per_lane_per_seed = float(f * P_rec / delta_eff)
            eff = L * n_seeds * eff_per_lane_per_seed
            rr["expected_raw_saved_configs_per_lane_per_seed"] = raw_per_lane_per_seed
            rr["expected_effective_configs_per_lane_per_seed"] = eff_per_lane_per_seed
            rr["expected_effective_configs_pooled"] = float(eff)
            rr["expected_raw_saved_configs_per_seed"] = raw_per_lane_per_seed * L
            rr["expected_raw_saved_configs_per_regime"] = (
                raw_per_lane_per_seed * L * int(n_seeds))
            rr["reaches_target"] = bool(eff >= target_ess)
        else:
            rr["expected_raw_saved_configs_per_lane_per_seed"] = 0
            rr["expected_effective_configs_per_lane_per_seed"] = 0.0
            rr["expected_effective_configs_pooled"] = 0.0
            rr["expected_raw_saved_configs_per_seed"] = 0
            rr["expected_raw_saved_configs_per_regime"] = 0
            rr["reaches_target"] = False
        # Phase 8/12: a regime is DEFINITIVE only when it reaches target AND the
        # regime labels are stable across seeds; otherwise it is provisional.
        rr["status"] = ("definitive" if (rr["reaches_target"]
                        and regime_labels_stable) else "provisional")
        all_reach = all_reach and rr["reaches_target"]
    limiting = max(required_P, key=lambda r: required_P[r]) if required_P else None
    overall_status = ("definitive" if (all_reach and regime_labels_stable)
                      else "provisional")
    return {
        "per_regime": per_regime,
        "status": overall_status,
        "regime_labels_stable": bool(regime_labels_stable),
        "recommended_production_cycles_per_seed": int(P_rec),
        "target_effective_per_regime": int(target_ess),
        "limiting_regime": limiting,
        "post_burnin_frac": f,
        "snapshot_stride": stride,
        "n_seeds": int(n_seeds),
        "every_regime_reaches_target": bool(all_reach),
        "autocorrelation_convention": (
            "ESS = n/tau_int (diagnostics); production uses Delta_eff = "
            "max(snapshot_stride, tau_int_cycles) so ESS_saved = n_saved/"
            "max(1, tau_saved) with tau_saved = tau_int_cycles/snapshot_stride "
            "-- NO extra factor of two."),
        "rationale": ("effective spacing Delta_eff = max(snapshot_stride, "
                      "tau_int_cycles); effective saved ~ post_burnin_cycles/"
                      "Delta_eff summed over retained lanes and seeds; raw saved ~ "
                      "post_burnin_cycles/snapshot_stride; recommended length is "
                      "the max required length over regimes and key observables."),
    }


# ---------------------------------------------------------------------------
# Sampling / mixing gates (Phase 8)
# ---------------------------------------------------------------------------

def _gate(passed, value, threshold, explanation):
    return {"passed": bool(passed), "value": (None if value is None else float(value)),
            "threshold": float(threshold), "explanation": explanation}


def _band_gate(min_value, min_threshold, max_value, max_threshold, explanation):
    """Explicit two-sided (band) gate that records BOTH bounds (Part 10)."""
    passed = ((min_value is not None and min_value >= min_threshold)
              and (max_value is not None and max_value <= max_threshold))
    return {
        "passed": bool(passed),
        "minimum_value": (None if min_value is None else float(min_value)),
        "minimum_threshold": float(min_threshold),
        "maximum_value": (None if max_value is None else float(max_value)),
        "maximum_threshold": float(max_threshold),
        "explanation": explanation,
    }


def evaluate_sampling_gates(metrics, thresholds=None):
    """Structured gate results from aggregated mixing metrics."""
    t = dict(DEFAULT_GATE_THRESHOLDS)
    if thresholds:
        t.update(thresholds)
    g = {}
    g["min_ess"] = _gate(metrics.get("min_ess", 0) >= t["min_ess"],
                         metrics.get("min_ess", 0), t["min_ess"],
                         "min ESS over key observables (m, Rg2, m_global_scaled, "
                         "Smax, largest_component_fraction)")
    g["min_round_trips"] = _gate(
        metrics.get("min_round_trips", 0) >= t["min_round_trips"],
        metrics.get("min_round_trips", 0), t["min_round_trips"],
        "minimum walker round trips low<->high")
    g["min_temp_coverage"] = _gate(
        metrics.get("min_temp_coverage", 0) >= t["min_temp_coverage"],
        metrics.get("min_temp_coverage", 0), t["min_temp_coverage"],
        "minimum fraction of temperatures visited by a walker")
    g["min_adjacent_overlap"] = _gate(
        metrics.get("min_adjacent_overlap", 0) >= t["min_adjacent_overlap"],
        metrics.get("min_adjacent_overlap", 0), t["min_adjacent_overlap"],
        "minimum POOLED adjacent contact-histogram Bhattacharyya overlap")
    # Phase 10: a separate worst-SEED gate so a single seed with poor ladder
    # connectivity cannot be hidden by a healthy pooled overlap.
    g["worst_seed_adjacent_overlap"] = _gate(
        metrics.get("min_worst_seed_adjacent_overlap", 0)
        >= t["min_worst_seed_adjacent_overlap"],
        metrics.get("min_worst_seed_adjacent_overlap", 0),
        t["min_worst_seed_adjacent_overlap"],
        "minimum worst-SEED adjacent overlap (per-seed ladder connectivity)")
    g["swap_rate"] = _band_gate(
        metrics.get("min_swap_rate", 0), t["min_swap_rate"],
        metrics.get("max_swap_rate", 1), t["max_swap_rate"],
        "swap acceptance rates within the [min_swap_rate, max_swap_rate] band")
    g["drift"] = _gate(metrics.get("max_drift_in_std", 0) <= t["max_drift_in_std"],
                       metrics.get("max_drift_in_std", 0), t["max_drift_in_std"],
                       "absence of severe early/late drift (in std units)")
    g["state_changing_acceptance"] = _gate(
        metrics.get("min_state_changing_acceptance", 0)
        >= t["min_state_changing_acceptance"],
        metrics.get("min_state_changing_acceptance", 0),
        t["min_state_changing_acceptance"],
        "minimum state-changing local-move acceptance")
    # Phase 9: the STRONG tail gate is authoritative -- pooled raw + effective
    # counts, seed coverage, minimum per-seed tail, and the common-threshold
    # accounting identity (pooled raw == sum of per-seed raw).  It is never
    # replaced by a weaker probability-only recompute.
    tail_gate = metrics.get("tail_gate")
    if tail_gate is None:
        tail_gate = evaluate_tail_gate({
            "raw_tail_count_pooled": metrics.get("raw_tail_count_pooled", 0.0),
            "effective_tail_count_pooled": metrics.get(
                "effective_tail_count_pooled", 0.0),
            "raw_tail_count_per_seed": metrics.get("raw_tail_count_per_seed", []),
            "n_seeds": metrics.get("n_seeds", 1),
            "seed_coverage": metrics.get("seed_coverage", 0),
            "pooled_raw_equals_seed_sum": metrics.get(
                "pooled_raw_equals_seed_sum", True),
        }, thresholds)
    g["high_contact_tail_support"] = tail_gate
    g["seed_agreement"] = _gate(
        metrics.get("seed_relative_spread", 1.0) <= t["max_seed_relative_spread"],
        metrics.get("seed_relative_spread", 1.0), t["max_seed_relative_spread"],
        "endpoint Δ<m> agrees across seeds (relative spread)")
    all_pass = all(v["passed"] for v in g.values())
    g["_all_passed"] = bool(all_pass)
    return g


# ---------------------------------------------------------------------------
# Run + provenance (Phase 11)
# ---------------------------------------------------------------------------

def _sha256(path):
    if path is None or not Path(path).exists():
        return "unknown"
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# Bump when the resume-validation SEMANTICS change (a certificate written by an
# older revision must not be trusted for reuse).
VALIDATOR_REVISION = 3


def _certificate_path(feat_path):
    return Path(str(feat_path) + ".validation.json")


def _canonical_json(obj):
    """Deterministic, key-sorted JSON for fingerprinting (JSON-safe first)."""
    return json.dumps(sch.json_safe(obj), sort_keys=True, separators=(",", ":"))


def _stage_fingerprint(fields):
    """SHA-256 of the canonical calibration-stage fingerprint fields (Phase 4)."""
    return hashlib.sha256(_canonical_json(fields).encode("utf-8")).hexdigest()


def build_stage_fingerprint_fields(*, N, seed, ladder, K_ladder, info,
                                   fit_summary_path, n_cycles, steps_per_swap,
                                   n_workers, structural_stride, snapshot_stride,
                                   burnin_frac, context=None):
    """Canonical fields that uniquely identify a requested calibration stage.

    Any change to N, seed, the ladders, the fitted model (path/hash/params), the
    sampler control parameters, the definitions, or the schema/validator
    revisions produces a different fingerprint, so a resumed stage can be reused
    only when it was produced for exactly the requested configuration.

    The definitions path/hash/version and the output schema versions are taken
    from ONE resolved :class:`DefinitionsContext` (``context`` or ``sch.CONTEXT``)
    rather than a mix of frozen module globals and dynamic accessors (Phase 8).
    """
    ctx = context if context is not None else sch.CONTEXT
    sv = ctx.schema_versions
    return {
        "N": int(N),
        "seed": int(seed),
        "temperature_ladder": [round(float(t), 6) for t in ladder],
        "K_ladder": [round(float(k), 8) for k in K_ladder],
        "fit_summary_path": str(fit_summary_path),
        "fit_summary_sha256": _sha256(fit_summary_path),
        "model_name": str(info["model_name"]),
        "param_names": [str(x) for x in info.get("param_names", [])],
        "model_params": [float(x) for x in info["params"]],
        "Tref": float(info["Tref"]),
        "Tscale": float(info["Tscale"]),
        "n_cycles": int(n_cycles),
        "steps_per_swap": int(steps_per_swap),
        "n_workers": int(n_workers),
        "structural_stride": int(structural_stride),
        "snapshot_stride": int(snapshot_stride),
        "burnin_frac": float(burnin_frac),
        "definitions_path": str(ctx.resolved_path),
        "definitions_sha256": str(ctx.sha256),
        "definitions_version": str(ctx.definitions_version),
        "sampler_schema_version": int(sv["distributions_schema_version"]),
        "snapshot_schema_version": int(sv["snapshot_schema_version"]),
        "feature_schema_version": int(sv["feature_schema_version"]),
        "diagnostic_trajectory_schema_version": int(
            sv["diagnostic_trajectory_schema_version"]),
        "validator_revision": int(VALIDATOR_REVISION),
    }


def _embed_fingerprint_hdf5(path, fingerprint, fields):
    """Embed the stage fingerprint (+canonical field record) into an HDF5 file.

    Adds a top-level ``stage_fingerprint`` attribute and, when a ``/metadata``
    group is present, the fingerprint plus the JSON field record there, so the
    configuration/feature artifact carries an independent copy of the fingerprint
    it was produced for (Phase 4).
    """
    import h5py
    if not Path(path).exists():
        return
    with h5py.File(path, "r+") as f:
        f.attrs["stage_fingerprint"] = str(fingerprint)
        if "metadata" in f:
            f["metadata"].attrs["stage_fingerprint"] = str(fingerprint)
            f["metadata"].attrs["stage_fingerprint_fields"] = json.dumps(
                sch.json_safe(fields))


def _embed_fingerprint_feature(feat_path, fingerprint, fields):
    """Embed the fingerprint into the feature file manifest AND its companion
    ``<feature>.manifest.json`` (Phase 4)."""
    import h5py
    if Path(feat_path).exists():
        with h5py.File(feat_path, "r+") as f:
            f.attrs["stage_fingerprint"] = str(fingerprint)
            man = json.loads(f["metadata"].attrs["manifest"])
            man["stage_fingerprint"] = str(fingerprint)
            man["stage_fingerprint_fields"] = sch.json_safe(fields)
            f["metadata"].attrs["manifest"] = json.dumps(man, default=str)
    comp = Path(str(feat_path) + ".manifest.json")
    if comp.exists():
        m = json.loads(comp.read_text(encoding="utf-8"))
        m["stage_fingerprint"] = str(fingerprint)
        m["stage_fingerprint_fields"] = sch.json_safe(fields)
        comp.write_text(json.dumps(m, indent=2), encoding="utf-8")


def _embed_fingerprint_npz(path, fingerprint, fields):
    """Embed the fingerprint into a diagnostic-trajectory NPZ (Phase 4)."""
    if not Path(path).exists():
        return
    with np.load(path, allow_pickle=True) as z:
        data = {k: z[k] for k in z.files
                if k not in ("stage_fingerprint", "stage_fingerprint_fields")}
    data["stage_fingerprint"] = np.array(str(fingerprint))
    data["stage_fingerprint_fields"] = np.array(
        json.dumps(sch.json_safe(fields)))
    np.savez(path, **data)


def _companion_paths(prefix, cfg, feat):
    """Every per-seed artifact a stage must produce, including the feature
    manifest, validation certificate, run summary, and the artifact manifest
    (Phase 6/7).  This is the single canonical companion set used by resume,
    audit, and artifact-manifest generation."""
    paths = _full_companion_paths(prefix, cfg, feat)
    paths["artifact_manifest_json"] = f"{prefix}_artifact_manifest.json"
    return paths


def _deep_validate_and_certify(feat_path, *, stage_fingerprint=None,
                               stage_fingerprint_fields=None,
                               source_config=None):
    """Deep-validate a completed feature file and write a full certificate.

    Raises ``ExtractionError`` on any structural/identity failure (the caller
    treats that as "the stage must rerun or fail clearly").  The certificate
    records the stage fingerprint and source-configuration hash so a later
    ``--resume`` can prove the file belongs to the exact requested stage.
    """
    info = ext.validate_feature_file_hdf5(
        str(feat_path), deep=True, require_source=True,
        source_path_override=(str(source_config) if source_config else None))
    src_sha = (_sha256(source_config)
               if source_config and Path(source_config).exists() else None)
    cert = {
        "feature_path": str(feat_path),
        "feature_sha256": _sha256(feat_path),
        "source_configuration_path": (str(source_config)
                                      if source_config else None),
        "source_configuration_sha256": src_sha,
        "stage_fingerprint": stage_fingerprint,
        "stage_fingerprint_fields": (sch.json_safe(stage_fingerprint_fields)
                                     if stage_fingerprint_fields is not None
                                     else None),
        "validator_revision": int(VALIDATOR_REVISION),
        "validation_timestamp": _dt.datetime.now().isoformat(),
        "validator_schema_version": int(ext.FEATURE_SCHEMA_VERSION),
        "feature_schema_version": int(ext.FEATURE_SCHEMA_VERSION),
        "definitions_version": info.get("definitions_version"),
        "definitions_sha256": _sha256(str(sch.RESOLVED_DEFINITIONS_PATH)),
        "source_verification_status": info.get("source_verification_status"),
        "validation_result": "passed",
    }
    _certificate_path(feat_path).write_text(
        json.dumps(cert, indent=2), encoding="utf-8")
    return cert


def validate_feature_certificate(feature_path, *, source_configuration_path,
                                 expected_stage_fingerprint,
                                 expected_validator_revision,
                                 expected_definitions_sha256,
                                 expected_definitions_version=None,
                                 expected_feature_schema_version=None):
    """The ONE shared certificate-trust function (Phase 5).

    Returns ``(trusted: bool, reason: str)``.  A certificate is trusted ONLY when
    every one of the following holds, so no older certificate is accepted merely
    because its feature hash still matches:

    * ``validation_result == "passed"``;
    * the certificate references this exact feature path;
    * the recorded feature SHA-256 equals the current file hash;
    * the source configuration exists and its SHA-256 equals the recorded hash;
    * the stage fingerprint equals the expected stage fingerprint;
    * the validator revision equals the expected revision;
    * the definitions SHA-256 (and, when given, version) equal the expected ones;
    * the feature schema version equals the expected one (when given).

    Used by resume validation, the seed-artifact audit, and scientific
    calibration analysis alike.
    """
    feature_path = Path(feature_path)
    if not feature_path.exists():
        return False, "feature file missing"
    cert_p = _certificate_path(feature_path)
    if not cert_p.exists():
        return False, "no validation certificate"
    try:
        cert = json.loads(cert_p.read_text(encoding="utf-8"))
    except Exception:
        return False, "unreadable certificate"
    if cert.get("validation_result") != "passed":
        return False, "certificate validation_result != passed"
    cert_feat = cert.get("feature_path")
    if cert_feat is not None and Path(str(cert_feat)).name != feature_path.name:
        return False, "certificate references a different feature path"
    if cert.get("feature_sha256") != _sha256(feature_path):
        return False, "feature file hash changed since certification"
    if int(cert.get("validator_revision", -1)) != int(expected_validator_revision):
        return False, "certificate validator revision differs from expected"
    if cert.get("stage_fingerprint") != expected_stage_fingerprint:
        return False, "stage fingerprint differs from the expected stage"
    if (not source_configuration_path
            or not Path(source_configuration_path).exists()):
        return False, "source configuration missing"
    if cert.get("source_configuration_sha256") != _sha256(source_configuration_path):
        return False, "source configuration hash changed"
    if cert.get("definitions_sha256") != expected_definitions_sha256:
        return False, "definitions hash differs from expected"
    if (expected_definitions_version is not None
            and str(cert.get("definitions_version"))
            != str(expected_definitions_version)):
        return False, "definitions version differs from expected"
    if (expected_feature_schema_version is not None
            and int(cert.get("feature_schema_version",
                             cert.get("validator_schema_version", -1)))
            != int(expected_feature_schema_version)):
        return False, "feature schema version differs from expected"
    return True, "certificate trusted"


def _current_certificate_expectations():
    """Expected (defs_sha256, defs_version, feature_schema_version) for the
    installed code + active definitions."""
    return (_sha256(str(sch.RESOLVED_DEFINITIONS_PATH)),
            str(sch.DEFINITIONS_VERSION), int(ext.FEATURE_SCHEMA_VERSION))


def stage_reusable(feat, cfg, stage_fingerprint, companions):
    """(reusable, reason) — a resumed stage may be reused only if EVERYTHING
    matches the requested stage (Phase 4/5).

    Delegates certificate trust to :func:`validate_feature_certificate` (the one
    shared trust function) and additionally requires the source configuration and
    every companion artifact to be present.
    """
    if not Path(cfg).exists():
        return False, "source configuration missing"
    defs_sha, defs_ver, feat_ver = _current_certificate_expectations()
    ok, reason = validate_feature_certificate(
        feat, source_configuration_path=cfg,
        expected_stage_fingerprint=stage_fingerprint,
        expected_validator_revision=VALIDATOR_REVISION,
        expected_definitions_sha256=defs_sha,
        expected_definitions_version=defs_ver,
        expected_feature_schema_version=feat_ver)
    if not ok:
        return False, reason
    for name, path in companions.items():
        if not Path(path).exists():
            return False, f"missing companion artifact: {name}"
    return True, "reusable"


# ---------------------------------------------------------------------------
# Artifact manifest: SHA-256 + SEMANTIC validation of every companion (Phase 6)
# ---------------------------------------------------------------------------

ARTIFACT_MANIFEST_SCHEMA_VERSION = 1


class ArtifactValidationError(RuntimeError):
    """Raised when a companion artifact fails its hash or semantic validation."""


def _full_companion_paths(prefix, cfg, feat):
    """Every artifact a seed stage must produce (Phase 6/7)."""
    feat = str(feat)
    return {
        "configuration_h5": str(cfg),
        "feature_h5": feat,
        "feature_manifest_json": feat + ".manifest.json",
        "validation_certificate": feat + ".validation.json",
        "results_csv": f"{prefix}_results.csv",
        "swap_rates_csv": f"{prefix}_swap_rates.csv",
        "move_acceptance_csv": f"{prefix}_move_acceptance.csv",
        "diagnostics_json": f"{prefix}_diagnostics.json",
        "diagnostic_trajectories_npz": f"{prefix}_diagnostic_trajectories.npz",
        "distributions_npz": f"{prefix}_distributions.npz",
        "run_summary_json": f"{prefix}_run_summary.json",
    }


def build_artifact_manifest(paths, out_path):
    """Write an artifact manifest recording the SHA-256 of every artifact."""
    entries = {}
    for name, p in paths.items():
        entries[name] = {"path": str(p), "exists": bool(Path(p).exists()),
                         "sha256": _sha256(p)}
    man = {"artifact_manifest_schema_version": ARTIFACT_MANIFEST_SCHEMA_VERSION,
           "artifacts": entries}
    Path(out_path).write_text(json.dumps(sch.json_safe(man), indent=2),
                              encoding="utf-8")
    return man


def _ladder_close(a, b, tol=1e-3):
    a = np.asarray(a, float).ravel()
    b = np.asarray(b, float).ravel()
    return a.size == b.size and np.allclose(a, b, atol=tol, rtol=0.0)


def _read_csv_rows(path):
    import csv as _csv
    with open(path, newline="", encoding="utf-8") as f:
        r = _csv.DictReader(f)
        return list(r.fieldnames or []), list(r)


def validate_config_artifact(path, *, N, seed, ladder, stage_fingerprint=None,
                             run_id=None):
    import h5py
    with h5py.File(path, "r") as f:
        if str(f.attrs.get("status", f["snapshots"].attrs.get("status"))) \
                != "complete":
            raise ArtifactValidationError("configuration status != complete")
        snap = f["snapshots"]
        coords = snap["coordinates"]
        if int(coords.shape[2]) != int(N):
            raise ArtifactValidationError(
                f"configuration N {coords.shape[2]} != {N}")
        if int(cio.committed_rows(snap)) <= 0:
            raise ArtifactValidationError("configuration has no committed rows")
        meta = {k: v for k, v in f["metadata"].attrs.items()}
        ms = meta.get("seed")
        if ms is not None and int(require_exact_integer_from(ms)) != int(seed):
            raise ArtifactValidationError("configuration seed mismatch")
        temps = np.asarray(meta.get("temperatures", []), float).ravel()
        if temps.size and not _ladder_close(temps, ladder):
            raise ArtifactValidationError("configuration temperature ladder mismatch")
        if meta.get("model_name") is None:
            raise ArtifactValidationError("configuration missing model_name")
        rid = meta.get("run_id")
        if isinstance(rid, bytes):
            rid = rid.decode()
        if run_id is not None and rid is not None and str(rid) != str(run_id):
            raise ArtifactValidationError("configuration run_id mismatch")
        if stage_fingerprint is not None:
            if str(f.attrs.get("stage_fingerprint")) != str(stage_fingerprint):
                raise ArtifactValidationError("configuration stage fingerprint mismatch")


def require_exact_integer_from(v):
    """Reject a fractional value from an artifact scalar (delegates to ext)."""
    return ext.require_exact_integer_scalar(v, field_name="artifact scalar")


def validate_feature_artifact(path, *, source_config, stage_fingerprint):
    ext.validate_feature_file_hdf5(
        str(path), deep=True, require_source=True,
        source_path_override=(str(source_config) if source_config else None))
    if stage_fingerprint is not None:
        import h5py
        with h5py.File(path, "r") as f:
            if str(f.attrs.get("stage_fingerprint")) != str(stage_fingerprint):
                raise ArtifactValidationError("feature stage fingerprint mismatch")


def validate_distributions_artifact(path, *, N, ladder):
    with np.load(path, allow_pickle=True) as z:
        if "Ts" not in z or not _ladder_close(z["Ts"], ladder):
            raise ArtifactValidationError("distributions Ts != requested ladder")
        if "temps" in z and not _ladder_close(z["temps"], z["Ts"]):
            raise ArtifactValidationError("distributions temps alias != Ts")
        nT = len(np.asarray(ladder).ravel())
        for key in ("Pc", "Prg"):
            if key not in z:
                raise ArtifactValidationError(f"distributions missing {key}")
            if int(np.asarray(z[key]).shape[0]) != nT:
                raise ArtifactValidationError(
                    f"distributions {key} lane count != {nT}")
        if "model_name" not in z:
            raise ArtifactValidationError("distributions missing model_name")
        for key in ("n_beads", "N"):
            if key in z and int(np.asarray(z[key]).ravel()[0]) != int(N):
                raise ArtifactValidationError("distributions N mismatch")


def validate_diagnostic_trajectories_artifact(path, *, ladder,
                                              stage_fingerprint=None):
    with np.load(path, allow_pickle=True) as z:
        if "Ts" not in z or not _ladder_close(z["Ts"], ladder):
            raise ArtifactValidationError(
                "diagnostic trajectories Ts != requested ladder")
        nT = len(np.asarray(ladder).ravel())
        required = ("contacts_post", "rg2_post", "m_long_post",
                    "m_global_scaled_post", "smax_post",
                    "largest_component_fraction_post")
        for key in required:
            if key not in z:
                raise ArtifactValidationError(
                    f"diagnostic trajectories missing {key}")
            if int(np.asarray(z[key]).shape[0]) != nT:
                raise ArtifactValidationError(
                    f"diagnostic trajectories {key} lane count != {nT}")
        if stage_fingerprint is not None and "stage_fingerprint" in z:
            if str(z["stage_fingerprint"]) != str(stage_fingerprint):
                raise ArtifactValidationError(
                    "diagnostic trajectories stage fingerprint mismatch")


def validate_diagnostics_json_artifact(path, *, ladder):
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    nT = len(np.asarray(ladder).ravel())
    temps = d.get("temperatures")
    if temps is None:
        temps = (d.get("summary", {}) or {}).get("temperatures")
    if temps is not None:
        if len(temps) != nT or not _ladder_close(temps, ladder):
            raise ArtifactValidationError("diagnostics temperatures != ladder")
    lc = d.get("n_temperatures", d.get("lane_count"))
    if lc is not None and int(lc) != nT:
        raise ArtifactValidationError("diagnostics lane count != ladder size")
    if "burnin_frac" not in d and "burnin_frac" not in (d.get("summary", {}) or {}):
        raise ArtifactValidationError("diagnostics missing burnin_frac")


def validate_results_csv_artifact(path, *, ladder):
    header, rows = _read_csv_rows(path)
    nT = len(np.asarray(ladder).ravel())
    if len(rows) != nT:
        raise ArtifactValidationError(
            f"results CSV has {len(rows)} rows != {nT} lanes")
    for col in ("T", "C_mean", "Rg2_mean_lattice"):
        if col not in header:
            raise ArtifactValidationError(f"results CSV missing column {col}")
    temps = [float(r["T"]) for r in rows]
    if len(set(round(t, 4) for t in temps)) != nT:
        raise ArtifactValidationError("results CSV has duplicate lane temperatures")
    if not _ladder_close(temps, ladder):
        raise ArtifactValidationError("results CSV temperatures != ladder")


def validate_swap_rates_csv_artifact(path, *, ladder):
    header, rows = _read_csv_rows(path)
    nT = len(np.asarray(ladder).ravel())
    if len(rows) != nT - 1:
        raise ArtifactValidationError(
            f"swap-rates CSV has {len(rows)} rows != {nT - 1} pairs")
    for col in ("pair", "T_lo", "T_hi"):
        if col not in header:
            raise ArtifactValidationError(f"swap-rates CSV missing column {col}")
    for k, r in enumerate(rows):
        if int(r["pair"]) != k:
            raise ArtifactValidationError("swap-rates pair indices not contiguous")
        if (abs(float(r["T_lo"]) - float(ladder[k])) > 1e-3
                or abs(float(r["T_hi"]) - float(ladder[k + 1])) > 1e-3):
            raise ArtifactValidationError(
                f"swap-rates pair {k} T_lo/T_hi disagree with adjacent ladder")


def validate_move_acceptance_csv_artifact(path, *, ladder):
    header, rows = _read_csv_rows(path)
    nT = len(np.asarray(ladder).ravel())
    for col in ("temperature_index", "temperature", "move_type"):
        if col not in header:
            raise ArtifactValidationError(f"move-acceptance CSV missing {col}")
    lanes = sorted(set(int(r["temperature_index"]) for r in rows))
    if lanes != list(range(nT)):
        raise ArtifactValidationError("move-acceptance CSV lanes != 0..nT-1")
    move_names = set(remd.MOVE_NAMES)
    for i in range(nT):
        lane_moves = set(r["move_type"] for r in rows
                         if int(r["temperature_index"]) == i)
        if lane_moves != move_names:
            raise ArtifactValidationError(
                f"move-acceptance CSV lane {i} move types {lane_moves} "
                f"!= {move_names}")
        lane_temp = [float(r["temperature"]) for r in rows
                     if int(r["temperature_index"]) == i][0]
        if abs(lane_temp - float(ladder[i])) > 1e-3:
            raise ArtifactValidationError(
                f"move-acceptance CSV lane {i} temperature != ladder")


def validate_run_summary_artifact(path, *, N, seed, ladder,
                                  stage_fingerprint=None):
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    if int(d.get("N", d.get("n_beads", -1))) != int(N):
        raise ArtifactValidationError("run-summary N mismatch")
    if int(d.get("seed", -999)) != int(seed):
        raise ArtifactValidationError("run-summary seed mismatch")
    temps = d.get("temperatures")
    if temps is None or not _ladder_close(temps, ladder):
        raise ArtifactValidationError("run-summary temperatures != ladder")
    if d.get("model") is None and d.get("model_name") is None:
        raise ArtifactValidationError("run-summary missing model metadata")
    if stage_fingerprint is not None and "stage_fingerprint" in d:
        if str(d["stage_fingerprint"]) != str(stage_fingerprint):
            raise ArtifactValidationError("run-summary stage fingerprint mismatch")


_ARTIFACT_SEMANTIC = {
    "configuration_h5": lambda p, C: validate_config_artifact(
        p, N=C["N"], seed=C["seed"], ladder=C["ladder"],
        stage_fingerprint=C["stage_fingerprint"], run_id=C.get("run_id")),
    "feature_h5": lambda p, C: validate_feature_artifact(
        p, source_config=C["source_config"],
        stage_fingerprint=C["stage_fingerprint"]),
    "distributions_npz": lambda p, C: validate_distributions_artifact(
        p, N=C["N"], ladder=C["ladder"]),
    "diagnostic_trajectories_npz": lambda p, C:
        validate_diagnostic_trajectories_artifact(
            p, ladder=C["ladder"], stage_fingerprint=C["stage_fingerprint"]),
    "diagnostics_json": lambda p, C: validate_diagnostics_json_artifact(
        p, ladder=C["ladder"]),
    "results_csv": lambda p, C: validate_results_csv_artifact(
        p, ladder=C["ladder"]),
    "swap_rates_csv": lambda p, C: validate_swap_rates_csv_artifact(
        p, ladder=C["ladder"]),
    "move_acceptance_csv": lambda p, C: validate_move_acceptance_csv_artifact(
        p, ladder=C["ladder"]),
    "run_summary_json": lambda p, C: validate_run_summary_artifact(
        p, N=C["N"], seed=C["seed"], ladder=C["ladder"],
        stage_fingerprint=C["stage_fingerprint"]),
}


def validate_artifact_manifest(manifest_path, *, N, seed, ladder, source_config,
                               stage_fingerprint, run_id=None):
    """Validate every artifact's HASH and SEMANTIC content (Phase 6).

    Returns ``(ok, report)`` where ``report`` maps each artifact to ``"ok"`` or a
    failure reason.  A recorded-hash mismatch OR any semantic failure marks that
    artifact (and the whole manifest) invalid.
    """
    man = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    entries = man.get("artifacts", {})
    ctx = {"N": N, "seed": seed, "ladder": ladder,
           "source_config": source_config, "stage_fingerprint": stage_fingerprint,
           "run_id": run_id}
    report = {}
    ok_all = True
    for name, entry in entries.items():
        p = entry.get("path")
        if not p or not Path(p).exists():
            report[name] = "missing"
            ok_all = False
            continue
        if entry.get("sha256") != _sha256(p):
            report[name] = "hash mismatch"
            ok_all = False
            continue
        sem = _ARTIFACT_SEMANTIC.get(name)
        if sem is None:
            report[name] = "ok"
            continue
        try:
            sem(p, ctx)
            report[name] = "ok"
        except Exception as exc:  # ArtifactValidationError / ExtractionError / ...
            report[name] = f"semantic: {exc}"
            ok_all = False
    return bool(ok_all), report


def _fingerprint_field_names():
    """Deterministic sorted list of stage-fingerprint field names."""
    fields = build_stage_fingerprint_fields(
        N=0, seed=0, ladder=[0.0], K_ladder=[0.0],
        info={"model_name": "", "param_names": [], "params": [],
              "Tref": 0.0, "Tscale": 0.0},
        fit_summary_path="__no_such_fit_summary__", n_cycles=0, steps_per_swap=0,
        n_workers=0, structural_stride=0, snapshot_stride=0, burnin_frac=0.0)
    return sorted(fields)


def render_artifact_provenance_md():
    """Deterministic markdown for ``docs/artifact_provenance.md`` (Phase 13).

    Explains the configuration -> feature -> certificate -> seed-audit ->
    calibration-report chain, which SHA-256 binds each stage, which fingerprint
    fields identify a run, which artifacts are authoritative, and which validation
    status is required for scientific use.  Contains NO timestamps or absolute
    paths so it regenerates byte-for-byte.
    """
    artifacts = list(_full_companion_paths("<prefix>", "<cfg>", "<feat>"))
    lines = [
        "# Artifact provenance and trust chain",
        "",
        "> Generated by `run_structural_regime_pilot.render_artifact_provenance_md()`."
        "  Do not edit by hand; regenerate with `python -c \"import "
        "run_structural_regime_pilot as p; p.write_artifact_provenance_md()\"`.",
        "",
        "## Stage chain",
        "",
        "1. **configuration** (`<prefix>_configurations.h5`) — the REMD sampler "
        "writes committed coordinate snapshots plus run metadata (N, seed, "
        "run_id, temperature ladder, fitted-model record).  The stage fingerprint "
        "is embedded as the `stage_fingerprint` attribute.",
        "2. **feature extraction** (`<prefix>_features.h5`) — offline extraction "
        "reads ONLY committed rows, validating raw source integers before any "
        "cast, and writes one validated feature row per (snapshot, lane).  The "
        "manifest records `source_configuration_sha256` binding the feature file "
        "to the exact configuration bytes.",
        "3. **certificate** (`<prefix>_features.h5.validation.json`) — a deep, "
        "SOURCE-VERIFIED validation writes `feature_sha256`, "
        "`source_configuration_sha256`, `stage_fingerprint`, `validator_revision`, "
        "`definitions_sha256`, `definitions_version`, and "
        "`source_verification_status`.",
        "4. **artifact manifest** (`<prefix>_artifact_manifest.json`) — records "
        "the SHA-256 of every companion artifact for hash + semantic validation.",
        "5. **seed audit + calibration report** — every requested seed must be "
        "complete (all artifacts present), certificate-trusted for exactly its "
        "stage fingerprint, and cross-seed aligned before any multi-seed "
        "inference; otherwise the report marks inference `not_assessable`.",
        "",
        "## SHA-256 bindings",
        "",
        "| binding | recorded in | ties |",
        "| --- | --- | --- |",
        "| `source_configuration_sha256` | feature manifest + certificate | "
        "feature file to its configuration bytes |",
        "| `feature_sha256` | certificate | certificate to the feature bytes |",
        "| `definitions_sha256` | certificate + stage fingerprint | run to the "
        "frozen `project_definitions.json` |",
        "| `fit_summary_sha256` | stage fingerprint | run to the fitted-model file |",
        "| artifact `sha256` map | artifact manifest | report to every companion |",
        "",
        "## Stage-fingerprint fields (identify a requested run)",
        "",
    ] + [f"- `{name}`" for name in _fingerprint_field_names()] + [
        "",
        "## Authoritative artifacts (required per seed)",
        "",
    ] + [f"- `{name}`" for name in artifacts] + [
        "- `artifact_manifest_json`",
        "",
        "## Validation status required for scientific use",
        "",
        "- Feature validation must be `fully_source_verified` (the independent "
        "source configuration was found, hash-matched, and agreed on the "
        "temperature ladder, model, per-row b(T), seed, N, and definitions "
        "version).  `internally_consistent_only` is acceptable ONLY for "
        "historical inspection, never for resume or seed analysis.",
        "- A certificate is trusted only when `validation_result == passed` and "
        "its feature hash, source hash, stage fingerprint, validator revision, "
        "definitions hash, definitions version, and feature schema version all "
        "match the expected stage.",
        "- Multi-seed inference is assessable only when every requested seed is "
        "complete, certificate-trusted, and cross-seed aligned.",
        "",
    ]
    return "\n".join(lines)


def write_artifact_provenance_md(path=None):
    p = Path(path) if path is not None else (HERE / "docs" / "artifact_provenance.md")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(render_artifact_provenance_md(), encoding="utf-8")
    return str(p)


def _feature_certified(feat_path):
    """True only when ``feat_path`` is a currently-trustworthy, source-verified
    certified feature file (Phase 5 self-consistency check).

    An existing certificate is reused ONLY when it references this file's current
    hash, was written by the CURRENT validator revision, and matches the current
    definitions hash; otherwise a fresh deep, SOURCE-VERIFIED validation is rerun
    (and a new certificate written).  Any failure returns False so the seed stage
    reruns or fails clearly.  This is the self-consistency check; cross-checking a
    file against an EXTERNALLY expected stage fingerprint is done by
    :func:`validate_feature_certificate`.
    """
    feat_path = Path(feat_path)
    if not feat_path.exists():
        return False
    cert_p = _certificate_path(feat_path)
    defs_sha, _defs_ver, _feat_ver = _current_certificate_expectations()
    if cert_p.exists():
        try:
            cert = json.loads(cert_p.read_text(encoding="utf-8"))
            if (cert.get("validation_result") == "passed"
                    and cert.get("feature_sha256") == _sha256(feat_path)
                    and int(cert.get("validator_revision", -1))
                    == int(VALIDATOR_REVISION)
                    and cert.get("definitions_sha256") == defs_sha):
                return True
        except Exception:
            pass
    try:
        _deep_validate_and_certify(feat_path)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Seed-artifact audit (Phase 5) + cross-seed alignment (Phase 6)
# ---------------------------------------------------------------------------

# The complete artifact set every requested seed must produce before it can
# contribute to a multi-seed statistical analysis (Phase 7).
REQUIRED_SEED_ARTIFACTS = (
    "configuration_h5", "feature_h5", "feature_manifest_json",
    "validation_certificate", "results_csv", "swap_rates_csv",
    "move_acceptance_csv", "diagnostics_json", "diagnostic_trajectories_npz",
    "distributions_npz", "run_summary_json", "artifact_manifest_json")

ALIGN_TOL = 1e-6


class SeedAlignmentError(RuntimeError):
    """Raised when seeds disagree on the physics-defining run configuration."""


def audit_seed_artifacts(seed_companions, *, feature_validator=None,
                         expectations=None):
    """Classify each requested seed as complete / missing / invalid (Phase 5/7).

    ``seed_companions`` maps each requested seed to its companion-artifact path
    dict (as returned by :func:`_companion_paths`).  A seed is COMPLETE only when
    EVERY required artifact (including the artifact manifest) exists and, when
    ``expectations[seed]`` is supplied, its artifact manifest passes full
    HASH + SEMANTIC validation (:func:`validate_artifact_manifest`) AND its
    feature certificate is trusted for exactly the expected stage
    (:func:`validate_feature_certificate`).  Otherwise the lighter
    ``feature_validator`` is used.  Missing/invalid artifacts never silently drop
    a seed; they are reported with a per-artifact reason so scientific inference
    can be marked not assessable.
    """
    validator = feature_validator or _feature_certified
    complete, missing, invalid = [], [], []
    failures = {}
    for seed in sorted(seed_companions):
        comp = seed_companions[seed]
        miss = [n for n in REQUIRED_SEED_ARTIFACTS
                if n not in comp or not Path(comp[n]).exists()]
        if miss:
            missing.append(seed)
            failures[seed] = {"missing_artifacts": miss}
            continue
        if expectations is not None and seed in expectations:
            exp = expectations[seed]
            ok, report = validate_artifact_manifest(
                comp["artifact_manifest_json"], N=exp["N"], seed=exp["seed"],
                ladder=exp["ladder"], source_config=exp["source_config"],
                stage_fingerprint=exp["stage_fingerprint"],
                run_id=exp.get("run_id"))
            if not ok:
                invalid.append(seed)
                failures[seed] = {"artifact_validation": {
                    k: v for k, v in report.items() if v != "ok"}}
                continue
            defs_sha, defs_ver, feat_ver = _current_certificate_expectations()
            cok, creason = validate_feature_certificate(
                comp["feature_h5"],
                source_configuration_path=comp["configuration_h5"],
                expected_stage_fingerprint=exp["stage_fingerprint"],
                expected_validator_revision=VALIDATOR_REVISION,
                expected_definitions_sha256=defs_sha,
                expected_definitions_version=defs_ver,
                expected_feature_schema_version=feat_ver)
            if not cok:
                invalid.append(seed)
                failures[seed] = {"certificate": creason}
                continue
        elif not validator(comp["feature_h5"]):
            invalid.append(seed)
            failures[seed] = {"feature_validation": "failed"}
            continue
        complete.append(seed)
    requested = sorted(seed_companions)
    return {
        "requested_seeds": requested,
        "complete_seeds": complete,
        "missing_seeds": missing,
        "invalid_seeds": invalid,
        "artifact_failures_by_seed": failures,
        "all_requested_complete": bool(len(complete) == len(requested)),
        "assessable": bool(len(complete) == len(requested) and len(complete) >= 2),
    }


def require_seed_alignment(records, tol=ALIGN_TOL):
    """Raise :class:`SeedAlignmentError` unless every seed agrees EXACTLY (within
    a serialized-float tolerance) on the physics-defining quantities (Phase 6).

    Each record is a dict with keys ``N``, ``temperatures`` (per-lane order),
    ``K`` (per-lane order), ``model_name``, ``model_params``, ``Tref``,
    ``Tscale``, ``definitions_version``, ``trajectory_obs`` (iterable of names),
    ``trajectory_shapes`` (name -> shape).  Comparison is against the first
    record but flags ANY deviating seed, so the pass/fail outcome is invariant to
    seed ordering.  Temperatures/K are compared in per-lane order, so a reversed
    or shifted lane assignment fails even though the SORTED ladders would match.
    """
    if not records:
        raise SeedAlignmentError("no seed records to align")
    ref = records[0]

    def _arr(r, k):
        return np.asarray(r[k], float).ravel()

    for i, r in enumerate(records):
        if i == 0:
            continue
        if int(r["N"]) != int(ref["N"]):
            raise SeedAlignmentError(f"seed index {i}: N {r['N']} != {ref['N']}")
        t0, t1 = _arr(ref, "temperatures"), _arr(r, "temperatures")
        if t1.size != t0.size:
            raise SeedAlignmentError(
                f"seed index {i}: temperature count {t1.size} != {t0.size} "
                f"(a seed is missing a lane)")
        if not np.allclose(t1, t0, atol=tol, rtol=0.0):
            if np.allclose(np.sort(t1), np.sort(t0), atol=tol, rtol=0.0):
                raise SeedAlignmentError(
                    f"seed index {i}: temperature lanes are reordered relative "
                    f"to the reference (same sorted ladder, different order)")
            raise SeedAlignmentError(
                f"seed index {i}: temperature ladder differs across seeds")
        k0, k1 = _arr(ref, "K"), _arr(r, "K")
        if k1.size != k0.size or not np.allclose(k1, k0, atol=tol, rtol=0.0):
            raise SeedAlignmentError(f"seed index {i}: K ladder differs")
        if str(r.get("model_name")) != str(ref.get("model_name")):
            raise SeedAlignmentError(f"seed index {i}: model_name differs")
        p0 = [float(x) for x in ref.get("model_params", [])]
        p1 = [float(x) for x in r.get("model_params", [])]
        if len(p1) != len(p0) or any(abs(a - b) > tol for a, b in zip(p0, p1)):
            raise SeedAlignmentError(f"seed index {i}: model parameters differ")
        for key in ("Tref", "Tscale"):
            if ref.get(key) is not None and r.get(key) is not None:
                if abs(float(r[key]) - float(ref[key])) > tol:
                    raise SeedAlignmentError(f"seed index {i}: {key} differs")
        if str(r.get("definitions_version")) != str(ref.get("definitions_version")):
            raise SeedAlignmentError(
                f"seed index {i}: definitions_version differs")
        if set(r.get("trajectory_obs", [])) != set(ref.get("trajectory_obs", [])):
            raise SeedAlignmentError(
                f"seed index {i}: trajectory observable names differ")
        for name, shp in ref.get("trajectory_shapes", {}).items():
            if tuple(r.get("trajectory_shapes", {}).get(name, ())) != tuple(shp):
                raise SeedAlignmentError(
                    f"seed index {i}: trajectory array {name!r} shape differs")
    return {
        "n_seeds": len(records),
        "N": int(ref["N"]),
        "n_temperatures": int(_arr(ref, "temperatures").size),
        "aligned": True,
    }


def run_seed(out_dir, N, seed, ladder, summary_path, n_cycles, steps_per_swap,
             structural_stride, snapshot_stride, resume=False, commands=None,
             stage_fingerprint=None, stage_fingerprint_fields=None,
             burnin_frac=0.5, n_workers=1):
    prefix = out_dir / f"calib_N{N}_s{seed}"
    cfg = f"{prefix}_configurations.h5"
    feat = out_dir / f"calib_N{N}_s{seed}_features.h5"
    companions = _companion_paths(prefix, cfg, feat)
    if resume:
        ok, reason = stage_reusable(feat, cfg, stage_fingerprint, companions)
        if ok:
            print(f"RESUME: seed {seed} reused (stage fingerprint + companions "
                  f"match; {feat})")
            return str(prefix)
        print(f"RESUME: seed {seed} NOT reusable ({reason}); rerunning.")
    temps = ",".join(f"{t:.4f}" for t in ladder)
    cmd = [PY, REMD, "--N", str(N), "--temps", temps,
           "--fit-summary-json", summary_path,
           "--steps-per-swap", str(steps_per_swap), "--n-cycles", str(n_cycles),
           "--n-workers", str(int(n_workers)), "--seed", str(seed),
           "--burnin-frac", str(float(burnin_frac)),
           "--structural-observables", "--structural-stride", str(structural_stride),
           "--diagnostics", "--diagnostic-trajectories",
           "--save-configurations", "--snapshot-stride", str(snapshot_stride),
           "--overwrite-configurations", "--no-plots",
           "--out-prefix", str(prefix)]
    if commands is not None:
        commands.append(" ".join(cmd))
    print("RUN:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(HERE))
    # Embed the stage fingerprint (+field record) into the configuration HDF5 and
    # the diagnostic-trajectory NPZ BEFORE extraction/hashing (Phase 4).
    _embed_fingerprint_hdf5(cfg, stage_fingerprint, stage_fingerprint_fields)
    _embed_fingerprint_npz(companions["diagnostic_trajectories_npz"],
                           stage_fingerprint, stage_fingerprint_fields)
    ecmd = [PY, EXTRACT, "--input", cfg, "--output", str(feat),
            "--validate", "--overwrite"]
    if commands is not None:
        commands.append(" ".join(ecmd))
    subprocess.run(ecmd, check=True, cwd=str(HERE))
    # Embed the same fingerprint into the feature manifest + companion JSON, then
    # deep-validate and write a full certificate (stage fingerprint + field record
    # + source-config hash) so a later --resume can prove the stage belongs to
    # exactly this requested configuration.
    _embed_fingerprint_feature(feat, stage_fingerprint, stage_fingerprint_fields)
    _deep_validate_and_certify(
        feat, stage_fingerprint=stage_fingerprint,
        stage_fingerprint_fields=stage_fingerprint_fields, source_config=cfg)
    # Phase 6: record the SHA-256 of every produced artifact.
    build_artifact_manifest(_full_companion_paths(prefix, cfg, feat),
                            f"{prefix}_artifact_manifest.json")
    return str(prefix)


def _read_results(prefix):
    return list(csv.DictReader(open(f"{prefix}_results.csv", newline="")))


def _read_diag(prefix):
    with open(f"{prefix}_diagnostics.json") as fh:
        return json.load(fh)


def _K_of(info, T):
    return -remd.reduced_bias(info["model_name"], info["params"], float(T),
                              info["Tref"], info["Tscale"])


def _bhattacharyya(p, q):
    p = np.asarray(p, float); q = np.asarray(q, float)
    p = np.where(np.isfinite(p), p, 0.0); q = np.where(np.isfinite(q), q, 0.0)
    sp, sq = p.sum(), q.sum()
    if sp <= 0 or sq <= 0:
        return 0.0
    return float(np.sqrt((p / sp) * (q / sq)).sum())


# ---------------------------------------------------------------------------
# Trend + mixing analysis
# ---------------------------------------------------------------------------

def _load_seed_trajectories(prefix, info):
    """Per-lane post-burn-in trajectories for the hierarchical bootstrap."""
    traj = Path(f"{prefix}_diagnostic_trajectories.npz")
    if not traj.exists():
        return None
    t = np.load(traj)
    Ts = np.asarray(t["Ts"], float)
    K = np.array([_K_of(info, float(x)) for x in Ts])
    out = {
        "K": K,
        "contacts": np.asarray(t["contacts_post"], float),
        "rg2": np.asarray(t["rg2_post"], float),
        "m_global": np.asarray(t["m_global_scaled_post"], float),
        "smax": np.asarray(t["smax_post"], float),
    }
    if "largest_component_fraction_post" in t:
        out["lcf"] = np.asarray(t["largest_component_fraction_post"], float)
    return out


def _seed_alignment_record(prefix, feat_path, info, N):
    """Build a cross-seed alignment record from a completed seed's artifacts."""
    rows = _read_results(prefix)
    temps = [float(r["T"]) for r in rows]        # per-lane (run) order
    K = [_K_of(info, t) for t in temps]
    traj = Path(f"{prefix}_diagnostic_trajectories.npz")
    obs, shapes = [], {}
    if traj.exists():
        t = np.load(traj)
        for k in ("contacts_post", "rg2_post", "m_global_scaled_post",
                  "smax_post", "largest_component_fraction_post"):
            if k in t:
                obs.append(k)
                shapes[k] = tuple(int(x) for x in np.asarray(t[k]).shape)
    defs_ver = None
    mparams = [float(x) for x in info["params"]]
    man_path = Path(str(feat_path) + ".manifest.json")
    if man_path.exists():
        try:
            man = json.loads(man_path.read_text(encoding="utf-8"))
            defs_ver = man.get("definitions_version")
            mr = man.get("model_record") or {}
            if mr.get("model_params") is not None:
                mparams = [float(x) for x in mr["model_params"]]
        except Exception:
            pass
    return {
        "N": int(N), "temperatures": temps, "K": K,
        "model_name": str(info["model_name"]), "model_params": mparams,
        "Tref": float(info["Tref"]), "Tscale": float(info["Tscale"]),
        "definitions_version": defs_ver,
        "trajectory_obs": obs, "trajectory_shapes": shapes,
    }


def analyze_trends(prefixes, info, block_len=20, n_boot=2000):
    obs = ["C_mean", "Rg2_mean", "Ree2_mean", "m_long_mean",
           "m_global_scaled_mean", "Smax_mean",
           "largest_component_fraction_mean"]
    per_seed = []
    endpoint_dm, endpoint_drg2 = [], []
    for prefix in prefixes:
        rows = _read_results(prefix)
        T = np.array([float(r["T"]) for r in rows])
        K = np.array([_K_of(info, t) for t in T])
        order = np.argsort(K)
        rec = {"temperatures": T[order].tolist(), "K": K[order].tolist()}
        for o in obs:
            y = np.array([float(r[o]) for r in rows])[order]
            rec[o] = {
                "values": y.tolist(),
                "spearman_vs_K": spearman(K[order], y),
                "slope_vs_K": (float(np.polyfit(K[order], y, 1)[0])
                               if y.size >= 2 else float("nan")),
                "endpoint_high_minus_low": float(y[-1] - y[0]),
            }
        per_seed.append(rec)
        endpoint_dm.append(rec["C_mean"]["endpoint_high_minus_low"])
        endpoint_drg2.append(rec["Rg2_mean"]["endpoint_high_minus_low"])

    n_seeds = len(prefixes)
    # -- genuine hierarchical (seed x block) bootstrap (Part 4) --------------
    seed_data = [d for d in (_load_seed_trajectories(p, info) for p in prefixes)
                 if d is not None]
    hb = (hierarchical_bootstrap(seed_data, n_boot=n_boot, block_len=block_len,
                                 n_requested_seeds=len(prefixes))
          if seed_data else None)

    # Point estimates / directional signs from the aggregate endpoint deltas.
    dm_point = float(np.mean(endpoint_dm)) if endpoint_dm else float("nan")
    drg2_point = float(np.mean(endpoint_drg2)) if endpoint_drg2 else float("nan")
    signs_ok = (dm_point > 0) and (drg2_point < 0)

    # -- statistical support: NOT ASSESSABLE with < 2 independent seeds ------
    if n_seeds < 2 or hb is None:
        stat_supported = False
        stat_status = "not_assessable"
    else:
        dm_ci = hb["delta_contacts"]["ci95"]
        drg2_ci = hb["delta_rg2"]["ci95"]
        support = (ci_excludes_zero(*dm_ci) and dm_ci[0] > 0
                   and ci_excludes_zero(*drg2_ci) and drg2_ci[1] < 0)
        stat_supported = bool(support)
        stat_status = "supported" if support else "not_supported"

    dm_arr = np.asarray(endpoint_dm, float)
    spread = (float(np.std(dm_arr) / abs(np.mean(dm_arr)))
              if dm_arr.size > 1 and abs(np.mean(dm_arr)) > 0 else 0.0)
    return {
        "per_seed": per_seed,
        "endpoint_delta_m_point": dm_point,
        "endpoint_delta_Rg2_point": drg2_point,
        "hierarchical_bootstrap": hb,
        "seed_relative_spread_delta_m": spread,
        "sampled_direction_is_LCST_compatible": bool(signs_ok),
        "structural_change_is_statistically_supported": bool(stat_supported),
        "statistical_support_status": stat_status,
        "n_seeds": n_seeds,
        "note": ("expected (LCST): Δ<m> > 0 and Δ<Rg2> < 0 from low-K to high-K; "
                 "statistical support requires the HIERARCHICAL (seed x block) "
                 "bootstrap CI95 to exclude 0 with the correct sign; with < 2 "
                 "independent seeds support is 'not_assessable'."),
    }


def _pad_pc(Pc, width):
    """Pad a per-lane P(m) block (0-based integer contact grid) to ``width``."""
    Pc = np.asarray(Pc, float)
    if Pc.shape[1] >= width:
        return Pc[:, :width]
    out = np.zeros((Pc.shape[0], width), dtype=float)
    out[:, :Pc.shape[1]] = Pc
    return out


def analyze_mixing(prefixes, info, support_maximum=None, support_source=None):
    """Mixing diagnostics with SEED-AGGREGATED overlap and tail support (Part 6).

    Adjacent contact-histogram overlap is reported per seed and aggregated
    (median, minimum, and pooled-COUNT overlap using actual sample counts);
    ladder refinement and the overlap gate use the pooled/worst-seed values, so
    they are invariant to seed ordering.  The high-contact tail pools raw contact
    counts across all seeds and lanes with actual counts.
    """
    key_obs = ("contacts", "rg2", "m_global_scaled", "smax",
               "largest_component_fraction")
    slowest_tau_cycles = 0.0
    min_ess = float("inf"); min_rt = float("inf"); min_cov = float("inf")
    min_swap = float("inf"); max_swap = 0.0; max_drift = 0.0
    min_state_changing = float("inf")
    tau_by_lane = None
    # Phase 11: per-lane slowest-tau observable (the limiting observable), tracked
    # across seeds -- the observable achieving each lane's global maximum tau.
    limiting_obs_by_lane = {}
    limiting_tau_by_lane = {}
    # Phase 11: worst finite drift across seeds, lanes, AND key observables.
    drift_limit = {"value": 0.0, "seed": None, "lane": None, "observable": None}
    total_round_trips = 0
    median_round_trips = []

    # First pass: find the common (widest) integer contact grid across seeds.
    dists = []
    c_width = 0
    for prefix in prefixes:
        npz = np.load(f"{prefix}_distributions.npz")
        Pc_raw = np.asarray(npz["Pc"], float)
        dists.append(Pc_raw)
        c_width = max(c_width, Pc_raw.shape[1])
    c_vals_common = np.arange(c_width, dtype=float)

    seed_reports = []
    per_seed_Pc = []; per_seed_counts = []; per_seed_ess = []; per_seed_tau = []
    for si, prefix in enumerate(prefixes):
        diag = _read_diag(prefix)
        lanes = diag["lane_convergence"]
        Pc = _pad_pc(dists[si], c_width)
        nT = Pc.shape[0]
        lane_tau = np.full(nT, np.nan)
        ess_lane = np.zeros(nT); n_lane = np.zeros(nT)
        per_lane = []
        for i, lane in enumerate(lanes):
            row = {"temperature": lane["temperature"]}
            for o in key_obs:
                if o in lane:
                    tc = lane[o].get("tau_int_cycles")
                    e = lane[o].get("ess")
                    row[f"tau_int_cycles_{o}"] = tc
                    row[f"ess_{o}"] = e
                    if tc and np.isfinite(tc):
                        slowest_tau_cycles = max(slowest_tau_cycles, float(tc))
                        lane_tau[i] = np.nanmax([lane_tau[i], float(tc)])
                        if (i not in limiting_tau_by_lane
                                or float(tc) > limiting_tau_by_lane[i]):
                            limiting_tau_by_lane[i] = float(tc)
                            limiting_obs_by_lane[i] = o
                    if e and np.isfinite(e):
                        min_ess = min(min_ess, float(e))
            c = lane.get("contacts", {})
            if c.get("ess"):
                ess_lane[i] = float(c["ess"])
            if c.get("n_samples"):
                n_lane[i] = float(c["n_samples"])
            # Worst finite drift across every key observable (not only contacts).
            for o in key_obs:
                lo = lane.get(o, {})
                d = lo.get("drift_in_std")
                if d is not None and np.isfinite(d):
                    ad = abs(float(d))
                    if ad > max_drift:
                        max_drift = ad
                    if ad > drift_limit["value"]:
                        drift_limit = {"value": ad, "seed": Path(prefix).name,
                                       "lane": i, "observable": o}
            per_lane.append(row)
        tau_by_lane = lane_tau if tau_by_lane is None else np.fmax(tau_by_lane, lane_tau)
        adj = [float(_bhattacharyya(Pc[a], Pc[a + 1])) for a in range(nT - 1)]
        srates = diag["swap_rates"]
        if srates:
            min_swap = min(min_swap, min(srates)); max_swap = max(max_swap, max(srates))
        min_cov = min(min_cov, diag["summary"].get("min_temp_coverage", 0) or 0)
        # Phase 11: gate on the MINIMUM round trips PER WALKER, not the total.
        rt_pw = diag["summary"].get("min_round_trips_per_walker")
        if rt_pw is None:
            rt_pw = diag["summary"].get("total_round_trips_low", 0) or 0
        min_rt = min(min_rt, float(rt_pw))
        total_round_trips += int(diag["summary"].get("total_round_trips_low", 0) or 0)
        med = diag["summary"].get("median_round_trips_per_walker")
        if med is not None:
            median_round_trips.append(float(med))
        summ = json.load(open(f"{prefix}_run_summary.json"))
        scr = [float(x) for x in summ.get("state_changing_acceptance_rates", [])]
        if scr:
            min_state_changing = min(min_state_changing, min(scr))
        per_seed_Pc.append(Pc); per_seed_counts.append(n_lane)
        per_seed_ess.append(ess_lane); per_seed_tau.append(lane_tau)
        seed_reports.append({
            "seed_prefix": Path(prefix).name,
            "lanes": per_lane,
            "swap_rates": srates,
            "min_temp_coverage": diag["summary"].get("min_temp_coverage"),
            "total_round_trips_low": diag["summary"].get("total_round_trips_low"),
            "min_round_trips_per_walker": diag["summary"].get(
                "min_round_trips_per_walker"),
            "median_round_trips_per_walker": diag["summary"].get(
                "median_round_trips_per_walker"),
            "adjacent_overlap_bhattacharyya": adj,
            "state_changing_acceptance_rates": scr,
        })

    # -- seed-aggregated adjacent overlap (median/min/pooled-count) ----------
    S = len(prefixes)
    counts = np.array(per_seed_counts)            # (S, nT)
    Pc_stack = np.nan_to_num(np.array(per_seed_Pc))   # (S, nT, width)
    pooled_count = (Pc_stack * counts[:, :, None]).sum(axis=0)   # (nT, width)
    nT = pooled_count.shape[0]
    adjacent_aggregate = []
    for a in range(nT - 1):
        ov = [float(_bhattacharyya(per_seed_Pc[s][a], per_seed_Pc[s][a + 1]))
              for s in range(S)]
        adjacent_aggregate.append({
            "gap_index": a,
            "overlap_per_seed": ov,
            "median_overlap": float(np.median(ov)) if ov else float("nan"),
            "minimum_overlap": float(np.min(ov)) if ov else float("nan"),
            "pooled_count_overlap": float(_bhattacharyya(
                pooled_count[a], pooled_count[a + 1])),
        })
    pooled_adjacent = [d["pooled_count_overlap"] for d in adjacent_aggregate]
    worst_seed_adjacent = [d["minimum_overlap"] for d in adjacent_aggregate]
    min_adj_overlap = (min(pooled_adjacent) if pooled_adjacent else float("inf"))

    # -- ONE common tail threshold across every seed (Phase 9) --------------
    tail_pooled = common_tail_statistics(
        per_seed_Pc, per_seed_counts, per_seed_ess, c_vals_common,
        support_maximum=support_maximum, support_source=support_source)
    min_worst_seed_overlap = (min(worst_seed_adjacent)
                              if worst_seed_adjacent else float("inf"))

    metrics = {
        "min_ess": (None if min_ess == float("inf") else min_ess),
        "min_round_trips": (None if min_rt == float("inf") else min_rt),
        "min_temp_coverage": (None if min_cov == float("inf") else min_cov),
        "min_adjacent_overlap": (None if min_adj_overlap == float("inf") else min_adj_overlap),
        "min_worst_seed_adjacent_overlap": (
            None if min_worst_seed_overlap == float("inf")
            else min_worst_seed_overlap),
        "min_swap_rate": (None if min_swap == float("inf") else min_swap),
        "max_swap_rate": max_swap,
        "max_drift_in_std": max_drift,
        "min_state_changing_acceptance": (
            None if min_state_changing == float("inf") else min_state_changing),
        "tail_probability": float(tail_pooled["tail_probability_pooled"]),
        "raw_tail_count_pooled": float(tail_pooled["raw_tail_count_pooled"]),
        "effective_tail_count_pooled": float(
            tail_pooled["effective_tail_count_pooled"]),
    }
    return {"seeds": seed_reports,
            "slowest_tau_int_cycles": slowest_tau_cycles,
            "tau_by_lane": [None if not np.isfinite(x) else float(x)
                            for x in (tau_by_lane if tau_by_lane is not None else [])],
            "limiting_observable_by_lane": [
                limiting_obs_by_lane.get(i)
                for i in range(len(tau_by_lane) if tau_by_lane is not None else 0)],
            "round_trips": {
                "total_round_trips_low": int(total_round_trips),
                "min_round_trips_per_walker": (
                    None if min_rt == float("inf") else float(min_rt)),
                "median_round_trips_per_walker": (
                    float(np.median(median_round_trips))
                    if median_round_trips else None)},
            "drift_limiting": drift_limit,
            "adjacent_overlap_aggregate": adjacent_aggregate,
            "pooled_adjacent_overlap": pooled_adjacent,
            "worst_seed_adjacent_overlap": worst_seed_adjacent,
            "high_contact_tail_pooled": tail_pooled,
            "tail_gate": evaluate_tail_gate(tail_pooled),
            "metrics": metrics}


# ---------------------------------------------------------------------------
# Driver (Phase 11 immutability + gate exit)
# ---------------------------------------------------------------------------

def _regime_lane_map(labels):
    m = {r: [] for r in REGIME_NAMES}
    for i, lab in enumerate(labels):
        m.setdefault(lab, []).append(i)
    return {r: m.get(r, []) for r in REGIME_NAMES}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fit-summary-json", required=True, dest="fit_summary_json")
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--n-temperatures", type=int, default=8)
    ap.add_argument("--n-cycles", type=int, default=400)
    ap.add_argument("--steps-per-swap", type=int, default=60)
    ap.add_argument("--structural-stride", type=int, default=5)
    ap.add_argument("--snapshot-stride", type=int, default=5)
    ap.add_argument("--t-min", type=float, default=None)
    ap.add_argument("--t-max", type=float, default=None)
    ap.add_argument("--branch", choices=["auto", "low", "high"], default="auto")
    ap.add_argument("--physical-branch", type=int, default=None)
    ap.add_argument("--min-dT", type=float, default=0.25)
    ap.add_argument("--min-dK", type=float, default=1e-4)
    ap.add_argument("--allow-n-mismatch", action="store_true")
    ap.add_argument("--allow-decreasing-k", action="store_true",
                    help="DIAGNOSTIC override: permit a scientific run on a branch "
                         "where K decreases with T (recorded in the manifest)")
    ap.add_argument("--allow-failed-calibration", action="store_true")
    # Phase 12: production planning is decoupled from the calibration run; the
    # future production seed count is NOT inferred from the calibration seeds.
    ap.add_argument("--production-seeds", type=int, default=8,
                    help="planned number of INDEPENDENT production seeds "
                         "(default 8; not inferred from calibration seeds)")
    ap.add_argument("--production-burnin-frac", type=float, default=0.5)
    ap.add_argument("--production-snapshot-stride", type=int, default=None,
                    help="production snapshot stride (defaults to --snapshot-stride)")
    ap.add_argument("--target-effective-per-regime", type=int,
                    default=TARGET_EFF_PER_REGIME)
    ap.add_argument("--smoke-test", action="store_true",
                    help="tiny software check; never scientifically validated")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    # Phase 11.3: JSON definitions and compatibility constants must agree.
    sch.check_definitions_consistency()

    seeds = args.seeds
    if args.smoke_test:
        seeds = args.seeds[:1] or [1]

    # -- immutable run directory (Phase 11) ---------------------------------
    run_id = args.run_id or _dt.datetime.now().strftime("run_%Y%m%d_%H%M%S")
    base = Path(args.output_dir) if args.output_dir else (HERE / f"calibration_N{args.N}")
    out_dir = base / run_id
    if out_dir.exists() and not (args.overwrite or args.resume):
        raise SystemExit(
            f"run directory {out_dir} exists; pass --overwrite or --resume, or "
            f"choose a new --run-id (outputs are immutable by default).")
    out_dir.mkdir(parents=True, exist_ok=True)

    info, (T_lo, T_hi), nmeta = load_fitted_model(
        args.fit_summary_json, args.t_min, args.t_max, args.N,
        allow_n_mismatch=args.allow_n_mismatch)

    # preserve inputs by copying + hashing
    fit_copy = out_dir / "input_fit_summary.json"
    if not fit_copy.exists():
        shutil.copy(args.fit_summary_json, fit_copy)
    defs_copy = out_dir / "project_definitions.json"
    if not defs_copy.exists():
        shutil.copy(str(sch.RESOLVED_DEFINITIONS_PATH), defs_copy)

    kinfo = evaluate_K_grid(info, T_lo, T_hi)
    branch = select_branch(kinfo, args.branch, args.physical_branch)
    # Part 9: T-ordered inversion; recompute K by calling the ACTUAL fitted model.
    ladder_info = build_k_ladder(
        branch["Ts"], branch["K"], args.n_temperatures,
        min_dT=args.min_dT, min_dK=args.min_dK,
        k_of_T=lambda t: _K_of(info, t))
    ladder = ladder_info["temperatures"]
    K_ladder = [float(_K_of(info, t)) for t in ladder]

    # Part 9.2: a scientific PNIPAM run refuses a decreasing-K branch (unless an
    # explicit diagnostic override is recorded) and a ladder that cannot meet the
    # spacing/count constraints.
    branch_decreasing = (branch["K_direction"] == "decreasing")
    decreasing_override_used = bool(branch_decreasing and args.allow_decreasing_k)
    if not args.smoke_test:
        if branch_decreasing and not args.allow_decreasing_k:
            raise SystemExit(
                "selected branch has K DECREASING with T (not PNIPAM-LCST "
                "compatible); pass --allow-decreasing-k to override (recorded), "
                "or choose the other --branch.")
        if not ladder_info["spacing_ok"]:
            raise SystemExit(
                "K ladder cannot satisfy the spacing/count constraints: "
                f"{ladder_info['recommendation']} (reduce --n-temperatures or "
                f"relax --min-dT/--min-dK).")

    # -- per-seed stage fingerprints (Phase 4) ------------------------------
    BURNIN_FRAC = 0.5
    N_WORKERS = 1
    # Resolve ONE definitions context for the whole CLI run (Phase 8).
    defs_context = sch.active_definitions_context()
    stage_fingerprint_fields = {}
    stage_fingerprints = {}
    for s in seeds:
        ff = build_stage_fingerprint_fields(
            N=args.N, seed=s, ladder=ladder, K_ladder=K_ladder, info=info,
            fit_summary_path=args.fit_summary_json, n_cycles=args.n_cycles,
            steps_per_swap=args.steps_per_swap, n_workers=N_WORKERS,
            structural_stride=args.structural_stride,
            snapshot_stride=args.snapshot_stride, burnin_frac=BURNIN_FRAC,
            context=defs_context)
        stage_fingerprint_fields[s] = ff
        stage_fingerprints[s] = _stage_fingerprint(ff)

    commands = []
    manifest = {
        "calibration_report_schema_version": CALIBRATION_REPORT_SCHEMA_VERSION,
        "run_id": run_id, "status": "running",
        "created": _dt.datetime.now().isoformat(),
        "model": info["model_name"], "params": info["params"], "N": args.N,
        "requested_seeds": args.seeds, "effective_seeds": seeds,
        "smoke_test": bool(args.smoke_test),
        "fitted_temperature_interval": [T_lo, T_hi],
        "chain_length_check": nmeta,
        "branch_selection": {k: v for k, v in branch.items() if k not in ("Ts", "K")},
        "branch_decreasing_override_used": decreasing_override_used,
        "ladder": {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                   for k, v in ladder_info.items()},
        "K_ladder": K_ladder,
        "stage_fingerprints": {str(s): stage_fingerprints[s] for s in seeds},
        "stage_fingerprint_fields": {str(s): stage_fingerprint_fields[s]
                                     for s in seeds},
        "validator_revision": int(VALIDATOR_REVISION),
        "input_fit_summary_sha256": _sha256(args.fit_summary_json),
        "project_definitions_sha256": _sha256(str(sch.RESOLVED_DEFINITIONS_PATH)),
        "project_definitions_path": str(sch.RESOLVED_DEFINITIONS_PATH),
        "commands": commands,
    }
    (out_dir / "run_manifest.json").write_text(
        json.dumps(sch.json_safe(manifest), indent=2), encoding="utf-8")

    print(f"Run dir: {out_dir}")
    print(f"Model {info['model_name']} on [{T_lo:.2f}, {T_hi:.2f}]: "
          f"K_increases_with_T={kinfo['K_increases_with_T']} "
          f"branch={branch['branch_index']}/{branch['n_branches']} "
          f"dir={branch['K_direction']} zero_crossing={kinfo['zero_crossing_T']}")
    print(f"Ladder ({ladder.size}): T={[round(float(t),1) for t in ladder]}")
    print(f"K ladder: {[round(k,4) for k in K_ladder]}")

    status = "complete"
    try:
        prefixes = [run_seed(out_dir, args.N, s, ladder, args.fit_summary_json,
                             args.n_cycles, args.steps_per_swap,
                             args.structural_stride, args.snapshot_stride,
                             resume=args.resume, commands=commands,
                             stage_fingerprint=stage_fingerprints[s],
                             stage_fingerprint_fields=stage_fingerprint_fields[s],
                             burnin_frac=BURNIN_FRAC, n_workers=N_WORKERS)
                    for s in seeds]
    except Exception:
        manifest["status"] = "failed"
        (out_dir / "run_manifest.json").write_text(
            json.dumps(sch.json_safe(manifest), indent=2), encoding="utf-8")
        raise

    # -- seed-artifact audit (Phase 5) + cross-seed alignment (Phase 6) ------
    companions_by_seed = {}
    align_records = []
    seed_expectations = {}
    for s, prefix in zip(seeds, prefixes):
        feat = Path(f"{prefix}_features.h5")
        cfg = f"{prefix}_configurations.h5"
        companions_by_seed[s] = _companion_paths(prefix, cfg, feat)
        seed_expectations[s] = {
            "N": int(args.N), "seed": int(s),
            "ladder": [float(t) for t in ladder],
            "source_config": str(cfg),
            "stage_fingerprint": stage_fingerprints[s], "run_id": None}
        align_records.append(_seed_alignment_record(prefix, feat, info, args.N))
    seed_audit = audit_seed_artifacts(companions_by_seed,
                                      expectations=seed_expectations)
    alignment_ok, alignment_reason = True, None
    seed_alignment = {"aligned": True, "n_seeds": len(align_records)}
    if len(align_records) >= 2:
        try:
            seed_alignment = require_seed_alignment(align_records)
        except SeedAlignmentError as exc:
            alignment_ok, alignment_reason = False, str(exc)
            seed_alignment = {"aligned": False, "reason": alignment_reason}
    seeds_assessable = bool(seed_audit["assessable"] and alignment_ok)
    # Misaligned seeds must NEVER be combined; this is a data-integrity failure,
    # not a soft gate, so it aborts a scientific run outright.
    if not args.smoke_test and not alignment_ok:
        manifest["status"] = "failed"
        (out_dir / "run_manifest.json").write_text(
            json.dumps(sch.json_safe(manifest), indent=2), encoding="utf-8")
        raise SystemExit(f"cross-seed alignment failed: {alignment_reason}")

    trends = analyze_trends(prefixes, info)
    mixing = analyze_mixing(prefixes, info)

    # -- regime classification from SEED-AGGREGATED lane observables (Part 5) --
    agg = aggregate_lane_observables(prefixes, info, args.N)
    a = agg["aggregate"]
    regimes = classify_regimes(
        agg["K"], a["m"], a["rg2"], a["global_frac"], a["smax_over_N"],
        agg["within_lane_std"], DEFAULT_GATE_THRESHOLDS["min_regime_effect_size"])
    # per-seed labels + stability (unstable lanes disagree with the aggregate)
    per_seed_labels = []
    for s in range(agg["n_seeds"]):
        ps = agg["per_seed"]
        lab = classify_regimes(
            agg["K"], ps["m"][s], ps["rg2"][s], ps["global_frac"][s],
            ps["smax"][s] / max(1, args.N), agg["within_lane_std"],
            DEFAULT_GATE_THRESHOLDS["min_regime_effect_size"])["labels"]
        per_seed_labels.append(lab)
    # Phase 8: a lane is UNSTABLE when its per-seed agreement with the aggregate
    # label is below the configurable threshold; stable labels are required
    # before regimes may be called resolved or production called definitive.
    min_agreement = DEFAULT_GATE_THRESHOLDS["min_regime_label_agreement"]
    agree = []
    unstable_lanes = []
    for k in range(agg["n_temperatures"]):
        votes = [per_seed_labels[s][k] for s in range(agg["n_seeds"])]
        frac = (sum(1 for v in votes if v == regimes["labels"][k]) / len(votes)
                if votes else 0.0)
        agree.append(frac)
        if frac < min_agreement:
            unstable_lanes.append(k)
    regime_labels_stable = bool(len(unstable_lanes) == 0)
    regimes["seed_stability"] = {
        "per_seed_labels": per_seed_labels,
        "fraction_agreeing_with_aggregate": agree,
        "unstable_lanes": unstable_lanes,
        "min_regime_label_agreement": float(min_agreement),
        "regime_labels_stable": regime_labels_stable,
        "between_seed_std": {k: [float(x) for x in v]
                             for k, v in agg["between_seed_std"].items()},
    }
    # Resolved regimes additionally require STABLE labels across seeds.
    regimes["distinct_regimes_resolved"] = bool(
        regimes["distinct_regimes_resolved"] and regime_labels_stable)
    regime_map = _regime_lane_map(regimes["labels"])

    prod_stride = (args.production_snapshot_stride
                   if args.production_snapshot_stride is not None
                   else args.snapshot_stride)
    production = production_requirement(
        regime_map, mixing["tau_by_lane"],
        post_burnin_frac=args.production_burnin_frac,
        snapshot_stride=prod_stride, n_seeds=int(args.production_seeds),
        target_ess=int(args.target_effective_per_regime),
        limiting_observable_by_lane=mixing.get("limiting_observable_by_lane"),
        regime_labels_stable=regime_labels_stable)

    metrics = dict(mixing["metrics"])
    metrics["seed_relative_spread"] = trends["seed_relative_spread_delta_m"]
    metrics = {k: (0.0 if v is None else v) for k, v in metrics.items()}
    # Phase 9: the strengthened tail gate (pooled raw + effective + seed coverage
    # + per-seed support + accounting identity) is authoritative.
    metrics["tail_gate"] = mixing["tail_gate"]
    gates = evaluate_sampling_gates(metrics)

    model_K_ok = bool(branch["K_increases_with_T"])
    sampled_ok = trends["sampled_direction_is_LCST_compatible"]
    stat_ok = trends["structural_change_is_statistically_supported"]
    stat_status = trends["statistical_support_status"]
    # Phases 5-7: inference is assessable ONLY when every requested seed is
    # complete/valid AND aligned.  An incomplete/misaligned set is not
    # assessable regardless of any degenerate bootstrap interval.
    if not seeds_assessable:
        stat_ok = False
        stat_status = "not_assessable"
    distinct_ok = regimes["distinct_regimes_resolved"]
    sampling_ok = gates["_all_passed"]
    enough_seeds = len(seeds) >= 2
    run_mode = "smoke" if args.smoke_test else "scientific"

    calibration_gate_passed = bool(
        model_K_ok and sampled_ok and stat_ok and distinct_ok
        and sampling_ok and enough_seeds and seeds_assessable
        and not args.smoke_test)

    # Part 3: scientifically_validated is the gate result, NOT merely "not smoke".
    report = {
        "calibration_report_schema_version": CALIBRATION_REPORT_SCHEMA_VERSION,
        "run_id": run_id,
        "run_mode": run_mode,
        "scientific_run_attempted": bool(not args.smoke_test),
        "allow_failed_calibration_used": bool(args.allow_failed_calibration),
        "scientifically_validated": bool(calibration_gate_passed),
        "smoke_test": bool(args.smoke_test),
        "model": info["model_name"], "params": info["params"],
        "N": args.N, "seeds": seeds,
        "fitted_temperature_interval": [T_lo, T_hi],
        "chain_length_check": nmeta,
        "branch_selection": {k: v for k, v in branch.items() if k not in ("Ts", "K")},
        "branch_decreasing_override_used": decreasing_override_used,
        "K_analysis": {
            "K_increases_with_T": kinfo["K_increases_with_T"],
            "monotonic": kinfo["monotonic"],
            "zero_crossing_T": kinfo["zero_crossing_T"],
        },
        "ladder": {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                   for k, v in ladder_info.items()},
        "K_ladder": K_ladder,
        "stage_fingerprints": {str(s): stage_fingerprints[s] for s in seeds},
        "stage_fingerprint_fields": {str(s): stage_fingerprint_fields[s]
                                     for s in seeds},
        "validator_revision": int(VALIDATOR_REVISION),
        "seed_audit": seed_audit,
        "seed_alignment": seed_alignment,
        "trends": trends,
        "regimes": regimes,
        "mixing": mixing,
        "sampling_gates": gates,
        "recommended_production": production,
        "scientific_conclusions": {
            "model_K_direction_is_LCST_compatible": model_K_ok,
            "sampled_direction_is_LCST_compatible": sampled_ok,
            "structural_change_is_statistically_supported": stat_ok,
            "statistical_support_status": stat_status,
            "distinct_regimes_are_resolved": distinct_ok,
            "sampling_is_adequate": sampling_ok,
            "enough_seeds_for_calibration": enough_seeds,
            "all_requested_seeds_complete": seed_audit["all_requested_complete"],
            "seeds_are_aligned": alignment_ok,
            "seeds_assessable": seeds_assessable,
            "calibration_gate_passed": calibration_gate_passed,
        },
    }
    # Part 6: ladder refinement from the POOLED (seed-order-invariant) overlap.
    report["ladder_refinement"] = refine_ladder(
        ladder, mixing["adjacent_overlap_aggregate"])

    manifest["status"] = status
    manifest["commands"] = commands
    (out_dir / "run_manifest.json").write_text(
        json.dumps(sch.json_safe(manifest), indent=2), encoding="utf-8")
    (out_dir / "calibration_report.json").write_text(
        json.dumps(sch.json_safe(report), indent=2), encoding="utf-8")
    (out_dir / "recommended_production_config.json").write_text(
        json.dumps(sch.json_safe(production), indent=2), encoding="utf-8")
    _write_report_md(out_dir / "calibration_report.md", report)

    print(f"\nrun_mode: {run_mode}  "
          f"scientifically_validated: {report['scientifically_validated']}")
    print(f"calibration_gate_passed: {calibration_gate_passed}  "
          f"statistical_support: {stat_status}")
    for name, gr in gates.items():
        if name.startswith("_"):
            continue
        if "minimum_value" in gr:      # band gate
            print(f"  gate {name}: passed={gr['passed']} "
                  f"min={gr['minimum_value']}/{gr['minimum_threshold']} "
                  f"max={gr['maximum_value']}/{gr['maximum_threshold']}")
        else:
            print(f"  gate {name}: passed={gr['passed']} "
                  f"value={gr['value']} thr={gr['threshold']}")
    print(f"Reports in {out_dir}")

    if args.smoke_test:
        print("SMOKE TEST: software check only; NOT scientifically validated.")
        return
    if not calibration_gate_passed and not args.allow_failed_calibration:
        raise SystemExit(
            "SCIENTIFIC CALIBRATION GATE FAILED (see calibration_report.json). "
            "Pass --allow-failed-calibration to exit 0 anyway.")


def _write_report_md(path, report):
    c = report["scientific_conclusions"]
    prod = report["recommended_production"]
    lines = [
        f"# PNIPAM calibration report — N={report['N']} ({report['model']})",
        "",
        f"- run_id: {report['run_id']}",
        f"- scientifically validated: **{report['scientifically_validated']}**",
        f"- fitted interval: {report['fitted_temperature_interval']}",
        f"- branch: {report['branch_selection']}",
        "",
        "## Scientific conclusions (separate claims)",
    ] + [f"- {k}: **{v}**" for k, v in c.items()] + [
        "",
        f"- statistical support status: **{report['trends']['statistical_support_status']}**",
        "",
        "## Endpoint effect sizes (high-K minus low-K)",
        f"- Δ<m> (point) = {report['trends']['endpoint_delta_m_point']}",
        f"- Δ<Rg^2> (point) = {report['trends']['endpoint_delta_Rg2_point']}",
        f"- hierarchical bootstrap: {_hb_summary(report['trends']['hierarchical_bootstrap'])}",
        "",
        "## Regimes",
        f"- labels: {report['regimes']['labels']}",
        f"- rule: {report['regimes']['rule']}",
        "",
        "## Sampling gates",
    ] + [f"- {k}: passed={v['passed']} " + _gate_md(v)
         for k, v in report["sampling_gates"].items() if not k.startswith("_")] + [
        "",
        "## Recommended production (NOT launched)",
        f"- recommended cycles/seed: {prod['recommended_production_cycles_per_seed']}",
        f"- limiting regime: {prod['limiting_regime']}",
        f"- every regime reaches target: {prod['every_regime_reaches_target']}",
        "",
        "_Effective saved configs use Delta_eff = max(snapshot_stride, "
        "2*tau_int_cycles), not raw snapshots._",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _gate_md(v):
    if "minimum_value" in v:
        return (f"min={v['minimum_value']}/{v['minimum_threshold']} "
                f"max={v['maximum_value']}/{v['maximum_threshold']}")
    return f"value={v['value']} thr={v['threshold']}"


def _hb_summary(hb):
    if not hb:
        return "not available"
    dm = hb["delta_contacts"]; dr = hb["delta_rg2"]
    return (f"Δ<m> CI95={dm['ci95']}, Δ<Rg^2> CI95={dr['ci95']} "
            f"(n_seeds={dm['n_seeds']}, blocks L={dm['block_length']}, "
            f"reps={dm['n_bootstrap_replicates']})")


if __name__ == "__main__":
    main()
