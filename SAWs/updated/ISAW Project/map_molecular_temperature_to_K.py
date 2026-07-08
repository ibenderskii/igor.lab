#!/usr/bin/env python3
"""Empirically map every molecular REMD temperature to its best-matching direct
lattice coupling K.

For every molecular temperature T_j we ask which sampled direct-lattice coupling
K_i best reproduces (1) the molecular contact distribution, (2) the molecular Rg
distribution, and (3) a weighted combination of those two *marginal* comparisons.

The direct lattice ensemble is  P(C | K) proportional to exp[K * m(C)].

This script is a DIAGNOSTIC.  It never fits or assumes an analytic K(T) curve; it
produces the unconstrained empirical mapping  T_j -> K_best(T_j)  and compares it
with the existing analytic h/T - s mapping.

Histogram/JS conventions here mirror the reliable repository helpers in
``analyze_support_mismatch.py`` / ``fit_lattice_contact_model_2.py`` (piecewise-
constant density overlap rebinning; symmetric JS with natural logarithms), but are
re-implemented explicitly so the diagnostic is self-contained and directly
testable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Matplotlib is imported with a non-interactive backend so the script runs
# headless.  Only matplotlib is used for plotting (no seaborn).
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

LN2 = float(np.log(2.0))
# Tolerance for treating two nominally identical K values as the same coupling.
K_MATCH_ATOL = 1e-6


# ---------------------------------------------------------------------------
# Histogram helpers (explicit, self-contained)
# ---------------------------------------------------------------------------
def infer_edges_from_centers(centers: np.ndarray) -> np.ndarray:
    """Infer bin edges from evenly-spaced bin centers.

    Uses the mean spacing so a uniform grid maps to the exact half-open edges.
    """
    centers = np.asarray(centers, dtype=float)
    if centers.ndim != 1 or centers.size < 2:
        raise ValueError("need at least two bin centers to infer edges")
    d = float(np.mean(np.diff(centers)))
    if not np.isfinite(d) or d <= 0:
        raise ValueError("bin centers must be increasing with finite spacing")
    edges = np.empty(centers.size + 1, dtype=float)
    edges[:-1] = centers - 0.5 * d
    edges[-1] = centers[-1] + 0.5 * d
    return edges


def density_to_mass(density: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Convert a probability *density* on evenly-spaced centers to probability
    *mass* per bin (mass_j = density_j * bin_width), then safely normalize.
    """
    density = np.asarray(density, dtype=float)
    centers = np.asarray(centers, dtype=float)
    d = float(np.mean(np.diff(centers)))
    mass = density * d
    return normalize_mass(mass)


def normalize_mass(mass: np.ndarray) -> np.ndarray:
    """Normalize probability mass to sum to one; return zeros unchanged if the
    total is non-positive (rather than dividing by zero)."""
    mass = np.asarray(mass, dtype=float)
    s = float(mass.sum())
    if s > 0 and np.isfinite(s):
        return mass / s
    return mass.copy()


def _overlap_mass(
    source_edges: np.ndarray, source_mass: np.ndarray, target_edges: np.ndarray
) -> np.ndarray:
    """Mass-conserving overlap integration WITHOUT renormalization.

    ``source_mass[j]`` is treated as uniform density over
    ``[source_edges[j], source_edges[j+1])`` and that density is integrated over
    every target bin.  Target-bin widths are never used, so infinite outer target
    edges (for underflow/overflow catch-all bins) are allowed.  The total over a
    target grid that fully covers the source equals the total source mass.
    """
    source_edges = np.asarray(source_edges, dtype=float)
    source_mass = np.asarray(source_mass, dtype=float)
    target_edges = np.asarray(target_edges, dtype=float)

    if source_edges.ndim != 1 or target_edges.ndim != 1 or source_mass.ndim != 1:
        raise ValueError("edges and mass arrays must be 1D")
    if source_edges.size != source_mass.size + 1:
        raise ValueError("len(source_edges) must equal len(source_mass) + 1")
    if np.any(np.diff(source_edges) <= 0):
        raise ValueError("source_edges must be strictly increasing")
    if np.any(np.diff(target_edges) <= 0):
        raise ValueError("target_edges must be strictly increasing")

    widths = np.diff(source_edges)
    dens = np.zeros_like(source_mass, dtype=float)
    m = widths > 0
    dens[m] = source_mass[m] / widths[m]

    out = np.zeros(target_edges.size - 1, dtype=float)
    j = 0
    for i in range(out.size):
        a, b = float(target_edges[i]), float(target_edges[i + 1])
        while j < dens.size and source_edges[j + 1] <= a:
            j += 1
        jj = j
        while jj < dens.size and source_edges[jj] < b:
            left = max(a, source_edges[jj])
            right = min(b, source_edges[jj + 1])
            if right > left:
                out[i] += dens[jj] * (right - left)
            jj += 1
    return out


def redistribute_mass_to_grid(
    source_edges: np.ndarray,
    source_mass: np.ndarray,
    inner_edges: np.ndarray,
) -> np.ndarray:
    """Redistribute probability mass onto ``inner_edges`` with catch-all
    underflow and overflow bins so NO source mass is discarded.

    Returns a vector of length ``len(inner_edges) + 1``:
        [underflow, inner_bin_0, ..., inner_bin_{n-1}, overflow]
    where underflow collects mass below ``inner_edges[0]`` and overflow collects
    mass above ``inner_edges[-1]``.  The returned vector sums to the total source
    mass (i.e. mass is conserved, not renormalized here).
    """
    inner_edges = np.asarray(inner_edges, dtype=float)
    full_edges = np.concatenate(([-np.inf], inner_edges, [np.inf]))
    return _overlap_mass(source_edges, source_mass, full_edges)


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence with natural logarithms.

    Symmetric, bounded in [0, ln 2].  Zero for identical distributions, ln 2 for
    disjoint supports.  Zeros are handled with the 0*log0 = 0 convention (no
    epsilon flooring).  Returns NaN if either input has non-positive total mass.
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    sp, sq = float(p.sum()), float(q.sum())
    if not (sp > 0 and sq > 0 and np.isfinite(sp) and np.isfinite(sq)):
        return float("nan")
    p = p / sp
    q = q / sq
    m = 0.5 * (p + q)

    def _kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        # Wherever a>0, m=0.5*(a+..) > 0, so the ratio is well defined.
        return float(np.sum(a[mask] * np.log(a[mask] / b[mask])))

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


# ---------------------------------------------------------------------------
# Input loading and validation
# ---------------------------------------------------------------------------
def sha256_of_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _as_scalar(arr) -> Any:
    a = np.asarray(arr)
    return a.item() if a.ndim == 0 else a


def load_fit_summary(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        js = json.load(fh)
    if "contact_offset" not in js:
        raise ValueError(f"{path}: missing 'contact_offset'")
    if "rg_scale" not in js:
        raise ValueError(f"{path}: missing 'rg_scale'")
    contact_offset = float(js["contact_offset"])
    rg_scale = float(js["rg_scale"])
    if not np.isfinite(contact_offset):
        raise ValueError(f"{path}: contact_offset is not finite")
    if not (np.isfinite(rg_scale) and rg_scale > 0):
        raise ValueError(f"{path}: rg_scale must be finite and positive")
    model = js.get("model")
    params = js.get("params", {}) or {}
    h = params.get("h")
    s = params.get("s")
    return {
        "contact_offset": contact_offset,
        "rg_scale": rg_scale,
        "model": model,
        "h": None if h is None else float(h),
        "s": None if s is None else float(s),
        "raw": js,
    }


def load_molecular(path: str) -> Dict[str, Any]:
    d = np.load(path, allow_pickle=True)
    for key in ("temps", "ct_centers", "ct_hists", "rg_centers", "rg_hists"):
        if key not in d.files:
            raise ValueError(f"{path}: molecular NPZ missing '{key}'")
    temps = np.asarray(d["temps"], dtype=float)
    ct_centers = np.asarray(d["ct_centers"], dtype=float)
    ct_hists = np.asarray(d["ct_hists"], dtype=float)
    rg_centers = np.asarray(d["rg_centers"], dtype=float)
    rg_hists = np.asarray(d["rg_hists"], dtype=float)
    n = temps.size
    if ct_hists.shape != (n, ct_centers.size):
        raise ValueError(f"{path}: ct_hists shape {ct_hists.shape} incompatible")
    if rg_hists.shape != (n, rg_centers.size):
        raise ValueError(f"{path}: rg_hists shape {rg_hists.shape} incompatible")
    for name, arr in (("ct_hists", ct_hists), ("rg_hists", rg_hists),
                      ("temps", temps), ("ct_centers", ct_centers),
                      ("rg_centers", rg_centers)):
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{path}: molecular '{name}' has non-finite values")
    if np.any(ct_hists < 0) or np.any(rg_hists < 0):
        raise ValueError(f"{path}: molecular histograms have negative values")
    order = np.argsort(temps)
    return {
        "temps": temps[order],
        "ct_centers": ct_centers,
        "ct_hists": ct_hists[order],
        "rg_centers": rg_centers,
        "rg_hists": rg_hists[order],
    }


def _pick(d, *keys):
    for k in keys:
        if k in d.files:
            return d[k]
    return None


def load_direct_file(path: str) -> Dict[str, Any]:
    """Load and validate a single direct-K distributions NPZ."""
    d = np.load(path, allow_pickle=True)

    control_mode = _pick(d, "control_mode")
    if control_mode is None or str(_as_scalar(control_mode)) != "direct_K":
        raise ValueError(
            f"{path}: control_mode must be 'direct_K' (got {control_mode!r})"
        )
    tma = _pick(d, "temperature_mapping_applied")
    if tma is None or bool(_as_scalar(tma)) is not False:
        raise ValueError(
            f"{path}: temperature_mapping_applied must be false (got {tma!r})"
        )

    n_raw = _pick(d, "N")
    if n_raw is None:
        raise ValueError(f"{path}: missing chain length 'N'")
    N = int(_as_scalar(n_raw))

    seed_raw = _pick(d, "seed")
    if seed_raw is None:
        raise ValueError(f"{path}: missing 'seed'")
    seed = int(_as_scalar(seed_raw))

    K = _pick(d, "K_values", "coupling_K_by_temperature")
    if K is None:
        raise ValueError(f"{path}: missing 'K_values'")
    K = np.asarray(K, dtype=float)
    if K.ndim != 1 or K.size == 0:
        raise ValueError(f"{path}: K_values must be a non-empty 1D array")
    if not np.all(np.isfinite(K)):
        raise ValueError(f"{path}: K_values contains non-finite values")

    # Consistency of the K arrays within the file.
    ck = _pick(d, "coupling_K_by_temperature")
    if ck is not None:
        ck = np.asarray(ck, dtype=float)
        if ck.shape != K.shape or not np.allclose(ck, K, atol=1e-9, rtol=0.0):
            raise ValueError(
                f"{path}: K_values and coupling_K_by_temperature disagree"
            )

    c_counts = _pick(d, "c_vals", "ct_centers")
    if c_counts is None:
        raise ValueError(f"{path}: missing contact axis ('c_vals'/'ct_centers')")
    c_counts = np.asarray(c_counts, dtype=float)
    c_int = np.rint(c_counts)
    if not np.allclose(c_counts, c_int, atol=1e-8, rtol=0.0):
        raise ValueError(f"{path}: contact axis must be integer-valued")
    c_int = c_int.astype(int)
    if c_int[0] != 0 or np.any(np.diff(c_int) != 1):
        raise ValueError(f"{path}: contact axis must be 0,1,2,... contiguous")

    Pc = _pick(d, "Pc", "ct_hists")
    if Pc is None:
        raise ValueError(f"{path}: missing contact histogram ('Pc'/'ct_hists')")
    Pc = np.asarray(Pc, dtype=float)

    rg_edges = _pick(d, "rg_edges_lattice", "rg_edges")
    if rg_edges is None:
        rg_centers = _pick(d, "rg_centers_lattice", "rg_centers")
        if rg_centers is None:
            raise ValueError(f"{path}: missing lattice Rg edges/centers")
        rg_edges = infer_edges_from_centers(np.asarray(rg_centers, dtype=float))
    rg_edges = np.asarray(rg_edges, dtype=float)

    Prg = _pick(d, "Prg", "rg_hists")
    if Prg is None:
        raise ValueError(f"{path}: missing Rg histogram ('Prg'/'rg_hists')")
    Prg = np.asarray(Prg, dtype=float)

    nK = K.size
    if Pc.shape != (nK, c_int.size):
        raise ValueError(
            f"{path}: Pc shape {Pc.shape} incompatible with "
            f"({nK}, {c_int.size})"
        )
    if Prg.shape[0] != nK:
        raise ValueError(f"{path}: Prg lane count {Prg.shape[0]} != {nK}")
    if rg_edges.ndim != 1 or rg_edges.size != Prg.shape[1] + 1:
        raise ValueError(
            f"{path}: rg_edges length {rg_edges.size} must equal "
            f"n_rg_bins+1 ({Prg.shape[1] + 1})"
        )
    if np.any(np.diff(rg_edges) <= 0):
        raise ValueError(f"{path}: lattice Rg edges not strictly increasing")

    for name, arr in (("Pc", Pc), ("Prg", Prg)):
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{path}: '{name}' has non-finite values")
        if np.any(arr < 0):
            raise ValueError(f"{path}: '{name}' has negative values")

    return {
        "path": path,
        "N": N,
        "seed": seed,
        "K": K,
        "c_int": c_int,
        "Pc": Pc,
        "rg_edges": rg_edges,
        "Prg": Prg,
    }


# ---------------------------------------------------------------------------
# Pooled direct-K reference library
# ---------------------------------------------------------------------------
def cluster_unique_K(all_K: np.ndarray, atol: float = K_MATCH_ATOL) -> np.ndarray:
    """Cluster nominally identical K values (within ``atol``) into unique
    couplings.  Returns the sorted cluster representatives (cluster means)."""
    vals = np.sort(np.asarray(all_K, dtype=float))
    reps: List[float] = []
    group = [vals[0]]
    for v in vals[1:]:
        if v - group[-1] <= atol:
            group.append(v)
        else:
            reps.append(float(np.mean(group)))
            group = [v]
    reps.append(float(np.mean(group)))
    return np.asarray(reps, dtype=float)


def build_reference_library(
    direct_files: List[Dict[str, Any]],
    rg_scale: float,
    mol_rg_edges: np.ndarray,
) -> Dict[str, Any]:
    """Pool all direct-K files into a per-unique-K reference library.

    Contact distributions are placed on a common integer grid 0..C_max (zero
    padded); Rg distributions are scaled to molecular units and rebinned onto the
    molecular Rg grid (with underflow/overflow) so that pooling happens on common
    supports.  Duplicate K values are averaged first within a seed, then across
    seeds with equal seed weight.
    """
    # Common integer contact grid across every supplied file.
    c_max = max(int(f["c_int"][-1]) for f in direct_files)
    c_grid = np.arange(0, c_max + 1, dtype=int)
    n_rg_full = mol_rg_edges.size + 1  # underflow + inner + overflow

    # Gather per-file, per-K contributions keyed by unique K, then seed.
    all_K = np.concatenate([f["K"] for f in direct_files])
    uniqueK = cluster_unique_K(all_K)

    def match_K(k: float) -> int:
        idx = int(np.argmin(np.abs(uniqueK - k)))
        if abs(uniqueK[idx] - k) > K_MATCH_ATOL:
            raise ValueError(f"K={k} did not match any cluster")
        return idx

    # contributions[ik][seed] -> list of (contact_mass_on_cgrid, rg_mass_full)
    contributions: List[Dict[int, List[Tuple[np.ndarray, np.ndarray]]]] = [
        {} for _ in range(uniqueK.size)
    ]

    for f in direct_files:
        seed = f["seed"]
        for lane, k in enumerate(f["K"]):
            ik = match_K(float(k))
            # Contact mass on the common integer grid (zero-padded).
            cmass = np.zeros(c_grid.size, dtype=float)
            src = normalize_mass(f["Pc"][lane])
            cmass[: src.size] = src
            # Rg mass: scale lattice edges to molecular units, then rebin onto
            # the molecular Rg grid with underflow/overflow (mass conserving).
            scaled_edges = f["rg_edges"] * rg_scale
            rgmass = redistribute_mass_to_grid(
                scaled_edges, normalize_mass(f["Prg"][lane]), mol_rg_edges
            )
            rgmass = normalize_mass(rgmass)
            contributions[ik].setdefault(seed, []).append((cmass, rgmass))

    pooled_contact = np.zeros((uniqueK.size, c_grid.size), dtype=float)
    pooled_rg = np.zeros((uniqueK.size, n_rg_full), dtype=float)
    n_seeds = np.zeros(uniqueK.size, dtype=int)
    seed_lists: List[List[int]] = []

    for ik in range(uniqueK.size):
        seed_map = contributions[ik]
        seeds = sorted(seed_map.keys())
        seed_lists.append(seeds)
        n_seeds[ik] = len(seeds)
        seed_c = []
        seed_r = []
        for seed in seeds:
            runs = seed_map[seed]
            cavg = np.mean([r[0] for r in runs], axis=0)
            ravg = np.mean([r[1] for r in runs], axis=0)
            seed_c.append(normalize_mass(cavg))
            seed_r.append(normalize_mass(ravg))
        pooled_contact[ik] = normalize_mass(np.mean(seed_c, axis=0))
        pooled_rg[ik] = normalize_mass(np.mean(seed_r, axis=0))

    return {
        "K": uniqueK,
        "c_grid": c_grid,
        "c_max": c_max,
        "pooled_contact": pooled_contact,  # on integer grid 0..c_max
        "pooled_rg": pooled_rg,            # on [underflow, mol_rg_bins, overflow]
        "n_seeds": n_seeds,
        "seed_lists": seed_lists,
    }


# ---------------------------------------------------------------------------
# Molecular distribution mapping
# ---------------------------------------------------------------------------
def molecular_contact_mass(
    ct_centers: np.ndarray,
    ct_hist_row: np.ndarray,
    contact_offset: float,
    c_max: int,
) -> np.ndarray:
    """Shift the molecular contact density by ``contact_offset`` and integrate it
    into integer lattice-contact bins on the common contact vector

        [underflow, m=0, 1, ..., c_max, overflow].

    All molecular probability mass is preserved: mass below m=0 lands in
    underflow, mass above m=c_max lands in overflow.
    """
    native_edges = infer_edges_from_centers(ct_centers)
    shifted_edges = native_edges - contact_offset
    mass = density_to_mass(ct_hist_row, ct_centers)
    # Integer bin edges for m = 0..c_max are the half-integers -0.5 .. c_max+0.5.
    inner_edges = np.arange(-0.5, c_max + 1.5, 1.0, dtype=float)
    return redistribute_mass_to_grid(shifted_edges, mass, inner_edges)


def molecular_rg_mass(
    rg_centers: np.ndarray, rg_hist_row: np.ndarray, mol_rg_edges: np.ndarray
) -> np.ndarray:
    """Molecular Rg mass on the [underflow, molecular Rg bins, overflow] grid.

    The molecular grid defines the support, so all molecular mass falls in the
    inner bins (underflow and overflow are zero for the molecule)."""
    mass = density_to_mass(rg_hist_row, rg_centers)
    return np.concatenate(([0.0], mass, [0.0]))


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def contact_scores(
    mol_contact_full: np.ndarray, direct_contact_int: np.ndarray
) -> Tuple[float, float]:
    """Return (JS_contact_full, JS_contact_overlap).

    ``mol_contact_full`` is on [underflow, 0..c_max, overflow]; the direct pooled
    contact distribution is on the integer grid 0..c_max (no under/overflow).

    JS_contact_full is computed on the full vector (direct has zero underflow /
    overflow mass), so unsupported molecular mass is preserved and penalized.

    JS_contact_overlap restricts BOTH to the common integer support and
    renormalizes -- a diagnostic that hides support mismatch.
    """
    n_int = direct_contact_int.size
    direct_full = np.zeros(n_int + 2, dtype=float)
    direct_full[1:-1] = direct_contact_int
    js_full = js_divergence(mol_contact_full, direct_full)

    mol_int = mol_contact_full[1:-1]
    js_overlap = js_divergence(mol_int, direct_contact_int)
    return js_full, js_overlap


def quadratic_refine(
    K: np.ndarray, scores: np.ndarray, i_best: int
) -> float:
    """Local quadratic refinement of the combined-score minimum.

    Fits a parabola through the best grid point and its two neighbors.  Returns
    the vertex K only when the minimum is interior, the curvature is positive,
    and the vertex lies strictly between the neighbor K values; otherwise NaN.
    """
    if i_best <= 0 or i_best >= K.size - 1:
        return float("nan")
    x = K[i_best - 1 : i_best + 2].astype(float)
    y = scores[i_best - 1 : i_best + 2].astype(float)
    if not np.all(np.isfinite(y)):
        return float("nan")
    a, b, _c = np.polyfit(x, y, 2)
    if not (np.isfinite(a) and a > 0):
        return float("nan")
    vertex = -b / (2.0 * a)
    if not (x[0] < vertex < x[2]):
        return float("nan")
    return float(vertex)


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation with average ranks for ties."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2:
        return float("nan")

    def _rank(a: np.ndarray) -> np.ndarray:
        order = np.argsort(a, kind="mergesort")
        ranks = np.empty(a.size, dtype=float)
        ranks[order] = np.arange(a.size, dtype=float)
        # Average ranks over ties.
        _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
        sums = np.zeros(counts.size)
        np.add.at(sums, inv, ranks)
        avg = sums / counts
        return avg[inv]

    rx, ry = _rank(x), _rank(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = np.sqrt(np.sum(rx * rx) * np.sum(ry * ry))
    if denom <= 0:
        return float("nan")
    return float(np.sum(rx * ry) / denom)


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------
def _fmt(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, np.generic):  # normalize numpy scalars (float64/int64/bool_)
        x = x.item()
    if isinstance(x, bool):
        return "True" if x else "False"
    if isinstance(x, float):
        return "nan" if not np.isfinite(x) else repr(x)
    return str(x)


def write_csv(path: Path, header: List[str], rows: List[List[Any]]) -> None:
    lines = [",".join(header)]
    for row in rows:
        lines.append(",".join(_fmt(v) for v in row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return v if np.isfinite(v) else None
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return [json_safe(v) for v in obj.tolist()]
    return obj


# ---------------------------------------------------------------------------
# Plots (matplotlib only, default styles/colors, separate figures)
# ---------------------------------------------------------------------------
def plot_mapping_vs_temperature(
    temps, k_comb, k_contact, k_rg, k_old, tlo, thi, out_path
):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.axhspan(tlo, thi, alpha=0.15, label="transition interval")
    ax.plot(temps, k_comb, marker="o", label="K_best combined")
    ax.plot(temps, k_contact, marker="s", label="K_best contact")
    ax.plot(temps, k_rg, marker="^", label="K_best Rg")
    if k_old is not None:
        ax.plot(temps, k_old, marker="", linestyle="--", label="K_old = s - h/T")
    ax.set_xlabel("molecular temperature T")
    ax.set_ylabel("direct lattice coupling K")
    ax.set_title("Empirical K_best(T) vs analytic mapping")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_score_surface(temps, K, surface, tlo, thi, out_path):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # surface shape (nT, nK); show K on y, T on x.
    mesh = ax.pcolormesh(temps, K, surface.T, shading="auto")
    fig.colorbar(mesh, ax=ax, label="combined JS score")
    ax.axhspan(tlo, thi, alpha=0.25, label="transition interval")
    ax.set_xlabel("molecular temperature T")
    ax.set_ylabel("direct lattice coupling K")
    ax.set_title("Combined marginal-JS score surface")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_component_mappings(temps, k_contact, k_rg, tlo, thi, out_path):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.axhspan(tlo, thi, alpha=0.15, label="transition interval")
    ax.plot(temps, k_contact, marker="s", label="contact-only K_best")
    ax.plot(temps, k_rg, marker="^", label="Rg-only K_best")
    ax.set_xlabel("molecular temperature T")
    ax.set_ylabel("direct lattice coupling K")
    ax.set_title("Contact-only vs Rg-only preferred K")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run_mapping(args: argparse.Namespace) -> Dict[str, Any]:
    out_dir = Path(args.output_dir)
    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
        raise SystemExit(
            f"output dir {out_dir} is not empty; pass --overwrite to replace"
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    cw = float(args.contact_weight)
    rw = float(args.rg_weight)
    if cw < 0 or rw < 0 or (cw + rw) <= 0:
        raise SystemExit("contact-weight and rg-weight must be >=0 and not both 0")
    w_c = cw / (cw + rw)
    w_r = rw / (cw + rw)

    fit = load_fit_summary(args.fit_summary_json)
    mol = load_molecular(args.molecular_npz)

    direct_files = [load_direct_file(p) for p in args.direct_k_npz]
    ns = {f["N"] for f in direct_files}
    if len(ns) != 1:
        raise SystemExit(f"direct-K files disagree on chain length N: {sorted(ns)}")
    N = ns.pop()

    mol_rg_edges = infer_edges_from_centers(mol["rg_centers"])
    lib = build_reference_library(direct_files, fit["rg_scale"], mol_rg_edges)
    K = lib["K"]
    nK = K.size
    temps = mol["temps"]
    nT = temps.size

    # Precompute molecular mass vectors per temperature.
    mol_contact = np.array(
        [
            molecular_contact_mass(
                mol["ct_centers"], mol["ct_hists"][j], fit["contact_offset"],
                lib["c_max"],
            )
            for j in range(nT)
        ]
    )
    mol_rg = np.array(
        [
            molecular_rg_mass(mol["rg_centers"], mol["rg_hists"][j], mol_rg_edges)
            for j in range(nT)
        ]
    )

    # Score surfaces (nT, nK).
    js_c_full = np.full((nT, nK), np.nan)
    js_c_overlap = np.full((nT, nK), np.nan)
    js_rg = np.full((nT, nK), np.nan)
    for j in range(nT):
        for i in range(nK):
            full, overlap = contact_scores(
                mol_contact[j], lib["pooled_contact"][i]
            )
            js_c_full[j, i] = full
            js_c_overlap[j, i] = overlap
            js_rg[j, i] = js_divergence(mol_rg[j], lib["pooled_rg"][i])
    combined = w_c * js_c_full + w_r * js_rg

    # Old analytic mapping K_old(T) = s - h/T (direct exp[K m] ~ P0 exp[-b m]).
    has_old = fit["model"] == "hs" and fit["h"] is not None and fit["s"] is not None
    if has_old:
        k_old = fit["s"] - fit["h"] / temps
    else:
        k_old = np.full(nT, np.nan)

    # Per-temperature results.
    rows_map: List[List[Any]] = []
    mapping_records: List[Dict[str, Any]] = []
    k_best_comb = np.full(nT, np.nan)
    k_best_contact = np.full(nT, np.nan)
    k_best_rg = np.full(nT, np.nan)
    k_best_quad = np.full(nT, np.nan)

    # Component-specific boundary flags.  A flag is True when that optimization
    # target's grid optimum sits at a sampled-K endpoint and is therefore
    # censored (the true optimum may lie at or beyond that endpoint).
    c_bhl = np.zeros(nT, dtype=bool)
    c_bhh = np.zeros(nT, dtype=bool)
    r_bhl = np.zeros(nT, dtype=bool)
    r_bhh = np.zeros(nT, dtype=bool)
    m_bhl = np.zeros(nT, dtype=bool)
    m_bhh = np.zeros(nT, dtype=bool)

    thr = float(args.contact_rg_disagreement_threshold)

    for j in range(nT):
        i_c = int(np.argmin(js_c_full[j]))
        i_r = int(np.argmin(js_rg[j]))
        i_comb = int(np.argmin(combined[j]))
        k_best_contact[j] = K[i_c]
        k_best_rg[j] = K[i_r]
        k_best_comb[j] = K[i_comb]

        quad = quadratic_refine(K, combined[j], i_comb)
        k_best_quad[j] = quad

        # Second-best (next-smallest combined value over the remaining grid).
        order = np.argsort(combined[j])
        best_score = float(combined[j][order[0]])
        second_score = float(combined[j][order[1]]) if nK > 1 else float("nan")
        gap = second_score - best_score if nK > 1 else float("nan")

        # Boundary flags for contact-only, Rg-only, and combined optima.
        c_bhl[j] = i_c == 0
        c_bhh[j] = i_c == nK - 1
        r_bhl[j] = i_r == 0
        r_bhh[j] = i_r == nK - 1
        m_bhl[j] = i_comb == 0
        m_bhh[j] = i_comb == nK - 1
        contact_resolved = not (c_bhl[j] or c_bhh[j])
        rg_resolved = not (r_bhl[j] or r_bhh[j])
        pair_resolved = contact_resolved and rg_resolved
        combined_resolved_j = not (m_bhl[j] or m_bhh[j])

        # Distinguish each component's OWN minimum (at its argmin) from that
        # component's score evaluated AT the combined optimum.
        js_contact_min = float(js_c_full[j, i_c])
        js_contact_overlap_at_contact_min = float(js_c_overlap[j, i_c])
        js_rg_min = float(js_rg[j, i_r])
        js_contact_at_combined = float(js_c_full[j, i_comb])
        js_contact_overlap_at_combined = float(js_c_overlap[j, i_comb])
        js_rg_at_combined = float(js_rg[j, i_comb])

        crg_diff = abs(K[i_c] - K[i_r])
        disagree_raw = bool(crg_diff > thr)
        # A resolved disagreement is only meaningful when BOTH component optima
        # are interior; otherwise it is undefined (null).
        disagree_resolved = bool(crg_diff > thr) if pair_resolved else None

        k_old_j = float(k_old[j]) if has_old else float("nan")
        # The old-mapping difference is censored (null) when the combined
        # optimum sits at a boundary -- the true best K is not bracketed.
        if has_old and combined_resolved_j:
            k_diff_old: Optional[float] = float(k_best_comb[j] - k_old_j)
        else:
            k_diff_old = None
        old_cmp_resolved = bool(combined_resolved_j) if has_old else None

        rec = {
            "temperature": float(temps[j]),
            "K_best_contact_grid": float(K[i_c]),
            "K_best_rg_grid": float(K[i_r]),
            "K_best_combined_grid": float(K[i_comb]),
            "K_best_combined_quadratic": quad,
            "JS_contact_min": js_contact_min,
            "JS_contact_overlap_at_contact_min": js_contact_overlap_at_contact_min,
            "JS_rg_min": js_rg_min,
            "JS_contact_at_combined": js_contact_at_combined,
            "JS_contact_overlap_at_combined": js_contact_overlap_at_combined,
            "JS_rg_at_combined": js_rg_at_combined,
            "combined_score_best": best_score,
            "second_best_score": second_score,
            "second_best_score_gap": gap,
            "contact_rg_K_difference": float(crg_diff),
            "contact_rg_disagreement_raw": disagree_raw,
            "contact_rg_disagreement_resolved": disagree_resolved,
            "K_old": k_old_j if has_old else None,
            "K_difference_from_old": k_diff_old,
            "old_mapping_comparison_resolved": old_cmp_resolved,
            "contact_boundary_hit_low": bool(c_bhl[j]),
            "contact_boundary_hit_high": bool(c_bhh[j]),
            "rg_boundary_hit_low": bool(r_bhl[j]),
            "rg_boundary_hit_high": bool(r_bhh[j]),
            "combined_boundary_hit_low": bool(m_bhl[j]),
            "combined_boundary_hit_high": bool(m_bhh[j]),
            "n_direct_seeds_at_best_K": int(lib["n_seeds"][i_comb]),
        }
        mapping_records.append(rec)
        rows_map.append([
            rec["temperature"], rec["K_best_contact_grid"], rec["K_best_rg_grid"],
            rec["K_best_combined_grid"], rec["K_best_combined_quadratic"],
            rec["JS_contact_min"], rec["JS_contact_overlap_at_contact_min"],
            rec["JS_rg_min"], rec["JS_contact_at_combined"],
            rec["JS_contact_overlap_at_combined"], rec["JS_rg_at_combined"],
            rec["combined_score_best"], rec["second_best_score_gap"],
            rec["contact_rg_K_difference"], rec["contact_rg_disagreement_raw"],
            rec["contact_rg_disagreement_resolved"], rec["K_old"],
            rec["K_difference_from_old"], rec["old_mapping_comparison_resolved"],
            rec["contact_boundary_hit_low"], rec["contact_boundary_hit_high"],
            rec["rg_boundary_hit_low"], rec["rg_boundary_hit_high"],
            rec["combined_boundary_hit_low"], rec["combined_boundary_hit_high"],
            rec["n_direct_seeds_at_best_K"],
        ])

    # ------------------------------------------------------------------ CSVs
    write_csv(
        out_dir / "empirical_K_mapping.csv",
        [
            "temperature", "K_best_contact_grid", "K_best_rg_grid",
            "K_best_combined_grid", "K_best_combined_quadratic",
            "JS_contact_min", "JS_contact_overlap_at_contact_min", "JS_rg_min",
            "JS_contact_at_combined", "JS_contact_overlap_at_combined",
            "JS_rg_at_combined", "combined_score_best", "second_best_score_gap",
            "contact_rg_K_difference", "contact_rg_disagreement_raw",
            "contact_rg_disagreement_resolved", "K_old", "K_difference_from_old",
            "old_mapping_comparison_resolved",
            "contact_boundary_hit_low", "contact_boundary_hit_high",
            "rg_boundary_hit_low", "rg_boundary_hit_high",
            "combined_boundary_hit_low", "combined_boundary_hit_high",
            "n_direct_seeds_at_best_K",
        ],
        rows_map,
    )

    surf_rows: List[List[Any]] = []
    for j in range(nT):
        for i in range(nK):
            surf_rows.append([
                float(temps[j]), float(K[i]), float(js_c_full[j, i]),
                float(js_c_overlap[j, i]), float(js_rg[j, i]),
                float(combined[j, i]), int(lib["n_seeds"][i]),
            ])
    write_csv(
        out_dir / "empirical_K_score_surface.csv",
        ["temperature", "K", "JS_contact_full", "JS_contact_overlap", "JS_rg",
         "combined_score", "n_direct_seeds"],
        surf_rows,
    )

    # ------------------------------------------------------------- summaries
    # Resolved-versus-censored masks for the three optimization targets.
    contact_boundary = c_bhl | c_bhh
    rg_boundary = r_bhl | r_bhh
    comb_boundary = m_bhl | m_bhh
    pair_resolved_arr = (~contact_boundary) & (~rg_boundary)
    comb_resolved_arr = ~comb_boundary

    n_contact_boundary = int(contact_boundary.sum())
    n_rg_boundary = int(rg_boundary.sum())
    n_comb_boundary = int(comb_boundary.sum())
    frac_contact_boundary = n_contact_boundary / nT
    frac_rg_boundary = n_rg_boundary / nT
    frac_comb_boundary = n_comb_boundary / nT

    downward_steps = int(np.sum(np.diff(k_best_comb) < 0))
    rho = spearman(temps, k_best_comb)
    tlo, thi = float(args.transition_low), float(args.transition_high)

    crg_diffs = np.array([r["contact_rg_K_difference"] for r in mapping_records])

    # Raw contact-vs-Rg disagreement over ALL temperatures (transparency only).
    disagree_raw_arr = crg_diffs > thr
    contact_rg_raw = {
        "disagreement_threshold": thr,
        "n_disagree": int(disagree_raw_arr.sum()),
        "fraction_disagree": float(disagree_raw_arr.mean()),
        "mean_abs_contact_rg_K_diff": float(np.mean(crg_diffs)),
        "max_abs_contact_rg_K_diff": float(np.max(crg_diffs)),
    }

    # Resolved contact-vs-Rg disagreement: only rows where BOTH component optima
    # are interior.  Null (no claim) when no component pair is resolved.
    n_pairs_resolved = int(pair_resolved_arr.sum())
    n_pairs_unresolved = nT - n_pairs_resolved
    if n_pairs_resolved > 0:
        d_res = crg_diffs[pair_resolved_arr]
        n_disagree_res = int(np.sum(d_res > thr))
        contact_rg_resolved = {
            "disagreement_threshold": thr,
            "n_component_pairs_resolved": n_pairs_resolved,
            "n_component_pairs_unresolved": n_pairs_unresolved,
            "n_contact_rg_disagree_resolved": n_disagree_res,
            "fraction_contact_rg_disagree_resolved": n_disagree_res / n_pairs_resolved,
            "mean_abs_contact_rg_K_diff_resolved": float(np.mean(d_res)),
            "max_abs_contact_rg_K_diff_resolved": float(np.max(d_res)),
        }
    else:
        contact_rg_resolved = {
            "disagreement_threshold": thr,
            "n_component_pairs_resolved": 0,
            "n_component_pairs_unresolved": n_pairs_unresolved,
            "n_contact_rg_disagree_resolved": None,
            "fraction_contact_rg_disagree_resolved": None,
            "mean_abs_contact_rg_K_diff_resolved": None,
            "max_abs_contact_rg_K_diff_resolved": None,
        }

    # Old-mapping aggregate errors: raw (all rows, transparency) and resolved
    # (combined-interior rows only, the scientifically usable figure).
    if has_old:
        diffs_all = k_best_comb - k_old
        mad_raw = float(np.mean(np.abs(diffs_all)))
        rms_raw = float(np.sqrt(np.mean(diffs_all ** 2)))
        n_old_resolved = int(comb_resolved_arr.sum())
        n_old_unresolved = nT - n_old_resolved
        if n_old_resolved > 0:
            d_old = diffs_all[comb_resolved_arr]
            mad_res: Optional[float] = float(np.mean(np.abs(d_old)))
            rms_res: Optional[float] = float(np.sqrt(np.mean(d_old ** 2)))
        else:
            mad_res = None
            rms_res = None
    else:
        mad_raw = None
        rms_raw = None
        n_old_resolved = 0
        n_old_unresolved = nT
        mad_res = None
        rms_res = None

    # Transition reach: raw (all rows) and resolved (combined-interior rows).
    in_transition = [float(temps[j]) for j in range(nT)
                     if tlo <= k_best_comb[j] <= thi]
    in_transition_resolved = [float(temps[j]) for j in range(nT)
                              if comb_resolved_arr[j] and tlo <= k_best_comb[j] <= thi]

    if comb_resolved_arr.any():
        kr = k_best_comb[comb_resolved_arr]
        emp_K_min_res: Optional[float] = float(kr.min())
        emp_K_max_res: Optional[float] = float(kr.max())
    else:
        emp_K_min_res = None
        emp_K_max_res = None

    summary = {
        "n_temperatures": nT,
        "n_direct_K": nK,
        "K_min_sampled": float(K.min()),
        "K_max_sampled": float(K.max()),
        "empirical_K_min": float(np.min(k_best_comb)),
        "empirical_K_max": float(np.max(k_best_comb)),
        "empirical_K_min_resolved": emp_K_min_res,
        "empirical_K_max_resolved": emp_K_max_res,
        "n_downward_steps": downward_steps,
        "spearman_T_vs_Kbest": rho,
        "transition_low": tlo,
        "transition_high": thi,
        # Boundary diagnostics for all three optimization targets.
        "boundary_diagnostics": {
            "contact": {
                "n_boundary_optima": n_contact_boundary,
                "fraction_boundary_optima": frac_contact_boundary,
                "n_hit_low": int(c_bhl.sum()),
                "n_hit_high": int(c_bhh.sum()),
            },
            "rg": {
                "n_boundary_optima": n_rg_boundary,
                "fraction_boundary_optima": frac_rg_boundary,
                "n_hit_low": int(r_bhl.sum()),
                "n_hit_high": int(r_bhh.sum()),
            },
            "combined": {
                "n_boundary_optima": n_comb_boundary,
                "fraction_boundary_optima": frac_comb_boundary,
                "n_hit_low": int(m_bhl.sum()),
                "n_hit_high": int(m_bhh.sum()),
            },
        },
        # Flat named mirrors required by the reporting spec.
        "n_contact_boundary_optima": n_contact_boundary,
        "fraction_contact_boundary_optima": frac_contact_boundary,
        "n_rg_boundary_optima": n_rg_boundary,
        "fraction_rg_boundary_optima": frac_rg_boundary,
        "n_combined_boundary_optima": n_comb_boundary,
        "fraction_combined_boundary_optima": frac_comb_boundary,
        # Contact-vs-Rg agreement, raw and resolved.
        "contact_rg_raw": contact_rg_raw,
        "contact_rg_resolved": contact_rg_resolved,
        # Old-mapping comparison, raw (transparency) and resolved (scientific).
        "vs_old_mapping_raw": {
            "available": bool(has_old),
            "mean_abs_difference": mad_raw,
            "rms_difference": rms_raw,
        },
        "vs_old_mapping_resolved": {
            "available": bool(has_old),
            "n_resolved_for_old_mapping_comparison": n_old_resolved,
            "n_unresolved_for_old_mapping_comparison": n_old_unresolved,
            "mean_abs_difference_resolved": mad_res,
            "rms_difference_resolved": rms_res,
        },
        # Transition reach, raw and resolved.
        "transition_raw": {
            "temperatures_in_transition": in_transition,
            "n_temperatures_in_transition": len(in_transition),
        },
        "transition_resolved": {
            "temperatures_in_transition_resolved": in_transition_resolved,
            "n_temperatures_in_transition_resolved": len(in_transition_resolved),
        },
        "weights": {"contact": w_c, "rg": w_r},
    }

    inputs = {
        "molecular_npz": {
            "path": str(args.molecular_npz),
            "sha256": sha256_of_file(args.molecular_npz),
        },
        "fit_summary_json": {
            "path": str(args.fit_summary_json),
            "sha256": sha256_of_file(args.fit_summary_json),
        },
        "direct_k_npz": [
            {"path": str(p), "sha256": sha256_of_file(p)}
            for p in args.direct_k_npz
        ],
    }

    report_json = {
        "schema": "empirical_K_mapping_v2",
        "chain_length_N": N,
        "inputs": inputs,
        "contact_offset": fit["contact_offset"],
        "rg_scale": fit["rg_scale"],
        "weights": {"contact": w_c, "rg": w_r,
                    "raw_contact": cw, "raw_rg": rw},
        "combined_score_definition":
            "combined = w_contact * JS_contact_full + w_rg * JS_rg "
            "(weighted sum of marginal divergences; NOT a joint-distribution fit)",
        "sampled_K": [float(k) for k in K],
        "n_direct_seeds_per_K": {
            f"{float(K[i]):.6g}": int(lib["n_seeds"][i]) for i in range(nK)
        },
        "seeds_per_K": {
            f"{float(K[i]):.6g}": [int(s) for s in lib["seed_lists"][i]]
            for i in range(nK)
        },
        "contact_max_count": int(lib["c_max"]),
        "old_mapping": {
            "model": fit["model"],
            "h": fit["h"],
            "s": fit["s"],
            "formula": "K_old(T) = s - h/T",
            "K_old": [None if not has_old else float(v) for v in k_old],
        },
        "mapping": mapping_records,
        "summary": summary,
    }
    (out_dir / "empirical_K_mapping.json").write_text(
        json.dumps(json_safe(report_json), indent=2), encoding="utf-8"
    )

    # ---------------------------------------------------------------- report
    md = build_markdown_report(report_json, summary, K, k_best_comb, k_old,
                               has_old, direct_files)
    (out_dir / "empirical_K_mapping_report.md").write_text(md, encoding="utf-8")

    # ----------------------------------------------------------------- plots
    plot_mapping_vs_temperature(
        temps, k_best_comb, k_best_contact, k_best_rg,
        k_old if has_old else None, tlo, thi,
        out_dir / "K_mapping_vs_temperature.png",
    )
    plot_score_surface(
        temps, K, combined, tlo, thi,
        out_dir / "score_surface_combined.png",
    )
    plot_component_mappings(
        temps, k_best_contact, k_best_rg, tlo, thi,
        out_dir / "component_K_mappings.png",
    )

    return {"report": report_json, "summary": summary, "out_dir": str(out_dir)}


# Combined boundary fraction at or above which conclusions are gated behind
# expanding the direct-K library (a boundary-censored mapping cannot support
# statements about model adequacy, K(T) refits, or partial compactification).
BOUNDARY_FRACTION_THRESHOLD = 0.25
# Fraction of resolved component pairs that must disagree before contacts and Rg
# are declared to "strongly disagree".
COMPONENT_DISAGREE_FRACTION_THRESHOLD = 0.25
# Resolved old-mapping MAE above which a K(T) refit is worth recommending.
OLD_MAPPING_MAD_THRESHOLD = 0.10


def _f(x, fmt="{:.4g}"):
    """Format a value that may be None/NaN for the Markdown report."""
    if x is None:
        return "n/a"
    if isinstance(x, float) and not np.isfinite(x):
        return "n/a"
    return fmt.format(x)


def build_markdown_report(report_json, summary, K, k_best, k_old, has_old,
                          direct_files) -> str:
    nT = summary["n_temperatures"]
    tlo = summary["transition_low"]
    thi = summary["transition_high"]
    rho = summary["spearman_T_vs_Kbest"]
    bd = summary["boundary_diagnostics"]
    cr_raw = summary["contact_rg_raw"]
    cr_res = summary["contact_rg_resolved"]
    old_raw = summary["vs_old_mapping_raw"]
    old_res = summary["vs_old_mapping_resolved"]
    tr_raw = summary["transition_raw"]
    tr_res = summary["transition_resolved"]

    frac_comb_boundary = summary["fraction_combined_boundary_optima"]
    n_comb_boundary = summary["n_combined_boundary_optima"]
    n_comb_low = bd["combined"]["n_hit_low"]
    n_comb_high = bd["combined"]["n_hit_high"]
    n_pairs_resolved = cr_res["n_component_pairs_resolved"]
    frac_dis_res = cr_res["fraction_contact_rg_disagree_resolved"]
    mad_res = old_res["mean_abs_difference_resolved"]
    n_old_resolved = old_res["n_resolved_for_old_mapping_comparison"]
    n_in_trans_res = tr_res["n_temperatures_in_transition_resolved"]
    emp_K_max_res = summary["empirical_K_max_resolved"]

    boundary_censored = frac_comb_boundary >= BOUNDARY_FRACTION_THRESHOLD

    # Resolved contact/Rg agreement: only decidable when some pairs are resolved.
    if n_pairs_resolved > 0 and frac_dis_res is not None:
        components_agree_resolved: Optional[bool] = (
            frac_dis_res < COMPONENT_DISAGREE_FRACTION_THRESHOLD
        )
    else:
        components_agree_resolved = None

    # ------------------------------- recommendation hierarchy -------------- #
    recs: List[str] = []
    if boundary_censored:
        side = "low-K" if n_comb_low >= n_comb_high else "high-K"
        recs.append(
            "**Expand the direct-K library and rerun the mapping before assessing "
            "model adequacy or refitting K(T).** "
            f"{n_comb_boundary}/{nT} combined optima "
            f"({100 * frac_comb_boundary:.1f}%) sit at a sampled-K boundary "
            f"(predominantly the {side} side), so the true optima are not "
            "bracketed. Boundary-censored optima prevent any conclusion about "
            "model adequacy, contact/Rg agreement, a K(T) refit, or partial "
            "compactification."
        )
        next_action = "expand the direct-K library"
    else:
        if components_agree_resolved is False:
            recs.append(
                "Among resolved temperatures, contacts and Rg disagree on the "
                "preferred K: **consider a second structural coordinate or a "
                "richer Hamiltonian** -- one coupling cannot match both marginals."
            )
            next_action = "consider a richer Hamiltonian"
        elif components_agree_resolved is True:
            if has_old and mad_res is not None and mad_res > OLD_MAPPING_MAD_THRESHOLD:
                recs.append(
                    "Resolved contact and Rg optima agree, but the resolved "
                    "empirical mapping differs substantially from the old h/T - s "
                    "mapping: **refit the analytic K(T) mapping** to the empirical "
                    "interior optima."
                )
                next_action = "refit K(T)"
            else:
                next_action = "no change (resolved mapping is consistent)"
        else:
            next_action = "expand the direct-K library (no resolved pairs)"
        # Case 4 (a characterization of the resolved combined mapping, not a
        # boundary artefact): when combined optima are interior, bracketed, and
        # all below the transition band, the data show partial compactification.
        if (n_in_trans_res == 0 and emp_K_max_res is not None
                and emp_K_max_res < tlo):
            recs.append(
                "Resolved interior combined optima are bracketed yet never reach "
                f"the transition interval [{tlo}, {thi}]: **the molecular data map "
                "to partial compactification** rather than the intrinsic lattice "
                "crossover."
            )
        if not recs:
            recs.append(
                "No single failure mode dominates; the resolved mapping looks "
                "adequate."
            )

    # ----------------------------------- report body ---------------------- #
    lines: List[str] = []
    lines.append("# Empirical molecular-temperature -> direct-K mapping\n")
    lines.append(
        "This diagnostic maps each molecular REMD temperature to the sampled "
        "direct lattice coupling K that best reproduces its contact and Rg "
        "marginals. The combined score is a **weighted sum of marginal "
        "divergences**, not a joint-distribution fit. Results are reported at "
        "three levels: the **raw grid result** (every temperature), the "
        "**resolved interior result** (optima strictly inside the sampled-K "
        "range), and the **boundary-censored result** (optima pinned at an "
        "endpoint, where the true optimum is not bracketed).\n"
    )

    # Raw grid results ---------------------------------------------------- #
    lines.append("## Raw grid results (full dataset, transparency)\n")
    lines.append(
        f"All {len(direct_files)} direct-K file(s) validated (control_mode="
        f"direct_K, temperature_mapping_applied=false, chain length "
        f"N={report_json['chain_length_N']}, finite normalized histograms). "
        f"{summary['n_direct_K']} unique sampled K span "
        f"[{summary['K_min_sampled']:.4g}, {summary['K_max_sampled']:.4g}]; the "
        f"raw combined best-K spans [{summary['empirical_K_min']:.4g}, "
        f"{summary['empirical_K_max']:.4g}] with Spearman(T, K_best) = "
        f"{_f(rho, '{:.3f}')} and {summary['n_downward_steps']} downward step(s) "
        "(no monotonicity imposed).\n"
    )
    lines.append(
        f"Raw contact-vs-Rg disagreement: {cr_raw['n_disagree']}/{nT} temperatures "
        f"beyond threshold {cr_raw['disagreement_threshold']} "
        f"(mean |contact-K - Rg-K| = {_f(cr_raw['mean_abs_contact_rg_K_diff'])}, "
        f"max = {_f(cr_raw['max_abs_contact_rg_K_diff'])}). "
        f"Raw temperatures in transition [{tlo}, {thi}]: "
        f"{tr_raw['n_temperatures_in_transition']}. "
        + (
            f"Raw old-mapping MAE = {_f(old_raw['mean_abs_difference'])}, "
            f"RMS = {_f(old_raw['rms_difference'])} (these include censored rows "
            "and are NOT the scientific figure).\n"
            if has_old else "Old mapping unavailable (fit summary is not 'hs').\n"
        )
    )

    # Resolved interior results ------------------------------------------- #
    lines.append("## Resolved interior results (scientifically usable)\n")
    lines.append(
        f"**Q1. Contact-only optima at boundaries:** "
        f"{bd['contact']['n_boundary_optima']}/{nT} "
        f"({100 * bd['contact']['fraction_boundary_optima']:.1f}%; "
        f"{bd['contact']['n_hit_low']} low, {bd['contact']['n_hit_high']} high).\n"
    )
    lines.append(
        f"**Q2. Rg-only optima at boundaries:** "
        f"{bd['rg']['n_boundary_optima']}/{nT} "
        f"({100 * bd['rg']['fraction_boundary_optima']:.1f}%; "
        f"{bd['rg']['n_hit_low']} low, {bd['rg']['n_hit_high']} high).\n"
    )
    lines.append(
        f"**Q3. Combined optima at boundaries:** {n_comb_boundary}/{nT} "
        f"({100 * frac_comb_boundary:.1f}%; {n_comb_low} low, {n_comb_high} high).\n"
    )
    lines.append(
        f"**Q4. Temperatures with both component optima resolved:** "
        f"{n_pairs_resolved}/{nT} "
        f"({cr_res['n_component_pairs_unresolved']} unresolved).\n"
    )
    lines.append("**Q5. Do contacts and Rg agree among resolved temperatures?** ")
    if n_pairs_resolved == 0:
        lines.append(
            "No component pairs are resolved, so no agreement claim can be made.\n"
        )
    else:
        verdict = "agree" if components_agree_resolved else "disagree"
        lines.append(
            f"{cr_res['n_contact_rg_disagree_resolved']}/{n_pairs_resolved} "
            f"resolved pairs disagree beyond {cr_res['disagreement_threshold']} "
            f"(fraction {_f(frac_dis_res, '{:.3f}')}, mean "
            f"{_f(cr_res['mean_abs_contact_rg_K_diff_resolved'])}, max "
            f"{_f(cr_res['max_abs_contact_rg_K_diff_resolved'])}) -> they "
            f"**{verdict}** among resolved temperatures.\n"
        )
    lines.append("**Q6. Combined interior optima comparable with the old K(T):** ")
    if not has_old:
        lines.append("old mapping unavailable.\n")
    elif n_old_resolved == 0:
        lines.append(
            "No resolved interior empirical optima are available for comparison "
            "with the old analytic mapping.\n"
        )
    else:
        lines.append(
            f"{n_old_resolved} resolved temperature(s); resolved MAE = "
            f"{_f(mad_res)}, RMS = {_f(old_res['rms_difference_resolved'])} "
            f"(K_old(T) = s - h/T with h={_f(report_json['old_mapping']['h'])}, "
            f"s={_f(report_json['old_mapping']['s'])}).\n"
        )
    lines.append("**Q7. Does the resolved mapping reach the transition interval?** ")
    if boundary_censored and n_comb_low >= n_comb_high:
        lines.append(
            f"Unresolved: most combined optima are pinned at the low-K boundary, "
            f"so the direct-K library is insufficient on the low-K side to decide "
            f"whether the transition [{tlo}, {thi}] is reached.\n"
        )
    elif boundary_censored:
        lines.append(
            f"Unresolved: most combined optima are pinned at the high-K boundary; "
            f"the true optima may lie above the sampled range.\n"
        )
    elif n_in_trans_res > 0:
        temps_str = ", ".join(
            f"{t:.3f}" for t in tr_res["temperatures_in_transition_resolved"]
        )
        lines.append(
            f"Yes: {n_in_trans_res} resolved temperature(s) map into "
            f"[{tlo}, {thi}] ({temps_str}).\n"
        )
    else:
        lines.append(
            f"No resolved temperature maps into [{tlo}, {thi}]; resolved combined "
            f"best-K spans [{_f(summary['empirical_K_min_resolved'])}, "
            f"{_f(emp_K_max_res)}].\n"
        )
    lines.append(f"**Q8. Next action:** {next_action}.\n")

    # Recommendation ------------------------------------------------------ #
    lines.append("## Recommendation\n")
    for r in recs:
        lines.append(f"- {r}")
    lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Empirically map molecular REMD temperatures to best direct-K.")
    p.add_argument("--molecular-npz", required=True)
    p.add_argument("--direct-k-npz", required=True, nargs="+")
    p.add_argument("--fit-summary-json", required=True)
    p.add_argument("--contact-weight", type=float, default=0.5)
    p.add_argument("--rg-weight", type=float, default=0.5)
    p.add_argument("--contact-rg-disagreement-threshold", type=float, default=0.10)
    p.add_argument("--transition-low", type=float, default=0.58)
    p.add_argument("--transition-high", type=float, default=0.66)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_mapping(args)
    s = result["summary"]
    print(f"Wrote empirical K mapping to {result['out_dir']}")
    print(f"  temperatures={s['n_temperatures']}  unique K={s['n_direct_K']} "
          f"K in [{s['K_min_sampled']:.3g}, {s['K_max_sampled']:.3g}]")
    print(f"  combined boundary optima: {s['n_combined_boundary_optima']}/"
          f"{s['n_temperatures']}  resolved in-transition: "
          f"{s['transition_resolved']['n_temperatures_in_transition_resolved']}  "
          f"Spearman={s['spearman_T_vs_Kbest']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
