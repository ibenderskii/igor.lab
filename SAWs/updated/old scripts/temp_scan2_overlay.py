#!/usr/bin/env python3
"""
temp_scan2.py

Runs one of the SAW / ISAW model scripts at a range of temperatures and collects:

- scalar equilibrium summaries (E_mean, Rg_back_mean, C_mean)
- optional equilibrium distributions if the sim prints:
      DIST_FILE = <path_to_npz>

Those .npz files are expected to contain (at least):
  c_vals, c_prob, rg_edges, rg_prob, and metadata like T.

Outputs:
  - CSV of scalar summaries
  - plots of mean observables vs T
  - if DIST_FILE is available: heatmaps showing how P(C) and P(Rg) change with T,
    plus a merged .npz you can reuse for custom plotting.

Example:
  python temp_scan2.py --sim-script single_uniform_chain2_DBfix_dists.py --Tmin 0.5 --Tmax 2.0 --nT 16 \
      --steps 300000 --reps 3 --seed 1
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import re
import math
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from statistics import mean, stdev


DEFAULT_SIM_SCRIPT = "thermo_uniform_chain2_DBfix_dists_corrected.py"
# --- regexes for scalar outputs ---
FLOAT = r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
E_MEAN_RE       = re.compile(rf"E_mean\s*=\s*{FLOAT}\s*±\s*{FLOAT}")
RG_BACK_MEAN_RE = re.compile(rf"Rg_back_mean\s*=\s*{FLOAT}\s*±\s*{FLOAT}")
RG_FULL_MEAN_RE = re.compile(rf"Rg_full_mean\s*=\s*{FLOAT}\s*±\s*{FLOAT}")
C_MEAN_RE       = re.compile(rf"C_mean\s*=\s*{FLOAT}\s*±\s*{FLOAT}")
FINAL_RE        = re.compile(rf"Final E\s*=\s*{FLOAT},\s*Rg\s*=\s*{FLOAT}")

# distribution file line
DIST_FILE_RE = re.compile(r"^\s*DIST_FILE\s*=\s*(.+?)\s*$", flags=re.MULTILINE)


def _rebin_prob(old_edges: np.ndarray, old_prob: np.ndarray, new_edges: np.ndarray) -> np.ndarray:
    """
    Rebin a 1D histogram given as probability-per-bin (summing to 1) onto new_edges.
    We assume the probability density is piecewise constant within each old bin.
    """
    old_edges = np.asarray(old_edges, dtype=float)
    old_prob = np.asarray(old_prob, dtype=float)
    new_edges = np.asarray(new_edges, dtype=float)

    n_new = len(new_edges) - 1
    out = np.zeros(n_new, dtype=float)

    if old_prob.size == 0 or n_new <= 0:
        return out

    old_w = np.diff(old_edges)
    with np.errstate(divide="ignore", invalid="ignore"):
        dens = np.where(old_w > 0, old_prob / old_w, 0.0)

    for j in range(len(old_edges) - 1):
        l = old_edges[j]
        r = old_edges[j + 1]
        if not (np.isfinite(l) and np.isfinite(r)) or r <= l:
            continue
        if r <= new_edges[0] or l >= new_edges[-1]:
            continue

        k0 = np.searchsorted(new_edges, l, side="right") - 1
        k1 = np.searchsorted(new_edges, r, side="left")
        k0 = max(k0, 0)
        k1 = min(k1, n_new)

        for k in range(k0, k1):
            a = max(l, new_edges[k])
            b = min(r, new_edges[k + 1])
            overlap = b - a
            if overlap > 0:
                out[k] += dens[j] * overlap

    s = out.sum()
    if s > 0:
        out /= s
    return out


def run_simulation_once(sim_script: str,
                        T: float,
                        steps: int,
                        seed: int | None,
                        extra_args: list[str] | None = None,
                        timeout: int = 3600) -> tuple[dict | None, str | None]:
    """Run sim script and extract scalar observables + optional DIST_FILE path."""
    cmd = [sys.executable, sim_script,
           "--T", str(T),
           "--steps", str(steps)]
    if seed is not None:
        cmd += ["--seed", str(seed)]
    if extra_args:
        cmd += extra_args

    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           text=True, check=False, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None, f"timeout after {timeout}s"

    out = (p.stdout or "") + "\n" + (p.stderr or "")

    if p.returncode != 0:
        return None, out

    result: dict[str, float | str] = {}

    m = E_MEAN_RE.search(out)
    if m:
        result["E_mean"] = float(m.group(1))
        result["E_std"]  = float(m.group(2))

    m = RG_BACK_MEAN_RE.search(out)
    if m:
        result["Rg_back_mean"] = float(m.group(1))
        result["Rg_back_std"]  = float(m.group(2))

    m = RG_FULL_MEAN_RE.search(out)
    if m:
        result["Rg_full_mean"] = float(m.group(1))
        result["Rg_full_std"]  = float(m.group(2))

    m = C_MEAN_RE.search(out)
    if m:
        result["C_mean"] = float(m.group(1))
        result["C_std"]  = float(m.group(2))

    # optional dist file
    m = DIST_FILE_RE.search(out)
    if m:
        dist_path = m.group(1).strip().strip('"').strip("'")
        result["dist_file"] = dist_path

    # if we found at least E_mean and some Rg mean, accept
    if "E_mean" in result and ("Rg_back_mean" in result or "Rg_full_mean" in result):
        return result, None

    # fallback: old single-line format
    m = FINAL_RE.search(out)
    if m:
        result = {
            "E_mean": float(m.group(1)),
            "E_std": 0.0,
            "Rg_back_mean": float(m.group(2)),
            "Rg_back_std": 0.0,
        }
        # still keep dist_file if present
        m2 = DIST_FILE_RE.search(out)
        if m2:
            dist_path = m2.group(1).strip().strip('"').strip("'")
            result["dist_file"] = dist_path
        return result, None

    return None, out


def scan_temperatures(sim_script: str,
                      Ts: np.ndarray,
                      steps: int,
                      reps: int,
                      seed0: int | None,
                      extra_args: list[str] | None = None,
                      verbose: bool = True,
                      timeout: int = 3600,
                      n_workers: int = 1) -> list[dict]:

    def stats(values):
        clean = [v for v in values if not math.isnan(v)]
        if len(clean) == 0:
            return math.nan, math.nan
        if len(clean) == 1:
            return clean[0], 0.0
        return mean(clean), stdev(clean)

    # Phase A: pre-compute all (T_index, rep, T, seed) jobs so seeds are
    # assigned deterministically before any execution begins.
    jobs: list[tuple[int, int, float, int | None]] = []
    for i, T in enumerate(Ts):
        for r in range(reps):
            seed = None if seed0 is None else (seed0 + i * reps + r)
            jobs.append((i, r, float(T), seed))

    # Phase B: execute jobs, collect raw per-run results keyed by (i, r).
    raw: dict[tuple[int, int], tuple[dict | None, str | None]] = {}

    def _report(r, T, res, err):
        if err:
            print(f"[T={T:.3g} rep={r}] run failed / parse error; stdout/stderr:")
            print(err[:4000])
        else:
            msg = (f"[T={T:.3g} rep={r}] "
                   f"E={res.get('E_mean', math.nan):.4f}, "
                   f"Rg_back={res.get('Rg_back_mean', math.nan):.4f}, "
                   f"Rg_full={res.get('Rg_full_mean', math.nan):.4f}, "
                   f"C={res.get('C_mean', math.nan):.3f}")
            df = res.get("dist_file")
            if df:
                msg += f"  (dist: {df})"
            print(msg)

    if n_workers > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            future_to_meta = {
                executor.submit(run_simulation_once, sim_script, T, steps, seed,
                                extra_args, timeout): (i, r, T)
                for (i, r, T, seed) in jobs
            }
            for fut in as_completed(future_to_meta):
                i, r, T = future_to_meta[fut]
                res, err = fut.result()
                raw[(i, r)] = (res, err)
                if verbose:
                    _report(r, T, res, err)
    else:
        for (i, r, T, seed) in jobs:
            res, err = run_simulation_once(sim_script, T, steps, seed,
                                           extra_args=extra_args, timeout=timeout)
            raw[(i, r)] = (res, err)
            if verbose:
                _report(r, T, res, err)

    # Phase C: group by T index and compute per-T statistics.
    results = []
    for i, T in enumerate(Ts):
        Es, Rb_means, Rf_means, C_means = [], [], [], []
        dist_files: list[str] = []

        for r in range(reps):
            res, err = raw.get((i, r), (None, "missing"))
            if err:
                Es.append(math.nan); Rb_means.append(math.nan)
                Rf_means.append(math.nan); C_means.append(math.nan)
                continue

            Es.append(float(res.get("E_mean", math.nan)))
            Rb_means.append(float(res.get("Rg_back_mean", math.nan)))
            Rf_means.append(float(res.get("Rg_full_mean", math.nan)))
            C_means.append(float(res.get("C_mean", math.nan)))

            df = res.get("dist_file")
            if isinstance(df, str) and df.strip():
                dist_files.append(df.strip())

        E_mean, E_std = stats(Es)
        Rb_mean, Rb_std = stats(Rb_means)
        Rf_mean, Rf_std = stats(Rf_means)
        C_mean, C_std = stats(C_means)

        results.append({
            "T": float(T),
            "E_mean": E_mean, "E_std": E_std,
            "Rg_back_mean": Rb_mean, "Rg_back_std": Rb_std,
            "Rg_full_mean": Rf_mean, "Rg_full_std": Rf_std,
            "C_mean": C_mean, "C_std": C_std,
            "dist_files": dist_files,
        })
    return results


def save_results_csv(results, out_csv="temp_scan_results.csv"):
    keys = ["T", "E_mean", "E_std", "Rg_back_mean", "Rg_back_std", "Rg_full_mean", "Rg_full_std", "C_mean", "C_std"]
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(keys)
        for r in results:
            w.writerow([r.get(k, math.nan) for k in keys])
    print(f"Saved CSV to {out_csv}")


def plot_results(results, out_prefix="temp_scan"):
    Ts = np.array([r["T"] for r in results], dtype=float)

    Rg_back_means = np.array([r["Rg_back_mean"] for r in results], dtype=float)
    Rg_back_errs  = np.array([r["Rg_back_std"] for r in results], dtype=float)

    Rg_full_means = np.array([r["Rg_full_mean"] for r in results], dtype=float)
    Rg_full_errs  = np.array([r["Rg_full_std"] for r in results], dtype=float)

    E_means  = np.array([r["E_mean"] for r in results], dtype=float)
    E_errs   = np.array([r["E_std"] for r in results], dtype=float)

    C_means = np.array([r["C_mean"] for r in results], dtype=float)
    C_errs  = np.array([r["C_std"] for r in results], dtype=float)

    # Rg
    plt.figure(figsize=(6, 4))
    plt.errorbar(Ts, Rg_back_means, yerr=Rg_back_errs, marker='o', linestyle='-', capsize=3, label='Rg_backbone')
    if not np.all(np.isnan(Rg_full_means)):
        plt.errorbar(Ts, Rg_full_means, yerr=Rg_full_errs, marker='s', linestyle='--', capsize=3, label='Rg_full')
    plt.xlabel("Temperature T")
    plt.ylabel("Rg (mean ± std)")
    plt.title("Rg vs T")
    plt.legend()
    plt.tight_layout()
    png1 = f"{out_prefix}_Rg_vs_T.png"
    plt.savefig(png1, dpi=150)
    plt.close()
    print(f"Saved {png1}")

    # Energy
    plt.figure(figsize=(6, 4))
    plt.errorbar(Ts, E_means, yerr=E_errs, marker='o', linestyle='-', capsize=3)
    plt.xlabel("Temperature T")
    plt.ylabel("Energy E (mean ± std)")
    plt.title("E vs T")
    plt.tight_layout()
    png2 = f"{out_prefix}_E_vs_T.png"
    plt.savefig(png2, dpi=150)
    plt.close()
    print(f"Saved {png2}")

    # Contacts
    if not np.all(np.isnan(C_means)):
        plt.figure(figsize=(6, 4))
        plt.errorbar(Ts, C_means, yerr=C_errs, marker='o', linestyle='-', capsize=3)
        plt.xlabel("Temperature T")
        plt.ylabel("Nonbonded contacts (mean ± std)")
        plt.title("Contacts vs T")
        plt.tight_layout()
        png3 = f"{out_prefix}_contacts_vs_T.png"
        plt.savefig(png3, dpi=150)
        plt.close()
        print(f"Saved {png3}")


def aggregate_and_plot_distributions(results: list[dict],
                                     out_prefix: str = "temp_scan",
                                     rg_bins_common: int = 80) -> None:
    """
    If DIST_FILE outputs exist, build P(C|T) and P(Rg|T) over a *common* Rg axis
    so you can plot temperature evolution cleanly.
    """
    # flatten all dist files
    all_files = []
    for r in results:
        all_files.extend(r.get("dist_files", []) or [])
    all_files = [f for f in all_files if f]

    if not all_files:
        print("No DIST_FILE outputs found. Skipping distribution plots.")
        return

    # load all (ignore missing)
    loaded = []
    for f in all_files:
        try:
            loaded.append(np.load(f, allow_pickle=True))
        except Exception as e:
            print(f"Warning: couldn't load {f}: {e}")

    if not loaded:
        print("No distribution files could be loaded. Skipping distribution plots.")
        return

    # global contacts max
    maxC = 0
    rg_lo, rg_hi = math.inf, -math.inf
    for d in loaded:
        try:
            maxC = max(maxC, int(np.max(d["c_vals"])))
        except Exception:
            pass
        try:
            edges = np.asarray(d["rg_edges"], dtype=float)
            rg_lo = min(rg_lo, float(edges[0]))
            rg_hi = max(rg_hi, float(edges[-1]))
        except Exception:
            pass

    if not np.isfinite(rg_lo) or not np.isfinite(rg_hi) or rg_hi <= rg_lo:
        rg_lo, rg_hi = 0.0, 1.0

    rg_edges_common = np.linspace(rg_lo, rg_hi, int(rg_bins_common) + 1)
    rg_centers_common = 0.5 * (rg_edges_common[:-1] + rg_edges_common[1:])

    Ts = np.array([r["T"] for r in results], dtype=float)
    Pc = np.full((len(results), maxC + 1), np.nan, dtype=float)
    Prg = np.full((len(results), len(rg_centers_common)), np.nan, dtype=float)

    for i, r in enumerate(results):
        files = r.get("dist_files", []) or []
        if not files:
            continue

        # contacts
        pc_reps = []
        prg_reps = []

        for f in files:
            try:
                d = np.load(f, allow_pickle=True)
            except Exception:
                continue

            # contacts (discrete)
            try:
                row = np.zeros(maxC + 1, dtype=float)
                c_vals = np.asarray(d["c_vals"]).astype(int)
                c_prob = np.asarray(d["c_prob"], dtype=float)
                row[c_vals] = c_prob
                s = row.sum()
                if s > 0:
                    row /= s
                pc_reps.append(row)
            except Exception:
                pass

            # Rg (rebin onto common edges)
            try:
                old_edges = np.asarray(d["rg_edges"], dtype=float)
                old_prob = np.asarray(d["rg_prob"], dtype=float)
                new_prob = _rebin_prob(old_edges, old_prob, rg_edges_common)
                prg_reps.append(new_prob)
            except Exception:
                pass

        if pc_reps:
            Pc[i, :] = np.mean(pc_reps, axis=0)
            s = np.nansum(Pc[i, :])
            if s > 0:
                Pc[i, :] /= s

        if prg_reps:
            Prg[i, :] = np.mean(prg_reps, axis=0)
            s = np.nansum(Prg[i, :])
            if s > 0:
                Prg[i, :] /= s

    # save merged distributions
    out_npz = f"{out_prefix}_equilibrium_distributions.npz"
    np.savez_compressed(
        out_npz,
        Ts=Ts,
        c_vals=np.arange(maxC + 1, dtype=int),
        Pc=Pc,
        rg_edges=rg_edges_common,
        rg_centers=rg_centers_common,
        Prg=Prg,
    )
    print(f"Saved merged distributions to {out_npz}")

    # overlaid distributions (one curve per temperature)
    import matplotlib.cm as cm
    import matplotlib.colors as colors

    cmap = cm.coolwarm
    norm = colors.Normalize(vmin=float(np.nanmin(Ts)), vmax=float(np.nanmax(Ts)))
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    c_axis = np.arange(maxC + 1, dtype=int)

    fig, (ax_rg, ax_c) = plt.subplots(1, 2, figsize=(14, 5))

    for i, T in enumerate(Ts):
        col = cmap(norm(T))

        y_rg = Prg[i, :]
        if np.any(np.isfinite(y_rg)):
            ax_rg.plot(rg_centers_common, y_rg, color=col, alpha=0.65, linewidth=1.0)

        y_c = Pc[i, :]
        if np.any(np.isfinite(y_c)):
            ax_c.plot(c_axis, y_c, color=col, alpha=0.65, linewidth=1.0)

    ax_rg.set_xlabel("Rg")
    ax_rg.set_ylabel("P(Rg)")
    ax_rg.set_title("Equilibrium Rg distribution")

    ax_c.set_xlabel("Contacts")
    ax_c.set_ylabel("P(contacts)")
    ax_c.set_title("Equilibrium contact distribution")

    fig.colorbar(sm, ax=ax_rg, label="T")
    fig.colorbar(sm, ax=ax_c, label="T")

    fig.tight_layout()
    png_overlay = f"{out_prefix}_distributions_overlay.png"
    fig.savefig(png_overlay, dpi=200)
    plt.close(fig)
    print(f"Saved {png_overlay}")
    # Also save individual panels (same data, one axis each)
    # Rg-only
    fig_rg, ax = plt.subplots(figsize=(6.5, 4.5))
    for i, T in enumerate(Ts):
        y_rg = Prg[i, :]
        if np.any(np.isfinite(y_rg)):
            ax.plot(rg_centers_common, y_rg, color=cmap(norm(T)), alpha=0.65, linewidth=1.0)
    ax.set_xlabel("Rg")
    ax.set_ylabel("P(Rg)")
    ax.set_title("Equilibrium Rg distribution (colored by T)")
    fig_rg.colorbar(sm, ax=ax, label="T")
    fig_rg.tight_layout()
    png_prg = f"{out_prefix}_Prg_vs_T.png"
    fig_rg.savefig(png_prg, dpi=200)
    plt.close(fig_rg)
    print(f"Saved {png_prg}")

    # Contacts-only
    fig_c, ax = plt.subplots(figsize=(6.5, 4.5))
    for i, T in enumerate(Ts):
        y_c = Pc[i, :]
        if np.any(np.isfinite(y_c)):
            ax.plot(c_axis, y_c, color=cmap(norm(T)), alpha=0.65, linewidth=1.0)
    ax.set_xlabel("Contacts")
    ax.set_ylabel("P(contacts)")
    ax.set_title("Equilibrium contact distribution (colored by T)")
    fig_c.colorbar(sm, ax=ax, label="T")
    fig_c.tight_layout()
    png_pc = f"{out_prefix}_Pc_vs_T.png"
    fig_c.savefig(png_pc, dpi=200)
    plt.close(fig_c)
    print(f"Saved {png_pc}")



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Tmin", type=float, default=280)
    ap.add_argument("--Tmax", type=float, default=380)
    ap.add_argument("--nT", type=int, default=15)

    ap.add_argument("--reps", type=int, default=1, help="replicates per T")
    ap.add_argument("--steps", type=int, default=1000000)
    ap.add_argument("--seed", type=int, default=133)
    ap.add_argument("--sim-script", type=str, default=DEFAULT_SIM_SCRIPT)
    ap.add_argument("--extra-args", type=str, default="", help="extra args to pass to sim script (quoted)")
    ap.add_argument("--timeout", type=int, default=3600)

    ap.add_argument("--out-csv", type=str, default="temp_scan_results.csv")
    ap.add_argument("--out-prefix", type=str, default="temp_scan")
    ap.add_argument("--rg-bins-common", type=int, default=80,
                    help="common bin count for the merged P(Rg|T) heatmap (rebinning)")
    ap.add_argument("--n-workers", type=int, default=1,
                    help="number of parallel workers (ProcessPoolExecutor); default 1 = serial")

    args = ap.parse_args()

    Ts = np.linspace(args.Tmin, args.Tmax, args.nT)
    extra_args = args.extra_args.split() if args.extra_args.strip() else None

    print(f"Running scan: T in [{args.Tmin}, {args.Tmax}] ({args.nT} points), {args.reps} reps each, steps={args.steps}, workers={args.n_workers}")
    results = scan_temperatures(
        args.sim_script,
        Ts,
        args.steps,
        args.reps,
        args.seed,
        extra_args=extra_args,
        verbose=True,
        timeout=args.timeout,
        n_workers=args.n_workers,
    )

    save_results_csv(results, out_csv=args.out_csv)
    plot_results(results, out_prefix=args.out_prefix)
    aggregate_and_plot_distributions(results, out_prefix=args.out_prefix, rg_bins_common=args.rg_bins_common)


if __name__ == "__main__":
    main()
