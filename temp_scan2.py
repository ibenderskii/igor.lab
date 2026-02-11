#!/usr/bin/env python3
"""
temp_scan.py (updated to parse instrumented sim averages)

Usage:
    python temp_scan.py --sim-script polymer_hp_grafts_with_dimer_sidechains_instrumented.py
"""
import argparse
import subprocess
import sys
import re
import math
import csv
import numpy as np
import matplotlib.pyplot as plt
from statistics import mean, stdev

DEFAULT_SIM_SCRIPT = "single_uniform_chain.py"

# fallback regex (old single-line format)
#FINAL_RE = re.compile(r"Final E\s*=\s*([-\d\.]+),\s*Rg\s*=\s*([-\d\.]+)")

# regexes for the instrumented averaged output:
#E_MEAN_RE      = re.compile(r"E_mean\s*=\s*([-\d\.]+)\s*±\s*([-\d\.]+)")
#RG_BACK_MEAN_RE= re.compile(r"Rg_back_mean\s*=\s*([-\d\.]+)\s*±\s*([-\d\.]+)")
#RG_FULL_MEAN_RE= re.compile(r"Rg_full_mean\s*=\s*([-\d\.]+)\s*±\s*([-\d\.]+)")
#C_MEAN_RE     = re.compile(r"C_mean\s*=\s*([-\d\.]+)\s*±\s*([-\d\.]+)")

#new regex idk if it works if not switch back to one above
FLOAT = r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
E_MEAN_RE       = re.compile(rf"E_mean\s*=\s*{FLOAT}\s*±\s*{FLOAT}")
RG_BACK_MEAN_RE = re.compile(rf"Rg_back_mean\s*=\s*{FLOAT}\s*±\s*{FLOAT}")
RG_FULL_MEAN_RE = re.compile(rf"Rg_full_mean\s*=\s*{FLOAT}\s*±\s*{FLOAT}")
C_MEAN_RE       = re.compile(rf"C_mean\s*=\s*{FLOAT}\s*±\s*{FLOAT}")
FINAL_RE        = re.compile(rf"Final E\s*=\s*{FLOAT},\s*Rg\s*=\s*{FLOAT}")

def run_simulation_once(sim_script, T, steps, seed, extra_args=None, timeout=3600):
    """Run sim script and extract averaged observables (if present). Returns dict or (None,err)."""
    cmd = [sys.executable, sim_script,
           "--T", str(T),
           "--steps", str(steps),
           "--seed", str(seed)]
    if extra_args:
        cmd += extra_args
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           text=True, check=False, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None, f"timeout after {timeout}s"

    out = (p.stdout or "") + "\n" + (p.stderr or "")
    # Try instrumented averages first
    result = {}
    m = E_MEAN_RE.search(out)
    if m:
        try:
            result["E_mean"] = float(m.group(1))
            result["E_std"]  = float(m.group(2))
        except Exception:
            pass
    m = RG_BACK_MEAN_RE.search(out)
    if m:
        result["Rg_back_mean"] = float(m.group(1)); result["Rg_back_std"] = float(m.group(2))
    m = RG_FULL_MEAN_RE.search(out)
    if m:
        result["Rg_full_mean"] = float(m.group(1)); result["Rg_full_std"] = float(m.group(2))
    m = C_MEAN_RE.search(out)
    if m:
        result["C_mean"] = float(m.group(1)); result["C_std"] = float(m.group(2))

    # If we found at least E_mean and Rg_back_mean use those
    if "E_mean" in result and ("Rg_back_mean" in result or "Rg_full_mean" in result):
        return result, None

    # else try old single-line format as fallback
    m = FINAL_RE.search(out)
    if m:
        try:
            result = {
                "E_mean": float(m.group(1)),
                "E_std": 0.0,
                "Rg_back_mean": float(m.group(2)),
                "Rg_back_std": 0.0,
            }
            return result, None
        except Exception:
            pass

    # nothing parsed -> return stdout as error for debugging
    return None, out

def scan_temperatures(sim_script, Ts, steps, reps, seed0, extra_args=None, verbose=True):
    results = []
    for i, T in enumerate(Ts):
        Es, E_stds = [], []
        Rb_means, Rb_stds = [], []
        Rf_means, Rf_stds = [], []
        C_means, C_stds = [], []

        for r in range(reps):
            seed = None if seed0 is None else (seed0 + i*reps + r)
            res, err = run_simulation_once(sim_script, T, steps, seed, extra_args=extra_args)
            if err:
                if verbose:
                    print(f"[T={T:.3g} rep={r}] run failed / parse error; stdout/stderr:")
                    print(err)
                # record NaNs so later stats don't break
                Es.append(math.nan); E_stds.append(math.nan)
                Rb_means.append(math.nan); Rb_stds.append(math.nan)
                Rf_means.append(math.nan); Rf_stds.append(math.nan)
                C_means.append(math.nan); C_stds.append(math.nan)
                continue

            # res is a dict with some or all fields
            e = res.get("E_mean", math.nan); e_std = res.get("E_std", 0.0)
            rb = res.get("Rg_back_mean", math.nan); rb_std = res.get("Rg_back_std", 0.0)
            rf = res.get("Rg_full_mean", math.nan); rf_std = res.get("Rg_full_std", 0.0)
            C = res.get("C_mean", math.nan); C_std = res.get("C_std", 0.0)

            Es.append(e); E_stds.append(e_std)
            Rb_means.append(rb); Rb_stds.append(rb_std)
            Rf_means.append(rf); Rf_stds.append(rf_std)
            C_means.append(C); C_stds.append(C_std)

            if verbose:
                print(f"[T={T:.3g} rep={r}] E={e:.4f} (±{e_std:.3f}), Rg_back={rb:.4f} (±{rb_std:.3f}), Rg_full={rf:.4f} (±{rf_std:.3f}), C={C:.3f}")

        # clean NaNs for summary stats
        def stats(values, stds=None):
            clean = [v for v in values if not math.isnan(v)]
            if len(clean) == 0:
                return math.nan, math.nan
            if len(clean) == 1:
                return clean[0], 0.0
            return mean(clean), stdev(clean)

        E_mean, E_std = stats(Es)
        Rb_mean, Rb_std = stats(Rb_means)
        Rf_mean, Rf_std = stats(Rf_means)
        C_mean, C_std = stats(C_means)

        results.append({
            "T": T,
            "E_mean": E_mean, "E_std": E_std,
            "Rg_back_mean": Rb_mean, "Rg_back_std": Rb_std,
            "Rg_full_mean": Rf_mean, "Rg_full_std": Rf_std,
            "C_mean": C_mean, "C_std": C_std,
            "Es": Es, "Rg_backs": Rb_means, "Rg_fulls": Rf_means, "Cs": C_means
        })
    return results

def save_results_csv(results, out_csv="temp_scan_results.csv"):
    keys = ["T", "E_mean", "E_std", "Rg_back_mean", "Rg_back_std", "Rg_full_mean", "Rg_full_std", "C_mean", "C_std"]
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(keys)
        for r in results:
            w.writerow([
                r["T"], r["E_mean"], r["E_std"],
                r["Rg_back_mean"], r["Rg_back_std"],
                r["Rg_full_mean"], r["Rg_full_std"],
                r["C_mean"], r["C_std"]
            ])
    print(f"Saved CSV to {out_csv}")

def plot_results(results, out_prefix="temp_scan"):
    Ts = np.array([r["T"] for r in results])
    Rg_back_means = np.array([r["Rg_back_mean"] for r in results])
    Rg_back_errs  = np.array([r["Rg_back_std"] for r in results])
    Rg_full_means = np.array([r["Rg_full_mean"] for r in results])
    Rg_full_errs  = np.array([r["Rg_full_std"] for r in results])
    E_means  = np.array([r["E_mean"] for r in results])
    E_errs   = np.array([r["E_std"] for r in results])
    C_means = np.array([r["C_mean"] for r in results])
    C_errs  = np.array([r["C_std"] for r in results])

    # Rg backbone
    plt.figure(figsize=(6,4))
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
    plt.figure(figsize=(6,4))
    plt.errorbar(Ts, E_means, yerr=E_errs, marker='o', linestyle='-', capsize=3)
    plt.xlabel("Temperature T")
    plt.ylabel("Energy E (mean ± std)")
    plt.title("E vs T")
    plt.tight_layout()
    png2 = f"{out_prefix}_E_vs_T.png"
    plt.savefig(png2, dpi=150)
    plt.close()
    print(f"Saved {png2}")

    # Nonbonded contacts (if present)
    if not np.all(np.isnan(C_means)):
        plt.figure(figsize=(6,4))
        plt.errorbar(Ts, C_means, yerr=C_errs, marker='o', linestyle='-', capsize=3)
        plt.xlabel("Temperature T")
        plt.ylabel("Nonbonded contacts (mean ± std)")
        plt.title("Contacts vs T")
        plt.tight_layout()
        png3 = f"{out_prefix}_contacts_vs_T.png"
        plt.savefig(png3, dpi=150)
        plt.close()
        print(f"Saved {png3}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Tmin", type=float, default=1.5)
    ap.add_argument("--Tmax", type=float, default=2.5)
    ap.add_argument("--nT", type=int, default=20)
    
    ap.add_argument("--reps", type=int, default=3, help="replicates per T")
    ap.add_argument("--steps", type=int, default=500000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sim-script", type=str, default=DEFAULT_SIM_SCRIPT)
    ap.add_argument("--extra-args", type=str, default="", help="extra args to pass to sim script (quoted)")
    ap.add_argument("--out-csv", type=str, default="temp_scan_results.csv")
    ap.add_argument("--out-prefix", type=str, default="temp_scan")
    args = ap.parse_args()

    sim_script = args.sim_script
    Ts = np.linspace(args.Tmin, args.Tmax, args.nT)
    extra_args = args.extra_args.split() if args.extra_args.strip() else None

    print(f"Running scan: T in [{args.Tmin}, {args.Tmax}] ({args.nT} points), {args.reps} reps each, steps={args.steps}")
    results = scan_temperatures(sim_script, Ts, args.steps, args.reps, args.seed, extra_args=extra_args, verbose=True)

    save_results_csv(results, out_csv=args.out_csv)
    plot_results(results, out_prefix=args.out_prefix)

if __name__ == "__main__":
    main()
