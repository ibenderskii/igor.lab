#!/usr/bin/env python3
"""
temp_scan.py

Driver to run the lattice polymer script at many temperatures, collect final Rg and E,
and produce Rg(T) and E(T) plots + CSV output.

Usage:
    python temp_scan.py
or to change scan parameters:
    python temp_scan.py --Tmin 0.5 --Tmax 4.0 --nT 20 --reps 3 --steps 200000
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

# <-- change this if your simulation file name differs
SIM_SCRIPT = "polymer_hp_grafts_with_dimer_sidechain.py"

FINAL_RE = re.compile(r"Final E\s*=\s*([-\d\.]+),\s*Rg\s*=\s*([-\d\.]+)")

def run_simulation_once(T, steps, seed, extra_args=None):
    """Run the simulation script once and parse Final E and Rg from stdout."""
    cmd = [sys.executable, SIM_SCRIPT,
           "--T", str(T),
           "--steps", str(steps),
           "--seed", str(seed)]
    if extra_args:
        cmd += extra_args
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False, timeout=3600)
    except subprocess.TimeoutExpired:
        return None, None, f"timeout after 3600s"
    out = p.stdout + "\n" + p.stderr
    m = FINAL_RE.search(out)
    if not m:
        # return full stdout for debugging if parse fails
        return None, None, out
    E = float(m.group(1))
    Rg = float(m.group(2))
    return E, Rg, None

def scan_temperatures(Ts, steps, reps, seed0, extra_args=None, verbose=True):
    results = []
    for i, T in enumerate(Ts):
        Es, Rgs = [], []
        for r in range(reps):
            seed = None if seed0 is None else (seed0 + i*reps + r)
            E, Rg, err = run_simulation_once(T, steps, seed, extra_args=extra_args)
            if err:
                if verbose:
                    print(f"[T={T:.3g} rep={r}] run failed / parse error; returning stdout for debugging:")
                    print(err)
                # if a run fails, append NaNs and continue
                Es.append(math.nan); Rgs.append(math.nan)
            else:
                Es.append(E); Rgs.append(Rg)
                if verbose:
                    print(f"[T={T:.3g} rep={r}] E={E:.4f}, Rg={Rg:.4f}")
        # compute mean and std (ignore NaNs)
        Es_clean = [x for x in Es if not math.isnan(x)]
        Rgs_clean = [x for x in Rgs if not math.isnan(x)]
        e_mean, e_std = (mean(Es_clean), stdev(Es_clean)) if len(Es_clean) > 1 else (Es_clean[0] if Es_clean else math.nan, 0.0)
        r_mean, r_std = (mean(Rgs_clean), stdev(Rgs_clean)) if len(Rgs_clean) > 1 else (Rgs_clean[0] if Rgs_clean else math.nan, 0.0)
        results.append({
            "T": T,
            "E_mean": e_mean, "E_std": e_std,
            "Rg_mean": r_mean, "Rg_std": r_std,
            "Es": Es, "Rgs": Rgs
        })
    return results

def save_results_csv(results, out_csv="temp_scan_results.csv"):
    keys = ["T", "E_mean", "E_std", "Rg_mean", "Rg_std"]
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(keys)
        for r in results:
            w.writerow([r["T"], r["E_mean"], r["E_std"], r["Rg_mean"], r["Rg_std"]])
    print(f"Saved CSV to {out_csv}")

def plot_results(results, out_prefix="temp_scan"):
    Ts = np.array([r["T"] for r in results])
    Rg_means = np.array([r["Rg_mean"] for r in results])
    Rg_errs  = np.array([r["Rg_std"] for r in results])
    E_means  = np.array([r["E_mean"] for r in results])
    E_errs   = np.array([r["E_std"] for r in results])

    plt.figure(figsize=(6,4))
    plt.errorbar(Ts, Rg_means, yerr=Rg_errs, marker='o', linestyle='-', capsize=3)
    plt.xlabel("Temperature T")
    plt.ylabel("Rg (mean ± std)")
    plt.title("Rg vs T")
    plt.tight_layout()
    png1 = f"{out_prefix}_Rg_vs_T.png"
    plt.savefig(png1, dpi=150)
    plt.close()
    print(f"Saved {png1}")

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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Tmin", type=float, default=0.5)
    ap.add_argument("--Tmax", type=float, default=4.0)
    ap.add_argument("--nT", type=int, default=20)
    ap.add_argument("--reps", type=int, default=3, help="replicates per T")
    ap.add_argument("--steps", type=int, default=200000)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--sim-script", type=str, default=SIM_SCRIPT)
    ap.add_argument("--extra-args", type=str, default="", help="extra args to pass to sim script (quoted)")
    ap.add_argument("--out-csv", type=str, default="temp_scan_results.csv")
    ap.add_argument("--out-prefix", type=str, default="temp_scan")
    args = ap.parse_args()

    global SIM_SCRIPT
    SIM_SCRIPT = args.sim_script

    Ts = np.linspace(args.Tmin, args.Tmax, args.nT)
    extra_args = args.extra_args.split() if args.extra_args.strip() else None

    print(f"Running scan: T in [{args.Tmin}, {args.Tmax}] ({args.nT} points), {args.reps} reps each, steps={args.steps}")
    results = scan_temperatures(Ts, args.steps, args.reps, args.seed, extra_args=extra_args, verbose=True)

    save_results_csv(results, out_csv=args.out_csv)
    plot_results(results, out_prefix=args.out_prefix)

if __name__ == "__main__":
    main()
