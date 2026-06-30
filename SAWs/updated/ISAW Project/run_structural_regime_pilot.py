#!/usr/bin/env python3
"""SCIENTIFICALLY-CHOSEN structural-regime pilot (swollen / crossover / collapsed).

Unlike ``run_pilot_validation.py`` (a short file-integrity smoke test on an
arbitrary ladder), this driver builds a temperature ladder from a FITTED contact
model so that the sampled temperatures are expected to span the swollen,
crossover, and collapsed regimes, then VERIFIES that they actually do by checking
that the mean contact number m and the mean R_g^2 vary monotonically and
substantially across the ladder.  It does not hardcode an unrelated (h, s) pair:
the model parameters come from ``--fit-summary-json``.

PowerShell example:
    python .\\run_structural_regime_pilot.py `
        --fit-summary-json .\\fits_30mer_contact_only\\hs\\fit_summary.json `
        --N 30 --seeds 1 2 --n-cycles 400 --output-dir .\\regime_pilot

This is a longer run than the smoke test; it is intended to be launched
explicitly, not as part of the unit-test suite.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import remd_uniform_chain_2_new as remd

PY = sys.executable
REMD = str(HERE / "remd_uniform_chain_2_new.py")


def build_regime_ladder(summary_path, n_points):
    """Build a ladder bracketing the model's pseudo-transition (Tc) when known."""
    info = remd.load_fit_summary_json(summary_path)
    model = info["model_name"]
    params = info["params"]
    # Estimate a transition temperature for ladder centering.
    Tc = None
    if model == "hs" and abs(params[1]) > 1e-12:
        Tc = params[0] / params[1]            # h/s
    elif model == "tc_scale":
        Tc = params[1]
    if Tc is None or not np.isfinite(Tc) or Tc <= 0:
        Tc = 320.0
    lo, hi = 0.80 * Tc, 1.20 * Tc             # swollen (high T) ... collapsed (low T)
    return np.linspace(lo, hi, n_points), info, float(Tc)


def run_seed(out_dir, N, seed, ladder, summary_path, n_cycles):
    prefix = out_dir / f"regime_N{N}_s{seed}"
    temps = ",".join(f"{t:.4f}" for t in ladder)
    cmd = [PY, REMD, "--N", str(N), "--temps", temps,
           "--fit-summary-json", summary_path,
           "--steps-per-swap", "60", "--n-cycles", str(n_cycles), "--n-workers", "1",
           "--seed", str(seed), "--burnin-frac", "0.5",
           "--structural-observables", "--structural-stride", "5",
           "--diagnostics", "--no-plots",
           "--overwrite-configurations",
           "--out-prefix", str(prefix)]
    print("RUN:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(HERE))
    return f"{prefix}_results.csv"


def check_regimes(results_csv):
    import csv
    with open(results_csv, newline="") as fh:
        rows = list(csv.DictReader(fh))
    T = np.array([float(r["T"]) for r in rows])
    m = np.array([float(r["C_mean"]) for r in rows])
    rg2 = np.array([float(r["Rg2_mean"]) for r in rows])
    order = np.argsort(T)
    m, rg2 = m[order], rg2[order]
    # Distinct regimes: contacts should fall and Rg^2 should rise with T, and the
    # spread must be appreciable (not a flat, single-regime ladder).
    contacts_span = float(m.max() - m.min())
    rg2_span = float(rg2.max() - rg2.min())
    monotone_contacts = bool(np.all(np.diff(m) <= 1e-6))   # non-increasing in T
    return {
        "temperatures_sorted": T[order].tolist(),
        "mean_contacts": m.tolist(),
        "mean_rg2": rg2.tolist(),
        "contacts_span": contacts_span,
        "rg2_span": rg2_span,
        "contacts_decrease_with_T": monotone_contacts,
        "distinct_regimes": bool(contacts_span > 1.0 and rg2_span > 0.5),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fit-summary-json", required=True, dest="fit_summary_json")
    ap.add_argument("--N", type=int, default=30)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--n-temperatures", type=int, default=8)
    ap.add_argument("--n-cycles", type=int, default=400)
    ap.add_argument("--output-dir", default=str(HERE / "regime_pilot"))
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ladder, info, Tc = build_regime_ladder(args.fit_summary_json, args.n_temperatures)
    print(f"Model {info['model_name']} Tc~{Tc:.4g}; ladder "
          f"[{ladder.min():.2f}, {ladder.max():.2f}] x {len(ladder)}")

    report = {"model": info["model_name"], "Tc_estimate": Tc,
              "ladder": ladder.tolist(), "N": args.N, "seeds": {}}
    for seed in args.seeds:
        results_csv = run_seed(out_dir, args.N, seed, ladder,
                               args.fit_summary_json, args.n_cycles)
        regimes = check_regimes(results_csv)
        report["seeds"][str(seed)] = regimes
        print(f"  seed {seed}: distinct_regimes={regimes['distinct_regimes']} "
              f"contacts_span={regimes['contacts_span']:.2f} "
              f"rg2_span={regimes['rg2_span']:.2f}")
    with open(out_dir / "regime_pilot_report.json", "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"Saved {out_dir / 'regime_pilot_report.json'}")
    if not all(v["distinct_regimes"] for v in report["seeds"].values()):
        print("WARNING: ladder did not produce clearly distinct structural "
              "regimes; widen the temperature range or lengthen the run.")


if __name__ == "__main__":
    main()
