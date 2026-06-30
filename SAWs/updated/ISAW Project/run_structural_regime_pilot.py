#!/usr/bin/env python3
"""PNIPAM-aware structural CALIBRATION pilot (LCST-correct; K-based).

Builds a temperature ladder from a FITTED contact model, evaluates K(T) = -b(T)
densely over the fitted interval, verifies whether the model is consistent with
PNIPAM LCST behavior (collapse as T INCREASES because K increases with T), runs
short structural REMD with full saving, extracts validated features, and assesses
collapse trends against BOTH T and K using endpoint effect sizes, rank
correlation, slopes, and seed bootstrap.  It then recommends production settings
from the slowest autocorrelation time WITHOUT launching production.

PNIPAM convention
-----------------
P(C|T) ∝ exp[K(T) m(C)] with K(T) = -b(T); higher K favors more contacts.  For
the fitted LCST branch we EXPECT, from low-K to high-K: <m> up, <Rg^2> down,
global/long contacts up, connected structure up.  These are verified, never
hardcoded.

PowerShell example:
    python .\\run_structural_regime_pilot.py `
        --fit-summary-json .\\fits_30mer_contact_only\\hs\\fit_summary.json `
        --N 30 --seeds 1 2 --n-temperatures 8 --n-cycles 600 `
        --output-dir .\\calibration_N30

Run explicitly (longer than the smoke test); not part of the unit-test suite.
This script does NOT launch the eight-seed production campaign.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import remd_uniform_chain_2_new as remd
import isaw_schema as sch

PY = sys.executable
REMD = str(HERE / "remd_uniform_chain_2_new.py")
EXTRACT = str(HERE / "extract_contact_motif_features.py")

CALIBRATION_REPORT_SCHEMA_VERSION = sch.CALIBRATION_REPORT_SCHEMA_VERSION
TARGET_EFF_PER_REGIME = 5000   # computational-plan target


# ---------------------------------------------------------------------------
# Fitted model + K(T) analysis (D1, D2)
# ---------------------------------------------------------------------------

def load_fitted_model(summary_path, cli_tmin, cli_tmax):
    info = remd.load_fit_summary_json(summary_path)
    with open(summary_path) as fh:
        raw = json.load(fh)
    # Fitted temperature interval: explicit field, else temps_all, else require CLI.
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
    return info, rng


def analyze_K(info, T_lo, T_hi, n_dense=400):
    Ts = np.linspace(T_lo, T_hi, n_dense)
    b = np.array([remd.reduced_bias(info["model_name"], info["params"],
                                    float(T), info["Tref"], info["Tscale"])
                  for T in Ts])
    K = -b
    dKdT = np.gradient(K, Ts)
    inc = np.all(np.diff(K) > 0)
    dec = np.all(np.diff(K) < 0)
    monotonic = bool(inc or dec)
    # b(T)=0 crossing (K=0), if any.
    zero_cross = None
    sgn = np.sign(b)
    idx = np.where(np.diff(sgn) != 0)[0]
    if idx.size:
        i = int(idx[0])
        zero_cross = float(np.interp(0.0, [b[i + 1], b[i]], [Ts[i + 1], Ts[i]])
                           if b[i + 1] != b[i] else Ts[i])
    return {"T": Ts, "b": b, "K": K, "dKdT": dKdT,
            "K_increases_with_T": bool(inc), "monotonic": monotonic,
            "zero_crossing_T": zero_cross}


def build_K_ladder(kinfo, n_points, branch="auto"):
    """Choose temperatures ~uniform in K, with extra points where |dK/dT| is
    largest; returns sorted temperatures and the chosen-branch note."""
    Ts, K, dKdT = kinfo["T"], kinfo["K"], kinfo["dKdT"]
    note = "full_interval"
    if not kinfo["monotonic"]:
        # Identify the longest monotonic branch; require the user to confirm via
        # --branch low|high if ambiguous.
        dk = np.diff(K)
        sign = np.sign(dk)
        # longest run of equal sign
        best_s, best_e, s = 0, 0, 0
        for e in range(1, len(sign)):
            if sign[e] != sign[s]:
                if (e - s) > (best_e - best_s):
                    best_s, best_e = s, e
                s = e
        if (len(sign) - s) > (best_e - best_s):
            best_s, best_e = s, len(sign)
        Ts = Ts[best_s:best_e + 1]
        K = K[best_s:best_e + 1]
        note = f"nonmonotonic_K; using_longest_monotonic_branch[{branch}]"
    # uniform-in-K target levels
    K_lo, K_hi = float(K.min()), float(K.max())
    base = np.linspace(K_lo, K_hi, max(3, n_points - 2))
    # extra points where |dK/dT| largest (most structural sensitivity)
    extra_T = Ts[np.argsort(-np.abs(np.gradient(K, Ts)))[:2]]
    # invert K->T by interpolation (K monotone on this branch)
    order = np.argsort(K)
    Ks, Tsorted = K[order], Ts[order]
    chosen_T = np.interp(base, Ks, Tsorted)
    chosen_T = np.unique(np.concatenate([chosen_T, extra_T]))
    chosen_T.sort()
    if chosen_T.size > n_points:
        # thin to n_points keeping endpoints
        idx = np.linspace(0, chosen_T.size - 1, n_points).round().astype(int)
        chosen_T = np.unique(chosen_T[idx])
    return chosen_T, note


# ---------------------------------------------------------------------------
# Run + analysis (D3, D4, D5)
# ---------------------------------------------------------------------------

def run_seed(out_dir, N, seed, ladder, summary_path, n_cycles, steps_per_swap):
    prefix = out_dir / f"calib_N{N}_s{seed}"
    temps = ",".join(f"{t:.4f}" for t in ladder)
    cmd = [PY, REMD, "--N", str(N), "--temps", temps,
           "--fit-summary-json", summary_path,
           "--steps-per-swap", str(steps_per_swap), "--n-cycles", str(n_cycles),
           "--n-workers", "1", "--seed", str(seed), "--burnin-frac", "0.5",
           "--structural-observables", "--structural-stride", "5",
           "--diagnostics", "--diagnostic-trajectories",
           "--save-configurations", "--snapshot-stride", "5",
           "--overwrite-configurations", "--no-plots",
           "--out-prefix", str(prefix)]
    print("RUN:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(HERE))
    cfg = f"{prefix}_configurations.h5"
    feat = out_dir / f"calib_N{N}_s{seed}_features.h5"
    subprocess.run([PY, EXTRACT, "--input", cfg, "--output", str(feat),
                    "--validate", "--overwrite"], check=True, cwd=str(HERE))
    return prefix


def _spearman(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if x.size < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    rx = rx - rx.mean(); ry = ry - ry.mean()
    denom = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    return float((rx * ry).sum() / denom) if denom > 0 else float("nan")


def _read_results(prefix):
    rows = list(csv.DictReader(open(f"{prefix}_results.csv", newline="")))
    return rows


def _read_diag(prefix):
    with open(f"{prefix}_diagnostics.json") as fh:
        return json.load(fh)


def _K_of(info, T):
    return -remd.reduced_bias(info["model_name"], info["params"], float(T),
                              info["Tref"], info["Tscale"])


def analyze_trends(prefixes, info):
    """Per-seed trend analysis vs T and K with endpoint effect sizes, Spearman,
    slope vs K, and across-seed bootstrap of endpoint differences."""
    obs = ["C_mean", "Rg2_mean", "Ree2_mean", "m_long_mean",
           "m_global_scaled_mean", "Smax_mean",
           "largest_component_fraction_mean"]
    per_seed = []
    endpoint_dm = []   # high-K minus low-K, mean contacts
    endpoint_drg2 = []
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
                "spearman_vs_K": _spearman(K[order], y),
                "slope_vs_K": float(np.polyfit(K[order], y, 1)[0]) if y.size >= 2 else float("nan"),
                "endpoint_high_minus_low": float(y[-1] - y[0]),
            }
        per_seed.append(rec)
        endpoint_dm.append(rec["C_mean"]["endpoint_high_minus_low"])
        endpoint_drg2.append(rec["Rg2_mean"]["endpoint_high_minus_low"])

    def _boot(vals, n=2000, seed=0):
        vals = np.asarray(vals, float)
        if vals.size == 0:
            return (float("nan"), float("nan"), float("nan"))
        rng = np.random.RandomState(seed)
        bs = vals[rng.randint(0, vals.size, size=(n, vals.size))].mean(axis=1)
        return (float(vals.mean()), float(np.percentile(bs, 2.5)),
                float(np.percentile(bs, 97.5)))

    dm_mean, dm_lo, dm_hi = _boot(endpoint_dm)
    drg2_mean, drg2_lo, drg2_hi = _boot(endpoint_drg2)
    # PNIPAM expected signs: Δ<m> > 0 and Δ<Rg2> < 0 (high-K minus low-K).
    signs_ok = (dm_mean > 0) and (drg2_mean < 0)
    return {
        "per_seed": per_seed,
        "endpoint_delta_m_highK_minus_lowK": {
            "mean": dm_mean, "ci95": [dm_lo, dm_hi]},
        "endpoint_delta_Rg2_highK_minus_lowK": {
            "mean": drg2_mean, "ci95": [drg2_lo, drg2_hi]},
        "pnipam_signs_consistent": bool(signs_ok),
        "note": ("expected (LCST): Δ<m> > 0 and Δ<Rg2> < 0 from low-K to "
                 "high-K; sign conflict => NOT consistent."),
    }


def _bhattacharyya(p, q):
    p = np.asarray(p, float); q = np.asarray(q, float)
    p = np.where(np.isfinite(p), p, 0.0); q = np.where(np.isfinite(q), q, 0.0)
    sp, sq = p.sum(), q.sum()
    if sp <= 0 or sq <= 0:
        return 0.0
    return float(np.sqrt((p / sp) * (q / sq)).sum())


def analyze_mixing(prefixes, info):
    """tau_int_cycles / ESS / swap / coverage / round-trips / state-changing
    acceptance / contact support / high-contact tail / adjacent overlap."""
    seed_reports = []
    slowest_tau_cycles = 0.0
    min_ess = float("inf")
    for prefix in prefixes:
        diag = _read_diag(prefix)
        lanes = diag["lane_convergence"]
        npz = np.load(f"{prefix}_distributions.npz")
        Pc = npz["Pc"]; c_vals = npz["c_vals"]
        per_lane = []
        for i, lane in enumerate(lanes):
            row = {"temperature": lane["temperature"]}
            for o in ("contacts", "rg2", "m_global_scaled", "smax"):
                if o in lane:
                    row[f"tau_int_cycles_{o}"] = lane[o].get("tau_int_cycles")
                    row[f"ess_{o}"] = lane[o].get("ess")
                    tc = lane[o].get("tau_int_cycles")
                    if tc and np.isfinite(tc):
                        slowest_tau_cycles = max(slowest_tau_cycles, float(tc))
                    e = lane[o].get("ess")
                    if e and np.isfinite(e):
                        min_ess = min(min_ess, float(e))
            # contact support + high-contact tail
            pc = Pc[i] if i < Pc.shape[0] else np.array([])
            support = int(np.count_nonzero(np.nan_to_num(pc) > 0))
            tail = float(np.nan_to_num(pc)[c_vals >= np.nanpercentile(
                c_vals, 90)].sum()) if pc.size else 0.0
            row["contact_support_bins"] = support
            row["high_contact_tail_prob"] = tail
            per_lane.append(row)
        # adjacent-lane histogram overlap (Bhattacharyya) matrix
        nT = Pc.shape[0]
        overlap = np.zeros((nT, nT))
        for a in range(nT):
            for b in range(nT):
                overlap[a, b] = _bhattacharyya(Pc[a], Pc[b])
        adj = [float(overlap[a, a + 1]) for a in range(nT - 1)]
        seed_reports.append({
            "lanes": per_lane,
            "swap_rates": diag["swap_rates"],
            "min_temp_coverage": diag["summary"].get("min_temp_coverage"),
            "total_round_trips_low": diag["summary"].get("total_round_trips_low"),
            "adjacent_overlap_bhattacharyya": adj,
            "overlap_matrix": overlap.tolist(),
            "state_changing_acceptance_rates": [
                float(x) for x in json.load(open(f"{prefix}_run_summary.json"))
                .get("state_changing_acceptance_rates", [])],
        })
    return {"seeds": seed_reports,
            "slowest_tau_int_cycles": slowest_tau_cycles,
            "min_ess_observed": (None if min_ess == float("inf") else min_ess)}


# ---------------------------------------------------------------------------
# Production recommendation (D6)
# ---------------------------------------------------------------------------

def recommend_production(mixing, n_temperatures, n_seeds, structural_stride,
                         snapshot_stride):
    tau = mixing["slowest_tau_int_cycles"] or 1.0
    # cycles between effectively independent structural samples ~ 2*tau.
    cycles_per_indep = max(1.0, 2.0 * tau)
    # We need TARGET per regime (~3 regimes) per N; spread across seeds.
    target_total = TARGET_EFF_PER_REGIME * 3
    indep_per_seed = target_total / max(1, n_seeds)
    production_cycles = int(np.ceil(indep_per_seed * cycles_per_indep
                                    / max(1, n_temperatures)))
    burnin_cycles = int(np.ceil(10 * tau))
    snapshots = int(np.ceil(production_cycles / max(1, snapshot_stride)))
    return {
        "rationale": ("independent structural samples are estimated from "
                      "2*tau_int_cycles (NOT raw snapshot counts)"),
        "slowest_tau_int_cycles": float(tau),
        "cycles_per_independent_sample": float(cycles_per_indep),
        "target_effective_per_regime": TARGET_EFF_PER_REGIME,
        "recommended_steps_per_swap": 60,
        "recommended_n_temperatures": int(max(n_temperatures, 16)),
        "recommended_structural_stride": int(structural_stride),
        "recommended_snapshot_stride": int(snapshot_stride),
        "recommended_burnin_cycles": burnin_cycles,
        "recommended_production_cycles": production_cycles,
        "expected_raw_snapshots_per_seed": snapshots,
        "expected_effective_independent_per_seed": int(
            production_cycles * n_temperatures / cycles_per_indep),
        "storage_estimate_note": ("coords int16 gzip ~ "
                                  "n_temperatures*n_beads*3*2 bytes/snapshot"),
        "target_feasible_estimate": bool(
            production_cycles * n_seeds * n_temperatures / cycles_per_indep
            >= target_total),
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fit-summary-json", required=True, dest="fit_summary_json")
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--n-temperatures", type=int, default=8)
    ap.add_argument("--n-cycles", type=int, default=400)
    ap.add_argument("--steps-per-swap", type=int, default=60)
    ap.add_argument("--t-min", type=float, default=None)
    ap.add_argument("--t-max", type=float, default=None)
    ap.add_argument("--branch", choices=["auto", "low", "high"], default="auto")
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else (
        HERE / f"calibration_N{args.N}")
    out_dir.mkdir(parents=True, exist_ok=True)

    info, (T_lo, T_hi) = load_fitted_model(args.fit_summary_json,
                                           args.t_min, args.t_max)
    kinfo = analyze_K(info, T_lo, T_hi)
    print(f"Model {info['model_name']} on [{T_lo:.2f}, {T_hi:.2f}]: "
          f"K_increases_with_T={kinfo['K_increases_with_T']} "
          f"monotonic={kinfo['monotonic']} zero_crossing={kinfo['zero_crossing_T']}")

    ladder, branch_note = build_K_ladder(kinfo, args.n_temperatures, args.branch)
    if ladder.size < 2:
        raise SystemExit("could not build a >=2-point K ladder")
    K_ladder = [float(_K_of(info, t)) for t in ladder]
    print(f"Ladder ({ladder.size}): T={[round(float(t),1) for t in ladder]}")
    print(f"K ladder: {[round(k,4) for k in K_ladder]}")

    prefixes = [run_seed(out_dir, args.N, s, ladder, args.fit_summary_json,
                         args.n_cycles, args.steps_per_swap)
                for s in args.seeds]
    prefixes = [str(p) for p in prefixes]

    trends = analyze_trends(prefixes, info)
    mixing = analyze_mixing(prefixes, info)
    recommendation = recommend_production(
        mixing, ladder.size, len(args.seeds), 5, 5)

    pnipam_consistent = bool(kinfo["K_increases_with_T"]
                             and trends["pnipam_signs_consistent"])
    report = {
        "calibration_report_schema_version": CALIBRATION_REPORT_SCHEMA_VERSION,
        "model": info["model_name"], "params": info["params"],
        "N": args.N, "seeds": args.seeds,
        "fitted_temperature_interval": [T_lo, T_hi],
        "branch_note": branch_note,
        "K_analysis": {
            "K_increases_with_T": kinfo["K_increases_with_T"],
            "monotonic": kinfo["monotonic"],
            "zero_crossing_T": kinfo["zero_crossing_T"],
            "T_grid_endpoints": [float(kinfo["T"][0]), float(kinfo["T"][-1])],
            "K_grid_endpoints": [float(kinfo["K"][0]), float(kinfo["K"][-1])],
        },
        "temperature_ladder": [float(t) for t in ladder],
        "K_ladder": K_ladder,
        "trends": trends,
        "mixing": mixing,
        "recommended_production": recommendation,
        "pnipam_lcst_assessment": {
            "K_increases_with_T": kinfo["K_increases_with_T"],
            "m_increases_with_K": trends["endpoint_delta_m_highK_minus_lowK"]["mean"] > 0,
            "Rg2_decreases_with_K": trends["endpoint_delta_Rg2_highK_minus_lowK"]["mean"] < 0,
            "consistent_with_PNIPAM_LCST": pnipam_consistent,
            "ladder_spans_distinct_states": bool(
                trends["pnipam_signs_consistent"]),
            "sampling_mixing_adequate": bool(
                (mixing["min_ess_observed"] or 0) >= 20),
        },
    }
    (out_dir / "calibration_report.json").write_text(
        json.dumps(sch.json_safe(report), indent=2), encoding="utf-8")
    (out_dir / "recommended_production_config.json").write_text(
        json.dumps(sch.json_safe(recommendation), indent=2), encoding="utf-8")
    _write_report_md(out_dir / "calibration_report.md", report)
    print(f"\nPNIPAM LCST consistent: {pnipam_consistent}")
    print(f"Reports in {out_dir}")
    if not kinfo["K_increases_with_T"]:
        print("WARNING: fitted K(T) does NOT increase with T over this interval "
              "-> NOT a standard PNIPAM LCST direction. Check the model/interval.")
    if not trends["pnipam_signs_consistent"]:
        print("WARNING: structural endpoint signs conflict with PNIPAM "
              "expectation (Δ<m> and Δ<Rg2>); calibration NOT successful.")


def _write_report_md(path, report):
    a = report["pnipam_lcst_assessment"]
    lines = [
        f"# PNIPAM calibration report — N={report['N']} ({report['model']})",
        "",
        f"- fitted interval: {report['fitted_temperature_interval']}",
        f"- K increases with T: **{a['K_increases_with_T']}**",
        f"- <m> increases with K: **{a['m_increases_with_K']}**",
        f"- <Rg^2> decreases with K: **{a['Rg2_decreases_with_K']}**",
        f"- consistent with PNIPAM LCST: **{a['consistent_with_PNIPAM_LCST']}**",
        f"- ladder spans distinct states: **{a['ladder_spans_distinct_states']}**",
        f"- sampling/mixing adequate: **{a['sampling_mixing_adequate']}**",
        "",
        "## Endpoint effect sizes (high-K minus low-K)",
        f"- Δ<m> = {report['trends']['endpoint_delta_m_highK_minus_lowK']}",
        f"- Δ<Rg^2> = {report['trends']['endpoint_delta_Rg2_highK_minus_lowK']}",
        "",
        "## Recommended production (NOT launched)",
        f"- slowest tau_int_cycles: {report['recommended_production']['slowest_tau_int_cycles']}",
        f"- recommended production cycles: {report['recommended_production']['recommended_production_cycles']}",
        f"- target feasible: {report['recommended_production']['target_feasible_estimate']}",
        "",
        "_Independent-sample counts are from 2*tau_int_cycles, not raw snapshots._",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
