#!/usr/bin/env python3
"""SHORT INTEGRITY SMOKE TEST for the ISAW structural pipeline (NOT a regime pilot).

This driver runs short structural REMD on a small temperature ladder, extracts
offline features, and re-verifies every physical/structural invariant directly
from each coordinate HDF5 file AND from the generated feature HDF5 file.  It is a
FILE-INTEGRITY / END-TO-END smoke test: the ladder is arbitrary and short, so it
does NOT claim to span swollen / crossover / collapsed structural regimes.  For a
scientifically chosen regime-spanning pilot use ``run_structural_regime_pilot.py``.

All discrepancy counts must be zero.

PowerShell examples:
    python .\\run_pilot_validation.py --overwrite
    python .\\run_pilot_validation.py --output-dir .\\smoke_run --seeds 1 2 `
        --chain-lengths 30 44 --overwrite
    python .\\run_pilot_validation.py --keep-existing   # reuse existing files
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import h5py

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import isaw_contact_observables as ico
import isaw_config_io as cio
import extract_contact_motif_features as ext

PY = sys.executable
REMD = str(HERE / "remd_uniform_chain_2_new.py")
EXTRACT = str(HERE / "extract_contact_motif_features.py")

HS = ("--dh", "378.96", "--ds", "1.39686")
LADDER = "300,320,340,360"


def run_one(out_dir, N, seed, *, overwrite, keep_existing):
    prefix = out_dir / f"n{N}_s{seed}"
    cfg = f"{prefix}_configurations.h5"
    feat = out_dir / f"n{N}_s{seed}_features.h5"
    if keep_existing and Path(cfg).exists() and Path(feat).exists():
        print(f"REUSE: {cfg}")
        return str(cfg), str(feat)
    cmd = [PY, REMD, "--N", str(N), "--temps", LADDER,
           "--steps-per-swap", "40", "--n-cycles", "120", "--n-workers", "1",
           "--seed", str(seed), *HS,
           "--diagnostics", "--diagnostic-trajectories",
           "--structural-observables", "--structural-stride", "5",
           "--save-configurations", "--snapshot-stride", "5",
           "--no-plots", "--out-prefix", str(prefix)]
    if overwrite:
        cmd.append("--overwrite-configurations")
    print("RUN:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(HERE))
    ecmd = [PY, EXTRACT, "--input", cfg, "--output", str(feat), "--validate"]
    if overwrite:
        ecmd.append("--overwrite")
    subprocess.run(ecmd, check=True, cwd=str(HERE))
    return str(cfg), str(feat)


def validate_config_file(cfg_path):
    disc = {k: 0 for k in (
        "stored_m_vs_recomputed", "mr_sum_vs_m", "separations_odd",
        "pair_counts_vs_binomial", "graph_edges_vs_m", "rg2_match",
        "ree2_match", "walker_permutation")}
    with h5py.File(cfg_path, "r") as f:
        s = f["snapshots"]
        n_committed = cio.committed_rows(s)
        nb = int(f["metadata"].attrs["n_beads"])
        coords = s["coordinates"]
        nT = coords.shape[1]
        for si in range(n_committed):
            w = np.asarray(s["walker_id"][si], dtype=np.int64)
            if not np.array_equal(np.sort(w), np.arange(nT)):
                disc["walker_permutation"] += 1
            for k in range(nT):
                # Pass the raw HDF5 slice to the strict validator; no pre-cast,
                # so fractional/NaN/out-of-range coordinates cannot be hidden by
                # truncation.
                c = coords[si, k]
                cp, seps = ico.build_contact_map(c)
                m = cp.shape[0]
                if m != int(s["contacts"][si, k]):
                    disc["stored_m_vs_recomputed"] += 1
                m_r = ico.contact_separation_counts(cp, nb)
                if int(m_r.sum()) != m:
                    disc["mr_sum_vs_m"] += 1
                if m and not np.all((seps % 2) == 1):
                    disc["separations_odd"] += 1
                if ico.count_pair_motifs(cp)["pair_total"] != m * (m - 1) // 2:
                    disc["pair_counts_vs_binomial"] += 1
                g = ico.contact_graph_summary(cp, nb)
                if (g["contact_graph_edges"] != m or g["sum_component_edges"] != m
                        or g["contact_graph_cycle_rank"] != (
                            m - g["contact_vertices"] + g["contact_graph_components"])):
                    disc["graph_edges_vs_m"] += 1
                if abs(ico.radius_of_gyration_squared(c)
                       - float(s["rg2_lattice"][si, k])) > 1e-6:
                    disc["rg2_match"] += 1
                if abs(ico.end_to_end_distance_squared(c)
                       - float(s["ree2_lattice"][si, k])) > 1e-6:
                    disc["ree2_match"] += 1
    return disc, int(n_committed), int(nT), int(nb)


def validate_feature_file(feat_path, *, committed_rows, n_temperatures, n_beads):
    """Open and validate the generated feature HDF5 file against the input."""
    disc = {k: 0 for k in (
        "row_count", "scalar_dataset_lengths", "index_dataset_lengths",
        "m_r_shape", "m_r_sum_vs_m", "row_ordering", "status_complete",
        "committed_rows_vs_row_count", "manifest_row_count",
        "manifest_discrepancies_zero")}
    expected = committed_rows * n_temperatures
    with h5py.File(feat_path, "r") as f:
        if str(f.attrs.get("status")) != "complete":
            disc["status_complete"] += 1
        feats = f["features"]
        committed = int(feats.attrs.get("committed_feature_rows", -1))
        m = f["features/scalars/m"][()]
        n_rows = m.shape[0]
        if n_rows != expected:
            disc["row_count"] += 1
        if committed != n_rows:
            disc["committed_rows_vs_row_count"] += 1
        for name in f["features/scalars"]:
            if f["features/scalars/" + name].shape[0] != n_rows:
                disc["scalar_dataset_lengths"] += 1
        for name in f["features/sample_index"]:
            if f["features/sample_index/" + name].shape[0] != n_rows:
                disc["index_dataset_lengths"] += 1
        m_r = f["features/m_r"][()]
        if m_r.shape != (n_rows, n_beads):
            disc["m_r_shape"] += 1
        if not np.array_equal(m_r.sum(axis=1).astype(np.int64), m.astype(np.int64)):
            disc["m_r_sum_vs_m"] += 1
        snap = f["features/sample_index/snapshot_index"][()]
        ti = f["features/sample_index/temperature_index"][()]
        order = list(zip(snap.tolist(), ti.tolist()))
        if order != sorted(order):
            disc["row_ordering"] += 1
        man = json.loads(f["metadata"].attrs["manifest"])
        if int(man.get("row_count", -1)) != n_rows:
            disc["manifest_row_count"] += 1
        if not all(v == 0 for v in man.get("validation_discrepancy_counts", {1: 1}).values()):
            disc["manifest_discrepancies_zero"] += 1
    return disc, expected


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", default=None,
                    help="output directory (default: timestamped smoke_pilot_<ts>)")
    ap.add_argument("--overwrite", action="store_true",
                    help="overwrite existing configuration/feature outputs")
    ap.add_argument("--keep-existing", action="store_true",
                    help="reuse existing config+feature files instead of rerunning")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--chain-lengths", type=int, nargs="+", default=[30, 44])
    args = ap.parse_args()

    if args.overwrite and args.keep_existing:
        ap.error("--overwrite and --keep-existing are mutually exclusive")

    if args.output_dir is not None:
        out_dir = Path(args.output_dir)
    elif args.overwrite or args.keep_existing:
        out_dir = HERE / "pilot_outputs"
    else:
        # Safe default: a fresh timestamped directory (no destructive overwrite).
        ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = HERE / f"smoke_pilot_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {out_dir}")

    overall = {}
    for N in args.chain_lengths:
        for seed in args.seeds:
            cfg, feat = run_one(out_dir, N, seed,
                                overwrite=args.overwrite,
                                keep_existing=args.keep_existing)
            cfg_disc, committed, nT, nb = validate_config_file(cfg)
            feat_disc, expected_rows = validate_feature_file(
                feat, committed_rows=committed, n_temperatures=nT, n_beads=nb)
            all_zero = (all(v == 0 for v in cfg_disc.values())
                        and all(v == 0 for v in feat_disc.values()))
            report = {
                "test_kind": "integrity_smoke_test",
                "regime_spanning": False,
                "chain_length_N": N, "seed": seed,
                "snapshot_count": committed, "n_temperatures": nT,
                "feature_row_count": expected_rows,
                "config_discrepancy_counts": cfg_disc,
                "feature_discrepancy_counts": feat_disc,
                "all_zero": all_zero,
            }
            with open(out_dir / f"n{N}_s{seed}_validation.json", "w") as fh:
                json.dump(report, fh, indent=2)
            overall[f"n{N}_s{seed}"] = all_zero
            print(f"  validated N={N} seed={seed}: all_zero={all_zero} "
                  f"({committed} snapshots, {expected_rows} feature rows)")
    print("\nSUMMARY:", json.dumps(overall, indent=2))
    if not all(overall.values()):
        raise SystemExit("some smoke validations had nonzero discrepancies")
    print("ALL SMOKE VALIDATIONS PASSED (zero discrepancies)")


if __name__ == "__main__":
    main()
