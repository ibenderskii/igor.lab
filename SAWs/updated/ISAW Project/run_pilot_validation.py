#!/usr/bin/env python3
"""End-to-end pilot driver + invariant validation for N=30 and N=44.

Runs short structural REMD pilots (two independent seeds per N), extracts
offline features, then re-verifies every physical/structural invariant directly
from each coordinate HDF5 file and writes a per-run validation JSON report.  All
discrepancy counts must be zero.
"""
from __future__ import annotations

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

OUT = HERE / "pilot_outputs"
PY = sys.executable
REMD = str(HERE / "remd_uniform_chain_2_new.py")
EXTRACT = str(HERE / "extract_contact_motif_features.py")

HS = ("--dh", "378.96", "--ds", "1.39686")
LADDER = "300,320,340,360"


def run_one(N, seed):
    prefix = OUT / f"n{N}_s{seed}"
    cfg = f"{prefix}_configurations.h5"
    cmd = [PY, REMD, "--N", str(N), "--temps", LADDER,
           "--steps-per-swap", "40", "--n-cycles", "120", "--n-workers", "1",
           "--seed", str(seed), *HS,
           "--diagnostics", "--diagnostic-trajectories",
           "--structural-observables", "--structural-stride", "5",
           "--save-configurations", "--snapshot-stride", "5",
           "--no-plots", "--out-prefix", str(prefix)]
    print("RUN:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(HERE))
    feat = OUT / f"n{N}_s{seed}_features.h5"
    subprocess.run([PY, EXTRACT, "--input", cfg, "--output", str(feat),
                    "--validate", "--overwrite"], check=True, cwd=str(HERE))
    return str(cfg), str(feat)


def validate_file(cfg_path):
    disc = {k: 0 for k in (
        "stored_m_vs_recomputed", "mr_sum_vs_m", "separations_odd",
        "pair_counts_vs_binomial", "graph_edges_vs_m", "rg2_match",
        "ree2_match", "walker_permutation", "committed_rows_vs_extracted")}
    with h5py.File(cfg_path, "r") as f:
        s = f["snapshots"]
        n_committed = cio.committed_rows(s)
        nb = int(f["metadata"].attrs["n_beads"])
        coords = s["coordinates"]
        nT = coords.shape[1]
        extracted = 0
        for si in range(n_committed):
            w = np.asarray(s["walker_id"][si], dtype=np.int64)
            if not np.array_equal(np.sort(w), np.arange(nT)):
                disc["walker_permutation"] += 1
            for k in range(nT):
                c = coords[si, k].astype(np.int64)
                cp, seps = ico.build_contact_map(c)
                m = cp.shape[0]
                if m != int(s["contacts"][si, k]):
                    disc["stored_m_vs_recomputed"] += 1
                m_r = ico.contact_separation_counts(cp, nb)
                if int(m_r.sum()) != m:
                    disc["mr_sum_vs_m"] += 1
                if m and not np.all((seps % 2) == 1):
                    disc["separations_odd"] += 1
                motifs = ico.count_pair_motifs(cp)
                if motifs["pair_total"] != m * (m - 1) // 2:
                    disc["pair_counts_vs_binomial"] += 1
                g = ico.contact_graph_summary(cp, nb)
                if g["contact_graph_cycle_rank"] != (
                        m - g["contact_vertices"] + g["contact_graph_components"]):
                    disc["graph_edges_vs_m"] += 1
                rg2 = ico.radius_of_gyration_squared(c)
                ree2 = ico.end_to_end_distance_squared(c)
                if abs(rg2 - float(s["rg2_lattice"][si, k])) > 1e-6:
                    disc["rg2_match"] += 1
                if abs(ree2 - float(s["ree2_lattice"][si, k])) > 1e-6:
                    disc["ree2_match"] += 1
                extracted += 1
        info = ext.extract(cfg_path, cfg_path + ".features.tmp.h5",
                           validate=True, overwrite=True)
        if info["extracted_snapshot_rows"] != n_committed:
            disc["committed_rows_vs_extracted"] += 1
    os.remove(cfg_path + ".features.tmp.h5")
    if os.path.exists(cfg_path + ".features.tmp.h5.manifest.json"):
        os.remove(cfg_path + ".features.tmp.h5.manifest.json")
    return {"committed_rows": int(n_committed), "n_temperatures": int(nT),
            "n_beads": int(nb), "discrepancy_counts": disc,
            "all_zero": all(v == 0 for v in disc.values())}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    overall = {}
    for N in (30, 44):
        for seed in (1, 2):
            cfg, feat = run_one(N, seed)
            report = validate_file(cfg)
            rpath = OUT / f"n{N}_s{seed}_validation.json"
            with open(rpath, "w") as fh:
                json.dump(report, fh, indent=2)
            overall[f"n{N}_s{seed}"] = report["all_zero"]
            print(f"  validated {cfg}: all_zero={report['all_zero']} "
                  f"({report['committed_rows']} snapshots)")
    print("\nSUMMARY:", json.dumps(overall, indent=2))
    assert all(overall.values()), "some pilot validations had nonzero discrepancies"
    print("ALL PILOT VALIDATIONS PASSED (zero discrepancies)")


if __name__ == "__main__":
    main()
