#!/usr/bin/env python
"""Run the seeded golden-output cases against one code tree.

    python run_cases.py --code-root <.../SAWs/updated> --out <dir> \
        [--data-root <.../SAWs/updated>] [--inputs <dir>] \
        [--cases PATTERN[,PATTERN...]] [--workers 1,2] [--list]

``--code-root`` selects WHICH implementation runs (e.g. a git worktree of the
baseline commit, or the candidate working tree).  ``--data-root`` and
``--inputs`` select the input data and must be the same for the two runs being
compared, so only the code differs.  Each case runs once per worker count into
``<out>/<case>_w<W>/``; stdout/stderr go to ``<out>/_logs/`` and wall times to
``<out>/_timing.json`` (underscore-prefixed paths are skipped by
compare_outputs.py).

Cases are deliberately short (seconds to ~2 minutes each) but switch on every
output family: bootstrap + prediction bands, uncertainty diagnostics, split
sensitivity and the rg-weight scan for the fitter; diagnostics, trajectories,
structural observables and configuration snapshots for the samplers.

BLAS/OpenMP threads are pinned to 1 (as midway3_common.sh does) so results do
not depend on threading.  Python >= 3.8.
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List

HERE = Path(__file__).resolve().parent
RG_SCALE = "0.4632050331259017"

# ---------------------------------------------------------------------------
# Case definitions.  Each builder gets a context dict and returns an argv list
# (without the interpreter).  ctx: code, data, inputs, case_dir, workers.
# ---------------------------------------------------------------------------

def _fitter(ctx) -> List[str]:
    return [str(ctx["code"] / "auto" / "fit_lattice_contact_model_parallel.py")]


def _real44(ctx) -> List[str]:
    agg = ctx["data"] / "Aggregation Project"
    return ["--remd", str(agg / "remd_distributions_44mer.npz"),
            "--baseline", str(agg / "RnBaseline44.npz"), "--contact_offset", "43",
            "--rg-scale", RG_SCALE]


def _saw10(ctx) -> List[str]:
    return ["--remd", str(ctx["inputs"] / "saw_N10_target.npz"),
            "--baseline", str(ctx["inputs"] / "saw_N10_baseline.npz"),
            "--contact_offset", "0", "--rg-scale", "1.0"]


def _fit_common(ctx, model, loss, rg, boot, extra=(), restarts=4) -> List[str]:
    argv = _fitter(ctx) + ["--model", model, "--loss", loss, "--outdir", str(ctx["case_dir"]),
                           "--n_restarts", str(restarts), "--seed", "123", "--no-plots",
                           "--bootstrap", str(boot), "--bootstrap-seed", "7",
                           "--bootstrap-workers", str(ctx["workers"])]
    argv += (["--fit-rg", "--rg-weight", "0.5"] if rg else ["--no-fit-rg"])
    return argv + list(extra)


FULL_SUPPLEMENT = ["--bootstrap-save-prediction-bands", "--uncertainty-diagnostics",
                   "--split-sensitivity", "--split-schemes", "blocked_low,kfold",
                   "--split-kfold-k", "3", "--split-seed", "5"]


def fit_hs(ctx):
    return _fit_common(ctx, "hs", "js", True, 6, _real44(ctx) + FULL_SUPPLEMENT
                       + ["--rg-weight-grid", "0,0.5,1"])


def fit_sat(ctx):
    # 4-parameter model: lighter settings keep the case near a minute per run.
    return _fit_common(ctx, "saturating_cooperative_contact", "js", True, 3,
                       _real44(ctx) + ["--uncertainty-diagnostics", "--split-sensitivity",
                                       "--split-schemes", "blocked_low",
                                       "--rg-weight-grid", "0,0.5"], restarts=3)


def fit_hc_kl(ctx):
    return _fit_common(ctx, "heat_capacity", "kl", False, 4, _real44(ctx)
                       + ["--uncertainty-diagnostics", "--split-sensitivity",
                          "--split-schemes", "every_third_phase"])


def fit_lcs(ctx):
    return _fit_common(ctx, "local_coordination_saturation", "js", True, 6, _saw10(ctx)
                       + FULL_SUPPLEMENT + ["--rg-weight-grid", "0,0.5,1"])


def fit_lcs_kl(ctx):
    return _fit_common(ctx, "local_coordination_saturation", "kl", False, 4, _saw10(ctx)
                       + ["--uncertainty-diagnostics"])


def fit_hsq10(ctx):
    return _fit_common(ctx, "hs_quadratic", "js", True, 4, _saw10(ctx)
                       + ["--uncertainty-diagnostics", "--split-sensitivity",
                          "--split-schemes", "blocked_low", "--rg-weight-grid", "0,1"])


def _remd(ctx, n, model, params, extra=()) -> List[str]:
    return [str(ctx["code"] / "auto" / "remd_uniform_chain_2_new.py"),
            "--N", str(n), "--model", model, "--params", params,
            "--temps", "280,300,320,340,360", "--steps-per-swap", "200", "--n-cycles", "40",
            "--seed", "11", "--burnin-frac", "0.5", "--rg-bins", "20", "--rg-scale", RG_SCALE,
            "--diagnostics", "--diagnostic-trajectories", "--structural-observables",
            "--save-configurations", "--snapshot-stride", "5", "--no-plots",
            "--n-workers", str(ctx["workers"]),
            "--out-prefix", str(ctx["case_dir"] / "run")] + list(extra)


def remd_hs(ctx):
    return _remd(ctx, 30, "hs", "627.00515,1.86699")


def remd_hs_bend(ctx):
    return _remd(ctx, 30, "hs", "627.00515,1.86699", ["--kappa-bend", "0.6"])


def remd_hsq(ctx):
    return _remd(ctx, 30, "hs_quadratic", "631.21741,1.889921,0.112106",
                 ["--Tref", "320", "--Tscale", "80"])


def remd_sat(ctx):
    return _remd(ctx, 44, "saturating_cooperative_contact", "200.0,0.0,1.0,1.5",
                 ["--Tref", "320", "--Tscale", "80"])


def remd_lcs(ctx):
    return _remd(ctx, 30, "local_coordination_saturation", "173.7070586551577,-0.7663649589655334,2.018635667565305,1.8024265951628209",
                 ["--Tref", "320", "--Tscale", "80"])


def _mc(ctx, model, params, extra=()) -> List[str]:
    return [str(ctx["code"] / "Aggregation Project" / "remd_multichain.py"),
            "--n-chains", "3", "--N", "12", "--box-size", "8",
            "--temps", "290,320,350,380", "--n-cycles", "12",
            "--local-sweeps-per-swap", "1", "--translation-sweeps-per-swap", "1",
            "--reptation-sweeps-per-swap", "1", "--rotation-sweeps-per-swap", "1",
            "--model", model, "--params", params, "--Tref", "320", "--Tscale", "80",
            "--seed", "5", "--burnin-frac", "0.5", "--rg-bins", "10", "--rg-scale", RG_SCALE,
            "--cluster-contact-threshold", "1",
            "--diagnostics", "--diagnostic-trajectories", "--save-configurations",
            "--snapshot-stride", "3", "--no-plots", "--n-workers", str(ctx["workers"]),
            "--out-prefix", str(ctx["case_dir"] / "mc")] + list(extra)


def mc_hs(ctx):
    return _mc(ctx, "hs", "558.879803466925,2.0099937029926584")


def mc_hs_bend(ctx):
    return _mc(ctx, "hs", "558.879803466925,2.0099937029926584", ["--kappa-bend", "0.5"])


def mc_sat(ctx):
    return _mc(ctx, "saturating_cooperative_contact", "200.0,0.0,1.0,1.5")


def mc_lcs(ctx):
    return _mc(ctx, "local_coordination_saturation",
               "173.7070586551577,-0.7663649589655334,2.018635667565305,1.8024265951628209")


CASES: Dict[str, Callable] = {
    "fit_hs": fit_hs, "fit_sat": fit_sat, "fit_hc_kl": fit_hc_kl,
    "fit_lcs": fit_lcs, "fit_lcs_kl": fit_lcs_kl, "fit_hsq10": fit_hsq10,
    "remd_hs": remd_hs, "remd_hs_bend": remd_hs_bend, "remd_hsq": remd_hsq,
    "remd_sat": remd_sat, "remd_lcs": remd_lcs,
    "mc_hs": mc_hs, "mc_hs_bend": mc_hs_bend, "mc_sat": mc_sat, "mc_lcs": mc_lcs,
}

# Worker counts per case family.  The single-chain sampler uses one RNG scheme
# when serial (one global stream) and another in worker mode (a fresh seed per
# cycle and lane), so n_workers=1 and n_workers>=2 legitimately differ there;
# worker invariance is checked within the parallel family (2 vs 3).  The fitter
# and the multichain sampler are invariant across 1 vs 2.
FAMILY_WORKERS = {"fit": "1,2", "remd": "1,2,3", "mc": "1,2"}
CROSS_WORKER_PAIRS = {"fit": (1, 2), "remd": (2, 3), "mc": (1, 2)}


def family(name: str) -> str:
    return name.split("_", 1)[0]


PINNED_ENV = {
    "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1",
    "PYTHONHASHSEED": "0", "MPLBACKEND": "Agg", "PYTHONUNBUFFERED": "1",
}
# Removed so that worker counts come only from the explicit CLI flags.
UNSET_ENV = ("FIT_BOOTSTRAP_WORKERS", "SLURM_CPUS_PER_TASK", "ISAW_DEBUG_CONTACTS",
             "ISAW_PROJECT_DEFINITIONS")


def select(patterns: List[str]) -> List[str]:
    if not patterns:
        return list(CASES)
    out = []
    for name in CASES:
        if any(fnmatch.fnmatchcase(name, p) for p in patterns):
            out.append(name)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--code-root", help="SAWs/updated directory of the implementation to run")
    ap.add_argument("--data-root", default=str(HERE.parent),
                    help="SAWs/updated directory providing input data (default: this tree)")
    ap.add_argument("--inputs", default=None,
                    help="directory from make_inputs.py (default: <out>/../inputs, generated if missing)")
    ap.add_argument("--out")
    ap.add_argument("--cases", default="", help="comma-separated fnmatch patterns (default: all)")
    ap.add_argument("--workers", default="auto",
                    help="comma-separated worker counts, or 'auto' for the per-family "
                         "defaults " + str(FAMILY_WORKERS))
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--timeout", type=float, default=3600.0, help="per-case timeout (s)")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args(argv)

    names = select([p for p in args.cases.split(",") if p])
    if args.list:
        for n in names:
            print(n)
        return 0
    if not args.code_root or not args.out:
        ap.error("--code-root and --out are required (unless --list)")
    code = Path(args.code_root).resolve()
    data = Path(args.data_root).resolve()
    out = Path(args.out).resolve()
    inputs = Path(args.inputs).resolve() if args.inputs else out.parent / "inputs"
    if not (inputs / "MANIFEST.json").is_file():
        subprocess.run([args.python, str(HERE / "make_inputs.py"), "--out", str(inputs)],
                       check=True, stdout=subprocess.DEVNULL)
    out.mkdir(parents=True, exist_ok=True)
    (out / "_logs").mkdir(exist_ok=True)
    timing_path = out / "_timing.json"
    timing = json.loads(timing_path.read_text()) if timing_path.is_file() else {}

    env = dict(os.environ)
    for k in UNSET_ENV:
        env.pop(k, None)
    env.update(PINNED_ENV)

    failures = []
    for name in names:
        wspec = FAMILY_WORKERS[family(name)] if args.workers == "auto" else args.workers
        for w in [int(x) for x in wspec.split(",") if x]:
            case_id = f"{name}_w{w}"
            case_dir = out / case_id
            if case_dir.exists():
                for p in sorted(case_dir.rglob("*"), reverse=True):
                    p.unlink() if p.is_file() or p.is_symlink() else p.rmdir()
            case_dir.mkdir(parents=True, exist_ok=True)
            ctx = {"code": code, "data": data, "inputs": inputs, "case_dir": case_dir, "workers": w}
            cmd = [args.python] + CASES[name](ctx)
            t0 = time.perf_counter()
            with open(out / "_logs" / f"{case_id}.out", "w") as fo, \
                    open(out / "_logs" / f"{case_id}.err", "w") as fe:
                fo.write("# " + " ".join(cmd) + "\n")
                fo.flush()
                try:
                    rc = subprocess.run(cmd, cwd=str(case_dir), env=env, stdout=fo, stderr=fe,
                                        timeout=args.timeout).returncode
                except subprocess.TimeoutExpired:
                    rc = -999
            dt = time.perf_counter() - t0
            timing[case_id] = {"seconds": round(dt, 3), "returncode": rc}
            timing_path.write_text(json.dumps(timing, indent=2, sort_keys=True) + "\n")
            status = "ok" if rc == 0 else f"FAILED (rc={rc}; see _logs/{case_id}.err)"
            print(f"{case_id:22s} {dt:8.1f} s  {status}", flush=True)
            if rc != 0:
                failures.append(case_id)
    if failures:
        print("failed cases: " + ", ".join(failures))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
