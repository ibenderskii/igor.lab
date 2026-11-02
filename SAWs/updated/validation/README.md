# Validation harness (Stage 1 of the SU-reduction plan)

These tools prove that a performance change leaves the science **bit-identical**. Each one compares a *baseline* implementation (normally `origin/main`) with a *candidate* (the working tree). They don't depend on the code under test, and they run on Python ≥ 3.8 with numpy (h5py for HDF5 outputs, pytest optional).

| File | Purpose |
|---|---|
| `golden_check.sh` | One command: extract the baseline's `*.py`/`*.json` with `git archive`, run every case against both trees, and compare |
| `run_cases.py` | The seeded short cases (fitter, single-chain REMD, multichain REMD), run against any `--code-root` |
| `compare_outputs.py` | Bit-level tree comparison: NPZ, JSON, CSV/TXT, HDF5 |
| `golden_report.py` | Case-by-case baseline-vs-candidate comparison, plus worker-count invariance |
| `make_inputs.py` | Deterministic inputs: an **exact** enumerated 10-bead SAW baseline (scalar + local-coordination tables) and a synthetic target |
| `lockstep.py` | Python-vs-new-kernel lockstep test (state + RNG compared after every chunk), for the Numba stages |
| `test_validation_harness.py` | The harness checks itself: planted 1-ulp, signed-zero, dtype, CSV, HDF5 and RNG-skew differences must be caught |

## Quick start

```bash
cd igor.lab                                   # a git checkout with the candidate changes
source /scratch/midway3/ibenderskii/midway3_common.sh   # on Midway3: the production environment
bash SAWs/updated/validation/golden_check.sh -b origin/main   # ~25 min on one core for all cases
#   -c 'fit_*'   restrict to some cases;   -o DIR   choose the work directory
python -m pytest -q SAWs/updated/validation    # harness self-tests (seconds)
```

The result is printed and written to `<workdir>/golden_report.txt`. Every line must read `ok`, ending with `RESULT: ALL IDENTICAL`.

## What "identical" means

- **NPZ / HDF5 arrays:** dtype, shape and raw bytes must match. NaN payloads and signed zeros count.
- **JSON:** recursive equality.
- **CSV:** text equality (the first differing cells are reported).

A field is excluded only by **name**, and every excluded difference is still listed as `[ignored]`. The names excluded by default are:

- wall times and timestamps;
- host / Python / platform fields;
- `git_commit` and other `git_*` fields;
- anything matching `*_seconds`.

The output roots and code roots are normalized to placeholder tokens, because outputs record their own paths.

The cross-worker check also excludes the worker-count fields themselves: `execution`, `n_workers`, `workers`, `worker_source`, `command_line`.

## Cases

```text
fit_hs  fit_sat  fit_hc_kl      real 44-mer data (RnBaseline44 + remd_distributions_44mer)
fit_lcs fit_lcs_kl fit_hsq10    exact 10-bead SAW baseline (local-coordination + quadratic models)
remd_hs remd_hs_bend remd_hsq remd_sat remd_lcs      single-chain REMD, 5 lanes x 40 cycles
mc_hs mc_hs_bend mc_sat mc_lcs                       multichain REMD, M=3 N=12 L=8, 4 lanes x 12 cycles
```

The fitter cases switch on:

- bootstrap with prediction bands;
- uncertainty diagnostics;
- split sensitivity;
- the rg-weight scan.

The sampler cases switch on diagnostics, diagnostic trajectories, configuration snapshots (HDF5) and, for single chain, structural observables.

**Worker counts:**

| Family | Workers run | Invariance pair |
|---|---|---|
| fitter | 1, 2 | 1 vs 2 |
| multichain | 1, 2 | 1 vs 2 |
| single-chain REMD | 1, 2, 3 | 2 vs 3 |

## A finding the harness surfaced

On current `main`, **single-chain REMD with `--n-workers 1` doesn't produce the same trajectory as `--n-workers ≥ 2`.**

- **Serial mode** draws every sweep from one global `random` stream.
- **Worker mode** reseeds each lane every cycle (`_seed_all(base_seed + 100000*cycle + k)`).

Both are valid, but they are different streams. Results are identical for any W ≥ 2 (checked 2 vs 3 for hs, hs+κ, hs_quadratic, sat and lcs).

Production ran with 32 workers, so the parallel stream is the one to preserve. **Stage 3 must therefore give its 1-core Numba runs the worker-mode seeding.** Then a 1-core run reproduces the Oct 16–17 `--n-workers 32` outputs exactly. The fitter and the multichain sampler are invariant for 1 vs 2.
