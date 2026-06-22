# Lattice model suite code review

## Result

The three scripts now form a working end-to-end pipeline:

1. `run_model_suite.py` validates the configuration, target distributions, baselines, and the shared model contract.
2. It launches `fit_lattice_contact_model.py` once per baseline/model combination.
3. It validates the fit products and passes the resulting `fit_summary.json` to `remd_uniform_chain_new.py`.
4. It launches each configured REMD seed on the target's exact temperature ladder.
5. It validates and compares the analytic fit, simulated REMD distributions, and target distributions.
6. It writes comparison CSV/JSON files, per-temperature metrics, plots when requested, a report, a manifest, and per-job resume fingerprints.

All six current models pass the synthetic end-to-end test:

- `hs`
- `tc_scale`
- `hs_quadratic`
- `poly2`
- `poly3`
- `heat_capacity`

## Important bugs fixed

### `run_model_suite.py`

- Fixed a target-schema mismatch where suite preflight accepted either `temps` or `Ts`, while the fitter accepted only `temps`.
- Added strict target validation for temperature ordering, contact-grid shape, finite values, nonnegative histograms, and nonempty rows.
- Fixed Rg comparison so target-vs-REMD Rg is computed even when the analytic baseline is contact-only and cannot predict Rg.
- Prevented stale output files from making a failed rerun look successful.
- Added exact subprocess return-code checks before accepting outputs.
- Added per-job fingerprints using the exact command and SHA-256 hashes of material inputs. `--resume` now reuses a job only when its inputs and settings are unchanged.
- Added cached hashing so large target/baseline files are not rehashed for every model and seed.
- Added strict validation of fit summaries, fit NPZ metadata, REMD NPZ grids, normalization, run summaries, model parameters, and temperature metadata.
- Fixed reports that displayed `n/a` for the winning model when no validation split was configured. They now fall back to all-temperature metrics.
- Fixed comparison status reporting so fit success, partial REMD success, and complete REMD failure are distinguished.
- Fixed plots that represented missing values as zero-height bars.
- Added strict, standards-compliant JSON output. Non-finite missing values are written as `null`, not nonstandard `NaN` tokens.
- Added blank cells rather than `nan` strings in comparison CSV files.
- Added missing all-temperature Rg/combined metrics, MAE fields, validation status, and ranking notes to the comparison CSV.
- Relative paths are now resolved relative to the configuration file, not the shell's current working directory.
- Model availability is checked dynamically from the shared fitter/REMD model contract rather than only from a hard-coded suite list.
- Strengthened configuration validation for seeds, run lengths, burn-in, Rg scales, weights, and split options.
- Expanded `--quick-test` to cover all six models, the alternate `Ts` key, contact-only Rg comparison, job fingerprints, plots, reports, and normalized outputs.

### `fit_lattice_contact_model.py`

- Added support for both `temps` and `Ts` target keys.
- Added strict validation for positive, increasing temperatures; evenly spaced contact and observed-Rg centers; finite/nonnegative histograms; and positive row mass.
- Added early validation of `Tref`, `Tscale`, and heat-capacity `T0`, preventing partial outputs from being written before a late failure.
- Added standalone validation for restart counts, bootstrap counts, contact offset, Rg scale, and Rg objective weight.
- Added stricter baseline validation for discrete contacts, probability arrays, joint distributions, and contact histogram grids.
- Prevented silent truncation of noninteger `c_vals` and rejected duplicate contact values.
- Corrected rebinning of a joint baseline's contact marginal. `crg_prob.sum(axis=1)` is probability mass, so it is now rebinned using the actual `c_edges`, rather than being incorrectly treated as a density sampled at bin centers.
- Corrected baseline integer support detection for arbitrary contact edges.
- Optimizer selection now requires a finite, successful optimization result. Failed restarts are not silently selected because they happen to report a low objective.
- Bootstrap fits likewise retain only successful, finite optimizer results.
- Matplotlib is now optional when `--no-plots` is used, which matches the suite's default behavior.
- JSON writing is strict and rejects nonstandard non-finite values.
- Corrected the misleading `--rg-scale` help text, whose written default did not match the actual default.

### `remd_uniform_chain_new.py`

- Retained the correct reduced-potential local and replica-swap acceptance rules for temperature-dependent `b(T)` models.
- Retained exact shared model names, parameter ordering, and numerical `b(T)` agreement with the fitter.
- Improved multiprocessing substantially: workers no longer receive and return each replica's ever-growing trajectories every cycle. Only chain state and acceptance counters cross process boundaries, avoiding quadratic serialization growth with cycle count.
- Added robust seeding for arbitrary integer seeds while respecting NumPy's 32-bit seed requirement.
- Added automatic creation of nested output directories for standalone runs.
- Fixed the run-summary output list so the run-summary path is included in the JSON itself.
- Fixed swap-rate handling for adjacent pairs that received zero proposals. These are now reported as unavailable rather than incorrectly as zero acceptance.
- Added finite-value validation for Rg scale and linear temperature endpoints.
- Matplotlib is now optional when `--no-plots` is used.
- Run-summary JSON is strict and converts non-finite missing values to `null`.

## Tests performed

- Python syntax compilation for all three scripts.
- Full `remd_uniform_chain_new.py --quick-test`, including serial and two-worker execution, all six model forms, fit-summary loading, CSV loading, parameter ordering, temperature validation, and distribution normalization.
- Full `run_model_suite.py --quick-test`, fitting and simulating all six models end to end.
- End-to-end test using a target with `Ts` but no `temps` key.
- Rg comparison test using a contact-only baseline, confirming Rg contributes to the combined target-vs-REMD score.
- No-validation-split test, confirming the report uses all-temperature metrics rather than displaying an `n/a` winner.
- Stale-output failure test, confirming a failed subprocess cannot be marked successful from old files.
- Resume test, confirming identical work is skipped and a changed REMD setting reruns only the affected REMD job.
- Parallel REMD smoke tests after the worker-payload optimization.

## Configuration points that remain scientifically important

The code can validate consistency, but it cannot infer these choices reliably:

- `N` must match the intended lattice chain length.
- `contact_offset` must map the target contact definition onto the lattice nonbonded-contact definition.
- `rg_scale` must correctly convert lattice Rg units to the target Rg units.
- A baseline supplied as `c_vals, Pc` should genuinely represent an athermal baseline. Averaging biased multi-temperature distributions is not generally an athermal density of states.
- `fit_rg=true` is meaningful only when the joint baseline `P0(m,Rg)` has adequate Rg support overlap with the target.
- Validation temperatures should be chosen before comparing flexible models, especially `poly3` and `heat_capacity`.
- Production REMD runs still require convergence checks. Swap acceptance, local acceptance, seed-to-seed variation, and trajectory length should be examined rather than relying only on the model ranking.
- The fitter reports a summed divergence objective (and its current JS implementation uses natural logarithms), while the suite reports mean per-temperature base-2 JS. These are both valid for ranking within their own columns, but their raw numerical values should not be compared directly.

## Recommended invocation

Run the built-in checks first:

```powershell
python .\remd_uniform_chain_new.py --quick-test
python .\run_model_suite.py --quick-test
```

Then inspect the commands without launching simulations:

```powershell
python .\run_model_suite.py --config .\model_suite_config.json --dry-run
```

Launch or safely resume the suite:

```powershell
python .\run_model_suite.py --config .\model_suite_config.json
python .\run_model_suite.py --config .\model_suite_config.json --resume
```

Use `--continue-on-error` when one failed model or seed should not stop independent jobs. Use `--force` together with `--resume` when every configured job should be rerun.
