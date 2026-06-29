# ISAW structural-analysis correction — implementation plan (2026-06-29)

Canonical production sampler: `remd_uniform_chain_2_new.py`.
`remd_uniform_chain_2.py` is currently byte-identical and is referenced by the
existing test + `run_model_suite_2.py`; it is kept in sync (mirrored) so the
ecosystem keeps working. Docstring references to the non-`_2` legacy names are
removed.

## Sampling invariants that MUST NOT change
- Target weight `P(C|T) ∝ exp[-m(C) b(T)]`  (`reduced_potential`, `_b_*`).
- Local Metropolis `du = u_new - u_old; accept if du<=0 or rand<exp(-du)` (`mc_sweep`).
- Swap criterion `swap_log_accept` (unchanged).
- RNG call sequence inside the accepted/rejected MC path (unchanged unless a
  demonstrated bug forces a change, documented explicitly).
- Canonical NPZ keys and result-CSV column order (append only).

## Phase → file/function map

| Phase | File | Functions | Tests | Risk |
|---|---|---|---|---|
| 1 strict coords | `isaw_contact_observables.py` | `normalize_lattice_coordinates` (new); wire into build/validate/geometry/graph | `tests/test_geometry.py`, `test_contact_map.py` | geometry callers must allow non-SAW light mode |
| 2 full contact map | `isaw_contact_observables.py` | `validate_contact_map` (strict), `contact_separation_counts` | `test_contact_map.py` | stricter validation may reject previously-accepted partial maps |
| 3 bins | `isaw_contact_observables.py`, new `project_definitions.json` | `bin_contact_separations_fixed/scaled`, `validate_bin_definitions`, `FIXED/SCALED_BIN_DEFINITIONS` | `test_contact_bins.py` | old `bin_contact_separations`/`default_bin_definitions` kept as compat shim |
| 4 online storage | `remd_uniform_chain_2_new.py` | `run_remd`, `Replica`, CLI `--structural-observables`,`--save-m-r-trajectories` | `test_remd_regression.py` | default behavior change: m_r not retained unless enabled |
| 5 HDF5 writer | `isaw_config_io.py` | `SnapshotWriter` rewrite; `committed_rows`, `status`, `mark_complete` | `test_snapshot_io.py` | reader contract change (committed_rows) |
| 6 null moves | `remd_uniform_chain_2_new.py` | `mc_sweep`, `MOVE_COUNTER_COLS`, CSV, `detect_local_move_freezing` | `test_pair_motifs.py`?/inline | must preserve RNG sequence + detailed balance |
| 7 diagnostics | `remd_uniform_chain_2_new.py` | `compute_run_diagnostics`, `save_*` | `test_diagnostics.py` | add tau_int_cycles/ess; keep tau_int alias |
| 8 small fixes | several | `_parse_bool_env`, `q` overflow, SCHEMA bumps | inline | semantics version bump |
| 9 extractor | `extract_contact_motif_features.py` rewrite | committed_rows, status gate, manifest, full m_r, validation counts | `test_feature_extractor.py` | format default hdf5 |
| 10 tests | `tests/` | nine files | — | — |
| 11 pilot | run | N=30,N=44 | validation JSON | — |

## Schema/version bumps
- `SNAPSHOT_SCHEMA_VERSION` 1→2 (committed_rows, status, richer metadata).
- `FEATURE_SCHEMA_VERSION` 1→2 (manifest, /features/m_r, validation counts).
- `SCHEMA_VERSION` 1→2 (bin-definition semantics, move counters).
