# Contact-number-quadratic models: implementation report

Adds two new contact potentials — `hs_m2_const` and `hs_m2_hs` — to the fitting
and REMD pipeline, alongside a small generic potential interface that keeps
the six existing linear models bit-for-bit unchanged.

## Motivation

Every existing model reweights the athermal baseline with a bias linear in
the contact number:

```
P_model(m|T) ∝ P0(m) · exp[-b(T)·m]
```

The new models add a curvature term in `m`, normalized by chain length `N`:

```
u_contact(m, T; N) = b(T)·m + κ(T)·m²/(2N)
```

| Model | κ(T) | Parameters |
|---|---|---|
| `hs_m2_const` | constant | `h1, s1, kappa2` |
| `hs_m2_hs` | `h2/T - s2` | `h1, s1, h2, s2` |

Both nest `hs` exactly: at zero curvature parameters, `u_contact` reduces to
`(h1/T - s1)·m` bit-for-bit. The optimizer's first restart starts there
(`x0` has the curvature terms at `0.0`), so a fit can always fall back to the
hs solution if the data has no curvature signal.

`hs_quadratic` (already in the registry) is unrelated — it is quadratic in
**temperature**, not in `m`, and stays linear in `m`.

## Model API changes

### Generic potential interface

Every model in `MODEL_REGISTRY` now carries:

- `raw_q_fn` — quadratic coefficient `κ(T)`; `0.0` for the six linear models
- `potential_kind` — `"linear"` or `"contact_quadratic"`
- `quadratic_normalization` — `None` or `"m^2/(2N)"`
- `requires_chain_length` — `True` only for the two new models

New accessors alongside the existing `reduced_bias()` / `make_b_fn()`
(kept as backward-compatible linear-coefficient accessors):

```python
quadratic_bias(model, params, T, Tref, Tscale)      # κ(T); 0 for linear models
make_q_fn(model, Tref, Tscale)                        # closure over κ(T)
reduced_contact_potential(m, T, model, params, ...)   # u_contact; == b(T)*m for linear
make_contact_u_fn(model, Tref, Tscale, n_beads=None)  # closure used everywhere
```

For every legacy model, `reduced_contact_potential()` returns exactly
`reduced_bias(...) * m` (verified with `np.array_equal`, not `np.allclose`).

### Chain length resolution

A small resolver, duplicated in both fitters (mirrors the existing
duplicated-registry architecture — no new shared module):

```
read_baseline_chain_length(b_data)   # prefers n_beads, falls back to N,
                                      # raises on n_beads != N
resolve_chain_length(baseline_n, cli_n, model_name=..., baseline_path=...)
                                      # baseline wins; --N is fallback only;
                                      # conflict -> error; required for the
                                      # two new models; None is fine for
                                      # legacy linear fits
```

`--N` was added to both fitters' CLIs. Legacy baselines without `n_beads`/`N`
still fit any linear model unchanged.

### API version

`MODEL_API_VERSION` bumped **1 → 2** consistently in
`fit_lattice_contact_model_2.py`, `FIT_TO_DAT.py`, and
`remd_uniform_chain_2_new.py` (both copies), plus
`ISAW Project/project_definitions.json`'s `output_schema_versions.model_api_version`
(read by `isaw_schema.py`'s consistency check).

The suite's `check_model_contracts()` (`run_model_suite_2.py`) now also
compares `potential_kind`, `quadratic_normalization`, and numeric `κ(T)`
between the fitter and REMD registries, not just `b(T)`.

## Reweighting: linear coefficient → full potential

Every production reweighting path was converted from `exp[-b(T)·m]` to
`exp[-u_contact(m,T;N)]`, using a shared `u_fn(params, T, m)` closure in place
of the old `b_fn(params, T)`:

- `model_contact_mass`
- `objective` / `objective_combined` (contact-only and contact+Rg)
- `predict_rg_from_joint`
- `predict_rg_summary_from_joint` (scalar-Rg mode)
- per-temperature metrics (`per_temp_contact_losses`, `_predicted_means`)
- bootstrap (`run_bootstrap_uncertainty`, `run_rg_scalar_bootstrap`)
- split sensitivity (`run_split_sensitivity`, `run_rg_scalar_split_sensitivity`)
- uncertainty diagnostics (`run_uncertainty_diagnostics`)
- Rg-weight sensitivity / prediction bands
- both `--quick-test` suites

`b_fn` is retained everywhere purely for reporting (`b(T)` plots/CSVs); it
never re-enters the weight calculation.

### Support-aware max subtraction (bug fix)

The old stabilization took `x - x.max()` over *all* bins, including ones
where `p0 == 0`. That's harmless for a narrow linear bias, but with a
contact-quadratic term (or an extreme linear one) an unsupported bin can hold
the largest exponent, deflating every *supported* weight to underflow.

Measured on `RnBaseline30.npz` (54/81 unsupported bins), 8064 evaluations
across all six linear models and 21 parameter draws each:

| old partition function | n | worst distance from new |
|---|---|---|
| healthy | 6836 | 4.8e-15 (float noise) |
| denormal (~1e-319) | 199 | 5.0e-6 |
| **underflowed to 0** | **1029** | **0.99 — old silently returned a uniform distribution** |

That last row was a real defect: at extreme `b`, the old code's fallback
(`Z <= 0 → uniform`) fired, and under JS divergence a uniform distribution
can score *better* than the true model when the data lies outside the
baseline's support — an incentive for the optimizer to drift there.

The fix, `_stabilized_exponent(x, support)`:
- shifts by `max(x[support])`, not `max(x)`
- sets **unsupported bins to `-∞`** rather than leaving them at a large
  finite exponent (which would otherwise overflow `exp()` and turn the
  `0 · inf` product into `NaN` — this was a bug I introduced mid-fix and
  caught with `np.errstate(over="raise")` in the regression harness)
- reduces to exactly `x - x.max()` when everything is supported, so linear
  models are affected only in the (now-fixed) underflow regime

**Numerical result:** on the paper configuration (`--contact_offset 29`,
support overlap ~94%), all six linear models' fitted objectives agree to
**~1e-9 relative**; parameters differ by 1e-6–1e-4 relative (float-noise-scale
drift through a flat h–s optimizer valley, not a systematic shift). The
`FIT_TO_DAT.py` scalar-Rg path on `single.dat` is **bitwise identical**
end-to-end, including `rg_feasibility.csv`/`rg_feasibility_summary.json`.

## FIT_TO_DAT scalar-Rg feasibility for nonlinear models

The existing finite bias scan (`run_rg_feasibility_scan`) sweeps a single
scalar `b` and reads it as the model's reachable range — valid only because a
linear potential is indexed by one number. A contact-quadratic potential has
a two-parameter weight family, so a 1-D scan point isn't a model prediction
and the `b → ±∞` endpoints aren't limits the model ever takes.

Dispatch added in `run_rg_scalar_mode()`:

```python
bias_scan_applicable = potential_kind == "linear"
```

- **Linear models**: unchanged — same scan, same filenames
  (`rg_feasibility*.csv/.png/_summary.json`).
- **Contact-quadratic models**: `run_rg_contact_slice_diagnostic()` runs
  instead. It:
  - retains the support-overlap check
  - retains the rigorous `global_rg_outer_bounds()` (holds for **any**
    contact-only reweighting, linear or not — proven from the
    convex-combination structure over contact slices, independent of the
    potential's functional form)
  - writes `<prefix>_contact_slices.csv` — the model-independent
    per-contact-bin conditional-Rg table (bin, contact value, baseline mass,
    conditional Rg in lattice/observed units, raw moment)
  - writes `<prefix>_summary.json` with `bias_scan_applicable: false`,
    `endpoint_limits_applicable: false`, and a `not_applicable_reason`
    string explaining why
  - does **not** write `<prefix>.csv` / `<prefix>.png` (the scan artifacts)
  - still reports the fitted Rg(T) curve and `T_rg_max_slope`
    (`_rg_scalar_transitions` is potential-agnostic)

`classify_scientific_validity()` gained a matching short-circuit branch: for
a nonlinear model the status ladder collapses to
`zero_support_overlap → outside_global_outer_bound → unverified_no_bias_scan`,
never `supported` (there is no scan-scoped evidence to support it with).

## Bending compatibility

Unchanged behavior, extended to both new models and to `FIT_TO_DAT.py` (which
previously had no bending-metadata handling in its scalar-Rg path):

- baseline `kappa_bend` is read, never re-fitted
- missing legacy metadata reads as `0.0`
- CLI `--kappa-bend` is a consistency check against the baseline, not an
  override (mismatch raises)
- `kappa_bend`, `bending_enabled`, `bend_definition` recorded in all outputs
- no joint histogram over contacts *and* bends is required — the bending
  weight is already baked into `P0(m)` / `P0(m, Rg)`

## Outputs and plots

New fields in `fit_summary.json` / `fit_results.npz` (contact mode and
scalar-Rg mode):

- `reduced_bias_by_temperature` — kept as the linear-coefficient
  compatibility alias
- `linear_coefficient_by_temperature`, `quadratic_coefficient_by_temperature`
- `potential_kind`, `fit_chain_length`, `quadratic_normalization`

A `quadratic_coefficient_vs_T.png` (contact mode) / `rg_scalar_qT.png`
(scalar-Rg mode) plot is added **only** for `potential_kind != "linear"`.
Neither the linear coefficient's nor the quadratic coefficient's zero
crossing is described as *the* transition temperature anywhere in the code
or docstrings — `T_rg_max_slope` remains the primary finite-chain descriptor.

## REMD / sampler scope

The samplers now sample the full contact potential (linear and
contact-quadratic alike); the `require_linear_contact_potential` guard is no
longer called by either sampler (it is retained only as an unused helper for
any strictly linear-only sampler that might import it).

**Single-chain (`remd_uniform_chain_2_new.py`).** Acceptance, swaps, effective
energy, and observables route through the generic `reduced_contact_potential(m,
T; N)` (linear → `b(T)·m`; contact-quadratic → `b(T)·m + κ(T)·m²/(2N)`) plus the
fixed reduced bending penalty. The runtime chain length `N` drives the
`m²/(2N)` normalization; linear models remain bit-for-bit identical.

**Multi-chain (`remd_multichain.py`).** The reduced potential is

```
u(X, T) = lambda_intra · Σ_alpha u_contact(m_intra_alpha, T; N)
        + lambda_inter · b(T) · m_inter
        + kappa_bend · n_bend_total
```

The fitted `m²/(2N)` curvature applies **per chain** to intrachain contacts
only; interchain contacts stay **linear** (`b(T) = reduced_bias`) because the
single-chain fit does not identify an interchain quadratic term. To supply each
chain's `m_alpha` the authoritative state gained a per-chain cache
`intra_contacts_by_chain` (invariant `sum == counts.intra`), carried through
initialization, moves, worker serialization, and reconstruction. Production
code, observables, and swaps use the state-aware `reduced_potential_state` /
`swap_log_accept_state`; the aggregate-`ContactCounts` helpers
(`reduced_potential_counts`, …) remain for linear models and now **reject**
contact-quadratic models (aggregate totals cannot determine `Σ_alpha m_alpha²`).
Local moves on chain `alpha` score `du = lambda_intra·(u_contact(m_new) −
u_contact(m_old)) + lambda_inter·b(T)·Δm_inter + kappa_bend·Δbends` (never the
`b·Δm` approximation for nonlinear models); linear models keep their exact
historical arithmetic. Note that `(lambda_intra, lambda_inter) = (0, 0)`
disables the contact interactions but does **not** disable bending when
`kappa_bend != 0`. Serial and multiprocessing runs stay bit-identical.

## Files changed

| File | Change |
|---|---|
| `fit_lattice_contact_model_2.py` (×3 identical copies: `Aggregation Project`, `ISAW Project`, `auto`) | Registry, generic potential API, chain-length resolver, `u_fn` reweighting throughout, `--N`, new outputs/plot, quick-test 11 |
| `FIT_TO_DAT.py` | Same shared core (ported verbatim), plus bending metadata (previously absent in this file), scalar-Rg `u_fn` path, `run_rg_contact_slice_diagnostic`, feasibility dispatch, validity-ladder branch, `rg_scalar_qT.png` |
| `remd_uniform_chain_2_new.py` (`Aggregation Project`) | Full contact potential in acceptance/swaps/energy/observables/metadata (see single-chain section) |
| `remd_multichain.py` | Per-chain intrachain quadratic + linear interchain: state-aware potentials, quadratic move/swap, `intra_contacts_by_chain` cache threaded through workers, contract metadata |
| `multichain_state.py` | `intra_contacts_by_chain` field + `__post_init__`/`copy`/`make_state`/`validate_state` |
| `multichain_contacts.py` | `full_contacts_split` / `full_intra_contacts_by_chain*`; per-chain cache maintained in `apply_moved_beads`; per-chain debug assert |
| `test_remd_multichain.py` | +15 contact-quadratic tests (per-chain cache, additivity, per-chain swap, determinism, fit-summary round-trip, output metadata) |
| `run_model_suite_2.py` (×2: `ISAW Project`, `auto`) | Contract check extended to `potential_kind` / `quadratic_normalization` / numeric `κ(T)` |
| `ISAW Project/project_definitions.json` | `model_api_version` 1 → 2 |
| `test_contact_quadratic_models.py` | **New** — 60 tests |

## Tests

| Suite | Result |
|---|---|
| `test_contact_quadratic_models.py` (new) | 60 passed |
| Aggregation Project (`test_fit_kappa_bend`, `test_multichain_*`, `test_remd_multichain`) | 128 passed, 4 skipped |
| `SAWs/updated/tests/` | 38 passed |
| `ISAW Project` (`tests/`, `test_isaw_contact_observables.py`) | 305 passed, 185 skipped, 1 failed |
| `fit_lattice_contact_model_2.py --quick-test` | passed |
| `FIT_TO_DAT.py --quick-test` | passed |
| Suite contract preflight | `api v2, 8 models, 2 contact-quadratic, numeric b(T) and κ(T) equal` |

The one ISAW failure (`test_direct_k_diagnostic_smoke_produces_reports`)
reproduces identically against the pre-change committed code — it needs
`h5py`, which isn't installed in this environment. Unrelated to this change.

New-suite coverage maps directly to the nine requested areas: legacy
equivalence, zero-curvature nesting, exact `m²/(2N)` normalization,
vectorized/scalar equality, synthetic recovery for both new models, contact
+ joint-Rg weight consistency (including the unsupported-bin stabilization
fix), bending-enabled baseline compatibility, fit-summary/contract parity,
and nonlinear scalar-Rg feasibility dispatch.

## Known limitations

- `hs_m2_const` / `hs_m2_hs` have `derived_Tc = None` — no closed-form Tc,
  so `T_bias_zero_model_derived` is `null` for them (by design: neither
  model has one).
- `run_model_suite_2.py`'s `SUPPORTED_MODELS` tuple is unchanged (still the
  six linear models) — its self-test drives REMD per model, which would hit
  the new guard. Wiring the new models through the suite is out of scope
  here.
- The older, unrelated pair `fit_lattice_contact_model.py` /
  `remd_uniform_chain_new.py` (no `_2`) is untouched, stays at API v1.
- `pytest` was not installed on any interpreter in this environment (despite
  committed `.pyc` cache files implying it once was) and was installed via
  `pip install --user pytest` to run the suites above.
