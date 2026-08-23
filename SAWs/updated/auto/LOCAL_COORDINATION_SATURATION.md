# Local per-monomer coordination saturation

`local_coordination_saturation` is an additive model alongside every existing
contact model. It does not replace or reinterpret
`saturating_cooperative_contact`.

For a conformation `X`, let `k_i` be bead `i`'s nonbonded nearest-neighbour
degree and let `q_i = k_i/2`. Covalent predecessor/successor beads are excluded.
The fitted reduced contact potential is

```text
u(X,T) = (h_b/T - s_b) m(X)
         - A0 sum_i q_i^2 / [1 + (q_i/q_sat)^2]
```

The parameter order is `[h_b, s_b, A0, q_sat]`, with `A0 >= 0` and
`q_sat > 0`. Setting `A0 = 0` recovers `hs` exactly. The bending penalty, when
enabled, remains the separate additive term `kappa_bend*n_bend`.

## 1. Generate the fitting baseline

Run `single_chain_wang_landau.py` as usual. For the supplied 30-mer suite
configuration, the matching default production command is:

```bash
python single_chain_wang_landau.py \
  --N 30 --target_npz remd_distributions_30mer.npz --dist_dir dists \
  --n_workers 12 --steps_per_worker 400000000 --base_seed 42
```

This writes
`dists/single_chain_wang_landau_N30_workers12_steps400000000_seed42.npz`, the
baseline named by `config_local_coordination_30.json`. If any filename-forming
production option changes, update the config path to the emitted `DIST_FILE`.

Adaptive Wang-Landau learning is unchanged. During the existing frozen-weight
production sampling, the script also records the degree histogram
`(h_0,...,h_6)` and writes a compressed state table to the output NPZ:

- `local_coord_histograms`
- `local_coord_contact_counts`
- `local_coord_state_mass`
- `local_coord_rg_state_index`, `local_coord_rg_bin_index`, and
  `local_coord_rg_joint_mass` when the joint output is enabled

Every output is validated against `sum_k h_k = N`,
`sum_k k*h_k = 2*m`, `P0(m)`, and the Rg marginal. Production checkpoints are
version 2 and include the coordination samples. A version-1 checkpoint that
already contains samples is rejected because those historical conformations
cannot be reconstructed honestly.

## 2. Fit in `auto`

```bash
python fit_lattice_contact_model_2.py \
  --remd remd_distributions_30mer.npz \
  --baseline path/to/wang_landau_baseline.npz \
  --contact_offset 29 \
  --model local_coordination_saturation \
  --loss js \
  --outdir fits/30mer_local_coordination
```

Add `--fit-rg --rg-weight <value>` only when the baseline contains the sparse
coordination-state/Rg joint statistics. Legacy baselines continue to work for
legacy models, but the local model fails clearly if its required state table is
missing.

`FIT_TO_DAT.py` advertises the shared model contract but deliberately refuses
this model because its scalar-contact/Rg workflow cannot represent a
configuration-dependent Hamiltonian. Use `fit_lattice_contact_model_2.py`.

## 3. Simulate the fitted single chain

```bash
python remd_uniform_chain_2_new.py \
  --N 30 \
  --fit-summary-json fits/30mer_local_coordination/fit_summary.json \
  --Tmin 280 --Tmax 380 --nT 12 \
  --steps-per-swap 1000 --n-cycles 5000 \
  --out-prefix runs/local_coordination_30mer
```

The sampler caches the degree histogram with each chain state and evaluates it
in local moves and all four terms of the replica-exchange swap rule. Scalar
`u(m,T)` accessors reject this model rather than substituting a global `m/N`
potential.

## 4. Transfer the fit to aggregation

```bash
python remd_multichain.py \
  --fit-summary-json fits/30mer_local_coordination/fit_summary.json \
  --n-chains 8 --N 30 --box-size 24 \
  --lambda-intra 1 --lambda-inter 1 \
  --out-prefix runs/aggregation_local_coordination
```

For this model, an omitted `--cooperativity` resolves to `local`. An explicit
`--cooperativity global` is rejected. Intrachain and interchain neighbours both
enter the same `k_i`, and unequal `lambda_intra`/`lambda_inter` values are
rejected because the fitted Hamiltonian is label blind. The historical
`saturating_cooperative_contact` model retains its historical global default and
its optional local mode.

## Compatibility

The shared model API is version 4. Version-3 fit summaries for existing models
remain loadable. Existing model parameters, potential definitions, acceptance
arithmetic, and defaults are unchanged.
