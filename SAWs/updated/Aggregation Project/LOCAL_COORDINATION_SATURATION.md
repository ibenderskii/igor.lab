# Local coordination saturation in aggregation

The registry model `local_coordination_saturation` consumes the fit summary
written by `../auto/fit_lattice_contact_model_parallel.py`. Its single- and
multichain Hamiltonian is

```text
u(X,T) = (h_b/T - s_b) m(X)
         - A0 sum_i q_i^2 / [1 + (q_i/q_sat)^2],
q_i = k_i/2
```

Here `k_i` is the nonbonded nearest-neighbour degree of monomer `i`. In the
multichain state it counts intrachain and interchain neighbours identically.
Covalent neighbours never count. The exact graph invariant is
`sum_i k_i = 2*(m_intra + m_inter)`.

Run the aggregation sampler directly from the single-chain fit:

```bash
python remd_multichain.py \
  --fit-summary-json path/to/fit_summary.json \
  --n-chains 8 --N 30 --box-size 24 \
  --lambda-intra 1 --lambda-inter 1 \
  --out-prefix runs/aggregation_local_coordination
```

For this model:

- omitted `--cooperativity` resolves to `local`;
- explicit `--cooperativity global` is rejected;
- `lambda_intra` must equal `lambda_inter` because the fitted local degree is
  label blind;
- `A0 = 0` recovers the linear `hs` contact potential exactly;
- `kappa_bend*n_bend_total` remains a separate additive term.

The historical `saturating_cooperative_contact` model is unchanged: it still
defaults to its global `m_total/(M*N)` rule and continues to offer its existing
optional `--cooperativity local` interpretation.

## Intra/inter degree weights

Each monomer's degree is tracked split into intrachain and interchain
neighbours, `k_i = k_intra,i + k_inter,i`, and the cooperative term is
evaluated at

```text
k_eff,i = w_intra*k_intra,i + w_inter*k_inter,i      (inside g only)
u(X,T)  = lambda * [b(T)*m_total - A0 * sum_i g(k_eff,i)] + kappa_bend*n_bend_total
```

The linear `b(T)*m_total` term is unweighted. Set the weights with
`--degree-weight-intra` / `--degree-weight-inter` (finite, `>= 0`). The default
`(1, 1)` treats intra- and interchain neighbours identically and reproduces the
Hamiltonian above bit for bit. Non-default weights are rejected for every rule
other than `cooperativity="local"`; they apply equally to
`saturating_cooperative_contact --cooperativity local`. When the weights are not
`(1, 1)` the recorded `multichain_potential_definition` says so.

## Per-bead degree cache

Local-rule runs carry `MultiChainState.degrees` (`(M*N, 2)` int64, columns
`k_intra, k_inter`, indexed by global bead ID) and `degree_joint` (`(7, 7)`
int64 bead counts by `(k_intra, k_inter)`). Every other Hamiltonian leaves both
`None`. A move reads the old degrees from the cache and the new ones through an
overlay of the trial occupancy (no copy of `site_owner`), over the moved beads
and the occupants neighbouring their old and new sites. It commits them only on
acceptance. The contact deltas follow exactly from `sum_i k_intra,i = 2*m_intra`
and `sum_i k_inter,i = 2*m_inter`. Sums contract integer bead counts with `g` in
a fixed order, so serial and `--n-workers` runs stay bit-identical; the worker
receives and returns the degree array. `--debug-contacts` also checks the cache
element-wise against the full oracle after every accepted move.

At `M=8, N=44, L=24` (Python 3.8 environment) the local-model sweep went from
73.9 ms to 56.0 ms and a full-`u` evaluation from 603 µs to 8.9 µs; the `hs`
sweep is unchanged (45.5 vs 45.9 ms).

## Coordination outputs

Recorded for every model; the `S` quantities only for local-rule runs. The
post-burn-in window starts at `floor(n_cycles * burnin_frac)`, the same index as
the results CSV (`--burnin-frac`, default 0.5).

| output | key / column / dataset | shape, dtype | definition |
|---|---|---|---|
| NPZ | `coord_degree_values` | (7,) int64 | k = 0..6 |
| NPZ | `coord_degree_hist` | (nT, n_cycles, 7) int64 | per-cycle bead counts h_k |
| NPZ | `coord_post_burnin_start_cycle`, `coord_n_post_cycles` | (nT,) int64 | accumulator window |
| NPZ | `coord_P_degree` | (nT, 7) float | post-burn-in bead fraction at degree k |
| NPZ | `coord_joint_counts` / `coord_P_joint` | (nT, 7, 7) int64 / float | beads by (k_intra, k_inter): counts / fraction |
| NPZ | `coord_mean_degree_by_position` and `_intra_` / `_inter_` parts | (nT, N) float | <k>(i) averaged over chains and post-burn-in cycles |
| NPZ | `coord_S_mean`, `coord_S_std` | (nT,) float | local rule only |
| NPZ, run summary, snapshot metadata | `degree_cache_enabled`, `degree_weight_intra`, `degree_weight_inter`, `coordination_observables_schema_version` | scalars | provenance |
| results CSV (appended) | `degree_mean`, `frac_k0`..`frac_k6` | float | post-burn-in means |
| results CSV (appended) | `S_mean`, `u_linear_mean`, `u_coop_mean` | float, blank if not local | `u_linear + u_coop = u - kappa_bend*n_bend` |
| snapshot HDF5 | `snapshots/degree_histogram` | (n, nT, 7) int64 | every run |
| snapshot HDF5 | `snapshots/cooperative_sum` | (n, nT) float64 | local rule only |
| diagnostics | `mean_degree`, `cooperative_sum` | series | ESS / drift only, no warnings |

Schema versions: multichain output 4 → 5 and snapshot 1 → 2, both additive (no
existing key, column, dataset or meaning changed); coordination observables
v1. Per-bead degrees are not stored in snapshots;
`multichain_observables.snapshot_degrees(path)` rebuilds them from the
coordinates and `L`, and reads v1 files as well.

The complete baseline-generation and single-chain fitting workflow is in
[`../auto/LOCAL_COORDINATION_SATURATION.md`](../auto/LOCAL_COORDINATION_SATURATION.md).
