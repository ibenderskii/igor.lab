# Local coordination saturation in aggregation

The registry model `local_coordination_saturation` consumes the fit summary
written by `../auto/fit_lattice_contact_model_2.py`. Its single- and multichain
Hamiltonian is

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

The complete baseline-generation and single-chain fitting workflow is in
[`../auto/LOCAL_COORDINATION_SATURATION.md`](../auto/LOCAL_COORDINATION_SATURATION.md).
