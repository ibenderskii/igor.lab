# Local cooperativity and the restored intra/inter contact split

Two changes to `remd_multichain.py`. Both are additive: the default behaviour of
every existing run is unchanged, and all 358 tests in the Aggregation Project
pass.

---

## 1. `hs` regains independent `lambda_intra` / `lambda_inter`

### What was wrong

`hs` had been grouped with `saturating_cooperative_contact` into a "label-blind"
set that forces `lambda_intra == lambda_inter`:

```
hs                              lambda=(1,0) -> ValueError: require lambda_intra == lambda_inter
saturating_cooperative_contact  lambda=(1,0) -> ValueError: ...
hs_m2_hs                        lambda=(1,0) -> u = -0.2275
```

The grouping is only justified for a model that is **nonlinear** in the contact
count: there, `m_intra` and `m_inter` cannot be given separate weights and still
be fed through one function of `m_total`. `hs` is **linear**, so

```
u = b(T) * [lambda_intra * m_intra + lambda_inter * m_inter]
```

is perfectly well defined for any pair of scales. Forcing them equal made pilot
Models B (`g_inter = 0`) and C (`g_intra = 0`) of AggregationPlan §15
unreachable — the two controls that isolate intrachain collapse from interchain
association. Routing `hs` through two `u_contact` evaluations per proposal also
cost it bit-for-bit equivalence with the historical linear arithmetic.

### The fix

`LABEL_BLIND_CONTACT_MODELS` now contains only
`saturating_cooperative_contact`. `hs` (and every other linear model) takes the
exact historical `b(T)*(l_i*d_intra + l_e*d_inter)` path. The metadata follows:
`nonlinear_contact_scope = "linear_intra_inter_split"`,
`interchain_contact_model = "same_linear_coefficient_independent_lambda"`.

The three controls now run and are clearly distinguishable (M=3, N=10, L=12,
15 cycles, seed 17):

| control | lambda | ⟨m_intra⟩ lane 0 | ⟨m_intra⟩ lane 3 | ⟨u⟩ lane 0 |
|---|---|---|---|---|
| A equal | (1, 1) | 4.47 | 7.67 | −0.490 |
| B collapse-only | (1, 0) | 3.13 | 4.93 | −0.329 |
| C association-only | (0, 1) | 3.07 | 3.27 | −0.049 |

The saturating model is untouched: it still rejects unequal scales, under both
cooperativity rules.

---

## 2. The globally coupled saturating potential

### Diagnosis

The multichain rule was

```
u = lambda * u_contact(m_total; M*N),   u_contact(m;n) = b(T)*m - n*A0*q^2/(1+(q/q_sat)^2),  q = m/n
```

Expanding the cooperative term at small `q` gives **`-A0 * m_total^2 / (M*N)`** —
an all-to-all coupling between every pair of contacts in the box, of Curie–Weiss
form. Two consequences, both measured:

**It is not additive over non-interacting subsystems.** Four chains split into
two well-separated pairs that share no contacts:

```
global:  u(whole) - [u(A) + u(B)] = +0.035617
local :  u(whole) - [u(A) + u(B)] = -0.000000
```

**It cannot see where the contacts are.** At fixed `m_total = 80` (M=8, N=44):

| allocation | `u` under the global rule | correct additive sum |
|---|---|---|
| 8 chains × 10 contacts | −13.4249 | −13.4249 |
| 2 chains × 40 contacts | −13.4249 | −11.7245 |
| 1 chain × 80 contacts | −13.4249 | −9.6290 |

One fully collapsed chain and seven coils has exactly the same contact energy as
eight uniformly half-collapsed chains. Because `g(q)` is concave past its
inflection, the global evaluation hands out a bonus for concentrating contacts
that the additive treatment does not — a spurious long-range attraction pointing
straight at condensation. Any aggregation transition observed under it is a
mean-field artifact of the coupling, not evidence that the transferred
single-chain physics predicts aggregation.

### The fix: cooperativity in the per-monomer contact degree

`--cooperativity local` replaces the box-average density with each monomer's own
nonbonded contact degree `k_i`, which counts intrachain **and** interchain
neighbours alike, so the two boost each other:

```
u = lambda * [ b(T)*m_total - A0 * sum_i g(k_i) ],
g(k) = kappa^2 / (1 + (kappa/q_sat)^2),    kappa = k/2
```

The factor of ½ is forced, not fitted. Every contact contributes one incidence
to each endpoint, so `sum_i k_i = 2*m_total` exactly, making `kappa` the local
analogue of `q = m/N`.

**Why no refit is needed.** When every degree is equal, `k_i = 2m/N`, then
`kappa = q` and `sum_i g(k_i) = N*q^2/(1+(q/q_sat)^2)` — the fitted single-chain
cooperative term, *exactly*, with the same `A0` and the same `q_sat`. The local
rule is the strictly-local generalisation whose mean-field limit is the fit.

**Properties, all under test:**

| property | status |
|---|---|
| `sum_i k_i == 2 * m_total` | exact, asserted in the debug oracle |
| uniform degree → fitted single-chain term | exact to 1e-12, all N and m |
| `A0 = 0` → `u == b(T)*m_total` | exact (`==`, hs nesting preserved) |
| additive over separated chains | exact to float precision |
| bounded: `g(k) <= q_sat^2` | same saturation as the fit |
| `k_i <= 5` on a cubic lattice | interior beads have 2 bonded neighbours |
| O(moved) delta vs full recount | max 3.3e-16 over 5046 moves, all 6 move types |
| sweep `du` vs full potential difference | max 8.9e-16 over 3409 moves |
| swap == manual four-potential calculation | exact |
| serial vs 2-worker | bit-identical |

**One implementation trap worth recording.** The first version summed `g(k_i)`
by iterating `site_owner`. The multiprocessing path *rebuilds* `site_owner` from
coordinates while the serial path updates it incrementally, so the two dicts
have different iteration orders for the same physical state — different
floating-point summation order — and serial/parallel runs diverged. Both
`cooperative_sum` and `delta_cooperative_sum` now accumulate **integer** counts
per degree (0..6) and contract with `g` in fixed ascending order, which is
order-independent by construction and faster besides.

### Scope

- Default is `global`. Nothing existing changes; `saturating_cooperative_model_spec.json` still describes the default rule accurately.
- `local` is refused for any model without `A0`/`q_sat`, and unequal lambdas are still refused under both rules.
- Implemented at the *multichain* level, not as a new registry model. The choice of multichain rule is a multichain decision; the single-chain fitters, the model registry, and the contract-parity machinery are untouched.
- Fitting: the local form is not a function of `m` alone, so it cannot be fitted by contact-number histogram reweighting against the existing athermal baseline. It transfers the mean-field parameters. A dedicated fit would need a baseline joint in `sum_i g(k_i)`; that is a separate piece of work and is not claimed here.

### Smoke run (M=4, N=12, L=14, 60 cycles)

Both rules run stably with comparable swap rates (0.67–0.87), and the local rule
preserves the LCST-like direction — contacts rise and Rg falls with temperature:

| T | global: m_intra / m_inter / Rg | local: m_intra / m_inter / Rg |
|---|---|---|
| 305 | 9.27 / 1.90 / 1.633 | 8.50 / 1.47 / 1.618 |
| 320 | 10.30 / 1.67 / 1.597 | 9.13 / 1.77 / 1.597 |
| 335 | 11.03 / 2.17 / 1.595 | 11.37 / 2.10 / 1.548 |
| 350 | 11.23 / 3.83 / 1.577 | 11.00 / 3.20 / 1.536 |

These are short equilibration-free runs — a smoke test that the sampler is
healthy, not a physics result.
