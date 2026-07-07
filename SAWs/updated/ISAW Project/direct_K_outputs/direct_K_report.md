# Direct-K crossover diagnostic — N30_direct_K_extended_v1

- **N (beads):** 30
- **K ladder:** -0.4, -0.32, -0.24, -0.16, -0.08, 0, 0.08, 0.14, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6
- **Seeds:** 101, 202 (2 independent)
- **Control mode:** direct_K (P(C|K) ∝ exp[K·m]; b = −K; no temperature mapping)

## Answers

1. **Did the run complete?** Yes (2 seed(s) sampled and extracted).
2. **Was the transition bracketed?** NO (16 lanes below and 0 above the contact-variance peak).
3. **Largest contact fluctuation (Var(m)) at:** K=0.6 (endpoint).
4. **Strongest size response (−d⟨Rg²⟩/dK) at:** K=0.45.
5. **Do observables agree?** partial/no. Peaks: contact-variance K=0.6 (endpoint), d⟨m⟩/dK K=0.5, size K=0.45, network K=0.55; consensus K ≈ 0.5.
6. **Are ESS, swap rates, and round trips adequate?** No — min ESS(contacts) 948.0 (≥200? ok), swap∈[0.87,0.91] (out-of-band), min round trips 739 (ok), min adjacent overlap 0.99 (ok).
7. **Next scan:** contact-variance peak is at the highest K=0.6; extend the K range HIGHER.

## Fluctuation–response check

d⟨m⟩/dK should equal Var(m). Compared 17 lanes: max |Δ| = 1.59, mean |Δ| = 0.538, correlation = 0.994.

## Peak estimates

| Estimator | K peak | at endpoint |
| --- | --- | --- |
| K_peak_contact_variance | 0.6 | yes |
| K_peak_contact_derivative | 0.5 | no |
| K_peak_Rg2_response | 0.45 | no |
| K_peak_network_response | 0.55 | no |

Consensus K (mean of interior peaks): **0.5**.

See `direct_K_report.json` and `direct_K_response_curves.csv` for the full per-lane numbers.
