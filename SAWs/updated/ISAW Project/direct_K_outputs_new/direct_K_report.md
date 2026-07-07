# Direct-K crossover diagnostic — N30_direct_K_refined_v1

- **N (beads):** 30
- **K ladder:** 0.4, 0.45, 0.5, 0.53, 0.55, 0.57, 0.59, 0.6, 0.61, 0.63, 0.66, 0.7, 0.75
- **Seeds:** 101, 202 (2 independent)
- **Control mode:** direct_K (P(C|K) ∝ exp[K·m]; b = −K; no temperature mapping)

## Answers

1. **Did the run complete?** Yes (2 seed(s) sampled and extracted).
2. **Was the transition bracketed?** YES (7 lanes below and 5 above the contact-variance peak).
3. **Largest contact fluctuation (Var(m)) at:** K=0.6.
4. **Strongest size response (−d⟨Rg²⟩/dK) at:** K=0.61.
5. **Do observables agree?** yes. Peaks: contact-variance K=0.6, d⟨m⟩/dK K=0.6, size K=0.61, network K=0.59; consensus K ≈ 0.6.
6. **Are ESS, swap rates, and round trips adequate?** No — min ESS(contacts) 1144.3 (≥200? ok), swap∈[0.87,0.98] (out-of-band), min round trips 1686 (ok), min adjacent overlap 0.99 (ok).
7. **Next scan:** interior peak near K=0.6; refine the ladder around the interior peak.

## Fluctuation–response check

d⟨m⟩/dK should equal Var(m). Compared 13 lanes: max |Δ| = 3.83, mean |Δ| = 1.78, correlation = 0.527.

## Peak estimates

| Estimator | K peak | at endpoint |
| --- | --- | --- |
| K_peak_contact_variance | 0.6 | no |
| K_peak_contact_derivative | 0.6 | no |
| K_peak_Rg2_response | 0.61 | no |
| K_peak_network_response | 0.59 | no |

Consensus K (mean of interior peaks): **0.6**.

See `direct_K_report.json` and `direct_K_response_curves.csv` for the full per-lane numbers.
