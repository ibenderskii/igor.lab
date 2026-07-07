# Direct-K crossover diagnostic — N30_direct_K_high_v2

- **N (beads):** 30
- **K ladder:** 0, 0.2, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.05
- **Seeds:** 101, 202 (2 independent)
- **Control mode:** direct_K (P(C|K) ∝ exp[K·m]; b = −K; no temperature mapping)

## Answers

1. **Did the run complete?** Yes (2 seed(s) sampled and extracted).
2. **Was the transition bracketed?** YES (4 lanes below and 5 above the contact-variance peak).
3. **Largest contact fluctuation (Var(m)) at:** K=0.55.
4. **Strongest size response (−d⟨Rg²⟩/dK) at:** K=0.55.
5. **Do observables agree?** yes. Peaks: contact-variance K=0.55, d⟨m⟩/dK K=0.55, size K=0.55, network K=0.55; consensus K ≈ 0.55.
6. **Are ESS, swap rates, and round trips adequate?** Yes — min ESS(contacts) 466.2 (≥200? ok), swap∈[0.64,0.79] (ok), min round trips 402 (ok), min adjacent overlap 0.94 (ok).
7. **Next scan:** interior peak near K=0.55; refine the ladder around the interior peak.

## Fluctuation–response check

d⟨m⟩/dK should equal Var(m). Compared 10 lanes: max |Δ| = 2.73, mean |Δ| = 0.795, correlation = 0.986.

## Peak estimates

| Estimator | K peak | at endpoint |
| --- | --- | --- |
| K_peak_contact_variance | 0.55 | no |
| K_peak_contact_derivative | 0.55 | no |
| K_peak_Rg2_response | 0.55 | no |
| K_peak_network_response | 0.55 | no |

Consensus K (mean of interior peaks): **0.55**.

See `direct_K_report.json` and `direct_K_response_curves.csv` for the full per-lane numbers.
