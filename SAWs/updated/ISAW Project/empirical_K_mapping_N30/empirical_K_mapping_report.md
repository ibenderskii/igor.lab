# Empirical molecular-temperature -> direct-K mapping

This diagnostic maps each molecular REMD temperature to the sampled direct lattice coupling K that best reproduces its contact and Rg marginals. The combined score is a **weighted sum of marginal divergences**, not a joint-distribution fit. Results are reported at three levels: the **raw grid result** (every temperature), the **resolved interior result** (optima strictly inside the sampled-K range), and the **boundary-censored result** (optima pinned at an endpoint, where the true optimum is not bracketed).

## Raw grid results (full dataset, transparency)

All 6 direct-K file(s) validated (control_mode=direct_K, temperature_mapping_applied=false, chain length N=30, finite normalized histograms). 29 unique sampled K span [-0.4, 1.05]; the raw combined best-K spans [-0.32, 0.14] with Spearman(T, K_best) = 0.977 and 1 downward step(s) (no monotonicity imposed).

Raw contact-vs-Rg disagreement: 64/64 temperatures beyond threshold 0.1 (mean |contact-K - Rg-K| = 0.2363, max = 0.32). Raw temperatures in transition [0.58, 0.66]: 0. Raw old-mapping MAE = 0.09303, RMS = 0.1002 (these include censored rows and are NOT the scientific figure).

## Resolved interior results (scientifically usable)

**Q1. Contact-only optima at boundaries:** 11/64 (17.2%; 11 low, 0 high).

**Q2. Rg-only optima at boundaries:** 0/64 (0.0%; 0 low, 0 high).

**Q3. Combined optima at boundaries:** 0/64 (0.0%; 0 low, 0 high).

**Q4. Temperatures with both component optima resolved:** 53/64 (11 unresolved).

**Q5. Do contacts and Rg agree among resolved temperatures?** 
53/53 resolved pairs disagree beyond 0.1 (fraction 1.000, mean 0.2189, max 0.32) -> they **disagree** among resolved temperatures.

**Q6. Combined interior optima comparable with the old K(T):** 
64 resolved temperature(s); resolved MAE = 0.09303, RMS = 0.1002 (K_old(T) = s - h/T with h=647.7, s=1.874).

**Q7. Does the resolved mapping reach the transition interval?** 
No resolved temperature maps into [0.58, 0.66]; resolved combined best-K spans [-0.32, 0.14].

**Q8. Next action:** consider a richer Hamiltonian.

## Recommendation

- Among resolved temperatures, contacts and Rg disagree on the preferred K: **consider a second structural coordinate or a richer Hamiltonian** -- one coupling cannot match both marginals.
- Resolved interior combined optima are bracketed yet never reach the transition interval [0.58, 0.66]: **the molecular data map to partial compactification** rather than the intrinsic lattice crossover.
