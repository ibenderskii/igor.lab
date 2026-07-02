# Model suite report

Generated: 2026-06-30T22:08:11.522545+00:00

## Configuration

- Target REMD: `/scratch/midway2/ibenderskii/auto/remd_distributions_30mer.npz`
- Models: hs, hs_quadratic, poly2, poly3, heat_capacity
- Baselines: final_joint_baseline_30mer
- Output root: `/scratch/midway2/ibenderskii/auto/model_suite_output_30mer`

## Jobs

- Fits succeeded: 5; fits failed/incomplete: 0
- REMD seeds succeeded: 50; failed: 0

## Baseline: final_joint_baseline_30mer

- **Simulation winner:** heat_capacity (all-temperature combined JS 0.112)
- **Analytic-fit winner:** poly3 (all-temperature contact loss 3.177)

### Analytic fit ranking

| rank | model | val contact loss | all contact loss |
|---|---|---|---|
| 1 | poly3 | n/a | 3.177 |
| 2 | poly2 | n/a | 3.177 |
| 3 | hs_quadratic | n/a | 3.177 |
| 4 | heat_capacity | n/a | 3.177 |
| 5 | hs | n/a | 3.178 |

### REMD simulation ranking

| rank | model | combined JS (mean±std) | contact JS | Rg JS | note |
|---|---|---|---|---|---|
| 1 | heat_capacity | 0.112±0.000299 | 0.07428 | 0.07546 |  |
| 2 | poly3 | 0.1121±0.0003455 | 0.07426 | 0.07578 |  |
| 3 | poly2 | 0.1122±0.0006859 | 0.07452 | 0.07536 |  |
| 4 | hs | 0.1123±0.0004156 | 0.07428 | 0.076 | within ~1e-3 of rank-1 with fewer parameters (2 vs 3); consider preferring on parsimony |
| 5 | hs_quadratic | 0.1125±0.00058 | 0.07455 | 0.07587 |  |

### Diagnostics

- Possible overfitting (val ≫ train fit loss): none
- REMD does not reproduce analytic prediction (REMD↔fit contact JS > 0.05): none
- Convergence warnings (min swap rate < 0.05): none

### Convergence diagnostics

_Raw ranks above are unchanged; the status column below qualifies whether a low JS is trustworthy. A model flagged **unreliable** failed one or more convergence thresholds and its score should not be trusted regardless of rank._

| model | status | min ESS | max τ | round trips | min coverage | max drift | max Rhat | flags |
|---|---|---|---|---|---|---|---|---|
| hs | ok | 892.2 | 1.681 | 1.412e+04 | 1 | 0.1459 | 1.001 |  |
| hs_quadratic | ok | 858.8 | 1.747 | 1.424e+04 | 1 | 0.1544 | 1.001 |  |
| poly2 | ok | 433.6 | 3.459 | 1.429e+04 | 1 | 0.1726 | 1.001 |  |
| poly3 | ok | 765.3 | 1.96 | 1.447e+04 | 1 | 0.2071 | 1 |  |
| heat_capacity | ok | 960 | 1.562 | 1.421e+04 | 1 | 0.2082 | 1.001 |  |

### Statistical model comparison (paired across seeds)

- **Raw winner (lowest mean combined JS):** heat_capacity
- **Statistically supported differences:** none survive multiple-testing correction at this sample size
- **Practically equivalent pairs (P|delta|<eps >= 0.5):** hs~hs_quadratic (P=1), hs~poly2 (P=0.9993), hs~poly3 (P=1), hs~heat_capacity (P=1), hs_quadratic~poly2 (P=0.9808), hs_quadratic~poly3 (P=0.9968), hs_quadratic~heat_capacity (P=0.9903), poly2~poly3 (P=1), poly2~heat_capacity (P=0.9999), poly3~heat_capacity (P=1)
- **Parsimonious recommendation:** hs (hs is statistically/practically indistinguishable from the best model heat_capacity but has fewer parameters)
- **Convergence-qualified recommendation:** hs passes convergence diagnostics (status=reliable).

## Fit robustness and parameter identifiability

_Fitter-side diagnostics. These qualify but never replace the REMD simulation ranking. Bootstrap intervals are empirical temperature-resampling ranges, not likelihood standard errors._

### Baseline: final_joint_baseline_30mer

| model | bootstrap success | widest rel CI | max |corr| | split stability | Rg-weight | optimizer/identifiability |
|---|---|---|---|---|---|---|
| hs | 200/200 | 0.047 | 0.9986 | stable | stable | warn |
| hs_quadratic | 200/200 | 0.37 | 0.9955 | stable | sensitive | warn |
| poly2 | 200/200 | 0.17 | 0.7641 | stable | sensitive | warn |
| poly3 | 200/200 | 0.98 | 0.9134 | moderate | sensitive | warn |
| heat_capacity | 200/200 | 0.58 | 0.9998 | stable | sensitive | warn |

#### Bootstrap uncertainty

- **hs**: strong parameter correlation(s): h~s=0.9986 (possible non-identifiability).
- **hs_quadratic**: strong parameter correlation(s): h~s=0.9955 (possible non-identifiability).
- **poly2**: parameters appear well constrained; no intervals bracket zero and no strong correlations flagged.
- **poly3**: strong parameter correlation(s): a1~a3=-0.9134 (possible non-identifiability).
- **heat_capacity**: strong parameter correlation(s): dh0~ds0=0.9998 (possible non-identifiability).

#### Validation-split sensitivity

- **hs**: stable across 16/16 successful splits (max parameter CV 0.017); held-out combined loss range 0.02125.
- **hs_quadratic**: stable across 16/16 successful splits (max parameter CV 0.16); held-out combined loss range 0.02141.
- **poly2**: stable across 16/16 successful splits (max parameter CV 0.06); held-out combined loss range 0.02146.
- **poly3**: moderate across 16/16 successful splits (max parameter CV 0.41); held-out combined loss range 0.02146.
- **heat_capacity**: stable across 16/16 successful splits (max parameter CV 0.18); held-out combined loss range 0.0214.

_Estimates are considered stable when parameter estimates and ranking change little across interpolation, blocked-low/-mid/-high, k-fold, and random holdouts that were enabled._

#### Rg-weight sensitivity

- **hs**: production weight 0.5 lies on the Pareto frontier; heuristic knee at weight 1; conclusions are robust to the weight.
- **hs_quadratic**: production weight 0.5 lies on the Pareto frontier; heuristic knee at weight 1; conclusions change materially with the weight.
- **poly2**: production weight 0.5 lies on the Pareto frontier; heuristic knee at weight 1; conclusions change materially with the weight.
- **poly3**: production weight 0.5 lies on the Pareto frontier; heuristic knee at weight 1; conclusions change materially with the weight.
- **heat_capacity**: production weight 0.5 lies on the Pareto frontier; heuristic knee at weight 1; conclusions change materially with the weight.

### Recommendation qualified by robustness

- **final_joint_baseline_30mer:** primary REMD pick is **heat_capacity** (combined JS 0.112); robustness-qualified recommendation: **hs** (5 model(s) are within ~1e-3 of the best REMD score; among them, **hs** is preferred on stability/identifiability/parsimony).

## Global cross-baseline ranking

_Not produced: baselines differ in offset/Rg-units/metric config, or only one baseline. Per-baseline rankings above apply._

## Outputs

- `/scratch/midway2/ibenderskii/auto/model_suite_output_30mer/comparison/model_comparison.csv`
- `/scratch/midway2/ibenderskii/auto/model_suite_output_30mer/comparison/per_temperature_metrics.csv`
- `/scratch/midway2/ibenderskii/auto/model_suite_output_30mer/comparison/plots`
- `/scratch/midway2/ibenderskii/auto/model_suite_output_30mer/manifest.json`
- `/scratch/midway2/ibenderskii/auto/model_suite_output_30mer/comparison/pairwise_model_comparison.csv`
- `/scratch/midway2/ibenderskii/auto/model_suite_output_30mer/comparison/seed_level_model_scores.csv`
- `/scratch/midway2/ibenderskii/auto/model_suite_output_30mer/comparison/model_rank_stability.csv`
- `/scratch/midway2/ibenderskii/auto/model_suite_output_30mer/comparison/model_statistics_summary.json`
- `/scratch/midway2/ibenderskii/auto/model_suite_output_30mer/comparison/fit_robustness_summary.json`
- `/scratch/midway2/ibenderskii/auto/model_suite_output_30mer/comparison/fit_robustness.csv`
