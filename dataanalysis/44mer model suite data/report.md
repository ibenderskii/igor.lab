# Model suite report

Generated: 2026-06-30T00:49:40.787597+00:00

## Configuration

- Target REMD: `/scratch/midway2/ibenderskii/auto/remd_distributions_44mer.npz`
- Models: hs, hs_quadratic, poly2, poly3, heat_capacity
- Baselines: final_joint_baseline_44mer
- Output root: `/scratch/midway2/ibenderskii/auto/model_suite_output_44mer`

## Jobs

- Fits succeeded: 5; fits failed/incomplete: 0
- REMD seeds succeeded: 50; failed: 0

## Baseline: final_joint_baseline_44mer

- **Simulation winner:** poly3 (all-temperature combined JS 0.08742)
- **Analytic-fit winner:** poly3 (all-temperature contact loss 3.202)

### Analytic fit ranking

| rank | model | val contact loss | all contact loss |
|---|---|---|---|
| 1 | poly3 | n/a | 3.202 |
| 2 | heat_capacity | n/a | 3.202 |
| 3 | hs_quadratic | n/a | 3.203 |
| 4 | poly2 | n/a | 3.204 |
| 5 | hs | n/a | 3.246 |

### REMD simulation ranking

| rank | model | combined JS (mean±std) | contact JS | Rg JS | note |
|---|---|---|---|---|---|
| 1 | poly3 | 0.08742±0.0006943 | 0.07629 | 0.02227 |  |
| 2 | heat_capacity | 0.08743±0.0005943 | 0.07616 | 0.02254 | within ~1e-3 of rank-1 with fewer parameters (3 vs 4); consider preferring on parsimony |
| 3 | poly2 | 0.08753±0.0005206 | 0.07645 | 0.02216 | within ~1e-3 of rank-1 with fewer parameters (3 vs 4); consider preferring on parsimony |
| 4 | hs_quadratic | 0.08776±0.0004469 | 0.07653 | 0.02246 | within ~1e-3 of rank-1 with fewer parameters (3 vs 4); consider preferring on parsimony |
| 5 | hs | 0.08843±0.0007639 | 0.07713 | 0.02261 |  |

### Diagnostics

- Possible overfitting (val ≫ train fit loss): none
- REMD does not reproduce analytic prediction (REMD↔fit contact JS > 0.05): none
- Convergence warnings (min swap rate < 0.05): none

### Convergence diagnostics

_Raw ranks above are unchanged; the status column below qualifies whether a low JS is trustworthy. A model flagged **unreliable** failed one or more convergence thresholds and its score should not be trusted regardless of rank._

| model | status | min ESS | max τ | round trips | min coverage | max drift | max Rhat | flags |
|---|---|---|---|---|---|---|---|---|
| hs | ok | 692.7 | 2.165 | 1.227e+04 | 1 | 0.1954 | 1 |  |
| hs_quadratic | ok | 859 | 1.746 | 1.228e+04 | 1 | 0.1501 | 1.001 |  |
| poly2 | ok | 906.5 | 1.655 | 1.236e+04 | 1 | 0.1753 | 1.001 |  |
| poly3 | ok | 749.7 | 2.001 | 1.193e+04 | 1 | 0.1748 | 1.001 |  |
| heat_capacity | ok | 842.1 | 1.781 | 1.225e+04 | 1 | 0.1926 | 1.001 |  |

### Statistical model comparison (paired across seeds)

- **Raw winner (lowest mean combined JS):** poly3
- **Statistically supported differences (Holm p < alpha):** hs vs poly3 (Holm p=0.03516), hs vs heat_capacity (Holm p=0.01953)
- **Practically equivalent pairs (P|delta|<eps >= 0.5):** hs~hs_quadratic (P=0.8352), hs~poly2 (P=0.6124), hs~poly3 (P=0.5072), hs~heat_capacity (P=0.5264), hs_quadratic~poly2 (P=0.9992), hs_quadratic~poly3 (P=0.9942), hs_quadratic~heat_capacity (P=0.9913), poly2~poly3 (P=0.9999), poly2~heat_capacity (P=1), poly3~heat_capacity (P=0.996)
- **Parsimonious recommendation:** heat_capacity (heat_capacity is statistically/practically indistinguishable from the best model poly3 but has fewer parameters)
- **Convergence-qualified recommendation:** heat_capacity passes convergence diagnostics (status=reliable).

## Fit robustness and parameter identifiability

_Fitter-side diagnostics. These qualify but never replace the REMD simulation ranking. Bootstrap intervals are empirical temperature-resampling ranges, not likelihood standard errors._

### Baseline: final_joint_baseline_44mer

| model | bootstrap success | widest rel CI | max |corr| | split stability | Rg-weight | optimizer/identifiability |
|---|---|---|---|---|---|---|
| hs | 200/200 | 0.13 | 0.9982 | stable | stable | ok |
| hs_quadratic | 200/200 | 0.14 | 0.9955 | stable | stable | ok |
| poly2 | 200/200 | 0.12 | 0.6671 | stable | stable | warn |
| poly3 | 200/200 | 0.76 | 0.9495 | moderate | stable | warn |
| heat_capacity | 200/200 | 0.11 | 0.9938 | stable | stable | warn |

#### Bootstrap uncertainty

- **hs**: strong parameter correlation(s): h~s=0.9982 (possible non-identifiability).
- **hs_quadratic**: strong parameter correlation(s): h~s=0.9955 (possible non-identifiability).
- **poly2**: parameters appear well constrained; no intervals bracket zero and no strong correlations flagged.
- **poly3**: strong parameter correlation(s): a1~a3=-0.9495 (possible non-identifiability).
- **heat_capacity**: strong parameter correlation(s): dh0~ds0=0.9938 (possible non-identifiability).

#### Validation-split sensitivity

- **hs**: stable across 16/16 successful splits (max parameter CV 0.047); held-out combined loss range 0.004275.
- **hs_quadratic**: stable across 16/16 successful splits (max parameter CV 0.087); held-out combined loss range 0.006764.
- **poly2**: stable across 16/16 successful splits (max parameter CV 0.069); held-out combined loss range 0.006667.
- **poly3**: moderate across 16/16 successful splits (max parameter CV 0.23); held-out combined loss range 0.007124.
- **heat_capacity**: stable across 16/16 successful splits (max parameter CV 0.052); held-out combined loss range 0.007165.

_Estimates are considered stable when parameter estimates and ranking change little across interpolation, blocked-low/-mid/-high, k-fold, and random holdouts that were enabled._

#### Rg-weight sensitivity

- **hs**: production weight 0.5 lies on the Pareto frontier; heuristic knee at weight 0.5; conclusions are robust to the weight.
- **hs_quadratic**: production weight 0.5 lies on the Pareto frontier; heuristic knee at weight 0.5; conclusions are robust to the weight.
- **poly2**: production weight 0.5 lies on the Pareto frontier; heuristic knee at weight 0.5; conclusions are robust to the weight.
- **poly3**: production weight 0.5 lies on the Pareto frontier; heuristic knee at weight 0.5; conclusions are robust to the weight.
- **heat_capacity**: production weight 0.5 lies on the Pareto frontier; heuristic knee at weight 0.5; conclusions are robust to the weight.

### Recommendation qualified by robustness

- **final_joint_baseline_44mer:** primary REMD pick is **poly3** (combined JS 0.08742); robustness-qualified recommendation: **heat_capacity** (4 model(s) are within ~1e-3 of the best REMD score; among them, **heat_capacity** is preferred on stability/identifiability/parsimony).

## Global cross-baseline ranking

_Not produced: baselines differ in offset/Rg-units/metric config, or only one baseline. Per-baseline rankings above apply._

## Outputs

- `/scratch/midway2/ibenderskii/auto/model_suite_output_44mer/comparison/model_comparison.csv`
- `/scratch/midway2/ibenderskii/auto/model_suite_output_44mer/comparison/per_temperature_metrics.csv`
- `/scratch/midway2/ibenderskii/auto/model_suite_output_44mer/comparison/plots`
- `/scratch/midway2/ibenderskii/auto/model_suite_output_44mer/manifest.json`
- `/scratch/midway2/ibenderskii/auto/model_suite_output_44mer/comparison/pairwise_model_comparison.csv`
- `/scratch/midway2/ibenderskii/auto/model_suite_output_44mer/comparison/seed_level_model_scores.csv`
- `/scratch/midway2/ibenderskii/auto/model_suite_output_44mer/comparison/model_rank_stability.csv`
- `/scratch/midway2/ibenderskii/auto/model_suite_output_44mer/comparison/model_statistics_summary.json`
- `/scratch/midway2/ibenderskii/auto/model_suite_output_44mer/comparison/fit_robustness_summary.json`
- `/scratch/midway2/ibenderskii/auto/model_suite_output_44mer/comparison/fit_robustness.csv`
