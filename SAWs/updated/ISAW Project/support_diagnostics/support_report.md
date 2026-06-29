# Support-mismatch diagnostic report

- Target: `.\remd_distributions_44mer.npz`
- Baseline: `.\single_uniform_chain2_athermal_dists_joint_N44_T1_seed42.npz`
- Temperatures: 64 in [280, 360]
- Contact offsets: 43.0, 44.0
- Positive-support threshold: 0
- Baseline integer contact range: [0, 31]
- Baseline internal gaps (P0 <= threshold inside range): 28, 29, 30

Reweighting `P_model(m|T) ∝ P0(m)·exp[-b(T)·m]` cannot place mass at any contact where `P0(m)=0`. Target mass that is below the baseline range, above it, or in an internal gap is therefore unreachable by any `b(T)` and bounds the fit error from below.

## Per-offset contact support (averaged over temperatures)

| offset | mean unsup | max unsup | mean below | mean gap | mean above | mean pos-support | mean geom-support | max neg mass |
|---|---|---|---|---|---|---|---|---|
| 43 | 0.06181 | 0.1343 | 0.0003508 | 0.03292 | 0.02853 | 0.9382 | 0.9711 | 0.001 |
| 44 | 0.05238 | 0.1141 | 0.002273 | 0.0275 | 0.02261 | 0.9476 | 0.9751 | 0.004665 |

`mean geom-support` − `mean pos-support` equals the internal-gap mass: geometric support counts everything inside [min, max]; positive support counts only bins with `P0 > threshold`.

## Rg support (offset-independent)

- Scaled baseline Rg range: [0.8709, 3.331]  (rg_scale = 0.463205)
- Mean total unsupported Rg mass: 0.001508
- Max total unsupported Rg mass: 0.004997
- Mean below / internal-gap / above: 0 / 0.001508 / 0

## Best offset by criterion (reported separately)

- **Smallest mean unsupported contact mass:** 44
- **Smallest maximum unsupported contact mass:** 44
- **No negative shifted-contact support:** none — every offset places some target mass at negative shifted contacts.
- **Best Rg support:** 43 (Rg support does not depend on the contact offset; all offsets are equivalent under this criterion.)

These criteria are intentionally not combined into a single score; choose the offset that matches your modeling priority.
