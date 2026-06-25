# Rg mapping diagnostic report

- Production scalar mapping: `Rg_obs = 0.463205033126 * Rg_lat`
- Bootstrap reps: 60, seed 123, confidence 0.95
- Split schemes (blocked + interleaved): blocked_low,blocked_mid,blocked_high,every_third_phase

> **Guardrail.** A temperature-dependent geometric scale is physically suspicious — a genuine unit conversion is constant. A nonconstant s(T) more likely reflects contact-model misspecification than a real conversion. The constant mapping is preferred unless a richer mapping improves held-out Rg JS on *every* split and by a practically meaningful margin.

## hs:boot_smoke  (model: hs)

- Per-temperature optimal scale s(T): mean **0.46357**, std 0.00337, range [0.45947, 0.47418]
- Production scalar 0.463205 vs s(T) mean 0.463573: relative offset +0.1%
- Median JS at optimal scale: 0.01162
- Temperature-linear slope s1 = -0.003438 (CI [-0.004821, 0.005233]); distinguishable from zero: **False**; relative scale change across T range ~ 0.74%
- Held-out improvement, affine vs constant: mean +4.95e-05 JS/temp (+0.4%), improves all splits: False
- Held-out improvement, tlinear_scale vs constant: mean -5.68e-05 JS/temp (-0.3%), improves all splits: False

**Recommendation: constant multiplicative.**

The temperature-linear scale does not clear the stability + meaningfulness bar, so the constant mapping is retained.

## hs_quadratic:split_sens_run  (model: hs_quadratic)

- Per-temperature optimal scale s(T): mean **0.46372**, std 0.00358, range [0.45953, 0.47424]
- Production scalar 0.463205 vs s(T) mean 0.463722: relative offset +0.1%
- Median JS at optimal scale: 0.0116
- Temperature-linear slope s1 = -0.002728 (CI [-0.003491, 7.065e-05]); distinguishable from zero: **False**; relative scale change across T range ~ 0.59%
- Held-out improvement, affine vs constant: mean +2.01e-05 JS/temp (+0.1%), improves all splits: False
- Held-out improvement, tlinear_scale vs constant: mean -2.43e-05 JS/temp (-0.2%), improves all splits: False

**Recommendation: constant multiplicative.**

The temperature-linear scale does not clear the stability + meaningfulness bar, so the constant mapping is retained.

## Verdict on the production scalar

- Production scalar: `0.463205033126`
- Mean optimal s(T) across models: 0.463648 (per-model means: 0.46357, 0.46372)
- **No model shows stable, meaningful evidence for a temperature-dependent or affine mapping.** The constant scalar is adequate; Phase B integration is NOT warranted at this time.
