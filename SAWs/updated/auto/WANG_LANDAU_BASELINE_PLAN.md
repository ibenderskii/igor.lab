# Single-chain Wang-Landau athermal baseline plan

## 1. Scientific target

For an `N`-bead cubic-lattice self-avoiding walk (SAW), let `X` denote a
conformation and let `m(X)` be its number of non-bonded nearest-neighbour
contacts.  The baseline required by the fitter is still the uniform athermal
measure

\[
P_0(X)=\frac{1}{|\Omega_N|},
\]

not a flat distribution in contact number.  Its joint observable distribution
is

\[
P_0(m,R_g)=\frac{g(m,R_g)}{\sum_{m,R_g}g(m,R_g)},
\]

where `g(m,Rg)` counts conformations in a contact and radius-of-gyration bin.

Wang-Landau is used as an importance-sampling device to visit rare compact
contact levels.  It does not redefine the physical baseline.

## 2. Markov-chain validity

The script uses the same three move families as the direct athermal sampler:

1. proper cubic pivot rotations of the tail;
2. local 90-degree crankshaft flips;
3. symmetric end moves.

Move families are chosen with fixed probabilities, and each valid proposal has
a reverse proposal with the same probability.  The proposal kernel is therefore
symmetric.  Self-intersecting proposals are rejected.

For a frozen estimate `log_g_hat(m)`, the multicanonical conformation weight is

\[
W(X)=W(m(X))=\exp[-\log \hat g(m(X))].
\]

The Metropolis acceptance probability is consequently

\[
a(X\to X')=\min\left(1,
\exp[\log\hat g(m(X))-\log\hat g(m(X'))]\right).
\]

This satisfies detailed balance for the frozen-weight production distribution.

## 3. Two strictly separated phases

### Phase A: Wang-Landau learning

Initialize `log_g_hat(m)=0` in a declared integer window `[0,m_max]`.  After
every attempted move, accepted or rejected, update the occupied bin:

\[
\log\hat g(m_{current})\leftarrow
\log\hat g(m_{current})+\log f.
\]

The declared window has three tiers.  Tier 2 is required to reach
`wl_min_visits` and the `wl_flatness` minimum-to-mean ratio.  Tier 1 is excluded
from that ratio but must reach `wl_min_cover_visits`.  Tier 0 is outside the
declared window or is an independently verified internal gap.  The default
boundary between tiers 2 and 1 is derived from the worst-temperature molecular
target tail; the default tier-1 ceiling remains the exact geometric maximum.

With the default halving schedule, reset the histogram and halve `log(f)` after
the tier-specific checks pass.  The optional Belardinelli-Pereyra schedule uses
the cumulative Monte Carlo time `t = attempted_moves / included_levels`; its
time origin is never reset between stages.  After the initial coverage stages,
`log(f)=1/t` is updated after every attempted move without a histogram-flatness
criterion.

The adaptive samples are never used in the reported baseline.  Stage
checkpoints contain the current chain, density estimate, next modification
factor, and cumulative diagnostics.  A resumed stage restarts its flatness
histogram and random-number stream, which is statistically valid but not a
bitwise continuation.

### Phase B: fixed-weight production

Freeze `log_g_hat`.  Run independent production chains with unique seeds, a
burn-in period, and no further density updates.  Record `m`, `Rg`, and bend count
at fixed intervals in attempted Monte Carlo steps.

For production samples `X_i`, reconstruct any athermal expectation with
self-normalized importance weights

\[
q_i=\frac{\exp[\log\hat g(m_i)]}
{\sum_j\exp[\log\hat g(m_j)]}.
\]

For example,

\[
\hat P_0(m,R_g)=\sum_i q_i
\mathbf{1}[m_i=m, R_{g,i}\in R_g\text{ bin}].
\]

This estimator is consistent even when `log_g_hat` is imperfect, because the
same frozen bias used in production is removed explicitly.  DOS accuracy affects
mixing and variance, not the limiting target measure.

## 4. Contact-window policy

`m_max` must be the independently verified exact geometric contact maximum for
the selected chain length.  Proposals above that maximum are impossible, so
rejecting them does not truncate the athermal ensemble.  Using any lower ceiling
instead samples a conditional distribution rather than the full athermal
baseline.  The script uses verified maxima 30, 50, and 74 automatically for
`N=30`, `44`, and `60`; other chain lengths require an externally verified
exact maximum.

By default, every integer contact level from `0` through `m_max` remains in the
window, but only the target-supported tier must be flat.  The coverage tier is
still required to reach its minimum visit count.  A finite simulation is never
used to classify an unvisited level as geometrically unreachable.  Known
internal gaps may be supplied with `--excluded_contact_levels`, but only after
independent geometric verification.  If learning or production ever encounters
an excluded internal level, the run fails and writes no output.

Tail truncation is off by default.  It is enabled only by an explicit `m_cover`
or nonzero `cover_tail_threshold`, is printed as a declared conditional window,
and records the omitted molecular target mass.  Its omitted athermal mass
cannot be estimated from the truncated run itself.

If the requested ceiling is unreachable or an unlisted internal gap exists, the
run stops at `wl_max_steps` and reports the deficient bins.  The user must then
verify the geometry or any proposed internal gap independently.  The contact
window cannot be reduced based on finite sampling, and its endpoints cannot be
excluded.

For the current project, use the encoded exact geometric maxima.  Shifted REMD
contact support can establish the minimum support needed by a fit, but it cannot
set the ceiling of a complete athermal baseline.

## 5. Diagnostics required before accepting a production baseline

The output records:

- all Wang-Landau refinement stages and their flatness statistics;
- learning and production acceptance counts;
- learning and per-worker production round trips across the full contact window;
- per-worker reweighted mean contact and mean `Rg`;
- importance-sampling effective sample size (ESS);
- exact normalization of `P(m)`, `P(Rg)`, and `P(m,Rg)`;
- agreement of both joint-distribution marginals with their separately built
  one-dimensional distributions.

At least one summed production round trip is required by default.  Production
must also meet the tier-specific minimum counts before output is written:
`wl_min_visits` in tier 2 and `wl_min_cover_visits` in tier 1, with
`min_production_samples_per_level` retained as an optional stricter common
floor.  A flat adaptive histogram alone is not evidence of adequate
fixed-weight production mixing.

## 6. Output compatibility

The NPZ contains the athermal, reweighted versions of the existing fields:

- `c_vals`, `c_prob`;
- `c_edges`, `rg_edges`, `rg_prob`, `crg_prob`;
- `N`, `T`, `eps`, worker seeds and sampling controls;
- acceptance, worker means, bend summaries, and optional raw samples.

The stored raw `c_samples`, `rg_samples`, and `bend_samples` are obtained by
systematic importance resampling, so their semantics remain athermal rather than
multicanonical.  The weighted histograms, not the resampled arrays, are the
authoritative output.  `c_counts` is constructed from the same athermal
resampled contacts.  The raw fixed-weight visit counts are stored separately as
`production_c_counts`.  `wl_*` and importance-diagnostic fields are additive
and can be ignored by legacy readers.

## 7. Validation sequence

1. **Move invariants:** verify self-avoidance, unit bonds, 23 proper
   non-identity rotations, and exact incremental contact deltas against full
   recounts.
2. **Exact small-chain validation:** enumerate every rooted six-bead 3D SAW
   (3,534 walks), run learning plus frozen production, and compare estimated
   `P(m)` and mean `Rg` with enumeration.
3. **Schema validation:** create a short NPZ, require all legacy fields, check
   shapes and normalization, and load it through the existing fitter's baseline
   loader.
4. **Bulk cross-validation:** for the actual chain length, compare the
   Wang-Landau result with the direct athermal baseline over well-sampled contact
   bins.  Differences should be consistent with independent-run uncertainty.
5. **Tail and fit validation:** rerun `analyze_support_mismatch.py`, then refit
   with only the baseline changed.  Report how support, fitted parameters,
   residuals, and chain-length transferability change.

The script's built-in `--self-test` performs steps 1 and 2.  Steps 4 and 5 need
the project production data and should be treated as the scientific acceptance
test.

## 8. Recommended first production workflow

Pilot the 30-mer first as the end-to-end smoke test, then the 44-mer and 60-mer,
using the exact geometric `m_max` selected automatically for each chain length.
Run with checkpointing, multiple fixed-weight workers, and a production length
sufficient for repeated window round trips.  Retain the direct athermal baseline
as an independent bulk comparison rather than replacing or deleting it.
