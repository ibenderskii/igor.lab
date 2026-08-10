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

The script uses four move families.  The first three are shared with the direct
athermal sampler:

1. proper cubic pivot rotations of the tail;
2. local 90-degree corner flips (kink jumps, one bead across a corner);
3. symmetric end moves;
4. Lesh-Mitzenmacher-Whitesides pull moves, at probability
   `--pull_move_weight` (default 0.25).

Families are chosen with fixed, state-independent probabilities.  That
independence is required: a mixture weight that varied with the contact number
or the chain density would not cancel in the acceptance ratio and would leave an
uncorrected factor in `q`.  Self-intersecting proposals are rejected.

The first three families are symmetric — each valid proposal has a reverse
proposal of equal probability — but pull moves are not, so the kernel as a whole
is *not* symmetric and a Hastings term is mandatory.  Every move function
therefore returns its own `log q(X'\to X) - log q(X\to X')`, which is exactly
zero for families 1-3.

For a frozen estimate `log_g_hat(m)`, the multicanonical conformation weight is

\[
W(X)=W(m(X))=\exp[-\log \hat g(m(X))].
\]

The Metropolis-Hastings acceptance probability is consequently

\[
a(X\to X')=\min\left(1,
\frac{q(X'\to X)}{q(X\to X')}
\exp[\log\hat g(m(X))-\log\hat g(m(X'))]\right).
\]

Each family satisfies detailed balance with respect to the frozen-weight
production distribution, and a mixture of such kernels with state-independent
weights satisfies it too.  No accounting across families is needed: `q` is never
summed over the several families that could produce the same `X'`.

### Pull moves and reversibility

A pull move is anchored at one bead and propagates a vacancy along the backbone,
so it can relocate a bead whose own neighbours are all occupied.  Neither a
corner flip nor an end move can do that, and pivot acceptance falls to
approximately zero in dense conformations, so pull moves are what supplies
mobility in the compact region of the contact window.

Their proposal ratio is obtained by enumeration, never by a hand-derived count:
the deduplicated catalog is built at `X` and again at the proposed `X'`, giving
`q(X\to X')=1/n_f` and `q(X'\to X)=1/n_r`, hence a term `log(n_f) - log(n_r)`.

The move set is **not** closed under inversion.  Lesh, Mitzenmacher and
Whitesides claimed reversibility, but the proof is wrong: Györffy, Závodszky and
Szilágyi (arXiv:1210.0495, *J. Comput. Chem.* 2013) showed some pull moves have
no inverse pull move, which biases estimated parameters.  Measured in this
implementation, single-bead outcomes always invert while multi-bead outcomes do
so only about 60% of the time.

The sampler therefore rejects any pull proposal whose inverse is absent from the
reverse catalog.  Writing `R` for the set of ordered pairs that are mutually
reachable — symmetric by construction — this gives

\[
\pi(X)T(X\to X')=\mathbf{1}[(X,X')\in R]\,
\min\!\left(\frac{\pi(X)}{n_f},\frac{\pi(X')}{n_r}\right),
\]

which is symmetric under exchange of `X` and `X'`, so detailed balance holds
exactly.  Rejected proposals are ordinary self-loops and cost only efficiency;
35-55% of pull proposals are discarded this way.  Note that `n_f` and `n_r`
remain the *full* catalog sizes — the irreversible members stay proposable and
are merely never accepted, so removing them from the counts would destroy the
balance this restores.

Omitting the rejection is not a small effect.  On the exactly enumerable N=6
test it moves the sampled `P(m=0)` from 0.712 to 0.493 against an exact 0.712,
a total variation distance of 0.219 rather than 0.002.

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
time origin is never reset between stages.

That schedule can leave the halving phase by either of two triggers, and the
two are not interchangeable.

- **Belardinelli-Pereyra crossing**, `log(f) <= 1/t`.  The two quantities are
  equal at the crossing, so adopting `log(f)=1/t` there is rate-neutral.  This
  enters the asymptotic phase: `log(f)=1/t` is thereafter recomputed after every
  attempted move as a running minimum, without a histogram-flatness criterion.
  Coverage is then judged against a visit histogram reset at entry, so the bias
  that is ultimately frozen must have been exercised over the whole included
  window at its final resolution.
- **Stage stall**, `--wl_stage_stall_steps` attempted moves in one incomplete
  stage.  A stall gives no reason to believe `log(f)` and `1/t` are comparable;
  at a stall `1/t` is typically orders of magnitude smaller, so adopting it
  would collapse the modification factor and freeze a barely-learned density.
  The stall therefore leaves `log(f)` untouched and halving continues.  It
  relaxes only how a stage may advance: coverage is judged against a visit
  histogram reset at the stall rather than per stage, and the flatness ratio is
  taken over the tier-2 levels actually visited.  The per-level minimum visit
  counts are never relaxed, so a level the chain cannot reach still blocks every
  stage and still fails loudly with its per-level counts.  The
  Belardinelli-Pereyra trigger stays armed, so a stall-relaxed run can still
  cross into the asymptotic phase later.

The wall-clock cap `--wl_max_seconds` is cumulative across resumes by default;
`--wl_max_seconds_scope per_invocation` restores per-invocation accounting.

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

The optional arrays `c_samples_resampled`, `rg_samples_resampled`, and
`bend_samples_resampled` are systematic importance resamples and therefore
contain duplicates.  They must not be used for variance or error-bar
estimation.  The deprecated names `c_samples`, `rg_samples`, and `bend_samples`
are written only with `--legacy_sample_aliases`.  The script does not write a
`c_counts` field.

The weighted histograms are authoritative.  `production_c_counts` stores raw
fixed-weight visits, `c_naive_count_error` is retained only as a coverage
indicator, and `c_blocked_stderr` is the per-level batch-means standard error
that accounts for within-chain autocorrelation when blocks are sufficiently
long.  Raw-sample provenance and duplicate fraction are recorded explicitly.

For N=30, 44, and 60, `rg_edges` retains the historical grid verbatim and
extends it on the same spacing to cover the compact-cluster reference and exact
rod limit.  Contact edges always span the complete declared window, including
zero-mass levels.  Histogramming raises if any sample falls outside either grid;
out-of-range mass is never silently discarded and renormalized away.

## 7. Validation sequence

1. **Move invariants:** verify self-avoidance, unit bonds, 23 proper
   non-identity rotations, and exact incremental contact deltas against full
   recounts.  `tests/test_wl_moves.py` additionally checks that every accepted
   pull move has its inverse in the reverse catalog, that a pull-move chain
   reproduces the exact N=6 frozen-weight distribution, and that
   `--pull_move_weight 0` is bit-identical to the sampler that predates pull
   moves.
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

The script's built-in `--self-test` performs steps 1 through 3.
`run_wl_pilot.py` performs steps 4 and 5 in the mandatory order N=30, N=44,
then N=60, stopping after any failed gate.  Its `--dry-run` mode prints all
commands and the measured-throughput wall-time estimate without creating
outputs.  The production pilot needs the project data and should be treated as
the scientific acceptance test.

## 8. Recommended first production workflow

Pilot the 30-mer first as the end-to-end smoke test, then the 44-mer and 60-mer,
using the exact geometric `m_max` selected automatically for each chain length.
Run with checkpointing, multiple fixed-weight workers, and a production length
sufficient for repeated window round trips.  Retain the direct athermal baseline
as an independent bulk comparison rather than replacing or deleting it.

Preview the complete gated workflow with:

```bash
python run_wl_pilot.py --dry-run
```

Run it by removing `--dry-run`.  A failure to reach the declared upper contact
level is reported as evidence; the runner never lowers `m_cover` to manufacture
a passing result.

To measure whether pull moves restore re-reachability of the compact window for
a given chain length, use

```bash
python single_chain_wang_landau.py --pull_move_probe --N 44 \
    --wl_schedule halving --pull_move_probe_steps 8000000
```

The probe first drives a chain to a genuine stage-1 completion — the same
flatness and visit criterion the learner uses, not merely first coverage — then
replays that one warmed state with and without pull moves from an identical
random stream.  It reports the spread of the warmed `log_g` against the analytic
scale `N ln(mu)` and warns loudly when the spread is too small to represent the
state that actually starves, because a result measured from a mildly inflated
bias is not evidence either way.  It is far too slow for CI and is not run by
the test suite.

Both arms are judged on the full stage criterion — per-level minimum visits and
the tier-2 flatness ratio — not on first coverage.  Touching every level once is
a much weaker bar than a stage has to clear, and it is not the bar that fails: a
starving stage has already reached the top of the window and still cannot
accumulate the minimum counts there.

#### Measured status

The N=44 warm-up reproduces the reference stage-1 exit closely: 7.3M steps,
min/mean 0.870 against a reference 0.868, `min_tier2` 124543.  Its `log_g`
spread is 20395 against an analytic scale of 67.9, roughly 300 times, so by
bias amplitude alone it is the pathological regime.

Nevertheless **neither arm starves on that state**: both meet the stage
criterion at 500,000 steps, with minimum per-level visits of 8159 with pull
moves and 8876 without.  The stage-1 exit state is therefore *not* the state
that fails at stage 3, and bias amplitude alone does not reproduce the failure —
the stage-3 starvation must also depend on the halved `log_f` and on the shape
`log_g` takes after two refinement stages, not merely on how large it has grown.

The acceptance question in the pull-move brief — whether a repeat sweep reaches
`m=50` in fewer than the 7.5M steps stage 3 fails in — is therefore **not yet
demonstrated for N=44**.  Answering it needs a genuine stage-3 checkpoint fed to
`--pull_move_probe_log_g`; no such checkpoint exists in the repository yet.

At N=30 the same probe does discriminate, and shows pull moves working as
intended.  From a warmed state (900k steps, spread 1428 against a scale of
46.3), the stage criterion is met at 100,000 steps with pull moves against
500,000 without, a factor of 5.  The effect is concentrated exactly where the
argument for pull moves predicts: the first hit of `m=29` falls from step
401,013 to step 77, and of `m=28` from 53,485 to 170.

#### Throughput, and the honest trade

Cost at `--pull_move_weight 0.25`: **15x** slower per attempted move at N=30 and
**22x** at N=44.  This is inherent rather than an implementation defect — an
accepted pull move builds the full catalog twice, once forward and once to
obtain `n_r` — and it has not been traded away, because approximating either
count would break the proposal ratio.

The consequence must not be glossed over.  At N=30 pull moves need 5x fewer
steps but 15x more time per step, so reaching the same stage criterion took
20.9s with them against 6.8s without: **a 5x mixing gain and a 3x wall-clock
loss on the same state.**  Steps measure mixing; wall clock is what a run
actually spends.  Pull moves are therefore worth their cost only where the
pull-free chain cannot complete a stage at all — which is precisely the reported
N=44 stage-3 failure, and precisely the case not yet reproducible here.  The
default weight has not been tuned against probe outcomes.
