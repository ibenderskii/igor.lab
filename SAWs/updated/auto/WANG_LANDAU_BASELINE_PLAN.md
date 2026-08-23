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

#### Learning initializer

`--wl_init` selects the starting conformation.  The default `rod` is the
straight chain `[(i,0,0)]`, which is what every earlier run and every golden
test uses.

`--wl_init compact` instead seeds a boustrophedon snake through the verified
optimal bounding box, which starts the learner at the exact geometric `m_max`.
This exists because reaching the compact end of the window from the rod is a
*search* problem, not a sampling one, and at `N=60` the search does not
succeed.  The reported probe plateaus at `m=70` after 300,000 steps at
`log_f=1` and never reaches 71 through 74; an independent repeat here, same
budget and `--pull_move_weight 0`, got no further than `m=60`, first reached at
stage step 142,977.  Either way stage 1 never completes and the run hard-fails
before any refinement.  With the initializer in place the same configuration
reports `range=True` and `highest_m=74` at the very first check, first reached
at stage step 0.

The construction is exact, not heuristic.  `m = e(occupied site set) - (N-1)`,
so any Hamiltonian path on an optimal site set realises `m_max`; the snake is
such a path provided an odd extent sits in the middle axis position, which
`_boustrophedon` requires explicitly.  Boxes are encoded only for `N=30`, `44`,
and `60` (`2x3x5`, `3x3x5`, `3x4x5`); any other chain length is refused rather
than guessed, and the realised contact count is asserted against the encoded
geometric maximum on every call, never assumed.  Recommended for `N=44` and
`N=60`.

Two limits.  The seed sits at `m_max` by construction, so it is out of window
whenever the ceiling has been narrowed — a lowered `--m_max`, or a declared
truncation — and learning then refuses to start and names the initializer.  And
the resume path never re-seeds: `--wl_init` applies only to a fresh run.

Starting *at* the compact end is not the same as being able to *return* to it
once the bias has built up.  The initializer removes the search problem; pull
moves remain what makes the compact end re-reachable afterwards.

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

#### Production budget gate

Learning reports which of its two caps actually binds.  Production now does the
same before it commits anything.  Immediately before the workers are submitted,
a short throwaway burst (20,000 steps) runs on the frozen `log_g` from the
learned chain, and the measured rate, the projected per-worker wall clock, and
the projected core-hours are printed.

`--production_max_seconds` (default infinite, so the default never refuses)
turns that projection into a gate: when the projected per-worker wall clock
exceeds it, the run refuses to launch and reports the `steps_per_worker` that
would fit.  Neither value is altered and no result is silently shortened —
the operator chooses between a smaller budget and checkpointed resumption.

At the defaults `--n_workers 12 --steps_per_worker 400000000`, and with the
workers parallel so wall clock equals per-worker time:

| N  | steps/s (pull 0.25) | per-worker wall clock |
|----|---------------------|-----------------------|
| 30 | 2,841               | 39.1 h                |
| 44 | 1,673               | 66.4 h                |
| 60 | 951                 | 116.8 h (4.9 days)    |

Against a 36 h partition limit, `N=44` and `N=60` are killed mid-production
having already paid for a successful learning phase.

#### Production checkpoint and resume

`--production_checkpoint STEM` makes worker `i` write `{STEM}_prod_w{i}.npz`
every `--checkpoint_every_seconds`, and once more on completion.  Each file is
written to a temporary name and renamed into place, so a kill mid-write cannot
corrupt it.  `--resume_production_checkpoint STEM` restarts every worker from
those files.

Contents are plain numeric arrays, so production checkpoints load with
`allow_pickle=False` exactly as WL checkpoints do: the chain and its contact
count, steps completed, accepted and geometrically valid counts, the
accumulated contact, `Rg` and bend sample arrays, the round-trip counter state,
and the generator state.  `random.Random.getstate()` is stored as a `uint32`
array of 625 plus two scalars, with `NaN` standing for an absent cached normal
variate.

On resume, `N`, the contact window, `log_g` and the tier array are checked
against the current run and any mismatch is refused, mirroring the WL resume.
Burn-in is measured against the **total** step budget rather than
steps-this-invocation, so a resumed worker does not re-burn and discard samples
it has already earned; this does mean the resume must be given the same
`--steps_per_worker`, `--burnin` and `--sample_every` as the run it continues.

Unlike a resumed WL stage, a resumed production worker restores the full
generator state at a step boundary, so it *is* a bitwise continuation of the
interrupted chain.  `tests/test_wl_production_checkpoint.py` asserts this
directly: an interrupted-and-resumed worker reproduces the uninterrupted run's
samples, acceptance count and round trips exactly.

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

### Scope of the flat-tier tail

`--flat_tail_scope` selects the target against which `--flat_tail_threshold` is
measured when deriving `m_flat`.  The default `full` measures it against the
whole molecular target, **including** mass sitting above `m_max`.

That inclusion is a defect, not a conservatism.  Mass above the geometric
maximum is at contact numbers the lattice cannot realise at all — 0.123% at the
worst temperature for `N=60`, reaching shifted `m` near 82 against `m_max=74` —
so no choice of `m_flat` can ever satisfy a threshold that charges for it.  The
tail therefore never crosses, `m_flat` clamps to `m_max`, and every level
becomes tier 2, making the flatness requirement maximally strict exactly where
sampling is hardest.  `support_report` already reports that out-of-window mass
separately, so counting it in the tail counts it twice.

`--flat_tail_scope in_window` renormalises the target to `m <= m_max` before
computing the tail.  Measured:

| N  | `m_flat` under `full` | `m_flat` under `in_window` | coverage tier |
|----|-----------------------|----------------------------|----------------|
| 30 | 25                    | 25                         | unchanged      |
| 44 | 49                    | 45                         | 46..50         |
| 60 | clamped to 74         | 72                         | 73..74         |

This is a budget and correctness improvement, not the `N=60` unblock — that is
`--wl_init compact`.  It **changes `N=44`'s tier boundaries**, so it stays
opt-in behind the flag and `full` remains the default pending a deliberate
scientific sign-off.  `--cover_tail_threshold` is deliberately unaffected:
declared truncation must report its omitted mass against the full target.

The clamp warning is retained under both scopes and now prints the actual
worst-temperature tail at the clamp point next to the threshold, so the operator
can see how far off it is.  At `N=60` under `full` the tail is 0.0012288 against
a threshold of 0.001 — short by 0.00023.

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
   moves.  `tests/test_wl_compact_seed.py` checks that each encoded compact seed
   is a valid SAW of `N` distinct sites joined by unit steps, that it attains
   the verified geometric maximum, that it fits inside its declared box, and
   that an unencoded chain length or a narrowed ceiling is refused rather than
   guessed at.  `tests/test_wl_production_checkpoint.py` round-trips the
   generator state through an `allow_pickle=False` NPZ, checks that an
   interrupted worker resumes as an exact continuation, that burn-in is counted
   against the total budget, and that a mismatched `log_g` or contact window is
   refused.
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

Use `--wl_init compact` for `N=44` and `N=60`; `N=60` does not complete stage 1
from the rod at all.  Checkpoint both phases and set a production gate matched to
the partition limit, for example against a 36 h wall:

```bash
python single_chain_wang_landau.py --N 60 --wl_init compact \
    --checkpoint runs/n60.npz --production_checkpoint runs/n60 \
    --production_max_seconds 129600
```

If the gate refuses, it names the `steps_per_worker` that would fit.  Prefer
resuming the full budget over shortening it: rerun the same command with
`--resume_production_checkpoint runs/n60` and the same
`--steps_per_worker`, `--burnin` and `--sample_every`.

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

At N=60 the compact initializer and pull moves address two different halves of
the same problem, and both halves are needed.  Measured over 400,000 steps at
`log_f=1` from `--wl_init compact`:

| levels 70..74 after 400k steps | m=70 | m=71 | m=72 | m=73 | m=74 |
|--------------------------------|------|------|------|------|------|
| `--pull_move_weight 0.25`      | 4080 | 2545 | 0    | 2251 | 3    |

Against `--pull_move_weight 0`, where all of 70 through 73 stand at zero after
*3M* steps, this is the pull-move argument working exactly as stated: the
initializer puts the chain at `m=74` at step 0, and pull moves are what let it
come back after the bias has pushed it away.  `range=True` and `highest_m=74`
hold from the first check under both weights, so range coverage alone is not
the discriminating diagnostic here — the per-level counts are.

Stage 1 still does not complete in that budget: `m=72` has no visits and `m=74`
has three.  **`m=72` must not be read as a geometric gap on this evidence.**
Section 4 forbids exactly that inference, and nothing in this measurement
distinguishes an unreachable level from a level of very low density that
400,000 steps did not resolve; a level sandwiched between two neighbours with
thousands of visits is a reason to look harder, not a reason to exclude it.
Excluding it would require an independent geometric verification and
`--excluded_contact_levels`, never a finite run.

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
