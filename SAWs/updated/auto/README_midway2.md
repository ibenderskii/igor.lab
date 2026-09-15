# Running the umbrella athermal baseline sampler on RCC Midway2

Files here: `sbatch_umbrella_44.sh`, `sbatch_umbrella_60.sh`, `inspect_umbrella.py`,
`verify_umbrella_fixes.py`.

Everything runs from `SAWs/updated/auto` — the sampler imports
`single_chain_wang_landau`, which imports `baseline_grids` and `target_support`.

---

## 0. One-time checks on a login node

```bash
cd ~/igor.lab/SAWs/updated/auto
module load python/anaconda-2021.05
python --version                      # expect 3.8.8
python single_chain_umbrella_sampling.py --self-test
```

The self-test takes a few minutes and must end `SELF-TEST PASSED` (23 checks).
It runs two short N=6 simulations against exhaustive enumeration, so if it
passes, the Markov kernel, the WHAM algebra and the pull-move Hastings
correction are all behaving on this machine's numpy.

**Python version.** `python/anaconda-2021.05` is Python 3.8.8. The sampler now
works on 3.8, but this loop is pure-Python and a newer interpreter is
substantially faster. Run `module avail python` and prefer a 3.10/3.11 module
if your partition has one — it costs nothing to try, and the self-test tells
you whether it works.

---

## 1. Pick the ladder before you burn a node

`--tail_slope` is the only input nothing else validates, and it is mandatory
(no default) because it is a property of the chain, not of the sampler. It is
`|d log P0/dm|` at the geometric maximum, and it sets how far past `m_max` the
harmonic centres run:

> a harmonic window's mode sits `|s|/k` **below** its centre, so a ladder that
> stops at `m_max` peaks several contacts short of it and starves the compact
> tail it exists to reach.

Under-extending loses the tail. Over-extending costs windows linearly and is
otherwise harmless — WHAM absorbs the surplus. **So err high.**

Your existing `RnBaseline*.npz` cannot tell you this number: they stop at m=33
of 50 (N=44) and m≈46 of 74 (N=60), and their top bins are the starved ones, so
extrapolating from them underestimates. Use the pilot in step 2 instead.

Ladder size is free to check:

```bash
python single_chain_umbrella_sampling.py --N 44 --tail_slope 6 --show_window_design
```

| | `--tail_slope` | extension | windows | note |
|---|---|---|---|---|
| N=44, m_max=50 | 4 | 14 | 23 | |
| | **6** | 20 | **25** | starting point |
| | 8 | 27 | 27 | |
| N=60, m_max=74 | 4 | 14 | 31 | |
| | **5** | 17 | **32** | exactly fills 32 cpus |
| | 6 | 20 | 33 | **33 > 32 → two waves → 2× wall time** |

Set `--cpus-per-task` equal to the window count. `--n_processes` is omitted from
the scripts on purpose: it defaults to `SLURM_CPUS_PER_TASK`, capped at the
window count. Going one window over your cpu count roughly doubles wall time,
so check this number before submitting.

---

## 2. Pilot run (~1 h), to measure the tail slope

Edit `MODE="pilot"` at the top of `sbatch_umbrella_44.sh` and submit:

```bash
sbatch sbatch_umbrella_44.sh
```

The pilot runs 4M steps/window with **all gates disabled**, so it always
completes and always writes its diagnostics. Then:

```bash
python inspect_umbrella.py umbrella_runs/N44/pilot_N44.npz
```

That prints the recovered `|dlogP0/dm|` at `m_max` against the pinning headroom
the ladder provided, and tells you either "ladder is adequate" or the exact
`--tail_slope` to rerun with. It also prints the measured moves/s, so you can
size the production job for this hardware rather than trusting my numbers.

If it says **UNDER-EXTENDED**, raise `--tail_slope` and rerun the pilot. The
ladder is fixed when a run starts, so this is the one problem a checkpoint
cannot be extended out of — which is exactly why the pilot is cheap.

---

## 3. Production run

Set `MODE="production"` and `TAIL_SLOPE` to what the pilot recommended, then
submit. Gates are on: pooled coverage, adjacent overlap, exchange acceptance,
per-walker round trips, and per-level relative error.

**Set `--steps_per_window` to the final target on the first submission.**
`burn_steps = round(burnin * steps_per_window)` is computed once and frozen in
the checkpoint. Raising the target later keeps the old burn boundary and only
prints a note. Splitting the work across jobs is the wall-time checkpoint's
job, not `--steps_per_window`'s.

The job stops `--shutdown_margin_seconds` inside the SLURM limit, writes a
checkpoint and exits. If it printed `Run segment checkpointed successfully` and
no `DIST_FILE =`, resubmit the same script — it detects the checkpoint and
resumes. Repeat until `DIST_FILE = ...` appears. Resume is bitwise identical to
an uninterrupted run, so chaining costs nothing statistically.

### Throughput, and the pull-move cost

Measured on one core here (Midway2's Broadwell nodes will be slower — use the
pilot's own number):

| `--pull_move_weight` | N=44 moves/s | N=60 moves/s | N=44 h for 20M | N=60 h for 20M |
|---|---|---|---|---|
| 0.00 | 62,700 | 56,700 | 0.1 | 0.1 |
| 0.05 | 12,100 | 8,900 | 0.5 | 0.6 |
| 0.10 | 6,700 | 4,800 | 0.8 | 1.1 |
| 0.25 | 2,900 | 2,000 | 1.9 | 2.7 |
| 0.30 | 2,400 | 1,700 | 2.3 | 3.2 |

One pull move costs ~1.35 ms at N=44 and ~1.9 ms at N=60, versus ~16 µs for a
local move — it builds the full forward and reverse catalogs to get its
Hastings ratio. So `--pull_move_weight` is the dominant wall-time knob: 0.25
costs **26–33×** the throughput of local moves alone.

Whether that is worth it is an empirical question the pilot can answer. The
figure of merit is round trips per wall-hour, not moves/s: run the pilot twice,
at `--pull_move_weight 0.25` and `0.10`, and compare
`umbrella_walker_round_trips` divided by `total_wall_seconds`. Unlike the
Wang-Landau case, the umbrella bias is already pushing the chain into the
compact tail, so pull moves may be doing less work here than they do there.

---

## 4. Reading the result

```bash
python inspect_umbrella.py umbrella_runs/N44/RnBaseline44_umbrella.npz
```

The NPZ carries `c_prob`, `rg_prob`, `crg_prob` with the umbrella bias removed
(WHAM-reweighted, not raw biased trajectories), plus `c_blocked_stderr` and
`c_blocked_rel_stderr` for per-level error bars. `c_samples`/`rg_samples` are
systematic importance **resamples with duplicates** — do not use them for
variance or error bars; use `c_blocked_stderr`.

### When a gate fails

The run raises and writes no baseline, on purpose — but **every sample is in
the checkpoint**, and the error names the checkpoint. Nothing is lost.

| failure | response |
|---|---|
| levels below `--min_samples_per_level` | run longer (resubmit) |
| overlap / exchange acceptance too low | lower `--umbrella_k` or `--window_spacing`; needs a fresh run |
| round trips below floor | run longer, or lower `--exchange_every` |
| relative stderr too high | run longer |
| **not pinned at m_max** | raise `--tail_slope`; **fresh run required** |

To re-analyse existing samples without more sampling, resubmit with
`--resume_checkpoint` and `--steps_per_window` equal to the steps already done.

---

## 5. Regression suite (optional)

`verify_umbrella_fixes.py` re-checks the seven fixes — resume determinism,
compaction invariance, label alignment, checkpoint-on-exception, and both
directions of each new gate. It needs `ORIG_single_chain_umbrella_sampling.py`
(a copy of the pre-fix module) beside it to test old-checkpoint compatibility;
drop that requirement by deleting the two blocks that reference `ORIG`.
