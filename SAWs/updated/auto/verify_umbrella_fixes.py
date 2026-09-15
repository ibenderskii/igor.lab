"""Verify each applied fix and check for regressions, N=7 vs exact enumeration."""
import math, tempfile, copy
from pathlib import Path
import numpy as np

import single_chain_wang_landau as W
import single_chain_umbrella_sampling as U
import ORIG_single_chain_umbrella_sampling as ORIG

N = 7
values, probs, radii = W.enumerate_rooted_saws(N)
M_MAX = int(values.max())
exact = np.zeros(M_MAX + 1); exact[values] = probs
print(f"exact N={N}: m_max={M_MAX} P(m)={np.round(exact,6).tolist()}")

def mk(mod, steps, ckpt, resume=None, nproc=1, seed=4242, tail=2.0, pull=0.25,
       ck_secs=1e9, **kw):
    a = mod.parse_args([])
    a.N, a.m_min, a.m_max = N, 0, M_MAX
    a.umbrella_k, a.window_spacing, a.tail_slope = 0.30, 3, tail
    a.steps_per_window, a.burnin = steps, 0.0
    a.sample_every, a.exchange_every = 20, 200
    a.pull_move_weight, a.init, a.n_processes, a.base_seed = pull, "rod", nproc, seed
    a.rg_bins, a.no_joint, a.max_wall_seconds = 30, False, math.inf
    a.n_blocks = 8
    a.min_samples_per_level = a.min_round_trips = 0
    a.min_adjacent_overlap = a.min_swap_acceptance = 0.0
    if hasattr(a, "max_level_rel_stderr"):
        a.max_level_rel_stderr = 0.0
    a.checkpoint, a.resume_checkpoint = ckpt, resume
    a.checkpoint_every_seconds = ck_secs
    for k, v in kw.items():
        setattr(a, k, v)
    mod.validate_args(a)
    return a

EXT = U.center_extension(2.0, 0.30)
CEN = U.make_window_centers(0, M_MAX, 3, EXT)
print(f"ladder ext={EXT} centers={CEN.tolist()}")
tmp = tempfile.TemporaryDirectory(); root = Path(tmp.name)
results = {}
TOTAL, SPLIT = 160_000, 60_000

# --- regression: exactness, resume determinism, compaction-invariance --------
a1 = mk(U, TOTAL, str(root/"a.npz"))
s1 = U.run_replica_exchange(a1, CEN, progress=False)

a2 = mk(U, SPLIT, str(root/"b.npz"));   U.run_replica_exchange(a2, CEN, progress=False)
a3 = mk(U, TOTAL, str(root/"b.npz"), resume=str(root/"b.npz"))
s2 = U.run_replica_exchange(a3, CEN, progress=False)

# many compaction events (checkpoint every 0.01 s) must not perturb anything
a4 = mk(U, TOTAL, str(root/"c.npz"), ck_secs=0.01)
s3 = U.run_replica_exchange(a4, CEN, progress=False)

def same(x, y):
    return all(np.array_equal(x["samples"][k], y["samples"][k]) for k in U.SAMPLE_KEYS) \
       and all(np.array_equal(np.asarray(x[k]), np.asarray(y[k]))
               for k in ("swap_attempts","swap_accepts","round_trips","window_by_walker"))

results["resume is bitwise identical to an uninterrupted run"] = same(s1, s2)
results["frequent compaction leaves every sample array identical"] = same(s1, s3)

an1 = U.analyse_samples(a1, CEN, s1, enforce_checks=False)
tvd = 0.5*float(np.abs(an1["built"]["c_prob"] - exact).sum())
rg = float(np.dot(an1["weights"], s1["samples"]["rg_samples"]))
rgerr = abs(rg - radii.mean())/radii.mean()
print(f"\nwith pull moves: TVD={tvd:.5f}  mean Rg rel err={rgerr:.3e}  "
      f"wham_hist_err={an1['wham_histogram_error']:.2e}")
results["N=7 with pull moves reproduces exact P(m)"] = tvd < 0.01
results["N=7 with pull moves reproduces exact mean Rg"] = rgerr < 0.01

# --- fix 8: labelling + dtype ----------------------------------------------
samp = s1["samples"]
lens = {k: samp[k].shape[0] for k in U.SAMPLE_KEYS}
results["all seven sample arrays stay the same length"] = len(set(lens.values())) == 1
# every sample's window label must be one a walker actually held, and each
# window must receive exactly the same number of samples
hist = U.build_window_histograms(samp["contact_samples"], samp["sample_windows"],
                                 CEN.size, 0, M_MAX)
results["every window still receives an equal sample count"] = \
    len(set(hist.sum(axis=1).tolist())) == 1
# a step must carry exactly one sample per walker
uniq, cnt = np.unique(samp["sample_steps"], return_counts=True)
results["each sampled step carries exactly one row per walker"] = bool(
    np.all(cnt == CEN.size))
# and each (step, walker) pair must map to a unique window
pairs = samp["sample_steps"].astype(np.int64) * 1000 + samp["sample_walkers"]
results["step/walker pairs are unique (no duplicated or dropped rows)"] = \
    np.unique(pairs).size == pairs.size
with np.load(root/"a.npz") as z:
    coord_dtype = z["coordination_histogram_samples"].dtype
    results["coordination samples stored as int16"] = coord_dtype == np.int16
    results["labels still stored as int64"] = all(
        z[k].dtype == np.int64 for k in ("sample_windows","sample_walkers","sample_steps"))
    print(f"coordination dtype in checkpoint: {coord_dtype}")
    # the histogram must still agree with the contact count it was recorded with
    ch = z["coordination_histogram_samples"].astype(np.int64)
    agree = np.all(ch @ np.arange(7, dtype=np.int64) == 2*z["contact_samples"])
    results["coordination histograms still match their own m after narrowing"] = bool(agree)
    results["coordination bins never exceed N"] = bool(ch.max() <= N)

# --- fix 8/backward compat: an OLD (int64) checkpoint must still load --------
ao = mk(ORIG, SPLIT, str(root/"old.npz"))
ORIG.run_replica_exchange(ao, CEN, progress=False)
with np.load(root/"old.npz") as z:
    print(f"old checkpoint coordination dtype: {z['coordination_histogram_samples'].dtype}")
an_old = mk(U, SPLIT, None, resume=str(root/"old.npz"))
loaded = U.load_checkpoint(Path(root/"old.npz"), an_old, CEN)
results["a checkpoint written by the old code still loads"] = bool(loaded["states"])

# --- fix 1: an exception mid-run must still leave a checkpoint ---------------
ckpt = root/"boom.npz"
ab = mk(U, 120_000, str(ckpt), ck_secs=1800.0)   # the shipped default
orig_adv, calls = U.advance_walker, {"n": 0}
def boom(*a, **k):
    calls["n"] += 1
    if calls["n"] > 400: raise RuntimeError("simulated worker consistency failure")
    return orig_adv(*a, **k)
U.advance_walker = boom
try:
    U.run_replica_exchange(ab, CEN, progress=False)
except RuntimeError as e:
    print(f"\nraised as expected: {e}")
finally:
    U.advance_walker = orig_adv
results["a mid-run exception still writes the checkpoint"] = ckpt.exists()
if ckpt.exists():
    with np.load(ckpt) as z:
        print(f"  preserved {z['contact_samples'].size} samples, "
              f"steps_done={z['steps_done'][0]}")
        results["the salvaged checkpoint is synchronised at a block boundary"] = \
            len(set(z["steps_done"].tolist())) == 1
    ar = mk(U, 120_000, None, resume=str(ckpt))
    results["the salvaged checkpoint reloads and can be resumed"] = bool(
        U.load_checkpoint(ckpt, ar, CEN)["states"])

# --- fix 3: the pinning gate, both directions -------------------------------
cen0 = U.make_window_centers(0, M_MAX, 3, 0)
a0 = mk(U, SPLIT, str(root/"d.npz"), tail=0.0)
s0 = U.run_replica_exchange(a0, cen0, progress=False)
an0 = U.analyse_samples(a0, cen0, s0, enforce_checks=False)
fired = any("not pinned at m_max" in f for f in an0["diagnostic_failures"])
results["pinning gate fires on the unextended ladder"] = fired
results["pinning gate stays quiet on the extended ladder"] = not any(
    "not pinned at m_max" in f for f in an1["diagnostic_failures"])
print(f"\nunextended: recovered |s|={an0['recovered_tail_slope']:.3f} vs headroom "
      f"{an0['pinning_headroom']:.3f}, top-window mode m={an0['top_window_mode']}")
print(f"extended  : recovered |s|={an1['recovered_tail_slope']:.3f} vs headroom "
      f"{an1['pinning_headroom']:.3f}, top-window mode m={an1['top_window_mode']}")

# --- fix 5: relative-stderr gate --------------------------------------------
strict = copy.copy(a1); strict.max_level_rel_stderr = 1e-6
f_strict = U.analyse_samples(strict, CEN, s1, enforce_checks=False)["diagnostic_failures"]
loose = copy.copy(a1); loose.max_level_rel_stderr = 10.0
f_loose = U.analyse_samples(loose, CEN, s1, enforce_checks=False)["diagnostic_failures"]
results["relative-stderr gate fires when strict"] = any("max_level_rel_stderr" in f for f in f_strict)
results["relative-stderr gate quiet when loose"] = not any("max_level_rel_stderr" in f for f in f_loose)
print(f"\nc_blocked_rel_stderr = {np.round(an1['c_blocked_rel_stderr'],5).tolist()}")

# --- fix 4: per-walker round-trip floor -------------------------------------
probe = copy.copy(a1); probe.min_round_trips = 1
sim_probe = dict(s1); sim_probe["round_trips"] = np.array([0]+[9999]*(CEN.size-1), dtype=np.int64)
f_rt = U.analyse_samples(probe, CEN, sim_probe, enforce_checks=False)["diagnostic_failures"]
results["per-walker floor rejects one walker doing all the traversing"] = any(
    "floor per walker" in f for f in f_rt)
sim_ok = dict(s1); sim_ok["round_trips"] = np.full(CEN.size, 3, dtype=np.int64)
results["per-walker floor accepts a uniformly mixing ladder"] = not any(
    "floor per walker" in f for f in
    U.analyse_samples(probe, CEN, sim_ok, enforce_checks=False)["diagnostic_failures"])

# --- fix 9 ------------------------------------------------------------------
try:
    U.empirical_adjacent_overlap(np.array([1.0, 2.0, 3.0]))
    results["1-D overlap input raises the intended ValueError"] = False
except ValueError as e:
    results["1-D overlap input raises the intended ValueError"] = "two-dimensional" in str(e)

print("\n" + "="*72)
bad = 0
for k, v in results.items():
    print(f"  {'PASS' if v else 'FAIL'}: {k}")
    bad += not v
print("="*72)
print("ALL VERIFICATIONS PASSED" if not bad else f"{bad} FAILED")
tmp.cleanup()
