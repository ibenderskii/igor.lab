"""Re-analyse an umbrella checkpoint to size the ladder from data.

The pinning gate reports |d log P0/dm| only at m_max, from the last two WHAM
bins -- the noisiest number in the run.  This prints the whole tail slope
profile plus each window's empirical mode, so --tail_slope for the next run is
chosen from where the statistics actually are rather than extrapolated.

    python reanalyse_tail.py pilot_44.npz
"""
import os
import sys

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import single_chain_umbrella_sampling as U


def main(path):
    with np.load(path, allow_pickle=False) as z:
        m_min = int(z["m_min"])
        m_max = int(z["m_max"])
        k = float(z["umbrella_k"])
        spacing = int(z["window_spacing"])
        centers = np.asarray(z["window_centers"], dtype=np.int64)
        contacts = np.asarray(z["contact_samples"], dtype=np.int64)
        windows = np.asarray(z["sample_windows"], dtype=np.int64)
        N = int(z["N"])
        assumed = float(z["tail_slope"]) if "tail_slope" in z.files else float("nan")
        steps_done = np.asarray(z["steps_done"]).ravel()[0] if "steps_done" in z.files else -1

    print(f"checkpoint            : {path}")
    print(f"N                     : {N}")
    print(f"m range               : {m_min}..{m_max}")
    print(f"k / spacing           : {k} / {spacing}")
    print(f"centers ({centers.size:2d})          : {centers.tolist()}")
    print(f"assumed --tail_slope  : {assumed:g}")
    print(f"steps done per window : {steps_done:,}")
    print(f"total samples         : {contacts.size:,}")

    contact_values = np.arange(m_min, m_max + 1, dtype=np.int64)
    bias = U.harmonic_bias_matrix(centers, contact_values, k)
    hist = U.build_window_histograms(contacts, windows, centers.size, m_min, m_max)
    wham = U.solve_wham(hist, bias, tolerance=1e-11, max_iterations=400_000)
    logp = np.asarray(wham["log_probability"], dtype=np.float64)
    pooled = hist.sum(axis=0)

    print("\n=== per-window empirical mode -> implied local slope ===")
    print("   center   mode   samples   implied |s| = k*(center-mode)")
    for j, c in enumerate(centers):
        n = hist[j].sum()
        if n == 0:
            print(f"   {c:6d}      -  {int(n):9,}   (empty)")
            continue
        mode = int(np.argmax(hist[j])) + m_min
        tag = "  <- PINNED at m_max" if mode == m_max else ""
        print(f"   {c:6d} {mode:6d}  {int(n):9,}   {k*(c-mode):8.3f}{tag}")

    print("\n=== recovered tail: slope profile ===")
    print("      m    pooled counts        log P0     |s(m)| = logP0(m-1)-logP0(m)")
    lo = max(m_min + 1, m_max - 20)
    for m in range(lo, m_max + 1):
        i = m - m_min
        s = logp[i - 1] - logp[i]
        c = int(pooled[i])
        flag = ""
        if c == 0:
            flag = "   <- NO SAMPLES"
        elif c < 50:
            flag = "   <- thin"
        lp = f"{logp[i]:13.4f}" if np.isfinite(logp[i]) else "         -inf"
        ss = f"{s:9.3f}" if np.isfinite(s) else "     n/a"
        print(f"   {m:4d}   {c:14,}  {lp}   {ss}{flag}")

    headroom = k * (float(centers[-1]) - m_max)
    recovered = logp[-2] - logp[-1]
    print(f"\ngate arithmetic for THIS run:")
    print(f"   recovered |s(m_max)|              = {recovered:.3f}")
    print(f"   headroom k*(top_center - m_max)   = {headroom:.3f}")
    print(f"   gate threshold headroom + k/2     = {headroom + 0.5*k:.3f}")
    print(f"   -> {'PASS' if recovered <= headroom + 0.5*k else 'FAIL'}")

    print("\n=== ladder cost vs assumed --tail_slope for the NEXT run ===")
    print("  tail_slope   ext   top center   n_windows   covers recovered |s| up to")
    seen = {}
    for ts in [round(0.1 * i, 1) for i in range(5, 46)]:
        ext = U.center_extension(ts, k)
        cen = U.make_window_centers(m_min, m_max, spacing, ext)
        key = int(cen[-1])
        if key in seen:
            continue
        seen[key] = ts
        cover = k * (key - m_max) + 0.5 * k
        print(f"    {ts:6.1f}    {ext:3d}   {key:10d}   {cen.size:9d}   {cover:24.2f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "pilot_44.npz")
