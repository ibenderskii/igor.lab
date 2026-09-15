#!/usr/bin/env python3
"""Read an umbrella baseline or pilot NPZ and say what to do next.

    python inspect_umbrella.py umbrella_runs/N44/pilot_N44.npz
"""
import sys
import numpy as np


def main(path: str) -> int:
    z = np.load(path, allow_pickle=True)
    g = lambda k, d=None: z[k] if k in z.files else d

    N = int(g("N", -1))
    centers = np.asarray(g("umbrella_window_centers", []))
    k = float(g("umbrella_k", float("nan")))
    assumed = float(g("umbrella_tail_slope", float("nan")))
    recovered = float(g("umbrella_recovered_tail_slope", float("nan")))
    headroom = float(g("umbrella_pinning_headroom", float("nan")))
    mode = int(g("umbrella_top_window_mode", -1))
    c_prob = np.asarray(g("c_prob", []))
    m_max = c_prob.size - 1

    print(f"N={N}  k={k:g}  windows={centers.size}  top centre={centers[-1] if centers.size else '?'}")
    print(f"contact range 0..{m_max}, levels with weight: "
          f"{int((c_prob > 0).sum())}/{c_prob.size}")

    print("\n--- tail slope (the one input nothing else validates) ---")
    print(f"  assumed  --tail_slope : {assumed:.3f}")
    print(f"  recovered |dlogP0/dm| : {recovered:.3f}")
    print(f"  pinning headroom k*(top_centre - m_max): {headroom:.3f}")
    if np.isfinite(recovered) and recovered > headroom + 0.5 * k:
        need = int(np.ceil(recovered / k))
        print(f"  -> UNDER-EXTENDED. Rerun with --tail_slope {recovered:.1f} or more "
              f"(extension {need}, top centre >= {m_max + need}).")
        print("     The ladder is fixed at run start, so the existing checkpoint "
              "cannot be extended; this needs a fresh run.")
    else:
        print(f"  -> ladder is adequate (top window mode sits at m={mode}).")
        slack = headroom - recovered
        if np.isfinite(slack) and slack > 3.0:
            lean = max(recovered, 0.0)
            print(f"     You have {slack:.1f} of spare headroom; --tail_slope "
                  f"{lean + 0.5:.1f} would still pin and cost fewer windows.")

    print("\n--- mixing ---")
    sw = np.asarray(g("umbrella_swap_acceptance", []))
    ov = np.asarray(g("umbrella_empirical_adjacent_overlap", []))
    rt = np.asarray(g("umbrella_walker_round_trips", []))
    if sw.size:
        print(f"  adjacent exchange acceptance: {np.nanmin(sw):.3f} .. {np.nanmax(sw):.3f}")
    if ov.size:
        print(f"  adjacent overlap            : {ov.min():.3f} .. {ov.max():.3f}")
    if rt.size:
        print(f"  round trips per walker      : min {int(rt.min())}, "
              f"median {int(np.median(rt))}, total {int(rt.sum())}")
        if rt.min() < 1:
            print("     -> some walkers never made a full traverse; run longer "
                  "or lower --exchange_every.")

    print("\n--- statistical resolution ---")
    rel = np.asarray(g("c_blocked_rel_stderr", []))
    if rel.size:
        with np.errstate(invalid="ignore"):
            worst = np.nanmax(rel)
            idx = int(np.nanargmax(rel))
        print(f"  worst relative blocked stderr: {worst:.3f} at m={idx}")
        bad = np.flatnonzero(np.nan_to_num(rel, nan=0.0) > 0.25)
        if bad.size:
            print(f"  levels above 0.25: {bad.tolist()}")
    ess = g("importance_effective_sample_size")
    ns = g("n_samples")
    if ess is not None and ns is not None:
        print(f"  importance ESS: {float(ess):,.0f} of {int(ns):,} "
              f"({100*float(ess)/int(ns):.1f}%)")

    print("\n--- throughput (use this to size the production job) ---")
    secs = float(g("total_wall_seconds", float("nan")))
    spw = int(g("steps_per_window", 0))
    if np.isfinite(secs) and secs > 0 and spw:
        print(f"  {spw:,} steps/window in {secs/3600:.2f} h "
              f"-> {spw/secs:,.0f} moves/s per walker")
        for target in (20_000_000, 60_000_000, 120_000_000):
            print(f"     {target//1_000_000:>4}M steps/window would take "
                  f"{target/(spw/secs)/3600:6.1f} h")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1]))
