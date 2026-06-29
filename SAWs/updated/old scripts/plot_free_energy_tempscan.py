#!/usr/bin/env python3
"""
Plot free energy profiles from the temp_scan aggregated distributions NPZ.

Expected NPZ keys (from temp_scan2_updated / overlay workflow):
  - Ts:        (nT,) temperatures
  - c_vals:    (nC,) integer contact values (support)
  - Pc:        (nT,nC) probability mass over c_vals (rows sum to 1)
  - rg_edges:  (nR+1,) bin edges for Rg
  - rg_centers:(nR,) bin centers
  - Prg:       (nT,nR) probability mass over rg bins (rows sum to 1)

We plot:
  F(x;T) = -kB*T*ln P(x|T)   or reduced F/(kBT) = -ln P(x|T)

Because Pc/Prg are probability *masses*, using mass vs pdf differs only by
an additive constant if bin widths are uniform (we shift min to 0 anyway).

Example:
  python plot_free_energy_tempscan.py --npz temp_scan_30mer_dists.npz --kind both --outdir fe_scan
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def mass_to_pdf(mass: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Convert probability mass per bin to pdf values per bin."""
    w = np.diff(edges).astype(float)
    w[w <= 0] = np.nan
    pdf = mass / w
    return pdf


def free_energy(P: np.ndarray, T: float, kB: float, eps: float, reduced: bool) -> np.ndarray:
    p = np.clip(np.asarray(P, dtype=float), eps, None)
    if reduced:
        return -np.log(p)
    return -kB * float(T) * np.log(p)


def choose_indices(n: int, n_curves: int, stride: int | None) -> np.ndarray:
    if stride is not None and stride > 0:
        return np.arange(0, n, stride, dtype=int)
    n_curves = max(1, min(int(n_curves), n))
    if n_curves == n:
        return np.arange(n, dtype=int)
    return np.linspace(0, n - 1, n_curves).astype(int)


def plot_overlay(x: np.ndarray,
                 Ys: np.ndarray,
                 Ts: np.ndarray,
                 xlabel: str,
                 title: str,
                 outpath: Path,
                 shift_min: bool = True,
                 add_colorbar: bool = True) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    cmap = plt.cm.coolwarm
    tmin, tmax = float(np.min(Ts)), float(np.max(Ts))
    norm = plt.Normalize(vmin=tmin, vmax=tmax)

    for y, T in zip(Ys, Ts):
        yy = np.asarray(y, dtype=float)
        if shift_min:
            yy = yy - np.nanmin(yy)
        ax.plot(x, yy, color=cmap(norm(float(T))), alpha=0.85, lw=1.25)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Free energy (shifted)")
    ax.set_title(title)
    ax.grid(True, alpha=0.2)

    if add_colorbar:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.02)
        cbar.set_label("T")

    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.show()
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--npz", type=str, default='temp_scan_44mer_dists.npz', help="temp_scan aggregated dists .npz")
    ap.add_argument("--kind", choices=["contacts", "rg", "both"], default="both")
    ap.add_argument("--outdir", type=str, default="free_energy_plots_scan")
    ap.add_argument("--n_curves", type=int, default=14)
    ap.add_argument("--stride", type=int, default=0)
    ap.add_argument("--kB", type=float, default=1.0)
    ap.add_argument("--eps", type=float, default=1e-12)
    ap.add_argument("--reduced", action="store_true", help="plot -ln P (dimensionless)")
    ap.add_argument("--no_shift", action="store_true", help="do not shift each curve to min=0")
    ap.add_argument("--rg_use_pdf", action="store_true",
                    help="convert Prg mass -> pdf using rg_edges before computing F (only changes by const if uniform bins)")
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    Ts_all = np.asarray(d["Ts"], dtype=float)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    stride = args.stride if args.stride and args.stride > 0 else None
    idx = choose_indices(len(Ts_all), args.n_curves, stride)
    Ts = Ts_all[idx]

    shift_min = (not args.no_shift)

    if args.kind in ("contacts", "both"):
        c_vals = np.asarray(d["c_vals"], dtype=float)
        Pc = np.asarray(d["Pc"], dtype=float)[idx, :]
        # Pc should already be mass and sum to 1 per T
        Fc = np.stack([free_energy(Pc[i], Ts[i], args.kB, args.eps, args.reduced)
                       for i in range(len(idx))], axis=0)

        title = "temp_scan free energy vs contacts"
        if args.reduced:
            title += " (reduced: -ln P)"
        outpath = outdir / "F_contacts_overlay.png"
        plot_overlay(c_vals, Fc, Ts, xlabel="contacts (m)", title=title, outpath=outpath,
                     shift_min=shift_min, add_colorbar=True)
        print(f"Saved {outpath}")

    if args.kind in ("rg", "both"):
        rg_centers = np.asarray(d["rg_centers"], dtype=float)
        Prg = np.asarray(d["Prg"], dtype=float)[idx, :]
        if args.rg_use_pdf:
            rg_edges = np.asarray(d["rg_edges"], dtype=float)
            Prg_use = np.stack([mass_to_pdf(Prg[i], rg_edges) for i in range(len(idx))], axis=0)
        else:
            Prg_use = Prg

        Frg = np.stack([free_energy(Prg_use[i], Ts[i], args.kB, args.eps, args.reduced)
                        for i in range(len(idx))], axis=0)

        title = "temp_scan free energy vs $R_g$"
        if args.reduced:
            title += " (reduced: -ln P)"
        outpath = outdir / "F_Rg_overlay.png"
        plot_overlay(rg_centers, Frg, Ts, xlabel=r"$R_g$", title=title, outpath=outpath,
                     shift_min=shift_min, add_colorbar=True)
        print(f"Saved {outpath}")


if __name__ == "__main__":
    main()
