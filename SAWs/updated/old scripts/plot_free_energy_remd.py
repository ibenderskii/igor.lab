#!/usr/bin/env python3
"""
Plot free energy profiles from the REMD distributions NPZ.

Expected NPZ keys:
  - temps:      (nT,)
  - ct_centers: (nC,)   contact bin centers (pdf grid)
  - ct_hists:   (nT,nC) contact pdf values (integral over centers ≈ 1)
  - rg_centers: (nR,)   Rg bin centers (pdf grid)
  - rg_hists:   (nT,nR) Rg pdf values (integral over centers ≈ 1)

We plot (optionally reduced) free energy:
  F(x;T) = -kB*T*ln P(x|T)   or   F/(kB T) = -ln P(x|T)

Because P can be very small, we clip with eps before log.
We also shift each curve so min(F)=0 by default (only differences matter).

Example:
  python plot_free_energy_remd.py --npz remd_distributions_30mer.npz --kind both --outdir fe_plots
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def pdf_integral(pdf: np.ndarray, centers: np.ndarray) -> float:
    d = float(np.mean(np.diff(centers)))
    return float(np.sum(pdf) * d)


def free_energy_from_pdf(pdf: np.ndarray, T: float, kB: float, eps: float, reduced: bool) -> np.ndarray:
    p = np.clip(np.asarray(pdf, dtype=float), eps, None)
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

    # map temperature -> color
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
        sm.set_array([])  # needed for colorbar
        cbar = fig.colorbar(sm, ax=ax, pad=0.02)
        cbar.set_label("T")

    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.show()
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--npz", type=str, default='remd_distributions_44mer.npz', help="REMD distributions .npz")
    ap.add_argument("--kind", choices=["contacts", "rg", "both"], default="both")
    ap.add_argument("--outdir", type=str, default="free_energy_plots")
    ap.add_argument("--n_curves", type=int, default=14, help="how many temperatures to overlay")
    ap.add_argument("--stride", type=int, default=0, help="if >0, plot every stride-th temperature instead of n_curves")
    ap.add_argument("--kB", type=float, default=1.0)
    ap.add_argument("--eps", type=float, default=1e-12, help="probability floor for log")
    ap.add_argument("--reduced", action="store_true", help="plot -ln P (dimensionless) instead of -kBT ln P")
    ap.add_argument("--no_shift", action="store_true", default='false', help="do not shift each curve to min=0")
    ap.add_argument("--contact_offset", type=float, default=43,
                    help="subtract this constant from ct_centers (for plotting only)")
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    temps = np.asarray(d["temps"], dtype=float)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    stride = args.stride if args.stride and args.stride > 0 else None
    idx = choose_indices(len(temps), args.n_curves, stride)

    Ts = temps[idx]

    shift_min = (not args.no_shift)

    # Contacts
    if args.kind in ("contacts", "both"):
        c_centers = np.asarray(d["ct_centers"], dtype=float) - float(args.contact_offset)
        ct_hists = np.asarray(d["ct_hists"], dtype=float)[idx, :]

        # sanity check normalization (pdf integral ~ 1)
        integ0 = pdf_integral(ct_hists[0], c_centers)
        if not (0.8 <= integ0 <= 1.2):
            print(f"Warning: contact pdf integral ~ {integ0:.3g} (expected ~1). Proceeding anyway.")

        Fcs = np.stack([free_energy_from_pdf(ct_hists[i], Ts[i], args.kB, args.eps, args.reduced)
                        for i in range(len(idx))], axis=0)

        title = "REMD free energy vs contacts"
        if args.reduced:
            title += " (reduced: -ln P)"
        outpath = outdir / "F_contacts_overlay.png"
        plot_overlay(c_centers, Fcs, Ts, xlabel="contacts (m)", title=title, outpath=outpath,
                     shift_min=shift_min, add_colorbar=True)
        print(f"Saved {outpath}")

    # Rg
    if args.kind in ("rg", "both"):
        rg_centers = np.asarray(d["rg_centers"], dtype=float)
        rg_hists = np.asarray(d["rg_hists"], dtype=float)[idx, :]

        integ0 = pdf_integral(rg_hists[0], rg_centers)
        if not (0.8 <= integ0 <= 1.2):
            print(f"Warning: Rg pdf integral ~ {integ0:.3g} (expected ~1). Proceeding anyway.")

        Frg = np.stack([free_energy_from_pdf(rg_hists[i], Ts[i], args.kB, args.eps, args.reduced)
                        for i in range(len(idx))], axis=0)

        title = "REMD free energy vs $R_g$"
        if args.reduced:
            title += " (reduced: -ln P)"
        outpath = outdir / "F_Rg_overlay.png"
        plot_overlay(rg_centers, Frg, Ts, xlabel=r"$R_g$", title=title, outpath=outpath,
                     shift_min=shift_min, add_colorbar=True)
        print(f"Saved {outpath}")


if __name__ == "__main__":
    main()
