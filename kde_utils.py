# kde_utils.py
import pickle
import numpy as np
from pathlib import Path

def load_kde_as_u0(pkl_path, T, r, R_domain=None, shift_min_to_zero=True):
    """
    Load KDE {rg_points, kde_values} at temperature T from a pickle and
    convert to solver IC u(r) = 4π r² p(r) on grid r.

    - shift_min_to_zero=True: subtract min(rg) so the smallest Rg becomes 0
    - R_domain: if given, rescale the (shifted) support so max maps to R_domain
    """
    with Path(pkl_path).open("rb") as f:
        raw = pickle.load(f)

    # normalize temperature keys to ints if possible
    kde = {}
    for k, v in raw.items():
        try: kde[int(k)] = v
        except Exception: kde[k] = v
    if T not in kde:
        raise KeyError(f"T={T} not in pickle. Available keys: {sorted(kde.keys())}")

    rg  = np.asarray(kde[T]["rg_points"], dtype=float)
    pdf = np.asarray(kde[T]["kde_values"], dtype=float)

    # clean + sort
    m = np.isfinite(rg) & np.isfinite(pdf)
    rg, pdf = rg[m], pdf[m]
    order = np.argsort(rg)
    rg, pdf = rg[order], pdf[order]

    # 1) shift so min -> 0 (align different files)
    if shift_min_to_zero:
        rg_min = rg.min()
        rg = rg - rg_min  # now starts at 0

    # 2) optional rescale so max -> R_domain (align to solver domain)
    if R_domain is not None:
        rg_max = rg.max()
        if rg_max <= 0:
            raise ValueError("rg_max <= 0 after shift; cannot rescale.")
        scale = R_domain / rg_max
        rg = rg * scale
        # We changed the abscissa; renormalize numerically in the new coordinate
        # (safer than analytic change-of-variables for arbitrary sampled KDE)
    
    # (Re)normalize the PDF on its (possibly shifted/rescaled) grid
    Z = np.trapz(pdf, rg)
    if not np.isfinite(Z) or Z <= 0:
        raise ValueError("KDE normalization failed (area ≤ 0).")
    pdf = pdf / Z

    # 3) interpolate p(r) onto solver grid; zero outside support
    p0 = np.interp(r, rg, pdf, left=0.0, right=0.0)

    # 4) convert to shell mass u(r) = 4π r² p(r), renormalize for safety
    u0 = 4.0 * np.pi * (r**2) * p0
    Zu = np.trapz(u0, r)
    if not np.isfinite(Zu) or Zu <= 0:
        raise ValueError("u0 normalization failed on r-grid.")
    u0 /= Zu
    return u0


# --- example usage ---
# r = np.linspace(0, R, Nr)  # your existing grid
# u = load_kde_as_u0(r"C:\Users\ibend\data\rg_kde_data_44mer.pkl", 350, r, R_domain=R)
# then set solver state:
# u_current = u.copy()