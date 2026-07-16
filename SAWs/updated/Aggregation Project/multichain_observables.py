#!/usr/bin/env python3
"""
Minimal multi-chain observables for the first aggregation pilot (Stage 2).

Only the observables required to test whether the single-chain contact free
energy predicts aggregation are implemented:

* per-chain radius of gyration (from UNWRAPPED coordinates);
* raw lattice contact counts (m_intra, m_inter, m_total) and f_inter;
* an explicit CHAIN-level cluster graph (nodes are chains; two chains are
  connected when they share at least ``cluster_contact_threshold`` interchain
  contacts) with its component count, largest cluster size, and largest cluster
  fraction f_max = s_max / M.

The single-chain bead-level contact-graph S_max is deliberately NOT reused as an
aggregation measure; the chain-level graph is built here from interchain contact
pairs.  Post-run statistics and distribution histograms operate on plain arrays
so this module stays free of any sampler/model state.
"""
from __future__ import annotations

import math
from typing import Dict, List, Sequence

import numpy as np

import remd_uniform_chain_2_new as remd  # integrated_autocorr_time (ESS for SEMs)
from multichain_state import MultiChainState, per_chain_rg2
from multichain_contacts import ContactCounts, interchain_pair_counts


# ---------------------------------------------------------------------------
# Per-cycle observable extraction
# ---------------------------------------------------------------------------

def f_inter(counts: ContactCounts) -> float:
    """f_inter = m_inter / m_total, with f_inter = 0 when m_total == 0."""
    total = int(counts.total)
    if total == 0:
        return 0.0
    return float(counts.inter) / float(total)


def chain_geometry_summary(state: MultiChainState) -> dict:
    """Per-chain Rg summary in LATTICE units (rg_scale applied only downstream).

    Returns mean per-chain Rg, mean per-chain Rg^2, the std of per-chain Rg
    across chains, and the full per-chain Rg vector (for diagnostic output).
    """
    rg2 = per_chain_rg2(state)
    rg = np.sqrt(rg2)
    return {
        "mean_chain_rg2_lattice": float(rg2.mean()),
        "mean_chain_rg_lattice": float(rg.mean()),
        "std_chain_rg_lattice": float(rg.std(ddof=0)),
        "per_chain_rg_lattice": rg,
    }


def mean_pair_com_distance_over_L(state: MultiChainState) -> float:
    """Dimensionless mean interchain center-of-mass distance, Dc/L (paper eq. 3).

    Dc = < |r_c^alpha - r_c^beta| > over all unordered chain pairs (alpha < beta).
    Chain COMs come from the UNWRAPPED coordinates (each chain is internally
    contiguous there and is never wrapped for geometry, exactly as in
    :func:`multichain_state.per_chain_rg2`), so a COM may legitimately lie
    outside ``[0, L)^3``.  Pairwise separations then use the minimum-image
    convention with L = ``state.box_size``.  Dc/L is the dimensionless form
    reported by the paper (Dc for an ideal-gas arrangement scales with L).
    With M < 2 there are no interchain pairs, so the result is NaN.

    Sanity bounds: the value lies in [0, sqrt(3)/2 = 0.866].  Uncorrelated
    uniform COMs give 0.4803 (= E|u| for u uniform on [-L/2, L/2)^3);
    aggregation drives Dc/L well below 0.48.
    """
    M = state.n_chains
    if M < 2:
        return float("nan")
    L = float(state.box_size)
    coms = state.coords_unwrapped.astype(np.float64).mean(axis=1)  # (M, 3)
    d = coms[:, None, :] - coms[None, :, :]                        # (M, M, 3)
    # Min-imaging the SEPARATION (never the COMs individually) is exact: each
    # chain COM is a single well-defined point derived from contiguous unwrapped
    # coordinates, so the nearest periodic image of the difference is the
    # physical distance even when a COM sits outside the primary box (L >=
    # MIN_BOX_SIZE keeps the nearest image unambiguous).
    d -= L * np.round(d / L)
    dist = np.sqrt((d * d).sum(axis=-1))
    Dc = float(dist[np.triu_indices(M, k=1)].mean())
    return Dc / L


def _cluster_components(n_chains: int, edges: Sequence[tuple]) -> List[int]:
    """Union-find component sizes over ``n_chains`` nodes given chain-pair edges."""
    parent = list(range(n_chains))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[ra] = rb

    sizes: Dict[int, int] = {}
    for i in range(n_chains):
        r = find(i)
        sizes[r] = sizes.get(r, 0) + 1
    return list(sizes.values())


def cluster_summary(state: MultiChainState, cluster_contact_threshold: int = 1
                    ) -> dict:
    """Chain-level aggregation cluster statistics.

    Two chains are connected when they share at least
    ``cluster_contact_threshold`` interchain contacts.  Isolated chains are
    singleton components.  ``largest_cluster_fraction`` = s_max / M.
    """
    M = state.n_chains
    thr = int(cluster_contact_threshold)
    if thr < 1:
        raise ValueError("cluster_contact_threshold must be >= 1")
    pair_counts = interchain_pair_counts(state)
    edges = [pair for pair, cnt in pair_counts.items() if cnt >= thr]
    sizes = _cluster_components(M, edges)
    largest = max(sizes) if sizes else 0
    return {
        "n_clusters": int(len(sizes)),
        "largest_cluster_size": int(largest),
        "largest_cluster_fraction": float(largest) / float(M) if M else 0.0,
        "n_cluster_edges": int(len(edges)),
    }


def per_chain_contacts(m_intra, m_inter, n_chains: int) -> dict:
    """Intensive (per-chain) contact normalizations for cross-M comparison.

    Raw ``m_intra`` and ``m_inter`` are extensive, globally-unique contact-PAIR
    counts (each pair counted once).  Dividing by the number of chains M removes
    the leading extensive dependence on chain number so runs at different M are
    comparable (it does NOT remove all finite-size or concentration effects, and
    it never touches the Hamiltonian).  Two distinct interchain normalizations
    are provided and must not be conflated:

    * ``m_inter_pairs_per_chain      = m_inter / M``   (globally-unique pairs / M)
    * ``m_inter_incidences_per_chain = 2 * m_inter / M`` (contact endpoints / M;
      each interchain pair contributes one incidence to each of its two chains)
    """
    M = int(n_chains)
    if M <= 0:
        nan = float("nan")
        return {
            "m_intra_per_chain": nan, "m_inter_pairs_per_chain": nan,
            "m_inter_incidences_per_chain": nan, "m_total_pairs_per_chain": nan,
        }
    mi = float(m_intra)
    me = float(m_inter)
    return {
        "m_intra_per_chain": mi / M,
        "m_inter_pairs_per_chain": me / M,
        "m_inter_incidences_per_chain": 2.0 * me / M,
        "m_total_pairs_per_chain": (mi + me) / M,
    }


def cycle_observables(state: MultiChainState, cluster_contact_threshold: int = 1
                      ) -> dict:
    """All per-cycle structural observables for one lane (lattice units).

    Raw contact counts (``m_intra, m_inter, m_total``) are the extensive
    Hamiltonian variables and are returned unchanged.  Intensive per-chain
    normalizations (dividing by the number of chains M) are added for
    ANALYSIS ONLY so runs with different M can be compared; they never enter the
    reduced potential, sampling, or acceptance.  See :func:`per_chain_contacts`
    for the exact definitions (note the distinct pairs vs incidences meanings of
    the interchain normalization).
    """
    geo = chain_geometry_summary(state)
    clu = cluster_summary(state, cluster_contact_threshold)
    dc_over_L = mean_pair_com_distance_over_L(state)
    counts = state.counts
    out = {
        "m_intra": int(counts.intra),
        "m_inter": int(counts.inter),
        "m_total": int(counts.total),
        "f_inter": f_inter(counts),
        "mean_chain_rg_lattice": geo["mean_chain_rg_lattice"],
        "mean_chain_rg2_lattice": geo["mean_chain_rg2_lattice"],
        "std_chain_rg_lattice": geo["std_chain_rg_lattice"],
        "per_chain_rg_lattice": geo["per_chain_rg_lattice"],
        "largest_cluster_size": int(clu["largest_cluster_size"]),
        "largest_cluster_fraction": float(clu["largest_cluster_fraction"]),
        "n_clusters": int(clu["n_clusters"]),
        # Dc/L is the primary (dimensionless) form; the lattice-unit Dc is
        # recovered exactly by multiplying back, since L is a deterministic
        # factor and NaN * L stays NaN for the M < 2 case.
        "dc_over_L": dc_over_L,
        "dc_lattice": dc_over_L * float(state.box_size),
    }
    out.update(per_chain_contacts(int(counts.intra), int(counts.inter),
                                  state.n_chains))
    return out


# ---------------------------------------------------------------------------
# Post-run statistics (per lane) and distributions
# ---------------------------------------------------------------------------

def _mean(arr) -> float:
    a = np.asarray(arr, dtype=float)
    return float(np.nanmean(a)) if a.size else float("nan")


def _std(arr) -> float:
    a = np.asarray(arr, dtype=float)
    return float(np.nanstd(a, ddof=0)) if a.size else float("nan")


def _sem(arr) -> float:
    """Autocorrelation-corrected standard error of the mean: std / sqrt(ESS).

    MC cycles are correlated, so the naive std/sqrt(n) understates the error on a
    mean.  ESS comes from :func:`remd_uniform_chain_2_new.integrated_autocorr_time`
    (Geyer initial-positive-sequence, ess = n / tau_int), which IS the effective
    independent-sample count: Geyer's tau = 2*sum(Gamma_m) - 1 expands to
    1 + 2*sum_{k>=1} rho_k, the usual definition, with a robust truncation.  So the
    std/sqrt(ESS) formula is used as-is with no rescaling.  Because tau is clamped
    to >= 1, this is always >= std/sqrt(n).

    Non-finite entries are dropped first so std and ESS are computed over the same
    samples (integrated_autocorr_time filters internally).  Returns NaN for an
    empty series or when ESS is not >= 1 -- e.g. an all-NaN Dc lane at M < 2, which
    yields method="insufficient_samples" rather than raising.
    """
    a = np.asarray(arr, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float("nan")
    ess = remd.integrated_autocorr_time(a)["ess"]
    if not np.isfinite(ess) or ess < 1.0:
        return float("nan")
    return float(np.std(a, ddof=0) / math.sqrt(ess))


def _post_burnin(n: int, burnin_frac: float) -> int:
    return int(math.floor(int(n) * float(burnin_frac)))


def compute_statistics(
    traj: List[dict], temps: Sequence[float], burnin_frac: float = 0.5,
    rg_scale: float = 1.0, n_chains: int | None = None,
) -> List[dict]:
    """Per-lane post-burn-in means for the results CSV.

    ``traj`` is one dict per lane whose values are per-cycle sequences with keys
    ``u, effective_energy, m_intra, m_inter, mean_chain_rg_lattice,
    mean_chain_rg2_lattice, f_inter, largest_cluster_size,
    largest_cluster_fraction``.  Acceptance rates are added by the sampler.

    The number of chains M is taken from each lane's ``n_chains`` entry when
    present, else from the ``n_chains`` argument (M is never inferred from the
    cluster-size series).  When M is known, intensive per-chain contact
    normalizations are added; a per-chain mean/std is exactly the raw mean/std
    divided by M (2/M for the incidence form) because normalization is linear at
    fixed M.
    """
    results = []
    for lane, T in zip(traj, temps):
        n = len(lane["u"])
        s = _post_burnin(n, burnin_frac)
        sl = slice(s, None)
        rg_lat = np.asarray(lane["mean_chain_rg_lattice"], dtype=float)[sl]
        rg2_lat = np.asarray(lane["mean_chain_rg2_lattice"], dtype=float)[sl]
        m_intra = np.asarray(lane["m_intra"], dtype=float)[sl]
        m_inter = np.asarray(lane["m_inter"], dtype=float)[sl]
        # Per-chain Rg heterogeneity and chain-cluster count (optional keys; a
        # bare fake trajectory may omit them -> reported as NaN).
        std_rg_lat = np.asarray(
            lane.get("std_chain_rg_lattice", []), dtype=float)[sl] \
            if lane.get("std_chain_rg_lattice") is not None else np.array([])
        n_clusters = np.asarray(
            lane.get("n_clusters", []), dtype=float)[sl] \
            if lane.get("n_clusters") is not None else np.array([])
        # Mean interchain COM distance (Dc, Dc/L).  NaN for M < 2 lanes; the
        # nan-aware helpers below propagate that as NaN rather than raising.
        dc_lat = np.asarray(
            lane.get("dc_lattice", []), dtype=float)[sl] \
            if lane.get("dc_lattice") is not None else np.array([])
        dc_over_L = np.asarray(
            lane.get("dc_over_L", []), dtype=float)[sl] \
            if lane.get("dc_over_L") is not None else np.array([])

        lcf = np.asarray(lane["largest_cluster_fraction"], dtype=float)[sl]

        mi_mean, mi_std = _mean(m_intra), _std(m_intra)
        me_mean, me_std = _mean(m_inter), _std(m_inter)
        mt_mean = _mean(m_intra + m_inter)
        # Error bars on the reported means.  These are standard errors
        # (std/sqrt(ESS)) and are NOT the *_std ensemble fluctuation widths above,
        # which stay untouched and must not be drawn as error bars.
        rg_sem_lat = _sem(rg_lat)
        rg_ac = remd.integrated_autocorr_time(rg_lat)

        r = {
            "T": float(T),
            "u_mean": _mean(np.asarray(lane["u"], dtype=float)[sl]),
            "u_std": _std(np.asarray(lane["u"], dtype=float)[sl]),
            "effective_energy_mean": _mean(
                np.asarray(lane["effective_energy"], dtype=float)[sl]),
            "m_intra_mean": mi_mean,
            "m_intra_std": mi_std,
            "m_intra_sem": _sem(m_intra),
            "m_inter_mean": me_mean,
            "m_inter_std": me_std,
            "m_inter_sem": _sem(m_inter),
            "m_total_mean": mt_mean,
            "Rg_mean_lattice": _mean(rg_lat),
            "Rg_std_lattice": _std(rg_lat),
            "Rg_sem_lattice": rg_sem_lat,
            "Rg_mean": rg_scale * _mean(rg_lat),
            "Rg_std": rg_scale * _std(rg_lat),
            # rg_scale is a deterministic linear rescale, so the SEM rescales
            # exactly with it, just as Rg_std does.
            "Rg_sem": rg_scale * rg_sem_lat,
            "Rg_tau_int": float(rg_ac["tau_int"]),
            "Rg_ess": float(rg_ac["ess"]),
            "Rg2_mean_lattice": _mean(rg2_lat),
            "Rg2_mean": (rg_scale ** 2) * _mean(rg2_lat),
            "f_inter_mean": _mean(np.asarray(lane["f_inter"], dtype=float)[sl]),
            "largest_cluster_size_mean": _mean(
                np.asarray(lane["largest_cluster_size"], dtype=float)[sl]),
            "largest_cluster_fraction_mean": _mean(lcf),
            "largest_cluster_fraction_sem": _sem(lcf),
            # Mean interchain COM distance: raw Dc in lattice units and the
            # dimensionless Dc/L reported by the paper.
            "Dc_lattice_mean": _mean(dc_lat),
            "Dc_lattice_std": _std(dc_lat),
            "Dc_over_L_mean": _mean(dc_over_L),
            "Dc_over_L_std": _std(dc_over_L),
            "Dc_over_L_sem": _sem(dc_over_L),
            "Dc_over_L_ess": float(
                remd.integrated_autocorr_time(dc_over_L)["ess"]),
            "std_chain_rg_mean_lattice": _mean(std_rg_lat),
            "std_chain_rg_mean": rg_scale * _mean(std_rg_lat),
            "n_clusters_mean": _mean(n_clusters),
        }
        # Intensive per-chain contact statistics (analysis only; M from lane).
        M = lane.get("n_chains", n_chains)
        if M is not None and int(M) > 0:
            Mf = float(int(M))
            r.update({
                "n_chains": int(M),
                "m_intra_per_chain_mean": mi_mean / Mf,
                "m_intra_per_chain_std": mi_std / Mf,
                "m_inter_pairs_per_chain_mean": me_mean / Mf,
                "m_inter_pairs_per_chain_std": me_std / Mf,
                "m_inter_incidences_per_chain_mean": 2.0 * me_mean / Mf,
                "m_inter_incidences_per_chain_std": 2.0 * me_std / Mf,
                "m_total_pairs_per_chain_mean": mt_mean / Mf,
            })
        else:
            for k in ("m_intra_per_chain_mean", "m_intra_per_chain_std",
                      "m_inter_pairs_per_chain_mean", "m_inter_pairs_per_chain_std",
                      "m_inter_incidences_per_chain_mean",
                      "m_inter_incidences_per_chain_std",
                      "m_total_pairs_per_chain_mean"):
                r[k] = float("nan")
        results.append(r)
    return results


def _int_hist_matrix(seqs: List[np.ndarray]):
    """Per-lane P(k) over integer values [0, kmax]; returns (vals, P (nT,kmax+1))."""
    kmax = max((int(np.nanmax(a)) for a in seqs if a.size > 0), default=0)
    vals = np.arange(kmax + 1, dtype=int)
    P = np.full((len(seqs), kmax + 1), np.nan, dtype=float)
    for i, a in enumerate(seqs):
        if a.size == 0:
            continue
        k = np.rint(a).astype(int)
        row = np.zeros(kmax + 1, dtype=float)
        valid = (k >= 0) & (k <= kmax)
        np.add.at(row, k[valid], 1)
        tot = row.sum()
        if tot > 0:
            P[i] = row / tot
    return vals, P


def _float_hist_matrix(seqs: List[np.ndarray], n_bins: int, lo=None, hi=None):
    """Per-lane P over a shared padded linspace grid; returns (edges, centers, P)."""
    allv = np.concatenate([a[np.isfinite(a)] for a in seqs if a.size > 0]) \
        if any(a.size for a in seqs) else np.array([])
    if allv.size > 0:
        vlo = float(allv.min()) if lo is None else float(lo)
        vhi = float(allv.max()) if hi is None else float(hi)
        pad = 0.02 * (vhi - vlo) if vhi > vlo else 1e-9
    else:
        vlo, vhi, pad = 0.0, 1.0, 0.0
    edges = np.linspace(vlo - pad, vhi + pad, int(n_bins) + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    P = np.full((len(seqs), int(n_bins)), np.nan, dtype=float)
    for i, a in enumerate(seqs):
        v = a[np.isfinite(a)]
        if v.size == 0:
            continue
        counts, _ = np.histogram(v, bins=edges)
        tot = counts.sum()
        if tot > 0:
            P[i] = counts.astype(float) / tot
    return edges, centers, P


def build_distributions(
    traj: List[dict], temps: Sequence[float], *, burnin_frac: float = 0.5,
    rg_bins: int = 40, fmax_bins: int = 40, rg_scale: float = 1.0,
) -> dict:
    """Build P(m_intra|T), P(m_inter|T), P(mean_chain_Rg|T), P(f_max|T).

    Rg histograms are built in lattice units and the saved axis is converted to
    output units by ``rg_scale`` (probability mass unchanged).
    """
    Ts = np.asarray(temps, dtype=float)
    mi_seqs, me_seqs, rg_seqs, fx_seqs, chain_rg_seqs = [], [], [], [], []
    for lane in traj:
        n = len(lane["u"])
        s = _post_burnin(n, burnin_frac)
        mi_seqs.append(np.asarray(lane["m_intra"], dtype=float)[s:])
        me_seqs.append(np.asarray(lane["m_inter"], dtype=float)[s:])
        rg_seqs.append(np.asarray(lane["mean_chain_rg_lattice"], dtype=float)[s:])
        fx_seqs.append(np.asarray(lane["largest_cluster_fraction"], dtype=float)[s:])
        # Pool ALL individual-chain post-burn-in Rg samples for this lane; this
        # is semantically distinct from the cycle-level MEAN-across-chains
        # distribution above and resolves expanded/collapsed chain mixtures.
        per_chain = lane.get("per_chain_rg_lattice")
        if per_chain is not None and len(per_chain) > s:
            pooled = np.concatenate(
                [np.asarray(v, dtype=float).ravel() for v in per_chain[s:]]) \
                if len(per_chain[s:]) else np.array([])
        else:
            pooled = np.array([])
        chain_rg_seqs.append(pooled)

    m_intra_vals, P_m_intra = _int_hist_matrix(mi_seqs)
    m_inter_vals, P_m_inter = _int_hist_matrix(me_seqs)
    rg_edges_lat, rg_centers_lat, P_rg = _float_hist_matrix(rg_seqs, rg_bins)
    chain_rg_edges_lat, chain_rg_centers_lat, P_chain_rg = _float_hist_matrix(
        chain_rg_seqs, rg_bins)
    fmax_edges, fmax_centers, P_fmax = _float_hist_matrix(
        fx_seqs, fmax_bins, lo=0.0, hi=1.0)

    out = {
        "Ts": Ts,
        "m_intra_vals": m_intra_vals,
        "P_m_intra": P_m_intra,
        "m_inter_vals": m_inter_vals,
        "P_m_inter": P_m_inter,
        "rg_edges": rg_scale * rg_edges_lat,
        "rg_centers": rg_scale * rg_centers_lat,
        "rg_edges_lattice": rg_edges_lat,
        "rg_centers_lattice": rg_centers_lat,
        "P_mean_chain_rg": P_rg,
        # Pooled individual-chain Rg distribution (distinct from the mean-across-
        # chains distribution) with its own length axes.
        "chain_rg_edges": rg_scale * chain_rg_edges_lat,
        "chain_rg_centers": rg_scale * chain_rg_centers_lat,
        "chain_rg_edges_lattice": chain_rg_edges_lat,
        "chain_rg_centers_lattice": chain_rg_centers_lat,
        "P_chain_rg": P_chain_rg,
        "fmax_edges": fmax_edges,
        "fmax_centers": fmax_centers,
        "P_fmax": P_fmax,
        "rg_scale": float(rg_scale),
    }
    # Intensive contact axes for cross-M comparison.  Normalization by M is a
    # deterministic rescaling of the value axis at fixed M, so the probability
    # matrices are REUSED (no histogram recompute).  Only added when M is known.
    M = next((lane["n_chains"] for lane in traj if lane.get("n_chains")), None)
    if M is not None and int(M) > 0:
        Mf = float(int(M))
        out.update({
            "n_chains": int(M),
            "m_intra_per_chain_vals": m_intra_vals / Mf,
            "P_m_intra_per_chain": P_m_intra,
            "m_inter_pairs_per_chain_vals": m_inter_vals / Mf,
            "P_m_inter_pairs_per_chain": P_m_inter,
            "m_inter_incidences_per_chain_vals": 2.0 * m_inter_vals / Mf,
            "P_m_inter_incidences_per_chain": P_m_inter,
        })
    return out
