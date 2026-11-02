#!/usr/bin/env python
"""Deterministic, code-independent inputs for the golden-output cases.

Writes (into ``--out``):

* ``saw_N{n}_baseline.npz`` -- the EXACT athermal ensemble of n-bead
  self-avoiding walks on the simple cubic lattice, by complete enumeration
  (first bond fixed along +x; the other five directions are equivalent, so
  every probability is unchanged).  It carries both the scalar joint baseline
  (c_vals, c_prob, c_edges, rg_edges, rg_prob, crg_prob, N/n_beads) and the
  local-coordination state table with its sparse state x Rg joint
  (local_coord_* keys, schema 1), so it serves every model family.
* ``saw_N{n}_target.npz`` -- a synthetic "molecular" target (temps, ct_centers,
  ct_hists, rg_centers, rg_hists) made by reweighting that ensemble with a
  local-coordination potential and adding fixed-seed multiplicative noise.
* ``MANIFEST.json`` -- sha256 of each file, the generating parameters, and the
  enumeration counts.

Nothing here imports the code under test, so baseline and candidate runs see
byte-identical inputs.  Python >= 3.8, numpy only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

NN = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))


def enumerate_saws(n_beads: int):
    """Return (m, hist7, rg2_numerator) arrays over all walks with bond 0 = +x.

    rg2 = rg2_numerator / n_beads**2 exactly (integer numerator)."""
    if n_beads < 3:
        raise ValueError("n_beads must be >= 3")
    pos = [(0, 0, 0), (1, 0, 0)]
    occ = {(0, 0, 0): 0, (1, 0, 0): 1}
    deg = [0] * n_beads
    out_m, out_h, out_r = [], [], []
    state = {"m": 0}

    def leaf():
        h = [0] * 7
        for k in deg:
            h[k] += 1
        sx = sum(p[0] for p in pos); sy = sum(p[1] for p in pos); sz = sum(p[2] for p in pos)
        s2 = sum(p[0] * p[0] + p[1] * p[1] + p[2] * p[2] for p in pos)
        out_m.append(state["m"])
        out_h.append(h)
        out_r.append(n_beads * s2 - (sx * sx + sy * sy + sz * sz))

    def grow(j):
        if j == n_beads:
            leaf()
            return
        x, y, z = pos[-1]
        for dx, dy, dz in NN:
            p = (x + dx, y + dy, z + dz)
            if p in occ:
                continue
            # nonbonded neighbours of the new bead j (excluding its predecessor j-1)
            partners = []
            for ex, ey, ez in NN:
                q = (p[0] + ex, p[1] + ey, p[2] + ez)
                i = occ.get(q)
                if i is not None and i != j - 1:
                    partners.append(i)
            for i in partners:
                deg[i] += 1
            deg[j] = len(partners)
            state["m"] += len(partners)
            occ[p] = j
            pos.append(p)
            grow(j + 1)
            pos.pop()
            del occ[p]
            state["m"] -= len(partners)
            deg[j] = 0
            for i in partners:
                deg[i] -= 1

    grow(2)
    return (np.asarray(out_m, dtype=np.int64), np.asarray(out_h, dtype=np.int64),
            np.asarray(out_r, dtype=np.int64))


def build_baseline(n_beads: int, n_rg_bins: int):
    m, hist, rnum = enumerate_saws(n_beads)
    n_walks = int(m.size)
    rg = np.sqrt(rnum.astype(float) / float(n_beads * n_beads))
    lo, hi = float(rg.min()), float(rg.max())
    pad = 0.5 * (hi - lo) / n_rg_bins
    rg_edges = np.linspace(lo - pad, hi + pad, n_rg_bins + 1)
    rg_bin = np.clip(np.searchsorted(rg_edges, rg, side="right") - 1, 0, n_rg_bins - 1)

    m_max = int(m.max())
    c_vals = np.arange(m_max + 1, dtype=np.int64)
    c_counts = np.bincount(m, minlength=m_max + 1)
    c_prob = c_counts / float(n_walks)
    crg_counts = np.zeros((m_max + 1, n_rg_bins), dtype=np.int64)
    np.add.at(crg_counts, (m, rg_bin), 1)
    crg_prob = crg_counts / float(n_walks)
    c_edges = np.arange(-0.5, m_max + 1.0, 1.0)

    # local-coordination state table (unique histograms, lexicographic order)
    states, state_idx = np.unique(hist, axis=0, return_inverse=True)
    state_idx = np.asarray(state_idx).reshape(-1)
    S = states.shape[0]
    s_counts = np.bincount(state_idx, minlength=S)
    state_contacts = (states @ np.arange(7, dtype=np.int64)) // 2
    pair = state_idx * n_rg_bins + rg_bin
    upairs, pcounts = np.unique(pair, return_counts=True)
    arrays = dict(
        c_vals=c_vals, c_prob=c_prob, c_edges=c_edges, rg_edges=rg_edges,
        rg_prob=crg_prob.sum(axis=0), crg_prob=crg_prob,
        N=np.int64(n_beads), n_beads=np.int64(n_beads),
        local_coord_schema_version=np.int64(1),
        local_coord_degree_values=np.arange(7, dtype=np.int64),
        local_coord_histograms=states.astype(np.int64),
        local_coord_contact_counts=state_contacts.astype(np.int64),
        local_coord_state_mass=s_counts / float(n_walks),
        local_coord_rg_state_index=(upairs // n_rg_bins).astype(np.int64),
        local_coord_rg_bin_index=(upairs % n_rg_bins).astype(np.int64),
        local_coord_rg_joint_mass=pcounts / float(n_walks),
        enumeration_n_walks_first_bond_fixed=np.int64(n_walks),
    )
    return arrays, dict(n_walks=n_walks, n_states=int(S), m_max=m_max,
                        n_rg_records=int(upairs.size))


def build_target(base: dict, params, temps, noise: float, seed: int):
    """Reweight the exact ensemble with u = b(T) m - A0 sum_i g(k_i)."""
    h_b, s_b, A0, q_sat = params
    q = 0.5 * np.arange(7, dtype=float)
    g = q * q / (1.0 + (q / q_sat) ** 2)
    hist = base["local_coord_histograms"].astype(float)
    m = base["local_coord_contact_counts"].astype(float)
    p0 = base["local_coord_state_mass"]
    coop = hist @ g
    rng = np.random.default_rng(seed)
    n_m = base["c_vals"].size
    rg_edges = base["rg_edges"]
    n_rg = rg_edges.size - 1
    widths = np.diff(rg_edges)
    ct = np.zeros((temps.size, n_m))
    rgh = np.zeros((temps.size, n_rg))
    for i, T in enumerate(temps):
        u = (h_b / T - s_b) * m - A0 * coop
        w = p0 * np.exp(-(u - u.min()))
        w /= w.sum()
        pm = np.bincount(base["local_coord_contact_counts"], weights=w, minlength=n_m)
        sw = w / np.where(p0 > 0, p0, 1.0)
        pr = np.bincount(base["local_coord_rg_bin_index"],
                         weights=base["local_coord_rg_joint_mass"] * sw[base["local_coord_rg_state_index"]],
                         minlength=n_rg)
        pm = pm * (1.0 + noise * rng.standard_normal(n_m)).clip(0.05)
        pr = pr * (1.0 + noise * rng.standard_normal(n_rg)).clip(0.05)
        ct[i] = pm / pm.sum()                   # unit-width bins: density == mass
        rgh[i] = (pr / pr.sum()) / widths      # density on the Rg grid
    return dict(temps=temps, ct_centers=base["c_vals"].astype(float), ct_hists=ct,
                rg_centers=0.5 * (rg_edges[:-1] + rg_edges[1:]), rg_hists=rgh)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-beads", type=int, default=10)
    ap.add_argument("--n-rg-bins", type=int, default=24)
    ap.add_argument("--n-temps", type=int, default=16)
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    n = int(args.n_beads)
    base, info = build_baseline(n, int(args.n_rg_bins))
    temps = 280.0 * (360.0 / 280.0) ** (np.arange(args.n_temps) / (args.n_temps - 1))
    params = (300.0, 0.8, 0.4, 1.5)
    target = build_target(base, params, temps, noise=0.04, seed=20261102)
    pb = out / f"saw_N{n}_baseline.npz"
    pt = out / f"saw_N{n}_target.npz"
    np.savez(pb, **base)
    np.savez(pt, **target)
    manifest = {
        "generator": "validation/make_inputs.py",
        "n_beads": n, "n_rg_bins": int(args.n_rg_bins), "n_temps": int(args.n_temps),
        "target_model": "local_coordination_saturation", "target_params": list(params),
        "target_noise": 0.04, "target_seed": 20261102, **info,
        "files": {pb.name: sha256(pb), pt.name: sha256(pt)},
    }
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
