#!/usr/bin/env python3
"""pytest suite for the kappa_bend metadata path in fit_lattice_contact_model_2.

The bending penalty is baked into the BASELINE distribution; the fitter only
reads it, validates it against the CLI, and propagates it into its outputs. It
must never change the contact/Rg objectives.

Run:  python -m pytest "test_fit_kappa_bend.py" -q
"""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import fit_lattice_contact_model_2 as fit

FITTER = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "fit_lattice_contact_model_2.py")


# ---------------------------------------------------------------------------
# Synthetic inputs (small enough that the fitter runs in a couple of seconds)
# ---------------------------------------------------------------------------

def _write_baseline(path, kappa_bend=None):
    """Joint baseline NPZ; omit kappa_bend entirely to emulate a legacy file."""
    m_vals = np.arange(0, 21)
    rg_edges = np.linspace(1.0, 6.0, 21)
    rg_centers = 0.5 * (rg_edges[:-1] + rg_edges[1:])

    p_m = np.exp(-0.5 * ((m_vals - 7.0) / 4.0) ** 2)
    p_m /= p_m.sum()
    # Rg shrinks as contacts grow: a plausible, strictly positive joint.
    joint = np.exp(
        -0.5 * ((rg_centers[None, :] - (5.0 - 0.12 * m_vals[:, None])) / 0.8) ** 2
    )
    joint *= p_m[:, None] / joint.sum(axis=1, keepdims=True)
    joint /= joint.sum()

    kwargs = dict(
        c_vals=m_vals.astype(np.int64),
        c_prob=joint.sum(axis=1),
        c_edges=np.arange(m_vals[0] - 0.5, m_vals[-1] + 1.5, 1.0),
        rg_edges=rg_edges,
        rg_prob=joint.sum(axis=0),
        crg_prob=joint,
        N=20,
    )
    if kappa_bend is not None:
        kwargs["kappa_bend"] = float(kappa_bend)
        kwargs["bending_enabled"] = bool(kappa_bend != 0.0)
        kwargs["bend_definition"] = fit.BEND_DEFINITION
    np.savez_compressed(path, **kwargs)
    return path


def _write_remd(path, baseline_path, h=600.0, s=2.0):
    """REMD-style observed histograms generated from the baseline by exp[-b(T) m]."""
    b = np.load(baseline_path)
    m_vals = np.asarray(b["c_vals"], dtype=float)
    p0 = np.asarray(b["c_prob"], dtype=float)

    temps = np.linspace(290.0, 350.0, 10)
    hists = np.zeros((temps.size, m_vals.size), dtype=float)
    for i, T in enumerate(temps):
        w = p0 * np.exp(-(h / T - s) * m_vals)
        hists[i] = w / w.sum()

    np.savez_compressed(
        path, temps=temps, ct_centers=m_vals, ct_hists=hists,
    )
    return path


def _run_fitter(tmpdir, remd, baseline, outdir, *extra):
    cmd = [
        sys.executable, FITTER,
        "--remd", remd, "--baseline", baseline,
        "--contact_offset", "0", "--no-plots",
        "--n_restarts", "1", "--seed", "123",
        "--outdir", outdir,
        *extra,
    ]
    return subprocess.run(cmd, cwd=tmpdir, capture_output=True, text=True)


# ---------------------------------------------------------------------------
# 8: legacy baselines are interpreted as kappa_bend = 0
# ---------------------------------------------------------------------------

def test_legacy_baseline_reads_as_zero(tmp_path):
    path = _write_baseline(str(tmp_path / "legacy.npz"), kappa_bend=None)
    data = np.load(path)
    assert "kappa_bend" not in data.files
    assert fit.read_baseline_kappa_bend(data) == 0.0
    # With no CLI value the legacy baseline resolves to a plain athermal fit.
    assert fit.resolve_kappa_bend(fit.read_baseline_kappa_bend(data), None) == 0.0


def test_baseline_kappa_is_read_back(tmp_path):
    path = _write_baseline(str(tmp_path / "stiff.npz"), kappa_bend=1.25)
    assert fit.read_baseline_kappa_bend(np.load(path)) == 1.25


def test_baseline_kappa_must_be_finite_and_nonnegative(tmp_path):
    for bad in (-0.5, float("nan")):
        path = str(tmp_path / f"bad_{bad}.npz")
        np.savez_compressed(path, c_vals=np.arange(3), c_prob=np.full(3, 1 / 3),
                            kappa_bend=float(bad))
        with pytest.raises(ValueError):
            fit.read_baseline_kappa_bend(np.load(path))


# ---------------------------------------------------------------------------
# 9: the fitter rejects a CLI/baseline mismatch
# ---------------------------------------------------------------------------

def test_resolve_rejects_mismatch():
    with pytest.raises(ValueError, match="does not match the baseline"):
        fit.resolve_kappa_bend(0.0, 0.5)
    with pytest.raises(ValueError, match="does not match the baseline"):
        fit.resolve_kappa_bend(1.5, 0.0)


def test_resolve_accepts_match_within_tight_tolerance():
    assert fit.resolve_kappa_bend(0.5, 0.5) == 0.5
    assert fit.resolve_kappa_bend(0.5, 0.5 + 1e-12) == 0.5
    # ... but not a difference far above the tolerance.
    with pytest.raises(ValueError):
        fit.resolve_kappa_bend(0.5, 0.5 + 1e-6)


def test_resolve_rejects_bad_cli_values():
    with pytest.raises(ValueError, match="must be >= 0"):
        fit.resolve_kappa_bend(0.0, -1.0)
    with pytest.raises(ValueError, match="must be finite"):
        fit.resolve_kappa_bend(0.0, float("inf"))


def test_cli_mismatch_fails_the_run(tmp_path):
    baseline = _write_baseline(str(tmp_path / "stiff.npz"), kappa_bend=1.0)
    remd = _write_remd(str(tmp_path / "remd.npz"), baseline)
    proc = _run_fitter(str(tmp_path), remd, baseline, str(tmp_path / "out_bad"),
                       "--kappa-bend", "0.0")
    assert proc.returncode != 0
    assert "does not match the baseline" in proc.stderr


# ---------------------------------------------------------------------------
# 10: kappa metadata propagates without disturbing the contact fit
# ---------------------------------------------------------------------------

def test_kappa_metadata_propagates_without_changing_the_fit(tmp_path):
    baseline = _write_baseline(str(tmp_path / "stiff.npz"), kappa_bend=1.0)
    remd = _write_remd(str(tmp_path / "remd.npz"), baseline)

    out_implicit = tmp_path / "out_implicit"
    out_explicit = tmp_path / "out_explicit"
    p1 = _run_fitter(str(tmp_path), remd, baseline, str(out_implicit))
    p2 = _run_fitter(str(tmp_path), remd, baseline, str(out_explicit),
                     "--kappa-bend", "1.0")
    assert p1.returncode == 0, p1.stderr
    assert p2.returncode == 0, p2.stderr

    r1 = np.load(out_implicit / "fit_results.npz", allow_pickle=True)
    r2 = np.load(out_explicit / "fit_results.npz", allow_pickle=True)

    # Passing the (matching) CLI value changes nothing about the fit itself.
    assert np.array_equal(r1["params"], r2["params"])
    assert float(r1["train_loss"]) == float(r2["train_loss"])
    assert float(r1["all_loss"]) == float(r2["all_loss"])

    for r in (r1, r2):
        assert float(r["kappa_bend"]) == 1.0
        assert bool(r["bending_enabled"]) is True
        assert str(r["bend_definition"]) == fit.BEND_DEFINITION

    for outdir in (out_implicit, out_explicit):
        with open(outdir / "fit_summary.json") as fh:
            summary = json.load(fh)
        assert summary["kappa_bend"] == 1.0
        assert summary["bending_enabled"] is True
        assert summary["bend_definition"] == fit.BEND_DEFINITION
        # The contract the suite depends on is untouched.
        assert summary["model_api_version"] == fit.MODEL_API_VERSION


def test_legacy_baseline_end_to_end_reports_zero(tmp_path):
    baseline = _write_baseline(str(tmp_path / "legacy.npz"), kappa_bend=None)
    remd = _write_remd(str(tmp_path / "remd.npz"), baseline)
    outdir = tmp_path / "out_legacy"
    proc = _run_fitter(str(tmp_path), remd, baseline, str(outdir))
    assert proc.returncode == 0, proc.stderr

    res = np.load(outdir / "fit_results.npz", allow_pickle=True)
    assert float(res["kappa_bend"]) == 0.0
    assert bool(res["bending_enabled"]) is False
    with open(outdir / "fit_summary.json") as fh:
        assert json.load(fh)["kappa_bend"] == 0.0
