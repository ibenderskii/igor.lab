#!/usr/bin/env python3
"""HDF5 snapshot metadata regression for the Aggregation entrypoints.

A saved ``_configurations.h5`` must describe the Hamiltonian that was actually
sampled.  For a contact-quadratic model run with a nonzero bending penalty the
``/metadata`` group must carry all six provenance fields -- the fixed bending
penalty (``kappa_bend``/``bending_enabled``/``bend_definition``) and the
contact-potential contract (``potential_kind``/``quadratic_normalization``/
``fit_chain_length``) -- with values that match the run.  ``fit_chain_length``
must be the chain length the model was FIT at (from the fit summary), NOT the
runtime chain length used in m^2/(2N); the run below deliberately uses a runtime
``--N`` different from the fit length to pin that distinction down.

``saturating_cooperative_contact`` has no ``m^2/(2N)`` term at all, so a snapshot
of such a run must NOT claim a quadratic normalization; it has to state its own
contract instead -- ``potential_definition``, ``potential_normalization``
(``q = m/N``), ``m_ref``, and the fitted ``A0``/``q_sat`` -- for both the
single-chain and the multi-chain entrypoint.
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
_SCRIPT = _HERE / "remd_uniform_chain_2_new.py"
_MULTICHAIN_SCRIPT = _HERE / "remd_multichain.py"

SAT_MODEL = "saturating_cooperative_contact"
SAT_PARAMS = {"h_b": 700.0, "s_b": 2.4, "A0": 5.0, "q_sat": 0.15}
Q_NORMALIZATION = "q = m/N"


def _h5py_or_skip():
    try:
        import h5py  # noqa: F401
        return h5py
    except Exception:  # pragma: no cover - depends on local stack
        pytest.skip("h5py unavailable")


def _run_or_fail(script, argv, cwd):
    proc = subprocess.run(
        [sys.executable, str(script)] + list(argv),
        cwd=str(cwd), capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, (
        f"entrypoint failed.\nstdout tail:\n{proc.stdout[-2000:]}\n"
        f"stderr tail:\n{proc.stderr[-2000:]}"
    )
    return proc


def test_quadratic_snapshot_records_full_hamiltonian_metadata():
    if not _SCRIPT.is_file():
        pytest.skip(f"entry-point script not found: {_SCRIPT}")
    h5py = _h5py_or_skip()
    import lattice_bending  # co-located; the authoritative bend definition

    fit_chain_length = 30
    runtime_N = 20
    kappa_bend = 0.5
    summary = {
        "model_api_version": 2,
        "model": "hs_m2_const",
        "params": {"h1": 700.0, "s1": 2.4, "kappa2": 0.9},
        "Tref": 320.0,
        "Tscale": 80.0,
        "kappa_bend": kappa_bend,
        "fit_chain_length": fit_chain_length,
    }

    with tempfile.TemporaryDirectory() as tmp:
        summ_path = os.path.join(tmp, "fit_summary.json")
        with open(summ_path, "w") as fh:
            json.dump(summary, fh)
        out_prefix = os.path.join(tmp, "run")
        proc = subprocess.run(
            [sys.executable, str(_SCRIPT),
             "--fit-summary-json", summ_path,
             "--N", str(runtime_N),
             "--Tmin", "300", "--Tmax", "360", "--nT", "4",
             "--steps-per-swap", "8", "--n-cycles", "5", "--seed", "1",
             "--save-configurations", "--out-prefix", out_prefix],
            cwd=str(_SCRIPT.parent),
            capture_output=True, text=True, timeout=600,
        )
        assert proc.returncode == 0, (
            f"entrypoint failed.\nstdout tail:\n{proc.stdout[-2000:]}\n"
            f"stderr tail:\n{proc.stderr[-2000:]}"
        )
        cfg = out_prefix + "_configurations.h5"
        assert os.path.exists(cfg), f"no configurations file written: {cfg}"

        with h5py.File(cfg, "r") as f:
            meta = f["metadata"].attrs
            for key in ("kappa_bend", "bending_enabled", "bend_definition",
                        "potential_kind", "quadratic_normalization",
                        "fit_chain_length"):
                assert key in meta, f"missing snapshot metadata field: {key}"
            assert float(meta["kappa_bend"]) == pytest.approx(kappa_bend)
            assert bool(meta["bending_enabled"]) is True
            assert str(meta["bend_definition"]) == lattice_bending.BEND_DEFINITION
            assert str(meta["potential_kind"]) == "contact_quadratic"
            assert str(meta["quadratic_normalization"]) == "m^2/(2N)"
            # The FIT chain length, not the runtime chain length.
            assert int(meta["fit_chain_length"]) == fit_chain_length
            assert int(meta["n_beads"]) == runtime_N
            assert int(meta["fit_chain_length"]) != int(meta["n_beads"])


def test_saturating_snapshot_records_full_hamiltonian_metadata():
    """Single-chain snapshot of a saturating-cooperative run."""
    if not _SCRIPT.is_file():
        pytest.skip(f"entry-point script not found: {_SCRIPT}")
    h5py = _h5py_or_skip()
    import lattice_bending  # co-located; the authoritative bend definition
    import remd_uniform_chain_2_new as remd

    fit_chain_length = 30
    runtime_N = 20
    kappa_bend = 0.5
    summary = {
        "model_api_version": remd.MODEL_API_VERSION,
        "model": SAT_MODEL,
        "params": dict(SAT_PARAMS),
        "Tref": 320.0,
        "Tscale": 80.0,
        "kappa_bend": kappa_bend,
        "fit_chain_length": fit_chain_length,
    }

    with tempfile.TemporaryDirectory() as tmp:
        summ_path = os.path.join(tmp, "fit_summary.json")
        with open(summ_path, "w") as fh:
            json.dump(summary, fh)
        out_prefix = os.path.join(tmp, "run")
        _run_or_fail(_SCRIPT, [
            "--fit-summary-json", summ_path,
            "--N", str(runtime_N),
            "--Tmin", "300", "--Tmax", "360", "--nT", "4",
            "--steps-per-swap", "8", "--n-cycles", "5", "--seed", "1",
            "--save-configurations", "--out-prefix", out_prefix], _SCRIPT.parent)
        cfg = out_prefix + "_configurations.h5"
        assert os.path.exists(cfg), f"no configurations file written: {cfg}"

        with h5py.File(cfg, "r") as f:
            meta = f["metadata"].attrs
            for key in ("kappa_bend", "bending_enabled", "bend_definition",
                        "potential_kind", "quadratic_normalization",
                        "potential_definition", "potential_normalization",
                        "m_ref", "fit_chain_length"):
                assert key in meta, f"missing snapshot metadata field: {key}"
            assert float(meta["kappa_bend"]) == pytest.approx(kappa_bend)
            assert bool(meta["bending_enabled"]) is True
            assert str(meta["bend_definition"]) == lattice_bending.BEND_DEFINITION
            assert str(meta["potential_kind"]) == "saturating_cooperative"
            # This family has no m^2/(2N) term, so the quadratic slot is null and
            # the real contract lives in the saturating fields.
            assert str(meta["quadratic_normalization"]) == "null"
            assert str(meta["potential_definition"]) == \
                remd.SATURATING_COOPERATIVE_DEFINITION
            assert str(meta["potential_normalization"]) == Q_NORMALIZATION
            assert int(meta["m_ref"]) == 0
            # The FIT chain length, not the runtime chain length.
            assert int(meta["fit_chain_length"]) == fit_chain_length
            assert int(meta["n_beads"]) == runtime_N
            assert int(meta["fit_chain_length"]) != int(meta["n_beads"])


def test_multichain_saturating_snapshot_records_full_hamiltonian_metadata():
    """Snapshot metadata records the label-blind saturation Hamiltonian."""
    if not _MULTICHAIN_SCRIPT.is_file():
        pytest.skip(f"entry-point script not found: {_MULTICHAIN_SCRIPT}")
    h5py = _h5py_or_skip()
    import lattice_bending
    import remd_uniform_chain_2_new as remd
    import remd_multichain as rmc

    fit_chain_length = 30
    runtime_N = 8
    kappa_bend = 0.4
    summary = {
        "model_api_version": remd.MODEL_API_VERSION,
        "model": SAT_MODEL,
        "params": dict(SAT_PARAMS),
        "Tref": 320.0,
        "Tscale": 80.0,
        "kappa_bend": kappa_bend,
        "fit_chain_length": fit_chain_length,
    }

    with tempfile.TemporaryDirectory() as tmp:
        summ_path = os.path.join(tmp, "fit_summary.json")
        with open(summ_path, "w") as fh:
            json.dump(summary, fh)
        out_prefix = os.path.join(tmp, "mc_run")
        _run_or_fail(_MULTICHAIN_SCRIPT, [
            "--fit-summary-json", summ_path,
            "--n-chains", "2", "--N", str(runtime_N), "--box-size", "12",
            "--Tmin", "310", "--Tmax", "345", "--nT", "3",
            "--n-cycles", "6", "--seed", "1", "--no-plots",
            "--save-configurations", "--out-prefix", out_prefix],
            _MULTICHAIN_SCRIPT.parent)
        cfg = out_prefix + "_configurations.h5"
        assert os.path.exists(cfg), f"no configurations file written: {cfg}"

        with h5py.File(cfg, "r") as f:
            meta = f["metadata"].attrs
            for key in ("kappa_bend", "bending_enabled", "bend_definition",
                        "potential_kind", "quadratic_normalization",
                        "quadratic_contact_scope", "nonlinear_contact_scope",
                        "interchain_contact_model", "potential_definition",
                        "potential_normalization", "m_ref",
                        "multichain_potential_definition", "A0", "q_sat",
                        "runtime_chain_length", "fit_chain_length"):
                assert key in meta, f"missing snapshot metadata field: {key}"
            assert float(meta["kappa_bend"]) == pytest.approx(kappa_bend)
            assert bool(meta["bending_enabled"]) is True
            assert str(meta["bend_definition"]) == lattice_bending.BEND_DEFINITION
            assert str(meta["potential_kind"]) == "saturating_cooperative"
            assert str(meta["quadratic_normalization"]) == "null"
            assert str(meta["potential_definition"]) == \
                remd.SATURATING_COOPERATIVE_DEFINITION
            assert str(meta["potential_normalization"]) == Q_NORMALIZATION
            assert int(meta["m_ref"]) == 0
            assert str(meta["multichain_potential_definition"]) == \
                rmc.MULTICHAIN_POTENTIAL_DEFINITION
            assert str(meta["quadratic_contact_scope"]) == \
                "all_contacts_global"
            assert str(meta["nonlinear_contact_scope"]) == \
                "all_contacts_global"
            assert str(meta["interchain_contact_model"]) == \
                "same_single_chain_potential"
            assert float(meta["A0"]) == pytest.approx(SAT_PARAMS["A0"])
            assert float(meta["q_sat"]) == pytest.approx(SAT_PARAMS["q_sat"])
            # Runtime N drives q = m/N; the fit length is provenance only.
            assert int(meta["runtime_chain_length"]) == runtime_N
            assert int(meta["fit_chain_length"]) == fit_chain_length
            assert int(meta["N"]) == runtime_N


def test_multichain_quadratic_snapshot_metadata_unchanged():
    """The legacy contact-quadratic snapshot contract is untouched."""
    if not _MULTICHAIN_SCRIPT.is_file():
        pytest.skip(f"entry-point script not found: {_MULTICHAIN_SCRIPT}")
    h5py = _h5py_or_skip()

    with tempfile.TemporaryDirectory() as tmp:
        out_prefix = os.path.join(tmp, "mc_q")
        _run_or_fail(_MULTICHAIN_SCRIPT, [
            "--model", "hs_m2_const", "--params", "700,2.4,0.9",
            "--Tref", "320", "--Tscale", "80",
            "--n-chains", "2", "--N", "8", "--box-size", "12",
            "--Tmin", "310", "--Tmax", "345", "--nT", "3",
            "--n-cycles", "6", "--seed", "2", "--no-plots",
            "--save-configurations", "--out-prefix", out_prefix],
            _MULTICHAIN_SCRIPT.parent)
        with h5py.File(out_prefix + "_configurations.h5", "r") as f:
            meta = f["metadata"].attrs
            assert str(meta["potential_kind"]) == "contact_quadratic"
            assert str(meta["quadratic_normalization"]) == "m_chain^2/(2*N)"
            assert str(meta["quadratic_contact_scope"]) == "intra_per_chain"
            assert str(meta["potential_normalization"]) == "m^2/(2N)"
            assert int(meta["m_ref"]) == 0
            assert "A0" not in meta and "q_sat" not in meta


if __name__ == "__main__":
    sys.exit(pytest.main([os.path.abspath(__file__), "-q"]))
