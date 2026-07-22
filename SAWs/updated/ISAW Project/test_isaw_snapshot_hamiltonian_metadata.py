#!/usr/bin/env python3
"""HDF5 snapshot metadata regression for the ISAW single-chain entrypoint.

A saved ``_configurations.h5`` must describe the Hamiltonian that was actually
sampled.  For a contact-quadratic model run with a nonzero bending penalty the
``/metadata`` group must carry all six provenance fields -- the fixed bending
penalty (``kappa_bend``/``bending_enabled``/``bend_definition``) and the
contact-potential contract (``potential_kind``/``quadratic_normalization``/
``fit_chain_length``) -- with values that match the run.  ``fit_chain_length``
must be the chain length the model was FIT at (from the fit summary), NOT the
runtime chain length used in m^2/(2N); the run below deliberately uses a runtime
``--N`` different from the fit length to pin that distinction down.
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parent / "remd_uniform_chain_2_new.py"


def _h5py_or_skip():
    try:
        import h5py  # noqa: F401
        return h5py
    except Exception:  # pragma: no cover - depends on local stack
        pytest.skip("h5py unavailable")


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


if __name__ == "__main__":
    sys.exit(pytest.main([os.path.abspath(__file__), "-q"]))
