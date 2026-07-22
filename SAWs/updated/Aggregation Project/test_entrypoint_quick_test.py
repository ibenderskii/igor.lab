#!/usr/bin/env python3
"""Entry-point regression: the single-chain Aggregation script must run directly.

    python "SAWs/updated/Aggregation Project/remd_uniform_chain_2_new.py" --quick-test

must succeed from the repository root WITHOUT the caller setting
``ISAW_PROJECT_DEFINITIONS`` or ``PYTHONPATH`` by hand.  The script locates the
authoritative ``project_definitions.json`` and ``extract_contact_motif_features``
(which live in the sibling ``ISAW Project`` directory) via paths derived from
``__file__``; a regression there reintroduces the schema-resolution crash that
aborts the run before ``check_definitions_consistency`` even starts.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

# repo_root/SAWs/updated/Aggregation Project/<this file>
_SCRIPT = Path(__file__).resolve().parent / "remd_uniform_chain_2_new.py"
_REPO_ROOT = Path(__file__).resolve().parents[3]


def test_quick_test_runs_from_repo_root_without_path_config():
    if not _SCRIPT.is_file():
        pytest.skip(f"entry-point script not found: {_SCRIPT}")
    # Invoke exactly as documented: a repository-relative script path with the
    # working directory at the repository root.
    rel_script = _SCRIPT.relative_to(_REPO_ROOT)
    # Sanitize the environment so nothing external supplies the definitions path
    # or import roots -- the fix must stand on its own.
    env = dict(os.environ)
    env.pop("ISAW_PROJECT_DEFINITIONS", None)
    env.pop("PYTHONPATH", None)
    proc = subprocess.run(
        [sys.executable, str(rel_script), "--quick-test"],
        cwd=str(_REPO_ROOT), env=env,
        capture_output=True, text=True, timeout=600,
    )
    assert proc.returncode == 0, (
        "entry-point quick-test failed from the repository root without manual "
        f"path configuration.\nstdout tail:\n{proc.stdout[-2000:]}\n"
        f"stderr tail:\n{proc.stderr[-2000:]}"
    )
    # It must get all the way through (past the schema-consistency check that used
    # to crash), not merely start.
    assert "quick-test complete." in proc.stdout


if __name__ == "__main__":
    sys.exit(pytest.main([os.path.abspath(__file__), "-q"]))
