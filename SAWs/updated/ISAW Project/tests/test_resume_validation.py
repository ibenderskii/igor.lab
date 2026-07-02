"""Part 2/14: resume accepts a file ONLY when it deep-validates (certificate)."""
import json
import os
import shutil
import tempfile

import numpy as np
import pytest

import isaw_contact_observables as ico
import isaw_config_io as cio
import extract_contact_motif_features as ext
import run_structural_regime_pilot as cal

pytestmark = pytest.mark.skipif(not cio.h5py_available(), reason="h5py missing")

HAIRPIN6 = [(0, 0, 0), (1, 0, 0), (2, 0, 0), (2, 1, 0), (1, 1, 0), (0, 1, 0)]
MODEL_META = {"model_name": "hs", "param_names": ["h", "s"],
              "model_params": [647.7, 1.874], "Tref": 320.0, "Tscale": 80.0}


def _make_feature_file(tmp):
    nb, nT = len(HAIRPIN6), 2
    cp, _ = ico.build_contact_map(HAIRPIN6)
    m = cp.shape[0]
    rg2 = ico.radius_of_gyration_squared(HAIRPIN6)
    ree2 = ico.end_to_end_distance_squared(HAIRPIN6)
    meta = {"run_id": "cf", "seed": 1, "temperatures": [300.0, 350.0]}
    meta.update(MODEL_META)
    inp = os.path.join(tmp, "snap.h5")
    w = cio.SnapshotWriter(inp, n_beads=nb, n_temperatures=nT, metadata=meta)
    for c in range(3):
        w.append(cycle=c,
                 coordinates=np.stack([np.asarray(HAIRPIN6, dtype=np.int64)] * nT),
                 walker_id=np.array([0, 1] if c % 2 == 0 else [1, 0]),
                 contacts=np.full(nT, m), rg2_lattice=np.full(nT, rg2),
                 ree2_lattice=np.full(nT, ree2))
    w.mark_complete()
    w.close()
    out = os.path.join(tmp, "feat.h5")
    ext.extract(inp, out, validate=True, output_format="hdf5", overwrite=True)
    return out


def _corrupt(out, fn):
    dst = out + ".bad.h5"
    shutil.copy(out, dst)
    import h5py
    with h5py.File(dst, "r+") as f:
        fn(f)
    return dst


def _set(f, path, idx, value):
    d = f[path]; arr = d[()]; arr[idx] = value; d[...] = arr


CORRUPT = {
    "bonded_pair": lambda f: _set(f, "features/contact_pairs/pairs", 0, [0, 1]),
    "motif": lambda f: _set(f, "features/scalars/pair_nested", 0,
                            int(f["features/scalars/pair_nested"][0]) + 1),
    "thermodynamic": lambda f: _set(f, "features/scalars/K_T", 0, 12345.0),
    "graph_float": lambda f: _set(
        f, "features/scalars/largest_component_fraction_of_N", 0, 0.99),
}


def test_resume_accepts_valid_file_and_writes_certificate():
    with tempfile.TemporaryDirectory() as tmp:
        out = _make_feature_file(tmp)
        assert cal._feature_certified(out) is True
        cert = cal._certificate_path(out)
        assert cert.exists()
        data = json.loads(cert.read_text(encoding="utf-8"))
        assert data["validation_result"] == "passed"
        assert data["feature_sha256"] == cal._sha256(out)


@pytest.mark.parametrize("name", list(CORRUPT))
def test_resume_rejects_corruption(name):
    with tempfile.TemporaryDirectory() as tmp:
        out = _make_feature_file(tmp)
        bad = _corrupt(out, CORRUPT[name])
        # No certificate exists for the corrupted copy -> deep validation reruns
        # and fails -> not certified.
        assert cal._feature_certified(bad) is False


def test_resume_rejects_stale_certificate():
    with tempfile.TemporaryDirectory() as tmp:
        out = _make_feature_file(tmp)
        assert cal._feature_certified(out) is True      # writes a certificate
        # Corrupt the file in place; the certificate hash no longer matches, so
        # deep validation reruns and fails.
        import h5py
        with h5py.File(out, "r+") as f:
            _set(f, "features/contact_pairs/pairs", 0, [0, 1])
        assert cal._feature_certified(out) is False


def test_shallow_check_is_not_sufficient():
    # A corrupted file that a shallow (deep=False) validator might miss must
    # still be rejected by the certificate path (which is deep).
    with tempfile.TemporaryDirectory() as tmp:
        out = _make_feature_file(tmp)
        bad = _corrupt(out, CORRUPT["motif"])
        assert cal._feature_certified(bad) is False
