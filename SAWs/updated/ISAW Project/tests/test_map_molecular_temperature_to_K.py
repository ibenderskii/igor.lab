"""Targeted tests for map_molecular_temperature_to_K.py.

Covers: density->mass normalization, contact offset + integer-bin integration,
underflow/overflow preservation, Rg scaling+rebinning, JS(identical)=0, synthetic
recovery of a known best K, duplicate-K seed pooling with equal seed weights,
boundary-hit detection, quadratic-refinement rejection at endpoints, and an
end-to-end smoke run producing all required files.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import map_molecular_temperature_to_K as m

HERE = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- #
# synthetic-input helpers
# --------------------------------------------------------------------------- #
def make_direct_npz(path, K_values, Pc, rg_edges, Prg, seed, N=30):
    np.savez(
        path,
        control_mode="direct_K",
        temperature_mapping_applied=np.bool_(False),
        K_values=np.asarray(K_values, float),
        coupling_K_by_temperature=np.asarray(K_values, float),
        c_vals=np.arange(np.asarray(Pc).shape[1], dtype=int),
        Pc=np.asarray(Pc, float),
        rg_edges_lattice=np.asarray(rg_edges, float),
        Prg=np.asarray(Prg, float),
        seed=int(seed),
        N=int(N),
    )


def make_molecular_npz(path, temps, ct_centers, ct_hists, rg_centers, rg_hists):
    np.savez(
        path,
        temps=np.asarray(temps, float),
        ct_centers=np.asarray(ct_centers, float),
        ct_hists=np.asarray(ct_hists, float),
        rg_centers=np.asarray(rg_centers, float),
        rg_hists=np.asarray(rg_hists, float),
    )


def make_fit_summary(path, contact_offset=0.0, rg_scale=1.0, h=100.0, s=1.0):
    path.write_text(json.dumps({
        "model": "hs", "params": {"h": h, "s": s},
        "contact_offset": contact_offset, "rg_scale": rg_scale,
    }), encoding="utf-8")


def _args(**kw):
    base = dict(
        molecular_npz=None, direct_k_npz=None, fit_summary_json=None,
        contact_weight=0.5, rg_weight=0.5,
        contact_rg_disagreement_threshold=0.10,
        transition_low=0.58, transition_high=0.66,
        output_dir=None, overwrite=True,
    )
    base.update(kw)
    return argparse.Namespace(**base)


# 1. density -> mass normalization ------------------------------------------ #
def test_density_to_mass_normalization():
    centers = np.array([0.0, 1.0, 2.0, 3.0])
    mass = m.density_to_mass(np.array([1.0, 1.0, 1.0, 1.0]), centers)
    assert mass == pytest.approx([0.25, 0.25, 0.25, 0.25])
    assert mass.sum() == pytest.approx(1.0)
    # A spike density integrates to a single unit bin.
    mass2 = m.density_to_mass(np.array([0.0, 2.0, 0.0, 0.0]), centers)
    assert mass2 == pytest.approx([0.0, 1.0, 0.0, 0.0])
    # Non-unit spacing: mass = density * width, then normalized.
    c2 = np.array([0.0, 0.5, 1.0])
    mass3 = m.density_to_mass(np.array([2.0, 0.0, 2.0]), c2)
    assert mass3 == pytest.approx([0.5, 0.0, 0.5])


# 2. contact offset + integration into integer bins ------------------------- #
def test_contact_offset_and_integer_binning():
    ct_centers = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    row = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])  # unit at native 3
    full = m.molecular_contact_mass(ct_centers, row, contact_offset=2.0, c_max=5)
    # full = [underflow, m0..m5, overflow]; native 3 - offset 2 -> m=1.
    assert full.size == 5 + 2 + 1
    expected = np.zeros(8)
    expected[1 + 1] = 1.0  # underflow(0) then m=1
    assert full == pytest.approx(expected)
    assert full.sum() == pytest.approx(1.0)


# 3. underflow / overflow mass preservation --------------------------------- #
def test_underflow_overflow_preserved():
    ct_centers = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    row = np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.5])  # half native 0, half native 5
    # offset 3 -> native 0 => shifted -3 (underflow); native 5 => shifted 2.
    # c_max=1 => m in {0,1}; shifted 2 > 1.5 => overflow.
    full = m.molecular_contact_mass(ct_centers, row, contact_offset=3.0, c_max=1)
    assert full.size == 1 + 2 + 1  # underflow, m0, m1, overflow
    assert full[0] == pytest.approx(0.5)   # underflow
    assert full[-1] == pytest.approx(0.5)  # overflow
    assert full[1:-1] == pytest.approx([0.0, 0.0])
    assert full.sum() == pytest.approx(1.0)  # nothing discarded


# 4. Rg scaling + rebinning -------------------------------------------------- #
def test_rg_scaling_and_rebinning():
    # Lattice edges [1,2,3] with mass [0.5,0.5]; scale 2 -> [2,4,6].
    lattice_edges = np.array([1.0, 2.0, 3.0])
    mass = np.array([0.5, 0.5])
    scaled = lattice_edges * 2.0
    # Molecular grid exactly matches -> identity, no under/overflow.
    out = m.redistribute_mass_to_grid(scaled, mass, np.array([2.0, 4.0, 6.0]))
    assert out == pytest.approx([0.0, 0.5, 0.5, 0.0])
    # Molecular grid covers only the first scaled bin -> second bin overflows.
    out2 = m.redistribute_mass_to_grid(scaled, mass, np.array([2.0, 4.0]))
    assert out2 == pytest.approx([0.0, 0.5, 0.5])  # underflow, [2,4], overflow
    assert out2.sum() == pytest.approx(1.0)


# 5. JS divergence is zero for identical distributions ---------------------- #
def test_js_zero_and_bounds():
    p = np.array([0.2, 0.3, 0.5])
    assert m.js_divergence(p, p) == pytest.approx(0.0, abs=1e-15)
    # Disjoint supports -> ln 2 (natural log).
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 0.0, 1.0])
    assert m.js_divergence(a, b) == pytest.approx(m.LN2, abs=1e-12)
    # Symmetry.
    q = np.array([0.5, 0.25, 0.25])
    assert m.js_divergence(p, q) == pytest.approx(m.js_divergence(q, p), abs=1e-15)


# 6. synthetic recovery of a known best K ----------------------------------- #
def _recovery_setup(tmp_path, molecular_index):
    K = [0.2, 0.5, 0.8]
    Pc = np.array([[0.6, 0.3, 0.1],
                   [0.2, 0.5, 0.3],
                   [0.1, 0.3, 0.6]])
    Prg = np.array([[0.6, 0.3, 0.1],
                    [0.2, 0.5, 0.3],
                    [0.1, 0.3, 0.6]])
    rg_edges = np.array([0.5, 1.5, 2.5, 3.5])
    d1 = tmp_path / "d1.npz"
    make_direct_npz(d1, K, Pc, rg_edges, Prg, seed=101)
    mol = tmp_path / "mol.npz"
    make_molecular_npz(
        mol, temps=[300.0], ct_centers=[0.0, 1.0, 2.0],
        ct_hists=[Pc[molecular_index]], rg_centers=[1.0, 2.0, 3.0],
        rg_hists=[Prg[molecular_index]],
    )
    fs = tmp_path / "fit.json"
    make_fit_summary(fs, contact_offset=0.0, rg_scale=1.0)
    out = tmp_path / "out"
    args = _args(molecular_npz=str(mol), direct_k_npz=[str(d1)],
                 fit_summary_json=str(fs), output_dir=str(out))
    return m.run_mapping(args), K


def test_recovery_of_known_best_K(tmp_path):
    result, K = _recovery_setup(tmp_path, molecular_index=1)
    rec = result["report"]["mapping"][0]
    assert rec["K_best_combined_grid"] == pytest.approx(0.5)
    assert rec["K_best_contact_grid"] == pytest.approx(0.5)
    assert rec["K_best_rg_grid"] == pytest.approx(0.5)
    assert rec["combined_score_best"] == pytest.approx(0.0, abs=1e-12)
    assert rec["contact_rg_disagreement_raw"] is False
    # Middle K is interior, so both component optima are resolved.
    assert rec["contact_rg_disagreement_resolved"] is False
    # Component minimum equals the component score at the (same) combined optimum.
    assert rec["JS_contact_min"] == pytest.approx(rec["JS_contact_at_combined"])
    assert rec["JS_rg_min"] == pytest.approx(rec["JS_rg_at_combined"])


# 7. duplicate-K pooling with equal seed weights ---------------------------- #
def test_duplicate_K_pooling_equal_seed_weights():
    rg_edges = np.array([1.0, 2.0, 3.0])
    Prg = np.array([[0.5, 0.5]])
    A = np.array([[0.7, 0.2, 0.1]])   # seed 1, run 1
    B = np.array([[0.1, 0.2, 0.7]])   # seed 1, run 2
    C = np.array([[0.3, 0.4, 0.3]])   # seed 2, run 1
    f1 = {"path": "A", "N": 30, "seed": 1, "K": np.array([0.5]),
          "c_int": np.array([0, 1, 2]), "Pc": A,
          "rg_edges": rg_edges, "Prg": Prg}
    f2 = {"path": "B", "N": 30, "seed": 1, "K": np.array([0.5]),
          "c_int": np.array([0, 1, 2]), "Pc": B,
          "rg_edges": rg_edges, "Prg": Prg}
    f3 = {"path": "C", "N": 30, "seed": 2, "K": np.array([0.5]),
          "c_int": np.array([0, 1, 2]), "Pc": C,
          "rg_edges": rg_edges, "Prg": Prg}
    lib = m.build_reference_library([f1, f2, f3], rg_scale=1.0,
                                    mol_rg_edges=np.array([1.0, 2.0, 3.0]))
    assert lib["K"] == pytest.approx([0.5])
    assert lib["n_seeds"][0] == 2
    # Seed 1 averaged first (A,B) -> midpoint; then equal-weighted with seed 2 (C).
    seed1 = 0.5 * (A[0] + B[0])
    expected = 0.5 * seed1 + 0.5 * C[0]
    assert lib["pooled_contact"][0] == pytest.approx(expected)


# 8. boundary-hit detection ------------------------------------------------- #
def test_boundary_hit_detection(tmp_path):
    # Molecular matches the LOWEST K -> all component optima at the low boundary.
    result, K = _recovery_setup(tmp_path, molecular_index=0)
    rec = result["report"]["mapping"][0]
    assert rec["K_best_combined_grid"] == pytest.approx(min(K))
    assert rec["combined_boundary_hit_low"] is True
    assert rec["combined_boundary_hit_high"] is False
    assert rec["contact_boundary_hit_low"] is True
    assert rec["rg_boundary_hit_low"] is True
    # Both component optima are censored -> no resolved agreement claim.
    assert rec["contact_rg_disagreement_resolved"] is None
    # A boundary combined optimum censors the old-mapping difference.
    assert rec["K_difference_from_old"] is None
    assert rec["old_mapping_comparison_resolved"] is False


# 9. quadratic-refinement rejection at endpoints ---------------------------- #
def test_quadratic_refinement_rules():
    K = np.array([0.0, 1.0, 2.0])
    # Interior convex minimum -> vertex returned.
    assert m.quadratic_refine(K, np.array([1.0, 0.0, 1.0]), 1) == pytest.approx(1.0)
    # Endpoint minimum -> rejected (NaN).
    assert np.isnan(m.quadratic_refine(K, np.array([0.0, 1.0, 2.0]), 0))
    assert np.isnan(m.quadratic_refine(K, np.array([2.0, 1.0, 0.0]), 2))
    # Concave (negative curvature) -> rejected.
    assert np.isnan(m.quadratic_refine(K, np.array([-1.0, 0.0, -1.0]), 1))
    # Positive curvature but vertex (at K=-0.5) falls outside the bracket.
    assert np.isnan(m.quadratic_refine(K, np.array([0.0, 1.0, 3.0]), 1))


# 10. end-to-end smoke run produces all required files ---------------------- #
def test_smoke_run_produces_all_files(tmp_path):
    K = [0.2, 0.5, 0.8]
    Pc = np.array([[0.6, 0.3, 0.1],
                   [0.2, 0.5, 0.3],
                   [0.1, 0.3, 0.6]])
    Prg = np.array([[0.6, 0.3, 0.1],
                    [0.2, 0.5, 0.3],
                    [0.1, 0.3, 0.6]])
    rg_edges = np.array([0.5, 1.5, 2.5, 3.5])
    d1 = tmp_path / "d1.npz"
    d2 = tmp_path / "d2.npz"
    make_direct_npz(d1, K, Pc, rg_edges, Prg, seed=101)
    make_direct_npz(d2, K, Pc, rg_edges + 0.01, Prg, seed=202)
    mol = tmp_path / "mol.npz"
    make_molecular_npz(
        mol, temps=[300.0, 320.0, 340.0],
        ct_centers=[0.0, 1.0, 2.0],
        ct_hists=[Pc[0], Pc[1], Pc[2]],
        rg_centers=[1.0, 2.0, 3.0],
        rg_hists=[Prg[0], Prg[1], Prg[2]],
    )
    fs = tmp_path / "fit.json"
    make_fit_summary(fs, contact_offset=0.0, rg_scale=1.0)
    out = tmp_path / "smoke_out"
    cmd = [
        sys.executable, str(HERE / "map_molecular_temperature_to_K.py"),
        "--molecular-npz", str(mol),
        "--direct-k-npz", str(d1), str(d2),
        "--fit-summary-json", str(fs),
        "--output-dir", str(out), "--overwrite",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(HERE))
    assert res.returncode == 0, res.stdout + "\n" + res.stderr
    for fname in (
        "empirical_K_mapping.csv", "empirical_K_score_surface.csv",
        "empirical_K_mapping.json", "empirical_K_mapping_report.md",
        "K_mapping_vs_temperature.png", "score_surface_combined.png",
        "component_K_mappings.png",
    ):
        assert (out / fname).exists(), f"missing {fname}"


# ========================================================================== #
# Review-fix tests: boundary gating, resolved-vs-raw statistics, score names
# ========================================================================== #
def _build_and_run(tmp_path, mol_indices, temps, K, Pc, Prg, rg_edges,
                   mol_rg_indices=None, contact_offset=0.0, rg_scale=1.0,
                   h=100.0, s=1.0, **argkw):
    """Build a single synthetic direct-K file + molecular file and run the
    mapping.  Molecular temperature j takes its contact distribution from direct
    grid index ``mol_indices[j]`` and its Rg distribution from
    ``mol_rg_indices[j]`` (defaulting to the same index), so each per-component
    optimum is known -- and can be made to differ between contact and Rg."""
    Pc = np.asarray(Pc, float)
    Prg = np.asarray(Prg, float)
    if mol_rg_indices is None:
        mol_rg_indices = mol_indices
    tmp_path = Path(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    d1 = tmp_path / "d.npz"
    make_direct_npz(d1, K, Pc, rg_edges, Prg, seed=101)
    mol = tmp_path / "mol.npz"
    ct_centers = list(range(Pc.shape[1]))              # integers 0..nC-1
    rg_centers = [float(i + 1) for i in range(Prg.shape[1])]  # 1..nRg (spacing 1)
    make_molecular_npz(
        mol, temps=temps, ct_centers=ct_centers,
        ct_hists=[Pc[i] for i in mol_indices],
        rg_centers=rg_centers, rg_hists=[Prg[i] for i in mol_rg_indices],
    )
    fs = tmp_path / "fit.json"
    make_fit_summary(fs, contact_offset=contact_offset, rg_scale=rg_scale, h=h, s=s)
    out = tmp_path / "out"
    args = _args(molecular_npz=str(mol), direct_k_npz=[str(d1)],
                 fit_summary_json=str(fs), output_dir=str(out), **argkw)
    result = m.run_mapping(args)
    md = (out / "empirical_K_mapping_report.md").read_text(encoding="utf-8")
    return result, md


_K3 = [0.2, 0.5, 0.8]
_DIST3 = np.array([[0.6, 0.3, 0.1],
                   [0.2, 0.5, 0.3],
                   [0.1, 0.3, 0.6]])
_RG_EDGES3 = np.array([0.5, 1.5, 2.5, 3.5])


# Review Test 1: every optimum pinned at the lower boundary ------------------ #
def test_all_optima_at_lower_boundary(tmp_path):
    result, md = _build_and_run(
        tmp_path, mol_indices=[0, 0, 0], temps=[280.0, 300.0, 320.0],
        K=_K3, Pc=_DIST3, Prg=_DIST3, rg_edges=_RG_EDGES3,
    )
    s = result["summary"]
    # All three component optima are at a boundary for every temperature.
    assert s["n_contact_boundary_optima"] == 3
    assert s["n_rg_boundary_optima"] == 3
    assert s["n_combined_boundary_optima"] == 3
    assert s["fraction_combined_boundary_optima"] == pytest.approx(1.0)
    for rec in result["report"]["mapping"]:
        assert rec["contact_boundary_hit_low"] is True
        assert rec["rg_boundary_hit_low"] is True
        assert rec["combined_boundary_hit_low"] is True
    # No resolved component pairs; old-mapping resolved errors are null.
    assert s["contact_rg_resolved"]["n_component_pairs_resolved"] == 0
    assert s["contact_rg_resolved"]["fraction_contact_rg_disagree_resolved"] is None
    assert s["vs_old_mapping_resolved"]["mean_abs_difference_resolved"] is None
    assert s["vs_old_mapping_resolved"]["rms_difference_resolved"] is None
    # No resolved transition matches.
    assert s["transition_resolved"]["n_temperatures_in_transition_resolved"] == 0
    # Recommendation is ONLY to expand the library.  (The Case-1 message names
    # "refitting K(T)"/"partial compactification" as things it PREVENTS, so we
    # check that neither actionable recommendation phrase is present.)
    assert "Expand the direct-K library" in md
    assert "refit the analytic K(T) mapping" not in md
    assert "map to partial compactification" not in md


# Review Test 2: combined boundary while component optima differ ------------- #
def test_combined_boundary_components_differ(tmp_path):
    # Contact nearly flat across K (favors index 0); Rg strongly separated
    # (favors index 2).  Rg dominates -> combined pinned at the HIGH boundary,
    # while the component optima sit at opposite ends.
    Pc = np.array([[0.40, 0.30, 0.30],
                   [0.30, 0.40, 0.30],
                   [0.30, 0.30, 0.40]])
    Prg = np.array([[0.80, 0.15, 0.05],
                    [0.10, 0.80, 0.10],
                    [0.05, 0.15, 0.80]])
    result, md = _build_and_run(
        tmp_path, mol_indices=[0], mol_rg_indices=[2], temps=[300.0], K=_K3,
        Pc=Pc, Prg=Prg, rg_edges=_RG_EDGES3,
    )
    rec = result["report"]["mapping"][0]
    s = result["summary"]
    assert rec["K_best_contact_grid"] == pytest.approx(0.2)   # i_c = 0 (low)
    assert rec["K_best_rg_grid"] == pytest.approx(0.8)        # i_r = 2 (high)
    assert rec["contact_boundary_hit_low"] is True
    assert rec["rg_boundary_hit_high"] is True
    assert rec["combined_boundary_hit_high"] is True
    # Raw disagreement IS reported...
    assert rec["contact_rg_disagreement_raw"] is True
    assert rec["contact_rg_K_difference"] == pytest.approx(0.6)
    assert s["contact_rg_raw"]["mean_abs_contact_rg_K_diff"] > 0
    # ...but no resolved agreement conclusion is drawn.
    assert rec["contact_rg_disagreement_resolved"] is None
    assert s["contact_rg_resolved"]["n_component_pairs_resolved"] == 0
    assert "no agreement claim" in md


# Review Test 3: fully interior optimum -------------------------------------- #
def test_interior_optimum_resolved(tmp_path):
    # Middle K is interior; transition band is chosen to contain it.
    result, md = _build_and_run(
        tmp_path, mol_indices=[1], temps=[300.0], K=_K3, Pc=_DIST3, Prg=_DIST3,
        rg_edges=_RG_EDGES3, transition_low=0.4, transition_high=0.6,
    )
    rec = result["report"]["mapping"][0]
    s = result["summary"]
    for flag in ("contact_boundary_hit_low", "contact_boundary_hit_high",
                 "rg_boundary_hit_low", "rg_boundary_hit_high",
                 "combined_boundary_hit_low", "combined_boundary_hit_high"):
        assert rec[flag] is False
    # Resolved contact/Rg disagreement is defined and calculated.
    assert rec["contact_rg_disagreement_resolved"] is False
    assert s["contact_rg_resolved"]["n_component_pairs_resolved"] == 1
    assert s["contact_rg_resolved"]["fraction_contact_rg_disagree_resolved"] == pytest.approx(0.0)
    # Old-mapping error IS included for this resolved row.
    assert rec["old_mapping_comparison_resolved"] is True
    assert rec["K_difference_from_old"] is not None
    assert s["vs_old_mapping_resolved"]["n_resolved_for_old_mapping_comparison"] == 1
    assert s["vs_old_mapping_resolved"]["mean_abs_difference_resolved"] is not None
    # Transition counting works (K=0.5 lies in [0.4, 0.6]).
    assert s["transition_resolved"]["n_temperatures_in_transition_resolved"] == 1


# Review Test 4: mixed resolved / unresolved temperatures -------------------- #
def test_mixed_resolved_and_unresolved(tmp_path):
    # T=300 -> interior (index 1); T=200 -> lower boundary (index 0).
    result, _md = _build_and_run(
        tmp_path, mol_indices=[1, 0], temps=[300.0, 200.0], K=_K3,
        Pc=_DIST3, Prg=_DIST3, rg_edges=_RG_EDGES3, h=100.0, s=1.0,
    )
    s = result["summary"]
    assert s["n_combined_boundary_optima"] == 1
    old = s["vs_old_mapping_resolved"]
    assert old["n_resolved_for_old_mapping_comparison"] == 1
    assert old["n_unresolved_for_old_mapping_comparison"] == 1
    # Resolved MAE uses ONLY the interior T=300 row: |0.5 - (1 - 100/300)|.
    k_old_300 = 1.0 - 100.0 / 300.0
    expected_res = abs(0.5 - k_old_300)
    assert old["mean_abs_difference_resolved"] == pytest.approx(expected_res)
    # Raw MAE (transparency) also includes the censored T=200 row and differs.
    k_old_200 = 1.0 - 100.0 / 200.0
    expected_raw = 0.5 * (abs(0.5 - k_old_300) + abs(0.2 - k_old_200))
    assert s["vs_old_mapping_raw"]["mean_abs_difference"] == pytest.approx(expected_raw)
    assert old["mean_abs_difference_resolved"] != pytest.approx(expected_raw)
    # Only the interior row forms a resolved component pair.
    assert s["contact_rg_resolved"]["n_component_pairs_resolved"] == 1


# Review Test 5: score-field names track the correct argmin ------------------ #
def test_score_field_naming(tmp_path):
    # Scenario A: Rg dominates -> combined follows Rg (index 2), contact opt is
    # index 0, so the contact fields must come from DIFFERENT indices.
    Pc_flat = np.array([[0.40, 0.30, 0.30],
                        [0.30, 0.40, 0.30],
                        [0.30, 0.30, 0.40]])
    Prg_sep = np.array([[0.80, 0.15, 0.05],
                        [0.10, 0.80, 0.10],
                        [0.05, 0.15, 0.80]])
    resA, _ = _build_and_run(
        tmp_path / "a", mol_indices=[0], mol_rg_indices=[2], temps=[300.0],
        K=_K3, Pc=Pc_flat, Prg=Prg_sep, rg_edges=_RG_EDGES3,
    )
    recA = resA["report"]["mapping"][0]
    assert recA["K_best_contact_grid"] == pytest.approx(0.2)   # i_c = 0
    assert recA["K_best_combined_grid"] == pytest.approx(0.8)  # i_comb = 2
    # JS_contact_min is the contact minimum (at i_c, an exact match -> 0);
    # JS_contact_at_combined is evaluated at i_comb and is strictly larger.
    assert recA["JS_contact_min"] == pytest.approx(0.0, abs=1e-12)
    assert recA["JS_contact_at_combined"] > 1e-6
    assert recA["JS_contact_min"] < recA["JS_contact_at_combined"]

    # Scenario B: contact dominates -> combined follows contact (index 0), Rg
    # opt is index 2, so the Rg fields must come from DIFFERENT indices.
    resB, _ = _build_and_run(
        tmp_path / "b", mol_indices=[0], mol_rg_indices=[2], temps=[300.0],
        K=_K3, Pc=Prg_sep, Prg=Pc_flat, rg_edges=_RG_EDGES3,
    )
    recB = resB["report"]["mapping"][0]
    assert recB["K_best_rg_grid"] == pytest.approx(0.8)        # i_r = 2
    assert recB["K_best_combined_grid"] == pytest.approx(0.2)  # i_comb = 0
    assert recB["JS_rg_min"] == pytest.approx(0.0, abs=1e-12)
    assert recB["JS_rg_at_combined"] > 1e-6
    assert recB["JS_rg_min"] < recB["JS_rg_at_combined"]
