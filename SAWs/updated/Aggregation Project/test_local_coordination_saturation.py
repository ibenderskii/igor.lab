from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import FIT_TO_DAT as dat
import fit_lattice_contact_model_2 as fit
import multichain_state as mcs
import remd_multichain as multi
import remd_uniform_chain_2_new as remd


MODEL = "local_coordination_saturation"
PARAMS = np.array([700.0, 2.3, 1.2, 0.4])
TREF = 320.0
TSCALE = 80.0


def _hist_for(n_beads: int, contacts: int) -> np.ndarray:
    hist = np.zeros(7, dtype=np.int64)
    full, remainder = divmod(2 * int(contacts), 6)
    hist[6] = full
    used = full
    if remainder:
        hist[remainder] += 1
        used += 1
    hist[0] = int(n_beads) - used
    return hist


def test_v4_contract_is_shared_and_marks_complete_state() -> None:
    contracts = [
        fit.get_model_contract(), remd.get_model_contract(), dat.get_model_contract()
    ]
    assert all(contract["model_api_version"] == 4 for contract in contracts)
    entries = [contract["models"][MODEL] for contract in contracts]
    assert all(entry == entries[0] for entry in entries[1:])
    entry = contracts[0]["models"][MODEL]
    assert entry["param_names"] == ["h_b", "s_b", "A0", "q_sat"]
    assert entry["configuration_dependent"] is True
    assert entry["state_observable"] == "nonbonded_contact_degree_histogram_0_6"
    assert entry["potential_kind"] == "local_coordination_saturation"


def test_aggregation_spec_records_the_new_model_without_rewriting_legacy() -> None:
    path = Path(__file__).with_name("aggregation_model_spec_v1.json")
    spec = json.loads(path.read_text(encoding="utf-8"))
    assert spec["model_api_version"] == 4
    assert spec["spec_revision"] == 5
    local = spec["reduced_potential"]["local_coordination_contact_term"]
    assert local["models"] == [MODEL]
    assert "q_i = k_i/2" in local["equation"]
    assert "intrachain and interchain" in local["degree_definition"]
    history = spec["spec_revision_history"]
    assert history[-2]["spec_revision"] == 4
    assert "historical" in history[-1]["change"].lower()


@pytest.mark.parametrize("module", [fit, remd, dat])
def test_scalar_contact_accessor_refuses_local_model(module) -> None:
    with pytest.raises(ValueError, match="requires.*degree|requires.*histogram"):
        module.make_contact_u_fn(MODEL, TREF, TSCALE, n_beads=20)


def test_single_chain_state_potential_matches_definition() -> None:
    chain = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
    occupied = set(chain)
    m, hist = remd.contact_count_and_coordination_histogram(chain, occupied)
    assert m == 1
    assert hist == (2, 2, 0, 0, 0, 0, 0)
    T = 310.0
    q = 0.5 * np.arange(7, dtype=float)
    g_sum = float(np.dot(hist, q * q / (1.0 + (q / PARAMS[3]) ** 2)))
    expected = (PARAMS[0] / T - PARAMS[1]) * m - PARAMS[2] * g_sum
    got = remd.reduced_contact_potential_state(
        m, hist, T, MODEL, PARAMS, TREF, TSCALE, len(chain)
    )
    assert got == pytest.approx(expected, abs=1e-15)


def test_A0_zero_nests_hs_exactly_for_every_coordination_state() -> None:
    state_u = remd.make_contact_state_u_fn(MODEL, TREF, TSCALE, 20)
    hs_u = remd.make_contact_u_fn("hs", TREF, TSCALE, 20)
    local_params = np.array([812.5, 2.6125, 0.0, 0.35])
    hs_params = local_params[:2]
    for m in range(9):
        hist = _hist_for(20, m)
        for T in (285.0, 320.0, 355.0):
            assert state_u(local_params, T, m, hist) == hs_u(hs_params, T, m)


def test_local_baseline_validation_and_reweighting(tmp_path) -> None:
    n_beads = 20
    contacts = np.arange(5, dtype=np.int64)
    hist = np.array([_hist_for(n_beads, m) for m in contacts])
    state_mass = np.array([0.1, 0.2, 0.3, 0.25, 0.15])
    rg_edges = np.array([1.0, 2.0, 3.0])
    state_idx = np.repeat(np.arange(5), 2)
    rg_idx = np.tile(np.arange(2), 5)
    joint = np.repeat(state_mass, 2) * np.tile([0.4, 0.6], 5)
    path = tmp_path / "baseline.npz"
    np.savez_compressed(
        path,
        N=n_beads,
        c_vals=contacts,
        c_prob=state_mass,
        rg_edges=rg_edges,
        local_coord_schema_version=1,
        local_coord_degree_values=np.arange(7),
        local_coord_histograms=hist,
        local_coord_contact_counts=contacts,
        local_coord_state_mass=state_mass,
        local_coord_rg_state_index=state_idx,
        local_coord_rg_bin_index=rg_idx,
        local_coord_rg_joint_mass=joint,
    )
    with np.load(path) as saved:
        baseline = fit.load_local_coordination_baseline(
            saved, n_beads, contacts.astype(float), rg_edges
        )
    state_u = fit.make_contact_state_u_fn(MODEL, TREF, TSCALE, n_beads)
    contact_mass = fit.local_coordination_contact_mass(
        baseline, 320.0, PARAMS, state_u, len(contacts)
    )
    rg_mass = fit.local_coordination_rg_mass(
        baseline, 320.0, PARAMS, state_u
    )
    assert contact_mass.sum() == pytest.approx(1.0)
    assert rg_mass.sum() == pytest.approx(1.0)


def test_multichain_defaults_to_the_fitted_local_hamiltonian() -> None:
    state = mcs.initialize_dispersed_state(3, 8, 12, seed=4)
    new = multi.reduced_contact_potential_state(
        state, 330.0, MODEL, PARAMS, TREF, TSCALE, 1.0, 1.0
    )
    legacy_local = multi.reduced_contact_potential_state(
        state, 330.0, "saturating_cooperative_contact", PARAMS,
        TREF, TSCALE, 1.0, 1.0, cooperativity="local",
    )
    assert new == legacy_local
    assert multi._validate_cooperativity(None, MODEL) == "local"
    assert multi._validate_cooperativity(None, "saturating_cooperative_contact") == "global"

    metadata = {}
    multi.attach_metadata(
        metadata, M=3, N=8, L=12, Ts=np.array([300.0, 340.0]), seed=4,
        model_name=MODEL, param_names=["h_b", "s_b", "A0", "q_sat"],
        model_params=PARAMS, Tref=TREF, Tscale=TSCALE, lambda_intra=1.0,
        lambda_inter=1.0, local_sweeps_per_swap=1,
        translation_sweeps_per_swap=1, n_cycles=1, burnin_frac=0.5,
        cluster_contact_threshold=1, parameter_source="test",
        fit_summary_json="",
    )
    assert metadata["cooperativity"] == "local"


def test_multichain_refuses_global_reinterpretation_and_unequal_lambdas() -> None:
    state = mcs.initialize_dispersed_state(2, 8, 12, seed=4)
    with pytest.raises(ValueError, match="requires cooperativity='local'"):
        multi.reduced_contact_potential_state(
            state, 330.0, MODEL, PARAMS, TREF, TSCALE, 1.0, 1.0,
            cooperativity="global",
        )
    with pytest.raises(ValueError, match="lambda_intra == lambda_inter"):
        multi.reduced_contact_potential_state(
            state, 330.0, MODEL, PARAMS, TREF, TSCALE, 1.0, 0.0
        )


def test_fit_summary_handoff_preserves_model_and_chain_length(tmp_path) -> None:
    summary = {
        "model_api_version": 4,
        "model": MODEL,
        "params": {name: float(value) for name, value in zip(
            ["h_b", "s_b", "A0", "q_sat"], PARAMS
        )},
        "Tref": TREF,
        "Tscale": TSCALE,
        "kappa_bend": 0.0,
        "fit_chain_length": 30,
    }
    path = tmp_path / "fit_summary.json"
    path.write_text(json.dumps(summary), encoding="utf-8")
    loaded = remd.load_fit_summary_json(str(path))
    assert loaded["model_name"] == MODEL
    assert loaded["fit_chain_length"] == 30
    assert loaded["params"] == pytest.approx(PARAMS)
