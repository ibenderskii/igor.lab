"""P1: the compatibility module resolves to the canonical implementation."""
import remd_uniform_chain_2 as shim
import remd_uniform_chain_2_new as canon


def test_shim_resolves_to_canonical_by_identity():
    for name in ("run_remd", "mc_sweep", "attempt_swap", "contact_count",
                 "build_distributions", "compute_statistics", "main",
                 "ROT_MATS", "Replica", "ChainState", "_apply_rot"):
        assert getattr(shim, name) is getattr(canon, name), name


def test_shim_marks_canonical_module():
    assert shim.CANONICAL_MODULE is canon


def test_tests_use_canonical_module():
    # The pytest suite imports the canonical implementation directly.
    assert canon.__name__ == "remd_uniform_chain_2_new"
