"""Shared test helper: load the single-chain fitter by explicit file path.

The fitter lives in ``../auto/fit_lattice_contact_model_parallel.py``.  ``auto/``
also holds same-named copies of modules in this folder (e.g.
``remd_uniform_chain_2_new.py``), so it is never put on ``sys.path``; the fitter
is loaded from its file instead and the Aggregation Project copies are checked
to still be the ones imported.
"""
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FITTER = os.path.normpath(os.path.join(
    HERE, os.pardir, "auto", "fit_lattice_contact_model_parallel.py"))
MODULE_NAME = "fit_lattice_contact_model_parallel"


def load_fitter():
    """Return the fitter module (loaded once per process)."""
    module = sys.modules.get(MODULE_NAME)
    if module is None:
        spec = importlib.util.spec_from_file_location(MODULE_NAME, FITTER)
        module = importlib.util.module_from_spec(spec)
        sys.modules[MODULE_NAME] = module
        spec.loader.exec_module(module)
    import remd_uniform_chain_2_new as remd
    if os.path.dirname(os.path.abspath(remd.__file__)) != HERE:
        raise ImportError(
            f"remd_uniform_chain_2_new resolved to {remd.__file__}, not the "
            f"Aggregation Project copy in {HERE}")
    return module
