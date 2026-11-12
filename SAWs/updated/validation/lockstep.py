"""Lockstep comparison of two Monte Carlo kernels started from one state.

A *kernel* advances a state in place: ``kernel(state, rng, steps)`` where
``rng`` is a ``random.Random`` carrying the stream the kernel must consume.
Both kernels start from deep copies of the same state and identical RNG
states; after every chunk the comparable snapshot of each state AND the RNG
state are compared, so the first divergent chunk is reported, not just "the
end differs".  This is the test the Numba kernels (Stage 3/5) must pass
against the Python reference.

``single_chain_python_kernel`` adapts ``remd_uniform_chain_2_new.mc_sweep``
(which draws from the module-level ``random``) to this interface without
disturbing the caller's global RNG.  Python >= 3.8.
"""
from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

Kernel = Callable[[Any, random.Random, int], None]


@dataclass
class LockstepResult:
    ok: bool
    chunks_compared: int
    first_divergent_chunk: Optional[int] = None
    differing_fields: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        if self.ok:
            return f"identical over {self.chunks_compared} chunks (state + RNG)"
        return (f"DIVERGED at chunk {self.first_divergent_chunk}: "
                + ", ".join(self.differing_fields))


def run_lockstep(make_start: Callable[[], Tuple[Any, Any]], kernel_a: Kernel, kernel_b: Kernel,
                 snapshot: Callable[[Any], Dict[str, Any]], n_chunks: int,
                 steps_per_chunk: int) -> LockstepResult:
    """``make_start()`` returns (state, rng_state) with rng_state from ``random.Random.getstate``."""
    state0, rng_state0 = make_start()
    sa, sb = copy.deepcopy(state0), copy.deepcopy(state0)
    ra, rb = random.Random(), random.Random()
    ra.setstate(rng_state0)
    rb.setstate(rng_state0)
    for chunk in range(n_chunks):
        kernel_a(sa, ra, steps_per_chunk)
        kernel_b(sb, rb, steps_per_chunk)
        snap_a, snap_b = snapshot(sa), snapshot(sb)
        bad = sorted(k for k in set(snap_a) | set(snap_b) if not _same(snap_a.get(k), snap_b.get(k)))
        if ra.getstate() != rb.getstate():
            bad.append("rng_state")
        if bad:
            return LockstepResult(False, chunk + 1, chunk, bad)
    return LockstepResult(True, n_chunks)


def _same(a: Any, b: Any) -> bool:
    try:
        import numpy as np
        if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
            a, b = np.asarray(a), np.asarray(b)
            return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
    except ImportError:  # pragma: no cover
        pass
    if isinstance(a, float) and isinstance(b, float):
        return a == b or (a != a and b != b)
    return type(a) is type(b) and a == b


# ---------------------------------------------------------------------------
# Single-chain adapters (remd_uniform_chain_2_new)
# ---------------------------------------------------------------------------

def single_chain_python_kernel(R, model_name: str, params, Tref: float, Tscale: float,
                               kappa_bend: float = 0.0, n_beads: Optional[int] = None) -> Kernel:
    """Wrap ``R.mc_sweep`` (global-``random`` consumer) as a lockstep kernel."""
    def kernel(replica, rng: random.Random, steps: int) -> None:
        saved = random.getstate()
        try:
            random.setstate(rng.getstate())
            R.mc_sweep(replica, steps, model_name, params, Tref, Tscale, kappa_bend, n_beads)
            rng.setstate(random.getstate())
        finally:
            random.setstate(saved)
    return kernel


def single_chain_snapshot(replica) -> Dict[str, Any]:
    st = replica.state
    return {
        "chain": tuple(tuple(p) for p in st.chain),
        "occ": frozenset(st.occ),
        "m": st.m, "E": st.E, "n_bend": st.n_bend,
        "coordination_hist": None if st.coordination_hist is None else tuple(st.coordination_hist),
        "local_acc": replica.local_acc, "local_prop": replica.local_prop,
        "move_counters": replica.move_counters.copy(),
    }


def single_chain_start(R, N: int, T: float, model_name: str, params, Tref: float, Tscale: float,
                       seed: int, burn_steps: int = 5000, kappa_bend: float = 0.0):
    """A replica moved off the straight initial chain, plus a fresh RNG state."""
    def make():
        saved = random.getstate()
        try:
            random.seed(seed)
            st = R.ChainState.initial_straight(N, T, model_name, params, Tref, Tscale)
            rep = R.Replica(T=T, state=st)
            if burn_steps:
                R.mc_sweep(rep, burn_steps, model_name, params, Tref, Tscale, kappa_bend, N)
            return rep, random.Random(seed + 1).getstate()
        finally:
            random.setstate(saved)
    return make
