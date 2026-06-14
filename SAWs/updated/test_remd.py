"""
Tests for the swap acceptance criterion in remd_uniform_chain.py.

All expected values are derived analytically from the formula:
    log_accept = u(C_i,T_i) + u(C_j,T_j) - u(C_j,T_i) - u(C_i,T_j)
    u(C, T) = m(C) * (dh/T - ds)

Run with:  python -m pytest SAWs/updated/test_remd.py -v
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from remd_uniform_chain import reduced_potential, swap_log_accept


# ---------------------------------------------------------------------------
# reduced_potential
# ---------------------------------------------------------------------------

def test_reduced_potential_ds_zero():
    # u = m * dh/T   when ds=0
    assert abs(reduced_potential(4.0, 2.0, 1.0, 0.0) - 2.0) < 1e-12


def test_reduced_potential_full():
    # u = m * (dh/T - ds) = 3 * (2/1 - 0.5) = 3 * 1.5 = 4.5
    assert abs(reduced_potential(3.0, 1.0, 2.0, 0.5) - 4.5) < 1e-12


def test_reduced_potential_zero_contacts():
    # m=0 → u=0 regardless of T, dh, ds
    assert reduced_potential(0.0, 1.5, 10.0, 3.0) == 0.0


# ---------------------------------------------------------------------------
# swap_log_accept: analytical cases
# ---------------------------------------------------------------------------

def test_swap_log_accept_equal_contacts():
    # m_i = m_j → all four u terms cancel → log_accept = 0
    result = swap_log_accept(5.0, 5.0, 1.0, 2.0, dh=1.0, ds=0.5)
    assert abs(result) < 1e-12


def test_swap_log_accept_positive():
    # m_i=2, m_j=0, T_i=1, T_j=2, dh=1, ds=0
    # u(2,1)=2, u(0,2)=0, u(0,1)=0, u(2,2)=1
    # log_accept = 2 + 0 - 0 - 1 = 1
    result = swap_log_accept(2.0, 0.0, 1.0, 2.0, dh=1.0, ds=0.0)
    assert abs(result - 1.0) < 1e-12


def test_swap_log_accept_negative():
    # m_i=0, m_j=2, T_i=1, T_j=2, dh=1, ds=0
    # u(0,1)=0, u(2,2)=1, u(2,1)=2, u(0,2)=0
    # log_accept = 0 + 1 - 2 - 0 = -1
    result = swap_log_accept(0.0, 2.0, 1.0, 2.0, dh=1.0, ds=0.0)
    assert abs(result - (-1.0)) < 1e-12


def test_swap_log_accept_with_ds():
    # m_i=3, m_j=1, T_i=1, T_j=2, dh=2, ds=1
    # u(3,1) = 3*(2-1)=3   u(1,2) = 1*(1-1)=0
    # u(1,1) = 1*(2-1)=1   u(3,2) = 3*(1-1)=0
    # log_accept = 3 + 0 - 1 - 0 = 2
    result = swap_log_accept(3.0, 1.0, 1.0, 2.0, dh=2.0, ds=1.0)
    assert abs(result - 2.0) < 1e-12


# ---------------------------------------------------------------------------
# swap_log_accept: structural properties
# ---------------------------------------------------------------------------

def test_swap_log_accept_antisymmetric():
    # Swapping (i,j) and (j,i) must give opposite signs:
    # log_accept(m_i, m_j, T_i, T_j) == -log_accept(m_j, m_i, T_i, T_j)
    dh, ds = 2.0, 0.5
    a = swap_log_accept(3.0, 1.0, 1.0, 2.0, dh, ds)
    b = swap_log_accept(1.0, 3.0, 1.0, 2.0, dh, ds)
    assert abs(a + b) < 1e-12


def test_swap_log_accept_same_temperature():
    # At T_i = T_j the swap has no thermodynamic driving force:
    # u(C_i,T) + u(C_j,T) - u(C_j,T) - u(C_i,T) = 0
    result = swap_log_accept(4.0, 2.0, 300.0, 300.0, dh=378.96, ds=1.39686)
    assert abs(result) < 1e-9
