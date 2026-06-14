"""
Unit tests for contact_count, radius_of_gyration, and energy.

All chains are hand-built on the 3-D cubic lattice so expected values
can be verified by inspection.

Run with:  python -m pytest SAWs/updated/test_thermo_chain.py -v
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from thermo_uniform_chain2_DBfix_dists_corrected import (
    contact_count,
    radius_of_gyration,
    energy,
)

# ---------------------------------------------------------------------------
# Test chains
# ---------------------------------------------------------------------------

# Four monomers along x — no non-bonded contacts.
STRAIGHT = [(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)]

# Four monomers forming a square — one non-bonded contact: monomers 0 and 3.
#   0 - 1
#   |   |
#   3 - 2  (3 is bonded to 2, not to 0)
SQUARE = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]

# Six monomers in a U shape — two non-bonded contacts: {0,5} and {1,4}.
#   0 - 1 - 2
#   |   |   |   (only the two vertical pairs are non-bonded NN)
#   5 - 4 - 3
U_SHAPE = [(0, 0, 0), (1, 0, 0), (2, 0, 0), (2, 1, 0), (1, 1, 0), (0, 1, 0)]


def _occ(chain):
    return set(chain)


# ---------------------------------------------------------------------------
# contact_count
# ---------------------------------------------------------------------------

def test_contact_count_straight_zero():
    assert contact_count(STRAIGHT, _occ(STRAIGHT)) == 0.0


def test_contact_count_square_one():
    assert contact_count(SQUARE, _occ(SQUARE)) == 1.0


def test_contact_count_u_shape_two():
    assert contact_count(U_SHAPE, _occ(U_SHAPE)) == 2.0


# ---------------------------------------------------------------------------
# radius_of_gyration
# ---------------------------------------------------------------------------

def test_rg_two_monomers():
    # com = (0.5, 0, 0); each monomer 0.5 from com → Rg = 0.5
    chain = [(0, 0, 0), (1, 0, 0)]
    assert abs(radius_of_gyration(chain) - 0.5) < 1e-12


def test_rg_three_inline():
    # com = (1, 0, 0); squared distances: 1, 0, 1 → mean = 2/3 → Rg = sqrt(2/3)
    chain = [(0, 0, 0), (1, 0, 0), (2, 0, 0)]
    assert abs(radius_of_gyration(chain) - math.sqrt(2 / 3)) < 1e-12


def test_rg_single_monomer():
    chain = [(0, 0, 0)]
    assert radius_of_gyration(chain) == 0.0


# ---------------------------------------------------------------------------
# energy  H = n_contacts * (dh - T * ds)
# ---------------------------------------------------------------------------

def test_energy_no_contacts():
    # straight chain → 0 contacts → E = 0 regardless of params
    assert energy(STRAIGHT, _occ(STRAIGHT), dh=1.0, ds=1.0, T=5.0) == 0.0


def test_energy_one_contact_ds_zero():
    # square: 1 contact, ds=0 → E = 1 * dh, T-independent
    assert abs(energy(SQUARE, _occ(SQUARE), dh=3.7, ds=0.0, T=99.0) - 3.7) < 1e-12


def test_energy_temperature_scaling():
    # U shape: 2 contacts, dh=2, ds=1, T=1.5 → E = 2*(2 - 1.5*1) = 1.0
    result = energy(U_SHAPE, _occ(U_SHAPE), dh=2.0, ds=1.0, T=1.5)
    assert abs(result - 1.0) < 1e-12
