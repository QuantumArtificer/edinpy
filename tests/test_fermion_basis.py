import math

import pytest

from edinpy import fermion as edf


def test_dof_is_pure_descriptor():
    site = edf.DoF(5, name="site")
    assert site.size == 5
    assert site.name == "site"


@pytest.mark.parametrize("size", [0, -1])
def test_dof_rejects_nonpositive_size(size):
    with pytest.raises(ValueError):
        edf.DoF(size)


def test_fermion_modes_mapping_and_strides():
    site = edf.DoF(3, "site")
    spin = edf.DoF(2, "spin")
    orbital = edf.DoF(4, "orbital")
    modes = edf.FermionModes(site, spin, orbital)
    assert modes.strides == (1, 3, 6)
    assert modes.n_modes == 24
    assert modes.resolve((2, 1, 3)) == 23


def test_fermion_modes_round_trip():
    modes = edf.FermionModes(edf.DoF(4), edf.DoF(3), edf.DoF(2))
    for mode in range(modes.n_modes):
        assert modes.resolve(modes.unravel(mode)) == mode


def test_fermion_modes_rejects_wrong_number_of_indices():
    modes = edf.FermionModes(edf.DoF(4), edf.DoF(2))
    with pytest.raises(ValueError):
        modes.resolve((1,))


def test_single_dof_accepts_bare_integer_index():
    modes = edf.FermionModes(edf.DoF(4))
    assert modes.resolve(3) == 3


@pytest.mark.parametrize("n_modes", range(1, 9))
def test_fock_basis_all_particle_numbers(n_modes):
    for N in range(n_modes + 1):
        basis = edf.FockBasis(n_modes, N)
        assert len(basis) == math.comb(n_modes, N)
        assert list(basis.states) == sorted(basis.states)
        assert all(state.bit_count() == N for state in basis.states)
        assert all(state < (1 << n_modes) for state in basis.states)


def test_fock_basis_contains_states():
    basis = edf.FockBasis(5, 2)
    assert 0b00101 in basis
    assert edf.FockState(0b00101, n_modes=5) in basis
    assert 0b00111 not in basis


def test_fock_state_keeps_local_mode_count():
    state = edf.FockState(1, n_modes=7)
    assert "0000001" in str(state)


def test_n_particle_sector_owns_its_basis():
    modes = edf.FermionModes(edf.DoF(6))
    sector = edf.NParticleSector(modes, N=3)
    assert sector.dimension == 20
    assert sector.basis.n_modes == 6
    assert sector.basis.N == 3


def test_independent_sectors_do_not_share_state():
    modes_a = edf.FermionModes(edf.DoF(5))
    modes_b = edf.FermionModes(edf.DoF(7))
    sector_a = edf.NParticleSector(modes_a, 2)
    sector_b = edf.NParticleSector(modes_b, 3)
    assert sector_a.dimension == 10
    assert sector_b.dimension == 35
    assert sector_a.basis.n_modes == 5
    assert sector_b.basis.n_modes == 7


def test_large_vacuum_sector_preserves_mode_count():
    modes = edf.FermionModes(edf.DoF(70))
    sector = edf.NParticleSector(modes, 0)
    assert sector.basis.states == (0,)
    assert sector.basis.n_modes == 70

