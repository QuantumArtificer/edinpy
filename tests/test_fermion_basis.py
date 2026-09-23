import math

import pytest

from edinpy import fermion as edf


def test_dof_is_pure_descriptor():
    site = edf.DoF(5, name="site")
    assert site.size == 5
    assert site.name == "site"
    assert site.labels is None


def test_dof_accepts_ordered_labels():
    spin = edf.DoF(2, name="spin", labels=("up", "down"))

    assert spin.size == 2
    assert spin.labels == ("up", "down")


@pytest.mark.parametrize(
    "labels, error",
    [
        (("up",), ValueError),
        (("up", "up"), ValueError),
        (("up", 1), TypeError),
    ],
)
def test_dof_rejects_invalid_labels(labels, error):
    with pytest.raises(error):
        edf.DoF(2, name="spin", labels=labels)


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
