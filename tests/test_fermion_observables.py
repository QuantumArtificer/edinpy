import numpy as np
import pytest

from edinpy import fermion as edf


def test_sector_vector_round_trip_and_read_only_coefficients():
    modes = edf.FermionModes(edf.DoF(4))
    sector = edf.NParticleSector(modes, 2).build()
    coefficients = np.arange(sector.dimension, dtype=float)

    psi = sector.from_vector(coefficients)

    assert isinstance(psi, edf.FockVector)
    assert psi.sector is sector
    assert np.array_equal(psi.to_vector(), coefficients)
    assert not psi.coefficients.flags.writeable
    with pytest.raises(ValueError):
        psi.coefficients[0] = 3.0


def test_basis_backed_vector_inner_product_matches_numpy_vdot():
    modes = edf.FermionModes(edf.DoF(5))
    sector = edf.NParticleSector(modes, 2).build()
    rng = np.random.default_rng(91)
    a = rng.normal(size=sector.dimension) + 1j * rng.normal(size=sector.dimension)
    b = rng.normal(size=sector.dimension) + 1j * rng.normal(size=sector.dimension)
    psi = sector.from_vector(a)
    phi = sector.from_vector(b)

    expected = np.vdot(a, b)
    assert np.allclose(psi.inner(phi), expected)
    assert np.allclose(psi.dag * phi, expected)


def test_fock_state_bra_ket_algebra():
    modes = edf.FermionModes(edf.DoF(3))
    n = edf.set_notation(edf.Number, modes)
    ket = edf.FockState(0b101, n_modes=3)

    assert ket.dag * ket == 1
    assert ket.dag * n(0) * ket == 1
    assert ket.dag * n(1) * ket == 0
    assert ket.inner(n(2) * ket) == 1


def test_eigenstate_conversion_preserves_solver_basis_coordinates():
    modes = edf.FermionModes(edf.DoF(6))
    sector = edf.NParticleSector(modes, 3).build()
    H = sum((edf.Hopping(i, i + 1, -1.0, modes) for i in range(5)), start=0)
    ham = edf.Hamiltonian(H, sector)
    _, eigenvectors = ham.eigsolve(k=3, which="SA", tol=1e-12)

    psi = ham.eigenstate(0)
    states = ham.eigenstates()

    assert isinstance(psi, edf.FockVector)
    assert psi.sector is sector
    assert np.allclose(psi.coefficients, eigenvectors[:, 0])
    assert len(states) == 3
    for index, state in enumerate(states):
        assert np.allclose(state.coefficients, eigenvectors[:, index])
        assert np.allclose(state.dag * state, 1.0, atol=1e-12)


def test_eigenstate_requires_a_completed_eigensolve():
    modes = edf.FermionModes(edf.DoF(3))
    sector = edf.NParticleSector(modes, 1).build()
    ham = edf.Hamiltonian(edf.Onsite(0, 1.0, modes), sector)

    with pytest.raises(RuntimeError):
        ham.eigenstate()
    with pytest.raises(RuntimeError):
        ham.eigenstates()


def test_literal_expectation_matches_matrix_expression():
    modes = edf.FermionModes(edf.DoF(6))
    sector = edf.NParticleSector(modes, 3).build()
    H = sum((edf.Hopping(i, i + 1, -1.0, modes) for i in range(5)), start=0)
    ham = edf.Hamiltonian(H, sector)
    _, eigenvectors = ham.eigsolve(k=2, which="SA", tol=1e-12)
    psi = ham.eigenstate(0)

    O = edf.Onsite(2, 1.0, modes) + 0.37 * edf.Hopping(0, 4, 1.0, modes)
    O_matrix = edf.Hamiltonian(O, sector).matrix
    expected = np.vdot(eigenvectors[:, 0], O_matrix @ eigenvectors[:, 0])

    assert np.allclose(psi.dag * O * psi, expected, atol=1e-12)
    assert np.allclose(psi.inner(O * psi), expected, atol=1e-12)


def test_transition_matrix_element_matches_matrix_expression():
    modes = edf.FermionModes(edf.DoF(5))
    sector = edf.NParticleSector(modes, 2).build()
    H = sum((edf.Hopping(i, i + 1, -1.0, modes) for i in range(4)), start=0)
    ham = edf.Hamiltonian(H, sector)
    _, eigenvectors = ham.eigsolve(sparse=False, k=3, which="SA")
    psi = ham.eigenstate(0)
    phi = ham.eigenstate(1)

    O = edf.Creation(0, modes=modes) * edf.Annihilation(3, modes=modes)
    O_matrix = edf.Hamiltonian(O, sector).matrix
    expected = np.vdot(eigenvectors[:, 1], O_matrix @ eigenvectors[:, 0])

    assert np.allclose(phi.dag * O * psi, expected, atol=1e-12)


def test_number_changing_expectation_vanishes_without_projection_artifact():
    modes = edf.FermionModes(edf.DoF(4))
    sector = edf.NParticleSector(modes, 2).build()
    H = sum((edf.Onsite(i, i + 1.0, modes) for i in range(4)), start=0)
    ham = edf.Hamiltonian(H, sector)
    ham.eigsolve(sparse=False, k=1, which="SA")
    psi = ham.eigenstate(0)
    c = edf.set_notation(edf.Annihilation, modes)

    assert psi.dag * c(0) * psi == 0
    assert psi.inner(c(0) * psi) == 0


def test_fock_vectors_from_different_mode_definitions_do_not_mix():
    modes_a = edf.FermionModes(edf.DoF(4))
    modes_b = edf.FermionModes(edf.DoF(4))
    psi = edf.NParticleSector(modes_a, 2).build().from_vector(np.ones(6))
    phi = edf.NParticleSector(modes_b, 2).build().from_vector(np.ones(6))

    with pytest.raises(ValueError):
        psi.inner(phi)
    with pytest.raises(ValueError):
        _ = psi + phi


def test_different_particle_number_sectors_are_orthogonal():
    modes = edf.FermionModes(edf.DoF(4))
    psi = edf.NParticleSector(modes, 1).build().from_vector(np.ones(4))
    phi = edf.NParticleSector(modes, 2).build().from_vector(np.ones(6))

    assert psi.dag * phi == 0


def test_operator_action_rejects_fock_vector_with_different_modes():
    modes_a = edf.FermionModes(edf.DoF(4))
    modes_b = edf.FermionModes(edf.DoF(4))
    psi = edf.NParticleSector(modes_a, 2).build().from_vector(np.ones(6))
    O = edf.Onsite(0, 1.0, modes_b)

    with pytest.raises(ValueError):
        _ = O * psi
    with pytest.raises(ValueError):
        _ = psi.dag * O * psi
