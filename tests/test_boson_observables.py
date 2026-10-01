import numpy as np
import pytest

from edinpy import boson as edb


def test_sector_vector_round_trip_and_read_only_coefficients():
    modes = edb.BosonModes(edb.DoF(4))
    sector = edb.NParticleSector(modes, 2).build()
    coefficients = np.arange(sector.dimension, dtype=float)

    psi = sector.from_vector(coefficients)

    assert isinstance(psi, edb.FockVector)
    assert psi.sector is sector
    assert np.array_equal(psi.to_vector(), coefficients)
    assert not psi.coefficients.flags.writeable
    with pytest.raises(ValueError):
        psi.coefficients[0] = 3.0


def test_basis_backed_vector_inner_product_matches_numpy_vdot():
    modes = edb.BosonModes(edb.DoF(5))
    sector = edb.NParticleSector(modes, 2).build()
    rng = np.random.default_rng(91)
    a = rng.normal(size=sector.dimension) + 1j * rng.normal(size=sector.dimension)
    b = rng.normal(size=sector.dimension) + 1j * rng.normal(size=sector.dimension)
    psi = sector.from_vector(a)
    phi = sector.from_vector(b)

    expected = np.vdot(a, b)
    assert np.allclose(psi.inner(phi), expected)
    assert np.allclose(psi.dag * phi, expected)


def test_fock_state_bra_ket_algebra():
    modes = edb.BosonModes(edb.DoF(3))
    n = edb.set_notation(edb.Number, modes)
    ket = edb.FockState((2, 0, 1))

    assert ket.dag * ket == 1
    assert ket.dag * n(0) * ket == 2
    assert ket.dag * n(1) * ket == 0
    assert ket.inner(n(2) * ket) == 1


def test_eigenstate_conversion_preserves_solver_basis_coordinates():
    modes = edb.BosonModes(edb.DoF(5))
    sector = edb.NParticleSector(modes, 2).build()
    H = sum((edb.Hopping(i, i + 1, -1.0, modes) for i in range(4)), start=0)
    ham = edb.Hamiltonian(H, sector)
    _, eigenvectors = ham.eigsolve(k=3, which="SA", tol=1e-12)

    psi = ham.eigenstate(0)
    states = ham.eigenstates()

    assert isinstance(psi, edb.FockVector)
    assert psi.sector is sector
    assert np.allclose(psi.coefficients, eigenvectors[:, 0])
    assert len(states) == 3
    for index, state in enumerate(states):
        assert np.allclose(state.coefficients, eigenvectors[:, index])
        assert np.allclose(state.dag * state, 1.0, atol=1e-12)


def test_eigenstate_requires_a_completed_eigensolve():
    modes = edb.BosonModes(edb.DoF(3))
    sector = edb.NParticleSector(modes, 1).build()
    ham = edb.Hamiltonian(edb.Onsite(0, 1.0, modes), sector)

    with pytest.raises(RuntimeError):
        ham.eigenstate()
    with pytest.raises(RuntimeError):
        ham.eigenstates()


def test_literal_expectation_matches_matrix_expression():
    modes = edb.BosonModes(edb.DoF(5))
    sector = edb.NParticleSector(modes, 2).build()
    H = sum((edb.Hopping(i, i + 1, -1.0, modes) for i in range(4)), start=0)
    ham = edb.Hamiltonian(H, sector)
    _, eigenvectors = ham.eigsolve(k=2, which="SA", tol=1e-12)
    psi = ham.eigenstate(0)

    O = edb.Onsite(2, 1.0, modes) + 0.37 * edb.Hopping(0, 4, 1.0, modes)
    O_matrix = edb.Hamiltonian(O, sector).matrix
    expected = np.vdot(eigenvectors[:, 0], O_matrix @ eigenvectors[:, 0])

    assert np.allclose(psi.dag * O * psi, expected, atol=1e-12)
    assert np.allclose(psi.inner(O * psi), expected, atol=1e-12)


def test_transition_matrix_element_matches_matrix_expression():
    modes = edb.BosonModes(edb.DoF(4))
    sector = edb.NParticleSector(modes, 2).build()
    H = sum((edb.Hopping(i, i + 1, -1.0, modes) for i in range(3)), start=0)
    ham = edb.Hamiltonian(H, sector)
    _, eigenvectors = ham.eigsolve(sparse=False, k=3, which="SA")
    psi = ham.eigenstate(0)
    phi = ham.eigenstate(1)

    O = edb.Creation(0, modes=modes) * edb.Annihilation(3, modes=modes)
    O_matrix = edb.Hamiltonian(O, sector).matrix
    expected = np.vdot(eigenvectors[:, 1], O_matrix @ eigenvectors[:, 0])

    assert np.allclose(phi.dag * O * psi, expected, atol=1e-12)


def test_number_changing_expectation_vanishes_without_projection_artifact():
    modes = edb.BosonModes(edb.DoF(4))
    sector = edb.NParticleSector(modes, 2).build()
    H = sum((edb.Onsite(i, i + 1.0, modes) for i in range(4)), start=0)
    ham = edb.Hamiltonian(H, sector)
    ham.eigsolve(sparse=False, k=1, which="SA")
    psi = ham.eigenstate(0)
    b = edb.set_notation(edb.Annihilation, modes)

    assert psi.dag * b(0) * psi == 0
    assert psi.inner(b(0) * psi) == 0


def test_fock_vectors_from_different_mode_definitions_do_not_mix():
    modes_a = edb.BosonModes(edb.DoF(4))
    modes_b = edb.BosonModes(edb.DoF(4))
    psi = edb.NParticleSector(modes_a, 2).build().from_vector(np.ones(10))
    phi = edb.NParticleSector(modes_b, 2).build().from_vector(np.ones(10))

    with pytest.raises(ValueError):
        psi.inner(phi)
    with pytest.raises(ValueError):
        _ = psi + phi


def test_different_particle_number_sectors_are_orthogonal():
    modes = edb.BosonModes(edb.DoF(4))
    psi = edb.NParticleSector(modes, 1).build().from_vector(np.ones(4))
    phi = edb.NParticleSector(modes, 2).build().from_vector(np.ones(10))

    assert psi.dag * phi == 0


def test_operator_action_rejects_fock_vector_with_different_modes():
    modes_a = edb.BosonModes(edb.DoF(4))
    modes_b = edb.BosonModes(edb.DoF(4))
    psi = edb.NParticleSector(modes_a, 2).build().from_vector(np.ones(10))
    O = edb.Onsite(0, 1.0, modes_b)

    with pytest.raises(ValueError):
        _ = O * psi
    with pytest.raises(ValueError):
        _ = psi.dag * O * psi
