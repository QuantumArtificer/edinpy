import numpy as np
import pytest
from scipy.sparse.linalg import LinearOperator

from edinpy import boson as edb
from edinpy._core._solvers import solve_hermitian_operator


def _boson_test_hamiltonian(*, complex_hopping=False):
    modes = edb.BosonModes(edb.DoF(4))
    sector = edb.NParticleSector(modes, N=3).build()
    hopping = -0.7j if complex_hopping else -0.7
    operator = (
        edb.Hopping(0, 1, hopping, modes)
        + edb.Hopping(1, 2, -0.4, modes)
        + edb.Hopping(2, 3, -0.25, modes)
        + edb.Onsite(0, 0.2, modes)
        + edb.Onsite(3, -0.1, modes)
        + edb.Hubbard(2, 0.3, modes)
    )
    return edb.Hamiltonian(operator, sector)


def test_aslinearoperator_matches_csc_matvec_without_materializing_matrix():
    ham = _boson_test_hamiltonian()
    linear = ham.aslinearoperator()
    assert isinstance(linear, LinearOperator)
    assert ham._matrix is None

    vector = np.linspace(-0.8, 1.1, ham.sector.dimension)
    matrix_free_result = linear @ vector
    assert ham._matrix is None

    matrix = ham.matrix
    assert np.allclose(matrix_free_result, matrix @ vector)


def test_linear_operator_preserves_real_and_complex_dtype():
    real_ham = _boson_test_hamiltonian()
    complex_ham = _boson_test_hamiltonian(complex_hopping=True)
    assert real_ham.aslinearoperator().dtype == np.dtype(np.float64)
    assert complex_ham.aslinearoperator().dtype == np.dtype(np.complex128)


def test_linear_operator_rmatvec_uses_symbolic_adjoint():
    modes = edb.BosonModes(edb.DoF(3))
    sector = edb.NParticleSector(modes, N=2).build()
    operator = edb.Creation(0, modes=modes) * edb.Annihilation(1, modes=modes)
    ham = edb.Hamiltonian(operator, sector)
    linear = ham.aslinearoperator()

    vector = np.arange(1, sector.dimension + 1, dtype=float)
    matrix_free = linear.rmatvec(vector)
    assert ham._matrix is None
    explicit = ham.matrix.getH() @ vector
    assert np.allclose(matrix_free, explicit)


def test_matrix_free_eigsolve_matches_explicit_sparse_path_and_keeps_matrix_empty():
    reference = _boson_test_hamiltonian()
    expected, _ = reference.eigsolve(
        sparse=True,
        k=3,
        which="SA",
        tol=1e-12,
    )

    ham = _boson_test_hamiltonian()
    values, vectors = ham.eigsolve(
        sparse=True,
        k=3,
        which="SA",
        tol=1e-12,
        matrix_free=True,
        check_hermitian=False,
    )

    assert ham._matrix is None
    assert np.allclose(values, expected, atol=1e-10)
    assert vectors.shape == (ham.sector.dimension, 3)

    residuals = np.linalg.norm(
        ham.aslinearoperator() @ vectors - vectors * values,
        axis=0,
    )
    assert np.all(residuals < 1e-8)



def test_complex_matrix_free_eigsolve_matches_explicit_sparse_path():
    reference = _boson_test_hamiltonian(complex_hopping=True)
    expected, _ = reference.eigsolve(
        sparse=True,
        k=2,
        which="SA",
        tol=1e-12,
    )

    ham = _boson_test_hamiltonian(complex_hopping=True)
    values, _ = ham.eigsolve(
        sparse=True,
        k=2,
        which="SA",
        tol=1e-12,
        matrix_free=True,
        check_hermitian=False,
    )

    assert ham._matrix is None
    assert np.allclose(values, expected, atol=1e-10)

def test_matrix_free_solver_requires_partial_sparse_problem():
    matrix = np.diag([1.0, 2.0, 3.0])
    operator = LinearOperator(matrix.shape, matvec=lambda x: matrix @ x, dtype=float)

    with pytest.raises(ValueError, match="sparse=True"):
        solve_hermitian_operator(operator, sparse=False, k=1)
    with pytest.raises(ValueError, match="finite 'k'"):
        solve_hermitian_operator(operator, k=None)
    with pytest.raises(ValueError, match="k < Hamiltonian dimension"):
        solve_hermitian_operator(operator, k=3)


def test_matrix_free_eigsolve_requires_explicit_hermiticity_opt_out():
    ham = _boson_test_hamiltonian()
    with pytest.raises(ValueError, match="Hermiticity"):
        ham.eigsolve(matrix_free=True, k=2)
    assert ham._matrix is None


def test_projected_boson_matrix_free_matches_csc():
    site = edb.DoF(3, name="site")
    species = edb.DoF(2, name="species", labels=("a", "b"))
    modes = edb.BosonModes(site, species)
    sector = (
        edb.NParticleSector(modes, N=3)
        .project_particles("species", a=2, b=1)
        .build()
    )
    operator = (
        edb.Hopping((0, 0), (1, 0), -0.7, modes)
        + edb.Hopping((1, 1), (2, 1), -0.4, modes)
        + edb.DensityDensity((1, 0), (1, 1), 0.35, modes)
    )
    ham = edb.Hamiltonian(operator, sector)
    vector = np.linspace(-0.5, 0.75, sector.dimension)

    actual = ham.aslinearoperator() @ vector
    assert ham._matrix is None
    expected = ham.matrix @ vector
    assert np.allclose(actual, expected, atol=1e-12)


def test_boson_matrix_free_handles_multiword_basis():
    modes = edb.BosonModes(edb.DoF(70))
    sector = edb.NParticleSector(modes, N=1).build()
    operator = edb.Hopping(0, 69, -0.8, modes) + edb.Onsite(65, 0.4, modes)
    ham = edb.Hamiltonian(operator, sector)
    vector = np.linspace(-1.0, 1.0, sector.dimension)

    actual = ham.aslinearoperator() @ vector
    assert ham._matrix is None
    expected = ham.matrix @ vector
    assert np.allclose(actual, expected, atol=1e-12)
