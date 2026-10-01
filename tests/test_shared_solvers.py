import numpy as np
import pytest
from scipy.sparse import csc_matrix

from edinpy import boson as edb
from edinpy._core._solvers import (
    is_hermitian_matrix,
    solve_hermitian_matrix,
    spectral_order,
)


def test_spectral_order_matches_existing_which_semantics():
    values = np.array([-4.0, -1.0, 2.0, 7.0])
    assert np.array_equal(spectral_order(values, "SA"), [0, 1, 2, 3])
    assert np.array_equal(spectral_order(values, "LA"), [3, 2, 1, 0])
    assert np.array_equal(spectral_order(values, "SM"), [1, 2, 0, 3])
    assert np.array_equal(spectral_order(values, "LM"), [3, 0, 2, 1])


def test_shared_solver_dense_and_sparse_paths_agree():
    matrix = csc_matrix(
        np.array(
            [
                [2.0, -1.0, 0.0, 0.0],
                [-1.0, 1.0, -0.5, 0.0],
                [0.0, -0.5, 3.0, 0.4],
                [0.0, 0.0, 0.4, 4.0],
            ]
        )
    )
    sparse_values, _ = solve_hermitian_matrix(
        matrix, sparse=True, k=2, which="SA", tol=1e-12
    )
    dense_values, _ = solve_hermitian_matrix(
        matrix, sparse=False, k=2, which="SA"
    )
    assert np.allclose(sparse_values, dense_values, atol=1e-11)


def test_shared_solver_keeps_complex_hermitian_support():
    matrix = csc_matrix(np.array([[0.0, 1j], [-1j, 0.0]], dtype=complex))
    assert is_hermitian_matrix(matrix)
    values, vectors = solve_hermitian_matrix(matrix, sparse=False, k=None)
    assert np.allclose(values, [-1.0, 1.0])
    assert vectors.shape == (2, 2)


def test_shared_solver_rejects_nonhermitian_matrix():
    matrix = csc_matrix(np.array([[0.0, 1.0], [0.0, 0.0]]))
    assert not is_hermitian_matrix(matrix)
    with pytest.raises(ValueError, match="Hermitian"):
        solve_hermitian_matrix(matrix, sparse=False, k=None)


def test_hamiltonian_eigsolve_delegates_without_changing_results():
    modes = edb.BosonModes(edb.DoF(3))
    sector = edb.NParticleSector(modes, N=2).build()
    operator = (
        edb.Hopping(0, 1, -0.7, modes)
        + edb.Hopping(1, 2, -0.4, modes)
        + edb.Onsite(0, 0.2, modes)
        + edb.Onsite(2, -0.1, modes)
    )
    hamiltonian = edb.Hamiltonian(operator, sector)
    expected = np.linalg.eigvalsh(hamiltonian.toarray())[:3]
    values, vectors = hamiltonian.eigsolve(
        sparse=True, k=3, which="SA", tol=1e-12
    )
    assert np.allclose(values, expected, atol=1e-11)
    assert vectors.shape == (sector.dimension, 3)
    assert np.allclose(hamiltonian.eigvals, values)
    assert np.allclose(hamiltonian.eigvecs, vectors)
