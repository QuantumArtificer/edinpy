import numpy as np
import pytest

from edinpy import fermion as edf


def _fermion_chain(*, n_modes=6, particles=3, complex_hopping=False):
    modes = edf.FermionModes(edf.DoF(n_modes))
    sector = edf.NParticleSector(modes, N=particles).build()
    hopping = -0.6j if complex_hopping else -0.6
    H = 0
    for i in range(n_modes - 1):
        amplitude = hopping if i == 0 else -0.35
        H += edf.Hopping(i, i + 1, amplitude, modes)
        H += edf.DensityDensity(i, i + 1, 0.2, modes)
    H += edf.Onsite(0, -0.15, modes)
    return edf.Hamiltonian(H, sector)


def _assert_linear_matches_matrix(ham, vector, *, atol=1e-12):
    actual = ham.aslinearoperator() @ vector
    assert ham._matrix is None

    reference = edf.Hamiltonian(ham.operator, ham.sector).matrix @ vector
    assert np.allclose(actual, reference, atol=atol)


def test_fermion_matrix_free_real_matvec_matches_csc_without_materialization():
    ham = _fermion_chain()
    vector = np.linspace(-0.7, 1.1, ham.sector.dimension)
    _assert_linear_matches_matrix(ham, vector)


def test_fermion_matrix_free_complex_matvec_and_dtype_match_csc():
    ham = _fermion_chain(complex_hopping=True)
    linear = ham.aslinearoperator()
    assert linear.dtype == np.dtype(np.complex128)

    rng = np.random.default_rng(8127)
    vector = rng.normal(size=ham.sector.dimension) + 1j * rng.normal(
        size=ham.sector.dimension
    )
    _assert_linear_matches_matrix(ham, vector)


def test_fermion_linear_operator_rmatvec_matches_explicit_adjoint():
    modes = edf.FermionModes(edf.DoF(5))
    sector = edf.NParticleSector(modes, N=2).build()
    c = edf.set_notation(edf.Annihilation, modes)
    cd = edf.set_notation(edf.Creation, modes)
    H = (1.0 + 0.3j) * cd(0) * c(3)
    ham = edf.Hamiltonian(H, sector)

    vector = np.arange(1, sector.dimension + 1, dtype=float)
    actual = ham.aslinearoperator().rmatvec(vector)
    assert ham._matrix is None

    expected = edf.Hamiltonian(H, sector).matrix.getH() @ vector
    assert np.allclose(actual, expected, atol=1e-12)


def test_fermion_linear_operator_matmat_matches_explicit_matrix():
    ham = _fermion_chain()
    rng = np.random.default_rng(91)
    vectors = rng.normal(size=(ham.sector.dimension, 3))

    actual = ham.aslinearoperator() @ vectors
    assert ham._matrix is None
    expected = edf.Hamiltonian(ham.operator, ham.sector).matrix @ vectors
    assert np.allclose(actual, expected, atol=1e-12)


def test_fermion_matrix_free_eigsolve_matches_explicit_sparse_path():
    reference = _fermion_chain(n_modes=8, particles=4)
    expected, _ = reference.eigsolve(
        sparse=True,
        k=3,
        which="SA",
        tol=1e-12,
        ncv=8,
        v0=np.ones(reference.sector.dimension),
    )

    ham = _fermion_chain(n_modes=8, particles=4)
    values, vectors = ham.eigsolve(
        sparse=True,
        k=3,
        which="SA",
        tol=1e-12,
        ncv=8,
        v0=np.ones(ham.sector.dimension),
        matrix_free=True,
        check_hermitian=False,
    )

    assert ham._matrix is None
    assert np.allclose(values, expected, atol=1e-10)
    residuals = np.linalg.norm(
        ham.aslinearoperator() @ vectors - vectors * values,
        axis=0,
    )
    assert np.all(residuals < 1e-8)


def test_fermion_matrix_free_projected_sector_matches_csc():
    site = edf.DoF(3, name="site")
    spin = edf.DoF(2, name="spin", labels=("up", "down"))
    modes = edf.FermionModes(site, spin)
    sector = (
        edf.NParticleSector(modes, N=3)
        .project_particles("spin", up=2, down=1)
        .build()
    )

    H = 0
    for sigma in (0, 1):
        H += edf.Hopping((0, sigma), (1, sigma), -0.8, modes)
        H += edf.Hopping((1, sigma), (2, sigma), -0.5, modes)
    H += edf.Hubbard((1, 0), (1, 1), 1.2, modes)

    ham = edf.Hamiltonian(H, sector)
    vector = np.linspace(-1.0, 1.0, sector.dimension)
    _assert_linear_matches_matrix(ham, vector)


def test_fermion_matrix_free_above_64_modes_matches_csc():
    modes = edf.FermionModes(edf.DoF(70))
    sector = edf.NParticleSector(modes, N=1).build()
    H = edf.Hopping(0, 69, -0.75, modes) + edf.Onsite(66, 0.2, modes)
    ham = edf.Hamiltonian(H, sector)

    vector = np.linspace(-0.3, 0.9, sector.dimension)
    _assert_linear_matches_matrix(ham, vector)


@pytest.mark.parametrize("which", ["SA", "LA", "SM", "LM"])
def test_fermion_matrix_free_spectral_selection_matches_explicit(which):
    modes = edf.FermionModes(edf.DoF(5))
    sector = edf.NParticleSector(modes, N=1).build()
    onsite = (-3.0, -1.0, 0.5, 2.0, 5.0)
    H = sum(
        (edf.Onsite(i, value, modes) for i, value in enumerate(onsite)),
        start=0,
    )

    expected, _ = edf.Hamiltonian(H, sector).eigsolve(
        sparse=True,
        k=2,
        which=which,
        tol=1e-13,
        ncv=4,
        v0=np.ones(sector.dimension),
    )

    ham = edf.Hamiltonian(H, sector)
    actual, _ = ham.eigsolve(
        sparse=True,
        k=2,
        which=which,
        tol=1e-13,
        ncv=4,
        v0=np.ones(sector.dimension),
        matrix_free=True,
        check_hermitian=False,
    )

    assert ham._matrix is None
    assert np.allclose(actual, expected, atol=1e-10)


def test_fermion_number_changing_term_is_projected_out_matrix_free():
    modes = edf.FermionModes(edf.DoF(5))
    sector = edf.NParticleSector(modes, N=2).build()
    H = edf.Creation(0, modes=modes)
    ham = edf.Hamiltonian(H, sector)

    vector = np.arange(1, sector.dimension + 1, dtype=float)
    result = ham.aslinearoperator() @ vector

    assert ham._matrix is None
    assert np.allclose(result, 0)


def test_fermion_matrix_free_pair_hopping_matches_csc():
    modes = edf.FermionModes(edf.DoF(6))
    sector = edf.NParticleSector(modes, N=3).build()
    H = (
        edf.PairHopping(0, 1, 4, 5, 0.7, modes)
        + edf.Hopping(1, 2, -0.25, modes)
        + edf.Onsite(3, 0.4, modes)
    )
    ham = edf.Hamiltonian(H, sector)
    vector = np.linspace(-0.9, 0.8, sector.dimension)
    _assert_linear_matches_matrix(ham, vector)


def test_fermion_matrix_free_chunked_complete_basis_matches_csc(monkeypatch):
    from edinpy.fermion import _execution as execution

    monkeypatch.setattr(execution, "_DEFAULT_MATVEC_BLOCK_SIZE", 3)
    ham = _fermion_chain(n_modes=8, particles=4)
    vector = np.linspace(-0.8, 0.9, ham.sector.dimension)
    _assert_linear_matches_matrix(ham, vector)


def test_fermion_matrix_free_chunked_projected_basis_matches_csc(monkeypatch):
    from edinpy.fermion import _execution as execution

    monkeypatch.setattr(execution, "_DEFAULT_MATVEC_BLOCK_SIZE", 2)
    site = edf.DoF(3, name="site")
    spin = edf.DoF(2, name="spin", labels=("up", "down"))
    modes = edf.FermionModes(site, spin)
    sector = (
        edf.NParticleSector(modes, N=3)
        .project_particles("spin", up=2, down=1)
        .build()
    )
    H = (
        edf.Hopping((0, 0), (1, 0), -0.4 + 0.2j, modes)
        + edf.Hopping((1, 1), (2, 1), -0.7, modes)
        + edf.Hubbard((1, 0), (1, 1), 0.8, modes)
        + edf.PairHopping((0, 0), (0, 1), (2, 0), (2, 1), 0.3, modes)
    )
    ham = edf.Hamiltonian(H, sector)
    vector = np.linspace(-0.5, 1.0, sector.dimension) + 0.1j
    _assert_linear_matches_matrix(ham, vector)
