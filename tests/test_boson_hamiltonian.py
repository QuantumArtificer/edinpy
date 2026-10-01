import numpy as np
import pytest

from edinpy import boson as edb


def literal_matrix(operator, sector):
    basis = sector.basis
    matrix = np.zeros((basis.dimension, basis.dimension), dtype=complex)
    for column in range(basis.dimension):
        result = operator * basis.state(column)
        if isinstance(result, edb.NullState):
            continue
        states = (result,) if isinstance(result, edb.FockState) else result.states
        for state in states:
            if state.N != sector.N:
                continue
            try:
                row = basis.index(state.occupations)
            except ValueError:
                continue
            matrix[row, column] += state.amp
    return matrix


def test_hamiltonian_matches_literal_reference_for_bose_hubbard_terms():
    modes = edb.BosonModes(edb.DoF(3))
    sector = edb.NParticleSector(modes, N=3).build()
    b = edb.set_notation(edb.Annihilation, modes)
    bd = edb.set_notation(edb.Creation, modes)
    n = edb.set_notation(edb.Number, modes)

    operator = 0
    for site in range(3):
        operator += 0.7 * n(site) + 0.4 * bd(site) * bd(site) * b(site) * b(site)
    for left, right in ((0, 1), (1, 2)):
        operator += -1.1 * (bd(left) * b(right) + bd(right) * b(left))

    hamiltonian = edb.Hamiltonian(operator, sector)
    assert np.allclose(hamiltonian.toarray(), literal_matrix(operator, sector))
    assert hamiltonian.is_hermitian()
    assert hamiltonian.matrix.dtype == np.float64


def test_number_changing_terms_are_projected_out_of_fixed_n_hamiltonian():
    modes = edb.BosonModes(edb.DoF(2))
    sector = edb.NParticleSector(modes, N=2).build()
    b = edb.set_notation(edb.Annihilation, modes)
    bd = edb.set_notation(edb.Creation, modes)
    n = edb.set_notation(edb.Number, modes)
    operator = n(0) + 7 * b(0) + 5 * bd(1)

    hamiltonian = edb.Hamiltonian(operator, sector)
    expected = np.diag([2.0, 1.0, 0.0])
    assert np.array_equal(hamiltonian.toarray(), expected)
    assert hamiltonian.compiler_stats["projected_out"] == 2


def test_pair_hopping_uses_correct_bosonic_ladder_factor():
    modes = edb.BosonModes(edb.DoF(2))
    sector = edb.NParticleSector(modes, N=2).build()
    b = edb.set_notation(edb.Annihilation, modes)
    bd = edb.set_notation(edb.Creation, modes)
    operator = bd(0) * bd(0) * b(1) * b(1)

    matrix = edb.Hamiltonian(operator, sector).toarray()
    source = sector.basis.index((0, 2))
    target = sector.basis.index((2, 0))
    assert matrix[target, source] == pytest.approx(2.0)
    assert np.count_nonzero(matrix) == 1


def test_complex_hermitian_hopping_preserves_complex_dtype():
    modes = edb.BosonModes(edb.DoF(2))
    sector = edb.NParticleSector(modes, N=1).build()
    b = edb.set_notation(edb.Annihilation, modes)
    bd = edb.set_notation(edb.Creation, modes)
    operator = 1j * bd(0) * b(1) - 1j * bd(1) * b(0)

    hamiltonian = edb.Hamiltonian(operator, sector)
    assert hamiltonian.matrix.dtype == np.complex128
    assert hamiltonian.is_hermitian()
    assert np.allclose(hamiltonian.toarray(), [[0, 1j], [-1j, 0]])


def test_two_mode_one_boson_hopping_has_known_spectrum():
    modes = edb.BosonModes(edb.DoF(2))
    sector = edb.NParticleSector(modes, N=1).build()
    b = edb.set_notation(edb.Annihilation, modes)
    bd = edb.set_notation(edb.Creation, modes)
    t = 1.7
    operator = -t * (bd(0) * b(1) + bd(1) * b(0))

    hamiltonian = edb.Hamiltonian(operator, sector)
    values, vectors = hamiltonian.eigsolve(sparse=False, k=None)
    assert np.allclose(values, [-t, t])
    assert vectors.shape == (2, 2)
    assert hamiltonian.eigenstate(0).norm() == pytest.approx(1.0)


def test_projected_sector_matrix_skips_transitions_outside_sector():
    site = edb.DoF(2, name="site")
    species = edb.DoF(2, name="species", labels=("a", "b"))
    modes = edb.BosonModes(site, species)
    sector = (
        edb.NParticleSector(modes, N=2)
        .project_particles("species", a=1, b=1)
        .build()
    )
    b = edb.set_notation(edb.Annihilation, modes)
    bd = edb.set_notation(edb.Creation, modes)

    # This converts species b at site 0 into species a at site 0, so the
    # result leaves the fixed (Na, Nb)=(1,1) projected sector.
    a0 = modes.resolve((0, 0))
    b0 = modes.resolve((0, 1))
    operator = edb.Creation(a0, modes=modes) * edb.Annihilation(b0, modes=modes)

    matrix = edb.Hamiltonian(operator, sector).matrix
    assert matrix.nnz == 0


def test_basis_backed_bra_operator_ket_uses_sector_matrix():
    modes = edb.BosonModes(edb.DoF(2))
    sector = edb.NParticleSector(modes, N=2).build()
    b = edb.set_notation(edb.Annihilation, modes)
    bd = edb.set_notation(edb.Creation, modes)
    operator = bd(0) * b(1) + bd(1) * b(0)
    left = sector.from_vector([1, 2, 3])
    right = sector.from_vector([3, -1, 2])

    expected = np.vdot(left.coefficients, edb.Hamiltonian(operator, sector).matrix @ right.coefficients)
    assert left.dag * operator * right == pytest.approx(expected)


class OccupationPlusOne(edb.Operator):
    """Small custom operator used to exercise the literal fallback path."""

    def _action(self, state):
        return edb.FockState(
            state.occupations,
            amp=(state.occupation(0) + 1) * state.amp,
            index=state.index,
        )

    @property
    def dag(self):
        return self

    @property
    def string(self):
        return "occupation_plus_one"


def test_custom_operator_uses_generic_fallback():
    modes = edb.BosonModes(edb.DoF(2))
    sector = edb.NParticleSector(modes, N=2).build()
    operator = OccupationPlusOne()
    hamiltonian = edb.Hamiltonian(operator, sector)

    assert hamiltonian.compiler_stats["generic"] == 1
    assert np.array_equal(hamiltonian.toarray(), np.diag([3.0, 2.0, 1.0]))

    vector = np.array([1.0, -2.0, 0.5])
    mixed = hamiltonian.aslinearoperator(execution="mixed") @ vector
    assert hamiltonian._resolve_execution("mixed") == "numpy"
    assert np.allclose(mixed, hamiltonian.matrix @ vector)

    with pytest.raises(NotImplementedError, match="generic symbolic fallback"):
        hamiltonian.aslinearoperator(execution="numba-serial")


def test_multiword_sparse_diagonal_handles_split_occupation_field():
    modes = edb.BosonModes(edb.DoF(22))
    sector = edb.NParticleSector(modes, N=4).build()
    n = edb.set_notation(edb.Number, modes)

    # N=4 requires three bits per occupation. Mode 21 begins at bit 63, so
    # its field is split between the first and second uint64 words.
    assert sector.basis.bits_per_mode == 3
    assert 21 * sector.basis.bits_per_mode == 63
    assert sector.basis.packed_width == 66

    operator = 2.0 * n(20) + 3.0 * n(21) + 0.5 * n(21) * n(21)
    matrix = edb.Hamiltonian(operator, sector).matrix
    expected = np.asarray(
        [
            2.0 * state.occupation(20)
            + 3.0 * state.occupation(21)
            + 0.5 * state.occupation(21) ** 2
            for state in (sector.basis.state(i) for i in range(sector.dimension))
        ]
    )

    assert sector.basis._uint64_words().shape == (sector.dimension, 2)
    assert np.array_equal(matrix.diagonal(), expected)


def test_multiword_sparse_hopping_and_pair_hopping_match_ladder_factors():
    modes = edb.BosonModes(edb.DoF(22))
    sector = edb.NParticleSector(modes, N=4).build()
    operator = (
        edb.Hopping(0, 21, -0.7, modes)
        + edb.PairHopping(0, 21, 0.25, modes)
    )
    matrix = edb.Hamiltonian(operator, sector).matrix

    assert sector.basis.packed_width == 66
    assert 21 * sector.basis.bits_per_mode == 63
    assert sector.basis._uint64_words().shape[1] == 2

    # One-boson transfer from mode 21 to mode 0.
    source = (0,) * 21 + (4,)
    target = (1,) + (0,) * 20 + (3,)
    column = sector.basis.index(source)
    row = sector.basis.index(target)
    assert matrix[row, column] == pytest.approx(-0.7 * np.sqrt(4.0))

    # Two-boson transfer from mode 21 to mode 0.
    pair_target = (2,) + (0,) * 20 + (2,)
    pair_row = sector.basis.index(pair_target)
    expected = 0.25 * np.sqrt(1 * 2 * 4 * 3)
    assert matrix[pair_row, column] == pytest.approx(expected)


def test_projected_multiword_sparse_transitions_respect_projection():
    site = edb.DoF(11, name="site")
    species = edb.DoF(2, name="species", labels=("a", "b"))
    modes = edb.BosonModes(site, species)
    sector = (
        edb.NParticleSector(modes, N=4)
        .project_particles("species", a=2, b=2)
        .build()
    )

    # With three bits per occupation, mode 21 begins at bit 63 and crosses
    # the uint64 boundary. Modes 20 and 21 both belong to species b.
    assert sector.basis.bits_per_mode == 3
    assert sector.basis.packed_width == 66
    assert 21 * sector.basis.bits_per_mode == 63

    operator = (
        edb.Hopping(20, 21, -0.7, modes)
        + edb.PairHopping(20, 21, 0.25, modes)
    )
    matrix = edb.Hamiltonian(operator, sector).matrix

    source = (2,) + (0,) * 20 + (2,)
    one_target = (2,) + (0,) * 19 + (1, 1)
    pair_target = (2,) + (0,) * 19 + (2, 0)
    column = sector.basis.index(source)
    one_row = sector.basis.index(one_target)
    pair_row = sector.basis.index(pair_target)

    assert matrix[one_row, column] == pytest.approx(-0.7 * np.sqrt(2.0))
    assert matrix[pair_row, column] == pytest.approx(0.5)


def test_extended_bose_hubbard_chain_is_real_float64():
    modes = edb.BosonModes(edb.DoF(4))
    sector = edb.NParticleSector(modes, 3).build()
    H = 0
    for i in range(3):
        H += edb.Hopping(i, i + 1, -0.7, modes)
        H += edb.DensityDensity(i, i + 1, 0.2, modes)
    for i in range(4):
        H += edb.Onsite(i, 0.13 * i, modes)
        H += edb.Hubbard(i, 0.9, modes)

    matrix = edb.Hamiltonian(H, sector).matrix

    assert matrix.dtype == np.float64
    assert np.allclose(matrix.toarray(), matrix.toarray().T)


def test_complex_typed_zero_imaginary_coefficient_remains_real():
    modes = edb.BosonModes(edb.DoF(3))
    sector = edb.NParticleSector(modes, 2).build()
    H = edb.Hopping(0, 1, -1.0 + 0.0j, modes) + edb.Onsite(2, 0.5 + 0.0j, modes)

    matrix = edb.Hamiltonian(H, sector).matrix

    assert matrix.dtype == np.float64


def test_nonhermitian_hamiltonian_is_rejected_by_eigsolve():
    modes = edb.BosonModes(edb.DoF(3))
    sector = edb.NParticleSector(modes, 2).build()
    H = edb.Creation(0, modes=modes) * edb.Annihilation(1, modes=modes)
    ham = edb.Hamiltonian(H, sector)

    assert not ham.is_hermitian()
    with pytest.raises(ValueError, match="Hermitian"):
        ham.eigsolve()


def test_dense_and_sparse_eigenvalues_agree():
    modes = edb.BosonModes(edb.DoF(5))
    sector = edb.NParticleSector(modes, 3).build()
    H = sum((edb.Hopping(i, i + 1, -0.8, modes) for i in range(4)), start=0)
    H += sum((edb.Onsite(i, 0.17 * i, modes) for i in range(5)), start=0)
    H += sum((edb.Hubbard(i, 0.31, modes) for i in range(5)), start=0)

    sparse_values, _ = edb.Hamiltonian(H, sector).eigsolve(
        sparse=True, k=5, which="SA", tol=1e-12
    )
    dense_values, _ = edb.Hamiltonian(H, sector).eigsolve(
        sparse=False, k=5, which="SA"
    )

    assert np.allclose(sparse_values, dense_values, atol=1e-10)


def test_k_none_returns_complete_eigensystem():
    modes = edb.BosonModes(edb.DoF(3))
    sector = edb.NParticleSector(modes, 2).build()
    H = edb.Onsite(0, 0.2, modes) + edb.Onsite(1, 0.7, modes) + edb.Onsite(2, 1.9, modes)
    ham = edb.Hamiltonian(H, sector)

    values, vectors = ham.eigsolve(k=None, which="SA")

    assert values.shape == (sector.dimension,)
    assert vectors.shape == (sector.dimension, sector.dimension)
    assert np.allclose(vectors.conj().T @ vectors, np.eye(sector.dimension), atol=1e-12)


@pytest.mark.parametrize("which", ["SA", "LA", "SM", "LM"])
def test_eigenvalue_order_matches_requested_spectrum(which):
    modes = edb.BosonModes(edb.DoF(4))
    sector = edb.NParticleSector(modes, 1).build()
    energies = (0.3, -1.7, 2.4, 4.1)
    H = sum((edb.Onsite(i, energy, modes) for i, energy in enumerate(energies)), start=0)

    values, _ = edb.Hamiltonian(H, sector).eigsolve(sparse=False, k=None, which=which)
    expected = np.asarray(energies, dtype=float)
    if which == "SA":
        expected = np.sort(expected)
    elif which == "LA":
        expected = np.sort(expected)[::-1]
    elif which == "SM":
        expected = expected[np.argsort(np.abs(expected))]
    else:
        expected = expected[np.argsort(np.abs(expected))[::-1]]

    assert np.allclose(values, expected)


def test_operator_and_sector_mode_mismatch_is_rejected():
    modes_a = edb.BosonModes(edb.DoF(3))
    modes_b = edb.BosonModes(edb.DoF(3))
    sector = edb.NParticleSector(modes_a, 2).build()
    H = edb.Onsite(0, 1.0, modes_b)

    with pytest.raises(ValueError, match="different BosonModes"):
        edb.Hamiltonian(H, sector)


def test_hamiltonian_requires_built_sector():
    modes = edb.BosonModes(edb.DoF(3))
    sector = edb.NParticleSector(modes, 2)

    with pytest.raises(RuntimeError, match="sector.build"):
        edb.Hamiltonian(edb.Onsite(0, 1.0, modes), sector)


def test_projected_sector_hamiltonian_matches_full_sector_submatrix():
    site = edb.DoF(3, name="site")
    species = edb.DoF(2, name="species", labels=("a", "b"))
    modes = edb.BosonModes(site, species)
    full = edb.NParticleSector(modes, 3).build()
    projected = (
        edb.NParticleSector(modes, 3)
        .project_particles("species", a=2, b=1)
        .build()
    )

    H = 0
    for species_index in range(2):
        for i in range(2):
            H += edb.Hopping((i, species_index), (i + 1, species_index), -0.6, modes)
    for mode in range(modes.n_modes):
        H += edb.Onsite(mode, 0.07 * mode, modes)

    full_matrix = edb.Hamiltonian(H, full).matrix
    projected_matrix = edb.Hamiltonian(H, projected).matrix
    keep = np.asarray([full.basis.index(state) for state in projected.basis.states])
    reference = full_matrix[keep][:, keep]

    assert np.allclose(projected_matrix.toarray(), reference.toarray())


def test_two_dof_projected_hamiltonian_matches_full_sector_submatrix():
    site = edb.DoF(2, name="site")
    species = edb.DoF(2, name="species", labels=("a", "b"))
    layer = edb.DoF(2, name="layer", labels=("top", "bottom"))
    modes = edb.BosonModes(site, species, layer)
    full = edb.NParticleSector(modes, 3).build()
    projected = (
        edb.NParticleSector(modes, 3)
        .project_particles("species", a=2, b=1)
        .project_particles("layer", top=1, bottom=2)
        .build()
    )

    H = 0
    for species_index in range(2):
        for layer_index in range(2):
            H += edb.Hopping(
                (0, species_index, layer_index),
                (1, species_index, layer_index),
                -0.4,
                modes,
            )
    for mode in range(modes.n_modes):
        H += edb.Onsite(mode, 0.05 * mode, modes)

    full_matrix = edb.Hamiltonian(H, full).matrix
    projected_matrix = edb.Hamiltonian(H, projected).matrix
    keep = np.asarray([full.basis.index(state) for state in projected.basis.states])
    reference = full_matrix[keep][:, keep]

    assert np.allclose(projected_matrix.toarray(), reference.toarray())
