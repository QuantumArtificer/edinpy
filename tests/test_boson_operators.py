import numpy as np
import pytest

from edinpy import boson as edb


def test_hopping_matches_literal_expression_and_is_hermitian():
    modes = edb.BosonModes(edb.DoF(2))
    sector = edb.NParticleSector(modes, N=2).build()
    t = 0.7 + 0.3j

    helper = edb.Hopping(0, 1, t, modes)
    literal = (
        t * edb.Creation(0, modes=modes) * edb.Annihilation(1, modes=modes)
        + t.conjugate()
        * edb.Creation(1, modes=modes)
        * edb.Annihilation(0, modes=modes)
    )

    helper_h = edb.Hamiltonian(helper, sector)
    literal_h = edb.Hamiltonian(literal, sector)
    assert np.allclose(helper_h.toarray(), literal_h.toarray())
    assert helper_h.is_hermitian()
    assert helper_h.compiler_stats["hopping_groups"] == 1
    assert helper_h.compiler_stats["hopping_terms"] == 2
    assert helper_h.compiler_stats["generic"] == 0


def test_onsite_uses_number_kernel():
    modes = edb.BosonModes(edb.DoF(2))
    sector = edb.NParticleSector(modes, N=3).build()
    operator = edb.Onsite(0, 2.5, modes)
    hamiltonian = edb.Hamiltonian(operator, sector)

    for state in sector.basis:
        occupations = state.occupations
        index = sector.basis.index(state)
        assert hamiltonian.toarray()[index, index] == pytest.approx(
            2.5 * occupations[0]
        )
    assert hamiltonian.compiler_stats["number_product"] == 1
    assert hamiltonian.compiler_stats["generic"] == 0


def test_density_density_has_expected_diagonal():
    modes = edb.BosonModes(edb.DoF(2))
    sector = edb.NParticleSector(modes, N=3).build()
    V = 1.25
    matrix = edb.Hamiltonian(
        edb.DensityDensity(0, 1, V, modes), sector
    ).toarray()

    for state in sector.basis:
        occupations = state.occupations
        index = sector.basis.index(state)
        assert matrix[index, index] == pytest.approx(
            V * occupations[0] * occupations[1]
        )


def test_hubbard_is_u_over_two_n_n_minus_one():
    modes = edb.BosonModes(edb.DoF(2))
    sector = edb.NParticleSector(modes, N=4).build()
    U = 1.7
    hamiltonian = edb.Hamiltonian(
        edb.Hubbard(0, U, modes), sector
    )
    matrix = hamiltonian.toarray()

    for state in sector.basis:
        occupations = state.occupations
        index = sector.basis.index(state)
        n = occupations[0]
        assert matrix[index, index] == pytest.approx(0.5 * U * n * (n - 1))

    stats = hamiltonian.compiler_stats
    assert stats["generic"] == 0
    assert stats["number_product"] == 2


def test_pair_hopping_has_correct_ladder_factor_and_hermitian_reverse():
    modes = edb.BosonModes(edb.DoF(2))
    sector = edb.NParticleSector(modes, N=2).build()
    J = 0.4 + 0.2j
    matrix = edb.Hamiltonian(edb.PairHopping(0, 1, J, modes), sector).toarray()

    source = sector.basis.index((0, 2))
    target = sector.basis.index((2, 0))
    assert matrix[target, source] == pytest.approx(2 * J)
    assert matrix[source, target] == pytest.approx(2 * J.conjugate())
    assert np.allclose(matrix, matrix.conj().T)

    stats = edb.Hamiltonian(edb.PairHopping(0, 1, J, modes), sector).compiler_stats
    assert stats["pair_hopping_groups"] == 1
    assert stats["pair_hopping_terms"] == 2
    assert stats["monomial"] == 0
    assert stats["generic"] == 0


def test_bose_hubbard_helper_sum_has_known_two_site_n2_spectrum():
    modes = edb.BosonModes(edb.DoF(2))
    sector = edb.NParticleSector(modes, N=2).build()
    t = 1.0
    U = 2.0
    operator = (
        edb.Hopping(0, 1, -t, modes)
        + edb.Hubbard(0, U, modes)
        + edb.Hubbard(1, U, modes)
    )

    hamiltonian = edb.Hamiltonian(operator, sector)
    values, _vectors = hamiltonian.eigsolve(sparse=False, k=None)

    # In the {|2,0>, |1,1>, |0,2>} basis, the antisymmetric doublon has
    # energy U and the remaining 2x2 block has eigenvalues
    # (U ± sqrt(U^2 + 16 t^2))/2.
    expected = np.sort(
        [
            U,
            0.5 * (U - np.sqrt(U * U + 16 * t * t)),
            0.5 * (U + np.sqrt(U * U + 16 * t * t)),
        ]
    )
    assert np.allclose(values, expected)


def test_helpers_respect_multidof_mode_labels():
    site = edb.DoF(2, name="site")
    species = edb.DoF(2, name="species")
    modes = edb.BosonModes(site, species)
    sector = edb.NParticleSector(modes, N=1).build()

    operator = edb.Hopping((0, 1), (1, 1), -1.0, modes)
    matrix = edb.Hamiltonian(operator, sector).matrix

    first = sector.basis.index((0, 0, 1, 0))
    second = sector.basis.index((0, 0, 0, 1))
    assert matrix[first, second] == pytest.approx(-1.0)
    assert matrix[second, first] == pytest.approx(-1.0)
    assert matrix.nnz == 2


def test_schwinger_spin_ladder_factors_are_bosonic():
    modes = edb.BosonModes(edb.DoF(2, labels=("up", "down")))
    sector = edb.NParticleSector(modes, N=3).build()
    source = sector.basis[sector.basis.index((1, 2))]

    raised = edb.SpinPlus(0, 1, modes) * source
    lowered = edb.SpinMinus(0, 1, modes) * source

    assert raised.occupations == (2, 1)
    assert raised.amp == pytest.approx(2.0)
    assert lowered.occupations == (0, 3)
    assert lowered.amp == pytest.approx(np.sqrt(3.0))


def test_schwinger_spin_commutator_matches_two_sz():
    modes = edb.BosonModes(edb.DoF(2, labels=("up", "down")))
    sector = edb.NParticleSector(modes, N=4).build()
    sp = edb.SpinPlus(0, 1, modes)
    sm = edb.SpinMinus(0, 1, modes)
    sz = edb.SpinZ(0, 1, modes)

    commutator = sp * sm - sm * sp
    left = edb.Hamiltonian(commutator, sector).toarray()
    right = edb.Hamiltonian(2 * sz, sector).toarray()

    assert np.allclose(left, right)


def test_spin_xyz_are_hermitian():
    modes = edb.BosonModes(edb.DoF(2, labels=("up", "down")))
    sector = edb.NParticleSector(modes, N=3).build()

    for operator in (
        edb.SpinX(0, 1, modes),
        edb.SpinY(0, 1, modes),
        edb.SpinZ(0, 1, modes),
    ):
        assert edb.Hamiltonian(operator, sector).is_hermitian()


def test_schwinger_heisenberg_exchange_has_spin_half_singlet_triplet_spectrum():
    site = edb.DoF(2, name="site", labels=("left", "right"))
    component = edb.DoF(2, name="component", labels=("up", "down"))
    modes = edb.BosonModes(site, component)
    sector = (
        edb.NParticleSector(modes, N=2)
        .project_particles("site", left=1, right=1)
        .build()
    )
    J = 1.7
    exchange = edb.HeisenbergExchange(
        (0, 0), (0, 1), (1, 0), (1, 1), J, modes
    )

    values, _vectors = edb.Hamiltonian(exchange, sector).eigsolve(
        sparse=False, k=None
    )

    expected = np.array([-0.75 * J, 0.25 * J, 0.25 * J, 0.25 * J])
    assert np.allclose(np.sort(values), expected)


def test_spin_helpers_accept_multidof_labels():
    site = edb.DoF(2, name="site")
    component = edb.DoF(2, name="component", labels=("up", "down"))
    modes = edb.BosonModes(site, component)
    sector = edb.NParticleSector(modes, N=1).build()

    sx = edb.SpinX((1, 0), (1, 1), modes)
    matrix = edb.Hamiltonian(sx, sector).toarray()
    up = sector.basis.index((0, 1, 0, 0))
    down = sector.basis.index((0, 0, 0, 1))

    assert matrix[up, down] == pytest.approx(0.5)
    assert matrix[down, up] == pytest.approx(0.5)
