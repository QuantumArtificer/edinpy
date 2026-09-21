import numpy as np

from edinpy import fermion as edf


def matrices_equal(left, right, sector):
    a = edf.Hamiltonian(left, sector).toarray()
    b = edf.Hamiltonian(right, sector).toarray()
    return np.allclose(a, b, atol=1e-12)


def test_hopping_matches_literal_algebra():
    modes = edf.FermionModes(edf.DoF(4))
    sector = edf.NParticleSector(modes, 2)
    c = edf.set_notation(edf.Annihilation, modes)
    cd = edf.set_notation(edf.Creation, modes)
    t = -0.7 + 0.2j
    literal = t * cd(0) * c(3) + np.conjugate(t) * cd(3) * c(0)
    assert matrices_equal(edf.Hopping(0, 3, t, modes), literal, sector)


def test_onsite_matches_literal_algebra():
    modes = edf.FermionModes(edf.DoF(4))
    sector = edf.NParticleSector(modes, 2)
    n = edf.set_notation(edf.Number, modes)
    assert matrices_equal(edf.Onsite(2, 1.7, modes), 1.7 * n(2), sector)


def test_density_density_and_hubbard_match_literal_algebra():
    modes = edf.FermionModes(edf.DoF(2, "site"), edf.DoF(2, "spin"))
    sector = edf.NParticleSector(modes, 2)
    n = edf.set_notation(edf.Number, modes)
    literal = 3.2 * n(0, 0) * n(0, 1)
    assert matrices_equal(edf.Hubbard((0, 0), (0, 1), 3.2, modes), literal, sector)
    assert matrices_equal(edf.DensityDensity((0, 0), (0, 1), 3.2, modes), literal, sector)


def test_spin_operators_match_standard_fermionic_definitions():
    modes = edf.FermionModes(edf.DoF(1, "site"), edf.DoF(2, "spin"))
    sector = edf.NParticleSector(modes, 1)
    c = edf.set_notation(edf.Annihilation, modes)
    cd = edf.set_notation(edf.Creation, modes)
    n = edf.set_notation(edf.Number, modes)
    up, down = (0, 0), (0, 1)
    assert matrices_equal(edf.SpinPlus(up, down, modes), cd(*up) * c(*down), sector)
    assert matrices_equal(edf.SpinMinus(up, down, modes), cd(*down) * c(*up), sector)
    assert matrices_equal(edf.SpinZ(up, down, modes), 0.5 * (n(*up) - n(*down)), sector)


def test_heisenberg_exchange_matches_spin_definition():
    modes = edf.FermionModes(edf.DoF(2, "site"), edf.DoF(2, "spin"))
    sector = edf.NParticleSector(modes, 2)
    i_up, i_down = (0, 0), (0, 1)
    j_up, j_down = (1, 0), (1, 1)
    J = 0.8
    literal = J * (
        edf.SpinZ(i_up, i_down, modes) * edf.SpinZ(j_up, j_down, modes)
        + 0.5 * (
            edf.SpinPlus(i_up, i_down, modes) * edf.SpinMinus(j_up, j_down, modes)
            + edf.SpinMinus(i_up, i_down, modes) * edf.SpinPlus(j_up, j_down, modes)
        )
    )
    library = edf.HeisenbergExchange(i_up, i_down, j_up, j_down, J, modes)
    assert matrices_equal(library, literal, sector)


def test_pair_hopping_matches_literal_algebra():
    modes = edf.FermionModes(edf.DoF(2, "site"), edf.DoF(2, "spin"))
    sector = edf.NParticleSector(modes, 2)
    c = edf.set_notation(edf.Annihilation, modes)
    cd = edf.set_notation(edf.Creation, modes)
    i_up, i_down = (0, 0), (0, 1)
    j_up, j_down = (1, 0), (1, 1)
    J = 0.3 - 0.4j
    forward = cd(*i_up) * cd(*i_down) * c(*j_down) * c(*j_up)
    literal = J * forward + np.conjugate(J) * forward.dag
    library = edf.PairHopping(i_up, i_down, j_up, j_down, J, modes)
    assert matrices_equal(library, literal, sector)
