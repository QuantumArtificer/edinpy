import numpy as np
import pytest

from edinpy import fermion as edf


def make_spinful_chain(L, N):
    modes = edf.FermionModes(edf.DoF(L, "site"), edf.DoF(2, "spin"))
    sector = edf.NParticleSector(modes, N=N)
    c = edf.set_notation(edf.Annihilation, modes)
    cd = edf.set_notation(edf.Creation, modes)
    n = edf.set_notation(edf.Number, modes)
    return modes, sector, c, cd, n


def independent_apply_sequence(state, factors):
    amp = 1
    current = state
    for action, mode in reversed(factors):
        bit = 1 << mode
        occupied = bool(current & bit)
        if action == "a":
            if not occupied:
                return None, 0
        else:
            if occupied:
                return None, 0
        parity = (current & (bit - 1)).bit_count() & 1
        if parity:
            amp = -amp
        current ^= bit
    return current, amp


def test_extended_hubbard_chain_is_real_float64():
    L = 4
    modes, sector, c, cd, n = make_spinful_chain(L, N=4)
    H = 0
    t, U, V = 1.0, 4.0, 1.5
    for i in range(L - 1):
        for spin in (0, 1):
            H += -t * (cd(i, spin) * c(i + 1, spin) + cd(i + 1, spin) * c(i, spin))
    for i in range(L):
        H += U * n(i, 0) * n(i, 1)
    for i in range(L - 1):
        ni = n(i, 0) + n(i, 1)
        nj = n(i + 1, 0) + n(i + 1, 1)
        H += V * ni * nj
    ham = edf.Hamiltonian(H, sector)
    assert ham.matrix.dtype == np.float64
    assert ham.matrix.shape == (70, 70)
    assert ham.is_hermitian()


def test_genuinely_complex_hopping_uses_complex128():
    modes = edf.FermionModes(edf.DoF(4))
    sector = edf.NParticleSector(modes, 2)
    H = edf.Hopping(0, 1, 1j, modes) + edf.Hopping(2, 3, 0.3 + 0.7j, modes)
    ham = edf.Hamiltonian(H, sector)
    assert ham.matrix.dtype == np.complex128
    assert ham.is_hermitian()


def test_complex_typed_zero_imaginary_coefficient_remains_real():
    modes = edf.FermionModes(edf.DoF(3))
    sector = edf.NParticleSector(modes, 1)
    H = edf.Onsite(0, np.complex128(2 + 0j), modes)
    assert edf.Hamiltonian(H, sector).matrix.dtype == np.float64


def test_nonhermitian_hamiltonian_is_rejected_by_eigsolve():
    modes = edf.FermionModes(edf.DoF(3))
    sector = edf.NParticleSector(modes, 1)
    c = edf.set_notation(edf.Annihilation, modes)
    cd = edf.set_notation(edf.Creation, modes)
    ham = edf.Hamiltonian(cd(0) * c(1), sector)
    assert not ham.is_hermitian()
    with pytest.raises(ValueError):
        ham.eigsolve(k=1)


def test_dense_and_sparse_eigenpairs_agree():
    modes = edf.FermionModes(edf.DoF(8))
    sector = edf.NParticleSector(modes, 4)
    H = sum((edf.Hopping(i, i + 1, -1.0, modes) for i in range(7)), start=0)
    ham = edf.Hamiltonian(H, sector)
    sparse_vals, _ = ham.eigsolve(sparse=True, k=4, which="SA", tol=1e-12)
    dense_vals, _ = ham.eigsolve(sparse=False, k=4, which="SA")
    assert np.allclose(sparse_vals, dense_vals, atol=1e-10)


def test_k_none_returns_complete_eigensystem():
    modes = edf.FermionModes(edf.DoF(5))
    sector = edf.NParticleSector(modes, 2)
    H = sum((edf.Onsite(i, i + 1.0, modes) for i in range(5)), start=0)
    values, vectors = edf.Hamiltonian(H, sector).eigsolve(sparse=False, k=None)
    assert values.shape == (sector.dimension,)
    assert vectors.shape == (sector.dimension, sector.dimension)


@pytest.mark.parametrize("which", ["SA", "LA", "SM", "LM"])
def test_eigenvalue_order_matches_requested_spectrum(which):
    modes = edf.FermionModes(edf.DoF(4))
    sector = edf.NParticleSector(modes, 1)
    onsite = [-3.0, -1.0, 2.0, 5.0]
    H = sum((edf.Onsite(i, value, modes) for i, value in enumerate(onsite)), start=0)
    values, _ = edf.Hamiltonian(H, sector).eigsolve(sparse=False, k=2, which=which)
    if which == "SA":
        expected = [-3.0, -1.0]
    elif which == "LA":
        expected = [5.0, 2.0]
    elif which == "SM":
        expected = [-1.0, 2.0]
    else:
        expected = [5.0, -3.0]
    assert np.allclose(values, expected)


def test_random_number_conserving_hamiltonian_against_independent_reference():
    rng = np.random.default_rng(8127)
    n_modes = 6
    N = 3
    modes = edf.FermionModes(edf.DoF(n_modes))
    sector = edf.NParticleSector(modes, N)
    c = edf.set_notation(edf.Annihilation, modes)
    cd = edf.set_notation(edf.Creation, modes)
    n = edf.set_notation(edf.Number, modes)

    onsite = rng.normal(size=n_modes)
    hops = {(i, i + 1): rng.normal() + 1j * rng.normal() for i in range(n_modes - 1)}
    interactions = {(i, i + 2): rng.normal() for i in range(n_modes - 2)}

    H = 0
    for i, epsilon in enumerate(onsite):
        H += epsilon * n(i)
    for (i, j), t in hops.items():
        H += t * cd(i) * c(j) + np.conjugate(t) * cd(j) * c(i)
    for (i, j), V in interactions.items():
        H += V * n(i) * n(j)

    actual = edf.Hamiltonian(H, sector).toarray()
    expected = np.zeros_like(actual, dtype=np.complex128)
    lookup = {state: index for index, state in enumerate(sector.basis.states)}

    for col, state in enumerate(sector.basis.states):
        expected[col, col] += sum(onsite[i] for i in range(n_modes) if state & (1 << i))
        expected[col, col] += sum(
            V
            for (i, j), V in interactions.items()
            if state & (1 << i) and state & (1 << j)
        )
        for (i, j), t in hops.items():
            for coefficient, destination, source in ((t, i, j), (np.conjugate(t), j, i)):
                final, sign = independent_apply_sequence(
                    state,
                    (("c", destination), ("a", source)),
                )
                if final is not None:
                    expected[lookup[final], col] += coefficient * sign

    assert np.allclose(actual, expected, atol=1e-12)


def test_vacuum_sector_above_64_modes_does_not_enter_uint64_backend():
    modes = edf.FermionModes(edf.DoF(70))
    sector = edf.NParticleSector(modes, 0)
    H = edf.Onsite(69, 3.0, modes)
    matrix = edf.Hamiltonian(H, sector).matrix
    assert matrix.shape == (1, 1)
    assert matrix.dtype == np.float64
    assert matrix[0, 0] == 0


def test_operator_and_sector_mode_mismatch_is_rejected():
    modes_a = edf.FermionModes(edf.DoF(4))
    modes_b = edf.FermionModes(edf.DoF(4))
    sector = edf.NParticleSector(modes_b, 2)
    with pytest.raises(ValueError):
        edf.Hamiltonian(edf.Onsite(0, 1.0, modes_a), sector)
