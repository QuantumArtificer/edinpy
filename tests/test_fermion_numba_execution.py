"""Correctness tests for the optional fermion Numba executors."""

import numpy as np
import pytest

from edinpy import fermion as edf

pytest.importorskip("numba")


def _complete_problem(*, complex_terms: bool = False) -> edf.Hamiltonian:
    """Return a small complete-basis Hamiltonian covering all lowered families."""
    modes = edf.FermionModes(edf.DoF(8, name="mode"))
    sector = edf.NParticleSector(modes, N=4).build()
    hopping = -0.55 + (0.17j if complex_terms else 0.0)
    pair = 0.23j if complex_terms else 0.23

    operator = 0
    for index in range(7):
        coefficient = hopping if index == 0 else -0.31
        operator += edf.Hopping(index, index + 1, coefficient, modes)
        operator += edf.DensityDensity(index, index + 1, 0.14, modes)
    operator += edf.Onsite(3, -0.27, modes)
    operator += edf.PairHopping(0, 1, 6, 7, pair, modes)
    return edf.Hamiltonian(operator, sector)


def _vector(ham: edf.Hamiltonian, *, complex_valued: bool, seed: int) -> np.ndarray:
    """Return a deterministic real or complex vector for ``ham``."""
    rng = np.random.default_rng(seed)
    vector = rng.normal(size=ham.sector.dimension)
    if complex_valued:
        vector = vector + 1j * rng.normal(size=ham.sector.dimension)
    return vector


@pytest.mark.parametrize("execution", ["numba-serial", "numba-parallel"])
@pytest.mark.parametrize("complex_terms", [False, True])
def test_numba_execution_matches_numpy_and_csc(execution, complex_terms):
    ham = _complete_problem(complex_terms=complex_terms)
    vector = _vector(ham, complex_valued=complex_terms, seed=3101)

    actual = ham.aslinearoperator(execution=execution) @ vector
    expected_numpy = ham.aslinearoperator(execution="numpy") @ vector
    expected_csc = ham.matrix @ vector

    np.testing.assert_allclose(actual, expected_numpy, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(actual, expected_csc, rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("execution", ["numba-serial", "numba-parallel"])
def test_numba_execution_adjoint_matches_explicit_adjoint(execution):
    modes = edf.FermionModes(edf.DoF(6))
    sector = edf.NParticleSector(modes, N=3).build()
    c = edf.set_notation(edf.Annihilation, modes)
    cd = edf.set_notation(edf.Creation, modes)
    operator = (0.8 + 0.35j) * cd(0) * c(4) + 0.2 * edf.Number(2, modes=modes)
    ham = edf.Hamiltonian(operator, sector)
    vector = np.linspace(-0.7, 1.1, sector.dimension) + 0.13j

    actual = ham.aslinearoperator(execution=execution).rmatvec(vector)
    expected = ham.matrix.getH() @ vector
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)


def test_parallel_numba_still_rejects_projected_basis():
    site = edf.DoF(3, name="site")
    spin = edf.DoF(2, name="spin", labels=("up", "down"))
    modes = edf.FermionModes(site, spin)
    projected = (
        edf.NParticleSector(modes, N=3)
        .project_particles("spin", up=2, down=1)
        .build()
    )
    projected_ham = edf.Hamiltonian(edf.Number((0, 0), modes=modes), projected)
    with pytest.raises(NotImplementedError, match="projected/constrained"):
        projected_ham.aslinearoperator(execution="numba-parallel")


@pytest.mark.parametrize("execution", ["numba-serial", "numba-parallel"])
def test_strict_numba_execution_rejects_wide_basis(execution):
    wide_modes = edf.FermionModes(edf.DoF(70))
    wide_sector = edf.NParticleSector(wide_modes, N=1).build()
    wide_ham = edf.Hamiltonian(
        edf.Hopping(0, 69, -0.4, wide_modes),
        wide_sector,
    )
    with pytest.raises(NotImplementedError, match="more than 64"):
        wide_ham.aslinearoperator(execution=execution)


def test_mixed_uses_numba_serial_for_projected_and_complete_bases():
    site = edf.DoF(3, name="site")
    spin = edf.DoF(2, name="spin", labels=("up", "down"))
    modes = edf.FermionModes(site, spin)
    projected = (
        edf.NParticleSector(modes, N=3)
        .project_particles("spin", up=2, down=1)
        .build()
    )
    projected_ham = edf.Hamiltonian(
        edf.Hopping((0, 0), (1, 0), -0.6, modes)
        + edf.Hubbard((1, 0), (1, 1), 0.9, modes),
        projected,
    )
    vector = np.linspace(-1.0, 1.0, projected.dimension)
    np.testing.assert_allclose(
        projected_ham.aslinearoperator(execution="mixed") @ vector,
        projected_ham.aslinearoperator(execution="numpy") @ vector,
        rtol=1e-13,
        atol=1e-13,
    )
    assert projected_ham._resolve_execution("mixed") == "numba-serial"

    complete_ham = _complete_problem()
    assert complete_ham._resolve_execution("mixed") == "numba-serial"


def _projected_spin_problem(*, noncontiguous: bool = False) -> edf.Hamiltonian:
    """Return a spin-conserving projected Hamiltonian with mixed kernel families."""
    site = edf.DoF(4, name="site")
    spin = edf.DoF(2, name="spin", labels=("up", "down"))
    modes = (
        edf.FermionModes(spin, site)
        if noncontiguous
        else edf.FermionModes(site, spin)
    )
    sector = (
        edf.NParticleSector(modes, N=4)
        .project_particles("spin", up=2, down=2)
        .build()
    )

    def label(site_index, spin_index):
        return (
            (spin_index, site_index)
            if noncontiguous
            else (site_index, spin_index)
        )

    operator = 0
    for site_index in range(3):
        for spin_index in (0, 1):
            operator += edf.Hopping(
                label(site_index, spin_index),
                label(site_index + 1, spin_index),
                -0.41 + (0.07j if spin_index else 0.0),
                modes,
            )
    for site_index in range(4):
        operator += edf.Hubbard(
            label(site_index, 0),
            label(site_index, 1),
            0.63,
            modes,
        )
    operator += edf.PairHopping(
        label(0, 0),
        label(0, 1),
        label(3, 0),
        label(3, 1),
        0.19,
        modes,
    )
    return edf.Hamiltonian(operator, sector)


def test_serial_numba_pairs_exactly_reversible_monomials():
    from edinpy.fermion._numba_execution import _get_execution_plan

    modes = edf.FermionModes(edf.DoF(6))
    sector = edf.NParticleSector(modes, N=2).build()
    c = edf.set_notation(edf.Annihilation, modes)
    cd = edf.set_notation(edf.Creation, modes)

    forward = 0.37 * cd(4) * cd(5) * c(1) * c(0)
    ham = edf.Hamiltonian(forward + forward.dag, sector)
    plan = _get_execution_plan(ham._compiled, sector.basis).operator

    assert plan.monomial_transition_masks.size == 2
    assert plan.monomial_pair_forward_indices.size == 1
    assert plan.monomial_pair_reverse_indices.size == 1
    assert plan.monomial_unpaired_indices.size == 0

    vector = _vector(ham, complex_valued=False, seed=3310)
    actual = ham.aslinearoperator(execution="numba-serial") @ vector
    expected = ham.matrix @ vector
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)


def test_serial_numba_leaves_unpaired_monomial_on_residual_path():
    from edinpy.fermion._numba_execution import _get_execution_plan

    modes = edf.FermionModes(edf.DoF(6))
    sector = edf.NParticleSector(modes, N=2).build()
    c = edf.set_notation(edf.Annihilation, modes)
    cd = edf.set_notation(edf.Creation, modes)

    operator = (0.23 + 0.11j) * cd(4) * cd(5) * c(1) * c(0)
    ham = edf.Hamiltonian(operator, sector)
    plan = _get_execution_plan(ham._compiled, sector.basis).operator

    assert plan.monomial_pair_forward_indices.size == 0
    assert plan.monomial_pair_reverse_indices.size == 0
    assert plan.monomial_unpaired_indices.size == 1

    vector = _vector(ham, complex_valued=True, seed=3311)
    actual = ham.aslinearoperator(execution="numba-serial") @ vector
    expected = ham.matrix @ vector
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)


def test_projected_numba_rank_strategy_matches_basis_geometry():
    from edinpy.fermion._numba_execution import _get_execution_plan

    contiguous = _projected_spin_problem(noncontiguous=False)
    contiguous_plan = _get_execution_plan(
        contiguous._compiled,
        contiguous.sector.basis,
    )
    assert contiguous_plan.lookup_strategy == "factorized-contiguous"
    assert contiguous_plan.factorized_meta.shape == (2, 4)
    assert contiguous_plan.factorized_lookup.size == 2 * (1 << 4)

    interleaved = _projected_spin_problem(noncontiguous=True)
    interleaved_plan = _get_execution_plan(
        interleaved._compiled,
        interleaved.sector.basis,
    )
    assert interleaved_plan.lookup_strategy == "binary-search"
    assert interleaved_plan.factorized_meta.size == 0
    assert interleaved_plan.factorized_lookup.size == 0


@pytest.mark.parametrize("noncontiguous", [False, True])
def test_projected_numba_serial_matches_numpy_and_csc(noncontiguous):
    ham = _projected_spin_problem(noncontiguous=noncontiguous)
    vector = _vector(ham, complex_valued=True, seed=3301)

    actual = ham.aslinearoperator(execution="numba-serial") @ vector
    expected_numpy = ham.aslinearoperator(execution="numpy") @ vector
    expected_csc = ham.matrix @ vector

    np.testing.assert_allclose(actual, expected_numpy, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(actual, expected_csc, rtol=1e-13, atol=1e-13)


def test_projected_numba_serial_drops_transitions_leaving_sector():
    site = edf.DoF(3, name="site")
    spin = edf.DoF(2, name="spin", labels=("up", "down"))
    modes = edf.FermionModes(site, spin)
    sector = (
        edf.NParticleSector(modes, N=3)
        .project_particles("spin", up=2, down=1)
        .build()
    )

    spin_flip = edf.Hopping((1, 0), (1, 1), 0.73, modes)
    ham = edf.Hamiltonian(spin_flip, sector)
    vector = _vector(ham, complex_valued=False, seed=3302)

    actual = ham.aslinearoperator(execution="numba-serial") @ vector
    expected_numpy = ham.aslinearoperator(execution="numpy") @ vector
    expected_csc = ham.matrix @ vector

    np.testing.assert_allclose(actual, 0.0, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(actual, expected_numpy, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(actual, expected_csc, rtol=0.0, atol=0.0)


def test_projected_numba_serial_supports_multiple_population_constraints():
    site = edf.DoF(2, name="site")
    spin = edf.DoF(2, name="spin", labels=("up", "down"))
    layer = edf.DoF(2, name="layer", labels=("top", "bottom"))
    modes = edf.FermionModes(site, spin, layer)
    sector = (
        edf.NParticleSector(modes, N=4)
        .project_particles("spin", up=2, down=2)
        .project_particles("layer", top=2, bottom=2)
        .build()
    )

    operator = 0
    for spin_index in (0, 1):
        for layer_index in (0, 1):
            operator += edf.Hopping(
                (0, spin_index, layer_index),
                (1, spin_index, layer_index),
                -0.37,
                modes,
            )
    for site_index in (0, 1):
        for layer_index in (0, 1):
            operator += edf.Hubbard(
                (site_index, 0, layer_index),
                (site_index, 1, layer_index),
                0.52,
                modes,
            )

    ham = edf.Hamiltonian(operator, sector)
    vector = _vector(ham, complex_valued=False, seed=3303)

    actual = ham.aslinearoperator(execution="numba-serial") @ vector
    expected_numpy = ham.aslinearoperator(execution="numpy") @ vector
    expected_csc = ham.matrix @ vector

    assert sector.dimension == 18
    np.testing.assert_allclose(actual, expected_numpy, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(actual, expected_csc, rtol=1e-13, atol=1e-13)


def test_projected_numba_serial_eigsolve_converges():
    ham = _projected_spin_problem(noncontiguous=True)
    rng = np.random.default_rng(3304)
    values, vectors = ham.eigsolve(
        sparse=True,
        k=2,
        which="SA",
        tol=1e-10,
        ncv=10,
        v0=rng.normal(size=ham.sector.dimension),
        check_hermitian=False,
        matrix_free=True,
        execution="numba-serial",
    )

    diagnostics = ham.solver_diagnostics
    assert diagnostics.execution_resolved == "numba-serial"
    assert diagnostics.matvec_calls > 0

    residuals = np.linalg.norm(
        ham.aslinearoperator(execution="numba-serial") @ vectors
        - vectors * values,
        axis=0,
    )
    assert np.all(residuals < 1e-8)


def test_execution_selector_validation_and_matrix_free_guard():
    ham = _complete_problem()

    with pytest.raises(ValueError, match="Unknown execution mode"):
        ham.aslinearoperator(execution="jit")
    for execution in ("numba-serial", "numba-parallel"):
        with pytest.raises(ValueError, match="requires matrix_free=True"):
            ham.eigsolve(execution=execution)


@pytest.mark.parametrize("execution", ["numba-serial", "numba-parallel"])
def test_numba_eigsolve_records_execution_and_converges(execution):
    ham = _complete_problem()
    rng = np.random.default_rng(3204)
    values, vectors = ham.eigsolve(
        sparse=True,
        k=2,
        which="SA",
        tol=1e-10,
        ncv=8,
        v0=rng.normal(size=ham.sector.dimension),
        check_hermitian=False,
        matrix_free=True,
        execution=execution,
    )

    diagnostics = ham.solver_diagnostics
    assert diagnostics.execution == execution
    assert diagnostics.execution_resolved == execution
    assert diagnostics.matvec_calls > 0
    assert diagnostics.matvec_block_size is None
    assert diagnostics.chunks_per_matvec is None

    residuals = np.linalg.norm(
        ham.aslinearoperator(execution=execution) @ vectors - vectors * values,
        axis=0,
    )
    assert np.all(residuals < 1e-8)


@pytest.mark.parametrize(
    "n_modes, particles",
    [(1, 0), (1, 1), (8, 4), (64, 32), (64, 64)],
)
def test_numba_combinadic_rank_matches_complete_basis_order(n_modes, particles):
    from edinpy.fermion._execution import _complete_rank_tables
    from edinpy.fermion._numba_execution import _complete_rank_uint64

    basis = edf.FockBasis(n_modes, particles)
    contributions, popcounts = _complete_rank_tables(n_modes, particles)
    candidates = {0, max(0, basis.dimension - 1)}
    if basis.dimension <= 100:
        candidates.update(range(basis.dimension))
    elif basis.dimension > 2:
        candidates.add(basis.dimension // 2)

    for index in sorted(candidates):
        state = np.uint64(basis.state_at(index))
        rank = _complete_rank_uint64(state, contributions, popcounts)
        assert int(rank) == index


def test_numba_execution_plan_preserves_structure_and_is_reused():
    from edinpy.fermion._numba_execution import (
        _get_execution_plan,
        _pack_operator_plan,
    )

    modes = edf.FermionModes(edf.DoF(6))
    sector = edf.NParticleSector(modes, N=3).build()
    operator = (
        edf.Hopping(0, 1, -0.4, modes)
        + edf.Hopping(2, 3, 0.2 + 0.1j, modes)
        + edf.Hopping(0, 5, -0.3, modes)
    )
    ham = edf.Hamiltonian(operator, sector)

    operator_plan = _pack_operator_plan(ham._compiled)
    first = _get_execution_plan(ham._compiled, ham.sector.basis)
    second = _get_execution_plan(ham._compiled, ham.sector.basis)

    assert operator_plan.simple_hopping_transition_masks.size == 1
    assert operator_plan.parity_free_hopping_transition_masks.size == 1
    assert operator_plan.signed_hopping_transition_masks.size == 1
    assert first is second
    assert first.operator is operator_plan
    assert first.states is ham.sector.basis._execution_states()
