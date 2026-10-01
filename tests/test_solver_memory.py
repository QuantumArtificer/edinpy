import numpy as np
import pytest

from edinpy import boson as edb
from edinpy import fermion as edf
from edinpy._core import (
    arpack_ncv_for_memory_budget,
    estimate_arpack_memory,
    ncv_for_total_memory,
    plan_solver_memory,
    resolve_arpack_ncv,
)


def test_resolve_arpack_ncv_matches_scipy_default_policy():
    assert resolve_arpack_ncv(100, 2) == 20
    assert resolve_arpack_ncv(10, 2) == 10
    assert resolve_arpack_ncv(100, 2, 7) == 7


@pytest.mark.parametrize("ncv", [0, 2, 101])
def test_resolve_arpack_ncv_rejects_invalid_subspace_sizes(ncv):
    with pytest.raises(ValueError):
        resolve_arpack_ncv(100, 2, ncv)


def test_real_arpack_memory_estimate_counts_dominant_arrays():
    estimate = estimate_arpack_memory(1000, np.float64, k=2, ncv=8)

    assert estimate.dtype == "float64"
    assert estimate.solver_path == "real_symmetric_arpack"
    assert estimate.vector_bytes == 8000
    assert estimate.lanczos_basis_bytes == 64000
    assert estimate.iteration_workspace_bytes == 97024
    assert estimate.returned_eigenvectors_bytes == 16000
    assert estimate.extraction_workspace_bytes == 80048
    assert estimate.estimated_arpack_bytes == 177072
    assert estimate.estimated_peak_bytes == 177072
    assert estimate.as_dict()["ncv"] == 8
    assert "estimated_arpack_bytes" in estimate.as_dict()
    assert "estimated_peak_bytes" not in estimate.as_dict()


def test_complex_arpack_memory_estimate_uses_complex_workspace_model():
    real = estimate_arpack_memory(1000, np.float64, k=2, ncv=8)
    complex_estimate = estimate_arpack_memory(
        1000,
        np.complex128,
        k=2,
        ncv=8,
    )

    assert complex_estimate.dtype == "complex128"
    assert complex_estimate.solver_path == "complex_nonsymmetric_arpack"
    assert complex_estimate.vector_bytes == 16000
    assert complex_estimate.estimated_arpack_bytes > real.estimated_arpack_bytes


def test_raw_arpack_budget_selects_largest_ncv_that_fits():
    budget = estimate_arpack_memory(
        1000,
        np.float64,
        k=2,
        ncv=8,
    ).estimated_arpack_bytes

    assert arpack_ncv_for_memory_budget(
        1000,
        np.float64,
        k=2,
        budget_bytes=budget,
    ) == 8


def test_raw_arpack_budget_rejects_budget_below_minimum_workspace():
    minimum = estimate_arpack_memory(
        1000,
        np.float64,
        k=2,
        ncv=3,
    ).estimated_arpack_bytes

    with pytest.raises(ValueError, match="too small"):
        arpack_ncv_for_memory_budget(
            1000,
            np.float64,
            k=2,
            budget_bytes=minimum - 1,
        )


def test_solver_memory_plan_applies_reserve_safety_and_basis_storage():
    plan = plan_solver_memory(
        1000,
        np.float64,
        total_memory_bytes=1_000_000,
        k=2,
        ncv=8,
        reserve_bytes=100_000,
        reserve_fraction=0.10,
        safety_factor=1.5,
        basis_execution_bytes=10_000,
    )

    assert plan.reserve_bytes == 200_000
    assert plan.usable_memory_bytes == 800_000
    assert plan.estimated_arpack_bytes == 177072
    assert plan.scaled_arpack_bytes == 265608
    assert plan.basis_execution_bytes == 10_000
    assert plan.planned_bytes == 275608
    assert plan.headroom_bytes == 524392
    assert plan.fits


def test_solver_memory_plan_reports_when_minimum_does_not_fit():
    minimum = plan_solver_memory(
        1000,
        np.float64,
        total_memory_bytes=90_000,
        k=2,
        ncv=3,
        reserve_fraction=0.0,
        safety_factor=1.0,
    )
    assert not minimum.fits

    with pytest.raises(ValueError, match="too small"):
        ncv_for_total_memory(
            1000,
            np.float64,
            total_memory_bytes=90_000,
            k=2,
            reserve_fraction=0.0,
            safety_factor=1.0,
        )


def test_total_memory_ncv_selector_respects_upper_bound():
    assert ncv_for_total_memory(
        1000,
        np.float64,
        total_memory_bytes=10_000_000,
        k=2,
        ncv=8,
        reserve_fraction=0.25,
        safety_factor=1.25,
        basis_execution_bytes=8000,
    ) == 8


def test_memory_planning_input_validation():
    with pytest.raises(ValueError, match="reserve_fraction"):
        plan_solver_memory(
            100,
            np.float64,
            total_memory_bytes=1_000_000,
            reserve_fraction=1.0,
        )
    with pytest.raises(ValueError, match="safety_factor"):
        plan_solver_memory(
            100,
            np.float64,
            total_memory_bytes=1_000_000,
            safety_factor=0.9,
        )


def test_hamiltonian_solver_memory_estimate_stays_matrix_free():
    modes = edb.BosonModes(edb.DoF(5))
    sector = edb.NParticleSector(modes, N=3).build()
    operator = sum(
        (edb.Hopping(i, i + 1, -1.0, modes) for i in range(4)),
        start=0,
    )
    ham = edb.Hamiltonian(operator, sector)

    estimate = ham.estimate_solver_memory(k=2, ncv=6)

    assert estimate.dimension == sector.dimension
    assert estimate.k == 2
    assert estimate.ncv == 6
    assert estimate.dtype == "float64"
    assert ham._matrix is None
    assert not sector.basis.is_materialized


def test_hamiltonian_memory_plan_includes_expected_boson_execution_storage():
    modes = edb.BosonModes(edb.DoF(5))
    sector = edb.NParticleSector(modes, N=3).build()
    ham = edb.Hamiltonian(edb.Onsite(0, 1.0, modes), sector)

    plan = ham.plan_solver_memory(
        1_000_000,
        k=2,
        ncv=5,
        reserve_fraction=0.1,
        safety_factor=1.2,
    )

    assert plan.basis_execution_bytes == sector.dimension * 8
    assert plan.fits
    assert ham._matrix is None
    assert not sector.basis.is_materialized


def test_hamiltonian_memory_plan_includes_expected_fermion_execution_storage():
    modes = edf.FermionModes(edf.DoF(20))
    sector = edf.NParticleSector(modes, N=10).build()
    ham = edf.Hamiltonian(edf.Onsite(0, 1.0, modes), sector)

    assert sector.basis.estimated_execution_storage_bytes == sector.dimension * 8
    plan = ham.plan_solver_memory(
        100_000_000,
        k=2,
        ncv=8,
        reserve_fraction=0.1,
    )
    assert plan.basis_execution_bytes == sector.dimension * 8
    assert not sector.basis.is_materialized


def test_hamiltonian_raw_and_total_budget_helpers_stay_matrix_free():
    modes = edb.BosonModes(edb.DoF(5))
    sector = edb.NParticleSector(modes, N=3).build()
    ham = edb.Hamiltonian(edb.Onsite(0, 1.0, modes), sector)
    raw_budget = ham.estimate_solver_memory(k=2, ncv=5).estimated_arpack_bytes

    assert ham.ncv_for_arpack_budget(raw_budget, k=2) == 5
    assert ham.ncv_for_memory_budget(
        10_000_000,
        k=2,
        ncv=5,
        reserve_fraction=0.25,
        safety_factor=1.25,
    ) == 5
    assert ham._matrix is None
    assert not sector.basis.is_materialized
