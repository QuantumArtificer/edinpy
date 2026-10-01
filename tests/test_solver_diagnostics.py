import numpy as np
import pytest

from edinpy import boson as edb
from edinpy import fermion as edf
from edinpy._core import MatrixFreeSolveDiagnostics


def _fermion_chain():
    modes = edf.FermionModes(edf.DoF(8, name="mode"))
    sector = edf.NParticleSector(modes, 4).build()
    H = 0
    for i in range(7):
        H += edf.Hopping(i, i + 1, -1.0, modes)
        H += edf.DensityDensity(i, i + 1, 0.2, modes)
    for i in range(8):
        H += edf.Onsite(i, 0.017 * i, modes)
    return edf.Hamiltonian(H, sector)


def _boson_chain():
    modes = edb.BosonModes(edb.DoF(5, name="mode"))
    sector = edb.NParticleSector(modes, 4).build()
    H = 0
    for i in range(4):
        H += edb.Hopping(i, i + 1, -1.0, modes)
        H += edb.DensityDensity(i, i + 1, 0.15, modes)
    for i in range(5):
        H += edb.Onsite(i, 0.023 * i, modes)
    return edb.Hamiltonian(H, sector)


@pytest.mark.parametrize("factory", [_fermion_chain, _boson_chain])
def test_matrix_free_eigsolve_records_solver_diagnostics(factory):
    ham = factory()
    rng = np.random.default_rng(20260930)
    v0 = rng.normal(size=ham.sector.dimension)

    ham.eigsolve(
        sparse=True,
        k=2,
        which="SA",
        tol=1e-10,
        ncv=8,
        v0=v0,
        check_hermitian=False,
        matrix_free=True,
    )

    diagnostics = ham.solver_diagnostics
    assert isinstance(diagnostics, MatrixFreeSolveDiagnostics)
    assert diagnostics.dimension == ham.sector.dimension
    assert diagnostics.solver_wall_seconds > 0.0
    assert diagnostics.matvec_calls > 0
    assert diagnostics.matvec_total_seconds > 0.0
    assert diagnostics.matvec_min_seconds > 0.0
    assert diagnostics.matvec_min_seconds <= diagnostics.matvec_mean_seconds
    assert diagnostics.matvec_mean_seconds <= diagnostics.matvec_max_seconds
    assert diagnostics.rmatvec_calls >= 0
    assert diagnostics.hamiltonian_apply_seconds >= diagnostics.matvec_total_seconds
    assert diagnostics.solver_overhead_seconds >= 0.0
    assert 0.0 <= diagnostics.hamiltonian_apply_fraction <= 1.0
    assert diagnostics.basis_states_processed == (
        ham.sector.dimension
        * (diagnostics.matvec_calls + diagnostics.rmatvec_calls)
    )
    assert diagnostics.basis_states_per_second > 0.0

    payload = diagnostics.as_dict()
    assert payload["dimension"] == ham.sector.dimension
    assert payload["matvec_calls"] == diagnostics.matvec_calls


def test_solver_diagnostics_are_reset_and_snapshotted_per_solve():
    ham = _fermion_chain()
    operator = ham.aslinearoperator()
    vector = np.ones(ham.sector.dimension)

    # This call must not leak into the subsequent eigensolve diagnostics.
    operator @ vector

    rng = np.random.default_rng(20260930)
    v0 = rng.normal(size=ham.sector.dimension)
    ham.eigsolve(
        sparse=True,
        k=2,
        which="SA",
        tol=1e-10,
        ncv=8,
        v0=v0,
        check_hermitian=False,
        matrix_free=True,
    )
    diagnostics = ham.solver_diagnostics
    payload = diagnostics.as_dict()

    # Post-solve residual-style applications mutate the live callback timer but
    # must not change the stored snapshot from the eigensolve.
    operator @ vector
    assert ham.solver_diagnostics is diagnostics
    assert ham.solver_diagnostics.as_dict() == payload

    ham.eigsolve(sparse=False, k=2, which="SA")
    assert ham.solver_diagnostics is None


def test_fermion_diagnostics_include_chunk_geometry():
    ham = _fermion_chain()
    rng = np.random.default_rng(7)
    ham.eigsolve(
        sparse=True,
        k=1,
        which="SA",
        tol=1e-9,
        ncv=6,
        v0=rng.normal(size=ham.sector.dimension),
        check_hermitian=False,
        matrix_free=True,
    )

    diagnostics = ham.solver_diagnostics
    assert diagnostics.matvec_block_size == ham.sector.dimension
    assert diagnostics.chunks_per_matvec == 1
    assert diagnostics.estimated_total_chunks == (
        diagnostics.matvec_calls + diagnostics.rmatvec_calls
    )


def test_boson_diagnostics_leave_chunk_geometry_unspecified():
    ham = _boson_chain()
    rng = np.random.default_rng(11)
    ham.eigsolve(
        sparse=True,
        k=1,
        which="SA",
        tol=1e-9,
        ncv=6,
        v0=rng.normal(size=ham.sector.dimension),
        check_hermitian=False,
        matrix_free=True,
    )

    diagnostics = ham.solver_diagnostics
    assert diagnostics.matvec_block_size is None
    assert diagnostics.chunks_per_matvec is None
    assert diagnostics.estimated_total_chunks is None
