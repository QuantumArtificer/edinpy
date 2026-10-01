import numpy as np

from edinpy import boson as edb
from edinpy import fermion as edf


def test_dirac_state_layer_is_shared_but_elementary_fock_states_are_not():
    assert edf.NullState is edb.NullState
    assert edf.StateSum is edb.StateSum
    assert edf.FockVector is edb.FockVector
    assert edf.FockBra is edb.FockBra
    assert edf.FockState is not edb.FockState


def test_shared_vector_arithmetic_preserves_fermion_and_boson_sectors():
    fmodes = edf.FermionModes(edf.DoF(4))
    fsector = edf.NParticleSector(fmodes, 2).build()
    fpsi = fsector.from_vector(np.ones(fsector.dimension))
    assert isinstance(fpsi, edf.FockVector)
    assert np.allclose((2 * fpsi).coefficients, 2)

    bmodes = edb.BosonModes(edb.DoF(3))
    bsector = edb.NParticleSector(bmodes, 2).build()
    bpsi = bsector.from_vector(np.ones(bsector.dimension))
    assert isinstance(bpsi, edb.FockVector)
    assert np.allclose((2 * bpsi).coefficients, 2)


def test_shared_dirac_layer_keeps_statistics_specific_ladder_action():
    fmodes = edf.FermionModes(edf.DoF(2))
    fstate = edf.FockState(0b01, n_modes=2)
    assert edf.Creation(1, modes=fmodes) * fstate == edf.FockState(
        0b11, amp=-1, n_modes=2
    )

    bmodes = edb.BosonModes(edb.DoF(2))
    bstate = edb.FockState((2, 0))
    result = edb.Annihilation(0, modes=bmodes) * bstate
    assert result.occupations == (1, 0)
    assert np.isclose(result.amp, np.sqrt(2))


def test_basis_backed_dirac_products_still_match_explicit_matrices():
    for module, modes in (
        (edf, edf.FermionModes(edf.DoF(4))),
        (edb, edb.BosonModes(edb.DoF(3))),
    ):
        sector = module.NParticleSector(modes, 2).build()
        coefficients = np.arange(1, sector.dimension + 1, dtype=float)
        psi = sector.from_vector(coefficients).normalized()
        operator = module.Number(0, modes=modes)
        matrix = module.Hamiltonian(operator, sector).matrix
        expected = np.vdot(psi.coefficients, matrix @ psi.coefficients)
        assert np.allclose(psi.dag * operator * psi, expected)
