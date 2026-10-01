import numpy as np

from edinpy import boson as edb
from edinpy import fermion as edf


def test_hamiltonian_shell_is_shared_between_statistics():
    assert edf.Hamiltonian is edb.Hamiltonian


def test_shared_hamiltonian_preserves_statistics_specific_backends():
    fmodes = edf.FermionModes(edf.DoF(2))
    fsector = edf.NParticleSector(fmodes, 1).build()
    fham = edf.Hamiltonian(edf.Hopping(0, 1, -1.0, fmodes), fsector)

    bmodes = edb.BosonModes(edb.DoF(2))
    bsector = edb.NParticleSector(bmodes, 1).build()
    bham = edb.Hamiltonian(edb.Hopping(0, 1, -1.0, bmodes), bsector)

    expected = np.array([[-0.0, -1.0], [-1.0, -0.0]])
    assert np.allclose(fham.toarray(), expected)
    assert np.allclose(bham.toarray(), expected)
    assert np.allclose(
        np.sort(np.linalg.eigvalsh(fham.toarray())),
        np.sort(np.linalg.eigvalsh(bham.toarray())),
    )


def test_shared_hamiltonian_keeps_compiler_diagnostics():
    fmodes = edf.FermionModes(edf.DoF(3))
    fsector = edf.NParticleSector(fmodes, 1).build()
    fham = edf.Hamiltonian(edf.Hopping(0, 1, -1.0, fmodes), fsector)

    bmodes = edb.BosonModes(edb.DoF(3))
    bsector = edb.NParticleSector(bmodes, 1).build()
    bham = edb.Hamiltonian(edb.Hopping(0, 1, -1.0, bmodes), bsector)

    assert fham.compiler_stats["hopping_groups"] == 1
    assert bham.compiler_stats["hopping_groups"] == 1


def test_shared_dirac_path_matches_explicit_matrix():
    for module, modes in (
        (edf, edf.FermionModes(edf.DoF(4))),
        (edb, edb.BosonModes(edb.DoF(3))),
    ):
        sector = module.NParticleSector(modes, 2).build()
        coefficients = np.arange(1, sector.dimension + 1, dtype=float)
        psi = sector.from_vector(coefficients).normalized()
        operator = module.Number(0, modes=modes)
        ham = module.Hamiltonian(operator, sector)
        expected = np.vdot(psi.coefficients, ham.matrix @ psi.coefficients)
        assert np.allclose(psi.dag * operator * psi, expected)
