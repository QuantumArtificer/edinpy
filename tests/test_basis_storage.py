import numpy as np

from edinpy import boson as edb
from edinpy import fermion as edf


def test_complete_fermion_basis_starts_implicit_and_rank_unrank_round_trip():
    basis = edf.FockBasis(10, 5)

    assert basis.dimension == 252
    assert basis.is_complete
    assert not basis.is_materialized
    assert basis.storage_bytes == 0

    reference = edf.FockBasis._enumerate_states(10, 5)
    for index, state in enumerate(reference):
        assert basis.state_at(index) == state
        assert basis.index(state) == index

    assert not basis.is_materialized


def test_complete_fermion_execution_view_is_compact_without_public_tuple():
    basis = edf.FockBasis(16, 8)
    states = basis._execution_states()

    assert states.dtype == np.uint64
    assert states.shape == (12870,)
    assert basis.storage_bytes == states.nbytes
    assert basis._states is None
    assert basis.state_at(-1) == int(states[-1])


def test_projected_fermion_basis_uses_compact_uint64_storage():
    site = edf.DoF(4, name="site")
    spin = edf.DoF(2, name="spin", labels=("up", "down"))
    modes = edf.FermionModes(site, spin)
    basis = (
        edf.NParticleSector(modes, 4)
        .project_particles("spin", up=2, down=2)
        .build()
        .basis
    )

    assert not basis.is_complete
    assert basis._states is None
    assert basis._state_array.dtype == np.uint64
    assert basis.storage_bytes == basis.dimension * np.dtype(np.uint64).itemsize


def test_complete_fermion_above_64_modes_remains_implicit_for_rank_unrank():
    basis = edf.FockBasis(70, 2)
    probes = (0, 1, 17, basis.dimension - 1)

    for index in probes:
        state = basis.state_at(index)
        assert basis.index(state) == index

    assert not basis.is_materialized
    assert basis.storage_bytes == 0


def test_complete_boson_basis_starts_implicit_and_rank_unrank_round_trip():
    basis = edb.FockBasis(5, 4)

    assert basis.dimension == 70
    assert basis.is_complete
    assert not basis.is_materialized
    assert basis.storage_bytes == 0

    reference = edb.FockBasis._enumerate_packed_states(5, 4)
    for index, packed in enumerate(reference):
        state = basis.state_at(index)
        assert basis.pack(state.occupations) == packed
        assert basis.index(state) == index

    assert not basis.is_materialized


def test_complete_boson_execution_view_is_compact_without_public_tuple():
    basis = edb.FockBasis(10, 10)
    words = basis._uint64_words()

    assert words.dtype == np.uint64
    assert words.shape == (92378, 1)
    assert basis.storage_bytes == words.nbytes
    assert basis._packed_states is None
    assert basis.pack(basis.state_at(-1).occupations) == int(words[-1, 0])


def test_projected_boson_basis_uses_compact_multiword_storage():
    site = edb.DoF(17, name="site")
    species = edb.DoF(2, name="species", labels=("a", "b"))
    modes = edb.BosonModes(site, species)
    basis = (
        edb.NParticleSector(modes, 3)
        .project_particles("species", a=2, b=1)
        .build()
        .basis
    )

    assert not basis.is_complete
    assert basis._packed_states is None
    assert basis._packed_words.dtype == np.uint64
    assert basis._packed_words.shape[1] > 1
    assert basis.storage_bytes == basis._packed_words.nbytes


def test_matrix_free_action_does_not_materialize_public_state_tuples():
    fmodes = edf.FermionModes(edf.DoF(10))
    fsector = edf.NParticleSector(fmodes, 5).build()
    fham = edf.Hamiltonian(edf.Hopping(0, 1, -1.0, fmodes), fsector)
    fham.aslinearoperator() @ np.ones(fsector.dimension)
    assert fsector.basis._states is None
    assert fsector.basis._state_array is not None

    bmodes = edb.BosonModes(edb.DoF(6))
    bsector = edb.NParticleSector(bmodes, 5).build()
    bham = edb.Hamiltonian(edb.Hopping(0, 1, -1.0, bmodes), bsector)
    bham.aslinearoperator() @ np.ones(bsector.dimension)
    assert bsector.basis._packed_states is None
    assert bsector.basis._packed_words is not None


def test_projected_fermion_above_64_modes_uses_compact_multiword_storage():
    site = edf.DoF(35, name="site")
    spin = edf.DoF(2, name="spin", labels=("up", "down"))
    modes = edf.FermionModes(site, spin)
    basis = (
        edf.NParticleSector(modes, 2)
        .project_particles("spin", up=1, down=1)
        .build()
        .basis
    )

    assert not basis.is_complete
    assert basis._states is None
    assert basis._state_words.dtype == np.uint64
    assert basis._state_words.shape[1] == 2
    assert basis.storage_bytes == basis._state_words.nbytes
    for index in (0, basis.dimension // 2, basis.dimension - 1):
        state = basis.state_at(index)
        assert basis.index(state) == index
