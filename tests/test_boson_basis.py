from math import comb

import pytest

from edinpy import boson as edb


def test_fixed_n_basis_dimension_and_states():
    basis = edb.FockBasis(n_modes=3, N=2)

    assert basis.dimension == comb(4, 2) == 6
    assert basis.states == (
        (2, 0, 0),
        (1, 1, 0),
        (0, 2, 0),
        (1, 0, 1),
        (0, 1, 1),
        (0, 0, 2),
    )
    assert all(sum(state) == 2 for state in basis.states)


def test_basis_pack_unpack_and_index_are_exact():
    basis = edb.FockBasis(n_modes=4, N=5)
    state = (0, 2, 1, 2)
    packed = basis.pack(state)

    assert basis.unpack(packed) == state
    assert basis.index(state) == basis.index(packed)
    assert state in basis
    assert packed in basis


def test_packed_representation_has_no_64_bit_ceiling():
    basis = edb.FockBasis(n_modes=33, N=3)

    assert basis.packed_width == 66
    packed = basis.pack((0,) * 32 + (3,))
    assert packed.bit_length() > 64
    assert basis.unpack(packed) == (0,) * 32 + (3,)


def test_zero_particle_basis_contains_only_vacuum():
    basis = edb.FockBasis(n_modes=5, N=0)
    assert basis.dimension == 1
    assert basis.states == ((0, 0, 0, 0, 0),)


def test_fock_state_is_readable_and_validated():
    state = edb.FockState((2, 0, 1), amp=2)
    assert state.N == 3
    assert state.occupation(2) == 1
    assert str(state) == "2 |2,0,1>"

    with pytest.raises(ValueError, match="non-negative"):
        edb.FockState((1, -1))


def test_fock_basis_state_returns_indexed_ket():
    basis = edb.FockBasis(n_modes=3, N=2)
    state = basis.state(2)
    assert isinstance(state, edb.FockState)
    assert state.occupations == basis.states[2]
    assert state.index == 2


def test_fock_state_norm_and_normalization():
    state = edb.FockState((2, 1), amp=3 + 4j)
    assert state.norm() == 5
    normalized = state.normalized()
    assert normalized.norm() == pytest.approx(1)
    assert normalized.occupations == state.occupations


def test_complete_composition_ranks_match_complete_and_projected_order():
    modes = edb.BosonModes(
        edb.DoF(3, name="site"),
        edb.DoF(2, name="species", labels=("a", "b")),
    )
    complete = edb.NParticleSector(modes, N=4).build().basis
    assert tuple(complete._complete_composition_ranks()) == tuple(
        range(complete.dimension)
    )

    projected = (
        edb.NParticleSector(modes, N=4)
        .project_particles("species", a=2, b=2)
        .build()
        .basis
    )
    ranks = projected._complete_composition_ranks()
    assert ranks is projected._complete_composition_ranks()
    assert not ranks.flags.writeable
    assert all(left < right for left, right in zip(ranks, ranks[1:]))
    for row, packed in enumerate(projected.packed_states):
        assert complete.packed_states[ranks[row]] == packed
