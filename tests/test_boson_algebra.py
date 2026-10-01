import itertools

import numpy as np
import pytest

from edinpy import boson as edb


def as_map(result):
    if isinstance(result, edb.NullState):
        return {}
    if isinstance(result, edb.FockState):
        return {result.occupations: result.amp}
    if isinstance(result, edb.StateSum):
        out = {}
        for state in result.states:
            out[state.occupations] = out.get(state.occupations, 0) + state.amp
        return {state: amp for state, amp in out.items() if amp != 0}
    raise TypeError(type(result))


def assert_maps_close(actual, expected, atol=1e-12):
    assert set(actual) == set(expected)
    for key in actual:
        assert actual[key] == pytest.approx(expected[key], abs=atol)


def add_maps(*maps, scales=None):
    if scales is None:
        scales = (1,) * len(maps)
    out = {}
    for scale, mapping in zip(scales, maps):
        for state, amp in mapping.items():
            out[state] = out.get(state, 0) + scale * amp
    return {state: amp for state, amp in out.items() if abs(amp) > 1e-14}


def occupation_vectors(n_modes, max_occupation=2):
    yield from itertools.product(range(max_occupation + 1), repeat=n_modes)


def test_set_notation_leaves_symbol_choice_to_user():
    modes = edb.BosonModes(edb.DoF(4, "site"), edb.DoF(2, "species"))
    beta = edb.set_notation(edb.Annihilation, modes)
    assert beta(2, 1).mode == modes.resolve((2, 1))
    assert beta.operator_type is edb.Annihilation
    assert beta.modes is modes


def test_set_notation_accepts_tuple_input():
    modes = edb.BosonModes(edb.DoF(3), edb.DoF(2))
    b = edb.set_notation(edb.Annihilation, modes)
    assert b((2, 1)).mode == b(2, 1).mode


def test_bound_notation_rejects_flat_mode_for_multidof_modes():
    modes = edb.BosonModes(edb.DoF(3), edb.DoF(2))
    b = edb.set_notation(edb.Annihilation, modes)
    with pytest.raises(ValueError):
        b(4)


def test_primitive_constructor_allows_explicit_flat_mode():
    modes = edb.BosonModes(edb.DoF(3), edb.DoF(2))
    assert edb.Annihilation(4, modes=modes).mode == 4


def test_annihilation_and_creation_have_bosonic_ladder_factors():
    modes = edb.BosonModes(edb.DoF(3))
    b = edb.set_notation(edb.Annihilation, modes)
    bd = edb.set_notation(edb.Creation, modes)
    ket = edb.FockState((2, 0, 3), amp=2)

    assert_maps_close(as_map(b(0) * ket), {(1, 0, 3): 2 * np.sqrt(2)})
    assert_maps_close(as_map(bd(2) * ket), {(2, 0, 4): 4})
    assert as_map(b(1) * ket) == {}


@pytest.mark.parametrize("n_modes", [1, 2, 3])
def test_canonical_commutation_relations(n_modes):
    modes = edb.BosonModes(edb.DoF(n_modes))
    b = edb.set_notation(edb.Annihilation, modes)
    bd = edb.set_notation(edb.Creation, modes)

    for occupations in occupation_vectors(n_modes, max_occupation=2):
        ket = edb.FockState(occupations)
        for i in range(n_modes):
            for j in range(n_modes):
                bb = add_maps(
                    as_map(b(i) * (b(j) * ket)),
                    as_map(b(j) * (b(i) * ket)),
                    scales=(1, -1),
                )
                dd = add_maps(
                    as_map(bd(i) * (bd(j) * ket)),
                    as_map(bd(j) * (bd(i) * ket)),
                    scales=(1, -1),
                )
                mixed = add_maps(
                    as_map(b(i) * (bd(j) * ket)),
                    as_map(bd(j) * (b(i) * ket)),
                    scales=(1, -1),
                )
                assert_maps_close(bb, {})
                assert_maps_close(dd, {})
                expected = {occupations: 1} if i == j else {}
                assert_maps_close(mixed, expected)


def test_number_operator_matches_creation_annihilation():
    modes = edb.BosonModes(edb.DoF(4))
    b = edb.set_notation(edb.Annihilation, modes)
    bd = edb.set_notation(edb.Creation, modes)
    n = edb.set_notation(edb.Number, modes)

    for occupations in occupation_vectors(4, max_occupation=2):
        ket = edb.FockState(occupations)
        for i in range(4):
            assert_maps_close(
                as_map(n(i) * ket),
                as_map((bd(i) * b(i)) * ket),
            )


def test_number_operator_multiplies_by_occupation():
    modes = edb.BosonModes(edb.DoF(3))
    n = edb.set_notation(edb.Number, modes)
    ket = edb.FockState((4, 0, 2), amp=3)

    assert as_map(n(0) * ket) == {(4, 0, 2): 12}
    assert as_map(n(1) * ket) == {}
    assert as_map(n(2) * ket) == {(4, 0, 2): 6}


def test_unary_negation_and_numeric_scalars():
    modes = edb.BosonModes(edb.DoF(2))
    b = edb.set_notation(edb.Annihilation, modes)
    ket = edb.FockState((1, 0))

    assert as_map((-b(0)) * ket) == {(0, 0): -1}
    result = np.complex128(1 + 2j) * b(0) * ket
    assert as_map(result) == {(0, 0): 1 + 2j}


def test_operator_sum_distributes_over_state_action():
    modes = edb.BosonModes(edb.DoF(2))
    b = edb.set_notation(edb.Annihilation, modes)
    ket = edb.FockState((1, 2))

    result = (b(0) + 2 * b(1)) * ket
    assert_maps_close(
        as_map(result),
        {(0, 2): 1, (1, 1): 2 * np.sqrt(2)},
    )


def test_product_adjoint_reverses_order():
    modes = edb.BosonModes(edb.DoF(3))
    b = edb.set_notation(edb.Annihilation, modes)
    bd = edb.set_notation(edb.Creation, modes)
    expression = bd(0) * b(2)
    adjoint = expression.dag

    for occupations in occupation_vectors(3, max_occupation=2):
        ket = edb.FockState(occupations)
        assert_maps_close(
            as_map(adjoint * ket),
            as_map((bd(2) * b(0)) * ket),
        )


def test_operator_expression_rejects_mixed_mode_owners():
    left = edb.BosonModes(edb.DoF(2))
    right = edb.BosonModes(edb.DoF(2))
    b_left = edb.set_notation(edb.Annihilation, left)
    b_right = edb.set_notation(edb.Annihilation, right)

    expression = b_left(0) + b_right(0)
    with pytest.raises(ValueError, match="different BosonModes"):
        from edinpy.boson._algebra import operator_modes

        operator_modes(expression)


def test_fock_state_sum_combines_duplicate_occupations():
    result = edb.FockState((1, 2), amp=2) + edb.FockState((1, 2), amp=-0.5)
    assert isinstance(result, edb.FockState)
    assert result.occupations == (1, 2)
    assert result.amp == 1.5


def test_state_sum_scalar_multiplication():
    state_sum = edb.FockState((1, 0)) + edb.FockState((0, 1), amp=2)
    result = 3 * state_sum
    assert as_map(result) == {(1, 0): 3, (0, 1): 6}


def test_bra_ket_inner_products_use_complex_conjugation():
    left = edb.FockState((1, 0), amp=1 + 2j)
    right = edb.FockState((1, 0), amp=3 - 1j)
    orthogonal = edb.FockState((0, 1), amp=7)

    assert left.dag * right == pytest.approx(np.conjugate(1 + 2j) * (3 - 1j))
    assert left.dag * orthogonal == 0


def test_bra_operator_ket_literal_matrix_element():
    modes = edb.BosonModes(edb.DoF(2))
    b = edb.set_notation(edb.Annihilation, modes)
    bd = edb.set_notation(edb.Creation, modes)
    bra = edb.FockState((2, 0))
    ket = edb.FockState((1, 1))

    element = bra.dag * bd(0) * b(1) * ket
    assert element == pytest.approx(np.sqrt(2))


def test_fock_vector_roundtrip_and_inner_product():
    modes = edb.BosonModes(edb.DoF(3))
    sector = edb.NParticleSector(modes, N=2).build()
    coefficients = np.arange(1, sector.dimension + 1, dtype=float)
    psi = sector.from_vector(coefficients)

    assert isinstance(psi, edb.FockVector)
    assert np.array_equal(psi.to_vector(), coefficients)
    assert psi.inner(psi) == pytest.approx(np.vdot(coefficients, coefficients))
    assert psi.normalized().norm() == pytest.approx(1.0)
    assert not psi.coefficients.flags.writeable


def test_number_changing_operator_on_fock_vector_is_not_projected_back():
    modes = edb.BosonModes(edb.DoF(2))
    sector = edb.NParticleSector(modes, N=1).build()
    b = edb.set_notation(edb.Annihilation, modes)
    psi = sector.from_vector([1, 0])

    result = b(0) * psi
    assert as_map(result) == {(0, 0): 1}


def test_fock_vector_operator_requires_same_modes_object():
    modes = edb.BosonModes(edb.DoF(2))
    other_modes = edb.BosonModes(edb.DoF(2))
    sector = edb.NParticleSector(modes, N=1).build()
    psi = sector.from_vector([1, 0])
    b = edb.set_notation(edb.Annihilation, other_modes)

    with pytest.raises(ValueError, match="different BosonModes"):
        b(0) * psi
