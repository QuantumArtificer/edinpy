import numpy as np
import pytest

from edinpy import fermion as edf


def as_map(result):
    if isinstance(result, edf.NullState):
        return {}
    if isinstance(result, edf.FockState):
        return {result.state: result.amp}
    if isinstance(result, edf.StateSum):
        out = {}
        for state in result.states:
            out[state.state] = out.get(state.state, 0) + state.amp
        return {state: amp for state, amp in out.items() if amp != 0}
    raise TypeError(type(result))


def add_maps(*maps):
    out = {}
    for mapping in maps:
        for state, amp in mapping.items():
            out[state] = out.get(state, 0) + amp
    return {state: amp for state, amp in out.items() if amp != 0}


def test_set_notation_leaves_symbol_choice_to_user():
    modes = edf.FermionModes(edf.DoF(4, "site"), edf.DoF(2, "spin"))
    psi = edf.set_notation(edf.Annihilation, modes)
    assert psi(2, 1).mode == modes.resolve((2, 1))
    assert psi.operator_type is edf.Annihilation
    assert psi.modes is modes


def test_set_notation_accepts_tuple_input():
    modes = edf.FermionModes(edf.DoF(3), edf.DoF(2))
    c = edf.set_notation(edf.Annihilation, modes)
    assert c((2, 1)).mode == c(2, 1).mode


def test_bound_notation_rejects_flat_mode_for_multidof_modes():
    modes = edf.FermionModes(edf.DoF(3), edf.DoF(2))
    c = edf.set_notation(edf.Annihilation, modes)
    with pytest.raises(ValueError):
        c(4)


def test_primitive_constructor_allows_explicit_flat_mode():
    modes = edf.FermionModes(edf.DoF(3), edf.DoF(2))
    assert edf.Annihilation(4, modes=modes).mode == 4


@pytest.mark.parametrize("n_modes", [1, 2, 3, 4])
def test_canonical_anticommutation_relations(n_modes):
    modes = edf.FermionModes(edf.DoF(n_modes))
    c = edf.set_notation(edf.Annihilation, modes)
    cd = edf.set_notation(edf.Creation, modes)

    for state_int in range(1 << n_modes):
        ket = edf.FockState(state_int, n_modes=n_modes)
        for i in range(n_modes):
            for j in range(n_modes):
                cc = add_maps(
                    as_map(c(i) * (c(j) * ket)),
                    as_map(c(j) * (c(i) * ket)),
                )
                dd = add_maps(
                    as_map(cd(i) * (cd(j) * ket)),
                    as_map(cd(j) * (cd(i) * ket)),
                )
                mixed = add_maps(
                    as_map(c(i) * (cd(j) * ket)),
                    as_map(cd(j) * (c(i) * ket)),
                )
                assert cc == {}
                assert dd == {}
                expected = {state_int: 1} if i == j else {}
                assert mixed == expected


def test_number_operator_matches_creation_annihilation():
    modes = edf.FermionModes(edf.DoF(5))
    c = edf.set_notation(edf.Annihilation, modes)
    cd = edf.set_notation(edf.Creation, modes)
    n = edf.set_notation(edf.Number, modes)
    for state_int in range(1 << 5):
        ket = edf.FockState(state_int, n_modes=5)
        for i in range(5):
            assert as_map(n(i) * ket) == as_map((cd(i) * c(i)) * ket)


def test_unary_negation_is_valid_symbolic_algebra():
    modes = edf.FermionModes(edf.DoF(2))
    c = edf.set_notation(edf.Annihilation, modes)
    ket = edf.FockState(0b01, n_modes=2)
    result = (-c(0)) * ket
    assert as_map(result) == {0: -1}


def test_state_sum_scalar_multiplication():
    state_sum = edf.FockState(1, n_modes=2) + edf.FockState(2, amp=2, n_modes=2)
    result = 3 * state_sum
    assert as_map(result) == {1: 3, 2: 6}


@pytest.mark.parametrize("value", [np.float64(2.0), np.complex128(1 + 2j), 3, 4.5])
def test_numpy_and_python_numeric_scalars(value):
    modes = edf.FermionModes(edf.DoF(2))
    n = edf.set_notation(edf.Number, modes)
    expression = value * n(0)
    result = expression * edf.FockState(1, n_modes=2)
    assert as_map(result) == {1: value}


def test_product_adjoint_reverses_order():
    modes = edf.FermionModes(edf.DoF(3))
    c = edf.set_notation(edf.Annihilation, modes)
    cd = edf.set_notation(edf.Creation, modes)
    expression = cd(0) * c(2)
    adjoint = expression.dag
    for state_int in range(8):
        ket = edf.FockState(state_int, n_modes=3)
        assert as_map(adjoint * ket) == as_map((cd(2) * c(0)) * ket)
