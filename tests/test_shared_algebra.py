"""Shared symbolic-expression infrastructure for fermions and bosons."""

import math

from edinpy import boson as edb
from edinpy import fermion as edf
from edinpy._core import operator_domains


def test_fermion_and_boson_reuse_shared_expression_node_types():
    assert edf.Operator is edb.Operator
    assert edf.OperatorSum is edb.OperatorSum
    assert edf.OperatorProduct is edb.OperatorProduct
    assert edf.set_notation is edb.set_notation


def test_statistics_specific_primitives_remain_distinct():
    assert edf.Annihilation is not edb.Annihilation
    assert edf.Creation is not edb.Creation
    assert edf.Number is not edb.Number


def test_shared_ast_tracks_fermion_operator_domain():
    modes = edf.FermionModes(edf.DoF(3))
    expression = edf.Creation(2, modes=modes) * edf.Annihilation(0, modes=modes)

    assert operator_domains(expression) == frozenset((modes,))


def test_shared_ast_tracks_boson_operator_domain():
    modes = edb.BosonModes(edb.DoF(3))
    expression = edb.Creation(2, modes=modes) * edb.Annihilation(0, modes=modes)

    assert operator_domains(expression) == frozenset((modes,))


def test_shared_expression_tree_preserves_fermionic_literal_action():
    modes = edf.FermionModes(edf.DoF(2))
    c = edf.set_notation(edf.Annihilation, modes)
    cd = edf.set_notation(edf.Creation, modes)
    state = edf.FockState(0b01, n_modes=2)

    result = cd(1) * c(0) * state

    assert result.state == 0b10
    assert result.amp == 1


def test_shared_expression_tree_preserves_bosonic_ladder_action():
    modes = edb.BosonModes(edb.DoF(2))
    b = edb.set_notation(edb.Annihilation, modes)
    bd = edb.set_notation(edb.Creation, modes)
    state = edb.FockState((2, 1))

    result = bd(1) * b(0) * state

    assert result.occupations == (1, 2)
    assert math.isclose(result.amp, 2.0)
