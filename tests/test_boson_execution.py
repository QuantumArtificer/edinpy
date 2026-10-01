import numpy as np
import pytest

from edinpy import boson as edb
from edinpy.boson._execution import compile_operator


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


def test_compiler_selects_number_hopping_and_monomial_kernels():
    modes = edb.BosonModes(edb.DoF(2))
    b = edb.set_notation(edb.Annihilation, modes)
    bd = edb.set_notation(edb.Creation, modes)
    n = edb.set_notation(edb.Number, modes)

    operator = (
        2 * n(0) * n(1)
        - 3 * bd(0) * b(1)
        - 3 * bd(1) * b(0)
        + 0.5 * bd(0) * bd(0) * b(1) * b(1)
        + b(0)
    )
    compiled = compile_operator(operator)

    assert compiled.stats == {
        "number_product": 1,
        "hopping_groups": 1,
        "hopping_terms": 2,
        "pair_hopping_groups": 0,
        "pair_hopping_terms": 0,
        "monomial": 2,
        "generic": 0,
        "projected_out": 1,
    }
    assert compiled.fully_lowered


def test_compiled_action_matches_literal_symbolic_action():
    modes = edb.BosonModes(edb.DoF(2))
    b = edb.set_notation(edb.Annihilation, modes)
    bd = edb.set_notation(edb.Creation, modes)
    n = edb.set_notation(edb.Number, modes)
    operator = (
        1.25 * n(0) * n(1)
        + (2 - 0.5j) * bd(0) * b(1)
        + 0.75 * bd(0) * bd(0) * b(1) * b(1)
        + 0.4 * b(0)
    )
    compiled = compile_operator(operator)

    for occupations in ((0, 0), (1, 0), (0, 2), (2, 1), (1, 3)):
        ket = edb.FockState(occupations, amp=1.2 - 0.3j)
        assert_maps_close(as_map(compiled.apply(ket)), as_map(operator * ket))


def test_general_monomial_preserves_non_normal_ordered_bosonic_action():
    modes = edb.BosonModes(edb.DoF(1))
    b = edb.set_notation(edb.Annihilation, modes)
    bd = edb.set_notation(edb.Creation, modes)
    operator = b(0) * bd(0)
    compiled = compile_operator(operator)

    assert compiled.stats["monomial"] == 1
    for occupation in range(6):
        ket = edb.FockState((occupation,))
        assert_maps_close(as_map(compiled.apply(ket)), as_map(operator * ket))
        assert_maps_close(as_map(compiled.apply(ket)), {(occupation,): occupation + 1})


def test_compiled_creation_can_exceed_original_particle_number_field_width():
    modes = edb.BosonModes(edb.DoF(1))
    bd = edb.set_notation(edb.Creation, modes)
    compiled = compile_operator(bd(0))
    ket = edb.FockState((3,))

    result = compiled.apply(ket)
    assert result.occupations == (4,)
    assert result.amp == pytest.approx(2.0)


@pytest.mark.parametrize("execution", ("numba-serial", "numba-parallel"))
def test_numba_execution_matches_numpy_for_lowered_operator(execution):
    pytest.importorskip("numba")
    modes = edb.BosonModes(edb.DoF(5))
    sector = edb.NParticleSector(modes, N=4).build()
    b = edb.set_notation(edb.Annihilation, modes)
    bd = edb.set_notation(edb.Creation, modes)
    n = edb.set_notation(edb.Number, modes)
    operator = (
        edb.Hopping(0, 4, -0.7 + 0.2j, modes)
        + edb.PairHopping(1, 3, 0.25, modes)
        + 0.3 * n(0) * n(2)
        + 0.2 * bd(0) * bd(1) * b(4) * b(3)
    )
    hamiltonian = edb.Hamiltonian(operator, sector)
    vector = np.linspace(-0.8, 0.9, sector.dimension) + 0.1j

    reference = hamiltonian.aslinearoperator(execution="numpy")
    linear = hamiltonian.aslinearoperator(execution=execution)
    expected = reference @ vector
    actual = linear @ vector

    assert np.allclose(actual, expected, atol=1e-12)
    assert np.allclose(linear.rmatvec(vector), reference.rmatvec(vector), atol=1e-12)
    assert hamiltonian._resolve_execution(execution) == execution


def test_numba_execution_supports_projected_boson_basis():
    pytest.importorskip("numba")
    site = edb.DoF(3, name="site")
    species = edb.DoF(2, name="species", labels=("a", "b"))
    modes = edb.BosonModes(site, species)
    sector = (
        edb.NParticleSector(modes, N=3)
        .project_particles("species", a=2, b=1)
        .build()
    )
    operator = (
        edb.Hopping((0, 0), (1, 0), -0.6, modes)
        + edb.Hopping((1, 1), (2, 1), -0.4, modes)
        + edb.DensityDensity((1, 0), (1, 1), 0.35, modes)
    )
    hamiltonian = edb.Hamiltonian(operator, sector)
    vector = np.linspace(-0.5, 0.75, sector.dimension)

    expected = hamiltonian.aslinearoperator(execution="numpy") @ vector
    serial = hamiltonian.aslinearoperator(execution="numba-serial") @ vector
    parallel = hamiltonian.aslinearoperator(execution="numba-parallel") @ vector

    assert np.allclose(serial, expected, atol=1e-12)
    assert np.allclose(parallel, expected, atol=1e-12)


def test_mixed_keeps_boson_numpy_execution_conservative():
    pytest.importorskip("numba")
    modes = edb.BosonModes(edb.DoF(4))
    sector = edb.NParticleSector(modes, N=3).build()
    hamiltonian = edb.Hamiltonian(edb.Hopping(0, 3, -0.5, modes), sector)

    assert hamiltonian._resolve_execution("mixed") == "numpy"


def test_numba_execution_handles_multiword_boson_basis():
    pytest.importorskip("numba")
    modes = edb.BosonModes(edb.DoF(70))
    sector = edb.NParticleSector(modes, N=1).build()
    operator = edb.Hopping(0, 69, -0.8, modes) + edb.Onsite(65, 0.4, modes)
    hamiltonian = edb.Hamiltonian(operator, sector)
    vector = np.linspace(-1.0, 1.0, sector.dimension)

    expected = hamiltonian.aslinearoperator(execution="numpy") @ vector
    actual = hamiltonian.aslinearoperator(execution="numba-serial") @ vector

    assert np.allclose(actual, expected, atol=1e-12)


def test_numba_execution_plan_is_reused_for_repeated_matvecs():
    pytest.importorskip("numba")
    modes = edb.BosonModes(edb.DoF(4))
    sector = edb.NParticleSector(modes, N=3).build()
    hamiltonian = edb.Hamiltonian(edb.Hopping(0, 3, -0.5, modes), sector)
    vector = np.ones(sector.dimension)

    hamiltonian.aslinearoperator(execution="numba-serial") @ vector
    plans = hamiltonian._compiled._numba_execution_plans
    assert len(plans) == 1
    first_plan = next(iter(plans.values()))

    hamiltonian.aslinearoperator(execution="numba-serial") @ vector
    assert next(iter(plans.values())) is first_plan
