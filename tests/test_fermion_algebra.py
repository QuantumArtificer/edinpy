import numpy as np
import pytest

from edinpy import fermion as edf


def make_model(neff=2, nf=1):
    """Initialize a simple fermionic model with `neff` effective orbitals."""
    edf.clear()
    edf.DoF(neff, name="orbital")
    return edf.model(nf)


def ket(state):
    """Construct a Fock state from its integer occupation representation."""
    if state == 0:
        return edf.vacuum()
    return edf.fockstate(state)


def components(result):
    """Return {state: amplitude} for an EDinPy state-like result."""
    if isinstance(result, edf.null):
        return {}

    if isinstance(result, edf.fockstate):
        return {result.state: complex(result.amp)}

    if isinstance(result, edf.statesum):
        out = {}
        for state in result.states:
            out[state.state] = out.get(state.state, 0.0j) + complex(state.amp)
        return {s: a for s, a in out.items() if not np.isclose(a, 0.0)}

    raise TypeError(f"Unsupported result type: {type(result)}")


def apply_product(*operators, state):
    """
    Apply O_1 O_2 ... O_n |state>, with operators supplied
    in the same left-to-right order.
    """
    result = ket(state)

    for op in reversed(operators):
        result = op * result

    return result


def add_results(*results):
    """Add state amplitudes from several operator-action results."""
    out = {}

    for result in results:
        for state, amp in components(result).items():
            out[state] = out.get(state, 0.0j) + amp

    return {s: a for s, a in out.items() if not np.isclose(a, 0.0)}


def assert_single_state(result, state, amplitude=1.0):
    result = components(result)

    assert set(result) == {state}
    assert np.isclose(result[state], amplitude)


def assert_zero(result):
    assert result == {}


def test_fixed_particle_number_basis_enumeration():
    model = make_model(neff=4, nf=2)

    assert model.Neff == 4
    assert model.Nbasis == 6

    # All four-bit states with exactly two occupied orbitals,
    # ordered by their integer representation.
    assert model.fockspace.ls == [3, 5, 6, 9, 10, 12]


def test_single_mode_creation_and_annihilation():
    make_model(neff=2, nf=1)

    c0 = edf.operator(0)
    cd0 = c0.dag

    assert isinstance(c0 * ket(0), edf.null)
    assert_single_state(cd0 * ket(0), 1)

    assert_single_state(c0 * ket(1), 0)
    assert isinstance(cd0 * ket(1), edf.null)


def test_number_operator():
    make_model(neff=2, nf=1)

    n0 = edf.number(0)

    assert_single_state(n0 * ket(1), 1)
    assert isinstance(n0 * ket(2), edf.null)


@pytest.mark.parametrize("state", [0, 1, 2, 3])
def test_same_mode_canonical_anticommutator(state):
    """
    {c_0, c_0^dagger} = 1.
    """
    make_model(neff=2, nf=1)

    c0 = edf.operator(0)
    cd0 = c0.dag

    result = add_results(
        apply_product(c0, cd0, state=state),
        apply_product(cd0, c0, state=state),
    )

    assert set(result) == {state}
    assert np.isclose(result[state], 1.0)


def test_annihilation_has_fermionic_parity():
    """
    For mode ordering 0,1:

        c_1 |11> = -|10>

    where the integer state representation is:
        |11> -> 3
        |10> -> 1

    because mode 0 is occupied and precedes mode 1.
    """
    make_model(neff=2, nf=1)

    c1 = edf.operator(1)

    assert_single_state(c1 * ket(3), 1, amplitude=-1.0)


def test_creation_has_fermionic_parity():
    """
    c_1^dagger |10> = -|11>.
    """
    make_model(neff=2, nf=1)

    cd1 = edf.operator(1).dag

    assert_single_state(cd1 * ket(1), 3, amplitude=-1.0)


def test_different_mode_annihilators_anticommute():
    """
    {c_0, c_1} = 0.
    """
    make_model(neff=2, nf=1)

    c0 = edf.operator(0)
    c1 = edf.operator(1)

    result = add_results(
        apply_product(c0, c1, state=3),
        apply_product(c1, c0, state=3),
    )

    assert_zero(result)


def test_different_mode_creation_annihilation_anticommute():
    """
    {c_0, c_1^dagger} = 0 for 0 != 1.
    """
    make_model(neff=2, nf=1)

    c0 = edf.operator(0)
    cd1 = edf.operator(1).dag

    result = add_results(
        apply_product(c0, cd1, state=1),
        apply_product(cd1, c0, state=1),
    )

    assert_zero(result)


def test_different_mode_creators_anticommute():
    """
    {c_0^dagger, c_1^dagger} = 0.
    """
    make_model(neff=2, nf=1)

    cd0 = edf.operator(0).dag
    cd1 = edf.operator(1).dag

    result = add_results(
        apply_product(cd0, cd1, state=0),
        apply_product(cd1, cd0, state=0),
    )

    assert_zero(result)


def test_exhaustive_canonical_anticommutation_relations():
    """
    Verify the canonical anticommutation relations on the complete local
    Fock space of four fermionic modes:

        {c_i, c_j} = 0
        {c_i^dagger, c_j^dagger} = 0
        {c_i, c_j^dagger} = delta_ij
    """
    neff = 4
    make_model(neff=neff, nf=2)

    annihilation = [edf.operator(i) for i in range(neff)]
    creation = [op.dag for op in annihilation]

    for state in range(1 << neff):
        for i in range(neff):
            for j in range(neff):
                cc = add_results(
                    apply_product(
                        annihilation[i],
                        annihilation[j],
                        state=state,
                    ),
                    apply_product(
                        annihilation[j],
                        annihilation[i],
                        state=state,
                    ),
                )
                assert_zero(cc)

                cdcd = add_results(
                    apply_product(
                        creation[i],
                        creation[j],
                        state=state,
                    ),
                    apply_product(
                        creation[j],
                        creation[i],
                        state=state,
                    ),
                )
                assert_zero(cdcd)

                ccd = add_results(
                    apply_product(
                        annihilation[i],
                        creation[j],
                        state=state,
                    ),
                    apply_product(
                        creation[j],
                        annihilation[i],
                        state=state,
                    ),
                )

                if i == j:
                    assert set(ccd) == {state}
                    assert np.isclose(ccd[state], 1.0)
                else:
                    assert_zero(ccd)


def test_number_operator_matches_creation_annihilation():
    """
    n_i = c_i^dagger c_i on every local Fock state.
    """
    neff = 4
    make_model(neff=neff, nf=2)

    for state in range(1 << neff):
        for i in range(neff):
            n_i = edf.number(i)
            c_i = edf.operator(i)

            direct = components(n_i * ket(state))
            composite = components(
                apply_product(c_i.dag, c_i, state=state)
            )

            assert direct == composite


def test_operator_class_hierarchy():
    make_model(neff=4, nf=2)

    c = edf.Annihilation(1)
    cd = edf.Creation(1)
    n = edf.Number(1)

    assert isinstance(c, edf.Operator)
    assert isinstance(cd, edf.Operator)
    assert isinstance(n, edf.Operator)

    product = cd * c
    operator_sum = c + cd

    assert isinstance(product, edf.Operator)
    assert isinstance(operator_sum, edf.Operator)


def test_legacy_operator_api_aliases():
    make_model(neff=4, nf=2)

    assert edf.operator is edf.Annihilation
    assert edf.dagger is edf.Creation
    assert edf.number is edf.Number

    legacy_c = edf.operator(1)
    modern_c = edf.Annihilation(1)

    assert type(legacy_c) is type(modern_c)
    assert legacy_c.site == modern_c.site

    legacy_cd = legacy_c.dag
    modern_cd = edf.Creation(1)

    assert type(legacy_cd) is type(modern_cd)
    assert legacy_cd.site == modern_cd.site


def test_number_is_not_annihilation_operator():
    make_model(neff=4, nf=2)

    n = edf.Number(1)

    assert isinstance(n, edf.Operator)
    assert not isinstance(n, edf.Annihilation)


def test_creation_is_not_annihilation_operator():
    make_model(neff=4, nf=2)

    cd = edf.Creation(1)

    assert isinstance(cd, edf.Operator)
    assert not isinstance(cd, edf.Annihilation)
