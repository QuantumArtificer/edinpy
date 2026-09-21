"""Common number-conserving fermionic operators expressed in literal Fock algebra."""

from __future__ import annotations

from ._algebra import Annihilation, Creation, Number, OperatorProduct, OperatorSum


def Hopping(i, j, t, modes):
    r"""Return a Hermitian one-body hopping term.

    Parameters
    ----------
    i, j : int or iterable of int
        Mode labels for the two fermionic modes.
    t : numbers.Number
        Matrix element multiplying :math:`c_i^\dagger c_j`. The reverse
        process uses ``conj(t)``. A conventional tight-binding term
        :math:`-t_0(c_i^\dagger c_j+\mathrm{H.c.})` is obtained by passing
        ``t=-t0`` for real ``t0``.
    modes : FermionModes
        Fermionic modes used to resolve ``i`` and ``j``.

    Returns
    -------
    OperatorSum
        Literal symbolic expression
        :math:`t c_i^\dagger c_j+t^* c_j^\dagger c_i`.
    """
    t_conj = t.conjugate() if hasattr(t, "conjugate") else t
    return OperatorSum((
        t * Creation(i, modes=modes) * Annihilation(j, modes=modes),
        t_conj * Creation(j, modes=modes) * Annihilation(i, modes=modes),
    ))


def Onsite(i, epsilon, modes):
    r"""Return an onsite one-body energy :math:`\epsilon_i n_i`.

    Parameters
    ----------
    i : int or iterable of int
        Mode label.
    epsilon : numbers.Number
        Onsite energy.
    modes : FermionModes
        Fermionic modes used to resolve ``i``.

    Returns
    -------
    OperatorProduct
        Literal symbolic expression :math:`\epsilon_i n_i`.
    """
    return epsilon * Number(i, modes=modes)


def DensityDensity(i, j, V, modes):
    r"""Return a density-density interaction :math:`V_{ij}n_i n_j`.

    Parameters
    ----------
    i, j : int or iterable of int
        Mode labels.
    V : numbers.Number
        Density-density interaction strength.
    modes : FermionModes
        Fermionic modes used to resolve ``i`` and ``j``.

    Returns
    -------
    OperatorProduct
        Literal symbolic density-density interaction.
    """
    return V * Number(i, modes=modes) * Number(j, modes=modes)


def Hubbard(up, down, U, modes):
    r"""Return the local Hubbard interaction :math:`U n_\uparrow n_\downarrow`.

    Parameters
    ----------
    up, down : int or iterable of int
        Mode labels of the two local fermionic components.
    U : numbers.Number
        Local interaction strength.
    modes : FermionModes
        Fermionic modes used to resolve the labels.

    Returns
    -------
    OperatorProduct
        Literal symbolic Hubbard interaction.
    """
    return DensityDensity(up, down, U, modes)


def SpinPlus(up, down, modes):
    r"""Return :math:`S^+=c_\uparrow^\dagger c_\downarrow` for two modes.

    Parameters
    ----------
    up, down : int or iterable of int
        Mode labels identified with the up and down components.
    modes : FermionModes
        Fermionic modes used to resolve the labels.

    Returns
    -------
    OperatorProduct
        Literal symbolic spin-raising operator.
    """
    return Creation(up, modes=modes) * Annihilation(down, modes=modes)


def SpinMinus(up, down, modes):
    r"""Return :math:`S^-=c_\downarrow^\dagger c_\uparrow` for two modes.

    Parameters
    ----------
    up, down : int or iterable of int
        Mode labels identified with the up and down components.
    modes : FermionModes
        Fermionic modes used to resolve the labels.

    Returns
    -------
    OperatorProduct
        Literal symbolic spin-lowering operator.
    """
    return Creation(down, modes=modes) * Annihilation(up, modes=modes)


def SpinX(up, down, modes):
    r"""Return :math:`S^x=(S^++S^-)/2` for two fermionic modes.

    Parameters
    ----------
    up, down : int or iterable of int
        Mode labels identified with the up and down components.
    modes : FermionModes
        Fermionic modes used to resolve the labels.

    Returns
    -------
    OperatorSum
        Literal symbolic :math:`S^x` operator.
    """
    return 0.5 * (SpinPlus(up, down, modes) + SpinMinus(up, down, modes))


def SpinY(up, down, modes):
    r"""Return :math:`S^y=(S^+-S^-)/(2i)` for two fermionic modes.

    Parameters
    ----------
    up, down : int or iterable of int
        Mode labels identified with the up and down components.
    modes : FermionModes
        Fermionic modes used to resolve the labels.

    Returns
    -------
    OperatorSum
        Literal symbolic :math:`S^y` operator.
    """
    return (-0.5j) * (SpinPlus(up, down, modes) - SpinMinus(up, down, modes))


def SpinZ(up, down, modes):
    r"""Return :math:`S^z=(n_\uparrow-n_\downarrow)/2` for two modes.

    Parameters
    ----------
    up, down : int or iterable of int
        Mode labels identified with the up and down components.
    modes : FermionModes
        Fermionic modes used to resolve the labels.

    Returns
    -------
    OperatorSum
        Literal symbolic :math:`S^z` operator.
    """
    return 0.5 * (
        Number(up, modes=modes) - Number(down, modes=modes)
    )


def HeisenbergExchange(up_i, down_i, up_j, down_j, J, modes):
    r"""Return :math:`J\mathbf{S}_i\cdot\mathbf{S}_j` for two spinful sites.

    Parameters
    ----------
    up_i, down_i : int or iterable of int
        Up and down mode labels at site ``i``.
    up_j, down_j : int or iterable of int
        Up and down mode labels at site ``j``.
    J : numbers.Number
        Heisenberg exchange coupling.
    modes : FermionModes
        Fermionic modes used to resolve all labels.

    Returns
    -------
    Operator
        Literal symbolic expression equivalent to
        :math:`J[S_i^zS_j^z+(S_i^+S_j^-+S_i^-S_j^+)/2]`.
    """
    transverse = 0.5 * (
        SpinPlus(up_i, down_i, modes) * SpinMinus(up_j, down_j, modes)
        + SpinMinus(up_i, down_i, modes) * SpinPlus(up_j, down_j, modes)
    )
    longitudinal = SpinZ(up_i, down_i, modes) * SpinZ(
        up_j, down_j, modes
    )
    return J * (longitudinal + transverse)


def PairHopping(i_up, i_down, j_up, j_down, J, modes):
    r"""Return a Hermitian pair-hopping interaction between two mode pairs.

    Parameters
    ----------
    i_up, i_down : int or iterable of int
        Mode labels of the pair at location ``i``.
    j_up, j_down : int or iterable of int
        Mode labels of the pair at location ``j``.
    J : numbers.Number
        Matrix element for transferring the pair from ``j`` to ``i``.
    modes : FermionModes
        Fermionic modes used to resolve all labels.

    Returns
    -------
    OperatorSum
        Literal number-conserving pair-hopping expression and its Hermitian
        conjugate.
    """
    forward = (
        Creation(i_up, modes=modes)
        * Creation(i_down, modes=modes)
        * Annihilation(j_down, modes=modes)
        * Annihilation(j_up, modes=modes)
    )
    reverse = forward.dag
    J_conj = J.conjugate() if hasattr(J, "conjugate") else J
    return J * forward + J_conj * reverse
