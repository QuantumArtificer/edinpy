"""Common number-conserving fermionic operators expressed in literal Fock algebra."""

from __future__ import annotations

from edinpy._core._operator_builders import (
    conjugate as _conjugate,
    density_density as _density_density,
    heisenberg_exchange as _heisenberg_exchange,
    hopping as _hopping,
    onsite as _onsite,
    spin_minus as _spin_minus,
    spin_plus as _spin_plus,
    spin_x as _spin_x,
    spin_y as _spin_y,
    spin_z as _spin_z,
)
from ._algebra import Annihilation, Creation, Number


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
    return _hopping(Creation, Annihilation, i, j, t, modes)


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
    return _onsite(Number, i, epsilon, modes)


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
    return _density_density(Number, i, j, V, modes)


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
    return _spin_plus(Creation, Annihilation, up, down, modes)


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
    return _spin_minus(Creation, Annihilation, up, down, modes)


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
    return _spin_x(SpinPlus, SpinMinus, up, down, modes)


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
    return _spin_y(SpinPlus, SpinMinus, up, down, modes)


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
    return _spin_z(Number, up, down, modes)


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
    return _heisenberg_exchange(
        SpinPlus,
        SpinMinus,
        SpinZ,
        up_i,
        down_i,
        up_j,
        down_j,
        J,
        modes,
    )


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
    return J * forward + _conjugate(J) * forward.dag
