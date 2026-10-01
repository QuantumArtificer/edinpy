"""Common number-conserving bosonic operators in literal Fock algebra."""

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
    r"""Return a Hermitian one-boson hopping term.

    Parameters
    ----------
    i, j : int or iterable of int
        Mode labels for the two bosonic modes.
    t : numbers.Number
        Matrix element multiplying :math:`b_i^\dagger b_j`. The reverse
        process uses ``conj(t)``. The conventional Bose-Hubbard kinetic term
        :math:`-t_0(b_i^\dagger b_j+\mathrm{H.c.})` is obtained with
        ``t=-t0`` for real ``t0``.
    modes : BosonModes
        Bosonic modes used to resolve ``i`` and ``j``.

    Returns
    -------
    OperatorSum
        Literal symbolic expression
        :math:`t b_i^\dagger b_j+t^* b_j^\dagger b_i`.
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
    modes : BosonModes
        Bosonic modes used to resolve ``i``.

    Returns
    -------
    OperatorProduct
        Literal symbolic expression :math:`\epsilon_i n_i`.
    """
    return _onsite(Number, i, epsilon, modes)


def DensityDensity(i, j, V, modes):
    r"""Return a density-density interaction :math:`V_{ij} n_i n_j`.

    Parameters
    ----------
    i, j : int or iterable of int
        Mode labels.
    V : numbers.Number
        Density-density interaction strength.
    modes : BosonModes
        Bosonic modes used to resolve ``i`` and ``j``.

    Returns
    -------
    OperatorProduct
        Literal symbolic density-density interaction.
    """
    return _density_density(Number, i, j, V, modes)


def Hubbard(i, U, modes):
    r"""Return the standard bosonic onsite interaction.

    The operator is

    .. math::

        \frac{U}{2} n_i(n_i-1),

    which is equal to :math:`(U/2)b_i^\dagger b_i^\dagger b_i b_i`.

    Parameters
    ----------
    i : int or iterable of int
        Mode label.
    U : numbers.Number
        Onsite interaction strength.
    modes : BosonModes
        Bosonic modes used to resolve ``i``.

    Returns
    -------
    Operator
        Literal symbolic onsite interaction.
    """
    n_i = Number(i, modes=modes)
    return 0.5 * U * n_i * (n_i - 1)


def PairHopping(i, j, J, modes):
    r"""Return Hermitian two-boson pair hopping between two modes.

    The operator is

    .. math::

        J (b_i^\dagger)^2 b_j^2 + J^* (b_j^\dagger)^2 b_i^2.

    Parameters
    ----------
    i, j : int or iterable of int
        Mode labels between which a boson pair is transferred.
    J : numbers.Number
        Pair-transfer matrix element.
    modes : BosonModes
        Bosonic modes used to resolve ``i`` and ``j``.

    Returns
    -------
    OperatorSum
        Literal number-conserving pair-hopping expression.
    """
    forward = (
        Creation(i, modes=modes)
        * Creation(i, modes=modes)
        * Annihilation(j, modes=modes)
        * Annihilation(j, modes=modes)
    )
    return J * forward + _conjugate(J) * forward.dag

def SpinPlus(up, down, modes):
    r"""Return the Schwinger-boson raising operator.

    The operator is :math:`S^+=b_\uparrow^\dagger b_\downarrow`.

    Parameters
    ----------
    up, down : int or iterable of int
        Mode labels identified with the two internal bosonic components.
    modes : BosonModes
        Bosonic modes used to resolve the labels.

    Returns
    -------
    OperatorProduct
        Literal symbolic pseudospin-raising operator.

    Notes
    -----
    Together with :func:`SpinMinus` and :func:`SpinZ`, this is the Schwinger-
    boson representation of angular momentum.  On a subspace with fixed local
    occupation ``n_up + n_down = 2S`` it realizes a spin-``S`` representation.
    """
    return _spin_plus(Creation, Annihilation, up, down, modes)


def SpinMinus(up, down, modes):
    r"""Return the Schwinger-boson lowering operator.

    The operator is :math:`S^-=b_\downarrow^\dagger b_\uparrow`.

    Parameters
    ----------
    up, down : int or iterable of int
        Mode labels identified with the two internal bosonic components.
    modes : BosonModes
        Bosonic modes used to resolve the labels.

    Returns
    -------
    OperatorProduct
        Literal symbolic pseudospin-lowering operator.
    """
    return _spin_minus(Creation, Annihilation, up, down, modes)


def SpinX(up, down, modes):
    r"""Return :math:`S^x=(S^++S^-)/2` for two bosonic components.

    Parameters
    ----------
    up, down : int or iterable of int
        Mode labels identified with the two internal bosonic components.
    modes : BosonModes
        Bosonic modes used to resolve the labels.

    Returns
    -------
    OperatorSum
        Literal symbolic :math:`S^x` operator.
    """
    return _spin_x(SpinPlus, SpinMinus, up, down, modes)


def SpinY(up, down, modes):
    r"""Return :math:`S^y=(S^+-S^-)/(2i)` for two bosonic components.

    Parameters
    ----------
    up, down : int or iterable of int
        Mode labels identified with the two internal bosonic components.
    modes : BosonModes
        Bosonic modes used to resolve the labels.

    Returns
    -------
    OperatorSum
        Literal symbolic :math:`S^y` operator.
    """
    return _spin_y(SpinPlus, SpinMinus, up, down, modes)


def SpinZ(up, down, modes):
    r"""Return :math:`S^z=(n_\uparrow-n_\downarrow)/2`.

    Parameters
    ----------
    up, down : int or iterable of int
        Mode labels identified with the two internal bosonic components.
    modes : BosonModes
        Bosonic modes used to resolve the labels.

    Returns
    -------
    OperatorSum
        Literal symbolic :math:`S^z` operator.
    """
    return _spin_z(Number, up, down, modes)


def HeisenbergExchange(up_i, down_i, up_j, down_j, J, modes):
    r"""Return :math:`J\mathbf{S}_i\cdot\mathbf{S}_j` in Schwinger-boson form.

    Parameters
    ----------
    up_i, down_i : int or iterable of int
        Two component-mode labels at location ``i``.
    up_j, down_j : int or iterable of int
        Two component-mode labels at location ``j``.
    J : numbers.Number
        Exchange coupling.
    modes : BosonModes
        Bosonic modes used to resolve all labels.

    Returns
    -------
    Operator
        Literal expression
        :math:`J[S_i^zS_j^z+(S_i^+S_j^-+S_i^-S_j^+)/2]`.

    Notes
    -----
    When each location is constrained to one boson, this reduces to the usual
    spin-1/2 Heisenberg exchange.  More generally, a fixed local occupation
    ``n_up + n_down = 2S`` realizes spin ``S`` at that location.
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
