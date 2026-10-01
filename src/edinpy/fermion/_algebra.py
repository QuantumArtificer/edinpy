"""Literal symbolic algebra for fermionic creation, annihilation, and number operators."""

from __future__ import annotations

from edinpy._core._algebra import (
    Operator,
    OperatorProduct,
    OperatorSum,
    Scalar,
    SingleModeOperator,
    set_notation,
    single_operator_modes,
)
from ._basis import FockState, NullState
from ._modes import FermionModes


class _SingleModeOperator(SingleModeOperator):
    """Base class for primitive operators acting on one fermionic mode."""

    modes_type = FermionModes


class Annihilation(_SingleModeOperator):
    r"""Fermionic annihilation operator :math:`c_i`.

    Parameters
    ----------
    indices : int or iterable of int
        Flat mode index or degree-of-freedom coordinates resolved by ``modes``.
    modes : FermionModes
        Ordered mode specification on which the operator acts.
    """

    symbol = "c"

    def _action(self, state):
        """Apply fermionic annihilation with the canonical occupation ordering."""
        bit = 1 << self.mode
        if not (state.state & bit):
            return NullState()
        parity = (state.state & (bit - 1)).bit_count() & 1
        sign = -1 if parity else 1
        return FockState(
            state.state ^ bit,
            amp=sign * state.amp,
            n_modes=state.n_modes,
        )

    @property
    def dag(self):
        """Creation: Hermitian adjoint of the annihilation operator."""
        return Creation(self.indices, modes=self.modes)


class Creation(_SingleModeOperator):
    r"""Fermionic creation operator :math:`c_i^\dagger`.

    Parameters
    ----------
    indices : int or iterable of int
        Flat mode index or degree-of-freedom coordinates resolved by ``modes``.
    modes : FermionModes
        Ordered mode specification on which the operator acts.
    """

    symbol = "c†"

    def _action(self, state):
        """Apply fermionic creation with the canonical occupation ordering."""
        bit = 1 << self.mode
        if state.state & bit:
            return NullState()
        parity = (state.state & (bit - 1)).bit_count() & 1
        sign = -1 if parity else 1
        return FockState(
            state.state ^ bit,
            amp=sign * state.amp,
            n_modes=state.n_modes,
        )

    @property
    def dag(self):
        """Annihilation: Hermitian adjoint of the creation operator."""
        return Annihilation(self.indices, modes=self.modes)


class Number(_SingleModeOperator):
    r"""Fermionic number operator :math:`n_i=c_i^\dagger c_i`.

    Parameters
    ----------
    indices : int or iterable of int
        Flat mode index or degree-of-freedom coordinates resolved by ``modes``.
    modes : FermionModes
        Ordered mode specification on which the operator acts.
    """

    symbol = "n"

    def _action(self, state):
        """Return the input state when the mode is occupied and zero otherwise."""
        if state.state & (1 << self.mode):
            return FockState(
                state.state,
                amp=state.amp,
                n_modes=state.n_modes,
                index=state.index,
            )
        return NullState()

    @property
    def dag(self):
        """Number: Hermitian adjoint of the number operator."""
        return self




def operator_modes(operator):
    """Return the unique :class:`FermionModes` used by an expression."""
    return single_operator_modes(operator, FermionModes)
