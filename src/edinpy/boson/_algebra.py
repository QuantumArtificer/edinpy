"""Literal symbolic algebra for bosonic creation, annihilation, and number operators."""

from __future__ import annotations

from math import sqrt

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
from ._modes import BosonModes


class _SingleModeOperator(SingleModeOperator):
    """Base class for primitive operators acting on one bosonic mode."""

    modes_type = BosonModes


class Annihilation(_SingleModeOperator):
    r"""Bosonic annihilation operator :math:`b_i`.

    Parameters
    ----------
    indices : int or iterable of int
        Flat mode index or degree-of-freedom coordinates resolved by ``modes``.
    modes : BosonModes
        Ordered mode specification on which the operator acts.
    """

    symbol = "b"

    def _action(self, state):
        occupation = state.occupation(self.mode)
        if occupation == 0:
            return NullState()
        occupations = list(state.occupations)
        occupations[self.mode] -= 1
        return FockState(
            occupations,
            amp=state.amp * sqrt(occupation),
        )

    @property
    def dag(self):
        return Creation(self.indices, modes=self.modes)


class Creation(_SingleModeOperator):
    r"""Bosonic creation operator :math:`b_i^\dagger`.

    Parameters
    ----------
    indices : int or iterable of int
        Flat mode index or degree-of-freedom coordinates resolved by ``modes``.
    modes : BosonModes
        Ordered mode specification on which the operator acts.
    """

    symbol = "b†"

    def _action(self, state):
        occupation = state.occupation(self.mode)
        occupations = list(state.occupations)
        occupations[self.mode] += 1
        return FockState(
            occupations,
            amp=state.amp * sqrt(occupation + 1),
        )

    @property
    def dag(self):
        return Annihilation(self.indices, modes=self.modes)


class Number(_SingleModeOperator):
    r"""Bosonic number operator :math:`n_i=b_i^\dagger b_i`.

    Parameters
    ----------
    indices : int or iterable of int
        Flat mode index or degree-of-freedom coordinates resolved by ``modes``.
    modes : BosonModes
        Ordered mode specification on which the operator acts.
    """

    symbol = "n"

    def _action(self, state):
        occupation = state.occupation(self.mode)
        if occupation == 0:
            return NullState()
        return FockState(
            state.occupations,
            amp=occupation * state.amp,
            index=state.index,
        )

    @property
    def dag(self):
        return self




def operator_modes(operator):
    """Return the unique :class:`BosonModes` used by an expression."""
    return single_operator_modes(operator, BosonModes)
