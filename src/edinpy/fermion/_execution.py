"""Internal execution kernels for fermionic symbolic operators.

The symbolic Fock algebra remains the source of truth.  This module lowers
common operator forms to direct bitwise kernels while retaining a generic
symbolic fallback for operators that are not recognized.
"""

from __future__ import annotations

from dataclasses import dataclass

from .fermion import (
    Annihilation,
    Creation,
    Number,
    OperatorProduct,
    OperatorSum,
    fockstate,
    null,
    scalar,
    statesum,
)


def _fermionic_sign(state: int, site: int) -> int:
    """Return (-1)**N_<site for an integer occupation state."""
    preceding_mask = (1 << site) - 1
    return -1 if (state & preceding_mask).bit_count() & 1 else 1


@dataclass(frozen=True, slots=True)
class _NumberProductKernel:
    coefficient: complex
    sites: tuple[int, ...]

    def apply(self, ket):
        state = ket.state

        for site in self.sites:
            if not state & (1 << site):
                return ()

        return ((state, self.coefficient * ket.amp),)


@dataclass(frozen=True, slots=True)
class _HoppingKernel:
    coefficient: complex
    destination: int
    source: int

    def apply(self, ket):
        state = ket.state
        source_bit = 1 << self.source

        if not state & source_bit:
            return ()

        sign = _fermionic_sign(state, self.source)
        intermediate = state ^ source_bit

        destination_bit = 1 << self.destination

        if intermediate & destination_bit:
            return ()

        sign *= _fermionic_sign(intermediate, self.destination)
        final_state = intermediate ^ destination_bit

        return ((final_state, self.coefficient * sign * ket.amp),)


@dataclass(frozen=True, slots=True)
class _GenericKernel:
    term: object

    def apply(self, ket):
        result = self.term * ket

        if isinstance(result, null):
            return ()

        if isinstance(result, fockstate):
            return ((result.state, result.amp),)

        if isinstance(result, statesum):
            return tuple(
                (state.state, state.amp)
                for state in result.states
            )

        raise TypeError(
            "Operator action returned unsupported type "
            f"{type(result).__name__}."
        )


class CompiledOperator:
    """Lowered representation of a symbolic fermionic operator."""

    def __init__(self, kernels):
        self.kernels = tuple(kernels)

    def apply(self, ket):
        """Apply all lowered terms and combine duplicate output states."""
        output = {}

        for kernel in self.kernels:
            for state, amplitude in kernel.apply(ket):
                if amplitude != 0:
                    output[state] = output.get(state, 0.0j) + amplitude

        return output

    @property
    def stats(self):
        counts = {
            "number_product": 0,
            "hopping": 0,
            "generic": 0,
        }

        for kernel in self.kernels:
            if isinstance(kernel, _NumberProductKernel):
                counts["number_product"] += 1
            elif isinstance(kernel, _HoppingKernel):
                counts["hopping"] += 1
            else:
                counts["generic"] += 1

        return counts


def _flatten_sum(operator):
    """Yield additive terms from a possibly nested OperatorSum."""
    if isinstance(operator, OperatorSum):
        for term in operator.os:
            yield from _flatten_sum(term)
    else:
        yield operator


def _split_product(term):
    """Return numerical coefficient and non-scalar factors."""
    coefficient = 1.0

    if isinstance(term, scalar):
        return term.value, []

    if isinstance(term, OperatorProduct):
        factors = []

        for factor in term.op:
            if isinstance(factor, scalar):
                coefficient *= factor.value
            else:
                factors.append(factor)

        return coefficient, factors

    return coefficient, [term]


def _lower_term(term):
    coefficient, factors = _split_product(term)

    # Constants and products of number operators are diagonal.
    if all(isinstance(factor, Number) for factor in factors):
        return _NumberProductKernel(
            coefficient=coefficient,
            sites=tuple(factor.site for factor in factors),
        )

    # c_i^\dagger c_j: generic one-body fermionic transition.
    if (
        len(factors) == 2
        and isinstance(factors[0], Creation)
        and isinstance(factors[1], Annihilation)
    ):
        return _HoppingKernel(
            coefficient=coefficient,
            destination=factors[0].site,
            source=factors[1].site,
        )

    # Anything else retains the original Fock-algebra implementation.
    return _GenericKernel(term)


def compile_operator(operator):
    """Lower a symbolic operator into executable kernels."""
    return CompiledOperator(
        [_lower_term(term) for term in _flatten_sum(operator)]
    )
