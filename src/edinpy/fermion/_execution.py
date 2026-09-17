"""Internal execution kernels for fermionic symbolic operators.

The symbolic Fock algebra remains the source of truth. Common operator
patterns are lowered to direct bitwise kernels, while arbitrary expressions
retain the generic symbolic fallback.
"""

from __future__ import annotations

from bisect import bisect_left
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


@dataclass(frozen=True, slots=True)
class _NumberProductKernel:
    """Diagonal product of number operators."""

    coefficient: complex
    mask: int


@dataclass(frozen=True, slots=True)
class _HoppingKernel:
    """One-body fermionic transition c_i^dagger c_j."""

    coefficient: complex
    source_bit: int
    destination_bit: int
    transition_mask: int
    parity_mask: int


@dataclass(frozen=True, slots=True)
class _GenericKernel:
    """Fallback retaining the original symbolic Fock algebra."""

    term: object


class CompiledOperator:
    """Lowered executable representation of a symbolic operator."""

    def __init__(self, kernels):
        self._number_kernels = tuple(
            kernel
            for kernel in kernels
            if isinstance(kernel, _NumberProductKernel)
        )
        self._hopping_kernels = tuple(
            kernel
            for kernel in kernels
            if isinstance(kernel, _HoppingKernel)
        )
        self._generic_kernels = tuple(
            kernel
            for kernel in kernels
            if isinstance(kernel, _GenericKernel)
        )

    @property
    def fully_lowered(self):
        """True when no symbolic fallback is required."""
        return not self._generic_kernels

    def emit_sparse(
        self,
        ket,
        column,
        basis_states,
        nbasis,
        rows,
        columns,
        data,
    ):
        """Emit sparse-matrix entries directly for a fully lowered operator.

        Duplicate matrix entries are intentionally allowed here. The sparse
        assembly stage combines them with ``sum_duplicates()``.
        """
        if not self.fully_lowered:
            raise RuntimeError(
                "Direct sparse emission requires a fully lowered operator."
            )

        state = ket.state
        ket_amp = ket.amp

        # All diagonal terms contribute to the same matrix element.
        diagonal = 0.0j

        for kernel in self._number_kernels:
            if state & kernel.mask == kernel.mask:
                diagonal += kernel.coefficient

        if diagonal != 0:
            rows.append(column)
            columns.append(column)
            data.append(diagonal * ket_amp)

        # One-body transitions.
        for kernel in self._hopping_kernels:
            if not state & kernel.source_bit:
                continue

            if state & kernel.destination_bit:
                continue

            parity = (state & kernel.parity_mask).bit_count() & 1
            sign = -1 if parity else 1

            final_state = state ^ kernel.transition_mask
            row = bisect_left(basis_states, final_state)

            if row == nbasis or basis_states[row] != final_state:
                continue

            amplitude = kernel.coefficient * sign * ket_amp

            if amplitude != 0:
                rows.append(row)
                columns.append(column)
                data.append(amplitude)

    def apply(self, ket):
        """Apply the compiled operator and combine duplicate output states.

        This general path is retained for operators containing symbolic
        fallback kernels and for direct use of the execution API.
        """
        state = ket.state
        ket_amp = ket.amp
        output = {}

        diagonal = 0.0j

        for kernel in self._number_kernels:
            if state & kernel.mask == kernel.mask:
                diagonal += kernel.coefficient

        if diagonal != 0:
            output[state] = diagonal * ket_amp

        for kernel in self._hopping_kernels:
            if not state & kernel.source_bit:
                continue

            if state & kernel.destination_bit:
                continue

            parity = (state & kernel.parity_mask).bit_count() & 1
            sign = -1 if parity else 1

            final_state = state ^ kernel.transition_mask
            amplitude = kernel.coefficient * sign * ket_amp

            if amplitude != 0:
                output[final_state] = (
                    output.get(final_state, 0.0j) + amplitude
                )

        for kernel in self._generic_kernels:
            result = kernel.term * ket

            if isinstance(result, null):
                continue

            if isinstance(result, fockstate):
                if result.amp != 0:
                    output[result.state] = (
                        output.get(result.state, 0.0j)
                        + result.amp
                    )
                continue

            if isinstance(result, statesum):
                for result_state in result.states:
                    if result_state.amp != 0:
                        output[result_state.state] = (
                            output.get(result_state.state, 0.0j)
                            + result_state.amp
                        )
                continue

            raise TypeError(
                "Operator action returned unsupported type "
                f"{type(result).__name__}."
            )

        return output

    @property
    def stats(self):
        return {
            "number_product": len(self._number_kernels),
            "hopping": len(self._hopping_kernels),
            "generic": len(self._generic_kernels),
        }


def _flatten_sum(operator):
    """Yield additive terms from a nested OperatorSum."""
    if isinstance(operator, OperatorSum):
        for term in operator.os:
            yield from _flatten_sum(term)
    else:
        yield operator


def _split_product(term):
    """Separate the numerical coefficient from operator factors."""
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


def _number_mask(factors):
    """Return the occupancy mask required by number operators."""
    mask = 0

    for factor in factors:
        mask |= 1 << factor.site

    return mask


def _between_mask(site_i, site_j):
    """Return the bit mask strictly between two fermionic modes."""
    low = min(site_i, site_j)
    high = max(site_i, site_j)

    if high - low <= 1:
        return 0

    return (1 << high) - (1 << (low + 1))


def _lower_term(term):
    coefficient, factors = _split_product(term)

    # Constants and arbitrary products of number operators.
    if all(isinstance(factor, Number) for factor in factors):
        return _NumberProductKernel(
            coefficient=coefficient,
            mask=_number_mask(factors),
        )

    # General one-body transition c_i^dagger c_j.
    if (
        len(factors) == 2
        and isinstance(factors[0], Creation)
        and isinstance(factors[1], Annihilation)
    ):
        destination = factors[0].site
        source = factors[1].site

        # c_i^dagger c_i = n_i.
        if destination == source:
            return _NumberProductKernel(
                coefficient=coefficient,
                mask=1 << source,
            )

        source_bit = 1 << source
        destination_bit = 1 << destination

        return _HoppingKernel(
            coefficient=coefficient,
            source_bit=source_bit,
            destination_bit=destination_bit,
            transition_mask=source_bit | destination_bit,
            parity_mask=_between_mask(destination, source),
        )

    return _GenericKernel(term)


def compile_operator(operator):
    """Lower a symbolic fermionic operator into executable kernels."""
    return CompiledOperator(
        [_lower_term(term) for term in _flatten_sum(operator)]
    )
