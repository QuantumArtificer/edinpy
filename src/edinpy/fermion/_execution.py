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
class _HoppingPairKernel:
    """Combined hopping transitions between one unordered pair of modes."""

    low_bit: int
    high_bit: int
    transition_mask: int
    parity_mask: int
    low_from_high: complex
    high_from_low: complex


@dataclass(frozen=True, slots=True)
class _MonomialKernel:
    """Compiled product of fermionic creation and annihilation operators."""

    coefficient: complex
    required_occupied_mask: int
    required_empty_mask: int
    transition_mask: int
    parity_mask: int


@dataclass(frozen=True, slots=True)
class _GenericKernel:
    """Fallback retaining the original symbolic Fock algebra."""

    term: object


def _group_hopping_kernels(kernels):
    """Combine directed hopping terms acting on the same pair of modes."""
    groups = {}

    for kernel in kernels:
        low_bit = min(kernel.source_bit, kernel.destination_bit)
        high_bit = max(kernel.source_bit, kernel.destination_bit)

        key = (
            low_bit,
            high_bit,
            kernel.parity_mask,
        )

        low_from_high, high_from_low = groups.get(
            key,
            (0.0j, 0.0j),
        )

        if kernel.source_bit == low_bit:
            high_from_low += kernel.coefficient
        else:
            low_from_high += kernel.coefficient

        groups[key] = (
            low_from_high,
            high_from_low,
        )

    return tuple(
        _HoppingPairKernel(
            low_bit=low_bit,
            high_bit=high_bit,
            transition_mask=low_bit | high_bit,
            parity_mask=parity_mask,
            low_from_high=coefficients[0],
            high_from_low=coefficients[1],
        )
        for (low_bit, high_bit, parity_mask), coefficients
        in groups.items()
    )


class CompiledOperator:
    """Lowered executable representation of a symbolic operator."""

    def __init__(self, kernels):
        self._number_kernels = tuple(
            kernel
            for kernel in kernels
            if isinstance(kernel, _NumberProductKernel)
        )
        hopping_kernels = tuple(
            kernel
            for kernel in kernels
            if isinstance(kernel, _HoppingKernel)
        )
        self._hopping_term_count = len(hopping_kernels)
        self._hopping_kernels = _group_hopping_kernels(
            hopping_kernels
        )
        self._monomial_kernels = tuple(
            kernel
            for kernel in kernels
            if isinstance(kernel, _MonomialKernel)
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
        assembly stage combines them with sum_duplicates().
        """
        if not self.fully_lowered:
            raise RuntimeError(
                "Direct sparse emission requires a fully lowered operator."
            )

        state = ket.state
        ket_amp = ket.amp
        diagonal = 0.0j

        for kernel in self._number_kernels:
            if state & kernel.mask == kernel.mask:
                diagonal += kernel.coefficient

        if diagonal != 0:
            rows.append(column)
            columns.append(column)
            data.append(diagonal * ket_amp)

        for kernel in self._hopping_kernels:
            occupation = state & kernel.transition_mask

            if occupation == kernel.low_bit:
                coefficient = kernel.high_from_low
            elif occupation == kernel.high_bit:
                coefficient = kernel.low_from_high
            else:
                continue

            if coefficient == 0:
                continue

            parity = (state & kernel.parity_mask).bit_count() & 1
            sign = -1 if parity else 1
            final_state = state ^ kernel.transition_mask
            row = bisect_left(basis_states, final_state)

            if row == nbasis or basis_states[row] != final_state:
                continue

            amplitude = coefficient * sign * ket_amp

            if amplitude != 0:
                rows.append(row)
                columns.append(column)
                data.append(amplitude)

        for kernel in self._monomial_kernels:
            if (
                state & kernel.required_occupied_mask
                != kernel.required_occupied_mask
            ):
                continue

            if state & kernel.required_empty_mask:
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

    def emit_csc(self, basis_states):
        """Emit CSC data arrays directly from integer Fock basis states."""
        if not self.fully_lowered:
            raise RuntimeError(
                "Direct matrix emission requires a fully lowered operator."
            )

        basis_index = {
            state: index
            for index, state in enumerate(basis_states)
        }

        indices = []
        indptr = [0]
        data = []

        append_index = indices.append
        append_pointer = indptr.append
        append_data = data.append
        get_index = basis_index.get

        number_kernels = self._number_kernels
        hopping_kernels = self._hopping_kernels
        monomial_kernels = self._monomial_kernels

        for column, state in enumerate(basis_states):
            diagonal = 0.0j

            for kernel in number_kernels:
                if state & kernel.mask == kernel.mask:
                    diagonal += kernel.coefficient

            if diagonal != 0:
                append_index(column)
                append_data(diagonal)

            for kernel in hopping_kernels:
                occupation = state & kernel.transition_mask

                if occupation == kernel.low_bit:
                    coefficient = kernel.high_from_low
                elif occupation == kernel.high_bit:
                    coefficient = kernel.low_from_high
                else:
                    continue

                if coefficient == 0:
                    continue

                parity = (
                    state & kernel.parity_mask
                ).bit_count() & 1

                final_state = state ^ kernel.transition_mask
                row = get_index(final_state)

                if row is None:
                    continue

                amplitude = -coefficient if parity else coefficient

                if amplitude != 0:
                    append_index(row)
                    append_data(amplitude)

            for kernel in monomial_kernels:
                if (
                    state & kernel.required_occupied_mask
                    != kernel.required_occupied_mask
                ):
                    continue

                if state & kernel.required_empty_mask:
                    continue

                parity = (
                    state & kernel.parity_mask
                ).bit_count() & 1

                final_state = state ^ kernel.transition_mask
                row = get_index(final_state)

                if row is None:
                    continue

                coefficient = kernel.coefficient
                amplitude = -coefficient if parity else coefficient

                if amplitude != 0:
                    append_index(row)
                    append_data(amplitude)

            append_pointer(len(data))

        return indices, indptr, data

    def apply(self, ket):
        """Apply the compiled operator and combine duplicate output states."""
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
            occupation = state & kernel.transition_mask

            if occupation == kernel.low_bit:
                coefficient = kernel.high_from_low
            elif occupation == kernel.high_bit:
                coefficient = kernel.low_from_high
            else:
                continue

            if coefficient == 0:
                continue

            parity = (state & kernel.parity_mask).bit_count() & 1
            sign = -1 if parity else 1
            final_state = state ^ kernel.transition_mask
            amplitude = coefficient * sign * ket_amp

            if amplitude != 0:
                output[final_state] = (
                    output.get(final_state, 0.0j) + amplitude
                )

        for kernel in self._monomial_kernels:
            if (
                state & kernel.required_occupied_mask
                != kernel.required_occupied_mask
            ):
                continue

            if state & kernel.required_empty_mask:
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
            "hopping": self._hopping_term_count,
            "hopping_groups": len(self._hopping_kernels),
            "monomial": len(self._monomial_kernels),
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


def _compile_monomial(factors, coefficient):
    """Compile a fermion monomial into occupancy, transition, and parity masks."""
    required_occupied_mask = 0
    required_empty_mask = 0
    transition_mask = 0
    parity_mask = 0
    phase = 1

    for factor in reversed(factors):
        creation = isinstance(factor, Creation)
        site = factor.site
        bit = 1 << site
        toggled = bool(transition_mask & bit)

        # Occupancy immediately before this operation equals the initial
        # occupation XOR the parity of earlier toggles on the same mode.
        requires_occupied = toggled if creation else not toggled

        if requires_occupied:
            required_occupied_mask |= bit
        else:
            required_empty_mask |= bit

        lower_mask = bit - 1
        parity_mask ^= lower_mask

        if (transition_mask & lower_mask).bit_count() & 1:
            phase = -phase

        transition_mask ^= bit

    return _MonomialKernel(
        coefficient=phase * coefficient,
        required_occupied_mask=required_occupied_mask,
        required_empty_mask=required_empty_mask,
        transition_mask=transition_mask,
        parity_mask=parity_mask,
    )


def _lower_term(term):
    coefficient, factors = _split_product(term)

    if all(isinstance(factor, Number) for factor in factors):
        return _NumberProductKernel(
            coefficient=coefficient,
            mask=_number_mask(factors),
        )

    if (
        len(factors) == 2
        and isinstance(factors[0], Creation)
        and isinstance(factors[1], Annihilation)
    ):
        destination = factors[0].site
        source = factors[1].site

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

    if (
        factors
        and all(
            isinstance(factor, (Creation, Annihilation))
            for factor in factors
        )
    ):
        return _compile_monomial(
            factors,
            coefficient,
        )

    return _GenericKernel(term)


def compile_operator(operator):
    """Lower a symbolic fermionic operator into executable kernels."""
    return CompiledOperator(
        [_lower_term(term) for term in _flatten_sum(operator)]
    )
