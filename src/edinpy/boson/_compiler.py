"""Compile lowered bosonic terms into sparse-execution kernels."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from numbers import Number as Numeric
from typing import Iterator

from ._algebra import Operator
from ._ir import LoweredOperator, LoweredTerm, Primitive, PrimitiveKind


@dataclass(frozen=True, slots=True)
class NumberKernel:
    """Diagonal product of bosonic number operators.

    Parameters
    ----------
    coefficient : numbers.Number
        Scalar multiplying the complete number-operator product.
    powers : tuple of (int, int)
        ``(mode, power)`` pairs. Repeated number operators on one mode are
        combined into a single integer power during compilation.
    """

    coefficient: Numeric
    powers: tuple[tuple[int, int], ...]

    def evaluate(self, state: int, bits_per_mode: int, mask: int) -> Numeric:
        value = self.coefficient
        for mode, power in self.powers:
            occupation = (state >> (mode * bits_per_mode)) & mask
            if occupation == 0:
                return 0
            value *= occupation**power
        return value


@dataclass(frozen=True, slots=True)
class HoppingKernel:
    """One-boson transfer between an unordered pair of modes.

    Parameters
    ----------
    low, high : int
        Ordered flat mode indices with ``low < high``.
    low_from_high : numbers.Number
        Coefficient for moving one boson from ``high`` to ``low``.
    high_from_low : numbers.Number
        Coefficient for moving one boson from ``low`` to ``high``. A zero
        coefficient disables that direction.
    """

    low: int
    high: int
    low_from_high: Numeric
    high_from_low: Numeric

    def transitions(
        self, state: int, bits_per_mode: int, mask: int
    ) -> Iterator[tuple[int, Numeric]]:
        low_shift = self.low * bits_per_mode
        high_shift = self.high * bits_per_mode
        n_low = (state >> low_shift) & mask
        n_high = (state >> high_shift) & mask
        unit_low = 1 << low_shift
        unit_high = 1 << high_shift

        if self.low_from_high != 0 and n_high:
            yield (
                state + unit_low - unit_high,
                self.low_from_high * sqrt((n_low + 1) * n_high),
            )
        if self.high_from_low != 0 and n_low:
            yield (
                state - unit_low + unit_high,
                self.high_from_low * sqrt((n_high + 1) * n_low),
            )


@dataclass(frozen=True, slots=True)
class PairHoppingKernel:
    """Two-boson transfer between an unordered pair of modes.

    Parameters
    ----------
    low, high : int
        Ordered flat mode indices with ``low < high``.
    low_from_high : numbers.Number
        Coefficient for transferring two bosons from ``high`` to ``low``.
    high_from_low : numbers.Number
        Coefficient for the reverse transfer.
    """

    low: int
    high: int
    low_from_high: Numeric
    high_from_low: Numeric

    def transitions(
        self, state: int, bits_per_mode: int, mask: int
    ) -> Iterator[tuple[int, Numeric]]:
        low_shift = self.low * bits_per_mode
        high_shift = self.high * bits_per_mode
        n_low = (state >> low_shift) & mask
        n_high = (state >> high_shift) & mask
        two_low = 2 << low_shift
        two_high = 2 << high_shift

        if self.low_from_high != 0 and n_high >= 2:
            factor = (n_low + 1) * (n_low + 2) * n_high * (n_high - 1)
            yield (
                state + two_low - two_high,
                self.low_from_high * sqrt(factor),
            )
        if self.high_from_low != 0 and n_low >= 2:
            factor = (n_high + 1) * (n_high + 2) * n_low * (n_low - 1)
            yield (
                state - two_low + two_high,
                self.high_from_low * sqrt(factor),
            )


@dataclass(frozen=True, slots=True)
class MonomialKernel:
    """General fully lowered bosonic primitive monomial.

    Parameters
    ----------
    coefficient : numbers.Number
        Scalar multiplying the monomial.
    operations : tuple of (str, int)
        Primitive operations in actual ket-action order, so the rightmost
        symbolic factor appears first.
    touched_modes : tuple of int
        Sorted unique flat modes whose occupations are needed.
    operation_slots : tuple of int
        For each operation, the position of its mode inside ``touched_modes``.
    particle_delta : int
        Net number of created minus annihilated bosons. Fixed-particle
        Hamiltonians project nonzero values out of matrix execution.
    """

    coefficient: Numeric
    operations: tuple[tuple[PrimitiveKind, int], ...]
    touched_modes: tuple[int, ...]
    operation_slots: tuple[int, ...]
    particle_delta: int

    def transition(
        self, state: int, bits_per_mode: int, mask: int
    ) -> tuple[int, Numeric] | None:
        initial = [
            (state >> (mode * bits_per_mode)) & mask
            for mode in self.touched_modes
        ]
        occupations = list(initial)
        amplitude = self.coefficient

        for (kind, _mode), slot in zip(self.operations, self.operation_slots):
            occupation = occupations[slot]
            if kind == "annihilate":
                if occupation == 0:
                    return None
                amplitude *= sqrt(occupation)
                occupations[slot] = occupation - 1
            elif kind == "create":
                amplitude *= sqrt(occupation + 1)
                occupations[slot] = occupation + 1
            elif kind == "number":
                if occupation == 0:
                    return None
                amplitude *= occupation
            else:
                raise RuntimeError(f"Unknown bosonic primitive kind {kind!r}.")

        if amplitude == 0:
            return None

        final_state = state
        for mode, before, after in zip(self.touched_modes, initial, occupations):
            delta = after - before
            if delta:
                final_state += delta * (1 << (mode * bits_per_mode))
        return final_state, amplitude


@dataclass(frozen=True, slots=True)
class CompiledPlan:
    """Immutable kernel groups produced by bosonic compilation.

    Specialized number, hopping, pair-hopping, and general monomial kernels
    are kept separate so sparse, NumPy, and Numba execution can share one
    compilation result. ``generic_terms`` contains only custom symbolic terms
    that could not be lowered numerically.
    """

    number_kernels: tuple[NumberKernel, ...]
    hopping_kernels: tuple[HoppingKernel, ...]
    pair_hopping_kernels: tuple[PairHoppingKernel, ...]
    monomial_kernels: tuple[MonomialKernel, ...]
    generic_terms: tuple[Operator, ...]
    projected_out_terms: int
    real_valued: bool


def _is_real(value: Numeric) -> bool:
    try:
        return complex(value).imag == 0
    except (TypeError, ValueError):
        return False


def _number_powers(
    primitives: tuple[Primitive, ...],
) -> tuple[tuple[int, int], ...] | None:
    counts = {}
    for primitive in primitives:
        if primitive.kind != "number":
            return None
        counts[primitive.mode] = counts.get(primitive.mode, 0) + 1
    return tuple(sorted(counts.items()))


def _canonical_hopping(
    primitives: tuple[Primitive, ...],
) -> tuple[int, int] | None:
    if len(primitives) != 2:
        return None
    left, right = primitives
    if left.kind != "create" or right.kind != "annihilate":
        return None
    return left.mode, right.mode


def _canonical_pair_hopping(
    primitives: tuple[Primitive, ...],
) -> tuple[int, int] | None:
    """Return destination/source for ``(b†_dst)^2 b_src^2`` or ``None``."""
    if len(primitives) != 4:
        return None
    first, second, third, fourth = primitives
    if (
        first.kind != "create"
        or second.kind != "create"
        or third.kind != "annihilate"
        or fourth.kind != "annihilate"
    ):
        return None
    if first.mode != second.mode or third.mode != fourth.mode:
        return None
    if first.mode == third.mode:
        return None
    return first.mode, third.mode


def _general_monomial(term: LoweredTerm) -> MonomialKernel:
    # Symbolic factors are stored left-to-right; the rightmost acts first.
    operations = tuple(
        (primitive.kind, primitive.mode)
        for primitive in reversed(term.primitives)
    )
    touched_modes = tuple(sorted({mode for _kind, mode in operations}))
    slot_for_mode = {mode: slot for slot, mode in enumerate(touched_modes)}
    operation_slots = tuple(slot_for_mode[mode] for _kind, mode in operations)
    return MonomialKernel(
        coefficient=term.coefficient,
        operations=operations,
        touched_modes=touched_modes,
        operation_slots=operation_slots,
        particle_delta=(
            int(term.particle_delta)
            if term.particle_delta is not None
            else 0
        ),
    )


def compile_lowered(lowered: LoweredOperator) -> CompiledPlan:
    """Compile primitive terms into the bosonic execution kernel portfolio.

    Parameters
    ----------
    lowered : LoweredOperator
        Primitive additive representation returned by :func:`lower_operator`.
        Equivalent number products and directed transfers are combined before
        execution so repeated symbolic terms do not produce duplicate kernels.

    Returns
    -------
    CompiledPlan
        Immutable kernel groups used by sparse, NumPy matrix-free, and Numba
        matrix-free execution. Unsupported custom symbolic terms are retained
        in ``generic_terms`` for the literal NumPy fallback.
    """
    if not isinstance(lowered, LoweredOperator):
        raise TypeError("'lowered' must be a LoweredOperator instance.")

    number_coefficients = {}
    hopping_coefficients = {}
    pair_hopping_coefficients = {}
    pair_hopping_terms = {}
    monomial_coefficients = {}
    monomial_examples = {}
    generic_terms = []
    projected_out_terms = 0
    real_valued = True

    for term in lowered.terms:
        if not term.fully_lowered:
            generic_terms.append(term.source)
            real_valued = False
            continue

        coefficient = term.coefficient
        real_valued = real_valued and _is_real(coefficient)

        if term.particle_delta != 0:
            # Keep the kernel for literal compiled action, but fixed-N matrix
            # emission will project it out exactly.
            projected_out_terms += 1

        number_powers = _number_powers(term.primitives)
        if number_powers is not None:
            number_coefficients[number_powers] = (
                number_coefficients.get(number_powers, 0) + coefficient
            )
            continue

        hopping = _canonical_hopping(term.primitives)
        if hopping is not None:
            destination, source = hopping
            if destination == source:
                powers = ((destination, 1),)
                number_coefficients[powers] = (
                    number_coefficients.get(powers, 0) + coefficient
                )
                continue
            low, high = sorted((destination, source))
            key = (low, high)
            low_from_high, high_from_low = hopping_coefficients.get(key, (0, 0))
            if destination == low:
                low_from_high += coefficient
            else:
                high_from_low += coefficient
            hopping_coefficients[key] = (low_from_high, high_from_low)
            continue

        pair_hopping = _canonical_pair_hopping(term.primitives)
        if pair_hopping is not None:
            destination, source = pair_hopping
            low, high = sorted((destination, source))
            key = (low, high)
            low_from_high, high_from_low = pair_hopping_coefficients.get(
                key, (0, 0)
            )
            if destination == low:
                low_from_high += coefficient
            else:
                high_from_low += coefficient
            pair_hopping_coefficients[key] = (low_from_high, high_from_low)
            pair_hopping_terms.setdefault(key, []).append(term)
            continue

        kernel = _general_monomial(term)
        key = kernel.operations
        monomial_coefficients[key] = monomial_coefficients.get(key, 0) + coefficient
        monomial_examples.setdefault(key, kernel)

    number_kernels = tuple(
        NumberKernel(coefficient, powers)
        for powers, coefficient in sorted(number_coefficients.items())
        if coefficient != 0
    )
    hopping_kernels = tuple(
        HoppingKernel(low, high, low_from_high, high_from_low)
        for (low, high), (low_from_high, high_from_low) in sorted(
            hopping_coefficients.items()
        )
        if low_from_high != 0 or high_from_low != 0
    )

    pair_hopping_kernels = []
    for (low, high), (low_from_high, high_from_low) in sorted(
        pair_hopping_coefficients.items()
    ):
        # Specialize only complete two-way pair transfer. A lone directed
        # quartic monomial remains on the fully general monomial path so the
        # compiler preserves existing behavior for arbitrary expressions.
        if low_from_high != 0 and high_from_low != 0:
            pair_hopping_kernels.append(
                PairHoppingKernel(low, high, low_from_high, high_from_low)
            )
            continue

        for term in pair_hopping_terms[(low, high)]:
            kernel = _general_monomial(term)
            key = kernel.operations
            monomial_coefficients[key] = (
                monomial_coefficients.get(key, 0) + term.coefficient
            )
            monomial_examples.setdefault(key, kernel)

    monomial_kernels = []
    for key, coefficient in monomial_coefficients.items():
        if coefficient == 0:
            continue
        example = monomial_examples[key]
        monomial_kernels.append(
            MonomialKernel(
                coefficient=coefficient,
                operations=example.operations,
                touched_modes=example.touched_modes,
                operation_slots=example.operation_slots,
                particle_delta=example.particle_delta,
            )
        )

    return CompiledPlan(
        number_kernels=number_kernels,
        hopping_kernels=hopping_kernels,
        pair_hopping_kernels=tuple(pair_hopping_kernels),
        monomial_kernels=tuple(monomial_kernels),
        generic_terms=tuple(generic_terms),
        projected_out_terms=projected_out_terms,
        real_valued=real_valued,
    )
