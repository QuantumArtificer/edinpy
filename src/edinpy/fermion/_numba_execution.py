"""Numba matrix-free execution for complete fixed-particle fermion bases."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray

from ._execution import _complete_rank_tables

if TYPE_CHECKING:
    from ._basis import FockBasis
    from ._execution import CompiledOperator

try:  # Optional dependency: imported only by explicit execution selection.
    from numba import njit, prange
except ImportError:  # pragma: no cover - exercised on installations without Numba.
    njit = None
    prange = None


@dataclass(frozen=True, slots=True)
class _NumbaOperatorPlan:
    """Homogeneous arrays describing one fully lowered fermion operator.

    The symbolic compiler groups terms by execution behavior before this plan
    is built. Keeping diagonal, simple-hopping, parity-free-hopping, signed-
    hopping, and general monomial arrays separate lets each Numba kernel avoid
    branches that do not apply to its operator family.
    """

    number_masks: NDArray[np.uint64]
    number_coefficients: NDArray[np.generic]
    simple_hopping_transition_masks: NDArray[np.uint64]
    simple_hopping_coefficients: NDArray[np.generic]
    parity_free_hopping_low_bits: NDArray[np.uint64]
    parity_free_hopping_high_bits: NDArray[np.uint64]
    parity_free_hopping_transition_masks: NDArray[np.uint64]
    parity_free_hopping_low_from_high: NDArray[np.generic]
    parity_free_hopping_high_from_low: NDArray[np.generic]
    signed_hopping_low_bits: NDArray[np.uint64]
    signed_hopping_high_bits: NDArray[np.uint64]
    signed_hopping_transition_masks: NDArray[np.uint64]
    signed_hopping_parity_masks: NDArray[np.uint64]
    signed_hopping_low_from_high: NDArray[np.generic]
    signed_hopping_high_from_low: NDArray[np.generic]
    monomial_occupied_masks: NDArray[np.uint64]
    monomial_empty_masks: NDArray[np.uint64]
    monomial_transition_masks: NDArray[np.uint64]
    monomial_parity_masks: NDArray[np.uint64]
    monomial_coefficients: NDArray[np.generic]


@dataclass(frozen=True, slots=True)
class _NumbaExecutionPlan:
    """Cached operator and basis data shared by repeated Numba matvec calls.

    ``operator`` contains arrays that depend only on the compiled Hamiltonian.
    ``states`` fixes the complete-basis ordering. ``rank_contributions`` and
    ``byte_popcounts`` map native occupation words directly to basis indices
    for off-diagonal transitions. Purely diagonal plans leave the rank tables
    empty because no destination lookup is required.
    """

    operator: _NumbaOperatorPlan
    states: NDArray[np.uint64]
    rank_contributions: NDArray[np.int64]
    byte_popcounts: NDArray[np.uint8]


_NumbaMode = Literal["numba-serial", "numba-parallel"]


def numba_available() -> bool:
    """Return ``True`` when EDinPy can import the optional Numba runtime."""
    return njit is not None


def _unsupported_reason(
    compiled: "CompiledOperator",
    basis: "FockBasis",
) -> str | None:
    """Return why a compiled operator/basis pair cannot use Numba execution."""
    if not numba_available():
        return (
            "Numba is not installed. Install EDinPy with the optional "
            "'numba' extra or use execution='numpy'."
        )
    if not compiled.fully_lowered:
        return "generic symbolic fallback kernels are present"
    if int(basis.n_modes) > 64:
        return "more than 64 fermion modes require the arbitrary-width executor"
    if not bool(getattr(basis, "is_complete", False)):
        return "projected/constrained bases are not supported by Numba execution yet"
    return None


def numba_supported(compiled: "CompiledOperator", basis: "FockBasis") -> bool:
    """Return whether the operator and basis satisfy Numba executor requirements.

    Parameters
    ----------
    compiled : CompiledOperator
        Fully compiled fermion operator. Numba execution requires every term to
        have a lowered bitwise representation; generic symbolic fallback terms
        are not accepted.
    basis : FockBasis
        Built basis on which the operator acts. The Numba backend currently
        accepts complete fixed-particle bases with at most 64 fermion modes.

    Returns
    -------
    bool
        ``True`` when both serial and parallel Numba execution are available.
    """
    return _unsupported_reason(compiled, basis) is None


def require_numba_supported(
    compiled: "CompiledOperator",
    basis: "FockBasis",
    execution: _NumbaMode,
) -> None:
    """Validate one strict Numba execution request.

    Parameters
    ----------
    compiled : CompiledOperator
        Fermion operator prepared by the symbolic compiler.
    basis : FockBasis
        Complete fixed-particle basis associated with the Hamiltonian sector.
    execution : {"numba-serial", "numba-parallel"}
        Strict execution mode requested by the caller. The value is included in
        any error message so unsupported explicit requests remain easy to
        diagnose.

    Raises
    ------
    ValueError
        If ``execution`` is not one of the two strict Numba modes.
    NotImplementedError
        If the operator, basis, or installed dependencies cannot satisfy the
        requested Numba execution mode.
    """
    if execution not in {"numba-serial", "numba-parallel"}:
        raise ValueError(f"Unknown strict Numba execution mode {execution!r}.")
    reason = _unsupported_reason(compiled, basis)
    if reason is not None:
        raise NotImplementedError(
            f"execution={execution!r} is unavailable for this Hamiltonian: "
            f"{reason}. Use execution='mixed' or execution='numpy'."
        )


def _pack_operator_plan(compiled: "CompiledOperator") -> _NumbaOperatorPlan:
    """Pack lowered operator kernels into arrays suitable for Numba.

    Parameters
    ----------
    compiled : CompiledOperator
        Fully lowered fermion operator. Its kernel families have already been
        grouped by the symbolic compiler.

    Returns
    -------
    _NumbaOperatorPlan
        Immutable collection of homogeneous masks and coefficient arrays. The
        plan is cached on ``compiled`` and reused by every Numba execution.
    """
    cached = getattr(compiled, "_numba_operator_plan", None)
    if cached is not None:
        return cached

    coefficient_dtype = np.dtype(compiled.data_dtype)

    number_masks = np.asarray(
        [kernel.mask for kernel in compiled._number_kernels],
        dtype=np.uint64,
    )
    number_coefficients = np.asarray(
        [compiled._coerce_coefficient(kernel.coefficient) for kernel in compiled._number_kernels],
        dtype=coefficient_dtype,
    )

    simple_hopping_transition_masks = np.asarray(
        [kernel.transition_mask for kernel in compiled._simple_hopping_kernels],
        dtype=np.uint64,
    )
    simple_hopping_coefficients = np.asarray(
        [
            compiled._coerce_coefficient(kernel.coefficient)
            for kernel in compiled._simple_hopping_kernels
        ],
        dtype=coefficient_dtype,
    )

    parity_free_hopping_low_bits = np.asarray(
        [kernel.low_bit for kernel in compiled._parity_free_hopping_kernels],
        dtype=np.uint64,
    )
    parity_free_hopping_high_bits = np.asarray(
        [kernel.high_bit for kernel in compiled._parity_free_hopping_kernels],
        dtype=np.uint64,
    )
    parity_free_hopping_transition_masks = np.asarray(
        [kernel.transition_mask for kernel in compiled._parity_free_hopping_kernels],
        dtype=np.uint64,
    )
    parity_free_hopping_low_from_high = np.asarray(
        [
            compiled._coerce_coefficient(kernel.low_from_high)
            for kernel in compiled._parity_free_hopping_kernels
        ],
        dtype=coefficient_dtype,
    )
    parity_free_hopping_high_from_low = np.asarray(
        [
            compiled._coerce_coefficient(kernel.high_from_low)
            for kernel in compiled._parity_free_hopping_kernels
        ],
        dtype=coefficient_dtype,
    )

    signed_hopping_low_bits = np.asarray(
        [kernel.low_bit for kernel in compiled._signed_hopping_kernels],
        dtype=np.uint64,
    )
    signed_hopping_high_bits = np.asarray(
        [kernel.high_bit for kernel in compiled._signed_hopping_kernels],
        dtype=np.uint64,
    )
    signed_hopping_transition_masks = np.asarray(
        [kernel.transition_mask for kernel in compiled._signed_hopping_kernels],
        dtype=np.uint64,
    )
    signed_hopping_parity_masks = np.asarray(
        [kernel.parity_mask for kernel in compiled._signed_hopping_kernels],
        dtype=np.uint64,
    )
    signed_hopping_low_from_high = np.asarray(
        [
            compiled._coerce_coefficient(kernel.low_from_high)
            for kernel in compiled._signed_hopping_kernels
        ],
        dtype=coefficient_dtype,
    )
    signed_hopping_high_from_low = np.asarray(
        [
            compiled._coerce_coefficient(kernel.high_from_low)
            for kernel in compiled._signed_hopping_kernels
        ],
        dtype=coefficient_dtype,
    )

    number_conserving_monomials = []
    for kernel in compiled._monomial_kernels:
        annihilated = (
            kernel.transition_mask & kernel.required_occupied_mask
        ).bit_count()
        created = (
            kernel.transition_mask & kernel.required_empty_mask
        ).bit_count()
        if created == annihilated:
            number_conserving_monomials.append(kernel)

    monomial_occupied_masks = np.asarray(
        [kernel.required_occupied_mask for kernel in number_conserving_monomials],
        dtype=np.uint64,
    )
    monomial_empty_masks = np.asarray(
        [kernel.required_empty_mask for kernel in number_conserving_monomials],
        dtype=np.uint64,
    )
    monomial_transition_masks = np.asarray(
        [kernel.transition_mask for kernel in number_conserving_monomials],
        dtype=np.uint64,
    )
    monomial_parity_masks = np.asarray(
        [kernel.parity_mask for kernel in number_conserving_monomials],
        dtype=np.uint64,
    )
    monomial_coefficients = np.asarray(
        [
            compiled._coerce_coefficient(kernel.coefficient)
            for kernel in number_conserving_monomials
        ],
        dtype=coefficient_dtype,
    )

    plan = _NumbaOperatorPlan(
        number_masks=number_masks,
        number_coefficients=number_coefficients,
        simple_hopping_transition_masks=simple_hopping_transition_masks,
        simple_hopping_coefficients=simple_hopping_coefficients,
        parity_free_hopping_low_bits=parity_free_hopping_low_bits,
        parity_free_hopping_high_bits=parity_free_hopping_high_bits,
        parity_free_hopping_transition_masks=parity_free_hopping_transition_masks,
        parity_free_hopping_low_from_high=parity_free_hopping_low_from_high,
        parity_free_hopping_high_from_low=parity_free_hopping_high_from_low,
        signed_hopping_low_bits=signed_hopping_low_bits,
        signed_hopping_high_bits=signed_hopping_high_bits,
        signed_hopping_transition_masks=signed_hopping_transition_masks,
        signed_hopping_parity_masks=signed_hopping_parity_masks,
        signed_hopping_low_from_high=signed_hopping_low_from_high,
        signed_hopping_high_from_low=signed_hopping_high_from_low,
        monomial_occupied_masks=monomial_occupied_masks,
        monomial_empty_masks=monomial_empty_masks,
        monomial_transition_masks=monomial_transition_masks,
        monomial_parity_masks=monomial_parity_masks,
        monomial_coefficients=monomial_coefficients,
    )
    compiled._numba_operator_plan = plan
    return plan


def _get_execution_plan(
    compiled: "CompiledOperator",
    basis: "FockBasis",
) -> _NumbaExecutionPlan:
    """Return cached execution data for one operator/basis pair.

    Parameters
    ----------
    compiled : CompiledOperator
        Lowered operator whose packed masks and coefficients are shared across
        repeated matrix-vector products.
    basis : FockBasis
        Complete fixed-particle basis. Its native occupation words determine
        the execution ordering and its ``(n_modes, N)`` pair determines the
        direct combinadic rank tables.

    Returns
    -------
    _NumbaExecutionPlan
        Reusable basis states, operator arrays, and destination-rank tables.
    """
    cache = getattr(compiled, "_numba_execution_plans", None)
    if cache is None:
        cache = {}
        compiled._numba_execution_plans = cache

    plan = cache.get(basis)
    if plan is not None:
        return plan

    operator = _pack_operator_plan(compiled)
    needs_rank = (
        operator.simple_hopping_transition_masks.size
        + operator.parity_free_hopping_transition_masks.size
        + operator.signed_hopping_transition_masks.size
        + operator.monomial_transition_masks.size
    ) != 0
    if needs_rank:
        rank_contributions, byte_popcounts = _complete_rank_tables(
            basis.n_modes, basis.N
        )
    else:
        rank_contributions = np.empty((0, 0, 0), dtype=np.int64)
        byte_popcounts = np.empty(0, dtype=np.uint8)

    plan = _NumbaExecutionPlan(
        operator=operator,
        states=np.asarray(basis._execution_states(), dtype=np.uint64),
        rank_contributions=rank_contributions,
        byte_popcounts=byte_popcounts,
    )
    cache[basis] = plan
    return plan


def _coefficient_arrays(
    plan: _NumbaOperatorPlan,
    dtype: np.dtype,
) -> tuple[NDArray[np.generic], ...]:
    """Return plan coefficients cast to the current vector result dtype."""
    return (
        np.asarray(plan.number_coefficients, dtype=dtype),
        np.asarray(plan.simple_hopping_coefficients, dtype=dtype),
        np.asarray(plan.parity_free_hopping_low_from_high, dtype=dtype),
        np.asarray(plan.parity_free_hopping_high_from_low, dtype=dtype),
        np.asarray(plan.signed_hopping_low_from_high, dtype=dtype),
        np.asarray(plan.signed_hopping_high_from_low, dtype=dtype),
        np.asarray(plan.monomial_coefficients, dtype=dtype),
    )


def _hopping_kernel_count(plan: _NumbaOperatorPlan) -> int:
    """Return the number of packed one-body hopping kernels in ``plan``."""
    return int(
        plan.simple_hopping_transition_masks.size
        + plan.parity_free_hopping_transition_masks.size
        + plan.signed_hopping_transition_masks.size
    )


if njit is not None:

    @njit(cache=True, nogil=True, inline="always")
    def _parity_uint64(value):
        value ^= value >> np.uint64(32)
        value ^= value >> np.uint64(16)
        value ^= value >> np.uint64(8)
        value ^= value >> np.uint64(4)
        value ^= value >> np.uint64(2)
        value ^= value >> np.uint64(1)
        return value & np.uint64(1)


    @njit(cache=True, nogil=True, inline="always")
    def _complete_rank_uint64(state, rank_contributions, byte_popcounts):
        """Return complete-basis rank with cached bytewise combinadic lookup."""
        rank = np.int64(0)
        order = 1
        value = state
        for byte_index in range(rank_contributions.shape[0]):
            byte_value = int(value & np.uint64(0xFF))
            rank += rank_contributions[byte_index, order, byte_value]
            order += int(byte_popcounts[byte_value])
            value >>= np.uint64(8)
        return rank


    @njit(cache=True, nogil=True)
    def _matvec_numba_serial_diagonal_kernel(
        states,
        vector,
        number_masks,
        number_coefficients,
    ):
        """Apply an arbitrary pure diagonal number-product Hamiltonian.

        Pure diagonal plans have no scatter writes, so each output element is
        private to one source state.  Let Numba optimize repeated updates to
        that element directly instead of forcing a long scalar reduction
        dependency chain.  Mixed kernels retain scalar diagonal accumulation
        because their output element also participates in off-diagonal work.
        """
        result = np.zeros(vector.size, dtype=vector.dtype)
        if number_masks.size == 0:
            return result

        for column in range(states.size):
            input_amplitude = vector[column]
            if input_amplitude == 0:
                continue
            state = states[column]

            for index in range(number_masks.size):
                mask = number_masks[index]
                if state & mask == mask:
                    result[column] += (
                        number_coefficients[index] * input_amplitude
                    )

        return result


    @njit(cache=True, nogil=True)
    def _matvec_numba_serial_pure_monomial_kernel(
        states,
        vector,
        rank_contributions,
        byte_popcounts,
        monomial_occupied_masks,
        monomial_empty_masks,
        monomial_transition_masks,
        monomial_parity_masks,
        monomial_coefficients,
    ):
        """Apply a plan containing only number-conserving monomials."""
        result = np.zeros(vector.size, dtype=vector.dtype)

        for column in range(states.size):
            input_amplitude = vector[column]
            if input_amplitude == 0:
                continue
            state = states[column]

            for index in range(monomial_transition_masks.size):
                occupied = monomial_occupied_masks[index]
                if state & occupied != occupied:
                    continue
                empty = monomial_empty_masks[index]
                if state & empty:
                    continue

                final_state = state ^ monomial_transition_masks[index]
                row = _complete_rank_uint64(
                    final_state, rank_contributions, byte_popcounts
                )
                amplitude = monomial_coefficients[index] * input_amplitude
                parity_mask = monomial_parity_masks[index]
                if parity_mask != 0 and _parity_uint64(state & parity_mask) != 0:
                    amplitude = -amplitude
                result[row] += amplitude

        return result


    @njit(cache=True, nogil=True)
    def _matvec_numba_serial_diagonal_monomial_kernel(
        states,
        vector,
        rank_contributions,
        byte_popcounts,
        number_masks,
        number_coefficients,
        monomial_occupied_masks,
        monomial_empty_masks,
        monomial_transition_masks,
        monomial_parity_masks,
        monomial_coefficients,
    ):
        """Apply a plan containing diagonal and number-conserving monomial terms."""
        result = np.zeros(vector.size, dtype=vector.dtype)

        for column in range(states.size):
            input_amplitude = vector[column]
            if input_amplitude == 0:
                continue
            state = states[column]

            if number_masks.size != 0:
                diagonal = number_coefficients[0] * 0
                for index in range(number_masks.size):
                    mask = number_masks[index]
                    if state & mask == mask:
                        diagonal += number_coefficients[index]
                result[column] += diagonal * input_amplitude

            for index in range(monomial_transition_masks.size):
                occupied = monomial_occupied_masks[index]
                if state & occupied != occupied:
                    continue
                empty = monomial_empty_masks[index]
                if state & empty:
                    continue

                final_state = state ^ monomial_transition_masks[index]
                row = _complete_rank_uint64(
                    final_state, rank_contributions, byte_popcounts
                )
                amplitude = monomial_coefficients[index] * input_amplitude
                parity_mask = monomial_parity_masks[index]
                if parity_mask != 0 and _parity_uint64(state & parity_mask) != 0:
                    amplitude = -amplitude
                result[row] += amplitude

        return result


    @njit(cache=True, nogil=True)
    def _matvec_numba_serial_pure_hopping_kernel(
        states,
        vector,
        rank_contributions,
        byte_popcounts,
        simple_hopping_transition_masks,
        simple_hopping_coefficients,
        parity_free_hopping_low_bits,
        parity_free_hopping_high_bits,
        parity_free_hopping_transition_masks,
        parity_free_hopping_low_from_high,
        parity_free_hopping_high_from_low,
        signed_hopping_low_bits,
        signed_hopping_high_bits,
        signed_hopping_transition_masks,
        signed_hopping_parity_masks,
        signed_hopping_low_from_high,
        signed_hopping_high_from_low,
    ):
        """Apply a plan containing only specialized one-body hopping."""
        result = np.zeros(vector.size, dtype=vector.dtype)

        for column in range(states.size):
            input_amplitude = vector[column]
            if input_amplitude == 0:
                continue
            state = states[column]

            for index in range(simple_hopping_transition_masks.size):
                transition_mask = simple_hopping_transition_masks[index]
                occupation = state & transition_mask
                if occupation == 0 or occupation == transition_mask:
                    continue
                row = _complete_rank_uint64(
                    state ^ transition_mask, rank_contributions, byte_popcounts
                )
                result[row] += simple_hopping_coefficients[index] * input_amplitude

            for index in range(parity_free_hopping_transition_masks.size):
                transition_mask = parity_free_hopping_transition_masks[index]
                occupation = state & transition_mask
                coefficient = parity_free_hopping_low_from_high[index]
                if occupation == parity_free_hopping_low_bits[index]:
                    coefficient = parity_free_hopping_high_from_low[index]
                elif occupation != parity_free_hopping_high_bits[index]:
                    continue
                if coefficient == 0:
                    continue
                row = _complete_rank_uint64(
                    state ^ transition_mask, rank_contributions, byte_popcounts
                )
                result[row] += coefficient * input_amplitude

            for index in range(signed_hopping_transition_masks.size):
                transition_mask = signed_hopping_transition_masks[index]
                occupation = state & transition_mask
                coefficient = signed_hopping_low_from_high[index]
                if occupation == signed_hopping_low_bits[index]:
                    coefficient = signed_hopping_high_from_low[index]
                elif occupation != signed_hopping_high_bits[index]:
                    continue
                if coefficient == 0:
                    continue

                row = _complete_rank_uint64(
                    state ^ transition_mask, rank_contributions, byte_popcounts
                )
                amplitude = coefficient * input_amplitude
                if _parity_uint64(
                    state & signed_hopping_parity_masks[index]
                ) != 0:
                    amplitude = -amplitude
                result[row] += amplitude

        return result


    @njit(cache=True, nogil=True)
    def _matvec_numba_serial_mixed_kernel(
        states,
        vector,
        rank_contributions,
        byte_popcounts,
        number_masks,
        number_coefficients,
        simple_hopping_transition_masks,
        simple_hopping_coefficients,
        parity_free_hopping_low_bits,
        parity_free_hopping_high_bits,
        parity_free_hopping_transition_masks,
        parity_free_hopping_low_from_high,
        parity_free_hopping_high_from_low,
        signed_hopping_low_bits,
        signed_hopping_high_bits,
        signed_hopping_transition_masks,
        signed_hopping_parity_masks,
        signed_hopping_low_from_high,
        signed_hopping_high_from_low,
        monomial_occupied_masks,
        monomial_empty_masks,
        monomial_transition_masks,
        monomial_parity_masks,
        monomial_coefficients,
    ):
        """Apply a mixed plan containing diagonal, hopping, or monomial kernels."""
        result = np.zeros(vector.size, dtype=vector.dtype)

        for column in range(states.size):
            input_amplitude = vector[column]
            if input_amplitude == 0:
                continue
            state = states[column]

            if number_masks.size != 0:
                diagonal = number_coefficients[0] * 0
                for index in range(number_masks.size):
                    mask = number_masks[index]
                    if state & mask == mask:
                        diagonal += number_coefficients[index]
                result[column] += diagonal * input_amplitude

            for index in range(simple_hopping_transition_masks.size):
                transition_mask = simple_hopping_transition_masks[index]
                occupation = state & transition_mask
                if occupation == 0 or occupation == transition_mask:
                    continue

                final_state = state ^ transition_mask
                row = _complete_rank_uint64(
                    final_state, rank_contributions, byte_popcounts
                )
                result[row] += simple_hopping_coefficients[index] * input_amplitude

            for index in range(parity_free_hopping_transition_masks.size):
                transition_mask = parity_free_hopping_transition_masks[index]
                occupation = state & transition_mask
                coefficient = parity_free_hopping_low_from_high[index]
                if occupation == parity_free_hopping_low_bits[index]:
                    coefficient = parity_free_hopping_high_from_low[index]
                elif occupation != parity_free_hopping_high_bits[index]:
                    continue
                if coefficient == 0:
                    continue

                final_state = state ^ transition_mask
                row = _complete_rank_uint64(
                    final_state, rank_contributions, byte_popcounts
                )
                result[row] += coefficient * input_amplitude

            for index in range(signed_hopping_transition_masks.size):
                transition_mask = signed_hopping_transition_masks[index]
                occupation = state & transition_mask
                coefficient = signed_hopping_low_from_high[index]
                if occupation == signed_hopping_low_bits[index]:
                    coefficient = signed_hopping_high_from_low[index]
                elif occupation != signed_hopping_high_bits[index]:
                    continue
                if coefficient == 0:
                    continue

                final_state = state ^ transition_mask
                row = _complete_rank_uint64(
                    final_state, rank_contributions, byte_popcounts
                )
                amplitude = coefficient * input_amplitude
                if _parity_uint64(
                    state & signed_hopping_parity_masks[index]
                ) != 0:
                    amplitude = -amplitude
                result[row] += amplitude

            for index in range(monomial_transition_masks.size):
                occupied = monomial_occupied_masks[index]
                if state & occupied != occupied:
                    continue
                empty = monomial_empty_masks[index]
                if state & empty:
                    continue

                final_state = state ^ monomial_transition_masks[index]
                row = _complete_rank_uint64(
                    final_state, rank_contributions, byte_popcounts
                )
                amplitude = monomial_coefficients[index] * input_amplitude
                parity_mask = monomial_parity_masks[index]
                if parity_mask != 0 and _parity_uint64(state & parity_mask) != 0:
                    amplitude = -amplitude
                result[row] += amplitude

        return result


    @njit(cache=True, nogil=True, parallel=True)
    def _matvec_numba_parallel_diagonal_kernel(
        states,
        vector,
        number_masks,
        number_coefficients,
    ):
        """Apply a pure diagonal plan with independent output rows."""
        result = np.zeros(vector.size, dtype=vector.dtype)
        for row in prange(states.size):
            input_amplitude = vector[row]
            if input_amplitude == 0:
                continue
            state = states[row]
            for index in range(number_masks.size):
                mask = number_masks[index]
                if state & mask == mask:
                    result[row] += number_coefficients[index] * input_amplitude
        return result


    @njit(cache=True, nogil=True, parallel=True)
    def _matvec_numba_parallel_pure_monomial_kernel(
        states,
        vector,
        rank_contributions,
        byte_popcounts,
        monomial_occupied_masks,
        monomial_empty_masks,
        monomial_transition_masks,
        monomial_parity_masks,
        monomial_coefficients,
    ):
        """Apply a pure monomial plan with row-parallel gather."""
        result = np.zeros(vector.size, dtype=vector.dtype)
        for row in prange(states.size):
            state = states[row]
            value = vector[row] * 0
            for index in range(monomial_transition_masks.size):
                source_state = state ^ monomial_transition_masks[index]
                occupied = monomial_occupied_masks[index]
                if source_state & occupied != occupied:
                    continue
                empty = monomial_empty_masks[index]
                if source_state & empty:
                    continue
                column = _complete_rank_uint64(
                    source_state, rank_contributions, byte_popcounts
                )
                source_amplitude = vector[column]
                if source_amplitude == 0:
                    continue
                amplitude = monomial_coefficients[index] * source_amplitude
                parity_mask = monomial_parity_masks[index]
                if (
                    parity_mask != 0
                    and _parity_uint64(source_state & parity_mask) != 0
                ):
                    amplitude = -amplitude
                value += amplitude
            result[row] = value
        return result


    @njit(cache=True, nogil=True, parallel=True)
    def _matvec_numba_parallel_pure_hopping_kernel(
        states,
        vector,
        rank_contributions,
        byte_popcounts,
        simple_hopping_transition_masks,
        simple_hopping_coefficients,
        parity_free_hopping_low_bits,
        parity_free_hopping_high_bits,
        parity_free_hopping_transition_masks,
        parity_free_hopping_low_from_high,
        parity_free_hopping_high_from_low,
        signed_hopping_low_bits,
        signed_hopping_high_bits,
        signed_hopping_transition_masks,
        signed_hopping_parity_masks,
        signed_hopping_low_from_high,
        signed_hopping_high_from_low,
    ):
        """Apply a pure hopping plan with row-parallel gather."""
        result = np.zeros(vector.size, dtype=vector.dtype)
        for row in prange(states.size):
            state = states[row]
            value = vector[row] * 0

            for index in range(simple_hopping_transition_masks.size):
                transition_mask = simple_hopping_transition_masks[index]
                occupation = state & transition_mask
                if occupation == 0 or occupation == transition_mask:
                    continue
                source_state = state ^ transition_mask
                column = _complete_rank_uint64(
                    source_state, rank_contributions, byte_popcounts
                )
                source_amplitude = vector[column]
                if source_amplitude != 0:
                    value += simple_hopping_coefficients[index] * source_amplitude

            for index in range(parity_free_hopping_transition_masks.size):
                transition_mask = parity_free_hopping_transition_masks[index]
                occupation = state & transition_mask
                if occupation == parity_free_hopping_high_bits[index]:
                    coefficient = parity_free_hopping_high_from_low[index]
                elif occupation == parity_free_hopping_low_bits[index]:
                    coefficient = parity_free_hopping_low_from_high[index]
                else:
                    continue
                if coefficient == 0:
                    continue
                source_state = state ^ transition_mask
                column = _complete_rank_uint64(
                    source_state, rank_contributions, byte_popcounts
                )
                source_amplitude = vector[column]
                if source_amplitude != 0:
                    value += coefficient * source_amplitude

            for index in range(signed_hopping_transition_masks.size):
                transition_mask = signed_hopping_transition_masks[index]
                occupation = state & transition_mask
                if occupation == signed_hopping_high_bits[index]:
                    coefficient = signed_hopping_high_from_low[index]
                elif occupation == signed_hopping_low_bits[index]:
                    coefficient = signed_hopping_low_from_high[index]
                else:
                    continue
                if coefficient == 0:
                    continue
                source_state = state ^ transition_mask
                column = _complete_rank_uint64(
                    source_state, rank_contributions, byte_popcounts
                )
                source_amplitude = vector[column]
                if source_amplitude == 0:
                    continue
                amplitude = coefficient * source_amplitude
                if _parity_uint64(
                    source_state & signed_hopping_parity_masks[index]
                ) != 0:
                    amplitude = -amplitude
                value += amplitude

            result[row] = value
        return result


    @njit(cache=True, nogil=True, parallel=True)
    def _matvec_numba_parallel_mixed_kernel(
        states,
        vector,
        rank_contributions,
        byte_popcounts,
        number_masks,
        number_coefficients,
        simple_hopping_transition_masks,
        simple_hopping_coefficients,
        parity_free_hopping_low_bits,
        parity_free_hopping_high_bits,
        parity_free_hopping_transition_masks,
        parity_free_hopping_low_from_high,
        parity_free_hopping_high_from_low,
        signed_hopping_low_bits,
        signed_hopping_high_bits,
        signed_hopping_transition_masks,
        signed_hopping_parity_masks,
        signed_hopping_low_from_high,
        signed_hopping_high_from_low,
        monomial_occupied_masks,
        monomial_empty_masks,
        monomial_transition_masks,
        monomial_parity_masks,
        monomial_coefficients,
    ):
        """Apply a mixed complete-basis plan with race-free row-parallel gather.

        Every ``prange`` iteration owns one output row.  All off-diagonal
        transitions are inverted to their source occupation word, so threads
        only read the input vector and immutable execution plan while writing
        one private ``result[row]``.
        """
        result = np.zeros(vector.size, dtype=vector.dtype)

        for row in prange(states.size):
            state = states[row]
            value = vector[row] * 0

            input_amplitude = vector[row]
            if input_amplitude != 0:
                for index in range(number_masks.size):
                    mask = number_masks[index]
                    if state & mask == mask:
                        value += number_coefficients[index] * input_amplitude

            for index in range(simple_hopping_transition_masks.size):
                transition_mask = simple_hopping_transition_masks[index]
                occupation = state & transition_mask
                if occupation == 0 or occupation == transition_mask:
                    continue
                source_state = state ^ transition_mask
                column = _complete_rank_uint64(
                    source_state, rank_contributions, byte_popcounts
                )
                source_amplitude = vector[column]
                if source_amplitude != 0:
                    value += (
                        simple_hopping_coefficients[index] * source_amplitude
                    )

            for index in range(parity_free_hopping_transition_masks.size):
                transition_mask = parity_free_hopping_transition_masks[index]
                occupation = state & transition_mask
                if occupation == parity_free_hopping_high_bits[index]:
                    coefficient = parity_free_hopping_high_from_low[index]
                elif occupation == parity_free_hopping_low_bits[index]:
                    coefficient = parity_free_hopping_low_from_high[index]
                else:
                    continue
                if coefficient == 0:
                    continue
                source_state = state ^ transition_mask
                column = _complete_rank_uint64(
                    source_state, rank_contributions, byte_popcounts
                )
                source_amplitude = vector[column]
                if source_amplitude != 0:
                    value += coefficient * source_amplitude

            for index in range(signed_hopping_transition_masks.size):
                transition_mask = signed_hopping_transition_masks[index]
                occupation = state & transition_mask
                if occupation == signed_hopping_high_bits[index]:
                    coefficient = signed_hopping_high_from_low[index]
                elif occupation == signed_hopping_low_bits[index]:
                    coefficient = signed_hopping_low_from_high[index]
                else:
                    continue
                if coefficient == 0:
                    continue
                source_state = state ^ transition_mask
                column = _complete_rank_uint64(
                    source_state, rank_contributions, byte_popcounts
                )
                source_amplitude = vector[column]
                if source_amplitude == 0:
                    continue
                amplitude = coefficient * source_amplitude
                if _parity_uint64(
                    source_state & signed_hopping_parity_masks[index]
                ) != 0:
                    amplitude = -amplitude
                value += amplitude

            for index in range(monomial_transition_masks.size):
                transition_mask = monomial_transition_masks[index]
                source_state = state ^ transition_mask
                occupied = monomial_occupied_masks[index]
                if source_state & occupied != occupied:
                    continue
                empty = monomial_empty_masks[index]
                if source_state & empty:
                    continue

                column = _complete_rank_uint64(
                    source_state, rank_contributions, byte_popcounts
                )
                source_amplitude = vector[column]
                if source_amplitude == 0:
                    continue
                amplitude = monomial_coefficients[index] * source_amplitude
                parity_mask = monomial_parity_masks[index]
                if (
                    parity_mask != 0
                    and _parity_uint64(source_state & parity_mask) != 0
                ):
                    amplitude = -amplitude
                value += amplitude

            result[row] = value

        return result



else:
    _matvec_numba_serial_diagonal_kernel = None
    _matvec_numba_serial_pure_monomial_kernel = None
    _matvec_numba_serial_diagonal_monomial_kernel = None
    _matvec_numba_serial_pure_hopping_kernel = None
    _matvec_numba_serial_mixed_kernel = None
    _matvec_numba_parallel_diagonal_kernel = None
    _matvec_numba_parallel_pure_monomial_kernel = None
    _matvec_numba_parallel_pure_hopping_kernel = None
    _matvec_numba_parallel_mixed_kernel = None


def matvec_numba_serial(
    compiled: "CompiledOperator",
    basis: "FockBasis",
    vector: NDArray[np.generic],
) -> NDArray[np.generic]:
    """Apply a supported fermion Hamiltonian with serial Numba execution.

    Parameters
    ----------
    compiled : CompiledOperator
        Fully lowered fermion operator produced by the symbolic compiler.
    basis : FockBasis
        Complete fixed-particle basis with at most 64 fermion modes.
    vector : numpy.ndarray
        One-dimensional input state. Its length must equal ``basis.dimension``;
        real and complex numerical dtypes are accepted.

    Returns
    -------
    numpy.ndarray
        Hamiltonian action in the basis ordering. Serial execution uses
        source-state scatter and direct combinadic destination ranking.
    """
    require_numba_supported(compiled, basis, "numba-serial")

    dimension = int(basis.dimension)
    vector = compiled._validate_matvec_input(vector, dimension)
    dtype = np.result_type(vector.dtype, compiled.data_dtype)
    vector = np.asarray(vector, dtype=dtype)
    execution_plan = _get_execution_plan(compiled, basis)
    plan = execution_plan.operator
    coefficient_dtype = np.dtype(dtype)
    (
        number_coefficients,
        simple_coefficients,
        parity_low_from_high,
        parity_high_from_low,
        signed_low_from_high,
        signed_high_from_low,
        monomial_coefficients,
    ) = _coefficient_arrays(plan, coefficient_dtype)

    states = execution_plan.states
    rank_contributions = execution_plan.rank_contributions
    byte_popcounts = execution_plan.byte_popcounts
    hopping_count = _hopping_kernel_count(plan)
    monomial_count = int(plan.monomial_transition_masks.size)

    if hopping_count == 0 and monomial_count == 0:
        return _matvec_numba_serial_diagonal_kernel(
            states,
            vector,
            plan.number_masks,
            number_coefficients,
        )

    if hopping_count == 0:
        if plan.number_masks.size == 0:
            return _matvec_numba_serial_pure_monomial_kernel(
                states,
                vector,
                rank_contributions,
                byte_popcounts,
                plan.monomial_occupied_masks,
                plan.monomial_empty_masks,
                plan.monomial_transition_masks,
                plan.monomial_parity_masks,
                monomial_coefficients,
            )
        return _matvec_numba_serial_diagonal_monomial_kernel(
            states,
            vector,
            rank_contributions,
            byte_popcounts,
            plan.number_masks,
            number_coefficients,
            plan.monomial_occupied_masks,
            plan.monomial_empty_masks,
            plan.monomial_transition_masks,
            plan.monomial_parity_masks,
            monomial_coefficients,
        )

    if plan.number_masks.size == 0 and monomial_count == 0:
        return _matvec_numba_serial_pure_hopping_kernel(
            states,
            vector,
            rank_contributions,
            byte_popcounts,
            plan.simple_hopping_transition_masks,
            simple_coefficients,
            plan.parity_free_hopping_low_bits,
            plan.parity_free_hopping_high_bits,
            plan.parity_free_hopping_transition_masks,
            parity_low_from_high,
            parity_high_from_low,
            plan.signed_hopping_low_bits,
            plan.signed_hopping_high_bits,
            plan.signed_hopping_transition_masks,
            plan.signed_hopping_parity_masks,
            signed_low_from_high,
            signed_high_from_low,
        )

    return _matvec_numba_serial_mixed_kernel(
        states,
        vector,
        rank_contributions,
        byte_popcounts,
        plan.number_masks,
        number_coefficients,
        plan.simple_hopping_transition_masks,
        simple_coefficients,
        plan.parity_free_hopping_low_bits,
        plan.parity_free_hopping_high_bits,
        plan.parity_free_hopping_transition_masks,
        parity_low_from_high,
        parity_high_from_low,
        plan.signed_hopping_low_bits,
        plan.signed_hopping_high_bits,
        plan.signed_hopping_transition_masks,
        plan.signed_hopping_parity_masks,
        signed_low_from_high,
        signed_high_from_low,
        plan.monomial_occupied_masks,
        plan.monomial_empty_masks,
        plan.monomial_transition_masks,
        plan.monomial_parity_masks,
        monomial_coefficients,
    )


def matvec_numba_parallel(
    compiled: "CompiledOperator",
    basis: "FockBasis",
    vector: NDArray[np.generic],
) -> NDArray[np.generic]:
    """Apply a supported fermion Hamiltonian with parallel Numba execution.

    Parameters
    ----------
    compiled : CompiledOperator
        Fully lowered fermion operator produced by the symbolic compiler.
    basis : FockBasis
        Complete fixed-particle basis with at most 64 fermion modes.
    vector : numpy.ndarray
        One-dimensional input state. Its length must equal ``basis.dimension``;
        real and complex numerical dtypes are accepted.

    Returns
    -------
    numpy.ndarray
        Hamiltonian action in the basis ordering. Parallel execution assigns
        each output row to one ``prange`` iteration and gathers all contributing
        source amplitudes, so no thread writes to another row.

    Notes
    -----
    Thread count is controlled by Numba. Applications may use
    ``numba.set_num_threads`` or ``NUMBA_NUM_THREADS`` before the matrix-vector
    product. EDinPy does not change the configured thread count automatically.
    """
    require_numba_supported(compiled, basis, "numba-parallel")

    dimension = int(basis.dimension)
    vector = compiled._validate_matvec_input(vector, dimension)
    dtype = np.result_type(vector.dtype, compiled.data_dtype)
    vector = np.asarray(vector, dtype=dtype)
    execution_plan = _get_execution_plan(compiled, basis)
    plan = execution_plan.operator
    coefficient_dtype = np.dtype(dtype)
    (
        number_coefficients,
        simple_coefficients,
        parity_low_from_high,
        parity_high_from_low,
        signed_low_from_high,
        signed_high_from_low,
        monomial_coefficients,
    ) = _coefficient_arrays(plan, coefficient_dtype)

    hopping_count = _hopping_kernel_count(plan)
    monomial_count = int(plan.monomial_transition_masks.size)

    if hopping_count == 0 and monomial_count == 0:
        return _matvec_numba_parallel_diagonal_kernel(
            execution_plan.states,
            vector,
            plan.number_masks,
            number_coefficients,
        )

    if hopping_count == 0 and plan.number_masks.size == 0:
        return _matvec_numba_parallel_pure_monomial_kernel(
            execution_plan.states,
            vector,
            execution_plan.rank_contributions,
            execution_plan.byte_popcounts,
            plan.monomial_occupied_masks,
            plan.monomial_empty_masks,
            plan.monomial_transition_masks,
            plan.monomial_parity_masks,
            monomial_coefficients,
        )

    if hopping_count != 0 and plan.number_masks.size == 0 and monomial_count == 0:
        return _matvec_numba_parallel_pure_hopping_kernel(
            execution_plan.states,
            vector,
            execution_plan.rank_contributions,
            execution_plan.byte_popcounts,
            plan.simple_hopping_transition_masks,
            simple_coefficients,
            plan.parity_free_hopping_low_bits,
            plan.parity_free_hopping_high_bits,
            plan.parity_free_hopping_transition_masks,
            parity_low_from_high,
            parity_high_from_low,
            plan.signed_hopping_low_bits,
            plan.signed_hopping_high_bits,
            plan.signed_hopping_transition_masks,
            plan.signed_hopping_parity_masks,
            signed_low_from_high,
            signed_high_from_low,
        )

    return _matvec_numba_parallel_mixed_kernel(
        execution_plan.states,
        vector,
        execution_plan.rank_contributions,
        execution_plan.byte_popcounts,
        plan.number_masks,
        number_coefficients,
        plan.simple_hopping_transition_masks,
        simple_coefficients,
        plan.parity_free_hopping_low_bits,
        plan.parity_free_hopping_high_bits,
        plan.parity_free_hopping_transition_masks,
        parity_low_from_high,
        parity_high_from_low,
        plan.signed_hopping_low_bits,
        plan.signed_hopping_high_bits,
        plan.signed_hopping_transition_masks,
        plan.signed_hopping_parity_masks,
        signed_low_from_high,
        signed_high_from_low,
        plan.monomial_occupied_masks,
        plan.monomial_empty_masks,
        plan.monomial_transition_masks,
        plan.monomial_parity_masks,
        monomial_coefficients,
    )
