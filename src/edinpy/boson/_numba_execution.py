"""Numba matrix-free execution for fixed-particle boson sectors."""

from __future__ import annotations

from dataclasses import dataclass
from math import comb
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from ._basis import FockBasis
    from ._execution import CompiledOperator

try:  # Optional dependency used only by explicit Numba execution modes.
    from numba import njit, prange
except ImportError:  # pragma: no cover - installation without the optional extra.
    njit = None
    prange = None


@dataclass(frozen=True, slots=True)
class _BosonOperatorPlan:
    """Homogeneous arrays describing one fully lowered boson operator."""

    number_coefficients: NDArray[np.generic]
    number_offsets: NDArray[np.int64]
    number_modes: NDArray[np.int64]
    number_powers: NDArray[np.int64]
    hopping_low: NDArray[np.int64]
    hopping_high: NDArray[np.int64]
    hopping_low_from_high: NDArray[np.generic]
    hopping_high_from_low: NDArray[np.generic]
    pair_low: NDArray[np.int64]
    pair_high: NDArray[np.int64]
    pair_low_from_high: NDArray[np.generic]
    pair_high_from_low: NDArray[np.generic]
    monomial_coefficients: NDArray[np.generic]
    monomial_mode_offsets: NDArray[np.int64]
    monomial_modes: NDArray[np.int64]
    monomial_net_deltas: NDArray[np.int64]
    monomial_operation_offsets: NDArray[np.int64]
    monomial_operation_modes: NDArray[np.int64]
    monomial_operation_kinds: NDArray[np.int8]
    monomial_operation_adjustments: NDArray[np.int64]
    monomial_boundary_start: NDArray[np.int64]
    monomial_boundary_stop: NDArray[np.int64]
    monomial_boundary_deltas: NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class _BosonExecutionPlan:
    """Cached operator and basis data reused by repeated bosonic matvec calls."""

    operator: _BosonOperatorPlan
    cumulative_populations: NDArray[np.integer]
    complete_ranks: NDArray[np.int64]
    rank_contributions: NDArray[np.int64]
    complete_basis: bool
    total_particles: int
    n_modes: int


_NumMode = Literal["numba-serial", "numba-parallel"]


def numba_available() -> bool:
    """Return whether the optional Numba runtime can be imported."""
    return njit is not None


def _unsupported_reason(compiled: "CompiledOperator", basis: "FockBasis") -> str | None:
    """Return an explanatory reason when Numba execution is unavailable."""
    if not compiled.fully_lowered:
        return "generic symbolic fallback terms are present"
    if int(basis.N) > np.iinfo(np.int64).max:
        return "particle count exceeds signed 64-bit execution arithmetic"
    ranks = basis._complete_composition_ranks()
    if ranks is None:
        return "complete-basis ranks exceed signed 64-bit indexing"
    if not numba_available():
        return (
            "Numba is not installed. Install EDinPy with the optional 'numba' "
            "extra or use execution='numpy'."
        )
    return None


def require_numba_supported(
    compiled: "CompiledOperator", basis: "FockBasis", execution: _NumMode
) -> None:
    """Validate an explicit bosonic Numba execution request.

    Parameters
    ----------
    compiled : CompiledOperator
        Compiled bosonic Hamiltonian expression.
    basis : FockBasis
        Fixed-particle basis defining the matrix-free vector ordering.
    execution : {"numba-serial", "numba-parallel"}
        Strict Numba execution mode requested by the caller.

    Raises
    ------
    ValueError
        If ``execution`` is not a recognized strict Numba mode.
    NotImplementedError
        If the operator, basis, or installed dependencies cannot satisfy the
        requested execution mode.
    """
    if execution not in {"numba-serial", "numba-parallel"}:
        raise ValueError(f"Unknown strict Numba execution mode {execution!r}.")
    reason = _unsupported_reason(compiled, basis)
    if reason is not None:
        raise NotImplementedError(
            f"execution={execution!r} is unavailable for this Hamiltonian: "
            f"{reason}. Use execution='mixed' or execution='numpy'."
        )


def _flatten_number_kernels(compiled: "CompiledOperator"):
    offsets = [0]
    modes: list[int] = []
    powers: list[int] = []
    for kernel in compiled._number_kernels:
        for mode, power in kernel.powers:
            modes.append(int(mode))
            powers.append(int(power))
        offsets.append(len(modes))
    return offsets, modes, powers


def _pack_monomials(compiled: "CompiledOperator", n_modes: int):
    kernels = [
        kernel for kernel in compiled._monomial_kernels if kernel.particle_delta == 0
    ]
    mode_offsets = [0]
    modes: list[int] = []
    net_deltas: list[int] = []
    operation_offsets = [0]
    operation_modes: list[int] = []
    operation_kinds: list[int] = []
    operation_adjustments: list[int] = []
    boundary_start: list[int] = []
    boundary_stop: list[int] = []
    boundary_rows: list[list[int]] = []

    kind_code = {"annihilate": 0, "create": 1, "number": 2}

    for kernel in kernels:
        net_by_mode = {mode: 0 for mode in kernel.touched_modes}
        prefix_by_mode = {mode: 0 for mode in kernel.touched_modes}
        for kind, mode in kernel.operations:
            if kind == "create":
                net_by_mode[mode] += 1
            elif kind == "annihilate":
                net_by_mode[mode] -= 1

        for mode in kernel.touched_modes:
            modes.append(int(mode))
            net_deltas.append(int(net_by_mode[mode]))
        mode_offsets.append(len(modes))

        for kind, mode in kernel.operations:
            operation_modes.append(int(mode))
            operation_kinds.append(kind_code[kind])
            operation_adjustments.append(
                int(prefix_by_mode[mode] - net_by_mode[mode])
            )
            if kind == "create":
                prefix_by_mode[mode] += 1
            elif kind == "annihilate":
                prefix_by_mode[mode] -= 1
        operation_offsets.append(len(operation_modes))

        cumulative_delta = 0
        row = [0] * max(0, n_modes - 1)
        active = []
        for boundary in range(n_modes - 1):
            cumulative_delta += net_by_mode.get(boundary, 0)
            row[boundary] = cumulative_delta
            if cumulative_delta:
                active.append(boundary)
        if active:
            boundary_start.append(active[0])
            boundary_stop.append(active[-1] + 1)
        else:
            boundary_start.append(0)
            boundary_stop.append(0)
        boundary_rows.append(row)

    return (
        kernels,
        mode_offsets,
        modes,
        net_deltas,
        operation_offsets,
        operation_modes,
        operation_kinds,
        operation_adjustments,
        boundary_start,
        boundary_stop,
        boundary_rows,
    )


def _pack_operator_plan(
    compiled: "CompiledOperator", n_modes: int
) -> _BosonOperatorPlan:
    """Pack one lowered operator into homogeneous arrays for Numba.

    Parameters
    ----------
    compiled : CompiledOperator
        Fully lowered bosonic operator. Coefficients are packed using the
        operator's real or complex double-precision dtype.
    n_modes : int
        Number of flat bosonic modes in the execution basis. It fixes the
        cumulative-boundary representation used by general monomials.

    Returns
    -------
    _BosonOperatorPlan
        Immutable array representation reused by every compatible basis with
        the same number of modes.
    """
    cache = compiled._numba_operator_plans
    cached = cache.get(n_modes)
    if cached is not None:
        return cached

    dtype = np.dtype(compiled.data_dtype)
    number_offsets, number_modes, number_powers = _flatten_number_kernels(compiled)
    (
        monomials,
        mode_offsets,
        monomial_modes,
        monomial_net_deltas,
        operation_offsets,
        operation_modes,
        operation_kinds,
        operation_adjustments,
        boundary_start,
        boundary_stop,
        boundary_rows,
    ) = _pack_monomials(compiled, n_modes)

    plan = _BosonOperatorPlan(
        number_coefficients=np.asarray(
            [compiled._coerce_coefficient(k.coefficient) for k in compiled._number_kernels],
            dtype=dtype,
        ),
        number_offsets=np.asarray(number_offsets, dtype=np.int64),
        number_modes=np.asarray(number_modes, dtype=np.int64),
        number_powers=np.asarray(number_powers, dtype=np.int64),
        hopping_low=np.asarray([k.low for k in compiled._hopping_kernels], dtype=np.int64),
        hopping_high=np.asarray([k.high for k in compiled._hopping_kernels], dtype=np.int64),
        hopping_low_from_high=np.asarray(
            [compiled._coerce_coefficient(k.low_from_high) for k in compiled._hopping_kernels], dtype=dtype
        ),
        hopping_high_from_low=np.asarray(
            [compiled._coerce_coefficient(k.high_from_low) for k in compiled._hopping_kernels], dtype=dtype
        ),
        pair_low=np.asarray([k.low for k in compiled._pair_hopping_kernels], dtype=np.int64),
        pair_high=np.asarray([k.high for k in compiled._pair_hopping_kernels], dtype=np.int64),
        pair_low_from_high=np.asarray(
            [compiled._coerce_coefficient(k.low_from_high) for k in compiled._pair_hopping_kernels], dtype=dtype
        ),
        pair_high_from_low=np.asarray(
            [compiled._coerce_coefficient(k.high_from_low) for k in compiled._pair_hopping_kernels], dtype=dtype
        ),
        monomial_coefficients=np.asarray(
            [compiled._coerce_coefficient(k.coefficient) for k in monomials], dtype=dtype
        ),
        monomial_mode_offsets=np.asarray(mode_offsets, dtype=np.int64),
        monomial_modes=np.asarray(monomial_modes, dtype=np.int64),
        monomial_net_deltas=np.asarray(monomial_net_deltas, dtype=np.int64),
        monomial_operation_offsets=np.asarray(operation_offsets, dtype=np.int64),
        monomial_operation_modes=np.asarray(operation_modes, dtype=np.int64),
        monomial_operation_kinds=np.asarray(operation_kinds, dtype=np.int8),
        monomial_operation_adjustments=np.asarray(operation_adjustments, dtype=np.int64),
        monomial_boundary_start=np.asarray(boundary_start, dtype=np.int64),
        monomial_boundary_stop=np.asarray(boundary_stop, dtype=np.int64),
        monomial_boundary_deltas=np.asarray(
            boundary_rows,
            dtype=np.int64,
        ).reshape((len(monomials), max(0, n_modes - 1))),
    )
    cache[n_modes] = plan
    return plan


def _rank_contribution_table(n_modes: int, particles: int) -> NDArray[np.int64]:
    table = np.zeros((max(0, n_modes - 1), particles + 1), dtype=np.int64)
    for boundary in range(n_modes - 1):
        for population in range(1, particles + 1):
            table[boundary, population] = comb(
                population + boundary, boundary + 1
            )
    return table


def _cumulative_population_table(basis: "FockBasis") -> NDArray[np.integer]:
    n_boundaries = max(0, basis.n_modes - 1)
    if n_boundaries == 0:
        return np.empty((basis.dimension, 0), dtype=np.uint8)

    if basis.N <= np.iinfo(np.uint8).max:
        dtype = np.uint8
    elif basis.N <= np.iinfo(np.uint16).max:
        dtype = np.uint16
    elif basis.N <= np.iinfo(np.uint32).max:
        dtype = np.uint32
    else:
        dtype = np.uint64

    words = basis._uint64_words()
    bits = basis.bits_per_mode
    mask = np.uint64((1 << bits) - 1) if bits < 64 else np.uint64(2**64 - 1)
    cumulative = np.empty((basis.dimension, n_boundaries), dtype=dtype)
    running = np.zeros(basis.dimension, dtype=np.uint64)
    for mode in range(n_boundaries):
        bit_offset = mode * bits
        word = bit_offset // 64
        offset = bit_offset % 64
        if offset + bits <= 64:
            occupation = (words[:, word] >> np.uint64(offset)) & mask
        else:
            low = words[:, word] >> np.uint64(offset)
            high = words[:, word + 1] << np.uint64(64 - offset)
            occupation = (low | high) & mask
        running += occupation
        cumulative[:, mode] = running
    cumulative.setflags(write=False)
    return cumulative


def _get_execution_plan(
    compiled: "CompiledOperator", basis: "FockBasis"
) -> _BosonExecutionPlan:
    """Return cached basis-dependent arrays for repeated matrix-free action.

    The plan stores cumulative occupations at weak-composition boundaries, the
    retained complete-basis ranks for projected sectors, and the combinatorial
    rank contributions needed to invert bosonic transfers. No sparse matrix or
    dense connectivity table is materialized.
    """
    cache = compiled._numba_execution_plans
    cached = cache.get(basis)
    if cached is not None:
        return cached

    ranks = basis._complete_composition_ranks()
    if ranks is None:
        raise RuntimeError("Bosonic Numba execution requires signed 64-bit ranks.")
    plan = _BosonExecutionPlan(
        operator=_pack_operator_plan(compiled, basis.n_modes),
        cumulative_populations=_cumulative_population_table(basis),
        complete_ranks=np.asarray(ranks, dtype=np.int64),
        rank_contributions=_rank_contribution_table(basis.n_modes, basis.N),
        complete_basis=bool(basis.is_complete),
        total_particles=int(basis.N),
        n_modes=int(basis.n_modes),
    )
    cache[basis] = plan
    return plan


def _coefficient_arrays(
    plan: _BosonOperatorPlan, dtype: np.dtype
) -> tuple[NDArray[np.generic], ...]:
    """Cast coefficient groups once to the input/output arithmetic dtype."""
    return (
        np.asarray(plan.number_coefficients, dtype=dtype),
        np.asarray(plan.hopping_low_from_high, dtype=dtype),
        np.asarray(plan.hopping_high_from_low, dtype=dtype),
        np.asarray(plan.pair_low_from_high, dtype=dtype),
        np.asarray(plan.pair_high_from_low, dtype=dtype),
        np.asarray(plan.monomial_coefficients, dtype=dtype),
    )


if njit is not None:

    @njit(cache=True, nogil=True, inline="always")
    def _occupation(cumulative, row, mode, particles, n_modes):
        if n_modes == 1:
            return particles
        if mode == 0:
            return int(cumulative[row, 0])
        if mode == n_modes - 1:
            return particles - int(cumulative[row, n_modes - 2])
        return int(cumulative[row, mode]) - int(cumulative[row, mode - 1])

    @njit(cache=True, nogil=True, inline="always")
    def _basis_row(complete_ranks, rank, complete_basis):
        if complete_basis:
            if rank < 0 or rank >= complete_ranks.size:
                return -1
            return rank
        low = 0
        high = complete_ranks.size
        while low < high:
            middle = (low + high) // 2
            if complete_ranks[middle] < rank:
                low = middle + 1
            else:
                high = middle
        if low < complete_ranks.size and complete_ranks[low] == rank:
            return low
        return -1

    @njit(cache=True, nogil=True, inline="always")
    def _source_rank_pair_transfer(
        destination_rank,
        cumulative,
        row,
        rank_contributions,
        low_mode,
        high_mode,
        destination_minus_source,
    ):
        source_rank = destination_rank
        for boundary in range(low_mode, high_mode):
            destination_population = int(cumulative[row, boundary])
            source_population = destination_population - destination_minus_source
            source_rank += (
                rank_contributions[boundary, destination_population]
                - rank_contributions[boundary, source_population]
            )
        return source_rank

    @njit(cache=True, nogil=True, inline="always")
    def _source_rank_monomial(
        destination_rank,
        cumulative,
        row,
        rank_contributions,
        boundary_deltas,
        start,
        stop,
    ):
        source_rank = destination_rank
        for boundary in range(start, stop):
            delta = boundary_deltas[boundary]
            if delta == 0:
                continue
            destination_population = int(cumulative[row, boundary])
            source_population = destination_population - delta
            if source_population < 0 or source_population >= rank_contributions.shape[1]:
                return -1
            source_rank += (
                rank_contributions[boundary, destination_population]
                - rank_contributions[boundary, source_population]
            )
        return source_rank

    @njit(cache=True, nogil=True, inline="always")
    def _diagonal_value(
        row,
        cumulative,
        particles,
        n_modes,
        coefficients,
        offsets,
        modes,
        powers,
    ):
        value = coefficients[0] * 0 if coefficients.size else 0.0
        for kernel in range(coefficients.size):
            term = coefficients[kernel]
            for position in range(offsets[kernel], offsets[kernel + 1]):
                occupation = _occupation(
                    cumulative, row, modes[position], particles, n_modes
                )
                if occupation == 0:
                    term = term * 0
                    break
                power = powers[position]
                factor = 1
                for _ in range(power):
                    factor *= occupation
                term *= factor
            value += term
        return value

    @njit(cache=True, nogil=True, inline="always")
    def _monomial_amplitude(
        kernel,
        row,
        cumulative,
        particles,
        n_modes,
        coefficients,
        mode_offsets,
        modes,
        net_deltas,
        operation_offsets,
        operation_modes,
        operation_kinds,
        operation_adjustments,
    ):
        for position in range(mode_offsets[kernel], mode_offsets[kernel + 1]):
            mode = modes[position]
            source_occupation = (
                _occupation(cumulative, row, mode, particles, n_modes)
                - net_deltas[position]
            )
            if source_occupation < 0 or source_occupation > particles:
                return coefficients[kernel] * 0

        amplitude = coefficients[kernel]
        for position in range(operation_offsets[kernel], operation_offsets[kernel + 1]):
            mode = operation_modes[position]
            occupation = (
                _occupation(cumulative, row, mode, particles, n_modes)
                + operation_adjustments[position]
            )
            kind = operation_kinds[position]
            if kind == 0:
                if occupation <= 0:
                    return amplitude * 0
                amplitude *= np.sqrt(occupation)
            elif kind == 1:
                if occupation < 0:
                    return amplitude * 0
                amplitude *= np.sqrt(occupation + 1)
            else:
                if occupation <= 0:
                    return amplitude * 0
                amplitude *= occupation
        return amplitude

    @njit(cache=True, nogil=True)
    def _gather_row_value(
        row,
        vector,
        cumulative,
        complete_ranks,
        rank_contributions,
        complete_basis,
        particles,
        n_modes,
        number_coefficients,
        number_offsets,
        number_modes,
        number_powers,
        hopping_low,
        hopping_high,
        hopping_low_from_high,
        hopping_high_from_low,
        pair_low,
        pair_high,
        pair_low_from_high,
        pair_high_from_low,
        monomial_coefficients,
        monomial_mode_offsets,
        monomial_modes,
        monomial_net_deltas,
        monomial_operation_offsets,
        monomial_operation_modes,
        monomial_operation_kinds,
        monomial_operation_adjustments,
        monomial_boundary_start,
        monomial_boundary_stop,
        monomial_boundary_deltas,
    ):
        value = _diagonal_value(
            row,
            cumulative,
            particles,
            n_modes,
            number_coefficients,
            number_offsets,
            number_modes,
            number_powers,
        ) * vector[row]
        destination_rank = complete_ranks[row]

        for kernel in range(hopping_low.size):
            low = hopping_low[kernel]
            high = hopping_high[kernel]
            n_low = _occupation(cumulative, row, low, particles, n_modes)
            n_high = _occupation(cumulative, row, high, particles, n_modes)
            if hopping_low_from_high[kernel] != 0 and n_low >= 1:
                source_rank = _source_rank_pair_transfer(
                    destination_rank,
                    cumulative,
                    row,
                    rank_contributions,
                    low,
                    high,
                    1,
                )
                source = _basis_row(complete_ranks, source_rank, complete_basis)
                if source >= 0:
                    value += (
                        hopping_low_from_high[kernel]
                        * np.sqrt(n_low * (n_high + 1))
                        * vector[source]
                    )
            if hopping_high_from_low[kernel] != 0 and n_high >= 1:
                source_rank = _source_rank_pair_transfer(
                    destination_rank,
                    cumulative,
                    row,
                    rank_contributions,
                    low,
                    high,
                    -1,
                )
                source = _basis_row(complete_ranks, source_rank, complete_basis)
                if source >= 0:
                    value += (
                        hopping_high_from_low[kernel]
                        * np.sqrt(n_high * (n_low + 1))
                        * vector[source]
                    )

        for kernel in range(pair_low.size):
            low = pair_low[kernel]
            high = pair_high[kernel]
            n_low = _occupation(cumulative, row, low, particles, n_modes)
            n_high = _occupation(cumulative, row, high, particles, n_modes)
            if pair_low_from_high[kernel] != 0 and n_low >= 2:
                source_rank = _source_rank_pair_transfer(
                    destination_rank,
                    cumulative,
                    row,
                    rank_contributions,
                    low,
                    high,
                    2,
                )
                source = _basis_row(complete_ranks, source_rank, complete_basis)
                if source >= 0:
                    factor = n_low * (n_low - 1) * (n_high + 1) * (n_high + 2)
                    value += (
                        pair_low_from_high[kernel] * np.sqrt(factor) * vector[source]
                    )
            if pair_high_from_low[kernel] != 0 and n_high >= 2:
                source_rank = _source_rank_pair_transfer(
                    destination_rank,
                    cumulative,
                    row,
                    rank_contributions,
                    low,
                    high,
                    -2,
                )
                source = _basis_row(complete_ranks, source_rank, complete_basis)
                if source >= 0:
                    factor = n_high * (n_high - 1) * (n_low + 1) * (n_low + 2)
                    value += (
                        pair_high_from_low[kernel]
                        * np.sqrt(factor)
                        * vector[source]
                    )

        for kernel in range(monomial_coefficients.size):
            amplitude = _monomial_amplitude(
                kernel,
                row,
                cumulative,
                particles,
                n_modes,
                monomial_coefficients,
                monomial_mode_offsets,
                monomial_modes,
                monomial_net_deltas,
                monomial_operation_offsets,
                monomial_operation_modes,
                monomial_operation_kinds,
                monomial_operation_adjustments,
            )
            if amplitude == 0:
                continue
            source_rank = _source_rank_monomial(
                destination_rank,
                cumulative,
                row,
                rank_contributions,
                monomial_boundary_deltas[kernel],
                monomial_boundary_start[kernel],
                monomial_boundary_stop[kernel],
            )
            source = _basis_row(complete_ranks, source_rank, complete_basis)
            if source >= 0:
                value += amplitude * vector[source]
        return value

    @njit(cache=True, nogil=True)
    def _matvec_gather(
        vector,
        result,
        cumulative,
        complete_ranks,
        rank_contributions,
        complete_basis,
        particles,
        n_modes,
        number_coefficients,
        number_offsets,
        number_modes,
        number_powers,
        hopping_low,
        hopping_high,
        hopping_low_from_high,
        hopping_high_from_low,
        pair_low,
        pair_high,
        pair_low_from_high,
        pair_high_from_low,
        monomial_coefficients,
        monomial_mode_offsets,
        monomial_modes,
        monomial_net_deltas,
        monomial_operation_offsets,
        monomial_operation_modes,
        monomial_operation_kinds,
        monomial_operation_adjustments,
        monomial_boundary_start,
        monomial_boundary_stop,
        monomial_boundary_deltas,
    ):
        for row in range(vector.size):
            result[row] = _gather_row_value(
                row,
                vector,
                cumulative,
                complete_ranks,
                rank_contributions,
                complete_basis,
                particles,
                n_modes,
                number_coefficients,
                number_offsets,
                number_modes,
                number_powers,
                hopping_low,
                hopping_high,
                hopping_low_from_high,
                hopping_high_from_low,
                pair_low,
                pair_high,
                pair_low_from_high,
                pair_high_from_low,
                monomial_coefficients,
                monomial_mode_offsets,
                monomial_modes,
                monomial_net_deltas,
                monomial_operation_offsets,
                monomial_operation_modes,
                monomial_operation_kinds,
                monomial_operation_adjustments,
                monomial_boundary_start,
                monomial_boundary_stop,
                monomial_boundary_deltas,
            )

    @njit(cache=True, nogil=True, parallel=True)
    def _matvec_gather_parallel(
        vector,
        result,
        cumulative,
        complete_ranks,
        rank_contributions,
        complete_basis,
        particles,
        n_modes,
        number_coefficients,
        number_offsets,
        number_modes,
        number_powers,
        hopping_low,
        hopping_high,
        hopping_low_from_high,
        hopping_high_from_low,
        pair_low,
        pair_high,
        pair_low_from_high,
        pair_high_from_low,
        monomial_coefficients,
        monomial_mode_offsets,
        monomial_modes,
        monomial_net_deltas,
        monomial_operation_offsets,
        monomial_operation_modes,
        monomial_operation_kinds,
        monomial_operation_adjustments,
        monomial_boundary_start,
        monomial_boundary_stop,
        monomial_boundary_deltas,
    ):
        for row in prange(vector.size):
            result[row] = _gather_row_value(
                row,
                vector,
                cumulative,
                complete_ranks,
                rank_contributions,
                complete_basis,
                particles,
                n_modes,
                number_coefficients,
                number_offsets,
                number_modes,
                number_powers,
                hopping_low,
                hopping_high,
                hopping_low_from_high,
                hopping_high_from_low,
                pair_low,
                pair_high,
                pair_low_from_high,
                pair_high_from_low,
                monomial_coefficients,
                monomial_mode_offsets,
                monomial_modes,
                monomial_net_deltas,
                monomial_operation_offsets,
                monomial_operation_modes,
                monomial_operation_kinds,
                monomial_operation_adjustments,
                monomial_boundary_start,
                monomial_boundary_stop,
                monomial_boundary_deltas,
            )


def _matvec_numba(
    compiled: "CompiledOperator",
    basis: "FockBasis",
    vector: NDArray[np.generic],
    *,
    parallel: bool,
) -> NDArray[np.generic]:
    """Apply the shared Numba gather executor in serial or parallel mode."""
    require_numba_supported(
        compiled, basis, "numba-parallel" if parallel else "numba-serial"
    )
    vector = compiled._validate_matvec_input(vector, basis.dimension)
    result_dtype = np.result_type(vector.dtype, compiled.data_dtype)
    vector = np.asarray(vector, dtype=result_dtype)
    result = np.zeros(basis.dimension, dtype=result_dtype)
    execution_plan = _get_execution_plan(compiled, basis)
    plan = execution_plan.operator
    (
        number_coefficients,
        hopping_low_from_high,
        hopping_high_from_low,
        pair_low_from_high,
        pair_high_from_low,
        monomial_coefficients,
    ) = _coefficient_arrays(plan, result_dtype)

    kernel = _matvec_gather_parallel if parallel else _matvec_gather
    kernel(
        vector,
        result,
        execution_plan.cumulative_populations,
        execution_plan.complete_ranks,
        execution_plan.rank_contributions,
        execution_plan.complete_basis,
        execution_plan.total_particles,
        execution_plan.n_modes,
        number_coefficients,
        plan.number_offsets,
        plan.number_modes,
        plan.number_powers,
        plan.hopping_low,
        plan.hopping_high,
        hopping_low_from_high,
        hopping_high_from_low,
        plan.pair_low,
        plan.pair_high,
        pair_low_from_high,
        pair_high_from_low,
        monomial_coefficients,
        plan.monomial_mode_offsets,
        plan.monomial_modes,
        plan.monomial_net_deltas,
        plan.monomial_operation_offsets,
        plan.monomial_operation_modes,
        plan.monomial_operation_kinds,
        plan.monomial_operation_adjustments,
        plan.monomial_boundary_start,
        plan.monomial_boundary_stop,
        plan.monomial_boundary_deltas,
    )
    return result


def matvec_numba_serial(
    compiled: "CompiledOperator",
    basis: "FockBasis",
    vector: NDArray[np.generic],
) -> NDArray[np.generic]:
    """Apply a supported boson Hamiltonian with serial Numba execution.

    Parameters
    ----------
    compiled : CompiledOperator
        Fully lowered bosonic operator.
    basis : FockBasis
        Complete or projected fixed-particle basis defining vector ordering.
    vector : array_like
        One-dimensional numerical state vector with length ``basis.dimension``.

    Returns
    -------
    numpy.ndarray
        Matrix-free Hamiltonian action in basis order.
    """
    return _matvec_numba(compiled, basis, vector, parallel=False)


def matvec_numba_parallel(
    compiled: "CompiledOperator",
    basis: "FockBasis",
    vector: NDArray[np.generic],
) -> NDArray[np.generic]:
    """Apply a supported boson Hamiltonian with parallel Numba execution.

    The gather kernel assigns each destination basis row to exactly one Numba
    iteration. Threads therefore read shared immutable plan data and the input
    vector while writing disjoint output entries, avoiding atomics and
    thread-local copies of the full result vector.

    Parameters
    ----------
    compiled : CompiledOperator
        Fully lowered bosonic operator.
    basis : FockBasis
        Complete or projected fixed-particle basis defining vector ordering.
    vector : array_like
        One-dimensional numerical state vector with length ``basis.dimension``.

    Returns
    -------
    numpy.ndarray
        Matrix-free Hamiltonian action in basis order.
    """
    return _matvec_numba(compiled, basis, vector, parallel=True)
