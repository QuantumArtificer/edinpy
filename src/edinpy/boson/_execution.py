"""Execute compiled bosonic operators directly on packed occupation states."""

from __future__ import annotations

from math import comb, sqrt
from numbers import Integral, Number
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._algebra import Operator
from ._basis import FockState, NullState, StateSum
from ._compiler import CompiledPlan, compile_lowered
from ._ir import lower_operator

if TYPE_CHECKING:
    from ._basis import FockBasis


class CompiledOperator:
    """Executable bosonic operator with specialized sparse-matrix kernels."""

    __slots__ = (
        "_number_kernels",
        "_hopping_kernels",
        "_pair_hopping_kernels",
        "_monomial_kernels",
        "_generic_terms",
        "_projected_out_terms",
        "_real_valued",
        "_numba_operator_plans",
        "_numba_execution_plans",
    )

    def __init__(self, plan: CompiledPlan) -> None:
        self._number_kernels = plan.number_kernels
        self._hopping_kernels = plan.hopping_kernels
        self._pair_hopping_kernels = plan.pair_hopping_kernels
        self._monomial_kernels = plan.monomial_kernels
        self._generic_terms = plan.generic_terms
        self._projected_out_terms = plan.projected_out_terms
        self._real_valued = plan.real_valued
        self._numba_operator_plans = {}
        self._numba_execution_plans = {}

    @property
    def data_dtype(self) -> np.dtype:
        """numpy.dtype: Double-precision matrix-element dtype."""
        return np.dtype(np.float64 if self._real_valued else np.complex128)

    @property
    def fully_lowered(self) -> bool:
        """bool: Whether no literal symbolic fallback terms remain."""
        return not self._generic_terms

    def _coerce_coefficient(self, value: Number) -> float | complex:
        """Return a scalar compatible with the compiled matrix dtype."""
        if self._real_valued:
            return complex(value).real
        return complex(value)

    @property
    def stats(self) -> dict[str, int]:
        """dict[str, int]: Counts of compiled bosonic execution kernels."""
        return {
            "number_product": len(self._number_kernels),
            "hopping_groups": len(self._hopping_kernels),
            "hopping_terms": sum(
                int(kernel.low_from_high != 0) + int(kernel.high_from_low != 0)
                for kernel in self._hopping_kernels
            ),
            "pair_hopping_groups": len(self._pair_hopping_kernels),
            "pair_hopping_terms": sum(
                int(kernel.low_from_high != 0) + int(kernel.high_from_low != 0)
                for kernel in self._pair_hopping_kernels
            ),
            "monomial": len(self._monomial_kernels),
            "generic": len(self._generic_terms),
            "projected_out": self._projected_out_terms,
        }

    @staticmethod
    def _occupations_from_words(words, bit_offset, bits_per_mode, mask):
        """Vectorize one occupation field from little-endian ``uint64`` words."""
        word = bit_offset // 64
        offset = bit_offset % 64
        mask64 = np.uint64(mask)

        if offset + bits_per_mode <= 64:
            return (words[:, word] >> np.uint64(offset)) & mask64

        low = words[:, word] >> np.uint64(offset)
        high = words[:, word + 1] << np.uint64(64 - offset)
        return (low | high) & mask64

    def _diagonal_values(self, basis, number_specs, mask):
        """Return vectorized diagonal values for arbitrary packed-state widths."""
        if not number_specs:
            return None

        words = basis._uint64_words()
        dtype = np.float64 if self._real_valued else np.complex128
        diagonal = np.zeros(basis.dimension, dtype=dtype)
        bits_per_mode = basis.bits_per_mode

        for coefficient, powers in number_specs:
            coefficient = self._coerce_coefficient(coefficient)
            values = np.full(basis.dimension, coefficient, dtype=dtype)
            for shift, power in powers:
                occupations = self._occupations_from_words(
                    words,
                    shift,
                    bits_per_mode,
                    mask,
                )
                if power == 1:
                    values *= occupations
                else:
                    values *= occupations**power
            diagonal += values
        return diagonal

    @staticmethod
    def _complete_fixed_n_basis(basis):
        """Return whether ``basis`` is the complete weak-composition basis."""
        return basis.is_complete

    def _common_occupation_views(self, basis, endpoint_modes, boundaries):
        """Return vectorized occupations and cumulative populations.

        The canonical packed state may span any number of 64-bit words.  Mode
        occupations are decoded from the cached multiword execution view once,
        while cumulative populations are retained only at boundaries crossed by
        a hopping kernel.
        """
        words = basis._uint64_words()
        bits = basis.bits_per_mode
        mask = (1 << bits) - 1
        endpoint_modes = set(endpoint_modes)
        boundaries = set(boundaries)
        occupations = {}
        cumulative = {}
        running = np.zeros(basis.dimension, dtype=np.int64)

        max_mode = max(endpoint_modes | boundaries, default=-1)
        for mode in range(max_mode + 1):
            values = self._occupations_from_words(
                words,
                mode * bits,
                bits,
                mask,
            ).astype(np.int64, copy=False)
            if mode in endpoint_modes:
                occupations[mode] = values
            running += values
            if mode in boundaries:
                cumulative[mode] = running.copy()

        return occupations, cumulative, mask

    @staticmethod
    def _rank_step_tables(boundaries, N):
        """Return combinatorial row-step tables for selected mode boundaries."""
        down = {}
        up_one = {}
        up_two = {}
        for boundary in boundaries:
            down[boundary] = np.fromiter(
                (comb(population + boundary, boundary) for population in range(N + 2)),
                dtype=np.int64,
                count=N + 2,
            )
            up_one[boundary] = np.fromiter(
                (
                    0
                    if population == 0
                    else comb(population - 1 + boundary, boundary)
                    for population in range(N + 1)
                ),
                dtype=np.int64,
                count=N + 1,
            )
            up_two[boundary] = np.fromiter(
                (
                    0
                    if population < 2
                    else comb(population - 2 + boundary, boundary)
                    for population in range(N + 1)
                ),
                dtype=np.int64,
                count=N + 1,
            )
        return down, up_one, up_two

    @staticmethod
    def _assemble_csc_streams(dimension, rows_chunks, columns_chunks, data_chunks):
        """Assemble column-sorted transition streams directly into CSC arrays.

        Every specialized stream contains at most one entry per source column
        and its column indices are strictly increasing. Instead of globally
        sorting all emitted ``(column, row)`` pairs, count entries per column,
        build ``indptr``, and scatter each stream into the next free slot of its
        columns. SciPy later sorts the small row neighborhoods within columns.
        """
        if not rows_chunks:
            index_dtype = (
                np.int32 if dimension <= np.iinfo(np.int32).max else np.int64
            )
            return (
                np.empty(0, dtype=index_dtype),
                np.zeros(dimension + 1, dtype=index_dtype),
                np.empty(0, dtype=np.float64),
            )

        nnz = sum(len(columns) for columns in columns_chunks)
        counts = np.zeros(dimension, dtype=np.int64)
        for columns in columns_chunks:
            counts[columns] += 1

        indptr64 = np.empty(dimension + 1, dtype=np.int64)
        indptr64[0] = 0
        np.cumsum(counts, out=indptr64[1:])

        rows64 = np.empty(nnz, dtype=np.int64)
        data_dtype = np.result_type(
            *(np.asarray(chunk).dtype for chunk in data_chunks)
        )
        data = np.empty(nnz, dtype=data_dtype)
        next_slot = indptr64[:-1].copy()

        for rows, columns, values in zip(rows_chunks, columns_chunks, data_chunks):
            positions = next_slot[columns]
            rows64[positions] = rows
            data[positions] = values
            next_slot[columns] = positions + 1

        index_dtype = (
            np.int32
            if max(dimension, nnz, 1) <= np.iinfo(np.int32).max
            else np.int64
        )
        return (
            rows64.astype(index_dtype, copy=False),
            indptr64.astype(index_dtype, copy=False),
            data,
        )

    def _emit_common_vectorized_csc(self, basis):
        """Emit complete-basis number/hopping Hamiltonians without state lookup.

        A complete fixed-``N`` bosonic basis is an ordered set of weak
        compositions.  Transferring one boson across the boundary between
        modes ``k`` and ``k+1`` changes the composition rank by a binomial
        coefficient depending only on the cumulative population through mode
        ``k``.  Summing those rank steps across all crossed boundaries gives
        the destination row directly.  Pair hopping uses the two consecutive
        one-boson rank steps.

        This avoids constructing final packed Python integers and avoids a
        per-transition dictionary lookup.  Occupations themselves come from
        the cached multiword ``uint64`` execution view, so there is no 64-bit
        state-width boundary.
        """
        dimension = basis.dimension
        bits = basis.bits_per_mode
        mask = (1 << bits) - 1
        columns_all = np.arange(dimension, dtype=np.int64)

        number_specs = tuple(
            (
                kernel.coefficient,
                tuple((mode * bits, power) for mode, power in kernel.powers),
            )
            for kernel in self._number_kernels
        )
        diagonal = self._diagonal_values(basis, number_specs, mask)

        if not self._hopping_kernels and not self._pair_hopping_kernels:
            if diagonal is None:
                index_dtype = np.int32 if dimension <= np.iinfo(np.int32).max else np.int64
                return (
                    np.empty(0, dtype=index_dtype),
                    np.zeros(dimension + 1, dtype=index_dtype),
                    np.empty(0, dtype=np.float64),
                )
            nonzero = diagonal != 0
            indices64 = columns_all[nonzero]
            counts = nonzero.astype(np.int64, copy=False)
            indptr64 = np.empty(dimension + 1, dtype=np.int64)
            indptr64[0] = 0
            np.cumsum(counts, out=indptr64[1:])
            index_dtype = (
                np.int32
                if max(dimension, len(indices64), 1) <= np.iinfo(np.int32).max
                else np.int64
            )
            return (
                indices64.astype(index_dtype, copy=False),
                indptr64.astype(index_dtype, copy=False),
                np.asarray(diagonal[nonzero]),
            )

        endpoint_modes = set()
        boundaries = set()
        for kernel in (*self._hopping_kernels, *self._pair_hopping_kernels):
            endpoint_modes.update((kernel.low, kernel.high))
            boundaries.update(range(kernel.low, kernel.high))

        occupations, cumulative, _mask = self._common_occupation_views(
            basis,
            endpoint_modes,
            boundaries,
        )
        down, up_one, up_two = self._rank_step_tables(boundaries, basis.N)

        rows_chunks = []
        columns_chunks = []
        data_chunks = []

        if diagonal is not None:
            nonzero = diagonal != 0
            if np.any(nonzero):
                diagonal_columns = columns_all[nonzero]
                rows_chunks.append(diagonal_columns)
                columns_chunks.append(diagonal_columns)
                data_chunks.append(diagonal[nonzero])

        for kernel in self._hopping_kernels:
            low = kernel.low
            high = kernel.high
            n_low = occupations[low]
            n_high = occupations[high]

            if kernel.low_from_high != 0:
                valid = np.flatnonzero(n_high > 0)
                if valid.size:
                    delta = np.zeros(valid.size, dtype=np.int64)
                    for boundary in range(low, high):
                        delta += down[boundary][cumulative[boundary][valid]]
                    rows_chunks.append(valid - delta)
                    columns_chunks.append(valid)
                    data_chunks.append(
                        kernel.low_from_high
                        * np.sqrt((n_low[valid] + 1) * n_high[valid])
                    )

            if kernel.high_from_low != 0:
                valid = np.flatnonzero(n_low > 0)
                if valid.size:
                    delta = np.zeros(valid.size, dtype=np.int64)
                    for boundary in range(low, high):
                        delta += up_one[boundary][cumulative[boundary][valid]]
                    rows_chunks.append(valid + delta)
                    columns_chunks.append(valid)
                    data_chunks.append(
                        kernel.high_from_low
                        * np.sqrt((n_high[valid] + 1) * n_low[valid])
                    )

        for kernel in self._pair_hopping_kernels:
            low = kernel.low
            high = kernel.high
            n_low = occupations[low]
            n_high = occupations[high]

            if kernel.low_from_high != 0:
                valid = np.flatnonzero(n_high >= 2)
                if valid.size:
                    delta = np.zeros(valid.size, dtype=np.int64)
                    for boundary in range(low, high):
                        population = cumulative[boundary][valid]
                        delta += (
                            down[boundary][population]
                            + down[boundary][population + 1]
                        )
                    factor = (
                        (n_low[valid] + 1)
                        * (n_low[valid] + 2)
                        * n_high[valid]
                        * (n_high[valid] - 1)
                    )
                    rows_chunks.append(valid - delta)
                    columns_chunks.append(valid)
                    data_chunks.append(kernel.low_from_high * np.sqrt(factor))

            if kernel.high_from_low != 0:
                valid = np.flatnonzero(n_low >= 2)
                if valid.size:
                    delta = np.zeros(valid.size, dtype=np.int64)
                    for boundary in range(low, high):
                        population = cumulative[boundary][valid]
                        delta += (
                            up_one[boundary][population]
                            + up_two[boundary][population]
                        )
                    factor = (
                        (n_high[valid] + 1)
                        * (n_high[valid] + 2)
                        * n_low[valid]
                        * (n_low[valid] - 1)
                    )
                    rows_chunks.append(valid + delta)
                    columns_chunks.append(valid)
                    data_chunks.append(kernel.high_from_low * np.sqrt(factor))

        return self._assemble_csc_streams(
            dimension, rows_chunks, columns_chunks, data_chunks
        )

    def _emit_projected_ranked_csc(self, basis, retained_ranks):
        """Emit common kernels in a projected basis through complete-basis ranks.

        The projected basis is an ordered subset of the complete fixed-N weak
        compositions.  Each retained state therefore has a monotonically
        increasing complete-basis rank. Closed-form weak-composition rank
        steps give the destination complete rank for hopping and pair hopping without
        constructing a final packed state.  ``searchsorted`` on the retained
        rank array then answers whether that destination survives the projection
        and, if so, gives its projected row.

        This path is independent of packed-state width: occupations come from
        the cached multiword ``uint64`` execution view, while transition lookup
        uses signed 64-bit composition ranks.
        """
        dimension = basis.dimension
        bits = basis.bits_per_mode
        mask = (1 << bits) - 1
        columns_all = np.arange(dimension, dtype=np.int64)

        number_specs = tuple(
            (
                kernel.coefficient,
                tuple((mode * bits, power) for mode, power in kernel.powers),
            )
            for kernel in self._number_kernels
        )
        diagonal = self._diagonal_values(basis, number_specs, mask)

        if not self._hopping_kernels and not self._pair_hopping_kernels:
            if diagonal is None:
                index_dtype = np.int32 if dimension <= np.iinfo(np.int32).max else np.int64
                return (
                    np.empty(0, dtype=index_dtype),
                    np.zeros(dimension + 1, dtype=index_dtype),
                    np.empty(0, dtype=np.float64),
                )
            nonzero = diagonal != 0
            indices64 = columns_all[nonzero]
            counts = nonzero.astype(np.int64, copy=False)
            indptr64 = np.empty(dimension + 1, dtype=np.int64)
            indptr64[0] = 0
            np.cumsum(counts, out=indptr64[1:])
            index_dtype = (
                np.int32
                if max(dimension, len(indices64), 1) <= np.iinfo(np.int32).max
                else np.int64
            )
            return (
                indices64.astype(index_dtype, copy=False),
                indptr64.astype(index_dtype, copy=False),
                np.asarray(diagonal[nonzero]),
            )

        endpoint_modes = set()
        boundaries = set()
        for kernel in (*self._hopping_kernels, *self._pair_hopping_kernels):
            endpoint_modes.update((kernel.low, kernel.high))
            boundaries.update(range(kernel.low, kernel.high))

        occupations, cumulative, _mask = self._common_occupation_views(
            basis,
            endpoint_modes,
            boundaries,
        )
        down, up_one, up_two = self._rank_step_tables(boundaries, basis.N)

        rows_chunks = []
        columns_chunks = []
        data_chunks = []

        if diagonal is not None:
            nonzero = diagonal != 0
            if np.any(nonzero):
                diagonal_columns = columns_all[nonzero]
                rows_chunks.append(diagonal_columns)
                columns_chunks.append(diagonal_columns)
                data_chunks.append(diagonal[nonzero])

        def retained_rows(target_ranks):
            rows = np.searchsorted(retained_ranks, target_ranks).astype(
                np.int64,
                copy=False,
            )
            present = rows < dimension
            if np.any(present):
                positions = np.flatnonzero(present)
                present[positions] = (
                    retained_ranks[rows[positions]] == target_ranks[positions]
                )
            return rows, present

        for kernel in self._hopping_kernels:
            low = kernel.low
            high = kernel.high
            n_low = occupations[low]
            n_high = occupations[high]

            if kernel.low_from_high != 0:
                valid = np.flatnonzero(n_high > 0)
                if valid.size:
                    delta = np.zeros(valid.size, dtype=np.int64)
                    for boundary in range(low, high):
                        delta += down[boundary][cumulative[boundary][valid]]
                    rows, present = retained_rows(retained_ranks[valid] - delta)
                    if np.any(present):
                        kept = valid[present]
                        rows_chunks.append(rows[present])
                        columns_chunks.append(kept)
                        data_chunks.append(
                            kernel.low_from_high
                            * np.sqrt((n_low[kept] + 1) * n_high[kept])
                        )

            if kernel.high_from_low != 0:
                valid = np.flatnonzero(n_low > 0)
                if valid.size:
                    delta = np.zeros(valid.size, dtype=np.int64)
                    for boundary in range(low, high):
                        delta += up_one[boundary][cumulative[boundary][valid]]
                    rows, present = retained_rows(retained_ranks[valid] + delta)
                    if np.any(present):
                        kept = valid[present]
                        rows_chunks.append(rows[present])
                        columns_chunks.append(kept)
                        data_chunks.append(
                            kernel.high_from_low
                            * np.sqrt((n_high[kept] + 1) * n_low[kept])
                        )

        for kernel in self._pair_hopping_kernels:
            low = kernel.low
            high = kernel.high
            n_low = occupations[low]
            n_high = occupations[high]

            if kernel.low_from_high != 0:
                valid = np.flatnonzero(n_high >= 2)
                if valid.size:
                    delta = np.zeros(valid.size, dtype=np.int64)
                    for boundary in range(low, high):
                        population = cumulative[boundary][valid]
                        delta += (
                            down[boundary][population]
                            + down[boundary][population + 1]
                        )
                    rows, present = retained_rows(retained_ranks[valid] - delta)
                    if np.any(present):
                        kept = valid[present]
                        factor = (
                            (n_low[kept] + 1)
                            * (n_low[kept] + 2)
                            * n_high[kept]
                            * (n_high[kept] - 1)
                        )
                        rows_chunks.append(rows[present])
                        columns_chunks.append(kept)
                        data_chunks.append(kernel.low_from_high * np.sqrt(factor))

            if kernel.high_from_low != 0:
                valid = np.flatnonzero(n_low >= 2)
                if valid.size:
                    delta = np.zeros(valid.size, dtype=np.int64)
                    for boundary in range(low, high):
                        population = cumulative[boundary][valid]
                        delta += (
                            up_one[boundary][population]
                            + up_two[boundary][population]
                        )
                    rows, present = retained_rows(retained_ranks[valid] + delta)
                    if np.any(present):
                        kept = valid[present]
                        factor = (
                            (n_high[kept] + 1)
                            * (n_high[kept] + 2)
                            * n_low[kept]
                            * (n_low[kept] - 1)
                        )
                        rows_chunks.append(rows[present])
                        columns_chunks.append(kept)
                        data_chunks.append(kernel.high_from_low * np.sqrt(factor))

        return self._assemble_csc_streams(
            dimension, rows_chunks, columns_chunks, data_chunks
        )

    def _emit_lowered_csc(self, basis):
        """Emit CSC arrays with basis-specific bit operations precomputed once."""
        packed_states = basis.packed_states
        dimension = len(packed_states)
        bits_per_mode = basis.bits_per_mode
        mask = (1 << bits_per_mode) - 1
        lookup = {state: index for index, state in enumerate(packed_states)}
        get_row = lookup.get
        missing = -1

        number_specs = tuple(
            (
                kernel.coefficient,
                tuple(
                    (mode * bits_per_mode, power)
                    for mode, power in kernel.powers
                ),
            )
            for kernel in self._number_kernels
        )
        hopping_specs = tuple(
            (
                kernel.low * bits_per_mode,
                kernel.high * bits_per_mode,
                1 << (kernel.low * bits_per_mode),
                1 << (kernel.high * bits_per_mode),
                kernel.low_from_high,
                kernel.high_from_low,
            )
            for kernel in self._hopping_kernels
        )
        pair_hopping_specs = tuple(
            (
                kernel.low * bits_per_mode,
                kernel.high * bits_per_mode,
                2 << (kernel.low * bits_per_mode),
                2 << (kernel.high * bits_per_mode),
                kernel.low_from_high,
                kernel.high_from_low,
            )
            for kernel in self._pair_hopping_kernels
        )

        diagonal_values = self._diagonal_values(basis, number_specs, mask)
        diagonal_possible = diagonal_values is not None
        max_per_column = (
            int(diagonal_possible)
            + 2 * len(hopping_specs)
            + 2 * len(pair_hopping_specs)
            + len(self._monomial_kernels)
        )
        capacity = max(1, dimension * max_per_column)
        index_dtype = (
            np.int32
            if max(dimension, capacity) <= np.iinfo(np.int32).max
            else np.int64
        )
        indices = np.empty(capacity, dtype=index_dtype)
        data = np.empty(capacity, dtype=np.complex128)
        indptr = np.empty(dimension + 1, dtype=index_dtype)
        indptr[0] = 0
        cursor = 0

        for column, state in enumerate(packed_states):
            if diagonal_possible:
                diagonal = diagonal_values[column]
                if diagonal != 0:
                    indices[cursor] = column
                    data[cursor] = diagonal
                    cursor += 1

            for (
                low_shift,
                high_shift,
                unit_low,
                unit_high,
                low_from_high,
                high_from_low,
            ) in hopping_specs:
                n_low = (state >> low_shift) & mask
                n_high = (state >> high_shift) & mask

                if low_from_high != 0 and n_high:
                    final_state = state + unit_low - unit_high
                    row = get_row(final_state, missing)
                    if row != missing:
                        indices[cursor] = row
                        data[cursor] = low_from_high * sqrt((n_low + 1) * n_high)
                        cursor += 1

                if high_from_low != 0 and n_low:
                    final_state = state - unit_low + unit_high
                    row = get_row(final_state, missing)
                    if row != missing:
                        indices[cursor] = row
                        data[cursor] = high_from_low * sqrt((n_high + 1) * n_low)
                        cursor += 1

            for (
                low_shift,
                high_shift,
                two_low,
                two_high,
                low_from_high,
                high_from_low,
            ) in pair_hopping_specs:
                n_low = (state >> low_shift) & mask
                n_high = (state >> high_shift) & mask

                if low_from_high != 0 and n_high >= 2:
                    final_state = state + two_low - two_high
                    row = get_row(final_state, missing)
                    if row != missing:
                        factor = (
                            (n_low + 1)
                            * (n_low + 2)
                            * n_high
                            * (n_high - 1)
                        )
                        indices[cursor] = row
                        data[cursor] = low_from_high * sqrt(factor)
                        cursor += 1

                if high_from_low != 0 and n_low >= 2:
                    final_state = state - two_low + two_high
                    row = get_row(final_state, missing)
                    if row != missing:
                        factor = (
                            (n_high + 1)
                            * (n_high + 2)
                            * n_low
                            * (n_low - 1)
                        )
                        indices[cursor] = row
                        data[cursor] = high_from_low * sqrt(factor)
                        cursor += 1

            for kernel in self._monomial_kernels:
                if kernel.particle_delta != 0:
                    continue
                result = kernel.transition(state, bits_per_mode, mask)
                if result is None:
                    continue
                final_state, amplitude = result
                row = get_row(final_state, missing)
                if row != missing and amplitude != 0:
                    indices[cursor] = row
                    data[cursor] = amplitude
                    cursor += 1

            indptr[column + 1] = cursor

        return indices[:cursor], indptr, data[:cursor]

    @staticmethod
    def _append_literal_result(entries, result, basis):
        if isinstance(result, NullState):
            return
        if not isinstance(result, (FockState, StateSum)):
            raise TypeError(
                "Literal bosonic operator action returned unsupported type "
                f"{type(result).__name__}."
            )
        states = (result,) if isinstance(result, FockState) else result.states
        for state in states:
            if state.N != basis.N or state.n_modes != basis.n_modes:
                continue
            try:
                row = basis.index(state.occupations)
            except ValueError:
                continue
            if state.amp != 0:
                entries[row] = entries.get(row, 0) + state.amp

    def _emit_generic_csc(self, basis):
        packed_states = basis.packed_states
        bits_per_mode = basis.bits_per_mode
        mask = (1 << bits_per_mode) - 1
        lookup = {state: index for index, state in enumerate(packed_states)}
        missing = -1

        indices = []
        data = []
        indptr = [0]

        for column, state in enumerate(packed_states):
            entries = {}

            diagonal = 0
            for kernel in self._number_kernels:
                diagonal += kernel.evaluate(state, bits_per_mode, mask)
            if diagonal != 0:
                entries[column] = entries.get(column, 0) + diagonal

            for kernel in self._hopping_kernels:
                for final_state, amplitude in kernel.transitions(
                    state,
                    bits_per_mode,
                    mask,
                ):
                    row = lookup.get(final_state, missing)
                    if row != missing and amplitude != 0:
                        entries[row] = entries.get(row, 0) + amplitude

            for kernel in self._pair_hopping_kernels:
                for final_state, amplitude in kernel.transitions(
                    state,
                    bits_per_mode,
                    mask,
                ):
                    row = lookup.get(final_state, missing)
                    if row != missing and amplitude != 0:
                        entries[row] = entries.get(row, 0) + amplitude

            for kernel in self._monomial_kernels:
                if kernel.particle_delta != 0:
                    continue
                result = kernel.transition(state, bits_per_mode, mask)
                if result is None:
                    continue
                final_state, amplitude = result
                row = lookup.get(final_state, missing)
                if row != missing and amplitude != 0:
                    entries[row] = entries.get(row, 0) + amplitude

            if self._generic_terms:
                ket = basis.state(column)
                for term in self._generic_terms:
                    self._append_literal_result(entries, term * ket, basis)

            for row in sorted(entries):
                amplitude = entries[row]
                if amplitude != 0:
                    indices.append(row)
                    data.append(amplitude)
            indptr.append(len(indices))

        index_dtype = (
            np.int32
            if max(len(indices), len(indptr), basis.dimension, 1)
            <= np.iinfo(np.int32).max
            else np.int64
        )
        return (
            np.asarray(indices, dtype=index_dtype),
            np.asarray(indptr, dtype=index_dtype),
            np.asarray(data, dtype=np.complex128),
        )

    @staticmethod
    def _validate_matvec_input(
        vector: ArrayLike, dimension: int
    ) -> NDArray[np.generic]:
        """Return one-dimensional numerical input for matrix-free action."""
        vector = np.asarray(vector)
        if vector.ndim == 2 and vector.shape == (dimension, 1):
            vector = vector[:, 0]
        if vector.ndim != 1 or vector.shape[0] != dimension:
            raise ValueError(
                "Matrix-free input vector length must equal basis.dimension."
            )
        if not np.issubdtype(vector.dtype, np.number):
            raise TypeError("Matrix-free input vector must contain numerical values.")
        return vector

    def _matvec_common_ranked(self, basis, vector, retained_ranks=None):
        """Apply number, hopping, and pair-hopping kernels through rank steps."""
        dimension = basis.dimension
        dtype = np.result_type(vector.dtype, self.data_dtype)
        result = np.zeros(dimension, dtype=dtype)
        bits = basis.bits_per_mode
        mask = (1 << bits) - 1

        number_specs = tuple(
            (
                kernel.coefficient,
                tuple((mode * bits, power) for mode, power in kernel.powers),
            )
            for kernel in self._number_kernels
        )
        diagonal = self._diagonal_values(basis, number_specs, mask)
        if diagonal is not None:
            result += diagonal * vector

        if not self._hopping_kernels and not self._pair_hopping_kernels:
            return result

        endpoint_modes = set()
        boundaries = set()
        for kernel in (*self._hopping_kernels, *self._pair_hopping_kernels):
            endpoint_modes.update((kernel.low, kernel.high))
            boundaries.update(range(kernel.low, kernel.high))

        occupations, cumulative, _mask = self._common_occupation_views(
            basis,
            endpoint_modes,
            boundaries,
        )
        down, up_one, up_two = self._rank_step_tables(boundaries, basis.N)

        if retained_ranks is None:
            def retained_rows(columns, target_ranks):
                return columns, target_ranks
        else:
            def retained_rows(columns, target_ranks):
                rows = np.searchsorted(retained_ranks, target_ranks).astype(
                    np.int64, copy=False
                )
                present = rows < dimension
                if np.any(present):
                    positions = np.flatnonzero(present)
                    present[positions] = (
                        retained_ranks[rows[positions]]
                        == target_ranks[positions]
                    )
                return columns[present], rows[present]

        for kernel in self._hopping_kernels:
            low = kernel.low
            high = kernel.high
            n_low = occupations[low]
            n_high = occupations[high]

            if kernel.low_from_high != 0:
                columns = np.flatnonzero(n_high > 0)
                if columns.size:
                    delta = np.zeros(columns.size, dtype=np.int64)
                    for boundary in range(low, high):
                        delta += down[boundary][cumulative[boundary][columns]]
                    target = (
                        columns - delta
                        if retained_ranks is None
                        else retained_ranks[columns] - delta
                    )
                    kept, rows = retained_rows(columns, target)
                    if rows.size:
                        amplitude = kernel.low_from_high * np.sqrt(
                            (n_low[kept] + 1) * n_high[kept]
                        )
                        result[rows] += amplitude * vector[kept]

            if kernel.high_from_low != 0:
                columns = np.flatnonzero(n_low > 0)
                if columns.size:
                    delta = np.zeros(columns.size, dtype=np.int64)
                    for boundary in range(low, high):
                        delta += up_one[boundary][cumulative[boundary][columns]]
                    target = (
                        columns + delta
                        if retained_ranks is None
                        else retained_ranks[columns] + delta
                    )
                    kept, rows = retained_rows(columns, target)
                    if rows.size:
                        amplitude = kernel.high_from_low * np.sqrt(
                            (n_high[kept] + 1) * n_low[kept]
                        )
                        result[rows] += amplitude * vector[kept]

        for kernel in self._pair_hopping_kernels:
            low = kernel.low
            high = kernel.high
            n_low = occupations[low]
            n_high = occupations[high]

            if kernel.low_from_high != 0:
                columns = np.flatnonzero(n_high >= 2)
                if columns.size:
                    delta = np.zeros(columns.size, dtype=np.int64)
                    for boundary in range(low, high):
                        population = cumulative[boundary][columns]
                        delta += (
                            down[boundary][population]
                            + down[boundary][population + 1]
                        )
                    target = (
                        columns - delta
                        if retained_ranks is None
                        else retained_ranks[columns] - delta
                    )
                    kept, rows = retained_rows(columns, target)
                    if rows.size:
                        factor = (
                            (n_low[kept] + 1)
                            * (n_low[kept] + 2)
                            * n_high[kept]
                            * (n_high[kept] - 1)
                        )
                        result[rows] += (
                            kernel.low_from_high * np.sqrt(factor) * vector[kept]
                        )

            if kernel.high_from_low != 0:
                columns = np.flatnonzero(n_low >= 2)
                if columns.size:
                    delta = np.zeros(columns.size, dtype=np.int64)
                    for boundary in range(low, high):
                        population = cumulative[boundary][columns]
                        delta += (
                            up_one[boundary][population]
                            + up_two[boundary][population]
                        )
                    target = (
                        columns + delta
                        if retained_ranks is None
                        else retained_ranks[columns] + delta
                    )
                    kept, rows = retained_rows(columns, target)
                    if rows.size:
                        factor = (
                            (n_high[kept] + 1)
                            * (n_high[kept] + 2)
                            * n_low[kept]
                            * (n_low[kept] - 1)
                        )
                        result[rows] += (
                            kernel.high_from_low * np.sqrt(factor) * vector[kept]
                        )

        return result

    def _matvec_lowered_direct(self, basis, vector):
        """Apply fully lowered kernels directly on exact packed Python integers."""
        from bisect import bisect_left

        packed_states = basis.packed_states
        dimension = basis.dimension
        bits = basis.bits_per_mode
        mask = (1 << bits) - 1
        dtype = np.result_type(vector.dtype, self.data_dtype)
        result = np.zeros(dimension, dtype=dtype)

        number_specs = tuple(
            (
                kernel.coefficient,
                tuple((mode * bits, power) for mode, power in kernel.powers),
            )
            for kernel in self._number_kernels
        )
        diagonal = self._diagonal_values(basis, number_specs, mask)
        if diagonal is not None:
            result += diagonal * vector

        hopping_specs = tuple(
            (
                kernel.low * bits,
                kernel.high * bits,
                1 << (kernel.low * bits),
                1 << (kernel.high * bits),
                kernel.low_from_high,
                kernel.high_from_low,
            )
            for kernel in self._hopping_kernels
        )
        pair_specs = tuple(
            (
                kernel.low * bits,
                kernel.high * bits,
                2 << (kernel.low * bits),
                2 << (kernel.high * bits),
                kernel.low_from_high,
                kernel.high_from_low,
            )
            for kernel in self._pair_hopping_kernels
        )

        for column, input_amplitude in enumerate(vector):
            if input_amplitude == 0:
                continue
            state = packed_states[column]

            for (
                low_shift,
                high_shift,
                unit_low,
                unit_high,
                low_from_high,
                high_from_low,
            ) in hopping_specs:
                n_low = (state >> low_shift) & mask
                n_high = (state >> high_shift) & mask

                if low_from_high != 0 and n_high:
                    final_state = state + unit_low - unit_high
                    row = bisect_left(packed_states, final_state)
                    if row < dimension and packed_states[row] == final_state:
                        amplitude = self._coerce_coefficient(low_from_high) * sqrt(
                            (n_low + 1) * n_high
                        )
                        result[row] += amplitude * input_amplitude

                if high_from_low != 0 and n_low:
                    final_state = state - unit_low + unit_high
                    row = bisect_left(packed_states, final_state)
                    if row < dimension and packed_states[row] == final_state:
                        amplitude = self._coerce_coefficient(high_from_low) * sqrt(
                            (n_high + 1) * n_low
                        )
                        result[row] += amplitude * input_amplitude

            for (
                low_shift,
                high_shift,
                two_low,
                two_high,
                low_from_high,
                high_from_low,
            ) in pair_specs:
                n_low = (state >> low_shift) & mask
                n_high = (state >> high_shift) & mask

                if low_from_high != 0 and n_high >= 2:
                    final_state = state + two_low - two_high
                    row = bisect_left(packed_states, final_state)
                    if row < dimension and packed_states[row] == final_state:
                        factor = (
                            (n_low + 1) * (n_low + 2) * n_high * (n_high - 1)
                        )
                        result[row] += (
                            self._coerce_coefficient(low_from_high)
                            * sqrt(factor)
                            * input_amplitude
                        )

                if high_from_low != 0 and n_low >= 2:
                    final_state = state - two_low + two_high
                    row = bisect_left(packed_states, final_state)
                    if row < dimension and packed_states[row] == final_state:
                        factor = (
                            (n_high + 1) * (n_high + 2) * n_low * (n_low - 1)
                        )
                        result[row] += (
                            self._coerce_coefficient(high_from_low)
                            * sqrt(factor)
                            * input_amplitude
                        )

            for kernel in self._monomial_kernels:
                if kernel.particle_delta != 0:
                    continue
                transition = kernel.transition(state, bits, mask)
                if transition is None:
                    continue
                final_state, amplitude = transition
                row = bisect_left(packed_states, final_state)
                if row < dimension and packed_states[row] == final_state:
                    if self._real_valued:
                        amplitude = complex(amplitude).real
                    result[row] += amplitude * input_amplitude

        return result

    def _matvec_literal_fallback(self, basis, vector):
        """Apply operators that still contain symbolic fallback terms."""
        output = np.zeros(
            basis.dimension,
            dtype=np.result_type(vector.dtype, self.data_dtype),
        )

        for column, coefficient in enumerate(vector):
            if coefficient == 0:
                continue
            basis_ket = basis.state(column)
            ket = FockState(
                basis_ket.occupations,
                amp=coefficient,
                index=column,
            )
            result = self.apply(ket)

            if isinstance(result, NullState):
                continue
            states = result.states if isinstance(result, StateSum) else (result,)
            for state in states:
                try:
                    row = basis.index(state)
                except ValueError:
                    continue
                output[row] += state.amp

        return output

    def matvec(
        self, basis: "FockBasis", vector: ArrayLike
    ) -> NDArray[np.generic]:
        """Return the projected matrix-free action ``H @ vector``.

        Common fully lowered bosonic Hamiltonians use the same direct
        weak-composition rank transitions as sparse emission, but accumulate
        immediately into the output vector. General lowered monomials retain
        an exact packed-state path; only genuinely generic symbolic terms fall
        back to literal ``FockState`` action.
        """
        vector = self._validate_matvec_input(vector, basis.dimension)

        if not self.fully_lowered:
            return self._matvec_literal_fallback(basis, vector)

        if not self._monomial_kernels:
            if self._complete_fixed_n_basis(basis):
                return self._matvec_common_ranked(basis, vector)
            retained_ranks = basis._complete_composition_ranks()
            if retained_ranks is not None:
                return self._matvec_common_ranked(
                    basis, vector, retained_ranks=retained_ranks
                )

        return self._matvec_lowered_direct(basis, vector)

    def emit_csc(
        self, basis: "FockBasis"
    ) -> tuple[NDArray[np.integer], NDArray[np.integer], NDArray[np.generic]]:
        """Return CSC arrays for this operator projected into ``basis``."""
        if self.fully_lowered:
            common_only = not self._monomial_kernels
            if common_only:
                if self._complete_fixed_n_basis(basis):
                    return self._emit_common_vectorized_csc(basis)
                retained_ranks = basis._complete_composition_ranks()
                if retained_ranks is not None:
                    return self._emit_projected_ranked_csc(basis, retained_ranks)
            return self._emit_lowered_csc(basis)
        return self._emit_generic_csc(basis)

    def apply(self, ket: FockState) -> FockState | StateSum | NullState:
        """Apply the original compiled expression to one literal Fock state.

        This literal-state path preserves exact symbolic behavior for custom
        operators and for direct state algebra. Sparse Hamiltonian construction
        uses packed basis states through :meth:`emit_csc`.
        """
        if not isinstance(ket, FockState):
            raise TypeError("'ket' must be a bosonic FockState.")

        output = {}
        occupations = ket.occupations
        n_modes = ket.n_modes
        N = ket.N
        extra_creations = max(
            (
                sum(kind == "create" for kind, _mode in kernel.operations)
                for kernel in self._monomial_kernels
            ),
            default=0,
        )
        # Specialized hopping kernels conserve particle number, so only the
        # general monomial path can temporarily require a wider occupation field.
        bits = max(1, (N + extra_creations).bit_length())
        mask = (1 << bits) - 1
        packed = 0
        for mode, occupation in enumerate(occupations):
            packed |= occupation << (mode * bits)

        diagonal = 0
        for kernel in self._number_kernels:
            diagonal += kernel.evaluate(packed, bits, mask)
        if diagonal != 0:
            output[occupations] = output.get(occupations, 0) + diagonal * ket.amp

        for kernel in self._hopping_kernels:
            for final_state, amplitude in kernel.transitions(packed, bits, mask):
                final = tuple(
                    (final_state >> (mode * bits)) & mask for mode in range(n_modes)
                )
                output[final] = output.get(final, 0) + amplitude * ket.amp

        for kernel in self._pair_hopping_kernels:
            for final_state, amplitude in kernel.transitions(packed, bits, mask):
                final = tuple(
                    (final_state >> (mode * bits)) & mask for mode in range(n_modes)
                )
                output[final] = output.get(final, 0) + amplitude * ket.amp

        for kernel in self._monomial_kernels:
            result = kernel.transition(packed, bits, mask)
            if result is None:
                continue
            final_state, amplitude = result
            final = tuple(
                (final_state >> (mode * bits)) & mask for mode in range(n_modes)
            )
            output[final] = output.get(final, 0) + amplitude * ket.amp

        for term in self._generic_terms:
            result = term * ket
            if isinstance(result, FockState):
                output[result.occupations] = (
                    output.get(result.occupations, 0) + result.amp
                )
            elif isinstance(result, StateSum):
                for state in result.states:
                    output[state.occupations] = (
                        output.get(state.occupations, 0) + state.amp
                    )
            elif not isinstance(result, NullState):
                raise TypeError(
                    "Literal bosonic operator action returned unsupported type "
                    f"{type(result).__name__}."
                )

        states = [
            FockState(state, amp=amplitude)
            for state, amplitude in sorted(output.items())
            if amplitude != 0
        ]
        if not states:
            return NullState()
        if len(states) == 1:
            return states[0]
        return StateSum(states)


def compile_operator(operator: Operator) -> CompiledOperator:
    """Compile a bosonic symbolic expression into an executable operator.

    Parameters
    ----------
    operator : Operator
        Bosonic symbolic expression built from public algebra primitives or a
        compatible custom :class:`Operator` subclass. Built-in terms are
        lowered into numerical kernels; unsupported custom terms retain their
        literal action for NumPy execution.

    Returns
    -------
    CompiledOperator
        Reusable execution object shared by sparse and matrix-free Hamiltonian
        backends.
    """
    return CompiledOperator(compile_lowered(lower_operator(operator)))
