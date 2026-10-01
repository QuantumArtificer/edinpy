"""Internal execution kernels for fermionic symbolic operators.

The symbolic Fock algebra remains the source of truth. Common operator
patterns are lowered to direct bitwise kernels, while arbitrary expressions
retain the generic symbolic fallback.
"""

from __future__ import annotations

from bisect import bisect_left
from functools import lru_cache
from math import comb

import numpy as np

from ._compiler import lower_operator
from ._ir import (
    _GenericKernel,
    _HoppingKernel,
    _HoppingPairKernel,
    _MonomialKernel,
    _NumberProductKernel,
    _ParityFreeHoppingKernel,
    _SimpleHoppingKernel,
)
from ._basis import (
    FockState,
    NullState,
    StateSum,
)


_DEFAULT_MATVEC_BLOCK_SIZE = 32768


@lru_cache(maxsize=None)
def _complete_rank_tables(
    n_modes: int,
    particles: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build bytewise combinadic lookup tables for a complete fixed-N basis.

    Parameters
    ----------
    n_modes : int
        Number of fermionic modes represented by the occupation word. The
        Numba executor uses these tables only for native ``uint64`` bases.
    particles : int
        Fixed particle number of the complete basis. This determines the
        combinadic order carried from one byte of the occupation word to the
        next.

    Returns
    -------
    rank_contributions : numpy.ndarray
        Integer lookup table whose entries give the combinadic rank
        contribution of each possible occupation byte at each byte position
        and incoming particle order.
    byte_popcounts : numpy.ndarray
        Population count for every possible byte value. The ranker uses these
        counts to advance the particle order between adjacent bytes.

    Notes
    -----
    Eight-bit chunks keep the table small while replacing per-bit combinadic
    arithmetic with a handful of cache-friendly lookups. The tables are
    immutable and memoized by ``(n_modes, particles)``.
    """
    n_modes = int(n_modes)
    particles = int(particles)
    chunk_bits = 8
    table_size = 1 << chunk_bits
    n_chunks = (n_modes + chunk_bits - 1) // chunk_bits

    contributions = np.zeros(
        (n_chunks, particles + 2, table_size),
        dtype=np.int64,
    )
    popcounts = np.fromiter(
        (value.bit_count() for value in range(table_size)),
        dtype=np.uint8,
        count=table_size,
    )

    for chunk_index in range(n_chunks):
        base_position = chunk_bits * chunk_index
        for starting_order in range(particles, 0, -1):
            row = contributions[chunk_index, starting_order]
            next_row = contributions[chunk_index, starting_order + 1]
            for chunk_value in range(1, table_size):
                lowest = chunk_value & -chunk_value
                bit = lowest.bit_length() - 1
                rest = chunk_value ^ lowest
                position = base_position + bit
                contribution = (
                    comb(position, starting_order) if position < n_modes else 0
                )
                row[chunk_value] = contribution + next_row[rest]

    contributions.setflags(write=False)
    popcounts.setflags(write=False)
    return contributions, popcounts


def _group_hopping_kernels(kernels):
    """Combine directed hopping terms acting on the same unordered mode pair.

    Parameters
    ----------
    kernels : iterable of _HoppingKernel
        Directed one-body transitions produced by the compiler.

    Returns
    -------
    tuple[_HoppingPairKernel, ...]
        Pair kernels containing independent coefficients for both directions.

    Notes
    -----
    Grouping uses a hash key containing the two mode bits and the parity mask.
    The work therefore scales linearly with the number of hopping terms and
    does not require pairwise term comparisons.
    """
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
            (0.0, 0.0),
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
        """Organize lowered kernels into specialized execution categories.

        Parameters
        ----------
        kernels : iterable
            Intermediate-representation kernels produced by
            :func:`edinpy.fermion._compiler.lower_operator`.

        Notes
        -----
        Directed hopping terms are first grouped by unordered mode pair.
        Pair kernels are then separated into symmetric parity-free,
        asymmetric parity-free, and signed transitions so each class can use
        the simplest matrix-emission path available.
        """
        kernels = tuple(kernels)
        coefficients = [
            kernel.coefficient
            for kernel in kernels
            if hasattr(kernel, "coefficient")
        ]
        self._real_valued = all(
            complex(coefficient).imag == 0
            for coefficient in coefficients
        )

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

        simple_hopping = []
        parity_free_hopping = []
        signed_hopping = []

        for kernel in self._hopping_kernels:
            if kernel.parity_mask == 0:
                if kernel.low_from_high == kernel.high_from_low:
                    if kernel.low_from_high != 0:
                        simple_hopping.append(
                            _SimpleHoppingKernel(
                                coefficient=kernel.low_from_high,
                                transition_mask=kernel.transition_mask,
                            )
                        )
                elif (
                    kernel.low_from_high != 0
                    or kernel.high_from_low != 0
                ):
                    parity_free_hopping.append(
                        _ParityFreeHoppingKernel(
                            low_bit=kernel.low_bit,
                            high_bit=kernel.high_bit,
                            transition_mask=kernel.transition_mask,
                            low_from_high=kernel.low_from_high,
                            high_from_low=kernel.high_from_low,
                        )
                    )
            elif (
                kernel.low_from_high != 0
                or kernel.high_from_low != 0
            ):
                signed_hopping.append(kernel)

        self._simple_hopping_kernels = tuple(simple_hopping)
        self._parity_free_hopping_kernels = tuple(
            parity_free_hopping
        )
        self._signed_hopping_kernels = tuple(signed_hopping)

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
    def data_dtype(self):
        """numpy.dtype: Double-precision matrix-element dtype for lowered kernels."""
        import numpy as np

        return np.float64 if self._real_valued else np.complex128

    def _coerce_coefficient(self, value):
        """Return a scalar compatible with the selected matrix-element dtype."""
        if self._real_valued:
            return complex(value).real
        return complex(value)

    @property
    def fully_lowered(self):
        """bool: Whether no symbolic fallback kernels are present."""
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

        Duplicate matrix entries are allowed here. The sparse assembly stage
        combines them with ``sum_duplicates()``.
        """
        if not self.fully_lowered:
            raise RuntimeError(
                "Direct sparse emission requires a fully lowered operator."
            )

        state = ket.state
        ket_amp = ket.amp
        diagonal = 0.0 if self._real_valued else 0.0j

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

    @property
    def vectorized_simple_hopping(self):
        """Whether the operator admits the vectorized simple-hopping path."""
        return (
            bool(self._simple_hopping_kernels)
            and not self._number_kernels
            and not self._parity_free_hopping_kernels
            and not self._signed_hopping_kernels
            and not self._monomial_kernels
            and not self._generic_kernels
        )

    @property
    def vectorized_simple_terms(self):
        """Whether only vectorizable hopping and number products are present."""
        return (
            bool(
                self._simple_hopping_kernels
                or self._number_kernels
            )
            and not self._parity_free_hopping_kernels
            and not self._signed_hopping_kernels
            and not self._monomial_kernels
            and not self._generic_kernels
        )

    @property
    def vectorized_number_conserving_monomials(self):
        """Whether pure monomial terms admit blocked vectorized execution."""
        if (
            not self._monomial_kernels
            or self._number_kernels
            or self._simple_hopping_kernels
            or self._parity_free_hopping_kernels
            or self._signed_hopping_kernels
            or self._generic_kernels
        ):
            return False

        for kernel in self._monomial_kernels:
            annihilated = (
                kernel.transition_mask
                & kernel.required_occupied_mask
            ).bit_count()

            created = (
                kernel.transition_mask
                & kernel.required_empty_mask
            ).bit_count()

            if created != annihilated:
                return False

        return True

    def _emit_simple_hopping_csc_vectorized(
        self,
        basis_states,
        n_modes,
        block_size=32768,
    ):
        """Emit symmetric parity-free hopping directly into CSC arrays.

        Parameters
        ----------
        basis_states : sequence of int
            Ordered fixed-particle-number occupation bit strings.
        n_modes : int
            Total number of fermionic modes.
        block_size : int, optional
            Number of basis columns processed per NumPy block.

        Returns
        -------
        indices, indptr, data : numpy.ndarray
            Arrays defining the CSC matrix representation.

        Notes
        -----
        This path is used only for at most 64 modes because NumPy's native
        unsigned-integer vectorization is used for occupation masks and state
        transitions. Larger mode sets fall back to Python-integer execution.
        """
        import math
        import numpy as np

        nbasis = len(basis_states)

        if nbasis == 0:
            return (
                np.empty(0, dtype=np.int32),
                np.zeros(1, dtype=np.int32),
                np.empty(0, dtype=self.data_dtype),
            )

        neff = n_modes
        n_particles = basis_states[0].bit_count()

        # NumPy integer bitwise kernels are currently limited to a native
        # 64-bit occupation representation. Larger systems retain the scalar
        # Python-integer execution path.
        if neff > 64:
            raise RuntimeError(
                "Vectorized hopping emission supports at most 64 modes."
            )

        kernels = self._simple_hopping_kernels

        if (
            neff < 2
            or n_particles == 0
            or n_particles == neff
        ):
            directed_count = 0
        else:
            directed_count = math.comb(
                neff - 2,
                n_particles - 1,
            )

        capacity = (
            2
            * len(kernels)
            * directed_count
        )

        max_int32 = np.iinfo(np.int32).max

        index_dtype = (
            np.int32
            if max(nbasis, capacity) <= max_int32
            else np.int64
        )

        states = np.asarray(
            basis_states,
            dtype=np.uint64,
        )

        masks = np.asarray(
            [
                kernel.transition_mask
                for kernel in kernels
            ],
            dtype=np.uint64,
        )

        coefficients = np.asarray(
            [
                self._coerce_coefficient(kernel.coefficient)
                for kernel in kernels
            ],
            dtype=self.data_dtype,
        )

        indices = np.empty(
            capacity,
            dtype=index_dtype,
        )

        data = np.empty(
            capacity,
            dtype=self.data_dtype,
        )

        indptr = np.empty(
            nbasis + 1,
            dtype=index_dtype,
        )

        indptr[0] = 0
        cursor = 0

        for start in range(0, nbasis, block_size):
            stop = min(
                start + block_size,
                nbasis,
            )

            block_states = states[start:stop]

            occupation = (
                block_states[:, None]
                & masks[None, :]
            )

            active = (
                (occupation != 0)
                & (occupation != masks[None, :])
            )

            counts = active.sum(
                axis=1,
                dtype=np.int64,
            )

            cumulative = np.cumsum(
                counts,
                dtype=np.int64,
            )

            indptr[start + 1:stop + 1] = (
                cursor + cumulative
            )

            if cumulative.size == 0:
                continue

            nactive = int(cumulative[-1])

            if nactive == 0:
                continue

            final_states = (
                block_states[:, None]
                ^ masks[None, :]
            )

            selected_final_states = (
                final_states[active]
            )

            rows = np.searchsorted(
                states,
                selected_final_states,
            )

            bond_indices = np.nonzero(
                active
            )[1]

            end = cursor + nactive

            indices[cursor:end] = rows
            data[cursor:end] = (
                coefficients[bond_indices]
            )

            cursor = end

        if cursor != capacity:
            raise RuntimeError(
                "Vectorized hopping emission produced an "
                "unexpected number of sparse entries."
            )

        return indices, indptr, data

    def _emit_simple_terms_csc_vectorized(
        self,
        basis_states,
        n_modes,
        block_size=32768,
    ):
        """Emit diagonal number products and simple hopping in blocked form.

        Parameters
        ----------
        basis_states : sequence of int
            Ordered fixed-particle-number occupation bit strings.
        n_modes : int
            Total number of fermionic modes.
        block_size : int, optional
            Number of basis columns processed per NumPy block.

        Returns
        -------
        indices, indptr, data : numpy.ndarray
            Arrays defining the CSC matrix representation.

        Notes
        -----
        All diagonal number products acting on one basis state are accumulated
        into one diagonal matrix element before the off-diagonal hopping entries
        are emitted.
        """
        import math
        import numpy as np

        nbasis = len(basis_states)

        if nbasis == 0:
            return (
                np.empty(0, dtype=np.int32),
                np.zeros(1, dtype=np.int32),
                np.empty(0, dtype=self.data_dtype),
            )

        neff = n_modes
        n_particles = basis_states[0].bit_count()

        if neff > 64:
            raise RuntimeError(
                "Vectorized sparse emission supports at most 64 modes."
            )

        hopping_kernels = self._simple_hopping_kernels
        number_kernels = self._number_kernels

        if (
            neff >= 2
            and 1 <= n_particles <= neff - 1
        ):
            directed_count = math.comb(
                neff - 2,
                n_particles - 1,
            )
        else:
            directed_count = 0

        hopping_capacity = (
            2
            * len(hopping_kernels)
            * directed_count
        )

        # All diagonal number products are accumulated into at most one
        # matrix entry per basis state.
        capacity = hopping_capacity

        if number_kernels:
            capacity += nbasis

        max_int32 = np.iinfo(np.int32).max

        index_dtype = (
            np.int32
            if max(nbasis, capacity) <= max_int32
            else np.int64
        )

        states = np.asarray(
            basis_states,
            dtype=np.uint64,
        )

        hopping_masks = np.asarray(
            [
                kernel.transition_mask
                for kernel in hopping_kernels
            ],
            dtype=np.uint64,
        )

        hopping_coefficients = np.asarray(
            [
                self._coerce_coefficient(kernel.coefficient)
                for kernel in hopping_kernels
            ],
            dtype=self.data_dtype,
        )

        number_masks = np.asarray(
            [
                kernel.mask
                for kernel in number_kernels
            ],
            dtype=np.uint64,
        )

        number_coefficients = np.asarray(
            [
                self._coerce_coefficient(kernel.coefficient)
                for kernel in number_kernels
            ],
            dtype=self.data_dtype,
        )

        indices = np.empty(
            capacity,
            dtype=index_dtype,
        )

        data = np.empty(
            capacity,
            dtype=self.data_dtype,
        )

        indptr = np.empty(
            nbasis + 1,
            dtype=index_dtype,
        )

        indptr[0] = 0
        cursor = 0
        nhopping = len(hopping_kernels)

        for start in range(0, nbasis, block_size):
            stop = min(
                start + block_size,
                nbasis,
            )

            block_states = states[start:stop]
            nblock = stop - start

            # ---------------------------------------------------------
            # Diagonal number products.
            # ---------------------------------------------------------

            diagonal = np.zeros(
                nblock,
                dtype=self.data_dtype,
            )

            for mask, coefficient in zip(
                number_masks,
                number_coefficients,
            ):
                occupied = (
                    block_states & mask
                ) == mask

                diagonal[occupied] += coefficient

            diagonal_active = diagonal != 0

            # ---------------------------------------------------------
            # Symmetric parity-free hopping.
            # ---------------------------------------------------------

            if nhopping:
                occupation = (
                    block_states[:, None]
                    & hopping_masks[None, :]
                )

                hopping_active = (
                    (occupation != 0)
                    & (
                        occupation
                        != hopping_masks[None, :]
                    )
                )
            else:
                hopping_active = np.empty(
                    (nblock, 0),
                    dtype=bool,
                )

            # One possible diagonal entry plus one entry per hopping
            # bond. Row-major masking preserves CSC column grouping.
            active = np.empty(
                (nblock, nhopping + 1),
                dtype=bool,
            )

            active[:, 0] = diagonal_active

            if nhopping:
                active[:, 1:] = hopping_active

            counts = active.sum(
                axis=1,
                dtype=np.int64,
            )

            cumulative = np.cumsum(
                counts,
                dtype=np.int64,
            )

            indptr[start + 1:stop + 1] = (
                cursor + cumulative
            )

            if cumulative.size == 0:
                continue

            nactive = int(cumulative[-1])

            if nactive == 0:
                continue

            row_buffer = np.empty(
                (nblock, nhopping + 1),
                dtype=index_dtype,
            )

            value_buffer = np.empty(
                (nblock, nhopping + 1),
                dtype=self.data_dtype,
            )

            # Diagonal column.
            row_buffer[:, 0] = np.arange(
                start,
                stop,
                dtype=index_dtype,
            )

            value_buffer[:, 0] = diagonal

            if nhopping:
                final_states = (
                    block_states[:, None]
                    ^ hopping_masks[None, :]
                )

                hopping_rows = row_buffer[:, 1:]
                hopping_values = value_buffer[:, 1:]

                selected_final_states = (
                    final_states[hopping_active]
                )

                hopping_rows[hopping_active] = np.searchsorted(
                    states,
                    selected_final_states,
                )

                active_bonds = np.nonzero(
                    hopping_active
                )[1]

                hopping_values[hopping_active] = (
                    hopping_coefficients[active_bonds]
                )

            end = cursor + nactive

            indices[cursor:end] = row_buffer[active]
            data[cursor:end] = value_buffer[active]

            cursor = end

        return (
            indices[:cursor],
            indptr,
            data[:cursor],
        )

    def _emit_number_conserving_monomials_csc_vectorized(
        self,
        basis_states,
        n_modes,
        block_size=32768,
    ):
        """Emit number-conserving fermionic monomials in vectorized blocks.

        Parameters
        ----------
        basis_states : sequence of int
            Ordered fixed-particle-number occupation bit strings.
        n_modes : int
            Total number of fermionic modes.
        block_size : int, optional
            Number of basis columns processed per NumPy block.

        Returns
        -------
        indices, indptr, data : numpy.ndarray
            Arrays defining the CSC matrix representation.

        Notes
        -----
        Initial occupied/empty constraints select active monomials. Final
        states are generated by XOR with the compiled transition mask, and
        the fermionic sign is obtained from the parity mask.
        """
        import math
        import numpy as np

        nbasis = len(basis_states)

        if nbasis == 0:
            return (
                np.empty(0, dtype=np.int32),
                np.zeros(1, dtype=np.int32),
                np.empty(0, dtype=self.data_dtype),
            )

        neff = n_modes
        n_particles = basis_states[0].bit_count()

        if neff > 64:
            raise RuntimeError(
                "Vectorized monomial emission supports at most 64 modes."
            )

        kernels = self._monomial_kernels

        capacity = 0

        for kernel in kernels:
            occupied = kernel.required_occupied_mask
            empty = kernel.required_empty_mask

            required_occupied = occupied.bit_count()
            required_empty = empty.bit_count()

            free_modes = (
                neff
                - required_occupied
                - required_empty
            )

            free_particles = (
                n_particles
                - required_occupied
            )

            if (
                0 <= free_particles <= free_modes
            ):
                capacity += math.comb(
                    free_modes,
                    free_particles,
                )

        max_int32 = np.iinfo(np.int32).max

        index_dtype = (
            np.int32
            if max(nbasis, capacity) <= max_int32
            else np.int64
        )

        states = np.asarray(
            basis_states,
            dtype=np.uint64,
        )

        required_occupied = np.asarray(
            [
                kernel.required_occupied_mask
                for kernel in kernels
            ],
            dtype=np.uint64,
        )

        required_empty = np.asarray(
            [
                kernel.required_empty_mask
                for kernel in kernels
            ],
            dtype=np.uint64,
        )

        transition_masks = np.asarray(
            [
                kernel.transition_mask
                for kernel in kernels
            ],
            dtype=np.uint64,
        )

        parity_masks = np.asarray(
            [
                kernel.parity_mask
                for kernel in kernels
            ],
            dtype=np.uint64,
        )

        coefficients = np.asarray(
            [
                self._coerce_coefficient(kernel.coefficient)
                for kernel in kernels
            ],
            dtype=self.data_dtype,
        )

        indices = np.empty(
            capacity,
            dtype=index_dtype,
        )

        data = np.empty(
            capacity,
            dtype=self.data_dtype,
        )

        indptr = np.empty(
            nbasis + 1,
            dtype=index_dtype,
        )

        indptr[0] = 0
        cursor = 0

        for start in range(0, nbasis, block_size):
            stop = min(
                start + block_size,
                nbasis,
            )

            block_states = states[start:stop]

            occupied_ok = (
                (
                    block_states[:, None]
                    & required_occupied[None, :]
                )
                == required_occupied[None, :]
            )

            empty_ok = (
                (
                    block_states[:, None]
                    & required_empty[None, :]
                )
                == 0
            )

            active = occupied_ok & empty_ok

            counts = active.sum(
                axis=1,
                dtype=np.int64,
            )

            cumulative = np.cumsum(
                counts,
                dtype=np.int64,
            )

            indptr[start + 1:stop + 1] = (
                cursor + cumulative
            )

            if cumulative.size == 0:
                continue

            nactive = int(cumulative[-1])

            if nactive == 0:
                continue

            state_indices, kernel_indices = np.nonzero(
                active
            )

            selected_states = block_states[
                state_indices
            ]

            final_states = (
                selected_states
                ^ transition_masks[kernel_indices]
            )

            rows = np.searchsorted(
                states,
                final_states,
            )

            parity_values = (
                selected_states
                & parity_masks[kernel_indices]
            ).copy()

            parity_values ^= parity_values >> np.uint64(32)
            parity_values ^= parity_values >> np.uint64(16)
            parity_values ^= parity_values >> np.uint64(8)
            parity_values ^= parity_values >> np.uint64(4)
            parity_values ^= parity_values >> np.uint64(2)
            parity_values ^= parity_values >> np.uint64(1)

            parity = parity_values & np.uint64(1)

            amplitudes = coefficients[
                kernel_indices
            ].copy()

            amplitudes[parity != 0] *= -1

            end = cursor + nactive

            indices[cursor:end] = rows
            data[cursor:end] = amplitudes

            cursor = end

        if cursor != capacity:
            raise RuntimeError(
                "Vectorized monomial emission produced an "
                "unexpected number of sparse entries."
            )

        return (
            indices,
            indptr,
            data,
        )

    def emit_csc(self, basis_states, n_modes):
        """Emit preallocated CSC arrays from integer Fock basis states.

        Parameters
        ----------
        basis_states : sequence of int
            Ordered occupation bit strings.
        n_modes : int
            Total number of fermionic modes. This value is supplied explicitly
            because it cannot be inferred from vacuum occupation bit strings.

        Returns
        -------
        indices, indptr, data : numpy.ndarray
            CSC storage arrays.
        """
        if not self.fully_lowered:
            raise RuntimeError(
                "Direct matrix emission requires a fully lowered operator."
            )

        neff = int(n_modes)
        if neff < 0:
            raise ValueError("'n_modes' must be non-negative.")

        if neff <= 64:
            if self.vectorized_simple_hopping:
                return self._emit_simple_hopping_csc_vectorized(
                    basis_states,
                    neff,
                )

            if self.vectorized_simple_terms:
                return self._emit_simple_terms_csc_vectorized(
                    basis_states,
                    neff,
                )

            if self.vectorized_number_conserving_monomials:
                return (
                    self._emit_number_conserving_monomials_csc_vectorized(
                        basis_states,
                        neff,
                    )
                )

        import math
        import numpy as np

        nbasis = len(basis_states)

        if nbasis == 0:
            return (
                np.empty(0, dtype=np.int32),
                np.zeros(1, dtype=np.int32),
                np.empty(0, dtype=self.data_dtype),
            )

        n_particles = basis_states[0].bit_count()
        neff = int(n_modes)

        number_kernels = self._number_kernels
        simple_hopping_kernels = self._simple_hopping_kernels
        parity_free_hopping_kernels = (
            self._parity_free_hopping_kernels
        )
        signed_hopping_kernels = self._signed_hopping_kernels
        monomial_kernels = self._monomial_kernels

        # Upper bound for diagonal output. Number terms are accumulated into
        # one matrix element per basis state.
        capacity = nbasis if number_kernels else 0

        # For a fixed-N basis, a directed c_i^dagger c_j transition is
        # allowed for C(L - 2, N - 1) states. A symmetric pair therefore
        # contributes twice this number.
        if (
            neff >= 2
            and 1 <= n_particles <= neff - 1
        ):
            directed_hopping_count = math.comb(
                neff - 2,
                n_particles - 1,
            )
        else:
            directed_hopping_count = 0

        capacity += (
            2
            * len(simple_hopping_kernels)
            * directed_hopping_count
        )

        for kernel in parity_free_hopping_kernels:
            if kernel.low_from_high != 0:
                capacity += directed_hopping_count
            if kernel.high_from_low != 0:
                capacity += directed_hopping_count

        for kernel in signed_hopping_kernels:
            if kernel.low_from_high != 0:
                capacity += directed_hopping_count
            if kernel.high_from_low != 0:
                capacity += directed_hopping_count

        # A monomial contributes once for each basis state satisfying its
        # initial occupied/empty constraints, provided the operation
        # conserves the fixed particle number.
        for kernel in monomial_kernels:
            occupied = kernel.required_occupied_mask
            empty = kernel.required_empty_mask

            if occupied & empty:
                continue

            required_occupied = occupied.bit_count()
            required_empty = empty.bit_count()

            created = (
                kernel.transition_mask & empty
            ).bit_count()

            annihilated = (
                kernel.transition_mask & occupied
            ).bit_count()

            if created != annihilated:
                continue

            remaining_modes = (
                neff
                - required_occupied
                - required_empty
            )

            remaining_particles = (
                n_particles
                - required_occupied
            )

            if (
                remaining_particles < 0
                or remaining_particles > remaining_modes
            ):
                continue

            capacity += math.comb(
                remaining_modes,
                remaining_particles,
            )

        index_dtype = (
            np.int32
            if max(nbasis, capacity) <= np.iinfo(np.int32).max
            else np.int64
        )

        basis_index = {
            state: index
            for index, state in enumerate(basis_states)
        }
        get_index = basis_index.get
        missing_index = None

        indices = np.empty(
            capacity,
            dtype=index_dtype,
        )

        data = np.empty(
            capacity,
            dtype=self.data_dtype,
        )

        indptr = np.empty(
            nbasis + 1,
            dtype=index_dtype,
        )

        indptr[0] = 0
        cursor = 0

        for column, state in enumerate(basis_states):
            diagonal = 0.0 if self._real_valued else 0.0j

            for kernel in number_kernels:
                if state & kernel.mask == kernel.mask:
                    diagonal += kernel.coefficient

            if diagonal != 0:
                indices[cursor] = column
                data[cursor] = diagonal
                cursor += 1

            for kernel in simple_hopping_kernels:
                occupation = state & kernel.transition_mask

                if (
                    occupation == 0
                    or occupation == kernel.transition_mask
                ):
                    continue

                final_state = state ^ kernel.transition_mask
                row = get_index(final_state)

                if row != missing_index:
                    indices[cursor] = row
                    data[cursor] = self._coerce_coefficient(kernel.coefficient)
                    cursor += 1

            for kernel in parity_free_hopping_kernels:
                occupation = state & kernel.transition_mask

                if occupation == kernel.low_bit:
                    coefficient = kernel.high_from_low
                elif occupation == kernel.high_bit:
                    coefficient = kernel.low_from_high
                else:
                    continue

                if coefficient == 0:
                    continue

                final_state = state ^ kernel.transition_mask
                row = get_index(final_state)

                if row != missing_index:
                    indices[cursor] = row
                    data[cursor] = coefficient
                    cursor += 1

            for kernel in signed_hopping_kernels:
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

                if row != missing_index:
                    indices[cursor] = row
                    data[cursor] = (
                        -coefficient if parity else coefficient
                    )
                    cursor += 1

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

                if row == missing_index:
                    continue

                coefficient = kernel.coefficient

                indices[cursor] = row
                data[cursor] = (
                    -coefficient if parity else coefficient
                )
                cursor += 1

            indptr[column + 1] = cursor

        return (
            indices[:cursor],
            indptr,
            data[:cursor],
        )

    def apply(self, ket):
        """Apply the compiled operator to one Fock state.

        Parameters
        ----------
        ket : FockState
            Input occupation-number state.

        Returns
        -------
        dict[int, numbers.Number]
            Mapping from output occupation bit strings to accumulated
            amplitudes. Duplicate contributions from different symbolic terms
            are summed before returning.
        """
        state = ket.state
        ket_amp = ket.amp
        output = {}
        diagonal = 0.0 if self._real_valued else 0.0j

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

            if isinstance(result, NullState):
                continue

            if isinstance(result, FockState):
                if result.amp != 0:
                    output[result.state] = (
                        output.get(result.state, 0.0j)
                        + result.amp
                    )
                continue

            if isinstance(result, StateSum):
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


    @staticmethod
    def _uint64_parity(values):
        """Return parity bits for a ``uint64`` NumPy array."""
        values = np.asarray(values, dtype=np.uint64).copy()
        values ^= values >> np.uint64(32)
        values ^= values >> np.uint64(16)
        values ^= values >> np.uint64(8)
        values ^= values >> np.uint64(4)
        values ^= values >> np.uint64(2)
        values ^= values >> np.uint64(1)
        return values & np.uint64(1)

    @staticmethod
    def _validate_matvec_input(vector, dimension):
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

    @staticmethod
    def _matvec_block_size(block_size, dimension):
        """Return a bounded native-word matrix-free execution block size."""
        if block_size is None:
            block_size = _DEFAULT_MATVEC_BLOCK_SIZE
        try:
            block_size = int(block_size)
        except (TypeError, ValueError) as exc:
            raise TypeError("Matrix-free block size must be an integer.") from exc
        if block_size <= 0:
            raise ValueError("Matrix-free block size must be positive.")
        return min(block_size, max(int(dimension), 1))

    def _matvec_lowered_uint64(
        self,
        basis,
        vector,
        *,
        block_size=None,
    ):
        """Apply lowered native-word kernels in bounded source-state blocks.

        The NumPy executor processes contiguous source-state blocks so
        temporary transition arrays stay bounded as Hilbert-space dimension
        grows. Destination states are mapped back to basis rows with vectorized
        ``searchsorted``; projected bases additionally verify that each
        candidate destination is present.
        """
        states = np.asarray(basis._execution_states(), dtype=np.uint64)
        dimension = states.size
        dtype = np.result_type(vector.dtype, self.data_dtype)
        result = np.zeros(dimension, dtype=dtype)

        if dimension == 0:
            return result

        block_size = self._matvec_block_size(block_size, dimension)
        complete_basis = bool(getattr(basis, "is_complete", False))
        diagonal_scratch = (
            np.empty(block_size, dtype=self.data_dtype)
            if self._number_kernels
            else None
        )

        for start in range(0, dimension, block_size):
            stop = min(start + block_size, dimension)
            block_states = states[start:stop]
            block_vector = vector[start:stop]
            block_dimension = stop - start

            if self._number_kernels:
                diagonal = diagonal_scratch[:block_dimension]
                diagonal.fill(0)
                for kernel in self._number_kernels:
                    mask = np.uint64(kernel.mask)
                    active = (block_states & mask) == mask
                    if np.any(active):
                        diagonal[active] += self._coerce_coefficient(
                            kernel.coefficient
                        )
                result[start:stop] += diagonal * block_vector

            def retained_rows(local_columns, final_states):
                rows = np.searchsorted(states, final_states)
                if complete_basis:
                    return local_columns, rows

                present = rows < dimension
                if np.any(present):
                    positions = np.flatnonzero(present)
                    present[positions] = (
                        states[rows[positions]] == final_states[positions]
                    )
                return local_columns[present], rows[present]

            for kernel in self._hopping_kernels:
                transition_mask = np.uint64(kernel.transition_mask)
                occupation = block_states & transition_mask

                directions = (
                    (
                        np.uint64(kernel.low_bit),
                        kernel.high_from_low,
                    ),
                    (
                        np.uint64(kernel.high_bit),
                        kernel.low_from_high,
                    ),
                )
                for occupied_pattern, coefficient in directions:
                    if coefficient == 0:
                        continue
                    local_columns = np.flatnonzero(
                        occupation == occupied_pattern
                    )
                    if local_columns.size == 0:
                        continue
                    source_states = block_states[local_columns]
                    final_states = source_states ^ transition_mask
                    kept_local, rows = retained_rows(
                        local_columns,
                        final_states,
                    )
                    if rows.size == 0:
                        continue

                    source_columns = start + kept_local
                    amplitude = self._coerce_coefficient(coefficient)
                    values = vector[source_columns] * amplitude
                    if kernel.parity_mask:
                        parity = self._uint64_parity(
                            states[source_columns]
                            & np.uint64(kernel.parity_mask)
                        )
                        if np.any(parity):
                            values = values.copy()
                            values[parity != 0] *= -1
                    result[rows] += values

            for kernel in self._monomial_kernels:
                annihilated = (
                    kernel.transition_mask & kernel.required_occupied_mask
                ).bit_count()
                created = (
                    kernel.transition_mask & kernel.required_empty_mask
                ).bit_count()
                if created != annihilated:
                    continue

                occupied = np.uint64(kernel.required_occupied_mask)
                empty = np.uint64(kernel.required_empty_mask)
                active = (
                    ((block_states & occupied) == occupied)
                    & ((block_states & empty) == 0)
                )
                local_columns = np.flatnonzero(active)
                if local_columns.size == 0:
                    continue

                source_states = block_states[local_columns]
                final_states = (
                    source_states ^ np.uint64(kernel.transition_mask)
                )
                kept_local, rows = retained_rows(
                    local_columns,
                    final_states,
                )
                if rows.size == 0:
                    continue

                source_columns = start + kept_local
                values = vector[source_columns] * self._coerce_coefficient(
                    kernel.coefficient
                )
                if kernel.parity_mask:
                    parity = self._uint64_parity(
                        states[source_columns]
                        & np.uint64(kernel.parity_mask)
                    )
                    if np.any(parity):
                        values = values.copy()
                        values[parity != 0] *= -1
                result[rows] += values

        return result

    def _matvec_lowered_scalar(self, basis, vector):
        """Apply lowered kernels through the basis rank/state interface.

        This path keeps complete bases above 64 modes implicit: source states
        are obtained by combinadic unranking and destinations by combinadic
        ranking rather than by materializing an arbitrary-width Python-int
        tuple.
        """
        dimension = basis.dimension
        dtype = np.result_type(vector.dtype, self.data_dtype)
        result = np.zeros(dimension, dtype=dtype)

        for column, input_amplitude in enumerate(vector):
            if input_amplitude == 0:
                continue
            state = basis.state_at(column)

            diagonal = 0.0 if self._real_valued else 0.0j
            for kernel in self._number_kernels:
                if state & kernel.mask == kernel.mask:
                    diagonal += kernel.coefficient
            if diagonal != 0:
                result[column] += self._coerce_coefficient(diagonal) * input_amplitude

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

                final_state = state ^ kernel.transition_mask
                try:
                    row = basis.index(final_state)
                except ValueError:
                    continue
                parity = (state & kernel.parity_mask).bit_count() & 1
                amplitude = self._coerce_coefficient(coefficient) * input_amplitude
                result[row] += -amplitude if parity else amplitude

            for kernel in self._monomial_kernels:
                annihilated = (
                    kernel.transition_mask & kernel.required_occupied_mask
                ).bit_count()
                created = (
                    kernel.transition_mask & kernel.required_empty_mask
                ).bit_count()
                if created != annihilated:
                    continue
                if (
                    state & kernel.required_occupied_mask
                    != kernel.required_occupied_mask
                ):
                    continue
                if state & kernel.required_empty_mask:
                    continue

                final_state = state ^ kernel.transition_mask
                try:
                    row = basis.index(final_state)
                except ValueError:
                    continue
                parity = (state & kernel.parity_mask).bit_count() & 1
                amplitude = (
                    self._coerce_coefficient(kernel.coefficient) * input_amplitude
                )
                result[row] += -amplitude if parity else amplitude

        return result

    def _matvec_literal_fallback(self, basis, n_modes, vector):
        """Apply operators containing generic symbolic fallback terms."""
        dimension = basis.dimension
        dtype = np.result_type(vector.dtype, self.data_dtype)
        result = np.zeros(dimension, dtype=dtype)

        for column, coefficient in enumerate(vector):
            if coefficient == 0:
                continue
            ket = FockState(
                basis.state_at(column),
                amp=coefficient,
                n_modes=n_modes,
                index=column,
            )
            output = self.apply(ket)
            for final_state, amplitude in output.items():
                try:
                    row = basis.index(final_state)
                except ValueError:
                    continue
                if self._real_valued:
                    amplitude = complex(amplitude).real
                result[row] += amplitude
        return result

    def matvec(self, basis, n_modes, vector):
        """Apply the compiled operator without materializing a sparse matrix.

        Parameters
        ----------
        basis : FockBasis-like
            Built basis defining the row and column ordering of the projected
            Hilbert space.
        n_modes : int
            Number of fermionic modes represented by each basis state.
        vector : array_like
            One-dimensional state vector with length ``basis.dimension``.

        Returns
        -------
        numpy.ndarray
            Matrix-free Hamiltonian action in the same basis ordering.

        Notes
        -----
        Fully lowered bases with at most 64 modes use the native-word NumPy
        executor. Wider complete bases use scalar combinadic indexing, and
        operators containing generic symbolic kernels use the literal fallback.
        """
        dimension = basis.dimension
        vector = self._validate_matvec_input(vector, dimension)

        if not self.fully_lowered:
            return self._matvec_literal_fallback(basis, n_modes, vector)
        if int(n_modes) <= 64:
            return self._matvec_lowered_uint64(basis, vector)
        return self._matvec_lowered_scalar(basis, vector)

    @property
    def stats(self):
        """dict[str, int]: Counts of lowered kernels by execution category."""
        return {
            "number_product": len(self._number_kernels),
            "hopping": self._hopping_term_count,
            "hopping_groups": len(self._hopping_kernels),
            "monomial": len(self._monomial_kernels),
            "generic": len(self._generic_kernels),
        }


def compile_operator(operator):
    """Compile symbolic fermionic algebra into executable kernels.

    Parameters
    ----------
    operator : Operator
        Literal symbolic operator expression.

    Returns
    -------
    CompiledOperator
        Execution container produced from the lowered intermediate
        representation.
    """
    return CompiledOperator(
        lower_operator(operator)
    )
