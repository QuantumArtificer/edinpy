"""Occupation-number states and bases for bosonic Fock space."""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Iterable
from math import comb
from numbers import Integral, Number

import numpy as np
from numpy.typing import NDArray

from edinpy._core._states import (
    BaseFockState,
    FockBra as _CoreFockBra,
    FockVector as _CoreFockVector,
    NullState as _CoreNullState,
    StateSum as _CoreStateSum,
    _inner_product as _core_inner_product,
)

# Shared Dirac/state-vector primitives. Bosonic state-specific behavior is
# provided by FockState and FockBasis below; sums, vectors, bras, and the null
# ket use the statistics-independent core implementation.
NullState = _CoreNullState
StateSum = _CoreStateSum
FockVector = _CoreFockVector
FockBra = _CoreFockBra
_inner_product = _core_inner_product


class _OccupationCodec:
    """Pack fixed-sector occupations into exact Python integer keys.

    Parameters
    ----------
    n_modes : int
        Number of bosonic modes represented by every packed state.
    N : int
        Fixed total particle number. It determines the minimum number of bits
        reserved for each mode occupation.
    """

    __slots__ = ("n_modes", "N", "bits_per_mode", "mask", "total_bits")

    def __init__(self, n_modes: int, N: int) -> None:
        if not isinstance(n_modes, Integral) or not isinstance(N, Integral):
            raise TypeError("'n_modes' and 'N' must be integers.")
        n_modes = int(n_modes)
        N = int(N)
        if n_modes <= 0:
            raise ValueError("'n_modes' must be positive.")
        if N < 0:
            raise ValueError("'N' must be non-negative.")

        bits_per_mode = max(1, N.bit_length())
        self.n_modes = n_modes
        self.N = N
        self.bits_per_mode = bits_per_mode
        self.mask = (1 << bits_per_mode) - 1
        self.total_bits = bits_per_mode * n_modes

    def pack(self, occupations: Iterable[int], *, require_total: bool = True) -> int:
        """Return the exact packed key for one occupation vector."""
        try:
            occupations = tuple(occupations)
        except TypeError as exc:
            raise TypeError("'occupations' must be an iterable of integers.") from exc
        if len(occupations) != self.n_modes:
            raise ValueError(
                f"Expected {self.n_modes} occupations, received "
                f"{len(occupations)}."
            )

        state = 0
        total = 0
        for mode, occupation in enumerate(occupations):
            if not isinstance(occupation, Integral):
                raise TypeError("Bosonic occupations must be integers.")
            occupation = int(occupation)
            if occupation < 0:
                raise ValueError("Bosonic occupations must be non-negative.")
            if occupation > self.N:
                raise ValueError(
                    f"Occupation {occupation} exceeds the sector particle "
                    f"number N={self.N}."
                )
            total += occupation
            state |= occupation << (mode * self.bits_per_mode)

        if require_total and total != self.N:
            raise ValueError(
                f"Occupations sum to {total}, but this basis has N={self.N}."
            )
        return state

    def unpack(self, state: int) -> tuple[int, ...]:
        """Decode one packed key into an occupation tuple."""
        if not isinstance(state, Integral):
            raise TypeError("Packed bosonic states must be integers.")
        state = int(state)
        if state < 0:
            raise ValueError("Packed bosonic states must be non-negative.")
        if state.bit_length() > self.total_bits:
            raise ValueError("Packed state exceeds the codec width.")
        return tuple(
            (state >> (mode * self.bits_per_mode)) & self.mask
            for mode in range(self.n_modes)
        )

    def occupation(self, state: int, mode: int) -> int:
        """Return the occupation of one mode from a packed key."""
        if not isinstance(mode, Integral):
            raise TypeError("'mode' must be an integer.")
        mode = int(mode)
        if mode < 0 or mode >= self.n_modes:
            raise IndexError(f"Mode {mode} is outside [0, {self.n_modes}).")
        return (int(state) >> (mode * self.bits_per_mode)) & self.mask


class FockState(BaseFockState):
    """Bosonic occupation-number basis ket.

    Parameters
    ----------
    occupations : iterable of int
        Non-negative occupation number for each bosonic mode.
    amp : numbers.Number, optional
        State amplitude. The default is one.
    index : int, optional
        Basis index associated with the state.
    """

    __slots__ = ("occupations", "amp", "index")

    def __init__(
        self, occupations: Iterable[int], amp: Number = 1, index: int | None = None
    ) -> None:
        try:
            occupations = tuple(occupations)
        except TypeError as exc:
            raise TypeError("'occupations' must be an iterable of integers.") from exc
        if not occupations:
            raise ValueError("A bosonic Fock state requires at least one mode.")
        if not all(isinstance(value, Integral) for value in occupations):
            raise TypeError("Bosonic occupations must be integers.")
        occupations = tuple(int(value) for value in occupations)
        if any(value < 0 for value in occupations):
            raise ValueError("Bosonic occupations must be non-negative.")
        if not isinstance(amp, Number):
            raise TypeError("'amp' must be numeric.")
        if index is not None and not isinstance(index, Integral):
            raise TypeError("'index' must be an integer or None.")

        self.occupations = occupations
        self.amp = amp
        self.index = None if index is None else int(index)

    def _with_amp(self, amp):
        """Return the same bosonic occupation ket with a new amplitude."""
        return FockState(self.occupations, amp=amp, index=self.index)

    @property
    def state(self):
        """tuple[int, ...]: Occupation-number tuple."""
        return self.occupations

    @property
    def N(self):
        """int: Total particle number in the state."""
        return sum(self.occupations)

    @property
    def n_modes(self):
        """int: Number of bosonic modes represented by the state."""
        return len(self.occupations)

    def occupation(self, mode):
        """Return the occupation number of one bosonic mode.

        Parameters
        ----------
        mode : int
            Zero-based bosonic mode index.

        Returns
        -------
        int
            Occupation number of ``mode``.
        """
        if not isinstance(mode, Integral):
            raise TypeError("'mode' must be an integer.")
        mode = int(mode)
        if mode < 0 or mode >= self.n_modes:
            raise IndexError(f"Mode {mode} is outside [0, {self.n_modes}).")
        return self.occupations[mode]

    def is_occupied(self, mode):
        """Return whether one bosonic mode has nonzero occupation.

        Parameters
        ----------
        mode : int
            Zero-based bosonic mode index.

        Returns
        -------
        bool
            ``True`` when the occupation of ``mode`` is nonzero.
        """
        return self.occupation(mode) != 0

    def __pos__(self):
        """Return the state unchanged."""
        return self

    def __neg__(self):
        """Return the state with its amplitude multiplied by minus one."""
        return FockState(self.occupations, amp=-self.amp, index=self.index)

    def __add__(self, other):
        """Add another bosonic ket and simplify equal occupation states."""
        if isinstance(other, NullState):
            return self
        if isinstance(other, FockState):
            return StateSum((self, other)).simplified()
        if isinstance(other, StateSum):
            return StateSum((self, *other.states)).simplified()
        return NotImplemented

    def __radd__(self, other):
        """Add from the left, including Python's ``sum`` identity zero."""
        if other == 0:
            return self
        return self + other

    def __sub__(self, other):
        """Subtract another bosonic ket."""
        if isinstance(other, (FockState, StateSum, NullState)):
            return self + (-other)
        return NotImplemented

    def __mul__(self, scalar):
        """Multiply the state amplitude by a numerical scalar."""
        if not isinstance(scalar, Number):
            return NotImplemented
        return FockState(
            self.occupations,
            amp=self.amp * scalar,
            index=self.index,
        )

    def __rmul__(self, scalar):
        """Multiply the state amplitude by a numerical scalar."""
        return self * scalar

    def __eq__(self, other):
        """Compare occupation vectors and amplitudes."""
        return (
            isinstance(other, FockState)
            and self.occupations == other.occupations
            and self.amp == other.amp
        )

    def __str__(self):
        """Return the occupation-number representation of the state."""
        contents = ",".join(str(value) for value in self.occupations)
        return f"{self.amp} |{contents}>"

    def __repr__(self):
        """Return an unambiguous representation of the Fock state."""
        return (
            f"FockState(occupations={self.occupations!r}, amp={self.amp!r}, "
            f"index={self.index!r})"
        )

    @property
    def dag(self):
        """FockBra: Hermitian adjoint of the ket."""
        return FockBra(self)

    def inner(self, other):
        """Return the Hilbert-space inner product with another bosonic ket.

        Parameters
        ----------
        other : FockState, StateSum, FockVector, or NullState
            Ket on the right-hand side of the inner product.

        Returns
        -------
        numbers.Number
            Inner product ``<self|other>``.
        """
        return _inner_product(self, other)

    def norm(self):
        """Return the Hilbert-space norm of the state."""
        return float(abs(self.amp))

    def normalized(self):
        """Return a unit-normalized copy of the state."""
        norm = self.norm()
        if norm == 0:
            raise ValueError("The zero state cannot be normalized.")
        return (1 / norm) * self


class FockBasis:
    """Occupation-number basis for a fixed number of bosons.

    Parameters
    ----------
    n_modes : int
        Number of bosonic modes.
    N : int
        Total number of indistinguishable bosons in every basis state.

    Notes
    -----
    Complete fixed-particle-number bases are represented implicitly using the
    weak-composition structure of bosonic occupations. Decoded occupation
    tuples are created only when :attr:`states` is requested. Bases with
    additional particle-number constraints retain only the allowed occupation
    configurations. Packed integer keys provide an exact compact encoding for
    indexing and storage.
    """

    __slots__ = (
        "n_modes",
        "N",
        "_codec",
        "_packed_states",
        "_packed_words",
        "_composition_ranks",
        "_complete",
        "_dimension",
    )

    def __init__(self, n_modes: int, N: int) -> None:
        """Construct the complete weak-composition basis without materializing it."""
        self._codec = _OccupationCodec(n_modes, N)
        self.n_modes = self._codec.n_modes
        self.N = self._codec.N
        self._dimension = comb(self.N + self.n_modes - 1, self.N)
        self._complete = True
        self._packed_states = None
        self._packed_words = None
        self._composition_ranks = None

    @staticmethod
    def _iter_packed_states(n_modes, N):
        """Yield weak compositions directly in packed-key order."""
        codec = _OccupationCodec(n_modes, N)
        bits = codec.bits_per_mode

        def recurse(mode, remaining, packed):
            if mode == 0:
                yield packed | remaining
                return
            shift = mode * bits
            for occupation in range(remaining + 1):
                yield from recurse(
                    mode - 1,
                    remaining - occupation,
                    packed | (occupation << shift),
                )

        yield from recurse(n_modes - 1, N, 0)

    @staticmethod
    def _enumerate_packed_states(n_modes, N):
        """Return the complete packed-state tuple for compatibility."""
        return tuple(FockBasis._iter_packed_states(n_modes, N))

    @staticmethod
    def _words_from_packed_states(packed_states, total_bits):
        """Return a compact read-only little-endian multiword state array."""
        if hasattr(packed_states, "__len__"):
            source = packed_states
        else:
            source = tuple(packed_states)
        dimension = len(source)
        n_words = max(1, (int(total_bits) + 63) // 64)
        words = np.empty((dimension, n_words), dtype=np.uint64)
        word_mask = (1 << 64) - 1
        for word in range(n_words):
            shift = 64 * word
            words[:, word] = np.fromiter(
                ((int(state) >> shift) & word_mask for state in source),
                dtype=np.uint64,
                count=dimension,
            )
        words.setflags(write=False)
        return words

    @classmethod
    def _from_sorted_packed_states(cls, n_modes, N, packed_states):
        """Construct a compact projected basis from trusted sorted packed keys."""
        basis = cls.__new__(cls)
        basis._codec = _OccupationCodec(n_modes, N)
        basis.n_modes = basis._codec.n_modes
        basis.N = basis._codec.N
        basis._complete = False
        basis._packed_states = None
        basis._packed_words = cls._words_from_packed_states(
            packed_states,
            basis._codec.total_bits,
        )
        basis._dimension = int(basis._packed_words.shape[0])
        basis._composition_ranks = None
        return basis

    def __len__(self):
        return self._dimension

    def __iter__(self):
        for index in range(self._dimension):
            yield self.state(index)

    def __getitem__(self, index):
        if isinstance(index, slice):
            indices = range(*index.indices(len(self)))
            return tuple(self[item] for item in indices)
        return self.state(index)

    @property
    def is_complete(self):
        """bool: Whether this is the complete fixed-particle-number basis."""
        return self._complete

    @property
    def is_materialized(self):
        """bool: Whether explicit packed-state storage has been allocated."""
        return self._packed_states is not None or self._packed_words is not None

    @property
    def estimated_execution_storage_bytes(self):
        """int: Persistent compact storage expected by matrix-free execution.

        Bosonic matrix-free kernels cache a little-endian multiword ``uint64``
        state view. Projected ranked execution may additionally cache one
        ``int64`` complete-basis rank per retained state. The estimate is
        computed without materializing either array.
        """
        n_words = max(1, (self.packed_width + 63) // 64)
        words = self._dimension * n_words * np.dtype(np.uint64).itemsize
        ranks = 0 if self._complete else self._dimension * np.dtype(np.int64).itemsize
        return words + ranks

    @property
    def storage_bytes(self):
        """int: Bytes used by cached explicit basis-state storage."""
        import sys

        total = 0
        if self._packed_words is not None:
            total += int(self._packed_words.nbytes)
        if self._packed_states is not None:
            total += sys.getsizeof(self._packed_states)
            total += sum(sys.getsizeof(state) for state in self._packed_states)
        if self._composition_ranks is not None:
            total += int(self._composition_ranks.nbytes)
        return total

    @property
    def dimension(self):
        """int: Number of basis states."""
        return self._dimension

    @property
    def bits_per_mode(self):
        """int: Number of bits allocated to each occupation field."""
        return self._codec.bits_per_mode

    @property
    def packed_width(self):
        """int: Total number of logical bits in each packed key."""
        return self._codec.total_bits

    def _composition_index(self, occupations):
        """Return complete-basis rank of one weak composition."""
        occupations = tuple(int(value) for value in occupations)
        if len(occupations) != self.n_modes:
            raise ValueError("Occupation vector length does not match n_modes.")
        if any(value < 0 for value in occupations) or sum(occupations) != self.N:
            raise ValueError("State is outside the fixed-N bosonic sector.")

        combinadic_rank = 0
        cumulative = 0
        for boundary in range(self.n_modes - 1):
            cumulative += occupations[boundary]
            if cumulative:
                combinadic_rank += comb(
                    cumulative + boundary,
                    boundary + 1,
                )
        return self._dimension - 1 - combinadic_rank

    def _composition_at(self, index):
        """Return one weak composition by reverse combinadic unranking."""
        if self.n_modes == 1:
            return (self.N,)

        rank = self._dimension - 1 - index
        n_bars = self.n_modes - 1
        upper = self.N + self.n_modes - 2
        bars = [0] * n_bars

        for order in range(n_bars, 0, -1):
            low = order - 1
            high = upper
            while low < high:
                middle = (low + high + 1) // 2
                if comb(middle, order) <= rank:
                    low = middle
                else:
                    high = middle - 1
            bars[order - 1] = low
            rank -= comb(low, order)
            upper = low - 1

        occupations = [0] * self.n_modes
        previous = 0
        for boundary, bar in enumerate(bars):
            cumulative = bar - boundary
            occupations[boundary] = cumulative - previous
            previous = cumulative
        occupations[-1] = self.N - previous
        return tuple(occupations)

    def _packed_from_words(self, index):
        """Reconstruct one exact Python-int key from the compact word array."""
        packed = 0
        for word, value in enumerate(self._packed_words[index]):
            packed |= int(value) << (64 * word)
        return packed

    def packed_at(self, index):
        """Return the packed occupation key at one basis index.

        Parameters
        ----------
        index : int
            Zero-based basis index. Negative indices follow Python indexing.

        Returns
        -------
        int
            Exact packed representation of the bosonic occupation vector.
        """
        if not isinstance(index, Integral):
            raise TypeError("'index' must be an integer.")
        index = int(index)
        if index < 0:
            index += self._dimension
        if index < 0 or index >= self._dimension:
            raise IndexError("Fock-basis index is out of range.")

        if self._packed_states is not None:
            return self._packed_states[index]
        if self._packed_words is not None:
            return self._packed_from_words(index)
        return self._codec.pack(self._composition_at(index))

    @property
    def packed_states(self):
        """tuple[int, ...]: Exact packed occupation keys in basis order.

        This compatibility tuple is created lazily. Matrix-free execution uses
        the compact multiword view directly and does not require it.
        """
        if self._packed_states is None:
            if self._packed_words is None and self._complete:
                self._packed_states = tuple(
                    self._iter_packed_states(self.n_modes, self.N)
                )
            else:
                self._packed_states = tuple(
                    self.packed_at(index) for index in range(self._dimension)
                )
        return self._packed_states

    @property
    def states(self):
        """tuple[tuple[int, ...], ...]: Decoded occupation vectors."""
        return tuple(
            self._codec.unpack(self.packed_at(index))
            for index in range(self._dimension)
        )

    def _uint64_words(self) -> NDArray[np.uint64]:
        """Return a cached compact little-endian ``uint64`` execution view."""
        if self._packed_words is None:
            n_words = max(1, (self.packed_width + 63) // 64)
            if n_words == 1:
                words = np.fromiter(
                    self._iter_packed_states(self.n_modes, self.N),
                    dtype=np.uint64,
                    count=self._dimension,
                ).reshape(self._dimension, 1)
            else:
                words = np.empty(
                    (self._dimension, n_words),
                    dtype=np.uint64,
                )
                word_mask = (1 << 64) - 1
                for row, state in enumerate(
                    self._iter_packed_states(self.n_modes, self.N)
                ):
                    for word in range(n_words):
                        words[row, word] = (
                            state >> (64 * word)
                        ) & word_mask
            words.setflags(write=False)
            self._packed_words = words
        return self._packed_words

    def _complete_composition_ranks(self):
        """Return retained-state ranks in the complete fixed-N bosonic basis."""
        if self._composition_ranks is not None:
            return self._composition_ranks

        full_dimension = comb(self.N + self.n_modes - 1, self.N)
        if full_dimension - 1 > np.iinfo(np.int64).max:
            return None

        if self._complete:
            ranks = np.arange(self._dimension, dtype=np.int64)
            ranks.setflags(write=False)
            self._composition_ranks = ranks
            return ranks

        words = self._uint64_words()
        bits = self.bits_per_mode
        mask = (1 << bits) - 1
        mask64 = np.uint64(mask)
        ranks = np.full(self._dimension, full_dimension - 1, dtype=np.int64)
        cumulative = np.zeros(self._dimension, dtype=np.int64)

        for boundary in range(self.n_modes - 1):
            bit_offset = boundary * bits
            word = bit_offset // 64
            offset = bit_offset % 64
            if offset + bits <= 64:
                occupation = (
                    (words[:, word] >> np.uint64(offset)) & mask64
                ).astype(np.int64, copy=False)
            else:
                low = words[:, word] >> np.uint64(offset)
                high = words[:, word + 1] << np.uint64(64 - offset)
                occupation = ((low | high) & mask64).astype(np.int64, copy=False)

            cumulative += occupation
            table = np.fromiter(
                (
                    0
                    if population == 0
                    else comb(population + boundary, boundary + 1)
                    for population in range(self.N + 1)
                ),
                dtype=np.int64,
                count=self.N + 1,
            )
            ranks -= table[cumulative]

        ranks.setflags(write=False)
        self._composition_ranks = ranks
        return ranks

    def pack(self, occupations: Iterable[int]) -> int:
        """Encode a bosonic occupation vector as an exact integer key.

        Parameters
        ----------
        occupations : iterable of int
            Non-negative occupation of each mode. The occupations must sum to
            this basis's particle number ``N``.

        Returns
        -------
        int
            Packed occupation key.
        """
        return self._codec.pack(occupations)

    def unpack(self, packed_state: int) -> tuple[int, ...]:
        """Decode an exact packed state key.

        Parameters
        ----------
        packed_state : int
            Packed bosonic occupation key.

        Returns
        -------
        tuple[int, ...]
            Occupation number of every mode.
        """
        occupations = self._codec.unpack(packed_state)
        if sum(occupations) != self.N:
            raise ValueError("Packed state is outside the fixed-N sector.")
        return occupations

    def occupation(self, packed_state: int, mode: int) -> int:
        """Return one mode occupation from a packed state key.

        Parameters
        ----------
        packed_state : int
            Packed bosonic occupation key.
        mode : int
            Zero-based bosonic mode index.

        Returns
        -------
        int
            Occupation number of ``mode``.
        """
        return self._codec.occupation(packed_state, mode)

    def index(self, state):
        """Return the basis index of a bosonic occupation state.

        Parameters
        ----------
        state : FockState, int, or iterable of int
            Basis ket, packed state key, or occupation-number vector.

        Returns
        -------
        int
            Zero-based position in this basis.

        Raises
        ------
        ValueError
            If the state does not belong to the basis.
        """
        if isinstance(state, FockState):
            occupations = state.occupations
            packed = self.pack(occupations)
        elif isinstance(state, Integral):
            packed = int(state)
            occupations = self.unpack(packed)
        else:
            occupations = tuple(state)
            packed = self.pack(occupations)

        if self._complete:
            return self._composition_index(occupations)

        if self._packed_words.shape[1] == 1:
            column = self._packed_words[:, 0]
            if packed < 0 or packed > np.iinfo(np.uint64).max:
                raise ValueError("State is not present in this Fock basis.")
            position = int(np.searchsorted(column, np.uint64(packed)))
            if position >= self._dimension or int(column[position]) != packed:
                raise ValueError("State is not present in this Fock basis.")
            return position

        low = 0
        high = self._dimension
        while low < high:
            middle = (low + high) // 2
            candidate = self._packed_from_words(middle)
            if candidate < packed:
                low = middle + 1
            else:
                high = middle
        if low >= self._dimension or self._packed_from_words(low) != packed:
            raise ValueError("State is not present in this Fock basis.")
        return low

    def __contains__(self, state):
        try:
            self.index(state)
        except (TypeError, ValueError):
            return False
        return True

    def state(self, index: int) -> FockState:
        """Return one unit-amplitude bosonic basis ket.

        Parameters
        ----------
        index : int
            Zero-based basis index. Negative indices follow Python indexing.

        Returns
        -------
        FockState
            Basis ket with its ``index`` field set to the resolved index.
        """
        if not isinstance(index, Integral):
            raise TypeError("'index' must be an integer.")
        index = int(index)
        if index < 0:
            index += self._dimension
        packed = self.packed_at(index)
        return FockState(self._codec.unpack(packed), index=index)

    def state_at(self, index: int) -> FockState:
        """Return one basis ket at ``index`` without materializing :attr:`states`."""
        return self.state(index)

    @staticmethod
    def expected_dimension(n_modes: int, N: int) -> int:
        """Return the dimension of the complete fixed-particle-number basis.

        Parameters
        ----------
        n_modes : int
            Number of bosonic modes.
        N : int
            Number of bosons.

        Returns
        -------
        int
            Exact binomial coefficient ``binomial(N + n_modes - 1, N)``.
        """
        if not isinstance(n_modes, Integral) or not isinstance(N, Integral):
            raise TypeError("'n_modes' and 'N' must be integers.")
        n_modes = int(n_modes)
        N = int(N)
        if n_modes <= 0:
            raise ValueError("'n_modes' must be positive.")
        if N < 0:
            raise ValueError("'N' must be non-negative.")
        return comb(N + n_modes - 1, N)
