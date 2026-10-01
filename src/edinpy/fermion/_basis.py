"""Occupation-number states and bases for fermionic Fock space."""

from __future__ import annotations

from bisect import bisect_left
from math import comb
from numbers import Integral, Number

import numpy as np

from edinpy._core._states import (
    BaseFockState,
    FockBra as _CoreFockBra,
    FockVector as _CoreFockVector,
    NullState as _CoreNullState,
    StateSum as _CoreStateSum,
    _inner_product as _core_inner_product,
)

# Shared Dirac/state-vector primitives. Fermion-specific occupation behavior is
# provided by FockState and FockBasis below; sums, vectors, bras, and the null
# ket use the statistics-independent core implementation.
NullState = _CoreNullState
StateSum = _CoreStateSum
FockVector = _CoreFockVector
FockBra = _CoreFockBra
_inner_product = _core_inner_product


class FockState(BaseFockState):
    """Fermionic occupation-number basis ket represented by an integer bit string.

    Parameters
    ----------
    state : int
        Non-negative integer whose set bits indicate occupied fermionic modes.
    amp : numbers.Number, optional
        State amplitude. The default is one.
    n_modes : int, optional
        Number of fermionic modes used for formatting and validation.
    index : int, optional
        Basis index associated with the state.

    Notes
    -----
    Bit ``p`` records the occupation of fermionic mode ``p``. ``FockState``
    represents one occupation-number basis ket with an optional amplitude.
    Basis-backed superpositions such as eigensolver vectors are represented by
    :class:`FockVector`.
    """

    __slots__ = ("state", "amp", "n_modes", "index")

    def __init__(self, state, amp=1, n_modes=None, index=None):
        """Construct a Fock state from an occupation bit string."""
        if not isinstance(state, Integral):
            raise TypeError("'state' must be an integer bit string.")
        state = int(state)
        if state < 0:
            raise ValueError("'state' must be non-negative.")
        if not isinstance(amp, Number):
            raise TypeError("'amp' must be numeric.")
        if n_modes is not None:
            if not isinstance(n_modes, Integral):
                raise TypeError("'n_modes' must be an integer or None.")
            n_modes = int(n_modes)
            if n_modes < 0:
                raise ValueError("'n_modes' must be non-negative.")
            if state.bit_length() > n_modes:
                raise ValueError("The occupation bit string exceeds 'n_modes'.")
        if index is not None and not isinstance(index, Integral):
            raise TypeError("'index' must be an integer or None.")

        self.state = state
        self.amp = amp
        self.n_modes = n_modes
        self.index = None if index is None else int(index)

    def _with_amp(self, amp):
        """Return the same fermionic occupation ket with a new amplitude."""
        return FockState(
            self.state,
            amp=amp,
            n_modes=self.n_modes,
            index=self.index,
        )

    def __pos__(self):
        """Return the state unchanged."""
        return self

    def __neg__(self):
        """Return the state with the amplitude multiplied by minus one."""
        return FockState(
            self.state,
            amp=-self.amp,
            n_modes=self.n_modes,
            index=self.index,
        )

    def __add__(self, other):
        """Add another Fock state and return a simplified state sum."""
        if isinstance(other, NullState):
            return self
        if isinstance(other, FockState):
            return StateSum((self, other)).simplified()
        if isinstance(other, StateSum):
            return StateSum((self, *other.states)).simplified()
        return NotImplemented

    def __sub__(self, other):
        """Subtract another Fock state and return a simplified state sum."""
        if isinstance(other, (FockState, StateSum, NullState)):
            return self + (-other)
        return NotImplemented

    def __mul__(self, scalar):
        """Multiply the state amplitude by a numerical scalar."""
        if not isinstance(scalar, Number):
            return NotImplemented
        return FockState(
            self.state,
            amp=self.amp * scalar,
            n_modes=self.n_modes,
            index=self.index,
        )

    def __rmul__(self, scalar):
        """Multiply the state amplitude by a numerical scalar."""
        return self * scalar

    def __eq__(self, other):
        """Compare occupation bit strings and amplitudes."""
        return (
            isinstance(other, FockState)
            and self.state == other.state
            and self.amp == other.amp
        )

    def __str__(self):
        """Return the occupation-number representation of the state."""
        width = self.n_modes if self.n_modes is not None else self.state.bit_length()
        width = max(width, 1)
        return f"{self.amp} |{self.state:0{width}b}>"

    def __repr__(self):
        """Return an unambiguous representation of the Fock state."""
        return (
            f"FockState(state={self.state}, amp={self.amp!r}, "
            f"n_modes={self.n_modes!r}, index={self.index!r})"
        )

    @property
    def dag(self):
        """FockBra: Hermitian adjoint of the ket."""
        return FockBra(self)

    def inner(self, other):
        """Return the inner product with another fermionic ket.

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
        """Return a unit-normalized copy of the state.

        Raises
        ------
        ValueError
            If the state has zero amplitude.
        """
        norm = self.norm()
        if norm == 0:
            raise ValueError("The zero state cannot be normalized.")
        return (1 / norm) * self

    def is_occupied(self, mode):
        """Return whether a fermionic mode is occupied.

        Parameters
        ----------
        mode : int
            Zero-based fermionic-mode index.

        Returns
        -------
        bool
            ``True`` when the mode is occupied.
        """
        if not isinstance(mode, Integral):
            raise TypeError("'mode' must be an integer.")
        mode = int(mode)
        if mode < 0:
            raise IndexError("'mode' must be non-negative.")
        if self.n_modes is not None and mode >= self.n_modes:
            raise IndexError(f"Mode {mode} is outside [0, {self.n_modes}).")
        return bool(self.state & (1 << mode))


class FockBasis:
    """Occupation-number basis for a fixed number of fermions.

    Parameters
    ----------
    n_modes : int
        Number of fermionic modes.
    N : int
        Number of fermions. Must satisfy ``0 <= N <= n_modes``.

    Notes
    -----
    Complete fixed-particle-number bases are represented implicitly. Their
    dimension and state-index maps follow combinatorial ranking, so the full
    tuple of occupation bit strings is created only when :attr:`states` is
    requested. Bases with additional particle-number constraints retain only
    the allowed occupation states.

    Basis states are ordered by ascending occupation bit string, equivalent to
    combinadic colex order at fixed particle number.
    """

    __slots__ = (
        "n_modes",
        "N",
        "_complete",
        "_dimension",
        "_states",
        "_state_array",
        "_state_words",
    )

    def __init__(self, n_modes, N):
        """Construct an implicit complete fixed-``N`` occupation basis."""
        if not isinstance(n_modes, Integral) or not isinstance(N, Integral):
            raise TypeError("'n_modes' and 'N' must be integers.")
        n_modes = int(n_modes)
        N = int(N)
        if n_modes < 0:
            raise ValueError("'n_modes' must be non-negative.")
        if N < 0 or N > n_modes:
            raise ValueError("'N' must satisfy 0 <= N <= n_modes.")

        self.n_modes = n_modes
        self.N = N
        self._complete = True
        self._dimension = comb(n_modes, N)
        self._states = None
        self._state_array = None
        self._state_words = None

    @staticmethod
    def _enumerate_states(n_modes, N):
        """Return complete fixed-population states in ascending integer order.

        This compatibility helper intentionally materializes a tuple. Normal
        complete-basis construction no longer calls it.
        """
        return tuple(FockBasis(n_modes, N))

    @classmethod
    def _from_sorted_states(cls, n_modes, N, states):
        """Construct a projected basis from trusted sorted integer states."""
        basis = cls.__new__(cls)
        basis.n_modes = int(n_modes)
        basis.N = int(N)
        basis._complete = False

        basis._states = None
        basis._state_array = None
        basis._state_words = None
        if basis.n_modes <= 64:
            array = np.fromiter(
                (int(state) for state in states),
                dtype=np.uint64,
            )
            array.setflags(write=False)
            basis._state_array = array
            basis._dimension = int(array.size)
        else:
            source = states if hasattr(states, "__len__") else tuple(states)
            dimension = len(source)
            n_words = (basis.n_modes + 63) // 64
            words = np.empty((dimension, n_words), dtype=np.uint64)
            word_mask = (1 << 64) - 1
            for word in range(n_words):
                shift = 64 * word
                words[:, word] = np.fromiter(
                    (
                        (int(state) >> shift) & word_mask
                        for state in source
                    ),
                    dtype=np.uint64,
                    count=dimension,
                )
            words.setflags(write=False)
            basis._state_words = words
            basis._dimension = dimension
        return basis

    def __len__(self):
        """Return the Hilbert-space dimension of the basis."""
        return self._dimension

    def __iter__(self):
        """Iterate over integer occupation bit strings in basis order."""
        if self._states is not None:
            yield from self._states
            return
        if self._state_array is not None:
            yield from (int(state) for state in self._state_array)
            return
        if self._state_words is not None:
            for index in range(self._dimension):
                yield self._state_from_words(index)
            return

        if self.N == 0:
            yield 0
            return
        if self.N == self.n_modes:
            yield (1 << self.n_modes) - 1
            return

        state = (1 << self.N) - 1
        for index in range(self._dimension):
            yield state
            if index + 1 == self._dimension:
                break
            lowest = state & -state
            ripple = state + lowest
            state = ripple | (((state ^ ripple) >> 2) // lowest)

    def __getitem__(self, index):
        """Return an occupation bit string by basis index or slice."""
        if isinstance(index, slice):
            return tuple(self.state_at(i) for i in range(*index.indices(len(self))))
        return self.state_at(index)

    @property
    def is_complete(self):
        """bool: Whether this is the complete fixed-particle-number basis."""
        return self._complete

    @property
    def is_materialized(self):
        """bool: Whether any explicit state storage has been allocated."""
        return (
            self._states is not None
            or self._state_array is not None
            or self._state_words is not None
        )

    @property
    def estimated_execution_storage_bytes(self):
        """int: Persistent compact storage expected by matrix-free execution.

        Complete bases up to 64 modes use one ``uint64`` per state. Complete
        wider bases remain implicit. Projected bases report their existing
        compact storage. Calling this property never materializes states.
        """
        if not self._complete:
            return self.storage_bytes
        if self.n_modes <= 64:
            return self._dimension * np.dtype(np.uint64).itemsize
        return 0

    @property
    def storage_bytes(self):
        """int: Bytes used by cached explicit basis-state storage.

        The value is zero for a freshly constructed implicit complete basis.
        It excludes the small fixed-size Python ``FockBasis`` object itself.
        """
        import sys

        total = 0
        if self._state_array is not None:
            total += int(self._state_array.nbytes)
        if self._state_words is not None:
            total += int(self._state_words.nbytes)
        if self._states is not None:
            total += sys.getsizeof(self._states)
            total += sum(sys.getsizeof(state) for state in self._states)
        return total

    def _complete_rank(self, state):
        """Return the combinadic rank of a validated complete-basis state."""
        if state < 0 or state.bit_length() > self.n_modes:
            raise ValueError("The occupation state is not contained in this basis.")
        if state.bit_count() != self.N:
            raise ValueError("The occupation state is not contained in this basis.")

        rank = 0
        occupied_index = 1
        value = state
        position = 0
        while value:
            if value & 1:
                rank += comb(position, occupied_index)
                occupied_index += 1
            value >>= 1
            position += 1
        return rank

    def _complete_state_at(self, index):
        """Return one complete-basis state by combinadic unranking."""
        if self.N == 0:
            return 0

        rank = index
        state = 0
        upper = self.n_modes - 1
        for order in range(self.N, 0, -1):
            low = order - 1
            high = upper
            while low < high:
                middle = (low + high + 1) // 2
                if comb(middle, order) <= rank:
                    low = middle
                else:
                    high = middle - 1
            position = low
            state |= 1 << position
            rank -= comb(position, order)
            upper = position - 1
        return state

    def _state_from_words(self, index):
        """Reconstruct one arbitrary-width state from compact uint64 words."""
        state = 0
        for word, value in enumerate(self._state_words[index]):
            state |= int(value) << (64 * word)
        return state

    def state_at(self, index):
        """Return the integer occupation bit string at one basis index.

        Parameters
        ----------
        index : int
            Zero-based basis index. Negative indices follow Python indexing.

        Returns
        -------
        int
            Integer whose set bits mark occupied fermionic modes.
        """
        if not isinstance(index, Integral):
            raise TypeError("'index' must be an integer.")
        index = int(index)
        if index < 0:
            index += self._dimension
        if index < 0 or index >= self._dimension:
            raise IndexError("Fock-basis index is out of range.")

        if self._states is not None:
            return self._states[index]
        if self._state_array is not None:
            return int(self._state_array[index])
        if self._state_words is not None:
            return self._state_from_words(index)
        return self._complete_state_at(index)

    def _execution_states(self):
        """Return a compact native-word execution view when available.

        Complete bases up to 64 modes are enumerated lazily into a read-only
        ``uint64`` array. This avoids the much larger permanent tuple of Python
        integers while preserving vectorized execution.
        """
        if self.n_modes > 64:
            raise RuntimeError(
                "Native-word execution states are available only for <=64 modes."
            )
        if self._state_array is None:
            array = np.fromiter(
                self,
                dtype=np.uint64,
                count=self._dimension,
            )
            array.setflags(write=False)
            self._state_array = array
        return self._state_array

    def __contains__(self, state):
        """Return whether an integer occupation bit string belongs to the basis."""
        if isinstance(state, FockState):
            state = state.state
        if not isinstance(state, Integral):
            return False
        try:
            self.index(int(state))
        except ValueError:
            return False
        return True

    def index(self, state):
        """Return the basis index of an occupation-number state.

        Parameters
        ----------
        state : int or FockState
            Fermionic occupation bit string or basis ket.

        Returns
        -------
        int
            Zero-based position in this basis.

        Raises
        ------
        ValueError
            If the occupation state does not belong to the basis.
        """
        if isinstance(state, FockState):
            state = state.state
        if not isinstance(state, Integral):
            raise TypeError("'state' must be an integer or FockState.")
        state = int(state)

        if self._complete:
            return self._complete_rank(state)

        if self._state_array is not None:
            if state < 0 or state.bit_length() > 64:
                raise ValueError("The occupation state is not contained in this basis.")
            position = int(np.searchsorted(self._state_array, np.uint64(state)))
            if (
                position >= self._dimension
                or int(self._state_array[position]) != state
            ):
                raise ValueError("The occupation state is not contained in this basis.")
            return position

        if self._state_words is not None:
            if state < 0 or state.bit_length() > self.n_modes:
                raise ValueError("The occupation state is not contained in this basis.")
            low = 0
            high = self._dimension
            while low < high:
                middle = (low + high) // 2
                candidate = self._state_from_words(middle)
                if candidate < state:
                    low = middle + 1
                else:
                    high = middle
            if low >= self._dimension or self._state_from_words(low) != state:
                raise ValueError("The occupation state is not contained in this basis.")
            return low

        position = bisect_left(self._states, state)
        if position >= self._dimension or self._states[position] != state:
            raise ValueError("The occupation state is not contained in this basis.")
        return position

    @property
    def states(self):
        """tuple[int, ...]: Integer occupation states in basis order.

        For an implicit or compact basis this compatibility tuple is created
        lazily on first access.
        """
        if self._states is None:
            self._states = tuple(self)
        return self._states

    @property
    def dimension(self):
        """int: Number of basis states."""
        return self._dimension

    def state(self, index):
        """Return one unit-amplitude basis ket.

        Parameters
        ----------
        index : int
            Zero-based basis index. Negative indices follow Python indexing.

        Returns
        -------
        FockState
            Basis ket with its ``index`` field set to the resolved index.
        """
        index = int(index) if isinstance(index, Integral) else index
        value = self.state_at(index)
        if index < 0:
            index += self._dimension
        return FockState(value, n_modes=self.n_modes, index=index)

    @staticmethod
    def expected_dimension(n_modes, N):
        """Return the dimension of the complete fixed-particle-number basis.

        Parameters
        ----------
        n_modes : int
            Number of fermionic modes.
        N : int
            Number of fermions.

        Returns
        -------
        int
            Exact binomial coefficient ``binomial(n_modes, N)``.
        """
        if not isinstance(n_modes, Integral) or not isinstance(N, Integral):
            raise TypeError("'n_modes' and 'N' must be integers.")
        n_modes = int(n_modes)
        N = int(N)
        if n_modes < 0:
            raise ValueError("'n_modes' must be non-negative.")
        if N < 0 or N > n_modes:
            raise ValueError("'N' must satisfy 0 <= N <= n_modes.")
        return comb(n_modes, N)
