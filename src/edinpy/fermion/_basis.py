"""Occupation-number states and fixed-particle-number fermionic sectors."""

from __future__ import annotations

from bisect import bisect_left
from math import comb
from numbers import Integral, Number

from ._modes import FermionModes


class FockState:
    """Fermionic occupation-number state represented by an integer bit string.

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


class NullState:
    """Additive zero for symbolic Fock-state actions."""

    __slots__ = ()
    amp = 0

    def __neg__(self):
        """Return the additive zero unchanged."""
        return self

    def __add__(self, other):
        """Return the other state when adding the additive zero."""
        if isinstance(other, (FockState, StateSum, NullState)):
            return other
        return NotImplemented

    def __radd__(self, other):
        """Return the other state when adding the additive zero."""
        return self + other

    def __sub__(self, other):
        """Return the negative of the other state."""
        if isinstance(other, (FockState, StateSum, NullState)):
            return -other
        return NotImplemented

    def __mul__(self, other):
        """Return the additive zero under scalar or operator multiplication."""
        return self

    def __rmul__(self, other):
        """Return the additive zero under scalar or operator multiplication."""
        return self

    def __bool__(self):
        """Return ``False`` for the additive zero."""
        return False

    def __repr__(self):
        """Return a representation of the additive zero state."""
        return "NullState()"


class StateSum:
    """Finite linear combination of fermionic Fock states.

    Parameters
    ----------
    states : iterable of FockState or NullState
        Terms entering the linear combination.
    """

    __slots__ = ("_states",)

    def __init__(self, states):
        """Construct a state sum without modifying input state objects."""
        normalized = []
        for state in states:
            if isinstance(state, NullState):
                continue
            if not isinstance(state, FockState):
                raise TypeError("StateSum entries must be FockState objects.")
            normalized.append(state)
        self._states = tuple(normalized)

    @property
    def states(self):
        """tuple[FockState, ...]: Terms in the state sum."""
        return self._states

    def simplified(self):
        """Combine duplicate occupation bit strings and remove zero amplitudes.

        Returns
        -------
        FockState, StateSum, or NullState
            Simplified linear combination.
        """
        amplitudes = {}
        metadata = {}
        for state in self._states:
            amplitudes[state.state] = amplitudes.get(state.state, 0) + state.amp
            metadata.setdefault(state.state, (state.n_modes, state.index))

        result = []
        for state_int in sorted(amplitudes):
            amp = amplitudes[state_int]
            if amp == 0:
                continue
            n_modes, index = metadata[state_int]
            result.append(FockState(state_int, amp=amp, n_modes=n_modes, index=index))

        if not result:
            return NullState()
        if len(result) == 1:
            return result[0]
        return StateSum(result)

    def __neg__(self):
        """Multiply every amplitude by minus one."""
        return StateSum((-state for state in self._states))

    def __add__(self, other):
        """Add a Fock state or another state sum."""
        if isinstance(other, NullState):
            return self
        if isinstance(other, FockState):
            return StateSum((*self._states, other)).simplified()
        if isinstance(other, StateSum):
            return StateSum((*self._states, *other.states)).simplified()
        return NotImplemented

    def __radd__(self, other):
        """Add a Fock state or additive zero from the left."""
        return self + other

    def __sub__(self, other):
        """Subtract a Fock state or another state sum."""
        if isinstance(other, (FockState, StateSum, NullState)):
            return self + (-other)
        return NotImplemented

    def __mul__(self, scalar):
        """Multiply every amplitude by a numerical scalar."""
        if not isinstance(scalar, Number):
            return NotImplemented
        return StateSum((state * scalar for state in self._states)).simplified()

    def __rmul__(self, scalar):
        """Multiply every amplitude by a numerical scalar."""
        return self * scalar

    def __repr__(self):
        """Return an unambiguous representation of the state sum."""
        return f"StateSum(states={self._states!r})"


class FockBasis:
    """Occupation-number basis for a fixed fermion number.

    Parameters
    ----------
    n_modes : int
        Number of fermionic modes.
    N : int
        Number of fermions.

    Notes
    -----
    Basis states are ordered by their integer occupation bit strings. The
    fixed-population successor is the standard Gosper combination algorithm
    [Gosper1972]_. Anderson gives a widely used implementation reference for
    the same bit-combination construction [Anderson2005]_.

    References
    ----------
    .. [Gosper1972] R. W. Gosper, in *HAKMEM*, MIT AI Memo 239 (1972),
       Item 175.
    .. [Anderson2005] S. E. Anderson, "Bit Twiddling Hacks: Compute the
       lexicographically next bit permutation," Stanford University
       (1997--2005).
    """

    __slots__ = ("n_modes", "N", "_states")

    def __init__(self, n_modes, N):
        """Construct the complete fixed-``N`` occupation-number basis."""
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
        self._states = self._enumerate_states(n_modes, N)

    @staticmethod
    def _enumerate_states(n_modes, N):
        """Enumerate fixed-population occupation bit strings in ascending order."""
        if N == 0:
            return (0,)
        if N == n_modes:
            return ((1 << n_modes) - 1,)

        count = comb(n_modes, N)
        states = [0] * count
        state = (1 << N) - 1
        limit = 1 << n_modes

        for index in range(count):
            states[index] = state
            if index + 1 == count:
                break
            lowest = state & -state
            ripple = state + lowest
            state = ripple | (((state ^ ripple) >> 2) // lowest)
            if state >= limit:
                raise RuntimeError("Fixed-population basis enumeration overflowed.")

        return tuple(states)

    def __len__(self):
        """Return the Hilbert-space dimension of the basis."""
        return len(self._states)

    def __iter__(self):
        """Iterate over integer occupation bit strings."""
        return iter(self._states)

    def __getitem__(self, index):
        """Return an integer occupation bit string by basis index."""
        return self._states[index]

    def __contains__(self, state):
        """Return whether an integer occupation bit string belongs to the basis."""
        if isinstance(state, FockState):
            state = state.state
        if not isinstance(state, Integral):
            return False
        state = int(state)
        position = bisect_left(self._states, state)
        return position < len(self._states) and self._states[position] == state

    @property
    def states(self):
        """tuple[int, ...]: Integer occupation bit strings in basis order."""
        return self._states

    @property
    def dimension(self):
        """int: Number of basis states."""
        return len(self._states)

    def state(self, index):
        """Return a basis vector as a :class:`FockState`.

        Parameters
        ----------
        index : int
            Zero-based basis index.

        Returns
        -------
        FockState
            Unit-amplitude basis vector.
        """
        if not isinstance(index, Integral):
            raise TypeError("'index' must be an integer.")
        index = int(index)
        state = self._states[index]
        return FockState(state, n_modes=self.n_modes, index=index)


class NParticleSector:
    """Fixed-particle-number sector of fermionic Fock space.

    Parameters
    ----------
    modes : FermionModes
        Fermionic modes defining the occupation-number representation.
    N : int
        Number of fermions in the sector.
    """

    __slots__ = ("modes", "N", "basis")

    def __init__(self, modes, N):
        """Construct a fixed-particle-number sector and its Fock basis."""
        if not isinstance(modes, FermionModes):
            raise TypeError("'modes' must be a FermionModes instance.")
        if not isinstance(N, Integral):
            raise TypeError("'N' must be an integer.")
        N = int(N)
        if N < 0 or N > modes.n_modes:
            raise ValueError("'N' must satisfy 0 <= N <= modes.n_modes.")

        self.modes = modes
        self.N = N
        self.basis = FockBasis(modes.n_modes, N)

    @property
    def dimension(self):
        """int: Hilbert-space dimension of the fixed-particle-number sector."""
        return self.basis.dimension
