"""Occupation-number states and fixed-particle-number fermionic sectors."""

from __future__ import annotations

from bisect import bisect_left
from math import comb
from numbers import Integral, Number

import numpy as np

from ._modes import FermionModes


class FockState:
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

    @property
    def dag(self):
        """FockBra: Hermitian adjoint of the additive zero."""
        return FockBra(self)

    def inner(self, other):
        """Return zero for the inner product with any compatible ket."""
        if isinstance(other, (FockState, StateSum, FockVector, NullState)):
            return 0
        raise TypeError("'other' must be a fermionic ket.")

    def norm(self):
        """Return zero for the Hilbert-space norm."""
        return 0.0

    def normalized(self):
        """Raise because the additive zero cannot be normalized."""
        raise ValueError("The zero state cannot be normalized.")

    def __bool__(self):
        """Return ``False`` for the additive zero."""
        return False

    def __repr__(self):
        """Return a representation of the additive zero state."""
        return "NullState()"


class StateSum:
    """Finite symbolic linear combination of fermionic Fock states.

    Parameters
    ----------
    states : iterable of FockState or NullState
        Terms entering the linear combination.

    Notes
    -----
    ``StateSum`` is used for sparse symbolic state algebra, including the
    result of literal operator action that may change particle number.
    Eigensolver vectors in one fixed sector are represented by
    :class:`FockVector`.
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

    @property
    def dag(self):
        """FockBra: Hermitian adjoint of the linear combination."""
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
        """Return the Hilbert-space norm of the linear combination."""
        return float(np.sqrt(np.real_if_close(self.inner(self))))

    def normalized(self):
        """Return a unit-normalized linear combination.

        Raises
        ------
        ValueError
            If the state sum is zero.
        """
        norm = self.norm()
        if norm == 0:
            raise ValueError("The zero state cannot be normalized.")
        return (1 / norm) * self

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

    def index(self, state):
        """Return the basis index of an occupation-number state.

        Parameters
        ----------
        state : int or FockState
            Occupation bit string or corresponding Fock state.

        Returns
        -------
        int
            Zero-based position in the ordered Fock basis.

        Raises
        ------
        ValueError
            If the state is not contained in the basis.
        TypeError
            If ``state`` is neither an integer nor a ``FockState``.
        """
        if isinstance(state, FockState):
            state = state.state
        if not isinstance(state, Integral):
            raise TypeError("'state' must be an integer or FockState.")
        state = int(state)
        position = bisect_left(self._states, state)
        if position >= len(self._states) or self._states[position] != state:
            raise ValueError("The occupation state is not contained in this basis.")
        return position

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

    def from_vector(self, coefficients):
        """Construct a basis-backed Fock vector from coefficient coordinates.

        Parameters
        ----------
        coefficients : array_like
            One-dimensional coefficient array in ``self.basis`` ordering.

        Returns
        -------
        FockVector
            Ket represented in the fixed-``N`` Fock basis of this sector.
        """
        return FockVector(coefficients, self)


class FockVector:
    r"""Ket represented by coefficients in an :class:`NParticleSector` basis.

    Parameters
    ----------
    coefficients : array_like
        One-dimensional expansion coefficients :math:`c_\alpha` in the ordered
        Fock basis of ``sector``.
    sector : NParticleSector
        Fixed-particle-number sector defining the basis and fermionic modes.

    Notes
    -----
    ``FockVector`` is the basis-backed representation of an eigensolver vector,

    .. math::

        |\psi\rangle = \sum_\alpha c_\alpha |\alpha\rangle.

    The coefficient array is copied on construction and stored read-only.
    """

    __slots__ = ("sector", "_coefficients")

    def __init__(self, coefficients, sector):
        """Validate and store coefficients in the sector's Fock-basis order."""
        if not isinstance(sector, NParticleSector):
            raise TypeError("'sector' must be an NParticleSector instance.")
        array = np.asarray(coefficients)
        if array.ndim != 1:
            raise ValueError("'coefficients' must be one-dimensional.")
        if array.shape[0] != sector.dimension:
            raise ValueError(
                "The coefficient vector length must equal sector.dimension."
            )
        if not np.issubdtype(array.dtype, np.number):
            raise TypeError("'coefficients' must contain numerical values.")
        if np.iscomplexobj(array):
            array = np.asarray(array, dtype=np.complex128)
        else:
            array = np.asarray(array, dtype=np.float64)
        array = array.copy()
        array.setflags(write=False)
        self.sector = sector
        self._coefficients = array

    def __len__(self):
        """Return the number of Fock-basis coefficients."""
        return self._coefficients.shape[0]

    def __getitem__(self, index):
        """Return one coefficient by Fock-basis index."""
        return self._coefficients[index]

    def __pos__(self):
        """Return the ket unchanged."""
        return self

    def __neg__(self):
        """Return the ket multiplied by minus one."""
        return FockVector(-self._coefficients, self.sector)

    def __add__(self, other):
        """Add another Fock vector in the same sector."""
        if isinstance(other, NullState):
            return self
        if not isinstance(other, FockVector):
            return NotImplemented
        _require_compatible_sectors(self.sector, other.sector)
        return FockVector(self._coefficients + other._coefficients, self.sector)

    def __radd__(self, other):
        """Add another Fock vector or additive zero from the left."""
        return self + other

    def __sub__(self, other):
        """Subtract another Fock vector in the same sector."""
        if not isinstance(other, FockVector):
            return NotImplemented
        _require_compatible_sectors(self.sector, other.sector)
        return FockVector(self._coefficients - other._coefficients, self.sector)

    def __mul__(self, scalar):
        """Multiply the ket by a numerical scalar."""
        if not isinstance(scalar, Number):
            return NotImplemented
        return FockVector(self._coefficients * scalar, self.sector)

    def __rmul__(self, scalar):
        """Multiply the ket by a numerical scalar from the left."""
        return self * scalar

    @property
    def coefficients(self):
        """numpy.ndarray: Read-only coefficients in Fock-basis order."""
        return self._coefficients

    @property
    def basis(self):
        """FockBasis: Ordered Fock basis associated with the ket."""
        return self.sector.basis

    @property
    def modes(self):
        """FermionModes: Fermionic modes associated with the ket."""
        return self.sector.modes

    @property
    def dag(self):
        """FockBra: Hermitian adjoint of the ket."""
        return FockBra(self)

    def inner(self, other):
        """Return the inner product with another fermionic ket.

        Parameters
        ----------
        other : FockVector, FockState, StateSum, or NullState
            Ket on the right-hand side.

        Returns
        -------
        numbers.Number
            Inner product ``<self|other>``.

        Notes
        -----
        ``psi.inner(phi)`` is equivalent to ``psi.dag * phi``.
        """
        return _inner_product(self, other)

    def norm(self):
        """Return the Hilbert-space norm of the ket."""
        return float(np.linalg.norm(self._coefficients))

    def normalized(self):
        """Return a unit-normalized Fock vector.

        Raises
        ------
        ValueError
            If the vector has zero norm.
        """
        norm = self.norm()
        if norm == 0:
            raise ValueError("The zero vector cannot be normalized.")
        return FockVector(self._coefficients / norm, self.sector)

    def to_vector(self, *, copy=True):
        """Return the coefficient array in the sector basis.

        Parameters
        ----------
        copy : bool, optional
            Return a writable copy when ``True``. When ``False``, return the
            internal read-only array.

        Returns
        -------
        numpy.ndarray
            Coefficients in ``self.sector.basis`` ordering.
        """
        return self._coefficients.copy() if copy else self._coefficients

    def __repr__(self):
        """Return an unambiguous representation of the Fock vector."""
        return (
            f"FockVector(coefficients={self._coefficients!r}, "
            f"sector={self.sector!r})"
        )


class FockBra:
    r"""Hermitian adjoint of a fermionic ket.

    ``FockBra`` objects are normally obtained from the ``dag`` property of a
    :class:`FockState`, :class:`FockVector`, or symbolic state sum. They support
    literal Dirac-algebra products such as ``psi.dag * O * psi``.
    """

    __slots__ = ("_ket",)

    def __init__(self, ket):
        """Construct a bra as the Hermitian adjoint of a fermionic ket."""
        if not isinstance(ket, (FockState, StateSum, FockVector, NullState)):
            raise TypeError("'ket' must be a fermionic ket.")
        self._ket = ket

    @property
    def dag(self):
        """Return the ket whose Hermitian adjoint defines this bra."""
        return self._ket

    def __mul__(self, other):
        """Form a bra-ket inner product or append an operator to the bra."""
        from ._algebra import Operator

        if isinstance(other, Operator):
            return _BraOperatorProduct(self, other)
        if isinstance(other, (FockState, StateSum, FockVector, NullState)):
            return _inner_product(self._ket, other)
        if isinstance(other, Number):
            return FockBra(_conjugate(other) * self._ket)
        return NotImplemented

    def __rmul__(self, other):
        """Multiply the bra by a numerical scalar from the left."""
        if isinstance(other, Number):
            return FockBra(_conjugate(other) * self._ket)
        return NotImplemented

    def __repr__(self):
        """Return an unambiguous representation of the bra."""
        return f"FockBra(ket={self._ket!r})"


class _BraOperatorProduct:
    """Deferred product of a fermionic bra with one or more operators."""

    __slots__ = ("_bra", "_operator")

    def __init__(self, bra, operator):
        """Store the bra and ordered operator expression."""
        self._bra = bra
        self._operator = operator

    def __mul__(self, other):
        """Append an operator or complete the matrix element with a ket.

        Basis-backed vectors in the same fixed-``N`` sector use the compiled
        sparse operator matrix directly. Other bra-ket products fall back to
        literal operator action in Fock space.
        """
        from ._algebra import Operator

        if isinstance(other, Operator):
            return _BraOperatorProduct(self._bra, self._operator * other)
        if isinstance(other, FockVector) and isinstance(self._bra.dag, FockVector):
            left = self._bra.dag
            if left.sector.modes is not other.sector.modes:
                raise ValueError("Bra and ket use different FermionModes objects.")
            if left.sector.N == other.sector.N:
                from ._hamiltonian import Hamiltonian

                matrix = Hamiltonian(self._operator, other.sector).matrix
                return np.vdot(left.coefficients, matrix @ other.coefficients)
        if isinstance(other, (FockState, StateSum, FockVector, NullState)):
            return self._bra * (self._operator * other)
        if isinstance(other, Number):
            return _BraOperatorProduct(self._bra, self._operator * other)
        return NotImplemented

    def __repr__(self):
        """Return an unambiguous representation of the deferred product."""
        return (
            f"_BraOperatorProduct(bra={self._bra!r}, "
            f"operator={self._operator!r})"
        )


def _conjugate(value):
    """Return the complex conjugate of a numerical value."""
    conjugate = getattr(value, "conjugate", None)
    return conjugate() if conjugate is not None else value


def _require_compatible_sectors(left, right):
    """Validate that two basis-backed vectors use the same Fock-space sector."""
    if left.modes is not right.modes:
        raise ValueError("Fock vectors use different FermionModes objects.")
    if left.N != right.N:
        raise ValueError("Fock vectors belong to different particle-number sectors.")


def _state_sum_amplitudes(state):
    """Return a mapping from occupation bit strings to amplitudes."""
    if isinstance(state, NullState):
        return {}
    if isinstance(state, FockState):
        return {state.state: state.amp}
    if isinstance(state, StateSum):
        amplitudes = {}
        for term in state.states:
            amplitudes[term.state] = amplitudes.get(term.state, 0) + term.amp
        return {key: value for key, value in amplitudes.items() if value != 0}
    raise TypeError("Expected FockState, StateSum, or NullState.")


def _inner_product(left, right):
    """Evaluate the Hilbert-space inner product between two fermionic kets."""
    ket_types = (FockState, StateSum, FockVector, NullState)
    if not isinstance(left, ket_types) or not isinstance(right, ket_types):
        raise TypeError("Inner products require fermionic ket objects.")

    if isinstance(left, NullState) or isinstance(right, NullState):
        return 0

    if isinstance(left, FockVector) and isinstance(right, FockVector):
        if left.sector.modes is not right.sector.modes:
            raise ValueError("Fock vectors use different FermionModes objects.")
        if left.sector.N != right.sector.N:
            return 0
        return np.vdot(left.coefficients, right.coefficients)

    if isinstance(left, FockVector):
        right_map = _state_sum_amplitudes(right)
        total = 0
        for state_int, amplitude in right_map.items():
            if state_int not in left.basis:
                continue
            index = left.basis.index(state_int)
            total += np.conjugate(left.coefficients[index]) * amplitude
        return total

    if isinstance(right, FockVector):
        left_map = _state_sum_amplitudes(left)
        total = 0
        for state_int, amplitude in left_map.items():
            if state_int not in right.basis:
                continue
            index = right.basis.index(state_int)
            total += _conjugate(amplitude) * right.coefficients[index]
        return total

    left_map = _state_sum_amplitudes(left)
    right_map = _state_sum_amplitudes(right)
    total = 0
    for state_int, left_amp in left_map.items():
        right_amp = right_map.get(state_int)
        if right_amp is not None:
            total += _conjugate(left_amp) * right_amp
    return total
