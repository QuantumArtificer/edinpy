"""Statistics-independent sparse-state and Dirac-algebra infrastructure."""

from __future__ import annotations

from numbers import Number

import numpy as np


class BaseFockState:
    """Shared algebra for one elementary occupation-number basis ket.

    Statistics-specific subclasses own the actual occupation representation
    and implement :meth:`_with_amp` to rebuild the same ket with a different
    amplitude.
    """

    __slots__ = ()

    def _with_amp(self, amp):
        """Return the same occupation state with amplitude ``amp``."""
        raise NotImplementedError

    def __pos__(self):
        return self

    def __neg__(self):
        return self._with_amp(-self.amp)

    def __add__(self, other):
        if isinstance(other, NullState):
            return self
        if isinstance(other, BaseFockState):
            return StateSum((self, other)).simplified()
        if isinstance(other, StateSum):
            return StateSum((self, *other.states)).simplified()
        return NotImplemented

    def __radd__(self, other):
        if other == 0:
            return self
        return self + other

    def __sub__(self, other):
        if isinstance(other, (BaseFockState, StateSum, NullState)):
            return self + (-other)
        return NotImplemented

    def __mul__(self, scalar):
        if not isinstance(scalar, Number):
            return NotImplemented
        return self._with_amp(self.amp * scalar)

    def __rmul__(self, scalar):
        return self * scalar

    @property
    def dag(self):
        return FockBra(self)

    def inner(self, other):
        return _inner_product(self, other)

    def norm(self):
        return float(abs(self.amp))

    def normalized(self):
        norm = self.norm()
        if norm == 0:
            raise ValueError("The zero state cannot be normalized.")
        return (1 / norm) * self


class NullState:
    """Additive zero for Fock-space state algebra.

    ``NullState`` is returned when an operator annihilates a basis state or
    when a symbolic state sum simplifies to zero. It participates in state
    addition, scalar multiplication, Hermitian conjugation, and inner
    products without requiring a special numerical representation.
    """

    __slots__ = ()
    amp = 0

    def __neg__(self):
        return self

    def __add__(self, other):
        if isinstance(other, (BaseFockState, StateSum, FockVector, NullState)):
            return other
        return NotImplemented

    def __radd__(self, other):
        if other == 0:
            return self
        return self + other

    def __sub__(self, other):
        if isinstance(other, (BaseFockState, StateSum, FockVector, NullState)):
            return -other
        return NotImplemented

    def __mul__(self, other):
        return self

    def __rmul__(self, other):
        return self

    @property
    def dag(self):
        """Return the Hermitian-adjoint bra of the additive zero."""
        return FockBra(self)

    def inner(self, other):
        """Return zero for the inner product with any compatible ket."""
        if isinstance(other, (BaseFockState, StateSum, FockVector, NullState)):
            return 0
        raise TypeError("'other' must be a Fock-space ket.")

    def norm(self):
        """Return zero for the Hilbert-space norm."""
        return 0.0

    def normalized(self):
        """Raise because the additive zero cannot be normalized."""
        raise ValueError("The zero state cannot be normalized.")

    def __bool__(self):
        return False

    def __repr__(self):
        return "NullState()"


class StateSum:
    """Finite symbolic linear combination of occupation-number basis states.

    Parameters
    ----------
    states : iterable of FockState
        Elementary Fock states belonging to the same statistics-specific
        representation. Amplitudes are stored on the individual states.

    Notes
    -----
    :meth:`simplified` combines repeated occupation configurations and removes
    terms with zero total amplitude.
    """

    __slots__ = ("_states", "_state_type")

    def __init__(self, states):
        normalized = []
        state_type = None
        for state in states:
            if isinstance(state, NullState):
                continue
            if not isinstance(state, BaseFockState):
                raise TypeError("StateSum entries must be FockState objects.")
            if state_type is None:
                state_type = type(state)
            elif type(state) is not state_type:
                raise TypeError(
                    "StateSum entries must use the same FockState representation."
                )
            normalized.append(state)
        self._states = tuple(normalized)
        self._state_type = state_type

    @property
    def states(self):
        """tuple: Return the elementary Fock-state terms in the sum."""
        return self._states

    def simplified(self):
        """Combine duplicate occupation states and remove zero amplitudes."""
        amplitudes = {}
        exemplars = {}
        for state in self._states:
            key = state.state
            amplitudes[key] = amplitudes.get(key, 0) + state.amp
            exemplars.setdefault(key, state)

        result = []
        for key in sorted(amplitudes):
            amp = amplitudes[key]
            if amp == 0:
                continue
            result.append(exemplars[key]._with_amp(amp))

        if not result:
            return NullState()
        if len(result) == 1:
            return result[0]
        return StateSum(result)

    def __neg__(self):
        return StateSum((-state for state in self._states))

    def __add__(self, other):
        if isinstance(other, NullState):
            return self
        if isinstance(other, BaseFockState):
            return StateSum((*self._states, other)).simplified()
        if isinstance(other, StateSum):
            return StateSum((*self._states, *other.states)).simplified()
        return NotImplemented

    def __radd__(self, other):
        if other == 0:
            return self
        return self + other

    def __sub__(self, other):
        if isinstance(other, (BaseFockState, StateSum, NullState)):
            return self + (-other)
        return NotImplemented

    def __mul__(self, scalar):
        if not isinstance(scalar, Number):
            return NotImplemented
        return StateSum((state * scalar for state in self._states)).simplified()

    def __rmul__(self, scalar):
        return self * scalar

    @property
    def dag(self):
        """Return the Hermitian-adjoint bra of the state sum."""
        return FockBra(self)

    def inner(self, other):
        """Return the Hilbert-space inner product with another ket.

        Parameters
        ----------
        other : FockState, StateSum, FockVector, or NullState
            Ket on the right-hand side of the inner product. Basis-aware
            vectors must belong to a compatible sector.

        Returns
        -------
        numbers.Number
            Inner product ``<self|other>``.
        """
        return _inner_product(self, other)

    def norm(self):
        """Return the Hilbert-space norm of the state sum."""
        return float(np.sqrt(np.real_if_close(self.inner(self))))

    def normalized(self):
        """Return a unit-normalized state sum."""
        norm = self.norm()
        if norm == 0:
            raise ValueError("The zero state cannot be normalized.")
        return (1 / norm) * self

    def __repr__(self):
        return f"StateSum(states={self._states!r})"


class FockVector:
    """Ket represented by coordinates in a built Fock-space sector.

    Parameters
    ----------
    coefficients : array_like
        One-dimensional real or complex coefficients in the ordering of
        ``sector.basis``. The length must equal ``sector.dimension``.
    sector : NParticleSector
        Built fermionic or bosonic sector that defines the basis and mode
        ordering.

    Notes
    -----
    Coefficients are stored as a read-only NumPy array. Operator action,
    Hermitian conjugation, inner products, normalization, and expectation
    values preserve the associated sector.
    """

    __slots__ = ("sector", "_coefficients")

    def __init__(self, coefficients, sector):
        if not hasattr(sector, "is_built"):
            raise TypeError("'sector' must be a fixed-sector object.")
        if not sector.is_built:
            raise RuntimeError(
                "The sector has not been built. Call sector.build() before "
                "constructing a FockVector."
            )
        for attribute in ("dimension", "basis", "modes", "N"):
            if not hasattr(sector, attribute):
                raise TypeError(
                    "'sector' must provide dimension, basis, modes, and N."
                )

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
        return self._coefficients.shape[0]

    def __getitem__(self, index):
        return self._coefficients[index]

    def __pos__(self):
        return self

    def __neg__(self):
        return FockVector(-self._coefficients, self.sector)

    def __add__(self, other):
        if isinstance(other, NullState):
            return self
        if not isinstance(other, FockVector):
            return NotImplemented
        _require_compatible_sectors(self.sector, other.sector)
        return FockVector(self._coefficients + other._coefficients, self.sector)

    def __radd__(self, other):
        if other == 0:
            return self
        return self + other

    def __sub__(self, other):
        if not isinstance(other, FockVector):
            return NotImplemented
        _require_compatible_sectors(self.sector, other.sector)
        return FockVector(self._coefficients - other._coefficients, self.sector)

    def __mul__(self, scalar):
        if not isinstance(scalar, Number):
            return NotImplemented
        return FockVector(self._coefficients * scalar, self.sector)

    def __rmul__(self, scalar):
        return self * scalar

    @property
    def coefficients(self):
        """numpy.ndarray: Return the read-only coefficients in basis order."""
        return self._coefficients

    @property
    def basis(self):
        """Return the ordered Fock basis associated with the ket."""
        return self.sector.basis

    @property
    def modes(self):
        """Return the mode specification associated with the ket."""
        return self.sector.modes

    @property
    def dag(self):
        """Return the Hermitian-adjoint bra of the ket."""
        return FockBra(self)

    def inner(self, other):
        """Return the Hilbert-space inner product with another ket.

        Parameters
        ----------
        other : FockState, StateSum, FockVector, or NullState
            Ket on the right-hand side. A second FockVector must use a
            compatible sector.

        Returns
        -------
        numbers.Number
            Inner product ``<self|other>``.
        """
        return _inner_product(self, other)

    def norm(self):
        """Return the Hilbert-space norm of the ket."""
        return float(np.linalg.norm(self._coefficients))

    def normalized(self):
        """Return a unit-normalized copy of the ket.

        Raises
        ------
        ValueError
            If the coefficient vector has zero norm.
        """
        norm = self.norm()
        if norm == 0:
            raise ValueError("The zero vector cannot be normalized.")
        return FockVector(self._coefficients / norm, self.sector)

    def to_vector(self, *, copy=True):
        """Return coefficients in the sector-basis ordering.

        Parameters
        ----------
        copy : bool, optional
            Return a writable copy when ``True``. When ``False``, return the
            read-only array stored by the Fock vector.

        Returns
        -------
        numpy.ndarray
            One-dimensional real or complex coefficient array.
        """
        return self._coefficients.copy() if copy else self._coefficients

    def __repr__(self):
        return (
            f"FockVector(coefficients={self._coefficients!r}, "
            f"sector={self.sector!r})"
        )


class FockBra:
    """Hermitian adjoint of a Fock-space ket.

    Parameters
    ----------
    ket : FockState, StateSum, FockVector, or NullState
        Ket whose Hermitian adjoint is represented by the bra.

    Notes
    -----
    Multiplication by a compatible ket evaluates an inner product.
    Multiplication by an operator returns a bra-operator product that can be
    completed with a ket, as in ``psi.dag * O * psi``.
    """

    __slots__ = ("_ket",)

    def __init__(self, ket):
        if not isinstance(ket, (BaseFockState, StateSum, FockVector, NullState)):
            raise TypeError("'ket' must be a Fock-space ket.")
        self._ket = ket

    @property
    def dag(self):
        """Return the ket whose Hermitian adjoint defines this bra."""
        return self._ket

    def __mul__(self, other):
        from ._algebra import Operator

        if isinstance(other, Operator):
            return _BraOperatorProduct(self, other)
        if isinstance(other, (BaseFockState, StateSum, FockVector, NullState)):
            return _inner_product(self._ket, other)
        if isinstance(other, Number):
            return FockBra(_conjugate(other) * self._ket)
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, Number):
            return FockBra(_conjugate(other) * self._ket)
        return NotImplemented

    def __repr__(self):
        return f"FockBra(ket={self._ket!r})"


class _BraOperatorProduct:
    """Deferred product of a bra with one or more symbolic operators."""

    __slots__ = ("_bra", "_operator")

    def __init__(self, bra, operator):
        self._bra = bra
        self._operator = operator

    def __mul__(self, other):
        from ._algebra import Operator

        if isinstance(other, Operator):
            return _BraOperatorProduct(self._bra, self._operator * other)

        if isinstance(other, FockVector) and isinstance(self._bra.dag, FockVector):
            left = self._bra.dag
            if left.sector.modes is not other.sector.modes:
                modes_name = type(other.sector.modes).__name__
                raise ValueError(f"Bra and ket use different {modes_name} objects.")
            if left.sector.N == other.sector.N:
                from ._hamiltonian import Hamiltonian

                matrix = Hamiltonian(self._operator, other.sector).matrix
                return np.vdot(left.coefficients, matrix @ other.coefficients)

        if isinstance(other, (BaseFockState, StateSum, FockVector, NullState)):
            return self._bra * (self._operator * other)
        if isinstance(other, Number):
            return _BraOperatorProduct(self._bra, self._operator * other)
        return NotImplemented

    def __repr__(self):
        return (
            f"_BraOperatorProduct(bra={self._bra!r}, "
            f"operator={self._operator!r})"
        )


def _conjugate(value):
    conjugate = getattr(value, "conjugate", None)
    return conjugate() if conjugate is not None else value


def _require_compatible_sectors(left, right):
    if left.modes is not right.modes:
        modes_name = type(right.modes).__name__
        raise ValueError(f"Fock vectors use different {modes_name} objects.")
    if left.N != right.N:
        raise ValueError("Fock vectors belong to different particle-number sectors.")


def _state_sum_amplitudes(state):
    if isinstance(state, NullState):
        return {}
    if isinstance(state, BaseFockState):
        return {state.state: state.amp}
    if isinstance(state, StateSum):
        amplitudes = {}
        for term in state.states:
            amplitudes[term.state] = amplitudes.get(term.state, 0) + term.amp
        return {key: value for key, value in amplitudes.items() if value != 0}
    raise TypeError("Expected FockState, StateSum, or NullState.")


def _inner_product(left, right):
    ket_types = (BaseFockState, StateSum, FockVector, NullState)
    if not isinstance(left, ket_types) or not isinstance(right, ket_types):
        raise TypeError("Inner products require Fock-space ket objects.")

    if isinstance(left, NullState) or isinstance(right, NullState):
        return 0

    if isinstance(left, FockVector) and isinstance(right, FockVector):
        if left.sector.modes is not right.sector.modes:
            modes_name = type(right.sector.modes).__name__
            raise ValueError(f"Fock vectors use different {modes_name} objects.")
        if left.sector.N != right.sector.N:
            return 0
        return np.vdot(left.coefficients, right.coefficients)

    if isinstance(left, FockVector):
        right_map = _state_sum_amplitudes(right)
        total = 0
        for state_key, amplitude in right_map.items():
            if state_key not in left.basis:
                continue
            index = left.basis.index(state_key)
            total += np.conjugate(left.coefficients[index]) * amplitude
        return total

    if isinstance(right, FockVector):
        left_map = _state_sum_amplitudes(left)
        total = 0
        for state_key, amplitude in left_map.items():
            if state_key not in right.basis:
                continue
            index = right.basis.index(state_key)
            total += _conjugate(amplitude) * right.coefficients[index]
        return total

    left_map = _state_sum_amplitudes(left)
    right_map = _state_sum_amplitudes(right)
    total = 0
    for state_key, left_amp in left_map.items():
        right_amp = right_map.get(state_key)
        if right_amp is not None:
            total += _conjugate(left_amp) * right_amp
    return total
