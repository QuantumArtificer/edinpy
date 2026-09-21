"""Literal symbolic algebra for fermionic creation, annihilation, and number operators."""

from __future__ import annotations

from numbers import Number as Numeric

from ._basis import FockState, FockVector, NullState, StateSum
from ._modes import FermionModes


class Operator:
    """Base class for symbolic fermionic operators and operator expressions."""

    def __pos__(self):
        """Return the operator expression unchanged."""
        return self

    def __neg__(self):
        """Return the operator expression multiplied by minus one."""
        return _multiply_expressions(Scalar(-1), self)

    def __add__(self, other):
        """Return the symbolic sum with another operator or numerical scalar."""
        if isinstance(other, Numeric):
            if other == 0:
                return self
            other = Scalar(other)
        if isinstance(other, Operator):
            return OperatorSum((self, other))
        return NotImplemented

    def __radd__(self, other):
        """Return the symbolic sum with a numerical scalar from the left."""
        if isinstance(other, Numeric) and other == 0:
            return self
        return self + other

    def __sub__(self, other):
        """Return the symbolic difference from another operator or scalar."""
        if isinstance(other, Numeric):
            other = Scalar(other)
        if isinstance(other, Operator):
            return OperatorSum((self, -other))
        return NotImplemented

    def __rsub__(self, other):
        """Return a numerical scalar minus the operator expression."""
        if isinstance(other, Numeric):
            return Scalar(other) + (-self)
        return NotImplemented

    def __mul__(self, other):
        """Multiply by an operator, scalar, or fermionic ket.

        Multiplication by a ket applies the operator expression. For a
        :class:`~edinpy.fermion.FockVector`, the action is evaluated in literal
        Fock space from its fixed-sector basis expansion, so number-changing
        operators are not silently projected back into the original sector.
        """
        if isinstance(other, FockState):
            return self._action(other)
        if isinstance(other, NullState):
            return other
        if isinstance(other, StateSum):
            outputs = []
            for state in other.states:
                _extend_state_terms(outputs, self._action(state))
            return _state_terms_result(outputs)
        if isinstance(other, FockVector):
            modes = operator_modes(self)
            if modes is not None and modes is not other.sector.modes:
                raise ValueError(
                    "Operator and FockVector use different FermionModes objects."
                )
            outputs = []
            basis = other.sector.basis
            for index, amplitude in enumerate(other.coefficients):
                if amplitude == 0:
                    continue
                state = FockState(
                    basis.states[index],
                    amp=amplitude,
                    n_modes=basis.n_modes,
                    index=index,
                )
                _extend_state_terms(outputs, self._action(state))
            return _state_terms_result(outputs)
        if isinstance(other, Numeric):
            other = Scalar(other)
        if isinstance(other, Operator):
            return _multiply_expressions(self, other)
        return NotImplemented

    def __rmul__(self, other):
        """Multiply by a numerical scalar from the left."""
        if isinstance(other, Numeric):
            return _multiply_expressions(Scalar(other), self)
        return NotImplemented

    def __str__(self):
        """Return the compact symbolic representation of the expression."""
        return self.string

    def __format__(self, spec):
        """Format the operator using its compact symbolic representation."""
        return format(self.string, spec)

    def _action(self, state):
        """Apply the operator expression to one Fock state."""
        raise NotImplementedError

    @property
    def dag(self):
        """Operator: Hermitian adjoint of the operator expression."""
        raise NotImplementedError

    @property
    def string(self):
        """str: Compact symbolic representation of the operator expression."""
        raise NotImplementedError


class Scalar(Operator):
    """Numerical scalar embedded in a symbolic operator expression.

    Parameters
    ----------
    value : numbers.Number
        Numerical value.
    """

    __slots__ = ("value",)

    def __init__(self, value):
        """Construct a numerical scalar expression."""
        if not isinstance(value, Numeric):
            raise TypeError("'value' must be numeric.")
        self.value = value

    def _action(self, state):
        """Multiply a Fock-state amplitude by the scalar value."""
        return FockState(
            state.state,
            amp=self.value * state.amp,
            n_modes=state.n_modes,
            index=state.index,
        )

    @property
    def dag(self):
        """Scalar: Complex-conjugated scalar expression."""
        conjugate = getattr(self.value, "conjugate", None)
        return Scalar(conjugate() if conjugate is not None else self.value)

    @property
    def string(self):
        """str: String representation of the numerical value."""
        return str(self.value)


class _SingleModeOperator(Operator):
    """Base class for primitive operators acting on one fermionic mode."""

    symbol = "?"

    __slots__ = ("_indices", "_modes", "_mode")

    def __init__(self, indices, modes):
        """Resolve a degree-of-freedom label or explicit mode index."""
        if not isinstance(modes, FermionModes):
            raise TypeError("'modes' must be a FermionModes instance.")
        self._modes = modes

        if isinstance(indices, int):
            if indices < 0 or indices >= modes.n_modes:
                raise IndexError(
                    f"Mode index {indices} is outside [0, {modes.n_modes})."
                )
            self._indices = int(indices)
            self._mode = int(indices)
        else:
            self._indices = tuple(indices)
            self._mode = modes.resolve(self._indices)

    @property
    def modes(self):
        """FermionModes: Fermionic modes on which the operator is defined."""
        return self._modes

    @property
    def mode(self):
        """int: Zero-based resolved fermionic-mode index."""
        return self._mode

    @property
    def site(self):
        """int: Alias of :attr:`mode` used by the compiler backend."""
        return self._mode

    @property
    def indices(self):
        """int or tuple[int, ...]: Original index specification."""
        return self._indices

    def _label_indices(self):
        """Return the user-facing index tuple used in symbolic strings."""
        if isinstance(self._indices, int):
            return (self._indices,)
        return self._indices

    @property
    def string(self):
        """str: Compact symbolic representation including operator indices."""
        indices = ",".join(str(index) for index in self._label_indices())
        return f"{self.symbol}[{indices}]"


class Annihilation(_SingleModeOperator):
    r"""Fermionic annihilation operator :math:`c_i`.

    Parameters
    ----------
    indices : int or iterable of int
        Explicit fermionic-mode index or degree-of-freedom indices.
    modes : FermionModes
        Fermionic modes used to resolve degree-of-freedom indices.
    """

    symbol = "c"

    def _action(self, state):
        """Apply fermionic annihilation with the canonical occupation ordering."""
        bit = 1 << self.mode
        if not (state.state & bit):
            return NullState()
        parity = (state.state & (bit - 1)).bit_count() & 1
        sign = -1 if parity else 1
        return FockState(
            state.state ^ bit,
            amp=sign * state.amp,
            n_modes=state.n_modes,
        )

    @property
    def dag(self):
        """Creation: Hermitian adjoint of the annihilation operator."""
        return Creation(self.indices, modes=self.modes)


class Creation(_SingleModeOperator):
    r"""Fermionic creation operator :math:`c_i^\dagger`.

    Parameters
    ----------
    indices : int or iterable of int
        Explicit fermionic-mode index or degree-of-freedom indices.
    modes : FermionModes
        Fermionic modes used to resolve degree-of-freedom indices.
    """

    symbol = "c†"

    def _action(self, state):
        """Apply fermionic creation with the canonical occupation ordering."""
        bit = 1 << self.mode
        if state.state & bit:
            return NullState()
        parity = (state.state & (bit - 1)).bit_count() & 1
        sign = -1 if parity else 1
        return FockState(
            state.state ^ bit,
            amp=sign * state.amp,
            n_modes=state.n_modes,
        )

    @property
    def dag(self):
        """Annihilation: Hermitian adjoint of the creation operator."""
        return Annihilation(self.indices, modes=self.modes)


class Number(_SingleModeOperator):
    r"""Fermionic number operator :math:`n_i=c_i^\dagger c_i`.

    Parameters
    ----------
    indices : int or iterable of int
        Explicit fermionic-mode index or degree-of-freedom indices.
    modes : FermionModes
        Fermionic modes used to resolve degree-of-freedom indices.
    """

    symbol = "n"

    def _action(self, state):
        """Return the input state when the mode is occupied and zero otherwise."""
        if state.state & (1 << self.mode):
            return FockState(
                state.state,
                amp=state.amp,
                n_modes=state.n_modes,
                index=state.index,
            )
        return NullState()

    @property
    def dag(self):
        """Number: Hermitian adjoint of the Hermitian number operator."""
        return self


class OperatorSum(Operator):
    """Finite symbolic sum of operator expressions.

    Parameters
    ----------
    terms : iterable of Operator
        Operator expressions entering the sum.
    """

    __slots__ = ("_terms",)

    def __init__(self, terms):
        """Flatten nested sums while preserving additive term order."""
        flattened = []
        for term in terms:
            if isinstance(term, Numeric):
                term = Scalar(term)
            if isinstance(term, OperatorSum):
                flattened.extend(term.os)
            elif isinstance(term, Scalar) and term.value == 0:
                continue
            elif isinstance(term, Operator):
                flattened.append(term)
            else:
                raise TypeError("OperatorSum terms must be operator expressions.")
        if not flattened:
            flattened = [Scalar(0)]
        self._terms = tuple(flattened)

    @property
    def os(self):
        """tuple[Operator, ...]: Additive terms in the expression."""
        return self._terms

    def _action(self, state):
        """Apply every additive term to a Fock state and combine the outputs."""
        result = NullState()
        for term in self._terms:
            result = result + term._action(state)
        return result

    @property
    def dag(self):
        """OperatorSum: Hermitian adjoint of every additive term."""
        return OperatorSum(term.dag for term in self._terms)

    @property
    def string(self):
        """str: Sum of the compact symbolic representations of all terms."""
        return " + ".join(term.string for term in self._terms)


class OperatorProduct(Operator):
    """Ordered symbolic product of operator expressions.

    Parameters
    ----------
    factors : iterable of Operator
        Ordered factors. The rightmost factor acts first on a Fock state.
    """

    __slots__ = ("_factors",)

    def __init__(self, factors):
        """Flatten nested products while preserving operator order."""
        flattened = []
        coefficient = 1
        for factor in factors:
            if isinstance(factor, Numeric):
                factor = Scalar(factor)
            if isinstance(factor, OperatorProduct):
                nested = factor.op
            elif isinstance(factor, Operator):
                nested = (factor,)
            else:
                raise TypeError("OperatorProduct factors must be operator expressions.")

            for item in nested:
                if isinstance(item, Scalar):
                    coefficient *= item.value
                else:
                    flattened.append(item)

        if coefficient != 1 or not flattened:
            flattened.insert(0, Scalar(coefficient))
        self._factors = tuple(flattened)

    @property
    def op(self):
        """tuple[Operator, ...]: Ordered factors in the product."""
        return self._factors

    def _action(self, state):
        """Apply product factors from right to left to a Fock state."""
        result = state
        for factor in reversed(self._factors):
            if isinstance(result, NullState):
                return result
            if isinstance(result, FockState):
                result = factor._action(result)
                continue
            if isinstance(result, StateSum):
                next_result = NullState()
                for term_state in result.states:
                    next_result = next_result + factor._action(term_state)
                result = next_result
                continue
            raise TypeError(
                f"Unsupported intermediate state type {type(result).__name__}."
            )
        return result

    @property
    def dag(self):
        """OperatorProduct: Reversed product of Hermitian-adjoint factors."""
        return OperatorProduct(factor.dag for factor in reversed(self._factors))

    @property
    def string(self):
        """str: Concatenated symbolic representation of the product factors."""
        return " ".join(factor.string for factor in self._factors)


def _extend_state_terms(target, result):
    """Append a symbolic operator-action result to a flat state-term list."""
    if isinstance(result, NullState):
        return
    if isinstance(result, FockState):
        target.append(result)
        return
    if isinstance(result, StateSum):
        target.extend(result.states)
        return
    raise TypeError(f"Unsupported operator-action result {type(result).__name__}.")


def _state_terms_result(states):
    """Return the canonical symbolic state result for a list of Fock terms."""
    if not states:
        return NullState()
    return StateSum(states).simplified()


def _multiply_expressions(left, right):
    """Multiply expressions, distributing products over symbolic sums."""
    if isinstance(left, OperatorSum):
        return OperatorSum(_multiply_expressions(term, right) for term in left.os)
    if isinstance(right, OperatorSum):
        return OperatorSum(_multiply_expressions(left, term) for term in right.os)

    left_factors = left.op if isinstance(left, OperatorProduct) else (left,)
    right_factors = right.op if isinstance(right, OperatorProduct) else (right,)
    return OperatorProduct((*left_factors, *right_factors))


def operator_modes(operator):
    """Return the unique :class:`FermionModes` object used by an expression.

    Parameters
    ----------
    operator : Operator
        Symbolic operator expression.

    Returns
    -------
    FermionModes or None
        Unique mode specification, or ``None`` when the expression contains no
        mode-resolved primitive operators.

    Raises
    ------
    ValueError
        If primitives from different ``FermionModes`` objects are mixed.
    """
    modes = set()

    def visit(expression):
        """Collect mode specifications recursively from one expression node."""
        if isinstance(expression, _SingleModeOperator):
            modes.add(expression.modes)
        elif isinstance(expression, OperatorSum):
            for term in expression.os:
                visit(term)
        elif isinstance(expression, OperatorProduct):
            for factor in expression.op:
                visit(factor)

    visit(operator)
    if len(modes) > 1:
        raise ValueError("Operator expression mixes different FermionModes objects.")
    return next(iter(modes), None)


class _OperatorNotation:
    """Callable notation bound to one primitive operator class and mode set."""

    __slots__ = ("_operator_type", "_modes")

    def __init__(self, operator_type, modes):
        """Store the primitive operator class and associated fermionic modes."""
        if operator_type not in (Annihilation, Creation, Number):
            raise TypeError(
                "'operator_type' must be Annihilation, Creation, or Number."
            )
        if not isinstance(modes, FermionModes):
            raise TypeError("'modes' must be a FermionModes instance.")
        self._operator_type = operator_type
        self._modes = modes

    def __call__(self, *indices):
        """Construct the bound primitive operator from degree-of-freedom indices.

        Parameters
        ----------
        *indices : int
            Degree-of-freedom indices. A single tuple or list is also accepted.

        Returns
        -------
        Operator
            Primitive operator associated with the bound ``FermionModes``.
        """
        if len(indices) == 1 and isinstance(indices[0], (tuple, list)):
            indices = tuple(indices[0])
        return self._operator_type(indices, modes=self._modes)

    @property
    def modes(self):
        """FermionModes: Fermionic modes associated with the notation."""
        return self._modes

    @property
    def operator_type(self):
        """type: Primitive operator class constructed by the notation."""
        return self._operator_type


def set_notation(operator_type, modes):
    """Return a callable notation for a primitive fermionic operator.

    Parameters
    ----------
    operator_type : {Annihilation, Creation, Number}
        Primitive operator class to construct.
    modes : FermionModes
        Fermionic modes to which the notation is bound.

    Returns
    -------
    callable
        Callable accepting degree-of-freedom indices. The Python variable to
        which the callable is assigned determines the user's notation. EDinPy
        does not impose symbols such as ``c``, ``d``, or ``f``.

    Examples
    --------
    >>> from edinpy import fermion as edf
    >>> site = edf.DoF(4, name="site")
    >>> spin = edf.DoF(2, name="spin")
    >>> modes = edf.FermionModes(site, spin)
    >>> c = edf.set_notation(edf.Annihilation, modes)
    >>> c(2, 1).mode == modes.resolve((2, 1))
    True
    """
    return _OperatorNotation(operator_type, modes)
