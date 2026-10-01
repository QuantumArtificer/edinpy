"""Statistics-independent symbolic operator-expression infrastructure."""

from __future__ import annotations

from numbers import Integral, Number as Numeric

from ._states import BaseFockState, FockVector, NullState, StateSum


class Operator:
    """Base class for symbolic second-quantized operator expressions.

    Operator expressions support scalar multiplication, addition, subtraction,
    ordered products, Hermitian conjugation through :attr:`dag`, and direct
    action on compatible Fock states or basis-aware Fock vectors.
    """

    def __pos__(self):
        """Return the operator expression unchanged."""
        return self

    def __neg__(self):
        """Return the expression multiplied by minus one."""
        return _multiply_expressions(Scalar(-1), self)

    def __add__(self, other):
        """Return the symbolic sum with another operator or scalar."""
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
        """Multiply by another operator, scalar, or shared Fock-space ket."""
        if isinstance(other, Numeric):
            other = Scalar(other)
        if isinstance(other, Operator):
            return _multiply_expressions(self, other)
        if isinstance(other, BaseFockState):
            return self._action(other)
        if isinstance(other, NullState):
            return other
        if isinstance(other, StateSum):
            outputs = []
            for state in other.states:
                _extend_state_terms(outputs, self._action(state))
            return _state_terms_result(outputs)
        if isinstance(other, FockVector):
            domains = operator_domains(self)
            if domains and domains != frozenset((other.sector.modes,)):
                modes_name = type(other.sector.modes).__name__
                raise ValueError(
                    f"Operator and FockVector use different {modes_name} objects."
                )
            outputs = []
            basis = other.sector.basis
            for index, amplitude in enumerate(other.coefficients):
                if amplitude == 0:
                    continue
                state = basis.state(index) * amplitude
                _extend_state_terms(outputs, self._action(state))
            return _state_terms_result(outputs)
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
        """Apply the expression to one elementary Fock state."""
        raise NotImplementedError

    @property
    def dag(self):
        """Operator: Hermitian adjoint of the expression."""
        raise NotImplementedError

    @property
    def string(self):
        """str: Compact symbolic representation."""
        raise NotImplementedError


class Scalar(Operator):
    """Numerical scalar embedded in a symbolic operator expression."""

    __slots__ = ("value",)

    def __init__(self, value):
        if not isinstance(value, Numeric):
            raise TypeError("'value' must be numeric.")
        self.value = value

    def _action(self, state):
        return state * self.value

    @property
    def dag(self):
        conjugate = getattr(self.value, "conjugate", None)
        return Scalar(conjugate() if conjugate is not None else self.value)

    @property
    def string(self):
        return str(self.value)


class SingleModeOperator(Operator):
    """Shared mode-resolution logic for one-mode primitive operators."""

    symbol = "?"
    modes_type = None

    __slots__ = ("_indices", "_modes", "_mode")

    def __init__(self, indices, modes):
        modes_type = type(self).modes_type
        if modes_type is None:
            raise TypeError("Primitive operator class does not define modes_type.")
        if not isinstance(modes, modes_type):
            raise TypeError(
                f"'modes' must be a {modes_type.__name__} instance."
            )
        self._modes = modes

        if isinstance(indices, Integral):
            mode = int(indices)
            if mode < 0 or mode >= modes.n_modes:
                raise IndexError(
                    f"Mode index {mode} is outside [0, {modes.n_modes})."
                )
            self._indices = mode
            self._mode = mode
        else:
            self._indices = tuple(indices)
            self._mode = modes.resolve(self._indices)

    @property
    def modes(self):
        """Mode specification on which the primitive is defined."""
        return self._modes

    @property
    def mode(self):
        """int: Zero-based resolved mode index."""
        return self._mode

    @property
    def site(self):
        """int: Alias used by compiler backends."""
        return self._mode

    @property
    def indices(self):
        """Original flat mode index or degree-of-freedom coordinates."""
        return self._indices

    def _label_indices(self):
        if isinstance(self._indices, int):
            return (self._indices,)
        return self._indices

    @property
    def string(self):
        indices = ",".join(str(index) for index in self._label_indices())
        return f"{self.symbol}[{indices}]"


class OperatorSum(Operator):
    """Finite symbolic sum of operator expressions.

    Parameters
    ----------
    terms : iterable of Operator or numbers.Number
        Additive terms. Nested :class:`OperatorSum` objects are flattened and
        numerical scalars are promoted to symbolic scalar operators.
    """

    __slots__ = ("_terms",)

    def __init__(self, terms):
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
        result = None
        for term in self._terms:
            output = term._action(state)
            result = output if result is None else result + output
        return result

    @property
    def dag(self):
        return OperatorSum(term.dag for term in self._terms)

    @property
    def string(self):
        return " + ".join(term.string for term in self._terms)


class OperatorProduct(Operator):
    """Ordered symbolic product of operator expressions.

    Parameters
    ----------
    factors : iterable of Operator or numbers.Number
        Factors written in operator order. The rightmost factor acts first on
        a ket. Nested products are flattened and numerical factors are
        collected into one scalar coefficient.
    """

    __slots__ = ("_factors",)

    def __init__(self, factors):
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
                raise TypeError(
                    "OperatorProduct factors must be operator expressions."
                )

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
        """tuple[Operator, ...]: Ordered factors in the expression."""
        return self._factors

    def _action(self, state):
        result = state
        for factor in reversed(self._factors):
            result = factor * result
        return result

    @property
    def dag(self):
        return OperatorProduct(factor.dag for factor in reversed(self._factors))

    @property
    def string(self):
        return " ".join(factor.string for factor in self._factors)



def _extend_state_terms(target, result):
    """Append a literal operator-action result to a flat state-term list."""
    if isinstance(result, NullState):
        return
    if isinstance(result, BaseFockState):
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


def operator_domains(operator):
    """Return mode specifications referenced by an operator expression.

    The shared expression tree deliberately permits more than one mode domain.
    Statistics-specific frontends may impose a single-domain requirement.
    """
    domains = set()

    def visit(expression):
        if isinstance(expression, SingleModeOperator):
            domains.add(expression.modes)
        elif isinstance(expression, OperatorSum):
            for term in expression.os:
                visit(term)
        elif isinstance(expression, OperatorProduct):
            for factor in expression.op:
                visit(factor)

    visit(operator)
    return frozenset(domains)


def single_operator_modes(operator, modes_type):
    """Return the unique mode specification for one statistics-specific tree."""
    domains = operator_domains(operator)
    if any(not isinstance(domain, modes_type) for domain in domains):
        raise ValueError(
            f"Operator expression contains modes outside {modes_type.__name__}."
        )
    if len(domains) > 1:
        raise ValueError(
            f"Operator expression mixes different {modes_type.__name__} objects."
        )
    return next(iter(domains), None)


class OperatorNotation:
    """Callable shorthand for one primitive operator class and mode set.

    Parameters
    ----------
    operator_type : type
        Primitive creation, annihilation, or number-operator class.
    modes : FermionModes or BosonModes
        Mode specification used to resolve every subsequent call.
    """

    __slots__ = ("_operator_type", "_modes")

    def __init__(self, operator_type, modes):
        if not isinstance(operator_type, type) or not issubclass(
            operator_type, SingleModeOperator
        ):
            raise TypeError(
                "'operator_type' must be a single-mode primitive operator class."
            )
        modes_type = operator_type.modes_type
        if modes_type is None or not isinstance(modes, modes_type):
            expected = getattr(modes_type, "__name__", "compatible modes")
            raise TypeError(f"'modes' must be a {expected} instance.")
        self._operator_type = operator_type
        self._modes = modes

    def __call__(self, *indices):
        if len(indices) == 1 and isinstance(indices[0], (tuple, list)):
            indices = tuple(indices[0])
        return self._operator_type(indices, modes=self._modes)

    @property
    def modes(self):
        return self._modes

    @property
    def operator_type(self):
        return self._operator_type


def set_notation(operator_type, modes):
    """Bind a primitive operator class to a mode specification.

    Parameters
    ----------
    operator_type : type
        Primitive creation, annihilation, or number-operator class compatible
        with ``modes``.
    modes : FermionModes or BosonModes
        Ordered mode specification used to resolve integer mode indices or
        degree-of-freedom coordinates.

    Returns
    -------
    OperatorNotation
        Callable shorthand. For example, ``c = set_notation(Annihilation,
        modes)`` allows ``c(site, spin)`` instead of
        ``Annihilation((site, spin), modes=modes)``.
    """
    return OperatorNotation(operator_type, modes)
