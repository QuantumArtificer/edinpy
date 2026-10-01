"""Lower bosonic operator expressions into a small primitive representation."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Number as Numeric
from typing import Literal, TypeAlias

from ._algebra import (
    Annihilation,
    Creation,
    Number,
    Operator,
    OperatorProduct,
    OperatorSum,
    Scalar,
)

PrimitiveKind: TypeAlias = Literal["create", "annihilate", "number"]


@dataclass(frozen=True, slots=True)
class Primitive:
    """One elementary bosonic operation.

    Parameters
    ----------
    kind : {"create", "annihilate", "number"}
        Operation represented by the primitive. The symbolic expression is
        stored left-to-right; execution later reverses that order so the
        rightmost operator acts on the ket first.
    mode : int
        Flat bosonic mode index on which the primitive acts.
    """

    kind: PrimitiveKind
    mode: int


@dataclass(frozen=True, slots=True)
class LoweredTerm:
    """One additive term after primitive lowering.

    Parameters
    ----------
    coefficient : numbers.Number
        Scalar multiplying the primitive sequence.
    primitives : tuple of Primitive
        Primitive operators in the same left-to-right order as the symbolic
        term.
    source : Operator
        Original symbolic term. It is retained so unsupported custom operators
        can use the literal NumPy fallback without losing their semantics.
    fully_lowered : bool
        Whether every non-scalar factor was recognized as a primitive bosonic
        creation, annihilation, or number operator.
    """

    coefficient: Numeric
    primitives: tuple[Primitive, ...]
    source: Operator
    fully_lowered: bool

    @property
    def particle_delta(self) -> int | None:
        """Return the net particle-number change, or ``None`` when unsupported."""
        if not self.fully_lowered:
            return None
        delta = 0
        for primitive in self.primitives:
            if primitive.kind == "create":
                delta += 1
            elif primitive.kind == "annihilate":
                delta -= 1
        return delta


@dataclass(frozen=True, slots=True)
class LoweredOperator:
    """Flat additive representation consumed by the bosonic compiler.

    Parameters
    ----------
    terms : tuple of LoweredTerm
        Additive terms in deterministic symbolic order.
    """

    terms: tuple[LoweredTerm, ...]


def _additive_terms(operator: Operator) -> tuple[Operator, ...]:
    """Flatten nested sums while preserving symbolic term order."""
    if isinstance(operator, OperatorSum):
        terms: list[Operator] = []
        for term in operator.os:
            terms.extend(_additive_terms(term))
        return tuple(terms)
    return (operator,)


def _term_factors(term: Operator) -> tuple[Operator, ...]:
    """Return multiplicative factors of one additive symbolic term."""
    if isinstance(term, OperatorProduct):
        return tuple(term.op)
    return (term,)


def _lower_term(term: Operator) -> LoweredTerm:
    """Lower one symbolic term when all non-scalar factors are recognized."""
    coefficient: Numeric = 1
    primitives: list[Primitive] = []

    for factor in _term_factors(term):
        if isinstance(factor, Scalar):
            coefficient *= factor.value
        elif isinstance(factor, Creation):
            primitives.append(Primitive("create", factor.mode))
        elif isinstance(factor, Annihilation):
            primitives.append(Primitive("annihilate", factor.mode))
        elif isinstance(factor, Number):
            primitives.append(Primitive("number", factor.mode))
        else:
            return LoweredTerm(
                coefficient=1,
                primitives=(),
                source=term,
                fully_lowered=False,
            )

    return LoweredTerm(
        coefficient=coefficient,
        primitives=tuple(primitives),
        source=term,
        fully_lowered=True,
    )


def lower_operator(operator: Operator) -> LoweredOperator:
    """Lower a bosonic expression into additive primitive terms.

    Parameters
    ----------
    operator : Operator
        Bosonic symbolic expression to compile. Built-in creation,
        annihilation, number, scalar, product, and sum nodes are lowered
        numerically. Custom operator subclasses remain attached as literal
        fallback terms.

    Returns
    -------
    LoweredOperator
        Flat additive representation suitable for kernel compilation.
    """
    if not isinstance(operator, Operator):
        raise TypeError("'operator' must be a bosonic Operator expression.")
    return LoweredOperator(tuple(_lower_term(term) for term in _additive_terms(operator)))
