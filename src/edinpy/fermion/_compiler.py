"""Structural compiler for fermionic symbolic operators.

The compiler lowers literal Fock-algebra expressions in three stages:

1. flatten additive expressions;
2. normalize each term into a coefficient, ordered factors, and a structural
   primitive signature;
3. dispatch exact signatures or broad algebraic categories to lowered IR
   kernels.

Unsupported symbolic structures are preserved as generic fallback kernels.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

from ._ir import (
    _GenericKernel,
    _HoppingKernel,
    _MonomialKernel,
    _NumberProductKernel,
)
from ._algebra import (
    Annihilation,
    Creation,
    Number,
    OperatorProduct,
    OperatorSum,
    Scalar,
)


class _PrimitiveKind(Enum):
    """Primitive operator kinds used for structural compiler dispatch."""

    CREATION = auto()
    ANNIHILATION = auto()
    NUMBER = auto()
    OTHER = auto()


@dataclass(frozen=True, slots=True)
class _NormalizedTerm:
    """One additive operator term in normalized compiler form."""

    coefficient: complex
    factors: tuple
    signature: tuple
    original: object


def _flatten_sum(operator):
    """Yield additive terms from a nested OperatorSum."""
    if isinstance(operator, OperatorSum):
        for term in operator.os:
            yield from _flatten_sum(term)
    else:
        yield operator


def _split_product(term):
    """Separate the numerical coefficient from ordered operator factors."""
    coefficient = 1.0

    if isinstance(term, Scalar):
        return term.value, ()

    if isinstance(term, OperatorProduct):
        factors = []

        for factor in term.op:
            if isinstance(factor, Scalar):
                coefficient *= factor.value
            else:
                factors.append(factor)

        return coefficient, tuple(factors)

    return coefficient, (term,)


def _primitive_kind(factor):
    """Return the compiler opcode associated with a symbolic primitive."""
    if isinstance(factor, Creation):
        return _PrimitiveKind.CREATION

    if isinstance(factor, Annihilation):
        return _PrimitiveKind.ANNIHILATION

    if isinstance(factor, Number):
        return _PrimitiveKind.NUMBER

    return _PrimitiveKind.OTHER


def _normalize_term(term):
    """Normalize one additive term and compute its structural signature."""
    coefficient, factors = _split_product(term)

    return _NormalizedTerm(
        coefficient=coefficient,
        factors=factors,
        signature=tuple(
            _primitive_kind(factor)
            for factor in factors
        ),
        original=term,
    )


def _number_mask(factors):
    """Return the occupancy mask required by number operators."""
    mask = 0

    for factor in factors:
        mask |= 1 << factor.site

    return mask


def _between_mask(site_i, site_j):
    """Return the bit mask strictly between two fermionic modes."""
    low = min(site_i, site_j)
    high = max(site_i, site_j)

    if high - low <= 1:
        return 0

    return (1 << high) - (1 << (low + 1))


def _compile_monomial(factors, coefficient):
    """Compile a fermionic monomial into occupation and parity masks.

    Parameters
    ----------
    factors : tuple
        Ordered creation and annihilation operators in the symbolic monomial.
    coefficient : numbers.Number
        Numerical prefactor of the monomial.

    Returns
    -------
    _MonomialKernel
        Lowered kernel containing the initial occupation constraints, bit
        transition, Jordan--Wigner parity mask, and accumulated phase.

    Notes
    -----
    Operators act on a bit-encoded Fock state from right to left. The
    fermionic sign is the parity of occupied modes preceding each acted-on
    mode, with additional phase changes accumulated as earlier right-acting
    operators toggle occupations.
    """
    required_occupied_mask = 0
    required_empty_mask = 0
    transition_mask = 0
    parity_mask = 0
    phase = 1

    for factor in reversed(factors):
        creation = isinstance(factor, Creation)
        site = factor.site
        bit = 1 << site
        toggled = bool(transition_mask & bit)

        requires_occupied = toggled if creation else not toggled

        if requires_occupied:
            required_occupied_mask |= bit
        else:
            required_empty_mask |= bit

        lower_mask = bit - 1
        parity_mask ^= lower_mask

        if (transition_mask & lower_mask).bit_count() & 1:
            phase = -phase

        transition_mask ^= bit

    return _MonomialKernel(
        coefficient=phase * coefficient,
        required_occupied_mask=required_occupied_mask,
        required_empty_mask=required_empty_mask,
        transition_mask=transition_mask,
        parity_mask=parity_mask,
    )


def _lower_number_product(term):
    """Lower a product containing only number operators."""
    return _NumberProductKernel(
        coefficient=term.coefficient,
        mask=_number_mask(term.factors),
    )


def _lower_creation_annihilation(term):
    r"""Lower :math:`c_i^\dagger c_j` to a specialized one-body kernel.

    Equal source and destination modes reduce exactly to a number operator.
    Otherwise the lowered hopping kernel stores the two occupation bits, the
    transition mask, and the parity mask spanning the modes strictly between
    source and destination.
    """
    creation, annihilation = term.factors
    destination = creation.site
    source = annihilation.site

    if destination == source:
        return _NumberProductKernel(
            coefficient=term.coefficient,
            mask=1 << source,
        )

    source_bit = 1 << source
    destination_bit = 1 << destination

    return _HoppingKernel(
        coefficient=term.coefficient,
        source_bit=source_bit,
        destination_bit=destination_bit,
        transition_mask=source_bit | destination_bit,
        parity_mask=_between_mask(destination, source),
    )


def _lower_fermion_monomial(term):
    """Lower an arbitrary creation/annihilation product."""
    return _compile_monomial(
        term.factors,
        term.coefficient,
    )


def _is_number_product(term):
    """Whether every factor is a number operator."""
    return all(
        kind is _PrimitiveKind.NUMBER
        for kind in term.signature
    )


def _is_fermion_monomial(term):
    """Whether the term contains only creation/annihilation primitives."""
    return (
        bool(term.signature)
        and all(
            kind in (
                _PrimitiveKind.CREATION,
                _PrimitiveKind.ANNIHILATION,
            )
            for kind in term.signature
        )
    )


_EXACT_LOWERERS = {
    (
        _PrimitiveKind.CREATION,
        _PrimitiveKind.ANNIHILATION,
    ): _lower_creation_annihilation,
}


_CATEGORY_LOWERERS = (
    (_is_number_product, _lower_number_product),
    (_is_fermion_monomial, _lower_fermion_monomial),
)


def _lower_normalized_term(term):
    """Lower one normalized term through structural dispatch."""
    lowerer = _EXACT_LOWERERS.get(term.signature)

    if lowerer is not None:
        return lowerer(term)

    for predicate, lowerer in _CATEGORY_LOWERERS:
        if predicate(term):
            return lowerer(term)

    return _GenericKernel(term.original)


def lower_operator(operator):
    """Lower symbolic Fock algebra to a tuple of executable IR kernels."""
    normalized_terms = (
        _normalize_term(term)
        for term in _flatten_sum(operator)
    )

    return tuple(
        _lower_normalized_term(term)
        for term in normalized_terms
    )
