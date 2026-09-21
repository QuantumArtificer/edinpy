"""Lowered intermediate representation for fermionic operators.

These immutable kernel records describe executable algebraic structures after
symbolic normalization and recognition. They contain no matrix-construction or
solver logic.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class _NumberProductKernel:
    """Diagonal product of number operators."""

    coefficient: complex
    mask: int


@dataclass(frozen=True, slots=True)
class _HoppingKernel:
    """One-body fermionic transition c_i^dagger c_j."""

    coefficient: complex
    source_bit: int
    destination_bit: int
    transition_mask: int
    parity_mask: int


@dataclass(frozen=True, slots=True)
class _HoppingPairKernel:
    """Combined hopping transitions between one unordered pair of modes."""

    low_bit: int
    high_bit: int
    transition_mask: int
    parity_mask: int
    low_from_high: complex
    high_from_low: complex


@dataclass(frozen=True, slots=True)
class _SimpleHoppingKernel:
    """Parity-free hopping with the same coefficient in both directions."""

    coefficient: complex
    transition_mask: int


@dataclass(frozen=True, slots=True)
class _ParityFreeHoppingKernel:
    """Parity-free hopping with independent directional coefficients."""

    low_bit: int
    high_bit: int
    transition_mask: int
    low_from_high: complex
    high_from_low: complex


@dataclass(frozen=True, slots=True)
class _MonomialKernel:
    """Compiled product of fermionic creation and annihilation operators."""

    coefficient: complex
    required_occupied_mask: int
    required_empty_mask: int
    transition_mask: int
    parity_mask: int


@dataclass(frozen=True, slots=True)
class _GenericKernel:
    """Fallback retaining the original symbolic Fock algebra."""

    term: object
