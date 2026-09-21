# Fermionic correctness tests

The fermionic validation strategy checks the algebra, the optimized execution paths, and physically interpretable benchmarks independently. The goal is to avoid a circular test in which one optimized implementation is only compared with another path sharing the same assumptions.

## Canonical algebra

Small Fock spaces are used to test the canonical anticommutation relations exhaustively. Creation, annihilation, number operators, products, sums, Hermitian conjugation, operator-state action, and bra-ket products are compared against explicit occupation-bit actions.

The observable tests also verify that

```python
psi.dag * O * psi
```

agrees with the corresponding sparse matrix contraction, and that the method form

```python
psi.inner(O * psi)
```

returns the same scalar.

## Basis enumeration

Fixed-$N$ basis dimensions and populations are checked against $\binom{M}{N}$. Dedicated regressions cover mode counts larger than 64, including the vacuum sector where the number of modes cannot be inferred from occupied bits.

## Compiler and sparse matrix construction

Random small Hamiltonians are compared with an independent state-by-state bit-string reference calculation. This is important because it tests the optimized compiler against a path that does not share the same lowering implementation.

## Common-operator equivalence

`Hopping`, `Onsite`, `DensityDensity`, `Hubbard`, spin operators, `HeisenbergExchange`, and `PairHopping` are compared with the corresponding handwritten primitive expressions. The helpers are algebraic conveniences and use the same numerical implementations as the corresponding primitive expressions.

## Dense and sparse eigensolvers

Tests cover real and complex dtypes, Hermiticity checks, dense/sparse agreement, eigenstate conversion, simultaneous independent mode/sector objects, and ownership errors when an operator is applied to a state built from another `FermionModes` object.

## Analytic Hubbard-dimer benchmark

The documentation figure below provides a physics-level benchmark in addition to unit tests. The EDinPy points are obtained by rebuilding and diagonalizing the symmetric half-filled Hubbard dimer. The lines are analytic results for the ground-state energy, double occupancy, local moment, spin correlation, and singlet-triplet gap. The dimer example gives the formulas and reference.[^carrascal]

```{figure} ../_static/figures/hubbard_dimer_observables.svg
:width: 90%
:alt: Analytic Hubbard dimer benchmark for EDinPy

EDinPy reproduces several independent analytic Hubbard-dimer observables over the plotted interaction range. Checking quantities with different operator content is a stronger physics-level regression than checking the ground-state energy alone.
```

The complete executable example is in {doc}`../fermion/examples/hubbard_dimer`.

[^carrascal]: D. J. Carrascal, J. Ferrer, J. C. Smith, and K. Burke, "The Hubbard dimer: a density functional case study of a many-body problem," *J. Phys.: Condens. Matter* **27**, 393001 (2015), [doi:10.1088/0953-8984/27/39/393001](https://doi.org/10.1088/0953-8984/27/39/393001).
