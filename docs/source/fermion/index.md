# Fermionic exact diagonalization

The fermionic interface is organized around four explicit objects:

- `FermionModes` defines the ordered single-particle modes
- `NParticleSector` defines the fixed-$N$ many-body basis
- operator expressions represent the second-quantized algebra
- `Hamiltonian` connects an operator expression to a numerical matrix and eigensolver

A calculation can therefore stay close to the notation used to define the model. For example,

```python
hop = cd(i, spin) * c(j, spin)
H += -t * (hop + hop.dag)
```

constructs a Hermitian hopping term, while

```python
psi.dag * O * psi
```

evaluates the expectation value of an operator in a basis-backed Fock vector.

::::{grid} 2
:::{grid-item-card} Tutorial
:link: getting_started
:link-type: doc
Build and solve the half-filled Hubbard dimer step by step.
:::
:::{grid-item-card} User guide
:link: user_guide/index
:link-type: doc
Modes, sectors, operator algebra, Hamiltonian construction, eigensolvers, observables, and numerical considerations.
:::
:::{grid-item-card} Worked examples
:link: examples/index
:link-type: doc
Complete calculations for several standard finite fermion models.
:::
:::{grid-item-card} Theory and methods
:link: theory/index
:link-type: doc
Fock-space conventions, fermionic signs, sparse construction, and eigensolvers.
:::
:::{grid-item-card} API reference
:link: reference/index
:link-type: doc
Signatures and docstrings for the public fermionic interface.
:::
:::{grid-item-card} Validation
:link: ../validation/index
:link-type: doc
Algebra tests, independent matrix checks, analytic benchmarks, and performance measurements.
:::
::::

## Literal algebra with a compiled backend

EDinPy does not require a separate model-description language for common fermionic Hamiltonians. Primitive creation, annihilation, and number operators form sums and products directly in Python.

Before matrix construction, the expression is compiled. Recognized number-conserving structures are lowered to compact bit-mask representations and sparse matrix entries are emitted with specialized kernels. General symbolic expressions still have a fallback execution path.

This split keeps implementation details out of the model definition. It also means that readable operator expressions can use optimized matrix-construction paths when their structure is recognized.

## Fixed-particle-number scope

Version 0.2.0 builds fermionic Hamiltonians in one fixed total-particle-number sector at a time. Full Fock-space, parity, momentum, translation, and fixed-spin-population sectors are not yet part of the public API. See {doc}`reference/limitations` for the current numerical and feature limits.

```{toctree}
:maxdepth: 2
:hidden:

getting_started
user_guide/index
examples/index
theory/index
reference/index
```
