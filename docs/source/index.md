# EDinPy

Exact diagonalization for finite quantum many-body systems

EDinPy builds finite many-body problems directly in Fock space. The package keeps mode definitions, basis states, second-quantized operators, Hamiltonian matrices, eigenstates, and observables visible in the calculation.

The fermionic interface is designed around literal Fock algebra. A hopping term can be written as

```python
-t * (cd(i, spin) * c(j, spin) + cd(j, spin) * c(i, spin))
```

and an expectation value as

```python
psi.dag * O * psi
```

When a Hamiltonian matrix is needed, EDinPy recognizes common operator structures and compiles them to sparse execution kernels. The symbolic expression remains the public representation of the model.

::::{grid} 2
:::{grid-item-card} Getting started
:link: getting_started
:link-type: doc
Install EDinPy, choose a particle-statistics module, and see the basic exact-diagonalization workflow.
:::
:::{grid-item-card} Fermions
:link: fermion/index
:link-type: doc
Modes, fixed-particle-number sectors, operator algebra, sparse Hamiltonians, eigensolvers, observables, examples, and API reference.
:::
:::{grid-item-card} Bosons
:link: boson/index
:link-type: doc
Status and scope of the bosonic interface included with EDinPy.
:::
:::{grid-item-card} Validation
:link: validation/index
:link-type: doc
Correctness tests, analytic checks, and reproducible performance benchmarks.
:::
:::{grid-item-card} Development
:link: development/index
:link-type: doc
Testing, benchmarking, documentation, contribution, and release procedures.
:::
:::{grid-item-card} References
:link: references
:link-type: doc
Scientific and numerical references used by the documentation.
:::
::::

## What EDinPy is for

Exact diagonalization solves the represented finite Hilbert space directly, up to the tolerances and floating-point accuracy of the numerical solver. This makes it useful for small-cluster studies, teaching, testing analytical calculations, and benchmarking approximate many-body methods.[^lin]

The main limitation is the size of the Hilbert space. For $M$ fermionic modes at fixed particle number $N$,

$$
\dim \mathcal H_N = \binom{M}{N}.
$$

The dimension therefore grows rapidly even before the cost of matrix construction and diagonalization is considered. Sparse matrices and iterative eigensolvers extend the useful range, but they do not change this fundamental scaling.

## Package layout

EDinPy exposes fermionic and bosonic functionality through separate namespaces:

```python
from edinpy import fermion as edf
from edinpy import boson as edb
```

The fermionic API is the primary documented interface. It uses explicit mode ownership, fixed-$N$ sectors, optional particle-number projections on labeled degrees of freedom, sparse Hamiltonian construction, basis-backed eigenstates, and literal operator-state algebra.

The bosonic module uses a separate API and currently has less validation and documentation coverage than the fermionic module. Read {doc}`boson/index` before starting a bosonic calculation.

## Performance

EDinPy uses direct constrained-basis generation for particle-resolved sectors and optimized sparse construction paths for several common number-conserving fermionic structures. These optimizations reduce avoidable basis and matrix-construction work while keeping the public API close to the Fock algebra.

Performance depends on the basis dimension, matrix sparsity, operator structure, requested eigenpairs, hardware, and NumPy/SciPy build. The package therefore provides reproducible benchmark scripts instead of a single speed claim. See {doc}`validation/fermion_performance`.

```{toctree}
:maxdepth: 2
:hidden:

getting_started
fermion/index
boson/index
validation/index
development/index
references
```

[^lin]: H. Q. Lin, J. E. Gubernatis, H. Gould, and J. Tobochnik, "Exact Diagonalization Methods for Quantum Systems," *Computers in Physics* **7**, 400-407 (1993), [doi:10.1063/1.4823192](https://doi.org/10.1063/1.4823192).
