# EDinPy

Exact diagonalization for finite quantum many-body systems

EDinPy is a Python package for building and solving finite many-body problems directly in Fock space. It contains fermionic and bosonic exact-diagonalization functionality. The Hilbert space, second-quantized operators, Hamiltonian, eigensystem, and observables remain explicit throughout the calculation.

The fermionic interface is the current reference architecture. It provides explicit mode ownership, fixed-particle-number sectors, literal operator-state algebra, optimized sparse Hamiltonian construction, basis-backed eigenstates, and direct evaluation of observables. The bosonic implementation is also included in EDinPy, but its public interface predates the current fermionic API and remains provisional.

::::{grid} 2
:::{grid-item-card} Start here
:link: getting_started
:link-type: doc
Install EDinPy, understand the package layout, and choose the fermionic or bosonic workflow appropriate to the problem.
:::
:::{grid-item-card} Fermionic exact diagonalization
:link: fermion/index
:link-type: doc
The current reference workflow: literal fermionic Fock algebra, fixed-$N$ sectors, sparse Hamiltonians, eigenstates, observables, examples, theory, and API reference.
:::
:::{grid-item-card} Bosonic exact diagonalization
:link: boson/index
:link-type: doc
Bosonic Fock-space functionality is part of EDinPy. Read the current status, scope, and design target of the provisional bosonic interface.
:::
:::{grid-item-card} Validation and benchmarks
:link: validation/index
:link-type: doc
Correctness strategy, analytic checks, and performance benchmarks. The present quantitative validation suite focuses on the fermionic backend.
:::
:::{grid-item-card} Development
:link: development/index
:link-type: doc
Testing, benchmarking, contribution conventions, and the package-level development philosophy.
:::
::::

## Scope and design

Exact diagonalization is a finite-system method. EDinPy is intended for controlled finite calculations, benchmark problems, and small-cluster studies where the Hilbert space can be represented explicitly. The package does not claim to remove the exponential scaling of exact diagonalization.

The public interface emphasizes readable Fock-space algebra. Recognized operator structures are compiled to sparse execution kernels so that this literal notation does not require a separate slow execution path. Performance still depends on Hilbert-space dimension, sparsity, operator structure, and the numerical environment. Quantitative benchmarks are reported separately in {doc}`validation/index`.

## One package, two particle statistics

Fermionic and bosonic exact diagonalization share the same broad computational workflow:

1. define the one-particle labels or modes
2. construct a many-body Fock basis or sector
3. write a Hamiltonian in second quantization
4. construct its matrix representation
5. solve the required part of the eigensystem
6. evaluate observables, correlations, and derived quantities

The two statistics differ in their algebra and Hilbert-space structure. Fermionic modes obey canonical anticommutation relations and have occupations $n_p\in\{0,1\}$. Bosonic modes obey canonical commutation relations and may require an explicit occupation or Hilbert-space truncation in practical finite calculations. Separate public namespaces keep these differences visible.

## Current implementation status

### Fermions

The fermionic API is the reference implementation documented in detail on this site. It is organized around `FermionModes`, `NParticleSector`, literal creation, annihilation, and number operators, `FockVector` eigenstates, sparse Hamiltonian construction, and direct bra-ket expressions such as

```python
psi.dag * O * psi
```

for expectation values and transition amplitudes.

### Bosons

The bosonic implementation remains available in EDinPy, but its interface still reflects the earlier package architecture. Its current names and global-state conventions are provisional. The bosonic documentation remains high-level until the interface receives the same API, numerical, and validation review as the fermionic backend.

The separate `edinpy.fermion` and `edinpy.boson` namespaces keep the two Fock algebras explicit within one package.

```{toctree}
:maxdepth: 2
:hidden:

getting_started
fermion/index
boson/index
validation/index
development/index
```
