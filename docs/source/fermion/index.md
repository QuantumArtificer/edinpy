# Fermionic exact diagonalization

Literal second-quantized algebra, explicit Fock sectors, and many-body observables

The fermionic interface is the current reference architecture of EDinPy. A calculation defines ordered fermionic modes, chooses a fixed-particle-number sector, writes the Hamiltonian directly with creation, annihilation, and number operators, and associates that operator with the sector when a matrix representation is needed.

The same algebra remains available after diagonalization. Eigensolver columns can be represented as basis-backed `FockVector` objects, so expectation values and transition amplitudes retain standard Dirac notation:

```python
psi0 = hamiltonian.eigenstate(0)
expectation = psi0.dag * O * psi0
transition = psi1.dag * O * psi0
```

::::{grid} 2
:::{grid-item-card} Start with fermions
:link: getting_started
:link-type: doc
Solve an exactly checkable Hubbard dimer while inspecting the modes, Fock basis, symbolic Hamiltonian, matrix, eigenstates, and observables.
:::
:::{grid-item-card} User guide
:link: user_guide/index
:link-type: doc
Learn mode ordering, sectors, operator algebra, Hamiltonian construction, eigensolvers, diagnostics, observables, and numerical tradeoffs.
:::
:::{grid-item-card} API reference
:link: reference/index
:link-type: doc
Detailed signatures, parameters, return values, properties, and methods for the complete public fermionic API.
:::
:::{grid-item-card} Worked examples
:link: examples/index
:link-type: doc
Use analytically checkable models and finite interacting systems to study eigensystems, correlations, structure factors, exchange, and persistent current.
:::
:::{grid-item-card} Theory and numerical methods
:link: theory/index
:link-type: doc
Review the Fock-space conventions, fermionic signs, sparse construction strategy, and Hermitian eigensolvers behind the public workflow.
:::
::::

## A direct interface to the algebra

For a spinful lattice, the mode structure remains explicit:

```python
from edinpy import fermion as edf

site = edf.DoF(6, name="site")
spin = edf.DoF(2, name="spin")
modes = edf.FermionModes(site, spin)
sector = edf.NParticleSector(modes, N=6)
```

Operator notation is chosen by the user:

```python
c = edf.set_notation(edf.Annihilation, modes)
cd = edf.set_notation(edf.Creation, modes)
n = edf.set_notation(edf.Number, modes)
```

The Hamiltonian can then be written in the same operator language used on paper:

```python
H = -1.0 * (cd(0, 0) * c(1, 0) + cd(1, 0) * c(0, 0))
H += 4.0 * n(0, 0) * n(0, 1)

hamiltonian = edf.Hamiltonian(H, sector)
energies, vectors = hamiltonian.eigsolve(k=4, which="SA")
```

Common terms such as `Hopping`, `Hubbard`, and `HeisenbergExchange` expand to the same symbolic algebra. They use the same matrix-construction backend as handwritten expressions.

## Scope of the current fermionic backend

The present Hamiltonian representation uses a fixed total-particle-number sector. Full Fock-space Hamiltonians, fermion-parity-only sectors, translation/momentum blocks, and other symmetry-resolved sector types are not yet part of the public API. See {doc}`reference/limitations` for the current boundaries and {doc}`../validation/index` for the validation status.

```{toctree}
:maxdepth: 2
:hidden:

getting_started
user_guide/index
reference/index
examples/index
theory/index
```
