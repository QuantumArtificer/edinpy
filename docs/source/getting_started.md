# Getting started

EDinPy organizes exact diagonalization around explicit many-body objects. A calculation specifies the modes and Fock space, writes the Hamiltonian in second quantization, solves the required eigensystem, and evaluates observables.

This page introduces the package as a whole. The complete worked tutorial uses the fermionic interface, which is the fully documented reference interface in version 0.1.0.

## Installation

Install a release from PyPI with

```bash
python -m pip install edinpy
```

For a development checkout with the test and documentation dependencies:

```bash
python -m pip install -e ".[test,docs]"
```

Verify the installation:

```python
import edinpy
print(edinpy.__version__)
```

```text
0.1.0
```

## Choose the statistics of the problem

EDinPy exposes fermionic and bosonic functionality separately because their Fock algebras and basis structures are physically different.

::::{grid} 2
:::{grid-item-card} Fermions
:link: fermion/getting_started
:link-type: doc
Use the current reference interface for lattice fermions, multiorbital models, Hubbard-like Hamiltonians, spinful fermion problems, and other systems built from canonical fermionic modes.
:::
:::{grid-item-card} Bosons
:link: boson/index
:link-type: doc
The bosonic backend is included in EDinPy, but its public API is provisional. The bosonic overview describes its present scope and design target.
:::
::::

The corresponding namespaces are distinct:

```python
from edinpy import fermion as edf
from edinpy import boson as edb
```

The names `edf` and `edb` are local Python aliases. EDinPy does not prescribe them.

## The common ED workflow

Although the algebra differs, a finite exact-diagonalization calculation usually answers the same sequence of questions.

1. Identify the one-particle degrees of freedom. A lattice calculation may require site and spin labels. A multiorbital problem may also require orbital, layer, or valley indices.
2. Choose the many-body Hilbert space. Particle-number constraints and occupation rules determine the basis dimension and therefore the computational cost.
3. Write the Hamiltonian. EDinPy uses explicit second-quantized expressions so that the mathematical operator remains recognizable in the Python code.
4. Choose the required eigenstates. Small problems may justify a complete dense eigensystem. Larger sparse problems usually require only a few low-energy eigenpairs.
5. Evaluate the quantities relevant to the problem. Local occupations, double occupancy, spin and charge correlations, structure factors, currents, order-parameter fluctuations, and transition matrix elements are standard ED observables once the eigenstates are available.

## Where to continue

For a complete calculation, continue with {doc}`fermion/getting_started`. It solves the half-filled Hubbard dimer, prints the symbolic Hamiltonian and Fock basis, diagonalizes the many-body matrix, converts an eigenvector into a Fock-space ket, and evaluates observables with standard bra-ket expressions.

The {doc}`fermion/examples/index` section then develops larger and more physical calculations. The {doc}`boson/index` page describes the present status and design target of the bosonic backend.
