# Getting started

A finite exact-diagonalization calculation has a simple structure:

1. define the one-particle labels or modes
2. choose a finite many-body basis or symmetry sector
3. write the Hamiltonian in second quantization
4. construct its matrix representation
5. solve the part of the spectrum that is needed
6. evaluate observables in the resulting states

EDinPy keeps each of these steps explicit.

## Installation

Install a release from PyPI:

```bash
python -m pip install edinpy
```

For a development checkout:

```bash
python -m pip install -e ".[test,docs,dev]"
```

Verify the installation:

```python
import edinpy
print(edinpy.__version__)
```

## Choose the particle statistics

Fermions and bosons use different Fock algebras, so EDinPy keeps them in separate namespaces.

```python
from edinpy import fermion as edf
from edinpy import boson as edb
```

The local names `edf` and `edb` are ordinary Python aliases. You may choose any names you prefer.

::::{grid} 2
:::{grid-item-card} Fermions
:link: fermion/getting_started
:link-type: doc
Use the documented fixed-particle-number interface for lattice fermions, spinful fermion models, multiorbital problems, and related finite systems.
:::
:::{grid-item-card} Bosons
:link: boson/index
:link-type: doc
Read the bosonic status page before using the older bosonic interface in new code.
:::
::::

## A note on Hilbert-space size

For $M$ fermionic modes with exactly $N$ particles, the basis dimension is

$$
\binom{M}{N}.
$$

For example, 12 spin-orbitals at half filling give

$$
\binom{12}{6}=924
$$

basis states, while 24 spin-orbitals at half filling already give

$$
\binom{24}{12}=2\,704\,156.
$$

This growth is the central practical limit of exact diagonalization. Before building a large Hamiltonian, estimate the basis size and decide whether you need the complete spectrum or only a few extremal eigenpairs.

## Continue with the fermionic tutorial

The {doc}`fermion/getting_started` tutorial solves the half-filled Hubbard dimer from start to finish. It shows how mode ordering determines fermionic signs, how to inspect the basis and sparse matrix, how to use dense and sparse eigensolvers, and how to evaluate observables with bra-ket expressions.

The {doc}`fermion/examples/index` section then develops larger examples, including a Hubbard chain, an extended Hubbard ring, a flux-threaded ring, and fermionic spin exchange.
