# EDinPy

[![CI](https://github.com/QuantumArtificer/edinpy/actions/workflows/ci.yml/badge.svg?branch=main&event=push)](https://github.com/QuantumArtificer/edinpy/actions/workflows/ci.yml?query=branch%3Amain)
[![Docs](https://github.com/QuantumArtificer/edinpy/actions/workflows/docs.yml/badge.svg)](https://quantumartificer.github.io/edinpy/)
[![Python](https://img.shields.io/badge/python-%3E%3D3.10-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/github/license/QuantumArtificer/edinpy.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/747491440.svg)](https://doi.org/10.5281/zenodo.23090945)

EDinPy is a Python package for exact diagonalization of finite quantum many-body systems in Fock space. Its central aim is to keep numerical calculations close to the second-quantized algebra used to define the physics.

Fermionic and bosonic calculations can define labeled modes, construct finite particle-number sectors, write Hamiltonians and observables from literal creation-annihilation algebra or standard helper functions, solve finite eigensystems, and evaluate observables with basis-aware states. Sparse matrices and matrix-free execution are available when the represented Hilbert space grows beyond dense linear algebra.

EDinPy is suited to finite-cluster studies, interacting few-body and lattice problems, checks of analytical limits, teaching, and benchmarks for approximate many-body methods. Practical system size remains controlled by the growth of the many-body Hilbert space.

Documentation: [quantumartificer.github.io/edinpy](https://quantumartificer.github.io/edinpy/)

## Installation

Install EDinPy from source:

```bash
git clone https://github.com/QuantumArtificer/edinpy.git
cd edinpy
python -m pip install -e .
```

Optional Numba execution is available with

```bash
python -m pip install -e ".[numba]"
```

For development, install the test, documentation, and release tools as well:

```bash
python -m pip install -e ".[test,docs,dev]"
```

EDinPy requires Python 3.10 or newer, NumPy, and SciPy.

## A first calculation

The half-filled two-site Hubbard model is small enough to inspect directly and already contains hopping, local interactions, correlated eigenstates, and spin physics:

$$
H=-t\sum_\sigma\left(c_{0\sigma}^\dagger c_{1\sigma}+c_{1\sigma}^\dagger c_{0\sigma}\right)
+U\sum_i n_{i\uparrow}n_{i\downarrow}.
$$

```python
from edinpy import fermion as edf

L = 2
UP, DOWN = 0, 1
t = 1.0
U = 4.0

site = edf.DoF(L, name="site")
spin = edf.DoF(2, name="spin", labels=("up", "down"))
modes = edf.FermionModes(site, spin)
sector = edf.NParticleSector(modes, N=2).build()

c = edf.set_notation(edf.Annihilation, modes)
cd = edf.set_notation(edf.Creation, modes)
n = edf.set_notation(edf.Number, modes)

H = 0
for sigma in (UP, DOWN):
    hop = cd(0, sigma) * c(1, sigma)
    H += -t * (hop + hop.dag)

for i in range(L):
    H += U * n(i, UP) * n(i, DOWN)

hamiltonian = edf.Hamiltonian(H, sector)
energies, _ = hamiltonian.eigsolve(k=None)
psi0 = hamiltonian.eigenstate(0)

D = sum(
    (n(i, UP) * n(i, DOWN) for i in range(L)),
    start=0,
)

print(energies)
print(psi0.dag * D * psi0)
```

```text
[-0.82842712  0.          0.          0.          4.          4.82842712]
0.1464466094067262
```

The operator expression remains readable throughout the calculation, and the same literal algebra is available for custom observables. Standard constructors such as `Hopping`, `Hubbard`, and `HeisenbergExchange` provide compact forms for common physical terms.

## Fermions and bosons

The two particle statistics are exposed through separate namespaces:

```python
from edinpy import fermion as edf
from edinpy import boson as edb
```

Fermionic modes obey the occupation constraint $n_i\in\{0,1\}$ and canonical anticommutation relations. Bosonic modes admit multiple occupation and obey canonical commutation relations. Both interfaces use the same broad workflow from mode definition and sector construction through diagonalization and observable evaluation.

## Core workflow

1. Define the one-particle labels with `DoF` and `FermionModes` or `BosonModes`.
2. Build the fixed-particle-number sector with `NParticleSector`, adding symmetry constraints for separately conserved mode populations when appropriate.
3. Construct Hamiltonians and observables from primitive operators, literal sums and products, or standard helper functions.
4. Use `Hamiltonian` to obtain sparse or matrix-free numerical action and solve the required eigenpairs.
5. Recover basis-aware states and evaluate expectation values or custom correlation functions directly from the operator algebra.

Performance depends on basis dimension, operator structure, sparsity, requested eigenpairs, and hardware. Reproducible benchmark entry points are listed in [`benchmarks/README.md`](benchmarks/README.md).

## Documentation

- [Getting started](https://quantumartificer.github.io/edinpy/getting_started.html)
- [User guide](https://quantumartificer.github.io/edinpy/user_guide/index.html)
- [API reference](https://quantumartificer.github.io/edinpy/reference/index.html)
- [Worked examples](https://quantumartificer.github.io/edinpy/examples/index.html)
- [Theory and numerical methods](https://quantumartificer.github.io/edinpy/theory/index.html)
- [Validation and benchmarks](https://quantumartificer.github.io/edinpy/validation/index.html)

Executable examples are split between [`examples/fermion/`](examples/fermion/) and [`examples/boson/`](examples/boson/).

## Testing and development

Run the test suite with

```bash
python -m pytest
```

Build the documentation with warnings treated as errors:

```bash
python -m sphinx -W --keep-going -b html docs/source docs/_build/html
```

Release and contribution notes are in the [development documentation](https://quantumartificer.github.io/edinpy/development/index.html).

## Citation and provenance

If EDinPy contributes to published work, cite the archived software release used in the calculation. [`CITATION.cff`](CITATION.cff) contains the package citation metadata. [`PROVENANCE.md`](PROVENANCE.md) records algorithmic sources and numerical-library references used by the implementation.

Scientific references used in the documentation are collected on the [references page](https://quantumartificer.github.io/edinpy/references.html).

## License

EDinPy is distributed under the [MIT License](LICENSE).
