# EDinPy

[![CI](https://github.com/QuantumArtificer/edinpy/actions/workflows/ci.yml/badge.svg)](https://github.com/QuantumArtificer/edinpy/actions/workflows/ci.yml)
[![Docs](https://github.com/QuantumArtificer/edinpy/actions/workflows/docs.yml/badge.svg)](https://quantumartificer.github.io/edinpy/)
[![Python](https://img.shields.io/badge/python-%3E%3D3.10-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/github/license/QuantumArtificer/edinpy.svg)](LICENSE)

EDinPy is a Python package for exact diagonalization of finite quantum many-body systems in Fock space. Its main goal is to keep the calculation close to the algebra written on paper while still using sparse numerical methods where they are useful.

The fermionic interface lets you define modes, choose a fixed-particle-number sector, optionally resolve conserved particle populations by labeled degrees of freedom, write operators directly with creation and annihilation operators, build a sparse Hamiltonian, diagonalize it, and evaluate observables with ordinary bra-ket expressions.

EDinPy is well suited to small-system calculations, teaching and exploration, checks of analytical results, and benchmarks for approximate many-body methods. Exact diagonalization still scales exponentially with system size, so EDinPy does not remove the usual Hilbert-space limits of the method.

Documentation: [quantumartificer.github.io/edinpy](https://quantumartificer.github.io/edinpy/)

## Installation

EDinPy is currently installed from source:

```bash
git clone https://github.com/QuantumArtificer/edinpy.git
cd edinpy
python -m pip install -e .
```

For development, install the test, documentation, and release tools as well:

```bash
python -m pip install -e ".[test,docs,dev]"
```

EDinPy requires Python 3.10 or newer, NumPy, and SciPy.

## A first fermionic calculation

The two-site Hubbard model is small enough to inspect directly and already contains the main EDinPy workflow.

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

double_occupancy = sum(
    (n(i, UP) * n(i, DOWN) for i in range(L)),
    start=0,
)

print(energies)
print(psi0.dag * double_occupancy * psi0)
```

The symbolic expression remains readable throughout the calculation. For example,

```python
-t * (cd(i, sigma) * c(j, sigma) + cd(j, sigma) * c(i, sigma))
```

is an ordinary EDinPy operator expression. Common helpers such as `Hopping`, `Hubbard`, and `HeisenbergExchange` produce the same underlying algebra.

## Design

EDinPy keeps three parts of a fermionic calculation explicit.

1. **Modes and sectors.** `FermionModes` defines the ordering of single-particle labels. `NParticleSector` defines a fixed-$N$ many-body sector and can resolve separately conserved particle populations before the basis is built.
2. **Fock algebra.** `Creation`, `Annihilation`, `Number`, sums, products, and Hermitian conjugation behave as symbolic second-quantized operators.
3. **Numerical representation.** When a Hamiltonian matrix is requested, recognized operator structures are compiled to sparse execution kernels. This keeps the user-facing notation simple without requiring every term to be interpreted state by state.

The implementation includes optimized paths for common number-conserving structures, but performance depends strongly on basis dimension, sparsity, operator structure, and the numerical environment. Use the included [benchmark script](benchmarks/README.md) to measure the package on the problem and machine that matter to you.

## Particle-resolved sectors

When particle number is conserved separately for labels of a degree of freedom, specify those populations before building the basis:

```python
spin = edf.DoF(2, name="spin", labels=("up", "down"))
modes = edf.FermionModes(site, spin)

sector = (
    edf.NParticleSector(modes, N=4)
    .project_particles("spin", up=2, down=2)
    .build()
)
```

Several labeled degrees of freedom can be resolved in the same sector. Their constraints are solved jointly and the requested basis is generated directly rather than by constructing the complete fixed-$N$ basis and filtering it. See the [modes and sectors guide](https://quantumartificer.github.io/edinpy/fermion/user_guide/modes_and_sectors.html) for examples with spin, layer, and orbital labels.

## Documentation

The documentation is organized around the calculation workflow rather than the source-code layout.

- [Getting started](https://quantumartificer.github.io/edinpy/getting_started.html)
- [Fermionic tutorial](https://quantumartificer.github.io/edinpy/fermion/getting_started.html)
- [User guide](https://quantumartificer.github.io/edinpy/fermion/user_guide/index.html)
- [Worked examples](https://quantumartificer.github.io/edinpy/fermion/examples/index.html)
- [Theory and numerical methods](https://quantumartificer.github.io/edinpy/fermion/theory/index.html)
- [API reference](https://quantumartificer.github.io/edinpy/fermion/reference/index.html)
- [Limitations](https://quantumartificer.github.io/edinpy/fermion/reference/limitations.html)
- [Validation and benchmarks](https://quantumartificer.github.io/edinpy/validation/index.html)

The repository also contains executable examples in [`examples/`](examples/).

## Fermions and bosons

EDinPy contains separate `edinpy.fermion` and `edinpy.boson` namespaces because the two Fock algebras have different basis rules and operator identities.

The fermionic interface is the primary documented interface. The bosonic module uses a separate API and currently has less validation and documentation coverage. See the [bosonic documentation](https://quantumartificer.github.io/edinpy/boson/index.html) before starting a bosonic calculation.

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
