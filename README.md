# EDinPy

EDinPy is a Python package for exact diagonalization of discrete finite many-body systems. The fermionic API is organized around explicit fermionic modes, fixed-particle-number sectors, literal second-quantized operator algebra, structural compilation, sparse Hamiltonian construction, and Hermitian eigensolution.

## Design

The fermionic interface keeps the physical algebra explicit. User expressions such as

```python
-t * (cd(i, s) * c(j, s) + cd(j, s) * c(i, s))
```

are symbolic Fock-algebra expressions. The compiler recognizes their algebraic structure and lowers supported terms to optimized sparse execution kernels. Common operators such as `Hopping` and `Hubbard` expand to the same literal algebra and therefore use the same compiler paths as handwritten expressions.

EDinPy does not impose symbols for creation, annihilation, or number operators. `set_notation` associates a primitive operator class with a `FermionModes` object; the Python variable name remains the user's notation.

## Installation

```bash
python -m pip install edinpy
```

For development:

```bash
python -m pip install -e ".[test,docs]"
```

## Extended Hubbard chain

The spinful one-dimensional extended Hubbard Hamiltonian

\[
H=-t\sum_{\langle i,j\rangle,\sigma}
(c^\dagger_{i\sigma}c_{j\sigma}+\mathrm{H.c.})
+U\sum_i n_{i\uparrow}n_{i\downarrow}
+V\sum_{\langle i,j\rangle} n_i n_j
\]

can be constructed directly:

```python
from edinpy import fermion as edf

L = 8
N = 8
t = 1.0
U = 4.0
V = 1.5

site = edf.DoF(L, name="site")
spin = edf.DoF(2, name="spin")

modes = edf.FermionModes(site, spin)
sector = edf.NParticleSector(modes, N=N)

c = edf.set_notation(edf.Annihilation, modes)
cd = edf.set_notation(edf.Creation, modes)
n = edf.set_notation(edf.Number, modes)

UP, DOWN = 0, 1
H = 0

for i in range(L - 1):
    j = i + 1
    for sigma in (UP, DOWN):
        H += -t * (
            cd(i, sigma) * c(j, sigma)
            + cd(j, sigma) * c(i, sigma)
        )

for i in range(L):
    H += U * n(i, UP) * n(i, DOWN)

for i in range(L - 1):
    j = i + 1
    n_i = n(i, UP) + n(i, DOWN)
    n_j = n(j, UP) + n(j, DOWN)
    H += V * n_i * n_j

hamiltonian = edf.Hamiltonian(H, sector)
eigenvalues, eigenvectors = hamiltonian.eigsolve(k=4, which="SA")
```

## Fermionic public API

Core objects:

- `DoF`: discrete degree-of-freedom descriptor.
- `FermionModes`: ordered fermionic modes generated from arbitrary discrete degrees of freedom.
- `NParticleSector`: fixed-`N` sector of fermionic Fock space.
- `FockBasis` and `FockState`: occupation-number basis and states.
- `Annihilation`, `Creation`, `Number`: primitive second-quantized operators.
- `OperatorSum`, `OperatorProduct`: literal symbolic expressions.
- `set_notation`: user-selected notation bound to a `FermionModes` object.
- `Hamiltonian`: sparse matrix construction and Hermitian eigensolution.

Transparent common operators:

- `Hopping`
- `Onsite`
- `DensityDensity`
- `Hubbard`
- `SpinPlus`, `SpinMinus`, `SpinX`, `SpinY`, `SpinZ`
- `HeisenbergExchange`
- `PairHopping`

## Numerical behavior

Real Hamiltonians are stored in `float64`; genuinely complex Hamiltonians are stored in `complex128`. Sparse low-energy eigensolution uses SciPy's ARPACK interface. Dense partial Hermitian eigensolution uses `scipy.linalg.eigh` with index subsets when applicable.

The current fermionic basis is restricted to a fixed particle number. Number-changing operators can act on individual `FockState` objects, but a Hamiltonian constructed in `NParticleSector` contains only matrix elements within that sector. Full Fock-space, fermion-parity, translation, and momentum sectors are not part of version 0.1.0.

## Provenance and references

`PROVENANCE.md` records algorithmic sources, literature references, repositories reviewed for comparison, licensing constraints, and public-code similarity checks. External projects inspected during provenance review are not implementation sources unless explicitly identified as such.

## License

EDinPy is distributed under the MIT License.
