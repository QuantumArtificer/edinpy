# Building Hamiltonians

A Hamiltonian in EDinPy is an ordinary symbolic operator expression associated with a fixed-particle-number sector. Research models can be written term by term, while common helpers return the same literal algebra for standard interactions.

The examples below use a two-site spinful mode set:

```python
from edinpy import fermion as edf

site = edf.DoF(2, name="site")
spin = edf.DoF(2, name="spin")
modes = edf.FermionModes(site, spin)
sector = edf.NParticleSector(modes, N=2).build()

c = edf.set_notation(edf.Annihilation, modes)
cd = edf.set_notation(edf.Creation, modes)
n = edf.set_notation(edf.Number, modes)

UP, DOWN = 0, 1
```

## Write a term exactly as it appears in the Hamiltonian

For one spin-up hopping process,

```python
term = cd(0, UP) * c(1, UP)
print(term)
print(term.dag)
```

```text
c†[0,0] c[1,0]
c†[1,0] c[0,0]
```

A Hermitian hopping bond is therefore

```python
bond = -(term + term.dag)
print(bond)
```

```text
-1 c†[0,0] c[1,0] + -1 c†[1,0] c[0,0]
```

EDinPy does not silently add a Hermitian conjugate. If the physical model contains both directions, both directions must be present in the expression.

## Common operators remain transparent

The same bond can be built with `Hopping`:

```python
bond_helper = edf.Hopping(
    (0, UP),
    (1, UP),
    -1,
    modes,
)
print(bond_helper)
```

```text
-1 c†[0,0] c[1,0] + -1 c†[1,0] c[0,0]
```

Likewise, the local interaction

```python
local_U = edf.Hubbard((0, UP), (0, DOWN), 4, modes)
print(local_U)
```

```text
4 n[0,0] n[0,1]
```

is the same primitive number-operator product one would write by hand. The helper provides a standard convention, not a hidden model object.

## Assemble a complete model

```python
t = 1.0
U = 4.0
H = 0

for sigma in (UP, DOWN):
    H += edf.Hopping(
        (0, sigma),
        (1, sigma),
        -t,
        modes,
    )

for i in range(2):
    H += edf.Hubbard(
        (i, UP),
        (i, DOWN),
        U,
        modes,
    )

print(H)
```

```text
-1.0 c†[0,0] c[1,0] + -1.0 c†[1,0] c[0,0] + -1.0 c†[0,1] c[1,1] + -1.0 c†[1,1] c[0,1] + 4.0 n[0,0] n[0,1] + 4.0 n[1,0] n[1,1]
```

The same expression can freely mix helpers and handwritten algebra.

## Associate the algebra with a sector

```python
hamiltonian = edf.Hamiltonian(H, sector)

print(hamiltonian.matrix.shape)
print(hamiltonian.matrix.dtype)
print(hamiltonian.is_hermitian())
```

```text
(6, 6)
float64
True
```

Construction checks that the operator expression and sector share the same `FermionModes` object. This prevents two equally sized but differently ordered mode spaces from being mixed silently.

## Complex hopping phases

Complex coefficients are ordinary scalars in the symbolic algebra:

```python
import numpy as np

phase = 0.3
forward = np.exp(1j * phase) * cd(0, UP) * c(1, UP)
complex_bond = -(forward + forward.dag)

print(complex_bond)
```

```text
(-0.955336489125606-0.29552020666133955j) c†[0,0] c[1,0] + (-0.955336489125606+0.29552020666133955j) c†[1,0] c[0,0]
```

A Hamiltonian containing genuinely complex matrix elements uses `complex128`. Real Hamiltonians use `float64`.

## What sparsity looks like

A four-site half-filled Hubbard chain has a 70-dimensional fixed-$N$ basis at $L=4$. At $U/t=4$, the matrix contains 294 nonzero entries, compared with $70^2=4900$ entries in dense storage.

```{figure} ../../_static/figures/hubbard_matrix_sparsity.svg
:width: 65%
:alt: Sparse matrix pattern of a four-site Hubbard Hamiltonian

Sparse structure of the four-site open Hubbard Hamiltonian at half filling and $U/t=4$. Hopping produces off-diagonal connections. Density interactions contribute diagonal entries.
```

Sparse construction matters because exact diagonalization quickly reaches Hilbert spaces where storing a dense Hamiltonian is unnecessary or impossible even though a few extremal eigenpairs remain accessible.

## Public common-operator library

The current fermionic library includes `Hopping`, `Onsite`, `DensityDensity`, `Hubbard`, `SpinPlus`, `SpinMinus`, `SpinX`, `SpinY`, `SpinZ`, `HeisenbergExchange`, and `PairHopping`. Their exact signatures and sign conventions are documented in {doc}`../reference/operators`.
