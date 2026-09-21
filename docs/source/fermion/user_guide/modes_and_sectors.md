# Modes and particle-number sectors

The occupation-number basis is only unambiguous after the fermionic modes have been ordered. EDinPy makes that ordering an explicit object so that the same convention is shared by symbolic operators, Fock states, Hamiltonians, and eigenvectors.

## Degrees of freedom are labels

`DoF` describes one finite label set. Site, spin, orbital, layer, valley, and discrete momentum labels can all be represented in the same way:

```python
from edinpy import fermion as edf

site = edf.DoF(3, name="site")
spin = edf.DoF(2, name="spin")

print(site)
print(spin)
```

```text
DoF(size=3, name='site')
DoF(size=2, name='spin')
```

A `DoF` is not a wavefunction and is not a Hilbert space. It only specifies the allowed integer labels.

## Constructing the ordered mode set

```python
modes = edf.FermionModes(site, spin)

print("number of modes:", modes.n_modes)
for p in range(modes.n_modes):
    print(p, modes.unravel(p))
```

```text
number of modes: 6
0 (0, 0)
1 (1, 0)
2 (2, 0)
3 (0, 1)
4 (1, 1)
5 (2, 1)
```

The first supplied degree of freedom varies fastest. For sizes $(d_0,d_1,\ldots)$, the mixed-radix map begins

$$
p=i_0+d_0 i_1+d_0d_1 i_2+\cdots.
$$

Use `resolve()` and `unravel()` to apply this mapping consistently:

```python
print(modes.resolve((2, 1)))
print(modes.unravel(4))
```

```text
5
(1, 1)
```

The ordering matters because the canonical fermionic sign for an operator acting on mode $p$ counts occupied modes with index smaller than $p$.

## Fixed-particle-number sectors

A conserved total particle number reduces the full Fock space to

$$
\mathcal H_N=\operatorname{span}\{|n_0n_1\ldots n_{M-1}\rangle:\sum_p n_p=N\},
$$

with dimension

$$
D=\binom{M}{N}.
$$

For the six modes above and three particles:

```python
sector = edf.NParticleSector(modes, N=3)
print("dimension:", sector.dimension)
```

```text
dimension: 20
```

The basis itself is available through `sector.basis`:

```python
for i in range(5):
    print(i, sector.basis.state(i))
```

```text
0 1 |000111>
1 1 |001011>
2 1 |001101>
3 1 |001110>
4 1 |010011>
```

`FockState` prints the conventional binary representation with the most significant bit on the left. Mode 0 is therefore the rightmost bit. Internally the state is stored as the corresponding non-negative Python integer.

## Why fixed-$N$ sectors help

For a spin-$1/2$ chain with $L$ sites there are $M=2L$ fermionic modes. The full Fock space has dimension $4^L$, while the half-filled sector contains only

$$
\binom{2L}{L}
$$

states. Both still grow exponentially, but fixing $N$ removes a large amount of irrelevant Hilbert space when particle number is conserved.

```{figure} ../../_static/figures/hilbert_space_growth.svg
:width: 80%
:alt: Full and half-filled Hilbert-space dimensions for a spinful chain

Growth of the full spinful Fock space and the fixed-$N$ half-filled sector. The reduction is substantial, but exact diagonalization remains exponentially limited.
```

The figure is generated with

```bash
make -C docs figures
```

which writes the plotted dimensions from the same combinatorial definition used by `NParticleSector`.

## Basis-backed many-body kets

`FockState` is one occupation-number basis ket. `FockVector` is a general linear combination in a specific `NParticleSector`. A coefficient vector can be attached to the sector explicitly:

```python
import numpy as np

coefficients = np.zeros(sector.dimension)
coefficients[0] = 1.0
psi = sector.from_vector(coefficients)

print(psi.norm())
print(psi.dag * psi)
```

```text
1.0
1.0
```

The same conversion is performed automatically by `Hamiltonian.eigenstate()` after diagonalization.

## Choosing a sector

Use the smallest implemented sector that matches the conserved quantities of the Hamiltonian. EDinPy 0.2.0 fixes total particle number only. A spin-conserving Hamiltonian may decompose further into fixed $N_\uparrow$ and $N_\downarrow$ blocks, but those additional symmetry sectors are not yet used to reduce the basis automatically.

That limitation affects performance, not the literal operator algebra: spin-resolved operators and observables are still valid in the larger fixed-$N$ sector.
