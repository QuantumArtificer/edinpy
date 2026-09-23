# Modes and particle-number sectors

The occupation-number basis is only unambiguous after the fermionic modes have been ordered. EDinPy makes that ordering an explicit object so that the same convention is shared by symbolic operators, Fock states, Hamiltonians, and eigenvectors.

## Degrees of freedom are labels

`DoF` describes one finite label set. Site, spin, orbital, layer, valley, and discrete momentum labels can all be represented in the same way:

```python
from edinpy import fermion as edf

site = edf.DoF(3, name="site")
spin = edf.DoF(2, name="spin", labels=("up", "down"))

print(site)
print(spin)
```

```text
DoF(size=3, name='site', labels=None)
DoF(size=2, name='spin', labels=('up', 'down'))
```

A `DoF` is not a wavefunction and is not a Hilbert space. Integer indices remain the canonical mode coordinates. Optional `labels` give those values readable names that can also be used when defining particle-number projections.

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
sector = edf.NParticleSector(modes, N=3).build()
print("dimension:", sector.dimension)
```

`NParticleSector(...)` records the sector specification. The explicit `build()`
step generates the many-body basis. Separating specification from generation
allows additional sector restrictions to be supplied before basis enumeration.

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

## Resolving particle number by degree of freedom

If the Hamiltonian conserves particle number separately for values of a degree of freedom, those populations can be fixed before the basis is generated. For example, a spin-conserving calculation can fix $N_\uparrow$ and $N_\downarrow$ directly:

```python
sector = (
    edf.NParticleSector(modes, N=3)
    .project_particles("spin", up=2, down=1)
    .build()
)

print("dimension:", sector.dimension)
```

```text
dimension: 9
```

The projection is part of the sector specification. `build()` generates only states with the requested populations. For three sites, the dimension is therefore

$$
\binom{3}{2}\binom{3}{1}=9,
$$

rather than $\binom{6}{3}=20$. The complete fixed-$N$ basis is not generated and filtered.

The projection syntax uses labels already attached to the `DoF`. EDinPy does not assign physical meaning to names such as `spin`, `layer`, or `orbital`. A projection may also leave labels unspecified. If an orbital degree of freedom has labels `("a", "b", "c")` and only `a=2` is supplied, the remaining particles are free to occupy `b` and `c`.

Two degrees of freedom can be resolved at the same time. Their projections are solved jointly because the corresponding mode groups overlap. For example:

```python
site = edf.DoF(4, name="site")
spin = edf.DoF(2, name="spin", labels=("up", "down"))
layer = edf.DoF(2, name="layer", labels=("top", "bottom"))
modes = edf.FermionModes(site, spin, layer)

sector = (
    edf.NParticleSector(modes, N=4)
    .project_particles("spin", up=2, down=2)
    .project_particles("layer", top=2, bottom=2)
    .build()
)

print("dimension:", sector.dimension)
```

```text
dimension: 328
```

The builder partitions the modes into the intersections `(up, top)`, `(up, bottom)`, `(down, top)`, and `(down, bottom)`. It first finds the allowed particle counts in these four groups, then generates only the Fock states associated with those counts. The spin projection is therefore not generated first and filtered by the layer projection afterward.

Three or more degrees of freedom can be projected in the same sector. For example, an additional orbital population can be imposed before the basis is built:

```python
orbital = edf.DoF(2, name="orbital", labels=("a", "b"))
modes = edf.FermionModes(site, spin, layer, orbital)

sector = (
    edf.NParticleSector(modes, N=4)
    .project_particles("spin", up=2, down=2)
    .project_particles("layer", top=2, bottom=2)
    .project_particles("orbital", a=2, b=2)
    .build()
)
```

For one projected degree of freedom, EDinPy uses a direct product of fixed-population mode groups. For two projected degrees of freedom, it uses a bounded row-and-column occupation solver. For three or more, it forms the joint intersections of all projected labels and solves the remaining bounded occupation constraints recursively. These are separate generation paths selected by `build()`; the public projection syntax is unchanged.

In every case, the requested sector is generated directly. EDinPy does not construct the complete fixed-$N$ basis and then apply the projections as filters.

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

Use a projected sector only when the Hamiltonian preserves the corresponding particle populations. Fixing $N_\uparrow$ and $N_\downarrow$, for example, is appropriate when those two numbers are separately conserved. The projection reduces the basis used for matrix construction and diagonalization; it does not change the literal operator algebra.
