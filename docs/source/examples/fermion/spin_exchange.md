# Heisenberg dimer in a constrained fermionic Hilbert space

Exchange Hamiltonians describe the low-energy spin dynamics of Mott insulators, quantum magnets, and Hubbard systems after high-energy charge fluctuations are integrated out. For two spin-$1/2$ sites,

$$
H=J\,\mathbf S_0\cdot\mathbf S_1.
$$

The local spin operators are represented by fermionic bilinears,

$$
S_i^+=c_{i\uparrow}^\dagger c_{i\downarrow},\qquad
S_i^-=c_{i\downarrow}^\dagger c_{i\uparrow},\qquad
S_i^z=\frac12(n_{i\uparrow}-n_{i\downarrow}).
$$

A pure two-spin problem occupies the subspace with exactly one fermion on each site. The local-occupancy constraint reduces the $N=2$ Fock sector from six states to four.

```python
from edinpy import fermion as edf

UP, DOWN = 0, 1

site = edf.DoF(2, name="site", labels=("left", "right"))
spin = edf.DoF(2, name="spin", labels=("up", "down"))
modes = edf.FermionModes(site, spin)

sector = (
    edf.NParticleSector(modes, N=2)
    .project_particles("site", left=1, right=1)
    .build()
)

Sdot = edf.HeisenbergExchange(
    (0, UP), (0, DOWN),
    (1, UP), (1, DOWN),
    J=1.0,
    modes=modes,
)
```

The analytic energies follow from

$$
\mathbf S_0\cdot\mathbf S_1
=\frac12\left(\mathbf S_{\mathrm{tot}}^2-\mathbf S_0^2-\mathbf S_1^2\right),
$$

so that

$$
E_{S=0}=-\frac{3J}{4},\qquad
E_{S=1}=\frac{J}{4}.
$$

```{figure} ../../_static/figures/example_fermion_exchange_spectrum.svg
:width: 76%
:alt: Singlet and triplet energies of the constrained fermionic Heisenberg dimer

Spectrum of the two-spin Heisenberg dimer. Antiferromagnetic $J>0$ selects the singlet, while ferromagnetic $J<0$ selects the triplet manifold.
```

## Total-spin diagnostic

The total-spin operator can be constructed literally from the summed raising, lowering, and $z$ components,

$$
\mathbf S_{\mathrm{tot}}^2
=(S_{\mathrm{tot}}^z)^2
+\frac12\left(S_{\mathrm{tot}}^+S_{\mathrm{tot}}^-+S_{\mathrm{tot}}^-S_{\mathrm{tot}}^+\right).
$$

```python
Splus = sum(
    (edf.SpinPlus((i, UP), (i, DOWN), modes) for i in range(2)),
    start=0,
)
Sminus = sum(
    (edf.SpinMinus((i, UP), (i, DOWN), modes) for i in range(2)),
    start=0,
)
Sz = sum(
    (edf.SpinZ((i, UP), (i, DOWN), modes) for i in range(2)),
    start=0,
)
S2 = Sz * Sz + 0.5 * (Splus * Sminus + Sminus * Splus)
```

```{figure} ../../_static/figures/example_fermion_exchange_total_spin.svg
:width: 72%
:alt: Total spin squared for the four eigenstates of the fermionic Heisenberg dimer

Total-spin quantum number at $J=1$. The singlet has $\langle S_{\mathrm{tot}}^2\rangle=0$ and the three triplets have $\langle S_{\mathrm{tot}}^2\rangle=2$.
```

The occupancy constraint removes the doublon-holon states and leaves the four-dimensional effective spin Hilbert space.
