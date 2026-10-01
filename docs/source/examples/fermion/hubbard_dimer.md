# Hubbard dimer: charge fluctuations and superexchange

The half-filled Hubbard dimer is the smallest interacting fermion problem that contains both charge fluctuations and antiferromagnetic superexchange. It appears as the elementary bond problem behind strong-coupling expansions of the Hubbard model and provides a compact analytic benchmark for an exact-diagonalization calculation.[^carrascal]

The Hamiltonian is

$$
H=-t\sum_{\sigma=\uparrow,\downarrow}
\left(c_{0\sigma}^\dagger c_{1\sigma}+c_{1\sigma}^\dagger c_{0\sigma}\right)
+U\sum_{i=0}^{1}n_{i\uparrow}n_{i\downarrow},
$$

with two fermions in four spin-orbitals. The fixed-particle-number sector therefore has dimension

$$
\dim\mathcal H_{N=2}=\binom{4}{2}=6.
$$

## Hamiltonian and eigensystem

```python
from edinpy import fermion as edf

UP, DOWN = 0, 1
t = 1.0
U = 4.0

site = edf.DoF(2, name="site")
spin = edf.DoF(2, name="spin")
modes = edf.FermionModes(site, spin)
sector = edf.NParticleSector(modes, N=2).build()

c = edf.set_notation(edf.Annihilation, modes)
cd = edf.set_notation(edf.Creation, modes)
n = edf.set_notation(edf.Number, modes)

H = 0
for sigma in (UP, DOWN):
    hop = cd(0, sigma) * c(1, sigma)
    H += -t * (hop + hop.dag)
for i in range(2):
    H += U * n(i, UP) * n(i, DOWN)

hamiltonian = edf.Hamiltonian(H, sector)
energies, _ = hamiltonian.eigsolve(k=None)
psi0 = hamiltonian.eigenstate(0)
```

The exact ground-state energy is

$$
E_0=\frac{U-\sqrt{U^2+16t^2}}{2}.
$$

The low-energy singlet separates continuously from the charge-like singlet branches as repulsion grows.

```{figure} ../../_static/figures/hubbard_dimer_spectrum.svg
:width: 78%
:alt: Complete half-filled Hubbard-dimer spectrum as a function of U over t

Complete $N=2$ spectrum of the symmetric Hubbard dimer. The three triplet states remain degenerate at zero energy in this convention, while the correlated singlet ground state approaches the triplet manifold at strong coupling.
```

## Ground-state amplitudes

The ground state can be inspected directly in the Fock basis through `psi0.coefficients`. At $U/t=4$ the largest weights lie in the two singly occupied configurations with opposite spins, while the doublon-holon configurations retain smaller finite amplitudes.

```{figure} ../../_static/figures/hubbard_dimer_wavefunction.svg
:width: 78%
:alt: Fock-state probabilities of the Hubbard-dimer ground state at U over t equals four

Ground-state Fock probabilities at $U/t=4$. The singly occupied configurations dominate, while finite hopping keeps a nonzero doublon-holon admixture.
```

## Double occupancy

The total doublon operator is

$$
D=n_{0\uparrow}n_{0\downarrow}+n_{1\uparrow}n_{1\downarrow}.
$$

Its expectation value follows directly from the symbolic state algebra,

```python
D = n(0, UP) * n(0, DOWN) + n(1, UP) * n(1, DOWN)
double_occupancy = psi0.dag * D * psi0
```

and the exact result is

$$
\langle D\rangle
=\frac12\left(1-\frac{U}{\sqrt{U^2+16t^2}}\right).
$$

```{figure} ../../_static/figures/example_hubbard_dimer_double_occupancy.svg
:width: 76%
:alt: Hubbard-dimer double occupancy versus U over t

Total double occupancy of the half-filled dimer. Repulsion progressively removes doublon-holon weight from the ground state; exact diagonalization follows the analytic result throughout the crossover.
```

## Spin correlations

The rotationally invariant bond correlation is

$$
C_S=\left\langle\mathbf S_0\cdot\mathbf S_1\right\rangle.
$$

The operator is available as the exchange bilinear with unit coupling,

```python
Sdot = edf.HeisenbergExchange(
    (0, UP), (0, DOWN),
    (1, UP), (1, DOWN),
    J=1.0,
    modes=modes,
)
spin_correlation = psi0.dag * Sdot * psi0
```

For the symmetric dimer,

$$
C_S=-\frac38\left(1+\frac{U}{\sqrt{U^2+16t^2}}\right).
$$

```{figure} ../../_static/figures/example_hubbard_dimer_spin_correlation.svg
:width: 76%
:alt: Hubbard-dimer spin correlation versus U over t

Bond spin correlation of the half-filled dimer. Suppressing charge fluctuations drives the ground state toward a pure two-spin singlet with $\langle\mathbf S_0\cdot\mathbf S_1\rangle=-3/4$.
```

## Superexchange scale

The triplet energy is $E_T=0$ in the present convention, giving

$$
\Delta_{ST}=E_T-E_0
=\frac{\sqrt{U^2+16t^2}-U}{2}.
$$

At strong coupling,

$$
\Delta_{ST}=\frac{4t^2}{U}+\mathcal O\!\left(\frac{t^4}{U^3}\right).
$$

```{figure} ../../_static/figures/example_hubbard_dimer_gap.svg
:width: 76%
:alt: Hubbard-dimer singlet triplet gap and strong-coupling superexchange scale

Singlet-triplet gap of the Hubbard dimer. The exact fermionic spectrum approaches the antiferromagnetic superexchange scale $4t^2/U$ as charge excitations are pushed to high energy.
```

[^carrascal]: D. J. Carrascal, J. Ferrer, J. C. Smith, and K. Burke, "The Hubbard dimer: a density functional case study of a many-body problem," J. Phys.: Condens. Matter 27, 393001 (2015), [doi:10.1088/0953-8984/27/39/393001](https://doi.org/10.1088/0953-8984/27/39/393001).
