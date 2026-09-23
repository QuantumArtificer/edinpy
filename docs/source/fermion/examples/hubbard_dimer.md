# Hubbard dimer: charge fluctuations, local moments, and superexchange

The half-filled two-site Hubbard model is the smallest fermionic system that already contains the central competition of the Hubbard problem: hopping lowers the kinetic energy by delocalizing the particles, whereas the on-site interaction suppresses double occupation and favors local moments. Because the symmetric dimer is analytically solvable, it is also an unusually transparent exact-diagonalization benchmark.[^carrascal]

The Hamiltonian is

$$
H=-t\sum_{\sigma=\uparrow,\downarrow}
\left(c_{0\sigma}^\dagger c_{1\sigma}+c_{1\sigma}^\dagger c_{0\sigma}\right)
+U\sum_{i=0}^{1}n_{i\uparrow}n_{i\downarrow}.
$$

Here $c_{i\sigma}^\dagger$ and $c_{i\sigma}$ create and annihilate a fermion of spin $\sigma$ on site $i$. The operator $n_{i\sigma}=c_{i\sigma}^\dagger c_{i\sigma}$ measures the corresponding occupation. The hopping amplitude is $t$, and $U$ is the repulsive energy cost of double occupation. The example uses half filling with $N=2$ and takes $t=1$ as the energy unit.

## Construct the literal Hamiltonian

```python
from edinpy import fermion as edf

L = 2
N = 2
UP, DOWN = 0, 1
t = 1.0
U = 4.0

site = edf.DoF(L, name="site")
spin = edf.DoF(2, name="spin")
modes = edf.FermionModes(site, spin)
sector = edf.NParticleSector(modes, N=N).build()

c = edf.set_notation(edf.Annihilation, modes)
cd = edf.set_notation(edf.Creation, modes)
n = edf.set_notation(edf.Number, modes)

H = 0
for sigma in (UP, DOWN):
    hop = cd(0, sigma) * c(1, sigma)
    H += -t * (hop + hop.dag)
for i in range(L):
    H += U * n(i, UP) * n(i, DOWN)

print(H)
```

```text
-1.0 c†[0,0] c[1,0] + -1.0 c†[1,0] c[0,0] + -1.0 c†[0,1] c[1,1] + -1.0 c†[1,1] c[0,1] + 4.0 n[0,0] n[0,1] + 4.0 n[1,0] n[1,1]
```

The printed expression is already the physical Hamiltonian. There is no separate model specification that is later translated into second-quantized form.

## Solve the complete six-dimensional problem

The fixed-$N$ sector contains

$$
\dim\mathcal H_{N=2}=\binom{4}{2}=6
$$

basis states. Since the problem is tiny, the complete eigensystem is the most informative choice.

```python
hamiltonian = edf.Hamiltonian(H, sector)
energies, vectors = hamiltonian.eigsolve(k=None)

print("sector dimension:", sector.dimension)
print("energies:", energies)
```

```text
sector dimension: 6
energies: [-0.82842712  0.          0.          0.          4.          4.82842712]
```

The three states at zero energy form the triplet manifold. The ground state is a correlated singlet, while the two higher singlet states carry stronger charge character.

For the symmetric dimer the exact ground-state energy is

$$
E_0=\frac{U-\sqrt{U^2+16t^2}}{2}.
$$

At $U/t=4$ this gives $E_0/t=-0.82842712$, exactly matching the diagonalization.

## Read the ground state in the Fock basis

An eigenvector becomes physically more useful once its coefficients are attached to the sector basis.

```python
psi0 = hamiltonian.eigenstate(0)

for state, amplitude in zip(sector.basis.states, psi0.coefficients):
    print(f"{state:04b}  {amplitude:+.8f}")
```

```text
0011  +0.00000000
0101  -0.27059805
0110  -0.65328148
1001  -0.65328148
1010  -0.27059805
1100  +0.00000000
```

The two largest weights are the singly occupied configurations with opposite spins on the two sites. The smaller components are the two doublon-holon configurations. The fully spin-polarized basis states have zero ground-state weight because the ground state is a singlet.

```{figure} ../../_static/figures/hubbard_dimer_wavefunction.svg
:width: 82%
:alt: Fock-basis probabilities of the Hubbard-dimer ground state

Ground-state probability distribution in the six-dimensional $N=2$ Fock basis at $U/t=4$. Repulsion suppresses the doublon-holon configurations but does not eliminate them at finite $U/t$.
```

## Double occupancy: the charge-fluctuation diagnostic

The total double-occupancy operator is

$$
D=n_{0\uparrow}n_{0\downarrow}+n_{1\uparrow}n_{1\downarrow}.
$$

In EDinPy the expectation value is written exactly in Dirac form:

```python
D = n(0, UP) * n(0, DOWN) + n(1, UP) * n(1, DOWN)

print(D)
print(psi0.dag * D * psi0)
```

```text
n[0,0] n[0,1] + n[1,0] n[1,1]
0.1464466094067262
```

The Hellmann-Feynman theorem gives the analytic result

$$
\langle D\rangle
=\frac{\partial E_0}{\partial U}
=\frac12\left(1-\frac{U}{\sqrt{U^2+16t^2}}\right).
$$

Thus the probability carried by doubly occupied configurations decreases continuously as $U/t$ grows.

## Local moment formation

A useful local-spin diagnostic is

$$
\mu^2=\frac12\sum_{i=0}^{1}
\left\langle(n_{i\uparrow}-n_{i\downarrow})^2\right\rangle.
$$

For the half-filled symmetric dimer, $\mu^2=1-\langle D\rangle$: every suppression of doublon weight transfers probability into singly occupied local-moment configurations.

```python
mu2 = 0.5 * sum(
    (
        (n(i, UP) - n(i, DOWN))
        * (n(i, UP) - n(i, DOWN))
        for i in range(L)
    ),
    start=0,
)

print(psi0.dag * mu2 * psi0)
```

```text
0.8535533905932736
```

## Spin correlations from the same algebra

The rotationally invariant two-site spin correlation is

$$
\langle\mathbf S_0\cdot\mathbf S_1\rangle.
$$

`HeisenbergExchange(..., J=1)` returns precisely the operator $\mathbf S_0\cdot\mathbf S_1$, so it can be used as an observable even when it is not part of the Hamiltonian.

```python
Sdot = edf.HeisenbergExchange(
    (0, UP), (0, DOWN),
    (1, UP), (1, DOWN),
    J=1.0,
    modes=modes,
)

print(psi0.dag * Sdot * psi0)
```

```text
-0.6401650429449552
```

Only the singly occupied component carries two local spins, and that component is a singlet. Consequently,

$$
\left\langle\mathbf S_0\cdot\mathbf S_1\right\rangle
=-\frac34\mu^2
=-\frac38\left(1+\frac{U}{\sqrt{U^2+16t^2}}\right).
$$

The four observables below summarize the crossover from an itinerant dimer to a local-moment dimer.

```{figure} ../../_static/figures/hubbard_dimer_observables.svg
:width: 100%
:alt: Hubbard-dimer energy, double occupancy, local moment, spin correlation, and singlet-triplet gap

Analytic results and EDinPy calculations for the symmetric half-filled Hubbard dimer. Increasing $U/t$ suppresses charge fluctuations, increases the local moment, drives the spin correlation toward the pure-singlet value $-3/4$, and lowers the singlet-triplet splitting.
```

## Superexchange from the exact gap

The lowest triplet remains at $E_T=0$ in the present energy convention. The singlet-triplet gap is therefore

$$
\Delta_{ST}=E_T-E_0
=\frac{\sqrt{U^2+16t^2}-U}{2}.
$$

At strong coupling,

$$
\Delta_{ST}=\frac{4t^2}{U}+\mathcal O\!\left(\frac{t^4}{U^3}\right),
$$

which is the antiferromagnetic superexchange scale of the low-energy Heisenberg description. The effective coupling is extracted from the exact fermionic eigensystem. It is not an input parameter of the calculation.

The complete executable calculation is `examples/hubbard_dimer.py`.

## References

[^carrascal]: D. J. Carrascal, J. Ferrer, J. C. Smith, and K. Burke, “The Hubbard dimer: a density functional case study of a many-body problem,” *J. Phys.: Condens. Matter* **27**, 393001 (2015), [doi:10.1088/0953-8984/27/39/393001](https://doi.org/10.1088/0953-8984/27/39/393001).
