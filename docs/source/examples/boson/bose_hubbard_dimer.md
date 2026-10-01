# Bose-Hubbard dimer: number squeezing in a bosonic Josephson junction

The two-mode Bose-Hubbard model describes interacting bosons in a double well and provides the minimal quantum model of a bosonic Josephson junction.[^milburn][^smerzi][^raghavan] At fixed total particle number,

$$
H=-J\left(b_L^\dagger b_R+b_R^\dagger b_L\right)
+\frac{U}{2}\sum_{\alpha=L,R}n_\alpha(n_\alpha-1).
$$

The competition is between tunneling, which favors phase coherence across the wells, and repulsion, which suppresses number imbalance.

For $N=8$, the fixed-$N$ Hilbert space contains $N+1=9$ Fock states.

## Schwinger representation

Define

$$
S_x=\frac12\left(b_L^\dagger b_R+b_R^\dagger b_L\right),
\qquad
S_z=\frac12(n_L-n_R).
$$

Up to an $N$-dependent constant,

$$
H=-2J S_x+U S_z^2.
$$

```python
from edinpy import boson as edb

N = 8
J = 1.0
U = 4.0

well = edb.DoF(2, name="well", labels=("left", "right"))
modes = edb.BosonModes(well)
sector = edb.NParticleSector(modes, N=N).build()

H = (
    edb.Hopping(0, 1, -J, modes)
    + edb.Hubbard(0, U, modes)
    + edb.Hubbard(1, U, modes)
)

Sx = edb.SpinX(0, 1, modes)
Sz = edb.SpinZ(0, 1, modes)
```

## Low-energy spectrum

```{figure} ../../_static/figures/example_bose_dimer_spectrum.svg
:width: 76%
:alt: Lowest excitation energies of the Bose-Hubbard dimer versus U over J

Lowest excitation energies of the $N=8$ Bose-Hubbard dimer. Repulsion changes the level spacing as fluctuations in the occupation imbalance become increasingly costly.
```

## First-order coherence

The normalized transverse coherence is

$$
\mathcal C=\frac{\langle S_x\rangle}{N/2}.
$$

At $U=0$ all particles occupy the bonding orbital and $\mathcal C=1$.

```python
hamiltonian = edb.Hamiltonian(H, sector)
hamiltonian.eigsolve(k=None)
psi0 = hamiltonian.eigenstate(0)
coherence = np.real(psi0.dag * Sx * psi0) / (N / 2)
```

```{figure} ../../_static/figures/example_bose_dimer_coherence.svg
:width: 76%
:alt: Interwell coherence of the Bose-Hubbard dimer versus U over J

Normalized interwell coherence. Repulsive interactions progressively reduce the bonding-orbital coherence of the ground state.
```

## Number squeezing

The normalized number-imbalance fluctuation is

$$
\mathcal V_z=\frac{\langle S_z^2\rangle}{(N/2)^2}.
$$

```python
imbalance = np.real(psi0.dag * Sz * Sz * psi0) / (N / 2) ** 2
```

```{figure} ../../_static/figures/example_bose_dimer_imbalance.svg
:width: 76%
:alt: Number-imbalance fluctuations of the Bose-Hubbard dimer versus U over J

Normalized imbalance fluctuations. Increasing $U/J$ squeezes the relative particle number between the two wells.
```

## Fock-space redistribution

The Fock basis is

$$
|n_L,N-n_L\rangle,
\qquad n_L=0,1,\ldots,N,
$$

with probabilities

$$
P(n_L)=|\langle n_L,N-n_L|\psi_0\rangle|^2.
$$

```{figure} ../../_static/figures/example_bose_dimer_probabilities.svg
:width: 76%
:alt: Ground-state occupation probabilities of the Bose-Hubbard dimer for three U over J values

Ground-state Fock probabilities. The broad binomial-like distribution at weak coupling narrows around balanced occupation as repulsion suppresses number fluctuations.
```

[^milburn]: G. J. Milburn, J. Corney, E. M. Wright, and D. F. Walls, "Quantum dynamics of an atomic Bose-Einstein condensate in a double-well potential," Phys. Rev. A 55, 4318 (1997), [doi:10.1103/PhysRevA.55.4318](https://doi.org/10.1103/PhysRevA.55.4318).
[^smerzi]: A. Smerzi, S. Fantoni, S. Giovanazzi, and S. R. Shenoy, "Quantum Coherent Atomic Tunneling between Two Trapped Bose-Einstein Condensates," Phys. Rev. Lett. 79, 4950 (1997), [doi:10.1103/PhysRevLett.79.4950](https://doi.org/10.1103/PhysRevLett.79.4950).
[^raghavan]: S. Raghavan, A. Smerzi, S. Fantoni, and S. R. Shenoy, "Coherent oscillations between two weakly coupled Bose-Einstein condensates: Josephson effects, pi oscillations, and macroscopic quantum self-trapping," Phys. Rev. A 59, 620 (1999), [doi:10.1103/PhysRevA.59.620](https://doi.org/10.1103/PhysRevA.59.620).
