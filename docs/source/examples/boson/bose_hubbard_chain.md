# Bose-Hubbard ring: coherence and interaction-driven localization

The Bose-Hubbard model describes lattice bosons in optical lattices and related interacting bosonic systems.[^fisher][^jaksch] In one dimension,

$$
H=-J\sum_i\left(b_i^\dagger b_{i+1}+b_{i+1}^\dagger b_i\right)
+\frac{U}{2}\sum_i n_i(n_i-1).
$$

At integer filling the thermodynamic model supports superfluid and Mott-insulating regimes. A six-site unit-filled ring is too small to define a phase boundary, but it resolves the suppression of number fluctuations and long-range one-body coherence as $U/J$ grows.[^zhang]

The calculation uses $L=N=6$, for which

$$
\dim\mathcal H_N=\binom{N+L-1}{N}=462.
$$

## Model

```python
import numpy as np
from edinpy import boson as edb

L = N = 6
J = 1.0
U = 8.0

modes = edb.BosonModes(edb.DoF(L, name="site"))
sector = edb.NParticleSector(modes, N=N).build()
b = edb.set_notation(edb.Annihilation, modes)
bd = edb.set_notation(edb.Creation, modes)
n = edb.set_notation(edb.Number, modes)

H = 0
for i in range(L):
    H += edb.Hopping(i, (i + 1) % L, -J, modes)
    H += edb.Hubbard(i, U, modes)

hamiltonian = edb.Hamiltonian(H, sector)
hamiltonian.eigsolve(k=1, which="SA")
psi0 = hamiltonian.eigenstate(0)
```

## One-body density matrix and natural occupations

The one-body density matrix is

$$
\rho^{(1)}_{ij}=\langle b_i^\dagger b_j\rangle.
$$

Its eigenvalues $n_\alpha$ are the natural occupations,

$$
\sum_j\rho^{(1)}_{ij}\phi_j^{(\alpha)}
=n_\alpha\phi_i^{(\alpha)}.
$$

The largest occupation $n_0$ defines the finite-system Penrose-Onsager condensate fraction $n_0/N$.[^penrose]

```python
def one_body_density_matrix(psi):
    return np.asarray(
        [
            [psi.dag * bd(i) * b(j) * psi for j in range(L)]
            for i in range(L)
        ],
        dtype=complex,
    )

rho = one_body_density_matrix(psi0)
natural_occupations = np.linalg.eigvalsh(rho)[::-1]
condensate_fraction = natural_occupations[0] / N
```

```{figure} ../../_static/figures/example_bose_ring_condensate_fraction.svg
:width: 76%
:alt: Finite-system condensate fraction of the six-site Bose-Hubbard ring versus U over J

Largest natural occupation divided by particle number. The dominant one-body mode loses weight as repulsive interactions localize particles in the site basis.
```

## Onsite number fluctuations

At unit filling the local number variance is

$$
\Delta n^2
=\frac1L\sum_i
\left(\langle n_i^2\rangle-\langle n_i\rangle^2\right).
$$

```python
def number_variance(psi):
    values = []
    for i in range(L):
        mean = np.real(psi.dag * n(i) * psi)
        mean2 = np.real(psi.dag * n(i) * n(i) * psi)
        values.append(mean2 - mean**2)
    return np.mean(values)
```

```{figure} ../../_static/figures/example_bose_ring_number_variance.svg
:width: 76%
:alt: Onsite number variance of the six-site Bose-Hubbard ring versus U over J

Site-averaged number variance. The reduction of $\Delta n^2$ is the finite-cluster signature of interaction-driven number localization at commensurate filling.
```

## Real-space one-body coherence

The translationally averaged first-order correlation function is

$$
g^{(1)}(r)=\frac1L\sum_i\langle b_i^\dagger b_{i+r}\rangle.
$$

```python
def g1(psi, r):
    return np.real(
        sum(
            psi.dag * bd(i) * b((i + r) % L) * psi
            for i in range(L)
        ) / L
    )
```

```{figure} ../../_static/figures/example_bose_ring_g1.svg
:width: 76%
:alt: First-order coherence of the six-site Bose-Hubbard ring as a function of separation

First-order coherence at three interaction strengths. Long-range coherence is strongest in the weakly interacting ring and is progressively suppressed as onsite repulsion increases.
```

## Density structure factor

With filling $\bar n=N/L$, define

$$
\delta n_q=\sum_j e^{-iqj}(n_j-\bar n),
$$

and

$$
S_n(q)=\frac1L\langle\delta n_q^\dagger\delta n_q\rangle.
$$

```python
def density_structure_factor(psi, q):
    delta_n_q = sum(
        (
            np.exp(-1j * q * j) * (n(j) - N / L)
            for j in range(L)
        ),
        start=0,
    )
    return np.real(
        psi.dag * delta_n_q.dag * delta_n_q * psi
    ) / L
```

```{figure} ../../_static/figures/example_bose_ring_density_structure.svg
:width: 76%
:alt: Density structure factor of the six-site Bose-Hubbard ring at three interaction strengths

Connected density structure factor. Repulsion redistributes density fluctuations over the discrete momenta while the overall fluctuation scale is suppressed toward the strongly interacting unit-filled limit.
```

The one-body density matrix, $g^{(1)}(r)$, and $S_n(q)$ are built directly from creation, annihilation, and number operators, so their definitions remain identical throughout the interaction sweep.

[^penrose]: O. Penrose and L. Onsager, "Bose-Einstein Condensation and Liquid Helium," Phys. Rev. 104, 576-584 (1956), [doi:10.1103/PhysRev.104.576](https://doi.org/10.1103/PhysRev.104.576).
[^fisher]: M. P. A. Fisher, P. B. Weichman, G. Grinstein, and D. S. Fisher, "Boson localization and the superfluid-insulator transition," Phys. Rev. B 40, 546 (1989), [doi:10.1103/PhysRevB.40.546](https://doi.org/10.1103/PhysRevB.40.546).
[^jaksch]: D. Jaksch, C. Bruder, J. I. Cirac, C. W. Gardiner, and P. Zoller, "Cold Bosonic Atoms in Optical Lattices," Phys. Rev. Lett. 81, 3108 (1998), [doi:10.1103/PhysRevLett.81.3108](https://doi.org/10.1103/PhysRevLett.81.3108).
[^zhang]: J. M. Zhang and R. X. Dong, "Exact diagonalization: the Bose-Hubbard model as an example," Eur. J. Phys. 31, 591-602 (2010), [doi:10.1088/0143-0807/31/3/016](https://doi.org/10.1088/0143-0807/31/3/016).
