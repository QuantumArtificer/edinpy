# Extended Hubbard ring: competing spin, charge, and bond correlations

The nearest-neighbor density interaction enriches the half-filled Hubbard chain by allowing several ordering tendencies to compete on comparable energy scales. The one-dimensional extended Hubbard model is

$$
H=-t\sum_{i,\sigma}
\left(c_{i\sigma}^\dagger c_{i+1,\sigma}+\mathrm{H.c.}\right)
+U\sum_i n_{i\uparrow}n_{i\downarrow}
+V\sum_i n_i n_{i+1},
$$

with $n_i=n_{i\uparrow}+n_{i\downarrow}$ and periodic boundary conditions. The parameters $t$, $U$, and $V$ are the hopping amplitude, on-site repulsion, and nearest-neighbor repulsion, respectively. The example uses a six-site half-filled ring with $t=1$ and fixed $U/t=4$ while $V/t$ is varied.

In the thermodynamic model, weak-to-intermediate coupling contains a spin-density-wave (SDW) regime, a charge-density-wave (CDW) regime, and an intervening bond-order-wave (BOW) region near the competition line $U\approx2V$.[^nakamura][^sandvik][^jeckelmann] A six-site ring cannot locate those phase boundaries quantitatively. It can show how finite-size spectral weight shifts among spin, charge, and bond channels.

## Build and print the Hamiltonian

```python
import numpy as np
from edinpy import fermion as edf

L = 6
N = L
UP, DOWN = 0, 1
t = 1.0
U = 4.0
V = 2.25

site = edf.DoF(L, name="site")
spin = edf.DoF(2, name="spin")
modes = edf.FermionModes(site, spin)
sector = edf.NParticleSector(modes, N=N).build()

c = edf.set_notation(edf.Annihilation, modes)
cd = edf.set_notation(edf.Creation, modes)
n = edf.set_notation(edf.Number, modes)

H = 0
bond_ops = []
for i in range(L):
    j = (i + 1) % L
    B_i = 0
    for sigma in (UP, DOWN):
        hop = cd(i, sigma) * c(j, sigma)
        H += -t * (hop + hop.dag)
        B_i += hop + hop.dag
    bond_ops.append(B_i)

for i in range(L):
    H += U * n(i, UP) * n(i, DOWN)

for i in range(L):
    j = (i + 1) % L
    n_i = n(i, UP) + n(i, DOWN)
    n_j = n(j, UP) + n(j, DOWN)
    H += V * n_i * n_j

print(H)
```

```text
-1.0 c†[0,0] c[1,0] + -1.0 c†[1,0] c[0,0] + -1.0 c†[0,1] c[1,1] + -1.0 c†[1,1] c[0,1] + -1.0 c†[1,0] c[2,0] + -1.0 c†[2,0] c[1,0] + -1.0 c†[1,1] c[2,1] + -1.0 c†[2,1] c[1,1] + -1.0 c†[2,0] c[3,0] + -1.0 c†[3,0] c[2,0] + -1.0 c†[2,1] c[3,1] + -1.0 c†[3,1] c[2,1] + -1.0 c†[3,0] c[4,0] + -1.0 c†[4,0] c[3,0] + -1.0 c†[3,1] c[4,1] + -1.0 c†[4,1] c[3,1] + -1.0 c†[4,0] c[5,0] + -1.0 c†[5,0] c[4,0] + -1.0 c†[4,1] c[5,1] + -1.0 c†[5,1] c[4,1] + -1.0 c†[5,0] c[0,0] + -1.0 c†[0,0] c[5,0] + -1.0 c†[5,1] c[0,1] + -1.0 c†[0,1] c[5,1] + 4.0 n[0,0] n[0,1] + 4.0 n[1,0] n[1,1] + 4.0 n[2,0] n[2,1] + 4.0 n[3,0] n[3,1] + 4.0 n[4,0] n[4,1] + 4.0 n[5,0] n[5,1] + 2.25 n[0,0] n[1,0] + 2.25 n[0,0] n[1,1] + 2.25 n[0,1] n[1,0] + 2.25 n[0,1] n[1,1] + 2.25 n[1,0] n[2,0] + 2.25 n[1,0] n[2,1] + 2.25 n[1,1] n[2,0] + 2.25 n[1,1] n[2,1] + 2.25 n[2,0] n[3,0] + 2.25 n[2,0] n[3,1] + 2.25 n[2,1] n[3,0] + 2.25 n[2,1] n[3,1] + 2.25 n[3,0] n[4,0] + 2.25 n[3,0] n[4,1] + 2.25 n[3,1] n[4,0] + 2.25 n[3,1] n[4,1] + 2.25 n[4,0] n[5,0] + 2.25 n[4,0] n[5,1] + 2.25 n[4,1] n[5,0] + 2.25 n[4,1] n[5,1] + 2.25 n[5,0] n[0,0] + 2.25 n[5,0] n[0,1] + 2.25 n[5,1] n[0,0] + 2.25 n[5,1] n[0,1]
```

The explicit output is useful here because the $V n_i n_{i+1}$ term expands into four spin-resolved number products on every bond. The printed algebra confirms that the nearest-neighbor density interaction was assembled as intended.

## Low-energy state at an intermediate coupling

```python
hamiltonian = edf.Hamiltonian(H, sector)
energies, _ = hamiltonian.eigsolve(k=4, which="SA")
psi0 = hamiltonian.eigenstate(0)

print("sector dimension:", sector.dimension)
print("lowest energies:", energies)
```

```text
sector dimension: 924
lowest energies: [7.56414755 8.6482437  9.328496   9.59918525]
```

The absolute ground-state energy is shifted upward by the repulsive nearest-neighbor interaction. The observables below carry the more useful information about how the state is reorganizing.

## Three order diagnostics from one ground state

Define the local spin and connected charge operators

$$
S_i^z=\frac12(n_{i\uparrow}-n_{i\downarrow}),
\qquad
\delta n_i=n_i-1,
$$

and the kinetic bond operator

$$
B_i=\sum_\sigma
\left(c_{i\sigma}^\dagger c_{i+1,\sigma}+\mathrm{H.c.}\right).
$$

Their Fourier components are

$$
O_q=\sum_j e^{-iqj}O_j,
$$

and the calculation uses

$$
S_s(q)=\frac1L\langle S_q^{z\dagger}S_q^z\rangle,
\qquad
S_c(q)=\frac1L\langle N_q^\dagger N_q\rangle,
$$

with the analogous bond structure factor $S_B(q)$. At $q=\pi$, these quantities measure staggered spin, charge, and bond fluctuations respectively.

```python
spin_z = [
    edf.SpinZ((i, UP), (i, DOWN), modes)
    for i in range(L)
]
delta_n = [
    n(i, UP) + n(i, DOWN) - 1
    for i in range(L)
]


def structure_factor(local_operators, q):
    O_q = 0
    for j, operator in enumerate(local_operators):
        O_q += np.exp(-1j * q * j) * operator
    return np.real(psi0.dag * O_q.dag * O_q * psi0) / L

print(f"S_s(pi): {structure_factor(spin_z, np.pi):.12f}")
print(f"S_c(pi): {structure_factor(delta_n, np.pi):.12f}")
print(f"S_B(pi): {structure_factor(bond_ops, np.pi):.12f}")
```

```text
S_s(pi): 0.284095781337
S_c(pi): 1.890419192675
S_B(pi): 2.910768481745
```

At this finite size and $V/t=2.25$, the staggered charge response is stronger than the staggered spin response, and the bond channel is also enhanced. These finite-size responses identify channels with large fluctuations. They do not establish simultaneous thermodynamic long-range order.

## Local charge reorganization

The site-averaged double occupancy is

```python
double_occupancy = sum(
    np.real(psi0.dag * n(i, UP) * n(i, DOWN) * psi0)
    for i in range(L)
) / L

print(f"{double_occupancy:.12f}")
```

```text
0.258651216123
```

Increasing $V$ favors an alternating charge pattern in which high- and low-density sites coexist. Once that tendency becomes strong enough, double occupation is no longer suppressed in the same way as in the pure repulsive Hubbard model: doublons participate in the emerging CDW-like pattern.

## Sweep $V/t$: which channel wins?

```{figure} ../../_static/figures/extended_hubbard_competition.svg
:width: 100%
:alt: Spin, charge, and bond structure factors and local observables across the extended-Hubbard interaction sweep

Six-site half-filled ring at $U/t=4$. The staggered spin response decreases as $V/t$ grows, the staggered charge response rises rapidly, and the bond response is enhanced near the crossover. Double occupancy increases as the charge-modulated side is approached. The vertical $V=U/2$ line is a strong-coupling reference, not a finite-size phase-boundary determination.
```

This evolution mirrors the established thermodynamic competition among SDW, BOW, and CDW regimes reported by Nakamura and later numerical studies.[^nakamura][^sandvik][^jeckelmann] The small ring broadens the transitions into crossovers and shifts finite-size extrema, so no phase boundary is inferred from the plotted values.

## Momentum profiles make the competition visible

The same three structure factors can be plotted over all discrete momenta of the six-site ring.

```{figure} ../../_static/figures/extended_hubbard_structure_profiles.svg
:width: 100%
:alt: Momentum-resolved spin, charge, and connected bond structure factors at three values of V

Momentum-resolved structure factors at fixed $U/t=4$. On the low-$V$ side, the strongest staggered response is spin-like. On the high-$V$ side, the $q=\pi$ charge peak dominates. Near the crossover, the connected bond structure factor is strongly enhanced at $q=\pi$, qualitatively consistent with the bond-order-wave regime found in larger-system studies.
```

For the bond panel the disconnected piece $|\langle B_q\rangle|^2/L$ is subtracted. This removes the uniform kinetic-energy expectation value and leaves the bond fluctuations.

The complete executable calculation at one representative parameter point is `examples/extended_hubbard_chain.py`. The parameter sweeps are generated by `docs/scripts/generate_figures.py`.

## References

[^nakamura]: M. Nakamura, “Mechanism of CDW-SDW Transition in One Dimension,” *J. Phys. Soc. Jpn.* **68**, 3123-3126 (1999), [doi:10.1143/JPSJ.68.3123](https://doi.org/10.1143/JPSJ.68.3123).
[^sandvik]: P. Sengupta, A. W. Sandvik, and D. K. Campbell, “Bond-order-wave phase and quantum phase transitions in the one-dimensional extended Hubbard model,” *Phys. Rev. B* **65**, 155113 (2002), [doi:10.1103/PhysRevB.65.155113](https://doi.org/10.1103/PhysRevB.65.155113).
[^jeckelmann]: E. Jeckelmann, “Ground-State Phase Diagram of a Half-Filled One-Dimensional Extended Hubbard Model,” *Phys. Rev. Lett.* **89**, 236401 (2002), [doi:10.1103/PhysRevLett.89.236401](https://doi.org/10.1103/PhysRevLett.89.236401).
