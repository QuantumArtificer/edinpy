# Extended Hubbard ring: competing spin, charge, and bond correlations

The one-dimensional extended Hubbard model adds a nearest-neighbor repulsion to the ordinary Hubbard Hamiltonian,

$$
H=-t\sum_{i,\sigma}
\left(c_{i\sigma}^\dagger c_{i+1,\sigma}+\mathrm{H.c.}\right)
+U\sum_i n_{i\uparrow}n_{i\downarrow}
+V\sum_i n_i n_{i+1}.
$$

At half filling the thermodynamic model contains spin-density-wave, charge-density-wave, and bond-order-wave regimes.[^nakamura][^sandvik][^jeckelmann] A six-site ring cannot determine the phase boundaries, but it resolves the redistribution of finite-size correlations as $V/t$ is increased at fixed $U/t=4$.

## Local operators for the three channels

The staggered spin and charge operators are built from

$$
S_i^z=\frac12(n_{i\uparrow}-n_{i\downarrow}),
\qquad
\delta n_i=n_i-1.
$$

The bond operator is

$$
B_i=\sum_\sigma
\left(c_{i\sigma}^\dagger c_{i+1,\sigma}+\mathrm{H.c.}\right).
$$

```python
spin_z = [edf.SpinZ((i, UP), (i, DOWN), modes) for i in range(L)]
delta_n = [n(i, UP) + n(i, DOWN) - 1 for i in range(L)]

bond = []
for i in range(L):
    j = (i + 1) % L
    B = 0
    for sigma in (UP, DOWN):
        hop = cd(i, sigma) * c(j, sigma)
        B += hop + hop.dag
    bond.append(B)
```

For any local field $O_j$,

$$
S_O(q)=\frac1L\langle O_q^\dagger O_q\rangle,
\qquad
O_q=\sum_j e^{-iqj}O_j.
$$

For the bond channel the disconnected contribution $|\langle B_q\rangle|^2/L$ is subtracted.

## Spin response

```{figure} ../../_static/figures/example_extended_hubbard_spin_pi.svg
:width: 76%
:alt: Staggered spin structure factor of the six-site extended Hubbard ring versus V over t

Staggered spin structure factor at $U/t=4$. Increasing nearest-neighbor repulsion weakens the spin-dominated response of the small ring as the system approaches the charge-dominated side.
```

## Charge response

```{figure} ../../_static/figures/example_extended_hubbard_charge_pi.svg
:width: 76%
:alt: Staggered charge structure factor of the six-site extended Hubbard ring versus V over t

Staggered charge structure factor at $U/t=4$. The strong growth near and beyond the reference line $V=U/2$ reflects the increasing tendency toward alternating high- and low-density sites.
```

## Bond fluctuations

```{figure} ../../_static/figures/example_extended_hubbard_bond_pi.svg
:width: 76%
:alt: Connected staggered bond structure factor of the six-site extended Hubbard ring versus V over t

Connected staggered bond structure factor. The enhanced bond fluctuations near the spin-charge competition region are the finite-cluster signature most closely associated with the bond-order-wave regime found in larger systems.
```

## Double occupancy

The site-averaged doublon density is

$$
D=\frac1L\sum_i\langle n_{i\uparrow}n_{i\downarrow}\rangle.
$$

```{figure} ../../_static/figures/example_extended_hubbard_double_occupancy.svg
:width: 76%
:alt: Double occupancy of the six-site extended Hubbard ring versus V over t

Double occupancy at $U/t=4$. On the charge-dominated side, doublons become favorable because an alternating doublon-rich and low-density pattern reduces the nearest-neighbor repulsion.
```

Taken together, the four observables separate the three competing tendencies of the finite ring. At small $V/t$, the staggered spin response is largest and doublon density remains suppressed. Around the spin-charge competition region, the spin response falls before the charge response has fully saturated, while connected bond fluctuations are enhanced. At larger $V/t$, the staggered charge response and double occupancy rise together. The broad crossover and the locations of the extrema are finite-size properties, not estimates of thermodynamic phase boundaries.

## Momentum profiles

The discrete momenta of the six-site ring show where the spectral weight accumulates. The three couplings below sample the spin-dominated side, the competition region, and the charge-dominated side.

```{figure} ../../_static/figures/example_extended_hubbard_spin_q.svg
:width: 76%
:alt: Momentum-resolved spin structure factor at three V over t values

Momentum-resolved spin structure factor. The $q=\pi$ peak is strongest on the low-$V$ side and loses weight as charge ordering becomes favorable.
```

```{figure} ../../_static/figures/example_extended_hubbard_charge_q.svg
:width: 76%
:alt: Momentum-resolved charge structure factor at three V over t values

Momentum-resolved charge structure factor. The $q=\pi$ component becomes dominant on the high-$V$ side.
```

```{figure} ../../_static/figures/example_extended_hubbard_bond_q.svg
:width: 76%
:alt: Momentum-resolved connected bond structure factor at three V over t values

Connected bond structure factor. The staggered bond channel is enhanced most strongly in the intermediate coupling region of the finite ring.
```

The finite-size extrema should not be interpreted as thermodynamic phase boundaries. They show instead how spin, charge, and bond fluctuations exchange spectral weight across the competition region.

[^nakamura]: M. Nakamura, "Mechanism of CDW-SDW Transition in One Dimension," J. Phys. Soc. Jpn. 68, 3123-3126 (1999), [doi:10.1143/JPSJ.68.3123](https://doi.org/10.1143/JPSJ.68.3123).
[^sandvik]: P. Sengupta, A. W. Sandvik, and D. K. Campbell, "Bond-order-wave phase and quantum phase transitions in the one-dimensional extended Hubbard model," Phys. Rev. B 65, 155113 (2002), [doi:10.1103/PhysRevB.65.155113](https://doi.org/10.1103/PhysRevB.65.155113).
[^jeckelmann]: E. Jeckelmann, "Ground-State Phase Diagram of a Half-Filled One-Dimensional Extended Hubbard Model," Phys. Rev. Lett. 89, 236401 (2002), [doi:10.1103/PhysRevLett.89.236401](https://doi.org/10.1103/PhysRevLett.89.236401).
