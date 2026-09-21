# Half-filled Hubbard ring: correlations and structure factors

The Hubbard dimer shows how interactions suppress charge fluctuations on two sites. A periodic chain adds the first genuinely collective question: how do those local changes reorganize correlations in real and momentum space?

Consider a six-site periodic ring at half filling,

$$
H=-t\sum_{i=0}^{L-1}\sum_{\sigma}
\left(c_{i\sigma}^\dagger c_{i+1,\sigma}+\mathrm{H.c.}\right)
+U\sum_{i=0}^{L-1}n_{i\uparrow}n_{i\downarrow},
$$

with $c_{L\sigma}\equiv c_{0\sigma}$. The hopping amplitude $t$ sets the energy scale, $U>0$ is the on-site repulsion, and $L=N=6$ gives one fermion per site on average. The thermodynamic one-dimensional Hubbard model is exactly solvable by the Lieb-Wu Bethe ansatz. At half filling, the exact solution is insulating for every repulsive $U>0$.[^liebwu]

This finite ring is not a thermodynamic calculation. Its purpose is to show how the same exact ground state can be interrogated through local observables, real-space correlation functions, Fourier-space structure factors, and a pair correlator.

## Build and print the Hamiltonian

```python
import numpy as np
from edinpy import fermion as edf

L = 6
N = L
UP, DOWN = 0, 1
t = 1.0
U = 4.0

site = edf.DoF(L, name="site")
spin = edf.DoF(2, name="spin")
modes = edf.FermionModes(site, spin)
sector = edf.NParticleSector(modes, N=N)

c = edf.set_notation(edf.Annihilation, modes)
cd = edf.set_notation(edf.Creation, modes)
n = edf.set_notation(edf.Number, modes)

H = 0
for i in range(L):
    j = (i + 1) % L
    for sigma in (UP, DOWN):
        hop = cd(i, sigma) * c(j, sigma)
        H += -t * (hop + hop.dag)
for i in range(L):
    H += U * n(i, UP) * n(i, DOWN)

print(H)
```

```text
-1.0 c†[0,0] c[1,0] + -1.0 c†[1,0] c[0,0] + -1.0 c†[0,1] c[1,1] + -1.0 c†[1,1] c[0,1] + -1.0 c†[1,0] c[2,0] + -1.0 c†[2,0] c[1,0] + -1.0 c†[1,1] c[2,1] + -1.0 c†[2,1] c[1,1] + -1.0 c†[2,0] c[3,0] + -1.0 c†[3,0] c[2,0] + -1.0 c†[2,1] c[3,1] + -1.0 c†[3,1] c[2,1] + -1.0 c†[3,0] c[4,0] + -1.0 c†[4,0] c[3,0] + -1.0 c†[3,1] c[4,1] + -1.0 c†[4,1] c[3,1] + -1.0 c†[4,0] c[5,0] + -1.0 c†[5,0] c[4,0] + -1.0 c†[4,1] c[5,1] + -1.0 c†[5,1] c[4,1] + -1.0 c†[5,0] c[0,0] + -1.0 c†[0,0] c[5,0] + -1.0 c†[5,1] c[0,1] + -1.0 c†[0,1] c[5,1] + 4.0 n[0,0] n[0,1] + 4.0 n[1,0] n[1,1] + 4.0 n[2,0] n[2,1] + 4.0 n[3,0] n[3,1] + 4.0 n[4,0] n[4,1] + 4.0 n[5,0] n[5,1]
```

The last hopping pair closes the ring between sites 5 and 0. Printing the Hamiltonian makes this boundary condition explicit.

## Low-energy eigensystem

The half-filled sector contains

$$
\binom{2L}{N}=\binom{12}{6}=924
$$

basis states. Only a few low-energy eigenpairs are needed for the observables below.

```python
hamiltonian = edf.Hamiltonian(H, sector)
energies, _ = hamiltonian.eigsolve(k=8, which="SA")
psi0 = hamiltonian.eigenstate(0)

print("sector dimension:", sector.dimension)
print("lowest energies:", energies)
```

```text
sector dimension: 924
lowest energies: [-3.66870618 -2.89838147 -2.89838147 -2.89838147 -2.51637687
 -2.42291126 -2.42291126 -2.42291126]
```

The spectrum alone does not identify which correlations are changing as $U/t$ grows. Ground-state observables distinguish charge, spin, and pairing responses.

## Local diagnostics: double occupancy and the local moment

The site-averaged double occupancy is

$$
d=\frac1L\sum_i\langle n_{i\uparrow}n_{i\downarrow}\rangle,
$$

while the local moment is

$$
\mu^2=\frac1L\sum_i\left\langle(n_{i\uparrow}-n_{i\downarrow})^2\right\rangle.
$$

```python
double_occupancy = sum(
    np.real(psi0.dag * n(i, UP) * n(i, DOWN) * psi0)
    for i in range(L)
) / L

local_moment = sum(
    np.real(
        psi0.dag
        * (n(i, UP) - n(i, DOWN))
        * (n(i, UP) - n(i, DOWN))
        * psi0
    )
    for i in range(L)
) / L

print(f"double occupancy/site: {double_occupancy:.12f}")
print(f"local moment/site: {local_moment:.12f}")
```

```text
double occupancy/site: 0.111065916660
local moment/site: 0.777868166680
```

At half filling these two quantities move in opposite directions: repulsion removes doublon weight and stabilizes singly occupied local moments.

## Real-space spin and charge correlations

Define

$$
S_i^z=\frac12(n_{i\uparrow}-n_{i\downarrow}),
\qquad
\delta n_i=n_i-1.
$$

For a translationally invariant ring, useful equal-time correlations are

$$
C_s(r)=\frac1L\sum_i\langle S_i^zS_{i+r}^z\rangle,
\qquad
C_c(r)=\frac1L\sum_i\langle\delta n_i\,\delta n_{i+r}\rangle.
$$

```python
spin_z = [
    edf.SpinZ((i, UP), (i, DOWN), modes)
    for i in range(L)
]
delta_n = [
    n(i, UP) + n(i, DOWN) - 1
    for i in range(L)
]


def correlation(local_operators, r):
    return sum(
        np.real(
            psi0.dag
            * local_operators[i]
            * local_operators[(i + r) % L]
            * psi0
        )
        for i in range(L)
    ) / L

print("C_s(r):", np.round([correlation(spin_z, r) for r in range(4)], 6))
print("C_c(r):", np.round([correlation(delta_n, r) for r in range(4)], 6))
```

```text
C_s(r): [ 0.194467 -0.108857  0.035074 -0.046901]
C_c(r): [ 0.222132 -0.095259 -0.010513 -0.010587]
```

The alternating sign of $C_s(r)$ is the finite-ring signature of antiferromagnetic correlations. Charge correlations are simultaneously reduced as double occupation becomes costly.

```{figure} ../../_static/figures/hubbard_chain_correlations.svg
:width: 100%
:alt: Real-space spin and charge correlations of the half-filled six-site Hubbard ring

Translationally averaged equal-time correlations. Increasing $U/t$ strengthens the alternating spin pattern and suppresses the connected charge fluctuations.
```

## Structure factors: the same information in momentum space

For any local operator $O_j$, define

$$
O_q=\sum_j e^{-iqj}O_j,
\qquad
S_O(q)=\frac1L\langle O_q^\dagger O_q\rangle.
$$

This definition maps directly onto the literal operator algebra:

```python
def structure_factor(local_operators, q):
    O_q = 0
    for j, operator in enumerate(local_operators):
        O_q += np.exp(-1j * q * j) * operator
    return np.real(psi0.dag * O_q.dag * O_q * psi0) / L

q = np.pi
print(f"S_s(pi): {structure_factor(spin_z, q):.12f}")
print(f"S_c(pi): {structure_factor(delta_n, q):.12f}")
```

```text
S_s(pi): 0.529230730431
S_c(pi): 0.402211908588
```

The spin structure factor grows at $q=\pi$ as the repulsive chain develops stronger antiferromagnetic correlations. The connected charge structure factor is suppressed. This finite-size trend is the expected counterpart of the spin-charge separation and Mott physics of the exact one-dimensional model.[^liebwu]

## A four-fermion observable: the pair structure factor

The on-site singlet-pair annihilation operator is

$$
\Delta_i=c_{i\downarrow}c_{i\uparrow}.
$$

Although $\Delta_i$ changes particle number, the product $\Delta_q^\dagger\Delta_q$ conserves it and therefore has a perfectly well-defined expectation value in the fixed-$N$ ground state:

$$
P(q)=\frac1L\langle\Delta_q^\dagger\Delta_q\rangle.
$$

```python
pairs = [
    c(i, DOWN) * c(i, UP)
    for i in range(L)
]

print(f"P(0): {structure_factor(pairs, 0.0):.12f}")
```

```text
P(0): 0.201105954294
```

This calculation uses the literal operator-state algebra beyond Hamiltonian construction. A number-changing intermediate operator can enter a number-conserving correlation function without a separate observable API.

```{figure} ../../_static/figures/hubbard_chain_structure_factors.svg
:width: 100%
:alt: Spin, charge, and pair structure factors of the six-site Hubbard ring

Momentum-resolved spin, charge, and on-site pair structure factors. Repulsive $U$ transfers weight toward the antiferromagnetic spin channel at $q=\pi$ while suppressing charge and on-site pair fluctuations.
```

## Follow the crossover with $U/t$

Repeating the calculation as $U/t$ changes gives a compact many-body summary.

```{figure} ../../_static/figures/hubbard_chain_spectrum_observables.svg
:width: 100%
:alt: Low-energy spectrum and several observables of the half-filled Hubbard ring as U changes

Six-site half-filled ring. The spectrum is shown together with double occupancy, local-moment formation, staggered spin/charge structure factors, and the zero-momentum on-site pair structure factor. The figures are generated directly from the public EDinPy API.
```

As $U/t$ increases, local charge motion is suppressed, local moments grow, antiferromagnetic correlations strengthen, and on-site pair fluctuations fall. The finite ring cannot establish thermodynamic critical behavior. Its suppression of charge fluctuations is consistent with the insulating half-filled ground state of the exact one-dimensional solution.[^liebwu]

The complete executable calculation is `examples/hubbard_chain.py`.

## References

[^liebwu]: E. H. Lieb and F. Y. Wu, “Absence of Mott Transition in an Exact Solution of the Short-Range, One-Band Model in One Dimension,” *Phys. Rev. Lett.* **20**, 1445 (1968), [doi:10.1103/PhysRevLett.20.1445](https://doi.org/10.1103/PhysRevLett.20.1445).
