# Two-site exchange: singlets, triplets, and total spin

The two-site Heisenberg interaction is one of the simplest places where an exact-diagonalization calculation can be read almost entirely from the algebra. It is also a useful test of EDinPy's spin operators because the same fixed-particle-number sector contains both the familiar singly occupied spin states and fermionic doublon states.

Consider two spinful sites with two fermions and Hamiltonian

$$
H=J\,\mathbf S_0\cdot\mathbf S_1
=J\left[
S_0^zS_1^z+\frac{1}{2}
\left(S_0^+S_1^-+S_0^-S_1^+\right)
\right].
$$

Here $J$ is the exchange coupling. For $J>0$ the interaction is antiferromagnetic and favors the singlet. For $J<0$ it is ferromagnetic and favors the triplet. The fermionic representation uses creation and annihilation operators $c_{i\sigma}^\dagger$ and $c_{i\sigma}$ for spin $\sigma$ on site $i$, with $n_{i\sigma}=c_{i\sigma}^\dagger c_{i\sigma}$. The local spin operators are

$$
S_i^z=\frac{1}{2}(n_{i\uparrow}-n_{i\downarrow}),\qquad
S_i^+=c_{i\uparrow}^\dagger c_{i\downarrow},\qquad
S_i^-=c_{i\downarrow}^\dagger c_{i\uparrow}.
$$

## Build the fermionic problem

```python
import numpy as np
from edinpy import fermion as edf

L = 2
N = 2
J = 1.0
UP, DOWN = 0, 1

site = edf.DoF(L, name="site")
spin = edf.DoF(2, name="spin")
modes = edf.FermionModes(site, spin)
sector = edf.NParticleSector(modes, N=N)

Sdot = edf.HeisenbergExchange(
    (0, UP), (0, DOWN),
    (1, UP), (1, DOWN),
    J=1.0,
    modes=modes,
)
H = J * Sdot

print(H)
```

EDinPy prints the exchange operator as the underlying density and spin-flip terms:

```text
0.25 n[0,0] n[1,0] + -0.25 n[0,0] n[1,1] + -0.25 n[0,1] n[1,0] + 0.25 n[0,1] n[1,1] + 0.5 c†[0,0] c[0,1] c†[1,1] c[1,0] + 0.5 c†[0,1] c[0,0] c†[1,0] c[1,1]
```

The printed expression is the literal second-quantized Hamiltonian compiled into the fixed-$N$ matrix.

## Predict the spectrum before diagonalizing

If each site is singly occupied, two spin-$1/2$ degrees of freedom combine into one singlet and three triplet states. Since

$$
\mathbf S_0\cdot\mathbf S_1
=\frac{1}{2}
\left(\mathbf S_{\mathrm{tot}}^2-\mathbf S_0^2-\mathbf S_1^2\right),
$$

and $\mathbf S_i^2=3/4$, the exchange energies are

$$
E_{S=0}=-\frac{3J}{4},\qquad
E_{S=1}=\frac{J}{4}.
$$

The fermionic $N=2$ sector contains two additional basis states, $|\uparrow\downarrow,0\rangle$ and $|0,\uparrow\downarrow\rangle$. Their local spin vanishes, so this exchange Hamiltonian assigns both of them zero energy. The complete six-dimensional spectrum is therefore known analytically before running the eigensolver.

```python
hamiltonian = edf.Hamiltonian(H, sector)
energies, _ = hamiltonian.eigsolve(k=None)
print(energies)
```

```text
[-0.75  0.    0.    0.25  0.25  0.25]
```

The numerical spectrum is exactly the expected singlet, two spinless doublon states, and three triplet states.

```{figure} ../../_static/figures/spin_exchange_spectrum.svg
:width: 92%
:alt: Two-site exchange spectrum and spin correlations as a function of exchange coupling

Two-site fermionic exchange as $J$ is varied. EDinPy reproduces the analytic singlet branch $-3J/4$ and triplet branch $J/4$. The two doublon states remain at zero because they carry no local spin. The lower panel tracks the ground-state spin correlation.
```

## Diagnose the states with $\mathbf S_{\mathrm{tot}}^2$

Energy alone identifies the multiplets here, but total spin provides a more general state diagnostic. Define

$$
\mathbf S_{\mathrm{tot}}^2
=(S^z_{\mathrm{tot}})^2
+\frac{1}{2}
\left(S^+_{\mathrm{tot}}S^-_{\mathrm{tot}}
+S^-_{\mathrm{tot}}S^+_{\mathrm{tot}}\right),
$$

with $S^a_{\mathrm{tot}}=S^a_0+S^a_1$. In an eigenstate of total spin $S$, its expectation value is $S(S+1)$.

```python
Splus = sum(
    (edf.SpinPlus((i, UP), (i, DOWN), modes) for i in range(L)),
    start=0,
)
Sminus = sum(
    (edf.SpinMinus((i, UP), (i, DOWN), modes) for i in range(L)),
    start=0,
)
Sz = sum(
    (edf.SpinZ((i, UP), (i, DOWN), modes) for i in range(L)),
    start=0,
)
S2 = Sz * Sz + 0.5 * (Splus * Sminus + Sminus * Splus)

values = [
    np.real(hamiltonian.eigenstate(i).dag * S2 * hamiltonian.eigenstate(i))
    for i in range(len(energies))
]
print(np.round(values, 8))
```

```text
[0. 0. 0. 2. 2. 2.]
```

The ground-state singlet has $\langle\mathbf S_{\mathrm{tot}}^2\rangle=0$. The two doublon states also have total spin zero, whereas the triplet manifold gives $S(S+1)=2$.

The exchange correlator itself is obtained with the same Dirac expression used everywhere else in EDinPy:

```python
psi0 = hamiltonian.eigenstate(0)
print(psi0.dag * Sdot * psi0)
```

```text
-0.7500000000000001
```

Thus the ground state saturates the singlet value $\langle\mathbf S_0\cdot\mathbf S_1\rangle=-3/4$.

## Hilbert-space interpretation

The spin model and the fermionic Hilbert space used to represent it are not identical. Fixing $N=2$ does not enforce one particle per site. If the physical model requires exactly one spin-$1/2$ on every site, the two doublon states are outside that spin-only subspace. They are retained here to make the Hilbert-space distinction explicit.

The same operator construction applies to exchange terms embedded in Hubbard-like models, where charge fluctuations and spin exchange are both present.
