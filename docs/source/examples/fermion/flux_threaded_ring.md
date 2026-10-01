# Flux-threaded ring: level flow and persistent current

A magnetic flux through a mesoscopic ring changes the boundary phase acquired by a charged particle around the loop. The resulting level flow produces an equilibrium persistent current, a standard manifestation of phase coherence in multiply connected conductors.[^byers-yang][^bil]

For a spinless ring of $L$ sites,

$$
H(\phi)=-t\sum_{j=0}^{L-1}
\left(
 e^{i\phi/L}c_j^\dagger c_{j+1}
+e^{-i\phi/L}c_{j+1}^\dagger c_j
\right),
$$

where

$$
\phi=2\pi\frac{\Phi}{\Phi_0},
\qquad
\Phi_0=\frac{h}{e}.
$$

The example uses $L=4$, $N=2$, and $t=1$.

## Complex hopping

```python
import numpy as np
from edinpy import fermion as edf

L = 4
N = 2
t = 1.0
phi = np.pi / 2

theta = phi / L
phase = np.exp(1j * theta)

modes = edf.FermionModes(edf.DoF(L, name="site"))
sector = edf.NParticleSector(modes, N=N).build()
c = edf.set_notation(edf.Annihilation, modes)
cd = edf.set_notation(edf.Creation, modes)

H = 0
for i in range(L):
    j = (i + 1) % L
    hop = cd(i) * c(j)
    H += -t * (phase * hop + phase.conjugate() * hop.dag)
```

The conjugate hopping amplitudes preserve Hermiticity while making the matrix genuinely complex.

## Single-particle level flow

For momenta

$$
k_m=\frac{2\pi m}{L},
$$

the one-particle energies are

$$
\varepsilon_m(\phi)=-2t\cos\left(k_m-\frac{\phi}{L}\right).
$$

```{figure} ../../_static/figures/example_flux_single_particle.svg
:width: 76%
:alt: Single-particle energy levels of a four-site ring versus magnetic flux

Single-particle level flow of the four-site ring. Crossings change which orbitals are occupied in the zero-temperature many-body ground state.
```

## Many-body ground-state energy

For noninteracting fermions,

$$
E_0(\phi)=\sum_{m\in\mathrm{occ}}\varepsilon_m(\phi).
$$

```{figure} ../../_static/figures/example_flux_ground_energy.svg
:width: 76%
:alt: Ground-state energy of the half-filled four-site ring versus magnetic flux

Many-body ground-state energy at half filling. Exact diagonalization of the complex Fock-space Hamiltonian reproduces the occupied single-particle result, including the cusps produced by level crossings.
```

## Persistent current as a custom operator

The equilibrium current is

$$
I(\phi)=-\frac{\partial E_0}{\partial\phi}
=\left\langle-\frac{\partial H}{\partial\phi}\right\rangle.
$$

Differentiating the Hamiltonian term by term gives

$$
\hat I=\frac{t}{L}\sum_j
\left[
 i e^{i\phi/L}c_j^\dagger c_{j+1}
-i e^{-i\phi/L}c_{j+1}^\dagger c_j
\right].
$$

The literal operator is:

```python
current = 0
for i in range(L):
    j = (i + 1) % L
    hop = cd(i) * c(j)
    current += (t / L) * (
        1j * phase * hop
        - 1j * phase.conjugate() * hop.dag
    )

hamiltonian = edf.Hamiltonian(H, sector)
hamiltonian.eigsolve(k=None)
psi0 = hamiltonian.eigenstate(0)
I = np.real(psi0.dag * current * psi0)
```

```{figure} ../../_static/figures/example_flux_current.svg
:width: 76%
:alt: Persistent current of the half-filled four-site ring versus magnetic flux

Persistent current of the half-filled ring. The discontinuous changes at zero temperature coincide with changes in the occupied single-particle levels.
```

[^byers-yang]: N. Byers and C. N. Yang, "Theoretical Considerations Concerning Quantized Magnetic Flux in Superconducting Cylinders," Phys. Rev. Lett. 7, 46 (1961), [doi:10.1103/PhysRevLett.7.46](https://doi.org/10.1103/PhysRevLett.7.46).
[^bil]: M. Büttiker, Y. Imry, and R. Landauer, "Josephson behavior in small normal one-dimensional rings," Phys. Lett. A 96, 365-367 (1983), [doi:10.1016/0375-9601(83)90011-7](https://doi.org/10.1016/0375-9601(83)90011-7).
