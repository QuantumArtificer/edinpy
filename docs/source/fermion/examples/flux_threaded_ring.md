# Flux-threaded ring: complex hopping and persistent current

A fermion moving around a ring acquires an Aharonov-Bohm phase when magnetic flux threads the ring. This makes a flux-threaded tight-binding ring a compact example of three EDinPy capabilities at once: genuinely complex Hamiltonian matrix elements, a parameter-dependent spectrum, and a response observable obtained directly from the Hamiltonian.

For a spinless ring of $L$ sites, distribute the total phase uniformly over the bonds and write

$$
H(\phi)=-t\sum_{j=0}^{L-1}
\left(
 e^{i\phi/L}c_j^\dagger c_{j+1}
+e^{-i\phi/L}c_{j+1}^\dagger c_j
\right),
$$

where $c_j^\dagger$ and $c_j$ create and annihilate a spinless fermion on site $j$, $t>0$ is the hopping amplitude, $j+1$ is understood modulo $L$, and

$$
\phi=2\pi\frac{\Phi}{\Phi_0}
$$

is the dimensionless flux in terms of the physical flux $\Phi$ and the single-particle flux quantum $\Phi_0=h/e$. The spectrum is periodic in $\phi$ with period $2\pi$, the lattice version of flux periodicity associated with Byers and Yang.[^byers-yang]

The example uses $L=4$, $N=2$, $t=1$, and $\phi=\pi/2$.

## Construct and print the complete Hamiltonian

```python
import numpy as np
from edinpy import fermion as edf

L = 4
N = 2
t = 1.0
phi = np.pi / 2

theta = phi / L
phase = np.exp(1j * theta)

site = edf.DoF(L, name="site")
modes = edf.FermionModes(site)
sector = edf.NParticleSector(modes, N=N).build()
c = edf.set_notation(edf.Annihilation, modes)
cd = edf.set_notation(edf.Creation, modes)

H = 0
for i in range(L):
    j = (i + 1) % L
    hop = cd(i) * c(j)
    H += -t * (phase * hop + phase.conjugate() * hop.dag)

print(H)
```

```text
(-0.9238795325112867-0.3826834323650898j) c†[0] c[1] + (-0.9238795325112867+0.3826834323650898j) c†[1] c[0] + (-0.9238795325112867-0.3826834323650898j) c†[1] c[2] + (-0.9238795325112867+0.3826834323650898j) c†[2] c[1] + (-0.9238795325112867-0.3826834323650898j) c†[2] c[3] + (-0.9238795325112867+0.3826834323650898j) c†[3] c[2] + (-0.9238795325112867-0.3826834323650898j) c†[3] c[0] + (-0.9238795325112867+0.3826834323650898j) c†[0] c[3]
```

The complex coefficients occur in conjugate pairs, so the many-body Hamiltonian remains Hermitian.

```python
hamiltonian = edf.Hamiltonian(H, sector)
energies, _ = hamiltonian.eigsolve(k=None)

print("matrix dtype:", hamiltonian.matrix.dtype)
print("Hermitian:", hamiltonian.is_hermitian())
print("energies:", np.round(energies, 8))
```

```text
matrix dtype: complex128
Hermitian: True
energies: [-2.61312593 -1.0823922   0.          0.          1.0823922   2.61312593]
```

## Analytic one-particle spectrum

Because the model is noninteracting, momentum diagonalizes the Hamiltonian. For

$$
k_m=\frac{2\pi m}{L},\qquad m=0,\ldots,L-1,
$$

the one-particle energies are

$$
\varepsilon_m(\phi)
=-2t\cos\left(k_m-\frac{\phi}{L}\right).
$$

At zero temperature the many-body ground-state energy is the sum of the $N$ lowest occupied one-particle levels,

$$
E_0(\phi)=\sum_{m\in\mathrm{occ}}\varepsilon_m(\phi).
$$

For the parameters above:

```python
k_values = 2 * np.pi * np.arange(L) / L
single_particle = -2 * t * np.cos(k_values - theta)
occupied = np.argsort(single_particle)[:N]
E0_exact = np.sum(single_particle[occupied])

print("E0 EDinPy:", energies[0])
print("E0 analytic:", E0_exact)
```

```text
E0 EDinPy: -2.6131259297527456
E0 analytic: -2.613125929752753
```

The agreement checks both the complex Peierls phases and the many-body occupation bookkeeping.

## Derive the persistent-current operator

The equilibrium persistent current is the derivative of the ground-state energy with respect to flux. With the dimensionless variable $\phi$ used here,

$$
I(\phi)=-\frac{\partial E_0}{\partial\phi}
       =\left\langle -\frac{\partial H}{\partial\phi}\right\rangle.
$$

Persistent equilibrium currents in normal mesoscopic rings were established theoretically by Büttiker, Imry, and Landauer.[^bil]

Because EDinPy exposes literal operators, the response operator can be differentiated from the Hamiltonian term by term:

```python
current = 0
for i in range(L):
    j = (i + 1) % L
    hop = cd(i) * c(j)
    current += (t / L) * (
        1j * phase * hop
        - 1j * phase.conjugate() * hop.dag
    )

psi0 = hamiltonian.eigenstate(0)
I_ed = np.real(psi0.dag * current * psi0)
I_exact = (2 * t / L) * np.sum(
    np.sin(k_values[occupied] - theta)
)

print("persistent current EDinPy:", I_ed)
print("persistent current analytic:", I_exact)
```

```text
persistent current EDinPy: 0.27059805007309856
persistent current analytic: 0.27059805007309845
```

The current operator is built from the same symbolic primitives as the Hamiltonian. No dedicated current routine is required.

```{figure} ../../_static/figures/flux_threaded_ring.svg
:width: 100%
:alt: Single-particle level flow, many-body ground-state energy, and persistent current of a flux-threaded fermion ring

Flux response of the half-filled four-site ring. The upper panel shows the analytic one-particle level flow. The lower panels compare EDinPy with the exact many-body ground-state energy and persistent current. Changes in the occupied one-particle levels generate the cusps in $E_0(\phi)$ and the corresponding jumps in the zero-temperature current.
```

## Local observables remain simple

The flux drives a circulating current without breaking translational invariance. The local density therefore remains uniform:

```python
n = edf.set_notation(edf.Number, modes)
densities = [
    round(float(np.real(psi0.dag * n(i) * psi0)), 12)
    for i in range(L)
]
print(densities)
```

```text
[0.5, 0.5, 0.5, 0.5]
```

The density remains uniform even though the phase-sensitive current is nonzero. The example illustrates why an eigensystem alone rarely exhausts the information contained in a many-body state.

[^byers-yang]: N. Byers and C. N. Yang, "Theoretical Considerations Concerning Quantized Magnetic Flux in Superconducting Cylinders," *Phys. Rev. Lett.* **7**, 46 (1961), [doi:10.1103/PhysRevLett.7.46](https://doi.org/10.1103/PhysRevLett.7.46).
[^bil]: M. Büttiker, Y. Imry, and R. Landauer, "Josephson behavior in small normal one-dimensional rings," *Phys. Lett. A* **96**, 365-367 (1983), [doi:10.1016/0375-9601(83)90011-7](https://doi.org/10.1016/0375-9601(83)90011-7).
