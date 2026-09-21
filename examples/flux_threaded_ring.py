"""Spinless four-site ring threaded by a magnetic flux."""

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
sector = edf.NParticleSector(modes, N=N)
c = edf.set_notation(edf.Annihilation, modes)
cd = edf.set_notation(edf.Creation, modes)
n = edf.set_notation(edf.Number, modes)

H = 0
current = 0
for i in range(L):
    j = (i + 1) % L
    hop = cd(i) * c(j)
    H += -t * (phase * hop + phase.conjugate() * hop.dag)
    current += (t / L) * (
        1j * phase * hop - 1j * phase.conjugate() * hop.dag
    )

hamiltonian = edf.Hamiltonian(H, sector)
energies, _ = hamiltonian.eigsolve(k=None)
psi0 = hamiltonian.eigenstate(0)

k_values = 2 * np.pi * np.arange(L) / L
single_particle = -2 * t * np.cos(k_values - theta)
occupied = np.argsort(single_particle)[:N]
E0_exact = np.sum(single_particle[occupied])
I_exact = (2 * t / L) * np.sum(np.sin(k_values[occupied] - theta))

print("Hamiltonian:")
print(H)
print("\nmatrix dtype:", hamiltonian.matrix.dtype)
print("energies:", np.round(energies, 8))
print("E0:", energies[0], "analytic:", E0_exact)
print("persistent current:", np.real(psi0.dag * current * psi0), "analytic:", I_exact)
print("local densities:", [
    round(float(np.real(psi0.dag * n(i) * psi0)), 12)
    for i in range(L)
])
