"""Half-filled Hubbard dimer: exact spectrum and ground-state observables."""

import numpy as np

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

hamiltonian = edf.Hamiltonian(H, sector)
energies, _ = hamiltonian.eigsolve(k=None)
psi0 = hamiltonian.eigenstate(0)

D = n(0, UP) * n(0, DOWN) + n(1, UP) * n(1, DOWN)
mu2 = 0.5 * sum(
    (
        (n(i, UP) - n(i, DOWN))
        * (n(i, UP) - n(i, DOWN))
        for i in range(L)
    ),
    start=0,
)
Sdot = edf.HeisenbergExchange(
    (0, UP), (0, DOWN), (1, UP), (1, DOWN), 1.0, modes
)

root = np.sqrt(U**2 + 16.0 * t**2)
E0_exact = 0.5 * (U - root)
D_exact = 0.5 * (1.0 - U / root)
mu2_exact = 1.0 - D_exact
Sdot_exact = -0.75 * mu2_exact

print("Hamiltonian:")
print(H)
print("\nsector dimension:", sector.dimension)
print("energies:", np.round(energies, 8))
print("E0 exact:", E0_exact)
print("<D>:", np.real(psi0.dag * D * psi0), "exact:", D_exact)
print("mu^2:", np.real(psi0.dag * mu2 * psi0), "exact:", mu2_exact)
print("<S0.S1>:", np.real(psi0.dag * Sdot * psi0), "exact:", Sdot_exact)
