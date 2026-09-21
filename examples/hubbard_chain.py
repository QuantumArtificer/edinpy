"""Six-site periodic Hubbard chain with correlations and structure factors."""

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

hamiltonian = edf.Hamiltonian(H, sector)
energies, _ = hamiltonian.eigsolve(k=8, which="SA")
psi0 = hamiltonian.eigenstate(0)

spin_z = [edf.SpinZ((i, UP), (i, DOWN), modes) for i in range(L)]
delta_n = [n(i, UP) + n(i, DOWN) - 1 for i in range(L)]
pairs = [c(i, DOWN) * c(i, UP) for i in range(L)]


def structure_factor(local_operators, q):
    O_q = 0
    for j, operator in enumerate(local_operators):
        O_q += np.exp(-1j * q * j) * operator
    return np.real(psi0.dag * O_q.dag * O_q * psi0) / L


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

print("Hamiltonian:")
print(H)
print("\nsector dimension:", sector.dimension)
print("lowest energies:", np.round(energies, 8))
print(f"double occupancy/site: {double_occupancy:.12f}")
print(f"local moment/site: {local_moment:.12f}")
print(f"S_s(pi): {structure_factor(spin_z, np.pi):.12f}")
print(f"S_c(pi): {structure_factor(delta_n, np.pi):.12f}")
print(f"P(0): {structure_factor(pairs, 0.0):.12f}")
