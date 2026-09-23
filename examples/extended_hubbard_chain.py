"""Six-site extended Hubbard ring with spin, charge, and bond diagnostics."""

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

hamiltonian = edf.Hamiltonian(H, sector)
energies, _ = hamiltonian.eigsolve(k=4, which="SA")
psi0 = hamiltonian.eigenstate(0)

spin_z = [edf.SpinZ((i, UP), (i, DOWN), modes) for i in range(L)]
delta_n = [n(i, UP) + n(i, DOWN) - 1 for i in range(L)]


def structure_factor(local_operators, q):
    O_q = 0
    for j, operator in enumerate(local_operators):
        O_q += np.exp(-1j * q * j) * operator
    return np.real(psi0.dag * O_q.dag * O_q * psi0) / L


double_occupancy = sum(
    np.real(psi0.dag * n(i, UP) * n(i, DOWN) * psi0)
    for i in range(L)
) / L

print("Hamiltonian:")
print(H)
print("\nsector dimension:", sector.dimension)
print("lowest energies:", np.round(energies, 8))
print(f"double occupancy/site: {double_occupancy:.12f}")
print(f"S_s(pi): {structure_factor(spin_z, np.pi):.12f}")
print(f"S_c(pi): {structure_factor(delta_n, np.pi):.12f}")
print(f"S_B(pi): {structure_factor(bond_ops, np.pi):.12f}")
