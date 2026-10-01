"""Six-site Bose-Hubbard ring with custom density-matrix observables."""

import numpy as np

from edinpy import boson as edb

L = 6
N = 6
J = 1.0
U = 8.0

site = edb.DoF(L, name="site")
modes = edb.BosonModes(site)
sector = edb.NParticleSector(modes, N=N).build()

b = edb.set_notation(edb.Annihilation, modes)
bd = edb.set_notation(edb.Creation, modes)
n = edb.set_notation(edb.Number, modes)

H = 0
for i in range(L):
    H += edb.Hopping(i, (i + 1) % L, -J, modes)
    H += edb.Hubbard(i, U, modes)

hamiltonian = edb.Hamiltonian(H, sector)
energies, _ = hamiltonian.eigsolve(k=2, which="SA")
psi0 = hamiltonian.eigenstate(0)


def one_body_density_matrix(psi):
    return np.asarray(
        [
            [psi.dag * bd(i) * b(j) * psi for j in range(L)]
            for i in range(L)
        ],
        dtype=complex,
    )




def g1(psi, r):
    return float(
        np.real(
            sum(
                psi.dag * bd(i) * b((i + r) % L) * psi
                for i in range(L)
            ) / L
        )
    )


def number_variance(psi):
    values = []
    for i in range(L):
        mean = np.real(psi.dag * n(i) * psi)
        mean2 = np.real(psi.dag * n(i) * n(i) * psi)
        values.append(mean2 - mean**2)
    return float(np.mean(values))


def density_structure_factor(psi, q):
    delta_n_q = sum(
        (
            np.exp(-1j * q * j) * (n(j) - N / L)
            for j in range(L)
        ),
        start=0,
    )
    return float(np.real(psi.dag * delta_n_q.dag * delta_n_q * psi) / L)


rho = one_body_density_matrix(psi0)
natural_occupations = np.linalg.eigvalsh(rho)[::-1]
q_values = 2 * np.pi * np.arange(L) / L
structure = [density_structure_factor(psi0, q) for q in q_values]
coherence = [g1(psi0, r) for r in range(L // 2 + 1)]

print("dimension:", sector.dimension)
print("lowest energies:", np.round(energies, 8))
print("natural occupations:", np.round(natural_occupations, 8))
print("condensate fraction:", natural_occupations[0] / N)
print("number variance:", number_variance(psi0))
print("g1:", np.round(coherence, 8))
print("density structure factor:", np.round(structure, 8))
