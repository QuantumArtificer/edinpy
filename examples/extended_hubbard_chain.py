"""Ground-state spectrum of a spinful one-dimensional extended Hubbard chain."""

from edinpy import fermion as edf


L = 8
N = 8
t = 1.0
U = 4.0
V = 1.5

site = edf.DoF(L, name="site")
spin = edf.DoF(2, name="spin")
modes = edf.FermionModes(site, spin)
sector = edf.NParticleSector(modes, N=N)

c = edf.set_notation(edf.Annihilation, modes)
cd = edf.set_notation(edf.Creation, modes)
n = edf.set_notation(edf.Number, modes)

UP, DOWN = 0, 1
H = 0

for i in range(L - 1):
    j = i + 1
    for sigma in (UP, DOWN):
        H += -t * (
            cd(i, sigma) * c(j, sigma)
            + cd(j, sigma) * c(i, sigma)
        )

for i in range(L):
    H += U * n(i, UP) * n(i, DOWN)

for i in range(L - 1):
    j = i + 1
    n_i = n(i, UP) + n(i, DOWN)
    n_j = n(j, UP) + n(j, DOWN)
    H += V * n_i * n_j

hamiltonian = edf.Hamiltonian(H, sector)
eigenvalues, _ = hamiltonian.eigsolve(k=4, which="SA")
print(eigenvalues)
