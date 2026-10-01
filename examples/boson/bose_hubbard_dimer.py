"""Two-mode Bose-Hubbard dimer and Schwinger-spin observables."""

import numpy as np

from edinpy import boson as edb

N = 8
J = 1.0
U = 4.0

well = edb.DoF(2, name="well", labels=("left", "right"))
modes = edb.BosonModes(well)
sector = edb.NParticleSector(modes, N=N).build()

H = (
    edb.Hopping(0, 1, -J, modes)
    + edb.Hubbard(0, U, modes)
    + edb.Hubbard(1, U, modes)
)

Sx = edb.SpinX(0, 1, modes)
Sz = edb.SpinZ(0, 1, modes)

hamiltonian = edb.Hamiltonian(H, sector)
energies, _ = hamiltonian.eigsolve(k=None)
psi0 = hamiltonian.eigenstate(0)

print("dimension:", sector.dimension)
print("lowest energies:", np.round(energies[:5], 8))
print("coherence:", float(np.real(psi0.dag * Sx * psi0) / (N / 2)))
print("imbalance variance:", float(np.real(psi0.dag * Sz * Sz * psi0) / (N / 2) ** 2))
print("probabilities:", np.round(np.abs(psi0.coefficients) ** 2, 6))
