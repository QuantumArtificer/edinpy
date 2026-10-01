"""Two-site spin-1/2 Heisenberg exchange in a constrained fermion sector."""

import numpy as np

from edinpy import fermion as edf

UP, DOWN = 0, 1
J = 1.0

site = edf.DoF(2, name="site", labels=("left", "right"))
spin = edf.DoF(2, name="spin", labels=("up", "down"))
modes = edf.FermionModes(site, spin)
sector = (
    edf.NParticleSector(modes, N=2)
    .project_particles("site", left=1, right=1)
    .build()
)

Sdot = edf.HeisenbergExchange(
    (0, UP), (0, DOWN),
    (1, UP), (1, DOWN),
    J=1.0,
    modes=modes,
)

Splus = sum(
    (edf.SpinPlus((i, UP), (i, DOWN), modes) for i in range(2)),
    start=0,
)
Sminus = sum(
    (edf.SpinMinus((i, UP), (i, DOWN), modes) for i in range(2)),
    start=0,
)
Sz = sum(
    (edf.SpinZ((i, UP), (i, DOWN), modes) for i in range(2)),
    start=0,
)
S2 = Sz * Sz + 0.5 * (Splus * Sminus + Sminus * Splus)

hamiltonian = edf.Hamiltonian(J * Sdot, sector)
energies, _ = hamiltonian.eigsolve(k=None)
spin_squared = [
    float(np.real(hamiltonian.eigenstate(i).dag * S2 * hamiltonian.eigenstate(i)))
    for i in range(sector.dimension)
]

print("dimension:", sector.dimension)
print("energies:", np.round(energies, 8))
print("<S_tot^2>:", np.round(spin_squared, 8))
