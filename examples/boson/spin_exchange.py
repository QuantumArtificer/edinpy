"""Two spin-1/2 sites represented by projected Schwinger bosons."""

import numpy as np

from edinpy import boson as edb

site = edb.DoF(2, name="site", labels=("left", "right"))
component = edb.DoF(2, name="component", labels=("up", "down"))
modes = edb.BosonModes(site, component)

sector = (
    edb.NParticleSector(modes, N=2)
    .project_particles("site", left=1, right=1)
    .build()
)

UP, DOWN = 0, 1
Sdot = edb.HeisenbergExchange(
    (0, UP), (0, DOWN),
    (1, UP), (1, DOWN),
    J=1.0,
    modes=modes,
)

Splus = sum(
    (edb.SpinPlus((i, UP), (i, DOWN), modes) for i in range(2)),
    start=0,
)
Sminus = sum(
    (edb.SpinMinus((i, UP), (i, DOWN), modes) for i in range(2)),
    start=0,
)
Sz = sum(
    (edb.SpinZ((i, UP), (i, DOWN), modes) for i in range(2)),
    start=0,
)
S2 = Sz * Sz + 0.5 * (Splus * Sminus + Sminus * Splus)

hamiltonian = edb.Hamiltonian(Sdot, sector)
energies, _ = hamiltonian.eigsolve(k=None)
spin_squared = [
    float(np.real(hamiltonian.eigenstate(i).dag * S2 * hamiltonian.eigenstate(i)))
    for i in range(sector.dimension)
]

print("dimension:", sector.dimension)
print("energies:", np.round(energies, 8))
print("<S_tot^2>:", np.round(spin_squared, 8))
