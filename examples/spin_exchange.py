"""Two-site fermionic Heisenberg exchange and total-spin diagnostics."""

import numpy as np

from edinpy import fermion as edf

L = 2
N = 2
UP, DOWN = 0, 1
J = 1.0

site = edf.DoF(L, name="site")
spin = edf.DoF(2, name="spin")
modes = edf.FermionModes(site, spin)
sector = edf.NParticleSector(modes, N=N)

Sdot = edf.HeisenbergExchange(
    (0, UP), (0, DOWN),
    (1, UP), (1, DOWN),
    J=1.0,
    modes=modes,
)
H = J * Sdot

Splus = sum(
    (edf.SpinPlus((i, UP), (i, DOWN), modes) for i in range(L)),
    start=0,
)
Sminus = sum(
    (edf.SpinMinus((i, UP), (i, DOWN), modes) for i in range(L)),
    start=0,
)
Sz = sum(
    (edf.SpinZ((i, UP), (i, DOWN), modes) for i in range(L)),
    start=0,
)
S2 = Sz * Sz + 0.5 * (Splus * Sminus + Sminus * Splus)

hamiltonian = edf.Hamiltonian(H, sector)
energies, _ = hamiltonian.eigsolve(k=None)
psi0 = hamiltonian.eigenstate(0)

print("Hamiltonian:")
print(H)
print("\nenergies:", np.round(energies, 8))
print("<S0.S1> ground state:", np.real(psi0.dag * Sdot * psi0))
print("<S_tot^2> eigenstates:", [
    round(
        float(
            np.real(
                hamiltonian.eigenstate(i).dag
                * S2
                * hamiltonian.eigenstate(i)
            )
        ),
        8,
    )
    for i in range(len(energies))
])
