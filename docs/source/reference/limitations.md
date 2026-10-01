# Current limitations

## Symmetry sectors

The public sector builders support fixed total particle number and additional particle-number constraints associated with labeled degrees of freedom. Translation, crystal momentum, reflection, point-group, fermion-parity, and non-Abelian symmetry sectors are not part of the current API.

## Number-changing Hamiltonians

Creation and annihilation products can act on individual Fock states even when they change total particle number. A `Hamiltonian` constructed on one `NParticleSector` contains only matrix elements within that sector.

## Hermitian eigensolution

`Hamiltonian.eigsolve()` solves Hermitian eigenproblems. General non-Hermitian eigensolvers are not part of the current solver interface.

## Dynamics and finite temperature

The current API covers static finite-system Hamiltonians, eigenstates, and operator expectation values. Time evolution, thermal traces, spectral functions, and finite-temperature Lanczos methods are not included.

## Practical system-size limit

Exact diagonalization requires at least several vectors of length equal to the symmetry-sector dimension. When the reduced sector is too large for those vectors or for a converged Krylov calculation, changing from sparse to matrix-free Hamiltonian storage does not remove the fundamental Hilbert-space limit. Larger-system methods such as tensor-network approaches, quantum Monte Carlo where applicable, or other controlled approximations are then more appropriate for thermodynamic questions.

## Finite-size interpretation

Exact diagonalization is exact for the represented finite Hamiltonian up to floating-point error and eigensolver tolerance. Thermodynamic phases and critical points require finite-size analysis beyond a single cluster calculation. Different observables can converge at different rates, and boundary conditions can produce shell or edge effects that remain visible after the ground-state energy appears smooth.
