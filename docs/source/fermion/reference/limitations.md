# Current limitations

## Symmetry sectors

Version 0.1.0 constructs fermionic Hamiltonians in a fixed total-particle-number sector. Full Fock-space, fermion-parity, translation, momentum, fixed-spin-population, and other symmetry-reduced sectors are not yet implemented.

## Number-changing Hamiltonian terms

Creation and annihilation products that change total particle number can act on individual `FockState` objects. Their matrix elements connect different $N$ sectors and therefore are not represented by a `Hamiltonian` constructed in one `NParticleSector`.

## Hermitian problems

`Hamiltonian.eigsolve()` is a Hermitian eigensolver interface. General non-Hermitian diagonalization is outside the current API.

## Observables and finite-temperature calculations

The current release supports user-defined zero-temperature observables that can be expressed with the fermionic operator-state algebra. This includes expectation values, transition matrix elements, correlations, and structure factors. The package does not provide a separate high-level observable catalogue, thermal trace machinery, spectral functions, time evolution, or finite-temperature Lanczos methods.

## Bosonic implementation

The existing bosonic module predates the current fermionic architecture and is not covered by the 0.1.0 fermionic user guide. Its public interface is therefore documented separately from the current fermionic API.
