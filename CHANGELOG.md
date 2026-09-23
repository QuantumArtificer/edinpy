# Changelog

All notable changes to EDinPy are documented here.

## [Unreleased]

### Added

- Lazy `NParticleSector` construction with an explicit `build()` step.
- `NParticleSector.project_particles()` for resolving conserved particle populations on one or more labeled degrees of freedom before basis generation.
- Direct one-, two-, and multi-degree-of-freedom constrained basis generators.

### Changed

- Constrained sectors are generated directly instead of constructing the complete fixed-$N$ basis and filtering it.
- Constrained basis generation above 64 modes uses a multiword `uint64` backend while preserving Python-integer Fock states in the public API.

## [0.2.0] - 2026-09-21

### Added

- Explicit fermionic mode definitions with `DoF` and `FermionModes`.
- Fixed-particle-number sectors with `NParticleSector` and basis-backed eigenstates.
- Literal creation, annihilation, number, sum, product, Hermitian-conjugation, and bra-ket algebra.
- Common fermionic operators for hopping, onsite terms, density interactions, Hubbard interactions, spin operators, Heisenberg exchange, and pair hopping.
- Sparse Hamiltonian construction with specialized execution paths for common number-conserving operators.
- Dense and sparse Hermitian eigensolvers through SciPy.
- Worked examples, validation tests, performance benchmarks, Sphinx documentation, and citation metadata.

### Changed

- Fermionic model state is owned explicitly by mode and sector objects instead of module-level global state.
- Real Hamiltonians use `float64`; Hamiltonians with nonzero complex matrix elements use `complex128`.
- Fermionic operators and states validate that they belong to compatible mode definitions.

### Notes

The bosonic module uses a separate API and currently has less validation and documentation coverage than the fermionic module.

[Unreleased]: https://github.com/QuantumArtificer/edinpy/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/QuantumArtificer/edinpy/releases/tag/v0.2.0
