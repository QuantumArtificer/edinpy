# Changelog

All notable changes to EDinPy are documented here.

## [Unreleased]

## [0.3.0] - 2026-10-01

### Added

- Explicit bosonic modes, fixed-particle-number sectors, symbolic Fock algebra, common lattice operators, sparse Hamiltonians, basis-backed states, and matrix-free eigensolvers under `edinpy.boson`.
- Symmetry-constrained fixed-particle-number sectors for separately conserved populations on labeled degrees of freedom.
- Matrix-free Hamiltonian execution through public `numpy`, `numba-serial`, and `numba-parallel` backends, with conservative `mixed` fallback behavior.
- Solver-memory planning and matrix-free solve diagnostics for iterative eigensolves.
- Compact basis storage and execution views for large complete and symmetry-constrained sectors.
- Unified fermionic and bosonic documentation with a physics-oriented user guide, worked examples, shared API reference, theory notes, and numerical validation.

### Changed

- Statistics-independent mode, state, Hamiltonian, and eigensolver infrastructure is shared between the fermionic and bosonic APIs.
- Symmetry-constrained sectors are generated directly from their conserved-number constraints instead of constructing a complete fixed-particle-number basis and filtering it.
- Complete fixed-particle-number fermion execution uses direct combinatorial ranking in compiled Numba kernels; bosonic compiled execution ranks weak compositions directly.
- Matrix-free serial and parallel execution reuse compiled operator/basis plans across repeated Hamiltonian applications.
- Benchmarks are organized around durable basis, execution, matrix-free, and solver-memory measurements rather than implementation-specific experiments.

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

In 0.2.0, the bosonic module still used the older separate API and had less validation and documentation coverage than the fermionic module.

[Unreleased]: https://github.com/QuantumArtificer/edinpy/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/QuantumArtificer/edinpy/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/QuantumArtificer/edinpy/releases/tag/v0.2.0
