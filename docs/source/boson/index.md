# Bosonic exact diagonalization

## Current status

Bosonic exact-diagonalization functionality is part of EDinPy. Its public interface predates the current fermionic API and does not yet use the same explicit ownership, basis-backed eigenstate, bra-ket, documentation, and benchmark conventions.

This section provides a package-level overview. The existing bosonic module remains available, but its present class names, global-state conventions, and workflow are provisional.

EDinPy remains one package with separate fermionic and bosonic namespaces. Numerical infrastructure can be shared where the mathematics and data structures are genuinely common.

## Design target

The design target for the provisional bosonic API follows the same package-level principles as the fermionic interface while keeping the two algebras distinct:

- explicit mode definitions and many-body bases,
- literal creation, annihilation, and number-operator expressions,
- explicit association between states, operators, and the relevant mode space,
- transparent sparse Hamiltonian construction,
- basis-aware many-body eigenstates,
- standard matrix-element expressions such as

$$
\langle\psi|\hat O|\psi\rangle
$$

without requiring a special-purpose observable API for every quantity.

Bosons also require design choices that do not arise for fermions. A single bosonic mode is not restricted to occupations zero and one, so practical finite calculations may require an explicit local occupation cutoff, a fixed-total-boson sector, or another finite Hilbert-space restriction. Those choices must be visible in the public API because they determine both the physics represented and the Hilbert-space dimension.

## Relationship to the fermionic interface

The fermionic implementation provides an architectural reference, but the bosonic algebra requires its own explicit objects. Eigensolvers, sparse matrices, diagnostics, state-vector operations, and validation may support shared internal infrastructure. Fermionic sign handling and bosonic occupation truncation remain statistics-specific.

Detailed bosonic guide, example, theory, API, and validation pages are omitted while the public interface remains provisional. A complete bosonic reference should cover the same practical layers as the fermionic documentation: setup, basis construction, operator algebra, Hamiltonian assembly, eigensolution, observables, numerical limits, and validation.

## What to use today

For calculations that require the fully documented reference interface, use {doc}`../fermion/index`. The existing bosonic module remains accessible through `edinpy.boson`, but its public interface may change.

Package-level testing and benchmark information lives in {doc}`../validation/index`.
