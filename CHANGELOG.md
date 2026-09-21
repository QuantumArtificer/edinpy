# Changelog

## 0.1.0

- Replaced implicit global fermionic state with explicit `FermionModes` and `NParticleSector` ownership.
- Added user-defined primitive-operator notation through `set_notation`.
- Split the fermionic implementation into mode indexing, Fock basis, symbolic algebra, compiler, intermediate representation, execution, Hamiltonian, and common-operator modules.
- Added structural lowering and specialized sparse CSC emitters for hopping, number products, and number-conserving fermionic monomials.
- Standardized matrix precision to `float64` for real Hamiltonians and `complex128` for genuinely complex Hamiltonians.
- Added dense partial eigensolution and configurable ARPACK controls.
- Added transparent common operators for hopping, onsite energies, density interactions, Hubbard interactions, spin operators, Heisenberg exchange, and pair hopping.
- Added provenance records, scientific references, release tests, and repository hygiene checks.
