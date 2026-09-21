# Glossary

```{glossary}
mode
: One fermionic single-particle label combination in `FermionModes`. A mode is represented by one occupation bit in a Fock state.
Fock state
: An occupation-number configuration of the fermionic modes.
Fock vector
: A ket represented by coefficients in the ordered Fock basis of a fixed-$N$ sector.
Fock bra
: Hermitian adjoint of a fermionic ket, used in literal products such as `psi.dag * O * psi`.
fixed-$N$ sector
: The subspace containing Fock states with one specified total particle number $N$.
literal algebra
: The user-facing representation of a second-quantized expression in `Creation`, `Annihilation`, `Number`, sums, and products before backend lowering.
lowering
: Conversion of a recognized symbolic operator structure into a compact execution representation used for sparse matrix construction.
CSC
: Compressed sparse column format, the sparse matrix layout emitted by EDinPy's Hamiltonian builder.
```
