# Glossary

```{glossary}
mode
: One ordered single-particle label combination in `FermionModes` or `BosonModes`.
Fock state
: One occupation-number basis state. Fermionic occupations are 0 or 1; bosonic occupations are non-negative integers.
Fock basis
: Ordered occupation-number basis spanning a finite many-body sector.
Fock vector
: A ket represented by coefficients in the ordered basis of a built sector.
Fock bra
: Hermitian adjoint of a Fock-space ket, used in expressions such as `psi.dag * O * psi`.
fixed-particle-number sector
: Subspace with a fixed eigenvalue $N$ of the total number operator.
symmetry-constrained sector
: A fixed-particle-number sector with additional fixed eigenvalues of commuting number operators, such as $N_\uparrow$ and $N_\downarrow$. Such constraints are specified with `NParticleSector.project_particles(...)` before `build()`.
operator expression
: A second-quantized sum or ordered product constructed from creation, annihilation, number, and scalar operators.
CSC
: Compressed sparse column format used by the explicit sparse Hamiltonian matrix.
matrix-free eigensolve
: Partial eigensolution in which ARPACK receives the action of $H$ on vectors through a `LinearOperator` instead of an explicitly stored Hamiltonian matrix.
natural occupations
: Eigenvalues of the one-body density matrix $\rho^{(1)}_{ij}=\langle a_i^\dagger a_j\rangle$.
```
