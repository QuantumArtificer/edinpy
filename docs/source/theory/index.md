# Theory and numerical methods

Exact diagonalization represents a many-body Hamiltonian in a finite Fock-space sector and solves the resulting Hermitian eigenproblem. The mathematical structure is set by the particle statistics, the conserved quantum numbers used to block-diagonalize the Hamiltonian, and the ordered operator algebra. The numerical cost is then controlled primarily by the dimension of the selected sector and by whether the Hamiltonian is stored explicitly or applied as a linear operator.

The relevant ingredients are Fock-space dimensions and symmetry sectors, fermionic signs, sparse and matrix-free Hamiltonian representations, and Krylov eigensolution.

```{toctree}
:maxdepth: 2

fock_space
fermionic_signs
execution
eigensolvers
```
