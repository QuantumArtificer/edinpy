# Fermionic performance benchmarks

The repository benchmark `benchmarks/profile_operator_families.py` measures representative sparse matrix-construction workloads for hopping, density interactions, quartic fermionic monomials, and mixed Hamiltonians. Basis construction and matrix construction are timed separately, and the output records the Hilbert-space dimension, matrix nonzero count, and selected execution kernels.

The optimized fermionic backend was developed with fixed-size regression benchmarks, including a 22-mode, 11-particle sector with dimension 705432. These runs were used to detect performance regressions while preserving the literal public operator algebra. They are not cross-machine performance guarantees.

Runtime depends on the processor, Python/NumPy/SciPy build, Hilbert-space dimension, Hamiltonian sparsity, and operator structure. The benchmark script is the reproducible entry point for measuring performance on the machine used for a calculation.

Matrix construction is only one part of an exact-diagonalization workflow. For sufficiently large sectors, memory use and iterative eigensolution can dominate even when sparse matrix construction is fast. See {doc}`../fermion/user_guide/numerical_considerations` for the scaling discussion.
