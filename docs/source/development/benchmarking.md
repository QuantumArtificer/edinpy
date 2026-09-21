# Benchmarking

`benchmarks/profile_operator_families.py` provides representative matrix-construction workloads for the optimized fermionic kernels. Performance comparisons should use the same Hilbert-space sector, Hamiltonian terms, Python/NumPy/SciPy environment, and machine.

Record matrix dimension and nonzero count together with wall time. A raw timing without those quantities is difficult to compare across models because sparsity changes substantially with operator structure.
