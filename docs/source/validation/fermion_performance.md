# Fermionic performance benchmarks

EDinPy uses specialized sparse matrix-construction paths for several common number-conserving fermionic operator structures. The goal is to reduce the overhead of a literal Fock-algebra interface without changing the mathematical model written by the user.

The repository benchmark [`benchmarks/profile_operator_families.py`](https://github.com/QuantumArtificer/edinpy/blob/main/benchmarks/profile_operator_families.py) measures four representative operator families:

- nearest-neighbor hopping
- density-density interactions
- quartic number-conserving monomials
- a mixed hopping and density Hamiltonian

Run it with

```bash
python benchmarks/profile_operator_families.py
```

The script reports basis dimension, matrix nonzero count, basis construction time, compilation time, and repeated matrix-construction timings. Use `--json` to preserve the complete samples and software environment.

## What the benchmark does not claim

There is no single meaningful "EDinPy speed" or maximum system size. Exact-diagonalization cost depends on the basis dimension, the number and structure of Hamiltonian terms, matrix sparsity, the requested part of the spectrum, available memory, processor, and numerical-library build.

For this reason, the release documentation does not use one machine-specific timing as a performance guarantee. The included benchmark is the reproducible way to compare versions or implementation changes on a controlled setup.

Matrix construction is also only one part of the calculation. For larger sectors, storing sparse matrix data and the Krylov vectors used by iterative eigensolvers can dominate memory use. See {doc}`../fermion/user_guide/numerical_considerations` for the scaling discussion.
