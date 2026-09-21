# Benchmarks

The benchmark script measures sparse Hamiltonian construction for several common fermionic operator families. It is intended for regression testing and local comparisons, not as a hardware-independent performance claim.

Run it from an installed development checkout:

```bash
python benchmarks/profile_operator_families.py
```

A smaller run is useful while developing:

```bash
python benchmarks/profile_operator_families.py \
    --length 12 --particles 6 --repeat 3
```

Machine-readable results can be saved with `--json`:

```bash
python benchmarks/profile_operator_families.py \
    --json benchmark-results.json
```

## What is measured

For each operator family the script reports:

- fixed-particle-number basis construction time
- symbolic expression construction time in the JSON output
- Hamiltonian compilation time
- median sparse matrix construction time across repeated runs
- matrix dimension and number of nonzero entries

The default families cover nearest-neighbor hopping, density-density interactions, quartic number-conserving monomials, and a mixed Hamiltonian.

## Comparing runs

Compare timings only when the model, particle sector, Python version, NumPy and SciPy versions, and hardware are comparable. Exact diagonalization costs change rapidly with Hilbert-space dimension and matrix sparsity. A timing from one model or machine should not be treated as a general speed estimate.

For profiling, use Python's standard profiler on the same entry point, for example:

```bash
python -m cProfile -o matrix.prof \
    benchmarks/profile_operator_families.py \
    --length 16 --particles 8 --families mixed --repeat 1
```

Profiler output is local development data and is excluded from the repository.
