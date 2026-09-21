# Benchmarking

The benchmark entry point is [`benchmarks/profile_operator_families.py`](https://github.com/QuantumArtificer/edinpy/blob/main/benchmarks/profile_operator_families.py). It measures sparse matrix construction for representative hopping, density, quartic, and mixed fermionic operators.

Run the default benchmark with

```bash
python benchmarks/profile_operator_families.py
```

The script reports the fixed-$N$ basis dimension, matrix nonzero count, basis construction time, Hamiltonian compilation time, and median matrix-construction time across repeated runs. Use `--json` to save the complete timing samples and software environment.

## Interpreting timings

Exact-diagonalization timings depend strongly on the Hilbert-space dimension and the structure of the Hamiltonian. They also depend on Python, NumPy, SciPy, the processor, and linked numerical libraries.

For a meaningful regression comparison, keep the following fixed:

- mode count and particle number
- Hamiltonian terms and boundary conditions
- Python, NumPy, and SciPy versions
- hardware and thread settings

The benchmark is intended to detect regressions and compare implementation changes on a controlled setup. It is not a hardware-independent estimate of the largest system EDinPy can solve.

See {doc}`../validation/fermion_performance` for the performance scope of the released package.
