# Benchmarking

Benchmarks live in [`benchmarks/`](https://github.com/QuantumArtificer/edinpy/tree/main/benchmarks) and are separate from the unit-test suite. Tests establish numerical correctness and API behavior; benchmarks measure timing or memory for a specified physical workload and numerical environment.

The durable benchmark entry points are:

- `profile_fermion_execution.py` - fermion matrix-free execution modes across representative operator families
- `profile_boson_execution.py` - boson matrix-free execution modes on complete or symmetry-constrained sectors
- `profile_matrix_free.py` - explicit CSC versus matrix-free Hamiltonian workflows
- `profile_solver_memory.py` - ARPACK timing and memory as the Krylov subspace size changes
- `profile_basis_storage.py` - persistent basis storage and numerical execution views
- `profile_boson_sector_generation.py` - complete and symmetry-constrained bosonic sector construction

Each script provides `--help`. JSON output is written to `benchmarks/results/` when a simple filename is supplied; that directory is ignored by Git.

## Execution benchmarks

Matrix-free execution benchmarks measure repeated products

$$
y=Hx.
$$

The NumPy result is used as a numerical reference for the same input vector. Numba timings are taken after JIT warmup. Threaded results record the requested thread count and are compared with the optimized serial compiled path for the same operator family.

The operator portfolio includes diagonal terms, density interactions, real and complex hopping, extended-Hubbard terms, exchange and pair transfer, and generic number-conserving quartic monomials. The portfolio spans different amounts and types of per-state work, so the dependence on operator structure remains visible.

A compact reference result from this benchmark is presented in {doc}`../validation/performance`.

## Sparse versus matrix-free eigensolution

`profile_matrix_free.py` compares explicit CSC construction with matrix-free Hamiltonian action in an eigensolver calculation. The recorded quantities include sector dimension, matrix-construction time, solve time, peak memory, requested eigenpairs, and agreement of the converged eigenvalues.

Matrix-free execution removes sparse-matrix storage but not the $D$-component Ritz and Krylov vectors. A memory comparison therefore records the eigensolver configuration together with the Hamiltonian representation.

## Krylov-space memory

`profile_solver_memory.py` varies the ARPACK Krylov dimension `ncv`. A real double-precision vector contains approximately

$$
8D\ \mathrm{bytes},
$$

and a complex double-precision vector approximately

$$
16D\ \mathrm{bytes},
$$

before solver work arrays and orthogonalization storage are included. Timings and peak memory are recorded together because a larger Krylov subspace can improve convergence while increasing both storage and orthogonalization work.

## Basis construction and storage

`profile_basis_storage.py` measures persistent basis storage and temporary numerical views required for Hamiltonian action. `profile_boson_sector_generation.py` measures construction of complete and symmetry-constrained bosonic sectors.

Representation-boundary studies should include cases on both sides of the relevant storage transition. Comparisons between symmetry-constrained and unconstrained sectors must report both dimensions.

## Reproducible comparisons

Exact-diagonalization timings depend on Hilbert-space dimension, operator structure, requested eigenpairs, hardware, thread settings, Python, NumPy, SciPy, and optional Numba availability. A regression comparison should therefore keep the physical problem and numerical environment fixed and record at least

- EDinPy, Python, NumPy, SciPy, and Numba versions;
- sector definition and dimension;
- operator family or complete Hamiltonian;
- matrix representation and execution mode;
- thread count;
- eigensolver parameters when a solve is timed;
- warmup policy and repeat count.

Correctness precedes timing. A new performance path should first agree with an independent or simpler numerical route for the same represented Hamiltonian.
