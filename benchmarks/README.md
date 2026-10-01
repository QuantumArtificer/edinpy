# Benchmarks

The benchmark scripts measure stable EDinPy behaviors: basis storage, operator
application, matrix-free solving, solver memory, and execution backends. They
write optional JSON output to `benchmarks/results/`, which is intentionally
ignored by Git.

Benchmarks are not part of the unit-test suite. Unit tests check numerical
correctness and API behavior; benchmarks are for explicit performance and
memory measurements on a chosen machine.

## Fermion execution

`profile_fermion_execution.py` compares the public matrix-free fermion
executors across representative operator families:

- diagonal onsite and density interactions;
- local and long-range hopping;
- dense one-body hopping;
- an extended-Hubbard workload;
- exchange and pair hopping;
- general number-conserving quartic terms;
- complex hopping.

Each family runs in a fresh process. The first matrix-vector product is treated
as warm-up and excluded from the steady-state median. Every measured result is
checked against the NumPy executor after timing.

```bash
python benchmarks/profile_fermion_execution.py \
  --modes 20 \
  --particles 10 \
  --executions numpy numba-serial numba-parallel \
  --threads 1 2 4 8 \
  --repeat 3 \
  --json fermion_execution_20_10.json
```

`--families` selects a subset of workloads. `--threads` applies only to
`numba-parallel`; values above Numba's configured maximum are skipped.

## Matrix-free solvers

`profile_matrix_free.py` compares explicit sparse-matrix and matrix-free solver
workflows. `profile_solver_memory.py` reports solver timing, Hamiltonian
application timing, and measured/estimated memory as the ARPACK subspace size
changes. Pure memory-planning behavior is covered by the unit tests rather than
a separate benchmark script.

Examples:

```bash
python benchmarks/profile_matrix_free.py
python benchmarks/profile_solver_memory.py --execution numba-serial
```

## Basis storage

`profile_basis_storage.py` measures persistent basis storage and construction
cost for representative sectors.

```bash
python benchmarks/profile_basis_storage.py
```

## Boson execution

`profile_boson_execution.py` compares the public matrix-free boson executors
across diagonal, one-boson hopping, pair transfer, quartic, complex-hopping,
and mixed workloads. Complete fixed-particle sectors are measured by default;
`--projected` selects a two-species symmetry-constrained sector with fixed species populations.

```bash
python benchmarks/profile_boson_execution.py \
  --modes 14 \
  --particles 7 \
  --executions numpy numba-serial numba-parallel \
  --threads 1 2 4 8 \
  --repeat 3 \
  --json boson_execution_14_7.json

python benchmarks/profile_boson_execution.py \
  --modes 14 \
  --particles 7 \
  --projected \
  --species-a 3 \
  --executions numpy numba-serial numba-parallel \
  --threads 1 2 4 8 \
  --repeat 3 \
  --json boson_execution_projected_14_7.json
```

`profile_boson_sector_generation.py` remains separate because it measures basis
construction of complete and symmetry-constrained sectors rather than Hamiltonian execution.
