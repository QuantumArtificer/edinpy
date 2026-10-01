# Performance validation

Performance measurements are interpreted only after numerical equivalence has been established for the same Hamiltonian action. Absolute times depend on hardware and software environment, while scaling with sector size, operator structure, and thread count determines whether a faster execution mode is useful for a particular calculation.

## Matrix-free thread scaling

The fermionic matrix-free benchmark applies eight representative operator families to fixed-particle-number sectors and compares `numba-parallel` with the optimized `numba-serial` path. The plotted quantity is the geometric mean speedup

$$
S_p=
\left[
\prod_{f=1}^{N_f}
\frac{T_{f,\mathrm{serial}}}{T_{f,p}}
\right]^{1/N_f},
$$

where $f$ labels the $N_f=8$ operator families and $p$ is the requested thread count. Each family is checked numerically against the NumPy Hamiltonian action before its timing contributes to the result.

```{figure} ../_static/figures/validation_fermion_parallel_scaling.svg
:width: 76%
:alt: Geometric mean matrix-free fermion speedup versus Numba thread count for two Hilbert-space dimensions

Geometric mean speedup of matrix-free fermion Hamiltonian application relative to `numba-serial`. The smaller sector reaches its best mean result at four threads, while the larger sector continues to benefit through eight threads.
```

For $D=184{,}756$, the geometric mean speedup is $1.45$ at four threads and falls to $1.07$ at eight threads. For $D=2{,}704{,}156$, it increases from $1.88$ at two threads to $2.86$ at four threads and $3.37$ at eight threads. The larger sector provides enough work per Hamiltonian application to amortize threading overhead; the moderate sector does not benefit uniformly from the largest thread count.

The reference measurement used Python 3.12.9, NumPy 1.26.4, SciPy 1.15.1, Numba 0.68.0, EDinPy 0.3.0.dev0, and a maximum Numba thread count of eight. Each operator family was measured in an isolated process with three repeats after JIT warmup. The compact numerical snapshot used for the figure is stored in `docs/data/fermion_parallel_reference.json`.

The useful thread count depends on sector size and operator structure. If Hamiltonian application is a significant fraction of a production solve, the target workload should be timed directly.

## Other performance dimensions

Sparse-matrix construction, matrix-free eigensolution, Krylov-space memory, basis storage, and bosonic sector construction have dedicated benchmark entry points under `benchmarks/`. Their results are machine dependent and should be generated on the target environment. The benchmark protocols and commands are documented in {doc}`../development/benchmarking`.
