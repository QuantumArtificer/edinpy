# Numerical considerations

Exact diagonalization reduces the many-body problem to a finite matrix. Its main practical limitation is the combinatorial growth of the matrix dimension and of the vectors used by iterative eigensolvers.

## Estimate the Hilbert-space size first

For $M$ fermionic modes and fixed total particle number $N$,

$$
D=\binom{M}{N}.
$$

For a half-filled spin-$1/2$ chain, $M=2L$ and $N=L$:

```python
from math import comb

for L in (4, 8, 12, 16):
    print(L, comb(2 * L, L))
```

```text
4 70
8 12870
12 2704156
16 601080390
```

A single `float64` vector of length 601,080,390 already requires about 4.8 GB before accounting for the sparse Hamiltonian, additional Krylov vectors, Python objects, or solver work arrays.

```{figure} ../../_static/figures/hilbert_space_growth.svg
:width: 80%
:alt: Hilbert space growth for spinful fermions

Full Fock-space and fixed-$N$ dimensions for a spinful chain. Fixing particle number removes many irrelevant states but does not remove exponential scaling.
```

## Sparse storage does not make the Hilbert space small

EDinPy constructs sparse CSC matrices. A four-site half-filled Hubbard Hamiltonian can be built compactly for this check:

```python
from edinpy import fermion as edf

L = 4
site = edf.DoF(L, name="site")
spin = edf.DoF(2, name="spin")
modes = edf.FermionModes(site, spin)
sector = edf.NParticleSector(modes, N=L).build()
c = edf.set_notation(edf.Annihilation, modes)
cd = edf.set_notation(edf.Creation, modes)
n = edf.set_notation(edf.Number, modes)

H = 0
for i in range(L - 1):
    for sigma in (0, 1):
        term = cd(i, sigma) * c(i + 1, sigma)
        H += -(term + term.dag)
for i in range(L):
    H += 4.0 * n(i, 0) * n(i, 1)

hamiltonian = edf.Hamiltonian(H, sector)
print("dimension:", sector.dimension)
print("dense entries:", sector.dimension**2)
print("sparse nnz:", hamiltonian.matrix.nnz)
```

```text
dimension: 70
dense entries: 4900
sparse nnz: 294
```

The storage reduction can be dramatic, but Lanczos vectors still have length $D$. Eventually the vectors, not the Hamiltonian entries, become the limiting object.

The same estimate should be repeated after applying any separately conserved particle populations. For a spin-$1/2$ system with $L$ sites and fixed $N_\uparrow$ and $N_\downarrow$,

$$
D=\binom{L}{N_\uparrow}\binom{L}{N_\downarrow}.
$$

With several projected degrees of freedom there is generally no single product formula because their mode groups overlap. `sector.dimension` gives the dimension of the jointly constrained basis after `build()`.


## Matrix construction and eigensolution are separate costs

The symbolic compiler lowers recognized operator structures to sparse execution kernels before matrix construction. Number products, one-body hopping, and number-conserving fermionic monomials have optimized paths.

Matrix construction can become fast enough that the eigensolver dominates the wall time. Benchmark the two stages separately because faster operator construction does not necessarily reduce total runtime.

The repository benchmark

```bash
python benchmarks/profile_operator_families.py
```

A representative local run has the form

```text
L=16, N=8, Hilbert dimension=12870
hopping  basis=0.002541 s  matrix=0.010539 s  nnz=  102960  kernels={'number_product': 0, 'hopping': 30, 'hopping_groups': 15, 'monomial': 0, 'generic': 0}
density  basis=0.002480 s  matrix=0.001410 s  nnz=   12861  kernels={'number_product': 15, 'hopping': 0, 'hopping_groups': 0, 'monomial': 0, 'generic': 0}
quartic  basis=0.002368 s  matrix=0.003093 s  nnz=   12936  kernels={'number_product': 0, 'hopping': 0, 'hopping_groups': 0, 'monomial': 14, 'generic': 0}
mixed    basis=0.002465 s  matrix=0.032573 s  nnz=  115821  kernels={'number_product': 15, 'hopping': 30, 'hopping_groups': 15, 'monomial': 0, 'generic': 0}
```

The timings are examples, not performance guarantees. They depend on the processor, Python/NumPy/SciPy build, Hilbert-space dimension, Hamiltonian sparsity, and operator structure. The useful invariant is the report format: basis construction, matrix construction, nonzero count, and selected kernel families are shown separately.

## Vectorized and large-mode paths

For mode counts up to 64, selected Hamiltonian execution kernels use NumPy `uint64` vectorization. Larger mode spaces remain valid because the public Fock representation uses arbitrary-precision Python integers when necessary.

Constrained sector construction uses a separate multiword backend above 64 modes. Occupation masks are assembled as arrays of 64-bit words and converted to the ordinary integer Fock representation only after the requested basis has been generated and sorted. This keeps particle-projected basis construction vectorized across the 64-mode boundary without changing the public state representation.

A large number of modes does not necessarily imply a large fixed-$N$ sector. For example, a dilute few-particle problem can have many modes and a manageable basis. EDinPy therefore does not impose a 64-mode hard limit. Some Hamiltonian execution kernels still use different implementations above 64 modes, so basis-construction and matrix-construction timings should be considered separately.

## Precision

Real Hamiltonians use `float64`. Genuinely complex Hamiltonians use `complex128`.

```python
print(hamiltonian.matrix.dtype)
```

```text
float64
```

A Peierls phase or another genuinely complex term changes the matrix dtype to `complex128`. These matrix and eigensolver paths use double precision, not single-precision storage.

## Sparse versus dense diagonalization

Use sparse diagonalization when only a small number of extremal eigenpairs are needed:

```python
energies, vectors = hamiltonian.eigsolve(k=6, which="SA")
print(energies.shape, vectors.shape)
```

```text
(6,) (70, 6)
```

Use a complete dense eigensystem only when the matrix is small enough and the scientific question requires it:

```python
energies, vectors = hamiltonian.eigsolve(k=None)
print(energies.shape, vectors.shape)
```

```text
(70,) (70, 70)
```

The latter scales much more steeply in both memory and runtime.

## Reproducibility checklist

For a published calculation, record at least:

- EDinPy version and commit or archived release
- Python, NumPy, and SciPy versions
- mode ordering and all discrete degrees of freedom
- total particle number, any particle-number projections, and Hilbert-space dimension
- boundary conditions and Hamiltonian parameters
- whether the Hamiltonian is real or complex
- sparse/dense eigensolver choice and `k`, `which`, `tol`, `ncv`, and `maxiter` where relevant
- finite-size, basis-truncation, or model-parameter convergence checks relevant to the conclusion.

The numerical result is reproducible only when the finite problem being diagonalized is specified as carefully as the solver settings.
