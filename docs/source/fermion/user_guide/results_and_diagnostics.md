# Results, observables, and diagnostics

Exact diagonalization solves the specified finite matrix to numerical precision. A useful analysis separates three levels:

1. Does the matrix represent the intended Hamiltonian?
2. Are the requested eigenpairs converged?
3. Do the observables address the physical question?

The examples below continue with the two-site Hubbard model from {doc}`../getting_started`.

## Inspect the matrix before interpreting it

```python
A = hamiltonian.matrix

print("shape:", A.shape)
print("nnz:", A.nnz)
print("dtype:", A.dtype)
print("Hermitian:", hamiltonian.is_hermitian())
```

```text
shape: (6, 6)
nnz: 10
dtype: float64
Hermitian: True
```

The shape must match `sector.dimension`. The number of nonzero entries is a useful structural diagnostic: unexpected density can reveal an indexing error or an interaction that connects more basis states than intended, while an unexpectedly diagonal matrix can reveal a missing hopping or exchange term.

## Check the eigensolver residual

```python
import numpy as np

energies, vectors = hamiltonian.eigsolve(k=None)

checks = []
for i, energy in enumerate(energies):
    residual = np.linalg.norm(A @ vectors[:, i] - energy * vectors[:, i])
    checks.append(residual < 1e-10)

print(checks)
```

```text
[True, True, True, True, True, True]
```

Residuals validate the numerical eigensolution. They do not validate the Hamiltonian as a physical model.

## Work with eigenstates in the Fock basis

`eigsolve()` returns NumPy arrays so the result remains compatible with the SciPy ecosystem. The same columns are available as basis-aware kets:

```python
psi0 = hamiltonian.eigenstate(0)
psi1 = hamiltonian.eigenstate(1)

print(np.allclose(psi0.coefficients, vectors[:, 0]))
print(psi0.dag * psi0)
print(psi0.dag * psi1)
```

```text
True
0.9999999999999998
0.0
```

The second and third lines are literal Dirac products. `psi0.inner(psi1)` is the equivalent method form of `psi0.dag * psi1`.

## Expectation values are ordinary Fock algebra

For the Hubbard dimer, define the total double occupancy

```python
D = n(0, UP) * n(0, DOWN) + n(1, UP) * n(1, DOWN)
print(D)
```

```text
n[0,0] n[0,1] + n[1,0] n[1,1]
```

Then evaluate

```python
print(psi0.dag * D * psi0)
```

```text
0.1464466094067262
```

The equivalent explicit-state form is

```python
print(psi0.inner(D * psi0))
```

```text
0.1464466094067262
```

The first form follows the standard notation $\langle\psi_0|D|\psi_0\rangle$. For basis-backed vectors in one fixed-$N$ sector, it also uses the optimized path. The operator is compiled to its sparse representation and contracted with the coefficient vectors.

## Local densities

Observables use the same operator algebra as Hamiltonian terms:

```python
n0 = n(0, UP) + n(0, DOWN)
n1 = n(1, UP) + n(1, DOWN)

print("<n0> =", psi0.dag * n0 * psi0)
print("<n1> =", psi0.dag * n1 * psi0)
```

```text
<n0> = 1.0000000000000002
<n1> = 0.9999999999999996
```

The symmetric dimer is half filled, so each site carries one particle on average.

## Local moments and double occupancy

One-point densities are often not enough to distinguish different correlated states. For the half-filled dimer, two useful local diagnostics are the total double occupancy

$$
D=\sum_i n_{i\uparrow}n_{i\downarrow}
$$

and the site-averaged local moment

$$
\mu^2=\frac{1}{L}\sum_i
\left\langle(n_{i\uparrow}-n_{i\downarrow})^2\right\rangle.
$$

```python
D = sum(
    (n(i, UP) * n(i, DOWN) for i in range(2)),
    start=0,
)
local_moment = 0.5 * sum(
    (n(i, UP) - n(i, DOWN)) * (n(i, UP) - n(i, DOWN))
    for i in range(2)
)

print("<D> =", psi0.dag * D * psi0)
print("mu^2 =", psi0.dag * local_moment * psi0)
```

```text
<D> = 0.1464466094067262
mu^2 = 0.8535533905932736
```

For this half-filled dimer, $\mu^2=1-\langle D\rangle$. The two observables therefore make the same redistribution of wavefunction weight visible from complementary charge and spin perspectives.

## Real-space correlation functions

A correlation function is an expectation value of an operator product. Define the local spin projection

$$
S_i^z=\frac{1}{2}(n_{i\uparrow}-n_{i\downarrow})
$$

and the connected charge operator at half filling

$$
\delta n_i=n_i-1,
\qquad
n_i=n_{i\uparrow}+n_{i\downarrow}.
$$

```python
Sz = [
    edf.SpinZ((i, UP), (i, DOWN), modes)
    for i in range(2)
]
delta_n = [
    n(i, UP) + n(i, DOWN) - 1
    for i in range(2)
]
```

For a translationally invariant system, it is convenient to average over equivalent origins,

$$
C_s(r)=\frac{1}{L}\sum_i
\langle S_i^zS_{i+r}^z\rangle,
$$

$$
C_c(r)=\frac{1}{L}\sum_i
\langle\delta n_i\,\delta n_{i+r}\rangle.
$$

For the dimer:

```python
for r in (0, 1):
    Cs = sum(
        np.real(psi0.dag * Sz[i] * Sz[(i + r) % 2] * psi0)
        for i in range(2)
    ) / 2
    Cc = sum(
        np.real(psi0.dag * delta_n[i] * delta_n[(i + r) % 2] * psi0)
        for i in range(2)
    ) / 2
    print(r, Cs, Cc)
```

```text
0 0.2133883476483184 0.1464466094067262
1 -0.2133883476483184 -0.1464466094067262
```

The negative nearest-neighbor spin correlation is the finite-dimer signature of antiferromagnetic alignment. The negative connected charge correlation reflects the fact that a doublon on one site is accompanied by a hole on the other.

For larger lattices the same expressions produce full correlation profiles. The six-site Hubbard example plots both quantities as functions of separation:

```{figure} ../../_static/figures/hubbard_chain_correlations.svg
:width: 100%
:alt: Spin and connected charge correlations of a six-site Hubbard ring

Translationally averaged spin and connected charge correlations of the half-filled six-site Hubbard ring. Increasing repulsion suppresses charge fluctuations while strengthening the alternating spin correlations.
```

## Structure factors from the same algebra

Real-space correlations are often summarized in momentum space. Given any set of local operators $O_j$, define

$$
O_q=\sum_{j=0}^{L-1}e^{-iqj}O_j
$$

and the equal-time structure factor

$$
S_O(q)=\frac{1}{L}\langle O_q^\dagger O_q\rangle.
$$

The Fourier-space operator can be written directly in EDinPy:

```python
def structure_factor(psi, local_operators, q):
    L = len(local_operators)
    Oq = sum(
        (
            np.exp(-1j * q * j) * operator
            for j, operator in enumerate(local_operators)
        ),
        start=0,
    )
    return np.real(psi.dag * Oq.dag * Oq * psi) / L
```

For the two-site dimer:

```python
pair = [c(i, DOWN) * c(i, UP) for i in range(2)]

print("S_s(pi) =", structure_factor(psi0, Sz, np.pi))
print("S_c(pi) =", structure_factor(psi0, delta_n, np.pi))
print("P(0) =", structure_factor(psi0, pair, 0.0))
```

```text
S_s(pi) = 0.4267766952966368
S_c(pi) = 0.2928932188134524
P(0) = 0.1464466094067262
```

The first two quantities collect the staggered spin and charge fluctuations at $q=\pi$. The third uses the local pair-annihilation operator

$$
\Delta_j=c_{j\downarrow}c_{j\uparrow}
$$

and measures the zero-momentum on-site pair structure factor. Nothing in the helper function is specific to spin, charge, or pairing: changing `local_operators` changes the physical observable.

On a larger periodic chain, the complete momentum dependence is often more informative than a single special wavevector:

```{figure} ../../_static/figures/hubbard_chain_structure_factors.svg
:width: 100%
:alt: Spin charge and pair structure factors of a six-site Hubbard ring

Momentum-resolved spin, charge, and on-site pair structure factors for a half-filled six-site Hubbard ring. The discrete momenta are fixed by the finite periodic lattice. The curves diagnose the finite system and do not imply a thermodynamic singularity.
```

The same construction applies to bond-order, orbital, excitonic, and other composite observables. If the desired order parameter can be written as a Fock operator, its correlations can be evaluated without a model-specific observable API.

## Transition matrix elements

The bra-ket algebra is not restricted to expectation values. Once more than one eigenstate has been retained, transition amplitudes have the same syntax:

```python
psi1 = hamiltonian.eigenstate(1)
matrix_element = psi1.dag * Sz[0] * psi0
print(matrix_element)
```

The value of a transition matrix element depends on the basis chosen inside exactly degenerate eigenspaces, so individual amplitudes should be interpreted with the relevant symmetries in mind. The algebra is the same as the textbook expression $\langle\psi_1|O|\psi_0\rangle$.

## Number-changing operators remain meaningful

`O * psi` is a literal state action. It can therefore leave the original fixed-$N$ sector:

```python
removed = c(0, UP) * psi0
print(type(removed).__name__)
```

```text
StateSum
```

The resulting object is a state in the $(N-1)$ Fock sector represented by EDinPy's symbolic state algebra. Its expectation value in the original fixed-$N$ state vanishes:

```python
print(psi0.dag * c(0, UP) * psi0)
```

```text
0.0
```

This distinction is important: fixed-$N$ Hamiltonian matrices project onto one sector, but the operator-state algebra itself is not restricted to number-conserving actions.

## Use observables to interpret spectra

A spectrum identifies changes in the low-energy levels. Observables help determine the character of the corresponding states. The worked examples therefore pair eigensystems with quantities such as double occupancy, spin correlations, and staggered charge and spin fluctuations. See {doc}`../examples/hubbard_dimer`, {doc}`../examples/hubbard_chain`, and {doc}`../examples/extended_hubbard_chain`.

## Finite-size interpretation

An ED result is exact for the finite Hilbert space that was diagonalized. It is not automatically converged in system size, boundary conditions, local-state truncation, or effective-model parameters. When a scientific conclusion depends on those choices, repeat the calculation across the relevant finite-size or truncation sequence and report the convergence criterion alongside the observable.
