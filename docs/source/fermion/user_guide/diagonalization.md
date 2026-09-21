# Diagonalization

`Hamiltonian.eigsolve()` provides one Hermitian-eigensolver interface for both sparse low-energy calculations and complete dense spectra. The choice should follow the scientific question: use sparse methods when only a few extremal states are needed, and dense methods when the entire finite spectrum or a complete degenerate subspace is important.

The snippets below assume the two-site Hubbard Hamiltonian from {doc}`../getting_started`.

## A few low-energy eigenpairs

```python
import numpy as np

energies, vectors = hamiltonian.eigsolve(
    k=2,
    which="SA",
    tol=1e-10,
)

print(np.where(np.abs(energies) < 1e-12, 0.0, energies))
print(vectors.shape)
```

```text
[-0.82842712  0.        ]
(6, 2)
```

`which="SA"` means *smallest algebraic*, following `scipy.sparse.linalg.eigsh`. The other supported choices are `LA`, `SM`, and `LM`.

When `sparse=True` and `k` is smaller than the matrix dimension, EDinPy delegates to SciPy's ARPACK interface.

## Complete finite spectrum

For a small Hilbert space, request every eigenpair with `k=None`:

```python
energies, vectors = hamiltonian.eigsolve(k=None)
print(energies)
print(vectors.shape)
```

```text
[-0.82842712  0.          0.          0.          4.          4.82842712]
(6, 6)
```

This route uses dense Hermitian diagonalization. It costs much more memory and time as the basis grows, but it returns the complete finite-dimensional eigensystem.

## Recover basis-aware eigenstates

The NumPy columns are useful for interoperability with SciPy. For Fock algebra, recover the same states as `FockVector` objects:

```python
psi0 = hamiltonian.eigenstate(0)
psi1 = hamiltonian.eigenstate(1)

print(psi0.norm())
print(psi0.dag * psi1)
```

Typical output is

```text
0.9999999999999999
0.0
```

The last few digits of norms and overlaps can vary at roundoff level. `eigenstate(i)` always refers to the ordering returned by the most recent call to `eigsolve()`.

All returned eigenstates can be converted at once:

```python
states = hamiltonian.eigenstates()
print(len(states))
```

```text
6
```

## Residuals

A residual tests whether the numerical vector solves the matrix eigenproblem:

```python
import numpy as np

residuals = []
for i, energy in enumerate(energies):
    residual = np.linalg.norm(
        hamiltonian.matrix @ vectors[:, i]
        - energy * vectors[:, i]
    )
    residuals.append(residual)

print(max(residuals) < 1e-10)
```

```text
True
```

This is a numerical diagnostic only. A tiny residual does not validate the physical model, the finite-size extrapolation, or a truncation made before the Hamiltonian reached EDinPy.

## ARPACK controls

The sparse interface exposes the main SciPy controls directly:

```python
energies, vectors = hamiltonian.eigsolve(
    k=2,
    which="SA",
    tol=1e-12,
    maxiter=5000,
    ncv=5,
)
print(np.where(np.abs(energies) < 1e-12, 0.0, energies))
```

```text
[-0.82842712  0.        ]
```

`ncv` controls the Krylov-subspace dimension. Increasing it can help when targeted eigenvalues are clustered, at the cost of additional memory and orthogonalization work. `v0` can also be supplied explicitly when a physically motivated initial vector is available.

## Exact degeneracies

ARPACK starts from one vector. Converged Ritz residuals do not guarantee that every linearly independent vector in an exactly degenerate eigenspace has been returned. The Hubbard dimer makes that distinction visible: the spectrum contains a threefold zero-energy triplet manifold.

For a small system where the complete degenerate subspace matters, use dense diagonalization:

```python
energies, vectors = hamiltonian.eigsolve(k=None)
print(np.count_nonzero(np.isclose(energies, 0.0)))
```

```text
3
```

For larger systems, resolving additional symmetries into smaller blocks is usually preferable. EDinPy 0.2.0 currently provides only the fixed-total-particle-number sector.

## Hermiticity checks

`eigsolve()` checks Hermiticity by default. For a newly assembled model, keep that check enabled:

```python
print(hamiltonian.is_hermitian())
```

```text
True
```

Disabling `check_hermitian` only skips the validation step. It does not make a non-Hermitian matrix valid input to the present eigensolver interface.
