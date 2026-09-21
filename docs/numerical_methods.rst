Numerical methods
=================

Occupation-number representation
--------------------------------

Fermionic Fock states are represented by integer occupation bit strings.
Creation and annihilation signs are evaluated from the parity of occupied modes
preceding the acted-on mode in the canonical ordering.

Fixed-population basis enumeration
----------------------------------

``FockBasis`` enumerates fixed-particle-number configurations using the standard
Gosper next-combination construction. Algorithmic provenance is recorded in
``PROVENANCE.md`` and in the ``FockBasis`` docstring.

Sparse matrix construction
--------------------------

The symbolic compiler lowers recognized number products, one-body hopping terms,
and number-conserving fermionic monomials to bit-mask intermediate
representations. Matrix entries are emitted directly in compressed
sparse-column order. NumPy-vectorized bit operations are used for mode counts up
to 64 when the operator structure admits a vectorized execution path; larger
mode sets use Python integers.

Precision
---------

Real Hamiltonians use ``numpy.float64``. Hamiltonians with nonzero imaginary
matrix elements use ``numpy.complex128``. Complex-typed coefficients with exactly
zero imaginary component do not force complex matrix storage.

Eigensolution
-------------

Sparse Hermitian eigensolution uses ``scipy.sparse.linalg.eigsh`` and therefore
ARPACK's implicitly restarted Lanczos method. Dense Hermitian eigensolution uses
``scipy.linalg.eigh``. Partial dense spectra use LAPACK index subsets when the
requested ordering permits it.

The default sparse tolerance is ``1e-10`` for the double-precision backend.
ARPACK controls ``tol``, ``maxiter``, ``ncv``, and ``v0`` are exposed directly.
EDinPy does not generate automatic warm starts.
