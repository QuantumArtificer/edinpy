# Sparse and matrix-free Hamiltonians

In a basis $\{|\alpha\rangle\}_{\alpha=1}^{D}$, the Hamiltonian matrix is

$$
H_{\alpha\beta}=\langle\alpha|H|\beta\rangle.
$$

Local lattice Hamiltonians are usually sparse in the occupation basis because a single operator term connects each basis state to only a small subset of the full sector.

## Explicit sparse matrix

An explicit sparse representation stores the nonzero matrix elements and their indices. EDinPy uses compressed sparse column (CSC) storage. If $N_{\mathrm{nz}}$ matrix elements are nonzero, sparse storage scales with $N_{\mathrm{nz}}$ rather than $D^2$.

Materializing the sparse matrix is useful when matrix elements must be inspected, when the same matrix is reused by several numerical routines, or when an external method requires an explicit sparse object.

## Matrix-free action

Krylov eigensolvers do not require the individual entries of $H$; they require repeated products

$$
|y\rangle=H|x\rangle,
$$

or, in components,

$$
y_\alpha=\sum_\beta H_{\alpha\beta}x_\beta.
$$

A matrix-free representation evaluates this action directly from the operator terms and basis states without storing the complete sparse matrix. The eigenvectors and Krylov vectors still have length $D$, so matrix-free execution removes matrix storage rather than the Hilbert-space cost itself.

## Operator evaluation

Second-quantized sums and products are reduced to elementary actions on occupation states before repeated numerical application. The public matrix-free execution modes select NumPy or compiled Numba implementations of the same Hamiltonian action. Numerical equivalence between these paths is tested independently; the appropriate choice is therefore determined by problem size, operator structure, compilation overhead, and available memory rather than by a different physical approximation.

The practical selection between explicit sparse and matrix-free forms is discussed in {doc}`../user_guide/numerical_considerations`.
