# Hermitian eigensolvers

For a finite Hermitian Hamiltonian,

$$
H|\psi_n\rangle=E_n|\psi_n\rangle,
$$

with real eigenvalues $E_n$ and orthogonal eigenvectors belonging to distinct eigenvalues.

## Complete diagonalization

Dense Hermitian diagonalization returns the full spectrum and an orthonormal eigenbasis. It is the most direct choice when the sector is small or when an entire degenerate multiplet is required. Dense storage scales as $D^2$, and a complete diagonalization scales cubically with $D$ in the asymptotic dense-matrix limit.

## Krylov subspaces

For large sparse sectors, only a few extremal eigenpairs are usually required. Starting from a vector $|v_0\rangle$, a Krylov method constructs

$$
\mathcal K_m(H,v_0)
=\operatorname{span}\left\{
|v_0\rangle,
H|v_0\rangle,
\ldots,
H^{m-1}|v_0\rangle
\right\}.
$$

The Hamiltonian is projected into this much smaller subspace, whose Ritz values converge toward selected eigenvalues of the full problem. EDinPy uses SciPy's Hermitian ARPACK interface for partial spectra. ARPACK employs implicitly restarted Krylov methods designed to compute a small number of eigenpairs of large sparse or matrix-free problems.[^arpack]

The requested number of eigenpairs is `k`; `which` selects the part of the spectrum. The Krylov-space dimension `ncv` controls the number of basis vectors retained between restarts. A larger `ncv` can help separate clustered eigenvalues but increases memory use and orthogonalization work. `tol`, `maxiter`, and `v0` control convergence tolerance, iteration count, and the initial vector.

## Residuals and degeneracies

For a normalized approximate eigenvector $|\psi\rangle$ with Ritz value $E$, the residual

$$
r=\|H|\psi\rangle-E|\psi\rangle\|_2
$$

measures numerical convergence of that eigenpair. A small residual validates the eigenproblem for the represented finite Hamiltonian; it does not establish convergence with system size or other physical truncations.

Exactly or nearly degenerate levels require care. An iterative solver may return an arbitrary orthonormal basis within a degenerate eigenspace, and requesting fewer vectors than the multiplicity cannot recover the complete multiplet. Symmetry-constrained sectors or complete diagonalization of a small block are preferable when individual quantum numbers inside a degenerate manifold are required.

[^arpack]: R. B. Lehoucq, D. C. Sorensen, and C. Yang, "ARPACK Users' Guide: Solution of Large-Scale Eigenvalue Problems with Implicitly Restarted Arnoldi Methods," SIAM (1998), [doi:10.1137/1.9780898719628](https://doi.org/10.1137/1.9780898719628).
