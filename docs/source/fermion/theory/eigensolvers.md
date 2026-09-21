# Hermitian eigensolvers

`Hamiltonian.eigsolve()` solves Hermitian eigenproblems.

For a small number of extremal eigenpairs, EDinPy delegates to `scipy.sparse.linalg.eigsh`, SciPy's interface to ARPACK's implicitly restarted Lanczos method.[^arpack][^scipy] For complete spectra or dense partial spectra, EDinPy uses `scipy.linalg.eigh` and the LAPACK drivers selected by SciPy.

Lanczos methods are well suited to sparse many-body matrices when only a small part of the spectrum is required. They do not remove the memory cost of storing Krylov vectors. Exact or near degeneracies also require care because a converged single-vector Krylov run does not guarantee that every linearly independent vector in a degenerate eigenspace has been recovered.

The public controls `k`, `which`, `tol`, `maxiter`, `ncv`, and `v0` are passed to the corresponding SciPy routines where applicable. The eigensolver docstring documents their exact behavior.

[^arpack]: R. B. Lehoucq, D. C. Sorensen, and C. Yang, *ARPACK Users' Guide: Solution of Large-Scale Eigenvalue Problems with Implicitly Restarted Arnoldi Methods*, SIAM (1998), [doi:10.1137/1.9780898719628](https://doi.org/10.1137/1.9780898719628).
[^scipy]: P. Virtanen *et al.*, "SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python," *Nature Methods* **17**, 261-272 (2020), [doi:10.1038/s41592-019-0686-2](https://doi.org/10.1038/s41592-019-0686-2).
