# EDinPy provenance and algorithmic references

This file records published algorithms and external numerical libraries that are directly relevant to EDinPy's implementation. It complements the scientific references in the documentation.

## Fermionic fixed-particle-number basis

EDinPy represents fermionic occupation configurations as integer bit strings. A fixed-$N$ basis is enumerated with the standard next-integer-with-the-same-popcount construction commonly associated with Gosper's hack.

The method is described in HAKMEM Item 175 and in later references on combinatorial bit operations:

1. M. Beeler, R. W. Gosper, and R. Schroeppel, *HAKMEM*, MIT Artificial Intelligence Laboratory Memo AIM-239, Item 175 (1972).
2. D. E. Knuth, *The Art of Computer Programming*, Vol. 4A, Sec. 7.1.3, Addison-Wesley (2011).
3. H. S. Warren, Jr., *Hacker's Delight*, 2nd ed., Sec. 2-1, Addison-Wesley (2013).
4. S. E. Anderson, [Bit Twiddling Hacks](https://graphics.stanford.edu/~seander/bithacks.html).

## Bosonic basis enumeration

The lexicographic enumeration used by the bosonic basis routines follows the construction described by Zhang and Dong:

5. J. M. Zhang and R. X. Dong, "Exact diagonalization: the Bose-Hubbard model as an example," *European Journal of Physics* **31**, 591-602 (2010), [doi:10.1088/0143-0807/31/3/016](https://doi.org/10.1088/0143-0807/31/3/016), [arXiv:1102.4006](https://arxiv.org/abs/1102.4006).

## Fermionic bit strings and signs

Encoding occupation-number states as bit strings and obtaining fermionic signs from the parity of occupied lower-index modes are standard exact-diagonalization techniques. EDinPy uses these identities in its symbolic operator action and sparse matrix-construction kernels.

6. H. Q. Lin, J. E. Gubernatis, H. Gould, and J. Tobochnik, "Exact Diagonalization Methods for Quantum Systems," *Computers in Physics* **7**, 400-407 (1993), [doi:10.1063/1.4823192](https://doi.org/10.1063/1.4823192).
7. A. W. Sandvik, "Computational Studies of Quantum Spin Systems," *AIP Conference Proceedings* **1297**, 135-338 (2010), [arXiv:1101.3281](https://arxiv.org/abs/1101.3281).

The vectorized parity reduction uses the standard XOR-folding parity operation described in Anderson's public-domain collection cited above.

## Sparse and dense eigensolvers

EDinPy delegates sparse Hermitian eigensolution to `scipy.sparse.linalg.eigsh`, SciPy's ARPACK interface. Dense Hermitian eigensolution uses `scipy.linalg.eigh`, which dispatches to LAPACK routines provided through SciPy.

8. R. B. Lehoucq, D. C. Sorensen, and C. Yang, *ARPACK Users' Guide: Solution of Large-Scale Eigenvalue Problems with Implicitly Restarted Arnoldi Methods*, SIAM (1998), [doi:10.1137/1.9780898719628](https://doi.org/10.1137/1.9780898719628).
9. P. Virtanen *et al.*, "SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python," *Nature Methods* **17**, 261-272 (2020), [doi:10.1038/s41592-019-0686-2](https://doi.org/10.1038/s41592-019-0686-2).
10. C. R. Harris *et al.*, "Array programming with NumPy," *Nature* **585**, 357-362 (2020), [doi:10.1038/s41586-020-2649-2](https://doi.org/10.1038/s41586-020-2649-2).

## Licensing

EDinPy is distributed under the MIT License. New code derived from a published algorithm should cite the original source in the relevant documentation or docstring and should be recorded here when the implementation depends materially on that algorithm.

Third-party source code must not be copied into EDinPy unless its license is compatible with EDinPy's distribution terms and all required attribution is preserved.
