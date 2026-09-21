# EDinPy provenance and algorithmic references

This document records external literature, algorithms, software, and source-code
collections reviewed during the development and provenance audit of EDinPy.
Its purpose is to distinguish **algorithmic provenance** from **software reviewed
for comparison**.

Unless explicitly stated below, EDinPy does not incorporate source code from the
listed third-party repositories. Similarities in standard mathematical objects
(e.g. occupation-number bit strings, fermionic parity signs, sparse matrices, or
Lanczos/Arnoldi eigensolvers) reflect established methods rather than copied
implementations.

## Direct algorithmic sources

### Fixed-particle-number fermionic basis enumeration

EDinPy enumerates fixed-population fermionic occupation bit strings using the
classic "next higher integer with the same number of one bits" construction,
usually called **Gosper's hack**. The construction originates in HAKMEM,
Item 175, and is also discussed by Knuth and Warren.

The historical `fermionspace.nextvector` implementation was written with Sean
E. Anderson's *Bit Twiddling Hacks* page as an implementation reference. Anderson
states that the individual code snippets on that page are public domain. EDinPy's current fixed-N basis builder uses a Python implementation of the
same standard Gosper construction.

References:

1. M. Beeler, R. W. Gosper, and R. Schroeppel, *HAKMEM*, MIT Artificial
   Intelligence Laboratory Memo AIM-239, Item 175 (1972).
2. D. E. Knuth, *The Art of Computer Programming*, Vol. 4A, Sec. 7.1.3,
   Addison-Wesley (2011).
3. H. S. Warren, Jr., *Hacker's Delight*, 2nd ed., Addison-Wesley (2013),
   Sec. 2-1.
4. S. E. Anderson, *Bit Twiddling Hacks*,
   https://graphics.stanford.edu/~seander/bithacks.html

### Bosonic occupation-basis enumeration

The lexicographic enumeration used by the bosonic basis routines `findk` and
`nextvector` follows the algorithm described by Zhang and Dong for exact
diagonalization of the Bose-Hubbard model. EDinPy contains its own Python
implementation and has cited this source since the first public implementation
of the bosonic module.

Reference:

5. J. M. Zhang and R. X. Dong, "Exact diagonalization: the Bose-Hubbard model
   as an example," *European Journal of Physics* **31**, 591-602 (2010),
   doi:10.1088/0143-0807/31/3/016.

### Bit-string exact diagonalization and fermionic signs

Encoding occupation-number configurations as integers and obtaining the
fermionic sign from the parity of occupied lower-index modes are standard exact-
diagonalization techniques. EDinPy's compiler, lowered intermediate
representation, and execution kernels are EDinPy-specific implementations built
on these standard identities. The vectorized parity reduction uses the standard
XOR-folding parity technique; Anderson's public-domain collection was reviewed
as a reference for this bit operation.

Background references:

6. H. Q. Lin, J. E. Gubernatis, H. Gould, and J. Tobochnik, "Exact
   Diagonalization Methods for Quantum Systems," *Computers in Physics* **7**,
   400-407 (1993), doi:10.1063/1.4823192.
7. A. W. Sandvik, "Computational Studies of Quantum Spin Systems," in
   *AIP Conference Proceedings* **1297**, 135-338 (2010), arXiv:1101.3281.

### Sparse eigensolution

EDinPy delegates sparse Hermitian eigensolution to
`scipy.sparse.linalg.eigsh`, which is SciPy's ARPACK interface. ARPACK is an
external numerical library; EDinPy does not implement or vendor ARPACK.

References:

8. R. B. Lehoucq, D. C. Sorensen, and C. Yang, *ARPACK Users' Guide:
   Solution of Large-Scale Eigenvalue Problems with Implicitly Restarted
   Arnoldi Methods*, SIAM (1998), doi:10.1137/1.9780898719628.
9. P. Virtanen et al., "SciPy 1.0: fundamental algorithms for scientific
   computing in Python," *Nature Methods* **17**, 261-272 (2020),
   doi:10.1038/s41592-019-0686-2.
10. C. R. Harris et al., "Array programming with NumPy," *Nature* **585**,
    357-362 (2020), doi:10.1038/s41586-020-2649-2.

## Software repositories reviewed for independent comparison

The following projects were inspected during the 2026 provenance audit to check
for implementation-level overlap and to understand established exact-
diagonalization software designs. **No source code from these projects was copied
into EDinPy during this audit.** They are not implementation dependencies unless
listed separately in EDinPy's package metadata.

| Project | Relevant scope | License at audit | Relationship to EDinPy |
| --- | --- | --- | --- |
| QuSpin | Exact diagonalization; boson, fermion, spin bases and symmetries | BSD-3-Clause | Reviewed for basis/operator architecture and ED capabilities; no code copied. |
| OpenFermion | Symbolic fermionic operator algebra | Apache-2.0 | Reviewed for symbolic-operator representation; implementation differs from EDinPy's expression-tree/compiler design. |
| OpenFermion-FQE | Fermionic bit strings, number/spin sectors | Apache-2.0 | Reviewed for bit-string and sector representation; no code copied. |
| PySCF | Determinant strings and FCI infrastructure | Apache-2.0 | Reviewed `pyscf.fci.cistring`; its determinant enumeration differs from EDinPy's Gosper implementation. |
| NetKet | Fermionic Hilbert spaces and second-quantized operators | Apache-2.0 | Reviewed fermionic operator interface and Hilbert-space ownership; no code copied. |
| HΦ | Quantum-lattice exact diagonalization and bit operations | GPL-3.0 | **Comparison only. GPL source must not be incorporated into MIT-licensed EDinPy without an explicit licensing decision.** |
| EDLib | Exact diagonalization for quantum electron models | MIT | Reviewed solver architecture and ARPACK usage; no code copied. |
| ALPS / ALPSCore | Many-body simulation and ED infrastructure | MIT for ALPSCore | Reviewed as established many-body software; no code copied. |
| QuTiP | General quantum-system operators and solvers | BSD-3-Clause | Reviewed for scientific-Python API conventions; no code copied. |

Representative software references:

11. P. Weinberg and M. Bukov, "QuSpin: a Python package for dynamics and exact
    diagonalisation of quantum many body systems. Part I: spin chains,"
    *SciPost Physics* **2**, 003 (2017), doi:10.21468/SciPostPhys.2.1.003.
12. P. Weinberg and M. Bukov, "QuSpin: a Python package for dynamics and exact
    diagonalisation of quantum many body systems. Part II: bosons, fermions
    and higher spins," *SciPost Physics* **7**, 020 (2019),
    doi:10.21468/SciPostPhys.7.2.020.
13. J. R. McClean et al., "OpenFermion: The Electronic Structure Package for
    Quantum Computers," *Quantum Science and Technology* **5**, 034014 (2020),
    arXiv:1710.07629.
14. N. C. Rubin et al., "The Fermionic Quantum Emulator," arXiv:2104.13944
    (2021).
15. Q. Sun et al., "Recent developments in the PySCF program package,"
    *Journal of Chemical Physics* **153**, 024109 (2020),
    doi:10.1063/5.0006074.
16. F. Vicentini et al., "NetKet 3: Machine Learning Toolbox for Many-Body
    Quantum Systems," *SciPost Physics Codebases* **7** (2022),
    doi:10.21468/SciPostPhysCodeb.7.
17. M. Kawamura et al., "Quantum lattice model solver HΦ," *Computer Physics
    Communications* **217**, 180-192 (2017),
    doi:10.1016/j.cpc.2017.04.006.
18. B. Bauer et al., "The ALPS project release 2.0: Open source software for
    strongly correlated systems," *Journal of Statistical Mechanics* P05001
    (2011).
19. J. R. Johansson, P. D. Nation, and F. Nori, "QuTiP 2: A Python framework
    for the dynamics of open quantum systems," *Computer Physics
    Communications* **184**, 1234-1240 (2013),
    doi:10.1016/j.cpc.2012.11.019.

## Public-code similarity checks

Targeted GitHub code searches were performed for distinctive EDinPy compiler and
basis-enumeration identifiers and phrases. As of 2026-09-20:

- `_HoppingPairKernel` returned no public-code matches.
- `vectorized_number_conserving_monomials` returned no public-code matches.
- `required_empty_mask` together with `transition_mask` returned no public-code
  matches.
- `_calc_effsite` together with fermionic context returned only EDinPy source,
  build, and generated-documentation copies among the relevant results.
- the combined `fermionspace`/`firstvector`/`nextvector` query returned only
  EDinPy source/build copies.
- the distinctive bosonic `calc_tag` expression and lexicographic-enumeration
  comments returned only EDinPy source/build/generated-documentation copies.

These searches provide evidence against direct public-source copying, but they
cannot prove the universal negative that no unpublished or unindexed code is
similar.

## Contribution and licensing policy

EDinPy is distributed under the MIT License. To keep provenance unambiguous:

1. New implementations should be written independently from mathematical or
   algorithmic descriptions whenever practical.
2. When an implementation follows a published algorithm or a recognizable
   external implementation, cite it in the relevant NumPy-style docstring and
   record it here.
3. Do not copy source from GPL or otherwise license-incompatible projects into
   EDinPy without an explicit relicensing decision.
4. If third-party code is ever incorporated, preserve all legally required
   copyright, license, and NOTICE material and record the exact file/version in
   this document.
5. Repositories inspected only for comparison should not be described as code
   sources or dependencies.
