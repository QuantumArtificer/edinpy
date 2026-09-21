# Fermionic Fock space and fixed-$N$ sectors

For $M$ fermionic modes, the occupation-number basis is

$$
|n_0,n_1,\ldots,n_{M-1}\rangle,
\qquad n_p\in\{0,1\}.
$$

The full fermionic Fock space is the direct sum

$$
\mathcal F=\bigoplus_{N=0}^{M}\mathcal H_N,
$$

where $\mathcal H_N$ contains states with exactly $N$ occupied modes. EDinPy 0.2.0 constructs `NParticleSector`, i.e. one $\mathcal H_N$ at a time.

The dimension of this sector is

$$
\dim \mathcal H_N=\binom{M}{N}.
$$

EDinPy enumerates fixed-population occupation bit strings with the standard next-combination construction commonly associated with Gosper.[^hakmem] The implementation reference and provenance notes are also recorded in `PROVENANCE.md` and the `FockBasis` docstring.

[^hakmem]: R. W. Gosper, in *HAKMEM*, MIT AI Memo 239, Item 175 (1972).
