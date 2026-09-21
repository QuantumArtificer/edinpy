Current limitations
===================

Fermionic sectors
-----------------

Version 0.1.0 constructs Hamiltonians in fixed-particle-number sectors only.
Full Fock-space, fermion-parity, translation, and momentum sectors are not
implemented.

Number-changing Hamiltonian terms
---------------------------------

Creation or annihilation products that change total particle number can act on
individual ``FockState`` objects. Their matrix elements between different
particle-number sectors are not represented by ``Hamiltonian`` in an
``NParticleSector`` and are projected out.

Degenerate eigenspaces
----------------------

ARPACK uses a single-vector Krylov iteration. A converged residual does not by
itself guarantee recovery of every vector in an exactly degenerate eigenspace.
Dense diagonalization can be used when complete degenerate subspaces are
required and the Hilbert-space dimension permits it.

Bosonic module
--------------

The bosonic implementation is not part of the fermionic architecture refactor
in version 0.1.0.
