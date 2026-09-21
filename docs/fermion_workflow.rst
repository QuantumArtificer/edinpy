Fermionic workflow
==================

Mode labels
-----------

``DoF`` objects specify discrete labels. ``FermionModes`` forms their ordered
Cartesian product and maps tuples of labels to fermionic-mode indices. The first
``DoF`` is the fastest-varying index.

.. code-block:: python

   from edinpy import fermion as edf

   site = edf.DoF(8, name="site")
   spin = edf.DoF(2, name="spin")
   modes = edf.FermionModes(site, spin)

Fixed-particle-number sector
----------------------------

``NParticleSector`` constructs the occupation-number basis for fixed particle
number ``N``.

.. code-block:: python

   sector = edf.NParticleSector(modes, N=8)

The basis dimension is ``sector.dimension``. Basis states are integer bit strings
with exactly ``N`` occupied modes.

Operator notation
-----------------

Primitive operator notation is selected by the user.

.. code-block:: python

   c = edf.set_notation(edf.Annihilation, modes)
   cd = edf.set_notation(edf.Creation, modes)
   n = edf.set_notation(edf.Number, modes)

The assigned Python variable names have no semantic role in EDinPy. Alternative
conventions such as ``d``, ``f``, or ``psi`` are equivalent.

Literal algebra and compilation
-------------------------------

Operator sums and products are literal symbolic Fock algebra. Common structures
are recognized by the compiler and lowered to specialized sparse execution
kernels. The optimized path therefore does not require use of common-operator
helpers.

.. code-block:: python

   H = -1.0 * (cd(0, 0) * c(1, 0) + cd(1, 0) * c(0, 0))
   H += 4.0 * n(0, 0) * n(0, 1)

Hamiltonian and eigensolution
-----------------------------

.. code-block:: python

   hamiltonian = edf.Hamiltonian(H, sector)
   eigenvalues, eigenvectors = hamiltonian.eigsolve(
       k=4,
       which="SA",
       tol=1e-10,
   )
