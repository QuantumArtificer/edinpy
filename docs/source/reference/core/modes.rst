Degrees of freedom and mode indexing
====================================

DoF
---

Public names: ``edinpy.fermion.DoF`` and ``edinpy.boson.DoF``.

.. autoclass:: edinpy.fermion.DoF
   :members:

Shared mode-indexing interface
------------------------------

``FermionModes`` and ``BosonModes`` use the same ordered Cartesian-product
indexing of their :class:`DoF` objects.

.. autoattribute:: edinpy.fermion.FermionModes.dofs

.. autoattribute:: edinpy.fermion.FermionModes.strides

.. autoattribute:: edinpy.fermion.FermionModes.n_modes

.. automethod:: edinpy.fermion.FermionModes.resolve

.. automethod:: edinpy.fermion.FermionModes.unravel
