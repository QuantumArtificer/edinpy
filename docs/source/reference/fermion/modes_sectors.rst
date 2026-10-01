Fermionic modes, sectors, and basis states
==========================================

FermionModes
------------

.. autoclass:: edinpy.fermion.FermionModes

The shared indexing properties and methods are listed under
:doc:`../core/modes`.

NParticleSector
---------------

.. autoclass:: edinpy.fermion.NParticleSector
   :members: is_built, project_particles, build, basis, dimension, from_vector

FockBasis
---------

.. autoclass:: edinpy.fermion.FockBasis
   :members: dimension, is_complete, is_materialized, states, state, state_at, index, expected_dimension, storage_bytes, estimated_execution_storage_bytes

FockState
---------

.. autoclass:: edinpy.fermion.FockState
   :members: dag, inner, norm, normalized, is_occupied
