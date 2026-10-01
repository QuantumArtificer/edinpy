Bosonic modes, sectors, and basis states
========================================

BosonModes
----------

.. autoclass:: edinpy.boson.BosonModes

The shared indexing properties and methods are listed under
:doc:`../core/modes`.

NParticleSector
---------------

.. autoclass:: edinpy.boson.NParticleSector
   :members: is_built, project_particles, build, basis, dimension, from_vector

FockBasis
---------

.. autoclass:: edinpy.boson.FockBasis
   :members: dimension, is_complete, is_materialized, states, packed_states, state, state_at, packed_at, index, pack, unpack, occupation, expected_dimension, bits_per_mode, packed_width, storage_bytes, estimated_execution_storage_bytes

FockState
---------

.. autoclass:: edinpy.boson.FockState
   :members: state, N, n_modes, occupation, is_occupied, dag, inner, norm, normalized
