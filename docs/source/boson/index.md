# Bosonic exact diagonalization

EDinPy includes a bosonic exact-diagonalization module under `edinpy.boson`. It uses a separate public API from `edinpy.fermion` and currently has less validation, documentation, and performance coverage.

## Current status

The bosonic code can construct finite occupation bases and bosonic Hamiltonians, but it still uses global model state and older naming conventions. It has not yet received the same API review, validation suite, examples, or performance work as `edinpy.fermion`.

For this reason, the bosonic API should be treated as provisional in new code. Existing scripts can continue to use it, but future releases may change its public interface.

## Why the bosonic basis needs different choices

A fermionic mode has occupation $0$ or $1$. A bosonic mode can have arbitrarily many particles, so a numerical calculation must still define a finite Hilbert space. Common choices include a fixed total boson number, a local occupation cutoff, or both.

Those restrictions are part of the physical and numerical model. A future bosonic API will make them explicit in the same way that `NParticleSector` makes the fermionic particle-number restriction explicit.

## Current recommendation

Use {doc}`../fermion/index` for the fully documented EDinPy workflow. If you need the current bosonic module, inspect its API and validate the basis and Hamiltonian for the model being studied.

The bosonic basis-enumeration method is cited in {doc}`../references` and recorded in the repository's [`PROVENANCE.md`](https://github.com/QuantumArtificer/edinpy/blob/main/PROVENANCE.md).
