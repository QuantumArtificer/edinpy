"""Fermion-specific backend registration for the shared Hamiltonian shell."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from edinpy._core._hamiltonian import (
    Hamiltonian,
    HamiltonianBackend,
    register_hamiltonian_backend,
)

from ._basis import FockBasis, FockState
from ._execution import CompiledOperator, compile_operator
from ._modes import FermionModes


def _emit_fermion_csc(
    compiled: CompiledOperator,
    basis: FockBasis,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Emit CSC index arrays for a compiled fermion operator.

    Parameters
    ----------
    compiled : CompiledOperator
        Fermion operator produced by :func:`compile_operator`. Fully lowered
        operators use the optimized sparse emitter; operators retaining generic
        symbolic kernels use the literal basis-state fallback below.
    basis : FockBasis
        Built fermion basis defining the sparse matrix row and column ordering.

    Returns
    -------
    indices : numpy.ndarray
        CSC row indices, using 32-bit integers when the matrix dimensions allow
        them and 64-bit integers otherwise.
    indptr : numpy.ndarray
        CSC column pointer array with length ``basis.dimension + 1``.
    data : numpy.ndarray
        Nonzero matrix elements in column-major CSC order.
    """
    if compiled.fully_lowered:
        return compiled.emit_csc(
            basis.states,
            n_modes=basis.n_modes,
        )

    lookup = {state: index for index, state in enumerate(basis.states)}
    indices: list[int] = []
    data: list[complex] = []
    indptr = [0]

    for column, state_int in enumerate(basis.states):
        ket = FockState(
            state_int,
            n_modes=basis.n_modes,
            index=column,
        )
        output = compiled.apply(ket)
        entries = []
        for final_state, amplitude in output.items():
            row = lookup.get(final_state)
            if row is not None and amplitude != 0:
                entries.append((row, amplitude))
        entries.sort(key=lambda item: item[0])
        for row, amplitude in entries:
            indices.append(row)
            data.append(amplitude)
        indptr.append(len(indices))

    index_dtype = (
        np.int32
        if max(len(basis), len(indices), 1) <= np.iinfo(np.int32).max
        else np.int64
    )
    return (
        np.asarray(indices, dtype=index_dtype),
        np.asarray(indptr, dtype=index_dtype),
        np.asarray(data, dtype=np.complex128),
    )


def _fermion_matvec(
    compiled: CompiledOperator,
    basis: FockBasis,
    vector: NDArray[np.generic],
) -> NDArray[np.generic]:
    """Apply a compiled fermion operator with the NumPy executor.

    Parameters
    ----------
    compiled : CompiledOperator
        Lowered fermion operator to apply.
    basis : FockBasis
        Basis defining the input and output vector ordering.
    vector : numpy.ndarray
        One-dimensional numerical state vector with length ``basis.dimension``.

    Returns
    -------
    numpy.ndarray
        Matrix-free Hamiltonian action in the same basis ordering.
    """
    return compiled.matvec(
        basis,
        n_modes=basis.n_modes,
        vector=vector,
    )


def _fermion_matvec_execution(
    compiled: CompiledOperator,
    basis: FockBasis,
    vector: NDArray[np.generic],
    execution: str,
) -> NDArray[np.generic]:
    """Apply one public fermion matrix-free execution mode.

    Parameters
    ----------
    compiled : CompiledOperator
        Fermion operator prepared by the symbolic compiler.
    basis : FockBasis
        Built basis that defines the matrix-free state ordering.
    vector : numpy.ndarray
        Numerical state vector with length ``basis.dimension``.
    execution : {"numpy", "numba-serial", "numba-parallel", "mixed"}
        Matrix-free backend. ``"mixed"`` uses serial Numba when the operator
        and basis satisfy Numba's requirements and otherwise falls back to the
        NumPy executor. It never enables parallel execution automatically.

    Returns
    -------
    numpy.ndarray
        Hamiltonian action in the same basis ordering as ``vector``.
    """
    if execution == "numpy":
        return _fermion_matvec(compiled, basis, vector)

    from ._numba_execution import (
        matvec_numba_parallel,
        matvec_numba_serial,
        numba_supported,
        require_numba_supported,
    )

    if execution == "numba-serial":
        require_numba_supported(compiled, basis, execution)
        return matvec_numba_serial(compiled, basis, vector)
    if execution == "numba-parallel":
        require_numba_supported(compiled, basis, execution)
        return matvec_numba_parallel(compiled, basis, vector)
    if execution == "mixed":
        if numba_supported(compiled, basis):
            return matvec_numba_serial(compiled, basis, vector)
        return _fermion_matvec(compiled, basis, vector)
    raise ValueError(f"Unknown fermion execution mode {execution!r}.")


def _fermion_resolve_execution(
    compiled: CompiledOperator,
    basis: FockBasis,
    execution: str,
) -> str:
    """Resolve a requested selector to the concrete fermion matrix-free backend.

    Parameters
    ----------
    compiled : CompiledOperator
        Compiled fermion operator whose lowering status determines Numba
        eligibility.
    basis : FockBasis
        Basis whose width and projection structure determine Numba eligibility.
    execution : {"numpy", "numba-serial", "numba-parallel", "mixed"}
        Requested public execution selector.

    Returns
    -------
    str
        Concrete backend name. ``"mixed"`` resolves to ``"numba-serial"`` for
        supported complete native-word bases and to ``"numpy"`` otherwise.

    Raises
    ------
    NotImplementedError
        If an explicit Numba mode is requested for an unsupported operator or
        basis.
    """
    if execution == "numpy":
        return "numpy"

    from ._numba_execution import numba_supported, require_numba_supported

    if execution in {"numba-serial", "numba-parallel"}:
        require_numba_supported(compiled, basis, execution)
        return execution
    if execution == "mixed":
        return "numba-serial" if numba_supported(compiled, basis) else "numpy"
    raise ValueError(f"Unknown fermion execution mode {execution!r}.")


def _fermion_dtype(compiled: CompiledOperator) -> np.dtype:
    """Return the numerical dtype required by a compiled fermion operator."""
    return compiled.data_dtype


def _fermion_matvec_metadata(
    compiled: CompiledOperator,
    basis: FockBasis,
    execution: str = "numpy",
) -> dict[str, int | str]:
    """Describe the concrete executor and NumPy block geometry.

    Parameters
    ----------
    compiled : CompiledOperator
        Compiled fermion operator associated with the Hamiltonian.
    basis : FockBasis
        Basis whose dimension determines NumPy block sizing.
    execution : {"numpy", "numba-serial", "numba-parallel", "mixed"}, optional
        Requested matrix-free execution selector.

    Returns
    -------
    dict
        Always includes ``execution_resolved``. NumPy native-word execution
        additionally reports ``block_size`` and ``chunks_per_matvec``; Numba
        executors do not use the NumPy source-state blocking scheme.
    """
    resolved = _fermion_resolve_execution(compiled, basis, execution)
    if resolved != "numpy":
        return {"execution_resolved": resolved}
    if not compiled.fully_lowered or int(basis.n_modes) > 64:
        return {"execution_resolved": resolved}

    dimension = int(basis.dimension)
    if dimension == 0:
        return {
            "execution_resolved": resolved,
            "block_size": 0,
            "chunks_per_matvec": 0,
        }

    block_size = compiled._matvec_block_size(None, dimension)
    chunks = (dimension + block_size - 1) // block_size
    return {
        "execution_resolved": resolved,
        "block_size": int(block_size),
        "chunks_per_matvec": int(chunks),
    }


register_hamiltonian_backend(
    FermionModes,
    HamiltonianBackend(
        compile_operator=compile_operator,
        emit_csc=_emit_fermion_csc,
        matvec=_fermion_matvec,
        dtype=_fermion_dtype,
        matvec_metadata=_fermion_matvec_metadata,
        matvec_execution=_fermion_matvec_execution,
        resolve_execution=_fermion_resolve_execution,
    ),
)
