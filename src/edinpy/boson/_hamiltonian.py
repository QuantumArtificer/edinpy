"""Register bosonic compilation and execution with the shared Hamiltonian shell."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from edinpy._core._hamiltonian import (
    Hamiltonian,
    HamiltonianBackend,
    register_hamiltonian_backend,
)

from ._basis import FockBasis
from ._execution import CompiledOperator, compile_operator
from ._modes import BosonModes
from ._numba_execution import (
    matvec_numba_parallel,
    matvec_numba_serial,
    require_numba_supported,
)

_NumPyVector = NDArray[np.generic]
_NumPyExecutor = Callable[[CompiledOperator, FockBasis, _NumPyVector], _NumPyVector]
_NUMBA_EXECUTORS: dict[str, _NumPyExecutor] = {
    "numba-serial": matvec_numba_serial,
    "numba-parallel": matvec_numba_parallel,
}


def _emit_boson_csc(
    compiled: CompiledOperator,
    basis: FockBasis,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Emit the compiled bosonic operator in CSC storage arrays.

    Parameters
    ----------
    compiled : CompiledOperator
        Bosonic operator produced by :func:`compile_operator`.
    basis : FockBasis
        Built fixed-particle basis defining sparse row and column ordering.

    Returns
    -------
    indices, indptr, data : tuple of numpy.ndarray
        Row indices, column pointers, and numerical matrix elements accepted by
        :class:`scipy.sparse.csc_matrix`.
    """
    return compiled.emit_csc(basis)


def _boson_matvec(
    compiled: CompiledOperator,
    basis: FockBasis,
    vector: _NumPyVector,
) -> _NumPyVector:
    """Apply the vectorized NumPy boson executor.

    Parameters
    ----------
    compiled : CompiledOperator
        Lowered bosonic operator to apply.
    basis : FockBasis
        Basis defining input and output vector ordering.
    vector : numpy.ndarray
        One-dimensional numerical vector with length ``basis.dimension``.

    Returns
    -------
    numpy.ndarray
        Matrix-free Hamiltonian action in the same basis ordering.
    """
    return compiled.matvec(basis, vector)


def _boson_matvec_execution(
    compiled: CompiledOperator,
    basis: FockBasis,
    vector: _NumPyVector,
    execution: str,
) -> _NumPyVector:
    """Apply one normalized public bosonic matrix-free execution mode.

    Parameters
    ----------
    compiled : CompiledOperator
        Bosonic operator prepared by the symbolic compiler.
    basis : FockBasis
        Built fixed-particle basis associated with the Hamiltonian sector.
    vector : numpy.ndarray
        Numerical state vector with length ``basis.dimension``.
    execution : {"numpy", "numba-serial", "numba-parallel", "mixed"}
        Concrete execution request. ``"numpy"`` uses the vectorized NumPy
        backend. The two Numba values are strict requests. ``"mixed"`` is
        conservative for bosons and currently resolves to NumPy.

    Returns
    -------
    numpy.ndarray
        Hamiltonian action in the same basis ordering as ``vector``.
    """
    if execution in {"numpy", "mixed"}:
        return _boson_matvec(compiled, basis, vector)
    try:
        executor = _NUMBA_EXECUTORS[execution]
    except KeyError as exc:
        raise ValueError(f"Unknown boson execution mode {execution!r}.") from exc
    return executor(compiled, basis, vector)


def _boson_resolve_execution(
    compiled: CompiledOperator,
    basis: FockBasis,
    execution: str,
) -> str:
    """Resolve a public selector to the concrete bosonic execution backend.

    Parameters
    ----------
    compiled : CompiledOperator
        Compiled bosonic operator whose lowering status determines Numba
        eligibility.
    basis : FockBasis
        Fixed-particle basis whose rank representation determines Numba
        eligibility.
    execution : {"numpy", "numba-serial", "numba-parallel", "mixed"}
        Requested public execution selector. Explicit Numba requests are
        validated strictly; ``"mixed"`` resolves to ``"numpy"``.

    Returns
    -------
    str
        Concrete backend name used by the shared Hamiltonian shell.

    Raises
    ------
    NotImplementedError
        If an explicit Numba mode is requested for an unsupported operator,
        basis, or installation.
    """
    if execution in {"numpy", "mixed"}:
        return "numpy"
    if execution in _NUMBA_EXECUTORS:
        require_numba_supported(compiled, basis, execution)
        return execution
    raise ValueError(f"Unknown boson execution mode {execution!r}.")


def _boson_dtype(compiled: CompiledOperator) -> np.dtype:
    """Return the double-precision scalar dtype required by ``compiled``."""
    return np.dtype(compiled.data_dtype)


register_hamiltonian_backend(
    BosonModes,
    HamiltonianBackend(
        compile_operator=compile_operator,
        emit_csc=_emit_boson_csc,
        matvec=_boson_matvec,
        dtype=_boson_dtype,
        matvec_execution=_boson_matvec_execution,
        resolve_execution=_boson_resolve_execution,
    ),
)
