"""Shared Hermitian eigensolver utilities for EDinPy."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from numbers import Integral

import numpy as np
from scipy import linalg
from scipy.sparse.linalg import eigsh


@dataclass(frozen=True, slots=True)
class ArpackMemoryEstimate:
    """Estimated dominant SciPy/ARPACK arrays for one eigensolve.

    The estimate covers the Lanczos/Arnoldi basis, residual and work vectors,
    ARPACK's small ``O(ncv**2)`` work array, extraction workspace, and returned
    eigenvectors. It intentionally excludes the Hamiltonian representation,
    EDinPy basis/execution storage, Python/SciPy allocator overhead, linked
    libraries, and unrelated process memory. It is therefore an ARPACK-array
    estimate, not a total-process RSS prediction.
    """

    dimension: int
    k: int
    ncv: int
    dtype: str
    solver_path: str
    vector_bytes: int
    lanczos_basis_bytes: int
    iteration_workspace_bytes: int
    extraction_workspace_bytes: int
    returned_eigenvectors_bytes: int
    estimated_arpack_bytes: int

    @property
    def estimated_peak_bytes(self):
        """Compatibility alias for :attr:`estimated_arpack_bytes`."""
        return self.estimated_arpack_bytes

    def as_dict(self):
        """Return a JSON-serializable dictionary representation."""
        return asdict(self)


def _solver_dtype(dtype):
    """Return the floating dtype used by SciPy's ARPACK wrapper."""
    dtype = np.dtype(dtype)
    if np.issubdtype(dtype, np.complexfloating):
        return np.dtype(np.complex64 if dtype.itemsize <= 8 else np.complex128)
    if np.issubdtype(dtype, np.floating):
        return np.dtype(np.float32 if dtype.itemsize <= 4 else np.float64)
    return np.dtype(np.float64)


def resolve_arpack_ncv(dimension, k, ncv=None):
    """Resolve and validate ``ncv`` using SciPy/ARPACK's default policy.

    Parameters
    ----------
    dimension : int
        Matrix/operator dimension.
    k : int
        Number of requested eigenpairs.  Must satisfy ``0 < k < dimension``.
    ncv : int or None, optional
        Requested Krylov subspace size.  ``None`` resolves to
        ``min(dimension, max(2*k + 1, 20))``.
    """
    if not isinstance(dimension, Integral):
        raise TypeError("'dimension' must be an integer.")
    dimension = int(dimension)
    if dimension <= 0:
        raise ValueError("'dimension' must be positive.")
    if not isinstance(k, Integral):
        raise TypeError("'k' must be an integer.")
    k = int(k)
    if k <= 0 or k >= dimension:
        raise ValueError("'k' must satisfy 0 < k < dimension.")

    if ncv is None:
        return min(dimension, max(2 * k + 1, 20))
    if not isinstance(ncv, Integral):
        raise TypeError("'ncv' must be an integer or None.")
    ncv = int(ncv)
    if not k < ncv <= dimension:
        raise ValueError("'ncv' must satisfy k < ncv <= dimension.")
    return ncv


def estimate_arpack_memory(dimension, dtype, *, k=2, ncv=None):
    """Estimate dominant SciPy/ARPACK workspace for a partial eigensolve.

    The estimate follows the standard matrix-free ``eigsh`` path used by
    EDinPy.  Real symmetric problems use ARPACK's symmetric workspace model.
    Complex Hermitian inputs are handled by SciPy through the nonsymmetric
    complex ARPACK interface and therefore use a different work-array model.

    This is an implementation-informed estimate rather than a process-memory
    guarantee.  It is intended for comparing ``ncv`` choices before launching
    a large solve.
    """
    ncv = resolve_arpack_ncv(dimension, k, ncv)
    dimension = int(dimension)
    k = int(k)
    dtype = _solver_dtype(dtype)
    itemsize = dtype.itemsize
    vector_bytes = dimension * itemsize
    basis_bytes = ncv * vector_bytes
    residual_and_workd = 4 * vector_bytes  # residual + ARPACK workd[3*n]
    returned_bytes = k * vector_bytes

    if np.issubdtype(dtype, np.complexfloating):
        solver_path = "complex_nonsymmetric_arpack"
        workl_bytes = 3 * ncv * (ncv + 2) * itemsize
        real_itemsize = itemsize // 2
        rwork_bytes = ncv * real_itemsize
        iteration_bytes = (
            basis_bytes + residual_and_workd + workl_bytes + rwork_bytes
        )
        extraction_bytes = (
            returned_bytes
            + 3 * ncv * itemsize  # workev
            + ncv * np.dtype(np.int32).itemsize  # selection mask
            + k * itemsize
        )
    else:
        solver_path = "real_symmetric_arpack"
        workl_bytes = ncv * (ncv + 8) * itemsize
        iteration_bytes = basis_bytes + residual_and_workd + workl_bytes
        # SciPy allocates another n-by-ncv extraction array and copies the
        # converged n-by-k eigenvectors before releasing it.
        extraction_bytes = (
            basis_bytes
            + returned_bytes
            + ncv * np.dtype(np.int32).itemsize
            + k * itemsize
        )

    return ArpackMemoryEstimate(
        dimension=dimension,
        k=k,
        ncv=ncv,
        dtype=dtype.name,
        solver_path=solver_path,
        vector_bytes=vector_bytes,
        lanczos_basis_bytes=basis_bytes,
        iteration_workspace_bytes=iteration_bytes,
        extraction_workspace_bytes=extraction_bytes,
        returned_eigenvectors_bytes=returned_bytes,
        estimated_arpack_bytes=iteration_bytes + extraction_bytes,
    )


def arpack_ncv_for_memory_budget(
    dimension,
    dtype,
    *,
    k=2,
    budget_bytes,
    ncv=None,
):
    """Return the largest requested/default ``ncv`` fitting a workspace budget.

    ``budget_bytes`` applies only to the ARPACK-owned arrays represented by
    :func:`estimate_arpack_memory`; it is not a total-process memory limit.
    When ``ncv`` is ``None``, the search is capped at SciPy's default ``ncv``.
    Supplying ``ncv`` treats it as an explicit upper bound.
    """
    if not isinstance(budget_bytes, Integral):
        raise TypeError("'budget_bytes' must be an integer.")
    budget_bytes = int(budget_bytes)
    if budget_bytes <= 0:
        raise ValueError("'budget_bytes' must be positive.")

    target = resolve_arpack_ncv(dimension, k, ncv)
    minimum = int(k) + 1
    minimum_estimate = estimate_arpack_memory(
        dimension,
        dtype,
        k=k,
        ncv=minimum,
    )
    if minimum_estimate.estimated_arpack_bytes > budget_bytes:
        raise ValueError(
            "The ARPACK workspace budget is too small even for ncv=k+1."
        )

    low, high = minimum, target
    best = minimum
    while low <= high:
        trial = (low + high) // 2
        estimate = estimate_arpack_memory(
            dimension,
            dtype,
            k=k,
            ncv=trial,
        )
        if estimate.estimated_arpack_bytes <= budget_bytes:
            best = trial
            low = trial + 1
        else:
            high = trial - 1
    return best



@dataclass(frozen=True, slots=True)
class SolverMemoryPlan:
    """Conservative pre-flight memory plan for a matrix-free eigensolve.

    ``estimated_arpack_bytes`` comes from :func:`estimate_arpack_memory`.
    ``basis_execution_bytes`` is EDinPy's estimated persistent compact basis
    view required by the statistics-specific execution backend. ``reserve``
    and ``safety_factor`` deliberately leave room for Python/SciPy allocator
    overhead, temporary matrix-free work arrays, linked libraries, and the
    rest of the process.
    """

    dimension: int
    k: int
    ncv: int
    dtype: str
    total_memory_bytes: int
    reserve_bytes: int
    reserve_fraction: float
    usable_memory_bytes: int
    safety_factor: float
    basis_execution_bytes: int
    estimated_arpack_bytes: int
    scaled_arpack_bytes: int
    planned_bytes: int
    headroom_bytes: int
    fits: bool

    def as_dict(self):
        """Return a JSON-serializable dictionary representation."""
        return asdict(self)


def _validate_memory_planning_inputs(
    total_memory_bytes,
    *,
    reserve_bytes=0,
    reserve_fraction=0.25,
    safety_factor=1.25,
    basis_execution_bytes=0,
):
    if not isinstance(total_memory_bytes, Integral):
        raise TypeError("'total_memory_bytes' must be an integer.")
    total_memory_bytes = int(total_memory_bytes)
    if total_memory_bytes <= 0:
        raise ValueError("'total_memory_bytes' must be positive.")

    if not isinstance(reserve_bytes, Integral):
        raise TypeError("'reserve_bytes' must be an integer.")
    reserve_bytes = int(reserve_bytes)
    if reserve_bytes < 0:
        raise ValueError("'reserve_bytes' must be non-negative.")

    reserve_fraction = float(reserve_fraction)
    if not 0.0 <= reserve_fraction < 1.0:
        raise ValueError("'reserve_fraction' must satisfy 0 <= value < 1.")

    safety_factor = float(safety_factor)
    if safety_factor < 1.0:
        raise ValueError("'safety_factor' must be at least 1.")

    if not isinstance(basis_execution_bytes, Integral):
        raise TypeError("'basis_execution_bytes' must be an integer.")
    basis_execution_bytes = int(basis_execution_bytes)
    if basis_execution_bytes < 0:
        raise ValueError("'basis_execution_bytes' must be non-negative.")

    reserve_total = reserve_bytes + int(total_memory_bytes * reserve_fraction)
    usable = max(0, total_memory_bytes - reserve_total)
    return (
        total_memory_bytes,
        reserve_bytes,
        reserve_fraction,
        reserve_total,
        usable,
        safety_factor,
        basis_execution_bytes,
    )


def plan_solver_memory(
    dimension,
    dtype,
    *,
    total_memory_bytes,
    k=2,
    ncv=None,
    reserve_bytes=0,
    reserve_fraction=0.25,
    safety_factor=1.25,
    basis_execution_bytes=0,
):
    """Return a conservative matrix-free eigensolver memory plan.

    The plan is intentionally simple and portable. It does not fit constants
    to one machine's RSS measurements. Instead it reserves an explicit part of
    total memory and applies a safety multiplier to the implementation-informed
    ARPACK estimate before adding EDinPy's known persistent basis storage.
    """
    (
        total_memory_bytes,
        reserve_bytes,
        reserve_fraction,
        reserve_total,
        usable,
        safety_factor,
        basis_execution_bytes,
    ) = _validate_memory_planning_inputs(
        total_memory_bytes,
        reserve_bytes=reserve_bytes,
        reserve_fraction=reserve_fraction,
        safety_factor=safety_factor,
        basis_execution_bytes=basis_execution_bytes,
    )

    estimate = estimate_arpack_memory(dimension, dtype, k=k, ncv=ncv)
    scaled_arpack = int(np.ceil(safety_factor * estimate.estimated_arpack_bytes))
    planned = basis_execution_bytes + scaled_arpack
    headroom = usable - planned

    return SolverMemoryPlan(
        dimension=int(dimension),
        k=int(k),
        ncv=estimate.ncv,
        dtype=estimate.dtype,
        total_memory_bytes=total_memory_bytes,
        reserve_bytes=reserve_total,
        reserve_fraction=reserve_fraction,
        usable_memory_bytes=usable,
        safety_factor=safety_factor,
        basis_execution_bytes=basis_execution_bytes,
        estimated_arpack_bytes=estimate.estimated_arpack_bytes,
        scaled_arpack_bytes=scaled_arpack,
        planned_bytes=planned,
        headroom_bytes=headroom,
        fits=headroom >= 0,
    )


def ncv_for_total_memory(
    dimension,
    dtype,
    *,
    total_memory_bytes,
    k=2,
    ncv=None,
    reserve_bytes=0,
    reserve_fraction=0.25,
    safety_factor=1.25,
    basis_execution_bytes=0,
):
    """Choose the largest ``ncv`` fitting a conservative total-memory plan.

    ``ncv=None`` caps the search at SciPy's normal default. Supplying ``ncv``
    uses it as an explicit upper bound. The minimum admissible subspace is
    ``k+1``; a ``ValueError`` is raised when even that plan does not fit.
    """
    target = resolve_arpack_ncv(dimension, k, ncv)
    minimum = int(k) + 1

    minimum_plan = plan_solver_memory(
        dimension,
        dtype,
        total_memory_bytes=total_memory_bytes,
        k=k,
        ncv=minimum,
        reserve_bytes=reserve_bytes,
        reserve_fraction=reserve_fraction,
        safety_factor=safety_factor,
        basis_execution_bytes=basis_execution_bytes,
    )
    if not minimum_plan.fits:
        raise ValueError(
            "The total-memory plan is too small even for ncv=k+1 after "
            "reserve, safety factor, and basis execution storage."
        )

    low, high = minimum, target
    best = minimum
    while low <= high:
        trial = (low + high) // 2
        plan = plan_solver_memory(
            dimension,
            dtype,
            total_memory_bytes=total_memory_bytes,
            k=k,
            ncv=trial,
            reserve_bytes=reserve_bytes,
            reserve_fraction=reserve_fraction,
            safety_factor=safety_factor,
            basis_execution_bytes=basis_execution_bytes,
        )
        if plan.fits:
            best = trial
            low = trial + 1
        else:
            high = trial - 1
    return best

def spectral_order(eigenvalues, which):
    """Return indices ordered with SciPy/ARPACK ``which`` semantics."""
    values = np.asarray(eigenvalues)
    if which == "SA":
        return np.argsort(values)
    if which == "LA":
        return np.argsort(values)[::-1]
    if which == "SM":
        return np.argsort(np.abs(values))
    return np.argsort(np.abs(values))[::-1]


def is_hermitian_matrix(matrix, *, atol=1e-12):
    """Return whether a sparse matrix is Hermitian within ``atol``."""
    difference = matrix - matrix.getH()
    if difference.nnz == 0:
        return True
    return bool(np.max(np.abs(difference.data)) <= atol)


def solve_hermitian_matrix(
    matrix,
    *,
    sparse=True,
    k=2,
    which="SA",
    tol=1e-10,
    maxiter=None,
    ncv=None,
    v0=None,
    check_hermitian=True,
):
    """Solve a finite Hermitian matrix eigenproblem with SciPy.

    Parameters
    ----------
    matrix : scipy.sparse.spmatrix
        Square Hermitian matrix to diagonalize.
    sparse : bool, optional
        Use ARPACK through :func:`scipy.sparse.linalg.eigsh` when ``True`` and
        fewer than all eigenpairs are requested. Otherwise use the dense
        Hermitian solver :func:`scipy.linalg.eigh`.
    k : int or None, optional
        Number of eigenpairs to return. ``None`` requests the full spectrum.
    which : {"SA", "LA", "SM", "LM"}, optional
        ARPACK selection rule: smallest algebraic, largest algebraic, smallest
        magnitude, or largest magnitude, respectively.
    tol : float, optional
        ARPACK convergence tolerance.
    maxiter : int or None, optional
        Maximum ARPACK iteration count. ``None`` uses SciPy's default.
    ncv : int or None, optional
        Number of Lanczos vectors used by ARPACK. ``None`` lets SciPy choose.
    v0 : array-like or None, optional
        Starting vector supplied to ARPACK. ``None`` lets SciPy initialize it.
    check_hermitian : bool, optional
        Validate Hermiticity before solving when ``True``.

    Returns
    -------
    eigenvalues : numpy.ndarray
        Selected eigenvalues ordered according to ``which``.
    eigenvectors : numpy.ndarray
        Corresponding normalized eigenvectors stored by column.
    """
    dimension = matrix.shape[0]

    if check_hermitian and not is_hermitian_matrix(matrix):
        raise ValueError("Hamiltonian.eigsolve requires a Hermitian matrix.")
    if which not in {"SA", "LA", "SM", "LM"}:
        raise ValueError("'which' must be one of 'SA', 'LA', 'SM', or 'LM'.")
    if k is not None:
        if not isinstance(k, Integral):
            raise TypeError("'k' must be an integer or None.")
        k = int(k)
        if k <= 0:
            raise ValueError("'k' must be positive or None.")
        k = min(k, dimension)

    if dimension == 0:
        eigenvalues = np.empty(0, dtype=np.float64)
        eigenvectors = np.empty((0, 0), dtype=matrix.dtype)
    elif sparse and k is not None and k < dimension:
        eigenvalues, eigenvectors = eigsh(
            matrix,
            k=k,
            which=which,
            tol=tol,
            maxiter=maxiter,
            ncv=ncv,
            v0=v0,
        )
        order = spectral_order(eigenvalues, which)
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
    else:
        dense = matrix.toarray()
        if k is not None and which in {"SA", "LA"} and k < dimension:
            subset = (
                (0, k - 1)
                if which == "SA"
                else (dimension - k, dimension - 1)
            )
            eigenvalues, eigenvectors = linalg.eigh(
                dense,
                subset_by_index=subset,
                check_finite=False,
            )
            order = spectral_order(eigenvalues, which)
            eigenvalues = eigenvalues[order]
            eigenvectors = eigenvectors[:, order]
        else:
            eigenvalues, eigenvectors = linalg.eigh(
                dense,
                check_finite=False,
            )
            order = spectral_order(eigenvalues, which)
            if k is not None:
                order = order[:k]
            eigenvalues = eigenvalues[order]
            eigenvectors = eigenvectors[:, order]

    return eigenvalues, eigenvectors


def solve_hermitian_operator(
    operator,
    *,
    sparse=True,
    k=2,
    which="SA",
    tol=1e-10,
    maxiter=None,
    ncv=None,
    v0=None,
):
    """Solve a Hermitian partial eigenproblem from a matrix-free operator.

    Unlike :func:`solve_hermitian_matrix`, this function never materializes a
    dense or sparse matrix.  Consequently only the partial ARPACK path is
    supported; complete spectra and dense solves require an explicit matrix.
    """
    if not sparse:
        raise ValueError("Matrix-free eigensolve requires sparse=True.")

    dimension = operator.shape[0]
    if operator.shape != (dimension, dimension):
        raise ValueError("Matrix-free eigensolve requires a square operator.")
    if which not in {"SA", "LA", "SM", "LM"}:
        raise ValueError("'which' must be one of 'SA', 'LA', 'SM', or 'LM'.")
    if k is None:
        raise ValueError("Matrix-free eigensolve requires a finite 'k'.")
    if not isinstance(k, Integral):
        raise TypeError("'k' must be an integer.")
    k = int(k)
    if k <= 0:
        raise ValueError("'k' must be positive.")
    if dimension == 0:
        return (
            np.empty(0, dtype=np.float64),
            np.empty((0, 0), dtype=operator.dtype),
        )
    if k >= dimension:
        raise ValueError(
            "Matrix-free eigensolve requires k < Hamiltonian dimension."
        )

    eigenvalues, eigenvectors = eigsh(
        operator,
        k=k,
        which=which,
        tol=tol,
        maxiter=maxiter,
        ncv=ncv,
        v0=v0,
    )
    order = spectral_order(eigenvalues, which)
    return eigenvalues[order], eigenvectors[:, order]
