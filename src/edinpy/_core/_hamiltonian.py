"""Statistics-independent Hamiltonian orchestration for EDinPy."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from numbers import Integral
from time import perf_counter

import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import LinearOperator

from ._algebra import Operator, operator_domains
from ._solvers import (
    arpack_ncv_for_memory_budget,
    estimate_arpack_memory,
    is_hermitian_matrix,
    ncv_for_total_memory,
    plan_solver_memory,
    solve_hermitian_matrix,
    solve_hermitian_operator,
)


@dataclass(frozen=True, slots=True)
class HamiltonianBackend:
    """Statistics-specific compilation and sparse-emission callbacks."""

    compile_operator: object
    emit_csc: object
    matvec: object
    dtype: object
    matvec_metadata: object = None
    matvec_execution: object = None
    resolve_execution: object = None


@dataclass(frozen=True, slots=True)
class MatrixFreeSolveDiagnostics:
    """Timing and call-count diagnostics for one matrix-free eigensolve.

    Times are measured at the SciPy ``LinearOperator`` callback boundary.
    ``solver_overhead_seconds`` is the remaining eigensolver wall time after
    subtracting timed Hamiltonian applications; it therefore includes ARPACK
    bookkeeping, orthogonalization, Ritz extraction, and Python/SciPy overhead
    outside the registered backend ``matvec`` callbacks.
    """

    dimension: int
    execution: str
    execution_resolved: str
    solver_wall_seconds: float
    matvec_calls: int
    matvec_total_seconds: float
    matvec_mean_seconds: float
    matvec_min_seconds: float
    matvec_max_seconds: float
    rmatvec_calls: int
    rmatvec_total_seconds: float
    rmatvec_mean_seconds: float
    rmatvec_min_seconds: float
    rmatvec_max_seconds: float
    hamiltonian_apply_seconds: float
    solver_overhead_seconds: float
    hamiltonian_apply_fraction: float
    basis_states_processed: int
    basis_states_per_second: float
    matvec_block_size: int | None
    chunks_per_matvec: int | None
    estimated_total_chunks: int | None

    def as_dict(self):
        """Return a JSON-serializable dictionary representation."""
        return asdict(self)


class _CallTimer:
    """Low-overhead mutable timing accumulator for operator callbacks."""

    __slots__ = ("calls", "total_seconds", "min_seconds", "max_seconds")

    def __init__(self):
        self.reset()

    def reset(self):
        self.calls = 0
        self.total_seconds = 0.0
        self.min_seconds = float("inf")
        self.max_seconds = 0.0

    def record(self, seconds):
        seconds = float(seconds)
        self.calls += 1
        self.total_seconds += seconds
        self.min_seconds = min(self.min_seconds, seconds)
        self.max_seconds = max(self.max_seconds, seconds)

    @property
    def mean_seconds(self):
        return self.total_seconds / self.calls if self.calls else 0.0

    @property
    def bounded_min_seconds(self):
        return self.min_seconds if self.calls else 0.0


_BACKENDS = {}


def register_hamiltonian_backend(modes_type, backend):
    """Register the Hamiltonian backend associated with one mode-space type."""
    if not isinstance(backend, HamiltonianBackend):
        raise TypeError("'backend' must be a HamiltonianBackend instance.")
    _BACKENDS[modes_type] = backend
    return backend


def _backend_for_modes(modes):
    try:
        return _BACKENDS[type(modes)]
    except KeyError as exc:
        raise TypeError(
            "No Hamiltonian backend is registered for mode-space type "
            f"{type(modes).__name__}."
        ) from exc


class Hamiltonian:
    """Finite-sector Hamiltonian built from a symbolic operator expression.

    Parameters
    ----------
    operator : Operator
        Symbolic second-quantized operator expression. All primitive operators
        in the expression must use the same mode specification as ``sector``.
    sector : NParticleSector
        Built fermionic or bosonic sector that defines the ordered many-body
        basis.

    Notes
    -----
    The sparse CSC matrix is constructed lazily when :attr:`matrix`,
    :meth:`calc_matrix`, or an explicit-matrix eigensolve requires it. Partial
    eigenproblems can instead use :meth:`aslinearoperator` or
    ``eigsolve(matrix_free=True)``.
    """

    __slots__ = (
        "operator",
        "sector",
        "_backend",
        "_compiled",
        "_matrix",
        "_linear_operators",
        "_adjoint_compiled",
        "_eigvals",
        "_eigvecs",
        "_matvec_timer",
        "_rmatvec_timer",
        "_solver_diagnostics",
    )

    def __init__(self, operator, sector):
        if not isinstance(operator, Operator):
            raise TypeError("'operator' must be an Operator expression.")
        if not hasattr(sector, "is_built"):
            raise TypeError("'sector' must be a fixed-sector object.")
        if not sector.is_built:
            raise RuntimeError(
                "The sector has not been built. Call sector.build() before "
                "constructing a Hamiltonian."
            )
        for attribute in ("modes", "basis", "dimension", "N", "from_vector"):
            if not hasattr(sector, attribute):
                raise TypeError(
                    "'sector' must provide modes, basis, dimension, N, and "
                    "from_vector()."
                )

        domains = operator_domains(operator)
        if domains and domains != frozenset((sector.modes,)):
            modes_name = type(sector.modes).__name__
            raise ValueError(
                "The Hamiltonian operator and sector use different "
                f"{modes_name} objects."
            )

        self.operator = operator
        self.sector = sector
        self._backend = _backend_for_modes(sector.modes)
        self._compiled = self._backend.compile_operator(operator)
        self._matrix = None
        self._linear_operators = {}
        self._adjoint_compiled = None
        self._eigvals = None
        self._eigvecs = None
        self._matvec_timer = _CallTimer()
        self._rmatvec_timer = _CallTimer()
        self._solver_diagnostics = None

    @staticmethod
    def _canonicalize_data_dtype(data):
        data = np.asarray(data)
        if np.iscomplexobj(data):
            if data.size == 0 or np.all(data.imag == 0):
                return np.asarray(data.real, dtype=np.float64)
            return np.asarray(data, dtype=np.complex128)
        return np.asarray(data, dtype=np.float64)

    def calc_matrix(self):
        """Construct and cache the Hamiltonian in CSC sparse format.

        Returns
        -------
        scipy.sparse.csc_matrix
            Square Hamiltonian matrix in the ordering of ``sector.basis``.
        """
        basis = self.sector.basis
        indices, indptr, data = self._backend.emit_csc(self._compiled, basis)
        data = self._canonicalize_data_dtype(data)
        matrix = csc_matrix(
            (data, indices, indptr),
            shape=(basis.dimension, basis.dimension),
        )
        matrix.sum_duplicates()
        matrix.eliminate_zeros()
        matrix.sort_indices()
        self._matrix = matrix
        return matrix

    @property
    def matrix(self):
        """scipy.sparse.csc_matrix: Lazily constructed Hamiltonian matrix."""
        if self._matrix is None:
            self.calc_matrix()
        return self._matrix

    @staticmethod
    def _normalize_execution(execution: str) -> str:
        """Normalize and validate a public matrix-free execution selector."""
        if not isinstance(execution, str):
            raise TypeError("'execution' must be a string.")
        execution = execution.strip().lower()
        allowed = ("numpy", "numba-serial", "numba-parallel", "mixed")
        if execution not in allowed:
            options = ", ".join(repr(item) for item in allowed)
            raise ValueError(
                f"Unknown execution mode {execution!r}; expected one of {options}."
            )
        return execution

    def _resolve_execution(self, execution: str) -> str:
        """Return the concrete backend selected for one execution request."""
        execution = self._normalize_execution(execution)
        callback = self._backend.resolve_execution
        if callback is not None:
            return callback(self._compiled, self.sector.basis, execution)
        if execution in ("numpy", "mixed"):
            return "numpy"
        raise NotImplementedError(
            f"execution={execution!r} is not available for "
            f"{type(self.sector.modes).__name__}."
        )

    def _execute_matvec(self, compiled, vector, execution: str):
        """Apply one compiled operator through the selected matrix-free backend."""
        callback = self._backend.matvec_execution
        if callback is not None:
            return callback(compiled, self.sector.basis, vector, execution)
        if execution in ("numpy", "mixed"):
            return self._backend.matvec(compiled, self.sector.basis, vector)
        raise NotImplementedError(
            f"execution={execution!r} is not available for "
            f"{type(self.sector.modes).__name__}."
        )

    def aslinearoperator(self, *, execution: str = "numpy") -> LinearOperator:
        """Return the Hamiltonian as a matrix-free SciPy ``LinearOperator``.

        Parameters
        ----------
        execution : {"numpy", "numba-serial", "numba-parallel", "mixed"}, optional
            Hamiltonian-action implementation. ``"numpy"`` uses the NumPy
            implementation. ``"numba-serial"`` and ``"numba-parallel"`` are
            strict Numba requests and raise if the operator or basis is not
            supported. ``"numba-parallel"`` uses Numba's configured thread
            count. ``"mixed"`` uses the statistics-specific conservative
            selection rule and does not enable parallel execution automatically.

        Returns
        -------
        scipy.sparse.linalg.LinearOperator
            Matrix-free operator with forward and adjoint actions and shape
            ``(sector.dimension, sector.dimension)``.

        Notes
        -----
        The explicit CSC matrix is not constructed. The adjoint action is
        obtained from the symbolic Hermitian adjoint of the operator.
        """
        execution = self._normalize_execution(execution)
        self._resolve_execution(execution)
        cached = self._linear_operators.get(execution)
        if cached is not None:
            return cached

        dimension = self.sector.dimension
        dtype = np.dtype(self._backend.dtype(self._compiled))

        if self._adjoint_compiled is None:
            self._adjoint_compiled = self._backend.compile_operator(
                self.operator.dag
            )

        def matvec(vector):
            start = perf_counter()
            try:
                return self._execute_matvec(
                    self._compiled,
                    vector,
                    execution,
                )
            finally:
                self._matvec_timer.record(perf_counter() - start)

        def rmatvec(vector):
            start = perf_counter()
            try:
                return self._execute_matvec(
                    self._adjoint_compiled,
                    vector,
                    execution,
                )
            finally:
                self._rmatvec_timer.record(perf_counter() - start)

        linear_operator = LinearOperator(
            shape=(dimension, dimension),
            matvec=matvec,
            rmatvec=rmatvec,
            dtype=dtype,
        )
        self._linear_operators[execution] = linear_operator
        return linear_operator

    def _matrix_free_execution_metadata(self, execution):
        """Return backend-specific matrix-free execution metadata."""
        callback = self._backend.matvec_metadata
        if callback is None:
            return {}
        metadata = callback(self._compiled, self.sector.basis, execution)
        return {} if metadata is None else dict(metadata)

    def _snapshot_solver_diagnostics(self, solver_wall_seconds, execution):
        """Freeze callback timings from the just-completed matrix-free solve."""
        matvec = self._matvec_timer
        rmatvec = self._rmatvec_timer
        apply_seconds = matvec.total_seconds + rmatvec.total_seconds
        solver_wall_seconds = float(solver_wall_seconds)
        overhead = max(0.0, solver_wall_seconds - apply_seconds)
        fraction = (
            min(1.0, apply_seconds / solver_wall_seconds)
            if solver_wall_seconds > 0.0
            else 0.0
        )
        calls = matvec.calls + rmatvec.calls
        states_processed = int(self.sector.dimension) * calls
        throughput = (
            states_processed / apply_seconds
            if apply_seconds > 0.0
            else 0.0
        )
        execution_resolved = self._resolve_execution(execution)
        metadata = self._matrix_free_execution_metadata(execution)
        chunks_per_matvec = metadata.get("chunks_per_matvec")
        estimated_total_chunks = (
            int(chunks_per_matvec) * calls
            if chunks_per_matvec is not None
            else None
        )
        return MatrixFreeSolveDiagnostics(
            dimension=int(self.sector.dimension),
            execution=execution,
            execution_resolved=execution_resolved,
            solver_wall_seconds=solver_wall_seconds,
            matvec_calls=matvec.calls,
            matvec_total_seconds=matvec.total_seconds,
            matvec_mean_seconds=matvec.mean_seconds,
            matvec_min_seconds=matvec.bounded_min_seconds,
            matvec_max_seconds=matvec.max_seconds,
            rmatvec_calls=rmatvec.calls,
            rmatvec_total_seconds=rmatvec.total_seconds,
            rmatvec_mean_seconds=rmatvec.mean_seconds,
            rmatvec_min_seconds=rmatvec.bounded_min_seconds,
            rmatvec_max_seconds=rmatvec.max_seconds,
            hamiltonian_apply_seconds=apply_seconds,
            solver_overhead_seconds=overhead,
            hamiltonian_apply_fraction=fraction,
            basis_states_processed=states_processed,
            basis_states_per_second=throughput,
            matvec_block_size=metadata.get("block_size"),
            chunks_per_matvec=chunks_per_matvec,
            estimated_total_chunks=estimated_total_chunks,
        )

    @property
    def solver_diagnostics(self):
        """Diagnostics from the most recent matrix-free eigensolve, or ``None``.

        The object is a frozen snapshot taken immediately when ARPACK returns.
        Subsequent uses of :meth:`aslinearoperator`, such as residual checks,
        do not alter the stored diagnostics. Explicit-matrix solves clear it.
        """
        return self._solver_diagnostics

    @property
    def compiler_stats(self):
        """dict[str, int]: Statistics for the lowered operator representation."""
        stats = getattr(self._compiled, "stats", None)
        if stats is None:
            return {}
        return stats() if callable(stats) else stats

    def toarray(self):
        """Return the Hamiltonian as a dense NumPy array.

        Returns
        -------
        numpy.ndarray
            Dense matrix with shape ``(sector.dimension, sector.dimension)``.
        """
        return self.matrix.toarray()

    def is_hermitian(self, atol=1e-12):
        """Test Hermiticity of the explicit Hamiltonian matrix.

        Parameters
        ----------
        atol : float, optional
            Absolute tolerance used when comparing the matrix with its
            Hermitian adjoint.

        Returns
        -------
        bool
            ``True`` when the matrix is Hermitian within ``atol``.
        """
        return is_hermitian_matrix(self.matrix, atol=atol)

    def estimate_solver_memory(self, *, k=2, ncv=None):
        """Estimate dominant ARPACK workspace without materializing the matrix.

        Parameters
        ----------
        k : int, optional
            Number of requested eigenpairs.
        ncv : int or None, optional
            Krylov subspace size. ``None`` uses SciPy's default policy.

        Returns
        -------
        ArpackMemoryEstimate
            Implementation-informed estimate of the dominant SciPy/ARPACK
            arrays. It excludes the Hamiltonian, basis storage, and general
            process overhead.
        """
        dtype = np.dtype(self._backend.dtype(self._compiled))
        return estimate_arpack_memory(
            self.sector.dimension,
            dtype,
            k=k,
            ncv=ncv,
        )

    def ncv_for_arpack_budget(self, budget_bytes, *, k=2, ncv=None):
        """Choose the largest ``ncv`` fitting an ARPACK workspace budget.

        Parameters
        ----------
        budget_bytes : int
            Maximum bytes assigned to the ARPACK arrays represented by
            :meth:`estimate_solver_memory`.
        k : int, optional
            Number of requested eigenpairs.
        ncv : int or None, optional
            Upper bound for the Krylov subspace size. ``None`` caps the search
            at SciPy's normal default.

        Returns
        -------
        int
            Largest admissible ``ncv`` within the requested workspace budget.

        Notes
        -----
        The budget excludes basis storage, Python and SciPy allocator overhead,
        linked libraries, and unrelated process memory.
        """
        dtype = np.dtype(self._backend.dtype(self._compiled))
        return arpack_ncv_for_memory_budget(
            self.sector.dimension,
            dtype,
            k=k,
            budget_bytes=budget_bytes,
            ncv=ncv,
        )

    def plan_solver_memory(
        self,
        total_memory_bytes,
        *,
        k=2,
        ncv=None,
        reserve_bytes=0,
        reserve_fraction=0.25,
        safety_factor=1.25,
    ):
        """Estimate memory for a matrix-free ARPACK eigensolve.

        Parameters
        ----------
        total_memory_bytes : int
            Total memory envelope available to the process.
        k : int, optional
            Number of requested eigenpairs.
        ncv : int or None, optional
            Krylov subspace size. ``None`` uses SciPy's default policy.
        reserve_bytes : int, optional
            Fixed number of bytes withheld from the solver plan.
        reserve_fraction : float, optional
            Fraction of ``total_memory_bytes`` withheld in addition to
            ``reserve_bytes``. Must satisfy ``0 <= value < 1``.
        safety_factor : float, optional
            Multiplicative factor applied to the ARPACK array estimate. Must be
            at least one.

        Returns
        -------
        SolverMemoryPlan
            Estimated ARPACK storage, persistent basis storage, reserved
            memory, remaining headroom, and whether the plan fits.
        """
        dtype = np.dtype(self._backend.dtype(self._compiled))
        basis_bytes = int(
            getattr(self.sector.basis, "estimated_execution_storage_bytes", 0)
        )
        return plan_solver_memory(
            self.sector.dimension,
            dtype,
            total_memory_bytes=total_memory_bytes,
            k=k,
            ncv=ncv,
            reserve_bytes=reserve_bytes,
            reserve_fraction=reserve_fraction,
            safety_factor=safety_factor,
            basis_execution_bytes=basis_bytes,
        )

    def ncv_for_memory_budget(
        self,
        total_memory_bytes,
        *,
        k=2,
        ncv=None,
        reserve_bytes=0,
        reserve_fraction=0.25,
        safety_factor=1.25,
    ):
        """Choose the largest ``ncv`` fitting a total-memory envelope.

        Parameters
        ----------
        total_memory_bytes : int
            Total memory envelope available to the process.
        k : int, optional
            Number of requested eigenpairs.
        ncv : int or None, optional
            Upper bound for the Krylov subspace size. ``None`` caps the search
            at SciPy's normal default.
        reserve_bytes : int, optional
            Fixed number of bytes withheld from the solver plan.
        reserve_fraction : float, optional
            Fraction of total memory withheld in addition to ``reserve_bytes``.
        safety_factor : float, optional
            Multiplicative factor applied to the ARPACK array estimate.

        Returns
        -------
        int
            Largest admissible ``ncv`` satisfying the memory plan.
        """
        dtype = np.dtype(self._backend.dtype(self._compiled))
        basis_bytes = int(
            getattr(self.sector.basis, "estimated_execution_storage_bytes", 0)
        )
        return ncv_for_total_memory(
            self.sector.dimension,
            dtype,
            total_memory_bytes=total_memory_bytes,
            k=k,
            ncv=ncv,
            reserve_bytes=reserve_bytes,
            reserve_fraction=reserve_fraction,
            safety_factor=safety_factor,
            basis_execution_bytes=basis_bytes,
        )

    def eigsolve(
        self,
        *,
        sparse=True,
        k=2,
        which="SA",
        tol=1e-10,
        maxiter=None,
        ncv=None,
        v0=None,
        check_hermitian=True,
        matrix_free=False,
        execution="numpy",
    ):
        """Solve the Hermitian Hamiltonian eigenproblem.

        Parameters
        ----------
        sparse : bool, optional
            Use ARPACK through :func:`scipy.sparse.linalg.eigsh` when ``True``
            and fewer than all eigenpairs are requested. Otherwise use the
            dense Hermitian solver :func:`scipy.linalg.eigh`. Matrix-free
            solves require ``sparse=True``.
        k : int or None, optional
            Number of eigenpairs to return. ``None`` requests the complete
            spectrum and therefore requires an explicit matrix.
        which : {"SA", "LA", "SM", "LM"}, optional
            Spectral selection rule: smallest algebraic, largest algebraic,
            smallest magnitude, or largest magnitude, respectively.
        tol : float, optional
            ARPACK convergence tolerance.
        maxiter : int or None, optional
            Maximum number of ARPACK iterations. ``None`` uses SciPy's
            default.
        ncv : int or None, optional
            Number of Lanczos vectors used by ARPACK. It must satisfy
            ``k < ncv <= dimension``. ``None`` uses SciPy's default policy.
        v0 : array_like or None, optional
            Initial vector for ARPACK. ``None`` lets SciPy choose the starting
            vector.
        check_hermitian : bool, optional
            Check Hermiticity before an explicit-matrix solve. Matrix-free
            solves require ``False`` because an exact check would construct the
            matrix.
        matrix_free : bool, optional
            Apply the Hamiltonian through a SciPy ``LinearOperator`` instead of
            constructing the CSC matrix. Only partial sparse solves with
            ``k < sector.dimension`` are supported.
        execution : {"numpy", "numba-serial", "numba-parallel", "mixed"}, optional
            Hamiltonian-action implementation used when ``matrix_free=True``.
            Explicit-matrix solves require ``"numpy"``. Strict Numba modes
            raise when the current operator or basis is unsupported.

        Returns
        -------
        eigenvalues : numpy.ndarray
            Requested eigenvalues ordered according to ``which``.
        eigenvectors : numpy.ndarray
            Corresponding normalized eigenvectors stored by column in the
            sector-basis ordering.

        Notes
        -----
        The returned eigenpairs are also stored in :attr:`eigvals` and
        :attr:`eigvecs`. Use :meth:`eigenstate` or :meth:`eigenstates` to
        recover basis-aware Fock vectors.
        """
        execution = self._normalize_execution(execution)
        if not matrix_free and execution != "numpy":
            raise ValueError(
                "'execution' selects matrix-free Hamiltonian application and "
                "requires matrix_free=True."
            )
        if matrix_free:
            if check_hermitian:
                raise ValueError(
                    "Exact Hermiticity checking would materialize the matrix; "
                    "pass check_hermitian=False for a matrix-free solve."
                )
            self._matvec_timer.reset()
            self._rmatvec_timer.reset()
            self._solver_diagnostics = None
            start = perf_counter()
            eigenvalues, eigenvectors = solve_hermitian_operator(
                self.aslinearoperator(execution=execution),
                sparse=sparse,
                k=k,
                which=which,
                tol=tol,
                maxiter=maxiter,
                ncv=ncv,
                v0=v0,
            )
            self._solver_diagnostics = self._snapshot_solver_diagnostics(
                perf_counter() - start,
                execution,
            )
        else:
            self._solver_diagnostics = None
            eigenvalues, eigenvectors = solve_hermitian_matrix(
                self.matrix,
                sparse=sparse,
                k=k,
                which=which,
                tol=tol,
                maxiter=maxiter,
                ncv=ncv,
                v0=v0,
                check_hermitian=check_hermitian,
            )
        self._eigvals = eigenvalues
        self._eigvecs = eigenvectors
        return eigenvalues, eigenvectors

    def eigenstate(self, index=0):
        """Return one computed eigenvector as a basis-aware Fock ket.

        Parameters
        ----------
        index : int, optional
            Column index in the eigenvector array from the most recent
            :meth:`eigsolve`. Negative indices follow Python indexing.

        Returns
        -------
        FockVector
            Eigenstate associated with ``eigvals[index]``.
        """
        if self._eigvecs is None:
            raise RuntimeError("Call eigsolve() before requesting an eigenstate.")
        if not isinstance(index, Integral):
            raise TypeError("'index' must be an integer.")
        index = int(index)
        n_states = self._eigvecs.shape[1]
        if index < 0:
            index += n_states
        if index < 0 or index >= n_states:
            raise IndexError("Eigenstate index is outside the computed range.")
        return self.sector.from_vector(self._eigvecs[:, index])

    def eigenstates(self):
        """Return all computed eigenvectors as basis-aware Fock kets.

        Returns
        -------
        tuple[FockVector, ...]
            Eigenstates in the same order as :attr:`eigvals`.
        """
        if self._eigvecs is None:
            raise RuntimeError("Call eigsolve() before requesting eigenstates.")
        return tuple(
            self.sector.from_vector(self._eigvecs[:, index])
            for index in range(self._eigvecs.shape[1])
        )

    @property
    def eigvals(self):
        """numpy.ndarray or None: Eigenvalues from the most recent eigensolve."""
        return self._eigvals

    @property
    def eigvecs(self):
        """numpy.ndarray or None: Eigenvectors from the most recent eigensolve."""
        return self._eigvecs
