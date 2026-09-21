"""Hamiltonian matrix construction and Hermitian eigensolution."""

from __future__ import annotations

from numbers import Integral

import numpy as np
from scipy import linalg
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import eigsh

from ._algebra import Operator, operator_modes
from ._basis import FockState, NParticleSector
from ._execution import compile_operator


class Hamiltonian:
    """Hamiltonian represented by symbolic fermionic algebra in an ``N`` sector.

    Parameters
    ----------
    operator : Operator
        Symbolic fermionic Hamiltonian.
    sector : NParticleSector
        Fixed-particle-number sector in which the matrix is constructed.

    Notes
    -----
    Number-changing terms have no matrix elements within a fixed-``N`` sector
    and are therefore projected out. EDinPy does not currently provide a full
    Fock-space or fermion-parity sector.
    """

    __slots__ = (
        "operator",
        "sector",
        "_compiled",
        "_matrix",
        "_eigvals",
        "_eigvecs",
    )

    def __init__(self, operator, sector):
        """Validate ownership and initialize lazy matrix/eigensystem storage."""
        if not isinstance(operator, Operator):
            raise TypeError("'operator' must be a fermionic Operator expression.")
        if not isinstance(sector, NParticleSector):
            raise TypeError("'sector' must be an NParticleSector instance.")

        modes = operator_modes(operator)
        if modes is not None and modes is not sector.modes:
            raise ValueError(
                "The Hamiltonian operator and NParticleSector use different "
                "FermionModes objects."
            )

        self.operator = operator
        self.sector = sector
        self._compiled = compile_operator(operator)
        self._matrix = None
        self._eigvals = None
        self._eigvecs = None

    def _emit_generic_csc(self):
        """Construct CSC arrays by applying the compiled operator state by state."""
        basis = self.sector.basis
        lookup = {state: index for index, state in enumerate(basis.states)}
        indices = []
        data = []
        indptr = [0]

        for column, state_int in enumerate(basis.states):
            ket = FockState(
                state_int,
                n_modes=basis.n_modes,
                index=column,
            )
            output = self._compiled.apply(ket)
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

    @staticmethod
    def _canonicalize_data_dtype(data):
        """Normalize matrix-element storage without unnecessary large copies."""
        data = np.asarray(data)
        if np.iscomplexobj(data):
            if data.size == 0 or np.all(data.imag == 0):
                return np.asarray(data.real, dtype=np.float64)
            return np.asarray(data, dtype=np.complex128)
        return np.asarray(data, dtype=np.float64)

    def calc_matrix(self):
        """Construct and cache the Hamiltonian in compressed sparse-column form.

        Returns
        -------
        scipy.sparse.csc_matrix
            Hamiltonian matrix. Real-valued Hamiltonians use ``float64``;
            genuinely complex Hamiltonians use ``complex128``.
        """
        basis = self.sector.basis
        if self._compiled.fully_lowered:
            indices, indptr, data = self._compiled.emit_csc(
                basis.states,
                n_modes=basis.n_modes,
            )
        else:
            indices, indptr, data = self._emit_generic_csc()

        data = self._canonicalize_data_dtype(data)
        matrix = csc_matrix(
            (data, indices, indptr),
            shape=(basis.dimension, basis.dimension),
        )
        matrix.sum_duplicates()
        matrix.eliminate_zeros()
        self._matrix = matrix
        return matrix

    @property
    def matrix(self):
        """scipy.sparse.csc_matrix: Lazily constructed Hamiltonian matrix."""
        if self._matrix is None:
            self.calc_matrix()
        return self._matrix

    def toarray(self):
        """Return the Hamiltonian as a dense NumPy array.

        Returns
        -------
        numpy.ndarray
            Dense Hamiltonian matrix.
        """
        return self.matrix.toarray()

    def is_hermitian(self, atol=1e-12):
        """Return whether the Hamiltonian matrix is Hermitian within tolerance.

        Parameters
        ----------
        atol : float, optional
            Absolute tolerance applied to the largest element of ``H-H†``.

        Returns
        -------
        bool
            ``True`` when the matrix is Hermitian within ``atol``.
        """
        difference = self.matrix - self.matrix.getH()
        if difference.nnz == 0:
            return True
        return bool(np.max(np.abs(difference.data)) <= atol)

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
    ):
        """Solve the Hermitian Hamiltonian eigenproblem.

        Parameters
        ----------
        sparse : bool, optional
            Use ARPACK through :func:`scipy.sparse.linalg.eigsh` when ``True``
            and ``k`` is smaller than the matrix dimension. Use LAPACK through
            :func:`scipy.linalg.eigh` otherwise.
        k : int or None, optional
            Number of requested eigenpairs. ``None`` requests the complete
            eigensystem.
        which : {'SA', 'LA', 'SM', 'LM'}, optional
            Requested part of the spectrum. The strings follow SciPy/ARPACK:
            smallest algebraic, largest algebraic, smallest magnitude, or
            largest magnitude, respectively.
        tol : float, optional
            ARPACK convergence tolerance. The default is appropriate for the
            double-precision matrices used by EDinPy.
        maxiter : int, optional
            Maximum number of ARPACK iterations.
        ncv : int, optional
            Number of Lanczos vectors generated by ARPACK.
        v0 : array_like, optional
            Initial ARPACK vector. EDinPy does not generate warm starts
            automatically because a single-vector warm start can miss an
            exactly crossing symmetry sector.
        check_hermitian : bool, optional
            Verify Hermiticity before calling a Hermitian eigensolver.

        Returns
        -------
        eigenvalues : numpy.ndarray
            Eigenvalues ordered according to ``which``.
        eigenvectors : numpy.ndarray
            Corresponding eigenvectors stored by columns.

        Raises
        ------
        ValueError
            If ``k`` or ``which`` is invalid, or if the matrix is not
            Hermitian when ``check_hermitian=True``.

        Notes
        -----
        ``eigsh`` is an implicitly restarted Lanczos method provided by
        ARPACK. Small Ritz residuals establish convergence of returned
        eigenpairs but do not guarantee that a single-vector Lanczos run has
        recovered every vector in an exactly degenerate eigenspace.

        References
        ----------
        .. [1] R. B. Lehoucq, D. C. Sorensen, and C. Yang, *ARPACK Users'
           Guide: Solution of Large-Scale Eigenvalue Problems with Implicitly
           Restarted Arnoldi Methods*, SIAM (1998), doi:10.1137/1.9780898719628.
        .. [2] P. Virtanen et al., "SciPy 1.0: Fundamental Algorithms for
           Scientific Computing in Python," *Nature Methods* 17, 261--272
           (2020), doi:10.1038/s41592-019-0686-2.
        """
        matrix = self.matrix
        dimension = matrix.shape[0]

        if check_hermitian and not self.is_hermitian():
            raise ValueError(
                "Hamiltonian.eigsolve requires a Hermitian matrix."
            )
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
            order = _spectral_order(eigenvalues, which)
            eigenvalues = eigenvalues[order]
            eigenvectors = eigenvectors[:, order]
        else:
            dense = matrix.toarray()
            if k is not None and which in {"SA", "LA"} and k < dimension:
                if which == "SA":
                    subset = (0, k - 1)
                else:
                    subset = (dimension - k, dimension - 1)
                eigenvalues, eigenvectors = linalg.eigh(
                    dense,
                    subset_by_index=subset,
                    check_finite=False,
                )
                order = _spectral_order(eigenvalues, which)
                eigenvalues = eigenvalues[order]
                eigenvectors = eigenvectors[:, order]
            else:
                eigenvalues, eigenvectors = linalg.eigh(
                    dense,
                    check_finite=False,
                )
                order = _spectral_order(eigenvalues, which)
                if k is not None:
                    order = order[:k]
                eigenvalues = eigenvalues[order]
                eigenvectors = eigenvectors[:, order]

        self._eigvals = eigenvalues
        self._eigvecs = eigenvectors
        return eigenvalues, eigenvectors

    @property
    def eigvals(self):
        """numpy.ndarray or None: Eigenvalues from the most recent eigensolve."""
        return self._eigvals

    @property
    def eigvecs(self):
        """numpy.ndarray or None: Eigenvectors from the most recent eigensolve."""
        return self._eigvecs


def _spectral_order(eigenvalues, which):
    """Return indices that order eigenvalues according to SciPy/ARPACK semantics."""
    values = np.asarray(eigenvalues)
    if which == "SA":
        return np.argsort(values)
    if which == "LA":
        return np.argsort(values)[::-1]
    if which == "SM":
        return np.argsort(np.abs(values))
    return np.argsort(np.abs(values))[::-1]
