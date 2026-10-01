"""Shared statistics-independent foundations for EDinPy."""

from ._algebra import (
    Operator,
    OperatorProduct,
    OperatorSum,
    Scalar,
    SingleModeOperator,
    operator_domains,
    set_notation,
)
from ._hamiltonian import (
    Hamiltonian,
    HamiltonianBackend,
    MatrixFreeSolveDiagnostics,
)
from ._modes import DoF, DiscreteModes
from ._solvers import (
    ArpackMemoryEstimate,
    SolverMemoryPlan,
    arpack_ncv_for_memory_budget,
    estimate_arpack_memory,
    ncv_for_total_memory,
    plan_solver_memory,
    resolve_arpack_ncv,
)
from ._states import (
    BaseFockState,
    FockBra,
    FockVector,
    NullState,
    StateSum,
)

__all__ = [
    "DoF",
    "Hamiltonian",
    "HamiltonianBackend",
    "MatrixFreeSolveDiagnostics",
    "DiscreteModes",
    "ArpackMemoryEstimate",
    "SolverMemoryPlan",
    "estimate_arpack_memory",
    "resolve_arpack_ncv",
    "arpack_ncv_for_memory_budget",
    "plan_solver_memory",
    "ncv_for_total_memory",
    "Operator",
    "Scalar",
    "OperatorSum",
    "OperatorProduct",
    "SingleModeOperator",
    "operator_domains",
    "set_notation",
    "BaseFockState",
    "FockVector",
    "FockBra",
    "StateSum",
    "NullState",
]
