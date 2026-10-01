"""Profile matrix-free ARPACK memory as a function of ``ncv``.

Each ``ncv`` value is measured in a fresh subprocess so that peak-RSS state
from one eigensolve does not contaminate another.  The reported ARPACK memory
estimate covers dominant solver arrays only; observed process RSS includes the
full Python/SciPy process.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import resource
import subprocess
import sys
import time
from math import comb
from pathlib import Path

import numpy as np
import scipy


STATISTICS = ("fermion", "boson")
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def _current_rss_mib():
    status = Path("/proc/self/status")
    if status.exists():
        for line in status.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
    return _peak_rss_mib()


def _peak_rss_mib():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() == "Darwin":
        return value / (1024.0 * 1024.0)
    return value / 1024.0


def _environment_metadata():
    import edinpy

    return {
        "python": sys.version.split()[0],
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "edinpy": getattr(edinpy, "__version__", "unknown"),
    }


def _build_hamiltonian(statistics_name, n_modes, particles):
    if statistics_name == "fermion":
        from edinpy import fermion as ed

        modes = ed.FermionModes(ed.DoF(n_modes, name="mode"))
    else:
        from edinpy import boson as ed

        modes = ed.BosonModes(ed.DoF(n_modes, name="mode"))

    sector = ed.NParticleSector(modes, N=particles).build()
    H = 0
    for i in range(n_modes - 1):
        H += ed.Hopping(i, i + 1, -1.0, modes)
        H += ed.DensityDensity(i, i + 1, 0.35, modes)
    for i in range(n_modes):
        H += ed.Onsite(i, 0.03 * (i - 0.5 * (n_modes - 1)), modes)
    return ed.Hamiltonian(H, sector)


def _worker(args):
    ham = _build_hamiltonian(args.statistics, args.modes, args.particles)
    operator = ham.aslinearoperator(execution=args.execution)
    estimate = ham.estimate_solver_memory(k=args.k, ncv=args.ncv)

    rng = np.random.default_rng(args.seed)
    v0 = rng.normal(size=ham.sector.dimension)
    if np.issubdtype(operator.dtype, np.complexfloating):
        v0 = v0.astype(operator.dtype)
    else:
        v0 = v0.astype(np.float64)

    gc.collect()
    rss_before = _current_rss_mib()
    peak_before = _peak_rss_mib()

    start = time.perf_counter()
    eigenvalues, eigenvectors = ham.eigsolve(
        sparse=True,
        k=args.k,
        which=args.which,
        tol=args.tol,
        maxiter=args.maxiter,
        ncv=args.ncv,
        v0=v0,
        check_hermitian=False,
        matrix_free=True,
        execution=args.execution,
    )
    solve_seconds = time.perf_counter() - start
    solver_diagnostics = ham.solver_diagnostics

    rss_after_solve = _current_rss_mib()
    peak_after_solve = _peak_rss_mib()

    residuals = np.linalg.norm(
        operator @ eigenvectors - eigenvectors * eigenvalues,
        axis=0,
    )

    estimate_dict = estimate.as_dict()
    estimate_dict["estimated_arpack_mib"] = (
        estimate.estimated_arpack_bytes / (1024.0 * 1024.0)
    )

    payload = {
        "statistics": args.statistics,
        "modes": args.modes,
        "particles": args.particles,
        "dimension": ham.sector.dimension,
        "k": args.k,
        "ncv": args.ncv,
        "which": args.which,
        "tol": args.tol,
        "solver_dtype": str(operator.dtype),
        "execution": args.execution,
        "estimate": estimate_dict,
        "rss_before_solve_mib": rss_before,
        "rss_after_solve_mib": rss_after_solve,
        "peak_before_solve_mib": peak_before,
        "peak_after_solve_mib": peak_after_solve,
        "peak_increase_from_current_rss_mib": max(0.0, peak_after_solve - rss_before),
        "peak_increase_over_prior_peak_mib": max(0.0, peak_after_solve - peak_before),
        "eigsolve_seconds": solve_seconds,
        "solver_diagnostics": (
            None
            if solver_diagnostics is None
            else solver_diagnostics.as_dict()
        ),
        "residual_max": float(np.max(residuals)),
        "matrix_materialized": ham._matrix is not None,
        "eigenvalues": [float(value) for value in eigenvalues],
    }
    print(json.dumps(payload))


def _worker_command(args, ncv):
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--statistics",
        args.statistics,
        "--modes",
        str(args.modes),
        "--particles",
        str(args.particles),
        "--k",
        str(args.k),
        "--ncv",
        str(ncv),
        "--which",
        args.which,
        "--execution",
        args.execution,
        "--tol",
        repr(args.tol),
        "--seed",
        str(args.seed),
    ]
    if args.maxiter is not None:
        command.extend(("--maxiter", str(args.maxiter)))
    return command


def _run_ncv(args, ncv):
    result = subprocess.run(
        _worker_command(args, ncv),
        check=True,
        text=True,
        capture_output=True,
        env=os.environ.copy(),
    )
    return json.loads(result.stdout.strip().splitlines()[-1])


def _resolve_json_path(path):
    if path is None or path.is_absolute() or path.parent != Path("."):
        return path
    return RESULTS_DIR / path


def _print_results(results):
    print(
        f"{'ncv':>5} {'ARPACK MiB':>13} {'observed +MiB':>14} "
        f"{'solve [s]':>11} {'mv calls':>9} {'H [s]':>10} "
        f"{'H %':>7} {'residual':>12}"
    )
    print("-" * 101)
    for result in results:
        diagnostics = result.get("solver_diagnostics") or {}
        print(
            f"{result['ncv']:>5d} "
            f"{result['estimate']['estimated_arpack_mib']:>13.1f} "
            f"{result['peak_increase_from_current_rss_mib']:>14.1f} "
            f"{result['eigsolve_seconds']:>11.4f} "
            f"{diagnostics.get('matvec_calls', 0):>9d} "
            f"{diagnostics.get('hamiltonian_apply_seconds', 0.0):>10.3f} "
            f"{100.0 * diagnostics.get('hamiltonian_apply_fraction', 0.0):>6.1f}% "
            f"{result['residual_max']:>12.3e}"
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Measure matrix-free SciPy/ARPACK memory and runtime across ncv "
            "choices in isolated subprocesses."
        )
    )
    parser.add_argument("--statistics", choices=STATISTICS, default="fermion")
    parser.add_argument("--modes", type=int, default=16)
    parser.add_argument("--particles", type=int, default=8)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--ncvs", type=int, nargs="+", default=(4, 8, 12, 20))
    parser.add_argument("--which", choices=("SA", "LA", "SM", "LM"), default="SA")
    parser.add_argument(
        "--execution",
        choices=("numpy", "numba-serial", "numba-parallel", "mixed"),
        default="numpy",
    )
    parser.add_argument("--tol", type=float, default=1e-10)
    parser.add_argument("--maxiter", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--json", type=Path, default=None)

    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--ncv", type=int, help=argparse.SUPPRESS)

    args = parser.parse_args()
    if args.modes <= 0:
        parser.error("--modes must be positive")
    if args.particles < 0:
        parser.error("--particles must be non-negative")
    if args.statistics == "fermion" and args.particles > args.modes:
        parser.error("fermionic --particles must not exceed --modes")
    if args.k <= 0:
        parser.error("--k must be positive")
    if args.worker:
        if args.ncv is None:
            parser.error("worker mode requires --ncv")
        ncvs = (args.ncv,)
    else:
        ncvs = tuple(dict.fromkeys(args.ncvs))
    if any(ncv <= args.k for ncv in ncvs):
        parser.error("every ncv must be greater than --k")

    dimension = (
        comb(args.modes, args.particles)
        if args.statistics == "fermion"
        else comb(args.particles + args.modes - 1, args.particles)
    )
    if any(ncv > dimension for ncv in ncvs):
        parser.error("every ncv must not exceed the Hilbert-space dimension")
    args.ncvs = ncvs
    args.json = _resolve_json_path(args.json)
    return args


def main():
    args = parse_args()
    if args.worker:
        _worker(args)
        return

    metadata = _environment_metadata()
    dimension = (
        comb(args.modes, args.particles)
        if args.statistics == "fermion"
        else comb(args.particles + args.modes - 1, args.particles)
    )
    print(
        f"EDinPy {metadata['edinpy']} | Python {metadata['python']} | "
        f"NumPy {metadata['numpy']} | SciPy {metadata['scipy']}"
    )
    print(
        f"statistics={args.statistics}, modes={args.modes}, "
        f"particles={args.particles}, dimension={dimension}, k={args.k}, "
        f"execution={args.execution}"
    )
    print()

    results = [_run_ncv(args, ncv) for ncv in args.ncvs]
    _print_results(results)

    if args.json is not None:
        payload = {
            "environment": metadata,
            "parameters": {
                "statistics": args.statistics,
                "modes": args.modes,
                "particles": args.particles,
                "k": args.k,
                "ncvs": list(args.ncvs),
                "which": args.which,
                "execution": args.execution,
                "tol": args.tol,
                "maxiter": args.maxiter,
                "seed": args.seed,
            },
            "results": results,
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"\nWrote {args.json}")


if __name__ == "__main__":
    main()
