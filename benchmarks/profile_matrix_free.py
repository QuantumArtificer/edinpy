"""Compare explicit CSC and matrix-free EDinPy Hamiltonian execution.

Each representation is benchmarked in a fresh subprocess so that allocator
state, cached matrices, and peak-RSS accounting from one representation do not
contaminate the other.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
from math import comb, sqrt
from pathlib import Path

import numpy as np
import scipy


REPRESENTATIONS = ("csc", "matrix_free")
STATISTICS = ("fermion", "boson")
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def _current_rss_mib():
    """Return current resident memory in MiB when the platform exposes it."""
    status = Path("/proc/self/status")
    if status.exists():
        for line in status.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
    return _peak_rss_mib()


def _peak_rss_mib():
    """Return process peak resident memory in MiB."""
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


def _build_problem(statistics_name, n_modes, particles):
    if statistics_name == "fermion":
        from edinpy import fermion as ed

        modes = ed.FermionModes(ed.DoF(n_modes, name="mode"))
    else:
        from edinpy import boson as ed

        modes = ed.BosonModes(ed.DoF(n_modes, name="mode"))

    start = time.perf_counter()
    sector = ed.NParticleSector(modes, N=particles).build()
    basis_seconds = time.perf_counter() - start

    start = time.perf_counter()
    H = 0
    for i in range(n_modes - 1):
        H += ed.Hopping(i, i + 1, -1.0, modes)
        H += ed.DensityDensity(i, i + 1, 0.35, modes)
    for i in range(n_modes):
        H += ed.Onsite(i, 0.03 * (i - 0.5 * (n_modes - 1)), modes)
    expression_seconds = time.perf_counter() - start

    start = time.perf_counter()
    ham = ed.Hamiltonian(H, sector)
    compile_seconds = time.perf_counter() - start
    return ham, basis_seconds, expression_seconds, compile_seconds


def _matrix_storage_bytes(matrix):
    return int(matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes)


def _worker(args):
    ham, basis_seconds, expression_seconds, compile_seconds = _build_problem(
        args.statistics,
        args.modes,
        args.particles,
    )
    dimension = ham.sector.dimension

    rss_after_compile = _current_rss_mib()
    peak_after_compile = _peak_rss_mib()

    start = time.perf_counter()
    if args.representation == "csc":
        representation = ham.matrix
        matrix_bytes = _matrix_storage_bytes(representation)
        nnz = int(representation.nnz)
    else:
        representation = ham.aslinearoperator()
        matrix_bytes = 0
        nnz = None
    representation_seconds = time.perf_counter() - start

    rss_after_representation = _current_rss_mib()
    peak_after_representation = _peak_rss_mib()

    rng = np.random.default_rng(args.seed)
    vector = rng.normal(size=dimension)
    vector /= np.linalg.norm(vector)

    for _ in range(args.warmup):
        _ = representation @ vector

    matvec_samples = []
    for _ in range(args.repeat):
        start = time.perf_counter()
        result = representation @ vector
        matvec_samples.append(time.perf_counter() - start)

    rss_after_matvec = _current_rss_mib()
    peak_after_matvec = _peak_rss_mib()

    eigsolve_seconds = None
    residual_max = None
    eigenvalues = None
    if not args.no_eigsolve:
        v0 = np.full(dimension, 1.0 / sqrt(dimension), dtype=float)
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
            matrix_free=(args.representation == "matrix_free"),
        )
        eigsolve_seconds = time.perf_counter() - start

        action = ham.aslinearoperator() if args.representation == "matrix_free" else ham.matrix
        residuals = np.linalg.norm(
            action @ eigenvectors - eigenvectors * eigenvalues,
            axis=0,
        )
        residual_max = float(np.max(residuals))
        eigenvalues = [float(value) for value in eigenvalues]

    solver_dtype = np.dtype(representation.dtype)
    vector_storage_bytes = int(dimension * solver_dtype.itemsize)
    ncv_vector_storage_bytes = (
        None if args.ncv is None else int(args.ncv * vector_storage_bytes)
    )
    solver_memory = None
    if args.k < dimension:
        solver_memory = ham.estimate_solver_memory(k=args.k, ncv=args.ncv).as_dict()

    payload = {
        "statistics": args.statistics,
        "representation": args.representation,
        "modes": args.modes,
        "particles": args.particles,
        "dimension": dimension,
        "basis_seconds": basis_seconds,
        "expression_seconds": expression_seconds,
        "compile_seconds": compile_seconds,
        "representation_seconds": representation_seconds,
        "matvec_seconds_median": statistics.median(matvec_samples),
        "matvec_seconds_min": min(matvec_samples),
        "matvec_seconds_max": max(matvec_samples),
        "eigsolve_seconds": eigsolve_seconds,
        "residual_max": residual_max,
        "eigenvalues": eigenvalues,
        "matrix_materialized": ham._matrix is not None,
        "matrix_storage_bytes": matrix_bytes,
        "solver_vector_storage_bytes": vector_storage_bytes,
        "ncv_vector_storage_bytes": ncv_vector_storage_bytes,
        "solver_memory_estimate": solver_memory,
        "nnz": nnz,
        "rss_after_compile_mib": rss_after_compile,
        "rss_after_representation_mib": rss_after_representation,
        "rss_after_matvec_mib": rss_after_matvec,
        "peak_after_compile_mib": peak_after_compile,
        "peak_after_representation_mib": peak_after_representation,
        "peak_after_matvec_mib": peak_after_matvec,
        "peak_final_mib": _peak_rss_mib(),
    }
    print(json.dumps(payload))


def _worker_command(args, representation):
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--statistics",
        args.statistics,
        "--representation",
        representation,
        "--modes",
        str(args.modes),
        "--particles",
        str(args.particles),
        "--k",
        str(args.k),
        "--which",
        args.which,
        "--tol",
        repr(args.tol),
        "--repeat",
        str(args.repeat),
        "--warmup",
        str(args.warmup),
        "--seed",
        str(args.seed),
    ]
    if args.ncv is not None:
        command.extend(("--ncv", str(args.ncv)))
    if args.maxiter is not None:
        command.extend(("--maxiter", str(args.maxiter)))
    if args.no_eigsolve:
        command.append("--no-eigsolve")
    return command


def _run_representation(args, representation):
    result = subprocess.run(
        _worker_command(args, representation),
        check=True,
        text=True,
        capture_output=True,
        env=os.environ.copy(),
    )
    return json.loads(result.stdout.strip().splitlines()[-1])


def _format_mib(value):
    return f"{value:10.1f}"


def _print_results(results):
    print(
        f"{'repr':<12} {'dim':>10} {'nnz':>10} {'matrix MiB':>11} "
        f"{'matvec [s]':>12} {'eigsolve [s]':>13} {'peak RSS MiB':>13}"
    )
    print("-" * 88)
    for result in results:
        nnz = "-" if result["nnz"] is None else str(result["nnz"])
        matrix_mib = result["matrix_storage_bytes"] / (1024.0 * 1024.0)
        eigsolve = (
            "-"
            if result["eigsolve_seconds"] is None
            else f"{result['eigsolve_seconds']:.6f}"
        )
        print(
            f"{result['representation']:<12} {result['dimension']:>10d} "
            f"{nnz:>10} {matrix_mib:>11.1f} "
            f"{result['matvec_seconds_median']:>12.6f} "
            f"{eigsolve:>13} {_format_mib(result['peak_final_mib'])}"
        )

    if len(results) == 2:
        by_name = {result["representation"]: result for result in results}
        csc = by_name.get("csc")
        matrix_free = by_name.get("matrix_free")
        if csc is not None and matrix_free is not None:
            peak_saved = csc["peak_final_mib"] - matrix_free["peak_final_mib"]
            print(f"\nPeak RSS difference (CSC - matrix_free): {peak_saved:.1f} MiB")
            vector_mib = matrix_free["solver_vector_storage_bytes"] / (1024.0 * 1024.0)
            print(f"One solver vector: {vector_mib:.1f} MiB")
            if matrix_free["ncv_vector_storage_bytes"] is not None:
                ncv_mib = matrix_free["ncv_vector_storage_bytes"] / (1024.0 * 1024.0)
                print(
                    f"ncv × one-vector storage lower bound: {ncv_mib:.1f} MiB "
                    "(not total ARPACK workspace)"
                )
            estimate = matrix_free.get("solver_memory_estimate")
            if estimate is not None:
                peak_mib = estimate["estimated_arpack_bytes"] / (1024.0 * 1024.0)
                print(
                    f"Estimated dominant ARPACK peak arrays: {peak_mib:.1f} MiB "
                    f"(resolved ncv={estimate['ncv']})"
                )
            if csc["eigenvalues"] is not None and matrix_free["eigenvalues"] is not None:
                error = np.max(
                    np.abs(
                        np.asarray(csc["eigenvalues"])
                        - np.asarray(matrix_free["eigenvalues"])
                    )
                )
                print(f"Eigenvalue max |Δ|: {error:.3e}")


def _resolve_json_path(path):
    """Route bare result filenames into ``benchmarks/results``."""
    if path is None or path.is_absolute() or path.parent != Path("."):
        return path
    return RESULTS_DIR / path


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare EDinPy CSC and matrix-free Hamiltonian execution in "
            "isolated subprocesses."
        )
    )
    parser.add_argument("--statistics", choices=STATISTICS, default="fermion")
    parser.add_argument("--modes", type=int, default=14)
    parser.add_argument("--particles", type=int, default=7)
    parser.add_argument(
        "--representations",
        nargs="+",
        choices=REPRESENTATIONS,
        default=REPRESENTATIONS,
    )
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--which", choices=("SA", "LA", "SM", "LM"), default="SA")
    parser.add_argument("--tol", type=float, default=1e-10)
    parser.add_argument("--ncv", type=int, default=8)
    parser.add_argument("--maxiter", type=int, default=None)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--no-eigsolve", action="store_true")
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help=(
            "optional machine-readable output path; a bare filename is written "
            "under benchmarks/results"
        ),
    )

    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--representation", choices=REPRESENTATIONS, help=argparse.SUPPRESS)

    args = parser.parse_args()
    if args.modes <= 0:
        parser.error("--modes must be positive")
    if args.particles < 0:
        parser.error("--particles must be non-negative")
    if args.statistics == "fermion" and args.particles > args.modes:
        parser.error("fermionic --particles must not exceed --modes")
    if args.k <= 0:
        parser.error("--k must be positive")
    if args.ncv is not None and args.ncv <= args.k:
        parser.error("--ncv must be greater than --k")
    if args.repeat <= 0:
        parser.error("--repeat must be positive")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.worker and args.representation is None:
        parser.error("worker mode requires --representation")
    args.json = _resolve_json_path(args.json)
    return args


def main():
    args = parse_args()
    if args.worker:
        _worker(args)
        return

    metadata = _environment_metadata()
    if args.statistics == "fermion":
        expected_dimension = comb(args.modes, args.particles)
    else:
        expected_dimension = comb(args.particles + args.modes - 1, args.particles)

    print(
        f"EDinPy {metadata['edinpy']} | Python {metadata['python']} | "
        f"NumPy {metadata['numpy']} | SciPy {metadata['scipy']}"
    )
    print(
        f"statistics={args.statistics}, modes={args.modes}, "
        f"particles={args.particles}, dimension={expected_dimension}, "
        f"k={args.k}, ncv={args.ncv}"
    )
    print()

    results = [
        _run_representation(args, representation)
        for representation in args.representations
    ]
    _print_results(results)

    if args.json is not None:
        payload = {
            "environment": metadata,
            "parameters": {
                "statistics": args.statistics,
                "modes": args.modes,
                "particles": args.particles,
                "k": args.k,
                "which": args.which,
                "tol": args.tol,
                "ncv": args.ncv,
                "maxiter": args.maxiter,
                "repeat": args.repeat,
                "warmup": args.warmup,
                "seed": args.seed,
                "no_eigsolve": args.no_eigsolve,
            },
            "results": results,
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"\nWrote {args.json}")


if __name__ == "__main__":
    main()
