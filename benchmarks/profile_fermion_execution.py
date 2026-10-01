"""Benchmark public fermion matrix-free execution modes.

The benchmark spans representative diagonal, hopping, interaction, quartic,
and complex-valued Hamiltonians. Each operator family runs in a fresh Python
process. Numba parallel rows may be measured at several thread counts without
restarting the family worker, so JIT warm-up is kept separate from steady-state
timing.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
from math import comb
from pathlib import Path

import numpy as np
import scipy


RESULTS_DIR = Path(__file__).resolve().parent / "results"
EXECUTIONS = ("numpy", "numba-serial", "numba-parallel")
FAMILIES = (
    "diagonal",
    "hopping",
    "dense-one-body",
    "extended-hubbard",
    "exchange-pair",
    "quartic",
    "complex-hopping",
)
FAMILY_DESCRIPTIONS = {
    "diagonal": "onsite and density-density terms",
    "hopping": "local and long-range real hopping",
    "dense-one-body": "all-to-all real Hermitian hopping",
    "extended-hubbard": "spinful hopping with onsite and intersite density terms",
    "exchange-pair": "exchange and pair-hopping interactions",
    "quartic": "general number-conserving quartic monomials",
    "complex-hopping": "Hermitian hopping with complex phases",
}


def _peak_rss_mib() -> float:
    """Return process peak resident memory in MiB."""
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() == "Darwin":
        return value / (1024.0 * 1024.0)
    return value / 1024.0


def _environment_metadata() -> dict[str, object]:
    """Return versions and platform details relevant to execution timing."""
    import edinpy

    try:
        import numba
    except ImportError:
        numba_version = None
        numba_threads = None
    else:
        numba_version = numba.__version__
        numba_threads = numba.get_num_threads()

    return {
        "python": sys.version.split()[0],
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "numba": numba_version,
        "numba_threads": numba_threads,
        "edinpy": getattr(edinpy, "__version__", "unknown"),
    }


def _sum_terms(terms):
    expression = 0
    for term in terms:
        expression += term
    return expression


def _separated_pairs(n_modes: int) -> list[tuple[int, int]]:
    """Return unique mode pairs spanning short and long separations."""
    separations = (1, 2, max(3, n_modes // 3))
    pairs: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for separation in separations:
        if separation >= n_modes:
            continue
        for low in range(n_modes - separation):
            pair = (low, low + separation)
            if pair not in seen:
                seen.add(pair)
                pairs.append(pair)
    return pairs


def _build_expression(family: str, modes, n_modes: int, seed: int):
    """Build one representative operator family and return its source-term count."""
    from edinpy import fermion as edf

    rng = np.random.default_rng(seed)

    if family == "diagonal":
        terms = [
            edf.Onsite(index, float(rng.uniform(-0.5, 0.5)), modes)
            for index in range(n_modes)
        ]
        for index, (left, right) in enumerate(_separated_pairs(n_modes)):
            coefficient = 0.1 + 0.01 * (index % 7)
            terms.append(edf.DensityDensity(left, right, coefficient, modes))
        return _sum_terms(terms), len(terms)

    if family == "hopping":
        pairs = _separated_pairs(n_modes)
        terms = [
            edf.Hopping(left, right, -0.35 - 0.01 * (index % 5), modes)
            for index, (left, right) in enumerate(pairs)
        ]
        return _sum_terms(terms), len(terms)

    if family == "dense-one-body":
        terms = []
        for left in range(n_modes):
            for right in range(left + 1, n_modes):
                terms.append(
                    edf.Hopping(
                        left,
                        right,
                        float(rng.uniform(-0.08, 0.08)),
                        modes,
                    )
                )
        return _sum_terms(terms), len(terms)

    if family == "extended-hubbard":
        if n_modes < 4 or n_modes % 2:
            raise ValueError("extended-hubbard requires an even mode count >= 4")
        n_sites = n_modes // 2
        terms = []
        for site in range(n_sites - 1):
            for spin in (0, 1):
                left = 2 * site + spin
                right = 2 * (site + 1) + spin
                terms.append(edf.Hopping(left, right, -0.4, modes))
            up = 2 * site
            down = up + 1
            terms.append(edf.DensityDensity(up, down, 0.8, modes))
            terms.append(edf.DensityDensity(up, up + 2, 0.18, modes))
            terms.append(edf.DensityDensity(down, down + 2, 0.18, modes))
        terms.append(edf.DensityDensity(n_modes - 2, n_modes - 1, 0.8, modes))
        return _sum_terms(terms), len(terms)

    if family == "exchange-pair":
        terms = []
        for start in range(0, n_modes - 3, 4):
            a, b, c, d = start, start + 1, start + 2, start + 3
            terms.append(edf.HeisenbergExchange(a, b, c, d, 0.16, modes))
            terms.append(edf.PairHopping(a, b, c, d, 0.11, modes))
        if not terms:
            raise ValueError("exchange-pair requires at least four modes")
        return _sum_terms(terms), len(terms)

    if family == "quartic":
        c = edf.set_notation(edf.Annihilation, modes)
        cd = edf.set_notation(edf.Creation, modes)
        terms = []
        for index in range(max(1, n_modes)):
            positions = rng.choice(n_modes, size=4, replace=False)
            a, b, c_index, d = (int(value) for value in positions)
            coefficient = float(rng.uniform(-0.08, 0.08))
            term = coefficient * cd(a) * cd(b) * c(d) * c(c_index)
            terms.extend((term, term.dag))
        return _sum_terms(terms), len(terms)

    if family == "complex-hopping":
        terms = []
        for index, (left, right) in enumerate(_separated_pairs(n_modes)):
            phase = 0.17 * (index + 1)
            coefficient = -0.25 * np.exp(1j * phase)
            terms.append(edf.Hopping(left, right, coefficient, modes))
        return _sum_terms(terms), len(terms)

    raise ValueError(f"Unknown fermion benchmark family {family!r}.")


def _build_problem(family: str, n_modes: int, particles: int, seed: int):
    """Construct a Hamiltonian and compact family metadata."""
    from edinpy import fermion as edf

    modes = edf.FermionModes(edf.DoF(n_modes, name="mode"))
    sector = edf.NParticleSector(modes, N=particles).build()
    expression, source_terms = _build_expression(family, modes, n_modes, seed)
    ham = edf.Hamiltonian(expression, sector)
    return ham, {
        "family": family,
        "description": FAMILY_DESCRIPTIONS[family],
        "source_terms": source_terms,
        "compiled_kernels": ham._compiled.stats,
    }


def _vector(dimension: int, seed: int, *, complex_valued: bool) -> np.ndarray:
    """Return a normalized deterministic benchmark vector."""
    rng = np.random.default_rng(seed)
    vector = rng.normal(size=dimension)
    if complex_valued:
        vector = vector + 1j * rng.normal(size=dimension)
    vector /= np.linalg.norm(vector)
    return vector


def _measure(callback, repeat: int):
    """Warm one callable and return its final result and steady-state timings."""
    start = time.perf_counter()
    result = callback()
    warmup = time.perf_counter() - start

    samples = []
    for _ in range(repeat):
        start = time.perf_counter()
        result = callback()
        samples.append(time.perf_counter() - start)
    return result, warmup, samples


def _row(problem, execution, threads, result, warmup, samples, dimension, reference):
    """Create one JSON-serializable timing row."""
    median = statistics.median(samples)
    return {
        **problem,
        "execution": execution,
        "threads": threads,
        "worker_pid": os.getpid(),
        "warmup_seconds": warmup,
        "matvec_seconds_median": median,
        "matvec_seconds_min": min(samples),
        "matvec_seconds_max": max(samples),
        "states_per_second": float(dimension / median),
        "nanoseconds_per_state": float(1e9 * median / dimension),
        "peak_rss_mib": _peak_rss_mib(),
        "matrix_materialized": False,
        "result_norm": float(np.linalg.norm(result)),
        "max_abs_delta_vs_numpy": float(np.max(np.abs(result - reference))),
    }


def _run_worker(args):
    ham, problem = _build_problem(
        args._worker_family,
        args.modes,
        args.particles,
        args.seed,
    )
    dimension = int(ham.sector.dimension)
    vector = _vector(
        dimension,
        args.seed,
        complex_valued=np.dtype(ham._compiled.data_dtype).kind == "c",
    )
    reference = ham.aslinearoperator(execution="numpy") @ vector

    numba = None
    if any(value.startswith("numba-") for value in args.executions):
        try:
            import numba as _numba
        except ImportError:
            numba = None
        else:
            numba = _numba

    rows = []
    for execution in args.executions:
        if execution.startswith("numba-") and numba is None:
            rows.append({
                **problem,
                "execution": execution,
                "status": "unsupported",
                "reason": "Numba is not installed",
                "worker_pid": os.getpid(),
            })
            continue

        operator = ham.aslinearoperator(execution=execution)
        if execution != "numba-parallel":
            result, warmup, samples = _measure(lambda: operator @ vector, args.repeat)
            rows.append(
                _row(problem, execution, 1, result, warmup, samples, dimension, reference)
            )
            continue

        max_threads = numba.get_num_threads()
        thread_counts = [value for value in args.threads if value <= max_threads]
        if not thread_counts:
            thread_counts = [max_threads]
        for thread_count in thread_counts:
            numba.set_num_threads(thread_count)
            result, warmup, samples = _measure(lambda: operator @ vector, args.repeat)
            rows.append(
                _row(
                    problem,
                    execution,
                    numba.get_num_threads(),
                    result,
                    warmup,
                    samples,
                    dimension,
                    reference,
                )
            )
    return rows


def _worker_command(args, family: str) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--modes",
        str(args.modes),
        "--particles",
        str(args.particles),
        "--repeat",
        str(args.repeat),
        "--seed",
        str(args.seed),
        "--executions",
        *args.executions,
        "--threads",
        *(str(value) for value in args.threads),
        "--_worker-family",
        family,
    ]
    return command


def _run_subprocess(args, family: str):
    completed = subprocess.run(
        _worker_command(args, family),
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"fermion execution worker failed for {family}:\n{completed.stderr.strip()}"
        )
    return json.loads(completed.stdout)


def _resolve_json_path(path: Path | None) -> Path | None:
    if path is None or path.is_absolute() or path.parent != Path("."):
        return path
    return RESULTS_DIR / path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modes", type=int, default=20)
    parser.add_argument("--particles", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--families", nargs="+", choices=FAMILIES, default=list(FAMILIES))
    parser.add_argument(
        "--executions",
        nargs="+",
        choices=EXECUTIONS,
        default=list(EXECUTIONS),
        help="public matrix-free execution modes to measure",
    )
    parser.add_argument(
        "--threads",
        nargs="+",
        type=int,
        default=[1, 2, 4, 8],
        help="Numba thread counts used for numba-parallel rows",
    )
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--_worker-family", choices=FAMILIES, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if not 1 <= args.modes <= 64:
        parser.error("--modes must satisfy 1 <= modes <= 64")
    if not 0 <= args.particles <= args.modes:
        parser.error("--particles must satisfy 0 <= particles <= modes")
    if args.repeat <= 0:
        parser.error("--repeat must be positive")
    if any(value <= 0 for value in args.threads):
        parser.error("--threads values must be positive")
    return args


def _summary(rows):
    """Summarize best Numba speedups relative to serial execution per family."""
    summary = {}
    for family in FAMILIES:
        family_rows = [row for row in rows if row.get("family") == family]
        serial = next(
            (
                row
                for row in family_rows
                if row.get("execution") == "numba-serial" and "matvec_seconds_median" in row
            ),
            None,
        )
        parallel = [
            row
            for row in family_rows
            if row.get("execution") == "numba-parallel" and "matvec_seconds_median" in row
        ]
        if serial is None or not parallel:
            continue
        best = min(parallel, key=lambda row: row["matvec_seconds_median"])
        summary[family] = {
            "serial_seconds": serial["matvec_seconds_median"],
            "best_parallel_threads": best["threads"],
            "best_parallel_seconds": best["matvec_seconds_median"],
            "speedup_vs_serial": serial["matvec_seconds_median"]
            / best["matvec_seconds_median"],
        }
    if summary:
        speedups = [item["speedup_vs_serial"] for item in summary.values()]
        geometric_mean = math.exp(sum(math.log(value) for value in speedups) / len(speedups))
    else:
        geometric_mean = None
    return {
        "by_family": summary,
        "geometric_mean_best_parallel_speedup_vs_serial": geometric_mean,
    }


def main() -> None:
    args = parse_args()
    if args._worker_family is not None:
        print(json.dumps(_run_worker(args)))
        return

    environment = _environment_metadata()
    dimension = comb(args.modes, args.particles)
    print(
        f"EDinPy {environment['edinpy']} | Python {environment['python']} | "
        f"NumPy {environment['numpy']} | SciPy {environment['scipy']}"
    )
    print(
        f"fermion modes={args.modes}, particles={args.particles}, "
        f"dimension={dimension:,}"
    )

    rows = []
    for family in args.families:
        family_rows = _run_subprocess(args, family)
        rows.extend(family_rows)
        for row in family_rows:
            if "matvec_seconds_median" not in row:
                print(f"{family:>18} {row['execution']:>14} unsupported")
                continue
            thread_label = f"/{row['threads']}t" if row["execution"] == "numba-parallel" else ""
            print(
                f"{family:>18} {(row['execution'] + thread_label):>18} "
                f"{row['matvec_seconds_median']:.6f} s "
                f"delta={row['max_abs_delta_vs_numpy']:.2e}"
            )

    payload = {
        "environment": environment,
        "parameters": {
            "modes": args.modes,
            "particles": args.particles,
            "dimension": dimension,
            "families": args.families,
            "executions": args.executions,
            "threads": args.threads,
            "repeat": args.repeat,
            "seed": args.seed,
            "process_isolation": "per-family",
        },
        "family_descriptions": FAMILY_DESCRIPTIONS,
        "results": rows,
        "summary": _summary(rows),
    }

    output = _resolve_json_path(args.json)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {output}")


if __name__ == "__main__":
    main()
