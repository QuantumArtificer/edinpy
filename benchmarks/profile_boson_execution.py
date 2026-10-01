"""Benchmark public boson matrix-free execution modes.

The benchmark covers representative diagonal, hopping, pair-transfer, quartic,
and complex-valued Hamiltonians on complete or population-projected fixed-N
boson sectors. Each operator family runs in a fresh process; Numba parallel
rows may be measured at several thread counts after one JIT warm-up.
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
FAMILIES = ("diagonal", "hopping", "pair", "quartic", "complex-hopping", "mixed")
FAMILY_DESCRIPTIONS = {
    "diagonal": "onsite, Hubbard, and density-density interactions",
    "hopping": "real one-boson transfer over short and long separations",
    "pair": "Hermitian two-boson transfer",
    "quartic": "general number-conserving quartic monomials",
    "complex-hopping": "Hermitian hopping with complex phases",
    "mixed": "hopping with onsite and density interactions",
}


def _peak_rss_mib() -> float:
    """Return process peak resident memory in MiB."""
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() == "Darwin":
        return value / (1024.0 * 1024.0)
    return value / 1024.0


def _environment_metadata() -> dict[str, object]:
    """Return software versions relevant to execution timing."""
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


def _mode_pairs(modes, n_modes: int, projected: bool) -> list[tuple[int, int]]:
    """Return representative transfer pairs that preserve any projection."""
    if projected:
        pairs = []
        sites = n_modes // 2
        for species in (0, 1):
            chain = [modes.resolve((site, species)) for site in range(sites)]
            pairs.extend(zip(chain[:-1], chain[1:]))
            if sites > 2:
                pairs.extend(zip(chain[:-2], chain[2:]))
        return list(pairs)

    separations = (1, 2, max(3, n_modes // 3))
    pairs = []
    seen = set()
    for separation in separations:
        if separation >= n_modes:
            continue
        for low in range(n_modes - separation):
            pair = (low, low + separation)
            if pair not in seen:
                seen.add(pair)
                pairs.append(pair)
    return pairs


def _build_expression(family: str, modes, n_modes: int, projected: bool, seed: int):
    """Build one deterministic representative bosonic operator family."""
    from edinpy import boson as edb

    rng = np.random.default_rng(seed)
    pairs = _mode_pairs(modes, n_modes, projected)

    if family == "diagonal":
        terms = [
            edb.Onsite(mode, float(rng.uniform(-0.5, 0.5)), modes)
            + edb.Hubbard(mode, 0.15 + 0.01 * (mode % 5), modes)
            for mode in range(n_modes)
        ]
        terms.extend(
            edb.DensityDensity(left, right, 0.08 + 0.01 * (index % 5), modes)
            for index, (left, right) in enumerate(pairs)
        )
        return _sum_terms(terms), len(terms)

    if family == "hopping":
        terms = [
            edb.Hopping(left, right, -0.35 - 0.01 * (index % 5), modes)
            for index, (left, right) in enumerate(pairs)
        ]
        return _sum_terms(terms), len(terms)

    if family == "pair":
        terms = [
            edb.PairHopping(left, right, 0.08 + 0.01 * (index % 3), modes)
            for index, (left, right) in enumerate(pairs)
        ]
        return _sum_terms(terms), len(terms)

    if family == "quartic":
        b = edb.set_notation(edb.Annihilation, modes)
        bd = edb.set_notation(edb.Creation, modes)
        terms = []
        source_pairs = pairs[: max(1, min(len(pairs), n_modes))]
        for index, (left, right) in enumerate(source_pairs):
            if projected:
                species = modes.unravel(left)[1]
                sites = n_modes // 2
                same_species = [modes.resolve((site, species)) for site in range(sites)]
                a = same_species[index % len(same_species)]
                c = same_species[(index + 2) % len(same_species)]
            else:
                a = (left + 2) % n_modes
                c = (right + 3) % n_modes
            coefficient = 0.04 + 0.005 * (index % 4)
            term = (
                coefficient
                * bd(*modes.unravel(a))
                * bd(*modes.unravel(left))
                * b(*modes.unravel(right))
                * b(*modes.unravel(c))
            )
            terms.extend((term, term.dag))
        return _sum_terms(terms), len(terms)

    if family == "complex-hopping":
        terms = []
        for index, (left, right) in enumerate(pairs):
            coefficient = -0.25 * np.exp(1j * 0.17 * (index + 1))
            terms.append(edb.Hopping(left, right, coefficient, modes))
        return _sum_terms(terms), len(terms)

    if family == "mixed":
        terms = [
            edb.Onsite(mode, 0.03 * mode, modes) + edb.Hubbard(mode, 0.2, modes)
            for mode in range(n_modes)
        ]
        for left, right in pairs:
            terms.append(edb.Hopping(left, right, -0.3, modes))
            terms.append(edb.DensityDensity(left, right, 0.12, modes))
        return _sum_terms(terms), len(terms)

    raise ValueError(f"Unknown boson benchmark family {family!r}.")


def _build_problem(args, family: str):
    """Construct one complete or projected bosonic benchmark problem."""
    from edinpy import boson as edb

    if args.projected:
        sites = args.modes // 2
        site = edb.DoF(sites, name="site")
        species = edb.DoF(2, name="species", labels=("a", "b"))
        modes = edb.BosonModes(site, species)
        species_a = args.species_a
        species_b = args.particles - species_a
        sector = (
            edb.NParticleSector(modes, N=args.particles)
            .project_particles("species", a=species_a, b=species_b)
            .build()
        )
    else:
        modes = edb.BosonModes(edb.DoF(args.modes, name="mode"))
        sector = edb.NParticleSector(modes, N=args.particles).build()

    expression, source_terms = _build_expression(
        family, modes, args.modes, args.projected, args.seed
    )
    hamiltonian = edb.Hamiltonian(expression, sector)
    return hamiltonian, {
        "family": family,
        "description": FAMILY_DESCRIPTIONS[family],
        "source_terms": source_terms,
        "compiled_kernels": hamiltonian.compiler_stats,
        "basis": "projected" if args.projected else "complete",
    }


def _vector(dimension: int, seed: int, *, complex_valued: bool) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vector = rng.normal(size=dimension)
    if complex_valued:
        vector = vector + 1j * rng.normal(size=dimension)
    vector /= np.linalg.norm(vector)
    return vector


def _measure(callback, repeat: int):
    start = time.perf_counter()
    result = callback()
    warmup = time.perf_counter() - start
    samples = []
    for _ in range(repeat):
        start = time.perf_counter()
        result = callback()
        samples.append(time.perf_counter() - start)
    return result, warmup, samples


def _result_row(problem, execution, threads, result, warmup, samples, dimension, reference):
    median = statistics.median(samples)
    return {
        **problem,
        "dimension": dimension,
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
    hamiltonian, problem = _build_problem(args, args._worker_family)
    dimension = int(hamiltonian.sector.dimension)
    vector = _vector(
        dimension,
        args.seed,
        complex_valued=np.dtype(hamiltonian._compiled.data_dtype).kind == "c",
    )
    reference = hamiltonian.aslinearoperator(execution="numpy") @ vector

    try:
        import numba
    except ImportError:
        numba = None

    rows = []
    for execution in args.executions:
        if execution.startswith("numba-") and numba is None:
            rows.append(
                {
                    **problem,
                    "execution": execution,
                    "status": "unsupported",
                    "reason": "Numba is not installed",
                    "worker_pid": os.getpid(),
                }
            )
            continue
        operator = hamiltonian.aslinearoperator(execution=execution)
        if execution != "numba-parallel":
            result, warmup, samples = _measure(lambda: operator @ vector, args.repeat)
            rows.append(
                _result_row(
                    problem, execution, 1, result, warmup, samples, dimension, reference
                )
            )
            continue

        maximum = numba.get_num_threads()
        thread_counts = [count for count in args.threads if count <= maximum] or [maximum]
        for count in thread_counts:
            numba.set_num_threads(count)
            result, warmup, samples = _measure(lambda: operator @ vector, args.repeat)
            rows.append(
                _result_row(
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
    if args.projected:
        command.extend(("--projected", "--species-a", str(args.species_a)))
    return command


def _run_subprocess(args, family: str):
    completed = subprocess.run(
        _worker_command(args, family), check=False, capture_output=True, text=True
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"boson execution worker failed for {family}:\n{completed.stderr.strip()}"
        )
    return json.loads(completed.stdout)


def _summary(rows):
    summary = {}
    for family in FAMILIES:
        family_rows = [row for row in rows if row.get("family") == family]
        serial = next(
            (
                row
                for row in family_rows
                if row.get("execution") == "numba-serial"
                and "matvec_seconds_median" in row
            ),
            None,
        )
        parallel = [
            row
            for row in family_rows
            if row.get("execution") == "numba-parallel"
            and "matvec_seconds_median" in row
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
    speedups = [item["speedup_vs_serial"] for item in summary.values()]
    geometric_mean = (
        math.exp(sum(math.log(value) for value in speedups) / len(speedups))
        if speedups
        else None
    )
    return {
        "by_family": summary,
        "geometric_mean_best_parallel_speedup_vs_serial": geometric_mean,
    }


def _resolve_json_path(path: Path | None) -> Path | None:
    if path is None or path.is_absolute() or path.parent != Path("."):
        return path
    return RESULTS_DIR / path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modes", type=int, default=14)
    parser.add_argument("--particles", type=int, default=7)
    parser.add_argument("--projected", action="store_true")
    parser.add_argument(
        "--species-a",
        type=int,
        default=None,
        help="species-a population for --projected; defaults to floor(N/2)",
    )
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--families", nargs="+", choices=FAMILIES, default=list(FAMILIES))
    parser.add_argument(
        "--executions",
        nargs="+",
        choices=EXECUTIONS,
        default=list(EXECUTIONS),
    )
    parser.add_argument("--threads", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--_worker-family", choices=FAMILIES, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.modes <= 0:
        parser.error("--modes must be positive")
    if args.particles < 0:
        parser.error("--particles must be non-negative")
    if args.projected and (args.modes < 2 or args.modes % 2):
        parser.error("--projected requires an even --modes value >= 2")
    if args.species_a is None:
        args.species_a = args.particles // 2
    if not 0 <= args.species_a <= args.particles:
        parser.error("--species-a must lie between 0 and --particles")
    if args.repeat <= 0:
        parser.error("--repeat must be positive")
    if any(value <= 0 for value in args.threads):
        parser.error("--threads values must be positive")
    return args


def main() -> None:
    args = parse_args()
    if args._worker_family is not None:
        print(json.dumps(_run_worker(args)))
        return

    environment = _environment_metadata()
    if args.projected:
        dimension_label = "projected"
    else:
        dimension_label = f"{comb(args.particles + args.modes - 1, args.particles):,}"
    print(
        f"EDinPy {environment['edinpy']} | Python {environment['python']} | "
        f"NumPy {environment['numpy']} | SciPy {environment['scipy']}"
    )
    print(
        f"boson modes={args.modes}, particles={args.particles}, "
        f"basis={'projected' if args.projected else 'complete'}, dimension={dimension_label}"
    )

    rows = []
    for family in args.families:
        family_rows = _run_subprocess(args, family)
        rows.extend(family_rows)
        for row in family_rows:
            if "matvec_seconds_median" not in row:
                print(f"{family:>18} {row['execution']:>14} unsupported")
                continue
            thread_label = (
                f"/{row['threads']}t" if row["execution"] == "numba-parallel" else ""
            )
            print(
                f"{family:>18} {(row['execution'] + thread_label):>18} "
                f"{row['matvec_seconds_median']:.6f} s "
                f"delta={row['max_abs_delta_vs_numpy']:.2e}"
            )

    dimension = next(
        (row.get("dimension") for row in rows if row.get("dimension") is not None),
        None,
    )
    payload = {
        "environment": environment,
        "parameters": {
            "modes": args.modes,
            "particles": args.particles,
            "projected": args.projected,
            "species_a": args.species_a if args.projected else None,
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
