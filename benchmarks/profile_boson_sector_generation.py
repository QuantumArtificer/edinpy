"""Benchmark direct bosonic fixed-N and projected-sector basis construction."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from math import comb
from pathlib import Path

import numpy as np
import scipy

from edinpy import boson as edb

CASES = ("unconstrained", "one_dof", "two_dof", "three_dof")
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def _particle_split(total):
    """Return a deterministic two-value population split summing to ``total``."""
    first = total // 2
    return first, total - first


def _make_modes(sites):
    """Return one common mode space used by every benchmark case."""
    return edb.BosonModes(
        edb.DoF(sites, name="site"),
        edb.DoF(2, name="species", labels=("a", "b")),
        edb.DoF(2, name="layer", labels=("top", "bottom")),
        edb.DoF(2, name="orbital", labels=("x", "y")),
    )


def _make_sector(modes, particles, case):
    """Return an unbuilt sector specification for one projection case."""
    first, second = _particle_split(particles)
    sector = edb.NParticleSector(modes, N=particles)

    if case == "unconstrained":
        return sector
    sector.project_particles("species", a=first, b=second)
    if case == "one_dof":
        return sector
    sector.project_particles("layer", top=first, bottom=second)
    if case == "two_dof":
        return sector
    if case == "three_dof":
        return sector.project_particles("orbital", x=first, y=second)
    raise ValueError(f"Unknown sector benchmark case: {case}")


def benchmark_case(modes, particles, case, repeat, warmup):
    """Measure direct basis construction for one sector specification."""
    for _ in range(warmup):
        _make_sector(modes, particles, case).build()

    samples = []
    dimension = None
    for _ in range(repeat):
        sector = _make_sector(modes, particles, case)
        start = time.perf_counter()
        sector.build()
        samples.append(time.perf_counter() - start)
        dimension = sector.dimension

    median = statistics.median(samples)
    return {
        "case": case,
        "dimension": dimension,
        "build_seconds_median": median,
        "build_seconds_min": min(samples),
        "build_seconds_max": max(samples),
        "build_seconds_samples": samples,
        "states_per_second_median": dimension / median if median else None,
    }


def environment_metadata():
    """Return software versions needed to interpret benchmark results."""
    return {
        "python": sys.version.split()[0],
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "edinpy": getattr(__import__("edinpy"), "__version__", "unknown"),
    }


def _resolve_json_path(path):
    """Route bare result filenames into ``benchmarks/results``."""
    if path is None or path.is_absolute() or path.parent != Path("."):
        return path
    return RESULTS_DIR / path


def parse_args():
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark EDinPy bosonic sector generation. All cases use the same "
            "mode space; projected cases add one, two, or three conserved "
            "binary population marginals before direct basis generation."
        )
    )
    parser.add_argument(
        "--sites",
        type=int,
        default=2,
        help="number of site values; total mode count is 8 * sites",
    )
    parser.add_argument(
        "--particles",
        type=int,
        default=8,
        help="total boson number in the fixed-N sector",
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        default=CASES,
        choices=CASES,
        help="sector-generation cases to benchmark",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=5,
        help="timed basis constructions per case",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="untimed basis constructions per case",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help=(
            "optional machine-readable output path; a bare filename is written "
            "under benchmarks/results"
        ),
    )
    args = parser.parse_args()

    if args.sites <= 0:
        parser.error("--sites must be positive")
    if args.particles < 0:
        parser.error("--particles must be non-negative")
    if args.repeat <= 0:
        parser.error("--repeat must be positive")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")

    args.json = _resolve_json_path(args.json)
    return args


def main():
    """Run selected sector-generation benchmarks and print a compact summary."""
    args = parse_args()
    metadata = environment_metadata()
    modes = _make_modes(args.sites)
    full_dimension = comb(args.particles + modes.n_modes - 1, args.particles)

    print(
        f"EDinPy {metadata['edinpy']} | Python {metadata['python']} | "
        f"NumPy {metadata['numpy']} | SciPy {metadata['scipy']}"
    )
    print(
        f"sites={args.sites}, modes={modes.n_modes}, N={args.particles}, "
        f"full dimension={full_dimension}, repeat={args.repeat}, "
        f"warmup={args.warmup}"
    )
    print()
    print(
        f"{'case':<14} {'dimension':>12} {'fraction':>11} "
        f"{'median [s]':>13} {'states/s':>14}"
    )
    print("-" * 70)

    results = []
    for case in args.cases:
        result = benchmark_case(
            modes,
            args.particles,
            case,
            args.repeat,
            args.warmup,
        )
        result["fraction_of_full_basis"] = result["dimension"] / full_dimension
        results.append(result)
        print(
            f"{case:<14} {result['dimension']:>12d} "
            f"{result['fraction_of_full_basis']:>11.4f} "
            f"{result['build_seconds_median']:>13.6f} "
            f"{result['states_per_second_median']:>14.0f}"
        )

    if args.json is not None:
        payload = {
            "environment": metadata,
            "parameters": {
                "sites": args.sites,
                "n_modes": modes.n_modes,
                "particles": args.particles,
                "repeat": args.repeat,
                "warmup": args.warmup,
                "full_dimension": full_dimension,
            },
            "results": results,
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"\nWrote {args.json}")


if __name__ == "__main__":
    main()
