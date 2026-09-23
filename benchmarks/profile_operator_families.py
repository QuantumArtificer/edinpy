"""Benchmark sparse Hamiltonian construction for representative fermion operators."""

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

from edinpy import fermion as edf

FAMILIES = ("hopping", "density", "quartic", "mixed")


def _sum_terms(terms):
    """Return a symbolic sum without requiring a nonempty iterable."""
    expression = 0
    for term in terms:
        expression += term
    return expression


def build_expression(length, family, modes):
    """Build one representative symbolic operator expression."""
    c = edf.set_notation(edf.Annihilation, modes)
    cd = edf.set_notation(edf.Creation, modes)
    n = edf.set_notation(edf.Number, modes)

    if family == "hopping":
        terms = []
        for i in range(length - 1):
            terms.extend((-cd(i) * c(i + 1), -cd(i + 1) * c(i)))
    elif family == "density":
        terms = [n(i) * n(i + 1) for i in range(length - 1)]
    elif family == "quartic":
        terms = []
        for i in range(0, length - 3, 2):
            transfer = cd(i) * cd(i + 1) * c(i + 3) * c(i + 2)
            terms.extend((transfer, transfer.dag))
    elif family == "mixed":
        terms = []
        for i in range(length - 1):
            terms.extend(
                (
                    -cd(i) * c(i + 1),
                    -cd(i + 1) * c(i),
                    0.5 * n(i) * n(i + 1),
                )
            )
    else:
        raise ValueError(f"Unknown operator family: {family}")

    return _sum_terms(terms), len(terms)


def benchmark_family(length, particles, family, repeat, warmup):
    """Return timing and sparsity data for one operator family."""
    modes = edf.FermionModes(edf.DoF(length, name="mode"))

    start = time.perf_counter()
    sector = edf.NParticleSector(modes, N=particles).build()
    basis_seconds = time.perf_counter() - start

    start = time.perf_counter()
    expression, term_count = build_expression(length, family, modes)
    expression_seconds = time.perf_counter() - start

    start = time.perf_counter()
    hamiltonian = edf.Hamiltonian(expression, sector)
    compile_seconds = time.perf_counter() - start

    for _ in range(warmup):
        hamiltonian.calc_matrix()

    samples = []
    matrix = None
    for _ in range(repeat):
        start = time.perf_counter()
        matrix = hamiltonian.calc_matrix()
        samples.append(time.perf_counter() - start)

    return {
        "family": family,
        "terms": term_count,
        "dimension": sector.dimension,
        "nnz": matrix.nnz,
        "basis_seconds": basis_seconds,
        "expression_seconds": expression_seconds,
        "compile_seconds": compile_seconds,
        "matrix_seconds_median": statistics.median(samples),
        "matrix_seconds_min": min(samples),
        "matrix_seconds_max": max(samples),
        "matrix_seconds_samples": samples,
    }


def environment_metadata():
    """Return versions needed to interpret benchmark results."""
    return {
        "python": sys.version.split()[0],
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "edinpy": getattr(__import__("edinpy"), "__version__", "unknown"),
    }


def parse_args():
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark EDinPy sparse matrix construction. Compare timings only "
            "between runs with the same model, software environment, and hardware."
        )
    )
    parser.add_argument("--length", type=int, default=16, help="number of modes")
    parser.add_argument(
        "--particles",
        type=int,
        default=8,
        help="particle number in the fixed-N sector",
    )
    parser.add_argument(
        "--families",
        nargs="+",
        default=FAMILIES,
        choices=FAMILIES,
        help="operator families to benchmark",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=5,
        help="timed matrix constructions per family",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="untimed matrix constructions per family",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="optional path for machine-readable results",
    )
    args = parser.parse_args()

    if args.length <= 0:
        parser.error("--length must be positive")
    if not 0 <= args.particles <= args.length:
        parser.error("--particles must satisfy 0 <= N <= length")
    if args.repeat <= 0:
        parser.error("--repeat must be positive")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")

    return args


def main():
    """Run the selected benchmark families and print a compact summary."""
    args = parse_args()
    metadata = environment_metadata()
    expected_dimension = comb(args.length, args.particles)

    print(
        f"EDinPy {metadata['edinpy']} | Python {metadata['python']} | "
        f"NumPy {metadata['numpy']} | SciPy {metadata['scipy']}"
    )
    print(
        f"L={args.length}, N={args.particles}, dimension={expected_dimension}, "
        f"repeat={args.repeat}, warmup={args.warmup}"
    )
    print()
    print(
        f"{'family':<9} {'terms':>7} {'nnz':>10} "
        f"{'basis [s]':>11} {'compile [s]':>13} {'matrix median [s]':>18}"
    )
    print("-" * 74)

    results = []
    for family in args.families:
        result = benchmark_family(
            args.length,
            args.particles,
            family,
            args.repeat,
            args.warmup,
        )
        results.append(result)
        print(
            f"{family:<9} {result['terms']:>7d} {result['nnz']:>10d} "
            f"{result['basis_seconds']:>11.6f} "
            f"{result['compile_seconds']:>13.6f} "
            f"{result['matrix_seconds_median']:>18.6f}"
        )

    if args.json is not None:
        payload = {
            "environment": metadata,
            "parameters": {
                "length": args.length,
                "particles": args.particles,
                "repeat": args.repeat,
                "warmup": args.warmup,
            },
            "results": results,
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"\nWrote {args.json}")


if __name__ == "__main__":
    main()
