"""Profile the corrected pre-optimization EDinPy Hamiltonian construction."""

from __future__ import annotations

import argparse
import cProfile
import io
import os
import pstats
import time
from contextlib import redirect_stdout
from math import comb
from pathlib import Path

from edinpy import fermion as edf


def add_term(expression, term):
    return term if expression is None else expression + term


def build_spinless_chain(length: int, nfermions: int, t: float = 1.0, v: float = 1.0):
    """Construct an open spinless t-V chain."""
    edf.clear()
    edf.DoF(length, name="site")

    t0 = time.perf_counter()
    model = edf.model(nfermions)
    basis_time = time.perf_counter() - t0

    expression = None

    for i in range(length - 1):
        j = i + 1

        ci = edf.operator(i)
        cj = edf.operator(j)

        hopping = -t * (
            ci.dag * cj
            + cj.dag * ci
        )

        interaction = v * (
            edf.number(i) * edf.number(j)
        )

        expression = add_term(expression, hopping)
        expression = add_term(expression, interaction)

    return model, edf.hamiltonian(expression), basis_time


def calculate_silently(hamiltonian):
    """Run the legacy matrix builder without flooding the terminal."""
    with open(os.devnull, "w") as devnull, redirect_stdout(devnull):
        hamiltonian.calc_matrix()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--length", type=int, default=10)
    parser.add_argument("--particles", type=int, default=5)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/baseline_profile.prof"),
    )
    args = parser.parse_args()

    expected_basis = comb(args.length, args.particles)
    upper_candidates = expected_basis * (expected_basis + 1) // 2

    model, hamiltonian, basis_time = build_spinless_chain(
        args.length,
        args.particles,
    )

    assert model.Nbasis == expected_basis

    print(f"L                  : {args.length}")
    print(f"N                  : {args.particles}")
    print(f"Hilbert dimension  : {model.Nbasis}")
    print(f"upper-triangle tests: {upper_candidates:,}")
    print(f"basis build        : {basis_time:.6f} s")

    if args.profile:
        profiler = cProfile.Profile()

        t0 = time.perf_counter()
        profiler.enable()
        calculate_silently(hamiltonian)
        profiler.disable()
        matrix_time = time.perf_counter() - t0

        args.output.parent.mkdir(parents=True, exist_ok=True)
        profiler.dump_stats(args.output)

        print(f"matrix build       : {matrix_time:.6f} s")
        print(f"nnz                : {hamiltonian.sparse_matrix.nnz}")
        print(f"profile             : {args.output}")

        stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stream)
        stats.strip_dirs().sort_stats("cumtime").print_stats(30)

        print()
        print("Top cumulative-time functions")
        print(stream.getvalue())

    else:
        t0 = time.perf_counter()
        calculate_silently(hamiltonian)
        matrix_time = time.perf_counter() - t0

        print(f"matrix build       : {matrix_time:.6f} s")
        print(f"nnz                : {hamiltonian.sparse_matrix.nnz}")


if __name__ == "__main__":
    main()
