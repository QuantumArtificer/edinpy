"""Benchmark optimized fermionic operator families."""

from __future__ import annotations

import argparse
import os
import time
from contextlib import redirect_stdout
from math import comb

from edinpy import fermion as edf
from edinpy.fermion._execution import compile_operator


def _sum_terms(terms):
    expression = None
    for term in terms:
        expression = term if expression is None else expression + term
    return expression


def build_operator_family(length, particles, family):
    edf.clear()
    edf.DoF(length, name="site")

    t0 = time.perf_counter()
    model = edf.model(particles)
    basis_time = time.perf_counter() - t0

    c = [edf.Annihilation(i) for i in range(length)]

    if family == "hopping":
        terms = []
        for i in range(length - 1):
            terms.extend(
                (
                    -1.0 * c[i].dag * c[i + 1],
                    -1.0 * c[i + 1].dag * c[i],
                )
            )

    elif family == "density":
        terms = [
            edf.Number(i) * edf.Number(i + 1)
            for i in range(length - 1)
        ]

    elif family == "quartic":
        terms = []
        for i in range(0, length - 3, 2):
            pair_transfer = (
                c[i].dag * c[i + 1].dag * c[i + 3] * c[i + 2]
            )
            terms.extend((pair_transfer, pair_transfer.dag))

    elif family == "mixed":
        terms = []
        for i in range(length - 1):
            terms.extend(
                (
                    -1.0 * c[i].dag * c[i + 1],
                    -1.0 * c[i + 1].dag * c[i],
                    0.5 * edf.Number(i) * edf.Number(i + 1),
                )
            )

    else:
        raise ValueError(f"Unknown family: {family}")

    expression = _sum_terms(terms)
    compiled = compile_operator(expression)

    return model, edf.hamiltonian(expression), compiled, basis_time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--length", type=int, default=16)
    parser.add_argument("--particles", type=int, default=8)
    parser.add_argument(
        "--families",
        nargs="+",
        default=("hopping", "density", "quartic", "mixed"),
        choices=("hopping", "density", "quartic", "mixed"),
    )
    args = parser.parse_args()

    expected_basis = comb(args.length, args.particles)
    print(
        f"L={args.length}, N={args.particles}, "
        f"Hilbert dimension={expected_basis}"
    )

    for family in args.families:
        model, hamiltonian, compiled, basis_time = build_operator_family(
            args.length,
            args.particles,
            family,
        )

        with open(os.devnull, "w") as devnull, redirect_stdout(devnull):
            t0 = time.perf_counter()
            hamiltonian.calc_matrix()
            matrix_time = time.perf_counter() - t0

        print(
            f"{family:8s} "
            f"basis={basis_time:.6f} s  "
            f"matrix={matrix_time:.6f} s  "
            f"nnz={hamiltonian.sparse_matrix.nnz:8d}  "
            f"kernels={compiled.stats}"
        )


if __name__ == "__main__":
    main()
