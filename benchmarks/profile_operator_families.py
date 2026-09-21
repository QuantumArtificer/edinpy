"""Benchmark representative optimized fermionic operator families."""

from __future__ import annotations

import argparse
import time
from math import comb

import numpy as np
import scipy.sparse

from edinpy import fermion as edf
from edinpy.fermion._execution import compile_operator


def _sum_terms(terms):
    """Return the symbolic sum of an iterable of operator terms."""
    expression = 0
    for term in terms:
        expression = expression + term
    return expression


def build_operator_family(length, particles, family):
    """Construct one benchmark Hamiltonian family.

    Parameters
    ----------
    length : int
        Number of fermionic modes.
    particles : int
        Fixed fermion number.
    family : {'hopping', 'density', 'quartic', 'mixed'}
        Operator family to benchmark.

    Returns
    -------
    sector : NParticleSector
        Fixed-particle-number sector.
    hamiltonian : Hamiltonian
        Hamiltonian object for the benchmark expression.
    compiled : CompiledOperator
        Lowered execution object used for kernel statistics.
    basis_time : float
        Sector and basis construction time in seconds.
    """
    modes = edf.FermionModes(edf.DoF(length, name="mode"))

    start = time.perf_counter()
    sector = edf.NParticleSector(modes, N=particles)
    basis_time = time.perf_counter() - start

    c = edf.set_notation(edf.Annihilation, modes)
    cd = edf.set_notation(edf.Creation, modes)
    n = edf.set_notation(edf.Number, modes)

    if family == "hopping":
        terms = []
        for i in range(length - 1):
            terms.extend((
                -cd(i) * c(i + 1),
                -cd(i + 1) * c(i),
            ))
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
            terms.extend((
                -cd(i) * c(i + 1),
                -cd(i + 1) * c(i),
                0.5 * n(i) * n(i + 1),
            ))
    else:
        raise ValueError(f"Unknown operator family: {family}")

    expression = _sum_terms(terms)
    compiled = compile_operator(expression)
    hamiltonian = edf.Hamiltonian(expression, sector)
    return sector, hamiltonian, compiled, basis_time


def main():
    """Run the requested benchmark families and print timing statistics."""
    _ = np.empty(0)
    _ = scipy.sparse.csc_matrix((0, 0))

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
        sector, hamiltonian, compiled, basis_time = build_operator_family(
            args.length,
            args.particles,
            family,
        )
        start = time.perf_counter()
        matrix = hamiltonian.calc_matrix()
        matrix_time = time.perf_counter() - start

        print(
            f"{family:8s} "
            f"basis={basis_time:.6f} s  "
            f"matrix={matrix_time:.6f} s  "
            f"nnz={matrix.nnz:8d}  "
            f"kernels={compiled.stats}"
        )


if __name__ == "__main__":
    main()
