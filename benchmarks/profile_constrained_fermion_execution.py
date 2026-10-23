"""Benchmark constrained fermion matrix-free execution on a mixed workload.

The default problem is intentionally large enough to expose projected-basis
destination-indexing costs while remaining independent of any particular
material, lattice geometry, or point-group symmetry.

It uses two independently conserved population labels and a generic mixture of
number products, one-body hopping, and quartic number-conserving monomials.
"""

from __future__ import annotations

import argparse
import json
import platform
import resource
import statistics
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import scipy


RESULTS_DIR = Path(__file__).resolve().parent / "results"


def _peak_rss_mib() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() == "Darwin":
        return value / (1024.0 * 1024.0)
    return value / 1024.0


def _environment_metadata() -> dict[str, object]:
    import edinpy

    try:
        import numba
    except ImportError:
        numba_version = None
    else:
        numba_version = numba.__version__

    return {
        "python": sys.version.split()[0],
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "numba": numba_version,
        "edinpy": getattr(edinpy, "__version__", "unknown"),
    }


def _label(
    site: int,
    population: int,
    *,
    interleaved_populations: bool,
) -> tuple[int, int]:
    """Return coordinates for blocked or interleaved population ordering."""
    return (
        (population, site)
        if interleaved_populations
        else (site, population)
    )


def _build_expression(
    modes,
    *,
    sites: int,
    number_terms: int,
    hopping_groups: int,
    quartic_processes: int,
    interleaved_populations: bool,
):
    """Build a fully lowered mixed Hamiltonian with exact requested term counts."""
    from edinpy import fermion as edf

    labels = [
        _label(
            site,
            population,
            interleaved_populations=interleaved_populations,
        )
        for site in range(sites)
        for population in (0, 1)
    ]

    density_pairs = list(combinations(labels, 2))
    if number_terms > len(density_pairs):
        raise ValueError(
            f"Requested {number_terms} number products but only "
            f"{len(density_pairs)} distinct mode pairs are available."
        )

    hopping_pairs = []
    for population in (0, 1):
        population_labels = [
            _label(
                site,
                population,
                interleaved_populations=interleaved_populations,
            )
            for site in range(sites)
        ]
        hopping_pairs.extend(combinations(population_labels, 2))
    if hopping_groups > len(hopping_pairs):
        raise ValueError(
            f"Requested {hopping_groups} hopping groups but only "
            f"{len(hopping_pairs)} population-preserving pairs are available."
        )

    quartic_specs = []
    for source_a in range(sites):
        for destination_a in range(sites):
            if destination_a == source_a:
                continue
            for source_b in range(sites):
                for destination_b in range(sites):
                    if destination_b == source_b:
                        continue
                    forward = (
                        source_a,
                        destination_a,
                        source_b,
                        destination_b,
                    )
                    reverse = (
                        destination_a,
                        source_a,
                        destination_b,
                        source_b,
                    )
                    if forward < reverse:
                        quartic_specs.append(forward)
    if quartic_processes > len(quartic_specs):
        raise ValueError(
            f"Requested {quartic_processes} Hermitian quartic processes but only "
            f"{len(quartic_specs)} distinct reversible processes are available."
        )

    expression = 0

    for index, (left, right) in enumerate(density_pairs[:number_terms]):
        coefficient = 0.08 + 0.001 * (index % 31)
        expression += edf.DensityDensity(left, right, coefficient, modes)

    for index, (left, right) in enumerate(hopping_pairs[:hopping_groups]):
        coefficient = -0.21 - 0.002 * (index % 17)
        expression += edf.Hopping(left, right, coefficient, modes)

    c = edf.set_notation(edf.Annihilation, modes)
    cd = edf.set_notation(edf.Creation, modes)
    for index, (source_a, destination_a, source_b, destination_b) in enumerate(
        quartic_specs[:quartic_processes]
    ):
        coefficient = 0.012 + 0.0001 * (index % 29)
        term = (
            coefficient
            * cd(
                _label(
                    destination_a,
                    0,
                    interleaved_populations=interleaved_populations,
                )
            )
            * cd(
                _label(
                    destination_b,
                    1,
                    interleaved_populations=interleaved_populations,
                )
            )
            * c(
                _label(
                    source_b,
                    1,
                    interleaved_populations=interleaved_populations,
                )
            )
            * c(
                _label(
                    source_a,
                    0,
                    interleaved_populations=interleaved_populations,
                )
            )
        )
        expression += term + term.dag

    return expression


def _build_problem(args):
    from edinpy import fermion as edf

    population = edf.DoF(2, name="population", labels=("a", "b"))
    site = edf.DoF(args.sites, name="site")
    modes = (
        edf.FermionModes(population, site)
        if args.interleaved_populations
        else edf.FermionModes(site, population)
    )

    start = time.perf_counter()
    sector = (
        edf.NParticleSector(
            modes,
            N=args.particles_a + args.particles_b,
        )
        .project_particles(
            "population",
            a=args.particles_a,
            b=args.particles_b,
        )
        .build()
    )
    sector_seconds = time.perf_counter() - start

    start = time.perf_counter()
    expression = _build_expression(
        modes,
        sites=args.sites,
        number_terms=args.number_terms,
        hopping_groups=args.hopping_groups,
        quartic_processes=args.quartic_processes,
        interleaved_populations=args.interleaved_populations,
    )
    expression_seconds = time.perf_counter() - start

    start = time.perf_counter()
    ham = edf.Hamiltonian(expression, sector)
    compilation_seconds = time.perf_counter() - start

    return ham, {
        "sector_seconds": sector_seconds,
        "expression_seconds": expression_seconds,
        "compilation_seconds": compilation_seconds,
    }


def _normalized_vector(dimension: int, seed: int, *, complex_valued: bool):
    rng = np.random.default_rng(seed)
    vector = rng.normal(size=dimension)
    if complex_valued:
        vector = vector + 1j * rng.normal(size=dimension)
    vector /= np.linalg.norm(vector)
    return vector


def _measure_matvec(operator, vector, repeat: int):
    start = time.perf_counter()
    result = operator @ vector
    first_seconds = time.perf_counter() - start

    samples = []
    for _ in range(repeat):
        start = time.perf_counter()
        result = operator @ vector
        samples.append(time.perf_counter() - start)

    return result, first_seconds, samples


def _prepare_numba_plan(ham) -> float:
    from edinpy.fermion._numba_execution import _get_execution_plan

    start = time.perf_counter()
    _get_execution_plan(ham._compiled, ham.sector.basis)
    return time.perf_counter() - start


def _lookup_probe(ham, *, max_states: int, max_masks: int):
    """Measure isolated projected destination lookup throughput."""
    try:
        from numba import njit
    except ImportError:
        return None

    from edinpy.fermion._numba_execution import _basis_row_uint64

    states = np.asarray(ham.sector.basis._execution_states(), dtype=np.uint64)
    hopping_masks = []
    plan = ham._compiled._numba_operator_plan
    for values in (
        plan.simple_hopping_transition_masks,
        plan.parity_free_hopping_transition_masks,
        plan.signed_hopping_transition_masks,
    ):
        hopping_masks.extend(int(value) for value in values)

    if not hopping_masks:
        return None

    masks = np.asarray(hopping_masks[:max_masks], dtype=np.uint64)
    if states.size <= max_states:
        sampled_states = states
    else:
        indices = np.linspace(
            0,
            states.size - 1,
            max_states,
            dtype=np.int64,
        )
        sampled_states = states[indices]

    empty_rank = np.empty((0, 0, 0), dtype=np.int64)
    empty_popcounts = np.empty(0, dtype=np.uint8)

    @njit(nogil=True)
    def probe(sampled, transition_masks, basis_states):
        lookups = 0
        retained = 0
        checksum = np.int64(0)
        for index in range(sampled.size):
            state = sampled[index]
            for mask_index in range(transition_masks.size):
                mask = transition_masks[mask_index]
                occupation = state & mask
                if occupation == 0 or occupation == mask:
                    continue
                row = _basis_row_uint64(
                    state ^ mask,
                    basis_states,
                    False,
                    empty_rank,
                    empty_popcounts,
                )
                lookups += 1
                if row >= 0:
                    retained += 1
                    checksum ^= row
        return lookups, retained, checksum

    probe(sampled_states[: min(8, sampled_states.size)], masks, states)
    start = time.perf_counter()
    lookups, retained, checksum = probe(sampled_states, masks, states)
    seconds = time.perf_counter() - start

    return {
        "sampled_states": int(sampled_states.size),
        "masks": int(masks.size),
        "lookups": int(lookups),
        "retained": int(retained),
        "seconds": seconds,
        "nanoseconds_per_lookup": (
            float(1e9 * seconds / lookups) if lookups else None
        ),
        "checksum": int(checksum),
    }


def _matvec_row(
    execution,
    result,
    first_seconds,
    samples,
    reference,
):
    median = statistics.median(samples)
    return {
        "execution": execution,
        "first_matvec_seconds": first_seconds,
        "steady_matvec_seconds_median": median,
        "steady_matvec_seconds_min": min(samples),
        "steady_matvec_seconds_max": max(samples),
        "speedup_vs_numpy": None,
        "max_abs_delta_vs_numpy": float(np.max(np.abs(result - reference))),
        "result_norm": float(np.linalg.norm(result)),
        "peak_rss_mib": _peak_rss_mib(),
    }


def _resolve_json_path(path: Path | None) -> Path | None:
    if path is None or path.is_absolute() or path.parent != Path("."):
        return path
    return RESULTS_DIR / path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sites", type=int, default=12)
    parser.add_argument("--particles-a", type=int, default=6)
    parser.add_argument("--particles-b", type=int, default=6)
    parser.add_argument("--number-terms", type=int, default=156)
    parser.add_argument("--hopping-groups", type=int, default=72)
    parser.add_argument("--quartic-processes", type=int, default=288)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261023)
    parser.add_argument(
        "--interleaved-populations",
        action="store_true",
        help=(
            "Place the constrained population DoF in the fastest-varying mode "
            "coordinate. The default keeps each population in one contiguous "
            "occupation-bit block."
        ),
    )
    parser.add_argument(
        "--executions",
        nargs="+",
        choices=("numpy", "numba-serial"),
        default=["numpy", "numba-serial"],
    )
    parser.add_argument("--lookup-states", type=int, default=8192)
    parser.add_argument("--lookup-masks", type=int, default=16)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    if args.sites <= 0:
        parser.error("--sites must be positive")
    for name in ("particles_a", "particles_b"):
        value = getattr(args, name)
        if value < 0 or value > args.sites:
            parser.error(f"--{name.replace('_', '-')} must lie in [0, sites]")
    if args.repeat <= 0:
        parser.error("--repeat must be positive")
    if args.lookup_states <= 0 or args.lookup_masks <= 0:
        parser.error("lookup sampling values must be positive")
    return args


def main() -> None:
    args = parse_args()
    environment = _environment_metadata()
    ham, setup = _build_problem(args)
    dimension = int(ham.sector.dimension)

    print(
        f"EDinPy {environment['edinpy']} | Python {environment['python']} | "
        f"NumPy {environment['numpy']} | SciPy {environment['scipy']} | "
        f"Numba {environment['numba']}"
    )
    print(
        f"sites={args.sites}, modes={2 * args.sites}, "
        f"populations=({args.particles_a}, {args.particles_b}), "
        f"dimension={dimension:,}, "
        f"mode_order={'interleaved' if args.interleaved_populations else 'blocked'}"
    )
    print(f"compiled={ham._compiled.stats}")
    print(
        "setup: "
        f"sector={setup['sector_seconds']:.6f}s "
        f"expression={setup['expression_seconds']:.6f}s "
        f"compile={setup['compilation_seconds']:.6f}s"
    )

    vector = _normalized_vector(
        dimension,
        args.seed,
        complex_valued=np.dtype(ham._compiled.data_dtype).kind == "c",
    )

    rows = []
    reference = None
    plan_seconds = None

    for execution in args.executions:
        try:
            start = time.perf_counter()
            operator = ham.aslinearoperator(execution=execution)
            linear_operator_seconds = time.perf_counter() - start
        except (ImportError, NotImplementedError) as exc:
            rows.append(
                {
                    "execution": execution,
                    "status": "unsupported",
                    "reason": str(exc),
                }
            )
            print(f"{execution:>14}: unsupported: {exc}")
            continue

        if execution == "numba-serial":
            plan_seconds = _prepare_numba_plan(ham)

        result, first_seconds, samples = _measure_matvec(
            operator,
            vector,
            args.repeat,
        )

        if reference is None:
            if execution == "numpy":
                reference = result.copy()
            else:
                reference = ham.aslinearoperator(execution="numpy") @ vector

        row = _matvec_row(
            execution,
            result,
            first_seconds,
            samples,
            reference,
        )
        row["linear_operator_seconds"] = linear_operator_seconds
        if execution == "numba-serial":
            row["execution_plan_seconds"] = plan_seconds
        rows.append(row)

        print(
            f"{execution:>14}: first={first_seconds:.6f}s "
            f"steady={row['steady_matvec_seconds_median']:.6f}s "
            f"delta={row['max_abs_delta_vs_numpy']:.3e} "
            f"rss={row['peak_rss_mib']:.1f} MiB"
        )

    numpy_row = next(
        (
            row
            for row in rows
            if row.get("execution") == "numpy"
            and "steady_matvec_seconds_median" in row
        ),
        None,
    )
    if numpy_row is not None:
        baseline = numpy_row["steady_matvec_seconds_median"]
        for row in rows:
            if "steady_matvec_seconds_median" in row:
                row["speedup_vs_numpy"] = (
                    baseline / row["steady_matvec_seconds_median"]
                )

    lookup = None
    if any(row.get("execution") == "numba-serial" for row in rows):
        lookup = _lookup_probe(
            ham,
            max_states=args.lookup_states,
            max_masks=args.lookup_masks,
        )
        if lookup is not None:
            print(
                "projected binary lookup: "
                f"{lookup['lookups']:,} lookups in {lookup['seconds']:.6f}s "
                f"({lookup['nanoseconds_per_lookup']:.1f} ns/lookup)"
            )

    payload = {
        "environment": environment,
        "parameters": {
            "sites": args.sites,
            "modes": 2 * args.sites,
            "particles_a": args.particles_a,
            "particles_b": args.particles_b,
            "dimension": dimension,
            "number_terms": args.number_terms,
            "hopping_groups": args.hopping_groups,
            "quartic_processes": args.quartic_processes,
            "repeat": args.repeat,
            "seed": args.seed,
            "interleaved_populations": args.interleaved_populations,
        },
        "setup": setup,
        "compiled_kernels": ham._compiled.stats,
        "results": rows,
        "lookup_probe": lookup,
    }

    output = _resolve_json_path(args.json)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {output}")


if __name__ == "__main__":
    main()
