"""Measure fixed-N basis construction and compact execution-view memory.

The worker runs in a fresh subprocess so RSS measurements do not inherit
allocations from another statistics family or representation.
"""

from __future__ import annotations

import argparse
import json
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


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
    import scipy

    return {
        "python": sys.version.split()[0],
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "edinpy": getattr(edinpy, "__version__", "unknown"),
    }


def _build_basis(statistics_name, n_modes, particles):
    if statistics_name == "fermion":
        from edinpy import fermion as ed

        modes = ed.FermionModes(ed.DoF(n_modes, name="mode"))
    else:
        from edinpy import boson as ed

        modes = ed.BosonModes(ed.DoF(n_modes, name="mode"))

    start = time.perf_counter()
    sector = ed.NParticleSector(modes, N=particles).build()
    build_seconds = time.perf_counter() - start
    return sector.basis, build_seconds


def _worker(args):
    rss_before = _current_rss_mib()
    peak_before = _peak_rss_mib()

    basis, build_seconds = _build_basis(
        args.statistics,
        args.modes,
        args.particles,
    )
    rss_after_build = _current_rss_mib()
    peak_after_build = _peak_rss_mib()
    initial_storage = basis.storage_bytes
    initially_materialized = basis.is_materialized

    start = time.perf_counter()
    if args.statistics == "fermion":
        execution_view = basis._execution_states()
    else:
        execution_view = basis._uint64_words()
    execution_seconds = time.perf_counter() - start

    execution_storage = basis.storage_bytes
    rss_after_execution = _current_rss_mib()
    peak_after_execution = _peak_rss_mib()

    result = {
        "statistics": args.statistics,
        "modes": args.modes,
        "particles": args.particles,
        "dimension": basis.dimension,
        "build_seconds": build_seconds,
        "initially_materialized": initially_materialized,
        "initial_storage_bytes": initial_storage,
        "execution_view_seconds": execution_seconds,
        "execution_storage_bytes": execution_storage,
        "execution_view_shape": list(execution_view.shape),
        "execution_view_dtype": str(execution_view.dtype),
        "rss_before_mib": rss_before,
        "rss_after_build_mib": rss_after_build,
        "rss_after_execution_mib": rss_after_execution,
        "peak_before_mib": peak_before,
        "peak_after_build_mib": peak_after_build,
        "peak_after_execution_mib": peak_after_execution,
    }

    if args.materialize_public:
        start = time.perf_counter()
        if args.statistics == "fermion":
            public_states = basis.states
        else:
            public_states = basis.packed_states
        result["public_materialization_seconds"] = time.perf_counter() - start
        result["public_state_count"] = len(public_states)
        result["storage_after_public_bytes"] = basis.storage_bytes
        result["rss_after_public_mib"] = _current_rss_mib()
        result["peak_after_public_mib"] = _peak_rss_mib()

    print(json.dumps(result))


def _run_worker(args):
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
    ]
    if args.materialize_public:
        command.append("--materialize-public")
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout.strip().splitlines()[-1])


def _resolve_json_path(path):
    if path is None or path.is_absolute() or path.parent != Path("."):
        return path
    return RESULTS_DIR / path


def _print_result(result):
    compact_mib = result["execution_storage_bytes"] / (1024.0 * 1024.0)
    print(
        f"statistics={result['statistics']}, modes={result['modes']}, "
        f"particles={result['particles']}, dimension={result['dimension']}"
    )
    print(
        f"build={result['build_seconds']:.6f} s | "
        f"initial storage={result['initial_storage_bytes']} B | "
        f"compact view={compact_mib:.3f} MiB in "
        f"{result['execution_view_seconds']:.6f} s"
    )
    print(
        f"RSS after build={result['rss_after_build_mib']:.1f} MiB | "
        f"after compact view={result['rss_after_execution_mib']:.1f} MiB | "
        f"peak={result['peak_after_execution_mib']:.1f} MiB"
    )
    if "storage_after_public_bytes" in result:
        public_mib = result["storage_after_public_bytes"] / (1024.0 * 1024.0)
        print(
            f"after public tuple={public_mib:.3f} MiB | "
            f"peak={result['peak_after_public_mib']:.1f} MiB"
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Measure EDinPy implicit/compact fixed-N basis storage in an "
            "isolated subprocess."
        )
    )
    parser.add_argument("--statistics", choices=STATISTICS, default="fermion")
    parser.add_argument("--modes", type=int, default=20)
    parser.add_argument("--particles", type=int, default=10)
    parser.add_argument(
        "--materialize-public",
        action="store_true",
        help="also materialize the compatibility .states/.packed_states tuple",
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
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.modes <= 0:
        parser.error("--modes must be positive")
    if args.particles < 0:
        parser.error("--particles must be non-negative")
    if args.statistics == "fermion" and args.particles > args.modes:
        parser.error("fermionic --particles must not exceed --modes")
    args.json = _resolve_json_path(args.json)
    return args


def main():
    args = parse_args()
    if args.worker:
        _worker(args)
        return

    metadata = _environment_metadata()
    result = _run_worker(args)
    print(
        f"EDinPy {metadata['edinpy']} | Python {metadata['python']} | "
        f"NumPy {metadata['numpy']} | SciPy {metadata['scipy']}"
    )
    _print_result(result)

    if args.json is not None:
        payload = {
            "environment": metadata,
            "parameters": {
                "statistics": args.statistics,
                "modes": args.modes,
                "particles": args.particles,
                "materialize_public": args.materialize_public,
            },
            "result": result,
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"\nWrote {args.json}")


if __name__ == "__main__":
    main()
