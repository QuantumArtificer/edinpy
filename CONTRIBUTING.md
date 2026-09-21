# Contributing to EDinPy

Bug reports, documentation fixes, tests, and focused code changes are welcome.

## Development setup

Clone the repository and install the development dependencies:

```bash
python -m pip install -e ".[test,docs,dev]"
```

Run the test suite before submitting a change:

```bash
python -m pytest
```

Run the static checks with:

```bash
python -m ruff check src tests examples benchmarks docs/scripts
```

User-facing examples should use only public EDinPy interfaces. Changes to the fermionic algebra or matrix-construction backend should include a correctness test, preferably against an independent calculation or analytic result.

The full contribution guide covers documentation, tests, references, and implementation conventions:

- [Contribution guide](https://quantumartificer.github.io/edinpy/development/contributing.html)
- [Testing](https://quantumartificer.github.io/edinpy/development/testing.html)
- [Benchmarking](https://quantumartificer.github.io/edinpy/development/benchmarking.html)

EDinPy is distributed under the [MIT License](LICENSE). By contributing code, you agree that your contribution may be distributed under the same license.
