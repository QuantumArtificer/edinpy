# Contributing

Changes should keep the public calculation close to the underlying many-body problem. Mode labels, Hilbert-space restrictions, operator algebra, and numerical approximations should remain visible to the user.

## Public API

Public functions and classes should have NumPy-style docstrings. Document parameters, return values, exceptions, and numerical behavior that a user needs to interpret the result.

Common operator helpers should expand to the same symbolic algebra that a user could write by hand. Performance-specific bit masks, sparse emission buffers, and compiler details belong in private implementation modules.

## Tests

A change to the fermionic algebra or matrix-construction backend should include a correctness test. Prefer an independent mathematical reference when possible. Analytic small-system results are useful for tests that span several layers of the package.

Run

```bash
python -m pytest
```

before opening a pull request.

## Style

The repository uses Ruff for basic static checks:

```bash
python -m ruff check src tests examples benchmarks docs/scripts
```

Readable scientific code is preferred over compact code. Variables such as `modes`, `sector`, `H`, and `psi0` are appropriate when they match standard notation and remain clear in context.

## References and provenance

If an implementation follows a published algorithm, cite the original source in the relevant docstring or documentation page. Record implementation-level algorithmic provenance in [`PROVENANCE.md`](https://github.com/QuantumArtificer/edinpy/blob/main/PROVENANCE.md).
