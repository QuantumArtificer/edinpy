# Testing

Install a development checkout with the test dependencies:

```bash
python -m pip install -e ".[test]"
```

Run the complete test suite from the repository root:

```bash
python -m pytest
```

The fermionic tests cover mode indexing, fixed-$N$ basis enumeration, canonical anticommutation relations, symbolic state algebra, common operators, sparse matrix construction, dense and sparse eigensolvers, eigenstate conversion, and observables.

Several tests use an independent state-by-state bit-string calculation instead of EDinPy's compiler. This reduces the chance that a compiler error is hidden by comparing two code paths that share the same implementation.

## Examples

Executable examples are also checked in continuous integration:

```bash
for example in examples/*.py; do
    python "$example"
done
```

Examples are part of the user-facing package. They should use only public EDinPy interfaces.

## Documentation

Install the documentation dependencies and build with warnings treated as errors:

```bash
python -m pip install -e ".[docs]"
python -m sphinx -W --keep-going -b html docs/source docs/_build/html
```

A warning-free Sphinx build is required because broken cross-references and malformed API documentation are easy to miss in source files.

## Package build

The release checks also build both a source distribution and a wheel. See {doc}`releasing` for the complete release procedure.
