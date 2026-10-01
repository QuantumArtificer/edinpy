# Testing

Install a development checkout with the test dependencies:

```bash
python -m pip install -e ".[test]"
```

Run the complete suite from the repository root:

```bash
python -m pytest
```

The suite covers shared algebra and solver behavior together with statistics-specific basis construction, operator action, sparse Hamiltonians, matrix-free execution, symmetry-constrained sectors, and observables.

Independent small-system reference calculations are used for selected compiler and matrix-construction tests so that optimized execution is compared against a separate construction path.

## Examples

Run every executable example:

```bash
for example in examples/fermion/*.py examples/boson/*.py; do
    python "$example"
done
```

## Documentation

Install the documentation dependencies and treat warnings as errors:

```bash
python -m pip install -e ".[docs]"
python -m sphinx -W --keep-going -b html docs/source docs/_build/html
```

## Package build

The release checks build both a source distribution and a wheel. See {doc}`releasing` for the complete procedure.
