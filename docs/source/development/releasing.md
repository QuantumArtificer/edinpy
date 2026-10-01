# Releasing

## 1. Start from a clean checkout

Install the development dependencies in the checkout used for release validation:

```bash
python -m pip install -e ".[test,docs,dev,numba]"
```

The package version is defined in `src/edinpy/_version.py`. Release metadata in `CITATION.cff` and `CHANGELOG.md` should describe the same release.

## 2. Run source checks

```bash
python -m pytest
python -m ruff check src tests examples benchmarks docs/scripts
```

Run every example:

```bash
for example in examples/fermion/*.py examples/boson/*.py; do
    python "$example"
done
```

Build the documentation with warnings treated as errors:

```bash
python -m sphinx -W --keep-going -b html docs/source docs/_build/html
```

## 3. Build the distributions

```bash
rm -rf build dist src/*.egg-info
python -m build
python -m twine check dist/*
```

Inspect the wheel and source distribution before upload. Runtime source, package metadata, citation metadata, and the license belong in the distributions. Development-only tests, benchmarks, documentation sources, examples, generated documentation, caches, and local environments are excluded from the published archives.

## 4. Test the wheel

Install the wheel into a temporary target directory so the smoke test does not import the editable source tree:

```bash
rm -rf /tmp/edinpy-wheel-test
python -m pip install --no-deps --target /tmp/edinpy-wheel-test dist/*.whl
cd /tmp
PYTHONPATH=/tmp/edinpy-wheel-test python -c "import edinpy; print(edinpy.__version__); print(edinpy.__file__)"
```

The reported module path should point inside `/tmp/edinpy-wheel-test`. Run a small fermionic and bosonic calculation with the same `PYTHONPATH` to verify the installed wheel.

## 5. Tag and publish

Commit the release state on `main`, create an annotated version tag such as `vX.Y.Z`, and create a GitHub release from that tag.

Upload the exact files from `dist/` to PyPI. PyPI Trusted Publishing is preferred when the repository is configured for it.

The documentation workflow deploys the `main` documentation to GitHub Pages.

## 6. Archive the release

Archive the GitHub release in Zenodo or another long-term research-software archive. Add the release DOI to the citation metadata once it exists. Published work should cite the archived version used for the calculation.
