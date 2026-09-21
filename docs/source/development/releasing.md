# Releasing

This page describes the release checks for maintainers.

## 1. Start from a clean checkout

Create a fresh environment and install all development dependencies:

```bash
python -m pip install -e ".[test,docs,dev]"
```

Check that the version agrees in `pyproject.toml`, `src/edinpy/__init__.py`, and `CITATION.cff`.

## 2. Run the release checks

Run the tests and style checks:

```bash
python -m pytest
python -m ruff check src tests examples benchmarks docs/scripts
```

Run every example:

```bash
for example in examples/*.py; do
    python "$example"
done
```

Build the documentation with warnings treated as errors:

```bash
python -m sphinx -W --keep-going -b html docs/source docs/_build/html
```

## 3. Build the distributions

Remove old build products, then build a source distribution and wheel:

```bash
rm -rf build dist src/*.egg-info
python -m build
python -m twine check dist/*
```

Inspect the archive contents before upload. The source distribution should contain the tests, examples, documentation sources, benchmarks, citation metadata, and license. Generated documentation, profiler output, caches, and local environments should not be present.

## 4. Test the wheel in a clean environment

Create a separate environment and install the wheel from `dist/`. Import EDinPy, run the test suite against the installed package if practical, and run at least one example.

This step catches packaging errors that an editable install can hide.

## 5. Tag and publish

Commit the release state on `main`, create an annotated version tag such as `v0.2.0`, and create a GitHub release from that tag.

Upload the exact files from `dist/` to PyPI. PyPI Trusted Publishing is preferred over long-lived API tokens when the repository has been configured for it.

The documentation workflow deploys the `main` documentation to GitHub Pages. Before the first deployment, set **Settings > Pages > Build and deployment > Source** to **GitHub Actions** in the repository. A release should leave `main` at the same documented API as the published package.

## 6. Archive the release

Archive the GitHub release in Zenodo or another long-term research-software archive. Add the release DOI to the citation metadata once it exists. Published work should cite the archived version actually used for the calculation.
