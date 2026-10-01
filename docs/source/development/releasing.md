# Releasing

EDinPy releases are published from Git tags by GitHub Actions. Do not upload
distribution files to PyPI or GitHub Releases manually.

## 1. One-time PyPI setup

Configure a PyPI Trusted Publisher for this repository:

- PyPI project: `edinpy`
- GitHub owner: `QuantumArtificer`
- GitHub repository: `edinpy`
- Workflow: `release.yml`
- Environment: `pypi`

For the first release, use a pending Trusted Publisher if the `edinpy` project
does not yet exist on PyPI. No PyPI API token is required by the release
workflow.

## 2. Prepare the release on `main`

Update the release version in `src/edinpy/_version.py` and keep
`CITATION.cff` and `CHANGELOG.md` consistent with that version.

Before tagging, the normal CI and documentation workflows on `main` should be
green.

Local validation may be run with:

```bash
python -m pytest
python -m ruff check src tests examples benchmarks docs/scripts

for example in examples/fermion/*.py examples/boson/*.py; do
    python "$example"
done

python -m sphinx -W --keep-going -b html docs/source docs/_build/html
```

Commit the final release state to `main` before creating the tag.

## 3. Tag the release

Create an annotated tag whose version matches the package metadata:

```bash
git tag -a vX.Y.Z -m "EDinPy X.Y.Z"
git push origin vX.Y.Z
```

Pushing the tag starts `.github/workflows/release.yml`.

## 4. Automated release workflow

The release workflow:

1. checks that the tag, package version, `CITATION.cff`, and `CHANGELOG.md`
   agree;
2. runs the test suite, optional Numba tests, Ruff, examples, and
   documentation build;
3. builds the wheel and source distribution;
4. validates the distributions with Twine and smoke-tests the built wheel;
5. publishes the verified distributions to PyPI using Trusted Publishing;
6. creates the GitHub Release and attaches the exact wheel, source
   distribution, and `SHA256SUMS`.

Documentation figures are generated and reviewed during development. The
release workflow builds the committed documentation but does not require
byte-for-byte SVG regeneration on a different runner environment.

Do not create the GitHub Release before the workflow runs. Do not upload
release assets or invoke `twine upload` manually.

## 5. Documentation

The separate documentation workflow deploys the `main` documentation to
GitHub Pages.

## 6. Archive the release

After the GitHub Release exists, archive it in Zenodo or another long-term
research-software archive. If the repository is connected to Zenodo, the
GitHub Release can be ingested through that integration.

Published work should cite the archived version used for the calculation.
