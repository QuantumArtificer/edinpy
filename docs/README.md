# EDinPy documentation

The documentation is built with Sphinx, MyST, and the PyData Sphinx theme. Source files live in `docs/source`. Generated HTML is written to `docs/_build/html` and is not tracked by Git.

Install the documentation dependencies from the repository root:

```bash
python -m pip install -e ".[docs]"
```

Build the documentation with warnings treated as errors:

```bash
make -C docs html
```

The equivalent direct command is

```bash
python -m sphinx -W --keep-going -b html docs/source docs/_build/html
```

Scientific figures in `docs/source/_static/figures` are generated with EDinPy and Matplotlib. Regenerate them with

```bash
make -C docs figures
```

The documentation is deployed to GitHub Pages from the `main` branch by `.github/workflows/docs.yml`. The repository must use **GitHub Actions** as its Pages publishing source.
