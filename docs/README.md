# EDinPy documentation

The documentation source lives in `docs/source` and is built with Sphinx, MyST, and the PyData Sphinx theme. The site is organized from the package level downward: a short package orientation, separate fermionic and bosonic sections, package-level validation and benchmarks, and development notes.

The fermionic section currently contains the complete scientific manual: getting started, workflow-oriented user guide, worked examples, theory/numerical methods, and API reference. The bosonic section is intentionally a top-layer placeholder until the bosonic backend receives the corresponding architectural and documentation overhaul.

Install the documentation dependencies from the repository root:

```bash
python -m pip install -e ".[docs]"
```

Generate the scientific figures with EDinPy itself:

```bash
make -C docs figures
```

The generated SVG files are tracked in `docs/source/_static/figures`. Keeping the generation script with the source makes every plotted data set reproducible.

Build the site with warnings treated as errors:

```bash
make -C docs html
```

or directly:

```bash
sphinx-build -W -b html docs/source docs/_build/html
```

The generated HTML is written to `docs/_build/html` and is not tracked by Git.
