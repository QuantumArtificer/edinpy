# Testing

Run the fermionic test suite from the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m pytest tests -q
```

Docstring examples can be checked with:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
python -m pytest --doctest-modules src/edinpy/fermion -q
```

A Sphinx release build should be warning-free:

```bash
sphinx-build -W -b html docs/source docs/_build/html
```
