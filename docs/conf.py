"""Sphinx configuration for the EDinPy documentation."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

project = "EDinPy"
author = "Alex Santacruz"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
]

napoleon_numpy_docstring = True
napoleon_google_docstring = False
autodoc_typehints = "description"
add_module_names = False

exclude_patterns = ["_build"]
html_theme = "alabaster"
