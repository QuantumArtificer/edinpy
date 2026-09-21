from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from edinpy import __version__

project = "EDinPy"
author = "Alex Santacruz"
copyright = "2026, Alex Santacruz"
version = __version__
release = __version__

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinx_design",
]

autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_typehints = "description"
napoleon_numpy_docstring = True
napoleon_google_docstring = False

myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
]
myst_heading_anchors = 3

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

templates_path = ["_templates"]
html_theme = "pydata_sphinx_theme"
html_title = f"EDinPy {release}"
html_static_path = ["_static"]
html_theme_options = {
    "show_toc_level": 2,
    "navigation_depth": 4,
    "header_links_before_dropdown": 6,
    "icon_links": [
        {
            "name": "Source",
            "url": "https://github.com/QuantumArtificer/edinpy",
            "icon": "fa-brands fa-github",
        }
    ],
}

copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
