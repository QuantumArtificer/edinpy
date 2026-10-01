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
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinx_design",
]

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


exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "**/generated/**",
    "fermion/reference/**",
    "boson/reference/**",
]

html_theme = "pydata_sphinx_theme"
html_title = f"EDinPy {release}"
html_baseurl = "https://quantumartificer.github.io/edinpy/"
html_static_path = ["_static"]
html_theme_options = {
    "show_toc_level": 2,
    "navigation_depth": 4,
    "header_links_before_dropdown": 6,
    "use_edit_page_button": True,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/QuantumArtificer/edinpy",
            "icon": "fa-brands fa-github",
        }
    ],
}
html_context = {
    "github_user": "QuantumArtificer",
    "github_repo": "edinpy",
    "github_version": "main",
    "doc_path": "docs/source",
}

copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True
