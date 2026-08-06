"""Sphinx configuration for pyfe4ai documentation."""

import os
import sys

# -- Path setup --------------------------------------------------------------
sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------
project = "pyfe4ai"
copyright = "2026, Spire Studio"
author = "Spire Studio"

from pyfe4ai import __version__  # noqa: E402

version = __version__
release = __version__

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for autodoc -----------------------------------------------------
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
}

# -- Options for Napoleon (Google-style docstrings) --------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_include_special_with_doc = False
napoleon_use_param = True
napoleon_use_rtype = True

# -- Options for sphinx-autodoc-typehints ------------------------------------
always_document_param_types = True
typehints_defaults = "comma"

# -- Options for intersphinx -------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}

# -- Options for autodoc imports ---------------------------------------------
# Pairing-based schemes import charm-crypto lazily; mock it so the docs build
# does not require compiling PBC/charm in CI.
autodoc_mock_imports = ["charm"]

# -- Options for HTML output -------------------------------------------------
html_theme = "furo"
html_static_path = ["_static"]
html_title = f"pyfe4ai {release}"
html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#4d82c4",
        "color-brand-content": "#4d82c4",
    },
    "dark_css_variables": {
        "color-brand-primary": "#6ea3e0",
        "color-brand-content": "#6ea3e0",
    },
}
