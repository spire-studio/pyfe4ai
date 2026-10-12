"""Sphinx configuration for pyfe4ai documentation."""

import os
import re
import sys
from collections import Counter

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
    "gmpy2": ("https://gmpy2.readthedocs.io/en/latest", None),
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

# -- Scheme status ------------------------------------------------------------
# The status counts (|n_schemes| etc.) and the ``scheme-status-table``
# directive are rendered from pyfe4ai.schemes.status at build time, so the
# docs cannot drift from the code.
from docutils import nodes  # noqa: E402
from docutils.statemachine import StringList  # noqa: E402
from sphinx.util.docutils import SphinxDirective  # noqa: E402

from pyfe4ai.schemes import status as _status  # noqa: E402

_status_counts = Counter(entry.status for entry in _status.SCHEME_STATUS.values())
rst_prolog = f"""
.. |n_schemes| replace:: {len(_status.SCHEME_STATUS)}
.. |n_reviewed| replace:: {_status_counts[_status.REVIEWED]}
.. |n_experimental| replace:: {_status_counts[_status.EXPERIMENTAL]}
.. |n_unreviewed| replace:: {_status_counts[_status.UNREVIEWED]}
.. |n_known_insecure| replace:: {_status_counts[_status.KNOWN_INSECURE]}
"""


def _rst_text(text: str) -> str:
    """Escape reST inline markup in plain text."""
    return re.sub(r"([\\`*_|])", r"\\\1", text)


class SchemeStatusTable(SphinxDirective):
    """Render ``pyfe4ai.schemes.status.SCHEME_STATUS`` as a table."""

    def run(self):
        lines = [
            ".. list-table::",
            "   :header-rows: 1",
            "   :widths: 14 9 8 22 47",
            "",
            "   * - Scheme",
            "     - Status",
            "     - Fidelity",
            "     - Construction",
            "     - Notes",
        ]
        for (family, variant), entry in _status.SCHEME_STATUS.items():
            note = entry.issue if entry.status == _status.KNOWN_INSECURE else entry.note
            lines += [
                f"   * - :mod:`{family}/{variant} <{entry.module}>`",
                f"     - {entry.status}",
                f"     - {entry.fidelity}",
                f"     - {_rst_text(entry.reference)}",
                f"     - {_rst_text(note)}" if note else "     -",
            ]
        container = nodes.container()
        self.state.nested_parse(
            StringList(lines, source="pyfe4ai.schemes.status.SCHEME_STATUS"),
            self.content_offset,
            container,
        )
        return container.children


def setup(app):
    app.add_directive("scheme-status-table", SchemeStatusTable)
