from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

project = "fastnc"
author = "Sunao Sugiyama, Rafael Heringer Gomes"
copyright = "2026, Sunao Sugiyama and Rafael Heringer Gomes"

from fastnc import __version__

version = __version__
release = __version__

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "nbsphinx",
    "nbsphinx_link",
]

nbsphinx_execute = "never"
nbsphinx_allow_errors = False

autosummary_generate = True
autodoc_typehints = "description"
autodoc_class_signature = "mixed"
napoleon_google_docstring = True
napoleon_numpy_docstring = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "dev"]
html_theme = "furo"
html_title = f"fastnc {release}"
html_static_path = ["_static"]
html_css_files = ["custom.css"]

nitpicky = True
nitpick_ignore = [
    ("py:class", "array_like"),
    ("py:class", "callable"),
    ("py:class", "np.ndarray"),
    ("py:class", "fastnc.bispectrum.interpolation._InterpolationCacheSlot"),
]

autodoc_type_aliases = {
    "np.ndarray": "numpy.ndarray",
}
