"""Top-level package for fastnc."""
from __future__ import annotations

import logging as _logging

_logging.getLogger(__name__).addHandler(_logging.NullHandler())

from . import bispectrum
from . import coupling
from . import hankel
from . import multipole
from . import projection
from . import threepcf
from ._logging import configure_logging, disable_logging

__all__ = [
    "bispectrum",
    "coupling",
    "hankel",
    "multipole",
    "projection",
    "threepcf",
    "configure_logging",
    "disable_logging",
]

__version__ = "2.0.74"
__author__ = 'Sunao Sugiyama, Rafael Heringer Gomes'
__url__ = 'https://github.com/git-sunao/fastnc'
