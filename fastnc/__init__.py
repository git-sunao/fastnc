"""Top-level package for fastnc."""
from __future__ import annotations

import logging as _logging

_logging.getLogger(__name__).addHandler(_logging.NullHandler())

from . import bispectrum
from . import coupling
from . import hankel
from . import threepcf

__all__ = ["bispectrum", "coupling", "hankel", "threepcf"]

__version__ = "2.0.4"
__author__ = 'Sunao Sugiyama, Rafael Heringer Gomes'
__url__ = 'https://github.com/git-sunao/fastnc'
