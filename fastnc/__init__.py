"""Top-level package for fastnc.

The package is organized into four main subpackages:

- ``fastnc.bispectrum``: 3D/2D bispectrum models, LOS projection, and
  bispectrum multipole decomposition.
- ``fastnc.coupling``: spin-dependent multipole-coupling functions.
- ``fastnc.hankel``: FFTLog / double-Hankel transform wrappers.
- ``fastnc.threepcf``: 3PCF kernel construction, Hankel transforms,
  multipole resummation, and projection conversion.

Typical usage
-------------
    from fastnc import bispectrum as bs
    from fastnc import coupling as cg
    from fastnc import threepcf as tpcf
    from fastnc import hankel
"""

from __future__ import annotations

try:
    from . import bispectrum
except ImportError:
    bispectrum = None

try:
    from . import coupling
except ImportError:
    coupling = None

try:
    from . import hankel
except ImportError:
    hankel = None

try:
    from . import threepcf
except ImportError:
    threepcf = None

__all__ = [
    "bispectrum",
    "coupling",
    "hankel",
    "threepcf",
]

__version__ = "0.1.0"