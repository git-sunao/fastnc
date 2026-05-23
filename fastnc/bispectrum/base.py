"""Base bispectrum objects."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional
import numpy as np

from .support import Support3D, Support2D


class Bispectrum3D:
    """Base object for a 3D bispectrum ``B(k1,k2,k3,z)``."""
    support = Support3D()

    def __call__(self, k1, k2, k3, z, **params):
        return self.evaluate(k1, k2, k3, z, **params)

    def evaluate(self, k1, k2, k3, z, **params):
        raise NotImplementedError


class Bispectrum2D:
    """Base object for a 2D bispectrum ``B(ell1,ell2,ell3)``.

    In this package, 2D bispectra are angular bispectra by default, so the
    public class name intentionally omits ``Angular``.
    """
    support = Support2D()

    def __call__(self, ell1, ell2, ell3, **params):
        return self.evaluate(ell1, ell2, ell3, **params)

    def evaluate(self, ell1, ell2, ell3, **params):
        raise NotImplementedError

    def multipole(self, config=None, basis="fourier-even", regulator=None, **params):
        from .multipole import BispectrumMultipole2DCalculator
        return BispectrumMultipole2DCalculator(config=config, basis=basis).compute(
            self,
            regulator=regulator,
            **params,
        )
