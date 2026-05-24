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

    def interpolate(self, config, **params):
        from .interpolate import InterpolatedBispectrum3D
        return InterpolatedBispectrum3D.from_bispectrum(self, config, **params)


class Bispectrum2D:
    """Object for a 2D bispectrum ``B(ell1, ell2, ell3)``.

    In this package, 2D bispectra are angular bispectra by default, so the
    public class name intentionally omits ``Angular``.  The class can be used
    directly with an evaluator, or subclassed by analytic/tabulated models.
    """
    support = Support2D()

    def __init__(self, evaluator=None, support=None):
        self._evaluator = evaluator
        if support is not None:
            self.support = support

    def __call__(self, ell1, ell2, ell3, **params):
        return self.evaluate(ell1, ell2, ell3, **params)

    def evaluate(self, ell1, ell2, ell3, **params):
        if getattr(self, "_evaluator", None) is None:
            raise NotImplementedError
        return self._evaluator(ell1, ell2, ell3, **params)

    def interpolate(self, config, **params):
        from .interpolate import InterpolatedBispectrum2D
        return InterpolatedBispectrum2D.from_bispectrum(self, config, **params)

    def multipole(self, config=None, basis="fourier-even", regulator=None, **params):
        from .multipole import BispectrumMultipole2D
        return BispectrumMultipole2D.from_bispectrum2d(
            self,
            config=config,
            basis=basis,
            regulator=regulator,
            **params,
        )

