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


class FunctionBispectrum3D(Bispectrum3D):
    def __init__(self, func: Callable, support: Optional[Support3D] = None):
        self.func = func
        self.support = support or Support3D()

    def evaluate(self, k1, k2, k3, z, **params):
        return self.func(k1, k2, k3, z, **params)


class AngularBispectrum2D:
    """Base object for an angular bispectrum ``B(ell1,ell2,ell3)``."""
    support = Support2D()

    def __call__(self, ell1, ell2, ell3, **params):
        return self.evaluate(ell1, ell2, ell3, **params)

    def evaluate(self, ell1, ell2, ell3, **params):
        raise NotImplementedError

    def multipole(self, config=None, basis="fourier-even", regulator=None, **params):
        from .multipole import BispectrumMultipoleCalculator
        return BispectrumMultipoleCalculator(config=config, basis=basis).compute(
            self,
            regulator=regulator,
            **params,
        )


class FunctionAngularBispectrum2D(AngularBispectrum2D):
    def __init__(self, func: Callable, support: Optional[Support2D] = None, window: Optional[Callable] = None):
        self.func = func
        self.support = support or Support2D()
        self.window = window

    def evaluate(self, ell1, ell2, ell3, **params):
        val = self.func(ell1, ell2, ell3, **params)
        if self.window is not None:
            val = val * self.window(ell1, ell2, ell3)
        return val
