"""Base bispectrum objects."""
from __future__ import annotations

from dataclasses import dataclass
from functools import wraps
from typing import Callable, Optional
import numpy as np

from .support import Support3D, Support2D

class _BispectrumBase:
    """Mixin providing persistent default keyword arguments for evaluation.

    Defaults are stored per instance.  Runtime keyword arguments always take
    precedence and do not mutate the stored defaults.  ``__new__`` is used so
    subclasses do not need to call ``super().__init__()`` to get an independent
    empty dictionary.
    """

    def __new__(cls, *args, **kwargs):
        obj = super().__new__(cls)
        obj.default_kwargs = {}
        return obj

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        evaluate = cls.__dict__.get("evaluate")
        if evaluate is None or getattr(evaluate, "_fastnc_default_kwargs_wrapped", False):
            return

        @wraps(evaluate)
        def wrapped_evaluate(self, *args, **params):
            return evaluate(self, *args, **self._merge_default_kwargs(params))

        wrapped_evaluate._fastnc_default_kwargs_wrapped = True
        cls.evaluate = wrapped_evaluate

    def set_default_kwargs(self, **kwargs):
        """Set/update default keyword arguments used by :meth:`evaluate`.

        Calling ``evaluate(..., key=value)`` overrides a stored default for that
        call only.  The stored defaults themselves are left unchanged.
        """
        self.default_kwargs.update(kwargs)
        return self

    def clear_default_kwargs(self):
        """Remove all stored evaluation defaults and return ``self``."""
        self.default_kwargs.clear()
        return self

    def _merge_default_kwargs(self, kwargs):
        params = dict(self.default_kwargs)
        params.update(kwargs)
        return params


class Bispectrum3D(_BispectrumBase):
    """Base object for a 3D bispectrum ``B(k1,k2,k3,z)``.

    Concrete models may expose physical contributions through the route-
    capability collections :meth:`generic_terms`, :meth:`slepian_terms`, and
    :meth:`analytic_terms`.  Empty tuples are the default so legacy models are
    unchanged until they are explicitly converted to terms.
    """
    support = Support3D()
    supports_slepian = False
    slepian_los_kind = None

    def __call__(self, k1, k2, k3, z, **params):
        return self.evaluate(k1, k2, k3, z, **params)

    def evaluate(self, k1, k2, k3, z, **params):
        raise NotImplementedError

    def generic_terms(self, **params):
        """Return physical terms assigned to the generic numerical route.

        Models are converted incrementally.  An empty tuple therefore means
        "no term representation exposed yet", not that the physical
        bispectrum itself vanishes.
        """
        return ()

    def slepian_terms(self, **params):
        """Return terms supporting the future direct Slepian 3PCF route."""
        return ()

    def analytic_terms(self, **params):
        """Return terms for the future optimized regular analytic route."""
        return ()

    def interpolate(self, config, **params):
        from .interpolate import InterpolatedBispectrum3D
        return InterpolatedBispectrum3D.from_bispectrum(self, config, **params)


class Bispectrum2D(_BispectrumBase):
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

