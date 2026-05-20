"""Concrete model wrappers."""
from __future__ import annotations

from .base import Bispectrum3D
from .support import Support3D


class ExternalBispectrum3D(Bispectrum3D):
    """Adapter for external 3D bispectrum modules.

    The wrapped object must either be callable as ``obj(k1,k2,k3,z,**params)``
    or provide one of ``evaluate`` or ``get_bihalofit``.
    """
    def __init__(self, external, support: Support3D | None = None, method_name: str | None = None):
        self.external = external
        self.support = support or Support3D(policy="ignore")
        self.method_name = method_name

    def evaluate(self, k1, k2, k3, z, **params):
        if self.method_name is not None:
            return getattr(self.external, self.method_name)(k1, k2, k3, z, **params)
        if hasattr(self.external, "evaluate"):
            return self.external.evaluate(k1, k2, k3, z, **params)
        if hasattr(self.external, "get_bihalofit"):
            return self.external.get_bihalofit(k1, k2, k3, z, **params)
        return self.external(k1, k2, k3, z, **params)
