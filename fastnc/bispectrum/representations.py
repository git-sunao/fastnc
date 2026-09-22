"""Mathematical representations of additive bispectrum terms."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


class BispectrumRepresentation:
    """Marker base class for calculator-ready term representations."""


class BispectrumRepresentation3D(BispectrumRepresentation):
    """Marker for representations defined from ``B(k1, k2, k3, z)``."""


class BispectrumRepresentation2D(BispectrumRepresentation):
    """Marker for representations defined from ``B(ell1, ell2, ell3)``."""


@dataclass(frozen=True)
class NumericExpression3D(BispectrumRepresentation3D):
    """Direct evaluator of one term ``B(k1, k2, k3, z)``."""

    evaluator: Callable

    def __post_init__(self):
        if not callable(self.evaluator):
            raise TypeError("evaluator must be callable")

    def evaluate(self, k1, k2, k3, z, **params):
        return self.evaluator(k1, k2, k3, z, **params)

    __call__ = evaluate


@dataclass(frozen=True)
class NumericExpression2D(BispectrumRepresentation2D):
    """Direct evaluator of one angular term ``B(ell1, ell2, ell3)``."""

    evaluator: Callable

    def __post_init__(self):
        if not callable(self.evaluator):
            raise TypeError("evaluator must be callable")

    def evaluate(self, ell1, ell2, ell3, **params):
        return self.evaluator(ell1, ell2, ell3, **params)

    __call__ = evaluate
