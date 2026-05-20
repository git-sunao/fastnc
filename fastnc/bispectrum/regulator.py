"""Optional multiplicative windows/regulators for bispectrum evaluations."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Ell3HighPassRegulator:
    """Smooth high-pass regulator in the third side ``ell3``.

    The regulator is

        R(ell3) = [ell3^s / (ell3^s + ell_cut^s)]^(gamma/s).

    It is intended as an optional numerical/physical window.  It changes the
    evaluated bispectrum and should not be confused with a squeezed-safe branch,
    which evaluates the same model using a stable limiting expression.
    """

    ell_cut: float
    gamma: float = 1.0
    smooth: float = 4.0

    def __post_init__(self):
        if self.ell_cut <= 0.0:
            raise ValueError("ell_cut must be positive")
        if self.gamma <= 0.0:
            raise ValueError("gamma must be positive")
        if self.smooth <= 0.0:
            raise ValueError("smooth must be positive")

    def __call__(self, ell1, ell2, ell3):
        ell3 = np.asarray(ell3, dtype=float)
        x = (ell3 / self.ell_cut) ** self.smooth
        return (x / (1.0 + x)) ** (self.gamma / self.smooth)
