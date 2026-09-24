"""Route-independent lazy evaluation of angular bispectrum multipoles."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class BispectrumMultipole:
    """Lazily evaluated bispectrum multipoles with a route-independent API."""

    bispectrum: Any
    calculator: Any

    @classmethod
    def from_numeric(
        cls,
        config,
        b2d,
    ) -> "BispectrumMultipole":
        """Construct lazily evaluated numeric multipoles."""
        from .numeric import NumericBispectrumMultipoleCalculator

        calculator = NumericBispectrumMultipoleCalculator(config)
        return cls(bispectrum=b2d, calculator=calculator)

    @property
    def basis(self) -> str:
        """Basis convention used by the retained calculator."""
        return self.calculator.config.basis

    def evaluate(self, mode, ell2, ell3, **params) -> np.ndarray:
        """Evaluate one mode or a one-dimensional collection of modes."""
        return self.calculator.evaluate(
            self.bispectrum,
            mode,
            ell2=ell2,
            ell3=ell3,
            **params,
        )

    def evaluate_fourier(self, mode, ell2, ell3, **params) -> np.ndarray:
        """Evaluate canonical full-Fourier coefficients for integer modes."""
        requested = np.asarray(mode)
        scalar_mode = requested.ndim == 0
        if requested.ndim > 1 or requested.size == 0:
            raise ValueError("mode must be a scalar or non-empty 1D array")
        if not np.issubdtype(requested.dtype, np.integer):
            raise TypeError("mode values must be integers")
        modes = np.atleast_1d(requested).astype(int, copy=False)

        if self.basis == "fourier":
            values = self.evaluate(modes, ell2, ell3, **params)
        elif self.basis == "cosine":
            values = self.evaluate(np.abs(modes), ell2, ell3, **params)
            weights = np.where(modes == 0, 1.0, 0.5)
            values = values * weights.reshape((-1,) + (1,) * (values.ndim - 1))
        elif self.basis == "sine":
            shape = np.broadcast_shapes(np.shape(ell2), np.shape(ell3))
            values = np.zeros((modes.size,) + shape, dtype=complex)
            nonzero = modes != 0
            if np.any(nonzero):
                sine = self.evaluate(
                    np.abs(modes[nonzero]), ell2, ell3, **params
                )
                weights = -0.5j * np.sign(modes[nonzero])
                values[nonzero] = sine * weights.reshape(
                    (-1,) + (1,) * (sine.ndim - 1)
                )
        else:
            raise ValueError(f"unsupported multipole basis: {self.basis!r}")

        return values[0] if scalar_mode else values
