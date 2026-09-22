"""Radial kernels independent of any bispectrum or route API."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline

from .geometry import validate_los_coordinates


@dataclass(frozen=True)
class RadialKernel:
    z: np.ndarray
    chi: np.ndarray
    weight: np.ndarray
    name: str | None = None

    def __post_init__(self):
        z = np.asarray(self.z, dtype=float)
        chi = np.asarray(self.chi, dtype=float)
        weight = np.asarray(self.weight, dtype=float)
        if z.shape != chi.shape or chi.shape != weight.shape:
            raise ValueError("z, chi, and weight must have the same shape")
        if np.any(np.diff(chi) <= 0.0):
            order = np.argsort(chi)
            z, chi, weight = z[order], chi[order], weight[order]
        z, chi = validate_los_coordinates(z, chi)
        if np.any(~np.isfinite(weight)):
            raise ValueError("weight must be finite")
        object.__setattr__(self, "z", z)
        object.__setattr__(self, "chi", chi)
        object.__setattr__(self, "weight", weight)

    def evaluate(self, chi):
        order = min(3, self.chi.size - 1)
        spline = InterpolatedUnivariateSpline(
            self.chi,
            self.weight,
            k=order,
            ext=1,
        )
        return spline(chi)

    def integral(self):
        return np.trapezoid(self.weight, self.chi)

    def normalized(self, *, integral: float = 1.0):
        current = self.integral()
        if current == 0.0:
            raise ValueError("cannot normalize a kernel with zero integral")
        return type(self)(
            z=self.z,
            chi=self.chi,
            weight=self.weight * (float(integral) / current),
            name=self.name,
        )


class KernelSet:
    """Named radial kernels evaluated on a common requested LOS grid."""

    def __init__(self, kernels: Mapping[str, RadialKernel]):
        self._kernels = dict(kernels)
        if not self._kernels:
            raise ValueError("KernelSet requires at least one radial kernel")
        if not all(isinstance(kernel, RadialKernel) for kernel in self._kernels.values()):
            raise TypeError("KernelSet values must be RadialKernel objects")

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._kernels)

    def product(self, names, chi):
        chi = np.asarray(chi, dtype=float)
        result = np.ones_like(chi)
        for name in tuple(names):
            try:
                kernel = self._kernels[name]
            except KeyError as exc:
                raise KeyError(f"unknown radial kernel {name!r}") from exc
            result *= kernel.evaluate(chi)
        return result
