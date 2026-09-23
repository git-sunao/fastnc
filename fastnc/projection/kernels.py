"""Radial kernels independent of any bispectrum or route API."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline

from .geometry import validate_los_coordinates


@dataclass(frozen=True)
class Kernel1D:
    """One sampled line-of-sight kernel on a ``(z, chi)`` grid."""

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

    def copy(self, *, weight=None, name=None):
        return type(self)(
            z=self.z.copy(),
            chi=self.chi.copy(),
            weight=(
                self.weight.copy()
                if weight is None
                else np.asarray(weight, dtype=float)
            ),
            name=self.name if name is None else name,
        )

    def interpolator(self):
        return InterpolatedUnivariateSpline(
            self.chi,
            self.weight,
            k=min(3, self.chi.size - 1),
            ext=1,
        )

    def evaluate(self, chi):
        return self.interpolator()(chi)

    def resample(self, z, chi, *, name=None):
        z = np.asarray(z, dtype=float)
        chi = np.asarray(chi, dtype=float)
        if z.shape != chi.shape:
            raise ValueError("z and chi must have the same shape")
        return type(self)(
            z=z,
            chi=chi,
            weight=self.evaluate(chi),
            name=self.name if name is None else name,
        )

    def resample_like(self, other: "Kernel1D", *, name=None):
        if not isinstance(other, Kernel1D):
            raise TypeError("other must be a Kernel1D")
        return self.resample(other.z, other.chi, name=name)

    def integral(self):
        return np.trapezoid(self.weight, self.chi)

    def normalized(self, *, integral: float = 1.0, name=None):
        current = self.integral()
        if current == 0.0:
            raise ValueError("cannot normalize a kernel with zero integral")
        return self.copy(
            weight=self.weight * (float(integral) / current),
            name=self.name if name is None else name,
        )

    def _binary_kernel_op(self, other, operation, symbol: str):
        if np.isscalar(other):
            return self.copy(
                weight=operation(self.weight, float(other)),
                name=self.name,
            )
        if not isinstance(other, Kernel1D):
            return NotImplemented
        other_weight = (
            other.weight
            if np.array_equal(self.chi, other.chi)
            else other.evaluate(self.chi)
        )
        name = None
        if self.name is not None or other.name is not None:
            name = f"({self.name or 'kernel'}{symbol}{other.name or 'kernel'})"
        return self.copy(
            weight=operation(self.weight, other_weight),
            name=name,
        )

    def __add__(self, other):
        return self._binary_kernel_op(other, np.add, "+")

    __radd__ = __add__

    def __sub__(self, other):
        return self._binary_kernel_op(other, np.subtract, "-")

    def __rsub__(self, other):
        if np.isscalar(other):
            return self.copy(weight=float(other) - self.weight)
        if isinstance(other, Kernel1D):
            return other - self
        return NotImplemented

    def __mul__(self, other):
        return self._binary_kernel_op(other, np.multiply, "*")

    __rmul__ = __mul__

    def __truediv__(self, other):
        return self._binary_kernel_op(other, np.divide, "/")

    def __neg__(self):
        return self.copy(
            weight=-self.weight,
            name=None if self.name is None else f"(-{self.name})",
        )

    @staticmethod
    def _validate_z_grid(z, name: str):
        z = np.asarray(z, dtype=float)
        if z.ndim != 1 or z.size < 2:
            raise ValueError(
                f"{name} must be one-dimensional with at least two samples"
            )
        if np.any(~np.isfinite(z)) or np.any(np.diff(z) <= 0.0):
            raise ValueError(f"{name} must be finite and strictly increasing")
        return z

    @classmethod
    def _normalized_nz(cls, z_nz, nz):
        z_nz = cls._validate_z_grid(z_nz, "z_nz")
        nz = np.asarray(nz, dtype=float)
        if z_nz.shape != nz.shape:
            raise ValueError("z_nz and nz must have the same shape")
        if np.any(~np.isfinite(nz)):
            raise ValueError("nz must be finite")
        norm = np.trapezoid(nz, z_nz)
        if norm <= 0.0:
            raise ValueError("nz must have a positive integral")
        return nz / norm

    @staticmethod
    def _parse_nz_args(z_kernel, args, z_nz, method_name: str):
        if len(args) == 1:
            nz = args[0]
            if z_nz is None:
                z_nz = z_kernel
        elif len(args) == 2:
            if z_nz is not None:
                raise TypeError(
                    f"{method_name} received positional and keyword z_nz"
                )
            z_nz, nz = args
        else:
            raise TypeError(
                f"{method_name} expects (z, chi, nz) or (z, chi, z_nz, nz)"
            )
        return np.asarray(z_nz, dtype=float), np.asarray(nz, dtype=float)

    @classmethod
    def _nz_on_kernel_grid(cls, z_kernel, z_nz, nz):
        z_nz = cls._validate_z_grid(z_nz, "z_nz")
        if z_nz.shape != nz.shape:
            raise ValueError("z_nz and nz must have the same shape")
        spline = InterpolatedUnivariateSpline(
            z_nz,
            nz,
            k=min(3, z_nz.size - 1),
            ext=1,
        )
        return spline(z_kernel)

    @classmethod
    def from_nz(
        cls,
        z,
        chi,
        *args,
        z_nz=None,
        normalize: bool = True,
        name: str | None = None,
    ):
        """Convert a source distribution n(z) into n(chi)."""
        z = cls._validate_z_grid(z, "z_kernel")
        chi = np.asarray(chi, dtype=float)
        z_nz, nz = cls._parse_nz_args(z, args, z_nz, "from_nz")
        if normalize:
            nz = cls._normalized_nz(z_nz, nz)
        nz_on_kernel = cls._nz_on_kernel_grid(z, z_nz, nz)
        n_chi = nz_on_kernel * np.gradient(z, chi, edge_order=1)
        return cls(z=z, chi=chi, weight=n_chi, name=name)

    @classmethod
    def lensing_from_nz(
        cls,
        z,
        chi,
        *args,
        z_nz=None,
        omega_m: float = 0.3,
        h0_over_c: float = 100.0 / 299792.458,
        normalize_nz: bool = True,
        name: str | None = None,
    ):
        """Construct the weak-lensing efficiency from a source n(z)."""
        z = cls._validate_z_grid(z, "z_kernel")
        chi = np.asarray(chi, dtype=float)
        z_nz, nz = cls._parse_nz_args(z, args, z_nz, "lensing_from_nz")
        if normalize_nz:
            nz = cls._normalized_nz(z_nz, nz)
        if z.shape != chi.shape or np.any(chi <= 0.0):
            raise ValueError("z and positive chi must have the same shape")
        chi_s = InterpolatedUnivariateSpline(
            z,
            chi,
            k=min(3, z.size - 1),
            ext=1,
        )(z_nz)
        valid = chi_s > 0.0
        chi_s = chi_s[valid]
        z_src = z_nz[valid]
        nz_src = nz[valid]
        if chi_s.size < 2:
            raise ValueError("too few source samples inside the kernel z range")
        geometry = (
            np.maximum(chi_s[None, :] - chi[:, None], 0.0)
            / chi_s[None, :]
        )
        source_integral = np.trapezoid(
            nz_src[None, :] * geometry,
            z_src,
            axis=1,
        )
        prefactor = 1.5 * float(omega_m) * float(h0_over_c) ** 2
        weight = prefactor * chi * (1.0 + z) * source_integral
        return cls(z=z, chi=chi, weight=weight, name=name)

    @classmethod
    def nla_from_nz(
        cls,
        z,
        chi,
        *args,
        z_nz=None,
        amplitude: float = 1.0,
        omega_m: float = 0.3,
        c1rho_crit: float = 0.0134,
        growth=None,
        eta: float = 0.0,
        z0: float = 0.62,
        normalize_nz: bool = True,
        name: str | None = None,
    ):
        """Construct an NLA intrinsic-alignment kernel from a source n(z)."""
        z = cls._validate_z_grid(z, "z_kernel")
        chi = np.asarray(chi, dtype=float)
        z_nz, nz = cls._parse_nz_args(z, args, z_nz, "nla_from_nz")
        if normalize_nz:
            nz = cls._normalized_nz(z_nz, nz)
        nz_on_kernel = cls._nz_on_kernel_grid(z, z_nz, nz)
        n_chi = nz_on_kernel * np.gradient(z, chi, edge_order=1)
        if growth is None:
            growth_values = np.ones_like(z)
        elif callable(growth):
            growth_values = np.asarray(growth(z), dtype=float)
        else:
            growth_values = np.asarray(growth, dtype=float)
        if growth_values.shape != z.shape or np.any(growth_values == 0.0):
            raise ValueError("growth must be non-zero and match z")
        scaling = ((1.0 + z) / (1.0 + float(z0))) ** float(eta)
        weight = (
            -float(amplitude)
            * float(c1rho_crit)
            * float(omega_m)
            * scaling
            * n_chi
            / growth_values
        )
        return cls(z=z, chi=chi, weight=weight, name=name)

    @classmethod
    def lensing_plus_nla(
        cls,
        z,
        chi,
        *args,
        z_nz=None,
        name: str | None = None,
        **kwargs,
    ):
        """Construct the sum of lensing and NLA kernels."""
        lensing_kwargs = dict(kwargs.pop("lensing", {}))
        nla_kwargs = dict(kwargs.pop("nla", {}))
        if kwargs:
            raise TypeError(f"unexpected keyword(s): {', '.join(kwargs)}")
        z_nz, nz = cls._parse_nz_args(
            np.asarray(z, dtype=float),
            args,
            z_nz,
            "lensing_plus_nla",
        )
        lensing = cls.lensing_from_nz(
            z,
            chi,
            z_nz,
            nz,
            **lensing_kwargs,
        )
        nla = cls.nla_from_nz(z, chi, z_nz, nz, **nla_kwargs)
        return (lensing + nla).copy(name=name)

class KernelSet:
    """Named radial kernels evaluated on a common requested LOS grid."""

    def __init__(self, kernels: Mapping[str, Kernel1D]):
        self._kernels = dict(kernels)
        if not self._kernels:
            raise ValueError("KernelSet requires at least one radial kernel")
        if not all(isinstance(kernel, Kernel1D) for kernel in self._kernels.values()):
            raise TypeError("KernelSet values must be Kernel1D objects")

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._kernels)

    def __getitem__(self, name: str) -> Kernel1D:
        return self._kernels[name]

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
