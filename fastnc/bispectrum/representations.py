"""Mathematical representations of additive bispectrum terms."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


class BispectrumRepresentation:
    """Marker base class for calculator-ready term representations."""


class BispectrumRepresentation3D(BispectrumRepresentation):
    """Marker for representations defined from ``B(k1, k2, k3, z)``."""


class BispectrumRepresentation2D(BispectrumRepresentation):
    """Marker for representations defined from ``B(ell1, ell2, ell3)``."""


class NumericRepresentation3D(BispectrumRepresentation3D):
    """Capability marker for directly evaluable numeric 3D representations."""


class NumericRepresentation2D(BispectrumRepresentation2D):
    """Capability marker for directly evaluable numeric 2D representations."""


class SlepianRepresentation3D(BispectrumRepresentation3D):
    """Capability marker for separable 3D Slepian expressions."""


class SlepianRepresentation2D(BispectrumRepresentation2D):
    """Capability marker for separable native-2D Slepian expressions."""


@dataclass(frozen=True)
class SlepianRadialFactor3D:
    """One radial factor ``f(k, z)`` of a separable 3D bispectrum term."""

    evaluator: Callable | None = None
    is_constant: bool = False

    def __post_init__(self):
        if self.is_constant:
            if self.evaluator is not None:
                raise ValueError(
                    "a constant radial factor must not define an evaluator"
                )
        elif not callable(self.evaluator):
            raise TypeError(
                "a non-constant radial factor requires a callable evaluator"
            )

    @classmethod
    def constant(cls) -> "SlepianRadialFactor3D":
        return cls(is_constant=True)

    def evaluate(self, k, z):
        k = np.asarray(k, dtype=float)
        if self.is_constant:
            return np.ones_like(k)
        values = np.asarray(self.evaluator(k, z))
        if values.shape != k.shape:
            values = np.broadcast_to(values, k.shape)
        if np.any(~np.isfinite(values)):
            raise ValueError("radial factor returned non-finite values")
        return values

    __call__ = evaluate


@dataclass(frozen=True)
class SlepianExpression3D(SlepianRepresentation3D):
    """Separable expression ``C(z) product_i f_i(k_i,z) exp(i n_i phi_i)``."""

    coefficient: complex | Callable
    radial_factors: tuple[SlepianRadialFactor3D, ...]
    angular_orders: tuple[int, int, int] = (0, 0, 0)

    def __post_init__(self):
        factors = tuple(self.radial_factors)
        orders = tuple(int(order) for order in self.angular_orders)
        if len(factors) != 3:
            raise ValueError("radial_factors must contain exactly three factors")
        if not all(
            isinstance(factor, SlepianRadialFactor3D) for factor in factors
        ):
            raise TypeError(
                "radial_factors must contain SlepianRadialFactor3D objects"
            )
        if len(orders) != 3:
            raise ValueError("angular_orders must contain exactly three entries")
        if sum(orders) != 0:
            raise ValueError("angular_orders must sum to zero")
        coefficient = self.coefficient
        if not callable(coefficient):
            if not np.isfinite(coefficient):
                raise ValueError("coefficient must be finite")
            coefficient = complex(coefficient)
        object.__setattr__(self, "coefficient", coefficient)
        object.__setattr__(self, "radial_factors", factors)
        object.__setattr__(self, "angular_orders", orders)

    @property
    def constant_legs(self) -> tuple[int, ...]:
        return tuple(
            index
            for index, factor in enumerate(self.radial_factors)
            if factor.is_constant
        )

    def coefficient_at(self, z) -> complex:
        value = (
            self.coefficient(z)
            if callable(self.coefficient)
            else self.coefficient
        )
        value = np.asarray(value)
        if value.ndim != 0 or not np.isfinite(value):
            raise ValueError("coefficient must evaluate to one finite scalar")
        return complex(value)


@dataclass(frozen=True)
class SlepianRadialFactor2D:
    """One radial factor of a separable native-2D bispectrum term."""

    evaluator: Callable | None = None
    is_constant: bool = False

    def __post_init__(self):
        if self.is_constant:
            if self.evaluator is not None:
                raise ValueError("a constant radial factor must not define an evaluator")
        elif not callable(self.evaluator):
            raise TypeError("a non-constant radial factor requires a callable evaluator")

    @classmethod
    def constant(cls) -> "SlepianRadialFactor2D":
        return cls(is_constant=True)

    def evaluate(self, ell):
        ell = np.asarray(ell, dtype=float)
        if self.is_constant:
            return np.ones_like(ell)
        values = np.asarray(self.evaluator(ell))
        if values.shape != ell.shape:
            values = np.broadcast_to(values, ell.shape)
        if np.any(~np.isfinite(values)):
            raise ValueError("radial factor returned non-finite values")
        return values

    __call__ = evaluate


@dataclass(frozen=True)
class SlepianExpression2D(SlepianRepresentation2D):
    """Separable expression ``C product_i f_i(ell_i) exp(i n_i phi_i)``."""

    coefficient: complex
    radial_factors: tuple[SlepianRadialFactor2D, ...]
    angular_orders: tuple[int, int, int] = (0, 0, 0)

    def __post_init__(self):
        factors = tuple(self.radial_factors)
        orders = tuple(int(order) for order in self.angular_orders)
        if len(factors) != 3:
            raise ValueError("radial_factors must contain exactly three factors")
        if not all(isinstance(factor, SlepianRadialFactor2D) for factor in factors):
            raise TypeError("radial_factors must contain SlepianRadialFactor2D objects")
        if len(orders) != 3:
            raise ValueError("angular_orders must contain exactly three entries")
        if sum(orders) != 0:
            raise ValueError("angular_orders must sum to zero")
        if not np.isfinite(self.coefficient):
            raise ValueError("coefficient must be finite")
        object.__setattr__(self, "coefficient", complex(self.coefficient))
        object.__setattr__(self, "radial_factors", factors)
        object.__setattr__(self, "angular_orders", orders)

    @property
    def constant_legs(self) -> tuple[int, ...]:
        return tuple(
            index for index, factor in enumerate(self.radial_factors)
            if factor.is_constant
        )


@dataclass(frozen=True)
class NumericExpression3D(NumericRepresentation3D):
    """Direct evaluator of one term ``B(k1, k2, k3, z)``."""

    evaluator: Callable

    def __post_init__(self):
        if not callable(self.evaluator):
            raise TypeError("evaluator must be callable")

    def evaluate(self, k1, k2, k3, z, **params):
        return self.evaluator(k1, k2, k3, z, **params)

    __call__ = evaluate


@dataclass(frozen=True)
class NumericExpression2D(NumericRepresentation2D):
    """Direct evaluator of one angular term ``B(ell1, ell2, ell3)``."""

    evaluator: Callable

    def __post_init__(self):
        if not callable(self.evaluator):
            raise TypeError("evaluator must be callable")

    def evaluate(self, ell1, ell2, ell3, **params):
        return self.evaluator(ell1, ell2, ell3, **params)

    __call__ = evaluate
