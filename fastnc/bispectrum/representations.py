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


class SemiAnalyticRepresentation3D(BispectrumRepresentation3D):
    """Capability marker for coefficient-factorized 3D expressions."""


class SemiAnalyticRepresentation2D(BispectrumRepresentation2D):
    """Capability marker for native or projected semi-analytic expressions."""


@dataclass(frozen=True)
class SemiAnalyticLowRankProductExpression3D(SemiAnalyticRepresentation3D):
    r"""Low-rank three-leg product with shape-dependent amplitudes.

    The represented approximation is
    ``prod_i sum_a A_a(r1, r2) V_a(k_i, z)``.  ``A_a`` depends only on the
    triangle shape, while all source-state dependence is confined to the
    one-dimensional radial profiles ``V_a``.  The representation owns the
    trained shape basis; route calculators own angular and LOS quadrature.
    """

    rank: int
    amplitude_evaluator: Callable
    radial_evaluator: Callable
    trained_basis: str

    def __post_init__(self):
        if int(self.rank) < 1:
            raise ValueError("rank must be positive")
        if not callable(self.amplitude_evaluator):
            raise TypeError("amplitude_evaluator must be callable")
        if not callable(self.radial_evaluator):
            raise TypeError("radial_evaluator must be callable")
        if not str(self.trained_basis):
            raise ValueError("trained_basis must not be empty")
        object.__setattr__(self, "rank", int(self.rank))
        object.__setattr__(self, "trained_basis", str(self.trained_basis))

    def amplitudes(self, k1, k2, k3):
        """Return ``A_a`` with the low-rank index on the first axis."""
        k1, k2, k3 = np.broadcast_arrays(k1, k2, k3)
        values = np.asarray(self.amplitude_evaluator(k1, k2, k3), dtype=float)
        expected = (self.rank,) + k1.shape
        if values.shape != expected:
            raise ValueError(
                f"amplitude_evaluator must return shape {expected}, got {values.shape}"
            )
        return values

    def radial_profiles(self, k, z):
        """Return ``V_a(k,z)`` with the low-rank index on the first axis."""
        k, z = np.broadcast_arrays(k, z)
        values = np.asarray(self.radial_evaluator(k, z), dtype=float)
        expected = (self.rank,) + k.shape
        if values.shape != expected:
            raise ValueError(
                f"radial_evaluator must return shape {expected}, got {values.shape}"
            )
        return values

    def evaluate(self, k1, k2, k3, z):
        """Evaluate the low-rank approximation on a closed triangle."""
        k1, k2, k3, z = np.broadcast_arrays(k1, k2, k3, z)
        amplitudes = self.amplitudes(k1, k2, k3)
        legs = (
            np.sum(amplitudes * self.radial_profiles(k1, z), axis=0),
            np.sum(amplitudes * self.radial_profiles(k2, z), axis=0),
            np.sum(amplitudes * self.radial_profiles(k3, z), axis=0),
        )
        return legs[0] * legs[1] * legs[2]

    __call__ = evaluate


@dataclass(frozen=True)
class SemiAnalyticExpression3D(SemiAnalyticRepresentation3D):
    r"""Factorized expression used by the Appendix-C route.

    The represented term is
    ``U(k2/k, k3/k) V(k2, k3, z) (k1/k)**p W(k1, z)``, where
    ``k = sqrt(k2**2 + k3**2)`` and
    ``W(k1, z) = sum_n w_n(z) k1**nu_n``.  The calculator, rather than this
    passive object, performs the angular contraction and LOS projection.
    """

    exponents: np.ndarray
    coefficient_evaluator: Callable
    u_evaluator: Callable
    v_evaluator: Callable
    power: float = 0.0

    def __post_init__(self):
        exponents = np.asarray(self.exponents, dtype=complex)
        if exponents.ndim != 1 or exponents.size == 0:
            raise ValueError("exponents must be a non-empty one-dimensional array")
        if np.any(~np.isfinite(exponents)):
            raise ValueError("exponents must be finite")
        for name in ("coefficient_evaluator", "u_evaluator", "v_evaluator"):
            if not callable(getattr(self, name)):
                raise TypeError(f"{name} must be callable")
        exponents = np.array(exponents, copy=True)
        exponents.setflags(write=False)
        object.__setattr__(self, "exponents", exponents)
        object.__setattr__(self, "power", float(self.power))

    def coefficients(self, z) -> np.ndarray:
        """Return ``w_n(z)`` with the Mellin index on the final axis."""
        z = np.asarray(z, dtype=float)
        values = np.asarray(self.coefficient_evaluator(z), dtype=complex)
        expected = z.shape + (self.exponents.size,)
        try:
            values = np.broadcast_to(values, expected)
        except ValueError as exc:
            raise ValueError(
                f"coefficient_evaluator output must broadcast to {expected}"
            ) from exc
        if np.any(~np.isfinite(values)):
            raise ValueError("coefficient_evaluator returned non-finite values")
        return values

    def evaluate_u(self, ratio2, ratio3):
        return np.asarray(self.u_evaluator(ratio2, ratio3))

    def evaluate_v(self, k2, k3, z):
        return np.asarray(self.v_evaluator(k2, k3, z))


@dataclass(frozen=True)
class SemiAnalyticRadialExpression3D(SemiAnalyticRepresentation3D):
    r"""Grid-free declaration of the separable template in Appendix C.

    The represented term is
    ``U(k2/k, k3/k) V(k2, k3, z) (k1/k)**p W(k1, z) exp(i m phi23)``,
    where ``k = sqrt(k2**2 + k3**2)``.  The semi-analytic calculator owns
    the FFTLog expansion of ``W``, coefficient-level LOS projection, and
    contraction with the universal angular kernels.  This object owns no
    Mellin grid or projection grid.
    """

    u_evaluator: Callable
    v_evaluator: Callable
    w_evaluator: Callable
    power: float = 0.0
    angular_order: int = 0

    def __post_init__(self):
        for name in ("u_evaluator", "v_evaluator", "w_evaluator"):
            if not callable(getattr(self, name)):
                raise TypeError(f"{name} must be callable")
        object.__setattr__(self, "power", float(self.power))
        object.__setattr__(self, "angular_order", int(self.angular_order))

    def evaluate_u(self, ratio2, ratio3):
        return np.asarray(self.u_evaluator(ratio2, ratio3))

    def evaluate_v(self, k2, k3, z):
        return np.asarray(self.v_evaluator(k2, k3, z))

    def evaluate_w(self, k1, z):
        return np.asarray(self.w_evaluator(k1, z))

    def evaluate(self, k1, k2, k3, z):
        """Evaluate the represented term on a closed 3D triangle."""
        k1, k2, k3, z = np.broadcast_arrays(
            np.asarray(k1, dtype=float),
            np.asarray(k2, dtype=float),
            np.asarray(k3, dtype=float),
            np.asarray(z, dtype=float),
        )
        scale = np.sqrt(k2**2 + k3**2)
        with np.errstate(divide="ignore", invalid="ignore"):
            cosine = (k1**2 - k2**2 - k3**2) / (2.0 * k2 * k3)
            angle = np.arccos(np.clip(cosine, -1.0, 1.0))
            return (
                self.evaluate_u(k2 / scale, k3 / scale)
                * self.evaluate_v(k2, k3, z)
                * np.power(k1 / scale, self.power)
                * self.evaluate_w(k1, z)
                * np.exp(1j * self.angular_order * angle)
            )

    __call__ = evaluate


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
