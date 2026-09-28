"""Native-2D Slepian and Weber-Schafheitlin route kernels."""
from __future__ import annotations

from dataclasses import dataclass
import logging
import time
from numbers import Number

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.interpolate import CubicSpline
from scipy.special import jv, loggamma, rgamma

from fastnc.bispectrum import (
    Bispectrum2D,
    BispectrumTerm2D,
    SlepianExpression2D,
    SlepianExpression3D,
    SlepianRadialFactor2D,
    SlepianRepresentation2D,
)
from fastnc.projection import ProjectedSlepianRepresentation2D

from .config import SlepianConfig
from .conventions import as_effective_spin_triple


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FFTLogPowerSum:
    """Finite complex-power representation of one sampled radial factor."""

    coefficients: np.ndarray
    exponents: np.ndarray


@dataclass(frozen=True)
class _SlepianLegLayout:
    double_leg: int
    single_order: int
    double_orders: tuple[int, int]
    constant_orders: tuple[int, int]
    transpose_output: bool


def _slepian_leg_layout(
    expression,
    m: int,
    n: int,
    sigma: tuple[int, int, int] = (0, 0, 0),
) -> _SlepianLegLayout:
    sigma1, sigma2, sigma3 = as_effective_spin_triple(sigma).sigma
    constant_legs = expression.constant_legs
    if constant_legs == (0,):
        raise NotImplementedError(
            "the Slepian route cannot eliminate physical leg 1 because "
            "ZetaK retains the angle opposite leg 1"
        )
    if constant_legs == (2,):
        n1, n2, n3 = expression.angular_orders
        return _SlepianLegLayout(
            double_leg=1,
            single_order=n1 + sigma1,
            double_orders=(n2 + sigma2 - m, m),
            constant_orders=(n3 + sigma3 - n, n),
            transpose_output=False,
        )
    if constant_legs == (1,):
        n1, n2, n3 = expression.angular_orders
        return _SlepianLegLayout(
            double_leg=2,
            single_order=n1 + sigma1,
            double_orders=(n3 + sigma3 - n, n),
            constant_orders=(n2 + sigma2 - m, m),
            transpose_output=True,
        )
    raise NotImplementedError(
        "the Slepian route requires exactly physical leg 2 or leg 3 "
        "to be constant"
    )


@dataclass(frozen=True)
class WeberGeometry:
    """Scale and unique-ratio geometry for a two-Bessel kernel grid."""

    x: np.ndarray
    theta: np.ndarray
    scale: np.ndarray
    ratio: np.ndarray
    diagonal: np.ndarray

    @classmethod
    def from_coordinates(cls, x, theta) -> "WeberGeometry":
        x = np.asarray(x, dtype=float)
        theta = np.asarray(theta, dtype=float)
        if x.ndim != 1 or theta.ndim != 1:
            raise ValueError("x and theta must be one-dimensional")
        if np.any(x <= 0.0) or np.any(theta <= 0.0):
            raise ValueError("x and theta must be positive")
        xx = x[:, None]
        tt = theta[None, :]
        scale = np.maximum(xx, tt)
        ratio = np.minimum(xx, tt) / scale
        diagonal = np.isclose(xx, tt, rtol=1.0e-13, atol=0.0)
        return cls(x, theta, scale, ratio, diagonal)

    def unique_ratios(self, *, omit_diagonal: bool, max_ratio: float = 1.0):
        """Return unique log-ratios and indices reconstructing the full grid."""
        active = ~self.diagonal if omit_diagonal else np.ones_like(
            self.diagonal, dtype=bool
        )
        active &= self.ratio < float(max_ratio)
        # Log-grid ratios that differ only by roundoff represent the same
        # scale separation. Quantization avoids duplicate hypergeometric calls.
        log_ratio = np.round(np.log(self.ratio[active]), decimals=14)
        unique_log_ratio, inverse = np.unique(log_ratio, return_inverse=True)
        return np.exp(unique_log_ratio), active, inverse


@dataclass(frozen=True)
class ConstantLegKernel:
    """Distributional contact and regular pieces of a constant-leg kernel."""

    contact_coefficient: int
    regular_less: np.ndarray
    regular_greater: np.ndarray
    has_regular: bool


@dataclass(frozen=True)
class RegularMellinMatrix:
    """Passive full Mellin matrix for one regular constant-leg contraction."""

    values: np.ndarray
    double_exponents: np.ndarray
    single_exponents: np.ndarray
    x: np.ndarray
    theta: np.ndarray
    single_order: int
    double_orders: tuple[int, int]
    constant_orders: tuple[int, int]

    def __post_init__(self):
        values = np.asarray(self.values, dtype=complex)
        double_exponents = np.asarray(self.double_exponents, dtype=complex)
        single_exponents = np.asarray(self.single_exponents, dtype=complex)
        x = np.asarray(self.x, dtype=float)
        theta = np.asarray(self.theta, dtype=float)
        expected = (
            double_exponents.size,
            single_exponents.size,
            theta.size,
            theta.size,
        )
        if values.shape != expected:
            raise ValueError(f"values must have shape {expected}")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "double_exponents", double_exponents)
        object.__setattr__(self, "single_exponents", single_exponents)
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "theta", theta)
        object.__setattr__(self, "single_order", int(self.single_order))
        object.__setattr__(
            self, "double_orders", tuple(int(v) for v in self.double_orders)
        )
        object.__setattr__(
            self, "constant_orders", tuple(int(v) for v in self.constant_orders)
        )


@dataclass(frozen=True)
class LowRankRegularMellinMatrix:
    """SVD factors of a regular Mellin matrix at each theta pair."""

    left_vectors: np.ndarray
    singular_values: np.ndarray
    right_vectors: np.ndarray
    double_exponents: np.ndarray
    single_exponents: np.ndarray
    retained_rank: int
    relative_reconstruction_error: float

    def __post_init__(self):
        left = np.asarray(self.left_vectors, dtype=complex)
        singular = np.asarray(self.singular_values, dtype=float)
        right = np.asarray(self.right_vectors, dtype=complex)
        double_exponents = np.asarray(self.double_exponents, dtype=complex)
        single_exponents = np.asarray(self.single_exponents, dtype=complex)
        if left.ndim != 4 or singular.ndim != 3 or right.ndim != 4:
            raise ValueError("low-rank factors have incompatible dimensions")
        n_theta1, n_theta2, n_double, rank = left.shape
        if singular.shape != (n_theta1, n_theta2, rank):
            raise ValueError("singular_values have the wrong shape")
        if right.shape[:3] != (n_theta1, n_theta2, rank):
            raise ValueError("right_vectors have the wrong shape")
        if double_exponents.shape != (n_double,):
            raise ValueError("double_exponents have the wrong shape")
        if single_exponents.shape != (right.shape[3],):
            raise ValueError("single_exponents have the wrong shape")
        object.__setattr__(self, "left_vectors", left)
        object.__setattr__(self, "singular_values", singular)
        object.__setattr__(self, "right_vectors", right)
        object.__setattr__(self, "double_exponents", double_exponents)
        object.__setattr__(self, "single_exponents", single_exponents)
        object.__setattr__(self, "retained_rank", int(self.retained_rank))
        object.__setattr__(
            self,
            "relative_reconstruction_error",
            float(self.relative_reconstruction_error),
        )


@dataclass(frozen=True)
class LOSMellinFactors:
    """Factorized Mellin coefficients sampled on one immutable LOS grid."""

    z: np.ndarray
    chi: np.ndarray
    weight: np.ndarray
    single_coefficients: np.ndarray
    double_coefficients: np.ndarray
    single_exponents: np.ndarray
    double_exponents: np.ndarray

    def __post_init__(self):
        z = np.asarray(self.z, dtype=float)
        chi = np.asarray(self.chi, dtype=float)
        weight = np.asarray(self.weight, dtype=float)
        single = np.asarray(self.single_coefficients, dtype=complex)
        double = np.asarray(self.double_coefficients, dtype=complex)
        single_exponents = np.asarray(self.single_exponents, dtype=complex)
        double_exponents = np.asarray(self.double_exponents, dtype=complex)
        if z.ndim != 1 or z.size < 2:
            raise ValueError("z must contain at least two LOS nodes")
        if z.shape != chi.shape or z.shape != weight.shape:
            raise ValueError("z, chi, and weight must have the same shape")
        if np.any(np.diff(chi) <= 0.0):
            raise ValueError("chi must be strictly increasing")
        if single.shape != (z.size, single_exponents.size):
            raise ValueError("single_coefficients have the wrong shape")
        if double.shape != (z.size, double_exponents.size):
            raise ValueError("double_coefficients have the wrong shape")
        arrays = (
            z,
            chi,
            weight,
            single,
            double,
            single_exponents,
            double_exponents,
        )
        if any(np.any(~np.isfinite(values)) for values in arrays):
            raise ValueError("LOS Mellin data must be finite")
        names = (
            "z",
            "chi",
            "weight",
            "single_coefficients",
            "double_coefficients",
            "single_exponents",
            "double_exponents",
        )
        for name, values in zip(names, arrays):
            values = np.array(values, copy=True)
            values.setflags(write=False)
            object.__setattr__(self, name, values)


def _log_edge_taper(size: int, fraction: float) -> np.ndarray:
    weights = np.ones(int(size))
    edge = int(np.floor(float(fraction) * int(size)))
    if edge == 0:
        return weights
    phase = (np.arange(edge) + 1.0) / (edge + 1.0)
    ramp = 0.5 - 0.5 * np.cos(np.pi * phase)
    weights[:edge] = ramp
    weights[-edge:] = ramp[::-1]
    return weights


def _coefficient_window(size: int, fraction: float) -> np.ndarray:
    if fraction <= 0.0:
        return np.ones(int(size))
    frequency = np.abs(np.fft.fftfreq(int(size)))
    transition = (1.0 - float(fraction)) * np.max(frequency)
    weights = np.ones(int(size))
    mask = frequency > transition
    phase = (frequency[mask] - transition) / (np.max(frequency) - transition)
    weights[mask] = 0.5 + 0.5 * np.cos(np.pi * phase)
    return weights


def fftlog_power_sum(ell, values, config: SlepianConfig) -> FFTLogPowerSum:
    """Expand samples as ``sum_m c_m ell**nu_m`` on a log grid."""
    ell = np.asarray(ell, dtype=float)
    values = np.asarray(values, dtype=complex)
    if ell.ndim != 1 or ell.size < 8 or values.shape != ell.shape:
        raise ValueError("ell and values must be matching 1D arrays with at least 8 samples")
    spacing = np.diff(np.log(ell))
    if np.any(ell <= 0.0) or not np.allclose(
        spacing, spacing[0], rtol=1.0e-10, atol=1.0e-13
    ):
        raise ValueError("ell must be a positive logarithmically spaced grid")
    if np.any(~np.isfinite(values)):
        raise ValueError("radial samples must be finite")

    biased = (
        values
        * _log_edge_taper(ell.size, config.taper_fraction)
        / ell**config.bias
    )
    coefficients = np.fft.fft(biased) / ell.size
    eta = 2.0 * np.pi * np.fft.fftfreq(ell.size, d=spacing[0])
    coefficients *= _coefficient_window(ell.size, config.window_fraction)
    coefficients *= ell[0] ** (-1j * eta)
    return FFTLogPowerSum(coefficients, config.bias + 1j * eta)


def fftlog_power_sums_los(ell, values, config: SlepianConfig):
    """FFTLog-expand one radial factor independently at each LOS node."""
    values = np.asarray(values, dtype=complex)
    ell = np.asarray(ell, dtype=float)
    if values.ndim != 2 or values.shape[1:] != ell.shape:
        raise ValueError("values must have shape (n_z, n_ell)")
    sums = tuple(fftlog_power_sum(ell, row, config) for row in values)
    exponents = sums[0].exponents
    if any(
        not np.array_equal(power_sum.exponents, exponents)
        for power_sum in sums[1:]
    ):
        raise RuntimeError("FFTLog exponents changed between LOS nodes")
    return np.stack([power_sum.coefficients for power_sum in sums]), exponents


def _canonical_bessel_order(order: int) -> tuple[int, int]:
    order = int(order)
    if order >= 0:
        return order, 1
    positive = -order
    return positive, -1 if positive % 2 else 1


def _contact_coefficient(order_x: int, order_theta: int) -> int:
    """Return ``cos(pi * (order_x - order_theta) / 2)`` exactly."""
    difference = (int(order_x) - int(order_theta)) % 4
    if difference == 0:
        return 1
    if difference == 2:
        return -1
    return 0


def _regular_quadrature_supported(order_x: int, order_theta: int) -> bool:
    """Return whether unequal canonical orders require regular quadrature."""
    return abs(int(order_x)) != abs(int(order_theta))


def _single_bessel_factor(exponents, order: int):
    canonical, sign = _canonical_bessel_order(order)
    exponents = np.asarray(exponents, dtype=complex)
    return (
        sign
        * np.exp((exponents + 1.0) * np.log(2.0))
        * np.exp(loggamma((canonical + exponents + 2.0) / 2.0))
        * rgamma((canonical - exponents) / 2.0)
    )


def single_radial_transform(power_sum: FFTLogPowerSum, order: int, radius):
    """Evaluate ``int dlog(ell) ell^2 f(ell) J_order(ell r)``."""
    radius = np.asarray(radius, dtype=float)
    coefficients = power_sum.coefficients * _single_bessel_factor(
        power_sum.exponents, order
    )
    powers = radius[None, :] ** (-power_sum.exponents[:, None] - 2.0)
    return np.sum(coefficients[:, None] * powers, axis=0)


def _hyp2f1(a, b, c, z: float, rtol: float):
    term = 1.0 + 0.0j
    total = term
    for index in range(100000):
        term *= (
            (a + index)
            * (b + index)
            * z
            / ((c + index) * (index + 1.0))
        )
        total += term
        if abs(term) <= rtol * max(1.0, abs(total)):
            return total
    raise RuntimeError("hypergeometric series did not converge")


def _eta_zero_weber_unit_power(
    order_small: int, order_big: int, ratio: float
) -> complex:
    """Evaluate the regular eta-zero Weber kernel for even order differences."""
    mu, sign_small = _canonical_bessel_order(order_small)
    nu_big, sign_big = _canonical_bessel_order(order_big)
    difference = nu_big - mu
    if difference <= 0:
        return 0.0j
    if difference % 2:
        raise ValueError("eta-zero polynomial requires an even order difference")

    degree = difference // 2 - 1
    a = (mu + nu_big + 2) // 2
    b = 1 - difference // 2
    c = mu + 1
    z = float(ratio) ** 2
    term = 1.0
    polynomial = term
    for index in range(degree):
        term *= (
            (a + index)
            * (b + index)
            * z
            / ((c + index) * (index + 1.0))
        )
        polynomial += term

    log_prefactor = (
        np.log(2.0)
        + loggamma(mu + difference // 2 + 1)
        - loggamma(difference // 2)
        - loggamma(mu + 1)
    )
    return (
        sign_small
        * sign_big
        * ratio**mu
        * np.exp(log_prefactor)
        * polynomial
    )


_WEBER_REFLECTION_MAX_RATIO = 1.0 / np.sqrt(2.0)
_WEBER_REFLECTION_MIN_RATIO = 0.25
_WEBER_REFLECTION_INDEX_SCALE = 8.0


def _weber_reflection_ratio(exponent) -> float:
    """Return the Mellin-index-dependent connection-formula boundary."""
    imaginary_index = abs(complex(exponent).imag)
    if imaginary_index == 0.0:
        return _WEBER_REFLECTION_MAX_RATIO
    return max(
        _WEBER_REFLECTION_MIN_RATIO,
        min(
            _WEBER_REFLECTION_MAX_RATIO,
            _WEBER_REFLECTION_INDEX_SCALE / imaginary_index,
        ),
    )


def _reflected_weber_unit_power(
    exponent, order_small: int, order_big: int, ratio: float, rtol: float
) -> complex:
    """Evaluate the Weber kernel with hypergeometric series about ratio one."""
    mu, sign_small = _canonical_bessel_order(order_small)
    nu_big, sign_big = _canonical_bessel_order(order_big)
    lam = -complex(exponent) - 1.0
    A = (nu_big + mu - lam + 1.0) / 2.0
    B = (mu - nu_big - lam + 1.0) / 2.0
    C = complex(mu + 1.0)
    D = (nu_big - mu + lam + 1.0) / 2.0
    complement = 1.0 - float(ratio) ** 2
    common = mu * np.log(float(ratio)) - lam * np.log(2.0)
    log_coefficient_1 = (
        common
        + loggamma(A)
        + loggamma(lam)
        - loggamma(D)
        - loggamma(C - A)
        - loggamma(C - B)
    )
    log_coefficient_2 = (
        common
        + lam * np.log(complement)
        + loggamma(-lam)
        - loggamma(D)
        - loggamma(B)
    )
    hypergeometric_1 = _hyp2f1(A, B, 1.0 - lam, complement, rtol)
    hypergeometric_2 = _hyp2f1(
        C - A, C - B, 1.0 + lam, complement, rtol
    )
    return sign_small * sign_big * (
        np.exp(log_coefficient_1) * hypergeometric_1
        + np.exp(log_coefficient_2) * hypergeometric_2
    )


def _weber_unit_power(
    exponent, order_small: int, order_big: int, ratio: float, rtol: float
):
    mu, sign_small = _canonical_bessel_order(order_small)
    nu_big, sign_big = _canonical_bessel_order(order_big)
    if complex(exponent) == 0.0j and (nu_big - mu) % 2 == 0:
        return _eta_zero_weber_unit_power(order_small, order_big, ratio)
    lam = -complex(exponent) - 1.0
    integer_lambda = abs(lam.imag) < 1.0e-15 and np.isclose(
        lam.real, round(lam.real), atol=1.0e-12
    )
    if ratio >= _weber_reflection_ratio(exponent) and not integer_lambda:
        return _reflected_weber_unit_power(
            exponent, order_small, order_big, ratio, rtol
        )
    A = (nu_big + mu - lam + 1.0) / 2.0
    B = (mu - nu_big - lam + 1.0) / 2.0
    C = mu + 1.0
    D = (nu_big - mu + lam + 1.0) / 2.0
    if ratio == 0.0 and mu > 0:
        return 0.0j
    inverse_gamma_d = rgamma(D)
    if inverse_gamma_d == 0.0:
        return 0.0j
    prefactor = (
        ratio**mu
        * np.exp(-lam * np.log(2.0))
        * np.exp(loggamma(A))
        * inverse_gamma_d
        * rgamma(C)
    )
    return sign_small * sign_big * prefactor * _hyp2f1(A, B, C, ratio**2, rtol)


def _interpolated_weber_unit_power(
    exponent,
    order_small: int,
    order_big: int,
    ratios,
    *,
    nodes: int,
    max_ratio: float,
    rtol: float,
    table_cache: dict | None = None,
):
    """Evaluate the unit Weber function on a cubic table in ``log(-log(r))``."""
    ratios = np.asarray(ratios, dtype=float)
    if ratios.ndim != 1 or np.any(ratios <= 0.0) or np.any(ratios > 1.0):
        raise ValueError("ratios must be a one-dimensional array in (0, 1]")

    result = np.empty(ratios.shape, dtype=complex)
    direct = ratios >= float(max_ratio)
    if np.any(direct):
        result[direct] = np.array(
            [
                _weber_unit_power(
                    exponent, order_small, order_big, float(ratio), rtol
                )
                for ratio in ratios[direct]
            ]
        )

    regular = ~direct
    if not np.any(regular):
        return result
    if np.count_nonzero(regular) <= int(nodes):
        result[regular] = np.array(
            [
                _weber_unit_power(
                    exponent, order_small, order_big, float(ratio), rtol
                )
                for ratio in ratios[regular]
            ]
        )
        return result
    u = -np.log(ratios[regular])
    t = np.log(u)
    t_min = float(np.min(t))
    t_max = float(np.max(t))
    if np.isclose(t_min, t_max, rtol=1.0e-14, atol=0.0):
        result[regular] = _weber_unit_power(
            exponent, order_small, order_big, float(np.exp(-np.exp(t_min))), rtol
        )
        return result

    key = (
        complex(exponent),
        int(order_small),
        int(order_big),
        t_min,
        t_max,
        int(nodes),
        float(max_ratio),
        float(rtol),
    )
    table = None if table_cache is None else table_cache.get(key)
    if table is None:
        t_table = np.linspace(t_min, t_max, int(nodes))
        values = np.array(
            [
                _weber_unit_power(
                    exponent,
                    order_small,
                    order_big,
                    float(np.exp(-np.exp(t_value))),
                    rtol,
                )
                for t_value in t_table
            ]
        )
        table = CubicSpline(t_table, values)
        if table_cache is not None:
            table_cache[key] = table
    result[regular] = table(t)
    return result


def _powerlaw_double_kernel(
    exponent,
    order_x: int,
    order_theta: int,
    x,
    theta,
    *,
    rtol: float,
    omit_diagonal: bool,
    geometry: WeberGeometry | None = None,
    method: str = "direct",
    interpolation_nodes: int = 64,
    interpolation_max_ratio: float = 0.8,
    interpolation_cache: dict | None = None,
    max_ratio: float = 1.0,
):
    geometry = geometry or WeberGeometry.from_coordinates(x, theta)
    ratios, active, inverse = geometry.unique_ratios(
        omit_diagonal=omit_diagonal, max_ratio=max_ratio
    )
    result = np.zeros(geometry.scale.shape, dtype=complex)

    forward = geometry.x[:, None] <= geometry.theta[None, :]
    if int(order_x) == int(order_theta):
        orientations = ((active, order_x, order_theta),)
    else:
        orientations = (
            (active & forward, order_x, order_theta),
            (active & ~forward, order_theta, order_x),
        )
    for orientation, small_order, big_order in orientations:
        if not np.any(orientation):
            continue
        active_indices = np.flatnonzero(active)
        orientation_flat = orientation.ravel()[active_indices]
        needed = np.unique(inverse[orientation_flat])
        if method == "direct":
            unit_values = np.array(
                [
                    _weber_unit_power(
                        exponent,
                        small_order,
                        big_order,
                        ratios[index],
                        rtol,
                    )
                    for index in needed
                ]
            )
        elif method == "interpolated":
            unit_values = _interpolated_weber_unit_power(
                exponent,
                small_order,
                big_order,
                ratios[needed],
                nodes=interpolation_nodes,
                max_ratio=interpolation_max_ratio,
                rtol=rtol,
                table_cache=interpolation_cache,
            )
        else:
            raise ValueError("method must be 'direct' or 'interpolated'")
        unit = dict(zip(needed, unit_values))
        values = np.array([unit[index] for index in inverse[orientation_flat]])
        result[orientation] = (
            geometry.scale[orientation] ** (-complex(exponent) - 2.0)
            * values
        )
    return result


def constant_leg_kernel(
    order_x: int,
    order_theta: int,
    x,
    theta,
    config: SlepianConfig,
    *,
    geometry: WeberGeometry | None = None,
    interpolation_cache: dict | None = None,
) -> ConstantLegKernel:
    """Decompose a constant-leg Bessel closure kernel away from its diagonal."""
    if not isinstance(config, SlepianConfig):
        raise TypeError("config must be a SlepianConfig")
    geometry = geometry or WeberGeometry.from_coordinates(x, theta)
    regular = _powerlaw_double_kernel(
        0.0,
        order_x,
        order_theta,
        geometry.x,
        geometry.theta,
        rtol=config.weber_rtol,
        omit_diagonal=True,
        geometry=geometry,
        method=config.weber_method,
        interpolation_nodes=config.weber_interpolation_nodes,
        interpolation_max_ratio=config.weber_interpolation_max_ratio,
        interpolation_cache=interpolation_cache,
    )
    xx = geometry.x[:, None]
    tt = geometry.theta[None, :]
    regular_less = np.where(xx < tt, regular, 0.0)
    regular_greater = np.where(xx > tt, regular, 0.0)
    return ConstantLegKernel(
        _contact_coefficient(order_x, order_theta),
        regular_less,
        regular_greater,
        abs(int(order_x)) != abs(int(order_theta)),
    )


def regular_mellin_matrix(
    ell,
    single_exponents,
    double_exponents,
    single_order: int,
    double_orders: tuple[int, int],
    constant_orders: tuple[int, int],
    x,
    theta,
    config: SlepianConfig,
    *,
    geometry: WeberGeometry | None = None,
    constant_kernel: ConstantLegKernel | None = None,
    interpolation_cache: dict | None = None,
) -> RegularMellinMatrix:
    """Construct the uncompressed regular matrix before Mellin coefficients."""
    ell = np.asarray(ell, dtype=float)
    single_exponents = np.asarray(single_exponents, dtype=complex)
    double_exponents = np.asarray(double_exponents, dtype=complex)
    x = np.asarray(x, dtype=float)
    theta = np.asarray(theta, dtype=float)
    if ell.ndim != 1 or x.ndim != 1 or theta.ndim != 1:
        raise ValueError("ell, x, and theta must be one-dimensional")
    if config.regular_quadrature == "ratio_gauss":
        if config.regular_matrix_implementation == "vectorized":
            return _regular_mellin_matrix_ratio_vectorized(
                ell,
                single_exponents,
                double_exponents,
                single_order,
                double_orders,
                constant_orders,
                theta,
                config,
                interpolation_cache=interpolation_cache,
            )
        nodes, weights = leggauss(config.regular_n_ratio)
        ratio_min = config.regular_ratio_min
        ratio = 0.5 * ((1.0 - ratio_min) * nodes + 1.0 + ratio_min)
        weights = 0.5 * (1.0 - ratio_min) * weights
        values = np.zeros(
            (
                double_exponents.size,
                single_exponents.size,
                theta.size,
                theta.size,
            ),
            dtype=complex,
        )
        single_factor = _single_bessel_factor(single_exponents, single_order)
        double_order_x, double_order_theta = double_orders

        for constant_index, theta_constant in enumerate(theta):
            for branch, branch_x in (
                ("less", theta_constant * ratio),
                ("greater", theta_constant / ratio),
            ):
                branch_geometry = WeberGeometry.from_coordinates(branch_x, theta)
                kernel = constant_leg_kernel(
                    *constant_orders,
                    branch_x,
                    np.array([theta_constant]),
                    config,
                    interpolation_cache=interpolation_cache,
                )
                if branch == "less":
                    regular = kernel.regular_less[:, 0]
                    jacobian = theta_constant**2 * ratio
                else:
                    regular = kernel.regular_greater[:, 0]
                    jacobian = theta_constant**2 / ratio**3
                if not np.any(regular):
                    continue
                single_basis = (
                    single_factor[:, None]
                    * branch_x[None, :] ** (-single_exponents[:, None] - 2.0)
                )
                radial_weight = weights * jacobian * regular
                for double_index, exponent in enumerate(double_exponents):
                    double_basis = _powerlaw_double_kernel(
                        exponent,
                        double_order_x,
                        double_order_theta,
                        branch_x,
                        theta,
                        rtol=config.weber_rtol,
                        omit_diagonal=config.diagonal_correction == "brute",
                        geometry=branch_geometry,
                        method=config.weber_method,
                        interpolation_nodes=config.weber_interpolation_nodes,
                        interpolation_max_ratio=(
                            config.weber_interpolation_max_ratio
                        ),
                        interpolation_cache=interpolation_cache,
                        max_ratio=(
                            config.weber_brute_min_ratio
                            if config.diagonal_correction == "brute"
                            else 1.0
                        ),
                    )
                    if config.diagonal_correction == "brute":
                        brute_mask = (
                            branch_geometry.ratio
                            >= config.weber_brute_min_ratio
                        )
                        if np.any(brute_mask):
                            brute = _double_radial_brute(
                                ell,
                                ell**exponent,
                                double_order_x,
                                double_order_theta,
                                branch_x,
                                theta,
                            )
                            double_basis[brute_mask] = brute[brute_mask]
                    values[double_index, :, :, constant_index] += np.einsum(
                        "r,br,ri->bi",
                        radial_weight,
                        single_basis,
                        double_basis,
                    )
        return RegularMellinMatrix(
            values,
            double_exponents,
            single_exponents,
            ratio,
            theta,
            single_order,
            double_orders,
            constant_orders,
        )

    geometry = geometry or WeberGeometry.from_coordinates(x, theta)
    constant_kernel = constant_kernel or constant_leg_kernel(
        *constant_orders,
        x,
        theta,
        config,
        geometry=geometry,
        interpolation_cache=interpolation_cache,
    )
    regular = constant_kernel.regular_less + constant_kernel.regular_greater
    single_basis = (
        _single_bessel_factor(single_exponents, single_order)[:, None]
        * x[None, :] ** (-single_exponents[:, None] - 2.0)
    )
    values = np.empty(
        (
            double_exponents.size,
            single_exponents.size,
            theta.size,
            theta.size,
        ),
        dtype=complex,
    )
    double_order_x, double_order_theta = double_orders
    for index, exponent in enumerate(double_exponents):
        double_basis = _powerlaw_double_kernel(
            exponent,
            double_order_x,
            double_order_theta,
            x,
            theta,
            rtol=config.weber_rtol,
            omit_diagonal=config.diagonal_correction == "brute",
            geometry=geometry,
            method=config.weber_method,
            interpolation_nodes=config.weber_interpolation_nodes,
            interpolation_max_ratio=config.weber_interpolation_max_ratio,
            interpolation_cache=interpolation_cache,
        )
        if config.diagonal_correction == "brute" and np.any(geometry.diagonal):
            brute = _double_radial_brute(
                ell,
                ell**exponent,
                double_order_x,
                double_order_theta,
                x,
                theta,
            )
            double_basis[geometry.diagonal] = brute[geometry.diagonal]
        integrand = (
            x[:, None, None, None]
            * single_basis.T[:, :, None, None]
            * double_basis[:, None, :, None]
            * regular[:, None, None, :]
        )
        values[index] = np.trapezoid(integrand, x, axis=0)
    return RegularMellinMatrix(
        values,
        double_exponents,
        single_exponents,
        x,
        theta,
        single_order,
        double_orders,
        constant_orders,
    )


def _regular_mellin_matrix_ratio_vectorized(
    ell,
    single_exponents,
    double_exponents,
    single_order: int,
    double_orders: tuple[int, int],
    constant_orders: tuple[int, int],
    theta,
    config: SlepianConfig,
    *,
    interpolation_cache: dict | None = None,
) -> RegularMellinMatrix:
    """Construct ratio-Gauss ``F_ab`` by batching all radial coordinates."""
    nodes, weights = leggauss(config.regular_n_ratio)
    ratio_min = config.regular_ratio_min
    ratio = 0.5 * ((1.0 - ratio_min) * nodes + 1.0 + ratio_min)
    weights = 0.5 * (1.0 - ratio_min) * weights

    theta = np.asarray(theta, dtype=float)
    theta_column = theta[:, None]
    branch_x = np.stack(
        (theta_column * ratio[None, :], theta_column / ratio[None, :]),
        axis=1,
    )
    ntheta, nbranch, nratio = branch_x.shape
    flat_x = branch_x.reshape(-1)

    order_x, order_theta = constant_orders
    if config.weber_method == "direct":
        def evaluate_constant(left, right):
            return np.array(
                [
                    _weber_unit_power(
                        0.0, left, right, value, config.weber_rtol
                    )
                    for value in ratio
                ]
            )
    else:
        def evaluate_constant(left, right):
            return _interpolated_weber_unit_power(
                0.0,
                left,
                right,
                ratio,
                nodes=config.weber_interpolation_nodes,
                max_ratio=config.weber_interpolation_max_ratio,
                rtol=config.weber_rtol,
                table_cache=interpolation_cache,
            )

    constant_less = evaluate_constant(order_x, order_theta)
    constant_greater = evaluate_constant(order_theta, order_x)

    regular = np.empty_like(branch_x, dtype=complex)
    regular[:, 0, :] = theta_column**-2 * constant_less[None, :]
    regular[:, 1, :] = (
        theta_column / ratio[None, :]
    ) ** -2 * constant_greater[None, :]
    jacobian = np.empty_like(branch_x)
    jacobian[:, 0, :] = theta_column**2 * ratio[None, :]
    jacobian[:, 1, :] = theta_column**2 / ratio[None, :] ** 3
    radial_weight = weights[None, None, :] * jacobian * regular

    single_factor = _single_bessel_factor(single_exponents, single_order)
    single_basis = (
        single_factor[:, None]
        * flat_x[None, :] ** (-single_exponents[:, None] - 2.0)
    ).reshape(single_exponents.size, ntheta, nbranch, nratio)

    geometry = WeberGeometry.from_coordinates(flat_x, theta)
    values = np.empty(
        (
            double_exponents.size,
            single_exponents.size,
            ntheta,
            ntheta,
        ),
        dtype=complex,
    )
    double_order_x, double_order_theta = double_orders
    for index, exponent in enumerate(double_exponents):
        double_basis = _powerlaw_double_kernel(
            exponent,
            double_order_x,
            double_order_theta,
            flat_x,
            theta,
            rtol=config.weber_rtol,
            omit_diagonal=config.diagonal_correction == "brute",
            geometry=geometry,
            method=config.weber_method,
            interpolation_nodes=config.weber_interpolation_nodes,
            interpolation_max_ratio=config.weber_interpolation_max_ratio,
            interpolation_cache=interpolation_cache,
            max_ratio=(
                config.weber_brute_min_ratio
                if config.diagonal_correction == "brute"
                else 1.0
            ),
        ).reshape(ntheta, nbranch, nratio, ntheta)
        if config.diagonal_correction == "brute":
            brute_mask = geometry.ratio >= config.weber_brute_min_ratio
            if np.any(brute_mask):
                brute = _double_radial_brute(
                    ell,
                    ell**exponent,
                    double_order_x,
                    double_order_theta,
                    flat_x,
                    theta,
                )
                reshaped = double_basis.reshape(flat_x.size, theta.size)
                reshaped[brute_mask] = brute[brute_mask]
        values[index] = np.einsum(
            "cpr,bcpr,cpri->bic",
            radial_weight,
            single_basis,
            double_basis,
            optimize=True,
        )

    return RegularMellinMatrix(
        values,
        double_exponents,
        single_exponents,
        ratio,
        theta,
        single_order,
        double_orders,
        constant_orders,
    )


def contract_regular_mellin_matrix(
    matrix: RegularMellinMatrix,
    single_coefficients,
    double_coefficients,
):
    """Contract a full regular matrix with two sets of Mellin coefficients."""
    if not isinstance(matrix, RegularMellinMatrix):
        raise TypeError("matrix must be a RegularMellinMatrix")
    single = np.asarray(single_coefficients, dtype=complex)
    double = np.asarray(double_coefficients, dtype=complex)
    if single.shape != matrix.single_exponents.shape:
        raise ValueError("single_coefficients have the wrong shape")
    if double.shape != matrix.double_exponents.shape:
        raise ValueError("double_coefficients have the wrong shape")
    return np.einsum("a,b,abij->ij", double, single, matrix.values)


def compress_regular_mellin_matrix(
    matrix: RegularMellinMatrix,
    *,
    rank: int | None = None,
    rtol: float = 1.0e-6,
) -> LowRankRegularMellinMatrix:
    """Compress each theta-pair Mellin matrix with a common retained rank."""
    if not isinstance(matrix, RegularMellinMatrix):
        raise TypeError("matrix must be a RegularMellinMatrix")
    matrices = np.moveaxis(matrix.values, (0, 1), (-2, -1))
    left, singular, right = np.linalg.svd(matrices, full_matrices=False)
    maximum_rank = singular.shape[-1]
    if rank is None:
        total = np.sum(singular**2, axis=-1)
        tail = total[..., None] - np.cumsum(singular**2, axis=-1)
        relative_tail = np.sqrt(
            np.maximum(tail, 0.0) / np.maximum(total[..., None], 1.0e-300)
        )
        sufficient = relative_tail <= float(rtol)
        required = np.where(
            np.any(sufficient, axis=-1),
            np.argmax(sufficient, axis=-1) + 1,
            maximum_rank,
        )
        retained_rank = int(np.max(required))
    else:
        retained_rank = min(int(rank), maximum_rank)
    left = left[..., :retained_rank]
    singular = singular[..., :retained_rank]
    right = right[..., :retained_rank, :]
    reconstructed = np.einsum("ijar,ijr,ijrb->ijab", left, singular, right)
    denominator = max(float(np.linalg.norm(matrices)), 1.0e-300)
    relative_error = float(np.linalg.norm(reconstructed - matrices) / denominator)
    return LowRankRegularMellinMatrix(
        left,
        singular,
        right,
        matrix.double_exponents,
        matrix.single_exponents,
        retained_rank,
        relative_error,
    )


def contract_low_rank_regular_mellin_matrix(
    matrix: LowRankRegularMellinMatrix,
    single_coefficients,
    double_coefficients,
):
    """Contract cached low-rank factors with two Mellin coefficient vectors."""
    if not isinstance(matrix, LowRankRegularMellinMatrix):
        raise TypeError("matrix must be a LowRankRegularMellinMatrix")
    single = np.asarray(single_coefficients, dtype=complex)
    double = np.asarray(double_coefficients, dtype=complex)
    if double.shape != (matrix.left_vectors.shape[2],):
        raise ValueError("double_coefficients have the wrong shape")
    if single.shape != (matrix.right_vectors.shape[3],):
        raise ValueError("single_coefficients have the wrong shape")
    return np.einsum(
        "a,ijar,ijr,ijrb,b->ij",
        double,
        matrix.left_vectors,
        matrix.singular_values,
        matrix.right_vectors,
        single,
    )


def contract_low_rank_regular_mellin_matrix_los(
    matrix: LowRankRegularMellinMatrix,
    factors: LOSMellinFactors,
):
    """Contract low-rank ``F_ab`` before LOS quadrature.

    The dense LOS-integrated coefficient matrix is never materialized.
    """
    if not isinstance(matrix, LowRankRegularMellinMatrix):
        raise TypeError("matrix must be a LowRankRegularMellinMatrix")
    if not isinstance(factors, LOSMellinFactors):
        raise TypeError("factors must be LOSMellinFactors")
    if factors.double_coefficients.shape[1] != matrix.left_vectors.shape[2]:
        raise ValueError("double_coefficients have the wrong Mellin size")
    if factors.single_coefficients.shape[1] != matrix.right_vectors.shape[3]:
        raise ValueError("single_coefficients have the wrong Mellin size")
    if not np.array_equal(factors.double_exponents, matrix.double_exponents):
        raise ValueError("double_exponents do not match the low-rank matrix")
    if not np.array_equal(factors.single_exponents, matrix.single_exponents):
        raise ValueError("single_exponents do not match the low-rank matrix")
    left_projection = np.einsum(
        "za,ijar->zijr",
        factors.double_coefficients,
        matrix.left_vectors,
    )
    right_projection = np.einsum(
        "zb,ijrb->zijr",
        factors.single_coefficients,
        matrix.right_vectors,
    )
    integrand = np.einsum(
        "ijr,zijr,zijr->zij",
        matrix.singular_values,
        left_projection,
        right_projection,
    )
    return np.trapezoid(
        factors.weight[:, None, None] * integrand,
        factors.chi,
        axis=0,
    )


def _double_radial_brute(ell, values, order_x: int, order_theta: int, x, theta):
    ell = np.asarray(ell, dtype=float)
    values = np.asarray(values, dtype=complex)
    x = np.asarray(x, dtype=float)
    theta = np.asarray(theta, dtype=float)
    integrand = (
        ell[:, None, None] ** 2
        * values[:, None, None]
        * jv(order_x, ell[:, None, None] * x[None, :, None])
        * jv(order_theta, ell[:, None, None] * theta[None, None, :])
    )
    return np.trapezoid(integrand, np.log(ell), axis=0)


def double_radial_transform(
    ell,
    values,
    power_sum: FFTLogPowerSum,
    order_x: int,
    order_theta: int,
    x,
    theta,
    config: SlepianConfig,
    *,
    geometry: WeberGeometry | None = None,
    kernel_cache: dict | None = None,
    interpolation_cache: dict | None = None,
):
    """Evaluate a two-Bessel radial integral with finite FFTLog powers."""
    x = np.asarray(x, dtype=float)
    theta = np.asarray(theta, dtype=float)
    geometry = geometry or WeberGeometry.from_coordinates(x, theta)
    replace_near_diagonal = config.diagonal_correction == "brute"
    brute_mask = replace_near_diagonal & (
        geometry.ratio >= config.weber_brute_min_ratio
    )
    result = np.zeros((x.size, theta.size), dtype=complex)
    for coefficient, exponent in zip(
        power_sum.coefficients, power_sum.exponents
    ):
        if coefficient != 0.0:
            cache_key = (
                id(geometry),
                complex(exponent),
                int(order_x),
                int(order_theta),
                float(config.weber_rtol),
                bool(replace_near_diagonal),
                float(config.weber_brute_min_ratio),
                config.weber_method,
                config.weber_interpolation_nodes,
                config.weber_interpolation_max_ratio,
            )
            kernel = None if kernel_cache is None else kernel_cache.get(cache_key)
            if kernel is None:
                kernel = _powerlaw_double_kernel(
                    exponent,
                    order_x,
                    order_theta,
                    x,
                    theta,
                    rtol=config.weber_rtol,
                    omit_diagonal=replace_near_diagonal,
                    geometry=geometry,
                    method=config.weber_method,
                    interpolation_nodes=config.weber_interpolation_nodes,
                    interpolation_max_ratio=config.weber_interpolation_max_ratio,
                    interpolation_cache=interpolation_cache,
                    max_ratio=(
                        config.weber_brute_min_ratio
                        if replace_near_diagonal
                        else 1.0
                    ),
                )
                if kernel_cache is not None:
                    kernel_cache[cache_key] = kernel
            result += coefficient * kernel
    if np.any(brute_mask):
        tapered = np.asarray(values) * _log_edge_taper(
            len(ell), config.taper_fraction
        )
        brute = _double_radial_brute(
            ell, tapered, order_x, order_theta, x, theta
        )
        result[brute_mask] = brute[brute_mask]
    return result


class SlepianCalculator:
    """Calculator for direct native-2D separable-term contributions to ZetaK."""

    def __init__(self, config: SlepianConfig):
        if not isinstance(config, SlepianConfig):
            raise TypeError("config must be a SlepianConfig")
        self.config = config
        self._power_sums: dict[tuple[int, int], tuple[np.ndarray, FFTLogPowerSum]] = {}
        self._geometries: dict[tuple, WeberGeometry] = {}
        self._weber_kernels: dict[tuple, np.ndarray] = {}
        self._weber_interpolators: dict[tuple, CubicSpline] = {}
        self._constant_leg_kernels: dict[tuple, ConstantLegKernel] = {}
        self._regular_mellin_matrices: dict[tuple, RegularMellinMatrix] = {}
        self._low_rank_regular_mellin_matrices: dict[
            tuple, LowRankRegularMellinMatrix
        ] = {}
        self._timings: dict[str, float] = {}

    @property
    def timing_summary(self) -> dict[str, float]:
        """Return cumulative coarse-grained Slepian timing counters."""
        return dict(self._timings)

    def reset_timings(self) -> None:
        """Reset timing counters without invalidating numerical caches."""
        self._timings.clear()

    def _record_timing(self, name: str, value: float = 1.0) -> None:
        self._timings[name] = self._timings.get(name, 0.0) + float(value)

    def clear(self) -> None:
        self._power_sums.clear()
        self._geometries.clear()
        self._weber_kernels.clear()
        self._weber_interpolators.clear()
        self._constant_leg_kernels.clear()
        self._regular_mellin_matrices.clear()
        self._low_rank_regular_mellin_matrices.clear()
        self.reset_timings()

    def _clear_source_cache(self) -> None:
        self._power_sums.clear()

    def _geometry(self, x, theta) -> WeberGeometry:
        x = np.asarray(x, dtype=float)
        theta = np.asarray(theta, dtype=float)
        key = (x.shape, x.tobytes(), theta.shape, theta.tobytes())
        if key not in self._geometries:
            self._geometries[key] = WeberGeometry.from_coordinates(x, theta)
        return self._geometries[key]

    def _factor_data(self, expression, leg: int, ell):
        key = (id(expression), int(leg))
        if key not in self._power_sums:
            values = np.asarray(expression.radial_factors[leg].evaluate(ell))
            self._power_sums[key] = (
                values,
                fftlog_power_sum(ell, values, self.config),
            )
        return self._power_sums[key]

    def _constant_leg_kernel(self, order_x: int, order_theta: int, x, theta):
        geometry = self._geometry(x, theta)
        key = (
            id(geometry),
            int(order_x),
            int(order_theta),
            float(self.config.weber_rtol),
            self.config.weber_method,
            self.config.weber_interpolation_nodes,
            self.config.weber_interpolation_max_ratio,
        )
        if key not in self._constant_leg_kernels:
            self._constant_leg_kernels[key] = constant_leg_kernel(
                order_x,
                order_theta,
                x,
                theta,
                self.config,
                geometry=geometry,
                interpolation_cache=self._weber_interpolators,
            )
        return self._constant_leg_kernels[key]

    def _regular_x_grid(self, theta):
        theta = np.asarray(theta, dtype=float)
        padding = self.config.regular_x_padding
        base = np.geomspace(
            theta[0] / padding,
            theta[-1] * padding,
            self.config.regular_n_x,
        )
        return np.unique(np.concatenate((base, theta)))

    def _regular_ratio_rule(self):
        nodes, weights = leggauss(self.config.regular_n_ratio)
        ratio_min = self.config.regular_ratio_min
        ratio = 0.5 * ((1.0 - ratio_min) * nodes + 1.0 + ratio_min)
        weights = 0.5 * (1.0 - ratio_min) * weights
        return ratio, weights

    def _regular_matrix_inputs(self, layout: _SlepianLegLayout, theta):
        if self.config.regular_quadrature == "ratio_gauss":
            ratio, _ = self._regular_ratio_rule()
            return ratio, None
        x = self._regular_x_grid(theta)
        kernel = self._constant_leg_kernel(
            *layout.constant_orders, x, theta
        )
        return x, kernel

    def _regular_ratio_contribution(
        self,
        ell,
        values_double,
        power1: FFTLogPowerSum,
        power_double: FFTLogPowerSum,
        layout: _SlepianLegLayout,
        theta,
    ):
        """Integrate the two open constant-leg branches in ratio space."""
        theta = np.asarray(theta, dtype=float)
        ratio, weights = self._regular_ratio_rule()
        contribution = np.zeros((theta.size, theta.size), dtype=complex)

        for index, theta_constant in enumerate(theta):
            x_less = theta_constant * ratio
            x_greater = theta_constant / ratio
            branch_total = np.zeros(theta.size, dtype=complex)

            for branch, x in (("less", x_less), ("greater", x_greater)):
                radial1 = single_radial_transform(
                    power1, layout.single_order, x
                )
                radial_double = double_radial_transform(
                    ell,
                    values_double,
                    power_double,
                    *layout.double_orders,
                    x,
                    theta,
                    self.config,
                    geometry=self._geometry(x, theta),
                    kernel_cache=self._weber_kernels,
                    interpolation_cache=self._weber_interpolators,
                )
                kernel = self._constant_leg_kernel(
                    *layout.constant_orders, x, np.array([theta_constant])
                )
                if branch == "less":
                    regular = kernel.regular_less[:, 0]
                    jacobian = theta_constant**2 * ratio
                else:
                    regular = kernel.regular_greater[:, 0]
                    jacobian = theta_constant**2 / ratio**3
                branch_total += np.sum(
                    weights[:, None]
                    * jacobian[:, None]
                    * radial1[:, None]
                    * radial_double
                    * regular[:, None],
                    axis=0,
                )
            contribution[:, index] = branch_total
        return contribution

    def _regular_mellin_matrix(
        self,
        ell,
        power1: FFTLogPowerSum,
        power2: FFTLogPowerSum,
        single_order: int,
        double_orders: tuple[int, int],
        constant_orders: tuple[int, int],
        x,
        theta,
        constant_kernel: ConstantLegKernel | None,
    ) -> RegularMellinMatrix:
        geometry = self._geometry(x, theta)
        key = (
            id(geometry),
            power1.exponents.shape,
            power1.exponents.tobytes(),
            power2.exponents.shape,
            power2.exponents.tobytes(),
            int(single_order),
            tuple(int(v) for v in double_orders),
            tuple(int(v) for v in constant_orders),
            float(self.config.weber_rtol),
            self.config.weber_method,
            self.config.weber_interpolation_nodes,
            self.config.weber_interpolation_max_ratio,
            self.config.diagonal_correction,
            self.config.weber_brute_min_ratio,
            self.config.regular_quadrature,
            self.config.regular_matrix_implementation,
            self.config.regular_n_ratio,
            self.config.regular_ratio_min,
        )
        if key not in self._regular_mellin_matrices:
            self._regular_mellin_matrices[key] = regular_mellin_matrix(
                ell,
                power1.exponents,
                power2.exponents,
                single_order,
                double_orders,
                constant_orders,
                x,
                theta,
                self.config,
                geometry=geometry,
                constant_kernel=constant_kernel,
                interpolation_cache=self._weber_interpolators,
            )
        return self._regular_mellin_matrices[key]

    def _low_rank_regular_mellin_matrix(
        self,
        ell,
        power1: FFTLogPowerSum,
        power2: FFTLogPowerSum,
        single_order: int,
        double_orders: tuple[int, int],
        constant_orders: tuple[int, int],
        x,
        theta,
        constant_kernel: ConstantLegKernel | None,
    ) -> LowRankRegularMellinMatrix:
        geometry = self._geometry(x, theta)
        key = (
            id(geometry),
            power1.exponents.shape,
            power1.exponents.tobytes(),
            power2.exponents.shape,
            power2.exponents.tobytes(),
            int(single_order),
            tuple(int(v) for v in double_orders),
            tuple(int(v) for v in constant_orders),
            float(self.config.weber_rtol),
            self.config.weber_method,
            self.config.weber_interpolation_nodes,
            self.config.weber_interpolation_max_ratio,
            self.config.diagonal_correction,
            self.config.weber_brute_min_ratio,
            self.config.regular_quadrature,
            self.config.regular_matrix_implementation,
            self.config.regular_n_ratio,
            self.config.regular_ratio_min,
            self.config.regular_low_rank_rank,
            self.config.regular_low_rank_rtol,
        )
        if key not in self._low_rank_regular_mellin_matrices:
            logger.debug(
                "building low-rank regular Mellin matrix: double=%d single=%d theta=%d",
                power2.exponents.size,
                power1.exponents.size,
                np.asarray(theta).size,
            )
            started = time.perf_counter()
            full = regular_mellin_matrix(
                ell,
                power1.exponents,
                power2.exponents,
                single_order,
                double_orders,
                constant_orders,
                x,
                theta,
                self.config,
                geometry=geometry,
                constant_kernel=constant_kernel,
                interpolation_cache=self._weber_interpolators,
            )
            build_seconds = time.perf_counter() - started
            self._record_timing("regular_matrix_build_seconds", build_seconds)

            started = time.perf_counter()
            compressed = compress_regular_mellin_matrix(
                full,
                rank=self.config.regular_low_rank_rank,
                rtol=self.config.regular_low_rank_rtol,
            )
            compression_seconds = time.perf_counter() - started
            self._record_timing(
                "low_rank_compression_seconds", compression_seconds
            )
            self._record_timing("low_rank_matrix_builds")
            self._low_rank_regular_mellin_matrices[key] = compressed
            logger.debug(
                "low-rank matrix finished: F_ab=%.3f s SVD=%.3f s rank=%d/%d error=%.3e",
                build_seconds,
                compression_seconds,
                compressed.retained_rank,
                min(power1.exponents.size, power2.exponents.size),
                compressed.relative_reconstruction_error,
            )
        else:
            self._record_timing("low_rank_cache_hits")
            logger.debug("low-rank regular Mellin matrix cache hit")
        return self._low_rank_regular_mellin_matrices[key]

    @staticmethod
    def _weight_value(weight):
        value = weight() if callable(weight) else weight
        if not isinstance(value, Number):
            raise TypeError("a 2D term coefficient must evaluate to a number")
        return value

    def _evaluate_projected_expression(
        self,
        expression: ProjectedSlepianRepresentation2D,
        ell,
        theta,
        k_values,
        sigma,
    ):
        projector = expression.projector
        source = expression.source_representation
        if not isinstance(source, SlepianExpression3D):
            raise TypeError("unsupported SlepianRepresentation3D implementation")
        if not projector.is_delta_like:
            return self._evaluate_projected_los_expression(
                expression, ell, theta, k_values, sigma
            )

        z = float(projector.z[0])
        chi = float(projector.chi[0])
        shift = float(projector.shift)
        radial_factors = []
        for factor in source.radial_factors:
            if factor.is_constant:
                radial_factors.append(SlepianRadialFactor2D.constant())
                continue

            def angular_factor(ell_values, *, source_factor=factor):
                ell_values = np.asarray(ell_values, dtype=float)
                return source_factor.evaluate((ell_values + shift) / chi, z)

            radial_factors.append(SlepianRadialFactor2D(angular_factor))

        term_weight = expression.source_term.coefficient
        term_weight = term_weight(z) if callable(term_weight) else term_weight
        if not isinstance(term_weight, Number):
            raise TypeError("a projected 3D term coefficient must be scalar at z")
        native = SlepianExpression2D(
            coefficient=complex(term_weight) * source.coefficient_at(z),
            radial_factors=tuple(radial_factors),
            angular_orders=source.angular_orders,
        )
        native_bispectrum = Bispectrum2D(
            (BispectrumTerm2D(expression.source_term.term.name, (native,)),)
        )
        return self.evaluate_modes(
            native_bispectrum,
            ell,
            theta,
            k_values,
            sigma=sigma,
        )

    def _evaluate_projected_los_expression(
        self,
        expression: ProjectedSlepianRepresentation2D,
        ell,
        theta,
        k_values,
        sigma,
    ):
        """Evaluate a projected 3D expression without forming dense bar-C."""
        source = expression.source_representation
        base_layout = _slepian_leg_layout(source, 0, 0, sigma)
        projector = expression.projector
        ell = np.asarray(ell, dtype=float)
        theta = np.asarray(theta, dtype=float)
        z = projector.z
        chi = projector.chi
        radial_k = (ell[None, :] + projector.shift) / chi[:, None]
        radial_z = z[:, None]
        values1 = source.radial_factors[0].evaluate(radial_k, radial_z)
        values_double = source.radial_factors[base_layout.double_leg].evaluate(
            radial_k, radial_z
        )
        coefficients1, exponents1 = fftlog_power_sums_los(
            ell, values1, self.config
        )
        coefficients_double, exponents_double = fftlog_power_sums_los(
            ell, values_double, self.config
        )

        term_weight = expression.source_term.coefficient
        amplitudes = np.empty(z.size, dtype=complex)
        for index, redshift in enumerate(z):
            weight = term_weight(redshift) if callable(term_weight) else term_weight
            if not isinstance(weight, Number):
                raise TypeError(
                    "a projected 3D term coefficient must be scalar at z"
                )
            amplitudes[index] = complex(weight) * source.coefficient_at(redshift)

        power1_template = FFTLogPowerSum(coefficients1[0], exponents1)
        power_double_template = FFTLogPowerSum(
            coefficients_double[0], exponents_double
        )
        geometry = self._geometry(theta, theta)
        results = {
            float(k): np.zeros((theta.size, theta.size), dtype=complex)
            for k in k_values
        }
        effective = as_effective_spin_triple(sigma)
        for raw_k in k_values:
            k = float(raw_k)
            m, n = effective.bessel_orders(k)
            layout = _slepian_leg_layout(source, m, n, sigma)
            constant_kernel = self._constant_leg_kernel(
                *layout.constant_orders, theta, theta
            )
            contribution = np.zeros_like(results[k])
            if constant_kernel.contact_coefficient:
                samples = []
                for index in range(z.size):
                    power1 = FFTLogPowerSum(coefficients1[index], exponents1)
                    power_double = FFTLogPowerSum(
                        coefficients_double[index], exponents_double
                    )
                    radial1 = single_radial_transform(
                        power1, layout.single_order, theta
                    )
                    radial_double = double_radial_transform(
                        ell,
                        values_double[index],
                        power_double,
                        *layout.double_orders,
                        theta,
                        theta,
                        self.config,
                        geometry=geometry,
                        kernel_cache=self._weber_kernels,
                        interpolation_cache=self._weber_interpolators,
                    )
                    samples.append(
                        amplitudes[index]
                        * constant_kernel.contact_coefficient
                        * radial_double.T
                        * radial1[None, :]
                    )
                contribution += projector.integrate_coefficients(
                    np.stack(samples),
                    axis=0,
                    sample_combination=expression.sample_combination,
                )
            if constant_kernel.has_regular:
                if not _regular_quadrature_supported(
                    *layout.constant_orders
                ):
                    raise NotImplementedError(
                        "regular quadrature currently requires a positive "
                        "even difference between canonical Bessel orders"
                    )
                if self.config.regular_method == "quadrature":
                    samples = []
                    for index in range(z.size):
                        power1 = FFTLogPowerSum(coefficients1[index], exponents1)
                        power_double = FFTLogPowerSum(
                            coefficients_double[index], exponents_double
                        )
                        if self.config.regular_quadrature == "ratio_gauss":
                            regular_contribution = (
                                self._regular_ratio_contribution(
                                    ell,
                                    values_double[index],
                                    power1,
                                    power_double,
                                    layout,
                                    theta,
                                )
                            )
                        else:
                            x = self._regular_x_grid(theta)
                            regular_kernel = self._constant_leg_kernel(
                                *layout.constant_orders, x, theta
                            )
                            radial1_x = single_radial_transform(
                                power1, layout.single_order, x
                            )
                            radial_double_x = double_radial_transform(
                                ell,
                                values_double[index],
                                power_double,
                                *layout.double_orders,
                                x,
                                theta,
                                self.config,
                                geometry=self._geometry(x, theta),
                                kernel_cache=self._weber_kernels,
                                interpolation_cache=self._weber_interpolators,
                            )
                            regular = (
                                regular_kernel.regular_less
                                + regular_kernel.regular_greater
                            )
                            integrand = (
                                x[:, None, None]
                                * radial1_x[:, None, None]
                                * radial_double_x[:, :, None]
                                * regular[:, None, :]
                            )
                            regular_contribution = np.trapezoid(
                                integrand, x, axis=0
                            )
                        samples.append(
                            amplitudes[index] * regular_contribution
                        )
                    contribution += projector.integrate_coefficients(
                        np.stack(samples),
                        axis=0,
                        sample_combination=expression.sample_combination,
                    )
                elif self.config.regular_method == "full_matrix":
                    x, regular_kernel = self._regular_matrix_inputs(
                        layout, theta
                    )
                    matrix = self._regular_mellin_matrix(
                        ell,
                        power1_template,
                        power_double_template,
                        layout.single_order,
                        layout.double_orders,
                        layout.constant_orders,
                        x,
                        theta,
                        regular_kernel,
                    )
                    samples = np.stack(
                        [
                            contract_regular_mellin_matrix(
                                matrix,
                                amplitudes[index] * coefficients1[index],
                                coefficients_double[index],
                            )
                            for index in range(z.size)
                        ]
                    )
                    contribution += projector.integrate_coefficients(
                        samples,
                        axis=0,
                        sample_combination=expression.sample_combination,
                    )
                elif self.config.regular_method == "low_rank":
                    x, regular_kernel = self._regular_matrix_inputs(
                        layout, theta
                    )
                    matrix = self._low_rank_regular_mellin_matrix(
                        ell,
                        power1_template,
                        power_double_template,
                        layout.single_order,
                        layout.double_orders,
                        layout.constant_orders,
                        x,
                        theta,
                        regular_kernel,
                    )
                    factors = LOSMellinFactors(
                        z=z,
                        chi=chi,
                        weight=projector.weight(expression.sample_combination),
                        single_coefficients=amplitudes[:, None] * coefficients1,
                        double_coefficients=coefficients_double,
                        single_exponents=exponents1,
                        double_exponents=exponents_double,
                    )
                    started = time.perf_counter()
                    contribution += contract_low_rank_regular_mellin_matrix_los(
                        matrix, factors
                    )
                    self._record_timing(
                        "low_rank_contraction_seconds",
                        time.perf_counter() - started,
                    )
                else:  # guarded by SlepianConfig
                    raise ValueError("unsupported regular_method")
            if layout.transpose_output:
                contribution = contribution.T
            results[k] += (
                (-1j) ** effective.Sigma
                * contribution
                / (2.0 * np.pi) ** 2
            )
        return results

    def evaluate_modes(
        self,
        bispectrum,
        ell,
        theta,
        k_values,
        *,
        sigma=(0, 0, 0),
    ):
        """Return ZetaK arrays indexed by integer or half-integer mode."""
        if not isinstance(bispectrum, Bispectrum2D):
            raise TypeError("the Slepian route requires a Bispectrum2D")
        ell = np.asarray(ell, dtype=float)
        theta = np.asarray(theta, dtype=float)
        effective = as_effective_spin_triple(sigma)
        logger.debug(
            "Slepian modes: terms=%d modes=%d sigma=%s regular_method=%s",
            len(bispectrum.terms),
            len(k_values),
            effective.sigma,
            self.config.regular_method,
        )
        results = {
            float(k): np.zeros((theta.size, theta.size), dtype=complex)
            for k in k_values
        }
        geometry = self._geometry(theta, theta)
        for weighted_term in bispectrum.iter_terms():
            expression = weighted_term.term.get_representation(
                SlepianRepresentation2D
            )
            if isinstance(expression, ProjectedSlepianRepresentation2D):
                projected = self._evaluate_projected_expression(
                    expression,
                    ell,
                    theta,
                    k_values,
                    sigma,
                )
                for k, values in projected.items():
                    results[float(k)] += values
                continue
            if not hasattr(expression, "radial_factors"):
                raise TypeError("unsupported SlepianRepresentation2D implementation")
            _, power1 = self._factor_data(expression, 0, ell)
            coefficient = (
                self._weight_value(weighted_term.coefficient)
                * expression.coefficient
            )
            for raw_k in k_values:
                k = float(raw_k)
                m, n = effective.bessel_orders(k)
                layout = _slepian_leg_layout(expression, m, n, sigma)
                values_double, power_double = self._factor_data(
                    expression, layout.double_leg, ell
                )
                constant_kernel = self._constant_leg_kernel(
                    *layout.constant_orders, theta, theta
                )
                contribution = np.zeros_like(results[k])
                if constant_kernel.contact_coefficient:
                    radial1 = single_radial_transform(
                        power1, layout.single_order, theta
                    )
                    radial_double = double_radial_transform(
                        ell,
                        values_double,
                        power_double,
                        *layout.double_orders,
                        theta,
                        theta,
                        self.config,
                        geometry=geometry,
                        kernel_cache=self._weber_kernels,
                        interpolation_cache=self._weber_interpolators,
                    )
                    contribution += (
                        constant_kernel.contact_coefficient
                        * radial_double.T
                        * radial1[None, :]
                    )
                if constant_kernel.has_regular:
                    if not _regular_quadrature_supported(
                        *layout.constant_orders
                    ):
                        raise NotImplementedError(
                            "regular quadrature currently requires a positive "
                            "even difference between canonical Bessel orders"
                        )
                    if self.config.regular_method == "quadrature":
                        if self.config.regular_quadrature == "ratio_gauss":
                            contribution += self._regular_ratio_contribution(
                                ell,
                                values_double,
                                power1,
                                power_double,
                                layout,
                                theta,
                            )
                        else:
                            x = self._regular_x_grid(theta)
                            regular_kernel = self._constant_leg_kernel(
                                *layout.constant_orders, x, theta
                            )
                            radial1_x = single_radial_transform(
                                power1, layout.single_order, x
                            )
                            radial_double_x = double_radial_transform(
                                ell,
                                values_double,
                                power_double,
                                *layout.double_orders,
                                x,
                                theta,
                                self.config,
                                geometry=self._geometry(x, theta),
                                kernel_cache=self._weber_kernels,
                                interpolation_cache=self._weber_interpolators,
                            )
                            regular = (
                                regular_kernel.regular_less
                                + regular_kernel.regular_greater
                            )
                            integrand = (
                                x[:, None, None]
                                * radial1_x[:, None, None]
                                * radial_double_x[:, :, None]
                                * regular[:, None, :]
                            )
                            contribution += np.trapezoid(
                                integrand, x, axis=0
                            )
                    elif self.config.regular_method == "full_matrix":
                        x, regular_kernel = self._regular_matrix_inputs(
                            layout, theta
                        )
                        matrix = self._regular_mellin_matrix(
                            ell,
                            power1,
                            power_double,
                            layout.single_order,
                            layout.double_orders,
                            layout.constant_orders,
                            x,
                            theta,
                            regular_kernel,
                        )
                        contribution += contract_regular_mellin_matrix(
                            matrix,
                            power1.coefficients,
                            power_double.coefficients,
                        )
                    elif self.config.regular_method == "low_rank":
                        x, regular_kernel = self._regular_matrix_inputs(
                            layout, theta
                        )
                        matrix = self._low_rank_regular_mellin_matrix(
                            ell,
                            power1,
                            power_double,
                            layout.single_order,
                            layout.double_orders,
                            layout.constant_orders,
                            x,
                            theta,
                            regular_kernel,
                        )
                        started = time.perf_counter()
                        contribution += contract_low_rank_regular_mellin_matrix(
                            matrix,
                            power1.coefficients,
                            power_double.coefficients,
                        )
                        self._record_timing(
                            "low_rank_contraction_seconds",
                            time.perf_counter() - started,
                        )
                    else:  # guarded by SlepianConfig
                        raise ValueError("unsupported regular_method")
                if layout.transpose_output:
                    contribution = contribution.T
                results[k] += (
                    (-1j) ** effective.Sigma
                    * coefficient
                    * contribution
                    / (2.0 * np.pi) ** 2
                )
        return results
