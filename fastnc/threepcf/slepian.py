"""Native-2D Slepian and Weber-Schafheitlin route kernels."""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Number

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.special import jv, loggamma, rgamma

from fastnc.bispectrum import Bispectrum2D, SlepianRepresentation2D

from .config import SlepianConfig


@dataclass(frozen=True)
class FFTLogPowerSum:
    """Finite complex-power representation of one sampled radial factor."""

    coefficients: np.ndarray
    exponents: np.ndarray


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

    def unique_ratios(self, *, omit_diagonal: bool):
        """Return unique log-ratios and indices reconstructing the full grid."""
        active = ~self.diagonal if omit_diagonal else np.ones_like(
            self.diagonal, dtype=bool
        )
        # Log-grid ratios that differ only by roundoff represent the same
        # scale separation. Quantization avoids duplicate hypergeometric calls.
        log_ratio = np.round(np.log(self.ratio[active]), decimals=14)
        unique_log_ratio, inverse = np.unique(log_ratio, return_inverse=True)
        return np.exp(unique_log_ratio), active, inverse


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


def _canonical_bessel_order(order: int) -> tuple[int, int]:
    order = int(order)
    if order >= 0:
        return order, 1
    positive = -order
    return positive, -1 if positive % 2 else 1


def _contact_sign(order_x: int, order_theta: int) -> int | None:
    canonical_x, sign_x = _canonical_bessel_order(order_x)
    canonical_theta, sign_theta = _canonical_bessel_order(order_theta)
    if canonical_x != canonical_theta:
        return None
    return sign_x * sign_theta


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


def _weber_unit_power(
    exponent, order_small: int, order_big: int, ratio: float, rtol: float
):
    mu, sign_small = _canonical_bessel_order(order_small)
    nu_big, sign_big = _canonical_bessel_order(order_big)
    lam = -complex(exponent) - 1.0
    A = (nu_big + mu - lam + 1.0) / 2.0
    B = (mu - nu_big - lam + 1.0) / 2.0
    C = mu + 1.0
    D = (nu_big - mu + lam + 1.0) / 2.0
    if ratio == 0.0 and mu > 0:
        return 0.0j
    prefactor = (
        ratio**mu
        * np.exp(-lam * np.log(2.0))
        * np.exp(loggamma(A))
        * rgamma(D)
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
):
    geometry = geometry or WeberGeometry.from_coordinates(x, theta)
    ratios, active, inverse = geometry.unique_ratios(
        omit_diagonal=omit_diagonal
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
    has_diagonal = np.any(geometry.diagonal)
    replace_diagonal = has_diagonal and config.diagonal_correction == "brute"
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
                bool(replace_diagonal),
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
                    omit_diagonal=replace_diagonal,
                    geometry=geometry,
                    method=config.weber_method,
                    interpolation_nodes=config.weber_interpolation_nodes,
                    interpolation_max_ratio=config.weber_interpolation_max_ratio,
                    interpolation_cache=interpolation_cache,
                )
                if kernel_cache is not None:
                    kernel_cache[cache_key] = kernel
            result += coefficient * kernel
    if replace_diagonal:
        tapered = np.asarray(values) * _log_edge_taper(
            len(ell), config.taper_fraction
        )
        brute = _double_radial_brute(
            ell, tapered, order_x, order_theta, x, theta
        )
        result[geometry.diagonal] = brute[geometry.diagonal]
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

    def clear(self) -> None:
        self._power_sums.clear()
        self._geometries.clear()
        self._weber_kernels.clear()
        self._weber_interpolators.clear()

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

    @staticmethod
    def _weight_value(weight):
        value = weight() if callable(weight) else weight
        if not isinstance(value, Number):
            raise TypeError("a 2D term coefficient must evaluate to a number")
        return value

    def evaluate_modes(self, bispectrum, ell, theta, k_values):
        """Return scalar ZetaK arrays indexed by integer opening-angle mode."""
        if not isinstance(bispectrum, Bispectrum2D):
            raise TypeError("the Slepian route requires a Bispectrum2D")
        ell = np.asarray(ell, dtype=float)
        theta = np.asarray(theta, dtype=float)
        results = {
            int(k): np.zeros((theta.size, theta.size), dtype=complex)
            for k in k_values
        }
        geometry = self._geometry(theta, theta)
        for weighted_term in bispectrum.iter_terms():
            expression = weighted_term.term.get_representation(
                SlepianRepresentation2D
            )
            if not hasattr(expression, "radial_factors"):
                raise TypeError("unsupported SlepianRepresentation2D implementation")
            if expression.constant_legs != (2,):
                raise NotImplementedError(
                    "the initial Slepian route requires exactly the third radial leg "
                    "to be constant"
                )
            n1, n2, n3 = expression.angular_orders
            _, power1 = self._factor_data(expression, 0, ell)
            values2, power2 = self._factor_data(expression, 1, ell)
            coefficient = (
                self._weight_value(weighted_term.coefficient)
                * expression.coefficient
            )
            for raw_k in k_values:
                k = int(raw_k)
                if not np.isclose(raw_k, k):
                    raise NotImplementedError(
                        "the initial scalar Slepian route supports integer k only"
                    )
                m, n = k, -k
                contact_sign = _contact_sign(n3 - n, n)
                if contact_sign is None:
                    raise NotImplementedError(
                        "non-contact constant-leg kernels are not implemented yet"
                    )
                radial1 = single_radial_transform(power1, n1, theta)
                radial2 = double_radial_transform(
                    ell,
                    values2,
                    power2,
                    n2 - m,
                    m,
                    theta,
                    theta,
                    self.config,
                    geometry=geometry,
                    kernel_cache=self._weber_kernels,
                    interpolation_cache=self._weber_interpolators,
                )
                results[k] += (
                    coefficient
                    * contact_sign
                    * radial2.T
                    * radial1[None, :]
                    / (2.0 * np.pi) ** 2
                )
        return results
