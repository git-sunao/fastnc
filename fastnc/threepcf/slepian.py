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
    retained_rank: int
    relative_reconstruction_error: float

    def __post_init__(self):
        left = np.asarray(self.left_vectors, dtype=complex)
        singular = np.asarray(self.singular_values, dtype=float)
        right = np.asarray(self.right_vectors, dtype=complex)
        if left.ndim != 4 or singular.ndim != 3 or right.ndim != 4:
            raise ValueError("low-rank factors have incompatible dimensions")
        n_theta1, n_theta2, n_double, rank = left.shape
        if singular.shape != (n_theta1, n_theta2, rank):
            raise ValueError("singular_values have the wrong shape")
        if right.shape[:3] != (n_theta1, n_theta2, rank):
            raise ValueError("right_vectors have the wrong shape")
        object.__setattr__(self, "left_vectors", left)
        object.__setattr__(self, "singular_values", singular)
        object.__setattr__(self, "right_vectors", right)
        object.__setattr__(self, "retained_rank", int(self.retained_rank))
        object.__setattr__(
            self,
            "relative_reconstruction_error",
            float(self.relative_reconstruction_error),
        )


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


def _contact_coefficient(order_x: int, order_theta: int) -> int:
    """Return ``cos(pi * (order_x - order_theta) / 2)`` exactly."""
    difference = (int(order_x) - int(order_theta)) % 4
    if difference == 0:
        return 1
    if difference == 2:
        return -1
    return 0


def _regular_quadrature_supported(order_x: int, order_theta: int) -> bool:
    """Return whether the current direct quadrature has one-sided support."""
    difference = abs(abs(int(order_x)) - abs(int(order_theta)))
    return difference > 0 and difference % 2 == 0


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
        self._constant_leg_kernels: dict[tuple, ConstantLegKernel] = {}
        self._regular_mellin_matrices: dict[tuple, RegularMellinMatrix] = {}
        self._low_rank_regular_mellin_matrices: dict[
            tuple, LowRankRegularMellinMatrix
        ] = {}

    def clear(self) -> None:
        self._power_sums.clear()
        self._geometries.clear()
        self._weber_kernels.clear()
        self._weber_interpolators.clear()
        self._constant_leg_kernels.clear()
        self._regular_mellin_matrices.clear()
        self._low_rank_regular_mellin_matrices.clear()

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
        constant_kernel: ConstantLegKernel,
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
        constant_kernel: ConstantLegKernel,
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
            self.config.regular_low_rank_rank,
            self.config.regular_low_rank_rtol,
        )
        if key not in self._low_rank_regular_mellin_matrices:
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
            self._low_rank_regular_mellin_matrices[key] = (
                compress_regular_mellin_matrix(
                    full,
                    rank=self.config.regular_low_rank_rank,
                    rtol=self.config.regular_low_rank_rtol,
                )
            )
        return self._low_rank_regular_mellin_matrices[key]

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
                constant_kernel = self._constant_leg_kernel(
                    n3 - n, n, theta, theta
                )
                contribution = np.zeros_like(results[k])
                if constant_kernel.contact_coefficient:
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
                    contribution += (
                        constant_kernel.contact_coefficient
                        * radial2.T
                        * radial1[None, :]
                    )
                if constant_kernel.has_regular:
                    if not _regular_quadrature_supported(n3 - n, n):
                        raise NotImplementedError(
                            "regular quadrature currently requires a positive "
                            "even difference between canonical Bessel orders"
                        )
                    x = self._regular_x_grid(theta)
                    regular_kernel = self._constant_leg_kernel(
                        n3 - n, n, x, theta
                    )
                    if self.config.regular_method == "quadrature":
                        radial1_x = single_radial_transform(power1, n1, x)
                        radial2_x = double_radial_transform(
                            ell,
                            values2,
                            power2,
                            n2 - m,
                            m,
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
                            * radial2_x[:, :, None]
                            * regular[:, None, :]
                        )
                        contribution += np.trapezoid(integrand, x, axis=0)
                    elif self.config.regular_method == "full_matrix":
                        matrix = self._regular_mellin_matrix(
                            ell,
                            power1,
                            power2,
                            n1,
                            (n2 - m, m),
                            (n3 - n, n),
                            x,
                            theta,
                            regular_kernel,
                        )
                        contribution += contract_regular_mellin_matrix(
                            matrix,
                            power1.coefficients,
                            power2.coefficients,
                        )
                    elif self.config.regular_method == "low_rank":
                        matrix = self._low_rank_regular_mellin_matrix(
                            ell,
                            power1,
                            power2,
                            n1,
                            (n2 - m, m),
                            (n3 - n, n),
                            x,
                            theta,
                            regular_kernel,
                        )
                        contribution += contract_low_rank_regular_mellin_matrix(
                            matrix,
                            power1.coefficients,
                            power2.coefficients,
                        )
                    else:  # guarded by SlepianConfig
                        raise ValueError("unsupported regular_method")
                results[k] += coefficient * contribution / (2.0 * np.pi) ** 2
        return results
