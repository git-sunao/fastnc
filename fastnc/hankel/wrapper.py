"""Thin wrappers around the external two_Bessel implementation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import numpy as np

from .twobessel import two_Bessel
from .fftlog import fftlog as _FFTLog1D


@dataclass(frozen=True)
class DoubleHankelConfig:
    """Configuration passed to the external ``two_Bessel`` class."""

    xy: float = 1.0
    nu1: float = 1.01
    nu2: float = 1.01
    N_extrap_low: int = 0
    N_extrap_high: int = 0
    c_window_width: float = 0.25
    N_pad: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    def as_kwargs(self) -> dict[str, Any]:
        out = {
            "xy": self.xy,
            "nu1": self.nu1,
            "nu2": self.nu2,
            "N_extrap_low": self.N_extrap_low,
            "N_extrap_high": self.N_extrap_high,
            "c_window_width": self.c_window_width,
            "N_pad": self.N_pad,
        }
        out.update(self.extra)
        return out


def _order_sign(order: int) -> int:
    """Return the sign from J_{-n}(x)=(-1)^n J_n(x) for integer order."""
    order = int(order)
    if order < 0:
        return -1 if (abs(order) % 2) else 1
    return 1


def _transform_real(ell1: np.ndarray, ell2: np.ndarray, kernel: np.ndarray, m: int, n: int, config: DoubleHankelConfig, *, bin_width_logtheta: float | None = None):
    tb = two_Bessel(ell1, ell2, kernel, **config.as_kwargs())
    if bin_width_logtheta is None:
        theta1, theta2, out = tb.two_Bessel(abs(int(m)), abs(int(n)))
    else:
        theta1, theta2, out = tb.two_Bessel_binave(abs(int(m)), abs(int(n)), bin_width_logtheta, bin_width_logtheta)
    out = out * _order_sign(int(m)) * _order_sign(int(n))
    return theta1, theta2, out


def double_hankel_transform(
    ell1: np.ndarray,
    ell2: np.ndarray,
    kernel: np.ndarray,
    m: int,
    n: int,
    *,
    config: DoubleHankelConfig | None = None,
    bin_width_logtheta: float | None = None,
):
    r"""Evaluate a double Bessel transform with the 2D FFTLog backend.

    The backend computes

    .. math::

       T_{mn}(\theta_1,\theta_2)=\int d\ln\ell_1\,d\ln\ell_2\,
       K(\ell_1,\ell_2)J_m(\ell_1\theta_1)J_n(\ell_2\theta_2).

    ``kernel`` must already contain every desired power of :math:`\ell_i` and
    physical normalization. Complex input is transformed through separate real
    and imaginary calls. Negative integer orders use
    :math:`J_{-n}(x)=(-1)^nJ_n(x)`. If ``bin_width_logtheta`` is given, both
    output radii are logarithmically bin-averaged before return.

    Returns:
        ``(theta1, theta2, transformed)`` on the backend FFTLog grid.
    """
    cfg = config or DoubleHankelConfig()
    ell1 = np.asarray(ell1, dtype=float)
    ell2 = np.asarray(ell2, dtype=float)
    kernel = np.asarray(kernel)

    if np.iscomplexobj(kernel):
        t1, t2, real = _transform_real(ell1, ell2, np.asarray(kernel.real, dtype=float), int(m), int(n), cfg, bin_width_logtheta=bin_width_logtheta)
        _, _, imag = _transform_real(ell1, ell2, np.asarray(kernel.imag, dtype=float), int(m), int(n), cfg, bin_width_logtheta=bin_width_logtheta)
        return t1, t2, real + 1j * imag

    return _transform_real(ell1, ell2, np.asarray(kernel, dtype=float), int(m), int(n), cfg, bin_width_logtheta=bin_width_logtheta)


@dataclass(frozen=True)
class PowerLawFFTLogConfig:
    r"""Configure a discrete complex-power representation of a 1D function.

    The returned coefficients satisfy approximately

    .. math:: f(x)\simeq\sum_m c_m x^{\nu_m},\qquad \nu_m=q+i\eta_m.

    on the logarithmic input interval.  This is a lightweight interface for
    power-law decompositions used outside Hankel transforms, e.g. analytic
    angular kernels for bispectrum multipoles.  Internally this delegates to
    ``fastnc.hankel.fftlog.fftlog``, so the 1D FFTLog convention is shared
    with the Hankel-transform machinery.
    """

    bias: float = 0.0
    c_window_width: float = 0.0
    N_extrap_low: int = 0
    N_extrap_high: int = 0
    N_pad: int = 0


def power_law_fftlog_coefficients(x: np.ndarray, fx: np.ndarray, config: PowerLawFFTLogConfig | None = None):
    r"""Return the full complex FFTLog power-law expansion of ``fx``.

    For :math:`x_j=x_0e^{j\Delta}` and real bias :math:`q`, this transforms
    :math:`f(x_j)x_j^{-q}` and returns both frequency signs with

    .. math::

       \eta_m={2\pi m\over N\Delta},\qquad
       f(x)\simeq\sum_m c_m x^{q+i\eta_m}.

    The factor :math:`e^{-i\eta_m\ln x_0}/N` is included in ``c_m``. Callers
    must not apply another FFT normalization or origin phase. Windowing,
    extrapolation, and padding alter the numerical representation but not this
    convention.

    Args:
        x, fx: Matching positive logarithmic samples and real function values.
        config: Bias and endpoint-control settings.

    Returns:
        ``(coefficients, exponents)`` ordered by ``numpy.fft.fftfreq``.
    """
    cfg = config or PowerLawFFTLogConfig()
    x = np.asarray(x, dtype=float)
    fx = np.asarray(fx, dtype=float)
    if x.ndim != 1 or fx.ndim != 1 or x.size != fx.size:
        raise ValueError("x and fx must be one-dimensional arrays with the same length")
    if x.size < 2:
        raise ValueError("at least two samples are required")
    if np.any(x <= 0.0):
        raise ValueError("x must be positive")

    fl = _FFTLog1D(
        x,
        fx,
        nu=float(cfg.bias),
        N_extrap_low=int(cfg.N_extrap_low),
        N_extrap_high=int(cfg.N_extrap_high),
        c_window_width=float(cfg.c_window_width),
        N_pad=int(cfg.N_pad),
    )

    N = int(fl.N)
    dlnx = float(fl.dlnx)
    lnx0 = float(np.log(fl.x[0]))

    rcoeff = np.asarray(fl.c_m, dtype=complex)
    full = np.empty(N, dtype=complex)
    full[: N // 2 + 1] = rcoeff
    if N > 2:
        full[N // 2 + 1 :] = np.conj(rcoeff[1 : N // 2][::-1])

    eta = 2.0 * np.pi * np.fft.fftfreq(N, d=dlnx)
    coeff = full / N * np.exp(-1j * eta * lnx0)
    nu = float(cfg.bias) + 1j * eta
    return coeff, nu
