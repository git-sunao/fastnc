"""Thin wrappers around the external two_Bessel implementation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import numpy as np

from .twobessel import two_Bessel


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
    """Evaluate a double Bessel transform using the external 2D FFTLog code.

    ``kernel`` is the complete Fourier-grid integrand passed to the transform.
    Complex kernels are handled by applying the real-valued external code to
    real and imaginary parts separately.
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
