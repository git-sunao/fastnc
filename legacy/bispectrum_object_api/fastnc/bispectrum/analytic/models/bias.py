from __future__ import annotations

from typing import Mapping
import numpy as np

from fastnc.hankel.wrapper import PowerLawFFTLogConfig
from ..angular import PowerLawAngularKernelTableConfig, _a31, _t31
from ..composite import CompositeSemiAnalyticBispectrumMultipole3D
from ..fftlog import FFTLogComponent, _MutablePhysicalCallable
from ..terms import (DirectFourierTerm, LowRankVFunction, ProductVFunction, LeftVFunction, RightVFunction, SeparableMultipoleTerm)

def _value_at_z(value, z):
    """Evaluate a scalar bias/coefficient or a callable ``value(z)``."""
    return value(z) if callable(value) else value


def _pair_coefficients(coefficients):
    """Normalize pair coefficients to the ordered tuple ``(12, 23, 31)``."""
    if isinstance(coefficients, Mapping):
        unknown = set(coefficients) - {"12", "23", "31"}
        if unknown:
            raise ValueError(f"Unknown pair coefficient(s): {sorted(unknown)}")
        return tuple(coefficients.get(pair, 0.0) for pair in ("12", "23", "31"))
    values = tuple(coefficients)
    if len(values) != 3:
        raise ValueError("pair_coefficients must contain (C12, C23, C31)")
    return values


def _bias23_direct_coefficient(mode, linear_power, coefficient, fourier_factor):
    target = abs(int(mode))

    def evaluate(requested_mode, k2, k3, z):
        if abs(int(requested_mode)) != target:
            return np.zeros(np.broadcast(k2, k3).shape, dtype=float)
        return (
            _value_at_z(coefficient, z)
            * fourier_factor
            * linear_power(k2, z)
            * linear_power(k3, z)
        )

    return evaluate




class QuadraticBiasBispectrumMultipole3D(CompositeSemiAnalyticBispectrumMultipole3D):
    r"""Semi-analytic multipoles of the local quadratic-bias contribution.

    The represented bispectrum is

    .. math::
       B_{b_2}=C_{12}P_1P_2+C_{23}P_2P_3+C_{31}P_3P_1.

    The coefficients may be scalars or callables ``C_ij(z)``.  They include
    every physical prefactor, such as ``b2`` and the linear-bias factors on the
    two first-order legs.  This class intentionally contains no matter-tree or
    tidal contribution, so it can be tested and combined independently.
    """

    def __init__(self, linear_power, k_grid, *, pair_coefficients=(0.0, 0.0, 0.0),
                 fftlog_config: PowerLawFFTLogConfig | None = None,
                 angular_kernel_config: PowerLawAngularKernelTableConfig | None = None):
        linear_power = (linear_power if isinstance(linear_power, _MutablePhysicalCallable)
                        else _MutablePhysicalCallable(linear_power))
        c12, c23, c31 = _pair_coefficients(pair_coefficients)
        component = FFTLogComponent(
            name="linear_power_quadratic_bias",
            k_grid=np.asarray(k_grid, dtype=float),
            evaluator=linear_power,
            fftlog_config=fftlog_config or PowerLawFFTLogConfig(),
        )
        terms = [
            DirectFourierTerm(
                "quadratic-bias-23-L0",
                _bias23_direct_coefficient(0, linear_power, c23, 1.0),
            ),
            SeparableMultipoleTerm(
                "quadratic-bias-31",
                component,
                0,
                lambda x2, x3: np.ones(np.broadcast(x2, x3).shape),
                RightVFunction(
                    lambda k, z: _value_at_z(c31, z) * linear_power(k, z),
                    name="C31(z) P(k3)",
                ),
            ),
            SeparableMultipoleTerm(
                "quadratic-bias-12",
                component,
                0,
                lambda x2, x3: np.ones(np.broadcast(x2, x3).shape),
                LeftVFunction(
                    lambda k, z: _value_at_z(c12, z) * linear_power(k, z),
                    name="C12(z) P(k2)",
                ),
            ),
        ]
        super().__init__(terms, angular_kernel_config=angular_kernel_config)
        self.linear_power = linear_power
        self.k_grid = np.asarray(k_grid, dtype=float)
        self.pair_coefficients = (c12, c23, c31)
        self._linear_power_component = component

    def update_physics(self, *, linear_power):
        self.linear_power.update(linear_power)
        self.invalidate_components((self._linear_power_component,))
        return self


class TidalBiasBispectrumMultipole3D(CompositeSemiAnalyticBispectrumMultipole3D):
    r"""Semi-analytic multipoles of the tidal-bias contribution.

    The represented bispectrum is

    .. math::
       B_{K^2}=C_{12}S_{12}P_1P_2
                +C_{23}S_{23}P_2P_3
                +C_{31}S_{31}P_3P_1.

    ``C_ij`` contains the complete physical prefactor, conventionally
    ``2 b_K2`` times the linear-bias factors on the first-order legs.
    """

    def __init__(self, linear_power, k_grid, *, pair_coefficients=(0.0, 0.0, 0.0),
                 fftlog_config: PowerLawFFTLogConfig | None = None,
                 angular_kernel_config: PowerLawAngularKernelTableConfig | None = None):
        linear_power = (linear_power if isinstance(linear_power, _MutablePhysicalCallable)
                        else _MutablePhysicalCallable(linear_power))
        c12, c23, c31 = _pair_coefficients(pair_coefficients)
        component = FFTLogComponent(
            name="linear_power_tidal_bias",
            k_grid=np.asarray(k_grid, dtype=float),
            evaluator=linear_power,
            fftlog_config=fftlog_config or PowerLawFFTLogConfig(),
        )
        terms = [
            DirectFourierTerm(
                "tidal-bias-23-L0",
                _bias23_direct_coefficient(0, linear_power, c23, 1.0 / 6.0),
            ),
            DirectFourierTerm(
                "tidal-bias-23-L2",
                _bias23_direct_coefficient(2, linear_power, c23, 1.0 / 4.0),
            ),
        ]
        for pair, coefficient, swap, power_leg in (
            ("31", c31, False, 3),
            ("12", c12, True, 2),
        ):
            for shift in (0, 2, -2):
                if swap:
                    u = lambda x2, x3, p=shift: _t31(p, x3, x2)
                else:
                    u = lambda x2, x3, p=shift: _t31(p, x2, x3)
                if power_leg == 3:
                    v = RightVFunction(
                        lambda k, z, c=coefficient: _value_at_z(c, z) * linear_power(k, z),
                        name=f"C{pair}(z) P(k3)",
                    )
                else:
                    v = LeftVFunction(
                        lambda k, z, c=coefficient: _value_at_z(c, z) * linear_power(k, z),
                        name=f"C{pair}(z) P(k2)",
                    )
                terms.append(SeparableMultipoleTerm(
                    f"tidal-bias-{pair}-p{shift:+d}", component, shift, u, v
                ))
        super().__init__(terms, angular_kernel_config=angular_kernel_config)
        self.linear_power = linear_power
        self.k_grid = np.asarray(k_grid, dtype=float)
        self.pair_coefficients = (c12, c23, c31)
        self._linear_power_component = component

    def update_physics(self, *, linear_power):
        self.linear_power.update(linear_power)
        self.invalidate_components((self._linear_power_component,))
        return self


