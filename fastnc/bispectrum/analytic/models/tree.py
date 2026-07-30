from __future__ import annotations

import numpy as np

from fastnc.hankel.wrapper import PowerLawFFTLogConfig
from ..angular import PowerLawAngularKernelTableConfig, _a31
from ..composite import (
    CompositeSemiAnalyticBispectrumMultipole3D,
    _safe_prefactor_core_product,
)
from ..fftlog import FFTLogComponent, _MutablePhysicalCallable
from ..terms import (DirectFourierTerm, LowRankVFunction, ProductVFunction, LeftVFunction, RightVFunction, SeparableMultipoleTerm)

def _tree23_coefficient(mode, linear_power):
    abs_mode = abs(int(mode))

    def coefficient(requested_mode, k2, k3, z):
        if abs(int(requested_mode)) != abs_mode:
            return np.zeros(np.broadcast(k2, k3).shape, dtype=float)

        p1 = linear_power(k2, z)
        p2 = linear_power(k3, z)

        if abs_mode == 0:
            prefactor = 12.0 / 7.0
        elif abs_mode == 1:
            prefactor = 0.5 * (k2 / k3 + k3 / k2)
        elif abs_mode == 2:
            prefactor = 1.0 / 7.0
        else:
            raise ValueError(f"Unsupported tree 23 mode: {mode}")

        return prefactor * p1 * p2

    return coefficient


class TreeBispectrumMultipole3D(CompositeSemiAnalyticBispectrumMultipole3D):
    """Predefined tree-level matter-bispectrum multipole model.

    The list of terms is intentionally hard-coded here.  Custom models should
    subclass :class:`CompositeSemiAnalyticBispectrumMultipole3D` and define a
    corresponding explicit list, rather than composing models after creation.
    """

    def __init__(self, linear_power, k_grid, *, fftlog_config: PowerLawFFTLogConfig | None = None,
                 angular_kernel_config: PowerLawAngularKernelTableConfig | None = None,
                 regularize_squeezed: bool = True):
        linear_power = (linear_power if isinstance(linear_power, _MutablePhysicalCallable)
                        else _MutablePhysicalCallable(linear_power))
        component = FFTLogComponent(
            name="linear_power",
            k_grid=np.asarray(k_grid, dtype=float),
            evaluator=linear_power,
            fftlog_config=fftlog_config or PowerLawFFTLogConfig(),
        )
        terms = [
            DirectFourierTerm("tree-23-F2", _tree23_coefficient(0, linear_power)),
            DirectFourierTerm("tree-23-F2", _tree23_coefficient(1, linear_power)),
            DirectFourierTerm("tree-23-F2", _tree23_coefficient(2, linear_power)),
            SeparableMultipoleTerm("tree-31-F2-p0", component, 0,
                                   lambda x2, x3: 2.0 * _a31(0, x2, x3),
                                   RightVFunction(lambda k, z: linear_power(k, z), name="P(k3)")),
            SeparableMultipoleTerm("tree-31-F2-p2", component, 2,
                                   lambda x2, x3: 2.0 * _a31(2, x2, x3),
                                   RightVFunction(lambda k, z: linear_power(k, z), name="P(k3)")),
            SeparableMultipoleTerm("tree-12-F2-p0", component, 0,
                                   lambda x2, x3: 2.0 * _a31(0, x3, x2),
                                   LeftVFunction(lambda k, z: linear_power(k, z), name="P(k2)")),
            SeparableMultipoleTerm("tree-12-F2-p2", component, 2,
                                   lambda x2, x3: 2.0 * _a31(2, x3, x2),
                                   LeftVFunction(lambda k, z: linear_power(k, z), name="P(k2)")),
        ]
        if regularize_squeezed:
            def ureg(x2, x3):
                return (x2*x2 - x3*x3)**2 / (14.0*x2*x2*x3*x3)
            def vreg(k2, k3, z):
                p1, p2 = linear_power(k2, z), linear_power(k3, z)
                ksq = k2*k2 + k3*k3
                pbar = 0.5*(p1 + p2)
                denom = k2*k2 - k3*k3
                # Stable local derivative on the diagonal.
                with np.errstate(divide="ignore", invalid="ignore"):
                    dp = (p1 - p2) / denom
                diagonal = np.isclose(k2, k3, rtol=1.0e-8, atol=0.0)
                if np.any(diagonal):
                    eps = 1.0e-4
                    kp = k2 * np.exp(eps)
                    km = k2 * np.exp(-eps)
                    deriv = (linear_power(kp, z) - linear_power(km, z)) / (kp - km)
                    dp = np.where(diagonal, deriv / (2.0*k2), dp)
                return 2.0*pbar - (k2**4 + 5.0*k2*k2*k3*k3 + k3**4) / ksq * dp
            terms.append(SeparableMultipoleTerm("tree-31plus12-regularized", component, -2, ureg, vreg))
        else:
            terms.extend([
                SeparableMultipoleTerm("tree-31-F2-pminus2", component, -2,
                                       lambda x2, x3: 2.0 * _a31(-2, x2, x3),
                                       RightVFunction(lambda k, z: linear_power(k, z), name="P(k3)")),
                SeparableMultipoleTerm("tree-12-F2-pminus2", component, -2,
                                       lambda x2, x3: 2.0 * _a31(-2, x3, x2),
                                       LeftVFunction(lambda k, z: linear_power(k, z), name="P(k2)")),
            ])
        super().__init__(terms, angular_kernel_config=angular_kernel_config)
        self.linear_power = linear_power
        self.k_grid = np.asarray(k_grid, dtype=float)
        self.regularize_squeezed = bool(regularize_squeezed)

        self._linear_power_component = component

    def update_physics(self, *, linear_power):
        """Replace the linear spectrum without rebuilding model structure."""
        self.linear_power.update(linear_power)
        self.invalidate_components((self._linear_power_component,))
        return self

    def warm_cache(self, *, z, modes, shifts=None):
        """Warm only non-negative modes for the parity-even tree model."""
        modes = np.unique(np.abs(np.atleast_1d(np.asarray(modes, dtype=int))))
        return super().warm_cache(z=z, modes=modes, shifts=shifts)

    def evaluate_modes(self, modes, k2, k3, z, *, chunk_size=4096, **params):
        """Generic-array evaluation exploiting ``B_{-L}=B_L`` for tree level."""
        modes = np.atleast_1d(np.asarray(modes, dtype=int))
        work_modes = np.unique(np.abs(modes))
        work = super().evaluate_modes(
            work_modes,
            k2,
            k3,
            z,
            chunk_size=chunk_size,
            **params,
        )
        inverse = np.searchsorted(work_modes, np.abs(modes))
        return work[inverse]

    @staticmethod
    def _axis_values(evaluator, axis, z):
        values = np.asarray(evaluator(axis, z), dtype=float)
        try:
            values = np.broadcast_to(values, axis.shape)
        except ValueError as error:
            raise ValueError(
                "linear_power must return values broadcastable to its k input"
            ) from error
        return np.asarray(values)

    def _axis_divided_difference(self, geometry, p1_axis, p2_axis, z):
        """Return ``[P(k2)-P(k3)]/(k2^2-k3^2)`` on the prepared grid."""
        k2 = geometry.k2_axis[:, None]
        k3 = geometry.k3_axis[None, :]
        p1 = p1_axis[:, None]
        p2 = p2_axis[None, :]
        denominator = k2 * k2 - k3 * k3
        with np.errstate(divide="ignore", invalid="ignore"):
            divided = (p1 - p2) / denominator

        diagonal = np.isclose(k2, k3, rtol=1.0e-12, atol=0.0)
        if np.any(diagonal):
            # This is the continuous diagonal limit
            # D_P(k,k) = [2k]^{-1} dP/dk.  The derivative is evaluated only
            # once on the k2 axis, rather than once per diagonal grid point.
            eps = 1.0e-4
            kp = geometry.k2_axis * np.exp(eps)
            km = geometry.k2_axis * np.exp(-eps)
            derivative = (
                self._axis_values(self.linear_power, kp, z)
                - self._axis_values(self.linear_power, km, z)
            ) / (kp - km)
            diagonal_limit = derivative[:, None] / (2.0 * k2)
            divided = np.where(diagonal, diagonal_limit, divided)
        return divided

    def evaluate_modes_grid(
        self,
        modes,
        k2_axis=None,
        k3_axis=None,
        z=None,
        *,
        geometry: TensorProductGeometryCache | None = None,
        **params,
    ):
        """Fast tree-level evaluation on a tensor-product Fourier grid.

        The method evaluates ``P(k2;z)`` and ``P(k3;z)`` only on their
        one-dimensional axes, reuses the exact ratio grouping for identical
        logarithmic axes, and assembles the hard-coded tree terms in a fused
        expression.  It returns an array with shape
        ``(len(modes), len(k2_axis), len(k3_axis))``.
        """
        if z is None:
            raise ValueError("z is required")
        geometry = self._resolve_grid_geometry(k2_axis, k3_axis, geometry)
        modes = np.atleast_1d(np.asarray(modes, dtype=int))
        work_modes = np.unique(np.abs(modes))

        p1_axis = self._axis_values(self.linear_power, geometry.k2_axis, z)
        p2_axis = self._axis_values(self.linear_power, geometry.k3_axis, z)
        p1 = p1_axis[:, None]
        p2 = p2_axis[None, :]
        n_mode = work_modes.size
        result = np.zeros((n_mode,) + geometry.shape, dtype=complex)

        # Finite Fourier contribution from the 12 permutation.  These terms
        # require no FFTLog expansion and are assembled directly from the
        # axis-cached power spectra.
        p12 = p1 * p2
        k2 = geometry.k2_axis[:, None]
        k3 = geometry.k3_axis[None, :]
        ratio = k2 / k3
        abs_modes = np.abs(work_modes)
        result[abs_modes == 0] += (12.0 / 7.0) * p12
        result[abs_modes == 1] += 0.5 * (ratio + 1.0 / ratio) * p12
        result[abs_modes == 2] += (1.0 / 7.0) * p12

        shifts = (0, 2, -2)
        cores = self._component_cores_grid(
            work_modes,
            self._linear_power_component,
            shifts,
            geometry,
            z,
        )

        # The p=0,+2 23 and 31 pieces share the same core for a fixed p.
        for shift in (0, 2):
            prefactor = (
                2.0 * _a31(shift, geometry.x2, geometry.x3) * p2
                + 2.0 * _a31(shift, geometry.x3, geometry.x2) * p1
            )
            result += _safe_prefactor_core_product(prefactor, cores[shift])

        if self.regularize_squeezed:
            ureg = (
                (geometry.x2 * geometry.x2 - geometry.x3 * geometry.x3) ** 2
                / (14.0 * geometry.x2 * geometry.x2 * geometry.x3 * geometry.x3)
            )
            dp = self._axis_divided_difference(geometry, p1_axis, p2_axis, z)
            pbar = 0.5 * (p1 + p2)
            k1sq = k2 * k2
            k2sq = k3 * k3
            vreg = 2.0 * pbar - (
                k1sq * k1sq + 5.0 * k1sq * k2sq + k2sq * k2sq
            ) / (geometry.k * geometry.k) * dp
            result += _safe_prefactor_core_product(ureg * vreg, cores[-2])
        else:
            prefactor = (
                2.0 * _a31(-2, geometry.x2, geometry.x3) * p2
                + 2.0 * _a31(-2, geometry.x3, geometry.x2) * p1
            )
            result += _safe_prefactor_core_product(prefactor, cores[-2])

        inverse = np.searchsorted(work_modes, np.abs(modes))
        return result[inverse]

# -----------------------------------------------------------------------------
# Predefined physical models: quadratic and tidal galaxy-bias contributions
# -----------------------------------------------------------------------------

