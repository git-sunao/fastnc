"""Coefficient-level semi-analytic bispectrum multipoles."""
from __future__ import annotations

from dataclasses import dataclass
import logging

import numpy as np

from fastnc.bispectrum import Bispectrum2D
from fastnc.projection import ProjectedSemiAnalyticRepresentation2D


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SemiAnalyticConfig:
    """Numerical settings for the universal Appendix-C angular kernel."""

    angular_nodes: int = 1024

    def __post_init__(self):
        if int(self.angular_nodes) < 16:
            raise ValueError("angular_nodes must be at least 16")
        object.__setattr__(self, "angular_nodes", int(self.angular_nodes))


class SemiAnalyticCalculator:
    r"""Evaluate projected multipoles by LOS-integrating Mellin coefficients.

    For each projected representation this calculator first constructs
    ``d_n(ell2, ell3)`` from Eq. (C11), and only then contracts it with the
    cosmology-independent angular kernel ``K_L^(nu_n+p)`` in Eq. (C10).
    It never constructs a projected angular bispectrum on ``(ell1,ell2,ell3)``.
    """

    route = "semi_analytic"
    basis = "fourier"

    def __init__(self, config: SemiAnalyticConfig | None = None):
        self.config = SemiAnalyticConfig() if config is None else config
        if not isinstance(self.config, SemiAnalyticConfig):
            raise TypeError("config must be a SemiAnalyticConfig")
        self._kernel_cache = {}

    def clear_source_cache(self) -> None:
        """Clear source-dependent state while retaining universal kernels."""

    def clear_grid_cache(self) -> None:
        """Clear kernels tied to angular ratios or Mellin exponents."""
        self._kernel_cache.clear()

    def _angular_kernel(self, mode, exponent, ratio):
        ratio = np.asarray(ratio, dtype=float)
        key = (
            int(mode), complex(exponent), ratio.dtype.str, ratio.shape,
            np.ascontiguousarray(ratio).tobytes(),
        )
        cached = self._kernel_cache.get(key)
        if cached is not None:
            logger.debug("semi-analytic angular-kernel cache hit: mode=%d", int(mode))
            return cached
        nphi = self.config.angular_nodes
        phi = (np.arange(nphi, dtype=float) + 0.5) * (2.0 * np.pi / nphi)
        shape = (nphi,) + (1,) * ratio.ndim
        cosine = np.cos(phi).reshape(shape)
        phase = np.exp(-1j * int(mode) * phi).reshape(shape)
        s2 = 1.0 + ratio[None, ...] * cosine
        values = np.mean(
            np.power(s2.astype(complex), 0.5 * complex(exponent)) * phase,
            axis=0,
        )
        self._kernel_cache[key] = values
        logger.debug(
            "constructed semi-analytic angular kernel: mode=%d exponent=%s nodes=%d",
            int(mode),
            complex(exponent),
            nphi,
        )
        return values

    @staticmethod
    def _term_coefficient(weighted_term, z):
        coefficient = weighted_term.coefficient
        return coefficient(z) if callable(coefficient) else coefficient

    def _evaluate_representation(self, projected, modes, ell2, ell3):
        expression = projected.source_representation
        projector = projected.projector
        ell2, ell3 = np.broadcast_arrays(
            np.asarray(ell2, dtype=float), np.asarray(ell3, dtype=float)
        )
        q2 = ell2 + projector.shift
        q3 = ell3 + projector.shift
        scale = np.sqrt(q2**2 + q3**2)
        if np.any(scale <= 0.0):
            raise ValueError("semi-analytic angular scales must be positive")
        ratio = 2.0 * q2 * q3 / scale**2
        u = np.broadcast_to(
            expression.evaluate_u(q2 / scale, q3 / scale), scale.shape
        )

        los_samples = []
        for z, chi in zip(projector.z, projector.chi):
            coefficients = expression.coefficients(float(z))
            coefficient = self._term_coefficient(projected.source_term, float(z))
            v = expression.evaluate_v(q2 / chi, q3 / chi, float(z))
            los_samples.append(
                coefficient * v[..., None] * coefficients
                * np.power(float(chi), -expression.exponents)
            )
        integrated = projector.integrate_coefficients(
            np.stack(los_samples), axis=0,
            sample_combination=projected.sample_combination,
        )

        scale_powers = np.power(
            scale[..., None].astype(complex), expression.exponents
        )
        values = []
        for mode in modes:
            kernels = np.stack(
                [
                    self._angular_kernel(
                        int(mode), exponent + expression.power, ratio
                    )
                    for exponent in expression.exponents
                ], axis=-1,
            )
            values.append(u * np.sum(integrated * scale_powers * kernels, axis=-1))
        return np.stack(values)

    def evaluate(self, source, mode, *, ell2, ell3, **params):
        """Evaluate one or several Fourier multipoles of a projected source."""
        if params:
            names = ", ".join(sorted(params))
            raise TypeError(f"unused semi-analytic parameters: {names}")
        if not isinstance(source, Bispectrum2D):
            raise TypeError("source must be a Bispectrum2D")
        requested = np.asarray(mode)
        scalar = requested.ndim == 0
        if requested.ndim > 1 or requested.size == 0:
            raise ValueError("mode must be a scalar or non-empty 1D array")
        modes = np.atleast_1d(requested).astype(int, copy=False)
        logger.debug(
            "semi-analytic multipoles: terms=%d modes=%d ell_shape=%s",
            len(source.terms),
            modes.size,
            np.broadcast(np.asarray(ell2), np.asarray(ell3)).shape,
        )

        contributions = []
        for weighted in source.iter_terms():
            representations = [
                representation for representation in weighted.term.representations
                if isinstance(representation, ProjectedSemiAnalyticRepresentation2D)
            ]
            if not representations:
                raise TypeError(
                    f"term {weighted.term.name!r} has no projected "
                    "semi-analytic representation"
                )
            contributions.append(
                self._evaluate_representation(
                    representations[0], modes, ell2, ell3
                )
            )
        if not contributions:
            raise ValueError("source has no semi-analytic terms")
        result = sum(contributions)
        return result[0] if scalar else result
