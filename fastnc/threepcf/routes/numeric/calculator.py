"""Callable-based numeric angular-multipole calculator."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import NumericMultipoleConfig
from .multipoles import decompose_angular_multipoles, triangle_closing_side


@dataclass(frozen=True)
class AngularBispectrumSamples:
    """Passive storage for angular triangles and sampled bispectrum values."""

    delta_beta: np.ndarray
    ell1: np.ndarray
    ell2: np.ndarray
    ell3: np.ndarray
    values: np.ndarray
    output_shape: tuple[int, ...]


class NumericMultipoleCalculator:
    """Evaluate a 2D callable and decompose it into Fourier multipoles."""

    def __init__(self, config: NumericMultipoleConfig | None = None):
        self.config = config or NumericMultipoleConfig()
        if not isinstance(self.config, NumericMultipoleConfig):
            raise TypeError("config must be a NumericMultipoleConfig")

    def sample(self, evaluator, ell2, ell3, **params):
        if not callable(evaluator):
            raise TypeError(
                "evaluator must be callable as evaluator(ell1, ell2, ell3)"
            )
        ell2, ell3 = np.broadcast_arrays(
            np.asarray(ell2, dtype=float),
            np.asarray(ell3, dtype=float),
        )
        if np.any(ell2 <= 0.0) or np.any(ell3 <= 0.0):
            raise ValueError("ell2 and ell3 must be positive")
        output_shape = ell2.shape
        delta_beta = self.config.delta_beta
        angle_shape = (1,) * ell2.ndim + (delta_beta.size,)
        sampled_ell2 = np.broadcast_to(
            ell2[..., None], output_shape + (delta_beta.size,)
        )
        sampled_ell3 = np.broadcast_to(
            ell3[..., None], output_shape + (delta_beta.size,)
        )
        sampled_delta = delta_beta.reshape(angle_shape)
        ell1 = triangle_closing_side(
            sampled_ell2,
            sampled_ell3,
            sampled_delta,
        )
        values = np.asarray(
            evaluator(ell1, sampled_ell2, sampled_ell3, **params)
        )
        try:
            values = np.broadcast_to(values, ell1.shape)
        except ValueError as exc:
            raise ValueError(
                "evaluator output must broadcast to the sampled triangle shape"
            ) from exc
        return AngularBispectrumSamples(
            delta_beta=delta_beta,
            ell1=ell1,
            ell2=sampled_ell2,
            ell3=sampled_ell3,
            values=values,
            output_shape=output_shape,
        )

    def decompose(self, sampled: AngularBispectrumSamples, modes=None):
        if not isinstance(sampled, AngularBispectrumSamples):
            raise TypeError("sampled must be an AngularBispectrumSamples object")
        modes = self.config.modes if modes is None else modes
        return decompose_angular_multipoles(
            sampled.values,
            sampled.delta_beta,
            modes,
            basis=self.config.basis,
            axis=-1,
            decomposition_angle=self.config.decomposition_angle,
            method=self.config.method,
        )

    def evaluate(self, evaluator, ell2, ell3, *, modes=None, **params):
        sampled = self.sample(evaluator, ell2, ell3, **params)
        return self.decompose(sampled, modes=modes)
