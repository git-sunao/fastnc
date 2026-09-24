"""Numeric production of angular bispectrum multipoles."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import NumericMultipoleConfig
from .decompose import MultipoleCosine, MultipoleLegendre, MultipoleSine


def triangle_closing_side(ell2, ell3, delta_beta):
    """Return ``ell1 = |ell2 + ell3|`` in the outer-angle convention."""
    ell2 = np.asarray(ell2, dtype=float)
    ell3 = np.asarray(ell3, dtype=float)
    delta_beta = np.asarray(delta_beta, dtype=float)
    ell1_squared = ell2**2 + ell3**2 + 2.0 * ell2 * ell3 * np.cos(delta_beta)
    lower = np.abs(ell2 - ell3)
    upper = ell2 + ell3
    return np.clip(np.sqrt(np.maximum(ell1_squared, 0.0)), lower, upper)


def decompose_angular_multipoles(
    values,
    delta_beta,
    modes,
    *,
    basis: str = "cosine",
    axis: int = -1,
    decomposition_angle: str = "outer",
    method: str = "gauss-legendre",
):
    """Project sampled angular bispectra onto a specified angular basis.

    ``fourier`` denotes the coefficients of the even full-angle extension, so
    positive and negative modes are equal and are evaluated with cosine
    projection on ``[0, pi]``.
    """
    values = np.asarray(values)
    delta_beta = np.asarray(delta_beta, dtype=float)
    modes = np.atleast_1d(np.asarray(modes, dtype=int))
    basis = "cosine" if basis == "fourier-even" else basis
    if modes.size == 0:
        raise ValueError("at least one multipole mode is required")
    axis = np.core.numeric.normalize_axis_index(axis, values.ndim)
    if delta_beta.ndim != 1 or delta_beta.size != values.shape[axis]:
        raise ValueError(
            "delta_beta must be one-dimensional and match the decomposition axis"
        )
    if np.any(np.diff(delta_beta) <= 0.0):
        raise ValueError("delta_beta must be strictly increasing")
    if basis == "cosine":
        if np.any(modes < 0):
            raise ValueError("cosine modes must be non-negative")
        projected_modes = modes
        normalization = np.where(modes == 0, 1.0 / np.pi, 2.0 / np.pi)
        decomposer_type = MultipoleCosine
    elif basis == "sine":
        if np.any(modes <= 0):
            raise ValueError("sine modes must be positive")
        projected_modes = modes
        normalization = np.full(modes.shape, 2.0 / np.pi)
        decomposer_type = MultipoleSine
    elif basis == "fourier":
        projected_modes = np.abs(modes)
        normalization = np.full(modes.shape, 1.0 / np.pi)
        decomposer_type = MultipoleCosine
    elif basis == "legendre":
        if np.any(modes < 0):
            raise ValueError("Legendre modes must be non-negative")
        projected_modes = modes
        normalization = np.ones(modes.shape)
        decomposer_type = MultipoleLegendre
    else:
        raise ValueError(
            "basis must be 'cosine', 'sine', 'fourier', or 'legendre'"
        )

    if decomposition_angle == "outer":
        angle = delta_beta
        sampled = values
        sign = np.ones(modes.shape)
    elif decomposition_angle == "inner":
        angle = (np.pi - delta_beta)[::-1]
        sampled = np.flip(values, axis=axis)
        exponent = modes + 1 if basis == "sine" else modes
        sign = (-1.0) ** exponent
    else:
        raise ValueError("decomposition_angle must be 'outer' or 'inner'")

    coordinate = angle
    decomposer_values = sampled
    if basis == "legendre":
        coordinate = np.cos(angle)[::-1]
        decomposer_values = np.flip(sampled, axis=axis)

    raw = decomposer_type(
        coordinate,
        max(int(np.max(projected_modes)), 0),
        method=method,
    ).decompose(decomposer_values, projected_modes, axis=axis)
    reshape = (modes.size,) + (1,) * (raw.ndim - 1)
    return raw * (normalization * sign).reshape(reshape)


@dataclass(frozen=True)
class AngularBispectrumSamples:
    """Passive storage for angular triangles and sampled bispectrum values."""

    delta_beta: np.ndarray
    ell1: np.ndarray
    ell2: np.ndarray
    ell3: np.ndarray
    values: np.ndarray
    output_shape: tuple[int, ...]


class NumericBispectrumMultipoleCalculator:
    """Sample a numeric 2D bispectrum and decompose its angular dependence."""

    route = "numeric"

    def __init__(
        self,
        config: NumericMultipoleConfig | None = None,
        *,
        basis: str = "cosine",
    ):
        self.config = config or NumericMultipoleConfig()
        if not isinstance(self.config, NumericMultipoleConfig):
            raise TypeError("config must be a NumericMultipoleConfig")
        basis = "cosine" if basis == "fourier-even" else str(basis)
        if basis not in {"cosine", "sine", "fourier", "legendre"}:
            raise ValueError(
                "basis must be 'cosine', 'sine', 'fourier', or 'legendre'"
            )
        self.basis = basis

    def sample(self, evaluator, ell2, ell3, **params):
        if not callable(evaluator):
            raise TypeError(
                "source must be callable as source(ell1, ell2, ell3)"
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
        ell1 = triangle_closing_side(
            sampled_ell2,
            sampled_ell3,
            delta_beta.reshape(angle_shape),
        )
        values = np.asarray(
            evaluator(ell1, sampled_ell2, sampled_ell3, **params)
        )
        try:
            values = np.broadcast_to(values, ell1.shape)
        except ValueError as exc:
            raise ValueError(
                "source output must broadcast to the sampled triangle shape"
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
        if modes is None:
            raise ValueError("modes must be specified by BispectrumMultipole")
        modes = np.asarray(modes, dtype=int)
        values = decompose_angular_multipoles(
            sampled.values,
            sampled.delta_beta,
            modes,
            basis=self.basis,
            axis=-1,
            decomposition_angle=self.config.decomposition_angle,
            method=self.config.method,
        )
        return values

    def evaluate(self, source, mode, *, ell2, ell3, **params):
        requested = np.asarray(mode)
        scalar_mode = requested.ndim == 0
        if requested.ndim > 1 or requested.size == 0:
            raise ValueError("mode must be a scalar or non-empty 1D array")
        if not np.issubdtype(requested.dtype, np.integer):
            raise TypeError("mode values must be integers")
        modes = np.atleast_1d(requested).astype(int, copy=False)
        sampled = self.sample(source, ell2, ell3, **params)
        values = self.decompose(sampled, modes=modes)
        return values[0] if scalar_mode else values
