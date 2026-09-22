"""Numeric LOS projection of a 3D callable on angular triangle inputs."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import angular_to_comoving, validate_los_coordinates


@dataclass(frozen=True)
class LOSValues:
    """Values sampled on flattened angular inputs and the LOS axis."""

    values: np.ndarray
    output_shape: tuple[int, ...]
    scalar: bool


def evaluate_numeric_los(
    evaluator,
    ell1,
    ell2,
    ell3,
    *,
    z,
    chi,
    shift: float = 0.0,
    **params,
) -> LOSValues:
    """Evaluate ``evaluator(k1, k2, k3, z)`` at all LOS nodes."""
    if not callable(evaluator):
        raise TypeError("evaluator must be callable as evaluator(k1, k2, k3, z)")
    z, chi = validate_los_coordinates(z, chi)
    scalar = all(np.ndim(value) == 0 for value in (ell1, ell2, ell3))
    ell1, ell2, ell3 = np.broadcast_arrays(
        np.asarray(ell1, dtype=float),
        np.asarray(ell2, dtype=float),
        np.asarray(ell3, dtype=float),
    )
    output_shape = ell1.shape
    k1 = angular_to_comoving(ell1, chi, shift=shift)
    k2 = angular_to_comoving(ell2, chi, shift=shift)
    k3 = angular_to_comoving(ell3, chi, shift=shift)
    z_grid = np.broadcast_to(z.reshape(1, -1), k1.shape)
    values = np.asarray(evaluator(k1, k2, k3, z_grid, **params))
    try:
        values = np.broadcast_to(values, k1.shape)
    except ValueError as exc:
        raise ValueError(
            "evaluator output must broadcast to (n_angular, n_los)"
        ) from exc
    return LOSValues(values=values, output_shape=output_shape, scalar=scalar)


def integrate_numeric_los(sampled: LOSValues, chi, *, weight=1.0):
    """Integrate sampled numeric values over the final LOS axis."""
    if not isinstance(sampled, LOSValues):
        raise TypeError("sampled must be an LOSValues object")
    chi = np.asarray(chi, dtype=float)
    if chi.ndim != 1 or sampled.values.shape[-1] != chi.size:
        raise ValueError("chi must be one-dimensional and match the sampled LOS axis")
    weight = np.asarray(weight, dtype=float)
    try:
        integrand = sampled.values * np.broadcast_to(weight, chi.shape)[None, :]
    except ValueError as exc:
        raise ValueError("weight must be scalar or have the same shape as chi") from exc
    result = np.trapezoid(integrand, chi, axis=-1).reshape(sampled.output_shape)
    return result.item() if sampled.scalar else result
