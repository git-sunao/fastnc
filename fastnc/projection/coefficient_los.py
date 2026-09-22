"""LOS integration for route-produced coefficient arrays."""
from __future__ import annotations

import numpy as np


def integrate_coefficients(coefficients, chi, *, weight=1.0, axis: int = -1):
    """Integrate coefficient samples over a designated LOS axis."""
    coefficients = np.asarray(coefficients)
    chi = np.asarray(chi, dtype=float)
    if chi.ndim != 1:
        raise ValueError("chi must be one-dimensional")
    axis = int(axis)
    if axis < 0:
        axis += coefficients.ndim
    if axis < 0 or axis >= coefficients.ndim:
        raise ValueError("axis is outside the coefficient array")
    if coefficients.shape[axis] != chi.size:
        raise ValueError("coefficient LOS axis must have the same size as chi")
    weight = np.asarray(weight, dtype=float)
    try:
        weight = np.broadcast_to(weight, chi.shape)
    except ValueError as exc:
        raise ValueError("weight must be scalar or have the same shape as chi") from exc
    shape = [1] * coefficients.ndim
    shape[axis] = chi.size
    return np.trapezoid(coefficients * weight.reshape(shape), chi, axis=axis)
