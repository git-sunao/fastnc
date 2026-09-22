"""Pure Fourier decomposition of sampled angular bispectra."""
from __future__ import annotations

import numpy as np

from fastnc.bispectrum.decompose import MultipoleCosine


def triangle_closing_side(ell2, ell3, delta_beta):
    """Return ell1 = |ell2 + ell3| in the X1-reference convention."""
    ell2 = np.asarray(ell2, dtype=float)
    ell3 = np.asarray(ell3, dtype=float)
    delta_beta = np.asarray(delta_beta, dtype=float)
    ell1_squared = (
        ell2**2
        + ell3**2
        + 2.0 * ell2 * ell3 * np.cos(delta_beta)
    )
    lower = np.abs(ell2 - ell3)
    upper = ell2 + ell3
    return np.clip(np.sqrt(np.maximum(ell1_squared, 0.0)), lower, upper)


def decompose_angular_multipoles(
    values,
    delta_beta,
    modes,
    *,
    basis: str = "fourier-even",
    axis: int = -1,
    decomposition_angle: str = "outer",
    method: str = "gauss-legendre",
):
    """Project sampled side-only bispectra onto outer-angle Fourier modes.

    The returned mode axis is first. For fourier-even the coefficients satisfy
    B(delta) = c0 + sum cL cos(L delta). For fourier they are the full
    coefficients B_L; a side-only bispectrum is even, so B_L = B_-L.
    """
    values = np.asarray(values)
    delta_beta = np.asarray(delta_beta, dtype=float)
    modes = np.atleast_1d(np.asarray(modes, dtype=int))
    if modes.size == 0:
        raise ValueError("at least one multipole mode is required")
    axis = np.core.numeric.normalize_axis_index(axis, values.ndim)
    if delta_beta.ndim != 1 or delta_beta.size != values.shape[axis]:
        raise ValueError(
            "delta_beta must be one-dimensional and match the decomposition axis"
        )
    if np.any(np.diff(delta_beta) <= 0.0):
        raise ValueError("delta_beta must be strictly increasing")
    if basis == "fourier-even":
        if np.any(modes < 0):
            raise ValueError("fourier-even modes must be non-negative")
        projected_modes = modes
        normalization = np.where(modes == 0, 1.0 / np.pi, 2.0 / np.pi)
    elif basis == "fourier":
        projected_modes = np.abs(modes)
        normalization = np.full(modes.shape, 1.0 / np.pi)
    else:
        raise ValueError("basis must be 'fourier-even' or 'fourier'")

    if decomposition_angle == "outer":
        angle = delta_beta
        sampled = values
        sign = np.ones(modes.shape)
    elif decomposition_angle == "inner":
        angle = (np.pi - delta_beta)[::-1]
        sampled = np.flip(values, axis=axis)
        sign = (-1.0) ** modes
    else:
        raise ValueError("decomposition_angle must be 'outer' or 'inner'")

    raw = MultipoleCosine(
        angle,
        max(int(np.max(projected_modes)), 0),
        method=method,
    ).decompose(sampled, projected_modes, axis=axis)
    reshape = (modes.size,) + (1,) * (raw.ndim - 1)
    return raw * (normalization * sign).reshape(reshape)
