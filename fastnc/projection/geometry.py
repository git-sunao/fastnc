"""Pure flat-sky geometry used by line-of-sight projection."""
from __future__ import annotations

import numpy as np


def validate_los_coordinates(z, chi) -> tuple[np.ndarray, np.ndarray]:
    """Return validated one-dimensional LOS coordinates."""
    z = np.asarray(z, dtype=float)
    chi = np.asarray(chi, dtype=float)
    if z.ndim != 1 or chi.ndim != 1 or z.shape != chi.shape:
        raise ValueError("z and chi must be one-dimensional arrays with the same shape")
    if z.size < 2:
        raise ValueError("z and chi must contain at least two samples")
    if np.any(~np.isfinite(z)) or np.any(~np.isfinite(chi)):
        raise ValueError("z and chi must be finite")
    if np.any(chi <= 0.0) or np.any(np.diff(chi) <= 0.0):
        raise ValueError("chi must be positive and strictly increasing")
    return z, chi


def angular_to_comoving(ell, chi, *, shift: float = 0.0) -> np.ndarray:
    """Evaluate ``k = (ell + shift) / chi`` on an angular-by-LOS grid."""
    ell = np.asarray(ell, dtype=float)
    chi = np.asarray(chi, dtype=float)
    if chi.ndim != 1:
        raise ValueError("chi must be one-dimensional")
    return (ell.reshape(-1, 1) + float(shift)) / chi.reshape(1, -1)
