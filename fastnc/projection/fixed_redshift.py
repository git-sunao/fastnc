"""Fixed-redshift angularization of numeric 3D callables."""
from __future__ import annotations

import numpy as np


def angularize_numeric_3d(evaluator, *, z, chi):
    """Bind z and chi and return B2D(ell_i) = B3D(ell_i / chi, z)."""
    if not callable(evaluator):
        raise TypeError(
            "evaluator must be callable as evaluator(k1, k2, k3, z)"
        )
    z = float(z)
    chi = float(chi)
    if not np.isfinite(z):
        raise ValueError("z must be finite")
    if not np.isfinite(chi) or chi <= 0.0:
        raise ValueError("chi must be finite and positive")

    def angular_evaluator(ell1, ell2, ell3, **params):
        return evaluator(
            np.asarray(ell1) / chi,
            np.asarray(ell2) / chi,
            np.asarray(ell3) / chi,
            z,
            **params,
        )

    return angular_evaluator
