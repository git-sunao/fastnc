"""Pure array kernels for the numeric B2D -> HKernel -> ZetaK route."""
from __future__ import annotations

import numpy as np


def contract_hkernel(
    fourier_multipoles,
    coupling_values,
) -> np.ndarray:
    """Contract the Fourier-mode axis of B_L with G_Lk."""
    multipoles = np.asarray(fourier_multipoles)
    coupling = np.asarray(coupling_values)
    if multipoles.ndim < 1:
        raise ValueError("fourier_multipoles must have a leading mode axis")
    if coupling.shape != multipoles.shape:
        raise ValueError(
            "coupling_values must have the same shape as fourier_multipoles"
        )
    return np.sum(multipoles * coupling, axis=0)
