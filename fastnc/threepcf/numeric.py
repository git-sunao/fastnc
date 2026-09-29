"""Pure array kernels for the numeric B2D -> HKernel -> ZetaK route."""
from __future__ import annotations

import numpy as np


def contract_hkernel(
    fourier_multipoles,
    coupling_values,
) -> np.ndarray:
    r"""Contract bispectrum multipoles with the spin-coupling matrix.

    The leading axis is the angular-bispectrum mode :math:`L`, and the result
    is :math:`H_k(\ell_2,\ell_3)=\sum_L B_L(\ell_2,\ell_3)
    G_{Lk}(\sigma;\psi)`, where
    :math:`\psi=\arctan(\ell_3/\ell_2)`. No Fourier or radial measure is applied
    here; :class:`ThreePCF` inserts those factors before the Hankel transform.
    """
    multipoles = np.asarray(fourier_multipoles)
    coupling = np.asarray(coupling_values)
    if multipoles.ndim < 1:
        raise ValueError("fourier_multipoles must have a leading mode axis")
    if coupling.shape != multipoles.shape:
        raise ValueError(
            "coupling_values must have the same shape as fourier_multipoles"
        )
    return np.sum(multipoles * coupling, axis=0)
