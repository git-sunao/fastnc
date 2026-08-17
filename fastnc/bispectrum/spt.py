"""Direct standard-perturbation-theory bispectrum models.

This module contains ordinary configuration-space/Fourier-space evaluators of
SPT bispectra.  It is intentionally separate from :mod:`fastnc.bispectrum.analytic`,
whose models exploit separability to evaluate bispectrum multipoles directly.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from .base import Bispectrum3D
from .support import Support3D


def f2_kernel(k1, k2, mu12):
    r"""Return the symmetrized second-order SPT kernel :math:`F_2`.

    The Einstein--de Sitter form is used,

    .. math::

        F_2(\mathbf{k}_1,\mathbf{k}_2)
        = \frac{5}{7}
        + \frac{1}{2}\mu_{12}
          \left(\frac{k_1}{k_2}+\frac{k_2}{k_1}\right)
        + \frac{2}{7}\mu_{12}^2,

    where ``mu12`` is the cosine of the angle between the two wavevectors.
    Inputs follow NumPy broadcasting rules.
    """
    k1 = np.asarray(k1, dtype=float)
    k2 = np.asarray(k2, dtype=float)
    mu12 = np.asarray(mu12, dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio_sum = k1 / k2 + k2 / k1
        return 5.0 / 7.0 + 0.5 * mu12 * ratio_sum + 2.0 / 7.0 * mu12**2


def _pair_cosine(k1, k2, k3):
    r"""Cosine between ``k1`` and ``k2`` for a closed Fourier triangle.

    For :math:`\mathbf{k}_1+\mathbf{k}_2+\mathbf{k}_3=0`,

    .. math::

        \mu_{12}
        = \frac{k_3^2-k_1^2-k_2^2}{2 k_1 k_2}.
    """
    k1 = np.asarray(k1, dtype=float)
    k2 = np.asarray(k2, dtype=float)
    k3 = np.asarray(k3, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (k3**2 - k1**2 - k2**2) / (2.0 * k1 * k2)


class SPTMatterBispectrum3D(Bispectrum3D):
    r"""Direct tree-level real-space matter bispectrum in SPT.

    Parameters
    ----------
    linear_power : callable
        Callable ``linear_power(k, z)`` returning the linear matter power
        spectrum.  It should support NumPy-broadcastable ``k`` and ``z``.
    support : Support3D, optional
        Domain advertised to generic fastnc projection/interpolation machinery.
        If omitted, an unbounded support with policy ``"ignore"`` is used.

    Notes
    -----
    The model evaluates

    .. math::

        B_{mmm}^{\rm tree}(k_1,k_2,k_3;z)
        = 2F_2(\mathbf{k}_1,\mathbf{k}_2)P_L(k_1,z)P_L(k_2,z)
        + 2\ \mathrm{cyc.}

    directly for each triangle.  Unlike the semi-analytic SPT multipole model,
    no FFTLog/separable decomposition is used here.  This makes the class a
    useful independent reference for LOS projection and multipole validation.
    """

    def __init__(
        self,
        linear_power: Callable,
        support: Support3D | None = None,
    ):
        if not callable(linear_power):
            raise TypeError("linear_power must be callable as linear_power(k, z)")
        self.linear_power = linear_power
        self.support = support or Support3D(policy="ignore")

    def update_physics(self, *, linear_power: Callable):
        """Replace the linear power-spectrum evaluator in-place."""
        if not callable(linear_power):
            raise TypeError("linear_power must be callable as linear_power(k, z)")
        self.linear_power = linear_power
        return self

    def evaluate(self, k1, k2, k3, z, **params):
        k1 = np.asarray(k1, dtype=float)
        k2 = np.asarray(k2, dtype=float)
        k3 = np.asarray(k3, dtype=float)

        p1 = self.linear_power(k1, z)
        p2 = self.linear_power(k2, z)
        p3 = self.linear_power(k3, z)

        mu12 = _pair_cosine(k1, k2, k3)
        mu23 = _pair_cosine(k2, k3, k1)
        mu31 = _pair_cosine(k3, k1, k2)

        b12 = 2.0 * f2_kernel(k1, k2, mu12) * p1 * p2
        b23 = 2.0 * f2_kernel(k2, k3, mu23) * p2 * p3
        b31 = 2.0 * f2_kernel(k3, k1, mu31) * p3 * p1
        return b12 + b23 + b31
