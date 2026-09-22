"""Direct standard-perturbation-theory bispectrum models.

This module contains direct triangle-space reference implementations expressed
as typed term aggregates.
"""
from __future__ import annotations

from typing import Callable, Mapping

import numpy as np

from ..bispectrum import Bispectrum3D
from ..representations import NumericExpression3D
from ..support import Support3D
from ..terms import BispectrumTerm3D, WeightedTerm3D
from fastnc.utils.cosmology import (
    default_wmap_like_cosmology,
    eisenstein_hu_like_pklin,
    simple_debug_pklin,
    simple_linear_growth,
)


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


def tidal_kernel(mu):
    r"""Return the quadratic tidal-bias kernel :math:`S_2`.

    The convention matches the semi-analytic fastnc bias model,

    .. math::

        S_2(\mathbf{k}_1,\mathbf{k}_2) = \mu_{12}^2 - \frac{1}{3}.

    Inputs follow NumPy broadcasting rules.
    """
    mu = np.asarray(mu, dtype=float)
    return mu**2 - 1.0 / 3.0


def _value_at_z(value, z):
    """Evaluate a scalar bias parameter or a callable ``value(z)``."""
    return value(z) if callable(value) else value


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
        terms = (
            BispectrumTerm3D(
                name="tree:F2:12",
                representations=(NumericExpression3D(self._evaluate_12),),
            ),
            BispectrumTerm3D(
                name="tree:F2:23",
                representations=(NumericExpression3D(self._evaluate_23),),
            ),
            BispectrumTerm3D(
                name="tree:F2:31",
                representations=(NumericExpression3D(self._evaluate_31),),
            ),
        )
        super().__init__(terms, support=support)

    @classmethod
    def simple_debug(
        cls,
        *,
        cosmo: Mapping[str, float] | None = None,
        amplitude: float = 1.0e4,
        k_eq: float = 2.0e-2,
        transfer_power: float = 1.5,
        pklin_kind: str = "debug",
        support: Support3D | None = None,
    ) -> "SPTMatterBispectrum3D":
        """Return a self-initialized tree-level SPT matter model for debugging.

        The debug linear power spectrum is defined at ``z=0`` and promoted to
        arbitrary redshift using ``P_L(k,z) = D(z)^2 P_L(k,0)`` with the simple
        bundled growth factor.  These ingredients are intended for examples and
        validation only, not for scientific inference.

        Parameters mirror :meth:`BiHalofitBispectrum3D.simple_debug` where they
        are relevant.  Unlike Bihalofit, direct SPT evaluates the power spectrum
        callable on demand and therefore does not require pre-tabulated ``k`` or
        ``z`` grids.
        """
        cosmo_dict = default_wmap_like_cosmology() if cosmo is None else dict(cosmo)

        if pklin_kind == "debug":
            def pklin0(k):
                return simple_debug_pklin(
                    k,
                    cosmo=cosmo_dict,
                    amplitude=amplitude,
                    k_eq=k_eq,
                    transfer_power=transfer_power,
                )
        elif pklin_kind in {"eisenstein-hu", "eisenstein_hu", "eh"}:
            def pklin0(k):
                return eisenstein_hu_like_pklin(
                    k,
                    cosmo=cosmo_dict,
                    amplitude=amplitude,
                )
        else:
            raise ValueError("pklin_kind must be 'debug' or 'eisenstein-hu'.")

        def linear_power(k, z):
            growth = simple_linear_growth(z, cosmo=cosmo_dict)
            return pklin0(k) * growth**2

        return cls(linear_power=linear_power, support=support)

    def update_physics(self, *, linear_power: Callable):
        """Replace the linear power-spectrum evaluator in-place."""
        if not callable(linear_power):
            raise TypeError("linear_power must be callable as linear_power(k, z)")
        self.linear_power = linear_power
        self._state_updated()
        return self

    def _evaluate_pair(self, k_left, k_right, k_closing, z):
        k_left = np.asarray(k_left, dtype=float)
        k_right = np.asarray(k_right, dtype=float)
        k_closing = np.asarray(k_closing, dtype=float)
        mu = _pair_cosine(k_left, k_right, k_closing)
        return (
            2.0
            * f2_kernel(k_left, k_right, mu)
            * self.linear_power(k_left, z)
            * self.linear_power(k_right, z)
        )

    def _evaluate_12(self, k1, k2, k3, z, **params):
        return self._evaluate_pair(k1, k2, k3, z)

    def _evaluate_23(self, k1, k2, k3, z, **params):
        return self._evaluate_pair(k2, k3, k1, z)

    def _evaluate_31(self, k1, k2, k3, z, **params):
        return self._evaluate_pair(k3, k1, k2, z)

class SPTGalaxyBispectrum3D(Bispectrum3D):
    r"""Direct tree-level real-space galaxy bispectrum in SPT.

    Parameters
    ----------
    linear_power : callable
        Callable ``linear_power(k, z)`` returning the linear matter power
        spectrum.  It should support NumPy-broadcastable ``k`` and ``z``.
    b1, b2, bK2 : float or callable, optional
        Eulerian galaxy-bias parameters.  A callable is evaluated as
        ``bias(z)``.  The convention is the same as
        :class:`SPTGalaxyBispectrumMultipole3D` in the semi-analytic module.
    support : Support3D, optional
        Domain advertised to generic fastnc projection/interpolation machinery.
        If omitted, an unbounded support with policy ``"ignore"`` is used.

    Notes
    -----
    The deterministic Eulerian bias expansion is taken to be

    .. math::

        \delta_g = b_1\delta + \frac{b_2}{2}\delta^2 + b_{K^2}K^2 + \cdots.

    At tree level this gives

    .. math::

        B_g = b_1^3 B_{mmm}^{\rm tree}
            + b_1^2 b_2 \sum_{\rm cyc} P_iP_j
            + 2 b_1^2 b_{K^2}\sum_{\rm cyc} S_{ij}P_iP_j,

    where

    .. math::

        S_{ij}=\mu_{ij}^2-\frac13.

    This class evaluates the full triangle directly and does not use the
    semi-analytic multipole decomposition.
    """

    def __init__(
        self,
        linear_power: Callable,
        *,
        b1=1.0,
        b2=0.0,
        bK2=0.0,
        support: Support3D | None = None,
    ):
        if not callable(linear_power):
            raise TypeError("linear_power must be callable as linear_power(k, z)")
        self.linear_power = linear_power
        self.b1 = b1
        self.b2 = b2
        self.bK2 = bK2
        terms = []
        for pair, evaluator in (
            ("12", self._evaluate_tree_12),
            ("23", self._evaluate_tree_23),
            ("31", self._evaluate_tree_31),
        ):
            terms.append(
                WeightedTerm3D(
                    coefficient=self._tree_coefficient,
                    term=BispectrumTerm3D(
                        name=f"tree:F2:{pair}",
                        representations=(NumericExpression3D(evaluator),),
                    ),
                )
            )
        for pair, evaluator in (
            ("12", self._evaluate_quadratic_12),
            ("23", self._evaluate_quadratic_23),
            ("31", self._evaluate_quadratic_31),
        ):
            terms.append(
                WeightedTerm3D(
                    coefficient=self._quadratic_coefficient,
                    term=BispectrumTerm3D(
                        name=f"bias:quadratic:{pair}",
                        representations=(NumericExpression3D(evaluator),),
                    ),
                )
            )
        for pair, evaluator in (
            ("12", self._evaluate_tidal_12),
            ("23", self._evaluate_tidal_23),
            ("31", self._evaluate_tidal_31),
        ):
            terms.append(
                WeightedTerm3D(
                    coefficient=self._tidal_coefficient,
                    term=BispectrumTerm3D(
                        name=f"bias:tidal:{pair}",
                        representations=(NumericExpression3D(evaluator),),
                    ),
                )
            )
        super().__init__(terms, support=support)

    @classmethod
    def simple_debug(
        cls,
        *,
        b1=1.0,
        b2=0.0,
        bK2=0.0,
        cosmo: Mapping[str, float] | None = None,
        amplitude: float = 1.0e4,
        k_eq: float = 2.0e-2,
        transfer_power: float = 1.5,
        pklin_kind: str = "debug",
        support: Support3D | None = None,
    ) -> "SPTGalaxyBispectrum3D":
        """Return a self-initialized tree-level galaxy SPT model for debugging.

        The linear spectrum and growth prescription are identical to
        :meth:`SPTMatterBispectrum3D.simple_debug`.  Galaxy-bias parameters may
        be scalars or callables ``bias(z)``, exactly as in the ordinary
        constructor.
        """
        matter = SPTMatterBispectrum3D.simple_debug(
            cosmo=cosmo,
            amplitude=amplitude,
            k_eq=k_eq,
            transfer_power=transfer_power,
            pklin_kind=pklin_kind,
            support=support,
        )
        return cls(
            linear_power=matter.linear_power,
            b1=b1,
            b2=b2,
            bK2=bK2,
            support=support,
        )

    def update_physics(
        self,
        *,
        linear_power=None,
        b1=None,
        b2=None,
        bK2=None,
    ):
        """Update the power spectrum and/or galaxy-bias parameters in-place."""
        changed = False
        if linear_power is not None:
            if not callable(linear_power):
                raise TypeError("linear_power must be callable as linear_power(k, z)")
            self.linear_power = linear_power
            changed = True
        if b1 is not None:
            self.b1 = b1
            changed = True
        if b2 is not None:
            self.b2 = b2
            changed = True
        if bK2 is not None:
            self.bK2 = bK2
            changed = True
        if changed:
            self._state_updated()
        return self

    def _tree_coefficient(self, z):
        b1 = _value_at_z(self.b1, z)
        return b1**3

    def _quadratic_coefficient(self, z):
        b1 = _value_at_z(self.b1, z)
        b2 = _value_at_z(self.b2, z)
        return b1**2 * b2

    def _tidal_coefficient(self, z):
        b1 = _value_at_z(self.b1, z)
        bK2 = _value_at_z(self.bK2, z)
        return 2.0 * b1**2 * bK2

    def _evaluate_pair(self, kind, k_left, k_right, k_closing, z):
        k_left = np.asarray(k_left, dtype=float)
        k_right = np.asarray(k_right, dtype=float)
        k_closing = np.asarray(k_closing, dtype=float)
        product = self.linear_power(k_left, z) * self.linear_power(k_right, z)
        if kind == "quadratic":
            return product
        mu = _pair_cosine(k_left, k_right, k_closing)
        if kind == "tree":
            return 2.0 * f2_kernel(k_left, k_right, mu) * product
        return tidal_kernel(mu) * product

    def _evaluate_tree_12(self, k1, k2, k3, z, **params):
        return self._evaluate_pair("tree", k1, k2, k3, z)

    def _evaluate_tree_23(self, k1, k2, k3, z, **params):
        return self._evaluate_pair("tree", k2, k3, k1, z)

    def _evaluate_tree_31(self, k1, k2, k3, z, **params):
        return self._evaluate_pair("tree", k3, k1, k2, z)

    def _evaluate_quadratic_12(self, k1, k2, k3, z, **params):
        return self._evaluate_pair("quadratic", k1, k2, k3, z)

    def _evaluate_quadratic_23(self, k1, k2, k3, z, **params):
        return self._evaluate_pair("quadratic", k2, k3, k1, z)

    def _evaluate_quadratic_31(self, k1, k2, k3, z, **params):
        return self._evaluate_pair("quadratic", k3, k1, k2, z)

    def _evaluate_tidal_12(self, k1, k2, k3, z, **params):
        return self._evaluate_pair("tidal", k1, k2, k3, z)

    def _evaluate_tidal_23(self, k1, k2, k3, z, **params):
        return self._evaluate_pair("tidal", k2, k3, k1, z)

    def _evaluate_tidal_31(self, k1, k2, k3, z, **params):
        return self._evaluate_pair("tidal", k3, k1, k2, z)
