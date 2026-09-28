"""Direct standard-perturbation-theory bispectrum models.

This module contains direct triangle-space reference implementations expressed
as typed term aggregates.
"""
from __future__ import annotations

from typing import Callable, Mapping

import numpy as np

from ..bispectrum import Bispectrum3D
from ..representations import (
    NumericExpression3D,
    SlepianExpression2D,
    SlepianExpression3D,
    SlepianRadialFactor3D,
)
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


_SLEPIAN_PAIR_LAYOUTS_3D = {
    "12": ((0, 1), 2),
    "31": ((2, 0), 1),
}


def _safe_phase(k_left, k_right, k_opposite, mode):
    mu = np.nan_to_num(
        _pair_cosine(k_left, k_right, k_opposite),
        nan=0.0,
        posinf=1.0,
        neginf=-1.0,
    )
    return np.exp(1j * int(mode) * np.arccos(np.clip(mu, -1.0, 1.0)))


def _scaled_linear_power(linear_power, k, z, k_power):
    k = np.asarray(k, dtype=float)
    values = np.asarray(linear_power(k, z))
    target_shape = np.broadcast_shapes(k.shape, np.shape(z), values.shape)
    values = np.broadcast_to(values, target_shape)
    k = np.broadcast_to(k, target_shape)
    if k_power == 0:
        return values
    if k_power > 0:
        return values * k**k_power
    result = np.zeros(target_shape, dtype=values.dtype)
    return np.divide(values, k ** (-k_power), out=result, where=k != 0.0)


def _separable_pair_term_3d(
    owner,
    *,
    name,
    pair,
    mode,
    left_power,
    right_power,
    coefficient,
):
    (left, right), constant = _SLEPIAN_PAIR_LAYOUTS_3D[pair]
    powers = (left_power, right_power)

    def factor(index):
        radial_power = powers[index]
        return SlepianRadialFactor3D(
            lambda k, z, _power=radial_power: _scaled_linear_power(
                owner.linear_power, k, z, _power
            )
        )

    radial_factors = [None, None, None]
    radial_factors[left] = factor(0)
    radial_factors[right] = factor(1)
    radial_factors[constant] = SlepianRadialFactor3D.constant()
    angular_orders = [0, 0, 0]
    angular_orders[left] = mode
    angular_orders[right] = -mode

    def numeric(k1, k2, k3, z, **params):
        k = (k1, k2, k3)
        return (
            coefficient
            * _scaled_linear_power(owner.linear_power, k[left], z, left_power)
            * _scaled_linear_power(owner.linear_power, k[right], z, right_power)
            * _safe_phase(k[left], k[right], k[constant], mode)
        )

    return BispectrumTerm3D(
        name=name,
        representations=(
            NumericExpression3D(numeric),
            SlepianExpression3D(
                coefficient=coefficient,
                radial_factors=tuple(radial_factors),
                angular_orders=tuple(angular_orders),
            ),
        ),
    )


def _pair_terms_3d(owner, pair, kind):
    """Return exact finite Slepian components for one supported SPT pair."""
    if pair not in _SLEPIAN_PAIR_LAYOUTS_3D:
        raise ValueError("Slepian-compatible 3D pairs are '12' and '31'")
    if kind == "tree":
        specifications = [(0, 0, 0, 12.0 / 7.0, "")]
        for mode in (-1, 1):
            specifications.extend(
                (
                    (mode, 1, -1, 0.5, ":r+1"),
                    (mode, -1, 1, 0.5, ":r-1"),
                )
            )
        specifications.extend(
            ((-2, 0, 0, 1.0 / 7.0, ""), (2, 0, 0, 1.0 / 7.0, ""))
        )
        prefix = "tree:F2"
    elif kind == "quadratic":
        specifications = [(0, 0, 0, 1.0, "")]
        prefix = "bias:quadratic"
    elif kind == "tidal":
        specifications = [
            (0, 0, 0, 1.0 / 6.0, ""),
            (-2, 0, 0, 1.0 / 4.0, ""),
            (2, 0, 0, 1.0 / 4.0, ""),
        ]
        prefix = "bias:tidal"
    else:
        raise ValueError("kind must be 'tree', 'quadratic', or 'tidal'")

    return tuple(
        _separable_pair_term_3d(
            owner,
            name=f"{prefix}:{pair}:m{mode:+d}{suffix}",
            pair=pair,
            mode=mode,
            left_power=left_power,
            right_power=right_power,
            coefficient=coefficient,
        )
        for mode, left_power, right_power, coefficient, suffix in specifications
    )


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

    The pair-12 and pair-31 contributions are decomposed exactly into their
    finite Fourier harmonics and expose both numeric and Slepian
    representations. Pair 23 remains one numeric-only term because treating
    it as a constant-leg Slepian expression would eliminate physical leg 1.
    """

    def __init__(
        self,
        linear_power: Callable,
        support: Support3D | None = None,
    ):
        if not callable(linear_power):
            raise TypeError("linear_power must be callable as linear_power(k, z)")
        self.linear_power = linear_power
        terms = [*_pair_terms_3d(self, "12", "tree")]
        terms.append(
            BispectrumTerm3D(
                name="tree:F2:23",
                representations=(NumericExpression3D(self._evaluate_23),),
            )
        )
        terms.extend(_pair_terms_3d(self, "31", "tree"))
        super().__init__(tuple(terms), support=support)

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

    Terms from pairs 12 and 31 expose every exact finite Slepian harmonic.
    Pair-23 terms remain numeric-only because leg 1 cannot be the eliminated
    constant leg of the current ZetaK convention.
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
        terms = [
            WeightedTerm3D(self._tree_coefficient, term)
            for term in _pair_terms_3d(self, "12", "tree")
        ]
        terms.append(
            WeightedTerm3D(
                self._tree_coefficient,
                BispectrumTerm3D(
                    name="tree:F2:23",
                    representations=(NumericExpression3D(self._evaluate_tree_23),),
                ),
            )
        )
        terms.extend(
            WeightedTerm3D(self._tree_coefficient, term)
            for term in _pair_terms_3d(self, "31", "tree")
        )
        terms.extend(
            WeightedTerm3D(self._quadratic_coefficient, term)
            for term in _pair_terms_3d(self, "12", "quadratic")
        )
        terms.append(
            WeightedTerm3D(
                self._quadratic_coefficient,
                BispectrumTerm3D(
                    name="bias:quadratic:23",
                    representations=(
                        NumericExpression3D(self._evaluate_quadratic_23),
                    ),
                ),
            )
        )
        terms.extend(
            WeightedTerm3D(self._quadratic_coefficient, term)
            for term in _pair_terms_3d(self, "31", "quadratic")
        )
        terms.extend(
            WeightedTerm3D(self._tidal_coefficient, term)
            for term in _pair_terms_3d(self, "12", "tidal")
        )
        terms.append(
            WeightedTerm3D(
                self._tidal_coefficient,
                BispectrumTerm3D(
                    name="bias:tidal:23",
                    representations=(NumericExpression3D(self._evaluate_tidal_23),),
                ),
            )
        )
        terms.extend(
            WeightedTerm3D(self._tidal_coefficient, term)
            for term in _pair_terms_3d(self, "31", "tidal")
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
