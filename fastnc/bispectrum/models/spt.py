"""Direct standard-perturbation-theory bispectrum models.

This module contains ordinary configuration-space/Fourier-space evaluators of
SPT bispectra.  It is intentionally separate from :mod:`fastnc.bispectrum.analytic`,
whose models exploit separability to evaluate bispectrum multipoles directly.
"""
from __future__ import annotations

from typing import Callable, Mapping

import numpy as np

from ..base import Bispectrum3D
from ..terms import ModelBispectrumTerm
from ..slepian import ModelSlepianTerm, SlepianLOSMomentMetadata
from ..support import Support3D
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


_CYCLIC_LEGS = ((0, 1, 2), (1, 2, 0), (2, 0, 1))


class SPTMatterCyclicTerm(ModelBispectrumTerm):
    """One cyclic tree-level matter-SPT contribution ``2 F2_ij P_i P_j``."""

    def __init__(self, model, pair: str):
        super().__init__(model)
        try:
            self._cyclic_index = {"12": 0, "23": 1, "31": 2}[str(pair)]
        except KeyError as exc:
            raise ValueError("pair must be one of '12', '23', or '31'") from exc
        self.pair = str(pair)

    def evaluate(self, k1, k2, k3, z, **params):
        ks = [np.asarray(k1, dtype=float), np.asarray(k2, dtype=float), np.asarray(k3, dtype=float)]
        i, j, other = _CYCLIC_LEGS[self._cyclic_index]
        pi = self.model.linear_power(ks[i], z)
        pj = self.model.linear_power(ks[j], z)
        mu = _pair_cosine(ks[i], ks[j], ks[other])
        return 2.0 * f2_kernel(ks[i], ks[j], mu) * pi * pj


class SPTMatterSlepianTerm(ModelSlepianTerm):
    r"""One exponential/separable contribution to a cyclic matter-SPT term.

    The seven-term decomposition of ``2 F2_ij P_i P_j`` is

    .. math::

        \frac{12}{7} P_iP_j
        + \frac12 \sum_{s=\pm1} e^{is(\phi_i-\phi_j)}
          \left[\frac{k_i}{k_j}+\frac{k_j}{k_i}\right]P_iP_j
        + \frac17 \sum_{s=\pm1} e^{2is(\phi_i-\phi_j)}P_iP_j.

    The two dipole radial ratios are kept as distinct terms.  This is the
    physically identical but numerically useful 1 + 4 + 2 decomposition used
    by the future Slepian calculator, because ``P(k)``, ``k P(k)``, and
    ``P(k)/k`` can share FFTLog coefficients through integer exponent shifts.
    """

    def __init__(self, model, pair: str, *, coefficient: float, harmonic: int, shifts):
        super().__init__(model)
        try:
            cyclic_index = {"12": 0, "23": 1, "31": 2}[str(pair)]
        except KeyError as exc:
            raise ValueError("pair must be one of '12', '23', or '31'") from exc
        i, j, other = _CYCLIC_LEGS[cyclic_index]

        shifts = tuple(int(x) for x in shifts)
        if len(shifts) != 2:
            raise ValueError("shifts must contain the powers applied to the two paired legs")
        if sum(shifts) != 0:
            raise ValueError("SPT dipole power shifts must sum to zero")

        self.pair = str(pair)
        self.paired_legs = (i, j)
        self.other_leg = other
        self.coefficient = float(coefficient)
        self.harmonic = int(harmonic)
        self.pair_power_shifts = shifts

        orders = [0, 0, 0]
        orders[i] = self.harmonic
        orders[j] = -self.harmonic
        self._angular_orders = tuple(orders)

        global_shifts = [0, 0, 0]
        global_shifts[i] = shifts[0]
        global_shifts[j] = shifts[1]
        self.power_shifts = tuple(global_shifts)

        # Declarative only.  An arbitrary user-supplied linear_power(k,z) is
        # not assumed to factorize as D(z)^2 P0(k) until the LOS phase adds
        # an explicit capability/rule for that stronger statement.
        self._los_metadata = SlepianLOSMomentMetadata(
            kind="spt-two-linear-power",
            signature=(self.power_shifts,),
        )

    @property
    def angular_orders(self):
        return self._angular_orders

    @property
    def los_moment_metadata(self):
        return self._los_metadata

    @property
    def radial_signature(self):
        """Hashable identity for future FFTLog/grouping logic."""
        return ("linear-power-pair", self.paired_legs, self.power_shifts)

    def c(self, z, **params):
        return self.coefficient

    def _radial_leg(self, leg, k, z):
        if leg == self.other_leg:
            return np.ones_like(np.broadcast_arrays(np.asarray(k, dtype=float), np.asarray(z, dtype=float))[0])
        shift = self.power_shifts[leg]
        k = np.asarray(k, dtype=float)
        return self.model.linear_power(k, z) * k**shift

    def f1(self, k, z, **params):
        return self._radial_leg(0, k, z)

    def f2(self, k, z, **params):
        return self._radial_leg(1, k, z)

    def f3(self, k, z, **params):
        return self._radial_leg(2, k, z)


class SPTGalaxySlepianTreeTerm(SPTMatterSlepianTerm):
    r"""Galaxy-tree Slepian term reusing the matter-SPT radial/angular algebra.

    The only physical difference from :class:`SPTMatterSlepianTerm` is the
    tree-level galaxy-bias coefficient :math:`b_1(z)^3`.  Keeping this as a
    coefficient layer avoids duplicating the seven-term F2 decomposition and
    leaves the radial identities exactly shared with matter SPT.
    """

    def __init__(self, model, pair: str, *, coefficient: float, harmonic: int, shifts):
        super().__init__(
            model, pair, coefficient=coefficient, harmonic=harmonic, shifts=shifts
        )
        # Declarative only.  A future LOS rule may exploit the common two-power
        # structure together with the b1(z)^3 coefficient, but this phase does
        # not assume a factorized growth law for arbitrary linear_power(k,z).
        self._los_metadata = SlepianLOSMomentMetadata(
            kind="spt-galaxy-tree-two-linear-power",
            signature=(("b1", 3), self.power_shifts),
        )

    def c(self, z, **params):
        b1 = _value_at_z(self.model.b1, z)
        return self.coefficient * b1**3


_SPT_SLEPIAN_PAIR_SPECS = (
    # coefficient, harmonic, paired-leg power shifts
    (12.0 / 7.0,  0, ( 0,  0)),
    ( 1.0 / 2.0,  1, ( 1, -1)),
    ( 1.0 / 2.0, -1, ( 1, -1)),
    ( 1.0 / 2.0,  1, (-1,  1)),
    ( 1.0 / 2.0, -1, (-1,  1)),
    ( 1.0 / 7.0,  2, ( 0,  0)),
    ( 1.0 / 7.0, -2, ( 0,  0)),
)


def _build_slepian_pair_terms(model, pair: str, term_type):
    """Build one exact seven-term F2 decomposition using shared algebra."""
    return tuple(
        term_type(
            model, pair, coefficient=coefficient, harmonic=harmonic, shifts=shifts
        )
        for coefficient, harmonic, shifts in _SPT_SLEPIAN_PAIR_SPECS
    )


def _build_matter_slepian_pair_terms(model, pair: str):
    """Return the exact seven separable matter terms for one cyclic SPT pair."""
    return _build_slepian_pair_terms(model, pair, SPTMatterSlepianTerm)


def _build_galaxy_slepian_pair_terms(model, pair: str):
    """Return the galaxy-tree seven terms using the same matter F2 algebra."""
    return _build_slepian_pair_terms(model, pair, SPTGalaxySlepianTreeTerm)


class SPTGalaxyCyclicTerm(ModelBispectrumTerm):
    """One cyclic galaxy-SPT contribution of a selected physical kind."""

    def __init__(self, model, pair: str, kind: str):
        super().__init__(model)
        try:
            self._cyclic_index = {"12": 0, "23": 1, "31": 2}[str(pair)]
        except KeyError as exc:
            raise ValueError("pair must be one of '12', '23', or '31'") from exc
        if kind not in {"tree", "quadratic", "tidal"}:
            raise ValueError("kind must be 'tree', 'quadratic', or 'tidal'")
        self.pair = str(pair)
        self.kind = kind

    def evaluate(self, k1, k2, k3, z, **params):
        ks = [np.asarray(k1, dtype=float), np.asarray(k2, dtype=float), np.asarray(k3, dtype=float)]
        i, j, other = _CYCLIC_LEGS[self._cyclic_index]
        pi = self.model.linear_power(ks[i], z)
        pj = self.model.linear_power(ks[j], z)
        mu = _pair_cosine(ks[i], ks[j], ks[other])

        b1 = _value_at_z(self.model.b1, z)
        if self.kind == "tree":
            return b1**3 * 2.0 * f2_kernel(ks[i], ks[j], mu) * pi * pj
        if self.kind == "quadratic":
            b2 = _value_at_z(self.model.b2, z)
            return b1**2 * b2 * pi * pj
        bK2 = _value_at_z(self.model.bK2, z)
        return 2.0 * b1**2 * bK2 * tidal_kernel(mu) * pi * pj


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
        *,
        linear_power_z0: Callable | None = None,
        growth_factor: Callable | None = None,
    ):
        if not callable(linear_power):
            raise TypeError("linear_power must be callable as linear_power(k, z)")
        self.linear_power = linear_power
        if (linear_power_z0 is None) != (growth_factor is None):
            raise ValueError("linear_power_z0 and growth_factor must be provided together")
        self._linear_power_z0 = linear_power_z0
        self._growth_factor = growth_factor
        self.support = support or Support3D(policy="ignore")
        self._cyclic_terms = {
            pair: SPTMatterCyclicTerm(self, pair) for pair in ("12", "23", "31")
        }
        # Hybrid physical partition used from Phase 5 onward.  The regular B23
        # sector remains generic; B12+B31 are represented by 7+7 directly
        # evaluable Slepian terms.  SlepianConfig(mode="off") combines these
        # collections and therefore still sends the complete bispectrum through
        # the generic numerical route.
        self._generic_terms = (self._cyclic_terms["23"],)
        self._slepian_terms = (
            _build_matter_slepian_pair_terms(self, "12")
            + _build_matter_slepian_pair_terms(self, "31")
        )

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

        return cls(
            linear_power=linear_power,
            support=support,
            linear_power_z0=pklin0,
            growth_factor=lambda z: simple_linear_growth(z, cosmo=cosmo_dict),
        )

    supports_slepian = True
    slepian_los_kind = "factorized-growth"

    def generic_terms(self, **params):
        """Return the regular ``B23`` contribution for the hybrid route."""
        return self._generic_terms

    def slepian_terms(self, **params):
        """Return the exact 7+7 separable terms representing ``B12+B31``."""
        return self._slepian_terms

    @property
    def has_factorized_growth(self):
        return self._linear_power_z0 is not None and self._growth_factor is not None

    def factorized_linear_power(self):
        """Return explicit ``(P0, D)`` with ``P(k,z)=D(z)^2 P0(k)``.

        No factorization is inferred from sampled values.  Models constructed
        with an arbitrary ``linear_power(k,z)`` therefore return no capability
        unless the user explicitly installs one.
        """
        if not self.has_factorized_growth:
            raise RuntimeError("this SPT model has no explicit factorized-growth capability")
        return self._linear_power_z0, self._growth_factor

    def set_factorized_growth(self, linear_power_z0: Callable, growth_factor: Callable):
        if not callable(linear_power_z0) or not callable(growth_factor):
            raise TypeError("linear_power_z0 and growth_factor must be callable")
        self._linear_power_z0 = linear_power_z0
        self._growth_factor = growth_factor
        return self

    def update_physics(self, *, linear_power: Callable):
        """Replace the linear power evaluator and invalidate old factorization."""
        if not callable(linear_power):
            raise TypeError("linear_power must be callable as linear_power(k, z)")
        self.linear_power = linear_power
        self._linear_power_z0 = None
        self._growth_factor = None
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
        linear_power_z0: Callable | None = None,
        growth_factor: Callable | None = None,
    ):
        if not callable(linear_power):
            raise TypeError("linear_power must be callable as linear_power(k, z)")
        self.linear_power = linear_power
        if (linear_power_z0 is None) != (growth_factor is None):
            raise ValueError("linear_power_z0 and growth_factor must be provided together")
        self._linear_power_z0 = linear_power_z0
        self._growth_factor = growth_factor
        self.b1 = b1
        self.b2 = b2
        self.bK2 = bK2
        self.support = support or Support3D(policy="ignore")
        self._cyclic_terms = {
            (kind, pair): SPTGalaxyCyclicTerm(self, pair, kind)
            for kind in ("tree", "quadratic", "tidal")
            for pair in ("12", "23", "31")
        }

        # Phase 6 hybrid partition.  Only the high-L matter-tree 12+31 sector
        # is represented by Slepian terms.  The regular tree B23 piece and all
        # finite-angular-support quadratic/tidal bias terms stay generic.
        self._generic_terms = (
            self._cyclic_terms[("tree", "23")],
            *(self._cyclic_terms[("quadratic", pair)] for pair in ("12", "23", "31")),
            *(self._cyclic_terms[("tidal", pair)] for pair in ("12", "23", "31")),
        )
        self._slepian_terms = (
            _build_galaxy_slepian_pair_terms(self, "12")
            + _build_galaxy_slepian_pair_terms(self, "31")
        )

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
            linear_power_z0=matter._linear_power_z0,
            growth_factor=matter._growth_factor,
        )

    supports_slepian = True
    slepian_los_kind = "factorized-growth"

    def generic_terms(self, **params):
        """Return B23 tree plus all finite-support quadratic/tidal terms."""
        return self._generic_terms

    def slepian_terms(self, **params):
        """Return 7+7 galaxy-tree Slepian terms for the B12+B31 sector."""
        return self._slepian_terms

    @property
    def has_factorized_growth(self):
        return self._linear_power_z0 is not None and self._growth_factor is not None

    def factorized_linear_power(self):
        if not self.has_factorized_growth:
            raise RuntimeError("this SPT galaxy model has no explicit factorized-growth capability")
        return self._linear_power_z0, self._growth_factor

    def set_factorized_growth(self, linear_power_z0: Callable, growth_factor: Callable):
        if not callable(linear_power_z0) or not callable(growth_factor):
            raise TypeError("linear_power_z0 and growth_factor must be callable")
        self._linear_power_z0 = linear_power_z0
        self._growth_factor = growth_factor
        return self

    def update_physics(
        self,
        *,
        linear_power=None,
        b1=None,
        b2=None,
        bK2=None,
    ):
        """Update the power spectrum and/or galaxy-bias parameters in-place."""
        if linear_power is not None:
            if not callable(linear_power):
                raise TypeError("linear_power must be callable as linear_power(k, z)")
            self.linear_power = linear_power
            self._linear_power_z0 = None
            self._growth_factor = None
        if b1 is not None:
            self.b1 = b1
        if b2 is not None:
            self.b2 = b2
        if bK2 is not None:
            self.bK2 = bK2
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

        tree = (
            2.0 * f2_kernel(k1, k2, mu12) * p1 * p2
            + 2.0 * f2_kernel(k2, k3, mu23) * p2 * p3
            + 2.0 * f2_kernel(k3, k1, mu31) * p3 * p1
        )
        quadratic = p1 * p2 + p2 * p3 + p3 * p1
        tidal = (
            tidal_kernel(mu12) * p1 * p2
            + tidal_kernel(mu23) * p2 * p3
            + tidal_kernel(mu31) * p3 * p1
        )

        b1 = _value_at_z(self.b1, z)
        b2 = _value_at_z(self.b2, z)
        bK2 = _value_at_z(self.bK2, z)

        return (
            b1**3 * tree
            + b1**2 * b2 * quadratic
            + 2.0 * b1**2 * bK2 * tidal
        )
