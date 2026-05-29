"""Grid and triangle-coordinate helpers.

The bispectrum Fourier multipoles in this package are defined with respect to

the outer angle

    Delta beta = beta1 - beta2,

not the inner angle alpha.  The two are related by

    alpha = pi - Delta beta,
    cos(alpha) = -cos(Delta beta).

For the side-only angular bispectrum,

    ell3^2 = ell1^2 + ell2^2 + 2 ell1 ell2 cos(Delta beta).

The public grid returned by ``make_ell_psi_delta_beta_grid`` is therefore a
Delta-beta grid.  The endpoints are controlled directly by
``delta_beta_min`` and ``delta_beta_max``.  This is preferred over the old
``eps_mu`` convention because the Fourier basis itself is defined in
Delta beta.
"""
from __future__ import annotations

from dataclasses import dataclass
import warnings
import numpy as np


def loglinear(xmin, xmid, xmax, nlog, nlin):
    """Hybrid logarithmic+linear grid.

    The logarithmic part resolves the endpoint near ``xmin`` and the linear
    part covers the rest of the interval.  The returned grid is sorted and
    unique up to floating-point equality.
    """
    xmin = float(xmin)
    xmid = float(xmid)
    xmax = float(xmax)
    nlog = int(nlog)
    nlin = int(nlin)

    if xmin <= 0.0 or xmid <= 0.0 or not (xmin < xmid < xmax):
        return np.linspace(xmin, xmax, max(nlog + nlin, 2))

    a = np.logspace(np.log10(xmin), np.log10(xmid), max(nlog, 2), endpoint=False)
    b = np.linspace(xmid, xmax, max(nlin, 2), endpoint=True)
    return np.unique(np.concatenate([a, b]))


def delta_beta_min_from_eps_mu(eps_mu: float) -> float:
    """Convert the old ``eps_mu`` endpoint cut to ``Delta beta_min``.

    The old convention used ``mu = cos(Delta beta) <= 1 - eps_mu``.  Hence

        Delta beta_min = arccos(1 - eps_mu).
    """
    eps_mu = float(eps_mu)
    if not (0.0 < eps_mu < 2.0):
        raise ValueError("eps_mu must satisfy 0 < eps_mu < 2")
    return float(np.arccos(1.0 - eps_mu))


def ellpsi_to_ell1ell2(ell, psi):
    """Convert ``(ell, psi)`` to ``(ell1, ell2)``.

    Here ``ell1 = ell cos psi`` and ``ell2 = ell sin psi``.
    """
    return ell * np.cos(psi), ell * np.sin(psi)


def ell1ell2delta_beta_to_ell3(ell1, ell2, delta_beta):
    """Return the third side from the outer angle ``Delta beta``.

    The convention is

        ell3^2 = ell1^2 + ell2^2 + 2 ell1 ell2 cos(Delta beta).
    """
    ell3_sq = ell1**2 + ell2**2 + 2.0 * ell1 * ell2 * np.cos(delta_beta)
    ell3 = np.sqrt(np.maximum(ell3_sq, 0.0))

    # Roundoff-level safety at degenerate triangle boundaries.
    ell3 = np.minimum(ell3, ell1 + ell2)
    ell3 = np.maximum(ell3, np.abs(ell1 - ell2))
    return ell3


def ellpsidelta_beta_to_sides(ell, psi, delta_beta):
    """Convert ``(ell, psi, Delta beta)`` to ``(ell1, ell2, ell3)``."""
    ell1, ell2 = ellpsi_to_ell1ell2(ell, psi)
    ell3 = ell1ell2delta_beta_to_ell3(ell1, ell2, delta_beta)
    return ell1, ell2, ell3


def sides_to_ellpsidelta_beta(ell1, ell2, ell3):
    """Convert side lengths to ``(ell, psi, Delta beta)``.

    Since

        ell3^2 = ell1^2 + ell2^2 + 2 ell1 ell2 cos(Delta beta),

    we have

        cos(Delta beta) = (ell3^2 - ell1^2 - ell2^2) / (2 ell1 ell2).
    """
    ell1 = np.asarray(ell1, dtype=float)
    ell2 = np.asarray(ell2, dtype=float)
    ell3 = np.asarray(ell3, dtype=float)
    ell = np.sqrt(ell1**2 + ell2**2)
    psi = np.arctan2(ell2, ell1)
    cos_delta = (ell3**2 - ell1**2 - ell2**2) / (2.0 * ell1 * ell2)
    delta_beta = np.arccos(np.clip(cos_delta, -1.0, 1.0))
    return ell, psi, delta_beta


# Backward-compatible aliases for code that still imports the old names.
# These now use the outer-angle convention.  Prefer the explicit Delta-beta
# names in new code.
ell1ell2alpha_to_ell3 = ell1ell2delta_beta_to_ell3
ellpsialpha_to_sides = ellpsidelta_beta_to_sides
sides_to_ellpsialpha = sides_to_ellpsidelta_beta


def fold_psi(psi):
    """Fold ``psi`` to the fundamental range ``[0, pi/4]``."""
    psi = np.asarray(psi)
    return np.where(psi > np.pi / 4, np.pi / 2 - psi, psi)


@dataclass(frozen=True)
class MultipoleGridConfig:
    ell_min: float = 1.0e-1
    ell_max: float = 1.0e5
    n_ell: int = 100
    psi_min: float = 1.0e-4
    psi_max: float = np.pi / 4
    n_psi: int = 80

    # Direct endpoint control in the Fourier variable Delta beta.
    # Both exact endpoints are excluded by default because they correspond to
    # degenerate or flattened triangle boundaries for many bispectrum models.
    delta_beta_min: float = 0.0
    delta_beta_max: float = np.pi

    n_delta_beta_lin: int = 50
    n_delta_beta_log: int = 30
    delta_beta_transition: float = 5.0e-2

    # ``outer`` refines Delta beta -> 0.
    # ``inner`` refines alpha = pi - Delta beta -> 0, i.e. the squeezed/flattened endpoint.
    # ``both`` combines both refinements.
    angle_sampling: str = "inner"


class OuterAngleGrid:
    """Nonuniform grid for the outer angle ``Delta beta``.

    The grid is logarithmically refined near ``Delta beta_min`` and linearly
    spaced over the rest of the domain.  The main API uses ``delta_beta_min``
    and ``delta_beta_max`` directly.  Use ``from_eps_mu`` only when reproducing
    the old endpoint cut ``mu = cos(Delta beta) <= 1 - eps_mu``.
    """

    def __init__(
        self,
        delta_beta_min: float = 5.0e-4,
        delta_beta_max: float = np.pi - 5.0e-4,
        n_delta_beta_lin: int = 50,
        n_delta_beta_log: int = 30,
        transition: float = 5.0e-2,
        sampling: str = "inner",
    ):
        self.delta_beta_min = float(delta_beta_min)
        self.delta_beta_max = float(delta_beta_max)
        self.n_delta_beta_lin = int(n_delta_beta_lin)
        self.n_delta_beta_log = int(n_delta_beta_log)
        self.transition = float(transition)
        self.sampling = str(sampling)

    @classmethod
    def from_eps_mu(
        cls,
        eps_mu: float,
        delta_beta_max: float = np.pi - 5.0e-4,
        n_delta_beta_lin: int = 50,
        n_delta_beta_log: int = 30,
        transition: float = 5.0e-2,
        sampling: str = "inner",
    ):
        """Construct the grid using the old ``eps_mu`` lower-end convention."""
        return cls(
            delta_beta_min=delta_beta_min_from_eps_mu(eps_mu),
            delta_beta_max=delta_beta_max,
            n_delta_beta_lin=n_delta_beta_lin,
            n_delta_beta_log=n_delta_beta_log,
            transition=transition,
        )

    @property
    def delta_beta(self):
        dmin = self.delta_beta_min
        dmax = self.delta_beta_max
        trans = self.transition

        if not (0.0 <= dmin < dmax <= np.pi):
            raise ValueError("Require 0 <= delta_beta_min < delta_beta_max <= pi")

        # Numerical quadrature remains open at exact endpoints unless the user
        # explicitly chooses nonzero cuts.  The multipole definition is still
        # the full [0, pi] integral.
        dmin_open = max(dmin, np.nextafter(0.0, 1.0))
        dmax_open = min(dmax, np.nextafter(np.pi, 0.0))

        if not (0.0 < dmin_open < dmax_open < np.pi):
            raise ValueError("The requested Delta-beta interval has no open interior")

        def _outer_refined():
            return loglinear(
                dmin_open,
                trans,
                dmax_open,
                self.n_delta_beta_log,
                self.n_delta_beta_lin,
            )

        def _inner_refined():
            # alpha = pi - Delta beta.  Refining alpha -> 0 resolves the
            # squeezed/flattened endpoint Delta beta -> pi while returning a
            # monotonically increasing Delta-beta grid.
            amin = np.pi - dmax_open
            amax = np.pi - dmin_open
            a = loglinear(
                amin,
                trans,
                amax,
                self.n_delta_beta_log,
                self.n_delta_beta_lin,
            )
            return np.pi - a[::-1]

        if self.sampling == "outer":
            d = _outer_refined()
        elif self.sampling == "inner":
            d = _inner_refined()
        elif self.sampling == "both":
            d = np.unique(np.concatenate([_outer_refined(), _inner_refined()]))
        else:
            raise ValueError("sampling must be one of 'outer', 'inner', or 'both'")

        d = np.sort(np.unique(d))
        d = d[(d >= dmin_open) & (d <= dmax_open) & (d > 0.0) & (d < np.pi)]
        return d

    @property
    def mu(self):
        return np.cos(self.delta_beta)


# Backward-compatible class name.  The returned ``alpha`` property now means
# outer angle Delta beta.  Prefer ``OuterAngleGrid`` in new code.
class SqueezedAwareAngleGrid(OuterAngleGrid):
    def __init__(self, eps_alpha=1.0e-7, n_alpha_lin=50, n_alpha_log=30):
        warnings.warn(
            "SqueezedAwareAngleGrid is deprecated; use OuterAngleGrid with "
            "delta_beta_min/delta_beta_max.  The old eps_alpha argument is "
            "interpreted as eps_mu for compatibility.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(
            delta_beta_min=delta_beta_min_from_eps_mu(eps_alpha),
            delta_beta_max=np.pi - 5.0e-4,
            n_delta_beta_lin=n_alpha_lin,
            n_delta_beta_log=n_alpha_log,
        )

    @property
    def alpha(self):
        return self.delta_beta


def make_delta_beta_grid(config: MultipoleGridConfig):
    return OuterAngleGrid(
        delta_beta_min=config.delta_beta_min,
        delta_beta_max=config.delta_beta_max,
        n_delta_beta_lin=config.n_delta_beta_lin,
        n_delta_beta_log=config.n_delta_beta_log,
        transition=config.delta_beta_transition,
        sampling=config.angle_sampling,
    ).delta_beta


def make_ell_psi_delta_beta_grid(config: MultipoleGridConfig):
    """Return grids and sides for an outer-angle multipole calculation.

    Returns
    -------
    ell_grid, psi_grid, delta_beta_grid, E1, E2, E3
        ``E1``, ``E2`` and ``E3`` have shape
        ``(n_ell, n_psi_eff, n_delta_beta)``.
    """
    ell = np.logspace(np.log10(config.ell_min), np.log10(config.ell_max), config.n_ell)
    psi = loglinear(config.psi_min, 1.0e-3, config.psi_max, 50, config.n_psi)
    delta_beta = make_delta_beta_grid(config)

    ELL, PSI, DBETA = np.meshgrid(ell, psi, delta_beta, indexing="ij")
    E1, E2, E3 = ellpsidelta_beta_to_sides(ELL, PSI, DBETA)
    return ell, psi, delta_beta, E1, E2, E3


# Backward-compatible function name.  It now returns the outer-angle grid in the
# third position.  Prefer ``make_ell_psi_delta_beta_grid`` in new code.
def make_ell_psi_alpha_grid(config: MultipoleGridConfig):
    return make_ell_psi_delta_beta_grid(config)
