"""Angular-bispectrum multipole objects and calculators.

Fourier multipoles are defined in the X1-reference convention with

    Delta beta = beta2 - beta3,

and independent radii ``(ell2,ell3)``.  Fourier closure gives

    ell1^2 = ell2^2 + ell3^2 + 2 ell2 ell3 cos(Delta beta).

The default ``basis='fourier-even'`` convention is

    B(ell2, ell3, Delta beta)
      = c0(ell2, ell3) + sum_{q>=1} cq(ell2, ell3) cos(q Delta beta),

with

    c0 = (1/pi) int_0^pi dDelta beta B,
    cq = (2/pi) int_0^pi dDelta beta B cos(q Delta beta).

The ``basis='fourier'`` convention stores full complex Fourier coefficients

    B_L = (1/2pi) int_{-pi}^{pi} dDelta beta B exp(-i L Delta beta),

for ``L=-Lmax,...,+Lmax``.  For the present side-only bispectra this reduces
to ``B_0=c_0`` and ``B_{+L}=B_{-L}=c_L/2`` for ``L>0``.

In practice both exact endpoints are excluded.  The grid is controlled
directly by ``delta_beta_min`` and ``delta_beta_max``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional
import numpy as np
from scipy.special import eval_legendre

from .base import Bispectrum2D
from .decompose import MultipoleLegendre, MultipoleFourier, MultipoleCosine, MultipoleSine
from .grids import (
    MultipoleGridConfig,
    make_ell_psi_delta_beta_grid,
    make_delta_beta_grid,
    ell2ell3delta_beta_to_ell1,
    sides_to_ellpsidelta_beta,
)


@dataclass(frozen=True)
class BispectrumMultipole2DConfig:
    mode_max: int = 30
    mode_max_diag: Optional[int] = None
    ell_min: float = 1.0e-1
    ell_max: float = 1.0e5
    n_ell: int = 100
    psi_min: float = 1.0e-4
    psi_max: float = np.pi / 2 - 1.0e-4
    n_psi: int = 80

    # Direct endpoint control in the Fourier variable Delta beta.
    # The exact endpoints 0 and pi are excluded by default.
    delta_beta_min: float = 5.0e-4
    delta_beta_max: float = np.pi - 5.0e-4
    n_delta_beta_lin: int = 50
    n_delta_beta_log: int = 30
    delta_beta_transition: float = 5.0e-2

    # Grid sampling can be done in the inner angle alpha = pi - Delta beta,
    # while stored coefficients remain outer-angle coefficients.
    angle_sampling: str = "inner"
    decomposition_angle: str = "inner"

    interpolation: str = "linear"
    decomposer_method: str = "gauss-legendre"


# Backward-compatible alias.  New code should use BispectrumMultipole2DConfig.
BispectrumMultipoleConfig = BispectrumMultipole2DConfig


class BispectrumMultipole2D:
    """Object for a 2D angular-bispectrum multipole model.

    This is the fiducial public object for ``B_mode(ell2, ell3)``.  Different
    construction routes are represented by internal evaluators, not by public
    prefixes such as ``Projected`` or ``Decomposed``.
    """

    basis: str
    modes = None

    def __init__(self, evaluator=None, basis: str = "fourier-even", modes=None):
        self._evaluator = evaluator
        self.basis = basis
        self.modes = np.asarray(modes, dtype=int) if modes is not None else None

    @classmethod
    def from_bispectrum2d(
        cls,
        bispectrum: Bispectrum2D,
        config: Optional[BispectrumMultipole2DConfig] = None,
        basis: str = "fourier-even",
        regulator=None,
        **params,
    ):
        config = config or BispectrumMultipole2DConfig()
        evaluator = _Bispectrum2DMultipoleEvaluator(
            bispectrum=bispectrum,
            config=config,
            basis=basis,
            regulator=regulator,
            params=params,
        )
        modes = evaluator.default_modes()
        return cls(evaluator=evaluator, basis=basis, modes=modes)

    @classmethod
    def from_multipole3d(cls, multipole3d, projector, sample_combination=None, modes=None):
        evaluator = _LoSBispectrumMultipole2DEvaluator(
            multipole3d=multipole3d,
            projector=projector,
            sample_combination=sample_combination,
        )
        return cls(
            evaluator=evaluator,
            basis=multipole3d.basis,
            modes=modes,
        )

    def __call__(self, mode, ell2, ell3):
        return self.evaluate(mode, ell2, ell3)

    def evaluate(self, mode, ell2, ell3):
        """Evaluate ``B_L(ell2, ell3)`` in the X1-reference convention."""
        if getattr(self, "_evaluator", None) is None:
            raise NotImplementedError
        return self._evaluator(mode, ell2, ell3)

    def available_modes(self, mode_max=None):
        if self.modes is None:
            if mode_max is None:
                raise ValueError("mode_max is required when modes are not stored")
            mode_max = int(mode_max)
            if self.basis == "fourier":
                return np.arange(-mode_max, mode_max + 1)
            return np.arange(0, mode_max + 1)
        modes = np.asarray(self.modes, dtype=int)
        if mode_max is None:
            return modes
        return modes[np.abs(modes) <= int(mode_max)]

    @staticmethod
    def _sum_modes(coeff, weights):
        """Sum multipole coefficients against opening-angle weights.

        ``coeff`` has shape ``(nmode, *ell_shape)`` and ``weights`` has shape
        ``(nmode, *angle_shape)``.  The result has shape
        ``(*ell_shape, *angle_shape)``.
        """
        coeff = np.asarray(coeff)
        weights = np.asarray(weights)
        if coeff.shape[0] != weights.shape[0]:
            raise ValueError("mode axis mismatch between coefficients and weights")
        coeff_view = coeff.reshape(coeff.shape + (1,) * (weights.ndim - 1))
        weights_view = weights.reshape((weights.shape[0],) + (1,) * (coeff.ndim - 1) + weights.shape[1:])
        out = np.sum(coeff_view * weights_view, axis=0)
        return out.item() if out.shape == () else out

    def interpolate(self, config=None, **params):
        """Tabulate this lazy multipole object and return an interpolated wrapper.

        The public ``multipole()`` and LOS ``project()`` methods remain lazy.
        Interpolation is an explicit, opt-in step requested here.
        """
        config = config or BispectrumMultipole2DConfig()
        if getattr(self, "_evaluator", None) is not None and hasattr(self._evaluator, "interpolate"):
            return self._evaluator.interpolate(config, owner=self, **params)
        from .interpolate import InterpolatedBispectrumMultipole2D
        return InterpolatedBispectrumMultipole2D.from_multipole(self, config, **params)

    def resum(self, ell2, ell3, delta_beta, mode_max=None):
        """Reconstruct ``B(ell2,ell3,Delta beta)`` around ``ell1``."""
        modes = self.available_modes(mode_max)
        coeff = self.evaluate(modes, ell2, ell3)
        delta_beta = np.asarray(delta_beta)

        if self.basis == "legendre":
            # Legendre basis remains an inner-angle expansion.
            alpha = np.pi - delta_beta
            weights = np.array([eval_legendre(m, np.cos(alpha)) for m in modes])
            return self._sum_modes(coeff, weights)

        if self.basis == "fourier-even":
            if np.any(modes < 0):
                raise ValueError("fourier-even basis expects non-negative modes")
            weights = np.cos(modes.reshape((-1,) + (1,) * delta_beta.ndim) * delta_beta)
            return self._sum_modes(coeff, weights)

        if self.basis == "cosine":
            weights = np.cos(modes.reshape((-1,) + (1,) * delta_beta.ndim) * delta_beta)
            return self._sum_modes(coeff, weights)

        if self.basis == "sine":
            weights = np.sin(modes.reshape((-1,) + (1,) * delta_beta.ndim) * delta_beta)
            return self._sum_modes(coeff, weights)

        weights = np.exp(1j * modes.reshape((-1,) + (1,) * delta_beta.ndim) * delta_beta)
        return self._sum_modes(coeff, weights)

    def resum_alpha(self, ell2, ell3, alpha, mode_max=None):
        """Convenience wrapper for inner-angle inputs.

        Since ``alpha = pi - Delta beta``, this converts to the package's
        outer-angle convention before calling ``resum``.
        """
        return self.resum(ell2, ell3, np.pi - np.asarray(alpha), mode_max=mode_max)


class BispectrumMultipole3D:
    """Base object for a 3D bispectrum multipole ``B_L(k2,k3,z)``.

    This class represents the result of a multipole expansion performed before
    line-of-sight projection.  A projected angular multipole is obtained by
    integrating this object over the same LOS kernels used for ordinary 3D
    bispectrum projection.
    """

    basis: str = "fourier-even"
    support = None

    def __call__(self, mode, k2, k3, z, **params):
        return self.evaluate(mode, k2, k3, z, **params)

    def evaluate(self, mode, k2, k3, z, **params):
        raise NotImplementedError

    def interpolate(self, config, **params):
        from .interpolate import InterpolatedBispectrumMultipole3D
        return InterpolatedBispectrumMultipole3D.from_multipole(self, config, **params)

    def project_los(self, projector, sample_combination=None, modes=None, mode_max=None):
        """Project this 3D multipole with a line-of-sight projector.

        ``projector`` may be either a ``LineOfSightProjector`` or a
        ``MultipoleLineOfSightProjector``.  The conversion is delayed to avoid
        an import cycle between ``multipole.py`` and ``los.py``.
        """
        from .los import LineOfSightProjector

        if isinstance(projector, LineOfSightProjector):
            projector = projector.as_multipole_projector()
        out = projector.project(self, sample_combination=sample_combination)
        if modes is None and mode_max is not None:
            mode_max = int(mode_max)
            if getattr(self, "basis", "fourier-even") == "fourier":
                modes = np.arange(-mode_max, mode_max + 1)
            else:
                modes = np.arange(0, mode_max + 1)
        if modes is not None:
            out.modes = np.asarray(modes, dtype=int)
        return out


class _LoSBispectrumMultipole2DEvaluator:
    """Internal evaluator for the 3D-multipole -> 2D-multipole route."""

    def __init__(self, multipole3d, projector, sample_combination=None):
        self.multipole3d = multipole3d
        self.projector = projector
        self.sample_combination = tuple(sample_combination) if sample_combination is not None else None

    def warm(self, modes):
        """Warm the underlying 3D semi-analytic caches on the LOS grid."""
        warm_cache = getattr(self.multipole3d, "warm_cache", None)
        if warm_cache is None:
            raise TypeError(
                f"{type(self.multipole3d).__name__} does not implement warm_cache"
            )
        warm_cache(
            z=self.projector.z,
            modes=np.atleast_1d(np.asarray(modes, dtype=int)),
        )

    def __call__(self, mode, ell2, ell3):
        if np.isscalar(mode):
            return self.projector.evaluate(
                self.multipole3d,
                mode,
                ell2,
                ell3,
                sample_combination=self.sample_combination,
            )

        modes = np.atleast_1d(np.asarray(mode, dtype=int))
        return np.asarray([
            self.projector.evaluate(
                self.multipole3d,
                int(m),
                ell2,
                ell3,
                sample_combination=self.sample_combination,
            )
            for m in modes
        ])


class _Bispectrum2DMultipoleEvaluator:
    """Internal evaluator for the 2D-bispectrum -> 2D-multipole route."""

    def __init__(self, bispectrum, config, basis, regulator=None, params=None):
        self.bispectrum = bispectrum
        self.config = config
        self.basis = basis
        self.regulator = regulator
        self.params = dict(params or {})
        self.calculator = BispectrumMultipole2DCalculator(config=config, basis=basis)

    def default_modes(self):
        c = self.config
        if self.basis == "fourier":
            return np.arange(-c.mode_max, c.mode_max + 1)
        return np.arange(0, c.mode_max + 1)

    def __call__(self, mode, ell2, ell3):
        return self.calculator.evaluate_points(
            self.bispectrum,
            mode,
            ell2,
            ell3,
            regulator=self.regulator,
            **self.params,
        )

    def interpolate(self, config=None, owner=None, **params):
        run_params = dict(self.params)
        run_params.update(params)
        calculator = BispectrumMultipole2DCalculator(config=config or self.config, basis=self.basis)
        grid = calculator.compute(
            self.bispectrum,
            regulator=self.regulator,
            **run_params,
        )
        from .interpolate import InterpolatedBispectrumMultipole2D
        return InterpolatedBispectrumMultipole2D.from_grid(grid, base=owner)


@dataclass(frozen=True)
class BispectrumMultipole2DGrid:
    """Tabulated 2D multipole coefficients on an ``(ell, psi)`` grid.

    This is deliberately only data.  Use :meth:`as_interpolated` or
    :meth:`BispectrumMultipole2D.interpolate` when an evaluation object is
    required.
    """

    modes: np.ndarray
    ell_grid: np.ndarray
    psi_grid: np.ndarray
    values: np.ndarray
    basis: str
    method: str = "linear"
    angle: str = "delta_beta"

    def as_interpolated(self, base=None):
        from .interpolate import InterpolatedBispectrumMultipole2D
        return InterpolatedBispectrumMultipole2D.from_grid(self, base=base)



class BispectrumMultipole2DCalculator:
    def __init__(self, config: Optional[BispectrumMultipole2DConfig] = None, basis: str = "fourier-even"):
        self.config = config or BispectrumMultipole2DConfig()
        self.basis = basis

    def _grid_config(self):
        c = self.config
        return MultipoleGridConfig(
            ell_min=c.ell_min,
            ell_max=c.ell_max,
            n_ell=c.n_ell,
            psi_min=c.psi_min,
            psi_max=c.psi_max,
            n_psi=c.n_psi,
            delta_beta_min=c.delta_beta_min,
            delta_beta_max=c.delta_beta_max,
            n_delta_beta_lin=c.n_delta_beta_lin,
            n_delta_beta_log=c.n_delta_beta_log,
            delta_beta_transition=c.delta_beta_transition,
            angle_sampling=c.angle_sampling,
        )

    def _decomposer(self, x):
        c = self.config
        if self.basis == "legendre":
            return MultipoleLegendre(x, c.mode_max, method=c.decomposer_method)
        if self.basis in {"fourier-even", "cosine"}:
            return MultipoleCosine(x, c.mode_max, method=c.decomposer_method)
        if self.basis == "sine":
            return MultipoleSine(x, c.mode_max, method=c.decomposer_method)
        if self.basis == "fourier":
            return MultipoleFourier(x, c.mode_max, method=c.decomposer_method)
        raise ValueError(f"unsupported basis: {self.basis}")

    def evaluate_points(self, bispectrum: Bispectrum2D, mode, ell2, ell3, regulator=None, **params):
        """Evaluate selected multipole modes at arbitrary ``(ell2, ell3)``.

        This is the lazy/brute-force path used by ``Bispectrum2D.multipole``.
        It performs the angle decomposition only for the requested points, in
        contrast to ``compute`` which tabulates a full interpolation grid.
        """
        c = self.config
        grid = self._grid_config()
        delta_beta = make_delta_beta_grid(grid)

        scalar_mode = np.isscalar(mode)
        modes = np.atleast_1d(np.asarray(mode, dtype=int))

        ell2 = np.asarray(ell2, dtype=float)
        ell3 = np.asarray(ell3, dtype=float)
        ell2, ell3 = np.broadcast_arrays(ell2, ell3)
        shape = ell2.shape

        e2 = ell2[..., None]
        e3 = ell3[..., None]
        db = delta_beta.reshape((1,) * ell2.ndim + (-1,))
        e1 = ell2ell3delta_beta_to_ell1(e2, e3, db)

        # Some Bispectrum2D implementations accept broadcastable inputs, but
        # LOS-projected bispectra require ell1, ell2, and ell3 to have exactly
        # the same shape before they are flattened onto the LOS grid.  Broadcast
        # explicitly here so point-wise multipole evaluation works uniformly for
        # direct, projected, and interpolated bispectra.
        e1, e2, e3 = np.broadcast_arrays(e1, e2, e3)

        values = bispectrum(e1, e2, e3, **params)
        if regulator is not None:
            values = values * regulator(e1, e2, e3)

        if self.basis == "legendre":
            mu_inner = -np.cos(delta_beta)
            coeff = self._decomposer(mu_inner).decompose(values, modes, axis=-1)

        elif self.basis == "fourier-even":
            norm = np.full_like(modes, 2.0 / np.pi, dtype=float)
            norm[modes == 0] = 1.0 / np.pi
            if c.decomposition_angle == "outer":
                raw = self._decomposer(delta_beta).decompose(values, modes, axis=-1)
                coeff = raw * norm.reshape((-1,) + (1,) * len(shape))
            elif c.decomposition_angle == "inner":
                alpha_inc = (np.pi - delta_beta)[::-1]
                values_inc = values[..., ::-1]
                raw_inner = MultipoleCosine(
                    alpha_inc,
                    c.mode_max,
                    method=c.decomposer_method,
                ).decompose(values_inc, modes, axis=-1)
                sign = (-1.0) ** modes
                coeff = raw_inner * norm.reshape((-1,) + (1,) * len(shape)) * sign.reshape((-1,) + (1,) * len(shape))
            else:
                raise ValueError("decomposition_angle must be 'outer' or 'inner'")

        elif self.basis in {"cosine", "sine"}:
            if c.decomposition_angle == "outer":
                coeff = self._decomposer(delta_beta).decompose(values, modes, axis=-1)
            elif c.decomposition_angle == "inner":
                alpha_inc = (np.pi - delta_beta)[::-1]
                values_inc = values[..., ::-1]
                if self.basis == "cosine":
                    raw_inner = MultipoleCosine(
                        alpha_inc,
                        c.mode_max,
                        method=c.decomposer_method,
                    ).decompose(values_inc, modes, axis=-1)
                    sign = (-1.0) ** modes
                else:
                    raw_inner = MultipoleSine(
                        alpha_inc,
                        c.mode_max,
                        method=c.decomposer_method,
                    ).decompose(values_inc, modes, axis=-1)
                    sign = (-1.0) ** (modes + 1)
                coeff = raw_inner * sign.reshape((-1,) + (1,) * len(shape))
            else:
                raise ValueError("decomposition_angle must be 'outer' or 'inner'")

        elif self.basis == "fourier":
            # Full complex Fourier coefficients in the OUTER angle,
            #
            #   B_L = (1/2pi) int_{-pi}^{pi} B(delta) exp(-i L delta) ddelta.
            #
            # Current Bispectrum2D objects are side-only and therefore even in
            # delta, so this is equivalent to
            #
            #   B_L = (1/pi) int_0^pi B(delta) cos(L delta) ddelta.
            #
            # This avoids endpoint duplication and also supports the preferred
            # inner-angle sampling through cos[L(pi-alpha)] = (-1)^L cos(L alpha).
            modes_abs = np.abs(modes)
            norm = np.full_like(modes, 1.0 / np.pi, dtype=float)
            if c.decomposition_angle == "outer":
                raw = MultipoleCosine(
                    delta_beta,
                    c.mode_max,
                    method=c.decomposer_method,
                ).decompose(values, modes_abs, axis=-1)
                coeff = raw * norm.reshape((-1,) + (1,) * len(shape))
            elif c.decomposition_angle == "inner":
                alpha_inc = (np.pi - delta_beta)[::-1]
                values_inc = values[..., ::-1]
                raw_inner = MultipoleCosine(
                    alpha_inc,
                    c.mode_max,
                    method=c.decomposer_method,
                ).decompose(values_inc, modes_abs, axis=-1)
                sign = (-1.0) ** modes
                coeff = raw_inner * norm.reshape((-1,) + (1,) * len(shape)) * sign.reshape((-1,) + (1,) * len(shape))
            else:
                raise ValueError("decomposition_angle must be 'outer' or 'inner'")
        else:
            raise ValueError(f"unsupported basis: {self.basis}")

        return coeff[0] if scalar_mode else coeff

    def compute(self, bispectrum: Bispectrum2D, regulator=None, **params):
        c = self.config
        ell, psi, delta_beta, e1, e2, e3 = make_ell_psi_delta_beta_grid(self._grid_config())
        values = bispectrum(e1, e2, e3, **params)
        if regulator is not None:
            values = values * regulator(e1, e2, e3)

        if self.basis == "legendre":
            # Legendre multipoles are kept as inner-angle multipoles.  Since
            # alpha = pi - Delta beta, mu_inner = cos(alpha) = -cos(Delta beta).
            # The Delta-beta grid is increasing, so mu_inner is also increasing.
            mu_inner = -np.cos(delta_beta)
            modes = np.arange(0, c.mode_max + 1)
            coeff = self._decomposer(mu_inner).decompose(values, modes, axis=2)

        elif self.basis == "fourier-even":
            # Even Fourier/cosine series in the OUTER angle Delta beta:
            # B(delta)=c0+sum_{q>=1} c_q cos(q delta).
            #
            # For squeezed-aware quadrature we may decompose in the inner angle
            # alpha = pi - delta.  Since cos(q delta) = (-1)^q cos(q alpha),
            # the stored coefficient is converted back to the outer-angle
            # convention by multiplying by (-1)^q.
            modes = np.arange(0, c.mode_max + 1)
            norm = np.full_like(modes, 2.0 / np.pi, dtype=float)
            norm[0] = 1.0 / np.pi

            if c.decomposition_angle == "outer":
                raw = self._decomposer(delta_beta).decompose(values, modes, axis=2)
                coeff = raw * norm[:, None, None]
            elif c.decomposition_angle == "inner":
                alpha_inc = (np.pi - delta_beta)[::-1]
                values_inc = values[..., ::-1]
                raw_inner = MultipoleCosine(
                    alpha_inc,
                    c.mode_max,
                    method=c.decomposer_method,
                ).decompose(values_inc, modes, axis=2)
                sign = (-1.0) ** modes
                coeff = raw_inner * norm[:, None, None] * sign[:, None, None]
            else:
                raise ValueError("decomposition_angle must be 'outer' or 'inner'")

        elif self.basis in {"cosine", "sine"}:
            # Unnormalized projections.  Stored coefficients are always in the
            # outer-angle convention even when projected in the inner angle.
            modes = np.arange(0, c.mode_max + 1)
            if c.decomposition_angle == "outer":
                coeff = self._decomposer(delta_beta).decompose(values, modes, axis=2)
            elif c.decomposition_angle == "inner":
                alpha_inc = (np.pi - delta_beta)[::-1]
                values_inc = values[..., ::-1]
                if self.basis == "cosine":
                    raw_inner = MultipoleCosine(
                        alpha_inc,
                        c.mode_max,
                        method=c.decomposer_method,
                    ).decompose(values_inc, modes, axis=2)
                    sign = (-1.0) ** modes
                else:
                    raw_inner = MultipoleSine(
                        alpha_inc,
                        c.mode_max,
                        method=c.decomposer_method,
                    ).decompose(values_inc, modes, axis=2)
                    sign = (-1.0) ** (modes + 1)
                coeff = raw_inner * sign[:, None, None]
            else:
                raise ValueError("decomposition_angle must be 'outer' or 'inner'")

        elif self.basis == "fourier":
            # Full complex Fourier coefficients in the outer-angle convention:
            #
            #   B_L = (1/2pi) int_{-pi}^{pi} B(delta) exp(-i L delta) ddelta.
            #
            # For side-only bispectra B(delta)=B(-delta), so
            #
            #   B_L = (1/pi) int_0^pi B(delta) cos(L delta) ddelta,
            #
            # and B_{+L}=B_{-L}.  This is the full-Fourier representation used
            # by the spin-3PCF H-kernel pipeline.
            modes = np.arange(-c.mode_max, c.mode_max + 1)
            modes_abs = np.abs(modes)
            norm = np.full_like(modes, 1.0 / np.pi, dtype=float)

            if c.decomposition_angle == "outer":
                raw = MultipoleCosine(
                    delta_beta,
                    c.mode_max,
                    method=c.decomposer_method,
                ).decompose(values, modes_abs, axis=2)
                coeff = raw * norm[:, None, None]
            elif c.decomposition_angle == "inner":
                alpha_inc = (np.pi - delta_beta)[::-1]
                values_inc = values[..., ::-1]
                raw_inner = MultipoleCosine(
                    alpha_inc,
                    c.mode_max,
                    method=c.decomposer_method,
                ).decompose(values_inc, modes_abs, axis=2)
                sign = (-1.0) ** modes
                coeff = raw_inner * norm[:, None, None] * sign[:, None, None]
            else:
                raise ValueError("decomposition_angle must be 'outer' or 'inner'")

        else:
            raise ValueError(f"unsupported basis: {self.basis}")

        return BispectrumMultipole2DGrid(
            modes=np.asarray(modes, dtype=int),
            ell_grid=np.asarray(ell, dtype=float),
            psi_grid=np.asarray(psi, dtype=float),
            values=np.asarray(coeff),
            basis=self.basis,
            method=c.interpolation,
            angle="delta_beta",
        )
