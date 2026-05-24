"""Angular-bispectrum multipole objects and calculators.

Fourier multipoles are defined with respect to the outer angle

    Delta beta = beta1 - beta2,

not the inner angle alpha.  For side-only bispectra the third side is

    ell3^2 = ell1^2 + ell2^2 + 2 ell1 ell2 cos(Delta beta).

The default ``basis='fourier-even'`` convention is

    B(ell1, ell2, Delta beta)
      = c0(ell1, ell2) + sum_{q>=1} cq(ell1, ell2) cos(q Delta beta),

with

    c0 = (1/pi) int_0^pi dDelta beta B,
    cq = (2/pi) int_0^pi dDelta beta B cos(q Delta beta).

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
    ell1ell2delta_beta_to_ell3,
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
    psi_max: float = np.pi / 4
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

    This is the fiducial public object for ``B_mode(ell1, ell2)``.  Different
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

    def __call__(self, mode, ell1, ell2):
        return self.evaluate(mode, ell1, ell2)

    def evaluate(self, mode, ell1, ell2):
        if getattr(self, "_evaluator", None) is None:
            raise NotImplementedError
        return self._evaluator(mode, ell1, ell2)

    def available_modes(self, mode_max=None):
        if self.modes is None:
            if mode_max is None:
                raise ValueError("mode_max is required when modes are not stored")
            return np.arange(0, int(mode_max) + 1)
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

    def resum(self, ell1, ell2, delta_beta, mode_max=None):
        modes = self.available_modes(mode_max)
        coeff = self.evaluate(modes, ell1, ell2)
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

    def resum_alpha(self, ell1, ell2, alpha, mode_max=None):
        """Convenience wrapper for inner-angle inputs.

        Since ``alpha = pi - Delta beta``, this converts to the package's
        outer-angle convention before calling ``resum``.
        """
        return self.resum(ell1, ell2, np.pi - np.asarray(alpha), mode_max=mode_max)


class BispectrumMultipole3D:
    """Base object for a 3D bispectrum multipole ``B_L(k1,k2,z)``.

    This class represents the result of a multipole expansion performed before
    line-of-sight projection.  A projected angular multipole is obtained by
    integrating this object over the same LOS kernels used for ordinary 3D
    bispectrum projection.
    """

    basis: str = "fourier-even"
    support = None

    def __call__(self, mode, k1, k2, z, **params):
        return self.evaluate(mode, k1, k2, z, **params)

    def evaluate(self, mode, k1, k2, z, **params):
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
            modes = np.arange(0, int(mode_max) + 1)
        if modes is not None:
            out.modes = np.asarray(modes, dtype=int)
        return out


class _LoSBispectrumMultipole2DEvaluator:
    """Internal evaluator for the 3D-multipole -> 2D-multipole route."""

    def __init__(self, multipole3d, projector, sample_combination=None):
        self.multipole3d = multipole3d
        self.projector = projector
        self.sample_combination = tuple(sample_combination) if sample_combination is not None else None

    def __call__(self, mode, ell1, ell2):
        if np.isscalar(mode):
            return self.projector.evaluate(
                self.multipole3d,
                mode,
                ell1,
                ell2,
                sample_combination=self.sample_combination,
            )

        modes = np.atleast_1d(np.asarray(mode, dtype=int))
        return np.asarray([
            self.projector.evaluate(
                self.multipole3d,
                int(m),
                ell1,
                ell2,
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

    def __call__(self, mode, ell1, ell2):
        return self.calculator.evaluate_points(
            self.bispectrum,
            mode,
            ell1,
            ell2,
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

    def evaluate_points(self, bispectrum: Bispectrum2D, mode, ell1, ell2, regulator=None, **params):
        """Evaluate selected multipole modes at arbitrary ``(ell1, ell2)``.

        This is the lazy/brute-force path used by ``Bispectrum2D.multipole``.
        It performs the angle decomposition only for the requested points, in
        contrast to ``compute`` which tabulates a full interpolation grid.
        """
        c = self.config
        grid = self._grid_config()
        delta_beta = make_delta_beta_grid(grid)

        scalar_mode = np.isscalar(mode)
        modes = np.atleast_1d(np.asarray(mode, dtype=int))

        ell1 = np.asarray(ell1, dtype=float)
        ell2 = np.asarray(ell2, dtype=float)
        ell1, ell2 = np.broadcast_arrays(ell1, ell2)
        shape = ell1.shape

        e1 = ell1[..., None]
        e2 = ell2[..., None]
        db = delta_beta.reshape((1,) * ell1.ndim + (-1,))
        e3 = ell1ell2delta_beta_to_ell3(e1, e2, db)

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
            if c.decomposition_angle == "inner":
                raise NotImplementedError(
                    "inner-angle decomposition for complex Fourier basis is not implemented yet"
                )
            delta_signed = np.concatenate([-delta_beta[::-1], delta_beta])
            values_signed = np.concatenate([values[..., ::-1], values], axis=-1)
            raw = MultipoleFourier(delta_signed, c.mode_max, method=c.decomposer_method).decompose(
                values_signed, -modes, axis=-1
            )
            coeff = raw / (2.0 * np.pi)
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
            if c.decomposition_angle == "inner":
                raise NotImplementedError(
                    "inner-angle decomposition for complex Fourier basis is not implemented yet"
                )
            # Complex convention in Delta beta:
            # B_q = (1/2pi) int_{-pi}^{pi} B(delta) exp(-i q delta) d delta.
            # A side-only bispectrum is even in Delta beta, but this basis is
            # kept for API completeness and future asymmetric extensions.
            delta_signed = np.concatenate([-delta_beta[::-1], delta_beta])
            values_signed = np.concatenate([values[..., ::-1], values], axis=2)
            modes = np.arange(-c.mode_max, c.mode_max + 1)
            raw = MultipoleFourier(delta_signed, c.mode_max, method=c.decomposer_method).decompose(
                values_signed, -modes, axis=2
            )
            coeff = raw / (2.0 * np.pi)

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
