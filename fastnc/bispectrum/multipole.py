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
from scipy.interpolate import RegularGridInterpolator
from scipy.special import eval_legendre

from .base import Bispectrum2D
from .decompose import MultipoleLegendre, MultipoleFourier, MultipoleCosine, MultipoleSine
from .grids import (
    MultipoleGridConfig,
    make_ell_psi_delta_beta_grid,
    sides_to_ellpsidelta_beta,
    fold_psi,
)


@dataclass(frozen=True)
class BispectrumMultipoleConfig:
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


class BispectrumMultipole2D:
    basis: str

    def __call__(self, mode, ell1, ell2):
        return self.evaluate(mode, ell1, ell2)

    def evaluate(self, mode, ell1, ell2):
        raise NotImplementedError

    def resum(self, ell1, ell2, delta_beta, mode_max=None):
        raise NotImplementedError

    def resum_alpha(self, ell1, ell2, alpha, mode_max=None):
        """Convenience wrapper for inner-angle inputs.

        Since ``alpha = pi - Delta beta``, this converts to the package's
        outer-angle convention before calling ``resum``.
        """
        return self.resum(ell1, ell2, np.pi - np.asarray(alpha), mode_max=mode_max)


class AnalyticBispectrumMultipole2D(BispectrumMultipole2D):
    def __init__(self, func: Callable, basis: str = "fourier-even", modes=None):
        self.func = func
        self.basis = basis
        self.modes = np.asarray(modes) if modes is not None else None

    def evaluate(self, mode, ell1, ell2):
        return self.func(mode, ell1, ell2)

    def resum(self, ell1, ell2, delta_beta, mode_max=None):
        if self.modes is None and mode_max is None:
            raise ValueError("mode_max is required when modes are not stored")
        modes = self.modes if mode_max is None else np.arange(0, mode_max + 1)
        coeff = self.evaluate(modes[:, None], ell1, ell2)
        delta_beta = np.asarray(delta_beta)

        if self.basis == "legendre":
            # Legendre basis is still defined in the inner angle alpha.
            alpha = np.pi - delta_beta
            return np.sum(coeff * np.array([eval_legendre(m, np.cos(alpha)) for m in modes]), axis=0)

        if self.basis == "fourier-even":
            out = np.array(coeff[0], copy=True)
            if len(modes) > 1:
                out += np.sum(coeff[1:] * np.cos(modes[1:, None] * delta_beta), axis=0)
            return out

        return np.sum(coeff * np.exp(1j * modes[:, None] * delta_beta), axis=0)


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


class ProjectedBispectrumMultipole2D(AnalyticBispectrumMultipole2D):
    """LOS-projected angular multipole built from a 3D multipole object."""

    def __init__(self, multipole3d, projector, sample_combination=None, modes=None):
        self.multipole3d = multipole3d
        self.projector = projector
        self.sample_combination = tuple(sample_combination) if sample_combination is not None else None
        super().__init__(self._evaluate_mode, basis=multipole3d.basis, modes=modes)

    def _evaluate_mode(self, mode, ell1, ell2):
        return self.projector.evaluate(
            self.multipole3d,
            mode,
            ell1,
            ell2,
            sample_combination=self.sample_combination,
        )


class InterpolatedBispectrumMultipole2D(BispectrumMultipole2D):
    def __init__(self, modes, ell_grid, psi_grid, values, basis: str, method="linear", angle="delta_beta"):
        self.modes = np.asarray(modes, dtype=int)
        self.ell_grid = np.asarray(ell_grid, dtype=float)
        self.psi_grid = np.asarray(psi_grid, dtype=float)
        self.values = np.asarray(values)
        self.basis = basis
        self.method = method
        self.angle = angle
        self._mode_to_index = {int(m): i for i, m in enumerate(self.modes)}
        self._interpolators = {}
        for i, m in enumerate(self.modes):
            self._interpolators[int(m)] = RegularGridInterpolator(
                (np.log(self.ell_grid), np.log(self.psi_grid)),
                self.values[i],
                method=method,
                bounds_error=False,
                fill_value=None,
            )

    def available_modes(self, mode_max=None):
        if mode_max is None:
            return self.modes
        return self.modes[np.abs(self.modes) <= mode_max]

    def evaluate(self, mode, ell1, ell2):
        scalar_mode = np.isscalar(mode)
        modes = np.atleast_1d(np.asarray(mode, dtype=int))

        ell1 = np.asarray(ell1, dtype=float)
        ell2 = np.asarray(ell2, dtype=float)
        ell1, ell2 = np.broadcast_arrays(ell1, ell2)
        shape = ell1.shape

        ell = np.sqrt(ell1**2 + ell2**2)
        psi = fold_psi(np.arctan2(ell2, ell1))

        # Avoid uncontrolled extrapolation in the interpolation object.
        ell = np.clip(ell, self.ell_grid.min(), self.ell_grid.max())
        psi = np.clip(psi, self.psi_grid.min(), self.psi_grid.max())

        pts = np.column_stack([np.log(ell.ravel()), np.log(psi.ravel())])

        out = []
        for m in modes:
            if int(m) not in self._interpolators:
                raise ValueError(f"mode {m} is not available")
            out.append(self._interpolators[int(m)](pts).reshape(shape))
        out = np.asarray(out)
        return out[0] if scalar_mode else out

    def resum(self, ell1, ell2, delta_beta, mode_max=None):
        modes = self.available_modes(mode_max)
        coeff = self.evaluate(modes, ell1, ell2)
        delta_beta = np.asarray(delta_beta)

        if self.basis == "legendre":
            # Legendre basis remains an inner-angle expansion.
            alpha = np.pi - delta_beta
            p = np.array([eval_legendre(m, np.cos(alpha)) for m in modes])
            return np.sum(coeff * p, axis=0)

        if self.basis == "fourier-even":
            out = np.array(coeff[0], copy=True)
            if modes.size > 1:
                out += np.sum(coeff[1:] * np.cos(modes[1:, None] * delta_beta), axis=0)
            return out

        if self.basis == "cosine":
            return np.sum(coeff * np.cos(modes[:, None] * delta_beta), axis=0)

        if self.basis == "sine":
            return np.sum(coeff * np.sin(modes[:, None] * delta_beta), axis=0)

        return np.sum(coeff * np.exp(1j * modes[:, None] * delta_beta), axis=0)


class BispectrumMultipole2DCalculator:
    def __init__(self, config: Optional[BispectrumMultipoleConfig] = None, basis: str = "fourier-even"):
        self.config = config or BispectrumMultipoleConfig()
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

        return InterpolatedBispectrumMultipole2D(
            modes=modes,
            ell_grid=ell,
            psi_grid=psi,
            values=coeff,
            basis=self.basis,
            method=c.interpolation,
            angle="delta_beta",
        )
