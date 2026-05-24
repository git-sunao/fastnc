"""Interpolation wrappers for bispectrum-like objects.

The classes in this module are deliberately thin wrappers around an existing
bispectrum object.  They expose the same public evaluation signatures as the
objects they wrap, so downstream code can switch between brute-force and
interpolated evaluations without changing the rest of the pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from .base import Bispectrum2D, Bispectrum3D
from .multipole import (
    BispectrumMultipole2D, BispectrumMultipole2DConfig,
    BispectrumMultipole2DGrid, BispectrumMultipole3D,
)
from .support import Support2D, Support3D
from .grids import (
    MultipoleGridConfig,
    make_ell_psi_delta_beta_grid,
    ellpsi_to_ell1ell2,
    fold_psi,
)


@dataclass(frozen=True)
class Bispectrum3DInterpolationConfig:
    """Grid configuration for interpolating ``B(k1,k2,k3,z)``.

    The interpolation is performed in ``(log k1, log k2, log k3, z)`` by
    default.  Values may optionally be stored in signed-log form, which is more
    robust when the bispectrum is positive or mostly sign-definite but still
    permits sign changes.
    """

    k_min: float
    k_max: float
    n_k: int
    z_min: float
    z_max: float
    n_z: int
    method: str = "linear"
    log_k: bool = True
    value_transform: str = "identity"  # identity, log, signed-log
    floor: float = 1.0e-300

    @classmethod
    def from_support(
        cls,
        support: Support3D,
        *,
        n_k: int = 64,
        n_z: int = 32,
        method: str = "linear",
        log_k: bool = True,
        value_transform: str = "identity",
        floor: float = 1.0e-300,
    ) -> "Bispectrum3DInterpolationConfig":
        if not np.isfinite(support.k_min) or not np.isfinite(support.k_max):
            raise ValueError("finite support.k_min and support.k_max are required")
        if not np.isfinite(support.z_min) or not np.isfinite(support.z_max):
            raise ValueError("finite support.z_min and support.z_max are required")
        k_min = max(float(support.k_min), np.finfo(float).tiny)
        return cls(
            k_min=k_min,
            k_max=float(support.k_max),
            n_k=int(n_k),
            z_min=float(support.z_min),
            z_max=float(support.z_max),
            n_z=int(n_z),
            method=method,
            log_k=log_k,
            value_transform=value_transform,
            floor=floor,
        )

    def k_grid(self):
        if self.log_k:
            return np.geomspace(self.k_min, self.k_max, self.n_k)
        return np.linspace(self.k_min, self.k_max, self.n_k)

    def z_grid(self):
        return np.linspace(self.z_min, self.z_max, self.n_z)


@dataclass(frozen=True)
class Bispectrum2DInterpolationConfig:
    """Grid configuration for interpolating ``B(ell1,ell2,ell3)``."""

    ell_min: float
    ell_max: float
    n_ell: int
    method: str = "linear"
    log_ell: bool = True
    value_transform: str = "identity"  # identity, log, signed-log
    floor: float = 1.0e-300

    @classmethod
    def from_support(
        cls,
        support: Support2D,
        *,
        n_ell: int = 64,
        method: str = "linear",
        log_ell: bool = True,
        value_transform: str = "identity",
        floor: float = 1.0e-300,
    ) -> "Bispectrum2DInterpolationConfig":
        if not np.isfinite(support.ell_min) or not np.isfinite(support.ell_max):
            raise ValueError("finite support.ell_min and support.ell_max are required")
        ell_min = max(float(support.ell_min), np.finfo(float).tiny)
        return cls(
            ell_min=ell_min,
            ell_max=float(support.ell_max),
            n_ell=int(n_ell),
            method=method,
            log_ell=log_ell,
            value_transform=value_transform,
            floor=floor,
        )

    def ell_grid(self):
        if self.log_ell:
            return np.geomspace(self.ell_min, self.ell_max, self.n_ell)
        return np.linspace(self.ell_min, self.ell_max, self.n_ell)


@dataclass(frozen=True)
class BispectrumMultipole3DInterpolationConfig:
    """Grid configuration for interpolating ``B_L(k1,k2,z)``."""

    modes: tuple[int, ...]
    k_min: float
    k_max: float
    n_k: int
    z_min: float
    z_max: float
    n_z: int
    method: str = "linear"
    log_k: bool = True
    value_transform: str = "identity"  # identity, log, signed-log
    floor: float = 1.0e-300

    @classmethod
    def from_support(
        cls,
        support: Support3D,
        *,
        modes,
        n_k: int = 64,
        n_z: int = 32,
        method: str = "linear",
        log_k: bool = True,
        value_transform: str = "identity",
        floor: float = 1.0e-300,
    ) -> "BispectrumMultipole3DInterpolationConfig":
        if not np.isfinite(support.k_min) or not np.isfinite(support.k_max):
            raise ValueError("finite support.k_min and support.k_max are required")
        if not np.isfinite(support.z_min) or not np.isfinite(support.z_max):
            raise ValueError("finite support.z_min and support.z_max are required")
        k_min = max(float(support.k_min), np.finfo(float).tiny)
        return cls(
            modes=tuple(int(m) for m in modes),
            k_min=k_min,
            k_max=float(support.k_max),
            n_k=int(n_k),
            z_min=float(support.z_min),
            z_max=float(support.z_max),
            n_z=int(n_z),
            method=method,
            log_k=log_k,
            value_transform=value_transform,
            floor=floor,
        )

    def k_grid(self):
        if self.log_k:
            return np.geomspace(self.k_min, self.k_max, self.n_k)
        return np.linspace(self.k_min, self.k_max, self.n_k)

    def z_grid(self):
        return np.linspace(self.z_min, self.z_max, self.n_z)


def _axis_from_grid(grid, *, log_axis: bool):
    grid = np.asarray(grid, dtype=float)
    if log_axis:
        if np.any(grid <= 0.0):
            raise ValueError("log interpolation requires strictly positive grid values")
        return np.log(grid)
    return grid


def _coords_from_values(values, *, log_axis: bool):
    values = np.asarray(values, dtype=float)
    if log_axis:
        if np.any(values <= 0.0):
            raise ValueError("log interpolation requires strictly positive query values")
        return np.log(values)
    return values


def _pack_values(values, transform: str, floor: float):
    values = np.asarray(values)
    if transform == "identity":
        return values
    if transform == "log":
        if np.any(values <= 0.0):
            raise ValueError("value_transform='log' requires positive values")
        return np.log(np.maximum(values, floor))
    if transform == "signed-log":
        return np.sign(values) * np.log1p(np.abs(values) / floor)
    raise ValueError("value_transform must be 'identity', 'log', or 'signed-log'")


def _unpack_values(values, transform: str, floor: float):
    if transform == "identity":
        return values
    if transform == "log":
        return np.exp(values)
    if transform == "signed-log":
        return np.sign(values) * floor * np.expm1(np.abs(values))
    raise ValueError("value_transform must be 'identity', 'log', or 'signed-log'")


class InterpolatedBispectrum3D(Bispectrum3D):
    """Interpolated wrapper with the same signature as ``Bispectrum3D``."""

    def __init__(self, base: Bispectrum3D, config: Bispectrum3DInterpolationConfig, values=None, **params):
        self.base = base
        self.config = config
        self.k_grid = config.k_grid()
        self.z_grid = config.z_grid()
        self.support = Support3D(
            k_min=float(self.k_grid.min()),
            k_max=float(self.k_grid.max()),
            z_min=float(self.z_grid.min()),
            z_max=float(self.z_grid.max()),
            policy=base.support.policy,
        )
        if values is None:
            K1, K2, K3, Z = np.meshgrid(self.k_grid, self.k_grid, self.k_grid, self.z_grid, indexing="ij")
            values = base(K1, K2, K3, Z, **params)
        self.values = np.asarray(values)
        packed = _pack_values(self.values, config.value_transform, config.floor)
        axes = (
            _axis_from_grid(self.k_grid, log_axis=config.log_k),
            _axis_from_grid(self.k_grid, log_axis=config.log_k),
            _axis_from_grid(self.k_grid, log_axis=config.log_k),
            self.z_grid,
        )
        self.interpolator = RegularGridInterpolator(
            axes,
            packed,
            method=config.method,
            bounds_error=False,
            fill_value=None,
        )

    @classmethod
    def from_bispectrum(cls, base: Bispectrum3D, config: Bispectrum3DInterpolationConfig, **params):
        return cls(base, config, **params)

    def evaluate(self, k1, k2, k3, z, **params):
        if params:
            raise ValueError("InterpolatedBispectrum3D does not accept runtime model parameters")
        k1 = np.asarray(k1, dtype=float)
        k2 = np.asarray(k2, dtype=float)
        k3 = np.asarray(k3, dtype=float)
        z = np.asarray(z, dtype=float)
        k1, k2, k3, z = np.broadcast_arrays(k1, k2, k3, z)
        shape = k1.shape
        pts = np.column_stack([
            _coords_from_values(k1.ravel(), log_axis=self.config.log_k),
            _coords_from_values(k2.ravel(), log_axis=self.config.log_k),
            _coords_from_values(k3.ravel(), log_axis=self.config.log_k),
            z.ravel(),
        ])
        out = self.interpolator(pts).reshape(shape)
        out = _unpack_values(out, self.config.value_transform, self.config.floor)
        return out.item() if out.shape == () else out


class InterpolatedBispectrum2D(Bispectrum2D):
    """Interpolated wrapper with the same signature as ``Bispectrum2D``."""

    def __init__(self, base: Bispectrum2D, config: Bispectrum2DInterpolationConfig, values=None, **params):
        self.base = base
        self.config = config
        self.ell_grid = config.ell_grid()
        self.support = Support2D(
            ell_min=float(self.ell_grid.min()),
            ell_max=float(self.ell_grid.max()),
            policy=base.support.policy,
        )
        if values is None:
            E1, E2, E3 = np.meshgrid(self.ell_grid, self.ell_grid, self.ell_grid, indexing="ij")
            values = base(E1, E2, E3, **params)
        self.values = np.asarray(values)
        packed = _pack_values(self.values, config.value_transform, config.floor)
        axes = tuple(_axis_from_grid(self.ell_grid, log_axis=config.log_ell) for _ in range(3))
        self.interpolator = RegularGridInterpolator(
            axes,
            packed,
            method=config.method,
            bounds_error=False,
            fill_value=None,
        )

    @classmethod
    def from_bispectrum(cls, base: Bispectrum2D, config: Bispectrum2DInterpolationConfig, **params):
        return cls(base, config, **params)

    def evaluate(self, ell1, ell2, ell3, **params):
        if params:
            raise ValueError("InterpolatedBispectrum2D does not accept runtime model parameters")
        ell1 = np.asarray(ell1, dtype=float)
        ell2 = np.asarray(ell2, dtype=float)
        ell3 = np.asarray(ell3, dtype=float)
        ell1, ell2, ell3 = np.broadcast_arrays(ell1, ell2, ell3)
        shape = ell1.shape
        pts = np.column_stack([
            _coords_from_values(ell1.ravel(), log_axis=self.config.log_ell),
            _coords_from_values(ell2.ravel(), log_axis=self.config.log_ell),
            _coords_from_values(ell3.ravel(), log_axis=self.config.log_ell),
        ])
        out = self.interpolator(pts).reshape(shape)
        out = _unpack_values(out, self.config.value_transform, self.config.floor)
        return out.item() if out.shape == () else out


class InterpolatedBispectrumMultipole3D(BispectrumMultipole3D):
    """Interpolated wrapper with the same signature as ``BispectrumMultipole3D``."""

    def __init__(self, base: BispectrumMultipole3D, config: BispectrumMultipole3DInterpolationConfig, values=None, **params):
        self.base = base
        self.config = config
        self.modes = np.asarray(config.modes, dtype=int)
        self.basis = base.basis
        self.support = Support3D(
            k_min=float(config.k_grid().min()),
            k_max=float(config.k_grid().max()),
            z_min=float(config.z_grid().min()),
            z_max=float(config.z_grid().max()),
            policy=getattr(getattr(base, "support", None), "policy", "ignore"),
        )
        self.k_grid = config.k_grid()
        self.z_grid = config.z_grid()
        if values is None:
            K1, K2, Z = np.meshgrid(self.k_grid, self.k_grid, self.z_grid, indexing="ij")
            values = np.asarray([base(int(m), K1, K2, Z, **params) for m in self.modes])
        self.values = np.asarray(values)
        packed = _pack_values(self.values, config.value_transform, config.floor)
        axes = (
            self.modes,
            _axis_from_grid(self.k_grid, log_axis=config.log_k),
            _axis_from_grid(self.k_grid, log_axis=config.log_k),
            self.z_grid,
        )
        self.interpolator = RegularGridInterpolator(
            axes,
            packed,
            method=config.method,
            bounds_error=False,
            fill_value=None,
        )

    @classmethod
    def from_multipole(cls, base: BispectrumMultipole3D, config: BispectrumMultipole3DInterpolationConfig, **params):
        return cls(base, config, **params)

    def evaluate(self, mode, k1, k2, z, **params):
        if params:
            raise ValueError("InterpolatedBispectrumMultipole3D does not accept runtime model parameters")
        scalar_mode = np.isscalar(mode)
        modes = np.atleast_1d(np.asarray(mode, dtype=int)).ravel()
        k1 = np.asarray(k1, dtype=float)
        k2 = np.asarray(k2, dtype=float)
        z = np.asarray(z, dtype=float)
        k1, k2, z = np.broadcast_arrays(k1, k2, z)
        shape = k1.shape
        out = []
        for m in modes:
            pts = np.column_stack([
                np.full(k1.size, int(m), dtype=float),
                _coords_from_values(k1.ravel(), log_axis=self.config.log_k),
                _coords_from_values(k2.ravel(), log_axis=self.config.log_k),
                z.ravel(),
            ])
            vals = self.interpolator(pts).reshape(shape)
            out.append(_unpack_values(vals, self.config.value_transform, self.config.floor))
        out = np.asarray(out)
        return out[0] if scalar_mode else out

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
        self.base = None

    @classmethod
    def from_grid(cls, grid: BispectrumMultipole2DGrid, base=None):
        obj = cls(
            modes=grid.modes,
            ell_grid=grid.ell_grid,
            psi_grid=grid.psi_grid,
            values=grid.values,
            basis=grid.basis,
            method=grid.method,
            angle=grid.angle,
        )
        obj.base = base
        return obj

    @classmethod
    def from_multipole(cls, base: BispectrumMultipole2D, config=None, **params):
        if params:
            raise ValueError("Interpolating a BispectrumMultipole2D does not accept runtime parameters unless its evaluator handles them")
        config = config or BispectrumMultipole2DConfig()
        grid_config = MultipoleGridConfig(
            ell_min=config.ell_min,
            ell_max=config.ell_max,
            n_ell=config.n_ell,
            psi_min=config.psi_min,
            psi_max=config.psi_max,
            n_psi=config.n_psi,
            delta_beta_min=config.delta_beta_min,
            delta_beta_max=config.delta_beta_max,
            n_delta_beta_lin=config.n_delta_beta_lin,
            n_delta_beta_log=config.n_delta_beta_log,
            delta_beta_transition=config.delta_beta_transition,
            angle_sampling=config.angle_sampling,
        )
        ell, psi, _, _, _, _ = make_ell_psi_delta_beta_grid(grid_config)
        ELL, PSI = np.meshgrid(ell, psi, indexing="ij")
        E1, E2 = ellpsi_to_ell1ell2(ELL, PSI)
        modes = base.available_modes(config.mode_max)
        values = np.asarray([base(int(m), E1, E2) for m in modes])
        grid = BispectrumMultipole2DGrid(
            modes=np.asarray(modes, dtype=int),
            ell_grid=ell,
            psi_grid=psi,
            values=values,
            basis=base.basis,
            method=config.interpolation,
            angle="delta_beta",
        )
        return cls.from_grid(grid, base=base)

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

class NotYetInterpolatedBispectrum(Bispectrum2D):
    """Explicit placeholder for interpolation paths not implemented yet."""

    def __init__(self, message: str):
        self.message = message

    def evaluate(self, *args, **kwargs):
        raise NotImplementedError(self.message)


def sides_to_ruv(ell1, ell2, ell3):
    """TreeCorr-like unsigned triangle coordinates.

    The sides are sorted so that ``d1 >= d2 >= d3``.  We define ``r=d2``,
    ``u=d3/d2``, and ``v=(d1-d2)/d3``.  This removes permutation redundancy for
    symmetric bispectra.
    """
    arr = np.sort(np.stack([ell1, ell2, ell3], axis=0), axis=0)
    d3, d2, d1 = arr[0], arr[1], arr[2]
    r = d2
    u = np.divide(d3, d2, out=np.zeros_like(d2, dtype=float), where=d2 != 0)
    v = np.divide(d1 - d2, d3, out=np.zeros_like(d3, dtype=float), where=d3 != 0)
    return r, u, v


class RuvInterpolatedBispectrum2D(Bispectrum2D):
    def __init__(self, base: Bispectrum2D, r_grid, u_grid, v_grid, log_values,
                 method="linear", window=None):
        self.base = base
        self.r_grid = np.asarray(r_grid)
        self.u_grid = np.asarray(u_grid)
        self.v_grid = np.asarray(v_grid)
        self.log_values = np.asarray(log_values)
        self.interpolator = RegularGridInterpolator(
            (np.log(self.r_grid), np.log(self.u_grid), self.v_grid),
            self.log_values,
            method=method,
            bounds_error=False,
            fill_value=None,
        )
        self.support = base.support
        self.window = window

    @classmethod
    def from_bispectrum(cls, base: Bispectrum2D, r_grid, u_grid, v_grid, method="linear", floor=1.0e-300, **params):
        R, U, V = np.meshgrid(r_grid, u_grid, v_grid, indexing="ij")
        # Inverse of the convention above: d2=r, d3=u*r, d1=r+v*d3.
        d2 = R
        d3 = U * R
        d1 = R + V * d3
        vals = np.maximum(base(d1, d2, d3, **params), floor)
        return cls(base, r_grid, u_grid, v_grid, np.log(vals), method=method)

    def evaluate(self, ell1, ell2, ell3, **params):
        if params:
            raise ValueError("RuvInterpolatedBispectrum2D does not accept runtime model parameters")
        r, u, v = sides_to_ruv(ell1, ell2, ell3)
        pts = (np.log(r), np.log(u), v)
        val = np.exp(self.interpolator(pts))
        if self.window is not None:
            val = val * self.window(ell1, ell2, ell3)
        return val


# Backward-compatible alias.  New code should use RuvInterpolatedBispectrum2D.
RuvInterpolatedAngularBispectrum2D = RuvInterpolatedBispectrum2D
