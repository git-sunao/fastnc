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
    """Grid configuration for interpolating ``B(ell1,ell2,ell3)``.

    ``coordinate`` selects the tabulation coordinates.

    - ``"ell123"``: tabulate on a direct-product ``(ell1, ell2, ell3)`` grid.
    - ``"ruv"``: tabulate on TreeCorr-like triangle coordinates
      ``(r, u, v)`` after sorting the three sides.  This coordinate removes
      side-permutation information and should therefore only be used for
      side-symmetric bispectra.

    The value transform is shared by all coordinate systems.
    """

    # Direct side-length interpolation, coordinate="ell123".
    ell_min: float | None = None
    ell_max: float | None = None
    n_ell: int | None = None

    method: str = "linear"
    log_ell: bool = True
    value_transform: str = "identity"  # identity, log, signed-log
    floor: float = 1.0e-300
    coordinate: str = "ell123"  # ell123, ruv

    # TreeCorr-like triangle interpolation, coordinate="ruv".
    r_min: float | None = None
    r_max: float | None = None
    n_r: int | None = None
    u_min: float = 1.0e-3
    u_max: float = 1.0
    n_u: int = 64
    v_min: float = 0.0
    v_max: float = 1.0
    n_v: int = 64
    log_r: bool = True
    log_u: bool = True
    assume_symmetric: bool = False

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
        coordinate: str = "ell123",
        n_r: int | None = None,
        u_min: float = 1.0e-3,
        u_max: float = 1.0,
        n_u: int = 64,
        v_min: float = 0.0,
        v_max: float = 1.0,
        n_v: int = 64,
        log_r: bool = True,
        log_u: bool = True,
        assume_symmetric: bool = False,
    ) -> "Bispectrum2DInterpolationConfig":
        if not np.isfinite(support.ell_min) or not np.isfinite(support.ell_max):
            raise ValueError("finite support.ell_min and support.ell_max are required")
        ell_min = max(float(support.ell_min), np.finfo(float).tiny)
        ell_max = float(support.ell_max)
        coordinate = _normalize_2d_coordinate(coordinate)
        if coordinate == "ell123":
            return cls(
                ell_min=ell_min,
                ell_max=ell_max,
                n_ell=int(n_ell),
                method=method,
                log_ell=log_ell,
                value_transform=value_transform,
                floor=floor,
                coordinate="ell123",
            )
        return cls(
            method=method,
            value_transform=value_transform,
            floor=floor,
            coordinate="ruv",
            r_min=ell_min,
            r_max=ell_max,
            n_r=int(n_r if n_r is not None else n_ell),
            u_min=u_min,
            u_max=u_max,
            n_u=int(n_u),
            v_min=v_min,
            v_max=v_max,
            n_v=int(n_v),
            log_r=log_r,
            log_u=log_u,
            assume_symmetric=assume_symmetric,
        )

    @classmethod
    def from_ruv_grids(
        cls,
        *,
        r_grid,
        u_grid,
        v_grid,
        method: str = "linear",
        value_transform: str = "identity",
        floor: float = 1.0e-300,
        log_r: bool = True,
        log_u: bool = True,
        assume_symmetric: bool = True,
    ) -> "Bispectrum2DInterpolationConfig":
        r_grid = np.asarray(r_grid, dtype=float)
        u_grid = np.asarray(u_grid, dtype=float)
        v_grid = np.asarray(v_grid, dtype=float)
        return cls(
            method=method,
            value_transform=value_transform,
            floor=floor,
            coordinate="ruv",
            r_min=float(r_grid.min()),
            r_max=float(r_grid.max()),
            n_r=int(r_grid.size),
            u_min=float(u_grid.min()),
            u_max=float(u_grid.max()),
            n_u=int(u_grid.size),
            v_min=float(v_grid.min()),
            v_max=float(v_grid.max()),
            n_v=int(v_grid.size),
            log_r=log_r,
            log_u=log_u,
            assume_symmetric=assume_symmetric,
        )

    def ell_grid(self):
        if self.ell_min is None or self.ell_max is None or self.n_ell is None:
            raise ValueError("ell_min, ell_max, and n_ell are required for coordinate='ell123'")
        if self.log_ell:
            return np.geomspace(self.ell_min, self.ell_max, self.n_ell)
        return np.linspace(self.ell_min, self.ell_max, self.n_ell)

    def r_grid(self):
        if self.r_min is None or self.r_max is None or self.n_r is None:
            raise ValueError("r_min, r_max, and n_r are required for coordinate='ruv'")
        if self.log_r:
            return np.geomspace(self.r_min, self.r_max, self.n_r)
        return np.linspace(self.r_min, self.r_max, self.n_r)

    def u_grid(self):
        if self.log_u:
            return np.geomspace(self.u_min, self.u_max, self.n_u)
        return np.linspace(self.u_min, self.u_max, self.n_u)

    def v_grid(self):
        return np.linspace(self.v_min, self.v_max, self.n_v)


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



def _normalize_2d_coordinate(coordinate: str) -> str:
    coordinate = str(coordinate).lower().replace("-", "_")
    aliases = {
        "ell123": "ell123",
        "ell_123": "ell123",
        "sides": "ell123",
        "side_lengths": "ell123",
        "ruv": "ruv",
    }
    try:
        return aliases[coordinate]
    except KeyError as exc:
        raise ValueError("coordinate must be 'ell123' or 'ruv'") from exc


def ruv_to_sides(r, u, v):
    """Convert TreeCorr-like ``(r,u,v)`` coordinates to sorted sides.

    The returned sides are ``(d1, d2, d3)`` with ``d1 >= d2 >= d3``.
    """
    r = np.asarray(r, dtype=float)
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    d2 = r
    d3 = u * r
    d1 = r + v * d3
    return d1, d2, d3

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
    """Interpolated wrapper with the same signature as ``Bispectrum2D``.

    The tabulation coordinate is selected by
    ``Bispectrum2DInterpolationConfig.coordinate``.  Supported values are
    ``"ell123"`` and ``"ruv"``.
    """

    def __init__(self, base: Bispectrum2D, config: Bispectrum2DInterpolationConfig, values=None, **params):
        self.base = base
        self.config = config
        self.coordinate = _normalize_2d_coordinate(config.coordinate)
        self.support = base.support

        if self.coordinate == "ell123":
            axes, table_values = self._build_ell123_table(base, config, values, **params)
        elif self.coordinate == "ruv":
            axes, table_values = self._build_ruv_table(base, config, values, **params)
        else:  # pragma: no cover; guarded by _normalize_2d_coordinate
            raise ValueError("coordinate must be 'ell123' or 'ruv'")

        self.values = np.asarray(table_values)
        packed = _pack_values(self.values, config.value_transform, config.floor)
        self.interpolator = RegularGridInterpolator(
            axes,
            packed,
            method=config.method,
            bounds_error=False,
            fill_value=None,
        )

    def _build_ell123_table(self, base, config, values, **params):
        self.ell_grid = config.ell_grid()
        self.r_grid = None
        self.u_grid = None
        self.v_grid = None
        self.support = Support2D(
            ell_min=float(self.ell_grid.min()),
            ell_max=float(self.ell_grid.max()),
            policy=base.support.policy,
        )
        if values is None:
            E1, E2, E3 = np.meshgrid(
                self.ell_grid, self.ell_grid, self.ell_grid, indexing="ij"
            )
            values = base(E1, E2, E3, **params)
        axes = tuple(_axis_from_grid(self.ell_grid, log_axis=config.log_ell) for _ in range(3))
        return axes, values

    def _build_ruv_table(self, base, config, values, **params):
        if not config.assume_symmetric:
            raise ValueError(
                "coordinate='ruv' sorts the side lengths and is only valid for "
                "side-symmetric bispectra.  Set assume_symmetric=True explicitly."
            )
        self.ell_grid = None
        self.r_grid = config.r_grid()
        self.u_grid = config.u_grid()
        self.v_grid = config.v_grid()
        ell_min = float(min(self.r_grid.min(), (self.u_grid * self.r_grid.min()).min()))
        ell_max = float((self.r_grid.max() * (1.0 + self.u_grid.max() * self.v_grid.max())))
        self.support = Support2D(
            ell_min=ell_min,
            ell_max=ell_max,
            policy=base.support.policy,
        )
        if values is None:
            R, U, V = np.meshgrid(self.r_grid, self.u_grid, self.v_grid, indexing="ij")
            d1, d2, d3 = ruv_to_sides(R, U, V)
            values = base(d1, d2, d3, **params)
        axes = (
            _axis_from_grid(self.r_grid, log_axis=config.log_r),
            _axis_from_grid(self.u_grid, log_axis=config.log_u),
            self.v_grid,
        )
        return axes, values

    @classmethod
    def from_bispectrum(cls, base: Bispectrum2D, config: Bispectrum2DInterpolationConfig, **params):
        return cls(base, config, **params)

    def _query_points_ell123(self, ell1, ell2, ell3):
        return np.column_stack([
            _coords_from_values(ell1.ravel(), log_axis=self.config.log_ell),
            _coords_from_values(ell2.ravel(), log_axis=self.config.log_ell),
            _coords_from_values(ell3.ravel(), log_axis=self.config.log_ell),
        ])

    def _query_points_ruv(self, ell1, ell2, ell3):
        r, u, v = sides_to_ruv(ell1, ell2, ell3)
        return np.column_stack([
            _coords_from_values(r.ravel(), log_axis=self.config.log_r),
            _coords_from_values(u.ravel(), log_axis=self.config.log_u),
            v.ravel(),
        ])

    def evaluate(self, ell1, ell2, ell3, **params):
        if params:
            raise ValueError("InterpolatedBispectrum2D does not accept runtime model parameters")
        ell1 = np.asarray(ell1, dtype=float)
        ell2 = np.asarray(ell2, dtype=float)
        ell3 = np.asarray(ell3, dtype=float)
        ell1, ell2, ell3 = np.broadcast_arrays(ell1, ell2, ell3)
        shape = ell1.shape
        if self.coordinate == "ell123":
            pts = self._query_points_ell123(ell1, ell2, ell3)
        elif self.coordinate == "ruv":
            pts = self._query_points_ruv(ell1, ell2, ell3)
        else:  # pragma: no cover; guarded by construction
            raise ValueError("coordinate must be 'ell123' or 'ruv'")
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
        E2, E3 = ellpsi_to_ell1ell2(ELL, PSI)
        modes = base.available_modes(config.mode_max)
        values = np.asarray([base(int(m), E2, E3) for m in modes])
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

    def evaluate(self, mode, ell2, ell3):
        scalar_mode = np.isscalar(mode)
        modes = np.atleast_1d(np.asarray(mode, dtype=int))

        ell2 = np.asarray(ell2, dtype=float)
        ell3 = np.asarray(ell3, dtype=float)
        ell2, ell3 = np.broadcast_arrays(ell2, ell3)
        shape = ell2.shape

        ell = np.sqrt(ell2**2 + ell3**2)
        psi = np.arctan2(ell3, ell2)

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
