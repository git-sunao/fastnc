"""Unified manager for numeric, Slepian, and semi-analytic 3PCF routes."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from fastnc.bispectrum import (
    Bispectrum2D,
    NumericRepresentation2D,
    SlepianRepresentation2D,
)
from fastnc.coupling import CouplingCacheSession, CouplingMatrix
from fastnc.coupling.cache import resolve_coupling_cache_file
from fastnc.hankel import double_hankel_transform, make_fftlog_grid
from fastnc.multipole import BispectrumMultipole

from .config import ThreePCFConfig
from .conventions import SpinSpec, as_effective_spin_triple
from .conventions.projection import _projection_name
from .numeric import contract_hkernel
from .slepian import SlepianCalculator
from .tables import (
    ComponentModeKey,
    HKernelKey,
    HKernelTable,
    ZetaKKey,
    ZetaKTable,
    ZetaTable,
)


def _coordinate(values, name: str, *, positive: bool) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size < 1:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if np.any(~np.isfinite(array)):
        raise ValueError(f"{name} must contain finite values")
    if positive and np.any(array <= 0.0):
        raise ValueError(f"{name} must contain positive values")
    if np.any(np.diff(array) <= 0.0):
        raise ValueError(f"{name} must be strictly increasing")
    if name == "theta" and array.size > 1:
        spacing = np.diff(np.log(array))
        if not np.allclose(spacing, spacing[0]):
            raise ValueError("theta must be evenly spaced in log(theta)")
    result = np.array(array, copy=True)
    result.setflags(write=False)
    return result


class ThreePCF:
    """A 2D bispectrum prediction on fixed theta and phi coordinates."""

    _supported_routes = frozenset(
        {"numeric", "slepian", "semi_analytic", "hybrid"}
    )

    def __init__(
        self,
        config: ThreePCFConfig,
        bispectrum,
        theta,
        phi,
        *,
        route: str = "numeric",
        coupling_factory: Callable[[tuple[int, int, int], str], object] | None = None,
    ):
        if not isinstance(config, ThreePCFConfig):
            raise TypeError("config must be a ThreePCFConfig")
        if not callable(bispectrum):
            raise TypeError("bispectrum must be a callable 2D bispectrum")
        self.config = config
        self._route = self._validate_route(route)
        self._bispectrum = bispectrum
        self._bispectrum_state_token = self._source_state_token()
        self._theta = _coordinate(theta, "theta", positive=True)
        self._phi = _coordinate(phi, "phi", positive=False)
        self.grid = self._make_grid()

        self._coupling_factory = coupling_factory
        self._coupling_cache_sessions: dict[Path, CouplingCacheSession] = {}
        self._couplings: dict[tuple[int, int, int], object] = {}
        self._multipole: BispectrumMultipole | None = None
        self._numeric_semianalytic_source: Bispectrum2D | None = None
        self._slepian_source: Bispectrum2D | None = None
        self._hybrid_plan_ready = False
        self._slepian_calculator: SlepianCalculator | None = None
        self._hkernel_tables: dict[
            tuple[tuple[int, int, int], ...], HKernelTable
        ] = {}
        self._zetak_tables: dict[
            tuple[tuple[int, int, int], ...], ZetaKTable
        ] = {}
        self._zeta_tables: dict[
            tuple[tuple[int, int, int], ...], ZetaTable
        ] = {}

    @property
    def bispectrum(self):
        return self._bispectrum

    @property
    def route(self) -> str:
        return self._route

    @property
    def theta(self) -> np.ndarray:
        return self._theta

    @property
    def phi(self) -> np.ndarray:
        return self._phi

    def _make_grid(self):
        return make_fftlog_grid(
            self.config.ell_min,
            self.config.ell_max,
            self.config.n_ell,
            theta=self._theta,
        )

    def _clear_grid_results(self) -> None:
        self._hkernel_tables.clear()
        self._zetak_tables.clear()
        self._zeta_tables.clear()

    @classmethod
    def _validate_route(cls, route: str) -> str:
        route = str(route)
        if route not in cls._supported_routes:
            supported = ", ".join(sorted(cls._supported_routes))
            raise ValueError(f"route must be one of: {supported}")
        return route

    def _clear_route_results(self) -> None:
        self._multipole = None
        self._numeric_semianalytic_source = None
        self._slepian_source = None
        self._hybrid_plan_ready = False
        self._slepian_calculator = None
        self._clear_grid_results()

    def _clear_source_results(self) -> None:
        self._multipole = None
        self._numeric_semianalytic_source = None
        self._slepian_source = None
        self._hybrid_plan_ready = False
        if self._slepian_calculator is not None:
            self._slepian_calculator._clear_source_cache()
        self._clear_grid_results()

    def _source_state_token(self):
        token = getattr(self._bispectrum, "state_token", None)
        return None if token is None else tuple(token)

    def _sync_source_state(self) -> None:
        token = self._source_state_token()
        if token == self._bispectrum_state_token:
            return
        self._bispectrum_state_token = token
        self._clear_source_results()

    def set_theta(self, theta) -> None:
        """Replace target theta bins and rebuild the tuned FFT grid."""
        updated = _coordinate(theta, "theta", positive=True)
        if np.array_equal(updated, self._theta):
            return
        self._theta = updated
        self.grid = self._make_grid()
        self._slepian_calculator = None
        self._clear_grid_results()

    def set_phi(self, phi) -> None:
        """Replace opening-angle bins without invalidating radial results."""
        updated = _coordinate(phi, "phi", positive=False)
        if np.array_equal(updated, self._phi):
            return
        self._phi = updated
        self._zeta_tables.clear()

    def set_bispectrum(self, bispectrum) -> None:
        """Replace the angular bispectrum while preserving grid resources."""
        if not callable(bispectrum):
            raise TypeError("bispectrum must be a callable 2D bispectrum")
        if bispectrum is self._bispectrum:
            return
        self._bispectrum = bispectrum
        self._bispectrum_state_token = self._source_state_token()
        self._clear_source_results()

    def set_route(self, route: str) -> None:
        """Replace the route and invalidate all route-dependent results."""
        updated = self._validate_route(route)
        if updated == self._route:
            return
        self._route = updated
        self._clear_route_results()

    def _cache_session(self) -> CouplingCacheSession | None:
        if not self.config.use_coupling_cache:
            return None
        path = resolve_coupling_cache_file(self.config.coupling_cache_file)
        if path not in self._coupling_cache_sessions:
            self._coupling_cache_sessions[path] = CouplingCacheSession(path)
        return self._coupling_cache_sessions[path]

    def coupling(self, sigma: tuple[int, int, int]):
        """Return the retained coupling matrix for one effective spin."""
        sigma = tuple(int(value) for value in sigma)
        if len(sigma) != 3:
            raise ValueError("sigma must contain exactly three entries")
        if sigma not in self._couplings:
            if self._coupling_factory is not None:
                coupling = self._coupling_factory(sigma, self.config.basis)
            else:
                kwargs = self.config.coupling_kwargs()
                session = self._cache_session()
                if session is not None:
                    kwargs["cache_session"] = session
                coupling = CouplingMatrix(
                    *sigma,
                    basis=self.config.basis,
                    **kwargs,
                )
            if not callable(coupling):
                raise TypeError("coupling_factory must return a callable object")
            self._couplings[sigma] = coupling
        return self._couplings[sigma]

    @staticmethod
    def _term_supports(term, representation_type) -> bool:
        return any(
            isinstance(representation, representation_type)
            for representation in term.representations
        )

    def _plan_hybrid_sources(
        self,
    ) -> tuple[Bispectrum2D | None, Bispectrum2D | None]:
        """Partition terms once into direct-ZetaK and HKernel pipelines."""
        if not isinstance(self._bispectrum, Bispectrum2D):
            raise TypeError("the hybrid route requires a Bispectrum2D")
        if self._hybrid_plan_ready:
            return self._slepian_source, self._numeric_semianalytic_source

        slepian_names = []
        numeric_names = []
        unsupported = []
        for term in self._bispectrum.terms:
            if self._term_supports(term, SlepianRepresentation2D):
                slepian_names.append(term.name)
            elif self._term_supports(term, NumericRepresentation2D):
                numeric_names.append(term.name)
            else:
                unsupported.append(term.name)
        if unsupported:
            raise TypeError(
                "hybrid route found terms with no supported representation: "
                f"{unsupported}"
            )

        self._slepian_source = (
            self._bispectrum.select_terms(*slepian_names)
            if slepian_names
            else None
        )
        self._numeric_semianalytic_source = (
            self._bispectrum.select_terms(*numeric_names)
            if numeric_names
            else None
        )
        self._hybrid_plan_ready = True
        return self._slepian_source, self._numeric_semianalytic_source

    def _numeric_pipeline_source(self):
        if self.route != "hybrid":
            return self._bispectrum
        _, numeric_source = self._plan_hybrid_sources()
        if numeric_source is None:
            raise ValueError("hybrid route has no numeric/semi-analytic terms")
        return numeric_source

    def multipoles(self) -> BispectrumMultipole:
        """Return multipoles for terms assigned to the HKernel pipeline."""
        self._sync_source_state()
        if self._multipole is None:
            self._multipole = BispectrumMultipole.from_numeric(
                self.config.multipole,
                self._numeric_pipeline_source(),
                basis=self.config.basis,
            )
        return self._multipole

    def _epsilons(
        self,
        epsilons: Iterable[tuple[int, int, int]] | None,
    ) -> tuple[tuple[int, int, int], ...]:
        spin_spec = SpinSpec(self.config.spin)
        if epsilons is None:
            return spin_spec.representative_epsilons()
        requested = tuple(epsilons)
        if not requested:
            raise ValueError("epsilons must not be empty")
        representatives = set(spin_spec.representative_epsilons())
        result = []
        seen = set()
        for epsilon in requested:
            epsilon = tuple(int(value) for value in epsilon)
            spin_spec.sigma_from_epsilon(epsilon)
            invalid_scalar_indices = tuple(
                index
                for index, (spin, sign) in enumerate(
                    zip(spin_spec.spin, epsilon)
                )
                if spin == 0 and sign == -1
            )
            if invalid_scalar_indices:
                raise ValueError(
                    f"epsilon={epsilon} uses -1 at spin-zero vertices "
                    f"{invalid_scalar_indices}; epsilon must be +1 where "
                    "spin is zero"
                )
            if epsilon not in representatives:
                representative, conjugated = spin_spec.canonicalize_epsilon(
                    epsilon
                )
                relation = (
                    "its complex-conjugate representative is"
                    if conjugated
                    else "its representative is"
                )
                raise ValueError(
                    f"epsilon={epsilon} is not an independent "
                    f"representative for spin={spin_spec.spin}; {relation} "
                    f"{representative}"
                )
            if epsilon not in seen:
                result.append(epsilon)
                seen.add(epsilon)
        return tuple(result)

    @staticmethod
    def _hkey(sigma: tuple[int, int, int], k: float) -> HKernelKey:
        effective = as_effective_spin_triple(sigma)
        two_nu = int(round(2.0 * float(effective.nu(k))))
        return HKernelKey(sigma1=effective.sigma1, two_nu=two_nu)

    def _multipole_modes(self) -> np.ndarray:
        if self.config.basis == "fourier":
            return np.arange(-self.config.Lmax, self.config.Lmax + 1)
        if self.config.basis in {"cosine", "legendre"}:
            return np.arange(self.config.Lmax + 1)
        if self.config.basis == "sine":
            return np.arange(1, self.config.Lmax + 1)
        raise ValueError(f"unsupported basis: {self.config.basis!r}")

    def hkernel(
        self,
        *,
        epsilons: Iterable[tuple[int, int, int]] | None = None,
    ) -> HKernelTable:
        """Return numeric H kernels on the retained full FFT grid."""
        self._sync_source_state()
        requested_epsilons = self._epsilons(epsilons)
        if requested_epsilons in self._hkernel_tables:
            return self._hkernel_tables[requested_epsilons]

        spin_spec = SpinSpec(self.config.spin)
        ell2, ell3 = np.meshgrid(self.grid.ell, self.grid.ell, indexing="ij")
        psi = np.arctan2(ell3, ell2)
        psi_unique, inverse = np.unique(psi, return_inverse=True)
        modes = self._multipole_modes()
        coefficients = self.multipoles().evaluate(modes, ell2, ell3)

        values: dict[HKernelKey, np.ndarray] = {}
        aliases: dict[ComponentModeKey, HKernelKey] = {}
        for epsilon in requested_epsilons:
            sigma = spin_spec.sigma_from_epsilon(epsilon)
            effective = as_effective_spin_triple(sigma)
            coupling = self.coupling(sigma)
            for k in effective.k_values(self.config.kmax):
                key = self._hkey(sigma, float(k))
                aliases[ComponentModeKey.from_epsilon_k(epsilon, k)] = key
                if key in values:
                    continue
                coupling_values = np.stack(
                    [
                        np.asarray(
                            coupling(int(mode), float(k), psi_unique)
                        )[inverse].reshape(psi.shape)
                        for mode in modes
                    ]
                )
                values[key] = contract_hkernel(
                    coefficients, coupling_values
                )

        keys = tuple(values)
        if not keys:
            raise ValueError("spin and kmax produced no allowed k modes")
        table = HKernelTable(
            self.grid,
            keys,
            np.stack([values[key] for key in keys]),
            aliases=aliases,
        )
        self._hkernel_tables[requested_epsilons] = table
        return table

    def zetak(
        self,
        *,
        epsilons: Iterable[tuple[int, int, int]] | None = None,
    ) -> ZetaKTable:
        """Return cached or term-wise assembled opening-angle modes."""
        self._sync_source_state()
        if self.route == "semi_analytic":
            raise NotImplementedError(
                f"route={self.route!r} is not implemented"
            )
        requested_epsilons = self._epsilons(epsilons)
        if requested_epsilons in self._zetak_tables:
            return self._zetak_tables[requested_epsilons]

        if self.route == "slepian":
            table = self._zetak_slepian(
                requested_epsilons, self._bispectrum
            )
        elif self.route == "hybrid":
            slepian_source, numeric_source = self._plan_hybrid_sources()
            contributions = []
            if slepian_source is not None:
                contributions.append(
                    self._zetak_slepian(requested_epsilons, slepian_source)
                )
            if numeric_source is not None:
                contributions.append(
                    self._zetak_numeric_semianalytic(requested_epsilons)
                )
            table = self._sum_zetak_tables(contributions)
        else:
            table = self._zetak_numeric_semianalytic(requested_epsilons)

        self._zetak_tables[requested_epsilons] = table
        return table

    @staticmethod
    def _sum_zetak_tables(tables) -> ZetaKTable:
        tables = tuple(tables)
        if not tables:
            raise ValueError("at least one ZetaK contribution is required")
        reference = tables[0]
        aliases = dict(reference.aliases)
        keys = []
        for table in tables:
            if not np.array_equal(table.theta, reference.theta):
                raise ValueError("ZetaK contributions use different theta grids")
            if dict(table.aliases) != aliases:
                raise ValueError("ZetaK contributions use different mode aliases")
            for key in table.keys:
                if key not in keys:
                    keys.append(key)
        values = []
        for key in keys:
            total = np.zeros(
                (reference.theta.size, reference.theta.size), dtype=complex
            )
            for table in tables:
                if key in table.keys:
                    total += table.get(key)
            values.append(total)
        return ZetaKTable(
            reference.theta,
            tuple(keys),
            np.stack(values),
            aliases=aliases,
        )

    def _zetak_numeric_semianalytic(
        self,
        requested_epsilons: tuple[tuple[int, int, int], ...],
    ) -> ZetaKTable:
        """Transform the shared BispectrumMultipole/HKernel pipeline."""

        spin_spec = SpinSpec(self.config.spin)
        htable = self.hkernel(epsilons=requested_epsilons)
        ell2, ell3 = np.meshgrid(self.grid.ell, self.grid.ell, indexing="ij")
        hankel = replace(self.config.hankel, xy=self.grid.xy)

        values: dict[ZetaKKey, np.ndarray] = {}
        aliases: dict[ComponentModeKey, ZetaKKey] = {}
        for epsilon in requested_epsilons:
            sigma = spin_spec.sigma_from_epsilon(epsilon)
            effective = as_effective_spin_triple(sigma)
            for k in effective.k_values(self.config.kmax):
                hkey = self._hkey(sigma, float(k))
                m, n = effective.bessel_orders(float(k))
                key = ZetaKKey(hkey, m, n, effective.Sigma)
                aliases[ComponentModeKey.from_epsilon_k(epsilon, k)] = key
                if key in values:
                    continue
                prefactor = (-1j) ** effective.Sigma / (2.0 * np.pi) ** 3
                integrand = prefactor * htable.get(hkey) * ell2**2 * ell3**2
                theta1, theta2, full = double_hankel_transform(
                    self.grid.ell,
                    self.grid.ell,
                    integrand,
                    m,
                    n,
                    config=hankel,
                    bin_width_logtheta=self.config.bin_width_logtheta,
                )
                if not np.allclose(theta1, self.grid.theta) or not np.allclose(
                    theta2, self.grid.theta
                ):
                    raise RuntimeError(
                        "double Hankel output does not match the tuned theta grid"
                    )
                values[key] = self.grid.downsample_2d(full)

        keys = tuple(values)
        table = ZetaKTable(
            self.theta,
            keys,
            np.stack([values[key] for key in keys]),
            aliases=aliases,
        )
        return table

    def _zetak_slepian(
        self,
        requested_epsilons: tuple[tuple[int, int, int], ...],
        bispectrum,
    ) -> ZetaKTable:
        if self._slepian_calculator is None:
            self._slepian_calculator = SlepianCalculator(self.config.slepian)

        values: dict[ZetaKKey, np.ndarray] = {}
        aliases: dict[ComponentModeKey, ZetaKKey] = {}
        spin_spec = SpinSpec(self.config.spin)
        for epsilon in requested_epsilons:
            sigma = spin_spec.sigma_from_epsilon(epsilon)
            effective = as_effective_spin_triple(sigma)
            k_values = effective.k_values(self.config.kmax)
            mode_values = self._slepian_calculator.evaluate_modes(
                bispectrum,
                self.grid.ell,
                self.theta,
                k_values,
                sigma=sigma,
            )
            for k in k_values:
                hkey = self._hkey(sigma, float(k))
                m, n = effective.bessel_orders(float(k))
                key = ZetaKKey(hkey, m, n, effective.Sigma)
                aliases[ComponentModeKey.from_epsilon_k(epsilon, k)] = key
                if key not in values:
                    values[key] = mode_values[float(k)]

        keys = tuple(values)
        table = ZetaKTable(
            self.theta,
            keys,
            np.stack([values[key] for key in keys]),
            aliases=aliases,
        )
        return table

    def zeta(
        self,
        *,
        epsilons: Iterable[tuple[int, int, int]] | None = None,
        projection: str = "x",
    ) -> ZetaTable:
        """Return the final 3PCF in the requested projection convention."""
        self._sync_source_state()
        requested_epsilons = self._epsilons(epsilons)
        target_projection = _projection_name(projection)
        if requested_epsilons in self._zeta_tables:
            return self._zeta_tables[requested_epsilons].to_projection(
                target_projection
            )

        spin_spec = SpinSpec(self.config.spin)
        zetak = self.zetak(epsilons=requested_epsilons)
        values = []
        sigmas = []
        for epsilon in requested_epsilons:
            sigma = spin_spec.sigma_from_epsilon(epsilon)
            effective = as_effective_spin_triple(sigma)
            k_values = effective.k_values(self.config.kmax)
            modes = np.stack(
                [zetak.get_for_mode(epsilon, float(k)) for k in k_values]
            )
            phase_labels = effective.nu(k_values)
            phase = np.exp(1j * phase_labels[:, None] * self.phi[None, :])
            values.append(np.tensordot(modes, phase, axes=(0, 0)))
            sigmas.append(sigma)

        x_table = ZetaTable(
            theta=self.theta,
            phi=self.phi,
            components=requested_epsilons,
            values=np.stack(values),
            sigmas=tuple(sigmas),
            projection="x",
        )
        self._zeta_tables[requested_epsilons] = x_table
        return x_table.to_projection(target_projection)
