"""Unified manager for numeric, Slepian, and semi-analytic 3PCF routes."""
from __future__ import annotations

from dataclasses import replace
import logging
import time
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from fastnc.bispectrum import (
    Bispectrum2D,
    NumericRepresentation2D,
    SemiAnalyticRepresentation2D,
    SlepianRepresentation2D,
)
from fastnc.coupling import CouplingCacheSession, CouplingMatrix
from fastnc.coupling.cache import resolve_coupling_cache_file
from fastnc.hankel import double_hankel_transform, make_fftlog_grid
from fastnc.multipole import BispectrumMultipole
from fastnc._logging import log

from .config import ThreePCFConfig
from .conventions import SpinSpec, as_effective_spin_triple
from .conventions.projection import _projection_name
from .numeric import contract_hkernel
from .slepian import SlepianCalculator
from .semi_analytic import SemiAnalyticCalculator
from .tables import (
    ComponentModeKey,
    HKernelKey,
    HKernelTable,
    ZetaKKey,
    ZetaKTable,
    ZetaTable,
)


logger = logging.getLogger(__name__)


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
        self._numeric_source: Bispectrum2D | None = None
        self._semi_analytic_source: Bispectrum2D | None = None
        self._slepian_source: Bispectrum2D | None = None
        self._hybrid_plan_ready = False
        self._slepian_calculator: SlepianCalculator | None = None
        self._semi_analytic_calculator: SemiAnalyticCalculator | None = None
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
        self._numeric_source = None
        self._semi_analytic_source = None
        self._slepian_source = None
        self._hybrid_plan_ready = False
        self._slepian_calculator = None
        self._semi_analytic_calculator = None
        self._clear_grid_results()

    def _clear_source_results(self) -> None:
        self._multipole = None
        self._numeric_source = None
        self._semi_analytic_source = None
        self._slepian_source = None
        self._hybrid_plan_ready = False
        if self._slepian_calculator is not None:
            self._slepian_calculator._clear_source_cache()
        if self._semi_analytic_calculator is not None:
            self._semi_analytic_calculator.clear_source_cache()
        self._clear_grid_results()

    def _source_state_token(self):
        token = getattr(self._bispectrum, "state_token", None)
        return None if token is None else tuple(token)

    def _sync_source_state(self) -> None:
        token = self._source_state_token()
        if token == self._bispectrum_state_token:
            return
        log(logger, logging.INFO, "bispectrum state changed; clearing source-dependent results")
        self._bispectrum_state_token = token
        self._clear_source_results()

    def set_theta(self, theta) -> None:
        """Replace target theta bins and rebuild the tuned FFT grid."""
        updated = _coordinate(theta, "theta", positive=True)
        if np.array_equal(updated, self._theta):
            return
        log(logger, logging.INFO, "theta grid changed; rebuilding radial grid and results")
        self._theta = updated
        self.grid = self._make_grid()
        self._slepian_calculator = None
        if self._semi_analytic_calculator is not None:
            self._semi_analytic_calculator.clear_grid_cache()
        self._clear_grid_results()

    def set_phi(self, phi) -> None:
        """Replace opening-angle bins without invalidating radial results."""
        updated = _coordinate(phi, "phi", positive=False)
        if np.array_equal(updated, self._phi):
            return
        log(logger, logging.INFO, "phi grid changed; clearing resummed Zeta results")
        self._phi = updated
        self._zeta_tables.clear()

    def set_bispectrum(self, bispectrum) -> None:
        """Replace the angular bispectrum while preserving grid resources."""
        if not callable(bispectrum):
            raise TypeError("bispectrum must be a callable 2D bispectrum")
        if bispectrum is self._bispectrum:
            return
        log(logger, logging.INFO, "bispectrum replaced; clearing source-dependent results")
        self._bispectrum = bispectrum
        self._bispectrum_state_token = self._source_state_token()
        self._clear_source_results()

    def set_route(self, route: str) -> None:
        """Replace the route and invalidate all route-dependent results."""
        updated = self._validate_route(route)
        if updated == self._route:
            return
        log(logger, logging.INFO, "route changed from %s to %s", self._route, updated)
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
            log(logger, logging.DEBUG, "constructing coupling matrix sigma=%s basis=%s", sigma, self.config.basis)
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
        else:
            log(logger, logging.DEBUG, "coupling cache hit sigma=%s", sigma)
        return self._couplings[sigma]

    @staticmethod
    def _term_supports(term, representation_type) -> bool:
        return any(
            isinstance(representation, representation_type)
            for representation in term.representations
        )

    def _plan_hybrid_sources(
        self,
    ) -> tuple[Bispectrum2D | None, Bispectrum2D | None, Bispectrum2D | None]:
        """Assign each term once using Slepian, semi-analytic, numeric priority."""
        if not isinstance(self._bispectrum, Bispectrum2D):
            raise TypeError("the hybrid route requires a Bispectrum2D")
        if self._hybrid_plan_ready:
            return (
                self._slepian_source,
                self._semi_analytic_source,
                self._numeric_source,
            )

        slepian_names = []
        semi_analytic_names = []
        numeric_names = []
        unsupported = []
        for term in self._bispectrum.terms:
            if self._term_supports(term, SlepianRepresentation2D):
                slepian_names.append(term.name)
            elif self._term_supports(term, SemiAnalyticRepresentation2D):
                semi_analytic_names.append(term.name)
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
        self._semi_analytic_source = (
            self._bispectrum.select_terms(*semi_analytic_names)
            if semi_analytic_names
            else None
        )
        self._numeric_source = (
            self._bispectrum.select_terms(*numeric_names)
            if numeric_names
            else None
        )
        self._hybrid_plan_ready = True
        log(
            logger,
            logging.DEBUG,
            "hybrid plan: slepian=%s semi_analytic=%s numeric=%s",
            slepian_names,
            semi_analytic_names,
            numeric_names,
        )
        return self._slepian_source, self._semi_analytic_source, self._numeric_source

    def _numeric_pipeline_source(self):
        if self.route != "hybrid":
            return self._bispectrum
        _, semi_source, numeric_source = self._plan_hybrid_sources()
        sources = [source for source in (semi_source, numeric_source) if source]
        if not sources:
            raise ValueError("hybrid route has no numeric/semi-analytic terms")
        if len(sources) == 1:
            return sources[0]
        names = [term.name for source in sources for term in source.terms]
        return self._bispectrum.select_terms(*names)

    def _semi_calculator(self) -> SemiAnalyticCalculator:
        if self._semi_analytic_calculator is None:
            self._semi_analytic_calculator = SemiAnalyticCalculator(
                self.config.semi_analytic
            )
        return self._semi_analytic_calculator

    def multipoles(self) -> BispectrumMultipole:
        """Return multipoles for terms assigned to the HKernel pipeline."""
        self._sync_source_state()
        if self._multipole is None:
            if self.route == "semi_analytic":
                self._multipole = BispectrumMultipole.from_semi_analytic(
                    self._bispectrum, calculator=self._semi_calculator()
                )
            elif self.route == "hybrid":
                self._multipole = BispectrumMultipole.from_hybrid(
                    self.config.multipole,
                    self._numeric_pipeline_source(),
                    semi_analytic_calculator=self._semi_calculator(),
                    basis=self.config.basis,
                )
            else:
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
        r"""Return coupled Fourier kernels on the full internal ell grid.

        For each physical component ``epsilon`` and allowed opening-angle mode
        :math:`k`, this evaluates

        .. math::

           H_k(\ell_2,\ell_3)=\sum_L
           B_L(\ell_2,\ell_3)G_{Lk}(\sigma;\psi),\qquad
           \psi=\tan^{-1}(\ell_3/\ell_2),

        with :math:`\sigma_i=\epsilon_i s_i`. Canonically equivalent
        ``(sigma1, nu_k)`` combinations share one stored array; aliases retain
        the requested ``(epsilon, k)`` labels. The table remains on the dense
        FFTLog grid because radial transformation occurs downstream.
        """
        self._sync_source_state()
        requested_epsilons = self._epsilons(epsilons)
        if requested_epsilons in self._hkernel_tables:
            log(logger, logging.DEBUG, "HKernel cache hit epsilons=%s", requested_epsilons)
            return self._hkernel_tables[requested_epsilons]

        log(
            logger,
            logging.INFO,
            "building HKernel: route=%s epsilons=%d multipoles=%d ell_grid=%dx%d",
            self.route,
            len(requested_epsilons),
            self._multipole_modes().size,
            self.grid.ell.size,
            self.grid.ell.size,
        )
        started = time.perf_counter()

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
        log(logger, logging.INFO, "HKernel finished in %.3f s", time.perf_counter() - started)
        return table

    def zetak(
        self,
        *,
        epsilons: Iterable[tuple[int, int, int]] | None = None,
    ) -> ZetaKTable:
        r"""Return route-independent 3PCF opening-angle modes.

        ``ZetaK`` stores :math:`\zeta_k(\theta_1,\theta_2)` before the final
        opening-angle sum. Numeric and semi-analytic terms pass through
        ``HKernel`` and a double Hankel transform; Slepian terms construct the
        same physical modes directly. Hybrid evaluation sums equal physical
        keys from all routes, so route identity is deliberately absent from
        the cache key and result table.
        """
        self._sync_source_state()
        requested_epsilons = self._epsilons(epsilons)
        if requested_epsilons in self._zetak_tables:
            log(logger, logging.DEBUG, "ZetaK cache hit epsilons=%s", requested_epsilons)
            return self._zetak_tables[requested_epsilons]

        log(logger, logging.INFO, "building ZetaK with route=%s", self.route)
        started = time.perf_counter()

        if self.route == "slepian":
            table = self._zetak_slepian(
                requested_epsilons, self._bispectrum
            )
        elif self.route == "hybrid":
            slepian_source, semi_source, numeric_source = self._plan_hybrid_sources()
            contributions = []
            if slepian_source is not None:
                contributions.append(
                    self._zetak_slepian(requested_epsilons, slepian_source)
                )
            if semi_source is not None or numeric_source is not None:
                contributions.append(
                    self._zetak_numeric_semianalytic(requested_epsilons)
                )
            table = self._sum_zetak_tables(contributions)
        else:
            table = self._zetak_numeric_semianalytic(requested_epsilons)

        self._zetak_tables[requested_epsilons] = table
        log(logger, logging.INFO, "ZetaK finished in %.3f s", time.perf_counter() - started)
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
        r"""Double-Hankel transform the shared ``HKernel`` pipeline.

        For effective Bessel orders :math:`m_k,n_k` and total spin
        :math:`\Sigma`, the implemented convention is

        .. math::

           \zeta_k={(-i)^\Sigma\over(2\pi)^3}
           \int d\ln\ell_2\,d\ln\ell_3\,
           \ell_2^2\ell_3^2 H_k
           J_{m_k}(\ell_2\theta_1)J_{n_k}(\ell_3\theta_2).

        The prefactor and :math:`\ell_i^2` measures are inserted here because
        ``double_hankel_transform`` accepts the complete logarithmic-measure
        integrand. The tuned dense result is downsampled exactly onto the
        user-requested theta bins.
        """

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
            log(
                logger,
                logging.INFO,
                "evaluating Slepian contribution: epsilon=%s modes=%d",
                epsilon,
                k_values.size,
            )
            timing_before = self._slepian_calculator.timing_summary
            started = time.perf_counter()
            mode_values = self._slepian_calculator.evaluate_modes(
                bispectrum,
                self.grid.ell,
                self.theta,
                k_values,
                sigma=sigma,
            )
            elapsed = time.perf_counter() - started
            timing_after = self._slepian_calculator.timing_summary
            delta = {
                key: timing_after.get(key, 0.0) - timing_before.get(key, 0.0)
                for key in timing_after.keys() | timing_before.keys()
            }
            build_seconds = delta.get("regular_matrix_build_seconds", 0.0)
            compression_seconds = delta.get(
                "low_rank_compression_seconds", 0.0
            )
            contraction_seconds = delta.get(
                "low_rank_contraction_seconds", 0.0
            )
            tracked = build_seconds + compression_seconds + contraction_seconds
            log(
                logger,
                logging.INFO,
                "Slepian epsilon=%s finished in %.3f s: F_ab=%.3f s, SVD=%.3f s, contraction=%.3f s, other=%.3f s, builds=%d, hits=%d",
                epsilon,
                elapsed,
                build_seconds,
                compression_seconds,
                contraction_seconds,
                max(elapsed - tracked, 0.0),
                round(delta.get("low_rank_matrix_builds", 0.0)),
                round(delta.get("low_rank_cache_hits", 0.0)),
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
        r"""Resum opening-angle modes and return the requested projection.

        The cached ``x``-projection is assembled as
        :math:`\zeta(\theta_1,\theta_2,\phi)=\sum_k
        \zeta_k(\theta_1,\theta_2)e^{i\nu_k\phi}`. Projection conversion is a
        cheap phase rotation applied only to the returned table, so
        ``projection`` is not part of the expensive cache key.
        """
        self._sync_source_state()
        requested_epsilons = self._epsilons(epsilons)
        target_projection = _projection_name(projection)
        if requested_epsilons in self._zeta_tables:
            log(logger, logging.DEBUG, "Zeta cache hit; converting projection to %s", target_projection)
            return self._zeta_tables[requested_epsilons].to_projection(
                target_projection
            )

        spin_spec = SpinSpec(self.config.spin)
        log(
            logger,
            logging.INFO,
            "resumming Zeta: epsilons=%d phi_bins=%d projection=%s",
            len(requested_epsilons),
            self.phi.size,
            target_projection,
        )
        started = time.perf_counter()
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
        log(logger, logging.INFO, "Zeta finished in %.3f s", time.perf_counter() - started)
        return x_table.to_projection(target_projection)
