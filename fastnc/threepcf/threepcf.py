"""Unified manager for numeric, Slepian, and semi-analytic 3PCF routes."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from fastnc.coupling import CouplingCacheSession, CouplingMatrix
from fastnc.coupling.cache import resolve_coupling_cache_file
from fastnc.hankel import TunedFFTGrid, double_hankel_transform
from fastnc.multipole import BispectrumMultipole

from .config import NumericRouteConfig
from .conventions import SpinSpec, as_effective_spin_triple
from .numeric import contract_hkernel
from .tables import HKernelKey, HKernelTable, ZetaKKey, ZetaKTable


class ThreePCF:
    """Manage route resources and assemble passive 3PCF result tables."""

    def __init__(
        self,
        config: NumericRouteConfig | None = None,
        *,
        coupling_factory: Callable[[tuple[int, int, int]], object] | None = None,
    ):
        self.config = config or NumericRouteConfig()
        if not isinstance(self.config, NumericRouteConfig):
            raise TypeError("config must be a NumericRouteConfig")
        self._coupling_factory = coupling_factory
        self._coupling_cache_sessions: dict[Path, CouplingCacheSession] = {}
        self._couplings: dict[tuple[int, int, int], object] = {}
        self.hkernel_table: HKernelTable | None = None
        self.zetak_table: ZetaKTable | None = None

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
                coupling = self._coupling_factory(sigma)
            else:
                kwargs = self.config.coupling_kwargs()
                session = self._cache_session()
                if session is not None:
                    kwargs["cache_session"] = session
                coupling = CouplingMatrix(*sigma, **kwargs)
            if not callable(coupling):
                raise TypeError("coupling_factory must return a callable object")
            self._couplings[sigma] = coupling
        return self._couplings[sigma]

    @staticmethod
    def _epsilons(
        spin_spec: SpinSpec,
        epsilons: Iterable[tuple[int, int, int]] | None,
    ) -> tuple[tuple[int, int, int], ...]:
        if epsilons is None:
            return spin_spec.representative_epsilons()
        return tuple(
            tuple(int(value) for value in epsilon) for epsilon in epsilons
        )

    @staticmethod
    def _hkey(sigma: tuple[int, int, int], k: float) -> HKernelKey:
        effective = as_effective_spin_triple(sigma)
        two_nu = int(round(2.0 * float(effective.nu(k))))
        return HKernelKey(sigma1=effective.sigma1, two_nu=two_nu)

    def hkernel(
        self,
        multipole: BispectrumMultipole,
        grid: TunedFFTGrid,
        *,
        spin: tuple[int, int, int] = (0, 0, 0),
        epsilons: Iterable[tuple[int, int, int]] | None = None,
        **params,
    ) -> HKernelTable:
        """Evaluate all required numeric H kernels on the full FFT grid."""
        if not isinstance(multipole, BispectrumMultipole):
            raise TypeError("multipole must be a BispectrumMultipole")
        if not isinstance(grid, TunedFFTGrid):
            raise TypeError("grid must be a TunedFFTGrid")

        spin_spec = SpinSpec(tuple(spin))
        requested_epsilons = self._epsilons(spin_spec, epsilons)
        ell2, ell3 = np.meshgrid(grid.ell, grid.ell, indexing="ij")
        psi = np.arctan2(ell3, ell2)
        psi_unique, inverse = np.unique(psi, return_inverse=True)
        modes = np.arange(-self.config.Lmax, self.config.Lmax + 1)
        coefficients = multipole.evaluate_fourier(
            modes, ell2, ell3, **params
        )

        values: dict[HKernelKey, np.ndarray] = {}
        for epsilon in requested_epsilons:
            sigma = spin_spec.sigma_from_epsilon(epsilon)
            effective = as_effective_spin_triple(sigma)
            coupling = self.coupling(sigma)
            for k in effective.k_values(self.config.kmax):
                key = self._hkey(sigma, float(k))
                if key in values:
                    continue
                coupling_values = np.empty(coefficients.shape, dtype=float)
                for index, mode in enumerate(modes):
                    sampled = np.asarray(
                        coupling(int(mode), float(k), psi_unique), dtype=float
                    )
                    coupling_values[index] = sampled[inverse].reshape(psi.shape)
                values[key] = contract_hkernel(
                    coefficients, coupling_values
                )

        keys = tuple(values)
        table = HKernelTable(
            grid,
            keys,
            np.stack([values[key] for key in keys]),
        )
        self.hkernel_table = table
        self.zetak_table = None
        return table

    def zetak_numeric(
        self,
        multipole: BispectrumMultipole,
        grid: TunedFFTGrid,
        *,
        spin: tuple[int, int, int] = (0, 0, 0),
        epsilons: Iterable[tuple[int, int, int]] | None = None,
        **params,
    ) -> ZetaKTable:
        """Run multipole coupling and double FFTLog to the common ZetaK table."""
        spin_spec = SpinSpec(tuple(spin))
        requested_epsilons = self._epsilons(spin_spec, epsilons)
        htable = self.hkernel(
            multipole,
            grid,
            spin=spin,
            epsilons=requested_epsilons,
            **params,
        )
        ell2, ell3 = np.meshgrid(grid.ell, grid.ell, indexing="ij")
        hankel = replace(self.config.hankel, xy=grid.xy)

        values: dict[ZetaKKey, np.ndarray] = {}
        for epsilon in requested_epsilons:
            sigma = spin_spec.sigma_from_epsilon(epsilon)
            effective = as_effective_spin_triple(sigma)
            for k in effective.k_values(self.config.kmax):
                hkey = self._hkey(sigma, float(k))
                m, n = effective.bessel_orders(float(k))
                key = ZetaKKey(hkey, m, n, effective.Sigma)
                if key in values:
                    continue
                prefactor = (-1j) ** effective.Sigma / (2.0 * np.pi) ** 3
                integrand = prefactor * htable.get(hkey) * ell2**2 * ell3**2
                theta1, theta2, full = double_hankel_transform(
                    grid.ell,
                    grid.ell,
                    integrand,
                    m,
                    n,
                    config=hankel,
                    bin_width_logtheta=self.config.bin_width_logtheta,
                )
                if not np.allclose(theta1, grid.theta) or not np.allclose(
                    theta2, grid.theta
                ):
                    raise RuntimeError(
                        "double Hankel output does not match the tuned theta grid"
                    )
                values[key] = grid.downsample_2d(full)

        keys = tuple(values)
        table = ZetaKTable(
            grid.target_theta,
            keys,
            np.stack([values[key] for key in keys]),
        )
        self.zetak_table = table
        return table
