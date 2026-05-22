"""Low-level 3PCF calculator for the B_L -> H_k -> zeta_k -> zeta pipeline."""
from __future__ import annotations

import numpy as np

from ..coupling import CouplingMatrix
from ..coupling.cache import CouplingCacheSession
from ..hankel.wrapper import DoubleHankelConfig
from .bmultipole_grid import BMultipoleGrid
from .config import ThreePCFConfig
from .grid import FFTGrid
from .hkernel_grid import HKernelGrid
from .spin import SpinSpec, as_effective_spin_triple
from .zeta_grid import ZetaGrid
from .zetak_grid import ZetaKGrid


class ThreePCFCalculator:
    """Compute 3PCF grids from a bispectrum-multipole object.

    This is the low-level stage orchestrator.  It owns one common
    :class:`FFTGrid` and the stage grids

    - ``Bgrid`` for ``B_L(ell1, ell2)``,
    - ``Hgrid`` for angularly mixed ``H_k(ell1, ell2)``, and
    - ``ZKgrid`` for double-Hankel transformed ``zeta_k(theta1, theta2)``.

    The final real-space 3PCF is returned as :class:`ZetaGrid` by
    :meth:`compute_zeta`.
    """

    def __init__(
        self,
        bmultipole,
        config: ThreePCFConfig | None = None,
        *,
        coupling_kwargs: dict | None = None,
    ):
        if bmultipole is None:
            raise ValueError("bmultipole must be provided.")

        self.bmultipole = bmultipole
        self.config = config or ThreePCFConfig()

        self.coupling_kwargs = dict(self.config.coupling_kwargs)
        if coupling_kwargs is not None:
            self.coupling_kwargs.update(dict(coupling_kwargs))

        self.spin_spec = SpinSpec(self.config.spin)
        self.spin = self.spin_spec.spin
        self.grid = FFTGrid.from_config(self.config)

        basis = getattr(self.bmultipole, "basis", "fourier-even")
        self.Bgrid = BMultipoleGrid(grid=self.grid, Lmax=self.config.Lmax, basis=basis)
        self.Hgrid = HKernelGrid(spin=self.spin, kmax=self.config.kmax, grid=self.grid)
        self.ZKgrid = ZetaKGrid(spin=self.spin, kmax=self.config.kmax, grid=self.grid)
        # Backward-friendly attribute name for interactive inspection only.
        self.Zgrid: ZetaGrid | None = None

        self._coupling_cache_sessions: dict[str, CouplingCacheSession] = {}
        self._couplings: dict[tuple[int, int, int], CouplingMatrix] = {}

    # ------------------------------------------------------------------
    # Component bookkeeping
    @property
    def n_components(self) -> int:
        return self.spin_spec.n_components

    @property
    def components(self):
        return self.spin_spec.components()

    def epsilon_from_component(self, component: int) -> tuple[int, int, int]:
        return self.spin_spec.component(component).epsilon

    def sigma_from_epsilon(self, epsilon: tuple[int, int, int]) -> tuple[int, int, int]:
        return self.spin_spec.sigma_from_epsilon(epsilon)

    def _epsilons(
        self,
        *,
        epsilons=None,
        epsilon: tuple[int, int, int] | None = None,
        component: int | None = None,
        all_components: bool = False,
    ) -> tuple[tuple[int, int, int], ...]:
        if epsilons is not None:
            return tuple(tuple(int(e) for e in eps) for eps in epsilons)
        if epsilon is not None and component is not None:
            raise ValueError("Specify at most one of epsilon or component.")
        if epsilon is not None:
            idx, _ = self.spin_spec.component_index_from_epsilon(epsilon)
            return (self.spin_spec.component(idx).epsilon,)
        if component is not None:
            return (self.spin_spec.component(component).epsilon,)
        if all_components or self.config.epsilons is not None:
            eps = self.config.epsilons or self.spin_spec.representative_epsilons()
            return tuple(tuple(int(e) for e in ep) for ep in eps)
        return (self.spin_spec.component(0).epsilon,)

    def k_values(self, *, epsilon=None, component=None) -> np.ndarray:
        if epsilon is not None and component is not None:
            raise ValueError("Specify at most one of epsilon or component.")
        if epsilon is not None:
            sigma = self.spin_spec.sigma_from_epsilon(epsilon)
        elif component is not None:
            sigma = self.spin_spec.component(component).sigma
        else:
            sigma = self.spin_spec.component(0).sigma
        return as_effective_spin_triple(sigma).k_values(self.config.kmax)

    # ------------------------------------------------------------------
    # Coupling/session management
    def _cache_session_key(self) -> str:
        return str(self.coupling_kwargs.get("cache_file", "coupling_b_cache.h5"))

    def _get_cache_session(self) -> CouplingCacheSession | None:
        if not bool(self.coupling_kwargs.get("use_cache", True)):
            return None
        key = self._cache_session_key()
        if key not in self._coupling_cache_sessions:
            self._coupling_cache_sessions[key] = CouplingCacheSession(key)
        return self._coupling_cache_sessions[key]

    def _make_coupling(self, sigma: tuple[int, int, int]) -> CouplingMatrix:
        sig = tuple(int(x) for x in sigma)
        if sig not in self._couplings:
            kwargs = dict(self.coupling_kwargs)
            session = self._get_cache_session()
            if session is not None:
                kwargs["cache_session"] = session
            self._couplings[sig] = CouplingMatrix(sig[0], sig[1], sig[2], **kwargs)
        return self._couplings[sig]

    # ------------------------------------------------------------------
    # Stage execution
    def compute_bmultipoles(self, *, force: bool = False) -> BMultipoleGrid:
        """Compute and store ``B_L(ell1, ell2)`` on the managed FFT grid."""
        if force:
            self.Hgrid = HKernelGrid(spin=self.spin, kmax=self.config.kmax, grid=self.grid)
            self.ZKgrid = ZetaKGrid(spin=self.spin, kmax=self.config.kmax, grid=self.grid)
            self.Zgrid = None
        return self.Bgrid.compute(self.bmultipole, force=force)

    def compute_hkernels(
        self,
        *,
        epsilons=None,
        epsilon: tuple[int, int, int] | None = None,
        component: int | None = None,
        all_components: bool = False,
        force: bool = False,
    ) -> HKernelGrid:
        """Compute and store deduplicated ``H_k`` kernels."""
        eps = self._epsilons(
            epsilons=epsilons,
            epsilon=epsilon,
            component=component,
            all_components=all_components,
        )
        Bgrid = self.compute_bmultipoles()
        self.Hgrid.compute_all_epsilons(
            Bgrid,
            eps,
            coupling_factory=self._make_coupling,
            force=force,
        )
        return self.Hgrid

    def _hankel_config(self) -> DoubleHankelConfig:
        cfg = self.config
        return DoubleHankelConfig(
            xy=self.grid.xy,
            nu1=cfg.hankel.nu1,
            nu2=cfg.hankel.nu2,
            N_extrap_low=cfg.hankel.N_extrap_low,
            N_extrap_high=cfg.hankel.N_extrap_high,
            c_window_width=cfg.hankel.c_window_width,
            N_pad=cfg.hankel.N_pad,
            extra=dict(cfg.hankel.extra),
        )

    def compute_zetak(
        self,
        *,
        epsilons=None,
        epsilon: tuple[int, int, int] | None = None,
        component: int | None = None,
        all_components: bool = False,
        force: bool = False,
    ) -> ZetaKGrid:
        """Compute and store deduplicated ``zeta_k(theta1, theta2)`` modes."""
        eps = self._epsilons(
            epsilons=epsilons,
            epsilon=epsilon,
            component=component,
            all_components=all_components,
        )
        Hgrid = self.compute_hkernels(epsilons=eps)
        self.ZKgrid.compute_all_epsilons(
            Hgrid,
            eps,
            hankel_config=self._hankel_config(),
            bin_width_logtheta=self.config.effective_bin_width_logtheta(),
            force=force,
        )
        return self.ZKgrid

    # Explicit alias with separator for readability in prose.
    compute_zeta_k = compute_zetak

    def compute_zeta(
        self,
        delta_phi,
        *,
        phase: str = "nu",
        normalization: float = 1.0,
        bin_width: float | None = None,
        force: bool = False,
    ) -> ZetaGrid:
        """Compute the final all-component x-projection 3PCF grid.

        This is the preferred user-facing calculation method.  Internally it
        computes all representative ``ZetaKGrid`` components and then calls
        ``ZetaKGrid.resum(delta_phi)``.  Projection conversion is not performed
        here; call ``ZetaGrid.to_projection(...)`` on the returned object.
        """
        ZKgrid = self.compute_zetak(
            all_components=True,
            force=force,
        )
        self.Zgrid = ZKgrid.resum(
            delta_phi,
            phase=phase,
            normalization=normalization,
            bin_width=bin_width,
            config=self.config,
        )
        return self.Zgrid

    def compute(self, delta_phi, **kwargs) -> ZetaGrid:
        """Alias for :meth:`compute_zeta`."""
        return self.compute_zeta(delta_phi, **kwargs)
