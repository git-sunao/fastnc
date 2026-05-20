"""Minimal 3PCF engine connecting bispectrum multipoles, coupling, and 2D FFTLog."""
from __future__ import annotations

from dataclasses import dataclass
import time
import numpy as np

from ..hankel.grid import make_fftlog_grid, TunedFFTGrid
from ..hankel.wrapper import DoubleHankelConfig, double_hankel_transform
from .config import ThreePCFConfig
from .kernel import HKernelBuilder
from .spin import SpinTriple, as_spin_triple
from .resum import resum_multipoles
from .projection import convert_projection


@dataclass
class ThreePCFMultipoles:
    """Output container for radial 3PCF multipoles."""

    theta1: np.ndarray
    theta2: np.ndarray
    k_values: np.ndarray
    zeta_k: np.ndarray
    sigma: tuple[int, int, int]
    config: ThreePCFConfig

    def resum(
        self,
        delta_phi,
        *,
        phase: str = "nu",
        normalization: float = 1.0,
        bin_width: float | None = None,
        projection: str = "x",
        component: int | None = None,
    ):
        zeta = resum_multipoles(
            self.zeta_k,
            self.k_values,
            delta_phi,
            self.sigma,
            phase=phase,
            normalization=normalization,
            bin_width=bin_width,
        )
        if projection in ("x", "cross", "times"):
            return zeta
        return convert_projection(
            zeta,
            self.theta1,
            self.theta2,
            delta_phi,
            from_projection="x",
            to_projection=projection,
            component=component,
            sigma=self.sigma,
        )


class ThreePCFCalculator:
    """Compute 3PCF radial coefficients from a bispectrum multipole object.

    This class implements the first working milestone:

        BispectrumMultipole + CouplingMatrix -> H_k -> 2D FFTLog -> zeta_k
        -> zeta(DeltaPhi).

    Projection conversion and aperture statistics are deliberately not included
    here.
    """

    def __init__(self, sigma, coupling, config: ThreePCFConfig | None = None):
        self.spin = as_spin_triple(sigma)
        self.sigma = self.spin.sigma
        self.coupling = coupling
        self.config = config or ThreePCFConfig()
        self.grid: TunedFFTGrid | None = None
        self.ell1_fft: np.ndarray | None = None
        self.ell2_fft: np.ndarray | None = None
        self.theta1_fft: np.ndarray | None = None
        self.theta2_fft: np.ndarray | None = None
        self.down_sampler: np.ndarray | None = None
        self.ELL1: np.ndarray | None = None
        self.ELL2: np.ndarray | None = None
        self.result: ThreePCFMultipoles | None = None

    def _timer(self, label: str, t0: float) -> float:
        if self.config.timing or self.config.verbose:
            print(f"{label}: {time.perf_counter() - t0:.3f} s")
        return time.perf_counter()

    def setup_fft_grid(self):
        c = self.config
        grid = make_fftlog_grid(c.ell_min, c.ell_max, c.n_ell, theta=c.target_theta(), xy=c.xy)
        self.grid = grid
        self.ell1_fft = grid.ell
        self.ell2_fft = grid.ell
        self.theta1_fft = grid.theta
        self.theta2_fft = grid.theta
        self.down_sampler = grid.down_sampler
        self.ELL1, self.ELL2 = np.meshgrid(grid.ell, grid.ell, indexing="ij")
        return grid

    def k_values(self):
        return self.spin.k_values(self.config.kmax)

    def compute_multipoles(self, bmultipole) -> ThreePCFMultipoles:
        t0 = time.perf_counter()
        if self.grid is None:
            self.setup_fft_grid()
        t0 = self._timer("setup_fft_grid", t0)

        assert self.ell1_fft is not None and self.ell2_fft is not None
        assert self.theta1_fft is not None and self.theta2_fft is not None
        assert self.down_sampler is not None and self.ELL1 is not None and self.ELL2 is not None

        cfg = self.config
        # Ensure the Hankel xy matches any tuning performed by setup_fft_grid.
        hankel_cfg = cfg.hankel
        if self.grid is not None:
            hankel_cfg = DoubleHankelConfig(
                xy=self.grid.xy,
                nu1=cfg.hankel.nu1,
                nu2=cfg.hankel.nu2,
                N_extrap_low=cfg.hankel.N_extrap_low,
                N_extrap_high=cfg.hankel.N_extrap_high,
                c_window_width=cfg.hankel.c_window_width,
                N_pad=cfg.hankel.N_pad,
                extra=dict(cfg.hankel.extra),
            )

        builder = HKernelBuilder(self.spin, self.coupling, cfg.Lmax, bispectrum_basis=getattr(bmultipole, "basis", "fourier-even"))
        k_values = self.k_values()
        zeta_list = []

        prefactor = ((-1j) ** self.spin.Sigma) / (2.0 * np.pi) ** 3

        for k in k_values:
            t_loop = time.perf_counter()
            m, n = self.spin.bessel_orders(float(k))
            Hk = builder.compute(float(k), bmultipole, self.ELL1, self.ELL2)
            # Eq. radial integral has ell1^2 ell2^2 H_k dlnell1 dlnell2.
            kernel = Hk * self.ELL1**2 * self.ELL2**2
            theta1, theta2, zeta = double_hankel_transform(
                self.ell1_fft,
                self.ell2_fft,
                prefactor * kernel,
                m,
                n,
                config=hankel_cfg,
                bin_width_logtheta=cfg.effective_bin_width_logtheta(),
            )
            zeta = zeta[np.ix_(self.down_sampler, self.down_sampler)]
            zeta_list.append(zeta)
            if cfg.verbose:
                print(f"k={k:g}, m={m}, n={n}, elapsed={time.perf_counter()-t_loop:.3f} s")

        zeta_k = np.asarray(zeta_list)
        theta1 = theta1[self.down_sampler]
        theta2 = theta2[self.down_sampler]
        result = ThreePCFMultipoles(theta1=theta1, theta2=theta2, k_values=k_values, zeta_k=zeta_k, sigma=self.sigma, config=cfg)
        self.result = result
        self._timer("compute_multipoles", t0)
        return result

    def resum(
        self,
        delta_phi,
        *,
        phase: str = "nu",
        normalization: float = 1.0,
        bin_width: float | None = None,
        projection: str = "x",
        component: int | None = None,
    ):
        if self.result is None:
            raise RuntimeError("compute_multipoles() must be called before resum().")
        return self.result.resum(
            delta_phi,
            phase=phase,
            normalization=normalization,
            bin_width=bin_width,
            projection=projection,
            component=component,
        )
