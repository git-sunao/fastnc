"""FFTLog grid objects for 3PCF calculations.

The 3PCF pipeline uses one common logarithmic Fourier grid and the matching
FFTLog real-space grid.  Keeping both pieces in one object avoids accidental
mismatches between B_L, H_k, and zeta_k stages.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from ..hankel.grid import TunedFFTGrid, make_fftlog_grid
from .config import ThreePCFConfig


@dataclass(frozen=True)
class FFTGrid:
    """Common Fourier/real-space grid for the 3PCF pipeline.

    Parameters
    ----------
    ell
        One-dimensional Fourier-space FFTLog grid.
    theta_fft
        Full real-space FFTLog output grid.
    down_sampler
        Indices selecting the requested theta values from ``theta_fft``.
    xy
        FFTLog ``xy`` parameter.
    ELL1, ELL2
        Broadcast two-dimensional Fourier grids.
    psi_ell
        ``atan2(ELL2, ELL1)`` on the Fourier grid.
    theta
        User-facing real-space grid selected by ``down_sampler``.
    theta_user
        Alias of ``theta`` kept for user-facing APIs.
    THETA1, THETA2
        Broadcast user-facing real-space grids.
    THETA1_fft, THETA2_fft
        Broadcast full FFTLog real-space grids.
    """

    ell: np.ndarray
    theta_fft: np.ndarray
    down_sampler: np.ndarray
    xy: float
    ELL1: np.ndarray
    ELL2: np.ndarray
    psi_ell: np.ndarray
    theta: np.ndarray
    theta_user: np.ndarray
    THETA1: np.ndarray
    THETA2: np.ndarray
    THETA1_fft: np.ndarray
    THETA2_fft: np.ndarray

    @classmethod
    def from_tuned_grid(cls, tuned: TunedFFTGrid) -> "FFTGrid":
        ell = np.asarray(tuned.ell, dtype=float)
        theta_fft = np.asarray(tuned.theta, dtype=float)
        down_sampler = np.asarray(tuned.down_sampler, dtype=int)
        ELL1, ELL2 = np.meshgrid(ell, ell, indexing="ij")
        theta = theta_fft[down_sampler]
        theta_user = np.asarray(theta, dtype=float)
        THETA1, THETA2 = np.meshgrid(theta, theta, indexing="ij")
        THETA1_fft, THETA2_fft = np.meshgrid(theta_fft, theta_fft, indexing="ij")
        return cls(
            ell=ell,
            theta_fft=theta_fft,
            down_sampler=down_sampler,
            xy=float(tuned.xy),
            ELL1=ELL1,
            ELL2=ELL2,
            psi_ell=np.arctan2(ELL2, ELL1),
            theta=theta,
            theta_user=theta_user,
            THETA1=THETA1,
            THETA2=THETA2,
            THETA1_fft=THETA1_fft,
            THETA2_fft=THETA2_fft,
        )

    @classmethod
    def from_config(cls, config: ThreePCFConfig) -> "FFTGrid":
        tuned = make_fftlog_grid(
            config.ell_min,
            config.ell_max,
            config.n_ell,
            theta=config.target_theta(),
            xy=config.xy,
        )
        return cls.from_tuned_grid(tuned)

    @property
    def n_ell(self) -> int:
        return int(self.ell.size)

    @property
    def n_theta(self) -> int:
        return int(self.theta.size)

    @property
    def n_theta_fft(self) -> int:
        return int(self.theta_fft.size)

    @property
    def shape_ell(self) -> tuple[int, int]:
        return self.ELL1.shape

    @property
    def shape_theta(self) -> tuple[int, int]:
        return self.THETA1.shape

    @property
    def shape_theta_fft(self) -> tuple[int, int]:
        return self.THETA1_fft.shape

    def downsample_theta_array(self, values: np.ndarray, *, axis1: int = -2, axis2: int = -1) -> np.ndarray:
        """Select the user-facing theta grid from an array on the full FFT grid.

        The two theta axes are selected with ``down_sampler``.  This helper is
        used by grid-backed objects so that they can store full FFT-grid values
        while exposing downsampled/user-grid values through explicit getters.
        """
        arr = np.asarray(values)
        axis1 = axis1 % arr.ndim
        axis2 = axis2 % arr.ndim
        if axis1 == axis2:
            raise ValueError("axis1 and axis2 must be distinct.")
        arr = np.take(arr, self.down_sampler, axis=axis1)
        if axis2 > axis1:
            axis2 -= 0
        arr = np.take(arr, self.down_sampler, axis=axis2)
        return arr

    def validate_same(self, other: "FFTGrid") -> None:
        if self is other:
            return
        if not np.array_equal(self.ell, other.ell):
            raise ValueError("Fourier ell grids are inconsistent.")
        if not np.array_equal(self.theta, other.theta):
            raise ValueError("Real-space theta grids are inconsistent.")
        if float(self.xy) != float(other.xy):
            raise ValueError("FFTLog xy parameters are inconsistent.")


@dataclass
class GridBacked:
    """Mixin/base for objects defined on a common 3PCF FFT grid."""

    grid: FFTGrid

    @property
    def ell(self) -> np.ndarray:
        return self.grid.ell

    @property
    def theta(self) -> np.ndarray:
        return self.grid.theta

    @property
    def theta_user(self) -> np.ndarray:
        return self.grid.theta_user

    @property
    def theta_fft(self) -> np.ndarray:
        return self.grid.theta_fft

    @property
    def down_sampler(self) -> np.ndarray:
        return self.grid.down_sampler


