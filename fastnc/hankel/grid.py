"""FFTLog grid helpers.

These helpers are adapted from the old fastnc grid-tuning utilities.  They do
not modify the external FFTLog implementation; they only choose Fourier-space
and real-space grids that are convenient for downstream 2D-FFTLog calls.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class TunedFFTGrid:
    """Container for a tuned logarithmic FFT grid.

    Attributes
    ----------
    ell : ndarray
        Fourier-space grid, linearly spaced in log(ell).
    theta : ndarray
        Corresponding real-space grid, ``theta = xy / ell[::-1]`` in log-space.
    down_sampler : ndarray
        Indices selecting the requested theta bins from ``theta``.
    xy : float
        FFTLog xy parameter required to place requested theta bins on the grid.
    """

    ell: np.ndarray
    theta: np.ndarray
    down_sampler: np.ndarray
    xy: float


def tune_fft_grid_size(lmin: float, lmax: float, nfft_min: int, dt: float):
    """Tune a linear grid in log-space to align with a target bin spacing.

    Parameters are in log-space.  This is the old fastnc algorithm: choose an
    integer skip factor so that the requested bin spacing ``dt`` is an integer
    multiple of the FFT grid spacing, and keep an even grid length for FFTLog.
    """
    nskip = int(np.ceil((int(nfft_min) - 1) * float(dt) / (float(lmax) - float(lmin))))
    nskip = max(nskip, 1)
    dl = float(dt) / nskip
    nfft = int(np.floor((float(lmax) - float(lmin)) / dl + 1.0))
    ep = (float(lmax) - float(lmin)) / dl + 1.0 - nfft
    lmax_new = float(lmax) - dl * ep / 2.0
    lmin_new = float(lmin) + dl * ep / 2.0
    if nfft % 2 != 0:
        nfft -= 1
        lmax_new -= dl / 2.0
        lmin_new += dl / 2.0
    return lmin_new, lmax_new, nfft, nskip


def tune_fft_real_bin(log_ell: np.ndarray, log_theta_pivot: float) -> float:
    """Return the log-space xy shift that aligns a real-space grid point."""
    log_ell = np.asarray(log_ell, dtype=float)
    dl = log_ell[1] - log_ell[0]
    ep = np.modf((float(log_theta_pivot) + log_ell[-1]) / dl)[0]
    return float(ep * dl)


def get_tuned_fftgrid(log_ell_min: float, log_ell_max: float, nfft_min: int, log_theta: np.ndarray) -> TunedFFTGrid:
    """Return a tuned FFTLog grid for requested log(theta) bins.

    The returned Fourier grid is still logarithmically uniform, but its spacing
    and FFTLog ``xy`` parameter are chosen so that the FFTLog real-space grid

        log(theta_fft) = log(xy) - log(ell)[::-1]

    contains every requested ``log_theta`` value exactly at ``down_sampler``.
    The requested theta values must be equally spaced in log-space.
    """
    log_theta = np.asarray(log_theta, dtype=float)
    if log_theta.ndim != 1 or log_theta.size < 1:
        raise ValueError("log_theta must be a one-dimensional array with at least one entry.")
    if np.any(~np.isfinite(log_theta)):
        raise ValueError("log_theta must contain finite values.")
    if log_theta.size >= 2:
        dt = float(np.diff(log_theta)[0])
        if dt <= 0.0:
            raise ValueError("log_theta must be strictly increasing.")
        if not np.allclose(np.diff(log_theta), dt):
            raise ValueError("log_theta must be evenly spaced for tuned FFT grid construction.")
    else:
        # With one requested bin, keep roughly the input Fourier spacing.
        dt = (float(log_ell_max) - float(log_ell_min)) / max(int(nfft_min) - 1, 1)

    log_ell_min = float(log_ell_min)
    log_ell_max = float(log_ell_max)
    nfft_min = int(nfft_min)
    if nfft_min < 2:
        raise ValueError("nfft_min must be >= 2.")

    # Choose an integer skip factor so the requested theta spacing is an
    # integer multiple of the FFTLog grid spacing.
    nskip = int(np.ceil((nfft_min - 1) * dt / (log_ell_max - log_ell_min)))
    nskip = max(nskip, 1)
    dl = dt / nskip

    # Need enough points both to cover the requested ell range and to contain
    # all requested theta values at the selected skip factor.
    n_cover_ell = int(np.ceil((log_ell_max - log_ell_min) / dl)) + 1
    n_cover_theta = nskip * (log_theta.size - 1) + 1
    nfft = max(nfft_min, n_cover_ell, n_cover_theta)
    if nfft % 2 != 0:
        nfft += 1

    # Center the tuned ell range around the requested range.
    ell_center = 0.5 * (log_ell_min + log_ell_max)
    ell_width = dl * (nfft - 1)
    lmin = ell_center - 0.5 * ell_width
    lmax = ell_center + 0.5 * ell_width
    log_ell = np.linspace(lmin, lmax, nfft)

    # Place the first requested theta value exactly at theta_fft[0].
    # Then theta_fft[nskip * i] = log_theta[i].
    log_xy = float(log_theta[0] + log_ell[-1])
    log_theta_fft = log_xy - log_ell[::-1]
    down_sampler = nskip * np.arange(log_theta.size, dtype=int)

    if np.any(down_sampler < 0) or np.any(down_sampler >= log_theta_fft.size):
        raise RuntimeError("Requested theta bins are not covered by the tuned FFT grid.")
    if not np.allclose(log_theta, log_theta_fft[down_sampler], rtol=1.0e-12, atol=1.0e-12):
        maxerr = float(np.max(np.abs(log_theta - log_theta_fft[down_sampler])))
        raise RuntimeError(f"Failed to tune FFT grid to requested theta bins; max log-error={maxerr}.")

    return TunedFFTGrid(
        ell=np.exp(log_ell),
        theta=np.exp(log_theta_fft),
        down_sampler=down_sampler,
        xy=float(np.exp(log_xy)),
    )

def make_fftlog_grid(
    ell_min: float,
    ell_max: float,
    n_ell: int,
    *,
    theta: np.ndarray | None = None,
    xy: float = 1.0,
) -> TunedFFTGrid:
    """Create either a plain FFTLog grid or a tuned grid for requested theta bins."""
    if theta is None:
        n = int(n_ell)
        if n % 2 != 0:
            n -= 1
        ell = np.logspace(np.log10(ell_min), np.log10(ell_max), n)
        theta_fft = float(xy) / ell[::-1]
        return TunedFFTGrid(
            ell=ell,
            theta=theta_fft,
            down_sampler=np.arange(n),
            xy=float(xy),
        )

    theta = np.asarray(theta, dtype=float)
    if np.any(theta <= 0.0):
        raise ValueError("theta values must be positive.")
    return get_tuned_fftgrid(np.log(ell_min), np.log(ell_max), int(n_ell), np.log(theta))
