"""Configuration objects for 3PCF calculations."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import numpy as np

from ..hankel.wrapper import DoubleHankelConfig


def _as_1d_positive_array(x, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array.")
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if np.any(~np.isfinite(arr)) or np.any(arr <= 0.0):
        raise ValueError(f"{name} must contain positive finite values.")
    if np.any(np.diff(arr) <= 0.0):
        raise ValueError(f"{name} must be strictly increasing.")
    return arr


@dataclass(frozen=True)
class ThreePCFConfig:
    """Configuration for the 3PCF engine.

    If a target real-space theta grid is specified, the FFTLog ell grid is
    automatically tuned so that the FFTLog output grid contains those theta
    values exactly.  There are three supported ways to specify the target grid:

    1. ``theta``: direct target theta values, interpreted as bin centers.
    2. ``theta_bins``: alias of ``theta``; also interpreted as bin centers.
    3. ``theta_bin_edges``: logarithmic bin edges.  The target theta values are
       the geometric bin centers, and ``bin_width_logtheta`` is inferred from
       the edge spacing unless explicitly provided.

    ``theta``/``theta_bins`` are useful for point evaluation.  ``theta_bin_edges``
    is useful for bin-averaged double-Hankel transforms.
    """
    # User-facing physical field spin.  Natural-component effective spins
    # are generated internally as sigma_i = epsilon_i * spin_i.
    spin: tuple[int, int, int] = (0, 0, 0)


    Lmax: int = 30
    kmax: float = 30.0
    ell_min: float = 1.0e-1
    ell_max: float = 1.0e5
    n_ell: int = 200

    # Target theta centers for tuned FFTLog output grid.
    theta: np.ndarray | None = None
    theta_bins: np.ndarray | None = None
    theta_bin_edges: np.ndarray | None = None

    xy: float = 1.0
    bin_width_logtheta: float | None = None
    hankel: DoubleHankelConfig = field(default_factory=DoubleHankelConfig)
    verbose: bool = False
    timing: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    epsilons: tuple[tuple[int, int, int], ...] | None = None
    coupling_kwargs: dict[str, Any] = field(default_factory=dict)

    def target_theta(self) -> np.ndarray | None:
        """Return target theta centers, or ``None`` for the full FFTLog grid."""
        specified = [self.theta is not None, self.theta_bins is not None, self.theta_bin_edges is not None]
        if sum(specified) > 1:
            raise ValueError("Specify only one of theta, theta_bins, or theta_bin_edges.")

        if self.theta is not None:
            return _as_1d_positive_array(self.theta, "theta")

        if self.theta_bins is not None:
            return _as_1d_positive_array(self.theta_bins, "theta_bins")

        if self.theta_bin_edges is not None:
            edges = _as_1d_positive_array(self.theta_bin_edges, "theta_bin_edges")
            if edges.size < 2:
                raise ValueError("theta_bin_edges must contain at least two edges.")
            return np.sqrt(edges[:-1] * edges[1:])

        return None

    def effective_bin_width_logtheta(self) -> float | None:
        """Return log-theta bin width used for bin-averaged Hankel transforms."""
        if self.bin_width_logtheta is not None:
            return float(self.bin_width_logtheta)

        if self.theta_bin_edges is None:
            return None

        edges = _as_1d_positive_array(self.theta_bin_edges, "theta_bin_edges")
        dlog = np.diff(np.log(edges))
        if not np.allclose(dlog, dlog[0]):
            raise ValueError("theta_bin_edges must be evenly spaced in log(theta) to infer bin_width_logtheta.")
        return float(dlog[0])
