"""Prepared fixed-z Slepian radial geometry."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class SlepianRadialGrid:
    x: np.ndarray
    weights: np.ndarray

    @classmethod
    def from_config(cls, config):
        x = np.geomspace(config.slepian.x_min, config.slepian.x_max, config.slepian.n_x)
        # trapezoidal weights for integral dx x (...)
        w = np.empty_like(x)
        w[1:-1] = 0.5 * (x[2:] - x[:-2])
        w[0] = 0.5 * (x[1] - x[0])
        w[-1] = 0.5 * (x[-1] - x[-2])
        return cls(x=x, weights=w)
