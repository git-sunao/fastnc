"""Pre-integrated Slepian-kernel cache infrastructure.

This module defines the production-facing cache identity and contraction API
for kernels in which the auxiliary x integral has already been performed.
Concrete SPT K(q) and BiHalofit K(q,rho) builders can use the existing fixed-z
reference path as an offline/reference integrator while keeping online
calculation to matrix interpolation and contraction.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class IntegratedKernelGeometry:
    """Dimensionless angular geometry of a pre-integrated kernel."""

    q: float
    orientation: str
    rho: float | None = None
    diagonal: bool = False

    def __post_init__(self):
        q = float(self.q)
        if not 0.0 < q <= 1.0:
            raise ValueError("q must satisfy 0 < q <= 1")
        if self.orientation not in {"theta1_ge_theta2", "theta1_lt_theta2"}:
            raise ValueError("unsupported integrated-kernel orientation")
        object.__setattr__(self, "q", q)
        if self.rho is not None:
            rho = float(self.rho)
            if rho <= 0.0:
                raise ValueError("rho must be positive")
            object.__setattr__(self, "rho", rho)
        object.__setattr__(self, "diagonal", bool(self.diagonal or q == 1.0))


@dataclass(frozen=True)
class IntegratedKernelMatrix:
    """One coefficient-independent pre-integrated FFTLog kernel matrix."""

    geometry: IntegratedKernelGeometry
    values: np.ndarray

    def __post_init__(self):
        values = np.asarray(self.values, dtype=np.complex128)
        if values.ndim != 2 or values.shape[0] != values.shape[1]:
            raise ValueError("integrated kernel matrix must be square")
        object.__setattr__(self, "values", values)

    def contract(self, left, right=None):
        left = np.asarray(left, dtype=np.complex128)
        right = left if right is None else np.asarray(right, dtype=np.complex128)
        if left.shape != (self.values.shape[0],) or right.shape != left.shape:
            raise ValueError("coefficient vectors do not match kernel dimension")
        # Bilinear FFTLog contraction, deliberately without complex conjugation.
        return left @ self.values @ right

    def low_rank(self, rank: int):
        U, s, Vh = np.linalg.svd(self.values, full_matrices=False)
        rank = min(int(rank), len(s))
        return U[:, :rank], s[:rank], Vh[:rank, :], s


class IntegratedKernelCache:
    """Simple exact-key cache for coefficient-independent integrated kernels."""

    def __init__(self):
        self._matrices = {}

    @property
    def n_matrices(self):
        return len(self._matrices)

    def get(self, key):
        return self._matrices.get(key)

    def put(self, key, matrix: IntegratedKernelMatrix):
        self._matrices[key] = matrix
        return matrix

    def clear(self):
        self._matrices.clear()


__all__ = [
    "IntegratedKernelGeometry", "IntegratedKernelMatrix", "IntegratedKernelCache",
]
