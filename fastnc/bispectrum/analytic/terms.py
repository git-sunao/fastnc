from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .fftlog import FFTLogComponent

@dataclass(frozen=True, kw_only=True)
class SemiAnalyticMultipoleTerm:
    """Base class for one additive contribution to a semi-analytic multipole.

    ``weight`` is a scalar or a callable ``weight(z)``.  It is deliberately
    kept separate from the FFTLog component so scaling a term does not create
    a new coefficient or angular-kernel cache entry.
    """

    weight: object = 1.0

    def scaled_by(self, weight):
        """Return an immutable copy multiplied by an additional weight."""
        return replace(self, weight=_multiply_term_weights(self.weight, weight))

    def evaluate(self, mode, k2, k3, z, *, cache, kernel_tables):
        raise NotImplementedError


def _term_weight_value(weight, z):
    return weight(z) if callable(weight) else weight


def _multiply_term_weights(left, right):
    if not callable(left) and not callable(right):
        return left * right

    def combined(z):
        return _term_weight_value(left, z) * _term_weight_value(right, z)

    return combined


@dataclass(frozen=True)
class LowRankVFunction:
    r"""Explicit low-rank representation of ``V(k2, k3, z)``.

    The function is represented exactly as

    .. math::
       V(k_2,k_3,z)=\sum_{a=1}^{R} f_a(k_2,z)g_a(k_3,z).

    The generic callable interface is retained through :meth:`__call__`, while
    tensor-product LOS projection can evaluate the one-dimensional factors
    separately and avoid repeated evaluations on the full ``(k2,k3)`` grid.
    """

    left_factors: tuple[object, ...]
    right_factors: tuple[object, ...]
    name: str = "low-rank-v"

    def __post_init__(self):
        left = tuple(self.left_factors)
        right = tuple(self.right_factors)
        if not left or len(left) != len(right):
            raise ValueError("left_factors and right_factors must have equal nonzero length")
        if not all(callable(factor) for factor in left + right):
            raise TypeError("all low-rank V factors must be callable")
        object.__setattr__(self, "left_factors", left)
        object.__setattr__(self, "right_factors", right)

    @property
    def rank(self):
        return len(self.left_factors)

    @staticmethod
    def _evaluate_factors(factors, k, z):
        values = [np.asarray(factor(k, z)) for factor in factors]
        values = np.broadcast_arrays(*values, np.asarray(k), np.asarray(z))[:len(values)]
        return np.stack(values, axis=0)

    def evaluate_left(self, k2, z):
        return self._evaluate_factors(self.left_factors, k2, z)

    def evaluate_right(self, k3, z):
        return self._evaluate_factors(self.right_factors, k3, z)

    def __call__(self, k2, k3, z):
        left = self.evaluate_left(k2, z)
        right = self.evaluate_right(k3, z)
        return np.sum(left * right, axis=0)


def ProductVFunction(left, right, *, name="product-v"):
    """Return an exact rank-one ``V(k2,k3,z)=left(k2,z) right(k3,z)``."""
    return LowRankVFunction((left,), (right,), name=name)


def LeftVFunction(left, *, name="left-v"):
    """Return ``V(k2,k3,z)=left(k2,z)`` as an exact rank-one object."""
    return ProductVFunction(left, lambda k, z: np.ones(np.broadcast(k, z).shape), name=name)


def RightVFunction(right, *, name="right-v"):
    """Return ``V(k2,k3,z)=right(k3,z)`` as an exact rank-one object."""
    return ProductVFunction(lambda k, z: np.ones(np.broadcast(k, z).shape), right, name=name)


@dataclass(frozen=True)
class SeparableMultipoleTerm(SemiAnalyticMultipoleTerm):
    """One term ``U V (k3/k)^p W`` in the appendix convention."""

    name: str
    component: FFTLogComponent
    p: int
    u: object
    v: object
    amplitude: complex = 1.0
    modes: tuple[int, ...] | None = None

    def supports_mode(self, mode: int) -> bool:
        return self.modes is None or int(mode) in self.modes

    @property
    def kernel_shifts(self):
        return (int(self.p),)

    def evaluate(self, mode, k2, k3, z, *, cache, kernel_tables):
        if not self.supports_mode(int(mode)):
            shape = np.broadcast(np.asarray(k2), np.asarray(k3)).shape
            value = np.zeros(shape, dtype=complex)
            return value.item() if value.shape == () else value
        k2, k3 = np.broadcast_arrays(np.asarray(k2, dtype=float), np.asarray(k3, dtype=float))
        k = np.hypot(k2, k3)
        x2, x3 = k2 / k, k3 / k
        r = np.minimum(k2, k3) / k
        coeff, nu = cache.get(self.component, z)
        table = kernel_tables[id(self.component)]
        kernel = table.evaluate(int(mode), r, shift=int(self.p))
        powers = k.reshape((1,) + k.shape) ** nu.reshape((-1,) + (1,) * k.ndim)
        result = np.sum(coeff.reshape((-1,) + (1,) * k.ndim) * powers * kernel, axis=0)
        result = (
            _term_weight_value(self.weight, z)
            * self.amplitude
            * self.u(x2, x3)
            * self.v(k2, k3, z)
            * result
        )
        return result.item() if result.shape == () else result


@dataclass(frozen=True)
class DirectFourierTerm(SemiAnalyticMultipoleTerm):
    """Term with a finite, explicitly known Fourier multipole coefficient."""

    name: str
    coefficient: object
    amplitude: complex = 1.0

    def evaluate(self, mode, k2, k3, z, *, cache, kernel_tables):
        value = (
            _term_weight_value(self.weight, z)
            * self.amplitude
            * self.coefficient(int(mode), k2, k3, z)
        )
        return np.asarray(value).item() if np.asarray(value).shape == () else value


_PROJECTED_STATE_UNSET = object()


