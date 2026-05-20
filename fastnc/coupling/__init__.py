"""Spin coupling package for the Fourier-basis 3PCF formula."""

from .compute import (
    CouplingIndex,
    allowed_k,
    bar_beta,
    coupling_delta,
    coupling_delta_float,
    coupling_G,
    coupling_G_quad,
    coupling_index,
    exact_zero_by_support,
    exact_zero_delta,
    spin_phase_coeff,
    spin_phase_coeff_from_keys,
    two_delta_from_L_k,
)
from .cache import CouplingCache, BCacheKey, CachePolicy
from .evaluate import coupling, coupling_by_delta, coupling_delta_from_cache, coupling_from_cache
from .matrix import CouplingKernel, CouplingKernelConfig, CouplingMatrix, CouplingMatrixConfig

__all__ = [
    "BCacheKey",
    "CachePolicy",
    "CouplingCache",
    "CouplingIndex",
    "CouplingKernel",
    "CouplingKernelConfig",
    "CouplingMatrix",
    "CouplingMatrixConfig",
    "allowed_k",
    "bar_beta",
    "coupling",
    "coupling_by_delta",
    "coupling_delta",
    "coupling_delta_float",
    "coupling_delta_from_cache",
    "coupling_from_cache",
    "coupling_G",
    "coupling_G_quad",
    "coupling_index",
    "exact_zero_by_support",
    "exact_zero_delta",
    "spin_phase_coeff",
    "spin_phase_coeff_from_keys",
    "two_delta_from_L_k",
]
