"""Three-point correlation function pipeline.

The legacy high-level ``ThreePCF`` object and object-based calculator are
archived. Numerical data structures remain available while pure route kernels
and assembly are rebuilt under :mod:`fastnc.threepcf.routes`.
"""

from .config import ThreePCFConfig
from .grid import FFTGrid, GridBacked
from .bmultipole_grid import BMultipoleGrid
from .hkernel_grid import HKernel, HKernelGrid, HKernelKey
from .zetak_grid import ZetaKMode, ZetaKGrid, ZetaKKey
from .spin import (
    EffectiveSpinTriple,
    SpinSpec,
    ComponentSpec,
    as_effective_spin_triple,
    independent_epsilons,
    component_specs,
)
from .zeta_grid import ZetaGrid
from .bruteforce import (
    BruteForce3PCFAdaptiveTrial,
    BruteForce3PCFConfig,
    BruteForce3PCFResult,
    BruteForceX3PCF,
)

__all__ = [
    "ThreePCFConfig",
    "FFTGrid",
    "GridBacked",
    "BMultipoleGrid",
    "HKernel",
    "HKernelKey",
    "HKernelGrid",
    "ZetaKMode",
    "ZetaKKey",
    "ZetaKGrid",
    "EffectiveSpinTriple",
    "SpinSpec",
    "ComponentSpec",
    "as_effective_spin_triple",
    "independent_epsilons",
    "component_specs",
    "ZetaGrid",
    "BruteForce3PCFAdaptiveTrial",
    "BruteForce3PCFConfig",
    "BruteForce3PCFResult",
    "BruteForceX3PCF",
]
