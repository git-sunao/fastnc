"""Three-point correlation function pipeline.

Public API
----------
Use :class:`ThreePCF` for ordinary calculations and
:class:`GenericThreePCFCalculator` (or the compatibility alias
:class:`ThreePCFCalculator`) for low-level generic stage debugging.
"""

from .api import ThreePCF
from .calculator import HybridThreePCFCalculator, GenericThreePCFCalculator, ThreePCFCalculator
from .config import ThreePCFConfig, SlepianConfig
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
    "ThreePCF",
    "HybridThreePCFCalculator",
    "GenericThreePCFCalculator",
    "ThreePCFCalculator",
    "ThreePCFConfig",
    "SlepianConfig",
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

from .slepian import SlepianThreePCFCalculator
