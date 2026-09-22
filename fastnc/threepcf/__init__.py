"""Three-point correlation function pipeline.

The legacy high-level ``ThreePCF`` object depended on the retired bispectrum
object API and is archived. The numerical grids and calculator remain active
while route assembly is redesigned.
"""

from .calculator import ThreePCFCalculator
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
    "ThreePCFCalculator",
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
