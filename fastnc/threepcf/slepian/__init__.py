"""Direct Slepian-route numerical machinery."""
from .calculator import SlepianThreePCFCalculator
from .fftlog import FFTLogExpansion, decompose_log_powerlaw, reconstruct
from .geometry import SlepianRadialGrid
from .los_moments import (PreparedLOSMoments, PreparedBatchedLOSMoments, SlepianLOSMomentRule, FactorizedGrowthMomentRule, FactorizedGrowthBatchMomentRule, GeneralCoefficientMomentRule, GeneralCoefficientBatchMomentRule)
from .weber import WeberTableCache, canonical_bessel_order, single_bessel_factor

__all__ = [
    'SlepianThreePCFCalculator', 'FFTLogExpansion', 'decompose_log_powerlaw',
    'reconstruct', 'SlepianRadialGrid', 'WeberTableCache',
    'canonical_bessel_order', 'single_bessel_factor',
    'PreparedLOSMoments', 'PreparedBatchedLOSMoments', 'SlepianLOSMomentRule',
    'FactorizedGrowthMomentRule', 'FactorizedGrowthBatchMomentRule',
    'GeneralCoefficientMomentRule', 'GeneralCoefficientBatchMomentRule',
]

from .terms import RadialKernelKey, CompiledSlepianTerm, CompiledModePlan, compile_mode_plan
