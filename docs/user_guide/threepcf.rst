Three-point Correlation Functions
=================================

:class:`~fastnc.threepcf.ThreePCF` fixes the target ``theta`` and ``phi`` bins,
calculation grid, spin convention, and route. Expensive intermediate tables
are evaluated lazily and cached in memory.

Routes
------

``numeric``
   Computes angular bispectrum multipoles, HKernel, ZetaK, and Zeta entirely
   numerically.

``slepian``
   Uses separable Slepian representations to calculate ZetaK directly.

``semi_analytic``
   Computes analytic angular multipoles with coefficient-level LOS integration
   before entering the shared HKernel pipeline.

``hybrid``
   Assigns each bispectrum term to Slepian, semi-analytic, or numeric handling
   according to its available representations, then combines contributions at
   ZetaK.

Spin and projection
-------------------

``ThreePCFConfig.spin`` gives the physical spin at the three vertices.
Components are requested with epsilon triples. ``ThreePCF.zeta`` returns the
X projection by default and can convert the completed result to centroid or
orthocenter conventions without recomputing the radial transforms.
