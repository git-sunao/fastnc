Architecture and Data Flow
==========================

fastnc separates physical definitions from calculation algorithms and sampled
results. This separation matters because the same bispectrum term may be
evaluated numerically for validation, transformed with a Slepian identity, or
reduced semi-analytically without changing its physical meaning.

The principal object layers are:

``Bispectrum3D`` and ``Bispectrum2D``
   Additive collections of named physical terms. They define what is being
   calculated and expose the available mathematical representations.

``LOSProjector``
   The physical radial projection, including LOS nodes, kernels, geometric
   prefactor, and quadrature convention. It converts a 3D definition into an
   angular bispectrum recipe.

``ThreePCF``
   The calculation facade. It owns target angular coordinates, numerical
   configuration, the selected route, calculators, and in-memory results.

``HKernelTable``, ``ZetaKTable``, and ``ZetaTable``
   Passive sampled outputs. They validate axes and key conventions but do not
   choose a calculation route.

The standard workflow is:

.. code-block:: text

   Bispectrum3D + LOSProjector
               |
               v
          Bispectrum2D
               |
               v
            ThreePCF
               |
               +-- numeric ------> multipoles -> HKernel --+
               +-- semi-analytic -> multipoles -> HKernel --+-> ZetaK -> Zeta
               +-- Slepian ------------------------------->--+

Only assembly layers know the APIs of multiple subsystems. Mathematical
primitives such as coupling coefficients, FFTLog grids, and Weber kernels are
independent of bispectrum model classes. This makes them reusable and lets
route calculations be tested with native 2D toy terms.

Physical state and numerical state
----------------------------------

Model parameters and numerical hyperparameters have different lifetimes.
Cosmological parameters, bias coefficients, and power spectra may change at
every inference sample. Target theta bins, the ell and Mellin grids, and LOS
quadrature nodes normally remain fixed throughout an analysis.

fastnc therefore uses source-state tokens to invalidate predictions while
retaining resources that depend only on fixed numerical geometry. Examples
include coupling matrices and Mellin-space radial kernels. Configuration
objects are immutable; changing a numerical hyperparameter is an explicit new
calculation, while replacing a bispectrum source uses
:meth:`~fastnc.threepcf.ThreePCF.set_bispectrum`.

This distinction is especially important for projected structured routes.
Changing cosmology changes ``k = ell / chi(z)`` and redshift-dependent Mellin
coefficients, but need not change a Mellin basis defined on the fixed angular
ell grid. A projector's LOS grid is numerical projection geometry, whereas
its kernel values are physical projection state.
