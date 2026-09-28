Three-point Correlation Functions
=================================

:class:`~fastnc.threepcf.ThreePCF` fixes target ``theta`` and ``phi`` bins,
the calculation grid, spin convention, and route. Expensive intermediate
tables are evaluated lazily and cached in memory.

The calculation always begins from :class:`~fastnc.bispectrum.Bispectrum2D`.
That object may be native or may retain a 3D source and
:class:`~fastnc.projection.LOSProjector`. The route decides when a retained LOS
integral is applied; users do not pass a projector to ``ThreePCF``.

Routes
------

``numeric``
   Computes angular bispectrum multipoles, HKernel, ZetaK, and Zeta entirely
   numerically. Every term is required to have a numeric representation.

``slepian``
   Uses separable Slepian representations to calculate ZetaK directly.

``semi_analytic``
   Computes analytic angular multipoles with coefficient-level LOS integration
   before entering the shared HKernel pipeline.

``hybrid``
   Assigns each term to Slepian, semi-analytic, or numeric handling according
   to available representations, then combines contributions at ZetaK.

For the numeric route the flow is ``Bispectrum2D -> BispectrumMultipole ->
HKernelTable -> ZetaKTable -> ZetaTable``. Slepian contributions enter directly
at ZetaK. Semi-analytic contributions first construct analytic multipoles and
then share the HKernel-to-ZetaK pipeline with numeric contributions. Hybrid
results are added at ZetaK, so final cache identity describes physical
components rather than internal routes.

Lazy evaluation
---------------

Public methods expose intermediate levels for validation:

.. code-block:: python

   bm = threepcf.multipoles()
   hkernel = threepcf.hkernel(epsilons=((1, 1, 1),))
   zetak = threepcf.zetak(epsilons=((1, 1, 1),))
   zeta = threepcf.zeta(epsilons=((1, 1, 1),))

Calling only ``zeta`` is sufficient for ordinary use. Each earlier stage is
constructed on demand and retained. Repeating an identical request does not
rebuild its upstream tables.

Spin and projection
-------------------

``ThreePCFConfig.spin`` gives physical spin at the three vertices. Components
are requested with epsilon triples. ``ThreePCF.zeta`` returns the X projection
by default and can convert the completed result to centroid or orthocenter
conventions without recomputing radial transforms.

For a scalar vertex, only ``epsilon=+1`` is representative. A request with
``epsilon=-1`` at a spin-zero vertex raises an error rather than silently
replacing the requested component. This keeps computational symmetry reduction
separate from the physical component returned.

.. code-block:: python

   epsilons = ((1, 1, -1), (1, -1, 1), (-1, 1, 1))
   zeta_centroid = threepcf.zeta(
       epsilons=epsilons,
       projection="centroid",
   )

Projection conversion is inexpensive. fastnc caches the X-projected table and
applies another convention when returning the result.

Changing an existing calculation
---------------------------------

Explicit setters preserve resources whose mathematical inputs are unchanged:

``set_phi``
   Clears only final angular resummation.

``set_theta``
   Rebuilds the tuned FFTLog grid and radial results.

``set_bispectrum``
   Replaces source-dependent predictions while preserving compatible grid
   resources.

``set_route``
   Clears route-dependent calculators and result tables.

The configuration object is immutable. To change Fourier-grid or algorithmic
controls, construct a new ``ThreePCFConfig`` and ``ThreePCF`` instance.
