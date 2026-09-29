Architecture and Data Flow
==========================

fastnc separates a physical model from the algorithm used to transform it and
from the arrays produced by that algorithm. This is not only an organizational
choice. The same bispectrum term can be evaluated numerically for validation,
through a Slepian representation for speed, or through analytic angular
multipoles without changing the physical term that the user selected.

The complete calculation is assembled by :class:`~fastnc.threepcf.ThreePCF`.
Only this assembly layer needs to know about bispectra, projection, route
calculators, coupling, radial transforms, and output tables at the same time.
Lower-level mathematical components remain usable and testable independently.

.. figure:: /_static/architecture-flow.svg
   :alt: fastnc data flow from bispectrum definitions through route calculators to passive 3PCF tables
   :align: center
   :width: 100%
   :class: architecture-flow

   Physical definitions are green, assembly objects are gray, calculation
   algorithms are blue or amber, and passive sampled results are purple.
   Dashed inputs are reusable structural resources rather than physical model
   predictions.

Design contracts
----------------

The architecture follows four contracts. They are useful when deciding where
a new model, approximation, cache, or transformation belongs.

**Definitions describe physics.** A bispectrum is an additive collection of
named terms. Each term may expose numeric, Slepian, and semi-analytic
representations. A representation contains enough mathematical information for
the corresponding calculator, but it does not choose the route used by a
3PCF calculation.

**Calculators implement algorithms.** Angular quadrature, coupling
contractions, Mellin transforms, Weber kernels, coefficient-level LOS
integration, and Hankel transforms belong to calculators or mathematical
primitives. Reusable numerical caches live beside the algorithm that owns
their validity conditions, not inside a physical bispectrum term.

**Tables are passive.** :class:`~fastnc.threepcf.HKernelTable`,
:class:`~fastnc.threepcf.ZetaKTable`, and
:class:`~fastnc.threepcf.ZetaTable` store sampled values, axes, and physical
keys. They do not select a route or call an upstream model. This keeps results
from one route interchangeable with equivalent results from another route.

**Assembly owns cross-package dependencies.** ``ThreePCF`` assigns terms to
routes, invokes calculators, combines contributions with equal physical keys,
and manages invalidation. A primitive such as a coupling coefficient or Weber
kernel does not accept a ``ThreePCF`` or bispectrum model object.

Physical definitions
--------------------

``Bispectrum3D`` represents a function of ``(k1, k2, k3, z)``.
``Bispectrum2D`` represents a function of ``(ell1, ell2, ell3)``. Both are
collections of ``fastnc.bispectrum`` terms rather than monolithic evaluator
classes. Algebra on terms can therefore preserve names and available
representations for later route planning.

A 2D bispectrum has two construction paths:

* A native ``Bispectrum2D`` contains angular expressions supplied directly by
  a model or user.
* A projected ``Bispectrum2D`` retains its ``Bispectrum3D`` source and
  :class:`~fastnc.projection.LOSProjector`. Retaining this provenance allows a
  downstream structured route to apply LOS integration at the mathematically
  appropriate stage instead of forcing every calculation through a sampled
  angular bispectrum.

``LOSProjector`` owns the physical projection recipe: kernel values, redshift
and comoving-distance nodes, geometric prefactors, and quadrature convention.
Its :meth:`~fastnc.projection.LOSProjector.delta_like` constructor represents
an exact fixed-redshift evaluation rather than a narrow numerical window.

Assembly and planning
---------------------

``ThreePCFConfig`` contains immutable numerical choices such as spin, angular
basis, Fourier truncation, FFTLog grid, and route-specific controls.
``ThreePCF`` combines that configuration with one ``Bispectrum2D`` and fixed
target ``theta`` and ``phi`` coordinates. Fixing the target coordinates lets
the tuned FFTLog grid include the requested theta bins directly, avoiding a
final interpolation of the theoretical prediction.

:meth:`~fastnc.threepcf.ThreePCF.calculation_plan` creates the same term-wise
assignment consumed by execution. It can therefore inspect a hybrid
calculation without evaluating the model. The plan is not a second,
documentation-only interpretation of route selection.

For a hybrid calculation, each term follows the most structured compatible
representation:

1. A supported Slepian representation contributes directly to ZetaK.
2. Otherwise, a semi-analytic representation supplies angular multipoles.
3. Otherwise, the numeric representation is evaluated by angular quadrature.

Terms assigned to the same route are grouped before evaluation. Numeric
fallback is therefore local to unsupported terms; it does not force the whole
bispectrum onto the numeric route.

Route data flow
---------------

The **numeric route** evaluates ``BispectrumMultipole`` on the internal ell
grid, contracts the selected angular basis with spin-dependent coupling
matrices to form HKernel, and applies a two-dimensional Hankel transform to
obtain ZetaK.

The **semi-analytic route** supplies analytic or structured angular
multipoles. After that point it deliberately shares the same coupling,
HKernel, and Hankel pipeline as the numeric route. For a projected source,
redshift-dependent coefficients can be integrated along the LOS without first
sampling the complete 2D bispectrum.

The **Slepian route** uses separability and Mellin-space radial identities to
construct ZetaK directly. It bypasses both ``BispectrumMultipole`` and
``HKernelTable``. Weber kernels, regular Mellin matrices, and their low-rank
approximations are structural resources of the Slepian calculator.

All contributions meet at ``ZetaKTable``. Route identity is intentionally not
part of a ZetaK key: numeric, semi-analytic, and Slepian arrays describing the
same physical spin component and mode must be added, not stored as unrelated
observables. ``ZetaTable`` is then obtained by resumming opening-angle modes.
Changing the final shear projection is a cheap phase conversion and does not
rebuild ZetaK.

Projection timing
-----------------

Projection is not always a single preprocessing operation. Its location
depends on the mathematical representation while the ``LOSProjector`` remains
the common owner of projection geometry.

For numeric evaluation, the projector evaluates the 3D source at
``k_i = ell_i / chi(z)`` and integrates the resulting angular bispectrum. The
downstream route then sees an ordinary numeric ``Bispectrum2D`` expression.

For structured routes, applying the LOS integral later can preserve
separability. Semi-analytic calculations may integrate redshift-dependent
coefficients, while Slepian calculations can contract radial kernels with
coefficients at each quadrature node before LOS summation. The projected 2D
object retains the source and projector so that ``ThreePCF`` can make this
choice without asking the user to construct a separate slice object.

State and cache lifetimes
-------------------------

Model state and numerical geometry have different lifetimes in an inference
calculation. Cosmological parameters, bias coefficients, profile amplitudes,
and power spectra may change at every sample. Theta bins, ell and Mellin grids,
LOS quadrature nodes, basis choice, and truncation normally remain fixed.

.. list-table:: Invalidation boundaries
   :header-rows: 1
   :widths: 23 31 46

   * - Change
     - Recomputed
     - Preserved
   * - Bispectrum model state
     - Multipoles, HKernel, ZetaK, Zeta, model-dependent coefficients
     - FFTLog geometry, coupling matrices, compatible structural radial kernels
   * - Bispectrum object
     - Route assignment and all source-dependent predictions
     - Target grid and compatible numerical geometry
   * - ``phi`` bins
     - Final Zeta mode resummation
     - Multipoles, HKernel, and ZetaK
   * - ``theta`` bins
     - Tuned FFTLog grid and all radial results
     - Source definition and compatible route resources
   * - Route
     - Calculators, route assignment, and result tables
     - Immutable source and configuration objects
   * - Numerical configuration
     - Entire ``ThreePCF`` calculation
     - Physical model objects may be passed to a new instance

Source objects expose state tokens so that ``ThreePCF`` can detect a model
update lazily. Invalidation removes physical predictions but keeps resources
whose mathematical inputs have not changed. In particular, replacing a
BiHalofit source by another compatible source does not by itself alter a
coupling matrix or a Mellin basis defined on the fixed ell grid.

Before an inference loop,
:meth:`~fastnc.threepcf.ThreePCF.warm_up` explicitly builds reusable route
resources and then discards the temporary physical prediction. The first
subsequent ``zeta`` call is therefore a real warm calculation rather than a
return of the warm-up result.

Result identity
---------------

Spin components are requested by epsilon triples. Internally, effective spin
and mode conventions may allow several requested labels to share one sampled
array. Table aliases preserve the requested physical label while canonical
keys prevent duplicate storage. Scalar vertices admit only ``epsilon=+1``;
invalid spin-zero sign requests raise an error instead of being silently
canonicalized.

HKernel and ZetaK keys contain the physical information needed to identify a
mode, not the route that produced it. Zeta caching similarly excludes the
final projection convention because conversion from the cached X projection
is inexpensive.

Extending fastnc
----------------

A new bispectrum model should define named terms and attach representations;
it should not call a 3PCF calculator. A new route should consume a documented
representation contract and produce an existing passive table wherever
possible. A new mathematical kernel should accept arrays and scalar
parameters rather than model classes. Finally, a new cache must state exactly
which model state, projection geometry, and numerical hyperparameters enter
its identity.

These boundaries allow each layer to be validated independently: native 2D
toy terms test route mathematics, numeric evaluation provides a common
benchmark, and end-to-end projected models test assembly and state handling.
