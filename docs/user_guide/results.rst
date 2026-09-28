Results and Cache Behavior
==========================

Result classes are immutable, passive tables. They store coordinates, keys,
aliases, and read-only value arrays. Route calculators create them; the tables
do not know how their values were obtained.

HKernelTable
------------

:class:`~fastnc.threepcf.HKernelTable` stores angularly coupled kernels on the
full tuned FFTLog ell grid. Values have shape ``(n_key, n_ell, n_ell)``.
Physical ``(epsilon, k)`` requests are resolved through aliases to storage
keys. Symmetry-related requests can therefore reuse one computed array without
changing the physical component returned to the user.

ZetaKTable
----------

:class:`~fastnc.threepcf.ZetaKTable` stores opening-angle modes on the final
theta grid. Values have shape ``(n_key, n_theta, n_theta)``. Numeric and
semi-analytic routes reach this table through HKernel and a two-dimensional
Hankel transform; the Slepian route constructs compatible ZetaK modes
directly. Hybrid calculations add these contributions before resummation.

ZetaTable
---------

:class:`~fastnc.threepcf.ZetaTable` is the user-facing result. Values have
shape ``(n_component, n_theta, n_theta, n_phi)``. Access one component with:

.. code-block:: python

   component = zeta.get((1, 1, -1))

The first and second theta axes are the two sides defining the chosen vertex,
and the final axis is the opening angle. Arrays are read-only, preventing
accidental mutation of a cached result.

Projection conversion
---------------------

``ZetaTable.to_projection`` converts between X, centroid, and orthocenter
shear conventions using stored effective-spin triples. It does not redo
bispectrum multipoles, coupling, or radial transforms:

.. code-block:: python

   centroid = zeta.to_projection("centroid")

Calling ``threepcf.zeta(projection="centroid")`` performs the same conversion
while retaining only the canonical X-projected result in the expensive cache.

Cache invalidation
------------------

Caches are scoped to one ``ThreePCF`` instance and live in memory. Their
identity includes requested epsilon components and relevant physical source
state, but not final shear projection. Changing ``phi`` preserves ZetaK;
changing ``theta`` invalidates radial tables; replacing the bispectrum
invalidates source-dependent values; changing route invalidates route results.
