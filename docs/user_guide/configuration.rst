Configuration
=============

:class:`~fastnc.threepcf.ThreePCFConfig` collects physical conventions and
numerical controls fixed for one calculation instance. It is a frozen
dataclass, so changes are explicit and cannot silently leave stale caches.

Angular conventions
-------------------

``spin`` is the three-vertex physical spin tuple. ``basis`` selects the angular
multipole basis used consistently by numeric decomposition and coupling.
Fourier, cosine, sine, and Legendre bases are accepted; coupling internally
uses exact finite recombination of Fourier primitives where applicable.

``Lmax`` controls retained angular bispectrum multipoles. ``kmax`` controls
the opening-angle modes retained in ZetaK and final resummation. They are
distinct truncations and should be convergence-tested independently.

Fourier and real-space grids
----------------------------

``ell_min``, ``ell_max``, and ``n_ell`` define the logarithmic angular Fourier
grid. Supplied target ``theta`` bins tune the FFTLog grid so the
high-resolution transform grid down-samples directly onto requested
coordinates. This avoids another interpolation of the final prediction.

``theta`` must be positive, one-dimensional, strictly increasing, and evenly
spaced in ``log(theta)``. ``phi`` must be finite and strictly increasing. They
are constructor arguments of ``ThreePCF`` because they are result coordinates.

Route-specific controls
-----------------------

``multipole`` configures numeric angular integration. ``hankel`` configures
the two-dimensional FFTLog transform. ``slepian`` controls Mellin transforms,
Weber evaluation, regular radial integration, and optional low-rank
factorization. ``semi_analytic`` controls the Appendix-C-style angular
multipole calculation.

.. code-block:: python

   from fastnc.threepcf import SlepianConfig, ThreePCFConfig

   config = ThreePCFConfig(
       spin=(2, 2, 2),
       basis="fourier",
       Lmax=16,
       kmax=16,
       ell_min=1.0,
       ell_max=3.0e4,
       n_ell=128,
       slepian=SlepianConfig(
           weber_method="interpolated",
           regular_method="low_rank",
           regular_low_rank_rtol=1.0e-6,
       ),
   )

Approximation controls should not be selected from speed alone. Compare with
a higher-resolution numeric route or analytic toy problem while varying one
truncation at a time. Numeric and hybrid routes may disagree because finite
``Lmax`` misses squeezed-triangle power, rather than because the structured
route is inaccurate.

Coupling cache
--------------

Coupling matrices depend on spin, basis, mode truncation, and integration
settings, not cosmological amplitudes. ``use_coupling_cache`` enables reuse,
while ``coupling_cache_policy`` controls whether missing entries may be
created. Slepian radial resources are retained in memory rather than written
to a persistent disk cache.
