Quick Start
===========

This example follows the complete public workflow: construct a physical 3D
bispectrum, define a line-of-sight projection, obtain an angular bispectrum,
and calculate a scalar 3PCF. ``simple_debug`` supplies a self-contained power
spectrum for examples; scientific analyses should use their own cosmology and
linear-power implementation.

.. code-block:: python

   import numpy as np

   import fastnc
   from fastnc.bispectrum import SPTMatterBispectrum3D
   from fastnc.projection import LOSProjector
   from fastnc.threepcf import ThreePCF, ThreePCFConfig

   fastnc.configure_logging("INFO")

   b3d = SPTMatterBispectrum3D.simple_debug()
   projector = LOSProjector.delta_like(z=0.5, chi=1300.0)
   b2d = projector.project(b3d)

   theta = np.geomspace(1.0e-3, 3.0e-2, 12)
   phi = np.linspace(0.0, np.pi, 25)
   config = ThreePCFConfig(
       spin=(0, 0, 0),
       Lmax=8,
       kmax=8,
       ell_min=1.0,
       ell_max=3.0e3,
       n_ell=64,
   )

   threepcf = ThreePCF(config, b2d, theta, phi, route="numeric")
   zeta = threepcf.zeta()

``zeta`` is a passive :class:`~fastnc.threepcf.ZetaTable`. Its ``values``
array has shape ``(n_component, n_theta, n_theta, n_phi)``. For this scalar
example the default component is ``(1, 1, 1)``:

.. code-block:: python

   scalar = zeta.get((1, 1, 1))
   diagonal = scalar[np.arange(theta.size), np.arange(theta.size)]

``diagonal`` has shape ``(n_theta, n_phi)`` and contains isosceles triangles.

The same projected bispectrum can use a different route when its terms provide
the required representations:

.. code-block:: python

   hybrid = ThreePCF(config, b2d, theta, phi, route="hybrid")
   zeta_hybrid = hybrid.zeta()

The route is fixed on the instance but may be changed explicitly with
:meth:`~fastnc.threepcf.ThreePCF.set_route`. This clears route-dependent
results while preserving reusable grid resources.

The expensive stages are lazy. Constructing ``ThreePCF`` allocates no
bispectrum multipoles or radial transforms; the first call to ``zeta`` builds
the required chain. Repeating the same call returns the in-memory cached
result. Enable logging to see which stages are built or reused.
