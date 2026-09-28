Quick Start
===========

The following example constructs a three-dimensional tree-level SPT matter
bispectrum, evaluates it at one redshift, and computes a scalar 3PCF through
the numeric route.

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
array is indexed by component, the two radial coordinates, and opening angle.

The same projected bispectrum can be evaluated with a different route when
its terms provide the required representations:

.. code-block:: python

   hybrid = ThreePCF(config, b2d, theta, phi, route="hybrid")
   zeta_hybrid = hybrid.zeta()
