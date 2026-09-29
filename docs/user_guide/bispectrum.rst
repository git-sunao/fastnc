Bispectra
=========

A fastnc bispectrum is an additive collection of named terms. Each term may
carry one or more representations of the same physical contribution. Numeric
representations evaluate arbitrary Fourier triangles directly; Slepian and
semi-analytic representations expose additional mathematical structure to
specialized 3PCF routes.

The distinction between a *term* and a *representation* is central. A term
identifies one physical additive contribution. Its representations are
alternative, mathematically equivalent recipes for evaluating that same
contribution. They are capabilities consumed by calculators, not independent
terms to be summed.

Three-dimensional bispectra
----------------------------

A :class:`~fastnc.bispectrum.Bispectrum3D` is a function of
``(k1, k2, k3, z)``. Built-in models include tree-level SPT matter and galaxy
bispectra, Bihalofit, and one-halo models. Model objects do not own projection
or 3PCF grids.

.. code-block:: python

   from fastnc.bispectrum import SPTMatterBispectrum3D

   b3d = SPTMatterBispectrum3D.simple_debug()
   value = b3d.evaluate(k1=0.1, k2=0.12, k3=0.08, z=0.5)

Array inputs follow the broadcasting behavior of the underlying model. The
wave-number arguments describe a closed Fourier triangle through side lengths;
invalid-triangle handling is controlled by the model's support policy.

BiHalofit Bh1 representations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``BiHalofitBispectrum3D`` always constructs both the exact numeric Bh1
representation and a trained low-rank semi-analytic representation. The
model-level configuration controls how the latter is constructed; it does not
select a 3PCF route.

.. code-block:: python

   from fastnc.bispectrum import (
       BiHalofitBh1SemiAnalyticConfig,
       BiHalofitBispectrum3D,
   )

   bh1_config = BiHalofitBh1SemiAnalyticConfig(
       rank=8,
       trained_basis="broad-debug-v1",
   )
   b3d = BiHalofitBispectrum3D.simple_debug(
       config_semi_analytic=bh1_config,
   )

Omitting ``config_semi_analytic`` uses its default value rather than disabling
the representation. A downstream ``ThreePCF`` with the numeric route uses the
exact representation. The hybrid route may select the semi-analytic
representation. The ``broad-debug-v1`` basis is the current parameter-domain
validation basis and the name is intentionally explicit about its training
domain.

Two-dimensional bispectra
--------------------------

A :class:`~fastnc.bispectrum.Bispectrum2D` is a function of
``(ell1, ell2, ell3)``. It may be defined natively or constructed from a 3D
bispectrum and :class:`~fastnc.projection.LOSProjector`. Downstream calculators
use the same ``Bispectrum2D`` interface in either case.

.. code-block:: python

   from fastnc.bispectrum import (
       Bispectrum2D,
       BispectrumTerm2D,
       NumericExpression2D,
   )

   term = BispectrumTerm2D(
       name="toy",
       representations=(
           NumericExpression2D(
               lambda ell1, ell2, ell3: 1.0 / (ell1 * ell2 * ell3)
           ),
       ),
   )
   b2d = Bispectrum2D((term,))

Native angular models are useful for route development because cosmology and
LOS projection can be excluded from a validation problem.

Terms and representations
-------------------------

Weighted terms preserve additive structure so the hybrid route can choose the
best available representation term by term. The current priority is Slepian,
then semi-analytic, then numeric fallback.

.. code-block:: python

   selected = b3d.select_terms("tree:F2:12:m+0")
   rescaled = 2.0 * selected
   combined = selected + rescaled

For a 3D term the coefficient may be a scalar or callable ``coefficient(z)``.
For a 2D term it may be a scalar or zero-argument callable. Keeping terms
separate preserves hybrid route choices.

Numeric interpolation
---------------------

:meth:`~fastnc.bispectrum.Bispectrum2D.interpolate` replaces every numeric
representation with a self-contained interpolation representation. The table
uses ``(log ell2, log ell3, mu23)`` coordinates, where ``mu23`` determines the
closing side ``ell1``. Non-numeric representations remain attached.

.. code-block:: python

   import numpy as np
   from fastnc.bispectrum import TriangleInterpolationConfig

   interpolation = TriangleInterpolationConfig(
       ell2=np.geomspace(10.0, 5000.0, 80),
       ell3=np.geomspace(10.0, 5000.0, 80),
       mu23=np.linspace(-1.0, 1.0, 65),
   )
   b2d_interpolated = b2d.interpolate(interpolation, prepare=True)

The table owns a source-state token. If the source revision changes, the next
evaluation rebuilds it. Changing the interpolation grid requires a new
configuration. Selected numeric terms may first be combined to construct one
interpolation table for their sum.
