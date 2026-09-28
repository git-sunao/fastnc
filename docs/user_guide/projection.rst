Line-of-sight Projection
========================

:class:`~fastnc.projection.LOSProjector` owns the redshift and comoving-distance
nodes used by line-of-sight integration. Optional
:class:`~fastnc.projection.KernelSet` objects supply radial weights for samples.

For a finite-width projection, the numeric convention is

.. math::

   B_{\mathrm{2D}}(\ell_1,\ell_2,\ell_3)
   = \int d\chi\; P(\chi)\,
     \prod_{i=1}^{3}W_i(\chi)\,
     B_{\mathrm{3D}}\!\left(
       \frac{\ell_1}{\chi},\frac{\ell_2}{\chi},
       \frac{\ell_3}{\chi},z(\chi)\right),

where the default prefactor is :math:`P(\chi)=\chi^{-4}`. Kernels are not
normalized implicitly; their normalization remains part of the physical model.

Fixed-redshift calculations
---------------------------

``LOSProjector.delta_like`` evaluates the source exactly at one ``(z, chi)``.
It performs no quadrature and applies no ``chi**-4`` prefactor. This is an
exact fixed-redshift benchmark, not a narrow finite-width approximation.

.. code-block:: python

   projector = LOSProjector.delta_like(z=0.5, chi=1300.0)
   b2d = projector.project(b3d)
   value = b2d.evaluate(ell1=100.0, ell2=120.0, ell3=80.0)

The final line evaluates ``b3d`` at ``ki = elli / chi`` and the specified
redshift.

Finite-width projections
------------------------

For a finite range, construct a projector from strictly increasing ``z`` and
``chi`` arrays. Numeric projection integrates the 3D bispectrum into a
directly evaluable 2D numeric representation. Structured representations
retain the projector so Slepian or semi-analytic calculators can perform LOS
integration at the mathematically appropriate later stage.

.. code-block:: python

   projector = LOSProjector(z=z, chi=chi, kernels=kernel_set)
   b2d = projector.project(
       b3d,
       sample_combination=("source_a", "source_b", "source_c"),
   )

The projector owns both quadrature nodes and physical weights. Consequently, a
projected ``Bispectrum2D`` remains directly evaluable while retaining enough
information for delayed structured LOS integration.

Kernels
-------

:class:`~fastnc.projection.Kernel1D` stores one radial weight sampled on
``(z, chi)``. Factory methods construct a source distribution, lensing
efficiency, nonlinear-alignment kernel, or lensing-plus-IA combination.
Algebra between compatible kernels is supported, but automatic normalization
is deliberately absent. Call ``Kernel1D.normalized``
only when normalization is part of the intended convention.

:class:`~fastnc.projection.KernelSet` maps stable sample names to kernels.
``sample_combination`` selects which kernels are multiplied at projection
time; repeated names are allowed when vertices use the same sample.

Projection state
----------------

``grid_signature`` identifies immutable LOS nodes and the quadrature rule.
``physical_state_token`` identifies kernel values, prefactor, shift, and
fixed-redshift status. Geometry-only resources can therefore be reused while
source-dependent projected results follow changes in physical inputs.
