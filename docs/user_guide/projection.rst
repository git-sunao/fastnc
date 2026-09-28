Line-of-sight Projection
========================

:class:`~fastnc.projection.LOSProjector` owns the redshift and comoving-distance
nodes used by line-of-sight integration. Its default geometric prefactor is
``chi**-4``. Optional :class:`~fastnc.projection.KernelSet` objects supply the
radial weights for selected samples.

Fixed-redshift calculations
---------------------------

``LOSProjector.delta_like`` evaluates the source exactly at one ``(z, chi)``.
It performs no quadrature and applies no ``chi**-4`` prefactor. This mode is
useful for debugging route calculations at a fixed redshift.

Finite-width projections
------------------------

For a finite redshift range, construct a projector from strictly increasing
``z`` and ``chi`` arrays. Numeric projection integrates the 3D bispectrum into
a directly evaluable 2D numeric representation. Structured representations
retain the projector so Slepian or semi-analytic calculators can perform LOS
integration at the appropriate later stage.
