Bispectra
=========

A fastnc bispectrum is an additive collection of named terms. Each term may
carry one or more representations of the same physical contribution. Numeric
representations evaluate arbitrary Fourier triangles directly; Slepian and
semi-analytic representations expose additional mathematical structure to
specialized 3PCF routes.

Three-dimensional bispectra
----------------------------

A :class:`~fastnc.bispectrum.Bispectrum3D` is a function of
``(k1, k2, k3, z)``. Built-in models include tree-level SPT matter and galaxy
bispectra, Bihalofit, and one-halo models. Model objects do not own projection
or 3PCF grids.

Two-dimensional bispectra
--------------------------

A :class:`~fastnc.bispectrum.Bispectrum2D` is a function of
``(ell1, ell2, ell3)``. It may be defined natively or constructed from a 3D
bispectrum and a :class:`~fastnc.projection.LOSProjector`. Downstream
calculators use the same ``Bispectrum2D`` interface in either case.

Terms and representations
-------------------------

Weighted terms preserve additive structure so the hybrid route can choose the
best available representation term by term. The current priority is Slepian,
then semi-analytic, then numeric fallback.
