# Archived bispectrum object API

This directory is a reference archive created during the version 2 bispectrum
representation refactor. It is not an importable compatibility package and is
not part of the active `fastnc` architecture.

The archived code includes:

- the former `Bispectrum3D`/`Bispectrum2D` evaluator base classes;
- LOS, interpolation, multipole, regulator, collection, and grid wrappers
  coupled to those base classes;
- the unfinished and unvalidated `bispectrum/analytic` semi-analytic
  implementation;
- the former high-level `threepcf/api.py`, which called
  `Bispectrum2D.multipole()`;
- an unfinished Slepian-foundation test that referenced modules not present in
  the active branch.

Code may be read and individual numerical kernels may be recovered from here,
but active code must not import from this directory. Reusable pieces should be
extracted as array/callable-based functions with independent tests before they
return to `fastnc`.
