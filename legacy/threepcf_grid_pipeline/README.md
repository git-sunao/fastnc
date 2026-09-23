# Archived 3PCF Grid pipeline

This directory contains the version 2 prototype in which Grid classes also
performed calculations, selected spin modes, managed aliases, and owned
caches. It is reference material, not an importable compatibility package.

The archived modules include:

- the combined high-level 3PCF and FFT-grid configuration;
- computing `BMultipoleGrid`, `HKernelGrid`, `ZetaKGrid`, and `ZetaGrid`
  objects;
- the coupled spin/epsilon orchestration used by those Grid objects;
- the monolithic brute-force reference calculator.

Active code must not import from this directory. Reusable mathematics should
return to `fastnc` only as independently tested array/callable kernels, route
calculators, or passive result value types. The active spin and projection
conventions were retained under `fastnc/threepcf/conventions/`.
