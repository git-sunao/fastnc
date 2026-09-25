# fastnc development TODO

Status: Tracked deferred work. These items are not normative architecture;
accepted contracts belong in `docs/design/` and supporting evidence belongs in
`docs/notes/`.

This file records deferred work that is specific enough to affect future
implementation decisions. It is not a general wishlist. Keep completed items
in the relevant design document or Git history rather than accumulating them
here.

## Weber evaluator stability at large imaginary Mellin index

**Status:** Deferred. This is a known numerical residual in the current direct
and interpolated Weber evaluators, not an interpolation-only error.

For large `abs(Im(exponent))`, the direct evaluator's power series for
`hyp2f1(A, B; C; r**2)` becomes inaccurate as `r` approaches one. For
`exponent = -0.8 + 40j` and Bessel orders `(0, 0)`, comparison against an
80-digit `mpmath` reference showed relative errors of approximately
`6e-6` at `r=0.6`, `2e-4` at `r=0.7`, `1.7e-1` at `r=0.8`, and complete
failure beyond that. The jagged behavior seen in the validation notebook is
therefore artificial and is inherited by interpolation tables constructed
from the direct evaluator.

The preferred first repair is a piecewise hypergeometric evaluator. Keep the
current `r**2` series away from one and use the `z=1` connection formula, whose
series variable is `1-r**2`, near one. Evaluate Gamma-function coefficients in
log space and validate complex values over the actual FFTLog exponent and
Bessel-order grids. The switching rule must be based on demonstrated error,
not only on a fixed ratio.

A Hankel-function contour deformation is a later analytic/reference project.
Split `J_mu J_nu` into four Hankel products, rotate each component toward its
decaying half-plane, and track origin, branch-cut, contact, and Heaviside
contributions. This may provide an independent nonoscillatory reference, but
it is not required before continuing the staged Slepian implementation.

Completion requires:

- comparison against high-precision values over representative real biases,
  imaginary Mellin indices, Bessel-order pairs, and ratios;
- smooth direct values without artificial jagged structure near `r=1`;
- revalidation of the interpolated evaluator using the repaired direct path;
- final-error tests with actual FFTLog coefficients, rather than Weber-only
  relative errors;
- retention of a slower independent reference method.

The local diagnostic notebook is
`dev/slepian/threepcf_slepian_weber_interpolation.ipynb`; notebooks
remain excluded from Git.

## Public API docstrings

**Status:** Not started as a systematic pass.

Add consistent docstrings to the supported user-facing API after interfaces
have stabilized. Prioritize exported classes, constructors, class methods,
configuration fields, evaluators, state-changing setters, and returned result
objects. Internal helpers need docstrings only when their mathematical
contract, conventions, cache ownership, or numerical limitations are not
obvious from the implementation.

Each public docstring should state, where applicable:

- the mathematical object represented and the dimensional convention;
- parameter meanings, shapes, units, basis, spin, and epsilon conventions;
- accepted route or method choices and their default;
- return type and array-axis ordering;
- cache invalidation or recomputation caused by state changes;
- raised errors and numerically unsupported regions;
- whether an object is passive data or performs calculation.

Completion requires an inventory of names exported from package `__init__`
modules, docstrings for every supported public name, and a documentation or
introspection test that detects missing public docstrings. Do not document
legacy modules as supported API.
