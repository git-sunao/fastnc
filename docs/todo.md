# fastnc development TODO

Status: Tracked deferred work. These items are not normative architecture;
accepted contracts belong in `docs/design/` and supporting evidence belongs in
`docs/notes/`.

This file records deferred work that is specific enough to affect future
implementation decisions. It is not a general wishlist. Keep completed items
in the relevant design document or Git history rather than accumulating them
here.

## Structured calculation logging

**Status:** Deferred until the numerical routes and cache lifecycle stabilize.

Design logging as one coherent facility rather than adding isolated messages
inside individual kernels. For the Slepian low-rank route, record at least the
requested rank or matrix tolerance, the retained rank, the maximum available
rank, and the matrix reconstruction error whenever a compressed matrix is
created. Cache hits should not repeat construction messages.

The broader pass should identify similarly useful events across routes:
expensive table construction versus cache reuse, selected numerical method,
grid dimensions and ranges, automatic truncation choices, recomputation after
state changes, and measured stage timings. Define consistent log levels and
message fields, keep default library operation quiet, and avoid logging large
arrays or one message per inner-loop evaluation.

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
`dev/slepian-old/threepcf_slepian_weber_interpolation.ipynb`; notebooks
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

## User tutorials

**Status:** Not started. Begin after the public API and route-selection
contracts are sufficiently stable.

The codebase has grown beyond what can reasonably be learned from class names
and docstrings alone. Add a small, ordered tutorial set for users. These are
supported user documents and belong under `tutorials/`, unlike exploratory
notebooks under ignored `dev/` directories.

The first tutorial must be a minimal quick start that works through the full
public workflow

```text
Bispectrum3D + LOSProjector -> Bispectrum2D -> ThreePCF
```

without exposing internal calculators or cache objects. It should use one
maintained model, one concise LOS setup, one route, and one final 3PCF plot.
Optional alternatives must not obscure the shortest working path.

Follow the quick start with focused tutorials covering:

- bispectra: native 3D and 2D objects, terms and representations, model
  architecture, composition, projection provenance, and relevant config;
- LOS projection: `Kernel1D`, `KernelSet`, exact delta-like evaluation,
  finite-width kernels, lensing and intrinsic-alignment kernels, LOS grids,
  weights, and projector architecture;
- 3PCF calculation: numeric, Slepian, semi-analytic, and future hybrid routes,
  route-independent result tables, theta/phi grids, spin/epsilon conventions,
  config, and cache behavior.

Tutorial markdown is Japanese only when explicitly intended as a private
development aid; supported package tutorials and their code/output labels must
use the documentation project's chosen public language consistently. Every
tutorial must run in CI against the current public API. Completion requires a
clean-environment execution test and no imports from `dev/` or legacy code.

## Generated documentation and Read the Docs

**Status:** Not started. Depends on the public-docstring and tutorial passes.

Build a maintained documentation site, hosted through Read the Docs, from a
single source tree. Generate API reference pages from public docstrings and
include the supported tutorials in the same navigation. Prefer a standard
Python documentation stack with reproducible pinned build dependencies; do
not copy docstrings manually into separate pages.

The documentation build must:

- distinguish normative architecture, user tutorials, API reference, and
  historical development notes;
- omit `dev/`, `legacy`, private helpers, and unsupported experimental APIs;
- execute or otherwise verify tutorial code during CI;
- fail on broken internal links, missing public API pages, and documentation
  build warnings that indicate invalid references;
- build locally with the same command and dependency set used by Read the
  Docs.

## Physical models and representation coverage

**Status:** Not started systematically. Numeric representations exist broadly;
Slepian and semi-analytic coverage remains model- and term-dependent.

Add alternative representations to the maintained models term by term. Each
term must keep its numeric representation as the universal reference and may
add Slepian or semi-analytic information only when the mathematical expression
is complete and independently validated. Shared Weber, Mellin, coupling, and
transform caches belong to route calculators rather than model terms.

For every added representation, test the representation against the numeric
form at the bispectrum level and compare its final 3PCF contribution through
an independent route. Record unsupported terms explicitly instead of silently
approximating or dropping them. Start with one simple SPT matter term before
expanding to permutations, bias terms, one-halo, or fitted models.

## Term-wise hybrid route planning

**Status:** Not implemented. `ThreePCF` currently selects one route for the
whole bispectrum; projected representations can coexist on terms, but there is
no completed term-wise hybrid planner and assembler.

Implement an explicit route-selection policy for bispectra containing multiple
terms with different available representations. Under a `hybrid` policy, each
term should use the highest-priority supported route selected by configuration.
For example, when three terms support only numeric, Slepian, and semi-analytic
evaluation respectively, the assembled result must evaluate those terms with
their corresponding routes and add their contributions into one shared
`ZetaKTable`. A term without its preferred representation must follow an
explicit fallback order, ultimately reaching numeric when a numeric
representation exists.

The planner must operate on terms, not on the bispectrum as an indivisible
object. It must prevent double counting when one term exposes several
representations, preserve term coefficients and projection provenance, and
raise an informative error when no allowed route can evaluate a term. Route
labels must not enter physical result keys; contributions from different
routes with the same component/mode key must be summed.

Before implementation, define:

- configurable route priority and whether fallback is automatic or strict;
- the capability query used to determine whether a projected or native term
  supports a route;
- grouping rules for terms that must be evaluated together;
- cache ownership and invalidation when a term changes representation or
  selected route;
- diagnostics exposing the selected route for each term without making route
  identity part of the result.

Completion requires mixed-term tests for all representation combinations,
permutation-invariant assembly, equality with manually summed single-route
calculations, explicit missing-capability failures, and preservation of
source-independent caches across model updates.
