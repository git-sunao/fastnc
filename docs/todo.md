# fastnc development TODO

Status: Tracked deferred work. These items are not normative architecture;
accepted contracts belong in `docs/design/` and supporting evidence belongs in
`docs/notes/`.

This file records deferred work that is specific enough to affect future
implementation decisions. It is not a general wishlist. Keep completed items
in the relevant design document or Git history rather than accumulating them
here.

## Structured calculation logging

**Status:** Initial progress logging is implemented. `INFO` reports major
ThreePCF stages and elapsed times; `DEBUG` reports route planning, cache reuse,
grid sizes, modes, and calculator details. Further route coverage and stable
machine-readable fields remain pending.

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

## Calculation-graph visualization

**Status:** Deferred.

Add a `ThreePCF` inspection API that displays the planned calculation graph
before expensive evaluation. The graph must show how every bispectrum term is
assigned to numeric, Slepian, or semi-analytic processing, group terms that
share the same route, and show the shared downstream products such as
BispectrumMultipole, HKernel, ZetaK, and Zeta. It should be useful as both a
text representation and an inline notebook visualization, without executing
the numerical graph merely to inspect it. The visualization must be derived
from the same planner data used for execution so that documentation and actual
route selection cannot diverge.

## Weber evaluator stability at large imaginary Mellin index

**Status:** Partially resolved. The direct and interpolated evaluators now use
the `z = 1` connection formula above a Mellin-index-dependent ratio boundary,
while the central eta-zero mode uses an exact finite polynomial when the
canonical Bessel-order difference is even.

For large `abs(Im(exponent))`, the direct evaluator's power series for
`hyp2f1(A, B; C; r**2)` becomes inaccurate as `r` approaches one. For
`exponent = -0.8 + 40j` and Bessel orders `(0, 0)`, comparison against an
80-digit `mpmath` reference showed relative errors of approximately
`6e-6` at `r=0.6`, `2e-4` at `r=0.7`, `1.7e-1` at `r=0.8`, and complete
failure beyond that. The jagged behavior seen in the validation notebook is
therefore artificial and is inherited by interpolation tables constructed
from the direct evaluator.

The implemented switching point balances the ordinary and reflected series
variables at low imaginary index and moves down to a floor of `ratio = 0.25`
as the index grows. Pointwise tests against 80-digit values cover imaginary
Mellin indices through 100, Bessel orders through 30, and ratios through
0.9999. The analytic Gaussian end-to-end benchmark at `n_ell = 256` improved
from approximately `4e-2` to `3e-6` in ZetaK scaled error. The remaining work
includes other complete FFTLog grids and terms, and real integer connection
exponents outside the eta-zero branch.

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

## Mathematical implementation docstrings

**Status:** Deferred until the current numerical kernels and their interfaces
stabilize. This is related to, but distinct from, the public API docstring
pass.

Add concise mathematical docstrings to major kernels and to functions whose
implementation evaluates a transformed, decomposed, or stabilized expression
rather than the most obvious defining formula. These docstrings are for
advanced users and developers who need to connect the code to the actual
algorithm. Do not add equations mechanically to every helper.

For each selected function, state as applicable:

- the original mathematical quantity that the function contributes to;
- the equivalent or approximate expression actually evaluated;
- the intermediate pieces computed separately and the equation used to
  assemble them;
- branch, support, parity, index, and normalization conventions needed to map
  arguments to the formula;
- numerical switching conditions and why the direct expression is avoided;
- approximation parameters, expected error, and unsupported limiting cases;
- the next upstream and downstream mathematical objects when the function is
  one stage of a longer contraction.

Keep the equations local and concise. Long derivations, experiment history,
and benchmark plots belong in `docs/notes/` or development notebooks, while
the docstring should contain enough notation to understand the implementation
without reverse-engineering the function body.

Prioritize FFTLog coefficient construction, single and double radial
transforms, ordinary/reflected/eta-zero Weber evaluation, contact and regular
kernel decomposition, full and low-rank `F_ab` construction and contraction,
coefficient-level LOS integration, multipole coupling, Hankel transforms, and
the final `ZetaK`/`Zeta` assembly. Extend the inventory to similarly
non-obvious numeric, projection, and interpolation kernels before declaring
the pass complete.

Completion requires a reviewed inventory of such kernels, consistent notation
with the design documents, and tests or documentation checks that at least
guard the presence of the required mathematical sections without asserting
their prose verbatim.

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

**Status:** Started with the projected 3D tree-level SPT `F2` model. Numeric
representations exist broadly; Slepian and semi-analytic coverage remains
model- and term-dependent.

Add alternative representations to the maintained models term by term. Each
term must keep its numeric representation as the universal reference and may
add Slepian or semi-analytic information only when the mathematical expression
is complete and independently validated. Shared Weber, Mellin, coupling, and
transform caches belong to route calculators rather than model terms.

For every added representation, test the representation against the numeric
form at the bispectrum level and compare its final 3PCF contribution through
an independent route. Record unsupported terms explicitly instead of silently
approximating or dropping them. The supported `F2` pairs `12` and `13` now
cover their complete finite harmonic content `m=0,+/-1,+/-2`; pair `23` is
explicitly excluded because it would make physical leg 1 constant. Continue
with the hybrid fallback for pair `23`, then one-halo terms and fitted models.

The finite decomposition is attached directly to the maintained 3D matter
and galaxy SPT models. A separate 2D SPT model accepting `C_ell` is not part of
the physical-model API: users obtain the angular bispectrum by projecting the
3D model. Galaxy quadratic-bias and tidal-bias terms for pairs 12 and 31 are
covered as well. The remaining SPT work is therefore physical finite-width
LOS validation and semi-analytic coverage, not additional constant-leg
representations for these models.

## Physical LOS benchmarks for the Slepian route

**Status:** Structural LOS support exists, but physical end-to-end validation
is incomplete.

Validate projected Slepian predictions with maintained physical kernels rather
than relying only on native-2D toys and exact delta-like projection. Cover at
least one finite-width source distribution, one lensing kernel constructed
from a source distribution, and one intrinsic-alignment kernel. Use the same
3D bispectrum term, cosmology, angular grid, redshift quadrature, spin,
epsilon, and final theta/phi bins in the Slepian and numeric projection paths.

The comparison must separately inspect projected Mellin coefficients or the
earliest common intermediate quantity, every retained `ZetaK` mode, and final
`Zeta`. Plot the compared functions as well as residuals. Vary the LOS
quadrature sufficiently to distinguish projection error from FFTLog, Weber,
and radial-contraction error. Test that changing only model parameters reuses
structural Weber and `F_ab` resources, while changing the projector's redshift
grid invalidates all coefficient-dependent LOS products.

Completion requires convergence against a stricter numeric-projection
benchmark for each kernel type, documented accuracy and runtime at a practical
configuration, and an explicit account of any term or kernel for which the
coefficient-level LOS path is not mathematically supported.

## Slepian numerical defaults and diagnostics

**Status:** Individual controls exist and several focused experiments have
validated them, but supported defaults and failure diagnostics are not yet
established as a coherent policy.

Determine practical defaults and convergence guidance jointly for the FFTLog
ell range and `n_ell`, real Mellin bias, taper and window fractions, direct
versus interpolated Weber evaluation, the Mellin-index-dependent reflection
boundary, diagonal correction, regular quadrature, and full versus low-rank
`F_ab` contraction. A larger grid must not be presented as automatically more
accurate; diagnostics must expose aliasing, insufficient dynamic range,
endpoint sensitivity, unstable Mellin modes, and low-rank truncation error.

Separate correctness controls from performance controls. Correctness controls
must have conservative defaults or produce a clear warning when the requested
configuration is outside validated ranges. Performance controls may trade
accuracy for speed only through explicit configuration. Avoid automatically
tuning against the target result in a way that changes the mathematical
prediction during parameter inference.

Completion requires convergence studies for analytic toys, at least one SPT
term, and the physical LOS benchmarks above; recommended configurations for a
quick calculation and a production calculation; machine-readable diagnostics
that can later feed the structured logging facility; and regression tests that
pin the accepted accuracy without depending on one accidental grid choice.

## Term-wise hybrid route planning

**Status:** Numeric/Slepian/semi-analytic hybrid assembly is implemented. The
generic calculator and `U`, `V`, `W`, `p` representation exist; physical model
coverage remains incomplete. SPT matter pair 23 is implemented and has
independent numeric accuracy and warm-speed validation.

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

The current `hybrid` policy assigns every Slepian-capable term to the direct
Slepian-to-ZetaK path. Remaining terms use coefficient-level semi-analytic
multipoles when available and numeric angular decomposition otherwise. The
calculators are separate, but both feed the shared HKernel path. The planner
sums route-independent ZetaK keys and caches only the assembled table.
