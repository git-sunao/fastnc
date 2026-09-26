# fastnc development rules

These rules apply to all future development in this repository.

1. Increment the package version in `fastnc/__init__.py` whenever source code
   is changed. Use a patch increment for ordinary development changes unless a
   minor or major increment is explicitly requested.
2. After changing source code, report the principal files that were added or
   modified and briefly identify what changed in each file. This list is for
   the maintainer's review and must not be omitted.
3. For every new feature, create a focused Jupyter notebook that exercises the
   feature. Here, "minimal" means free of unrelated setup, exhaustive parameter
   scans, duplicated calculations, and tutorial boilerplate; it does not mean a
   one-cell smoke test. A notebook should normally contain enough short cells to
   show the intended API, inspect the important intermediate objects or arrays,
   and verify at least one defining identity or reference result. For composite
   features, also demonstrate the principal selection, composition, or alternate
   execution path needed to understand the design. Use concise markdown to state
   what each stage establishes. If part of the feature cannot yet be exercised,
   leave a concise `#` comment in the relevant code cell explaining what remains.
   Write development-notebook Markdown headings and explanations in Japanese.
   Inside code cells, use English for identifiers, string literals, plot titles,
   axis labels, legends, printed messages, and other displayed text so that
   rendering does not depend on Japanese fonts. Code comments may be Japanese.
   Keep public API names and established mathematical notation unchanged when
   that makes the code easier to follow.
   When comparing numerical methods, do not present only scalar residuals or a
   residual plot. Also plot the compared quantities themselves as functions of
   their arguments so that both the mathematical behavior and agreement are
   visible. Overlay methods for a one-dimensional function-versus-argument
   comparison. For a two-dimensional function, use three heatmap panels:
   benchmark, candidate, and residual. Use a shared symmetric color range with
   `vmax = max(abs(benchmark), abs(candidate))`, `vmin = -vmax`, and the `bwr`
   colormap. Put different indices or modes in separate figures rather than
   combining them into one crowded multipanel figure. For complex quantities,
   inspect the real and imaginary parts when both matter. Keep numerical residual summaries
   as supplementary diagnostics.
4. Never add or commit Jupyter notebooks to Git. Development notebooks remain
   local, untracked files.
5. Keep exploratory notebooks and calculation records under `dev/`, grouped by
   topic. Place notebooks, Python helpers, data, and final PDFs directly in the
   topic directory; only TeX sources belong in a `tex/` subdirectory. Reserve
   `tutorials/` for polished examples of the supported public API; never place
   development notebooks there. The entire `dev/` tree is local working state
   and must remain excluded from Git because notebooks, figures, data, and
   helper scripts change frequently. Preserve durable design decisions in
   `docs/design/`; promote reusable code and regression coverage into
   `fastnc/` and `tests/` rather than tracking selected files from `dev/`.

Changes limited to documentation, tests, notebooks, Git configuration, or
these development rules do not by themselves require a version increment or a
new validation notebook.

## Development environment

Run tests, validation scripts, and development notebooks with
`/Users/sugiyamasunao/miniforge3/envs/fastnc/bin/python`. Prefer the executable
directly so execution does not depend on shell activation.

## Architecture

Before modifying bispectrum, projection, route, multipole, interpolation, LOS,
or Grid code, read `docs/design/architecture.md` and follow its normative
contracts. Read `docs/notes/bispectrum-refactor.md` when the task requires
historical reasoning, migration context, rejected alternatives, or detailed
test plans.

Before modifying the Slepian route, Weber evaluation, constant-leg kernels, or
Mellin contractions, also read `docs/design/slepian.md`.
Follow its staged implementation order and preserve every established slower
method as a selectable benchmark for the next optimization. In particular, do
not combine Weber interpolation, full `F_ab` construction, and low-rank
compression in one implementation step.

Read `docs/notes/slepian-performance.md` when detailed derivations, experiment
history, or benchmark interpretation are needed. Notes are non-normative; when
they conflict with `docs/design/`, the design document wins.

Before planning or starting new implementation work, read `docs/todo.md` and
check whether the change resolves, depends on,
or must preserve a recorded deferred issue. Update the TODO when new evidence
changes the scope or acceptance criteria of an item.

Treat `docs/design/` as the authoritative source for this refactor; do not
duplicate its detailed design in `AGENTS.md`. If an implementation requires a
change to an agreed architectural decision, update the relevant design
document first and explain the proposed change to the maintainer before
changing the architecture.

The local directory `dev/legacy/fastnc-v2/bispectrum-object-api/` is a
non-importable reference
archive for the retired bispectrum object graph and the unfinished,
unvalidated semi-analytic implementation. Active code must never import from
that directory. Recover a useful numerical component only by extracting it
into the active architecture and adding an independent test for its contract.

The local directory `dev/legacy/fastnc-v2/threepcf-grid-pipeline/` is also a non-importable
reference archive. It contains the retired 3PCF prototype in which Grid
objects performed calculations and managed route state. Active code must not
import from it. Keep active `fastnc/threepcf` organized into shared conventions
and explicit route modules; new result/Grid types must be passive.

The local directory `dev/legacy/fastnc-v2/compat/` contains the retired adapters for the archived
object APIs. It is reference material only; active code must not import it or
use it to preserve compatibility with the retired architecture.

Line-of-sight kernels and geometry belong in `fastnc/projection`.
Route-independent multipole products and producers belong in
`fastnc/multipole`. Algorithms that map angular representations to 3PCF
products belong in the flat `fastnc/threepcf/{numeric,slepian,semi_analytic}`
modules. Keep route-independent projection code out of route modules, and keep
physical bispectrum models out of both projection and route modules.
