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
4. Never add or commit Jupyter notebooks to Git. Development notebooks remain
   local, untracked files.

Changes limited to documentation, tests, notebooks, Git configuration, or
these development rules do not by themselves require a version increment or a
new validation notebook.

## Architecture

Before modifying bispectrum, projection, route, multipole, interpolation, LOS,
or Grid code, read `docs/development/bispectrum-representations.md` and follow
the architecture and migration decisions recorded there.

Treat that document as the authoritative source for this refactor; do not
duplicate its detailed design in `AGENTS.md`. If an implementation requires a
change to an agreed architectural decision, update the design document first
and explain the proposed change to the maintainer before changing the
architecture.

The directory `legacy/bispectrum_object_api/` is a non-importable reference
archive for the retired bispectrum object graph and the unfinished,
unvalidated semi-analytic implementation. Active code must never import from
that directory. Recover a useful numerical component only by extracting it
into the active architecture and adding an independent test for its contract.

The directory `legacy/threepcf_grid_pipeline/` is also a non-importable
reference archive. It contains the retired 3PCF prototype in which Grid
objects performed calculations and managed route state. Active code must not
import from it. Keep active `fastnc/threepcf` organized into shared conventions
and explicit route modules; new result/Grid types must be passive.

The directory `legacy/compat/` contains the retired adapters for the archived
object APIs. It is reference material only; active code must not import it or
use it to preserve compatibility with the retired architecture.

Line-of-sight kernels and geometry belong in `fastnc/projection`.
Route-independent multipole products and producers belong in
`fastnc/multipole`. Algorithms that map angular representations to 3PCF
products belong in the flat `fastnc/threepcf/{numeric,slepian,semi_analytic}`
modules. Keep route-independent projection code out of route modules, and keep
physical bispectrum models out of both projection and route modules.
