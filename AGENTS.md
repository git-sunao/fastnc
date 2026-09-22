# fastnc development rules

These rules apply to all future development in this repository.

1. Increment the package version in `fastnc/__init__.py` whenever source code
   is changed. Use a patch increment for ordinary development changes unless a
   minor or major increment is explicitly requested.
2. After changing source code, report the principal files that were added or
   modified and briefly identify what changed in each file. This list is for
   the maintainer's review and must not be omitted.
3. For every new feature, create a minimal Jupyter notebook that exercises the
   feature. Keep the notebook as short as possible. If part of the feature
   cannot yet be exercised, leave a concise `#` comment in the relevant code
   cell explaining what remains.
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
