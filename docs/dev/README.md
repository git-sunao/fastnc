# Documentation policy

The repository documentation has three layers with different authority.

1. `dev/design/` contains the normative architecture and numerical contracts that
   active code must follow. These files describe the current decision, not its
   discussion history.
2. `dev/notes/` contains non-normative development context for future work:
   derivations, alternatives, migration history, experiments, and reasons for
   decisions. Notes may be longer and may describe superseded ideas. If a note
   conflicts with a design document, the design document wins.
3. `dev/todo.md` records deferred issues with their observed impact, proposed next
   step, and completion criteria.

Local notebooks, helper scripts, data, TeX sources, and human-oriented PDF
reports belong under `dev/` and are excluded from Git. Distill durable results
from those artifacts into `dev/notes/`, and promote accepted contracts into
`dev/design/`.
