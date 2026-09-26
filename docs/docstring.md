Python docstring rules:

Scope
- Add docstrings only to functions, methods, and classes that form part of the user-facing interface.
- Public helper functions do not need docstrings merely because they are technically public.
- Do not add docstrings to internal implementation details unless their behavior is unusually subtle and requires explanation.
- Functions or methods whose names start with `_` should normally have no docstring.
- Do not add trivial docstrings solely for documentation coverage.
- When modifying existing code, preserve useful existing docstrings even for internal functions unless there is a reason to remove them.

Style
- Use compact Google-style docstrings.
- Keep docstrings concise and avoid unnecessary vertical length.
- The first line must give a one-sentence summary of what the interface does.
- Use type hints in signatures; never repeat types in docstrings.
- Use only sections that add useful information: Args, Returns, Raises, Notes.
- Omit Args or Returns when they would merely restate the signature.
- Group closely related arguments on one line when they share meaning, units, or conventions.
- Keep individual descriptions on one line when reasonably possible.
- Do not add Examples, See Also, or similar sections unless genuinely useful.
- Prefer physical/mathematical meaning, units, normalization, conventions,
  assumptions, and non-obvious behavior over implementation details.

Mathematical interfaces
- Major scientific or numerical interfaces should explain the mathematical
  quantity or operation being implemented.
- Include the defining equation when it substantially clarifies the interface.
- Define symbols appearing in the equation when their meaning is not obvious
  from the argument descriptions.
- State important normalization conventions, coordinate conventions, units,
  approximations, and domain restrictions.
- Explain the mathematical contract of the interface, not the internal
  numerical implementation.
- Keep mathematical explanations compact; do not turn docstrings into
  derivations or documentation pages.
- Use standard mathematical notation supported by the project's documentation
  system. If no documentation renderer is assumed, prefer readable plain-text
  mathematics.

General principle
- Docstrings document interfaces, not every function in the implementation.
- The goal is not exhaustive API coverage. A researcher reading the interface
  should be able to understand what mathematical/physical quantity is computed,
  what conventions are used, and what the inputs and outputs mean without
  reading the implementation.