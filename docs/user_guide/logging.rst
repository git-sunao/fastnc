Logging
=======

fastnc is quiet by default. Notebook and script users can enable progress
messages with:

.. code-block:: python

   import fastnc

   fastnc.configure_logging("INFO")

``INFO`` reports major calculation stages and elapsed times. ``DEBUG`` adds
route planning, cache reuse, grid sizes, mode counts, and numerical-kernel
details. Disable the convenience handler with:

.. code-block:: python

   fastnc.disable_logging()

Applications with an existing logging configuration may configure the
``fastnc`` logger hierarchy directly.

Logging is diagnostic and is not part of cache identity. Switching between
``INFO`` and ``DEBUG`` never invalidates a calculation. Typical ``INFO``
messages describe HKernel, ZetaK, and Zeta construction with elapsed wall
time. ``DEBUG`` is useful when checking hybrid term assignment, unique Slepian
matrix counts, low-rank retention, or coupling-cache reuse.

Timing from a cold call includes construction of missing in-memory resources.
A warm call represents repeated model evaluation only when the changed model
state leaves those resources reusable.
