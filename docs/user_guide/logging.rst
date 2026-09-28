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
