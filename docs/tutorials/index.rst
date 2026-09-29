Tutorials
=========

These tutorials build complete calculations with the supported public API.
They are ordered from a minimal end-to-end prediction to the individual model,
projection, and 3PCF layers.

.. toctree::
   :maxdepth: 1

   quick_start
   bispectrum
   los_projection
   threepcf_routes

The examples use ``simple_debug`` power spectra and compact numerical grids so
that the workflow is reproducible without an external Boltzmann solver. These
choices are suitable for learning and software checks, not for scientific
parameter inference. Production calculations should supply a validated
cosmology and demonstrate convergence in every numerical grid and truncation.
