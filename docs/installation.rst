Installation
============

Development installation
------------------------

Clone the repository and install it into a dedicated environment:

.. code-block:: console

   python -m pip install -e .

The core dependencies are NumPy, SciPy, Astropy, pandas, mpi4py, and h5py.
For documentation development, install the additional requirements and build
the HTML pages with Sphinx:

.. code-block:: console

   python -m pip install -r docs/requirements.txt
   python -m sphinx -W --keep-going -b html docs docs/_build/html

Serve the generated pages over HTTP for reliable local viewing in all
browsers:

.. code-block:: console

   python -m http.server 8000 --directory docs/_build/html

Then open ``http://localhost:8000``. Direct ``file://`` access may prevent
Chrome from loading theme assets even when the build itself is valid.
