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
