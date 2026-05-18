Installation
============

Development installation
------------------------

Clone the repository and install the package in editable mode:

.. code-block:: bash

   git clone https://github.com/agimenezromero/demovuln.git
   cd demovuln
   python -m pip install -e ".[dev,docs]"

Check the installation
----------------------

.. code-block:: bash

   python -c "import demovuln; print(demovuln.__version__)"
