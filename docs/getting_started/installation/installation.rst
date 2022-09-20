============
Installation
============
.. image:: https://img.shields.io/badge/license-MIT-blue.svg
    :target: http://opensource.org/licenses/MIT

Install with `conda <https://repo.anaconda.com/miniconda/>`_
------------------------------------------------------------
::

    $ conda install -c conda-forge mosdef_gomc


Install an editable version from the source code
------------------------------------------------

It is good practice to use a pre-packaged Python distribution like
`Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_,
which ensures all of the dependencies are installed::

    $ git clone https://github.com/GOMC-WSU/MoSDeF-GOMC
    $ cd mosdef_gomc
    $ conda env create -f environment-dev.yml
    $ conda activate mosdef_gomc_dev
    $ pip install -e .

.. note::
    The above installation instructions are for OSX and Unix.  If you are using Windows, please use a virtual machine or the Linux subsystem, as some components of this software and its dependencies may not be fully compatible with Windows.


Install pre-commit
------------------

The `pre-commit <https://pre-commit.com/>`_ packages are utilized to maintain uniform code formatting, which is auto-installed in the dev environment.
Alternatively, pre-commit can be installed inside the active ``mosdef_gomc`` conda environment via the following commands::

     $ conda install -c conda-forge pre-commit

To check all the file, you can run::

     $ pre-commit run --all-files


Supported Python Versions
-------------------------

Python 3.9 are officially supported and tested during development and with the final product.
Python versions older than 3.9 may work, but there is no guarantee.

Testing your installation
-------------------------

MoSDeF-GOMC uses `pytest <https://docs.pytest.org/en/stable/>`_ to test the code for accuracy and possible errors.
The pytest package is installed for testing code changes or if the existing implementation is correct, which is auto-installed in the dev environment (``mosdef_gomc_dev``).
Alternatively, pytest can be installed inside the active ``mosdef_gomc`` conda environment via the following commands::

    $ conda install -c conda-forge pytest

To run these unit tests, run the following from the base directory::

    $ pytest -v

Building the documentation
--------------------------

MoSDeF-GOMC documentation is all built using `sphinx <https://www.sphinx-doc.org/en/master/index.html>`_.
After installing the software via the **Install an editable version from the source code** (``mosdef_gomc_dev`` environment), the ``docs`` can be built locally with the following commands when in the ``docs`` directory::

    $ conda activate mosdef_gomc_dev
    $ make html
