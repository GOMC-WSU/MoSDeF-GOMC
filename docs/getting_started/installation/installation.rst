============
Installation
============
.. image:: https://img.shields.io/badge/license-MIT-blue.svg
    :target: http://opensource.org/licenses/MIT

Install with `mamba <https://github.com/mamba-org/mamba>`_ (Recommended)
------------------------------------------------------------------------
::

    $ mamba install -c conda-forge mosdef-gomc


Install with `conda <https://repo.anaconda.com/miniconda/>`_
------------------------------------------------------------
::

    $ conda install -c conda-forge mosdef-gomc


NOTE: conda has had some issues pulling the most recent packages, so a mamba installation is recommended.

Install an editable version from the source code
------------------------------------------------

It is good practice to use a pre-packaged Python distribution like
`Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_,
which ensures all of the dependencies are installed::

    $ git clone https://github.com/GOMC-WSU/MoSDeF-GOMC
    $ cd mosdef_gomc
    $ conda env create -f environment.yml
    $ conda activate mosdef_gomc
    $ pip install -e .

.. note::
    If `pip install -e .` is not run, then then finding some of the packages files will not work.  Hence, the unit-tests will not pass.


.. note::
    The above installation instructions are for OSX and Unix.  If you are using Windows, please use a virtual machine or the Linux subsystem, as some components of this software and its dependencies may not be fully compatible with Windows.


Install pre-commit
------------------

The `pre-commit <https://pre-commit.com/>`_ packages are utilized to maintain uniform code formatting, which is auto-installed in the mosdef_gomc environment.

To check all the file, you can run::

     $ pre-commit run --all-files


Supported Python Versions
-------------------------

Python 3.12 and 3.13 are officially supported and tested during development and with the final product.
Python versions older than 3.12 may work, but there is no guarantee.

Testing your installation
-------------------------

MoSDeF-GOMC uses `pytest <https://docs.pytest.org/en/stable/>`_ to test the code for accuracy, possible errors, code changes, or if the existing implementation is correct.
The pytest package is auto-installed in the ``mosdef_gomc`` environment.

To run these unit tests, run the following from the base directory::

    $ pytest -v

Building the documentation
--------------------------

``MoSDeF-GOMC`` documentation is all built using `sphinx <https://www.sphinx-doc.org/en/master/index.html>`_.
The ``sphinx`` software may need to be installed separately to avoid dependency conflicts.
If ``sphinx`` is not automatically provided, the correct ``sphinx`` package can be build after creating
a new conda environment using the ``environment_docs.yml`` file in the ``MoSDeF-GOMC/docs``
directory, located on ``MoSDeF-GOMC`` GitHub's main repository or GitHub's releases for a specific version.

The ``docs`` can be built locally with the following commands when in the ``docs`` directory::

    $ make html
