=====
Units
=====
.. image:: https://img.shields.io/badge/license-MIT-blue.svg
    :target: http://opensource.org/licenses/MIT

**MoSDeF-GOMC** uses the `unyt <https://unyt.readthedocs.io/en/stable/>`_ package for all its unit systems, except if they are unitless values.

.. note::
    The **MoSDeF-GOMC** uses the `Molecular Simulation Design Framework (MoSDeF) <https://mosdef.org>`_ software. Therefore, for all the compatible unit systems, please see the following packages:

    * `mBuild <https://mbuild.mosdef.org/en/stable/>`_ -- A hierarchical, component based molecule builder

    * `foyer <https://foyer.mosdef.org/en/stable/>`_ -- A package for atom-typing as well as applying and disseminating forcefields

    * `GMSO <https://gmso.mosdef.org/en/stable/>`_ -- Flexible storage of chemical topology for molecular simulation


.. note::
    The **GOMC** and **NAMD** file units can be found in the :ref:`DataStructures` section for the PDB, PSF, and force field files, and the `GOMC input <https://gomc-wsu.github.io/Manual/input_file.html>`_ and `GOMC output <https://gomc-wsu.github.io/Manual/output_file.html>`_ (**GOMC** control file writer).  **NAMD's** file units are the same as **GOMC's** when using the non-bonded Lennard-Jones (LJ) interactions.
