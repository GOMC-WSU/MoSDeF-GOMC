.. _DataStructures:

===============
Data Structures
===============
.. image:: https://img.shields.io/badge/license-MIT-blue.svg
    :target: http://opensource.org/licenses/MIT


CHARMM-style PDB, PSF, and Force Field File Writers
---------------------------------------------------

	.. autoclass:: mosdef_gomc.formats.gmso_charmm_writer.Charmm
		:special-members: __init__
		:members:


GOMC Control File Writer
------------------------

	.. automodule:: mosdef_gomc.formats.gmso_gomc_conf_writer
    		:members: write_gomc_control_file, print_required_input





NAMD Control File Writer
------------------------

The NAMD control file writer is not currently available.



MoSDeF software functions
-------------------------

The **MosDeF-GOMC** was built on the **MosDeF** ecosystem.  Therefore, please see the **MoSDeF** software documentation for details on the respective packages:
    	* `mBuild <https://mbuild.mosdef.org/en/stable/>`_ -- A hierarchical, component based molecule builder

    	* `foyer <https://foyer.mosdef.org/en/stable/>`_ -- A package for atom-typing as well as applying and disseminating forcefields

    	* `GMSO <https://gmso.mosdef.org/en/stable/>`_ -- Flexible storage of chemical topology for molecular simulation
