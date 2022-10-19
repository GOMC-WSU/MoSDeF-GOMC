MoSDeF-GOMC: Python software for the creation of scientific workflows for the Monte Carlo simulation engine GOMC
================================================================================================================

.. image:: https://img.shields.io/badge/license-MIT-blue.svg
    :target: http://opensource.org/licenses/MIT


The `GPU Optimized Monte Carlo (GOMC) <http://gomc.eng.wayne.edu>`_ simulation engine is one of the fastest open-source Monte Carlo engines.  `Molecular Simulation Design Framework (MoSDeF) <https://mosdef.org>`_ software is an open-source software package that efficiently constructs simple or complex chemical systems, applies the force field parameters, and writes several simulation input files.
**MoSDeF-GOMC** uses the power of **MoSDeF** to build the traditional chemical engineering system, while **MoSDeF-GOMC**
generates all the required files to perform a **GOMC** simulation.

The **MoSDeF-GOMC** software is compatible with the following software or simulation methods, when the files are generated with the Lennard-Jones (LJ) non-bonded potential:
	#. The `NAMD <https://www.ks.uiuc.edu/Research/namd/>`_ molecular dynamics software: The NAMD control file needs to be created by another means, but all the other files (PDB, PSF, and force field files) are compatible.

	#. The `py-MCMD <https://py-mcmd.readthedocs.io/en/latest/>`_ software (Hybrid Monte Carlo and molecular dynamics simulations via **GOMC** and **NAMD**): All files are compatible except the GOMC control file (PDB, PSF, and force field files) are compatible.

	#.  Switching back and forth, in any order or duration, between a **GOMC** and **NAMD** simulations, allowing faster equilibrium for **GOMC** simulations and many other advantages. The newest **GOMC** versions also retain the atom/bead velocities and pass them back to **NAMD**.


**MoSDeF-GOMC Highlights**:
	#. With tens of lines of **MoSDeF-GOMC** code, you can generate all the required files to conduct a **GOMC** simulation (see `Simple MoSDeF-GOMC examples <https://github.com/GOMC-WSU/GOMC_Examples/tree/main/MoSDef-GOMC>`_).

	#. **MoSDeF-GOMC** is designed to automate the simulation workflow, allowing high-throughput workflows with **GOMC** or **NAMD**, where `signac <https://signac.io>`_ can be utilized to manage the simulations, data storage, and analysis.

	#. **MoSDeF-GOMC** lowers the barrier of entry for novice users.

	#. **MoSDeF-GOMC** and **MoSDeF** permit reproducible simulations, as these automated simulation workflows can build, simulate, and analyze the data in a repeatable manner. This allows the simulation workflows to be easily transferred, replicated, or expanded upon within or outside existing teams.



.. toctree::
	:caption: Overview
    	:maxdepth: 2

	overview/general_info
	overview/supported_chemical_systems

.. toctree::
	:caption: Getting Started
    	:maxdepth: 2

	getting_started/installation/installation
    	getting_started/quick_start/quick_start

.. toctree::
	:caption: Topic Guides
    	:maxdepth: 2

    	topic_guides/data_structures
	topic_guides/load_data

.. toctree::
    	:caption: Reference
    	:maxdepth: 2

	reference/units
	reference/user_notices
	reference/credits
    	reference/citing_mosdef_gomc_python
