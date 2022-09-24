
General Information
===================
.. image:: https://img.shields.io/badge/license-MIT-blue.svg
    :target: http://opensource.org/licenses/MIT


MoSDeF-GOMC Basics
------------------
The `Molecular Simulation Design Framework (MoSDeF) <https://mosdef.org>`_
software is the base software for **MoSDeF-GOMC**, which builds and force fields the systems inside the **MoSDeF-GOMC**.
This **MoSDeF-GOMC** software generates the required files for simulating a system using
the `GPU Optimized Monte Carlo (GOMC) <https://github.com/GOMC-WSU/GOMC>`_ or the
`NAMD <https://www.ks.uiuc.edu/Research/namd/>`_ molecular dynamics software.  The **MoSDeF-GOMC** software
builds all the required files (PDB, PSF, force field, and **GOMC** control files) to conduct a **GOMC** simulation.

The **MoSDeF-GOMC** software is compatible with the following software or simulation methods, when the files are generated with the Lennard-Jones (LJ) non-bonded potential:
	#. The `NAMD <https://www.ks.uiuc.edu/Research/namd/>`_ molecular dynamics software: The NAMD control file needs to be created by another means, but all the other files (PDB, PSF, and force field files) are compatible.

	#. The `py-MCMD <https://py-mcmd.readthedocs.io/en/latest/>`_ software (Hybrid Monte Carlo and molecular dynamics simulations via **GOMC** and **NAMD**): All files are compatible except the GOMC control file (PDB, PSF, and force field files) are compatible.

	#.  Switching back and forth, in any order or duration, between a **GOMC** and **NAMD** simulations, allowing faster equilibrium for **GOMC** simulations and many other advantages. The newest **GOMC** versions also retain the atom/bead velocities and pass them back to **NAMD**.


The **MoSDeF-GOMC** software creates all these files using only tens of lines of Python code.
The **MoSDeF-GOMC** and **MoSDeF** tools permit simulation reproducibility across a variety of simulation engines,
removing the requirement of expert knowledge in all the engines to repeat, continue, or advance the existing research.
Additionally, the **MoSDeF-GOMC** and **MoSDeF** software permits the auto-generation of numerous and distinct
systems, allowing large-scale screening of materials and chemicals via `signac <https://signac.io>`_ to manage
the simulations and data.

.. note::
	The using **MoSDeF-GOMC** software to setup the **GOMC** and **NAMD** simulations,
	is made even easier via the
	`MoSDeF-GOMC Examples <https://github.com/GOMC-WSU/GOMC_Examples/tree/main/MoSDef-GOMC>`_ and the
	`GOMC Documentation <http://gomc.eng.wayne.edu/documentation/>`_, which contains links to the GOMC Manual, with
	`GOMC YouTube tutorial videos <https://youtube.com/playlist?list=PLdxD0z6HRx8Y9VhwcODxAHNQBBJDRvxMf>`_.
	For the Lennard-Jones (LJ) non-bonded interactions, the MoSDeF-GOMC's PDB, PSF, and force field files are
	identical to the NAMD files for the traditional chemical engineering simulations, unless there are fixed bonds
	and angles in the GOMC force field files.  Changing the fixed bonds and angles between the GOMC and NAMD
	force field files is as simple as changing one (1) variable and rerunning that line of code,
	making a separate GOMC and NAMD files for the same system.  Making separate files may be needed when
	running a hybrid Monte Carlo and molecular dynamics (GOMC and NAMD) simulation with the
	`py-MCMD <https://py-mcmd.readthedocs.io/en/latest/>`_ software .

MoSDeF-GOMC is a part of the MoSDeF ecosystem
---------------------------------------------
The **MoSDeF-GOMC** software is a liaison between the
`Molecular Simulation Design Framework (MoSDeF) <https://mosdef.org>`_ software suite and the
**GOMC** simulation engine.
The MoSDeF libraries also supports various simulation engines,
including `Cassandra <https://cassandra.nd.edu>`_,
`GOMC <https://github.com/GOMC-WSU/GOMC>`_,
`GROMACS <https://www.gromacs.org>`_,
`HOOMD-blue <http://glotzerlab.engin.umich.edu/hoomd-blue/>`_, and
`LAMMPS <https://lammps.sandia.gov>`_.
The **MoSDeF-GOMC** and **MoSDeF** libraries permit reproducibility simulations
across a wide range of simulation engines, eliminating the need to be an expert in every simulation engine
to replicate or expand upon the existing research. These software packages can auto-produce a variety of different chemical systems,
permitting large-scale screening of chemicals and materials using `signac <https://signac.io>`_ to manage the simulations, data, and analysis.

The **MoSDeF** software is comprised the following core packages:
	* `mBuild <https://mbuild.mosdef.org/en/stable/>`_ -- A hierarchical, component based molecule builder

	* `foyer <https://foyer.mosdef.org/en/stable/>`_ -- A package for atom-typing as well as applying and disseminating forcefields

	* `GMSO <https://gmso.mosdef.org/en/stable/>`_ -- Flexible storage of chemical topology for molecular simulation
