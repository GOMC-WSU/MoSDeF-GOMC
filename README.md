[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/GOMC-WSU/MoSDeF-GOMC/master.svg)](https://results.pre-commit.ci/latest/github/GOMC-WSU/MoSDeF-GOMC/master)
[![CI](https://github.com/GOMC-WSU/MoSDeF-GOMC/actions/workflows/CI.yml/badge.svg)](https://github.com/GOMC-WSU/MoSDeF-GOMC/actions/workflows/CI.yml)
##  MoSDeF-GOMC:

### A User-Friendly Python Interface for Creating All of the files required to run a GPU Optimized Monte Carlo (GOMC) simulation engine via the Molecular Simulation Design Framework  [Molecular Simulation Design Framework (MoSDeF)](https://mosdef.org) ([MoSDeF Github](https://github.com/mosdef-hub)) software.
--------

### Overview

This Python code allows the auto-generation of the [GPU Optimized Monte Carlo (GOMC)](http://gomc.eng.wayne.edu) files for a simulation, which includes the coordinate (PDB), topology (PSF), force field (FF), and the GOMC control file.  This software supports various systems, force field types, and can also create the PDB, PSF, and FF files for the  [NAMD](https://www.ks.uiuc.edu/Research/namd/) simulation engine.  Since MoSDeF-GOMC was built from the  [Molecular Simulation Design Framework (MoSDeF)](https://mosdef.org) ([MoSDeF Github](https://github.com/mosdef-hub)) platform, it provides complete integration with the MoSDeF software.

### Warning
MoSDeF-GOMC is a new product continually adding functionality to provide an optimal user experience. Therefore, there could be some user-noticeable changes to this software when upgrading newer MoSDeF-GOMC versions.

The original version of MoSDeF-GOMC, which uses Parmed as the software backend, will be deprecated by December 2022.  This Parmed version is already replaced with the new MoSDeF-GOMC version, which uses MoSDeF's GMSO software as the new backend.  We recommend that the new **GMSO** version of MoSDeF-GOMC be used because it has many new features, and the Parmed is no longer supported.

### Resources
 - [GOMC Github repository](https://github.com/GOMC-WSU)
 - [Downloading GOMC](https://github.com/GOMC-WSU/GOMC)
 - [Installing GOMC via GOMC manual](https://github.com/GOMC-WSU/Manual)
 - [MoSDeF-GOMC tutorials and examples](https://github.com/GOMC-WSU/GOMC_Examples/tree/main/MoSDef-GOMC) with [MoSDeF-GOMC YouTube videos](https://www.youtube.com/watch?v=7StVoUCGkHs&list=PLdxD0z6HRx8Y9VhwcODxAHNQBBJDRvxMf)
 - [GOMC YouTube channel](https://www.youtube.com/channel/UCueLGE6tuOyu-mvxIt-U1HQ/playlists)
 - [MoSDeF tools](https://mosdef.org/)

### Citation

Please cite MoSDeF-GOMC, GOMC, and MoSDeF tools, which are provided [here](https://mosdef-gomc.readthedocs.io/en/latest/reference/citing_mosdef_gomc_python.html).

### Documentation

The MoSDeF-GOMC documentation is located [here](https://mosdef-gomc.readthedocs.io/en/latest/index.html).

### Installation

The MoSDeF-GOMC package is available via conda:

`conda create --name mosdef_gomc -c conda-forge mosdef-gomc`
