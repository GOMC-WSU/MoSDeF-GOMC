[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/GOMC-WSU/MoSDeF-GOMC/master.svg)](https://results.pre-commit.ci/latest/github/GOMC-WSU/MoSDeF-GOMC/master)
[![CI](https://github.com/GOMC-WSU/MoSDeF-GOMC/actions/workflows/CI.yml/badge.svg)](https://github.com/GOMC-WSU/MoSDeF-GOMC/actions/workflows/CI.yml)
##  MoSDeF-GOMC:

### A User-Friendly Python Interface for Creating All of the files required to run a GPU Optimized Monte Carlo (GOMC) simulation engine via the Molecular Simulation Design Framework  [Molecular Simulation Design Framework (MoSDeF)](https://mosdef.org) ([MoSDeF Github](https://github.com/mosdef-hub)) software.
--------

### Overview

This Python code allows the auto-generation of the [GPU Optimized Monte Carlo (GOMC)](http://gomc.eng.wayne.edu) files for a simulation, which includes the coordinate (PDB), topology (PSF), force field (FF), and the GOMC control file.  This software supports various systems, force field types, and can also create the PDB, PSF, and FF files for the  [NAMD](https://www.ks.uiuc.edu/Research/namd/) simulation engine.  Since MoSDeF-GOMC was built from the  [Molecular Simulation Design Framework (MoSDeF)](https://mosdef.org) ([MoSDeF Github](https://github.com/mosdef-hub)) platform, it provides complete integration with the MoSDeF software.

### Warning
MoSDeF-GOMC is a new product continually adding functionality to provide an optimal user experience. Therefore, there could be some user-noticeable changes to this software when upgrading newer MoSDeF-GOMC versions.

### Resources
 - [GOMC Github repository](https://github.com/GOMC-WSU)
 - [Downloading GOMC](https://github.com/GOMC-WSU/GOMC)
 - [Installing GOMC via GOMC manual](https://github.com/GOMC-WSU/GOMC/blob/main/GOMC_Manual.pdf)
 - [MoSDeF-GOMC tutorials and examples](https://github.com/GOMC-WSU/GOMC-MoSDeF) with [MoSDeF-GOMC YouTube videos](https://www.youtube.com/watch?v=7StVoUCGkHs&list=PLdxD0z6HRx8Y9VhwcODxAHNQBBJDRvxMf)
 - [GOMC YouTube channel](https://www.youtube.com/channel/UCueLGE6tuOyu-mvxIt-U1HQ/playlists)
 - [MoSDeF tools](https://mosdef.org/)

### Citation

Please cite MoSDeF-GOMC, GOMC, and MoSDeF tools, which are provided [here](XXX).

### Installation

The MoSDeF-GOMC package is available via conda:

conda create --name mosdef_gomc -c conda-forge mosdef_gomc
