Build A System and Write All Files
==================================
.. image:: https://img.shields.io/badge/license-MIT-blue.svg
    :target: http://opensource.org/licenses/MIT


All-Atom (AA) Hexane and Pentane System
---------------------------------------

.. note::
    NAMD can directly use these files, but the NAMD control file is not written with this software and needs to be provided by the user.

Import the required mbuild package.

.. code:: ipython3

    import mbuild as mb
    import unyt as u
    import mosdef_gomc.formats.gmso_charmm_writer as mf_charmm
    import mosdef_gomc.formats.gmso_gomc_conf_writer as gomc_control


Construct a hexane and pentane all-atom (AA) system using the OPLS-AA force field (FF),
which is a standard FF supplied with `foyer <https://foyer.mosdef.org/en/stable/>`_.
The molecules are built via `smiles strings <https://www.daylight.com/dayhtml/doc/theory/theory.smiles.html>`_.
The hexane and pentane residues are named `"HEX"` and `"PEN"`, respectively.
All the `GPU Optimized Monte Carlo (GOMC) <https://github.com/GOMC-WSU/GOMC>`_ files for conducting a are written, including the **GOMC** control file.  The GOMC control file selects many defaults, minimizes novice user errors, and allows expert users to change GOMC parameters.

.. code:: ipython3

    # GOMC Example for hexane and pentane the NPT Ensemble via MoSDeF.

    # Specify the box dimensions, number of molecules, and mol ratios.
    import mbuild as mb
    import unyt as u
    import mosdef_gomc.formats.gmso_charmm_writer as mf_charmm
    import mosdef_gomc.formats.gmso_gomc_conf_writer as gomc_control

    # Create the hexane and pentane molecules with residue names.
    hexane =mb.load('CCCCCC', smiles=True)
    hexane.name = 'HEX'

    pentane = mb.load('CCCCC', smiles=True)
    pentane.name = 'PEN'

    # Build the main liquid simulation box.
    # NOTE: mBuild dimensions are in nm, and density in kg/m^3.
    box_liq = mb.fill_box(compound=[hexane, pentane],
                          compound_ratio=[0.5, 0.5],
                          density=700,
                          box=[4.5, 4.5, 4.5]
                          )

    # Build the Charmm object, which is required to write the
    # FF (.inp), psf, pdb, and GOMC control files.
    charmm = mf_charmm.Charmm(box_liq,
                              'NPT_hexane_pentane',
                              structure_box_1=None,
                              filename_box_1=None,
                              ff_filename="NPT_hexane_pentane_FF" ,
                              forcefield_selection={
                                  hexane.name: 'oplsaa' ,
                                  pentane.name: 'oplsaa'
                              },
                              residues=[hexane.name, pentane.name],
                             )

    ## Write the write the FF (.inp), psf, pdb, and GOMC control files
    charmm.write_inp()
    charmm.write_psf()
    charmm.write_pdb()

    gomc_control.write_gomc_control_file(charmm,
                                         'in_NPT.conf',
                                         'NPT',
                                         100,
                                         300 * u.K,
                                         input_variables_dict={"Pressure": 10 * u.bar}
                                         )
