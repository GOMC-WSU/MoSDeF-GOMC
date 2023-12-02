class Charmm:
    def __init__(
        self,
        structure_box_0,
        filename_box_0,
        structure_box_1=None,
        filename_box_1=None,
        non_bonded_type="LJ",
        forcefield_selection=None,
        residues=None,
        detect_forcefield_style=True,
        gomc_fix_bonds_angles=None,
        gomc_fix_bonds=None,
        gomc_fix_angles=None,
        bead_to_atom_name_dict=None,
        fix_residue=None,
        fix_residue_in_box=None,
        set_residue_pdb_occupancy_to_1=None,
        ff_filename=None,
        reorder_res_in_pdb_psf=False,
    ):
        # depreciation error that charmm_writer has been depreciated
        depreciation_error = (
            "The mosdef_gomc 'charmm_writer.py' has been depreciated, and the entire 'charmm_writer.py' "
            "file will be removed/deleted soon."
            "The this only effects the mosdef-gomc 'charmm_writer' parmed version.  The GMSO version, "
            "'gmso_charmm_writer.py', has replaced it with more features, so please use this GMSO version."
        )
        raise ModuleNotFoundError(depreciation_error)
