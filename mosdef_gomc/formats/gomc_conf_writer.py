class GOMCControl:
    def __init__(
        self,
        charmm_object,
        ensemble_type,
        RunSteps,
        Temperature,
        ff_psf_pdb_file_directory=None,
        check_input_files_exist=True,
        Restart=False,
        RestartCheckpoint=False,
        ExpertMode=False,
        Parameters=None,
        Coordinates_box_0=None,
        Structure_box_0=None,
        Coordinates_box_1=None,
        Structure_box_1=None,
        binCoordinates_box_0=None,
        extendedSystem_box_0=None,
        binVelocities_box_0=None,
        binCoordinates_box_1=None,
        extendedSystem_box_1=None,
        binVelocities_box_1=None,
        input_variables_dict=None,
    ):
        depreciation_error = (
            "The mosdef_gomc 'gomc_conf_writer.py' has been depreciated, and the entire 'gomc_conf_writer.py' "
            "file will be removed/deleted soon."
            "The this only effects the mosdef-gomc 'gomc_conf_writer' parmed version.  The GMSO version, "
            "'gmso_gomc_conf_writer.py', has replaced it with more features, so please use this GMSO version."
        )
        raise ModuleNotFoundError(depreciation_error)


# user callable function to write the GOMC control file
def write_gomc_control_file(
    charmm_object,
    conf_filename,
    ensemble_type,
    RunSteps,
    Temperature,
    ff_psf_pdb_file_directory=None,
    check_input_files_exist=True,
    Restart=False,
    RestartCheckpoint=False,
    ExpertMode=False,
    Parameters=None,
    Coordinates_box_0=None,
    Structure_box_0=None,
    Coordinates_box_1=None,
    Structure_box_1=None,
    binCoordinates_box_0=None,
    extendedSystem_box_0=None,
    binVelocities_box_0=None,
    binCoordinates_box_1=None,
    extendedSystem_box_1=None,
    binVelocities_box_1=None,
    input_variables_dict=None,
):
    GOMCControl(
        charmm_object,
        ensemble_type,
        RunSteps,
        Temperature,
        ff_psf_pdb_file_directory=ff_psf_pdb_file_directory,
        check_input_files_exist=check_input_files_exist,
        Restart=Restart,
        RestartCheckpoint=RestartCheckpoint,
        ExpertMode=ExpertMode,
        Parameters=Parameters,
        Coordinates_box_0=Coordinates_box_0,
        Structure_box_0=Structure_box_0,
        Coordinates_box_1=Coordinates_box_1,
        Structure_box_1=Structure_box_1,
        binCoordinates_box_0=binCoordinates_box_0,
        extendedSystem_box_0=extendedSystem_box_0,
        binVelocities_box_0=binVelocities_box_0,
        binCoordinates_box_1=binCoordinates_box_1,
        extendedSystem_box_1=extendedSystem_box_1,
        binVelocities_box_1=binVelocities_box_1,
        input_variables_dict=input_variables_dict,
    )
