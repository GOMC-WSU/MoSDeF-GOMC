# GMSO and foyer use specific residues to apply force fields and mapping molecule number to atom numbers
import os
from warnings import warn
from xml.dom import minidom

import parmed as pmd

import mbuild as mb
from mbuild.compound import Compound
from mbuild.utils.io import has_foyer
from gmso.external import from_parmed
import gmso
from foyer.general_forcefield import Forcefield as gmsoFF
from foyer import Forcefield as oldFF

import datetime


def specific_ff_to_residue(
    structure,
    forcefield_selection=None,
    residues=None,
    reorder_res_in_pdb_psf=False,
    boxes_for_simulation=1,
):

    """
    Takes the mbuild Compound or mbuild Box structure and applies the selected
    force field to the corresponding residue via foyer and GMSO.
    Note: a residue is defined as a molecule in this case, so it is not
    designed for applying a force field to a protein.

    Parameters
    ----------
    structure: mbuild Compound object or mbuild Box object;
        The mBuild Compound object or mbuild Box object, which contains the molecules
        (or empty box) that will have the force field applied to them.
    forcefield_selection: str or dictionary, default=None
        Apply a force field to the output file by selecting a force field xml file with
        its path or by using the standard force field name provided the `foyer` package.
        Example dict for FF file: {'ETH' : 'oplsaa.xml', 'OCT': 'path_to file/trappe-ua.xml'}
        Example str for FF file: 'path_to file/trappe-ua.xml'
        Example dict for standard FF names : {'ETH' : 'oplsaa', 'OCT': 'trappe-ua'}
        Example str for standard FF names: 'trappe-ua'
        Example of a mixed dict with both : {'ETH' : 'oplsaa', 'OCT': 'path_to file/'trappe-ua.xml'}
    residues: list, [str, ..., str], default=None
        Labels of unique residues in the Compound. Residues are assigned by
        checking against Compound.name.  Only supply residue names as 4 characters
        strings, as the residue names are truncated to 4 characters to fit in the
        psf and pdb file.
    reorder_res_in_pdb_psf: bool, default=False
        This option provides the ability to reorder the residues/molecules from the original
        structure's order.  If True, the residues will be reordered as they appear in the residues
        variable.  If False, the order will be the same as entered in the original structure.
    boxes_for_simulation: int [1, 2], default = 1
        Gibbs (GEMC) or grand canonical (GCMC) ensembles are examples of where the boxes_for_simulation would be 2.
        Canonical (NVT) or isothermal–isobaric (NPT) ensembles are example with the boxes_for_simulation equal to 1.
        Note: the only valid options are 1 or 2.

    Returns
    -------
    list, [
        topology,
        residues_applied_list,
        electrostatics14Scale_dict,
        nonBonded14Scale_dict,
        atom_types_dict,
        bond_types_dict,
        angle_types_dict,
        dihedral_types_dict,
        improper_types_dict,
        combining_rule_dict,
        ]

    topology: gmso.Topology
        gmso Topology with applied force field
    residues_applied_list: list
        list of residues (i.e., list of stings).
        These are all the residues in which the force field actually applied.
    electrostatics14Scale_dict: dict
        A dictionary with the 1,4-electrostatic/Coulombic scalars for each residue,
        as the forcefields are specified by residue {'residue_name': '1-4_electrostatic_scaler'}.
    nonBonded14Scale_dict: dict
        A dictionary with the 1,4-non-bonded scalars for each residue,
        as the forcefields are specified by residue {'residue_name': '1-4_nonBonded_scaler'}.
    atom_types_dict: dict
        A dict with the all the residues as the keys. The values are a list containing,
        {'expression': confirmed singular atom types expression or equation,
        'atom_types': gmso Topology.atom_types}.
    bond_types_dict: dict
        A dict with the all the residues as the keys. The values are a list containing,
        {'expression': confirmed singular bond types expression or equation,
        'bond_types': gmso Topology.bond_types}.
    angle_types_dict: dict
        A dict with the all the residues as the keys. The values are a list containing,
        {'expression': confirmed singular angle types expression or equation,
        'angle_types': gmso Topology.angle_types}.
    dihedral_types_dict: dict
        A dict with the all the residues as the keys. The values are a list containing,
        {'expression': confirmed singular dihedral types expression or equation,
        'dihedral_types': gmso Topology.dihedral_types}.
    improper_types_dict: dict
        A dict with the all the residues as the keys. The values are a list containing,
        {'expression': confirmed singular improper types expression or equation,
        'improper_types': gmso Topology.improper_types}.
    combining_rule_dict: dict
        A dict with the all the residues as the keys and the gmso Topology._combining_rule as the values.
        {'residue_name': 'combining_rule_type'}.
    Notes
    -----
    To write the NAMD/GOMC force field, pdb, psf, and force field
    (.inp) files, the residues and forcefields must be provided in
    a str or dictionary. If a dictionary is provided all residues must
    be specified to a force field if the boxes_for_simulation is equal to 1.

    Generating an empty box (i.e., pdb and psf files):
    Enter residues = [], but the accompanying structure must be an empty mb.Box.
    However, when doing this, the forcefield_selection must be supplied,
    or it will provide an error (i.e., forcefield_selection can not be equal to None).

    In this current FF/psf/pdb writer, a residue type is essentially a molecule type.
    Therefore, it can only correctly write systems where every bead/atom in the molecule
    has the same residue name, and the residue name is specific to that molecule type.
    For example: a protein molecule with many residue names is not currently supported,
    but is planned to be supported in the future.
    """

    if has_foyer:
        from foyer import Forcefield
        from foyer.forcefields import forcefields
    else:
        print_error_message = (
            "Package foyer is not installed. "
            "Please install it using conda install -c conda-forge foyer"
        )
        raise ImportError(print_error_message)

    if not isinstance(structure, (Compound, mb.Box)):
        print_error_message = (
            "ERROR: The structure expected to be of type: "
            "{} or {}, received: {}".format(
                type(Compound()),
                type(mb.Box(lengths=[1, 1, 1])),
                type(structure),
            )
        )
        raise TypeError(print_error_message)

    print("forcefield_selection = " + str(forcefield_selection))
    if forcefield_selection is None:
        print_error_message = (
            "Please the force field selection (forcefield_selection) as a dictionary "
            "with all the residues specified to a force field "
            '-> Ex: {"Water" : "oplsaa", "OCT": "path/trappe-ua.xml"}, '
            "Note: the file path must be specified the force field file "
            "or by using the standard force field name provided the `foyer` package."
        )
        raise TypeError(print_error_message)

    elif forcefield_selection is not None and not isinstance(
        forcefield_selection, dict
    ):
        print_error_message = (
            "The force field selection (forcefield_selection) "
            "is not a dictionary. Please enter a dictionary "
            "with all the residues specified to a force field "
            '-> Ex: {"Water" : "oplsaa", "OCT": "path/trappe-ua.xml"}, '
            "Note: the file path must be specified the force field file "
            "or by using the standard force field name provided the `foyer` package."
        )
        raise TypeError(print_error_message)

    if residues is None or not isinstance(residues, list):
        print_error_message = (
            "Please enter the residues in the Specific_FF_to_residue function."
        )
        raise TypeError(print_error_message)

    if not isinstance(reorder_res_in_pdb_psf, bool):
        print_error_message = (
            "Please enter the reorder_res_in_pdb_psf "
            "in the Specific_FF_to_residue function (i.e., True or False)."
        )
        raise TypeError(print_error_message)

    print_error_message_for_boxes_for_simulatiion = (
        "ERROR: Please enter boxes_for_simulation equal " "the integer 1 or 2."
    )
    if not isinstance(boxes_for_simulation, int):
        raise TypeError(print_error_message_for_boxes_for_simulatiion)

    elif isinstance(boxes_for_simulation, int) and boxes_for_simulation not in [
        1,
        2,
    ]:
        raise ValueError(print_error_message_for_boxes_for_simulatiion)

    forcefield_keys_list = []
    if forcefield_selection is not None:
        for res in forcefield_selection.keys():
            forcefield_keys_list.append(res)
        ff_data = forcefield_selection

    if forcefield_keys_list == [] and len(residues) != 0:
        print_error_message = "The forcefield_selection variable are not provided, but there are residues provided."
        raise ValueError(print_error_message)

    elif forcefield_keys_list != [] and len(residues) == 0:
        print_error_message = (
            "The residues variable is an empty list but there are "
            "forcefield_selection variables provided."
        )
        raise ValueError(print_error_message)

    user_entered_ff_with_path_dict = (
        {}
    )  # True means user entered the path, False is a standard foyer FF with no path
    for z in range(0, len(forcefield_keys_list)):
        for res_i in range(0, len(residues)):
            if residues[res_i] == forcefield_keys_list[z]:
                if (
                    os.path.splitext(ff_data[forcefield_keys_list[z]])[1]
                    == ".xml"
                    and len(residues) != 0
                ):
                    user_entered_ff_with_path_dict.update(
                        {residues[res_i]: True}
                    )
                elif (
                    os.path.splitext(ff_data[forcefield_keys_list[z]])[1] == ""
                    and len(residues) != 0
                ):
                    user_entered_ff_with_path_dict.update(
                        {residues[res_i]: False}
                    )
                else:
                    print_error_message = (
                        r"Please make sure you are entering the correct "
                        "foyer FF name and not a path to a FF file. "
                        "If you are entering a path to a FF file, "
                        "please use the forcefield_files variable with the "
                        "proper XML extension (.xml)."
                    )
                    raise ValueError(print_error_message)

    electrostatics14Scale_dict = {}
    nonBonded14Scale_dict = {}

    atom_types_dict = {}
    bond_types_dict = {}
    angle_types_dict = {}
    dihedral_types_dict = {}
    improper_types_dict = {}

    combining_rule_dict = {}


    for j in range(0, len(forcefield_keys_list)):
        residue_iteration = forcefield_keys_list[j]
        if user_entered_ff_with_path_dict[residue_iteration]:
            ff_for_residue_iteration = ff_data[residue_iteration]
            try:
                read_xlm_iteration = minidom.parse(ff_for_residue_iteration)

            except:
                print_error_message = (
                    "Please make sure you are entering the correct foyer FF path, "
                    "including the FF file name.xml. "
                    "If you are using the pre-build FF files in foyer, "
                    "only use the string name without any extension. "
                    "The selected FF file could also could not formated properly, or "
                    "there may be errors in the FF file itself."
                )
                raise ValueError(print_error_message)
        elif not user_entered_ff_with_path_dict[residue_iteration]:
            ff_for_residue_iteration = ff_data[residue_iteration]
            ff_names_path_iteration = (
                forcefields.get_ff_path()[0]
                + "/xml/"
                + ff_for_residue_iteration
                + ".xml"
            )
            try:
                read_xlm_iteration = minidom.parse(ff_names_path_iteration)
            except:
                print_error_message = (
                    "Please make sure you are entering the correct foyer FF name, or the "
                    "correct file extension (i.e., .xml, if required)."
                )
                raise ValueError(print_error_message)

    # Check to see if it is an empty mbuild.Compound and set intial atoms to 0
    # note empty mbuild.Compound will read 1 atoms but there is really noting there
    if isinstance(structure, Compound):
        if len(structure.children) == 0:
            # there are no real atoms in the Compound so the test fails. User should use mbuild.Box
            print_error_message = (
                "ERROR: If you are not providing an empty box, "
                "you need to specify the atoms/beads as children in the mb.Compound. "
                "If you are providing and empty box, please do so by specifying and "
                "mbuild Box ({})".format(type(mb.Box(lengths=[1, 1, 1])))
            )
            raise TypeError(print_error_message)
        else:
            initial_no_atoms = len(structure.to_parmed().atoms)

    # calculate the initial number of atoms for later comparison
    if isinstance(structure, mb.Box):
        lengths = structure.lengths
        angles = structure.angles

        structure = mb.Compound()
        structure.box = mb.Box(lengths=lengths, angles=angles)
        initial_no_atoms = 0

    # add the FF to the residues
    compound_box_infor = structure.to_parmed(residues=residues)
    new_topology = gmso.Topology(name="main", box=compound_box_infor.box)

    # create a temporary list of gmso.Topology objects until the gmso.Topology object can be directly added together
    topology_per_residue_list = []

    # prepare all compound and remove nested compounds
    no_layers_to_check_for_residues = 3

    print_error_message_all_res_not_specified = (
        "ERROR: All the residues are not specified, or "
        "the residues entered does not match the residues that "
        "were found and built for structure."
    )
    for j in range(0, no_layers_to_check_for_residues):
        new_compound_iter = mb.Compound()
        new_compound_iter.periodicity = structure.periodicity
        if structure.name in residues:
            if len(structure.children) == 0:
                warn(
                    "Warning: This residue is the atom, and is a single atom., "
                    + str(structure.name)
                )
                new_compound_iter.add(mb.compound.clone(structure))

            elif len(structure.children) > 0:

                new_compound_iter.add(mb.compound.clone(structure))

        else:
            for child in structure.children:
                if len(child.children) == 0:
                    if child.name not in residues:
                        raise ValueError(
                            print_error_message_all_res_not_specified
                        )

                    else:
                        new_compound_iter.add(mb.compound.clone(child))

                elif len(child.children) > 0:
                    if child.name in residues:
                        new_compound_iter.add(mb.compound.clone(child))
                    else:
                        for sub_child in child.children:
                            if sub_child.name in residues:
                                new_compound_iter.add(
                                    mb.compound.clone(sub_child)
                                )

                            else:
                                if len(sub_child.children) == 0 and (
                                    child.name not in residues
                                ):

                                    raise ValueError(
                                        print_error_message_all_res_not_specified
                                    )

        structure = new_compound_iter

    residues_applied_list = []
    residue_orig_order_list = []
    for child in structure.children:
        if child.name not in residue_orig_order_list:
            residue_orig_order_list.append(child.name)
    for res_reorder_iter in range(0, len(residues)):
        if residues[res_reorder_iter] not in residue_orig_order_list:
            text_to_print_1 = (
                "All the residues were not used from the forcefield_selection "
                "string or dictionary. There may be residues below other "
                "specified residues in the mbuild.Compound hierarchy. "
                "If so, all the highest listed residues pass down the force "
                "fields through the hierarchy. Alternatively, residues that "
                "are not in the structure may have been specified. "
            )
            text_to_print_2 = (
                "Note: This warning will appear if you are using the CHARMM pdb and psf writers "
                + "2 boxes, and the boxes do not contain all the residues in each box."
            )
            if boxes_for_simulation == 1:
                warn(text_to_print_1)
                raise ValueError(text_to_print_1)
            if boxes_for_simulation == 2:
                warn(text_to_print_1 + text_to_print_2)

    if not reorder_res_in_pdb_psf:
        residues = residue_orig_order_list
    elif reorder_res_in_pdb_psf:
        print(
            "INFO: the output file are being reordered in via the residues list's sequence."
        )

    atom_number = 0 # 0 sets the 1st atom_number at 1
    molecule_number = 0 # 0 sets the 1st molecule_number at 1
    molecules_atom_number_dict = {}
    for i in range(0, len(residues)):
        children_in_iteration = False
        new_compound_iteration = mb.Compound()
        new_compound_iter.periodicity = structure.periodicity
        for child in structure.children:
            if ff_data.get(child.name) is None:
                print_error_message = "ERROR: All residues are not specified in the force_field dictionary"
                raise ValueError(print_error_message)

            if child.name == residues[i]:
                children_in_iteration = True
                new_compound_iteration.add(mb.compound.clone(child))

        if children_in_iteration:
            if user_entered_ff_with_path_dict[residues[i]]:
                ff_iteration = gmsoFF(forcefield_files=ff_data[residues[i]],
                                      strict=False)
                residues_applied_list.append(residues[i])
            elif not user_entered_ff_with_path_dict[residues[i]]:
                ff_iteration = gmsoFF(name=ff_data[residues[i]],
                                      strict=False)
                residues_applied_list.append(residues[i])

            new_compound_iteration.box = None



            ff_apply_start_time_s = datetime.datetime.today()
            new_topology_iteration = ff_iteration.apply(
                new_compound_iteration,
                residues=[residues[i]],
                assert_improper_params=False,
                name = residues[i],
                box=compound_box_infor.box
            )


            ff_apply_end_time_s = datetime.datetime.today()
            ff_apply_total_time_s = (ff_apply_end_time_s - ff_apply_start_time_s).total_seconds()
            write_log_data = f"*************************************************\n" \
                             f"residue name = {residues[i]} \n" \
                             f"ff_apply_total_time_s (s) = {ff_apply_total_time_s} \n"
            print(write_log_data)


            print('xxxxxxxxxxxxxxxxxxx')
            print('xxxxxxxxxxxxxxxxxxx')
            print('xxxxxxxxxxxxxxxxxxx')

            # add atom numbers (renumber) to the combined gmso topology
            for site in new_topology_iteration.sites:
                site.__dict__['label_'] = atom_number
                atom_number += 1

            bonded_atom_number_set = set()
            all_bonded_atoms_list = set()
            for bond in new_topology_iteration.bonds:
                bonded_atom_0_iter = bond.__dict__['connection_members_'][0].__dict__['label_']
                bonded_atom_1_iter = bond.__dict__['connection_members_'][1].__dict__['label_']

                if bonded_atom_0_iter < bonded_atom_1_iter:
                    bonded_atom_tuple_iter = (bonded_atom_0_iter, bonded_atom_1_iter)
                else:
                    bonded_atom_tuple_iter = (bonded_atom_1_iter, bonded_atom_0_iter)

                bonded_atom_number_set.add(bonded_atom_tuple_iter)
                all_bonded_atoms_list.add(bonded_atom_0_iter)
                all_bonded_atoms_list.add(bonded_atom_1_iter)

            # map all bonded atoms as molecules
            molecules_atom_number_list = []
            for site in new_topology_iteration.sites:
                atom_iter_k = site.__dict__['label_']

                if atom_iter_k in all_bonded_atoms_list:
                    for bonded_atoms_n in bonded_atom_number_set:
                        if atom_iter_k in bonded_atoms_n:
                            bonded_atoms_n_list_iter = list(bonded_atoms_n)
                            atom_found_iter = False
                            if len(molecules_atom_number_list) != 0:
                                for molecule_j in range(0, len(molecules_atom_number_list)):
                                    if atom_iter_k in molecules_atom_number_list[molecule_j]:
                                        molecules_atom_number_list[molecule_j].add(bonded_atoms_n_list_iter[0])
                                        molecules_atom_number_list[molecule_j].add(bonded_atoms_n_list_iter[1])
                                        atom_found_iter = True

                                    if (molecule_j == len(molecules_atom_number_list) - 1) \
                                            and atom_found_iter is False:
                                        molecules_atom_number_list.append(
                                            {
                                                bonded_atoms_n_list_iter[0],
                                                bonded_atoms_n_list_iter[1]
                                            }
                                        )
                            else:
                                molecules_atom_number_list.append(
                                    {
                                        bonded_atoms_n_list_iter[0],
                                        bonded_atoms_n_list_iter[1]
                                    }
                                )
                else:
                    molecules_atom_number_list.append({atom_iter_k})

            # create a molecule number to atom number dict
            # Example:  {molecule_number_x: {atom_number_1, ..., atom_number_y}, ...}
            for molecule_iter in range(0, len(molecules_atom_number_list)):
                molecules_atom_number_dict.update(
                    {molecule_number: molecules_atom_number_list[molecule_iter]}
                )
                molecule_number += 1

            for site in new_topology_iteration.sites:
                site_atom_number_iter = site.__dict__['label_']

                # get molecule number
                for mol_n, atom_set_n in molecules_atom_number_dict.items():
                    if site_atom_number_iter in atom_set_n:
                        molecule_p_number = mol_n

                # change 'residue_label_' to "residue_name_"
                # change 'residue_index_' to "residue_number_"
                site.__dict__['residue_label_'] = residues[i]
                site.__dict__['residue_index_'] = molecule_p_number + 1

            # get the non-bonded, bond, angle, dihedral and improper equations and impropers and other info
            atom_type_expression_set = set()
            for atom_type_k in new_topology_iteration.atom_types:
                atom_type_expression_set.add(atom_type_k.expression)
            if len(atom_type_expression_set) == 1:
                atom_types_dict.update({residues[i]: {'expression': list(atom_type_expression_set)[0],
                                                      'atom_types': new_topology_iteration.atom_types
                                                      }
                                        }
                                       )
            elif len(atom_type_expression_set) == 0:
                atom_types_dict.update({residues[i]: None})
            else:
                raise ValueError('ERROR: There is more than 1 {} equation types per residue or molecules '
                                 ''.format('non-bonded'))

            bond_type_expression_set = set()
            for bond_type_k in new_topology_iteration.bond_types:
                bond_type_expression_set.add(bond_type_k.expression)
            if len(bond_type_expression_set) == 1:
                bond_types_dict.update({residues[i]: {'expression': list(bond_type_expression_set)[0],
                                                      'bond_types': new_topology_iteration.bond_types}})
            elif len(bond_type_expression_set) == 0:
                bond_types_dict.update({residues[i]: None})
            else:
                raise ValueError('ERROR: There is more than 1 {} equation types per residue or molecules '
                                 ''.format('bond'))

            angle_type_expression_set = set()
            for angle_type_k in new_topology_iteration.angle_types:
                angle_type_expression_set.add(angle_type_k.expression)
            if len(angle_type_expression_set) == 1:
                angle_types_dict.update({residues[i]: {'expression': list(angle_type_expression_set)[0],
                                         'angle_types': new_topology_iteration.angle_types}})
            elif len(angle_type_expression_set) == 0:
                angle_types_dict.update({residues[i]: None})
            else:
                raise ValueError('ERROR: There is more than 1 {} equation types per residue or molecules '
                                 ''.format('angle'))

            dihedral_type_expression_set = set()
            for dihedral_type_k in new_topology_iteration.dihedral_types:
                dihedral_type_expression_set.add(dihedral_type_k.expression)
            if len(dihedral_type_expression_set) == 1:
                dihedral_types_dict.update({residues[i]: {'expression': list(dihedral_type_expression_set)[0],
                                                          'dihedral_types': new_topology_iteration.dihedral_types}})
            elif len(dihedral_type_expression_set) == 0:
                dihedral_types_dict.update({residues[i]: None})
            else:
                raise ValueError('ERROR: There is more than 1 {} equation types per residue or molecules '
                                 ''.format('dihedral'))

            improper_type_expression_set = set()
            for improper_type_k in new_topology_iteration.improper_types:
                improper_type_expression_set.add(improper_type_k.expression)
            if len(improper_type_expression_set) == 1:
                improper_types_dict.update({residues[i]: {'expression': list(improper_type_expression_set)[0],
                                                          'improper_types': new_topology_iteration.improper_types}})
            elif len(improper_type_expression_set) == 0:
                improper_types_dict.update({residues[i]: None})
            else:
                raise ValueError('ERROR: There is more than 1 {} equation types per residue or molecules '
                                 ''.format('improper'))

            nonBonded_1_4_scaling_factor_set = set()
            electro_1_4_scaling_factor_set = set()
            nonBonded_1_4_scaling_factor_set.add(new_topology_iteration.scaling_factors["nonBonded14Scale"])
            electro_1_4_scaling_factor_set.add(new_topology_iteration.scaling_factors["electrostatics14Scale"])
            if len(nonBonded_1_4_scaling_factor_set) == 1 and len(electro_1_4_scaling_factor_set) == 1:
                nonBonded14Scale_dict.update({residues[i]: list(nonBonded_1_4_scaling_factor_set)[0]})
                electrostatics14Scale_dict.update({residues[i]: list(electro_1_4_scaling_factor_set)[0]})
            elif len(nonBonded_1_4_scaling_factor_set) == 0 and len(electro_1_4_scaling_factor_set) == 0:
                nonBonded14Scale_dict.update({residues[i]: None})
                electrostatics14Scale_dict.update({residues[i]: None})
            else:
                raise ValueError('ERROR: There is more than 1 {} equation types per residue or molecules '
                                 ''.format('1-4 scaling facter'))

            combining_rule_dict.update({residues[i]: new_topology_iteration._combining_rule})

            topology_per_residue_list.append(new_topology_iteration)

    # iterate thru topologies
    for top_i in topology_per_residue_list:
        # iterate thru sites and add to empty topology
        for site_i in top_i.sites:
            new_topology.add_site(site_i)

        # iterate thru connections (bonds, angles, dihedrals, and impropers) and add to empty topology
        for connection_i in top_i.connections:
            new_topology.add_connection(connection_i)

    topology = new_topology

    # calculate the final number of atoms
    final_no_atoms = topology.n_sites

    if final_no_atoms != initial_no_atoms:
        print_error_message = (
            "ERROR: The initial number of atoms sent to the force field analysis is "
            "not the same as the final number of atoms analyzed. "
            "The initial number of atoms was {} and the final number of atoms was {}. "
            "Please ensure that all the residues names that are in the initial "
            "Compound are listed in the residues list "
            "(i.e., the residues variable).".format(
                initial_no_atoms, final_no_atoms
            )
        )
        raise ValueError(print_error_message)

    return [
        topology,
        residues_applied_list,
        electrostatics14Scale_dict,
        nonBonded14Scale_dict,
        atom_types_dict,
        bond_types_dict,
        angle_types_dict,
        dihedral_types_dict,
        improper_types_dict,
        combining_rule_dict,
    ]
