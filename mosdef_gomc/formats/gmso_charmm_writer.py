import datetime
import os
from warnings import warn
import mosdef_gomc

import numpy as np
import gmso
import unyt as u
from unyt.dimensions import (
    length,
    energy,
    temperature,
    angle
)

from mbuild.box import Box
from mbuild.compound import Compound
from mbuild.utils.conversion import (
    RB_to_CHARMM,
    OPLS_to_CHARMM,
)
from mosdef_gomc.utils.conversion import (
    base10_to_base16_alph_num,
    base10_to_base26_alph,
    base10_to_base52_alph,
    base10_to_base62_alph_num,
)
from mbuild.utils.sorting import natural_sort
from mosdef_gomc.utils.gmso_specific_ff_to_residue import specific_ff_to_residue
from mbuild.utils.gmso_equation_compare import (
    get_atom_type_expressions_and_scalars,
    evaluate_harmonic_bond_format_with_scaler,
    evaluate_harmonic_angle_format_with_scaler,
    evaluate_harmonic_torsion_format_with_scaler,
    evaluate_OPLS_torsion_format_with_scaler,
    evaluate_periodic_torsion_format_with_scaler,
    evaluate_RB_torsion_format_with_scaler,
)


def _check_convert_bond_k_constant_units(
        bond_class_input_str,
        bond_energy_input_unyt,
        bond_energy_output_units_str,
):
    """Checks to see if the value is a valid bond k-constant
    energy value and converts it to kcal/mol/angstroms**2

    Parameters
    ----------
    bond_class_input_str : str
        The bond class information from the gmso.bond_types.member_class
    bond_energy_input_unyt : unyt.unyt_quantity
        The bond energy in units of 'energy/mol/angstroms**2' or 'K/angstroms**2'.
        NOTE that the only valid temperature unit for thermal energy is Kelvin (K),
        which in 'K/angstroms**2'.
    bond_class_output_units_str : str ('kcal/mol/angstrom**2' or 'K/angstrom**2')
        The bond class information from the gmso.bond_types.member_class


    Returns
    -------
    If the bond_energy_input_unyt value is unyt.unyt_quantity of energy/length**2 : unyt.unyt_quantity
        The value is in 'kcal/mol/angstrom**2' or 'K/angstrom**2' units.
    If the bond_energy_input_unyt value is not unyt.unyt_quantity of energy/length**2 : raise TypeError
    If the bond_energy_output_units_str value is not a valid choice: raise ValueError
    """

    if bond_energy_output_units_str not in ['kcal/mol/angstrom**2', 'K/angstrom**2']:
        print_error_message = (
            "ERROR: The selected bond energy k-constant units via "
            "bond_energy_output_units_str "
            "are not 'kcal/mol/angstrom**2' or 'K/angstrom**2'."
        )
        raise ValueError(print_error_message)

    print_error_message = (
        f"ERROR: The bond class input, {bond_class_input_str} is {type(bond_energy_input_unyt)} "
        f"and needs to be a {u.array.unyt_quantity} "
        f"in energy/length**2 units, "
        f"such as 'kcal/mol/angstrom**2', 'kJ/mol/angstrom**2', or 'K/angstrom**2'."
    )
    if isinstance(bond_energy_input_unyt, u.array.unyt_quantity):
        if energy / length ** 2 == bond_energy_input_unyt.units.dimensions:
            if bond_energy_output_units_str == 'kcal/mol/angstrom**2':
                bond_energy_output_unyt = bond_energy_input_unyt.to('kcal/mol/angstrom**2')
                return bond_energy_output_unyt

            elif bond_energy_output_units_str == 'K/angstrom**2':
                bond_energy_output_unyt = bond_energy_input_unyt.to('kcal/mol/angstrom**2')
                bond_energy_output_unyt = bond_energy_output_unyt * u.angstrom**2
                bond_energy_output_unyt = bond_energy_output_unyt.to('K', equivalence='thermal')
                bond_energy_output_unyt = bond_energy_output_unyt / u.angstrom**2
                return bond_energy_output_unyt

        elif temperature / length ** 2 == bond_energy_input_unyt.units.dimensions:
            if bond_energy_output_units_str == 'kcal/mol/angstrom**2':
                bond_energy_output_unyt = bond_energy_input_unyt.to('K/angstrom**2')
                bond_energy_output_unyt = bond_energy_output_unyt * u.angstrom**2
                bond_energy_output_unyt = bond_energy_output_unyt.to('kcal/mol', equivalence='thermal')
                bond_energy_output_unyt = bond_energy_output_unyt / u.angstrom**2
                return bond_energy_output_unyt

            elif bond_energy_output_units_str == 'K/angstrom**2':
                bond_energy_output_unyt = bond_energy_input_unyt.to('K/angstrom**2')
                return bond_energy_output_unyt

        else:
            raise TypeError(print_error_message)
    else:
        raise TypeError(print_error_message)

def _check_convert_angle_k_constant_units(
        angle_class_input_str,
        angle_energy_input_unyt,
        angle_energy_output_units_str,
):
    """Checks to see if the value is a valid angle k-constant energy value
    and converts it to kcal/mol/rad**2 or 'K/rad**2'.

    Parameters
    ----------
    angle_class_input_str : str
        The angle class information from the gmso.angle_types.member_class
    angle_energy_input_unyt : unyt.unyt_quantity
        The angle energy in units of 'energy/mol/rad**2' or 'K/rad**2'.
        NOTE that the only valid temperature unit for thermal energy is Kelvin (K),
        which in 'K/rad**2'.
    angle_class_output_units_str : str ('kcal/mol/rad**2' or 'K/rad**2')
        The angle class information from the gmso.angle_types.member_class


    Returns
    -------
    If the angle_energy_input_unyt value is unyt.unyt_quantity of energy/angle**2 : unyt.unyt_quantity
        The value is in 'kcal/mol/rad**2' or 'K/rad**2' units.
    If the angle_energy_input_unyt value is not unyt.unyt_quantity of energy/angle**2 : raise TypeError
    If the angle_energy_output_units_str value is not a valid choice: raise ValueError
    """

    if angle_energy_output_units_str not in ['kcal/mol/rad**2', 'K/rad**2']:
        print_error_message = (
            "ERROR: The selected angle energy k-constant units via "
            "angle_energy_output_units_str "
            "are not 'kcal/mol/rad**2' or 'K/rad**2'."
        )
        raise ValueError(print_error_message)

    print_error_message = (
        f"ERROR: The angle class input, {angle_class_input_str} is {type(angle_energy_input_unyt)} "
        f"and needs to be a {u.array.unyt_quantity} "
        f"in energy/angle**2 units, "
        f"such as 'kcal/mol/rad**2', 'kJ/mol/rad**2', or 'K/rad**2'."
    )
    if isinstance(angle_energy_input_unyt, u.array.unyt_quantity):
        if energy / angle ** 2 == angle_energy_input_unyt.units.dimensions:
            if angle_energy_output_units_str == 'kcal/mol/rad**2':
                angle_energy_output_unyt = angle_energy_input_unyt.to('kcal/mol/rad**2')
                return angle_energy_output_unyt

            elif angle_energy_output_units_str == 'K/rad**2':
                angle_energy_output_unyt = angle_energy_input_unyt.to('kcal/mol/rad**2')
                angle_energy_output_unyt = angle_energy_output_unyt * u.rad**2
                angle_energy_output_unyt = angle_energy_output_unyt.to('K', equivalence='thermal')
                angle_energy_output_unyt = angle_energy_output_unyt / u.rad**2
                return angle_energy_output_unyt

        elif temperature / angle ** 2 == angle_energy_input_unyt.units.dimensions:
            if angle_energy_output_units_str == 'kcal/mol/rad**2':
                angle_energy_output_unyt = angle_energy_input_unyt.to('K/rad**2')
                angle_energy_output_unyt = angle_energy_output_unyt * u.rad**2
                angle_energy_output_unyt = angle_energy_output_unyt.to('kcal/mol', equivalence='thermal')
                angle_energy_output_unyt = angle_energy_output_unyt / u.rad**2
                return angle_energy_output_unyt

            elif angle_energy_output_units_str == 'K/rad**2':
                angle_energy_output_unyt = angle_energy_input_unyt.to('K/rad**2')
                return angle_energy_output_unyt

        else:
            raise TypeError(print_error_message)
    else:
        raise TypeError(print_error_message)

'''
def _check_convert_dihedral_k_constant_units(
        dihedral_energy_input_unyt,
):
    """Checks to see if the value is a valid dihedral k-constant energy value
    and (kcal/mol, kJ/mol or 'K', ...).

    Parameters
    ----------
    dihedral_class_input_str : str
        The dihedral class information from the gmso.dihedral_types.member_class
    dihedral_energy_input_unyt : unyt.unyt_quantity
        The dihedral energy in units of 'energy/mol' or 'K'.
        NOTE that the only valid temperature unit for thermal energy is Kelvin (K),
        which in 'K'.
    dihedral_class_output_units_str : str ('kcal/mol' or 'K')
        The dihedral class information from the gmso.dihedral_types.member_class


    Returns
    -------
    If the dihedral_energy_input_unyt value is unyt.unyt_quantity of energy : unyt.unyt_quantity
        The value is in 'kcal/mol' or 'K' units.
    If the dihedral_energy_input_unyt value is not unyt.unyt_quantity of energy : raise TypeError
    If the dihedral_energy_output_units_str value is not a valid choice: raise ValueError
    """
    if dihedral_energy_output_units_str not in ['kcal/mol', 'K']:
        print_error_message = (
            "ERROR: The selected dihedral energy k-constant units via "
            "dihedral_energy_output_units_str are not 'kcal/mol' or 'K'."
        )
        raise ValueError(print_error_message)

    print_error_message = (
        f"ERROR: The dihedral class input, {dihedral_class_input_str} is {type(dihedral_energy_input_unyt)} "
        f"and needs to be a {u.array.unyt_quantity} "
        f"in energy units, such as 'kcal/mol', 'kJ/mol', or 'K'."
    )
    raise TypeError(print_error_message)
    if isinstance(dihedral_energy_input_unyt, u.array.unyt_quantity):
        if energy == dihedral_energy_input_unyt.units.dimensions:
            if dihedral_energy_output_units_str in ['kcal/mol', 'K']:
                dihedral_energy_output_unyt = dihedral_energy_input_unyt.to(
                    dihedral_energy_output_units, equivalence='thermal')
                return dihedral_energy_output_unyt

        else:
            raise TypeError(print_error_message)

    else:
        raise TypeError(print_error_message)
'''


# this is needed later for the non-bonded fixes (NBFIX)
def _LJ_sigma_to_r_min(
    sigma
):
    """Convert sigma to Rmin for the non-bonded Lennard-Jones (LJ) potential energy equation.

    Parameters
    ----------
    sigma : int or float
        The sigma value for the non-bonded Lennard-Jones (LJ) potential energy equation.

    Returns
    ----------
    r_min : float
        The radius at the minimum energy (Rmin) for the non-bonded Lennard-Jones (LJ) potential energy equation.
    """
    r_min = float(sigma * 2**(1 / 6))

    return r_min

def _LJ_sigma_to_r_min_div_2(
    sigma
):
    """Convert sigma to Rmin/2 for the non-bonded Lennard-Jones (LJ) potential energy equation.

    Parameters
    ----------
    sigma : int or float
        The sigma value for the non-bonded Lennard-Jones (LJ) potential energy equation.

    Returns
    ----------
    r_min_div_2 : float
        The radius at the minimum energy divided by 2 (Rmin/2)
        for the non-bonded Lennard-Jones (LJ) potential energy equation.
    """
    r_min_div_2 = float(sigma * 2**(1 / 6) / 2)

    return r_min_div_2


def unique_atom_naming(
    topology, residue_id_list, residue_names_list, bead_to_atom_name_dict=None
):
    """
    Generates unique atom/bead names for each molecule, which is required for some
    simulation types (Example: The special Monte Carlo moves)

    Parameters
    ----------
    topology : gmso.Topology object
    residue_id_list : list, in sequential order
            The residue ID for every atom in the system
    residue_names_list : list, in sequential order
        The atom names for every atom in the system
    bead_to_atom_name_dict: dictionary ; optional, default =None
        For all atom names/elements/beads with 2 or less digits, this converts
        the atom name in the GOMC psf and pdb files to a unique atom name,
        provided they do not exceed 3844 atoms (62^2) of the same name/element/bead
        per residue. For all atom names/elements/beads with 3 digits, this converts
        the atom name in the GOMC psf and pdb files to a unique atom name,
        provided they do not exceed 62 of the same name/element pre residue.
        Example dictionary: {'_CH3':'C', '_CH2':'C', '_CH':'C', '_HC':'C'}

    Returns
     ----------
    unique_individual_atom_names_dict : dictionary
        All the unique atom names comno_piled into a dictionary.
    individual_atom_names_list : list, in sequential  order
        The atom names for every atom in the system
    missing_bead_to_atom_name : list, in sequential  order
        The bead names of any atoms beads that did not have a name specificed to them
        via the bead_to_atom_name_dict
    """
    unique_individual_atom_names_dict = {}
    individual_atom_names_list = []
    missing_bead_to_atom_name = []
    for i, site in enumerate(topology.sites):
        site_name = site.__dict__['name_']
        interate_thru_names = True
        j = 0
        while interate_thru_names is True:
            j = j + 1
            if str(site_name)[:1] == "_":
                if (
                    bead_to_atom_name_dict is not None
                    and (str(site_name) in bead_to_atom_name_dict) is True
                ):
                    if len(bead_to_atom_name_dict[str(site_name)]) > 2:
                        text_to_write = (
                            "ERROR: only enter atom names that have 2 or less digits"
                            + " in the Bead to atom naming dictionary (bead_to_atom_name_dict)."
                        )
                        warn(text_to_write)
                        return None, None, None
                    else:
                        atom_name_value = bead_to_atom_name_dict[str(site_name)]
                        no_digits_atom_name = 2
                else:
                    missing_bead_to_atom_name.append(1)
                    atom_name_value = "BD"
                    no_digits_atom_name = 2
            elif len(str(site_name)) > 2:
                if len(str(site_name)) == 3:
                    no_digits_atom_name = 1
                    atom_name_value = site_name
                else:
                    text_to_write = (
                        "ERROR: atom numbering will not work propery at"
                        + " the element has more than 4 charaters"
                    )
                    warn(text_to_write)
                    return None, None, None
            else:
                no_digits_atom_name = 2
                atom_name_value = site_name
            atom_name_iteration = str(atom_name_value) + str(
                base10_to_base62_alph_num(j)
            )
            atom_res_no_resname_atomname_iteration = (
                str(residue_id_list[i])
                + "_"
                + str(residue_names_list[i])
                + "_"
                + atom_name_iteration
            )

            if (
                unique_individual_atom_names_dict.get(
                    str(atom_res_no_resname_atomname_iteration)
                )
                is None
            ):
                unique_individual_atom_names_dict.update(
                    {atom_res_no_resname_atomname_iteration: i + 1}
                )
                interate_thru_names = False
                individual_atom_names_list.append(
                    str(atom_name_value)
                    + str(
                        str(base10_to_base62_alph_num(j))[-no_digits_atom_name:]
                    )
                )

    if sum(missing_bead_to_atom_name) > 0:
        warn(
            "NOTE: All bead names were not found in the Bead to atom naming dictionary (bead_to_atom_name_dict) "
        )

    return [
        unique_individual_atom_names_dict,
        individual_atom_names_list,
        missing_bead_to_atom_name,
    ]


def _lengths_angles_to_vectors(lengths, angles, precision=6):
    """Converts the length and angles into CellBasisVectors

    Parameters
    ----------
    lengths : list-like, shape=(3,), dtype=float
        Lengths of the edges of the box (user chosen units).
    angles : list-like, shape=(3,), dtype=float, default=None
        Angles (in degrees) that define the tilt of the edges of the box. If
        None is given, angles are assumed to be [90.0, 90.0, 90.0]. These are
        also known as alpha, beta, gamma in the crystallography community.
    precision : int, optional, default=6
        Control the precision of the floating point representation of box
        attributes. If none provided, the default is 6 decimals.

    Returns
    -------
    box_vectors: numpy.ndarray, [[float, float, float], [float, float, float], [float, float, float]]
        Three (3) sets vectors for box 0 each with 3 float values, which represent
        the vectors for the Charmm-style systems (units are the same as entered for lengths)

    """

    (a, b, c) = lengths

    (alpha, beta, gamma) = np.deg2rad(angles)
    cos_a = np.clip(np.cos(alpha), -1.0, 1.0)
    cos_b = np.clip(np.cos(beta), -1.0, 1.0)
    cos_g = np.clip(np.cos(gamma), -1.0, 1.0)

    sin_a = np.clip(np.sin(alpha), -1.0, 1.0)
    sin_b = np.clip(np.sin(beta), -1.0, 1.0)
    sin_g = np.clip(np.sin(gamma), -1.0, 1.0)
    a_vec = np.asarray([a, 0.0, 0.0])

    b_x = b * cos_g
    b_y = b * sin_g
    b_vec = np.asarray([b_x, b_y, 0.0])

    c_x = c * cos_b
    c_cos_y_term = (cos_a - (cos_b * cos_g)) / sin_g
    c_y = c * c_cos_y_term
    c_z = c * np.sqrt(1 - np.square(cos_b) - np.square(c_cos_y_term))
    c_vec = np.asarray([c_x, c_y, c_z])
    box_vectors = np.asarray((a_vec, b_vec, c_vec))
    box_vectors.reshape(3, 3)
    # still leaves some floating values in some cases
    box_vectors = np.around(box_vectors, decimals=precision)

    return box_vectors


def _check_fixed_bonds_angles_lists(
    gomc_fix_bonds_and_or_angles,
    gomc_fix_bonds_and_or_angles_selection,
    residues,
):
    """Check the GOMC fixed bonds and angles lists for input errors.

    Parameters
    ----------
    gomc_fix_bonds_and_or_angles : list of strings, [str, ..., str]
        A list of the residues (i.e., molecules since GOMC currently considers a
        a whole molecule as a residue) to have their bonds and/or angles held
        rigid/fixed for the GOMC simulation engine.
        The `gomc_fix_bonds_angles`, `gomc_fix_bonds`, `gomc_fix_angles` are the only possible
        variables from the `Charmm` object to be entered.
        In GOMC, the residues currently are the same for every bead or atom in
        the molecules. Therefore, when the residue is selected, the whole molecule
        is selected.
    gomc_fix_bonds_and_or_angles_selection : str
        The name of the variable that is used but formatted as a string, which is fed
        to the error and information outputs. The
        `gomc_fix_bonds_angles`, `gomc_fix_bonds`, `gomc_fix_angles` are the only possible
        variables from the `Charmm` object to be entered.
        Whichever variable you choose, the variable name is just input as a
        string here. For example, if `gomc_fix_bonds_and_or_angles` is equal to
        gomc_fix_bonds_angles, then this should be 'gomc_fix_bonds_angles'
        (i.e., `gomc_fix_bonds_and_or_angles_selection` = 'gomc_fix_bonds_angles').
    residues : list, [str, ..., str]
        Labels of unique residues in the Compound. Residues are assigned by
        checking against Compound.name.  Only supply residue names as 4 character
        strings, as the residue names are truncated to 4 characters to fit in the
        psf and pdb file.

    Returns
    -------
    Provides a ValueError or TypeError if the input is not correct.
    """

    if gomc_fix_bonds_and_or_angles is not None and not isinstance(
        gomc_fix_bonds_and_or_angles, list
    ):
        print_error_message = (
            "ERROR: Please ensure the residue names in the ({}) variable "
            "are in a list.".format(gomc_fix_bonds_and_or_angles_selection)
        )
        raise TypeError(print_error_message)

    if isinstance(gomc_fix_bonds_and_or_angles, list):
        for gomc_fix_i in gomc_fix_bonds_and_or_angles:
            if gomc_fix_i not in residues:
                print_error_message = (
                    "ERROR: Please ensure that all the residue names in the "
                    "{} list are also in the residues list.".format(
                        gomc_fix_bonds_and_or_angles_selection
                    )
                )
                raise ValueError(print_error_message)
            elif not isinstance(gomc_fix_i, str):
                print_error_message = "ERROR: Please enter a fix_res_bonds list with only string values."
                raise TypeError(print_error_message)
            else:
                print(
                    "INFORMATION: The following residues will have these fixed parameters: "
                    + "gomc_fix_bonds = {}".format(gomc_fix_bonds_and_or_angles)
                )


class Charmm:
    """Generates a Charmm object via foyer and gmso that is required to produce the Charmm style parameter
    (force field), PDB, PSF files, which are usable in the GOMC and NAMD engines.
    Additionally, this Charmm object is also used in generating the GOMC control file.

    The units for the GOMC and NAMD output data files.
        * Mw = g/mol
        * charge = e
        * Harmonic bonds : Kb = kcal/mol, b0 = Angstroms
        * Harmonic angles : Ktheta = kcal/mol/rad**2 , Theta0 = degrees
        * Dihedral angles: Ktheta = kcal/mol, n = interger (unitless), delta = degrees
        * Improper angles (currently unavailable) : TBD
        * Lennard-Jones (LJ)-NONBONDED: epsilon = kcal/mol, Rmin/2 = Angstroms
        * Mie-NONBONDED: epsilon = Kelvin, sigma = Angstroms, n = interger (unitless)
        * Buckingham-NONBONDED (currently unavailable): epsilon = Kelvin, sigma = Angstroms, n = interger (unitless)
        * Lennard-Jones (LJ)-NBFIX (currently unavailable) : epsilon = kcal/mol, Rmin = Angstroms
        * Mie-NBFIX (currently unavailable) : same as Mie-NONBONDED
        * Buckingham-NBFIX (currently unavailable) : same as Buckingham-NONBONDED

    Note: The units are only the same for GOMC and NAMD units, which using the Lennard-Jones (LJ) non-bonded type.

    Parameters
    ----------
    structure_box_0 : mbuild Compound object (mbuild.Compound) or mbuild Box object (mbuild.Box);
        If the structure has atoms/beads it must be an mbuild Compound.
        If the structure is empty it must be and mbuild Box object.
        Note: If 1 structures are provided (i.e., only structure_box_0),
        it must be an mbuild Compound.
        Note: If 2 structures are provided,
        only 1 structure can be an empty box (i.e., either structure_box_0 or structure_box_1)
    filename_box_0 : str
        The file name of the output file for structure_box_0.  Note: the extension should
        not be provided, as multiple extension (.pdb and .psf) are added to this name.
    structure_box_1 : mbuild Compound object (mbuild.Compound) or mbuild Box object (mbuild.Box), default = None;
        If the structure has atoms/beads it must be an mbuild Compound.
        Note: When running a GEMC or GCMC simulation the box 1 stucture should be input
        here.  Otherwise, there is no guarantee that any of the atom type and force field
        information will all work together correctly with box 0, if it is built separately.
        Note: If 2 structures are provided, only 1 structure can be an empty box
        (i.e., either structure_box_0 or structure_box_1).
    filename_box_1 : str , default = None
        The file name of the output file for structure_box_1 (Ex: for GCMC or GEMC simulations
        which have multiple simulation boxes).  Note: the extension should
        not be provided, as multiple extension (.pdb and .psf) are added to this name.
        Note: When running a GEMC or GCMC simulation the box 1 stucture should be input
        here. Otherwise, there is no guarantee that any of the atom type and force field
        information will all work together correctly with box 0, if it is built separately.
    residues : list, [str, ..., str]
        Labels of unique residues in the Compound. Residues are assigned by
        checking against Compound.name.  Only supply residue names as 4 character
        strings, as the residue names are truncated to 4 characters to fit in the
        psf and pdb file.
    forcefield_selection : str or dictionary, default = None
        Apply a forcefield to the output file by selecting a force field XML file with
        its path or by using the standard force field name provided the `foyer` package.
        Note: to write the NAMD/GOMC force field, pdb, and psf files, the
        residues and forcefields must be provided in a str or
        dictionary.  If a dictionary is provided all residues must
        be specified to a force field.
        * Example dict for FF file: {'ETH' : 'oplsaa.xml', 'OCT': 'path_to_file/trappe-ua.xml'}

        * Example str for FF file: 'path_to file/trappe-ua.xml'

        * Example dict for standard FF names : {'ETH' : 'oplsaa', 'OCT': 'trappe-ua'}

        * Example str for standard FF names: 'trappe-ua'

        * Example of a mixed dict with both : {'ETH' : 'oplsaa', 'OCT': 'path_to_file/'trappe-ua.xml'}
    ff_filename : str, default =None
        If a string, it will write the  force field files that work in
        GOMC and NAMD structures.
    gomc_fix_bonds_angles : list, default = None
        When list of residues is provided, the selected residues will have
        their bonds and angles fixed in the GOMC engine.  This is specifically
        for the GOMC engine and it changes the residue's bond constants (Kbs)
        and angle constants (Kthetas) values to 999999999999 in the
        FF file (i.e., the .inp file).
    bead_to_atom_name_dict : dict, optional, default =None
        For all atom names/elements/beads with 2 or less digits, this converts
        the atom name in the GOMC psf and pdb files to a unique atom name,
        provided they do not exceed 3844 atoms (62^2) of the same name/element/bead
        per residue. For all atom names/elements/beads with 3 digits, this converts
        the atom name in the GOMC psf and pdb files to a unique atom name,
        provided they do not exceed 62 of the same name/element pre residue.

        * Example dictionary: {'_CH3':'C', '_CH2':'C', '_CH':'C', '_HC':'C'}

        * Example name structure: {atom_type: first_part_pf atom name_without_numbering}

    fix_residue : list  or None, default = None
        Changes occcur in the pdb file only.
        When residues are listed here, all the atoms in the residue are
        fixed and can not move via setting the Beta values in the PDB file to 1.00.
        If neither fix_residue or fix_residue_in_box lists a
        residue or both equal None, then the Beta values for all the atoms
        in the residue are free to move in the simulation and Beta values
        in the PDB file is set to 0.00.
        NOTE that this is mainly for GOMC but also applies for NAMD (please see NAMD manual).
    fix_residue_in_box : list  or None, default = None
        Changes occcur in the pdb file only.
        When residues are listed here, all the atoms in the residue
        can move within the box but cannot be transferred between boxes
        via setting the Beta values in the PDB file to 2.00.
        If neither fix_residue or fix_residue_in_box lists a
        residue or both equal None, then the Beta values for all the atoms
        in the residue are free to move in the simulation and Beta values
        in the PDB file is set to 0.00.
        NOTE that this is mainly for GOMC but also applies for NAMD (please see NAMD manual).
    reorder_res_in_pdb_psf : bool, default =False
        If False, the order of of the atoms in the pdb file is kept in
        its original order, as in the Compound sent to the writer.
        If True, the order of the atoms is reordered based on their
        residue names in the 'residues' list that was entered.

    Attributes
    ----------
    input_error : bool
        This error is typically incurred from an error in the user's input values.
        However, it could also be due to a bug, provided the user is inputting
        the data as this Class intends.
    structure_box_0 : mbuild.compound.Compound
        The mbuild Compound for the input box 0
    structure_box_1 : mbuild.compound.Compound or None, default = None
        The mbuild Compound for the input box 1
    filename_box_0 : str
        The file name of the output file for structure_box_0.  Note: the extension should
        not be provided, as multiple extension (.pdb and .psf) are added to this name.
    filename_box_1 : str or None , default = None
        The file name of the output file for structure_box_1.  Note: the extension should
        not be provided, as multiple extension (.pdb and .psf) are added to this name.
        (i.e., either structure_box_0 or structure_box_1).
    residues : list, [str, ..., str]
        Labels of unique residues in the Compound. Residues are assigned by
        checking against Compound.name.  Only supply residue names as 4 character
        strings, as the residue names are truncated to 4 characters to fit in the
        psf and pdb file.
    forcefield_selection : str or dictionary, default = None
        Apply a forcefield to the output file by selecting a force field XML file with
        its path or by using the standard force field name provided the `foyer` package.
        Note: to write the NAMD/GOMC force field, pdb, and psf files, the
        residues and forcefields must be provided in a str or
        dictionary.  If a dictionary is provided all residues must
        be specified to a force field.

        * Example dict for FF file: {'ETH' : 'oplsaa.xml', 'OCT': 'path_to_file/trappe-ua.xml'}

        * Example str for FF file: 'path_to file/trappe-ua.xml'

        * Example dict for standard FF names : {'ETH' : 'oplsaa', 'OCT': 'trappe-ua'}

        * Example str for standard FF names: 'trappe-ua'

        * Example of a mixed dict with both : {'ETH' : 'oplsaa', 'OCT': 'path_to_file/'trappe-ua.xml'}

    ff_filename : str, default =None
        If a string, it will write the  force field files that work in
        GOMC and NAMD structures.
    gomc_fix_bonds_angles : list, default = None
        When list of residues is provided, the selected residues will have
        their bonds and angles fixed and will ignore the relative bond energies and
        related angle energies in the GOMC engine. Note that GOMC
        does not sample bond stretching. This is specifically
        for the GOMC engine and it changes the residue's bond constants (Kbs)
        and angle constants (Kthetas) values to 999999999999 in the
        FF file (i.e., the .inp file).
        If the residues are listed in either the gomc_fix_angles or the gomc_fix_bonds_angles
        lists, the angles will be fixed for that residue.
        If the residues are listed in either the gomc_fix_bonds or the gomc_fix_bonds_angles
        lists, the bonds will be fixed for that residue.
        NOTE if this option is utilized it may cause issues if using the FF file in NAMD.
    gomc_fix_bonds : list, default = None
        When list of residues is provided, the selected residues will have their
        relative bond energies ignored in the GOMC engine. Note that GOMC
        does not sample bond stretching. This is specifically
        for the GOMC engine and it changes the residue's bond constants (Kbs)
        values to 999999999999 in the FF file (i.e., the .inp file).
        If the residues are listed in either the gomc_fix_bonds or the gomc_fix_bonds_angles
        lists, the relative bond energy will be ignored.
        NOTE if this option is utilized it may cause issues if using the FF file in NAMD.
    gomc_fix_angles : list, default = None
        When list of residues is provided, the selected residues will have
        their angles fixed and will ignore the related angle energies in the GOMC engine.
        This is specifically for the GOMC engine and it changes the residue's angle
        constants (Kthetas) values to 999999999999 in the FF file (i.e., the .inp file),
        which fixes the angles and ignores related angle energy.
        If the residues are listed in either the gomc_fix_angles or the gomc_fix_bonds_angles
        lists, the angles will be fixed and the related angle energy will be ignored
        for that residue.
        NOTE if this option is utilized it may cause issues if using the FF file in NAMD.
    bead_to_atom_name_dict : dict, optional, default =None
        For all atom names/elements/beads with 2 or less digits, this converts
        the atom name in the GOMC psf and pdb files to a unique atom name,
        provided they do not exceed 3844 atoms (62^2) of the same name/element/bead
        per residue. For all atom names/elements/beads with 3 digits, this converts
        the atom name in the GOMC psf and pdb files to a unique atom name,
        provided they do not exceed 62 of the same name/element pre residue.

        * Example dictionary: {'_CH3':'C', '_CH2':'C', '_CH':'C', '_HC':'C'}

        * Example name structure: {atom_type: first_part_pf atom name_without_numbering}

    fix_residue : list  or None, default = None
        Changes occcur in the pdb file only.
        When residues are listed here, all the atoms in the residue are
        fixed and can not move via setting the Beta values in the PDB file to 1.00.
        If neither fix_residue or fix_residue_in_box lists a
        residue or both equal None, then the Beta values for all the atoms
        in the residue are free to move in the simulation and Beta values
        in the PDB file is set to 0.00.
        NOTE that this is mainly for GOMC but also applies for NAMD (please see NAMD manual).
    fix_residue_in_box : list  or None, default = None
        Changes occcur in the pdb file only.
        When residues are listed here, all the atoms in the residue
        can move within the box but cannot be transferred between boxes
        via setting the Beta values in the PDB file to 2.00.
        If neither fix_residue or fix_residue_in_box lists a
        residue or both equal None, then the Beta values for all the atoms
        in the residue are free to move in the simulation and Beta values
        in the PDB file is set to 0.00.
        NOTE that this is mainly for GOMC but also applies for NAMD (please see NAMD manual).
    reorder_res_in_pdb_psf : bool, default =False
        If False, the order of of the atoms in the pdb file is kept in
        its original order, as in the Compound sent to the writer.
        If True, the order of the atoms is reordered based on their
        residue names in the 'residues' list that was entered.
    box_0 : Box
        The Box class that contains the attributes Lx, Ly, Lz for the length
        of the box 0 (units in nanometers (nm)). It also contains the xy, xz, and yz Tilt factors
        needed to displace an orthogonal box's xy face to its
        parallelepiped structure for box 0.
    box_1 : Box
        The Box class that contains the attributes Lx, Ly, Lz for the length
        of the box 1 (units in nanometers (nm)). It also contains the xy, xz, and yz Tilt factors
        needed to displace an orthogonal box's xy face to its
        parallelepiped structure for box 0.
    box_0_vectors : numpy.ndarray, [[float, float, float], [float, float, float], [float, float, float]]
        Three (3) sets vectors for box 0 each with 3 float values, which represent
        the vectors for the Charmm-style systems (units in Angstroms (Ang))
    box_1_vectors : numpy.ndarray, [[float, float, float], [float, float, float], [float, float, float]]
        Three (3) sets vectors for box 1 each with 3 float values, which represent
        the vectors for the Charmm-style systems (units in Angstroms (Ang))
    topology_box_0_ff : gmso.Topology
        The box 0 topology (from structure_box_0) after all the provided
        force fields are applied.
    topology_box_1_ff : gmso.Topology
        The box 1 topology (from structure_box_1) after all the provided
        force fields are applied. This only exists if the box 1 structure
        (structure_box_1) is provided.
    residues_applied_list_box_0 : list
        The residues in box 0 that were found and had the force fields applied to them.
    residues_applied_list_box_1 : list
        The residues in box 1 that were found and had the force fields applied to them.
        This only exists if the box 1 structure (structure_box_1) is provided.
    boxes_for_simulation : int, [0, 1]
        The number of boxes used when writing the Charmm object and force fielding
        the system. If only box 0 is provided, the value is 0. If box 0 and box 1
        are provided, the value is 1.
    epsilon_kcal_per_mol_atom_type_dict : dict {str: float or int}
        The uniquely numbered atom type (key) and it's non-bonded epsilon coefficient in units
        of kcal/mol (value). The atom type is defined by the AtomClass_ResidueName
        (Example of a carbon atom in an ethane molecule, AtomClass_ResidueName --> CT_ETH).
    sigma_angstrom_atom_type_dict : dict {str: float or int}
        The uniquely numbered atom type (key) and it's non-bonded sigma coefficient in
        angstroms (value). The atom type is defined by the AtomClass_ResidueName
        (Example of a carbon atom in an ethane molecule, AtomClass_ResidueName --> CT_ETH).
    nonbonded_1_4_dict : dict {str: float or int}
        The uniquely numbered atom type (key) and it's non-bonded 1-4 scaling factor (value).
        The atom type is defined by the AtomClass_ResidueName
        (Example of a carbon atom in an ethane molecule, AtomClass_ResidueName --> CT_ETH).

        NOTE: NAMD and GOMC can have  multiple values for the LJ 1-4 scalers, since they are
        provided as an individual input for each atom type in the force field (.inp) file.
    electrostatic_1_4 : float or int
        The non-bonded 1-4 coulombic scaling factor, which is the same for all the
        residues/molecules, regardless if differenct force fields are utilized.  Note: if
        1-4 coulombic scaling factor is not the same for all molecules the Charmm object
        will fail with an error.

        NOTE: NAMD and GOMC can have  multiple values for the LJ 1-4 scalers, since they are
        provided as an individual input for each atom type in the force field (.inp) file.
    combined_1_4_electrostatic_dict_per_residue : dict, {str: float or int}
        The residue name/molecule (key) and it's non-bonded 1-4 coulombic scaling factor (value).

        Note: NAMD and GOMC can have  multiple values for the LJ 1-4 scalers, since they are
        provided as an individual input for each atom type in the force field (.inp) file.
    combined_combining_rule_dict_per_residue : dict, {str: str}
        The residue name/molecule (key) and it's combining or mixing rule (value).
        The combining or mixing rule is a either 'geometric' or 'lorentz'.

        NOTE: NAMD and GOMC can have  multiple values for the LJ 1-4 scalers, since they are
        provided as an individual input for each atom type in the force field (.inp) file.
    combining_rule : str ('geometric' or 'lorentz'"'),
        The combining or mixing rule which is used. The possible mixing/combining  rules are
        'geometric' or 'lorentz', which provide the  geometric and arithmetic mixing rule, respectively.

        NOTE: NAMD and GOMC can have  multiple values for the LJ 1-4 scalers, since they are
        provided as an individual input for each atom type in the force field (.inp) file.
    forcefield_selection : str or dictionary, default = None
        Apply a forcefield to the output file by selecting a force field XML file with
        its path or by using the standard force field name provided the `foyer` package.
        Note: to write the NAMD/GOMC force field, pdb, and psf files, the
        residues and forcefields must be provided in a str or
        dictionary.  If a dictionary is provided all residues must
        be specified to a force field.

        * Example dict for FF file: {'ETH' : 'oplsaa.xml', 'OCT': 'path_to_file/trappe-ua.xml'}

        * Example str for FF file: 'path_to file/trappe-ua.xml'

        * Example dict for standard FF names : {'ETH' : 'oplsaa', 'OCT': 'trappe-ua'}

        * Example str for standard FF names: 'trappe-ua'

        * Example of a mixed dict with both : {'ETH' : 'oplsaa', 'OCT': 'path_to_file/'trappe-ua.xml'}

    all_individual_atom_names_list : list
        A list of all the atom names for the combined structures
        (box 0 and box 1 (if supplied)), in order.
    all_residue_names_list : list
        A list of all the residue names for the combined structures
        (box 0 and box 1 (if supplied)), in order.
    max_residue_no : int
        The maximum number that the residue number will count to
        before restarting the counting back to 1, which is predetermined
        by the PDB format. This is a constant, which equals 9999
    max_resname_char : int
        The maximum number of characters allowed in the residue name,
        which is predetermined by the PDB format. This is a constant,
        which equals 4.
    all_res_unique_atom_name_dict : dict, {str : set(str, ..., str)}
        A dictionary that provides the residue names (keys) and a set
        of the unique atom names in the residue (value), for the
        combined structures (box 0 and box 1 (if supplied)).

    Notes
    -----
    Impropers, Urey-Bradleys, and NBFIX are not currenly supported.
    Currently the NBFIX is not available but will be in the near future.
    OPLS and CHARMM forcefield styles are supported (without impropers),
    AMBER forcefield styles are NOT supported.

    The atom typing is currently provided via a base 52 numbering (capital and lowercase lettering).
    This base 52 numbering allows for (52)^4 unique atom types.

    Unique atom names are provided if the system do not exceed 3844 atoms (62^2) of the same
    name/bead per residue (base 62 numbering). For all atom names/elements with 3 or less digits,
    this converts the atom name in the GOMC psf and pdb files to a unique atom name,
    provided they do not exceed 62 of the same name/element pre residue.

    Generating an empty box (i.e., pdb and psf files):
    Single Box system: Enter residues = [], but the accompanying structure (structure_box_0)
    must be an empty mb.Box. However, when doing this, the forcefield_selection
    must be supplied, or it will provide an error
    (i.e., forcefield_selection can not be equal to None).
    Dual Box System: Enter an empty mb.Box structure for either structure_box_0 or
    structure_box_1.

    In this current FF/psf/pdb writer, a residue type is essentially a molecule type.
    Therefore, it can only correctly write systems where every bead/atom in the molecule
    has the same residue name, and the residue name is specific to that molecule type.
    For example: a protein molecule with many residue names is not currently supported,
    but is planned to be supported in the future.
    """

    def __init__(
        self,
        structure_box_0,
        filename_box_0,
        structure_box_1=None,
        filename_box_1=None,
        forcefield_selection=None,
        residues=None,
        gomc_fix_bonds_angles=None,
        gomc_fix_bonds=None,
        gomc_fix_angles=None,
        bead_to_atom_name_dict=None,
        fix_residue=None,
        fix_residue_in_box=None,
        ff_filename=None,
        reorder_res_in_pdb_psf=False,
    ):

        # set all input variables to the class
        self.structure_box_0 = structure_box_0
        self.filename_box_0 = filename_box_0
        self.structure_box_1 = structure_box_1
        self.filename_box_1 = filename_box_1
        self.forcefield_selection = forcefield_selection
        self.residues = residues
        self.gomc_fix_bonds_angles = gomc_fix_bonds_angles
        self.gomc_fix_bonds = gomc_fix_bonds
        self.gomc_fix_angles = gomc_fix_angles
        self.bead_to_atom_name_dict = bead_to_atom_name_dict
        self.fix_residue = fix_residue
        self.fix_residue_in_box = fix_residue_in_box
        self.ff_filename = ff_filename
        self.reorder_res_in_pdb_psf = reorder_res_in_pdb_psf
        self.combining_rule = None

        # value to check for errors, with  self.input_error = True or False. Set to False initally
        self.input_error = False

        if not isinstance(self.structure_box_0, (Compound, Box)):
            self.input_error = True
            print_error_message = (
                "ERROR: The structure_box_0 expected to be of type: "
                "{} or {}, received: {}".format(
                    type(Compound()),
                    type(Box(lengths=[1, 1, 1])),
                    type(structure_box_0),
                )
            )
            raise TypeError(print_error_message)

        if self.structure_box_1 is not None and not isinstance(
            self.structure_box_1, (Compound, Box)
        ):
            self.input_error = True
            print_error_message = (
                "ERROR: The structure_box_1 expected to be of type: "
                "{} or {}, received: {}".format(
                    type(Compound()),
                    type(Box(lengths=[1, 1, 1])),
                    type(structure_box_1),
                )
            )
            raise TypeError(print_error_message)

        if isinstance(self.structure_box_0, Box) and isinstance(
            self.structure_box_1, Box
        ):
            self.input_error = True
            print_error_message = (
                "ERROR: Both structure_box_0 and structure_box_0 are empty Boxes {}. "
                "At least 1 structure must be an mbuild compound {} with 1 "
                "or more atoms in it".format(
                    type(Box(lengths=[1, 1, 1])), type(Compound())
                )
            )
            raise TypeError(print_error_message)

        if self.structure_box_1 is None and not isinstance(
            self.structure_box_0, Compound
        ):
            self.input_error = True
            print_error_message = (
                "ERROR: Only 1 structure is provided and it can not be an empty mbuild Box {}. "
                "it must be an mbuild compound {} with at least 1 "
                "or more atoms in it.".format(
                    type(Box(lengths=[1, 1, 1])), type(Compound())
                )
            )
            raise TypeError(print_error_message)

        if not isinstance(self.residues, list):
            self.input_error = True
            print_error_message = "ERROR: Please enter the residues list (residues) in a list format."
            raise TypeError(print_error_message)

        if isinstance(self.residues, list):
            for each_residue in self.residues:
                if not isinstance(each_residue, str):
                    self.input_error = True
                    print_error_message = "ERROR: Please enter a residues list (residues) with only string values."
                    raise TypeError(print_error_message)

        if self.residues is None:
            self.input_error = True
            print_error_message = (
                "ERROR: Please enter the residues list (residues)"
            )
            raise TypeError(print_error_message)
        if not isinstance(self.filename_box_0, str):
            self.input_error = True
            print_error_message = (
                "ERROR: Please enter the filename_box_0 as a string."
            )
            raise TypeError(print_error_message)

        unique_residue_test_name_list = []
        for res_m in range(0, len(self.residues)):
            if self.residues[res_m] not in unique_residue_test_name_list:
                unique_residue_test_name_list.append(self.residues[res_m])
        if len(unique_residue_test_name_list) != len(self.residues):
            self.input_error = True
            print_error_message = "ERROR: Please enter the residues list (residues) that has only unique residue names."
            raise ValueError(print_error_message)

        if self.filename_box_1 is not None and not isinstance(
            self.filename_box_1, str
        ):
            self.input_error = True
            print_error_message = (
                "ERROR: Please enter the filename_box_1 as a string."
            )
            raise TypeError(print_error_message)

        if self.ff_filename is not None:
            if not isinstance(self.ff_filename, str):
                self.input_error = True
                print_error_message = "ERROR: Please enter GOMC force field name (ff_filename) as a string."
                raise TypeError(print_error_message)
            if isinstance(self.ff_filename, str):
                extension_ff_name = os.path.splitext(self.ff_filename)[-1]
                if extension_ff_name == "":
                    self.ff_filename = self.ff_filename + ".inp"
                elif extension_ff_name == ".inp":
                    self.ff_filename = self.ff_filename + ""
                elif extension_ff_name != ".inp":
                    self.input_error = True
                    print_error_message = (
                        "ERROR: Please enter GOMC force field name without an "
                        "extention or the .inp extension."
                    )
                    raise ValueError(print_error_message)

        if self.forcefield_selection is not None:
            print(
                "write_gomcdata: forcefield_selection = "
                + str(self.forcefield_selection)
                + ", "
                + "residues = "
                + str(self.residues)
            )
            if not isinstance(self.forcefield_selection, (dict, str)):
                self.input_error = True
                print_error_message = (
                    "ERROR: The force field selection (forcefield_selection) "
                    "is not a string or a dictionary with all the residues specified "
                    'to a force field. -> String Ex: "path/trappe-ua.xml" or Ex: "trappe-ua" '
                    "Otherise provided a dictionary with all the residues specified "
                    "to a force field "
                    '->Dictionary Ex: {"Water" : "oplsaa", "OCT": "path/trappe-ua.xml"}, '
                    "Note: the file path must be specified the force field file if "
                    "a standard foyer force field is not used."
                )
                raise TypeError(print_error_message)

            if isinstance(self.forcefield_selection, str):
                ff_name = self.forcefield_selection
                self.forcefield_selection = {}
                for i in range(0, len(self.residues)):
                    self.forcefield_selection.update(
                        {self.residues[i]: ff_name}
                    )
                print(
                    "FF forcefield_selection = "
                    + str(self.forcefield_selection)
                )

        elif self.forcefield_selection is None:
            self.input_error = True
            print_error_message = "ERROR: Please enter the forcefield_selection as it was not provided."
            raise TypeError(print_error_message)

        if self.residues is not None and not isinstance(self.residues, list):
            self.input_error = True
            print_error_message = (
                "ERROR:  Please enter the residues (residues) in a list format"
            )
            raise TypeError(print_error_message)

        _check_fixed_bonds_angles_lists(
            self.gomc_fix_bonds_angles, "gomc_fix_bonds_angles", self.residues
        )
        _check_fixed_bonds_angles_lists(
            self.gomc_fix_bonds, "gomc_fix_bonds", self.residues
        )
        _check_fixed_bonds_angles_lists(
            self.gomc_fix_angles, "gomc_fix_angles", self.residues
        )

        if self.fix_residue is not None and not isinstance(
            self.fix_residue, list
        ):
            self.input_error = True
            print_error_message = (
                "ERROR: Please enter the fix_residue in a list format"
            )
            raise TypeError(print_error_message)

        if isinstance(self.fix_residue, list):
            for fix_residue_q in self.fix_residue:
                if fix_residue_q not in self.residues:
                    self.input_error = True
                    print_error_message = (
                        "Error: Please ensure that all the residue names in the fix_residue "
                        "list are also in the residues list."
                    )
                    raise ValueError(print_error_message)

        if self.fix_residue_in_box is not None and not isinstance(
            self.fix_residue_in_box, list
        ):
            self.input_error = True
            print_error_message = (
                "ERROR: Please enter the fix_residue_in_box in a list format."
            )
            raise TypeError(print_error_message)

        if isinstance(self.fix_residue_in_box, list):
            for fix_residue_in_box_q in self.fix_residue_in_box:
                if fix_residue_in_box_q not in self.residues:
                    self.input_error = True
                    print_error_message = (
                        "Error: Please ensure that all the residue names in the "
                        "fix_residue_in_box list are also in the residues list."
                    )
                    raise ValueError(print_error_message)

        if self.bead_to_atom_name_dict is not None and not isinstance(
            self.bead_to_atom_name_dict, dict
        ):
            self.input_error = True
            print_error_message = (
                "ERROR: Please enter the a bead type to atom in the dictionary "
                "(bead_to_atom_name_dict) so GOMC can properly evaluate the unique atom names"
            )
            raise TypeError(print_error_message)

        if isinstance(self.bead_to_atom_name_dict, dict):
            dict_list = []
            for key in self.bead_to_atom_name_dict.keys():
                dict_list.append(key)

            for dict_lis_i in dict_list:
                if not isinstance(dict_lis_i, str) or not isinstance(
                    self.bead_to_atom_name_dict[dict_lis_i], str
                ):
                    print_error_message = "ERROR: Please enter the bead_to_atom_name_dict with only string inputs."
                    raise TypeError(print_error_message)

        print("******************************")
        print("")

        if self.structure_box_1:
            self.boxes_for_simulation = 2
        else:
            self.boxes_for_simulation = 1

        # write the Force fields
        self.combined_1_4_nonbonded_dict_per_residue = {}
        self.combined_1_4_electrostatic_dict_per_residue = {}
        self.combined_combining_rule_dict_per_residue = {}

        self.atom_types_dict_per_residue = {}
        self.bond_types_dict_per_residue = {}
        self.angle_types_dict_per_residue = {}
        self.dihedral_types_dict_per_residue = {}
        self.improper_types_dict_per_residue = {}

        if self.structure_box_1:

            print(
                "GOMC FF writing each residues FF as a group for structure_box_0"
            )
            [
                self.topology_box_0_ff,
                self.residues_applied_list_box_0,
                self.electrostatics14Scale_dict_box_0,
                self.nonBonded14Scale_dict_box_0,
                self.atom_types_dict_box_0,
                self.bond_types_dict_box_0,
                self.angle_types_dict_box_0,
                self.dihedral_types_dict_box_0,
                self.improper_types_dict_box_0,
                self.combining_rule_dict_box_0,
            ] = specific_ff_to_residue(
                self.structure_box_0,
                forcefield_selection=self.forcefield_selection,
                residues=self.residues,
                reorder_res_in_pdb_psf=self.reorder_res_in_pdb_psf,
                boxes_for_simulation=self.boxes_for_simulation,
            )

            print(
                "GOMC FF writing each residues FF as a group for  structure_box_1"
            )
            [
                self.topology_box_1_ff,
                self.residues_applied_list_box_1,
                self.electrostatics14Scale_dict_box_1,
                self.nonBonded14Scale_dict_box_1,
                self.atom_types_dict_box_1,
                self.bond_types_dict_box_1,
                self.angle_types_dict_box_1,
                self.dihedral_types_dict_box_1,
                self.improper_types_dict_box_1,
                self.combining_rule_dict_box_1,
            ] = specific_ff_to_residue(
                self.structure_box_1,
                forcefield_selection=self.forcefield_selection,
                residues=self.residues,
                reorder_res_in_pdb_psf=self.reorder_res_in_pdb_psf,
                boxes_for_simulation=self.boxes_for_simulation,
            )

            # combine the topologies of box 0 and 1
            self.topology_box_0_and_1_ff = gmso.Topology()
            # iterate thru sites to combine the topologies of box 0 and 1
            for site_i in self.topology_box_0_ff.sites:
                self.topology_box_0_and_1_ff.add_site(site_i)
            for site_i in self.topology_box_1_ff.sites:
                self.topology_box_0_and_1_ff.add_site(site_i)
            # iterate thru connections (bonds, angles, dihedrals, and impropers) and add to empty topology
            # to combine the topologyies of box 0 and 1
            for connection_i in self.topology_box_0_ff.connections:
                self.topology_box_0_and_1_ff.add_connection(connection_i)
            for connection_i in self.topology_box_1_ff.connections:
                self.topology_box_0_and_1_ff.add_connection(connection_i)

            # create/add to alot of the dictionaries
            self.atom_types_dict_per_residue.update(
                self.atom_types_dict_box_0
            )
            self.atom_types_dict_per_residue.update(
                self.atom_types_dict_box_1
            )
            self.bond_types_dict_per_residue.update(
                self.bond_types_dict_box_0
            )
            self.bond_types_dict_per_residue.update(
                self.bond_types_dict_box_1
            )
            self.angle_types_dict_per_residue.update(
                self.angle_types_dict_box_0
            )
            self.angle_types_dict_per_residue.update(
                self.angle_types_dict_box_1
            )
            self.dihedral_types_dict_per_residue.update(
                self.dihedral_types_dict_box_0
            )
            self.dihedral_types_dict_per_residue.update(
                self.dihedral_types_dict_box_1
            )
            self.improper_types_dict_per_residue.update(
                self.improper_types_dict_box_0
            )
            self.improper_types_dict_per_residue.update(
                self.improper_types_dict_box_1
            )
            self.combined_1_4_nonbonded_dict_per_residue.update(
                self.nonBonded14Scale_dict_box_0
            )
            self.combined_1_4_nonbonded_dict_per_residue.update(
                self.nonBonded14Scale_dict_box_1
            )
            self.combined_1_4_electrostatic_dict_per_residue.update(
                self.electrostatics14Scale_dict_box_0
            )
            self.combined_1_4_electrostatic_dict_per_residue.update(
                self.electrostatics14Scale_dict_box_1
            )
            self.combined_combining_rule_dict_per_residue.update(
                self.combining_rule_dict_box_0
            )
            self.combined_combining_rule_dict_per_residue.update(
                self.combining_rule_dict_box_1
            )

            self.residues_applied_list_box_0_and_1 = (
                self.residues_applied_list_box_0
            )

            for res_iter in self.residues_applied_list_box_1:
                if res_iter not in self.residues_applied_list_box_0:
                    self.residues_applied_list_box_0_and_1.append(res_iter)

            for res_iter_0_1 in self.residues_applied_list_box_0_and_1:
                if res_iter_0_1 not in self.residues:
                    self.input_error = True
                    print_error_message = (
                        "ERROR: All the residues were not used from the forcefield_selection "
                        "string or dictionary.  There may be residues below other specified "
                        "residues in the mbuild.Compound hierarchy.  If so, the residues "
                        "acquire the residue's force fields, which is at the top of the "
                        "hierarchy.  Alternatively, residues that are not in the structure "
                        "may have been specified."
                    )
                    raise ValueError(print_error_message)

            for res_iter_0_1 in self.residues:
                if res_iter_0_1 not in self.residues_applied_list_box_0_and_1:
                    self.input_error = True
                    print_error_message = (
                        "ERROR: All the residues were not used from the forcefield_selection "
                        "string or dictionary.  There may be residues below other specified "
                        "residues in the mbuild.Compound hierarchy.  If so, the residues "
                        "acquire the residue's force fields, which is at the top of the "
                        "hierarchy.  Alternatively, residues that are not in the structure "
                        "may have been specified."
                    )
                    raise ValueError(print_error_message)


            # check that there are atoms in the system (checking box 0 and 1)
            site_list_box_0_and_1 = [site for site in self.topology_box_0_and_1_ff.sites
                                     ]
            if len(site_list_box_0_and_1) == 0:
                self.input_error = True
                print_error_message = (
                    "ERROR: the submitted structure has no PDB coordinates, "
                    "so the PDB writer has terminated. "
                )
                raise ValueError(print_error_message)

            # Check if the box 0's charges sum to zero
            charge_list_box_0 = [site.atom_type.__dict__['charge_'].to('C') / u.elementary_charge
                                 for site in self.topology_box_0_ff.sites
                                 ]
            if len(charge_list_box_0) != 0:
                total_charge_box_0 = sum(charge_list_box_0)
                total_charge_box_0 = total_charge_box_0.to_value('(dimensionless)')

                if round(total_charge_box_0, 6) != 0.0:
                    warn(
                        "System is not charge neutral for structure_box_0. "
                        "Total charge is {}.".format(total_charge_box_0)
                    )

            # Check if the box 1's charges sum to zero
            charge_list_box_1 = [site.atom_type.__dict__['charge_'].to('C') / u.elementary_charge
                                 for site in self.topology_box_1_ff.sites
                                 ]
            if len(charge_list_box_1) != 0:
                total_charge_box_1 = sum(charge_list_box_1)
                total_charge_box_1 = total_charge_box_1.to_value('(dimensionless)')

                if round(total_charge_box_1, 6) != 0.0:
                    warn(
                        "System is not charge neutral for structure_box_1. "
                        "Total charge is {}.".format(total_charge_box_1)
                    )

            # Check if the box 0 and 1's charges sum to zero
            charge_list_box_0_and_1 = [site.atom_type.__dict__['charge_'].to('C') / u.elementary_charge
                                       for site in self.topology_box_0_and_1_ff.sites
                                       ]
            total_charge_box_0_and_1 = sum(charge_list_box_0_and_1).to_value('(dimensionless)')
            if round(total_charge_box_0_and_1, 6) != 0.0:
                warn(
                    "System is not charge neutral for structure_0_and_1. "
                    "Total charge is {}.".format(total_charge_box_0_and_1)
                )

        else:
            print(
                "GOMC FF writing each residues FF as a group for structure_box_0"
            )
            [
                self.topology_box_0_ff,
                self.residues_applied_list_box_0,
                self.electrostatics14Scale_dict_box_0,
                self.nonBonded14Scale_dict_box_0,
                self.atom_types_dict_box_0,
                self.bond_types_dict_box_0,
                self.angle_types_dict_box_0,
                self.dihedral_types_dict_box_0,
                self.improper_types_dict_box_0,
                self.combining_rule_dict_box_0,
            ] = specific_ff_to_residue(
                self.structure_box_0,
                forcefield_selection=self.forcefield_selection,
                residues=self.residues,
                reorder_res_in_pdb_psf=self.reorder_res_in_pdb_psf,
                boxes_for_simulation=self.boxes_for_simulation,
            )

            self.atom_types_dict_per_residue.update(
                self.atom_types_dict_box_0
            )
            self.bond_types_dict_per_residue.update(
                self.bond_types_dict_box_0
            )
            self.angle_types_dict_per_residue.update(
                self.angle_types_dict_box_0
            )
            self.dihedral_types_dict_per_residue.update(
                self.dihedral_types_dict_box_0
            )
            self.improper_types_dict_per_residue.update(
                self.improper_types_dict_box_0
            )

            self.combined_1_4_nonbonded_dict_per_residue.update(
                self.nonBonded14Scale_dict_box_0
            )
            self.combined_1_4_electrostatic_dict_per_residue.update(
                self.electrostatics14Scale_dict_box_0
            )
            self.combined_combining_rule_dict_per_residue.update(
                self.combining_rule_dict_box_0
            )

            for res_iter_0 in self.residues_applied_list_box_0:
                if res_iter_0 not in self.residues:
                    self.input_error = True
                    print_error_message = (
                        "ERROR: All the residues were not used from the forcefield_selection "
                        "string or dictionary.  There may be residues below other specified "
                        "residues in the mbuild.Compound hierarchy.  If so, the residues "
                        "acquire the residue's force fields, which is at the top of the "
                        "hierarchy.  Alternatively, residues that are not in the structure "
                        "may have been specified."
                    )
                    raise ValueError(print_error_message)

            for res_iter_0 in self.residues:
                if res_iter_0 not in self.residues_applied_list_box_0:
                    self.input_error = True
                    print_error_message = (
                        "ERROR: All the residues were not used from the forcefield_selection "
                        "string or dictionary.  There may be residues below other specified "
                        "residues in the mbuild.Compound hierarchy.  If so, the residues "
                        "acquire the residue's force fields, which is at the top of the "
                        "hierarchy.  Alternatively, residues that are not in the structure "
                        "may have been specified."
                    )
                    raise ValueError(print_error_message)

            # check that there are atoms in the system (checking box 0 and 1)
            site_list_box_0 = [site for site in self.topology_box_0_ff.sites
                                     ]
            if len(site_list_box_0) == 0:
                self.input_error = True
                print_error_message = (
                    "ERROR: the submitted structure has no PDB coordinates, "
                    "so the PDB writer has terminated. "
                )
                raise ValueError(print_error_message)

            # Check if the box 0's charges sum to zero
            charge_list_box_0 = [site.atom_type.__dict__['charge_'].to('C') / u.elementary_charge
                                 for site in self.topology_box_0_ff.sites
                                 ]
            if len(charge_list_box_0) != 0:
                total_charge_box_0 = sum(charge_list_box_0)
                total_charge_box_0 = total_charge_box_0.to_value('(dimensionless)')

                if round(total_charge_box_0, 6) != 0.0:
                    warn(
                        "System is not charge neutral for structure_box_0. "
                        "Total charge is {}.".format(total_charge_box_0)
                    )

        print(
            "forcefield type from compound = " + str(self.forcefield_selection)
        )
        print(
            "coulomb14scale from compound = "
            + str(self.combined_1_4_electrostatic_dict_per_residue)
        )
        print(
            "nonbonded14scale from compound = "
            + str(self.combined_1_4_nonbonded_dict_per_residue)
        )

        # select all atoms in the or both boxes
        if self.structure_box_1:
            self.topology_selection = self.topology_box_0_and_1_ff
        else:
            self.topology_selection = self.topology_box_0_ff

        # get atom_type info
        # Example: {'ff_atom_name': {'atomclass': 'CT', 'description': 'alkane CH3',
        # 'definition': '[C;X4](C)(H)(H)H]', 'doi': 'doi.xxxx' }, ..., }
        self.atom_type_info_dict = {}
        for site in self.topology_selection.sites:
            key_iter = f"{site.atom_type.__dict__['name_']}_" \
                       f"{site.atom_type.__dict__['atomclass_']}_" \
                       f"{site.__dict__['residue_label_']}"
            if key_iter not in self.atom_type_info_dict.keys():
                charge_value = site.atom_type.__dict__['charge_'].to('C') / u.elementary_charge
                charge_value = charge_value.to_value('(dimensionless)')

                self.atom_type_info_dict.update(
                    {key_iter: {
                        #'tags': site.atom_type.__dict__['tags'],
                        'potential_expression_': site.atom_type.__dict__['potential_expression_'],
                        'mass_': site.atom_type.__dict__['mass_'].to('amu'),
                        'charge_': charge_value,
                        'atomclass_': site.atom_type.__dict__['atomclass_'],
                        'doi_': site.atom_type.__dict__['doi_'],
                        'overrides_': site.atom_type.__dict__['overrides_'],
                        'definition_': site.atom_type.__dict__['definition_'],
                        'description_': site.atom_type.__dict__['description_'],
                        'residue_label_': site.__dict__['residue_label_'],
                    }
                    }
                )

        # lock the atom_style and unit_style for GOMC. Can be inserted into variables
        # once more functionality is built in

        # change 'residue_label_' to "residue_name_"
        # change 'residue_index_' to "residue_number_"
        self.types = np.array(
            [   f"{site.atom_type.__dict__['name_']}_" \
                f"{site.atom_type.__dict__['atomclass_']}_" \
                f"{site.__dict__['residue_label_']}"
                for site in self.topology_selection.sites
            ]
        )

        self.unique_types = list(set(self.types))
        self.unique_types.sort(key=natural_sort)
        print('self.unique_types = ' +str(self.unique_types))

        self.classes = np.array(
            [
                f"{site.atom_type.__dict__['atomclass_']}_{site.__dict__['residue_label_']}"
                for site in self.topology_selection.sites
            ]
        )

        self.unique_classes = list(set(self.classes))
        self.unique_classes.sort(key=natural_sort)
        print('self.unique_classes = ' +str(self.unique_classes))

        # added an index so the atom classes can be converted to numbers as the type name is to long for insertion into
        # the pdb and psf files
        self.atom_class_to_index_value_dict = {}
        atom_class_numbering_iter = -1 # -1 means it will start at zero in the below for loop
        for unique_atom_class_name_iter in self.unique_classes:
            atom_class_numbering_iter += 1
            self.atom_class_to_index_value_dict.update(
                {unique_atom_class_name_iter: atom_class_numbering_iter}
            )

        self.masses = (
            np.array([site.atom_type.__dict__['mass_'].to_value('amu')
                      for site in self.topology_selection.sites])
        )

        self.mass_atom_type_dict = dict(
            [
                (atom_type, mass)
                for atom_type, mass in zip(self.types, self.masses)
            ]
        )


        self.mass_atom_class_dict = dict(
            [
                (atom_class, mass)
                for atom_class, mass in zip(self.classes, self.masses)
            ]
        )

        self.charges = (
            np.array([(site.atom_type.__dict__['charge_'].to('C') /
                      u.elementary_charge).to_value('(dimensionless)')
                      for site in self.topology_selection.sites])
        )
        self.charges_atom_type_dict = dict(
            [
                (atom_type, charge)
                for atom_type, charge in zip(self.types, self.charges)
            ]
        )

        # normalize by sigma
        self.box_0 = Box(
            lengths=np.array(
                [
                    (0.1 * val)
                    for val in self.topology_box_0_ff.box[0:3]
                ]
            ),
            angles=self.topology_box_0_ff.box[3:6],
        )

        # create box 0 vector list and convert from nm to Ang and round to 6 decimals.
        # note mbuild standard lengths are in nm, so round to 6+1 = 7 then mutlipy by 10
        box_0_lengths_ang = (
            self.box_0.lengths[0] * 10,
            self.box_0.lengths[1] * 10,
            self.box_0.lengths[2] * 10,
        )
        self.box_0_vectors = _lengths_angles_to_vectors(
            box_0_lengths_ang, self.box_0.angles, precision=6
        )

        # Internally use nm
        if self.structure_box_1:
            self.box_1 = Box(
                lengths=np.array(
                    [
                        (0.1 * val)
                        for val in self.topology_box_1_ff.box[0:3]
                    ]
                ),
                angles=self.topology_box_1_ff.box[3:6],
            )

            # create box 1 vector list and convert from nm to Ang and round to 6 decimals.
            # note mbuild standard lengths are in nm, so round to 6+1 = 7 then mutlipy by 10
            box_1_lengths_ang = (
                self.box_1.lengths[0] * 10,
                self.box_1.lengths[1] * 10,
                self.box_1.lengths[2] * 10,
            )
            self.box_1_vectors = _lengths_angles_to_vectors(
                box_1_lengths_ang, self.box_1.angles, precision=6
            )

        # need to add only the residue name
        residues_all_list = [
            site.__dict__['residue_label_'] for site in self.topology_selection.sites
        ]

        # Non-Bonded forces

        # Check if force fields have the same potential energy equations.
        # Only harmonic bond and angle potentials are accepted.
        # Impropers are not currently accepted

        # create dictionary to map atom type to a potential energy equation
        # GOMC standard LJ form = epsilon * ( (Rmin/r)**12 - 2*(Rmin/r)**6 )
        # --------------------> = 4*epsilon * ( (sigma/r)**12 - (sigma/r)**6 ), when converted to sigmas
        # Both forms above are accepted and compared automatically, but all input FFs have to be of the above
        # input forms, aside from the whole potential energy scaling factor.

        # GOMC standard Mie form = (n/(n-m)) * (n/m)**(m/(n-m)) * epsilon * ((sigma/r)**n - (sigma/r)**m)
        # where m = 6 --> (n/(n-6)) * (n/6)**(6/(n-6)) * epsilon * ((sigma/r)**n - (sigma/r)**6)
        # The above form is accepted but only if all input FFs have the same form,
        # aside from the whole potential energy scaling factor.

        # GOMC standard Buckingham (Exp-6) form =
        # alpha*epsilon/(alpha -6) * Exp( alpha*(1-r/Rmin) - (Rmin/r)**6 ), where r >= Rmax
        # infinity , where r < Rmax
        # The above form is accepted but only if all input FFs have the same form,
        # aside from the whole potential energy scaling factor.

        # Non-bonded potential energy (u).

        self.atom_type_experssion_and_scalar_combined = get_atom_type_expressions_and_scalars(
            self.atom_types_dict_per_residue)

        # find non-bonded expression to use
        all_NB_expression_forms_set = set()
        for atom_type_j in list(self.atom_type_experssion_and_scalar_combined.keys()):
            expression_form_iter = self.atom_type_experssion_and_scalar_combined[atom_type_j]['expression_form']
            all_NB_expression_forms_set.add(expression_form_iter)

        if len(all_NB_expression_forms_set) == 1:
            if 'LJ' in all_NB_expression_forms_set:
                self.utilized_NB_expression = 'LJ'
            elif 'Mie' in all_NB_expression_forms_set:
                self.utilized_NB_expression = 'Mie'
            elif 'Exp6' in all_NB_expression_forms_set:
                self.utilized_NB_expression = 'Exp6'
            else:
                raise ValueError("ERROR: The non-bonded equation type is not the LJ, Mie or Exp6 "
                                 "potential, which are the only available non-bonded equation potentials.")

        elif len(all_NB_expression_forms_set) == 2:
            if ('LJ' in all_NB_expression_forms_set) and ('Mie' in all_NB_expression_forms_set):
                self.utilized_NB_expression = 'Mie'

            elif 'Exp6' in all_NB_expression_forms_set:
                raise ValueError("ERROR: The 'Exp6' non-bonded equation type can not be used with the "
                                 "LJ or Mie potentials.")

        else:
            raise ValueError("ERROR: Only 1 or 2 differnt non-bonded equation types are supported at a time. "
                             "Only 'LJ' and 'Mie' potential combinations are allowed, and they change the "
                             "equation to the Mie potential form (see the GOMC manual).")

        epsilons_kcal_per_mol = (
            np.array([site.atom_type.parameters['epsilon'].to('kcal/mol', equivalence='thermal').to_value()
                      for site in self.topology_selection.sites])
        )
        sigmas_angstrom = (
            np.array([site.atom_type.parameters['sigma'].to('angstrom').to_value()
                      for site in self.topology_selection.sites])
        )

        if self.utilized_NB_expression == 'Mie':
            mie_n = []
            # The Mie m-constant must be six (m=6) for GOMC, per the general GMSO format
            # Therefore, we check if m=6 for all, and if not this writer will fail
            self.mie_m_required_value = 6

            for site in self.topology_selection.sites:
                atom_type_residue_iter = f"{site.atom_type.__dict__['name_']}_{site.__dict__['residue_label_']}"
                nonbonded_expresseion_iter = self.atom_type_experssion_and_scalar_combined[
                    atom_type_residue_iter]['expression_form']
                if nonbonded_expresseion_iter == 'Mie':
                    # This set the n parameter of the FF Mie if the iteration has it
                    mie_n_iter = site.atom_type.parameters['n'].to('dimensionless').to_value()
                    mie_n.append(mie_n_iter)

                    #check if m = 6 for all, and if not this writer will fail
                    if site.atom_type.parameters['m'].to('dimensionless').to_value() != self.mie_m_required_value:
                        print_error = f"ERROR: The Mie Potential atom class " \
                                      f"{site.atom_type.__dict__['atomclass_']}_" \
                                      f"{site.__dict__['residue_label_']} " \
                                      f"does not have an m-constant of 6 in the force field XML, " \
                                      f"which is required in GOMC and this file writer."
                        raise ValueError(print_error)
                elif nonbonded_expresseion_iter == 'LJ':
                    # ONly adding LJ here as it is the only other FF that currently works with Mie
                    mie_n_iter = 12
                    mie_n.append(mie_n_iter)

            mie_n = np.array(mie_n)

            self.mie_n_atom_type_dict = dict(
                [
                    (atom_type, mie_n_iter)
                    for atom_type, mie_n_iter in zip(self.types, mie_n)
                ]
            )
            self.mie_n_atom_class_dict = dict(
                [
                    (atom_class, mie_n_iter)
                    for atom_class, mie_n_iter in zip(self.classes, mie_n)
                ]
            )

        self.epsilon_kcal_per_mol_atom_type_dict = dict(
            [
                (atom_type, epsilon)
                for atom_type, epsilon in zip(self.types, epsilons_kcal_per_mol)
            ]
        )
        self.sigma_angstrom_atom_type_dict = dict(
            [
                (atom_type, sigma)
                for atom_type, sigma in zip(self.types, sigmas_angstrom)
            ]
        )

        self.epsilon_kcal_per_mol_atom_class_dict = dict(
            [
                (atom_class, epsilon)
                for atom_class, epsilon in zip(self.classes, epsilons_kcal_per_mol)
            ]
        )
        self.sigmas_angstrom_atom_class_dict = dict(
            [
                (atom_class, sigma)
                for atom_class, sigma in zip(self.classes, sigmas_angstrom)
            ]
        )

        # check to ensure all epsilons, sigmas, and masses with atom types in an atom class are the same
        for atom_type_k_iter in list(self.atom_type_info_dict.keys()):
            atom_class_k_iter = self.atom_type_info_dict[atom_type_k_iter]['atomclass_']

            # atom type values
            atom_type_residue_k_iter = atom_type_k_iter
            atom_type_epsilon_kcal_per_mol_k_iter = self.epsilon_kcal_per_mol_atom_type_dict[atom_type_residue_k_iter]
            atom_type_sigmas_angstrom_k_iter = self.sigma_angstrom_atom_type_dict[atom_type_residue_k_iter]
            atom_type_mass_amu_k_iter = self.mass_atom_type_dict[atom_type_residue_k_iter]

            # atom class values
            atom_class_k_iter = self.atom_type_info_dict[atom_type_k_iter]['atomclass_']
            atom_class_residue_k_iter = f"{atom_class_k_iter}_" \
                                        f"{self.atom_type_info_dict[atom_type_k_iter]['residue_label_']}"
            atom_class_number_k_iter = self.atom_class_to_index_value_dict[atom_class_residue_k_iter]
            atom_class_epsilon_kcal_per_mol_k_iter = self.epsilon_kcal_per_mol_atom_class_dict[atom_class_residue_k_iter]
            atom_class_sigmas_angstrom_k_iter = self.sigmas_angstrom_atom_class_dict[atom_class_residue_k_iter]
            atom_class_mass_amu_k_iter = self.mass_atom_class_dict[atom_class_residue_k_iter]

            if atom_class_epsilon_kcal_per_mol_k_iter != atom_type_epsilon_kcal_per_mol_k_iter:
                print_error = f"ERROR: Only the same epsilon values are permitted for an atom class. " \
                              f"The {atom_type_residue_k_iter} atom type has different epsilon values"
                raise ValueError(print_error)
            if atom_class_sigmas_angstrom_k_iter != atom_type_sigmas_angstrom_k_iter:
                print_error = f"ERROR: Only the same sigma values are permitted for an atom class. " \
                              f"The {atom_type_residue_k_iter} atom class has different sigma values"
                raise ValueError(print_error)
            if atom_class_mass_amu_k_iter != atom_type_mass_amu_k_iter:
                print_error = f"ERROR: Only the same atom mass values are permitted for an atom class. " \
                              f"The {atom_type_residue_k_iter} atom class has different atom mass values"
                raise ValueError(print_error)
            if self.utilized_NB_expression == 'Mie':
                atom_type_mie_n_k_iter = self.mie_n_atom_type_dict[atom_type_residue_k_iter]
                atom_class_mie_n_k_iter = self.mie_n_atom_class_dict[atom_class_residue_k_iter]

                if atom_class_mie_n_k_iter != atom_type_mie_n_k_iter:
                    print_error = f"ERROR: Only the same Mie n values are permitted for an atom class. " \
                                  f"The {atom_class_residue_k_iter} atom class has different n values"
                    raise ValueError(print_error)

        self.nonbonded_1_4_dict = dict(
            [
                (atom_class,
                 self.combined_1_4_nonbonded_dict_per_residue[residues_all_list],
                )
                for atom_class, residues_all_list in zip(
                    self.classes, residues_all_list
                )
            ]
        )

        # ensure all 1,4-coulombic or electrostatic scaling factors are the same,
        # and if not set to the same if None is provided
        electrostatic_1_4_set = set()
        for residue_p in self.combined_1_4_electrostatic_dict_per_residue.keys():
            if self.combined_1_4_electrostatic_dict_per_residue[residue_p] is None:
                warn("WARNING: The 1,4-electrostatic scaling factor for the {} residue "
                     "that was provided as None. This may mean that force field file "
                     "does not need the 1,4-electrostatic scaling factor, because it does not have 4 connected "
                     "atoms in the 1-4 configuration, or there may be an error in the "
                     "force field file. "
                     "".format(residue_p))
            else:
                electrostatic_1_4_set.add(self.combined_1_4_electrostatic_dict_per_residue[residue_p])
        if len(electrostatic_1_4_set) > 1:
            self.input_error = True
            print_error_message = (
                "ERROR: There are multiple 1,4-electrostatic scaling factors "
                "GOMC will only accept a singular input for the 1,4-electrostatic "
                "scaling factors."
            )
            raise ValueError(print_error_message)
        elif len(electrostatic_1_4_set) == 1:
            self.electrostatic_1_4 = list(electrostatic_1_4_set)[0]
        else:
            print_warning_message = (
                "WARNING: No 1,4-electrostatic scaling factors were provided, so "
                "it is being set to zero by default."
            )
            warn(print_warning_message)
            self.electrostatic_1_4 = 0

        # ensure all 1-4 nonbonded scaling factors are provided,
        # and if None is provided, it is set to 0 with a warning.
        for residue_r in self.combined_1_4_nonbonded_dict_per_residue.keys():
            if self.combined_1_4_nonbonded_dict_per_residue[residue_r] is None:
                warn("WARNING: The 1,4-nonbonded scaling factor for the {} residue "
                     "that was provided as None. This may mean that force field file "
                     "does not need the 1,4-nonbonded scaling factor, because it does not have 4 connected "
                     "atoms in the 1-4 configuration, or there may be an error in the force field file. "
                     "Since the {} residue provided None for the 1,4-nonbonded scaling factor, it "
                     "is being set to 0. "
                     "".format(residue_r, residue_r)
                     )
                self.combined_1_4_nonbonded_dict_per_residue[residue_r] = 0

        # ensure all the provided combining rules match,
        # and if None is provided, it is set to the only other matching rule with a warning.
        combining_rule_set = set()
        for residue_q in self.combined_combining_rule_dict_per_residue.keys():
            if self.combined_combining_rule_dict_per_residue[residue_q] is None:
                warn("WARNING: The combining or mixing rule for the {} residue "
                     "that was provided as None. This may mean that force field file "
                     "can use any combining or mixing rule, or there may be an error in the force field file. "
                     "Since the {} residue provided None for the combining or mixing rule, it "
                     "is being set to the same as all the other values. "
                     "".format(residue_q, residue_q)
                     )
            else:
                combining_rule_set.add(self.combined_combining_rule_dict_per_residue[residue_q])
        if len(combining_rule_set) > 1:
            self.input_error = True
            print_error_message = (
                "ERROR: There are multiple combining or mixing rules "
                "GOMC will only accept a singular input for the mixing rules. "
            )
            raise ValueError(print_error_message)
        elif len(combining_rule_set) == 1:
            self.combining_rule = list(combining_rule_set)[0]
        else:
            print_warning_message = (
                "WARNING: No combining or mixing rules were provided, so "
                "it is being set to 'lorentz' or arithmetic by default."
            )
            warn(print_warning_message)
            self.combining_rule = 'lorentz'

        # get all the unique atom name to check for the MEMC move in the gomc_conf_writer
        self.all_individual_atom_names_list = []
        self.all_residue_names_list = []
        if self.structure_box_1:
            list_of_topologies = [
                self.topology_box_0_ff,
                self.topology_box_1_ff,
            ]
            stuct_only = [self.topology_box_0_ff, self.topology_box_1_ff]
        else:
            list_of_topologies = [self.topology_box_0_ff]
            stuct_only = [self.topology_box_0_ff]

        for q_i in range(0, len(list_of_topologies)):
            stuct_only_iteration = stuct_only[q_i]

            # caluculate the atom name and unique atom names
            residue_names_list = []
            residue_id_list = []
            for site in stuct_only_iteration.sites:
                residue_id_list.append(
                    site.__dict__['residue_index_']
                )
                residue_names_list.append(
                    site.__dict__['residue_label_']
                )

            # this sets the residues chain length to a max limit
            self.max_residue_no = 9999
            self.max_resname_char = 4

            res_no_chain_iter_corrected = []
            for residue_id_int in residue_id_list:
                res_id_adder = int(
                    (residue_id_int % self.max_residue_no) % self.max_residue_no
                )
                if int(res_id_adder) == 0:
                    res_no_iteration_corrected = int(self.max_residue_no)
                else:
                    res_no_iteration_corrected = res_id_adder

                res_no_chain_iter_corrected.append(res_no_iteration_corrected)

            # This converts the atom name in the GOMC psf and pdb files to unique atom names
            [
                unique_individual_atom_names_dict_iter,
                individual_atom_names_list_iter,
                missing_bead_to_atom_name_iter,
            ] = unique_atom_naming(
                stuct_only_iteration,
                residue_id_list,
                residue_names_list,
                bead_to_atom_name_dict=self.bead_to_atom_name_dict,
            )

            if q_i == 0:
                self.all_individual_atom_names_list = (
                    individual_atom_names_list_iter
                )

                self.all_residue_names_list = residue_names_list
            else:

                self.all_individual_atom_names_list = (
                    self.all_individual_atom_names_list
                    + individual_atom_names_list_iter
                )

                self.all_residue_names_list = (
                    self.all_residue_names_list + residue_names_list
                )

        # put the self.all_individual_atom_names_list and self.all_residue_names_list in a list to match
        # the the atom name with a residue and find the unique matches
        if None in [
            unique_individual_atom_names_dict_iter,
            individual_atom_names_list_iter,
            missing_bead_to_atom_name_iter,
        ]:
            self.input_error = True
            print_error_message = (
                "ERROR: The unique_atom_naming function failed while "
                "running the charmm_writer function. Ensure the proper inputs are "
                "in the bead_to_atom_name_dict."
            )
            raise ValueError(print_error_message)

        else:
            self.all_res_unique_atom_name_dict = {}
            for res_i in range(0, len(self.all_individual_atom_names_list)):
                if self.all_res_unique_atom_name_dict \
                     in list(self.all_res_unique_atom_name_dict.keys()):
                    self.all_res_unique_atom_name_dict.setdefault(
                        self.all_residue_names_list[res_i], set()
                    ).add(self.all_individual_atom_names_list[res_i])

                else:
                    self.all_res_unique_atom_name_dict.setdefault(
                        self.all_residue_names_list[res_i], set()
                    ).add(self.all_individual_atom_names_list[res_i])

        print(
            "all_res_unique_atom_name_dict = {}".format(
                self.all_res_unique_atom_name_dict
            )
        )



    def write_inp(self):
        """This write_inp function writes the Charmm style parameter (force field) file, which can be utilized
        in the GOMC and NAMD engines."""
        print("******************************")
        print("")
        print(
            "The charmm force field file writer (the write_inp function) is running"
        )

        if self.ff_filename is None:
            self.input_error = True
            print_error_message = (
                "ERROR: The force field file name was not specified and in the "
                "Charmm object. "
                "Therefore, the force field file (.inp) can not be written. "
                "Please use the force field file name when building the Charmm object, "
                "then use the write_inp function."
            )
            raise TypeError(print_error_message)
        else:

            print("******************************")
            print("")
            print(
                "The charmm force field file writer (the write_inp function) is running"
            )
            print("******************************")
            print("")
            print("writing the GOMC force field file ")
            date_time = datetime.datetime.today()

            residue_id_list = []
            residue_names_list = []

            if self.structure_box_1:
                for k, site in enumerate(self.topology_box_0_ff.sites):
                    residue_id_list.append(site.__dict__['residue_index_'])
                    residue_names_list.append(site.__dict__['residue_label_'])

                for k, site in enumerate(self.topology_box_1_ff.sites):
                    residue_id_list.append(site.__dict__['residue_index_'])
                    residue_names_list.append(site.__dict__['residue_label_'])

            else:
                for k, site in enumerate(self.topology_box_0_ff.sites):
                    residue_id_list.append(site.__dict__['residue_index_'])
                    residue_names_list.append(site.__dict__['residue_label_'])

            for n in range(0, len(residue_names_list)):
                if residue_names_list[n] not in self.residues:
                    print(
                        "residue_names_list = "
                        + str(residue_names_list)
                    )
                    self.input_error = True
                    print_error_message = "ERROR: Please specifiy all residues (residues) in a list"
                    raise ValueError(print_error_message)

            # Start writing the force field (.inp) file
            with open(self.ff_filename, "w") as data:

                if self.structure_box_1:
                    data.write(
                        f"* {self.filename_box_0} and {self.filename_box_1} "
                        f"- created by mBuild using the on {date_time}\n"
                    )
                else:
                    data.write(
                        f"* {self.filename_box_0} - created by mBuild using the on {date_time}.\n"
                    )

                data.write(
                    f"* These parameters use the non-bonded {self.utilized_NB_expression} form "
                    f"--- with these force field(s) via MoSDef  {self.forcefield_selection}.\n"
                )
                data.write(
                    f"*  1-4 electrostatic scaling = {self.combined_1_4_electrostatic_dict_per_residue} "
                    f", and 1-4 non-bonded scaling = {self.combined_1_4_nonbonded_dict_per_residue}"
                    f", and non-bonded mixing rule = {self.combined_combining_rule_dict_per_residue}\n\n"
                )
                data.write(
                    "* {:15d} atoms\n".format(self.topology_selection.n_sites)
                )

                data.write("* {:15d} bonds\n".format(self.topology_selection.n_bonds))
                data.write("* {:15d} angles\n".format(self.topology_selection.n_angles))
                data.write(
                    "* {:15d} dihedrals\n".format(self.topology_selection.n_dihedrals)
                )
                data.write(
                    "* {:15d} impropers\n\n".format(self.topology_selection.n_impropers)
                )

                data.write(
                    "* {:15d} atom types\n".format(len(self.topology_selection.atom_types))
                )

                data.write(
                    "* {:15d} bond types\n".format(len(self.topology_selection.bond_types))
                    )
                data.write(
                    "* {:15d} angle types\n".format(len(self.topology_selection.angle_types))
                )
                data.write(
                    "* {:15d} dihedral types\n".format(len(self.topology_selection.dihedral_types))
                )
                data.write(
                    "* {:15d} improper types\n".format(len(self.topology_selection.improper_types))
                )
                data.write("\n")

                data.write("\n* masses\n\n")
                data.write("! {:15s} {:15s} ! {}\n".format(
                    'atom_types',
                    'mass',
                    'atomClass_ResidueName',
                )
                )

                atom_mass_decimals_round = 4
                for atom_class, mass in self.mass_atom_class_dict.items():
                    mass_format = "* {:15s} {:15s} ! {:25s}\n"
                    data.write(
                        mass_format.format(
                            base10_to_base52_alph(self.atom_class_to_index_value_dict[atom_class]
                            ),
                            str(np.round(mass, decimals=atom_mass_decimals_round)),
                            atom_class,
                        )
                    )


                # Bond coefficients
                if len(self.topology_selection.bond_types) > 0:
                    data.write("\n")
                    data.write("BONDS * harmonic\n")
                    data.write("! \n")
                    data.write("! V(bond) = Kb(b - b0)**2\n")
                    data.write("! \n")
                    data.write("! Kb: kcal/mol/A**2 (LJ) and  K/A**2 (Mie and Exp6)\n")
                    data.write("! b0: A\n")
                    data.write(
                        "! Kb (kcal/mol) = Kb_K (K) * Boltz. const.; (9999999999 if no stretching)\n"
                    )
                    data.write("! \n")
                    data.write(
                        "! {:8s} {:10s} {:15s} {:15s} ! {:20s} {:20s}\n".format(
                            "type_1",
                            "type_2",
                            "Kb",
                            "b0",
                            "extended_type_1",
                            "extended_type_2"
                        )
                    )

                    bond_k_kcal_per_mol_round_decimals = 6
                    bond_k_Kelvin_round_decimals = 4
                    bond_distance_round_decimals = 6

                    for res_x in self.bond_types_dict_per_residue.keys():
                        if self.bond_types_dict_per_residue[res_x] is not None:
                            for bond_type_x in self.bond_types_dict_per_residue[res_x]['bond_types']:
                                if bond_type_x.member_classes is not None:
                                    bond_members_iter = bond_type_x.member_classes
                                elif bond_type_x.member_types is not None:
                                    bond_members_iter = bond_type_x.member_types
                                    raise TypeError(f"ERROR: The {res_x} residue has a least one bond member_types "
                                                    f"that is not None. "
                                                    f"Currently, the Charmm writer only supports the member_class "
                                                    f"designations for bonds."
                                                    )
                                else:
                                    raise TypeError(f"ERROR: The {res_x} residue has a least one bond member_types "
                                                    f"and member_classes that is not None. "
                                                    f"There may be an issue with the XML file."
                                                    )

                                # ***************************
                                # ***************************
                                # ***************************
                                # ***************************
                                # need to set eqn back to "1 * k * (r-r_eq)**2", as it is wrong not to fix the
                                # bug which the foyer to GMSO conversion does not include the 1/2 constant
                                # (start)
                                # ***************************
                                # ***************************
                                # ***************************
                                # ***************************

                                gomc_bond_form_with_scalar = "2 * k * (r-r_eq)**2"

                                # ***************************
                                # ***************************
                                # ***************************
                                # ***************************
                                # need to set eqn back to "1 * k * (r-r_eq)**2", as it is wrong not to fix the
                                # bug which the foyer to GMSO conversion does not include the 1/2 constant
                                # (end)
                                # ***************************
                                # ***************************
                                # ***************************
                                # ***************************

                                [bond_type_str_iter, bond_eqn_scalar_iter] = \
                                    evaluate_harmonic_bond_format_with_scaler(
                                    bond_type_x.expression,
                                    gomc_bond_form_with_scalar
                                )

                                if bond_type_str_iter != 'HarmonicBondPotential':
                                    raise TypeError(f"ERROR: The {res_x} residue does not have a "
                                                    f"{'HarmonicBondPotential'} bond potential, which "
                                                    f"is the only supported bond potential.")

                                # the bond energy value is dependant of the non-bonded form
                                bond_energy_value_LJ_units_iter = _check_convert_bond_k_constant_units(
                                    str(bond_type_x.member_classes),
                                    bond_eqn_scalar_iter * bond_type_x.parameters['k'],
                                    'kcal/mol/angstrom**2',
                                ).to_value('kcal/mol/angstrom**2')

                                if self.utilized_NB_expression == 'LJ':
                                    bond_energy_value_iter = np.round(bond_energy_value_LJ_units_iter,
                                                                      decimals=bond_k_kcal_per_mol_round_decimals
                                                                      )
                                elif self.utilized_NB_expression in ['Mie', 'Exp6']:
                                    bond_energy_value_Mie_Exp6_units_iter = _check_convert_bond_k_constant_units(
                                        str(bond_type_x.member_classes),
                                        bond_eqn_scalar_iter * bond_type_x.parameters['k'],
                                        'K/angstrom**2',
                                    ).to_value('K/angstrom**2')
                                    bond_energy_value_iter = np.round(bond_energy_value_Mie_Exp6_units_iter,
                                                                      decimals=bond_k_Kelvin_round_decimals
                                                                      )

                                bond_format = "{:10s} {:10s} {:15s} {:15s} ! {:20s} {:20s}\n"

                                if (
                                    (self.gomc_fix_bonds_angles is not None)
                                    and (
                                            str(res_x)
                                            in self.gomc_fix_bonds_angles
                                    )
                                ) or (
                                    (
                                    (self.gomc_fix_bonds is not None)
                                    and (
                                            str(res_x)
                                            in self.gomc_fix_bonds
                                    )
                                    )
                                ):
                                    fix_bond_k_value = "999999999999"
                                    data.write(
                                        bond_format.format(
                                            base10_to_base52_alph(
                                                self.atom_class_to_index_value_dict[
                                                    f"{bond_members_iter[0]}_{res_x}"
                                                ]
                                            ),
                                            base10_to_base52_alph(
                                                self.atom_class_to_index_value_dict[
                                                    f"{bond_members_iter[1]}_{res_x}"
                                                ]
                                            ),
                                            str(fix_bond_k_value),
                                            str(np.round(bond_type_x.parameters['r_eq'].to_value('angstrom'),
                                                     decimals=bond_distance_round_decimals)
                                                ),
                                            f"{bond_members_iter[0]}_{res_x}",
                                            f"{bond_members_iter[1]}_{res_x}",
                                        )
                                    )

                                else:
                                    data.write(
                                        bond_format.format(
                                            base10_to_base52_alph(
                                                self.atom_class_to_index_value_dict[
                                                    f"{bond_members_iter[0]}_{res_x}"
                                                ]
                                            ),
                                            base10_to_base52_alph(
                                                self.atom_class_to_index_value_dict[
                                                    f"{bond_members_iter[1]}_{res_x}"
                                                ]
                                            ),
                                            str(bond_energy_value_iter),
                                            str(np.round(bond_type_x.parameters['r_eq'].to_value('angstrom'),
                                                     decimals=bond_distance_round_decimals)
                                                ),
                                            f"{bond_members_iter[0]}_{res_x}",
                                            f"{bond_members_iter[1]}_{res_x}",
                                        )
                                    )

                # Angle coefficients
                if len(self.topology_selection.angle_types):
                    data.write("\nANGLES * harmonic\n")
                    data.write("! \n")
                    data.write("! V(angle) = Ktheta(Theta - Theta0)**2\n")
                    data.write("! \n")
                    data.write("! Ktheta: kcal/mol/rad**2 (LJ) and  K/rad**2 (Mie and Exp6)\n")
                    data.write("! Theta0: degrees\n")
                    data.write("! \n")
                    data.write("!  Boltzmann = 0.0019872041 kcal / (mol * K)\n")
                    data.write("! \n")
                    data.write(
                        "! Ktheta (kcal/mol) = Ktheta_K (K) * Boltz. const.\n"
                    )
                    data.write("! \n")
                    data.write(
                        "! {:8s} {:10s} {:10s} {:15s} {:15s} ! {:20s} {:20s} {:20s}\n".format(
                            "type_1",
                            "type_2",
                            "type_3",
                            "Ktheta",
                            "Theta0",
                            "extended_type_1",
                            "extended_type_2",
                            "extended_type_3"
                        )
                    )

                    angle_k_kcal_per_mol_round_decimals = 6
                    angle_k_Kelvin_round_decimals = 4
                    angle_degree_round_decimals = 6

                    for res_x in self.angle_types_dict_per_residue.keys():
                        if self.angle_types_dict_per_residue[res_x] is not None:
                            for angle_type_x in self.angle_types_dict_per_residue[res_x]['angle_types']:
                                if angle_type_x.member_classes is not None:
                                    angle_members_iter = angle_type_x.member_classes
                                elif angle_type_x.member_types is not None:
                                    angle_members_iter = angle_type_x.member_types
                                    raise TypeError(f"ERROR: The {res_x} residue has a least one angle member_types "
                                                    f"that is not None. "
                                                    f"Currently, the Charmm writer only supports the member_class "
                                                    f"designations for angles."
                                                    )
                                else:
                                    raise TypeError(f"ERROR: The {res_x} residue has a least one angle member_types "
                                                    f"and member_classes that is not None. "
                                                    f"There may be an issue with the XML file."
                                                    )
                                # ***************************
                                # ***************************
                                # ***************************
                                # ***************************
                                # need to set eqn back to "1 * k * (theta - theta_eq)**2", as it is wrong not to fix the
                                # bug which the foyer to GMSO conversion does not include the 1/2 constant
                                # (start)
                                # ***************************
                                # ***************************
                                # ***************************
                                # ***************************

                                gomc_angle_form_with_scalar = "2 * k * (theta - theta_eq)**2"

                                # ***************************
                                # ***************************
                                # ***************************
                                # ***************************
                                # need to set eqn back to "1 * k * (theta - theta_eq)**2", as it is wrong not to fix the
                                # bug which the foyer to GMSO conversion does not include the 1/2 constant
                                # (end)
                                # ***************************
                                # ***************************
                                # ***************************
                                # ***************************

                                [angle_type_str_iter, angle_eqn_scalar_iter] = \
                                    evaluate_harmonic_angle_format_with_scaler(
                                    angle_type_x.expression,
                                    gomc_angle_form_with_scalar
                                )

                                if angle_type_str_iter != 'HarmonicAnglePotential':
                                    raise TypeError(f"ERROR: The {res_x} residue does not have a "
                                                    f"{'HarmonicAnglePotential'} angle potential, which "
                                                    f"is the only supported angle potential.")

                                # the angle energy value is dependant of the non-bonded form
                                angle_energy_value_LJ_units_iter = _check_convert_angle_k_constant_units(
                                    str(angle_type_x.member_classes),
                                    angle_eqn_scalar_iter * angle_type_x.parameters['k'],
                                    'kcal/mol/rad**2',
                                ).to_value('kcal/mol/rad**2')

                                if self.utilized_NB_expression == 'LJ':
                                    angle_energy_value_iter = np.round(angle_energy_value_LJ_units_iter,
                                                                  decimals=angle_k_kcal_per_mol_round_decimals
                                                                  )
                                elif self.utilized_NB_expression in ['Mie', 'Exp6']:
                                    angle_energy_value_Mie_Exp6_units_iter = _check_convert_angle_k_constant_units(
                                        str(angle_type_x.member_classes),
                                        angle_eqn_scalar_iter * angle_type_x.parameters['k'],
                                        'K/rad**2',
                                    ).to_value('K/rad**2')

                                    angle_energy_value_iter = np.round(angle_energy_value_Mie_Exp6_units_iter,
                                                                       decimals=angle_k_Kelvin_round_decimals
                                                                       )

                                angle_format = "{:10s} {:10s} {:10s} {:15s} {:15s} ! {:20s} {:20s} {:20s}\n"

                                if (
                                    (self.gomc_fix_bonds_angles is not None)
                                    and (
                                            str(res_x)
                                            in self.gomc_fix_bonds_angles
                                    )
                                ) or (
                                    (
                                        (self.gomc_fix_angles is not None)
                                        and (
                                                str(res_x)
                                                in self.gomc_fix_angles
                                        )
                                    )
                                ):
                                    fix_angle_k_value = "999999999999"
                                    data.write(
                                        angle_format.format(
                                            base10_to_base52_alph(
                                                self.atom_class_to_index_value_dict[
                                                    f"{angle_members_iter[0]}_{res_x}"
                                                ]
                                            ),
                                            base10_to_base52_alph(
                                                self.atom_class_to_index_value_dict[
                                                    f"{angle_members_iter[1]}_{res_x}"
                                                ]
                                            ),
                                            base10_to_base52_alph(
                                                self.atom_class_to_index_value_dict[
                                                    f"{angle_members_iter[2]}_{res_x}"
                                                ]
                                            ),
                                            str(fix_angle_k_value),
                                            str(np.round(angle_type_x.parameters['theta_eq'].to_value('degree'),
                                                     decimals=angle_degree_round_decimals)
                                                ),
                                            f"{angle_members_iter[0]}_{res_x}",
                                            f"{angle_members_iter[1]}_{res_x}",
                                            f"{angle_members_iter[2]}_{res_x}",
                                        )
                                    )

                                else:
                                    data.write(
                                        angle_format.format(
                                            base10_to_base52_alph(
                                                self.atom_class_to_index_value_dict[
                                                    f"{angle_members_iter[0]}_{res_x}"
                                                ]
                                            ),
                                            base10_to_base52_alph(
                                                self.atom_class_to_index_value_dict[
                                                    f"{angle_members_iter[1]}_{res_x}"
                                                ]
                                            ),
                                            base10_to_base52_alph(
                                                self.atom_class_to_index_value_dict[
                                                    f"{angle_members_iter[2]}_{res_x}"
                                                ]
                                            ),
                                            str(angle_energy_value_iter),
                                            str(np.round(angle_type_x.parameters['theta_eq'].to_value('degree'),
                                                     decimals=angle_degree_round_decimals)
                                                ),
                                            f"{angle_members_iter[0]}_{res_x}",
                                            f"{angle_members_iter[1]}_{res_x}",
                                            f"{angle_members_iter[2]}_{res_x}",
                                        )
                                    )

                # Dihedral coefficients
                if len(self.topology_selection.dihedral_types):
                    list_if_large_error_dihedral_overall = []

                    list_if_largest_error_abs_values_for_dihedral_overall = []
                    list_dihedral_overall_error = []

                    list_if_abs_max_values_for_dihedral_overall = []
                    list_dihedral_atoms_all_dihedral_overall = []

                    data.write("\nDIHEDRALS * CHARMM\n")
                    data.write("! \n")
                    data.write(
                        "! V(dihedral) = Kchi(1 + cos(n(chi) - delta))\n"
                    )
                    data.write("! \n")
                    data.write("! Kchi: kcal/mol (LJ) and K/mol (Mie and Exp6)\n")
                    data.write("! n: multiplicity\n")
                    data.write("! delta: degrees\n")
                    data.write("! \n")
                    data.write("! Kchi (kcal/mol) = Kchi_K (K) * Boltz. const.\n")
                    data.write("! Boltzmann = 0.0019872041 kcal / (mol * K)\n")
                    data.write("! \n")
                    data.write(
                        "! {:8s} {:10s} {:10s} {:10s} {:15s} {:10s} {:15s} ! {:20s} {:20s} {:20s} {:20s}\n".format(
                            "type_1",
                            "type_2",
                            "type_3",
                            "type_4",
                            "Kchi",
                            "n",
                            "delta",
                            "extended_type_1",
                            "extended_type_2",
                            "extended_type_3",
                            "extended_type_4"
                        )
                    )

                dihedral_k_kcal_per_mol_round_decimals = 6
                dihedral_k_Kelvin_round_decimals = 4
                dihedral_phase_degree_round_decimals = 6

                no_pi = np.pi
                dihedral_steps = 5 * 10 ** (-3)
                dihedral_range = 2 * no_pi
                dihedral_no_steps = (
                        int(dihedral_range / dihedral_steps) + 1
                )

                for res_x in self.dihedral_types_dict_per_residue.keys():
                    if self.dihedral_types_dict_per_residue[res_x] is not None:
                        for dihedral_type_x in self.dihedral_types_dict_per_residue[res_x]['dihedral_types']:
                            if dihedral_type_x.member_classes is not None:
                                dihedral_members_iter = dihedral_type_x.member_classes
                            elif dihedral_type_x.member_types is not None:
                                dihedral_members_iter = dihedral_type_x.member_types
                                raise TypeError(f"ERROR: The {res_x} residue has a least one dihedral member_types "
                                                f"that is not None. "
                                                f"Currently, the Charmm writer only supports the member_class "
                                                f"designations for dihedrals."
                                                )
                            else:
                                raise TypeError(f"ERROR: The {res_x} residue has a least one dihderal member_types "
                                                f"and member_classes that is not None. "
                                                f"There may be an issue with the XML file."
                                                )

                            dihedral_type_str_iter = None
                            dihedral_eqn_scalar_iter = None
                            if dihedral_type_str_iter is None and dihedral_eqn_scalar_iter is None:
                                # Check if OPLSTorsionPotential
                                OPLSTorsionPotential_form_with_scalar = "0.5 * k0 + " \
                                                                        "0.5 * k1 * (1 + cos(phi)) + " \
                                                                        "0.5 * k2 * (1 - cos(2*phi)) + " \
                                                                        "0.5 * k3 * (1 + cos(3*phi)) + " \
                                                                        "0.5 * k4 * (1 - cos(4*phi))"
                                [dihedral_type_str_iter, dihedral_eqn_scalar_iter] = \
                                    evaluate_OPLS_torsion_format_with_scaler(
                                        dihedral_type_x.expression,
                                        OPLSTorsionPotential_form_with_scalar
                                    )

                            if dihedral_type_str_iter is None and dihedral_eqn_scalar_iter is None:
                                # Check if OPLSTorsionPotential
                                RyckaertBellemansTorsionPotential_form_with_scalar = "c0 * cos(phi)**0 + " \
                                                                                     "c1 * cos(phi)**1 + " \
                                                                                     "c2 * cos(phi)**2 + " \
                                                                                     "c3 * cos(phi)**3 + " \
                                                                                     "c4 * cos(phi)**4 + " \
                                                                                     "c5 * cos(phi)**5"
                                [dihedral_type_str_iter, dihedral_eqn_scalar_iter] = \
                                    evaluate_RB_torsion_format_with_scaler(
                                        dihedral_type_x.expression,
                                        RyckaertBellemansTorsionPotential_form_with_scalar
                                    )

                            if dihedral_type_str_iter is None and dihedral_eqn_scalar_iter is None:
                                # Check if OPLSTorsionPotential
                                PeriodicTorsionPotential_form_with_scalar = "k0 + " \
                                                                            "k1 * (1 + cos(1 * phi - phi_eq1)) + " \
                                                                            "k2 * (1 + cos(2 * phi - phi_eq2)) + " \
                                                                            "k3 * (1 + cos(3 * phi - phi_eq3)) + " \
                                                                            "k4 * (1 + cos(4 * phi - phi_eq4)) + " \
                                                                            "k5 * (1 + cos(5 * phi - phi_eq5))"
                                [dihedral_type_str_iter, dihedral_eqn_scalar_iter] = \
                                    evaluate_periodic_torsion_format_with_scaler(
                                        dihedral_type_x.expression,
                                        PeriodicTorsionPotential_form_with_scalar
                                    )

                            if dihedral_type_str_iter is None and dihedral_eqn_scalar_iter is None:
                                # Check if HarmonicTorsionPotential
                                HarmonicTorsionPotential_form_with_scalar = "0.5 * k * (phi - phi_eq)**2"
                                [dihedral_type_str_iter, dihedral_eqn_scalar_iter] = \
                                    evaluate_harmonic_torsion_format_with_scaler(
                                        dihedral_type_x.expression,
                                        HarmonicTorsionPotential_form_with_scalar
                                    )

                            if dihedral_type_str_iter == 'HarmonicTorsionPotential':
                                raise TypeError(f"ERROR: The {res_x} residue has a "
                                                f"{'HarmonicTorsionPotential'} torsion potential, which "
                                                f"is not currently supported in this writer."
                                                )
                            elif dihedral_type_str_iter is None and dihedral_eqn_scalar_iter is None:
                                raise TypeError(f"ERROR: The {res_x} residue and associated force field "
                                                f"has at least one unsupported dihdedral. "
                                                f"The only supported dihedrals are {'HarmonicTorsionPotential'}, "
                                                f"{'OPLSTorsionPotential'}, {'PeriodicTorsionPotential'}, and "
                                                f"{'RyckaertBellemansTorsionPotential'}."
                                                )

                            # convert dihedral to CHARMM style
                            if dihedral_type_str_iter == 'OPLSTorsionPotential':
                                f0 = dihedral_eqn_scalar_iter * dihedral_type_x.parameters[
                                    'k0'].to_value('kcal/mol', equivalence='thermal')
                                f1 = dihedral_eqn_scalar_iter * dihedral_type_x.parameters[
                                    'k1'].to_value('kcal/mol', equivalence='thermal')
                                f2 = dihedral_eqn_scalar_iter * dihedral_type_x.parameters[
                                    'k2'].to_value('kcal/mol', equivalence='thermal')
                                f3 = dihedral_eqn_scalar_iter * dihedral_type_x.parameters[
                                    'k3'].to_value('kcal/mol', equivalence='thermal')
                                f4 = dihedral_eqn_scalar_iter * dihedral_type_x.parameters[
                                    'k4'].to_value('kcal/mol', equivalence='thermal')

                                [[K0, n0, d0],
                                 [K1, n1, d1],
                                 [K2, n2, d2],
                                 [K3, n3, d3],
                                 [K4, n4, d4],
                                 [K5, n5, d5]] = OPLS_to_CHARMM(f0, f1, f2, f3, f4)

                            elif dihedral_type_str_iter == 'RyckaertBellemansTorsionPotential':
                                c0 = dihedral_eqn_scalar_iter * dihedral_type_x.parameters[
                                    'c0'].to_value('kcal/mol', equivalence='thermal')
                                c1 = dihedral_eqn_scalar_iter * dihedral_type_x.parameters[
                                    'c1'].to_value('kcal/mol', equivalence='thermal')
                                c2 = dihedral_eqn_scalar_iter * dihedral_type_x.parameters[
                                    'c2'].to_value('kcal/mol', equivalence='thermal')
                                c3 = dihedral_eqn_scalar_iter * dihedral_type_x.parameters[
                                    'c3'].to_value('kcal/mol', equivalence='thermal')
                                c4 = dihedral_eqn_scalar_iter * dihedral_type_x.parameters[
                                    'c4'].to_value('kcal/mol', equivalence='thermal')
                                c5 = dihedral_eqn_scalar_iter * dihedral_type_x.parameters[
                                    'c5'].to_value('kcal/mol', equivalence='thermal')

                                [[K0, n0, d0],
                                 [K1, n1, d1],
                                 [K2, n2, d2],
                                 [K3, n3, d3],
                                 [K4, n4, d4],
                                 [K5, n5, d5]] = RB_to_CHARMM(c0, c1, c2, c3, c4, c5)

                            elif dihedral_type_str_iter == 'PeriodicTorsionPotential':
                                K0_input = dihedral_eqn_scalar_iter * dihedral_type_x.parameters[
                                    'k0'].to_value('kcal/mol', equivalence='thermal')

                                K1_input = dihedral_eqn_scalar_iter * dihedral_type_x.parameters[
                                    'k1'].to_value('kcal/mol', equivalence='thermal')
                                d1_input = dihedral_type_x.parameters[
                                    'phi_eq1'].to_value('degree')

                                K2_input =dihedral_eqn_scalar_iter * dihedral_type_x.parameters[
                                    'k2'].to_value('kcal/mol', equivalence='thermal')
                                d2_input = dihedral_type_x.parameters[
                                    'phi_eq2'].to_value('degree')

                                K3_input = dihedral_eqn_scalar_iter * dihedral_type_x.parameters[
                                    'k3'].to_value('kcal/mol', equivalence='thermal')
                                d3_input = dihedral_type_x.parameters[
                                    'phi_eq3'].to_value('degree')

                                K4_input = dihedral_eqn_scalar_iter * dihedral_type_x.parameters[
                                    'k4'].to_value('kcal/mol', equivalence='thermal')
                                d4_input = dihedral_type_x.parameters[
                                    'phi_eq4'].to_value('degree')

                                K5_input = dihedral_eqn_scalar_iter * dihedral_type_x.parameters[
                                    'k5'].to_value('kcal/mol', equivalence='thermal')
                                d5_input = dihedral_type_x.parameters[
                                    'phi_eq5'].to_value('degree')

                                K0 = K0_input
                                n0 = 0
                                d0 = 90.0

                                K1 = K1_input
                                n1 = 1
                                d1 = d1_input

                                K2 = K2_input
                                n2 = 2
                                d2 = d2_input

                                K3 = K3_input
                                n3 = 3
                                d3 = d3_input

                                K4 = K4_input
                                n4 = 4
                                d4 = d4_input

                                K5 = K5_input
                                n5 = 5
                                d5 = d5_input


                            # test dihedral conversion for errors
                            input_dihedral_to_charmm_abs_diff = []
                            for i in range(0, dihedral_no_steps + 1):
                                phi = i * dihedral_steps
                                psi = phi - no_pi

                                # calulate the charmm dihedral (PeriodicTorsionPotential)
                                charmm_dihedral_calc = (
                                   K0 * (1 + np.cos(n0 * phi - d0 * no_pi / 180)) +
                                   K1 * (1 + np.cos(n1 * phi - d1 * no_pi / 180)) +
                                   K2 * (1 + np.cos(n2 * phi - d2 * no_pi / 180)) +
                                   K3 * (1 + np.cos(n3 * phi - d3 * no_pi / 180)) +
                                   K4 * (1 + np.cos(n4 * phi - d4 * no_pi / 180)) +
                                   K5 * (1 + np.cos(n5 * phi - d5 * no_pi / 180))
                                )

                                if dihedral_type_str_iter == 'OPLSTorsionPotential':
                                    input_dihedral_calc = (
                                            f0 / 2 +
                                            f1 / 2 * (1 + np.cos(1 * phi)) +
                                            f2 / 2 * (1 - np.cos(2 * phi)) +
                                            f3 / 2 * (1 + np.cos(3 * phi)) +
                                            f4 / 2 * (1 - np.cos(4 * phi))

                                    )

                                elif dihedral_type_str_iter == 'RyckaertBellemansTorsionPotential':
                                    input_dihedral_calc = (
                                            c0 +
                                            c1 * np.cos(psi)**1 +
                                            c2 * np.cos(psi)**2 +
                                            c3 * np.cos(psi)**3 +
                                            c4 * np.cos(psi)**4 +
                                            c5 * np.cos(psi)**5
                                    )

                                elif dihedral_type_str_iter == 'PeriodicTorsionPotential':
                                    input_dihedral_calc = (
                                        K0_input +
                                        K1_input * (1 + np.cos(1 * phi - d1_input * no_pi / 180)) +
                                        K2_input * (1 + np.cos(2 * phi - d2_input * no_pi / 180)) +
                                        K3_input * (1 + np.cos(3 * phi - d3_input * no_pi / 180)) +
                                        K4_input * (1 + np.cos(4 * phi - d4_input * no_pi / 180)) +
                                        K5_input * (1 + np.cos(5 * phi - d5_input * no_pi / 180))
                                    )

                                input_to_charmm_absolute_difference = np.absolute(
                                            input_dihedral_calc - charmm_dihedral_calc
                                )

                                input_dihedral_to_charmm_abs_diff.append(input_to_charmm_absolute_difference)
                                list_if_large_error_dihedral_iteration = []
                                list_abs_max_dihedral_iteration = []

                                if max(input_dihedral_to_charmm_abs_diff) > 10 ** (-10):
                                    list_if_large_error_dihedral_iteration.append(1)
                                    list_abs_max_dihedral_iteration.append(
                                        max(input_dihedral_to_charmm_abs_diff)
                                    )

                                    list_if_large_error_dihedral_overall.append(1)
                                    list_if_largest_error_abs_values_for_dihedral_overall.append(
                                        max(input_dihedral_to_charmm_abs_diff))

                                    list_dihedral_overall_error.append(
                                        f"{dihedral_members_iter[0]}_{res_x}, "
                                        f"{dihedral_members_iter[1]}_{res_x}, "
                                        f"{dihedral_members_iter[2]}_{res_x}, "
                                        f"{dihedral_members_iter[3]}_{res_x}, "
                                    )

                                else:
                                    list_if_large_error_dihedral_iteration.append(0)
                                    list_if_abs_max_values_for_dihedral_overall.append(
                                        max(input_dihedral_to_charmm_abs_diff))
                                    list_dihedral_atoms_all_dihedral_overall.append(
                                        f"{dihedral_members_iter[0]}_{res_x}, "
                                        f"{dihedral_members_iter[1]}_{res_x}, "
                                        f"{dihedral_members_iter[2]}_{res_x}, "
                                        f"{dihedral_members_iter[3]}_{res_x}, "
                                    )

                            if self.utilized_NB_expression == 'LJ':
                                K0_output_energy_iter = np.round(K0, decimals=dihedral_k_kcal_per_mol_round_decimals)
                                K1_output_energy_iter = np.round(K1, decimals=dihedral_k_kcal_per_mol_round_decimals)
                                K2_output_energy_iter = np.round(K2, decimals=dihedral_k_kcal_per_mol_round_decimals)
                                K3_output_energy_iter = np.round(K3, decimals=dihedral_k_kcal_per_mol_round_decimals)
                                K4_output_energy_iter = np.round(K4, decimals=dihedral_k_kcal_per_mol_round_decimals)
                                K5_output_energy_iter = np.round(K5, decimals=dihedral_k_kcal_per_mol_round_decimals)

                            elif self.utilized_NB_expression in ['Mie', 'Exp6']:
                                K0_output_energy_iter = np.round(
                                    u.unyt_quantity(K0, 'kcal/mol').to_value('K', equivalence='thermal'),
                                    decimals=dihedral_k_Kelvin_round_decimals
                                )
                                K1_output_energy_iter = np.round(
                                    u.unyt_quantity(K1, 'kcal/mol').to_value('K', equivalence='thermal'),
                                    decimals=dihedral_k_Kelvin_round_decimals
                                )
                                K2_output_energy_iter = np.round(
                                    u.unyt_quantity(K2, 'kcal/mol').to_value('K', equivalence='thermal'),
                                    decimals=dihedral_k_Kelvin_round_decimals
                                )
                                K3_output_energy_iter = np.round(
                                    u.unyt_quantity(K3, 'kcal/mol').to_value('K', equivalence='thermal'),
                                    decimals=dihedral_k_Kelvin_round_decimals
                                )
                                K4_output_energy_iter = np.round(
                                    u.unyt_quantity(K4, 'kcal/mol').to_value('K', equivalence='thermal'),
                                    decimals=dihedral_k_Kelvin_round_decimals
                                )
                                K5_output_energy_iter = np.round(
                                    u.unyt_quantity(K5, 'kcal/mol').to_value('K', equivalence='thermal'),
                                    decimals=dihedral_k_Kelvin_round_decimals
                                )

                            # **************************************
                            # check the error between the convertions of RB_tortions to CHARMM DIHEDRALS (end)
                            # **************************************
                            dihedral_format = "{:10s} {:10s} {:10s} {:10s} {:15s} {:10s} {:15s} " \
                                              "! {:20s} {:20s} {:20s} {:20s}\n"

                            # write charmm dihedral K0 (zero order dihedral --- a constant) if Mie or Exp6,
                            # but not written for CHARMM as the K0 constant is defined as a
                            # harmonic function in CHARMM.
                            if self.utilized_NB_expression in ['Mie', 'Exp6']:
                                data.write(
                                    dihedral_format.format(
                                        base10_to_base52_alph(
                                            self.atom_class_to_index_value_dict[
                                                f"{dihedral_members_iter[0]}_{res_x}"
                                                ]
                                        ),
                                        base10_to_base52_alph(
                                            self.atom_class_to_index_value_dict[
                                                f"{dihedral_members_iter[1]}_{res_x}"
                                                ]
                                        ),
                                        base10_to_base52_alph(
                                            self.atom_class_to_index_value_dict[
                                                f"{dihedral_members_iter[2]}_{res_x}"
                                                ]
                                        ),
                                        base10_to_base52_alph(
                                            self.atom_class_to_index_value_dict[
                                                f"{dihedral_members_iter[3]}_{res_x}"
                                                ]
                                        ),
                                        str(K0_output_energy_iter),
                                        str(int(n0)),
                                        str(np.round(d0, decimals=dihedral_phase_degree_round_decimals)),
                                        f"{dihedral_members_iter[0]}_{res_x}",
                                        f"{dihedral_members_iter[1]}_{res_x}",
                                        f"{dihedral_members_iter[2]}_{res_x}",
                                        f"{dihedral_members_iter[3]}_{res_x}",
                                    )
                                )

                            # write charmm dihedral K1 (first order dihedral)
                            data.write(
                                dihedral_format.format(
                                    base10_to_base52_alph(
                                        self.atom_class_to_index_value_dict[
                                            f"{dihedral_members_iter[0]}_{res_x}"
                                        ]
                                    ),
                                    base10_to_base52_alph(
                                        self.atom_class_to_index_value_dict[
                                            f"{dihedral_members_iter[1]}_{res_x}"
                                        ]
                                    ),
                                    base10_to_base52_alph(
                                        self.atom_class_to_index_value_dict[
                                            f"{dihedral_members_iter[2]}_{res_x}"
                                        ]
                                    ),
                                    base10_to_base52_alph(
                                        self.atom_class_to_index_value_dict[
                                            f"{dihedral_members_iter[3]}_{res_x}"
                                        ]
                                    ),
                                    str(K1_output_energy_iter),
                                    str(int(n1)),
                                    str(np.round(d1, decimals=dihedral_phase_degree_round_decimals)),
                                    f"{dihedral_members_iter[0]}_{res_x}",
                                    f"{dihedral_members_iter[1]}_{res_x}",
                                    f"{dihedral_members_iter[2]}_{res_x}",
                                    f"{dihedral_members_iter[3]}_{res_x}",
                                )
                            )

                            # write charmm dihedral K2 (second order dihedral)
                            data.write(
                                dihedral_format.format(
                                    base10_to_base52_alph(
                                        self.atom_class_to_index_value_dict[
                                            f"{dihedral_members_iter[0]}_{res_x}"
                                        ]
                                    ),
                                    base10_to_base52_alph(
                                        self.atom_class_to_index_value_dict[
                                            f"{dihedral_members_iter[1]}_{res_x}"
                                        ]
                                    ),
                                    base10_to_base52_alph(
                                        self.atom_class_to_index_value_dict[
                                            f"{dihedral_members_iter[2]}_{res_x}"
                                        ]
                                    ),
                                    base10_to_base52_alph(
                                        self.atom_class_to_index_value_dict[
                                            f"{dihedral_members_iter[3]}_{res_x}"
                                        ]
                                    ),
                                    str(K2_output_energy_iter),
                                    str(int(n2)),
                                    str(np.round(d2, decimals=dihedral_phase_degree_round_decimals)),
                                    f"{dihedral_members_iter[0]}_{res_x}",
                                    f"{dihedral_members_iter[1]}_{res_x}",
                                    f"{dihedral_members_iter[2]}_{res_x}",
                                    f"{dihedral_members_iter[3]}_{res_x}",
                                )
                            )

                            # write charmm dihedral K3 (third order dihedral)
                            data.write(
                                dihedral_format.format(
                                    base10_to_base52_alph(
                                        self.atom_class_to_index_value_dict[
                                            f"{dihedral_members_iter[0]}_{res_x}"
                                        ]
                                    ),
                                    base10_to_base52_alph(
                                        self.atom_class_to_index_value_dict[
                                            f"{dihedral_members_iter[1]}_{res_x}"
                                        ]
                                    ),
                                    base10_to_base52_alph(
                                        self.atom_class_to_index_value_dict[
                                            f"{dihedral_members_iter[2]}_{res_x}"
                                        ]
                                    ),
                                    base10_to_base52_alph(
                                        self.atom_class_to_index_value_dict[
                                            f"{dihedral_members_iter[3]}_{res_x}"
                                        ]
                                    ),
                                    str(K3_output_energy_iter),
                                    str(int(n3)),
                                    str(np.round(d3, decimals=dihedral_phase_degree_round_decimals)),
                                    f"{dihedral_members_iter[0]}_{res_x}",
                                    f"{dihedral_members_iter[1]}_{res_x}",
                                    f"{dihedral_members_iter[2]}_{res_x}",
                                    f"{dihedral_members_iter[3]}_{res_x}",
                                )
                            )

                            # write charmm dihedral K4 (first order dihedral)
                            data.write(
                                dihedral_format.format(
                                    base10_to_base52_alph(
                                        self.atom_class_to_index_value_dict[
                                            f"{dihedral_members_iter[0]}_{res_x}"
                                        ]
                                    ),
                                    base10_to_base52_alph(
                                        self.atom_class_to_index_value_dict[
                                            f"{dihedral_members_iter[1]}_{res_x}"
                                        ]
                                    ),
                                    base10_to_base52_alph(
                                        self.atom_class_to_index_value_dict[
                                            f"{dihedral_members_iter[2]}_{res_x}"
                                        ]
                                    ),
                                    base10_to_base52_alph(
                                        self.atom_class_to_index_value_dict[
                                            f"{dihedral_members_iter[3]}_{res_x}"
                                        ]
                                    ),
                                    str(K4_output_energy_iter),
                                    str(int(n4)),
                                    str(np.round(d4, decimals=dihedral_phase_degree_round_decimals)),
                                    f"{dihedral_members_iter[0]}_{res_x}",
                                    f"{dihedral_members_iter[1]}_{res_x}",
                                    f"{dihedral_members_iter[2]}_{res_x}",
                                    f"{dihedral_members_iter[3]}_{res_x}",
                                )
                            )

                            # write charmm dihedral K5 (fifth order dihedral)
                            data.write(
                                dihedral_format.format(
                                    base10_to_base52_alph(
                                        self.atom_class_to_index_value_dict[
                                            f"{dihedral_members_iter[0]}_{res_x}"
                                        ]
                                    ),
                                    base10_to_base52_alph(
                                        self.atom_class_to_index_value_dict[
                                            f"{dihedral_members_iter[1]}_{res_x}"
                                        ]
                                    ),
                                    base10_to_base52_alph(
                                        self.atom_class_to_index_value_dict[
                                            f"{dihedral_members_iter[2]}_{res_x}"
                                        ]
                                    ),
                                    base10_to_base52_alph(
                                        self.atom_class_to_index_value_dict[
                                            f"{dihedral_members_iter[3]}_{res_x}"
                                        ]
                                    ),
                                    str(K5_output_energy_iter),
                                    str(int(n5)),
                                    str(np.round(d5, decimals=dihedral_phase_degree_round_decimals)),
                                    f"{dihedral_members_iter[0]}_{res_x}",
                                    f"{dihedral_members_iter[1]}_{res_x}",
                                    f"{dihedral_members_iter[2]}_{res_x}",
                                    f"{dihedral_members_iter[3]}_{res_x}",
                                )
                            )

                if len(self.topology_selection.dihedral_types):
                    if sum(list_if_large_error_dihedral_overall) > 0:
                        list_max_error_abs_dihedral_overall = max(
                            list_if_largest_error_abs_values_for_dihedral_overall
                        )
                        info_if_dihedral_error_too_large = (
                                f"! WARNING: The input dihedral type(s) to "
                                f"CHARMM dihedral conversion error"
                                f" is to large [error > 10^(-10)] \n"
                                f"! WARNING: Maximum( "
                                f"|(input dihedral calc)-(CHARMM dihedral calc)| ) =  "
                                f"{list_max_error_abs_dihedral_overall}\n"
                        )
                        warn(info_if_dihedral_error_too_large
                        )
                        data.write(info_if_dihedral_error_too_large)
                        print(info_if_dihedral_error_too_large)
                    else:
                        list_if_abs_max_values_for_dihedral_overall_max = (
                            max(list_if_abs_max_values_for_dihedral_overall)
                        )
                        info_if_dihedral_error_ok = (
                                f"! The input dihedral to CHARMM dihedral conversion error is OK "
                                f"[error <= 10^(-10)]\n"
                                f"! Maximum( |(input dihedral calc)-(CHARMM dihedral calc)| ) =  "
                                f"{list_if_abs_max_values_for_dihedral_overall_max}\n"
                        )
                        data.write(info_if_dihedral_error_ok)
                        print(info_if_dihedral_error_ok)


                # Improper coefficients
                if len(self.topology_selection.improper_types):
                    data.write(
                        "! ERROR: Impropers were found, however, "
                        "this writer currently does not support impropers. "
                        "Also, GOMC is not currently able to use improper in its calculations."
                    )

                # Pair coefficients
                print(
                    "NBFIX_Mixing not used or no mixing used for the non-bonded potentials out"
                )
                print("self.utilized_NB_expression = " + str(self.utilized_NB_expression))

                epsilon_kcal_per_mol_round_decimals = 10
                epsilon_Kelvin_round_decimals = 6
                sigma_round_decimals = 10
                mie_n_round_decimals = 8

                if self.utilized_NB_expression == 'LJ':
                    data.write("\n")
                    data.write("NONBONDED\n")
                    data.write("! \n")
                    data.write(
                        "! V(Lennard-Jones) = Epsilon,i,j[(Rmin,i,j/ri,j)**12 - 2(Rmin,i,j/ri,j)**6]\n"
                        "                                    or"
                        "! V(Lennard-Jones) = 4 * Epsilon,i,j[(Sigma,i,j/ri,j)**12 - (Sigma,i,j/ri,j)**6]\n"
                    )
                    data.write("! \n")

                    data.write("! {:8s} {:15s} {:15s} {:15s} {:15s} {:15s} {:15s} ! {:20s} {:20s}\n"
                               "".format("type_1",
                                         "ignored",
                                         "epsilon",
                                         "Rmin/2",
                                         "ignored",
                                         "epsilon,1-4",
                                         "Rmin/2,1-4",
                                         "extended_type_1",
                                         "extended_type_2",
                                         )
                               )

                    for class_x, epsilon_kcal_per_mol in self.epsilon_kcal_per_mol_atom_class_dict.items():
                        nb_format = "{:10s} {:15s} {:15s} {:15s} {:15s} {:15s} {:15s} ! {:20s} {:20s}\n"

                        # if the 1-4 non-bonded scalar is used.
                        # If 1-4 non-bonded scalar = 0, all 1-4 non-bonded values are set to zero (0).
                        # If 1-4 non-bonded scalar = 1, the epsilon 1-4 non-bonded interaction is scaled
                        # via another scalar.
                        if float(self.nonbonded_1_4_dict[class_x]) == 0:
                            scalar_used_binary = 0
                            scalar_sign_change = -1
                        else:
                            scalar_used_binary = 1
                            scalar_sign_change = 1

                        data.write(
                            nb_format.format(
                                str(base10_to_base52_alph(self.atom_class_to_index_value_dict[class_x])),
                                str(0.0),
                                str(np.round(-epsilon_kcal_per_mol,
                                             decimals=epsilon_kcal_per_mol_round_decimals)
                                    ),
                                str(np.round(_LJ_sigma_to_r_min_div_2(self.sigmas_angstrom_atom_class_dict[class_x]),
                                              decimals=sigma_round_decimals)
                                     ),
                                str(0.0),
                                str(np.round(scalar_used_binary *
                                             float(self.nonbonded_1_4_dict[class_x])
                                             * (-epsilon_kcal_per_mol) * scalar_sign_change,
                                             decimals=epsilon_kcal_per_mol_round_decimals)),
                                str(np.round(_LJ_sigma_to_r_min_div_2(
                                    scalar_used_binary * self.sigmas_angstrom_atom_class_dict[class_x]),
                                              decimals=sigma_round_decimals)
                                     ),
                                str(class_x),
                                str(class_x),
                            )
                        )

                elif self.utilized_NB_expression == 'Mie':
                    data.write("\n")
                    data.write("NONBONDED_MIE\n")
                    data.write("! \n")
                    data.write(
                        "! V(Mie) = (n/(n-6)) * (n/6)**(6/(n-6)) * Epsilon * ((sig/r)**n - (sig/r)**6)\n"
                    )
                    data.write("! \n")

                    data.write("! {:8s} {:15s} {:15s} {:15s} {:15s} {:15s} {:15s} ! {:20s} {:20s}\n"
                               "".format("type_1",
                                         "epsilon",
                                         "sigma",
                                         "n",
                                         "epsilon,1-4",
                                         "sigma,1-4",
                                         "n,1-4",
                                         "extended_type_1",
                                         "extended_type_2",
                                         )
                               )

                    for class_x, epsilon_kcal_per_mol in self.epsilon_kcal_per_mol_atom_class_dict.items():
                        nb_format = "{:10s} {:15s} {:15s} {:15s} {:15s} {:15s} {:15s} ! {:20s} {:20s}\n"

                        # if the 1-4 non-bonded scalar is used.
                        # If 1-4 non-bonded scalar = 0, all 1-4 non-bonded values are set to zero (0).
                        # If 1-4 non-bonded scalar = 1, only the epsilon 1-4 non-bonded interaction is scaled
                        # via another scalar.
                        if float(self.nonbonded_1_4_dict[class_x]) == 0:
                            scalar_used_binary = 0
                        else:
                            scalar_used_binary = 1

                        epsilon_Kelvin = u.unyt_quantity(
                            epsilon_kcal_per_mol, 'kcal/mol').to_value('K', equivalence='thermal')

                        data.write(
                            nb_format.format(
                                str(base10_to_base52_alph(self.atom_class_to_index_value_dict[class_x])),
                                str(np.round(epsilon_Kelvin,
                                             decimals=epsilon_Kelvin_round_decimals)
                                    ),
                                str(np.round(self.sigmas_angstrom_atom_class_dict[class_x],
                                             decimals=sigma_round_decimals)
                                    ),
                                str(np.round(self.mie_n_atom_class_dict[class_x],
                                             decimals=mie_n_round_decimals)
                                    ),
                                str(np.round(scalar_used_binary *
                                             float(self.nonbonded_1_4_dict[class_x]) * (epsilon_Kelvin),
                                             decimals=epsilon_Kelvin_round_decimals)
                                    ),
                                str(np.round(scalar_used_binary *
                                             self.sigmas_angstrom_atom_class_dict[class_x],
                                             decimals=sigma_round_decimals)
                                    ),
                                str(np.round(scalar_used_binary *
                                             self.mie_n_atom_class_dict[class_x],
                                             decimals=mie_n_round_decimals)
                                    ),
                                str(class_x),
                                str(class_x),
                            )
                        )

                elif self.utilized_NB_expression == 'Exp6':
                    printed_output = "ERROR: Currently the 'Exp6' potential (self.utilized_NB_expression) " \
                                     "is not supported in this MoSDeF GOMC parameter writer\n"
                    data.write(printed_output)
                    print_error_message = (printed_output)
                    raise ValueError(print_error_message)

                else:
                    printed_output = f"ERROR: Currently this potential ({self.utilized_NB_expression}) " \
                                     f"is not supported in this MoSDeF GOMC parameter writer\n"
                    data.write(printed_output)
                    print_error_message = (printed_output)
                    raise ValueError(print_error_message)

                # writing end in file
                data.write("\nEND\n")

        # **********************************
        # **********************************
        # FF writer (end)
        # **********************************
        # **********************************

    def write_psf(self):
        """This write_psf function writes the Charmm style PSF (topology) file, which can be utilized
        in the GOMC and NAMD engines."""
        # **********************************
        # **********************************
        # psf writer (start)
        # **********************************
        # **********************************

        print("******************************")
        print("")
        print(
            "The charmm X-plor format psf writer (the write_psf function) is running"
        )

        date_time = datetime.datetime.today()

        print(
            "write_psf: forcefield_selection = {}, residues = {}".format(
                self.forcefield_selection, self.residues
            )
        )

        print("******************************")
        print("")

        if self.structure_box_1:
            list_of_topologies = [
                self.topology_box_0_ff,
                self.topology_box_1_ff,
            ]
            list_of_file_names = [self.filename_box_0, self.filename_box_1]
            stuct_only = [self.topology_box_0_ff, self.topology_box_1_ff]
        else:
            list_of_topologies = [self.topology_box_0_ff]
            list_of_file_names = [self.filename_box_0]
            stuct_only = [self.topology_box_0_ff]

        for q in range(0, len(list_of_topologies)):
            stuct_iteration = list_of_topologies[q]
            file_name_iteration = list_of_file_names[q]
            output = str(file_name_iteration) + ".psf"
            stuct_only_iteration = stuct_only[q]

            no_atoms = stuct_iteration.n_sites
            no_bonds = stuct_iteration.n_bonds
            no_angles = stuct_iteration.n_angles

            no_dihedrals = stuct_iteration.n_dihedrals
            no_impropers = stuct_iteration.n_impropers

            no_donors = 0
            no_acceptors = 0
            no_groups = 0
            no_NNB = 0

            # psf printing (start)

            residue_names_list = []
            residue_id_list = []
            for site in stuct_only_iteration.sites:
                residue_id_list.append(site.__dict__['residue_index_'] )
                residue_names_list.append(site.__dict__['residue_label_'])

            # this sets the residues chain length to a max limit
            res_no_chain_iter_corrected = []
            for residue_id_int in residue_id_list:
                res_id_adder = int(
                    (residue_id_int % self.max_residue_no) % self.max_residue_no
                )
                if int(res_id_adder) == 0:
                    res_no_iteration_corrected = int(self.max_residue_no)
                else:
                    res_no_iteration_corrected = res_id_adder

                res_no_chain_iter_corrected.append(res_no_iteration_corrected)

            output_write = open(output, "w")

            first_indent = "%8s"
            psf_formating = (
                "%8s %-4s %-4s %-4s %-4s %4s %10.6f %13.4f" + 11 * " "
            )

            output_write.write("PSF ")
            output_write.write("\n\n")

            no_of_remarks = 3
            output_write.write(first_indent % no_of_remarks + " !NTITLE\n")
            output_write.write(
                " REMARKS this file "
                + file_name_iteration
                + " - created by mBuild/foyer/gmso using the"
                + "\n"
            )
            output_write.write(
                " REMARKS parameters from the "
                + str(self.forcefield_selection)
                + " force field via MoSDef\n"
            )
            output_write.write(
                " REMARKS created on " + str(date_time) + "\n\n\n"
            )

            # This converts the atom name in the GOMC psf and pdb files to unique atom names
            print(
                "bead_to_atom_name_dict = {}".format(
                    self.bead_to_atom_name_dict
                )
            )
            [
                unique_individual_atom_names_dict,
                individual_atom_names_list,
                missing_bead_to_atom_name,
            ] = unique_atom_naming(
                stuct_only_iteration,
                residue_id_list,
                residue_names_list,
                bead_to_atom_name_dict=self.bead_to_atom_name_dict,
            )

            if None in [
                unique_individual_atom_names_dict,
                individual_atom_names_list,
                missing_bead_to_atom_name,
            ]:
                self.input_error = True
                print_error_message = (
                    "ERROR: The unique_atom_naming function failed while "
                    "running the charmm_writer function. Ensure the proper inputs are "
                    "in the bead_to_atom_name_dict."
                )
                raise ValueError(print_error_message)

            # ATOMS: Calculate the atom data
            # psf_formating is conducted for the for CHARMM format (i.e., atom types are base 52, letters only)
            output_write.write(first_indent % no_atoms + " !NATOM\n")
            for i_atom, PSF_atom_iteration_1 in enumerate(
                stuct_iteration.sites
            ):
                segment_id = "SYS" # can also use residue segid for CHARMM

                charge_iter = PSF_atom_iteration_1.atom_type.__dict__['charge_'].to('C') / u.elementary_charge
                charge_iter = charge_iter.to_value('(dimensionless)')
                mass_iter = PSF_atom_iteration_1.atom_type.__dict__['mass_'].to_value('amu')
                #atom_ff_type_iter = PSF_atom_iteration_1.atom_type.__dict__['name_']
                residue_name_iter = PSF_atom_iteration_1.__dict__['residue_label_']
                atomclass_name_iter = PSF_atom_iteration_1.atom_type.__dict__['atomclass_']

                atom_type_iter = base10_to_base52_alph(
                    self.atom_class_to_index_value_dict[
                        f"{atomclass_name_iter}_{residue_name_iter}"
                    ]
                )

                atom_lines_iteration = psf_formating % (
                    i_atom + 1,
                    segment_id,
                    res_no_chain_iter_corrected[i_atom],
                    str(residue_names_list[i_atom])[: self.max_resname_char],
                    individual_atom_names_list[i_atom],
                    atom_type_iter,
                    charge_iter,
                    mass_iter,
                )

                output_write.write("%s\n" % atom_lines_iteration)

            output_write.write("\n")

            # BONDS: Calculate the bonding data
            output_write.write(first_indent % no_bonds + " !NBOND: bonds\n")
            for i_bond, bond_iteration in enumerate(
                stuct_iteration.bonds
            ):
                output_write.write(
                    (first_indent * 2)
                    % (
                        bond_iteration.connection_members[0].label + 1,
                        bond_iteration.connection_members[1].label + 1,
                    )
                )

                if (i_bond + 1) % 4 == 0:
                    output_write.write("\n")

            if no_bonds % 4 == 0:
                output_write.write("\n")
            else:
                output_write.write("\n\n")

            if no_bonds == 0:
                output_write.write("\n")

            # ANGLES: Calculate the angle data
            output_write.write(first_indent % no_angles + " !NTHETA: angles\n")
            for i_angle, angle_iteration in enumerate(stuct_iteration.angles):

                output_write.write(
                    (first_indent * 3)
                    % (
                        angle_iteration.connection_members[0].label + 1,
                        angle_iteration.connection_members[1].label + 1,
                        angle_iteration.connection_members[2].label + 1,
                    )
                )

                if (i_angle + 1) % 3 == 0:
                    output_write.write("\n")

            if no_angles % 3 == 0:
                output_write.write("\n")
            else:
                output_write.write("\n\n")

            if no_angles == 0:
                output_write.write("\n")

            # DIHEDRALS: Calculate the dihedral  data
            output_write.write(
                first_indent % no_dihedrals + " !NPHI: dihedrals\n"
            )
            for i_dihedral, dihedral_iter in enumerate(stuct_iteration.dihedrals):
                output_write.write(
                    (first_indent * 4)
                    % (
                        dihedral_iter.connection_members[0].label + 1,
                        dihedral_iter.connection_members[1].label + 1,
                        dihedral_iter.connection_members[2].label + 1,
                        dihedral_iter.connection_members[3].label + 1,
                    )
                )

                if (i_dihedral + 1) % 2 == 0:
                    output_write.write("\n")

            if no_dihedrals % 2 == 0:
                output_write.write("\n")
            else:
                output_write.write("\n\n")

            if no_dihedrals == 0:
                output_write.write("\n")

            # IMPROPERS: Calculate the improper data
            output_write.write(
                first_indent % no_impropers + " !NIMPHI: impropers\n"
            )
            for i_improper, improper_iter in enumerate(stuct_iteration.impropers):
                output_write.write(
                    (first_indent * 4)
                    % (
                        improper_iter.connection_members[0].label + 1,
                        improper_iter.connection_members[1].label + 1,
                        improper_iter.connection_members[2].label + 1,
                        improper_iter.connection_members[3].label + 1,
                    )
                )

                if (i_improper + 1) % 2 == 0:
                    output_write.write("\n")

            if no_impropers % 2 == 0:
                output_write.write("\n")
            else:
                output_write.write("\n\n")

            if no_impropers == 0:
                output_write.write("\n")

            # DONOR: calculate the donor data
            output_write.write(first_indent % no_donors + " !NDON: donors\n")
            output_write.write("\n")

            # ACCEPTOR: calculate the acceptor data
            output_write.write(
                first_indent % no_acceptors + " !NACC: acceptors\n"
            )
            output_write.write("\n")

            # NNB: calculate the NNB data
            output_write.write(first_indent % no_NNB + " !NNB\n")
            output_write.write("\n")

            # GROUP: calculate the group data
            output_write.write(
                first_indent % no_groups + " !NGRP \n"
            )
            output_write.write("\n")

            output_write.close()
        # **********************************
        # **********************************
        # psf writer (end)
        # **********************************
        # **********************************

    def write_pdb(self, space_group="P 1"):
        """This write_psf function writes the Charmm style PDB (coordinate file), which can be utilized
        in the GOMC and NAMD engines.

        Parameters
        ----------
        space_group : str (default="P 1")
            The space group of the structure

        """
        # **********************************
        # **********************************
        # pdb writer (start)
        # **********************************
        # **********************************

        date_time = datetime.datetime.today()
        print("******************************")
        print("")
        print("The charmm pdb writer (the write_pdb function) is running")
        print("write_charmm_pdb: residues == {}".format(self.residues))
        print("fix_residue = {}".format(self.fix_residue))
        print("fix_residue_in_box = {}".format(self.fix_residue_in_box))
        print("bead_to_atom_name_dict = {}".format(self.bead_to_atom_name_dict))

        if self.fix_residue is None and self.fix_residue_in_box is None:
            print(
                "INFORMATION: No atoms are fixed in this pdb file for the GOMC simulation engine. "
            )
        else:
            warn(
                "Some atoms are fixed in this pdb file for the GOMC simulation engine. "
            )

        print("******************************")
        print("")

        if self.structure_box_1:
            list_of_topologies = [
                self.topology_box_0_ff,
                self.topology_box_1_ff,
            ]
            list_of_file_names = [self.filename_box_0, self.filename_box_1]
            stuct_only = [self.topology_box_0_ff, self.topology_box_1_ff]
        else:
            list_of_topologies = [self.topology_box_0_ff]
            list_of_file_names = [self.filename_box_0]
            stuct_only = [self.topology_box_0_ff]

        for q in range(0, len(list_of_topologies)):
            file_name_iteration = list_of_file_names[q]
            output = str(file_name_iteration) + ".pdb"
            stuct_only_iteration = stuct_only[q]

            output_write = open(output, "w")
            # output_write.write(
            #'REMARK this file ' + file_name_iteration + ' - created by mBuild/foyer using the' + '\n')
            # output_write.write(
            #'REMARK parameters from the ' + str(self.forcefield_selection) + ' force field via MoSDef\n')
            # output_write.write('REMARK created on ' + str(date_time) + '\n')

            # caluculate the atom name and unique atom names and lock occupany factor at 1 (instead of: atom.occupancy)
            locked_occupany_factor = 1.00
            max_no_atoms_in_base10 = 99999  # 99,999 for atoms in psf/pdb

            atom_no_list = []
            element_list = []
            residue_names_list = []
            residue_id_list = []
            res_chain_iteration_corrected_list = []

            fix_atoms_list = []
            atom_alternate_location_list = []
            residue_code_insertion_list = []
            x_y_z_coor_list = []
            segment_id = []

            atom_alternate_location_all_values = ""
            residue_code_insertion_all_values = ""
            segment_id_all_values = ""
            for k, site in enumerate(stuct_only_iteration.sites):
                atom_no_list.append(site.__dict__['label_'])

                try:
                    element_list.append(site.__dict__['name_'])
                except:
                    element_list.append('EP')

                residue_names_list.append(site.__dict__['residue_label_'])
                residue_id_list.append(site.__dict__['residue_index_']  )

                if (self.fix_residue is not None) and (
                        site.__dict__['residue_label_'] in self.fix_residue
                ):
                    beta_iteration = 1.00
                elif (self.fix_residue_in_box is not None) and (
                        site.__dict__['residue_label_'] in self.fix_residue_in_box
                ):
                    beta_iteration = 2.00
                else:
                    beta_iteration = 0.00
                fix_atoms_list.append(beta_iteration)

                atom_alternate_location_list.append(atom_alternate_location_all_values)
                residue_code_insertion_list.append(residue_code_insertion_all_values)

                x_y_z_coor = site.__dict__['position_'].to_value('angstrom')
                x_y_z_coor_list.append(x_y_z_coor)

                segment_id.append(segment_id_all_values)

                res_chain_iteration_corrected_list.append(
                    base10_to_base26_alph(int(residue_id_list[-1] / (self.max_residue_no + 1)))[-1:])

            for n in range(0, len(residue_names_list)):
                if residue_names_list[n] not in self.residues:
                    self.input_error = True
                    print_error_message = "ERROR: Please specifiy all residues (residues) in a list"
                    raise ValueError(print_error_message)

            if (self.fix_residue is not None) and (
                self.fix_residue_in_box is not None
            ):
                for n in range(0, len(self.fix_residue)):
                    if self.fix_residue[n] in self.fix_residue_in_box:
                        self.input_error = True
                        print_error_message = (
                            "ERROR: residue type can not be specified to both "
                            "fix_residue and fix_residue_in_box"
                        )
                        raise ValueError(print_error_message)

            if stuct_only_iteration.box is not None:
                output_write.write(
                    "CRYST1%9.3f%9.3f%9.3f%7.2f%7."
                    "2f%7.2f %-11s%4s\n"
                    % (
                        stuct_only_iteration.box[0],
                        stuct_only_iteration.box[1],
                        stuct_only_iteration.box[2],
                        stuct_only_iteration.box[3],
                        stuct_only_iteration.box[4],
                        stuct_only_iteration.box[5],
                        space_group,
                        "",
                    )
                )

            pdb_atom_line_format = "ATOM  %5s %-4s%1s%-4s%1s%4d%1s   %8.3f%8.3f%8.3f%6.2f%6.2f      %-4s%2s%-2s\n"

            # This converts the atom name in the CHARMM psf and pdb files to unique atom names
            [
                unique_individual_atom_names_dict,
                individual_atom_names_list,
                missing_bead_to_atom_name,
            ] = unique_atom_naming(
                stuct_only_iteration,
                residue_id_list,
                residue_names_list,
                bead_to_atom_name_dict=self.bead_to_atom_name_dict,
            )

            if None in [
                unique_individual_atom_names_dict,
                individual_atom_names_list,
                missing_bead_to_atom_name,
            ]:
                self.input_error = True
                print_error_message = (
                    "ERROR: The unique_atom_naming function failed while "
                    "running the charmm_writer function. Ensure the proper inputs are "
                    "in the bead_to_atom_name_dict."
                )

                raise ValueError(print_error_message)

            for v, atom_iter_1 in enumerate(stuct_only_iteration.sites):
                if v + 1 > max_no_atoms_in_base10:
                    atom_number = base10_to_base16_alph_num(v + 1)

                else:
                    atom_number = v + 1

                output_write.write(
                    pdb_atom_line_format
                    % (
                        atom_number,
                        individual_atom_names_list[v],
                        atom_alternate_location_list[v],
                        str(residue_names_list[v])[: self.max_resname_char],
                        res_chain_iteration_corrected_list[v],  #res_chain_iteration_corrected_list[v],
                        residue_id_list[v],  #res_no_chain_iter_corrected[v],
                        residue_code_insertion_list[v],
                        x_y_z_coor_list[v][0],
                        x_y_z_coor_list[v][1],
                        x_y_z_coor_list[v][2],
                        locked_occupany_factor,
                        fix_atoms_list[v],
                        segment_id[v],
                        element_list[v],
                        "",
                    )
                    )

            output_write.write("%-80s\n" % "END")

            output_write.close()

            # **********************************
            # **********************************
            # pdb writer (end)
            # **********************************
            # **********************************
