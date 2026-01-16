import datetime
import os
from warnings import warn

import gmso
import numpy as np
import scipy
import unyt as u
from mbuild.box import Box
from mbuild.compound import Compound
from unyt.dimensions import angle, energy, length, temperature

from mosdef_gomc.utils.conversion import (
    OPLS_to_periodic,
    RB_to_periodic,
    base10_to_base16_alph_num,
    base10_to_base26_alph,
    base10_to_base44_alph,
    base10_to_base52_alph,
    base10_to_base62_alph_num,
)
from mosdef_gomc.utils.gmso_equation_compare import (
    evaluate_harmonic_angle_format_with_scaler,
    evaluate_harmonic_bond_format_with_scaler,
    evaluate_harmonic_improper_format_with_scaler,
    evaluate_harmonic_torsion_format_with_scaler,
    evaluate_OPLS_torsion_format_with_scaler,
    evaluate_periodic_improper_format_with_scaler,
    evaluate_periodic_torsion_format_with_scaler,
    evaluate_RB_torsion_format_with_scaler,
    get_atom_type_expressions_and_scalars,
)
from mosdef_gomc.utils.gmso_specific_ff_to_residue import specific_ff_to_residue


def _check_convert_bond_k_constant_units(
    bond_class_or_type_input_str,
    bond_energy_input_unyt,
    bond_energy_output_units_str,
):
    """Checks to see if the value is a valid bond k-constant
    energy value and converts it to kcal/mol/angstroms**2

    Parameters
    ----------
    bond_class_or_type_input_str: str
        The bond class information from the gmso.bond_types.member_class or gmso.bond_types.member_type
    bond_energy_input_unyt: unyt.unyt_quantity
        The bond energy in units of 'energy/mol/angstroms**2' or 'K/angstroms**2'.
        NOTE that the only valid temperature unit for thermal energy is Kelvin (K),
        which in 'K/angstroms**2'.
    bond_class_output_units_str: str ('kcal/mol/angstrom**2' or 'K/angstrom**2')
        The bond class information from the gmso.bond_types.member_class


    Returns
    -------
    If the bond_energy_input_unyt value is unyt.unyt_quantity of energy/length**2: unyt.unyt_quantity
        The value is in 'kcal/mol/angstrom**2' or 'K/angstrom**2' units.
    If the bond_energy_input_unyt value is not unyt.unyt_quantity of energy/length**2: raise TypeError
    If the bond_energy_output_units_str value is not a valid choice: raise ValueError
    """

    if bond_energy_output_units_str not in [
        "kcal/mol/angstrom**2",
        "K/angstrom**2",
    ]:
        print_error_message = (
            "ERROR: The selected bond energy k-constant units via "
            "bond_energy_output_units_str "
            "are not 'kcal/mol/angstrom**2' or 'K/angstrom**2'."
        )
        raise ValueError(print_error_message)

    print_error_message = (
        f"ERROR: The bond class input, {bond_class_or_type_input_str} is {type(bond_energy_input_unyt)} "
        f"and needs to be a {u.array.unyt_quantity} "
        f"in energy/length**2 units, "
        f"such as 'kcal/mol/angstrom**2', 'kJ/mol/angstrom**2', or 'K/angstrom**2'."
    )
    if isinstance(bond_energy_input_unyt, u.array.unyt_quantity):
        if energy / length**2 == bond_energy_input_unyt.units.dimensions:
            if bond_energy_output_units_str == "kcal/mol/angstrom**2":
                bond_energy_output_unyt = bond_energy_input_unyt.to(
                    "kcal/mol/angstrom**2"
                )
                return bond_energy_output_unyt

            elif bond_energy_output_units_str == "K/angstrom**2":
                bond_energy_output_unyt = bond_energy_input_unyt.to(
                    "kcal/mol/angstrom**2"
                )
                bond_energy_output_unyt = (
                    bond_energy_output_unyt * u.angstrom**2
                )
                bond_energy_output_unyt = bond_energy_output_unyt.to(
                    "K", equivalence="thermal"
                )
                bond_energy_output_unyt = (
                    bond_energy_output_unyt / u.angstrom**2
                )
                return bond_energy_output_unyt

        elif temperature / length**2 == bond_energy_input_unyt.units.dimensions:
            if bond_energy_output_units_str == "kcal/mol/angstrom**2":
                bond_energy_output_unyt = bond_energy_input_unyt.to(
                    "K/angstrom**2"
                )
                bond_energy_output_unyt = (
                    bond_energy_output_unyt * u.angstrom**2
                )
                bond_energy_output_unyt = bond_energy_output_unyt.to(
                    "kcal/mol", equivalence="thermal"
                )
                bond_energy_output_unyt = (
                    bond_energy_output_unyt / u.angstrom**2
                )
                return bond_energy_output_unyt

            elif bond_energy_output_units_str == "K/angstrom**2":
                bond_energy_output_unyt = bond_energy_input_unyt.to(
                    "K/angstrom**2"
                )
                return bond_energy_output_unyt

        else:
            raise TypeError(print_error_message)
    else:
        raise TypeError(print_error_message)


def _check_convert_angle_k_constant_units(
    angle_class_or_type_input_str,
    angle_energy_input_unyt,
    angle_energy_output_units_str,
):
    """Checks to see if the value is a valid angle k-constant energy value
    and converts it to kcal/mol/rad**2 or 'K/rad**2'.

    Parameters
    ----------
    angle_class_or_type_input_str: str
        The angle class information from the gmso.angle_types.member_class or gmso.bond_types.member_type
    angle_energy_input_unyt: unyt.unyt_quantity
        The angle energy in units of 'energy/mol/rad**2' or 'K/rad**2'.
        NOTE that the only valid temperature unit for thermal energy is Kelvin (K),
        which in 'K/rad**2'.
    angle_class_output_units_str: str ('kcal/mol/rad**2' or 'K/rad**2')
        The angle class information from the gmso.angle_types.member_class


    Returns
    -------
    If the angle_energy_input_unyt value is unyt.unyt_quantity of energy/angle**2: unyt.unyt_quantity
        The value is in 'kcal/mol/rad**2' or 'K/rad**2' units.
    If the angle_energy_input_unyt value is not unyt.unyt_quantity of energy/angle**2: raise TypeError
    If the angle_energy_output_units_str value is not a valid choice: raise ValueError
    """

    if angle_energy_output_units_str not in ["kcal/mol/rad**2", "K/rad**2"]:
        print_error_message = (
            "ERROR: The selected angle energy k-constant units via "
            "angle_energy_output_units_str "
            "are not 'kcal/mol/rad**2' or 'K/rad**2'."
        )
        raise ValueError(print_error_message)

    print_error_message = (
        f"ERROR: The angle class input, {angle_class_or_type_input_str} is {type(angle_energy_input_unyt)} "
        f"and needs to be a {u.array.unyt_quantity} "
        f"in energy/angle**2 units, "
        f"such as 'kcal/mol/rad**2', 'kJ/mol/rad**2', or 'K/rad**2'."
    )
    if isinstance(angle_energy_input_unyt, u.array.unyt_quantity):
        if energy / angle**2 == angle_energy_input_unyt.units.dimensions:
            if angle_energy_output_units_str == "kcal/mol/rad**2":
                angle_energy_output_unyt = angle_energy_input_unyt.to(
                    "kcal/mol/rad**2"
                )
                return angle_energy_output_unyt

            elif angle_energy_output_units_str == "K/rad**2":
                angle_energy_output_unyt = angle_energy_input_unyt.to(
                    "kcal/mol/rad**2"
                )
                angle_energy_output_unyt = angle_energy_output_unyt * u.rad**2
                angle_energy_output_unyt = angle_energy_output_unyt.to(
                    "K", equivalence="thermal"
                )
                angle_energy_output_unyt = angle_energy_output_unyt / u.rad**2
                return angle_energy_output_unyt

        elif temperature / angle**2 == angle_energy_input_unyt.units.dimensions:
            if angle_energy_output_units_str == "kcal/mol/rad**2":
                angle_energy_output_unyt = angle_energy_input_unyt.to(
                    "K/rad**2"
                )
                angle_energy_output_unyt = angle_energy_output_unyt * u.rad**2
                angle_energy_output_unyt = angle_energy_output_unyt.to(
                    "kcal/mol", equivalence="thermal"
                )
                angle_energy_output_unyt = angle_energy_output_unyt / u.rad**2
                return angle_energy_output_unyt

            elif angle_energy_output_units_str == "K/rad**2":
                angle_energy_output_unyt = angle_energy_input_unyt.to(
                    "K/rad**2"
                )
                return angle_energy_output_unyt

        else:
            raise TypeError(print_error_message)
    else:
        raise TypeError(print_error_message)


def _LJ_sigma_to_r_min(sigma):
    """Convert sigma to Rmin for the non-bonded Lennard-Jones (LJ) potential energy equation.

    Parameters
    ----------
    sigma: int or float
        The sigma value for the non-bonded Lennard-Jones (LJ) potential energy equation.

    Returns
    ----------
    r_min: float
        The radius at the minimum energy (Rmin) for the non-bonded Lennard-Jones (LJ) potential energy equation.
    """
    r_min = float(sigma * 2 ** (1 / 6))

    return r_min


def _LJ_sigma_to_r_min_div_2(sigma):
    """Convert sigma to Rmin/2 for the non-bonded Lennard-Jones (LJ) potential energy equation.

    Parameters
    ----------
    sigma: int or float
        The sigma value for the non-bonded Lennard-Jones (LJ) potential energy equation.

    Returns
    ----------
    r_min_div_2: float
        The radius at the minimum energy divided by 2 (Rmin/2)
        for the non-bonded Lennard-Jones (LJ) potential energy equation.
    """
    r_min_div_2 = float(sigma * 2 ** (1 / 6) / 2)

    return r_min_div_2


def _Exp6_Rmin_to_sigma(sigma, Rmin, alpha):
    """Get equation to convert Rmin to sigma for the Exponential-6 (Exp6) potential energy equation.

    .. math::
    Exp6_{potential} &= epsilon * alpha / (alpha - 6) *
                     &= (6 / alpha * np.exp(alpha * (1 - sigma / Rmin)) - (Rmin / sigma)**6)

    Parameters
    ----------
    sigma: variable for find root
        The sigma variable that will be solve for the non-bonded Exp6 potential
        energy equation via the epsilon, Rmin, and alpha parameters.
    Rmin: int or float
        The Rmin value for the non-bonded Exp6 potential energy equation.
    alpha: int or float
        The alpha value for the non-bonded Exp6 potential energy equation.

    Returns
    ----------
    exp6_eqn_with_sigma_only_variable: equation
        The Exp6 potential energy equation with sigma as the only variable.
        The other variables (Rmin and alpha) are entered as constants,
        as epsilon is not required.
    """
    exp6_eqn_with_sigma_only_variable = (
        alpha
        / (alpha - 6)
        * (6 / alpha * np.exp(alpha * (1 - sigma / Rmin)) - (Rmin / sigma) ** 6)
    )

    return exp6_eqn_with_sigma_only_variable


def _Exp6_Rmin_to_sigma_solver(
    Rmin_actual, alpha_actual, Rmin_fraction_for_sigma_findroot=0.95
):
    """
    Numerically solve the sigma value in non-bonded Exp6 potential (using epsilon, r_min, and alpha).

    .. math::
    Exp6_{potential} &= epsilon * alpha / (alpha - 6) *
                     &= (6 / alpha * np.exp(alpha * (1 - sigma / Rmin)) - (Rmin / sigma)**6)

    Parameters
    ----------
    Rmin_actual: variable
        The Rmin value for the non-bonded Exp6 potential energy equation.
    alpha_actual: int or float
        The alpha value for the non-bonded Exp6 potential energy equation.
    Rmin_fraction_for_sigma_findroot: float, default=0.95
        The fraction of the r_min value used to provide the starting input
        to the numerical solver (find root/scipy.optimize).
        This must be less than r_min, but not too much less than r_min,
        or it will find a non-logical root, due to the Exp6 potential's
        unrealistic other root at low atomic radii.

    Returns
    ----------
    sigma_calculated: float
        The numerically solved sigma value for the non-bonded Exp6 potential energy equation.
    """
    if alpha_actual == 6 or Rmin_actual == 0:
        raise ValueError(
            f"ERROR: The Exp6 potential Rmin --> sigma converter failed. "
            f"The Exp6 potential values can not be Rmin = 0 or alpha = 6, "
            f"as it divides by zero. "
            f"The entered values are Rmin = {Rmin_actual} and alpha = {alpha_actual}."
        )
    exp6_sigma_solver = scipy.optimize.root(
        lambda sigma: _Exp6_Rmin_to_sigma(sigma, Rmin_actual, alpha_actual),
        Rmin_actual * Rmin_fraction_for_sigma_findroot,
    )

    sigma_calculated = exp6_sigma_solver.x[0]

    # check for errors
    if (
        exp6_sigma_solver.message != "The solution converged."
        or sigma_calculated >= Rmin_actual
    ):
        raise ValueError(
            "ERROR: The Exp6 potential Rmin --> sigma converter failed. "
            "It did not converge, sigma_calculated >= Rmin_actual, or "
            "another issue."
        )

    return sigma_calculated


def _Exp6_sigma_to_Rmin(Rmin, sigma, alpha):
    """Get equation to convert Rmin to sigma for the Exponential-6 (Exp6) potential energy equation.

    .. math::
    Exp6_{potential} &= epsilon * alpha / (alpha - 6) *
                     &= (6 / alpha * np.exp(alpha * (1 - sigma / Rmin)) - (Rmin / sigma)**6)

    Parameters
    ----------
    Rmin: variable for find root
        The Rmin variable that will be solve for the non-bonded Exp6 potential
        energy equation via the epsilon, Rmin, and alpha parameters.
    sigma: int or float
        The sigma value for the non-bonded Exp6 potential energy equation.
    alpha: int or float
        The alpha value for the non-bonded Exp6 potential energy equation.

    Returns
    ----------
    exp6_eqn_with_r_min_only_variable: equation
        The Exp6 potential energy equation with Rmin as the only variable.
        The other variables (sigma and alpha) are entered as constants,
        as epsilon is not required.
    """
    exp6_eqn_with_r_min_only_variable = (
        1
        / (1 - 6 / alpha)
        * (6 / alpha * np.exp(alpha * (1 - sigma / Rmin)) - (Rmin / sigma) ** 6)
    )

    return exp6_eqn_with_r_min_only_variable


def _Exp6_sigma_to_Rmin_solver(
    sigma_actual, alpha_actual, sigma_fraction_for_Rmin_findroot=1.05
):
    """
    Numerically solve the sigma value in non-bonded Exp6 potential (using epsilon, r_min, and alpha).

    .. math::
    Exp6_{potential} &= epsilon * alpha / (alpha - 6) *
                     &= (6 / alpha * np.exp(alpha * (1 - sigma / Rmin)) - (Rmin / sigma)**6)

    Parameters
    ----------
    sigma_actual: variable
        The sigma value for the non-bonded Exp6 potential energy equation.
    alpha_actual: int or float
        The alpha value for the non-bonded Exp6 potential energy equation.
    sigma_fraction_for_Rmin_findroot: float, default=0.95
        The fraction of the sigma value used to provide the starting input
        to the numerical solver (find root/scipy.optimize).
        This must be greater than sigma, but not too much greater than sigma,
        or it will find a non-logical root, due to the Exp6 potential's
        convergence to zero (0) at large atomic radii.

    Returns
    ----------
    Rmin_calculated: float
        The numerically solved Rmin value for the non-bonded Exp6 potential energy equation.
    """
    if alpha_actual == 6 or sigma_actual == 0:
        raise ValueError(
            f"ERROR: The Exp6 potential sigma --> Rmin converter failed. "
            f"The Exp6 potential values can not be sigma = 0 or alpha = 6, "
            f"as it divides by zero. "
            f"The entered values are sigma = {sigma_actual} and alpha = {alpha_actual}."
        )

    exp6_Rmin_solver = scipy.optimize.root(
        lambda Rmin: _Exp6_sigma_to_Rmin(Rmin, sigma_actual, alpha_actual),
        sigma_actual * sigma_fraction_for_Rmin_findroot,
    )

    Rmin_calculated = exp6_Rmin_solver.x[0]

    # check for errors
    if (
        exp6_Rmin_solver.message != "The solution converged."
        or Rmin_calculated <= sigma_actual
    ):
        raise ValueError(
            "ERROR: The Exp6 potential sigma --> Rmin converter failed. "
            "It did not converge, Rmin_calculated <= sigma_actual, or "
            "another issue."
        )

    return Rmin_calculated


def unique_atom_naming(
    topology, residue_id_list, residue_names_list, bead_to_atom_name_dict=None
):
    """
    Generates unique atom/bead names for each molecule, which is required for some
    simulation types (Example: The special Monte Carlo moves)

    Parameters
    ----------
    topology: gmso.Topology object
    residue_id_list: list, in sequential order
            The residue ID for every atom in the system
    residue_names_list: list, in sequential order
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
    unique_individual_atom_names_dict: dictionary
        All the unique atom names comno_piled into a dictionary.
    individual_atom_names_list: list, in sequential  order
        The atom names for every atom in the system
    missing_bead_to_atom_name: list, in sequential  order
        The bead names of any atoms beads that did not have a name specificed to them
        via the bead_to_atom_name_dict
    """
    unique_individual_atom_names_dict = {}
    individual_atom_names_list = []
    missing_bead_to_atom_name = []
    for i, site in enumerate(topology.sites):
        site_name_unique_naming = site.__dict__["name_"]

        # extract element or atom name from mol2 without numbers (integers)
        element_name_unique_naming = ""
        for site_name_unique_naming_char_i in site_name_unique_naming:
            try:
                int(site_name_unique_naming_char_i)

            except:
                element_name_unique_naming += site_name_unique_naming_char_i

        if element_name_unique_naming == "":
            raise ValueError(
                "ERROR: The input file, likely mol2 file does not contain element names or char, only int."
            )

        interate_thru_names = True
        j = 0
        while interate_thru_names is True:
            j = j + 1
            if str(site_name_unique_naming)[:1] == "_":
                if (
                    bead_to_atom_name_dict is not None
                    and (str(site_name_unique_naming) in bead_to_atom_name_dict)
                    is True
                ):
                    if (
                        len(
                            bead_to_atom_name_dict[str(site_name_unique_naming)]
                        )
                        > 2
                    ):
                        text_to_write = (
                            "ERROR: only enter atom names that have 2 or less digits"
                            + " in the Bead to atom naming dictionary (bead_to_atom_name_dict)."
                        )
                        warn(text_to_write)
                        return None, None, None
                    else:
                        atom_name_value = bead_to_atom_name_dict[
                            str(site_name_unique_naming)
                        ]
                        no_digits_atom_name = 2
                else:
                    missing_bead_to_atom_name.append(1)
                    atom_name_value = "BD"
                    no_digits_atom_name = 2
            elif (
                len(str(element_name_unique_naming)) > 2
                and not str(site_name_unique_naming)[:1] == "_"
            ):
                if len(str(element_name_unique_naming)) == 3:
                    no_digits_atom_name = 1
                    atom_name_value = element_name_unique_naming
                else:
                    text_to_write = (
                        "ERROR: atom numbering will not work propery at"
                        + " the element has more than 4 charaters"
                    )
                    warn(text_to_write)
                    return None, None, None
            else:
                no_digits_atom_name = 2
                atom_name_value = element_name_unique_naming

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
    lengths: list-like, shape=(3,), dtype=float
        Lengths of the edges of the box (user chosen units).
    angles: list-like, shape=(3,), dtype=float, default=None
        Angles (in degrees) that define the tilt of the edges of the box. If
        None is given, angles are assumed to be [90.0, 90.0, 90.0]. These are
        also known as alpha, beta, gamma in the crystallography community.
    precision: int, optional, default=6
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
    gomc_fix_bonds_and_or_angles: list of strings, [str, ..., str]
        A list of the residues (i.e., molecules since GOMC currently considers a
        a whole molecule as a residue) to have their bonds and/or angles held
        rigid/fixed for the GOMC simulation engine.
        The `gomc_fix_bonds_angles`, `gomc_fix_bonds`, `gomc_fix_angles` are the only possible
        variables from the `Charmm` object to be entered.
        In GOMC, the residues currently are the same for every bead or atom in
        the molecules. Therefore, when the residue is selected, the whole molecule
        is selected.
    gomc_fix_bonds_and_or_angles_selection: str
        The name of the variable that is used but formatted as a string, which is fed
        to the error and information outputs. The
        `gomc_fix_bonds_angles`, `gomc_fix_bonds`, `gomc_fix_angles` are the only possible
        variables from the `Charmm` object to be entered.
        Whichever variable you choose, the variable name is just input as a
        string here. For example, if `gomc_fix_bonds_and_or_angles` is equal to
        gomc_fix_bonds_angles, then this should be 'gomc_fix_bonds_angles'
        (i.e., `gomc_fix_bonds_and_or_angles_selection` = 'gomc_fix_bonds_angles').
    residues: list, [str, ..., str]
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
        * Energy units = Lennard-Jones (LJ) -> kcal/mol, Mie and Exp-6 (Buckingham) -> Kelvin (K)
        * Harmonic bonds: Kb = Energy units, b0 = Angstroms
        * Harmonic angles: Ktheta = (Energy units)/rad**2 , Theta0 = degrees
        * Dihedral angles Harmonic (not unavailable): Ktheta = Energy units, n (LJ) = 0 (unitless integer), delta = degrees
        * Dihedral angles: Periodic: Ktheta = Energy units, n (unitless integer) -> n (LJ) = 1-5 and n (Mie or Buckingham) = 0-5, delta = degrees
        * Improper angles Harmonic (not unavailable): Ktheta = Energy units, n (LJ) = 0 (unitless integer), delta = degrees
        * Improper angles: Periodic: Ktheta = Energy units, n = 1-5 (unitless integer), delta = degrees
        * Lennard-Jones (LJ)-NONBONDED: epsilon = Energy units, Rmin/2 = Angstroms
        * Mie-NONBONDED: epsilon = Energy units, sigma = Angstroms, n = integer (unitless)
        * Exp6-NONBONDED: epsilon = Energy units, sigma = Angstroms (FF XMLs in Rmin), alpha = integer (unitless)
        * Lennard-Jones (LJ)-NBFIX (not unavailable): epsilon = Energy units, Rmin = Angstroms
        * Mie-NBFIX (not unavailable): same as Mie-NONBONDED
        * Exp6-NBFIX (not unavailable): same as Exp6-NONBONDED

    Note: The ``Charmm`` object is only compatible with molecules with a single residue (single residue name).

    Note: The units are only the same for GOMC and NAMD when using the Lennard-Jones (LJ) non-bonded type.

    Note: NAMD is only compatible with the Lennard-Jones (LJ) non-bonded type.

    Note: There are CHARMM style atom type naming conventions based on the force field xml provided and
    the GOMC simulation input parameters, outlined in the atom_type_naming_style input variable.

    Parameters
    ----------
    structure_box_0: mbuild Compound object (mbuild.Compound) or mbuild Box object (mbuild.Box);
        If the structure has atoms/beads it must be an mbuild Compound.
        If the structure is empty it must be and mbuild Box object.
        Note: If 1 structures are provided (i.e., only structure_box_0),
        it must be an mbuild Compound.
        Note: If 2 structures are provided,
        only 1 structure can be an empty box (i.e., either structure_box_0 or structure_box_1)
    filename_box_0: str
        The file name of the output file for structure_box_0.  Note: the extension should
        not be provided, as multiple extension (.pdb and .psf) are added to this name.
    structure_box_1: mbuild Compound object (mbuild.Compound) or mbuild Box object (mbuild.Box), default=None;
        If the structure has atoms/beads it must be an mbuild Compound.
        Note: When running a GEMC or GCMC simulation the box 1 stucture should be input
        here.  Otherwise, there is no guarantee that any of the atom type and force field
        information will all work together correctly with box 0, if it is built separately.
        Note: If 2 structures are provided, only 1 structure can be an empty box
        (i.e., either structure_box_0 or structure_box_1).
    filename_box_1: str , default=None
        The file name of the output file for structure_box_1 (Ex: for GCMC or GEMC simulations
        which have multiple simulation boxes).  Note: the extension should
        not be provided, as multiple extension (.pdb and .psf) are added to this name.
        Note: When running a GEMC or GCMC simulation the box 1 stucture should be input
        here. Otherwise, there is no guarantee that any of the atom type and force field
        information will all work together correctly with box 0, if it is built separately.
    residues: list, [str, ..., str]
        Labels of unique residues in the Compound. Residues are assigned by
        checking against Compound.name.  Only supply residue names as 4 character
        strings, as the residue names are truncated to 4 characters to fit in the
        psf and pdb file.
    forcefield_selection: str or dictionary, default=None
        Apply a forcefield to the output file by selecting a force field XML file with
        its path or by using the standard force field name provided the `foyer` package.
        Note: to write the NAMD/GOMC force field, pdb, and psf files, the
        residues and forcefields must be provided in a str or
        dictionary.  If a dictionary is provided all residues must
        be specified to a force field.
        * Example dict for FF file: {'ETH': 'oplsaa.xml', 'OCT': 'path_to_file/trappe-ua.xml'}

        * Example str for FF file: 'path_to file/trappe-ua.xml'

        * Example dict for standard FF names: {'ETH': 'oplsaa', 'OCT': 'trappe-ua'}

        * Example str for standard FF names: 'trappe-ua'

        * Example of a mixed dict with both: {'ETH': 'oplsaa', 'OCT': 'path_to_file/'trappe-ua.xml'}

    gmso_match_ff_by: str ("group" or "molecule"), default = "molecule"
        How the GMSO force field is applied, using the molecules name/residue name (mbuild.Compound().name)
        for GOMC and NAMD.  This is regardless number of levels in the (mbuild.Compound().name or mbuild.Box()).

        * "molecule" applies the force field using the molecule's name

            WARNING: This is the atom's name for a single atom molecule (1 atom/bead  = molecule).

            Molecule > 1 atom ----> uses the "mbuild.Compound().name" (1 level above the atoms/beads)
            is the molecule's name.  This "mb.Compound().name" (1 level above the atoms/beads) needs
            to be used in the Charmm object's residue_list and forcefield_selection (if >1 force field), and
            will be the residue name in the PSF, PDB, and FF files.

            Molecule = 1 atom/bead  ----> uses the "atom/bead's name" is the molecule's name.

            This "atom/bead's name" needs to be used in the Charmm object's residue_list and
            forcefield_selection (if >1 force field), and will be the residue name in the PSF, PDB, and FF files.

            NOTE: Non-bonded zeolites or other fixed structures without bonds will use the
            "atom/bead's name" as the molecule's name, if they are single non-bonded atoms.
            However, the user may want to use the "group" option instead for this type of system,
            if applicable (see the "group" option).

            Example (Charmm_writer selects the user changeable "ETH", in the Charmm object residue_list and
            forcefield_selection (if >1 force field), which sets the residue "ETH" in the PSF, PDB, and FF files).

            >>> ethane = mbuild.load("CC", smiles=True)
            >>> ethane.name = "ETH"

            >>> ethane_box = mbuild.fill_box(
            >>>     compound=[ethane],
            >>>     n_compounds=[100],
            >>>     box=[4, 4, 4]
            >>> )

            Example (Charmm_writer must to select the non-user changeable "_CH4" (per the foyer TraPPE force field),
            in the Charmm object residue_list and  forcefield_selection (if >1 force field),
            which sets the residue "_CH4" in the PSF, PDB, and FF files).

            >>> methane_ua_bead_name = "_CH4"
            >>> methane_child_bead = mbuild.Compound(name=methane_ua_bead_name)
            >>> methane_box = mbuild.fill_box(
            >>>    compound=methane_child_bead,
            >>>    n_compounds=4,
            >>>    box=[1, 2, 3]
            >>> )
            methane_box.name = "MET"

            Example (Charmm_writer must to select the non-user changeable "Na" and "Cl"
            in the Charmm object residue_list and  forcefield_selection (if >1 force field),
            which sets the residues "Na" and "Cl" in the PSF, PDB, and FF files).

            >>> sodium_atom_name = "Na"
            >>> sodium_child_atom = mbuild.Compound(name=sodium_atom_name)
            >>> sodium = mb.Compound(name="SOD")
            >>> sodium.add(sodium_child_atom, inherit_periodicity=False)

            >>> chloride_atom_name = "Cl"
            >>> chloride_child_bead = mbuild.Compound(name=chloride_atom_name)
            >>> chloride = mb.Compound(name="CHL")
            >>> chloride.add(chloride_child_atom, inherit_periodicity=False)

            >>> sodium_chloride_box = mbuild.fill_box(
            >>>     compound=[sodium, chloride],
            >>>     n_compounds=[4, 4],
            >>>     box=[1, 2, 3]
            >>> )

            Example zeolite (Charmm_writer must to select the non-user changeable "Si" and "O"
            in the Charmm object residue_list and  forcefield_selection (if >1 force field),
            which sets the residues "Si" and "O" in the PSF, PDB, and FF files):

            >>> lattice_cif_ETV_triclinic = load_cif(file_or_path=get_mosdef_gomc_fn("ETV_triclinic.cif"))
            >>> ETV_triclinic = lattice_cif_ETV_triclinic.populate(x=1, y=1, z=1)
            >>> ETV_triclinic.name = "ETV"

        * "group" applies the force field to any molecules under the mbuild.Compound().name (children).

            The "group" search starts at the top level if there is no mbuild.Compound().name found on that level,
            then the next level down (children) are searched, keeping moving down every time.

            WARNING: This "group" option will take all the molecules below it, regardless if they
            are selected to be separate from the group via the residue_list and forcefield_selection
            (if >1 force field).

            This is ideal to use when you are building simulation box(es) using mbuild.fill_box(),
            with molecules, and it allows you to add another level to single single atom molecules
            (1 atom/bead  = molecule) to rename the mbuild.Compound().name, changing the residue's
            name and allowing keeping the atom/bead name so the force field is applied properly.

            NOTE: This "group" option may be best for non-bonded zeolites or other fixed structures
            without bonds, if they are single non-bonded atoms. Using this "group" option, the user
            can select the residue name for the Charmm_writer's residue_list and  forcefield_selection
            (if >1 force field) to force field all the atoms with a single residue name, and output
            this residue name in the PSF, PDB, and FF files.

            Example (Charmm_writer select the user changeable "MET" in the Charmm object residue_list
            and  forcefield_selection (if >1 force field), which sets the residue "MET" in the
            PSF, PDB, and FF files):

            >>> methane_ua_bead_name = "_CH4"
            >>> methane_child_bead = mbuild.Compound(name=methane_ua_bead_name)
            >>> methane_box = mbuild.fill_box(compound=methane_child_bead, n_compounds=4, box=[1, 2, 3])
            >>> methane_box.name = "MET"

            Example (Charmm_writer select the user changeable "MET" in the Charmm object residue_list
            and  forcefield_selection (if >1 force field), which sets the residue "MET" in the
            PSF, PDB, and FF files):

            >>> methane_child_bead = mb.Compound(name="_CH4")
            >>> methane = mb.Compound(name="MET")
            >>> methane.add(methane_child_bead, inherit_periodicity=False)

            >>> box_liq = mb.fill_box(
            >>>     compound=methane,
            >>>     n_compounds=1230,
            >>>     box=[4.5, 4.5, 4.5]
            >>> )

            Example zeolite (Charmm_writer select the user changeable "ETV" in the Charmm object residue_list
            and  forcefield_selection (if >1 force field), which sets the residue
            "ETV" in the PSF, PDB, and FF files):

            >>> lattice_cif_ETV_triclinic = load_cif(file_or_path=get_mosdef_gomc_fn("ETV_triclinic.cif"))
            >>> ETV_triclinic = lattice_cif_ETV_triclinic.populate(x=1, y=1, z=1)
            >>> ETV_triclinic.name = "ETV"

    ff_filename: str, default =None
        If a string, it will write the  force field files that work in
        GOMC and NAMD structures.
    gomc_fix_bonds_angles: list, default=None
        When list of residues is provided, the selected residues will have
        their bonds and angles fixed in the GOMC engine.  This is specifically
        for the GOMC engine and it changes the residue's bond constants (Kbs)
        and angle constants (Kthetas) values to 999999999999 in the
        FF file (i.e., the .inp file).
    bead_to_atom_name_dict: dict, optional, default=None
        For all atom names/elements/beads with 2 or less digits, this converts
        the atom name in the GOMC psf and pdb files to a unique atom name,
        provided they do not exceed 3844 atoms (62^2) of the same name/element/bead
        per residue. For all atom names/elements/beads with 3 digits, this converts
        the atom name in the GOMC psf and pdb files to a unique atom name,
        provided they do not exceed 62 of the same name/element pre residue.

        * Example dictionary: {'_CH3':'C', '_CH2':'C', '_CH':'C', '_HC':'C'}

        * Example name structure: {atom_type: first_part_pf atom name_without_numbering}

    atom_type_naming_style: str, optional, default='all_unique', ('general' or 'all_unique')
        'general':
        WARNING: The 'general' convention is UNSAFE, and the EXPERT user SHOULD USE AT THEIR OWN RISK,
        making SURE ALL THE BONDED PARAMETERS HAVE THE SAME VALUES IN THE UTILIZED
        FORCE FIELD XMLs.  Also, this DOES NOT ENSURE that THERE ARE NO specific
        Foyer XML ATOM TYPE BONDED CONNECTIONS in the Foyer FORCE FIELD XMLs, instead of the Foyer
        atom class type bonded connections, which could RESULT IN AN INCORRECT FORCE FIELD
        PARAMETERIZATION.  This is UNSAFE to use even with the same force field XML file, so the
        EXPERT user SHOULD USE AT THEIR OWN RISK.

        The 'general' convention only tests if the sigma, epsilons, mass, and Mie-n values are
        identical between the different molecules (residues in this context) and their applied
        force fields and DOES NOT check that any or all of the bonded parameters have the same
        or conflicting values.

        The 'general' convention is where all the atom classes in the Foyer force field
        XML files are converted to the CHARMM-style atom types (FOYER ATOM CLASSES).
        The 'general' convention ONLY auto-checks that the sigma, epsilon, mass, and Mie-n values
        are the same and does not currently ensure all the bonded parameters are the same
        or conflicting between different force field XML files.
        If the sigma, epsilons, mass, and Mie-n values are the same between force fields
        the general method can be applied; if not, it defaults to the 'all_unique' method.

        Example of CHARMM style atom types in an all-atom ethane and ethanol system:
        * Ethane: alkane carbon = CT, alkane hydrogen = HC
        * Ethanol: alkane carbon = CT, alkane hydrogen = HC , oxygen in alcohol = OH, hydrogen in alcohol = OH

        This is only permitted when the following is true; otherwise it will default to the the 'all_unique':
        * All the MoSDeF force field XML's atom classes' non-bonded parameters
        (sigma, epsilon, mass, and Mie-n power constant) values are THE SAME.
        * If the general CHARMM style atom type in any residue/molecule's gomc_fix_bonds_angles,
        gomc_fix_bonds, or gomc_fix_angles NOT IN any other residue/molecule, the 'all_unique' type
        will be used.

        'all_unique':
        The 'all_unique' convention is the SAFE way to parameterize the system.
        The MoSDeF force field XML atom names within residue/molecule are all unique,
        where each one adds an alpha numberic value after the MoSDeF force field XML atom classes to
        ensure uniqueness within the within residue/molecule.
        The OPLS atom types/name do not require all the sigma, epsilon, mass values to be the same,
        but have less bonded class parameters.

        Example of CHARMM style atom types in an all-atom ethane and ethanol system:
        * Ethane: alkane carbon type 0 = CT0, alkane hydrogen type 0 = HC0
        * Ethanol: alkane carbon type 1 = CT1, alkane carbon type 2 = CT2,
        alkane hydrogen type 1 = HC1 , oxygen in alcohol type 0 = OH0, hydrogen in alcohol type 0 = OH0

        This is selected when auto-selected when:
        * All the MoSDeF force field XML's atom classes' non-bonded parameters
        (sigma, epsilon, mass, and Mie-n power constant) values are NOT THE SAME.
        * If the general CHARMM style atom type in any residue/molecule's gomc_fix_bonds_angles,
        gomc_fix_bonds, or gomc_fix_angles are IN any other residue/molecule.
    fix_residue: list  or None, default=None
        Changes occcur in the pdb file only.
        All the atoms in the residue are have their Beta values in the PDB file set to 1.00;
        Otherwise, they will be 0.00.
        NOTE: In GOMC, this only fixes atoms automatically as listed below.
        NOTE: In NAMD, these all fixes need to be manually set in the control file (please see NAMD manual).
        When residues are listed here, all the atoms in the residue are
        fixed and can not move via setting the Beta values in the PDB file to 1.00
        If neither fix_residue or fix_residue_in_box lists a
        residue or both equal None, then the Beta (Temperature factor) values for all the atoms
        in the residue are free to move in the simulation and Beta values
        in the PDB file is set to 0.00.
    fix_residue_in_box: list  or None, default=None
        Changes occcur in the pdb file only.
        All the atoms in the residue are have their Beta values in the PDB file set to 2.00;
        Otherwise, they will be 0.00.
        NOTE: In GOMC, this only fixes atoms automatically as listed below.
        NOTE: In NAMD, these all fixes need to be manually set in the control file (please see NAMD manual).
        When residues are listed here, all the atoms in the residue
        can move within the box but cannot be transferred between boxes
        via setting the Beta (Temperature factor) values in the PDB file to 2.00.
        If neither fix_residue or fix_residue_in_box lists a
        residue or both equal None, then the Beta values for all the atoms
        in the residue are free to move in the simulation and Beta values
        in the PDB file is set to 0.00.
        NOTE that this is mainly for GOMC but also applies for NAMD (please see NAMD manual).
    set_residue_pdb_occupancy_to_1: list  or None, default=None
        Changes occcur in the pdb file only.
        All the atoms in the residue are have their occupancy values in the PDB file set to 1.00;
        Otherwise, they will be 0.00.
        NOTE: In GOMC, This defines which atoms belong to which box for the GCMC and GEMC ensembles.
        NOTE: In NAMD, This can be used for fixes which are manually set in the control file (please see NAMD manual).

    Attributes
    ----------
    input_error: bool
        This error is typically incurred from an error in the user's input values.
        However, it could also be due to a bug, provided the user is inputting
        the data as this Class intends.
    structure_box_0: mbuild.compound.Compound
        The mbuild Compound for the input box 0
    structure_box_1: mbuild.compound.Compound or None, default=None
        The mbuild Compound for the input box 1
    filename_box_0: str
        The file name of the output file for structure_box_0.  Note: the extension should
        not be provided, as multiple extension (.pdb and .psf) are added to this name.
    filename_box_1: str or None , default=None
        The file name of the output file for structure_box_1.  Note: the extension should
        not be provided, as multiple extension (.pdb and .psf) are added to this name.
        (i.e., either structure_box_0 or structure_box_1).
    residues: list, [str, ..., str]
        Labels of unique residues in the Compound. Residues are assigned by
        checking against Compound.name.  Only supply residue names as 4 character
        strings, as the residue names are truncated to 4 characters to fit in the
        psf and pdb file.
    forcefield_selection: str or dictionary, default=None
        Apply a forcefield to the output file by selecting a force field XML file with
        its path or by using the standard force field name provided the `foyer` package.
        Note: to write the NAMD/GOMC force field, pdb, and psf files, the
        residues and forcefields must be provided in a str or
        dictionary.  If a dictionary is provided all residues must
        be specified to a force field.

        * Example dict for FF file: {'ETH': 'oplsaa.xml', 'OCT': 'path_to_file/trappe-ua.xml'}

        * Example str for FF file: 'path_to file/trappe-ua.xml'

        * Example dict for standard FF names: {'ETH': 'oplsaa', 'OCT': 'trappe-ua'}

        * Example str for standard FF names: 'trappe-ua'

        * Example of a mixed dict with both: {'ETH': 'oplsaa', 'OCT': 'path_to_file/'trappe-ua.xml'}

    ff_filename: str, default =None
        If a string, it will write the  force field files that work in
        GOMC and NAMD structures.
    gomc_fix_bonds_angles: list, default=None
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
    gomc_fix_bonds: list, default=None
        When list of residues is provided, the selected residues will have their
        relative bond energies ignored in the GOMC engine. Note that GOMC
        does not sample bond stretching. This is specifically
        for the GOMC engine and it changes the residue's bond constants (Kbs)
        values to 999999999999 in the FF file (i.e., the .inp file).
        If the residues are listed in either the gomc_fix_bonds or the gomc_fix_bonds_angles
        lists, the relative bond energy will be ignored.
        NOTE if this option is utilized it may cause issues if using the FF file in NAMD.
    gomc_fix_angles: list, default=None
        When list of residues is provided, the selected residues will have
        their angles fixed and will ignore the related angle energies in the GOMC engine.
        This is specifically for the GOMC engine and it changes the residue's angle
        constants (Kthetas) values to 999999999999 in the FF file (i.e., the .inp file),
        which fixes the angles and ignores related angle energy.
        If the residues are listed in either the gomc_fix_angles or the gomc_fix_bonds_angles
        lists, the angles will be fixed and the related angle energy will be ignored
        for that residue.
        NOTE if this option is utilized it may cause issues if using the FF file in NAMD.
    bead_to_atom_name_dict: dict, optional, default =None
        For all atom names/elements/beads with 2 or less digits, this converts
        the atom name in the GOMC psf and pdb files to a unique atom name,
        provided they do not exceed 3844 atoms (62^2) of the same name/element/bead
        per residue. For all atom names/elements/beads with 3 digits, this converts
        the atom name in the GOMC psf and pdb files to a unique atom name,
        provided they do not exceed 62 of the same name/element pre residue.

        * Example dictionary: {'_CH3':'C', '_CH2':'C', '_CH':'C', '_HC':'C'}

        * Example name structure: {atom_type: first_part_pf atom name_without_numbering}

    fix_residue: list  or None, default=None
        Changes occcur in the pdb file only.
        All the atoms in the residue are have their Beta values in the PDB file set to 1.00;
        Otherwise, they will be 0.00.
        NOTE: In GOMC, this only fixes atoms automatically as listed below.
        NOTE: In NAMD, these all fixes need to be manually set in the control file (please see NAMD manual).
        When residues are listed here, all the atoms in the residue are
        fixed and can not move via setting the Beta values in the PDB file to 1.00
        If neither fix_residue or fix_residue_in_box lists a
        residue or both equal None, then the Beta (Temperature factor) values for all the atoms
        in the residue are free to move in the simulation and Beta values
        in the PDB file is set to 0.00.
    fix_residue_in_box: list  or None, default=None
        Changes occcur in the pdb file only.
        All the atoms in the residue are have their Beta values in the PDB file set to 2.00;
        Otherwise, they will be 0.00.
        NOTE: In GOMC, this only fixes atoms automatically as listed below.
        NOTE: In NAMD, these all fixes need to be manually set in the control file (please see NAMD manual).
        When residues are listed here, all the atoms in the residue
        can move within the box but cannot be transferred between boxes
        via setting the Beta (Temperature factor) values in the PDB file to 2.00.
        If neither fix_residue or fix_residue_in_box lists a
        residue or both equal None, then the Beta values for all the atoms
        in the residue are free to move in the simulation and Beta values
        in the PDB file is set to 0.00.
        NOTE that this is mainly for GOMC but also applies for NAMD (please see NAMD manual).
    set_residue_pdb_occupancy_to_1: list  or None, default=None
        Changes occcur in the pdb file only.
        All the atoms in the residue are have their occupancy values in the PDB file set to 1.00;
        Otherwise, they will be 0.00.
        NOTE: In GOMC, This defines which atoms belong to which box for the GCMC and GEMC ensembles.
        NOTE: In NAMD, This can be used for fixes which are manually set in the control file (please see NAMD manual).
    box_0: Box
        The Box class that contains the attributes Lx, Ly, Lz for the length
        of the box 0 (units in nanometers (nm)). It also contains the xy, xz, and yz Tilt factors
        needed to displace an orthogonal box's xy face to its
        parallelepiped structure for box 0.
    box_1: Box
        The Box class that contains the attributes Lx, Ly, Lz for the length
        of the box 1 (units in nanometers (nm)). It also contains the xy, xz, and yz Tilt factors
        needed to displace an orthogonal box's xy face to its
        parallelepiped structure for box 0.
    box_0_vectors: numpy.ndarray, [[float, float, float], [float, float, float], [float, float, float]]
        Three (3) sets vectors for box 0 each with 3 float values, which represent
        the vectors for the Charmm-style systems (units in Angstroms (Ang))
    box_1_vectors: numpy.ndarray, [[float, float, float], [float, float, float], [float, float, float]]
        Three (3) sets vectors for box 1 each with 3 float values, which represent
        the vectors for the Charmm-style systems (units in Angstroms (Ang))
    topology_box_0_ff: gmso.Topology
        The box 0 topology (from structure_box_0) after all the provided
        force fields are applied.
    topology_box_1_ff: gmso.Topology
        The box 1 topology (from structure_box_1) after all the provided
        force fields are applied. This only exists if the box 1 structure
        (structure_box_1) is provided.
    residues_applied_list_box_0: list
        The residues in box 0 that were found and had the force fields applied to them.
    residues_applied_list_box_1: list
        The residues in box 1 that were found and had the force fields applied to them.
        This only exists if the box 1 structure (structure_box_1) is provided.
    boxes_for_simulation: int, [0, 1]
        The number of boxes used when writing the Charmm object and force fielding
        the system. If only box 0 is provided, the value is 0. If box 0 and box 1
        are provided, the value is 1.
    epsilon_kcal_per_mol_atom_type_dict: dict {str: float or int}
        The uniquely numbered atom type (key) and it's non-bonded epsilon coefficient in units
        of kcal/mol (value). The atom type is defined by the AtomClass_ResidueName
        (Example of a carbon atom in an ethane molecule, AtomClass_ResidueName --> CT_ETH).
    sigma_angstrom_atom_type_dict: dict {str: float or int}
        The uniquely numbered atom type (key) and it's non-bonded sigma coefficient in
        angstroms (value). The atom type is defined by the AtomClass_ResidueName
        (Example of a carbon atom in an ethane molecule, AtomClass_ResidueName --> CT_ETH).
    mie_n_atom_type_dict: dict {str: float, int, or None}
        The uniquely numbered atom type (key) and it's non-bonded unitless n coefficient (value).
        The atom type is defined by the AtomClass_ResidueName
        (Example of a carbon atom in an ethane molecule, AtomClass_ResidueName --> CT_ETH).
        NOTE: The value is None if a Mie FF is not used.
    exp6_alpha_atom_type_dict: dict {str: float, int, or None}
        The uniquely numbered atom type (key) and it's non-bonded unitless alpha coefficient (value).
        The atom type is defined by the AtomClass_ResidueName
        (Example of a carbon atom in an ethane molecule, AtomClass_ResidueName --> CT_ETH).
        NOTE: The value is None if a Exp6 FF is not used.
    nonbonded_1_4_dict: dict {str: float or int}
        The uniquely numbered atom type (key) and it's non-bonded 1-4 scaling factor (value).
        The atom type is defined by the AtomClass_ResidueName
        (Example of a carbon atom in an ethane molecule, AtomClass_ResidueName --> CT_ETH).

        NOTE: NAMD and GOMC can have multiple values for the LJ 1-4 scalers, since they are
        provided as an individual input for each atom type in the force field (.inp) file.
    combined_1_4_nonbonded_dict_per_residue
        The residue name/molecule (key) and it's non-bonded 1-4 coulombic scaling factor (value).

        NOTE: NAMD and GOMC can have multiple values for the LJ 1-4 scalers, since they are
        provided as an individual input for each atom type in the force field (.inp) file.
    electrostatic_1_4: float or int
        The non-bonded 1-4 coulombic scaling factor, which is the same for all the
        residues/molecules, regardless if differenct force fields are utilized.  Note: if
        1-4 coulombic scaling factor is not the same for all molecules the Charmm object
        will fail with an error.

        NOTE: NAMD and GOMC can not have mulitple electrostatic 1-4 scalers, since it is
        provided as a single input in their control files.
    combined_1_4_electrostatic_dict_per_residue: dict, {str: float or int}
        The residue name/molecule (key) and it's non-bonded 1-4 coulombic scaling factor (value).

        NOTE: NAMD and GOMC can not have mulitple electrostatic 1-4 scalers, since it is
        provided as a single input in their control files.
    combining_rule: str ('geometric' or 'lorentz'"'),
        The possible mixing/combining  rules are 'geometric' or 'lorentz',
        which provide the  geometric and arithmetic mixing rule, respectively.
        NOTE: Arithmetic means the 'lorentz' combining or mixing rule.
        NOTE: GMSO default to the 'lorentz' mixing rule if none is provided,
        and this writers default is the GMSO default.
        NOTE: NAMD and GOMC can not have multiple values for the combining_rule.
    forcefield_selection: str or dictionary, default=None
        Apply a forcefield to the output file by selecting a force field XML file with
        its path or by using the standard force field name provided the `foyer` package.
        Note: to write the NAMD/GOMC force field, pdb, and psf files, the
        residues and forcefields must be provided in a str or
        dictionary.  If a dictionary is provided all residues must
        be specified to a force field.

        * Example dict for FF file: {'ETH': 'oplsaa.xml', 'OCT': 'path_to_file/trappe-ua.xml'}

        * Example str for FF file: 'path_to file/trappe-ua.xml'

        * Example dict for standard FF names: {'ETH': 'oplsaa', 'OCT': 'trappe-ua'}

        * Example str for standard FF names: 'trappe-ua'

        * Example of a mixed dict with both: {'ETH': 'oplsaa', 'OCT': 'path_to_file/'trappe-ua.xml'}

    all_individual_atom_names_list: list
        A list of all the atom names for the combined structures
        (box 0 and box 1 (if supplied)), in order.
    all_residue_names_list: list
        A list of all the residue names for the combined structures
        (box 0 and box 1 (if supplied)), in order.
    max_residue_no: int
        The maximum number that the residue number will count to
        before restarting the counting back to 1, which is predetermined
        by the PDB format. This is a constant, which equals 9999
    max_resname_char: int
        The maximum number of characters allowed in the residue name,
        which is predetermined by the PDB format. This is a constant,
        which equals 4.
    all_res_unique_atom_name_dict: dict, {str: set(str, ..., str)}
        A dictionary that provides the residue names (keys) and a set
        of the unique atom names in the residue (value), for the
        combined structures (box 0 and box 1 (if supplied)).

    Notes
    -----
    Urey-Bradleys, and NBFIX are not currenly supported.
    Currently the NBFIX is not available but will be in the near future.
    OPLS, AMBER, and CHARMM forcefield styles are supported (without harmonic dihedrals),

    The atom typing is currently provided via a base 62 numbering (0 to 9, then capital and lowercase lettering A to Z).
    This base 62 numbering allows for (62)^2 unique atom types. There are 4 characters allowed
    for the atom type name and 2 for the base 62 addition making them unique if needed.  Otherwise,
    if the GMSO general class is used for the atom typing, then the 6 characters is allowed,
    because additional numbers are not required to make them unique.
    This specifically avoids X because this represents any atom type in the CHARMM-style force field files.

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

    GOMC standard LJ form = epsilon * ( (Rmin/r)**12 - 2*(Rmin/r)**6 )
    ---------> = 4*epsilon * ( (sigma/r)**12 - (sigma/r)**6 ), when converted to sigmas
    Both forms above are accepted and compared automatically, but all input FFs have to be of the above
    input forms, aside from the whole potential energy scaling factor.

    GOMC standard Mie form = (n/(n-m)) * (n/m)**(m/(n-m)) * epsilon * ((sigma/r)**n - (sigma/r)**m)
    where m = 6 --> (n/(n-6)) * (n/6)**(6/(n-6)) * epsilon * ((sigma/r)**n - (sigma/r)**6)
    The above form is accepted but only if all input FFs have the same form,
    aside from the whole potential energy scaling factor.

    GOMC standard Exp-6 form =
    alpha*epsilon/(alpha -6) * Exp( alpha*(1-r/Rmin) - (Rmin/r)**6 ), where r >= Rmax
    infinity , where r < Rmax
    The above form is accepted but only if all input FFs have the same form,
    aside from the whole potential energy scaling factor.
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
        atom_type_naming_style="all_unique",
        fix_residue=None,
        fix_residue_in_box=None,
        set_residue_pdb_occupancy_to_1=None,
        ff_filename=None,
        gmso_match_ff_by="molecule",
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
        self.set_residue_pdb_occupancy_to_1 = set_residue_pdb_occupancy_to_1
        self.ff_filename = ff_filename
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
                    '->Dictionary Ex: {"Water": "oplsaa", "OCT": "path/trappe-ua.xml"}, '
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
                print(f"FF forcefield_selection = {self.forcefield_selection}")

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

        if self.set_residue_pdb_occupancy_to_1 is not None and not isinstance(
            self.set_residue_pdb_occupancy_to_1, list
        ):
            self.input_error = True
            print_error_message = "ERROR: Please enter the set_residue_pdb_occupancy_to_1 in a list format."
            raise TypeError(print_error_message)

        if isinstance(self.set_residue_pdb_occupancy_to_1, list):
            for (
                set_residue_pdb_occupancy_to_1_q
            ) in self.set_residue_pdb_occupancy_to_1:
                if set_residue_pdb_occupancy_to_1_q not in self.residues:
                    self.input_error = True
                    print_error_message = (
                        "Error: Please ensure that all the residue names in the "
                        "set_residue_pdb_occupancy_to_1 list are also in the residues list."
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

        if not isinstance(
            atom_type_naming_style, str
        ) or atom_type_naming_style not in ["general", "all_unique"]:
            self.input_error = True
            print_error_message = (
                "ERROR: Please enter the atom_type_naming_style "
                "as a string, either 'general' or 'all_unique'."
            )
            raise TypeError(print_error_message)

        if not isinstance(gmso_match_ff_by, str) or gmso_match_ff_by not in [
            "molecule",
            "group",
        ]:
            self.input_error = True
            print_error_message = (
                "ERROR: Please enter the gmso_match_ff_by "
                "as a string, either 'molecule', 'group'."
            )
            raise TypeError(print_error_message)

        if self.structure_box_1:
            self.boxes_for_simulation = 2
        else:
            self.boxes_for_simulation = 1

        # Initialize is_tabulated flag for TABULATED potentials
        self.is_tabulated = False

        # Initialize ParaType flags for different force field formats
        self.ParaTypeCHARMM = False
        self.ParaTypeMie = False
        self.ParaTypeMARTINI = False

        # write the Force fields
        self.combined_1_4_nonbonded_dict_per_residue = {}
        self.combined_1_4_electrostatic_dict_per_residue = {}

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
                gmso_match_ff_by=gmso_match_ff_by,
                residues=self.residues,
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
                gmso_match_ff_by=gmso_match_ff_by,
                residues=self.residues,
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
            self.atom_types_dict_per_residue.update(self.atom_types_dict_box_0)
            self.atom_types_dict_per_residue.update(self.atom_types_dict_box_1)
            self.bond_types_dict_per_residue.update(self.bond_types_dict_box_0)
            self.bond_types_dict_per_residue.update(self.bond_types_dict_box_1)
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
            # check the combining rules for both boxes and merge them into 1 combining rule.
            # Remove the empty box's None value, using the combining rule for the box with molecules.
            if self.combining_rule_dict_box_0 == self.combining_rule_dict_box_1:
                self.combining_rule = self.combining_rule_dict_box_0
            elif self.combining_rule_dict_box_0 is None:
                self.combining_rule = self.combining_rule_dict_box_1
            elif self.combining_rule_dict_box_1 is None:
                self.combining_rule = self.combining_rule_dict_box_0
            else:
                print_error_message = (
                    f"ERROR: There are multiple combining or mixing rules "
                    f"GOMC will only accept a singular input for the mixing rules. "
                    f"The provided mixing rules are "
                    f"{self.combining_rule_dict_box_0} and {self.combining_rule_dict_box_1} "
                    f"for box 0 and 1, respectively"
                )
                raise ValueError(print_error_message)

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
            site_list_box_0_and_1 = [
                site for site in self.topology_box_0_and_1_ff.sites
            ]
            if len(site_list_box_0_and_1) == 0:
                self.input_error = True
                print_error_message = (
                    "ERROR: the submitted structure has no PDB coordinates, "
                    "so the PDB writer has terminated. "
                )
                raise ValueError(print_error_message)

            # Check if the box 0's charges sum to zero
            charge_list_box_0 = [
                site.atom_type.__dict__["charge_"].to("C") / u.elementary_charge
                for site in self.topology_box_0_ff.sites
            ]

            if len(charge_list_box_0) != 0:
                total_charge_box_0 = sum(charge_list_box_0)
                total_charge_box_0 = total_charge_box_0.to_value(
                    "(dimensionless)"
                )

                if round(total_charge_box_0, 6) != 0.0:
                    warn(
                        "System is not charge neutral for structure_box_0. "
                        "Total charge is {}.".format(total_charge_box_0)
                    )

            # Check if the box 1's charges sum to zero
            charge_list_box_1 = [
                site.atom_type.__dict__["charge_"].to("C") / u.elementary_charge
                for site in self.topology_box_1_ff.sites
            ]
            if len(charge_list_box_1) != 0:
                total_charge_box_1 = sum(charge_list_box_1)
                total_charge_box_1 = total_charge_box_1.to_value(
                    "(dimensionless)"
                )

                if round(total_charge_box_1, 6) != 0.0:
                    warn(
                        "System is not charge neutral for structure_box_1. "
                        "Total charge is {}.".format(total_charge_box_1)
                    )

            # Check if the box 0 and 1's charges sum to zero
            charge_list_box_0_and_1 = [
                site.atom_type.__dict__["charge_"].to("C") / u.elementary_charge
                for site in self.topology_box_0_and_1_ff.sites
            ]
            total_charge_box_0_and_1 = sum(charge_list_box_0_and_1).to_value(
                "(dimensionless)"
            )
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
                gmso_match_ff_by=gmso_match_ff_by,
                residues=self.residues,
                boxes_for_simulation=self.boxes_for_simulation,
            )

            self.atom_types_dict_per_residue.update(self.atom_types_dict_box_0)
            self.bond_types_dict_per_residue.update(self.bond_types_dict_box_0)
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
            self.combining_rule = self.combining_rule_dict_box_0

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
            site_list_box_0 = [site for site in self.topology_box_0_ff.sites]
            if len(site_list_box_0) == 0:
                self.input_error = True
                print_error_message = (
                    "ERROR: the submitted structure has no PDB coordinates, "
                    "so the PDB writer has terminated. "
                )
                raise ValueError(print_error_message)

            # Check if the box 0's charges sum to zero
            charge_list_box_0 = [
                site.atom_type.__dict__["charge_"].to("C") / u.elementary_charge
                for site in self.topology_box_0_ff.sites
            ]
            if len(charge_list_box_0) != 0:
                total_charge_box_0 = sum(charge_list_box_0)
                total_charge_box_0 = total_charge_box_0.to_value(
                    "(dimensionless)"
                )

                if round(total_charge_box_0, 6) != 0.0:
                    warn(
                        "System is not charge neutral for structure_box_0. "
                        "Total charge is {}.".format(total_charge_box_0)
                    )

        print(f"forcefield type from compound = {self.forcefield_selection}")
        print(
            f"coulomb14scale from compound = {self.combined_1_4_electrostatic_dict_per_residue}"
        )
        print(
            f"nonbonded14scale from compound = {self.combined_1_4_nonbonded_dict_per_residue}"
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
            key_iter = f"{site.__dict__['residue_name_']}_{site.atom_type.__dict__['name_']}"
            if key_iter not in self.atom_type_info_dict.keys():
                charge_value = (
                    site.atom_type.__dict__["charge_"].to("C")
                    / u.elementary_charge
                )
                charge_value = charge_value.to_value("(dimensionless)")

                self.atom_type_info_dict.update(
                    {
                        key_iter: {
                            "potential_expression_": site.atom_type.__dict__[
                                "potential_expression_"
                            ],
                            "mass_": site.atom_type.__dict__["mass_"].to("amu"),
                            "charge_": charge_value,
                            "atomclass_": site.atom_type.__dict__["atomclass_"],
                            "doi_": site.atom_type.__dict__["doi_"],
                            "overrides_": site.atom_type.__dict__["overrides_"],
                            "definition_": site.atom_type.__dict__[
                                "definition_"
                            ],
                            "description_": site.atom_type.__dict__[
                                "description_"
                            ],
                            "residue_name_": site.__dict__["residue_name_"],
                            "element_": site.__dict__["name_"],
                            "atom_type_": site.atom_type.__dict__["name_"],
                        }
                    }
                )

        # lock the atom_style and unit_style for GOMC. Can be inserted into variables
        # once more functionality is built in

        # change 'residue_name_' to "residue_name_"
        # change 'residue_number_' to "residue_number_"
        self.residues_types_classes = []
        for site in self.topology_selection.sites:
            self.residues_types_classes.append(
                (
                    str(site.__dict__["residue_name_"]),
                    str(site.atom_type.__dict__["atomclass_"]),
                    str(site.atom_type.__dict__["name_"]),
                )
            )
        self.unique_residues_types_classes = list(
            set(self.residues_types_classes)
        )
        self.unique_residues_types_classes.sort(
            key=lambda x: (x[0], x[1], x[2])
        )

        self.types = np.array(
            [
                f"{site_res_type_i[0]}_{site_res_type_i[2]}"
                for site_res_type_i in self.residues_types_classes
            ]
        )
        self.unique_types = list(set(self.types))
        self.unique_types = np.array(
            [
                f"{site_res_type_i[0]}_{site_res_type_i[2]}"
                for site_res_type_i in self.unique_residues_types_classes
            ]
        )
        print("self.unique_types = " + str(self.unique_types))

        # get the unique atom types
        self.classes = np.array(
            [
                f"{site_res_type_i[0]}_{site_res_type_i[1]}"
                for site_res_type_i in self.residues_types_classes
            ]
        )
        self.unique_classes = np.array(
            [
                f"{site_res_type_i[0]}_{site_res_type_i[1]}"
                for site_res_type_i in self.unique_residues_types_classes
            ]
        )

        # so the atom types can be converted the pdb and psf files
        # The version to use will be determined by if the sigma, epsilons, mie n,
        # non-boned, and bonded are the same for the general case
        # set the CHARMM style atom type character limit (without addition of number/letters for uniqueness)
        atom_type_char_limit = 6
        self.mosdef_residue_atom_name_to_unique_charmm_atom_type_dict = {}
        self.atom_name_to_all_unique_classes_added_alpha_nums = set()
        self.atom_name_to_general_classes = set()
        self.mosdef_residue_atom_name_to_general_charmm_atom_type_dict = {}
        self.mosdef_residue_atom_name_to_unique_charmm_atom_type_dict = {}
        self.mosdef_atom_name_to_general_charmm_atom_type_per_residue_dict = {}
        self.mosdef_atom_name_to_unique_charmm_atom_type_per_residue_dict = {}

        for (
            unique_residues_types_classes_k
        ) in self.unique_residues_types_classes:
            # if it is a bead use the x characters after the _ and if not use the first x

            if unique_residues_types_classes_k[1][0] == "_":
                mosdef_class_with_char_iter = unique_residues_types_classes_k[
                    1
                ][1:]

            else:
                mosdef_class_with_char_iter = unique_residues_types_classes_k[
                    1
                ][0:]

            # check to see if the mosdef atom type/name or atom class has an X in or _X in it.
            if (
                unique_residues_types_classes_k[1]
                or unique_residues_types_classes_k[2]
            ) == "X" or (
                unique_residues_types_classes_k[1][1:]
                or unique_residues_types_classes_k[2][1:]
            ) == "X":
                print_error = (
                    f"WARNING: The residue/molecule {unique_residues_types_classes_k[0]} has a"
                    f"mosdef atom type/name {unique_residues_types_classes_k[2]}  "
                    f"or atom class {unique_residues_types_classes_k[1]} with an X in it, "
                    f"which is not permitted in the gmso format.  If you want a wildcard "
                    f"type please use * instead of X."
                )
                raise ValueError(print_error)

            self.mosdef_residue_atom_name_to_general_charmm_atom_type_dict.update(
                {
                    f"{unique_residues_types_classes_k[0]}_{unique_residues_types_classes_k[2]}": mosdef_class_with_char_iter
                }
            )
            try:
                self.mosdef_atom_name_to_general_charmm_atom_type_per_residue_dict[
                    unique_residues_types_classes_k[0]
                ].update(
                    {
                        unique_residues_types_classes_k[
                            2
                        ]: mosdef_class_with_char_iter
                    }
                )
            except:
                self.mosdef_atom_name_to_general_charmm_atom_type_per_residue_dict.update(
                    {
                        unique_residues_types_classes_k[0]: {
                            unique_residues_types_classes_k[
                                2
                            ]: mosdef_class_with_char_iter
                        }
                    }
                )

            for unique_class_number_k in range(0, 10**6):
                unique_atom_name_and_classes_iter = (
                    f"{mosdef_class_with_char_iter}"
                    f"{base10_to_base62_alph_num(unique_class_number_k)}"
                )

                if (
                    unique_atom_name_and_classes_iter
                    not in self.atom_name_to_all_unique_classes_added_alpha_nums
                    or str(unique_atom_name_and_classes_iter) == "*"
                ):
                    self.atom_name_to_all_unique_classes_added_alpha_nums.add(
                        unique_atom_name_and_classes_iter
                    )

                    self.mosdef_residue_atom_name_to_unique_charmm_atom_type_dict.update(
                        {
                            f"{unique_residues_types_classes_k[0]}_{unique_residues_types_classes_k[2]}": unique_atom_name_and_classes_iter
                        }
                    )
                    try:
                        self.mosdef_atom_name_to_unique_charmm_atom_type_per_residue_dict[
                            unique_residues_types_classes_k[0]
                        ].update(
                            {
                                unique_residues_types_classes_k[
                                    2
                                ]: unique_atom_name_and_classes_iter
                            }
                        )
                    except:
                        self.mosdef_atom_name_to_unique_charmm_atom_type_per_residue_dict.update(
                            {
                                unique_residues_types_classes_k[0]: {
                                    unique_residues_types_classes_k[
                                        2
                                    ]: unique_atom_name_and_classes_iter
                                }
                            }
                        )

                    break

        self.masses = np.array(
            [
                site.atom_type.__dict__["mass_"].to_value("amu")
                for site in self.topology_selection.sites
            ]
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

        self.charges = np.array(
            [
                (
                    site.atom_type.__dict__["charge_"].to("C")
                    / u.elementary_charge
                ).to_value("(dimensionless)")
                for site in self.topology_selection.sites
            ]
        )
        self.charges_atom_type_dict = dict(
            [
                (atom_type, charge)
                for atom_type, charge in zip(self.types, self.charges)
            ]
        )
        self.charges_atom_class_dict = dict(
            [
                (atom_class, charge)
                for atom_class, charge in zip(self.classes, self.charges)
            ]
        )

        # normalize by sigma
        self.box_0 = Box(
            lengths=self.topology_box_0_ff.box.lengths,
            angles=self.topology_box_0_ff.box.angles,
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
                lengths=self.topology_box_1_ff.box.lengths,
                angles=self.topology_box_1_ff.box.angles,
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
            site.__dict__["residue_name_"]
            for site in self.topology_selection.sites
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

        # GOMC standard Exp-6 form =
        # alpha*epsilon/(alpha -6) * Exp( alpha*(1-r/Rmin) - (Rmin/r)**6 ), where r >= Rmax
        # infinity , where r < Rmax
        # The above form is accepted but only if all input FFs have the same form,
        # aside from the whole potential energy scaling factor.

        # Non-bonded potential energy (u).

        self.atom_type_experssion_and_scalar_combined = (
            get_atom_type_expressions_and_scalars(
                self.atom_types_dict_per_residue
            )
        )

        # find non-bonded expression to use
        all_NB_expression_forms_set = set()
        for atom_type_j in list(
            self.atom_type_experssion_and_scalar_combined.keys()
        ):
            expression_form_iter = (
                self.atom_type_experssion_and_scalar_combined[atom_type_j][
                    "expression_form"
                ]
            )
            all_NB_expression_forms_set.add(expression_form_iter)

        # Check if TABULATED potentials are being used
        self.is_tabulated = "TABULATED" in all_NB_expression_forms_set

        if len(all_NB_expression_forms_set) == 1:
            if "LJ" in all_NB_expression_forms_set:
                self.utilized_NB_expression = "LJ"
                self.ParaTypeCHARMM = True
            elif "Mie" in all_NB_expression_forms_set:
                self.utilized_NB_expression = "Mie"
                self.ParaTypeMie = True
            elif "Exp6" in all_NB_expression_forms_set:
                self.utilized_NB_expression = "Exp6"
                self.ParaTypeMARTINI = True
            elif "TABULATED" in all_NB_expression_forms_set:
                self.utilized_NB_expression = "TABULATED"
                self.ParaTypeMie = True
            else:
                raise ValueError(
                    "ERROR: The non-bonded equation type is not the LJ, Mie, Exp6, "
                    "or TABULATED potential, which are the only available non-bonded equation potentials."
                )

        elif len(all_NB_expression_forms_set) == 2:
            if ("LJ" in all_NB_expression_forms_set) and (
                "Mie" in all_NB_expression_forms_set
            ):
                self.utilized_NB_expression = "Mie"
                self.ParaTypeMie = True

            elif "Exp6" in all_NB_expression_forms_set:
                raise ValueError(
                    "ERROR: The 'Exp6' non-bonded equation type can not be used with the "
                    "LJ or Mie potentials."
                )

        else:
            raise ValueError(
                "ERROR: Only 1 or 2 differnt non-bonded equation types are supported at a time. "
                "Only 'LJ' and 'Mie' potential combinations are allowed, and they change the "
                "equation to the Mie potential form (see the GOMC manual)."
            )

        # calculate epsilons form LJ, Mie, and Exp6 forms
        # Also, scale the epsilon based on the gmso equation input scaler, compared to the standard form
        epsilons_kcal_per_mol = np.array(
            [
                site.atom_type.parameters["epsilon"]
                .to("kcal/mol", equivalence="thermal")
                .to_value()
                * self.atom_type_experssion_and_scalar_combined[
                    f'{site.__dict__["residue_name_"]}_{site.atom_type.__dict__["name_"]}'
                ]["expression_scalar"]
                for site in self.topology_selection.sites
            ]
        )
        self.epsilon_kcal_per_mol_atom_class_dict = dict(
            [
                (atom_class, epsilon)
                for atom_class, epsilon in zip(
                    self.classes, epsilons_kcal_per_mol
                )
            ]
        )
        self.epsilon_kcal_per_mol_atom_type_dict = dict(
            [
                (atom_type, epsilon)
                for atom_type, epsilon in zip(self.types, epsilons_kcal_per_mol)
            ]
        )

        # calculate the Mie FF expression and push LJ to Mie if a mix of both
        if self.utilized_NB_expression == "Mie":
            mie_n = []
            # The Mie m-constant must be six (m=6) for GOMC, per the general GMSO format
            # Therefore, we check if m=6 for all, and if not this writer will fail
            self.mie_m_required_value = 6

            for site in self.topology_selection.sites:
                atom_type_residue_iter = f"{site.__dict__['residue_name_']}_{site.atom_type.__dict__['name_']}"
                nonbonded_expresseion_iter = (
                    self.atom_type_experssion_and_scalar_combined[
                        atom_type_residue_iter
                    ]["expression_form"]
                )
                if nonbonded_expresseion_iter == "Mie":
                    # This set the n parameter of the FF Mie if the iteration has it
                    mie_n_iter = (
                        site.atom_type.parameters["n"]
                        .to("dimensionless")
                        .to_value()
                    )
                    mie_n.append(mie_n_iter)

                    # check if m = 6 for all, and if not this writer will fail
                    if (
                        site.atom_type.parameters["m"]
                        .to("dimensionless")
                        .to_value()
                        != self.mie_m_required_value
                    ):
                        print_error = (
                            f"ERROR: The Mie Potential atom class "
                            f"{site.__dict__['residue_name_']}_"
                            f"{site.atom_type.__dict__['atomclass_']} "
                            f"does not have an m-constant of 6 in the force field XML, "
                            f"which is required in GOMC and this file writer."
                        )
                        raise ValueError(print_error)
                elif nonbonded_expresseion_iter == "LJ":
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
        else:
            self.mie_n_atom_type_dict = None
            self.mie_n_atom_class_dict = None

        # get the sigma values for the LJ and Mie forms calculate sigmas based on FF type
        sigmas_angstrom = []
        if self.utilized_NB_expression in ["LJ", "Mie"]:
            for site in self.topology_selection.sites:
                sigmas_angstrom.append(
                    site.atom_type.parameters["sigma"].to("angstrom").to_value()
                )

            # List the sigma values for the LJ and Mie FF types
            self.sigma_angstrom_atom_type_dict = dict(
                [
                    (atom_type, sigma)
                    for atom_type, sigma in zip(self.types, sigmas_angstrom)
                ]
            )
            self.sigma_angstrom_atom_class_dict = dict(
                [
                    (atom_class, sigma)
                    for atom_class, sigma in zip(self.classes, sigmas_angstrom)
                ]
            )
            # For LJ and Mie, Exp6 attributes are not needed
            self.exp6_alpha_atom_class_dict = None
            self.exp6_alpha_atom_type_dict = None
            self.exp6_r_min_angstrom_atom_class_dict = None
            self.exp6_r_min_angstrom_atom_type_dict = None

        # get the sigma values for the Exp6 forms calculate sigmas based on FF type
        elif self.utilized_NB_expression in ["Exp6"]:
            for site in self.topology_selection.sites:
                atom_type_residue_iter = f"{site.__dict__['residue_name_']}_{site.atom_type.__dict__['name_']}"
                nonbonded_expresseion_iter = (
                    self.atom_type_experssion_and_scalar_combined[
                        atom_type_residue_iter
                    ]["expression_form"]
                )
            # Brad note: start here next time
            # get the exp6 alpha values
            exp6_alpha_unitless = np.array(
                [
                    site.atom_type.parameters["alpha"]
                    .to("dimensionless")
                    .to_value()
                    for site in self.topology_selection.sites
                ]
            )
            self.exp6_alpha_atom_class_dict = dict(
                [
                    (atom_class, alpha)
                    for atom_class, alpha in zip(
                        self.classes, exp6_alpha_unitless
                    )
                ]
            )
            self.exp6_alpha_atom_type_dict = dict(
                [
                    (atom_type, alpha)
                    for atom_type, alpha in zip(self.types, exp6_alpha_unitless)
                ]
            )

            # get the Rmin from the Exp6 solutions
            exp6_r_min_angstrom = np.array(
                [
                    site.atom_type.parameters["Rmin"].to("angstrom").to_value()
                    for site in self.topology_selection.sites
                ]
            )

            self.exp6_r_min_angstrom_atom_class_dict = dict(
                [
                    (atom_class, r_min)
                    for atom_class, r_min in zip(
                        self.classes, exp6_r_min_angstrom
                    )
                ]
            )
            self.exp6_r_min_angstrom_atom_type_dict = dict(
                [
                    (atom_type, r_min)
                    for atom_type, r_min in zip(self.types, exp6_r_min_angstrom)
                ]
            )

            # Get the Exp6 sigma atom_class_dict by looping the other atom_class_dict dictionaries.
            # Use the Exp6 alpha and Rmin values to numerically convert Rmin --> Sigma.
            # There is no analytical conversion.
            self.sigma_angstrom_atom_class_dict = {}
            for (
                exp6_key,
                exp6_value,
            ) in self.exp6_r_min_angstrom_atom_class_dict.items():
                r_min_exp6_iter = exp6_value

                # get the corresponding alpha for Exp6
                alpha_exp6_iter = self.exp6_alpha_atom_class_dict[exp6_key]

                # use scipy to numerically solve for sigma for atom class dict
                exp6_sigma_iter = _Exp6_Rmin_to_sigma_solver(
                    r_min_exp6_iter, alpha_exp6_iter
                )

                self.sigma_angstrom_atom_class_dict.update(
                    {exp6_key: exp6_sigma_iter}
                )

            # Get the Exp6 sigma atom_type_dict by looping the other atom_type_dict dictionaries.
            # Use the Exp6 alpha and Rmin values to numerically convert Rmin --> Sigma.
            # There is no analytical conversion.
            self.sigma_angstrom_atom_type_dict = {}
            for (
                exp6_key,
                exp6_value,
            ) in self.exp6_r_min_angstrom_atom_type_dict.items():
                r_min_exp6_iter = exp6_value

                # get the corresponding alpha for Exp6
                alpha_exp6_iter = self.exp6_alpha_atom_type_dict[exp6_key]

                # use scipy to numerically solve for sigma for atom type dict
                exp6_sigma_iter = _Exp6_Rmin_to_sigma_solver(
                    r_min_exp6_iter, alpha_exp6_iter
                )

                self.sigma_angstrom_atom_type_dict.update(
                    {exp6_key: exp6_sigma_iter}
                )

        # For TABULATED potentials, sigma and alpha values are not extracted from force field file
        # They are read from the tabulated data file instead
        elif self.utilized_NB_expression == "TABULATED":
            self.sigma_angstrom_atom_type_dict = None
            self.sigma_angstrom_atom_class_dict = None
            self.exp6_alpha_atom_class_dict = None
            self.exp6_alpha_atom_type_dict = None
            self.exp6_r_min_angstrom_atom_class_dict = None
            self.exp6_r_min_angstrom_atom_type_dict = None

        # Determine if we can use MOSDEF (foyer/gmso) atom classes or traditional CHARMM atom types,
        # instead of using MOSDEF (foyer/gmso) atom names.  MOSDEF (foyer/gmso) atom names usages for the
        # FF is designed for OPLS FF where the sigmas and epsilons are not always the same for the
        # atom types (atom class).  This means even for a different charge, there is a different
        # atom type (atom class), which we will avoid if at all possible.
        # For other FFs, where the sigma and epsilons are always the same for a given atom types (atom class),
        # allow the usage of the general atom types (atom class) to be written.  This means that as long
        # as the sigmas/epsilons and the non-bonded and bonded interactions match across the used FFs,
        # we can use the more general type of atom types (atom class).
        self.general_atom_type_class_style = True

        residues_for_all_fixed_bonds_angles = []
        if self.gomc_fix_bonds_angles is not None:
            for fixed_bond_angles_m in self.gomc_fix_bonds_angles:
                residues_for_all_fixed_bonds_angles.append(fixed_bond_angles_m)
        if gomc_fix_bonds is not None:
            for fixed_bond_angles_m in self.gomc_fix_bonds:
                residues_for_all_fixed_bonds_angles.append(fixed_bond_angles_m)
        if gomc_fix_angles is not None:
            for fixed_bond_angles_m in self.gomc_fix_angles:
                residues_for_all_fixed_bonds_angles.append(fixed_bond_angles_m)
        residues_for_all_fixed_bonds_angles = set(
            residues_for_all_fixed_bonds_angles
        )

        # check atom types to ensure all epsilons, sigmas, and masses with atom types in an atom class are the same
        print(
            f"self.epsilon_kcal_per_mol_atom_class_dict = {self.epsilon_kcal_per_mol_atom_class_dict}"
        )
        if self.utilized_NB_expression != "TABULATED":
            print(
                f"self.sigma_angstrom_atom_type_dict = {self.sigma_angstrom_atom_type_dict}"
            )
        print(f"self.mass_atom_type_dict = {self.mass_atom_type_dict}")
        atom_type_in_fixed_bond_angle_per_residue_dict = {}
        atom_class_in_fixed_bond_angle_per_residues_dict = {}
        atom_class_only_epsilon_dict = {}
        atom_class_only_sigma_dict = {}
        atom_class_only_mass_dict = {}
        atom_class_only_mie_n_dict = {}
        atom_class_only_exp6_alpha_dict = {}
        
        # Skip atom class consistency checks for TABULATED potentials
        # since parameters are read from the data file, not the force field
        if self.utilized_NB_expression != "TABULATED":
            for atom_type_key_iter in list(self.atom_type_info_dict.keys()):
                atom_class_j_iter = self.atom_type_info_dict[atom_type_key_iter][
                    "atomclass_"
                ]
                atom_residue_j_iter = self.atom_type_info_dict[atom_type_key_iter][
                    "residue_name_"
                ]
                atom_class_residue_j_iter = (
                    f"{atom_residue_j_iter}_{atom_class_j_iter}"
                )

                # atom class values
                atom_class_epsilon_kcal_per_mol_j_iter = (
                    self.epsilon_kcal_per_mol_atom_class_dict[
                        atom_class_residue_j_iter
                    ]
                )
                atom_class_sigmas_angstrom_j_iter = (
                    self.sigma_angstrom_atom_class_dict[atom_class_residue_j_iter]
                )
                atom_class_mass_amu_j_iter = self.mass_atom_class_dict[
                    atom_class_residue_j_iter
                ]

                atom_class_conficts_str = ""
                try:
                    if (
                        atom_class_epsilon_kcal_per_mol_j_iter
                        != atom_class_only_epsilon_dict[atom_class_j_iter]
                    ):
                        if atom_class_conficts_str == "":
                            atom_class_conficts_str = f"{'epsilon'}"
                        else:
                            atom_class_conficts_str = (
                                f"{atom_class_conficts_str}, {'epsilon'}"
                            )
                except:
                    atom_class_only_epsilon_dict.update(
                        {atom_class_j_iter: atom_class_epsilon_kcal_per_mol_j_iter}
                    )

                try:
                    if (
                        atom_class_sigmas_angstrom_j_iter
                        != atom_class_only_sigma_dict[atom_class_j_iter]
                    ):
                        if atom_class_conficts_str == "":
                            atom_class_conficts_str = f"{'sigma'}"
                        else:
                            atom_class_conficts_str = (
                                f"{atom_class_conficts_str}, {'sigma'}"
                            )
                except:
                    atom_class_only_sigma_dict.update(
                        {atom_class_j_iter: atom_class_sigmas_angstrom_j_iter}
                    )

                try:
                    if (
                        atom_class_mass_amu_j_iter
                        != atom_class_only_mass_dict[atom_class_j_iter]
                    ):
                        if atom_class_conficts_str == "":
                            atom_class_conficts_str = f"{'atom mass'}"
                        else:
                            atom_class_conficts_str = (
                                f"{atom_class_conficts_str}, {'atom mass'}"
                            )
                except:
                    atom_class_only_mass_dict.update(
                        {atom_class_j_iter: atom_class_mass_amu_j_iter}
                    )

                if self.utilized_NB_expression == "Mie":
                    atom_class_mie_n_iter = self.mie_n_atom_class_dict[
                        atom_class_residue_j_iter
                    ]
                    try:
                        if (
                            atom_class_mie_n_iter
                            != atom_class_only_mie_n_dict[atom_class_j_iter]
                        ):
                            if atom_class_conficts_str == "":
                                atom_class_conficts_str = f"{'mie_n'}"
                            else:
                                atom_class_conficts_str = (
                                    f"{atom_class_conficts_str}, {'mie_n'}"
                                )

                            print_error = (
                                f"ERROR: Only the same Mie n values are permitted for an atom class. "
                                f"The {atom_class_mie_n_iter} atom class has different n values"
                            )
                            raise ValueError(print_error)

                    except:
                        atom_class_only_mie_n_dict.update(
                            {atom_class_j_iter: atom_class_mie_n_iter}
                        )

                if self.utilized_NB_expression == "Exp6":
                    atom_class_exp6_alpha_iter = self.exp6_alpha_atom_class_dict[
                        atom_class_residue_j_iter
                    ]
                    try:
                        if (
                            atom_class_exp6_alpha_iter
                            != atom_class_only_exp6_alpha_dict[atom_class_j_iter]
                        ):
                            if atom_class_conficts_str == "":
                                atom_class_conficts_str = f"{'alpha'}"
                            else:
                                atom_class_conficts_str = (
                                    f"{atom_class_conficts_str}, {'alpha'}"
                                )

                            print_error = (
                                f"ERROR: Only the same Exp6 alpha values are permitted for an atom class. "
                                f"The {atom_class_exp6_alpha_iter} atom class has different alpha values"
                            )
                            raise ValueError(print_error)

                    except:
                        atom_class_only_exp6_alpha_dict.update(
                            {atom_class_j_iter: atom_class_exp6_alpha_iter}
                        )

                if atom_class_conficts_str != "":
                    print_warning = (
                        f"WARNING: The {atom_class_residue_j_iter} CHARMM atom type (Foyer atom class) "
                        f"has different {atom_class_conficts_str} values."
                        f"Only the same {atom_class_conficts_str} values are permitted to use the "
                        f"MoSDeF atom class as the CHARMM style atom type. "
                        f"Therefore the Foyer atom names are being used as "
                        f"the CHARMM style atom type, meaning atom_type_naming_style='all_unique'."
                    )
                    warn(print_warning)
                    self.general_atom_type_class_style = False

                if atom_residue_j_iter in residues_for_all_fixed_bonds_angles:
                    try:
                        atom_class_in_fixed_bond_angle_per_residues_dict[
                            atom_residue_j_iter
                        ].add(atom_class_j_iter)
                    except:
                        atom_class_in_fixed_bond_angle_per_residues_dict[
                            atom_residue_j_iter
                        ] = set([atom_class_j_iter])

            print(f"atom_class_conficts_str = {atom_class_conficts_str}")
        else:
            # For TABULATED potentials, skip the atom class consistency checks
            # since parameters are read from the data file
            pass

        # check if any atom classes are in the residue other than the fixed one
        # get all atom classes that are fixed
        all_fixed_atom_class_set = set()
        for (
            residue_p,
            all_fixed_class_set_p,
        ) in atom_class_in_fixed_bond_angle_per_residues_dict.items():
            for all_fixed_class_p in all_fixed_class_set_p:
                all_fixed_atom_class_set.add(all_fixed_class_p)
        for (
            residue_q,
            dict_q,
        ) in (
            self.mosdef_atom_name_to_general_charmm_atom_type_per_residue_dict.items()
        ):
            if residue_q not in residues_for_all_fixed_bonds_angles:
                for atom_class_fixed_iter in dict_q.values():
                    if atom_class_fixed_iter in all_fixed_atom_class_set:
                        self.general_atom_type_class_style = False

        # *********************
        # need a flag from gmso sorting to say we have only atom
        # classes or atom types in the FF params.
        # if any at types, need to use member_types not classes for the writeout (start)
        # *********************

        # determine if the general atom class is used or the atom type is needed
        self.combinded_residue_bond_types = []
        for res_x in self.bond_types_dict_per_residue.keys():
            if self.bond_types_dict_per_residue[res_x] is not None:
                for bond_type_x in self.bond_types_dict_per_residue[res_x][
                    "bond_types"
                ]:
                    self.combinded_residue_bond_types.append(bond_type_x)

        self.combinded_residue_angle_types = []
        for res_x in self.angle_types_dict_per_residue.keys():
            if self.angle_types_dict_per_residue[res_x] is not None:
                for angle_type_x in self.angle_types_dict_per_residue[res_x][
                    "angle_types"
                ]:
                    self.combinded_residue_angle_types.append(angle_type_x)

        self.combinded_residue_dihedral_types = []
        for res_x in self.dihedral_types_dict_per_residue.keys():
            if self.dihedral_types_dict_per_residue[res_x] is not None:
                for dihedral_type_x in self.dihedral_types_dict_per_residue[
                    res_x
                ]["dihedral_types"]:
                    self.combinded_residue_dihedral_types.append(
                        dihedral_type_x
                    )

        self.combinded_residue_improper_types = []
        for res_x in self.improper_types_dict_per_residue.keys():
            if self.improper_types_dict_per_residue[res_x] is not None:
                for improper_type_x in self.improper_types_dict_per_residue[
                    res_x
                ]["improper_types"]:
                    self.combinded_residue_improper_types.append(
                        improper_type_x
                    )

        # ********************************************************************************
        # **************************     ADD THIS HERE (START)      **********************
        # ********************************************************************************
        # if any of the bond, angle, dih, imp return a atom type specifed bonded param ,
        # return self.general_atom_type_class_style= False
        # ********************************************************************************
        # **************************     ADD THIS HERE (END)      **********************
        # ********************************************************************************

        # *********************
        # need a flag from gmso sorting to say we have only atom
        # classes or atom types in the FF params.
        # if any at types, need to use member_types not classes for the writeout (start)
        # *********************

        # determine if the general atom class is used or the atom type is needed
        if (
            self.general_atom_type_class_style
            and atom_type_naming_style == "general"
        ):
            self.mosdef_atom_name_to_atom_type_dict = (
                self.mosdef_residue_atom_name_to_general_charmm_atom_type_dict
            )
            atom_type_class_style_naming_str = "MoSDeF_atom_class"

            warn(
                f"WARNING: atom_type_naming_style = 'general'\n"
                f"WARNING: The 'general' convention is UNSAFE, and the EXPERT user SHOULD USE AT THEIR OWN RISK, "
                f"making SURE ALL THE BONDED PARAMETERS HAVE THE SAME VALUES IN THE UTILIZED "
                f"FORCE FIELD XMLs.  Also, this DOES NOT ENSURE that THERE ARE NO specific "
                f"Foyer XML ATOM TYPE BONDED CONNECTIONS in the Foyer FORCE FIELD XMLs, instead of the Foyer "
                f"atom class type bonded connections, which could RESULT IN AN INCORRECT FORCE FIELD "
                f"PARAMETERIZATION.  This is UNSAFE to use even with the same force field XML file, so the "
                f"EXPERT user SHOULD USE AT THEIR OWN RISK.\n"
                f"The 'general' convention only tests if the sigma, epsilons, mass, and Mie-n values are "
                f"identical between the different molecules (residues in this context) and their applied "
                f"force fields and DOES NOT check that any or all of the bonded parameters have the same "
                f"or conflicting values. "
            )

        else:
            self.mosdef_atom_name_to_atom_type_dict = (
                self.mosdef_residue_atom_name_to_unique_charmm_atom_type_dict
            )
            atom_type_class_style_naming_str = "MoSDeF_atom_type_or_name"

        print(
            f"mosdef_atom_name_to_atom_type_dict = {self.mosdef_atom_name_to_atom_type_dict}"
        )
        # determine if the charmm atom types are longer than 6 characters
        for (
            key_name_j,
            value_name_j,
        ) in self.mosdef_atom_name_to_atom_type_dict.items():
            if len(value_name_j) > atom_type_char_limit:
                print_error = (
                    f"ERROR: The {key_name_j} residue and mosdef atom name using the "
                    f"{atom_type_class_style_naming_str} methodology exceeds the "
                    f"character limit of {atom_type_char_limit}, which is required for "
                    f"the CHARMM style atom types. Please format the force field "
                    f"xml files to get them under these 6 characters by allowing the "
                    f"general {'MoSDeF_atom_class'} to be used or otherwise "
                    f"shortening the atom {'MoSDeF_atom_class'} names. "
                    f"NOTE: The {'MoSDeF_atom_class'} must allow for additional "
                    f"alphanumberic additions at the end of it, making unique CHARMM atom "
                    f"types (MoSDeF_atom_classes); typically this would allow for 4"
                    f"characters in the MoSDeF_atom_classes."
                )
                raise ValueError(print_error)

        self.nonbonded_1_4_dict = dict(
            [
                (
                    atom_type,
                    self.combined_1_4_nonbonded_dict_per_residue[
                        residues_all_list
                    ],
                )
                for atom_type, residues_all_list in zip(
                    self.types, residues_all_list
                )
            ]
        )

        # ensure all 1,4-coulombic or electrostatic scaling factors are the same,
        # and if not set to the same if None is provided
        electrostatic_1_4_set = set()
        for (
            residue_p
        ) in self.combined_1_4_electrostatic_dict_per_residue.keys():
            if (
                self.combined_1_4_electrostatic_dict_per_residue[residue_p]
                is None
            ):
                warn(
                    "WARNING: The 1,4-electrostatic scaling factor for the {} residue "
                    "that was provided as None. This may mean that force field file "
                    "does not need the 1,4-electrostatic scaling factor, because it does not have 4 connected "
                    "atoms in the 1-4 configuration, or there may be an error in the "
                    "force field file. "
                    "".format(residue_p)
                )
            else:
                electrostatic_1_4_set.add(
                    self.combined_1_4_electrostatic_dict_per_residue[residue_p]
                )
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
                warn(
                    "WARNING: The 1,4-nonbonded scaling factor for the {} residue "
                    "that was provided as None. This may mean that force field file "
                    "does not need the 1,4-nonbonded scaling factor, because it does not have 4 connected "
                    "atoms in the 1-4 configuration, or there may be an error in the force field file. "
                    "Since the {} residue provided None for the 1,4-nonbonded scaling factor, it "
                    "is being set to 0. "
                    "".format(residue_r, residue_r)
                )
                self.combined_1_4_nonbonded_dict_per_residue[residue_r] = 0

        if self.combining_rule not in ["lorentz", "geometric"]:
            self.input_error = True
            print_warning_message = (
                f"Error: The {self.combining_rule} combining or mixing rules were provided, but "
                f"only the 'lorentz' or 'geometric' mixing rules are allowed. "
                f"NOTE: Arithmetic means the 'lorentz' combining or mixing rule."
            )
            warn(print_warning_message)

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

        # this sets the residues chain length to a max limit
        self.max_residue_no = 9999
        self.max_resname_char = 4

        for q_i in range(0, len(list_of_topologies)):
            stuct_only_iteration = stuct_only[q_i]

            # caluculate the atom name and unique atom names
            residue_names_list = []
            residue_id_list = []
            res_no_chain_iter_corrected_list = []
            segment_id_list = []
            segment_id_iter = 0  # starts at 0 as it adds 1 the first iter (being 0 -> A for 1st iteration)
            for site in stuct_only_iteration.sites:
                residue_names_list.append(site.__dict__["residue_name_"])

                residue_id_list_iter = site.__dict__["residue_number_"]
                residue_id_list.append(residue_id_list_iter)

                res_id_adder = int(
                    (residue_id_list_iter % self.max_residue_no)
                    % self.max_residue_no
                )
                if int(res_id_adder) == 0:
                    res_no_iteration_q = int(self.max_residue_no)
                else:
                    res_no_iteration_q = res_id_adder

                if (
                    len(res_no_chain_iter_corrected_list) > 0
                    and res_no_chain_iter_corrected_list[-1]
                    == self.max_residue_no
                    and res_no_iteration_q == 1
                ):
                    segment_id_iter += 1

                res_no_chain_iter_corrected_list.append(res_no_iteration_q)
                segment_id_list.append(base10_to_base52_alph(segment_id_iter))

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

                self.residue_names_list_box_0 = residue_names_list
                self.residue_id_list_box_0 = residue_id_list
                self.res_no_chain_iter_corrected_list_box_0 = (
                    res_no_chain_iter_corrected_list
                )
                self.segment_id_list_box_0 = segment_id_list

            else:
                self.all_individual_atom_names_list = (
                    self.all_individual_atom_names_list
                    + individual_atom_names_list_iter
                )

                self.all_residue_names_list = (
                    self.all_residue_names_list + residue_names_list
                )

                self.residue_names_list_box_1 = residue_names_list
                self.residue_id_list_box_1 = residue_id_list
                self.res_no_chain_iter_corrected_list_box_1 = (
                    res_no_chain_iter_corrected_list
                )
                self.segment_id_list_box_1 = segment_id_list

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
                if self.all_res_unique_atom_name_dict in list(
                    self.all_res_unique_atom_name_dict.keys()
                ):
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
                    residue_id_list.append(site.__dict__["residue_number_"])
                    residue_names_list.append(site.__dict__["residue_name_"])

                for k, site in enumerate(self.topology_box_1_ff.sites):
                    residue_id_list.append(site.__dict__["residue_number_"])
                    residue_names_list.append(site.__dict__["residue_name_"])

            else:
                for k, site in enumerate(self.topology_box_0_ff.sites):
                    residue_id_list.append(site.__dict__["residue_number_"])
                    residue_names_list.append(site.__dict__["residue_name_"])

            for n in range(0, len(residue_names_list)):
                if residue_names_list[n] not in self.residues:
                    print("residue_names_list = " + str(residue_names_list))
                    self.input_error = True
                    print_error_message = "ERROR: Please specifiy all residues (residues) in a list"
                    raise ValueError(print_error_message)

            # Start writing the force field (.inp) file
            with open(self.ff_filename, "w") as data:
                if self.structure_box_1:
                    data.write(
                        f"* {self.filename_box_0} and {self.filename_box_1} "
                        f"- created by MoSDeF-GOMC using the on {date_time}\n"
                    )
                else:
                    data.write(
                        f"* {self.filename_box_0} - created by MoSDeF-GOMC using the on {date_time}.\n"
                    )

                data.write(
                    f"* These parameters use the non-bonded {self.utilized_NB_expression} form "
                    f"--- with these force field(s) via MoSDef  {self.forcefield_selection}.\n"
                )
                data.write(
                    f"*  1-4 electrostatic scaling = {self.combined_1_4_electrostatic_dict_per_residue} "
                    f", and 1-4 non-bonded scaling = {self.combined_1_4_nonbonded_dict_per_residue}"
                    f", and non-bonded mixing rule = {self.combining_rule}\n\n"
                )
                data.write(
                    "* {:15d} atoms\n".format(self.topology_selection.n_sites)
                )

                data.write(
                    "* {:15d} bonds\n".format(self.topology_selection.n_bonds)
                )
                data.write(
                    "* {:15d} angles\n".format(self.topology_selection.n_angles)
                )
                data.write(
                    "* {:15d} dihedrals\n".format(
                        self.topology_selection.n_dihedrals
                    )
                )
                data.write(
                    "* {:15d} impropers\n\n".format(
                        self.topology_selection.n_impropers
                    )
                )

                data.write("\n* masses\n\n")
                data.write(
                    "! {:15s} {:15s} ! {}\n".format(
                        "atom_types",
                        "mass",
                        "atomClass_ResidueName",
                    )
                )
                atom_mass_decimals_round = 4
                atom_mass_list = []
                for atom_type, mass in self.mass_atom_type_dict.items():
                    atom_mass_list.append(
                        [
                            self.mosdef_atom_name_to_atom_type_dict[atom_type],
                            str(
                                np.round(
                                    mass, decimals=atom_mass_decimals_round
                                ),
                            ),
                            atom_type,
                        ]
                    )

                    # check for duplicates, for duplicate class or atom type
                    mass_same_count = 0  # Start at 0 (1 always found) count numbr of values that are the same
                    for mass_check_i in range(0, len(atom_mass_list)):

                        # check if atomclass or atomtype is the same
                        if (
                            atom_mass_list[-1][0]
                            == atom_mass_list[mass_check_i][0]
                        ):
                            mass_same_count += 1

                            # check if values are the same since atomclass or atomtype is the same
                            if (
                                atom_mass_list[-1][1]
                                != atom_mass_list[mass_check_i][1]
                            ):
                                raise ValueError(
                                    f"ERROR: The same atomclass or atomtype in the "
                                    f"force field are have different {'mass'} values.\n"
                                    f"{atom_mass_list[-1]} != {atom_mass_list[mass_check_i]} "
                                )

                    # Only print for 1 time so mass_same_count=1 (starts at 0)
                    if mass_same_count == 1:
                        mass_format = "* {:15s} {:15s} ! {:25s}\n"
                        data.write(
                            mass_format.format(
                                atom_mass_list[-1][0],
                                atom_mass_list[-1][1],
                                atom_mass_list[-1][2],
                            )
                        )

                # Bond coefficients
                if len(self.topology_selection.bond_types) > 0:
                    data.write("\n")
                    data.write("BONDS * harmonic\n")
                    data.write("! \n")
                    data.write("! V(bond) = Kb(b - b0)**2\n")
                    data.write("! \n")
                    data.write(
                        "! Kb: kcal/mol/A**2 (LJ) and  K/A**2 (Mie and Exp6)\n"
                    )
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
                            "extended_type_2",
                        )
                    )

                    bond_k_kcal_per_mol_round_decimals = 6
                    bond_k_Kelvin_round_decimals = 4
                    bond_distance_round_decimals = 6

                    bond_values_list = []
                    for bond_type_x in self.combinded_residue_bond_types:
                        res_x = bond_type_x.__dict__["tags_"]["resname"]
                        bond_members_iter = bond_type_x.member_types

                        gomc_bond_form_with_scalar = "1 * k * (r-r_eq)**2"

                        [
                            bond_type_str_iter,
                            bond_eqn_scalar_iter,
                        ] = evaluate_harmonic_bond_format_with_scaler(
                            bond_type_x.expression, gomc_bond_form_with_scalar
                        )
                        if bond_type_str_iter != "HarmonicBondPotential":
                            raise TypeError(
                                f"ERROR: The {res_x} residue's "
                                f"bond types or classes does not have a "
                                f"{'HarmonicBondPotential'} bond potential, which "
                                f"is the only supported bond potential."
                            )

                        # the bond energy value is dependant of the non-bonded form
                        bond_energy_value_LJ_units_iter = (
                            _check_convert_bond_k_constant_units(
                                str(bond_members_iter),
                                bond_eqn_scalar_iter
                                * bond_type_x.parameters["k"],
                                "kcal/mol/angstrom**2",
                            ).to_value("kcal/mol/angstrom**2")
                        )

                        if self.utilized_NB_expression == "LJ":
                            bond_energy_value_iter = np.round(
                                bond_energy_value_LJ_units_iter,
                                decimals=bond_k_kcal_per_mol_round_decimals,
                            )
                        elif self.utilized_NB_expression in ["Mie", "Exp6"]:
                            bond_energy_value_Mie_Exp6_units_iter = (
                                _check_convert_bond_k_constant_units(
                                    str(bond_members_iter),
                                    bond_eqn_scalar_iter
                                    * bond_type_x.parameters["k"],
                                    "K/angstrom**2",
                                ).to_value("K/angstrom**2")
                            )
                            bond_energy_value_iter = np.round(
                                bond_energy_value_Mie_Exp6_units_iter,
                                decimals=bond_k_Kelvin_round_decimals,
                            )
                        elif self.utilized_NB_expression == "TABULATED":
                            # For TABULATED potentials, use same units as Mie/Exp6 (Kelvin)
                            bond_energy_value_Mie_Exp6_units_iter = (
                                _check_convert_bond_k_constant_units(
                                    str(bond_members_iter),
                                    bond_eqn_scalar_iter
                                    * bond_type_x.parameters["k"],
                                    "K/angstrom**2",
                                ).to_value("K/angstrom**2")
                            )
                            bond_energy_value_iter = np.round(
                                bond_energy_value_Mie_Exp6_units_iter,
                                decimals=bond_k_Kelvin_round_decimals,
                            )

                        if (
                            (self.gomc_fix_bonds_angles is not None)
                            and (str(res_x) in self.gomc_fix_bonds_angles)
                        ) or (
                            (
                                (self.gomc_fix_bonds is not None)
                                and (str(res_x) in self.gomc_fix_bonds)
                            )
                        ):
                            fix_bond_k_value = "999999999999"
                            bond_values_list.append(
                                [
                                    self.mosdef_atom_name_to_atom_type_dict[
                                        f"{res_x}_{bond_members_iter[0]}"
                                    ],
                                    self.mosdef_atom_name_to_atom_type_dict[
                                        f"{res_x}_{bond_members_iter[1]}"
                                    ],
                                    str(fix_bond_k_value),
                                    str(
                                        np.round(
                                            bond_type_x.parameters[
                                                "r_eq"
                                            ].to_value("angstrom"),
                                            decimals=bond_distance_round_decimals,
                                        )
                                    ),
                                    f"{res_x}_{bond_members_iter[0]}",
                                    f"{res_x}_{bond_members_iter[1]}",
                                ]
                            )

                        else:
                            bond_values_list.append(
                                [
                                    self.mosdef_atom_name_to_atom_type_dict[
                                        f"{res_x}_{bond_members_iter[0]}"
                                    ],
                                    self.mosdef_atom_name_to_atom_type_dict[
                                        f"{res_x}_{bond_members_iter[1]}"
                                    ],
                                    str(bond_energy_value_iter),
                                    str(
                                        np.round(
                                            bond_type_x.parameters[
                                                "r_eq"
                                            ].to_value("angstrom"),
                                            decimals=bond_distance_round_decimals,
                                        )
                                    ),
                                    f"{res_x}_{bond_members_iter[0]}",
                                    f"{res_x}_{bond_members_iter[1]}",
                                ]
                            )

                        # check for duplicates, for duplicate class or atom type
                        bond_same_count = 0  # Start at 0 (1 always found) count numbr of values that are the same
                        for bond_check_i in range(0, len(bond_values_list)):

                            # check if atomclass or atomtype is the same (check regular and reverse bond order)
                            if (
                                bond_values_list[-1][0]
                                == bond_values_list[bond_check_i][0]
                                and bond_values_list[-1][1]
                                == bond_values_list[bond_check_i][1]
                            ) or (
                                bond_values_list[-1][0]
                                == bond_values_list[bond_check_i][1]
                                and bond_values_list[-1][1]
                                == bond_values_list[bond_check_i][0]
                            ):
                                bond_same_count += 1

                                # check if values are the same since atomclass or atomtype is the same
                                if (
                                    bond_values_list[-1][2]
                                    != bond_values_list[bond_check_i][2]
                                    or bond_values_list[-1][3]
                                    != bond_values_list[bond_check_i][3]
                                ):
                                    raise ValueError(
                                        f"ERROR: The same atomclass or atomtype in the "
                                        f"force field are have different {'bond'} values.\n"
                                        f"{bond_values_list[-1]} != {bond_values_list[bond_check_i]} "
                                    )

                        # Only print for 1 time so bond_same_count=1 (starts at 0)
                        if bond_same_count == 1:
                            bond_format = (
                                "{:10s} {:10s} {:15s} {:15s} ! {:20s} {:20s}\n"
                            )
                            data.write(
                                bond_format.format(
                                    bond_values_list[-1][0],
                                    bond_values_list[-1][1],
                                    bond_values_list[-1][2],
                                    bond_values_list[-1][3],
                                    bond_values_list[-1][4],
                                    bond_values_list[-1][5],
                                )
                            )

                # Angle coefficients
                if len(self.topology_selection.angle_types):
                    data.write("\nANGLES * harmonic\n")
                    data.write("! \n")
                    data.write("! V(angle) = Ktheta(Theta - Theta0)**2\n")
                    data.write("! \n")
                    data.write(
                        "! Ktheta: kcal/mol/rad**2 (LJ) and  K/rad**2 (Mie and Exp6)\n"
                    )
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
                            "extended_type_3",
                        )
                    )

                    angle_k_kcal_per_mol_round_decimals = 6
                    angle_k_Kelvin_round_decimals = 4
                    angle_degree_round_decimals = 6
                    angle_values_list = []
                    for angle_type_x in self.combinded_residue_angle_types:
                        res_x = angle_type_x.__dict__["tags_"]["resname"]
                        angle_members_iter = angle_type_x.member_types

                        gomc_angle_form_with_scalar = (
                            "1 * k * (theta - theta_eq)**2"
                        )

                        [
                            angle_type_str_iter,
                            angle_eqn_scalar_iter,
                        ] = evaluate_harmonic_angle_format_with_scaler(
                            angle_type_x.expression, gomc_angle_form_with_scalar
                        )

                        if angle_type_str_iter != "HarmonicAnglePotential":
                            raise TypeError(
                                f"ERROR: The {res_x} residue's "
                                f"angle types or classes does not have a "
                                f"{'HarmonicAnglePotential'} angle potential, which "
                                f"is the only supported angle potential."
                            )

                        # the angle energy value is dependant of the non-bonded form
                        angle_energy_value_LJ_units_iter = (
                            _check_convert_angle_k_constant_units(
                                str(angle_members_iter),
                                angle_eqn_scalar_iter
                                * angle_type_x.parameters["k"],
                                "kcal/mol/rad**2",
                            ).to_value("kcal/mol/rad**2")
                        )

                        if self.utilized_NB_expression == "LJ":
                            angle_energy_value_iter = np.round(
                                angle_energy_value_LJ_units_iter,
                                decimals=angle_k_kcal_per_mol_round_decimals,
                            )
                        elif self.utilized_NB_expression in ["Mie", "Exp6"]:
                            angle_energy_value_Mie_Exp6_units_iter = (
                                _check_convert_angle_k_constant_units(
                                    str(angle_members_iter),
                                    angle_eqn_scalar_iter
                                    * angle_type_x.parameters["k"],
                                    "K/rad**2",
                                ).to_value("K/rad**2")
                            )

                            angle_energy_value_iter = np.round(
                                angle_energy_value_Mie_Exp6_units_iter,
                                decimals=angle_k_Kelvin_round_decimals,
                            )
                        elif self.utilized_NB_expression == "TABULATED":
                            # For TABULATED potentials, use same units as Mie/Exp6 (Kelvin)
                            angle_energy_value_Mie_Exp6_units_iter = (
                                _check_convert_angle_k_constant_units(
                                    str(angle_members_iter),
                                    angle_eqn_scalar_iter
                                    * angle_type_x.parameters["k"],
                                    "K/rad**2",
                                ).to_value("K/rad**2")
                            )

                            angle_energy_value_iter = np.round(
                                angle_energy_value_Mie_Exp6_units_iter,
                                decimals=angle_k_Kelvin_round_decimals,
                            )

                        if (
                            (self.gomc_fix_bonds_angles is not None)
                            and (str(res_x) in self.gomc_fix_bonds_angles)
                        ) or (
                            (
                                (self.gomc_fix_angles is not None)
                                and (str(res_x) in self.gomc_fix_angles)
                            )
                        ):
                            fix_angle_k_value = "999999999999"
                            angle_values_list.append(
                                [
                                    self.mosdef_atom_name_to_atom_type_dict[
                                        f"{res_x}_{angle_members_iter[0]}"
                                    ],
                                    self.mosdef_atom_name_to_atom_type_dict[
                                        f"{res_x}_{angle_members_iter[1]}"
                                    ],
                                    self.mosdef_atom_name_to_atom_type_dict[
                                        f"{res_x}_{angle_members_iter[2]}"
                                    ],
                                    str(fix_angle_k_value),
                                    str(
                                        np.round(
                                            angle_type_x.parameters[
                                                "theta_eq"
                                            ].to_value("degree"),
                                            decimals=angle_degree_round_decimals,
                                        )
                                    ),
                                    f"{res_x}_{angle_members_iter[0]}",
                                    f"{res_x}_{angle_members_iter[1]}",
                                    f"{res_x}_{angle_members_iter[2]}",
                                ]
                            )

                        else:
                            angle_values_list.append(
                                [
                                    self.mosdef_atom_name_to_atom_type_dict[
                                        f"{res_x}_{angle_members_iter[0]}"
                                    ],
                                    self.mosdef_atom_name_to_atom_type_dict[
                                        f"{res_x}_{angle_members_iter[1]}"
                                    ],
                                    self.mosdef_atom_name_to_atom_type_dict[
                                        f"{res_x}_{angle_members_iter[2]}"
                                    ],
                                    str(angle_energy_value_iter),
                                    str(
                                        np.round(
                                            angle_type_x.parameters[
                                                "theta_eq"
                                            ].to_value("degree"),
                                            decimals=angle_degree_round_decimals,
                                        )
                                    ),
                                    f"{res_x}_{angle_members_iter[0]}",
                                    f"{res_x}_{angle_members_iter[1]}",
                                    f"{res_x}_{angle_members_iter[2]}",
                                ]
                            )

                        # check for duplicates, for duplicate class or atom type
                        angle_same_count = 0  # Start at 0 (1 always found) count numbr of values that are the same
                        for angle_check_i in range(0, len(angle_values_list)):

                            # check if atomclass or atomtype is the same (check regular and reverse angle order)
                            if angle_values_list[-1][1] == angle_values_list[
                                angle_check_i
                            ][1] and (
                                (
                                    angle_values_list[-1][0]
                                    == angle_values_list[angle_check_i][0]
                                    and angle_values_list[-1][2]
                                    == angle_values_list[angle_check_i][2]
                                )
                                or (
                                    angle_values_list[-1][0]
                                    == angle_values_list[angle_check_i][2]
                                    and angle_values_list[-1][2]
                                    == angle_values_list[angle_check_i][0]
                                )
                            ):
                                angle_same_count += 1

                                # check if values are the same since atomclass or atomtype is the same
                                if (
                                    angle_values_list[-1][3]
                                    != angle_values_list[angle_check_i][3]
                                    or angle_values_list[-1][4]
                                    != angle_values_list[angle_check_i][4]
                                ):
                                    raise ValueError(
                                        f"ERROR: The same atomclass or atomtype in the "
                                        f"force field are have different {'angle'} values.\n"
                                        f"{angle_values_list[-1]} != {angle_values_list[angle_check_i]} "
                                    )

                        # Only print for 1 time so angle_same_count=1 (starts at 0)
                        if angle_same_count == 1:
                            angle_format = "{:10s} {:10s} {:10s} {:15s} {:15s} ! {:20s} {:20s} {:20s}\n"
                            data.write(
                                angle_format.format(
                                    angle_values_list[-1][0],
                                    angle_values_list[-1][1],
                                    angle_values_list[-1][2],
                                    angle_values_list[-1][3],
                                    angle_values_list[-1][4],
                                    angle_values_list[-1][5],
                                    angle_values_list[-1][6],
                                    angle_values_list[-1][7],
                                )
                            )

                # Dihedral coefficients
                if len(self.topology_selection.dihedral_types):
                    list_if_large_error_dihedral_overall = []

                    list_if_largest_error_abs_values_for_dihedral_overall = []
                    list_dihedral_overall_error = []

                    list_if_abs_max_values_for_dihedral_overall = []
                    list_dihedral_atoms_all_dihedral_overall = []

                    if self.utilized_NB_expression in ["LJ"]:
                        data.write("\nDIHEDRALS * CHARMM\n")
                    elif self.utilized_NB_expression in ["Mie"]:
                        data.write("\nDIHEDRALS * Mie\n")
                    elif self.utilized_NB_expression in ["Exp6"]:
                        data.write("\nDIHEDRALS * Exp6\n")
                    elif self.utilized_NB_expression in ["TABULATED"]:
                        data.write("\nDIHEDRALS * Mie\n")
                    data.write("! \n")
                    data.write(
                        "! V(dihedral) = Kchi(1 + cos(n(chi) - delta)), where delta also called chi0 \n"
                    )
                    data.write(
                        "! NOTE: For the CHARMM FF n=0 is a harmonic dihedral, which is not supported. "
                        "CHARMM FF where n=0 -->  V(dihedral) = Kchi(chi - chi0)**2) \n"
                    )
                    data.write("! \n")
                    data.write("! Kchi: kcal/mol (LJ) and K (Mie and Exp6)\n")
                    data.write("! n: multiplicity\n")
                    data.write("! delta: degrees\n")
                    data.write("! \n")
                    data.write(
                        "! Kchi (kcal/mol) = Kchi_K (K) * Boltz. const.\n"
                    )
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
                            "extended_type_4",
                        )
                    )

                dihedral_k_kcal_per_mol_round_decimals = 6
                dihedral_k_Kelvin_round_decimals = 4
                dihedral_phase_degree_round_decimals = 6

                no_pi = np.pi
                dihedral_steps = 5 * 10 ** (-3)
                dihedral_range = 2 * no_pi
                dihedral_no_steps = int(dihedral_range / dihedral_steps) + 1

                dihedral_values_list = []
                for dihedral_type_x in self.combinded_residue_dihedral_types:
                    res_x = dihedral_type_x.__dict__["tags_"]["resname"]
                    dihedral_members_iter = dihedral_type_x.member_types

                    dihedral_type_str_iter = None
                    dihedral_eqn_scalar_iter = None

                    # Check if OPLSTorsionPotential with 0.5
                    # NOTE: It does not compare if 0.5 and 1/2 (rounding error)
                    if (
                        dihedral_type_str_iter is None
                        and dihedral_eqn_scalar_iter is None
                    ):
                        # Check if OPLSTorsionPotential with 0.5
                        # NOTE: It does not compare if 0.5 and 1/2 (rounding error)
                        OPLSTorsionPotential_form_with_scalar = (
                            "1/2 * k0 + "
                            "1/2 * k1 * (1 + cos(phi)) + "
                            "1/2 * k2 * (1 - cos(2*phi)) + "
                            "1/2 * k3 * (1 + cos(3*phi)) + "
                            "1/2 * k4 * (1 - cos(4*phi))"
                        )
                        [
                            dihedral_type_str_iter,
                            dihedral_eqn_scalar_iter,
                        ] = evaluate_OPLS_torsion_format_with_scaler(
                            dihedral_type_x.expression,
                            OPLSTorsionPotential_form_with_scalar,
                        )

                    # Check if OPLSTorsionPotential with 0.5
                    # NOTE: It does not compare if 0.5 and 1/2 (rounding error)
                    if (
                        dihedral_type_str_iter is None
                        and dihedral_eqn_scalar_iter is None
                    ):
                        # Check if OPLSTorsionPotential with 0.5
                        # note it does not compare if 0.5 and 1/2 (rounding error)
                        OPLSTorsionPotential_form_with_scalar = (
                            "0.5 * k0 + "
                            "0.5 * k1 * (1 + cos(phi)) + "
                            "0.5 * k2 * (1 - cos(2*phi)) + "
                            "0.5 * k3 * (1 + cos(3*phi)) + "
                            "0.5 * k4 * (1 - cos(4*phi))"
                        )
                        [
                            dihedral_type_str_iter,
                            dihedral_eqn_scalar_iter,
                        ] = evaluate_OPLS_torsion_format_with_scaler(
                            dihedral_type_x.expression,
                            OPLSTorsionPotential_form_with_scalar,
                        )

                    if (
                        dihedral_type_str_iter is None
                        and dihedral_eqn_scalar_iter is None
                    ):
                        # Check if RyckaertBellemansTorsionPotential
                        RyckaertBellemansTorsionPotential_form_with_scalar = (
                            "c0 * cos(phi)**0 + "
                            "c1 * cos(phi)**1 + "
                            "c2 * cos(phi)**2 + "
                            "c3 * cos(phi)**3 + "
                            "c4 * cos(phi)**4 + "
                            "c5 * cos(phi)**5"
                        )
                        [
                            dihedral_type_str_iter,
                            dihedral_eqn_scalar_iter,
                        ] = evaluate_RB_torsion_format_with_scaler(
                            dihedral_type_x.expression,
                            RyckaertBellemansTorsionPotential_form_with_scalar,
                        )

                    if (
                        dihedral_type_str_iter is None
                        and dihedral_eqn_scalar_iter is None
                    ):
                        # Check if PeriodicTorsionPotential
                        PeriodicTorsionPotential_form_with_scalar = (
                            "k * (1 + cos(n * phi - phi_eq))"
                        )
                        [
                            dihedral_type_str_iter,
                            dihedral_eqn_scalar_iter,
                        ] = evaluate_periodic_torsion_format_with_scaler(
                            dihedral_type_x.expression,
                            PeriodicTorsionPotential_form_with_scalar,
                        )

                    if (
                        dihedral_type_str_iter is None
                        and dihedral_eqn_scalar_iter is None
                    ):
                        # Check if HarmonicTorsionPotential
                        HarmonicTorsionPotential_form_with_scalar = (
                            "0.5 * k * (phi - phi_eq)**2"
                        )
                        [
                            dihedral_type_str_iter,
                            dihedral_eqn_scalar_iter,
                        ] = evaluate_harmonic_torsion_format_with_scaler(
                            dihedral_type_x.expression,
                            HarmonicTorsionPotential_form_with_scalar,
                        )

                    if dihedral_type_str_iter == "HarmonicTorsionPotential":
                        raise TypeError(
                            f"ERROR: The {res_x} residue has a "
                            f"{'HarmonicTorsionPotential'} torsion potential, which "
                            f"is not currently supported in this writer."
                        )
                    elif (
                        dihedral_type_str_iter is None
                        and dihedral_eqn_scalar_iter is None
                    ):
                        raise TypeError(
                            f"ERROR: The {res_x} residue and associated force field "
                            f"has at least one unsupported dihdedral. "
                            f"The only supported dihedrals are "
                            f"{'OPLSTorsionPotential'}, {'PeriodicTorsionPotential'}, and "
                            f"{'RyckaertBellemansTorsionPotential'}."
                        )

                    # convert dihedral to Periodic style
                    if dihedral_type_str_iter == "OPLSTorsionPotential":
                        f0 = (
                            dihedral_eqn_scalar_iter
                            * dihedral_type_x.parameters["k0"].to_value(
                                "kcal/mol", equivalence="thermal"
                            )
                        )
                        f1 = (
                            dihedral_eqn_scalar_iter
                            * dihedral_type_x.parameters["k1"].to_value(
                                "kcal/mol", equivalence="thermal"
                            )
                        )
                        f2 = (
                            dihedral_eqn_scalar_iter
                            * dihedral_type_x.parameters["k2"].to_value(
                                "kcal/mol", equivalence="thermal"
                            )
                        )
                        f3 = (
                            dihedral_eqn_scalar_iter
                            * dihedral_type_x.parameters["k3"].to_value(
                                "kcal/mol", equivalence="thermal"
                            )
                        )
                        f4 = (
                            dihedral_eqn_scalar_iter
                            * dihedral_type_x.parameters["k4"].to_value(
                                "kcal/mol", equivalence="thermal"
                            )
                        )

                        [
                            [K0, n0, d0],
                            [K1, n1, d1],
                            [K2, n2, d2],
                            [K3, n3, d3],
                            [K4, n4, d4],
                            [K5, n5, d5],
                        ] = OPLS_to_periodic(f0, f1, f2, f3, f4)

                    elif (
                        dihedral_type_str_iter
                        == "RyckaertBellemansTorsionPotential"
                    ):
                        c0 = (
                            dihedral_eqn_scalar_iter
                            * dihedral_type_x.parameters["c0"].to_value(
                                "kcal/mol", equivalence="thermal"
                            )
                        )
                        c1 = (
                            dihedral_eqn_scalar_iter
                            * dihedral_type_x.parameters["c1"].to_value(
                                "kcal/mol", equivalence="thermal"
                            )
                        )
                        c2 = (
                            dihedral_eqn_scalar_iter
                            * dihedral_type_x.parameters["c2"].to_value(
                                "kcal/mol", equivalence="thermal"
                            )
                        )
                        c3 = (
                            dihedral_eqn_scalar_iter
                            * dihedral_type_x.parameters["c3"].to_value(
                                "kcal/mol", equivalence="thermal"
                            )
                        )
                        c4 = (
                            dihedral_eqn_scalar_iter
                            * dihedral_type_x.parameters["c4"].to_value(
                                "kcal/mol", equivalence="thermal"
                            )
                        )
                        c5 = (
                            dihedral_eqn_scalar_iter
                            * dihedral_type_x.parameters["c5"].to_value(
                                "kcal/mol", equivalence="thermal"
                            )
                        )

                        [
                            [K0, n0, d0],
                            [K1, n1, d1],
                            [K2, n2, d2],
                            [K3, n3, d3],
                            [K4, n4, d4],
                            [K5, n5, d5],
                        ] = RB_to_periodic(c0, c1, c2, c3, c4, c5)

                    elif dihedral_type_str_iter == "PeriodicTorsionPotential":
                        # get the number of periodic dihedral
                        dihedral_kx_ph_eqx_x_variables_set_iter = {
                            "k",
                            "n",
                            "phi_eq",
                        }

                        # get k values
                        if isinstance(
                            dihedral_type_x.parameters["k"],
                            u.array.unyt_quantity,
                        ):
                            periodic_dihedrals_k_values = u.unyt_array(
                                [
                                    dihedral_type_x.parameters["k"].to_value(
                                        "kcal/mol", equivalence="thermal"
                                    )
                                ],
                                "kcal/mol",
                            )
                        else:
                            periodic_dihedrals_k_values = (
                                dihedral_type_x.parameters["k"]
                            )

                        periodic_dihedrals_k_values_len = len(
                            periodic_dihedrals_k_values
                        )

                        # get n values
                        if isinstance(
                            dihedral_type_x.parameters["n"],
                            u.array.unyt_quantity,
                        ):
                            periodic_dihedrals_n_values = u.unyt_array(
                                [
                                    dihedral_type_x.parameters["n"].to_value(
                                        "(dimensionless)"
                                    )
                                ],
                                "(dimensionless)",
                            )
                        else:
                            periodic_dihedrals_n_values = (
                                dihedral_type_x.parameters["n"]
                            )

                        periodic_dihedrals_n_values_len = len(
                            periodic_dihedrals_n_values
                        )

                        # get phi_eq values
                        if isinstance(
                            dihedral_type_x.parameters["phi_eq"],
                            u.array.unyt_quantity,
                        ):
                            periodic_dihedrals_phi_eq_values = u.unyt_array(
                                [
                                    dihedral_type_x.parameters[
                                        "phi_eq"
                                    ].to_value("degree")
                                ],
                                "degree",
                            )
                        else:
                            periodic_dihedrals_phi_eq_values = (
                                dihedral_type_x.parameters["phi_eq"]
                            )

                        periodic_dihedrals_phi_eq_values_len = len(
                            periodic_dihedrals_phi_eq_values
                        )

                        if (
                            periodic_dihedrals_k_values_len
                            == periodic_dihedrals_n_values_len
                            == periodic_dihedrals_phi_eq_values_len
                        ):
                            number_of_periodic_dihedrals_int = (
                                periodic_dihedrals_k_values_len
                            )

                        else:
                            print_error = (
                                f"ERROR: The periodic dihedral values "
                                f"{dihedral_kx_ph_eqx_x_variables_set_iter} "
                                f"do not have all the values for each set."
                            )
                            raise ValueError(print_error)

                        # scale the Kx, nx, and phi_eqx to the kcal/mol values
                        Kx_dihedral = (
                            dihedral_eqn_scalar_iter
                            * periodic_dihedrals_k_values.to_value(
                                "kcal/mol", equivalence="thermal"
                            )
                        )
                        nx_dihedral = periodic_dihedrals_n_values.to_value(
                            "(dimensionless)"
                        )
                        phi_eqx_dihedral = (
                            periodic_dihedrals_phi_eq_values.to_value("degree")
                        )

                        # sort the Kx, nx, and phi_eq values and put them in the
                        # K0-K5, n0-n5, and phi_eq0-phi_eq5 form
                        # set all the values initially to zero
                        [
                            [K0, n0, d0],
                            [K1, n1, d1],
                            [K2, n2, d2],
                            [K3, n3, d3],
                            [K4, n4, d4],
                            [K5, n5, d5],
                        ] = [
                            [0, 0, 90],
                            [0, 1, 0],
                            [0, 2, 0],
                            [0, 3, 0],
                            [0, 4, 0],
                            [0, 5, 0],
                        ]

                        # test if all n values unique
                        unique_n_values_iter = []
                        for periodic_j in range(
                            0, number_of_periodic_dihedrals_int
                        ):
                            Kx_dihedral_iter = Kx_dihedral[periodic_j]
                            nx_dihedral_iter = nx_dihedral[periodic_j]
                            phi_eqx_dihedral_iter = phi_eqx_dihedral[periodic_j]

                            # for nx_dihedral_j in nx_dihedral_iter:
                            if nx_dihedral_iter in unique_n_values_iter:
                                print_error = (
                                    f"ERROR: mulitple PeriodicTorsionPotential "
                                    f"n values of {nx_dihedral_iter} "
                                    f"were found for the same torsion. Only 1 of each "
                                    f"n values are allowed per PeriodicTorsionPotential."
                                )
                                raise ValueError(print_error)
                            else:
                                unique_n_values_iter.append(nx_dihedral_iter)

                            # test if all n values are between 0 and 5
                            if nx_dihedral_iter not in [0, 1, 2, 3, 4, 5]:
                                print_value_error = (
                                    "ERROR: The "
                                    f"{dihedral_members_iter[0]}-"
                                    f"{dihedral_members_iter[1]}-"
                                    f"{dihedral_members_iter[2]}-"
                                    f"{dihedral_members_iter[3]} "
                                    f"dihedral "
                                    f"{'PeriodicTorsionPotential'} "
                                    "does not have n values from 0 to 5."
                                )
                                raise ValueError(print_value_error)

                            # change the Kx, nx, and dx values to non-zero if they exist
                            if nx_dihedral_iter == 0:
                                K0 = Kx_dihedral_iter * (
                                    1
                                    + np.cos(
                                        -phi_eqx_dihedral_iter * np.pi / 180
                                    )
                                )
                                n0 = nx_dihedral_iter
                                d0 = phi_eqx_dihedral_iter

                            elif nx_dihedral_iter == 1:
                                K1 = Kx_dihedral_iter
                                n1 = nx_dihedral_iter
                                d1 = phi_eqx_dihedral_iter

                            elif nx_dihedral_iter == 2:
                                K2 = Kx_dihedral_iter
                                n2 = nx_dihedral_iter
                                d2 = phi_eqx_dihedral_iter

                            elif nx_dihedral_iter == 3:
                                K3 = Kx_dihedral_iter
                                n3 = nx_dihedral_iter
                                d3 = phi_eqx_dihedral_iter

                            elif nx_dihedral_iter == 4:
                                K4 = Kx_dihedral_iter
                                n4 = nx_dihedral_iter
                                d4 = phi_eqx_dihedral_iter

                            elif nx_dihedral_iter == 5:
                                K5 = Kx_dihedral_iter
                                n5 = nx_dihedral_iter
                                d5 = phi_eqx_dihedral_iter

                    # test dihedral conversion for errors
                    input_dihedral_to_periodic_abs_diff = []
                    for i in range(0, dihedral_no_steps + 1):
                        phi = i * dihedral_steps
                        psi = phi - no_pi

                        # calulate the periodic dihedral (PeriodicTorsionPotential)
                        periodic_dihedral_calc = (
                            K0 * (1 + np.cos(n0 * phi - d0 * no_pi / 180))
                            + K1 * (1 + np.cos(n1 * phi - d1 * no_pi / 180))
                            + K2 * (1 + np.cos(n2 * phi - d2 * no_pi / 180))
                            + K3 * (1 + np.cos(n3 * phi - d3 * no_pi / 180))
                            + K4 * (1 + np.cos(n4 * phi - d4 * no_pi / 180))
                            + K5 * (1 + np.cos(n5 * phi - d5 * no_pi / 180))
                        )

                        if dihedral_type_str_iter == "OPLSTorsionPotential":
                            input_dihedral_calc = (
                                f0 / 2
                                + f1 / 2 * (1 + np.cos(1 * phi))
                                + f2 / 2 * (1 - np.cos(2 * phi))
                                + f3 / 2 * (1 + np.cos(3 * phi))
                                + f4 / 2 * (1 - np.cos(4 * phi))
                            )

                        elif (
                            dihedral_type_str_iter
                            == "RyckaertBellemansTorsionPotential"
                        ):
                            input_dihedral_calc = (
                                c0
                                + c1 * np.cos(psi) ** 1
                                + c2 * np.cos(psi) ** 2
                                + c3 * np.cos(psi) ** 3
                                + c4 * np.cos(psi) ** 4
                                + c5 * np.cos(psi) ** 5
                            )

                        elif (
                            dihedral_type_str_iter == "PeriodicTorsionPotential"
                        ):
                            input_dihedral_calc = (
                                K0 * (1 + np.cos(0 * phi - d0 * no_pi / 180))
                                + K1 * (1 + np.cos(1 * phi - d1 * no_pi / 180))
                                + K2 * (1 + np.cos(2 * phi - d2 * no_pi / 180))
                                + K3 * (1 + np.cos(3 * phi - d3 * no_pi / 180))
                                + K4 * (1 + np.cos(4 * phi - d4 * no_pi / 180))
                                + K5 * (1 + np.cos(5 * phi - d5 * no_pi / 180))
                            )

                        input_to_periodic_absolute_difference = np.absolute(
                            input_dihedral_calc - periodic_dihedral_calc
                        )

                        input_dihedral_to_periodic_abs_diff.append(
                            input_to_periodic_absolute_difference
                        )
                        list_if_large_error_dihedral_iteration = []
                        list_abs_max_dihedral_iteration = []

                        if max(input_dihedral_to_periodic_abs_diff) > 10 ** (
                            -10
                        ):
                            list_if_large_error_dihedral_iteration.append(1)
                            list_abs_max_dihedral_iteration.append(
                                max(input_dihedral_to_periodic_abs_diff)
                            )

                            list_if_large_error_dihedral_overall.append(1)
                            list_if_largest_error_abs_values_for_dihedral_overall.append(
                                max(input_dihedral_to_periodic_abs_diff)
                            )

                            list_dihedral_overall_error.append(
                                f"{res_x}_{dihedral_members_iter[0]}, "
                                f"{res_x}_{dihedral_members_iter[1]}, "
                                f"{res_x}_{dihedral_members_iter[2]}, "
                                f"{res_x}_{dihedral_members_iter[3]}, "
                            )

                        else:
                            list_if_large_error_dihedral_iteration.append(0)
                            list_if_abs_max_values_for_dihedral_overall.append(
                                max(input_dihedral_to_periodic_abs_diff)
                            )
                            list_dihedral_atoms_all_dihedral_overall.append(
                                f"{res_x}_{dihedral_members_iter[0]}, "
                                f"{res_x}_{dihedral_members_iter[1]}, "
                                f"{res_x}_{dihedral_members_iter[2]}, "
                                f"{res_x}_{dihedral_members_iter[3]}, "
                            )

                    if self.utilized_NB_expression == "LJ":
                        # K0_output_energy_iter in LJ CHARMM format is only a harmonic dihedral,
                        # which is not included yet because GOMC currently only treats K0 as a constant,
                        # not a harmonic dihedral.
                        K0_output_energy_iter = np.round(
                            0,
                            decimals=dihedral_k_Kelvin_round_decimals,
                        )

                        K1_output_energy_iter = np.round(
                            K1, decimals=dihedral_k_kcal_per_mol_round_decimals
                        )
                        K2_output_energy_iter = np.round(
                            K2, decimals=dihedral_k_kcal_per_mol_round_decimals
                        )
                        K3_output_energy_iter = np.round(
                            K3, decimals=dihedral_k_kcal_per_mol_round_decimals
                        )
                        K4_output_energy_iter = np.round(
                            K4, decimals=dihedral_k_kcal_per_mol_round_decimals
                        )
                        K5_output_energy_iter = np.round(
                            K5, decimals=dihedral_k_kcal_per_mol_round_decimals
                        )

                    elif self.utilized_NB_expression in ["Mie", "Exp6"]:
                        K0_output_energy_iter = np.round(
                            u.unyt_quantity(K0, "kcal/mol").to_value(
                                "K", equivalence="thermal"
                            ),
                            decimals=dihedral_k_Kelvin_round_decimals,
                        )
                        K1_output_energy_iter = np.round(
                            u.unyt_quantity(K1, "kcal/mol").to_value(
                                "K", equivalence="thermal"
                            ),
                            decimals=dihedral_k_Kelvin_round_decimals,
                        )
                        K2_output_energy_iter = np.round(
                            u.unyt_quantity(K2, "kcal/mol").to_value(
                                "K", equivalence="thermal"
                            ),
                            decimals=dihedral_k_Kelvin_round_decimals,
                        )
                        K3_output_energy_iter = np.round(
                            u.unyt_quantity(K3, "kcal/mol").to_value(
                                "K", equivalence="thermal"
                            ),
                            decimals=dihedral_k_Kelvin_round_decimals,
                        )
                        K4_output_energy_iter = np.round(
                            u.unyt_quantity(K4, "kcal/mol").to_value(
                                "K", equivalence="thermal"
                            ),
                            decimals=dihedral_k_Kelvin_round_decimals,
                        )
                        K5_output_energy_iter = np.round(
                            u.unyt_quantity(K5, "kcal/mol").to_value(
                                "K", equivalence="thermal"
                            ),
                            decimals=dihedral_k_Kelvin_round_decimals,
                        )

                    elif self.utilized_NB_expression == "TABULATED":
                        # For TABULATED potentials, use the same convention as Mie/Exp6
                        # Convert K values from kcal/mol to Kelvin
                        K0_output_energy_iter = np.round(
                            u.unyt_quantity(K0, "kcal/mol").to_value(
                                "K", equivalence="thermal"
                            ),
                            decimals=dihedral_k_Kelvin_round_decimals,
                        )
                        K1_output_energy_iter = np.round(
                            u.unyt_quantity(K1, "kcal/mol").to_value(
                                "K", equivalence="thermal"
                            ),
                            decimals=dihedral_k_Kelvin_round_decimals,
                        )
                        K2_output_energy_iter = np.round(
                            u.unyt_quantity(K2, "kcal/mol").to_value(
                                "K", equivalence="thermal"
                            ),
                            decimals=dihedral_k_Kelvin_round_decimals,
                        )
                        K3_output_energy_iter = np.round(
                            u.unyt_quantity(K3, "kcal/mol").to_value(
                                "K", equivalence="thermal"
                            ),
                            decimals=dihedral_k_Kelvin_round_decimals,
                        )
                        K4_output_energy_iter = np.round(
                            u.unyt_quantity(K4, "kcal/mol").to_value(
                                "K", equivalence="thermal"
                            ),
                            decimals=dihedral_k_Kelvin_round_decimals,
                        )
                        K5_output_energy_iter = np.round(
                            u.unyt_quantity(K5, "kcal/mol").to_value(
                                "K", equivalence="thermal"
                            ),
                            decimals=dihedral_k_Kelvin_round_decimals,
                        )

                    # **************************************
                    # check the error between the convertions of RB_tortions to Periodic DIHEDRALS (end)
                    # **************************************

                    # get the dihedral atoms
                    dihedral_atom_0 = self.mosdef_atom_name_to_atom_type_dict[
                        f"{res_x}_{dihedral_members_iter[0]}"
                    ]
                    dihedral_atom_1 = self.mosdef_atom_name_to_atom_type_dict[
                        f"{res_x}_{dihedral_members_iter[1]}"
                    ]
                    dihedral_atom_2 = self.mosdef_atom_name_to_atom_type_dict[
                        f"{res_x}_{dihedral_members_iter[2]}"
                    ]
                    dihedral_atom_3 = self.mosdef_atom_name_to_atom_type_dict[
                        f"{res_x}_{dihedral_members_iter[3]}"
                    ]

                    # get the energies in the list
                    dihedral_values_list.append(
                        [
                            [
                                dihedral_atom_0,
                                dihedral_atom_1,
                                dihedral_atom_2,
                                dihedral_atom_3,
                                str(K0_output_energy_iter),
                                str(int(n0)),
                                str(
                                    np.round(
                                        d0,
                                        decimals=dihedral_phase_degree_round_decimals,
                                    )
                                ),
                                f"{res_x}_{dihedral_members_iter[0]}",
                                f"{res_x}_{dihedral_members_iter[1]}",
                                f"{res_x}_{dihedral_members_iter[2]}",
                                f"{res_x}_{dihedral_members_iter[3]}",
                            ],
                            [
                                dihedral_atom_0,
                                dihedral_atom_1,
                                dihedral_atom_2,
                                dihedral_atom_3,
                                str(K1_output_energy_iter),
                                str(int(n1)),
                                str(
                                    np.round(
                                        d1,
                                        decimals=dihedral_phase_degree_round_decimals,
                                    )
                                ),
                                f"{res_x}_{dihedral_members_iter[0]}",
                                f"{res_x}_{dihedral_members_iter[1]}",
                                f"{res_x}_{dihedral_members_iter[2]}",
                                f"{res_x}_{dihedral_members_iter[3]}",
                            ],
                            [
                                dihedral_atom_0,
                                dihedral_atom_1,
                                dihedral_atom_2,
                                dihedral_atom_3,
                                str(K2_output_energy_iter),
                                str(int(n2)),
                                str(
                                    np.round(
                                        d2,
                                        decimals=dihedral_phase_degree_round_decimals,
                                    )
                                ),
                                f"{res_x}_{dihedral_members_iter[0]}",
                                f"{res_x}_{dihedral_members_iter[1]}",
                                f"{res_x}_{dihedral_members_iter[2]}",
                                f"{res_x}_{dihedral_members_iter[3]}",
                            ],
                            [
                                dihedral_atom_0,
                                dihedral_atom_1,
                                dihedral_atom_2,
                                dihedral_atom_3,
                                str(K3_output_energy_iter),
                                str(int(n3)),
                                str(
                                    np.round(
                                        d3,
                                        decimals=dihedral_phase_degree_round_decimals,
                                    )
                                ),
                                f"{res_x}_{dihedral_members_iter[0]}",
                                f"{res_x}_{dihedral_members_iter[1]}",
                                f"{res_x}_{dihedral_members_iter[2]}",
                                f"{res_x}_{dihedral_members_iter[3]}",
                            ],
                            [
                                dihedral_atom_0,
                                dihedral_atom_1,
                                dihedral_atom_2,
                                dihedral_atom_3,
                                str(K4_output_energy_iter),
                                str(int(n4)),
                                str(
                                    np.round(
                                        d4,
                                        decimals=dihedral_phase_degree_round_decimals,
                                    )
                                ),
                                f"{res_x}_{dihedral_members_iter[0]}",
                                f"{res_x}_{dihedral_members_iter[1]}",
                                f"{res_x}_{dihedral_members_iter[2]}",
                                f"{res_x}_{dihedral_members_iter[3]}",
                            ],
                            [
                                dihedral_atom_0,
                                dihedral_atom_1,
                                dihedral_atom_2,
                                dihedral_atom_3,
                                str(K5_output_energy_iter),
                                str(int(n5)),
                                str(
                                    np.round(
                                        d5,
                                        decimals=dihedral_phase_degree_round_decimals,
                                    )
                                ),
                                f"{res_x}_{dihedral_members_iter[0]}",
                                f"{res_x}_{dihedral_members_iter[1]}",
                                f"{res_x}_{dihedral_members_iter[2]}",
                                f"{res_x}_{dihedral_members_iter[3]}",
                            ],
                        ]
                    )

                    # check for duplicates, for duplicate class or atom type
                    dihedral_same_count = 0  # Start at 0 (1 always found) count numbr of values that are the same
                    for dihedral_check_i in range(0, len(dihedral_values_list)):
                        # check if atomclass or atomtype is the same (check regular and reverse angle order)
                        if (
                            dihedral_values_list[-1][0][0]
                            == dihedral_values_list[dihedral_check_i][0][0]
                            and dihedral_values_list[-1][0][1]
                            == dihedral_values_list[dihedral_check_i][0][1]
                            and dihedral_values_list[-1][0][2]
                            == dihedral_values_list[dihedral_check_i][0][2]
                            and dihedral_values_list[-1][0][3]
                            == dihedral_values_list[dihedral_check_i][0][3]
                        ) or (
                            dihedral_values_list[-1][0][0]
                            == dihedral_values_list[dihedral_check_i][0][3]
                            and dihedral_values_list[-1][0][1]
                            == dihedral_values_list[dihedral_check_i][0][2]
                            and dihedral_values_list[-1][0][2]
                            == dihedral_values_list[dihedral_check_i][0][1]
                            and dihedral_values_list[-1][0][3]
                            == dihedral_values_list[dihedral_check_i][0][0]
                        ):
                            dihedral_same_count += 1

                            # check if values are the same since atomclass or atomtype is the same
                            max_k_value_used_int = len(dihedral_values_list[-1])
                            for k_v_i in range(0, max_k_value_used_int):
                                if (
                                    dihedral_values_list[-1][k_v_i][4]
                                    != dihedral_values_list[dihedral_check_i][
                                        k_v_i
                                    ][4]
                                    or dihedral_values_list[-1][k_v_i][5]
                                    != dihedral_values_list[dihedral_check_i][
                                        k_v_i
                                    ][5]
                                    or dihedral_values_list[-1][k_v_i][6]
                                    != dihedral_values_list[dihedral_check_i][
                                        k_v_i
                                    ][6]
                                ):
                                    raise ValueError(
                                        f"ERROR: The same atomclass or atomtype in the "
                                        f"force field are have different {'dihedral'} values.\n"
                                        f"{dihedral_values_list[-1][k_v_i]} != {dihedral_values_list[dihedral_check_i][k_v_i]}."
                                    )

                    dihedral_format = (
                        "{:10s} {:10s} {:10s} {:10s} {:15s} {:10s} {:15s} "
                        "! {:20s} {:20s} {:20s} {:20s}\n"
                    )
                    # Only print for 1 time so dihedral_count=1 (starts at 0)
                    if dihedral_same_count == 1:

                        # write charmm dihedral K0 (zero order dihedral --- a constant) if Mie or Exp6,
                        # but not written for periodic as the K0 constant is defined as a
                        # harmonic potential function in periodic.
                        if self.utilized_NB_expression in ["Mie", "Exp6", "TABULATED"]:
                            # write charmm dihedral K0 (harmonic dihedral)
                            if K0_output_energy_iter != 0:
                                data.write(
                                    dihedral_format.format(
                                        dihedral_values_list[-1][0][0],
                                        dihedral_values_list[-1][0][1],
                                        dihedral_values_list[-1][0][2],
                                        dihedral_values_list[-1][0][3],
                                        dihedral_values_list[-1][0][4],
                                        dihedral_values_list[-1][0][5],
                                        dihedral_values_list[-1][0][6],
                                        dihedral_values_list[-1][0][7],
                                        dihedral_values_list[-1][0][8],
                                        dihedral_values_list[-1][0][9],
                                        dihedral_values_list[-1][0][10],
                                    )
                                )

                        # write charmm dihedral K1 (first order periodic dihedral)
                        if (
                            K1_output_energy_iter == 0
                            and K2_output_energy_iter == 0
                            and K3_output_energy_iter == 0
                            and K4_output_energy_iter == 0
                            and K5_output_energy_iter == 0
                        ) or (K1_output_energy_iter != 0):
                            data.write(
                                dihedral_format.format(
                                    dihedral_values_list[-1][1][0],
                                    dihedral_values_list[-1][1][1],
                                    dihedral_values_list[-1][1][2],
                                    dihedral_values_list[-1][1][3],
                                    dihedral_values_list[-1][1][4],
                                    dihedral_values_list[-1][1][5],
                                    dihedral_values_list[-1][1][6],
                                    dihedral_values_list[-1][1][7],
                                    dihedral_values_list[-1][1][8],
                                    dihedral_values_list[-1][1][9],
                                    dihedral_values_list[-1][1][10],
                                )
                            )

                        # write charmm dihedral K2 (second order periodic dihedral)
                        if K2_output_energy_iter != 0:
                            data.write(
                                dihedral_format.format(
                                    dihedral_values_list[-1][2][0],
                                    dihedral_values_list[-1][2][1],
                                    dihedral_values_list[-1][2][2],
                                    dihedral_values_list[-1][2][3],
                                    dihedral_values_list[-1][2][4],
                                    dihedral_values_list[-1][2][5],
                                    dihedral_values_list[-1][2][6],
                                    dihedral_values_list[-1][2][7],
                                    dihedral_values_list[-1][2][8],
                                    dihedral_values_list[-1][2][9],
                                    dihedral_values_list[-1][2][10],
                                )
                            )

                        # write charmm dihedral K3 (third order periodic dihedral)
                        if K3_output_energy_iter != 0:
                            data.write(
                                dihedral_format.format(
                                    dihedral_values_list[-1][3][0],
                                    dihedral_values_list[-1][3][1],
                                    dihedral_values_list[-1][3][2],
                                    dihedral_values_list[-1][3][3],
                                    dihedral_values_list[-1][3][4],
                                    dihedral_values_list[-1][3][5],
                                    dihedral_values_list[-1][3][6],
                                    dihedral_values_list[-1][3][7],
                                    dihedral_values_list[-1][3][8],
                                    dihedral_values_list[-1][3][9],
                                    dihedral_values_list[-1][3][10],
                                )
                            )

                        # write charmm dihedral K4 (fourth order periodic dihedral)
                        if K4_output_energy_iter != 0:
                            data.write(
                                dihedral_format.format(
                                    dihedral_values_list[-1][4][0],
                                    dihedral_values_list[-1][4][1],
                                    dihedral_values_list[-1][4][2],
                                    dihedral_values_list[-1][4][3],
                                    dihedral_values_list[-1][4][4],
                                    dihedral_values_list[-1][4][5],
                                    dihedral_values_list[-1][4][6],
                                    dihedral_values_list[-1][4][7],
                                    dihedral_values_list[-1][4][8],
                                    dihedral_values_list[-1][4][9],
                                    dihedral_values_list[-1][4][10],
                                )
                            )

                        # write charmm dihedral K5 (fifth order periodic dihedral)
                        if K5_output_energy_iter != 0:
                            data.write(
                                dihedral_format.format(
                                    dihedral_values_list[-1][5][0],
                                    dihedral_values_list[-1][5][1],
                                    dihedral_values_list[-1][5][2],
                                    dihedral_values_list[-1][5][3],
                                    dihedral_values_list[-1][5][4],
                                    dihedral_values_list[-1][5][5],
                                    dihedral_values_list[-1][5][6],
                                    dihedral_values_list[-1][5][7],
                                    dihedral_values_list[-1][5][8],
                                    dihedral_values_list[-1][5][9],
                                    dihedral_values_list[-1][5][10],
                                )
                            )

                if len(self.topology_selection.dihedral_types):
                    if sum(list_if_large_error_dihedral_overall) > 0:
                        list_max_error_abs_dihedral_overall = max(
                            list_if_largest_error_abs_values_for_dihedral_overall
                        )
                        info_if_dihedral_error_too_large = (
                            f"! WARNING: The input dihedral type(s) to "
                            f"periodic dihedral conversion error"
                            f" is to large [error > 10^(-10)] \n"
                            f"! WARNING: Maximum( "
                            f"|(input dihedral calc)-(periodic dihedral calc)| ) =  "
                            f"{list_max_error_abs_dihedral_overall}\n"
                        )
                        warn(info_if_dihedral_error_too_large)
                        data.write(info_if_dihedral_error_too_large)
                        print(info_if_dihedral_error_too_large)
                    else:
                        list_if_abs_max_values_for_dihedral_overall_max = max(
                            list_if_abs_max_values_for_dihedral_overall
                        )
                        info_if_dihedral_error_ok = (
                            f"! The input dihedral to periodic dihedral conversion error is OK "
                            f"[error <= 10^(-10)]\n"
                            f"! Maximum( |(input dihedral calc)-(periodic dihedral calc)| ) =  "
                            f"{list_if_abs_max_values_for_dihedral_overall_max}\n"
                        )
                        data.write(info_if_dihedral_error_ok)
                        print(info_if_dihedral_error_ok)

                # Improper coefficients
                if len(self.topology_selection.improper_types):
                    if self.utilized_NB_expression in ["Mie", "Exp6"]:
                        print_error = f"ERROR: Currently, the Mie and Exp6 potentials do not support impropers."
                        raise ValueError(print_error)

                    if self.utilized_NB_expression in ["LJ"]:
                        data.write("\nIMPROPER * CHARMM\n")
                    elif self.utilized_NB_expression in ["Mie"]:
                        data.write("\nIMPROPER * Mie\n")
                    elif self.utilized_NB_expression in ["Exp6"]:
                        data.write("\nIMPROPER * Exp6\n")
                    data.write("! \n")
                    data.write("! V(dihedral) = Kw(1 + cos(n(w) - w0))\n")
                    data.write(
                        "! NOTE: For the CHARMM FF n=0 is a harmonic improper. "
                        "CHARMM FF where n=0 -->  V(improper) = Kw(w - w0)**2) \n"
                    )
                    data.write("! \n")
                    data.write("! Kw: kcal/mol (LJ) and K (Mie and Exp6)\n")
                    data.write("! n: multiplicity\n")
                    data.write("! delta: degrees\n")
                    data.write("! \n")
                    data.write("! Kw (kcal/mol) = Kw_K (K) * Boltz. const.\n")
                    data.write("! Boltzmann = 0.0019872041 kcal / (mol * K)\n")
                    data.write("! \n")
                    data.write(
                        "! {:8s} {:10s} {:10s} {:10s} {:15s} {:10s} {:15s} ! {:20s} {:20s} {:20s} {:20s}\n".format(
                            "type_1",
                            "type_2",
                            "type_3",
                            "type_4",
                            "Kw",
                            "n",
                            "w0",
                            "extended_type_1",
                            "extended_type_2",
                            "extended_type_3",
                            "extended_type_4",
                        )
                    )

                improper_k_kcal_per_mol_round_decimals = 6
                improper_k_Kelvin_round_decimals = 4
                improper_phase_degree_round_decimals = 6

                impr_periodic_val_list = []
                for improper_type_x in self.combinded_residue_improper_types:
                    res_x = improper_type_x.__dict__["tags_"]["resname"]
                    improper_members_iter = improper_type_x.member_types

                    improper_type_str_iter = None
                    improper_eqn_scalar_iter = None
                    if (
                        improper_type_str_iter is None
                        and improper_eqn_scalar_iter is None
                    ):
                        # Check if HarmonicImproperPotential
                        HarmonicImproperPotential_form_with_scalar = (
                            "k * (phi - phi_eq)**2"
                        )

                        [
                            improper_type_str_iter,
                            improper_eqn_scalar_iter,
                        ] = evaluate_harmonic_improper_format_with_scaler(
                            improper_type_x.expression,
                            HarmonicImproperPotential_form_with_scalar,
                        )

                    if (
                        improper_type_str_iter is None
                        and improper_eqn_scalar_iter is None
                    ):
                        # Check if PeriodicImproperPotential
                        PeriodicImproperPotential_form_with_scalar = (
                            "k * (1 + cos(n * phi - phi_eq))"
                        )

                        [
                            improper_type_str_iter,
                            improper_eqn_scalar_iter,
                        ] = evaluate_periodic_improper_format_with_scaler(
                            improper_type_x.expression,
                            PeriodicImproperPotential_form_with_scalar,
                        )

                    elif (
                        improper_type_str_iter is None
                        and improper_eqn_scalar_iter is None
                    ):
                        raise TypeError(
                            f"ERROR: The {res_x} residue and associated force field "
                            f"has at least one unsupported dihdedral. "
                            f"The only supported impropers is the "
                            f" {'PeriodicImproperPotential'}."
                        )

                    # determine the allowable variables per the improper types
                    improper_type_x_parameters_keys_list = list(
                        improper_type_x.parameters.keys()
                    )
                    if improper_type_str_iter == "HarmonicImproperPotential":
                        raise TypeError(
                            f"ERROR: The {res_x} residue has a "
                            f"{'HarmonicImproperPotential'} torsion potential, which "
                            f"is not currently supported in this writer."
                        )
                        """

                        # KEEP THIS FOR LATER
                        # this section of code is not currently will but will work when GMSO
                        # is able to support both harmonic and periodic improper together

                        improper_k0_phi_eq0_n0_variables_set_iter = {'k', 'phi_eq'} # no n variables here
                        #improper_all_k_n_phi_eq_variables_set_iter = set()
                        improper_all_k_n_phi_eq_variables_set_iter = sorted(
                                set.union(
                                    improper_k0_phi_eq0_n0_variables_set_iter,
                                )
                        )
                        """

                    elif improper_type_str_iter == "PeriodicImproperPotential":
                        improper_kx_ph_eqx_x_variables_set_iter = {
                            "k",
                            "n",
                            "phi_eq",
                        }
                        improper_all_k_n_phi_eq_variables_set_iter = set()
                        improper_all_k_n_phi_eq_variables_set_iter = sorted(
                            set.union(
                                improper_kx_ph_eqx_x_variables_set_iter,
                            )
                        )

                    # see of the improper variables are of the correct type
                    for (
                        improper_type_x_parameters_key_i
                    ) in improper_type_x_parameters_keys_list:
                        if (
                            improper_type_x_parameters_key_i
                            not in improper_all_k_n_phi_eq_variables_set_iter
                        ):
                            print_error = (
                                f"ERROR: In molecule or residue {res_x}, "
                                f"the {improper_type_str_iter} "
                                f"types variables for the improper "
                                f"{improper_members_iter[0]}-"
                                f"{improper_members_iter[1]}-"
                                f"{improper_members_iter[2]}-"
                                f"{improper_members_iter[3]} "
                                f"are {improper_type_x_parameters_keys_list}, "
                                f"not the required types "
                                f"{improper_all_k_n_phi_eq_variables_set_iter}."
                            )
                            raise ValueError(print_error)

                    # get the improper values in k = kcal/mol, n= integer and phi_eq = degrees
                    if improper_type_str_iter == "HarmonicImproperPotential":
                        """
                        # KEEP THIS FOR LATER
                        # this section of code is not currently will but will work when GMSO
                        # is able to support both harmonic and periodic improper together

                        for improper_iter_i in improper_k0_phi_eq0_n0_variables_set_iter:
                            if improper_iter_i not in improper_type_x_parameters_keys_list:
                                print_error = f"ERROR: In molecule or residue {res_x}, " \
                                              f"the {'HarmonicImproperPotential'} "\
                                              f"types variables for the improper "\
                                              f"{improper_members_iter[0]}-"\
                                              f"{improper_members_iter[1]}-"\
                                              f"{improper_members_iter[2]}-"\
                                              f"{improper_members_iter[3]} "\
                                              f"are {improper_type_x_parameters_keys_list}, " \
                                              f"not the required types" \
                                              f"{improper_k0_phi_eq0_n0_variables_set_iter}."
                                raise ValueError(print_error)

                        K0_improper = dihedral_eqn_scalar_iter * improper_type_x.parameters[
                            'k'].to_value('kcal/mol', equivalence='thermal')
                        n0_improper = 0
                        phi_eq0_improper = improper_type_x.parameters['phi_eq'].to_value('degree')
                        """

                    elif improper_type_str_iter == "PeriodicImproperPotential":
                        # get the number of periodic impropers
                        # for improper_iter_i in improper_kx_ph_eqx_x_variables_set_iter:

                        # get k values
                        if isinstance(
                            improper_type_x.parameters["k"],
                            u.array.unyt_quantity,
                        ):
                            periodic_impropers_k_values = u.unyt_array(
                                [
                                    improper_type_x.parameters["k"].to_value(
                                        "kcal/mol", equivalence="thermal"
                                    )
                                ],
                                "kcal/mol",
                            )
                        else:
                            periodic_impropers_k_values = (
                                improper_type_x.parameters["k"]
                            )

                        periodic_impropers_k_values_len = len(
                            periodic_impropers_k_values
                        )

                        # get n values
                        if isinstance(
                            improper_type_x.parameters["n"],
                            u.array.unyt_quantity,
                        ):
                            periodic_impropers_n_values = u.unyt_array(
                                [
                                    improper_type_x.parameters["n"].to_value(
                                        "(dimensionless)"
                                    )
                                ],
                                "(dimensionless)",
                            )
                        else:
                            periodic_impropers_n_values = (
                                improper_type_x.parameters["n"]
                            )

                        periodic_improper_n_values_len = len(
                            periodic_impropers_n_values
                        )

                        # get phi_eq values
                        if isinstance(
                            improper_type_x.parameters["phi_eq"],
                            u.array.unyt_quantity,
                        ):
                            periodic_impropers_phi_eq_values = u.unyt_array(
                                [
                                    improper_type_x.parameters[
                                        "phi_eq"
                                    ].to_value("degree")
                                ],
                                "degree",
                            )
                        else:
                            periodic_impropers_phi_eq_values = (
                                improper_type_x.parameters["phi_eq"]
                            )

                        periodic_improper_phi_eq_values_len = len(
                            periodic_impropers_phi_eq_values
                        )

                        if (
                            periodic_impropers_k_values_len
                            == periodic_improper_n_values_len
                            == periodic_improper_phi_eq_values_len
                        ):
                            number_of_periodic_impropers_int = (
                                periodic_impropers_k_values_len
                            )

                        else:
                            print_error = (
                                f"ERROR: The periodic improper values "
                                f"{improper_kx_ph_eqx_x_variables_set_iter} "
                                f"do not have all the values for each set."
                            )
                            raise ValueError(print_error)

                        # scale the Kx, nx, and phi_eqx to the kcal/mol values
                        Kx_improper = (
                            improper_eqn_scalar_iter
                            * periodic_impropers_k_values.to_value(
                                "kcal/mol", equivalence="thermal"
                            )
                        )
                        nx_improper = periodic_impropers_n_values.to_value(
                            "(dimensionless)"
                        )
                        phi_eqx_improper = (
                            periodic_impropers_phi_eq_values.to_value("degree")
                        )

                        # test if all n values unique and Check if the any of the impropers is zero
                        unique_n_values_improper_iter = []
                        for periodic_j in range(
                            0, number_of_periodic_impropers_int
                        ):
                            nx_improper_iter = nx_improper[periodic_j]

                            # for nx_improper_j in nx_dihedral_iter:
                            if (
                                nx_improper_iter
                                in unique_n_values_improper_iter
                            ):
                                print_error = (
                                    f"ERROR: mulitple PeriodicImproperPotential "
                                    f"n values of {nx_improper_iter} "
                                    f"were found for the same improper. Only 1 of each "
                                    f"n values are allowed per PeriodicImproperPotential."
                                )
                                raise ValueError(print_error)
                            else:
                                unique_n_values_improper_iter.append(
                                    nx_improper_iter
                                )

                            nx_improper_iter = nx_improper[periodic_j]
                            if nx_improper_iter not in [1, 2, 3, 4, 5]:
                                print_value_error = (
                                    "ERROR: The "
                                    f"{improper_members_iter[0]}-"
                                    f"{improper_members_iter[1]}-"
                                    f"{improper_members_iter[2]}-"
                                    f"{improper_members_iter[3]} "
                                    f"improper "
                                    f"{'PeriodicImproperPotential'} "
                                    "does not have n values from 1 to 5."
                                )
                                raise ValueError(print_value_error)

                    # get the improper atom values
                    improper_atom_0 = self.mosdef_atom_name_to_atom_type_dict[
                        f"{res_x}_{improper_members_iter[0]}"
                    ]
                    improper_atom_1 = self.mosdef_atom_name_to_atom_type_dict[
                        f"{res_x}_{improper_members_iter[1]}"
                    ]
                    improper_atom_2 = self.mosdef_atom_name_to_atom_type_dict[
                        f"{res_x}_{improper_members_iter[2]}"
                    ]
                    improper_atom_3 = self.mosdef_atom_name_to_atom_type_dict[
                        f"{res_x}_{improper_members_iter[3]}"
                    ]

                    # The CHARMM as the K0 improper constant (harmonic) is defined as a
                    # get the improper values in k = kcal/mol, n= integer and phi_eq = degrees
                    if improper_type_str_iter == "HarmonicImproperPotential":
                        """
                        # KEEP THIS FOR LATER
                        # this section of code is not currently will but will work when GMSO
                        # is able to support both harmonic and periodic improper together

                        # not sure need: if K0_improper_output_energy_iter != 0:
                        if self.utilized_NB_expression == 'LJ':
                            K0_improper_output_energy_iter = np.round(
                                K0_improper, decimals=improper_k_kcal_per_mol_round_decimals)

                        elif self.utilized_NB_expression in ['Mie', 'Exp6']:
                            print_error = "ERROR: The {'HarmonicImproperPotential'} is not supported " \
                                          "for the {'Mie'} or {'Exp6'} potentials."
                            raise ValueError(print_error)

                        data.write(
                            improper_format.format(
                                improper_atom_0,
                                improper_atom_1,
                                improper_atom_2,
                                improper_atom_3,
                                str(K0_improper_output_energy_iter),
                                str(int(n0_improper)),
                                str(np.round(phi_eq0_improper, decimals=improper_phase_degree_round_decimals)),
                                f"{res_x}_{improper_members_iter[0]}",
                                f"{res_x}_{improper_members_iter[1]}",
                                f"{res_x}_{improper_members_iter[2]}",
                                f"{res_x}_{improper_members_iter[3]}",
                            )
                        )
                        """

                    elif improper_type_str_iter == "PeriodicImproperPotential":
                        # write charmm improper Kx (x order periodic improper)
                        for periodic_impropers_no_j in range(
                            0, number_of_periodic_impropers_int
                        ):
                            if self.utilized_NB_expression == "LJ":
                                Kx_improper_output_energy_iter = np.round(
                                    Kx_improper[periodic_impropers_no_j],
                                    decimals=improper_k_kcal_per_mol_round_decimals,
                                )

                            elif self.utilized_NB_expression in ["Mie", "Exp6"]:
                                Kx_improper_output_energy_iter = np.round(
                                    u.unyt_quantity(
                                        Kx_improper[periodic_impropers_no_j],
                                        "kcal/mol",
                                    ).to_value("K", equivalence="thermal"),
                                    decimals=improper_k_Kelvin_round_decimals,
                                )

                            nx_improper_output_iter = nx_improper[
                                periodic_impropers_no_j
                            ]
                            phi_eqx_improper_output_iter = phi_eqx_improper[
                                periodic_impropers_no_j
                            ]

                            impr_periodic_val_list.append(
                                [
                                    improper_atom_0,
                                    improper_atom_1,
                                    improper_atom_2,
                                    improper_atom_3,
                                    str(Kx_improper_output_energy_iter),
                                    str(int(nx_improper_output_iter)),
                                    str(
                                        np.round(
                                            phi_eqx_improper_output_iter,
                                            decimals=improper_phase_degree_round_decimals,
                                        )
                                    ),
                                    f"{res_x}_{improper_members_iter[0]}",
                                    f"{res_x}_{improper_members_iter[1]}",
                                    f"{res_x}_{improper_members_iter[2]}",
                                    f"{res_x}_{improper_members_iter[3]}",
                                ]
                            )

                            # check for duplicates, for duplicate class or atom type
                            improper_same_count = 0  # Start at 0 (1 always found) count numbr of values that are the same
                            for improper_check_i in range(
                                0, len(impr_periodic_val_list)
                            ):

                                # check if atomclass or atomtype is the same (check regular and reverse improper order)
                                if (
                                    impr_periodic_val_list[-1][0]
                                    == impr_periodic_val_list[improper_check_i][
                                        0
                                    ]
                                    and impr_periodic_val_list[-1][3]
                                    == impr_periodic_val_list[improper_check_i][
                                        3
                                    ]
                                    and impr_periodic_val_list[-1][5]
                                    == impr_periodic_val_list[improper_check_i][
                                        5
                                    ]
                                ) and (
                                    (
                                        impr_periodic_val_list[-1][1]
                                        == impr_periodic_val_list[
                                            improper_check_i
                                        ][1]
                                        and impr_periodic_val_list[-1][2]
                                        == impr_periodic_val_list[
                                            improper_check_i
                                        ][2]
                                    )
                                    or (
                                        impr_periodic_val_list[-1][1]
                                        == impr_periodic_val_list[
                                            improper_check_i
                                        ][2]
                                        and impr_periodic_val_list[-1][2]
                                        == impr_periodic_val_list[
                                            improper_check_i
                                        ][1]
                                    )
                                ):
                                    improper_same_count += 1

                                    # check if values are the same since atomclass or atomtype is the same
                                    if (
                                        impr_periodic_val_list[-1][4]
                                        != impr_periodic_val_list[
                                            improper_check_i
                                        ][4]
                                        or impr_periodic_val_list[-1][6]
                                        != impr_periodic_val_list[
                                            improper_check_i
                                        ][6]
                                    ):
                                        raise ValueError(
                                            f"ERROR: The same atomclass or atomtype in the "
                                            f"force field are have different {'improper'} values.\n"
                                            f"{impr_periodic_val_list[-1]} != {impr_periodic_val_list[improper_check_i]} "
                                        )

                            # Only print for 1 time so improper_same_count=1 (starts at 0)
                            if improper_same_count == 1:
                                improper_format = (
                                    "{:10s} {:10s} {:10s} {:10s} {:15s} {:10s} {:15s} "
                                    "! {:20s} {:20s} {:20s} {:20s}\n"
                                )
                                data.write(
                                    improper_format.format(
                                        impr_periodic_val_list[-1][0],
                                        impr_periodic_val_list[-1][1],
                                        impr_periodic_val_list[-1][2],
                                        impr_periodic_val_list[-1][3],
                                        impr_periodic_val_list[-1][4],
                                        impr_periodic_val_list[-1][5],
                                        impr_periodic_val_list[-1][6],
                                        impr_periodic_val_list[-1][7],
                                        impr_periodic_val_list[-1][8],
                                        impr_periodic_val_list[-1][9],
                                        impr_periodic_val_list[-1][10],
                                    )
                                )

                # Pair coefficients
                print("NBFIX_Mixing not used.")
                print(
                    "self.utilized_NB_expression = "
                    + str(self.utilized_NB_expression)
                )

                epsilon_kcal_per_mol_round_decimals = 10
                epsilon_Kelvin_round_decimals = 6
                sigma_round_decimals = 10
                mie_n_or_exp6_alpha_round_decimals = 8

                if self.utilized_NB_expression == "LJ":
                    data.write("\n")
                    data.write("NONBONDED * LJ\n")
                    data.write("! \n")
                    data.write(
                        "! V(Lennard-Jones) = Epsilon,i,j[(Rmin,i,j/ri,j)**12 - 2(Rmin,i,j/ri,j)**6]\n"
                        "!                                    or\n"
                        "! V(Lennard-Jones) = 4 * Epsilon,i,j[(Sigma,i,j/ri,j)**12 - (Sigma,i,j/ri,j)**6]\n"
                    )
                    data.write("! \n")

                    data.write(
                        "! {:8s} {:15s} {:15s} {:15s} {:15s} {:15s} {:15s} ! {:20s} {:20s}\n"
                        "".format(
                            "type_1",
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

                elif self.utilized_NB_expression in ["Mie", "Exp6"]:
                    if self.utilized_NB_expression == "Mie":
                        data.write("\n")
                        data.write("NONBONDED_MIE * Mie\n")

                        data.write("! \n")
                        data.write(
                            "! V(Mie) = (n/(n-6)) * (n/6)**(6/(n-6)) * Epsilon * ((sig/r)**n - (sig/r)**6)\n"
                        )
                        data.write("! \n")

                        data.write(
                            "! {:8s} {:15s} {:15s} {:15s} {:15s} {:15s} {:15s} ! {:20s} {:20s}\n"
                            "".format(
                                "type_1",
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

                    elif self.utilized_NB_expression == "Exp6":
                        data.write("\n")
                        data.write("NONBONDED_MIE * Exp6\n")
                        data.write("! \n")
                        data.write(
                            "! V(Exp6) = epsilon*alpha/(alpha-6) * (6/alpha*exp(alpha*(1-r/Rmin)) - (Rmin/r)**6)\n"
                        )
                        data.write("! \n")

                        data.write(
                            "! {:8s} {:15s} {:15s} {:15s} {:15s} {:15s} {:15s} ! {:20s} {:20s}\n"
                            "".format(
                                "type_1",
                                "epsilon",
                                "sigma",
                                "alpha",
                                "epsilon,1-4",
                                "sigma,1-4",
                                "alpha,1-4",
                                "extended_type_1",
                                "extended_type_2",
                            )
                        )

                elif self.utilized_NB_expression == "TABULATED":
                    # For TABULATED potentials, write NONBONDED section header based on ParaTypeMie
                    data.write("\n")
                    if self.ParaTypeMie:
                        data.write("NONBONDED_MIE * TABULATED\n")
                    else:
                        data.write("NONBONDED * TABULATED\n")
                    data.write("! \n")
                    data.write("! Tabulated pair interactions\n")
                    data.write("! \n")

                else:
                    printed_output = (
                        f"ERROR: Currently this potential ({self.utilized_NB_expression}) "
                        f"is not supported in this MoSDeF GOMC parameter writer\n"
                    )
                    data.write(printed_output)
                    print_error_message = printed_output
                    raise ValueError(print_error_message)

                # write out the non-bonded portion
                # For TABULATED potentials, write atom types with zero parameters
                if self.utilized_NB_expression == "TABULATED":
                    # For TABULATED, write atom types with placeholder zero parameters
                    nb_val_list = []
                    for (
                        class_x,
                        epsilon_kcal_per_mol,
                    ) in self.epsilon_kcal_per_mol_atom_type_dict.items():
                        if float(self.nonbonded_1_4_dict[class_x]) == 0:
                            scalar_used_binary = 0
                        else:
                            scalar_used_binary = 1
                        
                        # For TABULATED with ParaTypeMie, use Mie format with zeros except n=12
                        if self.ParaTypeMie:
                            nb_val_list.append(
                                [
                                    str(
                                        self.mosdef_atom_name_to_atom_type_dict[
                                            class_x
                                        ]
                                    ),
                                    str(0.0),  # epsilon = 0
                                    str(0.0),  # sigma = 0
                                    str(12.0),  # n = 12 (MIE exponent)
                                    str(0.0),  # epsilon,1-4 = 0
                                    str(0.0),  # sigma,1-4 = 0
                                    str(12.0),  # n,1-4 = 12
                                    str(class_x),
                                    str(class_x),
                                ]
                            )
                        else:
                            nb_val_list.append(
                                [
                                    str(
                                        self.mosdef_atom_name_to_atom_type_dict[
                                            class_x
                                        ]
                                    ),
                                    str(0.0),  # ignored
                                    str(0.0),  # epsilon = 0
                                    str(0.0),  # Rmin/2 = 0
                                    str(0.0),  # ignored
                                    str(0.0),  # epsilon,1-4 = 0
                                    str(0.0),  # Rmin/2,1-4 = 0
                                    str(class_x),
                                    str(class_x),
                                ]
                            )

                elif self.utilized_NB_expression != "TABULATED":
                    nb_val_list = []
                    for (
                        class_x,
                        epsilon_kcal_per_mol,
                    ) in self.epsilon_kcal_per_mol_atom_type_dict.items():
                            # if the 1-4 non-bonded scalar is used.
                            # If 1-4 non-bonded scalar = 0, all 1-4 non-bonded values are set to zero (0).
                            # If 1-4 non-bonded scalar = 1, the epsilon 1-4 non-bonded interaction is scaled
                            # via another scalar.
                            if float(self.nonbonded_1_4_dict[class_x]) == 0:
                                scalar_used_binary = 0
                            else:
                                scalar_used_binary = 1

                            # if in "LJ" form put in kcal/mol
                            if self.utilized_NB_expression == "LJ":
                                nb_val_list.append(
                                    [
                                        str(
                                            self.mosdef_atom_name_to_atom_type_dict[
                                                class_x
                                            ]
                                        ),
                                        str(0.0),
                                        str(
                                            np.round(
                                                -epsilon_kcal_per_mol,
                                                decimals=epsilon_kcal_per_mol_round_decimals,
                                            )
                                        ),
                                        str(
                                            np.round(
                                                _LJ_sigma_to_r_min_div_2(
                                                    self.sigma_angstrom_atom_type_dict[
                                                        class_x
                                                    ]
                                                ),
                                                decimals=sigma_round_decimals,
                                            )
                                        ),
                                        str(0.0),
                                        str(
                                            -np.round(
                                                scalar_used_binary
                                                * float(
                                                    self.nonbonded_1_4_dict[class_x]
                                                )
                                                * epsilon_kcal_per_mol,
                                                decimals=epsilon_kcal_per_mol_round_decimals,
                                            )
                                        ),
                                        str(
                                            np.round(
                                                _LJ_sigma_to_r_min_div_2(
                                                    self.sigma_angstrom_atom_type_dict[
                                                        class_x
                                                    ]
                                                ),
                                                decimals=sigma_round_decimals,
                                            )
                                        ),
                                        str(class_x),
                                        str(class_x),
                                    ]
                                )

                            # if in "Mie" or "Exp6" form put in K -- energy units
                            elif self.utilized_NB_expression in ["Mie", "Exp6"]:
                                # check that nb-vdw = 0 or 1 only for "Mie or Exp6"
                                for (
                                    res_nb_14,
                                    scaler_nb_14,
                                ) in self.nonbonded_1_4_dict.items():
                                    if scaler_nb_14 is None:
                                        print_error = (
                                            f"ERROR: The {res_nb_14} residue is provided a value of "
                                            f"{scaler_nb_14}, and a value from 0 to 1 needs to be provided. "
                                            f"Please check the force file xml file."
                                        )
                                        raise ValueError(print_error)

                                # select Mie n values or Exp6 alpha values
                                if self.utilized_NB_expression == "Mie":
                                    mie_n_or_exp6_alpha_atom_type_value = (
                                        self.mie_n_atom_type_dict[class_x]
                                    )
                                elif self.utilized_NB_expression == "Exp6":
                                    mie_n_or_exp6_alpha_atom_type_value = (
                                        self.exp6_alpha_atom_type_dict[class_x]
                                    )

                                epsilon_Kelvin = u.unyt_quantity(
                                    epsilon_kcal_per_mol, "kcal/mol"
                                ).to_value("K", equivalence="thermal")

                                nb_val_list.append(
                                    [
                                        str(
                                            self.mosdef_atom_name_to_atom_type_dict[
                                                class_x
                                            ]
                                        ),
                                        str(
                                            np.round(
                                                epsilon_Kelvin,
                                                decimals=epsilon_Kelvin_round_decimals,
                                            )
                                        ),
                                        str(
                                            np.round(
                                                self.sigma_angstrom_atom_type_dict[
                                                    class_x
                                                ],
                                                decimals=sigma_round_decimals,
                                            )
                                        ),
                                        str(
                                            np.round(
                                                mie_n_or_exp6_alpha_atom_type_value,
                                                decimals=mie_n_or_exp6_alpha_round_decimals,
                                            )
                                        ),
                                        str(
                                            np.round(
                                                scalar_used_binary
                                                * float(
                                                    self.nonbonded_1_4_dict[class_x]
                                                )
                                                * (epsilon_Kelvin),
                                                decimals=epsilon_Kelvin_round_decimals,
                                            )
                                        ),
                                        str(
                                            np.round(
                                                self.sigma_angstrom_atom_type_dict[
                                                    class_x
                                                ],
                                                decimals=sigma_round_decimals,
                                            )
                                        ),
                                        str(
                                            np.round(
                                                mie_n_or_exp6_alpha_atom_type_value,
                                                decimals=mie_n_or_exp6_alpha_round_decimals,
                                            )
                                        ),
                                        str(class_x),
                                        str(class_x),
                                    ]
                                )

                            # check for duplicates, for duplicate class or atom type
                            nb_same_count = 0  # Start at 0 (1 always found) count numbr of values that are the same
                            for nb_check_i in range(0, len(nb_val_list)):

                                # check if atomclass or atomtype is the same
                                if nb_val_list[-1][0] == nb_val_list[nb_check_i][0]:
                                    nb_same_count += 1

                                    # check if values are the same since atomclass or atomtype is the same
                                    if (
                                        nb_val_list[-1][1] != nb_val_list[nb_check_i][1]
                                        or nb_val_list[-1][2]
                                        != nb_val_list[nb_check_i][2]
                                        or nb_val_list[-1][3]
                                        != nb_val_list[nb_check_i][3]
                                        or nb_val_list[-1][4]
                                        != nb_val_list[nb_check_i][4]
                                        or nb_val_list[-1][5]
                                        != nb_val_list[nb_check_i][5]
                                        or nb_val_list[-1][6]
                                        != nb_val_list[nb_check_i][6]
                                    ):
                                        raise ValueError(
                                            f"ERROR: The same atomclass or atomtype in the "
                                            f"force field are have different {'non-bonded'} values.\n"
                                            f"{nb_val_list[-1]} != {nb_val_list[nb_check_i]} "
                                        )

                            # Only print for 1 time so nb_same_count=1 (starts at 0)
                            if nb_same_count == 1:
                                nb_format = "{:10s} {:15s} {:15s} {:15s} {:15s} {:15s} {:15s} ! {:20s} {:20s}\n"
                                data.write(
                                    nb_format.format(
                                        nb_val_list[-1][0],
                                        nb_val_list[-1][1],
                                        nb_val_list[-1][2],
                                        nb_val_list[-1][3],
                                        nb_val_list[-1][4],
                                        nb_val_list[-1][5],
                                        nb_val_list[-1][6],
                                        nb_val_list[-1][7],
                                        nb_val_list[-1][8],
                                    )
                                )

                # Write TABULATED atom types (outside the conditional block)
                if self.utilized_NB_expression == "TABULATED":
                    for nb_val in nb_val_list:
                        nb_format = "{:10s} {:15s} {:15s} {:15s} {:15s} {:15s} {:15s} ! {:20s} {:20s}\n"
                        data.write(
                            nb_format.format(
                                nb_val[0],
                                nb_val[1],
                                nb_val[2],
                                nb_val[3],
                                nb_val[4],
                                nb_val[5],
                                nb_val[6],
                                nb_val[7],
                                nb_val[8],
                            )
                        )

                # writing NBTABLE entries (only for TABULATED potentials)
                if self.is_tabulated:
                    data.write("\n")
                    data.write("NBTABLE * table\n")
                    data.write("! \n")
                    data.write("! Tabulated pair interactions for TABULATED potentials\n")
                    data.write("! Format: atom_type_1 atom_type_2 label\n")
                    
                    # Get all unique atom types from the force field
                    atom_types = sorted(set(self.mosdef_atom_name_to_atom_type_dict.values()))
                    
                    # Write all pair combinations
                    for atom_type_1 in atom_types:
                        for atom_type_2 in atom_types:
                            if atom_type_1 <= atom_type_2:  # Avoid duplicate pairs
                                pair_label = f"{atom_type_1}{atom_type_2}"
                                data.write(f"{atom_type_1:10s} {atom_type_2:10s} {pair_label}\n")

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
        print("The psf writer (the write_psf function) is running")

        date_time = datetime.datetime.today()

        print(
            "write_psf: forcefield_selection = {}, residues = {}".format(
                self.forcefield_selection, self.residues
            )
        )

        print("******************************")
        print("")

        # calculate the atom name and  (instead of: atom.occupancy)

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

            if q == 0:
                residue_names_list_psf = self.residue_names_list_box_0
                residue_id_list_psf = self.residue_id_list_box_0
                res_no_chain_iter_corrected_list_psf = (
                    self.res_no_chain_iter_corrected_list_box_0
                )
                segment_id_list_psf = self.segment_id_list_box_0

            else:
                residue_names_list_psf = self.residue_names_list_box_1
                residue_id_list_psf = self.residue_id_list_box_1
                res_no_chain_iter_corrected_list_psf = (
                    self.res_no_chain_iter_corrected_list_box_1
                )
                segment_id_list_psf = self.segment_id_list_box_1

            output_write = open(output, "w")

            first_indent = "%8s"
            psf_formating = (
                "%8s %-4s %-4s %-4s %-4s %-6s %10.6f %13.4f" + 11 * " "
            )

            output_write.write("PSF ")
            output_write.write("\n\n")

            no_of_remarks = 3
            output_write.write(first_indent % no_of_remarks + " !NTITLE\n")
            output_write.write(
                f" REMARKS this file "
                f"{file_name_iteration} "
                f"- created by using MoSDeF-GOMC. \n"
            )
            output_write.write(
                f" REMARKS parameters from the "
                f"{self.forcefield_selection} "
                f"force field via MoSDef\n"
            )
            output_write.write(f" REMARKS created on {date_time}\n\n\n")

            # This converts the atom name in the GOMC psf and pdb files to unique atom names
            print(f"bead_to_atom_name_dict = {self.bead_to_atom_name_dict}")
            [
                unique_individual_atom_names_dict,
                individual_atom_names_list,
                missing_bead_to_atom_name,
            ] = unique_atom_naming(
                stuct_only_iteration,
                residue_id_list_psf,
                residue_names_list_psf,
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
            # psf_formating is conducted for the for CHARMM format (i.e., atom types are base 44, letters only)
            output_write.write(first_indent % no_atoms + " !NATOM\n")
            for i_atom, PSF_atom_iteration_1 in enumerate(
                stuct_iteration.sites
            ):
                charge_iter = (
                    PSF_atom_iteration_1.atom_type.__dict__["charge_"].to("C")
                    / u.elementary_charge
                )
                charge_iter = charge_iter.to_value("(dimensionless)")
                mass_iter = PSF_atom_iteration_1.atom_type.__dict__[
                    "mass_"
                ].to_value("amu")
                residue_name_iter = PSF_atom_iteration_1.__dict__[
                    "residue_name_"
                ]
                atom_class_name_iter = PSF_atom_iteration_1.atom_type.__dict__[
                    "atomclass_"
                ]
                atom_type_name_iter = PSF_atom_iteration_1.atom_type.__dict__[
                    "name_"
                ]

                atom_type_iter = self.mosdef_atom_name_to_atom_type_dict[
                    f"{residue_name_iter}_{atom_type_name_iter}"
                ]

                atom_lines_iteration = psf_formating % (
                    i_atom + 1,
                    segment_id_list_psf[i_atom],
                    res_no_chain_iter_corrected_list_psf[i_atom],
                    str(residue_names_list_psf[i_atom])[
                        : self.max_resname_char
                    ],
                    individual_atom_names_list[i_atom],
                    atom_type_iter,
                    charge_iter,
                    mass_iter,
                )

                output_write.write("%s\n" % atom_lines_iteration)

            output_write.write("\n")

            # BONDS: Calculate the bonding data
            output_write.write(first_indent % no_bonds + " !NBOND: bonds\n")
            for i_bond, bond_iteration in enumerate(stuct_iteration.bonds):
                output_write.write(
                    (first_indent * 2)
                    % (
                        stuct_iteration.get_index(
                            bond_iteration.connection_members[0]
                        )
                        + 1,
                        stuct_iteration.get_index(
                            bond_iteration.connection_members[1]
                        )
                        + 1,
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
                        stuct_iteration.get_index(
                            angle_iteration.connection_members[0]
                        )
                        + 1,
                        stuct_iteration.get_index(
                            angle_iteration.connection_members[1]
                        )
                        + 1,
                        stuct_iteration.get_index(
                            angle_iteration.connection_members[2]
                        )
                        + 1,
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
            for i_dihedral, dihedral_iter in enumerate(
                stuct_iteration.dihedrals
            ):
                output_write.write(
                    (first_indent * 4)
                    % (
                        stuct_iteration.get_index(
                            dihedral_iter.connection_members[0]
                        )
                        + 1,
                        stuct_iteration.get_index(
                            dihedral_iter.connection_members[1]
                        )
                        + 1,
                        stuct_iteration.get_index(
                            dihedral_iter.connection_members[2]
                        )
                        + 1,
                        stuct_iteration.get_index(
                            dihedral_iter.connection_members[3]
                        )
                        + 1,
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
            for i_improper, improper_iter in enumerate(
                stuct_iteration.impropers
            ):
                output_write.write(
                    (first_indent * 4)
                    % (
                        stuct_iteration.get_index(
                            improper_iter.connection_members[0]
                        )
                        + 1,
                        stuct_iteration.get_index(
                            improper_iter.connection_members[1]
                        )
                        + 1,
                        stuct_iteration.get_index(
                            improper_iter.connection_members[2]
                        )
                        + 1,
                        stuct_iteration.get_index(
                            improper_iter.connection_members[3]
                        )
                        + 1,
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

            # DONOR: calculate the donor data (not calculated here printing the header)
            output_write.write(first_indent % no_donors + " !NDON: donors\n")
            output_write.write("\n")

            # ACCEPTOR: calculate the acceptor data (not calculated here printing the header)
            output_write.write(
                first_indent % no_acceptors + " !NACC: acceptors\n"
            )
            output_write.write("\n")

            # NNB: calculate the NNB data (not calculated here printing the header)
            output_write.write(first_indent % no_NNB + " !NNB\n")
            output_write.write("\n")

            # GROUP: calculate the group data  (not calculated here printing the header)
            output_write.write(first_indent % no_groups + " !NGRP \n")
            output_write.write("\n")

            output_write.close()
        # **********************************
        # **********************************
        # psf writer (end)
        # **********************************
        # **********************************

    def write_pdb(self, space_group="P 1"):
        """This write_pdb function writes the Charmm style PDB (coordinate file), which can be utilized
        in the GOMC and NAMD engines.

        Parameters
        ----------
        space_group: str (default="P 1")
            The space group of the structure

        """
        # **********************************
        # **********************************
        # pdb writer (start)
        # **********************************
        # **********************************

        print("******************************")
        print("")
        print("The charmm pdb writer (the write_pdb function) is running")
        print("write_charmm_pdb: residues == {}".format(self.residues))
        print("fix_residue = {}".format(self.fix_residue))
        print("fix_residue_in_box = {}".format(self.fix_residue_in_box))
        print(
            "set_residue_pdb_occupancy_to_1 = {}".format(
                self.set_residue_pdb_occupancy_to_1
            )
        )
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

            if q == 0:
                residue_names_list_pdb = self.residue_names_list_box_0
                residue_id_list_pdb = self.residue_id_list_box_0
                res_no_chain_iter_corrected_list_pdb = (
                    self.res_no_chain_iter_corrected_list_box_0
                )
                segment_id_list_pdb = self.segment_id_list_box_0

            else:
                residue_names_list_pdb = self.residue_names_list_box_1
                residue_id_list_pdb = self.residue_id_list_box_1
                res_no_chain_iter_corrected_list_pdb = (
                    self.res_no_chain_iter_corrected_list_box_1
                )
                segment_id_list_pdb = self.segment_id_list_box_1

            output_write = open(output, "w")
            # output_write.write(
            #'REMARK this file ' + file_name_iteration + ' - created by using MoSDeF-GOMC.' + '\n')
            # output_write.write(
            #'REMARK parameters from the ' + str(self.forcefield_selection) + ' force field via MoSDef\n')
            # output_write.write('REMARK created on ' + str(date_time) + '\n')

            # caluculate the atom name and unique atom names
            max_no_atoms_in_base10 = 99999  # 99,999 for atoms in psf/pdb

            atom_no_list = []
            element_list = []

            occupancy_values_atoms_list = []
            fix_atoms_list = []
            atom_alternate_location_list = []
            residue_code_insertion_list = []
            x_y_z_coor_list = []
            segment_id = []

            atom_alternate_location_all_values = ""
            residue_code_insertion_all_values = ""
            segment_id_all_values = ""
            for f, site in enumerate(stuct_only_iteration.sites):
                if residue_names_list_pdb[f] not in self.residues:
                    self.input_error = True
                    print_error_message = "ERROR: Please specifiy all residues (residues) in a list"
                    raise ValueError(print_error_message)

                # get other values
                atom_no_list.append(stuct_only_iteration.get_index(site))

                # only 2 character element names are allowed
                site_name = str(site.__dict__["name_"])

                # extract element or atom name from mol2 without numbers (integers)
                if site_name[0] == "_":
                    element_name = site_name
                else:
                    element_name = ""
                    for site_name_char_i in site_name:
                        try:
                            int(site_name_char_i)

                        except:
                            element_name += site_name_char_i

                try:
                    # check if element is bead (i.e., first part of name "_")
                    if element_name[0] == "_":
                        element_name = "BD"
                    elif len(element_name) > 2:
                        element_name = "TL"
                except:
                    element_name = "UN"
                element_list.append(element_name)

                if (self.fix_residue is not None) and (
                    site.__dict__["residue_name_"] in self.fix_residue
                ):
                    beta_iteration = 1.00
                elif (self.fix_residue_in_box is not None) and (
                    site.__dict__["residue_name_"] in self.fix_residue_in_box
                ):
                    beta_iteration = 2.00
                else:
                    beta_iteration = 0.00
                fix_atoms_list.append(beta_iteration)

                if (self.set_residue_pdb_occupancy_to_1 is not None) and (
                    site.__dict__["residue_name_"]
                    in self.set_residue_pdb_occupancy_to_1
                ):
                    occupancy_iteration = 1.00
                else:
                    occupancy_iteration = 0.00
                occupancy_values_atoms_list.append(occupancy_iteration)

                atom_alternate_location_list.append(
                    atom_alternate_location_all_values
                )
                residue_code_insertion_list.append(
                    residue_code_insertion_all_values
                )

                x_y_z_coor = site.__dict__["position_"].to_value("angstrom")
                x_y_z_coor_list.append(x_y_z_coor)

                segment_id.append(segment_id_all_values)

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
                        stuct_only_iteration.box.lengths[0].to("angstrom"),
                        stuct_only_iteration.box.lengths[1].to("angstrom"),
                        stuct_only_iteration.box.lengths[2].to("angstrom"),
                        stuct_only_iteration.box.angles[0].to("degree"),
                        stuct_only_iteration.box.angles[1].to("degree"),
                        stuct_only_iteration.box.angles[2].to("degree"),
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
                residue_id_list_pdb,
                residue_names_list_pdb,
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
                        str(residue_names_list_pdb[v])[: self.max_resname_char],
                        segment_id_list_pdb[v],
                        res_no_chain_iter_corrected_list_pdb[v],
                        residue_code_insertion_list[v],
                        x_y_z_coor_list[v][0],
                        x_y_z_coor_list[v][1],
                        x_y_z_coor_list[v][2],
                        occupancy_values_atoms_list[v],
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
