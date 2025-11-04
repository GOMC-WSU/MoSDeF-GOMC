import mbuild as mb
import numpy as np
import pytest
from foyer.forcefields import forcefields
from gmso import Topology
from gmso.exceptions import GMSOError
from gmso.external.convert_mbuild import from_mbuild, to_mbuild
from mbuild import Box, Compound
from mbuild.lattice import load_cif
from mbuild.utils.io import get_fn, has_foyer

from mosdef_gomc.formats.gmso_charmm_writer import (
    Charmm,
    _Exp6_Rmin_to_sigma,
    _Exp6_Rmin_to_sigma_solver,
    _Exp6_sigma_to_Rmin,
    _Exp6_sigma_to_Rmin_solver,
)
from mosdef_gomc.tests.base_test import BaseTest
from mosdef_gomc.utils.conversion import (
    base10_to_base16_alph_num,
    base10_to_base22_alph,
    base10_to_base26_alph,
    base10_to_base44_alph,
    base10_to_base52_alph,
    base10_to_base54_alph_num,
    base10_to_base62_alph_num,
)
from mosdef_gomc.utils.io import get_mosdef_gomc_fn


@pytest.mark.skipif(not has_foyer, reason="Foyer package not installed")
class TestCharmmWriterData(BaseTest):
    def test_save(self, ethane_gomc):
        box_0 = mb.fill_box(
            compound=[ethane_gomc], n_compounds=[1], box=[4, 4, 4]
        )

        Charmm(
            box_0,
            "ethane",
            ff_filename="ethane",
            residues=[ethane_gomc.name],
            forcefield_selection="oplsaa",
        )

    def test_save_charmm_gomc_ff(self, ethane_gomc):
        box_0 = mb.fill_box(
            compound=[ethane_gomc], n_compounds=[2], box=[4, 4, 4]
        )

        charmm = Charmm(
            box_0,
            "charmm_data",
            ff_filename="charmm_data",
            residues=[ethane_gomc.name],
            forcefield_selection="oplsaa",
            atom_type_naming_style="general",
        )
        charmm.write_inp()

        with open("charmm_data.inp", "r") as fp:
            masses_read = False
            bonds_read = False
            angles_read = False
            dihedrals_read = False
            nonbondeds_read = False
            out_gomc = fp.readlines()
            for i, line in enumerate(out_gomc):
                if (
                    "! atom_types" in line
                    and "mass" in line
                    and "atomClass_ResidueName" in line
                ):
                    masses_read = True
                    assert len(out_gomc[i + 1].split("!")[0].split()) == 3
                    assert out_gomc[i + 1].split("!")[0].split()[0:3] == [
                        "*",
                        "CT",
                        "12.011",
                    ]
                    assert len(out_gomc[i + 2].split("!")[0].split()) == 3
                    assert out_gomc[i + 2].split("!")[0].split()[0:3] == [
                        "*",
                        "HC",
                        "1.008",
                    ]
                    assert out_gomc[i + 1].split()[4:5] == ["ETH_opls_135"]
                    assert out_gomc[i + 2].split()[4:5] == ["ETH_opls_140"]

                elif (
                    "! type_1" in line
                    and "type_2" in line
                    and "Kb" in line
                    and "b0" in line
                    and "extended_type_1" in line
                    and "extended_type_2" in line
                ):
                    bonds_read = True
                    bond_types = [
                        ["CT", "CT", "268.0", "1.529"],
                        ["CT", "HC", "340.0", "1.09"],
                    ]
                    assert len(out_gomc[i + 1].split("!")[0].split()) == 4
                    assert len(out_gomc[i + 2].split("!")[0].split()) == 4
                    if (
                        out_gomc[i + 1].split("!")[0].split()[0:4]
                        == bond_types[0]
                    ):
                        assert (
                            out_gomc[i + 1].split("!")[0].split()[0:4]
                            == bond_types[0]
                        )
                        assert (
                            out_gomc[i + 2].split("!")[0].split()[0:4]
                            == bond_types[1]
                        )
                    elif (
                        out_gomc[i + 1].split("!")[0].split()[0:4]
                        == bond_types[1]
                    ):
                        assert (
                            out_gomc[i + 1].split("!")[0].split()[0:4]
                            == bond_types[1]
                        )
                        assert (
                            out_gomc[i + 2].split("!")[0].split()[0:4]
                            == bond_types[0]
                        )

                elif (
                    "! type_1 " in line
                    and "type_2" in line
                    and "type_3" in line
                    and "Ktheta" in line
                    and "Theta0" in line
                    and "extended_type_1" in line
                    and "extended_type_2" in line
                    and "extended_type_3" in line
                ):
                    angles_read = True
                    angle_types = [
                        ["CT", "CT", "HC", "37.5", "110.7"],
                        ["HC", "CT", "HC", "33.0", "107.8"],
                    ]
                    assert len(out_gomc[i + 1].split("!")[0].split()) == 5
                    assert len(out_gomc[i + 2].split("!")[0].split()) == 5
                    if (
                        out_gomc[i + 1].split("!")[0].split()[0:5]
                        == angle_types[0]
                    ):
                        assert (
                            out_gomc[i + 1].split("!")[0].split()[0:5]
                            == angle_types[0]
                        )
                        assert (
                            out_gomc[i + 2].split("!")[0].split()[0:5]
                            == angle_types[1]
                        )
                    elif (
                        out_gomc[i + 1].split("!")[0].split()[0:4]
                        == angle_types[1]
                    ):
                        assert (
                            out_gomc[i + 1].split("!")[0].split()[0:5]
                            == angle_types[1]
                        )
                        assert (
                            out_gomc[i + 2].split("!")[0].split()[0:5]
                            == angle_types[0]
                        )

                elif (
                    "! type_1" in line
                    and "type_2" in line
                    and "type_3" in line
                    and "type_4" in line
                    and "Kchi" in line
                    and "n" in line
                    and "delta" in line
                    and "extended_type_1" in line
                    and "extended_type_2" in line
                    and "extended_type_3" in line
                    and "extended_type_4" in line
                ):
                    dihedrals_read = True
                    dihed_types = [
                        ["HC", "CT", "CT", "HC", "-0.15", "3", "180.0"],
                    ]
                    for j in range(0, len(dihed_types)):
                        assert (
                            len(out_gomc[i + 1 + j].split("!")[0].split()) == 7
                        )
                        assert (
                            out_gomc[i + 1 + j].split("!")[0].split()[0:7]
                            == dihed_types[j]
                        )

                elif (
                    "! type_1" in line
                    and "ignored" in line
                    and "epsilon" in line
                    and "Rmin/2" in line
                    and "ignored" in line
                    and "epsilon,1-4" in line
                    and "Rmin/2,1-4" in line
                    and "extended_type_1" in line
                    and "extended_type_2" in line
                ):
                    nonbondeds_read = True
                    nb_types = [
                        [
                            "CT",
                            "0.0",
                            "-0.066",
                            "1.9643085845",
                            "0.0",
                            "-0.033",
                            "1.9643085845",
                        ],
                        [
                            "HC",
                            "0.0",
                            "-0.03",
                            "1.4030775604",
                            "0.0",
                            "-0.015",
                            "1.4030775604",
                        ],
                    ]

                    for j in range(0, len(nb_types)):
                        assert (
                            len(out_gomc[i + 1 + j].split("!")[0].split()) == 7
                        )
                        assert (
                            out_gomc[i + 1 + j].split("!")[0].split()[0:7]
                            == nb_types[j]
                        )

                else:
                    pass

        assert masses_read
        assert bonds_read
        assert angles_read
        assert dihedrals_read
        assert nonbondeds_read

    def test_save_charmm_psf(self, ethane_gomc):
        box_0 = mb.fill_box(
            compound=[ethane_gomc], n_compounds=[2], box=[4, 4, 4]
        )

        charmm = Charmm(
            box_0,
            "charmm_data",
            ff_filename="charmm_data",
            residues=[ethane_gomc.name],
            forcefield_selection="oplsaa",
            atom_type_naming_style="general",
        )
        charmm.write_psf()

        with open("charmm_data.psf", "r") as fp:
            charges_read = False
            out_gomc = fp.readlines()
            for i, line in enumerate(out_gomc):
                if "16 !NATOM" in line:
                    charges_read = True
                    atom_type_charge_etc_list = [
                        [
                            "1",
                            "A",
                            "1",
                            "ETH",
                            "C1",
                            "CT",
                            "-0.180000",
                            "12.0110",
                        ],
                        [
                            "2",
                            "A",
                            "1",
                            "ETH",
                            "C2",
                            "CT",
                            "-0.180000",
                            "12.0110",
                        ],
                        [
                            "3",
                            "A",
                            "1",
                            "ETH",
                            "H1",
                            "HC",
                            "0.060000",
                            "1.0080",
                        ],
                        [
                            "4",
                            "A",
                            "1",
                            "ETH",
                            "H2",
                            "HC",
                            "0.060000",
                            "1.0080",
                        ],
                        [
                            "5",
                            "A",
                            "1",
                            "ETH",
                            "H3",
                            "HC",
                            "0.060000",
                            "1.0080",
                        ],
                        [
                            "6",
                            "A",
                            "1",
                            "ETH",
                            "H4",
                            "HC",
                            "0.060000",
                            "1.0080",
                        ],
                        [
                            "7",
                            "A",
                            "1",
                            "ETH",
                            "H5",
                            "HC",
                            "0.060000",
                            "1.0080",
                        ],
                        [
                            "8",
                            "A",
                            "1",
                            "ETH",
                            "H6",
                            "HC",
                            "0.060000",
                            "1.0080",
                        ],
                        [
                            "9",
                            "A",
                            "2",
                            "ETH",
                            "C1",
                            "CT",
                            "-0.180000",
                            "12.0110",
                        ],
                        [
                            "10",
                            "A",
                            "2",
                            "ETH",
                            "C2",
                            "CT",
                            "-0.180000",
                            "12.0110",
                        ],
                        [
                            "11",
                            "A",
                            "2",
                            "ETH",
                            "H1",
                            "HC",
                            "0.060000",
                            "1.0080",
                        ],
                        [
                            "12",
                            "A",
                            "2",
                            "ETH",
                            "H2",
                            "HC",
                            "0.060000",
                            "1.0080",
                        ],
                        [
                            "13",
                            "A",
                            "2",
                            "ETH",
                            "H3",
                            "HC",
                            "0.060000",
                            "1.0080",
                        ],
                        [
                            "14",
                            "A",
                            "2",
                            "ETH",
                            "H4",
                            "HC",
                            "0.060000",
                            "1.0080",
                        ],
                        [
                            "15",
                            "A",
                            "2",
                            "ETH",
                            "H5",
                            "HC",
                            "0.060000",
                            "1.0080",
                        ],
                        [
                            "16",
                            "A",
                            "2",
                            "ETH",
                            "H6",
                            "HC",
                            "0.060000",
                            "1.0080",
                        ],
                    ]
                    for j in range(0, len(atom_type_charge_etc_list)):
                        assert (
                            out_gomc[i + 1 + j].split()[0:8]
                            == atom_type_charge_etc_list[j]
                        )

                else:
                    pass

        assert charges_read

    def test_save_charmm_pdb(self, ethane_gomc):
        box_0 = mb.fill_box(
            compound=[ethane_gomc], n_compounds=[2], box=[4, 4, 4]
        )

        charmm = Charmm(
            box_0,
            "charmm_data",
            ff_filename="charmm_data",
            residues=[ethane_gomc.name],
            forcefield_selection="oplsaa",
            atom_type_naming_style="general",
        )
        charmm.write_pdb()

        with open("charmm_data.pdb", "r") as fp:
            pdb_read = False
            out_gomc = fp.readlines()
            for i, line in enumerate(out_gomc):
                if "CRYST1" in line:
                    pdb_read = True
                    atom_type_res_part_1_list = [
                        ["ATOM", "1", "C1", "ETH", "A", "1"],
                        ["ATOM", "2", "C2", "ETH", "A", "1"],
                        ["ATOM", "3", "H1", "ETH", "A", "1"],
                        ["ATOM", "4", "H2", "ETH", "A", "1"],
                        ["ATOM", "5", "H3", "ETH", "A", "1"],
                        ["ATOM", "6", "H4", "ETH", "A", "1"],
                        ["ATOM", "7", "H5", "ETH", "A", "1"],
                        ["ATOM", "8", "H6", "ETH", "A", "1"],
                        ["ATOM", "9", "C1", "ETH", "A", "2"],
                        ["ATOM", "10", "C2", "ETH", "A", "2"],
                        ["ATOM", "11", "H1", "ETH", "A", "2"],
                        ["ATOM", "12", "H2", "ETH", "A", "2"],
                        ["ATOM", "13", "H3", "ETH", "A", "2"],
                        ["ATOM", "14", "H4", "ETH", "A", "2"],
                        ["ATOM", "15", "H5", "ETH", "A", "2"],
                        ["ATOM", "16", "H6", "ETH", "A", "2"],
                    ]
                    atom_type_res_part_2_list = [
                        ["0.00", "0.00", "C"],
                        ["0.00", "0.00", "C"],
                        ["0.00", "0.00", "H"],
                        ["0.00", "0.00", "H"],
                        ["0.00", "0.00", "H"],
                        ["0.00", "0.00", "H"],
                        ["0.00", "0.00", "H"],
                        ["0.00", "0.00", "H"],
                        ["0.00", "0.00", "C"],
                        ["0.00", "0.00", "C"],
                        ["0.00", "0.00", "H"],
                        ["0.00", "0.00", "H"],
                        ["0.00", "0.00", "H"],
                        ["0.00", "0.00", "H"],
                        ["0.00", "0.00", "H"],
                        ["0.00", "0.00", "H"],
                    ]

                    for j in range(0, len(atom_type_res_part_1_list)):
                        assert (
                            out_gomc[i + 1 + j].split()[0:6]
                            == atom_type_res_part_1_list[j]
                        )
                        assert (
                            out_gomc[i + 1 + j].split()[9:12]
                            == atom_type_res_part_2_list[j]
                        )

                else:
                    pass

        assert pdb_read

    def test_save_charmm_psf_2_ethane_from_single_mol2(self):
        two_ethane_mol2_file = Topology.load(
            get_mosdef_gomc_fn("2_ethane.mol2")
        )
        two_ethane_mol2_file = to_mbuild(two_ethane_mol2_file)
        two_ethane_mol2_file.name = "ETH"

        box_0 = mb.fill_box(
            compound=[two_ethane_mol2_file], n_compounds=[1], box=[4, 4, 4]
        )

        charmm = Charmm(
            box_0,
            "charmm_data_2_ethane_from_single_mol2",
            ff_filename="charmm_data_2_ethane_from_single_mol2",
            residues=[two_ethane_mol2_file.name],
            forcefield_selection="oplsaa",
            atom_type_naming_style="general",
            gmso_match_ff_by="group",
        )
        charmm.write_psf()

        with open("charmm_data_2_ethane_from_single_mol2.psf", "r") as fp:
            charges_read = False
            out_gomc = fp.readlines()
            for i, line in enumerate(out_gomc):
                if "16 !NATOM" in line:
                    charges_read = True
                    atom_type_charge_etc_list = [
                        [
                            "1",
                            "A",
                            "1",
                            "ETH",
                            "C1",
                            "CT",
                            "-0.180000",
                            "12.0110",
                        ],
                        [
                            "2",
                            "A",
                            "1",
                            "ETH",
                            "C2",
                            "CT",
                            "-0.180000",
                            "12.0110",
                        ],
                        [
                            "3",
                            "A",
                            "1",
                            "ETH",
                            "H1",
                            "HC",
                            "0.060000",
                            "1.0080",
                        ],
                        [
                            "4",
                            "A",
                            "1",
                            "ETH",
                            "H2",
                            "HC",
                            "0.060000",
                            "1.0080",
                        ],
                        [
                            "5",
                            "A",
                            "1",
                            "ETH",
                            "H3",
                            "HC",
                            "0.060000",
                            "1.0080",
                        ],
                        [
                            "6",
                            "A",
                            "1",
                            "ETH",
                            "H4",
                            "HC",
                            "0.060000",
                            "1.0080",
                        ],
                        [
                            "7",
                            "A",
                            "1",
                            "ETH",
                            "H5",
                            "HC",
                            "0.060000",
                            "1.0080",
                        ],
                        [
                            "8",
                            "A",
                            "1",
                            "ETH",
                            "H6",
                            "HC",
                            "0.060000",
                            "1.0080",
                        ],
                        [
                            "9",
                            "A",
                            "2",
                            "ETH",
                            "C1",
                            "CT",
                            "-0.180000",
                            "12.0110",
                        ],
                        [
                            "10",
                            "A",
                            "2",
                            "ETH",
                            "C2",
                            "CT",
                            "-0.180000",
                            "12.0110",
                        ],
                        [
                            "11",
                            "A",
                            "2",
                            "ETH",
                            "H1",
                            "HC",
                            "0.060000",
                            "1.0080",
                        ],
                        [
                            "12",
                            "A",
                            "2",
                            "ETH",
                            "H2",
                            "HC",
                            "0.060000",
                            "1.0080",
                        ],
                        [
                            "13",
                            "A",
                            "2",
                            "ETH",
                            "H3",
                            "HC",
                            "0.060000",
                            "1.0080",
                        ],
                        [
                            "14",
                            "A",
                            "2",
                            "ETH",
                            "H4",
                            "HC",
                            "0.060000",
                            "1.0080",
                        ],
                        [
                            "15",
                            "A",
                            "2",
                            "ETH",
                            "H5",
                            "HC",
                            "0.060000",
                            "1.0080",
                        ],
                        [
                            "16",
                            "A",
                            "2",
                            "ETH",
                            "H6",
                            "HC",
                            "0.060000",
                            "1.0080",
                        ],
                    ]
                    for j in range(0, len(atom_type_charge_etc_list)):
                        assert (
                            out_gomc[i + 1 + j].split()[0:8]
                            == atom_type_charge_etc_list[j]
                        )

                else:
                    pass

        assert charges_read

    def test_save_charmm_pdb_2_ethane_from_single_mol2(self):
        two_ethane_mol2_file = Topology.load(
            get_mosdef_gomc_fn("2_ethane.mol2")
        )
        two_ethane_mol2_file = to_mbuild(two_ethane_mol2_file)
        two_ethane_mol2_file.name = "ETH"
        box_0 = mb.fill_box(
            compound=[two_ethane_mol2_file], n_compounds=[1], box=[4, 4, 4]
        )

        charmm = Charmm(
            box_0,
            "charmm_data_2_ethane_from_single_mol2",
            ff_filename="charmm_data_2_ethane_from_single_mol2",
            residues=[two_ethane_mol2_file.name],
            forcefield_selection="oplsaa",
            atom_type_naming_style="general",
        )
        charmm.write_pdb()

        with open("charmm_data_2_ethane_from_single_mol2.pdb", "r") as fp:
            pdb_read = False
            out_gomc = fp.readlines()
            for i, line in enumerate(out_gomc):
                if "CRYST1" in line:
                    pdb_read = True
                    atom_type_res_part_1_list = [
                        ["ATOM", "1", "C1", "ETH", "A", "1"],
                        ["ATOM", "2", "C2", "ETH", "A", "1"],
                        ["ATOM", "3", "H1", "ETH", "A", "1"],
                        ["ATOM", "4", "H2", "ETH", "A", "1"],
                        ["ATOM", "5", "H3", "ETH", "A", "1"],
                        ["ATOM", "6", "H4", "ETH", "A", "1"],
                        ["ATOM", "7", "H5", "ETH", "A", "1"],
                        ["ATOM", "8", "H6", "ETH", "A", "1"],
                        ["ATOM", "9", "C1", "ETH", "A", "2"],
                        ["ATOM", "10", "C2", "ETH", "A", "2"],
                        ["ATOM", "11", "H1", "ETH", "A", "2"],
                        ["ATOM", "12", "H2", "ETH", "A", "2"],
                        ["ATOM", "13", "H3", "ETH", "A", "2"],
                        ["ATOM", "14", "H4", "ETH", "A", "2"],
                        ["ATOM", "15", "H5", "ETH", "A", "2"],
                        ["ATOM", "16", "H6", "ETH", "A", "2"],
                    ]
                    atom_type_res_part_2_list = [
                        ["0.00", "0.00", "C"],
                        ["0.00", "0.00", "C"],
                        ["0.00", "0.00", "H"],
                        ["0.00", "0.00", "H"],
                        ["0.00", "0.00", "H"],
                        ["0.00", "0.00", "H"],
                        ["0.00", "0.00", "H"],
                        ["0.00", "0.00", "H"],
                        ["0.00", "0.00", "C"],
                        ["0.00", "0.00", "C"],
                        ["0.00", "0.00", "H"],
                        ["0.00", "0.00", "H"],
                        ["0.00", "0.00", "H"],
                        ["0.00", "0.00", "H"],
                        ["0.00", "0.00", "H"],
                        ["0.00", "0.00", "H"],
                    ]

                    for j in range(0, len(atom_type_res_part_1_list)):
                        assert (
                            out_gomc[i + 1 + j].split()[0:6]
                            == atom_type_res_part_1_list[j]
                        )
                        assert (
                            out_gomc[i + 1 + j].split()[9:12]
                            == atom_type_res_part_2_list[j]
                        )

                else:
                    pass

        assert pdb_read

    def test_save_charmm_different_1_4_NB_interactions_gomc_ff(
        self, ethane_gomc, water
    ):
        box_0 = mb.fill_box(
            compound=[ethane_gomc, water], n_compounds=[1, 1], box=[4, 4, 4]
        )

        charmm = Charmm(
            box_0,
            "charmm_different_1_4_NB_interactions",
            ff_filename="charmm_different_1_4_NB_interactions",
            residues=[ethane_gomc.name, water.name],
            forcefield_selection={
                ethane_gomc.name: "oplsaa",
                water.name: get_mosdef_gomc_fn(
                    "spce_coul_14_half__LJ_14_zero.xml"
                ),
            },
            gomc_fix_bonds_angles=[water.name],
            atom_type_naming_style="general",
        )
        charmm.write_inp()

        with open("charmm_different_1_4_NB_interactions.inp", "r") as fp:
            nonbondeds_read = False
            out_gomc = fp.readlines()
            for i, line in enumerate(out_gomc):
                if (
                    "! type_1" in line
                    and "ignored" in line
                    and "epsilon" in line
                    and "Rmin/2" in line
                    and "ignored" in line
                    and "epsilon,1-4" in line
                    and "Rmin/2,1-4" in line
                    and "extended_type_1" in line
                    and "extended_type_2" in line
                ):
                    nonbondeds_read = True
                    nb_types = [
                        [
                            "CT",
                            "0.0",
                            "-0.066",
                            "1.9643085845",
                            "0.0",
                            "-0.033",
                            "1.9643085845",
                        ],
                        [
                            "HC",
                            "0.0",
                            "-0.03",
                            "1.4030775604",
                            "0.0",
                            "-0.015",
                            "1.4030775604",
                        ],
                        [
                            "OW",
                            "0.0",
                            "-0.1554000956",
                            "1.7766160931",
                            "0.0",
                            "-0.0",
                            "1.7766160931",
                        ],
                        [
                            "HW",
                            "0.0",
                            "-0.0",
                            "0.0",
                            "0.0",
                            "-0.0",
                            "0.0",
                        ],
                    ]

                    for j in range(0, len(nb_types)):
                        assert (
                            len(out_gomc[i + 1 + j].split("!")[0].split()) == 7
                        )
                        assert (
                            out_gomc[i + 1 + j].split("!")[0].split()[0:7]
                            == nb_types[j]
                        )

                else:
                    pass

        assert nonbondeds_read

    def test_save_charmm_gomc_ethyl_benzene_aa_ff(self):
        ethyl_benzene_aa = Topology.load(
            get_mosdef_gomc_fn("ethyl_benzene_aa.mol2")
        )
        ethyl_benzene_aa = to_mbuild(ethyl_benzene_aa)
        box_0 = mb.fill_box(
            compound=[ethyl_benzene_aa], n_compounds=[1], box=[4, 4, 4]
        )

        charmm = Charmm(
            box_0,
            "charmm_ethyl_benzene_aa_data",
            ff_filename="charmm_ethyl_benzene_aa_data",
            residues=["EBN"],
            forcefield_selection=get_mosdef_gomc_fn(
                "benzene_and_alkane_branched_benzene_aa.xml"
            ),
            atom_type_naming_style="general",
        )
        charmm.write_inp()

        with open("charmm_ethyl_benzene_aa_data.inp", "r") as fp:
            improper_read = False
            out_gomc = fp.readlines()
            for i, line in enumerate(out_gomc):
                if (
                    "! type_1" in line
                    and "type_2" in line
                    and "type_3" in line
                    and "type_4" in line
                    and "Kw" in line
                    and "n" in line
                    and "w0" in line
                    and "extended_type_1" in line
                    and "extended_type_2" in line
                    and "extended_type_3" in line
                    and "extended_type_4" in line
                ):
                    improper_read = True
                    improper_types = [
                        ["CE", "HCE", "CE", "CE", "0.956023", "1", "0.0"],
                        ["CE", "HCE", "CE", "CE", "1.075526", "3", "180.0"],
                        ["CE", "CT", "CE", "CE", "1.195029", "2", "0.0"],
                    ]
                    for j in range(0, len(improper_types)):
                        assert (
                            len(out_gomc[i + 1 + j].split("!")[0].split()) == 7
                        )
                        assert (
                            out_gomc[i + 1 + j].split("!")[0].split()[0:7]
                            == improper_types[j]
                        )

                else:
                    pass

        assert improper_read

    def test_save_charmm_gomc_methyl_benzene_aa_ff(self):
        methyl_benzene_aa = Topology.load(
            get_mosdef_gomc_fn("methyl_benzene_aa.mol2")
        )
        methyl_benzene_aa = to_mbuild(methyl_benzene_aa)
        methyl_benzene_aa.name = "BBB"

        box_0 = mb.fill_box(
            compound=[methyl_benzene_aa], n_compounds=[1], box=[4, 4, 4]
        )

        charmm = Charmm(
            box_0,
            "charmm_methyl_benzene_aa_data",
            ff_filename="charmm_methyl_benzene_aa_data",
            residues=[methyl_benzene_aa.name],
            forcefield_selection=get_mosdef_gomc_fn(
                "benzene_and_alkane_branched_benzene_aa.xml"
            ),
            atom_type_naming_style="all_unique",
            gmso_match_ff_by="group",
        )
        charmm.write_inp()

        with open("charmm_methyl_benzene_aa_data.inp", "r") as fp:
            improper_read = False
            out_gomc = fp.readlines()
            for i, line in enumerate(out_gomc):
                if (
                    "! type_1" in line
                    and "type_2" in line
                    and "type_3" in line
                    and "type_4" in line
                    and "Kw" in line
                    and "n" in line
                    and "w0" in line
                    and "extended_type_1" in line
                    and "extended_type_2" in line
                    and "extended_type_3" in line
                    and "extended_type_4" in line
                ):
                    improper_read = True
                    improper_types = [
                        ["CE0", "HCE0", "CE0", "CE1", "0.956023", "1", "0.0"],
                        ["CE0", "HCE0", "CE0", "CE1", "1.075526", "3", "180.0"],
                        ["CE0", "HCE0", "CE0", "CE0", "0.956023", "1", "0.0"],
                        ["CE0", "HCE0", "CE0", "CE0", "1.075526", "3", "180.0"],
                        ["CE1", "CT0", "CE0", "CE0", "1.195029", "2", "0.0"],
                    ]
                    for j in range(0, len(improper_types)):
                        assert (
                            len(out_gomc[i + 1 + j].split("!")[0].split()) == 7
                        )
                        assert (
                            out_gomc[i + 1 + j].split("!")[0].split()[0:7]
                            == improper_types[j]
                        )

                else:
                    pass

        assert improper_read

    def test_save_charmm_gomc_benzene_aa_ff(self):
        benzene_aa = Topology.load(get_mosdef_gomc_fn("benzene_aa.mol2"))
        benzene_aa = to_mbuild(benzene_aa)
        box_0 = mb.fill_box(
            compound=[benzene_aa], n_compounds=[1], box=[4, 4, 4]
        )

        charmm = Charmm(
            box_0,
            "charmm_benzene_aa_data",
            ff_filename="charmm_benzene_aa_data",
            residues=["BEN"],
            forcefield_selection=get_mosdef_gomc_fn(
                "benzene_and_alkane_branched_benzene_aa.xml"
            ),
            atom_type_naming_style="general",
        )
        charmm.write_inp()

        with open("charmm_benzene_aa_data.inp", "r") as fp:
            improper_read = False
            out_gomc = fp.readlines()
            for i, line in enumerate(out_gomc):
                if (
                    "! type_1" in line
                    and "type_2" in line
                    and "type_3" in line
                    and "type_4" in line
                    and "Kw" in line
                    and "n" in line
                    and "w0" in line
                    and "extended_type_1" in line
                    and "extended_type_2" in line
                    and "extended_type_3" in line
                    and "extended_type_4" in line
                ):
                    improper_read = True
                    improper_types = [
                        ["CE", "HCE", "CE", "CE", "0.956023", "1", "0.0"],
                        ["CE", "HCE", "CE", "CE", "1.075526", "3", "180.0"],
                    ]

                    for j in range(0, len(improper_types)):
                        assert (
                            len(out_gomc[i + 1 + j].split("!")[0].split()) == 7
                        )
                        assert (
                            out_gomc[i + 1 + j].split("!")[0].split()[0:7]
                            == improper_types[j]
                        )

                else:
                    pass

        assert improper_read

    def test_save_charmm_ethyl_benzene_aa_psf(self):
        ethyl_benzene_aa = Topology.load(
            get_mosdef_gomc_fn("ethyl_benzene_aa.mol2")
        )
        ethyl_benzene_aa = to_mbuild(ethyl_benzene_aa)

        box_0 = mb.fill_box(
            compound=[ethyl_benzene_aa], n_compounds=[1], box=[4, 4, 4]
        )

        charmm = Charmm(
            box_0,
            "charmm_ethyl_benzene_aa_data",
            ff_filename="charmm_ethyl_benzene_aa_data",
            residues=["EBN"],
            forcefield_selection=get_mosdef_gomc_fn(
                "benzene_and_alkane_branched_benzene_aa.xml"
            ),
            atom_type_naming_style="general",
        )
        charmm.write_psf()

        with open("charmm_ethyl_benzene_aa_data.psf", "r") as fp:
            charges_read = False
            out_gomc = fp.readlines()
            for i, line in enumerate(out_gomc):
                if "18 !NATOM" in line:
                    charges_read = True
                    atom_type_charge_etc_list = [
                        [
                            "1",
                            "A",
                            "1",
                            "EBN",
                            "C1",
                            "CE",
                            "0.010000",
                            "12.0110",
                        ],
                        [
                            "2",
                            "A",
                            "1",
                            "EBN",
                            "C2",
                            "CE",
                            "0.010000",
                            "12.0110",
                        ],
                        [
                            "3",
                            "A",
                            "1",
                            "EBN",
                            "C3",
                            "CE",
                            "0.010000",
                            "12.0110",
                        ],
                        [
                            "4",
                            "A",
                            "1",
                            "EBN",
                            "C4",
                            "CE",
                            "0.010000",
                            "12.0110",
                        ],
                        [
                            "5",
                            "A",
                            "1",
                            "EBN",
                            "C5",
                            "CE",
                            "0.010000",
                            "12.0110",
                        ],
                        [
                            "6",
                            "A",
                            "1",
                            "EBN",
                            "C6",
                            "CE",
                            "0.010000",
                            "12.0110",
                        ],
                        [
                            "7",
                            "A",
                            "1",
                            "EBN",
                            "C7",
                            "CT",
                            "0.030000",
                            "12.0110",
                        ],
                        [
                            "8",
                            "A",
                            "1",
                            "EBN",
                            "C8",
                            "CT",
                            "0.030000",
                            "12.0110",
                        ],
                        [
                            "9",
                            "A",
                            "1",
                            "EBN",
                            "H1",
                            "HCE",
                            "0.040000",
                            "1.0080",
                        ],
                        [
                            "10",
                            "A",
                            "1",
                            "EBN",
                            "H2",
                            "HCE",
                            "0.040000",
                            "1.0080",
                        ],
                        [
                            "11",
                            "A",
                            "1",
                            "EBN",
                            "H3",
                            "HCE",
                            "0.040000",
                            "1.0080",
                        ],
                        [
                            "12",
                            "A",
                            "1",
                            "EBN",
                            "H4",
                            "HCE",
                            "0.040000",
                            "1.0080",
                        ],
                        [
                            "13",
                            "A",
                            "1",
                            "EBN",
                            "H5",
                            "HCE",
                            "0.040000",
                            "1.0080",
                        ],
                        [
                            "14",
                            "A",
                            "1",
                            "EBN",
                            "H6",
                            "HCT",
                            "0.050000",
                            "1.0080",
                        ],
                        [
                            "15",
                            "A",
                            "1",
                            "EBN",
                            "H7",
                            "HCT",
                            "0.050000",
                            "1.0080",
                        ],
                        [
                            "16",
                            "A",
                            "1",
                            "EBN",
                            "H8",
                            "HCT",
                            "0.050000",
                            "1.0080",
                        ],
                        [
                            "17",
                            "A",
                            "1",
                            "EBN",
                            "H9",
                            "HCT",
                            "0.050000",
                            "1.0080",
                        ],
                        [
                            "18",
                            "A",
                            "1",
                            "EBN",
                            "HA",
                            "HCT",
                            "0.050000",
                            "1.0080",
                        ],
                    ]
                    for j in range(0, len(atom_type_charge_etc_list)):
                        assert (
                            out_gomc[i + 1 + j].split()[0:8]
                            == atom_type_charge_etc_list[j]
                        )

                else:
                    pass

        assert charges_read

    def test_save_charmm_ua_gomc_ff(self, two_propanol_ua):
        box_0 = mb.fill_box(
            compound=[two_propanol_ua], n_compounds=[1], box=[4, 4, 4]
        )

        charmm = Charmm(
            box_0,
            "charmm_UA",
            ff_filename="charmm_UA",
            residues=[two_propanol_ua.name],
            forcefield_selection="trappe-ua",
            bead_to_atom_name_dict={"_CH3": "C"},
            atom_type_naming_style="general",
        )
        charmm.write_inp()

        with open("charmm_UA.inp", "r") as fp:
            masses_read = False
            bonds_read = False
            angles_read = False
            dihedrals_read = False
            nonbondeds_read = False
            out_gomc = fp.readlines()
            for i, line in enumerate(out_gomc):
                if (
                    "! atom_types" in line
                    and "mass" in line
                    and "atomClass_ResidueName" in line
                ):
                    masses_read = True
                    atom_types_1 = [
                        ["*", "CH3", "15.035"],
                        ["*", "CH", "13.019"],
                        ["*", "O", "15.9994"],
                        ["*", "H", "1.008"],
                    ]
                    atom_types_2 = [
                        ["POL_CH3_sp3"],
                        ["POL_CH_O"],
                        ["POL_O"],
                        ["POL_H"],
                    ]

                    for j in range(0, len(atom_types_1)):
                        assert (
                            len(out_gomc[i + 1 + j].split("!")[0].split()) == 3
                        )
                        assert (
                            out_gomc[i + 1 + j].split("!")[0].split()[0:3]
                            == atom_types_1[j]
                        )
                        assert (
                            out_gomc[i + 1 + j].split()[4:5] == atom_types_2[j]
                        )

                elif (
                    "! type_1" in line
                    and "type_2" in line
                    and "Kb" in line
                    and "b0" in line
                    and "extended_type_1" in line
                    and "extended_type_2" in line
                ):
                    bonds_read = True
                    bond_types = [
                        ["CH3", "CH", "600.40153", "1.54"],
                        ["CH", "O", "600.40153", "1.43"],
                        ["O", "H", "600.40153", "0.945"],
                    ]
                    total_bonds_evaluated = []
                    total_bonds_evaluated_reorg = []
                    for j in range(0, len(bond_types)):
                        assert (
                            len(out_gomc[i + 1 + j].split("!")[0].split()) == 4
                        )

                        if (
                            out_gomc[i + 1 + j].split("!")[0].split()[0:4]
                            == bond_types[0]
                            or bond_types[1]
                            or bond_types[2]
                        ):
                            total_bonds_evaluated.append(
                                out_gomc[i + 1 + j].split("!")[0].split()[0:4]
                            )
                    for k in range(0, len(bond_types)):
                        if bond_types[k] in total_bonds_evaluated:
                            total_bonds_evaluated_reorg.append(bond_types[k])
                    assert total_bonds_evaluated_reorg == bond_types

                elif (
                    "! type_1 " in line
                    and "type_2" in line
                    and "type_3" in line
                    and "Ktheta" in line
                    and "Theta0" in line
                    and "extended_type_1" in line
                    and "extended_type_2" in line
                    and "extended_type_3" in line
                ):
                    angles_read = True
                    angle_types = [
                        ["CH3", "CH", "O", "50.077544", "109.469889"],
                        ["CH3", "CH", "CH3", "62.10013", "112.000071"],
                        ["CH", "O", "H", "55.045554", "108.499872"],
                    ]
                    total_angles_evaluated = []
                    total_angles_evaluated_reorg = []
                    for j in range(0, len(angle_types)):
                        assert (
                            len(out_gomc[i + 1 + j].split("!")[0].split()) == 5
                        )
                        if (
                            out_gomc[i + 1 + j].split("!")[0].split()[0:5]
                            == angle_types[0]
                            or angle_types[1]
                            or angle_types[2]
                        ):
                            total_angles_evaluated.append(
                                out_gomc[i + 1 + j].split("!")[0].split()[0:5]
                            )
                    for k in range(0, len(angle_types)):
                        if angle_types[k] in total_angles_evaluated:
                            total_angles_evaluated_reorg.append(angle_types[k])

                    assert total_angles_evaluated_reorg == angle_types

                elif (
                    "! type_1" in line
                    and "type_2" in line
                    and "type_3" in line
                    and "type_4" in line
                    and "Kchi" in line
                    and "n" in line
                    and "delta" in line
                    and "extended_type_1" in line
                    and "extended_type_2" in line
                    and "extended_type_3" in line
                    and "extended_type_4" in line
                ):
                    dihedrals_read = True
                    dihedral_types = [
                        ["CH3", "CH", "O", "H", "-0.392135", "1", "180.0"],
                        ["CH3", "CH", "O", "H", "-0.062518", "2", "0.0"],
                        ["CH3", "CH", "O", "H", "0.345615", "3", "180.0"],
                    ]
                    for j in range(0, len(dihedral_types)):
                        assert (
                            len(out_gomc[i + 1 + j].split("!")[0].split()) == 7
                        )
                        assert (
                            out_gomc[i + 1 + j].split("!")[0].split()[0:7]
                            == dihedral_types[j]
                        )

                elif (
                    "! type_1" in line
                    and "ignored" in line
                    and "epsilon" in line
                    and "Rmin/2" in line
                    and "ignored" in line
                    and "epsilon,1-4" in line
                    and "Rmin/2,1-4" in line
                    and "extended_type_1" in line
                    and "extended_type_2" in line
                ):
                    nonbondeds_read = True
                    nb_types = [
                        [
                            "CH3",
                            "0.0",
                            "-0.1947459369",
                            "2.1046163406",
                            "0.0",
                            "-0.0",
                            "2.1046163406",
                        ],
                        [
                            "CH",
                            "0.0",
                            "-0.0198720124",
                            "2.4301303346",
                            "0.0",
                            "-0.0",
                            "2.4301303346",
                        ],
                        [
                            "O",
                            "0.0",
                            "-0.1848099904",
                            "1.6949176929",
                            "0.0",
                            "-0.0",
                            "1.6949176929",
                        ],
                        [
                            "H",
                            "0.0",
                            "-0.0",
                            "5.6123102415",
                            "0.0",
                            "-0.0",
                            "5.6123102415",
                        ],
                    ]

                    for j in range(0, len(nb_types)):
                        assert (
                            len(out_gomc[i + 1 + j].split("!")[0].split()) == 7
                        )
                        assert (
                            out_gomc[i + 1 + j].split("!")[0].split()[0:7]
                            == nb_types[j]
                        )

                else:
                    pass

        assert masses_read
        assert bonds_read
        assert angles_read
        assert dihedrals_read
        assert nonbondeds_read

    def test_save_charmm_ua_atom_data_psf(self, two_propanol_ua):
        box_0 = mb.fill_box(
            compound=[two_propanol_ua], n_compounds=[1], box=[4, 4, 4]
        )

        charmm = Charmm(
            box_0,
            "charmm_atom_data_UA",
            ff_filename="charmm_atom_data_UA",
            residues=[two_propanol_ua.name],
            forcefield_selection="trappe-ua",
            bead_to_atom_name_dict={"_CH3": "C"},
            atom_type_naming_style="general",
        )
        charmm.write_psf()

        with open("charmm_atom_data_UA.psf", "r") as fp:
            read_psf = False
            out_gomc = fp.readlines()
            for i, line in enumerate(out_gomc):
                if "5 !NATOM" in line:
                    read_psf = True
                    atom_type_charge_etc_list = [
                        [
                            "1",
                            "A",
                            "1",
                            "POL",
                            "C1",
                            "CH3",
                            "0.000000",
                            "15.0350",
                        ],
                        [
                            "2",
                            "A",
                            "1",
                            "POL",
                            "BD1",
                            "CH",
                            "0.265000",
                            "13.0190",
                        ],
                        [
                            "3",
                            "A",
                            "1",
                            "POL",
                            "O1",
                            "O",
                            "-0.700000",
                            "15.9994",
                        ],
                        [
                            "4",
                            "A",
                            "1",
                            "POL",
                            "H1",
                            "H",
                            "0.435000",
                            "1.0080",
                        ],
                        [
                            "5",
                            "A",
                            "1",
                            "POL",
                            "C2",
                            "CH3",
                            "0.000000",
                            "15.0350",
                        ],
                    ]

                    for j in range(0, len(atom_type_charge_etc_list)):
                        assert (
                            out_gomc[i + 1 + j].split()[0:8]
                            == atom_type_charge_etc_list[j]
                        )

                else:
                    pass

        assert read_psf

    def test_save_charmm_ua_pdb(self, two_propanol_ua):
        box_0 = mb.fill_box(
            compound=[two_propanol_ua], n_compounds=[1], box=[4, 4, 4]
        )

        charmm = Charmm(
            box_0,
            "charmm_UA",
            ff_filename="charmm_UA",
            residues=[two_propanol_ua.name],
            forcefield_selection="trappe-ua",
            bead_to_atom_name_dict={"_CH3": "C"},
            atom_type_naming_style="general",
        )
        charmm.write_pdb()

        with open("charmm_UA.pdb", "r") as fp:
            read_pdb = False
            out_gomc = fp.readlines()
            for i, line in enumerate(out_gomc):
                if "CRYST1" in line:
                    read_pdb = True
                    atom_type_res_part_1_list = [
                        ["ATOM", "1", "C1", "POL", "A", "1"],
                        ["ATOM", "2", "BD1", "POL", "A", "1"],
                        ["ATOM", "3", "O1", "POL", "A", "1"],
                        ["ATOM", "4", "H1", "POL", "A", "1"],
                        ["ATOM", "5", "C2", "POL", "A", "1"],
                    ]
                    atom_type_res_part_2_list = [
                        ["0.00", "0.00", "BD"],
                        ["0.00", "0.00", "BD"],
                        ["0.00", "0.00", "O"],
                        ["0.00", "0.00", "H"],
                        ["0.00", "0.00", "BD"],
                    ]

                    for j in range(0, len(atom_type_res_part_1_list)):
                        assert (
                            out_gomc[i + 1 + j].split()[0:6]
                            == atom_type_res_part_1_list[j]
                        )
                        assert (
                            out_gomc[i + 1 + j].split()[9:12]
                            == atom_type_res_part_2_list[j]
                        )

                else:
                    pass

        assert read_pdb

    def test_save_charmm_ua_single_bond_data_with_6_char_psf(self):
        molecule = mb.load(
            get_mosdef_gomc_fn("psf_file_single_bond_test_ua.mol2")
        )
        molecule.box = mb.Box(lengths=[4, 4, 4])
        molecule.name = "MOL"

        charmm = Charmm(
            molecule,
            "charmm_single_bond_data_with_6_char_UA",
            ff_filename="charmm_single_bond_data_with_6_char_UA",
            residues=[molecule.name],
            forcefield_selection=get_mosdef_gomc_fn(
                "charmm_psf_6_char_atom_names_test_single_bonds_angles_dihedrals_impropers_ua.xml"
            ),
            bead_to_atom_name_dict={"_BD": "BD"},
            atom_type_naming_style="general",
        )
        charmm.write_psf()

        with open("charmm_single_bond_data_with_6_char_UA.psf", "r") as fp:
            read_psf = False
            out_gomc = fp.readlines()
            for i, line in enumerate(out_gomc):
                if "2 !NATOM" in line:
                    read_psf = True
                    atom_type_charge_etc_list = [
                        [
                            "1",
                            "A",
                            "1",
                            "MOL",
                            "BD1",
                            "BDABCD",
                            "0.000000",
                            "10.0000",
                        ],
                        [
                            "2",
                            "A",
                            "1",
                            "MOL",
                            "BD2",
                            "BDABCD",
                            "0.000000",
                            "10.0000",
                        ],
                    ]

                    for j in range(0, len(atom_type_charge_etc_list)):
                        assert (
                            out_gomc[i + 1 + j].split()[0:8]
                            == atom_type_charge_etc_list[j]
                        )

                else:
                    pass

        assert read_psf

    # *********************
    # new psf bonded atom tests (start)
    # *********************
    def test_save_charmm_ua_single_bond_data_psf(self):
        molecule = mb.load(
            get_mosdef_gomc_fn("psf_file_single_bond_test_ua.mol2")
        )
        molecule.box = mb.Box(lengths=[4, 4, 4])
        molecule.name = "MOL"

        charmm = Charmm(
            molecule,
            "charmm_single_bond_data_UA",
            ff_filename="charmm_single_bond_data_UA",
            residues=[molecule.name],
            forcefield_selection=get_mosdef_gomc_fn(
                "charmm_psf_test_single_bonds_angles_dihedrals_impropers_ua.xml"
            ),
            bead_to_atom_name_dict={"_BD": "BD"},
            atom_type_naming_style="general",
        )
        charmm.write_psf()

        with open("charmm_single_bond_data_UA.psf", "r") as fp:
            read_psf = False
            out_gomc = fp.readlines()
            for i, line in enumerate(out_gomc):
                if "1 !NBOND: bonds" in line:
                    read_psf = True
                    atom_number_arrangement_1 = [["1", "2"]]
                    atom_number_arrangement_2 = [["2", "1"]]

                    assert len(atom_number_arrangement_1) == len(
                        atom_number_arrangement_2
                    )

                    for j in range(0, len(atom_number_arrangement_1)):
                        assert (
                            out_gomc[i + 1 + j].split()[0:2]
                            == atom_number_arrangement_1[j]
                            or out_gomc[i + 1 + j].split()[0:2]
                            == atom_number_arrangement_2[j]
                        )

                else:
                    pass

        assert read_psf

    def test_save_charmm_ua_single_angle_data_psf(self):
        molecule = mb.load(
            get_mosdef_gomc_fn("psf_file_single_angle_test_ua.mol2")
        )
        molecule.box = mb.Box(lengths=[4, 4, 4])
        molecule.name = "MOL"

        charmm = Charmm(
            molecule,
            "charmm_single_angle_data_UA",
            ff_filename="charmm_single_angle_data_UA",
            residues=[molecule.name],
            forcefield_selection=get_mosdef_gomc_fn(
                "charmm_psf_test_single_bonds_angles_dihedrals_impropers_ua.xml"
            ),
            bead_to_atom_name_dict={"_BD": "BD"},
            atom_type_naming_style="general",
        )
        charmm.write_psf()

        with open("charmm_single_angle_data_UA.psf", "r") as fp:
            read_psf = False
            out_gomc = fp.readlines()
            for i, line in enumerate(out_gomc):
                if "1 !NTHETA: angles" in line:
                    read_psf = True
                    atom_number_arrangement_1 = [["3", "2", "1"]]
                    atom_number_arrangement_2 = [["1", "2", "3"]]

                    assert len(atom_number_arrangement_1) == len(
                        atom_number_arrangement_2
                    )

                    for j in range(0, len(atom_number_arrangement_1)):
                        assert (
                            out_gomc[i + 1 + j].split()[0:3]
                            == atom_number_arrangement_1[j]
                            or out_gomc[i + 1 + j].split()[0:3]
                            == atom_number_arrangement_2[j]
                        )
                else:
                    pass

        assert read_psf

    def test_save_charmm_ua_single_dihedral_data_psf(self):
        molecule = mb.load(
            get_mosdef_gomc_fn("psf_file_single_dihedral_test_ua.mol2")
        )
        molecule.box = mb.Box(lengths=[4, 4, 4])
        molecule.name = "MOL"

        charmm = Charmm(
            molecule,
            "charmm_single_dihedral_data_UA",
            ff_filename="charmm_single_dihedral_data_UA",
            residues=[molecule.name],
            forcefield_selection=get_mosdef_gomc_fn(
                "charmm_psf_test_single_bonds_angles_dihedrals_impropers_ua.xml"
            ),
            bead_to_atom_name_dict={"_BD": "BD"},
            atom_type_naming_style="general",
        )
        charmm.write_psf()

        with open("charmm_single_dihedral_data_UA.psf", "r") as fp:
            read_psf = False
            out_gomc = fp.readlines()
            for i, line in enumerate(out_gomc):
                if "1 !NPHI: dihedrals" in line:
                    read_psf = True
                    atom_number_arrangement_1 = [["4", "3", "2", "1"]]
                    atom_number_arrangement_2 = [["1", "2", "3", "4"]]

                    assert len(atom_number_arrangement_1) == len(
                        atom_number_arrangement_2
                    )

                    for j in range(0, len(atom_number_arrangement_1)):
                        assert (
                            out_gomc[i + 1 + j].split()[0:4]
                            == atom_number_arrangement_1[j]
                            or out_gomc[i + 1 + j].split()[0:4]
                            == atom_number_arrangement_2[j]
                        )
                else:
                    pass

        assert read_psf

    def test_save_charmm_ua_single_improper_data_psf(self):
        molecule = mb.load(
            get_mosdef_gomc_fn("psf_file_single_improper_test_ua.mol2")
        )
        molecule.box = mb.Box(lengths=[4, 4, 4])
        molecule.name = "MOL"

        charmm = Charmm(
            molecule,
            "charmm_single_improper_data_UA",
            ff_filename="charmm_single_improper_data_UA",
            residues=[molecule.name],
            forcefield_selection=get_mosdef_gomc_fn(
                "charmm_psf_test_single_bonds_angles_dihedrals_impropers_ua.xml"
            ),
            bead_to_atom_name_dict={"_BD": "BD"},
            atom_type_naming_style="general",
        )
        charmm.write_psf()

        with open("charmm_single_improper_data_UA.psf", "r") as fp:
            read_psf = False
            out_gomc = fp.readlines()
            for i, line in enumerate(out_gomc):
                if "1 !NIMPHI: impropers" in line:
                    read_psf = True
                    atom_number_arrangement_1 = [["1", "2", "3", "4"]]
                    atom_number_arrangement_2 = [["1", "2", "4", "3"]]
                    atom_number_arrangement_3 = [["1", "3", "2", "4"]]
                    atom_number_arrangement_4 = [["1", "3", "4", "2"]]
                    atom_number_arrangement_5 = [["1", "4", "2", "3"]]
                    atom_number_arrangement_6 = [["1", "4", "3", "2"]]

                    assert (
                        len(atom_number_arrangement_1)
                        == len(atom_number_arrangement_2)
                        == len(atom_number_arrangement_3)
                        == len(atom_number_arrangement_4)
                        == len(atom_number_arrangement_5)
                        == len(atom_number_arrangement_6)
                    )

                    for j in range(0, len(atom_number_arrangement_1)):
                        assert (
                            out_gomc[i + 1 + j].split()[0:4]
                            == atom_number_arrangement_1[j]
                            or out_gomc[i + 1 + j].split()[0:4]
                            == atom_number_arrangement_2[j]
                            or out_gomc[i + 1 + j].split()[0:4]
                            == atom_number_arrangement_3[j]
                            or out_gomc[i + 1 + j].split()[0:4]
                            == atom_number_arrangement_4[j]
                            or out_gomc[i + 1 + j].split()[0:4]
                            == atom_number_arrangement_5[j]
                            or out_gomc[i + 1 + j].split()[0:4]
                            == atom_number_arrangement_6[j]
                        )
                else:
                    pass

        assert read_psf

    def test_save_charmm_ua_number_bonds_angles_dihedrals_impropers_psf(self):
        molecule = mb.load(
            get_mosdef_gomc_fn("psf_file_single_improper_test_ua.mol2")
        )
        molecule.box = mb.Box(lengths=[4, 4, 4])
        molecule.name = "MOL"

        charmm = Charmm(
            molecule,
            "charmm_number_bonds_angles_dihedrals_impropers_UA",
            ff_filename="charmm_number_bonds_angles_dihedrals_impropers_UA",
            residues=[molecule.name],
            forcefield_selection=get_mosdef_gomc_fn(
                "charmm_psf_test_single_bonds_angles_dihedrals_impropers_ua.xml"
            ),
            bead_to_atom_name_dict={"_BD": "BD"},
            atom_type_naming_style="general",
        )
        charmm.write_psf()

        with open(
            "charmm_number_bonds_angles_dihedrals_impropers_UA.psf", "r"
        ) as fp:
            read_psf = False
            out_gomc = fp.readlines()
            correct_number_bonds = False
            correct_number_angles = False
            correct_number_dihedrals = False
            correct_number_impropers = False
            for i, line in enumerate(out_gomc):
                if "PSF" in line:
                    read_psf = True

                elif "6 !NBOND: bonds" in line:
                    correct_number_bonds = True

                elif "7 !NTHETA: angles" in line:
                    correct_number_angles = True

                elif " 4 !NPHI: dihedrals" in line:
                    correct_number_dihedrals = True

                elif "1 !NIMPHI: impropers" in line:
                    correct_number_impropers = True

                else:
                    pass

        assert read_psf
        assert correct_number_bonds
        assert correct_number_angles
        assert correct_number_dihedrals
        assert correct_number_impropers

    # *********************
    # new psf angleed atom tests (end)
    # *********************

    def test_save_charmm_mie_ua_gomc_ff(self, water, two_propanol_ua):
        box_0 = mb.fill_box(
            compound=[water, two_propanol_ua], n_compounds=[1, 1], box=[5, 4, 3]
        )

        charmm = Charmm(
            box_0,
            "charmm_data_Mie_UA",
            ff_filename="charmm_data_Mie_UA",
            residues=[water.name, two_propanol_ua.name],
            forcefield_selection={
                water.name: get_mosdef_gomc_fn(
                    "gmso_spce_water__lorentz_combining.xml"
                ),
                two_propanol_ua.name: get_mosdef_gomc_fn(
                    "gmso_two_propanol_Mie_ua.xml"
                ),
            },
            bead_to_atom_name_dict={"_CH3": "C", "_CH2": "C", "_HC": "C"},
            gomc_fix_bonds_angles=[water.name],
            atom_type_naming_style="general",
        )
        charmm.write_inp()

        with open("charmm_data_Mie_UA.inp", "r") as fp:
            masses_read = False
            bonds_read = False
            angles_read = False
            dihedrals_read = False
            nonbondeds_read = False
            out_gomc = fp.readlines()
            for i, line in enumerate(out_gomc):
                if (
                    "! atom_types" in line
                    and "mass" in line
                    and "atomClass_ResidueName" in line
                ):
                    masses_read = True
                    atom_types_1 = [
                        ["*", "OW", "15.999"],
                        ["*", "HW", "1.008"],
                        ["*", "CH3", "15.035"],
                        ["*", "CH", "13.019"],
                        ["*", "O", "15.9994"],
                        ["*", "H", "1.008"],
                    ]
                    atom_types_2 = [
                        ["WAT_o_spce"],
                        ["WAT_h_spce"],
                        ["POL_CH3_sp3"],
                        ["POL_CH_O"],
                        ["POL_O"],
                        ["POL_H"],
                    ]

                    for j in range(0, len(atom_types_1)):
                        assert (
                            len(out_gomc[i + 1 + j].split("!")[0].split()) == 3
                        )
                        assert (
                            out_gomc[i + 1 + j].split("!")[0].split()[0:3]
                            == atom_types_1[j]
                        )
                        assert (
                            out_gomc[i + 1 + j].split()[4:5] == atom_types_2[j]
                        )

                elif (
                    "! type_1" in line
                    and "type_2" in line
                    and "Kb" in line
                    and "b0" in line
                    and "extended_type_1" in line
                    and "extended_type_2" in line
                ):
                    bonds_read = True
                    bond_types = [
                        ["OW", "HW", "999999999999", "1.0"],
                        ["CH3", "CH", "604267.5897", "1.5401"],
                        ["CH", "O", "604267.5897", "1.4301"],
                        ["O", "H", "604267.5897", "0.9451"],
                    ]
                    total_bonds_evaluated = []
                    total_bonds_evaluated_reorg = []
                    for j in range(0, len(bond_types)):
                        assert (
                            len(out_gomc[i + 1 + j].split("!")[0].split()) == 4
                        )

                        if (
                            out_gomc[i + 1 + j].split("!")[0].split()[0:4]
                            == bond_types[0]
                            or bond_types[1]
                            or bond_types[2]
                        ):
                            total_bonds_evaluated.append(
                                out_gomc[i + 1 + j].split("!")[0].split()[0:4]
                            )
                    for k in range(0, len(bond_types)):
                        if bond_types[k] in total_bonds_evaluated:
                            total_bonds_evaluated_reorg.append(bond_types[k])
                    assert total_bonds_evaluated_reorg == bond_types

                elif (
                    "! type_1 " in line
                    and "type_2" in line
                    and "type_3" in line
                    and "Ktheta" in line
                    and "Theta0" in line
                    and "extended_type_1" in line
                    and "extended_type_2" in line
                    and "extended_type_3" in line
                ):
                    angles_read = True
                    angle_types = [
                        ["HW", "OW", "HW", "999999999999", "109.47"],
                        ["CH3", "CH", "CH3", "62500.0036", "112.01"],
                        ["CH3", "CH", "O", "50400.0029", "109.51"],
                        ["CH", "O", "H", "55400.0031", "108.51"],
                    ]
                    total_angles_evaluated = []
                    total_angles_evaluated_reorg = []
                    for j in range(0, len(angle_types)):
                        assert (
                            len(out_gomc[i + 1 + j].split("!")[0].split()) == 5
                        )
                        if (
                            out_gomc[i + 1 + j].split("!")[0].split()[0:5]
                            == angle_types[0]
                            or angle_types[1]
                            or angle_types[2]
                        ):
                            total_angles_evaluated.append(
                                out_gomc[i + 1 + j].split("!")[0].split()[0:5]
                            )
                    for k in range(0, len(angle_types)):
                        if angle_types[k] in total_angles_evaluated:
                            total_angles_evaluated_reorg.append(angle_types[k])

                    assert total_angles_evaluated_reorg == angle_types

                elif (
                    "! type_1" in line
                    and "type_2" in line
                    and "type_3" in line
                    and "type_4" in line
                    and "Kchi" in line
                    and "n" in line
                    and "delta" in line
                    and "extended_type_1" in line
                    and "extended_type_2" in line
                    and "extended_type_3" in line
                    and "extended_type_4" in line
                ):
                    dihedrals_read = True
                    dihedral_types = [
                        ["CH3", "CH", "O", "H", "209.82", "1", "0.0"],
                        ["CH3", "CH", "O", "H", "-29.17", "2", "180.0"],
                        ["CH3", "CH", "O", "H", "187.93", "3", "0.0"],
                    ]
                    for j in range(0, len(dihedral_types)):
                        assert (
                            len(out_gomc[i + 1 + j].split("!")[0].split()) == 7
                        )
                        assert (
                            out_gomc[i + 1 + j].split("!")[0].split()[0:7]
                            == dihedral_types[j]
                        )

                elif (
                    "! type_1" in line
                    and "epsilon" in line
                    and "sigma" in line
                    and "n" in line
                    and "epsilon,1-4" in line
                    and "sigma,1-4" in line
                    and "n,1-4" in line
                    and "extended_type_1" in line
                    and "extended_type_2" in line
                ):
                    nonbondeds_read = True
                    nb_types = [
                        [
                            "OW",
                            "78.200368",
                            "3.16557",
                            "12.0",
                            "0.0",
                            "3.16557",
                            "12.0",
                        ],
                        [
                            "HW",
                            "0.0",
                            "1.0",
                            "12.0",
                            "0.0",
                            "1.0",
                            "12.0",
                        ],
                        [
                            "CH3",
                            "98.000006",
                            "3.751",
                            "11.0",
                            "0.0",
                            "3.751",
                            "11.0",
                        ],
                        [
                            "CH",
                            "10.000001",
                            "4.681",
                            "12.0",
                            "0.0",
                            "4.681",
                            "12.0",
                        ],
                        [
                            "O",
                            "93.000005",
                            "3.021",
                            "13.0",
                            "0.0",
                            "3.021",
                            "13.0",
                        ],
                        [
                            "H",
                            "0.0",
                            "0.0",
                            "14.0",
                            "0.0",
                            "0.0",
                            "14.0",
                        ],
                    ]

                    for j in range(0, len(nb_types)):
                        assert (
                            len(out_gomc[i + 1 + j].split("!")[0].split()) == 7
                        )
                        assert (
                            out_gomc[i + 1 + j].split("!")[0].split()[0:7]
                            == nb_types[j]
                        )

                else:
                    pass

        assert masses_read
        assert bonds_read
        assert angles_read
        assert dihedrals_read
        assert nonbondeds_read

    def test_save_charmm_mie_ua_psf(self, water, two_propanol_ua):
        box_0 = mb.fill_box(
            compound=[water, two_propanol_ua], n_compounds=[1, 1], box=[5, 4, 3]
        )

        charmm = Charmm(
            box_0,
            "charmm_data_Mie_UA",
            ff_filename="charmm_data_Mie_UA",
            residues=[water.name, two_propanol_ua.name],
            forcefield_selection={
                water.name: get_mosdef_gomc_fn(
                    "gmso_spce_water__lorentz_combining.xml"
                ),
                two_propanol_ua.name: get_mosdef_gomc_fn(
                    "gmso_two_propanol_Mie_ua.xml"
                ),
            },
            bead_to_atom_name_dict={"_CH3": "C", "_CH2": "C", "_HC": "C"},
            gomc_fix_bonds_angles=[water.name],
            atom_type_naming_style="general",
        )
        charmm.write_psf()

        with open("charmm_data_Mie_UA.psf", "r") as fp:
            read_psf = False
            out_gomc = fp.readlines()
            for i, line in enumerate(out_gomc):
                if "8 !NATOM" in line:
                    read_psf = True
                    atom_type_charge_etc_list = [
                        [
                            "1",
                            "A",
                            "1",
                            "WAT",
                            "O1",
                            "OW",
                            "-0.847600",
                            "15.9990",
                        ],
                        [
                            "2",
                            "A",
                            "1",
                            "WAT",
                            "H1",
                            "HW",
                            "0.423800",
                            "1.0080",
                        ],
                        [
                            "3",
                            "A",
                            "1",
                            "WAT",
                            "H2",
                            "HW",
                            "0.423800",
                            "1.0080",
                        ],
                        [
                            "4",
                            "A",
                            "2",
                            "POL",
                            "C1",
                            "CH3",
                            "0.000000",
                            "15.0350",
                        ],
                        [
                            "5",
                            "A",
                            "2",
                            "POL",
                            "C2",
                            "CH",
                            "0.265000",
                            "13.0190",
                        ],
                        [
                            "6",
                            "A",
                            "2",
                            "POL",
                            "O1",
                            "O",
                            "-0.700000",
                            "15.9994",
                        ],
                        [
                            "7",
                            "A",
                            "2",
                            "POL",
                            "H1",
                            "H",
                            "0.435000",
                            "1.0080",
                        ],
                        [
                            "8",
                            "A",
                            "2",
                            "POL",
                            "C3",
                            "CH3",
                            "0.000000",
                            "15.0350",
                        ],
                    ]

                    for j in range(0, len(atom_type_charge_etc_list)):
                        assert (
                            out_gomc[i + 1 + j].split()[0:8]
                            == atom_type_charge_etc_list[j]
                        )

                else:
                    pass

        assert read_psf

    def test_save_charmm_mie_ua_pdb(self, water, two_propanol_ua):
        box_0 = mb.fill_box(
            compound=[water, two_propanol_ua], n_compounds=[1, 1], box=[5, 4, 3]
        )

        charmm = Charmm(
            box_0,
            "charmm_data_Mie_UA",
            ff_filename="charmm_data_Mie_UA",
            residues=[water.name, two_propanol_ua.name],
            forcefield_selection={
                water.name: get_mosdef_gomc_fn(
                    "gmso_spce_water__lorentz_combining.xml"
                ),
                two_propanol_ua.name: get_mosdef_gomc_fn(
                    "gmso_two_propanol_Mie_ua.xml"
                ),
            },
            bead_to_atom_name_dict={"_CH3": "C", "_CH2": "C", "_HC": "C"},
            gomc_fix_bonds_angles=[water.name],
            atom_type_naming_style="general",
        )
        charmm.write_pdb()

        with open("charmm_data_Mie_UA.pdb", "r") as fp:
            read_pdb = False
            out_gomc = fp.readlines()
            for i, line in enumerate(out_gomc):
                if (
                    "CRYST1" in line
                    and "50.000" in line
                    and "40.000" in line
                    and "30.000" in line
                ):
                    read_pdb = True
                    atom_type_res_part_1_list = [
                        ["ATOM", "1", "O1", "WAT", "A", "1"],
                        ["ATOM", "2", "H1", "WAT", "A", "1"],
                        ["ATOM", "3", "H2", "WAT", "A", "1"],
                        ["ATOM", "4", "C1", "POL", "A", "2"],
                        ["ATOM", "5", "C2", "POL", "A", "2"],
                        ["ATOM", "6", "O1", "POL", "A", "2"],
                        ["ATOM", "7", "H1", "POL", "A", "2"],
                        ["ATOM", "8", "C3", "POL", "A", "2"],
                    ]
                    atom_type_res_part_2_list = [
                        ["0.00", "0.00", "O"],
                        ["0.00", "0.00", "H"],
                        ["0.00", "0.00", "H"],
                        ["0.00", "0.00", "BD"],
                        ["0.00", "0.00", "BD"],
                        ["0.00", "0.00", "O"],
                        ["0.00", "0.00", "H"],
                        ["0.00", "0.00", "BD"],
                    ]

                    for j in range(0, len(atom_type_res_part_1_list)):
                        assert (
                            out_gomc[i + 1 + j].split()[0:6]
                            == atom_type_res_part_1_list[j]
                        )
                        assert (
                            out_gomc[i + 1 + j].split()[9:12]
                            == atom_type_res_part_2_list[j]
                        )

                else:
                    pass

        assert read_pdb

    def test_save_charmm_mie_ua_K_energy_units_periodic_ff(
        self, water, two_propanol_ua
    ):
        box_0 = mb.fill_box(
            compound=[water, two_propanol_ua], n_compounds=[4, 4], box=[5, 4, 3]
        )

        charmm = Charmm(
            box_0,
            "charmm_mie_ua_K_energy_units_periodic",
            ff_filename="charmm_mie_ua_K_energy_units_periodic",
            residues=[water.name, two_propanol_ua.name],
            forcefield_selection={
                water.name: get_mosdef_gomc_fn(
                    "gmso_spce_water__lorentz_combining.xml"
                ),
                two_propanol_ua.name: get_mosdef_gomc_fn(
                    "gmso_two_propanol_Mie_periodic_dihedral_ua_K_energy_units.xml"
                ),
            },
            bead_to_atom_name_dict={"_CH3": "C", "_CH2": "C", "_HC": "C"},
            gomc_fix_bonds_angles=[water.name],
            atom_type_naming_style="general",
        )
        charmm.write_inp()

        with open("charmm_mie_ua_K_energy_units_periodic.inp", "r") as fp:
            masses_read = False
            bonds_read = False
            angles_read = False
            dihedrals_read = False
            nonbondeds_read = False
            out_gomc = fp.readlines()
            for i, line in enumerate(out_gomc):
                if (
                    "! atom_types" in line
                    and "mass" in line
                    and "atomClass_ResidueName" in line
                ):
                    masses_read = True
                    atom_types_1 = [
                        ["*", "OW", "15.999"],
                        ["*", "HW", "1.008"],
                        ["*", "CH3", "15.035"],
                        ["*", "CH", "13.019"],
                        ["*", "O", "15.9994"],
                        ["*", "H", "1.008"],
                    ]
                    atom_types_2 = [
                        ["WAT_o_spce"],
                        ["WAT_h_spce"],
                        ["POL_CH3_sp3"],
                        ["POL_CH_O"],
                        ["POL_O"],
                        ["POL_H"],
                    ]

                    for j in range(0, len(atom_types_1)):
                        assert (
                            len(out_gomc[i + 1 + j].split("!")[0].split()) == 3
                        )
                        assert (
                            out_gomc[i + 1 + j].split("!")[0].split()[0:3]
                            == atom_types_1[j]
                        )
                        assert (
                            out_gomc[i + 1 + j].split()[4:5] == atom_types_2[j]
                        )

                elif (
                    "! type_1" in line
                    and "type_2" in line
                    and "Kb" in line
                    and "b0" in line
                    and "extended_type_1" in line
                    and "extended_type_2" in line
                ):
                    bonds_read = True
                    bond_types = [
                        ["OW", "HW", "999999999999", "1.0"],
                        ["CH3", "CH", "604267.5553", "1.54"],
                        ["CH", "O", "604267.5553", "1.43"],
                        ["O", "H", "604267.5553", "0.945"],
                    ]
                    total_bonds_evaluated = []
                    total_bonds_evaluated_reorg = []
                    for j in range(0, len(bond_types)):
                        assert (
                            len(out_gomc[i + 1 + j].split("!")[0].split()) == 4
                        )

                        if (
                            out_gomc[i + 1 + j].split("!")[0].split()[0:4]
                            == bond_types[0]
                            or bond_types[1]
                            or bond_types[2]
                        ):
                            total_bonds_evaluated.append(
                                out_gomc[i + 1 + j].split("!")[0].split()[0:4]
                            )
                    for k in range(0, len(bond_types)):
                        if bond_types[k] in total_bonds_evaluated:
                            total_bonds_evaluated_reorg.append(bond_types[k])
                    assert total_bonds_evaluated_reorg == bond_types

                elif (
                    "! type_1 " in line
                    and "type_2" in line
                    and "type_3" in line
                    and "Ktheta" in line
                    and "Theta0" in line
                    and "extended_type_1" in line
                    and "extended_type_2" in line
                    and "extended_type_3" in line
                ):
                    angles_read = True
                    angle_types = [
                        ["HW", "OW", "HW", "999999999999", "109.47"],
                        ["CH3", "CH", "O", "50400.0", "109.5"],
                        ["CH3", "CH", "CH3", "62500.0", "112.0"],
                        ["CH", "O", "H", "55400.0", "108.5"],
                    ]
                    total_angles_evaluated = []
                    total_angles_evaluated_reorg = []
                    for j in range(0, len(angle_types)):
                        assert (
                            len(out_gomc[i + 1 + j].split("!")[0].split()) == 5
                        )
                        if (
                            out_gomc[i + 1 + j].split("!")[0].split()[0:5]
                            == angle_types[0]
                            or angle_types[1]
                            or angle_types[2]
                        ):
                            total_angles_evaluated.append(
                                out_gomc[i + 1 + j].split("!")[0].split()[0:5]
                            )
                    for k in range(0, len(angle_types)):
                        if angle_types[k] in total_angles_evaluated:
                            total_angles_evaluated_reorg.append(angle_types[k])

                    assert total_angles_evaluated_reorg == angle_types

                elif (
                    "! type_1" in line
                    and "type_2" in line
                    and "type_3" in line
                    and "type_4" in line
                    and "Kchi" in line
                    and "n" in line
                    and "delta" in line
                    and "extended_type_1" in line
                    and "extended_type_2" in line
                    and "extended_type_3" in line
                    and "extended_type_4" in line
                ):
                    dihedrals_read = True
                    dihedral_types = [
                        ["CH3", "CH", "O", "H", "-18.75", "0", "90.0"],
                        ["CH3", "CH", "O", "H", "10.0", "1", "180.0"],
                        ["CH3", "CH", "O", "H", "-10.0", "2", "0.0"],
                        ["CH3", "CH", "O", "H", "10.0", "3", "180.0"],
                        ["CH3", "CH", "O", "H", "-0.625", "4", "0.0"],
                    ]
                    for j in range(0, len(dihedral_types)):
                        assert (
                            len(out_gomc[i + 1 + j].split("!")[0].split()) == 7
                        )
                        assert (
                            out_gomc[i + 1 + j].split("!")[0].split()[0:7]
                            == dihedral_types[j]
                        )

                elif (
                    "! type_1" in line
                    and "epsilon" in line
                    and "sigma" in line
                    and "n" in line
                    and "epsilon,1-4" in line
                    and "sigma,1-4" in line
                    and "n,1-4" in line
                    and "extended_type_1" in line
                    and "extended_type_2" in line
                ):
                    nonbondeds_read = True
                    nb_types = [
                        [
                            "OW",
                            "78.200368",
                            "3.16557",
                            "12.0",
                            "0.0",
                            "3.16557",
                            "12.0",
                        ],
                        [
                            "HW",
                            "0.0",
                            "1.0",
                            "12.0",
                            "0.0",
                            "1.0",
                            "12.0",
                        ],
                        [
                            "CH3",
                            "98.0",
                            "3.75",
                            "11.0",
                            "0.0",
                            "3.75",
                            "11.0",
                        ],
                        [
                            "CH",
                            "10.0",
                            "4.68",
                            "12.0",
                            "0.0",
                            "4.68",
                            "12.0",
                        ],
                        [
                            "O",
                            "93.0",
                            "3.02",
                            "13.0",
                            "0.0",
                            "3.02",
                            "13.0",
                        ],
                        [
                            "H",
                            "0.0",
                            "0.0",
                            "14.0",
                            "0.0",
                            "0.0",
                            "14.0",
                        ],
                    ]

                    for j in range(0, len(nb_types)):
                        assert (
                            len(out_gomc[i + 1 + j].split("!")[0].split()) == 7
                        )
                        assert (
                            out_gomc[i + 1 + j].split("!")[0].split()[0:7]
                            == nb_types[j]
                        )

                else:
                    pass

        assert masses_read
        assert bonds_read
        assert angles_read
        assert dihedrals_read
        assert nonbondeds_read

    def test_save_charmm_mie_ua_K_energy_units_OPLS_ff(
        self, water, two_propanol_ua
    ):
        box_0 = mb.fill_box(
            compound=[water, two_propanol_ua], n_compounds=[1, 1], box=[5, 4, 3]
        )

        charmm = Charmm(
            box_0,
            "charmm_mie_ua_K_energy_units_OPLS",
            ff_filename="charmm_mie_ua_K_energy_units_OPLS",
            residues=[water.name, two_propanol_ua.name],
            forcefield_selection={
                water.name: get_mosdef_gomc_fn(
                    "gmso_spce_water_one_for_nb_and_coul__lorentz_combining.xml"
                ),
                two_propanol_ua.name: get_mosdef_gomc_fn(
                    "gmso_two_propanol_Mie_OPLS_dihedral_ua_K_energy_units.xml"
                ),
            },
            bead_to_atom_name_dict={"_CH3": "C", "_CH2": "C", "_HC": "C"},
            gomc_fix_bonds_angles=[water.name],
            atom_type_naming_style="general",
        )
        charmm.write_inp()

        with open("charmm_mie_ua_K_energy_units_OPLS.inp", "r") as fp:
            dihedrals_read = False
            out_gomc = fp.readlines()
            for i, line in enumerate(out_gomc):
                if (
                    "! type_1" in line
                    and "type_2" in line
                    and "type_3" in line
                    and "type_4" in line
                    and "Kchi" in line
                    and "n" in line
                    and "delta" in line
                    and "extended_type_1" in line
                    and "extended_type_2" in line
                    and "extended_type_3" in line
                    and "extended_type_4" in line
                ):
                    dihedrals_read = True
                    dihedral_types = [
                        ["CH3", "CH", "O", "H", "-18.75", "0", "90.0"],
                        ["CH3", "CH", "O", "H", "10.0", "1", "180.0"],
                        ["CH3", "CH", "O", "H", "-10.0", "2", "0.0"],
                        ["CH3", "CH", "O", "H", "10.0", "3", "180.0"],
                        ["CH3", "CH", "O", "H", "-0.625", "4", "0.0"],
                    ]
                    for j in range(0, len(dihedral_types)):
                        assert (
                            len(out_gomc[i + 1 + j].split("!")[0].split()) == 7
                        )
                        assert (
                            out_gomc[i + 1 + j].split("!")[0].split()[0:7]
                            == dihedral_types[j]
                        )
                else:
                    pass

        assert dihedrals_read

    def test_save_charmm_mie_ua_K_energy_units_RB_ff(
        self, water, two_propanol_ua
    ):
        box_0 = mb.fill_box(
            compound=[water, two_propanol_ua], n_compounds=[1, 1], box=[5, 4, 3]
        )

        charmm = Charmm(
            box_0,
            "charmm_mie_ua_K_energy_units_RB",
            ff_filename="charmm_mie_ua_K_energy_units_RB",
            residues=[water.name, two_propanol_ua.name],
            forcefield_selection={
                water.name: get_mosdef_gomc_fn(
                    "gmso_spce_water__lorentz_combining.xml"
                ),
                two_propanol_ua.name: get_mosdef_gomc_fn(
                    "gmso_two_propanol_Mie_RB_dihedral_ua_K_energy_units.xml"
                ),
            },
            bead_to_atom_name_dict={"_CH3": "C", "_CH2": "C", "_HC": "C"},
            gomc_fix_bonds_angles=[water.name],
            atom_type_naming_style="general",
        )
        charmm.write_inp()

        with open("charmm_mie_ua_K_energy_units_RB.inp", "r") as fp:
            dihedrals_read = False
            out_gomc = fp.readlines()
            for i, line in enumerate(out_gomc):
                if (
                    "! type_1" in line
                    and "type_2" in line
                    and "type_3" in line
                    and "type_4" in line
                    and "Kchi" in line
                    and "n" in line
                    and "delta" in line
                    and "extended_type_1" in line
                    and "extended_type_2" in line
                    and "extended_type_3" in line
                    and "extended_type_4" in line
                ):
                    dihedrals_read = True
                    dihedral_types = [
                        ["CH3", "CH", "O", "H", "-18.75", "0", "90.0"],
                        ["CH3", "CH", "O", "H", "10.0", "1", "180.0"],
                        ["CH3", "CH", "O", "H", "-10.0", "2", "0.0"],
                        ["CH3", "CH", "O", "H", "10.0", "3", "180.0"],
                        ["CH3", "CH", "O", "H", "-0.625", "4", "0.0"],
                    ]
                    for j in range(0, len(dihedral_types)):
                        assert (
                            len(out_gomc[i + 1 + j].split("!")[0].split()) == 7
                        )
                        assert (
                            out_gomc[i + 1 + j].split("!")[0].split()[0:7]
                            == dihedral_types[j]
                        )

                else:
                    pass

        assert dihedrals_read

    def test_save_charmm_mie_ua_K_energy_units_periodic_ff_with_nb_half(
        self, water, two_propanol_ua
    ):
        box_0 = mb.fill_box(
            compound=[water, two_propanol_ua], n_compounds=[4, 4], box=[5, 4, 3]
        )

        charmm = Charmm(
            box_0,
            "charmm_mie_ua_K_energy_units_periodic_with_nb_half",
            ff_filename="charmm_mie_ua_K_energy_units_periodic_with_nb_half",
            residues=[water.name, two_propanol_ua.name],
            forcefield_selection={
                water.name: get_mosdef_gomc_fn(
                    "gmso_spce_water_half_for_nb__half_coul__lorentz_combining.xml"
                ),
                two_propanol_ua.name: get_mosdef_gomc_fn(
                    "gmso_two_propanol_Mie_half_for_nb__half_coul.xml"
                ),
            },
            bead_to_atom_name_dict={"_CH3": "C", "_CH2": "C", "_HC": "C"},
            gomc_fix_bonds_angles=[water.name],
            atom_type_naming_style="general",
        )
        charmm.write_inp()

        with open(
            "charmm_mie_ua_K_energy_units_periodic_with_nb_half.inp", "r"
        ) as fp:
            masses_read = False
            bonds_read = False
            angles_read = False
            dihedrals_read = False
            nonbondeds_read = False
            out_gomc = fp.readlines()
            for i, line in enumerate(out_gomc):
                if (
                    "! atom_types" in line
                    and "mass" in line
                    and "atomClass_ResidueName" in line
                ):
                    masses_read = True
                    atom_types_1 = [
                        ["*", "OW", "15.999"],
                        ["*", "HW", "1.008"],
                        ["*", "CH3", "15.035"],
                        ["*", "CH", "13.019"],
                        ["*", "O", "15.9994"],
                        ["*", "H", "1.008"],
                    ]
                    atom_types_2 = [
                        ["WAT_o_spce"],
                        ["WAT_h_spce"],
                        ["POL_CH3_sp3"],
                        ["POL_CH_O"],
                        ["POL_O"],
                        ["POL_H"],
                    ]

                    for j in range(0, len(atom_types_1)):
                        assert (
                            len(out_gomc[i + 1 + j].split("!")[0].split()) == 3
                        )
                        assert (
                            out_gomc[i + 1 + j].split("!")[0].split()[0:3]
                            == atom_types_1[j]
                        )
                        assert (
                            out_gomc[i + 1 + j].split()[4:5] == atom_types_2[j]
                        )

                elif (
                    "! type_1" in line
                    and "type_2" in line
                    and "Kb" in line
                    and "b0" in line
                    and "extended_type_1" in line
                    and "extended_type_2" in line
                ):
                    bonds_read = True
                    bond_types = [
                        ["OW", "HW", "999999999999", "1.0"],
                        ["CH3", "CH", "604267.5553", "1.54"],
                        ["CH", "O", "604267.5553", "1.43"],
                        ["O", "H", "604267.5553", "0.945"],
                    ]
                    total_bonds_evaluated = []
                    total_bonds_evaluated_reorg = []
                    for j in range(0, len(bond_types)):
                        assert (
                            len(out_gomc[i + 1 + j].split("!")[0].split()) == 4
                        )

                        if (
                            out_gomc[i + 1 + j].split("!")[0].split()[0:4]
                            == bond_types[0]
                            or bond_types[1]
                            or bond_types[2]
                        ):
                            total_bonds_evaluated.append(
                                out_gomc[i + 1 + j].split("!")[0].split()[0:4]
                            )
                    for k in range(0, len(bond_types)):
                        if bond_types[k] in total_bonds_evaluated:
                            total_bonds_evaluated_reorg.append(bond_types[k])
                    assert total_bonds_evaluated_reorg == bond_types

                elif (
                    "! type_1 " in line
                    and "type_2" in line
                    and "type_3" in line
                    and "Ktheta" in line
                    and "Theta0" in line
                    and "extended_type_1" in line
                    and "extended_type_2" in line
                    and "extended_type_3" in line
                ):
                    angles_read = True
                    angle_types = [
                        ["HW", "OW", "HW", "999999999999", "109.47"],
                        ["CH3", "CH", "O", "50400.0", "109.5"],
                        ["CH3", "CH", "CH3", "62500.0", "112.0"],
                        ["CH", "O", "H", "55400.0", "108.5"],
                    ]
                    total_angles_evaluated = []
                    total_angles_evaluated_reorg = []
                    for j in range(0, len(angle_types)):
                        assert (
                            len(out_gomc[i + 1 + j].split("!")[0].split()) == 5
                        )
                        if (
                            out_gomc[i + 1 + j].split("!")[0].split()[0:5]
                            == angle_types[0]
                            or angle_types[1]
                            or angle_types[2]
                        ):
                            total_angles_evaluated.append(
                                out_gomc[i + 1 + j].split("!")[0].split()[0:5]
                            )
                    for k in range(0, len(angle_types)):
                        if angle_types[k] in total_angles_evaluated:
                            total_angles_evaluated_reorg.append(angle_types[k])

                    assert total_angles_evaluated_reorg == angle_types

                elif (
                    "! type_1" in line
                    and "type_2" in line
                    and "type_3" in line
                    and "type_4" in line
                    and "Kchi" in line
                    and "n" in line
                    and "delta" in line
                    and "extended_type_1" in line
                    and "extended_type_2" in line
                    and "extended_type_3" in line
                    and "extended_type_4" in line
                ):
                    dihedrals_read = True
                    dihedral_types = [
                        ["CH3", "CH", "O", "H", "-18.75", "0", "90.0"],
                        ["CH3", "CH", "O", "H", "10.0", "1", "180.0"],
                        ["CH3", "CH", "O", "H", "-10.0", "2", "0.0"],
                        ["CH3", "CH", "O", "H", "10.0", "3", "180.0"],
                        ["CH3", "CH", "O", "H", "-0.625", "4", "0.0"],
                    ]
                    for j in range(0, len(dihedral_types)):
                        assert (
                            len(out_gomc[i + 1 + j].split("!")[0].split()) == 7
                        )
                        assert (
                            out_gomc[i + 1 + j].split("!")[0].split()[0:7]
                            == dihedral_types[j]
                        )

                elif (
                    "! type_1" in line
                    and "epsilon" in line
                    and "sigma" in line
                    and "n" in line
                    and "epsilon,1-4" in line
                    and "sigma,1-4" in line
                    and "n,1-4" in line
                    and "extended_type_1" in line
                    and "extended_type_2" in line
                ):
                    nonbondeds_read = True
                    nb_types = [
                        [
                            "OW",
                            "78.200368",
                            "3.16557",
                            "12.0",
                            "39.100184",
                            "3.16557",
                            "12.0",
                        ],
                        [
                            "HW",
                            "0.0",
                            "1.0",
                            "12.0",
                            "0.0",
                            "1.0",
                            "12.0",
                        ],
                        [
                            "CH3",
                            "98.0",
                            "3.75",
                            "11.0",
                            "49.0",
                            "3.75",
                            "11.0",
                        ],
                        [
                            "CH",
                            "10.0",
                            "4.68",
                            "12.0",
                            "5.0",
                            "4.68",
                            "12.0",
                        ],
                        [
                            "O",
                            "93.0",
                            "3.02",
                            "13.0",
                            "46.5",
                            "3.02",
                            "13.0",
                        ],
                        [
                            "H",
                            "0.0",
                            "0.0",
                            "14.0",
                            "0.0",
                            "0.0",
                            "14.0",
                        ],
                    ]

                    for j in range(0, len(nb_types)):
                        assert (
                            len(out_gomc[i + 1 + j].split("!")[0].split()) == 7
                        )
                        assert (
                            out_gomc[i + 1 + j].split("!")[0].split()[0:7]
                            == nb_types[j]
                        )

                else:
                    pass

        assert masses_read
        assert bonds_read
        assert angles_read
        assert dihedrals_read
        assert nonbondeds_read

    def test_charmm_pdb_fix_angle_bond_fix_atoms(
        self, ethane_gomc, ethanol_gomc
    ):
        test_box_ethane_ethanol = mb.fill_box(
            compound=[ethane_gomc, ethanol_gomc],
            n_compounds=[1, 1],
            box=[2.0, 2.0, 2.0],
        )
        charmm = Charmm(
            test_box_ethane_ethanol,
            "Test_fixes_angle_bond_atoms",
            ff_filename="Test_fixes_angle_bond_atoms",
            residues=[ethanol_gomc.name, ethane_gomc.name],
            forcefield_selection="oplsaa",
            fix_residue=[ethane_gomc.name],
            fix_residue_in_box=[ethanol_gomc.name],
            gomc_fix_bonds_angles=[ethane_gomc.name],
            atom_type_naming_style="general",
        )
        charmm.write_inp()
        charmm.write_pdb()

        with open("Test_fixes_angle_bond_atoms.inp", "r") as fp:
            masses_read = False
            bonds_read = False
            angles_read = False
            out_gomc = fp.readlines()
            for i, line in enumerate(out_gomc):
                if (
                    "! atom_types" in line
                    and "mass" in line
                    and "atomClass_ResidueName" in line
                ):
                    masses_read = True
                    mass_type_1 = [
                        ["*", "CT0", "12.011"],
                        ["*", "HC0", "1.008"],
                        ["*", "CT1", "12.011"],
                        ["*", "CT2", "12.011"],
                        ["*", "OH0", "15.9994"],
                        ["*", "HC1", "1.008"],
                        ["*", "HO0", "1.008"],
                    ]
                    mass_type_2 = [
                        ["ETH_opls_135"],
                        ["ETH_opls_140"],
                        ["ETO_opls_135"],
                        ["ETO_opls_157"],
                        ["ETO_opls_154"],
                        ["ETO_opls_140"],
                        ["ETO_opls_155"],
                    ]

                    for j in range(0, len(mass_type_1)):
                        assert (
                            len(out_gomc[i + 1 + j].split("!")[0].split()) == 3
                        )
                        assert (
                            out_gomc[i + 1 + j].split("!")[0].split()[0:3]
                            == mass_type_1[j]
                        )
                        assert (
                            out_gomc[i + 1 + j].split()[4:5] == mass_type_2[j]
                        )

                elif (
                    "! type_1" in line
                    and "type_2" in line
                    and "Kb" in line
                    and "b0" in line
                    and "extended_type_1" in line
                    and "extended_type_2" in line
                ):
                    bonds_read = True
                    bond_types = [
                        ["CT0", "CT0", "999999999999", "1.529"],
                        ["HC0", "CT0", "999999999999", "1.09"],
                        ["CT2", "CT1", "268.0", "1.529"],
                        ["HC1", "CT1", "340.0", "1.09"],
                        ["OH0", "CT2", "320.0", "1.41"],
                        ["HC1", "CT2", "340.0", "1.09"],
                        ["HO0", "OH0", "553.0", "0.945"],
                    ]
                    total_bonds_evaluated = []
                    total_fixed_bonds = []
                    for j in range(0, 7):
                        total_bonds_evaluated.append(
                            out_gomc[i + 1 + j].split("!")[0].split()[0:4]
                        )
                        if out_gomc[i + 1 + j].split("!")[0].split()[2:3] == [
                            "999999999999"
                        ]:
                            total_fixed_bonds.append(
                                out_gomc[i + 1 + j].split("!")[0].split()[0:4]
                            )
                    assert total_bonds_evaluated.sort() == bond_types.sort()
                    assert len(total_fixed_bonds) == 2

                elif (
                    "! type_1 " in line
                    and "type_2" in line
                    and "type_3" in line
                    and "Ktheta" in line
                    and "Theta0" in line
                    and "extended_type_1" in line
                    and "extended_type_2" in line
                    and "extended_type_3" in line
                ):
                    angles_read = True
                    fixed_angle_types = [
                        ["HC0", "CT0", "CT0", "999999999999", "110.7"],
                        ["HC0", "CT0", "HC0", "999999999999", "107.8"],
                    ]
                    total_angles_evaluated = []
                    total_fixed_angles = []
                    for j in range(0, 9):
                        if out_gomc[i + 1 + j].split("!")[0].split()[0:4] == (
                            fixed_angle_types[0] or fixed_angle_types[1]
                        ):
                            total_angles_evaluated.append(
                                out_gomc[i + 1 + j].split("!")[0].split()[0:4]
                            )
                        if out_gomc[i + 1 + j].split("!")[0].split()[3:4] == [
                            "999999999999"
                        ]:
                            total_fixed_angles.append(
                                out_gomc[i + 1 + j].split("!")[0].split()[0:4]
                            )
                    assert (
                        fixed_angle_types.sort()
                        == total_angles_evaluated.sort()
                    )
                    assert len(total_fixed_angles) == len(fixed_angle_types)

                else:
                    pass

        assert masses_read
        assert bonds_read
        assert angles_read

        with open("Test_fixes_angle_bond_atoms.pdb", "r") as fp:
            read_pdb_part_1 = False
            read_pdb_part_2 = False
            out_gomc = fp.readlines()
            for i, line in enumerate(out_gomc):
                if "CRYST1" in line:
                    read_pdb_part_1 = True
                    assert out_gomc[i].split()[0:7] == [
                        "CRYST1",
                        "20.000",
                        "20.000",
                        "20.000",
                        "90.00",
                        "90.00",
                        "90.00",
                    ]

                if "CRYST1" in line:
                    read_pdb_part_2 = True
                    atom_type_res_part_1_list = [
                        ["ATOM", "1", "C1", "ETH", "A", "1"],
                        ["ATOM", "2", "C2", "ETH", "A", "1"],
                        ["ATOM", "3", "H1", "ETH", "A", "1"],
                        ["ATOM", "4", "H2", "ETH", "A", "1"],
                        ["ATOM", "5", "H3", "ETH", "A", "1"],
                        ["ATOM", "6", "H4", "ETH", "A", "1"],
                        ["ATOM", "7", "H5", "ETH", "A", "1"],
                        ["ATOM", "8", "H6", "ETH", "A", "1"],
                        ["ATOM", "9", "C1", "ETO", "A", "2"],
                        ["ATOM", "10", "C2", "ETO", "A", "2"],
                        ["ATOM", "11", "O1", "ETO", "A", "2"],
                        ["ATOM", "12", "H1", "ETO", "A", "2"],
                        ["ATOM", "13", "H2", "ETO", "A", "2"],
                        ["ATOM", "14", "H3", "ETO", "A", "2"],
                        ["ATOM", "15", "H4", "ETO", "A", "2"],
                        ["ATOM", "16", "H5", "ETO", "A", "2"],
                        ["ATOM", "17", "H6", "ETO", "A", "2"],
                    ]
                    atom_type_res_part_2_list = [
                        ["0.00", "1.00", "C"],
                        ["0.00", "1.00", "C"],
                        ["0.00", "1.00", "H"],
                        ["0.00", "1.00", "H"],
                        ["0.00", "1.00", "H"],
                        ["0.00", "1.00", "H"],
                        ["0.00", "1.00", "H"],
                        ["0.00", "1.00", "H"],
                        ["0.00", "2.00", "C"],
                        ["0.00", "2.00", "C"],
                        ["0.00", "2.00", "O"],
                        ["0.00", "2.00", "H"],
                        ["0.00", "2.00", "H"],
                        ["0.00", "2.00", "H"],
                        ["0.00", "2.00", "H"],
                        ["0.00", "2.00", "H"],
                        ["0.00", "2.00", "H"],
                    ]

                    for j in range(0, len(atom_type_res_part_1_list)):
                        assert (
                            out_gomc[i + 1 + j].split()[0:6]
                            == atom_type_res_part_1_list[j]
                        )
                        assert (
                            out_gomc[i + 1 + j].split()[9:12]
                            == atom_type_res_part_2_list[j]
                        )

                else:
                    pass

        assert read_pdb_part_1
        assert read_pdb_part_2

    def test_charmm_pdb_set_residue_pdb_occupancy_to_1(
        self, ethane_gomc, ethanol_gomc
    ):
        test_box_ethane_ethanol = mb.fill_box(
            compound=[ethane_gomc, ethanol_gomc],
            n_compounds=[1, 1],
            box=[2.0, 2.0, 2.0],
        )
        charmm = Charmm(
            test_box_ethane_ethanol,
            "Test_set_residue_pdb_occupancy_to_1",
            ff_filename="Test_set_residue_pdb_occupancy_to_1",
            residues=[ethanol_gomc.name, ethane_gomc.name],
            forcefield_selection="oplsaa",
            fix_residue=None,
            fix_residue_in_box=None,
            set_residue_pdb_occupancy_to_1=[ethane_gomc.name],
            gomc_fix_bonds_angles=None,
            atom_type_naming_style="general",
        )
        charmm.write_pdb()

        with open("Test_set_residue_pdb_occupancy_to_1.pdb", "r") as fp:
            read_pdb_part_1 = False
            read_pdb_part_2 = False
            out_gomc = fp.readlines()
            for i, line in enumerate(out_gomc):
                if "CRYST1" in line:
                    read_pdb_part_1 = True
                    assert out_gomc[i].split()[0:7] == [
                        "CRYST1",
                        "20.000",
                        "20.000",
                        "20.000",
                        "90.00",
                        "90.00",
                        "90.00",
                    ]

                if "CRYST1" in line:
                    read_pdb_part_2 = True
                    atom_type_res_part_1_list = [
                        ["ATOM", "1", "C1", "ETH", "A", "1"],
                        ["ATOM", "2", "C2", "ETH", "A", "1"],
                        ["ATOM", "3", "H1", "ETH", "A", "1"],
                        ["ATOM", "4", "H2", "ETH", "A", "1"],
                        ["ATOM", "5", "H3", "ETH", "A", "1"],
                        ["ATOM", "6", "H4", "ETH", "A", "1"],
                        ["ATOM", "7", "H5", "ETH", "A", "1"],
                        ["ATOM", "8", "H6", "ETH", "A", "1"],
                        ["ATOM", "9", "C1", "ETO", "A", "2"],
                        ["ATOM", "10", "C2", "ETO", "A", "2"],
                        ["ATOM", "11", "O1", "ETO", "A", "2"],
                        ["ATOM", "12", "H1", "ETO", "A", "2"],
                        ["ATOM", "13", "H2", "ETO", "A", "2"],
                        ["ATOM", "14", "H3", "ETO", "A", "2"],
                        ["ATOM", "15", "H4", "ETO", "A", "2"],
                        ["ATOM", "16", "H5", "ETO", "A", "2"],
                        ["ATOM", "17", "H6", "ETO", "A", "2"],
                    ]
                    atom_type_res_part_2_list = [
                        ["1.00", "0.00", "C"],
                        ["1.00", "0.00", "C"],
                        ["1.00", "0.00", "H"],
                        ["1.00", "0.00", "H"],
                        ["1.00", "0.00", "H"],
                        ["1.00", "0.00", "H"],
                        ["1.00", "0.00", "H"],
                        ["1.00", "0.00", "H"],
                        ["0.00", "0.00", "C"],
                        ["0.00", "0.00", "C"],
                        ["0.00", "0.00", "O"],
                        ["0.00", "0.00", "H"],
                        ["0.00", "0.00", "H"],
                        ["0.00", "0.00", "H"],
                        ["0.00", "0.00", "H"],
                        ["0.00", "0.00", "H"],
                        ["0.00", "0.00", "H"],
                    ]

                    for j in range(0, len(atom_type_res_part_1_list)):
                        assert (
                            out_gomc[i + 1 + j].split()[0:6]
                            == atom_type_res_part_1_list[j]
                        )
                        assert (
                            out_gomc[i + 1 + j].split()[9:12]
                            == atom_type_res_part_2_list[j]
                        )

                else:
                    pass

        assert read_pdb_part_1
        assert read_pdb_part_2

    def test_charmm_pdb_fix_bonds_only(self, ethane_gomc, ethanol_gomc):
        test_box_ethane_ethanol = mb.fill_box(
            compound=[ethane_gomc, ethanol_gomc],
            n_compounds=[1, 1],
            box=[2.0, 2.0, 2.0],
        )
        charmm = Charmm(
            test_box_ethane_ethanol,
            "Test_fixes_bonds_only",
            ff_filename="Test_fixes_bonds_only",
            residues=[ethanol_gomc.name, ethane_gomc.name],
            forcefield_selection="oplsaa",
            gomc_fix_bonds=[ethane_gomc.name],
            atom_type_naming_style="general",
        )
        charmm.write_inp()

        with open("Test_fixes_bonds_only.inp", "r") as fp:
            bonds_read = False
            angles_read = False
            out_gomc = fp.readlines()
            for i, line in enumerate(out_gomc):
                if (
                    "! type_1" in line
                    and "type_2" in line
                    and "Kb" in line
                    and "b0" in line
                    and "extended_type_1" in line
                    and "extended_type_2" in line
                ):
                    bonds_read = True
                    bond_types = [
                        ["CT0", "CT0", "999999999999", "1.529"],
                        ["CT0", "HC0", "999999999999", "1.09"],
                        ["CT1", "CT2", "268.0", "1.529"],
                        ["CT1", "HC1", "340.0", "1.09"],
                        ["CT2", "OH0", "320.0", "1.41"],
                        ["CT2", "HC1", "340.0", "1.09"],
                        ["HO0", "OH0", "553.0", "0.945"],
                    ]
                    total_bonds_evaluated = []
                    total_fixed_bonds = []
                    for j in range(0, 7):
                        total_bonds_evaluated.append(
                            out_gomc[i + 1 + j].split("!")[0].split()[0:4]
                        )
                        if out_gomc[i + 1 + j].split("!")[0].split()[2:3] == [
                            "999999999999"
                        ]:
                            total_fixed_bonds.append(
                                out_gomc[i + 1 + j].split("!")[0].split()[0:4]
                            )
                    assert total_bonds_evaluated.sort() == bond_types.sort()
                    assert len(total_fixed_bonds) == 2

                elif (
                    "! type_1 " in line
                    and "type_2" in line
                    and "type_3" in line
                    and "Ktheta" in line
                    and "Theta0" in line
                    and "extended_type_1" in line
                    and "extended_type_2" in line
                    and "extended_type_3" in line
                ):
                    angles_read = True
                    fixed_angle_types = []
                    total_angles_evaluated = []
                    total_fixed_angles = []
                    for j in range(0, 9):
                        if len(fixed_angle_types) > 0:
                            if out_gomc[i + 1 + j].split("!")[0].split()[
                                0:4
                            ] == (fixed_angle_types[0] or fixed_angle_types[1]):
                                total_angles_evaluated.append(
                                    out_gomc[i + 1 + j]
                                    .split("!")[0]
                                    .split()[0:4]
                                )
                        if out_gomc[i + 1 + j].split("!")[0].split()[3:4] == [
                            "999999999999"
                        ]:
                            total_fixed_angles.append(
                                out_gomc[i + 1 + j].split("!")[0].split()[0:4]
                            )
                    assert (
                        fixed_angle_types.sort()
                        == total_angles_evaluated.sort()
                    )
                    assert len(total_fixed_angles) == len(fixed_angle_types)

                else:
                    pass

        assert bonds_read
        assert angles_read

    def test_charmm_pdb_fix_bonds_only_and_fix_bonds_angles(
        self, ethane_gomc, ethanol_gomc
    ):
        test_box_ethane_ethanol = mb.fill_box(
            compound=[ethane_gomc, ethanol_gomc],
            n_compounds=[1, 1],
            box=[2.0, 2.0, 2.0],
        )
        charmm = Charmm(
            test_box_ethane_ethanol,
            "Test_fixes_bonds_only_and_fix_bonds_angles",
            ff_filename="Test_fixes_bonds_only_and_fix_bonds_angles",
            residues=[ethanol_gomc.name, ethane_gomc.name],
            forcefield_selection="oplsaa",
            gomc_fix_bonds=[ethane_gomc.name],
            gomc_fix_bonds_angles=[ethane_gomc.name],
            atom_type_naming_style="general",
        )
        charmm.write_inp()

        with open("Test_fixes_bonds_only_and_fix_bonds_angles.inp", "r") as fp:
            bonds_read = False
            angles_read = False
            out_gomc = fp.readlines()
            for i, line in enumerate(out_gomc):
                if (
                    "! type_1" in line
                    and "type_2" in line
                    and "Kb" in line
                    and "b0" in line
                    and "extended_type_1" in line
                    and "extended_type_2" in line
                ):
                    bonds_read = True
                    bond_types = [
                        ["CT0", "CT0", "999999999999", "1.529"],
                        ["CT0", "HC0", "999999999999", "1.09"],
                        ["CT1", "CT2", "268.0", "1.529"],
                        ["CT1", "HC1", "340.0", "1.09"],
                        ["CT2", "OH0", "320.0", "1.41"],
                        ["CT2", "HC1", "340.0", "1.09"],
                        ["HO0", "OH0", "553.0", "0.945"],
                    ]
                    total_bonds_evaluated = []
                    total_fixed_bonds = []
                    for j in range(0, 7):
                        total_bonds_evaluated.append(
                            out_gomc[i + 1 + j].split("!")[0].split()[0:4]
                        )
                        if out_gomc[i + 1 + j].split("!")[0].split()[2:3] == [
                            "999999999999"
                        ]:
                            total_fixed_bonds.append(
                                out_gomc[i + 1 + j].split("!")[0].split()[0:4]
                            )
                    assert total_bonds_evaluated.sort() == bond_types.sort()
                    assert len(total_fixed_bonds) == 2

                elif (
                    "! type_1 " in line
                    and "type_2" in line
                    and "type_3" in line
                    and "Ktheta" in line
                    and "Theta0" in line
                    and "extended_type_1" in line
                    and "extended_type_2" in line
                    and "extended_type_3" in line
                ):
                    angles_read = True
                    fixed_angle_types = [
                        ["HC0", "CT0", "HC0", "999999999999", "107.80000"],
                        ["CT0", "CT0", "HC0", "999999999999", "110.70000"],
                    ]
                    total_angles_evaluated = []
                    total_fixed_angles = []
                    for j in range(0, 9):
                        if out_gomc[i + 1 + j].split("!")[0].split()[0:4] == (
                            fixed_angle_types[0] or fixed_angle_types[1]
                        ):
                            total_angles_evaluated.append(
                                out_gomc[i + 1 + j].split("!")[0].split()[0:4]
                            )
                        if out_gomc[i + 1 + j].split("!")[0].split()[3:4] == [
                            "999999999999"
                        ]:
                            total_fixed_angles.append(
                                out_gomc[i + 1 + j].split("!")[0].split()[0:4]
                            )
                    assert (
                        fixed_angle_types.sort()
                        == total_angles_evaluated.sort()
                    )
                    assert len(total_fixed_angles) == len(fixed_angle_types)

                else:
                    pass

        assert bonds_read
        assert angles_read

    def test_charmm_pdb_fix_angles_only(self, ethane_gomc, ethanol_gomc):
        test_box_ethane_ethanol = mb.fill_box(
            compound=[ethane_gomc, ethanol_gomc],
            n_compounds=[1, 1],
            box=[2.0, 2.0, 2.0],
        )
        charmm = Charmm(
            test_box_ethane_ethanol,
            "Test_fixes_angles_only",
            ff_filename="Test_fixes_angles_only",
            residues=[ethanol_gomc.name, ethane_gomc.name],
            forcefield_selection="oplsaa",
            gomc_fix_angles=[ethane_gomc.name],
            atom_type_naming_style="general",
        )
        charmm.write_inp()

        with open("Test_fixes_angles_only.inp", "r") as fp:
            bonds_read = False
            angles_read = False
            out_gomc = fp.readlines()
            for i, line in enumerate(out_gomc):
                if (
                    "! type_1" in line
                    and "type_2" in line
                    and "Kb" in line
                    and "b0" in line
                    and "extended_type_1" in line
                    and "extended_type_2" in line
                ):
                    bonds_read = True
                    bond_types = [
                        ["CT0", "CT0", "268.0", "1.529"],
                        ["CT0", "HC0", "340.0", "1.09"],
                        ["CT1", "CT2", "268.0", "1.529"],
                        ["CT1", "HC1", "340.0", "1.09"],
                        ["CT2", "OH0", "320.0", "1.41"],
                        ["CT2", "HC1", "340.0", "1.09"],
                        ["HO0", "OH0", "553.0", "0.945"],
                    ]
                    total_bonds_evaluated = []
                    total_fixed_bonds = []
                    for j in range(0, 7):
                        total_bonds_evaluated.append(
                            out_gomc[i + 1 + j].split("!")[0].split()[0:4]
                        )
                        if out_gomc[i + 1 + j].split("!")[0].split()[2:3] == [
                            "999999999999"
                        ]:
                            total_fixed_bonds.append(
                                out_gomc[i + 1 + j].split("!")[0].split()[0:4]
                            )
                    assert total_bonds_evaluated.sort() == bond_types.sort()
                    assert len(total_fixed_bonds) == 0

                elif (
                    "! type_1 " in line
                    and "type_2" in line
                    and "type_3" in line
                    and "Ktheta" in line
                    and "Theta0" in line
                    and "extended_type_1" in line
                    and "extended_type_2" in line
                    and "extended_type_3" in line
                ):
                    angles_read = True
                    fixed_angle_types = [
                        ["HC0", "CT0", "HC0", "999999999999", "107.80000"],
                        ["CT0", "CT0", "HC0", "999999999999", "110.70000"],
                    ]
                    total_angles_evaluated = []
                    total_fixed_angles = []
                    for j in range(0, 9):
                        if out_gomc[i + 1 + j].split("!")[0].split()[0:4] == (
                            fixed_angle_types[0] or fixed_angle_types[1]
                        ):
                            total_angles_evaluated.append(
                                out_gomc[i + 1 + j].split("!")[0].split()[0:4]
                            )
                        if out_gomc[i + 1 + j].split("!")[0].split()[3:4] == [
                            "999999999999"
                        ]:
                            total_fixed_angles.append(
                                out_gomc[i + 1 + j].split("!")[0].split()[0:4]
                            )
                    assert (
                        fixed_angle_types.sort()
                        == total_angles_evaluated.sort()
                    )
                    assert len(total_fixed_angles) == len(fixed_angle_types)

                else:
                    pass

        assert bonds_read
        assert angles_read

    def test_charmm_pdb_fix_angles_only_and_fix_bonds_angles(
        self, ethane_gomc, ethanol_gomc
    ):
        test_box_ethane_ethanol = mb.fill_box(
            compound=[ethane_gomc, ethanol_gomc],
            n_compounds=[1, 1],
            box=[2.0, 2.0, 2.0],
        )
        charmm = Charmm(
            test_box_ethane_ethanol,
            "Test_fixes_angles_only_and_fix_bonds_angles",
            ff_filename="Test_fixes_angles_only_and_fix_bonds_angles",
            residues=[ethanol_gomc.name, ethane_gomc.name],
            forcefield_selection="oplsaa",
            gomc_fix_angles=[ethane_gomc.name],
            gomc_fix_bonds_angles=[ethane_gomc.name],
            atom_type_naming_style="general",
        )
        charmm.write_inp()

        with open("Test_fixes_angles_only_and_fix_bonds_angles.inp", "r") as fp:
            bonds_read = False
            angles_read = False
            out_gomc = fp.readlines()
            for i, line in enumerate(out_gomc):
                if (
                    "! type_1" in line
                    and "type_2" in line
                    and "Kb" in line
                    and "b0" in line
                    and "extended_type_1" in line
                    and "extended_type_2" in line
                ):
                    bonds_read = True
                    bond_types = [
                        ["CT0", "CT0", "999999999999", "1.529"],
                        ["CT0", "HC0", "999999999999", "1.09"],
                        ["CT1", "CT2", "268.0", "1.529"],
                        ["CT1", "HC1", "340.0", "1.09"],
                        ["CT2", "OH0", "320.0", "1.41"],
                        ["CT2", "HC1", "340.0", "1.09"],
                        ["HO0", "OH0", "553.0", "0.945"],
                    ]
                    total_bonds_evaluated = []
                    total_fixed_bonds = []
                    for j in range(0, 7):
                        total_bonds_evaluated.append(
                            out_gomc[i + 1 + j].split("!")[0].split()[0:4]
                        )
                        if out_gomc[i + 1 + j].split("!")[0].split()[2:3] == [
                            "999999999999"
                        ]:
                            total_fixed_bonds.append(
                                out_gomc[i + 1 + j].split("!")[0].split()[0:4]
                            )
                    assert total_bonds_evaluated.sort() == bond_types.sort()
                    assert len(total_fixed_bonds) == 2

                elif (
                    "! type_1 " in line
                    and "type_2" in line
                    and "type_3" in line
                    and "Ktheta" in line
                    and "Theta0" in line
                    and "extended_type_1" in line
                    and "extended_type_2" in line
                    and "extended_type_3" in line
                ):
                    angles_read = True
                    fixed_angle_types = [
                        ["HC0", "CT0", "HC0", "999999999999", "107.80000"],
                        ["CT0", "CT0", "HC0", "999999999999", "110.70000"],
                    ]
                    total_angles_evaluated = []
                    total_fixed_angles = []
                    for j in range(0, 9):
                        if out_gomc[i + 1 + j].split("!")[0].split()[0:4] == (
                            fixed_angle_types[0] or fixed_angle_types[1]
                        ):
                            total_angles_evaluated.append(
                                out_gomc[i + 1 + j].split("!")[0].split()[0:4]
                            )
                        if out_gomc[i + 1 + j].split("!")[0].split()[3:4] == [
                            "999999999999"
                        ]:
                            total_fixed_angles.append(
                                out_gomc[i + 1 + j].split("!")[0].split()[0:4]
                            )
                    assert (
                        fixed_angle_types.sort()
                        == total_angles_evaluated.sort()
                    )
                    assert len(total_fixed_angles) == len(fixed_angle_types)

                else:
                    pass

        assert bonds_read
        assert angles_read

    def test_charmm_pdb_diff_1_4_NB_and_1_4_coul_scalars(
        self, two_propanol_ua, water
    ):
        test_box_ethane_two_propanol_ua = mb.fill_box(
            compound=[two_propanol_ua, water],
            n_compounds=[1, 1],
            box=[2.0, 2.0, 2.0],
        )

        water.box = mb.Box(lengths=[4, 4, 4])

        with pytest.raises(
            ValueError,
            match=r"ERROR: There are multiple 1,4-electrostatic scaling factors "
            "GOMC will only accept a singular input for the 1,4-electrostatic "
            "scaling factors.",
        ):
            Charmm(
                test_box_ethane_two_propanol_ua,
                "residue_reorder_box_sizing_box_0",
                structure_box_1=water,
                filename_box_1="residue_reorder_box_sizing_box_1",
                ff_filename="residue_reorder_box",
                residues=[two_propanol_ua.name, water.name],
                forcefield_selection={
                    two_propanol_ua.name: "trappe-ua",
                    water.name: get_mosdef_gomc_fn(
                        "gmso_spce_water_one_for_nb_and_coul__lorentz_combining.xml"
                    ),
                },
                fix_residue=None,
                fix_residue_in_box=None,
                gomc_fix_bonds_angles=None,
                bead_to_atom_name_dict={"_CH3": "C"},
                atom_type_naming_style="general",
            )

    # test utils base 10 to base 16 converter
    def test_base_10_to_base_16(self):
        list_base_10_and_16 = [
            [15, "f"],
            [16, "10"],
            [17, "11"],
            [200, "c8"],
            [1000, "3e8"],
            [5000, "1388"],
            [int(16**3 - 1), "fff"],
            [int(16**3), "1000"],
        ]

        for test_base_16_iter in range(0, len(list_base_10_and_16)):
            test_10_iter = list_base_10_and_16[test_base_16_iter][0]
            test_16_iter = list_base_10_and_16[test_base_16_iter][1]
            assert str(base10_to_base16_alph_num(test_10_iter)) == str(
                test_16_iter
            )

        unique_entries_base_16_list = []
        for test_unique_base_16 in range(0, 16**2):
            unique_entries_base_16_list.append(
                base10_to_base16_alph_num(test_unique_base_16)
            )

        verified_unique_entries_base_16_list = np.unique(
            unique_entries_base_16_list
        )
        assert len(verified_unique_entries_base_16_list) == len(
            unique_entries_base_16_list
        )

        add_same_values_list = ["1", "a"]
        for add_same_base_16 in range(0, len(add_same_values_list)):
            verified_unique_entries_base_16_list = np.append(
                verified_unique_entries_base_16_list,
                add_same_values_list[add_same_base_16],
            )
        assert len(verified_unique_entries_base_16_list) - len(
            add_same_values_list
        ) == len(unique_entries_base_16_list)

    # test utils base 10 to base 26 converter
    def test_base_10_to_base_26(self):
        list_base_10_and_26 = [
            [0, "A"],
            [5, "F"],
            [25, "Z"],
            [26, "BA"],
            [200, "HS"],
            [1000, "BMM"],
            [5000, "HKI"],
            [int(26**3 - 1), "ZZZ"],
            [int(26**3), "BAAA"],
        ]

        for test_base_26_iter in range(0, len(list_base_10_and_26)):
            test_10_iter = list_base_10_and_26[test_base_26_iter][0]
            test_26_iter = list_base_10_and_26[test_base_26_iter][1]
            assert str(base10_to_base26_alph(test_10_iter)) == str(test_26_iter)

        unique_entries_base_26_list = []
        for test_unique_base_26 in range(0, 26**2):
            unique_entries_base_26_list.append(
                base10_to_base26_alph(test_unique_base_26)
            )

        verified_unique_entries_base_26_list = np.unique(
            unique_entries_base_26_list
        )
        assert len(verified_unique_entries_base_26_list) == len(
            unique_entries_base_26_list
        )

        add_same_values_list = ["1", "a"]
        for add_same_base_26 in range(0, len(add_same_values_list)):
            verified_unique_entries_base_26_list = np.append(
                verified_unique_entries_base_26_list,
                add_same_values_list[add_same_base_26],
            )
        assert len(verified_unique_entries_base_26_list) - len(
            add_same_values_list
        ) == len(unique_entries_base_26_list)

    # test utils base 10 to base 52 converter
    def test_base_10_to_base_52(self):
        list_base_10_and_52 = [
            [17, "R"],
            [51, "z"],
            [52, "BA"],
            [53, "BB"],
            [200, "Ds"],
            [1000, "TM"],
            [5000, "BsI"],
            [int(52**3 - 1), "zzz"],
            [int(52**3), "BAAA"],
        ]

        for test_base_52_iter in range(0, len(list_base_10_and_52)):
            test_10_iter = list_base_10_and_52[test_base_52_iter][0]
            test_52_iter = list_base_10_and_52[test_base_52_iter][1]
            assert str(base10_to_base52_alph(test_10_iter)) == str(test_52_iter)

        unique_entries_base_52_list = []
        for test_unique_base_52 in range(0, 52**2):
            unique_entries_base_52_list.append(
                base10_to_base52_alph(test_unique_base_52)
            )

        verified_unique_entries_base_52_list = np.unique(
            unique_entries_base_52_list
        )
        assert len(verified_unique_entries_base_52_list) == len(
            unique_entries_base_52_list
        )

        add_same_values_list = ["1", "a"]
        for add_same_base_52 in range(0, len(add_same_values_list)):
            verified_unique_entries_base_52_list = np.append(
                verified_unique_entries_base_52_list,
                add_same_values_list[add_same_base_52],
            )
        assert len(verified_unique_entries_base_52_list) - len(
            add_same_values_list
        ) == len(unique_entries_base_52_list)

    # test utils base 10 to base 62 converter
    def test_base_10_to_base_62(self):
        list_base_10_and_62 = [
            [17, "H"],
            [61, "z"],
            [62, "10"],
            [63, "11"],
            [200, "3E"],
            [1000, "G8"],
            [5000, "1Ie"],
            [int(62**3 - 1), "zzz"],
            [int(62**3), "1000"],
        ]

        for test_base_62_iter in range(0, len(list_base_10_and_62)):
            test_10_iter = list_base_10_and_62[test_base_62_iter][0]
            test_62_iter = list_base_10_and_62[test_base_62_iter][1]
            assert str(base10_to_base62_alph_num(test_10_iter)) == str(
                test_62_iter
            )

        unique_entries_base_62_list = []
        for test_unique_base_62 in range(0, 62**2):
            unique_entries_base_62_list.append(
                base10_to_base62_alph_num(test_unique_base_62)
            )

        verified_unique_entries_base_62_list = np.unique(
            unique_entries_base_62_list
        )
        assert len(verified_unique_entries_base_62_list) == len(
            unique_entries_base_62_list
        )

        add_same_values_list = ["1", "a"]
        for add_same_base_62 in range(0, len(add_same_values_list)):
            verified_unique_entries_base_62_list = np.append(
                verified_unique_entries_base_62_list,
                add_same_values_list[add_same_base_62],
            )
        assert len(verified_unique_entries_base_62_list) - len(
            add_same_values_list
        ) == len(unique_entries_base_62_list)

    # test utils base 10 to base 22 converter
    def test_base_10_to_base_22(self):
        list_base_10_and_22 = [
            [0, "A"],
            [5, "F"],
            [21, "V"],
            [22, "BA"],
            [200, "JC"],
            [1000, "CBK"],
            [5000, "KHG"],
            [int(22**3 - 1), "VVV"],
            [int(22**3), "BAAA"],
        ]

        for test_base_22_iter in range(0, len(list_base_10_and_22)):
            test_10_iter = list_base_10_and_22[test_base_22_iter][0]
            test_22_iter = list_base_10_and_22[test_base_22_iter][1]
            assert str(base10_to_base22_alph(test_10_iter)) == str(test_22_iter)

        unique_entries_base_22_list = []
        for test_unique_base_22 in range(0, 22**2):
            unique_entries_base_22_list.append(
                base10_to_base22_alph(test_unique_base_22)
            )

        verified_unique_entries_base_22_list = np.unique(
            unique_entries_base_22_list
        )
        assert len(verified_unique_entries_base_22_list) == len(
            unique_entries_base_22_list
        )

        add_same_values_list = ["1", "a"]
        for add_same_base_22 in range(0, len(add_same_values_list)):
            verified_unique_entries_base_22_list = np.append(
                verified_unique_entries_base_22_list,
                add_same_values_list[add_same_base_22],
            )
        assert len(verified_unique_entries_base_22_list) - len(
            add_same_values_list
        ) == len(unique_entries_base_22_list)

    # test utils base 10 to base 44 converter
    def test_base_10_to_base_44(self):
        list_base_10_and_44 = [
            [17, "R"],
            [43, "v"],
            [44, "BA"],
            [45, "BB"],
            [200, "Ec"],
            [1000, "ak"],
            [5000, "Cdg"],
            [int(44**3 - 1), "vvv"],
            [int(44**3), "BAAA"],
        ]

        for test_base_44_iter in range(0, len(list_base_10_and_44)):
            test_10_iter = list_base_10_and_44[test_base_44_iter][0]
            test_44_iter = list_base_10_and_44[test_base_44_iter][1]
            assert str(base10_to_base44_alph(test_10_iter)) == str(test_44_iter)

        unique_entries_base_44_list = []
        for test_unique_base_44 in range(0, 44**2):
            unique_entries_base_44_list.append(
                base10_to_base44_alph(test_unique_base_44)
            )

        verified_unique_entries_base_44_list = np.unique(
            unique_entries_base_44_list
        )
        assert len(verified_unique_entries_base_44_list) == len(
            unique_entries_base_44_list
        )

        add_same_values_list = ["1", "a"]
        for add_same_base_44 in range(0, len(add_same_values_list)):
            verified_unique_entries_base_44_list = np.append(
                verified_unique_entries_base_44_list,
                add_same_values_list[add_same_base_44],
            )
        assert len(verified_unique_entries_base_44_list) - len(
            add_same_values_list
        ) == len(unique_entries_base_44_list)

    # test utils base 10 to base 54 converter
    def test_base_10_to_base_54(self):
        list_base_10_and_54 = [
            [17, "H"],
            [53, "v"],
            [54, "10"],
            [55, "11"],
            [200, "3g"],
            [1000, "IS"],
            [5000, "1ga"],
            [int(54**3 - 1), "vvv"],
            [int(54**3), "1000"],
        ]

        for test_base_54_iter in range(0, len(list_base_10_and_54)):
            test_10_iter = list_base_10_and_54[test_base_54_iter][0]
            test_54_iter = list_base_10_and_54[test_base_54_iter][1]
            assert str(base10_to_base54_alph_num(test_10_iter)) == str(
                test_54_iter
            )

        unique_entries_base_54_list = []
        for test_unique_base_54 in range(0, 54**2):
            unique_entries_base_54_list.append(
                base10_to_base54_alph_num(test_unique_base_54)
            )

        verified_unique_entries_base_54_list = np.unique(
            unique_entries_base_54_list
        )
        assert len(verified_unique_entries_base_54_list) == len(
            unique_entries_base_54_list
        )

        add_same_values_list = ["1", "a"]
        for add_same_base_54 in range(0, len(add_same_values_list)):
            verified_unique_entries_base_54_list = np.append(
                verified_unique_entries_base_54_list,
                add_same_values_list[add_same_base_54],
            )
        assert len(verified_unique_entries_base_54_list) - len(
            add_same_values_list
        ) == len(unique_entries_base_54_list)

    def test_save_charmm_ua_single_bond_data_with_7_char_psf(self):
        with pytest.raises(
            ValueError,
            match=r"ERROR: The MOL_BDABCDE residue and mosdef atom name using the "
            r"MoSDeF_atom_class methodology exceeds the "
            r"character limit of 6, which is required for "
            r"the CHARMM style atom types. Please format the force field "
            r"xml files to get them under these 6 characters by allowing the "
            r"general MoSDeF_atom_class to be used or otherwise "
            r"shortening the atom MoSDeF_atom_class names. "
            f"NOTE: The {'MoSDeF_atom_class'} must allow for additional "
            f"alphanumberic additions at the end of it, making unique CHARMM atom "
            f"types \(MoSDeF_atom_classes\); typically this would allow for 4"
            f"characters in the MoSDeF_atom_classes.",
        ):
            molecule = mb.load(
                get_mosdef_gomc_fn("psf_file_single_bond_test_ua.mol2")
            )
            molecule.box = mb.Box(lengths=[4, 4, 4])
            molecule.name = "MOL"

            charmm = Charmm(
                molecule,
                "charmm_single_bond_data_UA",
                ff_filename="charmm_single_bond_data_UA",
                residues=[molecule.name],
                forcefield_selection=get_mosdef_gomc_fn(
                    "charmm_psf_7_char_atom_names_test_single_bonds_angles_dihedrals_impropers_ua.xml"
                ),
                bead_to_atom_name_dict={"_BD": "BD"},
                atom_type_naming_style="general",
            )

    def test_charmm_correct_residue_format(self, ethane_gomc):
        box_0 = mb.fill_box(
            compound=[ethane_gomc], n_compounds=[1], box=[4, 4, 4]
        )

        test_value = Charmm(
            box_0,
            "box_0",
            structure_box_1=None,
            filename_box_1=None,
            ff_filename=None,
            residues=[ethane_gomc.name],
            forcefield_selection={ethane_gomc.name: "oplsaa"},
            atom_type_naming_style="general",
        )

        assert test_value.input_error is False

    def test_charmm_residue_not_list(self, ethane_gomc):
        with pytest.raises(
            TypeError,
            match=r"ERROR: Please enter the residues list \(residues\) in a list format.",
        ):
            box_0 = mb.fill_box(
                compound=[ethane_gomc], n_compounds=[1], box=[4, 4, 4]
            )

            Charmm(
                box_0,
                "box_0",
                structure_box_1=None,
                filename_box_1=None,
                ff_filename=None,
                residues=ethane_gomc.name,
                forcefield_selection={ethane_gomc.name: "oplsaa"},
                atom_type_naming_style="general",
            )

    def test_charmm_residue_string(self, ethane_gomc):
        with pytest.raises(
            TypeError,
            match=r"ERROR: Please enter the residues list \(residues\) in a list format.",
        ):
            box_0 = mb.fill_box(
                compound=[ethane_gomc], n_compounds=[1], box=[4, 4, 4]
            )

            Charmm(
                box_0,
                "box_0",
                structure_box_1=None,
                filename_box_1=None,
                ff_filename=None,
                residues="ethane_gomc.name",
                forcefield_selection={ethane_gomc.name: "oplsaa"},
                atom_type_naming_style="general",
            )

    def test_charmm_residue_is_none(self, ethane_gomc):
        with pytest.raises(
            TypeError,
            match=r"ERROR: Please enter the residues list \(residues\)",
        ):
            box_0 = mb.fill_box(
                compound=[ethane_gomc], n_compounds=[1], box=[4, 4, 4]
            )

            Charmm(
                box_0,
                "box_0",
                structure_box_1=None,
                filename_box_1=None,
                ff_filename=None,
                residues=None,
                forcefield_selection={ethane_gomc.name: "oplsaa"},
                atom_type_naming_style="general",
            )

    def test_charmm_filename_0_is_not_string(self, ethane_gomc):
        with pytest.raises(
            TypeError,
            match=r"ERROR: Please enter the filename_box_0 as a string.",
        ):
            box_0 = mb.fill_box(
                compound=[ethane_gomc], n_compounds=[1], box=[4, 4, 4]
            )

            Charmm(
                box_0,
                0,
                structure_box_1=None,
                filename_box_1=None,
                ff_filename=None,
                residues=[ethane_gomc.name],
                forcefield_selection={ethane_gomc.name: "oplsaa"},
                atom_type_naming_style="general",
            )

    def test_charmm_filename_box_1_is_not_string(self, ethane_gomc):
        with pytest.raises(
            TypeError,
            match=r"ERROR: Please enter the filename_box_1 as a string.",
        ):
            box_0 = mb.fill_box(
                compound=[ethane_gomc], n_compounds=[1], box=[4, 4, 4]
            )

            Charmm(
                box_0,
                "box_0",
                structure_box_1=ethane_gomc,
                filename_box_1=["box_0"],
                ff_filename=None,
                residues=[ethane_gomc.name],
                forcefield_selection={ethane_gomc.name: "oplsaa"},
                atom_type_naming_style="general",
            )

    def test_charmm_gomc_filename_not_string(self, ethane_gomc):
        with pytest.raises(
            TypeError,
            match=r"ERROR: Please enter GOMC force field name \(ff_filename\) as a string.",
        ):
            box_0 = mb.fill_box(
                compound=[ethane_gomc], n_compounds=[1], box=[4, 4, 4]
            )

            Charmm(
                box_0,
                "box_0",
                structure_box_1=None,
                filename_box_1=None,
                ff_filename=0,
                residues=[ethane_gomc.name],
                forcefield_selection={ethane_gomc.name: "oplsaa"},
                atom_type_naming_style="general",
            )

    def test_charmm_gomc_filename_ext_not_dot_inp(self, ethane_gomc):
        with pytest.raises(
            ValueError,
            match=r"ERROR: Please enter GOMC force field name without an "
            "extention or the .inp extension.",
        ):
            box_0 = mb.fill_box(
                compound=[ethane_gomc], n_compounds=[1], box=[4, 4, 4]
            )

            Charmm(
                box_0,
                "box_0",
                structure_box_1=None,
                filename_box_1=None,
                ff_filename="box.test",
                residues=[ethane_gomc.name],
                forcefield_selection={ethane_gomc.name: "oplsaa"},
                atom_type_naming_style="general",
            )

    def test_charmm_ffselection_not_dict(self, ethane_gomc):
        with pytest.raises(
            TypeError,
            match=r"ERROR: The force field selection \(forcefield_selection\) "
            "is not a string or a dictionary with all the residues specified "
            'to a force field. -> String Ex: "path/trappe-ua.xml" or Ex: "trappe-ua" '
            "Otherise provided a dictionary with all the residues specified "
            "to a force field "
            '->Dictionary Ex: {"Water": "oplsaa", "OCT": "path/trappe-ua.xml"}, '
            "Note: the file path must be specified the force field file if "
            "a standard foyer force field is not used.",
        ):
            box_0 = mb.fill_box(
                compound=[ethane_gomc], n_compounds=[1], box=[4, 4, 4]
            )

            Charmm(
                box_0,
                "box_0",
                structure_box_1=None,
                filename_box_1=None,
                ff_filename="box_0",
                residues=[ethane_gomc.name],
                forcefield_selection=["oplsaa", "oplsaa"],
                atom_type_naming_style="general",
            )

    def test_charmm_ffselection_string(self, ethane_gomc):
        box_0 = mb.fill_box(
            compound=[ethane_gomc], n_compounds=[1], box=[4, 4, 4]
        )

        test_value = Charmm(
            box_0,
            "box_0",
            structure_box_1=None,
            filename_box_1=None,
            ff_filename="box_0",
            residues=[ethane_gomc.name],
            forcefield_selection="oplsaa",
            atom_type_naming_style="general",
        )

        assert test_value.input_error is False

    def test_charmm_residue_name_not_in_residues(self, ethane_gomc):
        with pytest.raises(
            GMSOError,
            match=f"A particle named C cannot be associated with the\n        "
            f"custom_groups \['XXX'\]. "
            f"Be sure to specify a list of group names that will cover\n        "
            f"all particles in the compound. This particle is one level below ETH.",
        ):
            box_0 = mb.fill_box(
                compound=[ethane_gomc], n_compounds=[1], box=[4, 4, 4]
            )

            Charmm(
                box_0,
                "box_0",
                structure_box_1=None,
                filename_box_1=None,
                ff_filename="box_0",
                residues=["XXX"],
                forcefield_selection="oplsaa",
                atom_type_naming_style="general",
            )

    def test_ffselection_string(self, two_propanol_ua):
        box_0 = mb.fill_box(
            compound=[two_propanol_ua], n_compounds=[1], box=[4, 4, 4]
        )

        charmm = Charmm(
            box_0,
            "ffselection_string",
            ff_filename="ffselection_string",
            residues=[two_propanol_ua.name],
            forcefield_selection=f"{forcefields.get_ff_path()[0]}/xml/trappe-ua.xml",
            bead_to_atom_name_dict={"_CH3": "C"},
            atom_type_naming_style="general",
        )
        charmm.write_pdb()

        with open("ffselection_string.pdb", "r") as fp:
            pdb_read = False
            out_gomc = fp.readlines()
            for i, line in enumerate(out_gomc):
                if "CRYST1" in line:
                    pdb_read = True
                    atom_type_res_part_1_list = [
                        ["ATOM", "1", "C1", "POL", "A", "1"],
                        ["ATOM", "2", "BD1", "POL", "A", "1"],
                        ["ATOM", "3", "O1", "POL", "A", "1"],
                        ["ATOM", "4", "H1", "POL", "A", "1"],
                        ["ATOM", "5", "C2", "POL", "A", "1"],
                    ]
                    atom_type_res_part_2_list = [
                        ["0.00", "0.00", "BD"],
                        ["0.00", "0.00", "BD"],
                        ["0.00", "0.00", "O"],
                        ["0.00", "0.00", "H"],
                        ["0.00", "0.00", "BD"],
                    ]

                    for j in range(0, len(atom_type_res_part_1_list)):
                        assert (
                            out_gomc[i + 1 + j].split()[0:6]
                            == atom_type_res_part_1_list[j]
                        )
                        assert (
                            out_gomc[i + 1 + j].split()[9:12]
                            == atom_type_res_part_2_list[j]
                        )

                else:
                    pass

        assert pdb_read

    def test_ff_selection_list(self, two_propanol_ua):
        with pytest.raises(
            TypeError,
            match=r"ERROR: The force field selection \(forcefield_selection\) "
            "is not a string or a dictionary with all the residues specified "
            'to a force field. -> String Ex: "path/trappe-ua.xml" or Ex: "trappe-ua" '
            "Otherise provided a dictionary with all the residues specified "
            "to a force field "
            '->Dictionary Ex: {"Water": "oplsaa", "OCT": "path/trappe-ua.xml"}, '
            "Note: the file path must be specified the force field file if "
            "a standard foyer force field is not used.",
        ):
            box_0 = mb.fill_box(
                compound=[two_propanol_ua], n_compounds=[1], box=[4, 4, 4]
            )

            Charmm(
                box_0,
                "S",
                ff_filename="S",
                residues=[two_propanol_ua.name],
                forcefield_selection=[
                    f"{str(forcefields.get_ff_path()[0])}/xml/trappe-ua.xml"
                ],
                bead_to_atom_name_dict={"_CH3": "C"},
                atom_type_naming_style="general",
            )

    def test_residues_not_a_string(self, two_propanol_ua):
        with pytest.raises(
            TypeError,
            match=r"ERROR: Please enter a residues list "
            r"\(residues\) with only string values.",
        ):
            box_0 = mb.fill_box(
                compound=[two_propanol_ua], n_compounds=[1], box=[4, 4, 4]
            )

            Charmm(
                box_0,
                "box_0",
                ff_filename="box_0",
                residues=[2],
                forcefield_selection={two_propanol_ua.name: "trappe-ua"},
                bead_to_atom_name_dict={"_CH3": "C"},
                atom_type_naming_style="general",
            )

    def test_bead_atomname_equal_3(self, two_propanol_ua):
        # testing def unique_atom_naming in charmm_writer, expecting when failing
        with pytest.raises(
            ValueError,
            match=r"ERROR: The unique_atom_naming function failed while "
            "running the charmm_writer function. Ensure the proper inputs are "
            "in the bead_to_atom_name_dict.",
        ):
            box_reservior_0 = mb.fill_box(
                compound=[two_propanol_ua], box=[10, 10, 10], n_compounds=[10]
            )

            value_0 = Charmm(
                box_reservior_0,
                "test_bead_atomname_equal_3",
                ff_filename="test_bead_atomname_equal_3",
                residues=[two_propanol_ua.name],
                forcefield_selection="trappe-ua",
                bead_to_atom_name_dict={"_CH3": "Cx", "_HC": "Cxx"},
                atom_type_naming_style="general",
            )
            value_0.write_inp()
            value_0.write_pdb()
            value_0.write_psf()

    def test_gomc_fix_bonds_angles_string(self, two_propanol_ua):
        with pytest.raises(
            TypeError,
            match=r"ERROR: Please ensure the residue names in the \({}\) variable "
            r"are in a list.".format("gomc_fix_bonds_angles"),
        ):
            box_0 = mb.fill_box(
                compound=[two_propanol_ua], n_compounds=[1], box=[4, 4, 4]
            )

            Charmm(
                box_0,
                "charmm_data_UA",
                ff_filename="charmm_data_UA",
                residues=[two_propanol_ua.name],
                forcefield_selection="trappe-ua",
                bead_to_atom_name_dict={"_CH3": "C"},
                gomc_fix_bonds_angles="two_propanol_ua.name",
            )

    def test_gomc_fix_bonds_angles_residue_not_in_system(self, two_propanol_ua):
        with pytest.raises(
            ValueError,
            match=r"ERROR: Please ensure that all the residue names in the "
            r"{} list are also in the residues list.".format(
                "gomc_fix_bonds_angles"
            ),
        ):
            box_0 = mb.fill_box(
                compound=[two_propanol_ua], n_compounds=[1], box=[4, 4, 4]
            )

            Charmm(
                box_0,
                "charmm_data_UA",
                ff_filename="charmm_data_UA",
                residues=[two_propanol_ua.name],
                forcefield_selection="trappe-ua",
                bead_to_atom_name_dict={"_CH3": "C"},
                gomc_fix_bonds_angles=["WNG"],
            )

    def test_gomc_fix_bonds_string(self, two_propanol_ua):
        with pytest.raises(
            TypeError,
            match=r"ERROR: Please ensure the residue names in the \({}\) variable "
            r"are in a list.".format("gomc_fix_bonds"),
        ):
            box_0 = mb.fill_box(
                compound=[two_propanol_ua], n_compounds=[1], box=[4, 4, 4]
            )

            Charmm(
                box_0,
                "charmm_data_UA",
                ff_filename="charmm_data_UA",
                residues=[two_propanol_ua.name],
                forcefield_selection="trappe-ua",
                bead_to_atom_name_dict={"_CH3": "C"},
                gomc_fix_bonds="two_propanol_ua.name",
            )

    def test_gomc_fix_bonds_residue_not_in_system(self, two_propanol_ua):
        with pytest.raises(
            ValueError,
            match=r"ERROR: Please ensure that all the residue names in the "
            r"{} list are also in the residues list.".format("gomc_fix_bonds"),
        ):
            box_0 = mb.fill_box(
                compound=[two_propanol_ua], n_compounds=[1], box=[4, 4, 4]
            )

            Charmm(
                box_0,
                "charmm_data_UA",
                ff_filename="charmm_data_UA",
                residues=[two_propanol_ua.name],
                forcefield_selection="trappe-ua",
                bead_to_atom_name_dict={"_CH3": "C"},
                gomc_fix_bonds=["WNG"],
            )

    def test_gomc_fix_angles_string(self, two_propanol_ua):
        with pytest.raises(
            TypeError,
            match=r"ERROR: Please ensure the residue names in the \({}\) variable "
            r"are in a list.".format("gomc_fix_angles"),
        ):
            box_0 = mb.fill_box(
                compound=[two_propanol_ua], n_compounds=[1], box=[4, 4, 4]
            )

            Charmm(
                box_0,
                "charmm_data_UA",
                ff_filename="charmm_data_UA",
                residues=[two_propanol_ua.name],
                forcefield_selection="trappe-ua",
                bead_to_atom_name_dict={"_CH3": "C"},
                gomc_fix_angles="two_propanol_ua.name",
            )

    def test_gomc_fix_angles_residue_not_in_system(self, two_propanol_ua):
        with pytest.raises(
            ValueError,
            match=r"ERROR: Please ensure that all the residue names in the "
            r"{} list are also in the residues list.".format("gomc_fix_angles"),
        ):
            box_0 = mb.fill_box(
                compound=[two_propanol_ua], n_compounds=[1], box=[4, 4, 4]
            )

            Charmm(
                box_0,
                "charmm_data_UA",
                ff_filename="charmm_data_UA",
                residues=[two_propanol_ua.name],
                forcefield_selection="trappe-ua",
                bead_to_atom_name_dict={"_CH3": "C"},
                gomc_fix_angles=["WNG"],
            )

    def test_fix_residue_string(self, two_propanol_ua):
        with pytest.raises(
            TypeError,
            match=r"ERROR: Please enter the fix_residue in a list format",
        ):
            box_0 = mb.fill_box(
                compound=[two_propanol_ua], n_compounds=[1], box=[4, 4, 4]
            )

            Charmm(
                box_0,
                "charmm_data_UA",
                ff_filename="charmm_data_UA",
                residues=[two_propanol_ua.name],
                forcefield_selection="trappe-ua",
                bead_to_atom_name_dict={"_CH3": "C"},
                fix_residue="two_propanol_ua.name",
            )

    def test_fix_residue_string_residue_not_in_system(self, two_propanol_ua):
        with pytest.raises(
            ValueError,
            match=r"Error: Please ensure that all the residue names in the fix_residue "
            r"list are also in the residues list.",
        ):
            box_0 = mb.fill_box(
                compound=[two_propanol_ua], n_compounds=[1], box=[4, 4, 4]
            )

            Charmm(
                box_0,
                "charmm_data_UA",
                ff_filename="charmm_data_UA",
                residues=[two_propanol_ua.name],
                forcefield_selection="trappe-ua",
                bead_to_atom_name_dict={"_CH3": "C"},
                fix_residue=["WNG"],
            )

    def test_fix_residue_in_box_string(self, two_propanol_ua):
        with pytest.raises(
            TypeError,
            match=r"ERROR: Please enter the fix_residue_in_box in a list format.",
        ):
            box_0 = mb.fill_box(
                compound=[two_propanol_ua], n_compounds=[1], box=[4, 4, 4]
            )

            Charmm(
                box_0,
                "charmm_data_UA",
                ff_filename="charmm_data_UA",
                residues=[two_propanol_ua.name],
                forcefield_selection="trappe-ua",
                bead_to_atom_name_dict={"_CH3": "C"},
                fix_residue_in_box="two_propanol_ua.name",
            )

    def test_fix_residue_in_box_string_residue_not_in_system(
        self, two_propanol_ua
    ):
        with pytest.raises(
            ValueError,
            match=r"Error: Please ensure that all the residue names in the "
            r"fix_residue_in_box list are also in the residues list.",
        ):
            box_0 = mb.fill_box(
                compound=[two_propanol_ua], n_compounds=[1], box=[4, 4, 4]
            )

            Charmm(
                box_0,
                "charmm_data_UA",
                ff_filename="charmm_data_UA",
                residues=[two_propanol_ua.name],
                forcefield_selection="trappe-ua",
                bead_to_atom_name_dict={"_CH3": "C"},
                fix_residue_in_box=["WNG"],
            )

    def test_set_residue_pdb_occupancy_to_1_string(self, two_propanol_ua):
        with pytest.raises(
            TypeError,
            match=r"ERROR: Please enter the set_residue_pdb_occupancy_to_1 in a list format.",
        ):
            box_0 = mb.fill_box(
                compound=[two_propanol_ua], n_compounds=[1], box=[4, 4, 4]
            )

            Charmm(
                box_0,
                "charmm_data_UA",
                ff_filename="charmm_data_UA",
                residues=[two_propanol_ua.name],
                forcefield_selection="trappe-ua",
                bead_to_atom_name_dict={"_CH3": "C"},
                set_residue_pdb_occupancy_to_1="two_propanol_ua.name",
            )

    def test_set_residue_pdb_occupancy_to_1_string_residue_not_in_system(
        self, two_propanol_ua
    ):
        with pytest.raises(
            ValueError,
            match=r"Error: Please ensure that all the residue names in the "
            r"set_residue_pdb_occupancy_to_1 list are also in the residues list.",
        ):
            box_0 = mb.fill_box(
                compound=[two_propanol_ua], n_compounds=[1], box=[4, 4, 4]
            )

            Charmm(
                box_0,
                "charmm_data_UA",
                ff_filename="charmm_data_UA",
                residues=[two_propanol_ua.name],
                forcefield_selection="trappe-ua",
                bead_to_atom_name_dict={"_CH3": "C"},
                set_residue_pdb_occupancy_to_1=[1],  # ["WNG"],
            )

    def test_bead_to_atom_name_dict_list(self, two_propanol_ua):
        with pytest.raises(
            TypeError,
            match=r"ERROR: Please enter the a bead type to atom in the dictionary "
            r"\(bead_to_atom_name_dict\) so GOMC can properly evaluate the "
            r"unique atom names",
        ):
            box_0 = mb.fill_box(
                compound=[two_propanol_ua], n_compounds=[1], box=[4, 4, 4]
            )

            Charmm(
                box_0,
                "charmm_data_UA",
                ff_filename="charmm_data_UA",
                residues=[two_propanol_ua.name],
                forcefield_selection="trappe-ua",
                bead_to_atom_name_dict=["_CH3", "C"],
            )

    def test_bead_to_atom_name_dict_not_string_0(self, two_propanol_ua):
        with pytest.raises(
            TypeError,
            match=r"ERROR: Please enter the bead_to_atom_name_dict with only "
            r"string inputs.",
        ):
            box_0 = mb.fill_box(
                compound=[two_propanol_ua], n_compounds=[1], box=[4, 4, 4]
            )

            Charmm(
                box_0,
                "charmm_data_UA",
                ff_filename="charmm_data_UA",
                residues=[two_propanol_ua.name],
                forcefield_selection="trappe-ua",
                bead_to_atom_name_dict={"_CH3": 0},
            )

    def test_bead_to_atom_name_dict_not_string_1(self, two_propanol_ua):
        with pytest.raises(
            TypeError,
            match=r"ERROR: Please enter the bead_to_atom_name_dict with only "
            r"string inputs.",
        ):
            box_0 = mb.fill_box(
                compound=[two_propanol_ua], n_compounds=[1], box=[4, 4, 4]
            )

            Charmm(
                box_0,
                "charmm_data_UA",
                ff_filename="charmm_data_UA",
                residues=[two_propanol_ua.name],
                forcefield_selection="trappe-ua",
                bead_to_atom_name_dict={0: "C"},
            )

    def test_1_box_residues_not_all_listed_box_0(
        self, ethane_gomc, ethanol_gomc
    ):
        with pytest.raises(
            GMSOError,
            match=f"A particle named C cannot be associated with the\n        "
            f"custom_groups \['ETO'\]. "
            f"Be sure to specify a list of group names that will cover\n        "
            f"all particles in the compound. This particle is one level below ETH.",
        ):
            box_0 = mb.fill_box(
                compound=[ethane_gomc], n_compounds=[1], box=[4, 4, 4]
            )

            Charmm(
                box_0,
                "charmm_data_box_0",
                structure_box_1=None,
                filename_box_1=None,
                ff_filename="charmm_data",
                residues=[ethanol_gomc.name],
                forcefield_selection="oplsaa",
            )

    def test_2_box_residues_not_all_listed_box_0(
        self, ethane_gomc, ethanol_gomc
    ):
        with pytest.raises(
            GMSOError,
            match=f"A particle named C cannot be associated with the\n        "
            f"custom_groups \['XXX', 'ETO'\]. "
            f"Be sure to specify a list of group names that will cover\n        "
            f"all particles in the compound. This particle is one level below ETH.",
        ):
            box_0 = mb.fill_box(
                compound=[ethane_gomc], n_compounds=[1], box=[4, 4, 4]
            )

            box_1 = mb.fill_box(
                compound=[ethanol_gomc], n_compounds=[1], box=[4, 4, 4]
            )

            Charmm(
                box_0,
                "charmm_data_box_0",
                structure_box_1=box_1,
                filename_box_1="charmm_data_box_1",
                ff_filename="charmm_data",
                residues=["XXX", ethanol_gomc.name],
                forcefield_selection="oplsaa",
            )

    def test_2_box_residues_not_all_listed_box_1(
        self, ethane_gomc, ethanol_gomc
    ):
        with pytest.raises(
            GMSOError,
            match=f"A particle named C cannot be associated with the\n        "
            f"custom_groups \['XXX', 'ETH'\]. "
            f"Be sure to specify a list of group names that will cover\n        "
            f"all particles in the compound. This particle is one level below ETO.",
        ):
            box_0 = mb.fill_box(
                compound=[ethane_gomc], n_compounds=[1], box=[4, 4, 4]
            )

            box_1 = mb.fill_box(
                compound=[ethanol_gomc], n_compounds=[1], box=[4, 4, 4]
            )

            Charmm(
                box_0,
                "charmm_data_box_0",
                structure_box_1=box_1,
                filename_box_1="charmm_data_box_1",
                ff_filename="charmm_data",
                residues=["XXX", ethane_gomc.name],
                forcefield_selection="oplsaa",
            )

    def test_2_box_residues_listed_2x(self, ethane_gomc, ethanol_gomc):
        with pytest.raises(
            ValueError,
            match=r"ERROR: Please enter the residues list \(residues\) that has "
            r"only unique residue names.",
        ):
            box_0 = mb.fill_box(
                compound=[ethane_gomc], n_compounds=[1], box=[4, 4, 4]
            )

            box_1 = mb.fill_box(
                compound=[ethanol_gomc], n_compounds=[1], box=[4, 4, 4]
            )

            Charmm(
                box_0,
                "charmm_data_box_0",
                structure_box_1=box_1,
                filename_box_1="charmm_data_box_1",
                ff_filename="charmm_data",
                residues=[ethanol_gomc.name, ethanol_gomc.name],
                forcefield_selection="oplsaa",
            )

    def test_all_residues_are_listed(self, ethane_gomc, ethanol_gomc):
        with pytest.raises(
            GMSOError,
            match=f"A particle named C cannot be associated with the\n        "
            f"custom_groups \['ETO'\]. "
            f"Be sure to specify a list of group names that will cover\n        "
            f"all particles in the compound. This particle is one level below ETH.",
        ):
            box_0 = mb.fill_box(
                compound=[ethane_gomc], n_compounds=[1], box=[4, 4, 4]
            )

            box_1 = mb.fill_box(
                compound=[ethanol_gomc], n_compounds=[1], box=[4, 4, 4]
            )

            Charmm(
                box_0,
                "charmm_data_box_0",
                structure_box_1=box_1,
                filename_box_1="charmm_data_box_1",
                ff_filename="charmm_data",
                residues=[ethanol_gomc.name],
                forcefield_selection="oplsaa",
            )

    # Test that an empty box (psf and pdb files) can be created to start a simulation
    def test_box_1_empty_test_1(self, two_propanol_ua):
        empty_compound = Box(lengths=[2, 2, 2])

        box_0 = mb.fill_box(
            compound=[two_propanol_ua], n_compounds=[1], box=[4, 4, 4]
        )

        charmm = Charmm(
            box_0,
            "charmm_filled_box",
            structure_box_1=empty_compound,
            filename_box_1="charmm_empty_box",
            ff_filename="charmm_empty_box.inp",
            residues=[two_propanol_ua.name],
            forcefield_selection="trappe-ua",
            bead_to_atom_name_dict={"_CH3": "C"},
            atom_type_naming_style="general",
        )
        charmm.write_pdb()
        charmm.write_psf()

        with open("charmm_empty_box.pdb", "r") as fp:
            pdb_read = False
            out_gomc = fp.readlines()
            for i, line in enumerate(out_gomc):
                if "CRYST1" in line:
                    pdb_read = True
                    assert out_gomc[i].split()[0:7] == [
                        "CRYST1",
                        "20.000",
                        "20.000",
                        "20.000",
                        "90.00",
                        "90.00",
                        "90.00",
                    ]
                    assert out_gomc[i + 1].split() == ["END"]

                else:
                    pass

        assert pdb_read

        with open("charmm_filled_box.pdb", "r") as fp:
            pdb_read = False
            out_gomc = fp.readlines()
            for i, line in enumerate(out_gomc):
                if "CRYST1" in line:
                    pdb_read = True
                    atom_type_res_part_1_list = [
                        ["ATOM", "1", "C1", "POL", "A", "1"],
                        ["ATOM", "2", "BD1", "POL", "A", "1"],
                        ["ATOM", "3", "O1", "POL", "A", "1"],
                        ["ATOM", "4", "H1", "POL", "A", "1"],
                        ["ATOM", "5", "C2", "POL", "A", "1"],
                    ]
                    atom_type_res_part_2_list = [
                        ["0.00", "0.00", "BD"],
                        ["0.00", "0.00", "BD"],
                        ["0.00", "0.00", "O"],
                        ["0.00", "0.00", "H"],
                        ["0.00", "0.00", "BD"],
                    ]

                    for j in range(0, len(atom_type_res_part_1_list)):
                        assert (
                            out_gomc[i + 1 + j].split()[0:6]
                            == atom_type_res_part_1_list[j]
                        )
                        assert (
                            out_gomc[i + 1 + j].split()[9:12]
                            == atom_type_res_part_2_list[j]
                        )

                else:
                    pass

        assert pdb_read

    def test_box_1_empty_test_2(self, two_propanol_ua):
        empty_compound = Box(lengths=[3, 3, 3], angles=[90, 90, 90])

        box_0 = mb.fill_box(
            compound=[two_propanol_ua], n_compounds=[1], box=[4, 4, 4]
        )

        charmm = Charmm(
            box_0,
            "charmm_filled_box",
            structure_box_1=empty_compound,
            filename_box_1="charmm_empty_box",
            ff_filename="charmm_empty_box.inp",
            residues=[two_propanol_ua.name],
            forcefield_selection="trappe-ua",
            bead_to_atom_name_dict={"_CH3": "C"},
            atom_type_naming_style="general",
        )
        charmm.write_pdb()
        charmm.write_psf()

        with open("charmm_empty_box.pdb", "r") as fp:
            pdb_read = False
            out_gomc = fp.readlines()
            for i, line in enumerate(out_gomc):
                if "CRYST1" in line:
                    pdb_read = True
                    assert out_gomc[i].split()[0:7] == [
                        "CRYST1",
                        "30.000",
                        "30.000",
                        "30.000",
                        "90.00",
                        "90.00",
                        "90.00",
                    ]
                    assert out_gomc[i + 1].split() == ["END"]

                else:
                    pass

        assert pdb_read

        with open("charmm_filled_box.pdb", "r") as fp:
            pdb_read = False
            out_gomc = fp.readlines()
            for i, line in enumerate(out_gomc):
                if "CRYST1" in line:
                    pdb_read = True
                    atom_type_res_part_1_list = [
                        ["ATOM", "1", "C1", "POL", "A", "1"],
                        ["ATOM", "2", "BD1", "POL", "A", "1"],
                        ["ATOM", "3", "O1", "POL", "A", "1"],
                        ["ATOM", "4", "H1", "POL", "A", "1"],
                        ["ATOM", "5", "C2", "POL", "A", "1"],
                    ]
                    atom_type_res_part_2_list = [
                        ["0.00", "0.00", "BD"],
                        ["0.00", "0.00", "BD"],
                        ["0.00", "0.00", "O"],
                        ["0.00", "0.00", "H"],
                        ["0.00", "0.00", "BD"],
                    ]

                    for j in range(0, len(atom_type_res_part_1_list)):
                        assert (
                            out_gomc[i + 1 + j].split()[0:6]
                            == atom_type_res_part_1_list[j]
                        )
                        assert (
                            out_gomc[i + 1 + j].split()[9:12]
                            == atom_type_res_part_2_list[j]
                        )

                else:
                    pass

        assert pdb_read

    def test_box_1_empty_test_3(self, two_propanol_ua):
        empty_compound = Box(lengths=[4, 5, 6])

        test_box_two_propanol_ua_gomc = mb.fill_box(
            compound=[two_propanol_ua], n_compounds=[1], box=[3, 4, 5]
        )

        charmm = Charmm(
            empty_compound,
            "charmm_empty_box",
            structure_box_1=test_box_two_propanol_ua_gomc,
            filename_box_1="charmm_filled_box",
            ff_filename="charmm_empty_box",
            residues=[two_propanol_ua.name],
            forcefield_selection="trappe-ua",
            bead_to_atom_name_dict={"_CH3": "C"},
            atom_type_naming_style="general",
        )
        charmm.write_pdb()
        charmm.write_psf()

        with open("charmm_empty_box.pdb", "r") as fp:
            pdb_part_1_read = False
            out_gomc = fp.readlines()
            for i, line in enumerate(out_gomc):
                if "CRYST1" in line:
                    pdb_part_1_read = True
                    assert out_gomc[i].split()[0:7] == [
                        "CRYST1",
                        "40.000",
                        "50.000",
                        "60.000",
                        "90.00",
                        "90.00",
                        "90.00",
                    ]
                    assert out_gomc[i + 1].split() == ["END"]

                else:
                    pass

        assert pdb_part_1_read

        with open("charmm_filled_box.pdb", "r") as fp:
            pdb_part_2_read = False
            out_gomc = fp.readlines()
            for i, line in enumerate(out_gomc):
                if "CRYST1" in line:
                    pdb_part_2_read = True
                    atom_type_res_part_1_list = [
                        ["ATOM", "1", "C1", "POL", "A", "1"],
                        ["ATOM", "2", "BD1", "POL", "A", "1"],
                        ["ATOM", "3", "O1", "POL", "A", "1"],
                        ["ATOM", "4", "H1", "POL", "A", "1"],
                        ["ATOM", "5", "C2", "POL", "A", "1"],
                    ]
                    atom_type_res_part_2_list = [
                        ["0.00", "0.00", "BD"],
                        ["0.00", "0.00", "BD"],
                        ["0.00", "0.00", "O"],
                        ["0.00", "0.00", "H"],
                        ["0.00", "0.00", "BD"],
                    ]

                    for j in range(0, len(atom_type_res_part_1_list)):
                        assert (
                            out_gomc[i + 1 + j].split()[0:6]
                            == atom_type_res_part_1_list[j]
                        )
                        assert (
                            out_gomc[i + 1 + j].split()[9:12]
                            == atom_type_res_part_2_list[j]
                        )

                else:
                    pass

        assert pdb_part_2_read

    def test_box_1_empty_test_4(self):
        empty_compound_box_0 = Box(lengths=[2, 2, 2])
        empty_compound_box_1 = Box(lengths=[3, 3, 3])
        with pytest.raises(
            TypeError,
            match=r"ERROR: Both structure_box_0 and structure_box_0 are empty Boxes {}. "
            "At least 1 structure must be an mbuild compound {} with 1 "
            "or more atoms in it".format(
                type(Box(lengths=[1, 1, 1])), type(Compound())
            ),
        ):
            Charmm(
                empty_compound_box_0,
                "charmm_data_box_0",
                structure_box_1=empty_compound_box_1,
                filename_box_1="charmm_data_box_1",
                ff_filename="charmm_data",
                residues=[],
                forcefield_selection="oplsaa",
                atom_type_naming_style="general",
            )

    def test_box_1_empty_test_5(self):
        empty_compound_box_0 = Box(lengths=[2, 2, 2])
        with pytest.raises(
            TypeError,
            match=r"ERROR: Only 1 structure is provided and it can not be an empty "
            r"mbuild Box {}. "
            "it must be an mbuild compound {} with at least 1 "
            "or more atoms in it.".format(
                type(Box(lengths=[1, 1, 1])), type(Compound())
            ),
        ):
            Charmm(
                empty_compound_box_0,
                "charmm_data_box_0",
                structure_box_1=None,
                filename_box_1=None,
                ff_filename="charmm_data",
                residues=[],
                forcefield_selection="oplsaa",
                atom_type_naming_style="general",
            )

    def test_box_1_empty_test_6(self, two_propanol_ua):
        with pytest.raises(
            TypeError,
            match=r"ERROR: If you are not providing an empty box, "
            r"you need to specify the atoms/beads as children in the mb.Compound. "
            r"If you are providing and empty box, please do so by specifying and "
            r"mbuild Box \({}\)".format(type(Box(lengths=[1, 1, 1]))),
        ):
            test_box_two_propanol_ua_gomc = mb.fill_box(
                compound=[two_propanol_ua], n_compounds=[1], box=[3, 4, 5]
            )

            empty_compound = mb.Compound()
            empty_compound.box = mb.Box(lengths=[4, 4, 4])

            Charmm(
                empty_compound,
                "charmm_empty_box",
                structure_box_1=test_box_two_propanol_ua_gomc,
                filename_box_1="charmm_filled_box",
                ff_filename="charmm_empty_box",
                residues=[two_propanol_ua.name],
                forcefield_selection="trappe-ua",
                bead_to_atom_name_dict={"_CH3": "C"},
                atom_type_naming_style="general",
            )

    def test_structure_box_0_not_mb_compound(self, ethane_gomc):
        with pytest.raises(
            TypeError,
            match=r"ERROR: The structure_box_0 expected to be of type: "
            r"{} or {}, received: {}".format(
                type(Compound()),
                type(Box(lengths=[1, 1, 1])),
                type("ethane_gomc"),
            ),
        ):
            box_1 = mb.fill_box(
                compound=[ethane_gomc], n_compounds=[1], box=[4, 4, 4]
            )

            Charmm(
                "ethane_gomc",
                "charmm_data_box_0",
                structure_box_1=box_1,
                filename_box_1="charmm_data_box_1",
                ff_filename="charmm_data",
                residues=[ethane_gomc.name],
                forcefield_selection="oplsaa",
                atom_type_naming_style="general",
            )

    def test_structure_box_1_not_mb_compound(self, ethane_gomc):
        with pytest.raises(
            TypeError,
            match=r"ERROR: The structure_box_1 expected to be of type: "
            "{} or {}, received: {}".format(
                type(Compound()), type(Box(lengths=[1, 1, 1])), type(0)
            ),
        ):
            box_0 = mb.fill_box(
                compound=[ethane_gomc], n_compounds=[1], box=[4, 4, 4]
            )

            Charmm(
                box_0,
                "charmm_data_box_0",
                structure_box_1=0,
                filename_box_1="charmm_data_box_1",
                ff_filename="charmm_data",
                residues=[ethane_gomc.name],
                forcefield_selection="oplsaa",
                atom_type_naming_style="general",
            )

    def test_ff_dict_not_entered(self, ethane_gomc):
        with pytest.raises(
            TypeError,
            match=r"ERROR: Please enter the forcefield_selection as it was not provided.",
        ):
            box_0_and_1 = mb.fill_box(
                compound=[ethane_gomc], n_compounds=[1], box=[4, 4, 4]
            )

            Charmm(
                box_0_and_1,
                "charmm_data_box_0",
                structure_box_1=box_0_and_1,
                filename_box_1="charmm_data_box_1",
                ff_filename="charmm_data",
                residues=[ethane_gomc.name],
                forcefield_selection=None,
                atom_type_naming_style="general",
            )

    def test_diff_1_4_coul_scalars(self, two_propanol_ua, water):
        with pytest.raises(
            ValueError,
            match=r"ERROR: There are multiple 1,4-electrostatic scaling factors "
            "GOMC will only accept a singular input for the 1,4-electrostatic "
            "scaling factors.",
        ):
            box_0 = mb.fill_box(
                compound=[water], n_compounds=[1], box=[4, 4, 4]
            )

            box_1 = mb.fill_box(
                compound=[two_propanol_ua], n_compounds=[1], box=[4, 4, 4]
            )

            Charmm(
                box_0,
                "charmm_data_box_0",
                structure_box_1=box_1,
                filename_box_1="charmm_data_box_1",
                ff_filename="charmm_data",
                residues=[water.name, two_propanol_ua.name],
                forcefield_selection={
                    water.name: get_mosdef_gomc_fn(
                        "gmso_spce_water_zero_for_nb__half_coul__lorentz_combining.xml"
                    ),
                    two_propanol_ua.name: "trappe-ua",
                },
                atom_type_naming_style="general",
            )

    def test_write_inp_wo_ff_filename(self, ethane_gomc):
        with pytest.raises(
            TypeError,
            match=r"ERROR: The force field file name was not specified and in the "
            r"Charmm object. "
            r"Therefore, the force field file \(.inp\) can not be written. "
            r"Please use the force field file name when building the Charmm object, "
            r"then use the write_inp function.",
        ):
            box_0_and_1 = mb.fill_box(
                compound=[ethane_gomc], n_compounds=[1], box=[4, 4, 4]
            )

            charmm = Charmm(
                box_0_and_1,
                "charmm_data_box_0",
                structure_box_1=box_0_and_1,
                filename_box_1="charmm_data_box_1",
                ff_filename=None,
                forcefield_selection="oplsaa",
                residues=[ethane_gomc.name],
            )
            charmm.write_inp()

    def test_write_inp_with_2_boxes(self, ethane_gomc):
        box_0_and_1 = mb.fill_box(
            compound=[ethane_gomc], n_compounds=[1], box=[4, 4, 4]
        )

        charmm = Charmm(
            box_0_and_1,
            "charmm_data_box_0",
            structure_box_1=box_0_and_1,
            filename_box_1="charmm_data_box_1",
            ff_filename="charmm_data",
            residues=[ethane_gomc.name],
            forcefield_selection="oplsaa",
            atom_type_naming_style="general",
        )
        charmm.write_inp()

        with open("charmm_data.inp", "r") as fp:
            masses_read = False
            out_gomc = fp.readlines()
            for i, line in enumerate(out_gomc):
                if (
                    "! atom_types" in line
                    and "mass" in line
                    and "atomClass_ResidueName" in line
                ):
                    masses_read = True
                    mass_type_1 = [
                        ["*", "CT", "12.011"],
                        ["*", "HC", "1.008"],
                    ]
                    mass_type_2 = [["ETH_opls_135"], ["ETH_opls_140"]]
                    for j in range(0, len(mass_type_1)):
                        assert (
                            len(out_gomc[i + 1 + j].split("!")[0].split()) == 3
                        )
                        assert (
                            out_gomc[i + 1 + j].split("!")[0].split()[0:3]
                            == mass_type_1[j]
                        )
                        assert (
                            out_gomc[i + 1 + j].split()[4:5] == mass_type_2[j]
                        )
                else:
                    pass

        assert masses_read

    def test_write_inp_zeolite_non_othoganol_box_using_molecule_name(self):
        lattice_cif_ETV_triclinic = load_cif(
            file_or_path=get_mosdef_gomc_fn("ETV_triclinic.cif")
        )
        ETV_triclinic = lattice_cif_ETV_triclinic.populate(x=1, y=1, z=1)
        ETV_triclinic.name = "ETV"

        charmm = Charmm(
            ETV_triclinic,
            "ETV_triclinic",
            ff_filename="ETV_triclinic",
            forcefield_selection={
                "ETV": get_mosdef_gomc_fn(
                    "Charmm_writer_testing_only_zeolite.xml"
                )
            },
            residues=["ETV"],
            bead_to_atom_name_dict=None,
            fix_residue=["ETV"],
            gmso_match_ff_by="group",
            atom_type_naming_style="general",
        )
        charmm.write_inp()

        with open("ETV_triclinic.inp", "r") as fp:
            masses_read = False
            nonbondeds_read = False
            out_gomc = fp.readlines()
            for i, line in enumerate(out_gomc):
                if (
                    "! atom_types" in line
                    and "mass" in line
                    and "atomClass_ResidueName" in line
                ):
                    masses_read = True
                    mass_type_1 = [
                        ["*", "O", "15.9994"],
                        ["*", "Si", "28.0855"],
                    ]
                    mass_type_2 = [["ETV_O"], ["ETV_Si"]]
                    for j in range(0, len(mass_type_1)):
                        assert (
                            len(out_gomc[i + 1 + j].split("!")[0].split()) == 3
                        )
                        assert (
                            out_gomc[i + 1 + j].split("!")[0].split()[0:3]
                            == mass_type_1[j]
                        )
                        assert (
                            out_gomc[i + 1 + j].split()[4:5] == mass_type_2[j]
                        )

                elif (
                    "! type_1" in line
                    and "ignored" in line
                    and "epsilon" in line
                    and "Rmin/2" in line
                    and "ignored" in line
                    and "epsilon,1-4" in line
                    and "Rmin/2,1-4" in line
                    and "extended_type_1" in line
                    and "extended_type_2" in line
                ):
                    nonbondeds_read = True
                    nb_types = [
                        [
                            "O",
                            "0.0",
                            "-0.2941061185",
                            "2.0933917201",
                            "0.0",
                            "-0.0",
                            "2.0933917201",
                        ],
                        [
                            "Si",
                            "0.0",
                            "-0.0556417304",
                            "1.9081854821",
                            "0.0",
                            "-0.0",
                            "1.9081854821",
                        ],
                    ]

                    for j in range(0, len(nb_types)):
                        assert (
                            len(out_gomc[i + 1 + j].split("!")[0].split()) == 7
                        )
                        assert (
                            out_gomc[i + 1 + j].split("!")[0].split()[0:7]
                            == nb_types[j]
                        )

                else:
                    pass

        assert masses_read
        assert nonbondeds_read

    def test_write_inp_zeolite_non_othoganol_box_using_atom_name(self):
        lattice_cif_ETV_triclinic = load_cif(
            file_or_path=get_mosdef_gomc_fn("ETV_triclinic.cif")
        )
        ETV_triclinic = lattice_cif_ETV_triclinic.populate(x=1, y=1, z=1)
        ETV_triclinic.name = "ETV"

        charmm = Charmm(
            ETV_triclinic,
            "ETV_triclinic",
            ff_filename="ETV_triclinic",
            forcefield_selection={
                "O": get_mosdef_gomc_fn(
                    "Charmm_writer_testing_only_zeolite.xml"
                ),
                "Si": get_mosdef_gomc_fn(
                    "Charmm_writer_testing_only_zeolite.xml"
                ),
            },
            residues=["O", "Si"],
            bead_to_atom_name_dict=None,
            fix_residue=["O", "Si"],
            gmso_match_ff_by="molecule",
            atom_type_naming_style="general",
        )
        charmm.write_inp()

        with open("ETV_triclinic.inp", "r") as fp:
            masses_read = False
            nonbondeds_read = False
            out_gomc = fp.readlines()
            for i, line in enumerate(out_gomc):
                if (
                    "! atom_types" in line
                    and "mass" in line
                    and "atomClass_ResidueName" in line
                ):
                    masses_read = True
                    mass_type_1 = [
                        ["*", "O", "15.9994"],
                        ["*", "Si", "28.0855"],
                    ]
                    mass_type_2 = [["O_O"], ["Si_Si"]]
                    for j in range(0, len(mass_type_1)):
                        assert (
                            len(out_gomc[i + 1 + j].split("!")[0].split()) == 3
                        )
                        assert (
                            out_gomc[i + 1 + j].split("!")[0].split()[0:3]
                            == mass_type_1[j]
                        )
                        assert (
                            out_gomc[i + 1 + j].split()[4:5] == mass_type_2[j]
                        )

                elif (
                    "! type_1" in line
                    and "ignored" in line
                    and "epsilon" in line
                    and "Rmin/2" in line
                    and "ignored" in line
                    and "epsilon,1-4" in line
                    and "Rmin/2,1-4" in line
                    and "extended_type_1" in line
                    and "extended_type_2" in line
                ):
                    nonbondeds_read = True
                    nb_types = [
                        [
                            "O",
                            "0.0",
                            "-0.2941061185",
                            "2.0933917201",
                            "0.0",
                            "-0.0",
                            "2.0933917201",
                        ],
                        [
                            "Si",
                            "0.0",
                            "-0.0556417304",
                            "1.9081854821",
                            "0.0",
                            "-0.0",
                            "1.9081854821",
                        ],
                    ]

                    for j in range(0, len(nb_types)):
                        assert (
                            len(out_gomc[i + 1 + j].split("!")[0].split()) == 7
                        )
                        assert (
                            out_gomc[i + 1 + j].split("!")[0].split()[0:7]
                            == nb_types[j]
                        )

                else:
                    pass

        assert masses_read
        assert nonbondeds_read

    # test cif reader ETA psf writer outputs correct atom and residue numbering using non-orthoganol box
    def test_save_zeolite_non_othoganol_box_using_molecule_name_only_psf(self):
        lattice_cif_ETV_triclinic = load_cif(
            file_or_path=get_mosdef_gomc_fn("ETV_triclinic.cif")
        )
        ETV_triclinic = lattice_cif_ETV_triclinic.populate(x=1, y=1, z=1)
        ETV_triclinic.name = "ETV"

        charmm = Charmm(
            ETV_triclinic,
            "ETV_triclinic",
            ff_filename="ETV_triclinic_FF",
            forcefield_selection={
                "ETV": get_mosdef_gomc_fn(
                    "Charmm_writer_testing_only_zeolite.xml"
                )
            },
            residues=["ETV"],
            bead_to_atom_name_dict=None,
            fix_residue=["ETV"],
            gmso_match_ff_by="group",
            atom_type_naming_style="general",
        )

        charmm.write_psf()

        with open("ETV_triclinic.psf", "r") as fp:
            psf_read = False
            out_gomc = fp.readlines()
            for i, line in enumerate(out_gomc):
                if "42 !NATOM" in line:
                    psf_read = True
                    no_O_atoms = 28
                    no_Si_atoms = 14
                    atom_type_charge_etc_list = []
                    for f_i in range(0, no_O_atoms):
                        atom_type_charge_etc_list.append(
                            [
                                str(f_i + 1),
                                "A",
                                str(f_i + 1),
                                "ETV",
                                "O1",
                                "O",
                                "-0.400000",
                                "15.9994",
                            ],
                        )
                    for f_i in range(no_O_atoms, no_O_atoms + no_Si_atoms):
                        atom_type_charge_etc_list.append(
                            [
                                str(f_i + 1),
                                "A",
                                str(f_i + 1),
                                "ETV",
                                "Si1",
                                "Si",
                                "0.800000",
                                "28.0855",
                            ],
                        )

                    for j in range(0, len(atom_type_charge_etc_list)):
                        assert (
                            out_gomc[i + 1 + j].split()[0:8]
                            == atom_type_charge_etc_list[j]
                        )

                else:
                    pass

        assert psf_read

    # test cif reader ETA pdb writer outputs correct atom and residue numbering using non-orthoganol box
    def test_save_zeolite_non_othoganol_box_using_molecule_name_only_pdb(self):
        lattice_cif_ETV_triclinic = load_cif(
            file_or_path=get_mosdef_gomc_fn("ETV_triclinic.cif")
        )
        ETV_triclinic = lattice_cif_ETV_triclinic.populate(x=1, y=1, z=1)
        ETV_triclinic.name = "ETV"

        charmm = Charmm(
            ETV_triclinic,
            "ETV_triclinic",
            ff_filename="ETV_triclinic_FF",
            forcefield_selection={
                "ETV": get_mosdef_gomc_fn(
                    "Charmm_writer_testing_only_zeolite.xml"
                )
            },
            residues=["ETV"],
            bead_to_atom_name_dict=None,
            fix_residue=["ETV"],
            gmso_match_ff_by="group",
            atom_type_naming_style="general",
        )

        charmm.write_pdb()

        with open("ETV_triclinic.pdb", "r") as fp:
            pdb_read = False
            out_gomc = fp.readlines()
            for i, line in enumerate(out_gomc):
                if "CRYST1" in line:
                    pdb_read = True
                    crystal_box_length_angles = [
                        "CRYST1",
                        "8.750",
                        "9.648",
                        "10.272",
                        "105.72",
                        "100.19",
                        "97.02",
                    ]

                    no_O_atoms = 28
                    no_Si_atoms = 14
                    atom_type_res_part_1_list = []
                    for f_i in range(0, no_O_atoms):
                        atom_type_res_part_1_list.append(
                            [
                                "ATOM",
                                str(f_i + 1),
                                "O1",
                                "ETV",
                                "A",
                                str(f_i + 1),
                            ]
                        )
                    for f_i in range(no_O_atoms, no_O_atoms + no_Si_atoms):
                        atom_type_res_part_1_list.append(
                            [
                                "ATOM",
                                str(f_i + 1),
                                "Si1",
                                "ETV",
                                "A",
                                str(f_i + 1),
                            ]
                        )

                    atom_type_res_part_2_list = []
                    for f_i in range(0, no_O_atoms):
                        atom_type_res_part_2_list.append(["0.00", "1.00", "O"])
                    for f_i in range(no_O_atoms, no_O_atoms + no_Si_atoms):
                        atom_type_res_part_2_list.append(["0.00", "1.00", "Si"])

                    assert out_gomc[i].split()[0:7] == crystal_box_length_angles

                    for j in range(0, len(atom_type_res_part_1_list)):
                        assert (
                            out_gomc[i + 1 + j].split()[0:6]
                            == atom_type_res_part_1_list[j]
                        )
                        assert (
                            out_gomc[i + 1 + j].split()[9:12]
                            == atom_type_res_part_2_list[j]
                        )

                else:
                    pass

        assert pdb_read

    # test methane UA psf writer outputs correct atom and residue numbering using orthoganol box
    def test_save_othoganol_methane_ua_using_molecule_name_psf(self):
        methane_ua_bead_name = "_CH4"
        methane_child_bead = mb.Compound(name=methane_ua_bead_name)
        methane_box = mb.fill_box(
            compound=methane_child_bead, n_compounds=4, box=[1, 2, 3]
        )
        methane_box.name = "MET"

        charmm = Charmm(
            methane_box,
            "methane_box",
            ff_filename="methane_box_FF",
            forcefield_selection={methane_box.name: "trappe-ua"},
            residues=[methane_box.name],
            bead_to_atom_name_dict={methane_ua_bead_name: "C"},
            gmso_match_ff_by="group",
            atom_type_naming_style="general",
        )

        charmm.write_psf()

        with open("methane_box.psf", "r") as fp:
            psf_read = False
            out_gomc = fp.readlines()
            for i, line in enumerate(out_gomc):
                if "4 !NATOM" in line:
                    psf_read = True
                    no_methane_atoms = 4
                    atom_type_charge_etc_list = []
                    for f_i in range(0, no_methane_atoms):
                        atom_type_charge_etc_list.append(
                            [
                                str(f_i + 1),
                                "A",
                                str(f_i + 1),
                                "MET",
                                "C1",
                                "CH4",
                                "0.000000",
                                "16.0430",
                            ],
                        )

                    for j in range(0, len(atom_type_charge_etc_list)):
                        assert (
                            out_gomc[i + 1 + j].split()[0:8]
                            == atom_type_charge_etc_list[j]
                        )

                else:
                    pass

        assert psf_read

    # test methane UA psf writer a diff way: outputs correct atom and residue numbering using orthoganol box
    def test_save_othoganol_methane_ua_using_molecule_name_psf_diff_way(self):
        methane_ua_bead_name = "_CH4"
        methane = mb.Compound(name="MET")
        methane_child_bead = mb.Compound(name=methane_ua_bead_name)
        methane.add(methane_child_bead, inherit_periodicity=False)

        methane_box = mb.fill_box(
            compound=methane, n_compounds=4, box=[1, 2, 3]
        )
        charmm = Charmm(
            methane_box,
            "methane_box",
            ff_filename="methane_box_FF",
            forcefield_selection={methane.name: "trappe-ua"},
            residues=[methane.name],
            bead_to_atom_name_dict={methane_ua_bead_name: "C"},
            gmso_match_ff_by="group",
            atom_type_naming_style="general",
        )

        charmm.write_psf()

        with open("methane_box.psf", "r") as fp:
            psf_read = False
            out_gomc = fp.readlines()
            for i, line in enumerate(out_gomc):
                if "4 !NATOM" in line:
                    psf_read = True
                    no_methane_atoms = 4
                    atom_type_charge_etc_list = []
                    for f_i in range(0, no_methane_atoms):
                        atom_type_charge_etc_list.append(
                            [
                                str(f_i + 1),
                                "A",
                                str(f_i + 1),
                                "MET",
                                "C1",
                                "CH4",
                                "0.000000",
                                "16.0430",
                            ],
                        )

                    for j in range(0, len(atom_type_charge_etc_list)):
                        assert (
                            out_gomc[i + 1 + j].split()[0:8]
                            == atom_type_charge_etc_list[j]
                        )

                else:
                    pass

        assert psf_read

    # test methane UA pdb writer outputs in a slightly different format mbuild box construction
    def test_save_othoganol_methane_ua_using_molecule_name_pdb(self):
        methane_ua_bead_name = "_CH4"
        methane_molecule_name = "MET"
        methane = mb.Compound(name=methane_molecule_name)
        methane_child_bead = mb.Compound(name=methane_ua_bead_name)
        methane.add(methane_child_bead, inherit_periodicity=False)

        methane_box = mb.fill_box(
            compound=methane, n_compounds=10, box=[1, 2, 3]
        )

        charmm = Charmm(
            methane_box,
            "methane_box",
            ff_filename="methane_box_FF",
            forcefield_selection={methane_molecule_name: "trappe-ua"},
            residues=[methane_molecule_name],
            bead_to_atom_name_dict={methane_ua_bead_name: "C"},
            gmso_match_ff_by="group",
            atom_type_naming_style="general",
        )

        charmm.write_pdb()

        with open("methane_box.pdb", "r") as fp:
            pdb_read = False
            out_gomc = fp.readlines()
            for i, line in enumerate(out_gomc):
                if "CRYST1" in line:
                    pdb_read = True
                    crystal_box_length_angles = [
                        "CRYST1",
                        "10.000",
                        "20.000",
                        "30.000",
                        "90.00",
                        "90.00",
                        "90.00",
                    ]

                    no_methane_atoms = 4
                    atom_type_res_part_1_list = []
                    for f_i in range(0, no_methane_atoms):
                        # Note: If _CH4 used instead of MET, _CH4 and A are merged togther here
                        atom_type_res_part_1_list.append(
                            [
                                "ATOM",
                                str(f_i + 1),
                                "C1",
                                "MET",
                                "A",
                                str(f_i + 1),
                            ]
                        )

                    atom_type_res_part_2_list = []
                    for f_i in range(0, no_methane_atoms):
                        atom_type_res_part_2_list.append(["0.00", "0.00", "BD"])

                    assert out_gomc[i].split()[0:7] == crystal_box_length_angles

                    for j in range(0, len(atom_type_res_part_1_list)):
                        # Note: If _CH4 used instead of MET,
                        # _CH4 and A are merged togther here hence split()[0:5] not split()[0:6]

                        assert (
                            out_gomc[i + 1 + j].split()[0:6]
                            == atom_type_res_part_1_list[j]
                        )
                        # Note: If _CH4 used instead of MET,
                        # _CH4 and A are merged togther here hence split()[9:12] not split()[8:11]
                        assert (
                            out_gomc[i + 1 + j].split()[9:12]
                            == atom_type_res_part_2_list[j]
                        )

                else:
                    pass

        assert pdb_read

    # test cif reader ETA psf writer outputs correct atom and residue numbering using non-orthoganol box
    def test_save_zeolite_non_othoganol_box_using_atom_name_only_psf(self):
        lattice_cif_ETV_triclinic = load_cif(
            file_or_path=get_mosdef_gomc_fn("ETV_triclinic.cif")
        )
        ETV_triclinic = lattice_cif_ETV_triclinic.populate(x=1, y=1, z=1)
        ETV_triclinic.name = "ETV"

        charmm = Charmm(
            ETV_triclinic,
            "ETV_triclinic",
            ff_filename="ETV_triclinic_FF",
            forcefield_selection={
                "O": get_mosdef_gomc_fn(
                    "Charmm_writer_testing_only_zeolite.xml"
                ),
                "Si": get_mosdef_gomc_fn(
                    "Charmm_writer_testing_only_zeolite.xml"
                ),
            },
            residues=["O", "Si"],
            bead_to_atom_name_dict=None,
            fix_residue=["O", "Si"],
            gmso_match_ff_by="molecule",
            atom_type_naming_style="general",
        )

        charmm.write_psf()

        with open("ETV_triclinic.psf", "r") as fp:
            psf_read = False
            out_gomc = fp.readlines()
            for i, line in enumerate(out_gomc):
                if "42 !NATOM" in line:
                    psf_read = True
                    no_O_atoms = 28
                    no_Si_atoms = 14
                    atom_type_charge_etc_list = []
                    for f_i in range(0, no_O_atoms):
                        atom_type_charge_etc_list.append(
                            [
                                str(f_i + 1),
                                "A",
                                str(f_i + 1),
                                "O",
                                "O1",
                                "O",
                                "-0.400000",
                                "15.9994",
                            ],
                        )
                    for f_i in range(no_O_atoms, no_O_atoms + no_Si_atoms):
                        atom_type_charge_etc_list.append(
                            [
                                str(f_i + 1),
                                "A",
                                str(f_i + 1),
                                "Si",
                                "Si1",
                                "Si",
                                "0.800000",
                                "28.0855",
                            ],
                        )

                    for j in range(0, len(atom_type_charge_etc_list)):
                        assert (
                            out_gomc[i + 1 + j].split()[0:8]
                            == atom_type_charge_etc_list[j]
                        )

                else:
                    pass

        assert psf_read

    # test cif reader ETA pdb writer outputs correct atom and residue numbering using non-orthoganol box
    def test_save_zeolite_non_othoganol_box_using_atom_name_only_pdb(self):
        lattice_cif_ETV_triclinic = load_cif(
            file_or_path=get_mosdef_gomc_fn("ETV_triclinic.cif")
        )
        ETV_triclinic = lattice_cif_ETV_triclinic.populate(x=1, y=1, z=1)
        ETV_triclinic.name = "ETV"

        charmm = Charmm(
            ETV_triclinic,
            "ETV_triclinic",
            ff_filename="ETV_triclinic_FF",
            forcefield_selection={
                "O": get_mosdef_gomc_fn(
                    "Charmm_writer_testing_only_zeolite.xml"
                ),
                "Si": get_mosdef_gomc_fn(
                    "Charmm_writer_testing_only_zeolite.xml"
                ),
            },
            residues=["O", "Si"],
            bead_to_atom_name_dict=None,
            fix_residue=["O", "Si"],
            gmso_match_ff_by="molecule",
            atom_type_naming_style="general",
        )

        charmm.write_pdb()

        with open("ETV_triclinic.pdb", "r") as fp:
            pdb_read = False
            out_gomc = fp.readlines()
            for i, line in enumerate(out_gomc):
                if "CRYST1" in line:
                    pdb_read = True
                    crystal_box_length_angles = [
                        "CRYST1",
                        "8.750",
                        "9.648",
                        "10.272",
                        "105.72",
                        "100.19",
                        "97.02",
                    ]

                    no_O_atoms = 28
                    no_Si_atoms = 14
                    atom_type_res_part_1_list = []
                    for f_i in range(0, no_O_atoms):
                        atom_type_res_part_1_list.append(
                            [
                                "ATOM",
                                str(f_i + 1),
                                "O1",
                                "O",
                                "A",
                                str(f_i + 1),
                            ]
                        )
                    for f_i in range(no_O_atoms, no_O_atoms + no_Si_atoms):
                        atom_type_res_part_1_list.append(
                            [
                                "ATOM",
                                str(f_i + 1),
                                "Si1",
                                "Si",
                                "A",
                                str(f_i + 1),
                            ]
                        )

                    atom_type_res_part_2_list = []
                    for f_i in range(0, no_O_atoms):
                        atom_type_res_part_2_list.append(["0.00", "1.00", "O"])
                    for f_i in range(no_O_atoms, no_O_atoms + no_Si_atoms):
                        atom_type_res_part_2_list.append(["0.00", "1.00", "Si"])

                    assert out_gomc[i].split()[0:7] == crystal_box_length_angles

                    for j in range(0, len(atom_type_res_part_1_list)):
                        assert (
                            out_gomc[i + 1 + j].split()[0:6]
                            == atom_type_res_part_1_list[j]
                        )
                        assert (
                            out_gomc[i + 1 + j].split()[9:12]
                            == atom_type_res_part_2_list[j]
                        )

                else:
                    pass

        assert pdb_read

    # test methane UA psf writer outputs correct atom and residue numbering using orthoganol box
    def test_save_othoganol_methane_ua_using_atom_name_psf(self):
        methane_ua_bead_name = "_CH4"
        methane_child_bead = mb.Compound(name=methane_ua_bead_name)
        methane_box = mb.fill_box(
            compound=methane_child_bead, n_compounds=4, box=[1, 2, 3]
        )
        methane_box.name = "MET"

        charmm = Charmm(
            methane_box,
            "methane_box",
            ff_filename="methane_box_FF",
            forcefield_selection={methane_ua_bead_name: "trappe-ua"},
            residues=[methane_ua_bead_name],
            bead_to_atom_name_dict={methane_ua_bead_name: "C"},
            gmso_match_ff_by="molecule",
            atom_type_naming_style="general",
        )

        charmm.write_psf()

        with open("methane_box.psf", "r") as fp:
            psf_read = False
            out_gomc = fp.readlines()
            for i, line in enumerate(out_gomc):
                if "4 !NATOM" in line:
                    psf_read = True
                    no_methane_atoms = 4
                    atom_type_charge_etc_list = []
                    for f_i in range(0, no_methane_atoms):
                        atom_type_charge_etc_list.append(
                            [
                                str(f_i + 1),
                                "A",
                                str(f_i + 1),
                                "_CH4",
                                "C1",
                                "CH4",
                                "0.000000",
                                "16.0430",
                            ],
                        )

                    for j in range(0, len(atom_type_charge_etc_list)):
                        assert (
                            out_gomc[i + 1 + j].split()[0:8]
                            == atom_type_charge_etc_list[j]
                        )

                else:
                    pass

        assert psf_read

    # test methane UA pdb writer outputs correct atom and residue numbering using orthoganol box
    def test_save_othoganol_methane_ua_using_atom_name_pdb(self):
        methane_ua_bead_name = "_CH4"
        methane_child_bead = mb.Compound(name=methane_ua_bead_name)
        methane_box = mb.fill_box(
            compound=methane_child_bead, n_compounds=4, box=[1, 2, 3]
        )
        methane_box.name = "MET"

        charmm = Charmm(
            methane_box,
            "methane_box",
            ff_filename="methane_box_FF",
            forcefield_selection={methane_ua_bead_name: "trappe-ua"},
            residues=[methane_ua_bead_name],
            bead_to_atom_name_dict={methane_ua_bead_name: "C"},
            gmso_match_ff_by="molecule",
            atom_type_naming_style="general",
        )

        charmm.write_pdb()

        with open("methane_box.pdb", "r") as fp:
            pdb_read = False
            out_gomc = fp.readlines()
            for i, line in enumerate(out_gomc):
                if "CRYST1" in line:
                    pdb_read = True
                    crystal_box_length_angles = [
                        "CRYST1",
                        "10.000",
                        "20.000",
                        "30.000",
                        "90.00",
                        "90.00",
                        "90.00",
                    ]

                    no_methane_atoms = 4
                    atom_type_res_part_1_list = []
                    for f_i in range(0, no_methane_atoms):
                        # Note: If _CH4 used instead of MET, _CH4 and A are merged togther here
                        atom_type_res_part_1_list.append(
                            [
                                "ATOM",
                                str(f_i + 1),
                                "C1",
                                "_CH4A",
                                str(f_i + 1),
                            ]
                        )

                    atom_type_res_part_2_list = []
                    for f_i in range(0, no_methane_atoms):
                        atom_type_res_part_2_list.append(["0.00", "0.00", "BD"])

                    assert out_gomc[i].split()[0:7] == crystal_box_length_angles

                    for j in range(0, len(atom_type_res_part_1_list)):
                        # Note: If _CH4 used instead of MET,
                        # _CH4 and A are merged togther here hence split()[0:5] not split()[0:6]

                        assert (
                            out_gomc[i + 1 + j].split()[0:5]
                            == atom_type_res_part_1_list[j]
                        )
                        # Note: If _CH4 used instead of MET,
                        # _CH4 and A are merged togther here hence split()[9:12] not split()[8:11]
                        assert (
                            out_gomc[i + 1 + j].split()[8:11]
                            == atom_type_res_part_2_list[j]
                        )

                else:
                    pass

        assert pdb_read

    # test methane UA psf writer outputs correct atom and residue numbering using orthoganol box
    def test_save_othoganol_methane_ua_compound_and_subcompound_psf(self):
        methane_ua_bead_name = "_CH4"
        methane_child_bead = mb.Compound(name=methane_ua_bead_name)
        methane_box = mb.fill_box(
            compound=methane_child_bead, n_compounds=4, box=[1, 2, 3]
        )
        methane_box.name = "MET"

        charmm = Charmm(
            methane_box,
            "methane_box_compound_and_subcompound",
            ff_filename="methane_box_compound_and_subcompound_FF",
            forcefield_selection={methane_box.name: "trappe-ua"},
            residues=[methane_box.name],
            bead_to_atom_name_dict={methane_ua_bead_name: "C"},
            gmso_match_ff_by="group",
            atom_type_naming_style="general",
        )
        charmm.write_psf()

        with open("methane_box_compound_and_subcompound.psf", "r") as fp:
            psf_read = False
            out_gomc = fp.readlines()
            for i, line in enumerate(out_gomc):
                if "4 !NATOM" in line:
                    psf_read = True
                    no_methane_atoms = 4
                    atom_type_charge_etc_list = []
                    for f_i in range(0, no_methane_atoms):
                        atom_type_charge_etc_list.append(
                            [
                                str(f_i + 1),
                                "A",
                                str(f_i + 1),
                                "MET",
                                "C1",
                                "CH4",
                                "0.000000",
                                "16.0430",
                            ],
                        )

                    for j in range(0, len(atom_type_charge_etc_list)):
                        assert (
                            out_gomc[i + 1 + j].split()[0:8]
                            == atom_type_charge_etc_list[j]
                        )

                else:
                    pass

        assert psf_read

    # test methane UA pdb writer outputs correct atom and residue numbering using orthoganol box
    def test_save_othoganol_methane_ua_compound_and_subcompound_pdb(self):
        methane_ua_bead_name = "_CH4"
        methane_child_bead = mb.Compound(name=methane_ua_bead_name)
        methane_box = mb.fill_box(
            compound=methane_child_bead, n_compounds=4, box=[1, 2, 3]
        )
        methane_box.name = "MET"

        charmm = Charmm(
            methane_box,
            "methane_box_compound_and_subcompound",
            ff_filename="methane_box_compound_and_subcompound_FF",
            forcefield_selection={methane_box.name: "trappe-ua"},
            residues=[methane_box.name],
            bead_to_atom_name_dict={methane_ua_bead_name: "C"},
            gmso_match_ff_by="group",
            atom_type_naming_style="general",
        )

        charmm.write_pdb()

        with open("methane_box_compound_and_subcompound.pdb", "r") as fp:
            pdb_read = False
            out_gomc = fp.readlines()
            for i, line in enumerate(out_gomc):
                if "CRYST1" in line:
                    pdb_read = True
                    crystal_box_length_angles = [
                        "CRYST1",
                        "10.000",
                        "20.000",
                        "30.000",
                        "90.00",
                        "90.00",
                        "90.00",
                    ]

                    no_methane_atoms = 4
                    atom_type_res_part_1_list = []
                    for f_i in range(0, no_methane_atoms):
                        atom_type_res_part_1_list.append(
                            [
                                "ATOM",
                                str(f_i + 1),
                                "C1",
                                "MET",
                                "A",
                                str(f_i + 1),
                            ]
                        )

                    atom_type_res_part_2_list = []
                    for f_i in range(0, no_methane_atoms):
                        atom_type_res_part_2_list.append(["0.00", "0.00", "BD"])

                    assert out_gomc[i].split()[0:7] == crystal_box_length_angles

                    for j in range(0, len(atom_type_res_part_1_list)):
                        assert (
                            out_gomc[i + 1 + j].split()[0:6]
                            == atom_type_res_part_1_list[j]
                        )
                        assert (
                            out_gomc[i + 1 + j].split()[9:12]
                            == atom_type_res_part_2_list[j]
                        )

                else:
                    pass

        assert pdb_read

    # test warning the for non-zero charged system box 0
    def test_save_system_charge_non_zero_box_0_only_gomc_ff(self, water):
        with pytest.warns(
            UserWarning,
            match="System is not charge neutral for structure_box_0. Total charge is -0.8476.",
        ):
            box_0 = mb.fill_box(
                compound=[water], n_compounds=[1], box=[4, 4, 4]
            )

            Charmm(
                box_0,
                "system_charge_non_zero_gomc_ff_box_0",
                structure_box_1=None,
                filename_box_1=None,
                ff_filename="system_charge_non_zero_gomc_ff",
                residues=[water.name],
                forcefield_selection={
                    water.name: get_mosdef_gomc_fn(
                        "gmso_spce_water_bad_charges__lorentz_combining.xml"
                    ),
                },
            )

    # test warning the for non-zero charged system box 0
    def test_save_system_charge_non_zero_box_0_gomc_ff(
        self, water, two_propanol_ua
    ):
        with pytest.warns(
            UserWarning,
            match="System is not charge neutral for structure_box_0. Total charge is -0.8476.",
        ):
            box_0 = mb.fill_box(
                compound=[water], n_compounds=[1], box=[4, 4, 4]
            )

            box_1 = mb.fill_box(
                compound=[two_propanol_ua], n_compounds=[1], box=[4, 4, 4]
            )

            Charmm(
                box_0,
                "system_charge_non_zero_gomc_ff_box_0",
                structure_box_1=box_1,
                filename_box_1="system_charge_non_zero_gomc_ff_box_1",
                ff_filename="system_charge_non_zero_gomc_ff",
                residues=[water.name, two_propanol_ua.name],
                forcefield_selection={
                    water.name: get_mosdef_gomc_fn(
                        "gmso_spce_water_bad_charges__lorentz_combining.xml"
                    ),
                    two_propanol_ua.name: get_mosdef_gomc_fn(
                        "gmso_two_propanol_periodic_dihedrals_ua_all_bond_angles_dihedrals_k_times_half.xml"
                    ),
                },
            )

    def test_save_system_charge_non_zero_box_1_gomc_ff(
        self, water, two_propanol_ua
    ):
        with pytest.warns(
            UserWarning,
            match="System is not charge neutral for structure_box_1. Total charge is -0.8476.",
        ):
            box_0 = mb.fill_box(
                compound=[two_propanol_ua], n_compounds=[1], box=[4, 4, 4]
            )

            box_1 = mb.fill_box(
                compound=[water], n_compounds=[1], box=[4, 4, 4]
            )

            Charmm(
                box_0,
                "system_charge_non_zero_gomc_ff_box_0",
                structure_box_1=box_1,
                filename_box_1="system_charge_non_zero_gomc_ff_box_1",
                ff_filename="system_charge_non_zero_gomc_ff",
                residues=[water.name, two_propanol_ua.name],
                forcefield_selection={
                    water.name: get_mosdef_gomc_fn(
                        "gmso_spce_water_bad_charges__lorentz_combining.xml"
                    ),
                    two_propanol_ua.name: get_mosdef_gomc_fn(
                        "gmso_two_propanol_periodic_dihedrals_ua_all_bond_angles_dihedrals_k_times_half.xml"
                    ),
                },
            )

    # **** testing the different dihedral types produce the same values and work properly ****
    # test the gmso RB dihderal input with 1 times the RB torsion values
    def test_save_gmso_RB_dihedral_times_1_gomc_ff(self, two_propanol_ua):
        box_0 = mb.fill_box(
            compound=[two_propanol_ua], n_compounds=[1], box=[4, 4, 4]
        )

        charmm = Charmm(
            box_0,
            "gmso_RB_dihedral_gomc",
            ff_filename="gmso_RB_dihedral_gomc",
            residues=[two_propanol_ua.name],
            forcefield_selection={
                two_propanol_ua.name: get_mosdef_gomc_fn(
                    "gmso_two_propanol_RB_dihedrals_times_1_ua.xml"
                ),
            },
            bead_to_atom_name_dict={"_CH3": "C", "_CH2": "C", "_HC": "C"},
            atom_type_naming_style="general",
        )
        charmm.write_inp()

        with open("gmso_RB_dihedral_gomc.inp", "r") as fp:
            dihedrals_read = False
            out_gomc = fp.readlines()
            for i, line in enumerate(out_gomc):
                if (
                    "! type_1" in line
                    and "type_2" in line
                    and "type_3" in line
                    and "type_4" in line
                    and "Kchi" in line
                    and "n" in line
                    and "delta" in line
                    and "extended_type_1" in line
                    and "extended_type_2" in line
                    and "extended_type_3" in line
                    and "extended_type_4" in line
                ):
                    dihedrals_read = True
                    dihedral_types = [
                        ["CH3", "CH", "O", "H", "-0.392135", "1", "180.0"],
                        ["CH3", "CH", "O", "H", "-0.062518", "2", "0.0"],
                        ["CH3", "CH", "O", "H", "0.345615", "3", "180.0"],
                    ]
                    for j in range(0, len(dihedral_types)):
                        assert (
                            len(out_gomc[i + 1 + j].split("!")[0].split()) == 7
                        )
                        assert (
                            out_gomc[i + 1 + j].split("!")[0].split()[0:7]
                            == dihedral_types[j]
                        )

                else:
                    pass

        assert dihedrals_read

    # test the gmso RB dihderal input with 2 times the RB torsion values
    def test_save_gmso_RB_dihedral_times_2_gomc_ff(self, two_propanol_ua):
        box_0 = mb.fill_box(
            compound=[two_propanol_ua], n_compounds=[1], box=[4, 4, 4]
        )

        charmm = Charmm(
            box_0,
            "gmso_RB_dihedral_gomc",
            ff_filename="gmso_RB_dihedral_gomc",
            residues=[two_propanol_ua.name],
            forcefield_selection={
                two_propanol_ua.name: get_mosdef_gomc_fn(
                    "gmso_two_propanol_RB_dihedrals_times_2_ua.xml"
                ),
            },
            bead_to_atom_name_dict={"_CH3": "C", "_CH2": "C", "_HC": "C"},
            atom_type_naming_style="general",
        )
        charmm.write_inp()

        with open("gmso_RB_dihedral_gomc.inp", "r") as fp:
            dihedrals_read = False
            out_gomc = fp.readlines()
            for i, line in enumerate(out_gomc):
                if (
                    "! type_1" in line
                    and "type_2" in line
                    and "type_3" in line
                    and "type_4" in line
                    and "Kchi" in line
                    and "n" in line
                    and "delta" in line
                    and "extended_type_1" in line
                    and "extended_type_2" in line
                    and "extended_type_3" in line
                    and "extended_type_4" in line
                ):
                    dihedrals_read = True
                    dihedral_types = [
                        ["CH3", "CH", "O", "H", "-0.78427", "1", "180.0"],
                        ["CH3", "CH", "O", "H", "-0.125036", "2", "0.0"],
                        ["CH3", "CH", "O", "H", "0.69123", "3", "180.0"],
                    ]
                    for j in range(0, len(dihedral_types)):
                        assert (
                            len(out_gomc[i + 1 + j].split("!")[0].split()) == 7
                        )
                        assert (
                            out_gomc[i + 1 + j].split("!")[0].split()[0:7]
                            == dihedral_types[j]
                        )

                else:
                    pass

        assert dihedrals_read

    # test the gmso OPLS dihderal input with 0.5 times 1 OPLS torsion values
    def test_save_gmso_OPLS_dihedral_0_5_times_1_gomc_ff(self, two_propanol_ua):
        box_0 = mb.fill_box(
            compound=[two_propanol_ua], n_compounds=[1], box=[4, 4, 4]
        )

        charmm = Charmm(
            box_0,
            "gmso_OPLS_dihedral_gomc",
            ff_filename="gmso_OPLS_dihedral_gomc",
            residues=[two_propanol_ua.name],
            forcefield_selection={
                two_propanol_ua.name: get_mosdef_gomc_fn(
                    "gmso_two_propanol_OPLS_dihedrals_0_5_times_1_ua.xml"
                ),
            },
            bead_to_atom_name_dict={"_CH3": "C", "_CH2": "C", "_HC": "C"},
            atom_type_naming_style="general",
        )
        charmm.write_inp()

        with open("gmso_OPLS_dihedral_gomc.inp", "r") as fp:
            dihedrals_read = False
            out_gomc = fp.readlines()
            for i, line in enumerate(out_gomc):
                if (
                    "! type_1" in line
                    and "type_2" in line
                    and "type_3" in line
                    and "type_4" in line
                    and "Kchi" in line
                    and "n" in line
                    and "delta" in line
                    and "extended_type_1" in line
                    and "extended_type_2" in line
                    and "extended_type_3" in line
                    and "extended_type_4" in line
                ):
                    dihedrals_read = True
                    dihedral_types = [
                        ["CH3", "CH", "O", "H", "-0.392135", "1", "180.0"],
                        ["CH3", "CH", "O", "H", "-0.062518", "2", "0.0"],
                        ["CH3", "CH", "O", "H", "0.345615", "3", "180.0"],
                    ]
                    for j in range(0, len(dihedral_types)):
                        assert (
                            len(out_gomc[i + 1 + j].split("!")[0].split()) == 7
                        )
                        assert (
                            out_gomc[i + 1 + j].split("!")[0].split()[0:7]
                            == dihedral_types[j]
                        )

                else:
                    pass

        assert dihedrals_read

    # test the gmso OPLS dihderal input with 0.5 times 2 OPLS torsion values
    def test_save_gmso_OPLS_dihedral_0_5_times_2_gomc_ff(self, two_propanol_ua):
        box_0 = mb.fill_box(
            compound=[two_propanol_ua], n_compounds=[1], box=[4, 4, 4]
        )

        charmm = Charmm(
            box_0,
            "gmso_OPLS_dihedral_gomc",
            ff_filename="gmso_OPLS_dihedral_gomc",
            residues=[two_propanol_ua.name],
            forcefield_selection={
                two_propanol_ua.name: get_mosdef_gomc_fn(
                    "gmso_two_propanol_OPLS_dihedrals_0_5_times_2_ua.xml"
                ),
            },
            bead_to_atom_name_dict={"_CH3": "C", "_CH2": "C", "_HC": "C"},
            atom_type_naming_style="general",
        )
        charmm.write_inp()

        with open("gmso_OPLS_dihedral_gomc.inp", "r") as fp:
            dihedrals_read = False
            out_gomc = fp.readlines()
            for i, line in enumerate(out_gomc):
                if (
                    "! type_1" in line
                    and "type_2" in line
                    and "type_3" in line
                    and "type_4" in line
                    and "Kchi" in line
                    and "n" in line
                    and "delta" in line
                    and "extended_type_1" in line
                    and "extended_type_2" in line
                    and "extended_type_3" in line
                    and "extended_type_4" in line
                ):
                    dihedrals_read = True
                    dihedral_types = [
                        ["CH3", "CH", "O", "H", "-0.78427", "1", "180.0"],
                        ["CH3", "CH", "O", "H", "-0.125036", "2", "0.0"],
                        ["CH3", "CH", "O", "H", "0.69123", "3", "180.0"],
                    ]
                    for j in range(0, len(dihedral_types)):
                        assert (
                            len(out_gomc[i + 1 + j].split("!")[0].split()) == 7
                        )
                        assert (
                            out_gomc[i + 1 + j].split("!")[0].split()[0:7]
                            == dihedral_types[j]
                        )

                else:
                    pass

        assert dihedrals_read

    # test the gmso OPLS dihderal input with half times 1 OPLS torsion values
    def test_save_gmso_OPLS_dihedral_half_times_1_gomc_ff(
        self, two_propanol_ua
    ):
        box_0 = mb.fill_box(
            compound=[two_propanol_ua], n_compounds=[1], box=[4, 4, 4]
        )

        charmm = Charmm(
            box_0,
            "gmso_OPLS_dihedral_gomc",
            ff_filename="gmso_OPLS_dihedral_gomc",
            residues=[two_propanol_ua.name],
            forcefield_selection={
                two_propanol_ua.name: get_mosdef_gomc_fn(
                    "gmso_two_propanol_OPLS_dihedrals_half_times_1_ua.xml"
                ),
            },
            bead_to_atom_name_dict={"_CH3": "C", "_CH2": "C", "_HC": "C"},
            atom_type_naming_style="general",
        )
        charmm.write_inp()

        with open("gmso_OPLS_dihedral_gomc.inp", "r") as fp:
            dihedrals_read = False
            out_gomc = fp.readlines()
            for i, line in enumerate(out_gomc):
                if (
                    "! type_1" in line
                    and "type_2" in line
                    and "type_3" in line
                    and "type_4" in line
                    and "Kchi" in line
                    and "n" in line
                    and "delta" in line
                    and "extended_type_1" in line
                    and "extended_type_2" in line
                    and "extended_type_3" in line
                    and "extended_type_4" in line
                ):
                    dihedrals_read = True
                    dihedral_types = [
                        ["CH3", "CH", "O", "H", "-0.392135", "1", "180.0"],
                        ["CH3", "CH", "O", "H", "-0.062518", "2", "0.0"],
                        ["CH3", "CH", "O", "H", "0.345615", "3", "180.0"],
                    ]
                    for j in range(0, len(dihedral_types)):
                        assert (
                            len(out_gomc[i + 1 + j].split("!")[0].split()) == 7
                        )
                        assert (
                            out_gomc[i + 1 + j].split("!")[0].split()[0:7]
                            == dihedral_types[j]
                        )

                else:
                    pass

        assert dihedrals_read

    # test the gmso OPLS dihderal input with 0.5 times 2 OPLS torsion values
    def test_save_gmso_OPLS_dihedral_half_times_2_gomc_ff(
        self, two_propanol_ua
    ):
        box_0 = mb.fill_box(
            compound=[two_propanol_ua], n_compounds=[1], box=[4, 4, 4]
        )

        charmm = Charmm(
            box_0,
            "gmso_OPLS_dihedral_gomc",
            ff_filename="gmso_OPLS_dihedral_gomc",
            residues=[two_propanol_ua.name],
            forcefield_selection={
                two_propanol_ua.name: get_mosdef_gomc_fn(
                    "gmso_two_propanol_OPLS_dihedrals_half_times_2_ua.xml"
                ),
            },
            bead_to_atom_name_dict={"_CH3": "C", "_CH2": "C", "_HC": "C"},
            atom_type_naming_style="general",
        )
        charmm.write_inp()

        with open("gmso_OPLS_dihedral_gomc.inp", "r") as fp:
            dihedrals_read = False
            out_gomc = fp.readlines()
            for i, line in enumerate(out_gomc):
                if (
                    "! type_1" in line
                    and "type_2" in line
                    and "type_3" in line
                    and "type_4" in line
                    and "Kchi" in line
                    and "n" in line
                    and "delta" in line
                    and "extended_type_1" in line
                    and "extended_type_2" in line
                    and "extended_type_3" in line
                    and "extended_type_4" in line
                ):
                    dihedrals_read = True
                    dihedral_types = [
                        ["CH3", "CH", "O", "H", "-0.78427", "1", "180.0"],
                        ["CH3", "CH", "O", "H", "-0.125036", "2", "0.0"],
                        ["CH3", "CH", "O", "H", "0.69123", "3", "180.0"],
                    ]
                    for j in range(0, len(dihedral_types)):
                        assert (
                            len(out_gomc[i + 1 + j].split("!")[0].split()) == 7
                        )
                        assert (
                            out_gomc[i + 1 + j].split("!")[0].split()[0:7]
                            == dihedral_types[j]
                        )

                else:
                    pass

        assert dihedrals_read

    # test the gmso periodic dihderal input add k (bonds, angles, dihedrals times 1/2)
    def test_save_gmso_periodic_dihedral_gomc_ff_all_ks_times_half(
        self, two_propanol_ua
    ):
        box_0 = mb.fill_box(
            compound=[two_propanol_ua], n_compounds=[1], box=[4, 4, 4]
        )

        charmm = Charmm(
            box_0,
            "gmso_periodic_dihedral_gomc",
            ff_filename="gmso_periodic_dihedral_gomc",
            residues=[two_propanol_ua.name],
            forcefield_selection={
                two_propanol_ua.name: get_mosdef_gomc_fn(
                    "gmso_two_propanol_periodic_dihedrals_ua_all_bond_angles_dihedrals_k_times_half.xml"
                ),
            },
            bead_to_atom_name_dict={"_CH3": "C", "_CH2": "C", "_HC": "C"},
            atom_type_naming_style="general",
        )
        charmm.write_inp()

        with open("gmso_periodic_dihedral_gomc.inp", "r") as fp:
            masses_read = False
            bonds_read = False
            angles_read = False
            dihedrals_read = False
            out_gomc = fp.readlines()
            for i, line in enumerate(out_gomc):
                if (
                    "! atom_types" in line
                    and "mass" in line
                    and "atomClass_ResidueName" in line
                ):
                    masses_read = True
                    atom_types_1 = [
                        ["*", "CH3", "15.035"],
                        ["*", "CH", "13.019"],
                        ["*", "O", "15.9994"],
                        ["*", "H", "1.008"],
                    ]
                    atom_types_2 = [
                        ["POL_CH3_sp3"],
                        ["POL_CH_O"],
                        ["POL_O"],
                        ["POL_H"],
                    ]

                    for j in range(0, len(atom_types_1)):
                        assert (
                            len(out_gomc[i + 1 + j].split("!")[0].split()) == 3
                        )
                        assert (
                            out_gomc[i + 1 + j].split("!")[0].split()[0:3]
                            == atom_types_1[j]
                        )
                        assert (
                            out_gomc[i + 1 + j].split()[4:5] == atom_types_2[j]
                        )

                elif (
                    "! type_1" in line
                    and "type_2" in line
                    and "Kb" in line
                    and "b0" in line
                    and "extended_type_1" in line
                    and "extended_type_2" in line
                ):
                    bonds_read = True
                    bond_types = [
                        ["CH3", "CH", "600.40153", "1.54"],
                        ["CH", "O", "600.40153", "1.43"],
                        ["O", "H", "600.40153", "0.945"],
                    ]
                    total_bonds_evaluated = []
                    total_bonds_evaluated_reorg = []
                    for j in range(0, len(bond_types)):
                        assert (
                            len(out_gomc[i + 1 + j].split("!")[0].split()) == 4
                        )

                        if (
                            out_gomc[i + 1 + j].split("!")[0].split()[0:4]
                            == bond_types[0]
                            or bond_types[1]
                            or bond_types[2]
                        ):
                            total_bonds_evaluated.append(
                                out_gomc[i + 1 + j].split("!")[0].split()[0:4]
                            )
                    for k in range(0, len(bond_types)):
                        if bond_types[k] in total_bonds_evaluated:
                            total_bonds_evaluated_reorg.append(bond_types[k])
                    assert total_bonds_evaluated_reorg == bond_types

                elif (
                    "! type_1 " in line
                    and "type_2" in line
                    and "type_3" in line
                    and "Ktheta" in line
                    and "Theta0" in line
                    and "extended_type_1" in line
                    and "extended_type_2" in line
                    and "extended_type_3" in line
                ):
                    angles_read = True
                    angle_types = [
                        ["CH3", "CH", "O", "50.077544", "109.469889"],
                        ["CH3", "CH", "CH3", "62.10013", "112.000071"],
                        ["CH", "O", "H", "55.045554", "108.499872"],
                    ]
                    total_angles_evaluated = []
                    total_angles_evaluated_reorg = []
                    for j in range(0, len(angle_types)):
                        assert (
                            len(out_gomc[i + 1 + j].split("!")[0].split()) == 5
                        )
                        if (
                            out_gomc[i + 1 + j].split("!")[0].split()[0:5]
                            == angle_types[0]
                            or angle_types[1]
                            or angle_types[2]
                        ):
                            total_angles_evaluated.append(
                                out_gomc[i + 1 + j].split("!")[0].split()[0:5]
                            )
                    for k in range(0, len(angle_types)):
                        if angle_types[k] in total_angles_evaluated:
                            total_angles_evaluated_reorg.append(angle_types[k])

                    assert total_angles_evaluated_reorg == angle_types

                elif (
                    "! type_1" in line
                    and "type_2" in line
                    and "type_3" in line
                    and "type_4" in line
                    and "Kchi" in line
                    and "n" in line
                    and "delta" in line
                    and "extended_type_1" in line
                    and "extended_type_2" in line
                    and "extended_type_3" in line
                    and "extended_type_4" in line
                ):
                    dihedrals_read = True
                    dihedral_types = [
                        ["CH3", "CH", "O", "H", "-0.196067", "1", "180.0"],
                        ["CH3", "CH", "O", "H", "-0.031259", "2", "0.0"],
                        ["CH3", "CH", "O", "H", "0.172807", "3", "180.0"],
                    ]
                    for j in range(0, len(dihedral_types)):
                        assert (
                            len(out_gomc[i + 1 + j].split("!")[0].split()) == 7
                        )
                        assert (
                            out_gomc[i + 1 + j].split("!")[0].split()[0:7]
                            == dihedral_types[j]
                        )

        assert masses_read
        assert bonds_read
        assert angles_read
        assert dihedrals_read

    # test the gmso periodic dihderal input add k (bonds, angles, dihedrals times 1)
    def test_save_gmso_periodic_dihedral_gomc_ff_all_ks_times_1(
        self, two_propanol_ua
    ):
        box_0 = mb.fill_box(
            compound=[two_propanol_ua], n_compounds=[1], box=[4, 4, 4]
        )

        charmm = Charmm(
            box_0,
            "gmso_periodic_dihedral_gomc",
            ff_filename="gmso_periodic_dihedral_gomc",
            residues=[two_propanol_ua.name],
            forcefield_selection={
                two_propanol_ua.name: get_mosdef_gomc_fn(
                    "gmso_two_propanol_periodic_dihedrals_ua_all_bond_angles_dihedrals_k_times_1.xml"
                ),
            },
            bead_to_atom_name_dict={"_CH3": "C", "_CH2": "C", "_HC": "C"},
            atom_type_naming_style="general",
        )
        charmm.write_inp()

        with open("gmso_periodic_dihedral_gomc.inp", "r") as fp:
            masses_read = False
            bonds_read = False
            angles_read = False
            dihedrals_read = False
            out_gomc = fp.readlines()
            for i, line in enumerate(out_gomc):
                if (
                    "! atom_types" in line
                    and "mass" in line
                    and "atomClass_ResidueName" in line
                ):
                    masses_read = True
                    atom_types_1 = [
                        ["*", "CH3", "15.035"],
                        ["*", "CH", "13.019"],
                        ["*", "O", "15.9994"],
                        ["*", "H", "1.008"],
                    ]
                    atom_types_2 = [
                        ["POL_CH3_sp3"],
                        ["POL_CH_O"],
                        ["POL_O"],
                        ["POL_H"],
                    ]

                    for j in range(0, len(atom_types_1)):
                        assert (
                            len(out_gomc[i + 1 + j].split("!")[0].split()) == 3
                        )
                        assert (
                            out_gomc[i + 1 + j].split("!")[0].split()[0:3]
                            == atom_types_1[j]
                        )
                        assert (
                            out_gomc[i + 1 + j].split()[4:5] == atom_types_2[j]
                        )

                elif (
                    "! type_1" in line
                    and "type_2" in line
                    and "Kb" in line
                    and "b0" in line
                    and "extended_type_1" in line
                    and "extended_type_2" in line
                ):
                    bonds_read = True
                    bond_types = [
                        ["CH3", "CH", "1200.803059", "1.54"],
                        ["CH", "O", "1200.803059", "1.43"],
                        ["O", "H", "1200.803059", "0.945"],
                    ]
                    total_bonds_evaluated = []
                    total_bonds_evaluated_reorg = []
                    for j in range(0, len(bond_types)):
                        assert (
                            len(out_gomc[i + 1 + j].split("!")[0].split()) == 4
                        )

                        if (
                            out_gomc[i + 1 + j].split("!")[0].split()[0:4]
                            == bond_types[0]
                            or bond_types[1]
                            or bond_types[2]
                        ):
                            total_bonds_evaluated.append(
                                out_gomc[i + 1 + j].split("!")[0].split()[0:4]
                            )
                    for k in range(0, len(bond_types)):
                        if bond_types[k] in total_bonds_evaluated:
                            total_bonds_evaluated_reorg.append(bond_types[k])
                    assert total_bonds_evaluated_reorg == bond_types

                elif (
                    "! type_1 " in line
                    and "type_2" in line
                    and "type_3" in line
                    and "Ktheta" in line
                    and "Theta0" in line
                    and "extended_type_1" in line
                    and "extended_type_2" in line
                    and "extended_type_3" in line
                ):
                    angles_read = True
                    angle_types = [
                        ["CH3", "CH", "O", "100.155088", "109.469889"],
                        ["CH3", "CH", "CH3", "124.200261", "112.000071"],
                        ["CH", "O", "H", "110.091109", "108.499872"],
                    ]
                    total_angles_evaluated = []
                    total_angles_evaluated_reorg = []
                    for j in range(0, len(angle_types)):
                        assert (
                            len(out_gomc[i + 1 + j].split("!")[0].split()) == 5
                        )
                        if (
                            out_gomc[i + 1 + j].split("!")[0].split()[0:5]
                            == angle_types[0]
                            or angle_types[1]
                            or angle_types[2]
                        ):
                            total_angles_evaluated.append(
                                out_gomc[i + 1 + j].split("!")[0].split()[0:5]
                            )
                    for k in range(0, len(angle_types)):
                        if angle_types[k] in total_angles_evaluated:
                            total_angles_evaluated_reorg.append(angle_types[k])

                    assert total_angles_evaluated_reorg == angle_types

                elif (
                    "! type_1" in line
                    and "type_2" in line
                    and "type_3" in line
                    and "type_4" in line
                    and "Kchi" in line
                    and "n" in line
                    and "delta" in line
                    and "extended_type_1" in line
                    and "extended_type_2" in line
                    and "extended_type_3" in line
                    and "extended_type_4" in line
                ):
                    dihedrals_read = True
                    dihedral_types = [
                        ["CH3", "CH", "O", "H", "-0.392135", "1", "180.0"],
                        ["CH3", "CH", "O", "H", "-0.062518", "2", "0.0"],
                        ["CH3", "CH", "O", "H", "0.345615", "3", "180.0"],
                    ]
                    for j in range(0, len(dihedral_types)):
                        assert (
                            len(out_gomc[i + 1 + j].split("!")[0].split()) == 7
                        )
                        assert (
                            out_gomc[i + 1 + j].split("!")[0].split()[0:7]
                            == dihedral_types[j]
                        )

        assert masses_read
        assert bonds_read
        assert angles_read
        assert dihedrals_read

    # test the gmso periodic wildcard dihderal input
    def test_save_gmso_periodic_wildcard_dihedral_gomc_ff(
        self, two_propanol_ua
    ):
        box_0 = mb.fill_box(
            compound=[two_propanol_ua], n_compounds=[1], box=[4, 4, 4]
        )

        charmm = Charmm(
            box_0,
            "gmso_periodic_wildcard_dihedral_gomc",
            ff_filename="gmso_periodic_wildcard_dihedral_gomc",
            residues=[two_propanol_ua.name],
            forcefield_selection={
                two_propanol_ua.name: get_mosdef_gomc_fn(
                    "gmso_two_propanol_periodic_wildcard_dihedrals_ua.xml"
                ),
            },
            bead_to_atom_name_dict={"_CH3": "C", "_CH2": "C", "_HC": "C"},
            atom_type_naming_style="general",
        )
        charmm.write_inp()

        with open("gmso_periodic_wildcard_dihedral_gomc.inp", "r") as fp:
            dihedrals_read = False
            out_gomc = fp.readlines()
            for i, line in enumerate(out_gomc):
                if (
                    "! type_1" in line
                    and "type_2" in line
                    and "type_3" in line
                    and "type_4" in line
                    and "Kchi" in line
                    and "n" in line
                    and "delta" in line
                    and "extended_type_1" in line
                    and "extended_type_2" in line
                    and "extended_type_3" in line
                    and "extended_type_4" in line
                ):
                    dihedrals_read = True
                    dihedral_types = [
                        ["CH3", "CH", "O", "H", "-0.392135", "1", "180.0"],
                        ["CH3", "CH", "O", "H", "-0.062518", "2", "0.0"],
                        ["CH3", "CH", "O", "H", "0.345615", "3", "180.0"],
                    ]
                    for j in range(0, len(dihedral_types)):
                        assert (
                            len(out_gomc[i + 1 + j].split("!")[0].split()) == 7
                        )
                        assert (
                            out_gomc[i + 1 + j].split("!")[0].split()[0:7]
                            == dihedral_types[j]
                        )

                else:
                    pass

        assert dihedrals_read

    # test the gmso harmonic dihderal input
    def test_save_gmso_harmonic_dihedral_gomc_ff(self, two_propanol_ua):
        with pytest.raises(
            TypeError,
            match=f"ERROR: The {'POL'} residue has a "
            f"{'HarmonicTorsionPotential'} torsion potential, which "
            f"is not currently supported in this writer.",
        ):
            box_0 = mb.fill_box(
                compound=[two_propanol_ua], n_compounds=[1], box=[4, 4, 4]
            )

            charmm = Charmm(
                box_0,
                "gmso_harmonic_dihedral_gomc",
                ff_filename="gmso_harmonic_dihedral_gomc",
                residues=[two_propanol_ua.name],
                forcefield_selection={
                    two_propanol_ua.name: get_mosdef_gomc_fn(
                        "gmso_two_propanol_harmonic_dihedrals_ua.xml"
                    ),
                },
                bead_to_atom_name_dict={"_CH3": "C", "_CH2": "C", "_HC": "C"},
            )
            charmm.write_inp()

    # test the gmso other bad_form dihderal input equation
    def test_save_gmso_bad_form_dihedral_gomc_ff(self, two_propanol_ua):
        with pytest.raises(
            TypeError,
            match=f"ERROR: The {'POL'} residue and associated force field "
            f"has at least one unsupported dihdedral. "
            f"The only supported dihedrals are "
            f"{'OPLSTorsionPotential'}, {'PeriodicTorsionPotential'}, and "
            f"{'RyckaertBellemansTorsionPotential'}.",
        ):
            box_0 = mb.fill_box(
                compound=[two_propanol_ua], n_compounds=[1], box=[4, 4, 4]
            )

            charmm = Charmm(
                box_0,
                "gmso_bad_form_dihedral_gomc",
                ff_filename="gmso_bad_form_dihedral_gomc",
                residues=[two_propanol_ua.name],
                forcefield_selection={
                    two_propanol_ua.name: get_mosdef_gomc_fn(
                        "gmso_two_propanol_bad_form_dihedrals_ua.xml"
                    ),
                },
                bead_to_atom_name_dict={"_CH3": "C", "_CH2": "C", "_HC": "C"},
            )
            charmm.write_inp()

    # test the gmso other bad_form bond input equation
    def test_save_gmso_bad_form_bonds_gomc_ff(self, two_propanol_ua):
        with pytest.raises(
            TypeError,
            match=f"ERROR: The {'POL'} residue's "
            f"bond types or classes does not have a "
            f"{'HarmonicBondPotential'} bond potential, which "
            f"is the only supported bond potential.",
        ):
            box_0 = mb.fill_box(
                compound=[two_propanol_ua], n_compounds=[1], box=[4, 4, 4]
            )

            charmm = Charmm(
                box_0,
                "gmso_bad_form_bonds_gomc",
                ff_filename="gmso_bad_form_bonds_gomc",
                residues=[two_propanol_ua.name],
                forcefield_selection={
                    two_propanol_ua.name: get_mosdef_gomc_fn(
                        "gmso_two_propanol_bad_form_bonds_ua.xml"
                    ),
                },
                bead_to_atom_name_dict={"_CH3": "C", "_CH2": "C", "_HC": "C"},
            )
            charmm.write_inp()

    # test the gmso other bad_form angle input equation
    def test_save_gmso_bad_form_angles_gomc_ff(self, two_propanol_ua):
        with pytest.raises(
            TypeError,
            match=f"ERROR: The {'POL'} residue's "
            f"angle types or classes does not have a "
            f"{'HarmonicAnglePotential'} angle potential, which "
            f"is the only supported angle potential.",
        ):
            box_0 = mb.fill_box(
                compound=[two_propanol_ua], n_compounds=[1], box=[4, 4, 4]
            )

            charmm = Charmm(
                box_0,
                "gmso_bad_form_angles_gomc",
                ff_filename="gmso_bad_form_angles_gomc",
                residues=[two_propanol_ua.name],
                forcefield_selection={
                    two_propanol_ua.name: get_mosdef_gomc_fn(
                        "gmso_two_propanol_bad_form_angles_ua.xml"
                    ),
                },
                bead_to_atom_name_dict={"_CH3": "C", "_CH2": "C", "_HC": "C"},
            )
            charmm.write_inp()

    # test bond, angle, and dihedral k-constants are the correct type and units
    def test_save_charmm_mie_ff_with_m_not_equal_6(self, two_propanol_ua):
        with pytest.raises(
            ValueError,
            match=f"ERROR: The Mie Potential atom class "
            f"{'POL'}_"
            f"{'O'} "
            f"does not have an m-constant of 6 in the force field XML, "
            f"which is required in GOMC and this file writer.",
        ):
            box_0 = mb.fill_box(
                compound=[two_propanol_ua], n_compounds=[1], box=[4, 4, 4]
            )

            Charmm(
                box_0,
                "charmm_mie_ff_with_m_not_equal_6",
                ff_filename="charmm_mie_ff_with_m_not_equal_6",
                residues=[two_propanol_ua.name],
                forcefield_selection={
                    two_propanol_ua.name: get_mosdef_gomc_fn(
                        "gmso_two_propanol_Mie_m_not_equal_6_ua.xml"
                    ),
                },
                bead_to_atom_name_dict={"_CH3": "C", "_CH2": "C", "_HC": "C"},
            )

    def test_atom_type_naming_style_not_correct(self, two_propanol_ua):
        with pytest.raises(
            TypeError,
            match=r"ERROR: Please enter the atom_type_naming_style "
            "as a string, either 'general' or 'all_unique'.",
        ):
            box_0 = mb.fill_box(
                compound=[two_propanol_ua], n_compounds=[1], box=[4, 4, 4]
            )

            Charmm(
                box_0,
                "charmm_data_UA",
                ff_filename="charmm_data_UA",
                residues=[two_propanol_ua.name],
                forcefield_selection="trappe-ua",
                bead_to_atom_name_dict={"_CH3": "C"},
                atom_type_naming_style="None",
            )

    def test_save_charmm_ua_gomc_ff_all_unique_atom_type_naming_style(
        self, two_propanol_ua
    ):
        box_0 = mb.fill_box(
            compound=[two_propanol_ua], n_compounds=[1], box=[4, 4, 4]
        )

        charmm = Charmm(
            box_0,
            "charmm_UA_all_unique_atom_type_naming_style",
            ff_filename="charmm_UA_all_unique_atom_type_naming_style",
            residues=[two_propanol_ua.name],
            forcefield_selection="trappe-ua",
            bead_to_atom_name_dict={"_CH3": "C"},
        )
        charmm.write_inp()

        with open("charmm_UA_all_unique_atom_type_naming_style.inp", "r") as fp:
            masses_read = False
            out_gomc = fp.readlines()
            for i, line in enumerate(out_gomc):
                if (
                    "! atom_types" in line
                    and "mass" in line
                    and "atomClass_ResidueName" in line
                ):
                    masses_read = True
                    atom_types_1 = [
                        ["*", "CH30", "15.035"],
                        ["*", "CH0", "13.019"],
                        ["*", "O0", "15.9994"],
                        ["*", "H0", "1.008"],
                    ]
                    atom_types_2 = [
                        ["POL_CH3_sp3"],
                        ["POL_CH_O"],
                        ["POL_O"],
                        ["POL_H"],
                    ]

                    for j in range(0, len(atom_types_1)):
                        assert (
                            len(out_gomc[i + 1 + j].split("!")[0].split()) == 3
                        )
                        assert (
                            out_gomc[i + 1 + j].split("!")[0].split()[0:3]
                            == atom_types_1[j]
                        )
                        assert (
                            out_gomc[i + 1 + j].split()[4:5] == atom_types_2[j]
                        )
        assert masses_read

    # test the periodic dihedrals can not have the same 1 values per dihedral
    def test_save_charmm_Mie_periodic_dihedral_same_n_values_ff(
        self, water, two_propanol_ua
    ):
        with pytest.raises(
            ValueError,
            match=f"ERROR: mulitple PeriodicTorsionPotential "
            f"n values of {'1.0'} "
            f"were found for the same torsion. Only 1 of each "
            f"n values are allowed per PeriodicTorsionPotential.",
        ):
            box_0 = mb.fill_box(
                compound=[water, two_propanol_ua],
                n_compounds=[4, 4],
                box=[5, 4, 3],
            )

            charmm = Charmm(
                box_0,
                "charmm_mie_ua_K_energy_units_periodic_with_same_n_values",
                ff_filename="charmm_mie_ua_K_energy_units_periodic_with_same_n_values",
                residues=[water.name, two_propanol_ua.name],
                forcefield_selection={
                    water.name: get_mosdef_gomc_fn(
                        "gmso_spce_water__lorentz_combining.xml"
                    ),
                    two_propanol_ua.name: get_mosdef_gomc_fn(
                        "gmso_two_propanol_Mie_periodic_dihedral_ua_dihedral_with_same_n_values.xml"
                    ),
                },
                bead_to_atom_name_dict={"_CH3": "C", "_CH2": "C", "_HC": "C"},
                gomc_fix_bonds_angles=[water.name],
            )
            charmm.write_inp()

    def test_save_charmm_gomc_ua_charmm_periodic_improper_ff(
        self, two_propanol_ua
    ):
        molecule = two_propanol_ua
        molecule.box = mb.Box(lengths=[4, 4, 4])
        molecule.name = "POL"

        charmm = Charmm(
            molecule,
            "charmm_gomc_ua_charmm_periodic_improper_ff",
            ff_filename="charmm_gomc_ua_charmm_periodic_improper_ff",
            residues=[molecule.name],
            forcefield_selection=get_mosdef_gomc_fn(
                "gmso_two_propanol_periodic_impropers_ua.xml"
            ),
            atom_type_naming_style="general",
        )
        charmm.write_inp()

        with open("charmm_gomc_ua_charmm_periodic_improper_ff.inp", "r") as fp:
            impropers_read = False
            out_gomc = fp.readlines()
            for i, line in enumerate(out_gomc):
                if (
                    "! type_1" in line
                    and "type_2" in line
                    and "type_3" in line
                    and "type_4" in line
                    and "Kw" in line
                    and "n" in line
                    and "w0" in line
                    and "extended_type_1" in line
                    and "extended_type_2" in line
                    and "extended_type_3" in line
                    and "extended_type_4" in line
                ):
                    impropers_read = True
                    improp_types = [
                        ["CH", "O", "CH3", "CH3", "-2.0", "1", "180.0"],
                        ["CH", "O", "CH3", "CH3", "1.0", "3", "0.0"],
                    ]
                    for j in range(0, len(improp_types)):
                        assert (
                            len(out_gomc[i + 1 + j].split("!")[0].split()) == 7
                        )
                        assert (
                            out_gomc[i + 1 + j].split("!")[0].split()[0:7]
                            == improp_types[j]
                        )

                else:
                    pass

        assert impropers_read

    def test_save_charmm_gomc_ua_charmm_periodic_wildcard_improper_ff(
        self, two_propanol_ua
    ):
        molecule = two_propanol_ua
        molecule.box = mb.Box(lengths=[4, 4, 4])
        molecule.name = "POL"

        charmm = Charmm(
            molecule,
            "charmm_gomc_ua_charmm_periodic_wildcard_improper_ff",
            ff_filename="charmm_gomc_ua_charmm_periodic_wildcard_improper_ff",
            residues=[molecule.name],
            forcefield_selection=get_mosdef_gomc_fn(
                "gmso_two_propanol_periodic_wildcard_impropers_ua.xml"
            ),
            atom_type_naming_style="general",
        )
        charmm.write_inp()

        with open(
            "charmm_gomc_ua_charmm_periodic_wildcard_improper_ff.inp", "r"
        ) as fp:
            impropers_read = False
            out_gomc = fp.readlines()
            for i, line in enumerate(out_gomc):
                if (
                    "! type_1" in line
                    and "type_2" in line
                    and "type_3" in line
                    and "type_4" in line
                    and "Kw" in line
                    and "n" in line
                    and "w0" in line
                    and "extended_type_1" in line
                    and "extended_type_2" in line
                    and "extended_type_3" in line
                    and "extended_type_4" in line
                ):
                    impropers_read = True
                    improp_types = set(
                        [
                            ("CH", "CH3", "O", "CH3"),
                            ("CH", "O", "CH3", "CH3"),
                        ]
                    )
                    improp_params = set(
                        [
                            ("-2.0", "1", "180.0"),
                            ("1.0", "3", "0.0"),
                        ]
                    )
                    for j in range(len(improp_types)):
                        assert (
                            len(out_gomc[i + 1 + j].split("!")[0].split()) == 7
                        )
                        assert (
                            tuple(
                                out_gomc[i + 1 + j].split("!")[0].split()[0:4]
                            )
                            in improp_types
                        )
                        assert (
                            tuple(
                                out_gomc[i + 1 + j].split("!")[0].split()[4:7]
                            )
                            in improp_params
                        )

                else:
                    pass

        assert impropers_read

    def test_save_Mie_gomc_ua_charmm_periodic_improper_ff(
        self, two_propanol_ua
    ):
        with pytest.raises(
            ValueError,
            match=f"ERROR: Currently, the Mie and Exp6 potentials do not support impropers.",
        ):
            molecule = two_propanol_ua
            molecule.box = mb.Box(lengths=[4, 4, 4])
            molecule.name = "POL"

            charmm = Charmm(
                molecule,
                "charmm_gomc_ua_Mie_periodic_improper_ff",
                ff_filename="charmm_gomc_ua_Mie_periodic_improper_ff",
                residues=[molecule.name],
                forcefield_selection=get_mosdef_gomc_fn(
                    "gmso_two_propanol_Mie_periodic_impropers_ua.xml"
                ),
                atom_type_naming_style="general",
            )
            charmm.write_inp()

    def test_save_charmm_gomc_ua_charmm_harmonic_improper_ff(
        self, two_propanol_ua
    ):
        with pytest.raises(
            TypeError,
            match=f"ERROR: The {'POL'} residue has a "
            f"{'HarmonicImproperPotential'} torsion potential, which "
            f"is not currently supported in this writer.",
        ):
            molecule = two_propanol_ua
            molecule.box = mb.Box(lengths=[4, 4, 4])
            molecule.name = "POL"

            charmm = Charmm(
                molecule,
                "charmm_gomc_ua_charmm_harmonic_improper_ff",
                ff_filename="charmm_gomc_ua_charmm_harmonic_improper_ff",
                residues=[molecule.name],
                forcefield_selection=get_mosdef_gomc_fn(
                    "gmso_two_propanol_harmonic_impropers_ua.xml"
                ),
                atom_type_naming_style="general",
            )
            charmm.write_inp()

    def test_gmso_two_propanol_harmonic_impropers_ua_with_2_harmonics_in_2_diff_forms(
        self, two_propanol_ua
    ):
        with pytest.raises(
            TypeError,
            match=f"ERROR: The supplied force field xml for the POL residue is not a foyer or gmso xml, "
            f"or the xml has errors and it not able to load properly.",
        ):
            molecule = two_propanol_ua
            molecule.box = mb.Box(lengths=[4, 4, 4])
            molecule.name = "POL"

            charmm = Charmm(
                molecule,
                "charmm_gomc_ua_charmm_harmonic_improper_ff",
                ff_filename="charmm_gomc_ua_charmm_harmonic_improper_ff",
                residues=[molecule.name],
                forcefield_selection=get_mosdef_gomc_fn(
                    "gmso_two_propanol_harmonic_impropers_ua_with_2_harmonics_in_2_diff_forms.xml"
                ),
                atom_type_naming_style="general",
            )
            charmm.write_inp()

    def test_save_charmm_ua_bonded_types_and_classes_xml_ff(
        self, water, two_propanol_ua
    ):
        box_0 = mb.fill_box(
            compound=[water, two_propanol_ua], n_compounds=[1, 1], box=[5, 4, 3]
        )

        outside_box = mb.Compound()
        outside_box.box = mb.Box(lengths=[6, 6, 6])

        charmm = Charmm(
            box_0,
            "charmm_ua_bonded_types_not_classes_xml_ff",
            ff_filename="charmm_ua_bonded_types_not_classes_xml_ff",
            residues=[water.name, two_propanol_ua.name],
            forcefield_selection={
                water.name: get_mosdef_gomc_fn(
                    "gmso_spce_water_one_for_nb_and_coul__lorentz_combining.xml"
                ),
                two_propanol_ua.name: get_mosdef_gomc_fn(
                    "gmso_two_propanol_some_bonded_types_and_classes.xml"
                ),
            },
            bead_to_atom_name_dict={"_CH3": "C", "_CH2": "C", "_HC": "C"},
            gomc_fix_bonds_angles=[water.name],
            atom_type_naming_style="general",
        )
        charmm.write_inp()

        with open("charmm_ua_bonded_types_not_classes_xml_ff.inp", "r") as fp:
            masses_read = False
            bonds_read = False
            angles_read = False
            dihedrals_read = False
            impropers_read = False
            nonbondeds_read = False
            out_gomc = fp.readlines()
            for i, line in enumerate(out_gomc):
                if (
                    "! atom_types" in line
                    and "mass" in line
                    and "atomClass_ResidueName" in line
                ):
                    masses_read = True
                    atom_types_1 = [
                        ["*", "OW", "15.999"],
                        ["*", "HW", "1.008"],
                        ["*", "CH3", "15.035"],
                        ["*", "CH", "13.019"],
                        ["*", "O", "15.9994"],
                        ["*", "H", "1.008"],
                    ]
                    atom_types_2 = [
                        ["WAT_o_spce"],
                        ["WAT_h_spce"],
                        ["POL_CH3_sp3"],
                        ["POL_CH_O"],
                        ["POL_O"],
                        ["POL_H"],
                    ]

                    for j in range(0, len(atom_types_1)):
                        assert (
                            len(out_gomc[i + 1 + j].split("!")[0].split()) == 3
                        )
                        assert (
                            out_gomc[i + 1 + j].split("!")[0].split()[0:3]
                            == atom_types_1[j]
                        )
                        assert (
                            out_gomc[i + 1 + j].split()[4:5] == atom_types_2[j]
                        )

                elif (
                    "! type_1" in line
                    and "type_2" in line
                    and "Kb" in line
                    and "b0" in line
                    and "extended_type_1" in line
                    and "extended_type_2" in line
                ):
                    bonds_read = True
                    bond_types = [
                        ["OW", "HW", "999999999999", "1.0"],
                        ["CH3", "CH", "604267.555311", "1.54"],
                        ["CH", "O", "604267.555311", "1.43"],
                        ["O", "H", "604267.555311", "0.945"],
                    ]
                    total_bonds_evaluated = []
                    total_bonds_evaluated_reorg = []
                    for j in range(0, len(bond_types)):
                        assert (
                            len(out_gomc[i + 1 + j].split("!")[0].split()) == 4
                        )

                        if (
                            out_gomc[i + 1 + j].split("!")[0].split()[0:4]
                            == bond_types[0]
                            or bond_types[1]
                            or bond_types[2]
                        ):
                            total_bonds_evaluated.append(
                                out_gomc[i + 1 + j].split("!")[0].split()[0:4]
                            )
                    for k in range(0, len(bond_types)):
                        if bond_types[k] in total_bonds_evaluated:
                            total_bonds_evaluated_reorg.append(bond_types[k])
                    assert total_bonds_evaluated_reorg == bond_types

                elif (
                    "! type_1 " in line
                    and "type_2" in line
                    and "type_3" in line
                    and "Ktheta" in line
                    and "Theta0" in line
                    and "extended_type_1" in line
                    and "extended_type_2" in line
                    and "extended_type_3" in line
                ):
                    angles_read = True
                    angle_types = [
                        ["HW", "OW", "HW", "999999999999", "109.47"],
                        ["CH3", "CH", "O", "50400.0", "109.5"],
                        ["CH3", "CH", "CH3", "62500.0", "112.0"],
                        ["CH", "O", "H", "55400.0", "108.5"],
                    ]
                    total_angles_evaluated = []
                    total_angles_evaluated_reorg = []
                    for j in range(0, len(angle_types)):
                        assert (
                            len(out_gomc[i + 1 + j].split("!")[0].split()) == 5
                        )
                        if (
                            out_gomc[i + 1 + j].split("!")[0].split()[0:5]
                            == angle_types[0]
                            or angle_types[1]
                            or angle_types[2]
                        ):
                            total_angles_evaluated.append(
                                out_gomc[i + 1 + j].split("!")[0].split()[0:5]
                            )
                    for k in range(0, len(angle_types)):
                        if angle_types[k] in total_angles_evaluated:
                            total_angles_evaluated_reorg.append(angle_types[k])

                    assert total_angles_evaluated_reorg == angle_types

                elif (
                    "! type_1" in line
                    and "type_2" in line
                    and "type_3" in line
                    and "type_4" in line
                    and "Kchi" in line
                    and "n" in line
                    and "delta" in line
                    and "extended_type_1" in line
                    and "extended_type_2" in line
                    and "extended_type_3" in line
                    and "extended_type_4" in line
                ):
                    dihedrals_read = True
                    dihedral_types = [
                        ["CH3", "CH", "O", "H", "15.0", "1", "180.0"],
                        ["CH3", "CH", "O", "H", "10.0", "2", "0.0"],
                        ["CH3", "CH", "O", "H", "5.0", "3", "180.0"],
                    ]
                    for j in range(0, len(dihedral_types)):
                        assert (
                            len(out_gomc[i + 1 + j].split("!")[0].split()) == 7
                        )
                        assert (
                            out_gomc[i + 1 + j].split("!")[0].split()[0:7]
                            == dihedral_types[j]
                        )

                elif (
                    "! type_1" in line
                    and "type_2" in line
                    and "type_3" in line
                    and "type_4" in line
                    and "Kw" in line
                    and "n" in line
                    and "w0" in line
                    and "extended_type_1" in line
                    and "extended_type_2" in line
                    and "extended_type_3" in line
                    and "extended_type_4" in line
                ):
                    impropers_read = True
                    improp_types = [
                        ["CH", "O", "CH3", "CH3", "-8.0", "1", "180.0"],
                        ["CH", "O", "CH3", "CH3", "4.0", "3", "0.0"],
                    ]
                    for j in range(0, len(improp_types)):
                        assert (
                            len(out_gomc[i + 1 + j].split("!")[0].split()) == 7
                        )
                        assert (
                            out_gomc[i + 1 + j].split("!")[0].split()[0:7]
                            == improp_types[j]
                        )

                elif (
                    "! type_1" in line
                    and "ignored" in line
                    and "epsilon" in line
                    and "Rmin/2" in line
                    and "ignored" in line
                    and "epsilon,1-4" in line
                    and "Rmin/2,1-4" in line
                    and "extended_type_1" in line
                    and "extended_type_2" in line
                ):
                    nonbondeds_read = True
                    nb_types = [
                        [
                            "OW",
                            "0.0",
                            "-0.1554000956",
                            "1.7766160931",
                            "0.0",
                            "-0.1554000956",
                            "1.7766160931",
                        ],
                        [
                            "HW",
                            "0.0",
                            "-0.0",
                            "0.5612310242",
                            "0.0",
                            "-0.0",
                            "0.5612310242",
                        ],
                        [
                            "CH3",
                            "0.0",
                            "-98.0",
                            "2.1046163406",
                            "0.0",
                            "-98.0",
                            "2.1046163406",
                        ],
                        [
                            "CH",
                            "0.0",
                            "-10.0",
                            "2.626561193",
                            "0.0",
                            "-10.0",
                            "2.626561193",
                        ],
                        [
                            "O",
                            "0.0",
                            "-93.0",
                            "1.6949176929",
                            "0.0",
                            "-93.0",
                            "1.6949176929",
                        ],
                        [
                            "H",
                            "0.0",
                            "-0.0",
                            "0.0",
                            "0.0",
                            "-0.0",
                            "0.0",
                        ],
                    ]

                    for j in range(0, len(nb_types)):
                        assert (
                            len(out_gomc[i + 1 + j].split("!")[0].split()) == 7
                        )
                        assert (
                            out_gomc[i + 1 + j].split("!")[0].split()[0:7]
                            == nb_types[j]
                        )

                else:
                    pass

        assert masses_read
        assert bonds_read
        assert angles_read
        assert dihedrals_read
        assert impropers_read
        assert nonbondeds_read

    def test_residues_not_correct_with_gmso_forcefielding(
        self, two_propanol_ua
    ):
        with pytest.raises(
            ValueError,
            match=f"ERROR: There is something wrong with the MoSDeF-GOMC inputs when run through GMSO. "
            f"It could be something else, but please check the following also. "
            f"All the residues are not specified in the residue list, or "
            f"the entered residues does not match the residues that "
            f"were found in the foyer and GMSO force field application. "
            f"The residues were not used from the forcefield_selection string or dictionary. "
            "All the residues were not used from the forcefield_selection "
            "string or dictionary. There may be residues below other "
            "specified residues in the mbuild.Compound hierarchy. "
            "If so, all the highest listed residues pass down the force "
            "fields through the hierarchy. Alternatively, residues that "
            "are not in the structure may have been specified. ",
        ):
            molecule = two_propanol_ua
            molecule.box = mb.Box(lengths=[4, 4, 4])
            molecule.name = "POL"

            methane_ua_bead_name = "_CH4"
            methane_child_bead = mb.Compound(name=methane_ua_bead_name)
            methane_box = mb.fill_box(
                compound=methane_child_bead, n_compounds=4, box=[1, 2, 3]
            )
            methane_box.name = "MET"

            charmm = Charmm(
                methane_box,
                "methane_box_compound_and_subcompound",
                ff_filename="methane_box_compound_and_subcompound_FF",
                forcefield_selection={methane_box.name: "trappe-ua"},
                residues=[methane_box.name],
                bead_to_atom_name_dict={methane_ua_bead_name: "C"},
                gmso_match_ff_by="molecule",
                atom_type_naming_style="general",
            )

    def test_atom_type_style_general_warning(
        self, two_propanol_ua, alt_two_propanol_ua
    ):
        with pytest.warns(
            UserWarning,
            match=f"WARNING: atom_type_naming_style = 'general'\n"
            f"WARNING: The 'general' convention is UNSAFE, and the EXPERT user SHOULD USE AT THEIR OWN RISK, "
            f"making SURE ALL THE BONDED PARAMETERS HAVE THE SAME VALUES IN THE UTILIZED "
            f"FORCE FIELD XMLs.  Also, this DOES NOT ENSURE that THERE ARE NO specific "
            f"Foyer XML ATOM TYPE BONDED CONNECTIONS in the Foyer FORCE FIELD XMLs, instead of the Foyer "
            f"atom class type bonded connections, which could RESULT IN AN INCORRECT FORCE FIELD "
            f"PARAMETERIZATION.  This is UNSAFE to use even with the same force field XML file, so the "
            f"EXPERT user SHOULD USE AT THEIR OWN RISK.\n"
            f"The 'general' convention only tests if the sigma, epsilons, mass, and Mie-n values are "
            f"identical between the different molecules \(residues in this context\) and their applied "
            f"force fields and DOES NOT check that any or all of the bonded parameters have the same "
            f"or conflicting values. ",
        ):
            methane_box = mb.fill_box(
                compound=[two_propanol_ua, alt_two_propanol_ua],
                n_compounds=[1, 1],
                box=[3, 3, 3],
            )

            charmm = Charmm(
                methane_box,
                "test_atom_type_style_general_passes_tests",
                ff_filename="test_atom_typ_style_general_passes_tests",
                forcefield_selection={
                    two_propanol_ua.name: get_mosdef_gomc_fn(
                        "gmso_two_propanol_periodic_dihedrals_ua_all_bond_angles_dihedrals_k_times_half.xml"
                    ),
                    alt_two_propanol_ua.name: get_mosdef_gomc_fn(
                        "gmso_two_propanol_periodic_dihedrals_ua_all_bond_angles_dihedrals_k_times_half.xml"
                    ),
                },
                residues=[two_propanol_ua.name, alt_two_propanol_ua.name],
                gmso_match_ff_by="molecule",
                bead_to_atom_name_dict={"_CH3": "C", "_CH2": "C", "_HC": "C"},
                atom_type_naming_style="general",
            )

    def test_atom_type_style_general(
        self, two_propanol_ua, alt_two_propanol_ua
    ):
        box_0 = mb.fill_box(
            compound=[two_propanol_ua, alt_two_propanol_ua],
            n_compounds=[1, 1],
            box=[3, 3, 3],
        )

        charmm = Charmm(
            box_0,
            "test_atom_type_style_general",
            ff_filename="test_atom_type_style_general",
            forcefield_selection={
                two_propanol_ua.name: get_mosdef_gomc_fn(
                    "gmso_two_propanol_Mie_periodic_dihedral_ua_K_energy_units.xml"
                ),
                alt_two_propanol_ua.name: get_mosdef_gomc_fn(
                    "gmso_two_propanol_Mie_periodic_dihedral_ua_K_energy_units.xml"
                ),
            },
            residues=[two_propanol_ua.name, alt_two_propanol_ua.name],
            gmso_match_ff_by="molecule",
            bead_to_atom_name_dict={"_CH3": "C", "_CH2": "C", "_HC": "C"},
            atom_type_naming_style="general",
        )
        charmm.write_inp()

        with open("test_atom_type_style_general.inp", "r") as fp:
            masses_read = False
            bonds_read = False
            angles_read = False
            dihedrals_read = False
            nonbondeds_read = False
            out_gomc = fp.readlines()
            for i, line in enumerate(out_gomc):
                if (
                    "! atom_types" in line
                    and "mass" in line
                    and "atomClass_ResidueName" in line
                ):
                    masses_read = True
                    atom_types_1 = [
                        ["*", "CH3", "15.035"],
                        ["*", "CH", "13.019"],
                        ["*", "O", "15.9994"],
                        ["*", "H", "1.008"],
                    ]
                    atom_types_2 = [
                        ["POL_CH3_sp3"],
                        ["POL_CH_O"],
                        ["POL_O"],
                        ["POL_H"],
                    ]

                    for j in range(0, len(atom_types_1)):
                        assert (
                            len(out_gomc[i + 1 + j].split("!")[0].split()) == 3
                        )
                        assert (
                            out_gomc[i + 1 + j].split("!")[0].split()[0:3]
                            == atom_types_1[j]
                        )
                        assert (
                            out_gomc[i + 1 + j].split()[4:5] == atom_types_2[j]
                        )

                elif (
                    "! type_1" in line
                    and "type_2" in line
                    and "Kb" in line
                    and "b0" in line
                    and "extended_type_1" in line
                    and "extended_type_2" in line
                ):
                    bonds_read = True
                    bond_types = [
                        ["CH3", "CH", "604267.5553", "1.54"],
                        ["CH", "O", "604267.5553", "1.43"],
                        ["O", "H", "604267.5553", "0.945"],
                    ]
                    total_bonds_evaluated = []
                    total_bonds_evaluated_reorg = []
                    for j in range(0, len(bond_types)):
                        assert (
                            len(out_gomc[i + 1 + j].split("!")[0].split()) == 4
                        )

                        if (
                            out_gomc[i + 1 + j].split("!")[0].split()[0:4]
                            == bond_types[0]
                            or bond_types[1]
                            or bond_types[2]
                        ):
                            total_bonds_evaluated.append(
                                out_gomc[i + 1 + j].split("!")[0].split()[0:4]
                            )
                    for k in range(0, len(bond_types)):
                        if bond_types[k] in total_bonds_evaluated:
                            total_bonds_evaluated_reorg.append(bond_types[k])
                    assert total_bonds_evaluated_reorg == bond_types

                elif (
                    "! type_1 " in line
                    and "type_2" in line
                    and "type_3" in line
                    and "Ktheta" in line
                    and "Theta0" in line
                    and "extended_type_1" in line
                    and "extended_type_2" in line
                    and "extended_type_3" in line
                ):
                    angles_read = True
                    angle_types = [
                        ["CH3", "CH", "O", "50400.0", "109.5"],
                        ["CH3", "CH", "CH3", "62500.0", "112.0"],
                        ["CH", "O", "H", "55400.0", "108.5"],
                    ]
                    total_angles_evaluated = []
                    total_angles_evaluated_reorg = []
                    for j in range(0, len(angle_types)):
                        assert (
                            len(out_gomc[i + 1 + j].split("!")[0].split()) == 5
                        )
                        if (
                            out_gomc[i + 1 + j].split("!")[0].split()[0:5]
                            == angle_types[0]
                            or angle_types[1]
                            or angle_types[2]
                        ):
                            total_angles_evaluated.append(
                                out_gomc[i + 1 + j].split("!")[0].split()[0:5]
                            )
                    for k in range(0, len(angle_types)):
                        if angle_types[k] in total_angles_evaluated:
                            total_angles_evaluated_reorg.append(angle_types[k])

                    assert total_angles_evaluated_reorg == angle_types

                elif (
                    "! type_1" in line
                    and "type_2" in line
                    and "type_3" in line
                    and "type_4" in line
                    and "Kchi" in line
                    and "n" in line
                    and "delta" in line
                    and "extended_type_1" in line
                    and "extended_type_2" in line
                    and "extended_type_3" in line
                    and "extended_type_4" in line
                ):
                    dihedrals_read = True
                    dihedral_types = [
                        ["CH3", "CH", "O", "H", "-18.75", "0", "90.0"],
                        ["CH3", "CH", "O", "H", "10.0", "1", "180.0"],
                        ["CH3", "CH", "O", "H", "-10.0", "2", "0.0"],
                        ["CH3", "CH", "O", "H", "10.0", "3", "180.0"],
                        ["CH3", "CH", "O", "H", "-0.625", "4", "0.0"],
                    ]
                    for j in range(0, len(dihedral_types)):
                        assert (
                            len(out_gomc[i + 1 + j].split("!")[0].split()) == 7
                        )
                        assert (
                            out_gomc[i + 1 + j].split("!")[0].split()[0:7]
                            == dihedral_types[j]
                        )

                elif (
                    "! type_1" in line
                    and "epsilon" in line
                    and "sigma" in line
                    and "n" in line
                    and "epsilon,1-4" in line
                    and "sigma,1-4" in line
                    and "n,1-4" in line
                    and "extended_type_1" in line
                    and "extended_type_2" in line
                ):
                    nonbondeds_read = True
                    nb_types = [
                        [
                            "CH3",
                            "98.0",
                            "3.75",
                            "11.0",
                            "0.0",
                            "3.75",
                            "11.0",
                        ],
                        [
                            "CH",
                            "10.0",
                            "4.68",
                            "12.0",
                            "0.0",
                            "4.68",
                            "12.0",
                        ],
                        [
                            "O",
                            "93.0",
                            "3.02",
                            "13.0",
                            "0.0",
                            "3.02",
                            "13.0",
                        ],
                        [
                            "H",
                            "0.0",
                            "0.0",
                            "14.0",
                            "0.0",
                            "0.0",
                            "14.0",
                        ],
                    ]

                    for j in range(0, len(nb_types)):
                        assert (
                            len(out_gomc[i + 1 + j].split("!")[0].split()) == 7
                        )
                        assert (
                            out_gomc[i + 1 + j].split("!")[0].split()[0:7]
                            == nb_types[j]
                        )

                else:
                    pass

        assert masses_read
        assert bonds_read
        assert angles_read
        assert dihedrals_read
        assert nonbondeds_read

    def test_atom_type_style_all_unique_diff_epsilon(
        self, two_propanol_ua, alt_two_propanol_ua
    ):
        box_0 = mb.fill_box(
            compound=[two_propanol_ua, alt_two_propanol_ua],
            n_compounds=[1, 1],
            box=[3, 3, 3],
        )

        charmm = Charmm(
            box_0,
            "test_atom_type_style_all_unique",
            ff_filename="test_atom_type_style_all_unique",
            forcefield_selection={
                two_propanol_ua.name: get_mosdef_gomc_fn(
                    "gmso_two_propanol_Mie_periodic_dihedral_ua_K_energy_units.xml"
                ),
                alt_two_propanol_ua.name: get_mosdef_gomc_fn(
                    "gmso_two_propanol_Mie_periodic_dihedral_ua_K_energy_units_diff_eplison.xml"
                ),
            },
            residues=[two_propanol_ua.name, alt_two_propanol_ua.name],
            gmso_match_ff_by="molecule",
            bead_to_atom_name_dict={"_CH3": "C", "_CH2": "C", "_HC": "C"},
            atom_type_naming_style="general",
        )
        charmm.write_inp()

        with open("test_atom_type_style_all_unique.inp", "r") as fp:
            masses_read = False
            bonds_read = False
            angles_read = False
            dihedrals_read = False
            nonbondeds_read = False
            out_gomc = fp.readlines()
            for i, line in enumerate(out_gomc):
                if (
                    "! atom_types" in line
                    and "mass" in line
                    and "atomClass_ResidueName" in line
                ):
                    masses_read = True
                    atom_types_1 = [
                        ["*", "CH30", "15.035"],
                        ["*", "CH0", "13.019"],
                        ["*", "O0", "15.9994"],
                        ["*", "H0", "1.008"],
                        ["*", "CH31", "15.035"],
                        ["*", "CH1", "13.019"],
                        ["*", "O1", "15.9994"],
                        ["*", "H1", "1.008"],
                    ]
                    atom_types_2 = [
                        ["POL_CH3_sp3"],
                        ["POL_CH_O"],
                        ["POL_O"],
                        ["POL_H"],
                        ["SPL_CH3_sp3"],
                        ["SPL_CH_O_diff"],
                        ["SPL_O"],
                        ["SPL_H"],
                    ]

                    for j in range(0, len(atom_types_1)):
                        assert (
                            len(out_gomc[i + 1 + j].split("!")[0].split()) == 3
                        )
                        assert (
                            out_gomc[i + 1 + j].split("!")[0].split()[0:3]
                            == atom_types_1[j]
                        )
                        assert (
                            out_gomc[i + 1 + j].split()[4:5] == atom_types_2[j]
                        )

                elif (
                    "! type_1" in line
                    and "type_2" in line
                    and "Kb" in line
                    and "b0" in line
                    and "extended_type_1" in line
                    and "extended_type_2" in line
                ):
                    bonds_read = True
                    bond_types = [
                        ["CH30", "CH0", "604267.5553", "1.54"],
                        ["CH0", "O0", "604267.5553", "1.43"],
                        ["O0", "H0", "604267.5553", "0.945"],
                        ["CH31", "CH1", "604267.5553", "1.54"],
                        ["CH1", "O1", "604267.5553", "1.43"],
                        ["O1", "H1", "604267.5553", "0.945"],
                    ]
                    total_bonds_evaluated = []
                    total_bonds_evaluated_reorg = []
                    for j in range(0, len(bond_types)):
                        assert (
                            len(out_gomc[i + 1 + j].split("!")[0].split()) == 4
                        )

                        if (
                            out_gomc[i + 1 + j].split("!")[0].split()[0:4]
                            == bond_types[0]
                            or bond_types[1]
                            or bond_types[2]
                        ):
                            total_bonds_evaluated.append(
                                out_gomc[i + 1 + j].split("!")[0].split()[0:4]
                            )
                    for k in range(0, len(bond_types)):
                        if bond_types[k] in total_bonds_evaluated:
                            total_bonds_evaluated_reorg.append(bond_types[k])
                    assert total_bonds_evaluated_reorg == bond_types

                elif (
                    "! type_1 " in line
                    and "type_2" in line
                    and "type_3" in line
                    and "Ktheta" in line
                    and "Theta0" in line
                    and "extended_type_1" in line
                    and "extended_type_2" in line
                    and "extended_type_3" in line
                ):
                    angles_read = True
                    angle_types = [
                        ["CH30", "CH0", "O0", "50400.0", "109.5"],
                        ["CH30", "CH0", "CH30", "62500.0", "112.0"],
                        ["CH0", "O0", "H0", "55400.0", "108.5"],
                        ["CH31", "CH1", "O1", "50400.0", "109.5"],
                        ["CH31", "CH1", "CH31", "62500.0", "112.0"],
                        ["CH1", "O1", "H1", "55400.0", "108.5"],
                    ]
                    total_angles_evaluated = []
                    total_angles_evaluated_reorg = []
                    for j in range(0, len(angle_types)):
                        assert (
                            len(out_gomc[i + 1 + j].split("!")[0].split()) == 5
                        )
                        if (
                            out_gomc[i + 1 + j].split("!")[0].split()[0:5]
                            == angle_types[0]
                            or angle_types[1]
                            or angle_types[2]
                        ):
                            total_angles_evaluated.append(
                                out_gomc[i + 1 + j].split("!")[0].split()[0:5]
                            )
                    for k in range(0, len(angle_types)):
                        if angle_types[k] in total_angles_evaluated:
                            total_angles_evaluated_reorg.append(angle_types[k])

                    assert total_angles_evaluated_reorg == angle_types

                elif (
                    "! type_1" in line
                    and "type_2" in line
                    and "type_3" in line
                    and "type_4" in line
                    and "Kchi" in line
                    and "n" in line
                    and "delta" in line
                    and "extended_type_1" in line
                    and "extended_type_2" in line
                    and "extended_type_3" in line
                    and "extended_type_4" in line
                ):
                    dihedrals_read = True
                    dihedral_types = [
                        ["CH30", "CH0", "O0", "H0", "-18.75", "0", "90.0"],
                        ["CH30", "CH0", "O0", "H0", "10.0", "1", "180.0"],
                        ["CH30", "CH0", "O0", "H0", "-10.0", "2", "0.0"],
                        ["CH30", "CH0", "O0", "H0", "10.0", "3", "180.0"],
                        ["CH30", "CH0", "O0", "H0", "-0.625", "4", "0.0"],
                        ["CH31", "CH1", "O1", "H1", "-18.75", "0", "90.0"],
                        ["CH31", "CH1", "O1", "H1", "10.0", "1", "180.0"],
                        ["CH31", "CH1", "O1", "H1", "-10.0", "2", "0.0"],
                        ["CH31", "CH1", "O1", "H1", "10.0", "3", "180.0"],
                        ["CH31", "CH1", "O1", "H1", "-0.625", "4", "0.0"],
                    ]
                    for j in range(0, len(dihedral_types)):
                        assert (
                            len(out_gomc[i + 1 + j].split("!")[0].split()) == 7
                        )
                        assert (
                            out_gomc[i + 1 + j].split("!")[0].split()[0:7]
                            == dihedral_types[j]
                        )

                elif (
                    "! type_1" in line
                    and "epsilon" in line
                    and "sigma" in line
                    and "n" in line
                    and "epsilon,1-4" in line
                    and "sigma,1-4" in line
                    and "n,1-4" in line
                    and "extended_type_1" in line
                    and "extended_type_2" in line
                ):
                    nonbondeds_read = True
                    nb_types = [
                        [
                            "CH30",
                            "98.0",
                            "3.75",
                            "11.0",
                            "0.0",
                            "3.75",
                            "11.0",
                        ],
                        [
                            "CH0",
                            "10.0",
                            "4.68",
                            "12.0",
                            "0.0",
                            "4.68",
                            "12.0",
                        ],
                        [
                            "O0",
                            "93.0",
                            "3.02",
                            "13.0",
                            "0.0",
                            "3.02",
                            "13.0",
                        ],
                        [
                            "H0",
                            "0.0",
                            "0.0",
                            "14.0",
                            "0.0",
                            "0.0",
                            "14.0",
                        ],
                        [
                            "CH31",
                            "98.0",
                            "3.75",
                            "11.0",
                            "0.0",
                            "3.75",
                            "11.0",
                        ],
                        [
                            "CH1",
                            "2.0",
                            "4.68",
                            "12.0",
                            "0.0",
                            "4.68",
                            "12.0",
                        ],
                        [
                            "O1",
                            "93.0",
                            "3.02",
                            "13.0",
                            "0.0",
                            "3.02",
                            "13.0",
                        ],
                        [
                            "H1",
                            "0.0",
                            "0.0",
                            "14.0",
                            "0.0",
                            "0.0",
                            "14.0",
                        ],
                    ]

                    for j in range(0, len(nb_types)):
                        assert (
                            len(out_gomc[i + 1 + j].split("!")[0].split()) == 7
                        )
                        assert (
                            out_gomc[i + 1 + j].split("!")[0].split()[0:7]
                            == nb_types[j]
                        )

                else:
                    pass

        assert masses_read
        assert bonds_read
        assert angles_read
        assert dihedrals_read
        assert nonbondeds_read

    def test_atom_type_style_all_unique_diff_sigma(
        self, two_propanol_ua, alt_two_propanol_ua
    ):
        box_0 = mb.fill_box(
            compound=[two_propanol_ua, alt_two_propanol_ua],
            n_compounds=[1, 1],
            box=[3, 3, 3],
        )

        charmm = Charmm(
            box_0,
            "test_atom_type_style_all_unique",
            ff_filename="test_atom_type_style_all_unique",
            forcefield_selection={
                two_propanol_ua.name: get_mosdef_gomc_fn(
                    "gmso_two_propanol_Mie_periodic_dihedral_ua_K_energy_units.xml"
                ),
                alt_two_propanol_ua.name: get_mosdef_gomc_fn(
                    "gmso_two_propanol_Mie_periodic_dihedral_ua_K_energy_units_diff_sigma.xml"
                ),
            },
            residues=[two_propanol_ua.name, alt_two_propanol_ua.name],
            gmso_match_ff_by="molecule",
            bead_to_atom_name_dict={"_CH3": "C", "_CH2": "C", "_HC": "C"},
            atom_type_naming_style="general",
        )
        charmm.write_inp()

        with open("test_atom_type_style_all_unique.inp", "r") as fp:
            masses_read = False
            bonds_read = False
            angles_read = False
            dihedrals_read = False
            nonbondeds_read = False
            out_gomc = fp.readlines()
            for i, line in enumerate(out_gomc):
                if (
                    "! atom_types" in line
                    and "mass" in line
                    and "atomClass_ResidueName" in line
                ):
                    masses_read = True
                    atom_types_1 = [
                        ["*", "CH30", "15.035"],
                        ["*", "CH0", "13.019"],
                        ["*", "O0", "15.9994"],
                        ["*", "H0", "1.008"],
                        ["*", "CH31", "15.035"],
                        ["*", "CH1", "13.019"],
                        ["*", "O1", "15.9994"],
                        ["*", "H1", "1.008"],
                    ]
                    atom_types_2 = [
                        ["POL_CH3_sp3"],
                        ["POL_CH_O"],
                        ["POL_O"],
                        ["POL_H"],
                        ["SPL_CH3_sp3"],
                        ["SPL_CH_O_diff"],
                        ["SPL_O"],
                        ["SPL_H"],
                    ]

                    for j in range(0, len(atom_types_1)):
                        assert (
                            len(out_gomc[i + 1 + j].split("!")[0].split()) == 3
                        )
                        assert (
                            out_gomc[i + 1 + j].split("!")[0].split()[0:3]
                            == atom_types_1[j]
                        )
                        assert (
                            out_gomc[i + 1 + j].split()[4:5] == atom_types_2[j]
                        )

                elif (
                    "! type_1" in line
                    and "type_2" in line
                    and "Kb" in line
                    and "b0" in line
                    and "extended_type_1" in line
                    and "extended_type_2" in line
                ):
                    bonds_read = True
                    bond_types = [
                        ["CH30", "CH0", "604267.5553", "1.54"],
                        ["CH0", "O0", "604267.5553", "1.43"],
                        ["O0", "H0", "604267.5553", "0.945"],
                        ["CH31", "CH1", "604267.5553", "1.54"],
                        ["CH1", "O1", "604267.5553", "1.43"],
                        ["O1", "H1", "604267.5553", "0.945"],
                    ]
                    total_bonds_evaluated = []
                    total_bonds_evaluated_reorg = []
                    for j in range(0, len(bond_types)):
                        assert (
                            len(out_gomc[i + 1 + j].split("!")[0].split()) == 4
                        )

                        if (
                            out_gomc[i + 1 + j].split("!")[0].split()[0:4]
                            == bond_types[0]
                            or bond_types[1]
                            or bond_types[2]
                        ):
                            total_bonds_evaluated.append(
                                out_gomc[i + 1 + j].split("!")[0].split()[0:4]
                            )
                    for k in range(0, len(bond_types)):
                        if bond_types[k] in total_bonds_evaluated:
                            total_bonds_evaluated_reorg.append(bond_types[k])
                    assert total_bonds_evaluated_reorg == bond_types

                elif (
                    "! type_1 " in line
                    and "type_2" in line
                    and "type_3" in line
                    and "Ktheta" in line
                    and "Theta0" in line
                    and "extended_type_1" in line
                    and "extended_type_2" in line
                    and "extended_type_3" in line
                ):
                    angles_read = True
                    angle_types = [
                        ["CH30", "CH0", "O0", "50400.0", "109.5"],
                        ["CH30", "CH0", "CH30", "62500.0", "112.0"],
                        ["CH0", "O0", "H0", "55400.0", "108.5"],
                        ["CH31", "CH1", "O1", "50400.0", "109.5"],
                        ["CH31", "CH1", "CH31", "62500.0", "112.0"],
                        ["CH1", "O1", "H1", "55400.0", "108.5"],
                    ]
                    total_angles_evaluated = []
                    total_angles_evaluated_reorg = []
                    for j in range(0, len(angle_types)):
                        assert (
                            len(out_gomc[i + 1 + j].split("!")[0].split()) == 5
                        )
                        if (
                            out_gomc[i + 1 + j].split("!")[0].split()[0:5]
                            == angle_types[0]
                            or angle_types[1]
                            or angle_types[2]
                        ):
                            total_angles_evaluated.append(
                                out_gomc[i + 1 + j].split("!")[0].split()[0:5]
                            )
                    for k in range(0, len(angle_types)):
                        if angle_types[k] in total_angles_evaluated:
                            total_angles_evaluated_reorg.append(angle_types[k])

                    assert total_angles_evaluated_reorg == angle_types

                elif (
                    "! type_1" in line
                    and "type_2" in line
                    and "type_3" in line
                    and "type_4" in line
                    and "Kchi" in line
                    and "n" in line
                    and "delta" in line
                    and "extended_type_1" in line
                    and "extended_type_2" in line
                    and "extended_type_3" in line
                    and "extended_type_4" in line
                ):
                    dihedrals_read = True
                    dihedral_types = [
                        ["CH30", "CH0", "O0", "H0", "-18.75", "0", "90.0"],
                        ["CH30", "CH0", "O0", "H0", "10.0", "1", "180.0"],
                        ["CH30", "CH0", "O0", "H0", "-10.0", "2", "0.0"],
                        ["CH30", "CH0", "O0", "H0", "10.0", "3", "180.0"],
                        ["CH30", "CH0", "O0", "H0", "-0.625", "4", "0.0"],
                        ["CH31", "CH1", "O1", "H1", "-18.75", "0", "90.0"],
                        ["CH31", "CH1", "O1", "H1", "10.0", "1", "180.0"],
                        ["CH31", "CH1", "O1", "H1", "-10.0", "2", "0.0"],
                        ["CH31", "CH1", "O1", "H1", "10.0", "3", "180.0"],
                        ["CH31", "CH1", "O1", "H1", "-0.625", "4", "0.0"],
                    ]
                    for j in range(0, len(dihedral_types)):
                        assert (
                            len(out_gomc[i + 1 + j].split("!")[0].split()) == 7
                        )
                        assert (
                            out_gomc[i + 1 + j].split("!")[0].split()[0:7]
                            == dihedral_types[j]
                        )

                elif (
                    "! type_1" in line
                    and "epsilon" in line
                    and "sigma" in line
                    and "n" in line
                    and "epsilon,1-4" in line
                    and "sigma,1-4" in line
                    and "n,1-4" in line
                    and "extended_type_1" in line
                    and "extended_type_2" in line
                ):
                    nonbondeds_read = True
                    nb_types = [
                        [
                            "CH30",
                            "98.0",
                            "3.75",
                            "11.0",
                            "0.0",
                            "3.75",
                            "11.0",
                        ],
                        [
                            "CH0",
                            "10.0",
                            "4.68",
                            "12.0",
                            "0.0",
                            "4.68",
                            "12.0",
                        ],
                        [
                            "O0",
                            "93.0",
                            "3.02",
                            "13.0",
                            "0.0",
                            "3.02",
                            "13.0",
                        ],
                        [
                            "H0",
                            "0.0",
                            "0.0",
                            "14.0",
                            "0.0",
                            "0.0",
                            "14.0",
                        ],
                        [
                            "CH31",
                            "98.0",
                            "3.75",
                            "11.0",
                            "0.0",
                            "3.75",
                            "11.0",
                        ],
                        [
                            "CH1",
                            "10.0",
                            "2.0",
                            "12.0",
                            "0.0",
                            "2.0",
                            "12.0",
                        ],
                        [
                            "O1",
                            "93.0",
                            "3.02",
                            "13.0",
                            "0.0",
                            "3.02",
                            "13.0",
                        ],
                        [
                            "H1",
                            "0.0",
                            "0.0",
                            "14.0",
                            "0.0",
                            "0.0",
                            "14.0",
                        ],
                    ]

                    for j in range(0, len(nb_types)):
                        assert (
                            len(out_gomc[i + 1 + j].split("!")[0].split()) == 7
                        )
                        assert (
                            out_gomc[i + 1 + j].split("!")[0].split()[0:7]
                            == nb_types[j]
                        )

                else:
                    pass

        assert masses_read
        assert bonds_read
        assert angles_read
        assert dihedrals_read
        assert nonbondeds_read

    def test_atom_type_style_all_unique_diff_mass(
        self, two_propanol_ua, alt_two_propanol_ua
    ):
        box_0 = mb.fill_box(
            compound=[two_propanol_ua, alt_two_propanol_ua],
            n_compounds=[1, 1],
            box=[3, 3, 3],
        )

        charmm = Charmm(
            box_0,
            "test_atom_type_style_all_unique",
            ff_filename="test_atom_type_style_all_unique",
            forcefield_selection={
                two_propanol_ua.name: get_mosdef_gomc_fn(
                    "gmso_two_propanol_Mie_periodic_dihedral_ua_K_energy_units.xml"
                ),
                alt_two_propanol_ua.name: get_mosdef_gomc_fn(
                    "gmso_two_propanol_Mie_periodic_dihedral_ua_K_energy_units_diff_mass.xml"
                ),
            },
            residues=[two_propanol_ua.name, alt_two_propanol_ua.name],
            gmso_match_ff_by="molecule",
            bead_to_atom_name_dict={"_CH3": "C", "_CH2": "C", "_HC": "C"},
            atom_type_naming_style="general",
        )
        charmm.write_inp()

        with open("test_atom_type_style_all_unique.inp", "r") as fp:
            masses_read = False
            bonds_read = False
            angles_read = False
            dihedrals_read = False
            nonbondeds_read = False
            out_gomc = fp.readlines()
            for i, line in enumerate(out_gomc):
                if (
                    "! atom_types" in line
                    and "mass" in line
                    and "atomClass_ResidueName" in line
                ):
                    masses_read = True
                    atom_types_1 = [
                        ["*", "CH30", "15.035"],
                        ["*", "CH0", "13.019"],
                        ["*", "O0", "15.9994"],
                        ["*", "H0", "1.008"],
                        ["*", "CH31", "15.035"],
                        ["*", "CH1", "1.0"],
                        ["*", "O1", "15.9994"],
                        ["*", "H1", "1.008"],
                    ]
                    atom_types_2 = [
                        ["POL_CH3_sp3"],
                        ["POL_CH_O"],
                        ["POL_O"],
                        ["POL_H"],
                        ["SPL_CH3_sp3"],
                        ["SPL_CH_O_diff"],
                        ["SPL_O"],
                        ["SPL_H"],
                    ]

                    for j in range(0, len(atom_types_1)):
                        assert (
                            len(out_gomc[i + 1 + j].split("!")[0].split()) == 3
                        )
                        assert (
                            out_gomc[i + 1 + j].split("!")[0].split()[0:3]
                            == atom_types_1[j]
                        )
                        assert (
                            out_gomc[i + 1 + j].split()[4:5] == atom_types_2[j]
                        )

                elif (
                    "! type_1" in line
                    and "type_2" in line
                    and "Kb" in line
                    and "b0" in line
                    and "extended_type_1" in line
                    and "extended_type_2" in line
                ):
                    bonds_read = True
                    bond_types = [
                        ["CH30", "CH0", "604267.5553", "1.54"],
                        ["CH0", "O0", "604267.5553", "1.43"],
                        ["O0", "H0", "604267.5553", "0.945"],
                        ["CH31", "CH1", "604267.5553", "1.54"],
                        ["CH1", "O1", "604267.5553", "1.43"],
                        ["O1", "H1", "604267.5553", "0.945"],
                    ]
                    total_bonds_evaluated = []
                    total_bonds_evaluated_reorg = []
                    for j in range(0, len(bond_types)):
                        assert (
                            len(out_gomc[i + 1 + j].split("!")[0].split()) == 4
                        )

                        if (
                            out_gomc[i + 1 + j].split("!")[0].split()[0:4]
                            == bond_types[0]
                            or bond_types[1]
                            or bond_types[2]
                        ):
                            total_bonds_evaluated.append(
                                out_gomc[i + 1 + j].split("!")[0].split()[0:4]
                            )
                    for k in range(0, len(bond_types)):
                        if bond_types[k] in total_bonds_evaluated:
                            total_bonds_evaluated_reorg.append(bond_types[k])
                    assert total_bonds_evaluated_reorg == bond_types

                elif (
                    "! type_1 " in line
                    and "type_2" in line
                    and "type_3" in line
                    and "Ktheta" in line
                    and "Theta0" in line
                    and "extended_type_1" in line
                    and "extended_type_2" in line
                    and "extended_type_3" in line
                ):
                    angles_read = True
                    angle_types = [
                        ["CH30", "CH0", "O0", "50400.0", "109.5"],
                        ["CH30", "CH0", "CH30", "62500.0", "112.0"],
                        ["CH0", "O0", "H0", "55400.0", "108.5"],
                        ["CH31", "CH1", "O1", "50400.0", "109.5"],
                        ["CH31", "CH1", "CH31", "62500.0", "112.0"],
                        ["CH1", "O1", "H1", "55400.0", "108.5"],
                    ]
                    total_angles_evaluated = []
                    total_angles_evaluated_reorg = []
                    for j in range(0, len(angle_types)):
                        assert (
                            len(out_gomc[i + 1 + j].split("!")[0].split()) == 5
                        )
                        if (
                            out_gomc[i + 1 + j].split("!")[0].split()[0:5]
                            == angle_types[0]
                            or angle_types[1]
                            or angle_types[2]
                        ):
                            total_angles_evaluated.append(
                                out_gomc[i + 1 + j].split("!")[0].split()[0:5]
                            )
                    for k in range(0, len(angle_types)):
                        if angle_types[k] in total_angles_evaluated:
                            total_angles_evaluated_reorg.append(angle_types[k])

                    assert total_angles_evaluated_reorg == angle_types

                elif (
                    "! type_1" in line
                    and "type_2" in line
                    and "type_3" in line
                    and "type_4" in line
                    and "Kchi" in line
                    and "n" in line
                    and "delta" in line
                    and "extended_type_1" in line
                    and "extended_type_2" in line
                    and "extended_type_3" in line
                    and "extended_type_4" in line
                ):
                    dihedrals_read = True
                    dihedral_types = [
                        ["CH30", "CH0", "O0", "H0", "-18.75", "0", "90.0"],
                        ["CH30", "CH0", "O0", "H0", "10.0", "1", "180.0"],
                        ["CH30", "CH0", "O0", "H0", "-10.0", "2", "0.0"],
                        ["CH30", "CH0", "O0", "H0", "10.0", "3", "180.0"],
                        ["CH30", "CH0", "O0", "H0", "-0.625", "4", "0.0"],
                        ["CH31", "CH1", "O1", "H1", "-18.75", "0", "90.0"],
                        ["CH31", "CH1", "O1", "H1", "10.0", "1", "180.0"],
                        ["CH31", "CH1", "O1", "H1", "-10.0", "2", "0.0"],
                        ["CH31", "CH1", "O1", "H1", "10.0", "3", "180.0"],
                        ["CH31", "CH1", "O1", "H1", "-0.625", "4", "0.0"],
                    ]
                    for j in range(0, len(dihedral_types)):
                        assert (
                            len(out_gomc[i + 1 + j].split("!")[0].split()) == 7
                        )
                        assert (
                            out_gomc[i + 1 + j].split("!")[0].split()[0:7]
                            == dihedral_types[j]
                        )

                elif (
                    "! type_1" in line
                    and "epsilon" in line
                    and "sigma" in line
                    and "n" in line
                    and "epsilon,1-4" in line
                    and "sigma,1-4" in line
                    and "n,1-4" in line
                    and "extended_type_1" in line
                    and "extended_type_2" in line
                ):
                    nonbondeds_read = True
                    nb_types = [
                        [
                            "CH30",
                            "98.0",
                            "3.75",
                            "11.0",
                            "0.0",
                            "3.75",
                            "11.0",
                        ],
                        [
                            "CH0",
                            "10.0",
                            "4.68",
                            "12.0",
                            "0.0",
                            "4.68",
                            "12.0",
                        ],
                        [
                            "O0",
                            "93.0",
                            "3.02",
                            "13.0",
                            "0.0",
                            "3.02",
                            "13.0",
                        ],
                        [
                            "H0",
                            "0.0",
                            "0.0",
                            "14.0",
                            "0.0",
                            "0.0",
                            "14.0",
                        ],
                        [
                            "CH31",
                            "98.0",
                            "3.75",
                            "11.0",
                            "0.0",
                            "3.75",
                            "11.0",
                        ],
                        [
                            "CH1",
                            "10.0",
                            "4.68",
                            "12.0",
                            "0.0",
                            "4.68",
                            "12.0",
                        ],
                        [
                            "O1",
                            "93.0",
                            "3.02",
                            "13.0",
                            "0.0",
                            "3.02",
                            "13.0",
                        ],
                        [
                            "H1",
                            "0.0",
                            "0.0",
                            "14.0",
                            "0.0",
                            "0.0",
                            "14.0",
                        ],
                    ]

                    for j in range(0, len(nb_types)):
                        assert (
                            len(out_gomc[i + 1 + j].split("!")[0].split()) == 7
                        )
                        assert (
                            out_gomc[i + 1 + j].split("!")[0].split()[0:7]
                            == nb_types[j]
                        )

                else:
                    pass

        assert masses_read
        assert bonds_read
        assert angles_read
        assert dihedrals_read
        assert nonbondeds_read

    def test_atom_type_style_all_unique_diff_Mie_n(
        self, two_propanol_ua, alt_two_propanol_ua
    ):
        box_0 = mb.fill_box(
            compound=[two_propanol_ua, alt_two_propanol_ua],
            n_compounds=[1, 1],
            box=[3, 3, 3],
        )

        charmm = Charmm(
            box_0,
            "test_atom_type_style_all_unique",
            ff_filename="test_atom_type_style_all_unique",
            forcefield_selection={
                two_propanol_ua.name: get_mosdef_gomc_fn(
                    "gmso_two_propanol_Mie_periodic_dihedral_ua_K_energy_units.xml"
                ),
                alt_two_propanol_ua.name: get_mosdef_gomc_fn(
                    "gmso_two_propanol_Mie_periodic_dihedral_ua_K_energy_units_diff_Mie_n.xml"
                ),
            },
            residues=[two_propanol_ua.name, alt_two_propanol_ua.name],
            gmso_match_ff_by="molecule",
            bead_to_atom_name_dict={"_CH3": "C", "_CH2": "C", "_HC": "C"},
            atom_type_naming_style="general",
        )
        charmm.write_inp()

        with open("test_atom_type_style_all_unique.inp", "r") as fp:
            masses_read = False
            bonds_read = False
            angles_read = False
            dihedrals_read = False
            nonbondeds_read = False
            out_gomc = fp.readlines()
            for i, line in enumerate(out_gomc):
                if (
                    "! atom_types" in line
                    and "mass" in line
                    and "atomClass_ResidueName" in line
                ):
                    masses_read = True
                    atom_types_1 = [
                        ["*", "CH30", "15.035"],
                        ["*", "CH0", "13.019"],
                        ["*", "O0", "15.9994"],
                        ["*", "H0", "1.008"],
                        ["*", "CH31", "15.035"],
                        ["*", "CH1", "13.019"],
                        ["*", "O1", "15.9994"],
                        ["*", "H1", "1.008"],
                    ]
                    atom_types_2 = [
                        ["POL_CH3_sp3"],
                        ["POL_CH_O"],
                        ["POL_O"],
                        ["POL_H"],
                        ["SPL_CH3_sp3"],
                        ["SPL_CH_O_diff"],
                        ["SPL_O"],
                        ["SPL_H"],
                    ]

                    for j in range(0, len(atom_types_1)):
                        assert (
                            len(out_gomc[i + 1 + j].split("!")[0].split()) == 3
                        )
                        assert (
                            out_gomc[i + 1 + j].split("!")[0].split()[0:3]
                            == atom_types_1[j]
                        )
                        assert (
                            out_gomc[i + 1 + j].split()[4:5] == atom_types_2[j]
                        )

                elif (
                    "! type_1" in line
                    and "type_2" in line
                    and "Kb" in line
                    and "b0" in line
                    and "extended_type_1" in line
                    and "extended_type_2" in line
                ):
                    bonds_read = True
                    bond_types = [
                        ["CH30", "CH0", "604267.5553", "1.54"],
                        ["CH0", "O0", "604267.5553", "1.43"],
                        ["O0", "H0", "604267.5553", "0.945"],
                        ["CH31", "CH1", "604267.5553", "1.54"],
                        ["CH1", "O1", "604267.5553", "1.43"],
                        ["O1", "H1", "604267.5553", "0.945"],
                    ]
                    total_bonds_evaluated = []
                    total_bonds_evaluated_reorg = []
                    for j in range(0, len(bond_types)):
                        assert (
                            len(out_gomc[i + 1 + j].split("!")[0].split()) == 4
                        )

                        if (
                            out_gomc[i + 1 + j].split("!")[0].split()[0:4]
                            == bond_types[0]
                            or bond_types[1]
                            or bond_types[2]
                        ):
                            total_bonds_evaluated.append(
                                out_gomc[i + 1 + j].split("!")[0].split()[0:4]
                            )
                    for k in range(0, len(bond_types)):
                        if bond_types[k] in total_bonds_evaluated:
                            total_bonds_evaluated_reorg.append(bond_types[k])
                    assert total_bonds_evaluated_reorg == bond_types

                elif (
                    "! type_1 " in line
                    and "type_2" in line
                    and "type_3" in line
                    and "Ktheta" in line
                    and "Theta0" in line
                    and "extended_type_1" in line
                    and "extended_type_2" in line
                    and "extended_type_3" in line
                ):
                    angles_read = True
                    angle_types = [
                        ["CH30", "CH0", "O0", "50400.0", "109.5"],
                        ["CH30", "CH0", "CH30", "62500.0", "112.0"],
                        ["CH0", "O0", "H0", "55400.0", "108.5"],
                        ["CH31", "CH1", "O1", "50400.0", "109.5"],
                        ["CH31", "CH1", "CH31", "62500.0", "112.0"],
                        ["CH1", "O1", "H1", "55400.0", "108.5"],
                    ]
                    total_angles_evaluated = []
                    total_angles_evaluated_reorg = []
                    for j in range(0, len(angle_types)):
                        assert (
                            len(out_gomc[i + 1 + j].split("!")[0].split()) == 5
                        )
                        if (
                            out_gomc[i + 1 + j].split("!")[0].split()[0:5]
                            == angle_types[0]
                            or angle_types[1]
                            or angle_types[2]
                        ):
                            total_angles_evaluated.append(
                                out_gomc[i + 1 + j].split("!")[0].split()[0:5]
                            )
                    for k in range(0, len(angle_types)):
                        if angle_types[k] in total_angles_evaluated:
                            total_angles_evaluated_reorg.append(angle_types[k])

                    assert total_angles_evaluated_reorg == angle_types

                elif (
                    "! type_1" in line
                    and "type_2" in line
                    and "type_3" in line
                    and "type_4" in line
                    and "Kchi" in line
                    and "n" in line
                    and "delta" in line
                    and "extended_type_1" in line
                    and "extended_type_2" in line
                    and "extended_type_3" in line
                    and "extended_type_4" in line
                ):
                    dihedrals_read = True
                    dihedral_types = [
                        ["CH30", "CH0", "O0", "H0", "-18.75", "0", "90.0"],
                        ["CH30", "CH0", "O0", "H0", "10.0", "1", "180.0"],
                        ["CH30", "CH0", "O0", "H0", "-10.0", "2", "0.0"],
                        ["CH30", "CH0", "O0", "H0", "10.0", "3", "180.0"],
                        ["CH30", "CH0", "O0", "H0", "-0.625", "4", "0.0"],
                        ["CH31", "CH1", "O1", "H1", "-18.75", "0", "90.0"],
                        ["CH31", "CH1", "O1", "H1", "10.0", "1", "180.0"],
                        ["CH31", "CH1", "O1", "H1", "-10.0", "2", "0.0"],
                        ["CH31", "CH1", "O1", "H1", "10.0", "3", "180.0"],
                        ["CH31", "CH1", "O1", "H1", "-0.625", "4", "0.0"],
                    ]
                    for j in range(0, len(dihedral_types)):
                        assert (
                            len(out_gomc[i + 1 + j].split("!")[0].split()) == 7
                        )
                        assert (
                            out_gomc[i + 1 + j].split("!")[0].split()[0:7]
                            == dihedral_types[j]
                        )

                elif (
                    "! type_1" in line
                    and "epsilon" in line
                    and "sigma" in line
                    and "n" in line
                    and "epsilon,1-4" in line
                    and "sigma,1-4" in line
                    and "n,1-4" in line
                    and "extended_type_1" in line
                    and "extended_type_2" in line
                ):
                    nonbondeds_read = True
                    nb_types = [
                        [
                            "CH30",
                            "98.0",
                            "3.75",
                            "11.0",
                            "0.0",
                            "3.75",
                            "11.0",
                        ],
                        [
                            "CH0",
                            "10.0",
                            "4.68",
                            "12.0",
                            "0.0",
                            "4.68",
                            "12.0",
                        ],
                        [
                            "O0",
                            "93.0",
                            "3.02",
                            "13.0",
                            "0.0",
                            "3.02",
                            "13.0",
                        ],
                        [
                            "H0",
                            "0.0",
                            "0.0",
                            "14.0",
                            "0.0",
                            "0.0",
                            "14.0",
                        ],
                        [
                            "CH31",
                            "98.0",
                            "3.75",
                            "11.0",
                            "0.0",
                            "3.75",
                            "11.0",
                        ],
                        [
                            "CH1",
                            "10.0",
                            "4.68",
                            "10.0",
                            "0.0",
                            "4.68",
                            "10.0",
                        ],
                        [
                            "O1",
                            "93.0",
                            "3.02",
                            "13.0",
                            "0.0",
                            "3.02",
                            "13.0",
                        ],
                        [
                            "H1",
                            "0.0",
                            "0.0",
                            "14.0",
                            "0.0",
                            "0.0",
                            "14.0",
                        ],
                    ]

                    for j in range(0, len(nb_types)):
                        assert (
                            len(out_gomc[i + 1 + j].split("!")[0].split()) == 7
                        )
                        assert (
                            out_gomc[i + 1 + j].split("!")[0].split()[0:7]
                            == nb_types[j]
                        )

                else:
                    pass

        assert masses_read
        assert bonds_read
        assert angles_read
        assert dihedrals_read
        assert nonbondeds_read

    def test_write_inp_zeolite_non_othoganol_water_zeolite_level_group(
        self, water
    ):
        lattice_cif_ETV_triclinic = load_cif(
            file_or_path=get_mosdef_gomc_fn("ETV_triclinic.cif")
        )
        ETV_triclinic = lattice_cif_ETV_triclinic.populate(x=2, y=2, z=2)
        ETV_triclinic.name = "ETV"

        box_0 = mb.fill_box(
            compound=[water], n_compounds=[1], box=[1.5, 1.5, 1.5]
        )

        box_1 = mb.fill_box(
            compound=[water], n_compounds=[1], box=[1.5, 1.5, 1.5]
        )

        top_box = mb.Compound()
        top_box.box = mb.Box(lengths=[2, 2, 4])

        one_under_top = mb.Compound()
        one_under_top.box = mb.Box(lengths=[2, 2, 4])

        box_0.translate([0, 0, 2])
        one_under_top.add(box_0, inherit_periodicity=False)
        one_under_top.add(ETV_triclinic, inherit_periodicity=False)
        top_box.add(one_under_top, inherit_periodicity=False)

        charmm = Charmm(
            top_box,
            "ETV_triclinic_ethane_in_1_box",
            structure_box_1=box_1,
            filename_box_1="water_box_box_1",
            ff_filename="ETV_triclinic_ethane_in_1_box",
            forcefield_selection={
                ETV_triclinic.name: get_mosdef_gomc_fn(
                    "Charmm_writer_testing_only_zeolite.xml"
                ),
                water.name: get_mosdef_gomc_fn(
                    "gmso_spce_water__geometric_combining.xml"
                ),
            },
            residues=[water.name, ETV_triclinic.name],
            bead_to_atom_name_dict=None,
            fix_residue=[ETV_triclinic.name],
            gmso_match_ff_by="group",
            atom_type_naming_style="general",
        )
        charmm.write_inp()

        with open("ETV_triclinic_ethane_in_1_box.inp", "r") as fp:
            masses_read = False
            nonbondeds_read = False
            out_gomc = fp.readlines()
            for i, line in enumerate(out_gomc):
                if (
                    "! atom_types" in line
                    and "mass" in line
                    and "atomClass_ResidueName" in line
                ):
                    masses_read = True
                    mass_type_1 = [
                        ["*", "OW", "15.999"],
                        ["*", "HW", "1.008"],
                        ["*", "O", "15.9994"],
                        ["*", "Si", "28.0855"],
                    ]
                    mass_type_2 = [
                        ["WAT_o_spce"],
                        ["WAT_h_spce"],
                        ["ETV_O"],
                        ["ETV_Si"],
                    ]
                    for j in range(0, len(mass_type_1)):
                        assert (
                            len(out_gomc[i + 1 + j].split("!")[0].split()) == 3
                        )
                        assert (
                            out_gomc[i + 1 + j].split("!")[0].split()[0:3]
                            == mass_type_1[j]
                        )
                        assert (
                            out_gomc[i + 1 + j].split()[4:5] == mass_type_2[j]
                        )

                elif (
                    "! type_1" in line
                    and "ignored" in line
                    and "epsilon" in line
                    and "Rmin/2" in line
                    and "ignored" in line
                    and "epsilon,1-4" in line
                    and "Rmin/2,1-4" in line
                    and "extended_type_1" in line
                    and "extended_type_2" in line
                ):
                    nonbondeds_read = True
                    nb_types = [
                        [
                            "OW",
                            "0.0",
                            "-0.1554000956",
                            "1.7766160931",
                            "0.0",
                            "-0.0",
                            "1.7766160931",
                        ],
                        [
                            "HW",
                            "0.0",
                            "-0.0",
                            "0.5612310242",
                            "0.0",
                            "-0.0",
                            "0.5612310242",
                        ],
                        [
                            "O",
                            "0.0",
                            "-0.2941061185",
                            "2.0933917201",
                            "0.0",
                            "-0.0",
                            "2.0933917201",
                        ],
                        [
                            "Si",
                            "0.0",
                            "-0.0556417304",
                            "1.9081854821",
                            "0.0",
                            "-0.0",
                            "1.9081854821",
                        ],
                    ]

                    for j in range(0, len(nb_types)):
                        assert (
                            len(out_gomc[i + 1 + j].split("!")[0].split()) == 7
                        )
                        assert (
                            out_gomc[i + 1 + j].split("!")[0].split()[0:7]
                            == nb_types[j]
                        )

                else:
                    pass

        assert masses_read
        assert nonbondeds_read

    def test_write_inp_ethanol_ethane_in_a_single_box_with_group(
        self, ethanol_gomc, ethane_gomc
    ):
        box_0 = mb.fill_box(
            compound=[ethanol_gomc, ethane_gomc],
            n_compounds=[1, 1],
            box=[3, 3, 3],
        )

        charmm = Charmm(
            box_0,
            "ethanol_ethane_in_a_single_box_with_group",
            structure_box_1=None,
            filename_box_1=None,
            ff_filename="ethanol_ethane_in_a_single_box_with_group",
            forcefield_selection="oplsaa",
            residues=[ethanol_gomc.name, ethane_gomc.name],
            bead_to_atom_name_dict=None,
            fix_residue=None,
            gmso_match_ff_by="group",
            atom_type_naming_style="general",
        )
        charmm.write_inp()

        with open("ethanol_ethane_in_a_single_box_with_group.inp", "r") as fp:
            bonds_read = False
            out_gomc = fp.readlines()
            for i, line in enumerate(out_gomc):
                if (
                    "! type_1" in line
                    and "type_2" in line
                    and "Kb" in line
                    and "b0" in line
                    and "extended_type_1" in line
                    and "extended_type_2" in line
                ):
                    bonds_read = True
                    bond_types = [
                        ["CT", "CT", "268.0", "1.529"],
                        ["CT", "HC", "340.0", "1.09"],
                        ["CT", "OH", "320.0", "1.41"],
                        ["HO", "OH", "553.0", "0.945"],
                    ]
                    total_bonds_evaluated = []
                    total_bonds_evaluated_reorg = []
                    for j in range(0, len(bond_types)):
                        assert (
                            len(out_gomc[i + 1 + j].split("!")[0].split()) == 4
                        )

                        if (
                            out_gomc[i + 1 + j].split("!")[0].split()[0:4]
                            == bond_types[0]
                            or bond_types[1]
                            or bond_types[2]
                            or bond_types[3]
                            or bond_types[4]
                            or bond_types[5]
                            or bond_types[6]
                        ):
                            total_bonds_evaluated.append(
                                out_gomc[i + 1 + j].split("!")[0].split()[0:4]
                            )
                    for k in range(0, len(bond_types)):
                        if bond_types[k] in total_bonds_evaluated:
                            total_bonds_evaluated_reorg.append(bond_types[k])
                    print(
                        f" total_bonds_evaluated_reorg= {total_bonds_evaluated_reorg}"
                    )
                    print(f"bond_types = {bond_types}")
                    assert total_bonds_evaluated_reorg == bond_types

        assert bonds_read

    # This fails because box_0 is altered when added to the one_under_top and top_box,
    # which are not apparent changes under the hood
    def test_write_inp_zeolite_non_othoganol_water_box_0_altered_under_hood(
        self, water
    ):
        with pytest.raises(
            AssertionError,
            match=r"Compound is not a top level compound. Make a copy to pass to the "
            r"`compound`     argument that has no parents",
        ):
            lattice_cif_ETV_triclinic = load_cif(
                file_or_path=get_mosdef_gomc_fn("ETV_triclinic.cif")
            )
            ETV_triclinic = lattice_cif_ETV_triclinic.populate(x=2, y=2, z=2)
            ETV_triclinic.name = "ETV"

            box_0 = mb.fill_box(
                compound=[water], n_compounds=[1], box=[1.5, 1.5, 1.5]
            )

            top_box = mb.Compound()
            top_box.box = mb.Box(lengths=[2, 2, 4])

            one_under_top = mb.Compound()
            one_under_top.box = mb.Box(lengths=[2, 2, 4])

            box_0.translate([0, 0, 2])
            one_under_top.add(box_0, inherit_periodicity=False)
            one_under_top.add(ETV_triclinic, inherit_periodicity=False)
            top_box.add(one_under_top, inherit_periodicity=False)

            Charmm(
                top_box,
                "ETV_triclinic_ethane_in_1_box",
                structure_box_1=box_0,
                filename_box_1="water_box_box_1",
                ff_filename="ETV_triclinic_ethane_in_1_box",
                forcefield_selection={
                    ETV_triclinic.name: get_mosdef_gomc_fn(
                        "Charmm_writer_testing_only_zeolite.xml"
                    ),
                    water.name: get_mosdef_gomc_fn(
                        "gmso_spce_water__geometric_combining.xml"
                    ),
                },
                residues=[water.name, ETV_triclinic.name],
                bead_to_atom_name_dict=None,
                fix_residue=[ETV_triclinic.name],
                gmso_match_ff_by="group",
                atom_type_naming_style="general",
            )

    def test_save_charmm_benzene_gaff_gomc_ff_only_1_proper_and_improper_periodic_value(
        self,
    ):
        benzene = mb.load("c1ccccc1", smiles=True)
        benzene.name = "BEN"

        box_0 = mb.fill_box(compound=[benzene], n_compounds=[1], box=[4, 4, 4])

        charmm = Charmm(
            box_0,
            "charmm_benzene_gaff_data",
            ff_filename="charmm_benzene_gaff_data",
            residues=[benzene.name],
            forcefield_selection=get_mosdef_gomc_fn("gmso_benzene_GAFF.xml"),
            atom_type_naming_style="all_unique",
        )
        charmm.write_inp()

        with open("charmm_benzene_gaff_data.inp", "r") as fp:
            masses_read = False
            bonds_read = False
            angles_read = False
            dihedrals_read = False
            impropers_read = False
            nonbondeds_read = False
            out_gomc = fp.readlines()
            for i, line in enumerate(out_gomc):
                if (
                    "! atom_types" in line
                    and "mass" in line
                    and "atomClass_ResidueName" in line
                ):
                    masses_read = True
                    assert len(out_gomc[i + 1].split("!")[0].split()) == 3
                    assert out_gomc[i + 1].split("!")[0].split()[0:3] == [
                        "*",
                        "ca0",
                        "12.01",
                    ]
                    assert len(out_gomc[i + 2].split("!")[0].split()) == 3
                    assert out_gomc[i + 2].split("!")[0].split()[0:3] == [
                        "*",
                        "ha0",
                        "1.008",
                    ]
                    assert out_gomc[i + 1].split()[4:5] == ["BEN_ca"]
                    assert out_gomc[i + 2].split()[4:5] == ["BEN_ha"]

                elif (
                    "! type_1" in line
                    and "type_2" in line
                    and "Kb" in line
                    and "b0" in line
                    and "extended_type_1" in line
                    and "extended_type_2" in line
                ):
                    bonds_read = True
                    bond_types = [
                        ["ca0", "ca0", "461.1", "1.3984"],
                        ["ca0", "ha0", "345.8", "1.086"],
                    ]
                    assert len(out_gomc[i + 1].split("!")[0].split()) == 4
                    assert len(out_gomc[i + 2].split("!")[0].split()) == 4
                    if (
                        out_gomc[i + 1].split("!")[0].split()[0:4]
                        == bond_types[0]
                    ):
                        assert (
                            out_gomc[i + 1].split("!")[0].split()[0:4]
                            == bond_types[0]
                        )
                        assert (
                            out_gomc[i + 2].split("!")[0].split()[0:4]
                            == bond_types[1]
                        )
                    elif (
                        out_gomc[i + 1].split("!")[0].split()[0:4]
                        == bond_types[1]
                    ):
                        assert (
                            out_gomc[i + 1].split("!")[0].split()[0:4]
                            == bond_types[1]
                        )
                        assert (
                            out_gomc[i + 2].split("!")[0].split()[0:4]
                            == bond_types[0]
                        )

                elif (
                    "! type_1 " in line
                    and "type_2" in line
                    and "type_3" in line
                    and "Ktheta" in line
                    and "Theta0" in line
                    and "extended_type_1" in line
                    and "extended_type_2" in line
                    and "extended_type_3" in line
                ):
                    angles_read = True
                    angle_types = [
                        ["ca0", "ca0", "ha0", "48.2", "119.88"],
                        ["ca0", "ca0", "ca0", "66.6", "120.02"],
                    ]
                    assert len(out_gomc[i + 1].split("!")[0].split()) == 5
                    assert len(out_gomc[i + 2].split("!")[0].split()) == 5
                    if (
                        out_gomc[i + 1].split("!")[0].split()[0:5]
                        == angle_types[0]
                    ):
                        assert (
                            out_gomc[i + 1].split("!")[0].split()[0:5]
                            == angle_types[0]
                        )
                        assert (
                            out_gomc[i + 2].split("!")[0].split()[0:5]
                            == angle_types[1]
                        )
                    elif (
                        out_gomc[i + 1].split("!")[0].split()[0:4]
                        == angle_types[1]
                    ):
                        assert (
                            out_gomc[i + 1].split("!")[0].split()[0:5]
                            == angle_types[1]
                        )
                        assert (
                            out_gomc[i + 2].split("!")[0].split()[0:5]
                            == angle_types[0]
                        )

                elif (
                    "! type_1" in line
                    and "type_2" in line
                    and "type_3" in line
                    and "type_4" in line
                    and "Kchi" in line
                    and "n" in line
                    and "delta" in line
                    and "extended_type_1" in line
                    and "extended_type_2" in line
                    and "extended_type_3" in line
                    and "extended_type_4" in line
                ):
                    dihedrals_read = True
                    dihed_types = [
                        ["ha0", "ca0", "ca0", "ca0", "3.625", "2", "180.0"],
                        ["ha0", "ca0", "ca0", "ha0", "3.625", "2", "180.0"],
                        ["ca0", "ca0", "ca0", "ca0", "3.625", "2", "180.0"],
                    ]

                    # need to get and sort order as order seems to be random sometimes
                    actual_dihed = []

                    for j in range(0, len(dihed_types)):
                        assert (
                            len(out_gomc[i + 1 + j].split("!")[0].split()) == 7
                        )
                        actual_dihed.append(
                            out_gomc[i + 1 + j].split("!")[0].split()[0:7]
                        )

                    assert actual_dihed.sort() == dihed_types.sort()

                elif (
                    "! type_1" in line
                    and "type_2" in line
                    and "type_3" in line
                    and "type_4" in line
                    and "Kw" in line
                    and "n" in line
                    and "w0" in line
                    and "extended_type_1" in line
                    and "extended_type_2" in line
                    and "extended_type_3" in line
                    and "extended_type_4" in line
                ):
                    impropers_read = True
                    improp_types = [
                        ["ca0", "ca0", "ca0", "ha0", "1.1", "2", "180.0"],
                    ]
                    for j in range(0, len(improp_types)):
                        assert (
                            len(out_gomc[i + 1 + j].split("!")[0].split()) == 7
                        )
                        assert (
                            out_gomc[i + 1 + j].split("!")[0].split()[0:7]
                            == improp_types[j]
                        )

                elif (
                    "! type_1" in line
                    and "ignored" in line
                    and "epsilon" in line
                    and "Rmin/2" in line
                    and "ignored" in line
                    and "epsilon,1-4" in line
                    and "Rmin/2,1-4" in line
                    and "extended_type_1" in line
                    and "extended_type_2" in line
                ):
                    nonbondeds_read = True
                    nb_types = [
                        [
                            "ca0",
                            "0.0",
                            "-0.086",
                            "1.908",
                            "0.0",
                            "-0.043",
                            "1.908",
                        ],
                        [
                            "ha0",
                            "0.0",
                            "-0.015",
                            "1.459",
                            "0.0",
                            "-0.0075",
                            "1.459",
                        ],
                    ]

                    for j in range(0, len(nb_types)):
                        assert (
                            len(out_gomc[i + 1 + j].split("!")[0].split()) == 7
                        )
                        assert (
                            out_gomc[i + 1 + j].split("!")[0].split()[0:7]
                            == nb_types[j]
                        )

                else:
                    pass

        assert masses_read
        assert bonds_read
        assert angles_read
        assert dihedrals_read
        assert impropers_read
        assert nonbondeds_read

    """
    # ***************************
    # ***************************
    # THIS IS NOT AUTO TESTED BECAUSE IT TAKE TOO LONG.  (START)
    # HOWEVER THIS SHOULD BE MANUALLY TESTED OCCASIONALLY TO ENSURE THE SEGMENT_IDS CHANGE
    # EVERY 9999 RESIDUE_IDS
    # (EX: SEGMENT_IDS = 1 for the 1st RESIDUE_IDS=1-9999 --> SEGMENT_IDS = 2 for the 2nd RESIDUE_IDS=1-9999)
    # There should be SEGMENT_IDS A to C in this example
    # ***************************
    # ***************************
    def test_write_seg_id_changes_ethane_ua_correctly(self):
        ethane_ua = mb.load(get_mosdef_gomc_fn("ethane_ua.mol2"))
        ethane_ua.name = "ETH"
        ethane_ua_box = mb.fill_box(
            compound=ethane_ua,
            n_compounds=20002,
            box=[60, 60, 60]
        )

        charmm = Charmm(
            ethane_ua_box,
            "test_write_seg_id_changes_correctly_box_0",
            ff_filename="test_write_seg_id_changes_correctly",
            structure_box_1=ethane_ua_box,
            filename_box_1="test_write_seg_id_changes_correctly_box_1",
            residues=[ethane_ua.name],
            forcefield_selection=get_mosdef_gomc_fn("ethane_propane_ua_lorentz_combining.xml"),
            gmso_match_ff_by='molecule',
            bead_to_atom_name_dict={"_CH3": "C", "_CH2": "C", "_HC": "C"},
        )
        charmm.write_psf()
        charmm.write_pdb()

        with open("test_write_seg_id_changes_correctly_box_0.psf", "r") as fp:
            psf_read = False
            out_gomc = fp.readlines()
            for i, line in enumerate(out_gomc):
                no_ethane_atoms = 40004
                if f"{no_ethane_atoms} !NATOM" in line:
                    psf_read = True

                    atom_number = 1
                    seg_id_line_A_is_1 = [
                        str(atom_number),
                        "A",
                        "1",
                        "ETH",
                        "C1",
                        "CH30",
                        "0.000000",
                        "15.0350",
                    ]
                    assert out_gomc[i + atom_number].split()[0:8] == seg_id_line_A_is_1

                    atom_number = 19998
                    seg_id_line_A_is_9999 = [
                        str(atom_number),
                        "A",
                        "9999",
                        "ETH",
                        "C2",
                        "CH30",
                        "0.000000",
                        "15.0350",
                    ]
                    assert out_gomc[i + atom_number].split()[0:8] == seg_id_line_A_is_9999

                    atom_number = 19999
                    seg_id_line_B_is_1 = [
                        str(atom_number),
                        "B",
                        "1",
                        "ETH",
                        "C1",
                        "CH30",
                        "0.000000",
                        "15.0350",
                    ]
                    assert out_gomc[i + atom_number].split()[0:8] == seg_id_line_B_is_1

                    atom_number = 39996
                    seg_id_line_B_is_9999 = [
                        str(atom_number),
                        "B",
                        "9999",
                        "ETH",
                        "C2",
                        "CH30",
                        "0.000000",
                        "15.0350",
                    ]
                    assert out_gomc[i + atom_number].split()[0:8] == seg_id_line_B_is_9999

                    atom_number = 39997
                    seg_id_line_C_is_1 = [
                        str(atom_number),
                        "C",
                        "1",
                        "ETH",
                        "C1",
                        "CH30",
                        "0.000000",
                        "15.0350",
                    ]
                    assert out_gomc[i + atom_number].split()[0:8] == seg_id_line_C_is_1


                else:
                    pass

        assert psf_read

        with open("test_write_seg_id_changes_correctly_box_0.pdb", "r") as fp:
            pdb_read = False
            out_gomc = fp.readlines()
            for i, line in enumerate(out_gomc):
                if (
                        "CRYST1" in line
                        and "600.000" in line
                        and "600.000" in line
                        and "600.000" in line
                        and "90.00" in line
                        and "90.00" in line
                        and "90.00" in line
                ):
                    pdb_read = True

                    atom_number = 1
                    seg_id_line_A_is_1 = [
                        "ATOM",
                        str(atom_number),
                        "C1",
                        "ETH",
                        "A",
                        "1",
                    ]
                    assert out_gomc[i + atom_number].split()[0:6] == seg_id_line_A_is_1

                    atom_number = 19998
                    seg_id_line_A_is_9999 = [
                        "ATOM",
                        str(atom_number),
                        "C2",
                        "ETH",
                        "A9999",
                    ]
                    assert out_gomc[i + atom_number].split()[0:5] == seg_id_line_A_is_9999

                    atom_number = 19999
                    seg_id_line_B_is_1 = [
                        "ATOM",
                        str(atom_number),
                        "C1",
                        "ETH",
                        "B",
                        "1",
                    ]
                    assert out_gomc[i + atom_number].split()[0:6] == seg_id_line_B_is_1

                    atom_number = 39996
                    seg_id_line_B_is_9999 = [
                        "ATOM",
                        str(atom_number),
                        "C2",
                        "ETH",
                        "B9999",
                    ]
                    assert out_gomc[i + atom_number].split()[0:5] == seg_id_line_B_is_9999

                    atom_number = 39997
                    seg_id_line_C_is_1 = [
                        "ATOM",
                        str(atom_number),
                        "C1",
                        "ETH",
                        "C",
                        "1",
                    ]
                    assert out_gomc[i + atom_number].split()[0:6] == seg_id_line_C_is_1


                else:
                    pass

        assert pdb_read

        with open("test_write_seg_id_changes_correctly_box_1.psf", "r") as fp:
            psf_read = False
            out_gomc = fp.readlines()
            for i, line in enumerate(out_gomc):
                no_ethane_atoms = 40004
                if f"{no_ethane_atoms} !NATOM" in line:
                    psf_read = True

                    atom_number = 1
                    seg_id_line_A_is_1 = [
                        str(atom_number),
                        "A",
                        "1",
                        "ETH",
                        "C1",
                        "CH30",
                        "0.000000",
                        "15.0350",
                    ]
                    assert out_gomc[i + atom_number].split()[0:8] == seg_id_line_A_is_1

                    atom_number = 19998
                    seg_id_line_A_is_9999 = [
                        str(atom_number),
                        "A",
                        "9999",
                        "ETH",
                        "C2",
                        "CH30",
                        "0.000000",
                        "15.0350",
                    ]
                    assert out_gomc[i + atom_number].split()[0:8] == seg_id_line_A_is_9999

                    atom_number = 19999
                    seg_id_line_B_is_1 = [
                        str(atom_number),
                        "B",
                        "1",
                        "ETH",
                        "C1",
                        "CH30",
                        "0.000000",
                        "15.0350",
                    ]
                    assert out_gomc[i + atom_number].split()[0:8] == seg_id_line_B_is_1

                    atom_number = 39996
                    seg_id_line_B_is_9999 = [
                        str(atom_number),
                        "B",
                        "9999",
                        "ETH",
                        "C2",
                        "CH30",
                        "0.000000",
                        "15.0350",
                    ]
                    assert out_gomc[i + atom_number].split()[0:8] == seg_id_line_B_is_9999

                    atom_number = 39997
                    seg_id_line_C_is_1 = [
                        str(atom_number),
                        "C",
                        "1",
                        "ETH",
                        "C1",
                        "CH30",
                        "0.000000",
                        "15.0350",
                    ]
                    assert out_gomc[i + atom_number].split()[0:8] == seg_id_line_C_is_1


                else:
                    pass

        assert psf_read

        with open("test_write_seg_id_changes_correctly_box_1.pdb", "r") as fp:
            pdb_read = False
            out_gomc = fp.readlines()
            for i, line in enumerate(out_gomc):
                if (
                        "CRYST1" in line
                        and "600.000" in line
                        and "600.000" in line
                        and "600.000" in line
                        and "90.00" in line
                        and "90.00" in line
                        and "90.00" in line
                ):
                    pdb_read = True

                    atom_number = 1
                    seg_id_line_A_is_1 = [
                        "ATOM",
                        str(atom_number),
                        "C1",
                        "ETH",
                        "A",
                        "1",
                    ]
                    assert out_gomc[i + atom_number].split()[0:6] == seg_id_line_A_is_1

                    atom_number = 19998
                    seg_id_line_A_is_9999 = [
                        "ATOM",
                        str(atom_number),
                        "C2",
                        "ETH",
                        "A9999",
                    ]
                    assert out_gomc[i + atom_number].split()[0:5] == seg_id_line_A_is_9999

                    atom_number = 19999
                    seg_id_line_B_is_1 = [
                        "ATOM",
                        str(atom_number),
                        "C1",
                        "ETH",
                        "B",
                        "1",
                    ]
                    assert out_gomc[i + atom_number].split()[0:6] == seg_id_line_B_is_1

                    atom_number = 39996
                    seg_id_line_B_is_9999 = [
                        "ATOM",
                        str(atom_number),
                        "C2",
                        "ETH",
                        "B9999",
                    ]
                    assert out_gomc[i + atom_number].split()[0:5] == seg_id_line_B_is_9999

                    atom_number = 39997
                    seg_id_line_C_is_1 = [
                        "ATOM",
                        str(atom_number),
                        "C1",
                        "ETH",
                        "C",
                        "1",
                    ]
                    assert out_gomc[i + atom_number].split()[0:6] == seg_id_line_C_is_1


                else:
                    pass

        assert pdb_read

    # ***************************
    # ***************************
    # THIS IS NOT AUTO TESTED BECAUSE IT TAKE TOO LONG.  (END)
    # HOWEVER THIS SHOULD BE MANUALLY TESTED OCCASIONALLY TO ENSURE THE SEGMENT_IDS CHANGE
    # EVERY 9999 RESIDUE_IDS
    # (EX: SEGMENT_IDS = 1 for the 1st RESIDUE_IDS=1-9999 --> SEGMENT_IDS = 2 for the 2nd RESIDUE_IDS=1-9999)
    # There should be SEGMENT_IDS A to C in this example
    # ***************************
    # ***************************
    """

    def test_Exp6_Rmin_to_sigma_solver(self):
        exp6_sigma_value = _Exp6_Rmin_to_sigma_solver(4.0941137, 16)
        assert np.isclose(exp6_sigma_value, 3.6790000166)

    def test_Exp6_Rmin_to_sigma_solver_failing_alpha_equal_6(self):
        with pytest.raises(
            ValueError,
            match=f"ERROR: The Exp6 potential Rmin --> sigma converter failed. "
            f"The Exp6 potential values can not be Rmin = 0 or alpha = 6, "
            f"as it divides by zero. "
            f"The entered values are Rmin = 4.0941137 and alpha = 6.",
        ):
            exp6_sigma_value = _Exp6_Rmin_to_sigma_solver(4.0941137, 6)

    def test_Exp6_Rmin_to_sigma_solver_failing_Rmin_equal_0(self):
        with pytest.raises(
            ValueError,
            match=f"ERROR: The Exp6 potential Rmin --> sigma converter failed. "
            f"The Exp6 potential values can not be Rmin = 0 or alpha = 6, "
            f"as it divides by zero. "
            f"The entered values are Rmin = 0 and alpha = 16.",
        ):
            exp6_sigma_value = _Exp6_Rmin_to_sigma_solver(0, 16)

    def test_Exp6_Rmin_to_sigma_solver_failing_values_large_alpha(self):
        with pytest.raises(
            ValueError,
            match="ERROR: The Exp6 potential Rmin --> sigma converter failed. "
            "It did not converge, sigma_calculated >= Rmin_actual, or "
            "another issue.",
        ):
            exp6_sigma_value = _Exp6_Rmin_to_sigma_solver(4.0941137, 1000000.0)

    def test_Exp6_Rmin_to_sigma_solver_failing_values_sigma_greater_than_Rmin(
        self,
    ):
        with pytest.raises(
            ValueError,
            match="ERROR: The Exp6 potential Rmin --> sigma converter failed. "
            "It did not converge, sigma_calculated >= Rmin_actual, or "
            "another issue.",
        ):
            exp6_sigma_value = _Exp6_Rmin_to_sigma_solver(
                4.0941137, 16, Rmin_fraction_for_sigma_findroot=1.1
            )

    def test_Exp6_sigma_to_Rmin_solver(
        self,
    ):
        exp6_Rmin_value = _Exp6_sigma_to_Rmin_solver(3.6790000166, 16)

        assert np.isclose(exp6_Rmin_value, 4.0941137)

    def test_Exp6_sigma_to_Rmin_solver_failing_alpha_equal_6(self):
        with pytest.raises(
            ValueError,
            match=f"ERROR: The Exp6 potential sigma --> Rmin converter failed. "
            f"The Exp6 potential values can not be sigma = 0 or alpha = 6, "
            f"as it divides by zero. "
            f"The entered values are sigma = 4.0941137 and alpha = 6.",
        ):
            exp6_Rmin_value = _Exp6_sigma_to_Rmin_solver(4.0941137, 6)

    def test_Exp6_sigma_to_Rmin_solver_failing_sigma_equal_0(self):
        with pytest.raises(
            ValueError,
            match=f"ERROR: The Exp6 potential sigma --> Rmin converter failed. "
            f"The Exp6 potential values can not be sigma = 0 or alpha = 6, "
            f"as it divides by zero. "
            f"The entered values are sigma = 0 and alpha = 16.",
        ):
            exp6_Rmin_value = _Exp6_sigma_to_Rmin_solver(0, 16)

    def test_Exp6_sigma_to_Rmin_solver_failing_values_small_alpha(self):
        with pytest.raises(
            ValueError,
            match=f"ERROR: The Exp6 potential sigma --> Rmin converter failed. "
            "It did not converge, Rmin_calculated <= sigma_actual, or "
            "another issue.",
        ):
            exp6_Rmin_value = _Exp6_sigma_to_Rmin_solver(3.6790000166, 0.01)

    def test_Exp6_sigma_to_Rmin_solver_failing_values_Rmin_greater_than_sigma(
        self,
    ):
        with pytest.raises(
            ValueError,
            match=f"ERROR: The Exp6 potential sigma --> Rmin converter failed. "
            "It did not converge, Rmin_calculated <= sigma_actual, or "
            "another issue.",
        ):
            exp6_Rmin_value = _Exp6_sigma_to_Rmin_solver(
                3.6790000166, 0.1, sigma_fraction_for_Rmin_findroot=0.1
            )

    def test_save_Exp6_gomc_ff(self, hexane_ua):
        box_0 = mb.fill_box(
            compound=[hexane_ua], n_compounds=[2], box=[4, 4, 4]
        )

        charmm = Charmm(
            box_0,
            "exp6_data",
            ff_filename="exp6_data",
            residues=[hexane_ua.name],
            forcefield_selection=get_mosdef_gomc_fn(
                "gmso_hexane_Exp6_periodic_dihedral_ua_K_energy_units.xml"
            ),
            atom_type_naming_style="general",
        )
        charmm.write_inp()

        with open("exp6_data.inp", "r") as fp:
            masses_read = False
            bonds_read = False
            angles_read = False
            dihedrals_read = False
            nonbondeds_read = False
            out_gomc = fp.readlines()
            for i, line in enumerate(out_gomc):
                if (
                    "! atom_types" in line
                    and "mass" in line
                    and "atomClass_ResidueName" in line
                ):
                    masses_read = True
                    assert len(out_gomc[i + 1].split("!")[0].split()) == 3
                    assert out_gomc[i + 1].split("!")[0].split()[0:3] == [
                        "*",
                        "CH3",
                        "15.035",
                    ]
                    assert len(out_gomc[i + 2].split("!")[0].split()) == 3
                    assert out_gomc[i + 2].split("!")[0].split()[0:3] == [
                        "*",
                        "CH2",
                        "14.027",
                    ]
                    assert out_gomc[i + 1].split()[4:5] == ["HEX_CH3_sp3"]
                    assert out_gomc[i + 2].split()[4:5] == ["HEX_CH2_sp3"]

                elif (
                    "! type_1" in line
                    and "type_2" in line
                    and "Kb" in line
                    and "b0" in line
                    and "extended_type_1" in line
                    and "extended_type_2" in line
                ):
                    bonds_read = True
                    bond_types = [
                        ["CH3", "CH2", "604267.5553", "1.687"],
                        ["CH2", "CH2", "604267.5553", "1.535"],
                    ]
                    assert len(out_gomc[i + 1].split("!")[0].split()) == 4
                    assert len(out_gomc[i + 2].split("!")[0].split()) == 4
                    if (
                        out_gomc[i + 1].split("!")[0].split()[0:4]
                        == bond_types[0]
                    ):
                        assert (
                            out_gomc[i + 1].split("!")[0].split()[0:4]
                            == bond_types[0]
                        )
                        assert (
                            out_gomc[i + 2].split("!")[0].split()[0:4]
                            == bond_types[1]
                        )
                    elif (
                        out_gomc[i + 1].split("!")[0].split()[0:4]
                        == bond_types[1]
                    ):
                        assert (
                            out_gomc[i + 1].split("!")[0].split()[0:4]
                            == bond_types[1]
                        )
                        assert (
                            out_gomc[i + 2].split("!")[0].split()[0:4]
                            == bond_types[0]
                        )

                elif (
                    "! type_1 " in line
                    and "type_2" in line
                    and "type_3" in line
                    and "Ktheta" in line
                    and "Theta0" in line
                    and "extended_type_1" in line
                    and "extended_type_2" in line
                    and "extended_type_3" in line
                ):
                    angles_read = True
                    angle_types = [
                        ["CH2", "CH2", "CH3", "62500.0", "114.0"],
                        ["CH2", "CH2", "CH2", "62500.0", "114.0"],
                    ]
                    assert len(out_gomc[i + 1].split("!")[0].split()) == 5
                    assert len(out_gomc[i + 2].split("!")[0].split()) == 5
                    if (
                        out_gomc[i + 1].split("!")[0].split()[0:5]
                        == angle_types[0]
                    ):
                        assert (
                            out_gomc[i + 1].split("!")[0].split()[0:5]
                            == angle_types[0]
                        )
                        assert (
                            out_gomc[i + 2].split("!")[0].split()[0:5]
                            == angle_types[1]
                        )
                    elif (
                        out_gomc[i + 1].split("!")[0].split()[0:4]
                        == angle_types[1]
                    ):
                        assert (
                            out_gomc[i + 1].split("!")[0].split()[0:5]
                            == angle_types[1]
                        )
                        assert (
                            out_gomc[i + 2].split("!")[0].split()[0:5]
                            == angle_types[0]
                        )

                elif (
                    "! type_1" in line
                    and "type_2" in line
                    and "type_3" in line
                    and "type_4" in line
                    and "Kchi" in line
                    and "n" in line
                    and "delta" in line
                    and "extended_type_1" in line
                    and "extended_type_2" in line
                    and "extended_type_3" in line
                    and "extended_type_4" in line
                ):
                    dihedrals_read = True
                    dihed_types = [
                        ["CH3", "CH2", "CH2", "CH2", "1078.16", "0", "90.0"],
                        ["CH3", "CH2", "CH2", "CH2", "-177.515", "1", "180.0"],
                        ["CH3", "CH2", "CH2", "CH2", "34.095", "2", "0.0"],
                        ["CH3", "CH2", "CH2", "CH2", "-395.66", "3", "180.0"],
                        ["CH2", "CH2", "CH2", "CH2", "1078.16", "0", "90.0"],
                        ["CH2", "CH2", "CH2", "CH2", "-177.515", "1", "180.0"],
                        ["CH2", "CH2", "CH2", "CH2", "34.095", "2", "0.0"],
                        ["CH2", "CH2", "CH2", "CH2", "-395.66", "3", "180.0"],
                    ]
                    for j in range(0, len(dihed_types)):
                        assert (
                            len(out_gomc[i + 1 + j].split("!")[0].split()) == 7
                        )
                        assert (
                            out_gomc[i + 1 + j].split("!")[0].split()[0:7]
                            == dihed_types[j]
                        )

                elif (
                    "! type_1" in line
                    and "epsilon" in line
                    and "sigma" in line
                    and "alpha" in line
                    and "epsilon,1-4" in line
                    and "sigma,1-4" in line
                    and "alpha,1-4" in line
                    and "extended_type_1" in line
                    and "extended_type_2" in line
                ):
                    nonbondeds_read = True
                    nb_types = [
                        [
                            "CH3",
                            "98.0",
                            "3.6790000166",
                            "16.0",
                            "0.0",
                            "3.6790000166",
                            "16.0",
                        ],
                        [
                            "CH2",
                            "98.2",
                            "4.5818082699",
                            "16.2",
                            "0.0",
                            "4.5818082699",
                            "16.2",
                        ],
                    ]

                    for j in range(0, len(nb_types)):
                        assert (
                            len(out_gomc[i + 1 + j].split("!")[0].split()) == 7
                        )
                        assert (
                            out_gomc[i + 1 + j].split("!")[0].split()[0:7]
                            == nb_types[j]
                        )

                else:
                    pass

        assert masses_read
        assert bonds_read
        assert angles_read
        assert dihedrals_read
        assert nonbondeds_read

    # test the Exp6 eqn when it is 2x higher than the standard
    def test_save_Exp6_eqn_times_2_gomc_ff(self, hexane_ua):
        box_0 = mb.fill_box(
            compound=[hexane_ua], n_compounds=[2], box=[4, 4, 4]
        )

        charmm = Charmm(
            box_0,
            "test_save_Exp6_eqn_times_2_gomc_ff",
            ff_filename="test_save_Exp6_eqn_times_2_gomc_ff",
            residues=[hexane_ua.name],
            forcefield_selection=get_mosdef_gomc_fn(
                "gmso_hexane_Exp6_eqn_times_2_periodic_dihedral_ua_K_energy.xml"
            ),
            atom_type_naming_style="general",
        )
        charmm.write_inp()

        with open("test_save_Exp6_eqn_times_2_gomc_ff.inp", "r") as fp:
            nonbondeds_read = False
            out_gomc = fp.readlines()
            for i, line in enumerate(out_gomc):
                if (
                    "! type_1" in line
                    and "epsilon" in line
                    and "sigma" in line
                    and "alpha" in line
                    and "epsilon,1-4" in line
                    and "sigma,1-4" in line
                    and "alpha,1-4" in line
                    and "extended_type_1" in line
                    and "extended_type_2" in line
                ):
                    nonbondeds_read = True
                    nb_types = [
                        [
                            "CH3",
                            "196.0",
                            "3.6790000166",
                            "16.0",
                            "0.0",
                            "3.6790000166",
                            "16.0",
                        ],
                        [
                            "CH2",
                            "196.4",
                            "4.5818082699",
                            "16.2",
                            "0.0",
                            "4.5818082699",
                            "16.2",
                        ],
                    ]

                    for j in range(0, len(nb_types)):
                        assert (
                            len(out_gomc[i + 1 + j].split("!")[0].split()) == 7
                        )
                        assert (
                            out_gomc[i + 1 + j].split("!")[0].split()[0:7]
                            == nb_types[j]
                        )

                else:
                    pass

        assert nonbondeds_read

    # test the LJ equation when it is 2x higher than the standard
    def test_save_charmm_water_LJ_eqn_times_2(self, water):
        box_0 = mb.fill_box(compound=[water], n_compounds=[1], box=[5, 4, 3])

        charmm = Charmm(
            box_0,
            "ctest_save_charmm_water_LJ_eqn_times_2",
            ff_filename="test_save_charmm_water_LJ_eqn_times_2",
            residues=[water.name],
            forcefield_selection={
                water.name: get_mosdef_gomc_fn(
                    "gmso_spce_water_LJ_eqn_times_2_one_for_nb_and_coul__lorentz_combining.xml"
                ),
            },
            bead_to_atom_name_dict={"_CH3": "C", "_CH2": "C", "_HC": "C"},
            gomc_fix_bonds_angles=[water.name],
            atom_type_naming_style="general",
        )
        charmm.write_inp()

        with open("test_save_charmm_water_LJ_eqn_times_2.inp", "r") as fp:
            nonbondeds_read = False
            out_gomc = fp.readlines()
            for i, line in enumerate(out_gomc):
                if (
                    "! type_1" in line
                    and "ignored" in line
                    and "epsilon" in line
                    and "Rmin/2" in line
                    and "ignored" in line
                    and "epsilon,1-4" in line
                    and "Rmin/2,1-4" in line
                    and "extended_type_1" in line
                    and "extended_type_2" in line
                ):
                    nonbondeds_read = True
                    nb_types = [
                        [
                            "OW",
                            "0.0",
                            "-0.3108001912",
                            "1.7766160931",
                            "0.0",
                            "-0.3108001912",
                            "1.7766160931",
                        ],
                        [
                            "HW",
                            "0.0",
                            "-0.0",
                            "0.5612310242",
                            "0.0",
                            "-0.0",
                            "0.5612310242",
                        ],
                    ]

                    for j in range(0, len(nb_types)):
                        assert (
                            len(out_gomc[i + 1 + j].split("!")[0].split()) == 7
                        )
                        assert (
                            out_gomc[i + 1 + j].split("!")[0].split()[0:7]
                            == nb_types[j]
                        )

                else:
                    pass

        assert nonbondeds_read

    # test the Mie equation when it is 2x higher than the standard
    def test_save_charmm_Mie_eqn_times_2(self, water, two_propanol_ua):
        box_0 = mb.fill_box(
            compound=[water, two_propanol_ua], n_compounds=[1, 1], box=[5, 4, 3]
        )

        charmm = Charmm(
            box_0,
            "test_save_charmm_mie_LJ_eqn_times_2",
            ff_filename="test_save_charmm_mie_LJ_eqn_times_2",
            residues=[water.name, two_propanol_ua.name],
            forcefield_selection={
                water.name: get_mosdef_gomc_fn(
                    "gmso_spce_water__lorentz_combining.xml"
                ),
                two_propanol_ua.name: get_mosdef_gomc_fn(
                    "gmso_two_propanol_Mie_eqn_time_2_ua.xml"
                ),
            },
            bead_to_atom_name_dict={"_CH3": "C", "_CH2": "C", "_HC": "C"},
            gomc_fix_bonds_angles=[water.name],
            atom_type_naming_style="general",
        )
        charmm.write_inp()

        with open("test_save_charmm_mie_LJ_eqn_times_2.inp", "r") as fp:
            nonbondeds_read = False
            out_gomc = fp.readlines()
            for i, line in enumerate(out_gomc):
                if (
                    "! type_1" in line
                    and "epsilon" in line
                    and "sigma" in line
                    and "n" in line
                    and "epsilon,1-4" in line
                    and "sigma,1-4" in line
                    and "n,1-4" in line
                    and "extended_type_1" in line
                    and "extended_type_2" in line
                ):
                    nonbondeds_read = True
                    nb_types = [
                        [
                            "OW",
                            "78.200368",
                            "3.16557",
                            "12.0",
                            "0.0",
                            "3.16557",
                            "12.0",
                        ],
                        [
                            "HW",
                            "0.0",
                            "1.0",
                            "12.0",
                            "0.0",
                            "1.0",
                            "12.0",
                        ],
                        [
                            "CH3",
                            "196.000011",
                            "3.751",
                            "11.0",
                            "0.0",
                            "3.751",
                            "11.0",
                        ],
                        [
                            "CH",
                            "20.000001",
                            "4.681",
                            "12.0",
                            "0.0",
                            "4.681",
                            "12.0",
                        ],
                        [
                            "O",
                            "186.000011",
                            "3.021",
                            "13.0",
                            "0.0",
                            "3.021",
                            "13.0",
                        ],
                        [
                            "H",
                            "0.0",
                            "0.0",
                            "14.0",
                            "0.0",
                            "0.0",
                            "14.0",
                        ],
                    ]

                    for j in range(0, len(nb_types)):
                        print("**************")
                        print(
                            f'output = {out_gomc[i + 1 + j].split("!")[0].split()[0:7]}'
                        )
                        print("**************")
                        print(f"std = {nb_types[j]}")
                        print("**************")
                        assert (
                            len(out_gomc[i + 1 + j].split("!")[0].split()) == 7
                        )
                        assert (
                            out_gomc[i + 1 + j].split("!")[0].split()[0:7]
                            == nb_types[j]
                        )

                else:
                    pass

        assert nonbondeds_read

    def test_save_amber_aa_butane_with_fake_improper_gmso_ff(self):
        butane_aa = mb.load(get_mosdef_gomc_fn("butane_aa.mol2"))
        butane_aa.name = "BUT"
        box_0 = mb.fill_box(
            compound=[butane_aa], n_compounds=[2], box=[20, 20, 20]
        )

        charmm = Charmm(
            box_0,
            "charmm_ff_style_amber_ff_data",
            ff_filename="charmm_ff_style_amber_ff_data",
            residues=[butane_aa.name],
            forcefield_selection=get_mosdef_gomc_fn(
                "amber_aa_butane_CT_CT_CT_CT_with_fake_improper_gmso.xml"
            ),
            atom_type_naming_style="general",
        )
        charmm.write_inp()

        with open("charmm_ff_style_amber_ff_data.inp", "r") as fp:
            masses_read = False
            bonds_read = False
            angles_read = False
            dihedrals_read = False
            impropers_read = False
            nonbondeds_read = False
            out_gomc = fp.readlines()
            for i, line in enumerate(out_gomc):
                if (
                    "! atom_types" in line
                    and "mass" in line
                    and "atomClass_ResidueName" in line
                ):
                    masses_read = True
                    assert len(out_gomc[i + 1].split("!")[0].split()) == 3
                    assert out_gomc[i + 1].split("!")[0].split()[0:3] == [
                        "*",
                        "CT",
                        "12.011",
                    ]
                    assert len(out_gomc[i + 2].split("!")[0].split()) == 3
                    assert out_gomc[i + 2].split("!")[0].split()[0:3] == [
                        "*",
                        "HC",
                        "1.008",
                    ]
                    assert out_gomc[i + 1].split()[4:5] == ["BUT_C_CTH3"]
                    assert out_gomc[i + 2].split()[4:5] == ["BUT_H_CTH3"]

                elif (
                    "! type_1" in line
                    and "type_2" in line
                    and "Kb" in line
                    and "b0" in line
                    and "extended_type_1" in line
                    and "extended_type_2" in line
                ):
                    bonds_read = True
                    bond_types = [
                        ["CT", "CT", "310.0", "1.526"],
                        ["CT", "HC", "340.0", "1.09"],
                    ]
                    assert len(out_gomc[i + 1].split("!")[0].split()) == 4
                    assert len(out_gomc[i + 2].split("!")[0].split()) == 4
                    if (
                        out_gomc[i + 1].split("!")[0].split()[0:4]
                        == bond_types[0]
                    ):
                        assert (
                            out_gomc[i + 1].split("!")[0].split()[0:4]
                            == bond_types[0]
                        )
                        assert (
                            out_gomc[i + 2].split("!")[0].split()[0:4]
                            == bond_types[1]
                        )
                    elif (
                        out_gomc[i + 1].split("!")[0].split()[0:4]
                        == bond_types[1]
                    ):
                        assert (
                            out_gomc[i + 1].split("!")[0].split()[0:4]
                            == bond_types[1]
                        )
                        assert (
                            out_gomc[i + 2].split("!")[0].split()[0:4]
                            == bond_types[0]
                        )

                elif (
                    "! type_1 " in line
                    and "type_2" in line
                    and "type_3" in line
                    and "Ktheta" in line
                    and "Theta0" in line
                    and "extended_type_1" in line
                    and "extended_type_2" in line
                    and "extended_type_3" in line
                ):
                    angles_read = True
                    angle_types = [
                        ["CT", "CT", "HC", "50.0", "109.5"],
                        ["HC", "CT", "HC", "35.0", "109.5"],
                        ["CT", "CT", "CT", "40.0", "109.5"],
                    ]
                    assert len(out_gomc[i + 1].split("!")[0].split()) == 5
                    assert len(out_gomc[i + 2].split("!")[0].split()) == 5
                    if (
                        out_gomc[i + 1].split("!")[0].split()[0:5]
                        == angle_types[0]
                    ):
                        assert (
                            out_gomc[i + 1].split("!")[0].split()[0:5]
                            == angle_types[0]
                        )
                        assert (
                            out_gomc[i + 2].split("!")[0].split()[0:5]
                            == angle_types[1]
                        )
                    elif (
                        out_gomc[i + 1].split("!")[0].split()[0:4]
                        == angle_types[1]
                    ):
                        assert (
                            out_gomc[i + 1].split("!")[0].split()[0:5]
                            == angle_types[1]
                        )
                        assert (
                            out_gomc[i + 2].split("!")[0].split()[0:5]
                            == angle_types[0]
                        )

                elif (
                    "! type_1" in line
                    and "type_2" in line
                    and "type_3" in line
                    and "type_4" in line
                    and "Kchi" in line
                    and "n" in line
                    and "delta" in line
                    and "extended_type_1" in line
                    and "extended_type_2" in line
                    and "extended_type_3" in line
                    and "extended_type_4" in line
                ):
                    dihedrals_read = True
                    dihed_types = [
                        ["HC", "CT", "CT", "HC", "-0.15", "3", "180.0"],
                        ["CT", "CT", "CT", "HC", "-0.16", "3", "180.0"],
                        ["CT", "CT", "CT", "CT", "0.0", "1", "180.0"],
                    ]
                    for j in range(0, len(dihed_types)):
                        assert (
                            len(out_gomc[i + 1 + j].split("!")[0].split()) == 7
                        )
                        assert (
                            out_gomc[i + 1 + j].split("!")[0].split()[0:7]
                            == dihed_types[j]
                        )

                elif (
                    "! type_1" in line
                    and "type_2" in line
                    and "type_3" in line
                    and "type_4" in line
                    and "Kw" in line
                    and "n" in line
                    and "w0" in line
                    and "extended_type_1" in line
                    and "extended_type_2" in line
                    and "extended_type_3" in line
                    and "extended_type_4" in line
                ):
                    impropers_read = True
                    improper_types = [
                        ["CT", "HC", "HC", "HC", "-2.0", "1", "180.0"],
                        ["CT", "HC", "HC", "HC", "1.0", "3", "0.0"],
                    ]

                    for j in range(0, len(improper_types)):
                        assert (
                            len(out_gomc[i + 1 + j].split("!")[0].split()) == 7
                        )
                        assert (
                            out_gomc[i + 1 + j].split("!")[0].split()[0:7]
                            == improper_types[j]
                        )

                elif (
                    "! type_1" in line
                    and "ignored" in line
                    and "epsilon" in line
                    and "Rmin/2" in line
                    and "ignored" in line
                    and "epsilon,1-4" in line
                    and "Rmin/2,1-4" in line
                    and "extended_type_1" in line
                    and "extended_type_2" in line
                ):
                    nonbondeds_read = True
                    nb_types = [
                        [
                            "CT",
                            "0.0",
                            "-0.1094000956",
                            "1.9080002759",
                            "0.0",
                            "-0.0547000478",
                            "1.9080002759",
                        ],
                        [
                            "HC",
                            "0.0",
                            "-0.0157",
                            "1.4869984354",
                            "0.0",
                            "-0.00785",
                            "1.4869984354",
                        ],
                    ]

                    for j in range(0, len(nb_types)):
                        assert (
                            len(out_gomc[i + 1 + j].split("!")[0].split()) == 7
                        )
                        assert (
                            out_gomc[i + 1 + j].split("!")[0].split()[0:7]
                            == nb_types[j]
                        )

                else:
                    pass

        assert masses_read
        assert bonds_read
        assert angles_read
        assert dihedrals_read
        assert impropers_read
        assert nonbondeds_read
