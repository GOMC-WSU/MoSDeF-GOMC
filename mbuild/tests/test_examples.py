#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `mbuild.examples` module. """
import pytest
from base_test import BaseTest


class TestExample(BaseTest):

    def test_alkane(self):
        import mbuild.examples.alkane.alkane as example
        example.main()

    def test_alkane_monolayer(self):
        import mbuild.examples.alkane_monolayer.alkane_monolayer as example
        example.main()

    def test_bilayer(self):
        import mbuild.examples.bilayer.bilayer as example
        example.main()

    def test_ethane(self):
        import mbuild.examples.ethane.ethane as example
        ethane = example.main()
        assert ethane.n_atoms == 8
        assert ethane.n_bonds == 7
        ethane = ethane.to_trajectory()

        ethane.top.find_forcefield_terms()
        assert ethane.topology.n_ff_bonds == 7
        assert ethane.topology.n_ff_angles == 12
        assert ethane.topology.n_ff_dihedrals == 9

        ethane.save('ethane.hoomdxml')

    def test_methane(self):
        import mbuild.examples.methane.methane as example
        example.main()

    def test_pmpc(self):
        import mbuild.examples.pmpc.pmpc_brush_layer as example
        example.main()

    def test_reload(self):
        import mbuild.examples.reload.reload as example
        example.main()

    def test_solvate(self):
        import mbuild.examples.solvate.solvate as example
        example.main()

    def test_tnp(self):
        import mbuild.examples.tnp.tnp as example
        example.main()