import mbuild as mb
import pytest


class BaseTest:
    @pytest.fixture(autouse=True)
    def initdir(self, tmpdir):
        tmpdir.chdir()

    @pytest.fixture
    def water(self):
        water = mb.load("O", smiles=True)
        water.name = "WAT"

        return water

    @pytest.fixture
    def ethane_gomc(self):
        ethane_gomc = mb.load("CC", smiles=True)
        ethane_gomc.name = "ETH"

        return ethane_gomc

    @pytest.fixture
    def ethanol_gomc(self):
        ethanol_gomc = mb.load("CCO", smiles=True)
        ethanol_gomc.name = "ETO"

        return ethanol_gomc

    @pytest.fixture
    def methane_ua_gomc(self):
        methane_ua_gomc = mb.Compound(name="_CH4")

        return methane_ua_gomc

    @pytest.fixture
    def two_propanol_gomc(self):
        two_propanol_gomc = mb.load("CC(C)O", smiles=True)
        two_propanol_gomc.name = "TPR"
        return two_propanol_gomc

    @pytest.fixture
    def ethyl_ether_gomc(self):
        ethyl_ether_gomc = mb.load("CCOCC", smiles=True)
        ethyl_ether_gomc.name = "ETE"
        return ethyl_ether_gomc

    @pytest.fixture
    def methyl_ether_gomc(self):
        methyl_ether_gomc = mb.load("COC", smiles=True)
        methyl_ether_gomc.name = "MTE"
        return methyl_ether_gomc

    @pytest.fixture
    def two_propanol_ua(self):
        class TwoPropanolUA(mb.Compound):
            def __init__(self):
                super(TwoPropanolUA, self).__init__()
                self.name = "POL"

                CH3_1_1 = mb.Particle(pos=[0.2, 0.0, 0.0], name="_CH3")
                HC_1_1 = mb.Particle(pos=[0.4, 0.0, 0.0], name="_HC")
                O_1_1 = mb.Particle(pos=[0.8, 0.0, 0.0], name="O", element="O")
                H_1_1 = mb.Particle(pos=[1.0, 0.0, 0.0], name="H", element="H")
                CH3_1_2 = mb.Particle(pos=[0.6, 0.0, 0.0], name="_CH3")
                self.add([CH3_1_1, HC_1_1, O_1_1, H_1_1, CH3_1_2])

                port_R_CH3_1_1 = mb.Port(
                    anchor=CH3_1_1, orientation=[0.1, 0, 0], separation=0.05
                )
                port_L_HC_1_1 = mb.Port(
                    anchor=HC_1_1, orientation=[-0.1, 0, 0], separation=0.05
                )
                port_R_HC_1_1 = mb.Port(
                    anchor=HC_1_1, orientation=[0.1, 0, 0], separation=0.05
                )
                port_D_HC_1_1 = mb.Port(
                    anchor=HC_1_1, orientation=[0, -0.1, 0], separation=0.05
                )
                port_L_CH3_1_2 = mb.Port(
                    anchor=CH3_1_2, orientation=[-0.1, 0, 0], separation=0.05
                )
                port_L_O_1_1 = mb.Port(
                    anchor=O_1_1, orientation=[-0.1, 0, 0], separation=0.05
                )
                port_R_O_1_1 = mb.Port(
                    anchor=O_1_1, orientation=[0.1, 0, 0], separation=0.05
                )
                port_L_H_1_1 = mb.Port(
                    anchor=H_1_1, orientation=[-0.1, 0, 0], separation=0.05
                )

                self.add(port_R_CH3_1_1, label="port_R_CH3_1_1")
                self.add(port_L_HC_1_1, label="port_L_HC_1_1")
                self.add(port_R_HC_1_1, label="port_R_HC_1_1")
                self.add(port_L_CH3_1_2, label="port_L_CH3_1_2")
                self.add(port_D_HC_1_1, label="port_D_HC_1_1")
                self.add(port_L_O_1_1, label="port_L_O_1_1")
                self.add(port_R_O_1_1, label="port_R_O_1_1")
                self.add(port_L_H_1_1, label="port_L_H_1_1")

                mb.force_overlap(
                    move_this=HC_1_1,
                    from_positions=self["port_L_HC_1_1"],
                    to_positions=self["port_R_CH3_1_1"],
                )
                mb.force_overlap(
                    move_this=CH3_1_2,
                    from_positions=self["port_L_CH3_1_2"],
                    to_positions=self["port_R_HC_1_1"],
                )
                mb.force_overlap(
                    move_this=O_1_1,
                    from_positions=self["port_L_O_1_1"],
                    to_positions=self["port_D_HC_1_1"],
                )
                mb.force_overlap(
                    move_this=H_1_1,
                    from_positions=self["port_L_H_1_1"],
                    to_positions=self["port_R_O_1_1"],
                )

                self.energy_minimize(forcefield="trappe-ua", steps=10**9)

        return TwoPropanolUA()

    @pytest.fixture
    def alt_two_propanol_ua(self):
        class AltTwoPropanolUA(mb.Compound):
            def __init__(self):
                super(AltTwoPropanolUA, self).__init__()
                self.name = "SPL"

                CH3_1_1 = mb.Particle(pos=[0.2, 0.0, 0.0], name="_CH3")
                HC_1_1 = mb.Particle(pos=[0.4, 0.0, 0.0], name="_HC")
                O_1_1 = mb.Particle(pos=[0.8, 0.0, 0.0], name="O", element="O")
                H_1_1 = mb.Particle(pos=[1.0, 0.0, 0.0], name="H", element="H")
                CH3_1_2 = mb.Particle(pos=[0.6, 0.0, 0.0], name="_CH3")
                self.add([CH3_1_1, HC_1_1, O_1_1, H_1_1, CH3_1_2])

                port_R_CH3_1_1 = mb.Port(
                    anchor=CH3_1_1, orientation=[0.1, 0, 0], separation=0.05
                )
                port_L_HC_1_1 = mb.Port(
                    anchor=HC_1_1, orientation=[-0.1, 0, 0], separation=0.05
                )
                port_R_HC_1_1 = mb.Port(
                    anchor=HC_1_1, orientation=[0.1, 0, 0], separation=0.05
                )
                port_D_HC_1_1 = mb.Port(
                    anchor=HC_1_1, orientation=[0, -0.1, 0], separation=0.05
                )
                port_L_CH3_1_2 = mb.Port(
                    anchor=CH3_1_2, orientation=[-0.1, 0, 0], separation=0.05
                )
                port_L_O_1_1 = mb.Port(
                    anchor=O_1_1, orientation=[-0.1, 0, 0], separation=0.05
                )
                port_R_O_1_1 = mb.Port(
                    anchor=O_1_1, orientation=[0.1, 0, 0], separation=0.05
                )
                port_L_H_1_1 = mb.Port(
                    anchor=H_1_1, orientation=[-0.1, 0, 0], separation=0.05
                )

                self.add(port_R_CH3_1_1, label="port_R_CH3_1_1")
                self.add(port_L_HC_1_1, label="port_L_HC_1_1")
                self.add(port_R_HC_1_1, label="port_R_HC_1_1")
                self.add(port_L_CH3_1_2, label="port_L_CH3_1_2")
                self.add(port_D_HC_1_1, label="port_D_HC_1_1")
                self.add(port_L_O_1_1, label="port_L_O_1_1")
                self.add(port_R_O_1_1, label="port_R_O_1_1")
                self.add(port_L_H_1_1, label="port_L_H_1_1")

                mb.force_overlap(
                    move_this=HC_1_1,
                    from_positions=self["port_L_HC_1_1"],
                    to_positions=self["port_R_CH3_1_1"],
                )
                mb.force_overlap(
                    move_this=CH3_1_2,
                    from_positions=self["port_L_CH3_1_2"],
                    to_positions=self["port_R_HC_1_1"],
                )
                mb.force_overlap(
                    move_this=O_1_1,
                    from_positions=self["port_L_O_1_1"],
                    to_positions=self["port_D_HC_1_1"],
                )
                mb.force_overlap(
                    move_this=H_1_1,
                    from_positions=self["port_L_H_1_1"],
                    to_positions=self["port_R_O_1_1"],
                )

                self.energy_minimize(forcefield="trappe-ua", steps=10**9)

        return AltTwoPropanolUA()

    @pytest.fixture
    def hexane_ua(self):
        class HexaneUA(mb.Compound):
            def __init__(self):
                super(HexaneUA, self).__init__()
                self.name = "HEX"

                CH3_1_1 = mb.Particle(pos=[0.17, 0.0, 0.0], name="_CH3")
                CH2_1_2 = mb.Particle(pos=[0.34, 0.0, 0.0], name="_CH2")
                CH2_1_3 = mb.Particle(pos=[0.51, 0.0, 0.0], name="_CH2")
                CH2_1_4 = mb.Particle(pos=[0.68, 0.0, 0.0], name="_CH2")
                CH2_1_5 = mb.Particle(pos=[0.85, 0.0, 0.0], name="_CH2")
                CH3_1_6 = mb.Particle(pos=[1.02, 0.0, 0.0], name="_CH3")
                self.add([CH3_1_1, CH2_1_2, CH2_1_3, CH2_1_4, CH2_1_5, CH3_1_6])

                port_R_CH3_1_1 = mb.Port(
                    anchor=CH3_1_1, orientation=[0.1, 0, 0], separation=0.05
                )
                port_L_CH2_1_2 = mb.Port(
                    anchor=CH2_1_2, orientation=[-0.1, 0, 0], separation=0.05
                )
                port_R_CH2_1_2 = mb.Port(
                    anchor=CH2_1_2, orientation=[0.1, 0, 0], separation=0.05
                )
                port_L_CH2_1_3 = mb.Port(
                    anchor=CH2_1_3, orientation=[-0.1, 0, 0], separation=0.05
                )
                port_R_CH2_1_3 = mb.Port(
                    anchor=CH2_1_3, orientation=[0.1, 0, 0], separation=0.05
                )
                port_L_CH2_1_4 = mb.Port(
                    anchor=CH2_1_4, orientation=[-0.1, 0, 0], separation=0.05
                )
                port_R_CH2_1_4 = mb.Port(
                    anchor=CH2_1_4, orientation=[0.1, 0, 0], separation=0.05
                )
                port_L_CH2_1_5 = mb.Port(
                    anchor=CH2_1_5, orientation=[-0.1, 0, 0], separation=0.05
                )
                port_R_CH2_1_5 = mb.Port(
                    anchor=CH2_1_5, orientation=[0.1, 0, 0], separation=0.05
                )
                port_L_CH3_1_6 = mb.Port(
                    anchor=CH3_1_6, orientation=[-0.1, 0, 0], separation=0.05
                )

                self.add(port_R_CH3_1_1, label="port_R_CH3_1_1")
                self.add(port_L_CH2_1_2, label="port_L_CH2_1_2")
                self.add(port_R_CH2_1_2, label="port_R_CH2_1_2")
                self.add(port_L_CH2_1_3, label="port_L_CH2_1_3")
                self.add(port_R_CH2_1_3, label="port_R_CH2_1_3")
                self.add(port_L_CH2_1_4, label="port_L_CH2_1_4")
                self.add(port_R_CH2_1_4, label="port_R_CH2_1_4")
                self.add(port_L_CH2_1_5, label="port_L_CH2_1_5")
                self.add(port_R_CH2_1_5, label="port_R_CH2_1_5")
                self.add(port_L_CH3_1_6, label="port_L_CH3_1_6")

                mb.force_overlap(
                    move_this=CH2_1_2,
                    from_positions=self["port_L_CH2_1_2"],
                    to_positions=self["port_R_CH3_1_1"],
                )
                mb.force_overlap(
                    move_this=CH2_1_3,
                    from_positions=self["port_L_CH2_1_3"],
                    to_positions=self["port_R_CH2_1_2"],
                )
                mb.force_overlap(
                    move_this=CH2_1_4,
                    from_positions=self["port_L_CH2_1_4"],
                    to_positions=self["port_R_CH2_1_3"],
                )
                mb.force_overlap(
                    move_this=CH2_1_5,
                    from_positions=self["port_L_CH2_1_5"],
                    to_positions=self["port_R_CH2_1_4"],
                )
                mb.force_overlap(
                    move_this=CH3_1_6,
                    from_positions=self["port_L_CH3_1_6"],
                    to_positions=self["port_R_CH2_1_5"],
                )

        return HexaneUA()
