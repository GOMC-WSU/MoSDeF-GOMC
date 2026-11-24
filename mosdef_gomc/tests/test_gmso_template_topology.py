from types import SimpleNamespace

import pytest

from mosdef_gomc.formats.gmso_charmm_writer import unique_atom_naming
from mosdef_gomc.tests.base_test import BaseTest
from mosdef_gomc.utils.gmso_template_topology import (
    MockConnection,
    MockSite,
    TemplateTopologyProxy,
    combine_proxies,
)


class _FakeSite:
    """Hashable stand-in for a gmso.Site."""

    def __init__(self, name, resname):
        self.name = name
        self.atom_type = f"type_{name}"
        self.name_ = name
        self.residue_name_ = resname


class _FakeConnection:
    def __init__(self, members, type_attr, type_value):
        self.connection_members = members
        setattr(self, type_attr, type_value)


class _FakeTemplateTopology:
    """A 4-site residue 'ABC' with one of each connection, plus a 1-site 'X'."""

    def __init__(self):
        a, b, c, d = (_FakeSite(n, "ABC") for n in ("A", "B", "C", "D"))
        x = _FakeSite("X", "X")
        self.sites = [a, b, c, d, x]
        self.bonds = [_FakeConnection((a, b), "bond_type", "bt")]
        self.angles = [_FakeConnection((a, b, c), "angle_type", "at")]
        self.dihedrals = [_FakeConnection((a, b, c, d), "dihedral_type", "dt")]
        self.impropers = [_FakeConnection((b, a, c, d), "improper_type", "it")]
        self.box = "box"
        self.atom_types = ["type_A"]
        self.bond_types = ["bt"]
        self.angle_types = ["at"]
        self.dihedral_types = ["dt"]
        self.improper_types = ["it"]

    def unique_site_labels(self, label_type, name_only=False):
        return ("labels", label_type, name_only)

    def create_subtop(self, label_type, label):
        return ("subtop", label_type, label)

    def get_lj_scale(self, molecule_id):
        return ("lj", molecule_id)

    def get_electrostatics_scale(self, molecule_id):
        return ("elec", molecule_id)


class TestTemplateTopology(BaseTest):
    def test_mock_site_and_connection(self):
        site = MockSite("C1", "type_C", "RES", 3, 7, position=[1, 2, 3])
        assert site.name == "C1"
        assert site.atom_type == "type_C"
        assert site.index == 7
        assert site.occupancy == 0.0
        assert site.__dict__ == {
            "residue_name_": "RES",
            "residue_number_": 3,
            "name_": "C1",
            "position_": [1, 2, 3],
        }
        assert "position_" not in MockSite("C1", "t", "R", 1, 0).__dict__

        conn = MockConnection((site, site))
        assert conn.connection_members == (site, site)
        assert conn.connection_type is None

    def test_proxy_expands_all_connection_kinds(self):
        template = _FakeTemplateTopology()
        positions = [[float(i)] * 3 for i in range(9)]
        proxy = TemplateTopologyProxy(
            template, ["ABC", "X", "ABC"], positions=positions
        )

        assert proxy.box == "box"
        assert proxy.n_sites == 9
        assert [s.name for s in proxy.sites] == list("ABCDXABCD")
        assert [s.__dict__["residue_number_"] for s in proxy.sites] == [
            1, 1, 1, 1, 2, 3, 3, 3, 3,
        ]  # fmt: skip
        assert [proxy.get_index(s) for s in proxy.sites] == list(range(9))
        assert proxy.sites[5].__dict__["position_"] == [5.0] * 3

        assert proxy.n_bonds == 2
        assert proxy.n_angles == 2
        assert proxy.n_dihedrals == 2
        assert proxy.n_impropers == 2
        assert len(list(proxy.connections)) == 8

        # The second ABC residue's improper must point at its own sites.
        improper = proxy.impropers[1]
        assert improper.connection_type == "it"
        assert [s.index for s in improper.connection_members] == [6, 5, 7, 8]
        assert proxy.bonds[0].connection_type == "bt"
        assert proxy.angles[0].connection_type == "at"
        assert proxy.dihedrals[0].connection_type == "dt"

    def test_proxy_without_positions_and_delegation(self):
        template = _FakeTemplateTopology()
        proxy = TemplateTopologyProxy(template, ["X"])
        assert "position_" not in proxy.sites[0].__dict__

        assert proxy.atom_types == ["type_A"]
        assert proxy.bond_types == ["bt"]
        assert proxy.angle_types == ["at"]
        assert proxy.dihedral_types == ["dt"]
        assert proxy.improper_types == ["it"]
        assert proxy.unique_site_labels("molecule", True) == (
            "labels",
            "molecule",
            True,
        )
        assert proxy.create_subtop("molecule", "X") == (
            "subtop",
            "molecule",
            "X",
        )
        assert proxy.get_lj_scale("X") == ("lj", "X")
        assert proxy.get_electrostatics_scale("X") == ("elec", "X")

    def test_proxy_connection_type_fallback(self):
        template = _FakeTemplateTopology()
        a, b = template.sites[:2]
        bond = _FakeConnection((a, b), "connection_type", "generic")
        template.bonds = [bond]
        template.angles = []
        template.dihedrals = []
        template.impropers = []
        proxy = TemplateTopologyProxy(template, ["ABC"])
        assert proxy.bonds[0].connection_type == "generic"

    def test_combine_proxies(self):
        template = _FakeTemplateTopology()
        proxy_0 = TemplateTopologyProxy(template, ["ABC", "X"])
        proxy_1 = TemplateTopologyProxy(template, ["X", "ABC"])

        assert combine_proxies(proxy_0, None) is proxy_0

        combined = combine_proxies(proxy_0, proxy_1)
        assert combined.n_sites == 10
        assert [s.index for s in combined.sites] == list(range(10))
        assert [s.__dict__["residue_number_"] for s in combined.sites] == [
            1, 1, 1, 1, 2, 3, 4, 4, 4, 4,
        ]  # fmt: skip
        assert combined.n_bonds == 2
        assert combined.n_impropers == 2

    def test_combine_proxies_empty_first_box(self):
        template = _FakeTemplateTopology()
        proxy_0 = TemplateTopologyProxy(template, [])
        proxy_1 = TemplateTopologyProxy(template, ["X"])
        combined = combine_proxies(proxy_0, proxy_1)
        assert combined.sites[0].__dict__["residue_number_"] == 1


def _naming_topology(names):
    return SimpleNamespace(
        sites=[SimpleNamespace(name_=name) for name in names]
    )


class TestUniqueAtomNaming(BaseTest):
    def test_three_character_element_names(self):
        topology = _naming_topology(["ABC", "ABC", "C", "C"])
        unique_dict, names, missing = unique_atom_naming(
            topology, [1, 1, 1, 1], ["RES"] * 4
        )
        assert names == ["ABC1", "ABC2", "C1", "C2"]
        assert missing == []
        assert unique_dict == {
            "1_RES_ABC1": 1,
            "1_RES_ABC2": 2,
            "1_RES_C1": 3,
            "1_RES_C2": 4,
        }

    def test_names_restart_per_residue(self):
        topology = _naming_topology(["C", "C", "C"])
        _, names, _ = unique_atom_naming(topology, [1, 1, 2], ["RES"] * 3)
        assert names == ["C1", "C2", "C1"]

    def test_four_character_element_name_warns(self):
        topology = _naming_topology(["ABCD"])
        with pytest.warns(UserWarning, match="more than 4 charaters"):
            result = unique_atom_naming(topology, [1], ["RES"])
        assert result == (None, None, None)
