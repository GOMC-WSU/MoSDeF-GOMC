"""Lightweight topology proxies used when writing files with ``use_template_ff``.

When ``use_template_ff=True``, only one copy (template) of each residue is
parameterized by GMSO.  The classes here expand those templates back out to the
full system, following the residue order of the original structure, without
building a full ``gmso.Topology``.
"""

import itertools

# (connection attribute on the topology, attribute holding the potential type)
_CONNECTION_KINDS = (
    ("bonds", "bond_type"),
    ("angles", "angle_type"),
    ("dihedrals", "dihedral_type"),
    ("impropers", "improper_type"),
)


class MockSite:
    """A minimal stand-in for a ``gmso.Site`` in an expanded template topology."""

    __slots__ = ["name", "atom_type", "__dict__", "index", "occupancy"]

    def __init__(self, name, atom_type, resname, resnum, index, position=None):
        self.name = name
        self.atom_type = atom_type
        self.index = index
        self.occupancy = 0.0
        self.__dict__ = {
            "residue_name_": resname,
            "residue_number_": resnum,
            "name_": name,
        }
        if position is not None:
            self.__dict__["position_"] = position


class MockConnection:
    """A minimal stand-in for a gmso bond, angle, dihedral, or improper."""

    __slots__ = ["connection_members", "connection_type"]

    def __init__(self, members, connection_type=None):
        self.connection_members = members
        self.connection_type = connection_type


def _connection_type(connection, type_attr):
    """Return the potential type of a gmso connection (or None)."""
    return getattr(
        connection, type_attr, getattr(connection, "connection_type", None)
    )


class TemplateTopologyProxy:
    """Expand a GMSO template topology to the full system.

    Parameters
    ----------
    template_top : gmso.Topology
        Topology containing one parameterized copy of each residue.
    sequence : list of str
        The residue name of every molecule in the full system, in order.
    positions : unyt array (N, 3), optional
        Positions of every site in the full system.
    """

    def __init__(self, template_top, sequence, positions=None):
        self.template_top = template_top
        self.sequence = sequence
        if hasattr(template_top, "box"):
            self.box = template_top.box

        self.sites = []
        self.bonds = []
        self.angles = []
        self.dihedrals = []
        self.impropers = []

        # residue name -> {"sites": [...], "bonds": [...], ...}
        self.templates = {}
        # template site -> its index within its residue template
        self._site_to_template_idx = {}
        for resname in set(sequence):
            self._build_template(resname)

        for res_num, resname in enumerate(sequence, start=1):
            self._expand_residue(resname, res_num, positions)

    def _build_template(self, resname):
        """Collect the sites and connections of one residue template."""
        t_sites = [
            s
            for s in self.template_top.sites
            if s.__dict__.get("residue_name_") == resname
        ]
        site_set = set(t_sites)
        template = {"sites": t_sites}
        for kind, _ in _CONNECTION_KINDS:
            template[kind] = [
                c
                for c in getattr(self.template_top, kind)
                if c.connection_members[0] in site_set
            ]
        self.templates[resname] = template
        for i, site in enumerate(t_sites):
            self._site_to_template_idx[site] = i

    def _expand_residue(self, resname, res_num, positions):
        """Append a copy of the ``resname`` template as residue ``res_num``."""
        template = self.templates[resname]
        mol_sites = []
        for t_site in template["sites"]:
            index = len(self.sites)
            site = MockSite(
                t_site.name,
                t_site.atom_type,
                resname,
                res_num,
                index,
                position=positions[index] if positions is not None else None,
            )
            self.sites.append(site)
            mol_sites.append(site)

        for kind, type_attr in _CONNECTION_KINDS:
            out = getattr(self, kind)
            for conn in template[kind]:
                members = tuple(
                    mol_sites[self._site_to_template_idx[m]]
                    for m in conn.connection_members
                )
                out.append(
                    MockConnection(members, _connection_type(conn, type_attr))
                )

    @property
    def n_sites(self):
        """Number of sites in the expanded topology."""
        return len(self.sites)

    @property
    def n_bonds(self):
        """Number of bonds in the expanded topology."""
        return len(self.bonds)

    @property
    def n_angles(self):
        """Number of angles in the expanded topology."""
        return len(self.angles)

    @property
    def n_dihedrals(self):
        """Number of dihedrals in the expanded topology."""
        return len(self.dihedrals)

    @property
    def n_impropers(self):
        """Number of impropers in the expanded topology."""
        return len(self.impropers)

    @property
    def connections(self):
        """Iterate over all bonds, angles, dihedrals, and impropers."""
        return itertools.chain(
            self.bonds, self.angles, self.dihedrals, self.impropers
        )

    @property
    def bond_types(self):
        """Bond types of the template topology."""
        return getattr(self.template_top, "bond_types", [])

    @property
    def angle_types(self):
        """Angle types of the template topology."""
        return getattr(self.template_top, "angle_types", [])

    @property
    def dihedral_types(self):
        """Dihedral types of the template topology."""
        return getattr(self.template_top, "dihedral_types", [])

    @property
    def improper_types(self):
        """Improper types of the template topology."""
        return getattr(self.template_top, "improper_types", [])

    @property
    def atom_types(self):
        """Atom types of the template topology."""
        return getattr(self.template_top, "atom_types", [])

    @staticmethod
    def get_index(site):
        """Return the index of ``site`` in the expanded topology."""
        return site.index

    def unique_site_labels(self, label_type, name_only=False):
        """Delegate to the template topology."""
        return self.template_top.unique_site_labels(label_type, name_only)

    def create_subtop(self, label_type, label):
        """Delegate to the template topology."""
        return self.template_top.create_subtop(label_type, label)

    def get_lj_scale(self, molecule_id):
        """Delegate to the template topology."""
        return self.template_top.get_lj_scale(molecule_id)

    def get_electrostatics_scale(self, molecule_id):
        """Delegate to the template topology."""
        return self.template_top.get_electrostatics_scale(molecule_id)


def combine_proxies(proxy1, proxy2):
    """Combine the box 0 and box 1 proxies into one proxy.

    The sites of ``proxy2`` are re-indexed and re-numbered in place, so
    ``proxy2`` should not be used on its own afterwards.
    """
    if proxy2 is None:
        return proxy1

    combined = TemplateTopologyProxy(proxy1.template_top, [])
    for kind in ("sites", "bonds", "angles", "dihedrals", "impropers"):
        setattr(combined, kind, getattr(proxy1, kind) + getattr(proxy2, kind))

    site_offset = proxy1.n_sites
    res_offset = (
        proxy1.sites[-1].__dict__["residue_number_"] if proxy1.sites else 0
    )
    for site in proxy2.sites:
        site.index += site_offset
        site.__dict__["residue_number_"] += res_offset

    return combined
