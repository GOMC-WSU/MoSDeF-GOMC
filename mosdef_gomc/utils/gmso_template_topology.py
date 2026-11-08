class MockSite:
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
    __slots__ = ["connection_members", "connection_type"]

    def __init__(self, members, connection_type=None):
        self.connection_members = members
        self.connection_type = connection_type


class TemplateTopologyProxy:
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

        # We need to map residue names to their template components
        self.templates = {}

        # Map original site to its index in the template to reconstruct connections
        site_to_template_idx = {}
        for resname in set(sequence):
            # Extract sites
            t_sites = [
                s
                for s in template_top.sites
                if s.__dict__.get("residue_name_") == resname
            ]
            t_bonds = [
                b
                for b in template_top.bonds
                if b.connection_members[0] in t_sites
            ]
            t_angles = [
                a
                for a in template_top.angles
                if a.connection_members[0] in t_sites
            ]
            t_dihedrals = [
                d
                for d in template_top.dihedrals
                if d.connection_members[0] in t_sites
            ]
            t_impropers = [
                i
                for i in template_top.impropers
                if i.connection_members[0] in t_sites
            ]

            self.templates[resname] = {
                "sites": t_sites,
                "bonds": t_bonds,
                "angles": t_angles,
                "dihedrals": t_dihedrals,
                "impropers": t_impropers,
            }
            for i, s in enumerate(t_sites):
                site_to_template_idx[s] = i

        # Now expand
        site_offset = 0
        res_num = 1
        for resname in sequence:
            t = self.templates[resname]

            # Create MockSites
            mol_sites = []
            for t_site in t["sites"]:
                pos = positions[site_offset] if positions is not None else None
                s = MockSite(
                    t_site.name,
                    t_site.atom_type,
                    resname,
                    res_num,
                    site_offset,
                    position=pos,
                )
                # Copy tags_ if present on atom_type
                if (
                    hasattr(t_site.atom_type, "__dict__")
                    and "tags_" in t_site.atom_type.__dict__
                ):
                    # mock site atom type is a reference to the same object
                    pass
                self.sites.append(s)
                mol_sites.append(s)
                site_offset += 1

            # Create Mock connections
            for b in t["bonds"]:
                idx0 = site_to_template_idx[b.connection_members[0]]
                idx1 = site_to_template_idx[b.connection_members[1]]
                self.bonds.append(
                    MockConnection(
                        (mol_sites[idx0], mol_sites[idx1]),
                        getattr(
                            b, "bond_type", getattr(b, "connection_type", None)
                        ),
                    )
                )

            for a in t["angles"]:
                idx0 = site_to_template_idx[a.connection_members[0]]
                idx1 = site_to_template_idx[a.connection_members[1]]
                idx2 = site_to_template_idx[a.connection_members[2]]
                self.angles.append(
                    MockConnection(
                        (mol_sites[idx0], mol_sites[idx1], mol_sites[idx2]),
                        getattr(
                            a, "angle_type", getattr(a, "connection_type", None)
                        ),
                    )
                )

            for d in t["dihedrals"]:
                idx0 = site_to_template_idx[d.connection_members[0]]
                idx1 = site_to_template_idx[d.connection_members[1]]
                idx2 = site_to_template_idx[d.connection_members[2]]
                idx3 = site_to_template_idx[d.connection_members[3]]
                self.dihedrals.append(
                    MockConnection(
                        (
                            mol_sites[idx0],
                            mol_sites[idx1],
                            mol_sites[idx2],
                            mol_sites[idx3],
                        ),
                        getattr(
                            d,
                            "dihedral_type",
                            getattr(d, "connection_type", None),
                        ),
                    )
                )

            for i in t["impropers"]:
                idx0 = site_to_template_idx[i.connection_members[0]]
                idx1 = site_to_template_idx[i.connection_members[1]]
                idx2 = site_to_template_idx[i.connection_members[2]]
                idx3 = site_to_template_idx[i.connection_members[3]]
                self.impropers.append(
                    MockConnection(
                        (
                            mol_sites[idx0],
                            mol_sites[idx1],
                            mol_sites[idx2],
                            mol_sites[idx3],
                        ),
                        getattr(
                            i,
                            "improper_type",
                            getattr(i, "connection_type", None),
                        ),
                    )
                )

            res_num += 1

    @property
    def n_sites(self):
        return len(self.sites)

    @property
    def n_bonds(self):
        return len(self.bonds)

    @property
    def n_angles(self):
        return len(self.angles)

    @property
    def n_dihedrals(self):
        return len(self.dihedrals)

    @property
    def n_impropers(self):
        return len(self.impropers)

    @property
    def connections(self):
        import itertools

        return itertools.chain(
            self.bonds, self.angles, self.dihedrals, self.impropers
        )

    @property
    def bond_types(self):
        return getattr(self.template_top, "bond_types", [])

    @property
    def angle_types(self):
        return getattr(self.template_top, "angle_types", [])

    @property
    def dihedral_types(self):
        return getattr(self.template_top, "dihedral_types", [])

    @property
    def improper_types(self):
        return getattr(self.template_top, "improper_types", [])

    @property
    def atom_types(self):
        return getattr(self.template_top, "atom_types", [])

    def get_index(self, site):
        return site.index

    def unique_site_labels(self, label_type, name_only=False):
        return self.template_top.unique_site_labels(label_type, name_only)

    def create_subtop(self, label_type, label):
        return self.template_top.create_subtop(label_type, label)

    def get_lj_scale(self, molecule_id):
        return self.template_top.get_lj_scale(molecule_id)

    def get_electrostatics_scale(self, molecule_id):
        return self.template_top.get_electrostatics_scale(molecule_id)


def combine_proxies(proxy1, proxy2):
    if proxy2 is None:
        return proxy1
    # Create an empty proxy and merge
    combined = TemplateTopologyProxy(proxy1.template_top, [])
    # We can just concatenate the lists, but we must update the indices of proxy2!
    offset = proxy1.n_sites

    combined.sites = proxy1.sites + proxy2.sites
    combined.bonds = proxy1.bonds + proxy2.bonds
    combined.angles = proxy1.angles + proxy2.angles
    combined.dihedrals = proxy1.dihedrals + proxy2.dihedrals
    combined.impropers = proxy1.impropers + proxy2.impropers

    # Update indices for proxy2 sites in the combined topology
    for s in proxy2.sites:
        s.index += offset

    # We don't need to change connection members, because they still point to the proxy2 sites
    # which we just updated! Wait, if we change `s.index`, it mutates proxy2.
    # That's fine because proxy2 is not used separately after this.

    # Also update residue numbers to continue from proxy1
    res_offset = (
        proxy1.sites[-1].__dict__["residue_number_"] if proxy1.sites else 0
    )
    for s in proxy2.sites:
        s.__dict__["residue_number_"] += res_offset

    return combined
