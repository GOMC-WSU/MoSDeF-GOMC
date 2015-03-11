from collections import OrderedDict
from copy import deepcopy
import os
import sys

import numpy as np
import mdtraj as md
from orderedset import OrderedSet

from mbuild.formats.hoomdxml import HOOMDTopologyFile
#from mbuild.formats.lammps import LAMMPSTopologyFIle
from mbuild.formats.mol2 import write_mol2

from mbuild.atom import Atom
from mbuild.bond import Bond
from mbuild.box import Box
from mbuild.coordinate_transform import translate
from mbuild.has_parts_mixin import HasPartsMixin
from mbuild.part_mixin import PartMixin


__all__ = ['load', 'Compound']


def load(filename, relative_to_module=None, frame=-1, compound=None,
         coords_only=False, **kwargs):
    """ """

    # For mbuild *.py files with a class that wraps a structure file in its own
    # folder. E.g., you build a system from ~/foo.py and it imports from
    # ~/bar/baz.py where baz.py loads ~/bar/baz.pdb.
    if relative_to_module:
        current_dir = os.path.dirname(os.path.realpath(sys.modules[relative_to_module].__file__))
        filename = os.path.join(current_dir, filename)

    # This can return a md.Trajectory or an mb.Compound.
    loaded = md.load(filename, **kwargs)

    if not compound:
        if isinstance(loaded, Compound):
            return loaded
        else:
            compound = Compound()

    if isinstance(loaded, md.Trajectory):
        compound.from_trajectory(loaded, frame=frame, coords_only=coords_only)
    elif isinstance(loaded, Compound):  # Only updating coordinates.
        for atom, loaded_atom in zip(compound.atoms, loaded.atoms):
            atom.pos = loaded_atom.pos
    return compound


class Compound(PartMixin, HasPartsMixin):
    """A building block in the mBuild hierarchy.

    Compound is the superclass of all composite building blocks in the mBuild
    hierarchy. That is, all composite building blocks must inherit from
    compound, either directly or indirectly. The design of Compound follows the
    Composite design pattern (Gamma, Erich; Richard Helm; Ralph Johnson; John
    M. Vlissides (1995). Design Patterns: Elements of Reusable Object-Oriented
    Software. Addison-Wesley. p. 395. ISBN 0-201-63361-2.), with Compound being
    the composite, and Atom playing the role of the primitive (leaf) part.

    Compound maintains a list of parts (contained Compounds, Atoms, Bonds,
    etc., that inherit from PartMixin), and provides a means to tag the parts
    with labels, so that the parts can be easily looked up later. Labels may
    also point to objects outside the Compound's containment hierarchy.
    Compound has built-in support for copying and deepcopying Compound
    hierarchies, enumerating atoms or bonds in the hierarchy, proximity based
    searches, visualization, I/O operations, and a number of other convenience
    methods.

    Parameters
    ----------
    kind : str, optional, default=self.__class__.__name__
        The type of Compound.
    periodicity : np.ndarray, shape=(3,), dtype=float, optional
        The periodic lengths of the Compound in the x, y and z directions.
        Defaults to zeros which is treated as non-periodic.

    Attributes
    ----------
    kind : str, optional, default=self.__class__.__name__
        The type of Compound.
    periodicity : np.ndarray, shape=(3,), dtype=float, optional
        The periodic lengths of the Compound in the x, y and z directions.
        Defaults to zeros which is treated as non-periodic.
    parts : OrderedSet
        Contains all child parts. Parts can be Atom, Bond or Compound - anything
        that inherits from PartMixin.
    labels : OrderedDict
        Labels to Compound/Atom mappings. These do not necessarily need not be
        in parts.
    parent : mb.Compound
        The parent Compound that contains this part. Can be None if this
        compound is the root of the containment hierarchy.
    referrers : set
        Other compounds that reference this part with labels.

    """
    def __init__(self, kind=None, periodicity=None):
        super(Compound, self).__init__()

        if kind:
            self.kind = kind
        else:
            self.kind = self.__class__.__name__

        # A periodocity of zero in any direction is treated as non-periodic.
        if not periodicity:
            periodicity = np.array([0.0, 0.0, 0.0])
        self.periodicity = periodicity

        # Allow storing extra stuff in a dict (created on-demand).
        self._extras = None

    @property
    def extras(self):
        """Return the Compound's optional, extra attributes. """
        if self._extras is None:
            self._extras = dict()
        return self._extras

    def __getattr__(self, item):
        if self._extras and item in self._extras:
            return self._extras[item]
        else:
            return super(Compound, self).__getattr__(item)

    @property
    def atoms(self):
        """A list of all Atoms in the Compound and sub-Compounds. """
        return self.atom_list_by_kind(excludeG=True)

    def yield_atoms(self):
        """ """
        return self._yield_parts(Atom)

    @property
    def n_atoms(self):
        """Return the number of Atoms in the Compound. """
        return len(self.atoms)

    def atom_list_by_kind(self, kind='*', excludeG=False):
        """Return a list of Atoms filtered by their kind.

        Parameters
        ----------
        kind : str
            Return only atoms of this type. '*' indicates all.
        excludeG : bool
            Exclude Port particles of kind 'G' - reserved for Ports.

        Returns
        -------
        atom_list : list
            List of Atoms matching the inputs.
        """
        atom_list = []
        for atom in self.yield_atoms():
            if not (excludeG and atom.kind == "G"):
                if kind == '*':
                    atom_list.append(atom)
                elif atom.kind == kind:
                    atom_list.append(atom)
        return atom_list

    @property
    def bonds(self):
        """A list of all Bonds in the Compound and sub-Compounds. """
        return self.bond_list_by_kind()

    def yield_bonds(self):
        """ """
        return self._yield_parts(Bond)

    @property
    def n_bonds(self):
        """Return the number of Bonds in the Compound. """
        return len(self.bonds)

    def bond_list_by_kind(self, kind='*'):
        """Return a list of Bonds filtered by their kind. """
        bond_list = []
        for bond in self.yield_bonds():
            if kind == '*':
                bond_list.append(bond)
            elif bond.kind == kind:
                bond_list.append(bond)
        return bond_list

    def referenced_ports(self):
        """Return all Ports referenced by this Compound. """
        from mbuild.port import Port
        return [port for port in self.labels.values() if isinstance(port, Port)]

    def _remove(self, removed_part):
        """If removing an atom, make sure to remove the bonds it's part of. """
        super(Compound, self)._remove(removed_part)

        if isinstance(removed_part, Atom):
            for bond in removed_part.bonds:
                bond.other_atom(removed_part).bonds.remove(bond)
                if bond.parent is not None:
                    bond.parent.remove(bond)

    def _inherit_periodicity(self, periodicity):
        """Inherit the periodicity of a Compound that was added.  """
        self.periodicity = periodicity

    # Interface to Trajectory for reading/writing.
    # --------------------------------------------
    def from_trajectory(self, traj, frame=-1, coords_only=False):
        """Extract atoms and bonds from a md.Trajectory.

        Will create sub-compounds for every chain if there is more than one
        and sub-sub-compounds for every residue.

        Parameters
        ----------
        traj : md.Trajectory
            The trajectory to load.
        frame : int
            The frame to take coordinates from.

        """
        atom_mapping = dict()
        for chain in traj.topology.chains:
            if traj.topology.n_chains > 1:
                chain_compound = Compound()
                self.add(chain_compound, "chain[$]")
            else:
                chain_compound = self
            for res in chain.residues:
                for atom in res.atoms:
                    new_atom = Atom(str(atom.name), traj.xyz[frame, atom.index])
                    chain_compound.add(new_atom, label="{0}[$]".format(atom.name))
                    atom_mapping[atom] = new_atom

        if not coords_only:
            for a1, a2 in traj.topology.bonds:
                atom1 = atom_mapping[a1]
                atom2 = atom_mapping[a2]
                self.add(Bond(atom1, atom2))

            if np.any(traj.unitcell_lengths) and np.any(traj.unitcell_lengths[0]):
                self.periodicity = traj.unitcell_lengths[0]
            else:
                self.periodicity = np.array([0., 0., 0.])

    def to_trajectory(self, show_ports=False):
        """Convert to an md.Trajectory using the compound as the topology. """
        exclude = not show_ports
        atom_list = self.atom_list_by_kind('*', excludeG=exclude)

        # FLATTEN COMPOUND SO THAT ALL THE TOPOLOGY FUNCTIONS WORK
        # to_topology functionality with chain assignment

        # Coordinates.
        n_atoms = len(atom_list)
        xyz = np.ndarray(shape=(1, n_atoms, 3), dtype='float')
        for idx, atom in enumerate(atom_list):
            xyz[0, idx] = atom.pos
            atom.uid = idx

        # Unitcell information.
        box = self.boundingbox()
        unitcell_lengths = np.empty(3)
        for dim, val in enumerate(self.periodicity):
            if val:
                unitcell_lengths[dim] = val
            else:
                unitcell_lengths[dim] = box.lengths[dim]

        self._numAtoms = n_atoms
        #self.bonds = [(bond.atom1.uid, bond.atom2.uid) for bond in self.bonds]
        return md.Trajectory(xyz, self, unitcell_lengths=unitcell_lengths,
                             unitcell_angles=np.array([90, 90, 90]))

    def update_coordinates(self, filename):
        """ """
        load(filename, compound=self, coords_only=True)

    def save(self, filename, show_ports=False, **kwargs):
        """Save the Compound to a file.

        Parameters
        ----------
        filename : str
            Filesystem path in which to save the trajectory. The extension or
            prefix will be parsed and will control the format.

        Other Parameters
        ----------------
        force_overwrite : bool
            For .binpos, .xtc, .dcd. If `filename` already exists, overwrite it.

        """
        # grab the extension of the filename
        extension = os.path.splitext(filename)[-1]

        savers = {'.hoomdxml': self.save_hoomdxml,
                  #'.gro': self.save_gromacs,
                  #'.top': self.save_gromacs,
                  '.mol2': self.save_mol2,
                  #'.lammps': self.save_lammpsdata,
                  #'.lmp': self.save_lammpsdata,
                  }

        try:
            saver = savers[extension]
        except KeyError:  # TODO: better reporting
            saver = None

        if saver:  # mBuild supported saver.
            return saver(self, filename, show_ports=show_ports)
        else:  # MDTraj supported saver.
            traj = self.to_trajectory(show_ports=show_ports)
            return traj.save(filename, **kwargs)

    def save_hoomdxml(self, filename, force_overwrite=True, optional_nodes=None):
        """ """
        with HOOMDTopologyFile(filename, 'w', force_overwrite=True) as f:
            f.optional_nodes = optional_nodes
            f.write(self, optional_nodes=optional_nodes)

    def save_mol2(self, compound, filename, show_ports=False):
        write_mol2(compound, filename, show_ports=show_ports)

    # def save_gromacs(self):

    # def save_lammpsdata(self):

    # Convenience functions
    # ---------------------
    def visualize(self, show_ports=False):
        """Visualize the Compound using VMD.

        Assumes you have VMD installed and can call it from the command line via
        'vmd'.

        TODO: Look into pizza.py's vmd.py. See issue #32.
        """
        filename = 'visualize_{}.mol2'.format(self.__class__.__name__)
        self.save(filename, show_ports=show_ports)
        import os

        try:
            os.system('vmd {}'.format(filename))
        except OSError:
            print("Visualization with VMD failed. Make sure it is installed"
                  "correctly and launchable from the command line via 'vmd'.")

    @property
    def center(self):
        """The cartesian center of the Compound based on its Atoms. """
        try:
            return sum(atom.pos for atom in self.atoms) / self.n_atoms
        except ZeroDivisionError:  # Compound only contains 'G' atoms.
            atoms = self.atom_list_by_kind('G')
            return sum(atom.pos for atom in atoms) / len(atoms)

    def boundingbox(self, excludeG=True):
        """Compute the bounding box of the compound. """
        minx = np.inf
        miny = np.inf
        minz = np.inf
        maxx = -np.inf
        maxy = -np.inf
        maxz = -np.inf

        for atom in self.yield_atoms():
            if excludeG and atom.kind == 'G':
                continue
            if atom.pos[0] < minx:
                minx = atom.pos[0]
            if atom.pos[0] > maxx:
                maxx = atom.pos[0]
            if atom.pos[1] < miny:
                miny = atom.pos[1]
            if atom.pos[1] > maxy:
                maxy = atom.pos[1]
            if atom.pos[2] < minz:
                minz = atom.pos[2]
            if atom.pos[2] > maxz:
                maxz = atom.pos[2]

        min_coords = np.array([minx, miny, minz])
        max_coords = np.array([maxx, maxy, maxz])

        return Box(mins=min_coords, maxs=max_coords)

    def atoms_in_range(self, point, radius, max_items=10):
        """Return the indices of atoms within a radius of a point.

        Parameters
        ----------
        point : array-like, shape=(3,), dtype=float
            The reference point in cartesian coordinates.
        radius : float
            Find Atoms within this radius.
        max_items : int
            Maximum number of atoms to find.

        Returns
        -------
        atoms : list
            List of atoms within specified range.

        """
        atoms = self.atom_list_by_kind(excludeG=True)
        traj = self.to_trajectory()
        idxs = traj.atoms_in_range_idx(point, radius, max_items=max_items)
        return [atoms[idx] for idx in idxs]

    def wrap(self):
        """Wrap a periodic Compound. """
        assert np.any(self.periodicity)
        box = self.boundingbox()
        translate(self, -box.mins)
        for atom in self.yield_atoms():
            for i, coordinate in enumerate(atom.pos):
                if self.periodicity[i]:
                    if coordinate < 0.0:
                        atom.pos[i] = self.periodicity[i] + coordinate
                    if coordinate > self.periodicity[i]:
                        atom.pos[i] = coordinate - self.periodicity[i]

    def min_periodic_distance(self, xyz0, xyz1):
        """Vectorized distance calculation considering minimum image. """
        d = np.abs(xyz0 - xyz1)
        d = np.where(d > 0.5 * self.periodicity, self.periodicity - d, d)
        return np.sqrt((d ** 2).sum(axis=-1))

    def add_bond(self, type_a, type_b, dmin, dmax, kind=None):
        """Add Bonds between all Atom pairs of types a/b within [dmin, dmax].

        TODO: testing for periodic boundaries.
        """
        for a1 in self.atom_list_by_kind(type_a):
            nearest = self.atoms_in_range(a1.pos, dmax)
            for a2 in nearest:
                if (a2.kind == type_b) and (dmin <= self.min_periodic_distance(a2.pos, a1.pos) <= dmax):
                    self.add(Bond(a1, a2, kind=kind))

    def __deepcopy__(self, memo):
        cls = self.__class__
        newone = cls.__new__(cls)
        if len(memo) == 0:
            memo[0] = self
        memo[id(self)] = newone

        # First copy those attributes that don't need deepcopying.
        newone.kind = deepcopy(self.kind, memo)
        newone.periodicity = deepcopy(self.periodicity, memo)

        # Create empty containers.
        newone.parts = OrderedSet()
        newone.labels = OrderedDict()
        newone.referrers = set()

        # Copy the parent of everyone, except topmost Compound being deepcopied.
        if memo[0] == self:
            newone.parent = None
        else:
            newone.parent = deepcopy(self.parent, memo)

        # Copy parts, except bonds with atoms outside the hierarchy.
        for part in self.parts:
            if isinstance(part, Bond):
                if memo[0] in part.atom1.ancestors() and memo[0] in part.atom2.ancestors():
                    newone.parts.add(deepcopy(part, memo))
            else:
                newone.parts.add(deepcopy(part, memo))

        # Copy labels, except bonds with atoms outside the hierarchy
        for k, v in self.labels.items():
            if isinstance(v, Bond):
                if memo[0] in v.atom1.ancestors() and memo[0] in v.atom2.ancestors():
                    newone.labels[k] = deepcopy(v, memo)
                    newone.labels[k].referrers.add(newone)
            else:
                newone.labels[k] = deepcopy(v, memo)
                if not isinstance(newone.labels[k], list):
                    newone.labels[k].referrers.add(newone)

        # Copy referrers that do not point out of the hierarchy.
        for r in self.referrers:
            if memo[0] in r.ancestors():
                newone.referrers.add(deepcopy(r, memo))

        newone._extras = deepcopy(self._extras, memo)

        return newone
