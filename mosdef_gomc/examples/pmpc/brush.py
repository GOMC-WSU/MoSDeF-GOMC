from numpy import pi

import mbuild as mb

from mbuild.components.small_groups.silane import Silane
from mbuild.components.small_groups.ch3 import Ch3
from mbuild.examples.pmpc.mpc import MPC
from mbuild.examples.pmpc.initiator import Initiator


class Brush(mb.Compound):
    """ """
    def __init__(self, chain_length=4, alpha=pi/4):
        super(Brush, self).__init__(self)

        # Add parts
        self.add(Silane(), 'silane')
        self.add(Initiator(), 'initiator')
        self.add(mb.Polymer(MPC(alpha=alpha), n=chain_length,
                         port_labels=('up', 'down')), 'pmpc')
        self.add(Ch3(), 'methyl')

        mb.equivalence_transform(self.initiator, self.initiator.down, self.silane.up)
        mb.equivalence_transform(self.pmpc, self.pmpc.down, self.initiator.up)
        mb.equivalence_transform(self.methyl, self.methyl.up, self.pmpc.up)

        # Make self.port point to silane.bottom_port
        self.add(self.silane.down, label='down', containment=False)

if __name__ == "__main__":
    pmpc = Brush(chain_length=1, alpha=pi/4)
    pmpc.visualize(show_ports=True)
