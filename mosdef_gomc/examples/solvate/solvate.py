from __future__ import print_function

import mbuild as mb
from mbuild.components.small_groups.h2o import H2O
from mbuild.examples.ethane.ethane import Ethane


def main():
    """Solvate an ethane molecule in a Box of water. """
    ethane = Ethane()
    water = H2O()
    return mb.solvate(ethane, water, 500, box=[2, 2, 2])


if __name__ == "__main__":
    solvated_ethane = main()
    solvated_ethane.save('foo.mol2')

